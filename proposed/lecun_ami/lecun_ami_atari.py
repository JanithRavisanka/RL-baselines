"""
LeCun-AMI prototype for more complex discrete-action games.

This script adapts the proposed AMI/J-EPA architecture from Pendulum to Atari:

1. CNN perception encoder:
   stacked 84x84 grayscale frames -> compact latent state.
2. JEPA-style latent world model:
   predicts the next latent state for a discrete action, without pixel
   reconstruction.
3. Cost model:
   predicts immediate cost, using cost = -clipped_reward.
4. Critic:
   predicts future cost-to-go from the latent state.
5. Actor:
   fast reactive discrete policy trained from planner demonstrations.
6. Configurator:
   decides whether to use actor or model-based planning.
7. Discrete latent planner:
   samples candidate action sequences, rolls them out in latent space, and
   executes the first action of the lowest predicted cost sequence.

This is intentionally a research prototype, not a tuned Atari SOTA agent. Use it
to test whether the proposed architecture scales beyond simple continuous
control and to compare robustness/compute trade-offs against DQN-style baselines.
"""

import argparse
import csv
import datetime
import json
import os
import random
import time
from collections import deque

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation


gym.register_envs(ale_py)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = get_device()
print(f"Using device: {device}")


class AtariReplayBuffer:
    """
    Replay storage for pixel observations and optional planner labels.

    Observations are stored as uint8 stacked frames to keep memory practical.
    Conversion to float and normalization happens only inside the training batch.
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, planner_action=None):
        self.buffer.append((
            np.asarray(state, dtype=np.uint8),
            int(action),
            float(np.sign(reward)),
            np.asarray(next_state, dtype=np.uint8),
            float(done),
            None if planner_action is None else int(planner_action),
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, _ = zip(*batch)
        return (
            torch.tensor(np.stack(states), dtype=torch.float32, device=device) / 255.0,
            torch.tensor(actions, dtype=torch.long, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1),
            torch.tensor(np.stack(next_states), dtype=torch.float32, device=device) / 255.0,
            torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1),
        )

    def sample_n_step(self, batch_size, n_step, gamma):
        """
        Sample n-step critic targets from contiguous replay segments.

        The world model still learns one-step latent transitions, but the critic
        gets a less sparse target by accumulating several clipped rewards before
        bootstrapping. Episode boundaries stop the accumulation early.
        """
        n_step = max(1, int(n_step))
        states = []
        actions = []
        cost_targets = []
        bootstrap_states = []
        dones = []
        bootstrap_discounts = []
        max_start = max(1, len(self.buffer) - n_step + 1)

        for _ in range(batch_size):
            start = random.randrange(max_start)
            first = self.buffer[start]
            total_cost = 0.0
            discount = 1.0
            done = 0.0
            bootstrap_state = first[3]

            for offset in range(n_step):
                index = start + offset
                if index >= len(self.buffer):
                    break
                transition = self.buffer[index]
                total_cost += discount * (-transition[2])
                bootstrap_state = transition[3]
                done = transition[4]
                if done:
                    discount = 0.0
                    break
                discount *= gamma

            states.append(first[0])
            actions.append(first[1])
            cost_targets.append(total_cost)
            bootstrap_states.append(bootstrap_state)
            dones.append(done)
            bootstrap_discounts.append(discount)

        return (
            torch.tensor(np.stack(states), dtype=torch.float32, device=device) / 255.0,
            torch.tensor(actions, dtype=torch.long, device=device),
            torch.tensor(cost_targets, dtype=torch.float32, device=device).unsqueeze(1),
            torch.tensor(np.stack(bootstrap_states), dtype=torch.float32, device=device) / 255.0,
            torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1),
            torch.tensor(bootstrap_discounts, dtype=torch.float32, device=device).unsqueeze(1),
        )

    def sample_planner_demos(self, batch_size):
        demos = [item for item in self.buffer if item[5] is not None]
        if len(demos) < batch_size:
            return None
        batch = random.sample(demos, batch_size)
        states = [item[0] for item in batch]
        planner_actions = [item[5] for item in batch]
        return (
            torch.tensor(np.stack(states), dtype=torch.float32, device=device) / 255.0,
            torch.tensor(planner_actions, dtype=torch.long, device=device),
        )

    def __len__(self):
        return len(self.buffer)


class AtariEncoder(nn.Module):
    """
    CNN perception encoder for stacked Atari frames.

    The convolutional front-end mirrors the classic DQN shape, but the output is
    a latent representation used by the world model and planner rather than
    direct Q-values.
    """

    def __init__(self, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, obs):
        return self.head(self.conv(obs))


class DiscreteLatentWorldModel(nn.Module):
    """
    Predicts next latent state conditioned on a discrete action.

    Actions are embedded before being concatenated with the current latent state.
    A residual update is used because the next latent is usually close to the
    current latent over one Atari control step.
    """

    def __init__(self, latent_dim, action_dim, action_embed_dim, hidden_dim):
        super().__init__()
        self.action_embed = nn.Embedding(action_dim, action_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, latent, action):
        action_emb = self.action_embed(action)
        x = torch.cat([latent, action_emb], dim=-1)
        return latent + self.net(x)


class DiscreteCostModel(nn.Module):
    """Predicts immediate cost c(z_t, a_t) for a discrete action."""

    def __init__(self, latent_dim, action_dim, action_embed_dim, hidden_dim):
        super().__init__()
        self.action_embed = nn.Embedding(action_dim, action_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent, action):
        action_emb = self.action_embed(action)
        return self.net(torch.cat([latent, action_emb], dim=-1))


class Critic(nn.Module):
    """Latent future cost-to-go estimator."""

    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent):
        return self.net(latent)


class DiscreteActor(nn.Module):
    """Fast reactive policy over Atari actions."""

    def __init__(self, latent_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, latent):
        return self.net(latent)


def make_atari_env(env_name, render_mode=None, terminal_on_life_loss=True):
    """Create an Atari environment with DQN-style preprocessing."""
    env = gym.make(env_name, render_mode=render_mode, frameskip=1)
    env = AtariPreprocessing(
        env,
        frame_skip=4,
        grayscale_obs=True,
        scale_obs=False,
        terminal_on_life_loss=terminal_on_life_loss,
    )
    env = FrameStackObservation(env, stack_size=4)
    return env


def update_target_network(source, target, tau):
    """EMA update for target networks used as stable bootstrap references."""
    with torch.no_grad():
        for online_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(tau).add_(online_param.data, alpha=1.0 - tau)


def update_target_encoder(encoder, target_encoder, tau):
    """EMA update for the stop-gradient target encoder."""
    update_target_network(encoder, target_encoder, tau)


def latent_variance_loss(latent):
    """Lightweight anti-collapse regularizer for latent representations."""
    if latent.shape[0] < 2:
        return torch.zeros((), device=latent.device)
    std = torch.sqrt(latent.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(1.0 - std).mean()


class AtariAMIAgent:
    """
    AMI/J-EPA agent for discrete pixel games.

    This class keeps the same conceptual modules as the Pendulum prototype, but
    swaps the actor/planner interface for discrete Atari actions.
    """

    def __init__(self, action_dim, args):
        self.args = args
        self.action_dim = action_dim

        self.encoder = AtariEncoder(args.latent_dim).to(device)
        self.target_encoder = AtariEncoder(args.latent_dim).to(device)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad_(False)

        self.world_models = nn.ModuleList([
            DiscreteLatentWorldModel(args.latent_dim, action_dim, args.action_embed_dim, args.hidden_dim).to(device)
            for _ in range(args.ensemble_size)
        ])
        self.cost_model = DiscreteCostModel(args.latent_dim, action_dim, args.action_embed_dim, args.hidden_dim).to(device)
        self.critic = Critic(args.latent_dim, args.hidden_dim).to(device)
        self.target_critic = Critic(args.latent_dim, args.hidden_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for param in self.target_critic.parameters():
            param.requires_grad_(False)
        self.actor = DiscreteActor(args.latent_dim, action_dim, args.hidden_dim).to(device)

        self.model_optimizer = optim.Adam(
            list(self.encoder.parameters())
            + list(self.world_models.parameters())
            + list(self.cost_model.parameters())
            + list(self.critic.parameters()),
            lr=args.model_lr,
        )
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)

    def encode_numpy(self, state):
        obs = torch.tensor(np.asarray(state), dtype=torch.float32, device=device).unsqueeze(0) / 255.0
        return self.encoder(obs)

    def actor_action(self, state, epsilon=0.0):
        """Epsilon-greedy action from the fast actor."""
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        self.encoder.eval()
        self.actor.eval()
        with torch.no_grad():
            latent = self.encode_numpy(state)
            logits = self.actor(latent)
            return int(torch.argmax(logits, dim=-1).item())

    def one_step_uncertainty(self, state):
        """Configurator uncertainty from ensemble disagreement."""
        self.encoder.eval()
        self.actor.eval()
        for model in self.world_models:
            model.eval()
        with torch.no_grad():
            latent = self.encode_numpy(state)
            action = torch.argmax(self.actor(latent), dim=-1)
            preds = torch.stack([model(latent, action) for model in self.world_models], dim=0)
            return preds.var(dim=0, unbiased=False).mean().item()

    def should_plan(self, state, global_step, evaluate=False):
        """Decide whether to invoke latent-space planning."""
        if evaluate:
            if self.args.eval_mode == "planner":
                return True
            if self.args.eval_mode == "actor":
                return False
            if self.args.planning_interval > 0 and global_step % self.args.planning_interval == 0:
                return True
            return self.one_step_uncertainty(state) >= self.args.uncertainty_threshold
        if self.args.planning_mode == "always":
            return True
        if self.args.planning_mode == "actor":
            return False
        if global_step < self.args.warmup_plan_steps:
            return True
        if self.args.planning_interval > 0 and global_step % self.args.planning_interval == 0:
            return True
        return self.one_step_uncertainty(state) >= self.args.uncertainty_threshold

    def ensemble_next(self, latent, action):
        preds = torch.stack([model(latent, action) for model in self.world_models], dim=0)
        return preds.mean(dim=0), preds.var(dim=0, unbiased=False).mean(dim=-1, keepdim=True)

    @torch.no_grad()
    def plan(self, state):
        """
        Random-shooting planner for discrete action sequences.

        CEM over categorical distributions is possible, but random shooting is a
        simple first baseline: sample many action sequences, score them with the
        learned latent world, and execute the first action of the best sequence.
        """
        self.encoder.eval()
        self.actor.eval()
        self.cost_model.eval()
        self.critic.eval()
        for model in self.world_models:
            model.eval()

        latent = self.encode_numpy(state)
        sequences = self.sample_action_sequences(latent)
        costs, uncertainties = self.rollout_cost(latent, sequences)
        best_idx = torch.argmin(costs)
        return int(sequences[best_idx, 0].item()), {
            "predicted_cost": float(costs[best_idx].item()),
            "uncertainty": float(uncertainties[best_idx].item()),
        }

    def sample_action_sequences(self, latent):
        """Sample candidate action sequences, biased partly toward actor choices."""
        sequences = torch.randint(
            low=0,
            high=self.action_dim,
            size=(self.args.num_sequences, self.args.horizon),
            device=device,
        )
        if self.args.actor_seed_frac <= 0:
            return sequences

        actor_count = int(self.args.num_sequences * self.args.actor_seed_frac)
        if actor_count <= 0:
            return sequences

        # Seed some sequences with repeated actor actions plus small random
        # mutations. This helps planning focus near the reactive policy while
        # preserving exploration of other action sequences.
        logits = self.actor(latent)
        probs = F.softmax(logits, dim=-1).squeeze(0)
        actor_actions = torch.multinomial(probs, actor_count * self.args.horizon, replacement=True)
        sequences[:actor_count] = actor_actions.view(actor_count, self.args.horizon)
        mutation_mask = torch.rand(actor_count, self.args.horizon, device=device) < self.args.sequence_mutation_prob
        mutations = torch.randint(0, self.action_dim, (actor_count, self.args.horizon), device=device)
        sequences[:actor_count] = torch.where(mutation_mask, mutations, sequences[:actor_count])
        return sequences

    def rollout_cost(self, latent, sequences):
        """Roll out candidate discrete action sequences in latent space."""
        num_sequences = sequences.shape[0]
        latent_batch = latent.repeat(num_sequences, 1)
        total_cost = torch.zeros(num_sequences, 1, device=device)
        total_uncertainty = torch.zeros(num_sequences, 1, device=device)
        discount = 1.0

        for t in range(sequences.shape[1]):
            action_t = sequences[:, t]
            step_cost = self.cost_model(latent_batch, action_t)
            next_latent, uncertainty = self.ensemble_next(latent_batch, action_t)
            total_cost += discount * (step_cost + self.args.uncertainty_cost * uncertainty)
            total_uncertainty += uncertainty
            latent_batch = next_latent
            discount *= self.args.gamma

        total_cost += discount * self.critic(latent_batch)
        return total_cost.squeeze(1), total_uncertainty.squeeze(1) / sequences.shape[1]

    def update(self, replay):
        """Train encoder/world/cost/critic and actor imitation from replay."""
        if len(replay) < self.args.batch_size:
            return {}

        self.encoder.train()
        self.cost_model.train()
        self.critic.train()
        for model in self.world_models:
            model.train()

        states, actions, rewards, next_states, dones = replay.sample(self.args.batch_size)
        cost_targets = -rewards
        latent = self.encoder(states)
        with torch.no_grad():
            target_next_latent = self.target_encoder(next_states)

        world_loss = 0.0
        for model in self.world_models:
            pred_next_latent = model(latent, actions)
            world_loss = world_loss + F.mse_loss(pred_next_latent, target_next_latent)
        world_loss = world_loss / len(self.world_models)

        cost_loss = F.mse_loss(self.cost_model(latent, actions), cost_targets)
        (
            critic_states,
            _,
            critic_cost_targets,
            critic_bootstrap_states,
            critic_dones,
            critic_bootstrap_discounts,
        ) = replay.sample_n_step(self.args.batch_size, self.args.critic_n_step, self.args.gamma)
        critic_latent = self.encoder(critic_states)
        value = self.critic(critic_latent)
        with torch.no_grad():
            target_bootstrap_latent = self.target_encoder(critic_bootstrap_states)
            next_value = self.target_critic(target_bootstrap_latent)
            critic_target = (
                critic_cost_targets
                + critic_bootstrap_discounts * (1.0 - critic_dones) * next_value
            )
        critic_loss = F.mse_loss(value, critic_target)
        repr_loss = latent_variance_loss(latent)

        model_loss = (
            world_loss
            + self.args.cost_loss_coef * cost_loss
            + self.args.critic_loss_coef * critic_loss
            + self.args.repr_loss_coef * repr_loss
        )
        self.model_optimizer.zero_grad()
        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters())
            + list(self.world_models.parameters())
            + list(self.cost_model.parameters())
            + list(self.critic.parameters()),
            self.args.max_grad_norm,
        )
        self.model_optimizer.step()
        update_target_encoder(self.encoder, self.target_encoder, self.args.target_tau)
        update_target_network(self.critic, self.target_critic, self.args.critic_target_tau)

        actor_loss_value = 0.0
        demo_batch = replay.sample_planner_demos(self.args.batch_size)
        if demo_batch is not None:
            demo_states, planner_actions = demo_batch
            self.encoder.eval()
            self.actor.train()
            with torch.no_grad():
                demo_latent = self.encoder(demo_states)
            logits = self.actor(demo_latent)
            actor_loss = F.cross_entropy(logits, planner_actions)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
            self.actor_optimizer.step()
            actor_loss_value = actor_loss.item()

        return {
            "world_loss": float(world_loss.item()),
            "cost_loss": float(cost_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "repr_loss": float(repr_loss.item()),
            "actor_loss": float(actor_loss_value),
            "critic_n_step": float(self.args.critic_n_step),
        }


def select_action(agent, state, global_step, epsilon, evaluate=False):
    plan_now = agent.should_plan(state, global_step, evaluate=evaluate)
    if plan_now:
        action, info = agent.plan(state)
        return action, True, info
    return agent.actor_action(state, epsilon=0.0 if evaluate else epsilon), False, {
        "predicted_cost": None,
        "uncertainty": agent.one_step_uncertainty(state),
    }


def collect_seed_data(env, replay, seed_steps):
    state, _ = env.reset()
    for _ in range(seed_steps):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay.add(state, action, reward, next_state, done)
        state = next_state
        if done:
            state, _ = env.reset()


def epsilon_by_step(step, args):
    frac = min(1.0, step / max(1, args.epsilon_decay_steps))
    return args.epsilon_start + frac * (args.epsilon_end - args.epsilon_start)


def mean_or_zero(values):
    return float(np.mean(values)) if values else 0.0


def write_config(args, save_dir):
    config_path = os.path.join(save_dir, "config.json")
    payload = {
        "config": vars(args),
        "device": str(device),
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(config_path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(f"Config saved: {config_path}")


def save_checkpoint(agent, args, save_dir, filename, global_step=None):
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save({
        "global_step": global_step,
        "encoder": agent.encoder.state_dict(),
        "target_encoder": agent.target_encoder.state_dict(),
        "world_models": [model.state_dict() for model in agent.world_models],
        "cost_model": agent.cost_model.state_dict(),
        "critic": agent.critic.state_dict(),
        "target_critic": agent.target_critic.state_dict(),
        "actor": agent.actor.state_dict(),
        "model_optimizer": agent.model_optimizer.state_dict(),
        "actor_optimizer": agent.actor_optimizer.state_dict(),
        "config": vars(args),
        "action_dim": agent.action_dim,
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def get_fire_action(env):
    try:
        meanings = env.unwrapped.get_action_meanings()
    except Exception:
        return None
    return meanings.index("FIRE") if "FIRE" in meanings else None


def get_lives(env):
    try:
        return env.unwrapped.ale.lives()
    except Exception:
        return None


def evaluate_policy(agent, args, global_step, record_gif=False, save_dir=None):
    render_mode = "rgb_array" if record_gif else None
    env = make_atari_env(args.env, render_mode=render_mode, terminal_on_life_loss=False)
    rewards = []
    steps_per_episode = []
    planning_rates = []
    frames = []
    fire_action = get_fire_action(env)

    for episode in range(args.eval_episodes):
        state, _ = env.reset(seed=args.seed + 100000 + global_step + episode)
        done = False
        total_reward = 0.0
        steps = 0
        planned_steps = 0
        lives = get_lives(env)

        if fire_action is not None and steps < args.eval_max_steps:
            state, reward, terminated, truncated, _ = env.step(fire_action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            if record_gif:
                frames.append(env.render())

        while not done and steps < args.eval_max_steps:
            if record_gif:
                frames.append(env.render())
            action, planned, _ = select_action(
                agent,
                state,
                global_step + steps,
                epsilon=0.0,
                evaluate=True,
            )
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            planned_steps += int(planned)
            steps += 1

            new_lives = get_lives(env)
            if (
                fire_action is not None
                and lives is not None
                and new_lives is not None
                and new_lives < lives
                and not done
                and steps < args.eval_max_steps
            ):
                lives = new_lives
                state, reward, terminated, truncated, _ = env.step(fire_action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
                if record_gif:
                    frames.append(env.render())

        rewards.append(total_reward)
        steps_per_episode.append(steps)
        planning_rates.append(planned_steps / max(1, steps))

    env.close()

    gif_path = None
    if record_gif and frames and save_dir is not None:
        import imageio

        gif_path = os.path.join(save_dir, "lecun_ami_atari_agent.gif")
        imageio.mimsave(gif_path, frames, fps=30)
        print(f"Evaluation GIF saved: {gif_path}")

    return {
        "global_step": int(global_step),
        "eval_reward_mean": mean_or_zero(rewards),
        "eval_reward_std": float(np.std(rewards)) if rewards else 0.0,
        "eval_reward_min": float(np.min(rewards)) if rewards else 0.0,
        "eval_reward_max": float(np.max(rewards)) if rewards else 0.0,
        "eval_steps_mean": mean_or_zero(steps_per_episode),
        "eval_planning_rate_mean": mean_or_zero(planning_rates),
        "eval_episodes": int(args.eval_episodes),
        "gif_path": gif_path or "",
    }


def train(args, save_dir):
    env = make_atari_env(args.env)
    env.action_space.seed(args.seed)
    agent = AtariAMIAgent(env.action_space.n, args)
    replay = AtariReplayBuffer(args.replay_size)

    print(f"Collecting {args.seed_steps} random seed transitions...")
    collect_seed_data(env, replay, args.seed_steps)

    print("Training initial Atari latent world model...")
    latest_losses = {}
    for _ in range(args.initial_updates):
        latest_losses = agent.update(replay)

    state, _ = env.reset()
    episode_reward = 0.0
    episode_rewards = []
    planning_rates = []
    planned_steps = 0
    episode_steps = 0

    metrics_path = os.path.join(save_dir, "metrics.csv")
    training_log_path = os.path.join(save_dir, "training_log.csv")
    eval_metrics_path = os.path.join(save_dir, "eval_metrics.csv")
    metrics_fields = [
        "episode",
        "global_step",
        "reward",
        "planning_rate",
        "episode_steps",
        "epsilon",
        "completed",
        "elapsed_sec",
    ]
    training_fields = [
        "global_step",
        "elapsed_sec",
        "episodes",
        "epsilon",
        "replay_size",
        "avg_reward_10",
        "avg_reward_100",
        "avg_planning_10",
        "avg_planning_100",
        "avg_uncertainty",
        "avg_predicted_cost",
        "world_loss",
        "cost_loss",
        "critic_loss",
        "repr_loss",
        "actor_loss",
        "critic_n_step",
    ]
    eval_fields = [
        "global_step",
        "eval_reward_mean",
        "eval_reward_std",
        "eval_reward_min",
        "eval_reward_max",
        "eval_steps_mean",
        "eval_planning_rate_mean",
        "eval_episodes",
        "gif_path",
    ]

    recent_uncertainties = deque(maxlen=max(1, args.log_interval))
    recent_predicted_costs = deque(maxlen=max(1, args.log_interval))
    start_time = time.time()
    episode_index = 0

    print("Starting Atari LeCun-AMI training loop...")
    with (
        open(metrics_path, "w", newline="") as metrics_handle,
        open(training_log_path, "w", newline="") as training_handle,
        open(eval_metrics_path, "w", newline="") as eval_handle,
    ):
        metrics_writer = csv.DictWriter(metrics_handle, fieldnames=metrics_fields)
        training_writer = csv.DictWriter(training_handle, fieldnames=training_fields)
        eval_writer = csv.DictWriter(eval_handle, fieldnames=eval_fields)
        metrics_writer.writeheader()
        training_writer.writeheader()
        eval_writer.writeheader()

        for global_step in range(1, args.total_steps + 1):
            epsilon = epsilon_by_step(global_step, args)
            action, planned, action_info = select_action(agent, state, global_step, epsilon, evaluate=False)
            uncertainty = action_info.get("uncertainty")
            predicted_cost = action_info.get("predicted_cost")
            if uncertainty is not None:
                recent_uncertainties.append(float(uncertainty))
            if predicted_cost is not None:
                recent_predicted_costs.append(float(predicted_cost))

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay.add(state, action, reward, next_state, done, planner_action=action if planned else None)

            episode_reward += reward
            planned_steps += int(planned)
            episode_steps += 1
            state = next_state

            for _ in range(args.updates_per_step):
                latest_losses = agent.update(replay)

            if done:
                episode_index += 1
                planning_rate = planned_steps / max(1, episode_steps)
                episode_rewards.append(episode_reward)
                planning_rates.append(planning_rate)
                elapsed = time.time() - start_time
                metrics_writer.writerow({
                    "episode": episode_index,
                    "global_step": global_step,
                    "reward": episode_reward,
                    "planning_rate": planning_rate,
                    "episode_steps": episode_steps,
                    "epsilon": epsilon,
                    "completed": 1,
                    "elapsed_sec": elapsed,
                })
                metrics_handle.flush()
                state, _ = env.reset()
                episode_reward = 0.0
                planned_steps = 0
                episode_steps = 0

            if global_step % args.log_interval == 0:
                elapsed = time.time() - start_time
                avg_reward_10 = mean_or_zero(episode_rewards[-10:])
                avg_reward_100 = mean_or_zero(episode_rewards[-100:])
                avg_plan_10 = mean_or_zero(planning_rates[-10:])
                avg_plan_100 = mean_or_zero(planning_rates[-100:])
                training_writer.writerow({
                    "global_step": global_step,
                    "elapsed_sec": elapsed,
                    "episodes": len(episode_rewards),
                    "epsilon": epsilon,
                    "replay_size": len(replay),
                    "avg_reward_10": avg_reward_10,
                    "avg_reward_100": avg_reward_100,
                    "avg_planning_10": avg_plan_10,
                    "avg_planning_100": avg_plan_100,
                    "avg_uncertainty": mean_or_zero(recent_uncertainties),
                    "avg_predicted_cost": mean_or_zero(recent_predicted_costs),
                    "world_loss": latest_losses.get("world_loss", 0.0),
                    "cost_loss": latest_losses.get("cost_loss", 0.0),
                    "critic_loss": latest_losses.get("critic_loss", 0.0),
                    "repr_loss": latest_losses.get("repr_loss", 0.0),
                    "actor_loss": latest_losses.get("actor_loss", 0.0),
                    "critic_n_step": latest_losses.get("critic_n_step", args.critic_n_step),
                })
                training_handle.flush()
                print(
                    f"Step {global_step}/{args.total_steps} | "
                    f"Epsilon: {epsilon:.3f} | "
                    f"Avg reward (10/100): {avg_reward_10:.2f}/{avg_reward_100:.2f} | "
                    f"Planning rate (10/100): {avg_plan_10:.2f}/{avg_plan_100:.2f} | "
                    f"World loss: {latest_losses.get('world_loss', 0.0):.4f} | "
                    f"Critic loss: {latest_losses.get('critic_loss', 0.0):.4f} | "
                    f"Actor loss: {latest_losses.get('actor_loss', 0.0):.4f}"
                )

            if args.eval_interval > 0 and global_step % args.eval_interval == 0:
                eval_stats = evaluate_policy(agent, args, global_step, record_gif=False)
                eval_writer.writerow(eval_stats)
                eval_handle.flush()
                print(
                    f"Eval at step {global_step}: "
                    f"mean={eval_stats['eval_reward_mean']:.2f}, "
                    f"max={eval_stats['eval_reward_max']:.2f}, "
                    f"planning={eval_stats['eval_planning_rate_mean']:.2f}"
                )

            if args.checkpoint_interval > 0 and global_step % args.checkpoint_interval == 0:
                save_checkpoint(agent, args, save_dir, f"checkpoint_step_{global_step}.pth", global_step)

        if episode_steps > 0:
            episode_index += 1
            planning_rate = planned_steps / max(1, episode_steps)
            episode_rewards.append(episode_reward)
            planning_rates.append(planning_rate)
            metrics_writer.writerow({
                "episode": episode_index,
                "global_step": args.total_steps,
                "reward": episode_reward,
                "planning_rate": planning_rate,
                "episode_steps": episode_steps,
                "epsilon": epsilon_by_step(args.total_steps, args),
                "completed": 0,
                "elapsed_sec": time.time() - start_time,
            })
            metrics_handle.flush()

    env.close()
    return agent, episode_rewards, planning_rates


def save_metrics(rewards, planning_rates, save_dir, env_name, no_plots=False):
    metrics_path = os.path.join(save_dir, "metrics.csv")
    if os.path.exists(metrics_path):
        print(f"Episode metrics saved during training: {metrics_path}")
    else:
        with open(metrics_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["episode", "reward", "planning_rate"])
            for index, (reward, planning_rate) in enumerate(zip(rewards, planning_rates), start=1):
                writer.writerow([index, reward, planning_rate])
        print(f"Metrics saved: {metrics_path}")
    if no_plots:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots because matplotlib could not be imported: {exc}")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.7, label="Episode reward")
    if len(rewards) >= 10:
        moving = np.convolve(rewards, np.ones(10) / 10, mode="valid")
        plt.plot(range(9, len(rewards)), moving, label="Moving average (10)")
    plt.title(f"LeCun-AMI Atari Training on {os.path.basename(env_name)}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()
    plt.legend()
    reward_path = os.path.join(save_dir, "training_curve.png")
    plt.savefig(reward_path)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(planning_rates)
    plt.title("Configurator Planning Rate")
    plt.xlabel("Episode")
    plt.ylabel("Fraction of planned steps")
    plt.ylim(0.0, 1.05)
    plt.grid()
    planning_path = os.path.join(save_dir, "planning_rate.png")
    plt.savefig(planning_path)
    plt.close()
    print(f"Training curve saved: {reward_path}")
    print(f"Planning-rate curve saved: {planning_path}")


def evaluate_and_record(agent, args, save_dir):
    eval_stats = evaluate_policy(
        agent,
        args,
        args.total_steps,
        record_gif=not args.no_gif,
        save_dir=save_dir,
    )
    final_eval_path = os.path.join(save_dir, "final_eval_metrics.json")
    with open(final_eval_path, "w") as handle:
        json.dump(eval_stats, handle, indent=2, sort_keys=True)

    eval_metrics_path = os.path.join(save_dir, "eval_metrics.csv")
    write_header = not os.path.exists(eval_metrics_path) or os.path.getsize(eval_metrics_path) == 0
    with open(eval_metrics_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(eval_stats.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(eval_stats)

    print(
        f"Final full-game evaluation mean over {args.eval_episodes}: "
        f"{eval_stats['eval_reward_mean']:.2f}"
    )
    print(f"Final evaluation metrics saved: {final_eval_path}")


def build_args():
    parser = argparse.ArgumentParser(
        description="LeCun-AMI/J-EPA prototype for Atari-style discrete pixel games."
    )
    parser.add_argument("--env", type=str, default="ALE/Breakout-v5")
    parser.add_argument("--total-steps", type=int, default=100000)
    parser.add_argument("--seed-steps", type=int, default=5000)
    parser.add_argument("--initial-updates", type=int, default=1000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--replay-size", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--action-embed-dim", type=int, default=32)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--num-sequences", type=int, default=256)
    parser.add_argument("--actor-seed-frac", type=float, default=0.25)
    parser.add_argument("--sequence-mutation-prob", type=float, default=0.2)
    parser.add_argument("--planning-mode", choices=["adaptive", "always", "actor"], default="adaptive")
    parser.add_argument("--eval-mode", choices=["adaptive", "planner", "actor"], default="planner")
    parser.add_argument("--warmup-plan-steps", type=int, default=10000)
    parser.add_argument("--planning-interval", type=int, default=8)
    parser.add_argument("--uncertainty-threshold", type=float, default=0.01)
    parser.add_argument("--uncertainty-cost", type=float, default=0.1)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=50000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--model-lr", type=float, default=1e-4)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--target-tau", type=float, default=0.995)
    parser.add_argument("--critic-target-tau", type=float, default=0.995)
    parser.add_argument("--critic-n-step", type=int, default=5)
    parser.add_argument("--cost-loss-coef", type=float, default=1.0)
    parser.add_argument("--critic-loss-coef", type=float, default=0.5)
    parser.add_argument("--repr-loss-coef", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--eval-max-steps", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=50000)
    parser.add_argument("--checkpoint-interval", type=int, default=100000)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--no-final-eval", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    save_dir = os.path.join(base_dir, "results", "lecun_ami_atari", f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving all results to: {save_dir}")

    write_config(args, save_dir)
    trained_agent, rewards, planning_rates = train(args, save_dir)
    save_checkpoint(trained_agent, args, save_dir, "model.pth", args.total_steps)

    summary_path = os.path.join(save_dir, "final_summary.json")
    with open(summary_path, "w") as handle:
        json.dump({
            "episodes": len(rewards),
            "reward_mean": mean_or_zero(rewards),
            "reward_last_100_mean": mean_or_zero(rewards[-100:]),
            "reward_max": float(np.max(rewards)) if rewards else 0.0,
            "planning_rate_mean": mean_or_zero(planning_rates),
            "planning_rate_last_100_mean": mean_or_zero(planning_rates[-100:]),
            "total_steps": args.total_steps,
        }, handle, indent=2, sort_keys=True)
    print(f"Final summary saved: {summary_path}")

    save_metrics(rewards, planning_rates, save_dir, args.env, no_plots=args.no_plots)
    if not args.no_final_eval:
        evaluate_and_record(trained_agent, args, save_dir)
