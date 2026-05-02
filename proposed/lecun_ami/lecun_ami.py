"""
LeCun-AMI prototype for reinforcement learning.

This file implements a small, runnable subset of Yann LeCun's proposed
autonomous machine intelligence architecture:

1. Perception encoder:
   maps raw environment observations into compact latent states.
2. JEPA-style latent world model:
   predicts future latent states instead of reconstructing observations.
3. Cost model:
   predicts undesirable outcomes as costs; here cost is initially -reward.
4. Critic:
   estimates future cost-to-go so the planner does not need infinite rollouts.
5. Actor:
   provides fast reactive actions by imitating the planner.
6. Configurator:
   decides whether to use the actor or invoke deliberate latent-space planning.
7. Planner:
   performs bounded CEM/MPC search in latent space.

The implementation targets Pendulum-v1 by default because it is continuous,
low-dimensional, and comparable with existing DDPG/TD3/SAC/MPC baselines.
"""

import argparse
import csv
import datetime
import os
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ---------------------------------------------------------------------------
# Runtime device selection
# ---------------------------------------------------------------------------

def get_device():
    """Choose the fastest available PyTorch backend for local/server runs."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = get_device()
print(f"Using device: {device}")


# ---------------------------------------------------------------------------
# Replay memory
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Stores real environment transitions and optional planner demonstrations.

    Standard model-based learning needs real transitions:
        (state, action, reward, next_state, done)

    This implementation additionally stores `planner_action` when an action was
    selected by the planner. Those planner actions become supervised targets for
    the fast actor, which is the "distill planning into reactive policy" step.
    """

    def __init__(self, capacity):
        # A deque with maxlen automatically drops the oldest transition when full.
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, planner_action=None):
        """Insert one transition using float32 arrays for stable torch conversion."""
        self.buffer.append((
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            float(done),
            None if planner_action is None else np.asarray(planner_action, dtype=np.float32),
        ))

    def sample(self, batch_size):
        """Uniformly sample transitions for world-model, cost-model, and critic updates."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, _ = zip(*batch)
        return (
            torch.tensor(np.stack(states), dtype=torch.float32, device=device),
            torch.tensor(np.stack(actions), dtype=torch.float32, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1),
            torch.tensor(np.stack(next_states), dtype=torch.float32, device=device),
            torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1),
        )

    def sample_planner_demos(self, batch_size):
        """
        Sample only transitions where the planner was used.

        The actor is trained from these examples so it gradually approximates
        planner behavior without paying planning cost at every future step.
        """
        demos = [item for item in self.buffer if item[5] is not None]
        if len(demos) < batch_size:
            return None
        batch = random.sample(demos, batch_size)
        states = [item[0] for item in batch]
        actions = [item[5] for item in batch]
        return (
            torch.tensor(np.stack(states), dtype=torch.float32, device=device),
            torch.tensor(np.stack(actions), dtype=torch.float32, device=device),
        )

    @property
    def planner_demo_count(self):
        """Expose how many stored transitions can be used for actor imitation."""
        return sum(1 for item in self.buffer if item[5] is not None)

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Neural modules that correspond to the AMI architecture components
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    Perception module.

    Converts the raw Pendulum state [cos(theta), sin(theta), theta_dot] into a
    latent representation z_t. In image domains, this would be replaced by a CNN
    or ViT encoder, but the role is the same.
    """

    def __init__(self, state_dim, latent_dim, hidden_dim):
        super().__init__()
        # LayerNorm keeps the latent scale stable for prediction and planning.
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, state):
        """Return latent state z_t."""
        return self.net(state)


class LatentWorldModel(nn.Module):
    """
    JEPA-style predictive world model.

    It receives the current latent state and action, then predicts the next
    latent state. It does not reconstruct pixels or raw observations. The
    residual form `latent + delta` makes short-horizon dynamics easier to learn.
    """

    def __init__(self, latent_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, latent, action):
        """Predict z_{t+1} from z_t and a_t."""
        x = torch.cat([latent, action], dim=-1)
        return latent + self.net(x)


class CostModel(nn.Module):
    """
    Cost module.

    LeCun's architecture uses costs/objectives rather than only external reward.
    For a Gymnasium task, a practical first version is:
        cost = -reward
    Later, safety costs, uncertainty costs, or constraint costs can be added.
    """

    def __init__(self, latent_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent, action):
        """Predict immediate cost c(z_t, a_t)."""
        return self.net(torch.cat([latent, action], dim=-1))


class Critic(nn.Module):
    """
    Future cost-to-go estimator.

    The planner only rolls forward for a finite horizon. The critic estimates
    the remaining long-term cost after that horizon, acting as a terminal value.
    """

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
        """Predict V(z_t), interpreted as expected future cost."""
        return self.net(latent)


class Actor(nn.Module):
    """
    Fast reactive policy.

    This is the System-1-like part of the architecture. It maps latent states to
    continuous actions without search. It is trained by imitating actions chosen
    by the planner, so planning can be amortized over time.
    """

    def __init__(self, latent_dim, action_dim, hidden_dim, action_low, action_high):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        # Store action bounds as buffers so they move with the module device and
        # are saved in the state dict, but are not optimized as parameters.
        self.register_buffer("action_low", torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(action_high, dtype=torch.float32))

    def forward(self, latent):
        """Map tanh output from [-1, 1] into the environment action range."""
        squashed = self.net(latent)
        scale = (self.action_high - self.action_low) / 2.0
        center = (self.action_high + self.action_low) / 2.0
        return center + scale * squashed


# ---------------------------------------------------------------------------
# Representation learning utilities
# ---------------------------------------------------------------------------

def update_target_encoder(encoder, target_encoder, tau):
    """
    Exponential moving average target encoder.

    The target encoder provides a slowly moving prediction target for the
    JEPA-style latent objective, similar to BYOL/I-JEPA style target networks.
    """
    with torch.no_grad():
        for online, target in zip(encoder.parameters(), target_encoder.parameters()):
            target.data.mul_(tau).add_(online.data, alpha=1.0 - tau)


def latent_variance_loss(latent):
    """
    Simple anti-collapse regularizer for latent representations.

    A pure latent-prediction objective can collapse to constant vectors. This
    penalty encourages each latent dimension to keep at least unit-scale batch
    variation. It is intentionally lightweight for this FYP prototype.
    """
    if latent.shape[0] < 2:
        return torch.zeros((), device=latent.device)
    std = torch.sqrt(latent.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(1.0 - std).mean()


class AMIAgent:
    """
    Practical subset of LeCun's autonomous machine intelligence architecture:
    encoder, latent JEPA-style world model, cost model, actor, critic, and
    configurator-controlled planning.
    """

    def __init__(self, state_dim, action_dim, action_low, action_high, args):
        self.args = args
        self.action_dim = action_dim

        # Keep action bounds as tensors because both actor output clipping and
        # planner sampling happen in torch on the selected device.
        self.action_low = torch.tensor(action_low, dtype=torch.float32, device=device)
        self.action_high = torch.tensor(action_high, dtype=torch.float32, device=device)

        # Online encoder learns from gradients. Target encoder is only updated
        # by EMA and provides the stop-gradient representation target.
        self.encoder = Encoder(state_dim, args.latent_dim, args.hidden_dim).to(device)
        self.target_encoder = Encoder(state_dim, args.latent_dim, args.hidden_dim).to(device)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad_(False)

        # Ensemble of world models. Using more than one model gives a practical
        # uncertainty estimate: disagreement means the model is less confident.
        self.world_models = nn.ModuleList([
            LatentWorldModel(args.latent_dim, action_dim, args.hidden_dim).to(device)
            for _ in range(args.ensemble_size)
        ])

        # Cost, critic, and actor complete the AMI control loop:
        # predict what is bad, estimate long-term badness, and act quickly.
        self.cost_model = CostModel(args.latent_dim, action_dim, args.hidden_dim).to(device)
        self.critic = Critic(args.latent_dim, args.hidden_dim).to(device)
        self.actor = Actor(args.latent_dim, action_dim, args.hidden_dim, action_low, action_high).to(device)

        # Model optimizer updates representation, dynamics, cost, and critic.
        # Actor optimizer is separate because actor updates are supervised
        # imitation of planner actions, not part of the world-model loss.
        self.model_optimizer = optim.Adam(
            list(self.encoder.parameters())
            + list(self.world_models.parameters())
            + list(self.cost_model.parameters())
            + list(self.critic.parameters()),
            lr=args.model_lr,
        )
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)

    def encode(self, state):
        """Encode a single numpy environment state into a batched latent tensor."""
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        return self.encoder(state_t)

    def actor_action(self, state, add_noise=False):
        """
        Produce a fast action from the actor.

        During training, optional Gaussian noise keeps actor-only behavior from
        becoming completely deterministic before the actor has enough planner
        demonstrations. During evaluation, noise is disabled.
        """
        self.encoder.eval()
        self.actor.eval()
        with torch.no_grad():
            latent = self.encode(state)
            action = self.actor(latent).squeeze(0)
            if add_noise:
                noise = torch.randn_like(action) * self.args.actor_noise
                action = action + noise
            action = torch.max(torch.min(action, self.action_high), self.action_low)
        return action.cpu().numpy()

    def one_step_uncertainty(self, state):
        """
        Estimate model uncertainty for the configurator.

        The agent encodes the state, asks the current actor for an action, then
        measures disagreement among world-model ensemble predictions for the
        next latent. Higher disagreement means the state/action is less familiar.
        """
        self.encoder.eval()
        self.actor.eval()
        for model in self.world_models:
            model.eval()
        with torch.no_grad():
            latent = self.encode(state)
            action = self.actor(latent)
            preds = torch.stack([model(latent, action) for model in self.world_models], dim=0)
            return preds.var(dim=0, unbiased=False).mean().item()

    def should_plan(self, state, global_step, evaluate=False):
        """
        Configurator logic: decide whether to spend computation on planning.

        The default adaptive policy uses four gates:
        1. during warmup, always plan to collect high-quality demonstrations;
        2. if planning is forced, always plan;
        3. if actor-only mode is requested, never plan;
        4. otherwise plan periodically or when world-model uncertainty is high.
        """
        if evaluate:
            return self.args.eval_mode in ("planner", "adaptive")
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
        """
        Predict next latent using all world models.

        Returns both the ensemble mean prediction and a scalar uncertainty
        penalty based on ensemble variance.
        """
        preds = torch.stack([model(latent, action) for model in self.world_models], dim=0)
        return preds.mean(dim=0), preds.var(dim=0, unbiased=False).mean(dim=-1, keepdim=True)

    @torch.no_grad()
    def plan(self, state):
        """
        Deliberative planning module.

        Uses Cross-Entropy Method (CEM) over action sequences:
        - initialize a Gaussian over action sequences around the actor's action,
        - sample many candidate sequences,
        - roll each sequence through the latent world model,
        - keep the lowest-cost elite sequences,
        - refit the Gaussian to those elites,
        - execute only the first action of the best sequence.
        """
        self.encoder.eval()
        self.actor.eval()
        self.cost_model.eval()
        self.critic.eval()
        for model in self.world_models:
            model.eval()

        latent = self.encode(state)
        horizon = self.args.horizon
        sequences = self.args.num_sequences
        elite_count = max(1, int(sequences * self.args.elite_frac))

        # Center the first CEM distribution around the actor. This makes planning
        # refine the current reactive policy instead of starting from pure noise.
        actor_seed = self.actor(latent).squeeze(0)
        mean = actor_seed.repeat(horizon, 1)
        std = ((self.action_high - self.action_low) / 2.0).repeat(horizon, 1)
        best_sequence = None
        best_cost = None
        best_uncertainty = 0.0

        for _ in range(self.args.cem_iterations):
            # action_sequences has shape [horizon, num_sequences, action_dim].
            noise = torch.randn(horizon, sequences, self.action_dim, device=device)
            action_sequences = mean[:, None, :] + std[:, None, :] * noise
            action_sequences = torch.max(torch.min(action_sequences, self.action_high), self.action_low)

            # Lower cost is better because the objective is cost minimization.
            costs, uncertainties = self.rollout_cost(latent, action_sequences)
            elite_idx = torch.topk(costs, elite_count, largest=False).indices
            elites = action_sequences[:, elite_idx]

            # Refit CEM proposal distribution to the elite action sequences.
            mean = elites.mean(dim=1)
            std = elites.std(dim=1, unbiased=False).clamp_min(self.args.min_action_std)

            # Track the current best sequence so the first action can be applied.
            best_idx = torch.argmin(costs)
            best_sequence = action_sequences[:, best_idx]
            best_cost = costs[best_idx].item()
            best_uncertainty = uncertainties[best_idx].item()

        return best_sequence[0].cpu().numpy(), {
            "predicted_cost": best_cost,
            "uncertainty": best_uncertainty,
        }

    def rollout_cost(self, latent, action_sequences):
        """
        Score candidate action sequences inside the learned latent world.

        This is the core "imagination" path:
            z_t -> candidate actions -> predicted z futures -> predicted costs

        The planner minimizes:
            discounted predicted cost
            + uncertainty penalty
            + terminal critic cost
        """
        sequences = action_sequences.shape[1]
        latent_batch = latent.repeat(sequences, 1)
        total_cost = torch.zeros(sequences, 1, device=device)
        total_uncertainty = torch.zeros(sequences, 1, device=device)
        discount = 1.0

        for t in range(action_sequences.shape[0]):
            action_t = action_sequences[t]
            step_cost = self.cost_model(latent_batch, action_t)
            next_latent, uncertainty = self.ensemble_next(latent_batch, action_t)

            # Penalizing uncertainty discourages exploiting poorly learned model
            # regions, which is important for robust decision making.
            total_cost += discount * (step_cost + self.args.uncertainty_cost * uncertainty)
            total_uncertainty += uncertainty
            latent_batch = next_latent
            discount *= self.args.gamma

        # Terminal critic estimates cost beyond the finite planning horizon.
        total_cost += discount * self.critic(latent_batch)
        return total_cost.squeeze(1), total_uncertainty.squeeze(1) / action_sequences.shape[0]

    def update(self, replay):
        """
        Train all learnable components from replay.

        Updates are split into two parts:
        1. Representation/world/cost/critic update from real transitions.
        2. Actor imitation update from planner-labeled transitions.
        """
        if len(replay) < self.args.batch_size:
            return {}

        # Put predictive modules in training mode. The actor is updated later
        # only if enough planner demonstrations are available.
        self.encoder.train()
        self.cost_model.train()
        self.critic.train()
        for model in self.world_models:
            model.train()

        states, actions, rewards, next_states, dones = replay.sample(self.args.batch_size)
        cost_targets = -rewards
        latent = self.encoder(states)
        with torch.no_grad():
            # Stop-gradient target latent is the JEPA prediction target.
            target_next_latent = self.target_encoder(next_states)

        # Train every ensemble member to predict the same target latent. Ensemble
        # disagreement later provides uncertainty for the configurator/planner.
        world_loss = 0.0
        for model in self.world_models:
            pred_next_latent = model(latent, actions)
            world_loss = world_loss + F.mse_loss(pred_next_latent, target_next_latent)
        world_loss = world_loss / len(self.world_models)

        cost_loss = F.mse_loss(self.cost_model(latent, actions), cost_targets)
        value = self.critic(latent)
        with torch.no_grad():
            next_value = self.critic(target_next_latent)
            # Cost-space Bellman target: immediate cost + discounted future cost.
            critic_target = cost_targets + self.args.gamma * (1.0 - dones) * next_value
        critic_loss = F.mse_loss(value, critic_target)
        repr_loss = latent_variance_loss(latent)

        # Weighted objective keeps the implementation configurable for ablations.
        model_loss = (
            world_loss
            + self.args.cost_loss_coef * cost_loss
            + self.args.critic_loss_coef * critic_loss
            + self.args.repr_loss_coef * repr_loss
        )

        self.model_optimizer.zero_grad()
        model_loss.backward()
        # Clipping prevents unstable gradients from the coupled latent objectives.
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters())
            + list(self.world_models.parameters())
            + list(self.cost_model.parameters())
            + list(self.critic.parameters()),
            self.args.max_grad_norm,
        )
        self.model_optimizer.step()
        update_target_encoder(self.encoder, self.target_encoder, self.args.target_tau)

        # Actor distillation: once enough planned actions exist, train actor to
        # imitate them. This turns slow planning into fast reactive behavior.
        actor_loss_value = 0.0
        demo_batch = replay.sample_planner_demos(self.args.batch_size)
        if demo_batch is not None:
            demo_states, planner_actions = demo_batch
            self.encoder.eval()
            self.actor.train()
            with torch.no_grad():
                # Keep the encoder fixed for this supervised actor step.
                demo_latent = self.encoder(demo_states)
            actor_actions = self.actor(demo_latent)
            actor_loss = F.mse_loss(actor_actions, planner_actions)
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
        }


def select_action(agent, state, global_step, evaluate=False):
    """
    Single action-selection gateway used by training and evaluation.

    It asks the configurator whether to plan. If planning is selected, the
    action is planner-derived and stored later as a demonstration. Otherwise,
    the actor supplies a cheap reactive action.
    """
    plan_now = agent.should_plan(state, global_step, evaluate=evaluate)
    if plan_now:
        action, info = agent.plan(state)
        return action, True, info
    return agent.actor_action(state, add_noise=not evaluate), False, {
        "predicted_cost": None,
        "uncertainty": agent.one_step_uncertainty(state),
    }


def collect_seed_data(env, replay, seed_steps):
    """
    Fill replay with random transitions before learning begins.

    The world model cannot plan before it has seen any real dynamics. This seed
    dataset gives the encoder, world model, cost model, and critic enough data
    for their first supervised updates.
    """
    state, _ = env.reset()
    for _ in range(seed_steps):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay.add(state, action, reward, next_state, done)
        state = next_state
        if done:
            state, _ = env.reset()


def train(args):
    """
    Main training loop.

    High-level phases:
    1. Create environment, agent, and replay buffer.
    2. Collect random seed transitions.
    3. Pretrain the latent world model/cost/critic briefly.
    4. Run episodes where the configurator chooses actor or planner.
    5. Add real transitions to replay and update all modules online.
    """
    env = gym.make(args.env)
    env.action_space.seed(args.seed)

    # The default Pendulum observation is low-dimensional, but these dimensions
    # are read from the environment so the code remains Gymnasium-compatible.
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = AMIAgent(state_dim, action_dim, env.action_space.low, env.action_space.high, args)
    replay = ReplayBuffer(args.replay_size)

    print(f"Collecting {args.seed_steps} random seed transitions...")
    collect_seed_data(env, replay, args.seed_steps)

    # Initial updates reduce the chance that the first planner call is operating
    # on an entirely untrained latent dynamics model.
    print("Training initial latent world model...")
    latest_losses = {}
    for _ in range(args.initial_updates):
        latest_losses = agent.update(replay)

    episode_rewards = []
    planning_rates = []
    global_step = 0

    print("Starting LeCun-AMI training loop...")
    for episode in range(1, args.episodes + 1):
        state, _ = env.reset()
        done = False
        episode_reward = 0.0
        planned_steps = 0
        steps = 0

        while not done and steps < args.max_episode_steps:
            # The configurator decides whether this step uses deliberative
            # planning or the fast actor.
            action, planned, _ = select_action(agent, state, global_step, evaluate=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # If a planner produced the action, store it as a supervised target
            # for later actor distillation.
            replay.add(state, action, reward, next_state, done, planner_action=action if planned else None)

            episode_reward += reward
            planned_steps += int(planned)
            steps += 1
            global_step += 1
            state = next_state

            # Online updates let the world model improve as new transitions are
            # collected, rather than using a fixed offline model.
            for _ in range(args.updates_per_step):
                latest_losses = agent.update(replay)

        episode_rewards.append(episode_reward)
        planning_rate = planned_steps / max(1, steps)
        planning_rates.append(planning_rate)

        if episode % args.log_interval == 0 or episode == 1:
            avg_reward = np.mean(episode_rewards[-10:])
            print(
                f"Episode {episode}/{args.episodes} | "
                f"Reward: {episode_reward:.2f} | "
                f"Avg10: {avg_reward:.2f} | "
                f"Planning: {planning_rate:.2f} | "
                f"World loss: {latest_losses.get('world_loss', 0.0):.4f} | "
                f"Actor loss: {latest_losses.get('actor_loss', 0.0):.4f}"
            )

    env.close()
    return agent, episode_rewards, planning_rates


def plot_metrics(rewards, planning_rates, save_dir, no_plots=False):
    """
    Persist training metrics.

    CSV is always written because it is dependency-light and easy to analyze.
    Plots are optional because some local machines have incompatible matplotlib
    binaries; server training should still complete without plotting.
    """
    metrics_path = os.path.join(save_dir, "metrics.csv")
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
        # Do not fail a long training run just because plotting dependencies are
        # broken. The CSV already preserves the important metrics.
        print(f"Skipping plots because matplotlib could not be imported: {exc}")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.7, label="Episode reward")
    if len(rewards) >= 10:
        moving = np.convolve(rewards, np.ones(10) / 10, mode="valid")
        plt.plot(range(9, len(rewards)), moving, label="Moving average (10)")
    plt.title("LeCun-AMI Training on Pendulum-v1")
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
    """
    Run deterministic evaluation and save a GIF.

    Evaluation defaults to planner mode because the proposed method's strongest
    behavior is cost-guided planning. The CLI can switch this to actor-only or
    adaptive evaluation for ablation studies.
    """
    if args.no_gif:
        return

    import imageio

    env = gym.make(args.env, render_mode="rgb_array")
    frames = []
    rewards = []

    for episode in range(args.eval_episodes):
        # Use a different deterministic seed range from training.
        state, _ = env.reset(seed=args.seed + 1000 + episode)
        done = False
        total_reward = 0.0
        step = 0
        while not done and step < args.max_episode_steps:
            frames.append(env.render())
            action, _, _ = select_action(agent, state, step, evaluate=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1
        rewards.append(total_reward)

    env.close()
    gif_path = os.path.join(save_dir, "lecun_ami_pendulum_agent.gif")
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"Evaluation reward mean over {args.eval_episodes}: {np.mean(rewards):.2f}")
    print(f"Evaluation GIF saved: {gif_path}")


def build_args():
    """Define all experiment and ablation controls in one place."""
    parser = argparse.ArgumentParser(
        description="JEPA-inspired LeCun AMI prototype: latent world model, cost-guided planner, actor, and configurator."
    )

    # Environment and run length.
    parser.add_argument("--env", type=str, default="Pendulum-v1")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-episode-steps", type=int, default=200)

    # Replay and update schedule.
    parser.add_argument("--seed-steps", type=int, default=1000)
    parser.add_argument("--initial-updates", type=int, default=200)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--replay-size", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=128)

    # Neural network sizes.
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--ensemble-size", type=int, default=3)

    # CEM/MPC planner configuration.
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--num-sequences", type=int, default=256)
    parser.add_argument("--cem-iterations", type=int, default=4)
    parser.add_argument("--elite-frac", type=float, default=0.1)
    parser.add_argument("--min-action-std", type=float, default=0.05)

    # Configurator controls for actor/planner selection.
    parser.add_argument("--planning-mode", choices=["adaptive", "always", "actor"], default="adaptive")
    parser.add_argument("--eval-mode", choices=["adaptive", "planner", "actor"], default="planner")
    parser.add_argument("--warmup-plan-steps", type=int, default=1000)
    parser.add_argument("--planning-interval", type=int, default=5)
    parser.add_argument("--uncertainty-threshold", type=float, default=0.02)
    parser.add_argument("--uncertainty-cost", type=float, default=0.1)
    parser.add_argument("--actor-noise", type=float, default=0.1)

    # Learning objective coefficients and optimizers.
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--model-lr", type=float, default=3e-4)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--target-tau", type=float, default=0.99)
    parser.add_argument("--cost-loss-coef", type=float, default=1.0)
    parser.add_argument("--critic-loss-coef", type=float, default=0.5)
    parser.add_argument("--repr-loss-coef", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)

    # Logging and artifact controls.
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-gif", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    # Reproducibility seeds. Full determinism is not guaranteed across CUDA/MPS,
    # but this makes runs much easier to compare.
    args = build_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Match the repository convention: each run gets timestamped artifacts under
    # results/<algorithm>/run_<timestamp>/.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    save_dir = os.path.join(base_dir, "results", "lecun_ami", f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving all results to: {save_dir}")

    trained_agent, rewards, planning_rates = train(args)
    model_path = os.path.join(save_dir, "model.pth")
    # Save each module separately so later ablations can load only the encoder,
    # actor, planner models, or cost components if needed.
    torch.save({
        "encoder": trained_agent.encoder.state_dict(),
        "target_encoder": trained_agent.target_encoder.state_dict(),
        "world_models": [model.state_dict() for model in trained_agent.world_models],
        "cost_model": trained_agent.cost_model.state_dict(),
        "critic": trained_agent.critic.state_dict(),
        "actor": trained_agent.actor.state_dict(),
        "config": vars(args),
    }, model_path)
    print(f"Model saved: {model_path}")

    plot_metrics(rewards, planning_rates, save_dir, no_plots=args.no_plots)
    evaluate_and_record(trained_agent, args, save_dir)
