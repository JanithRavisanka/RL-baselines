import argparse
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
from torch.distributions import Normal


LOG_STD_MIN = -20
LOG_STD_MAX = 2


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SquashedGaussianActor(nn.Module):
    """SAC stochastic actor with tanh-squashed Gaussian actions."""

    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super().__init__()
        self.max_action = float(max_action)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        raw_action = normal.rsample()
        squashed = torch.tanh(raw_action)
        action = self.max_action * squashed
        log_prob = normal.log_prob(raw_action)
        log_prob -= torch.log(self.max_action * (1.0 - squashed.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        deterministic = self.max_action * torch.tanh(mean)
        return action, log_prob, deterministic

    def deterministic(self, state):
        mean, _ = self.forward(state)
        return self.max_action * torch.tanh(mean)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.float32),
            torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(-1),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(-1),
        )

    def __len__(self):
        return len(self.buffer)


def soft_update(source, target, tau):
    for src_param, tgt_param in zip(source.parameters(), target.parameters()):
        tgt_param.data.mul_(1.0 - tau).add_(tau * src_param.data)


def select_action(actor, state, dev, deterministic=True, low=None, high=None):
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32, device=dev).unsqueeze(0)
        if deterministic:
            action = actor.deterministic(state_t)
        else:
            action, _, _ = actor.sample(state_t)
        action = action.cpu().numpy()[0]
    if low is not None and high is not None:
        action = np.clip(action, low, high)
    return action


def train(args):
    dev = device()
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    actor = SquashedGaussianActor(state_dim, action_dim, max_action, args.hidden_dim).to(dev)
    q1 = QNetwork(state_dim, action_dim, args.hidden_dim).to(dev)
    q2 = QNetwork(state_dim, action_dim, args.hidden_dim).to(dev)
    q1_target = QNetwork(state_dim, action_dim, args.hidden_dim).to(dev)
    q2_target = QNetwork(state_dim, action_dim, args.hidden_dim).to(dev)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=args.lr)
    q1_opt = optim.Adam(q1.parameters(), lr=args.lr)
    q2_opt = optim.Adam(q2.parameters(), lr=args.lr)
    log_alpha = torch.tensor(np.log(args.alpha), dtype=torch.float32, device=dev, requires_grad=True)
    alpha_opt = optim.Adam([log_alpha], lr=args.lr)
    target_entropy = -float(action_dim)
    replay = ReplayBuffer(args.replay_size)
    episode_rewards = []
    total_steps = 0

    for episode in range(args.episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            if total_steps < args.start_steps:
                action = env.action_space.sample()
            else:
                action = select_action(actor, state, dev, deterministic=False, low=env.action_space.low, high=env.action_space.high)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay.add(state, action, reward, next_state, float(done))
            state = next_state
            total_reward += reward
            total_steps += 1

            if len(replay) >= args.batch_size:
                states, actions, rewards, next_states, dones = replay.sample(args.batch_size)
                states = states.to(dev)
                actions = actions.to(dev)
                rewards = rewards.to(dev)
                next_states = next_states.to(dev)
                dones = dones.to(dev)
                alpha = log_alpha.exp()

                with torch.no_grad():
                    next_actions, next_log_probs, _ = actor.sample(next_states)
                    target_q = torch.min(q1_target(next_states, next_actions), q2_target(next_states, next_actions))
                    target = rewards + args.gamma * (1.0 - dones) * (target_q - alpha * next_log_probs)

                q1_loss = F.mse_loss(q1(states, actions), target)
                q2_loss = F.mse_loss(q2(states, actions), target)
                q1_opt.zero_grad()
                q1_loss.backward()
                q1_opt.step()
                q2_opt.zero_grad()
                q2_loss.backward()
                q2_opt.step()

                new_actions, log_probs, _ = actor.sample(states)
                q_new = torch.min(q1(states, new_actions), q2(states, new_actions))
                actor_loss = (alpha.detach() * log_probs - q_new).mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()

                alpha_loss = -(log_alpha * (log_probs + target_entropy).detach()).mean()
                alpha_opt.zero_grad()
                alpha_loss.backward()
                alpha_opt.step()

                soft_update(q1, q1_target, args.tau)
                soft_update(q2, q2_target, args.tau)

        episode_rewards.append(total_reward)
        if (episode + 1) % args.log_interval == 0:
            print(f"Episode {episode + 1}/{args.episodes} | Avg reward: {np.mean(episode_rewards[-args.log_interval:]):.2f}")

    env.close()
    return actor, q1, q2, episode_rewards


def plot_rewards(rewards, save_dir):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    if len(rewards) >= 10:
        moving = np.convolve(rewards, np.ones(10) / 10, mode="valid")
        plt.plot(range(9, len(rewards)), moving, label="Moving average (10)")
        plt.legend()
    plt.title("SAC Training on Pendulum-v1")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid()
    out = os.path.join(save_dir, "training_curve.png")
    plt.savefig(out)
    print(f"Training curve saved: {out}")


def evaluate_and_record(actor, save_dir, env_name="Pendulum-v1"):
    import imageio

    dev = next(actor.parameters()).device
    env = gym.make(env_name, render_mode="rgb_array")
    state, _ = env.reset()
    done = False
    total_reward = 0.0
    frames = []

    while not done:
        frames.append(env.render())
        action = select_action(actor, state, dev, deterministic=True, low=env.action_space.low, high=env.action_space.high)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    env.close()
    filename = os.path.join(save_dir, "sac_pendulum_agent.gif")
    imageio.mimsave(filename, frames, fps=30)
    print(f"Evaluation reward: {total_reward:.2f}")
    print(f"Evaluation GIF saved: {filename}")


def build_args():
    parser = argparse.ArgumentParser(description="SAC maximum-entropy actor-critic baseline")
    parser.add_argument("--env", type=str, default="Pendulum-v1")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--start-steps", type=int, default=1000)
    parser.add_argument("--replay-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    save_dir = os.path.join(base_dir, "results", "sac", f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving all results to: {save_dir}")

    actor, q1, q2, rewards = train(args)
    model_path = os.path.join(save_dir, "model.pth")
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "q1_state_dict": q1.state_dict(),
            "q2_state_dict": q2.state_dict(),
            "config": vars(args),
        },
        model_path,
    )
    print(f"Model saved: {model_path}")
    plot_rewards(rewards, save_dir)
    evaluate_and_record(actor, save_dir, args.env)
