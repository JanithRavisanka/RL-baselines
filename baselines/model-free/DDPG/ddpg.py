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


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Actor(nn.Module):
    """DDPG deterministic actor: 400-300 ReLU MLP with tanh-scaled action."""

    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = float(max_action)
        self.fc1 = nn.Linear(state_dim, 400)
        self.bn1 = nn.BatchNorm1d(400)
        self.fc2 = nn.Linear(400, 300)
        self.bn2 = nn.BatchNorm1d(300)
        self.fc3 = nn.Linear(300, action_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.max_action * torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """DDPG critic: state pathway first, action injected at second hidden layer."""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.bn1 = nn.BatchNorm1d(400)
        self.fc2 = nn.Linear(400 + action_dim, 300)
        self.fc3 = nn.Linear(300, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc3.weight.data.uniform_(-3e-4, 3e-4)
        self.fc3.bias.data.uniform_(-3e-4, 3e-4)

    def forward(self, state, action):
        s = F.relu(self.bn1(self.fc1(state)))
        x = torch.cat([s, action], dim=-1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class OUNoise:
    """Ornstein-Uhlenbeck process used by the DDPG paper for exploration."""

    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(action_dim) * mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state = self.state + dx
        return self.state


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
    for src_buffer, tgt_buffer in zip(source.buffers(), target.buffers()):
        if torch.is_floating_point(tgt_buffer):
            tgt_buffer.data.mul_(1.0 - tau).add_(tau * src_buffer.data)
        else:
            tgt_buffer.data.copy_(src_buffer.data)


def hard_update(source, target):
    target.load_state_dict(source.state_dict())


def select_action(actor, state, dev, noise=None, low=None, high=None):
    was_training = actor.training
    actor.eval()
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32, device=dev).unsqueeze(0)
        action = actor(state_t).cpu().numpy()[0]
    if was_training:
        actor.train()
    if noise is not None:
        action = action + noise.sample()
    if low is not None and high is not None:
        action = np.clip(action, low, high)
    return action


def train(args):
    dev = device()
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    actor = Actor(state_dim, action_dim, max_action).to(dev)
    critic = Critic(state_dim, action_dim).to(dev)
    target_actor = Actor(state_dim, action_dim, max_action).to(dev)
    target_critic = Critic(state_dim, action_dim).to(dev)
    hard_update(actor, target_actor)
    hard_update(critic, target_critic)
    target_actor.eval()
    target_critic.eval()

    actor_opt = optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_opt = optim.Adam(critic.parameters(), lr=args.critic_lr, weight_decay=args.weight_decay)
    replay = ReplayBuffer(args.replay_size)
    noise = OUNoise(action_dim, sigma=args.ou_sigma)
    episode_rewards = []
    total_steps = 0

    for episode in range(args.episodes):
        state, _ = env.reset()
        noise.reset()
        done = False
        total_reward = 0.0

        while not done:
            if total_steps < args.start_steps:
                action = env.action_space.sample()
            else:
                action = select_action(actor, state, dev, noise, env.action_space.low, env.action_space.high)

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

                with torch.no_grad():
                    next_actions = target_actor(next_states)
                    target_q = target_critic(next_states, next_actions)
                    target_q = rewards + args.gamma * (1.0 - dones) * target_q

                critic_loss = F.mse_loss(critic(states, actions), target_q)
                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()

                actor_loss = -critic(states, actor(states)).mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()

                soft_update(actor, target_actor, args.tau)
                soft_update(critic, target_critic, args.tau)

        episode_rewards.append(total_reward)
        if (episode + 1) % args.log_interval == 0:
            print(f"Episode {episode + 1}/{args.episodes} | Avg reward: {np.mean(episode_rewards[-args.log_interval:]):.2f}")

    env.close()
    return actor, critic, episode_rewards


def plot_rewards(rewards, save_dir):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    if len(rewards) >= 10:
        moving = np.convolve(rewards, np.ones(10) / 10, mode="valid")
        plt.plot(range(9, len(rewards)), moving, label="Moving average (10)")
        plt.legend()
    plt.title("DDPG Training on Pendulum-v1")
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
        action = select_action(actor, state, dev, None, env.action_space.low, env.action_space.high)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    env.close()
    filename = os.path.join(save_dir, "ddpg_pendulum_agent.gif")
    imageio.mimsave(filename, frames, fps=30)
    print(f"Evaluation reward: {total_reward:.2f}")
    print(f"Evaluation GIF saved: {filename}")


def build_args():
    parser = argparse.ArgumentParser(description="DDPG baseline from Lillicrap et al. 2015")
    parser.add_argument("--env", type=str, default="Pendulum-v1")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--start-steps", type=int, default=1000)
    parser.add_argument("--replay-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.001)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--ou-sigma", type=float, default=0.2)
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
    save_dir = os.path.join(base_dir, "results", "ddpg", f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving all results to: {save_dir}")

    actor, critic, rewards = train(args)
    model_path = os.path.join(save_dir, "model.pth")
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "config": vars(args),
        },
        model_path,
    )
    print(f"Model saved: {model_path}")
    plot_rewards(rewards, save_dir)
    evaluate_and_record(actor, save_dir, args.env)
