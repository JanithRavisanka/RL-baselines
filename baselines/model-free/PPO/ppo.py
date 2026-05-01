import argparse
import datetime
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PPOActorCritic(nn.Module):
    """
    PPO clipped-objective actor-critic with separate policy/value heads.

    The PPO paper defines the algorithmic constraint through the clipped
    probability-ratio surrogate. CartPole uses vector observations, so this MLP
    replaces the larger MuJoCo/Atari front-ends while preserving policy/value
    heads and stochastic action sampling.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy = nn.Linear(hidden_dim, action_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        feat = self.shared(state)
        return self.policy(feat), self.value(feat)

    def dist(self, state):
        logits, value = self.forward(state)
        return Categorical(logits=logits), value


def compute_gae(rewards, dones, values, last_value, gamma, lam):
    advantages = []
    gae = 0.0
    values_ext = values + [last_value]
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values_ext[t + 1] * nonterminal - values_ext[t]
        gae = delta + gamma * lam * nonterminal * gae
        advantages.append(gae)
    advantages.reverse()
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


def train(args):
    dev = device()
    env = gym.make(args.env)
    model = PPOActorCritic(env.observation_space.shape[0], env.action_space.n, args.hidden_dim).to(dev)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    state, _ = env.reset(seed=args.seed)
    episode_rewards = []
    current_episode_reward = 0.0

    for update in range(args.updates):
        states, actions, rewards, dones, old_log_probs, values = [], [], [], [], [], []

        for _ in range(args.rollout_steps):
            state_t = torch.tensor(state, dtype=torch.float32, device=dev).unsqueeze(0)
            with torch.no_grad():
                dist, value = model.dist(state_t)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            current_episode_reward += reward

            states.append(state)
            actions.append(action.item())
            rewards.append(float(reward))
            dones.append(float(done))
            old_log_probs.append(log_prob.item())
            values.append(value.item())

            state = next_state
            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                state, _ = env.reset()

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=dev).unsqueeze(0)
            _, last_value = model.dist(state_t)
            last_value = last_value.item()

        advantages, returns = compute_gae(rewards, dones, values, last_value, args.gamma, args.gae_lambda)
        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=dev)
        actions_t = torch.tensor(actions, dtype=torch.long, device=dev)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32, device=dev)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=dev)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=dev)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)

        indices = np.arange(args.rollout_steps)
        for _ in range(args.epochs):
            np.random.shuffle(indices)
            for start in range(0, args.rollout_steps, args.minibatch_size):
                mb_idx = indices[start:start + args.minibatch_size]
                dist, value = model.dist(states_t[mb_idx])
                new_log_probs = dist.log_prob(actions_t[mb_idx])
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_log_probs - old_log_probs_t[mb_idx])
                unclipped = ratio * adv_t[mb_idx]
                clipped = torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef) * adv_t[mb_idx]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(value.squeeze(-1), returns_t[mb_idx])
                loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

        if (update + 1) % args.log_interval == 0:
            avg = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
            print(f"Update {update + 1}/{args.updates} | Avg reward (last 10): {avg:.2f}")
            if len(episode_rewards) >= 20 and np.mean(episode_rewards[-20:]) >= 475:
                print("Solved.")
                break

    env.close()
    return model, episode_rewards


def plot_rewards(rewards, save_dir):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    if len(rewards) >= 20:
        moving = np.convolve(rewards, np.ones(20) / 20, mode="valid")
        plt.plot(range(19, len(rewards)), moving, label="Moving average (20)")
        plt.legend()
    plt.title("PPO Training on CartPole-v1")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid()
    out = os.path.join(save_dir, "training_curve.png")
    plt.savefig(out)
    print(f"Training curve saved: {out}")


def evaluate_and_record(model, save_dir, env_name="CartPole-v1"):
    import imageio

    dev = next(model.parameters()).device
    env = gym.make(env_name, render_mode="rgb_array")
    state, _ = env.reset()
    done = False
    total_reward = 0.0
    frames = []
    model.eval()

    with torch.no_grad():
        while not done:
            frames.append(env.render())
            state_t = torch.tensor(state, dtype=torch.float32, device=dev).unsqueeze(0)
            logits, _ = model(state_t)
            action = torch.argmax(logits, dim=-1).item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

    env.close()
    filename = os.path.join(save_dir, "ppo_cartpole_agent.gif")
    imageio.mimsave(filename, frames, fps=30)
    print(f"Evaluation reward: {total_reward:.2f}")
    print(f"Evaluation GIF saved: {filename}")


def build_args():
    parser = argparse.ArgumentParser(description="PPO clipped-surrogate baseline from Schulman et al. 2017")
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    save_dir = os.path.join(base_dir, "results", "ppo", f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving all results to: {save_dir}")

    model, rewards = train(args)
    model_path = os.path.join(save_dir, "model.pth")
    torch.save({"model_state_dict": model.state_dict(), "config": vars(args)}, model_path)
    print(f"Model saved: {model_path}")
    plot_rewards(rewards, save_dir)
    evaluate_and_record(model, save_dir, args.env)
