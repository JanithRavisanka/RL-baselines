import argparse
import datetime
import os
import torch.multiprocessing as mp

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """
    A3C-style shared policy/value network for low-dimensional control.

    The original A3C paper uses separate policy and value heads over a shared
    representation. For CartPole's vector observations, an MLP replaces the Atari
    convolutional front-end while keeping the same actor/value factorization.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc = nn.Linear(state_dim, hidden_dim)
        self.policy = nn.Linear(hidden_dim, action_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc(state))
        return self.policy(x), self.value(x)


class SharedAdam(optim.Adam):
    """Adam optimizer with shared state tensors for multiprocessing workers."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr=lr, betas=betas, eps=eps)
        for group in self.param_groups:
            for param in group["params"]:
                state = self.state[param]
                state["step"] = torch.zeros(1)
                state["exp_avg"] = torch.zeros_like(param.data)
                state["exp_avg_sq"] = torch.zeros_like(param.data)
                state["step"].share_memory_()
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()


def ensure_shared_grads(local_model, global_model):
    for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
        global_param._grad = local_param.grad


def compute_returns(rewards, bootstrap_value, gamma):
    returns = []
    value = bootstrap_value
    for reward in reversed(rewards):
        value = reward + gamma * value
        returns.append(value)
    returns.reverse()
    return returns


def worker_loop(rank, args, global_model, optimizer, global_step, reward_queue):
    env = gym.make(args.env)
    local_model = ActorCritic(env.observation_space.shape[0], env.action_space.n, args.hidden_dim)
    episode_reward = 0.0
    state, _ = env.reset(seed=args.seed + rank)

    while True:
        with global_step.get_lock():
            if global_step.value >= args.max_steps:
                break

        local_model.load_state_dict(global_model.state_dict())
        log_probs, values, rewards, entropies = [], [], [], []
        done = False

        for _ in range(args.rollout_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits, value = local_model(state_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            episode_reward += reward

            log_probs.append(dist.log_prob(action))
            values.append(value.squeeze(0).squeeze(-1))
            rewards.append(float(reward))
            entropies.append(dist.entropy())

            with global_step.get_lock():
                global_step.value += 1
                reached_limit = global_step.value >= args.max_steps

            if done:
                reward_queue.put(episode_reward)
                episode_reward = 0.0
                next_state, _ = env.reset()

            state = next_state
            if done or reached_limit:
                break

        with torch.no_grad():
            if done:
                bootstrap = torch.tensor(0.0)
            else:
                next_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                _, next_value = local_model(next_tensor)
                bootstrap = next_value.squeeze()

        returns = compute_returns(rewards, bootstrap, args.gamma)
        returns = torch.stack([r if torch.is_tensor(r) else torch.tensor(r) for r in returns])
        values_t = torch.stack(values)
        log_probs_t = torch.stack(log_probs)
        entropy_t = torch.stack(entropies)
        advantage = returns.detach() - values_t

        policy_loss = -(log_probs_t * advantage.detach()).mean()
        value_loss = 0.5 * advantage.pow(2).mean()
        entropy_loss = entropy_t.mean()
        loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), args.max_grad_norm)
        ensure_shared_grads(local_model, global_model)
        optimizer.step()

    env.close()
    reward_queue.put(None)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = gym.make(args.env)
    global_model = ActorCritic(env.observation_space.shape[0], env.action_space.n, args.hidden_dim)
    env.close()
    global_model.share_memory()
    optimizer = SharedAdam(global_model.parameters(), lr=args.lr)
    global_step = mp.Value("i", 0)
    reward_queue = mp.Queue()

    workers = [
        mp.Process(target=worker_loop, args=(rank, args, global_model, optimizer, global_step, reward_queue))
        for rank in range(args.workers)
    ]
    for worker in workers:
        worker.start()

    episode_rewards = []
    finished = 0
    while finished < args.workers:
        item = reward_queue.get()
        if item is None:
            finished += 1
            continue
        episode_rewards.append(float(item))
        if len(episode_rewards) % 10 == 0:
            avg = np.mean(episode_rewards[-10:])
            print(f"Episode {len(episode_rewards)} | Avg reward (last 10): {avg:.2f} | steps {global_step.value}")

    for worker in workers:
        worker.join()
    return global_model, episode_rewards


def plot_rewards(rewards, save_dir):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    if len(rewards) >= 20:
        moving = np.convolve(rewards, np.ones(20) / 20, mode="valid")
        plt.plot(range(19, len(rewards)), moving, label="Moving average (20)")
        plt.legend()
    plt.title("A3C Training on CartPole-v1")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid()
    out = os.path.join(save_dir, "training_curve.png")
    plt.savefig(out)
    print(f"Training curve saved: {out}")


def evaluate_and_record(model, save_dir, env_name="CartPole-v1"):
    import imageio

    env = gym.make(env_name, render_mode="rgb_array")
    state, _ = env.reset()
    done = False
    total_reward = 0.0
    frames = []
    model.eval()

    with torch.no_grad():
        while not done:
            frames.append(env.render())
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits, _ = model(state_t)
            action = torch.argmax(logits, dim=-1).item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

    env.close()
    filename = os.path.join(save_dir, "a3c_cartpole_agent.gif")
    imageio.mimsave(filename, frames, fps=30)
    print(f"Evaluation reward: {total_reward:.2f}")
    print(f"Evaluation GIF saved: {filename}")


def build_args():
    parser = argparse.ArgumentParser(description="A3C baseline from Mnih et al. 2016")
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--workers", type=int, default=max(2, min(4, os.cpu_count() or 2)))
    parser.add_argument("--max-steps", type=int, default=200_000)
    parser.add_argument("--rollout-steps", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    args = build_args()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    save_dir = os.path.join(base_dir, "results", "a3c", f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving all results to: {save_dir}")

    model, rewards = train(args)
    model_path = os.path.join(save_dir, "model.pth")
    torch.save({"model_state_dict": model.state_dict(), "config": vars(args)}, model_path)
    print(f"Model saved: {model_path}")
    plot_rewards(rewards, save_dir)
    evaluate_and_record(model, save_dir, args.env)
