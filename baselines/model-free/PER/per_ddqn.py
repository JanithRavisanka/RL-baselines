import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
import datetime
import argparse
import sys
import time
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from atari_reporting import (  # noqa: E402
    EPISODE_FIELDS,
    EVAL_FIELDS,
    TRAINING_FIELDS,
    append_csv_row,
    ensure_csv,
    evaluate_q_policy,
    seed_everything,
    write_config,
    write_final_summary,
    write_json,
)

# Set up Hardware Acceleration
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


class DeepMindRMSprop(optim.Optimizer):
    """
    RMSProp variant used by the original DeepMind DQN implementation.

    PyTorch's RMSprop applies eps outside the square root and omits the DQN
    momentum accumulator unless configured separately. With eps=0.01, that is
    not the optimizer used in the paper.
    """
    def __init__(self, params, lr=2.5e-4, alpha=0.95, momentum=0.95, eps=0.01):
        defaults = dict(lr=lr, alpha=alpha, momentum=momentum, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            momentum = group["momentum"]
            eps = group["eps"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                state = self.state[param]

                if len(state) == 0:
                    state["square_avg"] = torch.zeros_like(param)
                    state["momentum_buffer"] = torch.zeros_like(param)

                square_avg = state["square_avg"]
                momentum_buffer = state["momentum_buffer"]

                square_avg.mul_(alpha).addcmul_(grad, grad, value=1.0 - alpha)
                denom = square_avg.add(eps).sqrt()
                momentum_buffer.mul_(momentum).addcdiv_(grad, denom, value=lr)
                param.add_(momentum_buffer, alpha=-1.0)

        return loss


class QNetwork(nn.Module):
    # Same as DDQN
    def __init__(self, action_dim):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
        return q_values

class SumTree:
    """
    A binary tree data structure where the parent's value is the sum of its children.
    Used for efficient O(log N) priority sampling in Prioritized Experience Replay.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        # Tree has 2 * capacity - 1 nodes
        # [0] is the root (total sum)
        # [1, 2] are its children, etc.
        # [capacity - 1 : 2*capacity - 1] are the leaf nodes storing the priorities
        self.tree = np.zeros(2 * capacity - 1)
        # Data array stores the actual memory transitions corresponding to the leaf nodes
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        # If we reach bottom, return index
        if left >= len(self.tree):
            return idx

        # Search left or right depending on the sum 's'
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        # Leaf index corresponding to current ring-buffer write pointer.
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplayBuffer:
    """
    Experience Replay Buffer using a SumTree to sample transitions based on TD-Error priority.
    """
    def __init__(self, capacity, alpha=0.6):
        # We need capacity to be a power of 2 for a perfect binary tree, 
        # but the SumTree implementation handles arbitrary sizes by keeping array bounds safe.
        self.tree = SumTree(capacity)
        self.alpha = alpha     # How much prioritization to use (0 = uniform random, 1 = full priority)
        self.epsilon = 0.01    # Small positive constant to ensure no transition has 0 priority
        self.capacity = capacity
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        # New transitions get maximum priority to guarantee they are sampled at least once
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = 1.0
            
        # Convert LazyFrames to numpy arrays to save memory
        state = np.array(state, dtype=np.uint8)
        next_state = np.array(next_state, dtype=np.uint8)
        
        self.tree.add(max_p, (state, action, reward, next_state, done))
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size, beta=0.4):
        # Stratified sampling over equal-priority mass segments.
        # This reduces variance versus drawing all samples from one interval.
        batch = []
        idxs = []
        priorities = []
        
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        # Importance Sampling correction:
        # High-priority samples are intentionally over-sampled, so we reweight loss
        # by (1 / (N * P(i)))^beta to reduce introduced bias.
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.size * sampling_probabilities, -beta)
        # Normalize weights so the maximum weight is 1.0
        is_weight /= is_weight.max()

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.bool_),
            idxs,
            np.array(is_weight, dtype=np.float32)
        )

    def update_priorities(self, idxs, td_errors):
        priorities = np.power(td_errors + self.epsilon, self.alpha)
        for idx, p in zip(idxs, priorities):
            self.tree.update(idx, p)
            
    def __len__(self):
        return self.size

def make_env(env_name="ALE/Breakout-v5", render_mode=None, terminal_on_life_loss=True):
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

def train(args, save_dir):
    """
    PER + Double DQN.

    Composition rationale:
    - Double DQN mitigates target overestimation.
    - PER focuses updates on transitions with higher TD error signal.
    - IS weights keep training approximately unbiased as beta -> 1.
    """
    env = make_env(args.env)
    seed_everything(args.seed, env)
    action_dim = env.action_space.n
    
    q_network = QNetwork(action_dim).to(device)
    target_network = QNetwork(action_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())
    
    optimizer = DeepMindRMSprop(q_network.parameters(), lr=2.5e-4, alpha=0.95, momentum=0.95, eps=0.01)
    
    # Hyperparameters
    # Directly storing stacked frames makes the paper's 1M replay buffer very
    # memory-heavy here; 100k keeps the prioritized replay mechanism practical.
    replay_buffer = PrioritizedReplayBuffer(capacity=args.replay_size, alpha=args.priority_alpha)
    batch_size = args.batch_size
    gamma = args.gamma
    
    epsilon_start = args.epsilon_start
    epsilon_end = args.epsilon_end
    # The wrapper repeats each selected action for 4 Atari frames, so 250k
    # agent decisions correspond to the paper's 1M-frame exploration anneal.
    epsilon_decay_steps = args.epsilon_decay_steps
    epsilon = epsilon_start
    
    # Beta annealing for Importance Sampling
    beta_start = args.priority_beta_start
    beta_frames = args.priority_beta_frames
    beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)
    
    max_frames = args.max_frames
    learning_starts = args.learning_starts
    target_update_frequency = args.target_update_frequency
    
    frame_idx = 0
    episode_rewards = []
    episode_index = 0
    episode_steps = 0
    start_time = time.time()
    metrics_path = os.path.join(save_dir, "metrics.csv")
    training_log_path = os.path.join(save_dir, "training_log.csv")
    eval_metrics_path = os.path.join(save_dir, "eval_metrics.csv")
    ensure_csv(metrics_path, EPISODE_FIELDS)
    ensure_csv(training_log_path, TRAINING_FIELDS)
    ensure_csv(eval_metrics_path, EVAL_FIELDS)
    
    state, _ = env.reset()
    episode_reward = 0
    
    print(f"Starting Prioritized Double DQN (PER DDQN) Training for {max_frames} frames...")
    
    while frame_idx < max_frames:
        frame_idx += 1
        
        # 1. Action Selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0).to(device) / 255.0
                q_values = q_network(state_tensor)
                action = q_values.argmax().item()
                
        # 2. Environment Step & Store Memory
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        clipped_reward = np.sign(reward)
        replay_buffer.add(state, action, clipped_reward, next_state, done)
        
        state = next_state
        episode_reward += reward
        episode_steps += 1
        
        if done:
            episode_index += 1
            append_csv_row(
                metrics_path,
                EPISODE_FIELDS,
                {
                    "episode": episode_index,
                    "global_step": frame_idx,
                    "reward": episode_reward,
                    "episode_steps": episode_steps,
                    "epsilon": epsilon,
                    "completed": 1,
                    "elapsed_sec": time.time() - start_time,
                },
            )
            state, _ = env.reset()
            episode_rewards.append(episode_reward)
            episode_reward = 0
            episode_steps = 0
            
        # 3. Decay Epsilon
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (frame_idx / epsilon_decay_steps))
        
        # 4. Prioritized Learning
        if len(replay_buffer) >= batch_size and frame_idx >= learning_starts:
            # Get annealed beta
            beta = beta_by_frame(frame_idx)
            
            # Sample using Priority and get Importance Sampling (IS) weights
            states_b, actions_b, rewards_b, next_states_b, dones_b, idxs, weights_b = replay_buffer.sample(batch_size, beta)
            
            states_tensor = torch.FloatTensor(states_b).to(device) / 255.0
            next_states_tensor = torch.FloatTensor(next_states_b).to(device) / 255.0
            actions_tensor = torch.LongTensor(actions_b).unsqueeze(1).to(device)
            rewards_tensor = torch.FloatTensor(rewards_b).unsqueeze(1).to(device)
            dones_tensor = torch.BoolTensor(dones_b).unsqueeze(1).to(device)
            weights_tensor = torch.FloatTensor(weights_b).unsqueeze(1).to(device)
            
            current_q_values = q_network(states_tensor).gather(1, actions_tensor)
            
            # Double DQN Target Calculation
            with torch.no_grad():
                best_next_actions = q_network(next_states_tensor).argmax(dim=1, keepdim=True)
                max_next_q_values = target_network(next_states_tensor).gather(1, best_next_actions)
                target_q_values = rewards_tensor + (gamma * max_next_q_values * (~dones_tensor))
                
            # TD error magnitude is the proxy for "learning progress potential".
            # Larger TD error => transition likely carries more corrective signal.
            td_errors = (target_q_values - current_q_values).abs().detach().cpu().numpy()
            
            # Update the SumTree with the new priorities
            replay_buffer.update_priorities(idxs, td_errors.flatten())
                
            # Compute Loss (Huber Loss / Smooth L1)
            # IMPORTANT: We multiply the loss by the IS Weights and use reduction='none' first
            elementwise_loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')
            loss = (weights_tensor * elementwise_loss).mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=10.0)
            optimizer.step()
            
        # 5. Update Target Network
        if frame_idx % target_update_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())
            
        if args.eval_interval > 0 and frame_idx % args.eval_interval == 0:
            eval_stats = evaluate_q_policy(
                q_network,
                make_env,
                device,
                args.env,
                frame_idx,
                args.eval_episodes,
                args.eval_max_steps,
                record_gif=False,
            )
            append_csv_row(eval_metrics_path, EVAL_FIELDS, eval_stats)
            print(
                f"Eval at frame {frame_idx}: "
                f"mean={eval_stats['eval_reward_mean']:.2f}, "
                f"max={eval_stats['eval_reward_max']:.2f}"
            )

        if args.checkpoint_interval > 0 and frame_idx % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_frame_{frame_idx}.pth")
            torch.save(q_network.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        if frame_idx % args.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-20:]) if episode_rewards else 0.0
            append_csv_row(
                training_log_path,
                TRAINING_FIELDS,
                {
                    "global_step": frame_idx,
                    "elapsed_sec": time.time() - start_time,
                    "episodes": len(episode_rewards),
                    "epsilon": epsilon,
                    "avg_reward_20": avg_reward,
                    "avg_reward_100": np.mean(episode_rewards[-100:]) if episode_rewards else 0.0,
                },
            )
            print(f"Frame: {frame_idx}/{max_frames} | Epsilon: {epsilon:.2f} | Avg Reward (Last 20): {avg_reward:.2f}")

    if episode_steps > 0:
        episode_index += 1
        episode_rewards.append(episode_reward)
        append_csv_row(
            metrics_path,
            EPISODE_FIELDS,
            {
                "episode": episode_index,
                "global_step": frame_idx,
                "reward": episode_reward,
                "episode_steps": episode_steps,
                "epsilon": epsilon,
                "completed": 0,
                "elapsed_sec": time.time() - start_time,
            },
        )

    env.close()
    return q_network, episode_rewards


def plot_rewards(rewards, save_dir):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.6)
    plt.title('PER DDQN Training on Breakout-v5')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    window = 20
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, len(rewards)), moving_avg, color='red', label='Moving Average (20 episodes)')
        plt.legend()

    plt.grid()
    save_path = os.path.join(save_dir, 'training_curve.png')
    plt.savefig(save_path)
    print(f"Training curve saved as '{save_path}'")

def evaluate_and_record(model, args, save_dir):
    filename = os.path.join(save_dir, 'breakout_per_ddqn_agent.gif')
    stats = evaluate_q_policy(
        model,
        make_env,
        device,
        args.env,
        args.max_frames,
        args.eval_episodes,
        args.eval_max_steps,
        record_gif=not args.no_gif,
        gif_path="" if args.no_gif else filename,
    )
    write_json(os.path.join(save_dir, "final_eval_metrics.json"), stats)
    append_csv_row(os.path.join(save_dir, "eval_metrics.csv"), EVAL_FIELDS, stats)
    print(f"Final evaluation mean over {args.eval_episodes}: {stats['eval_reward_mean']:.2f}")
    if not args.no_gif:
        print(f"Saved evaluation GIF to {filename}")
    return stats


def build_args():
    parser = argparse.ArgumentParser(description="PER Double DQN baseline for ALE/Breakout-v5")
    parser.add_argument("--env", type=str, default="ALE/Breakout-v5")
    parser.add_argument("--max-frames", type=int, default=1_000_000)
    parser.add_argument("--replay-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.1)
    parser.add_argument("--epsilon-decay-steps", type=int, default=250_000)
    parser.add_argument("--learning-starts", type=int, default=50_000)
    parser.add_argument("--target-update-frequency", type=int, default=10_000)
    parser.add_argument("--priority-alpha", type=float, default=0.6)
    parser.add_argument("--priority-beta-start", type=float, default=0.4)
    parser.add_argument("--priority-beta-frames", type=int, default=250_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-max-steps", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=50_000)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--no-final-eval", action="store_true")
    return parser.parse_args()

if __name__ == '__main__':
    args = build_args()
    seed_everything(args.seed)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    save_dir = os.path.join(base_dir, "results", "per_ddqn", f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving all results to: {save_dir}")
    write_config(args, save_dir, "per_ddqn", device)

    model, rewards = train(args, save_dir)
    
    model_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    write_final_summary(os.path.join(save_dir, "final_summary.json"), rewards, args.max_frames)
    if not args.no_plots:
        plot_rewards(rewards, save_dir)
    if not args.no_final_eval:
        evaluate_and_record(model, args, save_dir)
