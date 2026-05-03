import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
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
    """
    Deep Q-Network (DQN) architecture from the original DeepMind paper.
    Takes stacked 84x84 grayscale frames and outputs Q-values for each action.
    """
    def __init__(self, action_dim):
        super(QNetwork, self).__init__()
        
        # Convolutional Layers to process the 84x84x4 images
        # Input shape: (4 channels, 84, 84) -> 4 stacked frames
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        # Fully connected layers to process the flattened convolutional output
        # Output shape of conv3 is 64 channels x 7 x 7 image size = 3136
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, state):
        # We assume state is already normalized to [0, 1] before passing to the network
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the 3D tensor to 1D vector for the linear layers
        x = x.reshape(x.size(0), -1) 
        
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
        
        return q_values

class ReplayBuffer:
    """
    Experience Replay Buffer to store past experiences and sample random batches.
    This breaks correlation between consecutive states and stabilizes training.
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        # Convert LazyFrames to numpy arrays to save memory and avoid bugs
        state = np.array(state, dtype=np.uint8)
        next_state = np.array(next_state, dtype=np.uint8)
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.bool_)
        )
        
    def __len__(self):
        return len(self.buffer)

def make_env(env_name="ALE/Breakout-v5", render_mode=None, terminal_on_life_loss=True):
    """
    Creates the Atari environment and applies the necessary Wrappers.
    """
    env = gym.make(env_name, render_mode=render_mode, frameskip=1)
    
    # AtariPreprocessing applies: 
    # 1. Grayscale
    # 2. Resizes to 84x84
    # 3. Frameskip of 4 (Agent only plays every 4th frame, making it faster)
    env = AtariPreprocessing(
        env,
        frame_skip=4,
        grayscale_obs=True,
        scale_obs=False,
        terminal_on_life_loss=terminal_on_life_loss,
    )
    
    # FrameStack stacks the last 4 frames together to give the agent a sense of motion
    env = FrameStackObservation(env, stack_size=4)
    return env

def train(args, save_dir):
    """
    Double DQN training loop.

    Difference from vanilla DQN:
    - Online net selects argmax action at next state.
    - Target net evaluates that selected action value.
    This decoupling reduces positive overestimation bias.
    """
    env = make_env(args.env)
    seed_everything(args.seed, env)
    action_dim = env.action_space.n
    
    # Initialize the Primary Network (Used to select actions and trained at every step)
    q_network = QNetwork(action_dim).to(device)
    
    # Initialize the Target Network (Used to calculate stable targets, updated slowly)
    target_network = QNetwork(action_dim).to(device)
    target_network.load_state_dict(q_network.state_dict()) # Copy weights initially
    
    optimizer = DeepMindRMSprop(q_network.parameters(), lr=2.5e-4, alpha=0.95, momentum=0.95, eps=0.01)
    
    # Hyperparameters
    # Directly storing stacked frames makes the paper's 1M replay buffer very
    # memory-heavy here; 100k keeps the same replay mechanism for this baseline.
    replay_buffer = ReplayBuffer(capacity=args.replay_size)
    batch_size = args.batch_size
    gamma = args.gamma
    
    # Epsilon-Greedy Scheduler (Starts at 100% random, decays to 10% random)
    epsilon_start = args.epsilon_start
    epsilon_end = args.epsilon_end
    # The wrapper repeats each selected action for 4 Atari frames, so 250k
    # agent decisions correspond to the paper's 1M-frame exploration anneal.
    epsilon_decay_steps = args.epsilon_decay_steps
    epsilon = epsilon_start
    
    # Training Loop variables
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
    
    print(f"Starting Double DQN Training for {max_frames} frames...")
    
    while frame_idx < max_frames:
        frame_idx += 1
        
        # --- 1. Choose an Action (Epsilon-Greedy) ---
        if random.random() < epsilon:
            # Explore randomly
            action = env.action_space.sample()
        else:
            # Exploit learned Q-values
            with torch.no_grad():
                # state is shape (4, 84, 84). We add batch dimension -> (1, 4, 84, 84)
                # We also scale pixels to [0, 1] range before passing to the network
                state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0).to(device) / 255.0
                q_values = q_network(state_tensor)
                action = q_values.argmax().item()
                
        # --- 2. Take Step and Store in Replay Buffer ---
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Clip reward to [-1, 1] (Standard practice for Atari to stabilize training across games)
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
            
        # --- 3. Decay Epsilon ---
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (frame_idx / epsilon_decay_steps))
        
        # --- 4. Learn from Replay Buffer ---
        # Only start learning once we have enough memories in the buffer
        if len(replay_buffer) >= batch_size and frame_idx >= learning_starts:
            # Sample a random batch of memories
            states_b, actions_b, rewards_b, next_states_b, dones_b = replay_buffer.sample(batch_size)
            
            # Convert to PyTorch Tensors and move to Device (and normalize pixels)
            states_tensor = torch.FloatTensor(states_b).to(device) / 255.0
            next_states_tensor = torch.FloatTensor(next_states_b).to(device) / 255.0
            actions_tensor = torch.LongTensor(actions_b).unsqueeze(1).to(device)
            rewards_tensor = torch.FloatTensor(rewards_b).unsqueeze(1).to(device)
            dones_tensor = torch.BoolTensor(dones_b).unsqueeze(1).to(device)
            
            # Current Q-Values: What does the main network think the action we took was worth?
            current_q_values = q_network(states_tensor).gather(1, actions_tensor)
            
            # --- DOUBLE DQN TARGET CALCULATION ---
            with torch.no_grad():
                # 1) Action selection with online network:
                #    a* = argmax_a Q_online(s', a)
                best_next_actions = q_network(next_states_tensor).argmax(dim=1, keepdim=True)
                
                # 2) Action evaluation with target network:
                #    Q_target(s', a*)
                max_next_q_values = target_network(next_states_tensor).gather(1, best_next_actions)
                
                # Bellman Equation: Target = Reward + Gamma * Max Next Q
                # If done, there is no next Q value!
                target_q_values = rewards_tensor + (gamma * max_next_q_values * (~dones_tensor))
                
            # Compute Loss (Huber Loss / Smooth L1)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=10.0)
            optimizer.step()
            
        # --- 5. Update Target Network ---
        if frame_idx % target_update_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())
            
        # Logging
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
    plt.title('Double DQN Training on Breakout-v5')
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
    filename = os.path.join(save_dir, 'breakout_ddqn_agent.gif')
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
    parser = argparse.ArgumentParser(description="Double DQN baseline for ALE/Breakout-v5")
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
    save_dir = os.path.join(base_dir, "results", "ddqn", f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving all results to: {save_dir}")
    write_config(args, save_dir, "ddqn", device)

    model, rewards = train(args, save_dir)
    
    model_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    write_final_summary(os.path.join(save_dir, "final_summary.json"), rewards, args.max_frames)
    if not args.no_plots:
        plot_rewards(rewards, save_dir)
    if not args.no_final_eval:
        evaluate_and_record(model, args, save_dir)
