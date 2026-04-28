import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import imageio
import matplotlib.pyplot as plt
import os
import datetime
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

# Set up Hardware Acceleration
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

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

        # Importance Sampling (IS) weights calculations
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

def make_env(env_name="ALE/Breakout-v5", render_mode=None):
    env = gym.make(env_name, render_mode=render_mode, frameskip=1)
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=False, terminal_on_life_loss=True)
    env = FrameStackObservation(env, stack_size=4)
    return env

def train():
    env = make_env()
    action_dim = env.action_space.n
    
    q_network = QNetwork(action_dim).to(device)
    target_network = QNetwork(action_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())
    
    optimizer = optim.Adam(q_network.parameters(), lr=1e-4)
    
    # Hyperparameters
    replay_buffer = PrioritizedReplayBuffer(capacity=50000, alpha=0.6)
    batch_size = 32
    gamma = 0.99
    
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay_steps = 250000 
    epsilon = epsilon_start
    
    # Beta annealing for Importance Sampling
    beta_start = 0.4
    beta_frames = 250000
    beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)
    
    max_frames = 500000 
    target_update_frequency = 1000 
    
    frame_idx = 0
    episode_rewards = []
    
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
        
        if done:
            state, _ = env.reset()
            episode_rewards.append(episode_reward)
            episode_reward = 0
            
        # 3. Decay Epsilon
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (frame_idx / epsilon_decay_steps))
        
        # 4. Prioritized Learning
        if len(replay_buffer) > batch_size:
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
                
            # Compute TD Errors for Priority Update
            # Absolute difference between target and current Q values
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
            
        if frame_idx % 2000 == 0:
            avg_reward = np.mean(episode_rewards[-20:]) if episode_rewards else 0.0
            print(f"Frame: {frame_idx}/{max_frames} | Epsilon: {epsilon:.2f} | Avg Reward (Last 20): {avg_reward:.2f}")

    env.close()
    return q_network, episode_rewards


def plot_rewards(rewards, save_dir):
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

def evaluate_and_record(model, save_dir):
    filename = os.path.join(save_dir, 'breakout_per_ddqn_agent.gif')
    print(f"Evaluating agent and saving video to {filename}...")
    env = make_env(render_mode='rgb_array')
    state, _ = env.reset()
    frames = []
    done = False
    total_reward = 0
    
    with torch.no_grad():
        state, _, _, _, _ = env.step(1)
        frames.append(env.render())
        
        while not done:
            frames.append(env.render()) 
            state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0).to(device) / 255.0
            q_values = model(state_tensor)
            action = q_values.argmax().item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            if len(frames) > 500: 
                break
                
    env.close()
    print(f"Evaluation finished. Total Reward: {total_reward}")
    imageio.mimsave(filename, frames, fps=30)
    print("Saved successfully!")

if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    save_dir = os.path.join(base_dir, "results", "per_ddqn", f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving all results to: {save_dir}")

    model, rewards = train()
    
    model_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    plot_rewards(rewards, save_dir)
    evaluate_and_record(model, save_dir)
