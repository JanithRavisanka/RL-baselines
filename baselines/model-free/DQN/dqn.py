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

def make_env(env_name="ALE/Breakout-v5", render_mode=None):
    """
    Creates the Atari environment and applies the necessary Wrappers.
    """
    env = gym.make(env_name, render_mode=render_mode, frameskip=1)
    
    # AtariPreprocessing applies: 
    # 1. Grayscale
    # 2. Resizes to 84x84
    # 3. Frameskip of 4 (Agent only plays every 4th frame, making it faster)
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=False, terminal_on_life_loss=True)
    
    # FrameStack stacks the last 4 frames together to give the agent a sense of motion
    env = FrameStackObservation(env, stack_size=4)
    return env

def train():
    """
    Classic DQN training loop for Atari with replay + target network.

    Key stability ingredients:
    - Experience replay breaks temporal correlations in online Atari streams.
    - A lagged target network prevents chasing a moving bootstrap target.
    - Reward clipping standardizes scale across sparse/dense reward regimes.
    """
    env = make_env()
    action_dim = env.action_space.n
    
    # Initialize the Primary Network (Used to select actions and trained at every step)
    q_network = QNetwork(action_dim).to(device)
    
    # Initialize the Target Network (Used to calculate stable targets, updated slowly)
    target_network = QNetwork(action_dim).to(device)
    target_network.load_state_dict(q_network.state_dict()) # Copy weights initially
    
    optimizer = optim.Adam(q_network.parameters(), lr=1e-4)
    
    # Hyperparameters
    replay_buffer = ReplayBuffer(capacity=50000) # Increased capacity for better memory
    batch_size = 32
    gamma = 0.99
    
    # Epsilon-Greedy Scheduler (Starts at 100% random, decays to 10% random)
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay_steps = 250000 # Decay over 250k frames
    epsilon = epsilon_start
    
    # Training Loop variables
    max_frames = 500000 # Increased to 500k frames (will take some time, but agent will learn)
    target_update_frequency = 1000 # How often to copy Q-Net weights to Target-Net
    
    frame_idx = 0
    episode_rewards = []
    
    state, _ = env.reset()
    episode_reward = 0
    
    print(f"Starting DQN Training for {max_frames} frames...")
    
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
        
        if done:
            state, _ = env.reset()
            episode_rewards.append(episode_reward)
            episode_reward = 0
            
        # --- 3. Decay Epsilon ---
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (frame_idx / epsilon_decay_steps))
        
        # --- 4. Learn from Replay Buffer ---
        # Only start learning once we have enough memories in the buffer
        if len(replay_buffer) > batch_size:
            # Sample a random batch of memories
            states_b, actions_b, rewards_b, next_states_b, dones_b = replay_buffer.sample(batch_size)
            
            # Convert to PyTorch Tensors and move to Device (and normalize pixels)
            states_tensor = torch.FloatTensor(states_b).to(device) / 255.0
            next_states_tensor = torch.FloatTensor(next_states_b).to(device) / 255.0
            actions_tensor = torch.LongTensor(actions_b).unsqueeze(1).to(device)
            rewards_tensor = torch.FloatTensor(rewards_b).unsqueeze(1).to(device)
            dones_tensor = torch.BoolTensor(dones_b).unsqueeze(1).to(device)
            
            # Current Q(s_t, a_t) from online network.
            # gather() indexes predicted Q-values at executed actions.
            current_q_values = q_network(states_tensor).gather(1, actions_tensor)
            
            # Target Q-Values: What does the TARGET network think the best next action is worth?
            with torch.no_grad():
                max_next_q_values = target_network(next_states_tensor).max(dim=1, keepdim=True)[0]
                
                # 1-step Bellman bootstrap target:
                #   y_t = r_t + gamma * max_a' Q_target(s_{t+1}, a')
                # If done, there is no next Q value!
                target_q_values = rewards_tensor + (gamma * max_next_q_values * (~dones_tensor))
                
            # Huber loss is less sensitive to large TD errors than MSE and is the
            # common default in Atari DQN implementations.
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
        if frame_idx % 2000 == 0:
            avg_reward = np.mean(episode_rewards[-20:]) if episode_rewards else 0.0
            print(f"Frame: {frame_idx}/{max_frames} | Epsilon: {epsilon:.2f} | Avg Reward (Last 20): {avg_reward:.2f}")

    env.close()
    return q_network, episode_rewards


def plot_rewards(rewards, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.6)
    plt.title('DQN Training on Breakout-v5')
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
    filename = os.path.join(save_dir, 'breakout_dqn_agent.gif')
    print(f"Evaluating agent and saving video to {filename}...")
    env = make_env(render_mode='rgb_array')
    state, _ = env.reset()
    frames = []
    
    done = False
    total_reward = 0
    
    with torch.no_grad():
        # Play one episode
        
        # Breakout requires a "FIRE" action (Action 1) to launch the ball at the start of a life.
        # If the untrained agent never picks FIRE, the game stalls forever. We force it here.
        state, _, _, _, _ = env.step(1)
        frames.append(env.render())
        
        while not done:
            # We must use env.unwrapped.render() in some gymnasium versions to bypass wrappers for video,
            # but env.render() works fine if render_mode was passed.
            frames.append(env.render()) 
            
            state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0).to(device) / 255.0
            q_values = model(state_tensor)
            
            # Deterministic: Always pick the max Q-value action
            action = q_values.argmax().item()
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            if len(frames) > 500: # Breakout can get stuck if untrained, cap it to avoid huge GIFs
                break
                
    env.close()
    print(f"Evaluation finished. Total Reward: {total_reward}")
    
    imageio.mimsave(filename, frames, fps=30)
    print("Saved successfully!")

if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    save_dir = os.path.join(base_dir, "results", "dqn", f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving all results to: {save_dir}")

    model, rewards = train()
    
    model_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    plot_rewards(rewards, save_dir)
    evaluate_and_record(model, save_dir)
