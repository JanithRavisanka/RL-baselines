import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
import datetime
import imageio
import matplotlib.pyplot as plt

# Set up Hardware Acceleration
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

class DynamicsModel(nn.Module):
    """
    A Neural Network that learns the underlying physics of the environment.
    Instead of predicting the next state exactly, it predicts the *change* (delta) in the state.
    This makes the learning objective much easier.
    """
    def __init__(self, state_dim, action_dim):
        super(DynamicsModel, self).__init__()
        # Input is State + Action
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        # Output is Delta State
        self.fc3 = nn.Linear(128, state_dim)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        delta_state = self.fc3(x)
        return delta_state

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, next_state = map(np.stack, zip(*batch))
        return state, action, next_state

    def __len__(self):
        return len(self.buffer)

def pendulum_reward_fn(state, action):
    """
    Vectorized reward function for Pendulum-v1.
    State: [cos(theta), sin(theta), theta_dot]
    Reward: -(theta^2 + 0.1 * theta_dot^2 + 0.001 * action^2)
    """
    cos_th = state[:, 0]
    sin_th = state[:, 1]
    th_dot = state[:, 2]
    
    # Calculate angle from cos and sin
    th = torch.atan2(sin_th, cos_th)
    
    # Normalize angle between -pi and pi (atan2 already does this)
    
    cost = th**2 + 0.1 * (th_dot**2) + 0.001 * (action.squeeze(-1)**2)
    return -cost

class RandomShootingPlanner:
    """
    Uses the Dynamics Model to simulate thousands of imaginary trajectories,
    evaluates them, and picks the best immediate action.
    """
    def __init__(self, dynamics_model, action_dim, num_sequences=1000, horizon=15):
        self.model = dynamics_model
        self.action_dim = action_dim
        self.K = num_sequences # Number of parallel imaginary trajectories
        self.H = horizon       # How many steps into the future to simulate

    def get_action(self, current_state):
        self.model.eval()
        with torch.no_grad():
            # 1. Create K identical starting states
            # state shape: (K, state_dim)
            state_batch = torch.FloatTensor(current_state).unsqueeze(0).repeat(self.K, 1).to(device)
            
            # 2. Generate completely random action sequences
            # Pendulum actions are between -2.0 and 2.0
            # action_sequences shape: (H, K, action_dim)
            action_sequences = torch.FloatTensor(self.H, self.K, self.action_dim).uniform_(-2.0, 2.0).to(device)
            
            # Array to store the total cumulative reward for each of the K sequences
            cumulative_rewards = torch.zeros(self.K).to(device)
            
            # 3. Simulate the future!
            for t in range(self.H):
                actions_t = action_sequences[t]
                
                # Predict delta state using our learned physics model
                delta_s = self.model(state_batch, actions_t)
                
                # Predict next state
                next_state_batch = state_batch + delta_s
                
                # Normalize cos and sin to ensure it stays a valid angle (cos^2 + sin^2 = 1)
                norm = torch.sqrt(next_state_batch[:, 0]**2 + next_state_batch[:, 1]**2 + 1e-8)
                next_state_batch[:, 0] /= norm
                next_state_batch[:, 1] /= norm
                
                # Evaluate how good this state is
                rewards = pendulum_reward_fn(next_state_batch, actions_t)
                cumulative_rewards += rewards
                
                state_batch = next_state_batch
                
            # 4. Find the sequence that yielded the highest total reward
            best_sequence_idx = torch.argmax(cumulative_rewards).item()
            
            # 5. We only return the *first* action of the best sequence. 
            # In MPC, we replan from scratch at the next real step!
            best_first_action = action_sequences[0, best_sequence_idx].cpu().numpy()
            
        return best_first_action

def train_dynamics_model(model, optimizer, buffer, batch_size=256, epochs=10):
    if len(buffer) < batch_size:
        return 0.0
        
    model.train()
    total_loss = 0
    
    for _ in range(epochs):
        # Sample real-world transitions
        states, actions, next_states = buffer.sample(batch_size)
        
        states_t = torch.FloatTensor(states).to(device)
        actions_t = torch.FloatTensor(actions).to(device)
        next_states_t = torch.FloatTensor(next_states).to(device)
        
        # Calculate the true delta we want to predict
        true_deltas = next_states_t - states_t
        
        # Predict delta
        pred_deltas = model(states_t, actions_t)
        
        # MSE Loss
        loss = F.mse_loss(pred_deltas, true_deltas)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / epochs

def train():
    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    dynamics_model = DynamicsModel(state_dim, action_dim).to(device)
    optimizer = optim.Adam(dynamics_model.parameters(), lr=1e-3)
    
    buffer = ReplayBuffer(capacity=100000)
    
    # 1. Seed Dataset Collection
    # We must show the model *some* real physics before it can plan!
    print("Collecting seed dataset with random actions...")
    state, _ = env.reset()
    for _ in range(2000):
        action = env.action_space.sample()
        next_state, _, terminated, truncated, _ = env.step(action)
        buffer.push(state, action, next_state)
        
        if terminated or truncated:
            state, _ = env.reset()
        else:
            state = next_state
            
    # Train the initial physics model heavily
    print("Training initial Dynamics Model...")
    train_dynamics_model(dynamics_model, optimizer, buffer, batch_size=256, epochs=100)
    
    planner = RandomShootingPlanner(dynamics_model, action_dim, num_sequences=1000, horizon=15)
    
    num_episodes = 25 # MPC evaluates paths online, so it requires very few episodes compared to model-free!
    episode_rewards = []
    
    print("Starting MPC Control loop...")
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Plan and execute action
            action = planner.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Record the new transition (Data Aggregation)
            buffer.push(state, action, next_state)
            
            state = next_state
            total_reward += reward
            
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{num_episodes} | Total Reward: {total_reward:.2f}")
        
        # After every episode, retrain the dynamics model with the newly collected data
        # This fixes any inaccuracies the model had!
        loss = train_dynamics_model(dynamics_model, optimizer, buffer, batch_size=256, epochs=10)
        print(f"   -> Dynamics Model retrained. Avg MSE Loss: {loss:.5f}")

    env.close()
    return dynamics_model, planner, episode_rewards

def plot_rewards(rewards, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, marker='o')
    plt.title('Learned Dynamics MPC on Pendulum-v1')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid()
    save_path = os.path.join(save_dir, 'training_curve.png')
    plt.savefig(save_path)
    print(f"Training curve saved as '{save_path}'")

def evaluate_and_record(planner, save_dir):
    filename = os.path.join(save_dir, 'pendulum_mpc_agent.gif')
    print(f"Evaluating agent and saving video to {filename}...")
    env = gym.make('Pendulum-v1', render_mode='rgb_array')
    state, _ = env.reset()
    frames = []
    
    done = False
    total_reward = 0
    
    # We increase sequences during evaluation for even better planning!
    planner.K = 2000 
    
    while not done:
        frames.append(env.render())
        action = planner.get_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
    env.close()
    print(f"Evaluation finished. Total Reward: {total_reward:.2f}")
    
    imageio.mimsave(filename, frames, fps=30)
    print("Saved successfully!")

if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    save_dir = os.path.join(base_dir, "results", "mpc", f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving all results to: {save_dir}")

    model, planner, rewards = train()
    
    model_path = os.path.join(save_dir, "dynamics_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Dynamics Model saved to {model_path}")
    
    plot_rewards(rewards, save_dir)
    evaluate_and_record(planner, save_dir)
