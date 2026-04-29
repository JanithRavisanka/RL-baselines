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


class TransitionNormalizer:
    def __init__(self, state_dim, action_dim, eps=1e-6):
        self.eps = eps
        self.state_mean = torch.zeros(state_dim, device=device)
        self.state_std = torch.ones(state_dim, device=device)
        self.action_mean = torch.zeros(action_dim, device=device)
        self.action_std = torch.ones(action_dim, device=device)
        self.delta_mean = torch.zeros(state_dim, device=device)
        self.delta_std = torch.ones(state_dim, device=device)

    def update(self, buffer):
        # Fit running dataset statistics from replay.
        # Normalized inputs/targets make dynamics optimization better conditioned.
        states, actions, next_states = map(np.stack, zip(*buffer.buffer))
        states_t = torch.FloatTensor(states).to(device)
        actions_t = torch.FloatTensor(actions).to(device)
        deltas_t = torch.FloatTensor(next_states - states).to(device)
        self.state_mean = states_t.mean(0)
        self.state_std = states_t.std(0).clamp_min(self.eps)
        self.action_mean = actions_t.mean(0)
        self.action_std = actions_t.std(0).clamp_min(self.eps)
        self.delta_mean = deltas_t.mean(0)
        self.delta_std = deltas_t.std(0).clamp_min(self.eps)

    def inputs(self, states, actions):
        return (states - self.state_mean) / self.state_std, (actions - self.action_mean) / self.action_std

    def deltas(self, deltas):
        return (deltas - self.delta_mean) / self.delta_std

    def denorm_delta(self, deltas):
        return deltas * self.delta_std + self.delta_mean

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

class CEMPlanner:
    """
    Cross-entropy method MPC with an ensemble dynamics model.
    """
    def __init__(self, dynamics_models, normalizer, action_dim, action_low, action_high, num_sequences=512, horizon=20, elite_frac=0.1, iterations=4, gamma=0.99):
        self.models = dynamics_models
        self.normalizer = normalizer
        self.action_dim = action_dim
        self.action_low = torch.FloatTensor(action_low).to(device)
        self.action_high = torch.FloatTensor(action_high).to(device)
        self.K = num_sequences # Number of parallel imaginary trajectories
        self.H = horizon       # How many steps into the future to simulate
        self.elite_frac = elite_frac
        self.iterations = iterations
        self.gamma = gamma

    def predict_delta(self, states, actions):
        # Ensemble mean prediction reduces single-model exploitation artifacts.
        norm_states, norm_actions = self.normalizer.inputs(states, actions)
        preds = []
        for model in self.models:
            preds.append(self.normalizer.denorm_delta(model(norm_states, norm_actions)))
        return torch.stack(preds, dim=0).mean(0)

    def rollout_return(self, current_state, action_sequences):
        # Evaluate every candidate open-loop sequence under learned dynamics.
        # We apply geometric discount to emphasize near-term controllability.
        K = action_sequences.shape[1]
        state_batch = torch.FloatTensor(current_state).unsqueeze(0).repeat(K, 1).to(device)
        cumulative_rewards = torch.zeros(K).to(device)
        discount = 1.0
        for t in range(self.H):
            actions_t = action_sequences[t]
            next_state_batch = state_batch + self.predict_delta(state_batch, actions_t)
            norm = torch.sqrt(next_state_batch[:, 0]**2 + next_state_batch[:, 1]**2 + 1e-8)
            next_state_batch[:, 0] /= norm
            next_state_batch[:, 1] /= norm
            cumulative_rewards += discount * pendulum_reward_fn(next_state_batch, actions_t)
            state_batch = next_state_batch
            discount *= self.gamma
        return cumulative_rewards

    def get_action(self, current_state):
        for model in self.models:
            model.eval()
        with torch.no_grad():
            mean = torch.zeros(self.H, self.action_dim, device=device)
            std = torch.ones_like(mean) * (self.action_high - self.action_low) / 2.0
            elite_count = max(1, int(self.K * self.elite_frac))
            best_sequence = None
            # CEM loop:
            # 1) sample candidates from current Gaussian over action sequences,
            # 2) keep elite fraction,
            # 3) refit Gaussian to elites.
            for _ in range(self.iterations):
                noise = torch.randn(self.H, self.K, self.action_dim, device=device)
                action_sequences = mean[:, None, :] + std[:, None, :] * noise
                action_sequences = torch.max(torch.min(action_sequences, self.action_high), self.action_low)
                returns = self.rollout_return(current_state, action_sequences)
                elite_idx = torch.topk(returns, elite_count).indices
                elites = action_sequences[:, elite_idx]
                mean = elites.mean(dim=1)
                std = elites.std(dim=1).clamp_min(0.05)
                best_sequence = action_sequences[:, torch.argmax(returns)]

            best_first_action = best_sequence[0].cpu().numpy()
        return best_first_action

def train_dynamics_model(models, optimizers, normalizer, buffer, batch_size=256, epochs=10):
    if len(buffer) < batch_size:
        return 0.0

    normalizer.update(buffer)
    for model in models:
        model.train()
    total_loss = 0
    
    for _ in range(epochs):
        states, actions, next_states = buffer.sample(batch_size)
        states_t = torch.FloatTensor(states).to(device)
        actions_t = torch.FloatTensor(actions).to(device)
        next_states_t = torch.FloatTensor(next_states).to(device)
        true_deltas = next_states_t - states_t

        # Train in normalized delta-space:
        # model predicts standardized (s_{t+1} - s_t), then denormalization is only
        # needed for rollout/planning time.
        norm_states, norm_actions = normalizer.inputs(states_t, actions_t)
        target_deltas = normalizer.deltas(true_deltas)
        loss = 0.0
        for model, optimizer in zip(models, optimizers):
            pred_deltas = model(norm_states, norm_actions)
            model_loss = F.mse_loss(pred_deltas, target_deltas)
            optimizer.zero_grad()
            model_loss.backward()
            optimizer.step()
            loss += model_loss.item()
        loss /= len(models)
        total_loss += loss

    return total_loss / epochs


def predict_delta_ensemble(models, normalizer, states, actions):
    norm_states, norm_actions = normalizer.inputs(states, actions)
    preds = [normalizer.denorm_delta(model(norm_states, norm_actions)) for model in models]
    return torch.stack(preds, dim=0).mean(0)


def validate_multistep(models, normalizer, buffer, horizon=5, samples=256):
    # Multi-step validation catches compounding rollout error that one-step MSE misses.
    if len(buffer) < horizon + 1:
        return 0.0
    errors = []
    max_start = max(0, len(buffer.buffer) - horizon)
    for _ in range(min(samples, max_start)):
        start = random.randint(0, max_start - 1)
        state = torch.FloatTensor(buffer.buffer[start][0]).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = state.clone()
            for t in range(horizon):
                action = torch.FloatTensor(buffer.buffer[start + t][1]).unsqueeze(0).to(device)
                pred = pred + predict_delta_ensemble(models, normalizer, pred, action)
            actual = torch.FloatTensor(buffer.buffer[start + horizon - 1][2]).unsqueeze(0).to(device)
            errors.append(F.mse_loss(pred, actual).item())
    return float(np.mean(errors)) if errors else 0.0

def train():
    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    dynamics_models = [DynamicsModel(state_dim, action_dim).to(device) for _ in range(5)]
    optimizers = [optim.Adam(model.parameters(), lr=1e-3) for model in dynamics_models]
    normalizer = TransitionNormalizer(state_dim, action_dim)
    
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
    train_dynamics_model(dynamics_models, optimizers, normalizer, buffer, batch_size=256, epochs=100)
    planner = CEMPlanner(
        dynamics_models,
        normalizer,
        action_dim,
        env.action_space.low,
        env.action_space.high,
        num_sequences=512,
        horizon=20,
    )
    
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
        loss = train_dynamics_model(dynamics_models, optimizers, normalizer, buffer, batch_size=256, epochs=10)
        rollout_mse = validate_multistep(dynamics_models, normalizer, buffer, horizon=5)
        print(f"   -> Dynamics Model retrained. One-step MSE: {loss:.5f} | 5-step rollout MSE: {rollout_mse:.5f}")

    env.close()
    return dynamics_models, planner, episode_rewards

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
    planner.K = 1024
    
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

    models, planner, rewards = train()
    
    model_path = os.path.join(save_dir, "dynamics_ensemble.pth")
    torch.save([model.state_dict() for model in models], model_path)
    print(f"Dynamics ensemble saved to {model_path}")
    
    plot_rewards(rewards, save_dir)
    evaluate_and_record(planner, save_dir)
