import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import datetime
# Set up Hardware Acceleration
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")
class ActorCriticContinuous(nn.Module):
    """
    Continuous Actor-Critic Model for Pendulum-v1.
    Instead of outputting discrete probabilities, the Actor outputs the Mean (mu) 
    and Standard Deviation (sigma) of a Gaussian distribution to sample continuous force.
    """
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorCriticContinuous, self).__init__()
        self.max_action = max_action
        
        # Shared feature extractor
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Actor head: Outputs Mean (mu) and log standard deviation (for stability)
        self.mu_head = nn.Linear(128, action_dim)
        # Using a Parameter for log_std is a common trick to learn exploration independent of state
        # Or we can output it from the network. We'll output it from the network here.
        self.sigma_head = nn.Linear(128, action_dim)
        
        # Critic head: Outputs State Value
        self.critic = nn.Linear(128, 1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # ACTOR:
        # Mu is scaled to the environment's max action (e.g., [-2.0, 2.0] for Pendulum)
        mu = self.max_action * torch.tanh(self.mu_head(x))
        
        # Sigma must be positive, so we use Softplus (like ReLU but smooth)
        sigma = F.softplus(self.sigma_head(x)) + 1e-5 # add small epsilon to prevent 0
        
        # CRITIC:
        state_value = self.critic(x)
        
        return mu, sigma, state_value

def train():
    """
    Episodic actor-critic for continuous control.

    Policy parameterization:
    - Actor outputs (mu, sigma) for a Gaussian policy over actions.
    - Action is sampled during training for exploration, then clipped to env bounds.
    - Critic estimates V(s) and serves as baseline in advantage computation.
    """
    # Pendulum-v1 is a continuous environment
    env = gym.make('Pendulum-v1')
    
    state_dim = env.observation_space.shape[0]  # 3 (cos(theta), sin(theta), dot(theta))
    action_dim = env.action_space.shape[0]      # 1 (Joint effort)
    max_action = float(env.action_space.high[0]) # 2.0
    
    model = ActorCriticContinuous(state_dim, action_dim, max_action).to(device)
    # Using a slightly smaller learning rate as continuous spaces are more sensitive
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    gamma = 0.6
    num_episodes = 100000 # Pendulum might take more episodes to learn perfectly
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        
        log_probs = []
        values = []
        rewards = []
        
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # Get Gaussian parameters and state value
            mu, sigma, state_value = model(state_tensor)
            
            # Create a Normal distribution from mu and sigma.
            # log_prob under this distribution is the score-function term used for
            # policy-gradient updates.
            dist = Normal(mu, sigma)
            
            # Sample action for stochastic exploration.
            # In continuous spaces this is the main exploration mechanism.
            action = dist.sample()
            
            # Clip to valid action range expected by physics simulator.
            action_clipped = torch.clamp(action, -max_action, max_action)
            
            # Take step in environment
            next_state, reward, terminated, truncated, _ = env.step(action_clipped.cpu().detach().numpy()[0])
            done = terminated or truncated
            
            # Store memory
            log_probs.append(dist.log_prob(action).sum())
            values.append(state_value)
            rewards.append(reward)
            
            state = next_state
            
        episode_rewards.append(sum(rewards))
        
        # --- LEARNING ---
        # We compute Monte-Carlo returns, then use:
        # - actor loss:   -log pi(a|s) * advantage
        # - critic loss:  regression to return target
        returns = []
        R = 0
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        
        actor_loss = []
        critic_loss = []
        
        for log_prob, value, R in zip(log_probs, values, returns):
            # Advantage estimate A_t ~= G_t - V(s_t)
            advantage = R - value.item()
            
            actor_loss.append(-log_prob * advantage)
            critic_loss.append(F.smooth_l1_loss(value, torch.tensor([[R]], dtype=torch.float32).to(device)))
            
        optimizer.zero_grad()
        loss = torch.stack(actor_loss).sum() + torch.stack(critic_loss).sum()
        loss.backward()
        
        # Gradient clipping is especially useful here because log-prob terms can spike
        # when sigma becomes very small and create unstable updates.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode + 1}/{num_episodes} | Average Reward (last 50): {avg_reward:.2f}")
            
            # Pendulum max reward is 0. Anything above -250 is quite good.
            if len(episode_rewards) >= 100 and np.mean(episode_rewards[-100:]) > -200:
                print(f"Solved (Satisfactory performance) at episode {episode + 1}!")
                break
                
    env.close()
    return model, episode_rewards

def plot_rewards(rewards, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.6)
    plt.title('Continuous Actor-Critic Training on Pendulum-v1')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    window = 50
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, color='red', label='Moving Average (50 episodes)')
        plt.legend()
        
    plt.grid()
    save_path = os.path.join(save_dir, 'training_curve_continuous.png')
    plt.savefig(save_path)
    print(f"Training curve saved as '{save_path}'")

def evaluate_and_record(model, save_dir):
    filename = os.path.join(save_dir, 'pendulum_agent.gif')
    env = gym.make('Pendulum-v1', render_mode='rgb_array')
    state, _ = env.reset()
    frames = []
    
    done = False
    total_reward = 0
    max_action = float(env.action_space.high[0])
    
    with torch.no_grad():
        while not done:
            frames.append(env.render())
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            mu, _, _ = model(state_tensor)
            
            # During evaluation, we pick the exact Mean (deterministic) instead of sampling from the Bell Curve
            action_clipped = torch.clamp(mu, -max_action, max_action)
            
            state, reward, terminated, truncated, _ = env.step(action_clipped.cpu().numpy()[0])
            done = terminated or truncated
            total_reward += reward
            
    env.close()
    print(f"Evaluation finished. Total Reward: {total_reward:.2f}")
    
    print(f"Saving video as {filename}...")
    imageio.mimsave(filename, frames, fps=30)
    print("Saved successfully!")

if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    save_dir = os.path.join(base_dir, "results", "actor_critic_continuous", f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving all results to: {save_dir}")

    print("Starting Continuous Actor-Critic training on Pendulum-v1...")
    model, rewards = train()
    
    model_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    print("Plotting learning curve...")
    plot_rewards(rewards, save_dir)
    
    print("Evaluating trained agent and recording video...")
    evaluate_and_record(model, save_dir)
