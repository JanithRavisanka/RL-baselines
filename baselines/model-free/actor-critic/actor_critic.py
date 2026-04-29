import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import datetime
class ActorCritic(nn.Module):
    """
    The Actor-Critic model combines two different networks (often sharing a base representation):
    1. The Actor: Decides *what to do* (outputs action probabilities based on the current state).
    2. The Critic: Evaluates *how good* the current state is (outputs an estimated value score).
    """
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Shared feature extractor: Both actor and critic use this layer to understand the environment state
        self.fc1 = nn.Linear(state_dim, 128)
        
        # Actor head (Policy): Maps the shared features to action probabilities (e.g., Left vs Right)
        self.actor = nn.Linear(128, action_dim)
        
        # Critic head (Value): Maps the shared features to a single number representing the expected total future reward
        self.critic = nn.Linear(128, 1)
        
    def forward(self, state):
        # Pass state through the shared layer and apply ReLU activation function
        x = F.relu(self.fc1(state))
        
        # Action probabilities (using Softmax so the probabilities sum to 1.0)
        action_probs = F.softmax(self.actor(x), dim=-1)
        
        # State value (no activation needed, it can be any real number)
        state_value = self.critic(x)
        
        return action_probs, state_value

def train():
    """
    Monte-Carlo episodic Actor-Critic training loop.

    Why this structure:
    - We collect one full episode before updating so returns are unbiased samples.
    - The critic learns a baseline V(s) to reduce policy-gradient variance.
    - The actor is updated with advantage-weighted log-prob terms.
    """
    # Initialize the CartPole environment
    env = gym.make('CartPole-v1')
    
    # Get dimensions for our neural network from the environment
    state_dim = env.observation_space.shape[0]  # 4 variables: position, velocity, pole angle, pole angular velocity
    action_dim = env.action_space.n             # 2 actions: push cart left or push cart right
    
    # Create the model and the Adam optimizer
    model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    gamma = 0.99  # Discount factor: determines how much we care about future rewards compared to immediate ones
    num_episodes = 1000  # Maximum number of games to play during training
    
    episode_rewards = [] # Keep track of scores to see if the agent is learning over time
    
    for episode in range(num_episodes):
        state, _ = env.reset() # Start a new game
        
        # Lists to store memory of this specific episode for learning later
        log_probs = [] # Logarithm of the probability of the actions we ended up taking
        values = []    # The critic's predicted value for each state we visited
        rewards = []   # The actual rewards we received at each step
        
        done = False
        
        # --- PLAYING ONE EPISODE ---
        while not done:
            # Convert numpy state array to a PyTorch tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Ask the model for action probabilities and the estimated value of this state
            action_probs, state_value = model(state_tensor)
            
            # Sample an action based on the probabilities (this allows the agent to explore)
            m = Categorical(action_probs)
            action = m.sample()
            
            # Take the chosen action in the environment and observe what happens
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            # Store the log probability, critic's value, and reward for training later
            log_probs.append(m.log_prob(action))
            values.append(state_value)
            rewards.append(reward)
            
            state = next_state
            
        # Record the total score for this episode
        episode_rewards.append(sum(rewards))
        
        # --- LEARNING FROM THE EPISODE ---
        
        # 1. Calculate actual Returns (Discounted cumulative rewards)
        # For each step, we calculate how much total reward we got from that point until the end of the game
        returns = []
        R = 0
        # We iterate backwards through the rewards to easily calculate the cumulative discounted sum
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns)
        
        # Normalize returns (subtract mean, divide by std).
        # This does NOT change trajectory ordering; it rescales gradient magnitudes so
        # optimization remains stable across early/late training phases.
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        
        actor_loss = []
        critic_loss = []
        
        # 2. Calculate Losses for both the Actor and the Critic
        for log_prob, value, R in zip(log_probs, values, returns):
            # Advantage estimate:
            #   A_t ~= G_t - V(s_t)
            # where G_t is Monte-Carlo return and V(s_t) is critic baseline.
            # If Advantage > 0: The action resulted in a better outcome than the critic expected.
            # If Advantage < 0: The action resulted in a worse outcome than the critic expected.
            advantage = R - value.item()
            
            # Actor loss: We want to increase the probability of actions that had a positive advantage
            # (The negative sign is because PyTorch minimizes loss, but we want to maximize expected reward)
            actor_loss.append(-log_prob * advantage)
            
            # Critic loss: How far off was the critic's prediction from the actual return?
            # We use Smooth L1 Loss (Huber Loss) which is less sensitive to outliers than Mean Squared Error
            critic_loss.append(F.smooth_l1_loss(value, torch.tensor([[R]])))
            
        # 3. Backpropagation (joint actor+critic update)
        # We optimize a single scalar objective that is the sum of:
        # - policy gradient surrogate (actor),
        # - value regression loss (critic).
        # Shared trunk gradients therefore reflect both objectives.
        optimizer.zero_grad() # Clear old gradients
        # Combine actor and critic losses into a single total loss
        loss = torch.stack(actor_loss).sum() + torch.stack(critic_loss).sum()
        loss.backward()       # Calculate new gradients based on the loss
        optimizer.step()      # Apply the updates to the neural network weights
        
        # --- LOGGING & EARLY STOPPING ---
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}/{num_episodes} | Average Reward (last 50): {np.mean(episode_rewards[-50:]):.2f}")
            
        # CartPole is considered "solved" if the average score over the last 100 episodes is >= 475
        if len(episode_rewards) >= 100 and np.mean(episode_rewards[-100:]) >= 475:
            print(f"Solved at episode {episode + 1}!")
            break
            
    env.close()
    return model, episode_rewards

def plot_rewards(rewards, save_dir):
    """Plots the training curve to visualize learning progress."""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Actor-Critic Training on CartPole-v1')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Plot moving average for a smoother trend line
    window = 50
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, color='red', label='Moving Average (50 episodes)')
        plt.legend()
        
    plt.grid()
    save_path = os.path.join(save_dir, 'training_curve.png')
    plt.savefig(save_path)
    print(f"Training curve saved as '{save_path}'")

def evaluate_and_record(model, save_dir):
    filename = os.path.join(save_dir, 'cartpole_agent.gif')
    """Runs the trained agent and saves a GIF of its performance to see it in action."""
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    state, _ = env.reset()
    frames = []
    
    done = False
    total_reward = 0
    
    # torch.no_grad() disables gradient calculation to save memory and speed up evaluation
    with torch.no_grad():
        while not done:
            frames.append(env.render()) # Capture the screen for the GIF
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs, _ = model(state_tensor)
            
            # During evaluation, we pick the best action deterministically (highest probability) instead of sampling randomly
            action = torch.argmax(action_probs).item()
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
    env.close()
    print(f"Evaluation finished. Total Reward: {total_reward}")
    
    # Save the captured frames as a GIF
    print(f"Saving video as {filename}...")
    imageio.mimsave(filename, frames, fps=30)
    print("Saved successfully!")

if __name__ == '__main__':
    # Create save directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    save_dir = os.path.join(base_dir, "results", "actor_critic", f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving all results to: {save_dir}")

    print("Starting Actor-Critic training...")
    model, rewards = train()
    
    model_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    print("Plotting learning curve...")
    plot_rewards(rewards, save_dir)
    
    print("Evaluating trained agent and recording video...")
    evaluate_and_record(model, save_dir)
