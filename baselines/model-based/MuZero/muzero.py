import os
import datetime
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt
import imageio

# Hardware Acceleration
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ==============================================================================
# 1. Neural Networks (Representation, Dynamics, Prediction)
# ==============================================================================

class MuZeroNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # 1. Representation Function: h(o) -> s_0
        self.representation = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 2. Dynamics Function: g(s_{k-1}, a_k) -> r_k, s_k
        self.dynamics_state = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.dynamics_reward = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Predicts scalar reward
        )
        
        # 3. Prediction Function: f(s_k) -> p_k, v_k
        self.prediction_policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim) # Outputs logits
        )
        self.prediction_value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Predicts scalar value
        )

    def encode_action(self, action):
        # Convert integer action to one-hot tensor
        action_one_hot = torch.zeros((action.shape[0], self.action_dim), device=device)
        action_one_hot.scatter_(1, action.unsqueeze(1), 1.0)
        return action_one_hot

    def normalize_hidden(self, hidden_state):
        # Min-max normalization or tanh to keep Dynamics stable during unrolling
        return torch.tanh(hidden_state)

    def initial_inference(self, obs):
        """h(obs) and f(s_0)"""
        hidden_state = self.representation(obs)
        hidden_state = self.normalize_hidden(hidden_state)
        
        policy_logits = self.prediction_policy(hidden_state)
        value = self.prediction_value(hidden_state)
        return hidden_state, policy_logits, value

    def recurrent_inference(self, hidden_state, action):
        """g(s_{k-1}, a_k) and f(s_k)"""
        action_one_hot = self.encode_action(action)
        state_action = torch.cat([hidden_state, action_one_hot], dim=1)
        
        next_hidden_state = self.dynamics_state(state_action)
        next_hidden_state = self.normalize_hidden(next_hidden_state)
        reward = self.dynamics_reward(state_action)
        
        policy_logits = self.prediction_policy(next_hidden_state)
        value = self.prediction_value(next_hidden_state)
        return next_hidden_state, reward, policy_logits, value

# ==============================================================================
# 2. Monte Carlo Tree Search (MCTS)
# ==============================================================================

class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.reward = 0
        self.children = {} # action -> Node
        self.hidden_state = None
        
    def expanded(self):
        return len(self.children) > 0
        
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

def run_mctx(config, network, obs):
    """Runs MCTS to build a search tree and returns the improved policy."""
    root = Node(0)
    
    # 1. Initial Inference
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        hidden_state, policy_logits, value = network.initial_inference(obs_tensor)
        
    root.hidden_state = hidden_state
    
    # Expand root
    policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
    for a in range(config['action_dim']):
        root.children[a] = Node(policy_probs[a])
        
    # Add Dirichlet noise to root for exploration
    noise = np.random.dirichlet([config['dirichlet_alpha']] * config['action_dim'])
    for a in root.children:
        root.children[a].prior = root.children[a].prior * (1 - config['dirichlet_eps']) + noise[a] * config['dirichlet_eps']
        
    # 2. Run Simulations
    for _ in range(config['num_simulations']):
        node = root
        search_path = [node]
        history = [] # actions taken
        
        # Traverse
        while node.expanded():
            # UCB selection
            best_score = -float('inf')
            best_action = -1
            
            for a, child in node.children.items():
                # UCB Score = Q(s, a) + c * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
                q_value = child.value()
                exploration_term = config['pb_c_init'] * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
                score = q_value + exploration_term
                
                if score > best_score:
                    best_score = score
                    best_action = a
            
            action = best_action
            history.append(action)
            node = node.children[action]
            search_path.append(node)
            
        # Expand and Evaluate Leaf
        parent = search_path[-2]
        action = history[-1]
        action_tensor = torch.tensor([action], dtype=torch.long, device=device)
        
        with torch.no_grad():
            next_hidden, reward, policy_logits, value = network.recurrent_inference(parent.hidden_state, action_tensor)
            
        node.hidden_state = next_hidden
        node.reward = reward.item()
        v = value.item()
        
        policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
        for a in range(config['action_dim']):
            node.children[a] = Node(policy_probs[a])
            
        # Backpropagate
        for node in reversed(search_path):
            node.value_sum += v
            node.visit_count += 1
            # Discount value for the parent
            v = node.reward + config['discount'] * v
            
    # 3. Calculate Target Policy from Visit Counts
    visit_counts = np.array([root.children[a].visit_count for a in range(config['action_dim'])])
    policy = visit_counts / np.sum(visit_counts)
    return policy, root.value()

# ==============================================================================
# 3. Game History & Replay Buffer
# ==============================================================================

class GameHistory:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.target_policies = []
        self.target_values = []
        
    def store_search_statistics(self, obs, action, reward, policy, value):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.target_policies.append(policy)
        self.target_values.append(value)

class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = []
        self.capacity = capacity
        
    def save_game(self, game):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(game)
        
    def sample(self, batch_size, unroll_steps):
        # We sample an observation, and then sequences of length `unroll_steps`
        obs_batch, action_batch, reward_batch, policy_batch, value_batch = [], [], [], [], []
        
        for _ in range(batch_size):
            game = self.buffer[np.random.randint(len(self.buffer))]
            start = np.random.randint(len(game.observations))
            obs_batch.append(game.observations[start])
            
            actions = []
            rewards = []
            policies = []
            values = []
            
            for i in range(unroll_steps):
                if start + i < len(game.observations):
                    actions.append(game.actions[start + i])
                    rewards.append(game.rewards[start + i])
                    policies.append(game.target_policies[start + i])
                    values.append(game.target_values[start + i])
                else:
                    # Absorbing state (game ended)
                    actions.append(np.random.randint(2))
                    rewards.append(0.0)
                    policies.append(np.ones(2)/2)
                    values.append(0.0)
                    
            action_batch.append(actions)
            reward_batch.append(rewards)
            policy_batch.append(policies)
            value_batch.append(values)
            
        return (
            torch.FloatTensor(np.array(obs_batch)).to(device),
            torch.LongTensor(np.array(action_batch)).to(device),
            torch.FloatTensor(np.array(reward_batch)).to(device),
            torch.FloatTensor(np.array(policy_batch)).to(device),
            torch.FloatTensor(np.array(value_batch)).to(device)
        )

# ==============================================================================
# 4. Training Loop (BPTT)
# ==============================================================================

def train():
    env = gym.make("CartPole-v1")
    config = {
        'obs_dim': env.observation_space.shape[0],
        'action_dim': env.action_space.n,
        'num_simulations': 25, # MCTS runs per real environment step
        'discount': 0.99,
        'pb_c_init': 1.25, # UCB exploration constant
        'dirichlet_alpha': 0.25,
        'dirichlet_eps': 0.25,
        'unroll_steps': 5, # Unroll K steps during BPTT training
        'batch_size': 64,
        'num_games': 200,
        'lr': 0.002
    }
    
    network = MuZeroNetwork(config['obs_dim'], config['action_dim']).to(device)
    optimizer = optim.Adam(network.parameters(), lr=config['lr'], weight_decay=1e-4)
    buffer = ReplayBuffer(capacity=500)
    
    episode_rewards = []
    
    for episode in range(config['num_games']):
        game = GameHistory()
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # 1. Plan using MCTS
            policy, root_value = run_mctx(config, network, obs)
            
            # 2. Select Action (Explore based on MCTS visit counts)
            action = np.random.choice(config['action_dim'], p=policy)
            
            # 3. Take real environment step
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # Termination penalty for CartPole
            if terminated: reward = -1.0
            
            game.store_search_statistics(obs, action, reward, policy, root_value)
            obs = next_obs
            
        buffer.save_game(game)
        episode_rewards.append(total_reward)
        
        # --- Unrolled BPTT Training ---
        if len(buffer.buffer) > 10:
            for _ in range(2): # Optimize multiple times per game added
                obs_batch, act_batch, rew_batch, pol_batch, val_batch = buffer.sample(config['batch_size'], config['unroll_steps'])
                
                loss = 0
                
                # Initial Step: h(o) -> s_0
                hidden_state, policy_logits, value = network.initial_inference(obs_batch)
                
                # Step 0 Loss
                target_value = val_batch[:, 0].unsqueeze(1)
                target_policy = pol_batch[:, 0]
                
                loss += F.mse_loss(value, target_value)
                loss += -torch.sum(target_policy * F.log_softmax(policy_logits, dim=1), dim=1).mean()
                
                # Unroll for K steps
                for k in range(config['unroll_steps']):
                    action = act_batch[:, k]
                    
                    # g(s_{k-1}, a_k) -> r_k, s_k
                    hidden_state, reward, policy_logits, value = network.recurrent_inference(hidden_state, action)
                    
                    target_reward = rew_batch[:, k].unsqueeze(1)
                    target_value = val_batch[:, k].unsqueeze(1)
                    target_policy = pol_batch[:, k]
                    
                    # Scale losses by 1/K
                    loss += (1.0 / config['unroll_steps']) * F.mse_loss(reward, target_reward)
                    loss += (1.0 / config['unroll_steps']) * F.mse_loss(value, target_value)
                    loss += (1.0 / config['unroll_steps']) * -torch.sum(target_policy * F.log_softmax(policy_logits, dim=1), dim=1).mean()
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)
                optimizer.step()
                
        if (episode + 1) % 10 == 0:
            avg = np.mean(episode_rewards[-10:])
            print(f"Game {episode + 1}/{config['num_games']} | Avg Reward: {avg:.2f}")
            if len(episode_rewards) >= 20 and np.mean(episode_rewards[-20:]) >= 450:
                print("Solved!")
                break
                
    env.close()
    return network, episode_rewards, config

def plot_rewards(rewards, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('MuZero Training on CartPole-v1')
    plt.xlabel('Game')
    plt.ylabel('Total Reward')
    plt.grid()
    save_path = os.path.join(save_dir, 'training_curve.png')
    plt.savefig(save_path)
    print(f"Training curve saved as '{save_path}'")

def evaluate_and_record(network, config, save_dir):
    filename = os.path.join(save_dir, 'muzero_agent.gif')
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    obs, _ = env.reset()
    frames = []
    
    done = False
    total_reward = 0
    
    while not done:
        frames.append(env.render())
        # To act during evaluation, we use MCTS deterministically
        policy, _ = run_mctx(config, network, obs)
        action = np.argmax(policy)
        
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
            
    env.close()
    print(f"Evaluation finished. Total Reward: {total_reward}")
    print(f"Saving video as {filename}...")
    imageio.mimsave(filename, frames, fps=30)
    print("Saved successfully!")

if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    save_dir = os.path.join(base_dir, "results", "muzero", f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Saving all results to: {save_dir}")
    print("Starting MuZero training on CartPole-v1...")
    
    network, rewards, config = train()
    
    model_path = os.path.join(save_dir, "muzero_network.pth")
    torch.save(network.state_dict(), model_path)
    
    plot_rewards(rewards, save_dir)
    evaluate_and_record(network, config, save_dir)
