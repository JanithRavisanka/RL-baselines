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


class MinMaxStats:
    def __init__(self):
        self.minimum = float("inf")
        self.maximum = -float("inf")

    def update(self, value):
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


def ucb_score(config, parent, child, min_max_stats):
    pb_c = math.log((parent.visit_count + config["pb_c_base"] + 1) / config["pb_c_base"]) + config["pb_c_init"]
    pb_c *= math.sqrt(max(parent.visit_count, 1)) / (child.visit_count + 1)
    prior_score = pb_c * child.prior
    value_score = 0.0
    if child.visit_count > 0:
        q_value = child.reward + config["discount"] * child.value()
        value_score = min_max_stats.normalize(q_value)
    return prior_score + value_score


def select_child(config, node, min_max_stats):
    scores = [(ucb_score(config, node, child, min_max_stats), action, child) for action, child in node.children.items()]
    _, action, child = max(scores, key=lambda item: item[0])
    return action, child


def run_mctx(config, network, obs):
    """Runs MCTS to build a search tree and returns the improved policy."""
    root = Node(0)
    
    # 1. Initial Inference
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        hidden_state, policy_logits, value = network.initial_inference(obs_tensor)
        
    root.hidden_state = hidden_state
    
    # Expand root from policy priors predicted by f(h(o)).
    # Children are created lazily once with prior probabilities.
    policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
    for a in range(config['action_dim']):
        root.children[a] = Node(policy_probs[a])
        
    # Add Dirichlet noise only at root to diversify self-play trajectories.
    noise = np.random.dirichlet([config['dirichlet_alpha']] * config['action_dim'])
    for a in root.children:
        root.children[a].prior = root.children[a].prior * (1 - config['dirichlet_eps']) + noise[a] * config['dirichlet_eps']
        
    # 2. Run simulations:
    # selection (PUCT) -> expansion (dynamics step) -> backup (discounted return)
    min_max_stats = MinMaxStats()
    for _ in range(config['num_simulations']):
        node = root
        search_path = [node]
        history = [] # actions taken
        
        # Selection phase: descend tree by highest UCB score.
        while node.expanded():
            action, node = select_child(config, node, min_max_stats)
            history.append(action)
            search_path.append(node)
            
        # Expansion/evaluation: one recurrent model step from parent hidden state.
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
            
        # Backup value through search path:
        # parent value = reward + discount * child value
        for node in reversed(search_path):
            node.value_sum += v
            node.visit_count += 1
            min_max_stats.update(node.value())
            # Discount value for the parent
            v = node.reward + config['discount'] * v
            
    # 3. Calculate Target Policy from Visit Counts
    visit_counts = np.array([root.children[a].visit_count for a in range(config['action_dim'])])
    policy = visit_counts / max(np.sum(visit_counts), 1)
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
        
    def sample(self, batch_size, unroll_steps, config):
        # We sample an observation, and then sequences of length `unroll_steps`
        obs_batch, action_batch, reward_batch, policy_batch, value_batch = [], [], [], [], []
        reward_mask_batch, policy_mask_batch = [], []

        for _ in range(batch_size):
            game = self.buffer[np.random.randint(len(self.buffer))]
            start = np.random.randint(len(game.observations))
            obs_batch.append(game.observations[start])

            actions, rewards, reward_mask = [], [], []
            policies, values, policy_mask = [], [], []

            for i in range(unroll_steps):
                idx = start + i
                if idx < len(game.observations):
                    actions.append(game.actions[idx])
                    rewards.append(game.rewards[idx])
                    reward_mask.append(1.0)
                else:
                    # Absorbing/padding state — kept for tensor shape consistency
                    # but excluded from losses by masks.
                    actions.append(np.random.randint(config["action_dim"]))
                    rewards.append(0.0)
                    reward_mask.append(0.0)

            for i in range(unroll_steps + 1):
                idx = start + i
                if idx < len(game.observations):
                    policies.append(game.target_policies[idx])
                    values.append(make_value_target(game, idx, config))
                    policy_mask.append(1.0)
                else:
                    policies.append(np.ones(config["action_dim"]) / config["action_dim"])
                    values.append(0.0)
                    policy_mask.append(0.0)

            action_batch.append(actions)
            reward_batch.append(rewards)
            reward_mask_batch.append(reward_mask)
            policy_batch.append(policies)
            value_batch.append(values)
            policy_mask_batch.append(policy_mask)

        return (
            torch.FloatTensor(np.array(obs_batch)).to(device),
            torch.LongTensor(np.array(action_batch)).to(device),
            torch.FloatTensor(np.array(reward_batch)).to(device),
            torch.FloatTensor(np.array(policy_batch)).to(device),
            torch.FloatTensor(np.array(value_batch)).to(device),
            torch.FloatTensor(np.array(reward_mask_batch)).to(device),
            torch.FloatTensor(np.array(policy_mask_batch)).to(device),
        )


def make_value_target(game, start, config):
    # n-step bootstrap target:
    # sum_{i=0..n-1} gamma^i r_{t+i} + gamma^n v_{t+n}
    value = 0.0
    discount = 1.0
    bootstrap_index = start + config["td_steps"]
    for i in range(start, min(bootstrap_index, len(game.rewards))):
        value += discount * game.rewards[i]
        discount *= config["discount"]
    if bootstrap_index < len(game.target_values):
        value += discount * game.target_values[bootstrap_index]
    return value

# ==============================================================================
# 4. Training Loop (BPTT)
# ==============================================================================

def temperature_schedule(episode):
    if episode < 50:
        return 1.0
    if episode < 100:
        return 0.5
    return 0.25


def train():
    env = gym.make("CartPole-v1")
    config = {
        'obs_dim': env.observation_space.shape[0],
        'action_dim': env.action_space.n,
        'num_simulations': 25, # MCTS runs per real environment step
        'discount': 0.99,
        'pb_c_init': 1.25, # UCB exploration constant
        'pb_c_base': 19652,
        'dirichlet_alpha': 0.25,
        'dirichlet_eps': 0.25,
        'td_steps': 5,
        'unroll_steps': 5, # Unroll K steps during BPTT training
        'batch_size': 64,
        'num_games': 2000,
        'lr': 0.002,
        'temperature_fn': temperature_schedule,
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
        step = 0

        while not done:
            policy, root_value = run_mctx(config, network, obs)

            # Visit-count temperature schedule (paper: hot for exploration, cool later).
            temperature = config['temperature_fn'](episode)
            if temperature == 0:
                action = int(np.argmax(policy))
            else:
                logits = np.log(policy + 1e-9) / temperature
                exp_logits = np.exp(logits - logits.max())
                sampling_probs = exp_logits / exp_logits.sum()
                action = int(np.random.choice(config['action_dim'], p=sampling_probs))

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            game.store_search_statistics(obs, action, reward, policy, root_value)
            obs = next_obs
            step += 1

        buffer.save_game(game)
        episode_rewards.append(total_reward)

        # --- Unrolled BPTT Training ---
        if len(buffer.buffer) > 10:
            for _ in range(2):
                (
                    obs_batch,
                    act_batch,
                    rew_batch,
                    pol_batch,
                    val_batch,
                    rew_mask,
                    pol_mask,
                ) = buffer.sample(config['batch_size'], config['unroll_steps'], config)

                loss = 0.0

                # Initial step: h(o) -> s_0
                hidden_state, policy_logits, value = network.initial_inference(obs_batch)

                target_value0 = val_batch[:, 0].unsqueeze(1)
                target_policy0 = pol_batch[:, 0]
                policy_w0 = pol_mask[:, 0].unsqueeze(1)

                value_loss0 = ((value - target_value0) ** 2 * policy_w0).sum() / policy_w0.sum().clamp_min(1.0)
                policy_loss0 = -(target_policy0 * F.log_softmax(policy_logits, dim=1)).sum(-1, keepdim=True)
                policy_loss0 = (policy_loss0 * policy_w0).sum() / policy_w0.sum().clamp_min(1.0)
                loss = loss + value_loss0 + policy_loss0

                # Unroll recurrent dynamics for K steps and accumulate masked losses.
                for k in range(config['unroll_steps']):
                    action = act_batch[:, k]
                    hidden_state, reward, policy_logits, value = network.recurrent_inference(hidden_state, action)

                    rw = rew_mask[:, k].unsqueeze(1)
                    pw = pol_mask[:, k + 1].unsqueeze(1)

                    target_reward = rew_batch[:, k].unsqueeze(1)
                    target_value = val_batch[:, k + 1].unsqueeze(1)
                    target_policy = pol_batch[:, k + 1]

                    reward_loss = ((reward - target_reward) ** 2 * rw).sum() / rw.sum().clamp_min(1.0)
                    value_loss = ((value - target_value) ** 2 * pw).sum() / pw.sum().clamp_min(1.0)
                    policy_loss = -(target_policy * F.log_softmax(policy_logits, dim=1)).sum(-1, keepdim=True)
                    policy_loss = (policy_loss * pw).sum() / pw.sum().clamp_min(1.0)

                    scale = 1.0 / config['unroll_steps']
                    loss = loss + scale * (reward_loss + value_loss + policy_loss)

                    # Scale gradients to recurrent hidden state path (MuZero Appendix G)
                    # to stabilize very deep unroll updates.
                    hidden_state.register_hook(lambda grad: grad * 0.5)

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
