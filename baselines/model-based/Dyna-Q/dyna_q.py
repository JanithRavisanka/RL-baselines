import gymnasium as gym
import numpy as np
import random
import os
import datetime
import matplotlib.pyplot as plt
import imageio

class DynaQAgent:
    """
    Dyna-Q Agent following Sutton's original architecture.
    """
    def __init__(self, state_dim, action_dim, alpha=0.1, gamma=0.95, epsilon=0.1, planning_steps=50):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.planning_steps = planning_steps  # 'n' in Sutton's paper

        # Initialize Q(S,A) to zeros
        self.q_table = np.zeros((state_dim, action_dim))

        # Initialize Model(S,A) -> R, S'
        # Since CliffWalking is deterministic, a simple dictionary is perfect.
        self.model = {}
        
        # Keep track of states and actions previously observed for planning step (f)
        self.observed_states = []
        self.observed_actions = {} # Maps state -> list of actions taken in that state

    def choose_action(self, state):
        """ (b) A <- e-greedy(S, Q) """
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            # Break ties randomly
            best_actions = np.argwhere(self.q_table[state] == np.amax(self.q_table[state])).flatten()
            return np.random.choice(best_actions)

    def learn(self, state, action, reward, next_state):
        # (d) Direct RL Update
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

        # (e) Model Update
        if state not in self.observed_states:
            self.observed_states.append(state)
            self.observed_actions[state] = []
        if action not in self.observed_actions[state]:
            self.observed_actions[state].append(action)
            
        self.model[(state, action)] = (reward, next_state)

        # (f) Planning Loop
        for _ in range(self.planning_steps):
            # S <- random previously observed state
            sim_state = random.choice(self.observed_states)
            # A <- random action previously taken in S
            sim_action = random.choice(self.observed_actions[sim_state])
            
            # R, S' <- Model(S,A)
            sim_reward, sim_next_state = self.model[(sim_state, sim_action)]
            
            # Q(S,A) <- Q(S,A) + alpha [R + gamma max Q(S',a) - Q(S,A)]
            sim_best_next_action = np.argmax(self.q_table[sim_next_state])
            sim_td_target = sim_reward + self.gamma * self.q_table[sim_next_state][sim_best_next_action]
            sim_td_error = sim_td_target - self.q_table[sim_state][sim_action]
            self.q_table[sim_state][sim_action] += self.alpha * sim_td_error

def train():
    env = gym.make('CliffWalking-v0')
    state_dim = env.observation_space.n
    action_dim = env.action_space.n

    agent = DynaQAgent(state_dim, action_dim, alpha=0.1, gamma=0.99, epsilon=0.1, planning_steps=50)
    
    num_episodes = 500
    episode_rewards = []
    
    print(f"Starting Dyna-Q Training on CliffWalking-v0 for {num_episodes} episodes...")

    for episode in range(num_episodes):
        # (a) S <- current state
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # (b) A <- e-greedy(S, Q)
            action = agent.choose_action(state)
            
            # (c) Take action A; observe reward R and next state S'
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # (d, e, f) Direct RL, Model Update, and Planning Loop
            agent.learn(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            
        episode_rewards.append(total_reward)
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode + 1}/{num_episodes} | Avg Reward (Last 50): {avg_reward:.2f}")

    env.close()
    return agent, episode_rewards

def plot_rewards(rewards, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Dyna-Q Training on CliffWalking-v0')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    window = 50
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, color='red', label='Moving Average (50 episodes)')
        plt.legend()
        
    plt.grid()
    save_path = os.path.join(save_dir, 'training_curve.png')
    plt.savefig(save_path)
    print(f"Training curve saved as '{save_path}'")

def evaluate_and_record(agent, save_dir):
    filename = os.path.join(save_dir, 'cliffwalking_agent.gif')
    print(f"Evaluating agent and saving video to {filename}...")
    
    # CliffWalking does not have rgb_array render mode natively without some tricks in older gym versions,
    # but in Gymnasium it does!
    env = gym.make('CliffWalking-v0', render_mode='rgb_array')
    state, _ = env.reset()
    frames = []
    
    done = False
    total_reward = 0
    
    # Temporarily set epsilon to 0 for greedy evaluation
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0 
    
    step_count = 0
    while not done and step_count < 100: # Cap to prevent infinite loops if not solved
        frames.append(env.render())
        action = agent.choose_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1
        
    # Append final frame
    if not done:
        print("Warning: Evaluation did not reach terminal state within 100 steps.")
    frames.append(env.render())
        
    agent.epsilon = original_epsilon
    env.close()
    
    print(f"Evaluation finished. Total Reward: {total_reward}")
    imageio.mimsave(filename, frames, fps=5) # Lower fps is better for gridworlds
    print("Saved successfully!")

if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    save_dir = os.path.join(base_dir, "results", "dyna_q", f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving all results to: {save_dir}")

    agent, rewards = train()
    
    model_path = os.path.join(save_dir, "q_table.npy")
    np.save(model_path, agent.q_table)
    print(f"Q-table saved to {model_path}")
    
    plot_rewards(rewards, save_dir)
    evaluate_and_record(agent, save_dir)
