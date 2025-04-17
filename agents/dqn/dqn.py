# agents/dqn.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import os 
# from models.mlp import PolicyNet

from .network import ValueNetwork
from utils import Logger, ReplayBuffer


class DQN(nn.Module): 
    """
    A DQN agent with experience replay with target network 
    
    Args: 
        env (gym.Env): environment to train on 
        hidden_dims (list): list of hidden dimensions for the MLP 
        lr (float): learning rate 
        gamma (float): discount factor 
        epsilon_start (float): starting epsilon for epsilon-greedy policy 
        epsilon_end (float): ending epsilon for epsilon-greedy policy 
        epsilon_decay (float): decay steps for epsilon  
        buffer_size (int): size of the replay buffer 
        batch_size (int): batch size for training 
        target_update_frequency (int): interval for updating the target network 
        log_dir (str): directory to save the logs 
        plot_window (int): window size for the plot (moving average) 
        device (str): device to run the model on ('cpu' or 'cuda') <- not used yet 
    """
    def __init__(self, 
                 env, 
                 hidden_dims=[128, 128], lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=10000, 
                 buffer_size=10000, batch_size=64, target_update_frequency=1000, max_steps=200,
                 log_dir="logs/dqn_logs", plot_window=None, device='cpu'): 
        super(DQN, self).__init__() 

        self.env = env 
        self.state_dim = env.observation_space.shape[0] 
        assert isinstance(env.action_space, gym.spaces.Discrete), "Action space must be discrete for DQN"
        self.action_dim = env.action_space.n 

        self.q_network = ValueNetwork(self.state_dim, hidden_dims, self.action_dim, activation="ReLU") 
        self.target_network = ValueNetwork(self.state_dim, hidden_dims, self.action_dim, activation="ReLU") 

        # Initialize target network with same weights as q_network 
        self.target_network.load_state_dict(self.q_network.state_dict()) 
        self.target_network.eval()  # Set target_network to evaluation mode to avoid training 

        # Optimizer 
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr) 

        # Training parameters 
        self.gamma = gamma 
        self.epsilon = epsilon_start 
        self.epsilon_start = epsilon_start 
        self.epsilon_end = epsilon_end  
        self.epsilon_decay = epsilon_decay 
        self.batch_size = batch_size 
        self.max_steps = max_steps 
        self.timestamp = 0 
        self.tuf = target_update_frequency  
        self.gradient_clipping = True 

        # initialize replay buffer 
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)  

        # Initialize logger 
        self.logger = Logger(log_dir, log_name_prefix="dqn", plot_window=plot_window) 

        # Initialize device 
        self.device = device 
        self.to(device) 


    def act(self, state, explore=True): 
        """Select an action given the current state with epsilon-greedy policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device) 
        if explore and (np.random.rand() < self.epsilon):    # Exploration
            action = np.random.choice(self.action_dim) 
        else: 
            with torch.no_grad():                            # Greedy action 
                q_values = self.q_network(state_tensor) 
                action = q_values.argmax(dim=1).item()
        return action 
    
    def update_epsilon(self): 
        """Decay epsilon overtime"""    
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                        np.exp(-1.0 * self.timestamp / self.epsilon_decay) 
    

    def train_step(self, ): 
        """Training process for a single step""" 
        if len(self.replay_buffer) < self.batch_size: 
            return 0.0, 0.0 
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size) 
        states = torch.FloatTensor(states).to(self.device) 
        actions = torch.LongTensor(actions).to(self.device) 
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1) 
        next_states = torch.FloatTensor(next_states).to(self.device) 
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device) 

        # Compute Q-values 
        q_vals = self.q_network(states).gather(1, actions.unsqueeze(1))      # Gathers Q-values matching the actions (like indexing)
        
        # Compute target Q-values
        with torch.no_grad(): 
            next_q_vals = self.target_network(next_states).max(dim=1)[0].unsqueeze(1) 
            targets = rewards + self.gamma * (1-dones) * next_q_vals 
        
        # Compute Loss 
        loss = nn.MSELoss()(q_vals, targets) 

        # Optimize 
        self.optimizer.zero_grad() 
        loss.backward() 
        # Gradient clipping 
        if self.gradient_clipping: 
            nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step() 

        # Update target network 
        self.timestamp += 1  
        if self.timestamp % self.tuf == 0: 
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item(), 0.0   # placeholder for critic loss 



    def train_episode(self ): 
        """Training process for a single episode"""
        state, _ = self.env.reset() 
        total_reward = 0 
        total_loss = 0 
        steps = 0 

        for t in range(self.max_steps): 
            action = self.act(state) 
            next_state, reward, done, truncated, info = self.env.step(action)  

            # Store transition in replay buffer 
            self.replay_buffer.add(state, action, reward, next_state, float(done or truncated))

            # Update
            loss, _ = self.train_step()

            total_reward += reward 
            total_loss += loss 
            steps += 1 

            # Update state 
            state = next_state 
            # Update epsilon 
            self.update_epsilon() 

            if done or truncated: 
                break 
        
        avg_loss = total_loss / steps 
        return total_reward, avg_loss, 0.0 # Placeholder for critic loss 




    
    def train(self, num_episodes, max_timesteps, log_interval=100, save_interval=100): 
        """Train loop for the DQN agent"""
        self.max_timesteps = max_timesteps 
        for episode in range(num_episodes): 
            total_reward, avg_loss, _ = self.train_episode() 
            if episode % log_interval == 0: 
                self.logger.info(f"Episode(steps) {episode:>4d}({self.timestamp:>7d}) | Reward: {total_reward:>10.2f} "
                                 f"Loss: {avg_loss:>10.4f} | Epsilon: {self.epsilon:>6.4f}")
            metrics = {
                "total_reward": total_reward, 
                "avg_loss": avg_loss, 
                "epsilon": self.epsilon,
            }
            self.logger.log_metrics(episode, metrics)

            if episode % save_interval == 0 and episode > 0:
                self.save(f"checkpoints/dqn/dqn_checkpoint_{episode}.pth")
        self.logger.close()  


    def save(self, path): 
        """Save the model to a file"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save({
            "q_network_state_dict": self.q_network.state_dict(), 
            "target_network_state_dict": self.target_network.state_dict(), 
            "optimizer":  self.optimizer.state_dict(), 
            "timestamp": self.timestamp,
            "epsilon": self.epsilon, 
        }, path)


    def load(self, path): 
        """Load the model from a file"""
        checkpoint = torch.load(path, map_location=self.device) 
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"]) 
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"]) 
        self.optimizer.load_state_dict(checkpoint["optimizer"]) 
        self.timestamp = checkpoint["timestamp"]
        self.epsilon = checkpoint["epsilon"]

    def evaluate(self, num_episodes, max_timesteps): 
        """Evaluate the agent's performance (without exploration)"""
        total_rewards = [] 
        for _ in range(num_episodes): 
            state, _ = self.env.reset() 
            episode_reward = 0 
            for _ in range(max_timesteps): 
                action = self.act(state, explore=False) 
                next_state, reward, done, truncated, info = self.env.step(action) 
                episode_reward += reward 
                state = next_state 

                if done or truncated: 
                    break 
            total_rewards.append(episode_reward) 
        return np.mean(total_rewards) 




if __name__ == "__main__": 
    env = gym.make("CartPole-v1") 
    agent = DQN(env=env, 
                hidden_dims=[64, 64], 
                lr=1e-3, 
                gamma=0.99, 
                epsilon_start=1.0, 
                epsilon_end=0.01, 
                epsilon_decay=10000,
                buffer_size=10000, 
                batch_size=64, 
                target_update_frequency=100, 
                max_steps=200, 
                log_dir="logs/dqn_logs", 
                plot_window=100, 
                device='cpu')
    
    num_episodes = 2000
    max_timesteps = 200
    log_interval = 10 

    agent.train(num_episodes, max_timesteps, log_interval) 
    mean_reward = agent.evaluate(num_episodes=1, 
                                 max_timesteps=max_timesteps)
    print(f"Mean reward: {mean_reward:>10.2f}")

