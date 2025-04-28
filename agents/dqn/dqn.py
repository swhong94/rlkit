import torch 
import torch.nn as nn 
import torch.optim as optim 
import numpy as np 
import gymnasium as gym 

from .network import QNetwork, DuelingQNetwork 
from utils.noise import EpsilonGreedy 
from utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer 
from utils.logger import Logger 


class DQN: 
    def __init__(self, env, hidden_dims=[128, 128], lr=1e-4, gamma=0.99, tau=0.004, capacity=10000, 
                 batch_size=64, episode_length=200, log_dir='logs/dqn_logs', plot_window=None, device='cpu', 
                 per=False, dueling_dqn=False, double_dqn=False): 
        self.device = device 

        # Define Environment 
        self.env = env 
        self.state_dim = env.observation_space.shape[0] 
        self.action_dim = env.action_space.n 

        # Initialize Networks 
        q_network = DuelingQNetwork if dueling_dqn else QNetwork 
        self.q_network = q_network(self.state_dim, hidden_dims, self.action_dim).float().to(device) 
        self.target_q_network = q_network(self.state_dim, hidden_dims, self.action_dim).float().to(device) 
        self.target_q_network.load_state_dict(self.q_network.state_dict()) 

        # optimizer 
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)  

        # Buffer 
        self.buffer = PrioritizedReplayBuffer(capacity=capacity) if per else ReplayBuffer(capacity=capacity) 

        # Exploration Strategy: Epsilon Greedy 
        self.exploration = EpsilonGreedy()

        # Parameters and options 
        self.gamma = gamma 
        self.tau = tau 
        self.batch_size = batch_size 
        self.episode_length = episode_length 
        self.double_dqn = double_dqn 
        self.dueling_dqn = dueling_dqn 
        self.per = per 
        # Logging infos 
        self.logger = Logger(log_dir=log_dir, log_name_prefix='DQN', plot_window=plot_window) 
        self.timestamp = 0 

    def act(self, state, explore=True): 
        if explore and self.exploration.sample(): 
            return self.env.action_space.sample() 
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device) 
        with torch.no_grad(): 
            q_values = self.q_network(state) 
        return q_values.argmax().item() 

    def train_step(self, ): 
        if len(self.buffer) < self.batch_size: 
            return 0.0 

        # Sample batch 
        if self.per: 
            states, actions, rewards, next_states, dones, indices, weights = self.buffer.sample(self.batch_size) 
            weights= torch.FloatTensor(weights).to(self.device) 
        else: 
            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size) 
            weights = torch.ones(self.batch_size).to(self.device) 

        states = torch.FloatTensor(states).to(self.device) 
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device) 
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device) 
        next_states = torch.FloatTensor(next_states).to(self.device) 
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device) 

        with torch.no_grad(): 
            if self.double_dqn: 
                next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True) 
                target_q_values = self.target_q_network(next_states).gather(1, next_actions) 
            else:
                target_q_values = self.target_q_network(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + self.gamma * (1-dones) * target_q_values 

        # Compute Current Q-vals 
        current_q = self.q_network(states).gather(1, actions) 

        # Compute Loss 
        loss = (weights * (current_q - target_q)**2).mean() 

        # Optimize 
        self.optimizer.zero_grad() 
        loss.backward() 
        self.optimizer.step() 

        if self.per: 
            td_errors = (current_q - target_q).abs().detach().cpu().numpy().flatten() 
            self.buffer.update_priorities(indices, td_errors) 
        
        # Soft update target network 
        for param, target_param in zip(self.q_network.parameters(), self.target_q_network.parameters()): 
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data) 

        self.timestamp += 1

        return loss.item() 

    def train_episode(self, ): 
        state, _ = self.env.reset() 
        episode_reward, total_loss = 0, 0 
        step = 0 

        for t in range(self.episode_length): 
            action = self.act(state) 
            next_state, reward, done, truncated, info = self.env.step(action) 
            self.buffer.add(state, action, reward, next_state, (done or truncated)) 

            loss = self.train_step() 
            state = next_state 
            episode_reward += reward 
            total_loss += loss 
            step += 1 

            if done or truncated: 
                break 
        self.exploration.decay() 
        avg_loss = total_loss / max(step, 1) 

        return episode_reward, avg_loss 

    def train(self, num_episodes, max_timesteps, log_interval=100, save_interval=None): 
        for episode in range(num_episodes): 
            episode_reward, avg_loss = self.train_episode() 

            if episode % log_interval == 0: 
                self.logger.info(f"Episode[step] {episode:>4d}[{self.timestamp:>6d}] | "
                                 f"Reward: {episode_reward:>10.3f} | "
                                 f"Avg Loss: {avg_loss:>10.3f} ")
            metrics = {
                "episode_reward": episode_reward, 
                "avg_loss": avg_loss, 
            }
            self.logger.log_metrics(episode, metrics) 
        self.logger.close() 

    def save(self, path): 
        pass 

    def load(self, path): 
        pass 

    def evaluate(self, num_episodes, max_timesteps): 
        pass 


if __name__ == "__main__": 
    env = gym.make("CartPole-v1") 
    agent = DQN(
        env=env, 
        hidden_dims=[128, 128], 
        gamma=0.99, 
        tau=0.005, 
        lr=1e-4, 
        capacity=10000, 
        batch_size=64, 
        log_dir='logs/dqn_logs', 
        plot_window=30, 
        double_dqn=True, 
        dueling_dqn=False, 
        per=True, 
    )

    agent.train(
        num_episodes=500, 
        max_timesteps=10000,  
        log_interval=10, 
        save_interval=100, 
    )