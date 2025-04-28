import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim 
import gymnasium as gym 
from collections import deque 
import random

from utils import ReplayBuffer, OUNoise
from agents.ddpg.network import Actor, Critic 
from utils import Logger 

    


class DDPG(nn.Module): 
    def __init__(self, 
                 env, hidden_dims,
                 gamma=0.99, tau=0.005, 
                 actor_lr=1e-4, critic_lr=1e-3,
                 capacity=10000, batch_size=64, episode_length=200,
                 log_dir='logs/ddpg_logs', plot_window=None, 
                 device='cpu'):
        super(DDPG, self).__init__() 

        self.env = env 
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0] 
        self.action_low, self.action_high = float(env.action_space.low[0]), float(env.action_space.high[0])

        self.actor = Actor(input_dim=self.state_dim, 
                           hidden_dims=hidden_dims, 
                           output_dim=self.action_dim, 
                           max_action=self.action_high,).float()
        self.actor_target = Actor(input_dim=self.state_dim, 
                           hidden_dims=hidden_dims, 
                           output_dim=self.action_dim, 
                           max_action=self.action_high,).float() 
        self.actor_target.load_state_dict(self.actor.state_dict()) 
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr) 

        self.critic = Critic(input_dim=self.state_dim, 
                             hidden_dims=hidden_dims, 
                             output_dim=self.action_dim).float() 
        self.critic_target = Critic(input_dim=self.state_dim, 
                             hidden_dims=hidden_dims, 
                             output_dim=self.action_dim).float() 
        self.critic_target.load_state_dict(self.critic.state_dict()) 
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma 
        self.tau = tau 
        self.episode_length = episode_length 
        self.batch_size = batch_size 
        self.timestamp = 0 

        self.buffer = ReplayBuffer(capacity) 

        self.noise = OUNoise(self.action_dim)

        self.logger = Logger(log_dir=log_dir, log_name_prefix='ddpg', plot_window=plot_window)

        self.device = device 
        self.to(device) 


    def act(self, state, noise=True): 
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad(): 
            action = self.actor(state).cpu().numpy().flatten() 
        
        if noise: 
            action += self.noise.sample() 
        clipped_action = np.clip(action, self.action_low, self.action_high) 
        return clipped_action  
    
    def train_step(self, ):
        if len(self.buffer) < self.batch_size: 
            return 0.0, 0.0     # Placeholder for no training 
        
        # Sample a data from batch 
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size) 
        states = torch.FloatTensor(states).to(self.device) 
        actions = torch.FloatTensor(actions).to(self.device) 
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1) 
        next_states = torch.FloatTensor(next_states).to(self.device) 
        dones = torch.FloatTensor(dones).unsqueeze(1) 

        # 1. Update Critic 
        with torch.no_grad():
            next_actions = self.actor_target(next_states) 
            target_q = self.critic_target(next_states, next_actions) 
            target_q = rewards + self.gamma * (1-dones) * target_q 
        current_q = self.critic(states, actions) 

        # 2. Critic Loss 
        critic_loss = nn.MSELoss()(current_q, target_q) 

        # 3. Update Critic 
        self.critic_optimizer.zero_grad() 
        critic_loss.backward() 
        self.critic_optimizer.step() 

        # 4. Actor Loss 
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean() 

        # 5. Update Actor 
        self.actor_optimizer.zero_grad() 
        actor_loss.backward() 
        self.actor_optimizer.step() 

        # 6. Soft Update of Target Networks 
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0-self.tau) * target_param.data) 
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0-self.tau) * target_param.data) 
        
        self.timestamp += 1 

        return actor_loss.item(), critic_loss.item() 
    
    def train_episode(self): 
        state, _ = self.env.reset() 
        episode_reward = 0; total_actor_loss = 0; total_critic_loss = 0 
        step = 0 

        for t in range(self.episode_length): 
            action = self.act(state) 
            next_state, reward, done, truncated, info = self.env.step(action) 
            self.buffer.add(state, action, reward, next_state, (done or truncated)) 

            actor_loss, critic_loss = self.train_step() 
            state = next_state 
            episode_reward += reward 
            total_actor_loss += actor_loss 
            total_critic_loss += critic_loss 
            step += 1 

            if done or truncated: 
                break 
        
        avg_actor_loss = total_actor_loss / max(step, 1) 
        avg_critic_loss = total_critic_loss / max(step, 1) 
        return episode_reward, avg_actor_loss, avg_critic_loss 

    def train(self, max_episodes, max_timesteps, log_interval=10, save_interval=None): 
        self.max_timesteps = max_timesteps # Not used yet 
        for episode in range(max_episodes): 
            episode_reward, avg_actor_loss, avg_critic_loss = self.train_episode() 

            if episode % log_interval == 0: 
                self.logger.info(f"Episode[step]: {episode:>4d}[{self.timestamp:<6d}] | "
                                 f"Reward: {episode_reward:>10.3f} | "
                                 f"Avg. Actor Loss: {avg_actor_loss:>10.4f} | "
                                 f"Avg. Critic Loss: {avg_critic_loss:>10.4f} | "
                                 ) 
            metrics = {
                "episode_reward": episode_reward, 
                "actor_loss": avg_actor_loss, 
                "critic_loss": avg_critic_loss, 
            }
            self.logger.log_metrics(episode, metrics) 
        
        self.logger.close() 




if __name__ == "__main__": 
    env = gym.make("Pendulum-v1") 
    agent = DDPG(
        env=env, 
        hidden_dims=[256] * 2,
        gamma=0.99, 
        tau=0.005, 
        actor_lr=1e-4, 
        critic_lr=1e-3, 
        capacity=10000, 
        batch_size=64, 
        log_dir="logs/ddpg_logs",
        plot_window=30)
    
    agent.train(
        max_episodes=100, 
        max_timesteps=500, 
        log_interval=10, 
        save_interval=100
    )

                

        


