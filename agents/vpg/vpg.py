import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym 
     
from agents.vpg.network import PolicyNet, ValueNet 
from utils import Logger 


# class PolicyNet(nn.Module): 
#     def __init__(self, state_dim, hidden_dims, action_dim, activation="ReLU"): 
#         super(PolicyNet, self).__init__() 
#         layers = [] 
#         prev_dim = state_dim 
#         for h_dim in hidden_dims:  
#             layers.append(nn.Linear(prev_dim, h_dim))
#             layers.append(getattr(nn, activation)()) 
#             prev_dim = h_dim 
#         layers.append(nn.Linear(prev_dim, action_dim)) 
#         layers.append(nn.Softmax(dim=-1))     # Probability of each action
#         self.fc_net = nn.Sequential(*layers)  # Policy Network 

#     def forward(self, state): 
#         return self.fc_net(state)
    

# class ValueNet(nn.Module): 
#     def __init__(self, state_dim, hidden_dims, activation="ReLU"): 
#         super(ValueNet, self).__init__() 
#         layers = [] 
#         prev_dim = state_dim 
#         for h_dim in hidden_dims: 
#             layers.append(nn.Linear(prev_dim, h_dim)) 
#             layers.append(getattr(nn, activation)()) 
#             prev_dim = h_dim 
#         layers.append(nn.Linear(prev_dim, 1)) 
#         self.fc_net = nn.Sequential(*layers) 

#     def forward(self, state): 
#         return self.fc_net(state)
        
# REINFORCE Agnet 
class VanillaPolicyGradient(nn.Module): 
    """Vanilla Policy Gradient with baseline"""
    def __init__(self, 
                 env, 
                 hidden_dims, 
                 gamma=0.99, lambda_gae=0.95, gae=True, episode_length=200, 
                 policy_lr=1e-3,
                 value_lr=1e-3, 
                 log_dir="logs/reinforce_logs", plot_window=None,
                 device="cpu"):
        super(VanillaPolicyGradient, self).__init__() 
        self.env = env 
        self.state_dim = env.observation_space.shape[0] 
        self.action_dim = env.action_space.n 

        self.policy = PolicyNet(self.state_dim, hidden_dims, self.action_dim,) 
        self.value = ValueNet(self.state_dim, hidden_dims) 

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr) 
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=value_lr) 

        self.gamma = gamma 
        self.lambda_gae = lambda_gae 
        self.gae = gae 
        self.episode_length = episode_length 
        self.timestamp = 0 

        self.grad_clip = True 
        self.logger = Logger(log_dir, log_name_prefix="REINFORCE", plot_window=plot_window) 

        self.device = device
        self.to(self.device) 

    def select_action(self, state): 
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device) 
        probs = self.policy(state) 
        action_dist = torch.distributions.Categorical(probs) 
        action = action_dist.sample() 
        log_prob = action_dist.log_prob(action) 
        return action.item(), log_prob 
    
    def compute_gae(self, rewards, values, next_value, done):
        rewards = torch.FloatTensor(rewards)
        values = values.squeeze() 
        next_values = torch.cat([values[1:], next_value.unsqueeze(0)])
        # Compute TD errors 
        td_errors = rewards + self.gamma * next_values.detach() * (1-done) - values.detach()
        advantages = [] 
        gae = 0 
        for delta in reversed(td_errors): 
            gae = delta + self.gamma * self.lambda_gae * gae 
            advantages.insert(0, gae) 
        advantages = torch.FloatTensor(advantages).to(self.device) 
        returns = advantages + values 
        return advantages, returns 



    def update_trajectory(self, states, log_probs, rewards, next_state, done): 
        """Train the agent for one episode"""
        states = torch.FloatTensor(states).to(self.device) 
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device) 

        # compute values 
        values = self.value(states) 
        with torch.no_grad(): 
            next_values = self.value(next_state).squeeze()

        # compute GAE advantages and returns 
        advantages, returns = self.compute_gae(rewards, values, next_values, done) 
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  

        policy_loss = [] 
        for log_prob, advantage in zip(log_probs, advantages): 
            policy_loss.append(-log_prob * advantage) 
        policy_loss = torch.stack(policy_loss).sum() 

        # Value Loss 
        value_loss = nn.MSELoss()(returns, values.squeeze())

        # Update policy 
        self.policy_optimizer.zero_grad() 
        policy_loss.backward() 
        if self.grad_clip: 
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0) 
        self.policy_optimizer.step() 

        # Update value 
        self.value_optimizer.zero_grad() 
        value_loss.backward() 
        if self.grad_clip: 
            nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=1.0) 
        self.value_optimizer.step() 
        
        return policy_loss.item(), value_loss.item() 
    
    def train_episode(self): 
        """Train the agent for one episode"""
        state, _ = self.env.reset() 
        states, log_probs, rewards = [], [], [] 
        total_reward = 0
        step = 0 

        # collect trajectory 
        for t in range(self.episode_length): 
            action, log_prob = self.select_action(state) 
            next_state, reward, done, truncated, info = self.env.step(action) 

            states.append(state) 
            log_probs.append(log_prob) 
            rewards.append(reward) 
            state = next_state 
            self.timestamp += 1 
            # print(f"({done or truncated})")
            if done or truncated or (t==self.episode_length-1): 
                policy_loss, value_loss = self.update_trajectory(states, log_probs, rewards, next_state, done or truncated) 
                total_reward += sum(rewards) 
                break 
        
        return total_reward, policy_loss, value_loss 
    

    def train(self, max_episodes, max_steps, log_interval=10, save_interval=None): 
        self.max_steps = max_steps 
        """Train the agent"""
        for episode in range(max_episodes): 
            total_reward, policy_loss, value_loss = self.train_episode() 
            if episode % log_interval == 0: 
                self.logger.info(f"Episode {episode:>4d} | Episode Reward: {total_reward:>10.4f} |"
                                 f"Policy Loss: {policy_loss:>10.4f} | Value Loss: {value_loss:>10.4f}")
            metrics = {
                "episode_reward": total_reward, 
                "policy_loss": policy_loss, 
                "value_loss": value_loss, 
            }
            self.logger.log_metrics(episode, metrics) 

            if self.timestamp > max_steps: 
                break 
        self.logger.close() 


if __name__ == "__main__": 
    env = gym.make("CartPole-v1") 
    agent = VanillaPolicyGradient(env, 
                      hidden_dims=[128, 128], 
                      gamma=0.99, lambda_gae=0.95, gae=True, episode_length=500, 
                      policy_lr=1e-3, value_lr=1e-3, log_dir="logs/reinforce_logs", plot_window=30, device="cpu") 
    agent.train(max_episodes=500, max_steps=2_000_000, log_interval=50) 