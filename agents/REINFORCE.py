import torch 
import torch.nn as nn 
import torch.optim as optim 
import gymnasium as gym 
import numpy as np 

from torch.optim import Adam 
from torch.distributions import Normal 
from models.mlp import MLP_Policy, MLP_Value 
from utils import Logger 


class PolicyNet(nn.Module): 
    def __init__(self, state_dim, hidden_dim, action_dim): 
        super(PolicyNet, self).__init__() 
        self.mlp_block = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU()
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim) 
        self.log_std = nn.Parameter(torch.zeros(1, action_dim)) 

    def forward(self, x): 
        x = self.mlp_block(x) 
        mu = torch.tanh(self.mu_head(x)) * 2.0      # Scale to [-2, 2] 
        std = torch.exp(self.log_std) 
        return mu, std 
    

class ValueNet(nn.Module): 
    def __init__(self, state_dim, hidden_dim): 
        super(ValueNet, self).__init__() 
        self.mlp_block = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, 1) 
        )

    def forward(self, x): 
        return self.mlp_block(x) 
    


class REINFORCE: 
    def __init__(self,
                 env,
                 hidden_dims, 
                 policy_lr=0.001, 
                 value_lr=0.01, 
                 gamma=0.99, 
                 lambda_gae=0.95, 
                 gae=True, 
                 max_steps=200, 
                 log_dir="logs/reinforce_logs", 
                 plot_window=None): 
        self.env = env

        input_dim = env.observation_space.shape[0]
        if isinstance(env.action_space, gym.spaces.Box): 
            output_dim = env.action_space.shape[0]
        else: 
            output_dim = env.action_space.n 

        self.policy = MLP_Policy(input_dim, hidden_dims, output_dim) 
        self.value = MLP_Value(input_dim, hidden_dims)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=policy_lr) 
        self.value_optimizer = Adam(self.value.parameters(), lr=value_lr)
        self.gamma = gamma 
        self.lambda_gae = lambda_gae 
        self.gae = gae 
        self.max_steps = max_steps 
        self.min_max = [torch.FloatTensor(self.env.action_space.low), torch.FloatTensor(self.env.action_space.high)]
        self.timestamp = 0 
        self.logger = Logger(log_dir=log_dir, log_name_prefix="reinforce", plot_window=plot_window) 



    def act(self, state): 
        state_tensor = torch.FloatTensor(state)
        mu, std = self.policy(state_tensor) 
        dist = Normal(mu, std) 
        action = dist.sample() 
        action_clamped = torch.clamp(action, self.min_max[0], self.min_max[1]) 

        log_prob = dist.log_prob(action) 
        # Convert action to numpy and remove extra dimensions
        action_numpy = action_clamped.detach().numpy().flatten()
        return action_numpy, log_prob, state_tensor 


    def compute_gae(self, rewards, values, next_values): 
        deltas = rewards + self.gamma * next_values - values 
        advantages = [] 
        adv = 0 
        # print(f"Deltas: {deltas}")
        for delta in deltas.flip(dims=[0]):
            adv = delta + self.gamma * self.lambda_gae * adv 
            advantages.insert(0, adv) 
        return torch.FloatTensor(advantages) 
        

    def compute_returns(self, rewards): 
        discounted_rewards = [] 
        R = 0 
        for reward in rewards[::-1]: 
            R = reward + self.gamma * R 
            discounted_rewards.insert(0, R) 
        return torch.FloatTensor(discounted_rewards) 
    

    def train_episode(self): 
        state, _ = self.env.reset() 
        log_probs = [] 
        rewards = [] 
        states = [] 
        next_states = [] 
        step = 0 

        # collect trajectory 
        for t in range(self.max_steps): 
            action, log_prob, state_tensor = self.act(state) 
            # print(f"Action: {action}, Log Prob: {log_prob}, State Tensor: {state_tensor}")
            next_state, reward, done, truncated, _ = self.env.step(action) 
            log_probs.append(log_prob) 
            rewards.append(reward) 
            states.append(state_tensor) 
            next_states.append(torch.FloatTensor(next_state)) 
            state = next_state 
            step += 1 

            if done or truncated: 
                break 
        
        # compute values, returns, and gae 
        states_tensor = torch.stack(states) 
        next_states_tensor = torch.stack(next_states) 
        values = self.value(states_tensor).squeeze() 
        next_values = self.value(next_states_tensor).squeeze() 

        if self.gae: 
            advantages = self.compute_gae(torch.FloatTensor(rewards), values, next_values) 
            returns = advantages + values.detach() 
        else: 
            returns = self.compute_returns(rewards) 
            advantages = returns - values.detach() 
        
        # Normalize advantages 
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) 

        # Policy update 
        policy_loss = [] 
        for log_prob, advantage in zip(log_probs, advantages): 
            policy_loss.append(-log_prob * advantage) 
        policy_loss = torch.stack(policy_loss).sum() 

        self.policy_optimizer.zero_grad() 
        policy_loss.backward() 
        self.policy_optimizer.step() 

        # Value update 
        value_loss = (returns - values).pow(2).mean() 

        self.value_optimizer.zero_grad() 
        value_loss.backward() 
        self.value_optimizer.step() 

        avg_policy_loss = policy_loss.item() / step 
        avg_value_loss = value_loss.item() / step 
        return sum(rewards), avg_policy_loss, avg_value_loss 


    def train(self, max_episodes=1000, max_steps=2_000_000, log_interval=50):
        for episode in range(max_episodes): 
            total_reward, avg_policy_loss, avg_value_loss = self.train_episode() 
            if episode % log_interval == 0: 
                # print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Value Loss: {value_loss:.4f}")
                self.logger.info(f"Episode {episode:>4d} | Total Reward: {total_reward:>6.2f} | avg_policy_loss: {avg_policy_loss:>6.4f} | avg_value_loss: {avg_value_loss:>6.4f}")
            metrics = {
                "total_reward": total_reward, 
                "avg_policy_loss": avg_policy_loss,
                "avg_value_loss": avg_value_loss,   
            }
            self.logger.log_metrics(episode, metrics) 

            if self.timestamp > max_steps: 
                break 
        self.logger.close() 


if __name__ == "__main__": 
    env = gym.make("Pendulum-v1")
    agent = REINFORCE(env, 
                    #   input_dim=env.observation_space.shape[0], 
                      hidden_dims=[128, 128], 
                    #   output_dim=env.action_space.shape[0],
                      policy_lr=0.0005, 
                      value_lr=0.001, 
                      gamma=0.9, 
                      lambda_gae=0.95, 
                      gae=True, 
                      max_steps=200) 
    agent.train(max_episodes=2000, max_steps=2_000_000, log_interval=10) 


        
