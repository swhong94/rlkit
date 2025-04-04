import torch 
import torch.nn as nn 
import torch.optim as optim 
import gymnasium as gym 
import numpy as np 

from torch.distributions import Categorical 
from torch.optim import Adam 
from gymnasium.spaces import Discrete, Box 

from utils import Logger 
from models.mlp import PolicyNet, ValueNet 

# from models.mlp import MLP_Policy, MLP_Value  


class MLP_Policy(nn.Module): 
    def __init__(self, input_dim, hidden_dims, output_dim): 
        super(MLP_Policy, self).__init__() 
        layers = [] 
        prev_dim = input_dim 
        for h_dim in hidden_dims: 
            layers.append(nn.Linear(prev_dim, h_dim)) 
            layers.append(nn.ReLU()) 
            prev_dim = h_dim 
        self.mlp_block = nn.Sequential(*layers)  
        self.action_head = nn.Linear(hidden_dims[-1], output_dim) 

    def forward(self, x):  
        x = self.mlp_block(x)  
        logits = self.action_head(x) 
        return logits 
    
class MLP_Value(nn.Module): 
    def __init__(self, input_dim, hidden_dims): 
        super(MLP_Value, self).__init__()  
        layers = [] 
        prev_dim = input_dim 
        for h_dim in hidden_dims: 
            layers.append(nn.Linear(prev_dim, h_dim)) 
            layers.append(nn.ReLU()) 
            prev_dim = h_dim 
        layers.append(nn.Linear(prev_dim, 1)) 
        self.mlp_block = nn.Sequential(*layers) 
    
    
    def forward(self, x):  
        value = self.mlp_block(x) 
        return value 


class PPO:
    def __init__(self, 
                 env_name, 
                 hidden_dims, 
                 policy_lr=3e-4, 
                 value_lr=1e-3, 
                 gamma=0.99, 
                 gae_lambda=0.98, 
                 clip_ratio=0.2, 
                 epochs=10,
                 batch_size=64, 
                 max_steps=200, 
                 log_dir="logs/ppo_discrete_logs", 
                 plot_window=100, 
                 entropy_coeff=0.01, 
                 device="cpu", 
                 ):
        super(PPO, self).__init__() 

        # Environment informations 
        self.env = gym.make(env_name)
        self.env_name = env_name 
        self.log_prefix = "ppo_" + env_name.split("-")[0] 
        self.input_dim = self.env.observation_space.shape[0] 
        self.output_dim = self.env.action_space.n 
        self.action_type = "discrete" if isinstance(self.env.action_space, Discrete) else "continuous" 

        # self.actor = MLP_Policy(self.input_dim, hidden_dims, self.output_dim) 
        # self.critic = MLP_Value(self.input_dim, hidden_dims) 
        self.actor = PolicyNet(self.input_dim, hidden_dims, self.output_dim, output_type=self.action_type) 
        self.critic = ValueNet(self.input_dim, hidden_dims, activation="relu")

        # Optimizers  
        self.actor_optim = Adam(self.actor.parameters(), lr=policy_lr)  
        self.critic_optim = Adam(self.critic.parameters(), lr=value_lr)  

        # Training parameters 
        self.gamma = gamma 
        self.gae_lambda = gae_lambda 
        self.clip_ratio = clip_ratio 
        self.epochs = epochs 
        self.batch_size = batch_size 
        self.max_steps = max_steps 
        self.entropy_coeff = entropy_coeff 

        # Logger 
        self.logger = Logger(log_dir=log_dir, log_name_prefix=self.log_prefix, plot_window=plot_window)  
    
    def act(self, state): 
        state_tensor = torch.FloatTensor(state)
        logits = self.actor(state_tensor) 
        dist = Categorical(logits=logits)  
        action = dist.sample() 
        log_prob = dist.log_prob(action) 
        return action, log_prob, state_tensor 
    
    def compute_gae(self, rewards, values, next_values, dones): 
        td_errors = rewards + self.gamma * (1-dones) * next_values - values 
        advantages = [] 
        advantage = 0.0 
        for td_error, done in zip(td_errors.flip(0), dones.flip(0)): 
            advantage = td_error + self.gae_lambda * self.gamma * advantage * (1-done) 
            advantages.insert(0, advantage) 
        
        return torch.FloatTensor(advantages) 
    
    def train_episode(self): 
        state, _ = self.env.reset() 
        total_reward = 0 
        log_probs_old = [] 
        states, actions, next_states, rewards, dones = [], [], [], [], [] 
        values, next_values = [], [] 

        # Collect trajectories 
        for _ in range(self.max_steps): 
            action, log_prob, state_tensor = self.act(state) 
            next_state, reward, done, truncated, _ = self.env.step(action.item()) 

            states.append(state_tensor) 
            actions.append(action) 
            next_states.append(torch.FloatTensor(next_state)) 
            rewards.append(reward) 
            dones.append(float(done or truncated)) 
            log_probs_old.append(log_prob.detach()) 

            state = next_state 
            if done or truncated:  
                break 

        states = torch.stack(states)  
        actions = torch.stack(actions) 
        next_states = torch.stack(next_states) 
        rewards = torch.FloatTensor(rewards) 
        dones = torch.FloatTensor(dones) 
        log_probs_old = torch.stack(log_probs_old) 

        values = self.critic(states).squeeze() 
        next_values = self.critic(next_states).squeeze()  
        advantages = self.compute_gae(rewards, values.detach(), next_values.detach(), dones)  

        returns = advantages + values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  

        dataset_size = states.size(0) 
        total_policy_loss = 0 
        total_value_loss = 0 
        total_entropy = 0 
        update_count = 0

        for _ in range(self.epochs): 
            random_indices = torch.randperm(dataset_size)
            for i in range(0, dataset_size, self.batch_size): 
                batch_indices = random_indices[i:i+self.batch_size]  
                batch_states = states[batch_indices]  
                batch_actions = actions[batch_indices] 
                batch_returns = returns[batch_indices]  
                batch_advantages = advantages[batch_indices]  
                batch_log_probs_old = log_probs_old[batch_indices]

                # Compute new logits and values 
                logits = self.actor(batch_states) 
                dist = Categorical(logits=logits) 
                log_probs_new = dist.log_prob(batch_actions) 
                ratios = torch.exp(log_probs_new - batch_log_probs_old) 
                surr_obj1 = ratios * batch_advantages 
                surr_obj2 = torch.clamp(ratios, 1-self.clip_ratio, 1+self.clip_ratio) * batch_advantages 
                policy_loss = -torch.min(surr_obj1, surr_obj2).mean() 

                # Entropy bonus 
                entropy = dist.entropy().mean()


                # combined loss 
                loss = policy_loss - self.entropy_coeff * entropy  


                # Update 
                self.actor_optim.zero_grad() 
                loss.backward() 
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)  
                self.actor_optim.step() 

                # Value loss 
                values = self.critic(batch_states).squeeze() 
                value_loss = (batch_returns - values).pow(2).mean() 

                self.critic_optim.zero_grad() 
                value_loss.backward()  
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)   
                self.critic_optim.step()  

                total_policy_loss += policy_loss.item() 
                total_value_loss += value_loss.item() 
                total_entropy += entropy.item() 
                update_count += 1 

        avg_policy_loss = total_policy_loss / update_count 
        avg_value_loss = total_value_loss / update_count 
        avg_entropy = total_entropy / update_count 
        total_reward = rewards.sum().item() 

        return total_reward, avg_policy_loss, avg_value_loss, avg_entropy 



    def train(self, max_episodes=1000, log_interval=10): 
        for episode in range(max_episodes): 
            total_reward, avg_policy_loss, avg_value_loss, avg_entropy = self.train_episode() 
            metrics = {
                "total_reward": total_reward, 
                "avg_policy_loss": avg_policy_loss, 
                "avg_value_loss": avg_value_loss, 
                "avg_entropy": avg_entropy 
            }
            self.logger.log_metrics(episode, metrics) 

            if episode % log_interval == 0: 
                self.logger.info(f"Episode {episode} | Total Reward: {total_reward:>10.4f} | Avg Policy Loss: {avg_policy_loss:>10.4f} | Avg Value Loss: {avg_value_loss:>10.4f} | Avg Entropy: {avg_entropy:>10.4f}")
            
        self.logger.close() 
    
    def save(self, path): 
        pass 

    def load(self, path): 
        pass 

    def evaluate(self, num_episodes=10, max_steps=1000):
        env = gym.make(self.env_name, render_mode="human") 
        state, _ = env.reset() 
        total_rewards = [] 
        for ep in range(num_episodes): 
            episode_reward = 0 
            for _ in range(max_steps): 
                env.render() 
                action, log_prob, state_tensor = self.act(state) 
                next_state, reward, done, truncated, _ = env.step(action.item())
                state = next_state 
                episode_reward += reward 

                if done or truncated: 
                    env.reset() 
                    break 
            total_rewards.append(episode_reward) 
        env.close() 
        return np.mean(total_rewards) 






if __name__ == "__main__":  
    # env = gym.make("LunarLander-v3") 
    # env_name = "LunarLander-v3"
    env_name = "CartPole-v1"
    agent = PPO(
        env_name=env_name, 
        hidden_dims=[64, 64], 
        policy_lr=3e-4, 
        value_lr=1e-3, 
        gamma=0.99, 
        gae_lambda=0.98, 
        clip_ratio=0.2, 
        epochs=10,
        batch_size=64,
        max_steps=500, 
        log_dir="logs/ppo_discrete_logs", 
        plot_window=30,)
    
    agent.train(max_episodes=500, log_interval=50) 
    mean_reward = agent.evaluate(num_episodes=1, max_steps=500) 
    print(f"Mean Reward: {mean_reward:>10.4f}")



    