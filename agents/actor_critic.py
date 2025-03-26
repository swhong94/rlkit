import torch 
import torch.nn as nn 
import torch.optim as optim 
import numpy as np 
import gymnasium as gym 

from torch.distributions import Normal  
from models.mlp import MLP_Policy, MLP_Value 
from gymnasium.spaces import Box, Discrete 

from utils import Logger 


class ActorCritic(nn.Module): 
    """ 
    Actor-critic Agent 
    """
    def __init__(self, 
                 env, 
                 hidden_dims, 
                 activation="ReLU", 
                 policy_lr = 1e-4, 
                 value_lr = 1e-3, 
                 gamma = 0.99, 
                 max_steps = 200,
                 log_dir="logs/actor_critic_logs",
                 device="cpu",
                 plot_window=None):
        super(ActorCritic, self).__init__() 

        # Environment informations
        self.env = env 
        input_dim = env.observation_space.shape[0] 
        output_dim = env.action_space.shape[0] if isinstance(env.action_space, Box) else env.action_space.n 

        # Initialization (Actor and Critic) 
        self.actor = MLP_Policy(input_dim, hidden_dims, output_dim, activation) 
        self.critic = MLP_Value(input_dim, hidden_dims, activation) 

        # optimizers 
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=policy_lr) 
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=value_lr) 

        # Training parameters 
        self.gamma = gamma 
        self.timestep = 0 
        self.max_steps = max_steps 
        self.min_max = [torch.FloatTensor(self.env.action_space.low), torch.FloatTensor(self.env.action_space.high)]

        # Logger 
        self.logger = Logger(log_dir, log_name_prefix="actor_critic", plot_window=plot_window) 

        # Device 
        self.device = device 
        
    
    def act(self, state): 
        """
        Select an action from the policy 
        """
        state_tensor = torch.FloatTensor(state) 
        mu, sigma = self.actor(state_tensor) 
        mu_scaled = mu * self.min_max[1]                  # Scale to [action_low, action_high]
        dist = Normal(mu_scaled, sigma) 
        action = dist.sample() 
        action_clamped = torch.clamp(action, self.min_max[0], self.min_max[1])
        log_prob = dist.log_prob(action) 

        return action_clamped, log_prob, state_tensor 
    



    def train_step(self, state, action, reward, next_state, log_prob, done): 
        """ 
        Perform a single training step
        """
        self.timestep += 1 

        # compute state value 
        state_value = self.critic(state).squeeze() 
        next_state_tensor = torch.FloatTensor(next_state) 
        next_state_value = self.critic(next_state_tensor).squeeze()

        # Compute TD target 
        td_target = reward + self.gamma * (1 - done) * next_state_value 
        advantage = td_target - state_value 

        # Compute policy loss 
        actor_loss = -log_prob * advantage.detach() 

        # compute value loss 
        critic_loss = (td_target - state_value).pow(2).mean() 

        # update policy 
        self.actor_optimizer.zero_grad() 
        actor_loss.backward() 
        self.actor_optimizer.step()  

        # update value 
        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step() 

        return reward, actor_loss.item(), critic_loss.item()


    def train_episode(self,): 
        total_reward = 0 
        total_actor_loss = 0 
        total_critic_loss = 0 
        steps = 0

        state, _ = self.env.reset() 

        for t in range(self.max_steps): 
            action, log_prob, state_tensor = self.act(state) 
            next_state, reward, done, truncated, _ = self.env.step([action.item()]) 

            reward, actor_loss, critic_loss = self.train_step(state_tensor, action, reward, next_state, log_prob, done) 

            total_reward += reward 
            total_actor_loss += actor_loss 
            total_critic_loss += critic_loss 
            steps += 1 

            #  Update next_state             
            state = next_state 

            if done or truncated: 
                break 

        avg_policy_loss = total_actor_loss / steps 
        avg_critic_loss = total_critic_loss / steps 
        return total_reward, avg_policy_loss, avg_critic_loss 

    def train(self, max_episodes, max_timesteps, log_interval, save_interval=None):
        print(f"Training... {self.__class__.__name__}")
        self.max_timesteps = max_timesteps 
        for episode in range(max_episodes): 
            episode_reward, avg_policy_loss, avg_critic_loss  = self.train_episode() 
            if episode % log_interval == 0: 
                # print(f"Episode {episode:>4d} | Reward: {episode_reward:>10.2f} | Value Loss: {self.value_loss.item():>10.4f} | Policy Loss: {self.policy_loss.item():>10.4f}")
                self.logger.info(f"Episode {episode:>4d} | Reward: {episode_reward:>10.2f} | Policy Loss: {avg_policy_loss:>10.4f} | Critic Loss: {avg_critic_loss:>10.4f}")
            
            metrics = {
                "total_reward": episode_reward, 
                "avg_policy_loss": avg_policy_loss, 
                "avg_critic_loss": avg_critic_loss, 
            }
            self.logger.log_metrics(episode, metrics) 
        self.logger.close() 


    def save(self, path): 
        pass 

    def load(self, path): 
        pass 

    def evaluate(self, env, num_episodes): 
        pass 





if __name__ == "__main__": 
    env = gym.make("Pendulum-v1") 

    agent = ActorCritic(
        env = env, 
        hidden_dims = [128, 128], 
        policy_lr = 5e-6, 
        value_lr = 5e-5, 
        gamma = 0.9, 
        max_steps = 200,
        log_dir="logs/actor_critic_logs",
        device="cpu"
    )

    agent.train(max_episodes=2000, max_timesteps=200, log_interval=10) 
    env.close() 