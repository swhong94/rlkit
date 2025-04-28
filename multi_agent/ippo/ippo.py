import torch 
import torch.nn as nn 
import torch.optim as optim 
import gymnasium as gym 
import numpy as np 

from pettingzoo.mpe import simple_spread_v3
from torch.distributions import Categorical 
from torch.optim import Adam 
from gymnasium.spaces import Discrete 
from utils import Logger 
from .network import PolicyNet, ValueNet 


class IPPO(nn.Module):
    def __init__(self, 
                 env,
                 hidden_dims, 
                 policy_lr=3e-4, 
                 value_lr=1e-3, 
                 gamma=0.99, 
                 gae_lambda=0.98, 
                 clip_ratio=0.2, 
                 clip_grad=None,
                 epochs=10,
                 batch_size=64, 
                 max_steps=200, 
                 log_dir="logs/ippo_discrete_logs", 
                 plot_window=100, 
                 entropy_coeff=0.01, 
                 device="cpu", 
                 ):
        super(IPPO, self).__init__() 

        # Environment informations 
        self.env = env
        # Have to reset first to get agents 
        env.reset() 
        self.agents = env.agents    # List of agents 
        self.device = torch.device(device) 
        
        self.log_prefix = "ippo_" + "simple_spread"


        # Initialize agents 
        self.policies = {}; self.values = {}; self.policy_optims = {}; self.value_optims = {} 


        for agent in self.agents: 
            print("Observation space: ", self.env.observation_space(agent))
            obs_dim = self.env.observation_space(agent).shape[0] 
            assert isinstance(self.env.action_space(agent), Discrete), "Only discrete action spaces are supported for now (Box will be added later)"
            act_dim = self.env.action_space(agent).n 

            self.policies[agent] = PolicyNet(obs_dim, hidden_dims, act_dim).to(self.device) 
            self.values[agent] = ValueNet(obs_dim, hidden_dims).to(self.device) 
            self.policy_optims[agent] = Adam(self.policies[agent].parameters(), lr=policy_lr) 
            self.value_optims[agent] = Adam(self.values[agent].parameters(), lr=value_lr) 

        # Training parameters 
        self.gamma = gamma 
        self.gae_lambda = gae_lambda 
        self.clip_ratio = clip_ratio 
        self.epochs = epochs 
        self.batch_size = batch_size 
        self.max_steps = max_steps 
        self.entropy_coeff = entropy_coeff 
        self.clip_grad = clip_grad

        # Logger 
        self.logger = Logger(log_dir=log_dir, log_name_prefix=self.log_prefix, plot_window=plot_window)  
    
    def act(self, state, policy): 
        state_tensor = torch.FloatTensor(state).to(self.device) 
        logits = policy(state_tensor) 
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
        observations, _ = self.env.reset() 
        trajectories = {agent: {'states': [], 
                                'actions': [], 
                                'log_probs': [], 
                                'rewards': [], 
                                'values': [], 
                                'dones': [], 
                                'next_states': []} for agent in self.agents}
        total_rewards = {agent: 0 for agent in self.agents} 
        

        # Collect trajectories 
        for _ in range(self.max_steps): 
            actions = {} 

            for agent in self.agents: 
                action, log_prob, state_tensor = self.act(observations[agent], self.policies[agent]) 
                actions[agent] = action.item() 
                trajectories[agent]['states'].append(state_tensor) 
                trajectories[agent]['actions'].append(action) 
                trajectories[agent]['log_probs'].append(log_prob.detach()) 
                trajectories[agent]['values'].append(self.values[agent](state_tensor).detach())
            
            next_observations, rewards, dones, truncs, infos = self.env.step(actions) 
            for agent in self.agents: 
                trajectories[agent]['rewards'].append(rewards[agent]) 
                trajectories[agent]['dones'].append(float(dones[agent]) or float(truncs[agent]))
                trajectories[agent]['next_states'].append(torch.FloatTensor(next_observations[agent]).to(self.device))
                total_rewards[agent] += rewards[agent]
            
            observations = next_observations 
            if all(dones.values()) or all(truncs.values()): 
                break 
        
        # Train each agent independently 
        metrics = {} 

        for agent in self.agents: 
            tau_a = trajectories[agent]  
            states = torch.stack(tau_a['states'])
            actions = torch.stack(tau_a['actions']) 
            rewards = torch.FloatTensor(tau_a['rewards']).to(self.device) 
            next_states = torch.stack(tau_a['next_states'])
            dones = torch.FloatTensor(tau_a['dones']).to(self.device) 
            log_probs_old = torch.stack(tau_a['log_probs'])
            values = self.values[agent](states).squeeze() 
            next_values = self.values[agent](next_states).squeeze() 

            # Compute advantages (per agent) 
            advantages = self.compute_gae(rewards, values.detach(), next_values.detach(), dones)
            returns = advantages + values.detach() 
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            dataset_size = states.size(0) 
            total_policy_loss, total_value_loss, total_entropy, update_count = 0, 0, 0, 0 

            for _ in range(self.epochs): 
                random_indices = torch.randperm(dataset_size) 
                for i in range(0, dataset_size, self.batch_size): 
                    batch_indices = random_indices[i:i+self.batch_size] 
                    batch_states = states[batch_indices]
                    batch_actions = actions[batch_indices] 
                    batch_returns = returns[batch_indices] 
                    batch_advantages = advantages[batch_indices] 
                    batch_log_probs_old = log_probs_old[batch_indices] 

                    # Update policy (per agent) 
                    logits = self.policies[agent](batch_states) 
                    dist = Categorical(logits=logits) 
                    log_probs_new = dist.log_prob(batch_actions) 
                    ratios = torch.exp(log_probs_new - batch_log_probs_old) 
                    surr1 = ratios * batch_advantages 
                    surr2 = torch.clamp(ratios, 1-self.clip_ratio, 1+self.clip_ratio) * batch_advantages 
                    policy_loss = -torch.min(surr1, surr2).mean() 
                    entropy = dist.entropy().mean() 
                    loss = policy_loss - self.entropy_coeff * entropy 

                    self.policy_optims[agent].zero_grad() 
                    loss.backward() 
                    if self.clip_grad is not None: 
                        torch.nn.utils.clip_grad_norm_(self.policies[agent].parameters(), max_norm=self.clip_grad) 
                    self.policy_optims[agent].step() 

                    # Value update 
                    values = self.values[agent](batch_states).squeeze() 
                    value_loss = nn.MSELoss()(values, batch_returns) 

                    self.value_optims[agent].zero_grad()
                    value_loss.backward() 
                    if self.clip_grad is not None: 
                        torch.nn.utils.clip_grad_norm_(self.values[agent].parameters(), max_norm=self.clip_grad)
                    self.value_optims[agent].step() 

                    total_policy_loss += policy_loss.item() 
                    total_value_loss += value_loss.item() 
                    total_entropy += entropy.item() 
                    update_count += 1 

            avg_policy_loss = total_policy_loss / update_count 
            avg_value_loss = total_value_loss / update_count 
            avg_entropy = total_entropy / update_count 
            metrics[agent] = {
                "total_reward": total_rewards[agent], 
                "avg_policy_loss": avg_policy_loss, 
                "avg_value_loss": avg_value_loss, 
                "avg_entropy": avg_entropy
            }
        
        return metrics 
                 
                    


    def train(self, max_episodes=1000, log_interval=10): 
        for episode in range(max_episodes): 
            agent_metrics = self.train_episode() 
            avg_reward = np.mean([m["total_reward"] for m in agent_metrics.values()])
            avg_policy_loss = np.mean([m["avg_policy_loss"] for m in agent_metrics.values()])
            avg_value_loss = np.mean([m["avg_value_loss"] for m in agent_metrics.values()])
            avg_entropy = np.mean([m["avg_entropy"] for m in agent_metrics.values()])   
            metrics = {
                "avg_reward": avg_reward, 
                "avg_policy_loss": avg_policy_loss, 
                "avg_value_loss": avg_value_loss, 
                "avg_entropy": avg_entropy, 
            }
            for agent in self.agents: 
                metrics[f"{agent}_avg_reward"] = agent_metrics[agent]['total_reward'] 
                metrics[f"{agent}_avg_policy_loss"] = agent_metrics[agent]['avg_policy_loss'] 
                metrics[f"{agent}_avg_value_loss"] = agent_metrics[agent]['avg_value_loss'] 
                metrics[f"{agent}_avg_entropy"] = agent_metrics[agent]['avg_entropy'] 
            self.logger.log_metrics(episode, metrics) 

            if episode % log_interval == 0: 
                self.logger.info(f"Episode {episode} | Avg Reward: {avg_reward:>10.4f}")
            
        self.logger.close() 

    def save(self, path): 
        pass 

    def load(self, path): 
        pass 

    def evaluate(self, num_episodes=10, max_steps=1000):
        env = simple_spread_v3.parallel_env(N=3, max_cycles=25, continuous_actions=False, render_mode="human")
        total_rewards = []
        for _ in range(num_episodes):
            observations, _ = env.reset()
            episode_rewards = {agent: 0 for agent in self.agents}
            for _ in range(max_steps):
                actions = {agent: self.act(observations[agent], self.policies[agent])[0].item()
                        for agent in self.agents}
                observations, rewards, terminations, truncations, _ = self.env.step(actions)
                for agent in self.agents:
                    episode_rewards[agent] += rewards[agent]
                if all(terminations.values()) or all(truncations.values()):
                    break
            total_rewards.append(np.mean(list(episode_rewards.values())))
        env.close()
        return np.mean(total_rewards)





if __name__ == "__main__":  
    # env = gym.make("LunarLander-v3") 
    env = simple_spread_v3.parallel_env(N=3, max_cycles=200, continuous_actions=False,) 
    hidden_dims = [128] * 2 
    ippo = IPPO(
        env=env, 
        hidden_dims=hidden_dims, 
        policy_lr=0.0005, 
        value_lr=0.005, 
        gamma=0.99, 
        gae_lambda=0.9, 
        clip_ratio=0.2, 
        clip_grad=10.0,
        epochs=45,
        batch_size=64, 
        max_steps=200, 
        log_dir="logs/ippo_simple_spread_logs", 
        entropy_coeff=0.00,
        device='cpu',
        plot_window=30,
    )
    ippo.train(max_episodes=500, log_interval=10) 
    # rewards = ippo.evaluate(num_episodes=1, max_steps=200)
    # print("Average reward: ", rewards)