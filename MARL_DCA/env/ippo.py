import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import gymnasium as gym
from torch.distributions import Categorical
from pettingzoo.mpe import simple_spread_v3
from logger import Logger  


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64, activation="ReLU"):
        super(PolicyNet, self).__init__()
        layers= []
        prev_dim = obs_dim
        for h_dim in hidden_dim:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(getattr(nn, activation)())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, act_dim))
        self.fc = nn.Sequential(*layers)

    def forward(self, state):
        logits = self.fc(state)
        return logits
    
class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64, activation="ReLU"):
        super(ValueNet, self).__init__()
        layers= []
        prev_dim = obs_dim
        for h_dim in hidden_dim:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(getattr(nn, activation)())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.fc = nn.Sequential(*layers)

    def forward(self, state):
        value = self.fc(state)
        return value
    

class IPPO(nn.Module):
    def __init__(self, 
                env, 
                hidden_dims, 
                policy_lr = 3e-4,
                value_lr = 1e-3,
                gamma = 0.99, 
                gae_lambda = 0.98, 
                clip_ratio = 0.2, 
                clip_grad = None, 
                epochs = 10,
                batch_size = 64, 
                max_steps = 200, 
                log_dir = "logs/ippo_discrete_logs",
                plot_window = 100, 
                entropy_coeff = 0.01,
                device = "cpu"
                ):
        super(IPPO, self).__init__()

        # Enviornment
        self.env = env
        env.reset()
        self.agents = env.agents
        self.device = torch.device(device)

        self.log_prefix = "ippo_" + "simple_spread"

        self.policies = {}
        self.values = {}
        self.policy_optimizers = {}
        self.value_optimizers = {}
        self.obs_spaces = {} # for dictionary observation spaces

        for agent in self.agents:
            obs_space = self.env.observation_space(agent)

            # Compute total input dimension from discrete action space 
            if isinstance(obs_space, gym.spaces.Dict):
                obs_dim = sum(space.n if isinstance(space, gym.spaces.Discrete) else space.shape[0] for space in obs_space.spaces.values())
            else:
                obs_dim = obs_space.n if isinstance(obs_space, gym.spaces.Discrete) else obs_space.shape[0]
            # obs_dim = sum(space.n if isinstance(space, gym.spaces.Discrete) else space.shape[0] for space in obs_space.spaces.values()) 
            act_dim = self.env.action_space(agent).n
            # obs_dim = self.env.observation_space(agent).shape[0]
            # assert isinstance(self.env.action_space(agent), gym.spaces.Discrete), "only supports discrete action space"
            # act_dim = self.env.action_space(agent).n

            self.policies[agent] = PolicyNet(obs_dim, act_dim, hidden_dims).to(self.device)
            self.values[agent] = ValueNet(obs_dim, hidden_dims).to(self.device)
            self.policy_optimizers[agent] = optim.Adam(self.policies[agent].parameters(), lr=policy_lr)
            self.value_optimizers[agent] = optim.Adam(self.values[agent].parameters(), lr=value_lr)
            self.obs_spaces[agent] = obs_space

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.clip_grad = clip_grad
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.entropy_coeff = entropy_coeff

        self.logger = Logger(log_dir = log_dir, log_name_prefix = self.log_prefix, plot_window = plot_window)

    def preprocess_observation(self, obs, agent):
        # Convert Dictionary observation to a flat tensor
        obs_space = self.obs_spaces[agent]

        if isinstance(obs, gym.spaces.Dict):
            one_hots = []
            for key, value in obs.items():
                if isinstance(obs_space.spaces[key], gym.spaces.Discrete):
                    n = obs_space.spaces[key].n
                    one_hot = torch.zeros(n, device = self.device)
                    one_hot[value] = 1.0
                    one_hots.append(one_hot)
                else:
                    one_hots.append(torch.FloatTensor([value]))
            return torch.cat(one_hots)
        elif isinstance(obs, np.ndarray):
            return torch.FloatTensor(obs).to(self.device)
        else:
            raise TypeError(f"Unsupported observation type: {type(obs)}")
    

    def act(self, state, policy):
        state_tensor = torch.FloatTensor(state).to(self.device)
        logits = policy(state_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, state_tensor
    

    def compute_gae(self, rewards, values, next_values, dones):
        td_errors = rewards + self.gamma * next_values * (1 - dones) - values
        # Compute GAE
        gae = 0.0
        advantages = []
        for td_error, done in zip(td_errors.flip(0), dones.flip(0)):
            gae = td_error + self.gae_lambda * self.gamma * gae * (1 - done)
            advantages.insert(0,gae)
        
        return torch.FloatTensor(advantages)
    

    def train_episode(self):
        observations, _ = self.env.reset()
        trajectories = {agent: {'states':[],
                                'actions':[],
                                'log_probs':[],
                                'rewards':[],
                                'values':[],
                                'dones':[],
                                'next_states':[]} for agent in self.agents}
        total_rewards = {agent: 0 for agent in self.agents}

        # Collect trajectories
        for _ in range(self.max_steps):
            actions = {}

            for agent in self.agents:
                state_tensor = self.preprocess_observation(observations[agent], agent)
                action, log_prob, _ = self.act(observations[agent], self.policies[agent])
                actions[agent] = action.item()
                trajectories[agent]['states'].append(state_tensor)
                trajectories[agent]['log_probs'].append(log_prob.detach())
                trajectories[agent]['actions'].append(action)
                trajectories[agent]['values'].append(self.values[agent](state_tensor).detach())
            
            next_observations, rewards, dones, truncs, infos = self.env.step(actions)
            for agent in self.agents:
                trajectories[agent]['rewards'].append(rewards[agent])
                trajectories[agent]['dones'].append(float(dones[agent]) or float(truncs[agent]))
                next_state_tensor = self.preprocess_observation(next_observations[agent], agent)
                trajectories[agent]['next_states'].append(torch.FloatTensor(next_state_tensor))   
                total_rewards[agent] += rewards[agent]
            
            observations = next_observations
            if all(dones.values()) or all(truncs.values()):
                break

        #Train each agent independently
        metrics = {}

        for agent in self.agents:
            tau_a = trajectories[agent]
            states = torch.stack(tau_a['states'])
            actions = torch.stack(tau_a['actions'])
            log_probs_old = torch.stack(tau_a['log_probs'])
            rewards = torch.FloatTensor(tau_a['rewards']).to(self.device)
            dones = torch.FloatTensor(tau_a['dones']).to(self.device)
            next_states = torch.stack(tau_a['next_states'])
            values = self.values[agent](states).squeeze()
            next_values = self.values[agent](next_states).squeeze()

            # Compute GAE
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
                    batch_log_probs_old = log_probs_old[batch_indices]
                    batch_advantages = advantages[batch_indices]
                    batch_returns = returns[batch_indices]

                    # Compute new log probabilities
                    logits = self.policies[agent](batch_states)
                    dist = Categorical(logits=logits)
                    log_probs_new = dist.log_prob(batch_actions)

                    # Compute ratio
                    ratio = torch.exp(log_probs_new - batch_log_probs_old)

                    # Compute surrogate loss
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Compute entropy loss
                    entropy_loss = dist.entropy().mean()
                    
                    # Compute total loss
                    loss = policy_loss - self.entropy_coeff * entropy_loss
                    
                    # Update policy network
                    self.policy_optimizers[agent].zero_grad()
                    loss.backward()
                    
                    if self.clip_grad is not None:
                        torch.nn.utils.clip_grad_norm_(self.policies[agent].parameters(), max_norm = self.clip_grad)
                    self.policy_optimizers[agent].step()

                    # Compute value loss
                    values = self.values[agent](batch_states).squeeze()
                    value_loss = F.mse_loss(values, batch_returns)
                    # Update value network
                    self.value_optimizers[agent].zero_grad()
                    value_loss.backward()
                    if self.clip_grad is not None:
                        torch.nn.utils.clip_grad_norm_(self.values[agent].parameters(), max_norm = self.clip_grad)
                    self.value_optimizers[agent].step()

                    # Accumulate losses
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy_loss.item()
                    update_count += 1
                
            # Compute average losses
            avg_policy_loss = total_policy_loss / update_count
            avg_value_loss = total_value_loss / update_count
            avg_entropy = total_entropy / update_count
            metrics[agent] = {
                'avg_policy_loss': avg_policy_loss,
                'avg_value_loss': avg_value_loss,
                'avg_entropy': avg_entropy,
                'total_reward': total_rewards[agent]
            } 
        return metrics
    
    def train(self, max_episodes=1000, log_interval=10):
        for episode in range(max_episodes):
            agent_metrics = self.train_episode()
            avg_reward = np.mean([metrics['total_reward'] for metrics in agent_metrics.values()])
            avg_policy_loss = np.mean([metrics['avg_policy_loss'] for metrics in agent_metrics.values()])
            avg_value_loss = np.mean([metrics['avg_value_loss'] for metrics in agent_metrics.values()])
            avg_entropy = np.mean([metrics['avg_entropy'] for metrics in agent_metrics.values()])
            metrics = {
                'avg_reward': avg_reward,
                'avg_policy_loss': avg_policy_loss,
                'avg_value_loss': avg_value_loss,
                'avg_entropy': avg_entropy
            }
            for agent in self.agents:
                metrics[f"{agent}_avg_policy_loss"] = agent_metrics[agent]['avg_policy_loss']
                metrics[f"{agent}_avg_value_loss"] = agent_metrics[agent]['avg_value_loss']
                metrics[f"{agent}_avg_entropy"] = agent_metrics[agent]['avg_entropy']
                metrics[f"{agent}_total_reward"] = agent_metrics[agent]['total_reward']
            self.logger.log_metrics(metrics, episode)

            if episode % log_interval == 0:
                self.logger.info(f"Episode {episode} | Avg Reward: {avg_reward:>10.4f}")

        self.logger.close()
        
    def save(self, path):
        pass

    def load(self, path):
        pass

    def evaluate(self, num_episodes=10, max_steps = 1000):
        env = self.env
        total_rewards = []
        for _ in range(num_episodes):
            observations, _ = env.reset()
            episode_reward = {agent: 0 for agent in self.agents}
            for _ in range(max_steps):
                actions = {agent: self.act(observations[agent], self.policies[agent])[0].item() for agent in self.agents}
                next_obs, rewards, terminated, truncated, infos = self.env.step(actions)
                env.render()
                for agent in self.agents:
                    episode_reward[agent] += rewards[agent]
                if all(terminated.values()) or all(truncated.values()):
                    break
            total_rewards.append(np.mean(list(episode_reward.values())))
        env.close()
        return np.mean(total_rewards), np.mean(env.render_data["delay"])


if __name__ == "__main__":
    env = simple_spread_v3.parallel_env(N=3, max_cycles=200, continuous_actions = False)
    hidden_dims = [128] * 2
    ippo = IPPO(env=env, hidden_dims=hidden_dims,
                policy_lr = 0.0005,
                value_lr = 0.005,
                gamma = 0.99, 
                gae_lambda = 0.9, 
                clip_ratio = 0.2, 
                clip_grad = 10.0, 
                epochs = 45,
                batch_size = 64, 
                max_steps = 200, 
                log_dir = "logs/ippo_simple_spread_logs",
                plot_window = 30, 
                entropy_coeff = 0.00,
                device = "cpu"
                )
    ippo.train(max_episodes = 500, log_interval = 10)