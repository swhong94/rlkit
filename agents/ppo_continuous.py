import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from torch.distributions import Normal
from models.mlp import PolicyNet, ValueNet
from utils import Logger

class PPO:
    def __init__(self, env, hidden_layers=[64, 64], policy_lr=3e-4, value_lr=1e-3, gamma=0.99, 
                 gae_lambda=0.95, clip_ratio=0.2, epochs=10, batch_size=64, entropy_coeff=0.01, 
                 max_steps=200, log_dir="logs/ppo_continuous_logs", plot_window=100):
        self.env = env
        state_dim = env.observation_space.shape[0]  # 3 for Pendulum-v1
        action_dim = env.action_space.shape[0]     # 1 for Pendulum-v1

        # Use MLP from models.mlp
        self.actor = PolicyNet(state_dim, hidden_layers, action_dim, output_type="continuous")
        self.critic = ValueNet(state_dim, hidden_layers)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=policy_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=value_lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coeff = entropy_coeff
        self.max_steps = max_steps
        self.action_bounds = [torch.FloatTensor(env.action_space.low), 
                             torch.FloatTensor(env.action_space.high)]  # [-2, 2] for Pendulum

        self.logger = Logger(log_dir=log_dir, log_name_prefix="ppo_pendulum", plot_window=plot_window)

    def act(self, state):
        state_tensor = torch.FloatTensor(state)
        mu, std = self.actor(state_tensor)
        # Scale mu to action bounds [-2, 2]
        mu_scaled = torch.tanh(mu) * (self.action_bounds[1] - self.action_bounds[0]) / 2 + \
                    (self.action_bounds[1] + self.action_bounds[0]) / 2
        dist = Normal(mu_scaled, std)
        action = dist.sample()
        action_clamped = torch.clamp(action, self.action_bounds[0], self.action_bounds[1])
        log_prob = dist.log_prob(action_clamped)
        return action_clamped, log_prob, state_tensor

    def compute_gae(self, rewards, values, next_values, dones):
        deltas = rewards + self.gamma * (1 - dones) * next_values - values
        advantages = []
        advantage = 0.0
        for delta, done in zip(deltas.flip(0), dones.flip(0)):
            advantage = delta + self.gamma * self.gae_lambda * advantage * (1 - done)
            advantages.insert(0, advantage)
        return torch.FloatTensor(advantages)

    def train_episode(self):
        state, _ = self.env.reset()
        log_probs_old = []
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for _ in range(self.max_steps):
            action, log_prob, state_tensor = self.act(state)
            next_state, reward, done, truncated, _ = self.env.step([action.item()])  # Action as list

            log_probs_old.append(log_prob.detach())
            states.append(state_tensor)
            actions.append(action)
            rewards.append(reward)
            next_states.append(torch.FloatTensor(next_state))
            dones.append(float(done or truncated))

            state = next_state
            if done or truncated:
                break

        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.stack(next_states)
        dones = torch.FloatTensor(dones)
        log_probs_old = torch.stack(log_probs_old)

        # Normalize rewards 
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()
        advantages = self.compute_gae(rewards, values.detach(), next_values.detach(), dones)
        returns = advantages + values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = states.size(0)
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        updates = 0

        for _ in range(self.epochs):
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                batch_indices = indices[start:start + self.batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_log_probs_old = log_probs_old[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                mu, std = self.actor(batch_states)
                mu_scaled = torch.tanh(mu) * (self.action_bounds[1] - self.action_bounds[0]) / 2 + \
                            (self.action_bounds[1] + self.action_bounds[0]) / 2
                dist = Normal(mu_scaled, std)
                log_probs_new = dist.log_prob(batch_actions)  # batch_actions are clamped
                ratios = torch.exp(log_probs_new - batch_log_probs_old)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy = dist.entropy().mean()
                combined_policy_loss = policy_loss - self.entropy_coeff * entropy

                self.actor_optimizer.zero_grad()
                combined_policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 4.0)
                self.actor_optimizer.step()

                values = self.critic(batch_states).squeeze()
                value_loss = (values - batch_returns).pow(2).mean()

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 4.0)
                self.critic_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                updates += 1

        avg_policy_loss = total_policy_loss / updates
        avg_value_loss = total_value_loss / updates
        avg_entropy = total_entropy / updates
        total_reward = rewards.sum().item()

        print(f"Episode: {self.logger.logger.handlers[0].baseFilename.split('_')[-1].split('.')[0]}, "
              f"mu.mean(): {mu_scaled.mean().item():.4f}, std.mean(): {std.mean().item():.4f}, "
              f"action.mean(): {batch_actions.mean().item():.4f}, total_reward: {total_reward:.2f}")

        return total_reward, avg_policy_loss, avg_value_loss, avg_entropy

    def train(self, max_episodes=1000, log_interval=1):
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
                self.logger.info(f"Episode {episode} | Total Reward: {total_reward:>10.2f} | "
                                f"Avg Policy Loss: {avg_policy_loss:>10.4f} | Avg Value Loss: {avg_value_loss:>10.4f} | "
                                f"Avg Entropy: {avg_entropy:>10.4f}")
        self.logger.close()

if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    agent = PPO(
        env=env,
        hidden_layers=[128, 128],
        policy_lr=5e-4,
        value_lr=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        epochs=10,
        batch_size=64,
        entropy_coeff=0.0001,
        max_steps=1000,
        log_dir="logs/ppo_pendulum_logs",
        plot_window=100
    )

    agent.train(max_episodes=2000, log_interval=50)