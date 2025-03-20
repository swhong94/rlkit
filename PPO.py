import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value

def compute_advantages(rewards, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    next_value = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        next_value = values[t]
    returns = [a + v for a, v in zip(advantages, values)]
    return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

def ppo_update(policy, optimizer, states, actions, old_log_probs, returns, advantages, clip_epsilon=0.2, epochs=10):
    for _ in range(epochs):
        action_probs, state_values = policy(states)
        dist = Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        ratio = (new_log_probs - old_log_probs).exp()
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()
        value_loss = nn.MSELoss()(state_values.squeeze(), returns)
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def train_ppo(env, policy, optimizer, num_episodes=1000, gamma=0.99, lam=0.95, clip_epsilon=0.2, batch_size=64, epochs=10):
    for episode in range(num_episodes):
        state = env.reset()
        log_probs, states, actions, rewards, values = [], [], [], [], []
        done = False
        
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_probs, state_value = policy(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            
            next_state, reward, done, _ = env.step(action.item())
            
            log_probs.append(dist.log_prob(action))
            states.append(state_tensor)
            actions.append(action)
            rewards.append(reward)
            values.append(state_value.item())
            
            state = next_state
        
        advantages, returns = compute_advantages(rewards, values, gamma, lam)
        
        ppo_update(policy, optimizer, torch.stack(states), torch.tensor(actions), torch.tensor(log_probs), returns, advantages, clip_epsilon, epochs)
        
        if episode % 100 == 0:
            print(f"Episode {episode}: Reward = {sum(rewards)}")

# Example Usage
# env = gym.make("CartPole-v1")
# policy = ActorCritic(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
# optimizer = optim.Adam(policy.parameters(), lr=3e-4)
# train_ppo(env, policy, optimizer)
