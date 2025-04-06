import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

# Hyperparameters
gamma = 0.99          # Discount factor for future rewards
lr = 0.001            # Learning rate
num_episodes = 1000   # Number of episodes for training
entropy_weight = 0.01 # Entropy regularization to encourage exploration
batch_size = 5        # Number of steps before updating the model

# Define the neural network model for Actor and Critic
class A2CModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(A2CModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Actor output: Probabilities for each action
        self.actor = nn.Linear(128, output_dim)
        
        # Critic output: Value of the state
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc(x)
        action_probs = torch.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value


# Function to compute advantage
def compute_advantages(rewards, values, next_value, dones, gamma):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae * (1 - dones[t])
        advantages.insert(0, gae)
        next_value = values[t]
    return advantages


# A2C Training Loop
def train():
    env = gym.make('MountainCar-v0',render_mode = 'human')
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    
    model = A2CModel(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        total_reward = 0
        
        states = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        dones = []
        
        while not done:
            action_probs, value = model(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            next_state, reward, done, _, _ = env.step(action.item())
            next_state = torch.tensor(next_state, dtype=torch.float32)
            
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            dones.append(done)
            
            total_reward += reward
            state = next_state
            
            if done:
                next_value = 0 if done else model(state)[1].item()
                
                advantages = compute_advantages(rewards, values, next_value, dones, gamma)
                advantages = torch.tensor(advantages, dtype=torch.float32)
                
                returns = []
                for t in range(len(rewards)):
                    returns.append(advantages[t] + values[t].item())  # accumulate value
                
                # Convert lists to tensors
                states = torch.stack(states)
                actions = torch.stack(actions)
                log_probs = torch.stack(log_probs)
                returns = torch.tensor(returns, dtype=torch.float32)
                
                # Compute the loss
                action_probs, values = model(states)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                
                # Actor loss (policy gradient)
                actor_loss = -(new_log_probs * (returns - values.detach()).squeeze()).mean()
                
                # Critic loss (value function error)
                critic_loss = (returns - values.squeeze()).pow(2).mean()
                
                # Total loss
                total_loss = actor_loss + 0.5 * critic_loss - entropy_weight * entropy
                
                # Backpropagate and optimize
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # Clear the lists for the next episode
                states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
        
        print(f'Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}')

    env.close()

if __name__ == '__main__':
    train()
