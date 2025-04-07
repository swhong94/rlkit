import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import matplotlib.pyplot as plt

# Hyperparameters
gamma = 0.99          # Discount factor for future rewards
lr = 0.001            # Learning rate
num_episodes = 100    # Number of episodes for training
entropy_weight = 0.01 # Entropy regularization to encourage exploration
batch_size = 5        # Number of steps before updating the model

# Define the neural network model for Actor and Critic
class A2CModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(A2CModel, self).__init__()
        # Define the shared layers for both Actor and Critic
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
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t] # dt
        gae = delta + gamma * gae * (1 - dones[t]) # dt + gamma * dt+1 + ... + gamma^n * dt+n = Agae
        advantages.insert(0, gae)
        next_value = values[t]
    return advantages


# A2C Training Loop
def train():
    env = gym.make('MountainCar-v0')
    input_dim = env.observation_space.shape[0]  # State space dimension
    output_dim = env.action_space.n             # Action space dimension
    
    model = A2CModel(input_dim, output_dim)     #agent
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for episode in range(num_episodes): 
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32) # state는 항상 tensor로 변환)
        done = False
        total_reward = 0
        
        states = []     # St
        actions = []    # At
        log_probs = []  # log(π(At|St))
        rewards = []    # Rt
        values = []     # V(St)
        dones = []      # Done flag
        
        while not done:
            action_probs, value = model(state)
            dist = Categorical(action_probs) 
            action = dist.sample() #(1)
            log_prob = dist.log_prob(action) #(1)
            
            next_state, reward, done, _, _ = env.step(action.item()) # action.item()은 tensor를 int로 변환
            next_state = torch.tensor(next_state, dtype=torch.float32) #action에 의해 state가 한 개로 정해짐
            
            states.append(state)        # [s1, s2, s3, ...]
            actions.append(action)      # [a1, a2, a3, ...]
            log_probs.append(log_prob)  # [log(π(a1|s1)), log(π(a2|s2)), log(π(a3|s3)), ...]
            rewards.append(reward)      # [r1, r2, r3, ...]
            values.append(value)        # [V(s1), V(s2), V(s3), ...]
            dones.append(done)          # [0, 0, 0, ..., 1]
            
            total_reward += reward
            state = next_state
            
            if done: # policy update
                next_value = 0 if done else model(state)[1].item()
                
                advantages = compute_advantages(rewards, values, next_value, dones, gamma)
                advantages = torch.tensor(advantages, dtype=torch.float32)
                # 다 끝나면 advantages를 계산
                
                returns = []
                for t in range(len(rewards)): 
                    returns.append(advantages[t] + values[t].item())  # [Q1, Q2, Q3, ...]
                
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

    plt.plot(total_reward, label="total reward")
    plt.plot(total_loss, label="total loss")
    plt.xlabel("Episode")
    plt.ylabel("total reward")
    plt.title("A2C")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train()
