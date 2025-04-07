# QLBT Simulation Code (Simplified)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# ----- Hyperparameters -----
num_agents = 3
gamma = 0.99
epsilon = 0.1
learning_rate = 0.001
batch_size = 32
buffer_size = 10000
update_freq = 50
hidden_dim = 64
time_steps = 1000

# ----- Experience Replay Buffer -----
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# ----- Agent Network (GRU-based) -----
class AgentNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AgentNetwork, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)  # Wait or Transmit

    def forward(self, x):
        out, _ = self.gru(x)
        out = torch.relu(self.fc1(out[:, -1, :]))
        return self.fc2(out)

# ----- Mixing Network -----
class MixingNetwork(nn.Module):
    def __init__(self, num_agents, hidden_dim):
        super(MixingNetwork, self).__init__()
        self.fc1 = nn.Linear(num_agents, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, q_values):
        x = torch.elu(self.fc1(q_values))
        return self.fc2(x)

# ----- Environment (Simplified) -----
class DummyEnv:
    def __init__(self):
        self.state_dim = 10
        self.action_space = [0, 1]  # 0: Wait, 1: Transmit

    def reset(self):
        return np.zeros((num_agents, self.state_dim)), [0]*num_agents

    def step(self, actions):
        reward = [-1 if a == 1 else 0 for a in actions]
        next_state = np.random.rand(num_agents, self.state_dim)
        return next_state, reward

# ----- Initialize -----
agent_nets = [AgentNetwork(10, hidden_dim) for _ in range(num_agents)]
mixing_net = MixingNetwork(num_agents, hidden_dim)
replay_buffer = ReplayBuffer(buffer_size)
optimizers = [optim.Adam(agent.parameters(), lr=learning_rate) for agent in agent_nets]
mixing_optimizer = optim.Adam(mixing_net.parameters(), lr=learning_rate)
env = DummyEnv()

# ----- Training Loop -----
state, _ = env.reset()
tau = torch.zeros((num_agents, 1, 10))

for t in range(time_steps):
    actions = []
    q_values = []

    for i in range(num_agents):
        q = agent_nets[i](tau[i].unsqueeze(0))
        if random.random() < epsilon:
            action = random.choice([0, 1])
        else:
            action = q.argmax().item()
        actions.append(action)
        q_values.append(q[0, action])

    next_state, rewards = env.step(actions)
    next_tau = torch.tensor(next_state).float().unsqueeze(1)
    replay_buffer.push((tau.clone(), actions, rewards, next_tau))

    tau = next_tau.clone()

    if len(replay_buffer) >= batch_size:
        batch = replay_buffer.sample(batch_size)

        loss = 0
        for sample in batch:
            tau_b, actions_b, rewards_b, next_tau_b = sample
            q_vals = torch.stack([agent_nets[i](tau_b[i].unsqueeze(0))[0, actions_b[i]] for i in range(num_agents)])
            q_tot = mixing_net(q_vals.unsqueeze(0))
            target_q_vals = torch.stack([agent_nets[i](next_tau_b[i].unsqueeze(0)).max(1)[0] for i in range(num_agents)])
            target_q_tot = mixing_net(target_q_vals.unsqueeze(0))
            total_reward = sum(rewards_b)
            y_tot = total_reward + gamma * target_q_tot
            loss += (y_tot - q_tot).pow(2).mean()

        mixing_optimizer.zero_grad()
        for opt in optimizers:
            opt.zero_grad()
        loss.backward()
        mixing_optimizer.step()
        for opt in optimizers:
            opt.step()

    if t % 100 == 0:
        print(f"Step {t}, Loss: {loss.item():.4f}")
