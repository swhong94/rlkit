import torch
import random
from collections import deque
from MARL_DCA import QmixVariableSetup

import torch.nn as nn
import torch.optim as optim


args = QmixVariableSetup.parse_args()

# Initialize parameters from arguments
n = args.n
epsilon = args.epsilon
gamma = args.gamma
T = args.T
Nr = args.Nr
batch_size = args.batch_size
memory_size = args.memory_size
lr = args.lr
experience_memory = deque(maxlen=memory_size)

t = args.t
cnt = args.cnt
s = args.s0  # Initial global state
tau = [args.tau0] * n  # Initial local observations
a = [args.a0] * n  # Initial actions
z = [args.z0] * n  # Transmission status
beta = [args.beta0] * n  # Weighting factors
theta = args.theta0  # Neural network parameters
theta_target = args.theta_target0  # Target network parameters


# Define neural network for agents
class AgentNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AgentNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Initialize networks
agent_networks = [AgentNetwork(input_dim=10, output_dim=2) for _ in range(n)]
target_networks = [AgentNetwork(input_dim=10, output_dim=2) for _ in range(n)]
optimizers = [optim.Adam(agent.parameters(), lr=0.001) for agent in agent_networks]

# Copy parameters to target networks
for target, agent in zip(target_networks, agent_networks):
    target.load_state_dict(agent.state_dict())

# Training loop
while t < T:
    for i in range(n):
        # Compute tau[i][t] from tau[i][t-1], a[i][t-1], z[i][t-1]
        # (Placeholder logic, replace with actual computation)
        tau[i] = tau[i]

        if a[i] == "Transmit":
            if z[i] == 1:  # Transmission finished
                a[i] = "Wait"
            else:
                a[i] = "Transmit"
        else:
            if random.random() < 0.5:  # Simulate channel busy/idle
                a[i] = "Wait"
            else:
                # Input tau to agent network and output Q
                tau_tensor = torch.tensor(tau[i], dtype=torch.float32)
                Q = agent_networks[i](tau_tensor)

                # Generate action using epsilon-greedy policy
                if random.random() < epsilon:
                    a[i] = random.choice(["Wait", "Transmit"])
                else:
                    a[i] = "Wait" if Q[0] > Q[1] else "Transmit"

    # Simulate environment response
    s_next = None  # Placeholder for next global state
    r = None  # Placeholder for reward
    tau_next = tau  # Placeholder for next local observations

    # Store experience in memory
    experience_memory.append((s, tau, a, r, s_next, tau_next))

    # Sample a batch of experiences
    if len(experience_memory) >= batch_size:
        batch = random.sample(experience_memory, batch_size)

        # Compute loss and update networks
        for e in batch:
            s_batch, tau_batch, a_batch, r_batch, s_next_batch, tau_next_batch = e

            # Compute target values
            Q_next = [target(torch.tensor(tau_next_batch[i], dtype=torch.float32)).max().item()
                      for i, target in enumerate(target_networks)]
            y_tot = r_batch + gamma * sum(Q_next)

            # Compute individual losses
            losses = []
            for i in range(n):
                Q_pred = agent_networks[i](torch.tensor(tau_batch[i], dtype=torch.float32))
                y_ind = r_batch + gamma * Q_next[i]
                loss = (y_tot - Q_pred.sum()) ** 2 + beta[i] * (y_ind - Q_pred[a_batch[i]]) ** 2
                losses.append(loss)

            # Backpropagation
            for i in range(n):
                optimizers[i].zero_grad()
                losses[i].backward()
                optimizers[i].step()

    # Update target networks periodically
    if cnt % Nr == 0:
        for target, agent in zip(target_networks, agent_networks):
            target.load_state_dict(agent.state_dict())

    # Update counters and states
    cnt += 1
    s = s_next
    tau = tau_next
    t += 1