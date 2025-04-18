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

t = args.t
cnt = args.cnt
s = args.s0             # Initial global state
tau = [args.tau0] * n   # Initial local observations
a = [args.a0] * n       # Initial actions
z = [args.z0] * n       # Transmission status
beta = [args.beta0] * n  # Weighting factors
theta = args.theta0     # Neural network parameters
theta_target = args.theta_target0  # Target network parameters


# Define neural network for agents
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


class ExperienceMemory: #Use EM at AP
    def __init__(self, capacity=memory_size):  # Initialize
        self.buffer = deque(maxlen=capacity)

    def push(self, s, tau, a, r, s_next, tau_next): # Store experience in memory
        if len(self.buffer) == self.buffer.maxlen:
            self.buffer.popleft()  
        self.buffer.append((s, tau, a, r, s_next, tau_next))

    def sample(self, bs=batch_size):  # Use batch_size and sample experiences
        if len(self.buffer) < bs:
            return None
        batch = random.sample(self.buffer, bs)  # Randomly sample batch_size experiences
        s_batch, tau_batch, a_batch, r_batch, s_next_batch, tau_next_batch = zip(*batch)
        return (torch.stack(s_batch),
                torch.stack(tau_batch),
                torch.tensor(a_batch, dtype=torch.int64),
                torch.tensor(r_batch, dtype=torch.float32),
                torch.stack(s_next_batch),
                torch.stack(tau_next_batch))

    def __len__(self):
        return len(self.buffer)
    
class AgentNetwork:
    def __init__(self, input_dim, output_dim, lr=lr, gamma=gamma, epsilon=epsilon, 
                    buffer_capacity=memory_size, tarnet_update_frequency=Nr, epsilon_min=0.01, epsilon_decay=0.995, device=None):
        self.device = device if device else 'cpu'
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tarnet_update_frequency = tarnet_update_frequency

        # Experience replay memory
        self.buffer = ExperienceMemory(buffer_capacity)

        # Define Q-network and target network
        self.q_net = MLP(input_dim, output_dim).to(self.device)
        self.target_net = MLP(input_dim, output_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Counter for target network updates
        self.update_counter = 0

    def select_action(self, tau):
        if random.random() < self.epsilon:  # Exploration
            return random.randint(0, self.q_net.fc[-1].out_features - 1)
        else:  # Exploitation
            with torch.no_grad():
                tau_tensor = torch.tensor(tau, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.q_net(tau_tensor)
                return q_values.argmax().item()

    def store_experience(self, s, tau, a, r, s_next, tau_next):
        self.buffer.push(s, tau, a, r, s_next, tau_next)

    def update(self, beta):
        if len(self.buffer) < batch_size:
            return None

        # Sample a batch of experiences
        batch = self.buffer.sample(batch_size)
        if batch is None:
            return None

        s_batch, tau_batch, a_batch, r_batch, s_next_batch, tau_next_batch = batch

        # Compute Q-values and target Q-values
        q_values = self.q_net(tau_batch).gather(1, a_batch.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q_values = self.target_net(tau_next_batch).max(1)[0]
            target_q_values = r_batch + self.gamma * next_q_values

        # Compute loss
        loss = nn.functional.mse_loss(q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.tarnet_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()
    


    




