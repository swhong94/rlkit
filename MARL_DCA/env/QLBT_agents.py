import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class QLBTAgentNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=2):
        super(QLBTAgentNetwork, self).__init__()
        
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, obs_seq, hidden_state=None): #obs_seq: 과거 M개의 관측-행동 pair (tau)
        # obs_seq: (batch, seq_len, input_dim)
        # hidden_state: (1, batch, hidden_dim)        
        gru_out, hidden_state = self.gru(obs_seq, hidden_state)
        x = self.relu(self.fc1(gru_out[:,-1,:]))
        q = self.fc2(x)

        return q, hidden_state

class QLBTMixingNetwork(nn.Module):
    def __init__(self, n_agents, state_dim, embed_dim=32):
        super(QLBTMixingNetwork, self).__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim

        #Q total
        self.hyper_w1=nn.Sequential(
            nn.Linear(state_dim, embed_dim*n_agents),
            nn.Abs()
        )
        self.hyper_b1=nn.Linear(state_dim, embed_dim)

        self.hyper_w2=nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.Abs()
        )
        self.hyper_b2= nn.Linear(state_dim, 1)
        
        #Q individual
        self.individual_layer = nn.Sequential(
            nn.Linear(n_agents+state_dim, embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, n_agents)
        )

    def forward(self, agent_qs, state):
        """
        agent_qs: (batch, n_agents)
        state: (batch, state_dim)
        """

        bs=agent_qs.size(0)
        # Compute hypernetwork weights and biases
        w1 = self.hyper_w1(state).view(bs, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(bs, 1, self.embed_dim)
        hidden = torch.bmm(agent_qs.unsqueeze(1), w1) + b1  # (bs, 1, embed_dim)
        hidden = torch.elu(hidden)

        w2 = self.hyper_w2(state).view(bs, self.n_agents, 1)
        b2 = self.hyper_b2(state).view(bs, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2  # (bs, 1, 1)
        q_tot = q_tot.view(-1, 1)

        # Qind pathway
        ind_input = torch.cat([agent_qs, state], dim=-1)  # (bs, n_agents + state_dim)
        q_ind = self.individual_layer(ind_input)  # (bs, n_agents)

        return q_tot, q_ind
    

class QLBT_DQNAgent:
    def __init__(self, agent_id, obs_dim=2, action_dim=2, buffer_size=1000, batch_size=32, gamma=0.95, alpha=0.001, epsilon=0.1):
        self.agent_id = agent_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        ).to(self.device)

        self.target_net = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        ).to(self.device)

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=alpha)
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_freq = 50
        self.learn_step = 0
        self.prev_state = None
        self.prev_action = None

    def preprocess(self, obs):
        state = torch.tensor([obs["channel_state"], obs["collision"]], dtype=torch.float32)
        return state.to(self.device)  # shape: (1, 2)
    
    def act(self, obs):
        state = self.preprocess(obs)
        self.prev_state = state
        if torch.rand(1).item() < self.epsilon:
            action = torch.randint(0, 2, (1,)).item()
        else:
            with torch.no_grad():
                action = torch.argmax(self.q_net(state)).item()
        self.prev_action = action
        return action

    def store(self, obs_next, reward):
        next_state = self.preprocess(obs_next)
        self.memory.append((self.prev_state, self.prev_action, reward, next_state))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        
        
        states = torch.stack(states)                      # shape: (batch_size, 2)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1)  # (B, 1)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(1)
        next_states = torch.stack(next_states)  

        # print("▶️ DEBUG SHAPE")
        # print("states: ", states.shape)
        # print("actions: ", actions.shape)
        # print("q_net(states): ", self.q_net(states).shape)

        q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            q_next = self.target_net(next_states).max(1, keepdim=True)[0]
            target = rewards + self.gamma * q_next

        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())



