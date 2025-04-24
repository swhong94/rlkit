import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class QLBTAgentNetwork(nn.Module):
    
    def __init__(self, obs_dim, action_dim=2, hidden_dim=32): #output_dim: {0: wait, 1: transmit}
        """
        QLBTAgentNetwork: GRU-based network for Q-learning with temporal observation.
        input_dim: dimension of the observation
        hidden_dim: dimension of the hidden state
        output_dim: dimension of the action
        """
        super(QLBTAgentNetwork, self).__init__()
        self.gru = nn.GRUCell(obs_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs, prev_h): #obs_seq: 과거 M개의 관측-행동 pair (tau)
        """
        obs: (batch_size, obs_dim) - 관측값 텐서
        prev_h: (batch_size, hidden_dim) - 이전 hidden state
        """
        next_h = self.gru(obs, prev_h)
        x = self.relu(self.fc1(next_h))
        q = self.fc2(x)
        return q, next_h
    

class QLBTHyperNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QLBTHyperNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    # 그냥 선형 네트워크, 상태에 따라 가중치와 편향을 생성하는 네트워크

    def forward(self, state):
        return torch.abs(self.fc(state)) # 절대값을 취함으로써 가중치는 항상 양수로 유지


class QLBTMixingNetwork(nn.Module):

    def __init__(self, n_agents, state_dim, embed_dim=32):
        super(QLBTMixingNetwork, self).__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim

        #Q total
        self.hyper_w1=QLBTHyperNetwork(state_dim, n_agents * embed_dim)
        self.hyper_b1=nn.Linear(state_dim, embed_dim)

        self.hyper_w2=QLBTHyperNetwork(state_dim, embed_dim)
        self.hyper_b2=nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

        #Q individual
        self.individual_layer = nn.Linear(embed_dim+state_dim, n_agents)

    def forward(self, agent_qs, global_state):
        """
        q_total = f(Q_1, Q_2, ..., Q_n; s)
        = relu(w1 @ Q_agents + b1) @ w2 + b2

        q_ind = f(Q_1, Q_2, ..., Q_n; s)
        """

        bs=agent_qs.size(0)
        # Compute hypernetwork weights and biases
        w1 = self.hyper_w1(global_state).view(bs, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(global_state).view(bs, 1, self.embed_dim)
        hidden = torch.bmm(agent_qs.unsqueeze(1), w1) + b1  # (bs, 1, embed_dim)
        hidden = nn.functional.elu(hidden)

        w2 = self.hyper_w2(global_state).view(bs, self.embed_dim, 1)
        b2 = self.hyper_b2(global_state).view(bs, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2  # (bs, 1, 1)
        q_tot = q_tot.view(-1, 1)

        # Qind pathway
        # ind_input = torch.cat([agent_qs, global_state], dim=-1)  # (bs, n_agents + state_dim)
        # q_ind = self.individual_layer(ind_input)  # (bs, n_agents)
        hidden_detached = hidden.detach().squeeze(1)
        q_ind = self.individual_layer(torch.cat([hidden_detached, global_state], dim=-1))

        return q_tot, q_ind
    


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, observation, action, reward, next_state, next_observation, done):
        if len(self.buffer) == self.capacity:
            self.buffer.popleft()
        if isinstance(observation, torch.Tensor) and observation.ndim == 1:
            observation = observation.unsqueeze(0)  # (1, obs_dim)
        if isinstance(next_observation, torch.Tensor) and next_observation.ndim == 1:
            next_observation = next_observation.unsqueeze(0)  # (1, obs_dim)
        self.buffer.append((state, observation, action, reward, next_state, next_observation, done))
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    


class QLBT_Agent:
    def __init__(self, obs_dim, action_dim=2, hidden_dim=32, device=None):

        self.device = device if device else "cpu"
        self.net = QLBTAgentNetwork(obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(self.device)
        self.hidden_state = None

    def init_hidden(self,bs=1):
        self.hidden_state = torch.zeros(bs, self.net.gru.hidden_size).to(self.device)  # Initialize hidden state
        return self.hidden_state
    
    def select_action(self, obs, epsilon=0.1):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, obs_dim)
        q_values, self.hidden_state = self.net(obs, self.hidden_state)  # (1, action_dim)
        if random.random() < epsilon:
            action = random.randint(0, q_values.shape[-1]-1)
        else:
            action = q_values.argmax().item()
        return action
    
    def get_q_values(self, obs, hidden_state):
        obs = obs.to(self.device)
        q_values, next_hidden_state = self.net(obs, hidden_state)
        return q_values, next_hidden_state
    


class QLBT_AP:
    def __init__(self, n_agents, obs_dim, state_dim, hidden_dim = 32, 
                 buffer_size=10000, batch_size=32, gamma=0.99, lr=0.001, device=None):
        
        self.n_agents = n_agents
        self.gamma = gamma
        self.device = device if device else "cpu"
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        self.batch_size = batch_size
        
       
        self.agent_nets = [QLBT_Agent(obs_dim, action_dim=2, hidden_dim=hidden_dim, device = self.device) for _ in range(n_agents)]
        self.target_agent_nets = [QLBT_Agent(obs_dim, action_dim=2, hidden_dim=hidden_dim, device = self.device) for _ in range(n_agents)]

        self.mixing_net = QLBTMixingNetwork(n_agents, state_dim, embed_dim=hidden_dim).to(self.device)
        self.target_mixing_net = QLBTMixingNetwork(n_agents, state_dim, embed_dim=hidden_dim).to(self.device)

        self.parameters= list(self.mixing_net.parameters())
        # for agent in self.agent_nets:
        #     for param in agent.net.parameters():
        #         self.parameters += list(param)
        self.optimizer = optim.Adam(self.parameters, lr=lr)
        self.update_target_networks()  # Initialize target networks with the same weights as the main networks



    def update_target_networks(self):
        for target_net, net in zip(self.target_agent_nets, self.agent_nets):
            target_net.net.load_state_dict(net.net.state_dict())
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())


    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        states, observations, actions, rewards, next_states, next_observations, dones = zip(*batch)
        print(f"observations[0].shape: {observations[0].shape}")

        states = torch.stack([s.unsqueeze(0) if s.dim() == 1 else s for s in states]).to(self.device)
        next_states = torch.stack([s.unsqueeze(0) if s.dim() == 1 else s for s in next_states]).to(self.device)
        actions = torch.tensor(actions).long().to(self.device)  # (B, n_agents)
        if actions.dim() ==1:
            actions=actions.unsqueeze(1)
        print(f"actions.space: {actions.shape}")

        rewards = torch.tensor(rewards).float().to(self.device)  # (B,)
        dones = torch.tensor(dones).float().to(self.device)  # (B,)

        # Compute Q_i
        agent_qs = []
        with torch.no_grad():
            target_qs = []
        
        for i in range(self.n_agents):
            obs_i = torch.stack([torch.tensor(obs[i], dtype =torch.float32) for obs in observations]).to(self.device)
            next_obs_i = torch.stack([torch.tensor(obs[i], dtype =torch.float32) for obs in next_observations]).to(self.device)
            
            print(f"obs_i.shape: {obs_i.shape}")  # (batch_size, 1)
            hidden_state = self.agent_nets[i].init_hidden(bs = self.batch_size)
            q_values, _ = self.agent_nets[i].get_q_values(obs_i, hidden_state)  # (B, action_dim)
            print(f"q_values.shape: {q_values.shape}")  # (batch_size, action_dim)
            agent_q = q_values.gather(1, actions[:, i].unsqueeze(1)).squeeze(1)  # (B,)
            agent_qs.append(agent_q)


            with torch.no_grad():
                target_q_values, _ = self.target_agent_nets[i].get_q_values(next_obs_i, hidden_state)
                target_agent_q = target_q_values.max(1)[0]
                target_qs.append(target_agent_q)
        
        agent_qs = torch.stack(agent_qs, dim=1)
        with torch.no_grad():
            target_qs = torch.stack(target_qs, dim=1)


        # Compute Q_tot and Q_ind
        q_tot, q_ind = self.mixing_net(agent_qs, states)
        with torch.no_grad():
            target_q_tot, target_q_ind = self.target_mixing_net(target_qs, next_states)

        # Compute loss
        total_target = rewards + self.gamma * (1 - dones) * target_q_tot.squeeze(1)
        total_loss = nn.functional.mse_loss(q_tot.squeeze(1), total_target)

        individual_target = []
        for i in range(self.n_agents):
            ind_target = rewards + self.gamma * (1 - dones) * target_q_ind[:, i]
            individual_target.append(ind_target)
        individual_target = torch.stack(individual_target, dim=1)

        individual_loss = nn.functional.mse_loss(q_ind, individual_target)
        loss = total_loss + individual_loss

        # optimizing step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), total_loss.item(), individual_loss.item()


