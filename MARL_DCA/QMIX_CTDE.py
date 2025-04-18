import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

'''
QMIX: A Deep Reinforcement Learning Algorithm for Multi-Agent Systems
- Mixing Network: Monotonic Q_total(개별 Q값 증가 -> 전체 Q값 증가)
- Q_total: Q-value for the entire team
- Q_individual: Q-value for each agent
QMIX는 협력형 멀티 에이전트 알고리즘
학습: state+joint action,
실행: local observation+action
exploration: epsilon-greedy
'''

# 개별 에이전트의 Q-network
class AgentQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):  # state_dim -> MARL:obs_dim
        super(AgentQNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, obs):
        return self.net(obs)  # Q-values for all actions

# Mixing Network (Monotonic Q_total)
class MixingNetwork(nn.Module):
    def __init__(self, n_agents, state_dim, hidden_dim=32): # num of agents, global_state_dim
        super(MixingNetwork, self).__init__()
        self.hyper_w1 = nn.Linear(state_dim, n_agents * hidden_dim)
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)

        self.hyper_w2 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Linear(state_dim, 1) # output: Q_total

        self.n_agents = n_agents
        self.hidden_dim = hidden_dim

    def forward(self, agent_qs, global_state):  
        '''
        q_total = f(Q_1, Q_2, ..., Q_n; s)
        = relu(w1 @ Q_agents + b1) @ w2 + b2
        ''' 
        B = agent_qs.size(0)  # batch size = 샘플링 된 32개의 transition 각각에 대해 생성한 q값        agents_qs = (B, n_agents)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)  # (B, 1, n_agents)

        w1 = torch.abs(self.hyper_w1(global_state)).view(-1, self.n_agents, self.hidden_dim)
        b1 = self.hyper_b1(global_state).view(-1, 1, self.hidden_dim)

        h = torch.bmm(agent_qs, w1) + b1  # (B, 1, hidden_dim)
        h = torch.relu(h)

        w2 = torch.abs(self.hyper_w2(global_state)).view(-1, self.hidden_dim, 1)
        b2 = self.hyper_b2(global_state).view(-1, 1, 1)

        q_total = torch.bmm(h, w2) + b2  # (B, 1, 1)
        return q_total.view(-1, 1)

# QMIX 전체 구성 클래스
class QMIX:
    def __init__(self, n_agents, obs_dim, action_dim, state_dim, gamma=0.99, lr=0.001):
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.gamma = gamma

        self.agent_qnets = [AgentQNetwork(obs_dim, action_dim) for _ in range(n_agents)]
        self.target_qnets = [AgentQNetwork(obs_dim, action_dim) for _ in range(n_agents)]
        self.mixer = MixingNetwork(n_agents, state_dim)
        self.target_mixer = MixingNetwork(n_agents, state_dim)

        self.agent_qnets = nn.ModuleList(self.agent_qnets)
        self.target_qnets = nn.ModuleList(self.target_qnets)

        self.optim = optim.Adam(list(self.agent_qnets.parameters()) + list(self.mixer.parameters()), lr=lr)

    def train(self, batch):
        # batch: dict with keys 'obs', 'state', 'actions', 'rewards', 'next_obs', 'next_state', 'dones'
        obs, state, actions, rewards, next_obs, next_state, dones = \
            batch['obs'], batch['state'], batch['actions'], batch['rewards'], batch['next_obs'], batch['next_state'], batch['dones']

        # Q-values from each agent
        agent_qs = []
        next_agent_qs = []

        for i in range(self.n_agents):
            q = self.agent_qnets[i](obs[:, i, :])
            q = q.gather(1, actions[:, i].unsqueeze(1))
            agent_qs.append(q)

            next_q = self.target_qnets[i](next_obs[:, i, :])
            next_q = next_q.max(dim=1, keepdim=True)[0]
            next_agent_qs.append(next_q)

        agent_qs = torch.cat(agent_qs, dim=1)  # (B, n_agents)
        next_agent_qs = torch.cat(next_agent_qs, dim=1)

        q_total = self.mixer(agent_qs, state)
        with torch.no_grad():
            target_q_total = self.target_mixer(next_agent_qs, next_state)
            targets = rewards + self.gamma * (1 - dones) * target_q_total

        loss = nn.MSELoss()(q_total, targets)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item()

    def update_target_network(self):
        for i in range(self.n_agents):
            self.target_qnets[i].load_state_dict(self.agent_qnets[i].state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
