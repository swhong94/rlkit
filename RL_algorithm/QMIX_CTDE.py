import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from pettingzoo.mpe import simple_spread_v3


class AgentNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):  # input: obs_dim+action_dim, output: action_dim
        super(AgentNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim+action_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim) 
        self.q_out = nn.Linear(hidden_dim, action_dim)
        self.hidden_dim = hidden_dim

    # 위의 함수는 네트워크가 저렇게 생겼다는 거고 밑에 forward 함수에서 input과 output 정의

    def forward(self, obs, last_action, h_in):
        x = torch.cat([obs, last_action], dim=-1)
        x = nn.functional.relu(self.fc1(x))
        h = self.gru(x, h_in) # 기억 업데이트(partially observable)
        q = self.q_out(h)
        return q, h #각 에이전트의 Q-value와 hidden state(기억정보)를 반환

class HyperNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HyperNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    # 그냥 선형 네트워크, 상태에 따라 가중치와 편향을 생성하는 네트워크

    def forward(self, state):
        return torch.abs(self.fc(state)) # 절대값을 취함으로써 가중치는 항상 양수로 유지
    
class MixingNetwork(nn.Module):
    def __init__(self, n_agents, state_dim, hidden_dim=32): # num of agents' Q-values, global_state_dim
        super(MixingNetwork, self).__init__()
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim

        self.hyper_w1 = HyperNetwork(state_dim, n_agents * hidden_dim)
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)

        self.hyper_w2 = HyperNetwork(state_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) #Qtotal
        )

    def forward(self, agent_qs, global_state):  
        '''
        q_total = f(Q_1, Q_2, ..., Q_n; s)
        = relu(w1 @ Q_agents + b1) @ w2 + b2
        ''' 
        bs = agent_qs.size(0)  # batch size = 각 배치마다의 에이전트 Q 값 = (bs, n_agents)
        
        w1 = self.hyper_w1(global_state).view(bs, self.n_agents, self.hidden_dim) # .view -> reshape 함수
        b1 = self.hyper_b1(global_state).view(bs, 1, self.hidden_dim)
        hidden = nn.functional.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)  # hidden = aqent_qs @ w1 + b1

        w2 = self.hyper_w2(global_state).view(bs, self.hidden_dim, 1)
        b2 = self.hyper_b2(global_state).view(bs, 1, 1)
        q_total = torch.bmm(hidden, w2) + b2  # hidden @ w2 + b2

        return q_total.view(-1, 1)

#ReplayBuffer
Transition = namedtuple('Transition', ('obs', 'state', 'actions', 'rewards', 'next_obs', 'next_state', 'dones', 'last_actions'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)
    

# QMIX 전체 구성 클래스
class QMIX(nn.Module):
    def __init__(self, n_agents, obs_dim, state_dim, action_dim, hidden_dim=64, lr=0.001, gamma=0.99, device=None):
        super(QMIX, self).__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.gamma = gamma

        self.agent_qnets = nn.ModuleList([AgentNetwork(obs_dim, action_dim, hidden_dim) for _ in range(n_agents)])
        self.target_qnets = nn.ModuleList([AgentNetwork(obs_dim, action_dim, hidden_dim) for _ in range(n_agents)])
        self.mixing_net = MixingNetwork(n_agents, state_dim)
        self.target_mixing_net = MixingNetwork(n_agents, state_dim)

        self.device = device if device else 'cpu'
        self.to(self.device)

        self.update_target()

    def update_target(self):
        for i in range(self.n_agents):
            self.target_qnets[i].load_state_dict(self.agent_qnets[i].state_dict())
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())

    def forward_qs(self, obs, last_actions, h_states):
        q_vals, h_out = [], []
        for i in range(self.n_agents):
            q, h = self.agent_qnets[i](obs[:, i, :], last_actions[:, i, :], h_states[:, i, :])
            q_vals.append(q)
            h_out.append(h)
        
        return torch.stack(q_vals, dim=1), torch.stack(h_out, dim=1)
    
    def train(self, buffer, batch_size, optimizer):
        if len(buffer) < batch_size:
            return None
        # Sample a batch from the replay buffer
        batch = buffer.sample(batch_size)
        device = self.device

        obs = torch.tensor(batch.obs, dtype=torch.float32).to(device)  # (B, n_agents, obs_dim)
        state = torch.tensor(batch.state, dtype=torch.float32).to(device)  # (B, state_dim)
        actions = torch.tensor(batch.action, dtype=torch.long).to(device)  # (B, n_agents)
        rewards = torch.tensor(batch.reward, dtype=torch.float32).to(device).unsqueeze(-1)
        next_obs = torch.tensor(batch.next_obs, dtype=torch.float32).to(device)
        next_state = torch.tensor(batch.next_state, dtype=torch.float32).to(device)
        dones = torch.tensor(batch.done, dtype=torch.float32).to(device).unsqueeze(-1)
        last_actions = torch.tensor(batch.last_action, dtype=torch.float32).to(device)

        # Current Q-values
        h_states = [torch.zeros(batch_size, self.agent_qnets[i].idden_dim).to(device) for i in range(self.n_agents)]
        q_values, _ = self.forward_qs(obs, last_actions, h_states)
        chosen_qs = torch.gather(q_values, dim=2, index=actions.unsqueeze(-1)).squeeze(-1)
        q_tot = self.mixing_net(chosen_qs, state)

        # Target Q-values
        next_last_actions = nn.functional.one_hot(actions, num_classes=self.action_dim).float()
        h_states_target = [torch.zeros(batch_size, self.agent_net[i].hidden_dim).to(device) for i in range(self.n_agents)]

        next_q_vals, _ = self.forward_qs(next_obs, next_last_actions, h_states_target)
        next_actions = next_q_vals.argmax(dim=-1, keepdim=True)
        next_chosen_qs = torch.gather(next_q_vals, dim=2, index=next_actions).squeeze(-1)
        q_tot_target = self.target_mixing_net(next_chosen_qs, next_state)

        targets = rewards + self.gamma * (1 - dones) * q_tot_target
        loss = nn.functional.mse_loss(q_tot, targets.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()



if __name__ == "__main__":
    env = simple_spread_v3.env(render_mode = "human")
    env.reset(seed=42)

    n_agents = len(env.possible_agents)
    obs_dim=env.observation_space(env.possible_agents[0]).shape[0]
    state_dim = obs_dim * n_agents
    action_dim =env.action_space(env.possible_agents[0]).np_random

    qmix = QMIX(n_agents=n_agents, obs_dim=obs_dim, state_dim=state_dim, action_dim=action_dim)
    optimizer = optim.Adam(qmix.parameters(), lr=0.001)
    buffer = ReplayBuffer(capacity=10000)

    # Training parameters
    # max_steps = 500
    batch_size = 32
    epsilon = 0.1
    episodes = 1000

    for episode in episodes:
        env.reset(seed=42)
        observations = {agent: env.observe(agent) for agent in env.agents}
        dones = {agent: False for agent in env.agents}

        while not all(dones.values()): # 모든 에이전트가 종료될때까지 반복
            for agent in env.agent_iter():
                observation, reward, termination, truncation, info = env.last()
                action = None

                if not (termination or truncation):
                    obs_tensor = torch.tensor(observation, dtype =torch.float32).unsqueeze(0)
                    q_values, _ = qmix.forward_qs(obs_tensor, None, None)
                    action = q_values.argmax().item()
                else:
                    action = None
                env.step(action)

                next_observation, _, _, _, _ = env.last()
                buffer.add((observation,action,reward, next_observation,termination))
                observation = next_observation

                if len(buffer) >= batch_size:
                    loss = qmix.train(buffer, batch_size, optimizer)
                    print(f"Loss: {loss}")

                if termination:
                    break

                if episode % 100 == 0:
                    qmix.update_target()
                    print(f"Episode {episode}, Loss: {loss:.4f}")
        env.close()