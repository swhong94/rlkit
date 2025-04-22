import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from pettingzoo.butterfly import pistonball_v6
import supersuit as ss
'''
QMIX: A Deep Reinforcement Learning Algorithm for Multi-Agent Systems
- Mixing Network: Monotonic Q_total(개별 Q값 증가 -> 전체 Q값 증가)
- Q_total: Q-value for the entire team
- Q_individual: Q-value for each agent
QMIX는 협력형 멀티 에이전트 알고리즘
학습: state+joint action,
실행: local observation+action
exploration: epsilon-greedy


- AgnetNet : DRQN: 각 에이전트의 개별 Q-value를 계산하는 네트워크
- MixingNet : 개별 Q-value를 결합하여 joint Qtot-value를 계산하는 네트워크
- HyperNetwork : MixingNet의 가중치와 편향을 상태(global state)에 따라 생성하는 네트워크
'''

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
        h_states = [torch.zeros(batch_size, self.agent_net[i].hidden_dim).to(device) for i in range(self.n_agents)]
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


# def train_qmix():
#     env = pistonball_v6.parallel_env()
#     env = ss.color_reduction_v0(env, mode='B')
#     env = ss.resize_v1(env, 84, 84)
#     env = ss.frame_stack_v1(env, 4)
#     env.reset()
    
#     agents = env.agents
#     n_agents = len(agents)
#     obs_shape = env.observation_space(agents[0]).shape
#     obs_dim = int(np.prod(obs_shape))
#     state_dim = obs_dim * n_agents
#     action_dim = env.action_space(agents[0]).shape[0]

#     qmix = QMIX(n_agents=n_agents, obs_dim=obs_dim, state_dim=state_dim, action_dim=action_dim)
#     optimizer = optim.Adam(qmix.parameters(), lr=0.001)
#     buffer = ReplayBuffer(capacity=10000)


#     max_step = 500
#     batch_size = 32
#     epsilon = 0.1

#     for episode in range(1000):
#         obs_dict = env.reset()
        
#         if isinstance(obs_dict, tuple):
#             obs_dict= obs_dict[0]
#         done_dict = {agent: False for agent in agents}

#         last_action_onehot = {
#             agent: np.zeros(action_dim, dtype=np.float32) for agent in agents
#         }

#         h_states = {
#             agent: torch.zeros(1, qmix.agent_qnets[0].hidden_dim) for agent in agents
#         }
        
#         for step in range(max_step):
#             obs_array = np.array([obs_dict[agent].flatten() for agent in agents])
#             obs_tensor = torch.tensor(obs_array, dtype=torch.float32)
#             last_action_array = np.stack([last_action_onehot[a] for a in agents])
#             last_action_tensor = torch.tensor(last_action_array, dtype=torch.float32)

#             actions = []
#             onehots = []
#             with torch.no_grad():
#                 for i, agent in enumerate(agents):
#                     q_values, h = qmix.agent_qnets[i](obs_tensor[i].unsqueeze(0), last_action_tensor[i].unsqueeze(0), h_states[agent])
#                     h_states[agent] = h
#                     if np.random.rand() < epsilon:
#                         a = np.random.uniform(env.action_space(agent).low, env.action_space(agent).high)
#                     else:
#                         # 정책 네트워크에서 연속적인 값을 출력하도록 수정
#                         a = q_values.squeeze().detach().numpy()  # 연속적인 값으로 변환
#                     actions.append(a)
#                     onehot = np.zeros(action_dim, dtype=np.float32)
#                     #onehot[int(a)] = 1.0
#                     onehots.append(onehot)
#                     last_action_onehot[agent] = onehot
            
#             action_dict = {agent: actions[i] for i, agent in enumerate(agents)}
#             next_obs_dict, reward_dict, terminated, truncated, _ = env.step(action_dict)
#             done_dict = {agent: terminated[agent] or truncated[agent] for agent in agents}

#             next_obs_array = np.stack([next_obs_dict[agent].flatten() for agent in agents])
#             reward = np.mean([reward_dict[a] for a in agents])
#             done = any(done_dict.values())
#             joint_obs = obs_array
#             joint_next_obs = next_obs_array

#             buffer.push(
#                 joint_obs, joint_obs.flatten(),
#                 actions, reward,
#                 joint_next_obs, joint_next_obs.flatten(),
#                 float(done),
#                 last_action_array
#             )

#             obs_dict = next_obs_dict

#             if done:
#                 break
#         # Train the QMIX model
#         loss = qmix.train(buffer, batch_size=batch_size, optimizer=optimizer)
        
#         if episode % 100 == 0:
#             qmix.update_target()
#             print(f"Episode {episode}, Loss: {loss:.4f}")

# if __name__ == "__main__":
#     train_qmix()