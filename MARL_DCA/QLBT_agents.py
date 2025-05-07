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
        next_h = self.gru(obs, prev_h)  # parameter: input = obs, prev_h , output= next_h
        x = self.relu(self.fc1(next_h)) # parameter: input = next_h
        q = self.fc2(x)
        return q, next_h                #[q[transmit], q[wait], next_h]
    

class QLBTHyperNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QLBTHyperNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    # 그냥 선형 네트워크, 상태에 따라 가중치와 편향을 생성하는 네트워크

    def forward(self, state):
        return torch.abs(self.fc(state)) # 절대값을 취함으로써 가중치는 항상 양수로 유지


class QLBTMixingNetwork(nn.Module):
    '''
    입력: 각 에이전트의 Qi , 전역 상태 s
    출력: 전체 Qtot, [Qind_1, Qind_2, ..., Qind_n]
    구성:
    두 개의 2-layer fully-connected 네트워크:
        하나는 Q_tot 출력을 위한 mixing network
        하나는 Q_ind 출력을 위한 individual mixing network
    각 layer의 weight는 hypernetwork를 통해 전역 상태 s를 입력받아 생성됨
    weight는 non-negative constraint를 만족해야 하므로 abs() 사용
    '''

    def __init__(self, n_agents, state_dim, embed_dim=32):
        super(QLBTMixingNetwork, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        #Q total
        self.hyper_w1=QLBTHyperNetwork(state_dim, n_agents * embed_dim) #input으로 전역상태와 각 에이전트의 Qi
        self.hyper_b1=nn.Linear(state_dim, embed_dim)

        self.hyper_w2=QLBTHyperNetwork(state_dim, embed_dim)
        self.hyper_b2=nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

        #Q individual
        self.hyper_w1_ind = QLBTHyperNetwork(state_dim, n_agents*embed_dim)
        self.hyper_b1_ind = nn.Linear(state_dim, embed_dim)

        self.hyper_w2_ind = QLBTHyperNetwork(state_dim, embed_dim* n_agents)
        self.hyper_b2_ind = nn.Linear(state_dim, n_agents)

    def forward(self, agent_qs, state):
        """
        agent_qs = [bs, n_agents]
        state = [bs, state_dim]
        """

        bs=agent_qs.size(0) 

        #Q_tot
        w1 = self.hyper_w1(state).view(bs, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(bs, 1, self.embed_dim)
        hidden_tot = nn.functional.elu(torch.bmm(torch.tensor(agent_qs, dtype=torch.float32).unsqueeze(1), w1) + b1)  # (bs, 1, embed_dim)

        w2 = self.hyper_w2(state).view(bs, self.embed_dim, 1) ############ 1
        b2 = self.hyper_b2(state).view(bs, 1, 1)
        q_tot = torch.bmm(hidden_tot, w2) + b2  # (bs, 1, 1)
        q_tot = q_tot.view(-1, 1)

        # Q_ind
        w1_ind = self.hyper_w1_ind(state).view(bs, self.n_agents, self.embed_dim)
        b1_ind = self.hyper_b1_ind(state).view(bs, 1, self.embed_dim)
        hidden_ind = nn.functional.elu(torch.bmm(torch.tensor(agent_qs, dtype=torch.float32).unsqueeze(1), w1_ind)+b1_ind)

        w2_ind = self.hyper_w2_ind(state).view(bs, self.embed_dim, self.n_agents) ########## n_agents
        b2_ind = self.hyper_b2_ind(state).view(bs, 1, self.n_agents)
        q_ind = torch.bmm(hidden_ind, w2_ind) +b2_ind
        q_ind = q_ind.view(bs, self.n_agents)

        return q_tot, q_ind
    


''' 
QLBT_Agent: DE(decentralized execution)
    실제 env에서 env.step 호출시 사용
    매 step마다 observaton만 보고 action 선택(epsilon-greedy)
    학습 X, 단순 추론
'''
class QLBT_Agent:
    def __init__(self, obs_dim, action_dim=2, hidden_dim=32, device=None):

        self.device = device if device else "cpu"
        self.agent_net = QLBTAgentNetwork(obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(self.device)
        self.hidden_state = None

    def reset_hidden(self,bs=1):
        self.hidden_state = torch.zeros(bs, self.agent_net.gru.hidden_size).to(self.device)  # Initialize hidden state
        return self.hidden_state
    
    def select_action(self, obs, epsilon = 0.1):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, obs_dim)
        q_values, self.hidden_state = self.agent_net(obs, self.hidden_state)  # (1, action_dim)
        if random.random() < epsilon:
            action = random.randint(0, q_values.shape[1] - 1)
        else:
            action = torch.argmax(q_values, dim=1).item()
        return action
    
    def get_q_values(self, obs, hidden_state):
        obs = torch.tensor(obs, dtype = torch.float32).to(self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        q_values, next_hidden = self.agent_net(obs, hidden_state)
        return q_values, next_hidden
    



class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def push(self, transition):
        if len(self.buffer) == self.buffer_size:
            self.buffer.popleft()
        self.buffer.append(transition)
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    

'''
QLBT_AP:CT(centralized training)
    centralized trainer로 replay buffer 관리, mixing loss, optimizer step
    store_transition()
    -> train() 호출해서 학습
'''

class QLBT_AP:
    def __init__(self, n_agents, obs_dim, state_dim, hidden_dim = 32, 
                 buffer_size=10000, batch_size=32, gamma=0.99, lr=0.001, device=None):
        
        self.n_agents = n_agents
        self.gamma = gamma
        self.device = device if device else "cpu"
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        
       
        self.agent_nets = [QLBT_Agent(obs_dim, action_dim=2, hidden_dim=hidden_dim, device = self.device) for _ in range(n_agents)]
        self.target_agent_nets = [QLBT_Agent(obs_dim, action_dim=2, hidden_dim=hidden_dim, device = self.device) for _ in range(n_agents)]

        self.mixing_net = QLBTMixingNetwork(n_agents, state_dim, embed_dim=hidden_dim).to(self.device)
        self.target_mixing_net = QLBTMixingNetwork(n_agents, state_dim, embed_dim=hidden_dim).to(self.device)

        
        self.parameters = list(self.mixing_net.parameters())
        for agent in self.agent_nets:
            self.parameters += list(agent.agent_net.parameters())
        self.optimizer = optim.Adam(self.parameters, lr=lr)
        self.update_target_net()  # Initialize target networks with the same weights as the main networks



    def update_target_net(self):
        for target_net, net in zip(self.target_agent_nets, self.agent_nets):
            target_net.agent_net.load_state_dict(net.agent_net.state_dict())
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())

    def store_transition(self, transition):
        self.replay_buffer.push(transition)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        states, observations, hidden, actions, reward_tot, reward_ind, next_states, next_observations, hidden_next, dones = zip(*batch)

        s_batch = torch.stack(states).to(self.device)
        s_next_batch = torch.stack(next_states).to(self.device)
        r_tot = torch.tensor(reward_tot, dtype=torch.float32).unsqueeze(1).to(self.device)
        r_ind = torch.tensor(reward_ind, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device) 


        obs_batch = [torch.stack([obs[i] for obs in observations]).to(self.device) for i in range(self.n_agents)]
        obs_next_batch = [torch.stack([obs[i] for obs in next_observations]).to(self.device) for i in range(self.n_agents)]
        h_batch = [torch.stack([h[i] for h in hidden]).to(self.device) for i in range(self.n_agents)]
        h_next_batch = [torch.stack([h[i] for h in hidden_next]).to(self.device) for i in range(self.n_agents)]
        a_batch = [torch.tensor(a, dtype=torch.long).to(self.device) for a in zip(*actions)]
        

        # Compute Q_tot, Q_ind
        agent_qs = []
        for i in range(self.n_agents):
            q, _ = self.agent_nets[i].get_q_values(obs_batch[i], h_batch[i])
            q_i = q.gather(1, a_batch[i].unsqueeze(1).to(self.device))
            agent_qs.append(q_i)

        agent_qs_tensor = torch.cat(agent_qs, dim =1)
        q_tot, q_ind = self.mixing_net(agent_qs_tensor, s_batch)


        # Compute target Q_tot, target Q_ind
        with torch.no_grad():
            target_qs = []
            for i in range(self.n_agents):
                q, _ = self.target_agent_nets[i].get_q_values(obs_next_batch[i], h_next_batch[i])
                a_max =torch.argmax(q, dim=1, keepdim =True)
                q_i = q.gather(1, a_max)
                target_qs.append(q_i)
            target_qs_tensor = torch.cat(target_qs, dim=1)
            q_tot_target, q_ind_target = self.target_mixing_net(target_qs_tensor, s_next_batch)

        # Compute loss
        y_tot = r_tot + self.gamma * (1 - dones) * q_tot_target
        y_ind = r_ind + self.gamma * (1 - dones) * q_ind_target
        
        
        total_loss = nn.functional.mse_loss(q_tot, y_tot)
        individual_loss = nn.functional.mse_loss(q_ind, y_ind)
        loss = total_loss + individual_loss

        # optimizing step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item() #, total_loss.item(), individual_loss.item()


