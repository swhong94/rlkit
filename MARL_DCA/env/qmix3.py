import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque

import gymnasium as gym
from pettingzoo.mpe import simple_spread_v3
from pettingzoo.utils.conversions import aec_to_parallel
from logger import Logger

'''
QMIX
Lqmix: E(Sigma i=1^bs)[(yi_tot - Qi_tot(τ, u, s; θ))^2] 
yi_tot = r +γ maxu′ Qtot(τ ′, u′, s′; θ−)
Qtot(τ, u, s; θ) = f(Q1(τ1, u1, s; θ), Q2(τ2, u2, s; θ), ..., Qn(τn, un, s; θ); s)
f: mixing network
'''

# 1. Q individual
# 2. graph -> episode 0부터
# 3. soft update

class AgentNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(AgentNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)  # 히스토리 저장
        self.q_out = nn.Linear(hidden_dim, action_dim)
        # obs_dim + action+dim -> hidden_dim -> gru -> hidden_dim -> action_dim 

    def forward(self, obs, last_action, his_in):
        x = torch.cat([obs, last_action], dim=-1)
        x = F.relu(self.fc1(x))
        if his_in is None:
            his_in = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        elif his_in.dim() == 1:
            his_in = his_in.unsqueeze(0)
        elif his_in.dim() == 3:
            his_in = his_in.squeeze(1)
        his_out = self.gru(x, his_in) # f.relu() 할지 말지 
        q = self.q_out(his_out)
        return q, his_out # q + his_out


class HyperNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HyperNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )
    def forward(self, state):
        return torch.abs(self.fc(state))


class MixingNetwork(nn.Module):
    def __init__(self, n_agents, state_dim, hidden_dim=64):

        super(MixingNetwork, self).__init__()
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim

        self.hyper_w1 = HyperNetwork(state_dim, n_agents * hidden_dim)
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)

        self.hyper_w2 = HyperNetwork(state_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Q_tot
        )

    def forward(self, agents_q, state): 
        # agents_q : [q1, q2, ... , qn]
        # state : [state_dim] = obs_dim * n_agents

        agents_q = agents_q.unsqueeze(0) if agents_q.dim() == 1 else agents_q   # [1, n_agents]
        state = state.unsqueeze(0) if state.dim() ==1 else state                # [1, state_dim]

        w1 = self.hyper_w1(state).view(-1, self.n_agents, self.hidden_dim)  # W1 = [1, n_agents, hidden_dim]
        b1 = self.hyper_b1(state).view(-1, 1, self.hidden_dim)              # b1 = [1, 1, hidden_dim]
        
        hidden = F.elu(torch.bmm(agents_q.unsqueeze(1), w1) + b1).squeeze(1)             
        # hidden = [1, 1, n_agents] * [1, n_agents, hidden_dim] + [1, 1, hidden_dim]
        # hidden = [1, 1, hidden_dim(뒤에 2, 3차원이 곱해짐 앞에 batch_제외)] + [1, 1, hidden_dim] -> [1, hidden_dim]

        w2 = self.hyper_w2(state).view(-1, self.hidden_dim, 1)              # W2 = [1, hidden_dim, 1]
        b2 = self.hyper_b2(state)                                           # b2 = [1, 1] -> [1, 1]

        q_total = torch.bmm(hidden.unsqueeze(1), w2).squeeze(1) + b2
        # q_total = [1, 1, hidden_dim] * [1, hidden_dim, 1] + [1, 1]
        # q_total = [1, 1] + [1, 1] -> [1, 1]
        return q_total.squeeze()      # [1]
    

class ReplayBufferRNN: 
    '''
    state = joint_obs,
    joint_action,                
    reward,                        
    next_state = next_joint_obs, 
    joint_hidden_state,          
    joint_done        
    .detach()를 통해 gradient를 끊어, backpropagation을 하지 않음 -> 메모리 누수나 에러 방지            
    '''
    def __init__(self, capacity=10000, device="cpu"):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, hidden_seq, state_seq, action_seq, reward_seq, next_state_seq, dones):
        if len(self.buffer) == self.buffer.maxlen:
            self.buffer.popleft()
        data = (hidden_seq.detach(), 
                state_seq.detach(), 
                action_seq.detach(), 
                reward_seq.detach(), 
                next_state_seq.detach(), 
                dones.detach())
        self.buffer.append(data)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        h_lst, s_lst, a_lst, r_lst, ns_lst, dn_lst = zip(*batch)
        '''
        B = batch_size, T = max_step, N = n_agents, H_dim = hidden_dim, obs_dim = observation_dim
        '''
        h_tensor = torch.stack(h_lst).to(self.device)     # (B, T+1, N, H_dim) 
        s_tensor = torch.stack(s_lst).to(self.device)     # (B, T, N, obs_dim)
        a_tensor = torch.stack(a_lst).long().unsqueeze(-1).to(self.device)      # action.shape = (B, T, N) -> (B, T, N, 1)
        r_tensor = torch.stack(r_lst).unsqueeze(-1).to(self.device)             # reward.shape = (B, T) -> (B, T, 1), q_tot
        ns_tensor = torch.stack(ns_lst).to(self.device)   # (B, T, N, obs_dim)
        d_tensor = torch.stack(dn_lst).unsqueeze(-1).to(self.device)            # dones.shape = (B, T, N) -> (B, T, N, 1)

        return h_tensor, s_tensor, a_tensor, r_tensor, ns_tensor, d_tensor

    def __len__(self):
        return len(self.buffer)


def td_lambda_target(rewards, target_qs, gamma=0.95, td_lambda=0.8):
    if rewards.dim() > 2:
        rewards = rewards.squeeze(-1)
    if target_qs.dim() > 2:
        target_qs = target_qs.squeeze(-1)

    B, T = rewards.shape
    targets = torch.zeros_like(rewards).to(rewards.device)
    targets[:, -1] = target_qs[:, -1]
    for t in reversed(range(T - 1)):
        targets[:, t] = rewards[:, t] + gamma * (
            td_lambda * targets[:, t + 1] + (1 - td_lambda) * target_qs[:, t + 1]
        ) # td_lambda * target_qs[:, t + 1] + (1 - td_lambda) * targets[:, t + 1]
    return targets



class QMIX(nn.Module):
    def __init__(self,
                env,
                hidden_dims, 
                batch_size = 64, 
                buffer_capacity = 10000, 
                lr=0.0003, 
                gamma=0.95, 
                epochs = 10,
                max_steps = 200,
                log_dir = "logs/qmix_simple_spread_logs",
                plot_window = 100,
                clip_grad = None,
                update_interval=100, 
                device="cpu",
                tau=None
                ):
        super(QMIX, self).__init__()
        
        # Environment
        self.env = env
        self.env.reset()
        for landmark in env.aec_env.unwrapped.world.landmarks:
            landmark.state.p_vel = np.zeros(2)
            landmark.movable = False
            landmark.collide = False
        self.agents = env.agents
        self.n_agents = len(self.agents) # N
        self.device = torch.device(device)
        self.buffer = ReplayBufferRNN(buffer_capacity, device =self.device)

        self.log_prefix = "qmix_" + "simple_spread"


        self.agent_nets = nn.ModuleDict()
        self.target_agent_nets = nn.ModuleDict()
        self.obs_spaces = {}

        for agent in self.agents:
            obs_space = env.observation_space(agent)
            
            if isinstance(obs_space, gym.spaces.Dict):
                obs_dim = sum(space.n if isinstance(space, gym.spaces.Discrete) else space.shape[0] for space in obs_space.spaces.values())
            else:
                obs_dim = obs_space.n if isinstance(obs_space, gym.spaces.Discrete) else obs_space.shape[0]
            
            act_dim = self.env.action_space(agent).n

            self.agent_nets[agent] = AgentNetwork(obs_dim, act_dim, hidden_dims).to(self.device)
            self.target_agent_nets[agent] = AgentNetwork(obs_dim, act_dim, hidden_dims).to(self.device)
            self.obs_spaces[agent] = obs_space
        
        ''' agent_nets의 네트워크 파라미터도 optimizer에 포함시켜야 함'''
        agent_params = []
        for agent in self.agent_nets.values():
            agent_params += list(agent.parameters())

        self.mixing_net = MixingNetwork(self.n_agents, obs_dim * self.n_agents, hidden_dims).to(self.device)
        self.target_mixing_net = MixingNetwork(self.n_agents, obs_dim * self.n_agents, hidden_dims).to(self.device)

        self.optimizer = optim.Adam(agent_params+list(self.mixing_net.parameters()), lr=lr,amsgrad=True)

        self.batch_size = batch_size # B
        self.gamma = gamma
        self.update_interval = update_interval
        self.epochs = epochs
        self.max_steps = max_steps # T -> history 저장 길이..
        self.clip_grad = clip_grad
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.decay_ratio = 0.995
        self.step = 0
        self.tau = tau

        self.logger = Logger(log_dir, self.log_prefix, plot_window)

        self.update_target(tau=None)

    def preprocess_observation(self, obs, agent):
        # Convert Dictionary observation to a flat tensor
        obs_space = self.obs_spaces[agent]

        if isinstance(obs_space, gym.spaces.Dict):
            one_hots = []
            for key, value in obs.items():
                if isinstance(obs_space.spaces[key], gym.spaces.Discrete):
                    n = obs_space.spaces[key].n
                    one_hot = torch.zeros(n, device = self.device)
                    one_hot[value] = 1.0
                    one_hots.append(one_hot)
                else:
                    one_hots.append(torch.FloatTensor([value]))
            obs_tensor = torch.cat(one_hots)
        elif isinstance(obs, np.ndarray):
            obs_tensor = torch.FloatTensor(obs).to(self.device)
        else:
            raise TypeError(f"Unsupported observation type: {type(obs)}")
        return obs_tensor.unsqueeze(0)
    
    ### 3. soft update
    def update_target(self, tau=None):  # tau=None이면 hard update
        if tau is None:
            for agent in self.agents:
                self.target_agent_nets[agent].load_state_dict(self.agent_nets[agent].state_dict())
            self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())
        else:
            for agent in self.agents:
                for target_param, param in zip(self.target_agent_nets[agent].parameters(), self.agent_nets[agent].parameters()):
                    target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            for target_param, param in zip(self.target_mixing_net.parameters(), self.mixing_net.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


    def epsilon_decay(self):
        self.epsilon_start = max(self.epsilon_end, self.epsilon_start * self.decay_ratio)  #1, 0.995, 0.990, 0.985 ... 0.01
        return self.epsilon_start 


    def select_action(self, agent, obs, last_action, his_in):

        obs_tensor = torch.FloatTensor(obs).to(self.device) # (1, obs_dim)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        
        last_action_tensor = torch.FloatTensor(last_action).to(self.device) 
        if last_action_tensor.dim() == 1:
            last_action_tensor = last_action_tensor.unsqueeze(0)
        
        h_in = torch.FloatTensor(his_in).to(self.device) # (1, H_dim)
        if h_in.dim() == 1:
            h_in = h_in.unsqueeze(0)
        #print(f"[DEBUG] last_action.shape = {last_action.shape}")
        
        q_values, h_out = self.agent_nets[agent](obs_tensor, last_action_tensor, h_in)
        #print(f"[DEBUG] q_values.shape = {q_values.shape}")
        q_values = q_values.squeeze(0) if q_values.dim() > 1 and q_values.size(0) == 1 else q_values
            
        h_out = h_out.squeeze(0) # (H_dim)

        if random.random() < self.epsilon_start:
            action = random.randint(0, self.env.action_space(agent).n - 1)
        else:
            action = q_values.argmax().item()

        '''선택한 액션의 Q값만 추출, agent마다, step마다 다름'''
        # q_selected = q_values[action].item()  # scalar: float

        return action, h_out #, q_selected

        
    def update(self):
        ### 2. graph -> episode 0부터
        if len(self.buffer) < self.batch_size:
            return 0.0

        # 샘플링: (B, T+1, N, H), (B, T, N, obs), ...
        hidden_seq, state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        #print(f"[DEBUG] state.shape = {state.shape}")
        B, T, N, obs_dim = state.shape
        agent_qs, target_qs = [], []

        for i, agent in enumerate(self.agents):
            a_i = action[:, :, i]                      # (B, T, N)
            s_i = state[:, :, i, :]                    # (B, T, N, obs)
            ns_i = next_state[:, :, i, :]              # (B, T, N, obs)

            q_seq, tq_seq = [], []
            for t in range(T):
                h_i = hidden_seq[:, t, i, :] if t < hidden_seq.size(1) - 1 else hidden_seq[:, -1, i, :]
                a_onehot = F.one_hot(a_i[:, t], num_classes=self.env.action_space(agent).n).float()

                if a_onehot.dim() != s_i[:,t].dim():
                    a_onehot = a_onehot.view(s_i[:,t].size(0), -1)

                q_t, _ = self.agent_nets[agent](s_i[:, t], a_onehot, h_i)
                q_selected = q_t.gather(1, a_i[:, t].view(-1,1)).squeeze(-1)
                q_seq.append(q_selected)

                with torch.no_grad():
                    q_next = self.agent_nets[agent](ns_i[:, t], a_onehot, h_i)[0]
                    next_action = q_next.argmax(dim=1, keepdim=True)
                    q_target, _ = self.target_agent_nets[agent](ns_i[:, t], a_onehot, h_i)
                    tq = q_target.gather(1, next_action).squeeze(-1)
                    tq_seq.append(tq)

            agent_qs.append(torch.stack(q_seq, dim=1))     # (B, T)
            target_qs.append(torch.stack(tq_seq, dim=1))   # (B, T)

        # (B, T, N)
        agent_qs = torch.stack(agent_qs, dim=2)
        target_qs = torch.stack(target_qs, dim=2)

        '''Mixing Network'''
        # (B, T, global_obs)
        state = state.view(B, T, -1)
        next_state = next_state.view(B, T, -1)

        q_total = torch.stack([self.mixing_net(agent_qs[:,t], state[:,t]) for t in range(T)], dim=1).squeeze(-1)     
        tq_total = torch.stack([self.target_mixing_net(target_qs[:,t], next_state[:,t]) for t in range(T)], dim=1).squeeze(-1)   

        # reward sum across agents (optional: per-agent reward instead)
        r_total = reward.sum(dim=2).squeeze(-1)                    # (B, T)
        y_total = td_lambda_target(r_total, tq_total, gamma=self.gamma, td_lambda=0.8)

        # loss and optimization
        loss = F.mse_loss(q_total, y_total.detach())
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad is not None:
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.clip_grad)
        self.optimizer.step()

        self.epsilon_decay()
        self.step += 1
        self.update_target()
        
        # Per-agent metrics
        per_agent_qs = agent_qs.detach().mean(dim=(0, 1))   # agent_qs.shape: (B, T, N)
        avg_q_total = q_total.detach().mean().item()        # q_total.shape: (B, T)

        # per_agent_rewards = reward.mean(dim=1).squeeze(-1).mean(dim=0)  # [N]
        per_agent_losses = agent_qs.detach() - target_qs.detach()  # [B, T, N]
        per_agent_losses = (per_agent_losses ** 2).mean(dim=(0, 1))  # [N]

        ### 1. Q individual
        metrics = {
            'avg_loss': loss.item(),
            'avg_reward': r_total.sum().item() / (B*N),
            'avg_q_total': q_total.detach().mean().item(),
            'avg_entropy': self.epsilon_decay(),
            'per_agent_qs': {agent: per_agent_qs[i].item() for i, agent in enumerate(self.agents)},
            'per_agent_losses': {agent: per_agent_losses[i].item() for i, agent in enumerate(self.agents)}
        }
        return metrics
    
    def rollout_episode(self): ## 에피소드 단위로 collect 
        env = self.env
        obs_dict, _ = self.env.reset()
        for landmark in env.aec_env.unwrapped.world.landmarks:
            landmark.state.p_vel = np.zeros(2)
            landmark.movable = False
            landmark.collide = False
        episode_data = []

        obs = {agent: self.preprocess_observation(obs_dict[agent], agent) for agent in self.agents}
        h_states = {agent: torch.zeros(1, self.agent_nets[agent].hidden_dim, device=self.device) for agent in self.agents}
        last_actions = {agent: torch.zeros(1, self.env.action_space(agent).n, device=self.device) for agent in self.agents}

        for _ in range(self.max_steps):
            actions, h_next, q_selected_dict = {}, {}, {}
            for agent in self.agents:
                action, h_new = self.select_action(agent, obs[agent], last_actions[agent], h_states[agent])
                #print(f"[DEBUG] obs.shape = {obs[agent].shape}, last_actions.shape = {last_actions[agent].shape}, h_states.shape = {h_states[agent].shape}")
                actions[agent] = action
                h_next[agent] = h_new


            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            
            next_obs_proc = {agent: self.preprocess_observation(next_obs[agent], agent) for agent in self.agents}
            joint_obs = torch.stack([obs[agent].squeeze(0) if obs[agent].dim()==2 else obs[agent]
                                    for agent in self.agents])
            joint_next_obs = torch.stack([next_obs_proc[agent].squeeze(0) if next_obs_proc[agent].dim()==2 else next_obs_proc[agent]
                                           for agent in self.agents])
            joint_actions = torch.tensor([actions[agent] for agent in self.agents])
            joint_rewards = torch.tensor([rewards[agent] for agent in self.agents]).unsqueeze(-1)
            joint_dones = torch.tensor([terminations[agent] for agent in self.agents]).unsqueeze(-1)
            joint_hidden = torch.stack([h_states[agent].unsqueeze(0) if h_states[agent].dim()==1 else h_states[agent]
                                        for agent in self.agents])

            episode_data.append((joint_hidden, joint_obs, joint_actions, joint_rewards, joint_next_obs, joint_dones))

            obs = next_obs_proc
            h_states = h_next
            last_actions ={
                agent: F.one_hot(torch.tensor(actions[agent]), num_classes=self.env.action_space(agent).n).float().to(self.device) for agent in self.agents
            } 

            if all(terminations.values()) or all(truncations.values()):
                break
        
        # 버퍼에 푸시
        h_seq, s_seq, a_seq, r_seq, ns_seq, d_seq = zip(*episode_data)
        self.buffer.push(
            hidden_seq=torch.stack(h_seq),
            state_seq=torch.stack(s_seq),
            action_seq=torch.stack(a_seq),
            reward_seq=torch.stack(r_seq),
            next_state_seq=torch.stack(ns_seq),
            dones=torch.stack(d_seq)
        )


    
    def train(self, max_episode =1000, log_interval = 10):
        for episode in range(max_episode):
            self.rollout_episode() # returns: {agent_0: [q0, q1, ..., qT], ...}
            metrics = self.update()

            if not metrics:
                continue

            # Log overall metrics
            log_data = {
                'avg_reward': metrics['avg_reward'],
                'avg_q_total': metrics['avg_q_total'],
                'avg_loss': metrics['avg_loss'],
                #'avg_entropy': metrics['avg_entropy'],
            }

            # Log per-agent metrics
            for agent in self.agents:
                log_data[f'{agent}_q_indiv'] = metrics['per_agent_qs'][agent]
                log_data[f'{agent}_loss'] = metrics['per_agent_losses'][agent]
            
            self.logger.log_metrics(log_data, episode)

            if episode % log_interval == 0:
                self.logger.info(f"Episode {episode} | Avg Reward: {metrics['avg_reward']:.4f} | Avg Q total: {metrics['avg_q_total']:.4f} | Avg Loss: {metrics['avg_loss']:.4f}") #| Avg Entropy: {metrics['avg_entropy']:.4f}
                # self.logger.info(
                # f"Episode {episode} | Avg Reward: {metrics['avg_reward']:.4f} | "
                # f"Avg Loss: {metrics['avg_loss']:.4f} | Avg Entropy: {metrics['avg_entropy']:.4f} | "
                # + " | ".join([f"{agent}_Q: {avg_q_values[agent]:.2f}" for agent in self.agents])
                # )

        self.logger.close()

    def save(self, path):
        pass

    def load(self, path):
        pass



if __name__ == '__main__':
    env = simple_spread_v3.parallel_env(render_mode = 'None', N=3, max_cycles = 200, continuous_actions=False)

    for agent in env.aec_env.unwrapped.world.agents:
        agent.size = 0.02
    # env = aec_to_parallel(env)
    hidden_dims = 128

    qmix = QMIX(env=env, hidden_dims=hidden_dims, batch_size=64, buffer_capacity=10000, lr=0.0003, gamma=0.95,
                epochs=10, max_steps=200, log_dir="logs/qmix_simple_spread_logs", plot_window=100,
                update_interval=100, device="cpu", tau=0.01)

    qmix.train(max_episode=1000, log_interval=10)
    
