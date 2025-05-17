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
# 4. seq T=10

class AgentNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(AgentNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True) 
        self.q_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs_seq, last_action_seq, h_0 = None):
        """
        obs_seq: (B, T, obs_dim)
        last_action_seq: (B, T, action_dim)
        h_0: (1, B, hidden_dim) or None
        Returns:
            q_seq: (B, T, action_dim)
            h_T: (1, B, hidden_dim)
        """
        x = torch.cat([obs_seq, last_action_seq], dim=-1)   # (B, T, obs_dim + action_dim)
        x = F.relu(self.fc1(x))                             # (B, T, hidden_dim)
        if h_0 is None:
            h_0 = torch.zeros(1, x.size(0), self.hidden_dim, device=x.device)
        elif h_0.dim() == 2:                                  # (B, hidden_dim)
            h_0 = h_0.unsqueeze(0)                          # (1, B, hidden_dim)    
        elif h_0.dim() == 3:
            pass  # already correct
        else:
            raise ValueError(f"Unexpected h_0 shape: {h_0.shape}")
        
        gru_out, h_T = self.gru(x, h_0)                 # (B, T, hidden_dim), (1, B, hidden_dim)
        q_seq = self.q_out(gru_out)                     # (B, T, action_dim)
        return q_seq, h_T.squeeze(0)                    # (B, T, action_dim), (B, hidden_dim)


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

    def forward(self, agents_q: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        agents_q: shape (B, N)
        state: shape (B, state_dim)
        returns: q_total (B,)
        """
        B, N = agents_q.shape
        S = state.shape[1]

        agents_q = agents_q.view(B, 1, N)
        state = state.view(B, S)

        w1 = self.hyper_w1(state).view(B, N, self.hidden_dim)
        b1 = self.hyper_b1(state).view(B, 1, self.hidden_dim)

        hidden = F.elu(torch.bmm(agents_q, w1) + b1)

        w2 = self.hyper_w2(state).view(B, self.hidden_dim, 1)
        b2 = self.hyper_b2(state).view(B, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2  # (B, 1, 1)
        q_total = torch.clamp(q_total, -10.0, 10.0)
        return q_total.squeeze(-1).squeeze(-1)  # (B,)

    

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
                dones.detach()
            )
        self.buffer.append(data)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        h_lst, s_lst, a_lst, r_lst, ns_lst, dn_lst = zip(*batch)
        '''
        B = batch_size, T = max_step, N = n_agents, H_dim = hidden_dim, obs_dim = observation_dim
        '''
        h_tensor = torch.stack(h_lst).to(self.device)     # (B, T+1, N, H_dim) 
        s_tensor = torch.stack(s_lst).to(self.device)     # (B, T, N, obs_dim)
        a_tensor = torch.stack(a_lst).long().to(self.device)      # (B, T, N) 
        r_tensor = torch.stack(r_lst).to(self.device)             # (B, T, N) 
        ns_tensor = torch.stack(ns_lst).to(self.device)   # (B, T, N, obs_dim)
        d_tensor = torch.stack(dn_lst).to(self.device)            # (B, T, N) 
        return h_tensor, s_tensor, a_tensor, r_tensor, ns_tensor, d_tensor

    def __len__(self):
        return len(self.buffer)


def td_lambda_target(rewards, target_qs, gamma=0.95, td_lambda=0.8, done_mask=None):
    """
    Args:
        rewards: (B, T)
        target_qs: (B, T)
        done_mask: (B, T) or None
    Returns:
        targets: (B, T)
    """
    assert rewards.shape == target_qs.shape, "Shape mismatch between rewards and target_qs"
    B, T = rewards.shape

    targets = torch.zeros_like(rewards).to(rewards.device)
    targets[:, -1] = target_qs[:, -1]

    for t in reversed(range(T - 1)):
        if done_mask is not None:
            mask = 1.0 - done_mask[:, t + 1].float()
        else:
            mask = 1.0
        targets[:, t] = rewards[:, t] + gamma * mask * (
            td_lambda * targets[:, t + 1] + (1 - td_lambda) * target_qs[:, t + 1]
        )

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
        self.decay_ratio = 0.985
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
        last_action_tensor = torch.FloatTensor(last_action).to(self.device)     
        h_in = torch.FloatTensor(his_in).to(self.device) # (1, H_dim)

        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)   # (1, obs_dim)
        if last_action_tensor.dim() == 1:
            last_action_tensor = last_action_tensor.unsqueeze(0)  # (1, action_dim)
        if h_in.dim() == 1:
            h_in = h_in.unsqueeze(0)  # (1, H)

        obs_tensor = obs_tensor.unsqueeze(1)         # (1, 1, obs_dim)
        last_action_tensor = last_action_tensor.unsqueeze(1)  # (1, 1, action_dim)
        h_in = h_in.unsqueeze(0)                # (1, 1, H_dim)
        
        q_values, h_out = self.agent_nets[agent](obs_tensor, last_action_tensor, h_in)
        q_values = q_values.squeeze(0).squeeze(0) 
        h_out = h_out.squeeze(0) # (H_dim)

        if random.random() < self.epsilon_start:
            action = random.randint(0, self.env.action_space(agent).n - 1)
        else:
            action = q_values.argmax().item()

        '''선택한 액션의 Q값만 추출, agent마다, step마다 다름'''
        # q_selected = q_values[action].item()  # scalar: float

        return action, h_out 

        
    def update(self):
        if len(self.buffer) < self.batch_size:
            return 0.0

        hidden_seq, state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        B, T, N, obs_dim = state.shape

        agent_qs, target_qs = [], []

        for i, agent in enumerate(self.agents):
            a_i = action[:, :, i]                      # (B, T)
            s_i = state[:, :, i, :]                    # (B, T, obs_dim)
            ns_i = next_state[:, :, i, :]              # (B, T, obs_dim)
            h0_i = hidden_seq[:, 0, i, :]               # (B, T, H)
            h0_i = h0_i.squeeze(1) 
            h0_i = h0_i.unsqueeze(0)

            # Flatten last action to one-hot
            a_onehot_seq = F.one_hot(a_i, num_classes=self.env.action_space(agent).n).float()  # (B, T, A)

            # Compute Q(s, a)
            q_out, _ = self.agent_nets[agent](s_i, a_onehot_seq, h0_i)
            q_selected = q_out.gather(2, a_i.unsqueeze(-1)).squeeze(-1)  # (B, T)

            with torch.no_grad():
                # Target Q(s', a')
                q_next, _ = self.agent_nets[agent](ns_i, a_onehot_seq, h0_i)  # use main net for greedy
                next_action = q_next.argmax(dim=2, keepdim=True)  # (B, T, 1)
                next_a_onehot = F.one_hot(next_action.squeeze(-1), num_classes=self.env.action_space(agent).n).float()

                q_target, _ = self.target_agent_nets[agent](ns_i, next_a_onehot, h0_i)
                tq = q_target.gather(2, next_action).squeeze(-1)
                
            agent_qs.append(q_selected)
            target_qs.append(tq)

        agent_qs = torch.stack(agent_qs, dim=2)      # (B, T, N)
        target_qs = torch.stack(target_qs, dim=2)    # (B, T, N)

        state = state.view(B, T, -1)
        next_state = next_state.view(B, T, -1)

        q_total = torch.stack([self.mixing_net(agent_qs[:, t], state[:, t]) for t in range(T)], dim=1).squeeze(-1)
        tq_total = torch.stack([self.target_mixing_net(target_qs[:, t], next_state[:, t]) for t in range(T)], dim=1).squeeze(-1)
        print(f"[DEBUG] q_total mean: {q_total.mean().item()}, agent_qs mean: {agent_qs.mean().item()}")

        r_total = reward.sum(dim=2).squeeze(-1)
        # y_total = td_lambda_target(r_total, tq_total, gamma=self.gamma, td_lambda=0.8)
        y_total = r_total[:, :-1] + self.gamma * tq_total[:, 1:]
        y_total = F.pad(y_total, (0, 1))  # 마지막 timestep은 target 없이 그대로 유지

        loss = F.mse_loss(q_total, y_total.detach())
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad is not None:
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.clip_grad)
        self.optimizer.step()

        eps = self.epsilon_decay()
        self.step += 1
        self.update_target()

        per_agent_qs = agent_qs.detach().mean(dim=(0, 1))  
        per_agent_losses = ((agent_qs.detach() - target_qs.detach()) ** 2).mean(dim=(0, 1))
        avg_q_total = q_total.detach().mean().item()
        avg_reward = r_total.sum().item()

        metrics = {
            'avg_loss': loss.item(),
            'avg_reward': avg_reward,
            'avg_q_total': avg_q_total,
            'avg_entropy': eps,
            'per_agent_qs': {agent: per_agent_qs[i].item() for i, agent in enumerate(self.agents)},
            'per_agent_losses': {agent: per_agent_losses[i].item() for i, agent in enumerate(self.agents)}
        }

        return metrics
    
    def rollout_episode(self): 
        ''' 에피소드 단위로 collect 200 step 
        -> T = 10 시퀀스 길이로 설정 '''
        T = 10
        env = self.env
        obs_dict, _ = self.env.reset()
        episode_data = [] # 시퀀스 저장

        # for landmark in env.aec_env.unwrapped.world.landmarks:
        #     landmark.state.p_vel = np.zeros(2)
        #     landmark.movable = False
        #     landmark.collide = False
        

        obs = {agent: self.preprocess_observation(obs_dict[agent], agent) for agent in self.agents}
        h_states = {agent: torch.zeros(1, self.agent_nets[agent].hidden_dim, device=self.device) for agent in self.agents}
        last_actions = {agent: torch.zeros(1, self.env.action_space(agent).n, device=self.device) for agent in self.agents}

        for _ in range(self.max_steps):
            actions, h_next = {}, {}
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
            # for i, agent in enumerate(self.agents):
            #     print(f"[DEBUG] joint_obs.shape = {joint_obs[i].shape}, joint_actions.shape = {joint_actions[i].item()}, joint_rewards.shape = {joint_rewards[i].item()}")
                
            episode_data.append((joint_hidden, joint_obs, joint_actions, joint_rewards, joint_next_obs, joint_dones))

            # episode_data.append((joint_hidden.unsqueeze(0), 
            #                      joint_obs.unsqueeze(0), 
            #                      joint_actions.unsqueeze(0), 
            #                      joint_rewards.unsqueeze(0), 
            #                      joint_next_obs.unsqueeze(0), 
            #                      joint_dones.unsqueeze(0)
            #                     ))

            if len(episode_data) >= T:
                # 버퍼에 푸시 -> step마다 푸시하는게 아니라 episode 단위로 푸시 (수정 필요)
                h_seq, s_seq, a_seq, r_seq, ns_seq, d_seq = zip(*episode_data[-T:])
                self.buffer.push(
                    hidden_seq=torch.stack(h_seq),        # (T, N, H)
                    state_seq=torch.stack(s_seq),         # (T, N, obs_dim)
                    action_seq=torch.stack(a_seq),        # (T, N)
                    reward_seq=torch.stack(r_seq),        # (T, N, 1)
                    next_state_seq=torch.stack(ns_seq),   # (T, N, obs_dim)
                    dones=torch.stack(d_seq)              # (T, N, 1)
                )

            obs = next_obs_proc
            h_states = h_next
            last_actions ={
                agent: F.one_hot(torch.tensor(actions[agent]), num_classes=self.env.action_space(agent).n).float().to(self.device) 
                for agent in self.agents
            } 

            if all(terminations.values()) or all(truncations.values()):
                break


    
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
                self.logger.info(f"Episode {episode} | Avg Reward: {metrics['avg_reward']:.4f} | Avg Q total: {metrics['avg_q_total']:.4f} | Avg Loss: {metrics['avg_loss']:.4f} | Avg Entropy: {metrics['avg_entropy']:.4f}") 
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
    
