import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

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

class AgentNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(AgentNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.q_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs, last_action, his_in):
        x = torch.cat([obs, last_action], dim=-1)
        x = F.relu(self.fc1(x))
        his_out = self.gru(x, his_in) # f.relu() 할지 말지 
        q = self.q_out(his_out)
        return q, his_out # q = action_dim만큼 (q1, q2, ... , qn) + his_out


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

    def forward(self, agents_q, state): # agents_q : [q1, q2, ... , qn], state
        bs = agents_q.size(0) #agents_q = [bs, n_agents]
    
        w1 = self.hyper_w1(state).view(bs, self.n_agents, self.hidden_dim)  # W1 = [bs, n_agents, hidden_dim]
        b1 = self.hyper_b1(state).view(bs, 1, self.hidden_dim)              # b1 = [bs, 1, hidden_dim]
        hidden = F.elu(torch.bmm(agents_q.unsqueeze(1), w1) + b1).squeeze(1)             
        # hidden = [bs, 1, n_agents] * [bs, n_agents, hidden_dim] + [bs, 1, hidden_dim]
        # hidden = [bs, hidden_dim]

        w2 = self.hyper_w2(state).view(bs, self.hidden_dim, 1)              # W2 = [bs, hidden_dim, 1]
        b2 = self.hyper_b2(state)                                           # b2 = [bs, 1]

        q_total = torch.bmm(hidden.unsqueeze(1), w2).squeeze(1) + b2
        # q_total = [bs, hidden_dim] * [bs, hidden_dim, 1] + [bs, 1]
        # q_total = [bs, 1]
        return q_total
    

class ReplayBufferRNN:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.device = torch.device("cpu")

    def push(self, hidden_seq, state_seq, action_seq, reward_seq, next_state_seq, dones):
        data = tuple(tensor.detach().cpu().clone() for tensor in (
            hidden_seq, state_seq, action_seq, reward_seq, next_state_seq, dones
        ))
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = data
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        h_lst, s_lst, a_lst, r_lst, ns_lst, dn_lst = zip(*batch)
        # hi_tensor = torch.cat(hi_lst, dim=1).detach()
        # ho_tensor = torch.cat(ho_lst, dim=1).detach()
        # 텐서 스택 후 지정 디바이스로 이동
        h_tensor = torch.stack(h_lst).to(self.device)     # (B, T+1, N, H)
        s_tensor = torch.stack(s_lst).to(self.device)     # (B, T, N, obs)
        a_tensor = torch.stack(a_lst).to(self.device)     # (B, T, N)
        r_tensor = torch.stack(r_lst).to(self.device)     # (B, T, N)
        ns_tensor = torch.stack(ns_lst).to(self.device)   # (B, T, N, obs)
        d_tensor = torch.stack(dn_lst).to(self.device)    # (B, T, N)

        # 정수형 보정 (action)
        a_tensor = a_tensor.long()
        return h_tensor, s_tensor, a_tensor, r_tensor, ns_tensor, d_tensor

    def __len__(self):
        return len(self.buffer)


def td_lambda_target(rewards, target_qs, gamma=0.95, td_lambda=0.8):
    B, T = rewards.shape
    targets = torch.zeros_like(rewards).to(rewards.device)
    targets[:, -1] = target_qs[:, -1]
    for t in reversed(range(T - 1)):
        targets[:, t] = rewards[:, t] + gamma * (
            td_lambda * targets[:, t + 1] + (1 - td_lambda) * target_qs[:, t + 1]
        ) # td_lambda * target_qs[:, t + 1] + (1 - td_lambda) * targets[:, t + 1]
    return targets


# def plot_moving_average(rewards, window=20):
#     avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
#     plt.figure(figsize=(10, 4))
#     plt.plot(avg)
#     plt.title(f"Moving Average Reward (window={window})")
#     plt.xlabel("Episode")
#     plt.ylabel("Average Reward")
#     plt.grid(True)
#     plt.savefig("reward_moving_average.png")
#     plt.show()


class QMIX(nn.Module):
    def __init__(self,
                env,
                hidden_dims, 
                #n_agents,
                #obs_dim, 
                #state_dim, 
                #action_dim, 
                batch_size = 64, 
                buffer_capacity = 10000, 
                lr=0.0003, 
                gamma=0.95, 
                epochs = 10,
                max_steps = 200,
                log_dir = "logs/qmix_discrete_logs",
                plot_window = 100,
                # entropy_coeff = 0.01,
                clip_grad = None,
                update_interval=100, 
                device="cpu"
                ):
        super(QMIX, self).__init__()
        # self.n_agents = n_agents
        # self.obs_dim = obs_dim
        # self.state_dim = state_dim
        # self.action_dim = action_dim

        # Environment
        self.env = env
        env.reset()
        self.agents = env.agents
        self.n_agents = len(self.agents)
        self.device = torch.device(device)
        self.buffer = ReplayBufferRNN(buffer_capacity)

        self.log_prefix = "qmix_" + "simple_spread"


        self.agent_nets = nn.ModuleDict()
        self.target_agent_nets = nn.ModuleDict()
        self.mixing_net = {}
        self.target_mixing_net = {}
        self.optimizer = {}
        self.obs_spaces = {}

        for agent in self.agents:
            obs_space = env.observation_space(agent)
            # Compute total input dimension from discrete action space 
            if isinstance(obs_space, gym.spaces.Dict):
                obs_dim = sum(space.n if isinstance(space, gym.spaces.Discrete) else space.shape[0] for space in obs_space.spaces.values())
            else:
                obs_dim = obs_space.n if isinstance(obs_space, gym.spaces.Discrete) else obs_space.shape[0]
            # obs_dim = sum(space.n if isinstance(space, gym.spaces.Discrete) else space.shape[0] for space in obs_space.spaces.values()) 
            act_dim = self.env.action_space(agent).n

            self.agent_nets[agent] = AgentNetwork(obs_dim, act_dim, hidden_dims).to(self.device)
            self.target_agent_nets[agent] = AgentNetwork(obs_dim, act_dim, hidden_dims).to(self.device)
            self.obs_spaces[agent] = obs_space
        
        ''' agent_nets의 네트워크 파라미터도 optimizer에 포함시켜야 함'''
        agent_params = []
        for agent in self.agent_nets.values():
            agent_params += list(agent.parameters())

        self.mixing_net = MixingNetwork(self.n_agents, obs_dim, hidden_dims).to(self.device)
        self.target_mixing_net = MixingNetwork(self.n_agents, obs_dim, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(agent_params+list(self.mixing_net.parameters()), 
                                    lr=lr,
                                    amsgrad=True)

        self.batch_size = batch_size
        self.gamma = gamma
        self.update_interval = update_interval
        self.epochs = epochs
        self.max_steps = max_steps
        self.clip_grad = clip_grad
        # self.entropy_coeff = entropy_coeff -> epsilon decay로 대체

        self.logger = Logger(log_dir, self.log_prefix, plot_window)

        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.step = 0

        self.update_target(force=True)

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
            return torch.cat(one_hots)
        elif isinstance(obs, np.ndarray):
            return torch.FloatTensor(obs).to(self.device)
        else:
            raise TypeError(f"Unsupported observation type: {type(obs)}")


    def update_target(self, force=False): # hard update
        if force or (self.step % self.update_interval == 0):
            for agent in self.agents:
                self.target_agent_nets[agent].load_state_dict(self.agent_nets[agent].state_dict())
            self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())


    def epsilon_decay(self, step):
        decay_ratio = max(0, (1 - step / 20000))
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * decay_ratio


    def select_action(self, agent, obs, last_action, h_state):
        '''
        하나의 에이전트에 대한 action을 선택
        Args: observation,
        이전 action,
        hidden state
        
        Returns: action(epsilon-greedy),
        next hidden state
        '''
        epsilon = self.epsilon_decay(self.step)

        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        last_action_tensor = torch.FloatTensor(last_action).unsqueeze(0).to(self.device)
        h_in = h_state.unsqueeze(0).to(self.device)

        q_values, h_out = self.agent_nets[agent](obs_tensor, last_action_tensor, h_in)
        q_values = q_values.squeeze(0)
        h_out = h_out.squeeze(0)

        if random.random() < epsilon:
            action = random.randint(0, self.env.action_space(agent).n - 1)
        else:
            action = q_values.argmax().item()

        return action, h_out

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        # 샘플링: (B, T+1, N, H), (B, T, N, obs), ...
        hidden_seq, state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        B, T, N, obs_dim = state.shape

        device = self.device
        agent_qs, target_qs = [], []
        metrics = {}

        for i, agent in enumerate(self.agents):
            a_i = action[:, :, i].to(device)                      # (B, T)
            s_i = state[:, :, i, :].to(device)                    # (B, T, obs)
            ns_i = next_state[:, :, i, :].to(device)              # (B, T, obs)

            q_seq, tq_seq = [], []
            for t in range(T):
                h_i = hidden_seq[:, t, i, :].to(device) if t < hidden_seq.size(1) - 1 else None
                a_onehot = F.one_hot(a_i[:, t], num_classes=self.env.action_space(agent).n).float().to(device)

                q_t, _ = self.agent_nets[agent](s_i[:, t], a_onehot, h_i)
                q_selected = q_t.gather(1, a_i[:, t].unsqueeze(-1)).squeeze(-1)
                q_seq.append(q_selected)

                with torch.no_grad():
                    a_next_onehot = F.one_hot(next_action.squeeze(-1), num_classes=self.env.action_space(agent).n).float().to(device)
                    q_next = self.agent_nets[agent](ns_i[:, t], a_next_onehot, h_i)[0]
                    next_action = q_next.argmax(dim=1, keepdim=True)
                    q_target, _ = self.target_agent_nets[agent](ns_i[:, t], a_onehot, h_i)
                    tq = q_target.gather(1, next_action).squeeze(-1)
                    tq_seq.append(tq)

            agent_qs.append(torch.stack(q_seq, dim=1))     # (B, T)
            target_qs.append(torch.stack(tq_seq, dim=1))   # (B, T)

        # (B, T, N)
        agent_qs = torch.stack(agent_qs, dim=2)
        target_qs = torch.stack(target_qs, dim=2)

        # (B, T, global_obs)
        global_states = state.view(B, T, -1).to(device)
        global_next_states = next_state.view(B, T, -1).to(device)

        # Mixing network
        q_total_list, tq_total_list = [], []
        for t in range(T):
            q_total = self.mixing_net(agent_qs[:, t, :], global_states[:, t, :])
            tq_total = self.target_mixing_net(target_qs[:, t, :], global_next_states[:, t, :])
            q_total_list.append(q_total)
            tq_total_list.append(tq_total)

        q_total = torch.stack(q_total_list, dim=1).squeeze(-1)     # (B, T)
        tq_total = torch.stack(tq_total_list, dim=1).squeeze(-1)   # (B, T)

        # reward sum across agents (optional: per-agent reward instead)
        r_total = reward.sum(dim=2).to(device)                     # (B, T)
        y_total = td_lambda_target(r_total, tq_total, gamma=self.gamma, td_lambda=0.8)

        # loss and optimization
        loss = F.mse_loss(q_total, y_total.detach())
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad is not None:
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.clip_grad)
        self.optimizer.step()

        self.step += 1
        self.update_target()
        # avg_total_reward = r_total / self.step
        # avg_loss= loss / self.step
        # avg_entropy = self.epsilon_decay(self.step)  # ε 자체를 entropy 대용으로 기록

        for i, agent in enumerate(self.agents):
            metrics[agent] = {
                "loss": loss.item(),
                "total_reward": r_total.sum().item() / B,
                "entropy": self.epsilon_decay(self.step)  # ε 자체를 entropy 대용으로 기록
            }
        return metrics
    
    def train(self, max_episode =1000, log_interval = 10):
        for episode in range(max_episode):
            agent_metrics = self.update()

            if not agent_metrics:
                #print(f"Episode {episode}: Not enough data in buffer to update.")
                continue

            avg_reward = np.mean([metrics['total_reward'] for metrics in agent_metrics.values()])
            avg_loss = np.mean([metrics['loss'] for metrics in agent_metrics.values()])
            avg_entropy = np.mean([metrics['entropy'] for metrics in agent_metrics.values()])
            
            metrics = {
                'avg_reward': avg_reward,
                'avg_loss': avg_loss,
                'avg_entropy': avg_entropy
            }
            # for agent in self.agents:
            #     metrics["avg_loss"] = agent_metrics['avg_loss']
            #     metrics["avg_entropy"] = agent_metrics['avg_entropy']
            #     metrics["avg_total_reward"] = agent_metrics['avg_reward']
            self.logger.log_metrics(metrics, episode)

            if episode % log_interval == 0:
                self.logger.info(f"Episode {episode} | Avg Reward: {avg_reward:>10.4f}")

        self.logger.close()

    def save(self, path):
        pass

    def load(self, path):
        pass



if __name__ == '__main__':
    env = simple_spread_v3.parallel_env(N=3, max_cycles = 200, continuous_actions=False)
    # env = aec_to_parallel(env)
    hidden_dims = 128

    qmix = QMIX(env=env, hidden_dims=hidden_dims, batch_size=32, buffer_capacity=10000, lr=0.0003, gamma=0.95,
                epochs=10, max_steps=200, log_dir="logs/qmix_discrete_logs", plot_window=100,
                update_interval=100, device="cpu")

    qmix.train(max_episode=500, log_interval=10)
    
