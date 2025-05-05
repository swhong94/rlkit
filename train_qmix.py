import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from pettingzoo.mpe import simple_spread_v3
from pettingzoo.utils.conversions import aec_to_parallel
import matplotlib.pyplot as plt


class AgentNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(AgentNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.q_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs, last_action, h_in):
        x = torch.cat([obs, last_action], dim=-1)
        x = F.relu(self.fc1(x))
        h = self.gru(x, h_in)
        q = self.q_out(h)
        return q, h


class HyperNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HyperNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, state):
        return torch.abs(self.fc(state))


class MixingNetwork(nn.Module):
    def __init__(self, n_agents, state_dim, hidden_dim=64):
        '''
        Mixing Network for QMIX
        q_total = f(q1, q2, ..., qn; state) - state에 따라 달라지는 가중치
        '''
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
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, agents_q, state): # agents_q : [q1, q2, ... , qn], state
        bs = agents_q.size(0) #agents_q = [bs, n_agents]
    
        w1 = self.hyper_w1(state).view(bs, self.n_agents, self.hidden_dim)  # W1 = [bs, n_agents, hidden_dim]
        b1 = self.hyper_b1(state).view(bs, 1, self.hidden_dim)              # b1 = [bs, 1, hidden_dim]
        hidden = F.elu(torch.bmm(agents_q.unsqueeze(1), w1) + b1).squeeze(1)
        # hidden = [bs, 1, n_agents] * [bs, n_agents, hidden_dim] + [bs, 1, hidden_dim]
        # hidden = [bs, hidden_dim]

        w2 = self.hyper_w2(state).view(bs, self.hidden_dim, 1) # W2 = [bs, hidden_dim, 1]
        b2 = self.hyper_b2(state)         # b2 = [bs, 1]

        q_total = torch.bmm(hidden.unsqueeze(1), w2).squeeze(1) + b2
        # q_total = [bs, hidden_dim] * [bs, hidden_dim, 1] + [bs, 1]
        # q_total = [bs, 1]
        return q_total
    

class ReplayBufferRNN:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, hidden_in, hidden_out, state_seq, action_seq, reward_seq, next_state_seq):
        data = (hidden_in, hidden_out, state_seq, action_seq, reward_seq, next_state_seq)
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = data
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        hi_lst, ho_lst, s_lst, a_lst, r_lst, ns_lst = zip(*batch)
        hi_tensor = torch.cat(hi_lst, dim=1).detach()
        ho_tensor = torch.cat(ho_lst, dim=1).detach()
        s_tensor = torch.tensor(np.array(s_lst), dtype=torch.float32)
        a_tensor = torch.tensor(np.array(a_lst), dtype=torch.long)
        r_tensor = torch.tensor(np.array(r_lst), dtype=torch.float32)
        ns_tensor = torch.tensor(np.array(ns_lst), dtype=torch.float32)
        return hi_tensor, ho_tensor, s_tensor, a_tensor, r_tensor, ns_tensor

    def __len__(self):
        return len(self.buffer)


def td_lambda_target(rewards, target_qs, gamma=0.99, td_lambda=0.8):
    B, T = rewards.shape
    targets = torch.zeros_like(rewards).to(rewards.device)
    targets[:, -1] = target_qs[:, -1]
    for t in reversed(range(T - 1)):
        targets[:, t] = rewards[:, t] + gamma * (
            td_lambda * targets[:, t + 1] + (1 - td_lambda) * target_qs[:, t + 1]
        )
    return targets


def plot_moving_average(rewards, window=20):
    avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(10, 4))
    plt.plot(avg)
    plt.title(f"Moving Average Reward (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.savefig("reward_moving_average.png")
    plt.show()


class QMIX(nn.Module):
    def __init__(self, n_agents, obs_dim, state_dim, action_dim, batch_size, buffer_capacity, lr=0.0003, gamma=0.99, update_interval=200, device=None):
        super(QMIX, self).__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_interval = update_interval
        self.step = 0
        self.device = device if device else 'cpu'

        self.agent_nets = nn.ModuleList([AgentNetwork(obs_dim, action_dim).to(self.device) for _ in range(n_agents)])
        self.target_agent_nets = nn.ModuleList([AgentNetwork(obs_dim, action_dim).to(self.device) for _ in range(n_agents)])
        self.mixing_net = MixingNetwork(n_agents, state_dim).to(self.device)
        self.target_mixing_net = MixingNetwork(n_agents, state_dim).to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, amsgrad=True)
        self.buffer = ReplayBufferRNN(buffer_capacity)

        self.epsilon_start = 1.0
        self.epsilon_end = 0.05

        self.update_target(force=True)

    def update_target(self, force=False):
        if force or (self.step % self.update_interval == 0):
            for i in range(self.n_agents):
                self.target_agent_nets[i].load_state_dict(self.agent_nets[i].state_dict())
            self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())

    def epsilon_decay(self, step):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * step / 30000)

    def select_action(self, obs, last_actions, h_states, epsilon=None):
        actions = []
        next_h_states = []
        if epsilon is None:
            epsilon = self.epsilon_decay(self.step)

        for i in range(self.n_agents):
            obs_i = torch.tensor(obs[i], dtype=torch.float32).to(self.device)
            last_a_i = torch.tensor(last_actions[i], dtype=torch.float32).to(self.device)
            h_in = h_states[i].unsqueeze(0).to(self.device)

            q_values, h_out = self.agent_nets[i](obs_i.unsqueeze(0), last_a_i.unsqueeze(0), h_in)
            q_values = q_values.squeeze(0)
            h_out = h_out.squeeze(0)

            if random.random() < epsilon:
                action = random.randint(0, self.action_dim - 1)
            else:
                action = q_values.argmax().item()

            actions.append(action)
            next_h_states.append(h_out)

        return actions, next_h_states

    def update(self):
        if len(self.buffer) < self.batch_size * 5:
            return None

        hidden_in, hidden_out, state, action, reward, next_state = self.buffer.sample(self.batch_size)
        B, T, N, obs_dim = state.shape

        agent_qs = []
        target_qs = []

        for i in range(self.n_agents):
            h_i = hidden_in[0, :, i, :].to(self.device)
            a_i = action[:, :, i].to(self.device)
            s_i = state[:, :, i, :].to(self.device)
            ns_i = next_state[:, :, i, :].to(self.device)

            a_i_onehot = F.one_hot(a_i, num_classes=self.action_dim).float()

            q_seq = []
            tq_seq = []

            for t in range(T):
                q_t, h_i = self.agent_nets[i](s_i[:, t, :], a_i_onehot[:, t, :], h_i)
                q_selected = q_t.gather(1, a_i[:, t].unsqueeze(-1)).squeeze(-1)
                q_seq.append(q_selected)

                tq_t, _ = self.target_agent_nets[i](ns_i[:, t, :], a_i_onehot[:, t, :], h_i)
                tq_max = tq_t.max(dim=1)[0]
                tq_seq.append(tq_max)

            agent_q = torch.stack(q_seq, dim=1)
            target_q = torch.stack(tq_seq, dim=1)
            agent_qs.append(agent_q)
            target_qs.append(target_q)

        agent_qs = torch.stack(agent_qs, dim=2)
        target_qs = torch.stack(target_qs, dim=2)

        global_states = state.view(B, T, -1).to(self.device)
        global_next_states = next_state.view(B, T, -1).to(self.device)

        q_total_list = []
        next_q_total_list = []
        for t in range(T):
            q_total_t = self.mixing_net(agent_qs[:, t, :], global_states[:, t, :])
            next_q_total_t = self.target_mixing_net(target_qs[:, t, :], global_next_states[:, t, :])
            q_total_list.append(q_total_t)
            next_q_total_list.append(next_q_total_t)

        q_total = torch.stack(q_total_list, dim=1).squeeze(-1)
        next_q_total = torch.stack(next_q_total_list, dim=1).squeeze(-1)

        reward = torch.clamp(reward, -1.0, 0.0)
        r_total = reward.sum(dim=2).to(self.device)
        y_tot = td_lambda_target(r_total, next_q_total, gamma=self.gamma, td_lambda=0.8)

        loss = F.mse_loss(q_total, y_tot.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step += 1
        self.update_target()
        return loss.item()


if __name__ == '__main__':
    env = simple_spread_v3.env(render_mode="None")
    env = aec_to_parallel(env)
    env.reset(seed=42)

    n_agents = len(env.possible_agents)
    obs_dim = env.observation_space(env.agents[0]).shape[0]
    state_dim = obs_dim * n_agents
    action_dim = env.action_space(env.agents[0]).n

    trainer = QMIX(n_agents=n_agents, obs_dim=obs_dim, state_dim=state_dim, action_dim=action_dim,
                   batch_size=32, buffer_capacity=10000)

    episodes = 300
    episode_limit = 25
    episode_rewards = []

    for episode in range(episodes):
        observations, _ = env.reset(seed=42)
        last_actions = [np.zeros(action_dim) for _ in range(n_agents)]
        h_states = [torch.zeros(64) for _ in range(n_agents)]

        episode_state, episode_action, episode_last_action = [], [], []
        episode_reward, episode_next_state = [], []
        total_reward = 0

        for t in range(episode_limit):
            obs_list = [observations[agent] for agent in env.agents]
            actions, h_states = trainer.select_action(obs_list, last_actions, h_states)
            one_hot_actions = [np.eye(action_dim)[a] for a in actions]
            action_dict = {agent: act for agent, act in zip(env.agents, actions)}

            next_observations, rewards, terminated, truncated, infos = env.step(action_dict)
            done = all(terminated[agent] or truncated[agent] for agent in env.agents)

            next_obs_list = [next_observations[agent] for agent in env.possible_agents]
            reward_list = [float(rewards.get(agent, 0.0)) for agent in env.possible_agents]

            episode_state.append(obs_list)
            episode_action.append(actions)
            episode_last_action.append(one_hot_actions)
            episode_reward.append(reward_list)
            episode_next_state.append(next_obs_list)

            observations = next_observations
            last_actions = one_hot_actions
            total_reward += sum(reward_list)

            if done:
                break

        h_in = torch.zeros(1, 1, n_agents, 64)
        h_out = torch.zeros(1, 1, n_agents, 64)

        trainer.buffer.push(
            h_in, h_out,
            np.array(episode_state, dtype=np.float32),            # [T, N, obs_dim]
            np.array(episode_action, dtype=np.int32),             # [T, N]
            np.array(episode_reward, dtype=np.float32),          # [T, N]
            np.array(episode_next_state, dtype=np.float32)        # [T, N, obs_dim]
        )


        trainer.update()
        episode_rewards.append(total_reward)

        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {total_reward:.2f}, Avg(10): {np.mean(episode_rewards[-10:]):.2f}")
    
    plot_moving_average(episode_rewards, window=10)
    #plt.plot(episode_rewards)
    #plt.xlabel("Episode")
    #plt.ylabel("Total Reward")
    #plt.title("QMIX Training Rewards")
    #plt.grid(True)
    #plt.savefig("qmix_training_rewards.png")
    #plt.show()
