import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from pettingzoo.mpe import simple_spread_v3
from pettingzoo.utils.conversions import aec_to_parallel
import supersuit as ss
import matplotlib.pyplot as plt


class AgentNetwork(nn.Module):
    '''
    MARL problem -> partial observability, communication constraints among agents
    -> necessity of lr of decentralized policy 
    conditioning on the local action-observation history of each agent
    '''
    def __init__(self, obs_dim, action_dim, hidden_dim=64):  # hidden_dim -> history 저장
        super(AgentNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(obs_dim+action_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim) 
        self.q_out = nn.Linear(hidden_dim, action_dim)


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
        self.state_dim = state_dim

        self.hyper_w1 = HyperNetwork(state_dim, n_agents * hidden_dim) # n_agents * each history
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)

        self.hyper_w2 = HyperNetwork(state_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  #Q_total
        )

    def forward(self, agents_q, state):  
        '''
        q_total = f(Q_1, Q_2, ..., Q_n; s)
        = relu(w1 @ Q_agents + b1) @ w2 + b2
        ''' 
        bs = agents_q.size(0)  # batch size = 각 배치마다의 에이전트 Q 값 = (bs, n_agents)
        
        w1 = self.hyper_w1(state).view(bs, self.n_agents, self.hidden_dim) # .view -> reshape 함수
        b1 = self.hyper_b1(state).view(bs, 1, self.hidden_dim)
        hidden = nn.functional.elu(torch.bmm(agents_q.unsqueeze(1), w1) + b1)  # hidden = aqent_qs @ w1 + b1

        w2 = self.hyper_w2(state).view(bs, self.hidden_dim, 1)
        b2 = self.hyper_b2(state).view(bs, 1, 1)
        q_total = torch.bmm(hidden, w2) + b2  # hidden @ w2 + b2

        return q_total.view(-1, 1)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch

    def __len__(self):
        return len(self.buffer)
    

# QMIX 전체 구성 클래스
class QMIX(nn.Module):
    def __init__(self, n_agents, obs_dim, state_dim, action_dim, batch_size, buffer_capacity,
                lr=0.001, gamma=0.99, update_interval = 200, device=None):
        super(QMIX, self).__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.step = 0
        self.gamma = gamma
        
        self.agent_nets = nn.ModuleList([AgentNetwork(obs_dim, action_dim) for _ in range(n_agents)])
        self.target_agent_nets = nn.ModuleList([AgentNetwork(obs_dim, action_dim) for _ in range(n_agents)])
        self.mixing_net = MixingNetwork(n_agents, state_dim)
        self.target_mixing_net = MixingNetwork(n_agents, state_dim)

        self.optimizer = optim.Adam(list(self.agent_nets.parameters())+list(self.mixing_net.parameters()), lr=lr) 
        self.buffer = ReplayBuffer(buffer_capacity)

        self.epsilon_start = 1.0
        self.epsilon_end = 0.05

        self.device = device if device else 'cpu'
        self.to(self.device)

        self.update_target(self.step, self.update_interval)

    def to(self, device):
        for agent_net, target_net in zip(self.agent_nets, self.target_agent_nets):
            agent_net.to(device)
            target_net.to(device)
        self.mixing_net.to(device)
        self.target_mixing_net.to(device)

    def epsilon_decay(self, step):
        # epsilon = max(self.epsilon_end , self.epsilon_start*0.95)
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * step / 50000)
        
        return epsilon
    
    def select_action(self, obs, last_actions, h_states):
        actions = []
        next_h_states = []
        epsilon = self.epsilon_decay(self.step)

        for i in range(self.n_agents):
            # tau = torch.cat([obs[i], taus[i][-1][1]]) if taus[i] else obs[i]

            tau = torch.tensor(obs[i], dtype=torch.float32).to(self.device)
            last_action_tensor = torch.tensor(last_actions[i], dtype = torch.float32).to(self.device)
            h_in = h_states[i].to(self.device)
            q_values, h_out = self.agent_nets[i](tau.unsqueeze(0), last_action_tensor.unsqueeze(0), h_in.unsqueeze(0))
            q_values= q_values.squeeze(0)
            h_out= h_out.squeeze(0)

            if random.random() < epsilon:
                action = random.randint(0, self.action_dim - 1)
            else:
                action = q_values.argmax().item()
            actions.append(action)
            next_h_states.append(h_out)
        return actions, next_h_states

    def store_transition(self, transition):
        self.buffer.push(transition)

    def update_target(self, step, update_interval):
        if(step % update_interval ==0):
            for i in range(self.n_agents):
                self.target_agent_nets[i].load_state_dict(self.agent_nets[i].state_dict())
            self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())

    
    def update(self):
        if len(self.buffer) < self.batch_size:
            return None
        
        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.tensor(np.array(states, dtype=np.float32), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions, dtype=np.float32), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(np.array(rewards, dtype=np.float32), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states, dtype=np.float32), dtype=torch.float32).to(self.device)

        # obs = torch.tensor(batch.obs, dtype=torch.float32).to(self.device)  # (B, n_agents, obs_dim)
        # next_obs = torch.tensor(batch.next_obs, dtype=torch.float32).to(self.device)
        # dones = torch.tensor(batch.dones, dtype=torch.float32).to(self.device).unsqueeze(-1)
        # last_actions = torch.tensor(batch.last_actions, dtype=torch.float32).to(self.device)

        # agent Q-values
        agent_qs =[]
        next_agent_qs=[]

        for i in range(self.n_agents):
            h_in = torch.zeros(self.batch_size, 64).to(self.device)

            agent_action = actions[:,i,:].argmax(dim=-1)
            agent_action_onehot = nn.functional.one_hot(agent_action, num_classes = self.action_dim).float()
            
            q_values, _ = self.agent_nets[i](states[:,i,:], agent_action_onehot, h_in)
            agent_q = q_values.gather(1, q_values.argmax(dim=1,keepdim=True))
            agent_qs.append(agent_q)

            target_q_values, _ = self.agent_nets[i](next_states[:,i,:], actions[:,i,:], h_in)
            target_q = target_q_values.max(dim=1, keepdim = True)[0]
            next_agent_qs.append(target_q)

        agent_qs = torch.cat(agent_qs, dim=1)
        next_agent_qs = torch.cat(next_agent_qs, dim=1)

        global_states = states.view(self.batch_size,-1)
        next_global_states = next_states.view(self.batch_size,-1)

        q_total = self.mixing_net(agent_qs, global_states)
        next_q_total = self.target_mixing_net(next_agent_qs, next_global_states)
        
        y_tot = rewards.sum(dim=1, keepdim=True)+self.gamma*next_q_total

        loss = nn.functional.mse_loss(y_tot, q_total)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print(f"Step: {self.step}, Loss: {loss.item():.4f}")

        self.update_target(self.step, self.update_interval)

if __name__ == "__main__":
    env = simple_spread_v3.env(render_mode = "human")
    env = aec_to_parallel(env)
    env.reset(seed=42)
    # env = ss.pettingzoo_env_to_vec_env_v1(env)
    # env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='gymnasium')
    # env.reset(seed=42)

    n_agents = len(env.possible_agents)
    obs_dim=env.observation_space(env.agents[0]).shape[0]
    state_dim = obs_dim * n_agents
    action_dim =env.action_space(env.agents[0]).n

    trainer = QMIX(n_agents=n_agents, obs_dim=obs_dim, state_dim=state_dim, action_dim=action_dim,
                   batch_size=32, buffer_capacity=10000)

    episodes = 1000
    episode_limit = 30
    episode_reward = []

    for episode in range(episodes):
        
        observations, _ = env.reset(seed=42)
        # observations = [env.observe(agent) for agent in env.agents]
        last_actions = [np.zeros(trainer.action_dim) for _ in range(n_agents)]
        h_states = [torch.zeros(64) for _ in range(n_agents)]
        done = False
        t=0
        total_reward = 0

        while not done and t < episode_limit: # 모든 에이전트가 종료될때까지 반복
            obs_list = [observations[agent] for agent in env.agents]
            actions, h_states = trainer.select_action(obs_list, last_actions, h_states)
            one_hot_actions = [np.eye(trainer.action_dim)[a] for a in actions]
            action_dict = {agent: action for agent, action in zip(env.agents, actions)}
            
            observations, rewards, terminated, truncated, infos = env.step(action_dict)
            done = all(terminated[agent] or truncated[agent] for agent in env.agents)

            # for agent, action in zip(env.agents, actions):
            #     env.step(action)

            next_obs_list = [observations[agent] for agent in env.agents]
            reward_list = [float(rewards.get(agent, 0.0)) for agent in env.agents]
            # dones = [env.terminations[agent] or env.truncations[agent] for agent in env.agents]
            # done = np.all(dones)
            print(f"reward_list shape: {np.shape(reward_list)}, content: {reward_list}")
            if len(reward_list) != n_agents:
                print(f"[Warning] Incomplete reward list: {reward_list}")

            else:
                trainer.store_transition((
                    np.array(obs_list, dtype=object),
                    np.array(one_hot_actions, dtype=np.float32),
                    np.array(reward_list, dtype=np.float32),
                    np.array(next_obs_list, dtype=np.float32)
                ))

            # observations = next_observations
            last_actions = one_hot_actions
            trainer.step += 1
            t += 1

            trainer.update()
            total_reward+= sum(rewards.values())
  
        episode_reward.append(total_reward)

        if trainer.step % 500 == 0:
            avg_reward = np.mean(episode_reward[-10:])
            print(f"Step {trainer.step}, Average Reward (last 10 eps): {avg_reward:.2f}")

    # Plot after training
    plt.plot(episode_reward)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode Rewards Over Time")
    plt.grid(True)
    plt.savefig("episode_rewards.png")
    plt.show()
