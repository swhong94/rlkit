import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys
import time
import torch
from env.QLBT_agents import QLBT_Agent, QLBT_AP

if 'ipykernel' in sys.modules:
    from IPython import display

class AccessPoint: 
    def __init__(self):
        self.channel_state = "IDLE"  # IDLE, ACK, COLLISION
        self.channel_busy = False
        self.current_transmitter = None
        self.sifs= 2                # short inter-frame space
        self.ack = 0                # number of ACKs
        self.collision = 0          # number of collisions
        self.success = 0            # number of successful transmissions
    
    def receive(self, nodes):   # 수신한 노드 리스트를 받아서 채널 상태 확인
        if len(nodes) > 1:
            return "COLLISION"  # Collision occurred
        elif len(nodes) == 1:
            return "ACK"        # Successful transmission
        else:
            return "IDLE"       # No transmission

    def reset(self):
        self.channel_state = "IDLE"
        self.channel_busy = False ##채널 상태만 초기화 


class QLBT_DCAEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 10}  
    RENDER_SLOTS = 50 
    NOTEBOOK = 'ipykernel' in sys.modules 
    CHANNEL_STATE_MAP = {
        "IDLE": 0,
        "ACK": 1,
        "COLLISION": 2
    }

    def __init__(self, trainer: QLBT_AP, max_steps=1000, render_mode=None):
        super().__init__()
        self.num_nodes = trainer.n_agents
        self.max_steps = max_steps
        self.device = trainer.device
        self.access_point = AccessPoint()
        self.t = 0
        self.d2lt = np.zeros(self.num_nodes)
        self.joint_action = np.zeros(self.num_nodes)
        self.success_count = np.zeros(self.num_nodes)
        self.render_mode=render_mode
        self.trainer = trainer
        self.agents = trainer.agent_nets

        for agent in self.agents:
            agent.reset_hidden(bs=1)

        self._initialize_render_data()

    def reset(self): # 환경 초기화
        self.t = 0
        self.access_point.reset()
        self.d2lt = np.zeros(self.num_nodes)
        self.joint_action = np.zeros(self.num_nodes)
        self.success_count = np.zeros(self.num_nodes)
        for agent in self.agents:
            agent.reset_hidden(bs=1)
        self._initialize_render_data()
        return self._get_obs(), {} # 관측값과 정보 리턴

    def step(self, actions): # 액션에 따라 환경 상태 업데이트
        self.joint_action = np.array(actions)
        ready_nodes = np.where(self.joint_action == 1)[0].tolist()
        channel=self.access_point.receive(ready_nodes) # 채널 상태 확인

        rwd_tot, rwd_ind = self._compute_reward(channel, ready_nodes) # 보상 계산
        self.t += 1
        self.d2lt += 1 ###################### 패킷 전송 성공한 노드는 0으로 리셋 필요

        if channel == "ACK":
            winner = ready_nodes[0]
            self.d2lt[winner] = 0

        obs = self._get_obs() # 관측값 업데이트
        done = self.t >= self.max_steps
        info = {"ready nodes": ready_nodes, "channel": channel}
        
        # Update render data
        self.state_data = np.zeros((self.num_nodes, 1))
        if channel == 'ACK':
            self.state_data[ready_nodes[0]] = 1
        elif channel == 'COLLISION':
            for node in ready_nodes:
                self.state_data[node] = 2
        
        self.render_data['time_node'] = np.concatenate(
            (self.render_data['time_node'][:, 1:], self.state_data), axis=1
        )
        
        if self.render_mode == 'rgb_array' or self.render_mode == 'human':
            self.render_data['cumsum_success'].append(self.render_data['success'])
            self.render_data['cumsum_collision'].append(self.render_data['collision'])

        return obs, (rwd_tot, rwd_ind), done, False, info

    def get_state(self):
        channel_state = self.CHANNEL_STATE_MAP[self.access_point.channel_state]
        return np.concatenate((
            [channel_state],
            self.success_count/(self.t+1e-5),
            self.joint_action, 
            self.d2lt
        ))
    
    def _get_obs(self): # agent가 받는 관측값 
        obs = []
        channel_state = self.CHANNEL_STATE_MAP[self.access_point.channel_state]
        collision = 1 if self.access_point.channel_state == "COLLISION" else 0

        for i in range(self.num_nodes):
            own_d2lt = self.d2lt[i]
            others_d2lt = np.delete(self.d2lt, i)
            agent_obs = np.concatenate((
                [channel_state],
                [collision],
                [own_d2lt],
                others_d2lt
            ))
            obs.append(agent_obs)

        return np.array(obs)
    
    def _compute_reward(self, channel, ready_nodes):
        self.render_data['time_slot'] += 1
        reward_ind = np.zeros(self.num_nodes)
        reward_total = 0.0

        if channel == "ACK":
            winner = ready_nodes[0]
            self.render_data['success'] += 1
            self.render_data['node_success'][winner] += 1
            
            reward_total = 1.0
            reward_ind[winner] = 1.0 / (self.success_count[winner] + 1e-5)  # PF fairness
            self.d2lt[winner] = 0
            return reward_total, reward_ind
        
        elif channel == "COLLISION":
            self.render_data['collision'] += 1
            for node in ready_nodes:
                self.render_data['node_collision'][node] += 1
                reward_ind[node] = -1.0 / (self.success_count[node] + 1e-5)
            return -1.0, reward_ind
        
        else:
            return 0.0, reward_ind
        
    def render(self):
        if not self.render_mode:
            return
        
        time_node_data = self.render_data['time_node']
        
        if self.render_mode == "ansi":
            if self.NOTEBOOK:
                display.clear_output(wait=True)
            # print("*" * 30 + f"{'RENDERING DCA(t = ' + str(self.t) + ')':^40}" + "*" * 30)
            # for i, row in enumerate(time_node_data):
            #     print(f"Node {i+1} (D2LT={self.d2lt[i]:>4.1f}): ", end=' ')
            #     for rc in row:
            #         CCOLOR = '\033[44m' if rc == 1 else '\033[101m' if rc == 2 else '\33[7m'
            #         CEND = '\33[0m'
            #         print(f"{CCOLOR}{int(rc)}{CEND}", end=' ')
            #     print()
            print(f"Time Slot: {self.t}, "
                  f"Success: {self.render_data['success']} ({self.render_data['node_success'].squeeze()}), "
                  f"Collision: {self.render_data['collision']}, "
                  f"Throughput: {self.render_data['success'] / self.t if self.t > 0 else 0:.3f}")
            time.sleep(0.1)

        elif self.render_mode in ["rgb_array", "human"]:
            plt.clf()
            cmap = colors.ListedColormap(['white', 'blue', 'red', 'gray'])  # Gray for debugging
            norm = colors.BoundaryNorm([0, 1, 2, 3, 4], cmap.N)

            plt.subplot(211)
            plt.title(f"DCA (t = {self.t}, throughput: {self.render_data['success'] / self.render_data['time_slot']:.3f})")
            plt.imshow(time_node_data, cmap=cmap, norm=norm, aspect='auto')
            plt.ylabel('Node ID')

            plt.subplot(223)
            plt.plot(self.render_data['cumsum_success'], 'b-', label='Success')
            plt.plot(self.render_data['cumsum_collision'], 'r-', label='Collision')
            plt.legend()
            plt.grid(True)

            plt.subplot(224)
            plt.bar(range(self.num_nodes), self.render_data['node_success'].squeeze())
            plt.ylabel('Success Count')

            if self.NOTEBOOK:
                display.clear_output(wait=True)
                display.display(plt.gcf())
            else:
                plt.pause(0.001)
    
    def _initialize_render_data(self):
        self.render_data = {
            'time_node': np.zeros((self.num_nodes, self.RENDER_SLOTS)),
            'node_success': np.zeros((self.num_nodes, 1), dtype=int),
            'node_collision': np.zeros((self.num_nodes, 1), dtype=int),
            'time_slot': 0,
            'success': 0,
            'collision': 0,
            'cumsum_success': [],
            'cumsum_collision': [],
        }


if __name__ == "__main__":
    
    num_nodes = 5
    max_steps = 1000
    obs_dim = num_nodes + 2  # [channel_state, own_d2lt, others_d2lt...]
    state_dim = 1 + 3 * num_nodes  # [channel_state, success_ratio, joint_action, d2lt]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    trainer = QLBT_AP(
        n_agents=num_nodes,
        obs_dim=obs_dim,
        state_dim=state_dim,
        hidden_dim = 64,
        buffer_size=5000,
        batch_size=64,
        gamma=0.99,
        lr=0.001,
        device=device
    )
    env = QLBT_DCAEnv(trainer, max_steps=max_steps, render_mode='human')
    # agent = QLBT_gym(num_nodes, obs_dim=obs_dim, state_dim=state_dim, device=device)

    total_reward = 0
    obs, _ = env.reset()

    for t in range(max_steps):
        actions = [agent.select_action(obs[i]) for i, agent in enumerate(env.agents)]
        next_obs, (r_tot, r_ind), done, _, _ = env.step(actions)

        state = torch.tensor(env.get_state(), dtype=torch.float32).to(device)
        next_state = torch.tensor(env.get_state(), dtype=torch.float32).to(device)

        obs_tensor = [torch.tensor(o, dtype=torch.float32) for o in obs]
        next_obs_tensor = [torch.tensor(o, dtype=torch.float32) for o in next_obs]
        hidden_tensor = [agent.hidden_state.detach().squeeze(0) for agent in env.agents]

        trainer.store_transition((
            state,obs_tensor,hidden_tensor,actions, r_tot,r_ind,
            next_state, next_obs_tensor, hidden_tensor, done
        ))
        # for i in range(num_nodes):
        #     obs_tensor = torch.tensor(obs[i], dtype=torch.float32)
        #     next_obs_tensor = torch.tensor(next_obs[i], dtype=torch.float32)
        #     trainer.store_transition((state, obs_tensor, env.agents[i].hidden_state.detach(), actions[i],
        #                                r_tot, r_ind, next_state, next_obs_tensor, env.agents[i].hidden_state.detach(), done))

        trainer.train()
        obs = next_obs
        total_reward += r_tot
        env.render()
        if done:
            break

    if not env.NOTEBOOK:
        plt.show()

    print(f"✅ Average episode reward (QLBT): {total_reward / max_steps:.3f}")
