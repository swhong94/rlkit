import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('QtAgg')  
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import random
import sys
import time

if 'ipykernel' in sys.modules:
    from IPython import display

class AccessPoint: # CSMA/CA 환경의 AP는 simple하게 구현
    """
    Access Point (AP) for the CSMA/CA environment.
    This class simulates the behavior of an access point in a CSMA/CA network.
    It receives packets from nodes and determines the channel state (IDLE, ACK, COLLISION).
    """
    # packet-sifs-ack
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

class DCAEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 10}  
    RENDER_SLOTS = 50 
    NOTEBOOK = 'ipykernel' in sys.modules 
    CHANNEL_STATE_MAP = {
        "IDLE": 0,
        "ACK": 1,
        "COLLISION": 2
    }
    
    def __init__(self, num_nodes=5, max_steps=1000, render_mode=None):
        super().__init__() 

        self.num_nodes = num_nodes  # 네트워크 참여 노드 수 
        self.max_steps = max_steps  # 최대 time slot 수
        self.render_mode = render_mode.lower() if render_mode else None
        self.access_point = AccessPoint()
        self.t = 0

        # Define action and observation spaces
        self.action_space = gym.spaces.MultiBinary(num_nodes) # joint action space: 0 or 1 for each node
        

        # Observation space only includes channel state and collision info 
        self.observation_space = gym.spaces.Dict({    # joint observation space: AP to agent [channel state, D2LT]:MARL
            "channel_state": gym.spaces.Discrete(3),  # 0=IDLE, 1=ACK, 2=COLLISION -> channel_state
            "collision": gym.spaces.Discrete(2)       # 0=False, 1=True -> collision_occured
        })

        # Hidden state (not directly observable to agents) 
        self.hidden_state = {  
            "channel": "IDLE",        # Channel state 
            "ready_nodes": [],      # Ready Nodes
            "D2LT": [],             # Number of timeslots to last successful transmission
            "actions": [],          # Agent Actions 
            "others": [],           # Indicator of other nodes transmission (0: no transmission, 1: transmission)
        }

        self._initialize_render_data()


    def reset(self, ): # 환경 초기화
        self.t = 0
        self.access_point.reset()
        self.d2lt = np.zeros(self.num_nodes) 
        self.others = np.zeros(self.num_nodes) 
        self.hidden_state = {"channel": "IDLE",
                             "ready_nodes": [], 
                             "D2LT": self.d2lt, 
                             "actions": [0] * self.num_nodes, 
                             "others": self.others}
        self._initialize_render_data()

        # Get initial state and info 
        observation = self._get_obs() 
        info = {"hidden_state": self.hidden_state.copy()} 

        return observation, info


    def step(self, actions): # 액션에 따라 환경 상태 업데이트
        ready_nodes = np.where(actions == 1)[0].tolist()    # 액션이 1인 노드들(전송 시도한 노드들)을 리스트로 변환
        channel = self.access_point.receive(ready_nodes)    # AP에 전송시도 노드 전달해서 채널 상태 확인
        self.d2lt = self.d2lt + 1   
        self.others = np.zeros(self.num_nodes) 
        # for i in range(self.num_nodes):
        #     self.others[i] = np.sum([actions[j] for j in range(self.num_nodes) if j != i])  # Exclude the agent's own action
        
        # Compute reward 
        reward = self._compute_reward(channel, ready_nodes) #채널 상태와 전송 시도 노드에 따라 보상 계산
        self.hidden_state = {
            "channel": channel,
            "ready_nodes": ready_nodes,
            "D2LT": self.d2lt, 
            "actions": actions,
            "others": self.others,
        }

        # Get observation from AP to agent
        observation = self._get_obs() 

        # Update time and termination conditions 
        self.t += 1
        terminated = False 
        truncated = self.t >= self.max_steps 
        info = {
            "hidden_state": self.hidden_state.copy(),
        }

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


        return observation, reward, terminated, truncated, info


    def _get_obs(self, ): # agent가 받는 관측값 
        channel = self.hidden_state["channel"]
        channel_state = self.CHANNEL_STATE_MAP[channel]

        # D2LT (Number of time slots from last successful transmission)

        collision = 1 if channel == "COLLISION" else 0 
        return {"channel_state": channel_state, "collision": collision} 
    

    def _compute_reward(self, channel, ready_nodes):
        self.render_data['time_slot'] += 1
        self.others = np.ones(self.num_nodes)
        if channel == 'ACK':
            self.render_data['success'] += 1
            self.render_data['node_success'][ready_nodes[0]] += 1
            self.d2lt[ready_nodes[0]] = 0
            self.others[ready_nodes[0]] = 0
            return 1.0
        elif channel == 'COLLISION':
            self.render_data['collision'] += 1
            for node in ready_nodes:
                self.render_data['node_collision'][node] += 1
            return -1.0
        else: # channel == 'IDLE'
            self.others = np.zeros(self.num_nodes)
        return 0.0

    def render(self):
        if not self.render_mode:
            return
        
        time_node_data = self.render_data['time_node']
        
        if self.render_mode == "ansi":
            if self.NOTEBOOK:
                display.clear_output(wait=True)
            print("*" * 30 + f"{'RENDERING DCA(t = ' + str(self.t) + ')':^40}" + "*" * 30)
            for i, row in enumerate(time_node_data):
                print(f"Node {i+1} (D2LT={self.d2lt[i]:>4.1f}): ", end=' ')
                for rc in row:
                    CCOLOR = '\033[44m' if rc == 1 else '\033[101m' if rc == 2 else '\33[7m'
                    CEND = '\33[0m'
                    print(f"{CCOLOR}{int(rc)}{CEND}", end=' ')
                print()
            print(f"Time Slot: {self.t}, "
                  f"Success: {self.render_data['success']} ({self.render_data['node_success'].squeeze()}), "
                  f"Collision: {self.render_data['collision']}, "
                  f"Throughput: {self.render_data['success'] / self.t if self.t > 0 else 0:.3f}")
            time.sleep(0.1)

        elif self.render_mode == "rgb_array" or self.render_mode == "human":
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







class CSMA_gym():
    """
    action returns a value instead of a singleton list
    """
    def __init__(self, agent_id, cw_min=2, cw_max=16): 
        super().__init__(agent_id, cw_min, cw_max) 

    def act(self, observation): 
        action = super().act(observation) 
        return action[0]







class CSMA_CA_Agent: 
    """
    CSMA-CA Agent for the CSMA/CA environment.
    This agent implements the CSMA-CA protocol with two strategies: BEB (Binary Exponential Backoff) and Random Backoff.
    """
    
    def __init__(self, node_id, cw_min, cw_max, strategy='beb'):
        self.node_id = node_id  #노드 id
        self.cw_min = cw_min    #최소 대기 시간
        self.cw_max = cw_max    #최대 대기 시간
        self.current_cw = cw_min #현재 대기 시간
        self.strategy = strategy #전송 전략 beb, random
        self.backoff_timer = 0  
        self.action = [0]       # Initialize action = [wait, transmit]
        self.set_new_backoff()  # Set backoff timer

    def set_new_backoff(self):
        if self.strategy.lower() == 'random':
            self.backoff_timer = random.randint(1, self.cw_max) 
        elif self.strategy.lower() == 'beb':
            self.backoff_timer = random.randint(1, self.current_cw)

    def reset_backoff(self, collision_occured):     #충돌 발생 시 함수 
        if self.strategy.lower() == 'beb':          #beb일때만 충돌 영향있음 
            if collision_occured:   
                self.current_cw = min(self.current_cw * 2, self.cw_max)   # 2배, 최대값 중에 최소값 사용
            else:
                self.current_cw = self.cw_min
        self.set_new_backoff()

    def decrement_backoff(self):  # 대기시간 감소
        if self.backoff_timer > 0:  # 대기시간 >0 일때만 감소
            self.backoff_timer -= 1

    def ready(self): 
        return self.backoff_timer == 0

    def act(self, state):
        channel_state, collision_occured = state  # global state로부터 채널 상태, 충돌 발생 여부를 관찰  
        
        # Update backoff based on previous action and channel state
        if self.action[0] == 1:  # 전송 시도했을 때 충돌 발생 여부 부 확인 
            self.reset_backoff(collision_occured=collision_occured)
        else:   # 이전 액션이 대기라면
            self.decrement_backoff() # 대기시간 감소

        # Set new action
        self.action = [1] if self.ready() else [0] # 노드 전송 준비 상태일경우 전송 -> backoff timer=0
        return self.action

    def __repr__(self): #"node id: READY or WAITING (backoff timer / current cw)"
        return f"{self.node_id}: {'READY' if self.ready() else 'WAITING'} ({self.backoff_timer}/{self.current_cw})"
    


if __name__ == "__main__":  # CSMA/CA agent test
    num_nodes = 5
    max_steps = 1000
    env = DCAEnv(num_nodes=num_nodes, max_steps=max_steps, render_mode='human')
    agents = [CSMA_CA_Agent(i, cw_min=2, cw_max=16) for i in range(num_nodes)]
    
    total_reward = 0

    observation, _ = env.reset() 
    actions_csma = np.array([agent.act(observation) for agent in agents])
 
    for _ in range(max_steps):
        action = np.array([agent.act(observation) for agent in agents])
        next_obs, reward, terminated, truncated, info = env.step(action) 
        env.render()
        total_reward += reward
        observation = next_obs
        if terminated or truncated: #window+c -> 종료 
            break

    if not env.NOTEBOOK:
        plt.show()

    print(f"average episode reward (CSMA): {total_reward / max_steps}")