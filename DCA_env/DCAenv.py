import gymnasium as gym 
import numpy as np
import matplotlib
matplotlib.use('QtAgg')  # Use a non-interactive backend for rendering
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import random 
import sys 
import time 

if 'ipykernel' in sys.modules:
    from IPython import display

class AccessPoint: 
    """ 
    Handles channel access with multi-slot transmission 
    """
    SIF = 2         # time includes ACK returning slot 
    def __init__(self, ):
        self.channel_busy = False 
        self.current_transmitter = None   # ID of the node of current transmission 
        self.remaining_slots = 0          # Number of remaining slots for current transmission
        self.sif = self.SIF
        self.sif_mode = False 
        self.delay = 0 


    def receive(self, nodes, packet_length): 
        """ 
        Process transmission attempts and update channel state 

        Args: 
            nodes (list): List of node IDs attempting to transmit 
            packet_length (int): Length of the packet to be transmitted 
        
        Returns: 
            str: Channel state ('ACK', 'COLLISION', 'IDLE', or 'OCCUPIED')
        """

        if self.channel_busy:
            # print(f"Channel busy, remaining slots: {self.remaining_slots}, SIF: {self.sif}")
            if len(nodes) > 0: 
                self.reset()
                return "COLLISION"              # Interference during ongoing transmission.
            if self.remaining_slots >= 0 and self.sif > 0:              # Channel is busy but there are remaining slots for the ongoing transmission.
                if self.remaining_slots == 0: 
                    self.sif_mode = True 
                return "OCCUPIED"
            elif self.remaining_slots == 0 and self.sif == 0:
                # print("fjfjfjfjfjfjfjfjfjfjf")
                self.sif_mode = False 
                self.channel_busy = False
                # print("SIF=",self.sif, "SIF_MODE=",self.sif_mode)
                return "ACK" 
            # else: 
            #     return "ACK" 
        else:
            # print(f"Channel idle")
            if len(nodes) > 1:
                self.delay += 1
                return "COLLISION"              # Multiple nodes transmit simultaneously 
            elif len(nodes) == 1: 
                self.sif = self.SIF
                self.channel_busy = True 
                self.current_transmitter = nodes[0] 
                self.remaining_slots = packet_length
                return "OCCUPIED" 
            else:
                self.delay += 1
                return "IDLE"
    
    def update(self): 
        """Decrement remaining slots and update channel state"""
        
        if self.channel_busy and self.remaining_slots > 0:
            self.remaining_slots -= 1 
        if self.remaining_slots == 0 and self.sif > 0: 
            self.sif -= 1 
        # if self.remaining_slots == 0 and self.sif == 0: 
        #     self.channel_busy = False 
            # self.current_transmitter = None 
    
    def reset(self): 
        """Reset channel """
        self.channel_busy = False 
        self.current_transmitter = None 
        self.remaining_slots = 0 
        self.sif = self.SIF



class DCAEnv(gym.Env): 
    """ 
    Decentralized POMDP for DCA, for multi-slot packet transmission and listen-before-talk (LBT) 

    Each node (agent) decides wheter to transmit(=1) or wait(=0) in each time slot if the channel is "IDLE"
    Transmission lasts for multiple time slots, and success requires no interference from other nodes 
    "ACK" issued only when the full packet has fully arrived.
    """
    metadata = {"render_modes": ["human", "ansi", "rgb_array"]}
    RENDER_SLOTS = 200
    NOTEBOOK = 'ipykernel' in sys.modules 
    CHANNEL_STATE_MAPS = {"IDLE": 0, "OCCUPIED": 1, "COLLISION": 2, "ACK": 3}

    def __init__(self, 
                 num_agents=5, 
                 max_cycles=1000, 
                 packet_length=3,
                 render_mode=None):
        """ 
        Initialize the environment 

        Args: 
            num_agents (int): Number of nodes in the environment 
            max_steps (int): Maximum number of steps (time slots) the episode can last 
            packet_length (int): Number of time slots for each transmission 
            render_mode (str): The mode to render the environment (None, "human"="rgb_array", "ansi")

        """
        super().__init__() 

        self.num_agents = num_agents
        self.max_cycles = max_cycles 
        self.packet_length = packet_length
        self.render_mode = render_mode.lower() if render_mode else None 
        self.access_point = AccessPoint() 
        self.t = 0 

        self.agents = [f"agent_{i}" for i in range(num_agents)] 
        self.action_spaces = {agent_id: gym.spaces.Discrete(2) for agent_id in self.agents} 
        self.observation_spaces = {
            agent_id: gym.spaces.Dict({"channel_state": gym.spaces.Discrete(4), 
                                       "collision": gym.spaces.Discrete(2),
                                       "d2lt": gym.spaces.Box(low=0, high=1, shape=(1,))}) 
            for agent_id in self.agents 
        }


        self.hidden_state = {
            "channel": "IDLE", 
            "ready_nodes": [], 
            "successful": False,    # Tracks if the current packet transmission was successful 
        }

        self._initialize_render_data() 


    def reset(self, seed=None, options=None): 
        """
        Reset the environment 
        
        Args: 
            seed (int): Random seed for reproducibility 
            options (dict): additional options 
            
        Returns: 
            observation (dict): Initial observation 
            info (dict): Additional information 
        """
        super().reset(seed=seed) 
        if seed is not None:
            np.random.seed(seed) 
        
        self.t = 0 
        # Initialize access point and hidden state 
        self.access_point.reset() 
        self.hidden_state = {"channel": "IDLE", 
                             "ready_nodes": [], 
                             "successful": False, 
                             "D2LT": [0] * self.num_agents, # D2LT: number of time slots since last successful transmission 
                             "others": [0] * self.num_agents, # others: Indicator of other nodes (transmitting = 1, waiting = 0)
                             }
        self._initialize_render_data()  

        observation = self._get_observation() 
        infos = {agent_id: {"hidden_state": self.hidden_state.copy()} for agent_id in self.agents} 

        return observation, infos     


    def step(self, actions): 
        """Execute one step with joint actions 
        
        Args: 
            actions (dict): {agent_id: action} where action is 0 (wait) or 1 (transmit) 
            
        Returns: 
            observations (dict): Observations for each agent 
            rewards (dict): rewards per agent 
            terminated (dict): whether the episode is terminated 
            truncated (dict): whether the episode is truncated 
            info (dict): additional information (e.g., hidden_state)
        """

        action_array = np.array([actions[agent_id] for agent_id in self.agents])

        ready_nodes = np.where(action_array == 1)[0].tolist() 

        # Update channel state 
        channel = self.access_point.receive(ready_nodes, self.packet_length) 
        # print(f"Left packets: {self.access_point.remaining_slots}")

        self.hidden_state["channel"] = channel 
        self.hidden_state["ready_nodes"] = ready_nodes

        # Update D2LT and actions 
        for i in range(self.num_agents): 

            self.hidden_state["D2LT"][i] += 1 
            self.hidden_state["others"][i] = 1 if actions[self.agents[i]] == 1 else 0 

        # Update the ongoing packet transmission 
        if channel == "OCCUPIED": 
            self.hidden_state["successful"] = True 
        elif channel == "COLLISION": 
            self.hidden_state["successful"] = False 
        elif channel == "ACK":
            self.hidden_state["successful"] = True 

        rewards = self._compute_rewards() 
        self.access_point.update() 

        observations = self._get_observation()  
        self.t += 1 
        terminations = {agent_id: False for agent_id in self.agents}
        truncations = {agent_id: self.t >= self.max_cycles for agent_id in self.agents} 

        # update render data 
        self._update_render_data() 

        infos = {agent_id: {"hidden_state": self.hidden_state.copy()} for agent_id in self.agents}

        return observations, rewards, terminations, truncations, infos 


    def render(self):
        """Render th environment in accordance to the render_mode 
        
        Returns:
            None 
        """
        if not self.render_mode: 
            return 
        
        time_agent_data = self.render_data["time_agent"] 

        if self.render_mode == "ansi": 
            self._render_ansi(time_agent_data) 
        elif self.render_mode in ["human", "rgb_array"]:
            self._render_human(time_agent_data)


    def observation_space(self, agent): 
        return self.observation_spaces[agent]
    
    def action_space(self, agent): 
        return self.action_spaces[agent]


    def _get_observation(self): 
        """Get the current observation from the environment
        
        Returns: 
            observation (dict): Observation for each agent 
        """
        # ready_nodes = self.hidden_state["ready_nodes"]
        # print(self.hidden_state)
        observations = {} 
        channel = self.hidden_state["channel"] 
        channel_state = self.CHANNEL_STATE_MAPS[channel]  
        collision = 1 if channel == "COLLISION" else 0
        d2lt = self.hidden_state["D2LT"] 
        # normalize d2lt 
        # print(d2lt, end='')
        d2lt = [d2lt[i] / (sum(d2lt) + 1e-6) for i in range(self.num_agents)]
        # print(d2lt, end=' ')
        ### TODO: personalize by agent (when including D2LT)### 
        for i, agent_id in enumerate(self.agents): 
            observations[agent_id] = {"channel_state": channel_state,                          
                                      "collision": collision,
                                      "d2lt": d2lt[i]}
        #     print(f"{observations[agent_id]['d2lt']:.2f}", end=' ')
        # print()
        return observations 
    
    def _compute_rewards(self): 
        """Compute rewards based on state (observation + hidden_state)
        
        Returns: 
            rewards (dict): rewards per agent 
        """
        rewards = {agent_id: 0 for agent_id in self.agents}
        ready_nodes = self.hidden_state["ready_nodes"] 
        channel = self.hidden_state["channel"] 
        self.render_data["time_slot"] += 1 
        if channel == "ACK" and self.hidden_state["successful"]: 
            # For packet-based counts 
            self.render_data["success"] += 1 
            self.render_data["agent_success"][self.access_point.current_transmitter] += 1 
            # For bit-based counts 
            self.render_data["success_packet"] += (self.packet_length + self.access_point.sif)
            self.render_data["agent_success_packet"][self.access_point.current_transmitter] += (self.packet_length + self.access_point.sif)
            # rewards[self.agents[self.access_point.current_transmitter]] = 1 / self.hidden_state["D2LT"][self.access_point.current_transmitter]
            rewards = {agent_id: 1.0 for agent_id in self.agents}
            # Fairness handicap according to D2LT 
            for i in range(self.num_agents): 
                rewards[self.agents[i]] -= (self.hidden_state["D2LT"][i] * 10 / self.max_cycles)
            # print(rewards)
            self.hidden_state["D2LT"][self.access_point.current_transmitter] = 0 
            self.hidden_state["others"][self.access_point.current_transmitter] = 0 
            
        elif channel == "COLLISION": 
            self.render_data["collision"] += 1 
            for node in ready_nodes: 
                self.render_data["agent_collision"][node] += 1 
                rewards[self.agents[node]] = -1.0 
                
        return rewards 

    def _initialize_render_data(self): 
        """Initialize data for rendering """
        self.render_data = {
            "time_agent": np.zeros((self.num_agents, self.RENDER_SLOTS)), 
            "agent_success": np.zeros((self.num_agents, 1), dtype=int), 
            "agent_collision": np.zeros((self.num_agents, 1), dtype=int), 
            "agent_success_packet": np.zeros((self.num_agents, 1), dtype=int), 
            "time_slot": 0,
            "success": 0, 
            "success_packet": 0, 
            "collision": 0, 
            "cumsum_success": [], 
            "cumsum_success_packet": [], 
            "cumsum_collision": [],
            "throughput": [], 
            "throughput_packet": [],
            "delay": []
        }

    def _update_render_data(self): 
        """Update the render data based on the current state"""
        self.state_data = np.zeros((self.num_agents, 1))
        ready_nodes = self.hidden_state["ready_nodes"] 
        channel = self.hidden_state["channel"] 

        if channel == "ACK": 
            self.state_data[self.access_point.current_transmitter] = 3 
        elif channel == "COLLISION": 
            for node in ready_nodes: 
                self.state_data[node] = 2  
        elif channel == "OCCUPIED": 
            # if self.access_point.remaining_slots >= 0 and self.access_point.sif > 0:
            if self.access_point.remaining_slots >= 0 and not self.access_point.sif_mode:
                self.state_data[self.access_point.current_transmitter] = 1 
            elif self.access_point.remaining_slots > 0 and len(ready_nodes) > 0: 
                for node in ready_nodes: 
                    self.state_data[node] = 2 
        self.render_data["time_agent"] = np.concatenate(
            (self.render_data["time_agent"][:, 1:], self.state_data), axis=1
        )
        self.render_data["delay"].append(np.mean(self.hidden_state["D2LT"]))
        if self.render_mode in ["human", "rgb_array"]: 
            self.render_data["cumsum_success"].append(self.render_data["success"])
            self.render_data["cumsum_success_packet"].append(self.render_data["success_packet"])
            self.render_data["cumsum_collision"].append(self.render_data["collision"])
            self.render_data["throughput"].append(self.render_data["success"] / self.t)
            self.render_data["throughput_packet"].append(self.render_data["success_packet"] / self.t)
            


    def _render_ansi(self, time_node_data): 
        """Render the environment in ANSI mode""" 
        if self.NOTEBOOK: 
            display.clear_output(wait=True)  
        print("*" * 35 + f"{'RENDERING DCA(t = ' + str(self.t) + ')':^40}" + "*" * 35)
        for i, row in enumerate(time_node_data):
            print(f"Node {i+1} (D2LT={self.hidden_state['D2LT'][i]:>3d}): ", end=' ')
            for rc in row:
                CCOLOR = '\033[44m' if rc == 1 else '\033[101m' if rc == 2 else '\033[42m' if rc == 3 else '\33[7m'
                CEND = '\33[0m'
                print(f"{CCOLOR}{int(rc)}{CEND}", end=' ')
            print()
        print(f"Time Slot: {self.t}, "
                f"Success: {self.render_data['success']}, "
                f"Collision: {self.render_data['collision']}, "
                f"Throughput: {self.render_data['success_packet'] / self.t if self.t > 0 else 0:.3f}, " 
                # f"Mean Delay: {self.render_data['delay'][-1]}"
                f"Mean Delay: {np.mean(self.hidden_state['D2LT'])}"
                )
        print("*" * 110)
        print()
        time.sleep(0.1)
    
    def _render_human(self, time_node_data):
        plt.clf()
        cmap = colors.ListedColormap(['white', 'blue', 'red', 'lightgreen'])
        norm = colors.BoundaryNorm([0, 1, 2, 3, 4], cmap.N)

        plt.subplot(311)

        plt.title(f"DCA (t = {self.t}, throughput: {self.render_data['success_packet'] / self.render_data['time_slot']:.3f})")
        plt.imshow(time_node_data, cmap=cmap, norm=norm, aspect='auto')
        plt.ylabel('Node ID')

        plt.subplot(323)
        # plt.plot(self.render_data['cumsum_success'], 'b-', label='Success')
        plt.plot(self.render_data['cumsum_success_packet'], 'b-', label='Success')
        plt.plot(self.render_data['cumsum_collision'], 'r-', label='Collision')
        plt.legend()
        plt.grid(True)

        plt.subplot(324)
        plt.bar(range(self.num_agents), self.render_data['agent_success_packet'].squeeze())
        plt.ylabel('Success Count')

        plt.subplot(313)
        # plt.plot(self.render_data["throughput"], 'b-', label='Throughput')
        plt.plot(self.render_data["throughput_packet"], 'b-', label='Throughput Packet')
        plt.legend()
        plt.grid(True)


        if self.NOTEBOOK:
            display.clear_output(wait=True)
            display.display(plt.gcf())
        else:
            plt.pause(0.001)




class CSMA_CA_Agent(): 
    """
    CSMA/CA Agent (Multi-agent style) with listen-before-talk (LBT) 
    """
    def __init__(self, 
                 agent_id, 
                 cw_min=2, 
                 cw_max=16):
        self.agent_id = agent_id 
        self.cw_min = cw_min 
        self.cw_max = cw_max 
        self.current_cw = cw_min 
        self.backoff_timer = random.randint(1, self.current_cw) 
    
    def act(self, observation): 
        """
        Decide whether to wait(0) or transmit(1) based on the observation 
        """
        channel_state = observation["channel_state"] 
        collision = observation["collision"] 

        if channel_state == 0:   # IDLE 
            if self.backoff_timer > 0: 
                self.backoff_timer -= 1 
            elif self.backoff_timer == 0: 
                return 1   # Transmit   
        else:   # BUSY, ACK, or COLLISION 
            if self.backoff_timer == 0: 
                if collision == 1: 
                    self.current_cw = min(self.cw_max, self.current_cw * 2) 
                self.backoff_timer = random.randint(1, self.current_cw) 
            elif self.backoff_timer > 0: 
                self.backoff_timer -= 1 
        return 0 



        
if __name__ == "__main__":  
    num_agents = 3
    num_cycles = 1000
    packet_length = 10
    render_mode = "human"
    env = DCAEnv(num_agents=num_agents, max_cycles=num_cycles, packet_length=packet_length, render_mode=render_mode) 

    legacy_agents = {f"agent_{i}": CSMA_CA_Agent(i, cw_min=2, cw_max=32) for i in range(num_agents)}

    total_reward = 0 

    obs, _ = env.reset() 
    rewards_per_agent = {agent_id: 0 for agent_id in env.agents}
    for i in range(num_cycles): 
        # actions = {agent_id: env.action_spaces[agent_id].sample() for agent_id in env.agent_ids}
        actions = {agent_id: legacy_agents[agent_id].act(obs[agent_id]) for agent_id in env.agents}
        next_obs, rewards, terminated, truncated, info = env.step(actions) 
        env.render()
        total_reward += sum(rewards.values()) 
        rewards_per_agent = {agent_id: rewards_per_agent[agent_id] + rewards[agent_id] for agent_id in env.agents}
        obs = next_obs 
    
        if any(terminated.values()) or any(truncated.values()): 
            break 
    
    if not env.NOTEBOOK: 
        plt.show()

    avg_rewards_per_agent = {agent_id: rewards_per_agent[agent_id] / num_cycles for agent_id in env.agents}
    print(f"Avg rewards: {total_reward / num_cycles}")
    print(f"Avg delay: {np.mean(env.render_data['delay'])}")
    print(f"Avg rewards per agent: {avg_rewards_per_agent}")