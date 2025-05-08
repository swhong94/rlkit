import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

import gymnasium as gym
# from RL_algorithm.DQN import DQN
import matplotlib.pyplot as plt

class MLP(nn.Module): # nn.Module(Relu, Linear, Sequential 함수)을 상속
    def __init__(self, state_dim, action_dim, hidden_dim = 128): 
        super(MLP, self).__init__() 
        self.activation = nn.ReLU() # ReLU: 비선형 활성화 함수
        self.fc1 = nn.Linear(state_dim, hidden_dim) # 입력층(state) -> 은닉층
        self.fc2 = nn.Linear(hidden_dim, action_dim) # 은닉층 -> 출력층(action)
        self.mlp = nn.Sequential( # nn.Sequential을 이용해 MLP 구조 정의
            self.fc1,
            self.activation,
            self.fc2
        ) 
        # self.변수명: 클래스 내에서 사용하는 변수를 의미

    def forward(self, x):  # forward 함수 정의: 입력 x를 받아 MLP를 통과시키는 함수
        return self.mlp(x)

class ReplayBuffer:
    def __init__(self, capacity = 10000): # 초기화할 때 buffer의 크기 10000을 기본으로 지정
        self.buffer = deque(maxlen=capacity)  

    def push(self, state, action, reward, next_state, done):  
        if len(self.buffer) == self.buffer.maxlen:
            self.buffer.popleft() # 버퍼가 가득 찼을 때 가장 오래된 데이터 제거
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size): 
        batch = random.sample(self.buffer, batch_size)  # buffer에서 batch_size만큼 랜덤하게 샘플링
        states, actions, rewards, next_states, dones = zip(*batch)  # 샘플링한 데이터를 states, actions, rewards, next_states, dones로 각각 zip함
        return (np.stack(states),
                np.array(actions),
                np.array(rewards),
                np.stack(next_states),
                np.array(dones))

    def __len__(self):
        return len(self.buffer)

class DQN: 
    def __init__(self, state_dim, action_dim, hidden_dim = 128, buffer_capacity = 10000, 
                 lr = 1e-3, gamma = 0.99, eps = 1.0, eps_min = 0.01, eps_decay = 0.995, 
                 update_frequency = 100, target_net_hard_update = True, device= None):
        super(DQN, self).__init__()

        if device is None:
            self.device = 'cpu' # 강화학습에서는 CPU로 학습하는 것이 일반적
        else:
            self.device = device

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_net = MLP(state_dim, action_dim, hidden_dim).to(self.device) # DQN에서는 main-network,
        self.q_target = MLP(state_dim, action_dim, hidden_dim).to(self.device)  # target-network 두 개 정의

        self.replay_buffer = ReplayBuffer(buffer_capacity)  # replay buffer 정의

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr, weight_decay=1e-3)  #최적화 함수: Adam, learning rate: 1e-3, weight_decay: 1e-3
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.update_frequency = update_frequency
        self.target_net_hard_update = target_net_hard_update    # hard update=True or soft update=False
        self.update_counter = 0   # target-network을 업데이트하기 위한 counter
        self.update_target_network()   

    def _loss_info(self):   
        """CSV 파일의 헤더 정보 반환"""
        return ['loss']

    def sample_action(self, state):  # action 샘플링
        if random.random() < self.eps:   # exploration
            return random.randint(0, self.action_dim - 1)   #0, action_dim-1까지의 랜덤한 정수 반환(cartpole은 0 또는 1)
        else:   # exploitation
            with torch.no_grad():   # 액션 샘플링 할때는 gradient 계산 안함
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device) # state를 tensor로 변환 -> (1, state_dim)
                q_values = self.q_net(state)    # state를 q_net에 넣어 각 actio(왼, 오)에 대한 q_values 계산
                return q_values.argmax().item() # q_values 중 가장 큰 값(greedy action)의 index 반환

    def decay_epsilon(self):   
        self.eps = max(self.eps_min, self.eps * self.eps_decay)  #1, 0.995, 0.990, 0.985 ... 0.01

    def update_target_network(self, tau = 0.005):   # default: soft update(tau=0.005)
        if self.target_net_hard_update: # True
            for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
                target_param.data.copy_(param.data)
        else:   # False(soft_update)
            for target_param, param in zip(self.q_target.parameters(), self.q_net.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def update(self, batch_size):
        self.q_net.train() #nn.Module의 train 함수 호출
        if len(self.replay_buffer) < batch_size:
            return [0.0]

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size) #replay buffer에서 batch_size만큼 샘플링
        states = torch.FloatTensor(states).to(self.device)  #(batch_size, state_dim)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)    #(batch_size, 1, action_dim) 
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)   #(batch_size, 1, 1)
        next_states = torch.FloatTensor(next_states).to(self.device)    #(batch_size, state_dim)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)   #(batch_size, 1, 1)

        q_values = self.q_net(states).gather(1, actions)    #DQN output: Q( ,왼), Q( ,오)

        with torch.no_grad():
            next_q_values = self.q_target(next_states).max(1, keepdim=True)[0]  #DQN output: max Q(s',a')
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)    #target Q-value: r + gamma * max Q(s',a')

        loss = nn.functional.mse_loss(q_values, target_q_values)    #MSE loss 계산

        self.optimizer.zero_grad()      #gradient 초기화
        loss.backward()     #backpropagation
        self.optimizer.step()  #parameter 업데이트
        self.decay_epsilon()   #epsilon decay
        self.update_counter += 1    #update counter 증가
        if self.target_net_hard_update:
            if self.update_counter % self.update_frequency == 0:    #main_net 100번 업데이트 될 때마다 target_net 업데이트
                self.update_target_network()
        else:
            self.update_target_network()

        return [loss.item()]    #loss 반환

    def save_model(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load_model(self, path):
        self.q_net.load_state_dict(torch.load(path))


def train_dqn():
    env = gym.make("CartPole-v1")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    hard_update_agent = DQN(state_dim, action_dim, target_net_hard_update = True)  # DQN 에이전트 생성
    soft_update_agent = DQN(state_dim, action_dim, target_net_hard_update = False)  # DQN 에이전트 생성

    episodes = 200 # 학습할 에피소드 수
    batch_size = 64
    hard_update_avg_reward = [] #score 저장 
    soft_update_avg_reward = [] 

    for agent, reward_list in zip([hard_update_agent, soft_update_agent], [hard_update_avg_reward, soft_update_avg_reward]):
        avg_reward=0
        for episode in range(episodes):
        
            state = env.reset()[0]
            total_reward = 0
            done = False

            while not done: # 한 에피소드 내에서 동작
                action = agent.sample_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                agent.replay_buffer.push(state, action, reward, next_state, done)
                agent.update(batch_size)

                state = next_state
                total_reward += reward  # total reward = timestep

            avg_reward=0.9*avg_reward+0.1*total_reward if avg_reward !=0 else total_reward  # q(s,a)는 평균 reward = 0.9*avg_reward+0.1*total_reward
            reward_list.append(avg_reward)
            print(f"Episode {episode+1}, timestep:{total_reward:.0f}, avg_reward: {avg_reward:.3f}, Epsilon: {agent.eps:.3f}")
            
    env.close()

    plt.plot(hard_update_avg_reward, label="Hard Update")
    plt.plot(soft_update_avg_reward, label="Soft Update")
    plt.xlabel("Episode")
    plt.ylabel("avg reward")
    plt.title("Hard Update vs Soft Update")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_dqn()
