import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import numpy as np
import random

class ActorCritic(nn.Module): 
    def __init__(self, state_dim, action_dim, hidden_dim = 64):
        super(ActorCritic, self).__init__()
    
        #actor(Policy)
        self.actor = nn.Sequential( 
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1) # Softmax for action probabilities(discrete action space)
        )

        # critic(Value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # value function은 scalar이므로 출력 차원 1
        )

    def forward(self, state):
        action_probs = self.actor(state)    #state -> actor -> action_probs
        state_value = self.critic(state)
        return action_probs, state_value
  
class Buffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, log_prob, reward, next_state, done):
        if len(self.buffer) == self.buffer.maxlen:
            self.buffer.popleft()
        self.buffer.append((state, action, log_prob, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, log_probs, rewards, next_states, dones = zip(*batch)
        return (np.stack(states),
                np.array(actions),
                np.stack([log_prob.detach().numpy() for log_prob in log_probs]),  # detach() 추가
                np.array(rewards),
                np.stack(next_states),
                np.array(dones))

    # def reset(self):
    #     self.buffer.clear()  # 버퍼 비우기

    def __len__(self):
        return len(self.buffer)

class A2C:
    # critic, actor parameter initialization
    def __init__(self, state_dim, action_dim, hidden_dim = 64, lr = 1e-3, gamma = 0.99, capacity =10000, device = None):
        super(A2C, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.device = device if device else 'cpu'
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device) # policy = {actor, critic}
        self.optimizer = optim.Adam(self.policy.parameters(), lr = lr)
        self.buffer = Buffer(capacity) #Online A2C일 경우에는 매 step마다 업데이트, 그냥 A2C는 N개의 배치마다 update

    # action selection
    # tensor = multi dimensional matrix
    # unsqueeze(0): 0번째 인덱스에 1차원 추가 하는 역할
    def select_action(self, state):
      state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device) #(1, state_dim)
      action_probs, state_value = self.policy(state)  
      dist = Categorical(action_probs)  #actor만 필요
      action = dist.sample()
      log_prob = dist.log_prob(action)
      return action.item(), log_prob, state_value

    # update critic and actor(각각 따로 해도 되고 같이 해도 되는 듯?? 차이가 있는지 한번 확인해보면 좋을 것 같기도하고..)
    # buffer에 저장된 transition을 이용해 agent 업데이트
    def update(self, batch_size=100):
      self.policy.train()

      if len(self.buffer)< batch_size:
          return 0
      
      loss = []
      
      # loss function j(theta) = E[log(pi(at|st))*At]    
      states, actions, log_probs, rewards, next_states, dones = self.buffer.sample(batch_size)
      states = torch.tensor(states, dtype=torch.float32).to(self.device)
      actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
      rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
      next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
      dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
      log_probs = torch.tensor(log_probs, dtype=torch.float32).to(self.device)

      values = self.policy.critic(states).squeeze()  # (N, 1) -> (N)
      next_values = self.policy.critic(next_states).squeeze()  # (N, 1) -> (N)
      td_targets = rewards + self.gamma * next_values * (1 - dones)
      advantage = td_targets - values
      advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

      #Compute critic loss
      actor_loss = -(log_probs * advantage.detach()).mean() # detach(): 그래디언트 계산 안해도 돼서 분리
      critic_loss = nn.functional.mse_loss(values, td_targets)
      loss.append(actor_loss + critic_loss)
      loss = torch.stack(loss).sum()
          
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      # self.buffer.reset()
      return loss.item()


