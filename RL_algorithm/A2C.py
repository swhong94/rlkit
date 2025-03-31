import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

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
    pi = self.actor(state)    #state -> actor -> action_probs
    value = self.critic(state)
    return pi, value
    

class A2C:
  #critic, actor parameter initialization
  def __init__(self, state_dim, action_dim, hidden_dim = 64, lr = 1e-3, gamma = 0.99, device = None):
    super(A2C, self).__init__()
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.hidden_dim = hidden_dim
    self.lr = lr
    self.gamma = gamma
    self.device = device if device else 'cpu'
    self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
    self.optimizer = optim.Adam(self.policy.parameters(), lr = lr)
    self.buffer = []

  #action selection
  def select_action(self, state):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device) #(1, state_dim)
    pi, value = self.policy(state)  
    dist = Categorical(pi)  # 이산 action space인 경우 Categorical 분포 사용
    action = dist.sample()  
    log_prob = dist.log_prob(action)
    return action.item(), log_prob # at, log pi(at|st)

  # N timestep samples -> store transition 
  def store_transition(self, state, action, log_prob, reward, next_state, done):
    self.buffer.append((state, action, log_prob, reward, next_state, done))
  
  # update critic and actor(각각 따로 해도 되고 같이 해도 되는 듯?? 차이가 있는지 한번 확인해보면 좋을 것 같기도하고..)
  # buffer에 저장된 transition을 이용해 agent 업데이트
  def update(self):
  
    if len(self.buffer) == 0:
        return
    
    states, actions, log_probs, rewards, next_states, dones = zip(*self.buffer)
    states = torch.tensor(states, dtype=torch.float32).to(self.device)
    actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
    dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
    log_probs = torch.stack(log_probs).to(self.device)

    #Compute value estimates
    values = self.policy.critic(states).squeeze()  # (N, 1) -> (N,)
    next_values = self.policy.critic(next_states).squeeze()  # (N, 1) -> (N,)

    #Compute TD target and advantage
    with torch.no_grad():
      td_targets = rewards + self.gamma * next_values * (1 - dones)
      advantage = td_targets - values

    #Compute critic loss
    actor_loss = -(log_probs * advantage).mean()
    critic_loss = nn.MSELoss()(td_targets,values)
    loss= actor_loss + critic_loss
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    self.buffer = []  # 한번 업데이트 끝나면 버퍼 초기화 
    return loss.item()
  

