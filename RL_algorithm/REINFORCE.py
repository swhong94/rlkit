import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class PolicyNetwork(nn.Module): #continuous action space라서 정규분포를 따르는 정책을 사용
    def __init__(self, state_dim, action_dim, hidden_dim = 128):
        super(PolicyNetwork, self).__init__()
        self.fc_mean = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.fc_std = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softplus()
        )

    def forward(self, x):
        mean = self.fc_mean(x)
        std = self.fc_std(x)
        return mean, std   
    

class REINFORCE:
    def __init__(self, state_dim, action_dim, hidden_dim = 128, lr = 1e-3, gamma = 0.99, device = None):
        super(REINFORCE, self).__init__()
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr = lr)
        self.gamma = gamma    

        if device is None:
            self.device = 'cpu'
        else:
            self.device = device
    
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device) 
        mean,std = self.policy(state)   # pi(at|st): action_probs
        dist = Normal(mean, std)     
        action = dist.sample()  # action_probs로부터 action을 샘플링
        log_prob = dist.log_prob(action).sum(dim=-1) # log(pi(at|st))    
        return action.clamp(-1.0,1.0).item(), log_prob #at, log pi(at|st)
    
    def update(self, rewards, log_probs):
        returns = []
        policy_loss = [] 
        G = 0   
        # loss function J(theta) = E[log(pi(at|st)) * Gt]
        
        for r in rewards[::-1]: #[start:stop:step] -> 역순으로 rewards를 순회
            G = r + self.gamma*G
            returns.insert(0, G) # 0번째 인덱스에 G 삽입

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)   
        # normalize: reinforce 알고리즘은 eposode 끝날때까지 기다렸다가 한번에 업데이트해주므로 normalize해줌
        
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G) 
        
        policy_loss = torch.stack(policy_loss).sum()
        # policy_loss = Sigma(t=1 to T) -log(pi(at|st)) * Gt
        
        self.optimizer.zero_grad() # 기울기 초기화
        policy_loss.backward() # policy_loss를 이용해 역전파
        self.optimizer.step()

        return policy_loss.item()

