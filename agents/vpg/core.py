import torch 
from torch import nn 


class PolicyNet(nn.Module): 
    def __init__(self, state_dim, hidden_dims, action_dim, activation="ReLU"): 
        super(PolicyNet, self).__init__() 
        layers = [] 
        prev_dim = state_dim 
        for h_dim in hidden_dims:  
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(getattr(nn, activation)()) 
            prev_dim = h_dim 
        layers.append(nn.Linear(prev_dim, action_dim)) 
        layers.append(nn.Softmax(dim=-1))     # Probability of each action
        self.fc_net = nn.Sequential(*layers)  # Policy Network 

    def forward(self, state): 
        return self.fc_net(state)
    

class ValueNet(nn.Module): 
    def __init__(self, state_dim, hidden_dims, activation="ReLU"): 
        super(ValueNet, self).__init__() 
        layers = [] 
        prev_dim = state_dim 
        for h_dim in hidden_dims: 
            layers.append(nn.Linear(prev_dim, h_dim)) 
            layers.append(getattr(nn, activation)()) 
            prev_dim = h_dim 
        layers.append(nn.Linear(prev_dim, 1)) 
        self.fc_net = nn.Sequential(*layers) 

    def forward(self, state): 
        return self.fc_net(state)