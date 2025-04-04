import torch 
import torch.nn as nn 



class PolicyNet(nn.Module): 
    def __init__(self, input_dim, hidden_dims, output_dim, activation="ReLU"): 
        super(PolicyNet, self).__init__() 
        layers = [] 
        prev_dim = input_dim 
        for h_dim in hidden_dims: 
            layers.append(nn.Linear(prev_dim, h_dim)) 
            layers.append(getattr(nn, activation)()) 
            prev_dim = h_dim 
        layers.append(nn.Linear(prev_dim, output_dim))
        self.fc_net = nn.Sequential(*layers) 

    def forward(self, state): 
        logits = self.fc_net(state) 
        return logits 

class ValueNet(nn.Module): 
    def __init__(self, input_dim, hidden_dims, activation="ReLU"): 
        super(ValueNet, self).__init__() 
        layers = [] 
        prev_dim = input_dim 
        for h_dim in hidden_dims: 
            layers.append(nn.Linear(prev_dim, h_dim)) 
            layers.append(getattr(nn, activation)()) 
            prev_dim = h_dim 
        layers.append(nn.Linear(prev_dim, 1)) 
        self.fc_net = nn.Sequential(*layers) 

    def forward(self, state): 
        value = self.fc_net(state) 
        return value 