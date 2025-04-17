import torch 
import torch.nn as nn 



class ValueNetwork(nn.Module): 
    def __init__(self, input_dim, hidden_dims, output_dim, activation="ReLU"):
        super(ValueNetwork, self).__init__() 
        layers = [] 
        prev_dim = input_dim
        for h_dim in hidden_dims: 
            layers.append(nn.Linear(prev_dim, h_dim)) 
            layers.append(getattr(nn, activation)())
            prev_dim = h_dim 
        layers.append(nn.Linear(prev_dim, output_dim))

        self.fc_net = nn.Sequential(*layers)
    
    def forward(self, x): 
        return self.fc_net(x) 

