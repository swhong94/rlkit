import torch 
import torch.nn as nn 


class Actor(nn.Module): 
    """ 
    MLP for policy network
    """
    def __init__(self, 
                 input_dim, 
                 hidden_dims, 
                 output_dim, 
                 max_action,
                 activation="ReLU"):
        super(Actor, self).__init__() 
    
        # Create list of layers
        layers = []
        prev_dim = input_dim
        
        # Add hidden layers
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(getattr(nn, activation)())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())
        # Mean and std heads for the policy
        self.fc_net = nn.Sequential(*layers)
        self.max_action = max_action
    
    def forward(self, x): 
        return self.max_action * self.fc_net(x)


class Critic(nn.Module): 
    def __init__(self, input_dim, hidden_dims, output_dim, activation="ReLU"):
        super(Critic, self).__init__() 
        layers = [] 
        prev_dim = input_dim + output_dim
        for h_dim in hidden_dims: 
            layers.append(nn.Linear(prev_dim, h_dim)) 
            layers.append(getattr(nn, activation)())
            prev_dim = h_dim 
        layers.append(nn.Linear(prev_dim, 1))

        self.fc_net = nn.Sequential(*layers)
    
    def forward(self, state, action): 
        return self.fc_net(torch.cat([state, action], dim=1)) 