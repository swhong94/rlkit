import torch 
import torch.nn as nn 

from torch.distributions import Normal  
from models.utils import append_activation 

class MLP_Policy(nn.Module): 
    """ 
    MLP for policy network
    """
    def __init__(self, 
                 input_dim, 
                 hidden_dims, 
                 output_dim, 
                 activation="ReLU"):
        super(MLP_Policy, self).__init__() 
    
        # Create list of layers
        layers = []
        prev_dim = input_dim
        
        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(getattr(nn, activation)())
            prev_dim = hidden_dim
        
        # Mean and std heads for the policy
        self.shared_net = nn.Sequential(*layers)
        self.mu_head = nn.Linear(prev_dim, output_dim)
        self.log_std = nn.Parameter(torch.zeros(1, output_dim))
    
    def forward(self, x): 
        x = self.shared_net(x)
        mu = torch.tanh(self.mu_head(x))
        std = torch.exp(self.log_std)
        return mu, std 



class MLP_Value(nn.Module): 
    """
    MLP for value network 
    """
    def __init__(self, 
                 input_dim, 
                 hidden_dims, 
                 activation="ReLU"): 
        super(MLP_Value, self).__init__() 
        
        # Create list of layers
        layers = []
        prev_dim = input_dim
        
        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(getattr(nn, activation)())
            prev_dim = hidden_dim
            
        # Add output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x): 
        return self.net(x) 
    


class MLP(nn.Module): 
    """ 
    A flexible MLP model that can be used for both policy and value networks. 
    
    Args: 
        input_dim (int): The dimension of input_data (=state_dim)
        hidden_dims (list): The list of hidden_dimensions 
        output_dim (int): The dimension of output (number of actions or 1 for value)
        action_type (str): The type of action space (discrete or continuous or value) 
        activation (str): The activation function to use 
    """
    def __init__(self, input_dim, hidden_dims, output_dim, output_type="discrete", activation="ReLU"): 
        super(MLP, self).__init__()  

        self.output_type = output_type.lower() 
        assert self.output_type in ["discrete", "continuous", "value"], \
            f'output type "{self.output_type}" is not a valid type, must be one of: ["discrete", "continuous", "value"]' 


        # Shared MLP Backbone
        layers = [] 
        prev_dim = input_dim 
        for h_dim in hidden_dims: 
            layers.append(nn.Linear(prev_dim, h_dim)) 
            append_activation(layers, activation) 
            prev_dim = h_dim 
        self.backbone = nn.Sequential(*layers) 
    
        # Action head  
        if self.output_type == "discrete":  
            self.output_head = nn.Linear(prev_dim, output_dim) 
        elif self.output_type == "continuous": 
            self.mu_head = nn.Linear(prev_dim, output_dim) 
            self.log_std = nn.Parameter(torch.zeros(output_dim))
        elif self.output_type == "value": 
            self.output_head = nn.Linear(prev_dim, 1) 
        

    def forward(self, x): 
        x = self.backbone(x) 

        if self.output_type == "discrete": 
            logits = self.output_head(x) 
            return logits 
        elif self.output_type == "continuous": 
            mu = self.mu_head(x) 
            std = torch.exp(torch.clamp(self.log_std, -20, 2)) 
            return mu, std 
        elif self.output_type == "value": 
            value = self.output_head(x)  
            return value 

    

class PolicyNet(MLP): 
    """
    A policy network that outputs a distribution over actions. 
    """
    def __init__(self, input_dim, hidden_dims, output_dim, output_type='discrete', activation="ReLU"): 
        super(PolicyNet, self).__init__(input_dim, hidden_dims, output_dim, output_type=output_type, activation=activation) 


class ValueNet(MLP): 
    """
    A value network that outputs a single value. 
    """
    def __init__(self, input_dim, hidden_dims, activation="ReLU"): 
        super(ValueNet, self).__init__(input_dim, hidden_dims, 1, output_type="value", activation=activation) 
        
        


if __name__ == "__main__": 
    import torch 
    import torch.nn as nn 
    from torch.distributions import Normal, Categorical 

    # Test MLP 

    state = torch.randn(1, 10) 
     


    # 1. Test PolicyNet (discrete)
    discrete_actor = PolicyNet(input_dim=10, hidden_dims=[128, 128], output_dim=6) 
    logits = discrete_actor(state)  
    dist = Categorical(logits=logits) 
    action = dist.sample() 
    print("Probabilities: ", dist.probs.detach(), "Action: ", action) 

    # 2. Test PolicyNet (continuous) 
    continuous_actor = PolicyNet(input_dim=10, hidden_dims=[128, 128], output_dim=1, output_type='continuous')  
    mu, std = continuous_actor(state) 
    dist = Normal(mu, std) 
    action = dist.sample() 
    print(f"Continuous Action: Mu: {mu}, Std: {std}, Action: {action}")

    
    # 3. Test ValueNet 
    value_net = ValueNet(input_dim=10, hidden_dims=[32, 32, 32], activation="ReLU") 
    value = value_net(state) 
    print(f"Value: {value.detach()}") 
        
        
        
