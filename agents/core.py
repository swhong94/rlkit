# agents/core.py 

# Backbone of all agents 

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym


class Agent(nn.Module): 
    """ 
    Base class for all agents 
    """
    def __init__(self,): 
        super(Agent, self).__init__() 


    def act(self, state): 
        """Select an action given the current state"""
        pass 

    def save(self, path): 
        """Save the model to a file"""
        pass 

    def load(self, path): 
        pass 

    def evaluate(self, env, num_episodes): 
        """Evaluate the agent"""
        pass 



def soft_update(target, 
                source,
                tau: float=0.005):
    """Soft update the target network"""

    for target_param, source_param in zip(target.parameters(), source.parameters()): 
        target_param.data.copy_(tau * source_param.data + (1-tau) * target_param.data) 
    