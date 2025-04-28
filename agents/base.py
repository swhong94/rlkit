from abc import ABC, abstractmethod 
import torch 

class BaseAgent(ABC): 
    def __init__(self, env, device='cpu'): 
        self.env = env 
        self.device = device 

    @abstractmethod
    def act(self, observation, explore=True): 
        pass 

    @abstractmethod 
    def train_step(self, ): 
        pass 

    @abstractmethod 
    def train_episode(self, ): 
        pass 
    

