
from collections import deque 
import numpy as np 
import random 
import torch 


# class ReplayBuffer: 
#     """ 
#     A simple replay buffer for storing and sampling for experience replay 
    
#     Args: 
#         capacity (int): maximum size of the buffer 
#         device (str): device to store (request) the buffer ('cpu' or 'cuda')
#     """
#     def __init__(self, 
#                  capacity, 
#                  device='cpu'): 
#         self.buffer = deque(maxlen=capacity) 
#         self.device = device 

#     def push(self, state, action, reward, next_state, done): 
#         """
#         Add a transition (experience) to the buffer 
#         """
#         # Store as a tuple
#         transition = (state, action, reward, next_state, done) 
#         self.buffer.append(transition) 

#     def sample(self, batch_size): 
#         """Sample a batch from the buffer: 
        
#         Returns: Tuple of tensors (states, actions, rewards, next_states, dones)"""

#         # Randomly sample a batch of transitions 
#         transitions = random.sample(self.buffer, batch_size) 

#         # Unzip and convert to tensors 
#         states, actions, rewards, next_states, dones = zip(*transitions)            
#         states = torch.FloatTensor(np.array(states)).to(self.device)                    # tensor shape: (batch_size, state_dim)
#         actions = torch.LongTensor(np.array(actions)).to(self.device)      # tensor shape: (batch_size, 1)
#         rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)     # tensor shape: (batch_size, 1)
#         next_states = torch.FloatTensor(np.array(next_states)).to(self.device)          # tensor shape: (batch_size, state_dim)
#         dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)         # tensor shape: (batch_size, 1)

#         return states, actions, rewards, next_states, dones 

#     def __len__(self):
#         """Return the current size of the buffer"""
#         return len(self.buffer) 
    




class ReplayBuffer: 
    def __init__(self, capacity): 
        self.buffer = deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)) 
    
    def __len__(self):
        return len(self.buffer) 
    


