
from collections import deque 
import numpy as np 
import random 
import torch 



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
    

class PrioritizedReplayBuffer: 
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_annealing_steps=10000): 
        self.capacity = capacity 
        self.buffer = deque(maxlen=capacity) 
        self.priorities = deque(maxlen=capacity) 
        self.alpha = alpha 
        self.beta = beta 
        self.beta_annealing_steps = beta_annealing_steps 
        self.step = 0 

    def add(self, state, action, reward, next_state, done): 
        max_priority = max(self.priorities, default=1.0)
        self.buffer.append((state, action, reward, next_state, done)) 
        self.priorities.append(max_priority) 
    
    def sample(self, batch_size):
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha 
        probs /= probs.sum() 
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # Importance sampling weights 
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta) 
        weights /= weights.max() 

        # Anneal beta 
        self.step += 1 
        self.beta = min(1.0, self.beta + (1.0 - self.beta) * self.step / self.beta_annealing_steps)

        # Sample batch 
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch) 
        return (np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones), indices, weights)
    
    def update_priorities(self, indices, td_errors): 
        for idx, td_error in zip(indices, td_errors): 
            self.priorities[idx] = (abs(td_error) + 1e-6) ** self.alpha 
    
    def __len__(self): 
        return len(self.buffer) 
    