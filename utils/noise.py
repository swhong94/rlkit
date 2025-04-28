# utils/noise.py
import numpy as np


class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2): 
        self.mu = mu * np.ones(action_dim) 
        self.theta = theta 
        self.sigma = sigma 
        self.reset() 

    def reset(self): 
        self.state = self.mu 

    def sample(self): 
        x = self.state 
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x)) 
        self.state = x + dx 
        return self.state 
    

class EpsilonGreedy: 
    def __init__(self, 
                 epsilon_start=1.0,
                 epsilon_end=0.01, 
                 epsilon_decay=0.995): 
        self.epsilon = epsilon_start 
        self.epsilon_end = epsilon_end 
        self.epsilon_decay = epsilon_decay 

    def sample(self, ): 
        return np.random.rand() < self.epsilon 
    
    def decay(self, ): 
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)





# class OUNoise:
#     """
#     Ornstein-Uhlenbeck noise process for exploration in continuous action spaces.
    
#     Args:
#         size (int): Dimension of the action space.
#         mu (float): Mean of the noise.
#         theta (float): Rate of mean reversion.
#         sigma (float): Volatility of the noise.
#     """
#     def __init__(self, size, mu=0.0, theta=0.15, sigma=0.1):
#         self.size = size
#         self.mu = mu * np.ones(size)
#         self.theta = theta
#         self.sigma = sigma
#         self.reset()

#     def reset(self):
#         """
#         Reset the noise state to the mean.
#         """
# #         self.state = np.copy(self.mu)

#     def sample(self):
#         """
#         Generate a noise sample.
#         """
#         x = self.state
#         dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
#         self.state = x + dx
#         return self.state

