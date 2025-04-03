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

