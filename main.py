import gymnasium as gym 
import numpy as np

import torch 
import torch.nn as nn 

from agents.actor_critic import ActorCritic
from agents.REINFORCE import REINFORCE 

def main(): 
    env = gym.make("Pendulum-v1") 
    # agent = ActorCritic(env, 
    #                     hidden_dims=[128, 128], 
    #                     activation="ReLU", 
    #                     policy_lr=5e-6, 
    #                     value_lr=5e-5, 
    #                     gamma=0.9, 
    #                     max_steps=200, 
    #                     log_dir="logs/actor_critic_logs", 
    #                     device="cpu", 
    #                     plot_window=10) 
    # 
    # agent.train(max_episodes=100, max_timesteps=200, log_interval=10) 
    
    agent = REINFORCE(env=env, 
                      hidden_dims=[128, 128], 
                      policy_lr=0.0005, 
                      value_lr=0.001, 
                      gamma=0.9, 
                      lambda_gae=0.95, 
                      gae=True, 
                      max_steps=200, 
                      log_dir="logs/reinforce_logs", 
                      plot_window=100)

    agent.train(max_episodes=2000, max_steps=2_000_000, log_interval=10) 

    env.close() 

if __name__ == "__main__": 

    main() 
    