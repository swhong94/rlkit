import torch
import torch.nn as nn 

def append_activation(layers, activation): 
    if activation.lower() == 'relu': 
        layers.append(nn.ReLU()) 
    elif activation.lower() == 'tanh': 
        layers.append(nn.Tanh()) 
    elif activation.lower() == 'softplus': 
        layers.append(nn.Softplus()) 
    elif activation.lower() == 'sigmoid': 
        layers.append(nn.Sigmoid()) 
    elif activation.lower() == 'leakyrelu': 
        layers.append(nn.LeakyReLU()) 
    elif activation.lower() == 'elu': 
        layers.append(nn.ELU()) 
    elif activation.lower() == 'selu': 
        layers.append(nn.SELU())     
    elif activation.lower() == 'gelu': 
        layers.append(nn.GELU()) 
    else: 
        raise ValueError(f"Activation function {activation} not supported") 
        