import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

class Nutrition_MLP(nn.Module):
    def __init__(self, hidden_layers):
        super(Nutrition_MLP, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(100, hidden_layers[0])] + [nn.Linear(hidden_layers[i], hidden_layers[i+1]) for i in range(len(hidden_layers) - 1)] + [nn.Linear(hidden_layers[-1], 4)]) #output 4 dims
    
    def forward(self,x):
        for i in range(len(self.linears)):
            x = f.relu(self.linears[i](x))
        return x