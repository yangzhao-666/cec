import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class ValueNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear2(x)
        return x

class 
