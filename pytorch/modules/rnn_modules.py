"""
PyTorch implementation of
Recurrent neural network classes for encoding and decoding
@author: Sean A. Cantrell
"""
from tqdm import tqdm
import numpy as np

import torch
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class GeoGRU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GeoGRU, self).__init__()
        # Specify network dimensions
        self.in_dim = in_dim
        self.out_dim = out_dim
        # Create network functions w/ model params
        self.lin_up_gate = nn.Linear(in_dim + out_dim, out_dim)
        self.lin_forget_gate = nn.Linear(in_dim + out_dim, out_dim)
        self.lin_update = nn.Linear(in_dim + out_dim, out_dim)

        # Declare properties that class functions will assign values to
        self.h = None
        self.h_set = None

    def update_state(self, x, h):
        # Update gate
        u_input = torch.cat([x, h], dim=1)
        u = nn.Sigmoid()(self.lin_up_gate(u_input))

        # Forget gate
        r_input = torch.cat([x, h], dim=1)
        r = F.tanh(self.lin_forget_gate(r_input))

        # State update
        h_input = torch.cat([x, r * h], dim=1)
        h_update = F.tanh(self.lin_update(h_input))

        # New state
        h_new = (1 - u) * h + u * h_update
        return h_new

    def forward(self, x):
        shape = x.size()
        state = Variable(torch.zeros([shape[0], self.out_dim]),
                         requires_grad=True)
        for i in range(shape[1]):
            state = self.update_state(x[:,i], state)
        return state

    def __call__(self, x):
        return self.forward(x)
