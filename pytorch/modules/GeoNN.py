"""
PyTorch implementation of the Geodesic Neural Network
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

class GRU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GRU, self).__init__()
        