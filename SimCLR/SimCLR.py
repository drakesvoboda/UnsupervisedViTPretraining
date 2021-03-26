import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair 

import torchvision
import torchvision.transforms as transforms

class SimCLR(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size: int, representation_dim: int = 128, non_linearity=nn.ReLU):
        super().__init__()

        self.encoder = encoder

        # We use ... and a 2-layer MLP projection head to project the representation to a ... latent space.  - Quote from paper
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            non_linearity(),
            nn.Linear(hidden_size, representation_dim)
        )

    def forward(self, x_i, x_j):
        """ forward pass on a positive example pair """

        z_i = self.projection(self.encoder(x_i))
        z_j = self.projection(self.encoder(x_j))

        return z_i, z_j