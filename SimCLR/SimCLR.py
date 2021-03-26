from typing import Tuple

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

    def forward(self, input: Tuple[torch.tensor, torch.tensor]):
        """ forward pass on a positive example pair """

        bs = input[0].shape[0]

        input = torch.cat(input, dim=0)

        embeds = self.projection(self.encoder(input))

        z_i, z_j = torch.split(embeds, bs)

        return z_i, z_j