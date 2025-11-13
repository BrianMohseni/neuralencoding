from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

@dataclass
class EEGNetConfig:
    """
    this is a simple template for what your config file should look like.
    """
    input_size: int = 4
    chunk_size: int = 64

class EEGNet(nn.Module):
    def __init__(self, config):
        """
        this is a simple template for what your EEGNet should look like
        """
        super().__init__()

        self.linear = nn.Linear(4*config.chunk_size, 1)

    def forward(self, x):
        b, t, n = x.shape
        x = x.reshape(b, t*n)
        x = self.linear(x)

        return x
