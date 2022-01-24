from torch.nn import functional as nnf
from typing import Tuple, Optional
import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super().__init__()

        layers = []
        for i in range(len(sizes) - 1):
            layers.append(
                nn.Linear(sizes[i], sizes[i + 1], bias=bias)
            )

            if i < (len(sizes) - 2):
                layers.append(act())
        
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)