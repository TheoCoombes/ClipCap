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


class MLPTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.0):
        super().__init__()

        if out_d is None:
            out_d = in_dim

        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x