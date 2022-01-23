import torch.nn as nn
import torch

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.0):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5

        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        if y is None:
            y = x
        
        b, n, c = x.shape
        _, m, d = y.shape

        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)

        keys_values = self.to_keys_values(y).reshape(
            b, m, 2, self.num_heads, c // self.num_heads
        )

        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]

        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        
        attention = attention.softmax(dim=2)
    
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)

        return out, attention