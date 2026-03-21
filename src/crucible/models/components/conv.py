from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DepthwiseConv1D(nn.Module):
    """Causal depthwise 1D conv for local n-gram patterns before attention."""
    def __init__(self, dim: int, kernel: int = 3):
        super().__init__()
        self.pad = kernel - 1
        self.conv = nn.Conv1d(dim, dim, kernel, groups=dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x_t = F.pad(x.transpose(1, 2), (self.pad, 0))
        return self.conv(x_t).transpose(1, 2)


class FeatureConvBottleneck(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        if out_dim <= 0 or out_dim > in_dim:
            raise ValueError(f"expected 0 < out_dim <= in_dim, got in_dim={in_dim}, out_dim={out_dim}")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.group = math.ceil(in_dim / out_dim)
        self.padded_dim = self.group * out_dim
        self.left = nn.Parameter(torch.full((out_dim,), 0.125, dtype=torch.float32))
        self.center = nn.Parameter(torch.ones(out_dim, dtype=torch.float32))
        self.right = nn.Parameter(torch.full((out_dim,), 0.125, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        if self.padded_dim != self.in_dim:
            x = torch.cat((x, x.new_zeros(*x.shape[:-1], self.padded_dim - self.in_dim)), dim=-1)
        pooled = x.reshape(*x.shape[:-1], self.out_dim, self.group).mean(dim=-1)
        left = torch.cat((pooled[..., :1], pooled[..., :-1]), dim=-1)
        right = torch.cat((pooled[..., 1:], pooled[..., -1:]), dim=-1)
        return self.left.to(dtype=x.dtype)[None, None, :] * left + self.center.to(dtype=x.dtype)[None, None, :] * pooled + self.right.to(dtype=x.dtype)[None, None, :] * right
