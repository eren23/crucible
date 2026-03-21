from __future__ import annotations

import torch
from torch import Tensor, nn


class SmearGate(nn.Module):
    """Per-dimension sigmoid gate blending current token with previous token."""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate).to(dtype=x.dtype)[None, None, :]
        x_prev = torch.cat([x[:, :1], x[:, :-1]], dim=1)
        return g * x + (1 - g) * x_prev
