from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int = -1, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32)) if dim > 0 else None

    def forward(self, x: Tensor) -> Tensor:
        shape = (x.size(-1),)
        out = F.rms_norm(x.float(), shape, eps=self.eps).to(dtype=x.dtype)
        return out if self.weight is None else out * self.weight.to(dtype=x.dtype)
