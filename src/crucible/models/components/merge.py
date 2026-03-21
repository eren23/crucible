from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TokenMerger(nn.Module):
    """Blend adjacent tokens with high cosine similarity (soft merging without length change)."""
    def __init__(self, threshold: float = 0.9):
        super().__init__()
        self.threshold = threshold

    def forward(self, x: Tensor) -> Tensor:
        x_norm = F.normalize(x.float(), dim=-1)
        sim = (x_norm[:, :-1] * x_norm[:, 1:]).sum(dim=-1, keepdim=True)
        alpha = torch.sigmoid((sim - self.threshold) * 10.0).to(dtype=x.dtype)
        alpha = F.pad(alpha, (0, 0, 0, 1))  # (B, T, 1), last token alpha=0
        x_shifted = torch.cat([x[:, 1:], x[:, -1:]], dim=1)
        return x * (1 - alpha * 0.5) + x_shifted * (alpha * 0.5)
