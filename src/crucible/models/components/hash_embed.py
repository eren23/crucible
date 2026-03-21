from __future__ import annotations

import torch
from torch import Tensor, nn

from crucible.models.components.linear import CastedLinear


class BigramHash(nn.Module):
    """Hash-based bigram embedding: maps (prev_token, cur_token) to model_dim."""
    def __init__(self, vocab_size: int, num_buckets: int = 2048, embed_dim: int = 128, model_dim: int = 512):
        super().__init__()
        self.num_buckets = num_buckets
        self.embed = nn.Embedding(num_buckets, embed_dim)
        self.proj = CastedLinear(embed_dim, model_dim, bias=False)
        self.proj._zero_init = True

    def forward(self, prev_ids: Tensor, cur_ids: Tensor) -> Tensor:
        h = (prev_ids.long() * 1000003 + cur_ids.long()) % self.num_buckets
        return self.proj(self.embed(h))


class TrigramHash(nn.Module):
    """Hash-based trigram embedding: maps (tok[i-2], tok[i-1], tok[i]) to model_dim."""
    def __init__(self, vocab_size: int, num_buckets: int = 4096, embed_dim: int = 128, model_dim: int = 512):
        super().__init__()
        self.num_buckets = num_buckets
        self.embed = nn.Embedding(num_buckets, embed_dim)
        self.proj = CastedLinear(embed_dim, model_dim, bias=False)
        self.proj._zero_init = True

    def forward(self, ids: Tensor) -> Tensor:
        prev2 = torch.cat([ids[:, :2], ids[:, :-2]], dim=1)
        prev1 = torch.cat([ids[:, :1], ids[:, :-1]], dim=1)
        h = (prev2.long() * 1000003 * 1000003 + prev1.long() * 1000003 + ids.long()) % self.num_buckets
        return self.proj(self.embed(h))
