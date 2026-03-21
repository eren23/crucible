from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        self.weight.data = self.weight.data.float()
        if self.bias is not None:
            self.bias.data = self.bias.data.float()

    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)
