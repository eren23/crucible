from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from crucible.models.components.linear import CastedLinear


# ---------------------------------------------------------------------------
# Activation function registry
# ---------------------------------------------------------------------------

def _relu_sq(x: Tensor) -> Tensor:
    return torch.relu(x).square()

def _leaky01_sq(x: Tensor) -> Tensor:
    return F.leaky_relu(x, 0.1).square()

def _leaky02_sq(x: Tensor) -> Tensor:
    return F.leaky_relu(x, 0.2).square()

def _leaky08_sq(x: Tensor) -> Tensor:
    return F.leaky_relu(x, 0.8).square()

def _elu03_sq(x: Tensor) -> Tensor:
    return F.elu(x, 0.3).square()

def _mish_sq(x: Tensor) -> Tensor:
    return F.mish(x).square()

def _gelu_sq(x: Tensor) -> Tensor:
    return F.gelu(x).square()

def _x_absx(x: Tensor) -> Tensor:
    return x * x.abs()

def _log1p_relu_sq(x: Tensor) -> Tensor:
    return torch.log1p(torch.relu(x)).square()


ACTIVATIONS: dict[str, callable] = {
    "relu_sq": _relu_sq,
    "leaky01_sq": _leaky01_sq,
    "leaky02_sq": _leaky02_sq,
    "leaky08_sq": _leaky08_sq,
    "elu03_sq": _elu03_sq,
    "mish_sq": _mish_sq,
    "gelu_sq": _gelu_sq,
    "x_absx": _x_absx,
    "log1p_relu_sq": _log1p_relu_sq,
}


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int, activation: str = "relu_sq"):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

        if activation not in ACTIVATIONS:
            raise ValueError(
                f"Unknown activation {activation!r}. "
                f"Available: {', '.join(sorted(ACTIVATIONS))}"
            )
        self._activation = ACTIVATIONS[activation]

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self._activation(self.fc(x)))
