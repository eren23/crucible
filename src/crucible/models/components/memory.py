from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from crucible.models.components.linear import CastedLinear
from crucible.models.components.norm import RMSNorm
from crucible.models.components.mlp import MLP


class CausalPrefixMemory(nn.Module):
    def __init__(self, dim: int, state_dim: int):
        super().__init__()
        self.update_proj = CastedLinear(dim, state_dim, bias=False)
        self.gate_proj = CastedLinear(dim, state_dim, bias=False)
        self.out_proj = CastedLinear(state_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        memory = torch.cumsum(torch.sigmoid(self.gate_proj(x)) * self.update_proj(x), dim=1)
        denom = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=torch.float32)[None, :, None]
        return self.out_proj(F.rms_norm((memory / denom).to(dtype=x.dtype), (memory.size(-1),)))


class PrefixMemoryBlock(nn.Module):
    def __init__(self, dim: int, state_dim: int, mlp_mult: int, residual_variant: str = "standard", activation: str = "relu_sq"):
        super().__init__()
        if residual_variant not in {"standard", "gated"}:
            raise ValueError(f"Unsupported RESIDUAL_VARIANT={residual_variant!r}")
        self.memory_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.memory = CausalPrefixMemory(dim, state_dim)
        self.mlp = MLP(dim, mlp_mult, activation=activation)
        self.memory_mix = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.residual_variant = residual_variant
        if residual_variant == "gated":
            self.delta_gate = nn.Parameter(torch.full((dim,), 2.0, dtype=torch.float32))

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x_in = x
        x = x + self.memory_mix.to(dtype=x.dtype)[None, None, :] * self.memory(self.memory_norm(x))
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        if self.residual_variant == "gated":
            x = x_in + torch.sigmoid(self.delta_gate).to(dtype=x.dtype)[None, None, :] * (x - x_in)
        return x
