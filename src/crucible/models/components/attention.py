from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from crucible.models.components.linear import CastedLinear
from crucible.models.components.norm import RMSNorm
from crucible.models.components.rotary import Rotary, apply_rotary_emb
from crucible.models.components.mlp import MLP
from crucible.models.components.conv import DepthwiseConv1D


def _windowed_causal_mask(seqlen: int, window: int, device: torch.device) -> Tensor:
    """Boolean causal mask with sliding window: True = attend, False = block."""
    pos = torch.arange(seqlen, device=device)
    return (pos[:, None] >= pos[None, :]) & ((pos[:, None] - pos[None, :]) < window)


def _parse_block_pattern(pattern: str, num_layers: int, default_window: int = 64) -> list[int]:
    """Parse BLOCK_PATTERN into per-layer attention window sizes (0 = global)."""
    parts = [p.strip() for p in pattern.split(",")]
    if len(parts) != num_layers:
        raise ValueError(f"BLOCK_PATTERN has {len(parts)} entries but model has {num_layers} layers")
    result: list[int] = []
    for p in parts:
        if p in ("global", "g", "0"):
            result.append(0)
        elif p in ("local", "l"):
            result.append(default_window)
        else:
            result.append(int(p))
    return result


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        attention_variant: str = "standard",
        multiscale_window: int = 0,
        attention_window: int = 0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if attention_variant not in {"standard", "paired"}:
            raise ValueError(f"Unsupported ATTENTION_VARIANT={attention_variant!r}")
        if attention_variant == "paired" and num_heads % 2 != 0:
            raise ValueError("paired attention requires an even number of heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.attention_variant = attention_variant
        self.multiscale_window = multiscale_window
        self.attention_window = attention_window
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)
        if attention_variant == "paired":
            self.head_pair_mix = nn.Parameter(torch.eye(2, dtype=torch.float32).repeat(num_heads // 2, 1, 1))

    def forward(self, x: Tensor, q_delta=None, v_delta=None) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x) + (q_delta if q_delta is not None else 0)
        k = self.c_k(x)
        v = self.c_v(x) + (v_delta if v_delta is not None else 0)
        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        use_gqa = self.num_kv_heads != self.num_heads
        if self.attention_window > 0 and seqlen > self.attention_window:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=_windowed_causal_mask(seqlen, self.attention_window, x.device), enable_gqa=use_gqa)
        elif self.multiscale_window > 0 and seqlen > self.multiscale_window:
            n_local = self.num_heads // 2
            if use_gqa:
                rep = self.num_heads // self.num_kv_heads
                k_full, v_full = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
            else:
                k_full, v_full = k, v
            win = _windowed_causal_mask(seqlen, self.multiscale_window, x.device)
            y_local = F.scaled_dot_product_attention(q[:, :n_local], k_full[:, :n_local], v_full[:, :n_local], attn_mask=win)
            y_global = F.scaled_dot_product_attention(q[:, n_local:], k_full[:, n_local:], v_full[:, n_local:], is_causal=True)
            y = torch.cat([y_local, y_global], dim=1)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True, enable_gqa=use_gqa)
        if self.attention_variant == "paired":
            y = y.reshape(bsz, self.num_heads // 2, 2, seqlen, self.head_dim)
            y = torch.einsum("bpitd,poi->bpotd", y, self.head_pair_mix.to(dtype=y.dtype)).reshape(
                bsz, self.num_heads, seqlen, self.head_dim
            )
        return self.proj(y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        attention_variant: str = "standard",
        residual_variant: str = "standard",
        use_conv: bool = False,
        conv_kernel: int = 3,
        multiscale_window: int = 0,
        attention_window: int = 0,
        activation: str = "relu_sq",
    ):
        super().__init__()
        if residual_variant not in {"standard", "gated"}:
            raise ValueError(f"Unsupported RESIDUAL_VARIANT={residual_variant!r}")
        self.conv = DepthwiseConv1D(dim, conv_kernel) if use_conv else None
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, attention_variant, multiscale_window, attention_window)
        self.mlp = MLP(dim, mlp_mult, activation=activation)
        self.residual_variant = residual_variant
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        if residual_variant == "gated":
            self.delta_gate = nn.Parameter(torch.full((dim,), 2.0, dtype=torch.float32))

    def forward(self, x: Tensor, x0: Tensor, q_delta_fn=None, v_delta_fn=None) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x_in = x
        if self.conv is not None:
            x = x + self.conv(x)
        n = self.attn_norm(x)
        qd = q_delta_fn(n) if q_delta_fn is not None else None
        vd = v_delta_fn(n) if v_delta_fn is not None else None
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(n, qd, vd)
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        if self.residual_variant == "gated":
            x = x_in + torch.sigmoid(self.delta_gate).to(dtype=x.dtype)[None, None, :] * (x - x_in)
        return x
