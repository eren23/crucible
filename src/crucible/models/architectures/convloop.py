from __future__ import annotations

import torch
from torch import Tensor, nn

from crucible.core.types import ArgsNamespace
from crucible.models.base import TiedEmbeddingLM
from crucible.models.registry import register_model
from crucible.models.components.attention import Block
from crucible.models.components.conv import FeatureConvBottleneck
from crucible.models.components.linear import CastedLinear


class ConvLoopedTransformerLM(TiedEmbeddingLM):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        core_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        logical_steps: int,
        unique_blocks: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        attention_variant: str = "standard",
        residual_variant: str = "standard",
        embed_bottleneck_dim: int = 0,
        spectral_embed_init: bool = False,
        activation: str = "relu_sq",
    ):
        super().__init__(vocab_size, embed_dim, tie_embeddings, tied_embed_init_std, logit_softcap, embed_bottleneck_dim, spectral_embed_init)
        self.compress = FeatureConvBottleneck(embed_dim, core_dim)
        self.expand = CastedLinear(core_dim, embed_dim, bias=False)
        self.blocks = nn.ModuleList([
            Block(core_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, attention_variant, residual_variant, activation=activation)
            for _ in range(unique_blocks)
        ])
        self.logical_steps = logical_steps
        self.step_scales = nn.Parameter(torch.ones(logical_steps, core_dim, dtype=torch.float32))

    def hidden(self, input_ids: Tensor, lora=None) -> Tensor:
        x_wide = self.embed_tokens(input_ids)
        x = self.compress(x_wide)
        x0 = x
        for step in range(self.logical_steps):
            x_next = self.blocks[step % len(self.blocks)](x, x0)
            x = x + self.step_scales[step].to(dtype=x.dtype)[None, None, :] * (x_next - x)
        return self.expand(x) + x_wide


def _build_convloop(args: ArgsNamespace) -> ConvLoopedTransformerLM:
    common = dict(
        vocab_size=args.vocab_size,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        embed_bottleneck_dim=args.embed_bottleneck_dim,
        spectral_embed_init=getattr(args, 'spectral_embed_init', False),
    )
    core_dim = min(args.state_dim, args.model_dim)
    if core_dim % max(args.num_heads, 1) != 0:
        raise ValueError(f"STATE_DIM={core_dim} must be divisible by NUM_HEADS={args.num_heads} for convloop")
    logical_steps = args.recurrence_steps if args.recurrence_steps > 0 else args.num_layers
    return ConvLoopedTransformerLM(
        **common,
        embed_dim=args.model_dim,
        core_dim=core_dim,
        num_heads=args.num_heads,
        num_kv_heads=min(args.num_kv_heads, args.num_heads),
        mlp_mult=args.mlp_mult,
        logical_steps=logical_steps,
        unique_blocks=max(1, min(args.share_blocks, logical_steps)),
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        attention_variant=args.attention_variant,
        residual_variant=args.residual_variant,
        activation=getattr(args, 'activation', 'relu_sq'),
    )


register_model("convloop", _build_convloop)
