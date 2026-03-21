from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from crucible.models.base import TiedEmbeddingLM
from crucible.models.registry import register_model
from crucible.models.components.attention import Block


class LoopedTransformerLM(TiedEmbeddingLM):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
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
    ):
        super().__init__(vocab_size, model_dim, tie_embeddings, tied_embed_init_std, logit_softcap, embed_bottleneck_dim, spectral_embed_init)
        self.logical_steps = logical_steps
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, attention_variant, residual_variant)
            for _ in range(unique_blocks)
        ])
        self.step_scales = nn.Parameter(torch.ones(logical_steps, model_dim, dtype=torch.float32))

    def hidden(self, input_ids: Tensor, lora=None) -> Tensor:
        x = self.embed_tokens(input_ids)
        x0 = x
        for step in range(self.logical_steps):
            x_next = self.blocks[step % len(self.blocks)](x, x0)
            x = x + self.step_scales[step].to(dtype=x.dtype)[None, None, :] * (x_next - x)
        return x


def _build_looped(args: Any) -> LoopedTransformerLM:
    common = dict(
        vocab_size=args.vocab_size,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        embed_bottleneck_dim=args.embed_bottleneck_dim,
        spectral_embed_init=getattr(args, 'spectral_embed_init', False),
    )
    logical_steps = args.recurrence_steps if args.recurrence_steps > 0 else args.num_layers
    return LoopedTransformerLM(
        **common,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        logical_steps=logical_steps,
        unique_blocks=max(1, min(args.share_blocks, logical_steps)),
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        attention_variant=args.attention_variant,
        residual_variant=args.residual_variant,
    )


register_model("looped", _build_looped)
