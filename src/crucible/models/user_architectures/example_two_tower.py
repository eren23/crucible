"""Example plugin: Two-Tower architecture.

Demonstrates how to build a user architecture plugin for Crucible.
This model processes tokens through two parallel transformer stacks
(towers) and fuses their representations before the LM head.

Usage in an experiment design:
    MODEL_FAMILY=two_tower
    NUM_LAYERS=6        # layers per tower (total = 2x)
    MODEL_DIM=512
    NUM_HEADS=4
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from crucible.models.base import TiedEmbeddingLM
from crucible.models.registry import register_model
from crucible.models.components.attention import Block


class TwoTowerLM(TiedEmbeddingLM):
    """Two parallel transformer towers with learned fusion."""

    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        embed_bottleneck_dim: int = 0,
        spectral_embed_init: bool = False,
        attention_variant: str = "standard",
        residual_variant: str = "standard",
        activation: str = "relu_sq",
    ):
        super().__init__(
            vocab_size, model_dim, tie_embeddings,
            tied_embed_init_std, logit_softcap,
            embed_bottleneck_dim, spectral_embed_init,
        )
        # Tower A: processes tokens directly
        self.tower_a = nn.ModuleList([
            Block(
                model_dim, num_heads, num_kv_heads, mlp_mult,
                rope_base, qk_gain_init, attention_variant, residual_variant,
                activation=activation,
            )
            for _ in range(num_layers)
        ])
        # Tower B: processes tokens with a learned shift
        self.tower_b = nn.ModuleList([
            Block(
                model_dim, num_heads, num_kv_heads, mlp_mult,
                rope_base, qk_gain_init, attention_variant, residual_variant,
                activation=activation,
            )
            for _ in range(num_layers)
        ])
        # Learned gate to fuse the two towers
        self.fusion_gate = nn.Parameter(torch.zeros(model_dim))

    def hidden(self, input_ids: Tensor, lora=None) -> Tensor:
        x = self.embed_tokens(input_ids)

        # Tower A
        a = x
        for block in self.tower_a:
            a = block(a, x)

        # Tower B
        b = x
        for block in self.tower_b:
            b = block(b, x)

        # Gated fusion: sigmoid gate blends the two towers
        gate = torch.sigmoid(self.fusion_gate).to(dtype=x.dtype)
        return gate * a + (1 - gate) * b


def _build_two_tower(args: Any) -> TwoTowerLM:
    return TwoTowerLM(
        vocab_size=args.vocab_size,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        embed_bottleneck_dim=args.embed_bottleneck_dim,
        spectral_embed_init=getattr(args, "spectral_embed_init", False),
        attention_variant=args.attention_variant,
        residual_variant=args.residual_variant,
        activation=getattr(args, "activation", "relu_sq"),
    )


register_model("two_tower", _build_two_tower)
