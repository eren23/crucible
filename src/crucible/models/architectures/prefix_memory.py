from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from crucible.models.base import TiedEmbeddingLM
from crucible.models.registry import register_model
from crucible.models.components.memory import PrefixMemoryBlock


class PrefixMemoryLM(TiedEmbeddingLM):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        state_dim: int,
        num_layers: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        residual_variant: str = "standard",
        embed_bottleneck_dim: int = 0,
        spectral_embed_init: bool = False,
    ):
        super().__init__(vocab_size, model_dim, tie_embeddings, tied_embed_init_std, logit_softcap, embed_bottleneck_dim, spectral_embed_init)
        self.blocks = nn.ModuleList([
            PrefixMemoryBlock(model_dim, state_dim, mlp_mult, residual_variant)
            for _ in range(num_layers)
        ])
        self.step_scales = nn.Parameter(torch.ones(num_layers, model_dim, dtype=torch.float32))

    def hidden(self, input_ids: Tensor, lora=None) -> Tensor:
        x = self.embed_tokens(input_ids)
        x0 = x
        for step, block in enumerate(self.blocks):
            x_next = block(x, x0)
            x = x + self.step_scales[step].to(dtype=x.dtype)[None, None, :] * (x_next - x)
        return x


def _build_prefix_memory(args: Any) -> PrefixMemoryLM:
    common = dict(
        vocab_size=args.vocab_size,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        embed_bottleneck_dim=args.embed_bottleneck_dim,
        spectral_embed_init=getattr(args, 'spectral_embed_init', False),
    )
    return PrefixMemoryLM(
        **common,
        model_dim=args.model_dim,
        state_dim=min(args.state_dim, args.model_dim),
        num_layers=args.num_layers,
        mlp_mult=args.mlp_mult,
        residual_variant=args.residual_variant,
    )


# Register under both names to match the original build_model() behavior.
register_model("prefix_memory", _build_prefix_memory)
register_model("memory", _build_prefix_memory)
