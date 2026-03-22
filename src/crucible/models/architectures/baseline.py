from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from crucible.models.base import TiedEmbeddingLM
from crucible.models.registry import register_model, register_schema
from crucible.models.components.attention import Block, _parse_block_pattern
from crucible.models.components.gate import SmearGate
from crucible.models.components.hash_embed import BigramHash, TrigramHash
from crucible.models.components.merge import TokenMerger


class BaselineGPT(TiedEmbeddingLM):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        attention_variant: str = "standard",
        residual_variant: str = "standard",
        embed_bottleneck_dim: int = 0,
        use_smear_gate: bool = False,
        use_bigram_hash: bool = False,
        bigram_hash_buckets: int = 2048,
        bigram_hash_embed_dim: int = 128,
        ortho_init: bool = False,
        spectral_embed_init: bool = False,
        use_conv_block: bool = False,
        conv_kernel: int = 3,
        multiscale_window: int = 0,
        token_merge_layer: int = 0,
        token_merge_threshold: float = 0.9,
        block_pattern: str = "",
        use_trigram_hash: bool = False,
        trigram_hash_buckets: int = 4096,
        activation: str = "relu_sq",
        use_moe: bool = False,
        moe_num_experts: int = 4,
        moe_top_k: int = 2,
    ):
        super().__init__(vocab_size, model_dim, tie_embeddings, tied_embed_init_std, logit_softcap, embed_bottleneck_dim, spectral_embed_init)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(min(self.num_encoder_layers, self.num_decoder_layers), model_dim, dtype=torch.float32))
        _lw = _parse_block_pattern(block_pattern, num_layers) if block_pattern else [0] * num_layers
        self.blocks = nn.ModuleList([
            Block(
                model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                attention_variant, residual_variant,
                use_conv=use_conv_block, conv_kernel=conv_kernel,
                multiscale_window=multiscale_window if _lw[i] == 0 else 0,
                attention_window=_lw[i],
                activation=activation,
                use_moe=use_moe,
                moe_num_experts=moe_num_experts,
                moe_top_k=moe_top_k,
            )
            for i in range(num_layers)
        ])
        self.token_merger = TokenMerger(token_merge_threshold) if token_merge_layer > 0 else None
        self.token_merge_layer = token_merge_layer
        self.smear_gate_mod = SmearGate(model_dim) if use_smear_gate else None
        self.bigram_hash_mod = BigramHash(vocab_size, num_buckets=bigram_hash_buckets, embed_dim=bigram_hash_embed_dim, model_dim=model_dim) if use_bigram_hash else None
        self.trigram_hash_mod = TrigramHash(vocab_size, num_buckets=trigram_hash_buckets, embed_dim=bigram_hash_embed_dim, model_dim=model_dim) if use_trigram_hash else None
        if ortho_init:
            self._apply_ortho_init(num_layers)

    def _apply_ortho_init(self, num_layers: int) -> None:
        skip_patterns = ('tok_emb', 'embed_low', 'embed_proj', 'lm_head', 'bigram_hash', 'trigram_hash', 'smear_gate')
        for name, p in self.named_parameters():
            if p.ndim == 2 and p.numel() > 256 and not any(pat in name for pat in skip_patterns):
                nn.init.orthogonal_(p)
                if 'proj' in name:
                    p.data *= 1.0 / math.sqrt(2 * num_layers)

    def hidden(self, input_ids: Tensor, lora=None) -> Tensor:
        x = self.embed_tokens(input_ids)
        if self.smear_gate_mod is not None:
            x = self.smear_gate_mod(x)
        if self.bigram_hash_mod is not None:
            prev_ids = torch.cat([input_ids[:, :1], input_ids[:, :-1]], dim=1)
            x = x + self.bigram_hash_mod(prev_ids, input_ids)
        if self.trigram_hash_mod is not None:
            x = x + self.trigram_hash_mod(input_ids)
        x0 = x
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            qd = lora.q_loras[i] if lora else None
            vd = lora.v_loras[i] if lora else None
            x = self.blocks[i](x, x0, qd, vd)
            if self.token_merger is not None and i + 1 == self.token_merge_layer:
                x = self.token_merger(x)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            bi = self.num_encoder_layers + i
            qd = lora.q_loras[bi] if lora else None
            vd = lora.v_loras[bi] if lora else None
            x = self.blocks[bi](x, x0, qd, vd)
            if self.token_merger is not None and bi + 1 == self.token_merge_layer:
                x = self.token_merger(x)
        return x


def _build_baseline(args: Any) -> BaselineGPT:
    common = dict(
        vocab_size=args.vocab_size,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        embed_bottleneck_dim=args.embed_bottleneck_dim,
        spectral_embed_init=getattr(args, 'spectral_embed_init', False),
    )
    return BaselineGPT(
        **common,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        attention_variant=args.attention_variant,
        residual_variant=args.residual_variant,
        use_smear_gate=getattr(args, 'smear_gate', False),
        use_bigram_hash=getattr(args, 'bigram_hash', False),
        bigram_hash_buckets=getattr(args, 'bigram_hash_buckets', 2048),
        bigram_hash_embed_dim=getattr(args, 'bigram_hash_embed_dim', 128),
        ortho_init=getattr(args, 'ortho_init', False),
        use_conv_block=getattr(args, 'conv_block', False),
        conv_kernel=getattr(args, 'conv_kernel', 3),
        multiscale_window=getattr(args, 'multiscale_window', 0),
        token_merge_layer=getattr(args, 'token_merge_layer', 0),
        token_merge_threshold=getattr(args, 'token_merge_threshold', 0.9),
        block_pattern=getattr(args, 'block_pattern', ''),
        use_trigram_hash=getattr(args, 'trigram_hash', False),
        trigram_hash_buckets=getattr(args, 'trigram_hash_buckets', 4096),
        activation=getattr(args, 'activation', 'relu_sq'),
        use_moe=getattr(args, 'use_moe', False),
        moe_num_experts=getattr(args, 'moe_num_experts', 4),
        moe_top_k=getattr(args, 'moe_top_k', 2),
    )


register_model("baseline", _build_baseline)
register_schema("baseline", {
    "MODEL_DIM": {"type": "int", "default": 512, "description": "Model/embedding dimension"},
    "NUM_LAYERS": {"type": "int", "default": 9, "description": "Number of transformer layers"},
    "NUM_HEADS": {"type": "int", "default": 8, "description": "Number of attention heads"},
    "NUM_KV_HEADS": {"type": "int", "default": 4, "description": "Number of key-value heads (GQA)"},
    "MLP_MULT": {"type": "int", "default": 2, "description": "MLP expansion multiplier"},
    "ACTIVATION": {"type": "str", "default": "relu_sq", "description": "Activation function name"},
    "XSA_LAYERS": {"type": "str", "default": "", "description": "Comma-separated layer indices for XSA"},
    "SMEAR_GATE": {"type": "bool", "default": False, "description": "Enable SmearGate"},
    "BIGRAM_HASH": {"type": "bool", "default": False, "description": "Enable BigramHash embeddings"},
    "USE_MOE": {"type": "bool", "default": False, "description": "Enable Mixture of Experts"},
    "MOE_NUM_EXPERTS": {"type": "int", "default": 4, "description": "Number of MOE experts"},
    "MOE_TOP_K": {"type": "int", "default": 2, "description": "Top-k expert routing"},
})
