from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int = -1, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32)) if dim > 0 else None

    def forward(self, x: Tensor) -> Tensor:
        shape = (x.size(-1),)
        out = F.rms_norm(x.float(), shape, eps=self.eps).to(dtype=x.dtype)
        return out if self.weight is None else out * self.weight.to(dtype=x.dtype)


class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        self.weight.data = self.weight.data.float()
        if self.bias is not None:
            self.bias.data = self.bias.data.float()

    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


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


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class SmearGate(nn.Module):
    """Per-dimension sigmoid gate blending current token with previous token."""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate).to(dtype=x.dtype)[None, None, :]
        x_prev = torch.cat([x[:, :1], x[:, :-1]], dim=1)
        return g * x + (1 - g) * x_prev


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


class DepthwiseConv1D(nn.Module):
    """Causal depthwise 1D conv for local n-gram patterns before attention."""
    def __init__(self, dim: int, kernel: int = 3):
        super().__init__()
        self.pad = kernel - 1
        self.conv = nn.Conv1d(dim, dim, kernel, groups=dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x_t = F.pad(x.transpose(1, 2), (self.pad, 0))
        return self.conv(x_t).transpose(1, 2)


class TokenMerger(nn.Module):
    """Blend adjacent tokens with high cosine similarity (soft merging without length change)."""
    def __init__(self, threshold: float = 0.9):
        super().__init__()
        self.threshold = threshold

    def forward(self, x: Tensor) -> Tensor:
        x_norm = F.normalize(x.float(), dim=-1)
        sim = (x_norm[:, :-1] * x_norm[:, 1:]).sum(dim=-1, keepdim=True)
        alpha = torch.sigmoid((sim - self.threshold) * 10.0).to(dtype=x.dtype)
        alpha = F.pad(alpha, (0, 0, 0, 1))  # (B, T, 1), last token alpha=0
        x_shifted = torch.cat([x[:, 1:], x[:, -1:]], dim=1)
        return x * (1 - alpha * 0.5) + x_shifted * (alpha * 0.5)


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
    ):
        super().__init__()
        if residual_variant not in {"standard", "gated"}:
            raise ValueError(f"Unsupported RESIDUAL_VARIANT={residual_variant!r}")
        self.conv = DepthwiseConv1D(dim, conv_kernel) if use_conv else None
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, attention_variant, multiscale_window, attention_window)
        self.mlp = MLP(dim, mlp_mult)
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


class TiedEmbeddingLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        embed_bottleneck_dim: int = 0,
        spectral_embed_init: bool = False,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        if embed_bottleneck_dim < 0:
            raise ValueError(f"EMBED_BOTTLENECK_DIM must be non-negative, got {embed_bottleneck_dim}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.embed_bottleneck_dim = embed_bottleneck_dim
        self.spectral_embed_init = spectral_embed_init
        if embed_bottleneck_dim > 0:
            if embed_bottleneck_dim >= model_dim:
                raise ValueError(f"EMBED_BOTTLENECK_DIM={embed_bottleneck_dim} must be smaller than MODEL_DIM={model_dim}")
            self.embed_low = nn.Embedding(vocab_size, embed_bottleneck_dim)
            self.embed_proj = CastedLinear(embed_bottleneck_dim, model_dim, bias=False)
        else:
            self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            emb = self.embed_low if self.embed_bottleneck_dim > 0 else self.tok_emb
            if self.spectral_embed_init:
                # SVD-based spectral init: orthogonal embedding vectors for better
                # gradient flow through tied input/output projection.
                W = torch.randn_like(emb.weight)
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                emb.weight.data = U * (self.tied_embed_init_std * math.sqrt(emb.weight.size(0)))
            else:
                nn.init.normal_(emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def token_parameter_names(self) -> set[str]:
        return {"embed_low.weight", "embed_proj.weight"} if self.embed_bottleneck_dim > 0 else {"tok_emb.weight"}

    def embed_tokens(self, input_ids: Tensor) -> Tensor:
        x = self.embed_proj(self.embed_low(input_ids)) if self.embed_bottleneck_dim > 0 else self.tok_emb(input_ids)
        return F.rms_norm(x, (x.size(-1),))

    def tied_logits(self, x: Tensor) -> Tensor:
        if self.embed_bottleneck_dim > 0:
            return F.linear(F.linear(x, self.embed_proj.weight.T.to(x.dtype)), self.embed_low.weight.to(x.dtype))
        return F.linear(x, self.tok_emb.weight)

    def compute_loss(self, hidden: Tensor, target_ids: Tensor, lora=None) -> Tensor:
        x = self.final_norm(hidden)
        flat_x = x.reshape(-1, hidden.size(-1))
        logits_proj = self.tied_logits(flat_x) if self.tie_embeddings else self.lm_head(flat_x)
        if lora is not None:
            logits_proj = logits_proj + lora.lm_head_lora(x).reshape(-1, logits_proj.size(-1))
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        if lora is not None:
            bsz, sl = hidden.shape[:2]
            return F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction="none").reshape(bsz, sl)
        return F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction="mean", ignore_index=-100)

    def per_token_loss(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        """Per-token cross-entropy (no reduction), shape [batch*seq]."""
        hidden = self.hidden(input_ids)
        x = self.final_norm(hidden).reshape(-1, hidden.size(-1))
        logits_proj = self.tied_logits(x) if self.tie_embeddings else self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction="none")

    def hidden(self, input_ids: Tensor, lora=None) -> Tensor:
        raise NotImplementedError

    def forward(self, input_ids: Tensor, target_ids: Tensor, lora=None) -> Tensor:
        return self.compute_loss(self.hidden(input_ids, lora=lora), target_ids, lora=lora)


class BaselineGPT(TiedEmbeddingLM):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int, tie_embeddings: bool, tied_embed_init_std: float, logit_softcap: float, rope_base: float, qk_gain_init: float, attention_variant: str = "standard", residual_variant: str = "standard", embed_bottleneck_dim: int = 0, use_smear_gate: bool = False, use_bigram_hash: bool = False, bigram_hash_buckets: int = 2048, bigram_hash_embed_dim: int = 128, ortho_init: bool = False, spectral_embed_init: bool = False, use_conv_block: bool = False, conv_kernel: int = 3, multiscale_window: int = 0, token_merge_layer: int = 0, token_merge_threshold: float = 0.9, block_pattern: str = "", use_trigram_hash: bool = False, trigram_hash_buckets: int = 4096):
        super().__init__(vocab_size, model_dim, tie_embeddings, tied_embed_init_std, logit_softcap, embed_bottleneck_dim, spectral_embed_init)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(min(self.num_encoder_layers, self.num_decoder_layers), model_dim, dtype=torch.float32))
        _lw = _parse_block_pattern(block_pattern, num_layers) if block_pattern else [0] * num_layers
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, attention_variant, residual_variant, use_conv=use_conv_block, conv_kernel=conv_kernel, multiscale_window=multiscale_window if _lw[i] == 0 else 0, attention_window=_lw[i]) for i in range(num_layers)])
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


class LoopedTransformerLM(TiedEmbeddingLM):
    def __init__(self, vocab_size: int, model_dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int, logical_steps: int, unique_blocks: int, tie_embeddings: bool, tied_embed_init_std: float, logit_softcap: float, rope_base: float, qk_gain_init: float, attention_variant: str = "standard", residual_variant: str = "standard", embed_bottleneck_dim: int = 0, spectral_embed_init: bool = False):
        super().__init__(vocab_size, model_dim, tie_embeddings, tied_embed_init_std, logit_softcap, embed_bottleneck_dim, spectral_embed_init)
        self.logical_steps = logical_steps
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, attention_variant, residual_variant) for _ in range(unique_blocks)])
        self.step_scales = nn.Parameter(torch.ones(logical_steps, model_dim, dtype=torch.float32))

    def hidden(self, input_ids: Tensor, lora=None) -> Tensor:
        x = self.embed_tokens(input_ids)
        x0 = x
        for step in range(self.logical_steps):
            x_next = self.blocks[step % len(self.blocks)](x, x0)
            x = x + self.step_scales[step].to(dtype=x.dtype)[None, None, :] * (x_next - x)
        return x


class FeatureConvBottleneck(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        if out_dim <= 0 or out_dim > in_dim:
            raise ValueError(f"expected 0 < out_dim <= in_dim, got in_dim={in_dim}, out_dim={out_dim}")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.group = math.ceil(in_dim / out_dim)
        self.padded_dim = self.group * out_dim
        self.left = nn.Parameter(torch.full((out_dim,), 0.125, dtype=torch.float32))
        self.center = nn.Parameter(torch.ones(out_dim, dtype=torch.float32))
        self.right = nn.Parameter(torch.full((out_dim,), 0.125, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        if self.padded_dim != self.in_dim:
            x = torch.cat((x, x.new_zeros(*x.shape[:-1], self.padded_dim - self.in_dim)), dim=-1)
        pooled = x.reshape(*x.shape[:-1], self.out_dim, self.group).mean(dim=-1)
        left = torch.cat((pooled[..., :1], pooled[..., :-1]), dim=-1)
        right = torch.cat((pooled[..., 1:], pooled[..., -1:]), dim=-1)
        return self.left.to(dtype=x.dtype)[None, None, :] * left + self.center.to(dtype=x.dtype)[None, None, :] * pooled + self.right.to(dtype=x.dtype)[None, None, :] * right


class ConvLoopedTransformerLM(TiedEmbeddingLM):
    def __init__(self, vocab_size: int, embed_dim: int, core_dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int, logical_steps: int, unique_blocks: int, tie_embeddings: bool, tied_embed_init_std: float, logit_softcap: float, rope_base: float, qk_gain_init: float, attention_variant: str = "standard", residual_variant: str = "standard", embed_bottleneck_dim: int = 0, spectral_embed_init: bool = False):
        super().__init__(vocab_size, embed_dim, tie_embeddings, tied_embed_init_std, logit_softcap, embed_bottleneck_dim, spectral_embed_init)
        self.compress = FeatureConvBottleneck(embed_dim, core_dim)
        self.expand = CastedLinear(core_dim, embed_dim, bias=False)
        self.blocks = nn.ModuleList([Block(core_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, attention_variant, residual_variant) for _ in range(unique_blocks)])
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
    def __init__(self, dim: int, state_dim: int, mlp_mult: int, residual_variant: str = "standard"):
        super().__init__()
        if residual_variant not in {"standard", "gated"}:
            raise ValueError(f"Unsupported RESIDUAL_VARIANT={residual_variant!r}")
        self.memory_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.memory = CausalPrefixMemory(dim, state_dim)
        self.mlp = MLP(dim, mlp_mult)
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


class PrefixMemoryLM(TiedEmbeddingLM):
    def __init__(self, vocab_size: int, model_dim: int, state_dim: int, num_layers: int, mlp_mult: int, tie_embeddings: bool, tied_embed_init_std: float, logit_softcap: float, residual_variant: str = "standard", embed_bottleneck_dim: int = 0, spectral_embed_init: bool = False):
        super().__init__(vocab_size, model_dim, tie_embeddings, tied_embed_init_std, logit_softcap, embed_bottleneck_dim, spectral_embed_init)
        self.blocks = nn.ModuleList([PrefixMemoryBlock(model_dim, state_dim, mlp_mult, residual_variant) for _ in range(num_layers)])
        self.step_scales = nn.Parameter(torch.ones(num_layers, model_dim, dtype=torch.float32))

    def hidden(self, input_ids: Tensor, lora=None) -> Tensor:
        x = self.embed_tokens(input_ids)
        x0 = x
        for step, block in enumerate(self.blocks):
            x_next = block(x, x0)
            x = x + self.step_scales[step].to(dtype=x.dtype)[None, None, :] * (x_next - x)
        return x


def build_model(args: Any) -> TiedEmbeddingLM:
    common = dict(
        vocab_size=args.vocab_size,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        embed_bottleneck_dim=args.embed_bottleneck_dim,
        spectral_embed_init=getattr(args, 'spectral_embed_init', False),
    )
    attention_kwargs = dict(attention_variant=args.attention_variant, residual_variant=args.residual_variant)
    if args.model_family == "baseline":
        return BaselineGPT(**common, num_layers=args.num_layers, model_dim=args.model_dim, num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init, use_smear_gate=getattr(args, 'smear_gate', False), use_bigram_hash=getattr(args, 'bigram_hash', False), bigram_hash_buckets=getattr(args, 'bigram_hash_buckets', 2048), bigram_hash_embed_dim=getattr(args, 'bigram_hash_embed_dim', 128), ortho_init=getattr(args, 'ortho_init', False), use_conv_block=getattr(args, 'conv_block', False), conv_kernel=getattr(args, 'conv_kernel', 3), multiscale_window=getattr(args, 'multiscale_window', 0), token_merge_layer=getattr(args, 'token_merge_layer', 0), token_merge_threshold=getattr(args, 'token_merge_threshold', 0.9), block_pattern=getattr(args, 'block_pattern', ''), use_trigram_hash=getattr(args, 'trigram_hash', False), trigram_hash_buckets=getattr(args, 'trigram_hash_buckets', 4096), **attention_kwargs)
    if args.model_family == "looped":
        logical_steps = args.recurrence_steps if args.recurrence_steps > 0 else args.num_layers
        return LoopedTransformerLM(**common, model_dim=args.model_dim, num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult, logical_steps=logical_steps, unique_blocks=max(1, min(args.share_blocks, logical_steps)), rope_base=args.rope_base, qk_gain_init=args.qk_gain_init, **attention_kwargs)
    if args.model_family == "convloop":
        core_dim = min(args.state_dim, args.model_dim)
        if core_dim % max(args.num_heads, 1) != 0:
            raise ValueError(f"STATE_DIM={core_dim} must be divisible by NUM_HEADS={args.num_heads} for convloop")
        logical_steps = args.recurrence_steps if args.recurrence_steps > 0 else args.num_layers
        return ConvLoopedTransformerLM(**common, embed_dim=args.model_dim, core_dim=core_dim, num_heads=args.num_heads, num_kv_heads=min(args.num_kv_heads, args.num_heads), mlp_mult=args.mlp_mult, logical_steps=logical_steps, unique_blocks=max(1, min(args.share_blocks, logical_steps)), rope_base=args.rope_base, qk_gain_init=args.qk_gain_init, **attention_kwargs)
    if args.model_family in {"memory", "prefix_memory"}:
        return PrefixMemoryLM(**common, model_dim=args.model_dim, state_dim=min(args.state_dim, args.model_dim), num_layers=args.num_layers, mlp_mult=args.mlp_mult, residual_variant=args.residual_variant)
    # Fallback to crucible model registry for user-defined architectures
    try:
        from crucible.models.registry import build_model as crucible_build
    except ImportError:
        raise ValueError(f"unsupported MODEL_FAMILY={args.model_family!r}; expected baseline, looped, convloop, prefix_memory, or a crucible-registered family")
    return crucible_build(args)


# -----------------------------
# TTT LoRA MODULES
# -----------------------------

class BatchedLinearLoRA(nn.Module):
    """LoRA for a linear layer, with independent weights per batch element."""
    def __init__(self, bsz: int, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.in_features = in_features
        self.A = nn.Parameter(torch.empty(bsz, rank, in_features))
        self.B = nn.Parameter(torch.zeros(bsz, out_features, rank))
        self.reset()

    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.A.transpose(1, 2)) @ self.B.transpose(1, 2)

    def reset(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        with torch.no_grad():
            self.A.uniform_(-bound, bound)
            self.B.zero_()


class BatchedTTTLoRA(nn.Module):
    """All LoRA adapters for one batch: LM head and Q/V per block."""
    def __init__(self, bsz: int, model: TiedEmbeddingLM, rank: int):
        super().__init__()
        block0 = model.blocks[0]
        dim = block0.attn.c_q.in_features
        vocab = model.embed_low.num_embeddings if hasattr(model, 'embed_low') else model.tok_emb.num_embeddings
        self.lm_head_lora = BatchedLinearLoRA(bsz, dim, vocab, rank)
        self.q_loras = nn.ModuleList()
        self.v_loras = nn.ModuleList()
        for block in model.blocks:
            self.q_loras.append(BatchedLinearLoRA(bsz, dim, block.attn.c_q.weight.shape[0], rank))
            self.v_loras.append(BatchedLinearLoRA(bsz, dim, block.attn.c_v.weight.shape[0], rank))

    def reset(self) -> None:
        for m in self.modules():
            if isinstance(m, BatchedLinearLoRA):
                m.reset()
