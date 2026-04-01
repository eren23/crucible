"""Hybrid LE-WM: Latency-Optimal Hybrid World Model inspired by Nemotron-Flash.

Replaces the ViT encoder with a configurable mix of standard softmax attention
and O(N*d^2) linear attention blocks.  Linear attention uses an ELU+1 feature
map and is naturally bidirectional (no causal mask needed for the encoder).

Additional knobs:
    * Meta-tokens: K learnable prefix tokens prepended after CLS that
      participate in attention but are discarded at output.
    * WeightNormProjection: optional per-step hook that projects Q/K/V
      and MLP weight matrices onto the unit-norm sphere.

Env vars (all existing lewm vars PLUS):
    HYBRID_ENCODER_PATTERN:  Block type pattern (default: "ALALAL")
                             A = standard softmax attention, L = linear attention
    META_TOKENS:             Number of learnable prefix meta tokens (default: 4)
    USE_WEIGHT_NORM:         Enable weight norm projection hook (default: 0)
    LINEAR_ATTN_EPS:         Denominator epsilon for linear attention (default: 1e-6)

Existing lewm env vars:
    MODEL_DIM, PATCH_SIZE, IMAGE_SIZE, ENCODER_DEPTH, ENCODER_HEADS,
    PREDICTOR_DEPTH, PREDICTOR_HEADS, PREDICTOR_DIM_HEAD, PREDICTOR_MLP_DIM,
    ACTION_DIM, NUM_FRAMES, SIGREG_WEIGHT, SIGREG_PROJECTIONS, DROPOUT,
    PRETRAINED_WEIGHTS
"""
from __future__ import annotations

import math
import os
import sys
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from crucible.models.base import CrucibleModel
from crucible.models.registry import register_model


# ---------------------------------------------------------------------------
# Patch Embedding (shared with ViT encoder)
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Convert image to patch embeddings."""

    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        return self.proj(x).flatten(2).transpose(1, 2)


# ---------------------------------------------------------------------------
# Standard ViT Block (softmax attention)
# ---------------------------------------------------------------------------

class ViTBlock(nn.Module):
    """Standard ViT transformer block with pre-norm."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Bidirectional Linear Attention
# ---------------------------------------------------------------------------

class BidirectionalLinearAttention(nn.Module):
    """Linear attention using ELU+1 feature map.

    phi(x) = elu(x) + 1  (ensures non-negativity)

    Compute (right-to-left association for O(N*d^2)):
        numerator   = phi(Q) @ (phi(K)^T @ V)
        denominator = phi(Q) @ (phi(K)^T @ 1) + eps

    Naturally bidirectional -- no causal mask.

    Args:
        dim: Model/token dimension.
        num_heads: Number of attention heads.
        eps: Small constant for numerical stability in denominator.
    """

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

    @staticmethod
    def _phi(x: Tensor) -> Tensor:
        """ELU+1 feature map: ensures non-negative features."""
        return F.elu(x) + 1.0

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, N, D]
        Returns:
            [B, N, D]
        """
        B, N, D = x.shape
        H = self.num_heads
        d = self.head_dim

        q = self.q_proj(x).reshape(B, N, H, d).permute(0, 2, 1, 3)  # [B, H, N, d]
        k = self.k_proj(x).reshape(B, N, H, d).permute(0, 2, 1, 3)  # [B, H, N, d]
        v = self.v_proj(x).reshape(B, N, H, d).permute(0, 2, 1, 3)  # [B, H, N, d]

        # Apply feature map
        q = self._phi(q)  # [B, H, N, d]
        k = self._phi(k)  # [B, H, N, d]

        # O(N*d^2) linear attention via right-to-left association:
        # kv = K^T @ V : [B, H, d, d]
        kv = torch.einsum("bhnd,bhne->bhde", k, v)
        # numerator = Q @ kv : [B, H, N, d]
        numerator = torch.einsum("bhnd,bhde->bhne", q, kv)

        # denominator = Q @ (K^T @ 1) : [B, H, N]
        k_sum = k.sum(dim=2)  # [B, H, d]
        denominator = torch.einsum("bhnd,bhd->bhn", q, k_sum)  # [B, H, N]
        denominator = denominator.unsqueeze(-1) + self.eps  # [B, H, N, 1]

        out = numerator / denominator  # [B, H, N, d]
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)  # [B, N, D]
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Linear ViT Block
# ---------------------------------------------------------------------------

class LinearViTBlock(nn.Module):
    """Pre-norm + BidirectionalLinearAttention + residual + pre-norm + MLP + residual.

    Same structure as ViTBlock but uses linear attention instead of softmax.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = BidirectionalLinearAttention(dim, num_heads, eps=eps)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        h = self.attn(h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Hybrid ViT Encoder
# ---------------------------------------------------------------------------

class HybridViTEncoder(nn.Module):
    """Vision Transformer encoder with configurable mix of softmax and linear attention.

    Architecture:
        - Patch embedding with learnable position embeddings
        - CLS token prepended, then K learnable meta tokens, then patches
        - Position embeddings cover CLS + meta + patches
        - N blocks selected by ``block_pattern`` (A=softmax, L=linear)
        - Extract CLS token at output (meta tokens discarded)
        - BatchNorm projection (critical for SIGReg to work)

    Args:
        image_size: Input image spatial size.
        patch_size: Patch size for the Conv2d projection.
        in_channels: Number of input channels (default: 3 for RGB).
        embed_dim: Token/embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Attention heads per block.
        block_pattern: String of A/L characters controlling block type.
                       Repeated cyclically if shorter than ``depth``.
        meta_tokens: Number of learnable prefix tokens (default: 4).
        linear_attn_eps: Epsilon for linear attention denominator.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        block_pattern: str = "ALALAL",
        meta_tokens: int = 4,
        linear_attn_eps: float = 1e-6,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.meta_tokens = meta_tokens
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.meta_token = nn.Parameter(torch.zeros(1, meta_tokens, embed_dim)) if meta_tokens > 0 else None
        # Position embeddings cover CLS + meta + patches
        total_tokens = 1 + meta_tokens + num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens, embed_dim))

        # Build block sequence from pattern
        pattern = block_pattern.upper()
        blocks: list[nn.Module] = []
        for i in range(depth):
            char = pattern[i % len(pattern)]
            if char == "L":
                blocks.append(LinearViTBlock(embed_dim, num_heads, eps=linear_attn_eps))
            else:
                # Default to standard attention for "A" or any unknown char
                blocks.append(ViTBlock(embed_dim, num_heads))
        self.blocks = nn.ModuleList(blocks)

        self.norm = nn.LayerNorm(embed_dim)

        # Post-encoder projection with BatchNorm (critical for SIGReg)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.meta_token is not None:
            nn.init.trunc_normal_(self.meta_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Encode image to CLS token embedding.

        Args:
            x: [B, C, H, W]

        Returns:
            [B, embed_dim]
        """
        B = x.shape[0]

        # Patch embed
        patches = self.patch_embed(x)  # [B, N_patches, D]

        # Build token sequence: CLS + meta + patches
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        if self.meta_token is not None and self.meta_tokens > 0:
            meta = self.meta_token.expand(B, -1, -1)  # [B, K, D]
            x = torch.cat([cls, meta, patches], dim=1)  # [B, 1+K+N, D]
        else:
            x = torch.cat([cls, patches], dim=1)  # [B, 1+N, D]

        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Extract CLS token only (meta tokens discarded)
        x = self.norm(x[:, 0])  # [B, D]

        # BatchNorm projection
        x = self.proj(x)  # [B, D]

        return x


# ---------------------------------------------------------------------------
# Weight Norm Projection Hook
# ---------------------------------------------------------------------------

class WeightNormProjection:
    """Post-optimizer-step hook that projects weight matrices to unit norm sphere.

    After each optimizer step, for all targeted modules (attention Q/K/V and
    MLP weight matrices), each output-oriented row vector is rescaled:

        W_{i,:} = W_{i,:} / ||W_{i,:}||_2 * sqrt(dim)

    This encourages stable gradient flow through the linear attention blocks.
    Off by default, enabled via USE_WEIGHT_NORM=1 env var.

    Usage:
        hook = WeightNormProjection(model)
        # After optimizer.step():
        hook.apply()
    """

    def __init__(self, model: nn.Module):
        self._targets: list[nn.Parameter] = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Skip bias, BatchNorm, LayerNorm weights
                if hasattr(module, "weight") and module.weight.ndim == 2:
                    self._targets.append(module.weight)

    @torch.no_grad()
    def apply(self) -> None:
        """Project all target weight rows to unit norm * sqrt(dim)."""
        for w in self._targets:
            dim = w.shape[1]
            norms = w.norm(dim=1, keepdim=True).clamp(min=1e-8)
            w.mul_(math.sqrt(dim) / norms)


# ---------------------------------------------------------------------------
# AdaLN, DiTBlock, DiTPredictor — copied from lewm.py (no cross-plugin imports)
# ---------------------------------------------------------------------------

class AdaLN(nn.Module):
    """Adaptive Layer Normalization -- modulates norm output with action-conditioned scale/shift."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, dim * 2)  # scale + shift
        # Zero-initialize for training stability (LE-WM paper)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        scale_shift = self.proj(cond)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class DiTBlock(nn.Module):
    """DiT transformer block with AdaLN action conditioning.

    Attention uses inner_dim = dim_head * num_heads (can differ from dim).
    MLP uses mlp_dim as intermediate size.
    """

    def __init__(self, dim: int, num_heads: int, cond_dim: int, dim_head: int = 64, mlp_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        inner_dim = dim_head * num_heads
        self.adaln1 = AdaLN(dim, cond_dim)
        self.q_proj = nn.Linear(dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(dim, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, dim)
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.attn_dropout = nn.Dropout(dropout)

        self.adaln2 = AdaLN(dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        h = self.adaln1(x, cond)
        B, N, _ = h.shape
        q = self.q_proj(h).reshape(B, N, self.num_heads, self.dim_head).transpose(1, 2)
        k = self.k_proj(h).reshape(B, N, self.num_heads, self.dim_head).transpose(1, 2)
        v = self.v_proj(h).reshape(B, N, self.num_heads, self.dim_head).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, -1)
        attn_out = self.attn_dropout(self.out_proj(attn_out))
        x = x + attn_out
        x = x + self.mlp(self.adaln2(x, cond))
        return x


class DiTPredictor(nn.Module):
    """DiT-based dynamics predictor with AdaLN action conditioning.

    Predicts z_{t+1} from (z_t, action_t) using transformer blocks
    with adaptive layer normalization modulated by the action embedding.
    """

    def __init__(
        self,
        embed_dim: int = 192,
        action_dim: int = 10,
        depth: int = 6,
        num_heads: int = 16,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, embed_dim, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, z: Tensor, action: Tensor) -> Tensor:
        """Predict next-step embedding.

        Args:
            z: [B, embed_dim] -- current latent state
            action: [B, action_dim] -- action taken

        Returns:
            [B, embed_dim] -- predicted next latent state
        """
        cond = self.action_encoder(action)  # [B, embed_dim]
        h = z.unsqueeze(1)  # [B, 1, embed_dim]
        for block in self.blocks:
            h = block(h, cond.unsqueeze(1))
        h = self.norm(h.squeeze(1))  # [B, embed_dim]
        return h


# ---------------------------------------------------------------------------
# Hybrid LE-WM World Model
# ---------------------------------------------------------------------------

class HybridLeWMModel(CrucibleModel):
    """Hybrid LE-WM: Latency-Optimal Hybrid Latent Emergent World Model.

    Replaces the standard ViT encoder with a HybridViTEncoder that interleaves
    softmax attention and linear attention blocks according to a configurable
    pattern.  The DiT predictor with AdaLN action conditioning is unchanged.

    Training flow (same as lewm):
        1. Encode all frames: z_t = HybridEncoder(o_t)
        2. For each transition: z_hat_{t+1} = Predictor(z_t.detach(), a_t)
        3. Loss = MSE(z_hat_{t+1}, z_{t+1}.detach()) + lambda * SIGReg(Z)
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 192,
        encoder_depth: int = 6,
        encoder_heads: int = 3,
        block_pattern: str = "ALALAL",
        meta_tokens: int = 4,
        linear_attn_eps: float = 1e-6,
        predictor_depth: int = 6,
        predictor_heads: int = 16,
        predictor_dim_head: int = 64,
        predictor_mlp_dim: int = 2048,
        action_dim: int = 10,
        sigreg_weight: float = 1.0,
        sigreg_projections: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sigreg_weight = sigreg_weight
        self.sigreg_projections = sigreg_projections

        self.encoder = HybridViTEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            block_pattern=block_pattern,
            meta_tokens=meta_tokens,
            linear_attn_eps=linear_attn_eps,
        )

        self.predictor = DiTPredictor(
            embed_dim=embed_dim,
            action_dim=action_dim,
            depth=predictor_depth,
            num_heads=predictor_heads,
            dim_head=predictor_dim_head,
            mlp_dim=predictor_mlp_dim,
            dropout=dropout,
        )

        # Projectors matching HF config (192 -> 2048 -> 192)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, embed_dim),
        )
        self.pred_proj = nn.Sequential(
            nn.Linear(embed_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, embed_dim),
        )

    def forward(
        self,
        frames: Tensor,
        actions: Tensor,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        """Forward pass over a frame sequence.

        Args:
            frames: [B, T, C, H, W] -- sequence of observations
            actions: [B, T-1, A] -- actions between consecutive frames

        Returns:
            dict with loss, pred_loss, sigreg, pred_embeddings,
            target_embeddings, embeddings, z_std
        """
        B, T, C, H, W = frames.shape

        # Encode all frames
        flat_frames = frames.reshape(B * T, C, H, W)
        z_all = self.encoder(flat_frames).reshape(B, T, -1)  # [B, T, D]

        # Predict next-frame embeddings for each transition
        total_pred_loss = torch.tensor(0.0, device=frames.device)
        num_transitions = T - 1

        pred_embeddings_list = []
        for t in range(num_transitions):
            z_pred = self.predictor(z_all[:, t].detach(), actions[:, t])
            target = z_all[:, t + 1].detach()
            pred_loss = F.mse_loss(z_pred, target)
            total_pred_loss = total_pred_loss + pred_loss
            pred_embeddings_list.append(z_pred)

        total_pred_loss = total_pred_loss / max(num_transitions, 1)

        # SIGReg: simple variance-based regularization
        z_flat = z_all.reshape(-1, self.embed_dim)  # [B*T, D]
        z_std = z_flat.std(dim=0)  # [D]
        sigreg_approx = (1.0 - z_std).pow(2).mean()

        loss = total_pred_loss + self.sigreg_weight * sigreg_approx

        return {
            "loss": loss,
            "pred_loss": total_pred_loss,
            "sigreg": sigreg_approx,
            "pred_embeddings": torch.stack(pred_embeddings_list, dim=1) if pred_embeddings_list else z_all[:, 1:],
            "target_embeddings": z_all[:, 1:].detach(),
            "embeddings": z_flat,
            "z_std": z_std,
        }

    def training_step(self, **batch: Any) -> dict[str, Tensor]:
        return self.forward(**batch)

    def validation_step(self, **batch: Any) -> dict[str, Tensor]:
        return self.forward(**batch)

    def encode(self, frames: Tensor) -> Tensor:
        """Encode frames to CLS token embeddings (for inference/planning)."""
        if frames.dim() == 4:
            return self.encoder(frames)
        B, T, C, H, W = frames.shape
        return self.encoder(frames.reshape(B * T, C, H, W)).reshape(B, T, -1)

    def predict_next(self, z: Tensor, action: Tensor) -> Tensor:
        """Predict next embedding from current embedding + action (for planning)."""
        return self.predictor(z, action)

    def param_groups(self) -> list[dict[str, Any]]:
        """Separate encoder and predictor for different learning rates."""
        return [
            {"params": list(self.encoder.parameters()), "name": "encoder"},
            {"params": list(self.predictor.parameters()), "name": "predictor"},
        ]

    @classmethod
    def modality(cls) -> str:
        return "world_model"


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

def _build_hybrid_lewm(args: Any) -> HybridLeWMModel:
    """Build Hybrid LE-WM from Crucible args namespace or env vars."""
    model = HybridLeWMModel(
        image_size=int(getattr(args, "image_size", os.environ.get("IMAGE_SIZE", "224"))),
        patch_size=int(getattr(args, "patch_size", os.environ.get("PATCH_SIZE", "14"))),
        in_channels=int(getattr(args, "image_channels", 3)),
        embed_dim=int(getattr(args, "model_dim", os.environ.get("MODEL_DIM", "192"))),
        encoder_depth=int(getattr(args, "encoder_depth", os.environ.get("ENCODER_DEPTH", "6"))),
        encoder_heads=int(getattr(args, "encoder_heads", os.environ.get("ENCODER_HEADS", "3"))),
        block_pattern=str(getattr(args, "block_pattern", os.environ.get("HYBRID_ENCODER_PATTERN", "ALALAL"))),
        meta_tokens=int(getattr(args, "meta_tokens", os.environ.get("META_TOKENS", "4"))),
        linear_attn_eps=float(getattr(args, "linear_attn_eps", os.environ.get("LINEAR_ATTN_EPS", "1e-6"))),
        predictor_depth=int(getattr(args, "predictor_depth", os.environ.get("PREDICTOR_DEPTH", "6"))),
        predictor_heads=int(getattr(args, "predictor_heads", os.environ.get("PREDICTOR_HEADS", "16"))),
        predictor_dim_head=int(getattr(args, "predictor_dim_head", os.environ.get("PREDICTOR_DIM_HEAD", "64"))),
        predictor_mlp_dim=int(getattr(args, "predictor_mlp_dim", os.environ.get("PREDICTOR_MLP_DIM", "2048"))),
        action_dim=int(getattr(args, "action_dim", os.environ.get("ACTION_DIM", "10"))),
        sigreg_weight=float(getattr(args, "sigreg_weight", os.environ.get("SIGREG_WEIGHT", "1.0"))),
        sigreg_projections=int(getattr(args, "sigreg_projections", os.environ.get("SIGREG_PROJECTIONS", "128"))),
        dropout=float(getattr(args, "dropout", os.environ.get("DROPOUT", "0.1"))),
    )

    # Load pretrained weights if specified
    pretrained = os.environ.get("PRETRAINED_WEIGHTS", "")
    if pretrained and os.path.exists(pretrained):
        weights = torch.load(pretrained, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(weights, strict=False)
        if missing:
            print(f"hybrid_lewm: {len(missing)} missing keys (fine-tuning from partial checkpoint)", file=sys.stderr)
        if unexpected:
            print(f"hybrid_lewm: {len(unexpected)} unexpected keys ignored", file=sys.stderr)

    return model


try:
    register_model("hybrid_lewm", _build_hybrid_lewm, source="local")
except ValueError:
    pass  # Already registered (e.g. auto-discovery loaded before direct import)


# ---------------------------------------------------------------------------
# torch.compile(fullgraph=True) fix: the registry loads plugins via
# importlib.util.exec_module() but never registers them in sys.modules.
# torch._dynamo needs to import_module(__name__) during tracing.
# ---------------------------------------------------------------------------
if __name__ not in sys.modules:
    import types as _types
    _m = _types.ModuleType(__name__)
    _m.__file__ = globals().get("__file__", "")
    _m.__dict__.update(globals())
    sys.modules[__name__] = _m
