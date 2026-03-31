"""Elastic LE-WM: Latent Emergent World Model with difficulty-aware compute routing.

Inspired by NVIDIA Nemotron Elastic, this architecture extends the base LE-WM
with elastic width+depth on the predictor (and optionally the encoder) plus a
lightweight DifficultyRouter that selects per-transition compute budgets.

Training proceeds in two stages:
    Stage 1 (warmup): Full-model forward, standard LE-WM loss.
    Stage 2 (elastic): Sandwich strategy — min/max/random budget each step.
        Encoder always runs at full capacity (targets must be consistent).
        DifficultyRouter selects predictor budget via Gumbel-Softmax.
        Knowledge-distillation loss aligns sub-model with full-model predictions.

Env vars (all existing lewm vars PLUS):
    ELASTIC_NUM_BUDGETS:       Number of compute budget levels (default: 4)
    ELASTIC_WARMUP_FRACTION:   Fraction of training at full model (default: 0.3)
    ELASTIC_KD_WEIGHT:         Knowledge-distillation loss weight (default: 0.5)
    ELASTIC_ROUTER_COST:       Cost penalty for expected FLOPS (default: 0.01)
    ELASTIC_ROUTER_ENTROPY:    Entropy bonus for budget distribution (default: 0.005)
    ELASTIC_GUMBEL_TEMP:       Initial Gumbel-Softmax temperature (default: 5.0)
    ELASTIC_GUMBEL_TEMP_MIN:   Minimum Gumbel temperature (default: 0.5)
    ELASTIC_FIXED_BUDGET:      Fixed budget ratio for eval/deploy (default: 0 = use router)
    ELASTIC_SANDWICH:          Use sandwich strategy (default: 1)
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from crucible.models.base import CrucibleModel
from crucible.models.registry import register_model

# Register this module in sys.modules early — required for @dataclass on Python 3.9
# when loaded via importlib plugin discovery (exec_module doesn't add to sys.modules).
if __name__ not in sys.modules:
    import types as _types
    sys.modules[__name__] = _types.ModuleType(__name__)

# ---------------------------------------------------------------------------
# Budget configuration
# ---------------------------------------------------------------------------

@dataclass
class BudgetConfig:
    """Defines a single compute budget level."""
    encoder_active_layers: list[int]    # indices of active encoder layers
    encoder_active_heads: int           # number of active heads (prefix)
    encoder_active_mlp: int             # number of active MLP neurons (prefix)
    predictor_active_layers: list[int]  # indices of active predictor layers
    predictor_active_heads: int         # number of active predictor heads (prefix)
    predictor_active_mlp: int           # number of active MLP neurons (prefix)
    flops_ratio: float                  # approximate FLOPS fraction vs full model


def compute_budget_configs(
    encoder_depth: int,
    encoder_heads: int,
    encoder_mlp_dim: int,
    predictor_depth: int,
    predictor_heads: int,
    predictor_mlp_dim: int,
    num_budgets: int = 4,
) -> list[BudgetConfig]:
    """Generate N budget configs from full model spec.

    Budget 0 is the smallest, budget N-1 is the full model.
    Smaller budgets use contiguous prefix subsets of layers/heads/MLP neurons.
    Auto-reduces num_budgets if model is too small.
    """
    # Auto-reduce budget count if model is too small
    min_dim = min(encoder_heads, predictor_heads, encoder_depth, predictor_depth)
    num_budgets = min(num_budgets, max(min_dim, 1))
    num_budgets = max(num_budgets, 1)

    configs = []
    for i in range(num_budgets):
        # Fraction for this budget level: evenly spaced from 1/N to 1.0
        frac = (i + 1) / num_budgets

        # Encoder layers: contiguous prefix
        enc_n_layers = max(1, round(encoder_depth * frac))
        enc_layers = list(range(enc_n_layers))

        # Encoder heads: prefix
        enc_n_heads = max(1, round(encoder_heads * frac))

        # Encoder MLP: prefix neurons
        enc_n_mlp = max(1, round(encoder_mlp_dim * frac))

        # Predictor layers: contiguous prefix
        pred_n_layers = max(1, round(predictor_depth * frac))
        pred_layers = list(range(pred_n_layers))

        # Predictor heads: prefix
        pred_n_heads = max(1, round(predictor_heads * frac))

        # Predictor MLP: prefix neurons
        pred_n_mlp = max(1, round(predictor_mlp_dim * frac))

        # Approximate FLOPS ratio (simplified: proportional to active params)
        enc_ratio = (enc_n_layers / encoder_depth) * (enc_n_heads / encoder_heads)
        pred_ratio = (pred_n_layers / predictor_depth) * (pred_n_heads / predictor_heads)
        flops_ratio = 0.5 * enc_ratio + 0.5 * pred_ratio  # weighted average

        configs.append(BudgetConfig(
            encoder_active_layers=enc_layers,
            encoder_active_heads=enc_n_heads,
            encoder_active_mlp=enc_n_mlp,
            predictor_active_layers=pred_layers,
            predictor_active_heads=pred_n_heads,
            predictor_active_mlp=pred_n_mlp,
            flops_ratio=flops_ratio,
        ))

    return configs


# ---------------------------------------------------------------------------
# Elastic ViT Encoder
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Convert image to patch embeddings."""

    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class ElasticViTBlock(nn.Module):
    """ViT transformer block with elastic width+depth.

    Supports:
        - Head prefix masking: use first k of H attention heads
        - MLP neuron prefix masking: use first m of M hidden neurons
        - Depth gamma: if active_heads==0 or layer is skipped, return input unchanged
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.mlp_hidden = int(dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(dim)
        # Custom Q/K/V for head prefix slicing
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

        self.norm2 = nn.LayerNorm(dim)
        # MLP with separable fc1/fc2 for neuron prefix slicing
        self.fc1 = nn.Linear(dim, self.mlp_hidden)
        self.fc2 = nn.Linear(self.mlp_hidden, dim)

    def forward(
        self,
        x: Tensor,
        active_heads: int | None = None,
        active_mlp: int | None = None,
        depth_active: bool = True,
    ) -> Tensor:
        """Forward with elastic width+depth.

        Args:
            x: [B, N, D]
            active_heads: number of attention heads to use (prefix). None = all.
            active_mlp: number of MLP hidden neurons to use (prefix). None = all.
            depth_active: if False, return input unchanged (skip layer).
        """
        if not depth_active:
            return x

        B, N, D = x.shape
        heads = active_heads if active_heads is not None else self.num_heads
        heads = min(heads, self.num_heads)
        mlp_n = active_mlp if active_mlp is not None else self.mlp_hidden
        mlp_n = min(mlp_n, self.mlp_hidden)

        # ----- Attention with head prefix masking -----
        h = self.norm1(x)
        active_dim = heads * self.head_dim

        # Slice Q/K/V projections to only produce active heads
        q = F.linear(h, self.q_proj.weight[:active_dim], self.q_proj.bias[:active_dim] if self.q_proj.bias is not None else None)
        k = F.linear(h, self.k_proj.weight[:active_dim], self.k_proj.bias[:active_dim] if self.k_proj.bias is not None else None)
        v = F.linear(h, self.v_proj.weight[:active_dim], self.v_proj.bias[:active_dim] if self.v_proj.bias is not None else None)

        q = q.reshape(B, N, heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, active_dim)

        # Output projection: only use rows for active heads
        out = F.linear(attn_out, self.out_proj.weight[:, :active_dim], self.out_proj.bias)
        x = x + out

        # ----- MLP with neuron prefix masking -----
        h2 = self.norm2(x)
        mlp_out = F.linear(h2, self.fc1.weight[:mlp_n], self.fc1.bias[:mlp_n] if self.fc1.bias is not None else None)
        mlp_out = F.gelu(mlp_out)
        mlp_out = F.linear(mlp_out, self.fc2.weight[:, :mlp_n], self.fc2.bias)
        x = x + mlp_out

        return x


class ElasticViTEncoder(nn.Module):
    """Vision Transformer encoder with elastic width+depth.

    When no budget is specified, runs at full capacity.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 12,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_hidden = int(embed_dim * 4.0)

        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            ElasticViTBlock(embed_dim, num_heads) for _ in range(depth)
        ])

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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor, budget: BudgetConfig | None = None) -> Tensor:
        """Encode image to CLS token embedding.

        Args:
            x: [B, C, H, W]
            budget: optional BudgetConfig for elastic forward

        Returns:
            [B, embed_dim]
        """
        B = x.shape[0]

        patches = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, patches], dim=1)
        x = x + self.pos_embed

        if budget is not None:
            active_layers = set(budget.encoder_active_layers)
            for i, block in enumerate(self.blocks):
                x = block(
                    x,
                    active_heads=budget.encoder_active_heads,
                    active_mlp=budget.encoder_active_mlp,
                    depth_active=(i in active_layers),
                )
        else:
            for block in self.blocks:
                x = block(x)

        x = self.norm(x[:, 0])
        x = self.proj(x)

        return x


# ---------------------------------------------------------------------------
# Elastic DiT Predictor with AdaLN
# ---------------------------------------------------------------------------

class AdaLN(nn.Module):
    """Adaptive Layer Normalization with action-conditioned scale/shift."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, dim * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        scale_shift = self.proj(cond)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class ElasticDiTBlock(nn.Module):
    """DiT transformer block with elastic width+depth and AdaLN action conditioning.

    AdaLN always operates at full dim (cheap, needs full conditioning signal).
    Head prefix masking and MLP neuron prefix masking for elastic compute.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.inner_dim = dim_head * num_heads
        self.mlp_dim = mlp_dim

        self.adaln1 = AdaLN(dim, cond_dim)
        self.q_proj = nn.Linear(dim, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.inner_dim, bias=False)
        self.out_proj = nn.Linear(self.inner_dim, dim)
        self.attn_dropout = nn.Dropout(dropout)

        self.adaln2 = AdaLN(dim, cond_dim)
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.mlp_dropout1 = nn.Dropout(dropout)
        self.mlp_dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        active_heads: int | None = None,
        active_mlp: int | None = None,
        depth_active: bool = True,
    ) -> Tensor:
        """Forward with elastic width+depth.

        Args:
            x: [B, N, D]
            cond: [B, N, cond_dim] or [B, cond_dim] conditioning signal
            active_heads: number of attention heads to use (prefix). None = all.
            active_mlp: number of MLP hidden neurons to use (prefix). None = all.
            depth_active: if False, return input unchanged (skip layer).
        """
        if not depth_active:
            return x

        B, N, _ = x.shape
        heads = active_heads if active_heads is not None else self.num_heads
        heads = min(heads, self.num_heads)
        mlp_n = active_mlp if active_mlp is not None else self.mlp_dim
        mlp_n = min(mlp_n, self.mlp_dim)

        # ----- Attention with head prefix masking -----
        h = self.adaln1(x, cond)
        active_inner = heads * self.dim_head

        q = F.linear(h, self.q_proj.weight[:active_inner])
        k = F.linear(h, self.k_proj.weight[:active_inner])
        v = F.linear(h, self.v_proj.weight[:active_inner])

        q = q.reshape(B, N, heads, self.dim_head).transpose(1, 2)
        k = k.reshape(B, N, heads, self.dim_head).transpose(1, 2)
        v = v.reshape(B, N, heads, self.dim_head).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, active_inner)
        attn_out = self.attn_dropout(F.linear(attn_out, self.out_proj.weight[:, :active_inner], self.out_proj.bias))
        x = x + attn_out

        # ----- MLP with neuron prefix masking -----
        h2 = self.adaln2(x, cond)
        mlp_out = F.linear(h2, self.fc1.weight[:mlp_n], self.fc1.bias[:mlp_n] if self.fc1.bias is not None else None)
        mlp_out = F.gelu(mlp_out)
        mlp_out = self.mlp_dropout1(mlp_out)
        mlp_out = F.linear(mlp_out, self.fc2.weight[:, :mlp_n], self.fc2.bias)
        mlp_out = self.mlp_dropout2(mlp_out)
        x = x + mlp_out

        return x


class ElasticDiTPredictor(nn.Module):
    """DiT-based dynamics predictor with elastic width+depth.

    Action encoder stays at full width (cheap, needs accurate conditioning).
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
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim

        # Action encoder stays at full width
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.blocks = nn.ModuleList([
            ElasticDiTBlock(embed_dim, num_heads, embed_dim, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        z: Tensor,
        action: Tensor,
        budget: BudgetConfig | None = None,
    ) -> Tensor:
        """Predict next-step embedding with optional elastic budget.

        Args:
            z: [B, embed_dim] -- current latent state
            action: [B, action_dim] -- action taken
            budget: optional BudgetConfig for elastic forward

        Returns:
            [B, embed_dim] -- predicted next latent state
        """
        cond = self.action_encoder(action)
        h = z.unsqueeze(1)  # [B, 1, embed_dim]

        if budget is not None:
            active_layers = set(budget.predictor_active_layers)
            for i, block in enumerate(self.blocks):
                h = block(
                    h,
                    cond.unsqueeze(1),
                    active_heads=budget.predictor_active_heads,
                    active_mlp=budget.predictor_active_mlp,
                    depth_active=(i in active_layers),
                )
        else:
            for block in self.blocks:
                h = block(h, cond.unsqueeze(1))

        h = self.norm(h.squeeze(1))
        return h


# ---------------------------------------------------------------------------
# Difficulty Router
# ---------------------------------------------------------------------------

class DifficultyRouter(nn.Module):
    """2-layer MLP that selects compute budget per transition.

    Input: concat(z_t, action_t)
    Output: budget distribution over num_budgets levels
    Training: Gumbel-Softmax with temperature annealing
    Inference: argmax (or fixed budget via env var)
    """

    def __init__(
        self,
        embed_dim: int,
        action_dim: int,
        num_budgets: int = 4,
        temperature: float = 5.0,
    ):
        super().__init__()
        self.num_budgets = num_budgets
        self.temperature = temperature
        self.net = nn.Sequential(
            nn.Linear(embed_dim + action_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_budgets),
        )

    def forward(
        self,
        z_t: Tensor,
        action: Tensor,
        hard: bool = False,
        temperature: float | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Select budget for each sample in the batch.

        Args:
            z_t: [B, embed_dim] -- current latent state
            action: [B, action_dim] -- action
            hard: if True, use hard (one-hot) Gumbel-Softmax
            temperature: override temperature for Gumbel-Softmax

        Returns:
            (budget_weights [B, num_budgets], logits [B, num_budgets])
        """
        inp = torch.cat([z_t, action], dim=-1)
        logits = self.net(inp)  # [B, num_budgets]

        temp = temperature if temperature is not None else self.temperature

        if self.training:
            weights = F.gumbel_softmax(logits, tau=temp, hard=hard, dim=-1)
        else:
            # Inference: argmax
            idx = logits.argmax(dim=-1)
            weights = F.one_hot(idx, self.num_budgets).float()

        return weights, logits


# ---------------------------------------------------------------------------
# Elastic LE-WM World Model
# ---------------------------------------------------------------------------

class ElasticLeWMModel(CrucibleModel):
    """Elastic LE-WM: Latent Emergent World Model with difficulty-aware compute routing.

    Two-stage training:
        Stage 1 (warmup): Full-model forward, standard LE-WM loss.
        Stage 2 (elastic): Router selects predictor budget, sandwich strategy,
            knowledge distillation from full predictor to sub-model.

    The encoder ALWAYS runs at full capacity (targets must be consistent).
    Only the predictor has elastic compute.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 192,
        encoder_depth: int = 6,
        encoder_heads: int = 3,
        predictor_depth: int = 6,
        predictor_heads: int = 16,
        predictor_dim_head: int = 64,
        predictor_mlp_dim: int = 2048,
        action_dim: int = 10,
        sigreg_weight: float = 1.0,
        sigreg_projections: int = 128,
        dropout: float = 0.1,
        num_budgets: int = 4,
        warmup_fraction: float = 0.3,
        kd_weight: float = 0.5,
        router_cost: float = 0.01,
        router_entropy: float = 0.005,
        gumbel_temp: float = 5.0,
        gumbel_temp_min: float = 0.5,
        use_sandwich: bool = True,
        fixed_budget: float = 0.0,
        total_steps: int = 100000,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sigreg_weight = sigreg_weight
        self.sigreg_projections = sigreg_projections
        self.num_budgets = num_budgets
        self.warmup_fraction = warmup_fraction
        self.kd_weight = kd_weight
        self.router_cost = router_cost
        self.router_entropy = router_entropy
        self.gumbel_temp_init = gumbel_temp
        self.gumbel_temp_min = gumbel_temp_min
        self.use_sandwich = use_sandwich
        self.fixed_budget = fixed_budget
        self.total_steps = total_steps
        self.action_dim = action_dim

        # Track training step for stage transition + temperature annealing
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))

        # Encoder (always full capacity)
        self.encoder = ElasticViTEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
        )

        # Elastic predictor
        self.predictor = ElasticDiTPredictor(
            embed_dim=embed_dim,
            action_dim=action_dim,
            depth=predictor_depth,
            num_heads=predictor_heads,
            dim_head=predictor_dim_head,
            mlp_dim=predictor_mlp_dim,
            dropout=dropout,
        )

        # Projectors (matching HF config: embed_dim -> 2048 -> embed_dim)
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

        # Difficulty Router
        encoder_mlp_hidden = int(embed_dim * 4.0)
        self.budget_configs = compute_budget_configs(
            encoder_depth=encoder_depth,
            encoder_heads=encoder_heads,
            encoder_mlp_dim=encoder_mlp_hidden,
            predictor_depth=predictor_depth,
            predictor_heads=predictor_heads,
            predictor_mlp_dim=predictor_mlp_dim,
            num_budgets=num_budgets,
        )
        self.actual_num_budgets = len(self.budget_configs)

        self.router = DifficultyRouter(
            embed_dim=embed_dim,
            action_dim=action_dim,
            num_budgets=self.actual_num_budgets,
            temperature=gumbel_temp,
        )

    @property
    def current_step(self) -> int:
        return self._step.item()

    @property
    def is_elastic_stage(self) -> bool:
        warmup_steps = int(self.warmup_fraction * self.total_steps)
        return self.current_step > warmup_steps

    @property
    def current_gumbel_temp(self) -> float:
        """Anneal Gumbel temperature linearly from init to min over elastic stage."""
        if not self.is_elastic_stage:
            return self.gumbel_temp_init
        warmup_steps = int(self.warmup_fraction * self.total_steps)
        elastic_steps = self.total_steps - warmup_steps
        if elastic_steps <= 0:
            return self.gumbel_temp_min
        elapsed = self.current_step - warmup_steps
        progress = min(elapsed / elastic_steps, 1.0)
        return self.gumbel_temp_init + progress * (self.gumbel_temp_min - self.gumbel_temp_init)

    def _get_fixed_budget_config(self) -> BudgetConfig | None:
        """If ELASTIC_FIXED_BUDGET is set, return the matching budget config."""
        if self.fixed_budget <= 0.0:
            return None
        # Find closest budget by flops_ratio
        best = min(self.budget_configs, key=lambda b: abs(b.flops_ratio - self.fixed_budget))
        return best

    def _forward_full(
        self,
        frames: Tensor,
        actions: Tensor,
    ) -> dict[str, Tensor]:
        """Stage 1 forward: full model, standard LE-WM loss."""
        B, T, C, H, W = frames.shape

        # Encode all frames at full capacity
        flat_frames = frames.reshape(B * T, C, H, W)
        z_all = self.encoder(flat_frames).reshape(B, T, -1)

        # Predict next-frame embeddings
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

        # SIGReg approximation
        z_flat = z_all.reshape(-1, self.embed_dim)
        z_std = z_flat.std(dim=0)
        sigreg_approx = (1.0 - z_std).pow(2).mean()

        loss = total_pred_loss + self.sigreg_weight * sigreg_approx

        # Zero placeholders for elastic-stage losses
        zero = torch.tensor(0.0, device=frames.device)
        budget_dist = torch.zeros(self.actual_num_budgets, device=frames.device)
        budget_dist[-1] = 1.0  # full budget

        return {
            "loss": loss,
            "pred_loss": total_pred_loss,
            "sigreg": sigreg_approx,
            "kd_loss": zero,
            "router_loss": zero,
            "budget_distribution": budget_dist,
            "pred_embeddings": torch.stack(pred_embeddings_list, dim=1) if pred_embeddings_list else z_all[:, 1:],
            "target_embeddings": z_all[:, 1:].detach(),
            "embeddings": z_flat,
            "z_std": z_std,
        }

    def _forward_elastic(
        self,
        frames: Tensor,
        actions: Tensor,
    ) -> dict[str, Tensor]:
        """Stage 2 forward: elastic predictor with router, KD, sandwich."""
        B, T, C, H, W = frames.shape

        # Encoder ALWAYS at full capacity
        flat_frames = frames.reshape(B * T, C, H, W)
        z_all = self.encoder(flat_frames).reshape(B, T, -1)

        num_transitions = T - 1
        total_pred_loss = torch.tensor(0.0, device=frames.device)
        total_kd_loss = torch.tensor(0.0, device=frames.device)
        total_router_loss = torch.tensor(0.0, device=frames.device)
        budget_dist_accum = torch.zeros(self.actual_num_budgets, device=frames.device)
        pred_embeddings_list = []

        temp = self.current_gumbel_temp

        for t in range(num_transitions):
            z_t = z_all[:, t].detach()
            target = z_all[:, t + 1].detach()
            action_t = actions[:, t]

            # Full predictor forward (teacher)
            z_pred_full = self.predictor(z_t, action_t)
            pred_loss = F.mse_loss(z_pred_full, target)
            total_pred_loss = total_pred_loss + pred_loss
            pred_embeddings_list.append(z_pred_full)

            # Router selects budget
            fixed_budget = self._get_fixed_budget_config()
            if fixed_budget is not None:
                # Fixed budget mode: skip router
                z_pred_sub = self.predictor(z_t, action_t, budget=fixed_budget)
                kd_loss = F.mse_loss(z_pred_sub, z_pred_full.detach())
                total_kd_loss = total_kd_loss + kd_loss
                # Find index of fixed budget
                for bi, bc in enumerate(self.budget_configs):
                    if bc is fixed_budget:
                        budget_dist_accum[bi] += 1.0
                        break
            else:
                # Router-selected budget with sandwich strategy
                weights, logits = self.router(z_t.detach(), action_t.detach(), temperature=temp)
                budget_dist_accum = budget_dist_accum + weights.mean(dim=0).detach()

                if self.use_sandwich:
                    # Sandwich: min, max, and 1 random budget
                    budgets_to_run = [0, self.actual_num_budgets - 1]
                    if self.actual_num_budgets > 2:
                        rand_idx = torch.randint(1, self.actual_num_budgets - 1, (1,)).item()
                        if rand_idx not in budgets_to_run:
                            budgets_to_run.append(rand_idx)
                else:
                    budgets_to_run = list(range(self.actual_num_budgets))

                kd_loss_t = torch.tensor(0.0, device=frames.device)
                for bi in budgets_to_run:
                    bc = self.budget_configs[bi]
                    z_pred_sub = self.predictor(z_t, action_t, budget=bc)
                    kd = F.mse_loss(z_pred_sub, z_pred_full.detach())
                    # Weight by router's soft assignment
                    kd_loss_t = kd_loss_t + weights[:, bi].mean() * kd

                total_kd_loss = total_kd_loss + kd_loss_t / len(budgets_to_run)

                # Router loss: cost penalty + entropy bonus
                flops_ratios = torch.tensor(
                    [bc.flops_ratio for bc in self.budget_configs],
                    device=frames.device, dtype=weights.dtype,
                )
                expected_cost = (weights * flops_ratios.unsqueeze(0)).sum(dim=-1).mean()
                # Entropy of budget distribution
                entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean()
                router_loss = self.router_cost * expected_cost - self.router_entropy * entropy
                total_router_loss = total_router_loss + router_loss

        num_t = max(num_transitions, 1)
        total_pred_loss = total_pred_loss / num_t
        total_kd_loss = total_kd_loss / num_t
        total_router_loss = total_router_loss / num_t
        budget_dist_accum = budget_dist_accum / num_t

        # SIGReg
        z_flat = z_all.reshape(-1, self.embed_dim)
        z_std = z_flat.std(dim=0)
        sigreg_approx = (1.0 - z_std).pow(2).mean()

        loss = (
            total_pred_loss
            + self.sigreg_weight * sigreg_approx
            + self.kd_weight * total_kd_loss
            + total_router_loss
        )

        return {
            "loss": loss,
            "pred_loss": total_pred_loss,
            "sigreg": sigreg_approx,
            "kd_loss": total_kd_loss,
            "router_loss": total_router_loss,
            "budget_distribution": budget_dist_accum,
            "pred_embeddings": torch.stack(pred_embeddings_list, dim=1) if pred_embeddings_list else z_all[:, 1:],
            "target_embeddings": z_all[:, 1:].detach(),
            "embeddings": z_flat,
            "z_std": z_std,
        }

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
            dict with loss, pred_loss, sigreg, kd_loss, router_loss,
            budget_distribution, pred_embeddings, target_embeddings,
            embeddings, z_std
        """
        # Increment step counter during training
        if self.training:
            self._step += 1

        if self.is_elastic_stage:
            return self._forward_elastic(frames, actions)
        else:
            return self._forward_full(frames, actions)

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

    def predict_next(self, z: Tensor, action: Tensor, budget: BudgetConfig | None = None) -> Tensor:
        """Predict next embedding from current embedding + action."""
        return self.predictor(z, action, budget=budget)

    def param_groups(self) -> list[dict[str, Any]]:
        """Separate encoder, predictor, and router for different learning rates."""
        return [
            {"params": list(self.encoder.parameters()), "name": "encoder"},
            {"params": list(self.predictor.parameters()), "name": "predictor"},
            {"params": list(self.router.parameters()), "name": "router"},
        ]

    @classmethod
    def modality(cls) -> str:
        return "world_model"


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

def _build_elastic_lewm(args: Any) -> ElasticLeWMModel:
    """Build Elastic LE-WM from Crucible args namespace or env vars."""
    model = ElasticLeWMModel(
        image_size=int(getattr(args, "image_size", os.environ.get("IMAGE_SIZE", "224"))),
        patch_size=int(getattr(args, "patch_size", os.environ.get("PATCH_SIZE", "14"))),
        in_channels=int(getattr(args, "image_channels", 3)),
        embed_dim=int(getattr(args, "model_dim", os.environ.get("MODEL_DIM", "192"))),
        encoder_depth=int(getattr(args, "encoder_depth", os.environ.get("ENCODER_DEPTH", "6"))),
        encoder_heads=int(getattr(args, "encoder_heads", os.environ.get("ENCODER_HEADS", "3"))),
        predictor_depth=int(getattr(args, "predictor_depth", os.environ.get("PREDICTOR_DEPTH", "6"))),
        predictor_heads=int(getattr(args, "predictor_heads", os.environ.get("PREDICTOR_HEADS", "16"))),
        predictor_dim_head=int(getattr(args, "predictor_dim_head", os.environ.get("PREDICTOR_DIM_HEAD", "64"))),
        predictor_mlp_dim=int(getattr(args, "predictor_mlp_dim", os.environ.get("PREDICTOR_MLP_DIM", "2048"))),
        action_dim=int(getattr(args, "action_dim", os.environ.get("ACTION_DIM", "10"))),
        sigreg_weight=float(getattr(args, "sigreg_weight", os.environ.get("SIGREG_WEIGHT", "1.0"))),
        sigreg_projections=int(getattr(args, "sigreg_projections", os.environ.get("SIGREG_PROJECTIONS", "128"))),
        dropout=float(getattr(args, "dropout", os.environ.get("DROPOUT", "0.1"))),
        num_budgets=int(os.environ.get("ELASTIC_NUM_BUDGETS", "4")),
        warmup_fraction=float(os.environ.get("ELASTIC_WARMUP_FRACTION", "0.3")),
        kd_weight=float(os.environ.get("ELASTIC_KD_WEIGHT", "0.5")),
        router_cost=float(os.environ.get("ELASTIC_ROUTER_COST", "0.01")),
        router_entropy=float(os.environ.get("ELASTIC_ROUTER_ENTROPY", "0.005")),
        gumbel_temp=float(os.environ.get("ELASTIC_GUMBEL_TEMP", "5.0")),
        gumbel_temp_min=float(os.environ.get("ELASTIC_GUMBEL_TEMP_MIN", "0.5")),
        use_sandwich=bool(int(os.environ.get("ELASTIC_SANDWICH", "1"))),
        fixed_budget=float(os.environ.get("ELASTIC_FIXED_BUDGET", "0")),
    )

    # Load pretrained weights if specified
    pretrained = os.environ.get("PRETRAINED_WEIGHTS", "")
    if pretrained and os.path.exists(pretrained):
        weights = torch.load(pretrained, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(weights, strict=False)
        if missing:
            print(f"elastic_lewm: {len(missing)} missing keys (fine-tuning from partial checkpoint)", file=sys.stderr)
        if unexpected:
            print(f"elastic_lewm: {len(unexpected)} unexpected keys ignored", file=sys.stderr)

    return model


try:
    register_model("elastic_lewm", _build_elastic_lewm, source="local")
except ValueError:
    pass  # Already registered (e.g. auto-discovery loaded before direct import)

# ---------------------------------------------------------------------------
# torch.compile(fullgraph=True) fix: register module in sys.modules
# ---------------------------------------------------------------------------
if __name__ not in sys.modules:
    import types as _types
    _m = _types.ModuleType(__name__)
    _m.__file__ = globals().get("__file__", "")
    _m.__dict__.update(globals())
    sys.modules[__name__] = _m
