"""JEPA World Model — predicts future latent states from observations + actions.

A Joint Embedding Predictive Architecture (JEPA) that learns to predict
future visual observations in embedding space.  Inspired by
architectures like V-JEPA and Le-WM, scaled down for demonstration.

Architecture:
    - CNN encoder: maps frames [B, C, H, W] → latent embeddings [B, D]
    - MLP predictor: maps (z_t, action_t) → predicted z_{t+1}
    - EMA target encoder: provides stop-gradient targets for prediction
    - Loss: MSE in embedding space + variance regularization (VICReg-style)

Usage::

    MODEL_FAMILY=jepa_wm BATCH_SIZE=8 ITERATIONS=2000 NUM_FRAMES=4 \\
        PYTHONPATH=src python -m crucible.training.generic_backend
"""
from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from crucible.models.base import CrucibleModel
from crucible.models.registry import register_model


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class CNNEncoder(nn.Module):
    """Simple CNN that maps images to flat latent vectors."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 128,
        base_channels: int = 32,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Linear(base_channels * 4, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.net(x))


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class MLPPredictor(nn.Module):
    """Predicts next-frame embedding from current embedding + action."""

    def __init__(self, embed_dim: int = 128, action_dim: int = 2, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, z: Tensor, action: Tensor) -> Tensor:
        return self.net(torch.cat([z, action], dim=-1))


# ---------------------------------------------------------------------------
# JEPA World Model
# ---------------------------------------------------------------------------

class JEPAWorldModel(CrucibleModel):
    """JEPA-style world model with EMA target encoder.

    Given a sequence of frames and actions:
      1. Encode each frame with the online encoder
      2. For each transition (t → t+1), predict z_{t+1} from (z_t, a_t)
      3. Compute targets with the EMA encoder (stop-gradient)
      4. Loss = prediction MSE + variance regularization

    The variance regularization encourages the encoder to spread
    representations across the embedding dimensions, preventing collapse.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 128,
        action_dim: int = 2,
        base_channels: int = 32,
        predictor_hidden: int = 256,
        ema_decay: float = 0.99,
        var_weight: float = 0.1,
        var_target: float = 1.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ema_decay = ema_decay
        self.var_weight = var_weight
        self.var_target = var_target

        # Online encoder + predictor
        self.encoder = CNNEncoder(in_channels, embed_dim, base_channels)
        self.predictor = MLPPredictor(embed_dim, action_dim, predictor_hidden)

        # EMA target encoder (no gradients)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _update_ema(self) -> None:
        """Exponential moving average update for target encoder."""
        for online, target in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            target.data.mul_(self.ema_decay).add_(online.data, alpha=1.0 - self.ema_decay)

    def forward(
        self,
        frames: Tensor,
        actions: Tensor,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        """Forward pass over a frame sequence.

        Args:
            frames: [B, T, C, H, W] — sequence of observations
            actions: [B, T-1, A] — actions between consecutive frames

        Returns:
            dict with ``loss``, ``pred_loss``, ``var_reg``
        """
        B, T, C, H, W = frames.shape

        # Encode all frames with online encoder
        flat_frames = frames.reshape(B * T, C, H, W)
        z_online = self.encoder(flat_frames).reshape(B, T, -1)

        # Encode all frames with target encoder (stop-gradient)
        with torch.no_grad():
            z_target = self.target_encoder(flat_frames).reshape(B, T, -1)

        # Predict next-frame embeddings for each transition
        total_pred_loss = torch.tensor(0.0, device=frames.device)
        num_transitions = T - 1

        for t in range(num_transitions):
            z_pred = self.predictor(z_online[:, t], actions[:, t])
            pred_loss = F.mse_loss(z_pred, z_target[:, t + 1])
            total_pred_loss = total_pred_loss + pred_loss

        total_pred_loss = total_pred_loss / max(num_transitions, 1)

        # Variance regularization: prevent representation collapse
        # Compute std across batch dimension for each embedding dim
        z_std = z_online.reshape(-1, self.embed_dim).std(dim=0)
        var_reg = torch.relu(self.var_target - z_std).mean()

        # Combined loss
        loss = total_pred_loss + self.var_weight * var_reg

        # Update EMA target encoder
        self._update_ema()

        return {
            "loss": loss,
            "pred_loss": total_pred_loss,
            "var_reg": var_reg,
        }

    def training_step(self, **batch: Any) -> dict[str, Tensor]:
        return self.forward(**batch)

    def validation_step(self, **batch: Any) -> dict[str, Tensor]:
        return self.forward(**batch)

    def encode(self, frames: Tensor) -> Tensor:
        """Encode frames to embeddings (for inference)."""
        if frames.dim() == 4:
            return self.encoder(frames)
        B, T, C, H, W = frames.shape
        return self.encoder(frames.reshape(B * T, C, H, W)).reshape(B, T, -1)

    def predict_next(self, z: Tensor, action: Tensor) -> Tensor:
        """Predict next embedding from current embedding + action (for inference)."""
        return self.predictor(z, action)

    @classmethod
    def modality(cls) -> str:
        return "world_model"


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

def _build_jepa_wm(args: Any) -> JEPAWorldModel:
    """Build a JEPA world model from Crucible args namespace."""
    return JEPAWorldModel(
        in_channels=getattr(args, "image_channels", 3),
        embed_dim=getattr(args, "model_dim", 128),
        action_dim=getattr(args, "action_dim", 2),
        base_channels=getattr(args, "base_channels", 32),
        predictor_hidden=getattr(args, "predictor_hidden", 256),
        ema_decay=getattr(args, "ema_decay", 0.99),
        var_weight=getattr(args, "var_weight", 0.1),
    )


def register() -> None:
    """Register the JEPA world model family with Crucible."""
    register_model("jepa_wm", _build_jepa_wm)
