"""DDPM UNet — a small denoising diffusion model for image generation.

This is a real, trainable DDPM implementation designed to work with
Crucible's generic training backend.  It trains on 28x28 grayscale
images (MNIST) by default but generalizes to other resolutions.

Architecture:
    - Sinusoidal time embeddings
    - 3-level UNet with residual blocks and skip connections
    - GroupNorm + SiLU activations
    - Downsampling via strided conv, upsampling via transposed conv

Usage::

    MODEL_FAMILY=ddpm_unet BATCH_SIZE=32 ITERATIONS=5000 \\
        PYTHONPATH=src python -m crucible.training.generic_backend
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from crucible.models.base import CrucibleModel
from crucible.models.registry import register_model


# ---------------------------------------------------------------------------
# Diffusion schedule helpers
# ---------------------------------------------------------------------------

def linear_beta_schedule(timesteps: int) -> Tensor:
    """Linear beta schedule from DDPM paper."""
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = min(scale * 0.02, 0.999)  # Cap to avoid alpha < 0
    return torch.linspace(beta_start, beta_end, timesteps)


class DiffusionSchedule:
    """Precomputed diffusion coefficients."""

    def __init__(self, timesteps: int = 1000):
        self.timesteps = timesteps
        betas = linear_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register = {}
        self.register["betas"] = betas
        self.register["sqrt_alphas_cumprod"] = torch.sqrt(alphas_cumprod)
        self.register["sqrt_one_minus_alphas_cumprod"] = torch.sqrt(1.0 - alphas_cumprod)

    def to(self, device: torch.device) -> "DiffusionSchedule":
        for k, v in self.register.items():
            self.register[k] = v.to(device)
        return self

    def q_sample(self, x_start: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Forward diffusion: add noise at timestep t."""
        sqrt_alpha = self.register["sqrt_alphas_cumprod"][t]
        sqrt_one_minus = self.register["sqrt_one_minus_alphas_cumprod"][t]
        # Reshape for broadcasting: [B, 1, 1, 1]
        while sqrt_alpha.dim() < x_start.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)
        return sqrt_alpha * x_start + sqrt_one_minus * noise


# ---------------------------------------------------------------------------
# UNet building blocks
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock(nn.Module):
    """Residual block with time conditioning."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        # Add time embedding
        h = h + self.time_mlp(F.silu(t_emb))[:, :, None, None]
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------

class SmallUNet(nn.Module):
    """3-level UNet for 28x28 or 32x32 images."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        time_dim: int = 128,
    ):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Input conv
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = base_channels
        encoder_channels = [ch]
        for mult in channel_mults:
            out_ch = base_channels * mult
            self.down_blocks.append(ResBlock(ch, out_ch, time_dim))
            ch = out_ch
            encoder_channels.append(ch)
            self.downsamples.append(Downsample(ch))

        # Middle
        self.mid_block1 = ResBlock(ch, ch, time_dim)
        self.mid_block2 = ResBlock(ch, ch, time_dim)

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            self.upsamples.append(Upsample(ch))
            # Skip connection doubles input channels
            self.up_blocks.append(ResBlock(ch + encoder_channels.pop(), out_ch, time_dim))
            ch = out_ch

        # Output
        self.final_norm = nn.GroupNorm(min(8, ch), ch)
        self.final_conv = nn.Conv2d(ch, in_channels, 3, padding=1)
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        h = self.input_conv(x)

        # Encoder
        skips = [h]
        for down_block, downsample in zip(self.down_blocks, self.downsamples):
            h = down_block(h, t_emb)
            skips.append(h)
            h = downsample(h)

        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)

        # Decoder
        for up_block, upsample in zip(self.up_blocks, self.upsamples):
            h = upsample(h)
            skip = skips.pop()
            # Handle size mismatch from odd spatial dims
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = up_block(h, t_emb)

        return self.final_conv(F.silu(self.final_norm(h)))


# ---------------------------------------------------------------------------
# CrucibleModel wrapper
# ---------------------------------------------------------------------------

class DDPMModel(CrucibleModel):
    """DDPM wrapper that implements the Crucible training contract.

    The model handles its own loss computation: given a batch of images,
    it samples random timesteps, adds noise, predicts the noise, and
    returns the MSE loss.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        time_dim: int = 128,
        diffusion_steps: int = 1000,
    ):
        super().__init__()
        self.unet = SmallUNet(in_channels, base_channels, channel_mults, time_dim)
        self.schedule = DiffusionSchedule(diffusion_steps)
        self.diffusion_steps = diffusion_steps

    def forward(self, images: Tensor, **kwargs: Any) -> dict[str, Tensor]:
        """Noise-prediction training step."""
        device = images.device
        self.schedule.to(device)
        batch_size = images.shape[0]

        # Sample random timesteps
        t = torch.randint(0, self.diffusion_steps, (batch_size,), device=device)

        # Sample noise and create noisy images
        noise = torch.randn_like(images)
        x_noisy = self.schedule.q_sample(images, t, noise)

        # Predict noise
        t_emb = self.unet.time_embed(t)
        noise_pred = self.unet(x_noisy, t_emb)

        # MSE loss
        loss = F.mse_loss(noise_pred, noise)
        return {"loss": loss, "noise_mse": loss}

    def training_step(self, **batch: Any) -> dict[str, Tensor]:
        return self.forward(**batch)

    def validation_step(self, **batch: Any) -> dict[str, Tensor]:
        return self.forward(**batch)

    @classmethod
    def modality(cls) -> str:
        return "diffusion"


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

def _build_ddpm(args: Any) -> DDPMModel:
    """Build a DDPM model from Crucible args namespace."""
    in_channels = getattr(args, "image_channels", 1)
    base_channels = getattr(args, "model_dim", 64)
    diffusion_steps = getattr(args, "diffusion_steps", 1000)
    return DDPMModel(
        in_channels=in_channels,
        base_channels=base_channels,
        diffusion_steps=diffusion_steps,
    )


def register() -> None:
    """Register the DDPM model family with Crucible."""
    register_model("ddpm_unet", _build_ddpm)
