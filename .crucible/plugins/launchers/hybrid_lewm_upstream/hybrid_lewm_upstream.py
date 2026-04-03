"""Hybrid LE-WM upstream launcher for Crucible external projects.

Fork of lewm_upstream launcher that replaces the upstream HF ViT encoder with
a HybridViTEncoder (configurable mix of softmax + linear attention blocks).

Runs inside a cloned upstream ``lucas-maes/le-wm`` workspace.  All hybrid
encoder classes are defined inline to avoid import path issues on pods.

New env vars:
    HYBRID_ENCODER_PATTERN:  Block type pattern (default: "ALALAL")
    META_TOKENS:             Number of learnable prefix meta tokens (default: 4)
    USE_WEIGHT_NORM:         Enable weight norm projection hook (default: 0)
    LINEAR_ATTN_EPS:         Denominator epsilon for linear attention (default: 1e-6)
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path.cwd()))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Env helpers (same as lewm_upstream)
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name, "")
    return raw or default


def _set_nested_attr(obj: Any, dotted_path: str, value: Any) -> bool:
    current = obj
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        if not hasattr(current, part):
            return False
        current = getattr(current, part)
    if not hasattr(current, parts[-1]):
        return False
    setattr(current, parts[-1], value)
    return True


def _resolve_run_identity(default_name: str) -> tuple[str, str]:
    run_name = _env_str("LEWM_VARIANT", _env_str("WANDB_RUN_NAME", default_name))
    run_id = _env_str("CRUCIBLE_RUN_ID", _env_str("WANDB_RUN_NAME", run_name))
    return run_name, run_id


# ---------------------------------------------------------------------------
# Inline Hybrid Encoder classes (self-contained, no cross-plugin imports)
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Convert image to patch embeddings."""

    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


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


class BidirectionalLinearAttention(nn.Module):
    """Linear attention using ELU+1 feature map, O(N*d^2)."""

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

    @staticmethod
    def _phi(x: Tensor) -> Tensor:
        return F.elu(x) + 1.0

    def forward(self, x: Tensor) -> Tensor:
        B, N, D = x.shape
        H = self.num_heads
        d = self.head_dim

        q = self.q_proj(x).reshape(B, N, H, d).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, H, d).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, H, d).permute(0, 2, 1, 3)

        q = self._phi(q)
        k = self._phi(k)

        kv = torch.einsum("bhnd,bhne->bhde", k, v)
        numerator = torch.einsum("bhnd,bhde->bhne", q, kv)
        k_sum = k.sum(dim=2)
        denominator = torch.einsum("bhnd,bhd->bhn", q, k_sum).unsqueeze(-1) + self.eps

        out = numerator / denominator
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)
        return self.out_proj(out)


class LinearViTBlock(nn.Module):
    """Pre-norm + BidirectionalLinearAttention + residual + pre-norm + MLP + residual."""

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


class _HybridEncoderConfig:
    """Minimal config shim so encoder.config.hidden_size works for ARPredictor."""

    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size


class HybridViTEncoder(nn.Module):
    """Vision Transformer encoder with configurable mix of softmax and linear attention.

    Provides a ``.config`` attribute with ``hidden_size`` for compatibility
    with the upstream ARPredictor which reads ``encoder.config.hidden_size``.
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
        # Compatibility shim for upstream ARPredictor
        self.config = _HybridEncoderConfig(hidden_size=embed_dim)

        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.meta_token = nn.Parameter(torch.zeros(1, meta_tokens, embed_dim)) if meta_tokens > 0 else None
        total_tokens = 1 + meta_tokens + num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens, embed_dim))

        pattern = block_pattern.upper()
        blocks: list[nn.Module] = []
        for i in range(depth):
            char = pattern[i % len(pattern)]
            if char == "L":
                blocks.append(LinearViTBlock(embed_dim, num_heads, eps=linear_attn_eps))
            else:
                blocks.append(ViTBlock(embed_dim, num_heads))
        self.blocks = nn.ModuleList(blocks)

        self.norm = nn.LayerNorm(embed_dim)
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

    def forward(self, x: Tensor, **kwargs: Any) -> Any:
        """Encode image. Returns HF-compatible output with .last_hidden_state.

        The upstream JEPA calls ``output.last_hidden_state[:, 0]`` to get the
        CLS token, then applies its own projector (MLP + BatchNorm).  So we
        return raw transformer hidden states — NO internal projection here.
        """
        from types import SimpleNamespace

        B = x.shape[0]
        patches = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        if self.meta_token is not None and self.meta_tokens > 0:
            meta = self.meta_token.expand(B, -1, -1)
            x = torch.cat([cls, meta, patches], dim=1)
        else:
            x = torch.cat([cls, patches], dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # Return HF-style output — upstream uses last_hidden_state[:, 0]
        return SimpleNamespace(last_hidden_state=x)


class WeightNormProjectionCallback:
    """Post-optimizer-step hook for weight norm projection.

    Projects targeted weight matrices to unit norm sphere after each step:
        W_{i,:} = W_{i,:} / ||W_{i,:}||_2 * sqrt(dim)

    Integrates as a Lightning callback.
    """

    def __init__(self, model: nn.Module):
        self._targets: list[nn.Parameter] = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.weight.ndim == 2:
                self._targets.append(module.weight)

    @torch.no_grad()
    def apply(self) -> None:
        for w in self._targets:
            dim = w.shape[1]
            norms = w.norm(dim=1, keepdim=True).clamp(min=1e-8)
            w.mul_(math.sqrt(dim) / norms)


# ---------------------------------------------------------------------------
# Config loading + overrides (same as lewm_upstream)
# ---------------------------------------------------------------------------

def _load_config(dotlist: list[str]) -> Any:
    from hydra import compose, initialize_config_dir

    config_dir = Path.cwd() / "config" / "train"
    if not config_dir.exists():
        raise FileNotFoundError(f"Expected upstream LE-WM config directory at {config_dir}")
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="lewm", overrides=dotlist)


def _apply_common_overrides(cfg: Any, *, mode: str) -> dict[str, Any]:
    from omegaconf import open_dict

    run_name, run_id = _resolve_run_identity(str(getattr(cfg, "output_model_name", "hybrid_lewm")))
    num_workers = _env_int("SLIM_NUM_WORKERS", int(getattr(cfg, "num_workers", 4)))
    batch_size = _env_int("SLIM_BATCH_SIZE", int(getattr(cfg.loader, "batch_size", 128)))
    max_epochs = _env_int("SLIM_MAX_EPOCHS", int(getattr(cfg.trainer, "max_epochs", 100)))
    sigreg_num_proj = _env_int("SLIM_SIGREG_PROJ", int(cfg.loss.sigreg.kwargs.get("num_proj", 1024)))

    with open_dict(cfg):
        cfg.output_model_name = run_name
        cfg.subdir = run_id
        cfg.num_workers = num_workers
        cfg.loader.num_workers = num_workers
        cfg.loader.batch_size = batch_size
        cfg.loader.persistent_workers = num_workers > 0
        if hasattr(cfg.loader, "prefetch_factor") and num_workers <= 0:
            cfg.loader.prefetch_factor = None
        cfg.trainer.max_epochs = max_epochs
        cfg.loss.sigreg.kwargs.num_proj = sigreg_num_proj
        if getattr(cfg, "wandb", None) and getattr(cfg.wandb, "config", None):
            cfg.wandb.config.name = run_name
            cfg.wandb.config.id = run_id
            cfg.wandb.config.entity = _env_str("WANDB_ENTITY", str(getattr(cfg.wandb.config, "entity", "") or ""))
            cfg.wandb.config.project = _env_str("WANDB_PROJECT", str(getattr(cfg.wandb.config, "project", "") or ""))
            cfg.wandb.enabled = os.environ.get("WANDB_MODE", "online") != "disabled"

    return {
        "mode": mode,
        "run_name": run_name,
        "run_id": run_id,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "num_workers": num_workers,
        "sigreg_num_proj": sigreg_num_proj,
    }


def _apply_slim_overrides(cfg: Any) -> dict[str, Any]:
    from omegaconf import open_dict

    encoder_depth = _env_int("SLIM_ENC_DEPTH", 0)
    predictor_depth = _env_int("SLIM_PRED_DEPTH", int(getattr(cfg.predictor, "depth", 6)))
    embed_dim = _env_int("SLIM_DIM", int(getattr(cfg.wm, "embed_dim", 192)))
    if encoder_depth <= 0:
        raise ValueError("SLIM_ENC_DEPTH must be set to a positive integer for slim runs")

    with open_dict(cfg):
        cfg.wm.embed_dim = embed_dim
        cfg.predictor.depth = predictor_depth

    return {
        "embed_dim": embed_dim,
        "encoder_depth": encoder_depth,
        "predictor_depth": predictor_depth,
    }


def _write_launch_metadata(run_dir: Path, payload: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "crucible_launch_metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main training entry
# ---------------------------------------------------------------------------

def run_training(cfg: Any, *, mode: str, slim: bool) -> None:
    import lightning as pl
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from lightning.pytorch.loggers import WandbLogger
    from omegaconf import OmegaConf, open_dict

    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP, SIGReg
    from utils import ModelObjectCallBack, get_column_normalizer, get_img_preprocessor

    common = _apply_common_overrides(cfg, mode=mode)
    slim_details: dict[str, Any] = {}
    encoder_depth = None
    if slim:
        slim_details = _apply_slim_overrides(cfg)
        encoder_depth = slim_details["encoder_depth"]

    # --- Data setup ---
    batch_size = int(cfg.loader.batch_size)
    num_workers = int(cfg.loader.num_workers)

    precomputed = os.environ.get("PRECOMPUTED_DATASET", "")
    if precomputed:
        # Fast path: load precomputed HF dataset (pre-resized, pre-normalized)
        from datasets import load_dataset as _load_hf
        _hf_token = os.environ.get("HF_TOKEN") or None
        _hf_ds = _load_hf(precomputed, token=_hf_token)
        _hf_ds.set_format("torch")
        train_set = _hf_ds["train"]
        val_set = _hf_ds["validation"] if "validation" in _hf_ds else _hf_ds["test"]

        with open_dict(cfg):
            frameskip = int(cfg.data.dataset.get("frameskip", 1))
            for col in cfg.data.dataset.keys_to_load:
                if col.startswith("pixels"):
                    continue
                sample = train_set[0]
                if col in sample:
                    # Precomputed tensors already include frameskip expansion,
                    # but cfg.wm.*_dim must match get_dim() semantics (raw dim
                    # before frameskip), because effective_act_dim = frameskip * action_dim.
                    raw_dim = sample[col].shape[-1] if sample[col].dim() > 1 else 1
                    if col == "action" and frameskip > 1:
                        raw_dim = raw_dim // frameskip
                    setattr(cfg.wm, f"{col}_dim", raw_dim)

        train = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=num_workers, pin_memory=True,
            prefetch_factor=4 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )
        val = torch.utils.data.DataLoader(
            val_set, batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=num_workers, pin_memory=True,
            prefetch_factor=4 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )
    else:
        # Original HDF5 path
        dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
        transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]

        with open_dict(cfg):
            for col in cfg.data.dataset.keys_to_load:
                if col.startswith("pixels"):
                    continue
                normalizer = get_column_normalizer(dataset, col, col)
                transforms.append(normalizer)
                setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

        transform = spt.data.transforms.Compose(*transforms)
        dataset.transform = transform

        rnd_gen = torch.Generator().manual_seed(cfg.seed)
        train_set, val_set = spt.data.random_split(
            dataset,
            lengths=[cfg.train_split, 1 - cfg.train_split],
            generator=rnd_gen,
        )

        train = torch.utils.data.DataLoader(
            train_set, **cfg.loader, shuffle=True, drop_last=True,
            generator=rnd_gen,
        )
        val = torch.utils.data.DataLoader(
            val_set, **cfg.loader, shuffle=False, drop_last=False,
        )

    # --- Build Hybrid Encoder (replaces upstream HF ViT) ---
    block_pattern = _env_str("HYBRID_ENCODER_PATTERN", "ALALAL")
    meta_tokens = _env_int("META_TOKENS", 4)
    use_weight_norm = _env_int("USE_WEIGHT_NORM", 0)
    linear_attn_eps = _env_float("LINEAR_ATTN_EPS", 1e-6)

    embed_dim = cfg.wm.get("embed_dim", 192)
    enc_depth = encoder_depth if encoder_depth is not None else 6
    enc_heads = _env_int("SLIM_ENC_HEADS", 3)

    encoder = HybridViTEncoder(
        image_size=cfg.img_size,
        patch_size=cfg.patch_size,
        in_channels=3,
        embed_dim=embed_dim,
        depth=enc_depth,
        num_heads=enc_heads,
        block_pattern=block_pattern,
        meta_tokens=meta_tokens,
        linear_attn_eps=linear_attn_eps,
    )

    hidden_dim = encoder.config.hidden_size  # == embed_dim
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )
    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )

    # --- Weight norm hook (optional) ---
    weight_norm_hook = None
    if use_weight_norm:
        weight_norm_hook = WeightNormProjectionCallback(encoder)

    # --- Optimizer ---
    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    # --- Forward function ---
    def _forward(self, batch, stage, cfg_local):
        ctx_len = cfg_local.wm.history_size
        n_preds = cfg_local.wm.num_preds
        lambd = cfg_local.loss.sigreg.weight

        batch["action"] = torch.nan_to_num(batch["action"], 0.0)
        output = self.model.encode(batch)
        emb = output["emb"]
        act_emb = output["act_emb"]
        ctx_emb = emb[:, :ctx_len]
        ctx_act = act_emb[:, :ctx_len]
        tgt_emb = emb[:, n_preds:]
        pred_emb = self.model.predict(ctx_emb, ctx_act)
        output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
        output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
        output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]
        losses = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
        is_val = stage == "validate"
        self.log_dict(
            losses,
            on_step=not is_val,
            on_epoch=is_val,
            sync_dist=True,
        )
        return output

    data_module = spt.data.DataModule(train=train, val=val)
    module = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=lambda self, batch, stage: _forward(self, batch, stage, cfg),
        optim=optimizers,
    )

    run_dir = Path(swm.data.utils.get_cache_dir(), cfg.subdir)
    _write_launch_metadata(
        run_dir,
        {
            **common,
            **slim_details,
            "trainer_max_epochs": cfg.trainer.max_epochs,
            "loader_batch_size": cfg.loader.batch_size,
            "launcher": "hybrid_lewm_upstream",
            "block_pattern": block_pattern,
            "meta_tokens": meta_tokens,
            "use_weight_norm": use_weight_norm,
            "linear_attn_eps": linear_attn_eps,
            "hydra_config": OmegaConf.to_container(cfg, resolve=True),
        },
    )

    logger = None
    wandb_log_model = _env_int("WANDB_LOG_MODEL", 1)
    if cfg.wandb.enabled and os.environ.get("WANDB_MODE", "online") != "disabled":
        wandb_cfg = dict(cfg.wandb.config)
        if wandb_log_model:
            wandb_cfg["log_model"] = "all"
        logger = WandbLogger(**wandb_cfg)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir,
        filename=cfg.output_model_name,
        epoch_interval=1,
    )

    # --- Checkpoint callback for W&B model logging ---
    from lightning.pytorch.callbacks import ModelCheckpoint
    ckpt_callback = ModelCheckpoint(
        dirpath=run_dir,
        filename="{epoch:02d}-{step}",
        every_n_epochs=1,
        save_top_k=-1,
    )

    # --- Weight norm as Lightning callback ---
    callbacks = [object_dump_callback, ckpt_callback]
    if weight_norm_hook is not None:

        class _WeightNormLightningCallback(pl.Callback):
            def __init__(self, hook: WeightNormProjectionCallback):
                super().__init__()
                self._hook = hook

            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                self._hook.apply()

        callbacks.append(_WeightNormLightningCallback(weight_norm_hook))

    with open_dict(cfg):
        cfg.trainer.accelerator = "gpu"
        cfg.trainer.devices = 1

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )
    manager()


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    mode = _env_str("LEWM_MODE", "slim")
    cfg = _load_config(args)
    run_training(cfg, mode=mode, slim=(mode == "slim"))


if __name__ == "__main__":
    main()
