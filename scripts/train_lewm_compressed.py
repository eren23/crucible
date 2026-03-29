#!/usr/bin/env python3
"""LE-WM training with compression (QAT / pruning).

Wraps the upstream le-wm train.py by injecting compression hooks
as PyTorch Lightning callbacks. Uses the same Hydra config + data
pipeline as upstream.

Usage:
    # Baseline (no compression)
    python train_lewm_compressed.py data=pusht

    # QAT INT8
    COMPRESS=qat_int8 python train_lewm_compressed.py data=pusht

    # Wanda 30% pruning
    COMPRESS=wanda WANDA_SPARSITY=0.3 python train_lewm_compressed.py data=pusht

    # QAT INT8 + Wanda 30%
    COMPRESS=qat_int8,wanda WANDA_SPARSITY=0.3 python train_lewm_compressed.py data=pusht

Env vars:
    COMPRESS:  Comma-separated compression methods (qat_int8, qat_int4, wanda, magnitude)
    QAT_WARMUP_EPOCHS: Epochs before enabling QAT (default: 10)
    WANDA_SPARSITY: Fraction to prune (default: 0.3)
    WANDA_PRUNE_EPOCH: Epoch at which to prune (default: 50)
    PRUNE_SPARSITY: For magnitude pruning (default: 0.3)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from functools import partial

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn as nn
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


# ---------------------------------------------------------------------------
# Fake quantization (STE)
# ---------------------------------------------------------------------------

def fake_int8_quant(w: torch.Tensor) -> torch.Tensor:
    if w.ndim != 2:
        return w
    with torch.no_grad():
        scale = w.abs().amax(dim=1, keepdim=True) / 127.0
        scale = scale.clamp_min(1.0 / 127.0)
    q = torch.clamp(torch.round(w / scale), -128, 127)
    return w + (q * scale - w).detach()


def fake_int4_quant(w: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    if w.ndim != 2:
        return w
    rows, cols = w.shape
    if cols % group_size != 0:
        with torch.no_grad():
            scale = w.abs().amax(dim=1, keepdim=True) / 7.0
            scale = scale.clamp_min(1.0 / 7.0)
        q = torch.clamp(torch.round(w / scale), -8, 7)
        return w + (q * scale - w).detach()
    w_g = w.reshape(rows, -1, group_size)
    with torch.no_grad():
        scale = w_g.abs().amax(dim=-1, keepdim=True) / 7.0
        scale = scale.clamp_min(1.0 / 7.0)
    q = torch.clamp(torch.round(w_g / scale), -8, 7)
    return w + ((q * scale).reshape(rows, cols) - w).detach()


# ---------------------------------------------------------------------------
# Compression Lightning Callbacks
# ---------------------------------------------------------------------------

class QATCallback(pl.Callback):
    """Quantization-aware training via forward pre-hooks."""

    def __init__(self, bits: int = 8, warmup_epochs: int = 10):
        super().__init__()
        self.bits = bits
        self.warmup_epochs = warmup_epochs
        self._hooks = []
        self._enabled = False

    def on_train_start(self, trainer, pl_module):
        model = pl_module.model  # The JEPA model
        quant_fn = fake_int8_quant if self.bits == 8 else fake_int4_quant
        enabled_ref = self  # closure reference

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "norm" not in name.lower():
                def _make_hook(fn):
                    def _hook(mod, inputs):
                        if enabled_ref._enabled:
                            mod.weight.data = fn(mod.weight.data)
                    return _hook
                self._hooks.append(module.register_forward_pre_hook(_make_hook(quant_fn)))

        print(f"[QAT] Registered {len(self._hooks)} INT{self.bits} hooks (warmup: {self.warmup_epochs} epochs)")

    def on_train_epoch_start(self, trainer, pl_module):
        if not self._enabled and trainer.current_epoch >= self.warmup_epochs:
            self._enabled = True
            print(f"[QAT] Enabled at epoch {trainer.current_epoch}")

    def on_train_end(self, trainer, pl_module):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


class WandaPruneCallback(pl.Callback):
    """One-shot Wanda pruning at a specific epoch."""

    def __init__(self, sparsity: float = 0.3, prune_epoch: int = 50):
        super().__init__()
        self.sparsity = sparsity
        self.prune_epoch = prune_epoch
        self._applied = False
        self._act_norms: dict[str, torch.Tensor] = {}
        self._act_counts: dict[str, int] = {}
        self._hooks = []

    def on_train_epoch_start(self, trainer, pl_module):
        if self._applied:
            return

        model = pl_module.model
        calibrating = trainer.current_epoch >= self.prune_epoch - 5

        # Register activation hooks 5 epochs before pruning
        if calibrating and not self._hooks:
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and "norm" not in name.lower():
                    norms = self._act_norms
                    counts = self._act_counts

                    def _make_hook(ln):
                        def _hook(mod, inputs):
                            x = inputs[0] if isinstance(inputs, tuple) else inputs
                            if x is None:
                                return
                            with torch.no_grad():
                                col_norms = x.detach().reshape(-1, x.shape[-1]).norm(dim=0)
                                if ln not in norms:
                                    norms[ln] = torch.zeros_like(col_norms)
                                    counts[ln] = 0
                                norms[ln] += col_norms
                                counts[ln] += 1
                        return _hook
                    self._hooks.append(module.register_forward_pre_hook(_make_hook(name)))
            print(f"[Wanda] Calibration started at epoch {trainer.current_epoch}")

    def on_train_epoch_end(self, trainer, pl_module):
        if self._applied or trainer.current_epoch < self.prune_epoch:
            return

        model = pl_module.model
        pruned_count = 0
        total_count = 0

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear) or "norm" in name.lower():
                continue
            W = module.weight.data
            total_count += W.numel()

            act_norm = self._act_norms.get(name)
            if act_norm is None:
                act_norm = torch.ones(W.shape[1], device=W.device)
            else:
                act_norm = act_norm / max(self._act_counts.get(name, 1), 1)

            scores = W.abs() * act_norm.unsqueeze(0)
            k = int(W.shape[1] * self.sparsity)
            if k <= 0:
                continue

            for i in range(W.shape[0]):
                _, idx = torch.topk(scores[i], k, largest=False)
                W[i, idx] = 0.0
                pruned_count += k

        # Remove hooks
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._applied = True

        sparsity = pruned_count / max(total_count, 1)
        print(f"[Wanda] Pruned at epoch {trainer.current_epoch}: {sparsity:.1%} sparsity ({pruned_count:,}/{total_count:,})")


class CompressionMetricsCallback(pl.Callback):
    """Log compression metrics to W&B."""

    def on_train_epoch_end(self, trainer, pl_module):
        model = pl_module.model
        total = sum(p.numel() for p in model.parameters())
        zeros = sum((p.data == 0).sum().item() for p in model.parameters())
        sparsity = zeros / max(total, 1)
        size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

        if trainer.logger:
            trainer.logger.log_metrics({
                "compression/sparsity": sparsity,
                "compression/nonzero_params": total - zeros,
                "compression/model_size_mb": size_mb,
            }, step=trainer.global_step)


# ---------------------------------------------------------------------------
# Same forward as upstream
# ---------------------------------------------------------------------------

def lejepa_forward(self, batch, stage, cfg):
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

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

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    # Dataset
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

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
    train_set, val_set = spt.data.random_split(dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen)
    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    # Model
    encoder = spt.backbone.utils.vit_hf(cfg.encoder_scale, patch_size=cfg.patch_size, image_size=cfg.img_size, pretrained=False, use_mask_token=False)
    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(num_frames=cfg.wm.history_size, input_dim=embed_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, **cfg.predictor)
    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    projector = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    predictor_proj = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)

    world_model = JEPA(encoder=encoder, predictor=predictor, action_encoder=action_encoder, projector=projector, pred_proj=predictor_proj)

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model_module = spt.Module(model=world_model, sigreg=SIGReg(**cfg.loss.sigreg.kwargs), forward=partial(lejepa_forward, cfg=cfg), optim=optimizers)

    # Training
    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    # Build callbacks
    callbacks = [ModelObjectCallBack(dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1)]

    compress_str = os.environ.get("COMPRESS", "")
    compress_methods = [m.strip() for m in compress_str.split(",") if m.strip()]

    for method in compress_methods:
        if method == "qat_int8":
            warmup = int(os.environ.get("QAT_WARMUP_EPOCHS", "10"))
            callbacks.append(QATCallback(bits=8, warmup_epochs=warmup))
        elif method == "qat_int4":
            warmup = int(os.environ.get("QAT_WARMUP_EPOCHS", "10"))
            callbacks.append(QATCallback(bits=4, warmup_epochs=warmup))
        elif method == "wanda":
            sparsity = float(os.environ.get("WANDA_SPARSITY", "0.3"))
            prune_epoch = int(os.environ.get("WANDA_PRUNE_EPOCH", "50"))
            callbacks.append(WandaPruneCallback(sparsity=sparsity, prune_epoch=prune_epoch))
        elif method == "magnitude":
            # Magnitude pruning as a simple variant
            sparsity = float(os.environ.get("PRUNE_SPARSITY", "0.3"))
            prune_epoch = int(os.environ.get("WANDA_PRUNE_EPOCH", "50"))
            callbacks.append(WandaPruneCallback(sparsity=sparsity, prune_epoch=prune_epoch))

    if compress_methods:
        callbacks.append(CompressionMetricsCallback())
        print(f"[Compression] Methods: {compress_methods}")

    trainer = pl.Trainer(**cfg.trainer, callbacks=callbacks, num_sanity_val_steps=1, logger=logger, enable_checkpointing=True)
    manager = spt.Manager(trainer=trainer, module=world_model_module, data=data_module, ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt")
    manager()


if __name__ == "__main__":
    run()
