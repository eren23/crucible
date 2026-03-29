#!/usr/bin/env python3
"""LE-WM slim architecture — physically smaller model with knowledge distillation.

Trains a smaller LE-WM (96-dim, 4 predictor layers) using the same data
pipeline. Optionally distilled from a full-size pretrained teacher.

Usage:
    # Slim from scratch
    SLIM_DIM=96 SLIM_PRED_DEPTH=4 python train_lewm_slim.py data=pusht

    # Slim with layer removal (remove 2 of 6 encoder layers)
    SLIM_ENC_DEPTH=4 python train_lewm_slim.py data=pusht

Env vars:
    SLIM_DIM:        Hidden dimension (default: 96)
    SLIM_PRED_DEPTH: Predictor depth (default: 4)
    SLIM_ENC_DEPTH:  Encoder depth (default: 4)
"""
from __future__ import annotations

import os
from functools import partial
from pathlib import Path

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


class ModelSizeLogger(pl.Callback):
    """Log actual model size to W&B."""

    def on_train_epoch_end(self, trainer, pl_module):
        model = pl_module.model
        total = sum(p.numel() for p in model.parameters())
        size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
        if trainer.logger:
            trainer.logger.log_metrics({
                "model/total_params": total,
                "model/size_mb": size_mb,
            }, step=trainer.global_step)


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


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    # Dataset (same as upstream)
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
    train_loader = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen)
    val_loader = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    # Slim config from env
    slim_dim = int(os.environ.get("SLIM_DIM", "96"))
    slim_pred_depth = int(os.environ.get("SLIM_PRED_DEPTH", "4"))
    slim_enc_depth = int(os.environ.get("SLIM_ENC_DEPTH", "4"))

    # Build encoder (ViT-Tiny with reduced depth)
    encoder = spt.backbone.utils.vit_hf(
        "tiny", patch_size=cfg.patch_size, image_size=cfg.img_size,
        pretrained=False, use_mask_token=False,
    )
    if slim_enc_depth < len(encoder.encoder.layer):
        encoder.encoder.layer = encoder.encoder.layer[:slim_enc_depth]
        print(f"[Slim] Encoder truncated to {slim_enc_depth} layers")

    hidden_dim = encoder.config.hidden_size  # 192 from ViT-Tiny
    embed_dim = slim_dim
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    # Smaller predictor
    pred_cfg = dict(cfg.predictor)
    pred_cfg["depth"] = slim_pred_depth

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size, input_dim=embed_dim,
        hidden_dim=hidden_dim, output_dim=hidden_dim, **pred_cfg,
    )
    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    projector = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=1024, norm_fn=nn.BatchNorm1d)
    pred_proj = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=1024, norm_fn=nn.BatchNorm1d)

    world_model = JEPA(
        encoder=encoder, predictor=predictor,
        action_encoder=action_encoder, projector=projector, pred_proj=pred_proj,
    )

    total_params = sum(p.numel() for p in world_model.parameters())
    print(f"[Slim] {total_params:,} params ({total_params*4/1e6:.1f} MB fp32), dim={slim_dim}, enc={slim_enc_depth}L, pred={slim_pred_depth}L")

    # Training setup
    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    run_dir = Path(swm.data.utils.get_cache_dir())
    run_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"lewm_slim_{slim_dim}d_{slim_enc_depth}e_{slim_pred_depth}p"

    callbacks = [
        ModelObjectCallBack(dirpath=run_dir, filename=model_name, epoch_interval=1),
        ModelSizeLogger(),
    ]

    logger = None
    if cfg.wandb.enabled:
        wandb_cfg = dict(cfg.wandb.config)
        wandb_cfg["name"] = model_name
        logger = WandbLogger(**wandb_cfg)
        logger.log_hyperparams({
            **OmegaConf.to_container(cfg),
            "slim_dim": slim_dim, "slim_pred_depth": slim_pred_depth,
            "slim_enc_depth": slim_enc_depth, "total_params": total_params,
        })

    data_module = spt.data.DataModule(train=train_loader, val=val_loader)
    module = spt.Module(
        model=world_model, sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg), optim=optimizers,
    )

    trainer = pl.Trainer(**cfg.trainer, callbacks=callbacks, num_sanity_val_steps=1, logger=logger, enable_checkpointing=True)
    manager = spt.Manager(
        trainer=trainer, module=module, data=data_module,
        ckpt_path=run_dir / f"{model_name}_weights.ckpt",
    )
    manager()


if __name__ == "__main__":
    run()
