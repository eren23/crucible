"""Reusable LE-WM launcher bundle for Crucible external projects.

This launcher runs inside a cloned upstream ``lucas-maes/le-wm`` workspace.
It keeps the project spec thin while making the experiment entrypoint
shareable through local plugins, installed hub packages, or tap clones.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path.cwd()))


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        return int(raw)
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


def _trim_encoder_depth(encoder: Any, depth: int) -> None:
    import torch.nn as nn

    candidate_paths = (
        "encoder.layer",
        "vit.encoder.layer",
        "blocks",
        "layers",
    )

    for path in candidate_paths:
        current = encoder
        found = True
        for part in path.split("."):
            if not hasattr(current, part):
                found = False
                break
            current = getattr(current, part)
        if not found or not isinstance(current, (list, nn.ModuleList)):
            continue
        if len(current) < depth:
            raise ValueError(f"Requested encoder depth {depth}, but {path} only has {len(current)} layers")
        trimmed = type(current)(list(current)[:depth])
        if _set_nested_attr(encoder, path, trimmed):
            if hasattr(encoder, "config") and hasattr(encoder.config, "num_hidden_layers"):
                encoder.config.num_hidden_layers = depth
            return
    raise ValueError("Could not find a trimmable encoder layer stack on the upstream LE-WM encoder")


def _load_config(dotlist: list[str]) -> Any:
    from hydra import compose, initialize_config_dir

    config_dir = Path.cwd() / "config" / "train"
    if not config_dir.exists():
        raise FileNotFoundError(f"Expected upstream LE-WM config directory at {config_dir}")
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="lewm", overrides=dotlist)


def _apply_common_overrides(cfg: Any, *, mode: str) -> dict[str, Any]:
    from omegaconf import open_dict

    run_name, run_id = _resolve_run_identity(str(getattr(cfg, "output_model_name", "lewm")))
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


def run_training(cfg: Any, *, mode: str, slim: bool) -> None:
    import lightning as pl
    import stable_pretraining as spt
    import stable_worldmodel as swm
    import torch
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

    batch_size = int(cfg.loader.batch_size)
    num_workers = int(cfg.loader.num_workers)

    precomputed = os.environ.get("PRECOMPUTED_DATASET", "")
    if precomputed:
        # Fast path: load precomputed HF dataset (pre-resized, pre-normalized)
        from datasets import load_dataset as _load_hf
        _hf_ds = _load_hf(precomputed)
        _hf_ds.set_format("torch")
        train_set = _hf_ds["train"]
        val_set = _hf_ds["validation"] if "validation" in _hf_ds else _hf_ds["test"]

        # Still need column dims in cfg.wm for model construction
        with open_dict(cfg):
            for col in cfg.data.dataset.keys_to_load:
                if col.startswith("pixels"):
                    continue
                # Infer dim from the first sample
                sample = train_set[0]
                if col in sample:
                    dim = sample[col].shape[-1] if sample[col].dim() > 1 else 1
                    setattr(cfg.wm, f"{col}_dim", dim)

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

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )
    if slim and encoder_depth is not None:
        _trim_encoder_depth(encoder, encoder_depth)

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
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

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

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
            "launcher": "lewm_upstream",
            "hydra_config": OmegaConf.to_container(cfg, resolve=True),
        },
    )

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir,
        filename=cfg.output_model_name,
        epoch_interval=1,
    )
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
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
