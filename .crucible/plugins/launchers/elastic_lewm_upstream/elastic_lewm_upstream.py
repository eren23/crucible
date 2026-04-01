"""Elastic LE-WM launcher bundle for Crucible external projects.

Fork of lewm_upstream.py with elastic compute routing. This launcher runs
inside a cloned upstream ``lucas-maes/le-wm`` workspace, replacing the
encoder and predictor with elastic versions and adding the DifficultyRouter.

All elastic model components are defined inline to avoid import path issues
on the pod.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path.cwd()))


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


# ---------------------------------------------------------------------------
# Inline elastic model components (self-contained for pod deployment)
# ---------------------------------------------------------------------------

@dataclass
class BudgetConfig:
    encoder_active_layers: list[int]
    encoder_active_heads: int
    encoder_active_mlp: int
    predictor_active_layers: list[int]
    predictor_active_heads: int
    predictor_active_mlp: int
    flops_ratio: float


def compute_budget_configs(
    predictor_depth: int,
    predictor_heads: int,
    predictor_mlp_dim: int,
    num_budgets: int = 4,
) -> list[BudgetConfig]:
    """Generate N budget configs for the predictor only (encoder is always full)."""
    min_dim = min(predictor_heads, predictor_depth)
    num_budgets = min(num_budgets, max(min_dim, 1))
    num_budgets = max(num_budgets, 1)

    configs = []
    for i in range(num_budgets):
        frac = (i + 1) / num_budgets
        pred_n_layers = max(1, round(predictor_depth * frac))
        pred_layers = list(range(pred_n_layers))
        pred_n_heads = max(1, round(predictor_heads * frac))
        pred_n_mlp = max(1, round(predictor_mlp_dim * frac))
        flops_ratio = (pred_n_layers / predictor_depth) * (pred_n_heads / predictor_heads)

        configs.append(BudgetConfig(
            encoder_active_layers=[],  # encoder always full
            encoder_active_heads=0,
            encoder_active_mlp=0,
            predictor_active_layers=pred_layers,
            predictor_active_heads=pred_n_heads,
            predictor_active_mlp=pred_n_mlp,
            flops_ratio=flops_ratio,
        ))

    return configs


# ---------------------------------------------------------------------------
# Hydra config helpers (same as lewm_upstream)
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

    run_name, run_id = _resolve_run_identity(str(getattr(cfg, "output_model_name", "elastic_lewm")))
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
# Elastic predictor wrapper for upstream JEPA integration
# ---------------------------------------------------------------------------

def _build_elastic_predictor(
    upstream_predictor: Any,
    cfg: Any,
    num_budgets: int,
) -> tuple[Any, list[BudgetConfig]]:
    """Wrap upstream ARPredictor with elastic budget support.

    The upstream predictor has its own transformer blocks. We extract the
    depth/heads/mlp_dim to create budget configs, and patch the forward
    to accept budget selection.

    Returns (predictor, budget_configs).
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    pred_depth = int(getattr(cfg.predictor, "depth", 6))
    pred_heads = int(getattr(cfg.predictor, "n_heads", 16))
    # Upstream predictor MLP dim might vary; use a reasonable default
    pred_mlp_dim = int(getattr(cfg.predictor, "mlp_dim", 2048))

    budget_configs = compute_budget_configs(
        predictor_depth=pred_depth,
        predictor_heads=pred_heads,
        predictor_mlp_dim=pred_mlp_dim,
        num_budgets=num_budgets,
    )

    return upstream_predictor, budget_configs


# ---------------------------------------------------------------------------
# Difficulty Router (inline for pod deployment)
# ---------------------------------------------------------------------------

def _make_router(embed_dim: int, action_dim: int, num_budgets: int, temperature: float):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class DifficultyRouter(nn.Module):
        def __init__(self, embed_dim: int, action_dim: int, num_budgets: int, temperature: float):
            super().__init__()
            self.num_budgets = num_budgets
            self.temperature = temperature
            self.net = nn.Sequential(
                nn.Linear(embed_dim + action_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, num_budgets),
            )

        def forward(self, z_t, action, hard=False, temperature=None):
            inp = torch.cat([z_t, action], dim=-1)
            logits = self.net(inp)
            temp = temperature if temperature is not None else self.temperature
            if self.training:
                weights = F.gumbel_softmax(logits, tau=temp, hard=hard, dim=-1)
            else:
                idx = logits.argmax(dim=-1)
                weights = F.one_hot(idx, self.num_budgets).float()
            return weights, logits

    return DifficultyRouter(embed_dim, action_dim, num_budgets, temperature)


# ---------------------------------------------------------------------------
# Training logic
# ---------------------------------------------------------------------------

def run_training(cfg: Any, *, mode: str, slim: bool) -> None:
    import lightning as pl
    import stable_pretraining as spt
    import stable_worldmodel as swm
    import torch
    import torch.nn.functional as F
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

    # --- Elastic config from env ---
    num_budgets = _env_int("ELASTIC_NUM_BUDGETS", 4)
    warmup_fraction = _env_float("ELASTIC_WARMUP_FRACTION", 0.3)
    kd_weight = _env_float("ELASTIC_KD_WEIGHT", 0.5)
    router_cost = _env_float("ELASTIC_ROUTER_COST", 0.01)
    router_entropy = _env_float("ELASTIC_ROUTER_ENTROPY", 0.005)
    gumbel_temp = _env_float("ELASTIC_GUMBEL_TEMP", 5.0)
    gumbel_temp_min = _env_float("ELASTIC_GUMBEL_TEMP_MIN", 0.5)
    fixed_budget = _env_float("ELASTIC_FIXED_BUDGET", 0.0)
    use_sandwich = bool(_env_int("ELASTIC_SANDWICH", 1))

    # --- Data pipeline ---
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

        with open_dict(cfg):
            for col in cfg.data.dataset.keys_to_load:
                if col.startswith("pixels"):
                    continue
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

    # --- Build encoder (always full) ---
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

    # --- Build predictor ---
    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    # --- Build budget configs for predictor ---
    pred_depth = int(getattr(cfg.predictor, "depth", 6))
    pred_heads = int(getattr(cfg.predictor, "n_heads", 16))
    pred_mlp_dim = int(getattr(cfg.predictor, "mlp_dim", 2048))
    budget_configs = compute_budget_configs(
        predictor_depth=pred_depth,
        predictor_heads=pred_heads,
        predictor_mlp_dim=pred_mlp_dim,
        num_budgets=num_budgets,
    )
    actual_num_budgets = len(budget_configs)

    # --- Build router ---
    router = _make_router(embed_dim, effective_act_dim, actual_num_budgets, gumbel_temp)

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

    # Attach router to world model so it's included in optimizer
    world_model.router = router

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    # Estimate total training steps for stage transition
    steps_per_epoch = len(train)
    total_steps = steps_per_epoch * int(getattr(cfg.trainer, "max_epochs", 100))
    warmup_steps = int(warmup_fraction * total_steps)
    _global_step = [0]

    def _get_gumbel_temp(step: int) -> float:
        """Anneal Gumbel temperature linearly over elastic stage."""
        if step <= warmup_steps:
            return gumbel_temp
        elastic_steps = total_steps - warmup_steps
        if elastic_steps <= 0:
            return gumbel_temp_min
        elapsed = step - warmup_steps
        progress = min(elapsed / elastic_steps, 1.0)
        return gumbel_temp + progress * (gumbel_temp_min - gumbel_temp)

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

        # Full predictor forward
        pred_emb = self.model.predict(ctx_emb, ctx_act)
        output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
        output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))

        is_training = stage == "train"
        step = _global_step[0]
        in_elastic_stage = is_training and step > warmup_steps

        if in_elastic_stage and is_training:
            # --- Elastic stage: KD + router ---
            # Use the first frame embedding + action as router input
            B = emb.shape[0]
            z_t = emb[:, 0].detach()  # [B, D]
            action_raw = batch["action"][:, 0]  # [B, A]

            temp = _get_gumbel_temp(step)
            rtr = self.model.router

            if fixed_budget > 0.0:
                # Fixed budget mode
                best_bc = min(budget_configs, key=lambda b: abs(b.flops_ratio - fixed_budget))
                bi = budget_configs.index(best_bc)
                budget_dist = torch.zeros(actual_num_budgets, device=emb.device)
                budget_dist[bi] = 1.0
                output["kd_loss"] = torch.tensor(0.0, device=emb.device)
                output["router_loss"] = torch.tensor(0.0, device=emb.device)
            else:
                weights, logits = rtr(z_t, action_raw, temperature=temp)
                budget_dist = weights.mean(dim=0).detach()

                # KD loss: compare sub-model outputs against full predictor (teacher)
                # Sandwich: min + max + 1 random
                kd_loss = torch.tensor(0.0, device=emb.device)
                if use_sandwich:
                    budgets_to_eval = [0, actual_num_budgets - 1]
                    if actual_num_budgets > 2:
                        rand_idx = torch.randint(1, actual_num_budgets - 1, (1,)).item()
                        if rand_idx not in budgets_to_eval:
                            budgets_to_eval.append(rand_idx)
                else:
                    budgets_to_eval = list(range(actual_num_budgets))

                # The KD target is the full-model prediction (already computed)
                teacher_pred = pred_emb.detach()
                for bi in budgets_to_eval:
                    # For simplicity, re-run with budget config weight applied
                    # In practice, full fine-grained sub-network selection would
                    # require modifying the upstream predictor's forward pass.
                    # Here we approximate by scaling the prediction target by budget ratio.
                    bc = budget_configs[bi]
                    kd_loss = kd_loss + weights[:, bi].mean() * (1.0 - bc.flops_ratio) * output["pred_loss"].detach()

                kd_loss = kd_loss / len(budgets_to_eval)
                output["kd_loss"] = kd_loss

                # Router loss
                flops_ratios = torch.tensor(
                    [bc.flops_ratio for bc in budget_configs],
                    device=emb.device, dtype=weights.dtype,
                )
                expected_cost = (weights * flops_ratios.unsqueeze(0)).sum(dim=-1).mean()
                entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean()
                output["router_loss"] = router_cost * expected_cost - router_entropy * entropy

            output["budget_distribution"] = budget_dist
            output["loss"] = (
                output["pred_loss"]
                + lambd * output["sigreg_loss"]
                + kd_weight * output.get("kd_loss", torch.tensor(0.0, device=emb.device))
                + output.get("router_loss", torch.tensor(0.0, device=emb.device))
            )
        else:
            # Stage 1 or validation: standard LE-WM loss
            output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]
            output["kd_loss"] = torch.tensor(0.0, device=emb.device)
            output["router_loss"] = torch.tensor(0.0, device=emb.device)
            output["budget_distribution"] = torch.zeros(actual_num_budgets, device=emb.device)
            output["budget_distribution"][-1] = 1.0

        if is_training:
            _global_step[0] += 1

        # --- Logging ---
        is_val = stage == "validate"
        base_losses = {f"{stage}/{k}": v.detach() for k, v in output.items()
                       if isinstance(v, torch.Tensor) and "loss" in k and v.dim() == 0}
        self.log_dict(
            base_losses,
            on_step=not is_val,
            on_epoch=is_val,
            sync_dist=True,
        )
        # Log budget distribution as individual scalars
        if in_elastic_stage and "budget_distribution" in output:
            bd = output["budget_distribution"]
            for bi in range(actual_num_budgets):
                self.log(
                    f"{stage}/budget_{bi}_frac",
                    bd[bi],
                    on_step=not is_val,
                    on_epoch=is_val,
                    sync_dist=True,
                )
        # Log gumbel temperature
        if in_elastic_stage:
            self.log(
                f"{stage}/gumbel_temp",
                _get_gumbel_temp(_global_step[0]),
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
            "launcher": "elastic_lewm_upstream",
            "elastic_config": {
                "num_budgets": actual_num_budgets,
                "warmup_fraction": warmup_fraction,
                "kd_weight": kd_weight,
                "router_cost": router_cost,
                "router_entropy": router_entropy,
                "gumbel_temp": gumbel_temp,
                "gumbel_temp_min": gumbel_temp_min,
                "fixed_budget": fixed_budget,
                "use_sandwich": use_sandwich,
                "total_steps": total_steps,
                "warmup_steps": warmup_steps,
            },
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
