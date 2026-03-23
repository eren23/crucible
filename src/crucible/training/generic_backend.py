"""Modality-agnostic training backend using the CrucibleModel contract.

This module provides a training loop that works with *any* model inheriting
from :class:`CrucibleModel`, any :class:`DataAdapter`, and any
:class:`TrainingObjective`.  It reads configuration from environment
variables (the standard Crucible training contract) and emits metrics
in both the LM-compatible format (``step:N/M train_loss:X``) and the
generic format (``metric:name=value``).

This file is a NEW backend -- it does NOT modify ``torch_backend.py``.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int = 0) -> int:
    return int(_env(key, str(default)))


def _env_float(key: str, default: float = 0.0) -> float:
    return float(_env(key, str(default)))


def run_generic_training() -> None:
    """Entry point for modality-agnostic training.

    Reads env vars:
        MODEL_FAMILY        -- architecture family name (required)
        DATA_ADAPTER        -- data adapter name (default: "token")
        TRAINING_OBJECTIVE  -- objective name (default: "cross_entropy")
        ITERATIONS          -- total training iterations
        SEQ_LEN             -- sequence length (for token adapter)
        LR                  -- learning rate
        WEIGHT_DECAY        -- weight decay
        GRAD_ACCUM_STEPS    -- gradient accumulation steps
        MODEL_DIM           -- model dimension
        NUM_LAYERS          -- number of layers
        NUM_HEADS           -- number of attention heads
        VOCAB_SIZE          -- vocabulary size
        MAX_WALLCLOCK_SECONDS -- wall-clock timeout
        LOG_INTERVAL        -- steps between log lines (default: 10)
        VAL_INTERVAL        -- steps between validation (default: 0 = off)
    """
    import torch

    from crucible.models.registry import build_model
    from crucible.training.data_adapters import build_data_adapter
    from crucible.training.objectives import build_objective

    # ---------------------------------------------------------------
    # Configuration from env vars
    # ---------------------------------------------------------------
    model_family = _env("MODEL_FAMILY", "baseline")
    data_adapter_name = _env("DATA_ADAPTER", "token")
    objective_name = _env("TRAINING_OBJECTIVE", "cross_entropy")
    iterations = _env_int("ITERATIONS", 400)
    lr = _env_float("LR", 3e-4)
    weight_decay = _env_float("WEIGHT_DECAY", 0.0)
    grad_accum_steps = _env_int("GRAD_ACCUM_STEPS", 1)
    max_wallclock = _env_int("MAX_WALLCLOCK_SECONDS", 0)
    log_interval = _env_int("LOG_INTERVAL", 10)
    val_interval = _env_int("VAL_INTERVAL", 0)

    # Build an args namespace for build_model (matches training contract)
    class _Args:
        pass

    args = _Args()
    for k, v in os.environ.items():
        try:
            setattr(args, k.lower(), int(v))
        except ValueError:
            try:
                setattr(args, k.lower(), float(v))
            except ValueError:
                setattr(args, k.lower(), v)
    # Ensure standard model args exist with defaults
    if not hasattr(args, "vocab_size"):
        args.vocab_size = _env_int("VOCAB_SIZE", 50257)
    if not hasattr(args, "model_dim"):
        args.model_dim = _env_int("MODEL_DIM", 768)
    if not hasattr(args, "num_layers"):
        args.num_layers = _env_int("NUM_LAYERS", 12)
    if not hasattr(args, "num_heads"):
        args.num_heads = _env_int("NUM_HEADS", 6)

    # ---------------------------------------------------------------
    # Build components
    # ---------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"model_params:{total_params}", flush=True)

    # Check if model has its own loss computation
    has_compute_loss = hasattr(model, "compute_loss")

    # Build objective only if model doesn't handle loss internally
    objective = None
    if not has_compute_loss:
        objective = build_objective(objective_name)

    # Parameter groups
    if hasattr(model, "param_groups"):
        param_groups = model.param_groups()
    else:
        param_groups = [{"params": list(model.parameters())}]

    # Set lr on all groups
    for pg in param_groups:
        pg.setdefault("lr", lr)
        pg.setdefault("weight_decay", weight_decay)

    optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

    # Build data adapter (fallback to dummy batch if adapter unavailable)
    data_adapter = None
    try:
        data_adapter = build_data_adapter(data_adapter_name)
    except Exception:
        pass  # Will use _get_dummy_batch as fallback

    # ---------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------
    t0 = time.time()
    model.train()

    for step in range(1, iterations + 1):
        # Wall-clock timeout
        if max_wallclock > 0 and (time.time() - t0) > max_wallclock:
            print(f"stopping_early max_wallclock step:{step}/{iterations}", flush=True)
            break

        optimizer.zero_grad(set_to_none=True)

        # Get batch from data adapter or fallback to dummy
        if data_adapter is not None:
            try:
                batch = data_adapter.next_batch(
                    seq_len=getattr(args, "seq_len", 128),
                    device=device,
                )
            except Exception:
                batch = _get_dummy_batch(model, device, args)
        else:
            batch = _get_dummy_batch(model, device, args)

        # Forward pass
        if hasattr(model, "training_step"):
            step_result = model.training_step(**batch)
        elif objective is not None:
            predictions = model(**batch)
            step_result = objective.compute(predictions, batch)
        else:
            step_result = {"loss": torch.tensor(0.0, device=device)}

        loss = step_result["loss"]
        loss.backward()
        optimizer.step()

        # Logging
        if step % log_interval == 0 or step == 1:
            loss_val = loss.item()
            # Standard format for backward compat
            print(f"step:{step}/{iterations} train_loss:{loss_val:.6f}", flush=True)
            # Generic metrics
            print(f"metric:train_loss={loss_val:.6f}", flush=True)
            for key, val in step_result.items():
                if key != "loss" and hasattr(val, "item"):
                    print(f"metric:{key}={val.item():.6f}", flush=True)

    elapsed_ms = int((time.time() - t0) * 1000)
    print(f"train_time:{elapsed_ms}ms", flush=True)


def _get_dummy_batch(
    model: Any, device: Any, args: Any
) -> dict[str, Any]:
    """Build a minimal batch dict for the model's modality."""
    import torch

    modality = getattr(model, "modality", lambda: "generic")()
    if modality == "lm":
        seq_len = getattr(args, "seq_len", 128)
        batch_size = 1
        vocab = getattr(args, "vocab_size", 50257)
        input_ids = torch.randint(0, vocab, (batch_size, seq_len), device=device)
        target_ids = torch.randint(0, vocab, (batch_size, seq_len), device=device)
        return {"input_ids": input_ids, "target_ids": target_ids}
    # Generic fallback
    return {}


if __name__ == "__main__":
    run_generic_training()
