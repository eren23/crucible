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

import math
import os
import sys
import time
from pathlib import Path
from typing import Any

# Self-bootstrap: ensure src/ is on path when invoked directly
_src = str(Path(__file__).resolve().parent.parent.parent)
if _src not in sys.path:
    sys.path.insert(0, _src)


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
        BATCH_SIZE          -- batch size for non-token modalities (default: 8)
        SEQ_LEN             -- sequence length (for token adapter)
        LR                  -- learning rate
        WEIGHT_DECAY        -- weight decay
        GRAD_ACCUM_STEPS    -- gradient accumulation steps
        MODEL_DIM           -- model dimension
        NUM_LAYERS          -- number of layers
        NUM_HEADS           -- number of attention heads
        VOCAB_SIZE          -- vocabulary size
        IMAGE_SIZE          -- image resolution for vision modalities (default: 32)
        NUM_FRAMES          -- number of frames for video modalities (default: 4)
        MAX_WALLCLOCK_SECONDS -- wall-clock timeout
        LOG_INTERVAL        -- steps between log lines (default: 10)
        VAL_INTERVAL        -- steps between validation (default: 0 = off)
        LR_SCHEDULE         -- "cosine" | "constant" (default: "cosine")
        WARMUP_STEPS        -- LR warmup steps (default: 0)
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
    batch_size = _env_int("BATCH_SIZE", 8)
    lr = _env_float("LR", 3e-4)
    weight_decay = _env_float("WEIGHT_DECAY", 0.0)
    grad_accum_steps = _env_int("GRAD_ACCUM_STEPS", 1)
    max_wallclock = _env_int("MAX_WALLCLOCK_SECONDS", 0)
    log_interval = _env_int("LOG_INTERVAL", 10)
    val_interval = _env_int("VAL_INTERVAL", 0)
    lr_schedule = _env("LR_SCHEDULE", "cosine")
    warmup_steps = _env_int("WARMUP_STEPS", 0)
    image_size = _env_int("IMAGE_SIZE", 32)
    num_frames = _env_int("NUM_FRAMES", 4)

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

    objective = None
    objective_build_error: Exception | None = None
    if objective_name:
        try:
            objective = build_objective(objective_name)
        except Exception as exc:
            objective_build_error = exc

    # Parameter groups
    if hasattr(model, "param_groups"):
        param_groups = model.param_groups()
    else:
        param_groups = [{"params": list(model.parameters())}]

    # Set lr/weight_decay on each group so the optimizer factory sees them
    # per-group (don't pass as top-level kwargs — avoids PyTorch conflicts).
    for pg in param_groups:
        pg.setdefault("lr", lr)
        pg.setdefault("weight_decay", weight_decay)

    optimizer_name = _env("OPTIMIZER", "adamw")
    from crucible.training.optimizers import build_optimizer
    optimizer = build_optimizer(optimizer_name, param_groups)

    # LR scheduler
    from crucible.training.schedulers import build_scheduler as _build_sched
    scheduler = _build_sched(lr_schedule, optimizer, total_steps=iterations, warmup_steps=warmup_steps)

    # Build data adapter eagerly so misconfiguration fails fast.
    try:
        data_adapter = build_data_adapter(data_adapter_name)
    except Exception as exc:
        raise RuntimeError(f"Failed to build DATA_ADAPTER={data_adapter_name!r}: {exc}") from exc

    # ---------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------
    t0 = time.time()
    model.train()

    last_train_loss = float("nan")
    last_val_loss = float("nan")
    last_val_metrics: dict[str, float] = {}

    for step in range(1, iterations + 1):
        # Wall-clock timeout
        if max_wallclock > 0 and (time.time() - t0) > max_wallclock:
            print(f"stopping_early max_wallclock step:{step}/{iterations}", flush=True)
            break

        optimizer.zero_grad(set_to_none=True)

        # Get batch from the configured data adapter.
        batch = _get_batch(
            data_adapter, model, device, args,
            batch_size=batch_size,
            image_size=image_size,
            num_frames=num_frames,
        )

        # Forward pass
        step_result = _resolve_step_result(
            model,
            batch,
            objective=objective,
            objective_name=objective_name,
            objective_build_error=objective_build_error,
            stage="training",
        )

        loss = step_result["loss"]
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_val = loss.item()
        last_train_loss = loss_val

        # Logging
        if step % log_interval == 0 or step == 1:
            # Standard format for backward compat
            print(f"step:{step}/{iterations} train_loss:{loss_val:.6f}", flush=True)
            # Generic metrics
            print(f"metric:train_loss={loss_val:.6f}", flush=True)
            for key, val in step_result.items():
                if key != "loss" and hasattr(val, "item"):
                    # Handle tensors with multiple elements by averaging
                    try:
                        scalar_val = val.item() if val.numel() == 1 else val.mean().item()
                        print(f"metric:{key}={scalar_val:.6f}", flush=True)
                    except Exception as exc:
                        print(f"warning: could not convert metric {key}: {exc}", file=sys.stderr, flush=True)

        # Validation
        if val_interval > 0 and (step % val_interval == 0 or step == iterations):
            val_metrics = _run_validation(
                model, objective, data_adapter, device, args,
                objective_name=objective_name,
                objective_build_error=objective_build_error,
                batch_size=batch_size,
                image_size=image_size,
                num_frames=num_frames,
            )
            val_loss = val_metrics.get("val_loss", loss_val)
            last_val_loss = val_loss
            last_val_metrics = val_metrics
            # Emit in LM-compatible format (val_bpb = val_loss when not LM)
            val_bpb = val_metrics.get("val_bpb", val_loss)
            print(
                f"step:{step}/{iterations} val_loss:{val_loss:.6f} val_bpb:{val_bpb:.6f}",
                flush=True,
            )
            for mk, mv in val_metrics.items():
                print(f"metric:{mk}={mv:.6f}", flush=True)

    # ---------------------------------------------------------------
    # Final output
    # ---------------------------------------------------------------
    elapsed_ms = int((time.time() - t0) * 1000)
    print(f"train_time:{elapsed_ms}ms", flush=True)

    # Emit final result line so output parser's primary path matches.
    # Use val_loss if available, otherwise fall back to last train_loss.
    final_loss = last_val_loss if not math.isnan(last_val_loss) else last_train_loss
    final_bpb = last_val_metrics.get("val_bpb", final_loss)
    if not math.isnan(final_loss):
        print(
            f"final_generic val_loss:{final_loss:.6f} val_bpb:{final_bpb:.6f}",
            flush=True,
        )
    # Also emit all final metrics in generic format
    if last_val_metrics:
        for mk, mv in last_val_metrics.items():
            try:
                scalar_val = mv.item() if hasattr(mv, 'item') and mv.numel() == 1 else (mv.mean().item() if hasattr(mv, 'mean') else mv)
                print(f"metric:{mk}={scalar_val:.6f}", flush=True)
            except Exception as exc:
                print(f"warning: could not convert final metric {mk}: {exc}", file=sys.stderr, flush=True)
    else:
        print(f"metric:train_loss={last_train_loss:.6f}", flush=True)


def _resolve_step_result(
    model: Any,
    batch: dict[str, Any],
    *,
    objective: Any,
    objective_name: str,
    objective_build_error: Exception | None,
    stage: str,
) -> dict[str, Any]:
    """Return a step result dict guaranteed to contain a ``loss`` entry."""
    if hasattr(model, f"{stage}_step"):
        step_fn = getattr(model, f"{stage}_step")
        step_result = step_fn(**batch)
    elif stage == "validation" and hasattr(model, "training_step"):
        step_result = model.training_step(**batch)
    else:
        step_result = model(**batch)

    if not isinstance(step_result, dict):
        raise TypeError(f"{stage}_step must return a dict, got {type(step_result).__name__}")
    if "loss" in step_result:
        return step_result
    if objective is None:
        if objective_build_error is not None:
            raise RuntimeError(
                f"{stage}_step returned no 'loss' and TRAINING_OBJECTIVE={objective_name!r} failed to build: "
                f"{objective_build_error}"
            ) from objective_build_error
        raise KeyError(
            f"{stage}_step returned no 'loss'. Either return a loss directly or configure a valid "
            f"TRAINING_OBJECTIVE to consume the model outputs."
        )
    objective_result = objective.compute(step_result, batch)
    if not isinstance(objective_result, dict) or "loss" not in objective_result:
        raise KeyError(
            f"TRAINING_OBJECTIVE={objective_name!r} did not produce a 'loss' during {stage}."
        )
    return objective_result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _run_validation(
    model: Any,
    objective: Any,
    data_adapter: Any,
    device: Any,
    args: Any,
    *,
    objective_name: str,
    objective_build_error: Exception | None,
    batch_size: int = 8,
    image_size: int = 32,
    num_frames: int = 4,
    num_val_steps: int = 5,
) -> dict[str, float]:
    """Run a short validation pass and return averaged metrics."""
    import torch

    model.eval()
    accumulated: dict[str, float] = {}
    count = 0

    with torch.no_grad():
        for _ in range(num_val_steps):
            batch = _get_batch(
                data_adapter, model, device, args,
                batch_size=batch_size,
                image_size=image_size,
                num_frames=num_frames,
            )
            step_result = _resolve_step_result(
                model,
                batch,
                objective=objective,
                objective_name=objective_name,
                objective_build_error=objective_build_error,
                stage="validation",
            )

            for key, val in step_result.items():
                if hasattr(val, "item"):
                    accumulated[key] = accumulated.get(key, 0.0) + val.item()
            count += 1

    model.train()

    # Average and rename loss -> val_loss
    result: dict[str, float] = {}
    for key, total in accumulated.items():
        avg = total / max(count, 1)
        if key == "loss":
            result["val_loss"] = avg
        else:
            result[key] = avg
    return result


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def _get_batch(
    data_adapter: Any,
    model: Any,
    device: Any,
    args: Any,
    *,
    batch_size: int = 8,
    image_size: int = 32,
    num_frames: int = 4,
) -> dict[str, Any]:
    """Get a batch from the configured data adapter."""
    if data_adapter is None:
        raise RuntimeError("No data adapter is configured for generic training.")
    try:
        return data_adapter.next_batch(
            seq_len=getattr(args, "seq_len", 128),
            batch_size=batch_size,
            image_size=image_size,
            num_frames=num_frames,
            device=device,
        )
    except Exception as exc:
        raise RuntimeError(
            f"{type(data_adapter).__name__}.next_batch() failed: {exc}"
        ) from exc


def _get_dummy_batch(
    model: Any,
    device: Any,
    args: Any,
    batch_size: int = 8,
    image_size: int = 32,
    num_frames: int = 4,
) -> dict[str, Any]:
    """Build a minimal batch dict for the model's modality."""
    import torch

    modality = getattr(model, "modality", lambda: "generic")()
    if modality == "lm":
        seq_len = getattr(args, "seq_len", 128)
        vocab = getattr(args, "vocab_size", 50257)
        input_ids = torch.randint(0, vocab, (1, seq_len), device=device)
        target_ids = torch.randint(0, vocab, (1, seq_len), device=device)
        return {"input_ids": input_ids, "target_ids": target_ids}
    if modality in ("vision", "diffusion"):
        channels = getattr(args, "image_channels", 3)
        images = torch.randn(batch_size, channels, image_size, image_size, device=device)
        return {"images": images}
    if modality == "world_model":
        channels = getattr(args, "image_channels", 3)
        frames = torch.randn(
            batch_size, num_frames, channels, image_size, image_size, device=device
        )
        actions = torch.randn(batch_size, num_frames - 1, 2, device=device)
        return {"frames": frames, "actions": actions}
    # Generic fallback
    return {}


if __name__ == "__main__":
    run_generic_training()
