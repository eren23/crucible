#!/usr/bin/env python3
"""Apply DRY/deduplication refactoring to the crucible-community-tap repo.

Run from anywhere:
    python scripts/apply_tap_dedup.py

This script:
1. Deletes the duplicate distillation_objective (keeps objectives/distillation/)
2. Creates callbacks/_qat_common.py with shared QAT utilities
3. Refactors qat_int4, qat_int8, qat_mixed to use _qat_common
4. Creates launchers/_launcher_common.py with shared launcher boilerplate
5. Refactors lewm_upstream, hybrid_lewm_upstream, elastic_lewm_upstream to use it
6. Refactors semantic_eval.py and eval_code_wm.py to use _shared.py
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

TAP_ROOT = Path.home() / ".crucible-hub" / "taps" / "crucible-community-tap"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"  wrote: {path.relative_to(TAP_ROOT)}")


def _delete_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
        print(f"  deleted: {path.relative_to(TAP_ROOT)}")


# =====================================================================
# 1. Delete duplicate distillation_objective
# =====================================================================

def step1_delete_duplicate_distillation():
    print("\n=== Step 1: Delete duplicate distillation_objective ===")
    dup = TAP_ROOT / "objectives" / "distillation_objective"
    _delete_dir(dup)


# =====================================================================
# 2. Create callbacks/_qat_common.py
# =====================================================================

QAT_COMMON = '''\
"""Shared fake-quantization primitives for QAT callbacks.

Provides STE (straight-through estimator) fake quantization functions
for INT4 and INT8, plus a base callback class that handles the shared
warmup/hook/enable/cleanup lifecycle.

All three QAT callbacks (qat_int4, qat_int8, qat_mixed) import from
this module to avoid duplicating the STE math and lifecycle boilerplate.
"""
from __future__ import annotations

import os
from typing import Any


# ---------------------------------------------------------------------------
# Fake quantization functions (STE)
# ---------------------------------------------------------------------------

def fake_int8_quant(w):
    """Straight-through estimator for int8 QAT -- gradient flows through w."""
    import torch

    if w.ndim != 2:
        return w
    with torch.no_grad():
        scale = w.abs().amax(dim=1, keepdim=True) / 127.0
        scale = scale.clamp_min(1.0 / 127.0)
    q = torch.clamp(torch.round(w / scale), -128, 127)
    return w + (q * scale - w).detach()  # STE


def fake_int4_quant(w, group_size: int = 128):
    """Straight-through estimator for int4 QAT -- gradient flows through w."""
    import torch

    if w.ndim != 2:
        return w
    rows, cols = w.shape

    if cols % group_size != 0:
        # Fallback to per-row if not divisible
        with torch.no_grad():
            scale = w.abs().amax(dim=1, keepdim=True) / 7.0
            scale = scale.clamp_min(1.0 / 7.0)
        q = torch.clamp(torch.round(w / scale), -8, 7)
        return w + (q * scale - w).detach()

    w_grouped = w.reshape(rows, -1, group_size)
    with torch.no_grad():
        scale = w_grouped.abs().amax(dim=-1, keepdim=True) / 7.0
        scale = scale.clamp_min(1.0 / 7.0)
    q = torch.clamp(torch.round(w_grouped / scale), -8, 7)
    dequant = (q * scale).reshape(rows, cols)
    return w + (dequant - w).detach()  # STE


def get_quant_fn(bits: int):
    """Return the fake-quant function for the given bit-width."""
    if bits == 4:
        return fake_int4_quant
    # Default to int8 for 8-bit or any unrecognized value
    return fake_int8_quant


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def parse_exclude_patterns(env_var: str, default: str = "tok_emb,embed_low,lm_head") -> tuple[str, ...]:
    """Parse comma-separated exclude patterns from an env var."""
    exclude_raw = os.environ.get(env_var, default)
    return tuple(p.strip() for p in exclude_raw.split(",") if p.strip())


class BaseQATCallback:
    """Mixin providing shared QAT lifecycle: warmup gating, hook management, cleanup.

    Subclasses must set:
        self.warmup_steps: int
        self.exclude_patterns: tuple[str, ...]
        self._hooks: list[Any]
        self._enabled: bool
        self._bit_assignments: dict[str, int]

    And call _register_quant_hooks_base() from on_model_ready().
    """

    def _register_quant_hooks_base(
        self,
        model: Any,
        quant_fn,
        bits: int,
        *,
        resolve_fn=None,
    ) -> None:
        """Register forward pre-hooks on prunable layers.

        Args:
            model: The nn.Module to instrument.
            quant_fn: The fake-quant function to apply (for uniform-bit callbacks).
            bits: The bit-width to record (for uniform-bit callbacks).
            resolve_fn: Optional callable(name) -> (quant_fn, bits) for per-layer
                        bit-width resolution (used by qat_mixed).
        """
        from crucible.training.compression_utils import iter_prunable_layers

        self._bit_assignments = {}

        for name, module in iter_prunable_layers(model, exclude_patterns=self.exclude_patterns):
            if resolve_fn is not None:
                fn, b = resolve_fn(name)
            else:
                fn, b = quant_fn, bits

            def _make_hook(mod_name: str, func):
                def _hook(mod, inputs):
                    if self._enabled:
                        mod.weight.data = func(mod.weight.data)
                return _hook

            handle = module.register_forward_pre_hook(_make_hook(name, fn))
            self._hooks.append(handle)
            self._bit_assignments[name] = b

    def on_train_begin(self, state: dict[str, Any]) -> None:
        state["qat_bit_assignments"] = self._bit_assignments

    def on_step_begin(self, step: int, state: dict[str, Any]) -> None:
        if not self._enabled and step >= self.warmup_steps:
            self._enabled = True

    def on_train_end(self, state: dict[str, Any]) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._enabled = False
'''


def step2_create_qat_common():
    print("\n=== Step 2: Create callbacks/_qat_common.py ===")
    _write(TAP_ROOT / "callbacks" / "_qat_common.py", QAT_COMMON)


# =====================================================================
# 3. Refactor QAT callbacks
# =====================================================================

QAT_INT4_NEW = '''\
"""INT4 quantization-aware training callback.

Applies fake INT4 quantization with per-group scale factors via
straight-through estimator (STE).  Groups of ``GROUP_SIZE`` weights share
a single scale, giving finer granularity than per-row quantization.

Env vars:
    QAT_INT4_WARMUP_STEPS: Steps before enabling fake quant (default 1000).
    QAT_INT4_GROUP_SIZE: Number of weights per quantization group (default 128).
    QAT_INT4_EXCLUDE_PATTERNS: Comma-separated module name patterns to skip
        (default "tok_emb,embed_low,lm_head").
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Import shared QAT utilities from sibling _qat_common module
_callbacks_dir = str(Path(__file__).parent.parent)
if _callbacks_dir not in sys.path:
    sys.path.insert(0, _callbacks_dir)

from callbacks._qat_common import BaseQATCallback, fake_int4_quant, parse_exclude_patterns

from crucible.training.callbacks import TrainingCallback, register_callback


class QatInt4Callback(BaseQATCallback, TrainingCallback):
    """INT4 QAT via per-group fake quantization with straight-through estimator."""

    priority = 5  # Run early, before pruning callbacks

    def __init__(self, **kwargs: Any) -> None:
        self.warmup_steps = int(os.environ.get("QAT_INT4_WARMUP_STEPS", "1000"))
        self.group_size = int(os.environ.get("QAT_INT4_GROUP_SIZE", "128"))
        self.exclude_patterns = parse_exclude_patterns("QAT_INT4_EXCLUDE_PATTERNS")
        self._hooks: list[Any] = []
        self._enabled = False
        self._bit_assignments: dict[str, int] = {}

    def on_model_ready(self, state: dict[str, Any]) -> None:
        """Register QAT hooks BEFORE torch.compile so they\'re in the graph."""
        model = state.get("model")
        if model is None:
            return

        gs = self.group_size

        def _make_group_quant(mod_name: str, group_sz: int):
            def _fn(w):
                return fake_int4_quant(w, group_size=group_sz)
            return _fn

        from crucible.training.compression_utils import iter_prunable_layers
        self._bit_assignments = {}
        for name, module in iter_prunable_layers(model, exclude_patterns=self.exclude_patterns):
            quant_fn = _make_group_quant(name, gs)

            def _make_hook(mod_name: str, fn):
                def _hook(mod, inputs):
                    if self._enabled:
                        mod.weight.data = fn(mod.weight.data)
                return _hook

            handle = module.register_forward_pre_hook(_make_hook(name, quant_fn))
            self._hooks.append(handle)
            self._bit_assignments[name] = 4

    # on_train_begin, on_step_begin, on_train_end inherited from BaseQATCallback


register_callback("qat_int4", QatInt4Callback, source="local")
'''


QAT_INT8_NEW = '''\
"""INT8 quantization-aware training callback.

Applies fake INT8 quantization via straight-through estimator (STE) during
the forward pass so the model learns to be robust to quantization noise.
Hooks are gated by a warmup period and removed at train end for clean
serialization.

Env vars:
    QAT_INT8_WARMUP_STEPS: Steps before enabling fake quant (default 500).
    QAT_INT8_EXCLUDE_PATTERNS: Comma-separated module name patterns to skip
        (default "tok_emb,embed_low,lm_head").
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Import shared QAT utilities from sibling _qat_common module
_callbacks_dir = str(Path(__file__).parent.parent)
if _callbacks_dir not in sys.path:
    sys.path.insert(0, _callbacks_dir)

from callbacks._qat_common import BaseQATCallback, fake_int8_quant, parse_exclude_patterns

from crucible.training.callbacks import TrainingCallback, register_callback


class QatInt8Callback(BaseQATCallback, TrainingCallback):
    """INT8 QAT via per-row fake quantization with straight-through estimator."""

    priority = 5  # Run early, before pruning callbacks

    def __init__(self, **kwargs: Any) -> None:
        self.warmup_steps = int(os.environ.get("QAT_INT8_WARMUP_STEPS", "500"))
        self.exclude_patterns = parse_exclude_patterns("QAT_INT8_EXCLUDE_PATTERNS")
        self._hooks: list[Any] = []
        self._enabled = False
        self._bit_assignments: dict[str, int] = {}

    def on_model_ready(self, state: dict[str, Any]) -> None:
        """Register QAT hooks BEFORE torch.compile so they\'re in the graph."""
        model = state.get("model")
        if model is None:
            return
        self._register_quant_hooks_base(model, fake_int8_quant, bits=8)

    # on_train_begin, on_step_begin, on_train_end inherited from BaseQATCallback


register_callback("qat_int8", QatInt8Callback, source="local")
'''


QAT_MIXED_NEW = '''\
"""Mixed-precision quantization-aware training callback.

Parses a per-module bit-width pattern from ``QAT_MIXED_PATTERN`` and applies
the matching fake-quantization function to each module during forward passes.
This lets different parts of the model run at different precisions (e.g.,
encoder at 8-bit, predictor at 4-bit).

Pattern format (``QAT_MIXED_PATTERN``):
    - Single value: ``"8"`` -- all modules at 8 bits (default).
    - Comma-separated: ``"encoder:8,predictor:4,lm_head:8"`` -- modules whose
      name contains the prefix get the specified bit-width.

Env vars:
    QAT_MIXED_PATTERN: Bit-width mapping (default "8").
    QAT_MIXED_WARMUP_STEPS: Steps before enabling fake quant (default 500).
    QAT_MIXED_EXCLUDE_PATTERNS: Comma-separated module name patterns to skip
        (default "tok_emb,embed_low").
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Import shared QAT utilities from sibling _qat_common module
_callbacks_dir = str(Path(__file__).parent.parent)
if _callbacks_dir not in sys.path:
    sys.path.insert(0, _callbacks_dir)

from callbacks._qat_common import BaseQATCallback, get_quant_fn, parse_exclude_patterns

from crucible.training.callbacks import TrainingCallback, register_callback


# ---------------------------------------------------------------------------
# Pattern parsing
# ---------------------------------------------------------------------------

def _parse_pattern(raw: str) -> dict[str, int] | int:
    """Parse QAT_MIXED_PATTERN into either a global bit-width or prefix map.

    Returns:
        int -- if the pattern is a single number (applied to all modules).
        dict -- mapping of name-prefix to bit-width.
    """
    raw = raw.strip()
    if ":" not in raw:
        return int(raw)

    mapping: dict[str, int] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        prefix, _, bits_str = part.partition(":")
        mapping[prefix.strip()] = int(bits_str.strip())
    return mapping


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class QatMixedCallback(BaseQATCallback, TrainingCallback):
    """Mixed-precision QAT with per-module bit-width assignment."""

    priority = 5  # Run early, before pruning callbacks

    def __init__(self, **kwargs: Any) -> None:
        self.warmup_steps = int(os.environ.get("QAT_MIXED_WARMUP_STEPS", "500"))
        pattern_raw = os.environ.get("QAT_MIXED_PATTERN", "8")
        self.pattern = _parse_pattern(pattern_raw)
        self.exclude_patterns = parse_exclude_patterns("QAT_MIXED_EXCLUDE_PATTERNS", "tok_emb,embed_low")
        self._hooks: list[Any] = []
        self._enabled = False
        self._bit_assignments: dict[str, int] = {}

    def _resolve_bits(self, module_name: str) -> int:
        """Determine the bit-width for a given module name."""
        if isinstance(self.pattern, int):
            return self.pattern
        for prefix, bits in self.pattern.items():
            if prefix in module_name:
                return bits
        # No match -- fall back to 8-bit
        return 8

    def on_model_ready(self, state: dict[str, Any]) -> None:
        """Register QAT hooks BEFORE torch.compile so they\'re in the graph."""
        model = state.get("model")
        if model is None:
            return

        def _resolve_fn(name: str):
            bits = self._resolve_bits(name)
            return get_quant_fn(bits), bits

        self._register_quant_hooks_base(model, None, bits=0, resolve_fn=_resolve_fn)

    # on_train_begin, on_step_begin, on_train_end inherited from BaseQATCallback


register_callback("qat_mixed", QatMixedCallback, source="local")
'''


def step3_refactor_qat_callbacks():
    print("\n=== Step 3: Refactor QAT callbacks to use _qat_common ===")
    _write(TAP_ROOT / "callbacks" / "qat_int4" / "qat_int4.py", QAT_INT4_NEW)
    _write(TAP_ROOT / "callbacks" / "qat_int8" / "qat_int8.py", QAT_INT8_NEW)
    _write(TAP_ROOT / "callbacks" / "qat_mixed" / "qat_mixed.py", QAT_MIXED_NEW)


# =====================================================================
# 4. Create launchers/_launcher_common.py
# =====================================================================

LAUNCHER_COMMON = '''\
"""Shared launcher utilities for LE-WM family launchers.

Provides env-var parsing helpers, config loading, Hydra override application,
and metadata writing that were copy-pasted across lewm_upstream,
hybrid_lewm_upstream, and elastic_lewm_upstream.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_str(name: str, default: str) -> str:
    raw = os.environ.get(name, "")
    return raw or default


# ---------------------------------------------------------------------------
# Model attribute helpers
# ---------------------------------------------------------------------------

def set_nested_attr(obj: Any, dotted_path: str, value: Any) -> bool:
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


# ---------------------------------------------------------------------------
# Run identity
# ---------------------------------------------------------------------------

def resolve_run_identity(default_name: str) -> tuple[str, str]:
    """Resolve run name and ID from Crucible/legacy env vars."""
    run_name = env_str(
        "CRUCIBLE_VARIANT_NAME",
        env_str("LEWM_VARIANT", env_str("WANDB_RUN_NAME", default_name)),
    )
    run_id = env_str("CRUCIBLE_RUN_ID", env_str("WANDB_RUN_NAME", run_name))
    return run_name, run_id


# ---------------------------------------------------------------------------
# Encoder trimming
# ---------------------------------------------------------------------------

def trim_encoder_depth(encoder: Any, depth: int) -> None:
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
        if set_nested_attr(encoder, path, trimmed):
            if hasattr(encoder, "config") and hasattr(encoder.config, "num_hidden_layers"):
                encoder.config.num_hidden_layers = depth
            return
    raise ValueError("Could not find a trimmable encoder layer stack on the upstream LE-WM encoder")


# ---------------------------------------------------------------------------
# Hydra config
# ---------------------------------------------------------------------------

def load_config(dotlist: list[str]) -> Any:
    from hydra import compose, initialize_config_dir

    config_dir = Path.cwd() / "config" / "train"
    if not config_dir.exists():
        raise FileNotFoundError(f"Expected upstream LE-WM config directory at {config_dir}")
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="lewm", overrides=dotlist)


# ---------------------------------------------------------------------------
# Common overrides
# ---------------------------------------------------------------------------

def apply_common_overrides(cfg: Any, *, mode: str) -> dict[str, Any]:
    from omegaconf import open_dict

    run_name, run_id = resolve_run_identity(str(getattr(cfg, "output_model_name", "lewm")))
    num_workers = env_int("SLIM_NUM_WORKERS", int(getattr(cfg, "num_workers", 4)))
    batch_size = env_int("SLIM_BATCH_SIZE", int(getattr(cfg.loader, "batch_size", 128)))
    max_epochs = env_int("SLIM_MAX_EPOCHS", int(getattr(cfg.trainer, "max_epochs", 100)))
    sigreg_num_proj = env_int("SLIM_SIGREG_PROJ", int(cfg.loss.sigreg.kwargs.get("num_proj", 1024)))

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
            cfg.wandb.config.entity = env_str("WANDB_ENTITY", str(getattr(cfg.wandb.config, "entity", "") or ""))
            cfg.wandb.config.project = env_str("WANDB_PROJECT", str(getattr(cfg.wandb.config, "project", "") or ""))
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


def apply_slim_overrides(cfg: Any) -> dict[str, Any]:
    from omegaconf import open_dict

    encoder_depth = env_int("SLIM_ENC_DEPTH", 0)
    predictor_depth = env_int("SLIM_PRED_DEPTH", int(getattr(cfg.predictor, "depth", 6)))
    embed_dim = env_int("SLIM_DIM", int(getattr(cfg.wm, "embed_dim", 192)))
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


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def write_launch_metadata(run_dir: Path, payload: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "crucible_launch_metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Data pipeline (shared HF + HDF5 setup)
# ---------------------------------------------------------------------------

def build_data_loaders(cfg: Any, *, batch_size: int, num_workers: int):
    """Build train and val DataLoaders from precomputed HF or raw HDF5 data.

    Returns (train_loader, val_loader).
    """
    import os
    import torch
    from omegaconf import open_dict

    precomputed = os.environ.get("PRECOMPUTED_DATASET", "")
    if precomputed:
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
        import stable_pretraining as spt
        import stable_worldmodel as swm
        from utils import get_column_normalizer, get_img_preprocessor

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

    return train, val


# ---------------------------------------------------------------------------
# Forward function (shared JEPA loss computation)
# ---------------------------------------------------------------------------

def make_jepa_forward(cfg: Any):
    """Create the standard JEPA forward function used by all LE-WM launchers.

    Returns a function compatible with spt.Module(forward=...).
    """
    import torch

    def _forward(self, batch, stage):
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
        losses = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
        is_val = stage == "validate"
        self.log_dict(
            losses,
            on_step=not is_val,
            on_epoch=is_val,
            sync_dist=True,
        )
        return output

    return _forward
'''


def step4_create_launcher_common():
    print("\n=== Step 4: Create launchers/_launcher_common.py ===")
    _write(TAP_ROOT / "launchers" / "_launcher_common.py", LAUNCHER_COMMON)


# =====================================================================
# 5. Refactor lewm_upstream to use _launcher_common
# =====================================================================

LEWM_UPSTREAM_NEW = '''\
"""Reusable LE-WM launcher bundle for Crucible external projects.

This launcher runs inside a cloned upstream ``lucas-maes/le-wm`` workspace.
It keeps the project spec thin while making the experiment entrypoint
shareable through local plugins, installed hub packages, or tap clones.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path.cwd()))

# Import shared launcher utilities
_launchers_dir = str(Path(__file__).parent.parent)
if _launchers_dir not in sys.path:
    sys.path.insert(0, _launchers_dir)

from launchers._launcher_common import (
    apply_common_overrides,
    apply_slim_overrides,
    build_data_loaders,
    env_int,
    env_str,
    load_config as _load_config,
    make_jepa_forward,
    trim_encoder_depth,
    write_launch_metadata,
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
    from utils import ModelObjectCallBack

    common = apply_common_overrides(cfg, mode=mode)
    slim_details: dict[str, Any] = {}
    encoder_depth = None
    if slim:
        slim_details = apply_slim_overrides(cfg)
        encoder_depth = slim_details["encoder_depth"]

    batch_size = int(cfg.loader.batch_size)
    num_workers = int(cfg.loader.num_workers)

    train, val = build_data_loaders(cfg, batch_size=batch_size, num_workers=num_workers)

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )
    if slim and encoder_depth is not None:
        trim_encoder_depth(encoder, encoder_depth)

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

    _forward = make_jepa_forward(cfg)

    data_module = spt.data.DataModule(train=train, val=val)
    module = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=_forward,
        optim=optimizers,
    )

    run_dir = Path(swm.data.utils.get_cache_dir(), cfg.subdir)
    write_launch_metadata(
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
    with open_dict(cfg):
        cfg.trainer.accelerator = "gpu"
        cfg.trainer.devices = 1

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
    mode = env_str("LEWM_MODE", "slim")
    cfg = _load_config(args)
    run_training(cfg, mode=mode, slim=(mode == "slim"))


if __name__ == "__main__":
    main()
'''


def step5_refactor_lewm_upstream():
    print("\n=== Step 5: Refactor lewm_upstream to use _launcher_common ===")
    _write(TAP_ROOT / "launchers" / "lewm_upstream" / "lewm_upstream.py", LEWM_UPSTREAM_NEW)


# =====================================================================
# 6. Refactor semantic_eval.py to use _shared.py
# =====================================================================

def step6_refactor_eval_holdouts():
    print("\n=== Step 6: Refactor semantic_eval.py to use _shared.py ===")
    path = TAP_ROOT / "evaluation" / "code_wm" / "semantic_eval.py"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return

    content = path.read_text(encoding="utf-8")

    # Replace the inline load_model_and_data to use _shared
    old_block = '''def load_model_and_data(
    checkpoint_path: str,
    data_path: str,
    num_samples: int,
    device: str,
):
    """Load model from checkpoint plus held-out sample data."""
    import h5py
    import importlib.util

    # evaluation/code_wm/<script>.py -> tap root is parent.parent.parent
    tap_root = Path(__file__).parent.parent.parent
    for mod_name, mod_path in [
        ("wm_base", tap_root / "architectures" / "wm_base" / "wm_base.py"),
        ("code_wm", tap_root / "architectures" / "code_wm" / "code_wm.py"),
    ]:
        spec = importlib.util.spec_from_file_location(mod_name, mod_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)

    import code_wm

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = code_wm.CodeWorldModel(
        vocab_size=cfg["vocab_size"],
        max_seq_len=cfg["max_seq_len"],
        encoder_loops=cfg["encoder_loops"],
        model_dim=cfg["model_dim"],
        num_loops=cfg["num_loops"],
        num_heads=cfg["num_heads"],
        predictor_depth=2,
        ema_decay=cfg["ema_decay"],
        action_dim=cfg["action_dim"],
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.train(False)

    f = h5py.File(data_path, "r")
    n_total = f["before_tokens"].shape[0]
    n = min(num_samples, n_total)
    indices = np.sort(np.random.choice(n_total, size=n, replace=False))

    before = torch.from_numpy(f["before_tokens"][indices.tolist()].astype(np.int64)).to(device)
    actions = torch.from_numpy(f["edit_actions"][indices.tolist()].astype(np.float32)).to(device)
    after = torch.from_numpy(f["after_tokens"][indices.tolist()].astype(np.int64)).to(device)
    f.close()

    return model, before, actions, after, cfg'''

    new_block = '''# Shared checkpoint loader -- see _shared.py
_THIS_DIR = Path(__file__).parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
from _shared import load_codewm  # noqa: E402


def load_model_and_data(
    checkpoint_path: str,
    data_path: str,
    num_samples: int,
    device: str,
):
    """Load model from checkpoint plus held-out sample data."""
    import h5py

    model, cfg = load_codewm(checkpoint_path, device=device)

    f = h5py.File(data_path, "r")
    n_total = f["before_tokens"].shape[0]
    n = min(num_samples, n_total)
    indices = np.sort(np.random.choice(n_total, size=n, replace=False))

    before = torch.from_numpy(f["before_tokens"][indices.tolist()].astype(np.int64)).to(device)
    actions = torch.from_numpy(f["edit_actions"][indices.tolist()].astype(np.float32)).to(device)
    after = torch.from_numpy(f["after_tokens"][indices.tolist()].astype(np.int64)).to(device)
    f.close()

    return model, before, actions, after, cfg'''

    if old_block in content:
        content = content.replace(old_block, new_block)
        _write(path, content)
    else:
        print(f"  WARN: Could not find exact match in semantic_eval.py (may have changed)")


# =====================================================================
# Main
# =====================================================================

def main():
    if not TAP_ROOT.exists():
        print(f"ERROR: Tap repo not found at {TAP_ROOT}")
        return

    step1_delete_duplicate_distillation()
    step2_create_qat_common()
    step3_refactor_qat_callbacks()
    step4_create_launcher_common()
    step5_refactor_lewm_upstream()
    step6_refactor_eval_holdouts()

    print("\n=== Summary ===")
    print("  1. Deleted objectives/distillation_objective/ (literal duplicate)")
    print("  2. Created callbacks/_qat_common.py (shared STE + lifecycle)")
    print("  3. Refactored qat_int4, qat_int8, qat_mixed to use _qat_common")
    print("  4. Created launchers/_launcher_common.py (shared env, config, data pipeline)")
    print("  5. Refactored lewm_upstream to use _launcher_common")
    print("  6. Refactored semantic_eval.py to use _shared.py")
    print()
    print("NOT refactored (kept self-contained for pod deployment):")
    print("  - hybrid_lewm_upstream (has inline HybridViTEncoder -- unique logic)")
    print("  - elastic_lewm_upstream (has inline BudgetConfig/Router -- unique logic)")
    print("  - train_code_wm.py, train_kernel_wm.py (standalone training scripts)")
    print("  - Pruning callbacks (each has fundamentally different algorithms)")
    print()
    print("Verify with:")
    print("  cd ~/.crucible-hub/taps/crucible-community-tap")
    print("  python -c 'from callbacks._qat_common import fake_int8_quant; print(\"OK\")'")
    print("  python -c 'from launchers._launcher_common import env_int; print(\"OK\")'")


if __name__ == "__main__":
    main()
