"""Experiment presets: built-in defaults merged with crucible.yaml overrides.

Presets define baseline environment-variable configs for different run
scales.  The built-in defaults can be extended or overridden by adding a
``presets:`` section to crucible.yaml.

Usage::

    # Get a preset (merges yaml overrides on top of built-in defaults)
    preset = get_preset("proxy")

    # List all available presets
    names = list_presets()

    # Get preset with a ProjectConfig already loaded
    preset = get_preset("proxy", project_config=cfg)
"""
from __future__ import annotations

from crucible.core.config import ProjectConfig, load_config

# ---------------------------------------------------------------------------
# Built-in preset defaults (framework-agnostic training parameters)
# ---------------------------------------------------------------------------

PRESET_DEFAULTS: dict[str, dict[str, str]] = {
    "smoke": {
        "MAX_WALLCLOCK_SECONDS": "60",
        "ITERATIONS": "400",
        "TRAIN_BATCH_TOKENS": "8192",
        "GRAD_ACCUM_STEPS": "1",
        "VAL_LOSS_EVERY": "0",
        "VAL_BATCH_SIZE": "8192",
        "TRAIN_LOG_EVERY": "5",
        "WARMUP_STEPS": "5",
        "TRAIN_SHARD_LIMIT": "1",
    },
    "screen": {
        "MAX_WALLCLOCK_SECONDS": "3600",
        "ITERATIONS": "2000",
        "TRAIN_BATCH_TOKENS": "65536",
        "GRAD_ACCUM_STEPS": "1",
        "VAL_LOSS_EVERY": "500",
        "VAL_BATCH_SIZE": "8192",
        "TRAIN_LOG_EVERY": "20",
        "WARMUP_STEPS": "10",
        "TRAIN_SHARD_LIMIT": "8",
    },
    "proxy": {
        "MAX_WALLCLOCK_SECONDS": "1800",
        "ITERATIONS": "6000",
        "TRAIN_BATCH_TOKENS": "65536",
        "GRAD_ACCUM_STEPS": "1",
        "VAL_LOSS_EVERY": "0",
        "VAL_BATCH_SIZE": "8192",
        "TRAIN_LOG_EVERY": "50",
        "WARMUP_STEPS": "10",
        "TRAIN_SHARD_LIMIT": "8",
    },
    "medium": {
        "MAX_WALLCLOCK_SECONDS": "3600",
        "ITERATIONS": "15000",
        "TRAIN_BATCH_TOKENS": "65536",
        "GRAD_ACCUM_STEPS": "1",
        "VAL_LOSS_EVERY": "0",
        "VAL_BATCH_SIZE": "8192",
        "TRAIN_LOG_EVERY": "100",
        "WARMUP_STEPS": "10",
        "TRAIN_SHARD_LIMIT": "24",
    },
    "promotion": {
        "MAX_WALLCLOCK_SECONDS": "7200",
        "ITERATIONS": "100000",
        "TRAIN_BATCH_TOKENS": "65536",
        "GRAD_ACCUM_STEPS": "1",
        "VAL_LOSS_EVERY": "10000",
        "VAL_BATCH_SIZE": "8192",
        "TRAIN_LOG_EVERY": "100",
        "WARMUP_STEPS": "10",
        "TRAIN_SHARD_LIMIT": "0",
    },
    "overnight": {
        "MAX_WALLCLOCK_SECONDS": "3600",
        "ITERATIONS": "200000",
        "TRAIN_BATCH_TOKENS": "65536",
        "GRAD_ACCUM_STEPS": "1",
        "VAL_LOSS_EVERY": "10000",
        "VAL_BATCH_SIZE": "8192",
        "TRAIN_LOG_EVERY": "100",
        "WARMUP_STEPS": "20",
        "TRAIN_SHARD_LIMIT": "0",
    },
}

# CLI timeout hints per backend per preset (seconds).
# Used as sensible defaults when the caller does not specify a timeout.
CLI_TIMEOUT_DEFAULTS: dict[str, dict[str, int]] = {
    "mlx": {
        "smoke": 180,
        "screen": 900,
        "proxy": 3600,
        "medium": 5400,
        "promotion": 10800,
        "overnight": 7200,
    },
    "torch": {
        "smoke": 900,
        "screen": 600,
        "proxy": 3600,
        "medium": 5400,
        "promotion": 10800,
        "overnight": 5400,
    },
}


def get_preset(
    name: str,
    *,
    project_config: ProjectConfig | None = None,
) -> dict[str, str]:
    """Return a preset config dict, merging built-in defaults with yaml overrides.

    Lookup order:
      1. Built-in PRESET_DEFAULTS
      2. ``presets.<name>`` from crucible.yaml (if present)
      3. The yaml values override the built-in values key-by-key

    Raises ValueError if the preset name is unknown in both sources.
    """
    cfg = project_config
    yaml_presets: dict[str, dict[str, str]] = {}
    if cfg is not None:
        yaml_presets = cfg.presets or {}
    else:
        # Try loading config only if not provided
        try:
            cfg = load_config()
            yaml_presets = cfg.presets or {}
        except Exception:
            yaml_presets = {}

    builtin = PRESET_DEFAULTS.get(name)
    yaml_override = yaml_presets.get(name)

    if builtin is None and yaml_override is None:
        available = sorted(set(PRESET_DEFAULTS) | set(yaml_presets))
        raise ValueError(
            f"Unknown preset {name!r}. Available: {', '.join(available)}"
        )

    # Start from built-in, overlay yaml overrides
    result: dict[str, str] = {}
    if builtin is not None:
        result.update(builtin)
    if yaml_override is not None:
        result.update({k: str(v) for k, v in yaml_override.items()})

    return result


def list_presets(
    *,
    project_config: ProjectConfig | None = None,
) -> list[str]:
    """Return sorted list of all available preset names."""
    yaml_presets: dict[str, dict[str, str]] = {}
    try:
        cfg = project_config or load_config()
        yaml_presets = cfg.presets or {}
    except Exception:
        pass
    return sorted(set(PRESET_DEFAULTS) | set(yaml_presets))


def get_timeout_hint(backend: str, preset: str) -> int:
    """Return a sensible CLI timeout for backend+preset, or 3600 as fallback."""
    backend_defaults = CLI_TIMEOUT_DEFAULTS.get(backend, {})
    return backend_defaults.get(preset, 3600)
