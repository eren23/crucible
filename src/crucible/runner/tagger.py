"""Auto-tagging for recipes and experiment designs.

Borrows ml-intern's `namespace:value` tag scheme but adapts namespaces to
Crucible's domain. Tagging is a pure function over the saved artifact dict;
it does not require any I/O. Output: sorted, deduplicated list of strings
like ``preset:smoke``, ``architecture:baseline``, ``outcome:success``.

Auto-tags are merged with user-supplied tags at save time. The two coexist
in the same ``tags`` list, distinguished only by the ``:`` separator —
user tags conventionally don't include namespace prefixes.

Tagging is best-effort: missing or unexpected fields are silently skipped.
Unknown values still produce a tag (``preset:unknown_thing``) so future
namespaces remain discoverable in ``recipe_list`` queries.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

# Recognized presets — keep aligned with crucible.core.config preset list.
_KNOWN_PRESETS = frozenset({"smoke", "screen", "proxy", "medium", "promotion", "overnight"})

# Coarse modality signals: substrings that strongly imply a modality when
# present in a config key or value. Order doesn't matter; the first hit wins
# per modality (each artifact gets at most one modality tag).
_MODALITY_SIGNALS: list[tuple[str, tuple[str, ...]]] = [
    ("diffusion", ("DIFFUSION", "DDPM", "DDIM", "NOISE_SCHED", "TIMESTEPS")),
    ("world_model", ("WORLD_MODEL", "JEPA", "BOUNCING_BALL", "FRAME_HORIZON")),
    ("vision", ("IMAGE_SIZE", "PATCH_SIZE", "RESNET", "VIT")),
    ("rl", ("ENV_NAME", "POLICY", "ACTOR_CRITIC", "PPO", "GAE")),
    # MODEL_FAMILY intentionally NOT here: it appears in LM, diffusion,
    # world-model, and vision configs alike — using it as an LM signal would
    # mistag a diffusion config that happens to set MODEL_FAMILY=ddpm.
    # Both VOCAB_SIZE and VOCABULARY_SIZE are accepted (different repo
    # conventions); SEQUENCE_LENGTH is an alias for SEQ_LEN.
    ("lm", ("VOCAB_SIZE", "VOCABULARY_SIZE", "TOKENIZER", "SEQ_LEN", "SEQUENCE_LENGTH", "BPE")),
]

# val_bpb buckets — coarse so the tag space stays small.
def _val_bpb_bucket(value: float) -> str:
    if value < 1.0:
        return "<1.0"
    if value < 1.5:
        return "1.0-1.5"
    if value < 2.0:
        return "1.5-2.0"
    if value < 3.0:
        return "2.0-3.0"
    return "3.0+"


def _norm(s: Any) -> str:
    """Stringify and sanitize a value for use in a tag suffix."""
    return str(s).strip().lower().replace(" ", "_") or "unknown"


def _emit(out: set[str], namespace: str, value: Any) -> None:
    if value is None:
        return
    v = _norm(value)
    if not v:
        return
    out.add(f"{namespace}:{v}")


def _flatten_dict_keys(d: dict[str, Any]) -> Iterable[str]:
    """Yield uppercase versions of all keys, recursive over dict values."""
    for k, v in d.items():
        yield str(k).upper()
        if isinstance(v, dict):
            yield from _flatten_dict_keys(v)


def _detect_modality(config_blob: dict[str, Any]) -> str | None:
    """Return first matching modality, or None."""
    keys = set(_flatten_dict_keys(config_blob))
    for modality, signals in _MODALITY_SIGNALS:
        if any(sig in k for k in keys for sig in signals):
            return modality
    return None


def tag_recipe(recipe: dict[str, Any]) -> list[str]:
    """Return auto-derived tags for a recipe dict.

    Inspected fields: ``environment`` (preset, gpu), ``steps`` (tool names),
    ``results.status`` (outcome), ``project_spec`` (family).
    """
    if not isinstance(recipe, dict):
        return []
    out: set[str] = set()

    env = recipe.get("environment") or {}
    if isinstance(env, dict):
        # preset signal
        preset = env.get("PRESET") or env.get("preset")
        if preset:
            p = _norm(preset)
            _emit(out, "preset", p)
        # GPU signal
        gpu = env.get("GPU_TYPE") or env.get("gpu_type") or env.get("GPU")
        if gpu:
            _emit(out, "gpu", gpu)
        # W&B project family
        family = env.get("WANDB_PROJECT") or env.get("FAMILY")
        if family:
            _emit(out, "family", family)

    # outcome from results
    results = recipe.get("results") or {}
    if isinstance(results, dict):
        status = results.get("status") or results.get("outcome")
        if status:
            _emit(out, "outcome", status)
        primary = results.get("val_bpb") or results.get("primary_metric_value")
        if isinstance(primary, (int, float)):
            out.add(f"val_bpb_bucket:{_val_bpb_bucket(float(primary))}")

    # project_spec → family fallback (recipe may name a project file).
    # Accept either a logical name or a path; strip the YAML/YML extension and
    # any leading directory components so 'subdir/play.yaml' → 'play'.
    spec = recipe.get("project_spec")
    if spec and "family:" not in " ".join(out):
        stem = Path(str(spec)).name
        stem = stem.removesuffix(".yaml").removesuffix(".yml")
        _emit(out, "family", stem)

    # Steps → tool namespace. Skip if too many to keep tag list bounded.
    steps = recipe.get("steps") or []
    if isinstance(steps, list):
        tools = {str(s.get("tool", "")) for s in steps if isinstance(s, dict)}
        tools.discard("")
        for t in sorted(tools)[:8]:  # bound: at most 8 tool tags
            _emit(out, "tool", t)

    return sorted(out)


def tag_design(design: dict[str, Any]) -> list[str]:
    """Return auto-derived tags for an experiment_design dict.

    Inspected fields: ``base_preset``, ``backend``, ``family``, ``status``,
    ``config`` (modality detection).
    """
    if not isinstance(design, dict):
        return []
    out: set[str] = set()

    preset = design.get("base_preset")
    if preset:
        p = _norm(preset)
        if p in _KNOWN_PRESETS:
            out.add(f"preset:{p}")
        else:
            out.add(f"preset:{p}")  # unknown presets still tagged for discoverability

    backend = design.get("backend")
    if backend:
        _emit(out, "backend", backend)

    family = design.get("family")
    if family:
        _emit(out, "architecture", family)

    status = design.get("status")
    if status:
        _emit(out, "status", status)

    config_blob = design.get("config") or {}
    if isinstance(config_blob, dict):
        modality = _detect_modality(config_blob)
        if modality:
            _emit(out, "modality", modality)
        # Some configs encode the model family directly (MODEL_FAMILY).
        mf = config_blob.get("MODEL_FAMILY") or config_blob.get("model_family")
        if mf and "architecture:" not in " ".join(out):
            _emit(out, "architecture", mf)

    return sorted(out)


def merge_auto_tags(user_tags: list[str], auto_tags: list[str]) -> list[str]:
    """Combine user-supplied and auto-derived tags, dedup-preserving order:
    user tags first (in their original order), then any auto tags not already
    present. Keeps the user's ordering visible in YAML output.
    """
    seen: set[str] = set()
    out: list[str] = []
    for t in (user_tags or []):
        s = str(t)
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    for t in auto_tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out
