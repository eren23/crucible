"""Runtime profiles for notebook export.

A runtime profile declares environment-specific concerns: pip install order,
GPU-specific extras (flash-attn on H100, not on T4), session-limit guardrails,
and the shell prelude that sets up /content/project.

Profiles are flat dicts so tests can diff them and users can override via
`--runtime colab-h100 --runtime-extra 'pip:my-package'` in the future.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RuntimeProfile:
    """Declarative notebook runtime profile."""

    name: str
    description: str
    workdir: str                              # where project is cloned on the host
    session_limit_hours: float                # soft cap; notebooks print a warning above this
    extra_pip_packages: tuple[str, ...] = ()  # runtime-specific extras layered on top of spec.install
    flash_attn: bool = False                  # install + prefer flash-attn
    colab_secrets: bool = True                # use google.colab.userdata.get() for env forwarding
    shell_prelude: tuple[str, ...] = ()       # shell lines prepended to the install cell
    default_ckpt_dir: str = "/content/ckpts"  # where training writes checkpoints
    default_data_dir: str = "/content/data"   # where datasets are cached
    gpu_label: str = "unknown"                # human label for the default GPU
    training_caveats: tuple[str, ...] = field(default_factory=tuple)  # printed in a markdown cell


COLAB_H100 = RuntimeProfile(
    name="colab-h100",
    description="Colab Pro+ with H100 runtime (80GB HBM). bf16 + flash-attn default.",
    workdir="/content/project",
    session_limit_hours=24.0,
    extra_pip_packages=(
        "accelerate>=0.30",
        "bitsandbytes>=0.43",
        "peft>=0.11",
        "trl>=0.9",
        "transformers>=4.50",
    ),
    flash_attn=True,
    gpu_label="NVIDIA H100 80GB",
    training_caveats=(
        "H100 session limit is 24h on Colab Pro+ — checkpoint frequently.",
        "Flash-attn install may fail first try; the install cell falls back to SDPA automatically.",
    ),
)


COLAB_T4 = RuntimeProfile(
    name="colab-t4",
    description="Colab free tier, T4 16GB. Small models only (<= 7B in 4-bit).",
    workdir="/content/project",
    session_limit_hours=12.0,
    extra_pip_packages=(
        "accelerate>=0.30",
        "bitsandbytes>=0.43",
        "transformers>=4.50",
    ),
    flash_attn=False,
    gpu_label="NVIDIA T4 16GB",
    training_caveats=(
        "T4 free tier disconnects after ~12h idle. Do not launch >= 14B models without 4-bit.",
        "flash-attn is NOT supported on T4; SDPA is used.",
    ),
)


COLAB_A100 = RuntimeProfile(
    name="colab-a100",
    description="Colab Pro+ with A100 runtime (40GB). bf16 + flash-attn.",
    workdir="/content/project",
    session_limit_hours=24.0,
    extra_pip_packages=COLAB_H100.extra_pip_packages,
    flash_attn=True,
    gpu_label="NVIDIA A100 40GB",
    training_caveats=(
        "A100 40GB fits up to Qwen2.5-Coder-32B in 4-bit. 80GB variant exists on some Pro+ sessions.",
    ),
)


LOCAL = RuntimeProfile(
    name="local",
    description="Local Jupyter — no Colab userdata, uses os.environ directly.",
    workdir=".",
    session_limit_hours=999.0,
    extra_pip_packages=(),
    flash_attn=False,
    colab_secrets=False,
    gpu_label="local machine",
    training_caveats=("Running locally — env vars come from the host shell, not Colab userdata.",),
)


RUNTIME_PROFILES: dict[str, RuntimeProfile] = {
    p.name: p for p in (COLAB_H100, COLAB_A100, COLAB_T4, LOCAL)
}


def get_runtime(name: str) -> RuntimeProfile:
    """Return a runtime profile by name."""
    if name not in RUNTIME_PROFILES:
        known = ", ".join(sorted(RUNTIME_PROFILES))
        raise ValueError(f"Unknown runtime {name!r}. Known: {known}")
    return RUNTIME_PROFILES[name]


def list_runtimes() -> list[dict[str, str]]:
    """Return a small summary of every runtime profile, for `--help`-style tooling."""
    return [
        {"name": p.name, "description": p.description, "gpu": p.gpu_label}
        for p in RUNTIME_PROFILES.values()
    ]
