"""Shared RunPod + W&B contract enforcement for user-facing experiment entrypoints."""
from __future__ import annotations

import os
from typing import Any, Mapping

from crucible.core.config import ProjectConfig
from crucible.core.errors import ConfigError


def _cfg_attr(obj: Any, name: str, default: Any) -> Any:
    return getattr(obj, name, default)


def resolve_wandb_settings(
    config: ProjectConfig,
    *,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve W&B settings from config with environment fallbacks."""
    env_map = env or os.environ
    wandb_cfg = _cfg_attr(config, "wandb", None)
    project = str(_cfg_attr(wandb_cfg, "project", "") or env_map.get("WANDB_PROJECT", "")).strip()
    entity = str(_cfg_attr(wandb_cfg, "entity", "") or env_map.get("WANDB_ENTITY", "")).strip()
    mode = str(_cfg_attr(wandb_cfg, "mode", "online") or env_map.get("WANDB_MODE", "online")).strip() or "online"
    api_key_present = bool(str(env_map.get("WANDB_API_KEY", "")).strip())
    return {
        "required": bool(_cfg_attr(wandb_cfg, "required", True)),
        "project": project,
        "entity": entity or None,
        "mode": mode,
        "api_key_present": api_key_present,
    }


def contract_metadata(
    config: ProjectConfig,
    *,
    env: Mapping[str, str] | None = None,
    remote_node: str | None = None,
) -> dict[str, Any]:
    """Build persistable execution-contract metadata."""
    wandb = resolve_wandb_settings(config, env=env)
    return {
        "execution_provider": str(_cfg_attr(_cfg_attr(config, "provider", None), "type", "runpod")).lower(),
        "remote_node": remote_node,
        "contract_status": "compliant",
        "wandb": {
            "required": wandb["required"],
            "project": wandb["project"],
            "entity": wandb["entity"],
            "mode": wandb["mode"],
        },
    }


def validate_experiment_contract(
    config: ProjectConfig,
    *,
    action: str,
    execution_mode: str,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate the user-facing experiment contract and return resolved metadata.

    execution_mode:
      - "local": direct local launch from a user-facing entrypoint
      - "remote": queue/fleet/external-project launch path
    """
    provider = str(_cfg_attr(_cfg_attr(config, "provider", None), "type", "runpod") or "").lower()
    policy = _cfg_attr(config, "execution_policy", None)
    required_provider = str(_cfg_attr(policy, "required_provider", "runpod") or "runpod").lower()
    require_remote = bool(_cfg_attr(policy, "require_remote", True))
    allow_local_dev = bool(_cfg_attr(policy, "allow_local_dev", False))

    if require_remote and provider != required_provider:
        raise ConfigError(
            f"{action} requires provider.type={required_provider!r}; found {provider!r}."
        )

    if execution_mode == "local" and require_remote and not allow_local_dev:
        raise ConfigError(
            f"{action} is blocked by execution_policy.require_remote=true. "
            "Use the fleet/RunPod execution path or set execution_policy.allow_local_dev=true "
            "for explicit local development runs."
        )

    wandb = resolve_wandb_settings(config, env=env)
    if wandb["required"]:
        if not wandb["project"]:
            raise ConfigError(
                f"{action} requires W&B. Set wandb.project in crucible.yaml or WANDB_PROJECT in the environment."
            )
        if wandb["mode"] != "disabled" and not wandb["api_key_present"]:
            raise ConfigError(
                f"{action} requires WANDB_API_KEY when wandb.mode={wandb['mode']!r}."
            )

    return contract_metadata(config, env=env)
