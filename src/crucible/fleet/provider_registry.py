"""Pluggable fleet provider registry.

Providers are registered as factory callables that accept keyword arguments
(``ssh_key``, ``image_name``, ``gpu_type_ids``, ``defaults``, etc.) and
return a :class:`FleetProvider` instance.

Built-in providers (``runpod``, ``ssh``) are registered at import time.
External plugins are discovered from ``.crucible/plugins/providers/`` and
``~/.crucible-hub/plugins/providers/``.

Usage::

    from crucible.fleet.provider_registry import build_provider
    provider = build_provider("runpod", ssh_key="~/.ssh/id_ed25519", ...)
"""
from __future__ import annotations

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from crucible.core.plugin_registry import PluginRegistry

PROVIDER_REGISTRY: PluginRegistry[Callable[..., Any]] = PluginRegistry("fleet_provider")


# ---------------------------------------------------------------------------
# Convenience wrappers (public API)
# ---------------------------------------------------------------------------

def register_provider(name: str, factory: Callable[..., Any], *, source: str = "builtin") -> None:
    """Register a fleet provider factory under *name*."""
    PROVIDER_REGISTRY.register(name, factory, source=source)


def build_provider(name: str, **kwargs: Any) -> FleetProvider:
    """Build a fleet provider by name."""
    from crucible.fleet.provider import FleetProvider  # noqa: F811 — lazy to avoid circular

    factory = PROVIDER_REGISTRY.get(name)
    if factory is None:
        from crucible.core.errors import PluginError
        available = ", ".join(PROVIDER_REGISTRY.list_plugins()) or "(none)"
        raise PluginError(
            f"Unknown fleet provider {name!r}. Registered: {available}"
        )
    return factory(**kwargs)


def list_providers() -> list[str]:
    """Return sorted list of registered provider names."""
    return PROVIDER_REGISTRY.list_plugins()


def list_providers_detailed() -> list[dict[str, str]]:
    """Return providers with source metadata."""
    return PROVIDER_REGISTRY.list_plugins_detailed()


# ---------------------------------------------------------------------------
# Built-in provider factories
# ---------------------------------------------------------------------------


def _runpod_factory(
    *,
    ssh_key: str = "",
    image_name: str = "",
    gpu_type_ids: list[str] | None = None,
    defaults: dict[str, Any] | None = None,
    gpu_count: int = 1,
    network_volume_id: str = "",
    template_id: str = "",
    **kwargs: Any,
) -> FleetProvider:
    from crucible.fleet.provider import FleetProvider  # noqa: F811
    from crucible.fleet.providers.runpod import RunPodProvider
    return RunPodProvider(
        image_name=image_name or "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
        gpu_type_ids=gpu_type_ids,
        ssh_key=ssh_key,
        interruptible=bool(kwargs.get("interruptible", True)),
        defaults=defaults or {},
        gpu_count=gpu_count,
        network_volume_id=network_volume_id,
        template_id=template_id,
    )


def _ssh_factory(
    *,
    ssh_key: str = "",
    defaults: dict[str, Any] | None = None,
    **kwargs: Any,
) -> FleetProvider:
    from crucible.fleet.provider import FleetProvider  # noqa: F811
    from crucible.fleet.providers.ssh import SSHProvider
    return SSHProvider(
        ssh_key=ssh_key,
        defaults=defaults or {},
    )


# ---------------------------------------------------------------------------
# Register built-ins
# ---------------------------------------------------------------------------

register_provider("runpod", _runpod_factory)
register_provider("ssh", _ssh_factory)
