"""Abstract base class for fleet compute providers."""
from __future__ import annotations

import abc
from typing import Any

# Node records conform to ``crucible.core.types.NodeRecord`` structurally,
# but the abstract interface stays on ``dict[str, Any]`` because many
# call sites in ``fleet/manager.py`` still pass through plain dicts from
# ``load_nodes`` and the provider-returned records may carry extra
# provider-specific keys. Tightening the signature here cascades into a
# fleet-wide ``dict[str, Any]`` → ``NodeRecord`` migration which is out
# of scope for Agent 4. See ``docs/cleanup/04-type-consolidation.md``.


class FleetProvider(abc.ABC):
    """Interface every compute-provider backend must implement.

    Providers translate generic fleet operations (provision, destroy, refresh)
    into provider-specific API calls.  The FleetManager delegates to a
    provider instance and never touches provider internals directly.
    """

    @abc.abstractmethod
    def provision(
        self,
        *,
        count: int,
        name_prefix: str,
        start_index: int = 1,
        replacement: bool = False,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Create *count* new compute nodes.

        Returns a list of node-record dicts (conforming to NodeRecord).
        """

    @abc.abstractmethod
    def destroy(
        self,
        nodes: list[dict[str, Any]],
        *,
        selected_names: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Destroy nodes (all, or those whose name is in *selected_names*).

        Returns the list of surviving nodes.
        """

    @abc.abstractmethod
    def refresh(self, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Re-query the provider API and return updated node records."""

    @abc.abstractmethod
    def wait_ready(
        self,
        nodes: list[dict[str, Any]],
        *,
        timeout_seconds: int = 900,
        poll_seconds: int = 15,
        stalled_seconds: int | None = None,
    ) -> list[dict[str, Any]]:
        """Block until every node in *nodes* is reachable via SSH.

        Returns the updated node list.
        Raises ``TimeoutError`` if *timeout_seconds* elapses.
        """

    # -- Optional lifecycle methods (providers override if supported) ------

    def stop(
        self,
        nodes: list[dict[str, Any]],
        *,
        selected_names: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Stop running nodes to save cost.  Default: no-op (returns nodes unchanged)."""
        return nodes

    def start(
        self,
        nodes: list[dict[str, Any]],
        *,
        selected_names: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Start stopped nodes.  Default: no-op (returns nodes unchanged)."""
        return nodes
