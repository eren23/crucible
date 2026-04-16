"""Abstract base class for fleet compute providers."""
from __future__ import annotations

import abc
from typing import Any

from crucible.core.types import NodeRecord


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
    ) -> list[NodeRecord]:
        """Create *count* new compute nodes.

        Returns a list of node-record dicts (conforming to NodeRecord).
        """

    @abc.abstractmethod
    def destroy(
        self,
        nodes: list[NodeRecord],
        *,
        selected_names: set[str] | None = None,
    ) -> list[NodeRecord]:
        """Destroy nodes (all, or those whose name is in *selected_names*).

        Returns the list of surviving nodes.
        """

    @abc.abstractmethod
    def refresh(self, nodes: list[NodeRecord]) -> list[NodeRecord]:
        """Re-query the provider API and return updated node records."""

    @abc.abstractmethod
    def wait_ready(
        self,
        nodes: list[NodeRecord],
        *,
        timeout_seconds: int = 900,
        poll_seconds: int = 15,
        stalled_seconds: int | None = None,
    ) -> list[NodeRecord]:
        """Block until every node in *nodes* is reachable via SSH.

        Returns the updated node list.
        Raises ``TimeoutError`` if *timeout_seconds* elapses.
        """

    # -- Optional lifecycle methods (providers override if supported) ------

    def stop(
        self,
        nodes: list[NodeRecord],
        *,
        selected_names: set[str] | None = None,
    ) -> list[NodeRecord]:
        """Stop running nodes to save cost.  Default: no-op (returns nodes unchanged)."""
        return nodes

    def start(
        self,
        nodes: list[NodeRecord],
        *,
        selected_names: set[str] | None = None,
    ) -> list[NodeRecord]:
        """Start stopped nodes.  Default: no-op (returns nodes unchanged)."""
        return nodes
