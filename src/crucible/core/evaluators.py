"""Evaluator plugin system for crucible.

Phase 3.3 of the ecosystem-connections plan — a new plugin family
parallel to ``data_sources/``. Where ``data_sources/`` describes
*inputs* (training corpora, manifests), ``evaluators/`` describes
*outputs* (benchmark scores against a trained checkpoint).

Distinct from the Tier-13 ``eval_watch_*`` daemon, which schedules
arbitrary scripts on new checkpoints. Evaluators standardize the
plugin shape (a Python class with ``validate``/``evaluate`` methods)
so the eval-watcher daemon and any future MCP tool can pick adapters
from the same 3-tier registry (builtin < global < local) that every
other plugin family uses.

Builtin evaluators (Phase 3+):
- ``lm_eval_harness``: shells out to lm-evaluation-harness (HF)
- ``big_bench``: BIG-bench Lite tasks (planned)
- ``papers_with_code``: pull leaderboard priors (planned, lit-injection)

Each evaluator's :meth:`evaluate` returns an :class:`EvalResult` with
a ``scores`` dict whose keys are benchmark names + metric values. The
caller (eval-watcher daemon, autonomous loop) decides how to use them.

Registry pattern matches :mod:`crucible.core.data_sources` so dynamic
registration (``.crucible/plugins/evaluators/*.py``) works identically.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.core.plugin_registry import PluginRegistry


@dataclass
class EvalResult:
    """Output of an evaluator run.

    Attributes
    ----------
    scores:
        Map of metric name → numeric value. Convention: scalar metrics
        as ``float``; multi-task benchmarks expose nested-dotted keys
        (e.g., ``"hellaswag.acc": 0.62, "hellaswag.acc_norm": 0.65``).
    metadata:
        Free-form metadata (eval version, prompt template, task list).
    success:
        Whether the evaluator ran cleanly. ``False`` when validate or
        evaluate raised — ``error`` carries the message.
    error:
        Error message when ``success`` is False; ``None`` otherwise.
    duration_seconds:
        Wall-clock cost of the evaluate() call. ``None`` if untimed.
    """

    scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str | None = None
    duration_seconds: float | None = None


@dataclass
class EvalValidationResult:
    """Whether an evaluator is runnable in the current environment."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class EvaluatorPlugin(ABC):
    """Abstract base for benchmark evaluator plugins.

    Subclasses implement :meth:`validate` (cheap pre-flight check) and
    :meth:`evaluate` (actual benchmark run). Both must be safe to call
    on a missing or partial checkpoint — return an :class:`EvalResult`
    with ``success=False`` rather than raising into a fleet timer.
    """

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name = name
        self.config = config

    @abstractmethod
    def validate(self) -> EvalValidationResult:
        """Pre-flight: is this evaluator runnable here?

        Should not touch the network or run the benchmark. Typical
        checks: the underlying binary is on $PATH, required Python
        packages are importable, the task list is non-empty.
        """
        ...

    @abstractmethod
    def evaluate(self, checkpoint_path: Path) -> EvalResult:
        """Run the benchmark against the checkpoint.

        Implementations must catch their own exceptions and return an
        :class:`EvalResult` with ``success=False, error=<msg>`` on
        failure — the eval-watcher daemon will not catch raises for
        you.
        """
        ...

    def describe(self) -> dict[str, Any]:
        """Optional: structured metadata about this evaluator.

        Default implementation returns name + config keys. Override to
        surface task lists, version pins, expected runtime, etc.
        """
        return {"name": self.name, "config_keys": sorted(self.config.keys())}


# Singleton registry — 3-tier (builtin < global < local) per the
# project-wide PluginRegistry pattern.
_EVALUATOR_REGISTRY = PluginRegistry[EvaluatorPlugin]("evaluator")


def register_evaluator(
    name: str, cls: type[EvaluatorPlugin], source: str = "builtin"
) -> None:
    """Register an evaluator plugin under ``name``.

    ``source`` is one of ``"builtin"``, ``"global"`` (loaded from
    ``~/.crucible-hub/plugins/evaluators/``), or ``"local"`` (loaded
    from ``<project>/.crucible/plugins/evaluators/``). Local wins over
    global wins over builtin when the same name is registered at
    multiple tiers.
    """
    _EVALUATOR_REGISTRY.register(name, cls, source=source)


def list_evaluators() -> list[str]:
    """Return all registered evaluator names."""
    return _EVALUATOR_REGISTRY.list_plugins()


def describe_evaluator(name: str) -> dict[str, str] | None:
    """Return registry metadata for an evaluator: ``{name, type, source}``."""
    return _EVALUATOR_REGISTRY.describe_plugin(name)


def get_evaluator_class(name: str) -> type[EvaluatorPlugin] | None:
    """Return the registered class for an evaluator name, or None."""
    return _EVALUATOR_REGISTRY.get(name)


def instantiate_evaluator(
    name: str, config: dict[str, Any] | None = None
) -> EvaluatorPlugin:
    """Look up + instantiate an evaluator by name.

    Raises :class:`KeyError` if the name isn't registered (matches
    PluginRegistry semantics).
    """
    cls = _EVALUATOR_REGISTRY.get(name)
    if cls is None:
        raise KeyError(
            f"No evaluator registered as {name!r}. "
            f"Registered: {list_evaluators()}"
        )
    return cls(name=name, config=dict(config or {}))


def discover_evaluator_plugins(project_root: Path | None = None) -> None:
    """Trigger auto-discovery of evaluator plugins on disk.

    Loads ``.crucible/plugins/evaluators/*.py`` (local) and
    ``~/.crucible-hub/plugins/evaluators/*.py`` (global) so user
    plugins are available without explicit registration. Matches the
    pattern used by data_sources / optimizers / etc.
    """
    _EVALUATOR_REGISTRY.discover(project_root=project_root)


# Trigger builtin registration on import. Each builtin module
# self-registers via register_evaluator() at module load.
def _register_builtins() -> None:
    try:
        from crucible.evaluators import lm_eval_harness  # noqa: F401
    except ImportError:
        # The builtin module may not exist in a stripped install; that's fine.
        pass


_register_builtins()
