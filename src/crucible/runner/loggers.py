"""Pluggable training logger registry.

Provides an abstract ``TrainingLogger`` base class and a registry of
available logging backends.  Training backends can instantiate one or
more loggers by name (comma-separated env var ``LOGGING_BACKEND``).

Built-in loggers:
    ``wandb``   — adapter for the existing :class:`WandbLogger`
    ``console`` — print-based (metrics to stdout)
    ``jsonl``   — append-only JSONL file logger

Usage::

    from crucible.runner.loggers import build_logger, build_multi_logger
    logger = build_logger("wandb", run_id="exp-001", config={})
    # Or multiple simultaneously:
    logger = build_multi_logger("wandb,console", run_id="exp-001", config={})
"""
from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from crucible.core.plugin_registry import PluginRegistry

LOGGER_REGISTRY = PluginRegistry("logger")


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class TrainingLogger(ABC):
    """Interface for training logging backends."""

    @abstractmethod
    def log(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        """Log a dict of metrics at an optional step."""

    @abstractmethod
    def finish(self, exit_code: int = 0) -> None:
        """Finalize the logging session."""


# ---------------------------------------------------------------------------
# Built-in loggers
# ---------------------------------------------------------------------------

class ConsoleLogger(TrainingLogger):
    """Prints metrics to stdout."""

    def __init__(self, **kwargs: Any) -> None:
        self.prefix = kwargs.get("prefix", "")

    def log(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        parts = [f"{k}={v}" for k, v in sorted(metrics.items())]
        line = " ".join(parts)
        if step is not None:
            line = f"step:{step} {line}"
        if self.prefix:
            line = f"[{self.prefix}] {line}"
        print(line, flush=True)

    def finish(self, exit_code: int = 0) -> None:
        pass


class JsonlLogger(TrainingLogger):
    """Appends metrics as JSON lines to a file."""

    def __init__(self, *, run_id: str = "", log_dir: str = "", **kwargs: Any) -> None:
        log_dir = log_dir or os.environ.get("CRUCIBLE_LOG_DIR", ".crucible/logs")
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"{run_id or 'unknown'}.jsonl"
        self._f = open(self._path, "a")
        self._closed = False

    def log(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        if self._closed:
            return
        entry: dict[str, Any] = {"ts": time.time()}
        if step is not None:
            entry["step"] = step
        entry.update(metrics)
        self._f.write(json.dumps(entry) + "\n")
        self._f.flush()

    def finish(self, exit_code: int = 0) -> None:
        if not self._closed:
            self._f.close()
            self._closed = True

    def __del__(self) -> None:
        self.finish()


class WandbLoggerAdapter(TrainingLogger):
    """Adapter wrapping the existing WandbLogger for the registry interface."""

    def __init__(self, *, run_id: str = "", config: dict | None = None, **kwargs: Any) -> None:
        from crucible.runner.wandb_logger import WandbLogger
        self._inner = WandbLogger.create(
            run_id=run_id,
            config=config or {},
            backend=kwargs.get("backend", "generic"),
        )

    def log(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        if self._inner.enabled:
            self._inner.log(metrics, step=step)

    def finish(self, exit_code: int = 0) -> None:
        if self._inner.enabled and self._inner.run is not None:
            self._inner.run.finish(exit_code=exit_code)


# ---------------------------------------------------------------------------
# Multi-logger (composes multiple backends)
# ---------------------------------------------------------------------------

class MultiLogger(TrainingLogger):
    """Fans out log/finish calls to multiple backends."""

    def __init__(self, loggers: list[TrainingLogger]) -> None:
        self.loggers = loggers

    def log(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        for logger in self.loggers:
            logger.log(metrics, step=step)

    def finish(self, exit_code: int = 0) -> None:
        for logger in self.loggers:
            try:
                logger.finish(exit_code=exit_code)
            except Exception:
                pass  # Ensure all loggers get finalized even if one raises


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------

def register_logger(name: str, factory: Any, *, source: str = "builtin") -> None:
    LOGGER_REGISTRY.register(name, factory, source=source)


def build_logger(name: str, **kwargs: Any) -> TrainingLogger:
    """Build a single logger by name."""
    factory = LOGGER_REGISTRY.get(name)
    if factory is None:
        from crucible.core.errors import PluginError
        available = ", ".join(LOGGER_REGISTRY.list_plugins()) or "(none)"
        raise PluginError(f"Unknown logger {name!r}. Registered: {available}")
    return factory(**kwargs)


def build_multi_logger(names: str, **kwargs: Any) -> TrainingLogger:
    """Build one or more loggers from a comma-separated string.

    Returns a single logger if only one name, or a MultiLogger if multiple.
    """
    name_list = [n.strip() for n in names.split(",") if n.strip()]
    if not name_list:
        return ConsoleLogger(**kwargs)
    loggers = [build_logger(n, **kwargs) for n in name_list]
    if len(loggers) == 1:
        return loggers[0]
    return MultiLogger(loggers)


def list_loggers() -> list[str]:
    return LOGGER_REGISTRY.list_plugins()


def list_loggers_detailed() -> list[dict[str, str]]:
    return LOGGER_REGISTRY.list_plugins_detailed()


# ---------------------------------------------------------------------------
# Register built-ins
# ---------------------------------------------------------------------------

register_logger("console", ConsoleLogger)
register_logger("jsonl", JsonlLogger)
register_logger("wandb", WandbLoggerAdapter)
