"""Optional Weights & Biases integration.

WandbLogger wraps the wandb SDK behind a safe interface that stays inert
when WANDB_PROJECT is not set or wandb is not installed.  This lets the
runner always create a logger without gating on availability.

Environment variables consumed:
  WANDB_PROJECT   - W&B project name (required to enable)
  WANDB_ENTITY    - W&B entity / team
  WANDB_RUN_NAME  - Display name for the run (defaults to run_id)
  WANDB_RUN_GROUP - Grouping key
  WANDB_JOB_TYPE  - Job type label
  WANDB_TAGS      - Comma-separated extra tags
  WANDB_MODE      - "online" | "offline" | "disabled"
"""
from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING

from crucible.core.io import _json_ready

if TYPE_CHECKING:
    from crucible.runner.tracker import RunTracker


class WandbLogger:
    """Optional W&B logger that stays inert when WANDB_PROJECT is unset."""

    def __init__(
        self,
        run: Any | None = None,
        enabled: bool = False,
        error: str | None = None,
    ):
        self.run = run
        self.enabled = enabled
        self.error = error

    @property
    def url(self) -> str | None:
        """Return the W&B run URL, or None if not active."""
        if self.run is None:
            return None
        return getattr(self.run, "url", None)

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        config: dict[str, Any],
        backend: str,
        tracker: "RunTracker | None" = None,
        job_type: str | None = None,
        tags: list[str] | None = None,
    ) -> "WandbLogger":
        """Create a WandbLogger, initialising a W&B run if configured.

        Returns an inert logger if WANDB_PROJECT is unset or wandb is
        not installed -- callers never need to check availability.
        """
        project = os.environ.get("WANDB_PROJECT", "").strip()
        if not project:
            if tracker is not None:
                tracker.update(
                    wandb={"enabled": False, "reason": "WANDB_PROJECT unset"},
                    heartbeat=False,
                )
            return cls()

        try:
            import wandb  # type: ignore
        except ImportError:
            error = "wandb not installed"
            if tracker is not None:
                tracker.update(
                    wandb={"enabled": False, "error": error},
                    heartbeat=False,
                )
            return cls(error=error)

        env_tags = [
            tag.strip()
            for tag in os.environ.get("WANDB_TAGS", "").split(",")
            if tag.strip()
        ]
        final_tags = list(dict.fromkeys([backend, *(tags or []), *env_tags]))

        kwargs: dict[str, Any] = {
            "project": project,
            "entity": os.environ.get("WANDB_ENTITY") or None,
            "name": os.environ.get("WANDB_RUN_NAME", run_id),
            "group": os.environ.get("WANDB_RUN_GROUP") or None,
            "job_type": os.environ.get("WANDB_JOB_TYPE") or job_type,
            "tags": final_tags or None,
            "config": _json_ready(config),
            "mode": os.environ.get("WANDB_MODE") or None,
            "reinit": True,
        }
        run = wandb.init(**kwargs)
        logger = cls(run=run, enabled=True)

        if tracker is not None:
            tracker.update(
                wandb={
                    "enabled": True,
                    "url": logger.url,
                    "project": project,
                    "entity": os.environ.get("WANDB_ENTITY") or None,
                    "mode": os.environ.get("WANDB_MODE") or "online",
                },
                heartbeat=False,
            )
        return logger

    def log(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        """Log metrics to W&B. No-op if not enabled."""
        if not self.enabled or self.run is None:
            return
        payload = {k: v for k, v in _json_ready(metrics).items() if v is not None}
        if not payload:
            return
        if step is None:
            self.run.log(payload)
        else:
            self.run.log(payload, step=step)

    def update_summary(self, values: dict[str, Any]) -> None:
        """Update the W&B run summary. No-op if not enabled."""
        if not self.enabled or self.run is None:
            return
        for key, value in _json_ready(values).items():
            self.run.summary[key] = value

    def finish(self, exit_code: int = 0) -> None:
        """Finish the W&B run. No-op if not enabled."""
        if not self.enabled:
            return
        try:
            import wandb  # type: ignore
        except ImportError:
            return
        wandb.finish(exit_code=exit_code)
