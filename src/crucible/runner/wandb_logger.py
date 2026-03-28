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

import json
import os
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

from crucible.core.io import _json_ready, read_jsonl

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
        env: Mapping[str, str] | None = None,
    ) -> "WandbLogger":
        """Create a WandbLogger, initialising a W&B run if configured.

        Returns an inert logger if WANDB_PROJECT is unset or wandb is
        not installed -- callers never need to check availability.
        """
        env_map = env or os.environ
        project = env_map.get("WANDB_PROJECT", "").strip()
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
            for tag in env_map.get("WANDB_TAGS", "").split(",")
            if tag.strip()
        ]
        final_tags = list(dict.fromkeys([backend, *(tags or []), *env_tags]))

        kwargs: dict[str, Any] = {
            "project": project,
            "entity": env_map.get("WANDB_ENTITY") or None,
            "name": env_map.get("WANDB_RUN_NAME", run_id),
            "group": env_map.get("WANDB_RUN_GROUP") or None,
            "job_type": env_map.get("WANDB_JOB_TYPE") or job_type,
            "tags": final_tags or None,
            "config": _json_ready(config),
            "mode": env_map.get("WANDB_MODE") or None,
            "reinit": True,
        }
        try:
            run = wandb.init(**kwargs)
        except Exception as init_exc:
            error = f"wandb.init failed: {init_exc}"
            if tracker is not None:
                tracker.update(
                    wandb={"enabled": False, "error": error},
                    heartbeat=False,
                )
            return cls(error=error)
        logger = cls(run=run, enabled=True)

        if tracker is not None:
            tracker.update(
                wandb={
                    "enabled": True,
                    "url": logger.url,
                    "project": project,
                    "entity": env_map.get("WANDB_ENTITY") or None,
                    "mode": env_map.get("WANDB_MODE") or "online",
                    "run_name": env_map.get("WANDB_RUN_NAME", run_id),
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

    def log_image(self, image_path: str | Path, caption: str = "", key: str = "image") -> bool:
        """Log an image to W&B. Returns True if successful."""
        if not self.enabled or self.run is None:
            return False
        try:
            import wandb  # type: ignore
        except ImportError:
            return False
        img = wandb.Image(str(image_path), caption=caption)
        self.run.log({key: img})
        return True

    def annotate(self, notes: list[str] | None = None, findings: list[str] | None = None) -> bool:
        """Push notes and findings to W&B run summary."""
        if not self.enabled or self.run is None:
            return False
        if notes:
            self.run.summary["crucible_notes"] = notes
        if findings:
            self.run.summary["crucible_findings"] = findings
        return True

    def update_config(self, extra: dict[str, Any]) -> None:
        """Update W&B run config with additional crucible metadata."""
        if not self.enabled or self.run is None:
            return
        self.run.config.update(extra)

    def finish(self, exit_code: int = 0) -> None:
        """Finish the W&B run. No-op if not enabled."""
        if not self.enabled:
            return
        try:
            import wandb  # type: ignore
        except ImportError:
            return
        wandb.finish(exit_code=exit_code)


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def wandb_annotate_finished_run(
    wandb_url: str,
    notes: list[str] | None = None,
    findings: list[str] | None = None,
) -> bool:
    """Annotate an already-finished W&B run using the Public API."""
    try:
        import wandb  # type: ignore
    except ImportError:
        return False

    # Parse entity/project/run_id from URL
    # URL format: https://wandb.ai/{entity}/{project}/runs/{run_id}
    try:
        parts = wandb_url.rstrip("/").split("/")
        # Find the 'runs' segment and extract the pieces around it
        runs_idx = parts.index("runs")
        run_id = parts[runs_idx + 1]
        project = parts[runs_idx - 1]
        entity = parts[runs_idx - 2]
    except (ValueError, IndexError):
        return False

    try:
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")
        if notes:
            run.summary["crucible_notes"] = notes
        if findings:
            run.summary["crucible_findings"] = findings
        run.summary.update()
        return True
    except Exception:
        return False


def _resolve_wandb_url(run_id: str, config: Any) -> str | None:
    """Look up W&B URL for a crucible run_id from status sidecar or results."""
    # Check logs/{run_id}.status.json for wandb.url
    try:
        logs_dir = Path(config.project_root) / config.logs_dir
        status_path = logs_dir / f"{run_id}.status.json"
        if status_path.exists():
            data = json.loads(status_path.read_text(encoding="utf-8"))
            wandb_info = data.get("wandb", {})
            url = wandb_info.get("url")
            if url:
                return url
    except Exception:
        pass

    # Fallback: scan results JSONL
    try:
        results_path = Path(config.project_root) / config.results_file
        for record in read_jsonl(results_path):
            if record.get("id") == run_id or record.get("run_id") == run_id:
                wandb_info = record.get("wandb", {})
                if isinstance(wandb_info, dict):
                    url = wandb_info.get("url")
                    if url:
                        return url
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# External project WandB metric fetch
# ---------------------------------------------------------------------------

def fetch_wandb_metrics(
    project: str,
    entity: str | None = None,
    run_name: str | None = None,
) -> dict[str, float]:
    """Fetch final metrics from WandB API for a completed run.

    Returns a dict of metric names to float values from the run summary.
    Returns empty dict if wandb is unavailable or no matching run is found.
    """
    try:
        import wandb
    except ImportError:
        return {}

    try:
        run = _fetch_wandb_run(project=project, entity=entity, run_name=run_name)
        if run is None:
            return {}
        result: dict[str, float] = {}
        for key, val in run.summary.items():
            if key.startswith("_"):
                continue
            if isinstance(val, (int, float)):
                result[key] = float(val)
        return result
    except Exception:
        return {}


def fetch_wandb_run_info(
    project: str,
    entity: str | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Fetch metrics + URL for a completed W&B run."""
    try:
        run = _fetch_wandb_run(project=project, entity=entity, run_name=run_name)
        if run is None:
            return {}
        metrics: dict[str, float] = {}
        for key, val in run.summary.items():
            if key.startswith("_"):
                continue
            if isinstance(val, (int, float)):
                metrics[key] = float(val)
        return {
            "url": getattr(run, "url", None),
            "run_name": getattr(run, "name", None) or getattr(run, "display_name", None),
            "metrics": metrics,
        }
    except Exception:
        return {}


def _fetch_wandb_run(
    *,
    project: str,
    entity: str | None = None,
    run_name: str | None = None,
) -> Any | None:
    """Fetch a single W&B run, preferring the supplied display name."""
    try:
        import wandb
    except ImportError:
        return None

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    filters = {}
    if run_name:
        filters["display_name"] = run_name
    runs = api.runs(path, filters=filters, per_page=5)
    if not runs:
        return None
    return runs[0]
