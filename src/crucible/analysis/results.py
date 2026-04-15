"""Load, merge, and filter experiment results from local, project, and fleet JSONL files."""
from __future__ import annotations

from typing import Any

from crucible.core.config import ProjectConfig, load_config
from crucible.core.io import read_jsonl
from crucible.core.log import log_info, log_warn
from crucible.core.types import ExperimentResult

_PROJECT_RUNS_FILE = ".crucible/projects/runs.jsonl"


def _normalize_project_run(record: dict[str, Any]) -> ExperimentResult:
    """Convert a persisted project-run record into an ExperimentResult-like row."""
    run_id = str(record.get("run_id", ""))
    wandb = record.get("wandb")
    return {
        "id": run_id,
        "launch_id": record.get("launch_id"),
        "name": (
            record.get("result_name")
            or record.get("variant_name")
            or record.get("name")
            or f"{record.get('project', 'project')}-{run_id}"
        ),
        "timestamp": (
            record.get("completed_at")
            or record.get("updated_at")
            or record.get("launched_at")
            or ""
        ),
        "config": record.get("resolved_overrides") or record.get("overrides") or {},
        "result": record.get("result"),
        "status": record.get("status", "launched"),
        "project": record.get("project"),
        "remote_node": record.get("remote_node") or record.get("node_name"),
        "execution_provider": record.get("execution_provider"),
        "contract_status": record.get("contract_status"),
        "wandb": wandb,
        "launcher": record.get("launcher"),
        "launcher_source": record.get("launcher_source"),
        "failure_class": record.get("failure_class"),
        "status_reason": record.get("status_reason"),
        "remote_node_state": record.get("remote_node_state"),
        "source": "project",
    }


def _load_project_results(cfg: ProjectConfig) -> list[ExperimentResult]:
    path = cfg.project_root / _PROJECT_RUNS_FILE
    latest_by_run: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        run_id = str(record.get("run_id", ""))
        if not run_id:
            continue
        latest_by_run[run_id] = record
    return [_normalize_project_run(record) for record in latest_by_run.values()]


def load_results(
    cfg: ProjectConfig | None = None,
    *,
    source: str = "local",
) -> list[ExperimentResult]:
    """Load experiment results from a single JSONL source.

    Parameters
    ----------
    cfg:
        Project configuration.  Loaded automatically when *None*.
    source:
        ``"local"`` reads *results_file*, ``"project"`` reads
        ``.crucible/projects/runs.jsonl``, and ``"fleet"`` reads
        *fleet_results_file*.
    """
    if cfg is None:
        cfg = load_config()
    root = cfg.project_root

    if source == "project":
        records = _load_project_results(cfg)
        log_info(f"Loaded {len(records)} result(s) from {root / _PROJECT_RUNS_FILE}")
        return records
    if source == "fleet":
        path = root / cfg.fleet_results_file
    else:
        path = root / cfg.results_file

    records = read_jsonl(path)
    log_info(f"Loaded {len(records)} result(s) from {path}")
    return records  # type: ignore[return-value]


def merged_results(cfg: ProjectConfig | None = None) -> list[ExperimentResult]:
    """Load from local, project, and fleet results, deduplicate by run id (latest wins)."""
    if cfg is None:
        cfg = load_config()

    seen: dict[str, ExperimentResult] = {}
    for r in (
        load_results(cfg, source="local")
        + load_results(cfg, source="project")
        + load_results(cfg, source="fleet")
    ):
        key = str(r.get("id") or r.get("run_id") or r.get("name", ""))
        if not key:
            continue
        seen[key] = r
    return list(seen.values())


def completed_results(
    cfg: ProjectConfig | None = None,
    *,
    include_fleet: bool = True,
) -> list[ExperimentResult]:
    """Return only completed experiments that have a non-null result dict.

    Parameters
    ----------
    cfg:
        Project configuration.
    include_fleet:
        When *True* (default), merge fleet results before filtering.
    """
    source = merged_results(cfg) if include_fleet else load_results(cfg, source="local")
    completed = [
        r for r in source
        if r.get("status") == "completed" and r.get("result")
    ]
    if not completed:
        log_warn("No completed experiments found")
    return completed
