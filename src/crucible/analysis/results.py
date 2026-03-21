"""Load, merge, and filter experiment results from local and fleet JSONL files."""
from __future__ import annotations

from pathlib import Path

from crucible.core.config import ProjectConfig, load_config
from crucible.core.io import read_jsonl
from crucible.core.log import log_info, log_warn
from crucible.core.types import ExperimentResult


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
        ``"local"`` reads *results_file*, ``"fleet"`` reads
        *fleet_results_file*.
    """
    if cfg is None:
        cfg = load_config()
    root = cfg.project_root

    if source == "fleet":
        path = root / cfg.fleet_results_file
    else:
        path = root / cfg.results_file

    records = read_jsonl(path)
    log_info(f"Loaded {len(records)} result(s) from {path}")
    return records  # type: ignore[return-value]


def merged_results(cfg: ProjectConfig | None = None) -> list[ExperimentResult]:
    """Load from both local and fleet results, deduplicate by name (latest wins)."""
    if cfg is None:
        cfg = load_config()

    seen: dict[str, ExperimentResult] = {}
    for r in load_results(cfg, source="local") + load_results(cfg, source="fleet"):
        name = r.get("name") or r.get("id", "")
        seen[name] = r
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
