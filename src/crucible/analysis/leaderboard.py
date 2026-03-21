"""Leaderboard ranking, sensitivity analysis, and Pareto frontier."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from crucible.core.config import ProjectConfig
from crucible.core.log import log_info, log_warn
from crucible.core.types import ExperimentResult

from crucible.analysis.results import completed_results

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DEFAULT_METRIC = "val_bpb"


def _metric_value(result: ExperimentResult, metric: str) -> float:
    """Extract the primary metric from an experiment result dict.

    Raises *KeyError* when the metric key is missing from the result payload.
    """
    return result["result"][metric]  # type: ignore[index]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def leaderboard(
    top_n: int = 50,
    *,
    metric: str = _DEFAULT_METRIC,
    cfg: ProjectConfig | None = None,
) -> list[ExperimentResult]:
    """Return the *top_n* completed experiments ranked by *metric* (ascending).

    Lower values are better (e.g. loss, BPB).  Pass ``metric`` to rank by a
    different key inside the ``result`` dict.
    """
    results = completed_results(cfg)
    if not results:
        return []

    # Filter to results that actually contain the requested metric key
    valid = [r for r in results if metric in (r.get("result") or {})]
    if not valid:
        log_warn(f"No completed results contain metric '{metric}'")
        return []

    valid.sort(key=lambda r: _metric_value(r, metric))
    log_info(f"Leaderboard: {len(valid)} results ranked by {metric}, showing top {top_n}")
    return valid[:top_n]


def sensitivity_analysis(
    *,
    metric: str = _DEFAULT_METRIC,
    cfg: ProjectConfig | None = None,
) -> dict[str, list[tuple[Any, float]]]:
    """For each config key with >1 unique value, return ``(value, metric)`` pairs.

    Pairs are sorted by metric ascending so the first entry is the best
    observed value for that key.

    Returns
    -------
    dict mapping config-key names to sorted lists of ``(config_value, metric_value)``
    tuples.  Only keys whose values actually vary across experiments are included.
    """
    results = completed_results(cfg)
    if not results:
        return {}

    key_values: dict[str, list[tuple[Any, float]]] = defaultdict(list)
    for r in results:
        res = r.get("result") or {}
        if metric not in res:
            continue
        metric_val: float = res[metric]
        for k, v in (r.get("config") or {}).items():
            key_values[k].append((v, metric_val))

    sensitivity: dict[str, list[tuple[Any, float]]] = {}
    for k, pairs in key_values.items():
        distinct_values = {v for v, _ in pairs}
        if len(distinct_values) > 1:
            pairs.sort(key=lambda p: p[1])
            sensitivity[k] = pairs

    return sensitivity


def pareto_frontier(
    *,
    metric: str = _DEFAULT_METRIC,
    size_key: str = "model_bytes",
    cfg: ProjectConfig | None = None,
) -> list[ExperimentResult]:
    """Return Pareto-optimal experiments on two dimensions.

    Default dimensions: ``metric`` (lower-is-better) vs ``size_key``
    (lower-is-better, typically ``model_bytes``).

    An experiment is Pareto-optimal if no other experiment is strictly better
    on *both* dimensions simultaneously.

    Parameters
    ----------
    metric:
        Key inside ``result`` dict for the quality axis (default ``val_bpb``).
    size_key:
        Key on the top-level result record for the cost/size axis (default
        ``model_bytes``).
    cfg:
        Project configuration.
    """
    results = [
        r for r in completed_results(cfg)
        if r.get(size_key) and metric in (r.get("result") or {})
    ]
    if not results:
        return []

    # Sort by metric ascending (best quality first)
    results.sort(key=lambda r: _metric_value(r, metric))

    pareto: list[ExperimentResult] = []
    min_size = float("inf")
    for r in results:
        size = r[size_key]  # type: ignore[literal-required]
        if size <= min_size:
            pareto.append(r)
            min_size = size

    log_info(f"Pareto frontier: {len(pareto)} optimal point(s) from {len(results)} candidates")
    return pareto
