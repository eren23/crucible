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


def _resolve_metric(metric: str | None, cfg: ProjectConfig | None) -> str:
    """Return an explicit metric name, falling back to config or 'val_loss'."""
    if metric is not None:
        return metric
    if cfg is not None:
        return cfg.metrics.primary
    return "val_loss"


def _metric_value(result: ExperimentResult, metric: str) -> float:
    """Extract the primary metric from an experiment result dict.

    Raises *KeyError* when the metric key is missing from the result payload.
    """
    return result["result"][metric]  # type: ignore[index]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _resolve_direction(direction: str | None, cfg: ProjectConfig | None) -> str:
    """Return an explicit direction, falling back to config or 'minimize'."""
    if direction is not None:
        return direction
    if cfg is not None:
        return cfg.metrics.direction
    return "minimize"


def leaderboard(
    results: list[ExperimentResult] | None = None,
    top_n: int = 50,
    *,
    metric: str | None = None,
    direction: str | None = None,
    cfg: ProjectConfig | None = None,
) -> list[ExperimentResult]:
    """Return the *top_n* completed experiments ranked by *metric*.

    When ``direction`` is ``"minimize"`` (default) lower values are better
    (e.g. loss, BPB).  When ``"maximize"`` higher values are better
    (e.g. accuracy, reward).

    Parameters
    ----------
    results:
        Pre-loaded list of completed experiment results.  When *None*
        (default), results are loaded via :func:`completed_results`.
    metric:
        Result key to rank by.  When *None* (default), resolved from
        ``cfg.metrics.primary`` or falls back to ``"val_loss"``.
    direction:
        Sort direction ``"minimize"`` or ``"maximize"``.  When *None*,
        resolved from ``cfg.metrics.direction`` or falls back to
        ``"minimize"``.
    """
    metric = _resolve_metric(metric, cfg)
    direction = _resolve_direction(direction, cfg)
    if results is None:
        results = completed_results(cfg)
    if not results:
        return []

    # Filter to results that actually contain the requested metric key
    valid = [r for r in results if metric in (r.get("result") or {})]
    if not valid:
        log_warn(f"No completed results contain metric '{metric}'")
        return []

    reverse = direction == "maximize"
    valid.sort(key=lambda r: _metric_value(r, metric), reverse=reverse)
    log_info(f"Leaderboard: {len(valid)} results ranked by {metric} ({direction}), showing top {top_n}")
    return valid[:top_n]


def sensitivity_analysis(
    results: list[ExperimentResult] | None = None,
    *,
    metric: str | None = None,
    cfg: ProjectConfig | None = None,
) -> dict[str, list[tuple[Any, float]]]:
    """For each config key with >1 unique value, return ``(value, metric)`` pairs.

    Pairs are sorted by metric ascending so the first entry is the best
    observed value for that key.

    Parameters
    ----------
    results:
        Pre-loaded list of completed experiment results.  When *None*
        (default), results are loaded via :func:`completed_results`.
    metric:
        Result key to rank by.  When *None* (default), resolved from
        ``cfg.metrics.primary`` or falls back to ``"val_loss"``.

    Returns
    -------
    dict mapping config-key names to sorted lists of ``(config_value, metric_value)``
    tuples.  Only keys whose values actually vary across experiments are included.
    """
    metric = _resolve_metric(metric, cfg)
    if results is None:
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
    metric: str | None = None,
    size_key: str | None = None,
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
        Key inside ``result`` dict for the quality axis.  When *None*,
        resolved from ``cfg.metrics.primary`` or falls back to ``"val_loss"``.
    size_key:
        Key on the top-level result record for the cost/size axis.  When
        *None*, resolved from ``cfg.metrics.size`` or falls back to
        ``"model_bytes"``.
    cfg:
        Project configuration.
    """
    metric = _resolve_metric(metric, cfg)
    if size_key is None:
        size_key = cfg.metrics.size if cfg is not None else "model_bytes"
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
