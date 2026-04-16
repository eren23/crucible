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


# ---------------------------------------------------------------------------
# N-dimensional Pareto utilities
# ---------------------------------------------------------------------------


def dominates(
    a: list[float], b: list[float], directions: list[str]
) -> bool:
    """Return True iff point *a* dominates point *b* on every axis.

    An axis direction of ``"minimize"`` means lower is better; ``"maximize"``
    means higher is better. *a* dominates *b* when *a* is at least as good as
    *b* on all axes and strictly better on at least one.
    """
    if len(a) != len(b) or len(a) != len(directions):
        raise ValueError("Dimension mismatch between points and directions")

    strictly_better_anywhere = False
    for av, bv, d in zip(a, b, directions):
        if d == "minimize":
            if av > bv:
                return False
            if av < bv:
                strictly_better_anywhere = True
        elif d == "maximize":
            if av < bv:
                return False
            if av > bv:
                strictly_better_anywhere = True
        else:
            raise ValueError(f"Unknown direction {d!r} (expected 'minimize'/'maximize')")
    return strictly_better_anywhere


def pareto_frontier_nd(
    points: list[list[float]],
    directions: list[str],
    *,
    ids: list[Any] | None = None,
) -> list[int]:
    """Return indices of non-dominated points across N dimensions.

    Parameters
    ----------
    points:
        List of *N*-dimensional metric vectors (one per candidate).
    directions:
        Per-axis optimization direction: ``"minimize"`` or ``"maximize"``.
    ids:
        Optional labels for logging; indices are returned regardless.

    Returns
    -------
    List of indices into *points* that are on the Pareto frontier.
    """
    if not points:
        return []
    if any(len(p) != len(directions) for p in points):
        raise ValueError("All points must match the length of `directions`")

    n = len(points)
    frontier: list[int] = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if dominates(points[j], points[i], directions):
                dominated = True
                break
        if not dominated:
            frontier.append(i)

    if ids is not None:
        log_info(
            f"Pareto frontier (N-D): {len(frontier)} non-dominated of {n} points"
        )
    return frontier


def hypervolume_2d(
    points: list[list[float]],
    directions: list[str],
    *,
    reference: list[float] | None = None,
) -> float:
    """Compute 2D hypervolume (area dominated by the frontier).

    Dependency-free 2D implementation. Callers with higher dimensions should
    use a dedicated HV library (pygmo, moarchiving, etc.). Returns area in
    the "minimize both" reference frame; maximize axes are flipped internally.

    Parameters
    ----------
    points:
        Pareto frontier points (ideally already non-dominated).
    directions:
        Two-element list of ``"minimize"`` / ``"maximize"``.
    reference:
        Reference point in the *original* coordinate system. Defaults to the
        worst observed value on each axis plus a small margin.
    """
    if len(directions) != 2:
        raise ValueError("hypervolume_2d requires exactly 2 dimensions")
    if not points:
        return 0.0

    def _to_min(p: list[float]) -> list[float]:
        return [v if d == "minimize" else -v for v, d in zip(p, directions)]

    mins = [_to_min(p) for p in points]

    if reference is None:
        worst_x = max(p[0] for p in mins)
        worst_y = max(p[1] for p in mins)
        ref = [worst_x + 1.0, worst_y + 1.0]
    else:
        ref = _to_min(reference)

    # Filter non-dominated points worse than the reference on either axis.
    mins.sort(key=lambda p: p[0])
    filtered: list[list[float]] = []
    best_y = float("inf")
    for p in mins:
        if p[0] >= ref[0] or p[1] >= ref[1]:
            continue
        if p[1] < best_y:
            filtered.append(p)
            best_y = p[1]

    if not filtered:
        return 0.0

    # Sweep x ascending: contribution of point i is (x_{i+1} - x_i) * (ref_y - y_i)
    # where x_{n} = ref_x.
    area = 0.0
    for i, (x, y) in enumerate(filtered):
        next_x = filtered[i + 1][0] if i + 1 < len(filtered) else ref[0]
        width = next_x - x
        height = ref[1] - y
        if width > 0 and height > 0:
            area += width * height
    return area
