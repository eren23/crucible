"""Export experiment results: top configs, ranked tables, markdown summaries."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from crucible.core.config import ProjectConfig
from crucible.core.io import atomic_write_json, _json_ready
from crucible.core.log import log_info, log_step, log_warn
from crucible.core.types import ExperimentResult

from crucible.analysis.results import completed_results
from crucible.analysis.leaderboard import leaderboard, sensitivity_analysis

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


def _metric_val(r: ExperimentResult, metric: str) -> float:
    return (r.get("result") or {})[metric]


def _sorted_completed(
    metric: str,
    tag: str,
    cfg: ProjectConfig | None,
) -> list[ExperimentResult]:
    """Return completed results sorted by *metric*, optionally filtered by *tag*."""
    results = completed_results(cfg)
    if tag:
        results = [r for r in results if tag in (r.get("tags") or [])]
    if not results:
        return []
    results.sort(key=lambda r: _metric_val(r, metric))
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_top_configs(
    n: int = 5,
    out_dir: str | Path = "results/winners",
    *,
    tag: str = "",
    metric: str | None = None,
    cfg: ProjectConfig | None = None,
) -> None:
    """Write top *n* experiment configs as individual JSON files.

    Parameters
    ----------
    n:
        Number of configs to export.
    out_dir:
        Directory to write JSON files into.  Created if it does not exist.
    tag:
        Optional tag filter -- only experiments carrying this tag are considered.
    metric:
        Result key to rank by.  When *None* (default), resolved from
        ``cfg.metrics.primary`` or falls back to ``"val_loss"``.
    cfg:
        Project configuration.
    """
    metric = _resolve_metric(metric, cfg)
    results = _sorted_completed(metric, tag, cfg)
    if not results:
        label = f" (tag filter: {tag})" if tag else ""
        log_warn(f"No completed experiments found{label}")
        return

    secondary = (cfg.metrics.secondary if cfg is not None else "") or ""
    top = results[:n]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if secondary:
        header = f"{'Rank':<5} {'Name':<45} {metric:>10} {secondary:>10} {'MB':>10}"
    else:
        header = f"{'Rank':<5} {'Name':<45} {metric:>10} {'MB':>10}"
    print(header)
    print("-" * len(header))

    for i, r in enumerate(top, 1):
        res = r.get("result") or {}
        mb = r.get("model_bytes") or 0
        name = r.get("name", "unnamed")
        entry = {
            "name": name,
            "rank": i,
            metric: res.get(metric),
            "model_bytes": mb,
            "config": r.get("config", {}),
        }
        if secondary:
            entry[secondary] = res.get(secondary)
        fname = out / f"{i}_{name}.json"
        atomic_write_json(fname, entry)
        mb_s = str(mb) if mb else "N/A"
        metric_v = res.get(metric)
        metric_s = f"{metric_v:.4f}" if isinstance(metric_v, (int, float)) else str(metric_v)
        if secondary:
            sec_v = res.get(secondary)
            sec_s = f"{sec_v:.4f}" if isinstance(sec_v, (int, float)) else "N/A"
            print(f"{i:<5} {name:<45} {metric_s:>10} {sec_s:>10} {mb_s:>10}")
        else:
            print(f"{i:<5} {name:<45} {metric_s:>10} {mb_s:>10}")

    log_step(f"Exported {len(top)} configs to {out}/")


def print_rank(
    n: int = 10,
    *,
    tag: str = "",
    metric: str | None = None,
    cfg: ProjectConfig | None = None,
) -> None:
    """Print a terminal-friendly ranked table of completed experiments.

    Parameters
    ----------
    n:
        Number of top results to display.
    tag:
        Optional tag filter.
    metric:
        Result key to rank by.  When *None* (default), resolved from
        ``cfg.metrics.primary`` or falls back to ``"val_loss"``.
    cfg:
        Project configuration.
    """
    metric = _resolve_metric(metric, cfg)
    secondary = (cfg.metrics.secondary if cfg is not None else "") or ""
    results = _sorted_completed(metric, tag, cfg)
    if not results:
        label = f" (tag filter: {tag})" if tag else ""
        log_warn(f"No completed experiments found{label}")
        return

    top = results[:n]
    if secondary:
        header = f"{'Rank':<5} {'Name':<45} {metric:>10} {secondary:>10} {'MB':>10}"
    else:
        header = f"{'Rank':<5} {'Name':<45} {metric:>10} {'MB':>10}"
    print(header)
    print("-" * len(header))

    for i, r in enumerate(top, 1):
        res = r.get("result") or {}
        mb = r.get("model_bytes") or 0
        mb_s = str(mb) if mb else "N/A"
        metric_v = res.get(metric)
        metric_s = f"{metric_v:.4f}" if isinstance(metric_v, (int, float)) else str(metric_v)
        if secondary:
            sec_v = res.get(secondary)
            sec_s = f"{sec_v:.4f}" if isinstance(sec_v, (int, float)) else "N/A"
            print(f"{i:<5} {r.get('name', 'unnamed'):<45} {metric_s:>10} {sec_s:>10} {mb_s:>10}")
        else:
            print(f"{i:<5} {r.get('name', 'unnamed'):<45} {metric_s:>10} {mb_s:>10}")

    print(f"\n{len(results)} completed total, showing top {len(top)}")


def generate_summary(
    *,
    top_n: int = 10,
    metric: str | None = None,
    cfg: ProjectConfig | None = None,
) -> str:
    """Produce a markdown summary of experiment results.

    Includes a top-N leaderboard table, sensitivity analysis, and the best
    experiment's config exported as shell variables.

    Parameters
    ----------
    top_n:
        Number of results to include in the leaderboard table.
    metric:
        Result key to rank by.  When *None* (default), resolved from
        ``cfg.metrics.primary`` or falls back to ``"val_loss"``.
    cfg:
        Project configuration.
    """
    metric = _resolve_metric(metric, cfg)
    secondary = (cfg.metrics.secondary if cfg is not None else "") or ""
    results = completed_results(cfg)
    if not results:
        return "No experiments completed yet.\n"

    ranked = leaderboard(results, top_n=top_n, metric=metric, cfg=cfg)
    if not ranked:
        return "No experiments completed yet.\n"

    lines: list[str] = []

    # -- Header --
    lines.append(f"## Experiment Summary ({len(results)} completed)\n")

    # -- Leaderboard table --
    lines.append(f"### Top {top_n} by {metric}\n")
    if secondary:
        lines.append(f"| Rank | Name | {metric} | {secondary} | Model Bytes |")
        lines.append("|------|------|---------|----------|-------------|")
    else:
        lines.append(f"| Rank | Name | {metric} | Model Bytes |")
        lines.append("|------|------|---------|-------------|")
    for i, r in enumerate(ranked, 1):
        res = r.get("result") or {}
        mb = r.get("model_bytes", "N/A")
        metric_v = res.get(metric)
        metric_s = f"{metric_v:.4f}" if isinstance(metric_v, (int, float)) else str(metric_v)
        if secondary:
            sec_v = res.get(secondary)
            sec_s = f"{sec_v:.4f}" if isinstance(sec_v, (int, float)) else "N/A"
            lines.append(f"| {i} | {r.get('name', 'unnamed')} | {metric_s} | {sec_s} | {mb} |")
        else:
            lines.append(f"| {i} | {r.get('name', 'unnamed')} | {metric_s} | {mb} |")

    # -- Sensitivity --
    sens = sensitivity_analysis(metric=metric, cfg=cfg)
    if sens:
        lines.append(f"\n### Sensitivity Analysis\n")
        for key, pairs in sorted(sens.items()):
            distinct = {v for v, _ in pairs}
            if len(distinct) < 2:
                continue
            best_val, best_m = pairs[0]
            worst_val, worst_m = pairs[-1]
            spread = worst_m - best_m
            lines.append(
                f"- **{key}**: spread={spread:.4f} {metric}"
                f" | best={best_val} ({best_m:.4f})"
                f" worst={worst_val} ({worst_m:.4f})"
            )

    # -- Best config export --
    if ranked:
        best = ranked[0]
        lines.append("\n### Best Config\n")
        lines.append("```bash")
        for k, v in sorted((best.get("config") or {}).items()):
            lines.append(f"export {k}={v}")
        lines.append("```")

    return "\n".join(lines) + "\n"
