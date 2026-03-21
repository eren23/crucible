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

_DEFAULT_METRIC = "val_bpb"


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
    metric: str = _DEFAULT_METRIC,
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
        Result key to rank by (default ``val_bpb``).
    cfg:
        Project configuration.
    """
    results = _sorted_completed(metric, tag, cfg)
    if not results:
        label = f" (tag filter: {tag})" if tag else ""
        log_warn(f"No completed experiments found{label}")
        return

    top = results[:n]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    header = f"{'Rank':<5} {'Name':<45} {metric:>10} {'val_loss':>10} {'MB':>10}"
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
            "val_loss": res.get("val_loss"),
            "model_bytes": mb,
            "config": r.get("config", {}),
        }
        fname = out / f"{i}_{name}.json"
        atomic_write_json(fname, entry)
        mb_s = str(mb) if mb else "N/A"
        metric_v = res.get(metric)
        metric_s = f"{metric_v:.4f}" if isinstance(metric_v, (int, float)) else str(metric_v)
        val_loss = res.get("val_loss")
        loss_s = f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else "N/A"
        print(f"{i:<5} {name:<45} {metric_s:>10} {loss_s:>10} {mb_s:>10}")

    log_step(f"Exported {len(top)} configs to {out}/")


def print_rank(
    n: int = 10,
    *,
    tag: str = "",
    metric: str = _DEFAULT_METRIC,
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
        Result key to rank by (default ``val_bpb``).
    cfg:
        Project configuration.
    """
    results = _sorted_completed(metric, tag, cfg)
    if not results:
        label = f" (tag filter: {tag})" if tag else ""
        log_warn(f"No completed experiments found{label}")
        return

    top = results[:n]
    header = f"{'Rank':<5} {'Name':<45} {metric:>10} {'val_loss':>10} {'MB':>10}"
    print(header)
    print("-" * len(header))

    for i, r in enumerate(top, 1):
        res = r.get("result") or {}
        mb = r.get("model_bytes") or 0
        mb_s = str(mb) if mb else "N/A"
        metric_v = res.get(metric)
        metric_s = f"{metric_v:.4f}" if isinstance(metric_v, (int, float)) else str(metric_v)
        val_loss = res.get("val_loss")
        loss_s = f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else "N/A"
        print(f"{i:<5} {r.get('name', 'unnamed'):<45} {metric_s:>10} {loss_s:>10} {mb_s:>10}")

    print(f"\n{len(results)} completed total, showing top {len(top)}")


def generate_summary(
    *,
    top_n: int = 10,
    metric: str = _DEFAULT_METRIC,
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
        Result key to rank by (default ``val_bpb``).
    cfg:
        Project configuration.
    """
    results = completed_results(cfg)
    if not results:
        return "No experiments completed yet.\n"

    ranked = leaderboard(top_n, metric=metric, cfg=cfg)
    if not ranked:
        return "No experiments completed yet.\n"

    lines: list[str] = []

    # -- Header --
    lines.append(f"## Experiment Summary ({len(results)} completed)\n")

    # -- Leaderboard table --
    lines.append(f"### Top {top_n} by {metric}\n")
    lines.append(f"| Rank | Name | {metric} | val_loss | Model Bytes |")
    lines.append("|------|------|---------|----------|-------------|")
    for i, r in enumerate(ranked, 1):
        res = r.get("result") or {}
        mb = r.get("model_bytes", "N/A")
        metric_v = res.get(metric)
        metric_s = f"{metric_v:.4f}" if isinstance(metric_v, (int, float)) else str(metric_v)
        val_loss = res.get("val_loss")
        loss_s = f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else "N/A"
        lines.append(f"| {i} | {r.get('name', 'unnamed')} | {metric_s} | {loss_s} | {mb} |")

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
