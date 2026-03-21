"""Standalone analysis builder for experiment results.

Extracted from AutonomousResearcher.analyze() so that MCP tools and
other consumers can build the same rich analysis context without
instantiating the full researcher loop.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from crucible.core.config import ProjectConfig
from crucible.researcher.state import ResearchState


def build_analysis(config: ProjectConfig, state: ResearchState) -> str:
    """Build structured analysis of experiment results for LLM context.

    Returns a markdown-formatted string with leaderboard, family breakdown,
    sensitivity analysis, diminishing-returns detection, and research state.
    """
    try:
        from crucible.analysis.leaderboard import leaderboard, sensitivity_analysis
        from crucible.analysis.results import completed_results
    except ImportError:
        return "Analysis module not available. This is the first iteration."

    results = completed_results(config)
    if not results:
        return "No completed experiments yet. This is the first iteration."

    sections: list[str] = []
    primary = config.metrics.primary
    secondary = config.metrics.secondary or ""

    # Overall leaderboard
    top = leaderboard(results, top_n=10, cfg=config)
    board_lines = ["## Leaderboard (top 10)"]
    for i, r in enumerate(top, 1):
        res = r["result"]
        parts = [f"{primary}={res.get(primary, 'N/A')}"]
        if secondary:
            parts.append(f"{secondary}={res.get(secondary, 'N/A')}")
        parts.append(f"bytes={r.get('model_bytes', 'N/A')}")
        board_lines.append(f"  {i}. {r['name']}: {'  '.join(parts)}")
    sections.append("\n".join(board_lines))

    # Group by model family
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        family = r.get("config", {}).get("MODEL_FAMILY", "unknown")
        families[family].append(r)
    family_lines = ["## Results by Model Family"]
    for family, runs in sorted(families.items()):
        metrics = [r["result"].get(primary) for r in runs if r.get("result", {}).get(primary)]
        if metrics:
            best = min(metrics)
            worst = max(metrics)
            family_lines.append(
                f"  {family}: {len(runs)} runs, best={best:.4f}, worst={worst:.4f}, spread={worst - best:.4f}"
            )
    sections.append("\n".join(family_lines))

    # Sensitivity analysis
    sens = sensitivity_analysis(results, cfg=config)
    if sens:
        sens_lines = ["## Sensitivity Analysis (top parameters by spread)"]
        ranked_sens = sorted(sens.items(), key=lambda kv: kv[1][-1][1] - kv[1][0][1], reverse=True)
        for key, pairs in ranked_sens[:10]:
            best_val, best_metric = pairs[0]
            worst_val, worst_metric = pairs[-1]
            spread = worst_metric - best_metric
            sens_lines.append(
                f"  {key}: spread={spread:.4f}  best={best_val}({best_metric:.4f})  "
                f"worst={worst_val}({worst_metric:.4f})"
            )
        sections.append("\n".join(sens_lines))

    # Detect diminishing returns
    recent = state.history[-5:]
    if len(recent) >= 3:
        recent_metrics = []
        for rec in recent:
            metric_val = rec.get("result", {}).get(primary)
            if isinstance(metric_val, (int, float)):
                recent_metrics.append(metric_val)
        if len(recent_metrics) >= 3 and top:
            recent_best = min(recent_metrics)
            overall_best = top[0]["result"].get(primary, recent_best)
            if isinstance(overall_best, (int, float)) and abs(recent_best - overall_best) < 0.001:
                sections.append(
                    "## Warning: Diminishing Returns\n"
                    "  Last 5 experiments have not improved on the overall best by >0.001.\n"
                    "  Consider exploring a different model family or hyperparameter axis."
                )

    # Research state context
    sections.append(f"## Research State\n{state.get_history_summary(primary_metric=primary)}")
    if state.beliefs:
        beliefs_str = "\n".join(f"  - {b}" for b in state.beliefs)
        sections.append(f"## Current Beliefs\n{beliefs_str}")

    # Findings (if any)
    findings = state.get_findings()
    if findings:
        findings_lines = ["## Research Findings"]
        for f in findings[-10:]:
            findings_lines.append(f"  - [{f.get('category', 'observation')}] {f['finding']}")
        sections.append("\n".join(findings_lines))

    return "\n\n".join(sections)


def build_analysis_structured(config: ProjectConfig, state: ResearchState) -> dict[str, Any]:
    """Build analysis returning both markdown and structured data."""
    markdown = build_analysis(config, state)

    structured: dict[str, Any] = {"markdown": markdown}
    try:
        from crucible.analysis.leaderboard import leaderboard, sensitivity_analysis
        from crucible.analysis.results import completed_results

        results = completed_results(config)
        top = leaderboard(results, top_n=10, cfg=config)
        sens = sensitivity_analysis(results, cfg=config)
        structured["top_experiments"] = [
            {"name": r["name"], "metric": r.get("result", {}).get(config.metrics.primary), "model_bytes": r.get("model_bytes")}
            for r in top
        ]
        structured["sensitivity"] = {
            k: {"spread": pairs[-1][1] - pairs[0][1], "best": pairs[0], "worst": pairs[-1]}
            for k, pairs in sens.items()
        } if sens else {}
        structured["total_completed"] = len(results)
    except ImportError:
        structured["top_experiments"] = []
        structured["sensitivity"] = {}
        structured["total_completed"] = 0

    structured["beliefs"] = list(state.beliefs)
    structured["budget_remaining"] = state.budget_remaining
    structured["findings"] = state.get_findings()

    return structured
