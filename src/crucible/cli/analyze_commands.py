"""CLI handlers for analysis and model commands."""
from __future__ import annotations

import argparse
import sys

from crucible.core.config import load_config


def handle_analyze(args: argparse.Namespace) -> None:
    config = load_config()
    cmd = getattr(args, "analyze_command", None)

    if cmd == "rank":
        from crucible.analysis.leaderboard import leaderboard
        from crucible.analysis.results import completed_results

        primary = config.metrics.primary
        secondary = config.metrics.secondary or ""
        results = completed_results(config)
        top = leaderboard(results, top_n=getattr(args, "top", 20), cfg=config)
        if not top:
            print("No completed experiments.")
            return
        if secondary:
            print(f"{'Rank':>4}  {'Name':<40}  {primary:>8}  {secondary:>8}  {'bytes':>10}")
        else:
            print(f"{'Rank':>4}  {'Name':<40}  {primary:>8}  {'bytes':>10}")
        print("-" * 80)
        for i, r in enumerate(top, 1):
            res = r.get("result", {})
            pri = res.get(primary, "N/A")
            pri_str = f"{pri:.4f}" if isinstance(pri, (int, float)) else str(pri)
            if secondary:
                sec = res.get(secondary, "N/A")
                sec_str = f"{sec:.4f}" if isinstance(sec, (int, float)) else str(sec)
                print(f"{i:>4}  {r.get('name', '?'):<40}  {pri_str:>8}  {sec_str:>8}  {r.get('model_bytes', 'N/A'):>10}")
            else:
                print(f"{i:>4}  {r.get('name', '?'):<40}  {pri_str:>8}  {r.get('model_bytes', 'N/A'):>10}")

    elif cmd == "sensitivity":
        from crucible.analysis.leaderboard import sensitivity_analysis
        from crucible.analysis.results import completed_results

        results = completed_results(config)
        sens = sensitivity_analysis(results, cfg=config)
        if not sens:
            print("Not enough data for sensitivity analysis.")
            return
        ranked = sorted(sens.items(), key=lambda kv: kv[1][-1][1] - kv[1][0][1], reverse=True)
        for key, pairs in ranked[:15]:
            best_val, best_metric = pairs[0]
            worst_val, worst_metric = pairs[-1]
            spread = worst_metric - best_metric
            print(f"  {key}: spread={spread:.4f}  best={best_val}({best_metric:.4f})  worst={worst_val}({worst_metric:.4f})")

    elif cmd == "pareto":
        from crucible.analysis.leaderboard import pareto_frontier

        primary = config.metrics.primary
        frontier = pareto_frontier(cfg=config)
        if not frontier:
            print("No data for Pareto frontier.")
            return
        print(f"{'Name':<40}  {primary:>8}  {'bytes':>10}")
        print("-" * 60)
        for r in frontier:
            res = r.get("result", {})
            pri = res.get(primary, "N/A")
            pri_str = f"{pri:.4f}" if isinstance(pri, (int, float)) else str(pri)
            print(f"{r.get('name', '?'):<40}  {pri_str:>8}  {r.get('model_bytes', 'N/A'):>10}")

    elif cmd == "export":
        from crucible.analysis.export import export_top_configs

        out_dir = getattr(args, "out", "exported_configs")
        export_top_configs(n=getattr(args, "top", 5), out_dir=out_dir, cfg=config)

    elif cmd == "summary":
        from crucible.analysis.export import generate_summary

        print(generate_summary(cfg=config))

    else:
        print("Usage: crucible analyze {rank|sensitivity|pareto|export|summary}", file=sys.stderr)


def handle_models(args: argparse.Namespace) -> None:
    cmd = getattr(args, "models_command", None)

    if cmd == "list":
        try:
            from crucible.models.registry import list_families

            families = list_families()
            if not families:
                print("No model families registered. Install crucible[torch] to enable the model zoo.")
                return
            print("Registered model families:")
            for f in families:
                print(f"  - {f}")
        except ImportError:
            print("Model zoo requires PyTorch. Install with: pip install crucible-ml[torch]")
    else:
        print("Usage: crucible models {list}", file=sys.stderr)
