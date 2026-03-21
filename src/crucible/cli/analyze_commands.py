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

        results = completed_results(config)
        top = leaderboard(results, top_n=getattr(args, "top", 20))
        if not top:
            print("No completed experiments.")
            return
        print(f"{'Rank':>4}  {'Name':<40}  {'val_bpb':>8}  {'val_loss':>8}  {'bytes':>10}")
        print("-" * 80)
        for i, r in enumerate(top, 1):
            res = r.get("result", {})
            bpb = res.get("val_bpb", "N/A")
            loss = res.get("val_loss", "N/A")
            bpb_str = f"{bpb:.4f}" if isinstance(bpb, (int, float)) else str(bpb)
            loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else str(loss)
            print(f"{i:>4}  {r.get('name', '?'):<40}  {bpb_str:>8}  {loss_str:>8}  {r.get('model_bytes', 'N/A'):>10}")

    elif cmd == "sensitivity":
        from crucible.analysis.leaderboard import sensitivity_analysis
        from crucible.analysis.results import completed_results

        results = completed_results(config)
        sens = sensitivity_analysis(results)
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
        from crucible.analysis.results import completed_results

        results = completed_results(config)
        frontier = pareto_frontier(results)
        if not frontier:
            print("No data for Pareto frontier.")
            return
        print(f"{'Name':<40}  {'val_bpb':>8}  {'bytes':>10}")
        print("-" * 60)
        for r in frontier:
            res = r.get("result", {})
            bpb = res.get("val_bpb", "N/A")
            bpb_str = f"{bpb:.4f}" if isinstance(bpb, (int, float)) else str(bpb)
            print(f"{r.get('name', '?'):<40}  {bpb_str:>8}  {r.get('model_bytes', 'N/A'):>10}")

    elif cmd == "export":
        from crucible.analysis.export import export_top_configs
        from crucible.analysis.results import completed_results

        results = completed_results(config)
        out_dir = getattr(args, "out", "exported_configs")
        exported = export_top_configs(results, top_n=getattr(args, "top", 5), out_dir=out_dir)
        print(f"Exported {exported} configs to {out_dir}/")

    elif cmd == "summary":
        from crucible.analysis.export import generate_summary
        from crucible.analysis.results import completed_results

        results = completed_results(config)
        print(generate_summary(results))

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
