"""Convert hypotheses into executable experiment batches."""
from __future__ import annotations

from typing import Any

from crucible.researcher.state import ResearchState

# Default tier cost estimates (compute-hours per run)
DEFAULT_TIER_COSTS: dict[str, float] = {
    "smoke": 0.02,
    "proxy": 0.5,
    "medium": 1.0,
    "promotion": 2.0,
}


def design_batch(
    hypotheses: list[dict[str, Any]],
    state: ResearchState,
    tier: str,
    backend: str,
    iteration: int,
    tier_costs: dict[str, float] | None = None,
    baseline_config: dict[str, str] | None = None,
    baseline_name: str = "baseline_control",
) -> list[dict[str, Any]]:
    """Convert hypotheses to experiment configs.

    Always includes a baseline control. Mixes exploitation and exploration.
    Respects budget constraints.
    """
    costs = tier_costs or DEFAULT_TIER_COSTS
    batch: list[dict[str, Any]] = []
    tier_cost = costs.get(tier, 0.5)

    # Include baseline control if not run recently
    recent_names = {rec.get("experiment", {}).get("name") for rec in state.history[-20:]}
    if baseline_name not in recent_names and baseline_config:
        baseline_cfg = dict(baseline_config)
        # Auto-derive a distinguishable W&B run name unless caller already set one.
        # Default WANDB_RUN_NAME=exp_id collides across batch members in the W&B UI.
        baseline_cfg.setdefault("CRUCIBLE_VARIANT_NAME", f"{baseline_name}_iter_{iteration}")
        batch.append(
            {
                "name": baseline_name,
                "config": baseline_cfg,
                "tags": ["autonomous", "baseline", f"iter_{iteration}"],
                "tier": tier,
                "backend": backend,
                "priority": 0,
                "wave": f"auto_iter_{iteration}",
                "rationale": "Baseline control for comparison.",
            }
        )

    # Add hypotheses, respecting budget
    for hyp in hypotheses:
        if state.budget_remaining < tier_cost * (len(batch) + 1):
            print("  Budget limit reached, skipping remaining hypotheses.")
            break
        config = hyp.get("config", {})
        if not config:
            continue
        hyp_name = hyp.get("name", f"hyp_{iteration}")
        config_with_name = dict(config)
        # Same auto-derivation as the baseline path.
        config_with_name.setdefault("CRUCIBLE_VARIANT_NAME", f"{hyp_name}_iter_{iteration}")
        batch.append(
            {
                "name": hyp_name,
                "config": config_with_name,
                "tags": ["autonomous", hyp.get("family", "unknown"), f"iter_{iteration}"],
                "tier": tier,
                "backend": backend,
                "priority": int(hyp.get("confidence", 0.5) * 100),
                "wave": f"auto_iter_{iteration}",
                "rationale": hyp.get("rationale", ""),
                "hypothesis": hyp.get("hypothesis", ""),
            }
        )

    print(f"  Designed batch of {len(batch)} experiments (est. {len(batch) * tier_cost:.2f} compute-hours)")
    return batch
