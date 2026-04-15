"""Post-experiment reflection: compare predictions to outcomes, update beliefs."""
from __future__ import annotations

from crucible.researcher.llm_client import LLMClient, parse_json_from_text
from crucible.researcher.state import ResearchState


def reflect_and_update(
    state: ResearchState,
    llm: LLMClient,
    metric_key: str = "val_loss",
) -> tuple[list[str], list[str]]:
    """Post-experiment reflection. Returns (promote_names, kill_names).

    Parameters
    ----------
    metric_key:
        Primary metric key to extract from result dicts.  Callers with
        access to a ``ProjectConfig`` should pass ``config.metrics.primary``.
    """
    recent_results = state.history[-10:]
    if not recent_results:
        return [], []

    context_lines = ["Recent experiment results:"]
    for rec in recent_results:
        exp = rec.get("experiment", {})
        res = rec.get("result", {})
        name = exp.get("name", "unknown")
        primary_metric = res.get(metric_key, res.get("result", {}).get(metric_key))
        status = res.get("status", "unknown")
        metric_str = f"{primary_metric:.4f}" if isinstance(primary_metric, (int, float)) else "N/A"
        context_lines.append(f"  {name}: {metric_key}={metric_str} status={status}")

    if state.beliefs:
        context_lines.append("\nCurrent beliefs:")
        for b in state.beliefs:
            context_lines.append(f"  - {b}")

    # Show predictions vs actuals for tested hypotheses
    tested = {rec.get("experiment", {}).get("name") for rec in recent_results}
    predicted = [h for h in state.hypotheses if h.get("name") in tested]
    if predicted:
        context_lines.append("\nPredictions vs actuals:")
        for h in predicted:
            actual_rec = next(
                (rec for rec in recent_results if rec.get("experiment", {}).get("name") == h.get("name")),
                None,
            )
            if actual_rec:
                actual_metric = actual_rec.get("result", {}).get(
                    metric_key, actual_rec.get("result", {}).get("result", {}).get(metric_key)
                )
                expected = h.get("expected_impact", h.get("expected_bpb_impact", 0))
                context_lines.append(
                    f"  {h['name']}: predicted impact={expected:.4f}, "
                    f"actual metric={actual_metric if isinstance(actual_metric, (int, float)) else 'N/A'}"
                )

    system_prompt = (
        "You are reflecting on experiment results for an ML research project. "
        "Given the results and predictions, provide:\n"
        "1. Updated beliefs about what works and what does not (5-8 bullet points)\n"
        "2. Any surprising results and what they imply\n"
        "3. Which hypotheses should be promoted (run longer) or killed\n\n"
        "Return JSON with keys:\n"
        "- \"beliefs\": list of string belief statements\n"
        "- \"surprises\": list of string observations\n"
        "- \"promote\": list of experiment names to promote\n"
        "- \"kill\": list of experiment names/families to stop exploring"
    )

    response_text = llm.complete(system_prompt, "\n".join(context_lines))
    if response_text is None:
        return [], []

    parsed = parse_json_from_text(response_text)
    if parsed is None:
        return [], []

    beliefs = parsed.get("beliefs", [])
    if beliefs:
        state.update_beliefs(beliefs)
        print("  Updated beliefs:")
        for b in beliefs[:5]:
            print(f"    - {b}")

    surprises = parsed.get("surprises", [])
    if surprises:
        print("  Surprises noted:")
        for s in surprises[:3]:
            print(f"    ! {s}")

    return parsed.get("promote", []), parsed.get("kill", [])


def promote_or_kill(
    state: ResearchState,
    promote_names: list[str],
    kill_names: list[str],
    current_tier: str,
) -> None:
    """Auto-promote winners to next tier, kill dead ends."""
    tier_order = ["smoke", "proxy", "medium", "promotion"]
    current_idx = tier_order.index(current_tier) if current_tier in tier_order else 1
    next_tier = tier_order[min(current_idx + 1, len(tier_order) - 1)]

    if promote_names:
        print(f"  Promoting to {next_tier}: {', '.join(promote_names[:5])}")
        for name in promote_names:
            state.mark_hypothesis(name, "promoted")
            matching = [
                rec
                for rec in state.history
                if rec.get("experiment", {}).get("name") == name
                and rec.get("result", {}).get("status") == "completed"
            ]
            if matching:
                exp = matching[-1]["experiment"]
                promoted_hyp = {
                    "hypothesis": f"Promoted from {current_tier}: {name} showed promise, testing at {next_tier} tier",
                    "name": f"{name}_{next_tier}",
                    "config": exp.get("config", {}),
                    "expected_impact": 0.01,
                    "confidence": 0.7,
                    "family": exp.get("config", {}).get("MODEL_FAMILY", "unknown"),
                    "rationale": f"Promotion from {current_tier} tier based on strong results.",
                    "tier": next_tier,
                    "parent_run_id": matching[-1].get("result", {}).get("id"),
                }
                state.add_hypothesis(promoted_hyp)

    if kill_names:
        print(f"  Killing: {', '.join(kill_names[:5])}")
        for name in kill_names:
            state.mark_hypothesis(name, "killed")
