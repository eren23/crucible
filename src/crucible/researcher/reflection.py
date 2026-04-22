"""Post-experiment reflection: compare predictions to outcomes, update beliefs."""
from __future__ import annotations

from crucible.researcher.llm_client import LLMClient, parse_json_from_text
from crucible.researcher.state import ResearchState


REFLECTION_SYSTEM_PROMPT = (
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


def build_reflection_prompt(
    state: ResearchState,
    metric_key: str = "val_loss",
) -> tuple[str, str] | None:
    """Assemble system + user prompts for post-experiment reflection.

    Pure — no LLM call, no state mutation. Returns ``(system, user)`` or
    ``None`` if there is nothing to reflect on yet (empty history).
    """
    recent_results = state.history[-10:]
    if not recent_results:
        return None

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

    return REFLECTION_SYSTEM_PROMPT, "\n".join(context_lines)


def parse_reflection(text: str) -> dict[str, list] | None:
    """Parse a reflection LLM response into ``{beliefs, surprises, promote, kill}``.

    Returns ``None`` if the response is not a valid object. Missing keys
    default to empty lists.
    """
    parsed = parse_json_from_text(text)
    if parsed is None:
        return None
    return {
        "beliefs": parsed.get("beliefs", []) or [],
        "surprises": parsed.get("surprises", []) or [],
        "promote": parsed.get("promote", []) or [],
        "kill": parsed.get("kill", []) or [],
    }


def apply_reflection(state: ResearchState, parsed: dict[str, list]) -> dict[str, int]:
    """Apply a parsed reflection to ``state``. Returns counts summary."""
    beliefs = list(parsed.get("beliefs", []))
    if beliefs:
        state.update_beliefs(beliefs)
    return {
        "beliefs_updated": len(beliefs),
        "surprises": len(parsed.get("surprises", [])),
        "promote": len(parsed.get("promote", [])),
        "kill": len(parsed.get("kill", [])),
    }


def reflect_and_update(
    state: ResearchState,
    llm: LLMClient,
    metric_key: str = "val_loss",
) -> tuple[list[str], list[str]]:
    """Post-experiment reflection. Returns (promote_names, kill_names).

    Thin wrapper around the pure helpers — builds the prompt, calls the
    LLM, parses, applies beliefs, prints summary. Existing autonomous-
    mode callers see no behaviour change.
    """
    prompts = build_reflection_prompt(state, metric_key=metric_key)
    if prompts is None:
        return [], []
    system_prompt, user_prompt = prompts

    response_text = llm.complete(system_prompt, user_prompt)
    if response_text is None:
        return [], []

    parsed = parse_reflection(response_text)
    if parsed is None:
        return [], []

    apply_reflection(state, parsed)

    if parsed["beliefs"]:
        print("  Updated beliefs:")
        for b in parsed["beliefs"][:5]:
            print(f"    - {b}")
    if parsed["surprises"]:
        print("  Surprises noted:")
        for s in parsed["surprises"][:3]:
            print(f"    ! {s}")

    return parsed["promote"], parsed["kill"]


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
