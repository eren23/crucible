"""Hypothesis generation using LLM-driven analysis."""
from __future__ import annotations

from typing import Any

from crucible.researcher.llm_client import LLMClient, parse_json_from_text
from crucible.researcher.state import ResearchState


def generate_hypotheses(
    context: str,
    program_text: str,
    state: ResearchState,
    llm: LLMClient,
    iteration: int,
) -> list[dict[str, Any]]:
    """Use an LLM to generate ranked experiment hypotheses from the current research state."""

    system_prompt = (
        "You are an autonomous ML research agent. "
        "Your goal is to minimize the primary metric (e.g., validation loss or bits-per-byte) "
        "for a model within the given constraints.\n\n"
        "Given the current research state and results, propose 3-5 experiment hypotheses "
        "ranked by expected impact. Each hypothesis should test ONE change at a time "
        "unless you have strong evidence a combination helps.\n\n"
        "Return a JSON object with a single key \"hypotheses\" containing a list of objects, "
        "each with:\n"
        "- \"hypothesis\": string describing what you expect\n"
        "- \"name\": short experiment name (no spaces, lowercase + underscores)\n"
        "- \"expected_impact\": float, expected improvement (positive = better)\n"
        "- \"confidence\": float 0-1, your confidence this will help\n"
        "- \"config\": dict of env var overrides (all values must be strings)\n"
        "- \"rationale\": 1-2 sentence explanation\n"
        "- \"family\": which model family this tests\n\n"
        "IMPORTANT: All config values must be strings. The config dict contains environment "
        "variable overrides like {\"MODEL_FAMILY\": \"looped\", \"RECURRENCE_STEPS\": \"12\"}.\n\n"
        "Focus on the research priorities in the program document. "
        "Mix exploitation (refine what works) with exploration (test new ideas)."
    )

    user_prompt = f"{program_text}\n\n---\n\n{context}"

    response_text = llm.complete(system_prompt, user_prompt)
    if response_text is None:
        return []

    hypotheses = _parse_hypotheses(response_text, iteration)
    for hyp in hypotheses:
        state.add_hypothesis(hyp)

    print(f"  Generated {len(hypotheses)} hypotheses:")
    for h in hypotheses:
        print(
            f"    - {h.get('name', '?')}: {h.get('hypothesis', '?')[:80]}  "
            f"(impact={h.get('expected_impact', 0):.4f}, conf={h.get('confidence', 0):.2f})"
        )

    return hypotheses


def _parse_hypotheses(text: str, iteration: int) -> list[dict[str, Any]]:
    """Extract hypotheses list from LLM response text."""
    parsed = parse_json_from_text(text)
    if parsed is None:
        return []
    hypotheses = parsed.get("hypotheses", [])
    if not isinstance(hypotheses, list):
        return []
    valid = []
    for h in hypotheses:
        if not isinstance(h, dict):
            continue
        if not h.get("config"):
            continue
        config = h.get("config", {})
        h["config"] = {k: str(v) for k, v in config.items()}
        h.setdefault("name", f"hyp_{iteration}_{len(valid)}")
        h.setdefault("hypothesis", h.get("rationale", "No hypothesis stated"))
        h.setdefault("expected_impact", h.get("expected_bpb_impact", 0.0))
        h.setdefault("confidence", 0.5)
        h.setdefault("family", config.get("MODEL_FAMILY", "unknown"))
        valid.append(h)
    return valid
