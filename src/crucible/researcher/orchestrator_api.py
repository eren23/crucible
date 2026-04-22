"""Orchestrator-driven research loop — zero LLM keys in Crucible.

Crucible is infrastructure: pods, fleet, experiments, data, notes,
findings, plan, search. Taste (hypothesis generation, reflection,
planning) is supplied by an **external orchestrator** — Claude Code,
another agent, or a human via MCP. The orchestrator supplies its own
LLM.

This module exposes two pure primitives consumed by the two new MCP
tools ``research_request_prompt`` and ``research_submit``:

- :func:`request_prompt(stage, config, state, ...)` — build the
  system+user prompts Crucible would have sent to its own LLM, plus the
  JSON schema the response must match. No LLM call.
- :func:`submit_response(stage, response, config, state)` — parse the
  orchestrator-supplied response and apply it to the research state.
  No LLM call.

Stages:

- ``"hypothesis"`` — propose N experiment hypotheses.
- ``"reflection"`` — digest recent results, update beliefs, pick
  promote/kill lists.
- ``"briefing"`` — human-readable markdown summary of current project
  state (no schema, no submit counterpart — a read-only status probe).

The existing autonomous mode (``AutonomousResearcher`` + legacy
``generate_hypotheses`` / ``reflect_and_update``) is untouched; both
paths share the same prompt-builder / parser / applier helpers in
:mod:`crucible.researcher.hypothesis` and
:mod:`crucible.researcher.reflection`.
"""
from __future__ import annotations

import json
from typing import Any, Literal

from crucible.core.config import ProjectConfig
from crucible.core.errors import ResearcherError
from crucible.researcher.analysis import build_analysis
from crucible.researcher.briefing import build_briefing
from crucible.researcher.hypothesis import (
    HYPOTHESIS_SYSTEM_PROMPT,
    apply_hypotheses,
    build_hypothesis_prompt,
    parse_hypotheses,
    _validate_hypotheses,
)
from crucible.researcher.reflection import (
    apply_reflection,
    build_reflection_prompt,
    parse_reflection,
)
from crucible.researcher.state import ResearchState


Stage = Literal["hypothesis", "reflection", "briefing"]
_VALID_STAGES: tuple[Stage, ...] = ("hypothesis", "reflection", "briefing")


# ---------------------------------------------------------------------------
# JSON Schemas
# ---------------------------------------------------------------------------

HYPOTHESIS_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["hypotheses"],
    "properties": {
        "hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["hypothesis", "config"],
                "properties": {
                    "hypothesis": {"type": "string"},
                    "name": {"type": "string", "description": "lowercase + underscores, no spaces"},
                    "expected_impact": {"type": "number"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "config": {
                        "type": "object",
                        "description": "Env var overrides. All VALUES must be strings.",
                        "additionalProperties": {"type": "string"},
                    },
                    "rationale": {"type": "string"},
                    "family": {"type": "string"},
                },
            },
        },
    },
}


REFLECTION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "beliefs": {
            "type": "array",
            "items": {"type": "string"},
            "description": "5-8 belief statements about what works / does not",
        },
        "surprises": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Notable unexpected observations",
        },
        "promote": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Experiment names to promote to the next tier",
        },
        "kill": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Experiment names or families to stop exploring",
        },
    },
}


# ---------------------------------------------------------------------------
# request_prompt
# ---------------------------------------------------------------------------


def request_prompt(
    stage: Stage,
    config: ProjectConfig,
    state: ResearchState,
    *,
    focus_family: str = "",
    extra_context: str = "",
    literature_context: str = "",
    iteration: int = 0,
) -> dict[str, Any]:
    """Build the orchestrator-facing prompt + schema for *stage*.

    Returns ``{stage, system, user, schema, state_snapshot}``. The
    ``state_snapshot`` is an opaque marker (iteration counter + history
    length) the orchestrator can pass back in ``submit_response`` for
    sanity checking — not currently enforced but reserved.
    """
    if stage not in _VALID_STAGES:
        raise ResearcherError(f"Unknown stage {stage!r}. Valid: {_VALID_STAGES}")

    snapshot = {
        "iteration": iteration,
        "history_len": len(state.history),
        "hypotheses_len": len(state.hypotheses),
        "beliefs_len": len(state.beliefs),
    }

    if stage == "briefing":
        briefing = build_briefing(config)
        md = briefing.get("markdown_summary") or json.dumps(briefing, indent=2, default=str)
        return {
            "stage": stage,
            "system": None,
            "user": md,
            "schema": None,
            "state_snapshot": snapshot,
        }

    if stage == "hypothesis":
        analysis = build_analysis(config, state)
        if focus_family:
            analysis += f"\n\n## Focus Area\nFocus on the '{focus_family}' model family."
        if extra_context:
            analysis += f"\n\n## Additional Context\n{extra_context}"

        program_path = config.project_root / config.researcher.program_file
        program_text = program_path.read_text(encoding="utf-8") if program_path.exists() else ""

        system, user = build_hypothesis_prompt(
            analysis, program_text, literature_context=literature_context
        )
        return {
            "stage": stage,
            "system": system,
            "user": user,
            "schema": HYPOTHESIS_RESPONSE_SCHEMA,
            "state_snapshot": snapshot,
        }

    # stage == "reflection"
    prompts = build_reflection_prompt(state, metric_key=config.metrics.primary)
    if prompts is None:
        return {
            "stage": stage,
            "system": None,
            "user": (
                "## Reflection unavailable\n\n"
                "No completed experiments yet (state.history is empty). "
                "Run experiments and collect results, then request reflection again."
            ),
            "schema": None,
            "state_snapshot": snapshot,
        }
    system, user = prompts
    return {
        "stage": stage,
        "system": system,
        "user": user,
        "schema": REFLECTION_RESPONSE_SCHEMA,
        "state_snapshot": snapshot,
    }


# ---------------------------------------------------------------------------
# submit_response
# ---------------------------------------------------------------------------


def submit_response(
    stage: Stage,
    response: dict[str, Any] | str,
    config: ProjectConfig,  # noqa: ARG001  # reserved for future stage-scoped validation
    state: ResearchState,
    *,
    iteration: int = 0,
) -> dict[str, Any]:
    """Parse + apply an orchestrator-supplied response to *state*.

    *response* may be either a parsed dict (matching the stage's schema)
    or a raw string — JSON blobs and code-fenced JSON are both accepted
    via :func:`parse_json_from_text`.

    Returns a summary dict describing what was applied.
    """
    if stage not in _VALID_STAGES:
        raise ResearcherError(f"Unknown stage {stage!r}. Valid: {_VALID_STAGES}")

    if stage == "briefing":
        raise ResearcherError(
            "briefing stage is read-only — no submit needed (it only returns state markdown)."
        )

    if stage == "hypothesis":
        raw_list = _extract_hypotheses(response)
        hypotheses = _validate_hypotheses(raw_list, iteration)
        applied = apply_hypotheses(state, hypotheses)
        state.save()
        return {
            "applied": applied > 0,
            "hypotheses_added": applied,
            "summary": f"Added {applied} hypothesis item(s) to state.",
        }

    # stage == "reflection"
    parsed = _extract_reflection(response)
    if parsed is None:
        raise ResearcherError(
            "reflection submit: response is not a valid reflection object "
            "(expected {beliefs, surprises, promote, kill} — all optional lists)."
        )
    counts = apply_reflection(state, parsed)
    state.save()
    counts["applied"] = sum(counts.values()) > 0
    counts["summary"] = (
        f"Reflection applied: {counts['beliefs_updated']} beliefs, "
        f"{counts['promote']} promote, {counts['kill']} kill, "
        f"{counts['surprises']} surprises."
    )
    # Callers may still want the raw promote/kill lists to drive fleet tools:
    counts["promote_names"] = list(parsed.get("promote", []))
    counts["kill_names"] = list(parsed.get("kill", []))
    return counts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_hypotheses(response: Any) -> Any:
    """Accept a dict, a list, or a JSON string; return the inner hypothesis list."""
    if isinstance(response, list):
        return response
    if isinstance(response, dict):
        return response.get("hypotheses", response.get("items", []))
    if isinstance(response, str):
        parsed = parse_hypotheses(response, iteration=0)
        return parsed
    return []


def _extract_reflection(response: Any) -> dict[str, list] | None:
    if isinstance(response, dict):
        return {
            "beliefs": response.get("beliefs", []) or [],
            "surprises": response.get("surprises", []) or [],
            "promote": response.get("promote", []) or [],
            "kill": response.get("kill", []) or [],
        }
    if isinstance(response, str):
        return parse_reflection(response)
    return None
