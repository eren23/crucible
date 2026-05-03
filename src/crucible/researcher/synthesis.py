"""Findings-pair synthesis — GIANTS-style hypothesis seeding from the hub graph.

GIANTS (https://giants-insights.github.io/) trains a model to predict a
downstream paper's contribution from two parent papers it cited. We apply
the same shape to Crucible's cross-project findings ledger: mine pairs of
findings, ask the orchestrator's LLM to predict the experiment that
synthesizes both, feed the result into the existing
``design_batch_from_hypotheses`` → ``design_enqueue_batch`` pipeline.

Pure orchestrator-contract: no internal LLM call. ``build_synthesis_prompt``
returns ``{system, user, schema, parent_finding_ids}`` exactly like
``orchestrator_api.request_prompt``.
"""
from __future__ import annotations

import itertools
import random
from typing import Any

from crucible.core.errors import ResearcherError
from crucible.core.redact import redact_secrets
from crucible.researcher.hypothesis import _validate_hypotheses
from crucible.researcher.llm_client import parse_json_from_text


Pair = tuple[dict[str, Any], dict[str, Any]]


SYNTHESIS_SYSTEM_PROMPT = (
    "You are an autonomous ML research agent. You will be given two findings "
    "from a research-findings ledger — durable observations that each won on "
    "their own project or track. Your job is to propose 1-3 NEW experiment "
    "hypotheses that synergistically COMBINE both parents.\n\n"
    "A good synthesis is not a concatenation. It identifies the underlying "
    "mechanism each parent exploits and proposes an experiment whose outcome "
    "is informative whether the synthesis works or fails.\n\n"
    "Return a JSON object with key \"hypotheses\" (list). Each item has:\n"
    "  - name: short experiment name (lowercase + underscores, no spaces)\n"
    "  - hypothesis: one sentence stating what you expect\n"
    "  - config: dict of env-var overrides (ALL values must be strings)\n"
    "  - rationale: 1-2 sentence explanation of WHY this synthesis matters\n"
    "  - confidence: float 0-1\n"
    "  - expected_impact: float, expected improvement on primary metric\n"
)


SYNTHESIS_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["hypotheses"],
    "properties": {
        "hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["hypothesis", "config"],
                "properties": {
                    "name": {"type": "string"},
                    "hypothesis": {"type": "string"},
                    "config": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                    "rationale": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "expected_impact": {"type": "number"},
                },
            },
        },
    },
}


_VALID_POLICIES = ("random", "same_track", "cross_track")


def mine_pairs(
    findings: list[dict[str, Any]],
    *,
    k: int,
    policy: str = "random",
    seed: int | None = None,
    required_tags: set[str] | None = None,
) -> list[Pair]:
    """Sample ``k`` unordered finding pairs from ``findings`` per ``policy``.

    Policies:
      - ``random``: any unique unordered pair.
      - ``same_track``: both findings share a non-empty ``track``.
      - ``cross_track``: findings have different non-empty tracks.

    ``required_tags`` (optional) applies an OR filter at the pair level:
    a pair is eligible iff at least one finding carries at least one of
    the required tags. An empty / None set disables the filter.

    Returns at most ``k`` pairs; fewer if the eligible-pair pool is smaller.
    Empty list is a valid return when no pairs satisfy ``policy``.
    """
    if policy not in _VALID_POLICIES:
        raise ResearcherError(
            f"Unknown pair-mining policy {policy!r}. Valid: {_VALID_POLICIES}"
        )
    if len(findings) < 2:
        raise ResearcherError(
            "mine_pairs requires at least 2 findings; "
            f"got {len(findings)}"
        )

    tag_filter = set(required_tags) if required_tags else set()

    eligible: list[Pair] = []
    for a, b in itertools.combinations(findings, 2):
        if tag_filter:
            tags_a = set(a.get("tags") or [])
            tags_b = set(b.get("tags") or [])
            if not (tag_filter & (tags_a | tags_b)):
                continue
        track_a = a.get("track")
        track_b = b.get("track")
        if policy == "same_track":
            if track_a and track_b and track_a == track_b:
                eligible.append((a, b))
        elif policy == "cross_track":
            if track_a and track_b and track_a != track_b:
                eligible.append((a, b))
        else:
            eligible.append((a, b))

    if not eligible:
        return []

    rng = random.Random(seed)
    rng.shuffle(eligible)
    return eligible[:k]


def build_synthesis_prompt(pair: Pair) -> dict[str, Any]:
    """Build the orchestrator-facing prompt for a single finding pair.

    Returns ``{system, user, schema, parent_finding_ids}``. The orchestrator
    calls its own LLM with ``system``+``user``, parses against ``schema``,
    and submits via ``parse_synthesis_response``.
    """
    a, b = pair
    parent_ids = [a.get("id", ""), b.get("id", "")]

    user = (
        "## Parent Finding A\n"
        f"id: {a.get('id', '')}\n"
        f"title: {a.get('title', '')}\n"
        f"track: {a.get('track', '(none)')}\n"
        f"scope: {a.get('scope', '(none)')}\n"
        f"category: {a.get('category', 'observation')}\n"
        f"confidence: {a.get('confidence', 0.0)}\n\n"
        f"{a.get('body', '')}\n\n"
        "## Parent Finding B\n"
        f"id: {b.get('id', '')}\n"
        f"title: {b.get('title', '')}\n"
        f"track: {b.get('track', '(none)')}\n"
        f"scope: {b.get('scope', '(none)')}\n"
        f"category: {b.get('category', 'observation')}\n"
        f"confidence: {b.get('confidence', 0.0)}\n\n"
        f"{b.get('body', '')}\n\n"
        "## Task\n"
        "Propose 1-3 experiment hypotheses that synthesize both parents."
    )

    return {
        "system": SYNTHESIS_SYSTEM_PROMPT,
        "user": redact_secrets(user),
        "schema": SYNTHESIS_RESPONSE_SCHEMA,
        "parent_finding_ids": parent_ids,
    }


def parse_synthesis_response(
    response: dict[str, Any] | str,
    pair: Pair,
) -> list[dict[str, Any]]:
    """Validate an orchestrator-supplied synthesis response.

    Returns the list of validated hypothesis dicts, each annotated with
    ``parent_finding_ids`` so downstream callers can record provenance.
    """
    if isinstance(response, str):
        parsed = parse_json_from_text(response) or {}
    elif isinstance(response, dict):
        parsed = response
    else:
        return []

    raw_list = parsed.get("hypotheses", [])
    hypotheses = _validate_hypotheses(raw_list, iteration=0)

    parent_ids = [pair[0].get("id", ""), pair[1].get("id", "")]
    for h in hypotheses:
        h["parent_finding_ids"] = list(parent_ids)
        h["generation_method"] = "synthesis"
    return hypotheses
