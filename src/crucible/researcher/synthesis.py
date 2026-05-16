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


_VALID_POLICIES = ("random", "same_track", "cross_track", "memory_filter")


# Phase 4.2 — memory_filter scoring constants. Tunable, but the
# defaults reflect the plan's framing: confidence dominates, recency
# is a tiebreaker, cross-project diversity is a small bonus that
# breaks ties between equally-confident finding pairs.
_RECENCY_HALF_LIFE_DAYS = 90.0
_CROSS_PROJECT_BONUS = 0.15
_SAME_PROJECT_PENALTY = -0.05
_TAG_OVERLAP_BONUS = 0.10


def mine_pairs(
    findings: list[dict[str, Any]],
    *,
    k: int,
    policy: str = "random",
    seed: int | None = None,
    required_tags: set[str] | None = None,
    now: str | None = None,
) -> list[Pair]:
    """Sample ``k`` unordered finding pairs from ``findings`` per ``policy``.

    Policies:
      - ``random``: any unique unordered pair, shuffled uniformly.
      - ``same_track``: both findings share a non-empty ``track``.
      - ``cross_track``: findings have different non-empty tracks.
      - ``memory_filter`` (Phase 4.2): score every eligible pair by
        ``(confidence * recency_decay) + cross_project_diversity +
        tag_overlap_bonus`` and return the top-k. Useful when the hub
        has many findings and uniform random sampling washes out the
        high-confidence cross-project synthesis opportunities.

    ``required_tags`` (optional) applies an OR filter at the pair level.
    ``now`` (optional ISO-8601 string) pins the "current time" for
    recency decay; if None, uses ``utc_now_iso()``.

    Returns at most ``k`` pairs; fewer if the eligible-pair pool is smaller.
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
            # ``random`` and ``memory_filter`` both consider all pairs;
            # they differ only in how the top-k are selected.
            eligible.append((a, b))

    if not eligible:
        return []

    if policy == "memory_filter":
        ref_now = now or _now_iso()
        scored = [(score_pair(a, b, now=ref_now), (a, b)) for a, b in eligible]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [pair for _score, pair in scored[:k]]

    rng = random.Random(seed)
    rng.shuffle(eligible)
    return eligible[:k]


# ---------------------------------------------------------------------------
# memory_filter scoring (Phase 4.2)
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    from crucible.core.log import utc_now_iso
    return utc_now_iso()


def _parse_ts(ts: str | None) -> float | None:
    """Parse an ISO-8601 timestamp to a POSIX float. Tolerant: returns None
    on missing / malformed input."""
    if not ts or not isinstance(ts, str):
        return None
    from datetime import datetime
    try:
        # Handle trailing 'Z' that fromisoformat rejected before 3.11.
        clean = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(clean).timestamp()
    except (ValueError, TypeError):
        return None


def _recency_decay(created_at: str | None, now: str) -> float:
    """Exponential decay from a finding's age in days.

    Half-life = ``_RECENCY_HALF_LIFE_DAYS``. A finding created today
    scores ~1.0; one ~3 months old scores ~0.5; older fades toward 0.
    Returns 0.5 when timestamps can't be parsed (neutral default).
    """
    import math

    t_then = _parse_ts(created_at)
    t_now = _parse_ts(now)
    if t_then is None or t_now is None:
        return 0.5
    age_days = max(0.0, (t_now - t_then) / 86_400.0)
    return math.exp(-age_days * math.log(2) / _RECENCY_HALF_LIFE_DAYS)


def score_finding(finding: dict[str, Any], *, now: str) -> float:
    """Per-finding score: confidence × recency decay.

    Public helper for callers that want to surface individual finding
    salience (e.g., a "what should I look at first?" view in the
    briefing). The pair scorer multiplies these and adds bonuses.
    """
    confidence = float(finding.get("confidence", 0.0))
    return max(0.0, confidence) * _recency_decay(finding.get("created_at"), now)


def score_pair(a: dict[str, Any], b: dict[str, Any], *, now: str) -> float:
    """Score a candidate finding pair for the memory_filter policy.

    Components:
      - Average per-finding score (confidence × recency)
      - Cross-project diversity bonus (penalty when both share the same
        source_project; bonus when they don't)
      - Tag overlap bonus (rewards likely-synergistic pairs)
    """
    base = (score_finding(a, now=now) + score_finding(b, now=now)) * 0.5

    proj_a = a.get("source_project") or ""
    proj_b = b.get("source_project") or ""
    if proj_a and proj_b and proj_a == proj_b:
        diversity = _SAME_PROJECT_PENALTY
    elif proj_a and proj_b:
        diversity = _CROSS_PROJECT_BONUS
    else:
        diversity = 0.0

    tags_a = set(a.get("tags") or [])
    tags_b = set(b.get("tags") or [])
    overlap = _TAG_OVERLAP_BONUS if (tags_a & tags_b) else 0.0

    return base + diversity + overlap


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
