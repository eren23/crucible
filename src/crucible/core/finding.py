"""Finding validation and lifecycle helpers for hub and project scopes.

Findings represent durable research knowledge — observations, beliefs,
constraints, or rejected hypotheses that persist across experiments.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

import yaml

from crucible.core.log import utc_now_iso


# Valid finding statuses
FINDING_STATUSES = {"active", "superseded", "archived", "promoted"}

# Valid finding categories
FINDING_CATEGORIES = {
    "observation",
    "belief",
    "constraint",
    "rejected_hypothesis",
    "technique",
    "reference",
}

# Valid scopes and their promotion order
SCOPE_ORDER = ["project", "track", "global"]

# Minimum confidence for promotion
PROMOTION_RULES: dict[tuple[str, str], dict[str, Any]] = {
    ("project", "track"): {"min_confidence": 0.6},
    ("track", "global"): {"min_confidence": 0.8},
}


def make_finding_id(title: str, scope: str = "project", track: str | None = None) -> str:
    """Generate a deterministic slug-style finding ID from title and scope.

    Returns a slug like ``lr-warmup-matters`` (max 60 chars), with a short
    hash suffix for uniqueness.
    """
    # Slugify the title
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower().strip()).strip("-")
    slug = slug[:40] or "finding"

    # Add scope/track context to the hash for uniqueness
    context = f"{scope}:{track or ''}:{title}"
    suffix = hashlib.sha256(context.encode()).hexdigest()[:8]
    return f"{slug}-{suffix}"


def validate_finding(finding: dict[str, Any]) -> list[str]:
    """Validate a finding dict. Returns a list of error strings (empty = valid)."""
    errors: list[str] = []

    if not finding.get("title"):
        errors.append("Finding must have a non-empty 'title'.")

    if not finding.get("body"):
        from crucible.core.log import log_warn
        log_warn("Finding has no 'body' — consider adding a description")

    category = finding.get("category", "")
    if category and category not in FINDING_CATEGORIES:
        errors.append(
            f"Invalid category '{category}'. Must be one of: {sorted(FINDING_CATEGORIES)}"
        )

    status = finding.get("status", "")
    if status and status not in FINDING_STATUSES:
        errors.append(
            f"Invalid status '{status}'. Must be one of: {sorted(FINDING_STATUSES)}"
        )

    confidence = finding.get("confidence")
    if confidence is not None:
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            errors.append("Confidence must be a number between 0.0 and 1.0.")

    return errors


def can_promote(from_scope: str, to_scope: str) -> bool:
    """Check whether promoting from one scope to another is valid.

    Valid promotions move *up*: project -> track -> global.
    """
    if from_scope not in SCOPE_ORDER or to_scope not in SCOPE_ORDER:
        return False
    return SCOPE_ORDER.index(from_scope) < SCOPE_ORDER.index(to_scope)


# ---------------------------------------------------------------------------
# Markdown serialization
# ---------------------------------------------------------------------------

def render_finding_markdown(finding: dict[str, Any]) -> str:
    """Render a finding as markdown with YAML frontmatter."""
    frontmatter: dict[str, Any] = {}
    for key in ("id", "title", "scope", "status", "confidence", "tags",
                "source_project", "source_experiments", "supersedes",
                "superseded_by", "track", "created_at", "created_by",
                "promoted_from", "category"):
        if key in finding and finding[key] is not None:
            frontmatter[key] = finding[key]

    fm_text = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False, allow_unicode=True).rstrip()
    body = finding.get("body", "")
    return f"---\n{fm_text}\n---\n\n{body}\n"


def parse_finding_markdown(text: str) -> dict[str, Any]:
    """Parse markdown-with-frontmatter back to a Finding dict."""
    finding: dict[str, Any] = {}

    text = text.strip()
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            fm_raw = parts[1].strip()
            body = parts[2].strip()
            frontmatter = yaml.safe_load(fm_raw) or {}
            finding.update(frontmatter)
            finding["body"] = body
        else:
            finding["body"] = text
    else:
        finding["body"] = text

    return finding


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def new_finding(
    title: str,
    body: str,
    *,
    scope: str = "project",
    confidence: float = 0.5,
    tags: list[str] | None = None,
    source_project: str = "",
    source_experiments: list[str] | None = None,
    created_by: str = "unknown",
    track: str | None = None,
    category: str = "observation",
) -> dict[str, Any]:
    """Create a new Finding dict with sensible defaults."""
    return {
        "id": make_finding_id(title, scope, track),
        "title": title,
        "body": body,
        "scope": scope,
        "status": "active",
        "confidence": confidence,
        "tags": tags or [],
        "category": category,
        "source_project": source_project,
        "source_experiments": source_experiments or [],
        "supersedes": None,
        "superseded_by": None,
        "track": track,
        "created_at": utc_now_iso(),
        "created_by": created_by,
        "promoted_from": None,
    }
