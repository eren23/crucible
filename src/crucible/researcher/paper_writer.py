"""Paper draft generator (Phase 4.1).

One MCP-tool surface that takes a research track and produces a
structured markdown paper draft. Built on the orchestrator-contract
pattern (no LLM keys in Crucible): the tool emits a
``{system, user, schema}`` prompt bundle that the orchestrator runs
against its own LLM, then submits the response back via
:func:`parse_paper_draft_response` which validates the section
shape and returns the assembled markdown.

Input gathered automatically from the project state:
- Track metadata (name, description, project membership)
- Top-K findings from the hub (track + global)
- Top-K leaderboard rows (best primary-metric runs)
- Recent notes (pre-run / during-run / post-run)
- Active + recent hypotheses with status

The orchestrator's role is taste — picking a clean narrative line,
phrasing limitations honestly, choosing related work to cite.
Crucible doesn't second-guess any of that; it just stages the
evidence.

Output structure (the orchestrator must return all sections; any
missing one fails validation):
- ``abstract``
- ``introduction``
- ``method``
- ``results``
- ``discussion``
- ``limitations``
- ``related_work``

Plus optional:
- ``title`` (orchestrator may propose; defaults to track name)
- ``key_findings`` (bullet list)
"""
from __future__ import annotations

from typing import Any

from crucible.core.errors import CrucibleError
from crucible.core.redact import redact_secrets


# Required + optional sections. Order is also the order they appear
# in the rendered markdown.
_REQUIRED_SECTIONS = (
    "abstract",
    "introduction",
    "method",
    "results",
    "discussion",
    "limitations",
    "related_work",
)
_OPTIONAL_SECTIONS = ("title", "key_findings")


PAPER_SYSTEM_PROMPT = (
    "You are writing a tight ML research paper draft from real "
    "experiment evidence. Output ONLY valid JSON matching the supplied "
    "schema — no markdown fences, no prose preamble.\n\n"
    "Style:\n"
    "- Each section is a self-contained block of prose (markdown OK "
    "  inside string values; the renderer concatenates verbatim).\n"
    "- Abstract: 150-250 words. State the question, the method, the "
    "  numeric headline, the contribution.\n"
    "- Method: describe the experiment design, training setup, "
    "  evaluation protocol. No prose pads.\n"
    "- Results: lead with numbers; reference the specific runs by "
    "  name. Tables are fine (markdown).\n"
    "- Limitations: honest. If the results have known caveats — small "
    "  sample, single seed, narrow benchmark — say so. Anti-pattern: "
    "  vague 'future work could explore' filler.\n"
    "- Related work: cite the findings + literature explicitly; "
    "  groundedness over completeness.\n"
    "- Do not invent metrics, paper titles, or authors not present in "
    "  the supplied context.\n"
)

PAPER_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": list(_REQUIRED_SECTIONS),
    "properties": {
        "title": {"type": "string"},
        "abstract": {"type": "string"},
        "introduction": {"type": "string"},
        "method": {"type": "string"},
        "results": {"type": "string"},
        "discussion": {"type": "string"},
        "limitations": {"type": "string"},
        "related_work": {"type": "string"},
        "key_findings": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}


class PaperDraftError(CrucibleError):
    """Paper draft generation failed (missing track, bad orchestrator response)."""


def gather_track_context(
    *,
    track_name: str,
    hub_store: Any,
    leaderboard_rows: list[dict[str, Any]] | None = None,
    notes: list[dict[str, Any]] | None = None,
    hypotheses: list[dict[str, Any]] | None = None,
    max_findings: int = 25,
    max_leaderboard: int = 10,
    max_notes: int = 10,
) -> dict[str, Any]:
    """Pull every input the paper-draft prompt needs from the project.

    Pure data assembly: no LLM calls, no fleet ops. Caller wires up
    the data sources (hub store, leaderboard from analysis, notes from
    NoteStore, hypotheses from ResearchState) and passes them in.
    """
    track = _read_track(hub_store, track_name)

    findings = hub_store.load_context_for_track(
        track_name, include_global=True, max_findings=max_findings,
    )

    return {
        "track": {
            "name": track_name,
            "description": (track or {}).get("description", ""),
            "projects": (track or {}).get("projects", []),
        },
        "findings": findings,
        "leaderboard": (leaderboard_rows or [])[:max_leaderboard],
        "notes": (notes or [])[:max_notes],
        "hypotheses": hypotheses or [],
    }


def build_paper_draft_prompt(context: dict[str, Any]) -> dict[str, Any]:
    """Build the orchestrator-facing prompt for a paper draft.

    Returns ``{system, user, schema, sections}``. The orchestrator
    calls its own LLM with ``system``+``user``, parses against
    ``schema``, and the response goes back through
    :func:`parse_paper_draft_response`.
    """
    track = context.get("track", {})
    findings = context.get("findings", [])
    leaderboard = context.get("leaderboard", [])
    notes = context.get("notes", [])
    hypotheses = context.get("hypotheses", [])

    user_parts: list[str] = []
    user_parts.append(f"# Track: {track.get('name', '(unnamed)')}")
    desc = (track.get("description") or "").strip()
    if desc:
        user_parts.append(desc)
    projects = track.get("projects") or []
    if projects:
        user_parts.append(f"Projects in this track: {', '.join(projects)}")

    if findings:
        user_parts.append(f"\n## Findings (top {len(findings)})")
        for i, f in enumerate(findings, 1):
            user_parts.append(
                f"\n### Finding {i} — [{f.get('category', 'observation')}] "
                f"confidence={f.get('confidence', 0.0):.2f} "
                f"scope={f.get('_source_scope', f.get('scope', '?'))}"
            )
            title = f.get("title") or ""
            if title:
                user_parts.append(f"**{title}**")
            body = (f.get("body") or "").strip()
            if body:
                user_parts.append(body)
            source_exps = f.get("source_experiments") or []
            if source_exps:
                user_parts.append(
                    f"  _Source experiments: {', '.join(source_exps[:5])}_"
                )

    if leaderboard:
        user_parts.append(f"\n## Leaderboard (top {len(leaderboard)})")
        for row in leaderboard:
            name = row.get("name", "?")
            metric = row.get("primary_metric") or row.get("metric") or "score"
            value = row.get("primary_value") or row.get("metric_value") or row.get(metric)
            user_parts.append(
                f"- {name}: {metric}={value}"
                + (f" ({row.get('steps_completed', '?')} steps)" if "steps_completed" in row else "")
            )

    if hypotheses:
        user_parts.append(f"\n## Hypotheses (active + recent)")
        for h in hypotheses[:10]:
            user_parts.append(
                f"- [{h.get('status', '?')}] {h.get('name', '?')}: "
                f"{h.get('hypothesis', '')[:140]} (impact={h.get('expected_impact', '?')})"
            )

    if notes:
        user_parts.append(f"\n## Recent notes")
        for n in notes[:10]:
            stage = n.get("stage") or "?"
            preview = (n.get("body") or n.get("body_preview") or "").strip().replace("\n", " ")
            user_parts.append(f"- [{stage}] {preview[:160]}")

    user_parts.append(
        "\n## Task\n"
        "Write a complete paper draft as JSON matching the schema. "
        "Required sections: " + ", ".join(_REQUIRED_SECTIONS) + ". "
        "Optional: " + ", ".join(_OPTIONAL_SECTIONS) + "."
    )

    return {
        "system": PAPER_SYSTEM_PROMPT,
        "user": redact_secrets("\n".join(user_parts)),
        "schema": PAPER_RESPONSE_SCHEMA,
        "sections": list(_REQUIRED_SECTIONS),
    }


def parse_paper_draft_response(
    response: dict[str, Any] | str,
    track_name: str,
) -> dict[str, Any]:
    """Validate an orchestrator-supplied paper-draft response.

    Returns ``{title, sections, key_findings, markdown}`` where
    ``markdown`` is the fully-rendered paper draft. Raises
    :class:`PaperDraftError` if any required section is missing or
    empty.
    """
    from crucible.researcher.llm_client import parse_json_from_text

    if isinstance(response, str):
        parsed = parse_json_from_text(response) or {}
    elif isinstance(response, dict):
        parsed = response
    else:
        raise PaperDraftError(
            f"paper_writer: response must be dict or JSON string, got {type(response).__name__}."
        )

    missing = [
        s for s in _REQUIRED_SECTIONS
        if not isinstance(parsed.get(s), str) or not parsed[s].strip()
    ]
    if missing:
        raise PaperDraftError(
            f"paper_writer: response missing required section(s): {missing}. "
            f"Provide all of: {list(_REQUIRED_SECTIONS)}."
        )

    title = (parsed.get("title") or track_name).strip()
    key_findings = parsed.get("key_findings") or []
    if not isinstance(key_findings, list):
        key_findings = []

    sections = {s: parsed[s].strip() for s in _REQUIRED_SECTIONS}
    markdown = _render_markdown(title, sections, key_findings)
    return {
        "title": title,
        "sections": sections,
        "key_findings": key_findings,
        "markdown": markdown,
    }


def _read_track(hub_store: Any, track_name: str) -> dict[str, Any] | None:
    """Best-effort load of a track yaml from the hub."""
    reader = getattr(hub_store, "_read_track_yaml", None)
    if callable(reader):
        try:
            return reader(track_name)
        except Exception:
            return None
    return None


def _demote_top_level_headers(text: str) -> str:
    """Demote any leading-line H1/H2 ATX headers by one level.

    Section bodies arrive from the orchestrator's LLM as free-text
    markdown. If a body contains its own ``# Headline`` or ``## Sub``,
    those would collide with the renderer's structural headers
    (``# {title}`` and ``## Method``). Demote to keep the rendered
    document's outline intact: ``#`` → ``###``, ``##`` → ``###``.
    Deeper levels are left alone.
    """
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            indent = line[: len(line) - len(stripped)]
            out.append(f"{indent}### {stripped[2:]}")
        elif stripped.startswith("## ") and not stripped.startswith("### "):
            indent = line[: len(line) - len(stripped)]
            out.append(f"{indent}### {stripped[3:]}")
        else:
            out.append(line)
    return "\n".join(out)


def _render_markdown(
    title: str,
    sections: dict[str, str],
    key_findings: list[str],
) -> str:
    """Assemble the final markdown paper draft.

    Each section body is run through :func:`_demote_top_level_headers`
    so LLM-supplied ATX headers (``#`` / ``##``) don't compete with
    the renderer's own structure.
    """
    lines: list[str] = [f"# {title}", ""]

    abstract = sections.get("abstract", "").strip()
    if abstract:
        lines.append("## Abstract")
        lines.append("")
        lines.append(_demote_top_level_headers(abstract))
        lines.append("")

    if key_findings:
        lines.append("## Key Findings")
        lines.append("")
        for kf in key_findings:
            lines.append(f"- {kf}")
        lines.append("")

    # Render the remaining required sections in canonical order.
    section_titles = {
        "introduction": "Introduction",
        "method": "Method",
        "results": "Results",
        "discussion": "Discussion",
        "limitations": "Limitations",
        "related_work": "Related Work",
    }
    for key, header in section_titles.items():
        body = sections.get(key, "").strip()
        if not body:
            continue
        lines.append(f"## {header}")
        lines.append("")
        lines.append(_demote_top_level_headers(body))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
