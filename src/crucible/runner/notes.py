"""NoteStore: freeform markdown notes attached to experiment runs.

Storage layout:
    .crucible/notes/{run_id}/note_{timestamp}.md   -- markdown with YAML frontmatter
    .crucible/notes.jsonl                          -- append-only search index
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from crucible.core.errors import RunnerError
from crucible.core.io import append_jsonl, read_jsonl
from crucible.core.log import utc_now_iso, utc_stamp
from crucible.core.types import ExperimentNote


# ---------------------------------------------------------------------------
# Frontmatter helpers
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(.*?\n)---\s*\n(.*)",
    re.DOTALL,
)


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a markdown file into (YAML metadata dict, body)."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    return meta, m.group(2)


def _render_note(meta: dict[str, Any], body: str) -> str:
    """Render metadata + body into a markdown file with YAML frontmatter."""
    front = yaml.dump(meta, default_flow_style=False, sort_keys=True).rstrip("\n")
    return f"---\n{front}\n---\n\n{body}\n"


# ---------------------------------------------------------------------------
# NoteStore
# ---------------------------------------------------------------------------


class NoteStore:
    """Manage freeform markdown notes attached to experiment runs.

    Parameters
    ----------
    store_dir:
        The ``.crucible`` directory (or any directory used as the store root).
    """

    def __init__(self, store_dir: Path) -> None:
        self.notes_dir = store_dir / "notes"
        self.index_path = store_dir / "notes.jsonl"

    # -- public API ---------------------------------------------------------

    def add(
        self,
        run_id: str,
        body: str,
        *,
        stage: str = "",
        tags: list[str] | None = None,
        confidence: float | None = None,
        supersedes: str | None = None,
        finding_ids: list[str] | None = None,
        created_by: str = "unknown",
    ) -> ExperimentNote:
        """Create a new note for *run_id* and return its index entry.

        Raises
        ------
        RunnerError
            If *run_id* or *body* are empty.
        """
        if not run_id:
            raise RunnerError("run_id is required to add a note")
        if not body or not body.strip():
            raise RunnerError("note body cannot be empty")

        tags = tags or []
        finding_ids = finding_ids or []
        created_at = utc_now_iso()
        stamp = utc_stamp()
        note_id = f"note_{stamp}"

        meta: dict[str, Any] = {
            "run_id": run_id,
            "note_id": note_id,
            "stage": stage,
            "tags": tags,
            "confidence": confidence,
            "supersedes": supersedes,
            "finding_ids": finding_ids,
            "created_by": created_by,
            "created_at": created_at,
        }

        # Write .md file
        run_dir = self.notes_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        note_path = run_dir / f"{note_id}.md"
        note_path.write_text(_render_note(meta, body.strip()), encoding="utf-8")

        # Append to JSONL index (body is NOT stored in the index)
        index_entry: dict[str, Any] = dict(meta)
        index_entry["file"] = str(note_path.relative_to(self.notes_dir.parent))
        append_jsonl(self.index_path, index_entry)

        return index_entry

    def get_for_run(self, run_id: str) -> list[ExperimentNote]:
        """Return all index entries for *run_id*, newest first."""
        entries = [
            e for e in read_jsonl(self.index_path)
            if e.get("run_id") == run_id
        ]
        entries.sort(key=lambda e: e.get("created_at", ""), reverse=True)
        return entries

    def search(
        self,
        *,
        query: str = "",
        tags: list[str] | None = None,
        stage: str = "",
        run_id: str = "",
        limit: int = 50,
    ) -> list[ExperimentNote]:
        """Search the note index with optional filters.

        Parameters
        ----------
        query:
            Substring search against note_id, run_id, stage, and tags.
        tags:
            If provided, notes must contain *all* given tags.
        stage:
            Exact match on stage field.
        run_id:
            Exact match on run_id.
        limit:
            Maximum results to return.
        """
        results: list[dict[str, Any]] = []
        tags = tags or []

        for entry in read_jsonl(self.index_path):
            # run_id filter
            if run_id and entry.get("run_id") != run_id:
                continue

            # stage filter
            if stage and entry.get("stage") != stage:
                continue

            # tags filter (all must be present)
            if tags:
                entry_tags = set(entry.get("tags", []))
                if not all(t in entry_tags for t in tags):
                    continue

            # text query (substring across key fields)
            if query:
                q_lower = query.lower()
                searchable = " ".join([
                    entry.get("note_id", ""),
                    entry.get("run_id", ""),
                    entry.get("stage", ""),
                    entry.get("created_by", ""),
                    " ".join(entry.get("tags", [])),
                ]).lower()
                if q_lower not in searchable:
                    continue

            results.append(entry)
            if len(results) >= limit:
                break

        # newest first
        results.sort(key=lambda e: e.get("created_at", ""), reverse=True)
        return results

    def get_note(self, note_id: str) -> tuple[ExperimentNote, str] | None:
        """Return (index_entry, body) for a single note, or ``None``.

        Reads the markdown file from disk and parses frontmatter to extract
        the body.
        """
        # Find the entry in the index
        for entry in read_jsonl(self.index_path):
            if entry.get("note_id") == note_id:
                file_rel = entry.get("file", "")
                if not file_rel:
                    return entry, ""
                note_path = self.notes_dir.parent / file_rel
                if not note_path.exists():
                    return entry, ""
                text = note_path.read_text(encoding="utf-8")
                _, body = _parse_frontmatter(text)
                return entry, body.strip()
        return None
