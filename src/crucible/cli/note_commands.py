"""CLI handlers for experiment notes (crucible note ...).

All note operations use NoteStore as the canonical storage backend,
ensuring consistency with API and MCP entry points.
"""
from __future__ import annotations

import argparse
import sys

from crucible.core.config import load_config
from crucible.runner.notes import NoteStore


def _get_note_store() -> NoteStore:
    config = load_config()
    return NoteStore(config.project_root / config.store_dir)


def handle_note(args: argparse.Namespace) -> None:
    """Dispatch note subcommands."""
    cmd = getattr(args, "note_command", None)
    if cmd is None:
        print("Usage: crucible note {add|list|search}", file=sys.stderr)
        sys.exit(1)

    if cmd == "add":
        _cmd_add(args)
    elif cmd == "list":
        _cmd_list(args)
    elif cmd == "search":
        _cmd_search(args)
    else:
        print(f"Unknown note command: {cmd}", file=sys.stderr)
        sys.exit(1)


def _cmd_add(args: argparse.Namespace) -> None:
    """Add a note to an experiment run."""
    note_store = _get_note_store()

    run_id = args.run_id
    text = args.text
    stage = getattr(args, "stage", "") or ""
    tags = getattr(args, "tags", None) or []

    if not text:
        print("Error: note text is required.", file=sys.stderr)
        sys.exit(1)

    meta = note_store.add(
        run_id,
        body=text,
        stage=stage,
        tags=tags,
        created_by="cli",
    )
    print(f"Note added: {meta['note_id']} for run {run_id}")


def _cmd_list(args: argparse.Namespace) -> None:
    """List notes for an experiment run."""
    note_store = _get_note_store()
    run_id = args.run_id

    notes = note_store.get_for_run(run_id)
    if not notes:
        print(f"No notes for run {run_id}.")
        return

    for note in notes:
        ts = note.get("created_at", "")[:19]
        by = note.get("created_by", "unknown")
        tags = ", ".join(note.get("tags", []))
        stage = note.get("stage", "")
        body_preview = note.get("body", "")[:80].replace("\n", " ")
        tag_str = f" [{tags}]" if tags else ""
        stage_str = f" ({stage})" if stage else ""
        print(f"  {ts}  by:{by}{stage_str}{tag_str}")
        print(f"    {body_preview}")
        print()


def _cmd_search(args: argparse.Namespace) -> None:
    """Search notes across all experiments."""
    note_store = _get_note_store()

    query = getattr(args, "query", "") or ""
    tags = getattr(args, "tags", None) or []

    results = note_store.search(query=query, tags=tags)
    if not results:
        print("No matching notes found.")
        return

    for note in results:
        run_id = note.get("run_id", "")
        ts = note.get("created_at", "")[:19]
        by = note.get("created_by", "unknown")
        body_preview = note.get("body", "")[:80].replace("\n", " ")
        tag_list = ", ".join(note.get("tags", []))
        tag_str = f" [{tag_list}]" if tag_list else ""
        print(f"  [{run_id}] {ts}  by:{by}{tag_str}")
        print(f"    {body_preview}")
        print()
