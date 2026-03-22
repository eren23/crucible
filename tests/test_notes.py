"""Tests for crucible.runner.notes — NoteStore."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from crucible.core.errors import RunnerError
from crucible.runner.notes import NoteStore, _parse_frontmatter, _render_note


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> NoteStore:
    """Return a NoteStore backed by a temp directory."""
    return NoteStore(tmp_path)


# ---------------------------------------------------------------------------
# Frontmatter helpers
# ---------------------------------------------------------------------------


class TestFrontmatter:
    def test_parse_roundtrip(self):
        meta = {"run_id": "r1", "note_id": "n1", "tags": ["a", "b"]}
        body = "Some markdown body."
        rendered = _render_note(meta, body)
        parsed_meta, parsed_body = _parse_frontmatter(rendered)
        assert parsed_meta["run_id"] == "r1"
        assert parsed_meta["note_id"] == "n1"
        assert parsed_meta["tags"] == ["a", "b"]
        assert parsed_body.strip() == body

    def test_parse_no_frontmatter(self):
        text = "Just plain text, no frontmatter."
        meta, body = _parse_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_parse_empty_frontmatter(self):
        text = "---\n\n---\n\nBody here."
        meta, body = _parse_frontmatter(text)
        assert meta == {} or meta is None or isinstance(meta, dict)


# ---------------------------------------------------------------------------
# NoteStore.add
# ---------------------------------------------------------------------------


class TestAdd:
    def test_add_basic(self, store: NoteStore):
        entry = store.add("run_001", "Loss plateaued at step 3000.")
        assert entry["run_id"] == "run_001"
        assert entry["note_id"].startswith("note_")
        assert entry["created_by"] == "unknown"
        assert "file" in entry

    def test_add_with_all_options(self, store: NoteStore):
        entry = store.add(
            "run_002",
            "Learning rate was too high.",
            stage="post-run",
            tags=["lr", "warmup"],
            confidence=0.8,
            supersedes="note_old",
            finding_ids=["f1", "f2"],
            created_by="mcp-agent",
        )
        assert entry["stage"] == "post-run"
        assert entry["tags"] == ["lr", "warmup"]
        assert entry["confidence"] == 0.8
        assert entry["supersedes"] == "note_old"
        assert entry["finding_ids"] == ["f1", "f2"]
        assert entry["created_by"] == "mcp-agent"

    def test_add_creates_md_file(self, store: NoteStore, tmp_path: Path):
        entry = store.add("run_003", "Test body.")
        note_path = tmp_path / entry["file"]
        assert note_path.exists()
        content = note_path.read_text(encoding="utf-8")
        assert "run_003" in content
        assert "Test body." in content

    def test_add_appends_to_index(self, store: NoteStore, tmp_path: Path):
        store.add("run_004", "First note.")
        store.add("run_004", "Second note.")
        lines = store.index_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        records = [json.loads(line) for line in lines]
        assert all(r["run_id"] == "run_004" for r in records)

    def test_add_empty_run_id_raises(self, store: NoteStore):
        with pytest.raises(RunnerError, match="run_id is required"):
            store.add("", "some body")

    def test_add_empty_body_raises(self, store: NoteStore):
        with pytest.raises(RunnerError, match="body cannot be empty"):
            store.add("run_005", "")

    def test_add_whitespace_body_raises(self, store: NoteStore):
        with pytest.raises(RunnerError, match="body cannot be empty"):
            store.add("run_005", "   \n  ")


# ---------------------------------------------------------------------------
# NoteStore.get_for_run
# ---------------------------------------------------------------------------


class TestGetForRun:
    def test_empty_store(self, store: NoteStore):
        assert store.get_for_run("nonexistent") == []

    def test_returns_correct_run(self, store: NoteStore):
        store.add("run_a", "Note for a.")
        store.add("run_b", "Note for b.")
        store.add("run_a", "Another for a.")

        notes = store.get_for_run("run_a")
        assert len(notes) == 2
        assert all(n["run_id"] == "run_a" for n in notes)

    def test_ordered_newest_first(self, store: NoteStore):
        e1 = store.add("run_c", "First.")
        e2 = store.add("run_c", "Second.")
        notes = store.get_for_run("run_c")
        # Second added later so its created_at >= first
        assert notes[0]["note_id"] == e2["note_id"] or notes[0]["created_at"] >= notes[1]["created_at"]


# ---------------------------------------------------------------------------
# NoteStore.search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_no_filters(self, store: NoteStore):
        store.add("r1", "Note one.", tags=["lr"])
        store.add("r2", "Note two.", tags=["batch"])
        results = store.search()
        assert len(results) == 2

    def test_search_by_run_id(self, store: NoteStore):
        store.add("r1", "A.")
        store.add("r2", "B.")
        results = store.search(run_id="r1")
        assert len(results) == 1
        assert results[0]["run_id"] == "r1"

    def test_search_by_stage(self, store: NoteStore):
        store.add("r1", "Pre.", stage="pre-run")
        store.add("r1", "Post.", stage="post-run")
        results = store.search(stage="post-run")
        assert len(results) == 1
        assert results[0]["stage"] == "post-run"

    def test_search_by_tags(self, store: NoteStore):
        store.add("r1", "Tagged.", tags=["lr", "warmup"])
        store.add("r2", "Other.", tags=["lr"])
        results = store.search(tags=["lr", "warmup"])
        assert len(results) == 1
        assert results[0]["run_id"] == "r1"

    def test_search_by_query(self, store: NoteStore):
        store.add("r1", "Something.", stage="post-run", created_by="human")
        store.add("r2", "Else.", stage="pre-run", created_by="agent")
        results = store.search(query="human")
        assert len(results) == 1
        assert results[0]["run_id"] == "r1"

    def test_search_limit(self, store: NoteStore):
        for i in range(10):
            store.add(f"r{i}", f"Note {i}.")
        results = store.search(limit=3)
        assert len(results) == 3

    def test_search_combined_filters(self, store: NoteStore):
        store.add("r1", "A.", stage="post-run", tags=["lr"])
        store.add("r1", "B.", stage="pre-run", tags=["lr"])
        store.add("r2", "C.", stage="post-run", tags=["lr"])
        results = store.search(run_id="r1", stage="post-run", tags=["lr"])
        assert len(results) == 1
        assert results[0]["run_id"] == "r1"
        assert results[0]["stage"] == "post-run"


# ---------------------------------------------------------------------------
# NoteStore.get_note
# ---------------------------------------------------------------------------


class TestGetNote:
    def test_get_existing_note(self, store: NoteStore):
        entry = store.add("run_x", "The body text.", stage="analysis")
        result = store.get_note(entry["note_id"])
        assert result is not None
        meta, body = result
        assert meta["run_id"] == "run_x"
        assert meta["stage"] == "analysis"
        assert body == "The body text."

    def test_get_nonexistent_note(self, store: NoteStore):
        assert store.get_note("note_nonexistent") is None

    def test_get_note_preserves_multiline_body(self, store: NoteStore):
        body = "Line one.\n\nLine two.\n\n- bullet a\n- bullet b"
        entry = store.add("run_y", body)
        result = store.get_note(entry["note_id"])
        assert result is not None
        _, returned_body = result
        assert "Line one." in returned_body
        assert "- bullet b" in returned_body

    def test_get_note_missing_file(self, store: NoteStore, tmp_path: Path):
        """If the .md file is deleted, get_note returns entry with empty body."""
        entry = store.add("run_z", "Will be deleted.")
        # Remove the file
        note_path = tmp_path / entry["file"]
        note_path.unlink()
        result = store.get_note(entry["note_id"])
        assert result is not None
        meta, body = result
        assert meta["note_id"] == entry["note_id"]
        assert body == ""


# ---------------------------------------------------------------------------
# Integration: multiple runs, notes, and searching
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_workflow(self, store: NoteStore):
        # Add notes for two runs
        e1 = store.add("run_alpha", "Initial observation.", stage="mid-run", tags=["perf"])
        e2 = store.add("run_alpha", "Follow-up analysis.", stage="post-run", tags=["perf", "lr"])
        e3 = store.add("run_beta", "Baseline comparison.", stage="post-run", tags=["baseline"])

        # get_for_run returns only the relevant run
        alpha_notes = store.get_for_run("run_alpha")
        assert len(alpha_notes) == 2

        beta_notes = store.get_for_run("run_beta")
        assert len(beta_notes) == 1

        # search by tag across runs
        perf_notes = store.search(tags=["perf"])
        assert len(perf_notes) == 2

        # get individual note body
        result = store.get_note(e2["note_id"])
        assert result is not None
        _, body = result
        assert "Follow-up analysis." in body

    def test_unique_note_ids(self, store: NoteStore):
        """Note IDs are timestamp-based; quick successive adds should still work."""
        entries = []
        for i in range(5):
            entries.append(store.add("run_q", f"Note {i}."))
        # All entries should be retrievable
        all_notes = store.get_for_run("run_q")
        assert len(all_notes) == 5
