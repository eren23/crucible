"""Tests for CLI commands: notes, hub, and track.

All operations go through canonical stores (NoteStore, HubStore),
so tests verify the store-level artifacts exist in the correct locations.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from crucible.core.hub import HubStore
from crucible.runner.notes import NoteStore


# ---------------------------------------------------------------------------
# NoteStore tests (backing CLI note commands)
# ---------------------------------------------------------------------------


class TestNoteStore:
    """Test NoteStore — the canonical note backend used by CLI and API."""

    @pytest.fixture
    def note_store(self, tmp_path: Path) -> NoteStore:
        return NoteStore(tmp_path / ".crucible")

    def test_add_creates_note_file(self, note_store: NoteStore) -> None:
        meta = note_store.add("run_001", body="Test note", tags=["test"])
        assert meta["run_id"] == "run_001"
        assert meta["note_id"]
        assert meta["created_by"] == "unknown"

        # Verify note file exists in .crucible/notes/run_001/
        notes_dir = note_store.notes_dir / "run_001"
        assert notes_dir.exists()
        md_files = list(notes_dir.glob("*.md"))
        assert len(md_files) == 1

    def test_add_note_with_stage_and_tags(self, note_store: NoteStore) -> None:
        meta = note_store.add(
            "run_002",
            body="Overfitting detected",
            tags=["issue", "training"],
            stage="evaluation",
            created_by="cli",
        )
        assert meta["stage"] == "evaluation"
        assert meta["tags"] == ["issue", "training"]
        assert meta["created_by"] == "cli"

    def test_get_for_run_returns_notes(self, note_store: NoteStore) -> None:
        note_store.add("run_001", body="First note")
        note_store.add("run_001", body="Second note")
        note_store.add("run_002", body="Different run")

        notes = note_store.get_for_run("run_001")
        assert len(notes) == 2

    def test_get_for_run_empty(self, note_store: NoteStore) -> None:
        assert note_store.get_for_run("nonexistent") == []

    def test_search_by_run_id(self, note_store: NoteStore) -> None:
        note_store.add("run_001", body="Note A")
        note_store.add("run_002", body="Note B")

        results = note_store.search(run_id="run_001")
        assert len(results) == 1
        assert results[0]["run_id"] == "run_001"

    def test_search_by_tags(self, note_store: NoteStore) -> None:
        note_store.add("run_001", body="Tagged note", tags=["important"])
        note_store.add("run_001", body="Untagged note", tags=[])

        results = note_store.search(tags=["important"])
        assert len(results) == 1

    def test_notes_in_canonical_directory(self, note_store: NoteStore) -> None:
        """Notes must be in .crucible/notes/ directory."""
        note_store.add("run_001", body="Test")
        assert note_store.notes_dir.exists()
        assert note_store.index_path.exists()
        assert (note_store.notes_dir / "run_001").is_dir()

    def test_note_file_has_yaml_frontmatter(self, note_store: NoteStore) -> None:
        note_store.add("run_001", body="Some content", tags=["meta"])
        notes_dir = note_store.notes_dir / "run_001"
        md_files = list(notes_dir.glob("*.md"))
        assert len(md_files) == 1

        content = md_files[0].read_text(encoding="utf-8")
        assert content.startswith("---\n")
        parts = content.split("---", 2)
        assert len(parts) >= 3
        fm = yaml.safe_load(parts[1])
        assert fm["run_id"] == "run_001"
        assert "meta" in fm["tags"]

    def test_get_note_returns_body(self, note_store: NoteStore) -> None:
        meta = note_store.add("run_001", body="Hello world")
        result = note_store.get_note(meta["note_id"])
        assert result is not None
        _, body = result
        assert "Hello world" in body


# ---------------------------------------------------------------------------
# HubStore tests (backing CLI hub commands)
# ---------------------------------------------------------------------------


class TestHubStore:
    """Test HubStore — the canonical hub backend used by CLI hub commands."""

    @pytest.fixture
    def hub(self, tmp_path: Path) -> HubStore:
        return HubStore.init(tmp_path / "test-hub")

    def test_init_creates_hub_yaml(self, hub: HubStore) -> None:
        assert (hub.hub_dir / "hub.yaml").exists()

    def test_init_creates_tracks_dir(self, hub: HubStore) -> None:
        assert (hub.hub_dir / "tracks").is_dir()

    def test_initialized_property(self, hub: HubStore) -> None:
        assert hub.initialized is True

    def test_link_project(self, hub: HubStore, tmp_path: Path) -> None:
        project_dir = tmp_path / "my-project"
        project_dir.mkdir()
        entry = hub.link_project("my-project", project_dir)
        assert entry["name"] == "my-project"
        projects = hub.list_projects()
        assert len(projects) == 1
        assert projects[0]["name"] == "my-project"

    def test_link_multiple_projects(self, hub: HubStore, tmp_path: Path) -> None:
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        hub.link_project("proj-a", tmp_path / "a")
        hub.link_project("proj-b", tmp_path / "b")
        projects = hub.list_projects()
        assert len(projects) == 2
        names = {p["name"] for p in projects}
        assert names == {"proj-a", "proj-b"}


class TestHubStoreTracks:
    """Test HubStore track operations — backing CLI track commands."""

    @pytest.fixture
    def hub(self, tmp_path: Path) -> HubStore:
        return HubStore.init(tmp_path / "test-hub")

    def test_create_track(self, hub: HubStore) -> None:
        meta = hub.create_track("exploration", description="Initial exploration")
        assert meta["name"] == "exploration"
        assert meta["description"] == "Initial exploration"

        track_dir = hub.hub_dir / "tracks" / "exploration"
        assert track_dir.exists()
        assert (track_dir / "track.yaml").exists()

    def test_create_duplicate_track_raises(self, hub: HubStore) -> None:
        from crucible.core.errors import HubError
        hub.create_track("my-track")
        with pytest.raises(HubError):
            hub.create_track("my-track")

    def test_list_tracks(self, hub: HubStore) -> None:
        hub.create_track("alpha", description="First")
        hub.create_track("beta", description="Second")
        tracks = hub.list_tracks()
        assert len(tracks) == 2
        names = {t["name"] for t in tracks}
        assert names == {"alpha", "beta"}

    def test_list_tracks_empty(self, hub: HubStore) -> None:
        assert hub.list_tracks() == []

    def test_get_track(self, hub: HubStore) -> None:
        hub.create_track("test-track", description="A test", tags=["v1"])
        track = hub.get_track("test-track")
        assert track is not None
        assert track["name"] == "test-track"
        assert track["description"] == "A test"

    def test_get_track_nonexistent(self, hub: HubStore) -> None:
        assert hub.get_track("no-such-track") is None

    def test_activate_track(self, hub: HubStore) -> None:
        hub.create_track("my-track")
        hub.activate_track("my-track")
        assert hub.get_active_track() == "my-track"

    def test_activate_nonexistent_raises(self, hub: HubStore) -> None:
        from crucible.core.errors import HubError
        with pytest.raises(HubError):
            hub.activate_track("ghost")

    def test_get_active_track_none(self, hub: HubStore) -> None:
        assert hub.get_active_track() is None

    def test_switch_active_track(self, hub: HubStore) -> None:
        hub.create_track("track-a")
        hub.create_track("track-b")
        hub.activate_track("track-a")
        assert hub.get_active_track() == "track-a"
        hub.activate_track("track-b")
        assert hub.get_active_track() == "track-b"

    def test_tracks_in_hub_directory(self, hub: HubStore) -> None:
        hub.create_track("my-track")
        tracks_dir = hub.hub_dir / "tracks"
        assert (tracks_dir / "my-track" / "track.yaml").exists()


# ---------------------------------------------------------------------------
# HubStore.resolve_hub_dir tests
# ---------------------------------------------------------------------------


class TestResolveHubDir:
    def test_explicit_wins(self) -> None:
        result = HubStore.resolve_hub_dir(explicit="/custom/hub")
        assert result == Path("/custom/hub")

    def test_env_fallback(self, monkeypatch) -> None:
        monkeypatch.setenv("CRUCIBLE_HUB_DIR", "/env/hub")
        result = HubStore.resolve_hub_dir()
        assert result == Path("/env/hub")

    def test_config_fallback(self) -> None:
        result = HubStore.resolve_hub_dir(config_hub_dir="/config/hub")
        assert result == Path("/config/hub")

    def test_default_fallback(self, monkeypatch) -> None:
        monkeypatch.delenv("CRUCIBLE_HUB_DIR", raising=False)
        result = HubStore.resolve_hub_dir()
        assert result == Path.home() / ".crucible-hub"
