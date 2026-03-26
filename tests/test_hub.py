"""Tests for crucible.core.hub — HubStore init, projects, tracks, findings."""
from __future__ import annotations

from pathlib import Path

import pytest

from crucible.core.hub import HubStore
from crucible.core.errors import HubError
from crucible.core.finding import (
    can_promote,
    make_finding_id,
    validate_finding,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hub(tmp_path: Path) -> HubStore:
    """An initialized hub in a temp directory."""
    return HubStore.init(hub_dir=tmp_path / "test-hub", name="test-hub")


@pytest.fixture
def project_path(tmp_path: Path) -> Path:
    """A fake project directory."""
    p = tmp_path / "my-project"
    p.mkdir()
    (p / "crucible.yaml").write_text("name: my-project\n")
    return p


# ---------------------------------------------------------------------------
# Finding helpers
# ---------------------------------------------------------------------------


class TestFindingHelpers:
    def test_make_finding_id(self):
        fid = make_finding_id("LR warmup matters", "track", "transformers")
        assert fid.startswith("lr-warmup-matters-")
        assert len(fid) <= 60

    def test_make_finding_id_deterministic(self):
        a = make_finding_id("Width vs depth", "global")
        b = make_finding_id("Width vs depth", "global")
        assert a == b

    def test_make_finding_id_different_scopes(self):
        a = make_finding_id("Width matters", "track", "t1")
        b = make_finding_id("Width matters", "global")
        assert a != b

    def test_validate_finding_valid(self):
        finding = {
            "title": "LR warmup helps",
            "category": "belief",
            "status": "active",
            "confidence": 0.8,
        }
        assert validate_finding(finding) == []

    def test_validate_finding_missing_title(self):
        errors = validate_finding({"category": "belief"})
        assert any("title" in e for e in errors)

    def test_validate_finding_bad_category(self):
        errors = validate_finding({"title": "X", "category": "nonsense"})
        assert any("category" in e.lower() for e in errors)

    def test_validate_finding_bad_confidence(self):
        errors = validate_finding({"title": "X", "confidence": 1.5})
        assert any("confidence" in e.lower() for e in errors)

    def test_can_promote_valid(self):
        assert can_promote("project", "track") is True
        assert can_promote("track", "global") is True
        assert can_promote("project", "global") is True

    def test_can_promote_invalid(self):
        assert can_promote("global", "track") is False
        assert can_promote("global", "project") is False
        assert can_promote("track", "project") is False

    def test_can_promote_same_scope(self):
        assert can_promote("track", "track") is False


# ---------------------------------------------------------------------------
# HubStore initialization
# ---------------------------------------------------------------------------


class TestHubInit:
    def test_init_creates_dir(self, tmp_path):
        hub_dir = tmp_path / "new-hub"
        hub = HubStore.init(hub_dir=hub_dir, name="my-hub")
        assert hub.hub_dir == hub_dir
        assert hub.initialized is True
        assert (hub_dir / "hub.yaml").exists()
        assert (hub_dir / "tracks").is_dir()
        assert (hub_dir / "global").is_dir()

    def test_init_hub_yaml_content(self, tmp_path):
        import yaml

        hub_dir = tmp_path / "hub"
        HubStore.init(hub_dir=hub_dir, name="test-hub")
        data = yaml.safe_load((hub_dir / "hub.yaml").read_text())
        assert data["name"] == "test-hub"
        assert data["version"] == "1"
        assert data["default_track"] == ""
        assert "created_at" in data

    def test_init_raises_if_already_exists(self, hub):
        with pytest.raises(HubError, match="already initialized"):
            HubStore.init(hub_dir=hub.hub_dir)

    def test_uninitialized_hub_raises(self, tmp_path):
        hub = HubStore(hub_dir=tmp_path / "nonexistent")
        assert hub.initialized is False
        with pytest.raises(HubError, match="not initialized"):
            hub.list_projects()

    def test_discover_returns_none_when_no_hub(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CRUCIBLE_HUB_DIR", str(tmp_path / "nope"))
        monkeypatch.setattr("crucible.core.hub._DEFAULT_HUB_DIR", tmp_path / "nope")
        assert HubStore.discover() is None

    def test_discover_finds_hub(self, hub, monkeypatch):
        monkeypatch.setenv("CRUCIBLE_HUB_DIR", str(hub.hub_dir))
        found = HubStore.discover()
        assert found == hub.hub_dir


# ---------------------------------------------------------------------------
# Project registry
# ---------------------------------------------------------------------------


class TestProjectRegistry:
    def test_link_project(self, hub, project_path):
        record = hub.link_project("my-project", project_path)
        assert record["name"] == "my-project"
        assert record["path"] == str(project_path)
        assert "linked_at" in record

    def test_list_projects(self, hub, project_path):
        hub.link_project("proj-a", project_path)
        projects = hub.list_projects()
        assert len(projects) == 1
        assert projects[0]["name"] == "proj-a"

    def test_link_duplicate_raises(self, hub, project_path):
        hub.link_project("proj-a", project_path)
        with pytest.raises(HubError, match="already linked"):
            hub.link_project("proj-a", project_path)

    def test_link_nonexistent_path_raises(self, hub, tmp_path):
        with pytest.raises(HubError, match="does not exist"):
            hub.link_project("ghost", tmp_path / "nonexistent")

    def test_unlink_project(self, hub, project_path):
        hub.link_project("proj-a", project_path)
        assert hub.unlink_project("proj-a") is True
        assert hub.list_projects() == []

    def test_unlink_nonexistent(self, hub):
        assert hub.unlink_project("nope") is False

    def test_multiple_projects(self, hub, tmp_path):
        for name in ["a", "b", "c"]:
            p = tmp_path / name
            p.mkdir()
            hub.link_project(name, p)
        assert len(hub.list_projects()) == 3
        hub.unlink_project("b")
        names = [p["name"] for p in hub.list_projects()]
        assert "b" not in names
        assert len(names) == 2


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------


class TestTracks:
    def test_create_track(self, hub):
        track = hub.create_track("transformer-pretraining", description="Small transformers", tags=["transformers"])
        assert track["name"] == "transformer-pretraining"
        assert track["description"] == "Small transformers"
        assert "transformers" in track["tags"]
        assert track["active"] is True
        assert "created_at" in track

    def test_get_track(self, hub):
        hub.create_track("my-track", description="test")
        track = hub.get_track("my-track")
        assert track is not None
        assert track["name"] == "my-track"

    def test_get_track_nonexistent(self, hub):
        assert hub.get_track("nonexistent") is None

    def test_create_duplicate_raises(self, hub):
        hub.create_track("dup")
        with pytest.raises(HubError, match="already exists"):
            hub.create_track("dup")

    def test_list_tracks(self, hub):
        hub.create_track("track-a")
        hub.create_track("track-b")
        tracks = hub.list_tracks()
        assert len(tracks) == 2
        names = {t["name"] for t in tracks}
        assert names == {"track-a", "track-b"}

    def test_list_tracks_empty(self, hub):
        assert hub.list_tracks() == []

    def test_activate_track(self, hub):
        hub.create_track("my-track")
        hub.activate_track("my-track")
        assert hub.get_active_track() == "my-track"

    def test_activate_nonexistent_raises(self, hub):
        with pytest.raises(HubError, match="not found"):
            hub.activate_track("nonexistent")

    def test_get_active_track_none_by_default(self, hub):
        assert hub.get_active_track() is None

    def test_switch_active_track(self, hub):
        hub.create_track("a")
        hub.create_track("b")
        hub.activate_track("a")
        assert hub.get_active_track() == "a"
        hub.activate_track("b")
        assert hub.get_active_track() == "b"

    def test_link_project_to_track(self, hub, project_path):
        hub.link_project("my-project", project_path)
        hub.create_track("my-track")
        hub.link_project_to_track("my-track", "my-project")

        track = hub.get_track("my-track")
        assert "my-project" in track["linked_projects"]

    def test_link_unregistered_project_to_track_raises(self, hub):
        hub.create_track("my-track")
        with pytest.raises(HubError, match="not linked to the hub"):
            hub.link_project_to_track("my-track", "ghost-project")

    def test_link_project_to_track_idempotent(self, hub, project_path):
        hub.link_project("proj", project_path)
        hub.create_track("track")
        hub.link_project_to_track("track", "proj")
        hub.link_project_to_track("track", "proj")  # no error
        track = hub.get_track("track")
        assert track["linked_projects"].count("proj") == 1


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


SAMPLE_FINDING = {
    "title": "LR warmup helps convergence",
    "category": "belief",
    "confidence": 0.85,
    "tags": ["lr", "warmup"],
    "source_experiments": ["exp-001", "exp-002"],
}


class TestFindings:
    def test_store_finding_global(self, hub):
        result = hub.store_finding(SAMPLE_FINDING, scope="global")
        assert result["title"] == "LR warmup helps convergence"
        assert result["scope"] == "global"
        assert result["status"] == "active"
        assert result["version"] == 1
        assert "id" in result
        assert "created_at" in result

    def test_store_finding_track(self, hub):
        hub.create_track("my-track")
        result = hub.store_finding(SAMPLE_FINDING, scope="track", track="my-track")
        assert result["scope"] == "track"
        assert result["track"] == "my-track"

    def test_store_invalid_finding_raises(self, hub):
        with pytest.raises(HubError, match="Invalid finding"):
            hub.store_finding({"category": "belief"}, scope="global")  # missing title

    def test_get_finding(self, hub):
        stored = hub.store_finding(SAMPLE_FINDING, scope="global")
        retrieved = hub.get_finding(stored["id"], scope="global")
        assert retrieved is not None
        assert retrieved["title"] == SAMPLE_FINDING["title"]
        assert retrieved["id"] == stored["id"]

    def test_get_finding_nonexistent(self, hub):
        assert hub.get_finding("nonexistent-id", scope="global") is None

    def test_list_findings_global(self, hub):
        hub.store_finding({"title": "Finding A", "category": "observation"}, scope="global")
        hub.store_finding({"title": "Finding B", "category": "belief"}, scope="global")
        findings = hub.list_findings("global")
        assert len(findings) == 2

    def test_list_findings_status_filter(self, hub):
        hub.store_finding({"title": "Active one"}, scope="global")
        f2 = hub.store_finding({"title": "Will be superseded"}, scope="global")
        hub.supersede_finding(
            f2["id"],
            {"title": "Superseding finding"},
            scope="global",
        )
        active = hub.list_findings("global", status="active")
        # Finding A is active, superseding finding is active, but "Will be superseded" is superseded
        active_titles = {f["title"] for f in active}
        assert "Will be superseded" not in active_titles
        assert "Superseding finding" in active_titles

    def test_list_findings_tag_filter(self, hub):
        hub.store_finding({"title": "Tagged", "tags": ["important"]}, scope="global")
        hub.store_finding({"title": "Untagged"}, scope="global")
        tagged = hub.list_findings("global", tags=["important"])
        assert len(tagged) == 1
        assert tagged[0]["title"] == "Tagged"

    def test_list_findings_track_scope(self, hub):
        hub.create_track("t1")
        hub.create_track("t2")
        hub.store_finding({"title": "In T1"}, scope="track", track="t1")
        hub.store_finding({"title": "In T2"}, scope="track", track="t2")

        t1_findings = hub.list_findings("track", track="t1")
        assert len(t1_findings) == 1
        assert t1_findings[0]["title"] == "In T1"

    def test_supersede_finding(self, hub):
        original = hub.store_finding(
            {"title": "Old understanding", "confidence": 0.6},
            scope="global",
        )
        new = hub.supersede_finding(
            original["id"],
            {"title": "New understanding", "confidence": 0.9},
            scope="global",
        )
        assert new["version"] == 2
        assert new["status"] == "active"
        assert new["supersedes_version"] == 1
        assert new["id"] == original["id"]

        # Old version is marked superseded in its YAML
        old_reloaded = hub._read_finding_yaml(original["id"], "global", version=1)
        assert old_reloaded["status"] == "superseded"

    def test_supersede_nonexistent_raises(self, hub):
        with pytest.raises(HubError, match="not found"):
            hub.supersede_finding("ghost-id", {"title": "New"}, scope="global")

    def test_promote_finding_track_to_global(self, hub):
        hub.create_track("my-track")
        original = hub.store_finding(
            {"title": "Track-scoped insight", "tags": ["lr"]},
            scope="track",
            track="my-track",
        )

        promoted = hub.promote_finding(
            original["id"],
            from_scope="track",
            to_scope="global",
            from_track="my-track",
        )
        assert promoted["scope"] == "global"
        assert promoted["status"] == "active"
        assert promoted["promoted_from"]["scope"] == "track"
        assert promoted["promoted_from"]["track"] == "my-track"

        # Source should be marked as promoted
        source = hub.get_finding(original["id"], "track", "my-track")
        assert source["status"] == "promoted"

        # Should appear in global findings
        global_findings = hub.list_findings("global")
        assert any(f["title"] == "Track-scoped insight" for f in global_findings)

    def test_promote_invalid_direction_raises(self, hub):
        hub.create_track("t1")
        original = hub.store_finding(
            {"title": "Global insight"}, scope="global"
        )
        with pytest.raises(HubError, match="Cannot promote"):
            hub.promote_finding(
                original["id"],
                from_scope="global",
                to_scope="track",
                to_track="t1",
            )

    def test_promote_nonexistent_raises(self, hub):
        with pytest.raises(HubError, match="not found"):
            hub.promote_finding("ghost", from_scope="track", to_scope="global", from_track="t")

    def test_finding_disk_layout(self, hub):
        result = hub.store_finding({"title": "Test layout"}, scope="global")
        fid = result["id"]
        fdir = hub._global_dir / "findings" / fid
        assert (fdir / "v1.yaml").exists()
        assert (fdir / "current.yaml").exists()
        assert (hub._global_dir / "findings.jsonl").exists()


# ---------------------------------------------------------------------------
# Context loading
# ---------------------------------------------------------------------------


class TestContextLoading:
    def test_load_context_for_track(self, hub):
        hub.create_track("my-track")
        hub.store_finding({"title": "Track finding"}, scope="track", track="my-track")
        hub.store_finding({"title": "Global finding"}, scope="global")

        context = hub.load_context_for_track("my-track", include_global=True)
        assert len(context) == 2
        titles = {f["title"] for f in context}
        assert "Track finding" in titles
        assert "Global finding" in titles

    def test_load_context_without_global(self, hub):
        hub.create_track("my-track")
        hub.store_finding({"title": "Track only"}, scope="track", track="my-track")
        hub.store_finding({"title": "Global"}, scope="global")

        context = hub.load_context_for_track("my-track", include_global=False)
        assert len(context) == 1
        assert context[0]["title"] == "Track only"

    def test_load_context_respects_max(self, hub):
        hub.create_track("t")
        for i in range(10):
            hub.store_finding({"title": f"Finding {i}"}, scope="track", track="t")

        context = hub.load_context_for_track("t", include_global=False, max_findings=3)
        assert len(context) == 3

    def test_load_context_sorted_newest_first(self, hub):
        hub.create_track("t")
        hub.store_finding({"title": "First"}, scope="track", track="t")
        hub.store_finding({"title": "Second"}, scope="track", track="t")
        hub.store_finding({"title": "Third"}, scope="track", track="t")

        context = hub.load_context_for_track("t", include_global=False)
        # Newest first
        assert context[0]["title"] == "Third"
        assert context[-1]["title"] == "First"


# ---------------------------------------------------------------------------
# Git sync
# ---------------------------------------------------------------------------


class TestGitSync:
    def test_sync_without_remote(self, hub):
        # Should not crash even without a remote
        hub.store_finding({"title": "Something"}, scope="global")
        result = hub.sync()
        assert "committed" in result
        assert "pushed" in result
        assert "pulled" in result

    def test_set_remote(self, hub):
        hub.set_remote("https://github.com/test/hub.git")
        # Verify it was set
        proc = hub._git_run("remote", "get-url", "origin", check=False)
        assert "github.com" in proc.stdout


# ---------------------------------------------------------------------------
# Persistence / reload
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_hub_survives_reload(self, tmp_path, project_path):
        hub_dir = tmp_path / "persist-hub"
        hub1 = HubStore.init(hub_dir=hub_dir, name="persist")
        hub1.link_project("proj", project_path)
        hub1.create_track("t1")
        hub1.store_finding({"title": "F1"}, scope="global")
        hub1.store_finding({"title": "F2"}, scope="track", track="t1")
        hub1.activate_track("t1")

        # Reload
        hub2 = HubStore(hub_dir=hub_dir)
        assert hub2.initialized is True
        assert len(hub2.list_projects()) == 1
        assert len(hub2.list_tracks()) == 1
        assert hub2.get_active_track() == "t1"
        assert len(hub2.list_findings("global")) == 1
        assert len(hub2.list_findings("track", track="t1")) == 1

    def test_invalid_scope_raises(self, hub):
        with pytest.raises(HubError, match="Invalid scope"):
            hub.store_finding({"title": "X"}, scope="nonsense")

    def test_track_scope_without_track_raises(self, hub):
        with pytest.raises(HubError, match="Track name required"):
            hub.store_finding({"title": "X"}, scope="track")


# ---------------------------------------------------------------------------
# Architecture storage
# ---------------------------------------------------------------------------


class TestArchitectures:
    def test_store_code_architecture(self, hub):
        record = hub.store_architecture(
            name="code_arch",
            code="from crucible.models.registry import register_model\nregister_model('code_arch', lambda a: None)\n",
            source_project="proj-a",
        )
        assert record["name"] == "code_arch"
        assert record["kind"] == "code"
        assert record["relative_path"].endswith("architectures/plugins/code_arch.py")
        assert hub.get_architecture_code("code_arch") is not None
        assert hub.get_architecture_content("code_arch") is not None

    def test_store_spec_architecture(self, hub):
        record = hub.store_architecture(
            name="spec_arch",
            code="name: spec_arch\nbase: tied_embedding_lm\nembedding: {}\nblock: {}\nstack: {}\n",
            source_project="proj-a",
            kind="spec",
        )
        assert record["name"] == "spec_arch"
        assert record["kind"] == "spec"
        assert record["relative_path"].endswith("architectures/specs/spec_arch.yaml")
        assert hub.get_architecture_code("spec_arch") is None
        content = hub.get_architecture_content("spec_arch")
        assert content is not None
        assert "name: spec_arch" in content

    def test_list_architectures_normalizes_old_records(self, hub):
        registry_path = hub._arch_registry_path
        registry_path.write_text(
            '{"name": "legacy_arch", "added_at": "2024-01-01T00:00:00Z", "source_project": "", "tags": []}\n',
            encoding="utf-8",
        )
        (hub._arch_plugins_dir / "legacy_arch.py").write_text(
            "from crucible.models.registry import register_model\nregister_model('legacy_arch', lambda a: None)\n",
            encoding="utf-8",
        )
        records = hub.list_architectures()
        assert records[0]["kind"] == "code"
        assert records[0]["relative_path"].endswith("architectures/plugins/legacy_arch.py")
