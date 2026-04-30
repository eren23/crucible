"""Phase 2 — read-side HF tooling.

Covers:
  - hf_search.fetch_prior_runs (filter, sort, malformed-row tolerance)
  - hf_discussions.list_discussions / post_discussion (fake HfApi)
  - 3 read-side MCP tools (research_hf_prior_attempts, research_hf_discussions,
    note_post_to_hf_discussions)
  - briefing._hf_prior_runs section presence/absence
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from crucible.core.config import HfCollabConfig, ProjectConfig
from crucible.core.errors import HfError


# ---------------------------------------------------------------------------
# Fake huggingface_hub for Phase 2: separate from Phase 1 module so the
# read-side surface (discussions, paper-style fetches) is exercised.
# ---------------------------------------------------------------------------


class _FakeDiscussion:
    def __init__(self, num: int, title: str, status: str = "open"):
        self.num = num
        self.title = title
        self.status = status
        self.author = "agent-x"
        self.created_at = "2026-04-30T10:00:00Z"
        self.url = f"https://huggingface.co/datasets/demo/repo/discussions/{num}"
        self.is_pull_request = False


class FakeHfApiPhase2:
    def __init__(self, token=None):
        self.token = token
        # Class-level state so multiple instances of the API share the bus.
        self.discussions: dict[str, list[_FakeDiscussion]] = FakeHfApiPhase2._db
        self.created_discussions: list[dict[str, Any]] = FakeHfApiPhase2._created

    _db: dict[str, list[_FakeDiscussion]] = {}
    _created: list[dict[str, Any]] = []

    @classmethod
    def reset(cls) -> None:
        cls._db = {}
        cls._created = []

    def get_repo_discussions(self, repo_id, repo_type="dataset", discussion_status=None):
        items = self._db.get(repo_id, [])
        if discussion_status in ("open", "closed"):
            items = [d for d in items if d.status == discussion_status]
        return iter(items)

    def create_discussion(self, repo_id, repo_type, title, description):
        num = len(self._db.get(repo_id, [])) + 1
        d = _FakeDiscussion(num=num, title=title)
        self._db.setdefault(repo_id, []).append(d)
        self._created.append({
            "repo_id": repo_id, "repo_type": repo_type,
            "title": title, "description": description, "token": self.token,
        })
        return d


class FakeHfModulePhase2(types.ModuleType):
    def __init__(self):
        super().__init__("huggingface_hub")
        self.HfApi = FakeHfApiPhase2
        FakeHfApiPhase2.reset()
        self.downloads: list[dict[str, Any]] = []

    def hf_hub_download(self, **kw):
        # Materialize a pre-staged leaderboard.jsonl into kw['local_dir'].
        local_dir = Path(kw["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        out = local_dir / kw["filename"]
        # Tests overwrite this content via fake.staged_jsonl before calling.
        out.write_text(self._staged.get(kw["repo_id"], ""))
        self.downloads.append(dict(kw))
        return str(out)

    _staged: dict[str, str] = {}


@pytest.fixture()
def fake_hf2(monkeypatch):
    fake = FakeHfModulePhase2()
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    return fake


# ---------------------------------------------------------------------------
# fetch_prior_runs
# ---------------------------------------------------------------------------


class TestFetchPriorRuns:
    def test_empty_repo_id_returns_empty(self):
        from crucible.researcher.hf_search import fetch_prior_runs

        assert fetch_prior_runs("") == []

    def test_pulls_filters_sorts(self, fake_hf2):
        rows = [
            {"name": "a", "challenge": "param-golf-2026", "val_loss": 3.5, "rank": 3},
            {"name": "b", "challenge": "param-golf-2026", "val_loss": 2.1, "rank": 1},
            {"name": "c", "challenge": "other-comp", "val_loss": 1.0, "rank": 1},
            {"name": "d", "challenge": "param-golf-2026", "val_loss": 2.8, "rank": 2},
        ]
        fake_hf2._staged["demo/lb"] = "\n".join(json.dumps(r) for r in rows)

        from crucible.researcher.hf_search import fetch_prior_runs

        out = fetch_prior_runs(
            "demo/lb",
            challenge_id="param-golf-2026",
            primary_metric="val_loss",
            direction="minimize",
            top_k=2,
        )
        assert [r["name"] for r in out] == ["b", "d"]

    def test_malformed_lines_skipped(self, fake_hf2):
        fake_hf2._staged["demo/lb"] = (
            '{"name":"good","val_loss":1.0,"rank":1}\n'
            "this is not json\n"
            '{"name":"good2","val_loss":2.0,"rank":2}\n'
        )
        from crucible.researcher.hf_search import fetch_prior_runs

        out = fetch_prior_runs("demo/lb", primary_metric="val_loss", top_k=10)
        assert {r["name"] for r in out} == {"good", "good2"}

    def test_network_failure_returns_empty(self, monkeypatch):
        # No fake_hf2 fixture → huggingface_hub is the real one (or absent),
        # but we intercept hf_writer.pull_file to simulate a network error.
        from crucible.core import hf_writer
        from crucible.researcher.hf_search import fetch_prior_runs

        def boom(**kw):
            raise HfError("network down")

        monkeypatch.setattr(hf_writer, "pull_file", boom)
        out = fetch_prior_runs("demo/lb", primary_metric="val_loss")
        assert out == []

    def test_maximize_direction(self, fake_hf2):
        rows = [
            {"name": "a", "accuracy": 0.5},
            {"name": "b", "accuracy": 0.9},
            {"name": "c", "accuracy": 0.7},
        ]
        fake_hf2._staged["demo/lb"] = "\n".join(json.dumps(r) for r in rows)
        from crucible.researcher.hf_search import fetch_prior_runs

        out = fetch_prior_runs(
            "demo/lb", primary_metric="accuracy", direction="maximize", top_k=2,
        )
        assert [r["name"] for r in out] == ["b", "c"]

    def test_missing_metric_dropped(self, fake_hf2):
        # Rows missing the requested primary metric must be filtered out,
        # not retained with an inf sentinel — that previously sorted them
        # to the TOP under direction='maximize'.
        rows = [
            {"name": "complete", "val_loss": 1.0},
            {"name": "missing", "model_bytes": 100},   # no val_loss
            {"name": "non_numeric", "val_loss": "n/a"},
            {"name": "complete2", "val_loss": 2.0},
        ]
        fake_hf2._staged["demo/lb"] = "\n".join(json.dumps(r) for r in rows)
        from crucible.researcher.hf_search import fetch_prior_runs

        out = fetch_prior_runs(
            "demo/lb",
            primary_metric="val_loss",
            direction="maximize",
            top_k=10,
        )
        names = [r["name"] for r in out]
        assert "missing" not in names
        assert "non_numeric" not in names
        assert names == ["complete2", "complete"]  # maximize → 2.0 first


# ---------------------------------------------------------------------------
# list_discussions / post_discussion
# ---------------------------------------------------------------------------


class TestDiscussions:
    def test_list_filters_status(self, fake_hf2):
        FakeHfApiPhase2._db["demo/comm"] = [
            _FakeDiscussion(1, "open one", "open"),
            _FakeDiscussion(2, "closed one", "closed"),
            _FakeDiscussion(3, "another open", "open"),
        ]
        from crucible.researcher.hf_discussions import list_discussions

        opens = list_discussions("demo/comm", status="open")
        assert {d["num"] for d in opens} == {1, 3}
        closeds = list_discussions("demo/comm", status="closed")
        assert {d["num"] for d in closeds} == {2}
        all_ = list_discussions("demo/comm", status="all")
        assert len(all_) == 3

    def test_list_handles_missing_sdk(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        from crucible.researcher.hf_discussions import list_discussions

        assert list_discussions("any/repo") == []

    def test_post_creates_and_returns_num(self, fake_hf2):
        from crucible.researcher.hf_discussions import post_discussion

        out = post_discussion(
            "demo/comm", title="hello", description="world", token="t",
        )
        assert out["num"] == 1
        assert out["title"] == "hello"
        # FakeHfApi recorded the create call.
        assert FakeHfApiPhase2._created[0]["title"] == "hello"
        assert FakeHfApiPhase2._created[0]["token"] == "t"

    def test_post_without_sdk_raises(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        from crucible.researcher.hf_discussions import post_discussion

        with pytest.raises(HfError):
            post_discussion("any/repo", title="t", description="b")


# ---------------------------------------------------------------------------
# MCP tools — fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def hf_enabled_config(tmp_path: Path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    cfg = ProjectConfig()
    cfg.project_root = project
    cfg.store_dir = ".crucible"
    cfg.name = "demo"
    cfg.hf_collab = HfCollabConfig(
        enabled=True,
        leaderboard_repo="demo-org/lb",
        findings_repo="demo-org/find",
        recipes_repo="demo-org/rec",
        artifacts_repo="demo-org/art-{project}",
        private=True,
        briefing_auto_pull=True,  # opt-in for tests so briefing pulls
    )
    monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: cfg)
    return cfg


@pytest.fixture()
def hf_disabled_config(tmp_path: Path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    cfg = ProjectConfig()
    cfg.project_root = project
    cfg.store_dir = ".crucible"
    monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: cfg)
    return cfg


# ---------------------------------------------------------------------------
# research_hf_prior_attempts
# ---------------------------------------------------------------------------


class TestResearchHfPriorAttempts:
    def test_no_repo_no_config_errors(self, hf_disabled_config):
        from crucible.mcp.tools import research_hf_prior_attempts

        out = research_hf_prior_attempts({})
        assert "leaderboard_repo is empty" in out["error"]

    def test_uses_config_default(self, hf_enabled_config, fake_hf2):
        rows = [{"name": "x", "val_loss": 1.0, "rank": 1}]
        fake_hf2._staged["demo-org/lb"] = json.dumps(rows[0])

        from crucible.mcp.tools import research_hf_prior_attempts

        out = research_hf_prior_attempts({})
        assert out["ok"] is True
        assert out["repo_id"] == "demo-org/lb"
        assert out["count"] == 1

    def test_explicit_repo_wins(self, hf_enabled_config, fake_hf2):
        fake_hf2._staged["other/repo"] = json.dumps({"name": "y", "val_loss": 2.0, "rank": 1})
        from crucible.mcp.tools import research_hf_prior_attempts

        out = research_hf_prior_attempts({"repo_id": "other/repo"})
        assert out["repo_id"] == "other/repo"

    def test_bad_template_returns_error(self, hf_enabled_config):
        from crucible.mcp.tools import research_hf_prior_attempts

        out = research_hf_prior_attempts({"repo_id": "demo-org/{nope}"})
        assert "ValueError" in out["error"]


# ---------------------------------------------------------------------------
# research_hf_discussions
# ---------------------------------------------------------------------------


class TestResearchHfDiscussions:
    def test_lists_open(self, hf_enabled_config, fake_hf2):
        FakeHfApiPhase2._db["demo-org/comm"] = [
            _FakeDiscussion(1, "first", "open"),
            _FakeDiscussion(2, "done", "closed"),
        ]
        from crucible.mcp.tools import research_hf_discussions

        out = research_hf_discussions({"repo_id": "demo-org/comm", "status": "open"})
        assert out["ok"] is True
        assert out["count"] == 1
        assert out["discussions"][0]["num"] == 1

    def test_missing_repo_id(self, hf_enabled_config):
        from crucible.mcp.tools import research_hf_discussions

        out = research_hf_discussions({})
        assert "repo_id is required" in out["error"]

    def test_invalid_status(self, hf_enabled_config):
        from crucible.mcp.tools import research_hf_discussions

        out = research_hf_discussions({"repo_id": "x/y", "status": "bogus"})
        assert "Invalid status" in out["error"]


# ---------------------------------------------------------------------------
# note_post_to_hf_discussions
# ---------------------------------------------------------------------------


class TestNotePostToHfDiscussions:
    def test_disabled_guard(self, hf_disabled_config):
        from crucible.mcp.tools import note_post_to_hf_discussions

        out = note_post_to_hf_discussions({"title": "t", "body": "b"})
        assert "disabled" in out["error"]

    def test_explicit_title_body_posts(self, hf_enabled_config, fake_hf2):
        from crucible.mcp.tools import note_post_to_hf_discussions

        out = note_post_to_hf_discussions({
            "title": "I tried muon, val_loss 2.3, lr too low",
            "body": "config: {...}\nfinding: warmup mattered.",
        })
        assert out["ok"] is True
        assert out["repo_id"] == "demo-org/find"
        assert out["num"] == 1
        assert FakeHfApiPhase2._created[0]["title"].startswith("I tried muon")

    def test_resolves_from_local_note(self, hf_enabled_config, fake_hf2, monkeypatch):
        # Fake NoteStore: get_note returns (meta, body)
        class _FakeStore:
            def get_note(self, note_id):
                return ({"stage": "post-run", "tags": ["muon", "warmup"]}, "ran muon, lr collapse at step 800")

        monkeypatch.setattr("crucible.mcp.tools._get_note_store", lambda: _FakeStore())

        from crucible.mcp.tools import note_post_to_hf_discussions

        out = note_post_to_hf_discussions({
            "run_id": "run-42",
            "note_id": "note-7",
        })
        assert out["ok"] is True
        created = FakeHfApiPhase2._created[0]
        assert "post-run" in created["title"]
        assert "run-42" in created["title"]
        assert "muon" in created["description"]
        assert "lr collapse" in created["description"]

    def test_missing_inputs_errors(self, hf_enabled_config, fake_hf2):
        from crucible.mcp.tools import note_post_to_hf_discussions

        out = note_post_to_hf_discussions({})
        assert "title+body" in out["error"]

    def test_secrets_redacted_before_post(self, hf_enabled_config, fake_hf2):
        # If a note body contains an API key (e.g. from a copy-pasted env
        # dump or a stack trace), it must NOT reach HF verbatim.
        from crucible.mcp.tools import note_post_to_hf_discussions

        out = note_post_to_hf_discussions({
            "title": "smoke test",
            "body": (
                "ran experiment, got NaN.\n"
                "WANDB_API_KEY=wandb_v2_abcdefghij1234567890\n"
                "config: HF_TOKEN=hf_abcdefghij1234567890\n"
                "raw token: sk-ant-api03-abc123def456ghi789"
            ),
        })
        assert out["ok"] is True
        posted = FakeHfApiPhase2._created[-1]
        assert "wandb_v2_abc" not in posted["description"]
        assert "hf_abc" not in posted["description"]
        assert "sk-ant-api03" not in posted["description"]
        assert "<REDACTED>" in posted["description"]


# ---------------------------------------------------------------------------
# Briefing extension
# ---------------------------------------------------------------------------


class TestBriefingHfSection:
    def test_disabled_returns_empty(self, hf_disabled_config):
        from crucible.researcher.briefing import build_briefing

        out = build_briefing(hf_disabled_config)
        assert "hf_prior_runs" in out
        assert out["hf_prior_runs"] == []

    def test_enabled_includes_runs(self, hf_enabled_config, fake_hf2):
        rows = [{"name": "peer-1", "val_loss": 2.5, "rank": 1}]
        fake_hf2._staged["demo-org/lb"] = json.dumps(rows[0])

        from crucible.researcher.briefing import build_briefing

        out = build_briefing(hf_enabled_config)
        assert len(out["hf_prior_runs"]) == 1
        assert out["hf_prior_runs"][0]["name"] == "peer-1"
        # Markdown must call out the section.
        assert "Peer Agents" in out["markdown_summary"]

    def test_enabled_but_no_auto_pull_skips_hf(self, tmp_path: Path, monkeypatch, fake_hf2):
        # Even with hf_collab.enabled=True, briefing must NOT hit HF unless
        # briefing_auto_pull is also true. This protects briefings from
        # adding 1-30s of HF latency to every call.
        project = tmp_path / "proj"
        project.mkdir()
        cfg = ProjectConfig()
        cfg.project_root = project
        cfg.store_dir = ".crucible"
        cfg.name = "demo"
        cfg.hf_collab = HfCollabConfig(
            enabled=True,
            leaderboard_repo="demo-org/lb",
            briefing_auto_pull=False,  # the gate under test
        )
        fake_hf2._staged["demo-org/lb"] = json.dumps(
            {"name": "should-not-appear", "val_loss": 1.0, "rank": 1}
        )

        from crucible.researcher.briefing import build_briefing

        out = build_briefing(cfg)
        assert out["hf_prior_runs"] == []
        # No HF download occurred.
        assert fake_hf2.downloads == []
