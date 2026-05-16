"""Tests for crucible.researcher.peer_sync — Phase 4.3."""
from __future__ import annotations

from typing import Any

import pytest

from crucible.researcher import peer_sync as ps


# ---------------------------------------------------------------------------
# Title + post rendering
# ---------------------------------------------------------------------------


class TestTitleAndRender:
    def test_title_uses_challenge_id_prefix(self):
        assert ps.title_for_challenge("paper3-jepa") == "crucible-peer-sync:paper3-jepa"

    def test_render_post_includes_required_fields(self):
        body = ps.render_finding_post(
            agent_id="agent-a",
            challenge_id="paper3-jepa",
            top_finding={
                "title": "VICReg lifts predictor rank",
                "body": "Effective rank >16 across 3 seeds.",
                "category": "observation",
                "confidence": 0.85,
            },
            leaderboard_row={
                "name": "paper3_pred_vicreg",
                "primary_metric": "MRR",
                "primary_value": 0.57,
            },
            iso_now="2026-05-16T12:00:00+00:00",
        )
        assert "crucible-peer-finding (v1)" in body
        assert "agent_id: agent-a" in body
        assert "challenge_id: paper3-jepa" in body
        assert "ts: 2026-05-16T12:00:00+00:00" in body
        assert "leaderboard_metric: MRR=0.57" in body
        assert "leaderboard_run: paper3_pred_vicreg" in body
        assert "VICReg lifts predictor rank" in body
        assert "Effective rank >16" in body

    def test_render_redacts_secrets_from_body(self):
        body = ps.render_finding_post(
            agent_id="a",
            challenge_id="c",
            top_finding={
                "title": "leak test",
                "body": "Discovered with HF_TOKEN=hf_abcdefABCDEF1234567890abcdefABCDEF12.",
            },
            iso_now="now",
        )
        assert "hf_abcdefABCDEF" not in body
        assert ("REDACTED" in body) or ("***" in body)


# ---------------------------------------------------------------------------
# _parse_peer_post
# ---------------------------------------------------------------------------


class TestPeerPostParser:
    def test_v1_header_parsed(self):
        body = (
            "## crucible-peer-finding (v1)\n"
            "agent_id: peer-b\n"
            "challenge_id: paper3-jepa\n"
            "ts: 2026-05-16T08:00:00+00:00\n"
            "leaderboard_metric: MRR=0.42\n"
            "leaderboard_run: foo\n\n"
            "### Top finding\nbody body body\n"
        )
        out = ps._parse_peer_post(body)
        assert out["agent_id"] == "peer-b"
        assert out["challenge_id"] == "paper3-jepa"
        assert out["leaderboard_metric"] == "MRR=0.42"
        assert out["leaderboard_run"] == "foo"
        assert "body body body" in out["body"]

    def test_missing_header_returns_none(self):
        assert ps._parse_peer_post("just some markdown") is None

    def test_present_marker_but_no_agent_id_returns_none(self):
        body = "## crucible-peer-finding (v1)\nchallenge_id: c\n"
        assert ps._parse_peer_post(body) is None


# ---------------------------------------------------------------------------
# sync_peer_finding — uses monkeypatched HF surface
# ---------------------------------------------------------------------------


class TestSyncPeerFinding:
    def _finding(self):
        return {
            "title": "VICReg works",
            "body": "Body.",
            "category": "observation",
            "confidence": 0.9,
        }

    def test_creates_new_thread_when_none_exists(self, monkeypatch):
        # No existing thread.
        monkeypatch.setattr(ps, "list_discussions", lambda *a, **kw: [])
        captured = {}

        def fake_post(repo_id, *, title, description, repo_type, token):
            captured["title"] = title
            captured["description"] = description
            return {"num": 7, "url": "https://hf.co/discussions/7", "title": title}

        monkeypatch.setattr(ps, "post_discussion", fake_post)

        # No peers fetchable (no HfApi); should still succeed.
        monkeypatch.setattr(ps, "_fetch_peer_findings", lambda **kw: [])

        out = ps.sync_peer_finding(
            repo_id="org/repo",
            challenge_id="paper3",
            agent_id="agent-a",
            top_finding=self._finding(),
            iso_now="2026-05-16T00:00:00+00:00",
        )
        assert out["thread_num"] == 7
        assert out["thread_url"].endswith("/discussions/7")
        assert out["posted_url"].endswith("/discussions/7")
        assert out["peer_count"] == 0
        assert captured["title"] == "crucible-peer-sync:paper3"
        assert "agent_id: agent-a" in captured["description"]

    def test_appends_comment_to_existing_thread(self, monkeypatch):
        existing = {
            "num": 12,
            "title": "crucible-peer-sync:paper3",
            "url": "https://hf.co/discussions/12",
        }
        monkeypatch.setattr(ps, "list_discussions", lambda *a, **kw: [existing])

        comment_calls = {"n": 0}

        def fake_comment(*, repo_id, num, body, repo_type, token):
            comment_calls["n"] += 1
            comment_calls["num"] = num
            comment_calls["body"] = body
            return "https://hf.co/discussions/12#comment-42"

        monkeypatch.setattr(ps, "_post_comment_or_new_discussion", fake_comment)
        monkeypatch.setattr(
            ps, "_fetch_peer_findings",
            lambda **kw: [
                {"agent_id": "peer-b", "ts": "2026-05-15T00:00:00+00:00",
                 "body": "## crucible-peer-finding\nagent_id: peer-b",
                 "leaderboard_metric": "MRR=0.40"},
            ],
        )

        out = ps.sync_peer_finding(
            repo_id="org/repo",
            challenge_id="paper3",
            agent_id="agent-a",
            top_finding=self._finding(),
            iso_now="2026-05-16T00:00:00+00:00",
        )
        assert out["thread_num"] == 12
        assert comment_calls["n"] == 1
        assert comment_calls["num"] == 12
        assert "agent_id: agent-a" in comment_calls["body"]
        assert out["peer_count"] == 1
        assert out["peers"][0]["agent_id"] == "peer-b"

    def test_write_failure_does_not_block_read(self, monkeypatch):
        """If posting fails, we still return whatever peers we could
        read from the existing thread."""
        from crucible.core.errors import HfError

        existing = {"num": 9, "title": "crucible-peer-sync:paper3",
                    "url": "https://hf.co/discussions/9"}
        monkeypatch.setattr(ps, "list_discussions", lambda *a, **kw: [existing])

        def fake_comment(**kw):
            raise HfError("network down")

        monkeypatch.setattr(ps, "_post_comment_or_new_discussion", fake_comment)
        monkeypatch.setattr(
            ps, "_fetch_peer_findings",
            lambda **kw: [{"agent_id": "peer-c", "ts": "x", "body": "y"}],
        )
        out = ps.sync_peer_finding(
            repo_id="org/repo", challenge_id="paper3", agent_id="me",
            top_finding=self._finding(),
            iso_now="2026-05-16T00:00:00+00:00",
        )
        assert out["posted_url"] == ""
        assert out["peer_count"] == 1
        assert out["peers"][0]["agent_id"] == "peer-c"

    def test_my_agent_id_filtered_from_peers(self, monkeypatch):
        """_fetch_peer_findings receives my_agent_id and is expected to
        filter our own posts. This verifies the kwarg is wired."""
        captured = {}

        def fake_fetch(**kw):
            captured.update(kw)
            return []

        monkeypatch.setattr(ps, "_fetch_peer_findings", fake_fetch)
        monkeypatch.setattr(ps, "list_discussions", lambda *a, **kw: [])
        monkeypatch.setattr(
            ps, "post_discussion",
            lambda repo_id, *, title, description, repo_type, token: {
                "num": 1, "url": "https://hf.co/discussions/1", "title": title,
            },
        )

        ps.sync_peer_finding(
            repo_id="org/repo",
            challenge_id="paper3",
            agent_id="my-special-id",
            top_finding=self._finding(),
            iso_now="2026-05-16T00:00:00+00:00",
        )
        assert captured.get("my_agent_id") == "my-special-id"


# ---------------------------------------------------------------------------
# MCP dispatch
# ---------------------------------------------------------------------------


class TestMCPDispatch:
    def test_missing_challenge_id_returns_error(self, monkeypatch, tmp_path):
        from crucible.mcp.tools import TOOL_DISPATCH

        class _C:
            project_root = tmp_path
            hf_collab = type("H", (), {"leaderboard_repo": "org/repo"})()
            name = "test"

        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: _C())
        out = TOOL_DISPATCH["research_peer_sync"]({})
        assert "error" in out
        assert "challenge_id" in out["error"]

    def test_missing_repo_returns_error(self, monkeypatch, tmp_path):
        from crucible.mcp.tools import TOOL_DISPATCH

        class _C:
            project_root = tmp_path
            hf_collab = None
            name = "test"

        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: _C())
        out = TOOL_DISPATCH["research_peer_sync"]({"challenge_id": "c"})
        assert "error" in out
        assert "repo_id" in out["error"]

    def test_happy_path_via_mcp(self, monkeypatch, tmp_path):
        from crucible.mcp.tools import TOOL_DISPATCH

        class _Hf:
            leaderboard_repo = "org/peer-repo"

        class _C:
            project_root = tmp_path
            hf_collab = _Hf()
            name = "test-agent"
            research_state_file = "research_state.jsonl"

        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: _C())
        # Stub the peer-sync core so we don't hit HF.
        monkeypatch.setattr(
            "crucible.researcher.peer_sync.sync_peer_finding",
            lambda **kw: {
                "challenge_id": kw["challenge_id"],
                "agent_id": kw["agent_id"],
                "thread_num": 1,
                "thread_url": "https://hf.co/discussions/1",
                "posted_url": "https://hf.co/discussions/1",
                "peer_count": 0,
                "peers": [],
            },
        )
        out = TOOL_DISPATCH["research_peer_sync"]({
            "challenge_id": "paper3",
            "top_finding": {
                "title": "T", "body": "B", "category": "observation",
                "confidence": 0.9,
            },
        })
        assert out["challenge_id"] == "paper3"
        assert out["agent_id"] == "test-agent"
        assert out["peer_count"] == 0

    def test_agent_id_from_env_overrides_project(self, monkeypatch, tmp_path):
        from crucible.mcp.tools import TOOL_DISPATCH

        class _Hf:
            leaderboard_repo = "org/peer-repo"

        class _C:
            project_root = tmp_path
            hf_collab = _Hf()
            name = "project-name"
            research_state_file = "research_state.jsonl"

        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: _C())
        monkeypatch.setenv("CRUCIBLE_AGENT_ID", "from-env")
        captured = {}
        monkeypatch.setattr(
            "crucible.researcher.peer_sync.sync_peer_finding",
            lambda **kw: captured.update(kw) or {
                "challenge_id": kw["challenge_id"],
                "agent_id": kw["agent_id"],
                "thread_num": 0, "thread_url": "", "posted_url": "",
                "peer_count": 0, "peers": [],
            },
        )
        TOOL_DISPATCH["research_peer_sync"]({
            "challenge_id": "c",
            "top_finding": {"title": "T", "body": "B", "confidence": 0.5,
                            "category": "observation"},
        })
        assert captured["agent_id"] == "from-env"
