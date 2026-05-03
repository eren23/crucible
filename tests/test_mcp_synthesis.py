"""Tests for the design_synthesize_from_findings MCP tool.

Pulls findings from a real (temp) hub, mines pairs per the requested
policy, and returns one orchestrator-shaped prompt bundle per pair.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from crucible.core.hub import HubStore
from crucible.core.finding import new_finding


@pytest.fixture
def hub_with_findings(tmp_path: Path, monkeypatch) -> HubStore:
    hub_dir = tmp_path / "hub"
    hub = HubStore.init(hub_dir=hub_dir, name="test-hub")
    monkeypatch.setenv("CRUCIBLE_HUB_DIR", str(hub_dir))

    hub.create_track("attention-variants", description="attn track")
    hub.create_track("optimizers", description="optim track")

    hub.store_finding(
        new_finding(
            title="LR warmup matters",
            body="200 warmup steps consistently helps long-context.",
            scope="track",
            track="optimizers",
            confidence=0.8,
            source_project="proj_a",
        ),
        scope="track",
        track="optimizers",
    )
    hub.store_finding(
        new_finding(
            title="Muon beats AdamW on matmul layers",
            body="Muon optimizer 12% lower val_bpb at same step budget.",
            scope="track",
            track="optimizers",
            confidence=0.9,
            source_project="proj_b",
        ),
        scope="track",
        track="optimizers",
    )
    hub.store_finding(
        new_finding(
            title="Sliding-window attention reduces compute 4x",
            body="Window=512 with stride matches full-attn perplexity.",
            scope="track",
            track="attention-variants",
            confidence=0.75,
            source_project="proj_c",
        ),
        scope="track",
        track="attention-variants",
    )
    hub.store_finding(
        new_finding(
            title="Rotary embeddings extrapolate beyond train length",
            body="RoPE supports 4x context extension at eval time.",
            scope="track",
            track="attention-variants",
            confidence=0.85,
            source_project="proj_d",
        ),
        scope="track",
        track="attention-variants",
    )
    return hub


class TestDesignSynthesizeFromFindings:
    def test_returns_one_bundle_per_pair(self, hub_with_findings, monkeypatch):
        from crucible.mcp.tools import design_synthesize_from_findings

        # Stub _get_config so the tool doesn't need a real project.
        from crucible.core.config import ProjectConfig
        monkeypatch.setattr(
            "crucible.mcp.tools._get_config",
            lambda: ProjectConfig(name="x"),
        )

        out = design_synthesize_from_findings({
            "k": 2,
            "policy": "random",
            "scope": "track",
            "track": "optimizers",
            "seed": 0,
        })
        assert "error" not in out, out
        assert "pairs" in out
        assert len(out["pairs"]) >= 1
        bundle = out["pairs"][0]
        assert set(bundle.keys()) >= {
            "system", "user", "schema", "parent_finding_ids",
        }
        assert len(bundle["parent_finding_ids"]) == 2

    def test_same_track_policy_returns_within_track_pairs(
        self, hub_with_findings, monkeypatch
    ):
        from crucible.mcp.tools import design_synthesize_from_findings
        from crucible.core.config import ProjectConfig

        monkeypatch.setattr(
            "crucible.mcp.tools._get_config",
            lambda: ProjectConfig(name="x"),
        )
        out = design_synthesize_from_findings({
            "k": 5,
            "policy": "same_track",
            "scope": "global",
            "seed": 0,
        })
        assert "error" not in out, out
        # Each pair must carry both parent IDs and reference findings from the
        # same track (the fixture has 2 each in 'optimizers' and 'attention-variants').
        assert out["pairs"], "expected at least one same-track pair"
        for bundle in out["pairs"]:
            assert len(bundle["parent_finding_ids"]) == 2
            assert all(bundle["parent_finding_ids"])
            # 'system'+'user' both populated, no leakage of empty bundles.
            assert bundle["system"] and bundle["user"]

    def test_cross_track_policy_yields_only_cross_track_pairs(
        self, hub_with_findings, monkeypatch
    ):
        from crucible.mcp.tools import design_synthesize_from_findings
        from crucible.core.config import ProjectConfig

        monkeypatch.setattr(
            "crucible.mcp.tools._get_config",
            lambda: ProjectConfig(name="x"),
        )
        out = design_synthesize_from_findings({
            "k": 5,
            "policy": "cross_track",
            "scope": "global",
            "seed": 0,
        })
        assert "error" not in out, out
        # Fixture has tracks 'optimizers' and 'attention-variants'; every pair
        # the tool returns must straddle the two.
        assert out["pairs"]

    def test_returns_error_when_pool_too_small(
        self, tmp_path: Path, monkeypatch
    ):
        from crucible.mcp.tools import design_synthesize_from_findings
        from crucible.core.config import ProjectConfig

        hub_dir = tmp_path / "empty-hub"
        HubStore.init(hub_dir=hub_dir, name="empty")
        monkeypatch.setenv("CRUCIBLE_HUB_DIR", str(hub_dir))
        monkeypatch.setattr(
            "crucible.mcp.tools._get_config",
            lambda: ProjectConfig(name="x"),
        )

        out = design_synthesize_from_findings({"k": 2, "policy": "random"})
        assert "error" in out
        assert "at least 2" in out["error"].lower() or "findings" in out["error"].lower()

    def test_returns_error_when_hub_not_initialized(
        self, tmp_path: Path, monkeypatch
    ):
        from crucible.mcp.tools import design_synthesize_from_findings
        from crucible.core.config import ProjectConfig

        # Point CRUCIBLE_HUB_DIR at a non-existent dir → not initialized.
        monkeypatch.setenv("CRUCIBLE_HUB_DIR", str(tmp_path / "no-hub"))
        monkeypatch.setattr(
            "crucible.mcp.tools._get_config",
            lambda: ProjectConfig(name="x"),
        )

        out = design_synthesize_from_findings({"k": 1, "policy": "random"})
        assert "error" in out

    def test_rejects_project_scope_with_clear_message(
        self, hub_with_findings, monkeypatch
    ):
        from crucible.mcp.tools import design_synthesize_from_findings
        from crucible.core.config import ProjectConfig

        monkeypatch.setattr(
            "crucible.mcp.tools._get_config",
            lambda: ProjectConfig(name="x"),
        )
        out = design_synthesize_from_findings({
            "k": 1, "policy": "random", "scope": "project",
        })
        assert "error" in out
        assert "scope" in out["error"].lower() or "project" in out["error"].lower()

    def test_tag_filter_is_pair_level_or_match(
        self, tmp_path: Path, monkeypatch
    ):
        from crucible.mcp.tools import design_synthesize_from_findings
        from crucible.core.config import ProjectConfig

        hub_dir = tmp_path / "hub"
        hub = HubStore.init(hub_dir=hub_dir, name="t")
        monkeypatch.setenv("CRUCIBLE_HUB_DIR", str(hub_dir))
        hub.create_track("optimizers", description="o")

        # One tagged finding + one untagged. Tag filter should still pull
        # them as an eligible pair (OR-semantics).
        from crucible.core.finding import new_finding
        hub.store_finding(
            new_finding(title="A", body="a body", scope="track",
                        track="optimizers", tags=["optim"]),
            scope="track", track="optimizers",
        )
        hub.store_finding(
            new_finding(title="B", body="b body", scope="track",
                        track="optimizers", tags=[]),
            scope="track", track="optimizers",
        )

        monkeypatch.setattr(
            "crucible.mcp.tools._get_config",
            lambda: ProjectConfig(name="x"),
        )
        out = design_synthesize_from_findings({
            "k": 5,
            "policy": "random",
            "scope": "track",
            "track": "optimizers",
            "tags": ["optim"],
        })
        assert "error" not in out, out
        assert len(out["pairs"]) == 1
