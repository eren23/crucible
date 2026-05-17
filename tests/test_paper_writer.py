"""Tests for crucible.researcher.paper_writer — Phase 4.1."""
from __future__ import annotations

from typing import Any

import pytest

from crucible.researcher.paper_writer import (
    PaperDraftError,
    build_paper_draft_prompt,
    gather_track_context,
    parse_paper_draft_response,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeHubStore:
    """Minimal HubStore stub for the paper-writer tests."""

    def __init__(self, *, findings=None, track=None):
        self._findings = findings or []
        self._track = track or {}

    def load_context_for_track(self, track_name, *, include_global=True, max_findings=50):
        return self._findings[:max_findings]

    def _read_track_yaml(self, track_name):
        return self._track


VALID_RESPONSE = {
    "title": "VICReg-Augmented JEPA",
    "abstract": (
        "We extend VICReg to JEPA predictors and show 2x improvement on "
        "the predictor effective rank across three seeds. The method "
        "preserves the encoder while regularizing the predictor's output "
        "to prevent collapse."
    ),
    "introduction": "Self-supervised learning has converged on predictor-encoder pairs...",
    "method": "We add a VICReg loss to the predictor's output, weight 0.1, ...",
    "results": "On 1000x5000 retrieval: MRR 0.42 → 0.57.",
    "discussion": "The result holds across three seeds and two pooling variants.",
    "limitations": "Only single-token edits; multi-step rollout still degrades.",
    "related_work": "BYOL, SimSiam, VICReg (Bardes et al. 2021).",
    "key_findings": [
        "Predictor-side VICReg lifts effective rank without breaking encoder",
        "Holds across three seeds",
    ],
}


# ---------------------------------------------------------------------------
# gather_track_context + build_paper_draft_prompt
# ---------------------------------------------------------------------------


class TestGatherAndBuild:
    def test_empty_track_still_produces_prompt(self):
        hub = _FakeHubStore(findings=[], track={"description": ""})
        context = gather_track_context(
            track_name="empty-track",
            hub_store=hub,
            leaderboard_rows=[],
            notes=[],
            hypotheses=[],
        )
        prompt = build_paper_draft_prompt(context)
        assert "system" in prompt
        assert "user" in prompt
        assert "schema" in prompt
        assert "sections" in prompt
        # Required sections list is part of the prompt envelope.
        assert "abstract" in prompt["sections"]
        # Task line shows up.
        assert "Required sections" in prompt["user"]

    def test_prompt_includes_findings_leaderboard_hypotheses_notes(self):
        findings = [
            {
                "id": "f1",
                "title": "VICReg lifts predictor rank",
                "body": "After 8K steps the rank stays >16.",
                "category": "observation",
                "confidence": 0.85,
                "_source_scope": "track",
                "source_experiments": ["paper3_pred_vicreg_output"],
            },
        ]
        leaderboard = [
            {"name": "paper3_pred_vicreg_output", "primary_metric": "MRR",
             "primary_value": 0.57, "steps_completed": 8000},
        ]
        hypotheses = [
            {
                "name": "v3_residual_mlp_bounded",
                "hypothesis": "Bounded residual predictor should help composition.",
                "expected_impact": 0.015, "status": "pending",
            },
        ]
        notes = [
            {"stage": "post-run", "body": "VICReg(0.1) worked across seeds."},
        ]
        hub = _FakeHubStore(findings=findings, track={
            "description": "Phase 3 paper on predictor-side VICReg.",
            "projects": ["codewm"],
        })
        context = gather_track_context(
            track_name="paper3-symbolic",
            hub_store=hub,
            leaderboard_rows=leaderboard,
            notes=notes,
            hypotheses=hypotheses,
        )
        prompt = build_paper_draft_prompt(context)
        user = prompt["user"]
        assert "VICReg lifts predictor rank" in user
        assert "paper3_pred_vicreg_output" in user
        assert "MRR" in user
        assert "v3_residual_mlp_bounded" in user
        assert "VICReg(0.1) worked across seeds." in user
        assert "Phase 3 paper on predictor-side VICReg." in user
        assert "codewm" in user


# ---------------------------------------------------------------------------
# parse_paper_draft_response
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_valid_dict_response_assembles_markdown(self):
        out = parse_paper_draft_response(VALID_RESPONSE, "paper3-symbolic")
        assert out["title"] == "VICReg-Augmented JEPA"
        assert out["sections"]["abstract"].startswith("We extend VICReg")
        assert out["key_findings"] == VALID_RESPONSE["key_findings"]
        md = out["markdown"]
        assert "# VICReg-Augmented JEPA" in md
        assert "## Abstract" in md
        assert "## Method" in md
        assert "## Results" in md
        assert "## Limitations" in md
        assert "## Related Work" in md
        assert "## Key Findings" in md

    def test_title_defaults_to_track_name(self):
        no_title = {k: v for k, v in VALID_RESPONSE.items() if k != "title"}
        out = parse_paper_draft_response(no_title, "my-track")
        assert out["title"] == "my-track"
        assert "# my-track" in out["markdown"]

    def test_missing_section_raises_typed_error(self):
        bad = dict(VALID_RESPONSE)
        del bad["limitations"]
        with pytest.raises(PaperDraftError, match="limitations"):
            parse_paper_draft_response(bad, "track")

    def test_empty_section_raises_typed_error(self):
        bad = dict(VALID_RESPONSE)
        bad["results"] = "   "
        with pytest.raises(PaperDraftError, match="results"):
            parse_paper_draft_response(bad, "track")

    def test_json_string_response_parses(self):
        import json
        out = parse_paper_draft_response(
            json.dumps(VALID_RESPONSE), "track",
        )
        assert "## Method" in out["markdown"]

    def test_non_dict_non_string_response_raises(self):
        with pytest.raises(PaperDraftError, match="must be dict"):
            parse_paper_draft_response([1, 2, 3], "track")

    def test_orchestrator_supplied_headers_get_demoted(self):
        """Phase 4 review fix: if a section body contains its own
        ``# Headline`` or ``## Sub``, those would collide with the
        renderer's structural headers. They're now demoted to ###
        so the document outline stays intact."""
        sneaky = dict(VALID_RESPONSE)
        sneaky["method"] = (
            "# Sneaky H1\n"
            "## Sneaky H2\n"
            "Body text under the heading.\n"
            "### Already deep — leave alone"
        )
        out = parse_paper_draft_response(sneaky, "track")
        md = out["markdown"]
        # Document still has exactly one H1 (the title) and the H2s
        # the renderer added.
        h1_count = sum(
            1 for line in md.splitlines()
            if line.startswith("# ") and not line.startswith("## ")
        )
        assert h1_count == 1, f"expected exactly 1 H1, got {h1_count}"
        # The sneaky headers got demoted.
        assert "### Sneaky H1" in md
        assert "### Sneaky H2" in md
        # And the already-deep ### header was left as-is (no double-demote).
        assert "### Already deep" in md

    def test_section_counts_present_in_rendered_markdown(self):
        """Each non-empty required section appears once in markdown."""
        out = parse_paper_draft_response(VALID_RESPONSE, "track")
        md = out["markdown"]
        assert md.count("## Introduction") == 1
        assert md.count("## Method") == 1
        assert md.count("## Discussion") == 1


# ---------------------------------------------------------------------------
# MCP dispatch
# ---------------------------------------------------------------------------


class TestMCPDispatch:
    def test_request_prompt_via_mcp(self, tmp_path, monkeypatch):
        """End-to-end: monkeypatch _get_config + _get_hub_store, ask for
        the prompt, verify shape."""
        from crucible.mcp import tools as _tools

        class _C:
            project_root = tmp_path
            store_dir = ".crucible"
            research_state_file = "research_state.jsonl"

        (tmp_path / ".crucible").mkdir()

        monkeypatch.setattr(_tools, "_get_config", lambda: _C())
        monkeypatch.setattr(
            _tools, "_get_hub_store",
            lambda: _FakeHubStore(
                findings=[{
                    "id": "f1", "title": "T1", "body": "B1",
                    "category": "observation", "confidence": 0.9,
                    "_source_scope": "track",
                }],
                track={"description": "test", "projects": []},
            ),
        )
        # Bypass leaderboard/notes/hypotheses by letting the tool's
        # try/except absorb the missing project state.
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["note_generate_paper_draft"]({
            "action": "request_prompt", "track_name": "test-track",
        })
        assert "system" in out
        assert "schema" in out
        assert "Findings" in out["user"]
        assert "T1" in out["user"]
        assert out["track_name"] == "test-track"

    def test_submit_via_mcp(self, monkeypatch, tmp_path):
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["note_generate_paper_draft"]({
            "action": "submit",
            "track_name": "test",
            "response": VALID_RESPONSE,
        })
        assert "markdown" in out
        assert "# VICReg-Augmented JEPA" in out["markdown"]
        assert out["section_counts"]["abstract"] > 0

    def test_submit_with_invalid_response_returns_error(self, monkeypatch):
        from crucible.mcp.tools import TOOL_DISPATCH
        bad = dict(VALID_RESPONSE)
        del bad["abstract"]
        out = TOOL_DISPATCH["note_generate_paper_draft"]({
            "action": "submit",
            "track_name": "test",
            "response": bad,
        })
        assert "error" in out
        assert "abstract" in out["error"]

    def test_missing_track_name_returns_error(self):
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["note_generate_paper_draft"]({
            "action": "request_prompt"
        })
        assert "error" in out
        assert "track_name" in out["error"]

    def test_unknown_action_returns_error(self):
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["note_generate_paper_draft"]({
            "action": "bogus", "track_name": "x",
        })
        assert "error" in out
        assert "Unknown action" in out["error"]
