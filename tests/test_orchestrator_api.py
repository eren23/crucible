"""Unit tests for researcher/orchestrator_api.py — no LLM calls."""
from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from crucible.core.config import load_config
from crucible.core.errors import ResearcherError
from crucible.researcher import orchestrator_api as oa
from crucible.researcher.state import ResearchState


@pytest.fixture
def state_in_project(project_dir: Path) -> tuple[ResearchState, Path]:
    """Project dir (from conftest) with an empty ResearchState."""
    import os

    os.chdir(project_dir)  # load_config reads cwd
    (project_dir / "program.md").write_text("Focus on minimising val_loss.", encoding="utf-8")
    state_file = project_dir / "research_state.jsonl"
    state = ResearchState(state_file, budget_hours=5.0)
    return state, project_dir


# ---------------------------------------------------------------------------
# request_prompt
# ---------------------------------------------------------------------------


def test_request_prompt_rejects_unknown_stage(state_in_project):
    state, _ = state_in_project
    config = load_config()
    with pytest.raises(ResearcherError, match="Unknown stage"):
        oa.request_prompt("banana", config, state)  # type: ignore[arg-type]


def test_request_prompt_hypothesis_returns_schema(state_in_project):
    state, _ = state_in_project
    config = load_config()
    out = oa.request_prompt("hypothesis", config, state)
    assert out["stage"] == "hypothesis"
    assert "system" in out and out["system"]
    assert "user" in out and out["user"]
    assert out["schema"] == oa.HYPOTHESIS_RESPONSE_SCHEMA
    assert "state_snapshot" in out


def test_request_prompt_hypothesis_injects_focus_family(state_in_project):
    state, _ = state_in_project
    config = load_config()
    out = oa.request_prompt("hypothesis", config, state, focus_family="looped")
    assert "looped" in out["user"]
    assert "Focus Area" in out["user"]


def test_request_prompt_hypothesis_appends_extra_context(state_in_project):
    state, _ = state_in_project
    config = load_config()
    out = oa.request_prompt(
        "hypothesis", config, state, extra_context="UNIQUE_PROBE_STRING_42"
    )
    assert "UNIQUE_PROBE_STRING_42" in out["user"]


def test_request_prompt_reflection_empty_history(state_in_project):
    state, _ = state_in_project
    config = load_config()
    out = oa.request_prompt("reflection", config, state)
    assert out["stage"] == "reflection"
    # No data to reflect on yet — system is None, user is the "no data" notice
    assert out["schema"] is None
    assert "empty" in out["user"].lower() or "no completed" in out["user"].lower()


def test_request_prompt_reflection_with_history(state_in_project):
    state, _ = state_in_project
    state.record_result(
        experiment={"name": "baseline", "config": {"LR": "0.001"}, "pod_hours": 0.5},
        result={"name": "baseline", "val_loss": 1.42, "status": "completed"},
    )
    state.save()
    config = load_config()
    out = oa.request_prompt("reflection", config, state)
    assert out["schema"] == oa.REFLECTION_RESPONSE_SCHEMA
    assert "baseline" in out["user"]
    assert "val_loss" in out["user"]


def test_request_prompt_briefing_is_readonly(state_in_project):
    state, _ = state_in_project
    config = load_config()
    out = oa.request_prompt("briefing", config, state)
    assert out["stage"] == "briefing"
    assert out["system"] is None
    assert out["schema"] is None
    assert len(out["user"]) > 0  # Markdown body


# ---------------------------------------------------------------------------
# Schemas validate canned responses
# ---------------------------------------------------------------------------


def test_hypothesis_schema_accepts_canned_response():
    canned = {
        "hypotheses": [
            {
                "hypothesis": "Increase recurrence depth improves val_loss",
                "name": "deeper_recurrence",
                "expected_impact": 0.01,
                "confidence": 0.6,
                "config": {"MODEL_FAMILY": "looped", "RECURRENCE_STEPS": "12"},
                "rationale": "Deeper recurrence → more iterative refinement.",
                "family": "looped",
            }
        ],
    }
    jsonschema.validate(canned, oa.HYPOTHESIS_RESPONSE_SCHEMA)


def test_hypothesis_schema_rejects_non_string_config_value():
    bad = {
        "hypotheses": [
            {
                "hypothesis": "x",
                "config": {"LR": 0.001},  # should be string
            }
        ],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, oa.HYPOTHESIS_RESPONSE_SCHEMA)


def test_reflection_schema_accepts_canned_response():
    canned = {
        "beliefs": ["Deeper is better", "High LR diverges"],
        "surprises": ["RMSprop beat AdamW unexpectedly"],
        "promote": ["deeper_recurrence"],
        "kill": ["high_lr_bad"],
    }
    jsonschema.validate(canned, oa.REFLECTION_RESPONSE_SCHEMA)


def test_reflection_schema_accepts_empty():
    """All reflection fields are optional."""
    jsonschema.validate({}, oa.REFLECTION_RESPONSE_SCHEMA)


# ---------------------------------------------------------------------------
# submit_response
# ---------------------------------------------------------------------------


def test_submit_rejects_unknown_stage(state_in_project):
    state, _ = state_in_project
    config = load_config()
    with pytest.raises(ResearcherError, match="Unknown stage"):
        oa.submit_response("banana", {}, config, state)  # type: ignore[arg-type]


def test_submit_briefing_is_forbidden(state_in_project):
    state, _ = state_in_project
    config = load_config()
    with pytest.raises(ResearcherError, match="briefing"):
        oa.submit_response("briefing", {}, config, state)


def test_submit_hypothesis_roundtrip_dict(state_in_project):
    state, _ = state_in_project
    config = load_config()
    response = {
        "hypotheses": [
            {
                "hypothesis": "Wider hidden dim",
                "name": "wider",
                "expected_impact": 0.008,
                "confidence": 0.55,
                "config": {"MODEL_FAMILY": "baseline", "D_MODEL": "256"},
                "rationale": "Capacity bump.",
                "family": "baseline",
            },
            {
                "hypothesis": "Lower LR",
                "name": "lower_lr",
                "expected_impact": 0.003,
                "confidence": 0.4,
                "config": {"LR": "1e-4"},
                "rationale": "Training noise floor.",
                "family": "baseline",
            },
        ],
    }
    out = oa.submit_response("hypothesis", response, config, state)
    assert out["applied"] is True
    assert out["hypotheses_added"] == 2
    # State mutated
    state2 = ResearchState(state.state_file, budget_hours=5.0)
    assert len(state2.hypotheses) == 2
    assert {h["name"] for h in state2.hypotheses} == {"wider", "lower_lr"}


def test_submit_hypothesis_accepts_json_string(state_in_project):
    state, _ = state_in_project
    config = load_config()
    response_str = json.dumps(
        {"hypotheses": [{"hypothesis": "x", "config": {"MODEL_FAMILY": "baseline"}}]}
    )
    out = oa.submit_response("hypothesis", response_str, config, state)
    assert out["hypotheses_added"] == 1


def test_submit_hypothesis_coerces_non_string_config_values(state_in_project):
    """Validator stringifies values — we accept even loose responses."""
    state, _ = state_in_project
    config = load_config()
    response = {
        "hypotheses": [
            {
                "hypothesis": "x",
                "config": {"D_MODEL": 256, "LR": 0.001},  # ints/floats
            }
        ],
    }
    out = oa.submit_response("hypothesis", response, config, state)
    assert out["hypotheses_added"] == 1
    state2 = ResearchState(state.state_file, budget_hours=5.0)
    cfg = state2.hypotheses[0]["config"]
    assert cfg == {"D_MODEL": "256", "LR": "0.001"}


def test_submit_hypothesis_empty_list(state_in_project):
    state, _ = state_in_project
    config = load_config()
    out = oa.submit_response("hypothesis", {"hypotheses": []}, config, state)
    assert out["applied"] is False
    assert out["hypotheses_added"] == 0


def test_submit_reflection_roundtrip(state_in_project):
    state, _ = state_in_project
    config = load_config()
    response = {
        "beliefs": ["A is better than B", "C helps when D"],
        "surprises": ["E emerged"],
        "promote": ["exp_1"],
        "kill": ["exp_2"],
    }
    out = oa.submit_response("reflection", response, config, state)
    assert out["applied"] is True
    assert out["beliefs_updated"] == 2
    assert out["promote"] == 1
    assert out["kill"] == 1
    assert out["promote_names"] == ["exp_1"]
    assert out["kill_names"] == ["exp_2"]
    state2 = ResearchState(state.state_file, budget_hours=5.0)
    assert state2.beliefs == response["beliefs"]


def test_submit_reflection_invalid_response(state_in_project):
    state, _ = state_in_project
    config = load_config()
    with pytest.raises(ResearcherError, match="reflection"):
        oa.submit_response("reflection", 42, config, state)  # type: ignore[arg-type]


def test_request_then_submit_roundtrip(state_in_project):
    """Full round-trip: orchestrator asks for prompt, builds canned response per schema, submits."""
    state, _ = state_in_project
    config = load_config()

    prompt = oa.request_prompt("hypothesis", config, state)
    assert prompt["schema"] is not None

    canned = {
        "hypotheses": [
            {
                "hypothesis": "Test hypothesis",
                "name": "test_hyp",
                "expected_impact": 0.01,
                "confidence": 0.5,
                "config": {"MODEL_FAMILY": "baseline"},
                "rationale": "test",
                "family": "baseline",
            }
        ]
    }
    # Orchestrator-side: validate response matches the schema Crucible handed back.
    jsonschema.validate(canned, prompt["schema"])

    result = oa.submit_response("hypothesis", canned, config, state)
    assert result["applied"] is True
    assert result["hypotheses_added"] == 1
