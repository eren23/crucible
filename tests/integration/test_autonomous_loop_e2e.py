"""End-to-end smoke for autonomous_research_loop.

Plan's Part D promised one of these per phase. This is the first. Drives
the full session lifecycle (start → submit hypothesis → submit reflection
→ status → cancel) against a tmp project with canned orchestrator
responses. No real LLM, no fleet, no $.

What this test catches that unit tests miss:
- Real yaml-write-then-read seams (the bug Part G discovered)
- Real `_load_state` / config wiring (not a monkeypatched dict)
- Real stale-submit guard against a content-hashed snapshot
- Real cancel transitioning a live session to a terminal state

Marked `integration` so it can be deselected via `-m "not integration"`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from crucible.core.config import load_config
from crucible.core.errors import StaleSubmitError
from crucible.researcher import autonomous_session as autos

pytestmark = pytest.mark.integration


# A minimal but valid crucible.yaml — enough for ResearchState + autonomous
# sessions to spin up without RunPod, W&B, or HF round-trips.
MINIMAL_YAML = """\
name: e2e-loop
version: "0.3.0"
provider:
  type: ssh
  ssh_key: ~/.ssh/id_ed25519
data:
  source: huggingface
  repo_id: test/ds
  local_root: ./data
  manifest: manifest.json
training:
  - backend: torch
    script: train.py
presets:
  smoke:
    MAX_WALLCLOCK_SECONDS: "60"
    ITERATIONS: "200"
researcher:
  budget_hours: 0.1
  max_iterations: 2
  program_file: program.md
wandb:
  required: false
hf_collab:
  enabled: false
results_file: experiments.jsonl
fleet_results_file: experiments_fleet.jsonl
logs_dir: logs
nodes_file: nodes.json
research_state_file: research_state.jsonl
"""


def _canned_hypotheses() -> dict:
    return {
        "hypotheses": [
            {
                "name": "widen_mlp",
                "hypothesis": "Wider MLP improves loss.",
                "config": {"MLP_DIM": "1024", "MODEL_DIM": "256"},
                "rationale": "Capacity bottleneck.",
                "expected_impact": 0.05,
                "confidence": 0.6,
            },
            {
                "name": "add_dropout",
                "hypothesis": "Dropout reduces overfit.",
                "config": {"DROPOUT": "0.1", "MODEL_DIM": "256"},
                "rationale": "Overfit suspected.",
                "expected_impact": 0.03,
                "confidence": 0.5,
            },
        ]
    }


def _canned_reflection() -> dict:
    # The reflection schema expects beliefs + promote/kill — schema-level
    # validation is permissive, so a minimal payload exercises the apply
    # path without forcing us to mirror every field name.
    return {
        "beliefs": [
            {"belief": "Wider MLPs help when capacity-limited.", "confidence": 0.7}
        ],
        "promote": ["widen_mlp"],
        "kill": [],
        "next_hypothesis_focus": "Test wider MLP at fixed param count.",
    }


@pytest.fixture
def loop_project(tmp_path: Path) -> Path:
    (tmp_path / "crucible.yaml").write_text(MINIMAL_YAML, encoding="utf-8")
    (tmp_path / "logs").mkdir()
    (tmp_path / ".crucible").mkdir()
    (tmp_path / "experiments.jsonl").touch()
    (tmp_path / "experiments_fleet.jsonl").touch()
    (tmp_path / "nodes.json").write_text("[]", encoding="utf-8")
    return tmp_path


class TestAutonomousLoopE2E:
    def test_start_returns_orchestrator_contract_shape(self, loop_project: Path) -> None:
        config = load_config(loop_project / "crucible.yaml")
        resp = autos.action_start(config, iterations=2, tier="smoke")

        assert resp["session_id"]
        assert resp["stage"] == "hypothesis"
        assert resp["iteration"] == 0
        assert isinstance(resp["system"], str) and resp["system"]
        assert isinstance(resp["user"], str) and resp["user"]
        assert isinstance(resp["schema"], dict)
        assert "hypotheses" in resp["schema"].get("properties", {})
        snap = resp["state_snapshot"]
        assert {"history_len", "hypotheses_len", "content_hash"} <= set(snap.keys())

    def test_submit_advances_hypothesis_to_reflection(self, loop_project: Path) -> None:
        config = load_config(loop_project / "crucible.yaml")
        started = autos.action_start(config, iterations=2, tier="smoke")
        sid = started["session_id"]

        applied = autos.action_submit(
            config,
            session_id=sid,
            response=_canned_hypotheses(),
            state_snapshot=started["state_snapshot"],
        )

        assert applied["stage_applied"] == "hypothesis"
        assert applied["next_stage"] == "reflection"
        assert applied["session_status"] == "running"
        next_prompt = applied["next_prompt"]
        assert next_prompt["stage"] == "reflection"
        # Snapshot must have advanced — different content_hash from start.
        assert next_prompt["state_snapshot"]["hypotheses_len"] == 2

    def test_submit_reflection_completes_iteration(self, loop_project: Path) -> None:
        config = load_config(loop_project / "crucible.yaml")
        started = autos.action_start(config, iterations=2, tier="smoke")
        sid = started["session_id"]

        after_hyp = autos.action_submit(
            config,
            session_id=sid,
            response=_canned_hypotheses(),
            state_snapshot=started["state_snapshot"],
        )
        reflection_prompt = after_hyp["next_prompt"]

        after_refl = autos.action_submit(
            config,
            session_id=sid,
            response=_canned_reflection(),
            state_snapshot=reflection_prompt["state_snapshot"],
        )
        # First iteration's reflection submit advances to iteration 1 / hypothesis.
        assert after_refl["iterations_completed"] == 1
        assert after_refl["next_prompt"]["iteration"] == 1
        assert after_refl["next_prompt"]["stage"] == "hypothesis"

    def test_stale_snapshot_rejected(self, loop_project: Path) -> None:
        config = load_config(loop_project / "crucible.yaml")
        started = autos.action_start(config, iterations=2, tier="smoke")
        sid = started["session_id"]

        bogus = {
            "history_len": 999,
            "hypotheses_len": 999,
            "beliefs_len": 999,
            "findings_len": 999,
            "content_hash": "deadbeefdeadbeef",
        }
        with pytest.raises(StaleSubmitError) as exc:
            autos.action_submit(
                config,
                session_id=sid,
                response=_canned_hypotheses(),
                state_snapshot=bogus,
            )
        msg = str(exc.value)
        assert "deadbeef" in msg
        assert "Re-request the prompt" in msg

    def test_status_reflects_current_stage(self, loop_project: Path) -> None:
        config = load_config(loop_project / "crucible.yaml")
        started = autos.action_start(config, iterations=2, tier="smoke")
        sid = started["session_id"]
        autos.action_submit(
            config,
            session_id=sid,
            response=_canned_hypotheses(),
            state_snapshot=started["state_snapshot"],
        )

        status = autos.action_status(config, session_id=sid)
        assert status["status"] == "running"
        assert status["current_stage"] == "reflection"
        # Session yaml is durable.
        assert Path(status["yaml_path"]).exists()

    def test_cancel_returns_checkpoint(self, loop_project: Path) -> None:
        config = load_config(loop_project / "crucible.yaml")
        started = autos.action_start(config, iterations=2, tier="smoke")
        sid = started["session_id"]

        cancel = autos.action_cancel(config, session_id=sid, reason="e2e done")
        assert cancel["session_id"] == sid
        assert cancel["session_status"] == "canceled"
        assert Path(cancel["checkpoint_path"]).exists()
        assert cancel["already_terminal"] is False

    def test_cancel_idempotent_on_terminal(self, loop_project: Path) -> None:
        config = load_config(loop_project / "crucible.yaml")
        started = autos.action_start(config, iterations=2, tier="smoke")
        sid = started["session_id"]
        autos.action_cancel(config, session_id=sid)

        again = autos.action_cancel(config, session_id=sid, reason="redundant")
        assert again["already_terminal"] is True
        assert again["session_status"] == "canceled"

    def test_research_state_persists_hypotheses(self, loop_project: Path) -> None:
        config = load_config(loop_project / "crucible.yaml")
        started = autos.action_start(config, iterations=2, tier="smoke")
        autos.action_submit(
            config,
            session_id=started["session_id"],
            response=_canned_hypotheses(),
            state_snapshot=started["state_snapshot"],
        )

        state_file = loop_project / "research_state.jsonl"
        assert state_file.exists()
        events = [json.loads(line) for line in state_file.read_text().splitlines() if line]
        kinds = [e["kind"] for e in events]
        # Two hypotheses written; budget_adjustment from session creation.
        assert kinds.count("hypothesis") == 2

    def test_session_artifacts_persist(self, loop_project: Path) -> None:
        config = load_config(loop_project / "crucible.yaml")
        started = autos.action_start(config, iterations=1, tier="smoke")
        sid = started["session_id"]

        sess_dir = loop_project / ".crucible" / "autonomous_sessions"
        assert (sess_dir / f"{sid}.yaml").exists()
        assert (sess_dir / f"{sid}.jsonl").exists()

        events = [
            json.loads(line)
            for line in (sess_dir / f"{sid}.jsonl").read_text().splitlines()
            if line
        ]
        kinds = [e["event"] for e in events]
        assert "started" in kinds
        assert "stage_prompted" in kinds
