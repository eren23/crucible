"""Tests for the Optuna bridge — Phase 3.4."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Config validation (no Optuna needed for these — they fail before import)
# ---------------------------------------------------------------------------


class TestSpecValidation:
    """These run even when optuna is absent — validation happens before
    Optuna is imported, so the spec gates catch errors fast."""

    def test_missing_optuna_raises_import_error(self, monkeypatch):
        # Simulate optuna not being installed.
        from crucible.training import hpo_bridge

        # Hide optuna from import. We do this at module level so
        # _build_optuna_study's lazy import fails.
        monkeypatch.setitem(sys.modules, "optuna", None)
        with pytest.raises(hpo_bridge.HPOImportError, match="Optuna isn't installed"):
            hpo_bridge.HPOStudy(
                name="s",
                params={"LR": {"type": "float", "low": 1e-5, "high": 1e-2}},
            )

    def test_empty_params_raises(self, monkeypatch):
        from crucible.training import hpo_bridge
        # Force optuna absent so we don't even reach Optuna code.
        monkeypatch.setitem(sys.modules, "optuna", None)
        with pytest.raises(hpo_bridge.HPOConfigError, match="non-empty dict"):
            hpo_bridge.HPOStudy(name="s", params={})

    def test_unknown_distribution_type_raises(self, monkeypatch):
        from crucible.training import hpo_bridge
        monkeypatch.setitem(sys.modules, "optuna", None)
        with pytest.raises(hpo_bridge.HPOConfigError, match="unknown type"):
            hpo_bridge.HPOStudy(
                name="s",
                params={"X": {"type": "wat", "low": 0, "high": 1}},
            )

    def test_float_without_bounds_raises(self, monkeypatch):
        from crucible.training import hpo_bridge
        monkeypatch.setitem(sys.modules, "optuna", None)
        with pytest.raises(hpo_bridge.HPOConfigError, match="requires 'low' and 'high'"):
            hpo_bridge.HPOStudy(
                name="s",
                params={"X": {"type": "float"}},
            )

    def test_inverted_bounds_raises(self, monkeypatch):
        from crucible.training import hpo_bridge
        monkeypatch.setitem(sys.modules, "optuna", None)
        with pytest.raises(hpo_bridge.HPOConfigError, match="must be <"):
            hpo_bridge.HPOStudy(
                name="s",
                params={"X": {"type": "float", "low": 10, "high": 1}},
            )

    def test_categorical_without_choices_raises(self, monkeypatch):
        from crucible.training import hpo_bridge
        monkeypatch.setitem(sys.modules, "optuna", None)
        with pytest.raises(hpo_bridge.HPOConfigError, match="non-empty\n? *'choices'"):
            hpo_bridge.HPOStudy(
                name="s",
                params={"X": {"type": "categorical", "choices": []}},
            )

    def test_invalid_direction_raises(self, monkeypatch):
        from crucible.training import hpo_bridge
        monkeypatch.setitem(sys.modules, "optuna", None)
        with pytest.raises(hpo_bridge.HPOConfigError, match="direction"):
            hpo_bridge.HPOStudy(
                name="s",
                params={"X": {"type": "float", "low": 0, "high": 1}},
                direction="random",
            )


# ---------------------------------------------------------------------------
# Optuna round-trip (skipped when not installed)
# ---------------------------------------------------------------------------


# Pytest marker that skips the round-trip class if optuna is missing.
_OPTUNA_AVAILABLE = True
try:
    import optuna  # noqa: F401
except ImportError:
    _OPTUNA_AVAILABLE = False

pytestmark_optuna = pytest.mark.skipif(
    not _OPTUNA_AVAILABLE, reason="optuna not installed"
)


@pytestmark_optuna
class TestAskTellRoundtrip:
    def test_ask_returns_trial_id_and_params_dict(self, tmp_path):
        from crucible.training.hpo_bridge import HPOStudy
        study = HPOStudy(
            name="ask-test",
            params={
                "LR": {"type": "log_float", "low": 1e-5, "high": 1e-2},
                "BATCH_SIZE": {"type": "categorical", "choices": [16, 32, 64]},
            },
            sampler="random",
            seed=42,
            storage_dir=tmp_path,
        )
        out = study.ask()
        assert "trial_id" in out
        assert isinstance(out["trial_id"], int)
        assert set(out["params"].keys()) == {"LR", "BATCH_SIZE"}
        # All emitted params are strings (Crucible env-var contract).
        assert all(isinstance(v, str) for v in out["params"].values())

    def test_tell_persists_and_updates_best(self, tmp_path):
        from crucible.training.hpo_bridge import HPOStudy
        study = HPOStudy(
            name="tell-test",
            params={"LR": {"type": "float", "low": 0.0, "high": 1.0}},
            sampler="random",
            seed=42,
            storage_dir=tmp_path,
            direction="minimize",
        )
        # Run 3 trials, telling smaller values.
        scores = []
        for s in (0.5, 0.3, 0.7):
            t = study.ask()
            study.tell(t["trial_id"], s)
            scores.append(s)

        best = study.best()
        assert best is not None
        assert best["score"] == 0.3

        # Persisted JSON exists + has the 3 trials.
        import json
        data = json.loads((tmp_path / "tell-test.json").read_text())
        assert len(data["trials"]) == 3
        assert all(t["status"] == "complete" for t in data["trials"])

    def test_tell_unknown_trial_raises(self, tmp_path):
        from crucible.training.hpo_bridge import HPOStudy, HPOStateError
        study = HPOStudy(
            name="bad-tell",
            params={"LR": {"type": "float", "low": 0, "high": 1}},
            sampler="random",
            storage_dir=tmp_path,
        )
        with pytest.raises(HPOStateError, match="Unknown trial_id"):
            study.tell(9999, 0.5)

    def test_tell_double_raises(self, tmp_path):
        from crucible.training.hpo_bridge import HPOStudy, HPOStateError
        study = HPOStudy(
            name="dup-tell",
            params={"LR": {"type": "float", "low": 0, "high": 1}},
            sampler="random",
            storage_dir=tmp_path,
        )
        t = study.ask()
        study.tell(t["trial_id"], 0.5)
        with pytest.raises(HPOStateError, match="already finalized"):
            study.tell(t["trial_id"], 0.3)

    def test_failed_trial_does_not_update_best(self, tmp_path):
        from crucible.training.hpo_bridge import HPOStudy
        study = HPOStudy(
            name="fail-test",
            params={"LR": {"type": "float", "low": 0, "high": 1}},
            sampler="random",
            storage_dir=tmp_path,
        )
        t = study.ask()
        study.tell(t["trial_id"], 0.0, status="failed")
        assert study.best() is None  # No completed trials

    def test_history_returns_all_records(self, tmp_path):
        from crucible.training.hpo_bridge import HPOStudy
        study = HPOStudy(
            name="hist",
            params={"LR": {"type": "float", "low": 0, "high": 1}},
            sampler="random",
            storage_dir=tmp_path,
        )
        for _ in range(3):
            t = study.ask()
            study.tell(t["trial_id"], 0.5)
        hist = study.history()
        assert len(hist) == 3
        assert all("trial_id" in r for r in hist)
        assert all(r["status"] == "complete" for r in hist)

    def test_maximize_direction(self, tmp_path):
        from crucible.training.hpo_bridge import HPOStudy
        study = HPOStudy(
            name="max",
            params={"LR": {"type": "float", "low": 0, "high": 1}},
            direction="maximize",
            sampler="random",
            storage_dir=tmp_path,
        )
        for s in (0.5, 0.9, 0.3):
            t = study.ask()
            study.tell(t["trial_id"], s)
        assert study.best()["score"] == 0.9
