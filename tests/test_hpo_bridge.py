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

    def test_load_replays_trials_through_optuna(self, tmp_path):
        """G.4-style live-test fix: reloading a study from disk used to
        keep _trial_records but leave Optuna's study at trial 0. Then
        the next ask returned trial_id=0 and tell with the persisted
        ids raised "Unknown trial_id". HPOStudy.load now replays
        trials via Optuna's create_trial + add_trial so the sampler's
        belief is current and trial_ids stay monotonic."""
        from crucible.training.hpo_bridge import HPOStudy

        # Run a fresh study and persist 3 trials.
        first = HPOStudy(
            name="resume-test",
            params={"LR": {"type": "float", "low": 0.0, "high": 1.0}},
            sampler="random", seed=42, storage_dir=tmp_path,
        )
        for s in (0.5, 0.3, 0.7):
            t = first.ask()
            first.tell(t["trial_id"], s)

        # Reload from disk.
        reloaded = HPOStudy.load(
            name="resume-test", storage_dir=tmp_path,
            sampler="random", seed=99,
        )

        # Best survives the reload (read from _trial_records).
        best = reloaded.best()
        assert best is not None
        assert best["score"] == 0.3

        # Next ask returns trial_id >= 3 — Optuna's counter advanced
        # because we replayed the 3 persisted trials via add_trial.
        next_trial = reloaded.ask()
        assert next_trial["trial_id"] >= 3, (
            f"sampler counter not replayed; got trial_id={next_trial['trial_id']}"
        )

        # And tell on the new id works.
        reloaded.tell(next_trial["trial_id"], 0.1)
        assert reloaded.best()["score"] == 0.1

    def test_load_missing_file_raises(self, tmp_path):
        from crucible.training.hpo_bridge import HPOStudy, HPOConfigError
        with pytest.raises(HPOConfigError, match="No persisted HPO study"):
            HPOStudy.load(name="nonexistent", storage_dir=tmp_path)

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


# ---------------------------------------------------------------------------
# MCP-level review-fix tests
# ---------------------------------------------------------------------------


@pytestmark_optuna
class TestMCPLayerFixes:
    """Tests for the review-driven fixes to the MCP wrappers."""

    def _fake_config(self, tmp_path):
        class _C:
            project_root = tmp_path
        return _C()

    def test_hpo_create_study_idempotency_guard(self, tmp_path, monkeypatch):
        from crucible.mcp.tools import TOOL_DISPATCH, _HPO_STUDY_CACHE
        _HPO_STUDY_CACHE.clear()
        monkeypatch.setattr(
            "crucible.mcp.tools._get_config",
            lambda: self._fake_config(tmp_path),
        )
        args = {
            "name": "dup-test",
            "params": {"LR": {"type": "float", "low": 0, "high": 1}},
            "sampler": "random", "seed": 1,
        }
        first = TOOL_DISPATCH["hpo_create_study"](args)
        assert "persisted_path" in first
        # Second call without force = idempotency error.
        second = TOOL_DISPATCH["hpo_create_study"](args)
        assert "error" in second
        assert "already exists" in second["error"]
        assert second.get("already_exists") is True
        # With force=True it overwrites.
        forced = TOOL_DISPATCH["hpo_create_study"](dict(args, force=True))
        assert "persisted_path" in forced
        assert "error" not in forced

    def test_hpo_tell_result_missing_trial_id_is_clean_error(
        self, tmp_path, monkeypatch
    ):
        from crucible.mcp.tools import TOOL_DISPATCH, _HPO_STUDY_CACHE
        _HPO_STUDY_CACHE.clear()
        monkeypatch.setattr(
            "crucible.mcp.tools._get_config",
            lambda: self._fake_config(tmp_path),
        )
        TOOL_DISPATCH["hpo_create_study"]({
            "name": "bad-args",
            "params": {"LR": {"type": "float", "low": 0, "high": 1}},
        })
        # Missing trial_id.
        out = TOOL_DISPATCH["hpo_tell_result"]({
            "name": "bad-args", "score": 0.5,
        })
        assert "error" in out
        assert "trial_id" in out["error"]

    def test_hpo_tell_result_non_numeric_score_is_clean_error(
        self, tmp_path, monkeypatch
    ):
        from crucible.mcp.tools import TOOL_DISPATCH, _HPO_STUDY_CACHE
        _HPO_STUDY_CACHE.clear()
        monkeypatch.setattr(
            "crucible.mcp.tools._get_config",
            lambda: self._fake_config(tmp_path),
        )
        TOOL_DISPATCH["hpo_create_study"]({
            "name": "score-coerce",
            "params": {"LR": {"type": "float", "low": 0, "high": 1}},
        })
        out = TOOL_DISPATCH["hpo_tell_result"]({
            "name": "score-coerce", "trial_id": 0, "score": "not-a-number",
        })
        assert "error" in out
        assert "score" in out["error"]

    def test_h14_concurrent_ask_serialized_no_trial_collision(
        self, tmp_path, monkeypatch
    ):
        """H.1.4: two concurrent hpo_ask_trial calls used to both miss
        the unsynchronized cache, both build a fresh HPOStudy, and the
        second write would silently discard the first's asked-but-not-
        told trial. Now serialized via _HPO_STUDY_CACHE_LOCK — each
        ask returns a distinct trial_id and no in-memory state is
        lost."""
        import threading
        from crucible.mcp.tools import TOOL_DISPATCH, _HPO_STUDY_CACHE
        _HPO_STUDY_CACHE.clear()
        monkeypatch.setattr(
            "crucible.mcp.tools._get_config",
            lambda: self._fake_config(tmp_path),
        )
        TOOL_DISPATCH["hpo_create_study"]({
            "name": "race",
            "params": {"LR": {"type": "float", "low": 0, "high": 1}},
            "sampler": "random", "seed": 1,
        })

        results: list[dict] = []
        errors: list[Exception] = []

        def worker():
            try:
                out = TOOL_DISPATCH["hpo_ask_trial"]({"name": "race"})
                results.append(out)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"concurrent asks raised: {errors}"
        # Each ask must yield a distinct trial_id — if the cache lock
        # was missing, two would collide on trial_id=0.
        trial_ids = [r["trial_id"] for r in results]
        assert len(set(trial_ids)) == len(trial_ids), (
            f"duplicate trial_ids under concurrent ask: {trial_ids}"
        )
        assert len(results) == 8

    def test_hpo_cross_process_resume_via_mcp(self, tmp_path, monkeypatch):
        """End-to-end: create + 3 trials + clear cache + tell continues
        without 'Unknown trial_id'."""
        from crucible.mcp.tools import TOOL_DISPATCH, _HPO_STUDY_CACHE
        _HPO_STUDY_CACHE.clear()
        monkeypatch.setattr(
            "crucible.mcp.tools._get_config",
            lambda: self._fake_config(tmp_path),
        )
        TOOL_DISPATCH["hpo_create_study"]({
            "name": "resume",
            "params": {"LR": {"type": "float", "low": 0, "high": 1}},
            "sampler": "random", "seed": 1,
        })
        for s in (0.5, 0.3, 0.7):
            t = TOOL_DISPATCH["hpo_ask_trial"]({"name": "resume"})
            TOOL_DISPATCH["hpo_tell_result"]({
                "name": "resume", "trial_id": t["trial_id"], "score": s,
            })

        # Simulate process restart by clearing the cache.
        _HPO_STUDY_CACHE.clear()

        # Next ask must return a trial_id > 2 (Optuna's counter caught up).
        nxt = TOOL_DISPATCH["hpo_ask_trial"]({"name": "resume"})
        assert nxt["trial_id"] > 2, (
            f"cross-process resume left Optuna counter at 0; got "
            f"trial_id={nxt['trial_id']}"
        )
        # And tell still works.
        out = TOOL_DISPATCH["hpo_tell_result"]({
            "name": "resume", "trial_id": nxt["trial_id"], "score": 0.1,
        })
        assert out.get("ok") is True
        assert out["best"]["score"] == 0.1


# ---------------------------------------------------------------------------
# code_mutation default registration (review fix)
# ---------------------------------------------------------------------------


def test_code_mutation_policy_default_name():
    """Stub is registered under the canonical 'code_mutation' name so
    callers can use get_code_mutation_policy() (no args) to reach it.
    """
    from crucible.researcher.code_mutation import (
        get_code_mutation_policy,
        list_code_mutation_policies,
        StubCodeMutationPolicy,
    )
    assert "code_mutation" in list_code_mutation_policies()
    # Legacy alias still works.
    assert "stub" in list_code_mutation_policies()
    # Default name resolves.
    assert isinstance(get_code_mutation_policy(), StubCodeMutationPolicy)
    assert isinstance(get_code_mutation_policy("code_mutation"), StubCodeMutationPolicy)
    assert isinstance(get_code_mutation_policy("stub"), StubCodeMutationPolicy)
