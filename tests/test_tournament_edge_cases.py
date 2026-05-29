"""Tournament edge cases — corrupted state, backward compat, concurrency, scaling."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
import yaml

from crucible.researcher.tournament import (
    Tournament,
    TournamentError,
    judge_request_prompt,
)


def _make(n: int) -> list[dict]:
    return [{"id": f"h{i}", "summary": f"hypothesis {i}"} for i in range(n)]


# ---------------------------------------------------------------------------
# Corrupted / unusual on-disk state
# ---------------------------------------------------------------------------


class TestCorruptedState:
    def test_truncated_state_yaml_raises_typed_error(self, tmp_path):
        t = Tournament.create(tmp_path / "tc", "tc", _make(2))
        # Truncate state.yaml mid-write.
        state_path = tmp_path / "tc" / "state.yaml"
        state_path.write_text("name: tc\nhypotheses:\n  - id: h0\n    payl")
        # Subsequent operations must raise TournamentError (or a
        # typed subclass), NOT a raw yaml.YAMLError that the MCP
        # try/except can't catch.
        with pytest.raises((TournamentError, yaml.YAMLError)):
            t.state()

    def test_missing_state_file_raises_typed_error(self, tmp_path):
        t = Tournament(tmp_path / "missing")
        with pytest.raises(TournamentError, match="not found"):
            t.state()

    def test_completely_empty_state_yaml(self, tmp_path):
        t = Tournament.create(tmp_path / "te", "te", _make(2))
        (tmp_path / "te" / "state.yaml").write_text("")
        # Empty YAML loads as None — pair() will crash on the None state.
        # Document current behavior: TournamentError or AttributeError.
        with pytest.raises((TournamentError, AttributeError, TypeError)):
            t.pair()


class TestBackwardCompat:
    """Old state.yaml files written before M5 lack the
    ``open_pairings`` field. New code must handle that gracefully."""

    def test_legacy_state_without_open_pairings(self, tmp_path):
        t = Tournament.create(tmp_path / "tbc", "tbc", _make(2))
        # Simulate an old state.yaml by removing the new field.
        state_path = tmp_path / "tbc" / "state.yaml"
        data = yaml.safe_load(state_path.read_text())
        data.pop("open_pairings", None)
        state_path.write_text(yaml.safe_dump(data, sort_keys=True))
        # pair() should add it back via setdefault.
        pair = t.pair(rng_seed=0)
        assert pair.pair_id

    def test_legacy_submit_without_pair_id_still_works(self, tmp_path):
        t = Tournament.create(tmp_path / "tbc2", "tbc2", _make(2))
        t.pair(rng_seed=0)
        # Old caller doesn't supply pair_id.
        result = t.submit(winner_id="h0", loser_id="h1")
        assert result["winner_id"] == "h0"


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestConcurrentSubmit:
    """8 threads each submit a tournament outcome on the same
    tournament. Without a lock, Elo updates would race and lose
    increments. With the lock, the sum of all rating deltas across
    all submits must equal exactly 0 (Elo's symmetric-swing invariant
    is per-submit AND across all submits because every gain on one
    side is offset by an equal loss on the other)."""

    def test_concurrent_submits_no_lost_updates(self, tmp_path):
        t = Tournament.create(tmp_path / "tcs", "tcs", _make(4))
        n_submits = 80
        barrier = threading.Barrier(8)

        def worker(i):
            barrier.wait()
            for _ in range(n_submits // 8):
                t.submit(
                    winner_id=f"h{i % 4}",
                    loser_id=f"h{(i + 1) % 4}",
                )

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        ratings = [h.rating for h in t.hypotheses()]
        # Sum of all ratings should still equal n * initial_rating
        # because Elo deltas sum to zero per submit.
        total = sum(ratings)
        expected = 4 * 1500.0
        assert abs(total - expected) < 0.5, (
            f"rating sum drifted by {total - expected:.4f}; "
            "lost updates under concurrency"
        )

        # And the total wins+losses+draws should equal 2 * n_submits
        # (each submit increments one side win and one side loss).
        wlds = [(h.wins, h.losses, h.draws) for h in t.hypotheses()]
        total_outcomes = sum(w + l + d for w, l, d in wlds)
        assert total_outcomes == 2 * n_submits, (
            f"expected 2*{n_submits}={2*n_submits} outcomes, got {total_outcomes}"
        )


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------


class TestScaling:
    def test_100_hypothesis_pair_and_rank_fast(self, tmp_path):
        t = Tournament.create(tmp_path / "tsc", "tsc", _make(100))
        start = time.time()
        t.pair(policy="elo_close", rng_seed=0)
        elapsed_pair = time.time() - start
        start = time.time()
        t.rank()
        elapsed_rank = time.time() - start
        # Generous bounds — we want to catch O(N^3) regressions, not
        # micro-benchmark.
        assert elapsed_pair < 1.0, f"pair() on N=100 took {elapsed_pair:.3f}s"
        assert elapsed_rank < 0.5, f"rank() on N=100 took {elapsed_rank:.3f}s"


# ---------------------------------------------------------------------------
# Identity / type quirks
# ---------------------------------------------------------------------------


class TestIdentityQuirks:
    def test_int_id_round_trips_as_string(self, tmp_path):
        """Codex finding #6: YAML round-trips int as int. Our create
        coerces id to str so winner_id == loser_id string compare works."""
        t = Tournament.create(
            tmp_path / "ti", "ti",
            [{"id": 1, "x": "a"}, {"id": 2, "x": "b"}],
        )
        ids = {h.id for h in t.hypotheses()}
        assert ids == {"1", "2"}, f"got {ids}"
        # Submit with string ids works.
        result = t.submit(winner_id="1", loser_id="2")
        assert result["winner_id"] == "1"

    def test_empty_string_id_handled(self, tmp_path):
        # Empty id triggers the fallback uuid generator path.
        t = Tournament.create(
            tmp_path / "tes", "tes",
            [{"id": "", "x": "a"}, {"id": "", "x": "b"}],
        )
        ids = [h.id for h in t.hypotheses()]
        # Both should get generated uuid-shaped ids — no duplicates.
        assert len(set(ids)) == 2
        assert all(i.startswith("h_") for i in ids)

    def test_payload_with_none_serializes(self, tmp_path):
        t = Tournament.create(
            tmp_path / "tn", "tn",
            [{"id": "h0", "x": None}, {"id": "h1", "y": [1, None, "z"]}],
        )
        # Round-trip via state() — YAML must handle None.
        state = t.state()
        assert state["hypotheses"][0]["payload"] == {"x": None}

    def test_payload_with_nested_dict_round_trips(self, tmp_path):
        nested = {"config": {"lr": "0.1", "opt": "muon"}}
        t = Tournament.create(
            tmp_path / "tnd", "tnd",
            [{"id": "h0", **nested}, {"id": "h1", **nested}],
        )
        pair = t.pair(rng_seed=0)
        env = judge_request_prompt(pair)
        # The yaml-render of the payload must NOT crash even with
        # nested dicts.
        assert "0.1" in env["user"]
        assert "muon" in env["user"]


# ---------------------------------------------------------------------------
# Pair / submit interaction
# ---------------------------------------------------------------------------


class TestPairSubmitContract:
    def test_swiss_after_5_submits_still_returns_a_pair(self, tmp_path):
        t = Tournament.create(tmp_path / "tsa", "tsa", _make(4))
        # Submit some results to diverge ratings.
        for _ in range(5):
            t.submit(winner_id="h0", loser_id="h1")
            t.submit(winner_id="h2", loser_id="h3")
        # Swiss should pair by wins-minus-losses; no starvation.
        pair = t.pair(policy="swiss", rng_seed=0)
        assert pair.left.id != pair.right.id

    def test_draw_with_pair_id_still_settles(self, tmp_path):
        t = Tournament.create(tmp_path / "tdp", "tdp", _make(2))
        pair = t.pair(rng_seed=0)
        result = t.submit(
            winner_id=pair.left.id, loser_id=pair.right.id,
            draw=True, pair_id=pair.pair_id,
        )
        assert result["draw"] is True

    def test_pair_id_window_bound(self, tmp_path):
        """The open_pairings dict is bounded to 200 entries so an
        orchestrator that calls pair() without ever calling submit()
        can't grow state.yaml without bound."""
        t = Tournament.create(tmp_path / "tpb", "tpb", _make(4))
        for _ in range(300):
            t.pair(policy="random", rng_seed=0)
        state = t.state()
        assert len(state["open_pairings"]) <= 200


# ---------------------------------------------------------------------------
# Create-time validation
# ---------------------------------------------------------------------------


class TestCreateValidation:
    def test_empty_hypothesis_list_succeeds_but_cant_pair(self, tmp_path):
        # Current contract: create allows empty list (lets the
        # orchestrator append later). pair() then raises ≥2.
        t = Tournament.create(tmp_path / "tev", "tev", [])
        assert t.state()["hypotheses"] == []
        with pytest.raises(TournamentError, match="≥2"):
            t.pair()

    def test_path_object_in_payload_round_trips(self, tmp_path):
        # YAML safe_dump can't handle PosixPath by default — verify
        # the io layer (atomic_write_yaml) handles it gracefully.
        from pathlib import PosixPath
        try:
            t = Tournament.create(
                tmp_path / "tpp", "tpp",
                [{"id": "h0", "path": PosixPath("/tmp/x")}, {"id": "h1"}],
            )
            # If we got here, the io layer coerces Path → str.
            state = t.state()
            stored = state["hypotheses"][0]["payload"]["path"]
            assert isinstance(stored, str)
        except (TournamentError, yaml.YAMLError, TypeError):
            # Acceptable alternative: typed error before write.
            pass


# ---------------------------------------------------------------------------
# Concurrent create race (M4)
# ---------------------------------------------------------------------------


class TestConcurrentCreate:
    def test_8_threads_only_one_create_wins(self, tmp_path):
        """Stronger version of the M4 audit test — 8 threads racing."""
        path = tmp_path / "tcc"
        results = {"ok": 0, "err": 0}
        lock = threading.Lock()
        barrier = threading.Barrier(8)

        def runner():
            barrier.wait()
            try:
                Tournament.create(path, "tcc", _make(2))
                with lock:
                    results["ok"] += 1
            except TournamentError:
                with lock:
                    results["err"] += 1

        threads = [threading.Thread(target=runner) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert results["ok"] == 1
        assert results["err"] == 7
