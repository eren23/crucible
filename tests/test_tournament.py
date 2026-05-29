"""Tests for the hypothesis tournament (Thrust B1)."""
from __future__ import annotations

import pytest

from crucible.researcher.tournament import (
    Hypothesis,
    Tournament,
    TournamentError,
    _DEFAULT_INITIAL_RATING,
    _expected_score,
    _update_elo,
    judge_request_prompt,
)


# ---------------------------------------------------------------------------
# Elo math
# ---------------------------------------------------------------------------


class TestElo:
    def test_equal_ratings_expect_half(self):
        assert _expected_score(1500.0, 1500.0) == pytest.approx(0.5)

    def test_higher_rating_expects_more(self):
        assert _expected_score(1700.0, 1500.0) > 0.7

    def test_update_winner_gains_loser_loses_symmetric(self):
        new_a, new_b = _update_elo(1500.0, 1500.0, 1.0)
        assert new_a > 1500.0
        assert new_b < 1500.0
        assert new_a - 1500.0 == pytest.approx(1500.0 - new_b)

    def test_draw_keeps_equal_ratings_constant(self):
        new_a, new_b = _update_elo(1500.0, 1500.0, 0.5)
        assert new_a == pytest.approx(1500.0)
        assert new_b == pytest.approx(1500.0)

    def test_underdog_win_swings_more(self):
        # Lower-rated A beating higher-rated B should produce a bigger swing
        # than equal-rated A beating equal-rated B.
        equal_swing = _update_elo(1500.0, 1500.0, 1.0)[0] - 1500.0
        upset_swing = _update_elo(1200.0, 1800.0, 1.0)[0] - 1200.0
        assert upset_swing > equal_swing


# ---------------------------------------------------------------------------
# Tournament lifecycle
# ---------------------------------------------------------------------------


def _make_hypotheses(n: int) -> list[dict]:
    return [{"id": f"h{i}", "summary": f"hypothesis {i}", "rank_hint": i} for i in range(n)]


class TestTournamentCreate:
    def test_create_persists_state(self, tmp_path):
        t = Tournament.create(tmp_path / "t1", "t1", _make_hypotheses(4))
        assert (tmp_path / "t1" / "state.yaml").exists()
        state = t.state()
        assert state["name"] == "t1"
        assert len(state["hypotheses"]) == 4
        assert state["round_number"] == 0
        assert all(e["rating"] == _DEFAULT_INITIAL_RATING for e in state["hypotheses"])

    def test_create_rejects_duplicate_ids(self, tmp_path):
        with pytest.raises(TournamentError, match="duplicate"):
            Tournament.create(
                tmp_path / "t2", "t2",
                [{"id": "h0", "x": 1}, {"id": "h0", "x": 2}],
            )

    def test_create_generates_ids_when_missing(self, tmp_path):
        t = Tournament.create(
            tmp_path / "t3", "t3",
            [{"summary": "no id"}, {"summary": "still no id"}],
        )
        ids = {h.id for h in t.hypotheses()}
        assert len(ids) == 2
        assert all(i.startswith("h_") for i in ids)

    def test_create_twice_raises(self, tmp_path):
        Tournament.create(tmp_path / "t4", "t4", _make_hypotheses(2))
        with pytest.raises(TournamentError, match="already exists"):
            Tournament.create(tmp_path / "t4", "t4", _make_hypotheses(2))


class TestTournamentPair:
    def test_pair_returns_two_distinct(self, tmp_path):
        t = Tournament.create(tmp_path / "tp", "tp", _make_hypotheses(4))
        result = t.pair(policy="random", rng_seed=42)
        assert result.left.id != result.right.id
        assert result.round_number == 1
        assert result.policy == "random"

    def test_pair_requires_two_hypotheses(self, tmp_path):
        t = Tournament.create(tmp_path / "tp2", "tp2", [{"id": "lonely"}])
        with pytest.raises(TournamentError, match="≥2"):
            t.pair()

    def test_pair_unknown_policy_raises(self, tmp_path):
        t = Tournament.create(tmp_path / "tp3", "tp3", _make_hypotheses(2))
        with pytest.raises(TournamentError, match="unknown policy"):
            t.pair(policy="bogus")

    def test_elo_close_picks_closest_ratings(self, tmp_path):
        # M8 fix: the previous test used `assert X or True` (always
        # True) — dead code. Now we drive ratings into a deterministic
        # configuration and check the closest pair is actually returned.
        t = Tournament.create(tmp_path / "tp4", "tp4", _make_hypotheses(3))
        # Submit one result to diverge h0↑ / h1↓; h2 stays at 1500.
        # Ratings: h0≈1516, h1≈1484, h2=1500. Closest deltas:
        #   (h0,h2)=16, (h1,h2)=16, (h0,h1)=32. elo_close must NOT
        # return the (h0,h1) pair (the widest gap).
        t.submit(winner_id="h0", loser_id="h1", rationale="seed")
        # Bypass the no_repeat_within window so elo_close considers all
        # pairs (the seed submit registered (h0,h1) in recent_pairings).
        result = t.pair(policy="elo_close", no_repeat_within=0)
        assert {result.left.id, result.right.id} != {"h0", "h1"}, (
            "elo_close picked the widest pair instead of the closest"
        )
        # And the picked pair has min delta 16 (either h0-h2 or h1-h2).
        ratings = {h.id: h.rating for h in t.hypotheses()}
        delta = abs(ratings[result.left.id] - ratings[result.right.id])
        assert delta == pytest.approx(16.0, abs=0.5)

    def test_pair_records_event(self, tmp_path):
        t = Tournament.create(tmp_path / "tp5", "tp5", _make_hypotheses(2))
        t.pair(policy="random", rng_seed=1)
        events = t.events()
        assert any(e["event"] == "pair" for e in events)


class TestTournamentSubmit:
    def test_winner_rating_goes_up(self, tmp_path):
        t = Tournament.create(tmp_path / "ts", "ts", _make_hypotheses(2))
        before = {h.id: h.rating for h in t.hypotheses()}
        out = t.submit(winner_id="h0", loser_id="h1", rationale="cleaner method")
        after = {h.id: h.rating for h in t.hypotheses()}
        assert after["h0"] > before["h0"]
        assert after["h1"] < before["h1"]
        assert out["winner_new_rating"] == pytest.approx(after["h0"])

    def test_draw_keeps_equal_ratings(self, tmp_path):
        t = Tournament.create(tmp_path / "ts2", "ts2", _make_hypotheses(2))
        t.submit(winner_id="h0", loser_id="h1", draw=True)
        assert all(h.rating == pytest.approx(_DEFAULT_INITIAL_RATING) for h in t.hypotheses())
        assert all(h.draws == 1 for h in t.hypotheses())

    def test_unknown_winner_raises(self, tmp_path):
        t = Tournament.create(tmp_path / "ts3", "ts3", _make_hypotheses(2))
        with pytest.raises(TournamentError, match="unknown winner"):
            t.submit(winner_id="nope", loser_id="h0")

    def test_self_pairing_raises(self, tmp_path):
        t = Tournament.create(tmp_path / "ts4", "ts4", _make_hypotheses(2))
        with pytest.raises(TournamentError, match="winner_id == loser_id"):
            t.submit(winner_id="h0", loser_id="h0")


class TestTournamentRank:
    def test_rank_descending_by_rating(self, tmp_path):
        t = Tournament.create(tmp_path / "tr", "tr", _make_hypotheses(3))
        t.submit(winner_id="h0", loser_id="h1", rationale="ok")
        t.submit(winner_id="h0", loser_id="h2", rationale="ok")
        t.submit(winner_id="h2", loser_id="h1", rationale="ok")
        ranks = t.rank()
        assert [r.id for r in ranks] == ["h0", "h2", "h1"]
        assert ranks[0].wins == 2
        assert ranks[-1].losses == 2

    def test_top_k_truncates(self, tmp_path):
        t = Tournament.create(tmp_path / "tr2", "tr2", _make_hypotheses(5))
        ranks = t.rank(top_k=2)
        assert len(ranks) == 2

    def test_convergence_after_many_rounds(self, tmp_path):
        """If h0 wins every matchup, it should dominate the ranking."""
        t = Tournament.create(tmp_path / "tc", "tc", _make_hypotheses(4))
        for _ in range(20):
            for loser in ("h1", "h2", "h3"):
                t.submit(winner_id="h0", loser_id=loser)
        ranks = t.rank()
        assert ranks[0].id == "h0"
        assert ranks[0].rating > 1700  # well above the starting 1500


# ---------------------------------------------------------------------------
# Orchestrator contract envelope
# ---------------------------------------------------------------------------


class TestJudgeRequestPrompt:
    def test_envelope_shape(self, tmp_path):
        t = Tournament.create(tmp_path / "te", "te", _make_hypotheses(2))
        pair = t.pair(policy="random", rng_seed=0)
        env = judge_request_prompt(pair, context="round-1 context")
        assert {"system", "user", "schema", "pair_id"} <= env.keys()
        assert pair.left.id in env["user"]
        assert pair.right.id in env["user"]
        assert env["schema"]["properties"]["winner_id"]["enum"] == [pair.left.id, pair.right.id]

    def test_envelope_schema_constrains_winner_to_pair(self, tmp_path):
        t = Tournament.create(tmp_path / "te2", "te2", _make_hypotheses(4))
        pair = t.pair(policy="random", rng_seed=0)
        env = judge_request_prompt(pair)
        enum = set(env["schema"]["properties"]["winner_id"]["enum"])
        assert enum == {pair.left.id, pair.right.id}


# ---------------------------------------------------------------------------
# MCP tool surface
# ---------------------------------------------------------------------------


class TestMcpTournamentTools:
    def test_full_lifecycle(self, tmp_path, monkeypatch):
        class _FakeConfig:
            project_root = tmp_path
        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: _FakeConfig())
        from crucible.mcp.tools import TOOL_DISPATCH

        # Create
        out = TOOL_DISPATCH["hypothesis_tournament_create"]({
            "name": "demo",
            "hypotheses": [{"id": "a", "x": 1}, {"id": "b", "x": 2}, {"id": "c", "x": 3}],
        })
        assert out["ok"]
        assert out["hypothesis_count"] == 3

        # Pair
        pair = TOOL_DISPATCH["hypothesis_tournament_pair"]({"name": "demo", "rng_seed": 0})
        assert "system" in pair and "user" in pair and "schema" in pair
        assert pair["left_id"] != pair["right_id"]

        # Submit
        result = TOOL_DISPATCH["hypothesis_tournament_submit"]({
            "name": "demo",
            "winner_id": pair["left_id"],
            "loser_id": pair["right_id"],
            "rationale": "test",
        })
        assert result["winner_new_rating"] > 1500
        assert result["loser_new_rating"] < 1500

        # Rank
        ranks = TOOL_DISPATCH["hypothesis_tournament_rank"]({"name": "demo"})
        assert ranks["count"] == 3
        assert ranks["ranking"][0]["id"] == pair["left_id"]

    def test_cluster_tool(self, tmp_path, monkeypatch):
        class _FakeConfig:
            project_root = tmp_path
        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: _FakeConfig())
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["hypothesis_cluster"]({
            "hypotheses": [
                {"id": "h1", "payload": {"s": "swap relu for gelu in mlp block of the model"}},
                {"id": "h2", "payload": {"s": "swap relu for gelu in mlp block of the model now"}},
                {"id": "h3", "payload": {"s": "completely orthogonal idea"}},
            ],
            "threshold": 0.5,
            "n": 2,
        })
        assert out["count"] == 2  # h1+h2 merged, h3 alone
        assert "h1" in out["keepers"]
        assert "h3" in out["keepers"]

    def test_meta_review_envelope(self, tmp_path, monkeypatch):
        class _FakeConfig:
            project_root = tmp_path
            store_dir = ".crucible"
            research_state_file = ".crucible/state.jsonl"
        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: _FakeConfig())
        from crucible.mcp.tools import TOOL_DISPATCH

        # Seed a tournament so meta_review picks up ranking.
        TOOL_DISPATCH["hypothesis_tournament_create"]({
            "name": "mr_t",
            "hypotheses": [{"id": "a", "summary": "alpha"}, {"id": "b", "summary": "beta"}],
        })
        TOOL_DISPATCH["hypothesis_tournament_submit"]({
            "name": "mr_t", "winner_id": "a", "loser_id": "b",
        })

        out = TOOL_DISPATCH["research_meta_review"]({
            "track_name": "alpha-stuff",
            "tournament_name": "mr_t",
        })
        assert "system" in out and "user" in out and "schema" in out
        assert out["track_name"] == "alpha-stuff"
        assert out["sources_used"]["tournament_entries"] == 2
        assert "alpha-stuff" in out["user"]
        # m7: loosened — assert tournament entries appear in user prompt
        # (by their id) rather than pinning the literal heading copy.
        assert "a" in out["user"] and "b" in out["user"]


# ---------------------------------------------------------------------------
# Regression tests for security/correctness fixes
# ---------------------------------------------------------------------------


class TestCreateConcurrencyM4:
    """M4 fix: Tournament.create holds the file lock for the full
    exists-check → write sequence so two processes can't both pass
    the existence check."""

    def test_second_create_after_first_fails(self, tmp_path):
        Tournament.create(tmp_path / "tc", "tc", _make_hypotheses(2))
        with pytest.raises(TournamentError, match="already exists"):
            Tournament.create(tmp_path / "tc", "tc", _make_hypotheses(2))

    def test_concurrent_creates_only_one_wins(self, tmp_path):
        # Threaded race — two threads racing on create. Exactly one
        # must succeed; the other must raise TournamentError. The lock
        # serialises them so the loser sees state.yaml already exists.
        import threading
        path = tmp_path / "tc_race"
        results: dict[str, int] = {"ok": 0, "err": 0}
        lock = threading.Lock()

        def runner():
            try:
                Tournament.create(path, "tc_race", _make_hypotheses(2))
                with lock:
                    results["ok"] += 1
            except TournamentError:
                with lock:
                    results["err"] += 1

        threads = [threading.Thread(target=runner) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert results["ok"] == 1
        assert results["err"] == 3


class TestPairIdIntegrityM5:
    """M5 fix: submit accepts an optional pair_id; when provided it
    must match an issued pair_id naming the same two hypotheses."""

    def test_pair_id_is_returned(self, tmp_path):
        t = Tournament.create(tmp_path / "t_pid", "t_pid", _make_hypotheses(2))
        pair = t.pair(rng_seed=0)
        assert pair.pair_id and pair.pair_id.startswith("p_")

    def test_submit_with_valid_pair_id_works(self, tmp_path):
        t = Tournament.create(tmp_path / "t_pid2", "t_pid2", _make_hypotheses(2))
        pair = t.pair(rng_seed=0)
        result = t.submit(
            winner_id=pair.left.id, loser_id=pair.right.id,
            pair_id=pair.pair_id,
        )
        assert result["winner_id"] == pair.left.id

    def test_submit_with_unknown_pair_id_rejected(self, tmp_path):
        t = Tournament.create(tmp_path / "t_pid3", "t_pid3", _make_hypotheses(2))
        pair = t.pair(rng_seed=0)
        with pytest.raises(TournamentError, match="not open"):
            t.submit(
                winner_id=pair.left.id, loser_id=pair.right.id,
                pair_id="p_fabricated",
            )

    def test_submit_with_mismatched_ids_rejected(self, tmp_path):
        t = Tournament.create(tmp_path / "t_pid4", "t_pid4", _make_hypotheses(4))
        pair = t.pair(rng_seed=0)
        # Pick two IDs that are NOT the pair's actual members.
        others = [h.id for h in t.hypotheses() if h.id not in {pair.left.id, pair.right.id}]
        assert len(others) >= 2
        with pytest.raises(TournamentError, match="issued for"):
            t.submit(
                winner_id=others[0], loser_id=others[1],
                pair_id=pair.pair_id,
            )

    def test_submit_consumes_pair_id_once(self, tmp_path):
        t = Tournament.create(tmp_path / "t_pid5", "t_pid5", _make_hypotheses(2))
        pair = t.pair(rng_seed=0)
        t.submit(
            winner_id=pair.left.id, loser_id=pair.right.id,
            pair_id=pair.pair_id,
        )
        # Replay of the same pair_id is rejected.
        with pytest.raises(TournamentError, match="not open"):
            t.submit(
                winner_id=pair.left.id, loser_id=pair.right.id,
                pair_id=pair.pair_id,
            )

    def test_submit_without_pair_id_still_works_backward_compat(self, tmp_path):
        t = Tournament.create(tmp_path / "t_pid6", "t_pid6", _make_hypotheses(2))
        t.pair(rng_seed=0)
        # No pair_id supplied — legacy path still works.
        result = t.submit(winner_id="h0", loser_id="h1")
        assert result["winner_id"] == "h0"


class TestNameHardeningM9:
    """M9 fix: _tournament_dir rejects null bytes, leading dots,
    backslashes — not just / and ..."""

    def test_null_byte_rejected(self, tmp_path, monkeypatch):
        class _FakeConfig:
            project_root = tmp_path
        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: _FakeConfig())
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["hypothesis_tournament_create"]({
            "name": "evil\x00name",
            "hypotheses": [{"id": "a"}, {"id": "b"}],
        })
        assert "error" in out
        assert "invalid tournament name" in out["error"]

    def test_leading_dot_rejected(self, tmp_path, monkeypatch):
        class _FakeConfig:
            project_root = tmp_path
        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: _FakeConfig())
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["hypothesis_tournament_create"]({
            "name": ".git",
            "hypotheses": [{"id": "a"}, {"id": "b"}],
        })
        assert "error" in out
        assert "invalid tournament name" in out["error"]

    def test_backslash_rejected(self, tmp_path, monkeypatch):
        class _FakeConfig:
            project_root = tmp_path
        monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: _FakeConfig())
        from crucible.mcp.tools import TOOL_DISPATCH
        out = TOOL_DISPATCH["hypothesis_tournament_create"]({
            "name": "bad\\name",
            "hypotheses": [{"id": "a"}, {"id": "b"}],
        })
        assert "error" in out
        assert "invalid tournament name" in out["error"]


class TestAsymmetricDrawM6:
    """m6 fix: draw between unequal-rated hypotheses transfers points
    from the higher-rated to the lower-rated player."""

    def test_draw_with_diverged_ratings_transfers_points(self, tmp_path):
        t = Tournament.create(tmp_path / "tad", "tad", _make_hypotheses(2))
        # Diverge ratings: 5 wins by h0 over h1.
        for _ in range(5):
            t.submit(winner_id="h0", loser_id="h1")
        ratings_before = {h.id: h.rating for h in t.hypotheses()}
        # Now a draw — h0 (higher) should LOSE points, h1 (lower) should GAIN.
        t.submit(winner_id="h0", loser_id="h1", draw=True)
        ratings_after = {h.id: h.rating for h in t.hypotheses()}
        assert ratings_after["h0"] < ratings_before["h0"]
        assert ratings_after["h1"] > ratings_before["h1"]
        # Symmetric-swing invariant still holds.
        delta_a = ratings_after["h0"] - ratings_before["h0"]
        delta_b = ratings_after["h1"] - ratings_before["h1"]
        assert delta_a + delta_b == pytest.approx(0.0, abs=0.01)
