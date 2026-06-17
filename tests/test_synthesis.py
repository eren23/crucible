"""Tests for findings-pair synthesis — GIANTS-style hypothesis seeding.

Mines pairs of hub findings and asks the orchestrator's LLM to predict
the experiment that synthesizes both. Pure orchestrator-contract: no
internal LLM call, returns ``{system, user, schema, parent_finding_ids}``.
"""
from __future__ import annotations

from datetime import UTC

import pytest

from crucible.core.errors import ResearcherError
from crucible.researcher.synthesis import (
    build_synthesis_prompt,
    mine_pairs,
    parse_synthesis_response,
)


def _f(fid: str, *, track: str | None = None, title: str | None = None) -> dict:
    return {
        "id": fid,
        "title": title or fid,
        "body": f"body of {fid}",
        "scope": "global" if track is None else "track",
        "track": track,
        "tags": [],
        "category": "observation",
        "confidence": 0.7,
    }


class TestMinePairs:
    def test_random_returns_k_pairs_from_pool(self) -> None:
        pool = [_f(f"f{i}") for i in range(6)]
        pairs = mine_pairs(pool, k=3, policy="random", seed=1)
        assert len(pairs) == 3
        for a, b in pairs:
            assert a["id"] != b["id"]

    def test_random_is_deterministic_with_seed(self) -> None:
        pool = [_f(f"f{i}") for i in range(8)]
        a = mine_pairs(pool, k=3, policy="random", seed=42)
        b = mine_pairs(pool, k=3, policy="random", seed=42)
        ids_a = [(p[0]["id"], p[1]["id"]) for p in a]
        ids_b = [(p[0]["id"], p[1]["id"]) for p in b]
        assert ids_a == ids_b

    def test_pairs_are_unique_unordered(self) -> None:
        pool = [_f(f"f{i}") for i in range(6)]
        pairs = mine_pairs(pool, k=5, policy="random", seed=2)
        seen: set[frozenset[str]] = set()
        for a, b in pairs:
            key = frozenset((a["id"], b["id"]))
            assert key not in seen
            seen.add(key)

    def test_same_track_only_pairs_within_same_track(self) -> None:
        pool = [
            _f("a", track="t1"),
            _f("b", track="t1"),
            _f("c", track="t1"),
            _f("d", track="t2"),
            _f("e", track="t2"),
        ]
        pairs = mine_pairs(pool, k=4, policy="same_track", seed=0)
        for a, b in pairs:
            assert a.get("track") == b.get("track")
            assert a.get("track") is not None

    def test_cross_track_only_pairs_across_different_tracks(self) -> None:
        pool = [
            _f("a", track="t1"),
            _f("b", track="t1"),
            _f("c", track="t2"),
            _f("d", track="t3"),
        ]
        pairs = mine_pairs(pool, k=3, policy="cross_track", seed=0)
        assert pairs
        for a, b in pairs:
            assert a.get("track") != b.get("track")

    def test_raises_when_pool_too_small(self) -> None:
        with pytest.raises(ResearcherError, match="at least 2"):
            mine_pairs([_f("a")], k=1, policy="random")

    def test_returns_fewer_pairs_when_pool_constrains_k(self) -> None:
        pool = [_f(f"f{i}") for i in range(3)]
        # 3 items → only 3 unordered pairs possible
        pairs = mine_pairs(pool, k=10, policy="random", seed=0)
        assert len(pairs) == 3

    def test_unknown_policy_rejected(self) -> None:
        pool = [_f(f"f{i}") for i in range(3)]
        with pytest.raises(ResearcherError, match="policy"):
            mine_pairs(pool, k=1, policy="not_real")

    def test_same_track_skips_pool_when_no_eligible_pairs(self) -> None:
        pool = [_f("a", track="t1"), _f("b", track="t2")]
        pairs = mine_pairs(pool, k=3, policy="same_track", seed=0)
        assert pairs == []


class TestMinePairsTagFilter:
    def _ft(self, fid: str, tags: list[str]) -> dict:
        d = _f(fid)
        d["tags"] = tags
        return d

    def test_pair_eligible_when_either_finding_has_tag(self) -> None:
        pool = [
            self._ft("a", ["optim"]),
            self._ft("b", []),
            self._ft("c", ["arch"]),
            self._ft("d", []),
        ]
        pairs = mine_pairs(pool, k=10, policy="random", seed=0, required_tags={"optim"})
        # (a,b): a has optim → ok. (a,c): a has optim → ok. (a,d): ok. (b,c): no. (b,d): no. (c,d): no.
        assert len(pairs) == 3
        for x, y in pairs:
            assert "optim" in (x.get("tags") or []) or "optim" in (y.get("tags") or [])

    def test_pair_rejected_when_neither_has_required_tag(self) -> None:
        pool = [
            self._ft("a", ["arch"]),
            self._ft("b", []),
            self._ft("c", []),
        ]
        pairs = mine_pairs(pool, k=10, policy="random", seed=0, required_tags={"optim"})
        assert pairs == []

    def test_empty_required_tags_does_not_filter(self) -> None:
        pool = [self._ft(f"f{i}", []) for i in range(4)]
        pairs = mine_pairs(pool, k=2, policy="random", seed=0, required_tags=set())
        assert len(pairs) == 2

    def test_required_tags_or_semantics_across_tag_set(self) -> None:
        # required_tags={"a","b"} → pair eligible if either finding has 'a' OR 'b'.
        pool = [
            self._ft("x", ["a"]),
            self._ft("y", ["b"]),
            self._ft("z", []),
        ]
        pairs = mine_pairs(pool, k=10, policy="random", seed=0, required_tags={"a", "b"})
        # (x,y) ok; (x,z) ok; (y,z) ok.
        assert len(pairs) == 3


class TestBuildSynthesisPrompt:
    def test_returns_orchestrator_contract_shape(self) -> None:
        pair = (_f("aaa", title="LR warmup matters"), _f("bbb", title="Muon beats AdamW"))
        out = build_synthesis_prompt(pair)
        assert set(out.keys()) >= {"system", "user", "schema", "parent_finding_ids"}
        assert out["parent_finding_ids"] == ["aaa", "bbb"]
        assert isinstance(out["system"], str) and out["system"]
        assert isinstance(out["user"], str) and out["user"]
        assert isinstance(out["schema"], dict)

    def test_user_prompt_includes_both_finding_titles(self) -> None:
        pair = (_f("aaa", title="LR warmup matters"), _f("bbb", title="Muon beats AdamW"))
        out = build_synthesis_prompt(pair)
        assert "LR warmup matters" in out["user"]
        assert "Muon beats AdamW" in out["user"]

    def test_schema_requires_synthesis_hypothesis(self) -> None:
        pair = (_f("aaa"), _f("bbb"))
        out = build_synthesis_prompt(pair)
        schema = out["schema"]
        assert schema["type"] == "object"
        assert "hypotheses" in schema["properties"]


class TestParseSynthesisResponse:
    def test_parses_minimal_response(self) -> None:
        pair = (_f("aaa"), _f("bbb"))
        response = {
            "hypotheses": [
                {
                    "name": "synth_aaa_bbb_warmup_muon",
                    "hypothesis": "Try Muon with LR warmup.",
                    "config": {"OPTIMIZER": "muon", "WARMUP_STEPS": "200"},
                    "rationale": "Combines both parents.",
                }
            ]
        }
        hyps = parse_synthesis_response(response, pair)
        assert len(hyps) == 1
        h = hyps[0]
        assert h["name"] == "synth_aaa_bbb_warmup_muon"
        assert h["config"]["OPTIMIZER"] == "muon"
        assert h["parent_finding_ids"] == ["aaa", "bbb"]

    def test_drops_invalid_hypothesis_without_config(self) -> None:
        pair = (_f("aaa"), _f("bbb"))
        response = {
            "hypotheses": [
                {"name": "no_config_hyp", "hypothesis": "incomplete"},
                {
                    "name": "ok",
                    "hypothesis": "good",
                    "config": {"X": "1"},
                },
            ]
        }
        hyps = parse_synthesis_response(response, pair)
        assert len(hyps) == 1
        assert hyps[0]["name"] == "ok"

    def test_accepts_string_response(self) -> None:
        pair = (_f("aaa"), _f("bbb"))
        raw = (
            '{"hypotheses": [{"name": "x", "hypothesis": "y", '
            '"config": {"K": "v"}, "rationale": "r"}]}'
        )
        hyps = parse_synthesis_response(raw, pair)
        assert len(hyps) == 1
        assert hyps[0]["parent_finding_ids"] == ["aaa", "bbb"]

    def test_normalizes_config_values_to_strings(self) -> None:
        pair = (_f("aaa"), _f("bbb"))
        response = {
            "hypotheses": [
                {
                    "name": "x",
                    "hypothesis": "y",
                    "config": {"STEPS": 200, "LR": 0.001},
                }
            ]
        }
        hyps = parse_synthesis_response(response, pair)
        assert hyps[0]["config"] == {"STEPS": "200", "LR": "0.001"}

    def test_marks_generation_method_as_synthesis(self) -> None:
        pair = (_f("aaa"), _f("bbb"))
        response = {
            "hypotheses": [
                {"name": "x", "hypothesis": "y", "config": {"K": "v"}}
            ]
        }
        hyps = parse_synthesis_response(response, pair)
        assert hyps[0]["generation_method"] == "synthesis"


class TestMemoryFilter:
    """Phase 4.2: memory_filter policy ranks pairs by confidence ×
    recency + cross-project diversity + tag overlap instead of uniform
    random sampling."""

    NOW = "2026-05-16T12:00:00+00:00"

    def _finding(
        self,
        fid: str,
        *,
        confidence: float = 0.7,
        days_old: int = 0,
        project: str = "p1",
        tags: list[str] | None = None,
    ) -> dict:
        from datetime import datetime, timedelta
        created = datetime(2026, 5, 16, 12, 0, 0, tzinfo=UTC) - timedelta(days=days_old)
        return {
            "id": fid,
            "title": fid,
            "body": f"body of {fid}",
            "confidence": confidence,
            "created_at": created.isoformat(),
            "source_project": project,
            "tags": tags or [],
            "category": "observation",
            "scope": "global",
            "track": None,
        }

    def test_score_finding_confidence_dominates(self):
        from crucible.researcher.synthesis import score_finding
        high = self._finding("h", confidence=0.9, days_old=0)
        low = self._finding("l", confidence=0.2, days_old=0)
        assert score_finding(high, now=self.NOW) > score_finding(low, now=self.NOW)

    def test_score_finding_recency_decay(self):
        from crucible.researcher.synthesis import score_finding
        fresh = self._finding("f", confidence=0.8, days_old=0)
        old = self._finding("o", confidence=0.8, days_old=180)
        # Fresh > old at the same confidence.
        assert score_finding(fresh, now=self.NOW) > score_finding(old, now=self.NOW) * 1.5

    def test_score_pair_cross_project_beats_same_project(self):
        from crucible.researcher.synthesis import score_pair
        a = self._finding("a", project="p1")
        b = self._finding("b", project="p2")
        c = self._finding("c", project="p1")
        # Cross-project pair (a, b) beats same-project pair (a, c) at
        # equal confidence/recency.
        assert score_pair(a, b, now=self.NOW) > score_pair(a, c, now=self.NOW)

    def test_score_pair_tag_overlap_bonus(self):
        from crucible.researcher.synthesis import score_pair
        a = self._finding("a", project="p1", tags=["jepa"])
        b = self._finding("b", project="p1", tags=["jepa"])  # shared tag
        c = self._finding("c", project="p1", tags=[])
        assert score_pair(a, b, now=self.NOW) > score_pair(a, c, now=self.NOW)

    def test_memory_filter_returns_top_k_by_score(self):
        """Mining returns the highest-scored pairs first, deterministically."""
        from crucible.researcher.synthesis import mine_pairs
        # 4 findings: gold (high-confidence, fresh, cross-project) vs
        # the rest.
        gold_a = self._finding("g_a", confidence=0.95, days_old=0, project="p1")
        gold_b = self._finding("g_b", confidence=0.95, days_old=0, project="p2")
        meh_a = self._finding("m_a", confidence=0.3, days_old=200, project="p1")
        meh_b = self._finding("m_b", confidence=0.3, days_old=200, project="p1")
        pairs = mine_pairs(
            [gold_a, gold_b, meh_a, meh_b],
            k=1,
            policy="memory_filter",
            now=self.NOW,
        )
        assert len(pairs) == 1
        ids = {pairs[0][0]["id"], pairs[0][1]["id"]}
        assert ids == {"g_a", "g_b"}, (
            f"memory_filter should rank the gold cross-project pair first; got {ids}"
        )

    def test_memory_filter_deterministic(self):
        """Same input → same output across runs (no random shuffle)."""
        from crucible.researcher.synthesis import mine_pairs
        findings = [self._finding(f"f{i}", confidence=0.5 + 0.1 * (i % 3)) for i in range(6)]
        a = mine_pairs(findings, k=3, policy="memory_filter", now=self.NOW)
        b = mine_pairs(findings, k=3, policy="memory_filter", now=self.NOW)
        assert [(p[0]["id"], p[1]["id"]) for p in a] == [(p[0]["id"], p[1]["id"]) for p in b]

    def test_missing_timestamp_defaults_to_no_penalty(self):
        """Phase 4 review fix: previously _recency_decay returned 0.5
        when timestamps were unparseable, which actively halved the
        score of timestamp-less findings. Now defaults to 1.0
        (treat-as-fresh) so missing metadata doesn't penalize."""
        from crucible.researcher.synthesis import score_finding
        f_no_ts = {"id": "f", "confidence": 0.8}  # no created_at
        f_with_ts = self._finding("g", confidence=0.8, days_old=0)
        # Without timestamps, score should match a fresh-timestamped
        # finding at the same confidence (within tolerance).
        s_no = score_finding(f_no_ts, now=self.NOW)
        s_with = score_finding(f_with_ts, now=self.NOW)
        assert abs(s_no - s_with) < 1e-3, (
            f"missing timestamp should not penalize; "
            f"got no_ts={s_no:.3f} vs fresh={s_with:.3f}"
        )

    def test_memory_filter_config_override(self):
        """Tunable: half_life_days override changes the ranking."""
        from crucible.researcher.synthesis import score_finding
        old = self._finding("o", confidence=0.9, days_old=180)
        # Default half-life = 90d → decay ≈ 0.25 → score ≈ 0.225
        default_score = score_finding(old, now=self.NOW)
        # Shorter half-life → faster decay → lower score.
        short_score = score_finding(
            old, now=self.NOW, config={"half_life_days": 30.0},
        )
        # Longer half-life → slower decay → higher score.
        long_score = score_finding(
            old, now=self.NOW, config={"half_life_days": 365.0},
        )
        assert short_score < default_score < long_score

    def test_memory_filter_config_propagates_through_mine_pairs(self):
        """memory_filter_config kwarg on mine_pairs reaches score_pair."""
        from crucible.researcher.synthesis import mine_pairs
        findings = [
            self._finding("a", project="p1", confidence=0.5),
            self._finding("b", project="p2", confidence=0.5),
            self._finding("c", project="p1", confidence=0.5),
        ]
        # With a huge cross-project bonus, the cross-project pair (a, b)
        # must come first.
        pairs = mine_pairs(
            findings, k=1, policy="memory_filter", now=self.NOW,
            memory_filter_config={"cross_project_bonus": 10.0},
        )
        ids = {pairs[0][0]["id"], pairs[0][1]["id"]}
        assert ids == {"a", "b"}

    def test_memory_filter_respects_tag_filter(self):
        from crucible.researcher.synthesis import mine_pairs
        f1 = self._finding("f1", tags=["jepa"])
        f2 = self._finding("f2", tags=["jepa"])
        f3 = self._finding("f3", tags=["unrelated"])
        f4 = self._finding("f4", tags=[])
        pairs = mine_pairs(
            [f1, f2, f3, f4], k=10, policy="memory_filter",
            required_tags={"jepa"}, now=self.NOW,
        )
        ids = {fid for pair in pairs for fid in (pair[0]["id"], pair[1]["id"])}
        # f4 has no tags, can only appear via f1/f2 carrying 'jepa'.
        # f3 has 'unrelated' — eligible only when paired with f1 or f2.
        assert "f1" in ids or "f2" in ids


class TestPromptSecretRedaction:
    def test_redacts_secrets_in_user_prompt(self) -> None:
        a = _f("aaa", title="LR finding")
        a["body"] = "Discovered with HF_TOKEN=hf_abcdefABCDEF1234567890abcdefABCDEF12 set."
        b = _f("bbb", title="Muon finding")
        out = build_synthesis_prompt((a, b))
        assert "hf_abcdefABCDEF" not in out["user"]
        # Some redaction marker must appear (***/REDACTED/etc.).
        assert "REDACTED" in out["user"] or "***" in out["user"]
