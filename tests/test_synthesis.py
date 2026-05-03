"""Tests for findings-pair synthesis — GIANTS-style hypothesis seeding.

Mines pairs of hub findings and asks the orchestrator's LLM to predict
the experiment that synthesizes both. Pure orchestrator-contract: no
internal LLM call, returns ``{system, user, schema, parent_finding_ids}``.
"""
from __future__ import annotations

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


class TestPromptSecretRedaction:
    def test_redacts_secrets_in_user_prompt(self) -> None:
        a = _f("aaa", title="LR finding")
        a["body"] = "Discovered with HF_TOKEN=hf_abcdefABCDEF1234567890abcdefABCDEF12 set."
        b = _f("bbb", title="Muon finding")
        out = build_synthesis_prompt((a, b))
        assert "hf_abcdefABCDEF" not in out["user"]
        # Some redaction marker must appear (***/REDACTED/etc.).
        assert "REDACTED" in out["user"] or "***" in out["user"]
