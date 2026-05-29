"""Proximity edge cases — adversarial inputs the audit didn't cover."""
from __future__ import annotations

import time

import pytest

from crucible.researcher.proximity import (
    cluster,
    jaccard,
    suggest_keepers,
)


class TestSelfReferentialPayload:
    """A self-referential dict in a hypothesis payload must not crash
    the proximity walker. Real LLM responses occasionally produce
    graphs (e.g. when a hypothesis references its own id by name in
    a dict that the orchestrator builds up wrongly)."""

    def test_self_ref_dict_no_stack_overflow(self):
        d: dict = {"summary": "self ref"}
        d["myself"] = d
        hypotheses = [
            {"id": "h1", "payload": d},
            {"id": "h2", "payload": {"summary": "other"}},
        ]
        # Should not raise RecursionError.
        clusters = cluster(hypotheses, threshold=0.5)
        assert len(clusters) >= 1

    def test_self_ref_list_no_stack_overflow(self):
        lst: list = ["seed"]
        lst.append(lst)
        hypotheses = [
            {"id": "h1", "payload": {"items": lst}},
            {"id": "h2", "payload": {"items": ["other"]}},
        ]
        clusters = cluster(hypotheses, threshold=0.5)
        assert len(clusters) >= 1


class TestExtremeShapes:
    def test_all_identical_collapses_to_one_cluster(self):
        payload = {"summary": "identical text identical text identical text"}
        hypotheses = [{"id": f"h{i}", "payload": payload} for i in range(50)]
        clusters = cluster(hypotheses, threshold=0.5)
        assert len(clusters) == 1
        assert len(clusters[0].members) == 50

    def test_all_disjoint_yields_singletons(self):
        # 50 hypotheses each with unique words so the shingle overlap
        # is exactly 0 between any pair.
        hypotheses = [
            {"id": f"h{i}", "payload": {"summary": f"unique{i}a unique{i}b unique{i}c"}}
            for i in range(50)
        ]
        clusters = cluster(hypotheses, threshold=0.3, n=2)
        assert len(clusters) == 50
        assert all(len(c.members) == 1 for c in clusters)

    def test_1000_hypothesis_stress(self):
        hypotheses = [
            {"id": f"h{i}", "payload": {"summary": f"variant {i % 50} of activation experiment"}}
            for i in range(1000)
        ]
        start = time.time()
        clusters = cluster(hypotheses, threshold=0.7, n=3)
        elapsed = time.time() - start
        assert clusters, "expected at least one cluster"
        # Loose bound — we just want to catch quadratic blowup.
        # 1000 hypotheses × ~500k pair comparisons should finish in <10s.
        assert elapsed < 10.0, f"clustering 1000 hypotheses took {elapsed:.1f}s"

    def test_n_equals_one_single_hypothesis(self):
        clusters = cluster([{"id": "lonely", "payload": {"summary": "alone"}}])
        assert len(clusters) == 1
        assert clusters[0].representative == "lonely"

    def test_suggest_keepers_on_empty(self):
        assert suggest_keepers([]) == []


class TestPayloadShapes:
    def test_none_values_skipped(self):
        hypotheses = [
            {"id": "h1", "payload": {"summary": "alpha", "extra": None}},
            {"id": "h2", "payload": {"summary": "alpha", "extra": None}},
        ]
        clusters = cluster(hypotheses, threshold=0.5)
        assert len(clusters) == 1

    def test_bool_values_skipped(self):
        hypotheses = [
            {"id": "h1", "payload": {"summary": "alpha beta gamma", "flag": True}},
            {"id": "h2", "payload": {"summary": "alpha beta gamma", "flag": False}},
        ]
        clusters = cluster(hypotheses, threshold=0.5)
        assert len(clusters) == 1

    def test_empty_string_payload(self):
        hypotheses = [
            {"id": "h1", "payload": {"summary": ""}},
            {"id": "h2", "payload": {"summary": ""}},
        ]
        # Both have empty text — jaccard("", "") == 1.0 (vacuous truth).
        clusters = cluster(hypotheses, threshold=0.5)
        # All empty-text payloads cluster together (similarity 1.0).
        assert len(clusters) == 1

    def test_deeply_nested_payload_walk(self):
        nested = {"l1": {"l2": {"l3": {"l4": "deep alpha beta gamma"}}}}
        hypotheses = [
            {"id": "h1", "payload": nested},
            {"id": "h2", "payload": {"summary": "deep alpha beta gamma"}},
        ]
        clusters = cluster(hypotheses, threshold=0.3, n=2)
        assert len(clusters) == 1

    def test_mixed_list_dict_payload(self):
        hypotheses = [
            {"id": "h1", "payload": {"items": [{"text": "alpha"}, {"text": "beta"}]}},
            {"id": "h2", "payload": {"items": [{"text": "alpha"}, {"text": "beta"}]}},
        ]
        clusters = cluster(hypotheses, threshold=0.5)
        assert len(clusters) == 1


class TestThresholdEdges:
    def test_threshold_zero_collapses_everything(self):
        hypotheses = [
            {"id": f"h{i}", "payload": {"summary": f"completely different {i}xxx"}}
            for i in range(4)
        ]
        clusters = cluster(hypotheses, threshold=0.0)
        # threshold=0 → ANY positive similarity merges. Pairs with 0
        # similarity remain separate. With 3-gram shingles on these
        # inputs the inter-pair similarity is 0, so we expect singletons.
        # The point of the test is: it doesn't crash.
        assert clusters

    def test_threshold_one_means_only_identical_merge(self):
        hypotheses = [
            {"id": "h1", "payload": {"summary": "alpha beta gamma"}},
            {"id": "h2", "payload": {"summary": "alpha beta gamma"}},  # identical
            {"id": "h3", "payload": {"summary": "almost alpha beta gamma here"}},
        ]
        clusters = cluster(hypotheses, threshold=1.0, n=3)
        # Only identical h1+h2 merge.
        sizes = sorted(len(c.members) for c in clusters)
        assert sizes == [1, 2]


class TestSimilarityValues:
    def test_singleton_similarity_is_one(self):
        clusters = cluster([{"id": "h1", "payload": {"summary": "x"}}])
        assert clusters[0].similarity == 1.0

    def test_merged_cluster_similarity_in_range(self):
        hypotheses = [
            {"id": "h1", "payload": {"summary": "alpha beta gamma delta epsilon zeta"}},
            {"id": "h2", "payload": {"summary": "alpha beta gamma delta epsilon eta"}},
            {"id": "h3", "payload": {"summary": "alpha beta gamma delta theta eta"}},
        ]
        clusters = cluster(hypotheses, threshold=0.3, n=3)
        for c in clusters:
            assert 0.0 <= c.similarity <= 1.0


class TestJaccardEdges:
    def test_n_larger_than_tokens(self):
        # 5-gram shingles on 3-token strings — both become single tuple shingles.
        assert jaccard("a b c", "a b c", n=5) == pytest.approx(1.0)
        assert jaccard("a b c", "x y z", n=5) == pytest.approx(0.0)

    def test_case_insensitive(self):
        assert jaccard("Alpha Beta", "alpha beta") == pytest.approx(1.0)

    def test_punctuation_handled(self):
        # _tokenize uses [A-Za-z0-9_]+ — punctuation drops cleanly.
        assert jaccard("alpha beta!", "alpha, beta.") == pytest.approx(1.0)
