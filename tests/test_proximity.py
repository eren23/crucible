"""Tests for the proximity / dedupe module (Thrust B2)."""
from __future__ import annotations

import pytest

from crucible.researcher.proximity import (
    cluster,
    jaccard,
    suggest_keepers,
)


class TestJaccard:
    def test_identical_strings(self):
        assert jaccard("the quick brown fox", "the quick brown fox") == pytest.approx(1.0)

    def test_disjoint_strings(self):
        assert jaccard("alpha beta gamma", "uno dos tres") == pytest.approx(0.0)

    def test_partial_overlap(self):
        # 3-gram shingles: "the quick brown" overlaps with "quick brown fox"
        # for "quick brown" (no, 3-grams not 2). Test the general behavior.
        sim = jaccard("the quick brown fox jumps", "the quick brown dog runs")
        assert 0.0 < sim < 1.0

    def test_empty_strings(self):
        assert jaccard("", "") == pytest.approx(1.0)
        assert jaccard("x y z", "") == pytest.approx(0.0)


class TestCluster:
    def test_singleton_per_unique_text(self):
        hypotheses = [
            {"id": "h1", "payload": {"summary": "increase batch size to 256"}},
            {"id": "h2", "payload": {"summary": "swap optimizer to muon"}},
            {"id": "h3", "payload": {"summary": "double learning rate"}},
        ]
        clusters = cluster(hypotheses, threshold=0.9)
        assert len(clusters) == 3
        assert all(len(c.members) == 1 for c in clusters)

    def test_near_duplicates_merged(self):
        hypotheses = [
            {"id": "h1", "payload": {"summary": "increase batch size to 256 for stability"}},
            {"id": "h2", "payload": {"summary": "increase batch size to 256 for stability and throughput"}},
            {"id": "h3", "payload": {"summary": "completely orthogonal idea about activations"}},
        ]
        clusters = cluster(hypotheses, threshold=0.4)
        # h1 + h2 should cluster; h3 stands alone.
        cluster_sizes = sorted(len(c.members) for c in clusters)
        assert cluster_sizes == [1, 2]

    def test_threshold_strictness(self):
        hypotheses = [
            {"id": "h1", "payload": {"summary": "swap relu for gelu in mlp block of the model"}},
            {"id": "h2", "payload": {"summary": "swap gelu for silu in mlp block of the model"}},
        ]
        # High threshold → no merge
        assert len(cluster(hypotheses, threshold=0.9)) == 2
        # Low threshold with bigram shingles → merge
        assert len(cluster(hypotheses, threshold=0.3, n=2)) == 1

    def test_payload_text_walks_nested_dicts(self):
        hypotheses = [
            {"id": "h1", "payload": {"config": {"activation": "relu", "lr": "0.1"}, "note": "baseline"}},
            {"id": "h2", "payload": {"config": {"activation": "relu", "lr": "0.1"}, "note": "baseline"}},
        ]
        clusters = cluster(hypotheses, threshold=0.5)
        assert len(clusters) == 1

    def test_numeric_leaves_ignored(self):
        # Two hypotheses with same string content but different floats should still merge.
        hypotheses = [
            {"id": "h1", "payload": {"summary": "swap relu for gelu", "expected": 0.1}},
            {"id": "h2", "payload": {"summary": "swap relu for gelu", "expected": 999.9}},
        ]
        clusters = cluster(hypotheses, threshold=0.5)
        assert len(clusters) == 1

    def test_empty_input(self):
        assert cluster([]) == []

    def test_suggest_keepers_one_per_cluster(self):
        hypotheses = [
            {"id": "h1", "payload": {"summary": "alpha beta gamma delta epsilon"}},
            {"id": "h2", "payload": {"summary": "alpha beta gamma delta epsilon"}},
            {"id": "h3", "payload": {"summary": "completely different content here"}},
        ]
        clusters = cluster(hypotheses, threshold=0.5)
        keepers = suggest_keepers(clusters)
        assert len(keepers) == len(clusters)
        # h1 wins its cluster over h2 (first wins).
        assert "h1" in keepers
        assert "h3" in keepers


class TestKmeansBackend:
    def test_missing_sklearn_raises(self, monkeypatch):
        # Skip if sklearn is actually installed; only test the error path.
        try:
            import sklearn  # noqa: F401
            pytest.skip("sklearn installed — skipping the missing-dep test path")
        except ImportError:
            pass
        from crucible.researcher.proximity import ProximityError
        with pytest.raises(ProximityError, match="scikit-learn"):
            cluster(
                [
                    {"id": "h1", "payload": {"x": "a"}},
                    {"id": "h2", "payload": {"x": "b"}},
                ],
                backend="kmeans",
            )


# ---------------------------------------------------------------------------
# Regression: C3 — threshold semantics
# ---------------------------------------------------------------------------


class TestThresholdSemantics:
    """C3: lower threshold = more merging = fewer clusters.
    Both backends should honor the same semantics. KMeans is hard to
    test without sklearn; verify the shingle backend explicitly so a
    future inversion regression gets caught."""

    def test_lower_threshold_means_fewer_clusters_shingle(self):
        hypotheses = [
            {"id": f"h{i}", "payload": {"summary": f"swap activation variant {i} from relu to gelu"}}
            for i in range(6)
        ]
        tight = cluster(hypotheses, threshold=0.95, n=2)  # almost no merging
        loose = cluster(hypotheses, threshold=0.25, n=2)  # aggressive merging
        assert len(tight) >= len(loose), (
            f"tight={len(tight)} should be >= loose={len(loose)} clusters"
        )

    def test_kmeans_formula_uses_threshold_not_inverse(self):
        """Inspect the source so a regression to the inverted formula
        is caught at module-load. (sklearn not available; testing the
        runtime behavior would require it.)"""
        import inspect

        from crucible.researcher import proximity
        src = inspect.getsource(proximity._cluster_kmeans)
        # The correct formula multiplies by `threshold`, not `(1 - threshold)`.
        assert "n_h * threshold" in src or "n_h*threshold" in src
        assert "(1.0 - threshold)" not in src and "(1 - threshold)" not in src
