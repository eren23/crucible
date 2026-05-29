"""Proximity / dedupe for hypothesis tournaments (Thrust B2).

Mirrors DeepMind Co-Scientist's Proximity agent: cluster
near-duplicate hypotheses before judging so the tournament doesn't
burn debate rounds on essentially-identical pairs.

Default backend is shingle Jaccard (pure stdlib — no sklearn
dependency). Optional sklearn KMeans backend is loaded lazily and
only when explicitly requested via ``backend='kmeans'``.

API:
    cluster(hypotheses, threshold=0.7) -> list[Cluster]
    suggest_keepers(clusters, policy='first') -> list[Hypothesis-id]
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from crucible.core.errors import CrucibleError


class ProximityError(CrucibleError):
    """Proximity / dedupe failure."""


_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class Cluster:
    """A group of near-duplicate hypotheses.

    ``similarity`` is the average pairwise Jaccard inside the cluster.
    For singletons (one member), there are no pairs so the value is
    ``1.0`` by convention — read as "perfectly internally consistent
    by vacuous truth", not as "1.0 similarity to anything else". If
    you need to distinguish singletons, check ``len(members) == 1``.
    """

    members: list[str]  # hypothesis ids
    representative: str  # the keeper (first member by input order)
    similarity: float  # average pairwise similarity inside the cluster


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _shingles(text: str, n: int = 3) -> set[tuple[str, ...]]:
    tokens = _tokenize(text)
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(a: str, b: str, *, n: int = 3) -> float:
    """Jaccard similarity over n-gram shingles. Pure stdlib."""
    sa = _shingles(a, n)
    sb = _shingles(b, n)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _payload_text(payload: dict[str, Any]) -> str:
    """Flatten a hypothesis payload into a single text blob for comparison.

    We concatenate string-valued leaves so the comparison is robust to
    schema differences. Numeric / boolean leaves are ignored — they
    don't carry the semantic signal we're deduplicating on.

    Self-referential structures are safe: ``_walk`` tracks visited
    container ids so a cycle (``d["self"] = d``) doesn't blow the
    Python recursion limit. Real LLM responses occasionally produce
    accidental cycles via orchestrator-side dict building.
    """
    parts: list[str] = []
    seen: set[int] = set()

    def _walk(value: Any) -> None:
        if isinstance(value, str):
            parts.append(value)
            return
        if isinstance(value, (dict, list, tuple)):
            ident = id(value)
            if ident in seen:
                return
            seen.add(ident)
            if isinstance(value, dict):
                for v in value.values():
                    _walk(v)
            else:
                for v in value:
                    _walk(v)

    _walk(payload)
    return " ".join(parts)


def cluster(
    hypotheses: list[dict[str, Any]],
    *,
    threshold: float = 0.7,
    backend: str = "shingle",
    n: int = 3,
) -> list[Cluster]:
    """Group near-duplicate hypotheses.

    Each hypothesis dict must have an ``id`` and a ``payload`` (a
    dict whose string leaves carry the semantic content).

    ``threshold`` is the minimum similarity to count two hypotheses
    as duplicates. ``backend='shingle'`` (default) uses Jaccard over
    n-gram shingles. ``backend='kmeans'`` requires sklearn (raises
    ProximityError if not installed).

    Returns clusters sorted by descending size. Singletons are
    included so the caller can round-trip every input.
    """
    if not hypotheses:
        return []
    if backend == "kmeans":
        return _cluster_kmeans(hypotheses, threshold=threshold, n=n)

    ids = [h["id"] for h in hypotheses]
    texts = [_payload_text(h.get("payload", {})) for h in hypotheses]

    # Union-find over pairs whose Jaccard ≥ threshold.
    parent = {hid: hid for hid in ids}

    def _find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: str, b: str) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    similarities: dict[tuple[str, str], float] = {}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            sim = jaccard(texts[i], texts[j], n=n)
            similarities[(ids[i], ids[j])] = sim
            if sim >= threshold:
                _union(ids[i], ids[j])

    groups: dict[str, list[str]] = {}
    for hid in ids:
        groups.setdefault(_find(hid), []).append(hid)

    clusters: list[Cluster] = []
    for members in groups.values():
        if len(members) == 1:
            avg = 1.0
        else:
            pair_sims = []
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    key = (members[i], members[j])
                    rev = (members[j], members[i])
                    pair_sims.append(similarities.get(key, similarities.get(rev, 0.0)))
            avg = sum(pair_sims) / len(pair_sims) if pair_sims else 0.0
        clusters.append(Cluster(
            members=members,
            representative=members[0],
            similarity=avg,
        ))

    clusters.sort(key=lambda c: (-len(c.members), -c.similarity))
    return clusters


def _cluster_kmeans(
    hypotheses: list[dict[str, Any]], *, threshold: float, n: int
) -> list[Cluster]:
    """Optional sklearn KMeans clusterer. Lazy import."""
    try:
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:
        raise ProximityError(
            "backend='kmeans' requires scikit-learn; pip install scikit-learn"
        ) from exc

    ids = [h["id"] for h in hypotheses]
    texts = [_payload_text(h.get("payload", {})) or " " for h in hypotheses]
    if len(set(texts)) < 2:
        return [Cluster(members=ids, representative=ids[0], similarity=1.0)]

    # Heuristic: k = ceil(n_hypotheses * threshold) bounded to
    # [1, n_hypotheses // 2]. Tighter threshold (higher value) →
    # fewer merges expected → more clusters. Looser threshold → more
    # merges → fewer clusters. The previous formula used (1-threshold)
    # which inverted the semantics (C3 fix).
    n_h = len(hypotheses)
    raw_k = max(1, int(round(n_h * threshold)))
    k = max(1, min(raw_k, max(1, n_h // 2)))

    vec = TfidfVectorizer(ngram_range=(1, n), min_df=1)
    matrix = vec.fit_transform(texts)
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = km.fit_predict(matrix)

    grouped: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        grouped.setdefault(int(label), []).append(idx)

    clusters: list[Cluster] = []
    for member_indices in grouped.values():
        members = [ids[i] for i in member_indices]
        if len(members) == 1:
            avg = 1.0
        else:
            sims = []
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    sims.append(jaccard(texts[member_indices[i]], texts[member_indices[j]], n=n))
            avg = sum(sims) / len(sims) if sims else 0.0
        clusters.append(Cluster(
            members=members, representative=members[0], similarity=avg
        ))
    clusters.sort(key=lambda c: (-len(c.members), -c.similarity))
    return clusters


def suggest_keepers(clusters: list[Cluster]) -> list[str]:
    """Return one representative id per cluster (input order preserved)."""
    return [c.representative for c in clusters]
