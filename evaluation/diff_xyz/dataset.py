"""Load the Diff-XYZ test set from HuggingFace.

Dataset: `JetBrains-Research/diff-xyz`. Single `test` split, 1000 rows,
columns per dataset card: repo, commit, path, lang, license, message,
old_code, new_code, n_added, n_removed, n_hunks, change_kind,
udiff, udiff-h, udiff-l, search-replace.

`load_samples(limit, langs, seed)` returns a list of `DiffXYZSample` objects.
It does NOT cache — callers that want caching should wrap with
`datasets.load_dataset`'s own cache, which kicks in automatically.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

DATASET_ID = "JetBrains-Research/diff-xyz"


@dataclass
class DiffXYZSample:
    """One row from the Diff-XYZ test set."""

    repo: str
    commit: str
    path: str
    lang: str
    old_code: str
    new_code: str
    udiff: str
    udiff_h: str
    udiff_l: str
    search_replace: str
    n_added: int
    n_removed: int
    n_hunks: int
    change_kind: str

    def diff_for(self, fmt: str) -> str:
        """Return the pre-rendered diff string for a given format."""
        if fmt == "udiff":
            return self.udiff
        if fmt == "udiff-h":
            return self.udiff_h
        if fmt == "udiff-l":
            return self.udiff_l
        if fmt == "search-replace":
            return self.search_replace
        raise ValueError(f"unknown format {fmt!r}")


def load_samples(
    limit: int | None = None,
    langs: Sequence[str] | None = None,
    seed: int = 0,
) -> list[DiffXYZSample]:
    """Load Diff-XYZ samples, optionally filtered by language and limit-capped.

    Args:
        limit: max samples to return (None = all).
        langs: list of language codes to keep (e.g., ['python', 'java']);
            None keeps all 5 languages.
        seed: RNG seed for reproducible subsampling when `limit` is set.

    Returns:
        list of `DiffXYZSample` in deterministic order.
    """
    from datasets import load_dataset  # type: ignore[import-not-found]

    ds = load_dataset(DATASET_ID, split="test")
    rows = [_row_to_sample(row) for row in ds]

    if langs:
        wanted = {ln.lower() for ln in langs}
        rows = [r for r in rows if r.lang.lower() in wanted]

    if limit is not None and limit < len(rows):
        import random
        rng = random.Random(seed)
        rows = rng.sample(rows, limit)

    return rows


def _row_to_sample(row: dict) -> DiffXYZSample:
    """Coerce one HF row into a DiffXYZSample."""
    return DiffXYZSample(
        repo=str(row.get("repo", "")),
        commit=str(row.get("commit", "")),
        path=str(row.get("path", "")),
        lang=str(row.get("lang", "")),
        old_code=str(row.get("old_code", "")),
        new_code=str(row.get("new_code", "")),
        udiff=str(row.get("udiff", "")),
        udiff_h=str(row.get("udiff-h", row.get("udiff_h", ""))),
        udiff_l=str(row.get("udiff-l", row.get("udiff_l", ""))),
        search_replace=str(row.get("search-replace", row.get("search_replace", ""))),
        n_added=int(row.get("n_added", 0)),
        n_removed=int(row.get("n_removed", 0)),
        n_hunks=int(row.get("n_hunks", 0)),
        change_kind=str(row.get("change_kind", "")),
    )
