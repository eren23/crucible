"""Diff-XYZ metrics (paper §3.1 + §3.2 verbatim).

- Stripped EM: whitespace-only lines removed, then exact string equality
- Stripped IoU: |unique lines A ∩ B| / |unique lines A ∪ B| after whitespace-strip
- F1 on added lines: F1 between {added_lines in ref} and {added_lines in pred}
- F1 on deleted lines: analogous
- Parsing rate / Applying rate: fractions of parseable / successfully-applied diffs

All four main metrics operate on strings; F1 functions take line-set inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


# ---------------------------------------------------------------------------
# Core text helpers
# ---------------------------------------------------------------------------


def strip_whitespace_lines(text: str) -> str:
    """Remove lines containing only whitespace, per paper §3.1."""
    kept = [line for line in text.splitlines() if line.strip()]
    return "\n".join(kept)


def unique_lines(text: str) -> set[str]:
    """Unique non-whitespace lines (after stripping ambient whitespace)."""
    return {line for line in text.splitlines() if line.strip()}


# ---------------------------------------------------------------------------
# Primary metrics
# ---------------------------------------------------------------------------


def stripped_em(predicted: str, reference: str) -> float:
    """1.0 iff the two snippets are identical after whitespace-line removal.

    Per paper §3.1: "Stripped Exact Match — EM — is 1 when two processed code
    snippets are exactly the same, 0 otherwise."
    """
    return 1.0 if strip_whitespace_lines(predicted) == strip_whitespace_lines(reference) else 0.0


def stripped_iou(predicted: str, reference: str) -> float:
    """Jaccard over unique non-empty lines. Empty union returns 1.0 (both empty)."""
    a = unique_lines(predicted)
    b = unique_lines(reference)
    if not a and not b:
        return 1.0
    union = a | b
    inter = a & b
    return len(inter) / len(union) if union else 0.0


# ---------------------------------------------------------------------------
# F1 on line sets (Diff-Gen, per paper §3.2)
# ---------------------------------------------------------------------------


def f1_score(predicted: Iterable[str], reference: Iterable[str]) -> float:
    """Set-based F1. Empty-set corner case returns 1.0 iff both empty, else 0.0."""
    p = set(predicted)
    r = set(reference)
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    tp = len(p & r)
    if tp == 0:
        return 0.0
    precision = tp / len(p)
    recall = tp / len(r)
    return 2 * precision * recall / (precision + recall)


def extract_added_lines(diff_text: str, fmt: str) -> set[str]:
    """Pull the set of added-line contents from a diff string, per format."""
    from evaluation.diff_xyz.formats import diff_added_lines
    return diff_added_lines(diff_text, fmt)


def extract_deleted_lines(diff_text: str, fmt: str) -> set[str]:
    """Pull the set of deleted-line contents from a diff string, per format."""
    from evaluation.diff_xyz.formats import diff_deleted_lines
    return diff_deleted_lines(diff_text, fmt)


def f1_added(predicted_diff: str, reference_diff: str, fmt: str) -> float:
    """F1+ per paper §3.2."""
    return f1_score(
        extract_added_lines(predicted_diff, fmt),
        extract_added_lines(reference_diff, fmt),
    )


def f1_deleted(predicted_diff: str, reference_diff: str, fmt: str) -> float:
    """F1- per paper §3.2."""
    return f1_score(
        extract_deleted_lines(predicted_diff, fmt),
        extract_deleted_lines(reference_diff, fmt),
    )


# ---------------------------------------------------------------------------
# Aggregate computation
# ---------------------------------------------------------------------------


@dataclass
class ApplyTaskMetrics:
    """Apply / Anti-Apply: EM + IoU on generated code snippet vs reference."""

    em: float
    iou: float


@dataclass
class DiffGenMetrics:
    """Diff-Gen: EM/IoU after application + F1+/F1- on line sets + rates."""

    em: float              # EM on (apply(old, gen_diff), new); 0 if apply fails
    iou: float             # IoU on same; 0 if apply fails
    parsing_rate: float    # fraction of diffs that parsed (0 or 1 here; aggregate over samples)
    applying_rate: float   # fraction that applied cleanly
    f1_plus: float         # set-based F1 on added lines (predicted vs reference diff)
    f1_minus: float        # set-based F1 on deleted lines


def compute_apply_metrics(predicted_code: str, reference_code: str) -> ApplyTaskMetrics:
    """Score an Apply or Anti-Apply prediction."""
    return ApplyTaskMetrics(
        em=stripped_em(predicted_code, reference_code),
        iou=stripped_iou(predicted_code, reference_code),
    )


def compute_diff_gen_metrics(
    predicted_diff: str,
    reference_diff: str,
    old_code: str,
    new_code: str,
    fmt: str,
) -> DiffGenMetrics:
    """Score a Diff-Gen prediction: parse → apply → compare; compute F1 on line sets."""
    from evaluation.diff_xyz.formats import ApplyError, ParseError, apply_diff, parse_diff

    parsing_rate = 0.0
    applying_rate = 0.0
    em = 0.0
    iou = 0.0
    try:
        ops = parse_diff(predicted_diff, fmt)
        parsing_rate = 1.0
    except ParseError:
        ops = None
    if ops is not None:
        try:
            applied = apply_diff(old_code, ops, fmt)
            applying_rate = 1.0
            em = stripped_em(applied, new_code)
            iou = stripped_iou(applied, new_code)
        except ApplyError:
            pass
    return DiffGenMetrics(
        em=em,
        iou=iou,
        parsing_rate=parsing_rate,
        applying_rate=applying_rate,
        f1_plus=f1_added(predicted_diff, reference_diff, fmt),
        f1_minus=f1_deleted(predicted_diff, reference_diff, fmt),
    )
