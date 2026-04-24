"""Diff-XYZ eval harness (arXiv 2510.12487).

Generative benchmark for LLM code-diff understanding:
  - Apply:       old_code + diff -> new_code
  - Anti-Apply:  new_code + diff -> old_code
  - Diff-Gen:    old_code + new_code -> diff

Public API:
    from evaluation.diff_xyz import compute_metrics, parse_diff, apply_diff
"""
from __future__ import annotations

from evaluation.diff_xyz.formats import (
    ApplyError,
    ParseError,
    apply_diff,
    parse_diff,
)
from evaluation.diff_xyz.metrics import (
    compute_apply_metrics,
    compute_diff_gen_metrics,
    f1_added,
    f1_deleted,
    stripped_em,
    stripped_iou,
)

__all__ = [
    "ApplyError",
    "ParseError",
    "apply_diff",
    "compute_apply_metrics",
    "compute_diff_gen_metrics",
    "f1_added",
    "f1_deleted",
    "parse_diff",
    "stripped_em",
    "stripped_iou",
]
