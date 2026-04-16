"""Tests for the evolution log (append-only JSONL per iteration)."""
from __future__ import annotations

from pathlib import Path

from crucible.researcher.evolution_log import (
    append_iteration,
    last_iteration,
    log_path,
    read_log,
)


def test_append_and_read(tmp_path: Path) -> None:
    rec = append_iteration(
        tmp_path,
        iteration=1,
        proposed=3,
        validated=2,
        benchmarked=2,
        frontier_summary={"frontier_size": 2},
        cost={"tokens": 100},
        notes="first run",
    )
    assert rec["iteration"] == 1
    assert rec["proposed"] == 3
    assert rec["timestamp"]
    records = read_log(tmp_path)
    assert len(records) == 1
    assert records[0]["notes"] == "first run"


def test_append_preserves_ordering(tmp_path: Path) -> None:
    for i in range(1, 4):
        append_iteration(tmp_path, iteration=i, proposed=i, validated=i)
    records = read_log(tmp_path)
    assert [r["iteration"] for r in records] == [1, 2, 3]


def test_last_iteration_empty_returns_zero(tmp_path: Path) -> None:
    assert last_iteration(tmp_path) == 0


def test_last_iteration_returns_max(tmp_path: Path) -> None:
    append_iteration(tmp_path, iteration=3, proposed=0, validated=0)
    append_iteration(tmp_path, iteration=1, proposed=0, validated=0)
    append_iteration(tmp_path, iteration=5, proposed=0, validated=0)
    assert last_iteration(tmp_path) == 5


def test_extra_fields_are_merged(tmp_path: Path) -> None:
    rec = append_iteration(
        tmp_path,
        iteration=1,
        proposed=1,
        validated=1,
        extra={"llm_model": "claude-opus-4-6"},
    )
    assert rec["llm_model"] == "claude-opus-4-6"


def test_log_path_points_to_jsonl(tmp_path: Path) -> None:
    assert log_path(tmp_path).name == "evolution_log.jsonl"
