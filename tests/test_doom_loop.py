"""Unit tests for core/doom_loop.py."""
from __future__ import annotations

import pytest

from crucible.core.doom_loop import detect, detect_research_loop


def _asst_tool(name: str, args: dict) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "name": name, "input": args}],
    }


def _asst_text(text: str) -> dict:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def test_empty_history_returns_none():
    assert detect([]) is None


def test_threshold_validation():
    with pytest.raises(ValueError):
        detect([], threshold=1)


def test_no_loop_on_varied_tools():
    history = [
        _asst_tool("search", {"q": "a"}),
        _asst_tool("read", {"path": "x"}),
        _asst_tool("search", {"q": "b"}),
    ]
    assert detect(history, threshold=3) is None


def test_detects_identical_args_loop():
    history = [_asst_tool("search", {"q": "a"}) for _ in range(3)]
    msg = detect(history, threshold=3)
    assert msg is not None
    assert "search" in msg
    assert "identical arguments" in msg


def test_detects_varied_args_loop():
    """Same tool but different args triggers on threshold+2."""
    history = [_asst_tool("search", {"q": f"q{i}"}) for i in range(5)]
    msg = detect(history, threshold=3)
    assert msg is not None
    assert "search" in msg
    assert "varying arguments" in msg


def test_detects_identical_consecutive_messages():
    history = [
        _asst_text("I need to think about this"),
        _asst_text("I need to think about this"),
    ]
    msg = detect(history, threshold=3)
    assert msg is not None
    assert "identical" in msg.lower()


def test_window_limits_detection():
    """Old repetitions outside the window should not trigger."""
    history = [_asst_tool("search", {"q": "same"}) for _ in range(3)]
    # Fresh messages pushing old ones out of window — all distinct tool names
    history.extend([_asst_tool(f"unique_{i}", {}) for i in range(5)])
    assert detect(history, window=5, threshold=3) is None


class _FakeState:
    def __init__(self, hypotheses=None, history=None):
        self.hypotheses = hypotheses or []
        self.history = history or []


def test_research_loop_hypothesis_repetition():
    state = _FakeState(hypotheses=[{"hypothesis": "try deeper net"} for _ in range(4)])
    msg = detect_research_loop(state, threshold=3)
    assert msg is not None
    assert "deeper net" in msg or "hypothesis" in msg.lower()


def test_research_loop_config_repetition():
    cfg = {"MODEL_FAMILY": "baseline", "LR": "0.001"}
    state = _FakeState(
        history=[{"experiment": {"config": cfg}} for _ in range(5)],
    )
    msg = detect_research_loop(state, threshold=3)
    assert msg is not None
    assert "config" in msg.lower()


def test_research_loop_no_false_positive():
    state = _FakeState(
        hypotheses=[{"hypothesis": f"try variant {i}"} for i in range(3)],
        history=[{"experiment": {"config": {"LR": f"0.00{i}"}}} for i in range(3)],
    )
    assert detect_research_loop(state, threshold=3) is None
