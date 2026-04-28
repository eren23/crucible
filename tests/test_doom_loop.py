"""Unit tests for core/doom_loop.py."""
from __future__ import annotations

import pytest

from crucible.core.doom_loop import detect, detect_research_loop


def _asst_tool(name: str, args: dict, tu_id: str | None = None) -> dict:
    block: dict = {"type": "tool_use", "name": name, "input": args}
    if tu_id:
        block["id"] = tu_id
    return {"role": "assistant", "content": [block]}


def _tool_result(tu_id: str, result: object) -> dict:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tu_id, "content": result}],
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


def test_polling_with_varying_results_suppressed():
    """Same tool + same args 5x but with varying results = legitimate polling."""
    history: list[dict] = []
    for i in range(5):
        tu = f"call_{i}"
        history.append(_asst_tool("get_fleet_status", {"project": "x"}, tu_id=tu))
        history.append(_tool_result(tu, {"running": i, "ts": i}))  # different each time
    assert detect(history, window=20, threshold=3) is None


def test_polling_stuck_with_identical_results_fires():
    """Same tool + same args + same result 3x = real stuck loop."""
    history: list[dict] = []
    for i in range(3):
        tu = f"call_{i}"
        history.append(_asst_tool("get_status", {"id": "x"}, tu_id=tu))
        history.append(_tool_result(tu, {"state": "queued"}))  # identical each time
    msg = detect(history, window=20, threshold=3)
    assert msg is not None
    assert "get_status" in msg


def test_cycle_detection_two_tool_pattern():
    """[A B] [A B] cycle at the tail should fire the cycle warning."""
    history = [
        _asst_tool("dispatch", {}),
        _asst_tool("get_status", {}),
        _asst_tool("dispatch", {}),
        _asst_tool("get_status", {}),
    ]
    msg = detect(history, window=20, threshold=10)  # threshold high so only cycle rule can fire
    assert msg is not None
    assert "cycle" in msg.lower()
    assert "dispatch" in msg
    assert "get_status" in msg


def test_cycle_detection_three_tool_pattern():
    history = []
    for _ in range(2):
        history.append(_asst_tool("a", {}))
        history.append(_asst_tool("b", {}))
        history.append(_asst_tool("c", {}))
    msg = detect(history, window=20, threshold=10)
    assert msg is not None
    assert "cycle" in msg.lower()


def test_cycle_no_false_positive_on_single_pass():
    """[A B C] only once — not a cycle."""
    history = [_asst_tool("a", {}), _asst_tool("b", {}), _asst_tool("c", {})]
    assert detect(history, window=20, threshold=10) is None


def test_multiple_id_less_tool_results_in_one_message_pair_in_order():
    """Regression: when an assistant message emits multiple id-less tool_use
    blocks and a single user message follows with multiple tool_result blocks
    (no tool_use_id), the earlier code stored only the FIRST result hash per
    message index, leaving the rest of the tool_uses with empty result_hash.

    With the queue-per-index fix, each tool_result is consumed in FIFO order,
    so all calls pair correctly. Concretely: two identical tool_use blocks
    (same args), each followed by DIFFERENT results, must be treated as
    legitimate polling — not a doom loop.
    """
    from crucible.core.doom_loop import _extract_tool_signatures

    history: list[dict] = [
        # Assistant message with TWO tool_use blocks (no ids)
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "name": "poll", "input": {"k": "x"}},
                {"type": "tool_use", "name": "poll", "input": {"k": "x"}},
            ],
        },
        # User message with TWO tool_result blocks (no tool_use_ids)
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "content": {"state": "running"}},
                {"type": "tool_result", "content": {"state": "done"}},
            ],
        },
    ]
    triples = _extract_tool_signatures(history)
    # Both tool_uses get distinct result_hashes (not "" and not the same).
    assert len(triples) == 2
    rh1, rh2 = triples[0][2], triples[1][2]
    assert rh1 and rh2, "both calls should be paired with non-empty result hashes"
    assert rh1 != rh2, "different tool_results should produce different hashes"


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
