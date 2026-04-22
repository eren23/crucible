"""Doom-loop detection for autonomous agent loops.

An autonomous loop is in a "doom loop" when the LLM keeps making the
same decision despite repeated tool output — e.g. running the same
search 5× without using the result, or regenerating the same hypothesis
iteration after iteration.

Two detectors are provided:

1. :func:`detect` — general-purpose, works on any list of message-dicts
   that carry tool_call structure. Matches the tool-call histories that
   :class:`crucible.researcher.subagent.ResearchSubagent` and
   :mod:`crucible.cli.chat` build.

2. :func:`detect_research_loop` — researcher-specific. Inspects the
   :class:`crucible.researcher.state.ResearchState` for repeated
   hypothesis text, repeated experiment-config hashes, and stalled
   beliefs.

Both return either a short **corrective prompt** the loop should inject
as a user message before the next LLM call, or ``None`` if no loop is
detected. The detectors are pure — no I/O, no mutation.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from crucible.researcher.state import ResearchState


MessageDict = dict[str, Any]


def _args_hash(args: Any) -> str:
    """Stable hash of tool-call arguments."""
    try:
        blob = json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        blob = repr(args)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def _extract_tool_calls(history: list[MessageDict]) -> list[tuple[str, str]]:
    """Return (tool_name, args_hash) pairs from assistant messages."""
    pairs: list[tuple[str, str]] = []
    for msg in history:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    name = str(block.get("name", ""))
                    pairs.append((name, _args_hash(block.get("input", {}))))
        # Some codepaths flatten tool calls onto the message directly.
        for tc in msg.get("tool_calls", []) or []:
            if isinstance(tc, dict):
                name = str(tc.get("name") or tc.get("tool_name") or "")
                args = tc.get("arguments") or tc.get("input") or tc.get("args") or {}
                if name:
                    pairs.append((name, _args_hash(args)))
    return pairs


def detect(
    history: list[MessageDict],
    window: int = 10,
    threshold: int = 3,
) -> str | None:
    """Detect repetitive tool-call patterns in the last *window* turns.

    Triggers on any of:

    - Same ``(tool_name, args_hash)`` ≥ *threshold* times.
    - Same ``tool_name`` (any args) ≥ *threshold* + 2 times.
    - Two identical consecutive assistant text messages.

    Returns a short corrective prompt (to inject before the next LLM
    call) or ``None``.
    """
    if threshold < 2:
        raise ValueError("threshold must be >= 2")
    if not history:
        return None

    # (a) + (b): tool-call repetition
    recent = history[-window:]
    pairs = _extract_tool_calls(recent)
    if pairs:
        pair_counts = Counter(pairs)
        for (name, ah), n in pair_counts.most_common(1):
            if n >= threshold:
                return _prompt_tool_loop(name, n, specific=True)
        name_counts = Counter(name for name, _ in pairs)
        for name, n in name_counts.most_common(1):
            if n >= threshold + 2:
                return _prompt_tool_loop(name, n, specific=False)

    # (c): identical consecutive assistant text messages
    assistant_texts: list[str] = []
    for msg in reversed(recent):
        if msg.get("role") != "assistant":
            continue
        c = msg.get("content", "")
        text = c if isinstance(c, str) else _concat_text_blocks(c)
        text = text.strip()
        if not text:
            continue
        assistant_texts.append(text)
        if len(assistant_texts) == 2:
            break
    if len(assistant_texts) == 2 and assistant_texts[0] == assistant_texts[1]:
        return _prompt_repeat_message()

    return None


def detect_research_loop(state: "ResearchState", threshold: int = 3) -> str | None:
    """Researcher-specific detector operating on :class:`ResearchState`.

    Triggers when:

    - The same hypothesis text has appeared ≥ *threshold* times.
    - The same experiment config (by content hash) has been recorded
      ≥ *threshold* times in the last 2 × *threshold* runs.
    - Beliefs have not changed across the last *threshold* recorded
      hypothesis entries (implies reflection is making no progress).
    """
    if threshold < 2:
        raise ValueError("threshold must be >= 2")

    # Hypothesis repetition
    hyp_texts = [str(h.get("hypothesis", "")) for h in state.hypotheses if h.get("hypothesis")]
    if hyp_texts:
        counts = Counter(hyp_texts)
        top_text, n = counts.most_common(1)[0]
        if n >= threshold:
            return (
                f"You've proposed this hypothesis {n} times: \"{top_text[:120]}...\". "
                "Pick a different direction — vary the model family, tier, or swap the "
                "hyperparameter axis. If you've run out of ideas, ask the user."
            )

    # Experiment-config repetition
    recent_history = state.history[-(2 * threshold):]
    config_hashes = [_args_hash(rec.get("experiment", {}).get("config", {})) for rec in recent_history]
    if config_hashes:
        counts = Counter(config_hashes)
        _, n = counts.most_common(1)[0]
        if n >= threshold:
            return (
                f"The same experiment config has run {n} times without you changing it. "
                "Inspect the collected results and either (a) change at least one config "
                "key, or (b) promote/kill the current branch so a new hypothesis is chosen."
            )

    return None


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------


def _prompt_tool_loop(tool_name: str, n: int, *, specific: bool) -> str:
    scope = "with identical arguments" if specific else "with varying arguments"
    return (
        f"DOOM LOOP DETECTED: you have called `{tool_name}` {n}× {scope}. "
        "Stop and reconsider: (1) Is the tool response being used? "
        "(2) Is a different approach or tool needed? "
        "(3) Should you ask the user for clarification before continuing? "
        "Do NOT call this tool again until you can name a new reason."
    )


def _prompt_repeat_message() -> str:
    return (
        "DOOM LOOP DETECTED: your last two assistant messages were identical. "
        "You are stuck. Produce a genuinely different response: either make "
        "progress on a new subtask, summarize what you've learned, or ask "
        "the user for input."
    )


def _concat_text_blocks(blocks: Any) -> str:
    if not isinstance(blocks, list):
        return str(blocks)
    out: list[str] = []
    for b in blocks:
        if isinstance(b, dict) and b.get("type") == "text":
            out.append(str(b.get("text", "")))
    return "\n".join(out)
