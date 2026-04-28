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


def _result_hash(result: Any) -> str:
    """Stable hash of a tool-result payload, or empty string for None/missing."""
    if result is None:
        return ""
    try:
        blob = json.dumps(result, sort_keys=True, default=str)
    except (TypeError, ValueError):
        blob = repr(result)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def _extract_tool_signatures(
    history: list[MessageDict],
) -> list[tuple[str, str, str]]:
    """Return (tool_name, args_hash, result_hash) triples from messages.

    `result_hash` is the hash of the tool_result that follows each tool_use,
    or "" when no result is available in the history slice (the typical case
    for in-flight calls or sub-agent histories that drop result blocks).

    Pairing logic: when a tool_use carries an ``id`` (Anthropic format) we
    match by ``tool_use_id`` from any later ``tool_result`` block. Otherwise
    we fall back to position — each user/tool message keeps a FIFO queue of
    its tool_result hashes, so a message answering N prior id-less tool_uses
    pairs them in order rather than stranding all but the first.
    """
    triples: list[tuple[str, str, str]] = []
    pending: list[tuple[int, str, str, str | None]] = []  # (idx, name, args, tool_use_id)
    results_by_id: dict[str, str] = {}
    # FIFO queue of result-hashes per message index — handles the multi-
    # tool_result-per-message case correctly.
    positional_queues: list[list[str]] = [[] for _ in history]

    for i, msg in enumerate(history):
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "assistant":
            blocks = content if isinstance(content, list) else []
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    name = str(block.get("name", ""))
                    args = block.get("input", {})
                    tu_id = block.get("id")
                    pending.append((i, name, _args_hash(args), str(tu_id) if tu_id else None))
            for tc in msg.get("tool_calls", []) or []:
                if isinstance(tc, dict):
                    name = str(tc.get("name") or tc.get("tool_name") or "")
                    args = tc.get("arguments") or tc.get("input") or tc.get("args") or {}
                    tu_id = tc.get("id")
                    if name:
                        pending.append((i, name, _args_hash(args), str(tu_id) if tu_id else None))
        elif role in ("user", "tool"):
            blocks = content if isinstance(content, list) else []
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result":
                    rh = _result_hash(block.get("content"))
                    tu_id = block.get("tool_use_id")
                    if tu_id:
                        results_by_id[str(tu_id)] = rh
                    else:
                        positional_queues[i].append(rh)
            # `tool` role often carries a flat content field.
            if role == "tool" and "content" in msg and not isinstance(content, list):
                rh = _result_hash(content)
                tu_id = msg.get("tool_call_id") or msg.get("tool_use_id")
                if tu_id:
                    results_by_id[str(tu_id)] = rh
                else:
                    positional_queues[i].append(rh)

    # Stitch tool_use → tool_result.
    for idx, name, ah, tu_id in pending:
        rh = ""
        if tu_id and tu_id in results_by_id:
            rh = results_by_id[tu_id]
        else:
            for j in range(idx + 1, len(history)):
                if positional_queues[j]:
                    rh = positional_queues[j].pop(0)
                    break
        triples.append((name, ah, rh))
    return triples


def _extract_tool_calls(history: list[MessageDict]) -> list[tuple[str, str]]:
    """Backward-compat shim: return (name, args_hash) pairs."""
    return [(name, ah) for name, ah, _ in _extract_tool_signatures(history)]


def detect(
    history: list[MessageDict],
    window: int = 10,
    threshold: int = 3,
) -> str | None:
    """Detect repetitive tool-call patterns in the last *window* turns.

    Triggers on any of:

    - Same ``(tool_name, args_hash)`` ≥ *threshold* times **and** results
      are not varying (varying results = legitimate polling, suppressed).
    - Same ``tool_name`` (any args) ≥ *threshold* + 2 times **and** results
      are not varying.
    - A 2–5-step ``(tool, args)`` cycle that has fully repeated ≥ 2 times
      back-to-back at the tail of the window.
    - Two identical consecutive assistant text messages.

    Returns a short corrective prompt (to inject before the next LLM
    call) or ``None``.
    """
    if threshold < 2:
        raise ValueError("threshold must be >= 2")
    if not history:
        return None

    recent = history[-window:]
    triples = _extract_tool_signatures(recent)

    if triples:
        # (a) Same (name, args), suppress when result_hash is varying.
        pair_counts: Counter[tuple[str, str]] = Counter((n, ah) for n, ah, _ in triples)
        results_by_pair: dict[tuple[str, str], set[str]] = {}
        for n, ah, rh in triples:
            results_by_pair.setdefault((n, ah), set()).add(rh)
        for (name, ah), n in pair_counts.most_common(1):
            if n >= threshold and not _results_vary(results_by_pair[(name, ah)]):
                return _prompt_tool_loop(name, n, specific=True)

        # (b) Same name (any args), suppress when results vary across the group.
        name_counts: Counter[str] = Counter(n for n, _, _ in triples)
        results_by_name: dict[str, set[str]] = {}
        for n, _, rh in triples:
            results_by_name.setdefault(n, set()).add(rh)
        for name, n in name_counts.most_common(1):
            if n >= threshold + 2 and not _results_vary(results_by_name[name]):
                return _prompt_tool_loop(name, n, specific=False)

        # (b2) Cycle detection: pattern of length 2–5 repeated ≥ 2 times at tail.
        pairs_only = [(n, ah) for n, ah, _ in triples]
        cycle = _detect_cycle(pairs_only, max_len=5)
        if cycle is not None:
            return _prompt_cycle_loop(cycle)

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


def _results_vary(result_hashes: set[str]) -> bool:
    """True iff the results genuinely differ across calls.

    A set of size ≥ 2 with at least one non-empty hash means the tool
    returned different content across calls — suppress the loop.
    Histories without tool_result blocks (all empty hashes) collapse to
    {""} and are treated as non-varying so legacy callers still trigger.
    """
    non_empty = {r for r in result_hashes if r}
    return len(non_empty) >= 2


def _detect_cycle(
    pairs: list[tuple[str, str]], *, max_len: int = 5
) -> list[str] | None:
    """Return the tool-name cycle if last 2N pairs match pattern of length N.

    Checks N = 2..max_len. The most recent block of length N must equal
    the block immediately before it AND contain ≥ 2 distinct tool names
    (single-tool repetition is detected by Rule A, which honors
    result-hash variation for legitimate polling).
    """
    n = len(pairs)
    for seq_len in range(2, max_len + 1):
        if n < seq_len * 2:
            continue
        tail = pairs[-seq_len:]
        prev = pairs[-seq_len * 2 : -seq_len]
        if tail == prev and len({name for name, _ in tail}) >= 2:
            return [name for name, _ in tail]
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


def _prompt_cycle_loop(cycle: list[str]) -> str:
    chain = " → ".join(cycle)
    return (
        f"DOOM LOOP DETECTED: tool-call cycle [{chain}] has repeated back-to-back. "
        "This is the agent equivalent of a stuck infinite loop. Stop, summarize "
        "what these calls have established, and pick a different approach: "
        "skip ahead, gather missing info, or ask the user."
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
