"""Diff-XYZ format parsers + appliers.

Formats (per paper §5):
  - udiff:          standard unified diff with `@@ -a,b +c,d @@` hunk headers
  - udiff-h:        unified diff with relaxed `@@...@@` headers (no line nums)
  - udiff-l:        verbose `ADD`/`DEL`/`CON` markers instead of `+`/`-`/` `
  - search-replace: list of <<<<<<< SEARCH / ======= / >>>>>>> REPLACE blocks

All parsers raise `ParseError` on malformed input; all appliers raise
`ApplyError` when the ops don't match the old code.

Per paper §3.2, line-number anchors in udiff hunk headers are IGNORED during
application (they're used to disambiguate near-duplicate contexts, which is
<1% of the dataset). We do best-effort anchor-by-content.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Literal


class ParseError(Exception):
    """Raised when a diff string cannot be parsed in the expected format."""


class ApplyError(Exception):
    """Raised when parsed ops cannot be applied cleanly to the old code."""


# ---------------------------------------------------------------------------
# IR: a list of Hunks, each a sequence of context / add / delete ops.
# ---------------------------------------------------------------------------


OpKind = Literal["context", "add", "delete"]


@dataclass
class Op:
    kind: OpKind
    text: str  # single line without trailing newline


@dataclass
class Hunk:
    ops: list[Op] = field(default_factory=list)


@dataclass
class ParsedDiff:
    """Format-agnostic intermediate representation."""

    fmt: str
    hunks: list[Hunk] = field(default_factory=list)
    # Search-replace has no hunks in the udiff sense — we still fit it into Hunks
    # where each hunk's ops are the search lines (as deletes) + replace lines (as adds).


SUPPORTED_FORMATS = {"udiff", "udiff-h", "udiff-l", "search-replace"}


def parse_diff(diff_text: str, fmt: str) -> ParsedDiff:
    """Parse a diff string into a ParsedDiff IR."""
    if fmt == "udiff":
        return _parse_udiff(diff_text, header_required=True)
    if fmt == "udiff-h":
        return _parse_udiff(diff_text, header_required=False)
    if fmt == "udiff-l":
        return _parse_udiff_l(diff_text)
    if fmt == "search-replace":
        return _parse_search_replace(diff_text)
    raise ParseError(f"Unknown format {fmt!r}; supported: {sorted(SUPPORTED_FORMATS)}")


def apply_diff(old_code: str, parsed: ParsedDiff, fmt: str) -> str:
    """Apply a parsed diff to old_code. Ignores line-number anchors per paper §3.2."""
    if fmt == "search-replace":
        return _apply_search_replace(old_code, parsed)
    return _apply_udiff_family(old_code, parsed)


# ---------------------------------------------------------------------------
# Udiff parsing (handles both udiff and udiff-h)
# ---------------------------------------------------------------------------


_HUNK_HEADER_RE = re.compile(r"^@@\s*-?\d*,?\d*\s*\+?\d*,?\d*\s*@@")
_HUNK_HEADER_RELAXED_RE = re.compile(r"^@@.*?@@")


def _parse_udiff(diff_text: str, header_required: bool) -> ParsedDiff:
    """Parse udiff / udiff-h. Hunks delimited by `@@ ... @@` lines."""
    hunks: list[Hunk] = []
    cur: Hunk | None = None
    header_re = _HUNK_HEADER_RE if header_required else _HUNK_HEADER_RELAXED_RE

    for raw_line in diff_text.splitlines():
        # Skip file metadata lines (rare in Diff-XYZ but present in some git output).
        if raw_line.startswith(("diff --git ", "index ", "--- ", "+++ ")):
            continue
        if header_re.match(raw_line):
            cur = Hunk()
            hunks.append(cur)
            continue
        if cur is None:
            # Pre-hunk text — tolerated only if it's blank (some models preamble).
            if raw_line.strip():
                # Be strict in strict-header mode; lenient otherwise.
                if header_required:
                    raise ParseError(f"line before first hunk header: {raw_line!r}")
            continue
        # Classify by first character.
        if not raw_line:
            cur.ops.append(Op("context", ""))
            continue
        tag, body = raw_line[0], raw_line[1:]
        if tag == "+":
            cur.ops.append(Op("add", body))
        elif tag == "-":
            cur.ops.append(Op("delete", body))
        elif tag == " ":
            cur.ops.append(Op("context", body))
        else:
            # Some tools omit the leading space on unchanged lines. Treat as context.
            cur.ops.append(Op("context", raw_line))

    if not hunks:
        raise ParseError("no hunks found in udiff input")
    return ParsedDiff(fmt="udiff", hunks=hunks)


# ---------------------------------------------------------------------------
# Udiff-l: verbose ADD/DEL/CON markers
# ---------------------------------------------------------------------------


def _parse_udiff_l(diff_text: str) -> ParsedDiff:
    """Parse udiff-l: lines start with 'ADD ', 'DEL ', or 'CON '."""
    hunks: list[Hunk] = []
    cur = Hunk()
    hunks.append(cur)
    for raw_line in diff_text.splitlines():
        if raw_line.startswith("@@"):
            cur = Hunk()
            hunks.append(cur)
            continue
        if not raw_line.strip():
            continue
        prefix = raw_line[:4]
        body = raw_line[4:]
        if prefix == "ADD ":
            cur.ops.append(Op("add", body))
        elif prefix == "DEL ":
            cur.ops.append(Op("delete", body))
        elif prefix == "CON ":
            cur.ops.append(Op("context", body))
        else:
            raise ParseError(f"unknown udiff-l prefix in line: {raw_line!r}")
    # Drop empty leading hunk if model never emitted an @@ and we pre-seeded one.
    hunks = [h for h in hunks if h.ops]
    if not hunks:
        raise ParseError("no ops found in udiff-l input")
    return ParsedDiff(fmt="udiff-l", hunks=hunks)


# ---------------------------------------------------------------------------
# Search-replace format (Aider-style)
# ---------------------------------------------------------------------------


_SR_SEARCH_MARK = "<<<<<<< SEARCH"
_SR_DIVIDER = "======="
_SR_REPLACE_MARK = ">>>>>>> REPLACE"


def _parse_search_replace(diff_text: str) -> ParsedDiff:
    """Parse `<<<<<<< SEARCH / ======= / >>>>>>> REPLACE` blocks."""
    lines = diff_text.splitlines()
    hunks: list[Hunk] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == _SR_SEARCH_MARK:
            search_body: list[str] = []
            replace_body: list[str] = []
            i += 1
            # search body until divider
            while i < len(lines) and lines[i].strip() != _SR_DIVIDER:
                search_body.append(lines[i])
                i += 1
            if i >= len(lines):
                raise ParseError("search-replace: unterminated SEARCH block (missing '=======')")
            i += 1  # skip divider
            # replace body until REPLACE mark
            while i < len(lines) and lines[i].strip() != _SR_REPLACE_MARK:
                replace_body.append(lines[i])
                i += 1
            if i >= len(lines):
                raise ParseError("search-replace: unterminated REPLACE block (missing '>>>>>>> REPLACE')")
            # Build a hunk: deletes for search body, adds for replace body.
            ops = [Op("delete", s) for s in search_body] + [Op("add", r) for r in replace_body]
            hunks.append(Hunk(ops=ops))
        i += 1
    if not hunks:
        raise ParseError("no SEARCH/REPLACE blocks found")
    return ParsedDiff(fmt="search-replace", hunks=hunks)


def _apply_search_replace(old_code: str, parsed: ParsedDiff) -> str:
    """Apply search-replace by literal string substitution, one hunk at a time."""
    result = old_code
    for hunk in parsed.hunks:
        search_lines = [op.text for op in hunk.ops if op.kind == "delete"]
        replace_lines = [op.text for op in hunk.ops if op.kind == "add"]
        search = "\n".join(search_lines)
        replace = "\n".join(replace_lines)
        if search == "":
            # Insertion at top (SEARCH block empty) — Aider convention.
            result = replace + ("\n" + result if result else "")
            continue
        if search not in result:
            raise ApplyError(f"SEARCH block not found in old code: {search[:60]!r}...")
        # Replace only the first occurrence (paper's app preserves ordering).
        result = result.replace(search, replace, 1)
    return result


# ---------------------------------------------------------------------------
# Udiff-family application (anchor-by-content, ignoring line numbers)
# ---------------------------------------------------------------------------


def _apply_udiff_family(old_code: str, parsed: ParsedDiff) -> str:
    """Anchor-by-content application. Finds each hunk's context+delete lines in
    old_code, splices in the adds.
    """
    old_lines = old_code.split("\n")
    out_lines: list[str] = []
    cursor = 0

    for hunk in parsed.hunks:
        # The "before" side = context + delete lines (in order).
        before: list[str] = [op.text for op in hunk.ops if op.kind in ("context", "delete")]
        # The "after" side = context + add lines (in order).
        after: list[str] = [op.text for op in hunk.ops if op.kind in ("context", "add")]

        if not before:
            # Pure insertion hunk (no anchors) — append to tail.
            out_lines.extend(old_lines[cursor:])
            cursor = len(old_lines)
            out_lines.extend(after)
            continue

        # Locate `before` sequence in old_lines starting at cursor.
        idx = _find_sequence(old_lines, before, start=cursor)
        if idx < 0:
            raise ApplyError(
                f"hunk does not match old_code at or after line {cursor}: "
                f"before[0..3]={before[:3]!r}"
            )
        # Emit everything up to the match, then the 'after' block.
        out_lines.extend(old_lines[cursor:idx])
        out_lines.extend(after)
        cursor = idx + len(before)

    out_lines.extend(old_lines[cursor:])
    return "\n".join(out_lines)


def _find_sequence(haystack: list[str], needle: list[str], start: int = 0) -> int:
    """Return index of needle in haystack (starting at `start`), or -1."""
    if not needle:
        return start
    n = len(needle)
    for i in range(start, len(haystack) - n + 1):
        if haystack[i:i + n] == needle:
            return i
    return -1


# ---------------------------------------------------------------------------
# Line-set extraction for F1+/F1-
# ---------------------------------------------------------------------------


def diff_added_lines(diff_text: str, fmt: str) -> set[str]:
    """Set of added-line contents, silent on parse errors (returns empty set)."""
    try:
        parsed = parse_diff(diff_text, fmt)
    except ParseError:
        return set()
    return _flatten_ops(parsed.hunks, "add")


def diff_deleted_lines(diff_text: str, fmt: str) -> set[str]:
    """Set of deleted-line contents, silent on parse errors."""
    try:
        parsed = parse_diff(diff_text, fmt)
    except ParseError:
        return set()
    return _flatten_ops(parsed.hunks, "delete")


def _flatten_ops(hunks: Iterable[Hunk], kind: OpKind) -> set[str]:
    return {op.text for hunk in hunks for op in hunk.ops if op.kind == kind}
