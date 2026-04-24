"""System + user prompt templates for Diff-XYZ tasks.

Paper Appendix A describes the prompts but the arxiv HTML export renders them
as figure placeholders without figure content. The templates below are a
best-effort reproduction matching the paper's description:

  - `w/o format` system prompt: literally "You are a helpful assistant."
  - `w/format` system prompt: describes the target diff format and gives one
    worked example (paper A.3 Figures 6–9).

User prompts for each task (A.2 Figures 3–5):
  - Apply:       given old_code + diff → produce new_code (code only).
  - Anti-Apply:  given new_code + diff → produce old_code (code only).
  - Diff-Gen:    given old_code + new_code → produce diff in specified format.

Tuning these to exactly match the paper's numbers is a reproduction task
(paper's Table 1 reference row: Claude 4 Sonnet udiff w/format Apply EM 0.96).
If our reproduction misses by > 2pp, revisit these.
"""
from __future__ import annotations

from evaluation.diff_xyz.formats import SUPPORTED_FORMATS

TASKS = ("apply", "anti_apply", "diff_gen")
SYSTEM_PROMPT_MODES = ("none", "format")


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------


GENERIC_SYSTEM = "You are a helpful assistant."


_FORMAT_DESCRIPTIONS: dict[str, str] = {
    "udiff": (
        "You will work with diffs in the unified diff format.\n"
        "Each hunk starts with a header like `@@ -a,b +c,d @@` where a,b are\n"
        "the start line and count in the old file and c,d are the start line\n"
        "and count in the new file. Lines in a hunk are prefixed with:\n"
        "  '+' for an added line,\n"
        "  '-' for a deleted line,\n"
        "  ' ' (single space) for an unchanged context line.\n"
        "Example:\n"
        "```\n"
        "@@ -1,3 +1,3 @@\n"
        " line_a\n"
        "-line_b\n"
        "+line_B\n"
        " line_c\n"
        "```\n"
    ),
    "udiff-h": (
        "You will work with diffs in a relaxed unified diff format.\n"
        "Each hunk starts with `@@ ... @@` (no line numbers). Lines inside a\n"
        "hunk are prefixed with '+' (added), '-' (deleted), or ' ' (context).\n"
        "Example:\n"
        "```\n"
        "@@ ... @@\n"
        " line_a\n"
        "-line_b\n"
        "+line_B\n"
        " line_c\n"
        "```\n"
    ),
    "udiff-l": (
        "You will work with diffs using explicit line tags.\n"
        "Each line is prefixed with `ADD `, `DEL `, or `CON ` (four chars\n"
        "including the trailing space). Hunks are optionally separated by\n"
        "`@@ ... @@`.\n"
        "Example:\n"
        "```\n"
        "CON line_a\n"
        "DEL line_b\n"
        "ADD line_B\n"
        "CON line_c\n"
        "```\n"
    ),
    "search-replace": (
        "You will work with diffs in the search-replace format.\n"
        "Each edit is a block delimited by these three markers on their own lines:\n"
        "  `<<<<<<< SEARCH`\n"
        "  `=======`\n"
        "  `>>>>>>> REPLACE`\n"
        "The SEARCH block contains the exact old lines to find; the REPLACE\n"
        "block contains the new lines to substitute.\n"
        "Example:\n"
        "```\n"
        "<<<<<<< SEARCH\n"
        "old_fn()\n"
        "=======\n"
        "new_fn()\n"
        ">>>>>>> REPLACE\n"
        "```\n"
        "Emit one block per distinct edit, in file order.\n"
    ),
}


def system_prompt(fmt: str, mode: str) -> str:
    """Return the system prompt for (format, mode) combination."""
    if mode not in SYSTEM_PROMPT_MODES:
        raise ValueError(f"unknown system prompt mode {mode!r}; use 'none' or 'format'")
    if mode == "none":
        return GENERIC_SYSTEM
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"unknown format {fmt!r}")
    return GENERIC_SYSTEM + "\n\n" + _FORMAT_DESCRIPTIONS[fmt]


# ---------------------------------------------------------------------------
# User prompts (per task × format)
# ---------------------------------------------------------------------------


_APPLY_USER = (
    "You are given the old version of a code file and a diff describing the\n"
    "edits. Produce the new version of the file by applying the diff.\n"
    "Output ONLY the new code — no explanations, no markdown fences.\n"
    "\n"
    "--- Old code ---\n{old_code}\n--- End old code ---\n"
    "\n"
    "--- Diff ({fmt}) ---\n{diff}\n--- End diff ---\n"
    "\n"
    "Now output the new code:"
)


_ANTI_APPLY_USER = (
    "You are given the new version of a code file and a diff describing the\n"
    "edits that produced it. Reconstruct the old version by reversing the diff.\n"
    "Output ONLY the old code — no explanations, no markdown fences.\n"
    "\n"
    "--- New code ---\n{new_code}\n--- End new code ---\n"
    "\n"
    "--- Diff ({fmt}) ---\n{diff}\n--- End diff ---\n"
    "\n"
    "Now output the old code:"
)


_DIFF_GEN_USER = (
    "You are given the old and new versions of a code file. Produce a diff in\n"
    "the {fmt} format that transforms the old version into the new version.\n"
    "Output ONLY the diff — no explanations, no markdown fences.\n"
    "\n"
    "--- Old code ---\n{old_code}\n--- End old code ---\n"
    "\n"
    "--- New code ---\n{new_code}\n--- End new code ---\n"
    "\n"
    "Now output the {fmt} diff:"
)


def user_prompt(task: str, fmt: str, sample) -> str:  # type: ignore[no-untyped-def]
    """Build the task-specific user prompt for a sample.

    `sample` is a `DiffXYZSample` (duck-typed to avoid circular import).
    """
    if task == "apply":
        return _APPLY_USER.format(old_code=sample.old_code, diff=sample.diff_for(fmt), fmt=fmt)
    if task == "anti_apply":
        return _ANTI_APPLY_USER.format(new_code=sample.new_code, diff=sample.diff_for(fmt), fmt=fmt)
    if task == "diff_gen":
        return _DIFF_GEN_USER.format(old_code=sample.old_code, new_code=sample.new_code, fmt=fmt)
    raise ValueError(f"unknown task {task!r}; use {TASKS}")


# ---------------------------------------------------------------------------
# Response post-processing
# ---------------------------------------------------------------------------


def strip_markdown_fence(response: str) -> str:
    """Remove ```lang ... ``` fences many models emit despite being told not to.

    Handles both ``` and ```python variants; leaves the response intact if
    no fence is found.
    """
    text = response.strip()
    if not text.startswith("```"):
        return response
    # Drop first line (```lang) and trailing fence if present.
    lines = text.splitlines()
    if len(lines) < 2:
        return response
    body = lines[1:]
    if body and body[-1].strip().startswith("```"):
        body = body[:-1]
    return "\n".join(body)
