"""Tests for evaluation/diff_xyz/prompts.py and dataset.py (offline portions)."""
from __future__ import annotations

import pytest

from evaluation.diff_xyz.dataset import DiffXYZSample, _row_to_sample
from evaluation.diff_xyz.prompts import (
    GENERIC_SYSTEM,
    SYSTEM_PROMPT_MODES,
    TASKS,
    strip_markdown_fence,
    system_prompt,
    user_prompt,
)


@pytest.fixture
def sample() -> DiffXYZSample:
    return DiffXYZSample(
        repo="org/repo",
        commit="abc123",
        path="src/main.py",
        lang="python",
        old_code="def foo():\n    return 1",
        new_code="def foo():\n    return 2",
        udiff="@@ -1,2 +1,2 @@\n def foo():\n-    return 1\n+    return 2\n",
        udiff_h="@@...@@\n def foo():\n-    return 1\n+    return 2\n",
        udiff_l="CON def foo():\nDEL     return 1\nADD     return 2\n",
        search_replace="<<<<<<< SEARCH\n    return 1\n=======\n    return 2\n>>>>>>> REPLACE\n",
        n_added=1,
        n_removed=1,
        n_hunks=1,
        change_kind="edit",
    )


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------


def test_system_none_returns_generic():
    assert system_prompt("udiff", "none") == GENERIC_SYSTEM


def test_system_format_includes_example_udiff():
    prompt = system_prompt("udiff", "format")
    assert GENERIC_SYSTEM in prompt
    assert "@@" in prompt
    assert "+" in prompt and "-" in prompt


def test_system_format_search_replace_has_markers():
    prompt = system_prompt("search-replace", "format")
    assert "<<<<<<< SEARCH" in prompt
    assert ">>>>>>> REPLACE" in prompt


def test_system_format_udiff_l_has_tags():
    prompt = system_prompt("udiff-l", "format")
    assert "ADD" in prompt and "DEL" in prompt and "CON" in prompt


def test_system_unknown_mode_raises():
    with pytest.raises(ValueError):
        system_prompt("udiff", "bogus")


def test_system_unknown_format_raises_in_format_mode():
    with pytest.raises(ValueError):
        system_prompt("bogus", "format")


# ---------------------------------------------------------------------------
# User prompts
# ---------------------------------------------------------------------------


def test_apply_prompt_contains_old_code_and_diff(sample: DiffXYZSample):
    prompt = user_prompt("apply", "udiff", sample)
    assert sample.old_code in prompt
    assert sample.udiff in prompt
    assert "new_code" not in prompt.lower()  # apply task shouldn't leak the target


def test_anti_apply_prompt_contains_new_code_and_diff(sample: DiffXYZSample):
    prompt = user_prompt("anti_apply", "udiff", sample)
    assert sample.new_code in prompt
    assert sample.udiff in prompt


def test_diff_gen_prompt_contains_both_codes(sample: DiffXYZSample):
    prompt = user_prompt("diff_gen", "udiff", sample)
    assert sample.old_code in prompt
    assert sample.new_code in prompt
    # Diff-Gen should NOT include any diff in the prompt (that's what we're generating)
    assert sample.udiff not in prompt


def test_user_prompt_respects_format_arg(sample: DiffXYZSample):
    p = user_prompt("apply", "search-replace", sample)
    assert sample.search_replace in p
    assert sample.udiff not in p


def test_user_prompt_unknown_task_raises(sample: DiffXYZSample):
    with pytest.raises(ValueError):
        user_prompt("bogus", "udiff", sample)


def test_all_tasks_and_formats_enumerated():
    assert TASKS == ("apply", "anti_apply", "diff_gen")
    assert set(SYSTEM_PROMPT_MODES) == {"none", "format"}


# ---------------------------------------------------------------------------
# Markdown fence stripping
# ---------------------------------------------------------------------------


def test_strip_markdown_fence_with_language():
    text = "```python\ndef foo():\n    return 1\n```"
    assert strip_markdown_fence(text) == "def foo():\n    return 1"


def test_strip_markdown_fence_bare():
    text = "```\nraw body\n```"
    assert strip_markdown_fence(text) == "raw body"


def test_strip_markdown_fence_no_fence_returns_input():
    text = "def foo(): return 1"
    assert strip_markdown_fence(text) == text


def test_strip_markdown_fence_only_opening():
    # Opening fence only (unterminated) — still drops the opener line.
    text = "```python\nbody"
    assert strip_markdown_fence(text) == "body"


# ---------------------------------------------------------------------------
# Row → sample coercion
# ---------------------------------------------------------------------------


def test_row_to_sample_handles_hyphenated_keys():
    # HF column names use hyphens: 'udiff-h', 'search-replace'.
    row = {
        "repo": "o/r", "commit": "c", "path": "p.py", "lang": "python",
        "old_code": "a", "new_code": "b",
        "udiff": "u", "udiff-h": "uh", "udiff-l": "ul", "search-replace": "sr",
        "n_added": 1, "n_removed": 2, "n_hunks": 3, "change_kind": "edit",
    }
    s = _row_to_sample(row)
    assert s.udiff_h == "uh"
    assert s.udiff_l == "ul"
    assert s.search_replace == "sr"


def test_row_to_sample_underscored_fallback():
    # Some loaders normalize hyphens to underscores; accept both.
    row = {"udiff_h": "uh", "udiff_l": "ul", "search_replace": "sr"}
    s = _row_to_sample(row)
    assert s.udiff_h == "uh"
    assert s.udiff_l == "ul"
    assert s.search_replace == "sr"


def test_row_to_sample_missing_fields_default_empty():
    s = _row_to_sample({})
    assert s.repo == "" and s.lang == ""
    assert s.old_code == "" and s.new_code == ""
    assert s.n_added == 0


# ---------------------------------------------------------------------------
# DiffXYZSample.diff_for
# ---------------------------------------------------------------------------


def test_diff_for_returns_correct_format(sample: DiffXYZSample):
    assert sample.diff_for("udiff") == sample.udiff
    assert sample.diff_for("udiff-h") == sample.udiff_h
    assert sample.diff_for("udiff-l") == sample.udiff_l
    assert sample.diff_for("search-replace") == sample.search_replace


def test_diff_for_unknown_raises(sample: DiffXYZSample):
    with pytest.raises(ValueError):
        sample.diff_for("bogus")
