"""Tests for evaluation/diff_xyz/metrics.py and formats.py."""
from __future__ import annotations

import pytest

from evaluation.diff_xyz.formats import (
    ApplyError,
    ParseError,
    apply_diff,
    diff_added_lines,
    diff_deleted_lines,
    parse_diff,
)
from evaluation.diff_xyz.metrics import (
    compute_apply_metrics,
    compute_diff_gen_metrics,
    f1_added,
    f1_deleted,
    f1_score,
    strip_whitespace_lines,
    stripped_em,
    stripped_iou,
)


# ---------------------------------------------------------------------------
# Low-level text helpers
# ---------------------------------------------------------------------------


def test_strip_whitespace_lines_drops_empty_and_whitespace_only():
    s = "a\n\n  \n\tb\n"
    assert strip_whitespace_lines(s) == "a\n\tb"


def test_strip_whitespace_lines_preserves_lines_with_content():
    assert strip_whitespace_lines("a\nb") == "a\nb"


# ---------------------------------------------------------------------------
# Stripped EM
# ---------------------------------------------------------------------------


def test_em_identical_is_one():
    assert stripped_em("a\nb", "a\nb") == 1.0


def test_em_different_is_zero():
    assert stripped_em("a", "b") == 0.0


def test_em_invariant_to_blank_lines():
    assert stripped_em("a\n\n\nb", "a\nb") == 1.0


def test_em_sensitive_to_content():
    assert stripped_em("a\nx", "a\nb") == 0.0


# ---------------------------------------------------------------------------
# Stripped IoU
# ---------------------------------------------------------------------------


def test_iou_identical_is_one():
    assert stripped_iou("a\nb", "a\nb") == 1.0


def test_iou_disjoint_is_zero():
    assert stripped_iou("a\nb", "c\nd") == 0.0


def test_iou_half_overlap():
    # A = {a, b}, B = {b, c}. Inter=1, union=3 → 1/3.
    assert stripped_iou("a\nb", "b\nc") == pytest.approx(1 / 3)


def test_iou_both_empty_is_one():
    assert stripped_iou("", "") == 1.0


def test_iou_unique_dedup():
    # Duplicated lines collapse in the set → identical IoU to the dedup.
    assert stripped_iou("a\na\nb", "a\nb") == 1.0


# ---------------------------------------------------------------------------
# F1 on line sets
# ---------------------------------------------------------------------------


def test_f1_both_empty_is_one():
    assert f1_score([], []) == 1.0


def test_f1_one_empty_is_zero():
    assert f1_score(["a"], []) == 0.0
    assert f1_score([], ["a"]) == 0.0


def test_f1_perfect():
    assert f1_score(["a", "b"], ["a", "b"]) == 1.0


def test_f1_half():
    # P = {a,b}, R = {b,c}. tp=1, precision=0.5, recall=0.5, F1=0.5.
    assert f1_score(["a", "b"], ["b", "c"]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Udiff parse + apply
# ---------------------------------------------------------------------------


UDIFF_SIMPLE = """\
@@ -1,3 +1,3 @@
 line_a
-line_b
+line_B
 line_c
"""


def test_udiff_parses():
    parsed = parse_diff(UDIFF_SIMPLE, "udiff")
    assert len(parsed.hunks) == 1
    kinds = [op.kind for op in parsed.hunks[0].ops]
    assert kinds == ["context", "delete", "add", "context"]


def test_udiff_applies():
    parsed = parse_diff(UDIFF_SIMPLE, "udiff")
    old = "line_a\nline_b\nline_c"
    new = apply_diff(old, parsed, "udiff")
    assert new == "line_a\nline_B\nline_c"


def test_udiff_parse_error_on_no_hunks():
    with pytest.raises(ParseError):
        parse_diff("just plain text", "udiff")


def test_udiff_apply_fails_on_wrong_anchor():
    # old_code doesn't contain line_b → apply fails
    parsed = parse_diff(UDIFF_SIMPLE, "udiff")
    with pytest.raises(ApplyError):
        apply_diff("x\ny\nz", parsed, "udiff")


def test_udiff_h_parses_relaxed_header():
    text = "@@ ...just anything... @@\n a\n-b\n+B\n"
    parsed = parse_diff(text, "udiff-h")
    assert len(parsed.hunks) == 1


# ---------------------------------------------------------------------------
# Udiff-l (verbose markers)
# ---------------------------------------------------------------------------


UDIFF_L_SIMPLE = """\
CON line_a
DEL line_b
ADD line_B
CON line_c
"""


def test_udiff_l_parses_and_applies():
    parsed = parse_diff(UDIFF_L_SIMPLE, "udiff-l")
    assert [op.kind for op in parsed.hunks[0].ops] == ["context", "delete", "add", "context"]
    assert apply_diff("line_a\nline_b\nline_c", parsed, "udiff-l") == "line_a\nline_B\nline_c"


# ---------------------------------------------------------------------------
# Search-replace
# ---------------------------------------------------------------------------


SR_SIMPLE = """\
<<<<<<< SEARCH
old_fn()
=======
new_fn()
>>>>>>> REPLACE
"""


def test_search_replace_parses():
    parsed = parse_diff(SR_SIMPLE, "search-replace")
    assert len(parsed.hunks) == 1


def test_search_replace_applies():
    parsed = parse_diff(SR_SIMPLE, "search-replace")
    old = "def main():\n    old_fn()\n    return 0"
    new = apply_diff(old, parsed, "search-replace")
    assert new == "def main():\n    new_fn()\n    return 0"


def test_search_replace_multiple_blocks():
    multi = """\
<<<<<<< SEARCH
a
=======
A
>>>>>>> REPLACE
<<<<<<< SEARCH
b
=======
B
>>>>>>> REPLACE
"""
    parsed = parse_diff(multi, "search-replace")
    assert len(parsed.hunks) == 2
    assert apply_diff("a\nb\nc", parsed, "search-replace") == "A\nB\nc"


def test_search_replace_missing_marker_parse_error():
    bad = "<<<<<<< SEARCH\nfoo\n=======\nbar\n"
    with pytest.raises(ParseError):
        parse_diff(bad, "search-replace")


def test_search_replace_apply_fails_when_search_absent():
    parsed = parse_diff(SR_SIMPLE, "search-replace")
    with pytest.raises(ApplyError):
        apply_diff("no match here", parsed, "search-replace")


# ---------------------------------------------------------------------------
# diff_added_lines / diff_deleted_lines
# ---------------------------------------------------------------------------


def test_extract_added_lines_from_udiff():
    assert diff_added_lines(UDIFF_SIMPLE, "udiff") == {"line_B"}


def test_extract_deleted_lines_from_udiff():
    assert diff_deleted_lines(UDIFF_SIMPLE, "udiff") == {"line_b"}


def test_extract_from_search_replace():
    added = diff_added_lines(SR_SIMPLE, "search-replace")
    deleted = diff_deleted_lines(SR_SIMPLE, "search-replace")
    assert added == {"new_fn()"}
    assert deleted == {"old_fn()"}


def test_extract_silent_on_parse_error():
    assert diff_added_lines("malformed", "udiff") == set()
    assert diff_deleted_lines("malformed", "udiff") == set()


# ---------------------------------------------------------------------------
# F1+ / F1-
# ---------------------------------------------------------------------------


def test_f1_plus_perfect_match():
    assert f1_added(UDIFF_SIMPLE, UDIFF_SIMPLE, "udiff") == 1.0


def test_f1_minus_perfect_match():
    assert f1_deleted(UDIFF_SIMPLE, UDIFF_SIMPLE, "udiff") == 1.0


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


def test_compute_apply_metrics():
    m = compute_apply_metrics("a\nb", "a\nb")
    assert m.em == 1.0
    assert m.iou == 1.0


def test_compute_diff_gen_metrics_happy_path():
    old = "line_a\nline_b\nline_c"
    new = "line_a\nline_B\nline_c"
    m = compute_diff_gen_metrics(
        predicted_diff=UDIFF_SIMPLE,
        reference_diff=UDIFF_SIMPLE,
        old_code=old,
        new_code=new,
        fmt="udiff",
    )
    assert m.em == 1.0
    assert m.iou == 1.0
    assert m.parsing_rate == 1.0
    assert m.applying_rate == 1.0
    assert m.f1_plus == 1.0
    assert m.f1_minus == 1.0


def test_compute_diff_gen_metrics_parse_failure():
    m = compute_diff_gen_metrics(
        predicted_diff="gibberish",
        reference_diff=UDIFF_SIMPLE,
        old_code="a",
        new_code="b",
        fmt="udiff",
    )
    assert m.parsing_rate == 0.0
    assert m.applying_rate == 0.0
    assert m.em == 0.0


def test_compute_diff_gen_metrics_apply_failure():
    # parses fine, but 'line_b' is not in old_code → apply fails
    m = compute_diff_gen_metrics(
        predicted_diff=UDIFF_SIMPLE,
        reference_diff=UDIFF_SIMPLE,
        old_code="x\ny\nz",
        new_code="whatever",
        fmt="udiff",
    )
    assert m.parsing_rate == 1.0
    assert m.applying_rate == 0.0
    assert m.em == 0.0


# ---------------------------------------------------------------------------
# Unknown format handling
# ---------------------------------------------------------------------------


def test_unknown_format_raises():
    with pytest.raises(ParseError):
        parse_diff("", "bogus")
