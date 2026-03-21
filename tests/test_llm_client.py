"""Tests for crucible.researcher.llm_client."""
from __future__ import annotations

import pytest

from crucible.researcher.llm_client import parse_json_from_text


# ---------------------------------------------------------------------------
# parse_json_from_text
# ---------------------------------------------------------------------------

class TestParseJsonFromText:
    def test_direct_json(self):
        text = '{"key": "value", "num": 42}'
        result = parse_json_from_text(text)
        assert result == {"key": "value", "num": 42}

    def test_json_with_whitespace(self):
        text = '  \n  {"key": "value"}  \n  '
        result = parse_json_from_text(text)
        assert result == {"key": "value"}

    def test_json_code_block(self):
        text = 'Here is the result:\n```json\n{"answer": true}\n```\nDone.'
        result = parse_json_from_text(text)
        assert result == {"answer": True}

    def test_code_block_without_json_tag(self):
        text = 'Result:\n```\n{"answer": false}\n```'
        result = parse_json_from_text(text)
        assert result == {"answer": False}

    def test_embedded_json_in_text(self):
        text = 'The analysis shows {"metric": 1.5, "status": "ok"} which is good.'
        result = parse_json_from_text(text)
        assert result == {"metric": 1.5, "status": "ok"}

    def test_nested_braces(self):
        text = 'Output: {"outer": {"inner": "value"}, "list": [1, 2]}'
        result = parse_json_from_text(text)
        assert result is not None
        assert result["outer"]["inner"] == "value"
        assert result["list"] == [1, 2]

    def test_invalid_json_returns_none(self):
        text = "This is not JSON at all, just plain text."
        result = parse_json_from_text(text)
        assert result is None

    def test_empty_string_returns_none(self):
        result = parse_json_from_text("")
        assert result is None

    def test_array_not_dict_returns_none(self):
        text = '[1, 2, 3]'
        result = parse_json_from_text(text)
        # The function only returns dicts, not arrays
        assert result is None

    def test_complex_hypothesis_response(self):
        text = """Based on my analysis, I recommend the following:

```json
{
    "hypothesis": "Reducing learning rate improves convergence",
    "config": {"LR": "0.0001", "WARMUP_STEPS": "20"},
    "confidence": 0.8,
    "family": "lr_schedule"
}
```

This should improve the model's training stability."""
        result = parse_json_from_text(text)
        assert result is not None
        assert result["hypothesis"] == "Reducing learning rate improves convergence"
        assert result["config"]["LR"] == "0.0001"
        assert result["confidence"] == 0.8

    def test_multiple_json_objects_returns_first(self):
        text = 'First: {"a": 1} Second: {"b": 2}'
        result = parse_json_from_text(text)
        assert result is not None
        assert result == {"a": 1}

    def test_json_with_unicode(self):
        text = '{"name": "test model", "description": "improved architecture"}'
        result = parse_json_from_text(text)
        assert result is not None
        assert result["name"] == "test model"
