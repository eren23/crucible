"""Tests for crucible.core.io."""
import json
import tempfile
from pathlib import Path

from crucible.core.io import atomic_write_json, read_jsonl, append_jsonl, write_jsonl, collect_public_attrs


def test_atomic_write_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        atomic_write_json(path, {"key": "value", "num": 42})
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["key"] == "value"
        assert data["num"] == 42


def test_read_jsonl_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "empty.jsonl"
        assert read_jsonl(path) == []


def test_write_and_read_jsonl():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        records = [{"a": 1}, {"b": 2}, {"c": 3}]
        write_jsonl(path, records)
        loaded = read_jsonl(path)
        assert len(loaded) == 3
        assert loaded[0]["a"] == 1


def test_append_jsonl():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        append_jsonl(path, {"first": True})
        append_jsonl(path, {"second": True})
        loaded = read_jsonl(path)
        assert len(loaded) == 2


def test_collect_public_attrs():
    class Obj:
        x = 1
        y = "hello"
        _private = "hidden"
        def method(self): pass

    attrs = collect_public_attrs(Obj())
    assert attrs["x"] == 1
    assert attrs["y"] == "hello"
    assert "_private" not in attrs
    assert "method" not in attrs
