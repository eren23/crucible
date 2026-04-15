"""Tests for crucible.core.io."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from crucible.core.io import (
    _json_ready,
    atomic_write_json,
    atomic_write_yaml,
    read_jsonl,
    read_yaml,
    write_jsonl,
    write_yaml,
    append_jsonl,
    collect_public_attrs,
)


# ---------------------------------------------------------------------------
# _json_ready
# ---------------------------------------------------------------------------

class TestJsonReady:
    def test_path_to_string(self):
        assert _json_ready(Path("/foo/bar")) == "/foo/bar"

    def test_none_passthrough(self):
        assert _json_ready(None) is None

    def test_primitives_passthrough(self):
        assert _json_ready(42) == 42
        assert _json_ready(3.14) == 3.14
        assert _json_ready("hello") == "hello"
        assert _json_ready(True) is True

    def test_nested_dict(self):
        result = _json_ready({"a": Path("/x"), "b": {"c": None}})
        assert result == {"a": "/x", "b": {"c": None}}

    def test_nested_list(self):
        result = _json_ready([Path("/x"), [Path("/y"), 1]])
        assert result == ["/x", ["/y", 1]]

    def test_tuple_becomes_list(self):
        result = _json_ready((1, 2, Path("/z")))
        assert result == [1, 2, "/z"]

    def test_dict_keys_become_strings(self):
        result = _json_ready({1: "a", 2: "b"})
        assert result == {"1": "a", "2": "b"}

    def test_unknown_type_becomes_str(self):
        class Custom:
            def __repr__(self):
                return "CustomObj"
        result = _json_ready(Custom())
        assert result == "CustomObj"

    def test_empty_dict(self):
        assert _json_ready({}) == {}

    def test_empty_list(self):
        assert _json_ready([]) == []


# ---------------------------------------------------------------------------
# atomic_write_json
# ---------------------------------------------------------------------------

class TestAtomicWriteJson:
    def test_writes_valid_json(self, tmp_path):
        path = tmp_path / "test.json"
        atomic_write_json(path, {"key": "value", "num": 42})
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["key"] == "value"
        assert data["num"] == 42

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "test.json"
        atomic_write_json(path, {"nested": True})
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["nested"] is True

    def test_overwrites_existing_file(self, tmp_path):
        path = tmp_path / "overwrite.json"
        atomic_write_json(path, {"v": 1})
        atomic_write_json(path, {"v": 2})
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["v"] == 2

    def test_handles_path_values(self, tmp_path):
        path = tmp_path / "paths.json"
        atomic_write_json(path, {"dir": Path("/foo/bar")})
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["dir"] == "/foo/bar"

    def test_sorted_keys(self, tmp_path):
        path = tmp_path / "sorted.json"
        atomic_write_json(path, {"z": 1, "a": 2, "m": 3})
        text = path.read_text(encoding="utf-8")
        # Keys should appear in alphabetical order
        a_pos = text.index('"a"')
        m_pos = text.index('"m"')
        z_pos = text.index('"z"')
        assert a_pos < m_pos < z_pos

    def test_file_ends_with_newline(self, tmp_path):
        path = tmp_path / "newline.json"
        atomic_write_json(path, {"x": 1})
        text = path.read_text(encoding="utf-8")
        assert text.endswith("\n")


# ---------------------------------------------------------------------------
# read_jsonl
# ---------------------------------------------------------------------------

class TestReadJsonl:
    def test_missing_file_returns_empty(self, tmp_path):
        path = tmp_path / "missing.jsonl"
        assert read_jsonl(path) == []

    def test_empty_file_returns_empty(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        assert read_jsonl(path) == []

    def test_reads_records(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text('{"a":1}\n{"b":2}\n', encoding="utf-8")
        records = read_jsonl(path)
        assert len(records) == 2
        assert records[0]["a"] == 1
        assert records[1]["b"] == 2

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "blanks.jsonl"
        path.write_text('{"a":1}\n\n\n{"b":2}\n', encoding="utf-8")
        records = read_jsonl(path)
        assert len(records) == 2

    def test_skips_invalid_json_lines(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"ok":true}\nnot json\n{"also_ok":true}\n', encoding="utf-8")
        records = read_jsonl(path)
        assert len(records) == 2


# ---------------------------------------------------------------------------
# write_jsonl
# ---------------------------------------------------------------------------

class TestWriteJsonl:
    def test_write_creates_file(self, tmp_path):
        path = tmp_path / "new.jsonl"
        write_jsonl(path, [{"x": 1}, {"x": 2}])
        assert path.exists()
        records = read_jsonl(path)
        assert len(records) == 2

    def test_write_overwrites(self, tmp_path):
        path = tmp_path / "over.jsonl"
        write_jsonl(path, [{"v": 1}])
        write_jsonl(path, [{"v": 2}, {"v": 3}])
        records = read_jsonl(path)
        assert len(records) == 2
        assert records[0]["v"] == 2

    def test_write_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "deep" / "data.jsonl"
        write_jsonl(path, [{"nested": True}])
        assert path.exists()

    def test_empty_write(self, tmp_path):
        path = tmp_path / "empty_write.jsonl"
        write_jsonl(path, [])
        assert path.exists()
        assert read_jsonl(path) == []


# ---------------------------------------------------------------------------
# append_jsonl
# ---------------------------------------------------------------------------

class TestAppendJsonl:
    def test_append_creates_file(self, tmp_path):
        path = tmp_path / "append.jsonl"
        append_jsonl(path, {"first": True})
        records = read_jsonl(path)
        assert len(records) == 1
        assert records[0]["first"] is True

    def test_append_adds_to_existing(self, tmp_path):
        path = tmp_path / "grow.jsonl"
        append_jsonl(path, {"a": 1})
        append_jsonl(path, {"b": 2})
        append_jsonl(path, {"c": 3})
        records = read_jsonl(path)
        assert len(records) == 3

    def test_append_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "append.jsonl"
        append_jsonl(path, {"nested": True})
        assert path.exists()


# ---------------------------------------------------------------------------
# collect_public_attrs
# ---------------------------------------------------------------------------

class TestCollectPublicAttrs:
    def test_collects_attributes(self):
        class Obj:
            x = 1
            y = "hello"
        attrs = collect_public_attrs(Obj())
        assert attrs["x"] == 1
        assert attrs["y"] == "hello"

    def test_excludes_private(self):
        class Obj:
            _private = "hidden"
            __dunder = "also_hidden"
            public = "visible"
        attrs = collect_public_attrs(Obj())
        assert "_private" not in attrs
        assert "__dunder" not in attrs
        assert "public" in attrs

    def test_excludes_methods(self):
        class Obj:
            x = 1
            def method(self): pass
        attrs = collect_public_attrs(Obj())
        assert "x" in attrs
        assert "method" not in attrs

    def test_converts_paths(self):
        class Obj:
            path = Path("/foo/bar")
        attrs = collect_public_attrs(Obj())
        assert attrs["path"] == "/foo/bar"


# ---------------------------------------------------------------------------
# read_yaml
# ---------------------------------------------------------------------------

class TestReadYaml:
    def test_missing_file_returns_none(self, tmp_path):
        assert read_yaml(tmp_path / "missing.yaml") is None

    def test_reads_dict(self, tmp_path):
        path = tmp_path / "data.yaml"
        path.write_text("key: value\nnum: 42\n", encoding="utf-8")
        data = read_yaml(path)
        assert data == {"key": "value", "num": 42}

    def test_reads_list(self, tmp_path):
        path = tmp_path / "list.yaml"
        path.write_text("- one\n- two\n- three\n", encoding="utf-8")
        data = read_yaml(path)
        assert data == ["one", "two", "three"]

    def test_empty_file_returns_none(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        assert read_yaml(path) is None

    def test_scalar_returns_none(self, tmp_path):
        path = tmp_path / "scalar.yaml"
        path.write_text("just a string\n", encoding="utf-8")
        assert read_yaml(path) is None

    def test_nested_structure(self, tmp_path):
        path = tmp_path / "nested.yaml"
        path.write_text("parent:\n  child: value\n  list:\n    - a\n    - b\n", encoding="utf-8")
        data = read_yaml(path)
        assert data == {"parent": {"child": "value", "list": ["a", "b"]}}


# ---------------------------------------------------------------------------
# write_yaml
# ---------------------------------------------------------------------------

class TestWriteYaml:
    def test_writes_dict(self, tmp_path):
        path = tmp_path / "out.yaml"
        write_yaml(path, {"key": "value", "num": 42})
        data = read_yaml(path)
        assert data["key"] == "value"
        assert data["num"] == 42

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "deep" / "out.yaml"
        write_yaml(path, {"nested": True})
        assert path.exists()
        data = read_yaml(path)
        assert data["nested"] is True

    def test_overwrites_existing(self, tmp_path):
        path = tmp_path / "overwrite.yaml"
        write_yaml(path, {"v": 1})
        write_yaml(path, {"v": 2})
        data = read_yaml(path)
        assert data["v"] == 2

    def test_sort_keys_option(self, tmp_path):
        path = tmp_path / "sorted.yaml"
        write_yaml(path, {"z": 1, "a": 2, "m": 3}, sort_keys=True)
        text = path.read_text(encoding="utf-8")
        a_pos = text.index("a:")
        m_pos = text.index("m:")
        z_pos = text.index("z:")
        assert a_pos < m_pos < z_pos

    def test_unicode_content(self, tmp_path):
        path = tmp_path / "unicode.yaml"
        write_yaml(path, {"emoji": "hello", "cjk": "test"})
        data = read_yaml(path)
        assert data["emoji"] == "hello"

    def test_writes_list(self, tmp_path):
        path = tmp_path / "list.yaml"
        write_yaml(path, [{"name": "a"}, {"name": "b"}])
        data = read_yaml(path)
        assert isinstance(data, list)
        assert len(data) == 2


# ---------------------------------------------------------------------------
# atomic_write_yaml
# ---------------------------------------------------------------------------

class TestAtomicWriteYaml:
    def test_writes_valid_yaml(self, tmp_path):
        path = tmp_path / "atomic.yaml"
        atomic_write_yaml(path, {"key": "value", "num": 42})
        data = read_yaml(path)
        assert data["key"] == "value"
        assert data["num"] == 42

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "atomic.yaml"
        atomic_write_yaml(path, {"nested": True})
        assert path.exists()

    def test_overwrites_existing(self, tmp_path):
        path = tmp_path / "overwrite.yaml"
        atomic_write_yaml(path, {"v": 1})
        atomic_write_yaml(path, {"v": 2})
        data = read_yaml(path)
        assert data["v"] == 2

    def test_no_temp_file_left_on_success(self, tmp_path):
        path = tmp_path / "clean.yaml"
        atomic_write_yaml(path, {"clean": True})
        # Only the target file should exist
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].name == "clean.yaml"

    def test_sort_keys_option(self, tmp_path):
        path = tmp_path / "sorted.yaml"
        atomic_write_yaml(path, {"z": 1, "a": 2}, sort_keys=True)
        text = path.read_text(encoding="utf-8")
        a_pos = text.index("a:")
        z_pos = text.index("z:")
        assert a_pos < z_pos

    def test_roundtrip_with_read_yaml(self, tmp_path):
        path = tmp_path / "roundtrip.yaml"
        original = {"name": "test", "items": [1, 2, 3], "nested": {"a": "b"}}
        atomic_write_yaml(path, original)
        loaded = read_yaml(path)
        assert loaded == original
