"""Tests for crucible.core.env."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from crucible.core.env import load_env_files, _parse_env_line


# ---------------------------------------------------------------------------
# _parse_env_line (internal helper)
# ---------------------------------------------------------------------------

class TestParseEnvLine:
    def test_simple_key_value(self):
        assert _parse_env_line("FOO=bar") == ("FOO", "bar")

    def test_strips_whitespace(self):
        assert _parse_env_line("  FOO=bar  ") == ("FOO", "bar")

    def test_comment_returns_none(self):
        assert _parse_env_line("# comment") is None

    def test_blank_line_returns_none(self):
        assert _parse_env_line("") is None
        assert _parse_env_line("   ") is None

    def test_export_prefix(self):
        assert _parse_env_line("export API_KEY=secret") == ("API_KEY", "secret")

    def test_double_quoted_value(self):
        assert _parse_env_line('MY_VAR="hello world"') == ("MY_VAR", "hello world")

    def test_single_quoted_value(self):
        assert _parse_env_line("MY_VAR='hello world'") == ("MY_VAR", "hello world")

    def test_no_equals_returns_none(self):
        assert _parse_env_line("JUST_KEY") is None

    def test_empty_key_returns_none(self):
        assert _parse_env_line("=value") is None

    def test_value_with_equals(self):
        result = _parse_env_line("URL=http://host:8080?key=val")
        assert result == ("URL", "http://host:8080?key=val")

    def test_export_with_quotes(self):
        result = _parse_env_line('export SECRET="my secret"')
        assert result == ("SECRET", "my secret")


# ---------------------------------------------------------------------------
# load_env_files
# ---------------------------------------------------------------------------

class TestLoadEnvFiles:
    def test_loads_env_file(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_VAR_LOAD=hello\n", encoding="utf-8")
        monkeypatch.delenv("TEST_VAR_LOAD", raising=False)

        loaded = load_env_files(tmp_path)
        assert len(loaded) == 1
        assert os.environ["TEST_VAR_LOAD"] == "hello"

        # Cleanup
        monkeypatch.delenv("TEST_VAR_LOAD", raising=False)

    def test_does_not_override_existing_by_default(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("EXISTING_VAR=new_value\n", encoding="utf-8")
        monkeypatch.setenv("EXISTING_VAR", "old_value")

        load_env_files(tmp_path)
        assert os.environ["EXISTING_VAR"] == "old_value"

    def test_override_true_replaces_existing(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("OVERRIDE_VAR=new\n", encoding="utf-8")
        monkeypatch.setenv("OVERRIDE_VAR", "old")

        load_env_files(tmp_path, override=True)
        assert os.environ["OVERRIDE_VAR"] == "new"

        monkeypatch.delenv("OVERRIDE_VAR", raising=False)

    def test_skips_comments_and_blank_lines(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "# This is a comment\n\nCOMMENT_VAR=works\n# another comment\n",
            encoding="utf-8",
        )
        monkeypatch.delenv("COMMENT_VAR", raising=False)

        load_env_files(tmp_path)
        assert os.environ["COMMENT_VAR"] == "works"
        monkeypatch.delenv("COMMENT_VAR", raising=False)

    def test_handles_export_prefix(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("export EXPORT_VAR=exported\n", encoding="utf-8")
        monkeypatch.delenv("EXPORT_VAR", raising=False)

        load_env_files(tmp_path)
        assert os.environ["EXPORT_VAR"] == "exported"
        monkeypatch.delenv("EXPORT_VAR", raising=False)

    def test_handles_quoted_values(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text(
            'DQUOTE_VAR="double quoted"\nSQUOTE_VAR=\'single quoted\'\n',
            encoding="utf-8",
        )
        monkeypatch.delenv("DQUOTE_VAR", raising=False)
        monkeypatch.delenv("SQUOTE_VAR", raising=False)

        load_env_files(tmp_path)
        assert os.environ["DQUOTE_VAR"] == "double quoted"
        assert os.environ["SQUOTE_VAR"] == "single quoted"

        monkeypatch.delenv("DQUOTE_VAR", raising=False)
        monkeypatch.delenv("SQUOTE_VAR", raising=False)

    def test_missing_files_returns_empty(self, tmp_path):
        loaded = load_env_files(tmp_path)
        assert loaded == []

    def test_loads_multiple_files(self, tmp_path, monkeypatch):
        (tmp_path / ".env").write_text("MULTI_A=1\n", encoding="utf-8")
        (tmp_path / ".env.local").write_text("MULTI_B=2\n", encoding="utf-8")
        monkeypatch.delenv("MULTI_A", raising=False)
        monkeypatch.delenv("MULTI_B", raising=False)

        loaded = load_env_files(tmp_path)
        assert len(loaded) == 2
        assert os.environ["MULTI_A"] == "1"
        assert os.environ["MULTI_B"] == "2"

        monkeypatch.delenv("MULTI_A", raising=False)
        monkeypatch.delenv("MULTI_B", raising=False)

    def test_custom_filenames(self, tmp_path, monkeypatch):
        (tmp_path / "custom.env").write_text("CUSTOM_VAR=yes\n", encoding="utf-8")
        monkeypatch.delenv("CUSTOM_VAR", raising=False)

        loaded = load_env_files(tmp_path, filenames=["custom.env"])
        assert len(loaded) == 1
        assert os.environ["CUSTOM_VAR"] == "yes"

        monkeypatch.delenv("CUSTOM_VAR", raising=False)

    def test_returns_loaded_paths(self, tmp_path):
        (tmp_path / ".env").write_text("X=1\n", encoding="utf-8")
        loaded = load_env_files(tmp_path)
        assert len(loaded) == 1
        assert loaded[0] == (tmp_path / ".env").resolve()
