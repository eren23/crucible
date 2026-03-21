"""Tests for crucible.runner.fingerprint."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from crucible.runner.fingerprint import (
    code_fingerprint,
    safe_git_sha,
    safe_git_dirty,
    safe_git_branch,
    _discover_files,
)


# ---------------------------------------------------------------------------
# code_fingerprint
# ---------------------------------------------------------------------------

class TestCodeFingerprint:
    def test_with_explicit_files(self, tmp_path):
        (tmp_path / "a.py").write_text("print('a')\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("print('b')\n", encoding="utf-8")
        result = code_fingerprint(tmp_path, extra_files=["a.py", "b.py"])
        assert "fingerprint" in result
        assert "files" in result
        assert "a.py" in result["files"]
        assert "b.py" in result["files"]
        assert len(result["fingerprint"]) == 16

    def test_fingerprint_changes_with_content(self, tmp_path):
        (tmp_path / "a.py").write_text("version = 1\n", encoding="utf-8")
        fp1 = code_fingerprint(tmp_path, extra_files=["a.py"])

        (tmp_path / "a.py").write_text("version = 2\n", encoding="utf-8")
        fp2 = code_fingerprint(tmp_path, extra_files=["a.py"])

        assert fp1["fingerprint"] != fp2["fingerprint"]
        assert fp1["files"]["a.py"] != fp2["files"]["a.py"]

    def test_fingerprint_stable_for_same_content(self, tmp_path):
        (tmp_path / "a.py").write_text("stable\n", encoding="utf-8")
        fp1 = code_fingerprint(tmp_path, extra_files=["a.py"])
        fp2 = code_fingerprint(tmp_path, extra_files=["a.py"])
        assert fp1 == fp2

    def test_missing_files_excluded(self, tmp_path):
        (tmp_path / "exists.py").write_text("ok\n", encoding="utf-8")
        result = code_fingerprint(tmp_path, extra_files=["exists.py", "missing.py"])
        assert "exists.py" in result["files"]
        assert "missing.py" not in result["files"]

    def test_empty_file_list(self, tmp_path):
        result = code_fingerprint(tmp_path, extra_files=[])
        assert result["fingerprint"] is not None
        assert result["files"] == {}

    def test_auto_discover(self, tmp_path):
        (tmp_path / "train.py").write_text("training code\n", encoding="utf-8")
        (tmp_path / "train_mlx.py").write_text("mlx code\n", encoding="utf-8")
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "model.py").write_text("model\n", encoding="utf-8")

        result = code_fingerprint(tmp_path)
        assert "train.py" in result["files"]
        assert "train_mlx.py" in result["files"]
        assert "src/model.py" in result["files"]


# ---------------------------------------------------------------------------
# _discover_files
# ---------------------------------------------------------------------------

class TestDiscoverFiles:
    def test_discovers_train_scripts(self, tmp_path):
        (tmp_path / "train.py").write_text("", encoding="utf-8")
        (tmp_path / "train_mlx.py").write_text("", encoding="utf-8")
        (tmp_path / "other.py").write_text("", encoding="utf-8")
        files = _discover_files(tmp_path)
        assert "train.py" in files
        assert "train_mlx.py" in files
        assert "other.py" not in files

    def test_discovers_src_files(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "module.py").write_text("", encoding="utf-8")
        files = _discover_files(tmp_path)
        assert "src/module.py" in files

    def test_returns_sorted(self, tmp_path):
        (tmp_path / "train_z.py").write_text("", encoding="utf-8")
        (tmp_path / "train_a.py").write_text("", encoding="utf-8")
        files = _discover_files(tmp_path)
        assert files == sorted(files)

    def test_empty_project(self, tmp_path):
        files = _discover_files(tmp_path)
        assert files == []


# ---------------------------------------------------------------------------
# safe_git_sha
# ---------------------------------------------------------------------------

class TestSafeGitSha:
    @patch("crucible.runner.fingerprint.subprocess.run")
    def test_returns_sha(self, mock_run):
        mock_run.return_value = MagicMock(stdout="abc123def456\n", returncode=0)
        result = safe_git_sha(Path("/fake"))
        assert result == "abc123def456"

    @patch("crucible.runner.fingerprint.subprocess.run")
    def test_returns_none_on_empty(self, mock_run):
        mock_run.return_value = MagicMock(stdout="", returncode=128)
        result = safe_git_sha(Path("/fake"))
        assert result is None

    @patch("crucible.runner.fingerprint.subprocess.run", side_effect=FileNotFoundError)
    def test_returns_none_on_exception(self, mock_run):
        result = safe_git_sha(Path("/fake"))
        assert result is None


# ---------------------------------------------------------------------------
# safe_git_dirty
# ---------------------------------------------------------------------------

class TestSafeGitDirty:
    @patch("crucible.runner.fingerprint.subprocess.run")
    def test_clean_repo(self, mock_run):
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        result = safe_git_dirty(Path("/fake"))
        assert result is False

    @patch("crucible.runner.fingerprint.subprocess.run")
    def test_dirty_repo(self, mock_run):
        mock_run.return_value = MagicMock(stdout=" M src/file.py\n", returncode=0)
        result = safe_git_dirty(Path("/fake"))
        assert result is True

    @patch("crucible.runner.fingerprint.subprocess.run")
    def test_returns_none_on_error(self, mock_run):
        mock_run.return_value = MagicMock(stdout="", returncode=128)
        result = safe_git_dirty(Path("/fake"))
        assert result is None

    @patch("crucible.runner.fingerprint.subprocess.run", side_effect=FileNotFoundError)
    def test_returns_none_on_exception(self, mock_run):
        result = safe_git_dirty(Path("/fake"))
        assert result is None


# ---------------------------------------------------------------------------
# safe_git_branch
# ---------------------------------------------------------------------------

class TestSafeGitBranch:
    @patch("crucible.runner.fingerprint.subprocess.run")
    def test_returns_branch(self, mock_run):
        mock_run.return_value = MagicMock(stdout="main\n", returncode=0)
        result = safe_git_branch(Path("/fake"))
        assert result == "main"

    @patch("crucible.runner.fingerprint.subprocess.run")
    def test_returns_none_on_empty(self, mock_run):
        mock_run.return_value = MagicMock(stdout="", returncode=128)
        result = safe_git_branch(Path("/fake"))
        assert result is None

    @patch("crucible.runner.fingerprint.subprocess.run", side_effect=FileNotFoundError)
    def test_returns_none_on_exception(self, mock_run):
        result = safe_git_branch(Path("/fake"))
        assert result is None
