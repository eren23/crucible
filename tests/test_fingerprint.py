"""Tests for crucible.runner.fingerprint."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from crucible.runner.fingerprint import (
    build_run_manifest,
    code_fingerprint,
    ensure_clean_commit,
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

    def test_discovers_nested_src_files(self, tmp_path):
        pkg = tmp_path / "src" / "crucible" / "core"
        pkg.mkdir(parents=True)
        (pkg / "config.py").write_text("", encoding="utf-8")
        files = _discover_files(tmp_path)
        assert "src/crucible/core/config.py" in files

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


# ---------------------------------------------------------------------------
# ensure_clean_commit
# ---------------------------------------------------------------------------

class TestEnsureCleanCommit:
    @patch("crucible.runner.fingerprint.safe_git_sha", return_value="abc123")
    @patch("crucible.runner.fingerprint.safe_git_dirty", return_value=False)
    def test_clean_repo_returns_sha(self, mock_dirty, mock_sha):
        result = ensure_clean_commit(Path("/fake"))
        assert result == "abc123"

    @patch("crucible.runner.fingerprint.safe_git_sha", return_value="abc123")
    @patch("crucible.runner.fingerprint.safe_git_dirty", return_value=True)
    def test_dirty_repo_raises_without_auto_commit(self, mock_dirty, mock_sha):
        with pytest.raises(RuntimeError, match="uncommitted changes"):
            ensure_clean_commit(Path("/fake"), auto_commit=False)

    @patch("crucible.runner.fingerprint.safe_git_sha", return_value=None)
    @patch("crucible.runner.fingerprint.safe_git_dirty", return_value=False)
    def test_no_git_repo_raises(self, mock_dirty, mock_sha):
        with pytest.raises(RuntimeError, match="Not in a git repository"):
            ensure_clean_commit(Path("/fake"))


# ---------------------------------------------------------------------------
# build_run_manifest
# ---------------------------------------------------------------------------

class TestBuildRunManifest:
    @patch("crucible.runner.fingerprint.safe_git_sha", return_value="abc123def")
    @patch("crucible.runner.fingerprint.safe_git_dirty", return_value=False)
    @patch("crucible.runner.fingerprint.safe_git_branch", return_value="main")
    def test_basic_manifest(self, mock_branch, mock_dirty, mock_sha, tmp_path):
        (tmp_path / "train.py").write_text("code", encoding="utf-8")
        manifest = build_run_manifest(tmp_path)
        assert manifest["git_sha"] == "abc123def"
        assert manifest["git_dirty"] is False
        assert manifest["git_branch"] == "main"
        assert "code_fingerprint" in manifest
        assert isinstance(manifest["tap_versions"], dict)

    @patch("crucible.runner.fingerprint.safe_git_sha", return_value="abc123def")
    @patch("crucible.runner.fingerprint.safe_git_dirty", return_value=False)
    @patch("crucible.runner.fingerprint.safe_git_branch", return_value="main")
    def test_manifest_includes_data_checksum(self, mock_branch, mock_dirty, mock_sha, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "manifest.json").write_text('{"shards": 10}', encoding="utf-8")
        manifest = build_run_manifest(tmp_path)
        assert manifest["data_manifest_checksum"] is not None
        assert len(manifest["data_manifest_checksum"]) == 16

    @patch("crucible.runner.fingerprint.safe_git_sha", return_value="abc123def")
    @patch("crucible.runner.fingerprint.safe_git_dirty", return_value=False)
    @patch("crucible.runner.fingerprint.safe_git_branch", return_value="main")
    def test_manifest_no_data_dir(self, mock_branch, mock_dirty, mock_sha, tmp_path):
        manifest = build_run_manifest(tmp_path)
        assert manifest["data_manifest_checksum"] is None
