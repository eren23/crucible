"""Tests for crucible.core.tap_scaffold — `crucible tap init`.

Two contracts:

1. The scaffolded directory contains all expected files.
2. Running `crucible tap lint` against the scaffolded directory yields 0
   issues — i.e. the scaffold is "born clean" and any plugin added on top
   only has to maintain that bar, not fix scaffold-shipped issues.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from crucible.core.errors import TapError
from crucible.core.tap_lint import lint_tap_directory
from crucible.core.tap_scaffold import scaffold_tap


class TestScaffoldOutput:
    def test_writes_all_expected_files(self, tmp_path: Path):
        target = tmp_path / "demo"
        result = scaffold_tap(target, name="demo", init_git=False)
        assert result["files_written"] >= 7

        # Top-level files
        for f in ("README.md", "LICENSE", "tap.yaml", ".gitignore"):
            assert (target / f).is_file(), f"missing {f}"

        # CI workflow
        assert (target / ".github" / "workflows" / "lint.yaml").is_file()

        # Example plugin bundle
        plug = target / "optimizers" / "example_optimizer"
        for f in ("plugin.yaml", "example_optimizer.py", "README.md"):
            assert (plug / f).is_file(), f"missing {plug / f}"

    def test_returns_next_steps(self, tmp_path: Path):
        result = scaffold_tap(tmp_path / "demo", name="demo", init_git=False)
        steps = result["next_steps"]
        assert isinstance(steps, list) and steps
        assert any("crucible tap lint" in s for s in steps)

    def test_refuses_to_clobber_non_empty_dir(self, tmp_path: Path):
        target = tmp_path / "demo"
        target.mkdir()
        (target / "preexisting.txt").write_text("hello", encoding="utf-8")
        with pytest.raises(TapError, match="not empty"):
            scaffold_tap(target, name="demo", init_git=False)

    def test_refuses_to_scaffold_into_file(self, tmp_path: Path):
        target = tmp_path / "is_a_file"
        target.write_text("x", encoding="utf-8")
        with pytest.raises(TapError, match="not a directory"):
            scaffold_tap(target, name="demo", init_git=False)

    def test_empty_existing_dir_is_ok(self, tmp_path: Path):
        target = tmp_path / "empty"
        target.mkdir()
        result = scaffold_tap(target, name="demo", init_git=False)
        assert result["files_written"] >= 7

    def test_author_propagated_to_files(self, tmp_path: Path):
        target = tmp_path / "demo"
        scaffold_tap(target, name="demo", author="alice@example", init_git=False)
        assert "alice@example" in (target / "LICENSE").read_text()
        assert "alice@example" in (target / "tap.yaml").read_text()

    def test_apache_license_template(self, tmp_path: Path):
        target = tmp_path / "demo"
        scaffold_tap(target, name="demo", license_id="Apache-2.0", init_git=False)
        assert "Apache License" in (target / "LICENSE").read_text()

    def test_unknown_license_falls_back_to_mit(self, tmp_path: Path):
        target = tmp_path / "demo"
        scaffold_tap(target, name="demo", license_id="UnknownLicense", init_git=False)
        # Falls back to MIT body silently.
        assert "MIT License" in (target / "LICENSE").read_text()


class TestScaffoldedTapPassesLint:
    """The headline guarantee: scaffold → lint → 0 issues."""

    def test_default_scaffold_is_clean(self, tmp_path: Path):
        target = tmp_path / "demo"
        scaffold_tap(target, name="demo", author="ci", init_git=False)
        issues = lint_tap_directory(target)
        if issues:
            details = "\n".join(
                f"  [{i.severity}] {i.code} {i.path}: {i.message}"
                for i in issues
            )
            raise AssertionError(
                f"scaffolded tap had {len(issues)} lint issues:\n{details}"
            )

    def test_apache_scaffold_is_clean(self, tmp_path: Path):
        target = tmp_path / "demo"
        scaffold_tap(target, name="demo", author="ci",
                    license_id="Apache-2.0", init_git=False)
        assert lint_tap_directory(target) == []

    def test_custom_name_passes_lint(self, tmp_path: Path):
        target = tmp_path / "my_tap_v2"
        scaffold_tap(target, name="my_tap_v2", author="ci", init_git=False)
        assert lint_tap_directory(target) == []
