"""Tests for crucible.core.tap_lint — the 10 built-in lint checks.

For each check we construct a synthetic broken tap, run the check in
isolation, and assert the expected LintIssue(s) fire. We also assert the
default registry on a clean tap (built by tap_scaffold) produces zero
issues — the "born clean" guarantee Phase A promised.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from crucible.core.errors import PluginError
from crucible.core.tap_lint import (
    L001_MissingTopLevelReadme,
    L002_MissingLicense,
    L003_MissingTapManifest,
    L004_MissingPerPluginReadme,
    L005_CruftDirectories,
    L006_LargeFileInTap,
    L007_FolderNameMismatch,
    L008_MultiLineDescription,
    L009_VersionStringUnquoted,
    L010_PythonSyntaxError,
    L011_PerPluginManifestValid,
    LintCheckRegistry,
    LintIssue,
    format_lint_report,
    get_default_registry,
    lint_tap_directory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tap(root: Path, *, with_readme: bool = True, with_license: bool = True,
              with_tap_yaml: bool = True) -> Path:
    """Create a barebones tap skeleton."""
    root.mkdir(parents=True, exist_ok=True)
    if with_readme:
        (root / "README.md").write_text("# Test Tap\n", encoding="utf-8")
    if with_license:
        (root / "LICENSE").write_text("MIT\n", encoding="utf-8")
    if with_tap_yaml:
        (root / "tap.yaml").write_text(
            yaml.dump({
                "name": "test-tap",
                "description": "test",
                "version": "0.1.0",
                "author": "ci",
                "license": "MIT",
                "crucible_compat": ">=0.2,<0.3",
            }),
            encoding="utf-8",
        )
    return root


def _add_plugin(root: Path, plugin_type: str, name: str, *,
                with_readme: bool = True, manifest_overrides: dict | None = None,
                py_content: str | None = None) -> Path:
    """Drop a plugin folder under {root}/{type}/{name}/."""
    p = root / plugin_type / name
    p.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": name,
        "type": plugin_type,
        "version": "0.1.0",
        "description": "Brief one-liner",
        "author": "ci",
        "tags": ["test"],
        "crucible_compat": ">=0.2,<0.3",
        "dependencies": [],
    }
    if manifest_overrides:
        manifest.update(manifest_overrides)
    (p / "plugin.yaml").write_text(
        # Use explicit string quoting for the version field to keep L009
        # passing on the synthetic test fixtures.
        yaml.dump(manifest, default_flow_style=False),
        encoding="utf-8",
    )
    if with_readme:
        (p / "README.md").write_text(f"# {name}\n", encoding="utf-8")
    if py_content is not None:
        (p / f"{name}.py").write_text(py_content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


class TestL001MissingTopLevelReadme:
    def test_fires_when_readme_missing(self, tmp_path: Path):
        _make_tap(tmp_path, with_readme=False)
        issues = L001_MissingTopLevelReadme().run(tmp_path)
        assert any(i.code == "L001" and i.severity == "error" for i in issues)

    def test_passes_when_readme_present(self, tmp_path: Path):
        _make_tap(tmp_path)
        assert L001_MissingTopLevelReadme().run(tmp_path) == []


class TestL002MissingLicense:
    def test_fires_when_license_missing(self, tmp_path: Path):
        _make_tap(tmp_path, with_license=False)
        issues = L002_MissingLicense().run(tmp_path)
        assert any(i.code == "L002" and i.severity == "error" for i in issues)

    @pytest.mark.parametrize("name", ["LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING"])
    def test_passes_with_any_accepted_name(self, tmp_path: Path, name: str):
        _make_tap(tmp_path, with_license=False)
        (tmp_path / name).write_text("MIT\n", encoding="utf-8")
        assert L002_MissingLicense().run(tmp_path) == []


class TestL003MissingTapManifest:
    def test_fires_when_tap_yaml_missing(self, tmp_path: Path):
        _make_tap(tmp_path, with_tap_yaml=False)
        issues = L003_MissingTapManifest().run(tmp_path)
        assert any(i.code == "L003" and i.severity == "warning" for i in issues)

    def test_passes_when_tap_yaml_well_formed(self, tmp_path: Path):
        _make_tap(tmp_path)
        # L003 may emit warnings from the inner schema (e.g. trailing
        # optional fields), but no errors and not for the missing-file case.
        issues = L003_MissingTapManifest().run(tmp_path)
        assert not any("missing at repo root" in i.message for i in issues)


class TestL004MissingPerPluginReadme:
    def test_fires_when_plugin_readme_missing(self, tmp_path: Path):
        _make_tap(tmp_path)
        _add_plugin(tmp_path, "optimizers", "naked", with_readme=False)
        issues = L004_MissingPerPluginReadme().run(tmp_path)
        assert any("naked" in str(i.path) for i in issues)

    def test_passes_when_readme_present(self, tmp_path: Path):
        _make_tap(tmp_path)
        _add_plugin(tmp_path, "optimizers", "clothed")
        assert L004_MissingPerPluginReadme().run(tmp_path) == []


class TestL005CruftDirectories:
    @pytest.mark.parametrize("dirname", ["data", "checkpoints", "wandb", "__pycache__"])
    def test_fires_for_each_cruft_dir(self, tmp_path: Path, dirname: str):
        _make_tap(tmp_path)
        (tmp_path / dirname).mkdir()
        issues = L005_CruftDirectories().run(tmp_path)
        assert any(dirname in i.message for i in issues)

    def test_fires_for_ds_store_file(self, tmp_path: Path):
        _make_tap(tmp_path)
        (tmp_path / ".DS_Store").write_text("", encoding="utf-8")
        issues = L005_CruftDirectories().run(tmp_path)
        assert any("DS_Store" in i.message for i in issues)

    def test_passes_on_clean_tap(self, tmp_path: Path):
        _make_tap(tmp_path)
        _add_plugin(tmp_path, "optimizers", "clean")
        assert L005_CruftDirectories().run(tmp_path) == []


class TestL006LargeFileInTap:
    def test_fires_for_file_over_1mb(self, tmp_path: Path):
        _make_tap(tmp_path)
        big = tmp_path / "big.bin"
        big.write_bytes(b"x" * (2 * 1024 * 1024))  # 2 MB
        issues = L006_LargeFileInTap().run(tmp_path)
        assert any("big.bin" in i.message for i in issues)

    def test_skips_git_directory(self, tmp_path: Path):
        _make_tap(tmp_path)
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "pack.bin").write_bytes(b"x" * (2 * 1024 * 1024))
        issues = L006_LargeFileInTap().run(tmp_path)
        assert issues == []

    def test_passes_when_all_files_small(self, tmp_path: Path):
        _make_tap(tmp_path)
        _add_plugin(tmp_path, "optimizers", "small")
        assert L006_LargeFileInTap().run(tmp_path) == []


class TestL007FolderNameMismatch:
    def test_fires_when_folder_name_differs_from_manifest(self, tmp_path: Path):
        _make_tap(tmp_path)
        _add_plugin(tmp_path, "optimizers", "actual_folder",
                   manifest_overrides={"name": "different_name"})
        issues = L007_FolderNameMismatch().run(tmp_path)
        assert any("actual_folder" in i.message and "different_name" in i.message
                   for i in issues)

    def test_passes_when_names_match(self, tmp_path: Path):
        _make_tap(tmp_path)
        _add_plugin(tmp_path, "optimizers", "matched")
        assert L007_FolderNameMismatch().run(tmp_path) == []


class TestL008MultiLineDescription:
    def test_fires_on_block_scalar_description(self, tmp_path: Path):
        _make_tap(tmp_path)
        p = _add_plugin(tmp_path, "optimizers", "blocky")
        # Re-write the manifest manually with a YAML block scalar — yaml.dump
        # would collapse it.
        (p / "plugin.yaml").write_text(
            "name: blocky\n"
            "type: optimizers\n"
            'version: "0.1.0"\n'
            "description: >\n"
            "  Multi line\n"
            "  description\n"
            "author: ci\n",
            encoding="utf-8",
        )
        issues = L008_MultiLineDescription().run(tmp_path)
        assert any("blocky" in str(i.path) for i in issues)

    def test_passes_on_single_line_description(self, tmp_path: Path):
        _make_tap(tmp_path)
        _add_plugin(tmp_path, "optimizers", "single",
                   manifest_overrides={"description": "single line"})
        assert L008_MultiLineDescription().run(tmp_path) == []


class TestL009VersionStringUnquoted:
    def test_fires_when_version_unquoted(self, tmp_path: Path):
        _make_tap(tmp_path)
        p = _add_plugin(tmp_path, "optimizers", "unquoted")
        # Overwrite manifest with explicitly unquoted version.
        (p / "plugin.yaml").write_text(
            "name: unquoted\n"
            "type: optimizers\n"
            "version: 0.1.0\n"
            "description: x\n",
            encoding="utf-8",
        )
        issues = L009_VersionStringUnquoted().run(tmp_path)
        assert any("unquoted" in str(i.path) for i in issues)

    def test_passes_when_version_quoted(self, tmp_path: Path):
        _make_tap(tmp_path)
        p = _add_plugin(tmp_path, "optimizers", "quoted")
        (p / "plugin.yaml").write_text(
            "name: quoted\n"
            "type: optimizers\n"
            'version: "0.1.0"\n'
            "description: x\n",
            encoding="utf-8",
        )
        assert L009_VersionStringUnquoted().run(tmp_path) == []


class TestL010PythonSyntaxError:
    def test_fires_on_syntax_error(self, tmp_path: Path):
        _make_tap(tmp_path)
        _add_plugin(tmp_path, "optimizers", "broken",
                   py_content="def x(:\n    pass\n")
        issues = L010_PythonSyntaxError().run(tmp_path)
        assert any("broken" in str(i.path) for i in issues)

    def test_passes_on_clean_python(self, tmp_path: Path):
        _make_tap(tmp_path)
        _add_plugin(tmp_path, "optimizers", "clean",
                   py_content="def build():\n    return None\n")
        assert L010_PythonSyntaxError().run(tmp_path) == []


class TestL011PerPluginManifestValid:
    def test_fires_when_manifest_missing(self, tmp_path: Path):
        _make_tap(tmp_path)
        # Create a plugin folder without plugin.yaml.
        d = tmp_path / "optimizers" / "naked_dir"
        d.mkdir(parents=True)
        (d / "naked_dir.py").write_text("", encoding="utf-8")
        issues = L011_PerPluginManifestValid().run(tmp_path)
        assert any("missing plugin.yaml" in i.message.lower() for i in issues)

    def test_surfaces_schema_errors(self, tmp_path: Path):
        _make_tap(tmp_path)
        _add_plugin(tmp_path, "optimizers", "noname",
                   manifest_overrides={"name": ""})  # missing name → error
        issues = L011_PerPluginManifestValid().run(tmp_path)
        assert any(i.severity == "error" for i in issues)


# ---------------------------------------------------------------------------
# Top-level integration
# ---------------------------------------------------------------------------


class TestLintTapDirectory:
    def test_raises_on_missing_root(self, tmp_path: Path):
        with pytest.raises(PluginError):
            lint_tap_directory(tmp_path / "does-not-exist")

    def test_raises_on_non_directory(self, tmp_path: Path):
        f = tmp_path / "file"
        f.write_text("", encoding="utf-8")
        with pytest.raises(PluginError):
            lint_tap_directory(f)

    def test_clean_tap_returns_no_issues(self, tmp_path: Path):
        """Scaffolded tap should pass lint with 0 issues (the 'born clean'
        guarantee). Use the real scaffolder, not synthetic _make_tap."""
        from crucible.core.tap_scaffold import scaffold_tap

        target = tmp_path / "fresh-tap"
        scaffold_tap(target, name="fresh-tap", author="ci", init_git=False)
        assert lint_tap_directory(target) == []

    def test_aggregates_across_checks(self, tmp_path: Path):
        # Synthetic tap with multiple issues across checks.
        _make_tap(tmp_path, with_readme=False, with_license=False)
        (tmp_path / "data").mkdir()
        _add_plugin(tmp_path, "optimizers", "no_readme", with_readme=False)
        issues = lint_tap_directory(tmp_path)
        codes = {i.code for i in issues}
        assert "L001" in codes  # missing README
        assert "L002" in codes  # missing LICENSE
        assert "L005" in codes  # cruft dir
        assert "L004" in codes  # plugin missing README

    def test_check_crash_is_caught_not_propagated(self, tmp_path: Path):
        """A buggy check shouldn't kill the whole lint run."""
        _make_tap(tmp_path)

        class _BoomCheck:
            code = "BOOM"
            severity = "error"

            def run(self, _root: Path):
                raise RuntimeError("boom")

        reg = LintCheckRegistry()
        reg.register(_BoomCheck())  # type: ignore[arg-type]
        issues = lint_tap_directory(tmp_path, registry=reg)
        assert len(issues) == 1
        assert "boom" in issues[0].message.lower()


class TestLintCheckRegistry:
    def test_register_rejects_duplicate_code(self):
        reg = LintCheckRegistry()
        reg.register(L001_MissingTopLevelReadme())
        with pytest.raises(PluginError, match="duplicate"):
            reg.register(L001_MissingTopLevelReadme())

    def test_register_rejects_no_code(self):
        class _Empty:
            code = ""
            severity = "error"
            def run(self, _): return []
        with pytest.raises(PluginError, match="no code"):
            LintCheckRegistry().register(_Empty())  # type: ignore[arg-type]

    def test_get_returns_registered(self):
        reg = LintCheckRegistry()
        c = L001_MissingTopLevelReadme()
        reg.register(c)
        assert reg.get("L001") is c
        assert reg.get("MISSING") is None

    def test_default_registry_contains_all_builtins(self):
        reg = get_default_registry()
        codes = {c.code for c in reg.all()}
        # All 10 numbered checks plus L011 manifest validator.
        assert {"L001", "L002", "L003", "L004", "L005",
                "L006", "L007", "L008", "L009", "L010", "L011"} <= codes


class TestFormatLintReport:
    def test_empty_report(self, tmp_path: Path):
        out = format_lint_report([], tmp_path)
        assert "0 issues" in out

    def test_non_empty_report_lists_issues(self, tmp_path: Path):
        issue = LintIssue(
            code="L005", severity="error",
            message="data committed", path=tmp_path / "data",
            fix_hint="rm -rf data",
        )
        out = format_lint_report([issue], tmp_path)
        assert "L005" in out
        assert "rm -rf data" in out
        assert "errors:" in out
