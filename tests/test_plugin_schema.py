"""Tests for the plugin manifest schema validator (Phase 3c)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from crucible.core.errors import PluginError
from crucible.core.plugin_schema import (
    KNOWN_PLUGIN_TYPES,
    ValidationIssue,
    validate_manifest_dict,
    validate_manifest_file,
    validate_tap_directory,
)


def _valid_manifest() -> dict:
    return {
        "name": "my_plugin",
        "type": "callbacks",
        "version": "0.1.0",
        "description": "Does a thing",
        "author": "tester",
        "tags": ["example"],
        "crucible_compat": ">=0.2,<0.3",
        "dependencies": ["torch>=2.0"],
    }


# ---------------------------------------------------------------------------
# validate_manifest_dict
# ---------------------------------------------------------------------------


class TestValidateManifestDict:
    def test_fully_populated_manifest_passes_cleanly(self):
        issues = validate_manifest_dict(_valid_manifest())
        assert issues == []

    def test_minimal_valid_has_warnings_only(self):
        """Required fields only → passes but warns about missing author/tags/compat/deps."""
        issues = validate_manifest_dict({
            "name": "x",
            "type": "callbacks",
            "version": "0.1.0",
            "description": "small",
        })
        errors = [i for i in issues if i.severity == "error"]
        warnings = [i for i in issues if i.severity == "warning"]
        assert errors == []
        assert len(warnings) >= 3  # author, tags, compat, deps all missing

    def test_missing_name_is_error(self):
        issues = validate_manifest_dict({"type": "callbacks", "version": "0.1.0", "description": "x"})
        assert any(i.severity == "error" and i.field == "name" for i in issues)

    def test_invalid_name_pattern_is_error(self):
        data = _valid_manifest()
        data["name"] = "has spaces"
        issues = validate_manifest_dict(data)
        assert any(i.severity == "error" and i.field == "name" for i in issues)

    def test_missing_type_is_error(self):
        data = _valid_manifest()
        del data["type"]
        issues = validate_manifest_dict(data)
        assert any(i.severity == "error" and i.field == "type" for i in issues)

    def test_unknown_type_is_warning_not_error(self):
        data = _valid_manifest()
        data["type"] = "not_a_real_category"
        issues = validate_manifest_dict(data)
        warnings_on_type = [
            i for i in issues
            if i.field == "type" and i.severity == "warning"
        ]
        assert len(warnings_on_type) == 1

    def test_all_known_types_accepted(self):
        for t in KNOWN_PLUGIN_TYPES:
            data = _valid_manifest()
            data["type"] = t
            issues = validate_manifest_dict(data)
            assert not any(
                i.severity == "error" and i.field == "type" for i in issues
            ), f"type {t!r} was rejected"

    def test_missing_version_is_error(self):
        data = _valid_manifest()
        del data["version"]
        issues = validate_manifest_dict(data)
        assert any(i.severity == "error" and i.field == "version" for i in issues)

    def test_non_semver_version_is_warning(self):
        data = _valid_manifest()
        data["version"] = "v1"
        issues = validate_manifest_dict(data)
        assert any(i.severity == "warning" and i.field == "version" for i in issues)

    def test_semver_prerelease_ok(self):
        data = _valid_manifest()
        data["version"] = "1.0.0-rc1"
        issues = validate_manifest_dict(data)
        assert not any(i.severity == "warning" and i.field == "version" for i in issues)

    def test_missing_description_is_error(self):
        data = _valid_manifest()
        del data["description"]
        issues = validate_manifest_dict(data)
        assert any(i.severity == "error" and i.field == "description" for i in issues)

    def test_overlong_description_is_warning(self):
        data = _valid_manifest()
        data["description"] = "x" * 600
        issues = validate_manifest_dict(data)
        assert any(
            i.severity == "warning" and i.field == "description" for i in issues
        )

    def test_root_must_be_mapping(self):
        issues = validate_manifest_dict(["not", "a", "dict"])  # type: ignore[arg-type]
        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert "mapping" in issues[0].message

    def test_tags_must_be_list(self):
        data = _valid_manifest()
        data["tags"] = "oops"
        issues = validate_manifest_dict(data)
        assert any(i.severity == "error" and i.field == "tags" for i in issues)

    def test_tags_items_must_be_strings(self):
        data = _valid_manifest()
        data["tags"] = ["ok", 42]
        issues = validate_manifest_dict(data)
        assert any("tags[1]" in i.field and i.severity == "error" for i in issues)

    def test_dependencies_must_be_list(self):
        data = _valid_manifest()
        data["dependencies"] = "torch"
        issues = validate_manifest_dict(data)
        assert any(i.severity == "error" and i.field == "dependencies" for i in issues)

    def test_dependencies_can_be_strings_or_dicts(self):
        data = _valid_manifest()
        data["dependencies"] = ["torch>=2.0", {"name": "numpy", "version": "*"}]
        issues = validate_manifest_dict(data)
        assert not any(i.severity == "error" for i in issues)

    def test_dependency_dict_must_have_name(self):
        data = _valid_manifest()
        data["dependencies"] = [{"version": "1.0"}]
        issues = validate_manifest_dict(data)
        assert any(
            i.severity == "error" and "dependencies[0]" in i.field for i in issues
        )


# ---------------------------------------------------------------------------
# validate_manifest_file
# ---------------------------------------------------------------------------


class TestValidateManifestFile:
    def test_missing_file(self, tmp_path: Path):
        result = validate_manifest_file(tmp_path / "nope.yaml")
        assert not result.ok
        assert len(result.errors) == 1

    def test_valid_file(self, tmp_path: Path):
        path = tmp_path / "plugin.yaml"
        path.write_text(yaml.safe_dump(_valid_manifest()))
        result = validate_manifest_file(path)
        assert result.ok
        assert result.errors == []

    def test_invalid_yaml(self, tmp_path: Path):
        path = tmp_path / "plugin.yaml"
        path.write_text("not: [valid yaml")
        result = validate_manifest_file(path)
        assert not result.ok
        assert any("yaml" in i.field.lower() for i in result.errors)

    def test_result_properties(self, tmp_path: Path):
        path = tmp_path / "plugin.yaml"
        path.write_text(yaml.safe_dump({
            "name": "x",
            "type": "callbacks",
            "version": "0.1.0",
            "description": "x",
        }))
        result = validate_manifest_file(path)
        assert result.ok
        assert result.errors == []
        assert len(result.warnings) > 0


# ---------------------------------------------------------------------------
# validate_tap_directory
# ---------------------------------------------------------------------------


class TestValidateTapDirectory:
    def _write_plugin(self, tap: Path, category: str, name: str, manifest: dict) -> Path:
        plugin_dir = tap / category / name
        plugin_dir.mkdir(parents=True)
        path = plugin_dir / "plugin.yaml"
        path.write_text(yaml.safe_dump(manifest))
        return path

    def test_walks_tap_and_finds_every_manifest(self, tmp_path: Path):
        tap = tmp_path / "tap"
        self._write_plugin(tap, "callbacks", "good_one", _valid_manifest())
        self._write_plugin(tap, "architectures", "another", _valid_manifest())
        results = validate_tap_directory(tap)
        assert len(results) == 2

    def test_skips_blocked_directories(self, tmp_path: Path):
        tap = tmp_path / "tap"
        # Real plugin
        self._write_plugin(tap, "callbacks", "real", _valid_manifest())
        # Should be skipped
        self._write_plugin(tap, "findings", "old_plugin", _valid_manifest())
        self._write_plugin(tap, "wandb", "artifact", _valid_manifest())
        self._write_plugin(tap, "_manuscript", "private", _valid_manifest())
        results = validate_tap_directory(tap)
        # Only the callbacks/real one
        assert len(results) == 1
        assert "callbacks" in str(results[0].path)

    def test_errors_aggregated_per_plugin(self, tmp_path: Path):
        tap = tmp_path / "tap"
        good = _valid_manifest()
        bad = {"name": "bad"}  # missing type, version, description
        self._write_plugin(tap, "callbacks", "good", good)
        self._write_plugin(tap, "callbacks", "bad", bad)
        results = validate_tap_directory(tap)
        by_name = {
            "good" if "good" in str(r.path) else "bad": r for r in results
        }
        assert by_name["good"].ok
        assert not by_name["bad"].ok
        assert len(by_name["bad"].errors) >= 3

    def test_missing_tap_raises(self, tmp_path: Path):
        with pytest.raises(PluginError, match="does not exist"):
            validate_tap_directory(tmp_path / "nope")

    def test_non_directory_raises(self, tmp_path: Path):
        f = tmp_path / "not_a_dir"
        f.write_text("x")
        with pytest.raises(PluginError, match="not a directory"):
            validate_tap_directory(f)
