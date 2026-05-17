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


# ---------------------------------------------------------------------------
# Phase A.3 — top-level tap.yaml schema
# ---------------------------------------------------------------------------


from crucible.core.plugin_schema import (
    parse_version_range,
    validate_tap_manifest_dict,
    validate_tap_manifest_file,
    version_matches_range,
)


class TestTapManifestSchema:
    def test_well_formed_returns_no_errors(self):
        data = {
            "name": "demo-tap",
            "description": "one-line desc",
            "version": "0.1.0",
            "author": "me",
            "license": "MIT",
            "crucible_compat": ">=0.2,<0.3",
            "homepage": "https://example.com",
            "maintainer_contact": "me@example.com",
        }
        errors = [i for i in validate_tap_manifest_dict(data) if i.severity == "error"]
        assert errors == []

    def test_missing_name_is_error(self):
        data = {"description": "x", "version": "0.1.0"}
        issues = validate_tap_manifest_dict(data)
        assert any(i.severity == "error" and i.field == "name" for i in issues)

    def test_missing_description_is_error(self):
        data = {"name": "x", "version": "0.1.0"}
        issues = validate_tap_manifest_dict(data)
        assert any(i.severity == "error" and i.field == "description" for i in issues)

    def test_missing_recommended_fields_are_warnings(self):
        data = {"name": "x", "description": "d", "version": "0.1.0"}
        issues = validate_tap_manifest_dict(data)
        warnings = {i.field for i in issues if i.severity == "warning"}
        assert {"author", "license", "crucible_compat"} <= warnings

    def test_multiline_description_is_warning(self):
        data = {
            "name": "x",
            "description": "line one\nline two",
            "version": "0.1.0",
        }
        issues = validate_tap_manifest_dict(data)
        assert any(i.severity == "warning" and i.field == "description" for i in issues)

    def test_non_string_root_is_error(self):
        issues = validate_tap_manifest_dict("not a dict")  # type: ignore[arg-type]
        assert any(i.severity == "error" for i in issues)

    def test_bad_name_pattern_is_error(self):
        issues = validate_tap_manifest_dict({
            "name": "has spaces", "description": "x", "version": "0.1.0",
        })
        assert any(i.severity == "error" and i.field == "name" for i in issues)

    def test_non_string_homepage_is_error(self):
        issues = validate_tap_manifest_dict({
            "name": "x", "description": "y", "version": "0.1.0", "homepage": 42,
        })
        assert any(i.severity == "error" and i.field == "homepage" for i in issues)

    def test_missing_file_is_warning_not_error(self, tmp_path: Path):
        # Optional manifest — its absence is OK.
        result = validate_tap_manifest_file(tmp_path / "tap.yaml")
        assert result.ok
        assert any(i.severity == "warning" for i in result.issues)

    def test_well_formed_file(self, tmp_path: Path):
        manifest = tmp_path / "tap.yaml"
        manifest.write_text(yaml.dump({
            "name": "x", "description": "y", "version": "0.1.0",
            "author": "me", "license": "MIT", "crucible_compat": ">=0.2,<0.3",
        }), encoding="utf-8")
        result = validate_tap_manifest_file(manifest)
        assert result.ok

    def test_invalid_yaml_in_file_is_error(self, tmp_path: Path):
        manifest = tmp_path / "tap.yaml"
        manifest.write_text("name: x\n  :::bad indent", encoding="utf-8")
        result = validate_tap_manifest_file(manifest)
        assert not result.ok


class TestParseVersionRange:
    def test_simple_geq(self):
        # Versions zero-padded to 3 segments for stable comparison.
        assert parse_version_range(">=0.2") == [(">=", (0, 2, 0))]

    def test_compound_range(self):
        assert parse_version_range(">=0.2,<0.3") == [
            (">=", (0, 2, 0)), ("<", (0, 3, 0)),
        ]

    def test_eq_normalizes(self):
        assert parse_version_range("=0.1.0") == [("==", (0, 1, 0))]
        assert parse_version_range("==0.1.0") == [("==", (0, 1, 0))]

    def test_strips_whitespace(self):
        assert parse_version_range(" >=0.1 , < 1.0 ") == [
            (">=", (0, 1, 0)), ("<", (1, 0, 0)),
        ]

    def test_empty_returns_empty_list(self):
        assert parse_version_range("") == []
        assert parse_version_range(None) == []  # type: ignore[arg-type]

    def test_malformed_raises_plugin_error(self):
        with pytest.raises(PluginError, match="Invalid version-range"):
            parse_version_range("not a range")

    def test_trailing_comma_skips_empty_token(self):
        assert parse_version_range(">=0.2,") == [(">=", (0, 2, 0))]


class TestVersionMatchesRange:
    def test_in_range(self):
        assert version_matches_range("0.2.1", ">=0.2,<0.3")

    def test_below_min(self):
        assert not version_matches_range("0.1.0", ">=0.2,<0.3")

    def test_at_or_above_upper_bound(self):
        assert not version_matches_range("0.3.0", ">=0.2,<0.3")
        assert version_matches_range("0.2.999", ">=0.2,<0.3")

    def test_pre_release_suffix_is_ignored(self):
        # 0.2.1-alpha should compare as 0.2.1 for range matching.
        assert version_matches_range("0.2.1-alpha", ">=0.2,<0.3")

    def test_eq_operator(self):
        assert version_matches_range("0.1.0", "==0.1.0")
        assert not version_matches_range("0.1.1", "==0.1.0")

    def test_lt_operator(self):
        assert version_matches_range("0.1.0", "<0.2")
        assert not version_matches_range("0.2.0", "<0.2")

    def test_gt_operator(self):
        assert version_matches_range("0.3.0", ">0.2")
        assert not version_matches_range("0.2.0", ">0.2")

    def test_le_operator(self):
        assert version_matches_range("0.2.0", "<=0.2")
        assert version_matches_range("0.1.9", "<=0.2")
        assert not version_matches_range("0.2.1", "<=0.2")

    def test_empty_range_is_permissive(self):
        assert version_matches_range("999.0.0", "")
        assert version_matches_range("0.0.0", None)  # type: ignore[arg-type]
