"""Tests for crucible.core.project_template."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from crucible.core.errors import ProjectTemplateError
from crucible.core.project_template import (
    TemplateInfo,
    _merge_dicts,
    _resolve_extends_text,
    _templates_dir,
    extract_vars,
    find_template,
    iter_required_missing,
    list_templates,
    render_template,
    required_vars,
    substitute,
    validate_spec,
    write_project_spec,
)


# ---------------------------------------------------------------------------
# substitute
# ---------------------------------------------------------------------------


class TestSubstitute:
    def test_replaces_simple_var(self):
        assert substitute("hello ${NAME}", {"NAME": "world"}) == "hello world"

    def test_replaces_multiple_occurrences(self):
        out = substitute("${A} and ${A} and ${B}", {"A": "x", "B": "y"})
        assert out == "x and x and y"

    def test_uses_default_when_missing(self):
        assert substitute("${NAME:default}", {}) == "default"

    def test_explicit_value_beats_default(self):
        assert substitute("${NAME:default}", {"NAME": "explicit"}) == "explicit"

    def test_empty_default(self):
        assert substitute("${EMPTY:}", {}) == ""

    def test_raises_on_missing_required(self):
        with pytest.raises(ProjectTemplateError) as exc:
            substitute("hello ${WHO}", {})
        assert "WHO" in str(exc.value)

    def test_raises_lists_all_missing(self):
        with pytest.raises(ProjectTemplateError) as exc:
            substitute("${A} ${B} ${C}", {"B": "ok"})
        msg = str(exc.value)
        assert "A" in msg and "C" in msg and "B" not in msg

    def test_ignores_lowercase_vars(self):
        # Pattern requires UPPER_SNAKE_CASE
        out = substitute("${lower} and ${UPPER}", {"UPPER": "val"})
        assert out == "${lower} and val"

    def test_numeric_value_is_stringified(self):
        assert substitute("port=${PORT}", {"PORT": 8080}) == "port=8080"


# ---------------------------------------------------------------------------
# extract_vars
# ---------------------------------------------------------------------------


class TestExtractVars:
    def test_extracts_names_and_defaults(self):
        text = "${A} ${B:default_b} ${C}"
        assert extract_vars(text) == [("A", None), ("B", "default_b"), ("C", None)]

    def test_preserves_duplicates(self):
        assert extract_vars("${X} ${X}") == [("X", None), ("X", None)]

    def test_empty_input(self):
        assert extract_vars("") == []

    def test_no_vars(self):
        assert extract_vars("plain text") == []


# ---------------------------------------------------------------------------
# Built-in template discovery
# ---------------------------------------------------------------------------


class TestBuiltinTemplates:
    def test_templates_dir_exists(self):
        assert _templates_dir().is_dir()

    def test_list_templates_returns_expected_names(self):
        names = {t.name for t in list_templates()}
        assert {"generic", "lm", "vision", "diffusion", "world_model"} <= names

    def test_list_templates_populates_descriptions(self):
        templates = list_templates()
        assert all(isinstance(t, TemplateInfo) for t in templates)
        # Every shipped template should have a description comment
        for t in templates:
            assert t.description, f"{t.name} has no description"

    def test_find_template_resolves_builtin(self):
        path = find_template("generic")
        assert path.exists() and path.suffix == ".yaml"

    def test_find_template_raises_on_unknown(self):
        with pytest.raises(ProjectTemplateError) as exc:
            find_template("does_not_exist_12345")
        assert "Unknown project template" in str(exc.value)

    def test_each_template_lists_required_vars(self):
        # Sanity: every template has at least PROJECT_NAME + REPO_URL required
        for t in list_templates():
            req = set(required_vars(t.name))
            assert "PROJECT_NAME" in req
            assert "REPO_URL" in req, f"{t.name} missing REPO_URL"

    def test_each_template_renders_with_minimal_vars(self):
        values = {
            "PROJECT_NAME": "test-proj",
            "REPO_URL": "https://github.com/test/repo",
        }
        for t in list_templates():
            text = render_template(t.name, values, include_env=False)
            # Must parse as yaml and be a dict
            data = yaml.safe_load(text)
            assert isinstance(data, dict), f"{t.name} did not render to a dict"
            assert data.get("name") == "test-proj"
            # No unresolved vars
            assert "${" not in text, f"{t.name} has unresolved vars:\n{text}"

    def test_no_personal_identities_in_templates(self):
        """Templates must not bake in personal usernames or dataset paths."""
        blocked = ("eren23", "willdepueoai")
        for t in list_templates():
            text = t.path.read_text(encoding="utf-8")
            for token in blocked:
                assert token not in text, (
                    f"Template {t.name} contains forbidden token {token!r}"
                )


# ---------------------------------------------------------------------------
# required_vars
# ---------------------------------------------------------------------------


class TestRequiredVars:
    def test_generic_template_requires_name_and_repo(self, tmp_path: Path):
        req = required_vars("generic")
        assert "PROJECT_NAME" in req
        assert "REPO_URL" in req

    def test_defaults_are_not_required(self):
        req = required_vars("generic")
        # GPU_TYPE has a default, should not be required
        assert "GPU_TYPE" not in req
        assert "CONTAINER_DISK" not in req
        assert "BRANCH" not in req

    def test_variable_required_if_any_occurrence_lacks_default(
        self, tmp_path: Path, monkeypatch
    ):
        # Write a temp template with a mixed var
        tdir = tmp_path / "templates" / "projects"
        tdir.mkdir(parents=True)
        (tdir / "mixed.yaml").write_text("a: ${FOO}\nb: ${FOO:default}\n")
        monkeypatch.setattr(
            "crucible.core.project_template._templates_dir", lambda: tdir
        )
        assert "FOO" in required_vars("mixed")


# ---------------------------------------------------------------------------
# _merge_dicts
# ---------------------------------------------------------------------------


class TestMergeDicts:
    def test_child_overrides_parent(self):
        parent = {"a": 1, "b": 2}
        child = {"b": 3, "c": 4}
        assert _merge_dicts(parent, child) == {"a": 1, "b": 3, "c": 4}

    def test_nested_dicts_merge(self):
        parent = {"pod": {"gpu_type": "A", "disk": 10}}
        child = {"pod": {"disk": 20, "new": 1}}
        assert _merge_dicts(parent, child) == {
            "pod": {"gpu_type": "A", "disk": 20, "new": 1}
        }

    def test_lists_are_replaced(self):
        parent = {"install": ["a", "b"]}
        child = {"install": ["c"]}
        assert _merge_dicts(parent, child) == {"install": ["c"]}

    def test_deeply_nested_merge(self):
        parent = {"x": {"y": {"z": 1, "w": 2}}}
        child = {"x": {"y": {"w": 3}}}
        assert _merge_dicts(parent, child) == {"x": {"y": {"z": 1, "w": 3}}}


# ---------------------------------------------------------------------------
# extends resolution
# ---------------------------------------------------------------------------


class TestExtends:
    def test_builtin_template_extends_generic(self, tmp_path: Path):
        # lm extends generic; rendered lm should have generic's default fields
        text = render_template(
            "lm",
            {"PROJECT_NAME": "x", "REPO_URL": "y"},
            include_env=False,
        )
        data = yaml.safe_load(text)
        # generic provides env_forward / env_set / metrics
        assert "env_forward" in data
        assert "metrics" in data

    def test_extends_with_temp_templates(self, tmp_path: Path, monkeypatch):
        tdir = tmp_path / "templates" / "projects"
        tdir.mkdir(parents=True)
        (tdir / "parent.yaml").write_text(
            "name: ${PROJECT_NAME}\ngpu_count: 1\nshared: parent_value\n"
        )
        (tdir / "child.yaml").write_text(
            "extends: parent\nname: ${PROJECT_NAME}\nshared: child_value\n"
        )
        monkeypatch.setattr(
            "crucible.core.project_template._templates_dir", lambda: tdir
        )
        text = render_template(
            "child", {"PROJECT_NAME": "p"}, include_env=False
        )
        data = yaml.safe_load(text)
        assert data["shared"] == "child_value"  # child wins
        assert data["gpu_count"] == 1  # inherited

    def test_extends_cycle_is_detected(self, tmp_path: Path, monkeypatch):
        tdir = tmp_path / "templates" / "projects"
        tdir.mkdir(parents=True)
        (tdir / "a.yaml").write_text("extends: b\nname: ${PROJECT_NAME}\n")
        (tdir / "b.yaml").write_text("extends: a\nname: ${PROJECT_NAME}\n")
        monkeypatch.setattr(
            "crucible.core.project_template._templates_dir", lambda: tdir
        )
        with pytest.raises(ProjectTemplateError) as exc:
            render_template("a", {"PROJECT_NAME": "x"})
        assert "extends depth" in str(exc.value)


# ---------------------------------------------------------------------------
# validate_spec
# ---------------------------------------------------------------------------


class TestValidateSpec:
    def test_valid_spec_passes(self):
        validate_spec("name: test\nrepo: https://example.com/foo.git\n")

    def test_missing_name_fails(self):
        with pytest.raises(ProjectTemplateError) as exc:
            validate_spec("repo: https://example.com/foo.git\n")
        assert "name" in str(exc.value)

    def test_unresolved_var_fails(self):
        with pytest.raises(ProjectTemplateError) as exc:
            validate_spec("name: test\nvar: ${UNRESOLVED}\n")
        assert "UNRESOLVED" in str(exc.value)

    def test_non_mapping_fails(self):
        with pytest.raises(ProjectTemplateError):
            validate_spec("- just\n- a\n- list\n")

    def test_bad_yaml_fails(self):
        with pytest.raises(ProjectTemplateError):
            validate_spec("name: [\n")


# ---------------------------------------------------------------------------
# write_project_spec
# ---------------------------------------------------------------------------


class TestWriteProjectSpec:
    def test_writes_to_crucible_projects_dir(self, tmp_path: Path):
        target = write_project_spec(
            "generic",
            "my-test-proj",
            {"REPO_URL": "https://github.com/me/repo"},
            project_root=tmp_path,
        )
        expected = tmp_path / ".crucible" / "projects" / "my-test-proj.yaml"
        assert target == expected
        assert expected.exists()
        data = yaml.safe_load(expected.read_text())
        assert data["name"] == "my-test-proj"
        assert data["repo"] == "https://github.com/me/repo"

    def test_auto_populates_project_name(self, tmp_path: Path):
        # PROJECT_NAME should be inferred from the project name arg
        target = write_project_spec(
            "generic",
            "auto-named",
            {"REPO_URL": "https://x/y"},
            project_root=tmp_path,
        )
        data = yaml.safe_load(target.read_text())
        assert data["name"] == "auto-named"

    def test_refuses_to_overwrite_by_default(self, tmp_path: Path):
        write_project_spec(
            "generic",
            "foo",
            {"REPO_URL": "https://x/y"},
            project_root=tmp_path,
        )
        with pytest.raises(ProjectTemplateError) as exc:
            write_project_spec(
                "generic",
                "foo",
                {"REPO_URL": "https://x/y"},
                project_root=tmp_path,
            )
        assert "already exists" in str(exc.value)

    def test_overwrite_flag_replaces(self, tmp_path: Path):
        write_project_spec(
            "generic",
            "foo",
            {"REPO_URL": "https://first/one"},
            project_root=tmp_path,
        )
        target = write_project_spec(
            "generic",
            "foo",
            {"REPO_URL": "https://second/two"},
            project_root=tmp_path,
            overwrite=True,
        )
        data = yaml.safe_load(target.read_text())
        assert data["repo"] == "https://second/two"

    def test_raises_on_missing_required_var(self, tmp_path: Path):
        # Don't supply REPO_URL and don't use env
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ProjectTemplateError) as exc:
                write_project_spec(
                    "generic",
                    "missing",
                    {},
                    project_root=tmp_path,
                )
            assert "REPO_URL" in str(exc.value)


# ---------------------------------------------------------------------------
# iter_required_missing
# ---------------------------------------------------------------------------


class TestIterRequiredMissing:
    def test_reports_missing_when_nothing_supplied(self):
        with patch.dict(os.environ, {}, clear=True):
            missing = iter_required_missing("generic", [])
            # PROJECT_NAME is auto-populated, so only REPO_URL should remain
            assert "REPO_URL" in missing
            assert "PROJECT_NAME" not in missing

    def test_supplied_keys_filter_out(self):
        with patch.dict(os.environ, {}, clear=True):
            missing = iter_required_missing("generic", ["REPO_URL"])
            assert "REPO_URL" not in missing

    def test_env_keys_filter_out(self):
        with patch.dict(os.environ, {"REPO_URL": "https://x/y"}, clear=True):
            missing = iter_required_missing("generic", [])
            assert "REPO_URL" not in missing


# ---------------------------------------------------------------------------
# end-to-end: rendered spec loads as ProjectSpec
# ---------------------------------------------------------------------------


class TestRenderedSpecLoads:
    @pytest.mark.parametrize(
        "template_name",
        ["generic", "lm", "vision", "diffusion", "world_model"],
    )
    def test_rendered_spec_loads_as_project_spec(
        self, tmp_path: Path, template_name: str
    ):
        """Every rendered template must be loadable by load_project_spec."""
        from crucible.core.config import load_project_spec

        write_project_spec(
            template_name,
            f"test-{template_name}",
            {"REPO_URL": "https://github.com/test/repo"},
            project_root=tmp_path,
        )
        spec = load_project_spec(f"test-{template_name}", project_root=tmp_path)
        assert spec.name == f"test-{template_name}"
        assert spec.repo == "https://github.com/test/repo"
