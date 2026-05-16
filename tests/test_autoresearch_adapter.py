"""Tests for the Karpathy-autoresearch importer (Phase 1.6)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from crucible.core.errors import CrucibleError
from crucible.runner.autoresearch_adapter import (
    AutoresearchSource,
    import_autoresearch,
    parse_autoresearch_source,
    sanitize_name,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_autoresearch_dir(
    parent: Path,
    *,
    name: str = "my_research",
    with_program: bool = True,
    siblings: tuple[str, ...] = (),
    requirements: tuple[str, ...] = (),
) -> Path:
    """Build a synthetic autoresearch source directory."""
    source = parent / name
    source.mkdir(parents=True, exist_ok=True)
    (source / "train.py").write_text(
        "import os\nprint('train_loss:', 1.0)\n", encoding="utf-8"
    )
    if with_program:
        (source / "program.md").write_text(
            "# Research program\nMinimise val_loss on the toy benchmark.\n",
            encoding="utf-8",
        )
    for sib in siblings:
        (source / sib).write_text("# sibling helper\n", encoding="utf-8")
    if requirements:
        (source / "requirements.txt").write_text(
            "\n".join(["# pinned deps", *requirements, ""]),
            encoding="utf-8",
        )
    return source


# ---------------------------------------------------------------------------
# sanitize_name
# ---------------------------------------------------------------------------


class TestSanitizeName:
    def test_basic_dir_name(self):
        assert sanitize_name("My-Cool-Project") == "my-cool-project"

    def test_strips_leading_nonalpha(self):
        assert sanitize_name("123abc") == "abc"

    def test_replaces_whitespace_and_dots(self):
        assert sanitize_name("foo bar.baz") == "foo_bar_baz"

    def test_empty_after_sanitize_raises(self):
        with pytest.raises(CrucibleError, match="Cannot derive"):
            sanitize_name("!!!")


# ---------------------------------------------------------------------------
# parse_autoresearch_source
# ---------------------------------------------------------------------------


class TestParseSource:
    def test_minimal_source(self, tmp_path: Path):
        source = _make_autoresearch_dir(tmp_path)
        parsed = parse_autoresearch_source(source)
        assert parsed.train_path == source / "train.py"
        assert parsed.program_path == source / "program.md"
        assert parsed.sibling_py == []
        assert parsed.requirements == []

    def test_detects_sibling_py_files(self, tmp_path: Path):
        source = _make_autoresearch_dir(
            tmp_path, siblings=("data.py", "model.py", "utils.py")
        )
        parsed = parse_autoresearch_source(source)
        sibling_names = {p.name for p in parsed.sibling_py}
        assert sibling_names == {"data.py", "model.py", "utils.py"}
        # train.py is NOT in siblings — it's tracked separately.
        assert source / "train.py" not in parsed.sibling_py

    def test_parses_requirements_strips_comments_and_blanks(self, tmp_path: Path):
        source = _make_autoresearch_dir(
            tmp_path,
            requirements=("torch==2.4.0", "", "# fix later", "numpy>=1.24"),
        )
        parsed = parse_autoresearch_source(source)
        assert parsed.requirements == ["torch==2.4.0", "numpy>=1.24"]

    def test_parses_requirements_skips_recursive_includes_and_editable(
        self, tmp_path: Path
    ):
        """Review-driven fix: -r and -e lines don't survive Crucible's
        per-entry install pattern, so the importer must skip them with a
        warning rather than poison the bootstrap."""
        source = _make_autoresearch_dir(
            tmp_path,
            requirements=(
                "torch==2.4.0",
                "-r other-requirements.txt",
                "--requirement deep.txt",
                "-e .",
                "--editable ./pkg",
                "numpy",
            ),
        )
        parsed = parse_autoresearch_source(source)
        assert parsed.requirements == ["torch==2.4.0", "numpy"]

    def test_missing_train_py_raises(self, tmp_path: Path):
        source = tmp_path / "broken"
        source.mkdir()
        (source / "program.md").write_text("desc", encoding="utf-8")
        with pytest.raises(CrucibleError, match="train.py"):
            parse_autoresearch_source(source)

    def test_missing_program_md_proceeds_with_warn(self, tmp_path: Path):
        source = _make_autoresearch_dir(tmp_path, with_program=False)
        parsed = parse_autoresearch_source(source)
        assert parsed.program_path is None

    def test_nonexistent_dir_raises(self, tmp_path: Path):
        with pytest.raises(CrucibleError, match="source directory not found"):
            parse_autoresearch_source(tmp_path / "no_such_dir")

    def test_local_files_property_includes_everything(self, tmp_path: Path):
        source = _make_autoresearch_dir(
            tmp_path,
            siblings=("data.py", "model.py"),
            requirements=("torch",),
        )
        parsed = parse_autoresearch_source(source)
        local = {p.name for p in parsed.local_files}
        assert local == {"train.py", "program.md", "data.py", "model.py", "requirements.txt"}


# ---------------------------------------------------------------------------
# import_autoresearch
# ---------------------------------------------------------------------------


class TestImport:
    def test_emits_valid_project_spec(self, tmp_path: Path):
        source = _make_autoresearch_dir(tmp_path, name="my_proj")
        result = import_autoresearch(source, project_root=tmp_path)

        assert result["name"] == "my_proj"
        assert result["validated"] is True

        spec_path = tmp_path / ".crucible" / "projects" / "my_proj.yaml"
        assert spec_path.exists()

        # Loads back through load_project_spec — the importer already
        # validated this, but assert independently.
        from crucible.core.config import load_project_spec
        spec = load_project_spec("my_proj", tmp_path)
        assert spec.name == "my_proj"
        assert spec.train == "python train.py"

    def test_local_files_includes_siblings(self, tmp_path: Path):
        source = _make_autoresearch_dir(
            tmp_path, siblings=("data.py", "model.py")
        )
        import_autoresearch(source, project_root=tmp_path)

        from crucible.core.config import load_project_spec
        spec = load_project_spec("my_research", tmp_path)
        local_names = {Path(p).name for p in spec.local_files}
        assert "train.py" in local_names
        assert "program.md" in local_names
        assert "data.py" in local_names
        assert "model.py" in local_names

    def test_install_populated_from_requirements(self, tmp_path: Path):
        source = _make_autoresearch_dir(
            tmp_path, requirements=("torch==2.4.0", "numpy")
        )
        import_autoresearch(source, project_root=tmp_path)

        from crucible.core.config import load_project_spec
        spec = load_project_spec("my_research", tmp_path)
        assert spec.install == ["torch==2.4.0", "numpy"]

    def test_program_md_copied_to_spec_dir(self, tmp_path: Path):
        source = _make_autoresearch_dir(tmp_path)
        result = import_autoresearch(source, project_root=tmp_path)

        program_dest = tmp_path / ".crucible" / "projects" / "my_research.md"
        assert program_dest.exists()
        assert "Research program" in program_dest.read_text(encoding="utf-8")
        assert result["program_path"] == str(program_dest)

    def test_no_program_md_no_sidecar_emitted(self, tmp_path: Path):
        source = _make_autoresearch_dir(tmp_path, with_program=False)
        result = import_autoresearch(source, project_root=tmp_path)

        assert result["program_path"] is None
        # Only the .yaml is in projects/, not a stray .md
        sidecars = list((tmp_path / ".crucible" / "projects").glob("*.md"))
        assert sidecars == []

    def test_existing_spec_without_force_raises(self, tmp_path: Path):
        source = _make_autoresearch_dir(tmp_path, name="dup_proj")
        import_autoresearch(source, project_root=tmp_path)
        # Second import without force.
        with pytest.raises(CrucibleError, match="already exists"):
            import_autoresearch(source, project_root=tmp_path)

    def test_force_overwrites(self, tmp_path: Path):
        source = _make_autoresearch_dir(tmp_path, name="dup_proj")
        import_autoresearch(source, project_root=tmp_path)
        # Mutate train.py, re-import with force.
        (source / "train.py").write_text("print('new')\n", encoding="utf-8")
        result = import_autoresearch(source, project_root=tmp_path, force=True)
        assert result["validated"] is True

    def test_custom_name(self, tmp_path: Path):
        source = _make_autoresearch_dir(tmp_path, name="raw_source")
        result = import_autoresearch(
            source, project_root=tmp_path, name="custom-name"
        )
        assert result["name"] == "custom-name"

    def test_validation_rollback_on_load_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """If load_project_spec raises during validation, the partial files
        are removed and the importer surfaces a clear error."""
        source = _make_autoresearch_dir(tmp_path, name="rollback_test")

        def failing_load(name: str, project_root: Path | None = None):  # type: ignore[no-redef]
            raise RuntimeError("simulated validation failure")

        monkeypatch.setattr(
            "crucible.core.config.load_project_spec", failing_load
        )
        with pytest.raises(CrucibleError, match="failed validation"):
            import_autoresearch(source, project_root=tmp_path)

        # No partial files left behind.
        spec_path = tmp_path / ".crucible" / "projects" / "rollback_test.yaml"
        program_dest = tmp_path / ".crucible" / "projects" / "rollback_test.md"
        assert not spec_path.exists()
        assert not program_dest.exists()

    def test_warns_when_local_files_outside_project_root(
        self, tmp_path: Path, capsys
    ):
        """Review-driven fix: when local_files paths are outside project_root,
        the emitted spec is non-portable. Surface that loudly at import time
        rather than letting the user discover it on first dispatch.

        ``log_warn`` writes directly to stderr (not through Python's
        ``logging``), so we capture stderr via ``capsys`` not ``caplog``.
        """
        source_parent = tmp_path / "external_source"
        source = _make_autoresearch_dir(source_parent, name="autoresearch_src")
        project_root = tmp_path / "project"
        project_root.mkdir()

        import_autoresearch(source, project_root=project_root)

        stderr = capsys.readouterr().err
        assert "non-portable" in stderr, (
            f"expected portability warning in stderr; got: {stderr!r}"
        )

    def test_no_warning_when_local_files_inside_project_root(
        self, tmp_path: Path, capsys
    ):
        """If the source is INSIDE the project root, no portability warning."""
        source = _make_autoresearch_dir(tmp_path, name="inside_src")
        # tmp_path acts as both project_root and parent of the source dir,
        # so local_files paths are inside project_root.
        import_autoresearch(source, project_root=tmp_path)

        stderr = capsys.readouterr().err
        assert "non-portable" not in stderr

    def test_metrics_passthrough(self, tmp_path: Path):
        source = _make_autoresearch_dir(tmp_path, name="metric_test")
        import_autoresearch(
            source,
            project_root=tmp_path,
            primary_metric="val_bpb",
            direction="minimize",
        )
        spec_path = tmp_path / ".crucible" / "projects" / "metric_test.yaml"
        raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        assert raw["metrics"]["primary"] == "val_bpb"
        assert raw["metrics"]["direction"] == "minimize"
