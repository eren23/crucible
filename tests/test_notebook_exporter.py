"""Tests for the notebook exporter.

Covers: spec loading, runtime resolution, source generation, cell layout,
env merging (env_set + variant + overrides), secrets injection, and the
`.py`/`.ipynb` write paths.

Does NOT require jupytext at import time — the `.ipynb` write path is
smoke-tested with `pytest.importorskip("jupytext")`.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from crucible.notebook import export_project
from crucible.notebook.exporter import NotebookExportError
from crucible.notebook.runtimes import get_runtime, list_runtimes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_spec(project_root: Path, name: str, body: dict) -> Path:
    """Materialize a project spec under <project_root>/.crucible/projects/<name>.yaml."""
    specs_dir = project_root / ".crucible" / "projects"
    specs_dir.mkdir(parents=True, exist_ok=True)
    path = specs_dir / f"{name}.yaml"
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def smoke_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Minimal spec resolvable via load_project_spec with tmp_path as cwd."""
    body = {
        "name": "smoke_test",
        "description": "minimal smoke project",
        "repo": "",
        "branch": "main",
        "install": ["datasets>=2.18"],
        "train": "python3 -c 'print(1)'",
        "env_forward": ["HF_TOKEN"],
        "env_set": {"PYTHONUNBUFFERED": "1", "SMOKE_KEY": "base"},
        "variants": {"heavy": {"SMOKE_KEY": "heavy", "EXTRA_VAR": "v"}},
        "eval_suite": [],
    }
    _write_spec(tmp_path, "smoke_test", body)
    monkeypatch.chdir(tmp_path)
    return "smoke_test"


@pytest.fixture
def eval_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Spec with a populated eval_suite for testing the eval cell."""
    body = {
        "name": "eval_test",
        "train": "python3 -c 'print(2)'",
        "eval_suite": [
            {"script": "evaluation/dummy.py", "args": ["--device", "cpu"]},
            {"script": "evaluation/other.py", "args": []},
        ],
    }
    _write_spec(tmp_path, "eval_test", body)
    monkeypatch.chdir(tmp_path)
    return "eval_test"


@pytest.fixture
def repo_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Spec with an external repo — tests the clone cell."""
    body = {
        "name": "repo_test",
        "repo": "https://github.com/example/proj",
        "branch": "dev",
        "shallow": True,
        "train": "make train",
    }
    _write_spec(tmp_path, "repo_test", body)
    monkeypatch.chdir(tmp_path)
    return "repo_test"


# ---------------------------------------------------------------------------
# Runtime profile tests
# ---------------------------------------------------------------------------


def test_runtime_profiles_listed():
    rows = list_runtimes()
    names = {r["name"] for r in rows}
    assert {"colab-h100", "colab-a100", "colab-t4", "local"} <= names


def test_get_runtime_unknown_raises():
    with pytest.raises(ValueError, match="Unknown runtime"):
        get_runtime("not-a-real-runtime")


def test_colab_h100_has_flash_attn():
    rt = get_runtime("colab-h100")
    assert rt.flash_attn is True
    assert "accelerate" in " ".join(rt.extra_pip_packages)


def test_colab_t4_no_flash_attn():
    rt = get_runtime("colab-t4")
    assert rt.flash_attn is False


def test_local_no_colab_secrets():
    rt = get_runtime("local")
    assert rt.colab_secrets is False


# ---------------------------------------------------------------------------
# Core export tests
# ---------------------------------------------------------------------------


def test_export_to_py(smoke_spec: str, tmp_path: Path):
    out = tmp_path / "smoke.py"
    result = export_project(
        project=smoke_spec, runtime="colab-h100", preset="smoke", out_path=out
    )
    assert out.exists()
    assert result.cells >= 7
    text = out.read_text(encoding="utf-8")
    # Jupytext header + percent markers
    assert text.startswith("# ---\n# jupyter:")
    assert "# %%" in text
    assert "# %% [markdown]" in text


def test_emits_all_expected_cells(smoke_spec: str, tmp_path: Path):
    out = tmp_path / "cells.py"
    export_project(project=smoke_spec, out_path=out)
    text = out.read_text(encoding="utf-8")
    # Each major cell has a recognizable anchor
    assert "# --- Install dependencies ---" in text
    assert "# --- Secrets ---" in text
    assert "# --- Effective configuration ---" in text
    assert "# --- Train ---" in text
    assert "# --- Upload (optional) ---" in text


def test_variant_overrides_layered(smoke_spec: str, tmp_path: Path):
    out = tmp_path / "variant.py"
    export_project(
        project=smoke_spec, runtime="local",
        variant="heavy",
        overrides={"SMOKE_KEY": "cli-winner", "NEW_ONE": "yes"},
        out_path=out,
    )
    text = out.read_text(encoding="utf-8")
    # Precedence: env_set < variant < overrides. cli-winner must win.
    assert '"SMOKE_KEY": "cli-winner"' in text
    # Variant-only key survives when no override clashes
    assert '"EXTRA_VAR": "v"' in text
    # Overrides introduce new keys
    assert '"NEW_ONE": "yes"' in text


def test_variant_without_overrides(smoke_spec: str, tmp_path: Path):
    out = tmp_path / "variant2.py"
    export_project(project=smoke_spec, variant="heavy", out_path=out)
    text = out.read_text(encoding="utf-8")
    assert '"SMOKE_KEY": "heavy"' in text


def test_no_variant_uses_env_set(smoke_spec: str, tmp_path: Path):
    out = tmp_path / "base.py"
    export_project(project=smoke_spec, out_path=out)
    text = out.read_text(encoding="utf-8")
    assert '"SMOKE_KEY": "base"' in text


def test_secrets_cell_includes_forward_keys(smoke_spec: str, tmp_path: Path):
    out = tmp_path / "secrets.py"
    export_project(project=smoke_spec, out_path=out)
    text = out.read_text(encoding="utf-8")
    # HF_TOKEN is in env_forward; should appear in the forward keys list.
    assert "HF_TOKEN" in text
    # Standard secrets are always included
    assert "WANDB_API_KEY" in text
    assert "ANTHROPIC_API_KEY" in text


def test_local_runtime_skips_colab_userdata(smoke_spec: str, tmp_path: Path):
    out = tmp_path / "local.py"
    export_project(project=smoke_spec, runtime="local", out_path=out)
    text = out.read_text(encoding="utf-8")
    assert "google.colab" not in text


def test_colab_runtime_uses_colab_userdata(smoke_spec: str, tmp_path: Path):
    out = tmp_path / "colab.py"
    export_project(project=smoke_spec, runtime="colab-h100", out_path=out)
    text = out.read_text(encoding="utf-8")
    assert "google.colab" in text
    assert "userdata.get" in text


def test_eval_cell_includes_all_scripts(eval_spec: str, tmp_path: Path):
    out = tmp_path / "evals.py"
    export_project(project=eval_spec, out_path=out)
    text = out.read_text(encoding="utf-8")
    assert "evaluation/dummy.py" in text
    assert "evaluation/other.py" in text
    assert "--checkpoint" in text


def test_empty_eval_suite_emits_markdown(smoke_spec: str, tmp_path: Path):
    out = tmp_path / "noeval.py"
    export_project(project=smoke_spec, out_path=out)
    text = out.read_text(encoding="utf-8")
    assert "No `eval_suite:`" in text


def test_clone_cell_when_repo_set(repo_spec: str, tmp_path: Path):
    out = tmp_path / "clone.py"
    export_project(project=repo_spec, out_path=out)
    text = out.read_text(encoding="utf-8")
    assert "git clone" in text
    assert "--branch dev" in text
    assert "https://github.com/example/proj" in text


def test_no_clone_cell_when_repo_empty(smoke_spec: str, tmp_path: Path):
    out = tmp_path / "noclone.py"
    export_project(project=smoke_spec, out_path=out)
    text = out.read_text(encoding="utf-8")
    assert "No `repo:` in project spec" in text


def test_missing_project_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        export_project(project="nope_does_not_exist", out_path=tmp_path / "x.py")


def test_inline_plugins_not_implemented(smoke_spec: str, tmp_path: Path):
    with pytest.raises(NotebookExportError, match="inline-plugins"):
        export_project(
            project=smoke_spec, inline_plugins=True, out_path=tmp_path / "x.py"
        )


def test_unsupported_suffix_raises(smoke_spec: str, tmp_path: Path):
    with pytest.raises(NotebookExportError, match="Unsupported output suffix"):
        export_project(project=smoke_spec, out_path=tmp_path / "x.txt")


# ---------------------------------------------------------------------------
# Round-trip via jupytext (optional)
# ---------------------------------------------------------------------------


def test_ipynb_output_via_jupytext(smoke_spec: str, tmp_path: Path):
    pytest.importorskip("jupytext")
    out = tmp_path / "smoke.ipynb"
    result = export_project(project=smoke_spec, out_path=out)
    assert out.exists()
    # Sibling .py also written for diffs
    assert result.source_path is not None
    assert result.source_path.suffix == ".py"
    assert result.source_path.exists()


def test_ipynb_roundtrip_preserves_cells(smoke_spec: str, tmp_path: Path):
    jupytext = pytest.importorskip("jupytext")
    out = tmp_path / "rt.ipynb"
    result = export_project(project=smoke_spec, out_path=out)
    nb = jupytext.read(out)
    # Cells in the notebook should match the exporter's count
    assert len(nb.cells) == result.cells


# ---------------------------------------------------------------------------
# Compilability check — exported .py parses as valid Python
# ---------------------------------------------------------------------------


def test_exported_py_is_valid_python(smoke_spec: str, tmp_path: Path):
    import ast

    out = tmp_path / "compile.py"
    export_project(project=smoke_spec, out_path=out)
    source = out.read_text(encoding="utf-8")
    ast.parse(source)  # raises SyntaxError if malformed


def test_exported_py_with_variant_is_valid_python(smoke_spec: str, tmp_path: Path):
    import ast

    out = tmp_path / "variant_compile.py"
    export_project(
        project=smoke_spec, variant="heavy",
        overrides={"X": "1", "Y": "with 'quote'"},
        out_path=out,
    )
    source = out.read_text(encoding="utf-8")
    ast.parse(source)


def test_exported_py_with_eval_suite_is_valid_python(eval_spec: str, tmp_path: Path):
    import ast

    out = tmp_path / "eval_compile.py"
    export_project(project=eval_spec, out_path=out)
    source = out.read_text(encoding="utf-8")
    ast.parse(source)


def test_exported_py_with_repo_is_valid_python(repo_spec: str, tmp_path: Path):
    import ast

    out = tmp_path / "repo_compile.py"
    export_project(project=repo_spec, out_path=out)
    source = out.read_text(encoding="utf-8")
    ast.parse(source)


def test_result_to_dict_serializable(smoke_spec: str, tmp_path: Path):
    import json

    out = tmp_path / "tod.py"
    result = export_project(project=smoke_spec, out_path=out)
    d = result.to_dict()
    # Must be JSON-serializable (for MCP tool responses)
    json.dumps(d)
    assert d["project"] == smoke_spec
    assert d["cells"] >= 7
