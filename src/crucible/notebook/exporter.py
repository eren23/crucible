"""Project spec → standalone notebook export.

`export_project(project=..., runtime=..., out_path=...)` is the single entry
point. It loads the project spec, resolves the runtime profile, renders the
jupytext source via `templates.assemble_source`, and writes either a `.py`
(percent-format) or `.ipynb` file depending on `out_path` suffix.

`.ipynb` output requires the optional `jupytext` dependency. `.py` output
is pure stdlib.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from crucible.core.config import load_project_spec
from crucible.core.errors import CrucibleError
from crucible.notebook.runtimes import get_runtime
from crucible.notebook.templates import RenderContext, assemble_source


class NotebookExportError(CrucibleError):
    """Raised when notebook export fails (missing deps, unwritable path, etc.)."""


@dataclass
class ExportResult:
    """Return value of `export_project`."""

    project: str
    runtime: str
    preset: str
    variant: str | None
    out_path: Path
    source_path: Path | None         # jupytext .py if we also wrote one
    cells: int                       # cell count for quick verification
    size_bytes: int
    open_in_colab_url: str | None    # github-derived colab URL, or None if unknown

    def to_dict(self) -> dict[str, Any]:
        """Serialize for MCP tool responses."""
        return {
            "project": self.project,
            "runtime": self.runtime,
            "preset": self.preset,
            "variant": self.variant,
            "out_path": str(self.out_path),
            "source_path": str(self.source_path) if self.source_path else None,
            "cells": self.cells,
            "size_bytes": self.size_bytes,
            "open_in_colab_url": self.open_in_colab_url,
        }


def export_project(
    project: str,
    runtime: str = "colab-h100",
    preset: str = "smoke",
    out_path: str | Path | None = None,
    variant: str | None = None,
    overrides: dict[str, str] | None = None,
    inline_plugins: bool = False,
    crucible_install: str | None = None,
    project_root: Path | None = None,
) -> ExportResult:
    """Export a project spec as a standalone notebook.

    Args:
        project: project spec name (resolved by `load_project_spec`).
        runtime: runtime profile name (colab-h100, colab-a100, colab-t4, local).
        preset: Crucible preset to annotate; the notebook does NOT re-apply
            preset step counts — that's the train script's job.
        out_path: destination. `.py` or `.ipynb`. Default: `<project>.ipynb`.
        variant: named variant from `spec.variants`; env is layered on top of env_set.
        overrides: additional env vars; win over variant + env_set.
        inline_plugins: reserved; not yet implemented. Plugins come via the
            cloned repo for MVP.
        crucible_install: pip-installable reference to Crucible itself.
            Default: `git+https://github.com/eren23/parameter-golf_dev@main`
            but callers should pin to a specific commit for reproducibility.
        project_root: override for spec resolution root.

    Returns:
        `ExportResult` with paths and metadata.
    """
    spec = load_project_spec(project, project_root=project_root)
    raw_spec = _load_raw_spec(project, project_root=project_root)
    rt = get_runtime(runtime)

    if inline_plugins:
        raise NotebookExportError(
            "--inline-plugins is not yet implemented. "
            "For now, plugins are delivered via the cloned project repo."
        )

    ctx = RenderContext(
        spec=spec,
        raw_spec=raw_spec,
        runtime=rt,
        preset=preset,
        variant=variant,
        overrides=dict(overrides or {}),
        crucible_install=crucible_install or _default_crucible_install(),
        inline_plugins=inline_plugins,
    )
    source = assemble_source(ctx)

    dest = Path(out_path) if out_path else Path(f"{spec.name}.ipynb")
    dest.parent.mkdir(parents=True, exist_ok=True)

    suffix = dest.suffix.lower()
    if suffix == ".py":
        dest.write_text(source, encoding="utf-8")
        source_path: Path | None = dest
        size = dest.stat().st_size
    elif suffix == ".ipynb":
        # Keep a sibling .py for diffs, then convert.
        source_path = dest.with_suffix(".py")
        source_path.write_text(source, encoding="utf-8")
        _convert_to_ipynb(source_path, dest)
        size = dest.stat().st_size
    else:
        raise NotebookExportError(
            f"Unsupported output suffix {suffix!r}. Use .py or .ipynb."
        )

    cell_count = sum(1 for line in source.splitlines() if line.startswith("# %%"))

    return ExportResult(
        project=spec.name,
        runtime=rt.name,
        preset=preset,
        variant=variant,
        out_path=dest,
        source_path=source_path,
        cells=cell_count,
        size_bytes=size,
        open_in_colab_url=_colab_url(spec.repo, spec.branch, dest, spec.name),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_raw_spec(project: str, project_root: Path | None) -> dict[str, Any]:
    """Re-load the YAML to access fields not on the ProjectSpec dataclass (eval_suite, description)."""
    root = project_root or Path.cwd()
    local = root / ".crucible" / "projects" / f"{project}.yaml"
    if local.exists():
        return yaml.safe_load(local.read_text(encoding="utf-8")) or {}
    # Mirror `load_project_spec`'s hub/tap search for the raw YAML too.
    try:
        from crucible.core.hub import HubStore
        hub_dir = HubStore.resolve_hub_dir()
    except (ImportError, OSError):
        hub_dir = None
    if hub_dir:
        candidates: list[Path] = [hub_dir / "projects" / f"{project}.yaml"]
        taps_root = hub_dir / "taps"
        if taps_root.is_dir():
            candidates.extend(sorted(taps_root.glob(f"*/projects/{project}.yaml")))
        for cand in candidates:
            if cand.exists():
                return yaml.safe_load(cand.read_text(encoding="utf-8")) or {}
    return {}


def _default_crucible_install() -> str:
    """Default pip spec for Crucible itself when the caller doesn't pin."""
    return "git+https://github.com/eren23/parameter-golf_dev.git@main"


def _convert_to_ipynb(source_py: Path, dest_ipynb: Path) -> None:
    """Convert a jupytext .py source to .ipynb. Requires jupytext + nbformat."""
    try:
        import jupytext  # type: ignore[import-not-found]
    except ImportError as exc:
        raise NotebookExportError(
            "jupytext is not installed. Install with: pip install 'crucible-ml[notebook]'"
        ) from exc
    nb = jupytext.read(source_py)
    jupytext.write(nb, dest_ipynb, fmt="ipynb")


def _colab_url(repo: str, branch: str, dest: Path, project_name: str) -> str | None:
    """Heuristic: if repo is a github URL and dest under the repo, build an Open-in-Colab URL."""
    if not repo:
        return None
    if "github.com" not in repo:
        return None
    # Best-effort: assume the notebook will be committed under notebooks/<project>.ipynb
    base = repo.rstrip("/").removesuffix(".git")
    base = base.replace("https://github.com/", "").replace("git@github.com:", "")
    # Prefer the actual file path if it's inside cwd-relative notebooks/.
    rel = Path("notebooks") / dest.name
    return f"https://colab.research.google.com/github/{base}/blob/{branch}/{rel.as_posix()}"


__all__ = ["ExportResult", "NotebookExportError", "export_project"]
