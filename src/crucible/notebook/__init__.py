"""Colab/Jupyter notebook exporter for Crucible project specs.

`crucible notebook export --project X --runtime colab-h100 --out path.ipynb`
produces a standalone runnable notebook that clones the project repo, installs
deps, runs the spec's `train` command with merged env, and executes the
`eval_suite` block.

Public API:
    export_project(project, runtime, preset, out_path, variant=None,
                   overrides=None, inline_plugins=False) -> ExportResult

The exporter is in core because notebook generation is build infrastructure
that every project/tap benefits from, not domain-specific experiment code.
"""
from __future__ import annotations

from crucible.notebook.exporter import ExportResult, export_project

__all__ = ["ExportResult", "export_project"]
