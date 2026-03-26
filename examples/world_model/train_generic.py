"""Local training entrypoint for the world-model example.

Registers the example model and data adapter, then delegates to Crucible's
generic backend. This file is referenced by the example project's
``crucible.yaml`` so fleet runs and local runs use the same startup path.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_import_paths() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    for path in (repo_root, repo_root / "src"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def main() -> None:
    _bootstrap_import_paths()

    from data_adapter import register as register_adapter
    from model import register as register_model
    from crucible.training.generic_backend import run_generic_training

    register_model()
    register_adapter()
    run_generic_training()


if __name__ == "__main__":
    main()
