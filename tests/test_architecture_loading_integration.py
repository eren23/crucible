"""Integration tests for architecture loading precedence."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_mirrored_global_and_local_precedence(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    project_root = tmp_path / "project"
    project_root.mkdir()

    (project_root / "crucible.yaml").write_text("name: arch-loading-test\n", encoding="utf-8")

    mirror_dir = project_root / ".crucible" / "architectures" / "_hub"
    mirror_dir.mkdir(parents=True)
    (mirror_dir / "shared_arch.py").write_text(
        "from crucible.models.registry import register_model\n"
        "def _build(args):\n"
        "    return 'global'\n"
        "register_model('shared_arch', _build)\n",
        encoding="utf-8",
    )

    local_dir = project_root / ".crucible" / "architectures"
    (local_dir / "shared_arch.py").write_text(
        "from crucible.models.registry import register_model\n"
        "def _build(args):\n"
        "    return 'local'\n"
        "register_model('shared_arch', _build)\n",
        encoding="utf-8",
    )
    (mirror_dir / "global_only.py").write_text(
        "from crucible.models.registry import register_model\n"
        "def _build(args):\n"
        "    return 'global-only'\n"
        "register_model('global_only', _build)\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import json
                import types
                from crucible.models.registry import build_model, list_families
                args = types.SimpleNamespace(model_family="shared_arch")
                global_args = types.SimpleNamespace(model_family="global_only")
                print(json.dumps({
                    "families": list_families(),
                    "shared_arch_result": build_model(args),
                    "global_only_result": build_model(global_args),
                }))
                """
            ),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(src_root)},
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = proc.stdout.strip().splitlines()[-1]
    data = __import__("json").loads(payload)
    assert "shared_arch" in data["families"]
    assert "global_only" in data["families"]
    assert data["shared_arch_result"] == "local"
    assert data["global_only_result"] == "global-only"
