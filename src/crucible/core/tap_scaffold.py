"""Scaffold a fresh community tap directory.

``crucible tap init <path>`` creates a tap repo from a clean template:

  {path}/
  ├── README.md          one-paragraph what-it-is + how-to-add
  ├── LICENSE            user-chosen (default MIT)
  ├── tap.yaml           top-level manifest (name, version, license, ...)
  ├── .gitignore         excludes data/, checkpoints/, wandb/, .DS_Store
  ├── .github/
  │   └── workflows/
  │       └── lint.yaml  runs `crucible tap lint .` on PR
  └── optimizers/
      └── example_optimizer/
          ├── plugin.yaml          well-formed manifest
          ├── example_optimizer.py minimal builder stub
          └── README.md            describes what the plugin does

The scaffolded tap passes ``crucible tap lint .`` with 0 issues — it's
born clean. New plugins added on top must keep it that way (CI enforces
via the included workflow).

The scaffolding lives in this module (not in tap_commands.py) so it can be
unit-tested against a tmp dir without booting the CLI.
"""
from __future__ import annotations

import subprocess
from datetime import UTC
from pathlib import Path
from typing import Any

import yaml

from crucible.core.errors import TapError


def _yaml_str_double_quoted_representer(dumper: Any, data: str) -> Any:
    """Render strings with double quotes — keeps version strings unambiguous."""
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')


class _QuotedStringDumper(yaml.SafeDumper):
    pass


_QuotedStringDumper.add_representer(str, _yaml_str_double_quoted_representer)


def _write_yaml_with_quoted_strings(path: Path, data: dict[str, Any]) -> None:
    """Write a YAML file where every string is double-quoted.

    Used for tap.yaml and plugin.yaml in the scaffold so the L009
    'version: unquoted' lint check doesn't fire on freshly-scaffolded
    files. Versions need quoting because '0.1.0' is ambiguous in YAML.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.dump(
        data,
        Dumper=_QuotedStringDumper,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )
    path.write_text(text, encoding="utf-8")

# ---------------------------------------------------------------------------
# Template strings
# ---------------------------------------------------------------------------

_README_TEMPLATE = """\
# {name}

{description}

## What's in here

Plugins organized by type — each plugin lives at `{{type}}/{{name}}/` with a
`plugin.yaml` manifest, a single-file `.py` (or a directory bundle for
launchers / evaluations / domain_specs), and a short `README.md` describing
what it does.

This repo's `tap.yaml` declares its identity, version, license, and
compatibility range against Crucible itself.

## Add this tap to your hub

```bash
crucible tap add https://github.com/<your-org>/{name}
crucible tap search <keyword>
crucible tap install <plugin_name>
```

## Add a plugin

```bash
# from inside this tap repo
mkdir -p optimizers/my_optimizer
$EDITOR optimizers/my_optimizer/plugin.yaml
$EDITOR optimizers/my_optimizer/my_optimizer.py
crucible tap lint .          # repo-quality checks (READMEs, no cruft, ...)
crucible tap validate .      # per-plugin manifest schema
git add . && git commit -m "Add optimizers/my_optimizer"
```

## Quality bar

PRs are CI-gated by `crucible tap lint .` (see `.github/workflows/lint.yaml`).
The lint catches:

- Missing per-plugin README.md
- Cruft directories (data/, checkpoints/, wandb/, .DS_Store)
- Files over 1 MB (move to HuggingFace instead)
- Plugin folder name != manifest `name:` field
- Multi-line `description:` blocks
- Unquoted version strings
- Python syntax errors in plugin files
- Per-plugin manifest schema violations

## License

{license}
"""

_LICENSE_MIT = """\
MIT License

Copyright (c) {year} {author}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

_LICENSE_APACHE = """\
Copyright {year} {author}

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

_LICENSE_BSD3 = """\
BSD 3-Clause License

Copyright (c) {year}, {author}
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED.
"""

# Map license-id → (filename suffix, body template)
_LICENSE_BODIES: dict[str, str] = {
    "MIT": _LICENSE_MIT,
    "Apache-2.0": _LICENSE_APACHE,
    "BSD-3-Clause": _LICENSE_BSD3,
}

_GITIGNORE = """\
# Cruft that shouldn't live in a tap
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.coverage
htmlcov/

# Editor / OS noise
.DS_Store
*.swp
.vscode/
.idea/

# Large artifacts — keep these on HuggingFace instead
data/
checkpoints/
wandb/
*.bin
*.safetensors
*.ckpt
*.pt
*.pth
"""

_CI_WORKFLOW = """\
name: lint

on:
  pull_request:
  push:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install Crucible (with extras)
        run: |
          python -m pip install --upgrade pip
          # Replace with the right install line for your environment.
          # If Crucible is published on PyPI:
          #   pip install crucible-ml
          # Or from source:
          #   pip install git+https://github.com/<org>/crucible
          pip install crucible-ml || pip install git+https://github.com/eren23/crucible@main
      - name: tap lint
        run: crucible tap lint . --warnings-as-errors
      - name: tap validate
        run: crucible tap validate . --warnings-as-errors
"""

_EXAMPLE_PLUGIN_PY = '''\
"""example_optimizer — a minimal Crucible optimizer plugin.

This is the scaffold ``crucible tap init`` drops in. Replace the body
with your real optimizer; keep the ``build`` function signature so
Crucible's PluginRegistry can resolve it.
"""
from __future__ import annotations

from typing import Any

import torch


def build(params, *, lr: float = 1e-3, **kwargs: Any) -> torch.optim.Optimizer:
    """Return an optimizer instance.

    This stub returns plain SGD. Replace with your custom update rule.
    """
    return torch.optim.SGD(params, lr=lr)
'''

_EXAMPLE_PLUGIN_MANIFEST: dict[str, Any] = {
    "name": "example_optimizer",
    "type": "optimizers",
    "version": "0.1.0",
    "description": "Example optimizer scaffold from crucible tap init",
    "author": "",  # filled in at scaffold time
    "tags": ["example", "scaffold"],
    "crucible_compat": ">=0.2,<0.3",
    "dependencies": ["torch>=2.0"],
    "parameters": {
        "lr": {"type": "float", "default": 1e-3, "description": "learning rate"},
    },
}

_EXAMPLE_PLUGIN_README = """\
# example_optimizer

Scaffold plugin generated by `crucible tap init`. Replace the body of
`example_optimizer.py` with your real optimizer and update the manifest
metadata (description, author, tags) before publishing.

## Usage

```bash
crucible tap install example_optimizer
# then in a project: OPTIMIZER=example_optimizer crucible run experiment
```

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `LR` | `1e-3` | learning rate |
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scaffold_tap(
    target: Path,
    *,
    name: str,
    author: str = "",
    license_id: str = "MIT",
    description: str = "",
    init_git: bool = True,
) -> dict[str, Any]:
    """Write a fresh community-tap scaffold under *target*.

    Returns a dict describing what was written. Raises :class:`TapError`
    if *target* is already a non-empty directory (we don't clobber).
    """
    from datetime import datetime

    if not description:
        description = f"{name} — Crucible community tap"

    target = Path(target).expanduser().resolve()
    if target.exists():
        if not target.is_dir():
            raise TapError(f"{target} exists and is not a directory")
        if any(target.iterdir()):
            raise TapError(
                f"{target} is not empty — refusing to scaffold over existing files. "
                f"Pick an empty directory or remove the contents first."
            )
    else:
        target.mkdir(parents=True)

    license_body = _LICENSE_BODIES.get(license_id, _LICENSE_MIT)
    year = datetime.now(UTC).year
    author_display = author or "the tap authors"

    files_written: list[str] = []

    def _write(rel: str, content: str) -> None:
        path = target / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        files_written.append(rel)

    _write("README.md", _README_TEMPLATE.format(
        name=name, description=description, license=license_id,
    ))
    _write("LICENSE", license_body.format(year=year, author=author_display))
    _write(".gitignore", _GITIGNORE)
    _write(".github/workflows/lint.yaml", _CI_WORKFLOW)

    # tap.yaml top-level manifest (Phase A.3). All string values are
    # double-quoted so the L009 'unquoted version' lint passes.
    tap_yaml = {
        "name": name,
        "description": description,
        "version": "0.1.0",
        "author": author or "TODO",
        "license": license_id,
        "crucible_compat": ">=0.2,<0.3",
        "homepage": "",
        "maintainer_contact": "",
    }
    _write_yaml_with_quoted_strings(target / "tap.yaml", tap_yaml)
    files_written.append("tap.yaml")

    # Example plugin
    example_dir_rel = "optimizers/example_optimizer"
    example_dir = target / example_dir_rel
    example_dir.mkdir(parents=True, exist_ok=True)

    plugin_manifest = dict(_EXAMPLE_PLUGIN_MANIFEST)
    plugin_manifest["author"] = author or "TODO"
    _write_yaml_with_quoted_strings(example_dir / "plugin.yaml", plugin_manifest)
    files_written.append(f"{example_dir_rel}/plugin.yaml")

    _write(f"{example_dir_rel}/example_optimizer.py", _EXAMPLE_PLUGIN_PY)
    _write(f"{example_dir_rel}/README.md", _EXAMPLE_PLUGIN_README)

    # Init git + first commit (optional)
    if init_git:
        try:
            subprocess.run(
                ["git", "init", "-q"],
                cwd=str(target),
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "add", "."],
                cwd=str(target),
                check=True,
                capture_output=True,
                text=True,
            )
            # Don't require commit signing or hooks; pre-commit hooks
            # aren't installed in a fresh scaffold anyway.
            subprocess.run(
                [
                    "git", "commit",
                    "-q",
                    "-m", f"Initial scaffold for {name} via 'crucible tap init'",
                ],
                cwd=str(target),
                check=True,
                capture_output=True,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            # Non-fatal — the scaffold is still useful without git.
            return {
                "path": str(target),
                "files_written": len(files_written),
                "example_plugin_dir": str(example_dir),
                "git_initialized": False,
                "git_error": str(exc),
                "next_steps": _next_steps(target, name, init_git=False),
            }

    return {
        "path": str(target),
        "files_written": len(files_written),
        "example_plugin_dir": str(example_dir),
        "git_initialized": init_git,
        "next_steps": _next_steps(target, name, init_git=init_git),
    }


def _next_steps(target: Path, name: str, *, init_git: bool) -> list[str]:
    steps = [
        f"cd {target}",
        "crucible tap lint . --warnings-as-errors  # confirms the scaffold is clean",
    ]
    if init_git:
        steps += [
            "git remote add origin https://github.com/<your-org>/" + name,
            "git push -u origin main",
        ]
    else:
        steps += [
            "git init && git add . && git commit -m 'initial scaffold'",
        ]
    return steps
