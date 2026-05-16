"""Import a Karpathy-style ``autoresearch`` source into a Crucible project spec.

Karpathy's ``autoresearch`` (released 2026-03-07, ~66k GitHub stars) is a
single-GPU autonomous ML research loop that edits a ``train.py`` between
5-minute trials and keeps/rolls-back based on a metric. Its hot loop is
in-process; it does not have multi-pod fleet, judge separation, or
cross-project memory.

This module is Crucible's bridge: it takes an autoresearch source
directory and emits a ``.crucible/projects/<name>.yaml`` plus the
sidecar files needed to run that ``train.py`` under Crucible's
env-var/stdout training contract on a multi-pod fleet, driven by the
autonomous-loop session from Phase 1.1.

The adapter does NOT translate ``train.py`` itself — Crucible's training
contract only requires the script to read env vars and print parseable
output. autoresearch's ``train.py`` already takes its hyperparameters
from a config object near the top of the file, so wrapping it as a
Crucible training script is a small env-var layer at most; users do
that themselves on first run.

Source layout recognised:
- ``train.py`` (required) — the training script
- ``program.md`` (optional) — research program description; kept as
  sidecar and pointed at by the emitted spec
- Any other ``*.py`` files in the same directory — treated as helper
  modules that ``train.py`` may import; bundled into ``local_files``
  so fleet sync brings them along
- ``requirements.txt`` (optional) — parsed line-by-line into the spec's
  ``install`` list (comments and blank lines stripped)

Output:
- ``.crucible/projects/<name>.yaml`` — Crucible ProjectSpec with the
  source files listed in ``local_files`` and ``train: "python train.py"``
- ``.crucible/projects/<name>.md`` — copy of ``program.md`` if present,
  for the orchestrator's hypothesis prompt context

The emitted yaml is validated by round-tripping through
:func:`crucible.core.config.load_project_spec`; if the validation fails
the importer raises before leaving any partial state on disk.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from crucible.core.errors import CrucibleError
from crucible.core.log import log_info, log_warn

_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
_DEFAULT_WORKSPACE_PREFIX = "/workspace"


@dataclass
class AutoresearchSource:
    """Parsed pieces of an autoresearch source directory."""

    source_dir: Path
    train_path: Path
    program_path: Path | None
    sibling_py: list[Path]
    requirements: list[str]

    @property
    def local_files(self) -> list[Path]:
        """All source files that need to ride to the fleet pod."""
        files: list[Path] = [self.train_path]
        if self.program_path is not None:
            files.append(self.program_path)
        files.extend(self.sibling_py)
        req_path = self.source_dir / "requirements.txt"
        if req_path.exists():
            files.append(req_path)
        return files


def sanitize_name(raw: str) -> str:
    """Coerce *raw* into a Crucible-spec-compatible name.

    Project specs are addressed by file stem; this needs to be a
    filesystem-safe identifier. Lowercases, replaces whitespace and
    dots with underscores, strips leading non-alphabetics.
    """
    s = raw.strip().lower()
    s = re.sub(r"[\s.]+", "_", s)
    s = re.sub(r"[^a-z0-9_-]", "", s)
    s = re.sub(r"^[^a-z]+", "", s)
    if not s:
        raise CrucibleError(
            f"Cannot derive a valid project name from {raw!r} — pass --name explicitly."
        )
    return s


def parse_autoresearch_source(source_dir: Path) -> AutoresearchSource:
    """Scan *source_dir* for an autoresearch layout. Raises if ``train.py`` is missing."""
    source_dir = Path(source_dir).resolve()
    if not source_dir.is_dir():
        raise CrucibleError(f"autoresearch source directory not found: {source_dir}")

    train_path = source_dir / "train.py"
    if not train_path.is_file():
        raise CrucibleError(
            f"autoresearch import expects a 'train.py' at {source_dir} — none found. "
            "If your training entrypoint has a different name, rename it to train.py "
            "before importing, or write a Crucible project spec by hand instead."
        )

    program_path: Path | None = source_dir / "program.md"
    if not program_path.is_file():
        log_warn(
            f"autoresearch import: no program.md at {source_dir}; importer will proceed "
            "but the orchestrator's hypothesis prompts will lack the program description."
        )
        program_path = None

    sibling_py: list[Path] = sorted(
        p for p in source_dir.glob("*.py")
        if p.is_file() and p.name != "train.py"
    )

    req_path = source_dir / "requirements.txt"
    requirements: list[str] = []
    if req_path.is_file():
        for line in req_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            requirements.append(stripped)

    return AutoresearchSource(
        source_dir=source_dir,
        train_path=train_path,
        program_path=program_path,
        sibling_py=sibling_py,
        requirements=requirements,
    )


def _project_spec_dict(
    parsed: AutoresearchSource,
    name: str,
    *,
    workspace: str = "",
    python: str = "",
    primary_metric: str = "val_loss",
    direction: str = "minimize",
) -> dict[str, Any]:
    """Build the ProjectSpec dict that gets serialized as YAML."""
    if not workspace:
        workspace = f"{_DEFAULT_WORKSPACE_PREFIX}/{name}"
    return {
        "name": name,
        # repo: empty — we ship via local_files instead of git clone.
        "repo": "",
        "branch": "main",
        "shallow": True,
        "workspace": workspace,
        "python": python,
        "install": parsed.requirements,
        "local_files": [str(p) for p in parsed.local_files],
        "system_packages": [],
        "setup": [],
        "train": "python train.py",
        "timeout": 0,
        "env_forward": ["WANDB_API_KEY", "WANDB_ENTITY"],
        "env_set": {},
        "metrics": {
            "source": "stdout",
            "primary": primary_metric,
            "direction": direction,
        },
    }


def import_autoresearch(
    source_dir: Path | str,
    project_root: Path | str | None = None,
    *,
    name: str = "",
    force: bool = False,
    primary_metric: str = "val_loss",
    direction: str = "minimize",
) -> dict[str, Any]:
    """Import an autoresearch source directory as a Crucible project spec.

    Returns a dict describing what was emitted:
    ``{name, spec_path, program_path, source_files, requirements, validated}``.

    On validation failure (the emitted yaml does not round-trip through
    :func:`load_project_spec`), the partial files are removed and a
    :class:`CrucibleError` is raised.
    """
    parsed = parse_autoresearch_source(Path(source_dir))
    root = Path(project_root or Path.cwd()).resolve()
    project_name = sanitize_name(name) if name else sanitize_name(parsed.source_dir.name)
    if not _NAME_RE.match(project_name):
        raise CrucibleError(
            f"Sanitized project name {project_name!r} is invalid — must match {_NAME_RE.pattern}"
        )

    spec_dir = root / ".crucible" / "projects"
    spec_path = spec_dir / f"{project_name}.yaml"
    if spec_path.exists() and not force:
        raise CrucibleError(
            f"Crucible project spec already exists at {spec_path}. "
            "Pass --force to overwrite, or pick a different name with --name."
        )
    spec_dir.mkdir(parents=True, exist_ok=True)

    program_dest: Path | None = None
    if parsed.program_path is not None:
        program_dest = spec_dir / f"{project_name}.md"
        shutil.copyfile(parsed.program_path, program_dest)

    spec_dict = _project_spec_dict(
        parsed,
        project_name,
        primary_metric=primary_metric,
        direction=direction,
    )
    spec_path.write_text(yaml.safe_dump(spec_dict, sort_keys=False), encoding="utf-8")

    # Validate by round-tripping through load_project_spec. If that fails,
    # roll back the partial write so the user isn't left with broken state.
    try:
        from crucible.core.config import load_project_spec
        load_project_spec(project_name, root)
    except Exception as exc:
        spec_path.unlink(missing_ok=True)
        if program_dest is not None:
            program_dest.unlink(missing_ok=True)
        raise CrucibleError(
            f"Emitted project spec at {spec_path} failed validation: {exc}. "
            "The partial files have been removed."
        ) from exc

    log_info(
        f"autoresearch import: emitted {project_name!r} project spec at {spec_path} "
        f"(local_files={len(parsed.local_files)}, requirements={len(parsed.requirements)})"
    )

    return {
        "name": project_name,
        "spec_path": str(spec_path),
        "program_path": str(program_dest) if program_dest else None,
        "source_files": [str(p) for p in parsed.local_files],
        "requirements": parsed.requirements,
        "validated": True,
    }


__all__ = [
    "AutoresearchSource",
    "import_autoresearch",
    "parse_autoresearch_source",
    "sanitize_name",
]
