"""Project-spec templating engine.

Turns a named template (e.g. ``lm``, ``diffusion``) plus a dict of user-supplied
variables into a resolved ``.crucible/projects/<name>.yaml`` file. The goal is
to make it trivial to bootstrap a new project without hand-editing a spec and
risking hardcoded personal identities.

Templates live in ``src/crucible/templates/projects/`` and use a simple
``${VAR}`` / ``${VAR:default}`` substitution syntax. Templates may also
``extends:`` another template for shared boilerplate.

Usage::

    from crucible.core.project_template import render_template
    text = render_template("lm", {"PROJECT_NAME": "my-lm", "REPO_URL": "..."})
    Path(".crucible/projects/my-lm.yaml").write_text(text)

The CLI wrapper (``crucible project new``) prompts for any required vars that
aren't in the environment or passed as ``--set KEY=VALUE``.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from crucible.core.errors import ProjectTemplateError

# ``${VAR}`` or ``${VAR:default}`` — variable names are uppercase snake_case.
# Default values can be empty but cannot contain ``}`` (keeps the parser simple).
_VAR_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::([^}]*))?\}")

# Maximum ``extends:`` depth — guards against cycles.
_MAX_EXTENDS_DEPTH = 5


def _templates_dir() -> Path:
    """Return the directory containing built-in project templates."""
    # ``core/`` is at ``src/crucible/core/``; templates at ``src/crucible/templates/projects/``.
    return Path(__file__).resolve().parent.parent / "templates" / "projects"


@dataclass(frozen=True)
class TemplateInfo:
    """Metadata about a discovered template."""
    name: str
    path: Path
    description: str


def list_templates() -> list[TemplateInfo]:
    """List every built-in project template discovered in the templates dir.

    Description is read from the first ``# description:`` comment line in the
    template file, or the YAML ``description:`` key if present.
    """
    templates_dir = _templates_dir()
    if not templates_dir.is_dir():
        return []
    results: list[TemplateInfo] = []
    for path in sorted(templates_dir.glob("*.yaml")):
        description = _extract_template_description(path)
        results.append(TemplateInfo(name=path.stem, path=path, description=description))
    return results


def _extract_template_description(path: Path) -> str:
    """Best-effort extract a one-line description from a template file."""
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("# description:"):
                return stripped.removeprefix("# description:").strip()
            if stripped.startswith("description:"):
                # Top-level YAML description — peel off key
                val = stripped.split(":", 1)[1].strip().strip("\"'")
                return val
    except OSError:
        pass
    return ""


def find_template(name: str) -> Path:
    """Resolve a template name to an absolute path.

    Raises ``ProjectTemplateError`` if the template does not exist.
    """
    templates_dir = _templates_dir()
    candidate = templates_dir / f"{name}.yaml"
    if not candidate.exists():
        available = [t.name for t in list_templates()]
        raise ProjectTemplateError(
            f"Unknown project template: {name!r}. "
            f"Available: {', '.join(available) if available else '(none)'}"
        )
    return candidate


def extract_vars(text: str) -> list[tuple[str, str | None]]:
    """Return every ``${VAR}`` reference in *text* as ``(name, default)`` pairs.

    The same variable may appear multiple times; this function preserves order
    and duplicates. Use :func:`required_vars` if you only want the names of
    variables with no default.
    """
    return [
        (m.group(1), m.group(2))
        for m in _VAR_PATTERN.finditer(text)
    ]


def required_vars(template_name: str) -> list[str]:
    """Return the de-duplicated list of variables that must be supplied.

    A variable is "required" if at least one of its occurrences has no default.
    Processes the template after ``extends:`` resolution so inherited requirements
    are also reported.
    """
    resolved_text = _resolve_extends_text(find_template(template_name).read_text(encoding="utf-8"))
    seen: dict[str, bool] = {}  # name -> has_required_occurrence
    for name, default in extract_vars(resolved_text):
        if name not in seen:
            seen[name] = default is None
        else:
            # Upgrade to required if any occurrence has no default
            seen[name] = seen[name] or (default is None)
    return [name for name, required in seen.items() if required]


# Maximum substitution passes — nested defaults like ``${A:${B}}`` need more
# than one pass because the regex default group stops at the first ``}``.
_MAX_SUBSTITUTE_PASSES = 10


def substitute(text: str, values: dict[str, str]) -> str:
    """Substitute ``${VAR}`` and ``${VAR:default}`` references in *text*.

    Supports nested defaults (e.g. ``${WANDB_PROJECT:${PROJECT_NAME}}``) by
    iterating until a fixed point is reached or a max pass count is hit.
    Raises ``ProjectTemplateError`` if a variable is missing and has no default.
    """
    def one_pass(s: str) -> str:
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            default = match.group(2)
            if name in values:
                return str(values[name])
            if default is not None:
                return default
            # Leave unresolved — detected after the loop ends
            return match.group(0)

        return _VAR_PATTERN.sub(replace, s)

    prev = None
    current = text
    for _ in range(_MAX_SUBSTITUTE_PASSES):
        if current == prev:
            break
        prev = current
        current = one_pass(current)

    # Any remaining ``${VAR}`` references are truly missing (no default, no value)
    missing_seen: list[str] = []
    for match in _VAR_PATTERN.finditer(current):
        name = match.group(1)
        default = match.group(2)
        if default is not None:
            # Shouldn't happen after fixed-point iteration, but guard anyway
            continue
        if name not in values and name not in missing_seen:
            missing_seen.append(name)
    if missing_seen:
        raise ProjectTemplateError(
            f"Missing required template variables: {', '.join(missing_seen)}"
        )
    return current


def _resolve_extends_text(raw: str, _depth: int = 0) -> str:
    """Recursively resolve ``extends:`` in a raw template yaml string.

    Returns the merged yaml as a string. Child keys override parent keys
    (dict merge at top level; lists are replaced, not concatenated).
    """
    if _depth > _MAX_EXTENDS_DEPTH:
        raise ProjectTemplateError(
            f"Template extends depth exceeded {_MAX_EXTENDS_DEPTH} — likely a cycle"
        )
    try:
        data = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        raise ProjectTemplateError(f"Template YAML parse error: {exc}") from exc
    if not isinstance(data, dict):
        return raw
    parent_name = data.pop("extends", None)
    if not parent_name:
        return raw
    if not isinstance(parent_name, str):
        raise ProjectTemplateError(
            f"extends: must be a string (the parent template name), got {type(parent_name).__name__}"
        )
    parent_path = find_template(parent_name)
    parent_resolved = _resolve_extends_text(
        parent_path.read_text(encoding="utf-8"), _depth=_depth + 1
    )
    parent_data = yaml.safe_load(parent_resolved) or {}
    if not isinstance(parent_data, dict):
        parent_data = {}
    merged = _merge_dicts(parent_data, data)
    return yaml.safe_dump(merged, sort_keys=False, default_flow_style=False)


def _merge_dicts(parent: dict, child: dict) -> dict:
    """Shallow-merge *child* into *parent*. Child wins on conflicts.

    Nested dicts are recursively merged. Lists are replaced (not concatenated),
    so a child template can blank out an inherited list.
    """
    result = dict(parent)
    for key, value in child.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def render_template(
    template_name: str,
    values: dict[str, str] | None = None,
    *,
    include_env: bool = True,
) -> str:
    """Render a template to a resolved yaml string.

    :param template_name: name of a built-in template (e.g. ``lm``, ``diffusion``)
    :param values: explicit variable overrides; highest precedence
    :param include_env: if True (default), also pull variables from ``os.environ``.
        Explicit *values* still win over env vars.
    :returns: fully-resolved yaml text ready to write to ``.crucible/projects/<name>.yaml``
    :raises ProjectTemplateError: if the template is unknown, has a cycle, or if
        required variables are missing.
    """
    template_path = find_template(template_name)
    raw = template_path.read_text(encoding="utf-8")
    resolved_extends = _resolve_extends_text(raw)

    merged_values: dict[str, str] = {}
    if include_env:
        # Only expose uppercase-snake_case keys from the environment to keep
        # the substitution surface predictable.
        for k, v in os.environ.items():
            if _VAR_PATTERN.fullmatch(f"${{{k}}}"):
                merged_values[k] = v
    if values:
        merged_values.update(values)

    return substitute(resolved_extends, merged_values)


def validate_spec(text: str) -> None:
    """Lightweight post-render validation.

    Ensures the rendered text parses as yaml, contains at minimum a ``name``
    key, and that no unresolved ``${...}`` markers remain.
    """
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ProjectTemplateError(f"Rendered spec is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ProjectTemplateError("Rendered spec must be a YAML mapping at the top level")
    if not data.get("name"):
        raise ProjectTemplateError("Rendered spec is missing required field: name")
    stragglers = _VAR_PATTERN.findall(text)
    if stragglers:
        names = sorted({m[0] for m in stragglers})
        raise ProjectTemplateError(
            f"Rendered spec still contains unresolved variables: {', '.join(names)}"
        )


def write_project_spec(
    template_name: str,
    project_name: str,
    values: dict[str, str] | None = None,
    *,
    project_root: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Render *template_name* and write it to ``.crucible/projects/<project_name>.yaml``.

    :param template_name: template to render
    :param project_name: file stem under ``.crucible/projects/``
    :param values: explicit variable overrides
    :param project_root: defaults to cwd
    :param overwrite: if False (default), refuses to clobber an existing spec
    :returns: the written path
    """
    merged_values = dict(values or {})
    merged_values.setdefault("PROJECT_NAME", project_name)

    text = render_template(template_name, merged_values)
    validate_spec(text)

    root = project_root or Path.cwd()
    target = root / ".crucible" / "projects" / f"{project_name}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise ProjectTemplateError(
            f"Project spec already exists: {target}. Pass overwrite=True to replace it."
        )
    target.write_text(text, encoding="utf-8")
    return target


def iter_required_missing(
    template_name: str,
    supplied: Iterable[str],
) -> list[str]:
    """Return required variables that are neither in *supplied* nor in os.environ."""
    supplied_set = set(supplied)
    missing: list[str] = []
    for var in required_vars(template_name):
        if var in supplied_set:
            continue
        if var in os.environ:
            continue
        # PROJECT_NAME is auto-populated by write_project_spec
        if var == "PROJECT_NAME":
            continue
        missing.append(var)
    return missing
