"""Plugin manifest schema and validation.

Every plugin in a community tap (and every local plugin under
``.crucible/plugins/``) must carry a ``plugin.yaml`` manifest that
follows this schema. The validator is intentionally lightweight — no
jsonschema dependency — because manifests are small and the checks are
clear enough to express as Python.

Required fields:
  - ``name``: unique identifier, matches ``[a-zA-Z_][a-zA-Z0-9_-]*``
  - ``type``: one of the known plugin categories
  - ``version``: semver-ish string (M.m.p, optionally with a -suffix)
  - ``description``: one-line human-readable description

Optional but strongly recommended:
  - ``author``: maintainer handle / email
  - ``tags``: list of short classification tags
  - ``crucible_compat``: a version range against Crucible itself
    (e.g. ``">=0.2,<0.3"``). If absent, the plugin is assumed to work
    with any Crucible version — which is the current de-facto state
    but should be tightened over time.
  - ``dependencies``: list of ``{"name": str, "version": str}`` dicts
    or plain strings ("torch>=2.1"). Used by ``hub install`` to warn
    on missing deps and (future) resolve inter-plugin requirements.
  - ``config``: free-form dict of default env vars the plugin sets
  - ``parameters``: free-form dict documenting runtime env var knobs

The validator returns a list of ``ValidationIssue`` records — empty
means the manifest passes. Issues are classified as ``error`` (must
fix) or ``warning`` (should fix but not blocking).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from crucible.core.errors import PluginError

# Every plugin category currently recognized by Crucible's PluginRegistry.
# Keeping this in sync with core/plugin_registry.py is a soft contract;
# unknown types get a warning (not an error) so new categories added by
# the community don't immediately break validation.
KNOWN_PLUGIN_TYPES: frozenset[str] = frozenset({
    "architectures",
    "callbacks",
    "optimizers",
    "schedulers",
    "data_adapters",
    "data_sources",
    "objectives",
    "loggers",
    "providers",
    "block_types",
    "stack_patterns",
    "augmentations",
    "activations",
    "launchers",  # tap-specific category, not in core registry
})

_NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_-]*$")

# Semver-ish: M.m.p with optional -suffix. Accepts "0.1.0", "1.0.0-rc1",
# "0.2.1-alpha", etc. Deliberately loose.
_VERSION_PATTERN = re.compile(
    r"^\d+\.\d+\.\d+(?:[-+][a-zA-Z0-9][a-zA-Z0-9.-]*)?$"
)


@dataclass
class ValidationIssue:
    """One problem found while validating a manifest."""
    severity: str  # "error" | "warning"
    field: str
    message: str


@dataclass
class ValidationResult:
    """Aggregate result of validating one plugin manifest."""
    path: Path
    ok: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]


def validate_manifest_dict(data: dict[str, Any]) -> list[ValidationIssue]:
    """Validate an already-loaded plugin.yaml dict.

    Returns a list of issues. Empty list = pass. Callers that want a
    single-shot pass/fail check can use ``validate_manifest_file``.
    """
    issues: list[ValidationIssue] = []
    if not isinstance(data, dict):
        return [ValidationIssue("error", "<root>", "manifest must be a YAML mapping")]

    # ── required fields ────────────────────────────────────────────────
    name = data.get("name")
    if not name:
        issues.append(ValidationIssue("error", "name", "missing or empty"))
    elif not isinstance(name, str):
        issues.append(ValidationIssue("error", "name", f"must be a string, got {type(name).__name__}"))
    elif not _NAME_PATTERN.match(name):
        issues.append(ValidationIssue(
            "error", "name",
            f"{name!r} does not match [a-zA-Z_][a-zA-Z0-9_-]*"
        ))

    plugin_type = data.get("type")
    if not plugin_type:
        issues.append(ValidationIssue("error", "type", "missing"))
    elif not isinstance(plugin_type, str):
        issues.append(ValidationIssue("error", "type", f"must be a string, got {type(plugin_type).__name__}"))
    elif plugin_type not in KNOWN_PLUGIN_TYPES:
        # Unknown type is a warning — community taps can introduce new
        # categories, and we don't want to hard-break on them.
        issues.append(ValidationIssue(
            "warning", "type",
            f"{plugin_type!r} is not a known plugin type "
            f"(expected one of: {', '.join(sorted(KNOWN_PLUGIN_TYPES))})"
        ))

    version = data.get("version")
    if version is None:
        issues.append(ValidationIssue("error", "version", "missing"))
    elif not isinstance(version, (str, int, float)):
        issues.append(ValidationIssue("error", "version", f"must be a string, got {type(version).__name__}"))
    else:
        version_str = str(version)
        if not _VERSION_PATTERN.match(version_str):
            issues.append(ValidationIssue(
                "warning", "version",
                f"{version_str!r} does not look like semver (expected M.m.p)"
            ))

    description = data.get("description")
    if not description:
        issues.append(ValidationIssue("error", "description", "missing or empty"))
    elif not isinstance(description, str):
        issues.append(ValidationIssue("error", "description", f"must be a string, got {type(description).__name__}"))
    elif len(description) > 500:
        issues.append(ValidationIssue(
            "warning", "description",
            f"is {len(description)} chars; prefer a concise one-line summary "
            f"(move detail to a separate README.md)"
        ))

    # ── optional-but-recommended fields ────────────────────────────────
    author = data.get("author")
    if author is None:
        issues.append(ValidationIssue("warning", "author", "missing — recommended"))
    elif not isinstance(author, str):
        issues.append(ValidationIssue("error", "author", f"must be a string, got {type(author).__name__}"))

    tags = data.get("tags")
    if tags is None:
        issues.append(ValidationIssue("warning", "tags", "missing — recommended"))
    elif not isinstance(tags, list):
        issues.append(ValidationIssue("error", "tags", f"must be a list, got {type(tags).__name__}"))
    else:
        for i, tag in enumerate(tags):
            if not isinstance(tag, str):
                issues.append(ValidationIssue(
                    "error", f"tags[{i}]",
                    f"must be a string, got {type(tag).__name__}"
                ))

    compat = data.get("crucible_compat")
    if compat is None:
        issues.append(ValidationIssue(
            "warning", "crucible_compat",
            "missing — recommended (e.g. '>=0.2,<0.3')"
        ))
    elif not isinstance(compat, str):
        issues.append(ValidationIssue(
            "error", "crucible_compat",
            f"must be a string like '>=0.2,<0.3', got {type(compat).__name__}"
        ))

    deps = data.get("dependencies")
    if deps is None:
        issues.append(ValidationIssue(
            "warning", "dependencies",
            "missing — declare Python deps your plugin needs"
        ))
    elif not isinstance(deps, list):
        issues.append(ValidationIssue(
            "error", "dependencies",
            f"must be a list, got {type(deps).__name__}"
        ))
    else:
        for i, dep in enumerate(deps):
            if isinstance(dep, str):
                if not dep.strip():
                    issues.append(ValidationIssue(
                        "error", f"dependencies[{i}]", "empty string"
                    ))
            elif isinstance(dep, dict):
                if "name" not in dep:
                    issues.append(ValidationIssue(
                        "error", f"dependencies[{i}]",
                        "dict entries must have a 'name' field"
                    ))
            else:
                issues.append(ValidationIssue(
                    "error", f"dependencies[{i}]",
                    f"must be a string or dict, got {type(dep).__name__}"
                ))

    return issues


def validate_manifest_file(path: Path) -> ValidationResult:
    """Validate the plugin.yaml at *path*, returning a ValidationResult."""
    if not path.exists():
        return ValidationResult(
            path=path,
            ok=False,
            issues=[ValidationIssue("error", "<file>", f"does not exist: {path}")],
        )
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        return ValidationResult(
            path=path,
            ok=False,
            issues=[ValidationIssue("error", "<yaml>", f"invalid YAML: {exc}")],
        )
    issues = validate_manifest_dict(raw)
    ok = not any(i.severity == "error" for i in issues)
    return ValidationResult(path=path, ok=ok, issues=issues)


def validate_tap_directory(root: Path) -> list[ValidationResult]:
    """Validate every ``plugin.yaml`` file discovered under *root*.

    Walks the tap repo recursively, skipping common cruft dirs
    (``.git``, ``wandb``, ``__pycache__``, ``checkpoints``, ``data``,
    ``_manuscript``) and the top-level ``findings/`` directory (those
    are research artifacts, not plugins).
    """
    if not root.exists():
        raise PluginError(f"Tap directory does not exist: {root}")
    if not root.is_dir():
        raise PluginError(f"Tap path is not a directory: {root}")

    skip_dirs = {
        ".git",
        "wandb",
        "__pycache__",
        "checkpoints",
        "data",
        "_manuscript",
        "findings",
    }
    results: list[ValidationResult] = []
    for manifest_path in sorted(root.rglob("plugin.yaml")):
        # Skip anything inside a blocked directory
        rel_parts = manifest_path.relative_to(root).parts
        if any(part in skip_dirs for part in rel_parts):
            continue
        results.append(validate_manifest_file(manifest_path))
    return results
