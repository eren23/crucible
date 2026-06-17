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
    "launchers",     # tap-specific category, not in core registry
    "evaluations",   # tap-specific: per-project eval script bundles
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
class ManifestValidationResult:
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


#: Backward-compatible alias -- external code may still import the old name.
ValidationResult = ManifestValidationResult


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


def validate_manifest_file(path: Path) -> ManifestValidationResult:
    """Validate the plugin.yaml at *path*, returning a ManifestValidationResult."""
    if not path.exists():
        return ManifestValidationResult(
            path=path,
            ok=False,
            issues=[ValidationIssue("error", "<file>", f"does not exist: {path}")],
        )
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        return ManifestValidationResult(
            path=path,
            ok=False,
            issues=[ValidationIssue("error", "<yaml>", f"invalid YAML: {exc}")],
        )
    issues = validate_manifest_dict(raw)
    ok = not any(i.severity == "error" for i in issues)
    return ManifestValidationResult(path=path, ok=ok, issues=issues)


#: Required field set for a top-level ``tap.yaml`` manifest. A tap.yaml is
#: optional today (filesystem walk still works without it), but if one
#: exists it must declare these.
_TAP_REQUIRED_FIELDS = ("name", "description", "version")

#: Optional-but-recommended fields. Missing → warning.
_TAP_RECOMMENDED_FIELDS = ("author", "license", "crucible_compat")


def validate_tap_manifest_dict(data: dict[str, Any]) -> list[ValidationIssue]:
    """Validate a loaded ``tap.yaml`` (top-level tap manifest) dict.

    A tap manifest describes the tap repo itself — name, version, license,
    upstream URL, maintainer — distinct from the per-plugin ``plugin.yaml``
    manifests living under ``{type}/{name}/``. Returns a list of issues.
    """
    issues: list[ValidationIssue] = []
    if not isinstance(data, dict):
        return [ValidationIssue("error", "<root>", "tap.yaml must be a YAML mapping")]

    # ── required ──────────────────────────────────────────────────────
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

    description = data.get("description")
    if not description:
        issues.append(ValidationIssue("error", "description", "missing or empty"))
    elif not isinstance(description, str):
        issues.append(ValidationIssue(
            "error", "description", f"must be a string, got {type(description).__name__}"
        ))
    elif "\n" in description.strip():
        # tap.yaml descriptions follow the same one-liner rule as plugin.yaml.
        issues.append(ValidationIssue(
            "warning", "description",
            "should be a single line (move detail to README.md)"
        ))

    version = data.get("version")
    if version is None:
        issues.append(ValidationIssue("error", "version", "missing"))
    elif not isinstance(version, (str, int, float)):
        issues.append(ValidationIssue(
            "error", "version", f"must be a string, got {type(version).__name__}"
        ))
    elif not _VERSION_PATTERN.match(str(version)):
        issues.append(ValidationIssue(
            "warning", "version",
            f"{version!r} does not look like semver (expected M.m.p)"
        ))

    # ── recommended (warnings only) ────────────────────────────────────
    for field_name in _TAP_RECOMMENDED_FIELDS:
        val = data.get(field_name)
        if val is None:
            issues.append(ValidationIssue(
                "warning", field_name, "missing — recommended"
            ))
        elif not isinstance(val, str):
            issues.append(ValidationIssue(
                "error", field_name,
                f"must be a string, got {type(val).__name__}",
            ))

    # ── purely optional fields ────────────────────────────────────────
    for opt in ("homepage", "maintainer_contact"):
        val = data.get(opt)
        if val is not None and not isinstance(val, str):
            issues.append(ValidationIssue(
                "error", opt, f"must be a string, got {type(val).__name__}"
            ))

    return issues


def validate_tap_manifest_file(path: Path) -> ManifestValidationResult:
    """Validate the ``tap.yaml`` at *path*, returning a ManifestValidationResult.

    Missing file is reported as a *warning* (not an error) because tap.yaml
    is optional — older taps may not have one yet.
    """
    if not path.exists():
        return ManifestValidationResult(
            path=path,
            ok=True,
            issues=[ValidationIssue(
                "warning", "<file>",
                f"tap.yaml missing at {path} — recommended at tap root",
            )],
        )
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        return ManifestValidationResult(
            path=path,
            ok=False,
            issues=[ValidationIssue("error", "<yaml>", f"invalid YAML: {exc}")],
        )
    issues = validate_tap_manifest_dict(raw)
    ok = not any(i.severity == "error" for i in issues)
    return ManifestValidationResult(path=path, ok=ok, issues=issues)


# ---------------------------------------------------------------------------
# crucible_compat version-range parsing & matching
# ---------------------------------------------------------------------------

# Each token is one of: >=X.Y, >X.Y, <=X.Y, <X.Y, ==X.Y, =X.Y. Tokens are
# separated by commas. Numeric versions may have any number of dot-segments.
_RANGE_TOKEN_RE = re.compile(r"^\s*(>=|<=|==|=|>|<)\s*([0-9]+(?:\.[0-9]+)*(?:[-+][a-zA-Z0-9.-]+)?)\s*$")


def _parse_version(version: str, *, pad_to: int = 3) -> tuple[int, ...]:
    """Parse a version string into a comparable integer tuple.

    Ignores pre-release / build suffixes for comparison purposes (matches
    the loose semver intent of crucible_compat: '>=0.2,<0.3' should match
    '0.2.5-dev'). Zero-pads to *pad_to* segments so '0.2' and '0.2.0'
    compare equal (otherwise Python tuple comparison treats (0,2) < (0,2,0)
    which is wrong for semver).
    """
    base = re.split(r"[-+]", str(version), maxsplit=1)[0]
    parts: list[int] = []
    for seg in base.split("."):
        try:
            parts.append(int(seg))
        except ValueError:
            parts.append(0)
    while len(parts) < pad_to:
        parts.append(0)
    return tuple(parts)


def parse_version_range(range_spec: str) -> list[tuple[str, tuple[int, ...]]]:
    """Parse a comma-separated range spec like ``'>=0.2,<0.3'``.

    Returns a list of ``(operator, version_tuple)`` constraints. Raises
    :class:`PluginError` on malformed tokens so a bad ``crucible_compat``
    field doesn't silently disable enforcement.
    """
    if not range_spec or not isinstance(range_spec, str):
        return []
    constraints: list[tuple[str, tuple[int, ...]]] = []
    for token in range_spec.split(","):
        if not token.strip():
            continue
        m = _RANGE_TOKEN_RE.match(token)
        if not m:
            raise PluginError(
                f"Invalid version-range token {token!r} in {range_spec!r}. "
                f"Expected operators >= > <= < == = followed by a version."
            )
        op, ver = m.group(1), m.group(2)
        if op == "=":  # normalize
            op = "=="
        constraints.append((op, _parse_version(ver)))
    return constraints


def version_matches_range(version: str, range_spec: str) -> bool:
    """Return True if *version* satisfies *range_spec*.

    Empty / None range spec is permissive — returns True. A range with no
    parseable constraints (e.g. trailing commas) also returns True so we
    don't reject installs on a no-op spec.
    """
    if not range_spec:
        return True
    constraints = parse_version_range(range_spec)
    if not constraints:
        return True
    actual = _parse_version(version)
    for op, expected in constraints:
        if op == ">=" and not (actual >= expected):
            return False
        if op == ">" and not (actual > expected):
            return False
        if op == "<=" and not (actual <= expected):
            return False
        if op == "<" and not (actual < expected):
            return False
        if op == "==" and not (actual == expected):
            return False
    return True


def validate_tap_directory(root: Path) -> list[ManifestValidationResult]:
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
    results: list[ManifestValidationResult] = []
    for manifest_path in sorted(root.rglob("plugin.yaml")):
        # Skip anything inside a blocked directory
        rel_parts = manifest_path.relative_to(root).parts
        if any(part in skip_dirs for part in rel_parts):
            continue
        results.append(validate_manifest_file(manifest_path))
    return results
