"""Tap-quality linter — catches the things ``validate_tap_directory``
doesn't.

``plugin_schema.validate_tap_directory`` enforces the *per-plugin* manifest
schema. This module enforces *repo-level* quality conventions: presence of
LICENSE / README, no cruft directories committed, plugin folder names
matching their manifests, no large binaries in git, etc.

Each lint check is a small subclass of :class:`LintCheck` registered with
:class:`LintCheckRegistry`. The CLI entrypoint
(``crucible tap lint <path>``) walks the registry and prints a structured
report; CI workflows can call :func:`lint_tap_directory` directly for a
machine-readable result.

Design notes:
- Checks return ``LintIssue`` instances with a *fix_hint* — a one-liner
  the user can paste into a terminal to fix the issue. Hints are
  load-bearing for adoption: a linter that points at problems without
  prescribing fixes pushes the work onto the contributor.
- Issues carry a stable ``code`` (e.g. ``L001``) so CI configs can pin
  expectations and downgrade-to-warning specific checks if needed.
- Severity is ``error`` or ``warning`` — error fails the CLI exit code.
"""
from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import yaml

from crucible.core.errors import PluginError
from crucible.core.plugin_schema import (
    KNOWN_PLUGIN_TYPES,
    validate_manifest_dict,
    validate_tap_manifest_file,
)


@dataclass
class LintIssue:
    """One quality issue found while linting a tap directory."""
    code: str                # stable identifier (e.g. "L003")
    severity: str            # "error" | "warning"
    message: str             # one-line description of the problem
    path: Path               # offending file (or the tap root for repo-level checks)
    fix_hint: str = ""       # suggested command or action to resolve


# ---------------------------------------------------------------------------
# LintCheck base class + registry
# ---------------------------------------------------------------------------


class LintCheck:
    """Base class for tap-quality checks.

    Subclasses declare a stable :attr:`code` and :attr:`severity` and
    implement :meth:`run` returning issues found under *tap_root*.
    """

    code: ClassVar[str] = ""
    severity: ClassVar[str] = "error"
    description: ClassVar[str] = ""

    def run(self, tap_root: Path) -> list[LintIssue]:
        raise NotImplementedError

    def _issue(self, message: str, path: Path, *, fix_hint: str = "",
               severity: str | None = None) -> LintIssue:
        return LintIssue(
            code=self.code,
            severity=severity or self.severity,
            message=message,
            path=path,
            fix_hint=fix_hint,
        )


class LintCheckRegistry:
    """Holds all registered LintCheck instances, in registration order."""

    def __init__(self) -> None:
        self._checks: list[LintCheck] = []

    def register(self, check: LintCheck) -> None:
        if not check.code:
            raise PluginError(f"LintCheck {check.__class__.__name__} has no code")
        if any(c.code == check.code for c in self._checks):
            raise PluginError(f"duplicate LintCheck code: {check.code}")
        self._checks.append(check)

    def all(self) -> list[LintCheck]:
        return list(self._checks)

    def get(self, code: str) -> LintCheck | None:
        for c in self._checks:
            if c.code == code:
                return c
        return None


# A module-level singleton so callers don't have to construct one. Tests
# can build their own registry if they want isolation.
_DEFAULT_REGISTRY: LintCheckRegistry | None = None


def get_default_registry() -> LintCheckRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = LintCheckRegistry()
        for check_cls in _BUILTIN_CHECKS:
            _DEFAULT_REGISTRY.register(check_cls())
    return _DEFAULT_REGISTRY


# ---------------------------------------------------------------------------
# Built-in checks
# ---------------------------------------------------------------------------


_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", ".pytest_cache", ".mypy_cache"}


def _walk_plugin_dirs(tap_root: Path):
    """Yield (plugin_type, plugin_dir) for every plugin folder under *tap_root*.

    A plugin folder is any directory under a recognized plugin_type whose
    parent directory matches one of ``KNOWN_PLUGIN_TYPES``. We don't follow
    symlinks (the tap.py security model already rejects them at install).
    """
    for type_dir in sorted(tap_root.iterdir()):
        if not type_dir.is_dir() or type_dir.name in _SKIP_DIRS:
            continue
        if type_dir.name not in KNOWN_PLUGIN_TYPES:
            continue
        for plugin_dir in sorted(type_dir.iterdir()):
            if not plugin_dir.is_dir():
                continue
            yield (type_dir.name, plugin_dir)


class _BaseTapCheck(LintCheck):
    """Helper base for checks that don't traverse plugin folders."""


class L001_MissingTopLevelReadme(_BaseTapCheck):
    code = "L001"
    severity = "error"
    description = "tap repo root must have a README.md"

    def run(self, tap_root: Path) -> list[LintIssue]:
        if (tap_root / "README.md").is_file():
            return []
        return [self._issue(
            "tap repo root is missing README.md",
            tap_root / "README.md",
            fix_hint="touch README.md && echo '# my-tap' > README.md",
        )]


class L002_MissingLicense(_BaseTapCheck):
    code = "L002"
    severity = "error"
    description = "tap repo root must have a LICENSE file"

    def run(self, tap_root: Path) -> list[LintIssue]:
        for candidate in ("LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING"):
            if (tap_root / candidate).is_file():
                return []
        return [self._issue(
            "tap repo root is missing LICENSE",
            tap_root / "LICENSE",
            fix_hint="add an MIT/Apache-2.0/BSD LICENSE file at the repo root",
        )]


class L003_MissingTapManifest(_BaseTapCheck):
    code = "L003"
    severity = "warning"  # optional in this plan — warn, don't error
    description = "tap repo root should have a tap.yaml manifest"

    def run(self, tap_root: Path) -> list[LintIssue]:
        manifest = tap_root / "tap.yaml"
        if not manifest.is_file():
            return [self._issue(
                "tap.yaml missing at repo root",
                manifest,
                fix_hint="crucible tap init . (or add tap.yaml with name/description/version)",
            )]
        # Delegate schema validation to plugin_schema.validate_tap_manifest_file
        result = validate_tap_manifest_file(manifest)
        return [
            self._issue(
                f"tap.yaml/{i.field}: {i.message}",
                manifest,
                severity=i.severity,
            )
            for i in result.issues
            if i.severity in ("error", "warning")
        ]


class L004_MissingPerPluginReadme(LintCheck):
    code = "L004"
    severity = "warning"
    description = "every plugin folder should have a README.md"

    def run(self, tap_root: Path) -> list[LintIssue]:
        issues: list[LintIssue] = []
        for _, plugin_dir in _walk_plugin_dirs(tap_root):
            if not (plugin_dir / "README.md").is_file():
                issues.append(self._issue(
                    "plugin folder missing README.md",
                    plugin_dir / "README.md",
                    fix_hint=f"add a one-paragraph README inside {plugin_dir}",
                ))
        return issues


class L005_CruftDirectories(_BaseTapCheck):
    code = "L005"
    severity = "error"
    description = "data/checkpoints/wandb/__pycache__/.DS_Store should not live in a tap"

    _CRUFT_NAMES = (
        "data", "checkpoints", "wandb", "_manuscript",
        "__pycache__", ".pytest_cache", ".mypy_cache", "venv", ".venv",
    )
    _CRUFT_FILES = (".DS_Store",)
    # When the bulk data has been migrated to HuggingFace, taps often want
    # to leave a small `data/` or `checkpoints/` directory containing only
    # a `README.md` pointer. That pattern is fine — flagging it punishes
    # the right cleanup. Whitelist: a "cruft-named" directory with ONLY
    # README.md inside is allowed.
    _POINTER_DIR_NAMES = ("data", "checkpoints")

    def _is_pointer_dir(self, entry: Path) -> bool:
        if entry.name not in self._POINTER_DIR_NAMES:
            return False
        contents = list(entry.iterdir())
        return len(contents) == 1 and contents[0].name == "README.md"

    def run(self, tap_root: Path) -> list[LintIssue]:
        issues: list[LintIssue] = []
        for entry in tap_root.iterdir():
            if entry.is_dir() and entry.name in self._CRUFT_NAMES:
                if self._is_pointer_dir(entry):
                    continue  # README-only pointer dir is allowed
                issues.append(self._issue(
                    f"cruft directory committed: {entry.name}/",
                    entry,
                    fix_hint=f"rm -rf {entry} && echo '{entry.name}/' >> .gitignore",
                ))
            elif entry.is_file() and entry.name in self._CRUFT_FILES:
                issues.append(self._issue(
                    f"cruft file committed: {entry.name}",
                    entry,
                    fix_hint=f"rm {entry} && echo '{entry.name}' >> .gitignore",
                ))
        # Scan plugin dirs too — sometimes the cruft is nested.
        for _, plugin_dir in _walk_plugin_dirs(tap_root):
            for sub in plugin_dir.iterdir():
                if sub.is_dir() and sub.name in self._CRUFT_NAMES:
                    issues.append(self._issue(
                        f"cruft directory inside plugin: {sub.relative_to(tap_root)}",
                        sub,
                        fix_hint=f"rm -rf {sub}",
                    ))
        return issues


class L006_LargeFileInTap(_BaseTapCheck):
    code = "L006"
    severity = "error"
    description = "files larger than 1 MB should not live in a tap"

    _LIMIT_BYTES = 1 * 1024 * 1024  # 1 MB

    def run(self, tap_root: Path) -> list[LintIssue]:
        issues: list[LintIssue] = []
        for dirpath, dirnames, filenames in os.walk(tap_root):
            # Prune skip dirs in-place so os.walk doesn't descend into .git.
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for fname in filenames:
                fpath = Path(dirpath) / fname
                try:
                    size = fpath.stat().st_size
                except OSError:
                    continue
                if size > self._LIMIT_BYTES:
                    issues.append(self._issue(
                        f"large file ({size / 1024 / 1024:.1f} MB): "
                        f"{fpath.relative_to(tap_root)}",
                        fpath,
                        fix_hint=(
                            "move to HuggingFace via `hf_push_artifact` and "
                            "leave a pointer in README.md"
                        ),
                    ))
        return issues


class L007_FolderNameMismatch(LintCheck):
    code = "L007"
    severity = "error"
    description = "plugin folder name must equal plugin.yaml's name field"

    def run(self, tap_root: Path) -> list[LintIssue]:
        issues: list[LintIssue] = []
        for _, plugin_dir in _walk_plugin_dirs(tap_root):
            manifest = plugin_dir / "plugin.yaml"
            if not manifest.is_file():
                continue
            try:
                data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
            except (yaml.YAMLError, OSError):
                continue
            if not isinstance(data, dict):
                continue
            declared = data.get("name", "")
            if declared and declared != plugin_dir.name:
                issues.append(self._issue(
                    f"folder name {plugin_dir.name!r} != manifest name "
                    f"{declared!r}",
                    manifest,
                    fix_hint=(
                        f"either rename folder to {declared!r} or update "
                        f"name: in {manifest.relative_to(tap_root)}"
                    ),
                ))
        return issues


class L008_MultiLineDescription(LintCheck):
    code = "L008"
    severity = "warning"
    description = "plugin description should be a single line"

    _MULTILINE_INDICATORS = re.compile(r"^\s*description\s*:\s*[>|]", re.MULTILINE)

    def run(self, tap_root: Path) -> list[LintIssue]:
        issues: list[LintIssue] = []
        for _, plugin_dir in _walk_plugin_dirs(tap_root):
            manifest = plugin_dir / "plugin.yaml"
            if not manifest.is_file():
                continue
            try:
                raw = manifest.read_text(encoding="utf-8")
            except OSError:
                continue
            # Block scalar (`description: >` or `description: |`) indicates
            # multi-line. Parsed-string with embedded newlines also flagged.
            if self._MULTILINE_INDICATORS.search(raw):
                issues.append(self._issue(
                    "description: is a YAML block scalar (multi-line); "
                    "collapse to a single line",
                    manifest,
                    fix_hint=(
                        "replace `description: >` / `description: |` with "
                        "a single-line string and move detail to README.md"
                    ),
                ))
                continue
            try:
                data = yaml.safe_load(raw)
            except yaml.YAMLError:
                continue
            if isinstance(data, dict):
                desc = data.get("description")
                if isinstance(desc, str) and "\n" in desc.strip():
                    issues.append(self._issue(
                        "description: contains embedded newlines",
                        manifest,
                        fix_hint="collapse description to one line",
                    ))
        return issues


class L009_VersionStringUnquoted(LintCheck):
    code = "L009"
    severity = "warning"
    description = "plugin.yaml version: should be a quoted string"

    # Matches `version: 1.0.0` (unquoted) but not `version: "1.0.0"` or
    # `version: '1.0.0'`. Catches the case where a number with two dots is
    # ambiguous to YAML readers and gets coerced inconsistently.
    _UNQUOTED_VERSION_RE = re.compile(
        r"^\s*version\s*:\s*([^'\"\s][^\n#]*?)\s*(?:#.*)?$",
        re.MULTILINE,
    )

    def run(self, tap_root: Path) -> list[LintIssue]:
        issues: list[LintIssue] = []
        for _, plugin_dir in _walk_plugin_dirs(tap_root):
            manifest = plugin_dir / "plugin.yaml"
            if not manifest.is_file():
                continue
            try:
                raw = manifest.read_text(encoding="utf-8")
            except OSError:
                continue
            m = self._UNQUOTED_VERSION_RE.search(raw)
            if m:
                issues.append(self._issue(
                    f"version: is unquoted ({m.group(1).strip()!r}); "
                    "use quoted semver",
                    manifest,
                    fix_hint=(
                        f"change to `version: \"{m.group(1).strip()}\"` "
                        f"in {manifest.relative_to(tap_root)}"
                    ),
                ))
        return issues


class L010_PythonSyntaxError(LintCheck):
    code = "L010"
    severity = "error"
    description = "plugin .py files must parse"

    def run(self, tap_root: Path) -> list[LintIssue]:
        issues: list[LintIssue] = []
        for _, plugin_dir in _walk_plugin_dirs(tap_root):
            for py_file in plugin_dir.rglob("*.py"):
                try:
                    raw = py_file.read_text(encoding="utf-8")
                except OSError:
                    continue
                try:
                    ast.parse(raw, filename=str(py_file))
                except SyntaxError as exc:
                    issues.append(self._issue(
                        f"SyntaxError at line {exc.lineno}: {exc.msg}",
                        py_file,
                        fix_hint=f"fix syntax in {py_file.relative_to(tap_root)}",
                    ))
        return issues


class L011_PerPluginManifestValid(LintCheck):
    """Run the existing per-plugin schema validator and surface its errors."""
    code = "L011"
    severity = "error"
    description = "every plugin.yaml must pass the manifest schema"

    def run(self, tap_root: Path) -> list[LintIssue]:
        issues: list[LintIssue] = []
        for _, plugin_dir in _walk_plugin_dirs(tap_root):
            manifest = plugin_dir / "plugin.yaml"
            if not manifest.is_file():
                issues.append(self._issue(
                    "plugin folder is missing plugin.yaml",
                    manifest,
                    fix_hint=f"add plugin.yaml inside {plugin_dir}",
                ))
                continue
            try:
                data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
            except (yaml.YAMLError, OSError) as exc:
                issues.append(self._issue(
                    f"plugin.yaml is unreadable: {exc}",
                    manifest,
                    fix_hint="check YAML syntax",
                ))
                continue
            if not isinstance(data, dict):
                issues.append(self._issue(
                    "plugin.yaml is not a YAML mapping",
                    manifest,
                ))
                continue
            for vi in validate_manifest_dict(data):
                issues.append(self._issue(
                    f"plugin.yaml/{vi.field}: {vi.message}",
                    manifest,
                    severity=vi.severity,
                ))
        return issues


# Order matters for the report — repo-level checks first, then per-plugin.
_BUILTIN_CHECKS: tuple[type[LintCheck], ...] = (
    L001_MissingTopLevelReadme,
    L002_MissingLicense,
    L003_MissingTapManifest,
    L005_CruftDirectories,
    L006_LargeFileInTap,
    L004_MissingPerPluginReadme,
    L007_FolderNameMismatch,
    L008_MultiLineDescription,
    L009_VersionStringUnquoted,
    L010_PythonSyntaxError,
    L011_PerPluginManifestValid,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def lint_tap_directory(
    tap_root: Path,
    *,
    registry: LintCheckRegistry | None = None,
) -> list[LintIssue]:
    """Run every registered LintCheck over *tap_root* and return a flat
    list of issues.

    Pass *registry* to use a custom set of checks (tests do this). When
    omitted, the default registry with all built-in checks is used.
    """
    if not tap_root.exists():
        raise PluginError(f"Tap directory does not exist: {tap_root}")
    if not tap_root.is_dir():
        raise PluginError(f"Tap path is not a directory: {tap_root}")
    reg = registry or get_default_registry()
    all_issues: list[LintIssue] = []
    for check in reg.all():
        try:
            all_issues.extend(check.run(tap_root))
        except Exception as exc:  # noqa: BLE001 — defensive: never crash on one bad check
            all_issues.append(LintIssue(
                code=check.code,
                severity="error",
                message=f"check crashed: {type(exc).__name__}: {exc}",
                path=tap_root,
                fix_hint="report this as a bug in crucible.core.tap_lint",
            ))
    return all_issues


def format_lint_report(issues: list[LintIssue], tap_root: Path) -> str:
    """Format issues for human terminal output."""
    if not issues:
        return f"✓ {tap_root}: 0 issues"
    by_severity: dict[str, list[LintIssue]] = {"error": [], "warning": []}
    for i in issues:
        by_severity.setdefault(i.severity, []).append(i)
    lines = [f"Linted {tap_root}"]
    lines.append(
        f"  errors:   {len(by_severity.get('error', []))}"
    )
    lines.append(
        f"  warnings: {len(by_severity.get('warning', []))}"
    )
    lines.append("")
    for issue in issues:
        try:
            rel = issue.path.relative_to(tap_root)
        except ValueError:
            rel = issue.path
        marker = "ERROR" if issue.severity == "error" else "WARN "
        lines.append(f"  [{marker}] {issue.code} {rel}")
        lines.append(f"          {issue.message}")
        if issue.fix_hint:
            lines.append(f"          fix: {issue.fix_hint}")
    return "\n".join(lines)
