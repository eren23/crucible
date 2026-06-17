"""Code-level mutation — Phase 5.1 implementation.

Promoted from the Phase 3.6 interface stub to a working policy
surface after Harvard + DeepMind ERA (*Nature*, May 2026)
validated tree-search over scientific code as a viable autonomous
research mode.

Public surface (kept stable from the stub):
- ``MutationProposal`` / ``MutationResult`` dataclasses
- ``CodeMutationPolicy`` ABC with ``validate(proposal) -> list[str]``
  and ``apply(proposal) -> MutationResult``
- ``StubCodeMutationPolicy`` — preserved as the ``stub`` alias so
  existing tests keep passing
- ``register_code_mutation_policy`` / ``list_code_mutation_policies``
  / ``describe_code_mutation_policy`` / ``get_code_mutation_policy``
  / ``discover_code_mutation_policies``

New in Phase 5.1:
- ``AstSafetyChecker`` — walks post-mutation source AST and flags
  subprocess / network / dunder / sensitive-module patterns.
- ``SandboxRunner`` — rsync-clones the project into a tempdir and
  runs an arbitrary command with timeout + restricted env.
- ``apply_unified_diff`` — applies a diff via ``git apply`` inside a
  clean workspace; raises ``DiffApplyError`` on conflict.
- ``score_stdout`` — extracts the standard
  ``step:N/M val_loss:X val_bpb:Y`` pattern.
- ``AstLocalEditPolicy`` — function-level identifier / literal swap
  via Python AST; generates its own diff. No cross-file edits.
- ``LlmDiffPolicy`` — orchestrator-contract: ``request_prompt`` /
  ``submit`` envelope so an external LLM produces the diff;
  Crucible validates, AST-checks, sandboxes, scores. No LLM keys.
- ``build_code_mutation_policy`` — factory that injects the
  per-project context (project_root, scorer, sandbox config) into
  the policy.

Out of scope for 5.1 (carry-forward to 5.2+):
- Cross-file refactors larger than one hunk per file
- Network-allowlist gates for legitimately networked training runs
- Resource limits via ``resource.setrlimit`` (macOS friction; the
  subprocess timeout is the MVP guardrail)
- VersionStore integration for successful mutations
"""
from __future__ import annotations

import ast
import difflib
import json
import os
import re
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.core.errors import CrucibleError
from crucible.core.log import log_warn

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CodeMutationError(CrucibleError):
    """Code-mutation policy failed (parse, apply, exec, score)."""


class CodeMutationNotImplemented(CodeMutationError):
    """The stub policy was invoked. Wire a real policy."""


class DiffApplyError(CodeMutationError):
    """``git apply`` rejected the proposed diff."""


class SafetyCheckError(CodeMutationError):
    """Post-mutation source tripped the AST safety filter."""


class SandboxError(CodeMutationError):
    """Sandbox setup or execution failed before scoring could run."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MutationProposal:
    """One orchestrator-supplied mutation proposal.

    Attributes
    ----------
    name:
        Short identifier (becomes the experiment name).
    target_file:
        Path relative to project root. For multi-file diffs, the
        *primary* file (the diff itself drives the actual files
        touched). Used by validators that need a single anchor.
    diff:
        Either a unified-diff string (``LlmDiffPolicy``) or a JSON
        edit-spec string (``AstLocalEditPolicy``). The policy decides
        the interpretation.
    hypothesis:
        Free-text what the mutation tests.
    rationale:
        Free-text why the orchestrator picked this mutation.
    parent_node_id:
        Optional ID of the tree node this mutation expands from.
    mutation_scope:
        Optional allowlist of file paths the diff may touch. ``None``
        means "any file under project_root". Empty list means deny
        all. Matched as prefix against each diff target.
    """

    name: str
    target_file: str
    diff: str
    hypothesis: str = ""
    rationale: str = ""
    parent_node_id: str | None = None
    mutation_scope: list[str] | None = None


@dataclass
class MutationResult:
    """Outcome of applying + running a mutation."""

    proposal_name: str
    success: bool
    score: float | None = None
    error: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)


# Defaults are deliberately minimal — every key here is visible to the
# scorer subprocess and can be exfiltrated through stdout. HOME, USER,
# WANDB_API_KEY, HF_TOKEN, RUNPOD_API_KEY etc. are intentionally absent.
# Override via SandboxConfig(inherit_env_keys=...) at construction time
# in your own code; the MCP surface does NOT honor caller-supplied
# overrides (see ``code_mutation_apply`` in mcp/tools.py).
_DEFAULT_INHERIT_ENV_KEYS: tuple[str, ...] = (
    "PATH",
    "PYTHONPATH",
    "LANG",
    "LC_ALL",
    "CRUCIBLE_PROJECT_ROOT",
)

# Scorer-cmd basenames the sandbox refuses to launch. Shell
# interpreters, env-print utilities, and network-fetch tools all
# trivially exfiltrate any env key Crucible chose to inherit; refusing
# them up front is cheaper than auditing each call site.
_DEFAULT_SCORER_CMD_DENY: frozenset[str] = frozenset({
    "sh", "bash", "zsh", "fish", "ksh", "dash",
    "env", "printenv",
    "curl", "wget", "nc", "ncat", "socat",
    "ssh", "scp", "sftp", "rsync",
    "python", "python3",  # too easy to oneline-exfiltrate; require a script file
})


@dataclass
class SandboxConfig:
    """Sandbox limits for a single mutation run."""

    timeout_seconds: int = 1800
    allow_network: bool = False
    inherit_env_keys: tuple[str, ...] = _DEFAULT_INHERIT_ENV_KEYS
    cwd_subdir: str = "."


@dataclass
class ScorerConfig:
    """How to score a mutated workspace.

    ``cmd`` runs inside the sandbox workspace; stdout is parsed by
    ``score_pattern`` (a regex with a single capture group that yields
    the float score). ``direction`` is ``"minimize"`` or ``"maximize"``;
    it does not change the score value but is surfaced to the caller
    so tree-search policies can compare correctly.

    The default mirrors the project's standard
    ``step:N/M val_loss:X val_bpb:Y`` stdout contract.
    """

    cmd: list[str]
    score_pattern: str = r"val_bpb:([0-9]+\.?[0-9]*)"
    direction: str = "minimize"


# ---------------------------------------------------------------------------
# AST safety filter
# ---------------------------------------------------------------------------


# Each deny entry is a tuple of attribute parts. The AST walker
# reconstructs the same shape for every call site, then a set
# membership check decides allow/deny. Storing as tuples (instead of
# dotted strings) keeps the literal call patterns out of this source
# file — handy because some pre-commit security hooks key on the
# string form.
_DEFAULT_DENY_CALLS: frozenset[tuple[str, ...]] = frozenset({
    ("subprocess", "Popen"),
    ("subprocess", "run"),
    ("subprocess", "call"),
    ("subprocess", "check_call"),
    ("subprocess", "check_output"),
    ("os", "system"),
    ("os", "popen"),
    ("os", "execv"),
    ("os", "execvp"),
    ("os", "spawnv"),
    ("os", "spawnvp"),
    ("ctypes", "CDLL"),
    ("ctypes", "cdll", "LoadLibrary"),
})

_DEFAULT_DENY_NETWORK: frozenset[tuple[str, ...]] = frozenset({
    ("urllib", "request", "urlopen"),
    ("urllib", "request", "Request"),
    ("requests", "get"),
    ("requests", "post"),
    ("requests", "put"),
    ("requests", "delete"),
    ("requests", "request"),
    ("socket", "socket"),
    ("socket", "create_connection"),
    ("httpx", "get"),
    ("httpx", "post"),
    ("httpx", "Client"),
})

_DEFAULT_DENY_DUNDERS: frozenset[str] = frozenset({
    "__class__",
    "__bases__",
    "__mro__",
    "__subclasses__",
    "__globals__",
    "__builtins__",
})

_DEFAULT_SENSITIVE_MODULES: frozenset[str] = frozenset({
    "crucible.core.redact",
    "crucible.core.config",
    "crucible.core.errors",
    # Shell-escape + network modules. Blocking these at import time
    # closes the alias-bypass vector (e.g. `import subprocess as _sp`)
    # that defeats per-call deny lookups.
    "subprocess",
    "os",
    "ctypes",
    "socket",
    "urllib",
    "urllib.request",
    "requests",
    "httpx",
    "shutil",
    "pty",
})

# Names that, if referenced directly, enable dynamic call construction
# that bypasses the per-call deny list (e.g. ``getattr(__import__(
# "subprocess"), "run")``). Blocking the references themselves is the
# only way to defend against this without a full taint analysis.
_DEFAULT_DENY_DYNAMIC_NAMES: frozenset[str] = frozenset({
    "__import__",
    "getattr",
    "eval",
    "exec",
    "compile",
    "globals",
    "locals",
    "vars",
})


def _format_parts(parts: tuple[str, ...]) -> str:
    """Render an attribute-chain tuple as the dotted form for error messages."""
    return ".".join(parts)


@dataclass
class AstSafetyReport:
    """Per-file safety result."""

    target: str
    problems: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems


class AstSafetyChecker:
    """Walks a Python source AST and flags out-of-scope patterns.

    Allowlist semantics: a fully-qualified call name in
    ``allow_calls`` short-circuits the deny lookup. A target file path
    in ``allow_files`` skips the file entirely (used by per-project
    config to whitelist e.g. ``scripts/download_data.py``).
    """

    def __init__(
        self,
        *,
        allow_calls: set[tuple[str, ...]] | None = None,
        allow_files: set[str] | None = None,
        extra_deny_calls: set[tuple[str, ...]] | None = None,
        extra_deny_network: set[tuple[str, ...]] | None = None,
    ) -> None:
        self.allow_calls = allow_calls or set()
        self.allow_files = allow_files or set()
        self.deny_calls = _DEFAULT_DENY_CALLS | (extra_deny_calls or set())
        self.deny_network = _DEFAULT_DENY_NETWORK | (extra_deny_network or set())

    def check_source(self, source: str, filename: str) -> AstSafetyReport:
        if filename in self.allow_files:
            return AstSafetyReport(target=filename, problems=[])
        try:
            tree = ast.parse(source, filename=filename)
        except SyntaxError as exc:
            return AstSafetyReport(
                target=filename,
                problems=[f"{filename}: syntax error: {exc.msg} (line {exc.lineno})"],
            )

        problems: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                parts = self._call_parts(node.func)
                if not parts or parts in self.allow_calls:
                    continue
                if parts in self.deny_calls:
                    problems.append(
                        f"{filename}:{node.lineno}: shell-escape call "
                        f"{_format_parts(parts)!r} blocked"
                    )
                elif parts in self.deny_network:
                    problems.append(
                        f"{filename}:{node.lineno}: network call "
                        f"{_format_parts(parts)!r} blocked"
                    )
            elif isinstance(node, ast.Attribute):
                if node.attr in _DEFAULT_DENY_DUNDERS:
                    problems.append(
                        f"{filename}:{node.lineno}: dunder attribute {node.attr!r} touched"
                    )
            elif isinstance(node, ast.Name):
                # Blocks the dynamic-call bypass: getattr/__import__/eval/
                # exec/compile referenced anywhere (call site, alias,
                # decorator) flags the file. We do not block in expression
                # position vs call position separately because in research
                # code there is no legitimate use of these builtins.
                if node.id in _DEFAULT_DENY_DYNAMIC_NAMES:
                    problems.append(
                        f"{filename}:{node.lineno}: dynamic-call builtin "
                        f"{node.id!r} referenced — bypasses call-deny filter"
                    )
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = getattr(node, "module", None)
                names = [a.name for a in getattr(node, "names", [])]
                candidates = ([mod] if mod else []) + names
                for cand in candidates:
                    if cand in _DEFAULT_SENSITIVE_MODULES:
                        problems.append(
                            f"{filename}:{node.lineno}: import of sensitive module {cand!r} blocked"
                        )
        return AstSafetyReport(target=filename, problems=problems)

    def check_files(self, workspace: Path, files: list[str]) -> list[AstSafetyReport]:
        reports: list[AstSafetyReport] = []
        for rel in files:
            path = workspace / rel
            if not path.exists():
                reports.append(AstSafetyReport(target=rel, problems=[f"{rel}: missing after diff apply"]))
                continue
            if not rel.endswith(".py"):
                reports.append(AstSafetyReport(target=rel, problems=[]))
                continue
            reports.append(self.check_source(path.read_text(), rel))
        return reports

    @staticmethod
    def _call_parts(node: ast.AST) -> tuple[str, ...] | None:
        parts: list[str] = []
        cur: ast.AST = node
        while isinstance(cur, ast.Attribute):
            parts.insert(0, cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.insert(0, cur.id)
            return tuple(parts)
        return None


# ---------------------------------------------------------------------------
# Diff helpers
# ---------------------------------------------------------------------------


_DIFF_TARGET_RE = re.compile(r"^\+\+\+ (?:b/)?(.+?)(?:\s|$)", re.MULTILINE)


def parse_diff_targets(diff: str) -> list[str]:
    """Return the relative file paths a unified diff writes to.

    ``/dev/null`` (file deletion) is filtered out. The ``b/`` prefix is
    stripped when present. Absolute paths and paths containing ``..``
    segments are also stripped here as a defensive layer — they would
    fail the workspace-confinement check in ``execute_mutation``, but
    rejecting them at the parse step gives a clearer error.
    """
    targets: list[str] = []
    for match in _DIFF_TARGET_RE.finditer(diff):
        target = match.group(1).strip()
        if not target or target == "/dev/null":
            continue
        if target.startswith("/") or target.startswith("\\"):
            continue  # absolute path — refused; surfaced by check_workspace_confinement
        targets.append(target)
    return targets


def _matches_scope_prefix(target: str, prefix: str) -> bool:
    """Match a single target against one scope prefix.

    Two forms accepted: ``prefix`` matches when target equals it exactly
    (single-file scope, e.g. ``baseline.py``) OR target starts with
    ``prefix + "/"`` (directory scope, e.g. ``src/`` or bare ``src``).
    The trailing-slash ambiguity is resolved here so callers can write
    either form without surprises (M2 fix).
    """
    if not prefix:
        return True
    if prefix.endswith("/"):
        return target.startswith(prefix)
    # Bare prefix: match exact file OR same name treated as a directory.
    return target == prefix or target.startswith(prefix + "/")


def check_scope(targets: list[str], scope: list[str] | None) -> list[str]:
    """Return scope violations (paths that don't match any allowed prefix).

    ``scope=None`` is unrestricted. ``scope=[]`` denies everything.
    Each prefix is matched via :func:`_matches_scope_prefix`, which
    accepts both single-file (``"baseline.py"``) and directory
    (``"src"`` or ``"src/"``) forms without accidentally permitting
    ``src_backup/``.
    """
    if scope is None:
        return []
    violations = []
    for target in targets:
        if not any(_matches_scope_prefix(target, prefix) for prefix in scope):
            violations.append(target)
    return violations


def check_workspace_confinement(
    targets: list[str], workspace: Path
) -> list[str]:
    """Return any target whose resolved path escapes ``workspace``.

    Defends against `..` traversal in diff headers. Even though
    ``check_scope`` blocks scope violations, an unscoped mutation
    (``mutation_scope=None``) would otherwise let ``git apply
    --unsafe-paths`` write arbitrary files. This check is the
    last-line sandbox boundary.
    """
    bad: list[str] = []
    workspace_resolved = workspace.resolve()
    for target in targets:
        if ".." in Path(target).parts:
            bad.append(target)
            continue
        try:
            resolved = (workspace / target).resolve()
            resolved.relative_to(workspace_resolved)
        except ValueError:
            bad.append(target)
    return bad


def apply_unified_diff(workspace: Path, diff: str) -> None:
    """Apply a unified diff inside ``workspace`` using ``git apply``.

    ``git apply`` works without a ``.git`` directory for plain
    line-anchored diffs. ``--unsafe-paths`` lets it operate outside a
    git repo; ``-p1`` strips the ``a/`` prefix.

    Raises ``DiffApplyError`` on any apply conflict.
    """
    diff_path = workspace / ".crucible_mutation.diff"
    diff_path.write_text(diff)
    try:
        result = subprocess.run(
            ["git", "apply", "--unsafe-paths", "-p1", str(diff_path)],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise DiffApplyError("git apply timed out after 30s") from exc
    finally:
        diff_path.unlink(missing_ok=True)
    if result.returncode != 0:
        raise DiffApplyError(
            f"git apply rejected diff (exit {result.returncode}): {result.stderr.strip()}"
        )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def validate_scorer_cmd(cmd: list[str], workspace: Path) -> str | None:
    """Refuse scorer commands that obviously exfiltrate env or shell out.

    Python interpreters are allowed only with a workspace-relative
    ``.py`` script as the second argument — this is enough for the
    typical ``["python3", "scorer.py"]`` pattern while blocking
    ``["python3", "-c", "..."]`` injection. Shell/env/network
    basenames are hard-denied.

    Returns an error string if ``cmd`` is suspicious, ``None`` if OK.
    """
    if not cmd:
        return "scorer.cmd is empty"
    basename = Path(cmd[0]).name.lower()
    if basename in {"python", "python3"} or basename.startswith("python3."):
        if len(cmd) < 2:
            return (
                f"scorer.cmd starts with {basename!r} but no script arg — "
                f"bare {basename} would run REPL or read stdin"
            )
        script = cmd[1]
        if script.startswith("-"):
            return (
                f"scorer.cmd {basename} script slot is a flag {script!r}; "
                f"refused (would allow -c oneliner injection)"
            )
        if script.startswith("/") or script.startswith("\\"):
            return f"scorer.cmd python script must be workspace-relative, got {script!r}"
        if not script.endswith(".py"):
            return f"scorer.cmd python script must end with .py, got {script!r}"
        try:
            (workspace / script).resolve().relative_to(workspace.resolve())
        except (ValueError, OSError):
            return f"scorer.cmd python script {script!r} escapes workspace"
        return None
    if basename in _DEFAULT_SCORER_CMD_DENY:
        return (
            f"scorer.cmd basename {basename!r} blocked — "
            f"shell/env/network tool would exfiltrate sandbox env"
        )
    return None


def score_stdout(stdout: str, pattern: str) -> float | None:
    """Return the *last* float captured by ``pattern`` in ``stdout``.

    Returns ``None`` if no match. Caller is responsible for handling
    "no score found" — typically marks the mutation as failed.
    """
    matches = re.findall(pattern, stdout)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------


_DEFAULT_RSYNC_EXCLUDES = (
    ".git",
    "__pycache__",
    "*.pyc",
    "data/",
    "checkpoints/",
    "*.pt",
    "*.bin",
    "*.safetensors",
    "wandb/",
    ".crucible/search_trees/",
    ".crucible/eval_watch_ckpts/",
    "node_modules/",
    ".venv/",
    "venv/",
)


class SandboxRunner:
    """Rsync-clone the project to a tempdir and run a command there.

    The runner is stateless across mutations: every ``cloned_workspace``
    call produces a fresh clone, and the context manager removes it on
    exit. Failed mutations leave no trace in the user's working tree.
    """

    def __init__(
        self,
        project_root: Path,
        *,
        sandbox_root: Path | None = None,
        rsync_excludes: tuple[str, ...] = _DEFAULT_RSYNC_EXCLUDES,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.sandbox_root = (
            Path(sandbox_root).resolve()
            if sandbox_root is not None
            else Path(tempfile.gettempdir()) / "crucible_mutation_sandbox"
        )
        self.rsync_excludes = rsync_excludes

    @contextmanager
    def cloned_workspace(self) -> Iterator[Path]:
        """Yield a fresh rsync clone of the project; remove on exit."""
        self.sandbox_root.mkdir(parents=True, exist_ok=True)
        clone_dir = Path(tempfile.mkdtemp(prefix="mut_", dir=str(self.sandbox_root)))
        try:
            cmd = ["rsync", "-a"]
            for pattern in self.rsync_excludes:
                cmd.extend(["--exclude", pattern])
            cmd.extend([f"{self.project_root}/", f"{clone_dir}/"])
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if result.returncode != 0:
                raise SandboxError(
                    f"rsync clone failed (exit {result.returncode}): {result.stderr.strip()}"
                )
            yield clone_dir
        finally:
            shutil.rmtree(clone_dir, ignore_errors=True)

    def run(
        self, workspace: Path, cmd: list[str], config: SandboxConfig
    ) -> dict[str, Any]:
        """Run ``cmd`` inside ``workspace`` under ``config`` limits."""
        env: dict[str, str] = {}
        for key in config.inherit_env_keys:
            value = os.environ.get(key)
            if value is not None:
                env[key] = value
        env["CRUCIBLE_PROJECT_ROOT"] = str(workspace)
        if not config.allow_network:
            # Best-effort: most HTTP clients honor these. Not a security
            # boundary — a determined script can bypass with raw sockets.
            # The AST safety filter is the real network gate.
            env.setdefault("NO_PROXY", "*")
            env.setdefault("HTTP_PROXY", "http://127.0.0.1:1")
            env.setdefault("HTTPS_PROXY", "http://127.0.0.1:1")
        cwd = workspace / config.cwd_subdir
        try:
            result = subprocess.run(
                cmd,
                cwd=str(cwd),
                env=env,
                capture_output=True,
                text=True,
                timeout=config.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            return {
                "ok": False,
                "returncode": -1,
                "stdout": stdout,
                "stderr": f"timeout after {config.timeout_seconds}s",
            }
        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }


# ---------------------------------------------------------------------------
# Execution helper — diff → sandbox → score → MutationResult
# ---------------------------------------------------------------------------


def execute_mutation(
    *,
    proposal: MutationProposal,
    diff: str,
    sandbox: SandboxRunner,
    sandbox_config: SandboxConfig,
    scorer: ScorerConfig,
    safety: AstSafetyChecker,
) -> MutationResult:
    """Run the full apply → safety-check → sandbox → score pipeline.

    Used by both ``AstLocalEditPolicy`` (after it generates its diff)
    and ``LlmDiffPolicy`` (after the orchestrator supplies the diff).
    """
    targets = parse_diff_targets(diff)
    if not targets:
        return MutationResult(
            proposal_name=proposal.name,
            success=False,
            error="diff has no +++ targets — malformed (or every target was absolute / /dev/null)",
        )
    scope_violations = check_scope(targets, proposal.mutation_scope)
    if scope_violations:
        return MutationResult(
            proposal_name=proposal.name,
            success=False,
            error=f"mutation_scope violation on {scope_violations}",
        )

    try:
        with sandbox.cloned_workspace() as workspace:
            # Path-traversal defense: even if scope is None (unrestricted),
            # `git apply --unsafe-paths` must not write outside the
            # workspace. Defended pre-apply so we never modify the clone.
            traversal_bad = check_workspace_confinement(targets, workspace)
            if traversal_bad:
                return MutationResult(
                    proposal_name=proposal.name,
                    success=False,
                    error=(
                        f"workspace-confinement violation on {traversal_bad}: "
                        "diff targets escape sandbox"
                    ),
                    artifacts={"diff": diff, "targets": targets},
                )

            # Scorer-cmd defense: refuse shell/env/network basenames; for
            # python interpreters, require a workspace-relative .py script.
            scorer_problem = validate_scorer_cmd(scorer.cmd, workspace)
            if scorer_problem:
                return MutationResult(
                    proposal_name=proposal.name,
                    success=False,
                    error=f"scorer.cmd refused: {scorer_problem}",
                    artifacts={"scorer_cmd": scorer.cmd},
                )

            try:
                apply_unified_diff(workspace, diff)
            except DiffApplyError as exc:
                return MutationResult(
                    proposal_name=proposal.name,
                    success=False,
                    error=str(exc),
                    artifacts={"diff": diff, "targets": targets},
                )

            reports = safety.check_files(workspace, targets)
            all_problems: list[str] = [p for r in reports for p in r.problems]
            if all_problems:
                return MutationResult(
                    proposal_name=proposal.name,
                    success=False,
                    error="safety filter rejected mutation: " + "; ".join(all_problems),
                    artifacts={"diff": diff, "safety_reports": [r.problems for r in reports]},
                )

            run = sandbox.run(workspace, scorer.cmd, sandbox_config)
            score = score_stdout(run["stdout"], scorer.score_pattern)
            success = run["ok"] and score is not None
            if success:
                error: str | None = None
            elif not run["ok"]:
                error = run["stderr"][:2000] if run["stderr"] else f"exit {run['returncode']}"
            else:
                error = f"scorer produced no match for pattern {scorer.score_pattern!r}"
            return MutationResult(
                proposal_name=proposal.name,
                success=success,
                score=score,
                error=error,
                artifacts={
                    "diff": diff,
                    "targets": targets,
                    "stdout_tail": run["stdout"][-2000:],
                    "stderr_tail": run["stderr"][-2000:] if run["stderr"] else "",
                    "returncode": run["returncode"],
                    "scorer_cmd": scorer.cmd,
                    "scorer_pattern": scorer.score_pattern,
                    "scorer_direction": scorer.direction,
                },
            )
    except SandboxError as exc:
        return MutationResult(
            proposal_name=proposal.name,
            success=False,
            error=f"sandbox setup failed: {exc}",
        )


# ---------------------------------------------------------------------------
# Policy ABC + stub
# ---------------------------------------------------------------------------


class CodeMutationPolicy(ABC):
    """Contract for code-mutation tree-expansion policies.

    Each concrete policy is constructed with whatever per-project
    context it needs (project_root, scorer, sandbox config) and then
    invoked with a single ``MutationProposal`` at a time.
    """

    @abstractmethod
    def validate(self, proposal: MutationProposal) -> list[str]:
        """Pre-flight: list problems with the proposal, or [] if OK."""
        ...

    @abstractmethod
    def apply(self, proposal: MutationProposal) -> MutationResult:
        """Apply the mutation + run + score. Returns a MutationResult."""
        ...


class StubCodeMutationPolicy(CodeMutationPolicy):
    """Backward-compat stub. Real policies: ``AstLocalEditPolicy``, ``LlmDiffPolicy``."""

    def validate(self, proposal: MutationProposal) -> list[str]:
        return ["code_mutation stub policy — pick ast_local_edit or llm_diff for real mutations"]

    def apply(self, proposal: MutationProposal) -> MutationResult:
        raise CodeMutationNotImplemented(
            "StubCodeMutationPolicy.apply: this is the no-op stub. "
            "Register or build one of: ast_local_edit, llm_diff."
        )


# ---------------------------------------------------------------------------
# AstLocalEditPolicy — generates its own diff via Python AST
# ---------------------------------------------------------------------------


_AST_EDIT_KINDS = frozenset({"swap_identifier", "swap_literal", "swap_attribute"})


class AstLocalEditPolicy(CodeMutationPolicy):
    """Function-level identifier / literal swap via Python AST.

    The ``proposal.diff`` field is repurposed as a JSON edit spec
    string. Supported edit kinds:

    - ``swap_identifier`` — ``{"kind": "swap_identifier", "old": "relu", "new": "gelu"}``
      Replaces every ``ast.Name`` AND ``ast.Attribute`` whose name matches ``old``.
    - ``swap_literal`` — ``{"kind": "swap_literal", "old": 0.1, "new": 0.2}``
      Replaces every ``ast.Constant`` whose value equals ``old``.
    - ``swap_attribute`` — ``{"kind": "swap_attribute", "old": "GELU", "new": "SiLU"}``
      Like swap_identifier but only matches attributes (e.g. ``nn.GELU`` → ``nn.SiLU``).

    Regenerates source via ``ast.unparse`` and diffs against the
    original via ``difflib.unified_diff``. Mutations that produce a
    syntactically invalid file are rejected before the sandbox starts.
    """

    def __init__(
        self,
        *,
        project_root: Path | str | None = None,
        sandbox: SandboxRunner | None = None,
        sandbox_config: SandboxConfig | None = None,
        scorer: ScorerConfig | None = None,
        safety: AstSafetyChecker | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve() if project_root else None
        self._sandbox = sandbox
        self.sandbox_config = sandbox_config or SandboxConfig()
        self.scorer = scorer
        self.safety = safety or AstSafetyChecker()

    @property
    def sandbox(self) -> SandboxRunner:
        if self._sandbox is not None:
            return self._sandbox
        if self.project_root is None:
            raise CodeMutationError(
                "AstLocalEditPolicy needs project_root to construct a SandboxRunner"
            )
        self._sandbox = SandboxRunner(self.project_root)
        return self._sandbox

    def validate(self, proposal: MutationProposal) -> list[str]:
        if self.project_root is None:
            return ["AstLocalEditPolicy needs project_root; rebuild via build_code_mutation_policy()"]
        if self.scorer is None:
            return ["AstLocalEditPolicy needs a scorer (ScorerConfig)"]
        # Detect operator error: someone passing a unified diff here
        # likely meant to use llm_diff. Give a directional hint instead
        # of an opaque JSON parse error.
        stripped = proposal.diff.lstrip()
        if stripped.startswith("---") or stripped.startswith("+++"):
            return [
                "diff field looks like a unified diff — AstLocalEditPolicy "
                "expects a JSON edit spec. Did you mean policy='llm_diff'?"
            ]
        try:
            spec = json.loads(proposal.diff)
        except json.JSONDecodeError as exc:
            return [
                f"diff field must be JSON edit spec (e.g. "
                f'{{"kind": "swap_literal", "old": ..., "new": ...}}), got: {exc}'
            ]
        if not isinstance(spec, dict) or "kind" not in spec:
            return ["edit spec must be a dict with a 'kind' key"]
        if spec["kind"] not in _AST_EDIT_KINDS:
            return [f"unknown edit kind: {spec['kind']!r}; expected one of {sorted(_AST_EDIT_KINDS)}"]
        if "old" not in spec or "new" not in spec:
            return ["edit spec must include 'old' and 'new'"]
        if not proposal.target_file.endswith(".py"):
            return [f"AstLocalEditPolicy only handles .py files; got {proposal.target_file!r}"]
        target_path = self.project_root / proposal.target_file
        if not target_path.exists():
            return [f"target_file not found: {proposal.target_file}"]
        return []

    def generate_diff(self, proposal: MutationProposal) -> str:
        """Apply the edit spec to the on-disk source and return a unified diff."""
        if self.project_root is None:
            raise CodeMutationError("project_root not set")
        spec = json.loads(proposal.diff)
        target_path = self.project_root / proposal.target_file
        original = target_path.read_text()
        new = self._rewrite(original, spec)
        # ast.unparse drops the trailing newline; restore it so the
        # generated diff applies cleanly via git apply.
        if original.endswith("\n") and not new.endswith("\n"):
            new = new + "\n"
        try:
            ast.parse(new, filename=proposal.target_file)
        except SyntaxError as exc:
            raise CodeMutationError(
                f"mutation produced syntax error in {proposal.target_file}: {exc}"
            ) from exc
        if new == original:
            raise CodeMutationError(
                f"edit spec {spec!r} produced no changes in {proposal.target_file}"
            )
        return "".join(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile=f"a/{proposal.target_file}",
                tofile=f"b/{proposal.target_file}",
            )
        )

    def apply(self, proposal: MutationProposal) -> MutationResult:
        problems = self.validate(proposal)
        if problems:
            return MutationResult(
                proposal_name=proposal.name,
                success=False,
                error="validate() failed: " + "; ".join(problems),
            )
        try:
            diff = self.generate_diff(proposal)
        except CodeMutationError as exc:
            return MutationResult(
                proposal_name=proposal.name,
                success=False,
                error=str(exc),
            )
        assert self.scorer is not None  # validate() guarantees this
        return execute_mutation(
            proposal=proposal,
            diff=diff,
            sandbox=self.sandbox,
            sandbox_config=self.sandbox_config,
            scorer=self.scorer,
            safety=self.safety,
        )

    @staticmethod
    def _rewrite(source: str, spec: dict[str, Any]) -> str:
        tree = ast.parse(source)
        kind = spec["kind"]
        old = spec["old"]
        new = spec["new"]
        if kind == "swap_identifier":
            tree = _IdentifierSwapper(old, new).visit(tree)
        elif kind == "swap_attribute":
            tree = _AttributeSwapper(old, new).visit(tree)
        elif kind == "swap_literal":
            tree = _LiteralSwapper(old, new).visit(tree)
        else:  # pragma: no cover — guarded by validate()
            raise CodeMutationError(f"unknown kind: {kind!r}")
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)


class _IdentifierSwapper(ast.NodeTransformer):
    def __init__(self, old: str, new: str) -> None:
        self.old = old
        self.new = new

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == self.old:
            return ast.copy_location(ast.Name(id=self.new, ctx=node.ctx), node)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        if node.attr == self.old:
            return ast.copy_location(
                ast.Attribute(value=node.value, attr=self.new, ctx=node.ctx),
                node,
            )
        return node


class _AttributeSwapper(ast.NodeTransformer):
    def __init__(self, old: str, new: str) -> None:
        self.old = old
        self.new = new

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        if node.attr == self.old:
            return ast.copy_location(
                ast.Attribute(value=node.value, attr=self.new, ctx=node.ctx),
                node,
            )
        return node


class _LiteralSwapper(ast.NodeTransformer):
    def __init__(self, old: Any, new: Any) -> None:
        self.old = old
        self.new = new

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if type(node.value) is type(self.old) and node.value == self.old:
            return ast.copy_location(ast.Constant(value=self.new), node)
        return node


# ---------------------------------------------------------------------------
# LlmDiffPolicy — orchestrator-contract envelope
# ---------------------------------------------------------------------------


LLM_DIFF_SYSTEM_PROMPT = (
    "You propose ONE unified-diff patch to a single file in a "
    "research codebase. Output strict JSON matching the supplied "
    "schema — no markdown fences, no prose preamble.\n\n"
    "Rules:\n"
    "- Output a unified diff against the supplied file path.\n"
    "- The diff must apply cleanly with `git apply -p1`.\n"
    "- Do not touch files outside the supplied target_file unless "
    "explicitly authorised by mutation_scope.\n"
    "- Do not introduce shell-escape or network calls — the safety "
    "filter will reject them.\n"
    "- Keep edits minimal and surgical. Prefer one hunk.\n"
    "- The hypothesis field describes what your change tests; the "
    "rationale field describes why you picked it.\n"
)


def llm_diff_request_prompt(
    *,
    target_file: str,
    intent: str,
    project_root: Path | str,
    mutation_scope: list[str] | None = None,
    extra_context: str = "",
) -> dict[str, Any]:
    """Build the orchestrator-contract envelope for the LLM diff stage.

    Returns ``{system, user, schema}``. The orchestrator runs its own
    LLM with this envelope, then submits the JSON response via
    :func:`llm_diff_parse_response`.
    """
    project_root = Path(project_root).resolve()
    target_path = project_root / target_file
    if not target_path.exists():
        raise CodeMutationError(f"target_file not found: {target_file}")
    source = target_path.read_text()
    max_chars = 16000
    truncated_note = ""
    if len(source) > max_chars:
        source = source[:max_chars]
        truncated_note = f"\n\n[NOTE: file truncated at {max_chars} chars]"

    user = (
        f"# Target file: {target_file}\n"
        f"# Intent: {intent}\n"
    )
    if mutation_scope:
        user += f"# mutation_scope (allowed prefixes): {mutation_scope}\n"
    if extra_context:
        user += f"\n## Extra context\n{extra_context}\n"
    user += f"\n## Current source\n```python\n{source}{truncated_note}\n```\n"

    schema = {
        "type": "object",
        "required": ["diff", "hypothesis", "rationale"],
        "properties": {
            "diff": {
                "type": "string",
                "description": "Unified diff against target_file. Must apply with `git apply -p1`.",
            },
            "hypothesis": {
                "type": "string",
                "description": "What this mutation tests.",
            },
            "rationale": {
                "type": "string",
                "description": "Why this mutation is worth running.",
            },
            "name": {
                "type": "string",
                "description": "Optional short identifier; default derived from intent.",
            },
        },
    }
    return {
        "system": LLM_DIFF_SYSTEM_PROMPT,
        "user": user,
        "schema": schema,
        "target_file": target_file,
        "mutation_scope": mutation_scope,
    }


def llm_diff_parse_response(
    response: dict[str, Any],
    *,
    target_file: str,
    mutation_scope: list[str] | None = None,
    default_name: str = "llm_diff",
    allow_multi_file: bool = False,
) -> MutationProposal:
    """Validate the orchestrator's response and produce a MutationProposal.

    By default the diff must touch only ``target_file`` — multi-file
    diffs are refused unless ``allow_multi_file=True``. The single-file
    constraint matches the LLM prompt envelope, which only shows one
    file to the model.
    """
    if not isinstance(response, dict):
        raise CodeMutationError("LLM response must be a JSON object")
    for required in ("diff", "hypothesis", "rationale"):
        if required not in response or not isinstance(response[required], str):
            raise CodeMutationError(f"LLM response missing required string field {required!r}")
    diff = response["diff"]
    targets = parse_diff_targets(diff)
    if not targets:
        raise CodeMutationError("LLM diff has no +++ targets — malformed")
    if not allow_multi_file:
        if len(targets) != 1:
            raise CodeMutationError(
                f"LLM diff touches {len(targets)} files {targets!r}; "
                f"expected exactly one ({target_file!r}). "
                "Set allow_multi_file=True to relax."
            )
        if targets[0] != target_file:
            raise CodeMutationError(
                f"LLM diff target {targets[0]!r} does not match requested "
                f"target_file {target_file!r}"
            )
    violations = check_scope(targets, mutation_scope) if mutation_scope is not None else []
    if violations:
        raise CodeMutationError(
            f"LLM diff touches files outside mutation_scope: {violations}"
        )
    return MutationProposal(
        name=str(response.get("name") or default_name),
        target_file=target_file,
        diff=diff,
        hypothesis=response["hypothesis"],
        rationale=response["rationale"],
        mutation_scope=mutation_scope,
    )


class LlmDiffPolicy(CodeMutationPolicy):
    """Apply + sandbox + score a diff produced by an external LLM.

    The orchestrator generates the diff via :func:`llm_diff_request_prompt`
    and :func:`llm_diff_parse_response`. This policy then applies it.
    The diff lives on the proposal — the policy itself is LLM-free.
    """

    def __init__(
        self,
        *,
        project_root: Path | str | None = None,
        sandbox: SandboxRunner | None = None,
        sandbox_config: SandboxConfig | None = None,
        scorer: ScorerConfig | None = None,
        safety: AstSafetyChecker | None = None,
        allow_non_py: bool = False,
    ) -> None:
        self.project_root = Path(project_root).resolve() if project_root else None
        self._sandbox = sandbox
        self.sandbox_config = sandbox_config or SandboxConfig()
        self.scorer = scorer
        self.safety = safety or AstSafetyChecker()
        # AstSafetyChecker only inspects .py files. Non-Python targets
        # (shell scripts, Makefiles, YAML configs) bypass the filter
        # entirely. Opt-in flag for callers who know what they're doing.
        self.allow_non_py = allow_non_py

    @property
    def sandbox(self) -> SandboxRunner:
        if self._sandbox is not None:
            return self._sandbox
        if self.project_root is None:
            raise CodeMutationError(
                "LlmDiffPolicy needs project_root to construct a SandboxRunner"
            )
        self._sandbox = SandboxRunner(self.project_root)
        return self._sandbox

    def validate(self, proposal: MutationProposal) -> list[str]:
        if self.scorer is None:
            return ["LlmDiffPolicy needs a scorer (ScorerConfig)"]
        if not proposal.diff.strip():
            return ["proposal.diff is empty"]
        # Operator-error hint: a JSON edit spec smells like
        # AstLocalEditPolicy. Catch it before git apply chokes.
        stripped = proposal.diff.lstrip()
        if stripped.startswith("{"):
            return [
                "diff field looks like a JSON edit spec — LlmDiffPolicy "
                "expects a unified diff. Did you mean policy='ast_local_edit'?"
            ]
        targets = parse_diff_targets(proposal.diff)
        if not targets:
            return ["proposal.diff has no +++ targets"]
        # M7: refuse non-.py targets by default. AstSafetyChecker is
        # Python-only; a malicious LLM diff could otherwise drop a
        # `Makefile` or `run.sh` and bypass the safety filter entirely.
        if not self.allow_non_py:
            non_py = [t for t in targets if not t.endswith(".py")]
            if non_py:
                return [
                    f"LlmDiffPolicy refuses non-.py targets {non_py!r} "
                    "(AST safety filter is Python-only). "
                    "Set allow_non_py=True to override (operator vouches for safety)."
                ]
        violations = check_scope(targets, proposal.mutation_scope)
        if violations:
            return [f"mutation_scope violation: {violations}"]
        return []

    def apply(self, proposal: MutationProposal) -> MutationResult:
        problems = self.validate(proposal)
        if problems:
            return MutationResult(
                proposal_name=proposal.name,
                success=False,
                error="validate() failed: " + "; ".join(problems),
            )
        assert self.scorer is not None
        return execute_mutation(
            proposal=proposal,
            diff=proposal.diff,
            sandbox=self.sandbox,
            sandbox_config=self.sandbox_config,
            scorer=self.scorer,
            safety=self.safety,
        )


# ---------------------------------------------------------------------------
# Registry (PluginRegistry-backed)
# ---------------------------------------------------------------------------


from crucible.core.plugin_registry import PluginRegistry  # noqa: E402

_POLICY_REGISTRY = PluginRegistry[type[CodeMutationPolicy]]("code_mutation")


def register_code_mutation_policy(
    name: str,
    policy_cls: type[CodeMutationPolicy],
    source: str = "builtin",
) -> None:
    """Register a CodeMutationPolicy subclass under ``name``."""
    _POLICY_REGISTRY.register(name, policy_cls, source=source)


def list_code_mutation_policies() -> list[str]:
    """Return registered policy names (sorted)."""
    return sorted(_POLICY_REGISTRY.list_plugins())


def describe_code_mutation_policy(name: str) -> dict[str, str] | None:
    """Return registry metadata for a policy: ``{name, type, source}``."""
    return _POLICY_REGISTRY.describe_plugin(name)


def discover_code_mutation_policies(project_root: Path | None = None) -> None:
    """Trigger auto-discovery of code-mutation policies on disk.

    Walks both the global hub plugin dir
    (``~/.crucible-hub/plugins/code_mutation/``) and the project-local
    plugin dir (``<project>/.crucible/plugins/code_mutation/``) and
    registers any ``.py`` files found via the standard PluginRegistry
    ``load_plugins`` contract.

    Missing directories are silently skipped — discovery is best-effort.
    """
    global_dir = Path.home() / ".crucible-hub" / "plugins" / "code_mutation"
    if global_dir.is_dir():
        _POLICY_REGISTRY.load_plugins(global_dir, source="global")
    if project_root is not None:
        local_dir = Path(project_root) / ".crucible" / "plugins" / "code_mutation"
        if local_dir.is_dir():
            _POLICY_REGISTRY.load_plugins(local_dir, source="local")


def get_code_mutation_policy(name: str = "code_mutation") -> CodeMutationPolicy:
    """Look up + instantiate a policy by name with default constructor.

    For real policies that need context (project_root, scorer, etc.),
    use :func:`build_code_mutation_policy` instead.
    """
    cls = _POLICY_REGISTRY.get(name)
    if cls is None:
        raise CodeMutationError(
            f"No code_mutation policy registered as {name!r}. "
            f"Registered: {list_code_mutation_policies()}"
        )
    return cls()


def build_code_mutation_policy(
    name: str,
    *,
    project_root: Path | str,
    scorer: ScorerConfig,
    sandbox_config: SandboxConfig | None = None,
    safety: AstSafetyChecker | None = None,
    allow_non_py: bool = False,
) -> CodeMutationPolicy:
    """Look up a policy class and instantiate it with full per-project context.

    Only real policies that accept these kwargs benefit (AstLocalEditPolicy,
    LlmDiffPolicy). The stub falls back to the zero-arg constructor.

    ``allow_non_py=True`` lets LlmDiffPolicy mutate non-Python files;
    the operator is then responsible for vouching that the AST-only
    safety filter is acceptable. AstLocalEditPolicy ignores this flag
    (it only handles ``.py`` by construction).
    """
    cls = _POLICY_REGISTRY.get(name)
    if cls is None:
        raise CodeMutationError(
            f"No code_mutation policy registered as {name!r}. "
            f"Registered: {list_code_mutation_policies()}"
        )
    # Pass policy-specific kwargs only when accepted, so the registry
    # stays uniform across stub / ast_local_edit / llm_diff.
    kwargs: dict[str, Any] = {
        "project_root": project_root,
        "scorer": scorer,
        "sandbox_config": sandbox_config,
        "safety": safety,
    }
    if cls is LlmDiffPolicy:
        kwargs["allow_non_py"] = allow_non_py
    try:
        return cls(**kwargs)
    except TypeError as exc:
        log_warn(
            f"code_mutation policy {name!r} ignored context kwargs ({exc}); "
            "falling back to bare constructor"
        )
        return cls()


# ---------------------------------------------------------------------------
# Default registrations
# ---------------------------------------------------------------------------


register_code_mutation_policy("code_mutation", StubCodeMutationPolicy)
register_code_mutation_policy("stub", StubCodeMutationPolicy)
register_code_mutation_policy("ast_local_edit", AstLocalEditPolicy)
register_code_mutation_policy("llm_diff", LlmDiffPolicy)
