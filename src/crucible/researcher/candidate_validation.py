"""Pre-dispatch validation for harness candidate source code.

Runs between proposal and tree insertion:
  * **Syntax** — source must compile.
  * **Interface** — a class with the expected name must exist and define
    every method listed in ``DomainSpec.interface.required_methods``.
  * **Constraints** — referenced parameters must fall within domain bounds.
  * **Duplicate** — hash collisions with already-stored candidates are flagged.

All failures are returned as structured results rather than raised; the
orchestrator decides whether to drop, re-prompt, or escalate. This keeps
validation cheap and side-effect-free.
"""
from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from typing import Any

from crucible.core.errors import CandidateValidationError
from crucible.researcher.domain_spec import DomainSpec


@dataclass
class ValidationResult:
    """Outcome of validating a single candidate."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    code_hash: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "code_hash": self.code_hash,
        }


def _hash_code(code: str) -> str:
    """SHA-256 of the candidate source, normalized on line endings."""
    normalized = code.replace("\r\n", "\n").encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def _check_syntax(code: str, errors: list[str]) -> ast.Module | None:
    try:
        return ast.parse(code)
    except SyntaxError as exc:
        errors.append(f"SyntaxError: {exc.msg} at line {exc.lineno}")
        return None


def _collect_classes(tree: ast.Module) -> dict[str, ast.ClassDef]:
    return {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}


def _class_method_names(node: ast.ClassDef) -> set[str]:
    names: set[str] = set()
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(item.name)
    return names


def _check_interface(
    tree: ast.Module, spec: DomainSpec, errors: list[str], warnings: list[str]
) -> None:
    expected_class = spec.class_name
    if not expected_class:
        # Spec didn't pin a class name — nothing to enforce.
        return
    classes = _collect_classes(tree)

    # Accept either (a) a class literally named expected_class, or (b) any
    # subclass syntactically referencing expected_class as a base.
    chosen: ast.ClassDef | None = None
    if expected_class in classes:
        chosen = classes[expected_class]
    else:
        for cls in classes.values():
            for base in cls.bases:
                base_name = _base_name(base)
                if base_name == expected_class:
                    chosen = cls
                    break
            if chosen is not None:
                break

    if chosen is None:
        errors.append(
            f"No class named {expected_class!r} or subclass of it was found"
        )
        return

    present = _class_method_names(chosen)
    missing = [m for m in spec.required_method_names if m not in present]
    if missing:
        errors.append(
            f"Class {chosen.name!r} missing required methods: {missing}"
        )


def _base_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _check_constraints(
    config: dict[str, Any],
    spec: DomainSpec,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate config values against ``spec.constraints`` bounds."""
    for key, bounds in spec.constraints.items():
        if key not in config:
            continue
        raw = config[key]
        # Coerce strings to numbers when a numeric type is declared.
        declared_type = (bounds.get("type") or "").lower()
        value: Any = raw
        if declared_type in ("int", "integer"):
            try:
                value = int(raw)
            except (TypeError, ValueError):
                errors.append(f"config[{key!r}] expected int, got {raw!r}")
                continue
        elif declared_type in ("float", "number"):
            try:
                value = float(raw)
            except (TypeError, ValueError):
                errors.append(f"config[{key!r}] expected float, got {raw!r}")
                continue

        mn = bounds.get("min")
        mx = bounds.get("max")
        if mn is not None and value < mn:
            errors.append(f"config[{key!r}]={value} below min {mn}")
        if mx is not None and value > mx:
            errors.append(f"config[{key!r}]={value} above max {mx}")

        enum = bounds.get("enum")
        if enum is not None and value not in enum:
            errors.append(
                f"config[{key!r}]={value!r} not in allowed set {enum}"
            )


def validate_candidate(
    code: str,
    spec: DomainSpec,
    *,
    config: dict[str, Any] | None = None,
    seen_hashes: set[str] | None = None,
) -> ValidationResult:
    """Validate a candidate against a domain spec.

    Parameters
    ----------
    code:
        Candidate source code string.
    spec:
        Domain spec defining the expected interface, metrics, constraints.
    config:
        Optional config dict (env var overrides) to constraint-check.
    seen_hashes:
        Set of already-stored candidate hashes. When provided, duplicates
        produce a warning (not an error — exact duplicates are sometimes
        harmless, e.g., re-proposing a known baseline).
    """
    errors: list[str] = []
    warnings: list[str] = []
    code_hash = _hash_code(code)

    tree = _check_syntax(code, errors)
    if tree is not None:
        _check_interface(tree, spec, errors, warnings)

    if config is not None:
        _check_constraints(config, spec, errors, warnings)

    if seen_hashes is not None and code_hash in seen_hashes:
        warnings.append(f"Duplicate of previously seen candidate ({code_hash[:12]})")

    return ValidationResult(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        code_hash=code_hash,
    )


def batch_validate(
    candidates: list[dict[str, Any]],
    spec: DomainSpec,
    *,
    seen_hashes: set[str] | None = None,
    require_any_valid: bool = False,
) -> list[dict[str, Any]]:
    """Validate a list of candidate dicts; return them augmented with results.

    Each input candidate must have a ``code`` key. An optional ``config``
    dict is constraint-checked. The returned list has the same ordering as
    the input, each entry augmented with a ``validation`` field and a
    ``valid`` boolean mirroring ``validation.valid``.

    When ``require_any_valid`` is True and every candidate fails, raises
    :class:`CandidateValidationError` so callers detect the empty-proposal
    case without defensive-checking counts.
    """
    if not candidates:
        return []
    seen = set(seen_hashes) if seen_hashes else set()
    out: list[dict[str, Any]] = []
    for cand in candidates:
        code = cand.get("code", "")
        result = validate_candidate(
            code,
            spec,
            config=cand.get("config"),
            seen_hashes=seen,
        )
        seen.add(result.code_hash)
        augmented = dict(cand)
        augmented["validation"] = result.as_dict()
        augmented["valid"] = result.valid
        out.append(augmented)
    if require_any_valid and not any(c["valid"] for c in out):
        all_errors = [e for c in out for e in c["validation"]["errors"]]
        raise CandidateValidationError(
            f"All {len(out)} candidates failed validation: {all_errors[:3]}"
        )
    return out
