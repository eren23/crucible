"""Predicate-based search over the experiment-results ledger.

Backs the `runs_search` MCP tool (Phase 2.5). Lets agents filter runs
with a SQL-ish expression like::

    val_loss < 2.0 and model_dim > 256 and name == 'smoke'

The expression is parsed via Python's ``ast`` module with a strict
whitelist of node types — no function calls, no attribute access on
arbitrary objects, no imports, no exec. Identifiers resolve against
each row dict via dotted access (``result.val_loss`` reads
``row['result']['val_loss']``); top-level config keys (the env vars
that drove the run) are also reachable directly (``model_dim``,
``LR``, etc.) for convenience.
"""
from __future__ import annotations

import ast
import re
from typing import Any, Callable

from crucible.core.config import ProjectConfig
from crucible.core.errors import CrucibleError


class SearchError(CrucibleError):
    """Raised when the predicate is malformed or references unknown columns."""


_ALLOWED_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Name,
    ast.Constant,
    ast.Attribute,
    ast.Load,
    ast.List,
    ast.Tuple,
    ast.Set,
)


def _validate(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise SearchError(
                f"Unsupported expression: {type(node).__name__}. "
                "Only comparisons, boolean ops, identifiers, and literals are allowed."
            )


def _resolve(row: dict[str, Any], path: list[str]) -> Any:
    """Walk a dotted path into the row dict, returning None on miss.

    Also folds the row's ``config`` sub-dict into the top-level namespace
    so callers can write ``model_dim`` instead of ``config.model_dim``.
    Explicit dotted access still works for disambiguation.
    """
    current: Any = row
    for i, key in enumerate(path):
        if isinstance(current, dict):
            if key in current:
                current = current[key]
                continue
            # Top-level miss: check the row's "config" sub-dict for env-var
            # style keys (LR, MODEL_FAMILY, etc.).
            if i == 0 and "config" in current and isinstance(current["config"], dict):
                if key in current["config"]:
                    current = current["config"][key]
                    continue
                # Case-insensitive fallback for env-var keys (model_dim → MODEL_DIM).
                upper = key.upper()
                if upper in current["config"]:
                    current = current["config"][upper]
                    continue
            return None
        return None
    return current


def _evaluate(node: ast.AST, row: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, row)
    if isinstance(node, ast.BoolOp):
        values = [_evaluate(v, row) for v in node.values]
        if isinstance(node.op, ast.And):
            return all(values)
        return any(values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _evaluate(node.operand, row)
    if isinstance(node, ast.Compare):
        left = _evaluate(node.left, row)
        for op, right_node in zip(node.ops, node.comparators):
            right = _evaluate(right_node, row)
            if not _apply_op(op, left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Name):
        return _resolve(row, [node.id])
    if isinstance(node, ast.Attribute):
        path = _flatten_attribute(node)
        return _resolve(row, path)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [_evaluate(e, row) for e in node.elts]
    raise SearchError(f"Unsupported node during evaluation: {type(node).__name__}")


def _flatten_attribute(node: ast.Attribute) -> list[str]:
    path: list[str] = []
    current: Any = node
    while isinstance(current, ast.Attribute):
        path.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        path.append(current.id)
    else:
        raise SearchError("Attribute chains must terminate in an identifier.")
    return list(reversed(path))


def _apply_op(op: ast.AST, left: Any, right: Any) -> bool:
    if isinstance(op, ast.Eq):
        return left == right
    if isinstance(op, ast.NotEq):
        return left != right
    if isinstance(op, ast.Lt):
        return _safe_lt(left, right)
    if isinstance(op, ast.LtE):
        return left == right or _safe_lt(left, right)
    if isinstance(op, ast.Gt):
        return _safe_lt(right, left)
    if isinstance(op, ast.GtE):
        return left == right or _safe_lt(right, left)
    if isinstance(op, ast.In):
        try:
            return left in right
        except TypeError:
            return False
    if isinstance(op, ast.NotIn):
        try:
            return left not in right
        except TypeError:
            return False
    raise SearchError(f"Unsupported comparator: {type(op).__name__}")


def _safe_lt(a: Any, b: Any) -> bool:
    """Numeric-aware <. Strings compare lex, numbers compare numerically.
    Mismatched types coerce strings to numbers if possible, else return
    False (no error — search results should degrade not explode)."""
    if a is None or b is None:
        return False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a < b
    if isinstance(a, str) and isinstance(b, str):
        return a < b
    # Mixed: try numeric coercion both ways.
    try:
        return float(a) < float(b)
    except (TypeError, ValueError):
        return False


def compile_predicate(expr: str) -> Callable[[dict[str, Any]], bool]:
    """Return a callable that evaluates the predicate against a row dict.

    Raises SearchError if the expression is malformed or uses
    disallowed syntax.
    """
    if not expr.strip():
        return lambda _row: True
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise SearchError(f"Predicate syntax error: {exc.msg}") from exc
    _validate(tree)
    return lambda row: bool(_evaluate(tree, row))


def search_runs(
    config: ProjectConfig,
    *,
    where: str = "",
    order_by: str = "",
    direction: str = "asc",
    limit: int | None = 50,
    source: str = "merged",
    select: list[str] | None = None,
) -> dict[str, Any]:
    """Filter + sort the run ledger.

    Parameters
    ----------
    where:
        Predicate expression. Empty string returns all rows.
    order_by:
        Field name (dotted access supported). Empty string preserves
        ledger order.
    direction:
        ``"asc"`` or ``"desc"``.
    limit:
        Maximum number of rows to return. ``None`` returns all.
    source:
        ``"local"``, ``"project"``, ``"fleet"``, or ``"merged"`` (default).
    select:
        Optional list of columns to keep. Each entry can be a dotted
        path. ``None`` returns the full row.

    Returns
    -------
    dict with keys: ``matched`` (int), ``returned`` (int), ``rows`` (list).
    """
    from crucible.analysis.results import load_results, merged_results

    if source == "merged":
        rows = merged_results(config)
    elif source in {"local", "project", "fleet"}:
        rows = load_results(config, source=source)
    else:
        raise SearchError(
            f"Unknown source {source!r}; expected one of merged/local/project/fleet."
        )

    predicate = compile_predicate(where)
    matched_rows = [r for r in rows if predicate(r)]

    if order_by:
        path = order_by.split(".")
        reverse = direction.lower() == "desc"
        matched_rows.sort(
            key=lambda r: _sort_key(_resolve(r, path)),
            reverse=reverse,
        )

    returned = matched_rows
    if limit is not None and limit >= 0:
        returned = matched_rows[:limit]

    if select:
        returned = [_project(r, select) for r in returned]

    return {
        "matched": len(matched_rows),
        "returned": len(returned),
        "rows": returned,
    }


def _sort_key(value: Any) -> tuple[int, Any]:
    """Stable sort key that handles None + mixed types without raising."""
    if value is None:
        return (1, 0)  # None always sorts last
    if isinstance(value, (int, float)):
        return (0, value)
    return (0, str(value))


def _project(row: dict[str, Any], select: list[str]) -> dict[str, Any]:
    """Keep only the named columns. Dotted paths nest under the leaf key."""
    out: dict[str, Any] = {}
    for col in select:
        path = col.split(".")
        value = _resolve(row, path)
        # Use the last segment as the key so callers get a flat dict.
        out[path[-1]] = value
    return out
