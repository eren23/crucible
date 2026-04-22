"""LLM-facing todo-list / plan store.

Mirrors the ``plan`` tool in HuggingFace ml-intern: a flat list of plan
items with statuses ``pending`` / ``in_progress`` / ``completed``, and
an enforced invariant that at most one item is ``in_progress`` at a
time. The store is file-backed at ``.crucible/plan.json`` (atomic
writes) so the same plan is visible across an MCP session, a
``crucible chat`` session, and the autonomous researcher.

The canonical API mirrors ml-intern's: callers send the full replacement
list with :meth:`set`; individual status flips use :meth:`update_item`.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from crucible.core.errors import PlanError
from crucible.core.io import atomic_write_json
from crucible.core.log import utc_now_iso


PlanStatus = Literal["pending", "in_progress", "completed"]
_VALID_STATUSES: tuple[PlanStatus, ...] = ("pending", "in_progress", "completed")


@dataclass
class PlanItem:
    id: str
    description: str
    status: PlanStatus = "pending"
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PlanItem":
        status = raw.get("status", "pending")
        if status not in _VALID_STATUSES:
            raise PlanError(f"Invalid plan item status: {status!r}")
        return cls(
            id=str(raw.get("id") or _new_id()),
            description=str(raw.get("description", "")).strip(),
            status=status,  # type: ignore[arg-type]
            created_at=str(raw.get("created_at") or utc_now_iso()),
            updated_at=str(raw.get("updated_at") or utc_now_iso()),
        )


class PlanStore:
    """File-backed plan store.

    The on-disk format is ``{"items": [PlanItem...], "updated_at": iso}``.
    Missing files are treated as an empty plan.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self) -> list[PlanItem]:
        if not self.path.exists():
            return []
        import json

        raw = json.loads(self.path.read_text(encoding="utf-8"))
        items_raw = raw.get("items", [])
        return [PlanItem.from_dict(i) for i in items_raw]

    def as_dicts(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.get()]

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def set(self, items: list[dict[str, Any] | PlanItem]) -> list[PlanItem]:
        """Replace the entire plan with *items*. Enforces invariants."""
        normalized = [_to_item(i) for i in items]
        self._validate_invariants(normalized)
        self._write(normalized)
        return normalized

    def update_item(self, item_id: str, status: PlanStatus) -> PlanItem:
        """Flip one item's status. Enforces invariants."""
        if status not in _VALID_STATUSES:
            raise PlanError(f"Invalid status: {status!r}")
        items = self.get()
        found = next((i for i in items if i.id == item_id), None)
        if found is None:
            raise PlanError(f"Plan item not found: {item_id!r}")
        found.status = status
        found.updated_at = utc_now_iso()
        self._validate_invariants(items)
        self._write(items)
        return found

    def clear(self) -> None:
        self._write([])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_invariants(items: list[PlanItem]) -> None:
        in_progress = [i for i in items if i.status == "in_progress"]
        if len(in_progress) > 1:
            raise PlanError(
                f"Plan invariant violated: {len(in_progress)} items are in_progress; "
                "exactly one may be in_progress at a time."
            )
        ids = [i.id for i in items]
        if len(ids) != len(set(ids)):
            raise PlanError("Plan invariant violated: duplicate item IDs")
        for item in items:
            if not item.description:
                raise PlanError(f"Plan item {item.id!r} has empty description")

    def _write(self, items: list[PlanItem]) -> None:
        payload = {
            "items": [i.to_dict() for i in items],
            "updated_at": utc_now_iso(),
        }
        atomic_write_json(self.path, payload)


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def _to_item(raw: dict[str, Any] | PlanItem) -> PlanItem:
    if isinstance(raw, PlanItem):
        return raw
    if not isinstance(raw, dict):
        raise PlanError(f"Plan item must be a dict or PlanItem, got {type(raw).__name__}")
    if "description" not in raw:
        raise PlanError("Plan item missing 'description'")
    return PlanItem.from_dict(raw)
