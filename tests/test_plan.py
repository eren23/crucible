"""Unit tests for core/plan.py."""
from __future__ import annotations

import json

import pytest

from crucible.core.errors import PlanError
from crucible.core.plan import PlanItem, PlanStore


@pytest.fixture
def plan_path(tmp_path):
    return tmp_path / "plan.json"


def test_empty_plan_on_missing_file(plan_path):
    store = PlanStore(plan_path)
    assert store.get() == []
    assert store.as_dicts() == []


def test_set_and_get_roundtrip(plan_path):
    store = PlanStore(plan_path)
    written = store.set([
        {"description": "step one"},
        {"description": "step two", "status": "in_progress"},
    ])
    assert len(written) == 2
    loaded = store.get()
    assert [i.description for i in loaded] == ["step one", "step two"]
    assert loaded[1].status == "in_progress"


def test_atomic_write_produces_json(plan_path):
    store = PlanStore(plan_path)
    store.set([{"description": "x"}])
    raw = json.loads(plan_path.read_text(encoding="utf-8"))
    assert "items" in raw
    assert "updated_at" in raw
    assert len(raw["items"]) == 1


def test_invariant_two_in_progress(plan_path):
    store = PlanStore(plan_path)
    with pytest.raises(PlanError, match="in_progress"):
        store.set(
            [
                {"description": "a", "status": "in_progress"},
                {"description": "b", "status": "in_progress"},
            ]
        )


def test_invariant_empty_description(plan_path):
    store = PlanStore(plan_path)
    with pytest.raises(PlanError, match="empty description"):
        store.set([{"description": ""}])


def test_invariant_duplicate_ids(plan_path):
    store = PlanStore(plan_path)
    with pytest.raises(PlanError, match="duplicate"):
        store.set(
            [
                {"id": "abc", "description": "a"},
                {"id": "abc", "description": "b"},
            ]
        )


def test_update_item_enforces_invariant(plan_path):
    store = PlanStore(plan_path)
    items = store.set(
        [
            {"description": "a", "status": "in_progress"},
            {"description": "b", "status": "pending"},
        ]
    )
    b_id = items[1].id
    with pytest.raises(PlanError, match="in_progress"):
        store.update_item(b_id, "in_progress")


def test_update_item_unknown_id(plan_path):
    store = PlanStore(plan_path)
    store.set([{"description": "only"}])
    with pytest.raises(PlanError, match="not found"):
        store.update_item("no-such-id", "completed")


def test_update_item_invalid_status(plan_path):
    store = PlanStore(plan_path)
    items = store.set([{"description": "only"}])
    with pytest.raises(PlanError, match="Invalid status"):
        store.update_item(items[0].id, "weird-status")


def test_clear(plan_path):
    store = PlanStore(plan_path)
    store.set([{"description": "a"}])
    assert len(store.get()) == 1
    store.clear()
    assert store.get() == []


def test_auto_id_assignment(plan_path):
    store = PlanStore(plan_path)
    items = store.set([{"description": "x"}, {"description": "y"}])
    assert items[0].id and items[1].id
    assert items[0].id != items[1].id


def test_plan_item_from_dict_rejects_unknown_status():
    with pytest.raises(PlanError, match="Invalid plan item status"):
        PlanItem.from_dict({"description": "x", "status": "????"})
