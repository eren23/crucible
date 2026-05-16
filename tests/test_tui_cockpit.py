"""Tests for the TUI cockpit panes — Phase 2.3.

Textual ships a ``Pilot`` test runner that can mount the app without a
real terminal, drive key/mouse events programmatically, and query the
widget tree. The tests below boot the app against a fixture project
with seeded nodes/queue/results, switch each tab, and assert the right
data appears.

The interactive UX itself (colors, alignment, focus rings) still needs
human verification with ``crucible tui`` — these tests cover the data
plumbing, not the visual rendering.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest


# Textual's test mode requires asyncio. pytest-asyncio is not in the
# project's hard deps, so wrap each coroutine in asyncio.run() via a
# small adapter — the tests stay synchronous from pytest's POV.
pytest.importorskip("textual")


def _run(coro):
    """Synchronous wrapper around an async test body."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Project fixture
# ---------------------------------------------------------------------------


PROJECT_YAML = """\
name: tui-cockpit-test
version: "0.2.1-alpha"

provider:
  type: ssh
  ssh_key: ~/.ssh/id_ed25519

training:
  - backend: torch
    script: train.py

metrics:
  primary: val_loss
  direction: minimize

results_file: experiments.jsonl
fleet_results_file: experiments_fleet.jsonl
logs_dir: logs
store_dir: .crucible
research_state_file: research_state.jsonl
nodes_file: nodes.json

wandb:
  required: false

execution_policy:
  require_remote: false
  required_provider: ""
  allow_local_dev: true
"""


def _write_node(rows: list[dict[str, Any]], path: Path) -> None:
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def project_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A project dir with seeded nodes, queue, and results."""
    (tmp_path / "crucible.yaml").write_text(PROJECT_YAML, encoding="utf-8")
    (tmp_path / "logs").mkdir(exist_ok=True)
    (tmp_path / ".crucible").mkdir(exist_ok=True)

    # Nodes (load_nodes expects a list of dicts under "nodes" or top-level
    # array — defer to whatever the loader supports; the fleet pane reads
    # the same path).
    _write_node(
        [
            {"name": "pod-a", "state": "ready", "gpu": "4090",
             "ssh_host": "1.2.3.4", "env_ready": True, "dataset_ready": True},
            {"name": "pod-b", "state": "starting", "gpu": "A6000",
             "ssh_host": None, "env_ready": False, "dataset_ready": False},
        ],
        tmp_path / "nodes.json",
    )

    # Queue
    _write_jsonl(
        [
            {"run_id": "run-1", "name": "exp-a", "tier": "smoke",
             "lease_state": "running", "node_name": "pod-a",
             "updated_at": "2026-05-16T10:00:00Z"},
            {"run_id": "run-2", "name": "exp-b", "tier": "proxy",
             "lease_state": "queued", "node_name": None,
             "updated_at": "2026-05-16T10:01:00Z"},
        ],
        tmp_path / "fleet_queue.jsonl",
    )

    # Results (one completed for leaderboard)
    _write_jsonl(
        [{
            "id": "exp-a",
            "name": "exp-a",
            "status": "completed",
            "config": {"LR": "0.001"},
            "result": {"val_loss": 1.23, "steps_completed": 500},
            "model_bytes": 14000000,
            "backend": "torch",
        }],
        tmp_path / "experiments.jsonl",
    )

    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Boot + tab navigation
# ---------------------------------------------------------------------------


def test_cockpit_boots_with_five_tabs(project_dir):
    async def _body():
        """All five tabs mount without exception. Designs is the initial tab."""
        from textual.widgets import TabbedContent

        from crucible.tui.app import CrucibleApp

        app = CrucibleApp()
        async with app.run_test() as pilot:
            tabbed = app.query_one("#cockpit", TabbedContent)
            # Tab IDs in declaration order.
            expected = ["designs", "fleet", "queue", "leaderboard", "briefing"]
            actual = [pane.id for pane in tabbed.query("TabPane")]
            assert actual == expected
            assert tabbed.active == "designs"
            await pilot.pause()


    _run(_body())
def test_number_keys_switch_tabs(project_dir):
    async def _body():
        """Pressing 1/2/3/4/5 activates the matching tab."""
        from textual.widgets import TabbedContent

        from crucible.tui.app import CrucibleApp

        app = CrucibleApp()
        async with app.run_test() as pilot:
            tabbed = app.query_one("#cockpit", TabbedContent)
            for key, expected in [
                ("2", "fleet"),
                ("3", "queue"),
                ("4", "leaderboard"),
                ("5", "briefing"),
                ("1", "designs"),
            ]:
                await pilot.press(key)
                await pilot.pause()
                assert tabbed.active == expected, f"key {key} → expected {expected}"


    _run(_body())
# ---------------------------------------------------------------------------
# Pane data plumbing
# ---------------------------------------------------------------------------


def test_fleet_pane_shows_nodes(project_dir):
    async def _body():
        from textual.widgets import DataTable

        from crucible.tui.app import CrucibleApp

        app = CrucibleApp()
        async with app.run_test() as pilot:
            await pilot.press("2")
            await pilot.pause()
            table = app.query_one("#fleet-table", DataTable)
            assert table.row_count == 2
            # First column is the node name.
            cell_a = table.get_cell_at((0, 0))
            cell_b = table.get_cell_at((1, 0))
            assert {cell_a, cell_b} == {"pod-a", "pod-b"}


    _run(_body())
def test_queue_pane_shows_rows(project_dir):
    async def _body():
        from textual.widgets import DataTable

        from crucible.tui.app import CrucibleApp

        app = CrucibleApp()
        async with app.run_test() as pilot:
            await pilot.press("3")
            await pilot.pause()
            table = app.query_one("#queue-table", DataTable)
            assert table.row_count == 2


    _run(_body())
def test_leaderboard_pane_shows_completed_run(project_dir):
    async def _body():
        from textual.widgets import DataTable

        from crucible.tui.app import CrucibleApp

        app = CrucibleApp()
        async with app.run_test() as pilot:
            await pilot.press("4")
            await pilot.pause()
            table = app.query_one("#lb-table", DataTable)
            assert table.row_count == 1
            # Rank, name, primary metric (val_loss=1.23)
            rank = table.get_cell_at((0, 0))
            name = table.get_cell_at((0, 1))
            metric = table.get_cell_at((0, 2))
            assert rank == "1"
            assert name == "exp-a"
            assert "1.2" in str(metric)


    _run(_body())
def test_briefing_pane_renders_markdown(project_dir):
    async def _body():
        from textual.widgets import Markdown

        from crucible.tui.app import CrucibleApp

        app = CrucibleApp()
        async with app.run_test() as pilot:
            await pilot.press("5")
            await pilot.pause()
            md = app.query_one("#briefing-md", Markdown)
            # Markdown widget has no public text accessor; check that it
            # has a non-empty source via the underlying attribute.
            assert getattr(md, "_markdown", None), "Markdown body should be populated"


    _run(_body())
# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


def test_empty_project_does_not_crash(tmp_path, monkeypatch):
    async def _body():
        """All four panes must render even on a fresh project with no
        nodes / queue / results."""
        from textual.widgets import DataTable

        (tmp_path / "crucible.yaml").write_text(PROJECT_YAML, encoding="utf-8")
        (tmp_path / "logs").mkdir(exist_ok=True)
        (tmp_path / ".crucible").mkdir(exist_ok=True)
        monkeypatch.chdir(tmp_path)

        from crucible.tui.app import CrucibleApp

        app = CrucibleApp()
        async with app.run_test() as pilot:
            for key in ("2", "3", "4"):
                await pilot.press(key)
                await pilot.pause()
            # All three data tables exist; row_count==0 is fine.
            for tid in ("#fleet-table", "#queue-table", "#lb-table"):
                table = app.query_one(tid, DataTable)
                assert table.row_count == 0

    _run(_body())