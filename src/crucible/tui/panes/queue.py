"""QueuePane — fleet queue + recent dispatch state.

Reads ``fleet_queue.jsonl`` and renders it as a sortable DataTable.
Live-refreshes every 5 seconds so an operator running a dispatch loop
sees rows move queued → running → finished without having to reload.
"""
from __future__ import annotations

from typing import Any

from textual.containers import Vertical
from textual.widgets import DataTable, Label

from crucible.core.config import ProjectConfig
from crucible.core.errors import CrucibleError
from crucible.core.log import log_warn


_COLUMNS = ["Run ID", "Name", "Tier", "Lease", "Node", "Updated"]


class QueuePane(Vertical):
    """Live fleet queue table."""

    DEFAULT_CSS = """
    QueuePane { padding: 1 2; }
    QueuePane > DataTable { height: 1fr; }
    QueuePane > #queue-summary {
        height: 1;
        padding: 0 1;
        background: $boost;
    }
    """

    def __init__(self, config: ProjectConfig, refresh_seconds: float = 5.0) -> None:
        super().__init__()
        self._config = config
        self._refresh_seconds = refresh_seconds

    def compose(self):
        yield Label("Loading queue…", id="queue-summary")
        table = DataTable(id="queue-table", zebra_stripes=True)
        table.add_columns(*_COLUMNS)
        yield table

    def on_mount(self) -> None:
        self.refresh_data()
        if self._refresh_seconds > 0:
            self.set_interval(self._refresh_seconds, self.refresh_data)

    def refresh_data(self) -> None:
        # Narrowed exception set — let unexpected errors propagate
        # rather than getting swallowed in a timer callback.
        try:
            rows = self._load_queue()
        except (CrucibleError, OSError, ValueError) as exc:
            log_warn(f"QueuePane: refresh failed: {exc}")
            self.query_one("#queue-summary", Label).update(
                f"[red]Refresh failed: {exc}[/red]"
            )
            return

        table = self.query_one("#queue-table", DataTable)
        table.clear()
        if not rows:
            self.query_one("#queue-summary", Label).update(
                "[dim]Queue is empty. Use `crucible run enqueue` to add experiments.[/dim]"
            )
            return

        # Sort: running → queued → finished/failed (visual priority).
        priority = {"running": 0, "queued": 1, "retryable": 2,
                    "completed": 3, "finished": 3, "failed": 4}
        rows_sorted = sorted(
            rows,
            key=lambda r: priority.get((r.get("lease_state") or "").lower(), 5),
        )

        summary = {"running": 0, "queued": 0, "finished": 0, "failed": 0}
        for r in rows_sorted:
            lease = (r.get("lease_state") or "queued").lower()
            if lease == "running":
                summary["running"] += 1
            elif lease == "queued":
                summary["queued"] += 1
            elif lease in {"completed", "finished"}:
                summary["finished"] += 1
            elif lease in {"failed", "error", "canceled", "retryable"}:
                summary["failed"] += 1
            updated = (r.get("updated_at") or "")[:19]
            table.add_row(
                (r.get("run_id") or "?")[:20],
                (r.get("name") or "?")[:24],
                r.get("tier") or "[dim]—[/dim]",
                _color_lease(lease),
                r.get("node_name") or r.get("remote_node") or "[dim]—[/dim]",
                updated or "[dim]—[/dim]",
            )

        self.query_one("#queue-summary", Label).update(
            f"[bold]{len(rows)}[/bold] total · "
            f"[dodger_blue1]{summary['running']} running[/dodger_blue1] · "
            f"[yellow]{summary['queued']} queued[/yellow] · "
            f"[green]{summary['finished']} finished[/green]"
            + (f" · [red]{summary['failed']} failed[/red]" if summary['failed'] else "")
        )

    def _load_queue(self) -> list[dict[str, Any]]:
        from crucible.fleet.queue import load_queue
        try:
            return load_queue(self._config.project_root / "fleet_queue.jsonl")
        except FileNotFoundError:
            return []


def _color_lease(state: str) -> str:
    palette = {
        "running": "dodger_blue1",
        "queued": "yellow",
        "completed": "green",
        "finished": "green",
        "failed": "red",
        "canceled": "bright_black",
        "error": "red",
        "retryable": "magenta",
    }
    color = palette.get(state, "white")
    return f"[{color}]{state}[/{color}]"
