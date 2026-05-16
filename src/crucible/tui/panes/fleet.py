"""FleetPane — live pod status, SSH endpoints, env-readiness.

Polls the project's nodes.json + the active project runs ledger. Updates
every 10 seconds via Textual's set_interval. Live GPU/memory probing is
left off the hot path because it requires SSH and would block the
event loop; humans wanting that data should run get_fleet_status with
include_metrics=true from the CLI.
"""
from __future__ import annotations

from typing import Any

from textual.containers import Vertical
from textual.widgets import DataTable, Label, Static

from crucible.core.config import ProjectConfig
from crucible.core.errors import CrucibleError
from crucible.core.log import log_warn


_COLUMNS = ["Node", "State", "GPU", "SSH", "Env", "Data", "Runs"]


class FleetPane(Vertical):
    """Live fleet inventory table."""

    DEFAULT_CSS = """
    FleetPane { padding: 1 2; }
    FleetPane > DataTable { height: 1fr; }
    FleetPane > #fleet-summary {
        height: 1;
        padding: 0 1;
        background: $boost;
    }
    """

    def __init__(self, config: ProjectConfig, refresh_seconds: float = 10.0) -> None:
        super().__init__()
        self._config = config
        self._refresh_seconds = refresh_seconds

    def compose(self):
        yield Label("Loading fleet…", id="fleet-summary")
        table = DataTable(id="fleet-table", zebra_stripes=True)
        table.add_columns(*_COLUMNS)
        yield table

    def on_mount(self) -> None:
        self.refresh_data()
        if self._refresh_seconds > 0:
            self.set_interval(self._refresh_seconds, self.refresh_data)

    def refresh_data(self) -> None:
        # Narrowed exception set — let genuinely unexpected errors
        # (programming bugs) propagate so they surface in tests
        # rather than getting silently swallowed in a timer callback.
        try:
            nodes = self._load_nodes()
            active_runs_by_node = self._load_active_runs_by_node()
        except (CrucibleError, OSError, ValueError) as exc:
            log_warn(f"FleetPane: refresh failed: {exc}")
            self.query_one("#fleet-summary", Label).update(
                f"[red]Refresh failed: {exc}[/red]"
            )
            return

        table = self.query_one("#fleet-table", DataTable)
        table.clear()
        if not nodes:
            self.query_one("#fleet-summary", Label).update(
                "[dim]No pods provisioned. Run `crucible fleet provision --count N` to start.[/dim]"
            )
            return

        ready = 0
        for n in nodes:
            state = (n.get("state") or "?").lower()
            ssh_host = n.get("ssh_host") or "[dim]—[/dim]"
            env_ready = _check(n.get("env_ready"))
            data_ready = _check(n.get("dataset_ready"))
            run_count = len(active_runs_by_node.get(n.get("name") or "", []))
            run_str = str(run_count) if run_count else "[dim]0[/dim]"
            if env_ready == "✓" and ssh_host != "[dim]—[/dim]":
                ready += 1
            table.add_row(
                n.get("name", "?"),
                _color_state(state),
                n.get("gpu") or "[dim]—[/dim]",
                ssh_host,
                env_ready,
                data_ready,
                run_str,
            )
        active_total = sum(len(v) for v in active_runs_by_node.values())
        self.query_one("#fleet-summary", Label).update(
            f"[bold]{len(nodes)} pods[/bold] · "
            f"[green]{ready} ready[/green] · "
            f"[dodger_blue1]{active_total} active runs[/dodger_blue1]"
        )

    def _load_nodes(self) -> list[dict[str, Any]]:
        from crucible.fleet.inventory import load_nodes
        try:
            return load_nodes(self._config.project_root / self._config.nodes_file)
        except FileNotFoundError:
            return []

    def _load_active_runs_by_node(self) -> dict[str, list[dict[str, Any]]]:
        """Group active project runs by node_name."""
        path = self._config.project_root / ".crucible" / "projects" / "runs.jsonl"
        if not path.exists():
            return {}
        from crucible.core.io import read_jsonl
        try:
            rows = read_jsonl(path)
        except Exception:
            return {}
        out: dict[str, list[dict[str, Any]]] = {}
        ACTIVE = {"queued", "running", "starting"}
        for r in rows:
            if (r.get("status") or "").lower() not in ACTIVE:
                continue
            key = r.get("node_name") or r.get("remote_node") or "?"
            out.setdefault(key, []).append(r)
        return out


def _check(value: Any) -> str:
    return "[green]✓[/green]" if value else "[dim]✗[/dim]"


def _color_state(state: str) -> str:
    palette = {
        "running": "dodger_blue1",
        "ready": "green",
        "starting": "yellow",
        "stopped": "bright_black",
        "destroyed": "red",
        "terminated": "red",
        "failed": "red",
    }
    color = palette.get(state, "white")
    return f"[{color}]{state}[/{color}]"
