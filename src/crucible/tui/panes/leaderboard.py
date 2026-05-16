"""LeaderboardPane — top-K runs by primary metric.

Reads via ``crucible.analysis.completed_results`` which already merges
local / project / fleet results. Re-fetched on tab activation rather
than on a timer since experiments don't change minute-to-minute.
"""
from __future__ import annotations

from typing import Any

from textual.containers import Vertical
from textual.widgets import DataTable, Label

from crucible.core.config import ProjectConfig
from crucible.core.log import log_warn


class LeaderboardPane(Vertical):
    """Sortable leaderboard table."""

    DEFAULT_CSS = """
    LeaderboardPane { padding: 1 2; }
    LeaderboardPane > DataTable { height: 1fr; }
    LeaderboardPane > #lb-summary {
        height: 1;
        padding: 0 1;
        background: $boost;
    }
    """

    def __init__(self, config: ProjectConfig, top_n: int = 25) -> None:
        super().__init__()
        self._config = config
        self._top_n = top_n
        self._columns_built = False

    def compose(self):
        yield Label("Loading leaderboard…", id="lb-summary")
        yield DataTable(id="lb-table", zebra_stripes=True)

    def on_mount(self) -> None:
        self.refresh_data()

    def refresh_data(self) -> None:
        try:
            entries = self._compute_leaderboard()
        except Exception as exc:
            log_warn(f"LeaderboardPane: refresh failed: {exc}")
            self.query_one("#lb-summary", Label).update(
                f"[red]Refresh failed: {exc}[/red]"
            )
            return

        primary = self._config.metrics.primary
        secondary = getattr(self._config.metrics, "secondary", "") or ""
        direction = getattr(self._config.metrics, "direction", "minimize")

        table = self.query_one("#lb-table", DataTable)
        table.clear(columns=True)

        cols = ["Rank", "Name", primary]
        if secondary:
            cols.append(secondary)
        cols += ["Bytes", "Steps"]
        table.add_columns(*cols)

        if not entries:
            self.query_one("#lb-summary", Label).update(
                "[dim]No completed experiments yet.[/dim]"
            )
            return

        for entry in entries:
            primary_val = entry.get("primary_value")
            primary_str = _fmt(primary_val)
            row = [
                str(entry.get("rank", "?")),
                (entry.get("name") or "?")[:32],
                primary_str,
            ]
            if secondary:
                row.append(_fmt(entry.get("secondary_value")))
            row.append(_fmt_bytes(entry.get("model_bytes")))
            row.append(str(entry.get("steps_completed") or "—"))
            table.add_row(*row)

        self.query_one("#lb-summary", Label).update(
            f"[bold]{len(entries)} ranked[/bold] · primary [cyan]{primary}[/cyan] "
            f"({direction})"
        )

    def _compute_leaderboard(self) -> list[dict[str, Any]]:
        from crucible.analysis.leaderboard import leaderboard
        from crucible.analysis.results import completed_results

        results = completed_results(self._config)
        if not results:
            return []
        primary = self._config.metrics.primary
        secondary = getattr(self._config.metrics, "secondary", "") or ""
        direction = getattr(self._config.metrics, "direction", "minimize")
        ranked = leaderboard(
            results,
            top_n=self._top_n,
            metric=primary,
            direction=direction,
        )
        out: list[dict[str, Any]] = []
        for rank, r in enumerate(ranked, start=1):
            metrics_dict = r.get("result") or {}
            primary_val = metrics_dict.get(primary) if isinstance(metrics_dict, dict) else None
            secondary_val = (
                metrics_dict.get(secondary)
                if (secondary and isinstance(metrics_dict, dict))
                else None
            )
            out.append({
                "rank": rank,
                "name": r.get("name"),
                "primary_value": primary_val,
                "secondary_value": secondary_val,
                "model_bytes": r.get("model_bytes"),
                "steps_completed": (
                    metrics_dict.get("steps_completed")
                    if isinstance(metrics_dict, dict) else None
                ),
            })
        return out


def _fmt(value: Any) -> str:
    if value is None:
        return "[dim]—[/dim]"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _fmt_bytes(value: Any) -> str:
    if not isinstance(value, (int, float)) or value <= 0:
        return "[dim]—[/dim]"
    mb = value / 1_000_000
    return f"{mb:.1f}M"
