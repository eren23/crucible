"""BriefingPane — research briefing + suggested-actions cockpit.

Renders the markdown_summary from ``build_briefing`` plus the
machine-readable ``next_actions`` block from Phase 2.2 as a scrollable
markdown view. Refreshes on the ``r`` keypress (Textual's
``action_refresh`` binding).
"""
from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Label, Markdown

from crucible.core.config import ProjectConfig
from crucible.core.log import log_warn


class BriefingPane(Vertical):
    """Markdown briefing + recommended next tool.

    Refresh is driven by the app-level ``r`` binding (see
    ``CrucibleApp.action_refresh_active_pane``) — TabbedContent keeps
    focus on its content-tab strip, so pane-scoped bindings never
    receive the keypress directly.
    """

    DEFAULT_CSS = """
    BriefingPane { padding: 1 2; }
    BriefingPane > #briefing-summary {
        height: 1;
        padding: 0 1;
        background: $boost;
    }
    BriefingPane > VerticalScroll { height: 1fr; }
    """

    def __init__(self, config: ProjectConfig) -> None:
        super().__init__()
        self._config = config

    def compose(self) -> ComposeResult:
        yield Label("Loading briefing…", id="briefing-summary")
        with VerticalScroll(id="briefing-scroll"):
            yield Markdown("", id="briefing-md")

    def on_mount(self) -> None:
        self.refresh_data()

    def refresh_data(self) -> None:
        try:
            briefing = self._build()
        except Exception as exc:
            log_warn(f"BriefingPane: refresh failed: {exc}")
            self.query_one("#briefing-summary", Label).update(
                f"[red]Refresh failed: {exc}[/red]"
            )
            return

        md_widget = self.query_one("#briefing-md", Markdown)
        markdown = briefing.get("markdown_summary") or ""
        # Append a structured "next action" line at the bottom so the
        # recommendation jumps out even on tall briefings.
        na = briefing.get("next_actions") or {}
        rec = na.get("recommended_tool")
        if rec:
            markdown += (
                f"\n\n---\n\n"
                f"## Next call (from tool_router)\n\n"
                f"**`{rec}`** — {na.get('rationale', '')}\n"
            )
            for alt in (na.get("alternatives") or []):
                markdown += f"- alt: `{alt.get('tool')}` — {alt.get('rationale', '')}\n"
        md_widget.update(markdown)

        proj = briefing.get("project") or {}
        leaderboard = briefing.get("leaderboard_top3") or []
        self.query_one("#briefing-summary", Label).update(
            f"[bold]{proj.get('name', '?')}[/bold] · "
            f"primary [cyan]{proj.get('primary_metric', '?')}[/cyan] · "
            f"{len(leaderboard)} on leaderboard · "
            f"[dim]press r to refresh[/dim]"
        )

    def _build(self) -> dict[str, Any]:
        from crucible.researcher.briefing import build_briefing
        return build_briefing(self._config)
