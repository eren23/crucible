"""Crucible TUI — interactive experiment design browser.

Launch:  crucible tui
         PYTHONPATH=src python -m crucible.tui.app
         PYTHONPATH=src python -m crucible.tui.app --screenshots docs/images
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Label, ListItem, ListView, Static

from crucible.core.config import load_config
from crucible.core.store import VersionStore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATUS_COLORS: dict[str, str] = {
    "draft": "yellow",
    "ready": "green",
    "running": "dodger_blue",
    "completed": "bright_green",
    "archived": "dim",
}

STATUS_DOTS: dict[str, str] = {
    "draft": "yellow",
    "ready": "green",
    "running": "dodger_blue",
    "completed": "bright_green",
    "archived": "bright_black",
}

GROUP_LABELS: dict[str, str] = {
    "activation-sweep": "Activation Sweep",
    "competition-sota": "Competition SOTA",
    "qlabs": "QLabs-Inspired",
}

SEPARATOR = "[dim]" + "\u2500" * 44 + "[/dim]"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_store() -> VersionStore:
    config = load_config()
    return VersionStore(config.project_root / config.store_dir)


def _load_designs(store: VersionStore) -> list[dict[str, Any]]:
    designs = []
    for meta in store.list_resources("experiment_design"):
        result = store.get_current("experiment_design", meta["resource_name"])
        if result:
            m, content = result
            designs.append({"meta": m, "content": content})
    return designs


def _group_designs(designs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for d in designs:
        tags = d["content"].get("tags", [])
        batch = "other"
        for t in tags:
            if t in GROUP_LABELS or t.startswith("batch"):
                batch = t
                break
        groups.setdefault(batch, []).append(d)
    return groups


def _status_counts(designs: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for d in designs:
        s = d["content"].get("status", "?")
        counts[s] = counts.get(s, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

class DesignItem(ListItem):
    def __init__(self, design: dict[str, Any]) -> None:
        self.design = design
        name = design["content"]["name"]
        status = design["content"].get("status", "draft")
        color = STATUS_DOTS.get(status, "white")
        # Truncate long names
        display_name = name if len(name) <= 28 else name[:25] + "\u2026"
        super().__init__(
            Label(f"  [{color}]\u25cf[/{color}] {display_name}"),
            name=name,
        )


class DesignDetail(Static):
    def update_design(self, design: dict[str, Any] | None) -> None:
        if design is None:
            self.update("[dim]Select a design to view details[/dim]")
            return

        content = design["content"]
        meta = design["meta"]
        config = content.get("config", {})
        linked = content.get("linked_run_ids", [])
        status = content.get("status", "draft")
        status_color = STATUS_COLORS.get(status, "white")

        # Config table with alignment
        if config:
            max_key = max(len(k) for k in config)
            config_lines = "\n".join(
                f"  {k:<{max_key}}  [bold]{v}[/bold]" for k, v in sorted(config.items())
            )
        else:
            config_lines = "  [dim](empty)[/dim]"

        linked_str = ", ".join(linked) if linked else "[dim](none)[/dim]"
        tags = content.get("tags", [])
        tags_str = "  ".join(f"[dim]#{t}[/dim]" for t in tags) if tags else "[dim](none)[/dim]"

        text = f"""\
[bold cyan]{content['name']}[/bold cyan]  [dim]v{meta['version']}[/dim]
[{status_color}]{status}[/{status_color}]  \u2502  {content.get('base_preset', '?')}  \u2502  {content.get('family', '?')}
{SEPARATOR}
[bold]Hypothesis[/bold]
  {content.get('hypothesis', '[dim]\u2014[/dim]')}
{SEPARATOR}
[bold]Config[/bold]
{config_lines}
{SEPARATOR}
[bold]Rationale[/bold]
  {content.get('rationale', '[dim]\u2014[/dim]')}
{SEPARATOR}
{tags_str}
[bold]Runs:[/bold] {linked_str}
[dim]{meta.get('created_at', '?')[:19]}  by {meta.get('created_by', '?')}[/dim]"""
        self.update(text)


class DiffView(Static):
    def show_diff(self, design_a: dict[str, Any], design_b: dict[str, Any]) -> None:
        ca = design_a["content"]
        cb = design_b["content"]
        all_keys = sorted(set(ca.keys()) | set(cb.keys()))

        changed = []
        unchanged = []
        for key in all_keys:
            va = ca.get(key)
            vb = cb.get(key)
            if va != vb:
                changed.append((key, va, vb))
            else:
                unchanged.append(key)

        lines = [
            f"[bold]Diff: [cyan]{ca['name']}[/cyan] \u2192 [cyan]{cb['name']}[/cyan][/bold]",
            f"[dim]{len(changed)} changed, {len(unchanged)} unchanged[/dim]",
            "",
        ]

        for key, va, vb in changed:
            # Expand config dicts key-by-key
            if key == "config" and isinstance(va, dict) and isinstance(vb, dict):
                lines.append(f"  [bold]{key}:[/bold]")
                all_config_keys = sorted(set(va.keys()) | set(vb.keys()))
                for ck in all_config_keys:
                    cv_a = va.get(ck, "[dim]<unset>[/dim]")
                    cv_b = vb.get(ck, "[dim]<unset>[/dim]")
                    if cv_a != cv_b:
                        lines.append(f"    [red]\u2212 {ck}: {cv_a}[/red]")
                        lines.append(f"    [green]+ {ck}: {cv_b}[/green]")
            else:
                va_str = str(va)[:60] if va is not None else "[dim]<unset>[/dim]"
                vb_str = str(vb)[:60] if vb is not None else "[dim]<unset>[/dim]"
                lines.append(f"  [bold]{key}:[/bold]")
                lines.append(f"    [red]\u2212 {va_str}[/red]")
                lines.append(f"    [green]+ {vb_str}[/green]")

        if unchanged:
            lines.append(f"\n[dim]Unchanged: {', '.join(unchanged[:8])}")
            if len(unchanged) > 8:
                lines.append(f"  ... +{len(unchanged) - 8} more[/dim]")
            else:
                lines[-1] += "[/dim]"

        self.update("\n".join(lines))


class HelpScreen(ModalScreen[None]):
    """Modal overlay showing keybindings."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("question_mark", "dismiss", "Close"),
    ]

    def compose(self) -> ComposeResult:
        yield Static(
            """\
[bold cyan]Crucible TUI \u2014 Keybindings[/bold cyan]

[bold]\u2191\u2193 / j k[/bold]     Navigate designs
[bold]Enter[/bold]       Select design
[bold]d[/bold]           Diff mode (select two designs)
[bold]s[/bold]           Cycle status (draft \u2192 ready \u2192 running \u2192 \u2026)
[bold]h[/bold]           Version history
[bold]c[/bold]           Research context view
[bold]r[/bold]           Run design (fleet)
[bold]?[/bold]           This help
[bold]q[/bold]           Quit

[dim]Press Escape or ? to close[/dim]""",
            id="help-content",
        )

    CSS = """
    #help-content {
        width: 52;
        height: auto;
        max-height: 20;
        padding: 1 2;
        border: thick $accent;
        background: $surface;
        margin: 4 auto;
    }
    """


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

class CrucibleApp(App):
    TITLE = "Crucible \u2014 Experiment Designs"

    CSS = """
    #main { height: 1fr; }
    #sidebar {
        width: 38;
        border-right: solid $accent;
    }
    #detail-scroll {
        width: 1fr;
        padding: 1 2;
    }
    #stats-bar {
        height: 1;
        dock: bottom;
        background: $boost;
        padding: 0 2;
    }
    DesignItem {
        padding: 0;
    }
    DesignItem:hover {
        background: $boost;
    }
    ListView > ListItem.-highlight {
        background: $accent 30%;
    }
    DesignDetail { padding: 0; }
    DiffView { padding: 0; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("d", "diff", "Diff"),
        Binding("s", "cycle_status", "Status"),
        Binding("h", "history", "History"),
        Binding("r", "run", "Run"),
        Binding("c", "context", "Context"),
        Binding("question_mark", "help", "Help"),
    ]

    show_diff_mode: reactive[bool] = reactive(False)
    _diff_anchor: dict[str, Any] | None = None
    _screenshot_dir: Path | None = None

    def __init__(self, screenshot_dir: Path | None = None) -> None:
        super().__init__()
        self._store = _load_store()
        self._designs = _load_designs(self._store)
        self._groups = _group_designs(self._designs)
        self._selected: dict[str, Any] | None = None
        self._screenshot_dir = screenshot_dir

    def compose(self) -> ComposeResult:
        yield Header()

        items: list[ListItem] = []
        for group_name, group_designs in self._groups.items():
            display = GROUP_LABELS.get(group_name, group_name.replace("-", " ").title())
            count = len(group_designs)
            items.append(ListItem(
                Label(f"[bold]{display}[/bold] [dim]({count})[/dim]"),
                disabled=True,
            ))
            for d in group_designs:
                items.append(DesignItem(d))

        with Horizontal(id="main"):
            with Vertical(id="sidebar"):
                yield ListView(*items)
            with VerticalScroll(id="detail-scroll"):
                yield DesignDetail(id="detail")
                yield DiffView(id="diff-view")

        # Stats bar
        counts = _status_counts(self._designs)
        ctx_count = len(self._store.list_resources("research_context"))
        parts = [f"[bold]{len(self._designs)}[/bold] designs"]
        for status, color in STATUS_COLORS.items():
            if counts.get(status, 0) > 0:
                parts.append(f"[{color}]{counts[status]} {status}[/{color}]")
        parts.append(f"{ctx_count} context")
        yield Label(" \u2502 ".join(parts), id="stats-bar")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#diff-view", DiffView).display = False
        if self._designs:
            self._selected = self._designs[0]
            self.query_one("#detail", DesignDetail).update_design(self._selected)

        if self._screenshot_dir:
            self.set_timer(0.5, self._take_screenshots)

    async def _take_screenshots(self) -> None:
        out = self._screenshot_dir
        if out is None:
            return
        out.mkdir(parents=True, exist_ok=True)

        # 1. Main view
        self.save_screenshot(str(out / "tui-main.svg"))

        # 2. Select a middle design for better detail
        if len(self._designs) > 4:
            self._selected = self._designs[4]
            self.query_one("#detail", DesignDetail).update_design(self._selected)
        self.save_screenshot(str(out / "tui-detail.svg"))

        # 3. Diff view
        if len(self._designs) >= 2:
            diff_view = self.query_one("#diff-view", DiffView)
            diff_view.show_diff(self._designs[0], self._designs[4] if len(self._designs) > 4 else self._designs[1])
            diff_view.display = True
            self.query_one("#detail", DesignDetail).display = False
            self.save_screenshot(str(out / "tui-diff.svg"))
            diff_view.display = False
            self.query_one("#detail", DesignDetail).display = True

        # 4. Context view
        self.action_context()
        self.save_screenshot(str(out / "tui-context.svg"))

        # 5. History view
        if self._designs:
            self._selected = self._designs[0]
            self.action_history()
            self.save_screenshot(str(out / "tui-history.svg"))

        self.notify(f"Screenshots saved to {out}/")
        self.exit()

    # --- Event handlers ---

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.item and isinstance(event.item, DesignItem):
            self._selected = event.item.design
            self.query_one("#detail", DesignDetail).update_design(self._selected)
            if self.show_diff_mode:
                self.show_diff_mode = False
                self.query_one("#diff-view", DiffView).display = False
                self.query_one("#detail", DesignDetail).display = True

    # --- Actions ---

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_diff(self) -> None:
        if self._selected is None:
            self.notify("No design selected", severity="warning")
            return
        if not self.show_diff_mode:
            self._diff_anchor = self._selected
            self.show_diff_mode = True
            name = self._selected["content"]["name"]
            self.notify(f"Diff: [{STATUS_COLORS['ready']}]{name}[/] \u2192 select another", severity="information")
        else:
            if self._diff_anchor and self._selected:
                diff_view = self.query_one("#diff-view", DiffView)
                diff_view.show_diff(self._diff_anchor, self._selected)
                diff_view.display = True
                self.query_one("#detail", DesignDetail).display = False
            self.show_diff_mode = False
            self._diff_anchor = None

    def action_cycle_status(self) -> None:
        if self._selected is None:
            return
        content = self._selected["content"]
        cycle = ["draft", "ready", "running", "completed", "archived"]
        current = content.get("status", "draft")
        idx = cycle.index(current) if current in cycle else 0
        new_status = cycle[(idx + 1) % len(cycle)]

        content["status"] = new_status
        meta = self._store.create(
            "experiment_design", content["name"], content,
            summary=f"Status: {current} \u2192 {new_status}",
            created_by="tui-user",
            tags=self._selected["meta"].get("tags", []),
        )
        self._selected["meta"] = meta
        self.query_one("#detail", DesignDetail).update_design(self._selected)
        color = STATUS_COLORS.get(new_status, "white")
        self.notify(f"{content['name']}: [{color}]{new_status}[/{color}]")

    def action_history(self) -> None:
        if self._selected is None:
            return
        name = self._selected["content"]["name"]
        versions = self._store.history("experiment_design", name)
        lines = [f"[bold cyan]Version history: {name}[/bold cyan]\n"]
        for v in versions:
            ts = v.get("created_at", "")[:19]
            by = v.get("created_by", "?")
            summary = v.get("summary", "")
            git = " [green]\u2713 git[/green]" if v.get("git_committed") else ""
            lines.append(f"  [bold]v{v['version']}[/bold]  {ts}  [dim]by {by}[/dim]{git}")
            if summary:
                lines.append(f"       {summary}")
        self.query_one("#detail", DesignDetail).update("\n".join(lines))

    def action_context(self) -> None:
        entries = self._store.list_resources("research_context")
        lines = ["[bold cyan]Research Context[/bold cyan]\n"]
        for meta in entries:
            result = self._store.get_current("research_context", meta["resource_name"])
            if result:
                _, content = result
                entry_type = content.get("entry_type", "?")
                status = content.get("status", "?")
                lines.append(f"[bold]{content.get('title', meta['resource_name'])}[/bold]")
                lines.append(f"  [dim]{entry_type} \u2502 {status}[/dim]")
                text = content.get("content", "")
                if len(text) > 300:
                    text = text[:300] + "\u2026"
                lines.append(f"  {text}\n")
        self.query_one("#detail", DesignDetail).update("\n".join(lines))

    def action_run(self) -> None:
        if self._selected is None:
            return
        name = self._selected["content"]["name"]
        self.notify(f"Run {name}? (connect fleet first)", severity="warning")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    screenshot_dir = None
    if "--screenshots" in sys.argv:
        idx = sys.argv.index("--screenshots")
        if idx + 1 < len(sys.argv):
            screenshot_dir = Path(sys.argv[idx + 1])
        else:
            screenshot_dir = Path("docs/images")

    app = CrucibleApp(screenshot_dir=screenshot_dir)
    app.run()


if __name__ == "__main__":
    main()
