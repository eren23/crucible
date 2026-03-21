"""Crucible TUI — interactive experiment design browser."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Footer, Header, Label, ListItem, ListView, Static

import yaml

from crucible.core.config import load_config
from crucible.core.store import VersionStore


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_store() -> VersionStore:
    config = load_config()
    return VersionStore(config.project_root / config.store_dir)


def _load_designs(store: VersionStore) -> list[dict[str, Any]]:
    """Load all designs with their content."""
    designs = []
    for meta in store.list_resources("experiment_design"):
        result = store.get_current("experiment_design", meta["resource_name"])
        if result:
            m, content = result
            designs.append({"meta": m, "content": content})
    return designs


def _group_designs(designs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group designs by their first batch tag."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for d in designs:
        tags = d["content"].get("tags", [])
        batch = "other"
        for t in tags:
            if t.startswith("batch") or t in ("activation-sweep", "competition-sota", "qlabs"):
                batch = t
                break
        groups.setdefault(batch, []).append(d)
    return groups


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

class DesignItem(ListItem):
    """A single design entry in the list."""

    def __init__(self, design: dict[str, Any]) -> None:
        self.design = design
        name = design["content"]["name"]
        status = design["content"].get("status", "?")
        ver = design["meta"]["version"]
        super().__init__(Label(f"  {name}"), name=name)
        self._status = status
        self._version = ver


class DesignDetail(Static):
    """Detail panel showing a single design's full info."""

    def update_design(self, design: dict[str, Any] | None) -> None:
        if design is None:
            self.update("[dim]No design selected[/dim]")
            return

        content = design["content"]
        meta = design["meta"]
        config = content.get("config", {})
        linked = content.get("linked_run_ids", [])

        config_lines = "\n".join(f"    {k}: {v}" for k, v in sorted(config.items()))
        linked_str = ", ".join(linked) if linked else "(none)"
        tags_str = ", ".join(content.get("tags", [])) or "(none)"

        text = f"""[bold cyan]{content['name']}[/bold cyan]  [dim]v{meta['version']}[/dim]
[bold]Status:[/bold] {content.get('status', '?')}    [bold]Preset:[/bold] {content.get('base_preset', '?')}    [bold]Family:[/bold] {content.get('family', '?')}

[bold]Hypothesis:[/bold]
  {content.get('hypothesis', '—')}

[bold]Config:[/bold]
{config_lines}

[bold]Rationale:[/bold]
  {content.get('rationale', '—')}

[bold]Tags:[/bold] {tags_str}
[bold]Linked runs:[/bold] {linked_str}
[bold]Created:[/bold] {meta.get('created_at', '?')[:19]}  [bold]By:[/bold] {meta.get('created_by', '?')}"""
        self.update(text)


class DiffView(Static):
    """Shows a diff between two designs."""

    def show_diff(self, design_a: dict[str, Any], design_b: dict[str, Any]) -> None:
        ca = design_a["content"]
        cb = design_b["content"]

        all_keys = sorted(set(ca.keys()) | set(cb.keys()))
        lines = [f"[bold]Diff: {ca['name']} → {cb['name']}[/bold]\n"]

        for key in all_keys:
            va = ca.get(key)
            vb = cb.get(key)
            if va != vb:
                lines.append(f"  [red]- {key}: {va!r}[/red]")
                lines.append(f"  [green]+ {key}: {vb!r}[/green]")

        if len(lines) == 1:
            lines.append("  [dim]No differences found.[/dim]")

        self.update("\n".join(lines))


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

class CrucibleApp(App):
    """Interactive TUI for Crucible experiment designs."""

    TITLE = "Crucible — Experiment Designs"
    CSS = """
    #main {
        height: 1fr;
    }
    #design-list {
        width: 32;
        border-right: solid $accent;
        overflow-y: auto;
    }
    #detail-scroll {
        width: 1fr;
        padding: 1 2;
    }
    .group-header {
        background: $boost;
        color: $text;
        text-style: bold;
        padding: 0 1;
    }
    DesignDetail {
        padding: 0;
    }
    DiffView {
        padding: 0;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("d", "diff", "Diff"),
        Binding("s", "cycle_status", "Status"),
        Binding("h", "history", "History"),
        Binding("r", "run", "Run"),
        Binding("c", "context", "Context"),
    ]

    show_diff_mode: reactive[bool] = reactive(False)
    _diff_anchor: dict[str, Any] | None = None

    def __init__(self) -> None:
        super().__init__()
        self._store = _load_store()
        self._designs = _load_designs(self._store)
        self._groups = _group_designs(self._designs)
        self._selected: dict[str, Any] | None = None

    def compose(self) -> ComposeResult:
        yield Header()

        # Build list items upfront (can't append after compose)
        items: list[ListItem] = []
        label_map = {
            "activation-sweep": "Activation Sweep",
            "competition-sota": "Competition SOTA",
            "qlabs": "QLabs-Inspired",
        }
        for group_name, group_designs in self._groups.items():
            display = label_map.get(group_name, group_name.title())
            items.append(ListItem(Label(f"[bold]{display}[/bold]"), disabled=True))
            for d in group_designs:
                items.append(DesignItem(d))

        with Horizontal(id="main"):
            with Vertical(id="design-list"):
                yield ListView(*items)
            with VerticalScroll(id="detail-scroll"):
                yield DesignDetail(id="detail")
                yield DiffView(id="diff-view")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#diff-view", DiffView).display = False
        if self._designs:
            self._selected = self._designs[0]
            self.query_one("#detail", DesignDetail).update_design(self._selected)

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.item and isinstance(event.item, DesignItem):
            self._selected = event.item.design
            detail = self.query_one("#detail", DesignDetail)
            detail.update_design(self._selected)
            if self.show_diff_mode:
                self.show_diff_mode = False
                self.query_one("#diff-view", DiffView).display = False
                self.query_one("#detail", DesignDetail).display = True

    def action_diff(self) -> None:
        if self._selected is None:
            self.notify("No design selected", severity="warning")
            return
        if not self.show_diff_mode:
            self._diff_anchor = self._selected
            self.show_diff_mode = True
            self.notify(f"Diff anchor: {self._selected['content']['name']}. Select another design to compare.", severity="information")
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
            "experiment_design",
            content["name"],
            content,
            summary=f"Status: {current} → {new_status}",
            created_by="tui-user",
            tags=self._selected["meta"].get("tags", []),
        )
        self._selected["meta"] = meta
        self.query_one("#detail", DesignDetail).update_design(self._selected)
        self.notify(f"{content['name']}: {current} → {new_status}")

    def action_history(self) -> None:
        if self._selected is None:
            return
        name = self._selected["content"]["name"]
        versions = self._store.history("experiment_design", name)
        lines = [f"[bold]Version history: {name}[/bold]\n"]
        for v in versions:
            lines.append(
                f"  v{v['version']}  {v.get('created_at', '')[:19]}  "
                f"by:{v.get('created_by', '?')}  {v.get('summary', '')}"
            )
        detail = self.query_one("#detail", DesignDetail)
        detail.update("\n".join(lines))

    def action_context(self) -> None:
        entries = self._store.list_resources("research_context")
        lines = ["[bold]Research Context[/bold]\n"]
        for meta in entries:
            result = self._store.get_current("research_context", meta["resource_name"])
            if result:
                _, content = result
                lines.append(f"[cyan]{content.get('title', meta['resource_name'])}[/cyan]")
                lines.append(f"  Type: {content.get('entry_type', '?')}  Status: {content.get('status', '?')}")
                text = content.get("content", "")
                if len(text) > 200:
                    text = text[:200] + "..."
                lines.append(f"  {text}\n")
        detail = self.query_one("#detail", DesignDetail)
        detail.update("\n".join(lines))

    def action_run(self) -> None:
        if self._selected is None:
            return
        name = self._selected["content"]["name"]
        self.notify(f"Run {name}? (not yet connected to fleet)", severity="warning")


def main() -> None:
    app = CrucibleApp()
    app.run()


if __name__ == "__main__":
    main()
