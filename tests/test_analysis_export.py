"""Tests for analysis/export.py — was at 11% coverage.

Pure-python module (no torch). Covers export_top_configs JSON output,
print_rank stdout shape, generate_summary markdown rendering, and the
tag-filter / metric-resolution branches.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from crucible.analysis.export import (
    _metric_val,
    _resolve_metric,
    _sorted_completed,
    export_top_configs,
    generate_summary,
    print_rank,
)


# ---------------------------------------------------------------------------
# Fake config + results — sidesteps the full ProjectConfig loader.
# ---------------------------------------------------------------------------


def _make_cfg(tmp_path: Path, *, primary: str = "val_loss", secondary: str = "",
              direction: str = "minimize") -> SimpleNamespace:
    results_file = tmp_path / "results.jsonl"
    return SimpleNamespace(
        results_file=str(results_file),
        fleet_results_file=str(tmp_path / "fleet.jsonl"),
        metrics=SimpleNamespace(primary=primary, secondary=secondary, direction=direction),
        project_root=tmp_path,
    )


def _write_results(cfg: SimpleNamespace, rows: list[dict]) -> None:
    p = Path(cfg.results_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    # fleet ledger empty so loader doesn't trip
    Path(cfg.fleet_results_file).write_text("", encoding="utf-8")


def _row(name: str, val_loss: float, *, tag: str | None = None,
         model_bytes: int = 50000, lr: str = "0.001",
         val_bpb: float | None = None) -> dict:
    res = {"val_loss": val_loss}
    if val_bpb is not None:
        res["val_bpb"] = val_bpb
    return {
        "name": name,
        "status": "completed",
        "result": res,
        "model_bytes": model_bytes,
        "tags": [tag] if tag else [],
        "config": {"LR": lr, "MODEL_DIM": "256"},
    }


class TestResolveMetric:
    def test_explicit_metric_wins(self):
        assert _resolve_metric("val_bpb", None) == "val_bpb"

    def test_falls_back_to_config_primary(self, tmp_path):
        cfg = _make_cfg(tmp_path, primary="val_bpb")
        assert _resolve_metric(None, cfg) == "val_bpb"

    def test_final_default_is_val_loss(self):
        assert _resolve_metric(None, None) == "val_loss"


class TestMetricVal:
    def test_extracts_nested_metric(self):
        r = {"result": {"val_loss": 1.42}}
        assert _metric_val(r, "val_loss") == 1.42

    def test_missing_metric_raises(self):
        r = {"result": {}}
        with pytest.raises(KeyError):
            _metric_val(r, "val_loss")


class TestSortedCompleted:
    def test_sorts_ascending_by_metric(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        _write_results(cfg, [
            _row("a", 1.50),
            _row("b", 1.40),
            _row("c", 1.45),
        ])
        ordered = _sorted_completed("val_loss", "", cfg)
        names = [r["name"] for r in ordered]
        assert names == ["b", "c", "a"]

    def test_tag_filter(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        _write_results(cfg, [
            _row("a", 1.50, tag="x"),
            _row("b", 1.40, tag="y"),
            _row("c", 1.45, tag="x"),
        ])
        ordered = _sorted_completed("val_loss", "x", cfg)
        assert [r["name"] for r in ordered] == ["c", "a"]

    def test_empty_when_no_results(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        _write_results(cfg, [])
        assert _sorted_completed("val_loss", "", cfg) == []


class TestExportTopConfigs:
    def test_writes_top_n_json_files(self, tmp_path, capsys):
        cfg = _make_cfg(tmp_path)
        _write_results(cfg, [
            _row("a", 1.50),
            _row("b", 1.40),
            _row("c", 1.45),
        ])
        out = tmp_path / "winners"
        export_top_configs(n=2, out_dir=out, cfg=cfg)
        files = sorted(out.glob("*.json"))
        assert len(files) == 2
        # Filenames are rank-prefixed.
        names = [f.name for f in files]
        assert any(n.startswith("1_b") for n in names)
        assert any(n.startswith("2_c") for n in names)

        # Payload shape.
        payload = json.loads(files[0].read_text())
        assert payload["rank"] == 1
        assert payload["name"] == "b"
        assert payload["val_loss"] == 1.40
        assert payload["config"]["LR"] == "0.001"

        # Stdout has the ranked table header.
        out_text = capsys.readouterr().out
        assert "Rank" in out_text
        assert "val_loss" in out_text

    def test_warns_when_no_results(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        _write_results(cfg, [])
        out = tmp_path / "winners"
        export_top_configs(n=5, out_dir=out, cfg=cfg)
        # No files written, no crash.
        assert not list(out.glob("*.json")) if out.exists() else True

    def test_secondary_metric_in_output(self, tmp_path, capsys):
        cfg = _make_cfg(tmp_path, secondary="val_bpb")
        _write_results(cfg, [
            _row("a", 1.50, val_bpb=0.95),
            _row("b", 1.40, val_bpb=0.90),
        ])
        out = tmp_path / "winners"
        export_top_configs(n=2, out_dir=out, cfg=cfg)
        payload = json.loads((out / "1_b.json").read_text())
        assert payload["val_bpb"] == 0.90
        # Header includes the secondary column.
        out_text = capsys.readouterr().out
        assert "val_bpb" in out_text


class TestPrintRank:
    def test_prints_ranked_table(self, tmp_path, capsys):
        cfg = _make_cfg(tmp_path)
        _write_results(cfg, [
            _row("a", 1.50),
            _row("b", 1.40),
            _row("c", 1.45),
        ])
        print_rank(n=10, cfg=cfg)
        out = capsys.readouterr().out
        # Rank 1 is the lowest val_loss.
        assert "b" in out
        # All three should appear.
        assert "a" in out and "c" in out
        # Summary line at the end.
        assert "completed total" in out

    def test_warns_when_no_results(self, tmp_path, capsys):
        cfg = _make_cfg(tmp_path)
        _write_results(cfg, [])
        print_rank(n=10, cfg=cfg)
        # No crash; nothing useful in stdout, warning on stderr.
        assert "completed total" not in capsys.readouterr().out

    def test_tag_filter_narrows_results(self, tmp_path, capsys):
        cfg = _make_cfg(tmp_path)
        _write_results(cfg, [
            _row("a", 1.50, tag="alpha"),
            _row("b", 1.40, tag="beta"),
        ])
        print_rank(n=10, tag="alpha", cfg=cfg)
        out = capsys.readouterr().out
        assert "a" in out
        # b carried tag "beta" so should NOT appear in the table body —
        # only the header has "beta" (no), and the row has "a".
        # Coarse check: only one row.
        assert "1 completed total" in out


class TestGenerateSummary:
    def test_empty_returns_placeholder(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        _write_results(cfg, [])
        text = generate_summary(top_n=10, cfg=cfg)
        assert "No experiments completed yet" in text

    def test_includes_leaderboard_table(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        _write_results(cfg, [
            _row("a", 1.50, lr="0.001"),
            _row("b", 1.40, lr="0.002"),
            _row("c", 1.45, lr="0.003"),
        ])
        text = generate_summary(top_n=5, cfg=cfg)
        # Markdown header for top-N.
        assert "Top 5 by val_loss" in text
        # Pipe-table format.
        assert "| Rank |" in text
        assert "| Name |" in text
        # All three rows present.
        assert "| 1 | b |" in text
        assert "| 2 | c |" in text
        assert "| 3 | a |" in text

    def test_includes_best_config_export(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        _write_results(cfg, [
            _row("a", 1.50, lr="0.001"),
            _row("b", 1.40, lr="0.005"),
        ])
        text = generate_summary(top_n=5, cfg=cfg)
        assert "### Best Config" in text
        assert "```bash" in text
        # Best config is row b — its LR is 0.005.
        assert "export LR=0.005" in text
        assert "export MODEL_DIM=256" in text

    def test_sensitivity_section_when_multiple_distinct_values(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        _write_results(cfg, [
            _row("a", 1.50, lr="0.001"),
            _row("b", 1.40, lr="0.002"),
            _row("c", 1.45, lr="0.003"),
        ])
        text = generate_summary(top_n=5, cfg=cfg)
        # Sensitivity over LR (3 distinct values).
        assert "Sensitivity Analysis" in text
        assert "LR" in text
        assert "spread=" in text
