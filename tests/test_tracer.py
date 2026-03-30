"""Tests for crucible.mcp.tracer — session trace recording."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from crucible.mcp.tracer import SessionTracer, load_trace, load_trace_meta


# ---------------------------------------------------------------------------
# SessionTracer.record
# ---------------------------------------------------------------------------


class TestSessionTracerRecord:
    def test_record_writes_valid_jsonl(self, tmp_path: Path):
        tracer = SessionTracer(tmp_path / "traces", session_id="test-session")
        tracer.record(
            tool="get_fleet_status",
            arguments={"include_metrics": False},
            result={"summary": "2 nodes ready"},
            duration_ms=42.5,
        )

        lines = tracer.trace_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["tool"] == "get_fleet_status"
        assert entry["seq"] == 1
        assert entry["duration_ms"] == 42.5
        assert entry["status"] == "ok"
        assert "error" not in entry

    def test_record_multiple_entries(self, tmp_path: Path):
        tracer = SessionTracer(tmp_path / "traces", session_id="multi")
        for i in range(5):
            tracer.record(
                tool=f"tool_{i}",
                arguments={"i": i},
                result={"ok": True},
                duration_ms=float(i),
            )

        entries = load_trace(tracer.trace_path)
        assert len(entries) == 5
        assert [e["seq"] for e in entries] == [1, 2, 3, 4, 5]

    def test_record_with_error(self, tmp_path: Path):
        tracer = SessionTracer(tmp_path / "traces", session_id="err")
        tracer.record(
            tool="provision_nodes",
            arguments={"count": 2},
            result=None,
            duration_ms=1500.0,
            status="error",
            error="RunPod API timeout",
        )

        entries = load_trace(tracer.trace_path)
        assert len(entries) == 1
        assert entries[0]["status"] == "error"
        assert entries[0]["error"] == "RunPod API timeout"

    def test_record_redacts_secrets(self, tmp_path: Path):
        tracer = SessionTracer(tmp_path / "traces", session_id="sec")
        tracer.record(
            tool="bootstrap_nodes",
            arguments={"WANDB_API_KEY": "wandb_v1_supersecretvalue12345"},
            result={"nodes": [{"name": "gpu-1", "MY_SECRET": "hidden"}]},
            duration_ms=100.0,
        )

        entries = load_trace(tracer.trace_path)
        assert entries[0]["arguments"]["WANDB_API_KEY"] == "<REDACTED>"
        assert entries[0]["result"]["nodes"][0]["MY_SECRET"] == "<REDACTED>"

    def test_record_truncates_long_string_result(self, tmp_path: Path):
        tracer = SessionTracer(tmp_path / "traces", session_id="trunc")
        long_result = "x" * 5000
        tracer.record(
            tool="get_run_logs",
            arguments={"run_id": "r1"},
            result=long_result,
            duration_ms=50.0,
        )

        entries = load_trace(tracer.trace_path)
        assert len(entries[0]["result"]) < 5000
        assert "truncated" in entries[0]["result"]

    def test_record_creates_trace_dir(self, tmp_path: Path):
        deep_dir = tmp_path / "a" / "b" / "c" / "traces"
        tracer = SessionTracer(deep_dir, session_id="deep")
        tracer.record(tool="test", arguments={}, result={}, duration_ms=1.0)
        assert deep_dir.exists()
        assert tracer.trace_path.exists()


# ---------------------------------------------------------------------------
# SessionTracer.finalize
# ---------------------------------------------------------------------------


class TestSessionTracerFinalize:
    def test_finalize_writes_valid_meta_yaml(self, tmp_path: Path):
        tracer = SessionTracer(tmp_path / "traces", session_id="fin")
        tracer.record(tool="get_fleet_status", arguments={}, result={}, duration_ms=10.0)
        tracer.record(tool="get_fleet_status", arguments={}, result={}, duration_ms=20.0)
        tracer.record(tool="get_leaderboard", arguments={}, result={}, duration_ms=30.0)

        trace_path = tracer.finalize()
        assert trace_path == tracer.trace_path

        meta = load_trace_meta(tracer.meta_path)
        assert meta["session_id"] == "fin"
        assert meta["trace_version"] == 2
        assert meta["tool_calls"] == 3
        assert meta["tool_counts"]["get_fleet_status"] == 2
        assert meta["tool_counts"]["get_leaderboard"] == 1
        assert meta["trace_file"] == "fin.jsonl"
        assert "started_at" in meta
        assert "ended_at" in meta

    def test_finalize_records_identifiers(self, tmp_path: Path):
        tracer = SessionTracer(tmp_path / "traces", session_id="ids")
        tracer.record(
            tool="run_project",
            arguments={"project_name": "lewm"},
            result={"run_id": "lewm_123", "nodes": [{"name": "node-1"}]},
            duration_ms=12.0,
            identifiers={"run_ids": ["lewm_123"], "project_names": ["lewm"], "node_names": ["node-1"]},
        )
        tracer.finalize()

        entries = load_trace(tracer.trace_path)
        assert entries[0]["identifiers"]["run_ids"] == ["lewm_123"]

        meta = load_trace_meta(tracer.meta_path)
        assert meta["identifiers"]["run_ids"] == ["lewm_123"]
        assert meta["identifiers"]["project_names"] == ["lewm"]

    def test_finalize_atomic_write(self, tmp_path: Path):
        """The .tmp file should not exist after finalize completes."""
        tracer = SessionTracer(tmp_path / "traces", session_id="atomic")
        tracer.record(tool="test", arguments={}, result={}, duration_ms=1.0)
        tracer.finalize()

        tmp_file = tracer.meta_path.with_suffix(".tmp")
        assert not tmp_file.exists()
        assert tracer.meta_path.exists()

    def test_finalize_empty_session(self, tmp_path: Path):
        """Finalize with zero tool calls should still produce valid meta."""
        tracer = SessionTracer(tmp_path / "traces", session_id="empty")
        tracer.finalize()

        meta = load_trace_meta(tracer.meta_path)
        assert meta["tool_calls"] == 0
        assert meta["tool_counts"] == {}


# ---------------------------------------------------------------------------
# SessionTracer default session_id
# ---------------------------------------------------------------------------


class TestSessionTracerSessionId:
    def test_default_session_id_format(self, tmp_path: Path):
        tracer = SessionTracer(tmp_path / "traces")
        # Default ID should be a UTC timestamp like 20260328T120000Z
        assert len(tracer.session_id) == 16
        assert tracer.session_id.endswith("Z")
        assert "T" in tracer.session_id


# ---------------------------------------------------------------------------
# load_trace round-trip
# ---------------------------------------------------------------------------


class TestLoadTrace:
    def test_round_trip(self, tmp_path: Path):
        tracer = SessionTracer(tmp_path / "traces", session_id="rt")
        tracer.record(tool="a", arguments={"x": 1}, result={"y": 2}, duration_ms=5.0)
        tracer.record(tool="b", arguments={"x": 3}, result={"y": 4}, duration_ms=10.0)

        entries = load_trace(tracer.trace_path)
        assert len(entries) == 2
        assert entries[0]["tool"] == "a"
        assert entries[0]["arguments"] == {"x": 1}
        assert entries[1]["tool"] == "b"

    def test_load_empty_file(self, tmp_path: Path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        assert load_trace(empty) == []

    def test_load_with_blank_lines(self, tmp_path: Path):
        f = tmp_path / "blanks.jsonl"
        f.write_text(
            json.dumps({"tool": "a"}) + "\n\n" + json.dumps({"tool": "b"}) + "\n",
            encoding="utf-8",
        )
        entries = load_trace(f)
        assert len(entries) == 2


# ---------------------------------------------------------------------------
# Tool counts tracking
# ---------------------------------------------------------------------------


class TestToolCounts:
    def test_tool_counts_increment(self, tmp_path: Path):
        tracer = SessionTracer(tmp_path / "traces", session_id="counts")
        for _ in range(3):
            tracer.record(tool="get_fleet_status", arguments={}, result={}, duration_ms=1.0)
        for _ in range(2):
            tracer.record(tool="dispatch_experiments", arguments={}, result={}, duration_ms=1.0)
        tracer.record(tool="collect_results", arguments={}, result={}, duration_ms=1.0)

        tracer.finalize()
        meta = load_trace_meta(tracer.meta_path)

        # Sorted by count descending
        counts = meta["tool_counts"]
        keys = list(counts.keys())
        assert keys[0] == "get_fleet_status"
        assert counts["get_fleet_status"] == 3
        assert counts["dispatch_experiments"] == 2
        assert counts["collect_results"] == 1


# ---------------------------------------------------------------------------
# CLI trace commands
# ---------------------------------------------------------------------------


def _make_trace(traces_dir: Path, session_id: str = "cli-test") -> None:
    """Helper: create a trace with a few entries and finalize it."""
    tracer = SessionTracer(traces_dir, session_id=session_id)
    tracer.record(
        tool="provision_nodes",
        arguments={"count": 2, "name_prefix": "crucible"},
        result={"created": 2, "new_nodes": ["n1", "n2"]},
        duration_ms=1532.5,
    )
    tracer.record(
        tool="fleet_refresh",
        arguments={},
        result={"nodes": 2, "ready": 2},
        duration_ms=823.1,
    )
    tracer.record(
        tool="bootstrap_nodes",
        arguments={"skip_data": False},
        result={"bootstrapped": 2},
        duration_ms=45200.0,
        status="error",
        error="SSH connection refused",
    )
    tracer.finalize()


class TestTraceCliList:
    def test_list_shows_sessions(self, tmp_path: Path, monkeypatch, capsys):
        traces_dir = tmp_path / ".crucible" / "traces"
        _make_trace(traces_dir, "session-alpha")
        _make_trace(traces_dir, "session-beta")

        # Patch load_config so _get_traces_dir returns our tmp path
        monkeypatch.setattr(
            "crucible.cli.trace_commands.load_config",
            lambda: type("C", (), {"project_root": tmp_path})(),
        )

        from crucible.cli.trace_commands import _cmd_list
        import argparse

        _cmd_list(argparse.Namespace())
        out = capsys.readouterr().out
        assert "session-alpha" in out
        assert "session-beta" in out
        assert "3" in out  # tool_calls count

    def test_list_empty_directory(self, tmp_path: Path, monkeypatch, capsys):
        traces_dir = tmp_path / ".crucible" / "traces"
        traces_dir.mkdir(parents=True)

        monkeypatch.setattr(
            "crucible.cli.trace_commands.load_config",
            lambda: type("C", (), {"project_root": tmp_path})(),
        )

        from crucible.cli.trace_commands import _cmd_list
        import argparse

        _cmd_list(argparse.Namespace())
        out = capsys.readouterr().out
        assert "No traces found" in out

    def test_list_no_traces_dir(self, tmp_path: Path, monkeypatch, capsys):
        monkeypatch.setattr(
            "crucible.cli.trace_commands.load_config",
            lambda: type("C", (), {"project_root": tmp_path})(),
        )

        from crucible.cli.trace_commands import _cmd_list
        import argparse

        _cmd_list(argparse.Namespace())
        out = capsys.readouterr().out
        assert "No traces directory" in out


class TestTraceCliShow:
    def test_show_prints_entries(self, tmp_path: Path, monkeypatch, capsys):
        traces_dir = tmp_path / ".crucible" / "traces"
        _make_trace(traces_dir, "show-test")

        monkeypatch.setattr(
            "crucible.cli.trace_commands.load_config",
            lambda: type("C", (), {"project_root": tmp_path})(),
        )

        from crucible.cli.trace_commands import _cmd_show
        import argparse

        _cmd_show(argparse.Namespace(session_id="show-test"))
        out = capsys.readouterr().out
        assert "provision_nodes" in out
        assert "fleet_refresh" in out
        assert "bootstrap_nodes" in out
        # Should be valid JSON blocks
        assert '"tool":' in out

    def test_show_missing_session(self, tmp_path: Path, monkeypatch):
        traces_dir = tmp_path / ".crucible" / "traces"
        traces_dir.mkdir(parents=True)

        monkeypatch.setattr(
            "crucible.cli.trace_commands.load_config",
            lambda: type("C", (), {"project_root": tmp_path})(),
        )

        from crucible.cli.trace_commands import _cmd_show
        import argparse

        with pytest.raises(SystemExit):
            _cmd_show(argparse.Namespace(session_id="nonexistent"))


class TestTraceCliExport:
    def test_export_produces_markdown(self, tmp_path: Path, monkeypatch, capsys):
        traces_dir = tmp_path / ".crucible" / "traces"
        _make_trace(traces_dir, "export-test")

        monkeypatch.setattr(
            "crucible.cli.trace_commands.load_config",
            lambda: type("C", (), {"project_root": tmp_path})(),
        )

        from crucible.cli.trace_commands import _cmd_export
        import argparse

        _cmd_export(argparse.Namespace(session_id="export-test", output=None))
        out = capsys.readouterr().out
        assert "# Crucible Session Trace: export-test" in out
        assert "**Started:**" in out
        assert "**Ended:**" in out
        assert "**Tool calls:** 3" in out
        assert "## Tool Call Sequence" in out
        assert "### 1. `provision_nodes`" in out
        assert "### 2. `fleet_refresh`" in out
        assert "### 3. `bootstrap_nodes`" in out
        assert "[error]" in out
        assert "SSH connection refused" in out

    def test_export_to_file(self, tmp_path: Path, monkeypatch, capsys):
        traces_dir = tmp_path / ".crucible" / "traces"
        _make_trace(traces_dir, "file-export")

        monkeypatch.setattr(
            "crucible.cli.trace_commands.load_config",
            lambda: type("C", (), {"project_root": tmp_path})(),
        )

        from crucible.cli.trace_commands import _cmd_export
        import argparse

        out_file = tmp_path / "export.md"
        _cmd_export(argparse.Namespace(session_id="file-export", output=str(out_file)))

        stdout = capsys.readouterr().out
        assert f"Exported to {out_file}" in stdout
        content = out_file.read_text(encoding="utf-8")
        assert "# Crucible Session Trace: file-export" in content
        assert "## Tool Call Sequence" in content

    def test_export_missing_meta_still_works(self, tmp_path: Path, monkeypatch, capsys):
        """Export should work even if .meta.yaml is missing (derives from entries)."""
        traces_dir = tmp_path / ".crucible" / "traces"
        _make_trace(traces_dir, "no-meta")
        # Remove the meta file
        meta_path = traces_dir / "no-meta.meta.yaml"
        meta_path.unlink()

        monkeypatch.setattr(
            "crucible.cli.trace_commands.load_config",
            lambda: type("C", (), {"project_root": tmp_path})(),
        )

        from crucible.cli.trace_commands import _cmd_export
        import argparse

        _cmd_export(argparse.Namespace(session_id="no-meta", output=None))
        out = capsys.readouterr().out
        assert "# Crucible Session Trace: no-meta" in out
        assert "**Tool calls:** 3" in out
