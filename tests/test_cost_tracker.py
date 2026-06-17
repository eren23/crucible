"""Tests for crucible.runner.cost_tracker (Phase 1.8)."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from crucible.core.config import load_config
from crucible.runner.cost_tracker import (
    DEFAULT_FALLBACK_HOURLY_USD,
    active_pod_hourly_rate,
    compute_session_spend,
)


@pytest.fixture
def project_with_nodes(tmp_path: Path):
    """Project root with a minimal crucible.yaml and a nodes.json."""
    (tmp_path / "crucible.yaml").write_text(
        yaml.safe_dump({
            "name": "cost_test",
            "metrics": {"primary": "val_loss", "direction": "minimize"},
        }),
        encoding="utf-8",
    )
    return tmp_path


def _write_nodes(project_root: Path, nodes: list[dict]) -> None:
    (project_root / "nodes.json").write_text(json.dumps(nodes), encoding="utf-8")


# ---------------------------------------------------------------------------
# active_pod_hourly_rate
# ---------------------------------------------------------------------------


class TestActiveRate:
    def test_no_nodes_file_returns_zero(self, project_with_nodes):
        config = load_config(project_with_nodes / "crucible.yaml")
        assert active_pod_hourly_rate(config) == 0.0

    def test_sums_active_pods(self, project_with_nodes):
        _write_nodes(project_with_nodes, [
            {"name": "p1", "state": "ready", "cost_per_hr": 0.50},
            {"name": "p2", "state": "ready", "cost_per_hr": 1.20},
        ])
        config = load_config(project_with_nodes / "crucible.yaml")
        assert active_pod_hourly_rate(config) == pytest.approx(1.70)

    def test_excludes_terminal_pods(self, project_with_nodes):
        _write_nodes(project_with_nodes, [
            {"name": "live", "state": "ready", "cost_per_hr": 0.50},
            {"name": "dead", "state": "destroyed", "cost_per_hr": 99.0},
            {"name": "stopped", "state": "stopped", "cost_per_hr": 99.0},
        ])
        config = load_config(project_with_nodes / "crucible.yaml")
        assert active_pod_hourly_rate(config) == pytest.approx(0.50)

    def test_missing_cost_uses_fallback(self, project_with_nodes):
        _write_nodes(project_with_nodes, [
            {"name": "p1", "state": "ready"},  # no cost_per_hr
        ])
        config = load_config(project_with_nodes / "crucible.yaml")
        assert active_pod_hourly_rate(config) == pytest.approx(DEFAULT_FALLBACK_HOURLY_USD)


# ---------------------------------------------------------------------------
# compute_session_spend
# ---------------------------------------------------------------------------


class TestComputeSpend:
    def test_zero_hours_zero_spend(self, project_with_nodes):
        config = load_config(project_with_nodes / "crucible.yaml")
        _write_nodes(project_with_nodes, [{"name": "p", "state": "ready", "cost_per_hr": 1.0}])
        now = datetime.now(UTC)
        out = compute_session_spend(
            config, session_started_at=now.isoformat(), now=now,
        )
        assert out["spend_usd"] == pytest.approx(0.0, abs=1e-3)
        assert out["hourly_rate"] == pytest.approx(1.0)

    def test_spend_grows_with_wall_clock(self, project_with_nodes):
        config = load_config(project_with_nodes / "crucible.yaml")
        _write_nodes(project_with_nodes, [{"name": "p", "state": "ready", "cost_per_hr": 2.0}])
        started = datetime.now(UTC) - timedelta(hours=1.5)
        out = compute_session_spend(
            config, session_started_at=started.isoformat(),
            now=datetime.now(UTC),
        )
        # 1.5h × $2.0/hr ≈ $3.00 (allow small jitter)
        assert out["spend_usd"] == pytest.approx(3.0, abs=0.05)
        assert out["active_pods"] == 1

    def test_unparseable_started_at_returns_zero_with_error(self, project_with_nodes):
        config = load_config(project_with_nodes / "crucible.yaml")
        out = compute_session_spend(config, session_started_at="not-a-timestamp")
        assert out["spend_usd"] == 0.0
        assert "error" in out

    def test_no_active_pods_zero_spend(self, project_with_nodes):
        config = load_config(project_with_nodes / "crucible.yaml")
        _write_nodes(project_with_nodes, [
            {"name": "p", "state": "destroyed", "cost_per_hr": 5.0},
        ])
        started = datetime.now(UTC) - timedelta(hours=10)
        out = compute_session_spend(
            config, session_started_at=started.isoformat(),
            now=datetime.now(UTC),
        )
        assert out["spend_usd"] == 0.0
        assert out["active_pods"] == 0
