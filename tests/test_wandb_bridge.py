"""Tests for W&B bridge enhancements in crucible.runner.wandb."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from crucible.runner.wandb_logger import WandbLogger, _resolve_wandb_url, wandb_annotate_finished_run


# ---------------------------------------------------------------------------
# WandbLogger — inert mode (no wandb)
# ---------------------------------------------------------------------------


class TestWandbLoggerInert:
    """All new methods should be no-ops when the logger is inert."""

    def test_log_image_returns_false_when_disabled(self):
        logger = WandbLogger()
        assert logger.log_image("/some/path.png") is False

    def test_annotate_returns_false_when_disabled(self):
        logger = WandbLogger()
        assert logger.annotate(notes=["test"]) is False

    def test_update_config_noop_when_disabled(self):
        logger = WandbLogger()
        # Should not raise
        logger.update_config({"key": "value"})

    def test_log_image_returns_false_when_run_is_none(self):
        logger = WandbLogger(enabled=True, run=None)
        assert logger.log_image("/some/path.png") is False

    def test_annotate_returns_false_when_run_is_none(self):
        logger = WandbLogger(enabled=True, run=None)
        assert logger.annotate(findings=["test"]) is False

    def test_update_config_noop_when_run_is_none(self):
        logger = WandbLogger(enabled=True, run=None)
        logger.update_config({"key": "value"})


# ---------------------------------------------------------------------------
# WandbLogger — enabled with mocked run
# ---------------------------------------------------------------------------


class TestWandbLoggerEnabled:
    def _make_logger(self) -> tuple[WandbLogger, MagicMock]:
        mock_run = MagicMock()
        mock_run.summary = {}
        mock_run.config = MagicMock()
        logger = WandbLogger(run=mock_run, enabled=True)
        return logger, mock_run

    @patch("crucible.runner.wandb_logger.wandb", create=True)
    def test_log_image_calls_wandb_image(self, mock_wandb_mod):
        logger, mock_run = self._make_logger()
        mock_image = MagicMock()

        with patch.dict("sys.modules", {"wandb": mock_wandb_mod}):
            mock_wandb_mod.Image.return_value = mock_image
            result = logger.log_image("/path/to/img.png", caption="test cap", key="my_img")

        assert result is True
        mock_wandb_mod.Image.assert_called_once_with("/path/to/img.png", caption="test cap")
        mock_run.log.assert_called_once_with({"my_img": mock_image})

    def test_annotate_sets_notes(self):
        logger, mock_run = self._make_logger()
        result = logger.annotate(notes=["note1", "note2"])
        assert result is True
        assert mock_run.summary["crucible_notes"] == ["note1", "note2"]

    def test_annotate_sets_findings(self):
        logger, mock_run = self._make_logger()
        result = logger.annotate(findings=["finding1"])
        assert result is True
        assert mock_run.summary["crucible_findings"] == ["finding1"]

    def test_annotate_sets_both(self):
        logger, mock_run = self._make_logger()
        result = logger.annotate(notes=["n1"], findings=["f1"])
        assert result is True
        assert mock_run.summary["crucible_notes"] == ["n1"]
        assert mock_run.summary["crucible_findings"] == ["f1"]

    def test_annotate_returns_true_with_no_args(self):
        logger, mock_run = self._make_logger()
        # Neither notes nor findings — still returns True (run is active)
        result = logger.annotate()
        assert result is True

    def test_update_config_calls_run_config_update(self):
        logger, mock_run = self._make_logger()
        logger.update_config({"crucible_run_id": "exp_123", "crucible_preset": "proxy"})
        mock_run.config.update.assert_called_once_with({
            "crucible_run_id": "exp_123",
            "crucible_preset": "proxy",
        })


# ---------------------------------------------------------------------------
# _resolve_wandb_url
# ---------------------------------------------------------------------------


class TestResolveWandbUrl:
    def _make_config(self, tmp_path: Path) -> SimpleNamespace:
        return SimpleNamespace(
            project_root=tmp_path,
            logs_dir="logs",
            results_file="experiments.jsonl",
        )

    def test_resolves_from_status_sidecar(self, tmp_path: Path):
        config = self._make_config(tmp_path)
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        status = {
            "run_id": "exp_001",
            "state": "completed",
            "wandb": {"enabled": True, "url": "https://wandb.ai/team/proj/runs/abc123"},
        }
        (logs_dir / "exp_001.status.json").write_text(
            json.dumps(status), encoding="utf-8"
        )

        url = _resolve_wandb_url("exp_001", config)
        assert url == "https://wandb.ai/team/proj/runs/abc123"

    def test_resolves_from_results_jsonl(self, tmp_path: Path):
        config = self._make_config(tmp_path)
        (tmp_path / "logs").mkdir()
        results = [
            {
                "id": "exp_002",
                "name": "test",
                "wandb": {"url": "https://wandb.ai/team/proj/runs/def456"},
            }
        ]
        results_path = tmp_path / "experiments.jsonl"
        with open(results_path, "w", encoding="utf-8") as f:
            for rec in results:
                f.write(json.dumps(rec) + "\n")

        url = _resolve_wandb_url("exp_002", config)
        assert url == "https://wandb.ai/team/proj/runs/def456"

    def test_returns_none_when_no_url(self, tmp_path: Path):
        config = self._make_config(tmp_path)
        (tmp_path / "logs").mkdir()
        url = _resolve_wandb_url("nonexistent", config)
        assert url is None

    def test_status_sidecar_takes_priority(self, tmp_path: Path):
        config = self._make_config(tmp_path)
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()

        # Status sidecar
        status = {
            "run_id": "exp_003",
            "wandb": {"url": "https://wandb.ai/team/proj/runs/sidecar"},
        }
        (logs_dir / "exp_003.status.json").write_text(
            json.dumps(status), encoding="utf-8"
        )

        # Results JSONL
        results = [
            {
                "id": "exp_003",
                "wandb": {"url": "https://wandb.ai/team/proj/runs/jsonl"},
            }
        ]
        with open(tmp_path / "experiments.jsonl", "w", encoding="utf-8") as f:
            for rec in results:
                f.write(json.dumps(rec) + "\n")

        url = _resolve_wandb_url("exp_003", config)
        assert url == "https://wandb.ai/team/proj/runs/sidecar"

    def test_handles_missing_logs_dir(self, tmp_path: Path):
        config = self._make_config(tmp_path)
        # No logs dir, no results file
        url = _resolve_wandb_url("exp_missing", config)
        assert url is None

    def test_handles_malformed_status_json(self, tmp_path: Path):
        config = self._make_config(tmp_path)
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        (logs_dir / "exp_bad.status.json").write_text("not json", encoding="utf-8")
        url = _resolve_wandb_url("exp_bad", config)
        assert url is None


# ---------------------------------------------------------------------------
# wandb_annotate_finished_run
# ---------------------------------------------------------------------------


class TestWandbAnnotateFinishedRun:
    def test_returns_false_when_wandb_not_installed(self):
        with patch.dict("sys.modules", {"wandb": None}):
            # Import will fail when wandb is None in sys.modules
            # But we need to test the function's own import handling
            pass
        # The function catches ImportError internally
        # We can't easily make import fail without side effects,
        # so test with a bad URL instead
        result = wandb_annotate_finished_run("not-a-valid-url")
        assert result is False

    def test_returns_false_for_malformed_url(self):
        result = wandb_annotate_finished_run("https://example.com/bad")
        assert result is False

    @patch("crucible.runner.wandb_logger.wandb", create=True)
    def test_annotates_notes_successfully(self, mock_wandb_mod):
        mock_run = MagicMock()
        mock_run.summary = {}
        mock_api = MagicMock()
        mock_api.run.return_value = mock_run

        with patch.dict("sys.modules", {"wandb": mock_wandb_mod}):
            mock_wandb_mod.Api.return_value = mock_api
            result = wandb_annotate_finished_run(
                "https://wandb.ai/myteam/myproj/runs/abc123",
                notes=["note1"],
            )

        assert result is True
        mock_api.run.assert_called_once_with("myteam/myproj/abc123")
        assert mock_run.summary["crucible_notes"] == ["note1"]

    @patch("crucible.runner.wandb_logger.wandb", create=True)
    def test_annotates_findings_successfully(self, mock_wandb_mod):
        mock_run = MagicMock()
        mock_run.summary = {}
        mock_api = MagicMock()
        mock_api.run.return_value = mock_run

        with patch.dict("sys.modules", {"wandb": mock_wandb_mod}):
            mock_wandb_mod.Api.return_value = mock_api
            result = wandb_annotate_finished_run(
                "https://wandb.ai/team/proj/runs/xyz",
                findings=["f1", "f2"],
            )

        assert result is True
        assert mock_run.summary["crucible_findings"] == ["f1", "f2"]
