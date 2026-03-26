"""Unit and integration tests for generic backend contract handling."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from crucible.models.base import CrucibleModel
from crucible.training.generic_backend import _get_batch, _resolve_step_result


class LossModel(CrucibleModel):
    def forward(self, **batch):
        return {
            "loss": torch.tensor(1.0, requires_grad=True),
            "score": torch.tensor(0.5),
        }


class PredictionModel(CrucibleModel):
    def forward(self, **batch):
        return {"output": torch.tensor([1.0], requires_grad=True)}


class EchoObjective:
    def compute(self, predictions, targets):
        return {"loss": predictions["output"].sum()}


class GoodAdapter:
    def next_batch(self, **kwargs):
        return {"x": torch.tensor([1.0])}


class BadAdapter:
    def next_batch(self, **kwargs):
        raise RuntimeError("boom")


class _Args:
    seq_len = 8


def test_resolve_step_result_uses_existing_loss():
    result = _resolve_step_result(
        LossModel(),
        {},
        objective=None,
        objective_name="mse",
        objective_build_error=None,
        stage="training",
    )
    assert "loss" in result
    assert "score" in result


def test_resolve_step_result_applies_objective_when_loss_missing():
    result = _resolve_step_result(
        PredictionModel(),
        {},
        objective=EchoObjective(),
        objective_name="mse",
        objective_build_error=None,
        stage="training",
    )
    assert "loss" in result
    assert result["loss"].item() == 1.0


def test_resolve_step_result_raises_clear_error_without_loss_or_objective():
    with pytest.raises(KeyError, match="Either return a loss directly or configure a valid TRAINING_OBJECTIVE"):
        _resolve_step_result(
            PredictionModel(),
            {},
            objective=None,
            objective_name="mse",
            objective_build_error=None,
            stage="training",
        )


def test_get_batch_returns_adapter_batch():
    batch = _get_batch(
        GoodAdapter(),
        LossModel(),
        torch.device("cpu"),
        _Args(),
        batch_size=2,
    )
    assert batch["x"].shape == (1,)


def test_get_batch_raises_on_adapter_failure():
    with pytest.raises(RuntimeError, match="BadAdapter.next_batch\\(\\) failed: boom"):
        _get_batch(
            BadAdapter(),
            LossModel(),
            torch.device("cpu"),
            _Args(),
            batch_size=2,
        )


class TestGenericBackendIntegration:
    def test_run_experiment_uses_objective_end_to_end(self, tmp_path):
        from crucible.core.config import load_config
        from crucible.runner.experiment import run_experiment

        src_root = Path(__file__).resolve().parents[1] / "src"
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "crucible.yaml").write_text(
            textwrap.dedent(
                """
                name: generic-integration
                training:
                  - backend: generic
                    script: train_generic.py
                    modality: generic
                metrics:
                  primary: val_loss
                  direction: minimize
                presets:
                  smoke:
                    MODEL_FAMILY: prediction_model
                    DATA_ADAPTER: toy_adapter
                    TRAINING_OBJECTIVE: toy_objective
                    ITERATIONS: "2"
                    VAL_INTERVAL: "1"
                    LOG_INTERVAL: "1"
                    BATCH_SIZE: "2"
                    MAX_WALLCLOCK_SECONDS: "10"
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        (project_root / "train_generic.py").write_text(
            textwrap.dedent(
                f"""
                from __future__ import annotations
                import sys
                sys.path.insert(0, {str(src_root)!r})

                import torch
                from crucible.models.base import CrucibleModel
                from crucible.models.registry import register_model
                from crucible.training.data_adapters import DataAdapter, register_data_adapter
                from crucible.training.objectives import TrainingObjective, register_objective
                from crucible.training.generic_backend import run_generic_training

                class PredictionModel(CrucibleModel):
                    def forward(self, target, **batch):
                        return {{"output": target.clone().requires_grad_(True)}}

                class ToyAdapter(DataAdapter):
                    def next_batch(self, batch_size=2, device=None, **kwargs):
                        target = torch.ones(batch_size, 1)
                        if device is not None:
                            target = target.to(device)
                        return {{"target": target}}

                class ToyObjective(TrainingObjective):
                    name = "toy_objective"
                    def compute(self, predictions, targets):
                        loss = ((predictions["output"] - targets["target"]) ** 2).mean()
                        return {{"loss": loss}}

                register_model("prediction_model", lambda args: PredictionModel())
                register_data_adapter("toy_adapter", ToyAdapter)
                register_objective("toy_objective", ToyObjective)

                if __name__ == "__main__":
                    run_generic_training()
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        cfg = load_config(project_root / "crucible.yaml")
        result = run_experiment(
            config={},
            name="objective-e2e",
            backend="generic",
            preset="smoke",
            project_root=project_root,
            project_config=cfg,
            stream_output=False,
            timeout_seconds=30,
        )

        assert result["status"] == "completed"
        assert result["result"] is not None
        assert "val_loss" in result["result"]

    def test_run_experiment_fails_fast_on_bad_adapter(self, tmp_path):
        from crucible.core.config import load_config
        from crucible.runner.experiment import run_experiment

        src_root = Path(__file__).resolve().parents[1] / "src"
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "crucible.yaml").write_text(
            textwrap.dedent(
                """
                name: generic-adapter-failure
                training:
                  - backend: generic
                    script: train_generic.py
                    modality: generic
                presets:
                  smoke:
                    MODEL_FAMILY: loss_model
                    DATA_ADAPTER: broken_adapter
                    ITERATIONS: "1"
                    LOG_INTERVAL: "1"
                    MAX_WALLCLOCK_SECONDS: "10"
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        (project_root / "train_generic.py").write_text(
            textwrap.dedent(
                f"""
                from __future__ import annotations
                import sys
                sys.path.insert(0, {str(src_root)!r})

                import torch
                from crucible.models.base import CrucibleModel
                from crucible.models.registry import register_model
                from crucible.training.data_adapters import DataAdapter, register_data_adapter
                from crucible.training.generic_backend import run_generic_training

                class LossModel(CrucibleModel):
                    def forward(self, x=None, **batch):
                        return {{"loss": torch.tensor(0.0, requires_grad=True)}}

                class BrokenAdapter(DataAdapter):
                    def next_batch(self, **kwargs):
                        raise RuntimeError("broken adapter path")

                register_model("loss_model", lambda args: LossModel())
                register_data_adapter("broken_adapter", BrokenAdapter)

                if __name__ == "__main__":
                    run_generic_training()
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        cfg = load_config(project_root / "crucible.yaml")
        result = run_experiment(
            config={},
            name="adapter-fail-e2e",
            backend="generic",
            preset="smoke",
            project_root=project_root,
            project_config=cfg,
            stream_output=False,
            timeout_seconds=30,
        )

        assert result["status"] == "failed"
        assert "broken adapter path" in (result["error"] or "")
