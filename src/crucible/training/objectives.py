"""Training objectives -- modality-agnostic loss functions.

Each objective wraps a standard loss function and conforms to a common
interface so the generic training backend can swap objectives via config.
"""
from __future__ import annotations

from typing import Any


class TrainingObjective:
    """Base class for training objectives."""

    name: str = ""

    def compute(
        self, predictions: dict[str, Any], targets: dict[str, Any]
    ) -> dict[str, Any]:
        """Compute loss from predictions and targets.

        Returns a dict with at least a ``'loss'`` key (a scalar tensor).
        """
        raise NotImplementedError

    def metric_names(self) -> list[str]:
        """Extra metric names beyond ``'loss'`` that :meth:`compute` returns."""
        return []


class CrossEntropyObjective(TrainingObjective):
    """Standard cross-entropy loss for classification / language modelling."""

    name = "cross_entropy"

    def __init__(self, ignore_index: int = -100, label_smoothing: float = 0.0):
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def compute(
        self, predictions: dict[str, Any], targets: dict[str, Any]
    ) -> dict[str, Any]:
        import torch.nn.functional as F

        logits = predictions["logits"]
        labels = targets["labels"]
        # Flatten to (N, C) and (N,)
        if logits.dim() > 2:
            logits = logits.reshape(-1, logits.size(-1))
            labels = labels.reshape(-1)
        loss = F.cross_entropy(
            logits.float(),
            labels,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )
        return {"loss": loss}


class MSEObjective(TrainingObjective):
    """Mean-squared-error loss for regression tasks."""

    name = "mse"

    def compute(
        self, predictions: dict[str, Any], targets: dict[str, Any]
    ) -> dict[str, Any]:
        import torch.nn.functional as F

        loss = F.mse_loss(predictions["output"], targets["target"])
        return {"loss": loss}


class KLDivergenceObjective(TrainingObjective):
    """KL-divergence loss for distribution matching."""

    name = "kl_divergence"

    def __init__(self, log_target: bool = False):
        self.log_target = log_target

    def compute(
        self, predictions: dict[str, Any], targets: dict[str, Any]
    ) -> dict[str, Any]:
        import torch.nn.functional as F

        loss = F.kl_div(
            predictions["log_probs"],
            targets["target_probs"],
            reduction="batchmean",
            log_target=self.log_target,
        )
        return {"loss": loss}


class CompositeObjective(TrainingObjective):
    """Weighted sum of multiple objectives."""

    name = "composite"

    def __init__(self, objectives: list[tuple[float, TrainingObjective]]):
        if not objectives:
            raise ValueError("CompositeObjective requires at least one sub-objective")
        self.objectives = objectives

    def compute(
        self, predictions: dict[str, Any], targets: dict[str, Any]
    ) -> dict[str, Any]:
        total_loss = None
        result: dict[str, Any] = {}
        for weight, obj in self.objectives:
            sub = obj.compute(predictions, targets)
            sub_loss = sub["loss"]
            weighted = weight * sub_loss
            total_loss = weighted if total_loss is None else total_loss + weighted
            # Collect sub-losses under namespaced keys
            result[f"loss_{obj.name}"] = sub_loss
        result["loss"] = total_loss
        return result

    def metric_names(self) -> list[str]:
        return [f"loss_{obj.name}" for _, obj in self.objectives]


# ---------------------------------------------------------------------------
# Diffusion objectives
# ---------------------------------------------------------------------------

class DiffusionLossObjective(TrainingObjective):
    """Noise-prediction MSE for denoising diffusion models.

    Expects:
        predictions: ``{"noise_pred": Tensor}``
        targets:     ``{"noise": Tensor}``

    Returns ``{"loss": scalar, "noise_mse": scalar}``.
    """

    name = "diffusion"

    def compute(
        self, predictions: dict[str, Any], targets: dict[str, Any]
    ) -> dict[str, Any]:
        import torch.nn.functional as F

        noise_pred = predictions["noise_pred"]
        noise = targets["noise"]
        mse = F.mse_loss(noise_pred, noise)
        return {"loss": mse, "noise_mse": mse}

    def metric_names(self) -> list[str]:
        return ["noise_mse"]


# ---------------------------------------------------------------------------
# World model / JEPA objectives
# ---------------------------------------------------------------------------

class JEPAObjective(TrainingObjective):
    """Joint Embedding Predictive Architecture loss.

    Combines prediction MSE with variance regularization (VICReg-style).

    Expects:
        predictions: ``{"pred_embeddings": Tensor, "z_std": Tensor}``
        targets:     ``{"target_embeddings": Tensor}``

    ``z_std`` is the standard deviation of the encoder output across the
    batch dimension -- the regularizer encourages it to stay above 1.0.

    Returns ``{"loss": scalar, "pred_loss": scalar, "var_reg": scalar}``.
    """

    name = "jepa"

    def __init__(self, var_weight: float = 0.1, var_target: float = 1.0):
        self.var_weight = var_weight
        self.var_target = var_target

    def compute(
        self, predictions: dict[str, Any], targets: dict[str, Any]
    ) -> dict[str, Any]:
        import torch
        import torch.nn.functional as F

        pred_emb = predictions["pred_embeddings"]
        target_emb = targets["target_embeddings"]

        # Prediction loss: MSE in embedding space
        pred_loss = F.mse_loss(pred_emb, target_emb)

        # Variance regularization: hinge loss pushing std above target
        z_std = predictions["z_std"]  # [embed_dim] or [batch, embed_dim]
        var_reg = torch.relu(self.var_target - z_std).mean()

        loss = pred_loss + self.var_weight * var_reg
        return {"loss": loss, "pred_loss": pred_loss, "var_reg": var_reg}

    def metric_names(self) -> list[str]:
        return ["pred_loss", "var_reg"]


# ---------------------------------------------------------------------------
# Registry (PluginRegistry-backed, with backward-compatible API)
# ---------------------------------------------------------------------------

from crucible.core.plugin_registry import PluginRegistry

_OBJECTIVE_REGISTRY = PluginRegistry[type["TrainingObjective"]]("objective")
OBJECTIVE_REGISTRY: dict[str, type["TrainingObjective"]] = _OBJECTIVE_REGISTRY._registry  # backward compat


def register_objective(name: str, cls: type["TrainingObjective"], *, source: str = "builtin") -> None:
    """Register an objective class under *name*.

    Supports 3-tier precedence (builtin < global < local) via *source*.
    """
    _OBJECTIVE_REGISTRY.register(name, cls, source=source)


def build_objective(name: str, **kwargs: Any) -> "TrainingObjective":
    """Instantiate a registered objective by name."""
    cls = _OBJECTIVE_REGISTRY.get(name)
    if cls is None:
        raise KeyError(
            f"Unknown objective '{name}'. Available: {sorted(_OBJECTIVE_REGISTRY.list_plugins())}"
        )
    return cls(**kwargs)


def list_objectives() -> list[str]:
    """Return sorted list of registered objective names."""
    return _OBJECTIVE_REGISTRY.list_plugins()


def list_objectives_detailed() -> list[dict[str, str]]:
    """Return objectives with source metadata."""
    return _OBJECTIVE_REGISTRY.list_plugins_detailed()


# Register built-ins
register_objective("cross_entropy", CrossEntropyObjective)
register_objective("mse", MSEObjective)
register_objective("kl_divergence", KLDivergenceObjective)
register_objective("composite", CompositeObjective)
register_objective("diffusion", DiffusionLossObjective)
register_objective("jepa", JEPAObjective)
