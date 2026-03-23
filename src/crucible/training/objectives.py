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
        import torch

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
# Registry
# ---------------------------------------------------------------------------

OBJECTIVE_REGISTRY: dict[str, type[TrainingObjective]] = {}


def register_objective(name: str, cls: type[TrainingObjective]) -> None:
    """Register an objective class under *name*."""
    OBJECTIVE_REGISTRY[name] = cls


def build_objective(name: str, **kwargs: Any) -> TrainingObjective:
    """Instantiate a registered objective by name."""
    cls = OBJECTIVE_REGISTRY.get(name)
    if cls is None:
        raise KeyError(
            f"Unknown objective '{name}'. Available: {sorted(OBJECTIVE_REGISTRY)}"
        )
    return cls(**kwargs)


# Register built-ins
register_objective("cross_entropy", CrossEntropyObjective)
register_objective("mse", MSEObjective)
register_objective("kl_divergence", KLDivergenceObjective)
register_objective("composite", CompositeObjective)
