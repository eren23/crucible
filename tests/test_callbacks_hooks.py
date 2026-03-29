"""Tests for crucible.training.callbacks — lifecycle hooks and ordering."""
from __future__ import annotations

from typing import Any

import pytest

from crucible.training.callbacks import (
    TrainingCallback,
    build_callbacks,
    register_callback,
    CALLBACK_REGISTRY,
)


class TestTrainingCallbackBase:
    def test_on_model_ready_called(self):
        """Verify on_model_ready exists on TrainingCallback and is callable."""
        cb = TrainingCallback()
        # Should not raise — default implementation is a no-op
        cb.on_model_ready({"model": None})

    def test_on_step_begin_called(self):
        """Verify on_step_begin hook exists and is callable."""
        cb = TrainingCallback()
        cb.on_step_begin(0, {"model": None})

    def test_on_validation_end_called(self):
        """Verify on_validation_end hook exists and is callable."""
        cb = TrainingCallback()
        cb.on_validation_end(0, {"val_loss": 1.0}, {"model": None})


class _RecordingCallback(TrainingCallback):
    """Callback that records which hooks were called in order."""

    priority = 50

    def __init__(self):
        self.calls: list[str] = []

    def on_model_ready(self, state: dict[str, Any]) -> None:
        self.calls.append("on_model_ready")

    def on_train_begin(self, state: dict[str, Any]) -> None:
        self.calls.append("on_train_begin")

    def on_step_begin(self, step: int, state: dict[str, Any]) -> None:
        self.calls.append("on_step_begin")

    def on_step_end(self, step: int, metrics: dict[str, Any], state: dict[str, Any]) -> None:
        self.calls.append("on_step_end")

    def on_validation_end(self, step: int, metrics: dict[str, Any], state: dict[str, Any]) -> None:
        self.calls.append("on_validation_end")

    def on_train_end(self, state: dict[str, Any]) -> None:
        self.calls.append("on_train_end")


class TestCallbackPriorityOrdering:
    def test_callback_priority_ordering(self):
        """build_callbacks should return callbacks sorted by priority (low first)."""
        cbs = build_callbacks("grad_clip,nan_detector,early_stopping")
        priorities = [cb.priority for cb in cbs]
        assert priorities == sorted(priorities)
        # grad_clip=10, nan_detector=20, early_stopping=90
        assert priorities == [10, 20, 90]


class TestOnModelReadyBeforeTrainBegin:
    def test_on_model_ready_before_train_begin(self):
        """Verify on_model_ready is a distinct hook that can be called before on_train_begin."""
        cb = _RecordingCallback()
        state: dict[str, Any] = {}
        cb.on_model_ready(state)
        cb.on_train_begin(state)
        assert cb.calls == ["on_model_ready", "on_train_begin"]


class TestRecordingCallbackHooks:
    def test_full_lifecycle(self):
        """Exercise all hooks in order and verify calls are recorded."""
        cb = _RecordingCallback()
        state: dict[str, Any] = {}
        metrics: dict[str, Any] = {"train_loss": 0.5}
        cb.on_model_ready(state)
        cb.on_train_begin(state)
        cb.on_step_begin(0, state)
        cb.on_step_end(0, metrics, state)
        cb.on_validation_end(0, {"val_loss": 0.4}, state)
        cb.on_train_end(state)
        assert cb.calls == [
            "on_model_ready",
            "on_train_begin",
            "on_step_begin",
            "on_step_end",
            "on_validation_end",
            "on_train_end",
        ]
