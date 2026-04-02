"""Tests for crucible.training.hyperparams — focuses on the epochs field."""

from __future__ import annotations

import os
import pytest


class TestEpochsParam:
    """Test EPOCHS env var parsing in Hyperparameters."""

    def test_epochs_default_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EPOCHS", raising=False)
        # Must re-import to pick up the patched env
        import importlib, crucible.training.hyperparams as hp_mod
        importlib.reload(hp_mod)
        from crucible.training.hyperparams import Hyperparameters
        h = Hyperparameters()
        assert h.epochs == 0

    def test_epochs_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EPOCHS", "3")
        import importlib, crucible.training.hyperparams as hp_mod
        importlib.reload(hp_mod)
        from crucible.training.hyperparams import Hyperparameters
        h = Hyperparameters()
        assert h.epochs == 3

    def test_epochs_invalid_uses_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EPOCHS", "notanumber")
        import importlib, crucible.training.hyperparams as hp_mod
        importlib.reload(hp_mod)
        from crucible.training.hyperparams import Hyperparameters
        h = Hyperparameters()
        assert h.epochs == 0
