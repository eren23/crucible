"""Baseline: minimal reactive agent. Returns a no-op action and stops fast."""
from __future__ import annotations

from typing import Any


class AgentScaffold:
    def __init__(self) -> None:
        self._task: str = ""
        self._turn: int = 0
        self._done: bool = False

    def initialize(self, task_description: str) -> None:
        self._task = task_description
        self._turn = 0
        self._done = False

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        self._turn += 1
        if self._turn >= 4 or observation.get("success"):
            self._done = True
        return {"action": "noop", "turn": self._turn}

    def is_done(self) -> bool:
        return self._done
