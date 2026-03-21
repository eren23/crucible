"""Crucible runner: execute experiments with streaming output, status tracking, and OOM retry."""
from __future__ import annotations

from crucible.runner.experiment import run_experiment
from crucible.runner.output_parser import OutputParser, parse_output, classify_failure
from crucible.runner.presets import get_preset, list_presets, PRESET_DEFAULTS
from crucible.runner.tracker import RunTracker
from crucible.runner.fingerprint import code_fingerprint, safe_git_sha, safe_git_dirty

__all__ = [
    "run_experiment",
    "OutputParser",
    "parse_output",
    "classify_failure",
    "get_preset",
    "list_presets",
    "PRESET_DEFAULTS",
    "RunTracker",
    "code_fingerprint",
    "safe_git_sha",
    "safe_git_dirty",
]
