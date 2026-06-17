"""Crucible runner: execute experiments with streaming output, status tracking, and OOM retry."""
from __future__ import annotations

from crucible.core.fingerprint import code_fingerprint, safe_git_dirty, safe_git_sha
from crucible.runner.experiment import run_experiment
from crucible.runner.output_parser import OutputParser, classify_failure, parse_output
from crucible.runner.presets import PRESET_DEFAULTS, get_preset, list_presets
from crucible.runner.tagger import merge_auto_tags, tag_design, tag_recipe
from crucible.runner.tracker import RunTracker

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
    # Auto-tagging — pure functions over recipe / design dicts.
    "tag_recipe",
    "tag_design",
    "merge_auto_tags",
]
