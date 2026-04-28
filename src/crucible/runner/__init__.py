"""Crucible runner: execute experiments with streaming output, status tracking, and OOM retry."""
from __future__ import annotations

from crucible.runner.experiment import run_experiment
from crucible.runner.output_parser import OutputParser, parse_output, classify_failure
from crucible.runner.presets import get_preset, list_presets, PRESET_DEFAULTS
from crucible.runner.tagger import tag_design, tag_recipe, merge_auto_tags
from crucible.runner.tracker import RunTracker
from crucible.core.fingerprint import code_fingerprint, safe_git_sha, safe_git_dirty

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
