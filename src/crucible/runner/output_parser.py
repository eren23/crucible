"""Regex-based parsers for training script stdout.

The training contract: the training script communicates results via stdout
lines that match known patterns. This module provides both a default set
of patterns (suitable for the reference training scripts) and the ability
to register custom patterns via OutputParser.

Default patterns recognised:
  - Final validation:   final_<tag> val_loss:<f> val_bpb:<f>
  - Step progress:      step:<n>/<total> train_loss:<f>
  - Validation step:    step:<n>/<total> val_loss:<f> val_bpb:<f>
  - Model size:         Serialized model ...: <n> bytes
  - Early stopping:     stopping_early...step:<n>/
  - Warmup:             warmup_step:<n>/<total>
  - Train time:         train_time:<n>ms
"""
from __future__ import annotations

import re
import signal
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Default regex patterns
# ---------------------------------------------------------------------------

# Final validation line (primary success signal)
FINAL_RE = re.compile(
    r"final_\w+\s+val_loss:(\d+\.\d+)\s+val_bpb:(\d+\.\d+)"
)

# Optional test-time-training / LoRA line
TTT_RE = re.compile(
    r"final_\w+_ttt_lora\s+val_loss:(\d+\.\d+)\s+val_bpb:(\d+\.\d+)"
)

# Per-step training loss
STEP_RE = re.compile(r"step:(\d+)/(\d+)\s+train_loss:")
TRAIN_LOSS_RE = re.compile(
    r"step:(\d+)/(\d+)\s+train_loss:(\d+\.\d+)"
)

# Validation during training
VAL_RE = re.compile(
    r"step:(\d+)/(\d+)\s+val_loss:(\d+\.\d+)\s+val_bpb:(\d+\.\d+)"
)

# Serialised model size in bytes
MODEL_BYTES_RE = re.compile(
    r"(?:serialized_model_int8_zlib:|Serialized model \w+\+\w+:)\s*(\d+)\s+bytes"
)

# Early stopping
STOPPING_RE = re.compile(r"stopping_early.*step:(\d+)/")

# Warmup progress
WARMUP_RE = re.compile(r"warmup_step:(\d+)/(\d+)")

# Wall-clock training time
TRAIN_TIME_RE = re.compile(r"train_time:(\d+)ms")


# ---------------------------------------------------------------------------
# OutputParser: configurable pattern-based parser
# ---------------------------------------------------------------------------

@dataclass
class OutputParser:
    """Configurable parser for training stdout lines.

    Holds compiled regex patterns and exposes methods for extracting
    structured data from raw output text.  The default instance matches
    the reference training script patterns; override individual patterns
    via constructor kwargs or by setting attributes after construction.

    Custom result extractors can be registered via ``add_extractor`` for
    domain-specific output lines.
    """

    final_re: re.Pattern[str] = field(default_factory=lambda: FINAL_RE)
    ttt_re: re.Pattern[str] = field(default_factory=lambda: TTT_RE)
    step_re: re.Pattern[str] = field(default_factory=lambda: STEP_RE)
    train_loss_re: re.Pattern[str] = field(default_factory=lambda: TRAIN_LOSS_RE)
    val_re: re.Pattern[str] = field(default_factory=lambda: VAL_RE)
    model_bytes_re: re.Pattern[str] = field(default_factory=lambda: MODEL_BYTES_RE)
    stopping_re: re.Pattern[str] = field(default_factory=lambda: STOPPING_RE)
    warmup_re: re.Pattern[str] = field(default_factory=lambda: WARMUP_RE)
    train_time_re: re.Pattern[str] = field(default_factory=lambda: TRAIN_TIME_RE)

    @classmethod
    def from_config(cls, patterns: dict[str, str] | None = None) -> "OutputParser":
        """Build a parser, optionally overriding regex strings from config."""
        if not patterns:
            return cls()
        kwargs: dict[str, Any] = {}
        mapping = {
            "final": "final_re",
            "ttt": "ttt_re",
            "step": "step_re",
            "train_loss": "train_loss_re",
            "val": "val_re",
            "model_bytes": "model_bytes_re",
            "stopping": "stopping_re",
            "warmup": "warmup_re",
            "train_time": "train_time_re",
        }
        for key, attr in mapping.items():
            if key in patterns:
                kwargs[attr] = re.compile(patterns[key])
        return cls(**kwargs)

    # -- Aggregate parse --------------------------------------------------

    def parse(self, text: str) -> dict[str, Any] | None:
        """Parse combined output text and return structured result or None.

        Returns a dict with keys:
          status:      "completed" | "partial_recoverable"
          result:      dict of extracted metrics
          model_bytes: int | None
        """
        final_match = self.final_re.search(text)
        if final_match:
            val_loss = float(final_match.group(1))
            val_bpb = float(final_match.group(2))
            bytes_match = self.model_bytes_re.search(text)
            model_bytes = int(bytes_match.group(1)) if bytes_match else None

            steps_completed = self.steps_seen(text)
            train_time_match = self.train_time_re.search(text)
            train_time_ms = int(train_time_match.group(1)) if train_time_match else None

            result_dict: dict[str, Any] = {
                "val_loss": val_loss,
                "val_bpb": val_bpb,
                "steps_completed": steps_completed,
                "train_time_ms": train_time_ms,
            }
            ttt_match = self.ttt_re.search(text)
            if ttt_match:
                result_dict["ttt_val_loss"] = float(ttt_match.group(1))
                result_dict["ttt_val_bpb"] = float(ttt_match.group(2))

            return {
                "status": "completed",
                "result": result_dict,
                "model_bytes": model_bytes,
            }

        # Fallback: partial metrics from train loss lines
        losses = self.train_loss_re.findall(text)
        if losses:
            step, _, last_loss = losses[-1]
            return {
                "status": "partial_recoverable",
                "result": {
                    "train_loss_fallback": float(last_loss),
                    "steps_completed": int(step),
                },
                "model_bytes": None,
            }

        return None

    # -- Line-level helpers ------------------------------------------------

    def steps_seen(self, text: str) -> int:
        """Return the highest step number observed in the output."""
        steps_completed = 0
        for match in self.step_re.finditer(text):
            steps_completed = max(steps_completed, int(match.group(1)))
        stop_match = self.stopping_re.search(text)
        if stop_match:
            steps_completed = max(steps_completed, int(stop_match.group(1)))
        return steps_completed

    def parse_line(self, line: str) -> dict[str, Any] | None:
        """Extract structured info from a single output line.

        Returns a dict with a "type" key indicating the match kind, or None.
        """
        stripped = line.strip()
        if not stripped:
            return None

        warmup = self.warmup_re.search(stripped)
        if warmup:
            return {
                "type": "warmup",
                "step": int(warmup.group(1)),
                "total": int(warmup.group(2)),
            }

        train_match = self.train_loss_re.search(stripped)
        if train_match:
            return {
                "type": "train_loss",
                "step": int(train_match.group(1)),
                "total_steps": int(train_match.group(2)),
                "train_loss": float(train_match.group(3)),
            }

        val_match = self.val_re.search(stripped)
        if val_match:
            return {
                "type": "val",
                "step": int(val_match.group(1)),
                "total_steps": int(val_match.group(2)),
                "val_loss": float(val_match.group(3)),
                "val_bpb": float(val_match.group(4)),
            }

        if stripped.startswith("saved_model:") or "Serialized model" in stripped:
            return {"type": "serializing", "line": stripped}

        if "_roundtrip" in stripped and stripped.startswith("final_"):
            return {"type": "final", "line": stripped}

        if "_ttt_lora" in stripped and stripped.startswith("final_"):
            return {"type": "final_ttt", "line": stripped}

        return None


# ---------------------------------------------------------------------------
# Module-level convenience functions (use default parser)
# ---------------------------------------------------------------------------

_default_parser = OutputParser()


def parse_output(text: str) -> dict[str, Any] | None:
    """Parse combined training output with the default parser."""
    return _default_parser.parse(text)


def steps_seen(text: str) -> int:
    """Count steps seen in output with the default parser."""
    return _default_parser.steps_seen(text)


def classify_failure(
    returncode: int | None,
    text: str,
    timed_out: bool,
) -> tuple[str, str | None]:
    """Classify a process exit into (status, failure_class).

    status:        completed | timeout | killed | failed
    failure_class: None | timeout | oom_suspected | sigkill | sigterm | ...
    """
    lower = text.lower()
    if timed_out:
        return "timeout", "timeout"
    if returncode == 0:
        return "completed", None
    if returncode is None:
        return "failed", "unknown_exit"

    # Negative returncode → killed by signal
    if returncode < 0:
        signal_num = -returncode
        signal_name = (
            signal.Signals(signal_num).name
            if signal_num in signal.Signals._value2member_map_
            else f"SIG{signal_num}"
        )
        if signal_num == signal.SIGKILL:
            if "out of memory" in lower or "oom" in lower:
                return "killed", "oom_suspected"
            return "killed", "sigkill"
        if signal_num == signal.SIGTERM:
            return "killed", "sigterm"
        return "killed", signal_name.lower()

    # Positive nonzero returncode → process error
    if "out of memory" in lower or "cuda out of memory" in lower:
        return "failed", "oom_suspected"
    if "traceback" in lower:
        return "failed", "runtime_error"
    if "notimplementederror" in lower or "valueerror" in lower:
        return "failed", "config_validation"
    if "nan" in lower or "fatal: train_loss" in lower:
        return "failed", "nan_divergence"
    return "failed", "nonzero_exit"


def tail(text: str, limit: int = 1200) -> str:
    """Return the last *limit* characters of text."""
    return text[-limit:]
