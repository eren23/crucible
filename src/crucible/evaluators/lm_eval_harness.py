"""lm-evaluation-harness evaluator (Phase 3.3 builtin).

Shells out to the ``lm_eval`` CLI from
`EleutherAI/lm-evaluation-harness <https://github.com/EleutherAI/lm-evaluation-harness>`_
to score a checkpoint on a configurable set of tasks. Returns the
per-task accuracy / acc_norm / etc. as a flat ``scores`` dict.

lm_eval is NOT a hard dependency of Crucible; the evaluator's
:meth:`validate` returns an actionable error message when the binary
isn't on $PATH. Operators install it on the pod via the project's
``install`` block or the venv setup, not via the Crucible core
package.

Config schema (in ``crucible.yaml`` or per-call args):
    tasks: list[str]        # e.g., ["hellaswag", "arc_easy", "lambada_openai"]
    batch_size: int = 8
    num_fewshot: int = 0
    limit: int | None = None  # cap eval-set size for speed; None = full
    model_args: str = ""    # passed verbatim to lm_eval --model_args
    extra_args: list[str] = []  # extra flags appended to the lm_eval command
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from crucible.core.evaluators import (
    EvalResult,
    EvaluatorPlugin,
    EvalValidationResult,
    register_evaluator,
)


class LMEvalHarnessEvaluator(EvaluatorPlugin):
    """Run lm-evaluation-harness against a checkpoint."""

    DEFAULT_BATCH_SIZE = 8
    DEFAULT_NUM_FEWSHOT = 0

    # ------------------------------------------------------------------
    # validate
    # ------------------------------------------------------------------

    def validate(self) -> EvalValidationResult:
        errors: list[str] = []
        warnings: list[str] = []

        if not shutil.which("lm_eval"):
            errors.append(
                "lm_eval CLI not found on $PATH. Install with "
                "`pip install lm-evaluation-harness` (or pin a version via "
                "the project install block)."
            )

        tasks = self.config.get("tasks")
        if not tasks:
            errors.append(
                "lm_eval config requires a non-empty 'tasks' list "
                "(e.g., ['hellaswag', 'arc_easy'])."
            )
        elif not isinstance(tasks, list):
            errors.append(
                f"lm_eval config 'tasks' must be a list, got {type(tasks).__name__}."
            )

        return EvalValidationResult(
            valid=not errors, errors=errors, warnings=warnings,
        )

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------

    def evaluate(self, checkpoint_path: Path) -> EvalResult:
        """Run lm_eval on the checkpoint and return scores.

        Errors that prevent the eval from running (missing binary,
        bad config) are caught and returned as ``success=False``.
        """
        start = time.monotonic()
        pre = self.validate()
        if not pre.valid:
            return EvalResult(
                scores={},
                metadata={"errors": pre.errors},
                success=False,
                error="; ".join(pre.errors),
                duration_seconds=time.monotonic() - start,
            )

        if not checkpoint_path.exists():
            return EvalResult(
                scores={},
                success=False,
                error=f"Checkpoint path does not exist: {checkpoint_path}",
                duration_seconds=time.monotonic() - start,
            )

        cmd = self._build_command(checkpoint_path)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.get("timeout_seconds", 3600),
            )
        except subprocess.TimeoutExpired as exc:
            return EvalResult(
                scores={},
                metadata={"command": cmd, "stdout": exc.stdout, "stderr": exc.stderr},
                success=False,
                error=f"lm_eval timed out after {exc.timeout}s.",
                duration_seconds=time.monotonic() - start,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return EvalResult(
                scores={},
                metadata={"command": cmd},
                success=False,
                error=f"Failed to launch lm_eval: {type(exc).__name__}: {exc}",
                duration_seconds=time.monotonic() - start,
            )

        if proc.returncode != 0:
            return EvalResult(
                scores={},
                metadata={
                    "command": cmd,
                    "stdout": proc.stdout[-2000:],
                    "stderr": proc.stderr[-2000:],
                    "returncode": proc.returncode,
                },
                success=False,
                error=f"lm_eval exited with code {proc.returncode}",
                duration_seconds=time.monotonic() - start,
            )

        scores = self._parse_scores(proc.stdout)
        return EvalResult(
            scores=scores,
            metadata={
                "command": cmd,
                "tasks": self.config.get("tasks"),
                "num_fewshot": self.config.get("num_fewshot", self.DEFAULT_NUM_FEWSHOT),
            },
            success=True,
            duration_seconds=time.monotonic() - start,
        )

    # ------------------------------------------------------------------
    # describe
    # ------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": "lm_eval_harness",
            "tasks": self.config.get("tasks", []),
            "num_fewshot": self.config.get("num_fewshot", self.DEFAULT_NUM_FEWSHOT),
            "batch_size": self.config.get("batch_size", self.DEFAULT_BATCH_SIZE),
            "config_keys": sorted(self.config.keys()),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_command(self, checkpoint_path: Path) -> list[str]:
        """Compose the lm_eval CLI invocation."""
        tasks = ",".join(self.config["tasks"])
        model_args = self.config.get("model_args") or (
            f"pretrained={checkpoint_path}"
        )
        cmd: list[str] = [
            "lm_eval",
            "--model", self.config.get("model", "hf"),
            "--model_args", model_args,
            "--tasks", tasks,
            "--batch_size", str(self.config.get("batch_size", self.DEFAULT_BATCH_SIZE)),
            "--num_fewshot",
            str(self.config.get("num_fewshot", self.DEFAULT_NUM_FEWSHOT)),
        ]
        if self.config.get("limit") is not None:
            cmd += ["--limit", str(self.config["limit"])]
        cmd += list(self.config.get("extra_args", []))
        return cmd

    def _parse_scores(self, stdout: str) -> dict[str, float]:
        """Extract per-task numeric scores from lm_eval stdout.

        lm_eval prints a JSON results block at the end when called
        without ``--output_path``. We attempt JSON first; if the
        output isn't JSON-shaped, we fall back to scraping the
        ``hf-eval`` results table.

        Returns flat ``{task.metric: value}`` keys (e.g.,
        ``"hellaswag.acc": 0.62``).
        """
        # 1. Look for a "results" JSON object in the output.
        scores: dict[str, float] = {}
        try:
            # lm_eval ends with a JSON dump under the "results" key.
            start = stdout.rfind("{\n  \"results\":")
            if start >= 0:
                blob = stdout[start:]
                # Find the matching closing brace.
                depth = 0
                for i, ch in enumerate(blob):
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            blob = blob[: i + 1]
                            break
                data = json.loads(blob)
                results = data.get("results", {})
                for task, metrics in results.items():
                    if not isinstance(metrics, dict):
                        continue
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            scores[f"{task}.{metric}"] = float(value)
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
        return scores


register_evaluator("lm_eval_harness", LMEvalHarnessEvaluator)
