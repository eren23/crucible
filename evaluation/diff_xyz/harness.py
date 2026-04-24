"""Diff-XYZ evaluation harness — the CLI that wires every piece together.

Usage:
    python evaluation/diff_xyz/harness.py \\
        --model anthropic:claude-sonnet-4-6 \\
        --task apply \\
        --format udiff \\
        --system-prompt format \\
        --limit 20 \\
        --out /tmp/smoke.json

Output JSON schema (mirrors the eval_watcher contract so the Tier-13 daemon
can consume it unchanged):

{
  "model": "anthropic:claude-sonnet-4-6",
  "task": "apply",
  "format": "udiff",
  "system_prompt": "format",
  "n_samples": 20,
  "metrics": { "EM": 0.95, "IoU": 0.99, ... },
  "per_lang": { "python": {...}, ... },
  "per_sample": [ {repo, lang, em, iou, ...}, ... ]
}
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from evaluation.diff_xyz.dataset import DiffXYZSample, load_samples
from evaluation.diff_xyz.formats import SUPPORTED_FORMATS
from evaluation.diff_xyz.metrics import (
    compute_apply_metrics,
    compute_diff_gen_metrics,
)
from evaluation.diff_xyz.models import ModelError, resolve_backend
from evaluation.diff_xyz.prompts import (
    TASKS,
    strip_markdown_fence,
    system_prompt,
    user_prompt,
)


# ---------------------------------------------------------------------------
# Per-sample scoring
# ---------------------------------------------------------------------------


@dataclass
class SampleResult:
    """Metrics + context for a single sample."""

    idx: int
    lang: str
    task: str
    fmt: str
    em: float
    iou: float
    parsing_rate: float
    applying_rate: float
    f1_plus: float
    f1_minus: float
    response_chars: int
    elapsed_s: float
    error: str = ""


def score_sample(
    backend,
    sample: DiffXYZSample,
    idx: int,
    task: str,
    fmt: str,
    sys_mode: str,
    max_tokens: int,
    temperature: float,
) -> SampleResult:
    sys_p = system_prompt(fmt, sys_mode)
    usr_p = user_prompt(task, fmt, sample)

    t0 = time.time()
    try:
        raw = backend.generate(
            sys_p, usr_p, max_tokens=max_tokens, temperature=temperature
        )
    except ModelError as exc:
        return SampleResult(
            idx=idx, lang=sample.lang, task=task, fmt=fmt,
            em=0.0, iou=0.0, parsing_rate=0.0, applying_rate=0.0,
            f1_plus=0.0, f1_minus=0.0,
            response_chars=0, elapsed_s=round(time.time() - t0, 2),
            error=str(exc),
        )
    elapsed = round(time.time() - t0, 2)

    cleaned = strip_markdown_fence(raw)

    if task == "apply":
        m = compute_apply_metrics(cleaned, sample.new_code)
        return SampleResult(
            idx=idx, lang=sample.lang, task=task, fmt=fmt,
            em=m.em, iou=m.iou,
            parsing_rate=0.0, applying_rate=0.0, f1_plus=0.0, f1_minus=0.0,
            response_chars=len(cleaned), elapsed_s=elapsed,
        )
    if task == "anti_apply":
        m = compute_apply_metrics(cleaned, sample.old_code)
        return SampleResult(
            idx=idx, lang=sample.lang, task=task, fmt=fmt,
            em=m.em, iou=m.iou,
            parsing_rate=0.0, applying_rate=0.0, f1_plus=0.0, f1_minus=0.0,
            response_chars=len(cleaned), elapsed_s=elapsed,
        )
    if task == "diff_gen":
        m = compute_diff_gen_metrics(
            predicted_diff=cleaned,
            reference_diff=sample.diff_for(fmt),
            old_code=sample.old_code,
            new_code=sample.new_code,
            fmt=fmt,
        )
        return SampleResult(
            idx=idx, lang=sample.lang, task=task, fmt=fmt,
            em=m.em, iou=m.iou,
            parsing_rate=m.parsing_rate, applying_rate=m.applying_rate,
            f1_plus=m.f1_plus, f1_minus=m.f1_minus,
            response_chars=len(cleaned), elapsed_s=elapsed,
        )
    raise ValueError(f"unknown task {task!r}")


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _aggregate(results: list[SampleResult], task: str) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {}
    agg: dict[str, Any] = {
        "EM": _mean([r.em for r in results]),
        "IoU": _mean([r.iou for r in results]),
    }
    if task == "diff_gen":
        agg.update({
            "parsing_rate": _mean([r.parsing_rate for r in results]),
            "applying_rate": _mean([r.applying_rate for r in results]),
            "f1_plus": _mean([r.f1_plus for r in results]),
            "f1_minus": _mean([r.f1_minus for r in results]),
        })
    errors = [r for r in results if r.error]
    if errors:
        agg["errors"] = len(errors)
    return agg


def _aggregate_per_lang(
    results: list[SampleResult], task: str
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[SampleResult]] = defaultdict(list)
    for r in results:
        buckets[r.lang or "(unknown)"].append(r)
    return {lang: _aggregate(rows, task) for lang, rows in buckets.items()}


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def run_benchmark(
    model_spec: str,
    task: str,
    fmt: str,
    sys_mode: str = "format",
    limit: int | None = None,
    langs: list[str] | None = None,
    seed: int = 0,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    progress: bool = True,
) -> dict[str, Any]:
    """Run Diff-XYZ benchmark for a (model, task, format, system_prompt) combination."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; use {TASKS}")
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"unknown format {fmt!r}; use {sorted(SUPPORTED_FORMATS)}")

    backend = resolve_backend(model_spec)
    samples = load_samples(limit=limit, langs=langs, seed=seed)

    results: list[SampleResult] = []
    t0 = time.time()
    for idx, sample in enumerate(samples):
        r = score_sample(
            backend, sample, idx, task, fmt, sys_mode, max_tokens, temperature
        )
        results.append(r)
        if progress and ((idx + 1) % 10 == 0 or idx + 1 == len(samples)):
            m = _aggregate(results, task)
            print(
                f"[{idx + 1}/{len(samples)}] EM={m.get('EM', 0):.3f} "
                f"IoU={m.get('IoU', 0):.3f}",
                file=sys.stderr,
                flush=True,
            )

    return {
        "model": model_spec,
        "task": task,
        "format": fmt,
        "system_prompt": sys_mode,
        "n_samples": len(results),
        "metrics": _aggregate(results, task),
        "per_lang": _aggregate_per_lang(results, task),
        "per_sample": [asdict(r) for r in results],
        "elapsed_s": round(time.time() - t0, 2),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="diff_xyz_harness",
        description="Diff-XYZ benchmark harness (arXiv 2510.12487).",
    )
    p.add_argument("--model", required=True,
                   help="backend:model_id (e.g., anthropic:claude-sonnet-4-6, dummy:echo).")
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--format", dest="fmt", default="udiff",
                   choices=sorted(SUPPORTED_FORMATS))
    p.add_argument("--system-prompt", dest="sys_mode", default="format",
                   choices=["none", "format"])
    p.add_argument("--limit", type=int, default=None,
                   help="Max samples (None = all 1000).")
    p.add_argument("--langs", nargs="*", default=None,
                   help="Subset of languages (python/javascript/java/kotlin/rust).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--out", required=True, help="Output JSON path.")
    # eval_watcher compatibility — unused here, but accepted so the watcher can
    # invoke us with its checkpoint arg.
    p.add_argument("--checkpoint", default="", help=argparse.SUPPRESS)
    p.add_argument("--no-progress", action="store_true", help="Suppress progress prints.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        out = run_benchmark(
            model_spec=args.model,
            task=args.task,
            fmt=args.fmt,
            sys_mode=args.sys_mode,
            limit=args.limit,
            langs=args.langs,
            seed=args.seed,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            progress=not args.no_progress,
        )
    except ModelError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    m = out["metrics"]
    print(
        f"✓ {args.task}/{args.fmt}/{args.sys_mode} n={out['n_samples']} "
        f"EM={m.get('EM', 0):.3f} IoU={m.get('IoU', 0):.3f}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
