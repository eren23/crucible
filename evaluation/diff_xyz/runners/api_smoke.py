"""API-driven Diff-XYZ smoke runner.

Invokes `evaluation.diff_xyz.harness.run_benchmark` against OpenAI / Anthropic /
Google APIs. Used to validate the harness on a small sample before committing
to a full sweep.

Env vars (all optional with sane defaults):
    DIFFXYZ_MODEL           model spec, default 'openai:gpt-4.1-mini'
    DIFFXYZ_TASK            apply | anti_apply | diff_gen, default 'apply'
    DIFFXYZ_FORMAT          udiff | udiff-h | udiff-l | search-replace, default 'udiff'
    DIFFXYZ_SYSTEM          none | format, default 'format'
    DIFFXYZ_LIMIT           int, default 20
    DIFFXYZ_LANGS           comma-separated, default '' (all 5)
    DIFFXYZ_SEED            int, default 0
    DIFFXYZ_OUT             output JSON path, default '/workspace/project/result.json'

Required keys (forwarded by Crucible env_forward):
    OPENAI_API_KEY    (if model is openai:*)
    ANTHROPIC_API_KEY (if model is anthropic:*)
    GOOGLE_API_KEY    (if model is google:*)

Stdout contract: emits ``RESULT EM=<float> IoU=<float>`` on the last line so
Crucible's stdout metrics parser (`metrics.source: stdout`) picks it up.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    model = os.environ.get("DIFFXYZ_MODEL", "openai:gpt-4.1-mini")
    task = os.environ.get("DIFFXYZ_TASK", "apply")
    fmt = os.environ.get("DIFFXYZ_FORMAT", "udiff")
    sys_mode = os.environ.get("DIFFXYZ_SYSTEM", "format")
    limit = int(os.environ.get("DIFFXYZ_LIMIT", "20"))
    seed = int(os.environ.get("DIFFXYZ_SEED", "0"))
    raw_langs = os.environ.get("DIFFXYZ_LANGS", "").strip()
    langs = [s.strip() for s in raw_langs.split(",") if s.strip()] or None
    out_path = Path(os.environ.get("DIFFXYZ_OUT", "/workspace/project/result.json"))

    print(f"[api_smoke] model={model} task={task} fmt={fmt} sys={sys_mode} "
          f"limit={limit} langs={langs}", flush=True)

    # Local sys.path tweak so the script runs whether invoked from the repo root
    # or any subdir; mirrors what the harness CLI does.
    repo_root = Path(__file__).resolve().parents[3]
    for p in (repo_root, repo_root / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    from evaluation.diff_xyz.harness import run_benchmark
    from evaluation.diff_xyz.models import ModelError

    try:
        result = run_benchmark(
            model_spec=model, task=task, fmt=fmt, sys_mode=sys_mode,
            limit=limit, langs=langs, seed=seed, progress=True,
        )
    except ModelError as exc:
        print(f"[api_smoke] FAIL ModelError: {exc}", file=sys.stderr, flush=True)
        return 2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    em = result["metrics"].get("EM", 0.0)
    iou = result["metrics"].get("IoU", 0.0)
    print(f"[api_smoke] wrote {out_path}", flush=True)
    print(f"RESULT EM={em:.4f} IoU={iou:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
