"""Scorer entrypoint — runs baseline.train_and_score and prints val_bpb."""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `python3 scorer.py` from anywhere — the sandbox runs it in the
# cloned workspace, so the import path is the workspace itself.
sys.path.insert(0, str(Path(__file__).parent))

from baseline import train_and_score  # noqa: E402


def main() -> int:
    try:
        score = train_and_score()
    except Exception as exc:  # noqa: BLE001 — surface to caller via exit code
        print(f"scorer-failed:{exc}", file=sys.stderr)
        return 1
    print(f"val_bpb:{score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
