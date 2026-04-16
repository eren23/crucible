"""Harness runner: remote entrypoint for MemorySystem-style candidates.

This launcher is invoked by the Crucible runner. It reads the candidate
identifier from ``HARNESS_CANDIDATE_ID``, loads the candidate source from
the tree's ``candidates/`` directory, instantiates the harness class, and
runs a predict / learn_from_batch loop, emitting metrics in the stdout
format recognized by Crucible's ``OutputParser``:

    step:{step}/{total} train_loss:{loss}
    step:{step}/{total} val_loss:{loss} val_bpb:{bpb}

Extra metrics declared in the domain spec are emitted as key=value tokens
appended to the final status line.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _load_candidate_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("harness_candidate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_harness_class(module: Any, class_name: str | None) -> type:
    """Locate the concrete candidate class.

    Prefers classes DEFINED in *module* over imported symbols so that when a
    candidate subclasses ``MemorySystem`` (which may itself be importable in
    the module), we instantiate the subclass instead of the imported ABC.
    """
    module_defined: list[type] = [
        obj for obj in vars(module).values()
        if isinstance(obj, type) and obj.__module__ == module.__name__
    ]

    if class_name:
        # Exact-match on a module-defined class.
        for cls in module_defined:
            if cls.__name__ == class_name:
                return cls
        # Otherwise: a module-defined subclass of the named base.
        for cls in module_defined:
            for base in cls.__mro__[1:]:
                if base.__name__ == class_name:
                    return cls
        # Last resort: an imported symbol matching by name (original lookup).
        cls = getattr(module, class_name, None)
        if cls is not None:
            return cls
    if module_defined:
        return module_defined[0]
    raise RuntimeError("No candidate class found in module")


def _load_dataset(spec_path: Path | None) -> list[dict[str, Any]]:
    """Minimal synthetic dataset so the launcher runs without external data.

    Real domains ship a data adapter — this keeps the launcher self-contained
    for unit testing and smoke runs. Override by setting ``HARNESS_DATASET``
    to a path to a JSONL of ``{"input": ..., "target": ...}`` records.
    """
    override = os.environ.get("HARNESS_DATASET")
    if override and Path(override).exists():
        return [
            json.loads(line)
            for line in Path(override).read_text().splitlines()
            if line.strip()
        ]
    # Tiny in-memory synthetic set (8 examples, binary classification).
    return [
        {"input": f"example {i}", "target": i % 2}
        for i in range(8)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default=os.environ.get("HARNESS_CANDIDATE_ID"))
    parser.add_argument("--candidates-dir", default=os.environ.get("HARNESS_CANDIDATES_DIR"))
    parser.add_argument("--class-name", default=os.environ.get("HARNESS_CLASS_NAME", "MemorySystem"))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("HARNESS_STEPS", "1")))
    args = parser.parse_args()

    if not args.candidate or not args.candidates_dir:
        print("ERROR: --candidate and --candidates-dir are required", file=sys.stderr)
        return 2

    path = Path(args.candidates_dir) / f"{args.candidate}.py"
    if not path.exists():
        print(f"ERROR: candidate source not found at {path}", file=sys.stderr)
        return 2

    module = _load_candidate_module(path)
    cls = _find_harness_class(module, args.class_name)
    harness = cls()

    dataset = _load_dataset(None)
    total = max(args.steps, 1)
    correct = 0
    token_total = 0

    for step in range(total):
        batch_results: list[dict[str, Any]] = []
        start = time.time()
        for example in dataset:
            pred = harness.predict(example)
            target = example.get("target")
            is_correct = int(pred == target or (isinstance(pred, dict) and pred.get("label") == target))
            correct += is_correct
            token_total += len(str(pred))
            batch_results.append({"example": example, "prediction": pred, "correct": is_correct})
        # Optional online learning hook.
        if hasattr(harness, "learn_from_batch"):
            harness.learn_from_batch(batch_results)
        elapsed = time.time() - start
        train_loss = 1.0 - (correct / max(len(dataset) * (step + 1), 1))
        print(f"step:{step + 1}/{total} train_loss:{train_loss:.4f} elapsed:{elapsed:.2f}")

    n_total = len(dataset) * total
    accuracy = correct / max(n_total, 1)
    tokens_per_example = token_total / max(n_total, 1)
    val_bpb = max(1e-6, 1.0 - accuracy)  # BPB-style proxy so the legacy metric stays populated.

    # Emit multi-metric axes in the generic form OutputParser collects.
    print(f"metric:accuracy={accuracy:.6f}")
    print(f"metric:tokens_per_example={tokens_per_example:.6f}")
    print(f"metric:latency_ms=0.0")

    # Completion signal: final_<tag> val_loss:<f> val_bpb:<f>
    print(f"final_harness val_loss:{val_bpb:.4f} val_bpb:{val_bpb:.4f}")
    # Model-bytes line in the regex-compatible form (must include the colon).
    print("serialized_model_int8_zlib: 0 bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
