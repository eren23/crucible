"""ERA-style code-mutation demo — exercises Crucible Phase 5.1 end-to-end.

What it does:
1. Builds an ``AstLocalEditPolicy`` aimed at ``baseline.py``.
2. Proposes three concrete mutations (one per knob: ACTIVATION,
   LEARNING_RATE, HIDDEN_DIM).
3. Applies each in a sandboxed clone, scores via ``scorer.py``,
   compares to the unmutated baseline.
4. Prints the leaderboard.

No LLM needed — the mutations are hand-picked. For a richer demo
that uses ``LlmDiffPolicy`` plus an orchestrator's LLM, replace
``_HAND_PICKED_MUTATIONS`` with diffs returned by your LLM via
``llm_diff_request_prompt`` / ``llm_diff_parse_response``.

Run:
    PYTHONPATH=src python3 examples/code_mutation_era_replica/run_demo.py
"""
from __future__ import annotations

import json
from pathlib import Path

from crucible.researcher.code_mutation import (
    AstLocalEditPolicy,
    MutationProposal,
    SandboxConfig,
    SandboxRunner,
    ScorerConfig,
    score_stdout,
)


_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE  # the example dir IS the sandboxed project for this demo


_HAND_PICKED_MUTATIONS = [
    MutationProposal(
        name="gelu_activation",
        target_file="baseline.py",
        diff=json.dumps({"kind": "swap_literal", "old": "relu", "new": "gelu"}),
        hypothesis="GELU is smoother than ReLU near zero",
        rationale="reduces dead-neuron failure mode on a 4-hidden-unit net",
        mutation_scope=["baseline.py"],
    ),
    MutationProposal(
        name="lr_0p2",
        target_file="baseline.py",
        diff=json.dumps({"kind": "swap_literal", "old": 0.5, "new": 0.2}),
        hypothesis="LR=0.5 may be too aggressive for 50 epochs",
        rationale="smaller LR stabilises late-epoch oscillation",
        mutation_scope=["baseline.py"],
    ),
    MutationProposal(
        name="hidden_dim_16",
        target_file="baseline.py",
        diff=json.dumps({"kind": "swap_literal", "old": 4, "new": 16}),
        hypothesis="4 hidden units underfit the sin+cos target",
        rationale="2x sinusoid superposition needs ≥8 units empirically",
        mutation_scope=["baseline.py"],
    ),
]


def _baseline_score() -> float | None:
    """Score the unmutated baseline so the leaderboard has a reference."""
    import subprocess  # noqa: S404 — scoring helper, not user-controlled

    result = subprocess.run(
        ["python3", str(_HERE / "scorer.py")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return score_stdout(result.stdout, r"val_bpb:([0-9]+\.?[0-9]*)")


def main() -> int:
    print(f"# ERA-style mutation demo — project: {_PROJECT_ROOT}")
    baseline = _baseline_score()
    print(f"baseline val_bpb: {baseline}")

    scorer = ScorerConfig(
        cmd=["python3", "scorer.py"],
        score_pattern=r"val_bpb:([0-9]+\.?[0-9]*)",
        direction="minimize",
    )
    sandbox = SandboxRunner(
        _PROJECT_ROOT, sandbox_root=Path("/tmp") / "code_mutation_era_replica_sandbox"
    )
    policy = AstLocalEditPolicy(
        project_root=_PROJECT_ROOT,
        scorer=scorer,
        sandbox=sandbox,
        sandbox_config=SandboxConfig(timeout_seconds=60),
    )

    leaderboard: list[tuple[str, float | None, str | None]] = [
        ("baseline", baseline, None),
    ]
    for proposal in _HAND_PICKED_MUTATIONS:
        problems = policy.validate(proposal)
        if problems:
            print(f"  [{proposal.name}] validate: {problems}")
            leaderboard.append((proposal.name, None, "; ".join(problems)))
            continue
        result = policy.apply(proposal)
        flag = "ok" if result.success else "fail"
        print(f"  [{proposal.name}] {flag} score={result.score} err={result.error}")
        leaderboard.append((proposal.name, result.score, result.error))

    print("\n# Leaderboard (lower val_bpb is better)")
    ranked = sorted(
        leaderboard,
        key=lambda row: (row[1] is None, row[1] if row[1] is not None else 0.0),
    )
    for name, score, err in ranked:
        if score is None:
            print(f"  {name:24s}  FAIL  {err}")
        else:
            print(f"  {name:24s}  {score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
