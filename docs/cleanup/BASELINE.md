# Crucible Cleanup — Baseline (captured 2026-04-16)

Captured just after commit `a227055` (hermes-agent extraction) and just before Agent 1 runs.

## Static analysis

| Tool | Errors | File |
|---|---|---|
| ruff | 344 | `baseline-ruff.txt` |
| mypy | 539 (85 files) | `baseline-mypy.txt` |

## Tests

`PYTHONPATH=src pytest tests/ --ignore=tests/integration`

- **Collected**: 1803
- **Passed**: 1787
- **Failed**: 16 (pre-existing — documented below)
- **Skipped**: 4
- **Time**: 75s

Full run: `baseline-tests-run.txt`.

### Pre-existing failures (do not treat as regressions)

```
tests/test_mcp_project_tools.py::TestProvisionProject::test_uses_next_index_and_interruptible_override
tests/test_project_remote_roundtrip.py::test_run_project_and_collect_results_roundtrip
tests/test_project_remote_roundtrip.py::test_multi_node_launch_uses_launch_id_and_per_node_run_ids
tests/test_scheduler_math.py::TestCosineScheduler::test_cosine_no_warmup_starts_at_1
tests/test_scheduler_math.py::TestCosineScheduler::test_cosine_decays_to_zero_at_end
tests/test_scheduler_math.py::TestCosineScheduler::test_cosine_midpoint_is_half
tests/test_scheduler_math.py::TestCosineScheduler::test_cosine_with_warmup
tests/test_scheduler_math.py::TestCosineScheduler::test_cosine_min_lr_scale
tests/test_scheduler_math.py::TestConstantScheduler::test_constant_with_warmup_ramps
tests/test_scheduler_math.py::TestLinearScheduler::test_linear_decays_to_zero
tests/test_scheduler_math.py::TestLinearScheduler::test_linear_with_min_lr_scale
tests/test_scheduler_math.py::TestLinearScheduler::test_linear_with_warmup
tests/test_scheduler_math.py::TestCosineRestartsScheduler::test_cosine_restarts_returns_scheduler
tests/test_world_model_example.py::TestBouncingBallAdapter::test_registration
```

## Pass/fail rule for every agent

- Ruff error count must not **increase** above 344.
- Mypy error count must not **increase** above 539.
- Test failure count must not **increase** above 16.
- Test **collected** count may decrease only if the agent explicitly removed tests (and explained why in its assessment doc).
- The import smoke (`python -c "import crucible; import crucible.mcp.server; import crucible.cli.main"`) must pass.

## Raw counts (for delta tracking)

| Metric | Baseline |
|---|---|
| LOC (src/crucible) | 37,508 |
| `: Any` / `-> Any` annotations | 116 |
| `try:` blocks | 374 |
| `except Exception` | 220 |
| `TODO` / `FIXME` / `XXX` | 0 (!) |

The 0 TODOs means Agent 8's focus shifts from stale TODO hunting to AI-slop docstrings, in-motion narration, and redundant comments.

220 `except Exception` across 374 `try:` blocks is a very high ratio (~59%). CLAUDE.md forbids bare `except Exception`, so Agent 6 has a clear mandate.
