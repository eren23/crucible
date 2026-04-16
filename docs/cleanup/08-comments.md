# Agent 8/8 — Comment / Slop / Stub Cleanup

## Summary

Final agent in the 8-agent sequential sweep. Scope: remove AI-slop docstrings,
in-motion narration, and agent-era meta-comments; update the two stale claims
in `CLAUDE.md` and `docs/crucible-config-hierarchy.md` that prior agents flagged.

| Metric | Before | After | Δ |
|---|---|---|---|
| ruff | 324 | 324 | 0 |
| mypy | 430 | 430 | 0 |
| pytest (non-integration) | 1787 passed / 16 failed / 4 skipped | 1787 / 16 / 4 | 0 |
| `src/crucible` LOC | 37,405 | 37,392 | −13 |
| Agent-era comments (`# Agent N`, `# cleanup`) | 2 | 0 | −2 |
| In-motion narration blocks (`extracted from`) | 2 | 0 | −2 |
| Stale doc claims on `env_defaults` / `variants:` | 3 files | 0 | −3 |

## Critical assessment

The codebase was already remarkably clean after Agents 1–7. The earlier agents'
high-signal comment additions (Agent 6's `# noqa: BLE001` reason suffixes,
Agent 5's `# Any: {reason}` guards) are all load-bearing and were kept.

The only substantial slop that remained:

1. **In-motion narration paragraphs** — two module docstrings narrated the
   "extracted from `train_gpt.py`" refactor. That's a historical fact about
   the code's provenance, not information a reader needs to use the module.
2. **Agent-era comment** — `fleet/provider.py` had a 6-line comment explaining
   what Agent 5 had done. Pure changelog.
3. **Stale doc claims** — `CLAUDE.md` and `docs/crucible-config-hierarchy.md`
   described `variants:` as "inert" and `env_defaults` as "dead", both of
   which are obsolete: `env_defaults` was removed from `ProjectSpec` entirely,
   and `variants:` is now consumed by `run_project(variant_name=...)` and
   `chain_project_variants` (grep confirms 7 call sites across `fleet/` and
   `mcp/`, including tests).

Beyond those specific leads, grep-based sweeps for common slop patterns
(`# now uses`, `# was`, `# previously`, `# for now`, `# temporary`, `TODO`,
`FIXME`, `XXX`, `HACK`) returned zero matches in `src/crucible/`. Sampling
the most-touched modules (experiment.py, registry.py, bootstrap.py) showed
only purpose-oriented docstrings — no "This function takes X and returns Y"
slop survives anywhere.

## Removals

### `src/crucible/fleet/provider.py` (−7 lines)

Removed Agent-5 changelog header describing the `list[dict[str, Any]] →
list[NodeRecord]` tightening. The type hint on the class already expresses
the current contract.

Before:
```python
# Node records use ``crucible.core.types.NodeRecord`` (a ``TypedDict`` with
# ``total=False``). Agent 5 tightened the provider interface from
# ``list[dict[str, Any]]`` to ``list[NodeRecord]``. Provider-specific keys
# (``pod_id``, ``api_state``, etc.) are already part of ``NodeRecord``, so
# existing call sites flow through cleanly. See
# ``docs/cleanup/05-weak-types.md``.
```

After: deleted.

### `src/crucible/training/__init__.py` (−3 lines)

Before:
```python
"""Crucible training module — extracted from train_gpt.py.

Contains the PyTorch training loop, hyperparameters, optimizers,
data loading, validation, quantization, and TTT evaluation.
"""
```

After:
```python
"""PyTorch training loop, optimizers, data loading, validation, quantization, TTT eval."""
```

### `src/crucible/training/torch_backend.py` (−3 lines)

Dropped the "extracted from the original `train_gpt.py`" paragraph. Kept the
functional "invoke directly or via crucible runner / MCP tools" hint because
that is usage info, not history.

## CLAUDE.md update

Only the `docs/crucible-config-hierarchy.md` summary bullet was touched, per
the project rule that CLAUDE.md is user-maintained.

Before:
> ... three real bugs that have already cost hours (`nodes.json` interruptible
> echo, inert `variants:` dict, dead `env_defaults` field), the correct
> playbook for running a project variant (you must inline the env dict in
> `run_project(overrides=...)` until the variants dict is wired up) ...

After:
> ... the `nodes.json` interruptible echo bug, the correct playbook for
> running a project variant (pass `variant_name` to `run_project` or inline
> env via `overrides`) ...

## `docs/crucible-config-hierarchy.md` updates

- Executive summary dropped from "four surprising facts" to three; deleted
  `env_defaults` dead-code item and rewrote the variants item to describe
  the current dispatch path.
- §4 rewritten from "Known bug: `variants:` dict is inert" to "Running a
  variant" with accurate file pointers (`mcp/tools.py:3496-3514`,
  `fleet/project_runner.py:112-203`).
- §6 worked-example table: removed the `env_defaults is dead code` row and
  the paragraph that followed. Replaced with a one-line statement of where
  env vars are allowed to come from.
- §9 gotcha list: removed items #1 and #4 (variants/env_defaults), added a
  new item explaining current variant dispatch. List shrunk from 10 to 9.
- "Bugs to fix" section: removed items #2 and #4 (now addressed).

## Kept (tempted to delete, but load-bearing)

- `mcp/tools.py:1947-1950` `# noqa: BLE001` block with 3-line reason comment
  — Agent 6 added it; the rationale ("user-submitted code may raise anything
  at import time") is exactly the kind of non-obvious constraint our rubric
  says to keep.
- `mcp/tools.py:2579` / `:2612` Step 1 / Step 2 markers — they navigate a
  two-phase fallback (local logs → SSH fetch). Useful for a reader scanning
  a 70-line function.
- `models/registry.py` 19-line register_model docstring — verbose, but every
  paragraph describes a non-obvious precedence rule, not the signature.
- Provider abstract-method docstrings — they specify the contract each
  backend must honour. Not slop.
- The remaining `# noqa: F401 — triggers registration` comments on
  architecture imports — they explain why an unused import is intentional.

## Final state after the 8-agent sweep

Comment hygiene is now at a point where grep-based pattern detection for
slop returns empty on all standard patterns. The remaining comments in
`src/crucible/` fall into two buckets:

1. Load-bearing `# why` comments (precedence rules, security guards,
   platform quirks, intentional bare-except rationales).
2. Module/class/function docstrings that describe purpose — none restate
   the signature, none narrate history.

Further comment pruning would need per-module subjective judgement and is
out of scope for automated sweeps.

## Suggestions for orchestrator SUMMARY.md

Metrics worth aggregating across all 8 agents:

- **Total LOC delta** — every agent shaved lines. Cumulative number
  demonstrates the scope of the sweep better than any single agent's diff.
- **Baseline parity** — ruff / mypy / test counts stayed flat across all
  8 commits. That flat line IS the result: cleaner code, same behaviour.
- **`Any` count** (Agent 5 metric), **`try/except` count** (Agent 6), and
  **legacy-flag count** (Agent 7) — each went to zero or near-zero in
  their respective phases. Worth a single "before → after" bar chart.
- **Comment hygiene signals** — `TODO`/`FIXME`/`XXX` baseline was 0,
  remained 0. Agent-era comments dropped from 2 to 0. "Extracted from"
  / "Agent N" narration count: 4 → 0.

Aggregate note for the final summary: the 8-agent sequential design worked
as intended — each agent's domain was distinct enough that they didn't
step on each other, and later agents (6/7/8) benefited materially from
earlier ones' structural cleanups.
