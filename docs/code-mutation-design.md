---
layout: default
title: Code Mutation Design (Phase 5+)
---

# Code Mutation Design — Phase 5+ Tracking Doc

> **Status:** Interface stub only (Phase 3.6 landed). Full
> implementation deferred to Phase 5+ per the AI-native-discovery
> plan — Sakana AI Scientist v2 spent ~6 weeks on the equivalent
> surface, so we explicitly scope it out of Phase 3 to keep the
> ecosystem deliverables shippable.

## What's Live Now

`src/crucible/researcher/code_mutation.py` ships:

- `MutationProposal` dataclass — the orchestrator-supplied diff envelope
- `MutationResult` dataclass — execution outcome shape
- `CodeMutationPolicy` ABC — the contract a real implementation satisfies
- `StubCodeMutationPolicy` — raises `CodeMutationNotImplemented` on `apply()`
- Tiny registry: `register_code_mutation_policy` / `get_code_mutation_policy`

Downstream callers (tree expansion policies, future Codex-MCP integration) write against this surface today. When a real policy lands, the call sites don't shift.

## What Phase 5+ Needs to Build

### 1. Diff parsing + validation (~1 week)

Use `unidiff` (already a transitive dep via several plugins) to:

- Parse the proposed unified diff
- Verify the target file matches the diff header
- Confirm hunks apply cleanly against the on-disk file at the proposal's commit hash
- Reject diffs that touch files outside the configured `mutation_scope` (e.g., refuse to patch `crucible.yaml` or `requirements.txt`)

### 2. AST-aware safety filters (~1 week)

Walk the post-mutation AST and reject patterns that are out of scope:

- Shell-escape calls (subprocess wrappers that take user-controlled input)
- Network calls (`requests.*`, `urllib.*`, `socket.*`) in training paths
- Filesystem writes outside the project tree
- Dunder-attribute overrides on locked classes (e.g., overriding `__class__` to bypass type checks)
- Modifications to security-sensitive modules (`src/crucible/core/redact.py`, etc.)

A user-supplied allowlist relaxes specific gates per project (e.g., a project that legitimately needs network access for data download whitelists `urllib.request`).

### 3. Sandboxed execution (~2 weeks)

Each mutation runs in a fresh subprocess with:

- Working directory: a tempdir clone of the project (rsync, hardlinks where possible)
- Resource limits via `resource.setrlimit` (CPU time, RSS, file size)
- No network unless the allowlist authorizes it
- Output captured into the standard Crucible runs.jsonl ledger
- Timeout enforced at the policy level (default 1h; configurable)

The existing `fleet/` SSH provider could host this — the mutation is "just" a Crucible experiment with a custom training script — but a local sandbox is the right MVP because mutations should fail fast before consuming pod time.

### 4. Score harvesting + tree integration (~1 week)

When the mutation completes:

- Parse the standard `step:N/M val_loss:X val_bpb:Y` stdout pattern
- Surface the primary metric as the policy's `score`
- Capture artifacts (model bytes, logs, the diff itself) under the run's `artifacts` dict
- Optional: record the diff as a git-style note on the parent tree node so the lineage is browsable

### 5. Rollback + version store integration (~1 week)

- Apply the diff via `git apply` or `unidiff.PatchSet.apply()` on the cloned tempdir
- Never touch the user's actual working tree
- Persist successful mutations as VersionStore entries (resource type `code_mutation`) so they're browseable, comparable, and reproducible

## Integration Points

### Tree expansion

A mutation policy registers as a `code_mutation` policy and becomes selectable in `tree_create({"expansion_policy": "code_mutation"})`. Each `tree_expand_node` call asks the policy for child mutations rather than env-var overrides.

### External MCP (Codex)

Phase 3.5's `external_mcp` infrastructure lets the autonomous loop call out to a Codex MCP server that generates the diffs. The loop:

1. `external_mcp_call("codex", "propose_diff", {...})` returns a unified diff
2. Wraps as `MutationProposal`
3. Hands to the active `CodeMutationPolicy.apply()` for sandboxed exec

This decouples diff generation (Codex's strength) from validation / sandbox / scoring (Crucible's strength). The plan called out exactly this split in Phase 3.5.

## Open Questions

- **Granularity of mutations.** One-line tweaks (replace `relu` → `gelu`) vs. multi-hunk refactors. The MVP supports both via unified-diff, but the policy may want a `max_hunks` gate to fail fast on overly ambitious proposals.
- **Cross-file mutations.** Real research moves can touch `model.py` AND `train.py`. Phase 5 supports this; the validator just walks each hunk's target.
- **Reverting failed mutations.** Each policy runs in a fresh tempdir clone, so the user's tree is never touched. No revert needed for the in-flight failure case. The mutation's success-or-not is just a row in `runs.jsonl`.

## Why Not Ship Now?

Per the plan refinement notes (`docs/positioning.md` and the main plan doc):

> Phase 3 code-mutation MVP is 4-6 weeks alone — the same surface Sakana AI Scientist v2 spent its bulk on.

Phase 3 ships 5 deliverables in ~3-4 weeks: arxiv + openreview ingestion, evaluators scaffold, Optuna bridge, external_mcp consumption. Squeezing in a 6-week implementation would either delay the rest of Phase 3 or ship a half-baked mutation MVP. The chosen split is:

1. Phase 3.6 (this doc + the interface stub) — ~1 day. Locks in the contract.
2. Phase 5+ — the real implementation, with the design above.

In the meantime, users who need code mutation today:

- Run autoresearch via Phase 1.6's `crucible research import autoresearch <path>` — autoresearch's mutation loop runs locally, Crucible orchestrates the fleet around it.
- Or wire a Codex MCP server via Phase 3.5's `external_mcp` and call its `propose_diff` tool manually from the autonomous loop.
- Or integrate Sakana AI Scientist v2 / AutoResearchClaw via the same `external_mcp` route — "don't reinvent autoresearch's mutation loop" is an explicit project value (see plan's "What NOT to do").

## Tracking

Issue to file when starting Phase 5: "Implement CodeMutationPolicy — diff/sandbox/score". Reference this doc + the plan's Part C Phase 3.4 entry.
