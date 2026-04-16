# Agent 2 — Circular Dependency Untangling

**Axis**: madge-equivalent (import cycles, rule violations, fossil lazy-imports).
**Baseline (post Agent 1)**: ruff 330, mypy 536, pytest 1787 passed / 16 failed / 4 skipped, LOC 37,501.
**Tools**: `grimp 3.14`, manual Tarjan SCC, pydeps (partial — graphviz not available locally).
**Date**: 2026-04-16.

## Summary

The codebase is **remarkably clean** on import cycles. After Agent 1 removed the stale `fleet/provider_registry.py` workaround, only **one SCC of size ≥2** remains in the whole 144-module graph, and it is the deliberate plugin-registration pattern rooted at `models/registry.py`. There are no runtime import cycles — every graph cycle in this SCC goes through **function-local imports** that are only executed when the corresponding public API is called.

- **SCCs**: 1 (size 7, all inside `models/`).
- **Direct reciprocal import pairs**: 2 (`registry`↔`architectures` package, `registry`↔`composer`).
- **Fossil lazy-imports with "circular" comments**: 0 (Agent 1 got the last one).
- **Layer-rule violations** (per CLAUDE.md): 5 (2 trivial/benign, 3 genuine but require structural moves — queued).
- **Redundant TYPE_CHECKING blocks**: 1 (`fleet/bootstrap.py`, half-stale).

## Critical assessment

### 1. Is CLAUDE.md's layering holding up?

**Mostly yes, with three architectural debts.** `core/` has two "violations" that are either benign (`core/config.py` imports `crucible.__version__`, which is just a string constant in the root `__init__.py`) or explicitly-commented deliberate compromises (`core/plugin_discovery.py` function-locally imports `crucible.data_sources` to trigger builtin registration — the comment at line 58 says `"Lives here — not at module level — because core/ must not depend on non-core crucible modules."`).

The **genuine** debt is the **fleet → runner trio**:
- `fleet/project_runner.py` (function-local) → `runner.output_parser`
- `fleet/project_runner.py` (function-local) → `runner.wandb_logger`
- `fleet/sync.py`               (function-local) → `runner.fingerprint`

These are **real** rule violations per CLAUDE.md (`fleet/` and `runner/` must be mutually independent). They're hidden today only because the imports are function-local. Removing the lazy-import escape hatch is a no-op improvement (would just expose the violation at top level). The **principled** fix is to move the three borrowed helpers (`classify_failure`, `fetch_wandb_run_info`, `safe_git_sha`/`ensure_clean_commit`) into `core/` where both layers can depend on them. That's a multi-site refactor that exceeds this agent's scope (see [Queued](#queued-for-user-review)).

### 2. The `models/` plugin-registry SCC is load-bearing

The one remaining SCC (size 7) is:

```
┌─ models.architectures (package __init__)
│   ├─> models.architectures.baseline    ──┐
│   ├─> models.architectures.looped      ──┤
│   ├─> models.architectures.convloop    ──┤
│   ├─> models.architectures.prefix_memory─┤
│   └─> models.registry    <───────────────┤ (each arch imports register_model)
│                                          │
└─> models.registry
        ├─> models.architectures  (function-local in list_families — triggers registration)
        └─> models.composer       (function-local in _register_from_spec_file)

    models.composer
        └─> models.registry       (function-local in register_from_spec — calls register_model)
```

This is the **standard plugin-registration dance**:

- `architectures/__init__.py` imports each sibling (`baseline.py`, …) to trigger their top-level `register_model()` calls.
- Each sibling imports `registry.register_model`. That's not a runtime cycle because `registry.py` has no module-level imports of `architectures`.
- `registry.list_families()` lazily imports the `architectures` package to force registration if needed. That is what closes the graph cycle — grimp records function-local imports too.
- `composer ↔ registry` is the same story: `composer.register_from_spec` takes a parsed spec and calls `registry.register_model`; `registry._register_from_spec_file` is a thin pass-through to `composer.register_from_spec`. Both use function-local imports.

**Breaking this cycle** requires extracting a bare `register_model`-only module (e.g. `models/_registry_core.py`) that both `composer` and `registry` can import without touching each other. That's a minor structural refactor, but it has non-trivial blast radius (tests, plugin author contract, MCP tool descriptions all reference `crucible.models.registry.register_model`). I'm queuing this for user review rather than forcing it — the cycle does **not** manifest as a runtime failure, only as a graph-theoretic annoyance.

### 3. The fossil that Agent 1 fixed was the only one left

Agent 1 removed three `# noqa: F811 lazy to avoid circular` imports from `fleet/provider_registry.py`. I searched the whole tree for similar patterns:

```
grep -rn "# avoid circular\|# lazy to avoid\|# deferred.*circular" src/crucible/
```

Zero hits. Every remaining function-local `from crucible.…` import is one of:
- **Optional-deps insulation** (torch, wandb, anthropic, huggingface) — correct per CLAUDE.md.
- **Start-up-time laziness for CLI dispatchers** (`cli/main.py` only loads the sub-command handler the user actually ran).
- **Cycle-obscuring** (the `models/` SCC above — deliberate).
- **Layer-rule escape hatch** (the fleet → runner three — known debt, queued).
- **`except ImportError` guards** inside `list_families()` style functions — correct.

## Cycle table

| SCC | Participants | Kind | Actionable? |
|---|---|---|---|
| #0 | `models.registry` ↔ `models.composer` ↔ `models.architectures` ↔ each arch file | Deliberate plugin-registration (function-local imports both directions) | **No** — queued |

Total graph cycles: **1 SCC**. Total direct reciprocal pairs: **2**. **No runtime cycles exist.**

## High-confidence fixes (auto-applied)

| # | File | Change | Rationale |
|---|---|---|---|
| 1 | `src/crucible/fleet/bootstrap.py` | Drop `ProjectConfig` from the TYPE_CHECKING block (line 16); keep `ProjectSpec` only. | Fossil. Line 18 eagerly imports `ProjectConfig` at runtime already, so the TYPE_CHECKING import was a stale artifact from before that line was added. Half of the guard was already redundant. `ProjectSpec` remains TYPE_CHECKING-only — it's used purely as a `from __future__ import annotations` type hint. |

That is the only safe, cycle-related auto-apply. Every other candidate either:
- Is a deliberate architectural compromise (commented explicitly),
- Is a runtime-optional import (torch / wandb / anthropic),
- Requires a cross-module move (queued).

**Net effect**: 0 SCCs change, 1 small fossil removed. The SCC count was already at its floor.

## Verification

| Metric | Pre-Agent-2 | Post-Agent-2 | Delta |
|---|---|---|---|
| grimp SCCs (size ≥ 2) | 1 | 1 | 0 |
| direct reciprocal pairs | 2 | 2 | 0 |
| ruff | 330 | 330 | 0 |
| mypy | 536 | 536 | 0 |
| pytest passed / failed | 1787 / 16 | 1787 / 16 | 0 / 0 |
| LOC | 37,501 | 37,500 | -1 |

Import smoke (`import crucible; import crucible.mcp.server; import crucible.cli.main`) passes.

## Queued for user review

| # | Finding | Why it needs user judgment |
|---|---|---|
| 1 | **`fleet/` ↔ `runner/` import leak.** `fleet/project_runner.py` function-locally imports `runner.output_parser.{classify_failure, OutputParser}` and `runner.wandb_logger.fetch_wandb_run_info`; `fleet/sync.py` function-locally imports `runner.fingerprint.{ensure_clean_commit, safe_git_sha}`. Per CLAUDE.md, these layers must be mutually independent. | Fix is to move the three helpers into `core/` — but `classify_failure`'s semantics are genuinely runner-specific (knows about training-script exit codes), and `fetch_wandb_run_info` is wandb-gated. User should decide whether the CLAUDE.md rule stands or whether `runner/` is a legitimate dependency for `fleet/`. |
| 2 | **`models.registry` ↔ `models.composer` SCC.** Function-local imports in both directions. No runtime cycle, but the graph cycle is load-bearing — breaking it requires extracting `register_model` into a sibling module (e.g. `models/_registration.py`) that both can depend on without touching each other. | Multi-site change: tests, plugin-author-facing docs, and MCP tool descriptions all import `crucible.models.registry.register_model` directly. Worth user sign-off before breaking the public import path. |
| 3 | **`models.registry._register_from_spec_file` is a single-call-site pass-through.** Its only in-repo caller is `load_global_architectures` itself. It could be inlined, but doing so doesn't reduce the cycle count (the function-local import in `load_global_architectures` remains) and it was likely kept as public API for plugin authors. | Pure cosmetic simplification. Flag for Agent 3 (dedup) or explicit user ack — may be an intentional extension point. |
| 4 | **`fleet/bootstrap.py:15–16` TYPE_CHECKING survivors.** After Agent 2's fix (`ProjectConfig` removed), the block still holds `ProjectSpec` alone. Could merge with the runtime import on line 18 if `ProjectSpec` is ever used as a runtime value (it isn't today) — but it would be cleaner for Agent 8 to just delete the TYPE_CHECKING block entirely since `from __future__ import annotations` makes `ProjectSpec` annotations stringy. | Micro-cleanup; not cycle-related. |
| 5 | **`core/plugin_discovery.py:60` function-local `import crucible.data_sources`.** Author left an explicit comment (`"core/ must not depend on non-core"`). The import exists to register builtin data-source plugins. The principled fix is to either (a) make `data_sources` a subpackage of `core/`, or (b) register builtins at app startup from the CLI instead of from core. | Architectural decision. |

## Out of scope (for later agents)

- **Type duplication that looks cycle-adjacent** but isn't: `ProjectSpec`, `NodeRecord`, `RunResult`, `FindingRecord` are all defined in `core/types.py` but also have near-siblings in `core/config.py` (`ProjectSpec` the dataclass vs a dict-shaped `project_spec` used in places). Agent 4 territory.
- **`research_dag/bridge.py`** has a long list of optional imports (`core.hub`, `research_dag.node_format`, etc.) that are all legitimate. Noted only to rule out.
- **`researcher/` uses `fleet/manager` function-locally** (`researcher/loop.py:182`, `researcher/harness_optimizer.py:336`). This is within-rules per CLAUDE.md (researcher may import fleet). Lazy to avoid import-time cost when running hypothesis-only flows; keep.
- **`runner/loggers.py:14`**, **`training/callbacks.py:10`**, **`training/optimizers.py:12`**, **`training/schedulers.py:13`** all have a `from crucible.X import build_X` inside a helper function that recursively references the same module. That looks suspicious at first glance but is the "pluggable-builder" idiom — the functions exist as the public `build_*` API and the inner import is a fallback for plugin-discovery callers. Agent 3 (dedup) should check if this can be unified.
- **63 `I001` ruff violations** (unsorted imports) — hands off; Agent 8.

## Specific leads

### For Agent 3 (dedup):
- The `build_X` pattern repeats across `training/optimizers.py`, `training/schedulers.py`, `training/callbacks.py`, `runner/loggers.py` — each has a `build_*` that dispatches via `PluginRegistry`, plus a self-referential function-local import. Probably consolidatable.
- `load_nodes_if_exists` vs `load_nodes` vs `load_nodes_snapshot` across `fleet/inventory.py`, `fleet/scheduler.py`, `fleet/manager.py`, `fleet/bootstrap.py`. Several similar names — check for near-duplication.
- `log_warn` is function-locally imported in ~15 sites (error paths). A `@log_warn_on_failure` decorator or a single module-level import would DRY it.

### For Agent 4 (type consolidation):
- `ProjectSpec` is defined in `core/config.py` and imported under TYPE_CHECKING in `fleet/bootstrap.py`, `fleet/project_runner.py`, `mcp/tools.py`. No duplication — just confirmed canonical location.
- `NodeRecord` (in `core/types.py`) is imported by `fleet/bootstrap.py`, `fleet/inventory.py`, `fleet/manager.py`. Again, canonical.
- **Potential duplication**: `VersionMeta` / `VersionStore` (in `core/store.py` and `core/types.py`). Two names, two modules — Agent 4 should confirm they're distinct or merge.
- **Potential duplication**: Spec/dict patterns. `core/config.py::ProjectSpec` is a dataclass, but many MCP tools pass a `dict[str, Any]` that mirrors the same shape. Canonicalizing would tighten a lot of `Any` leaks for Agent 5.
