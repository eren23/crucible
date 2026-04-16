# Agent 3 — Deduplication & DRY

**Axis**: near-duplicate helpers, repeated inline imports, YAML/JSON boilerplate.
**Baseline (post Agent 2)**: ruff 330, mypy 536, pytest 1787 passed / 16 failed / 4 skipped, LOC 37,501.
**Date**: 2026-04-16.

## Summary

Five focused leads inherited from Agents 1–2, plus the plan's YAML/atomic/error-wrap hotspots.
After manual survey and jscpd-skipped triage, **most of the "duplication" leads are already DRY** —
the five plugin-registry modules (`loggers`, `callbacks`, `optimizers`, `schedulers`,
`data_adapters`, `objectives`, `provider_registry`) already delegate to `PluginRegistry`.
Their ~6-line `build_X` dispatchers and 2-line `list_X_detailed` wrappers look duplicated
by shape but differ in **error class semantics** (`PluginError` vs `KeyError`) and **positional
argument conventions** (`(name, params, **kw)` vs `(name, optimizer, **kw)` vs `(name, **kw)`).
Test coverage explicitly asserts these differences (tests/test_objectives.py:172, tests/test_registry_migration.py:85, :146).

Real wins: **14 inline `from crucible.core.log import log_warn`** hoisted to module-top, and
**5 YAML file reads** routed through the canonical `core.io.read_yaml()` helper.

- **Duplication patterns surveyed**: 7
- **Consolidated**: 2 (inline log import, YAML file-read boilerplate)
- **Rejected as premature abstraction**: 4
- **Queued for user review**: 1 (fleet→runner helper borrowing, Agent 2 handoff)

## Critical assessment

### What kind of duplication dominates?

**Inline imports inside exception handlers**, not function-body duplication. The `log_warn`
pattern alone accounts for 25 inline imports across 15 files. Hoisting these to top-level
removes repetitive boilerplate without changing any behavior (import resolution is cached,
and the error paths are rare). This is the single cleanest DRY win.

The second-biggest category is **YAML boilerplate** — `yaml.safe_load(path.read_text(encoding="utf-8"))`
appears 20+ times. `core.io.read_yaml()` already exists as the canonical replacement; only 4–5
call sites can migrate cleanly (the rest have bespoke error handling or are inside `try/except
yaml.YAMLError` blocks that `read_yaml` doesn't emulate).

The `build_X` / `list_X` / `list_X_detailed` fan-out across 7 plugin-type modules *looks*
duplicated to a line-counter, but a closer read shows each module's dispatcher is ~5 lines of
boilerplate that correspond 1:1 to `PluginRegistry.build()` already — the per-module wrappers
exist specifically to expose a **stable public API name** (`build_optimizer`, `build_scheduler`,
etc.) that is imported from ~50 call sites. Collapsing these wrappers would require migrating
those call sites to `OPTIMIZER_REGISTRY.build(...)`, which is a cosmetic renaming with no
actual DRY payoff — the error-message differences are user-facing.

### What's been removed and what's been left alone

**Removed**: inline imports in the hot "log and swallow" pattern; yaml-read-then-cast
scattered across small modules.

**Left alone** (and documented why): `load_nodes` variants (already minimal — three distinct
semantics), the plugin-type `build_X`/`list_X_detailed` wrappers (API-stable public surface).

## Consolidations applied (high-confidence)

| Old locations | New canonical | Reason |
|---------------|--------------|--------|
| `models/architectures/__init__.py:25,62,78` (3 inline imports `as _lw1/2/3`) | top-level `from crucible.core.log import log_warn` | crucible.core.log is leaf — no cycle risk |
| `models/registry.py:143,154,187` (3 inline imports) | top-level | same |
| `core/hub.py:217,220,347,350` (4 inline imports) | top-level | same |
| `runner/wandb_logger.py:229,296,323` (3 inline imports) | top-level | same |
| `researcher/search_tree.py:755` (1 inline) | top-level | already imports from core.log |
| `researcher/llm_client.py:54` | top-level | leaf module |
| `researcher/loop.py:101` (inline `as _lit_warn`) | top-level alias-free | same |
| `runner/tracker.py:148` | top-level | already imports utc_now_iso |
| `runner/experiment.py:588` | top-level | already imports from core.log |
| `runner/loggers.py:143` | top-level | leaf-safe |
| `fleet/sync.py:335,369` (2 inline) | top-level | leaf-safe |
| `core/finding.py:65` | top-level | already imports utc_now_iso |
| `core/plugin_registry.py:203` | top-level (LOCAL import stays — critical file) | kept inline per hard-rule constraint |
| `core/config.py:333,567` | left inline | critical file per hard-rule constraint |
| **YAML file reads** | — | — |
| `mcp/tracer.py:127` (`yaml.safe_load(path.read_text(...))`) | `read_yaml(path)` | direct drop-in — same semantics |
| `researcher/search_tree.py:962,973` (2 inline yaml reads with dict checks) | `read_yaml(path)` + explicit type check | cleanup, same semantics |

## Rejected candidates

1. **`list_X` / `list_X_detailed` pass-throughs (12 copies across 6 modules)**. Each is
   a 2-line function: `return REGISTRY.list_plugins()` / `return REGISTRY.list_plugins_detailed()`.
   Removing them would require migrating ~40 call sites to `REGISTRY.list_plugins()` directly.
   Cost: rename 40 sites for zero code reduction. Tests explicitly verify these names exist.
   **Premature abstraction**: the public names *are* the abstraction.

2. **`build_X` dispatchers (6 copies)**. Each is ~6 lines that check factory presence and
   raise `PluginError` or `KeyError` with a specific message. The `PluginRegistry.build()`
   method does roughly the same, but:
   - optimizer: `factory(params, **kw)` — positional `params`
   - scheduler: `factory(optimizer, **kw)` — positional `optimizer`
   - objective/adapter: raise `KeyError` (not `PluginError`)
   - error message: "Unknown X plugin Y" (registry) vs "Unknown X Y" (wrapper)
   Unifying would need a `build_positional` variant AND force `KeyError`→`PluginError` error
   class change, breaking 4 tests. Not worth it.

3. **`load_nodes` / `load_nodes_if_exists` / `load_nodes_snapshot`** in `fleet/inventory.py`.
   Already 3 distinct semantics: strict-raise, lenient-return-empty, thread-safe-snapshot.
   Bodies are 4-6 lines each and non-overlapping. **Not duplicated** — properly factored.

4. **`yaml.safe_load` with custom error handling** (~8 call sites in `core/config.py`,
   `core/project_template.py`, `core/plugin_schema.py`, `core/tap.py`). Each wraps in
   bespoke `try/except yaml.YAMLError` or appends to a validation result list.
   `read_yaml()` simply returns `None` for missing files, which would change behavior.
   **Not duplicated** — each has distinct error semantics.

## High-confidence (auto-applied) — diff summary

### 1. `log_warn` import hoisting (15 files, ~22 line delta)

Hoisted `from crucible.core.log import log_warn` out of `except` blocks to module-top
imports. Where a top-level `from crucible.core.log import ...` already existed, `log_warn`
was appended to the existing import. Where none existed, a new single-symbol import was
added. Aliases (`_lw1`, `_lw2`, `_lw3`, `_lit_warn`) replaced with plain `log_warn`.

**Net**: 14 duplicate inline imports removed. No behavior change (import is cached;
`log_warn` has no side effects).

**Preserved**:
- `core/plugin_registry.py:203` (critical file per hard-rule constraint — kept inline)
- `core/config.py:333,567` (critical file — kept inline)

### 2. YAML file read consolidation

- `mcp/tracer.py:load_trace_meta` — now calls `read_yaml()`, falling back to empty dict if missing.
- `researcher/search_tree.py:_load_meta` and `_load_nodes` — now use `read_yaml()` with an
  explicit `isinstance(..., dict)` check (same as before, less boilerplate).

## Queued for user review

**Fleet → runner helper borrowing** (Agent 2 handoff). Three function-local imports
(`fleet/project_runner.py` → `runner.output_parser`/`wandb_logger`, `fleet/sync.py` →
`runner.fingerprint`) are layer-rule violations per CLAUDE.md. The fix is to move
`classify_failure`, `fetch_wandb_run_info`, and `safe_git_sha`/`ensure_clean_commit` into
`core/`. This is a ≥5-site refactor touching `fleet/` and `runner/` — deferred.

## Leads for Agent 4 (types)

- **Plugin factory type signatures**: `optimizers.py`, `schedulers.py`, `callbacks.py`,
  `loggers.py`, `data_adapters.py`, `objectives.py`, `provider_registry.py` all declare
  `Callable[..., Any]` for their factory types. A shared `PluginFactory = Callable[..., Any]`
  in `core/types.py` would DRY this.
- **`NodeRecord` (dict[str, Any] alias)** referenced in `fleet/inventory.py`,
  `fleet/manager.py`, `fleet/scheduler.py`, `fleet/bootstrap.py`, `fleet/providers/*`. Check
  `core/types.py` for canonical definition.

## Leads for Agent 5 (Any leaks)

- `collect_public_attrs(obj: Any) -> dict[str, Any]` in `core/io.py` — widely used in
  training backends. `Any` return is load-bearing (could be TypedDict).
- `PluginRegistry` is generic over `T` but the public `get/build` still returns `Any`.
  Tighten once Agent 4 fixes type shapes.

## Leads for Agent 7 (legacy)

- `OBJECTIVE_REGISTRY: dict[...] = _OBJECTIVE_REGISTRY._registry` — *convenience alias*
  exposed for backwards compat. `DATA_ADAPTER_REGISTRY` has the same pattern. Both
  reach into the private `._registry` dict (anti-pattern). Candidate for deprecation.
- Inline `_reg._CURRENT_REGISTER_SOURCE = "builtin"` / `"local"` in
  `models/architectures/__init__.py` is the only caller of that module-level state.
  Probably a refactor-leftover.
