# Agent 4 — Type Definition Consolidation

**Axis**: consolidate types split across modules; promote `dict[str, Any]` shapes with stable fields to typed dataclasses/TypedDicts; surface latent field divergences.
**Baseline (post Agent 3)**: ruff 329, mypy 536, pytest 1787 passed / 16 failed / 4 skipped.
**Date**: 2026-04-16.

## Summary

The codebase's type story is **mostly coherent**: `core/types.py` already hosts
18 TypedDicts covering the main data shapes (versioning, fleet, hub, search
tree, tap). What's scattered is a small set of factory callables and three
coincidentally-named `ValidationResult` classes. The plan's predicted
hotspots (run results, fleet nodes, design/spec) are already centralised;
Agents 1–3 didn't leave a mess behind.

- **Types inventory**: 54 declarations across 10 files.
- `core/types.py`: 18 TypedDicts (unchanged count; **3 fields added** to `NodeRecord`).
- `core/config.py`: 16 dataclasses (critical file, not touched per hard rule).
- `api/models.py`: 8 pydantic `BaseModel`s (HTTP wire boundary — stays pydantic).
- Remaining: 12 dataclasses spread across `models/composer.py`, `core/data_sources.py`, `core/plugin_schema.py`, `core/project_template.py`, `researcher/{candidate_validation,domain_spec}.py`, `runner/output_parser.py`.
- **High-confidence consolidations applied**: 2 (one alias, one TypedDict expansion).
- **Queued for user review**: 1 (triple `ValidationResult` naming collision).

## Critical assessment

### Is the codebase's type story coherent?

Mostly yes. Three observations:

1. **`core/types.py` is the de-facto canonical home** for cross-module
   record shapes, and it's used correctly. 18 TypedDicts, well-grouped by
   concern (versioning, experiment, knowledge, search tree, tap). No
   duplicate TypedDicts.
2. **`core/config.py` is a parallel canonical home** for 16 dataclasses
   representing parsed YAML config (`ProjectConfig`, `ProjectSpec`,
   `ProviderConfig`, etc.). The split between TypedDict (runtime records)
   and dataclass (parsed config) is deliberate and works. Per the
   approved plan and hard rules, `ProjectSpec` stays in `core/config.py`;
   any move is queued for the user.
3. **One fleet-layer leak**: `dict[str, Any]` flows through ~40 call sites
   in `fleet/manager.py` despite `NodeRecord` being defined. Tightening
   the abstract `FleetProvider` signature to `list[NodeRecord]` cascades
   into a fleet-wide migration (see *Attempted but reverted* below).

### What pattern (dataclass vs TypedDict vs BaseModel) is working?

The codebase uses each representation for a distinct purpose:

- **TypedDict**: runtime records that round-trip through JSON/YAML
  (fleet inventory, version ledger, search tree). Total=False throughout
  so partial records parse cleanly.
- **Dataclass**: parsed config with defaults (`ProjectConfig`) and internal
  records that need behaviour (`ValidationResult` in data sources has a
  `.errors` property; `DomainSpec` has `.metric_names`).
- **Pydantic `BaseModel`**: HTTP wire format (`api/models.py`). Validation
  is load-bearing at the API boundary.

This is a reasonable pattern and should be preserved.

### What's not working

**Triple `ValidationResult` collision** — three unrelated classes share the
same name in three different modules:

| File | Fields | Purpose |
|------|--------|---------|
| `core/data_sources.py:57` | `valid: bool, errors: list[str], warnings: list[str]` | Data-source plugin validation |
| `core/plugin_schema.py:83` | `path: Path, ok: bool, issues: list[ValidationIssue]` | Plugin manifest validation |
| `researcher/candidate_validation.py:26` | `valid: bool, errors: list[str], warnings: list[str], code_hash: str` | Harness candidate validation |

The first and third are nearly identical (first 3 fields match). They could
converge into one shared type, but doing so requires adding an optional
`code_hash` field — behavior change, queued. The second is genuinely
different and should be renamed `ManifestValidationResult` to end the
naming overlap.

## Consolidations applied (high-confidence)

| Change | From | To | Scope |
|--------|------|------|-------|
| `PluginFactory` type alias | 7 copies of `Callable[..., Any]` in plugin factories | `core/types.py::PluginFactory` | 5 modules updated |
| `NodeRecord` field expansion | missing `last_seen_at`, `pod_id`, `gpu_count`, `interruptible`, `api_state`, `replacement`, `network_volume_id`, `env_source` | canonical `NodeRecord` in `core/types.py` | removes 2 `typeddict-unknown-key` errors in `fleet/bootstrap.py` |

### 1. `PluginFactory` alias

Added `PluginFactory: TypeAlias = Callable[..., Any]` to `core/types.py`.
Applied to:
- `training/optimizers.py` — `OPTIMIZER_REGISTRY` + `register_optimizer` signature
- `training/schedulers.py` — `SCHEDULER_REGISTRY` + `register_scheduler` signature
- `fleet/provider_registry.py` — `PROVIDER_REGISTRY` + `register_provider` signature
- `models/registry.py` — `_REGISTRY` + `register_model` signature
- `models/composer.py` — `BLOCK_TYPES` and `AUGMENTATIONS` convenience aliases

**Net**: 5 modules now share one canonical factory type. `Callable[..., Any]`
remains the underlying shape (this is intentional — see the docstring at
the declaration; per-plugin-type tightening is Agent 5's target).

### 2. `NodeRecord` field expansion

Added 8 fields to `NodeRecord` that were being written by fleet code
without type acknowledgement:

- `pod_id: str` — RunPod-provider alias of `node_id`
- `gpu_count: int`
- `interruptible: bool`
- `env_source: str` — which `.env.*` file to sync for secrets
- `api_state: str` — provider-reported lifecycle (vs local `state`)
- `last_seen_at: str | None` — used by bootstrap and refresh flows
- `replacement: bool` — re-provisioned to replace a dead node
- `network_volume_id: str`

Also documented the full enum for `state`: `new | bootstrapped | running |
dead | destroyed | ready | unreachable`.

**Net**: clears 2 mypy `typeddict-unknown-key` errors in `fleet/bootstrap.py`
at lines 581 and 756, each on `node["last_seen_at"] = utc_now_iso()`. These
were **real field additions hiding in the wild** — the TypedDict was lagging
the fleet code by multiple releases.

## Attempted but reverted

Tightening `fleet/provider.py` (the `FleetProvider` abstract interface)
and `fleet/providers/ssh.py` to use `list[NodeRecord]` instead of
`list[dict[str, Any]]`. This introduced 15+ new mypy errors in
`fleet/manager.py` because the manager still uses `list[dict[str, Any]]`
for ~40 call sites. Tightening the interface cascades into a fleet-wide
migration that's out of scope for type consolidation; see *Leads for
Agent 5* for the migration target. Reverted; docstring note added at
`fleet/provider.py` explaining why it stays on `dict[str, Any]`.

## Queued for user review

### Q1. Triple `ValidationResult` naming collision

Three classes in three modules all named `ValidationResult`, two with
nearly identical shape, one structurally different. Options:

- **Rename `core/plugin_schema.py::ValidationResult` → `ManifestValidationResult`** (mechanical, 4 call sites updated). Fixes the worst overlap.
- **Merge `core/data_sources.py::ValidationResult` and `researcher/candidate_validation.py::ValidationResult`** into a single `core/types.py::ValidationResult` with optional `code_hash`. This is a behavior change for both — the researcher one uses it in `as_dict()`, the data-sources one doesn't expose it at all. Queued because one of the downstream `*.data_sources/*.py` plugins (huggingface, local_files, wandb_artifact) constructs `ValidationResult(valid=..., errors=..., warnings=...)` positionally. Adding a field breaks 6 call sites unless given a default.

Recommendation: rename `plugin_schema.ValidationResult` first (cheap),
defer the data-source / researcher merge (needs API contract decision).

### Q2. Consider whether `ProjectSpec` should live in `core/types.py`

Currently in `core/config.py:403` as a dataclass. Per the hard rule
(`core/config.py` is critical, don't edit). Since `core/types.py` hosts
TypedDicts and `ProjectSpec` is a dataclass with field defaults, moving
it doesn't improve the story unless the whole parsed-config dataclass
cluster moves. Keep it where it is.

## Leads for Agent 5 (weak types)

1. **`fleet/manager.py` `list[dict[str, Any]]` → `list[NodeRecord]`**. 40+
   call sites now have a clean target type in `core/types.py`. The
   `FleetProvider` abstract interface also wants this but cascades through
   manager — migrate manager first, then tighten the interface.

2. **`fleet/provider.py::FleetProvider`** abstract signatures. Once
   manager is migrated, lift `list[dict[str, Any]]` → `list[NodeRecord]` on
   all 6 abstract methods and the two optional `stop`/`start` defaults.
   Also applies to `fleet/providers/ssh.py` concrete methods.

3. **`runner/experiment.py` return shapes**. Likely returns `dict[str, Any]`
   matching `ExperimentResult`. Tighten to `ExperimentResult`.

4. **`analysis/leaderboard.py` result list entries**. Should be
   `list[ExperimentResult]` — Agent 3 didn't find duplicate types here,
   but the `dict[str, Any]` plumbing is the weak part.

5. **`core/plugin_registry.py::PluginRegistry[T].get()/build()` returning
   `Any`**. Generic is declared but return types don't propagate; see the
   `no-any-return` error at line 103. Needs TypeVar bound on `T` to fix.
   Critical file rule limits Agent 5 here — only the TypeVar bound is
   in-scope.

6. **`Callable[..., Any]`** signatures remain in `mcp/server.py:97,213`
   (excluded per hard rule — MCP dynamic dispatch).

## Leads for Agent 7 (legacy)

- **`BLOCK_TYPES` / `AUGMENTATIONS` convenience aliases** in
  `models/composer.py` (lines 195, 213) reach into `PluginRegistry._registry`
  (private). Agent 3 already flagged the same pattern on
  `OBJECTIVE_REGISTRY`/`DATA_ADAPTER_REGISTRY`. Candidates for deprecation
  once callers migrate to `BLOCK_TYPE_REGISTRY.list_plugins()` or
  `BLOCK_TYPE_REGISTRY._registry[name]`.

- **`pod_id` aliased to `node_id` in `NodeRecord`**. Old field name for
  backward compatibility with RunPod-era code. Worth auditing whether any
  call site still reads `pod_id` vs `node_id`; if all readers use
  `node_id`, the alias can be dropped.

## Verification

```
ruff:   329 → 327 (−2 unused Callable imports removed)
mypy:   536 → 533 (−3; 2 from NodeRecord expansion, 1 from dead Callable import)
pytest: 1787 passed / 16 failed / 4 skipped  (unchanged — same 16 baseline failures)
imports: ok
```
