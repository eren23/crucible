# Agent 7 — Legacy / Deprecated / Fallback Code

## Summary

Surveyed `src/crucible/` for deprecated symbols, dual code paths, rename
residue, feature-flag branches whose one side is dead, and thin compat
wrappers.  Found **13 candidate sites**; **12 are load-bearing** (on-disk
data compat, public API with tests, or genuine runtime fallbacks);
**1 was truly dead** and removed.

The scarcity isn't an oversight — Agents 1 (dead code), 3 (dedup), and 6
(exceptions) already swept the biggest legacy surface.  What remains is
either:

1. **On-disk / wire-format compatibility** — readers that tolerate
   records written by older Crucible versions (inventory files,
   search-tree metadata, architecture-registry entries, experiment
   results without the newer `contract_status` key).  Removing these
   would require a data migration.
2. **Public API aliases with test coverage** — the `BLOCK_TYPES`,
   `AUGMENTATIONS`, `STACK_PATTERNS`, `OBJECTIVE_REGISTRY`,
   `DATA_ADAPTER_REGISTRY` dicts backing the unified `PluginRegistry`.
   Plugin authors import these names; tests pin the aliasing invariant
   (`BLOCK_TYPES is BLOCK_TYPE_REGISTRY._registry`).
3. **Intended fallbacks** documented inline — fleet-unavailable ->
   local execution in the harness optimizer, val-loss missing -> train
   loss in the generic backend, single-metric trees -> wrap in the new
   multi-metric frontier code.

## Critical assessment

The remaining "legacy"-tagged code is structurally load-bearing, not
cruft.  The dominant kind is **on-disk data compat** — the codebase
reads old data formats (pre-rename `pod_id`, single-metric tree meta,
pre-contract result rows) to avoid breaking existing projects.  These
are not candidates for Agent 7; they'd need a dedicated data-migration
pass (`crucible migrate` or similar) plus a major version bump.

The second-largest category is **public registry aliases**.  These are
by the letter of CLAUDE.md anti-compat-shim rules candidates for
removal, but they are explicitly tested and used by the MCP tool layer
and the composer itself.  Removing them is a public-API change that
exceeds Agent 7's autonomy.

## Removal table

| Symbol / site | Kind | In-repo callers | Tap callers | Verdict |
|---|---|---|---|---|
| `_REGISTRY_META.get(name, {}).get("source", "unknown")` in `models/registry.py:97` | dead fallback (invariant: meta written in lockstep with registry) | 1 (self) | 0 | **Removed** |
| `"builtin"` default in `models/registry.py:32` | defensive guard against corrupted `_REGISTRY_META` | 1 (self) | 0 | Keep — cheap invariant guard |
| `train_gpt.py` (top-level wrapper) | compat shim delegating to `crucible.training.torch_backend.main` | `examples/parameter_golf/crucible.yaml:26`, `src/crucible/templates/projects/lm.yaml:36`, `src/crucible/training/torch_backend.py:4`, `src/crucible/training/__init__.py:1` | unknown | Keep — still the contract for project yamls and the lm template generator |
| `NodeRecord.pod_id` alias | backward-compat for inventory files written before the `pod_id` -> `node_id` rename | `fleet/inventory.py:83,85,137,143,147`, `fleet/providers/runpod.py:645,831,842,848,958,985`, `fleet/manager.py:360,371`, `mcp/tools.py:330,351,3380,3382,3392` | unknown | Keep — actively read by inventory merge and destroy paths |
| `OBJECTIVE_REGISTRY` alias (`training/objectives.py:209`) | public dict view of `_OBJECTIVE_REGISTRY._registry` | `mcp/tools.py:3165`, `tests/test_objectives.py:159` | unknown | Queue — public API, tests pin `__contains__` semantics |
| `DATA_ADAPTER_REGISTRY` alias (`training/data_adapters.py:250`) | same | `mcp/tools.py:3164`, `tests/test_data_adapters.py:11,71,72,84,85,88`, `tests/test_data_adapters_new.py:6,40,72,90` | unknown | Queue — public API |
| `BLOCK_TYPES` / `AUGMENTATIONS` / `STACK_PATTERNS` aliases (`models/composer.py:196,214,366`) | public dict views of the composer plugin registries | `mcp/tools.py:2282,2296,2315,2322`, `models/composer.py:404-406,506-508,634-636`, `tests/test_composer_registries.py` | unknown | Queue — public API, internal lookup sites |
| `_reg._CURRENT_REGISTER_SOURCE` mutation in `models/architectures/__init__.py:9,66` | refactor residue — side-channel state for source tagging | `models/registry.py:11,24,30,129-131,198` | n/a | Keep — built-in architectures rely on this being set before `register_model("baseline", ...)` runs; moving to explicit `source=` would touch every plugin |
| `variants:` dict (`core/config.py:431`) | CLAUDE.md stub calls this "inert" — but it's wired up now | `fleet/project_runner.py:115,130,143`, `mcp/tools.py:3504,3716`, `tests/test_project_spec.py:93`, `tests/test_mcp_project_tools.py:260,301` | n/a | Keep — CLAUDE.md note is stale; lead for Agent 8 |
| `env_defaults` field | CLAUDE.md calls this dead | **Already removed** from `src/crucible/` (not present; Agent 1 swept it) | n/a | n/a |
| `contract_status = "legacy_missing_contract"` defaults in `mcp/tools.py:209,272,3785,3877` | on-disk compat for result rows written before the experiment contract existed | 4 × self | n/a | Keep — data compat |
| `search_tree._get_metrics()` single-metric fallback (`researcher/search_tree.py:622-633`) | on-disk compat for trees created before multi-metric tracking | 1 (self) | n/a | Keep — data compat |
| `HubStore._normalize_architecture_record` (`core/hub.py:695-706`) | injects `relative_path` for old hub-registry entries | 2 (self) | n/a | Keep — data compat |
| `latest_val_bpb=parsed.get("val_bpb")` "kept for output_parser compat" (`runner/experiment.py:136`) | stdout parser still emits `val_bpb` | 1 (self) + parser | n/a | Keep — active parser contract |

## High-confidence removals (applied)

- **`models/registry.py:97`** — dropped the `_REGISTRY_META.get(name, {}).get("source", "unknown")` chain in
  `list_families_detailed`.  `_REGISTRY_META` is written in lockstep
  with `_REGISTRY` (`register_model` lines 37-38) and cleared in
  lockstep (`reset_registry` lines 220-221); every `name in _REGISTRY`
  has a populated meta entry, so the `"unknown"` fallback was
  unreachable.  Now a direct `_REGISTRY_META[name]["source"]` lookup.

## Queued for user review

- **Registry dict aliases** (`OBJECTIVE_REGISTRY`, `DATA_ADAPTER_REGISTRY`,
  `BLOCK_TYPES`, `AUGMENTATIONS`, `STACK_PATTERNS`).  Removing them is a
  refactor: internal call sites in `models/composer.py` and
  `mcp/tools.py` need rewriting from dict operations
  (`name in DICT`, `list(DICT)`, `DICT[name]`) to the
  `PluginRegistry` method API (`registry.is_registered(name)`,
  `registry.list_plugins()`, `registry.get(name)`).  Tests that pin the
  aliasing invariant (`BLOCK_TYPES is BLOCK_TYPE_REGISTRY._registry`)
  need updating.  Public API change; out of scope for an auto-sweep.
- **`train_gpt.py` top-level wrapper.**  Still referenced by
  `src/crucible/templates/projects/lm.yaml` (the default LM template
  copies this script path into generated projects) and by
  `examples/parameter_golf/crucible.yaml`.  Deletion would break every
  existing project yaml.  Queue for a coordinated migration: either
  bake the script into `training/torch_backend.py`'s module entry point
  and update the template, or leave as-is.
- **`_reg._CURRENT_REGISTER_SOURCE` side-channel.**  Aesthetically it's
  refactor residue; functionally it's required because
  `register_model` is called by side-effect from
  `import crucible.models.architectures.baseline` and the caller cannot
  pass `source=` through an import.  A cleaner fix is a
  `with register_source("builtin"): ...` context manager, but that's a
  structural refactor, not legacy cleanup.

## Leads for Agent 8 (comments describing stale refactors)

1. **`CLAUDE.md` itself** — the "What NOT to do" and
   `docs/crucible-config-hierarchy.md` still say `variants:` is "inert"
   and `env_defaults` is a "dead field", but the former is fully wired
   up in `fleet/project_runner.py` + `mcp/tools.py` (with tests) and
   the latter has already been removed from `src/crucible/` (Agent 1
   scope).  Agent 8 should update both docs, or flag them for user
   review.
2. **`src/crucible/fleet/provider.py:9-14`** — reads as an Agent-5
   changelog entry ("Agent 5 tightened the provider interface...").
   This is an "in-motion narration" comment describing a completed
   refactor; candidate for Agent 8's slop sweep.
3. **`src/crucible/runner/experiment.py:136`** — `# kept for
   output_parser compat` is terse but correct; keep.  Flagging it here
   for Agent 8 awareness so it isn't mistaken for stale narration.
4. **`src/crucible/fleet/bootstrap.py:548`** — `# Resolve download
   command: runtime override > config > legacy fineweb default`
   describes current behavior; "legacy" is a qualifier on the
   fineweb default (which *is* active when the Parameter Golf workflow
   runs), not a removable branch.
5. **Module docstrings referring to `train_gpt.py`** —
   `src/crucible/training/__init__.py:1` says "extracted from
   `train_gpt.py`" and `src/crucible/training/torch_backend.py:4`
   says "can be invoked directly from `train_gpt.py`".  Both are
   historical narration.  Candidate for Agent 8.

## Non-findings (verified actively used)

- `pod_id` / `node_id` dual read pattern — the on-disk inventory still
  contains `pod_id`-keyed records; removing the read path would orphan
  every existing RunPod inventory.  **Keep.**
- `contract_status` legacy defaults — same story for results written
  before the contract field existed.  **Keep.**
- `search_tree._get_metrics()` single-metric fallback — same for trees
  created before multi-metric tracking.  **Keep.**
