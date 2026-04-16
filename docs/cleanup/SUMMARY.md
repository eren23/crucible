# Crucible Cleanup Sweep — Summary (2026-04-16)

Eight sequential cleanup agents ran against `src/crucible/`. Each produced one
commit with its own assessment doc. The sweep preserved test behavior exactly
(1787 passed, same 16 pre-existing failures) while cutting broad-exception
handlers by 70%, mypy errors by 20%, and roughly 116 lines of code overall —
most of which were load-bearing-to-nothing compat shims, in-motion comments, or
swallow-the-error defaults.

## Aggregate metrics

| Metric | Baseline | Final | Delta |
|---|---|---|---|
| ruff errors | 344 | 324 | **−20** |
| mypy errors | 539 | 430 | **−109** |
| LOC | 37,508 | 37,392 | **−116** |
| `except Exception` | 220 | **65** | **−155 (−70%)** |
| `try:` blocks | 374 | 365 | −9 |
| `: Any` / `-> Any` annotations | 116 | 115 | −1 (but 40+ call sites now typed via cascade) |
| pytest passed | 1787 | 1787 | 0 (parity) |
| pytest failed | 16 pre-existing | 16 pre-existing | 0 (parity) |
| Collected tests | 1803 | 1803 | 0 |

Test parity is the **intended** outcome, not a null result: the sweep changed
~1600 lines across 12+ modules without any behavior regression caught by the
1803-test suite.

## Per-agent commits

| # | SHA | Commit | LOC Δ |
|---|---|---|---|
| pre | `a227055` | Extract nested `hermes-agent/` repo out of tree | — |
| pre | `0feeb30` | Capture baselines | — |
| 1 | `5408830` | Remove dead code (vulture + ruff F-rules) | −7 |
| 2 | `5c6f2ca` | Untangle circular imports | −1 |
| 3 | `61eb1b1` | Dedup common helpers (mostly inline→module-top imports) | −14 |
| 4 | `81c125f` | Consolidate type definitions | +33 (`NodeRecord` gained 8 real fields) |
| 5 | `9ead710` | Tighten weak types (cascaded `list[NodeRecord]` through ~40 fleet sites) | +11 |
| 6 | `99c815b` | Narrow / remove defensive exception handling | **−126** |
| 7 | `448778b` | Remove legacy / deprecated / dual-path code | 0 |
| 8 | `03a7f50` | Remove comment slop + in-motion narration | −13 |

Agent 6 did the heaviest lifting on LOC (−126) via exception narrowing in
`mcp/tools.py`. Agent 5 did the heaviest lifting on mypy (−102) via the
cascade from `FleetProvider.list[NodeRecord]` through every fleet consumer.

## Latent bugs surfaced and fixed

1. **`fleet/bootstrap.py:621` — undefined `remote_exec()`**. Hidden by a broad
   `except Exception` that swallowed the `NameError`. The cleanup path (kill
   old training processes before bootstrap) would silently skip. Surfaced by
   Agent 1 (ruff F821), fixed by Agent 6.
2. **`fleet/providers/runpod.py` — unreachable `dict(node)` fallback** in the
   refresh path. `previous_by_id[pod_id]` is always populated. Surfaced by
   Agent 5's type cascade, trimmed by Agent 6.
3. **`NodeRecord` TypedDict missing 8 real fields** (`last_seen_at`, `pod_id`,
   `gpu_count`, `interruptible`, `env_source`, `api_state`, `replacement`,
   `network_volume_id`). Fleet code was writing them without type checking.
   Fixed by Agent 4; cleared 2 mypy `typeddict-unknown-key` errors.

## Latent bugs surfaced, queued for user review

1. **Triple `ValidationResult` naming collision** across
   `core/data_sources.py`, `core/plugin_schema.py`,
   `researcher/candidate_validation.py`. Two have identical shape, the third
   adds `code_hash`. Not unified — behavior risk.
2. **`Finding` TypedDict does not declare `category`** but `core/finding.py`
   reads and writes it. Pre-existing mypy error, not part of Agent 4's remit.

## Architectural notes

- **Stable SCC**: `models.registry` ↔ `models.architectures.*` ↔
  `models.composer` forms a deliberate 7-module plugin-registration SCC using
  function-local imports. Not a runtime cycle; not a design flaw.
- **`fleet/` ↔ `runner/` leak**: three function-local imports
  (`fleet/project_runner.py` → `runner.output_parser` / `wandb_logger`,
  `fleet/sync.py` → `runner.fingerprint`) violate CLAUDE.md's layer rule.
  Fix is hoisting four helpers (`classify_failure`, `fetch_wandb_run_info`,
  `safe_git_sha`, `ensure_clean_commit`) into `core/`. Queued — ≥5-site
  cross-layer refactor.
- **Registry aliases** (`OBJECTIVE_REGISTRY`, `DATA_ADAPTER_REGISTRY`,
  `BLOCK_TYPES`, `AUGMENTATIONS`, `STACK_PATTERNS`) reach into
  `PluginRegistry._registry`. Used by `models/composer.py`, `mcp/tools.py`,
  and `tests/test_composer_registries.py`. Treated as public API for
  tap-authored plugins. Kept.

## Stale documentation fixed

`CLAUDE.md` and `docs/crucible-config-hierarchy.md` both described the
`env_defaults` field as "dead" and the `variants:` dict as "inert". Both were
out of date: `env_defaults` is fully gone, and `variants:` is consumed by
`run_project(variant_name=...)` plus `chain_project_variants` across 7 call
sites. Updated in Agent 8's commit.

## Queued for user review (full list)

Maintained in per-agent assessment docs under `docs/cleanup/0{1..8}-*.md`.
High-impact items:

- `fleet/` ↔ `runner/` helper borrowing (layer rule violation).
- `ProjectSpec` relocation from `core/config.py` → `core/types.py`.
- Merging of the three `ValidationResult` shapes.
- `Finding` TypedDict missing `category` field.
- `train_gpt.py` deletion (referenced by `templates/projects/lm.yaml` and
  `examples/parameter_golf/crucible.yaml` — coordinated migration needed).

## What wasn't touched

- `src/crucible/mcp/server.py` — MCP tool registration / external contract.
- `src/crucible/core/plugin_registry.py` — only a TypeVar bound was added.
- `src/crucible/core/config.py` — `ProjectSpec` stayed put.
- `src/crucible/fleet/providers/runpod.py` — RunPod API surface (one
  unreachable-fallback trim in Agent 6 was the single minimal edit).
- `src/crucible/runner/run_remote.py` — remote contract with scheduler.
- `src/crucible/api/models.py` — pydantic wire models for the REST API.
- `pyproject.toml`, `crucible.yaml` — user-facing schemas.
- `hermes-agent/` — extracted entirely (stray nested repo).

## Verification gauntlet (final run)

```
ruff check src/crucible/          324 errors (was 344)
mypy src/crucible/                430 errors in 83 files (was 539 / 85)
pytest tests/ --ignore=tests/integration -q
                                  1787 passed, 16 failed (pre-existing),
                                  4 skipped, in 262s
import crucible / mcp.server / cli.main     all ok
python -m crucible.cli.main --help          ok
```
