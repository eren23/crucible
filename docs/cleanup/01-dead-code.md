# Agent 1 — Dead Code Removal

**Axis**: knip-equivalent (unused imports, unreferenced locals, unused helpers).
**Baseline**: ruff 344, mypy 539, pytest 1787 passed / 16 failed / 4 skipped, LOC 37,508.
**Date**: 2026-04-16.

## Summary

Very small surface area. The static-analysis tools agreed with each other:

- **ruff F401 (unused imports)**: 11 hits in 7 files.
- **vulture (confidence 80+)**: 5 hits, 4 of them unavoidable callback signatures (torch forward pre-hook `(mod, inputs)`, `signal.signal` `(signum, frame)`). Only 1 genuine: `format_review_content` already double-counted with ruff.
- **vulture (confidence 60)**: ~195 hits but most are dataclass fields used for JSON serialization, TUI framework hooks (`BINDINGS`, `CSS`, `compose`, `on_mount`), MCP server dispatch-by-string functions, `_list_*`/`_list_*_detailed` plugin introspection helpers, and `__all__` re-exports. All either load-bearing at runtime (framework) or flagged for later agents.

The codebase is surprisingly lean on genuine dead code. What the static analysis revealed is not lots of dead weight but three structural patterns:

## Critical assessment

### 1. `fleet/provider_registry.py` has a stale circular-import workaround

Three function-local imports of `FleetProvider` carry the comment `# noqa: F811 lazy to avoid circular`, but:
- The file has `from __future__ import annotations`, so `FleetProvider` in return-type positions is a string and doesn't require an import at call time.
- `fleet/provider.py` does not import `fleet/provider_registry.py` — there is no circular. The original circular was likely broken by an earlier refactor, but the workaround stayed.
- Ruff flags the three imports as F401 (unused) AND separately flags three F821 (`FleetProvider` undefined in the module-level annotations), so the file is broken in two ways at once: it imports a name it doesn't use, and references a name it doesn't import.
- There is also a duplicate `from __future__ import annotations` at lines 16 and 18.

Fix: collapse to a single `TYPE_CHECKING` block at module top. One import, three F401s and three F821s gone, plus an I001.

### 2. `typing.Any` / `typing.TYPE_CHECKING` left behind after refactors

Four files (`cli/tap_commands.py`, `core/data_sources.py`, `fleet/sync.py`, `runner/design.py`) import `Any` or `TYPE_CHECKING` that no longer has a referent. These are the fossil record of "we used to annotate this as Any, then tightened it; nobody pruned the import." Cheap wins.

### 3. `research_dag/bridge.py` imports signals from across its own split

Two imports (`utc_now_iso`, `format_review_content`) reference symbols that this module doesn't actually call. `format_review_content` is used by `tests/test_research_dag.py` (so the function itself stays); the bridge just imported it speculatively.

### 4. Non-findings worth flagging for later agents

- Duplicate `from __future__ import annotations` in `fleet/provider_registry.py:16,18` — Agent 3 (dedup) or Agent 8 should also spot this; I will fix it as an I001 byproduct.
- `remote_exec` undefined in `fleet/bootstrap.py:621` — ruff F821. This is a latent bug, not dead code. Flagged for Agent 6 or user review.
- 100% vulture hits on `inputs`/`frame`/`signum` are false positives: they're mandatory parameters of torch/signal callback signatures. Leaving them alone.
- `provider_name = "runpod"` / `"ssh"` class attributes are unused in-repo but look like plugin-API hooks. Queued for user review — may be public contract.

## High-confidence removals (auto-applied)

| File | Line | Symbol | Kind | Verification |
|---|---|---|---|---|
| `src/crucible/cli/tap_commands.py` | 10 | `typing.Any` | unused import | No `Any` references anywhere in file |
| `src/crucible/core/data_sources.py` | 13 | `typing.TYPE_CHECKING` | unused import | No `TYPE_CHECKING` references anywhere in file |
| `src/crucible/core/hub.py` | 40 | `crucible.core.finding.FINDING_STATUSES` | unused import | Only used inside `finding.py` itself |
| `src/crucible/fleet/provider_registry.py` | 18 | duplicate `from __future__ import annotations` | duplicate statement | Ruff I001 |
| `src/crucible/fleet/provider_registry.py` | 39, 77, 97 | lazy `from crucible.fleet.provider import FleetProvider` (x3) | unused import | Replaced with one `TYPE_CHECKING` import — no runtime circular confirmed |
| `src/crucible/fleet/sync.py` | 7 | `typing.Any` | unused import | No `Any` references anywhere in file |
| `src/crucible/research_dag/bridge.py` | 21 | `crucible.core.log.utc_now_iso` | unused import | Only referenced in this file once, in the import |
| `src/crucible/research_dag/bridge.py` | 27 | `format_review_content` from `node_format` | unused import | Function still exists in `node_format.py` and is used by tests |
| `src/crucible/runner/design.py` | 8 | `typing.Any` | unused import | No `Any` references anywhere in file |

Net effect: 10 removals (one of them is a duplicate `__future__` statement), fixes 10 F401 violations, 3 F821 violations, and 1 I001 violation. Actual ruff delta: 344 -> 330.

## Verification (Phase 3 results)

| Metric | Baseline | After | Delta |
|---|---|---|---|
| ruff | 344 | 330 | -14 |
| mypy | 539 | 536 | -3 |
| pytest failed | 16 | 16 | 0 |
| pytest passed | 1787 | 1787 | 0 |
| LOC (src/crucible) | 37,508 | 37,501 | -7 |

Import smoke passed.

## Queued for user review

| File | Line | Symbol | Reason |
|---|---|---|---|
| `src/crucible/runner/wandb_logger.py` | 280 | `import wandb` (inside `fetch_wandb_metrics` try block) | Ruff says unused but it is acting as an `ImportError` probe; replacement is `importlib.util.find_spec` (behavior change — Agent 6 territory). |
| `src/crucible/fleet/bootstrap.py` | 621 | `remote_exec(...)` call | F821, symbol is genuinely undefined. This is a latent bug in a non-hot code path, not dead code. Escalate. |
| `src/crucible/fleet/providers/runpod.py` | 684 | `provider_name = "runpod"` | Class attribute never read in-repo. May be part of plugin-author public API for reflection. |
| `src/crucible/fleet/providers/ssh.py` | 52 | `provider_name = "ssh"` | Same as above. |
| `src/crucible/researcher/tree_loop.py` | 19 | `class TreeSearchResearcher` | vulture flags the class and 4 methods as unused. May be exposed via MCP or CLI — needs cross-repo grep the user can validate. |

Five items — below the 20-item halt threshold.

## Out of scope (for later agents)

- `src/crucible/core/types.py` has ~60 unused dataclass fields per vulture. These are load-bearing for serialization round-trips; Agent 4 (type consolidation) should review whether all of them still correspond to written YAML/JSON.
- `_list_*` / `_list_*_detailed` helpers across `runner/loggers.py`, `training/callbacks.py`, `training/optimizers.py`, etc. — vulture flags them as unused, but they implement a consistent introspection API. Agent 3 (dedup) should evaluate whether this pattern is worth keeping or can be moved to `PluginRegistry.list_plugins_detailed()` (which already exists).
- MCP `_status_event_name` in `src/crucible/mcp/tools.py:3403` — `mcp/` is a critical file per CLAUDE.md. Not touching.
- TUI `BINDINGS` / `CSS` / `compose` / `on_mount` in `src/crucible/tui/app.py` — Textual framework hooks, not dead.
- Composer `__new__` at `src/crucible/models/composer.py:382` — looks like metaclass machinery, out of scope without understanding the class hierarchy.
- `harness_optimizer.pareto_node_ids`, `literature.get_paper_detail`, `search_tree.load_candidate`, `tree_loop.*` — unused in-repo but the researcher/ module is young (recent commits) and these are likely MCP-exposed. Agent 7 (legacy/deprecated) should revisit after checking MCP dispatch.

## Out-of-scope observations (not dead code, noticed during sweep)

- 63 I001 (unsorted imports) — Agent 8 or a bulk `ruff --fix` at the end.
- 167 E501 (line too long) — stylistic, user call.
- 220 `except Exception` — Agent 6.
