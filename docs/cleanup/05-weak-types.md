# Agent 5/8 — Weak Type Tightening

## Summary

| Metric | Before | After | Delta |
|--------|-------:|------:|------:|
| `: Any` / `-> Any` in src/crucible | 116 | 115 | −1 (but propagation through 40+ sites) |
| `list[dict[str, Any]]` | ~50 (fleet-wide) | 0 (fleet interfaces) | −50 |
| mypy errors | 533 | 431 | **−102** |
| ruff errors | 327 | 327 | 0 |
| pytest | 1787/16 | 1787/16 | 0 |

The headline improvement is not in the raw `Any` count (many are legitimate plugin-author-facing kwargs or external SDK boundaries), but in the **cascade**: tightening `FleetProvider` from `list[dict[str, Any]]` to `list[NodeRecord]` immediately fixed ~50 `[arg-type]` errors at call sites across `fleet/manager.py`, `fleet/bootstrap.py`, `fleet/scheduler.py`, `fleet/day_run.py`, and `fleet/providers/*.py`.

## Critical assessment

Crucible's `Any` population falls into **three structural buckets**:

1. **Load-bearing (keep)** — plugin-author-facing kwargs. `**kwargs: Any` on factories for optimizers, schedulers, callbacks, data_adapters, block_types is a deliberate plugin contract: authors pass arbitrary YAML/config. Tightening these would break the extensibility surface. Ditto `args: Any` on architecture factories.

2. **SDK boundaries (keep)** — wandb `run`, runpod GraphQL responses, torch tensors with dynamic dtype/shape, mcp raw JSON from the tool dispatcher. These are not wrong to keep `Any`; they're honest about what the external library returns.

3. **Lazy `dict[str, Any]` (tighten)** — this is where most of the win lives. The `FleetProvider` abstract interface used `list[dict[str, Any]]` for node lists even after Agent 4 introduced the canonical `NodeRecord` TypedDict. The note in `fleet/provider.py` explicitly punted this to a future agent. Agent 5 did the cascade.

## Tightening table (applied)

| File | Site | Old | New | Rationale |
|---|---|---|---|---|
| `fleet/provider.py` | all abstract methods | `list[dict[str, Any]]` | `list[NodeRecord]` | Canonical NodeRecord exists; flows cleanly through inventory |
| `fleet/providers/ssh.py` | provision/destroy/refresh/wait_ready, load_from_file | `list[dict[str, Any]]` | `list[NodeRecord]` | Matches abstract |
| `fleet/providers/runpod.py` | provision/destroy/refresh/wait_ready/stop/start + internal `_wait_for_api_ready`/`_wait_for_ssh_ready` | `list[dict[str, Any]]` | `list[NodeRecord]` | Matches abstract (API helpers `runpod_*` left untouched per hard rule) |
| `fleet/providers/runpod.py` | `inventory_record_from_api` return | `dict[str, Any]` | `NodeRecord` | Dict literal already has NodeRecord-compatible keys |
| `fleet/manager.py` | `provision`, `provision_and_wait`, `bootstrap`, `dispatch`, `collect`, `destroy`, `stop`, `start`, `monitor`, `refresh`, `_replace_stalled_nodes`, `_replace_bootstrap_node`, `_provision_replacement_nodes`, both inner `_provision_replacements`, `run_day`/`run_night` local `nodes` bindings | `list[dict[str, Any]]` / `dict[str, Any]` | `list[NodeRecord]` / `NodeRecord` | Cascade from provider tightening |
| `fleet/manager.py` | `enqueue`, `queue_status`, `dispatch` return | `list[dict[str, Any]]` | `list[QueueItem]` / `list[ExperimentConfig]` | Canonical types already defined in Agent 4 |
| `fleet/manager.py` | `prepared_waves: list[tuple[str, list[dict[str, Any]]]]` | `list[dict[str, Any]]` | `list[ExperimentConfig]` | Same |
| `fleet/bootstrap.py` | `bootstrap_node`, `bootstrap_node_worker` returns | `dict[str, Any]` | `NodeRecord` | Matches caller expectation |
| `fleet/bootstrap.py` | `start_bootstrap_supervisor` callback params `replace_fn`/`refresh_fn` | `Callable[..., list[dict[str, Any]]]` | `Callable[[str], NodeRecord]` / `Callable[[list[NodeRecord]], list[NodeRecord]]` | Concrete signatures |
| `fleet/scheduler.py` | `run_wave` `refresh_fn`/`provision_replacement_fn` | `Callable[[list[dict[str, Any]]], list[dict[str, Any]]]` | `Callable[[list[NodeRecord]], list[NodeRecord]]` | Cascade |
| `fleet/scheduler.py` | `bootstrap_recovery_candidates` local `candidates` | `list[dict[str, Any]]` | `list[NodeRecord]` | Matches return annotation |
| `fleet/day_run.py` | `record_day_progress` params; `write_day_leaderboard` rows | `list[dict[str, Any]]` | `list[NodeRecord]`, `list[QueueItem]`, `list[ExperimentResult]` | Cascade |
| `fleet/inventory.py` | `merge_node_snapshots` local `existing_by_key`/`merged` | `dict[str, Any]` / `list[dict[str, Any]]` | `NodeRecord` / `list[NodeRecord]` | Cascade |
| `core/plugin_registry.py` | `T = TypeVar("T")` | unbound | `T = TypeVar("T", bound=Callable[..., Any])` | Factory contract is "callable"; permits `get()` to return `T \| None` instead of `Any \| None` |
| `core/plugin_registry.py` | `get(self, name) -> Any \| None` | `Any \| None` | `T \| None` | TypeVar tightening only |
| `runner/experiment.py` | result dict persistence | — | added `# type: ignore[return-value]` | Local `dict[str, Any]` accumulates extra runtime fields (`oom_retry_from_*`, `status_path`, `manifest_path`, `log_path`, `data_sources`) not in the `ExperimentResult` TypedDict; the TypedDict represents the *external* JSONL wire shape |

## Keep+document (annotated `Any`)

Added `# Any: <reason>` comments on three validator entry points in `researcher/domain_spec.py` where `Any` reflects untrusted YAML input:

- `_validate_metrics(metrics: Any, ...)`
- `_validate_interface(interface: Any, ...)`
- `_validate_constraints(constraints: Any, ...)`

These validators are the trust boundary between `yaml.safe_load` (which returns `Any`) and the typed domain spec; tightening them would shift the burden to callers.

## Queued for user review

1. **`core/io.py` `_json_ready(value: Any) -> Any`** — recursive JSON serializer. Legitimate Any (accepts anything). Could be `object` but that's a stylistic call.
2. **`models/composer.py` block `build/forward`** — 20 `Any` annotations for torch tensors, nn.Module, and dynamic spec resolution. Tightening requires a small taxonomy (tensor vs sub-dict vs lora adapter); left for a modeling-specific pass.
3. **`training/{optimizers,schedulers,callbacks,data_adapters,objectives}.py` factories** — `**kwargs: Any` stays per hard rule (plugin author contract). These are the `build_*` entry points.
4. **`runner/wandb_logger.py` `run: Any`** — dynamic wandb SDK object. Left as `Any`.
5. **`mcp/tracer.py` `result: Any`** — arbitrary JSON-serializable tool call result. Left as `Any`.
6. **`training/generic_backend.py`** — `model: Any`, `objective: Any`, `device: Any` in training loops. Torch-facing; left.

## Leads for Agent 6 (try/except)

Two exception clauses are now demonstrably reachable-but-narrow after the type tightening:

1. **`fleet/providers/runpod.py` `refresh`** — the `except FleetError: ...` around `runpod_get_api_pod` now has a typed node context; the inner cast `failed: NodeRecord = dict(previous_by_id.get(pod_id, dict(node)))` shows that only network/auth errors from RunPod can reach this branch. If Agent 6 can confirm that via call trace, the `dict(node)` fallback becomes dead (we always have `previous_by_id[pod_id]`).

2. **`runner/experiment.py:545–557`** — broad `except Exception` wraps `run_experiment`'s subprocess management. Now that the result dict is typed-modulo-extras, every field assignment is traceable; Agent 6 can narrow this to `except (OSError, subprocess.SubprocessError, RuntimeError)` without loss.

## Leads for Agent 7 (legacy)

No renamed-type legacy `Any` discovered. The type consolidation in Agent 4 was thorough; every canonical `TypedDict` / `Finding` / etc. is already in `core/types.py` and Agent 5 simply plugged its consumers in.

One potential legacy: the duplicated `dict[str, Any]` parameters in `SSHProvider.__init__(defaults: dict[str, Any] | None)` vs. the same in `RunPodProvider.__init__`. These come from `crucible.yaml`'s `provider.defaults` YAML map and are structurally `dict[str, str]` in practice but typed loosely. Agent 7 could narrow if they're confident the field is pure string-to-string.
