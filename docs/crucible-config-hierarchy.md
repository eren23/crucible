# Crucible Config Hierarchy & Provisioning Reference

> **Read this before touching project specs, crucible.yaml, taps, or any `provision_project` / `bootstrap_project` / `run_project` call.**
>
> This is the definitive map of which config layer wins at each stage of the
> fleet / bootstrap / run flow, plus the `nodes.json` interruptible-echo bug
> that has already cost hours of debugging. Everything in this doc is
> cross-referenced to exact `file:line` in the `src/crucible/` tree.

## Executive summary

Crucible pulls config from ~12 layers that interact in non-obvious ways. The
three most surprising facts:

1. **Project spec resolution is first-match-wins, not merge.** `local > hub > tap`. If you have both a local `.crucible/projects/<name>.yaml` AND a tap-provided `~/.crucible-hub/taps/*/projects/<name>.yaml`, the local one **completely replaces** the tap one — no layering. ([§1](#1-project-spec-loading))
2. **`nodes.json` records `pod.interruptible` from the RunPod API *response*, not the input.** RunPod's response often omits the field, so the fallback chain lands on either the previous record's value or `True` regardless of what you asked for. Your yaml edit may have worked at the API layer but `nodes.json` lies to you. ([§3](#3-known-bug-nodesjson-interruptible-echo))
3. **Variants are dispatched via `variant_name=...` on `run_project`.** The runner looks up `spec.variants[name]` and merges its env dict into overrides before launch; passing `overrides={...}` directly still works. ([§4](#4-running-a-variant))

If you don't read anything else in this doc, read [§7 Correct playbook for running a project variant](#7-correct-playbook-for-running-a-project-variant).

---

## 1. Project spec loading

**Function**: `crucible.core.config.load_project_spec(name, project_root)` at `src/crucible/core/config.py:452`.

**Resolution order (first-match-wins, NOT merge):**

| Rank | Path | File:line |
|---|---|---|
| 1 (highest) | `<project_root>/.crucible/projects/<name>.yaml` (local) | `config.py:466` |
| 2 | `~/.crucible-hub/projects/<name>.yaml` (hub, reserved) | `config.py:478` |
| 3 (lowest) | `~/.crucible-hub/taps/*/projects/<name>.yaml` (taps, alphabetical) | `config.py:480-483` |

**Commit**: rank 3 (tap walking) was added in `b5dcf5c` (Track C fix C11, 2026-04-10). Before that commit, users had to manually `cp` tap specs to `.crucible/projects/` to use them.

**The silent trap**: if a user has a local spec AND a tap spec with the same name, the local one wins completely and no one tells them the tap version exists. Edits to the tap version simply never apply. Detect this with:

```bash
ls .crucible/projects/<name>.yaml ~/.crucible-hub/taps/*/projects/<name>.yaml 2>&1
```

If both exist, delete or rename the local one to let the tap version take effect.

---

## 2. Provision flow: project spec → RunPod API

Walks from `provision_project` MCP call all the way to the RunPod GraphQL / REST call.

| Step | Code | What happens |
|---|---|---|
| 1 | `mcp/tools.py:3387` | `spec = load_project_spec(project_name, config.project_root)` — fresh read per call, no caching between invocations. |
| 2 | `mcp/tools.py:3406-3407` | Builds `provider_overrides` dict from `spec.pod` fields, only including keys where the value is non-`None`. For `interruptible`: `if spec.pod.interruptible is not None: provider_overrides["interruptible"] = spec.pod.interruptible`. Proper null-guard — omitted yaml fields let lower-precedence defaults through. |
| 3 | `mcp/tools.py:3416` | `fm.provision(count, provider_overrides=provider_overrides)` with a **fresh `FleetManager` per call** — no cross-call cache. |
| 4 | `fleet/manager.py:103-115` | `FleetManager._build_provider` reads `config.provider.interruptible` from `crucible.yaml` and passes it to the provider constructor as the default. |
| 5 | `fleet/providers/runpod.py:687` | `RunPodProvider.__init__` stores `self.interruptible = interruptible`. |
| 6 | `fleet/providers/runpod.py:718` | Per-call kwargs override the stored default: `eff_interruptible = bool(kwargs.pop("interruptible", self.interruptible))`. |
| 7 | `fleet/providers/runpod.py:753` | `create_api_pod(interruptible=eff_interruptible, ...)`. |
| 8 | `fleet/providers/runpod.py:485-548` | GraphQL vs REST payload builders. **GraphQL path does NOT pass `interruptible` as a flag**; it decides spot vs on-demand by whether `minMemoryInGb`/`minVcpuCount` are present (`if not interruptible:` at line 546). REST fallback does pass the flag explicitly at line 485. |

**Bottom line**: the yaml `pod.interruptible` value reaches the RunPod API correctly. The bug is in the bookkeeping that runs *after* pod creation — see §3.

---

## 3. Known bug: `nodes.json` interruptible echo

**Location**: `src/crucible/fleet/providers/runpod.py:625-627`

```python
"interruptible": bool(
    raw.get("interruptible", previous.get("interruptible", True)),
),
```

**What's broken**: RunPod's `podFindAndDeploy*` GraphQL response does NOT reliably include the `interruptible` field. So `raw.get("interruptible")` returns `None`, the fallback reads `previous.get("interruptible", True)`, and the ultimate default is `True`.

**Observable symptom**:
- You edit `pod.interruptible: true → false` in the yaml and reprovision.
- Pod IS created as on-demand/reserved at the API layer (verify with RunPod REST: `GET /v1/pods/<id>` → `costPerHr ≈ 0.59`, `podType: "RESERVED"`).
- But `nodes.json` records `interruptible: true` because of the fallback chain.
- Looks like your yaml edit was ignored. It wasn't — only the local bookkeeping is wrong.

**Diagnostic** — trust these instead of `nodes.json` for pod type:
- `costPerHr` from the RunPod REST API (`GET https://rest.runpod.io/v1/pods/<id>` with `Authorization: Bearer $RUNPOD_API_KEY`)
- Spot RTX 4090: ~$0.34/hr
- Reserved on-demand RTX 4090: ~$0.59–0.69/hr
- Secure cloud on-demand RTX 4090: ~$0.89+/hr

**Fix (one-line, future work)**: rewrite the fallback chain at `runpod.py:625-627` to prefer the **input** `eff_interruptible` value over the API echo. Requires threading the input through `inventory_record_from_api(raw, defaults, input_interruptible=...)` or stashing it on the `create_api_pod` result before the inventory record is built.

---

## 4. Running a variant

**Location**: `src/crucible/mcp/tools.py:3496-3514` (variant resolution in `run_project`) and `src/crucible/fleet/project_runner.py:112-203` (`chain_project_variants`).

Pass `variant_name` to `run_project` and the runner looks it up in `spec.variants`, merging that dict into overrides before launch (caller overrides still win). `chain_project_variants` runs an ordered list of variants on the same node.

```python
# Single variant
run_project(project_name="code_wm", variant_name="phase5_frozen_target", overrides={"WM_LR": "5e-4"})

# Sequence on one node
run_project_chain(project_name="code_wm", variants=["warmup", "phase5_frozen_target"])
```

Unknown variant names raise `FleetError` with the available set. Inlining the env dict directly in `overrides` still works for ad-hoc runs — see [§7](#7-correct-playbook-for-running-a-project-variant).

---

## 5. Bootstrap flow

**Function**: `bootstrap_project(project_name)` at `src/crucible/fleet/bootstrap.py:650`.

What gets synced to each pod during bootstrap:

1. **Repository** — `sync_repo` git-clones `spec.repo` at `spec.branch` to `{workspace}/project`. (default workspace: `/workspace/project`)
2. **Launcher bundle** — if `spec.launcher` is set, `resolve_launcher_bundle()` at `src/crucible/fleet/project_launchers.py:26` walks `local → hub → taps` and rsyncs the matching `launchers/<name>/` directory to `{workspace}/.crucible/launchers/<name>/`.
3. **Local files** — each entry in `spec.local_files` is `scp`'d to `{workspace}/{basename}` (path is flattened).
4. **`.env` file** — `fleet/sync.py:363` `write_remote_env` writes `{workspace}/.env` from:
   - `spec.env_forward` keys read from local `os.environ` / `.env.*` files
   - `spec.env_set` literal values
   - Denylist: `RUNPOD_API_KEY` and similar are rejected to prevent leaking to pods.
5. **System packages** — `spec.system_packages` installed via apt/apk/yum/dnf.
6. **Python venv** — created at `{workspace}/project/.venv` via `uv venv`.
7. **Torch** — `spec.install_torch` installed with `spec.install_flags` (index URL etc.) as a separate step so it doesn't get re-resolved by the general pip install.
8. **Pip packages** — `spec.install` list via `uv pip install`.
9. **Setup commands** — `spec.setup` array executed in sequence, with `.env` sourced beforehand.

**Bootstrap ends** before training starts. The `train:` command is NOT yet executed.

---

## 6. Env assembly at launch time

**Function**: `launch_project` at `src/crucible/fleet/project_runner.py:22`.

The actual SSH command that launches training is (paraphrased):

```bash
cd {workspace} && source .venv/bin/activate && \
if [ -f {workspace}/.env ]; then source {workspace}/.env; fi && \
export KEY1=VAL1 && export KEY2=VAL2 && ... && \
mkdir -p {log_dir} && \
python -c {launch_snippet}
```

**Precedence (later wins, naive bash semantics)**:

| Rank | Source | Applied | Notes |
|---|---|---|---|
| 1 (lowest) | System env on the SSH session | inherited | Varies by pod image |
| 2 | `{workspace}/.env` | `source .env` at `project_runner.py:54` | Written at bootstrap by `fleet/sync.py:363`. Contains `env_forward` keys + `env_set` values (merged in that order — env_set wins). |
| 3 (highest) | `override_exports` | `export KEY=VAL` after source | From `launch_overrides` dict at `mcp/tools.py:3583-3590` |

**How `launch_overrides` is built** at `mcp/tools.py:3529-3590`:

```python
# Step 1: contract_env extracted from crucible.yaml wandb block
contract_env = _project_contract_env(config, spec)
# Step 2: MUTATES the in-memory spec.env_set IN PLACE
overrides.update(contract_env)
spec.env_set.update(contract_env)

# Step 3: hardcoded per-run metadata added
launch_overrides = {
    **overrides,  # caller's overrides + contract_env
    "WANDB_RUN_NAME": wandb_run_name,
    "CRUCIBLE_REMOTE_NODE": node["name"],
    "CRUCIBLE_EXECUTION_PROVIDER": config.provider.type.lower(),
    "CRUCIBLE_ENFORCE_CONTRACT": "1",
    "CRUCIBLE_VARIANT_NAME": variant_name,
}
```

**Worked example** — for `code_wm` with a caller override of `{WM_LR: "5e-4", WANDB_RUN_NAME: "my-run"}`:

| Var | Final value | Source |
|---|---|---|
| `WANDB_API_KEY` | (from local `.env.runpod.local`) | `.env` (env_forward) |
| `WANDB_PROJECT` | `crucible-code-wm` | `.env` (env_set) + contract_env |
| `WANDB_ENTITY` | `eren23` | `.env` (env_set) + contract_env |
| `WANDB_MODE` | `online` | `.env` (env_set) + contract_env |
| `WM_LR` | `5e-4` | override_exports (caller) |
| `WANDB_RUN_NAME` | `my-run` | override_exports (caller) |
| `CRUCIBLE_REMOTE_NODE` | `code_wm-01` | override_exports (hardcoded metadata) |
| `WM_EMA_DECAY` | *unset* | Not in overrides, and no variant selected |

Values only land in the launch env if they appear in `env_forward`, `env_set`, `spec.variants[variant_name]` (when a variant is selected), or the caller's `overrides` dict.

---

## 7. Correct playbook for running a project variant

Either pass `variant_name` (if the variant is declared in the yaml) or inline the env dict via `overrides`. Here's the full canonical sequence for the CodeWM frozen-target ablation (2 seeds in parallel) using inline overrides:

```python
# 1. Provision (spot is fine — saves $)
mcp__crucible-fleet__provision_project(project_name="code_wm", count=2)

# 2. Wait 60-120s, then refresh until both pods show ssh_host
mcp__crucible-fleet__fleet_refresh()
# If ssh_host is still empty after 3 min, destroy + reprovision;
# US-NC-1 has an intermittent stuck-pod issue and the allocator
# will usually pick a different DC on retry.

# 3. Bootstrap BOTH pods
mcp__crucible-fleet__bootstrap_project(project_name="code_wm")

# 4. Launch seed 42 on pod 01 (full variant overrides inlined)
mcp__crucible-fleet__run_project(
    project_name="code_wm",
    node_names=["code_wm-01"],
    overrides={
        "WM_HDF5_PATH": "commitpack_python_trajectories_1.5m.h5",
        "WM_MODEL_DIM": "128",
        "WM_NUM_LOOPS": "6",
        "WM_NUM_HEADS": "4",
        "WM_ENCODER_LOOPS": "6",
        "ACTION_DIM": "7",
        "WM_LR": "1e-4",
        "WM_BATCH_SIZE": "128",
        "WM_STEPS": "15000",
        "WM_WINDOW_LEN": "3",
        "WM_EMA_DECAY": "1.0",
        "WM_SEED": "42",
        "WANDB_RUN_NAME": "phase5-frozen-target-15k-seed42",
    },
)

# 5. Launch seed 43 on pod 02 (same overrides, different seed + name)
mcp__crucible-fleet__run_project(
    project_name="code_wm",
    node_names=["code_wm-02"],
    overrides={
        # ... copy the dict from step 4 ...
        "WM_SEED": "43",
        "WANDB_RUN_NAME": "phase5-frozen-target-15k-seed43",
    },
)

# 6. Monitor lifecycle
mcp__crucible-fleet__get_project_run_status(run_id=...)

# 7. Collect results + metrics
mcp__crucible-fleet__collect_project_results(launch_id=...)

# 8. Destroy when done (do NOT forget — RunPod charges by the hour)
mcp__crucible-fleet__destroy_nodes(node_names=["code_wm-01", "code_wm-02"])
```

**DC-stuck mitigation**: if `fleet_refresh` shows `ssh_host: ""` after ~3 min, `destroy_nodes` and `provision_project` again. The US-NC-1 data center intermittently hits a "rented but no public IP allocation" failure mode. The allocator usually picks a different DC on retry.

**Cost sanity check**: while debugging, run `curl -s "https://rest.runpod.io/v1/pods/<pod_id>" -H "Authorization: Bearer $RUNPOD_API_KEY"` and read `costPerHr`. If you see `0.59–0.69` you're on on-demand (or at community rates); `0.89+` is secure cloud on-demand; `~0.34` is spot. Do NOT trust `nodes.json` — see [§3](#3-known-bug-nodesjson-interruptible-echo).

---

## 8. Full precedence table

Highest rank wins. This is the pod-provisioning + bootstrap config table; env-var conflicts at training time are described in [§6](#6-env-assembly-at-launch-time).

| Rank | Source | Path | Scope |
|---|---|---|---|
| 0 (baseline) | RunPodProvider hardcoded defaults | `fleet/providers/runpod.py:28-33` | All pods |
| 1 | Built-in preset defaults | `runner/presets.py:28-95` | Presets only |
| 2 | `crucible.yaml` → `provider` block | `<cwd>/crucible.yaml` | Per-project |
| 3 | `crucible.yaml` → `provider.defaults` dict | `<cwd>/crucible.yaml` | Per-project (SSH/workspace) |
| 4 | `crucible.yaml` → `presets.<name>` overlay | `<cwd>/crucible.yaml` | Per-project preset |
| 5 | `.env` → `.env.local` → `.env.runpod.local` | `<cwd>/...` | Per-project |
| 6 | `os.environ` at caller time | — | Per-machine |
| 7 | Project spec (tap) | `~/.crucible-hub/taps/<tap>/projects/<name>.yaml` | Per-spec |
| 8 | Project spec (hub, reserved) | `~/.crucible-hub/projects/<name>.yaml` | Per-spec |
| 9 | Project spec (local) | `<cwd>/.crucible/projects/<name>.yaml` | Per-spec |
| 10 | `spec.pod.*` fields (GPU, disk, interruptible) | within winning spec | Per-spec |
| 11 | MCP `provision_project` call kwargs | invocation | Per-call |
| 12 (highest) | MCP `run_project` `overrides` arg | invocation | Per-call (env only) |

### Important subtleties

- **Rank 7–9 are mutually exclusive**: `load_project_spec` returns the FIRST match. Local spec fully replaces tap spec; there is NO layering. You cannot "inherit from tap, override one field locally" — you have to copy the whole file.
- **Rank 10 is a subset of 7–9**: `spec.pod` fields are part of whichever spec wins at rank 7/8/9. `None` values in the spec pass through and let ranks 0–6 defaults apply.
- **Rank 11 can override rank 10** only for kwargs the caller explicitly passes in `provider_overrides`.
- **Rank 12 is run-time env only** (`run_project(overrides=...)`). It does NOT affect GPU type, container disk, interruptible, or any other pod-creation-time config.

---

## 9. Common gotchas (fast reference)

Ten real traps, in rough order of "likely to bite you soon":

1. **`nodes.json` lies about `interruptible`.** Trust RunPod REST API `costPerHr` instead. See [§3](#3-known-bug-nodesjson-interruptible-echo).
2. **Local spec silently shadows tap spec.** If both exist, edits to the tap never apply. Delete the local one or rename it.
3. **Variant dispatch**: pass `variant_name=...` to `run_project` to apply `spec.variants[name]` before caller overrides; otherwise inline the env dict in `overrides`. See [§4](#4-running-a-variant).
4. **`env_set` and `env_forward` with the same key**: `env_set` wins (written last by `write_remote_env`). Semantically confusing — avoid.
6. **`RUNPOD_API_KEY` in `env_forward`**: rejected with a loud ValueError at `fleet/sync.py:355`. The denylist is a feature, not a bug, but it confuses first-time users.
7. **Contract env (WANDB_*) mutates `spec.env_set` in place** at `mcp/tools.py:3531`. If you reuse a loaded spec across multiple `run_project` calls in the same process, subsequent calls see a mutated state. Always reload the spec.
8. **RunPod DC scheduling stall**: pods can sit in `uptimeInSeconds=0` with `publicIp=""` for 10+ minutes in some data centers. There is no automatic detection. Destroy + reprovision after 3 min if `fleet_refresh` shows empty `ssh_host`.
9. **Provider defaults only apply to new nodes**: `provider.defaults.env_source` is only read when a node record is FIRST created. If you change `env_source` in `crucible.yaml` but your `nodes.json` already has records, the old value sticks until you reprovision. Delete and reprovision.
10. **`WANDB_RUN_NAME` is hardcoded per call**: `run_project` always sets it (falling back to the auto-generated launch id). If you want a specific name, pass it explicitly in `overrides`.

---

## Cross-references

- **Top-level project guide**: `../CLAUDE.md`
- **Fleet provider code**: `../src/crucible/fleet/providers/runpod.py` (reference §2–3)
- **ProjectSpec dataclass**: `../src/crucible/core/config.py:402-426`
- **Project spec loader**: `../src/crucible/core/config.py:452` (reference §1)
- **Bootstrap flow**: `../src/crucible/fleet/bootstrap.py:650` (reference §5)
- **Launch flow**: `../src/crucible/fleet/project_runner.py:22` (reference §6)
- **MCP tool wrappers**: `../src/crucible/mcp/tools.py:3387-3590` (reference §2 and §6)
- **Test coverage**: `../tests/test_project_spec.py` (ProjectSpec loading, defaults)

## Bugs to fix (tracked for follow-up)

1. [Bug] `nodes.json` interruptible echo — `fleet/providers/runpod.py:625-627`. Prefer the input value over the API response echo. One-line fix.
2. [Feature] DC-stuck pod detection in `fleet_refresh` — timeout after N seconds of `uptimeInSeconds=0` and mark the pod as `failed_to_start` so agents auto-destroy instead of sitting.
3. [Cleanup] `provider.defaults` merges into new node records only. Document the "reprovision to pick up changes" gotcha or fix the merge to also apply to existing records on `fleet_refresh`.

---

*Last updated: 2026-04-11 (post three rounds of debugging RunPod US-NC-1 stuck pods + interruptible bookkeeping).*
