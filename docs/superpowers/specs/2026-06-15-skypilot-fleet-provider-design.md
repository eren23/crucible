# SkyPilot fleet provider

**Date:** 2026-06-15
**Status:** Approved design, pending implementation plan
**Branch:** `feat/skypilot-provider`

## Context

Crucible's fleet layer has two providers today — `runpod` and `ssh` (both in
`src/crucible/fleet/providers/`). The June-2026 positioning work identified **more fleet
providers** as the #1 unfilled gap in the OSS auto-research ecosystem and a roadmap "double
down" commitment. The user runs GCP at work and wants Crucible usable there.

Rather than a bespoke GCP provider, we bridge to **SkyPilot** — the "integrate, don't reinvent
multi-cloud" move already endorsed in `docs/positioning.md`. SkyPilot gives GCP (plus ~20 other
clouds + spot failover) through one provider. This is phase two of the agreed plan ("proof-first
evaluator work, then SkyPilot backend").

Two findings from exploration shape the design:

1. **The provider contract is small.** `FleetProvider` (`src/crucible/fleet/provider.py`) requires
   four methods — `provision`, `destroy`, `refresh`, `wait_ready` — plus optional `stop`/`start`,
   all returning `list[NodeRecord]`. A provider's *entire* job is VM lifecycle + populating the
   SSH fields (`ssh_host`, `ssh_port`, `user`, `ssh_key`) on each NodeRecord. `bootstrap`,
   `dispatch`, and `collect` are already provider-agnostic over SSH.
2. **SkyPilot's Python SDK is the wrong surface.** It is async (`RequestId` + `sky.get`),
   version-churny, and has *no clean call to extract an SSH endpoint*. The reliable, stable
   surface is the **`sky` CLI**: `sky launch`, `sky status --ip <cluster>` (head IP), `sky down`,
   plus the `Host <cluster>` block SkyPilot writes to `~/.ssh/config` (User + IdentityFile).

## Goals

- A `skypilot` provider, selectable via `provider.type: skypilot`, that provisions GPU VMs on GCP
  (spot + idle-autostop), hands Crucible SSH endpoints, and tears them down.
- Adds **zero required Python dependency** to Crucible core: the provider shells out to the `sky`
  CLI (like the codebase already shells to `ssh`/`rsync`); `sky` is an external binary the user
  installs.
- Unit-tested with all `sky`/subprocess interaction mocked (no cloud in CI).

## Non-goals (explicit follow-ups)

- `stop`/`start` (inherit the base no-op) and orphan cleanup / cluster listing.
- SkyPilot **managed jobs** (spot auto-recovery) — clusters only.
- Multi-node clusters.
- Non-GCP clouds — the provider is config-agnostic, but only GCP is dogfooded here.
- The `PluginRegistry` source-propagation bug (separate decision, prior session).

## Approach

### Placement (core builtin, CLI-wrapping)

- New file `src/crucible/fleet/providers/skypilot.py` → `class SkyPilotProvider(FleetProvider)`.
- Register in `src/crucible/fleet/provider_registry.py` next to runpod/ssh:
  `register_provider("skypilot", _skypilot_factory)` with a factory mirroring `_runpod_factory`'s
  signature (`ssh_key`, `gpu_type_ids`, `gpu_count`, `interruptible`, `defaults`, `project_name`,
  `**kwargs`).
- Fleet providers belong in core (unlike model architectures); this is consistent with runpod/ssh.
- `pyproject.toml`: add an **optional** extra `skypilot = ["skypilot[gcp]>=0.12"]` for install
  convenience (`pip install crucible[skypilot]`). The provider never imports `sky`.

### Config (no core schema change)

SkyPilot specifics ride in the free-form `provider.defaults` dict read by the factory:

```yaml
provider:
  type: skypilot
  interruptible: true            # -> --use-spot
  gpu_types: [L4]                # -> accelerators "L4:<gpu_count>" if `accelerators` absent
  gpu_count: 1
  defaults:
    cloud: gcp
    region: us-central1          # optional
    accelerators: L4:1           # optional explicit override
    disk_size: 100               # GB, optional
    idle_minutes_to_autostop: 30 # cost control
    cpus: "4+"                   # optional
    memory: "16+"                # optional
    workspace_path: /workspace/project
    python_bin: python3
    env_source: .env.local
```

### The four methods

- **`provision(count, name_prefix, start_index, replacement, **kwargs)`** — for each index, build a
  cluster name and run `sky launch`. Then extract the endpoint and emit a NodeRecord:
  - cluster name: `crucible-<project>-<name_prefix>-<idx>`, sanitized to SkyPilot's DNS-like rules
    (lowercase alphanumeric + hyphens; no `__`). Project encoded for isolation.
  - launch command (exact flags confirmed by the Task-1 spike): `sky launch -c <cluster> -y`
    `--cloud <cloud> --gpus <accel> [--use-spot] [--region <r>] [--idle-minutes-to-autostop <n>]`
    `[--disk-size <gb>]` plus a trivial entrypoint so the cluster comes up.
  - endpoint: `sky status --ip <cluster>` → head IP; parse `~/.ssh/config` `Host <cluster>` block →
    `User`, `IdentityFile`.
  - NodeRecord fields: `ssh_host=<ip>`, `ssh_port=22`, `user=<parsed>`, `ssh_key=<parsed identity>`
    (SkyPilot's generated key, **not** `provider.ssh_key`), `node_id=<cluster>`,
    `provider="skypilot"`, `state="new"`, plus `workspace_path`/`python_bin`/`env_source` from
    defaults.
  - partial failure: commit already-created clusters to the return list before re-raising
    (mirrors runpod's `PartialProvisionError` handling).
- **`refresh(nodes)`** — per node, `sky status --ip` / `sky status <cluster>` → update
  `ssh_host`/`state`; a vanished cluster → `state="lost"`.
- **`wait_ready(nodes, timeout_seconds, poll_seconds, stalled_seconds)`** — poll `refresh()` until
  `ssh_host` is present, then `ssh_ok(node)`, reusing `fleet.sync` helpers (same as ssh.py).
- **`destroy(nodes, selected_names)`** — `sky down -y <cluster>` per (selected) node; return
  survivors; best-effort on already-gone.
- **`stop`/`start`** — inherit the base no-op (deferred).

### Testability

All `sky`/subprocess/ssh-config interaction sits behind small isolated functions — e.g.
`_build_launch_cmd(...)`, `_sky_status_ip(cluster)`, `_sky_down(cluster)`,
`_parse_ssh_config(cluster, config_text)` — so unit tests monkeypatch them with no cloud.

### Error handling

- Preflight `shutil.which("sky")` → `FleetError` with an actionable install message (mirrors the
  lm-eval evaluator's `validate`).
- Non-zero `sky` exit → `FleetError` carrying the stderr tail.
- Docs note: clusters do not auto-recover from spot preemption (that's destroy + re-provision, or
  the deferred managed-jobs path).

## Data flow

`crucible.yaml provider.type=skypilot` → `FleetManager._build_provider` → `build_provider("skypilot", …)`
→ `SkyPilotProvider`. Then `provision_nodes` → `provision()` (sky launch + endpoint parse) →
NodeRecord persisted to `nodes.json` → `bootstrap_nodes`/`dispatch_experiments`/`collect_results`
(SSH, provider-agnostic) → `destroy_nodes` → `destroy()` (sky down).

## Files

- Create: `src/crucible/fleet/providers/skypilot.py` (the provider + isolated `sky` helpers).
- Modify: `src/crucible/fleet/provider_registry.py` (register `skypilot` + factory).
- Modify: `pyproject.toml` (optional `skypilot` extra).
- Create: `tests/test_skypilot_provider.py` (mocked-CLI unit tests).
- Create/extend: `docs/skypilot-provider.md` (GCP auth + `sky check` + config keys + spot/autostop
  caveats).

## Prerequisites

- `sky` CLI on PATH (`pip install skypilot[gcp]`), GCP credentials (service account /
  `GOOGLE_APPLICATION_CREDENTIALS` or `gcloud auth application-default login`), `sky check` green,
  and GCP GPU quota in the target region.

## Risk, front-loaded

The exact `sky status --ip` output and ssh-config `User`/`IdentityFile` details are
version-dependent (SkyPilot ~0.12, 2026). The implementation plan's **Task 1 is a manual spike on
the user's GCP** — `sky check` → `sky launch` a tiny L4 spot node → confirm precise IP/user/key
extraction → `ssh_ok` → `sky down` — *before* writing the typed provider. The parsing then targets
confirmed formats and lives behind the mockable helpers, so it can change without touching
lifecycle logic.

## Verification

1. `PYTHONPATH=src .venv/bin/python -m pytest tests/test_skypilot_provider.py -v` → all pass
   (mocked CLI, no cloud).
2. `build_provider("skypilot", …)` returns a `SkyPilotProvider`; `provider.type: skypilot` resolves
   end-to-end through `FleetManager`.
3. Manual GCP dogfood (environment-gated): `sky check` → `provision_nodes` → `bootstrap_nodes` →
   one tiny experiment → `collect_results` → `destroy_nodes`, confirming a real run on the user's
   GCP and clean teardown.
