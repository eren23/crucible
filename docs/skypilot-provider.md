# SkyPilot Fleet Provider

The `skypilot` provider runs Crucible's fleet on any cloud SkyPilot supports —
GCP first — by wrapping the [`sky` CLI](https://docs.skypilot.co). It's the
"bridge, don't reinvent multi-cloud" path: rather than a bespoke GCP integration,
Crucible drives `sky launch` / `sky status` / `sky down` and hands the resulting
SSH endpoints to the same provider-agnostic bootstrap → dispatch → collect
machinery that RunPod and SSH already use.

The provider **shells out** to `sky` — it never imports the SkyPilot Python SDK,
so Crucible core gains no dependency. `sky` is an external binary you install,
exactly like `ssh` and `rsync`.

## Setup

### 1. Install SkyPilot with GCP support

```bash
pip install "skypilot[gcp]"        # or: pip install "crucible-ml[skypilot]"
```

### 2. Authenticate to GCP and validate

```bash
# Service account (recommended for long-lived use)
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
gcloud auth activate-service-account --key-file="$GOOGLE_APPLICATION_CREDENTIALS"
gcloud config set project YOUR_PROJECT_ID

# …or user creds
gcloud auth application-default login

# Validate — must show "GCP: enabled"
sky check
```

You also need **GPU quota** in your target region (e.g. `NVIDIA_L4_GPUS`); request
an increase in the GCP console if `sky launch` fails with a quota error.

If `sky` is missing from `$PATH`, the provider raises a `FleetError` with this
install hint before any cloud call.

## Configure `crucible.yaml`

```yaml
provider:
  type: skypilot
  interruptible: true            # -> --use-spot (preemptible, ~70-90% cheaper)
  gpu_types: [L4]                # first entry + gpu_count -> accelerators "L4:1"
  gpu_count: 1
  defaults:
    cloud: gcp                   # default if omitted
    region: us-central1          # optional; SkyPilot picks the cheapest zone
    accelerators: L4:1           # optional explicit override of gpu_types/gpu_count
    disk_size: 100               # GB, optional
    idle_minutes_to_autostop: 30 # cost control — stop idle clusters
    cpus: "4+"                   # optional minimum
    memory: "16+"                # optional minimum
    workspace_path: /workspace/project
    python_bin: python3
    env_source: .env.local
```

All SkyPilot-specific settings live in the free-form `provider.defaults` dict, so
no Crucible config schema changes are needed. `interruptible`, `gpu_types`, and
`gpu_count` are the standard top-level provider fields (shared with RunPod).

## How it works

| Step | What the provider does |
|------|------------------------|
| `provision` | `sky launch -c <cluster> -y --cloud gcp --gpus <accel> [--use-spot] …` per node, then reads the head IP from `sky status --ip` and the `User`/`IdentityFile` from the `Host <cluster>` block SkyPilot writes to `~/.ssh/config`. |
| `refresh` | `sky status --ip <cluster>` → updates `ssh_host`; probes SSH. A vanished cluster is marked `lost`. |
| `wait_ready` | Polls SSH with exponential backoff (shared with the SSH provider). |
| `destroy` | `sky down -y <cluster>` per selected node. |

The provider uses **SkyPilot's own generated SSH key and user** (parsed from
`~/.ssh/config`), not `provider.ssh_key`.

### Cluster naming & project isolation

SkyPilot has no resource tags, so the project is encoded in the cluster name:

```
crucible-<project>-<name_prefix>-<index>
```

sanitized to SkyPilot's DNS-like rules (lowercase, hyphens). This keeps one
project's clusters distinguishable from another's.

## Cost control

- **Always set `idle_minutes_to_autostop`.** Idle clusters keep billing; autostop
  stops them after N idle minutes.
- **Spot is cheap but volatile.** `interruptible: true` uses GCP preemptible VMs
  (large savings), but a preempted *cluster* does **not** auto-recover — checkpoint
  your training and re-provision. (Auto-recovery requires SkyPilot managed jobs,
  which this provider does not use yet.)
- Cold start is ~5–15 minutes (provision + image + setup) before the first step.

## Limitations (current scope)

- **No `stop`/`start`.** Use `destroy` + re-provision, or `idle_minutes_to_autostop`.
- **No orphan cleanup.** `cleanup_orphans` is not implemented for this provider; use
  `sky status` / `sky down` directly to reconcile.
- **No spot auto-recovery / managed jobs.** Clusters only.
- **Single-node clusters.**
- **GCP is the dogfooded cloud.** Other SkyPilot clouds are config-reachable
  (`defaults.cloud`) but untested here.
- **Provision is fail-fast.** A failed `sky launch` raises immediately; any clusters
  already created in the same call are logged for manual `sky down`.

See `docs/ssh-provider.md` for the provider-agnostic bootstrap/dispatch/collect
flow, which is identical across providers.
