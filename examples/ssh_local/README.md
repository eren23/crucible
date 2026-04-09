# SSH Local Example

Run Crucible against a **single SSH-reachable machine** instead of RunPod. Good for:

- Testing Crucible without a cloud account
- Running on a home server with a consumer GPU
- Running on a non-RunPod cloud VM (Lambda, Paperspace, your own EC2)
- Debugging the fleet bootstrap flow locally (point SSH at localhost)

## What this demonstrates

- **SSH provider** as a first-class peer to RunPod — no provisioning, just reuse what you already have.
- `crucible.yaml` with `provider.type: ssh` and a handwritten `nodes.json`.
- The same fleet loop (`bootstrap → dispatch → collect → analyze`) works with zero RunPod dependencies.

## Prerequisites

- A remote machine (or localhost) reachable via SSH with public-key auth.
- Python 3.11+ on the remote.
- PyTorch installed on the remote (or skip `bootstrap` and run `dispatch` directly against an existing env).

Verify SSH works from your dev box to the target before starting:

```bash
ssh -i ~/.ssh/id_ed25519 user@host.example.com "echo ready && python3 --version"
```

If that prints `ready` and a Python version, you're set.

## Setup

```bash
cd examples/ssh_local

# Copy the template and edit it with your machine's details
# (nodes.json itself is in .gitignore because it holds per-machine state)
cp nodes.json.example nodes.json
nano nodes.json
```

Then run the standard Crucible flow — notice there's no `crucible fleet provision` call because the SSH provider doesn't create machines. You register existing ones instead.

```bash
# Sync code to the node, verify Python + torch, run the data probe
crucible fleet bootstrap

# Check bootstrap status per step
crucible fleet status

# Run a smoke experiment directly
crucible run experiment --preset smoke --name ssh-smoke
```

## Running a batch

```bash
# Enqueue a batch spec
crucible run enqueue --spec experiments.json

# Dispatch queued runs to the SSH node (same scheduler as RunPod)
crucible run dispatch

# Collect results when done
crucible run collect
crucible analyze rank --top 5
```

## Files

| File | Purpose |
|---|---|
| `crucible.yaml` | Project config with `provider.type: ssh` and a baseline preset |
| `nodes.json.example` | Template inventory — copy to `nodes.json` and edit |
| `experiments.json` | Sample 3-experiment batch spec |
| `README.md` | This file |

## Notes on the SSH provider

- **No provisioning**: `crucible fleet provision` is a no-op; you manage nodes by editing `nodes.json` directly.
- **No destroy**: same reason — `crucible fleet destroy` just removes nodes from inventory, it does not power them off.
- **Orphan detection doesn't apply**: there's no provider-side API to query, so `cleanup_orphans` raises a clear `FleetError` for SSH providers.
- **Everything else works the same**: bootstrap, dispatch, collect, status, and the autonomous researcher are all provider-agnostic.

## Troubleshooting

**`wait_for_ssh_ready` raises `SshAuthError`.**
Your SSH key or user is wrong. Run `ssh -v -i <key> <user>@<host>` manually and fix the auth before retrying.

**`wait_for_ssh_ready` raises `SshNotReadyError` (connection refused).**
The host is reachable but sshd isn't running. Check `systemctl status sshd` on the target.

**Bootstrap fails on `torch_import`.**
The remote Python doesn't have torch installed. Either install it manually on the node, or add a `pip_install` step to your project spec that pulls your deps.
