---
layout: default
title: SSH Provider
---

# SSH Provider

Crucible's **SSH provider** runs the full fleet workflow against one or more machines you already own or rent. No RunPod account required.

Use this when:
- You have a home server or workstation with a GPU
- You want to test Crucible against localhost before committing to cloud spend
- You're on a non-RunPod cloud (Lambda, Paperspace, EC2) and just want to SSH in
- You're running against an always-on cluster where pods shouldn't be created and destroyed

## What the SSH provider can and can't do

| Operation | RunPod | SSH |
|---|---|---|
| `provision` | Creates pods via API | No-op, warns |
| `destroy` | Terminates pods via API | Removes from inventory only |
| `refresh` | Queries API state | SSH ping to each node |
| `wait_ready` | Waits for API + SSH | Exponential SSH backoff (same as RunPod) |
| `bootstrap` | ✓ Full step tracking | ✓ Full step tracking |
| `dispatch` | ✓ | ✓ |
| `collect` | ✓ | ✓ |
| `cleanup_orphans` | ✓ Lists + destroys | ✗ Raises FleetError (no central API to query) |

Everything non-provider-specific works identically — the output parser, leaderboard, autonomous researcher, MCP tools, and tree search don't know or care which provider you're using.

## Minimal setup

### 1. Configure `crucible.yaml`

```yaml
name: my-ssh-project

provider:
  type: ssh
  ssh_key: ~/.ssh/id_ed25519
  defaults:
    workspace_path: /home/user/crucible-workspace
    python_bin: python3

# Tune the initial SSH connect for always-on machines — shorter
# budget than the RunPod default because there's no cold-boot phase.
fleet:
  ssh:
    initial_connect:
      max_attempts: 3
      backoff_base: 2
      max_wait: 30
    step_timeouts:
      default: 300
      pip_install: 1200

execution_policy:
  require_remote: false
  allow_local_dev: true
```

### 2. Write `nodes.json` by hand

The SSH provider never creates `nodes.json` — you own it. Minimum shape:

```json
[
  {
    "name": "gpu-01",
    "node_id": "gpu-01",
    "ssh_host": "gpu01.home.lan",
    "ssh_port": 22,
    "user": "user",
    "ssh_key": "~/.ssh/id_ed25519",
    "workspace_path": "/home/user/crucible-workspace",
    "python_bin": "python3",
    "state": "new",
    "env_ready": false,
    "dataset_ready": false
  }
]
```

Add one entry per machine. State fields (`env_ready`, `dataset_ready`, `state`) get updated by bootstrap and refresh; just leave them at their initial values.

### 3. Verify SSH works before bootstrapping

```bash
ssh -i ~/.ssh/id_ed25519 user@gpu01.home.lan "echo ready && python3 --version && nvidia-smi --query-gpu=name --format=csv,noheader"
```

If that prints successfully, you're ready. If not, fix the SSH config first — `crucible fleet bootstrap` will error out with a classified reason (`SshAuthError` / `SshNotReadyError` / `SshTimeoutError`) but it won't help you debug keys or networking.

### 4. Bootstrap and run

```bash
# No provision step — nodes.json is already populated
crucible fleet bootstrap

# Per-step bootstrap state is visible here — helpful for debugging
crucible fleet status

# Run experiments
crucible run experiment --preset smoke --name my-first
# or a batch
crucible run enqueue --spec experiments.json
crucible run dispatch
crucible run collect
crucible analyze rank --top 10
```

## Differences from RunPod

### No `provision` / `destroy`

`crucible fleet provision` and `crucible fleet destroy` are warnings-only for SSH. The SSH provider logs a warning and returns without touching anything. To add a machine, edit `nodes.json`; to remove one, edit `nodes.json` again.

### `cleanup_orphans` raises `FleetError`

Orphan detection depends on a central provider API (RunPod's list-all-pods endpoint). SSH has no such API — each machine is its own universe — so the operation is undefined. Calling `cleanup_orphans` against an SSH provider raises `FleetError` with a clear message. This is by design: it's better to fail loudly than to silently report "no orphans found" when we simply can't tell.

### Initial connect budget defaults to shorter

The SSH provider defaults to the same backoff schedule as RunPod (6 attempts, 180s total), but if your machines are always-on you can safely lower this. `examples/ssh_local/crucible.yaml` sets it to 3 attempts / 30s for a fast-fail on config mistakes.

### No cost tracking

RunPod's provider can show live pricing; SSH has no such concept. `crucible fleet status` still works but won't include dollar-per-hour data.

## Example

See [`examples/ssh_local/`](../examples/ssh_local/) for a complete working example:

- `crucible.yaml` with SSH provider configuration
- `nodes.json.example` template (copy to `nodes.json`)
- `experiments.json` sample batch
- README covering setup, troubleshooting, and the full run loop

## Troubleshooting

**`SshAuthError` during bootstrap.**
Your key is wrong or the user doesn't have authorized_keys set up. Run `ssh -v -i <key> <user>@<host>` manually, fix the issue, then retry.

**`SshNotReadyError` during bootstrap.**
The host is reachable but sshd isn't accepting connections yet. On a home server, check `systemctl status sshd`. If the host is an always-on machine and this happens repeatedly, bump `fleet.ssh.initial_connect.max_wait` — a networking flake might need more budget.

**`SshTimeoutError` during bootstrap.**
The connection hangs mid-handshake. Usually indicates firewall issues, an overloaded node, or MTU mismatch. Test with `ssh -vvv -o ConnectTimeout=5 <host>`.

**Bootstrap reaches `torch_import` and fails.**
PyTorch isn't installed on the remote. Either install it manually, or add a `pip_install` line to your project spec that pulls it for you. The default Crucible bootstrap assumes torch is pre-installed (for RunPod-style images that ship with it).

**`crucible fleet status` says my nodes are `unreachable`.**
Either the SSH config broke (network change, key rotated, sshd down) or the node is genuinely down. `crucible fleet refresh` re-pings every node and updates the state.

**I ran `cleanup_orphans` and got `FleetError: does not support`.**
That's expected for SSH. Use it only with RunPod-style providers that have a central listing API.
