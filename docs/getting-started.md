---
layout: default
title: Getting Started
---

# Getting Started

This guide walks through setting up Crucible and running your first experiment from scratch. It covers local dev, cloud fleet runs, and the autonomous researcher.

**Crucible is modality-agnostic.** The examples below use a diffusion model, but the same flow works for LM, vision, world models, or a bring-your-own training script.

## 1. Install

```bash
pip install -e ".[all]"           # from source
# or (once published to PyPI):
pip install crucible-ml[all]
```

Install specific extras only:

```bash
pip install crucible-ml              # core (orchestration, CLI, config, store)
pip install crucible-ml[tui]         # + Interactive TUI
pip install crucible-ml[torch]       # + PyTorch model zoo
pip install crucible-ml[mcp]         # + MCP server for Claude
pip install crucible-ml[anthropic]   # + Claude-powered autonomous researcher
pip install crucible-ml[wandb]       # + W&B integration
```

## 2. Initialize a project directory

```bash
mkdir my-research && cd my-research
crucible init
```

This creates:
- `crucible.yaml` — project-wide config (provider, presets, researcher budget)
- `.crucible/` — local store for designs, context, notes, and plugins

Edit `crucible.yaml` to set your compute provider (RunPod or SSH), metrics to rank by, and experiment presets.

Alternative: copy `crucible.yaml.example` from the Crucible repo instead of running `init` — it has more comments and a fully worked-out `fleet.ssh` config.

## 3. Pick a project template

Crucible ships five built-in templates. Run:

```bash
crucible project templates
```

```
Available project templates (5):
  diffusion
    Denoising diffusion (DDPM-style) on image data. Modeled on the examples/diffusion/ walkthrough.
    required vars: PROJECT_NAME, REPO_URL
  generic
    Minimal project-spec skeleton — bring your own everything. Start here if no other template fits.
    required vars: PROJECT_NAME, REPO_URL
  lm
    Language model training. Defaults favor fineweb-style token-level training.
    required vars: PROJECT_NAME, REPO_URL
  vision
    Image classification (torchvision ImageFolder or CIFAR-10).
    required vars: PROJECT_NAME, REPO_URL
  world_model
    JEPA-style world model for video / trajectory prediction.
    required vars: PROJECT_NAME, REPO_URL
```

Create a spec from your chosen template:

```bash
crucible project new my-first-project --template diffusion \
    --set REPO_URL=https://github.com/me/my-first-project
```

This writes `.crucible/projects/my-first-project.yaml`. Open it, check the values, and customize as needed (install deps, training command, env vars).

If you omit required vars on the command line, `crucible project new` prompts for them interactively. For CI use `--no-prompt` to fail fast instead.

## 4. Run it — local first

For a first sanity check, run the built-in diffusion example directly:

```bash
cd /path/to/crucible/examples/diffusion
python train_generic.py
```

This trains a small DDPM UNet on synthetic images, verifying your install works. Swap env vars to change the preset (`PRESET=screen`) or the model size (`MODEL_DIM=64`).

For your own project:

```bash
crucible run experiment --preset smoke --name my-test
```

## 5. Run it — on a cloud fleet

With a `crucible.yaml` provider configured (RunPod is the most-tested backend) and a project spec in `.crucible/projects/`:

```bash
# 1. Provision GPUs (transactional — partial failures don't orphan pods)
crucible fleet provision --count 2

# 2. Wait for SSH to come up, then sync code + install deps
crucible fleet bootstrap

# 3. Check per-node, per-step bootstrap progress
crucible fleet status

# 4. Enqueue a batch of experiments and dispatch them to idle nodes
crucible run enqueue --spec experiments.json
crucible run dispatch

# 5. Wait for runs to finish, collect results, and rank them
crucible fleet monitor --watch 60
crucible run collect
crucible analyze rank --top 10

# 6. Tear down when done
crucible fleet destroy
```

If provisioning or bootstrap has hiccups (it happens on rental GPUs), check:

```bash
crucible fleet status                       # per-step bootstrap failures are visible here
```

And if you suspect leaked pods on the provider side:

```bash
# via MCP: cleanup_orphans(destroy=false)    # list first
# via MCP: cleanup_orphans(destroy=true)     # then destroy
```

## 6. Go autonomous (optional)

Once you have a few experiments on the leaderboard, you can hand the loop to Claude:

```bash
# Dry run first — prints what it would do without running anything
crucible research start --budget-hours 10 --tier proxy --dry-run

# Real run
crucible research start --budget-hours 10 --tier proxy
```

The researcher analyzes existing results, generates hypotheses, designs batches, dispatches them, and reflects on the outcomes — all within your compute budget.

## What's next

- **Examples** — see [`examples/`](../examples/) for working projects across modalities
- **Modality guide** — [`docs/modality-guide.md`](modality-guide.md) shows how to plug in a new data adapter, objective, or model type
- **Plugin system** — [`docs/plugins.md`](plugins.md) walks through adding optimizers, callbacks, architectures
- **MCP tools** — [`docs/mcp-tools.md`](mcp-tools.md) lists every tool exposed to Claude
- **Architecture** — [`docs/architecture.md`](architecture.md) for the module boundary story

## Troubleshooting

**`crucible fleet bootstrap` hangs for a long time on a fresh pod.**
Initial SSH readiness uses exponential backoff (6 attempts, up to 180s total by default). If a pod is genuinely dead, `wait_for_ssh_ready` raises `SshNotReadyError` with the classified reason. Tune `fleet.ssh.initial_connect.max_wait` in `crucible.yaml` if your provider is slower than RunPod.

**A node is marked `bootstrap_failed` — which step broke?**
Run `crucible fleet status` and look at `bootstrap_steps` on the failing node. Each step records `ok` / `failed` / `running` with a timestamp and the error message.

**Provision succeeded partially and left pods I can't see.**
Use the `cleanup_orphans` MCP tool (or a follow-up `crucible fleet refresh` — reconciled orphans now get an explicit `reconciled_orphan` state so they show up in `fleet status`).

**My project needs a dataset that isn't fineweb.**
Set `data.probe.paths` and `data.probe.download_command` in `crucible.yaml`. Bootstrap will check the listed paths on the remote and run your download command if any are missing. For projects that download data at training time, set `data.probe.paths: []` and bootstrap will skip the data step entirely.
