# Crucible

> **Alpha software.** Crucible works for the author's use case (autonomous ML research on RunPod). It may work for yours. APIs will change. Bug reports and PRs welcome.

Autonomous ML research on rental GPUs. LLM-driven hypothesis generation + fleet orchestration on RunPod/SSH.

You bring a training script. Crucible decides what experiments to run, provisions the compute, executes them across tiers, and learns from the results.

## Why Crucible?

No single existing tool combines fleet orchestration on rental GPUs with autonomous experiment design. The closest alternatives:

- **SkyPilot** provisions GPUs across 20+ clouds but doesn't decide what experiments to run
- **Optuna/Ax** optimize hyperparameters mathematically but don't provision compute or reason about architectures
- **AI Scientist** generates hypotheses but runs single-machine with a 42% failure rate and no fleet management
- **W&B/MLflow** track experiments but don't execute them autonomously

Crucible connects these concerns into one loop: **analyze → hypothesize → provision → execute → reflect → promote or kill**.

## Origins

Born from [OpenAI Parameter Golf](https://github.com/openai/parameter-golf) (March–April 2026), a competition to train the best 16MB language model on 8xH100s in 10 minutes. The autonomous research infrastructure we built for the competition turned out to be general-purpose. Crucible extracts and generalizes it.

## What Works Today

- Fleet orchestration on RunPod (provision, bootstrap, dispatch, collect, destroy)
- Generic SSH provider for any machine you can SSH into
- Experiment execution with live output parsing, OOM retry, tier presets
- Claude-driven autonomous research loop (hypothesis → batch → execute → reflect)
- MCP server so Claude can control experiments via tool use
- Model zoo with transformer components (RMSNorm, RoPE, GQA, SmearGate, etc.)
- Analysis: leaderboard, sensitivity analysis, Pareto frontier
- YAML project configuration (`crucible.yaml`)
- Experiment notes (attach freeform observations to runs with YAML frontmatter)
- Research tracks (group projects by research direction in the Crucible Hub)
- Crucible Hub (`~/.crucible-hub/`) for cross-project knowledge sharing, git-synced
- Research briefing (LLM session orientation with project context and findings)
- REST API server (`crucible serve`) — 10 FastAPI endpoints wrapping MCP tools
- W&B bridge with image logging and run annotation support
- 77 MCP tools for AI agent integration (fleet, design, context, notes, hub, tracks, briefing, architecture composition, tree search, training generalization)
- Interactive TUI for browsing experiment designs grouped by status

## What's Coming

- SkyPilot provider (20+ cloud support)
- Optuna/Ax integration (mathematical HPO alongside LLM-driven search)
- Code-level search (LLM modifies training scripts, not just configs)
- Research strategy plugins (custom loop phases)
- PyPI release

## Quick Start

```bash
# Install from source
pip install -e ".[all]"

# Initialize a project
crucible init

# Edit crucible.yaml — point at your training script

# Run a smoke test
crucible run experiment --preset smoke

# Or go autonomous
crucible research start --budget-hours 10 --tier proxy
```

## Core Concepts

### crucible.yaml

Like docker-compose for ML experiments:

```yaml
name: my-project
provider:
  type: runpod
  gpu_types: ["NVIDIA GeForce RTX 4090"]
training:
  - backend: torch
    script: train.py
presets:
  smoke: { MAX_WALLCLOCK_SECONDS: "60", ITERATIONS: "400" }
  proxy: { MAX_WALLCLOCK_SECONDS: "1800", ITERATIONS: "6000" }
researcher:
  model: claude-sonnet-4-6-20250514
  budget_hours: 10.0
```

### Training Contract

Crucible doesn't own your training code. Any script that reads env vars and prints parseable output works:

**Input** (env vars):
- `ITERATIONS`, `MAX_WALLCLOCK_SECONDS`, `TRAIN_BATCH_TOKENS`
- `MODEL_FAMILY`, `NUM_LAYERS`, `MODEL_DIM`, etc.
- `RUN_ID`, `RUN_BACKEND`, `RUN_PRESET`

**Output** (stdout patterns):
- `step:{step}/{total} train_loss:{loss}`
- `step:{step}/{total} val_loss:{loss} val_bpb:{bpb}`
- `Serialized model ... {N} bytes`

### Experiment Tiers

Experiments earn their way to expensive compute:

| Tier | Duration | Use Case |
|------|----------|----------|
| smoke | ~1 min | Quick validation |
| proxy | ~30 min | Main exploration |
| medium | ~1 hr | Extended runs |
| promotion | ~2 hrs | Best candidates |

### Fleet Management

```bash
crucible fleet provision --count 4
crucible fleet bootstrap
crucible fleet status
crucible fleet destroy
```

### Autonomous Researcher

Claude-powered: analyze results, generate hypotheses, design batches, execute, reflect, promote or kill.

```bash
crucible research start --budget-hours 10 --tier proxy --dry-run
```

### MCP Integration

```bash
crucible mcp serve  # starts stdio MCP server for Claude (77 tools)
crucible serve      # starts REST API server (FastAPI, 10 endpoints)
```

## CLI Reference

```
crucible init
crucible fleet {status|provision|destroy|bootstrap|sync|monitor}
crucible run {experiment|queue|enqueue|dispatch|collect|day|night}
crucible analyze {rank|sensitivity|pareto|export|summary}
crucible research {start|status}
crucible data {download|sync|status}
crucible mcp serve
crucible models list
crucible hub {status|sync|findings}
crucible track {create|list|switch}
crucible note {add|get|search}
crucible serve [--port PORT]
crucible tui
crucible store {list|diff|get}
```

## Installation

```bash
pip install crucible-ml[all]        # everything
pip install crucible-ml             # minimal (orchestration only)
pip install crucible-ml[torch]      # model zoo
pip install crucible-ml[anthropic]  # autonomous researcher
pip install crucible-ml[mcp]        # MCP server
```

## Validated Workflow (Tested 2026-03-21)

This exact sequence was run and confirmed working on 2 RunPod pods:

```bash
cd /path/to/your-ml-project
crucible fleet provision --count 2 --name-prefix crucible-test
crucible fleet bootstrap --train-shards 1
crucible run enqueue --spec experiments.json --limit 3
crucible run dispatch
crucible fleet monitor --watch 60
crucible run collect
crucible analyze rank --top 10
crucible fleet destroy
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Highest-impact areas:
- **Compute providers**: Modal, Lambda, SkyPilot backends
- **Search strategies**: Optuna, Ax integration
- **Training script examples**: Show Crucible working with your framework
- **Bug reports**: File issues, we'll fix them

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full plan — what works, what's next, what we won't build, and honest competitive assessment.

## Project Structure

```
src/crucible/
├── core/          # Config, env, I/O, types, logging, finding, hub
├── fleet/         # Provider-abstracted fleet management
│   └── providers/ # RunPod, SSH backends
├── runner/        # Experiment execution, output parsing, presets, tracking, notes
├── models/        # Model zoo (components + architectures)
├── researcher/    # LLM-driven autonomous research loop, briefing
├── analysis/      # Leaderboard, sensitivity, Pareto frontier
├── data/          # Manifest-driven HuggingFace data pipeline
├── mcp/           # MCP server for Claude agent integration (77 tools)
├── training/      # Training backends (torch) — factored from train_gpt.py
├── api/           # Lightweight REST API server (FastAPI)
├── tui/           # Interactive experiment design browser (Textual)
└── cli/           # CLI entry points
```

## License

MIT
