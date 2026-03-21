# Crucible

An ML research platform for fleet orchestration, autonomous experimentation, and model development.

## Origins

Crucible was born from [parameter-golf](https://github.com/openai/parameter-golf), OpenAI's March–April 2026 competition to train the best language model fitting in 16MB on 8xH100s in under 10 minutes. During the competition, we built a full-stack autonomous research system far beyond what a simple training script needs:

- **Multi-pod GPU fleet orchestration** on RunPod with wave-based scheduling
- **Claude-driven autonomous hypothesis generation** and experiment design
- **Tier-based experiment promotion** (smoke → proxy → medium → promotion)
- **MCP agent integration** so Claude can directly control experiments
- **A model zoo** with reusable transformer components
- **Real-time analysis** with leaderboards, sensitivity analysis, and Pareto frontiers

None of that infrastructure is competition-specific. Crucible extracts and generalizes it into a platform you can use for any ML research project — on RunPod, bare metal via SSH, or wherever you run experiments.

## Quick Start

```bash
# Install
pip install -e ".[all]"

# Initialize a new project
crucible init

# Edit crucible.yaml to point at your training script and compute

# Run a quick smoke test
crucible run experiment --preset smoke --set MODEL_FAMILY=baseline

# Provision GPU nodes and run at scale
crucible fleet provision --count 4
crucible fleet bootstrap
crucible run day --count 4
```

## Core Concepts

### crucible.yaml

Like docker-compose for ML experiments. Defines your compute provider, training scripts, data sources, experiment presets, and autonomous research program.

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

Crucible doesn't own your training code. Any script that reads environment variables and prints parseable output works:

**Input** (env vars set by Crucible):
- Preset defaults: `ITERATIONS`, `MAX_WALLCLOCK_SECONDS`, `TRAIN_BATCH_TOKENS`
- Config overrides: `MODEL_FAMILY`, `NUM_LAYERS`, `MODEL_DIM`, etc.
- Metadata: `RUN_ID`, `RUN_BACKEND`, `RUN_PRESET`

**Output** (stdout patterns Crucible parses):
- `step:{step}/{total} train_loss:{loss}`
- `step:{step}/{total} val_loss:{loss} val_bpb:{bpb}`
- `Serialized model ... {N} bytes`

### Fleet Management

Provision and manage GPU nodes across providers. Currently supports RunPod (API-driven) and generic SSH (manual host list).

```bash
crucible fleet provision --count 4 --name-prefix my-run
crucible fleet bootstrap          # sync code + data to all nodes
crucible fleet status             # check node health
crucible fleet destroy            # tear down when done
```

### Experiment Tiers

Experiments progress through tiers of increasing cost:

| Tier | Duration | Use Case |
|------|----------|----------|
| smoke | ~1 min | Quick validation |
| proxy | ~30 min | Main exploration |
| medium | ~1 hr | Extended runs |
| promotion | ~2 hrs | Best candidates |

### Autonomous Researcher

Claude-powered experiment loop: analyze results → generate hypotheses → design batches → execute → reflect → promote or kill.

```bash
crucible research start --budget-hours 10 --tier proxy
```

### Model Zoo

Shipped transformer components and architectures, independently importable:

```python
from crucible.models.registry import build_model, list_families
from crucible.models.components.attention import CausalSelfAttention
from crucible.models.components.rotary import Rotary
```

Architectures: baseline, looped, convloop, prefix_memory. Extensible via `register_model()`.

### MCP Integration

All fleet and research operations available as MCP tools for Claude agent workflows:

```bash
crucible mcp serve  # starts stdio MCP server
```

## CLI Reference

```
crucible init                              # Create crucible.yaml
crucible fleet {status|provision|destroy|bootstrap|sync|monitor}
crucible run {experiment|queue|enqueue|dispatch|collect|day|night}
crucible analyze {rank|sensitivity|pareto|export|summary}
crucible research {start|status}
crucible data {download|sync|status}
crucible mcp serve
crucible models list
```

## Installation

```bash
# Full install
pip install crucible-ml[all]

# Minimal (just orchestration, no torch/models)
pip install crucible-ml

# With specific extras
pip install crucible-ml[torch]       # model zoo
pip install crucible-ml[anthropic]   # autonomous researcher
pip install crucible-ml[mcp]         # MCP server
pip install crucible-ml[data]        # HuggingFace data pipeline
pip install crucible-ml[wandb]       # W&B logging
```

## Project Structure

```
src/crucible/
├── core/          # Config, env, I/O, types, logging
├── fleet/         # Provider-abstracted fleet management
│   └── providers/ # RunPod, SSH backends
├── runner/        # Experiment execution, output parsing, presets
├── models/        # Model zoo (components + architectures)
├── researcher/    # LLM-driven autonomous research loop
├── analysis/      # Leaderboard, sensitivity, Pareto frontier
├── data/          # Manifest-driven HuggingFace data pipeline
├── mcp/           # MCP server for Claude agent integration
└── cli/           # CLI entry points
```

## License

MIT
