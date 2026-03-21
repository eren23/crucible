---
layout: default
title: Architecture
---

# Architecture

## Module Structure

```
src/crucible/
  core/           Config, I/O, types, logging, version store
  fleet/          Provider-abstracted fleet management
    providers/    RunPod, SSH backends
  runner/         Experiment execution, output parsing, presets
  models/         Model zoo — PyTorch transformer components
    architectures/  baseline, looped, convloop, prefix_memory
    components/     attention, MLP, normalization, embeddings
  researcher/     LLM-driven autonomous research loop
  analysis/       Leaderboard, sensitivity, Pareto frontier
  data/           Manifest-driven HuggingFace data pipeline
  mcp/            MCP server (26 tools)
  tui/            Interactive terminal UI (Textual)
  cli/            CLI entry points
```

## Design Principles

### Import Rules
- `core/` has NO dependencies on other crucible modules
- `fleet/`, `runner/`, `analysis/`, `data/` are independent of each other
- `researcher/` orchestrates fleet, runner, and analysis
- `mcp/` and `cli/` are thin wrapper layers that import from everything
- External deps (torch, anthropic, mcp) are lazy-imported

### Version Store

The `.crucible/` directory provides hybrid persistence:

```
.crucible/
  store.jsonl              JSONL ledger (source of truth)
  designs/{name}/
    current.yaml           Latest version
    v1.yaml, v2.yaml...   Version history
  context/{name}/
    current.yaml
    v1.yaml...
```

**YAML files** serve humans — browse in editor, diff in git, read without tooling.
**JSONL ledger** serves code — fast indexed access without filesystem scanning.

### Experiment Lifecycle

```
ExperimentDesign (versioned YAML)
       |
       v
design_to_experiment_config()
       |
       v
ExperimentConfig (env var overrides)
       |
       v
run_experiment() or enqueue to fleet
       |
       v
ExperimentResult (JSONL)
       |
       v
link_result_to_design() (back-link)
```

### Training Contract

Training scripts interface via:
- **Input**: Environment variables (config overrides)
- **Output**: Stdout patterns:
  - `step:{step}/{total} train_loss:{loss}`
  - `step:{step}/{total} val_loss:{loss} val_bpb:{bpb}`
  - `Serialized model ... {N} bytes`

### Error Hierarchy

```
CrucibleError
  ConfigError       Bad YAML, missing config
  FleetError        Provider, SSH failures
  RunnerError       Experiment execution failures
  ResearcherError   LLM, hypothesis failures
  DataError         Data pipeline failures
  StoreError        Version store failures
```

## Model Zoo

Four transformer architectures with 9 activation functions:

| Family | Key Feature |
|--------|------------|
| `baseline` | Standard transformer with U-Net skip connections |
| `looped` | Weight-sharing recurrent (more compute, same params) |
| `convloop` | Bottleneck compression + looped core |
| `prefix_memory` | Bounded internal memory variant |

Activation functions: `relu_sq`, `leaky01_sq`, `leaky02_sq`, `leaky08_sq`, `mish_sq`, `gelu_sq`, `elu03_sq`, `x_absx`, `log1p_relu_sq`
