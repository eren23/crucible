---
layout: default
title: Getting Started
---

# Getting Started

## Installation

```bash
pip install crucible-ml[all]
```

Or install specific components:
```bash
pip install crucible-ml            # Core only (config, store, CLI)
pip install crucible-ml[tui]       # + Interactive TUI
pip install crucible-ml[torch]     # + PyTorch model zoo
pip install crucible-ml[mcp]       # + MCP server
pip install crucible-ml[anthropic] # + Claude integration
```

## Initialize a Project

```bash
mkdir my-research && cd my-research
crucible init
```

This creates `crucible.yaml` with default configuration. Edit it to set your:
- **Provider**: RunPod or SSH compute backend
- **Metrics**: What to optimize (val_loss, val_bpb, etc.)
- **Presets**: Experiment tiers (smoke, proxy, medium, promotion)
- **Researcher**: Claude model and budget for autonomous loop

## Create Your First Experiment Design

```bash
# Launch the interactive TUI
crucible tui
```

Or via CLI:
```bash
crucible store list designs
```

Or via MCP (from Claude):
```json
{
  "tool": "version_save_design",
  "arguments": {
    "name": "my-first-experiment",
    "config": {"MODEL_FAMILY": "baseline", "NUM_LAYERS": "9"},
    "hypothesis": "9 layers should be a good starting point",
    "base_preset": "smoke"
  }
}
```

## Run an Experiment

### Locally
```bash
crucible run experiment --preset smoke --name my-test
```

### On Fleet (RunPod)
```bash
crucible fleet provision --count 1
crucible fleet bootstrap
crucible run enqueue --spec experiments.json
crucible run dispatch
crucible run collect
```

### Via MCP (Agent-Driven)
Agents call `version_run_design` to execute a versioned design:
```json
{
  "tool": "version_run_design",
  "arguments": {"design_name": "my-first-experiment"}
}
```

For an external-project example that starts from a fresh directory and runs through MCP, see [YOLO MCP Demo](yolo-mcp-demo).

## Analyze Results

```bash
crucible analyze rank --top 10
crucible analyze sensitivity
crucible analyze pareto
```

Or via TUI — press `c` to view research context, `h` for version history.
