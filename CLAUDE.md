# Crucible Development Guide

## What is this?

Crucible is an ML research platform for autonomous experimentation on rental GPUs. It combines LLM-driven hypothesis generation with fleet orchestration (RunPod, SSH). Currently alpha — born from the OpenAI Parameter Golf competition.

## Architecture

```
src/crucible/
├── core/          # Config, env, I/O, types, logging — no external deps except pyyaml
├── fleet/         # Provider-abstracted fleet management (RunPod, SSH)
│   └── providers/ # Compute backends (runpod.py, ssh.py)
├── runner/        # Experiment execution, output parsing, presets, tracking
├── models/        # Model zoo — PyTorch transformer components + architectures
├── researcher/    # LLM-driven autonomous research loop (Claude-first)
├── analysis/      # Leaderboard, sensitivity analysis, Pareto frontier
├── data/          # Manifest-driven HuggingFace data pipeline
├── mcp/           # MCP server exposing fleet ops as Claude tools
└── cli/           # CLI entry points (crucible command)
```

## Key Conventions

### Imports
- `core/` modules have NO dependencies on other crucible modules
- All other modules import from `core/` for shared utilities
- `fleet/`, `runner/`, `analysis/`, `data/` are independent of each other
- `researcher/` may import from `fleet/`, `runner/`, `analysis/` (it orchestrates them)
- `mcp/` imports from everything (thin wrapper layer)
- `cli/` imports from everything (entry points)
- External deps (torch, anthropic, mcp, wandb, huggingface_hub) are lazy-imported where optional

### Error Handling
- Use `CrucibleError` hierarchy from `core/errors.py` (not bare `except Exception`)
- `ConfigError` for bad YAML / missing config
- `FleetError` for provider / SSH / provisioning failures
- `RunnerError` for experiment execution failures
- `ResearcherError` for LLM / hypothesis failures
- Let unexpected errors propagate — don't catch and swallow

### Testing
- Tests in `tests/` mirror the source structure: `tests/test_config.py` tests `core/config.py`
- Run with: `PYTHONPATH=src pytest tests/`
- No torch dependency for non-model tests
- Use `tmp_path` fixture for file I/O tests
- Integration tests requiring network/GPUs go in `tests/integration/` and are skipped by default

### Configuration
- All paths derived from `ProjectConfig` — no hardcoded paths
- Environment variables for secrets (RUNPOD_API_KEY, ANTHROPIC_API_KEY, WANDB_API_KEY)
- `crucible.yaml` for project-level config
- Presets (smoke, proxy, medium, promotion) merge built-in defaults with yaml overrides

### Training Contract
External training scripts interface with Crucible via:
- **Input**: Environment variables (config overrides set before script launch)
- **Output**: Stdout patterns that `OutputParser` recognizes:
  - `step:{step}/{total} train_loss:{loss}`
  - `step:{step}/{total} val_loss:{loss} val_bpb:{bpb}`
  - `Serialized model ... {N} bytes`

### CLI
- Entry point: `crucible` (via pyproject.toml console_scripts)
- Subcommands: `fleet`, `run`, `analyze`, `research`, `data`, `mcp`, `models`
- Each subcommand group has its own file in `cli/`

## Common Commands

```bash
# Run tests
PYTHONPATH=src pytest tests/ -v

# Run a smoke experiment
PYTHONPATH=src python -m crucible.cli.main run experiment --preset smoke

# Start MCP server
PYTHONPATH=src python -m crucible.mcp.server

# Check imports
PYTHONPATH=src python -c "import crucible; print(crucible.__version__)"
```

## What NOT to do
- Don't add experiment tracking UI — use W&B/MLflow
- Don't build Kubernetes support — use SkyPilot when ready
- Don't reinvent HPO math — integrate Optuna/Ax
- Don't hardcode paths — derive from ProjectConfig
- Don't catch and swallow errors — let them propagate with context
