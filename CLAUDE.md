# Crucible Development Guide

## What is this?

Crucible is an ML research platform for autonomous experimentation on rental GPUs. It combines LLM-driven hypothesis generation with fleet orchestration (RunPod, SSH). Currently alpha — born from the OpenAI Parameter Golf competition.

## Architecture

```
src/crucible/
├── core/          # Config, env, I/O, types, logging, finding, hub — no external deps except pyyaml
├── fleet/         # Provider-abstracted fleet management (RunPod, SSH)
│   └── providers/ # Compute backends (runpod.py, ssh.py)
├── runner/        # Experiment execution, output parsing, presets, tracking, notes
├── models/        # Model zoo — PyTorch transformer components + architectures
├── researcher/    # LLM-driven autonomous research loop, briefing (Claude-first)
├── analysis/      # Leaderboard, sensitivity analysis, Pareto frontier
├── data/          # Manifest-driven HuggingFace data pipeline
├── mcp/           # MCP server exposing fleet ops as Claude tools
├── api/           # Lightweight REST API server (FastAPI)
├── tui/           # Interactive experiment design browser (Textual)
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
- `HubError` for hub sync / track / finding promotion failures
- `ApiError` for REST API server failures
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
- Presets (smoke, screen, proxy, medium, promotion, overnight) merge built-in defaults with yaml overrides

### Training Contract
External training scripts interface with Crucible via:
- **Input**: Environment variables (config overrides set before script launch)
- **Output**: Stdout patterns that `OutputParser` recognizes:
  - `step:{step}/{total} train_loss:{loss}`
  - `step:{step}/{total} val_loss:{loss} val_bpb:{bpb}`
  - `Serialized model ... {N} bytes`

### CLI
- Entry point: `crucible` (via pyproject.toml console_scripts)
- Subcommands: `fleet`, `run`, `analyze`, `research`, `data`, `mcp`, `models`, `hub`, `track`, `note`, `serve`, `tui`, `store`
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

### Version Store

The `.crucible/` directory provides per-project hybrid persistence. YAML files serve humans (browsable, diffable in git), while the JSONL ledger (`store.jsonl`) serves code (fast indexed access without filesystem scanning). Designs, contexts, and notes all live under `.crucible/` with versioned history (`v1.yaml`, `v2.yaml`, `current.yaml`).

### Hub

`~/.crucible-hub/` is the cross-project knowledge store. It holds research tracks (groupings of related projects/directions), findings promoted from individual projects, and a git-synced index for sharing across machines. Key concepts:

- **Tracks**: Named research directions that group projects (e.g., "attention-variants", "scaling-laws")
- **Findings**: Insights promoted from project-level context to the hub for cross-project visibility
- **Finding promotion**: `context_push_finding` records locally, `finding_promote` elevates to the hub
- **Git sync**: `hub_sync` pushes/pulls the hub directory as a git repo

### Fleet Operations (Running on Pods)

Crucible is self-contained — all fleet operations run FROM this repo:

```bash
cd /path/to/parameter-golf_dev
PYTHONPATH=src python3 -c "
from crucible.fleet.manager import FleetManager
from crucible.core.config import load_config
from crucible.core.env import load_env_files
load_env_files('.')
fm = FleetManager(load_config())
fm.run_day(count=6, wave_specs=[('wave1', Path('specs/screen_batch_1.json'))])
"
```

**Bootstrap sequence** (what happens on each pod):
1. `sync_repo()` — rsync project to pod's `workspace_path`
2. `sync_env_file()` — copy `.env.runpod.local` (WandB keys) to pod
3. Python + CUDA validation
4. `pip install -r requirements.txt` (skips torch — already in pod image)
5. Data probe + download via `data/cached_challenge_fineweb.py`
6. Node marked `state: ready`

**Secrets flow**: `.env.runpod.local` contains only pod-needed secrets (WandB). `RUNPOD_API_KEY` stays local in `.env`. The `env_source` in `provider.defaults` controls which file gets synced.

**Runner script**: `src/crucible/runner/run_remote.py` is the CLI entry point invoked on pods by the scheduler. It wraps `run_experiment()` from `experiment.py`.

**Provider defaults** (in `crucible.yaml` `provider.defaults`):
- `workspace_path` — where code lands on pod (default: `/workspace/project`)
- `python_bin` — Python binary on pod (default: `python3`)
- `env_source` — which .env file to sync (default: `.env.local`)

### Experiment Designs

Designs live in `.crucible/designs/` as versioned YAML. Wave specs in `specs/` are JSON arrays consumed by `run_day()`. Create wave specs from designs, not the other way around.

## What NOT to do
- Don't build a full experiment tracking UI — the TUI and REST API cover agent/developer needs; use W&B/MLflow for dashboards
- Don't build Kubernetes support — use SkyPilot when ready
- Don't reinvent HPO math — integrate Optuna/Ax
- Don't hardcode paths — derive from ProjectConfig
- Don't catch and swallow errors — let them propagate with context
