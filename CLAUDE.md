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
├── training/      # Training backends (torch) — extracted from train_gpt.py
├── models/        # Model zoo — components, architectures, declarative composer
│   ├── components/     # Reusable blocks (Attention, MLP, MoE, RMSNorm, etc.)
│   ├── architectures/  # 4 built-in architectures + plugin auto-discovery
│   ├── specs/          # YAML architecture specs (declarative definitions)
│   └── composer.py     # Declarative architecture composition engine
├── researcher/    # LLM-driven autonomous research loop, briefing (Claude-first)
├── analysis/      # Leaderboard, sensitivity analysis, Pareto frontier
├── data/          # Manifest-driven HuggingFace data pipeline
├── mcp/           # MCP server exposing fleet ops as Claude tools (64 tools)
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

### Experiment Lifecycle

1. **Design** → `version_save_design` or create YAML in `.crucible/designs/`
2. **Wave Spec** → JSON array in `specs/` (or use `design_enqueue_batch` directly)
3. **Provision** → `provision_nodes` creates RunPod pods
4. **Refresh** → `fleet_refresh` gets SSH endpoints
5. **Bootstrap** → `bootstrap_nodes` syncs code, installs deps, downloads data
6. **Enqueue** → `design_enqueue_batch` or `enqueue_experiment` adds to queue
7. **Dispatch** → `dispatch_experiments` assigns queued runs to idle nodes
8. **Monitor** → `get_fleet_status` + `get_queue_status` for progress
9. **Collect** → `collect_results` rsyncs results from pods
10. **Results** → `get_leaderboard` ranks by val_bpb

**Presets** control experiment scale:
- `smoke` — 60s, 400 steps. Quick syntax check.
- `screen` — 1h, 2000 steps. Directional signal for architecture screening.
- `proxy` — 30min, 6000 steps. Medium confidence.
- `medium` — 1h, 15K steps. Thorough comparison.
- `promotion` — 2h, 100K steps. Competition-grade.

### MCP Tools (64 total)

**Tier 1 — Core Experiment Flow** (use these to run experiments):
`provision_nodes` → `fleet_refresh` → `bootstrap_nodes` → `design_enqueue_batch` → `dispatch_experiments` → `collect_results` → `get_leaderboard`

Plus: `get_fleet_status`, `get_queue_status`, `destroy_nodes`, `cancel_experiment`, `clear_stale_queue`

**Tier 2 — Experiment Design:**
`version_save_design`, `version_list_designs`, `version_run_design`, `version_get_design`, `config_get_presets`, `config_get_project`

**Tier 3 — Research Context:**
`context_push_finding`, `context_get_findings`, `get_research_briefing`, `note_add`, `note_search`, `note_get`

**Tier 4 — Model Extensibility (Code Plugins):**
`model_list_families`, `model_add_architecture`, `model_generate_template`, `model_validate_config`

**Tier 5 — Declarative Architecture Composition (Lego Blocks):**
`model_compose`, `model_from_template`, `model_list_stack_patterns`, `model_list_block_types`, `model_preview_spec`, `model_get_spec`

**Important**: `bootstrap_nodes`, `dispatch_experiments`, `collect_results`, and `sync_code` are long-running operations (minutes). The MCP server runs them in background threads via `asyncio.to_thread()` to prevent stdio pipe timeouts.

### Architecture Plugins

Crucible has a compact core with 4 built-in architectures (baseline, looped, convloop, prefix_memory). Everything else is a **plugin** — created by users or agents at runtime, never baked into core.

**Two ways to create architectures:**

**Option A — Declarative Composition (recommended, no code):**
Compose from known components via YAML specs. Uses the `model_compose` MCP tool.
1. `model_list_stack_patterns()` — see available wiring patterns (sequential, looped, encoder_decoder_skip, etc.)
2. `model_list_block_types()` — see available blocks (attention_block, prefix_memory_block)
3. `model_compose(name="my_arch", spec={block: {...}, stack: {...}, augmentations: {...}})` — creates `.crucible/architectures/my_arch.yaml`
4. Or use `model_from_template(name="my_arch", base="baseline", overrides={...})` to fork an existing spec
5. Run it: `provision_nodes` → `bootstrap_nodes` → enqueue with `MODEL_FAMILY: my_arch` → dispatch → collect

Specs are YAML files — no Python written. The `ComposedArchitecture` class interprets them at runtime.

**Option B — Python Plugin (for novel forward logic):**
When you need custom forward passes that YAML can't express.
1. `model_generate_template(name="two_tower")` — get boilerplate
2. Edit the code to implement your architecture
3. `model_add_architecture(name="two_tower", code="...")` — saves to `.crucible/architectures/two_tower.py` and registers
4. Run it the same way

**Plugin discovery** (3-tier with precedence):
- **Builtin** (lowest): 4 core architectures in `src/crucible/models/architectures/`
- **Global** (hub): `~/.crucible-hub/architectures/plugins/*.py` + `*.yaml`
- **Local** (highest): `.crucible/architectures/*.py` + `*.yaml`

Both `.py` and `.yaml` files are auto-discovered. `.py` takes precedence over `.yaml` at the same scope.

**The contract:** A plugin (Python or YAML spec) produces an `nn.Module` from an `args` namespace (with `vocab_size`, `model_dim`, `num_layers`, etc.).

**What stays in core:** baseline, looped, convloop, prefix_memory — reference implementations from Parameter Golf. For new work, compose or build plugins.

### Known Limitations

- Only RunPod provider is fully tested (SSH provider is pass-through for manual hosts)
- `train_gpt.py` is a compatibility wrapper — actual training is in `src/crucible/training/`
- Hub features require explicit initialization (`crucible hub init`)
- W&B integration requires `wandb` package and `WANDB_API_KEY`

## What NOT to do
- Don't add new architectures to core — build plugins in `user_architectures/`
- Don't build a full experiment tracking UI — the TUI and REST API cover agent/developer needs; use W&B/MLflow for dashboards
- Don't build Kubernetes support — use SkyPilot when ready
- Don't reinvent HPO math — integrate Optuna/Ax
- Don't hardcode paths — derive from ProjectConfig
- Don't catch and swallow errors — let them propagate with context
