# Changelog

## v0.2.0-alpha (2026-03-22)

**Experiment tracking experience.** Major additions for cross-project knowledge sharing, experiment annotation, and agent-friendly APIs.

### New Modules

- **`api/`** — Lightweight REST API server (FastAPI) exposing 10 endpoints that wrap MCP tools. Start with `crucible serve`.
- **`core/hub.py`** — Crucible Hub (`~/.crucible-hub/`), a git-synced cross-project knowledge store for findings and research tracks.
- **`core/finding.py`** — Finding model and promotion logic for elevating project-level insights to the hub.
- **`runner/notes.py`** — Experiment notes system: freeform markdown with YAML frontmatter, attached to run IDs.
- **`researcher/briefing.py`** — Research briefing generator for LLM session orientation (project context, recent findings, track state).

### 27 New MCP Tools (53 total, was 26)

**Notes** (3 tools):
- `note_add` — Attach a markdown note to a run
- `note_get` — Retrieve notes for a run
- `note_search` — Full-text search across all notes

**W&B Bridge** (3 tools):
- `wandb_log_image` — Log an image to a W&B run
- `wandb_get_url` — Get the W&B dashboard URL for a run
- `wandb_annotate` — Add annotations to a W&B run

**Hub** (2 tools):
- `hub_status` — Hub state: active track, synced projects, finding count
- `hub_sync` — Push/pull hub directory via git

**Tracks** (3 tools):
- `track_create` — Create a named research track
- `track_list` — List all tracks with metadata
- `track_switch` — Switch the active research track

**Findings** (2 tools):
- `hub_findings_query` — Search findings across all projects in the hub
- `finding_promote` — Promote a project finding to the hub

**Briefing** (2 tools):
- `get_research_briefing` — Generate LLM session orientation summary
- `annotate_run` — Add structured annotations to a completed run

### New CLI Commands

- `crucible hub {status|sync|findings}` — Manage the Crucible Hub
- `crucible track {create|list|switch}` — Research track management
- `crucible note {add|get|search}` — Experiment note management
- `crucible serve [--port PORT]` — Start the REST API server
- `crucible store {list|diff|get}` — Version store inspection

### REST API

10 FastAPI endpoints wrapping core MCP tools:
- `GET /api/fleet/status`, `POST /api/fleet/provision`, `DELETE /api/fleet/destroy`
- `GET /api/experiments/queue`, `POST /api/experiments/enqueue`, `GET /api/experiments/{run_id}`
- `GET /api/analysis/leaderboard`, `GET /api/analysis/sensitivity`
- `GET /api/research/state`, `GET /api/research/briefing`

### W&B Bridge Enhancements

- Image logging support (`wandb_log_image`)
- Run URL retrieval (`wandb_get_url`)
- Run annotation with structured metadata (`wandb_annotate`)

### Research Tracks & Cross-Project Findings

- Research tracks group related projects under named directions
- Findings can be promoted from project-level context to the hub
- Hub is git-synced for sharing across machines and collaborators
- Briefing system orients new LLM sessions with accumulated knowledge

### Model Extensibility (12 tools)

- `model_list_families` — List registered model families
- `model_list_activations` — List available activation functions
- `model_list_components` — List model components
- `model_get_config_schema` — Get parameter schema for a family
- `model_validate_config` — Validate experiment config against schema
- `model_add_architecture` — Register a user architecture plugin
- `model_add_activation` — Register a custom activation function
- `model_generate_template` — Generate plugin boilerplate

**Config** (2 tools):
- `config_get_presets` — List all presets with resolved values
- `config_get_project` — Full project configuration

### Architecture Plugin System

- User architectures live in `models/user_architectures/` and auto-register on import
- `example_two_tower.py` — working example plugin (two-tower with gated fusion)
- Plugin contract: factory function `(args) -> nn.Module`
- Plugins sync to pods automatically via rsync

### Robustness Fixes

- Narrowed exception handling in model registry fallback — plugin errors now propagate with real tracebacks instead of being swallowed
- Added `TYPE_CHECKING` import guard in `lora.py` to fix circular import
- `list_families()` gracefully handles torch-absent environments

### Test Suite

Fleet orchestration, architecture forward-pass, component, and runner execution tests added. Previous: 296 tests in v0.1.0.

---

## v0.1.0-alpha (2026-03-21)

**First release.** Extracted from the [OpenAI Parameter Golf](https://github.com/openai/parameter-golf) competition infrastructure and generalized into a standalone ML research platform.

### What's in this release

**Core Platform** (74 source files, ~10,200 lines)

- **`core/`** — Project configuration via `crucible.yaml`, .env file loading, atomic JSON/JSONL I/O, structured logging, shared type definitions, error hierarchy (`CrucibleError` → `ConfigError`, `FleetError`, `RunnerError`, `ResearcherError`, `DataError`)
- **`fleet/`** — Provider-abstracted fleet management with RunPod REST API and generic SSH backends. Node inventory, experiment queue (JSONL), wave-based scheduling with dispatch and early stopping, SSH/rsync sync, bootstrap (pip install, data download, CUDA verification), day/night run orchestration, live fleet monitoring
- **`runner/`** — Single-experiment execution with live stdout streaming, configurable output parsing (regex-based), OOM retry with halved batch size, status sidecars and heartbeats, code fingerprinting for dedup, optional W&B integration, tiered presets (smoke/proxy/medium/promotion/overnight)
- **`models/`** — PyTorch model zoo with 12 reusable components (RMSNorm, CastedLinear, Rotary/RoPE, CausalSelfAttention with GQA/paired/windowed, MLP, SmearGate, BigramHash, TrigramHash, DepthwiseConv1D, TokenMerger, CausalPrefixMemory, BatchedTTTLoRA) and 4 architecture families (baseline, looped, convloop, prefix_memory) with auto-registration registry
- **`researcher/`** — LLM-driven autonomous research loop: analyze results → generate hypotheses (Claude) → design experiment batches → execute on fleet or locally → reflect on outcomes → promote winners / kill dead ends. JSONL-backed persistent state for hypotheses, beliefs, and budget tracking. Dry-run mode with fixture data for testing without LLM calls
- **`analysis/`** — Results loading with local + fleet merge and dedup, configurable metric leaderboard ranking, per-parameter sensitivity analysis, Pareto frontier (metric vs model size), markdown summary generation, JSON config export
- **`data/`** — Manifest-driven HuggingFace dataset download with shard selection, local cache management, fleet data sync via rsync
- **`mcp/`** — MCP server exposing 10 fleet/research tools for Claude agent integration (get_fleet_status, get_leaderboard, enqueue_experiment, provision_nodes, destroy_nodes, sync_code, get_research_state, get_sensitivity, get_experiment_result, get_queue_status)
- **`cli/`** — Full CLI with subcommands: `crucible {init, fleet, run, analyze, research, data, mcp, models}`

**Test Suite** (16 test files, ~3,100 lines, 296 tests)

- Core: config loading/generation, I/O operations, env file parsing
- Runner: output parser patterns, presets, tracker, fingerprinting
- Researcher: state persistence, LLM JSON parsing, batch design
- Analysis: leaderboard ranking, sensitivity, results loading/merging
- Fleet: queue operations, inventory management

**Configuration**

- `crucible.yaml` — Declarative project config (provider, training scripts, presets, metrics, researcher, data, sync excludes)
- Configurable metrics via `metrics.primary` — no hardcoded metric assumptions
- Training contract: env vars in, stdout patterns out — works with any training script
- Provider abstraction: RunPod and SSH today, extensible for others

**Validated End-to-End**

- Provisioned 2 RunPod pods via REST API
- Bootstrapped (code sync, pip install, CUDA check, dataset download)
- Enqueued and dispatched 3 SOTA experiments from JSON spec
- Live W&B logging with training loss curves
- Result collection via rsync
- Pod destruction via API

### What's NOT in this release

- No multi-cloud support (RunPod + SSH only)
- No Optuna/Ax integration for mathematical HPO
- No SkyPilot integration
- No PyPI package (install from source)
- No CI/CD pipeline
- No active process monitoring during dispatch (result collection is poll-based)
- Model zoo components are from one competition — not a general-purpose library

### Origin

Every line of code in this release was extracted from the `parameter-golf` competition fork (`dev` branch), refactored for generality, and restructured into a clean package. The original 3,089-line `fleet.py` monolith was decomposed into 13 focused modules. The 614-line `torch_models.py` was split into 20 component and architecture files. Competition-specific references (hardcoded metrics, dataset paths, experiment names) were replaced with configurable alternatives.
