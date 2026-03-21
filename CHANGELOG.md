# Changelog

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
