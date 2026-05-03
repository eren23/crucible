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
├── training/      # Training backends (torch, generic) — modality-agnostic pipeline
├── models/        # Model zoo — components, architectures, declarative composer
│   ├── components/     # Reusable blocks (Attention, MLP, MoE, RMSNorm, etc.)
│   ├── architectures/  # 4 built-in architectures + plugin auto-discovery
│   ├── specs/          # YAML architecture specs (declarative definitions)
│   └── composer.py     # Declarative architecture composition engine
├── researcher/    # LLM-driven autonomous research loop, briefing (Claude-first)
├── analysis/      # Leaderboard, sensitivity analysis, Pareto frontier
├── data/          # Manifest-driven HuggingFace data pipeline
├── mcp/           # MCP server exposing fleet ops as Claude tools (118 tools)
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
- `DataError` for data manifest / download failures
- `StoreError` for version store failures
- `ComposerError` for declarative architecture composition failures
- `SearchTreeError` for tree search failures
- `PluginError` for plugin registration / discovery / build failures
- `TapError` for tap clone / sync / install / publish failures
- `DomainSpecError` for harness domain spec loading / validation failures
- `CandidateValidationError` for harness candidate validation failures
- `HarnessOptimizerError` for harness optimizer orchestration failures
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
- Subcommands: `fleet`, `run`, `analyze`, `research`, `data`, `mcp`, `models`, `hub`, `tap`, `track`, `note`, `serve`, `tui`, `store`
- Each subcommand group has its own file in `cli/`

## Orchestrator contract (default path — no LLM keys in Crucible)

Crucible is **infrastructure** — pods, fleet, experiments, data, notes,
findings, plan, search. Taste (hypothesis generation, reflection,
planning) is supplied by an **external orchestrator** — Claude Code via
MCP, another agent, or a human. The orchestrator supplies its own LLM.

**No `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `litellm` needed inside
Crucible.** The only creds it consumes are infrastructure creds
(`RUNPOD_API_KEY`, `WANDB_API_KEY`).

### The loop

```
1. research_request_prompt(stage="hypothesis")    → {system, user, schema}
2. Orchestrator calls its own LLM with that prompt, parses per schema.
3. research_submit(stage="hypothesis", response)  → hypotheses land in state.
4. design_batch_from_hypotheses → design_enqueue_batch →
   dispatch_experiments → collect_results                (pure fleet ops)
5. research_request_prompt(stage="reflection")    → next prompt + schema.
6. Orchestrator reflects, then research_submit(stage="reflection", ...) →
   beliefs updated, promote/kill lists returned.
7. Loop.
```

`stage="briefing"` is a read-only probe — returns a markdown summary of
project state (no schema, no submit needed). Useful as a periodic
sanity check.

### Legacy autonomous mode (opt-in)

The pre-existing `AutonomousResearcher` (`crucible research start`)
still runs a closed-loop agent with an internal Anthropic SDK call. It
requires `ANTHROPIC_API_KEY`. It is **NOT the default** — it exists for
users who explicitly want it. New work should use the orchestrator
contract above.

## Reference docs (read BEFORE debugging config / fleet issues)

- **`docs/crucible-config-hierarchy.md`** — definitive map of which config layer wins for `provision_project` / `bootstrap_project` / `run_project`. Documents the full precedence table (ranks 0–12), the `nodes.json` interruptible echo bug, the correct playbook for running a project variant (pass `variant_name` to `run_project` or inline env via `overrides`), and 10 common gotchas. **If you are about to edit a project spec yaml or debug a RunPod pod that "looks stuck", read §3 and §4 of that doc first.**

## W&B contract — default behavior

`wandb.required=true` is the default in `crucible.yaml` (see `WandbConfig` in `src/crucible/core/config.py`). When `True`, both layers fail loudly:

1. **Enqueue-time** (`validate_experiment_contract`): `enqueue_experiment` / `design_enqueue_batch` / `version_run_design` raise `ConfigError` if `wandb.project` is empty (in yaml or env) OR if `wandb.mode != "disabled"` and `WANDB_API_KEY` is unset.
2. **Runtime** (`run_experiment`): the launcher raises `RunnerError` if `WandbLogger.create()` returned an inert logger. Gate condition: `CRUCIBLE_ENFORCE_CONTRACT=1` always enforces, `=0` always opts out, unset → honors `wandb.required`.

**Opting out for development**: set `wandb.required=false` in `crucible.yaml`. That single setting bypasses both layers. `wandb.mode=disabled` alone is NOT a full opt-out — it only relaxes the API-key check while keeping `wandb.project` required. `CRUCIBLE_ENFORCE_CONTRACT=0` only relaxes the runtime gate; enqueue still rejects.

**Discoverability**: agents can call `get_wandb_guide` (MCP) for the decision tree, checklist, common failures, and verification steps. The canonical workflow lives at `docs/recipes/wandb-tracked-experiment.yaml` (source-controlled). To make it accessible via `recipe_get(name='wandb-tracked-experiment')` in a project, copy the file into that project's `.crucible/recipes/` (the recipe system reads from there). Tap-based distribution is a follow-up so cross-project `recipe_get` works without the manual copy step.

**HF collab recipe**: `docs/recipes/hf-collab-parameter-golf.yaml` is the canonical end-to-end workflow for cross-agent collaboration via HuggingFace Hub. Operator setup (tokens, repo provisioning, opt-in flags, smolagents integration, HF Space deployment) is covered in `docs/hf-collab-recipe.md`. Same `cp .crucible/recipes/` install pattern as wandb until the tap layer ships.

**Auto-defaults**:
- If `WANDB_API_KEY` and `WANDB_PROJECT` are both present in env and `LOGGING_BACKEND` is unset, the runner sets `LOGGING_BACKEND=wandb,console` (generic backend only — `torch_backend.py` instantiates `WandbLogger` directly, so the env var is moot there).
- If only one of the two is present, runner falls back to `console` and emits a warning.

**Run-name policy**: `WANDB_RUN_NAME` defaults to `exp_id` (UUID-ish), which collides across variants in the same batch. `enqueue_experiment` and `design_enqueue_batch` return a `warnings` list when neither `WANDB_RUN_NAME` nor `CRUCIBLE_VARIANT_NAME` are set in the config. The `run_project` path auto-derives `WANDB_RUN_NAME` from the variant name.

## Judge-separation contract — LM-as-judge loops

Any LM-as-judge loop in Crucible (currently: `harness_iterate`, `tree_expand_grpo`) must declare a `JudgePanel` with distinct reward and eval judges in different families. Same model = identical reward hacks; same family = correlated failure modes. Mirrors the GIANTS recipe (https://giants-insights.github.io/) — small/cheap reward judge, different-family eval judge, optional independent audit judge.

Configure in `crucible.yaml`:

```yaml
judges:
  reward_judge: {model: gemini-2.5-flash, family: gemini}
  eval_judge:   {model: claude-opus-4-7, family: claude}
  audit_judge:  {model: qwen3-14b, family: qwen}   # optional
  enforce_separation: true   # default; set false to downgrade to warning
```

When `judges:` is absent or all model strings are blank, the panel is *unconfigured* and enforcement is skipped — opt-in only. Tools call `panel.assert_separated()` before any LLM work; mis-separated panels raise `ConfigError` before pod time is consumed. Full doc: `docs/judge-separation.md`. Canonical recipe: `docs/recipes/judge-separated-harness.yaml`.

## GIANTS-style synthesis hypotheses — `design_synthesize_from_findings`

Mines pairs of hub findings (cross-project / cross-track) and returns one orchestrator-shaped prompt bundle per pair so your LLM can predict the experiment that synthesizes both parents. Pure orchestrator-contract: no internal LLM call. Pair-mining policies: `random`, `same_track`, `cross_track`. Each generated hypothesis carries `parent_finding_ids` so provenance reaches the batch and W&B run tags. Recipe: `docs/recipes/synthesis-driven-research.yaml`.

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
5. **Bootstrap** → `bootstrap_nodes` syncs code, installs deps, probes data source status, skips download if PARTIAL
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

**W&B Best Practice**: Related experiments (e.g., architecture variants) should share one WANDB_PROJECT. Set the same `env_set.WANDB_PROJECT` across related project specs. The variant name (`CRUCIBLE_VARIANT_NAME` / `WANDB_RUN_NAME`) distinguishes individual runs within the project. Don't create separate W&B projects per architecture variant — this fragments the leaderboard.

### MCP Tools (157 total)

**Tier 1 — Core Experiment Flow** (use these to run experiments):
`provision_nodes` → `fleet_refresh` → `bootstrap_nodes` → `design_enqueue_batch` → `dispatch_experiments` → `collect_results` → `get_leaderboard`

Plus: `get_fleet_status` (with optional `include_metrics` for live GPU/memory/disk), `get_queue_status`, `destroy_nodes`, `stop_nodes`, `start_nodes`, `cancel_experiment`, `clear_stale_queue`, `purge_queue`

**Tier 1b — RunPod Enhanced Operations** (GraphQL-powered):
`runpod_gpu_availability` — GPU types with spot/on-demand pricing.
`runpod_list_volumes`, `runpod_create_volume`, `runpod_delete_volume` — Persistent shared storage.
`runpod_list_templates`, `runpod_create_template` — Reusable pod templates.
`provision_nodes` also accepts optional `network_volume_id` and `template_id` params.

**Tier 2 — Experiment Design:**
`version_save_design`, `version_list_designs`, `version_run_design`, `version_get_design`, `config_get_presets`, `config_get_project`, `design_synthesize_from_findings` (GIANTS-style hypothesis seeding from cross-project hub findings — pure orchestrator-contract, no internal LLM call)

**Tier 3 — Research Context:**
`context_push_finding`, `context_get_findings`, `get_research_briefing`, `note_add`, `note_search`, `note_get`

**Tier 4 — Model Extensibility (Code Plugins):**
`model_list_families`, `model_add_architecture`, `model_generate_template`, `model_validate_config`, `model_fetch_architecture`

**Tier 5 — Declarative Architecture Composition (Lego Blocks):**
`model_compose`, `model_from_template`, `model_list_stack_patterns`, `model_list_block_types`, `model_preview_spec`, `model_get_spec`

**Tier 6 — Agent Assistance:**
- `get_run_logs(run_id)` — Fetch training stdout/stderr (local logs or SSH fallback). Essential for debugging. Output is scrubbed for HF / Anthropic / OpenAI / GitHub / AWS / Bearer / env-style secrets before return.
- `model_fetch_architecture(family)` — Read source code/spec for any architecture. Enables read→modify→re-register workflow.
- `get_architecture_guide()` — Decision tree for declarative vs code plugin workflows.
- `agent_health_check(recent_calls)` — Optional doom-loop detector. Stateless: orchestrator passes its recent tool calls (oldest first), Crucible runs cycle / repetition detection and returns `{ok, pattern?, hint?}`. Suppresses false-positives on legitimate polling (same args, varying results). Equivalent library entry: `from crucible.core import detect_doom_loop`.

**Tier 7 — Tree Search (branching experiment exploration):**
`tree_create` → `tree_enqueue_pending` → `dispatch_experiments` → `collect_results` → `tree_sync_results` → `tree_get` → `tree_expand_node` or `tree_auto_expand` or `tree_expand_grpo` → repeat

Plus: `tree_prune`, `tree_list`. Supports UCB1, greedy, epsilon-greedy, agent-directed, and GRPO selection policies. `tree_expand_grpo` takes pre-scored candidates from the orchestrator's eval judge, computes group-relative advantage (z-score / min-max), keeps top-K, and stores `group_advantage` per kept node. Honors the judge-separation contract when `config.judges` is configured.

**Tier 8 — Training Generalization:**
`config_get_modalities` — List available training backends with modality tags, data adapters, and objectives.

**Tier 9 — Session Recipes:**
`recipe_save`, `recipe_list`, `recipe_get` — Save and retrieve step-by-step session playbooks. Captures MCP tool sequence, environment versions, gotchas with fixes, and results. Other agents follow a recipe to reproduce a successful session.

**Tier 10 — Plugin Registry (3 consolidated tools):**
`plugin_list(type)` — List registered plugins. Types: optimizers, schedulers, providers, loggers, callbacks, block_types, stack_patterns, augmentations.
`plugin_add(type, name, code)` — Register a new plugin from Python code at runtime.
`plugin_get_schema(type, name)` — Get config parameter schema (optimizers, schedulers only).

All plugin types use a unified `PluginRegistry` with 3-tier precedence (builtin < global < local). Plugins are auto-discovered from `.crucible/plugins/{type}/*.py` (local) and `~/.crucible-hub/plugins/{type}/*.py` (global).

**Tier 11 — Community Taps (plugin sharing):**
`hub_tap_add`, `hub_tap_remove`, `hub_tap_list`, `hub_tap_sync` — Manage git-based tap repositories.
`hub_search`, `hub_install`, `hub_uninstall`, `hub_installed` — Search, install, and manage community plugins.
`hub_publish`, `hub_tap_push`, `hub_submit_pr` — Publish plugins to taps, push, and open PRs.
`hub_package_info` — Get detailed package metadata and install status.

Taps are git repos containing plugins with `plugin.yaml` manifests. Install copies to `~/.crucible-hub/plugins/` (auto-discovered). Publish commits to a tap's local clone; user pushes or opens PR.

**Tier 12 — Harness Optimization (7 tools):**
`harness_init`, `harness_propose`, `harness_validate`, `harness_iterate`, `harness_frontier`, `harness_evolution_log` — Meta-harness-style evolutionary loop over task-specific harness code (memory systems, agent scaffolds) with N-dimensional Pareto frontier tracking.
`tree_pareto` — General-purpose Pareto frontier query for any search tree.

Candidates are stored as Python files under `.crucible/search_trees/{tree}/candidates/{node_id}.py`; domain specs ship as a `domain_specs` tap plugin type. See `docs/harness-optimization.md` and `.crucible/taps/meta-harness/` for the workflow and bundled templates.

**Tier 14 — HuggingFace Collab Publish (opt-in write side, 5 tools):**
`hf_push_artifact`, `hf_pull_artifact` — model checkpoints + eval bundles to/from a HF model repo (defaults to `hf_collab.artifacts_repo`; supports `{project}` substitution).
`hf_publish_leaderboard(top_n, challenge?)` — exports `get_leaderboard()` rows as `leaderboard.jsonl` + `README.md` to a HF Dataset repo. Each row carries a stable `challenge` field (default: project name) for cross-project filtering.
`hf_publish_findings` — exports project / track / global findings JSONL to a HF Dataset repo.
`hf_publish_recipes` — exports `.crucible/recipes/*.yaml` (optionally filtered) to a HF Dataset repo.

All five are gated by `hf_collab.enabled=true` in `crucible.yaml` (default `false` — never auto-pushes). Auth via `HF_TOKEN` env var. Backed by the `hub_remotes` plugin family (builtins: `git`, `hf_dataset`) so taps can drop in S3 / IPFS remotes without core changes. Lives in `src/crucible/core/hf_writer.py` and `src/crucible/core/hub_remotes.py`.

**Tier 15 — HuggingFace Collab Research-Front (read side, 3 tools):**
Work even with `hf_collab.enabled=false` (read-only flows do not require write opt-in).

`research_hf_prior_attempts(repo_id?, challenge_id?, top_k=10, primary_metric?, direction?)` — pulls peer agents' `leaderboard.jsonl` and returns top-k filtered/sorted rows. Default repo from `hf_collab.leaderboard_repo`. Best-effort: network/missing → `{ok, count: 0, runs: []}`. `challenge_id` matches against the stable `challenge` field that Tier-14 publishers embed.
`research_hf_discussions(repo_id, status='open'|'closed'|'all', limit=50)` — lists discussions on any HF repo as a peer-agent comm channel.
`note_post_to_hf_discussions(repo_id?, title?, body?, run_id?, note_id?)` — opens a HF Discussion containing a Crucible note. Resolves `title`/`body` from the local NoteStore when only `run_id`+`note_id` are given. Requires `hf_collab.enabled=true` (this is a write — sits in Tier 15 for surface coherence with the read tools).

Backed by `src/crucible/researcher/hf_search.py:fetch_prior_runs` (reads leaderboard.jsonl) and `src/crucible/researcher/hf_discussions.py` (list_discussions + post_discussion).

`get_research_briefing` includes a "Peer Agents — Prior Runs from HF" section only when **all three** are true: `hf_collab.enabled=true`, `hf_collab.briefing_auto_pull=true` (off by default — opt in or every briefing burns 1-30s of HF latency), and `hf_collab.leaderboard_repo` is set. Section is silently empty otherwise — never blocks.

`note_post_to_hf_discussions` runs `redact_secrets()` on title + body before posting (catches `HF_TOKEN`, `WANDB_API_KEY`, `sk-ant-*`, `sk-*`, GitHub PATs, AWS keys, bearer tokens, env-style `KEY=value`). A note containing a copy-pasted env dump or stack trace will not leak credentials to a public repo.

**Tier 13 — Eval Watcher (auto-eval daemon, 3 tools):**
`eval_watch_start(project_name, interval=300)` — start a daemon that polls running pods every `interval` seconds, SCPs new checkpoints to `.crucible/eval_watch_ckpts/`, and runs each script in the project's `eval_suite:` block on each new ckpt.
`eval_watch_status(recent=10)` — current daemon state + last N evaluation rows.
`eval_watch_stop()` — halt the daemon.

Each project YAML can declare an `eval_suite:` block listing scripts to run on every checkpoint. Each script must accept `--checkpoint <path>` and `--out <json_path>`. Results land in `.crucible/eval_watch.jsonl` (append-only, SHA-deduplicated). State persists across Crucible restarts. See `docs/eval-watcher.md` for usage, schema, and debugging.

**Important**: `bootstrap_nodes`, `dispatch_experiments`, `collect_results`, and `sync_code` are long-running operations (minutes). The MCP server runs them in background threads via `asyncio.to_thread()` to prevent stdio pipe timeouts.

**Tool descriptions**: All tools include REQUIRES/RETURNS/NEXT sections to guide autonomous agents on preconditions, return shapes, and workflow sequencing.

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

### Modality Extensibility

Crucible supports training **any model type** (diffusion, world models, vision, RL, etc.) via the generic training backend. The 4 extension points:

1. **Model**: Inherit `CrucibleModel` (not `TiedEmbeddingLM`), implement `forward(**batch) -> {"loss": tensor, ...}`
2. **Data Adapter**: Implement `DataAdapter.next_batch()`, register with `register_data_adapter()`
3. **Objective** (optional): Implement `TrainingObjective.compute()` or compute loss inside the model
4. **Config**: Set `metrics.primary` + `metrics.direction` in `crucible.yaml`

Built-in adapters: `token` (LM), `image_folder` (torchvision), `synthetic_images`, `synthetic_video` (bouncing balls).
Built-in objectives: `cross_entropy`, `mse`, `kl_divergence`, `composite`, `diffusion`, `jepa`.

Working examples in `examples/diffusion/` (DDPM on MNIST) and `examples/world_model/` (JEPA on bouncing balls).
Full guide: `docs/modality-guide.md`.

### Unified Plugin System

All extension points use `PluginRegistry` from `core/plugin_registry.py` with 3-tier precedence:
- **Builtin** (lowest): shipped with Crucible core
- **Global**: `~/.crucible-hub/plugins/{type}/*.py`
- **Local** (highest): `.crucible/plugins/{type}/*.py`

| Plugin Type | Registry Module | Builtins |
|-------------|-----------------|----------|
| Optimizers | `training/optimizers.py` | adam, adamw, muon, sgd, rmsprop |
| Schedulers | `training/schedulers.py` | cosine, constant, linear, cosine_restarts |
| Fleet Providers | `fleet/provider_registry.py` | runpod, ssh |
| Loggers | `runner/loggers.py` | wandb, console, jsonl |
| Callbacks | `training/callbacks.py` | grad_clip, nan_detector, early_stopping |
| Data Adapters | `training/data_adapters.py` | token, image_folder, synthetic_images, synthetic_video |
| Objectives | `training/objectives.py` | cross_entropy, mse, kl_divergence, composite, diffusion, jepa |
| Architectures | `models/registry.py` | baseline, looped, convloop, prefix_memory |
| Block Types | `models/composer.py` | attention_block, prefix_memory_block |
| Stack Patterns | `models/composer.py` | sequential, encoder_decoder_skip, looped, prefix_memory_stack |
| Augmentations | `models/composer.py` | smear_gate, bigram_hash, trigram_hash |
| Activations | `models/components/mlp.py` | relu_sq, gelu_sq, mish_sq, etc. |
| Data Sources | `core/data_sources.py` | huggingface, local_files, wandb_artifact |
| Domain Specs | `researcher/domain_spec.py` | nlp_classification, agent_scaffold (tap) — YAML contracts for harness optimization |

**Env vars** select plugins at runtime: `OPTIMIZER=lion`, `LR_SCHEDULE=cosine`, `EMBED_OPTIMIZER=adam`, `MATRIX_OPTIMIZER=muon`, `SCALAR_OPTIMIZER=adamw`, `LOGGING_BACKEND=wandb,console`, `CALLBACKS=grad_clip,nan_detector`.

### Community Taps

Taps are git repos that serve as community plugin registries. Managed by `TapManager` in `core/tap.py`.

```bash
crucible tap add https://github.com/user/crucible-community-tap
crucible tap search lion
crucible tap install lion        # copies to ~/.crucible-hub/plugins/optimizers/
crucible tap publish my_opt --type optimizers --tap my-tap
crucible tap push my-tap
crucible tap submit-pr my-tap    # opens PR via gh CLI
```

**Hub layout:**
- `~/.crucible-hub/taps.yaml` — configured taps
- `~/.crucible-hub/installed.yaml` — installed packages ledger
- `~/.crucible-hub/taps/{name}/` — cloned tap repos (shallow)
- `~/.crucible-hub/plugins/{type}/` — installed plugins (auto-discovered)

**Tap repo layout:**
```
optimizers/lion/
  plugin.yaml    # name, type, version, description, author, tags, benchmarks
  lion.py        # the plugin code
```

**Security:** Plugin names validated against `[a-zA-Z0-9][a-zA-Z0-9_-]*`. Symlink escape guards on all file operations. Atomic YAML writes (write-then-rename).

### Known Limitations

- Only RunPod provider is fully tested (SSH provider is pass-through for manual hosts)
- `train_gpt.py` is a compatibility wrapper — actual training is in `src/crucible/training/`
- Hub features require explicit initialization (`crucible hub init`)
- W&B integration requires `wandb` package and `WANDB_API_KEY`

## What NOT to do
- Don't add new architectures to core — build plugins in `.crucible/architectures/`
- Don't add new data source implementations to core — build plugins in `.crucible/plugins/data_sources/`
- Don't build a full experiment tracking UI — the TUI and REST API cover agent/developer needs; use W&B/MLflow for dashboards
- Don't build Kubernetes support — use SkyPilot when ready
- Don't reinvent HPO math — integrate Optuna/Ax
- Don't hardcode paths — derive from ProjectConfig
- Don't catch and swallow errors — let them propagate with context
