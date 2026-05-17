# Crucible

> **Alpha software.** APIs will change. Bug reports and PRs welcome.

**Crucible is the open research operating system for autonomous ML discovery on commodity GPUs** — where hypothesis synthesis, fleet orchestration, and judge-separated loops compose into one closed loop.

Short version: **for labs that can't afford DeepMind's compute but want Sakana's autonomy.**

The defensible niche is the seven-way intersection: **autonomous + reproducible + open + commodity-GPU-native + multi-pod fleet + judge-separated + cross-project memory + plugin-extensible**. Each individual property has a competitor — Sakana on autonomy, autoresearch on simplicity, SkyPilot on multi-cloud, Optuna on HPO math, W&B on tracking. None at the seven-way intersection. See [`docs/positioning.md`](docs/positioning.md) for the full landscape and what Crucible explicitly is NOT.

You bring a training script and a rented GPU. Crucible owns the loop — hypothesize, dispatch across a fleet, collect, reflect, synthesize across findings, repeat. Model-agnostic (Claude, GPT, Gemini, Llama via your orchestrator), modality-agnostic (LM, diffusion, vision, world models, custom), vendor-agnostic (RunPod today, SSH anywhere, SkyPilot coming).

## Why Crucible?

The AI-native discovery engine frontier is shifting from copilots to systems that run closed loops on their own. Today's options each cover one slice:

- **Sakana AI Scientist** — closed-loop and peer-review-validated, but single-machine and paper-shaped, no fleet, no plugin architecture.
- **AlphaEvolve / FunSearch** (DeepMind) — algorithm-discovery via evolution, but paper-only releases, monolithic Gemini, no orchestration story.
- **FutureHouse Platform** — open Aviary framework + biomedical SaaS, but copilot-shaped (human in the loop) and domain-locked.
- **DeepMind AI Co-Scientist / OpenAI Deep Research / Anthropic Managed Agents** — frontier-vendor SaaS, closed weights, no on-your-GPU reproducibility.
- **SkyPilot / Modal / Anyscale** — fleet orchestration, but no hypothesis loop.
- **Optuna / Ax** — mathematical HPO, but no LLM-driven exploration or cross-experiment synthesis.
- **W&B / MLflow** — tracking, not autonomous execution.

Crucible is the only stack that connects all of these concerns: **hypothesize → batch design → provision → dispatch → collect → reflect → synthesize across findings → promote or kill**, with judge separation enforced at the contract layer and full reproducibility from fleet logs to configs to model weights.

See [`docs/positioning.md`](docs/positioning.md) for the full landscape map and what Crucible explicitly is NOT.

## Quick Start

**New here?** The [5-minute quickstart](docs/quickstart-5-minutes.md) takes you from `git clone` to a leaderboard + tool-router next-action recommendation in five commands, no GPU required.

For the production lifecycle (fleet provisioning, presets beyond smoke, autonomous loops, W&B), the full path:

```bash
# Install from source
pip install -e ".[all]"

# Initialize a project from a template (lm, vision, diffusion, world_model, generic)
crucible init
crucible project new my-first-project --template diffusion \
    --set REPO_URL=https://github.com/me/my-first-project

# Review the generated spec, then run a smoke test locally
crucible run experiment --preset smoke

# Or go distributed: provision pods, run a batch, collect results
crucible fleet provision --count 2
crucible fleet bootstrap
crucible run enqueue --spec experiments.json
crucible run dispatch
crucible run collect
crucible analyze rank --top 10
```

See [`docs/getting-started.md`](docs/getting-started.md) for the full walkthrough and [`examples/`](examples/) for working projects in several modalities.

## Modalities supported out of the box

Crucible ships reference examples for every major modality:

| Modality | Example | Highlights |
|---|---|---|
| Language modeling | `examples/parameter_golf/` | Tied-embedding LM with SmearGate / BigramHash / RoPE |
| Diffusion | `examples/diffusion/` | DDPM UNet on MNIST, custom data adapter |
| World models | `examples/world_model/` | JEPA-style latent world model on bouncing balls |
| Vision / classification | `crucible project new vision-test --template vision` | torchvision + CIFAR-10 starting point |
| Bring-your-own-trainer | `examples/huggingface_finetune/` | HuggingFace Trainer wrapper (any 🤗 model) |

For anything else, start from `--template generic` and override.

## What works today

- **Fleet orchestration** — RunPod and generic SSH provider, transactional provisioning with orphan recovery
- **Project templates** — `crucible project new --template <modality>` generates a spec, no copy-paste editing
- **Reliable bootstrap** — per-step state tracking, SSH timeout with exponential backoff, config-driven data probe
- **Experiment execution** — live output parsing, OOM retry, tier presets, per-backend timeout maps
- **Orchestrator-contract researcher** — hypothesis → batch → execute → reflect via `research_request_prompt` / `research_submit`. No LLM keys baked into Crucible — bring your own orchestrator (Claude Code, GPT, Gemini, smolagents)
- **GIANTS-style cross-finding synthesis** — `design_synthesize_from_findings` mines hub findings across projects/tracks and emits orchestrator-shaped prompts with `parent_finding_ids` provenance
- **Judge-separation contract** — reward judge ≠ eval judge in different model families, enforced before pod time burns ([`docs/judge-separation.md`](docs/judge-separation.md))
- **Tree search + GRPO** — UCB1 / greedy / agent-directed / GRPO selection policies, multi-metric Pareto frontiers
- **Harness optimizer** — meta-harness evolutionary loop for memory systems / agent scaffolds with N-D Pareto tracking
- **Auto-eval daemon** — `eval_watch_start` polls running pods, SCPs new checkpoints, runs your eval suite on each, SHA-deduplicated
- **Model zoo** — RMSNorm, RoPE, GQA, SmearGate, BigramHash, MoE, declarative YAML composition
- **Analysis** — leaderboard, sensitivity analysis, Pareto frontier
- **Experiment notes** and **research tracks** persisted under `.crucible/`
- **Crucible Hub** (`~/.crucible-hub/`) — cross-project knowledge sharing, git-synced findings with confidence-gated promotion
- **HuggingFace collab** — publish leaderboards, findings, recipes, model artifacts to HF; read peer-agent prior attempts and discussions
- **REST API** (`crucible serve`) and **MCP server** (`crucible mcp serve`) exposing 200+ tools
- **Community taps** — Homebrew-style git-based plugin sharing across 15 plugin types
- **Unified plugin system** — 13 pluggable types (optimizers, schedulers, callbacks, loggers, providers, architectures, data adapters, objectives, block types, stack patterns, augmentations, activations, data sources) with 3-tier precedence and auto-discovery
- **Interactive TUI** for browsing experiment designs grouped by status

## What's coming

A five-phase plan toward closed-loop autonomous discovery. See [`ROADMAP.md`](ROADMAP.md) for the full breakdown.

- **Phase 1 — Close the autonomy loop**: `autonomous_research_loop` / `tree_autonomous_loop` / `harness_autonomous_loop` MCP tools so one call drives N iterations end-to-end
- **Phase 2 — Discoverability surge**: tool router for the 200+ MCP surface, live TUI cockpit (fleet + queue + leaderboard + briefing), 5-minute quickstart, semantic experiment search
- **Phase 3 — Ecosystem connections**: benchmark ingestion (lm-eval-harness, BIG-bench, papers-with-code), Optuna/Ax bridge, external MCP consumption, code-level mutation MVP, arxiv/OpenReview ingestion
- **Phase 4 — Niche showcase**: paper-draft generator, memory-aware GIANTS synthesis, real-time peer coordination via shared HF Discussions, end-to-end demo project
- **Always coming**: SkyPilot provider, PyPI release

## Core concepts

### Two kinds of config

Crucible has two config levels, and it's worth knowing the difference:

- **`crucible.yaml`** at your repo root — project-wide settings: provider, presets, researcher budget, metrics, sync excludes. Generated by `crucible init`.
- **`.crucible/projects/<name>.yaml`** — a *project spec* for running external code on fleet pods. Each spec points at a git repo, declares a training command, and sets env vars. Generated by `crucible project new`.

For local-only development you just need `crucible.yaml`. For fleet runs you also need a project spec.

### Training contract

Crucible doesn't own your training code. Any script that reads env vars and prints parseable output works:

**Input** (env vars):
- `ITERATIONS`, `MAX_WALLCLOCK_SECONDS`, `TRAIN_BATCH_TOKENS`
- `MODEL_FAMILY`, `NUM_LAYERS`, `MODEL_DIM`, etc. (depends on your architecture)
- `RUN_ID`, `RUN_BACKEND`, `RUN_PRESET`, `CRUCIBLE_VARIANT_NAME`

**Output** (stdout patterns):
- `step:{step}/{total} train_loss:{loss}`
- `step:{step}/{total} val_loss:{loss} val_bpb:{bpb}`
- `Serialized model ... {N} bytes`

### Experiment tiers

Experiments earn their way to expensive compute:

| Tier | Duration | Use case |
|---|---|---|
| `smoke` | ~1 min | Quick validation |
| `screen` | ~1 hr | Directional signal, architecture screening |
| `proxy` | ~30 min | Medium-confidence comparison |
| `medium` | ~1 hr | Thorough comparison |
| `promotion` | ~2 hrs | Best candidates |

### Fleet lifecycle

```bash
crucible fleet provision --count 4      # create pods (transactional: partial failures don't orphan)
crucible fleet bootstrap                # sync code + verify env, with per-step state tracking
crucible fleet status                   # show nodes, step-by-step bootstrap progress
crucible run enqueue --spec batch.json  # queue experiments
crucible run dispatch                   # assign queue → idle nodes
crucible run collect                    # rsync results and merge
crucible fleet destroy                  # tear down when done
```

When things go wrong, `crucible fleet status` shows per-node, per-step bootstrap results; the MCP `cleanup_orphans` tool finds and destroys pods that aren't in local inventory.

### Autonomous researcher

Claude-powered: analyze results, generate hypotheses, design batches, execute, reflect, promote or kill.

```bash
crucible research start --budget-hours 10 --tier proxy --dry-run
```

### MCP + REST

```bash
crucible mcp serve     # stdio MCP server for Claude (200+ tools)
crucible serve         # REST API server (FastAPI)
```

## Installation

```bash
pip install crucible-ml[all]         # everything
pip install crucible-ml              # minimal (orchestration only)
pip install crucible-ml[torch]       # + model zoo
pip install crucible-ml[anthropic]   # + autonomous researcher
pip install crucible-ml[mcp]         # + MCP server
pip install crucible-ml[wandb]       # + W&B integration
pip install crucible-ml[tui]         # + Interactive TUI
```

## Project structure

```
src/crucible/
├── core/          # Config, env, I/O, types, logging, errors, plugin registry — no cross-module deps
├── fleet/         # Provider-abstracted fleet management (RunPod, SSH)
│   └── providers/ # Compute backends
├── runner/        # Experiment execution, output parsing, presets, tracking, notes
├── training/      # Training backends (torch, generic) — modality-agnostic
├── models/        # Model zoo — components, architectures, declarative composer
│   ├── components/     # Reusable blocks (Attention, MLP, MoE, RMSNorm, etc.)
│   ├── architectures/  # Built-in reference architectures
│   ├── specs/          # YAML architecture specs (declarative)
│   └── composer.py     # Declarative architecture composition engine
├── researcher/    # LLM-driven autonomous research loop
├── analysis/      # Leaderboard, sensitivity analysis, Pareto frontier
├── data/          # Manifest-driven data pipeline
├── mcp/           # MCP server exposing fleet ops as Claude tools
├── api/           # Lightweight REST API server (FastAPI)
├── tui/           # Interactive experiment design browser (Textual)
├── templates/     # Built-in project-spec templates shipped with the package
└── cli/           # CLI entry points
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Highest-impact areas:
- **Compute providers** — Modal, Lambda, SkyPilot backends
- **Search strategies** — Optuna, Ax integration
- **Training examples** — show Crucible working with your framework
- **Bug reports** — file issues, we'll fix them

**Publishing a plugin?** `crucible tap init ~/my-tap` scaffolds a born-clean tap (README, LICENSE, tap.yaml, CI workflow, example plugin); `crucible tap lint .` enforces 11 quality checks before publish. Full guide: [`docs/community-plugins.md`](docs/community-plugins.md); recipe: [`docs/recipes/publish-first-plugin.yaml`](docs/recipes/publish-first-plugin.yaml).

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full plan — what works, what's next, what we won't build, and honest competitive assessment.

## Origin

Crucible was born from [OpenAI Parameter Golf](https://github.com/openai/parameter-golf) (March–April 2026), a competition to train the best 16MB language model on 8×H100s in 10 minutes. The autonomous research infrastructure we built for the competition turned out to be general-purpose, so we extracted and generalized it. The Parameter Golf project config lives at the repo root (`crucible.yaml`) as a working reference; `crucible.yaml.example` is the modality-agnostic template for new projects.

## License

MIT
