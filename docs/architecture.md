---
layout: default
title: Architecture
---

# Architecture

## Module Structure

```
src/crucible/
  core/           Config, I/O, types, logging, version store, finding, hub
  fleet/          Provider-abstracted fleet management
    providers/    RunPod, SSH backends
  runner/         Experiment execution, output parsing, presets, tracking, notes
  models/         Model zoo — PyTorch transformer components
    architectures/  baseline, looped, convloop, prefix_memory
    components/     attention, MLP, normalization, embeddings
  researcher/     LLM-driven autonomous research loop, briefing
  analysis/       Leaderboard, sensitivity, Pareto frontier
  data/           Manifest-driven HuggingFace data pipeline
  mcp/            MCP server (41 tools)
  api/            REST API server (FastAPI, 10 endpoints)
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
  HubError          Hub sync, track, finding promotion failures
  ApiError          REST API server failures
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

## Hub System

The Crucible Hub (`~/.crucible-hub/`) is a cross-project knowledge store designed for LLM-driven research workflows.

```
~/.crucible-hub/
  config.yaml           Hub configuration (active track, remote URL)
  tracks/
    {track-name}/
      meta.yaml         Track metadata (description, created date)
      findings.jsonl    Promoted findings for this track
  projects/
    {project-name}/     Symlink or metadata pointing to project location
  index.jsonl           Global finding index for fast queries
```

**Key concepts:**
- **Tracks** are named research directions (e.g., "attention-variants", "scaling-laws") that group related findings across projects.
- **Finding promotion** elevates a project-level insight (`context_push_finding`) to the hub (`finding_promote`) where it becomes visible to all projects on the same track.
- **Git sync** (`hub_sync`) pushes/pulls the hub as a git repository, enabling sharing across machines and collaborators.
- Hub data is read by the briefing system to orient new LLM sessions with cross-project knowledge.

## Notes System

Experiment notes provide freeform annotation attached to run IDs. Stored under `.crucible/notes/` as markdown files with YAML frontmatter.

```
.crucible/notes/
  {run_id}/
    001.md              First note (YAML frontmatter + markdown body)
    002.md              Second note
    ...
```

Each note file:
```yaml
---
run_id: exp-20260322-001
timestamp: 2026-03-22T14:30:00Z
tags: [lr-schedule, plateau]
---
## Observation
Loss plateaued after step 4000. Suspect learning rate is too high.
```

Notes are searchable via `note_search` (full-text + tag filtering) and retrievable via `note_get`. The MCP tool `note_add` creates notes; the CLI command `crucible note add` provides the same capability.

## API Server

The REST API (`crucible serve`) is a thin FastAPI layer over the MCP tools, intended for dashboards, CI integration, and non-MCP clients.

```
GET  /api/fleet/status              → get_fleet_status
POST /api/fleet/provision           → provision_nodes
DELETE /api/fleet/destroy           → destroy_nodes
GET  /api/experiments/queue         → get_queue_status
POST /api/experiments/enqueue       → enqueue_experiment
GET  /api/experiments/{run_id}      → get_experiment_result
GET  /api/analysis/leaderboard      → get_leaderboard
GET  /api/analysis/sensitivity      → get_sensitivity
GET  /api/research/state            → get_research_state
GET  /api/research/briefing         → get_research_briefing
```

The API server shares no state with the MCP server — both read from the same `.crucible/` directory and JSONL files. This means they can run concurrently without conflicts.

## Research Tracks

Research tracks provide organizational structure above individual projects:

```
Track: "attention-variants"
  ├── Project A (GQA experiments)
  ├── Project B (linear attention)
  └── Project C (sliding window)
```

- One track is "active" at a time (set via `track_switch`)
- New findings are filed under the active track
- `track_list` shows all tracks with finding counts and last activity
- Tracks live in the hub, so they persist across projects and machines
