---
layout: default
title: MCP Tools Reference
---

# MCP Tools Reference

Crucible exposes 112 MCP tools for AI agents. Start the server:
```bash
crucible mcp serve
```

Configure in Claude Desktop or any MCP client:
```json
{
  "crucible": {
    "command": "crucible",
    "args": ["mcp", "serve"]
  }
}
```

---

## Fleet Management (6 tools)

| Tool | Description |
|------|-------------|
| `get_fleet_status` | Node inventory, health, assignments |
| `provision_nodes` | Create N new compute nodes |
| `destroy_nodes` | Tear down tracked nodes |
| `sync_code` | Push local code to nodes via rsync |
| `fleet_refresh` | Refresh node states from cloud provider API |
| `bootstrap_nodes` | Sync code, install deps, download data on nodes |

## Experiment Queue (5 tools)

| Tool | Description |
|------|-------------|
| `get_queue_status` | Queue state: queued, running, completed |
| `enqueue_experiment` | Add one experiment to queue |
| `get_experiment_result` | Get result for a specific run_id |
| `cancel_experiment` | Cancel queued/running experiments by name, run_id, or wave |
| `clear_stale_queue` | Mark experiments failed if assigned to non-existent nodes |

## Analysis (2 tools)

| Tool | Description |
|------|-------------|
| `get_leaderboard` | Top N results sorted by primary metric |
| `get_sensitivity` | Parameter sensitivity analysis |

## Research State (1 tool)

| Tool | Description |
|------|-------------|
| `get_research_state` | Hypotheses, beliefs, budget remaining |

## Experiment Design (5 tools)

| Tool | Description |
|------|-------------|
| `design_browse_experiments` | Browse completed experiments with filters (name, family, tag, metric range, config) |
| `design_compare_experiments` | Side-by-side comparison of 2-5 experiments |
| `design_generate_hypotheses` | LLM-driven hypothesis generation with agent context |
| `design_batch_from_hypotheses` | Convert hypotheses to executable batch |
| `design_enqueue_batch` | Enqueue a full batch to fleet |

## Research Context (3 tools)

| Tool | Description |
|------|-------------|
| `context_get_analysis` | Full structured analysis: leaderboard, sensitivity, beliefs |
| `context_push_finding` | Record a research finding |
| `context_get_findings` | Query accumulated findings |

## Version Management (6 tools)

| Tool | Description |
|------|-------------|
| `version_save_design` | Create/update versioned experiment design (supports partial updates) |
| `version_list_designs` | List all designs with metadata |
| `version_diff` | Compare two versions of a design |
| `version_get_design` | Get full content + metadata |
| `version_run_design` | Execute a design: convert, enqueue, update status |
| `version_link_result` | Link a run_id back to a design |

## Configuration (2 tools)

| Tool | Description |
|------|-------------|
| `config_get_presets` | All presets with resolved config values |
| `config_get_project` | Full project configuration |

## Notes (3 tools)

| Tool | Description |
|------|-------------|
| `note_add` | Attach a markdown note to a run |
| `note_get` | Retrieve all notes for a run |
| `note_search` | Full-text search across all notes |

### note_add

Attach a freeform observation to a completed experiment run.

**Parameters:**
- `run_id` (string, required) — The experiment run ID
- `content` (string, required) — Markdown note content
- `tags` (list[string], optional) — Categorization tags

**Example:**
```json
{
  "run_id": "exp-20260322-001",
  "content": "## Observation\nLoss plateaued after step 4000. Suspect learning rate is too high for this architecture.",
  "tags": ["lr-schedule", "plateau"]
}
```

### note_get

Retrieve all notes attached to a specific run.

**Parameters:**
- `run_id` (string, required) — The experiment run ID

**Example:**
```json
{ "run_id": "exp-20260322-001" }
```

### note_search

Search across all notes by keyword or tag.

**Parameters:**
- `query` (string, optional) — Full-text search query
- `tags` (list[string], optional) — Filter by tags
- `limit` (int, optional) — Max results (default: 20)

**Example:**
```json
{ "query": "learning rate plateau", "tags": ["lr-schedule"], "limit": 10 }
```

## W&B Bridge (3 tools)

| Tool | Description |
|------|-------------|
| `wandb_log_image` | Log an image to a W&B run |
| `wandb_get_url` | Get the W&B dashboard URL for a run |
| `wandb_annotate` | Add annotations to a W&B run |

### wandb_log_image

Log an image file or PIL image to a W&B run for visual tracking.

**Parameters:**
- `run_id` (string, required) — The experiment run ID
- `image_path` (string, required) — Path to the image file
- `caption` (string, optional) — Image caption
- `key` (string, optional) — W&B log key (default: "images")

**Example:**
```json
{
  "run_id": "exp-20260322-001",
  "image_path": "/results/loss_curve.png",
  "caption": "Training loss over 10k steps"
}
```

### wandb_get_url

Get the W&B dashboard URL for a tracked run.

**Parameters:**
- `run_id` (string, required) — The experiment run ID

**Example:**
```json
{ "run_id": "exp-20260322-001" }
```

### wandb_annotate

Add structured annotations (key-value metadata) to a W&B run.

**Parameters:**
- `run_id` (string, required) — The experiment run ID
- `annotations` (object, required) — Key-value pairs to attach
- `tags` (list[string], optional) — W&B tags to add

**Example:**
```json
{
  "run_id": "exp-20260322-001",
  "annotations": { "verdict": "promising", "next_step": "try cosine annealing" },
  "tags": ["reviewed", "lr-experiment"]
}
```

## Hub (2 tools)

| Tool | Description |
|------|-------------|
| `hub_status` | Hub state: active track, synced projects, finding count |
| `hub_sync` | Push/pull hub directory via git |

### hub_status

Get the current state of the Crucible Hub.

**Parameters:** None.

**Returns:** Active track, list of synced projects, total finding count, last sync time.

**Example:**
```json
{}
```

### hub_sync

Synchronize the hub directory with its git remote.

**Parameters:**
- `direction` (string, optional) — "push", "pull", or "both" (default: "both")

**Example:**
```json
{ "direction": "pull" }
```

## Tracks (3 tools)

| Tool | Description |
|------|-------------|
| `track_create` | Create a named research track |
| `track_list` | List all tracks with metadata |
| `track_switch` | Switch the active research track |

### track_create

Create a new research track to group related projects.

**Parameters:**
- `name` (string, required) — Track name (slug-friendly)
- `description` (string, optional) — What this track explores

**Example:**
```json
{ "name": "attention-variants", "description": "Comparing GQA, MHA, and linear attention on proxy tasks" }
```

### track_list

List all research tracks with their metadata and project counts.

**Parameters:** None.

**Example:**
```json
{}
```

### track_switch

Switch the active research track. New findings will be filed under this track.

**Parameters:**
- `name` (string, required) — Track name to activate

**Example:**
```json
{ "name": "attention-variants" }
```

## Findings (2 tools)

| Tool | Description |
|------|-------------|
| `hub_findings_query` | Search findings across all projects in the hub |
| `finding_promote` | Promote a project finding to the hub |

### hub_findings_query

Search findings across all projects and tracks in the hub.

**Parameters:**
- `query` (string, optional) — Full-text search query
- `track` (string, optional) — Filter by track name
- `project` (string, optional) — Filter by source project
- `limit` (int, optional) — Max results (default: 20)

**Example:**
```json
{ "query": "RoPE scaling", "track": "attention-variants", "limit": 5 }
```

### finding_promote

Promote a project-level finding to the Crucible Hub for cross-project visibility.

**Parameters:**
- `finding_id` (string, required) — The local finding ID to promote
- `track` (string, optional) — Target track (defaults to active track)

**Example:**
```json
{ "finding_id": "finding-20260322-003", "track": "attention-variants" }
```

## Briefing (2 tools)

| Tool | Description |
|------|-------------|
| `get_research_briefing` | Generate LLM session orientation summary |
| `annotate_run` | Add structured annotations to a completed run |

### get_research_briefing

Generate a research briefing for a new LLM session. Includes project context, recent results, active hypotheses, accumulated findings, and track state.

**Parameters:**
- `include_hub` (bool, optional) — Include cross-project findings from the hub (default: true)
- `max_findings` (int, optional) — Max findings to include (default: 10)

**Example:**
```json
{ "include_hub": true, "max_findings": 5 }
```

### annotate_run

Add structured annotations to a completed experiment run. Useful for recording LLM analysis of results.

**Parameters:**
- `run_id` (string, required) — The experiment run ID
- `annotations` (object, required) — Key-value annotation data
- `summary` (string, optional) — Brief text summary

**Example:**
```json
{
  "run_id": "exp-20260322-001",
  "annotations": { "quality": "good", "convergence": "fast", "notable": "loss dropped 15% vs baseline" },
  "summary": "Strong result — consider promoting to medium tier"
}
```

## Model Extensibility — Code Plugins (8 tools)

| Tool | Description |
|------|-------------|
| `model_list_families` | List all registered model families (built-in + plugins + specs) |
| `model_list_activations` | List available activation functions |
| `model_list_components` | List model components with descriptions |
| `model_get_config_schema` | Get parameter schema for a model family |
| `model_validate_config` | Validate experiment config against family schema |
| `model_add_architecture` | Save and register a user architecture plugin (Python) |
| `model_add_activation` | Register a custom activation function |
| `model_generate_template` | Generate boilerplate for a new architecture plugin |

## Model Extensibility — Hub Promotion (3 tools)

| Tool | Description |
|------|-------------|
| `model_list_global_architectures` | List architecture plugins in the global hub |
| `model_promote_architecture` | Promote a local plugin to the global hub |
| `model_import_architecture` | Import a hub plugin into the local project |

## Declarative Architecture Composition (6 tools)

Compose architectures from known components via YAML specs — no Python code needed.

| Tool | Description |
|------|-------------|
| `model_compose` | Create architecture from declarative spec (block + stack + augmentations) |
| `model_from_template` | Fork an existing spec-based architecture with overrides |
| `model_list_stack_patterns` | List wiring patterns (sequential, looped, encoder_decoder_skip, etc.) |
| `model_list_block_types` | List block types (attention_block, prefix_memory_block) |
| `model_preview_spec` | Dry-run: instantiate on CPU, return param count + structure |
| `model_get_spec` | Retrieve the YAML spec for a family (null if code-defined) |

## Configuration (2 tools)

| Tool | Description |
|------|-------------|
| `config_get_presets` | All presets with resolved config values |
| `config_get_project` | Full project configuration |

---

## Example Agent Workflow

### Browse, Design, Run

```
1. config_get_project          → understand the project
2. design_browse_experiments   → see what's been tried
3. get_sensitivity             → identify high-impact params
4. version_save_design         → save a new design
5. version_run_design          → execute it
6. get_experiment_result       → check the result
7. version_link_result         → link result to design
8. context_push_finding        → record what you learned
```

### Full Research Cycle

```
1. context_get_analysis            → understand current state
2. design_generate_hypotheses      → generate candidates
3. design_batch_from_hypotheses    → build executable batch
4. design_enqueue_batch            → queue for fleet
5. context_push_finding            → record insights
```

### Session with Briefing and Notes

```
1. get_research_briefing          → orient new session (project + hub context)
2. design_browse_experiments      → see recent results
3. note_search                    → find relevant observations
4. version_save_design            → create new experiment
5. version_run_design             → execute it
6. get_experiment_result          → check result
7. note_add                       → record observations
8. annotate_run                   → add structured analysis
9. finding_promote                → promote insight to hub
10. hub_sync                      → sync hub to remote
```

---

## Agent Assistance (3 tools)

| Tool | Description |
|------|-------------|
| `get_run_logs` | Fetch training stdout/stderr for an experiment (local or SSH) |
| `model_fetch_architecture` | Read source code or YAML spec for any architecture family |
| `get_architecture_guide` | Decision tree for declarative vs Python plugin workflows |

### get_run_logs

Fetch training output logs for debugging experiments. Checks local collected logs first, falls back to SSH.

**Parameters:**
- `run_id` (string, required) — The experiment run_id
- `tail_lines` (int, optional) — Lines from end to return (default: 100, 0 = all)

### model_fetch_architecture

Read the full source code (Python) or spec (YAML) for any registered architecture. Searches local > global > builtin.

**Parameters:**
- `family` (string, required) — Model family name (e.g., "baseline", "looped")

### get_architecture_guide

Returns a decision tree for choosing between declarative composition (`model_compose`) and Python plugins (`model_add_architecture`), with step-by-step workflows for each approach.

---

## Tree Search (8 tools)

Branching experiment exploration. Instead of running all experiments equally, organize them as a tree where promising directions expand and poor ones get pruned.

| Tool | Description |
|------|-------------|
| `tree_create` | Create a search tree with root experiment(s) and selection policies |
| `tree_get` | Get tree structure, ASCII visualization, and best path |
| `tree_expand_node` | Add child experiments to a completed node |
| `tree_auto_expand` | LLM-generate children for a node using Claude |
| `tree_prune` | Prune a node or entire branch |
| `tree_enqueue_pending` | Move pending tree nodes to fleet queue |
| `tree_sync_results` | Match completed results back to tree nodes |
| `tree_list` | List all search trees with summary stats |

### tree_create

**Parameters:**
- `name` (string, required) — Tree name
- `roots` (array, optional) — Root experiment nodes with name, config, hypothesis
- `expansion_policy` (string, optional) — "agent_directed" (default), "ucb1", "greedy", "epsilon_greedy"
- `pruning_policy` (string, optional) — "agent_directed" (default), "threshold"
- `primary_metric` (string, optional) — Metric to optimize (default: val_bpb)
- `metric_direction` (string, optional) — "minimize" or "maximize"

### Workflow

```
1. tree_create         → create tree with baseline root
2. tree_enqueue_pending → move to fleet queue
3. dispatch_experiments → run on nodes
4. collect_results     → download results
5. tree_sync_results   → update tree with metrics
6. tree_get            → view results + ASCII tree
7. tree_expand_node    → add children to best nodes
   (or tree_auto_expand for LLM-generated children)
8. Repeat from step 2
```

---

## Training Generalization (1 tool)

| Tool | Description |
|------|-------------|
| `config_get_modalities` | List training backends with modality tags, data adapters, and objectives |

### config_get_modalities

Returns available training backends (torch, generic), registered data adapters (token), and training objectives (cross_entropy, mse, kl_divergence).

---

## Session Recipes (3 tools)

Recipes capture successful experiment sessions as step-by-step playbooks. Save what you did, what worked, what broke, and how you fixed it. Other agents can follow a recipe to reproduce the session.

| Tool | Description |
|------|-------------|
| `recipe_save` | Save a session recipe with steps, environment, gotchas, and results |
| `recipe_list` | List all saved recipes (optionally filter by tag) |
| `recipe_get` | Get full recipe details for reproduction |

### recipe_save

Save a reproducible session recipe. Validates name (lowercase slug), requires at least one step with a `tool` key.

**Parameters:**
- `name` (string, required) — Slug name, e.g. `yolo11-nano-coco128`
- `title` (string) — Human-readable title
- `goal` (string) — What this session aimed to achieve
- `project_spec` (string) — Name of the project spec used (from `.crucible/projects/`)
- `environment` (object) — Runtime environment: gpu, torch version, python, provider
- `steps` (array, required) — Ordered MCP tool calls, each with `tool`, `args`, `note`
- `results` (object) — Final metrics, W&B URLs, linked run IDs
- `gotchas` (array) — Problems encountered: `{issue, fix}` pairs
- `tags` (array[string]) — Searchable tags
- `created_by` (string) — Who created this (default: `mcp-agent`)

**Example:**
```json
{
  "name": "yolo11-nano-coco128",
  "title": "YOLOv11 Nano fine-tune on COCO128",
  "goal": "Fine-tune YOLOv11n on COCO128 using RunPod RTX 4090",
  "project_spec": "yolo11-demo",
  "environment": {
    "gpu": "NVIDIA GeForce RTX 4090",
    "torch": "2.6.0+cu124",
    "python": "3.11",
    "ultralytics": "8.4.30",
    "provider": "runpod"
  },
  "steps": [
    {"tool": "provision_project", "args": {"project_name": "yolo11-demo", "count": 1}, "note": "Spin up 1x RTX 4090"},
    {"tool": "fleet_refresh", "args": {}, "note": "Wait ~45s then refresh for SSH endpoint"},
    {"tool": "bootstrap_project", "args": {"project_name": "yolo11-demo"}, "note": "Clone repo, install deps"},
    {"tool": "run_project", "args": {"project_name": "yolo11-demo", "overrides": {"MODEL": "yolo11n.pt", "EPOCHS": "100"}}, "note": "Launch training; single-node runs return run_id, multi-node runs return launch_id plus per-node run_ids"},
    {"tool": "get_project_run_status", "args": {"run_id": "<from_run_project>"}, "note": "Probe lifecycle state and recent events while the run is active"},
    {"tool": "collect_project_results", "args": {"run_id": "<from_run_project>"}, "note": "Rsync log, parse metrics, persist terminal status"},
    {"tool": "destroy_nodes", "args": {"node_names": ["yolo11-demo-01"]}, "note": "Kill pod"}
  ],
  "results": {"map50_95_b": 0.733, "precision_b": 0.909},
  "gotchas": [
    {"issue": "Ultralytics pulls torch cu130, incompatible with CUDA 12.8", "fix": "Pin torch 2.6.0+cu124 via install_torch"},
    {"issue": "W&B disabled by default in Ultralytics 8.x", "fix": "Add yolo settings wandb=true before training"}
  ],
  "tags": ["yolo", "detection", "coco128", "4090"]
}
```

### recipe_list

List all saved recipes. Optionally filter by tag.

**Parameters:**
- `tag` (string, optional) — Filter recipes that have this tag

**Example:**
```json
{}
```

```json
{"tag": "yolo"}
```

**Returns:**
```json
{
  "recipes": [
    {"name": "yolo11-nano-coco128", "title": "YOLOv11 Nano fine-tune", "tags": ["yolo", "4090"], "goal": "..."}
  ],
  "total": 1
}
```

### recipe_get

Get the full recipe with all steps, environment, gotchas, and results. Pass the output to an agent to reproduce the session.

**Parameters:**
- `name` (string, required) — Recipe name from `recipe_list`

**Example:**
```json
{"name": "yolo11-nano-coco128"}
```

### Workflow

```
1. Run experiments via MCP (provision → bootstrap → run → collect)
2. recipe_save      → capture what worked + what broke
3. recipe_list      → browse saved recipes
4. recipe_get       → hand to another agent for reproduction
```

---

## Plugin Registry (15 tools)

Unified plugin system for all extensible components. Each plugin type supports `list_available`, `add`, and optionally `get_config_schema`.

| Tool | Description |
|------|-------------|
| `optimizer_list_available` | List registered optimizers (adam, adamw, muon, sgd, rmsprop + custom) |
| `optimizer_add` | Register a new optimizer from Python code |
| `optimizer_get_config_schema` | Get parameter schema for a named optimizer |
| `scheduler_list_available` | List registered LR schedulers (cosine, constant, linear, cosine_restarts + custom) |
| `scheduler_add` | Register a new scheduler from Python code |
| `scheduler_get_config_schema` | Get parameter schema for a named scheduler |
| `provider_list_available` | List registered fleet providers (runpod, ssh + custom) |
| `provider_add` | Register a new fleet provider from Python code |
| `logger_list_available` | List registered logging backends (wandb, console, jsonl + custom) |
| `logger_add` | Register a new logging backend from Python code |
| `callback_list_available` | List registered training callbacks (grad_clip, nan_detector, early_stopping + custom) |
| `callback_add` | Register a new training callback from Python code |
| `composer_add_block_type` | Register a new composer block type |
| `composer_add_stack_pattern` | Register a new composer stack wiring pattern |
| `composer_add_augmentation` | Register a new composer augmentation |

All plugins use 3-tier precedence: builtin < global (`~/.crucible-hub/plugins/`) < local (`.crucible/plugins/`).

---

## Community Taps (12 tools)

Homebrew-style plugin sharing via git repositories.

| Tool | Description |
|------|-------------|
| `hub_tap_add` | Add a tap (clone a git repo containing plugins) |
| `hub_tap_remove` | Remove a tap and its cloned repo |
| `hub_tap_list` | List all configured taps |
| `hub_tap_sync` | Pull latest from one or all taps |
| `hub_search` | Search for plugins across all taps |
| `hub_install` | Install a plugin from a tap into the hub |
| `hub_uninstall` | Remove an installed tap plugin |
| `hub_installed` | List all installed tap plugins |
| `hub_publish` | Publish a local plugin to a tap repo |
| `hub_tap_push` | Push a tap repo to its remote |
| `hub_submit_pr` | Open a PR from a tap fork (via gh CLI) |
| `hub_package_info` | Get detailed package metadata and install status |

### Workflow

```
1. hub_tap_add       → clone a community tap repo
2. hub_search        → find plugins by name/description/tag
3. hub_install       → copies .py to ~/.crucible-hub/plugins/ (auto-discovered)
4. hub_publish       → copy local plugin to tap, commit
5. hub_tap_push      → push tap to remote
6. hub_submit_pr     → open PR to upstream (if fork)
```
