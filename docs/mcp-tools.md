---
layout: default
title: MCP Tools Reference
---

# MCP Tools Reference

Crucible exposes 53 MCP tools for AI agents. Start the server:
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

## Fleet Management (4 tools)

| Tool | Description |
|------|-------------|
| `get_fleet_status` | Node inventory, health, assignments |
| `provision_nodes` | Create N new compute nodes |
| `destroy_nodes` | Tear down tracked nodes |
| `sync_code` | Push local code to nodes via rsync |

## Experiment Queue (3 tools)

| Tool | Description |
|------|-------------|
| `get_queue_status` | Queue state: queued, running, completed |
| `enqueue_experiment` | Add one experiment to queue |
| `get_experiment_result` | Get result for a specific run_id |

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

## Model Extensibility (8 tools)

| Tool | Description |
|------|-------------|
| `model_list_families` | List all registered model families (built-in + plugins) |
| `model_list_activations` | List available activation functions |
| `model_list_components` | List model components with descriptions |
| `model_get_config_schema` | Get parameter schema for a model family |
| `model_validate_config` | Validate experiment config against family schema |
| `model_add_architecture` | Save and register a user architecture plugin |
| `model_add_activation` | Register a custom activation function |
| `model_generate_template` | Generate boilerplate for a new architecture plugin |

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
