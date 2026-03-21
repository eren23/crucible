---
layout: default
title: MCP Tools Reference
---

# MCP Tools Reference

Crucible exposes 26 MCP tools for AI agents. Start the server:
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
