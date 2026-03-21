---
layout: default
title: Roadmap
---

# Roadmap

## Phase 1: Foundation (Done)

### Version Store
- Hybrid persistence: YAML files + JSONL ledger
- Full CRUD: create, get, list, history, diff
- Git integration (optional auto-commit)
- Checksum tracking for integrity

### MCP Tools (26 total)
- Fleet management (4): provision, destroy, sync, status
- Experiment queue (3): enqueue, status, results
- Analysis (2): leaderboard, sensitivity
- Design tools (5): browse, compare, generate hypotheses, batch design, enqueue batch
- Context tools (3): analysis, push finding, get findings
- Version tools (6): save, list, diff, get, run, link
- Config tools (2): presets, project config

### Interactive TUI
- Split-pane design browser with Textual
- Status color coding (draft/ready/running/completed/archived)
- Diff mode for comparing designs
- Research context and version history views
- Screenshot export (SVG)

### Model Zoo
- 4 transformer architectures
- 9 configurable activation functions
- 15+ env var knobs per model

---

## Phase 2: Agent Collaboration (Planned)

### Orchestration Tools
- `research_run_iteration` — one-shot full research cycle
- `research_collect_and_reflect` — collect results + LLM reflection

### Code Change Proposals
- `proposal_create` — structured code change proposals with diffs
- `proposal_list` / `proposal_get` — review and manage proposals
- Proposals are versioned but don't auto-apply (require review)

### Agent Session Management
- `agent_register` — announce agent presence + focus area
- `agent_list_sessions` — see active agents
- `agent_claim_work` — prevent duplicate effort
- Heartbeat-based liveness (30min timeout)

---

## Phase 3: Multi-Agent Platform (Future)

### Concurrent Agent Support
- Advisory locking for work claims
- Agent identity tracking across sessions
- Conflict resolution for overlapping designs

### Real-Time Fleet Dashboard
- Live TUI with fleet node status
- Running experiment progress bars
- Queue visualization with priority lanes

### Advanced Analysis
- Automated Pareto frontier tracking
- Diminishing returns alerts
- Cross-family comparison reports
- W&B integration for detailed metrics

### Infrastructure
- SkyPilot integration for multi-cloud
- Optuna/Ax integration for HPO
- Experiment replay from manifests
