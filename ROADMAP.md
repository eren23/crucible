# Roadmap

## The positioning anchor

**Crucible is the open research operating system for autonomous ML discovery on commodity GPUs — where hypothesis synthesis, fleet orchestration, and judge-separated loops compose into one closed loop.**

Every roadmap item below is scored against that statement. If a proposed feature drifts into wet-lab, frontier-model-scale, dev-engineering agent, or HPO math reinvention, it's not on this roadmap — see [`docs/positioning.md`](docs/positioning.md) for the exclusions.

## Where Crucible Sits Today

Crucible works. We've validated the full loop — provision RunPod pods, bootstrap, dispatch experiments, monitor, collect results, destroy. Training losses drop, W&B logs, experiments complete. Hub-based cross-project knowledge sharing, GIANTS-style synthesis, judge-separated LM-as-judge loops, the harness optimizer, the eval-watcher daemon, and a 200+-tool MCP server are all functional.

But it's alpha software. Here's an honest assessment and what comes next.

## What Actually Works

| Feature | Status | Notes |
|---------|--------|-------|
| RunPod fleet provisioning | Working | Multi-pod runs, project-tagged isolation, transactional orphan recovery |
| SSH provider | Structural | Not battle-tested |
| Bootstrap (sync + install + data) | Working | Per-step state tracking, exponential backoff |
| Experiment dispatch via SSH | Working | Detached subprocess on pods |
| Result collection via rsync | Working | Poll-based |
| W&B integration | Working | Logging, annotation, image support; `wandb.required=true` enqueue + runtime contract |
| CLI (`crucible` command) | Working | All subcommands wired |
| Orchestrator-contract researcher | Working | `research_request_prompt` / `research_submit` — no LLM keys in Crucible |
| Legacy `AutonomousResearcher` | Working | Closed-loop Python API (opt-in, requires `ANTHROPIC_API_KEY`) |
| GIANTS-style finding synthesis | Working | `design_synthesize_from_findings` mines hub finding pairs with provenance |
| Judge-separation contract | Working | Reward/eval/audit judges; enforced before pod time burns |
| Tree search (UCB1 / greedy / epsilon-greedy / agent-directed / GRPO) | Working | Multi-policy node expansion, N-D Pareto frontiers |
| Harness optimizer | Working | Meta-harness evolutionary loop with N-D Pareto |
| Eval-watcher daemon | Working | Auto-evals running pods, SHA-deduplicated, persists across restarts |
| Model zoo + plugin system | Working | 4 built-in + plugin auto-discovery, declarative composer |
| MCP server (200+ tools) | Working | Fleet, design, research, hub, models, taps, harness, eval-watch |
| REST API (10 endpoints) | Working | FastAPI thin wrapper |
| Hub (cross-project knowledge) | Working | 3-tier finding promotion, git-synced |
| Notes system | Working | Markdown + YAML frontmatter, HF Discussion crosspost |
| Research briefing | Working | LLM session orientation, optional HF prior-attempts pull |
| HuggingFace collab (Tier 14 + 15) | Working | Publish leaderboards/findings/recipes/artifacts; pull peer prior attempts and discussions |
| Output parser | Working | Regex patterns validated |
| Configurable metrics | Working | No hardcoded assumptions |
| Interactive TUI | Partial | Design browser + diff + history; no live fleet/queue/leaderboard yet |
| Test suite | Partial | Core/runner/analysis/architect/plugins covered; fleet/synthesis/code-search gaps |
| Community taps | Working | Homebrew-style plugin sharing across 15 plugin types |

## What's Not a Moat (Honest)

Before building more, we acknowledge what others do better:

- **Fleet orchestration**: [SkyPilot](https://github.com/skypilot-org/skypilot) supports 20+ clouds with cost optimization and spot failover. Our RunPod + SSH is functional but not competitive at the infra layer.
- **HPO math**: [Optuna](https://github.com/optuna/optuna) and [Ax](https://github.com/facebook/Ax) have mathematically superior search (TPE, Bayesian, CMA-ES). Our LLM-driven search is different, not necessarily better at parameter optimization.
- **Experiment tracking UI**: [W&B](https://wandb.ai) and [MLflow](https://mlflow.org) exist. We don't build dashboards.
- **Model zoo**: [HuggingFace Transformers](https://github.com/huggingface/transformers) exists. Our components are from one competition.
- **Paper generation**: Sakana AI Scientist already has a Nature 2026 peer-reviewed paper. We'll ship a draft generator (Phase 4), but writing isn't the moat — the research loop is.

## What IS Unique

1. **Full closed-loop autonomy on rental GPUs** — No other tool goes from API key → autonomous research loop on RunPod/SSH.
2. **MCP-first agent-native architecture** — Model Context Protocol is the native orchestration interface. Any LLM the orchestrator picks can drive Crucible; no LLM keys baked in.
3. **GIANTS-style cross-finding synthesis** — `design_synthesize_from_findings` pair-mines hub findings across projects/tracks; provenance carried through batches and W&B tags. No competitor has this.
4. **Judge-separation contract** — reward judge ≠ eval judge in different model families, enforced at the contract layer before pod time burns. Hard-coded into the API surface.
5. **Tiered experiment promotion** — Experiments earn expensive compute (smoke → screen → proxy → medium → promotion).
6. **Plugin registry with 3-tier precedence** — 15 extension points unified under one contract. Taps make sharing Homebrew-easy.
7. **Harness optimizer** — meta-harness evolutionary loop for agent scaffolds / memory systems with N-D Pareto frontier tracking.
8. **Auto-eval daemon** — polls running pods, SCPs new checkpoints, runs your eval suite on each, SHA-deduplicated, persists across restarts.
9. **HuggingFace collab as peer-research surface** — publish leaderboards/findings/recipes/artifacts; pull peer agents' prior attempts and discussions to inform your hypothesis stage.
10. **Cross-project knowledge hub** — git-synced findings with confidence-gated promotion (project → track → global).

## The Five-Phase Plan

Full plan + audit at `/Users/eren/.claude/plans/ai-native-discovery-engines-xuster-virtual-hare.md`. Realistic timeline: 10-13 weeks.

### Phase 0: Strategic Anchoring (DONE — `b284723`)

Lock the positioning statement in the codebase, prune scope.

- [x] Positioning statement drafted (May 2026 audit)
- [x] README.md leads with the niche, not the feature list
- [x] CLAUDE.md "What is this?" updated
- [x] `docs/positioning.md` 1-pager (landscape, what we are NOT, scope tests)
- [x] ROADMAP.md refreshed (this file)
- [x] `docs/index.md` aligned (Phase 2.7 learning-path reorg)

**Exit:** Positioning statement appears verbatim in README, CLAUDE.md, docs/positioning.md.

### Phase 1: Close the Autonomy Loop (DONE — `b89319a` merge)

Make one MCP call drive N iterations end-to-end while keeping the orchestrator-contract design intact.

- [x] `autonomous_research_loop` MCP tool — persisted-session driver (start/submit/status/cancel), doom-loop detector, content-hash state_snapshot, wall-clock × pod-rate budget cap (Phase 1.1/1.2/1.8)
- [x] `tree_autonomous_loop` MCP tool — same protocol over tree search; external_dispatch hint when pending nodes await fleet ops (Phase 1.4)
- [x] `harness_autonomous_loop` MCP tool — wraps HarnessOptimizer + judge-separation guard (Phase 1.5)
- [x] Auto-promote findings — `context_push_finding(auto_promote=True)` honors PROMOTION_RULES per-scope thresholds (Phase 1.7 small-batch)
- [x] `tree_prune(mode='auto')` — deterministic greedy kill via configured metric threshold (Phase 1.10, collapsed into the existing tool per "don't add a sibling tool when a parameter does the job")
- [x] Literature pre-injection — `with_literature=True` on autonomous_research_loop pulls top-K HF Papers via `literature.py` (Phase 1.7)
- [x] CLI: `crucible research run --iterations N --tier proxy --budget-usd $X` (Phase 1.1)
- [x] Plus the `autoresearch` importer (`crucible research import autoresearch <path>`) — strategic differentiator hedge against Lightning AI bolting multi-GPU onto autoresearch (Phase 1.6)
- [x] Part G — production-hardening sweep: 8 schema-skew / race-safety seams closed (state file_lock, atomic writes, content_hash, doom-loop, budget guard, file-lock factory typing, stale-submit MCP roundtrip, TUI partial-data)

**Exit:** One MCP call + one CLI command can drive a full closed loop with no orchestrator handholding beyond initial setup.

### Phase 2: Discoverability Surge (DONE — merged via `3518f75`)

Flatten 12-step onboarding to 5, make the 200+ MCP tools navigable, turn the TUI into a live cockpit.

- [x] `tool_router` MCP — 11-branch state-aware recommender (Phase 2.1)
- [x] Briefing `next_actions` field — `get_research_briefing()` returns structured tool recommendation + markdown "Recommended Next Tool" section (Phase 2.2)
- [x] TUI cockpit screens — FleetPane, QueuePane, LeaderboardPane, BriefingPane in TabbedContent (Phase 2.3, branch `phase-2-3-tui-cockpit`)
- [x] `docs/quickstart-5-minutes.md` — cold start to leaderboard + tool-router rec in 5 commands (Phase 2.4)
- [x] `runs_search` MCP tool — SQL-ish predicate with safe AST whitelist + optional `strict_fields` typo detection (Phase 2.5)
- [x] `crucible recipe list / get` CLI + recipe index in docs (Phase 2.6)
- [x] `docs/index.md` learning path — 5-section organization (Phase 2.7)
- [x] Plus `crucible mcp call <tool>` CLI single-shot dispatcher (Phase 2.4 hook)

**Exit:** First-day onboarding goes from 12 steps to 5; TUI shows live state; agents never have to guess "what should I call next?"

### Phase 3: Ecosystem Connections (DONE — merged via `fe5c2d3`)

Make Crucible bidirectional — publish outward today, ingest inward tomorrow.

- [x] Evaluators plugin family — `core/evaluators.py` parallel to data_sources/; builtin `lm_eval_harness` shells out to lm-eval CLI; `evaluator_list` MCP tool (Phase 3.3)
- [x] Optuna HPO bridge — `hpo_create_study` / `hpo_ask_trial` / `hpo_tell_result` / `hpo_status` MCP. Tell-and-ask over TPE/random/CMA-ES/BoTorch samplers; in-process cache (threading.Lock from Part H) + persisted JSON for resume (Phase 3.4)
- [x] External MCP consumption — `external_mcp_servers:` config + `external_mcp_list_servers` / `list_tools` / `call` MCP tools with per-call timeout (Phase 3.5)
- [x] Code-level mutation — Phase 5+ deferred per "Sakana spent 6 weeks on this"; interface stub + design doc + `code_mutation_list` MCP tool ship in Phase 3.6 (full impl in `docs/code-mutation-design.md`)
- [x] `research_arxiv_search` (Phase 3.1) + `research_openreview_search` (Phase 3.2) MCP tools — Atom-feed + v2/v1 fallback, defusedxml-safe parse

**Exit:** Crucible ingests external benchmark scores, consumes Optuna posteriors, calls external MCP, and mutates training code.

### Phase 4: Defensible-Niche Showcase (DONE — on `phase-4-showcase` branch pending merge)

Demonstrate the niche with an end-to-end story plus features competitors don't have.

- [x] `note_generate_paper_draft(track_name)` MCP tool — request_prompt/submit orchestrator contract, 7 required sections, markdown H1/H2 demotion to prevent header injection (Phase 4.1)
- [x] Memory-aware GIANTS synthesis — `design_synthesize_from_findings(policy='memory_filter')` with tunable `memory_filter_config` for half_life_days / cross_project_bonus / etc. (Phase 4.2)
- [x] `research_peer_sync(challenge_id)` — shared HF Discussion thread with v1 header convention; redact_secrets on both write and read sides (Phase 4.3 + Part H.1.2 read-side fix)
- [x] `examples/full_autonomous_discovery/` — 11-step playbook README exercising every Phase 1-4 surface end-to-end (Phase 4.4)
- [ ] Public showcase video / asciinema for the README (optional, deferred)

**Exit:** A new user can clone, run the demo, and have a paper draft + reproducibility bundle by lunch.

## Verification (cross-phase)

1. **Smoke gate every phase.** `tests/integration/test_phase_N_end_to_end.py` runs the headline deliverable on a fake-fleet (SSH provider pointed at localhost).
2. **Doc audit.** `docs/quickstart-5-minutes.md` must work verbatim after each phase. Tracked in CI doc-lint step.
3. **MCP tool descriptions.** Every new MCP tool includes REQUIRES/RETURNS/NEXT and is reachable from `tool_router`.
4. **Positioning gate.** Each phase exit reviews against the positioning statement: strengthens the niche, or drifts?

## What We Won't Build

- **Experiment tracking UI** — Use W&B or MLflow.
- **Kubernetes orchestration** — Use SkyPilot.
- **Model serving / inference** — Out of scope.
- **Dataset hosting** — Use HuggingFace.
- **Optimization math** — Use Optuna/Ax (integrate, don't reinvent).
- **Wet-lab integration** — Periodic Labs / Profluent / FutureHouse Phoenix own that space.
- **Frontier model training** — Crucible is model-agnostic; users bring their LLM via the orchestrator contract.
- **Enterprise dev-engineer agent** — Cognition Devin owns that. Different buyer.
- **Web UI (in this roadmap)** — TUI is enough for now; web UI is post-Phase 4 if at all.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Highest-impact contributions:

1. **Compute providers** — Modal, Lambda, Vast.ai, SkyPilot backends.
2. **Search strategies** — Optuna, Ax, code-mutation policies.
3. **Evaluator plugins** — lm-eval-harness, BIG-bench, papers-with-code adapters.
4. **Architecture plugins** — Mamba, SSM, MoD reference architectures.
5. **Domain spec plugins** — taps with task-specific harness templates.
6. **Bug reports** — File issues, we'll fix them.
