# Roadmap

## Where Crucible Sits Today

Crucible works. We've validated the full loop — provision RunPod pods, bootstrap, dispatch experiments, monitor, collect results, destroy. Training losses drop, W&B logs, experiments complete. Hub-based cross-project knowledge sharing, research briefing, notes, and a 53-tool MCP server are all functional.

But it's alpha software. Here's an honest assessment and what comes next.

## What Actually Works (v0.2.0-alpha)

| Feature | Status | Notes |
|---------|--------|-------|
| RunPod fleet provisioning | Working | Tested with multi-pod runs |
| SSH provider | Structural | Not battle-tested |
| Bootstrap (sync + install + data) | Working | Tested on RunPod |
| Experiment dispatch via SSH | Working | Detached subprocess on pods |
| Result collection via rsync | Working | Poll-based |
| W&B integration | Working | Logging, annotation, image support |
| CLI (`crucible` command) | Working | All subcommands wired |
| Autonomous researcher (Claude) | Structural | Loop exists, dry-run works |
| Model zoo + plugin system | Working | 4 built-in + user_architectures/ plugins |
| MCP server (53 tools) | Working | Fleet, design, research, hub, models |
| REST API (10 endpoints) | Working | FastAPI thin wrapper |
| Hub (cross-project knowledge) | Working | Git-synced findings and tracks |
| Notes system | Working | Markdown with YAML frontmatter |
| Research briefing | Working | LLM session orientation |
| Output parser | Working | Regex patterns validated |
| Configurable metrics | Working | No hardcoded assumptions |
| Interactive TUI | Working | Design browser, diff, history |
| Test suite | Partial | Core/runner/analysis covered; fleet/components gaps |

## What's Not a Moat (Honest)

Before building more, we acknowledge what others do better:

- **Fleet orchestration**: [SkyPilot](https://github.com/skypilot-org/skypilot) supports 20+ clouds with cost optimization and spot failover. Our RunPod + SSH is functional but not competitive at the infra layer.
- **HPO math**: [Optuna](https://github.com/optuna/optuna) and [Ax](https://github.com/facebook/Ax) have mathematically superior search (TPE, Bayesian, CMA-ES). Our LLM-driven search is different, not necessarily better at parameter optimization.
- **Experiment tracking UI**: [W&B](https://wandb.ai) and [MLflow](https://mlflow.org) exist. We don't build dashboards.
- **Model zoo**: [HuggingFace Transformers](https://github.com/huggingface/transformers) exists. Our components are from one competition.

## What IS Unique

1. **Full-loop autonomy on rental GPUs** — No other tool goes from API key → autonomous research loop on RunPod/SSH
2. **Tiered experiment promotion** — Experiments earn expensive compute (smoke → proxy → medium → promotion)
3. **MCP agent integration** — Claude as research partner, not just coding assistant
4. **Training script agnosticism** — Env vars in, stdout out. No framework lock-in
5. **LLM-driven research that can propose new axes** — Not just grid search, but creative exploration
6. **Architecture plugins** — Drop a .py file, it's auto-discovered and works on remote GPUs
7. **Cross-project knowledge hub** — Findings, tracks, and briefings persist across projects and machines

## Roadmap

### Phase 1: Harden What Exists (Current)

- [x] MCP server with 53 tools
- [x] Hub system for cross-project findings
- [x] Notes system for experiment annotation
- [x] Research briefing for LLM session orientation
- [x] REST API server
- [x] Architecture plugin system with auto-discovery
- [x] W&B bridge with image logging and annotation
- [ ] CI/CD pipeline (GitHub Actions, tests on every push)
- [ ] PyPI release (`pip install crucible-ml`)
- [ ] Battle-test autonomous researcher with real LLM calls
- [ ] SSH provider testing on bare metal

### Phase 1.5: Robustness (Next)

- [ ] **Test coverage push** — Fleet orchestration, architecture forward-pass, component, runner execution tests
- [ ] **Pre-dispatch validation** — Verify model family exists before sending work to pods
- [ ] **Provider plugin system** — Registry + auto-discovery pattern (like architecture plugins)
- [ ] **Better error messages** — Pre-flight checks before expensive operations
- [ ] **Active experiment monitoring** — Heartbeat-based, not just result collection
- [ ] **Integration tests with real RunPod** (manual gate)

### Phase 2: Integrate, Don't Reinvent (Weeks 3-6)

- [ ] **SkyPilot provider** — Replace RunPod-specific fleet with SkyPilot for 20+ cloud support
- [ ] **More provider plugins** — Modal, Lambda, Vast.ai as user_providers/
- [ ] **Optuna strategy** — `crucible research start --strategy optuna` alongside LLM-driven search
- [ ] **Ax integration** — Bayesian optimization for systematic exploitation
- [ ] **Configurable output patterns** — Users define their own stdout → metric patterns in YAML
- [ ] **W&B experiment grouping** — Auto-group by wave, tier, model family

### Phase 3: Build Unique Value (Weeks 6+)

- [ ] **Hybrid search** — LLM for creative exploration + Optuna for systematic exploitation, automatically blended
- [ ] **Research strategies as plugins** — Users define custom loop phases (Python classes)
- [ ] **Cross-experiment learning** — LLM sees full project history, recognizes patterns across runs
- [ ] **Code-level search (AIDE-style)** — LLM proposes training script modifications, not just config changes
- [ ] **Experiment reports** — Auto-generate markdown reports with methodology, results, insights

### What We Won't Build

- **Experiment tracking UI** — Use W&B or MLflow
- **Kubernetes orchestration** — Use SkyPilot
- **Model serving / inference** — Out of scope
- **Dataset hosting** — Use HuggingFace
- **Optimization math** — Use Optuna/Ax (integrate, don't reinvent)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to help. Highest-impact contributions:

1. **Compute providers** — Modal, Lambda, Vast.ai, SkyPilot backends
2. **Search strategies** — Optuna, Ax, random, grid implementations
3. **Architecture plugins** — Example plugins in `user_architectures/`
4. **Training script examples** — Show Crucible working with your framework
5. **Bug reports** — File issues, we'll fix them
