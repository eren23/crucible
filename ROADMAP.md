# Roadmap

## Where Crucible Sits Today

Crucible works. We've validated the full loop — provision RunPod pods, bootstrap, dispatch experiments, monitor, collect results, destroy. Training losses drop, W&B logs, experiments complete.

But it's alpha software built in one day from a competition codebase. Here's an honest assessment and what comes next.

## What Actually Works (v0.1.0-alpha)

| Feature | Status | Notes |
|---------|--------|-------|
| RunPod fleet provisioning | Working | Tested with 2-pod runs |
| SSH provider | Structural | Not battle-tested |
| Bootstrap (sync + install + data) | Working | Tested on RunPod |
| Experiment dispatch via SSH | Working | Detached subprocess on pods |
| Result collection via rsync | Working | Poll-based |
| W&B integration | Working | Logging confirmed live |
| CLI (`crucible` command) | Working | All subcommands wired |
| Autonomous researcher (Claude) | Structural | Loop exists, dry-run works, not battle-tested with real LLM |
| Model zoo | Structural | Components extracted, no integration tests |
| MCP server | Structural | Tools defined, not tested with real Claude session |
| Output parser | Working | Regex patterns validated against real training output |
| Configurable metrics | Working | No hardcoded val_bpb assumptions |
| Test suite (296 tests) | Passing | Core + runner + researcher + analysis + fleet covered |

## What's Not a Moat (Honest)

Before building more, we acknowledge what others do better:

- **Fleet orchestration**: [SkyPilot](https://github.com/skypilot-org/skypilot) (8K stars) supports 20+ clouds with cost optimization and spot failover. Our RunPod + SSH is functional but not competitive at the infra layer.
- **HPO math**: [Optuna](https://github.com/optuna/optuna) (11K stars) and [Ax](https://github.com/facebook/Ax) have mathematically superior search (TPE, Bayesian, CMA-ES). Our LLM-driven search is different, not necessarily better at parameter optimization.
- **Experiment tracking UI**: [W&B](https://wandb.ai) and [MLflow](https://mlflow.org) exist. We don't build dashboards.
- **Model zoo**: [HuggingFace Transformers](https://github.com/huggingface/transformers) exists. Our components are from one competition.

## What IS Unique

1. **Full-loop autonomy on rental GPUs** — No other tool goes from API key → autonomous research loop on RunPod/SSH
2. **Tiered experiment promotion** — Experiments earn expensive compute (smoke → proxy → medium → promotion)
3. **MCP agent integration** — Claude as research partner, not just coding assistant
4. **Training script agnosticism** — Env vars in, stdout out. No framework lock-in
5. **LLM-driven research that can propose new axes** — Not just grid search, but creative exploration

## Roadmap

### Phase 1: Harden What Exists (Next 2 weeks)

- [ ] CI/CD pipeline (GitHub Actions, tests on every push)
- [ ] PyPI release (`pip install crucible-ml`)
- [ ] Integration tests with real RunPod (manual gate)
- [ ] Battle-test autonomous researcher with real LLM calls
- [ ] Battle-test MCP server with real Claude session
- [ ] SSH provider testing on bare metal
- [ ] Better error messages and pre-flight validation
- [ ] Active experiment monitoring (heartbeat-based, not just result collection)

### Phase 2: Integrate, Don't Reinvent (Weeks 3-6)

- [ ] **SkyPilot provider** — Replace RunPod-specific fleet with SkyPilot for 20+ cloud support
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
3. **Training script examples** — Show Crucible working with your framework
4. **Bug reports** — File issues, we'll fix them
