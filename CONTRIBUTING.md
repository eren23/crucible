# Contributing to Crucible

Crucible is alpha software. Contributions are welcome — especially the kinds listed below.

## High-Impact Contributions

### 1. Compute Providers

We need backends beyond RunPod and SSH. Each provider implements the `FleetProvider` interface:

```python
class FleetProvider(ABC):
    def provision(self, *, count, name_prefix, **kwargs) -> list[dict]: ...
    def destroy(self, nodes, *, selected_names=None) -> list[dict]: ...
    def refresh(self, nodes) -> list[dict]: ...
    def wait_ready(self, nodes, *, timeout_seconds=900) -> list[dict]: ...
```

Wanted: **Modal**, **Lambda**, **Vast.ai**, **SkyPilot** (wraps all of these)

See `src/crucible/fleet/providers/runpod.py` for the reference implementation.

### 2. Search Strategies

The autonomous researcher currently uses LLM-driven hypothesis generation. We want pluggable strategies:

- **Optuna** — TPE, CMA-ES, Bayesian optimization
- **Ax** — Multi-objective Bayesian with Pareto analysis
- **Grid/Random** — Simple baselines
- **Hybrid** — LLM for exploration + mathematical methods for exploitation

### 3. Training Script Examples

Show Crucible working with different ML frameworks. An example needs:
- A training script that follows the [training contract](README.md#training-contract)
- A `crucible.yaml` config
- A brief README

### 4. Bug Reports

File issues at the GitHub repo. Include:
- What you ran (command, config)
- What happened (error, unexpected behavior)
- What you expected

## Development Setup

```bash
git clone <repo-url>
cd crucible
pip install -e ".[dev]"
PYTHONPATH=src pytest tests/ -v
```

## Code Conventions

See [CLAUDE.md](CLAUDE.md) for full conventions. Key points:

- `core/` has no dependencies on other crucible modules
- External deps (torch, anthropic, mcp) are lazy-imported
- Use the `CrucibleError` hierarchy, not bare `except Exception`
- Tests in `tests/` mirror source structure
- No hardcoded metrics — use `config.metrics.primary`

## Pull Requests

- One concern per PR
- Tests for new functionality
- Run `PYTHONPATH=src pytest tests/` before submitting
- Update CHANGELOG.md for user-facing changes
