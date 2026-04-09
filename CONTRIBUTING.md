# Contributing to Crucible

Crucible is alpha software. Contributions are welcome — especially the kinds listed below.

## High-Impact Contributions

### 1. Compute Providers

We need backends beyond RunPod and SSH. Each provider implements the `FleetProvider` interface and registers via the plugin system:

```python
from crucible.fleet.provider_registry import register_provider
from crucible.fleet.provider import FleetProvider

class MyProvider(FleetProvider):
    def provision(self, *, count, name_prefix, **kwargs) -> list[dict]: ...
    def destroy(self, nodes, *, selected_names=None) -> list[dict]: ...
    def refresh(self, nodes) -> list[dict]: ...
    def wait_ready(self, nodes, *, timeout_seconds=900) -> list[dict]: ...

register_provider("my_cloud", lambda **kw: MyProvider(**kw), source="local")
```

Wanted: **Modal**, **Lambda**, **Vast.ai**, **SkyPilot** (wraps all of these)

See `src/crucible/fleet/providers/runpod.py` for the reference implementation. Plugins go in `.crucible/plugins/providers/`.

### 2. Search Strategies

The autonomous researcher currently uses LLM-driven hypothesis generation. We want pluggable strategies:

- **Optuna** — TPE, CMA-ES, Bayesian optimization
- **Ax** — Multi-objective Bayesian with Pareto analysis
- **Grid/Random** — Simple baselines
- **Hybrid** — LLM for exploration + mathematical methods for exploitation

### 3. Plugins (Optimizers, Schedulers, Callbacks, Loggers, etc.)

Crucible has a unified plugin system with 13 pluggable types: optimizers, schedulers, providers, loggers, callbacks, data_adapters, data_sources, objectives, architectures, block_types, stack_patterns, augmentations, activations. Contributing a plugin is two files:

```
.crucible/plugins/optimizers/my_optimizer.py   # the code
.crucible/plugins/optimizers/my_optimizer.yaml # metadata (optional)
```

The `.py` file calls `register_*()` at module level:
```python
from crucible.training.optimizers import register_optimizer

def _my_factory(params, **kwargs):
    return MyOptimizer(params, **kwargs)

register_optimizer("my_optimizer", _my_factory, source="local")
```

To share plugins via community taps, see the [Community Taps](#5-community-taps) section.

### 4. Community Taps

A tap is a git repo containing plugins. To create one:

1. Create a repo with this structure:
   ```
   optimizers/my_opt/
     plugin.yaml    # required: name, type, version, description
     my_opt.py      # the plugin code
   ```

2. Validate locally before committing:
   ```bash
   crucible tap validate .
   crucible tap validate . --warnings-as-errors  # strict mode for CI
   ```

3. Others install with: `crucible tap add <your-repo-url> && crucible tap install my_opt`

The `plugin.yaml` schema is defined in `crucible.core.plugin_schema` and enforced by `crucible tap validate`. Required fields are `name`, `type`, `version`, `description`; `author`, `tags`, `crucible_compat`, and `dependencies` are recommended (warning, not error, if missing). See the tap's CONTRIBUTING.md for the full schema.

To contribute to an existing tap: fork it, add your plugin, open a PR.

### 5. Training Script Examples

Show Crucible working with different ML frameworks. An example needs:
- A training script that follows the [training contract](README.md#training-contract)
- A `crucible.yaml` config
- A brief README

### 6. Bug Reports

File issues at the GitHub repo. Include:
- What you ran (command, config)
- What happened (error, unexpected behavior)
- What you expected

Especially useful for fleet/bootstrap bugs: run `crucible fleet status` and attach the per-node `bootstrap_steps` output from the failing node.

## Development Setup

```bash
git clone <repo-url>
cd crucible
pip install -e ".[dev]"
PYTHONPATH=src pytest tests/ -v
```

For fleet / config / template / schema work (the most active areas), you can run just the relevant slice:

```bash
PYTHONPATH=src pytest \
    tests/test_config.py \
    tests/test_project_template.py \
    tests/test_fleet_manager.py \
    tests/test_transactional_provision.py \
    tests/test_bootstrap_state.py \
    tests/test_ssh_timeouts.py \
    tests/test_ssh_provider.py \
    tests/test_data_probe.py \
    tests/test_plugin_schema.py \
    -q
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
