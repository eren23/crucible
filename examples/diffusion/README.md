# Diffusion Model Example: DDPM on MNIST

A complete denoising diffusion probabilistic model (DDPM) that trains on MNIST
digits using Crucible's generic training backend.

## What this demonstrates

- Building a **non-LM model** with `CrucibleModel`
- Writing a **custom data adapter** (MNIST images)
- Using the **generic training backend** for diffusion training
- Configuring **modality-specific presets** in `crucible.yaml`
- Running on the **fleet** with the same workflow as LM experiments

## Architecture

- **Model**: 3-level UNet with sinusoidal time embeddings, residual blocks, skip connections
- **Training**: DDPM noise prediction — sample timestep, add noise, predict noise, MSE loss
- **Data**: MNIST 28x28 grayscale digits, normalized to [-1, 1]

## Quick start (local)

```bash
# From repo root
cd examples/diffusion

# Register the model and adapter, then train
MODEL_FAMILY=ddpm_unet DATA_ADAPTER=mnist_images \
    BATCH_SIZE=32 ITERATIONS=500 IMAGE_CHANNELS=1 MODEL_DIM=32 \
    DIFFUSION_STEPS=100 LR=0.001 LOG_INTERVAL=10 \
    python train_generic.py
```

## Using presets

```bash
# Smoke test (60s, 200 steps)
python -c "
from crucible.runner.experiment import run_experiment
result = run_experiment(
    config={},
    name='ddpm-smoke',
    backend='generic',
    preset='smoke',
    project_root='.',
)
print(f'Status: {result[\"status\"]}')
print(f'Metrics: {result.get(\"result\", {})}')
"
```

## Running on fleet

```bash
# Same Crucible workflow — just specify the backend and model
python -c "
from crucible.fleet.manager import FleetManager
from crucible.core.config import load_config
fm = FleetManager(load_config())

# Provision, bootstrap, then enqueue:
# MODEL_FAMILY=ddpm_unet, DATA_ADAPTER=mnist_images, etc.
"
```

Or via MCP tools:
1. `provision_nodes(count=1, gpu_type="RTX 4090")`
2. `fleet_refresh()` → `bootstrap_nodes()`
3. `enqueue_experiment(name="ddpm-mnist", config={"MODEL_FAMILY": "ddpm_unet", "DATA_ADAPTER": "mnist_images", ...}, backend="generic")`
4. `dispatch_experiments()` → `collect_results()` → `get_leaderboard()`

## Files

| File | Purpose |
|------|---------|
| `model.py` | DDPM UNet implementation (CrucibleModel subclass) |
| `data_adapter.py` | MNIST data adapter |
| `crucible.yaml` | Project config with diffusion-specific presets |
| `__init__.py` | Auto-registers model + adapter on import |

## Customization

- **Different dataset**: Swap `MNISTAdapter` for `ImageFolderAdapter` or write your own
- **Larger model**: Increase `MODEL_DIM` and add channel multipliers
- **Different schedule**: Override `DiffusionSchedule` in `model.py`
- **Conditional generation**: Add class labels to the time embedding
