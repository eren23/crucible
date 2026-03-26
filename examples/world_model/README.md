# World Model Example: JEPA on Bouncing Balls

A Joint Embedding Predictive Architecture (JEPA) world model that learns to
predict future visual states in latent space.  Trains on self-generated
synthetic video of bouncing colored squares.

## What this demonstrates

- Building a **world model** with `CrucibleModel`
- Writing a **self-contained data adapter** (no external datasets needed)
- **EMA target encoder** with stop-gradient (BYOL/VICReg-style)
- **Multi-step prediction loss** over frame sequences
- **Variance regularization** to prevent representation collapse
- Running on the **fleet** with the same workflow as LM experiments

## Architecture

- **Encoder**: 3-layer CNN → adaptive pool → linear projection → [B, D] embeddings
- **Predictor**: 3-layer MLP maps (z_t, action_t) → predicted z_{t+1}
- **Target encoder**: EMA copy of encoder (stop-gradient targets)
- **Loss**: MSE(predicted z, target z) + variance regularization

Inspired by V-JEPA and Le-WM, scaled down for demonstration.

## Quick start (local)

```bash
# From repo root
cd examples/world_model

# Register the model and adapter, then train
MODEL_FAMILY=jepa_wm DATA_ADAPTER=bouncing_balls \
    BATCH_SIZE=8 ITERATIONS=500 IMAGE_SIZE=32 NUM_FRAMES=4 \
    MODEL_DIM=64 ACTION_DIM=2 BASE_CHANNELS=16 LR=0.001 \
    LOG_INTERVAL=10 \
    python train_generic.py
```

## Using presets

```bash
# Smoke test (60s, 200 steps)
python -c "
from crucible.runner.experiment import run_experiment
result = run_experiment(
    config={},
    name='jepa-smoke',
    backend='generic',
    preset='smoke',
    project_root='.',
)
print(f'Status: {result[\"status\"]}')
print(f'Metrics: {result.get(\"result\", {})}')
"
```

## Running on fleet

Same as any Crucible experiment — provision, bootstrap, enqueue, dispatch:

```bash
# Via MCP tools:
# 1. provision_nodes(count=1, gpu_type="RTX 4090")
# 2. fleet_refresh() → bootstrap_nodes()
# 3. enqueue_experiment(name="jepa-balls", config={
#        "MODEL_FAMILY": "jepa_wm",
#        "DATA_ADAPTER": "bouncing_balls",
#        "BATCH_SIZE": "32",
#        "NUM_FRAMES": "6",
#        ...
#    }, backend="generic")
# 4. dispatch_experiments() → collect_results() → get_leaderboard()
```

## Files

| File | Purpose |
|------|---------|
| `model.py` | JEPA world model (encoder + predictor + EMA target) |
| `data_adapter.py` | Bouncing ball video generator |
| `crucible.yaml` | Project config with world model presets |
| `__init__.py` | Auto-registers model + adapter on import |

## Customization

- **Real video data**: Replace `BouncingBallAdapter` with a loader for Atari frames, RoboNet, etc.
- **Larger encoder**: Swap `CNNEncoder` for a ResNet or ViT
- **Action-conditioned**: The predictor already takes actions; connect to a real environment
- **Multi-step rollout**: Extend the forward pass to predict N steps ahead autoregressively
- **Contrastive loss**: Add an InfoNCE term alongside MSE for sharper representations
