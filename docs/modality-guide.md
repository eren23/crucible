# Modality Guide: Training Any Model with Crucible

Crucible was born as a language model research platform, but its training
pipeline is modality-agnostic.  This guide shows how to train diffusion
models, world models, vision classifiers, or anything else through the
same fleet orchestration and experiment tracking infrastructure.

## How it works

Crucible's training contract is simple: **environment variables in, stdout
metrics out**.  The runner launches a training script as a subprocess,
passes config via env vars, and parses structured output lines for metrics.

```
┌──────────────┐     env vars     ┌──────────────────┐     stdout      ┌─────────────┐
│  Experiment  │ ──────────────→  │  Training Script  │ ─────────────→  │  Output     │
│  Runner      │                  │  (any backend)    │                 │  Parser     │
└──────────────┘                  └──────────────────┘                 └─────────────┘
```

This means **any training script that reads env vars and prints metrics can
plug into Crucible** — fleet provisioning, dispatching, result collection,
and leaderboard ranking all work automatically.

## The 4 extension points

To add a new modality, you touch up to 4 things:

### 1. Model (`CrucibleModel` subclass)

```python
from crucible.models.base import CrucibleModel
from crucible.models.registry import register_model

class MyModel(CrucibleModel):
    def forward(self, **batch) -> dict[str, Tensor]:
        # Receive whatever your DataAdapter sends
        # Return dict with at least {"loss": scalar_tensor}
        ...

    def training_step(self, **batch) -> dict[str, Tensor]:
        return self.forward(**batch)

    @classmethod
    def modality(cls) -> str:
        return "my_modality"  # "diffusion", "world_model", "vision", etc.

register_model("my_model", lambda args: MyModel(...))
```

The model can compute loss internally (recommended for complex models like
diffusion) or return predictions and let an external Objective handle loss.

### 2. Data Adapter (`DataAdapter` subclass)

```python
from crucible.training.data_adapters import DataAdapter, register_data_adapter

class MyAdapter(DataAdapter):
    def next_batch(self, batch_size=8, device=None, **kwargs) -> dict[str, Any]:
        # Return a dict of tensors matching your model's forward() signature
        return {"images": images_tensor, "labels": labels_tensor}

    @classmethod
    def modality(cls) -> str:
        return "my_modality"

register_data_adapter("my_data", MyAdapter)
```

Built-in adapters: `token` (LM), `image_folder` (torchvision datasets),
`synthetic_images`, `synthetic_video` (bouncing balls).

### 3. Training Objective (optional)

Only needed if your model returns raw predictions instead of computing loss
internally.

```python
from crucible.training.objectives import TrainingObjective, register_objective

class MyObjective(TrainingObjective):
    name = "my_loss"

    def compute(self, predictions, targets) -> dict[str, Any]:
        loss = F.mse_loss(predictions["output"], targets["target"])
        return {"loss": loss, "my_metric": loss}

register_objective("my_loss", MyObjective)
```

Built-in objectives: `cross_entropy`, `mse`, `kl_divergence`, `composite`,
`diffusion` (noise MSE), `jepa` (prediction + variance regularization).

### 4. Config (`crucible.yaml`)

```yaml
training:
  - backend: generic
    script: src/crucible/training/generic_backend.py
    modality: my_modality

metrics:
  primary: my_metric     # which metric to rank experiments by
  direction: minimize    # or "maximize"

presets:
  smoke:
    MODEL_FAMILY: my_model
    DATA_ADAPTER: my_data
    TRAINING_OBJECTIVE: my_loss  # or omit if model handles loss
    ITERATIONS: "200"
    BATCH_SIZE: "16"
    LR: "0.001"
    MAX_WALLCLOCK_SECONDS: "60"
```

## Step-by-step: adding a new modality

### 1. Write your model

Create a file (e.g., `.crucible/architectures/my_model.py` or
`examples/my_modality/model.py`):

```python
class MyDiffusionModel(CrucibleModel):
    def __init__(self, ...):
        super().__init__()
        self.unet = ...
        self.schedule = ...

    def forward(self, images, **kw):
        noise = torch.randn_like(images)
        t = torch.randint(0, 1000, (images.shape[0],), device=images.device)
        noisy = self.schedule.q_sample(images, t, noise)
        pred = self.unet(noisy, t)
        loss = F.mse_loss(pred, noise)
        return {"loss": loss, "noise_mse": loss}

    @classmethod
    def modality(cls):
        return "diffusion"
```

### 2. Write your data adapter

```python
class CIFARAdapter(DataAdapter):
    def __init__(self):
        from torchvision import datasets, transforms
        tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
        self.dataset = datasets.CIFAR10("./data", download=True, transform=tfm)
        self._idx = 0

    def next_batch(self, batch_size=32, device=None, **kw):
        images = []
        for _ in range(batch_size):
            img, _ = self.dataset[self._idx % len(self.dataset)]
            images.append(img)
            self._idx += 1
        batch = torch.stack(images)
        if device: batch = batch.to(device)
        return {"images": batch}
```

### 3. Register everything

```python
register_model("my_diffusion", lambda args: MyDiffusionModel(...))
register_data_adapter("cifar_images", CIFARAdapter)
```

### 4. Configure `crucible.yaml`

Set `metrics.primary` to the metric your model reports. Set presets with
your model-specific env vars.

### 5. Run locally

```bash
PYTHONPATH=src MODEL_FAMILY=my_diffusion DATA_ADAPTER=cifar_images \
    BATCH_SIZE=32 ITERATIONS=500 LR=0.001 LOG_INTERVAL=10 \
    python3 -m crucible.training.generic_backend
```

### 6. Run on fleet

Exactly the same as LM experiments:

```
provision_nodes → fleet_refresh → bootstrap_nodes →
enqueue_experiment(config={"MODEL_FAMILY": "my_diffusion", ...}, backend="generic") →
dispatch_experiments → collect_results → get_leaderboard
```

## Output format

The generic backend emits metrics in two formats:

```
# Standard format (backward-compatible with LM parser)
step:100/1000 train_loss:0.456789

# Generic format (any metric name)
metric:noise_mse=0.456789
metric:pred_loss=0.123456

# Validation (emitted at VAL_INTERVAL steps)
step:500/1000 val_loss:0.389012 val_bpb:0.389012

# Final result (emitted at end of training)
final_generic val_loss:0.345678 val_bpb:0.345678
```

The output parser picks up both formats.  For non-LM models, `val_bpb` is
set equal to `val_loss` (it's only meaningful for language models).

Generic metrics (`metric:name=value`) are collected into the result dict
and available for leaderboard ranking via `metrics.primary` in config.

## Environment variables

The generic backend reads these env vars:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_FAMILY` | `baseline` | Architecture name in registry |
| `DATA_ADAPTER` | `token` | Data adapter name |
| `TRAINING_OBJECTIVE` | `cross_entropy` | Objective name (ignored if model has `compute_loss`) |
| `ITERATIONS` | `400` | Total training steps |
| `BATCH_SIZE` | `8` | Batch size (for non-token modalities) |
| `SEQ_LEN` | `128` | Sequence length (for token adapter) |
| `IMAGE_SIZE` | `32` | Image resolution (for vision/diffusion) |
| `NUM_FRAMES` | `4` | Frame count (for video/world model) |
| `LR` | `3e-4` | Learning rate |
| `WEIGHT_DECAY` | `0.0` | Weight decay |
| `GRAD_ACCUM_STEPS` | `1` | Gradient accumulation |
| `MAX_WALLCLOCK_SECONDS` | `0` | Wall-clock timeout (0 = no limit) |
| `LOG_INTERVAL` | `10` | Steps between log lines |
| `VAL_INTERVAL` | `0` | Steps between validation (0 = off) |
| `LR_SCHEDULE` | `cosine` | `"cosine"` or `"constant"` |
| `WARMUP_STEPS` | `0` | LR warmup steps |

All env vars are passed through to the model's `args` namespace, so you
can add model-specific variables (e.g., `DIFFUSION_STEPS`, `EMA_DECAY`)
and read them in your factory function.

## What you DON'T need to touch

These systems work unchanged for any modality:

- **Fleet management**: `provision_nodes`, `destroy_nodes`, `sync_code`, `bootstrap_nodes`
- **Queue + dispatch**: `enqueue_experiment`, `dispatch_experiments`, `cancel_experiment`
- **Result collection**: `collect_results`, `get_experiment_result`
- **Analysis**: `get_leaderboard` (uses `metrics.primary`), `get_sensitivity`
- **Tree search**: `tree_create` → `tree_expand_node` → etc.
- **Version store**: `version_save_design`, `version_diff`, etc.
- **Hub**: findings, tracks, cross-project sharing
- **Notes + context**: `note_add`, `context_push_finding`

## Reference implementations

| Example | Location | Modality | Data |
|---------|----------|----------|------|
| DDPM on MNIST | `examples/diffusion/` | Diffusion | MNIST (auto-download) |
| JEPA World Model | `examples/world_model/` | World model | Bouncing balls (synthetic) |
| Two-Tower LM | `src/crucible/models/user_architectures/example_two_tower.py` | LM | Token stream |

## torch_backend vs generic_backend

| Feature | `torch_backend` | `generic_backend` |
|---------|----------------|-------------------|
| Modality | LM only | Any |
| DDP | Yes | No (single GPU) |
| Optimizer | Adam + Muon | AdamW |
| LR schedule | Custom | Cosine with warmup |
| Quantization | Int6 QAT | No |
| TTT evaluation | Yes | No |
| OOM retry | TRAIN_BATCH_TOKENS | TRAIN_BATCH_TOKENS or BATCH_SIZE |
| Validation | Val BPB (SentencePiece) | Generic val_loss |

Use `torch_backend` for competitive LM training.  Use `generic_backend` for
everything else (diffusion, vision, world models, RL, etc.).
