# 2-Pod Experiment on RunPod

Run ML experiments on 2 RunPod GPU pods using Crucible.

## Prerequisites

- RunPod account with API key
- SSH key at `~/.ssh/id_ed25519_runpod` (public key uploaded to RunPod)
- `RUNPOD_API_KEY` in your `.env` file
- A training script that follows the [training contract](../../README.md#training-contract)

## Setup

```bash
# In your ML project directory
crucible init

# Edit crucible.yaml:
#   - Set provider.type to "runpod"
#   - Set provider.ssh_key to your RunPod SSH key path
#   - Set provider.gpu_types to your preferred GPUs
#   - Point training.script to your training script
#   - Configure presets with your experiment tiers
```

## The Workflow

### 1. Provision pods

```bash
crucible fleet provision --count 2 --name-prefix my-experiment
```

This creates 2 RunPod pods and waits for SSH connectivity. Takes ~2-5 minutes.

### 2. Bootstrap pods

```bash
crucible fleet bootstrap --train-shards 1
```

Syncs your code, installs pip dependencies, and downloads training data on each pod. Use `--skip-install` if deps are already installed, `--skip-data` if data is pre-cached.

### 3. Check pod health

```bash
crucible fleet status
```

All pods should show `state=ready`.

### 4. Prepare experiment spec

Create a JSON file with your experiments:

```json
[
  {
    "name": "baseline_10L",
    "tier": "proxy",
    "backend": "torch",
    "tags": ["baseline"],
    "config": {
      "MODEL_FAMILY": "baseline",
      "NUM_LAYERS": "10",
      "MODEL_DIM": "512"
    }
  },
  {
    "name": "looped_8L",
    "tier": "proxy",
    "backend": "torch",
    "tags": ["looped"],
    "config": {
      "MODEL_FAMILY": "looped",
      "NUM_LAYERS": "8",
      "RECURRENCE_STEPS": "12"
    }
  }
]
```

### 5. Enqueue experiments

```bash
crucible run enqueue --spec experiments.json
# Or limit to first N:
crucible run enqueue --spec experiments.json --limit 3
```

### 6. Dispatch to pods

```bash
crucible run dispatch
```

Assigns queued experiments to idle pods. With 2 pods and 3 experiments, 2 run in parallel and 1 stays queued.

### 7. Monitor progress

```bash
# One-shot status
crucible fleet monitor

# Live refresh every 60 seconds
crucible fleet monitor --watch 60
```

### 8. Collect results

```bash
crucible run collect
```

Rsyncs logs and results from pods. Run periodically or after experiments finish.

### 9. View results

```bash
crucible analyze rank --top 10
```

### 10. Clean up

```bash
crucible fleet destroy
```

## Full Example (Parameter Golf)

```bash
cd /path/to/parameter-golf

# Provision 2 GPU pods
crucible fleet provision --count 2 --name-prefix crucible-test

# Bootstrap with 1 training shard (fast)
crucible fleet bootstrap --train-shards 1

# Enqueue 3 SOTA experiments
crucible run enqueue --spec specs/next_batch_merged.json --limit 3

# Dispatch to pods
crucible run dispatch

# Monitor until done (~1 hour for overnight tier)
crucible fleet monitor --watch 60

# Collect final results
crucible run collect

# Check leaderboard
crucible analyze rank --top 10

# Tear down pods
crucible fleet destroy
```

## Cost Estimate

| GPUs | Duration | Estimated Cost |
|------|----------|---------------|
| 2x RTX 3090 | 1 hour | ~$0.80 |
| 2x RTX 4090 | 1 hour | ~$1.40 |
| 2x RTX 4090 | 4 hours | ~$5.60 |

Costs vary by availability and spot pricing.
