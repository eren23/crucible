# Local Smoke Test

Run a quick experiment locally to validate your training script works with Crucible.

## Setup

```bash
# In your ML project directory
crucible init
# Edit crucible.yaml — point training.script at your training script
```

## Run

```bash
# Quick smoke test (60 seconds)
crucible run experiment --preset smoke

# With config overrides
crucible run experiment --preset smoke --set MODEL_FAMILY=baseline --set NUM_LAYERS=9

# Named experiment
crucible run experiment --preset smoke --name my_first_test --set MODEL_DIM=256

# Proxy tier (30 minutes)
crucible run experiment --preset proxy --name longer_run
```

## Check Results

```bash
crucible analyze rank
```

## What the Smoke Preset Does

The `smoke` preset runs your training script with minimal settings:
- `MAX_WALLCLOCK_SECONDS=60` (1 minute cap)
- `ITERATIONS=400` (few training steps)

This validates that:
1. Your training script reads env vars correctly
2. It prints output patterns Crucible can parse
3. The full pipeline works (launch → parse → collect results)

## Training Script Requirements

Your script must:
- **Read config from environment variables** (e.g., `os.environ.get("ITERATIONS", "1000")`)
- **Print parseable output** to stdout:
  - `step:{step}/{total} train_loss:{loss}` — training progress
  - `step:{step}/{total} val_loss:{loss} val_bpb:{bpb}` — validation
  - `Serialized model ... {N} bytes` — model size

See `examples/basic/train.py` for a minimal example.
