# HuggingFace Fine-Tune Example

Demonstrates **bring-your-own-trainer** — how to use Crucible's fleet orchestration with an external training framework (HuggingFace 🤗 Transformers' `Trainer`) without plugging into the built-in training backends.

## What this demonstrates

- A **standalone training script** that reads Crucible env vars and prints the [training contract](../../README.md#training-contract) output lines, with zero Crucible imports.
- **DistilBERT fine-tune on SST-2** — small enough to complete on a single RTX 4090 in under 10 minutes.
- **Crucible presets** (`smoke`, `screen`, `proxy`) selecting batch size / epochs / eval cadence via env vars.
- **Fleet workflow**: `crucible project new → provision → bootstrap → dispatch → collect → analyze` works identically to the built-in examples.

## Why "bring your own trainer"?

Crucible ships two training backends (`torch`, `generic`) that register models, data adapters, and objectives via the plugin system. That's great when you want the full observability (live step parsing, OOM retry, tree search, research loop).

But sometimes you already have a training script you trust — HuggingFace `Trainer`, PyTorch Lightning, a custom loop, whatever — and you just want Crucible to orchestrate it. This example shows the contract: read env vars for config, print `step:X/Y train_loss:Z` lines to stdout, and everything downstream (output parser, leaderboard, sensitivity analysis) works.

## Files

| File | Purpose |
|---|---|
| `train.py` | Standalone fine-tune script (no Crucible imports) |
| `crucible.yaml` | Project-wide config (provider, presets, metrics) |
| `requirements.txt` | Python deps — torch, transformers, datasets, evaluate |
| `README.md` | This file |

## Quick start (local)

```bash
cd examples/huggingface_finetune
pip install -r requirements.txt

# Smoke test — ~60s on a single GPU
ITERATIONS=50 BATCH_SIZE=16 LR=2e-5 python train.py
```

Expected output:
```
step:10/50 train_loss:0.6931
step:20/50 train_loss:0.5214
step:30/50 train_loss:0.4103
step:40/50 train_loss:0.3567
step:50/50 val_loss:0.4102 val_acc:0.8165
Serialized model /tmp/hf-sst2-smoke 267849216 bytes
```

## Running on a fleet

Generate a project spec pointing at your fork:

```bash
crucible project new hf-sst2 --template generic \
    --set REPO_URL=https://github.com/me/my-hf-example
```

Edit `.crucible/projects/hf-sst2.yaml` to use this example's `train.py`:

```yaml
train: "python train.py"
install:
  - "-r requirements.txt"
env_set:
  MODEL_NAME: distilbert-base-uncased
  DATASET_NAME: sst2
```

Then run the standard fleet loop:

```bash
crucible fleet provision --count 1
crucible fleet bootstrap
crucible run experiment --preset screen
crucible run collect
```

## Customization

- **Different model**: set `MODEL_NAME=bert-base-uncased` (or any HF model ID).
- **Different task**: the script works for any GLUE single-sentence classification task; set `DATASET_NAME=cola` or similar.
- **Different metric**: `METRIC_NAME=matthews_correlation` for CoLA, `accuracy` for most others.
- **Regression instead of classification**: swap `AutoModelForSequenceClassification` for `AutoModelForSequenceClassification` with `num_labels=1, problem_type="regression"` and report MSE.
