"""HuggingFace fine-tune bring-your-own-trainer example.

Deliberately imports ZERO Crucible modules. The training contract is:

  - Read config from environment variables
  - Print `step:{step}/{total} train_loss:{loss}` during training
  - Print `step:{step}/{total} val_loss:{loss} val_acc:{acc}` at eval time
  - Print `Serialized model {path} {N} bytes` at the end

Any script that follows this contract works with Crucible's generic backend;
the output parser picks up the metric lines and feeds them into the
leaderboard and analysis tools.

Config env vars (all optional — reasonable defaults for a smoke test):

    MODEL_NAME:         HuggingFace model ID (default: distilbert-base-uncased)
    DATASET_NAME:       GLUE subset name (default: sst2)
    ITERATIONS:         Max training steps (default: 50)
    BATCH_SIZE:         Per-device train batch size (default: 16)
    EVAL_BATCH_SIZE:    Per-device eval batch size (default: 32)
    LR:                 Learning rate (default: 5e-5)
    WARMUP_STEPS:       Linear warmup steps (default: 5)
    LOG_INTERVAL:       Steps between train_loss log lines (default: 10)
    MAX_LENGTH:         Tokenizer max_length (default: 128)
    MAX_WALLCLOCK_SECONDS:  Hard wall-clock cap (default: 300)
    MODEL_SAVE_DIR:     Directory to save the final model (default: /tmp/hf-model)
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def main() -> int:
    model_name = os.environ.get("MODEL_NAME", "distilbert-base-uncased")
    dataset_name = os.environ.get("DATASET_NAME", "sst2")
    metric_name = os.environ.get("METRIC_NAME", "accuracy")
    max_steps = _env_int("ITERATIONS", 50)
    batch_size = _env_int("BATCH_SIZE", 16)
    eval_batch_size = _env_int("EVAL_BATCH_SIZE", 32)
    lr = _env_float("LR", 5e-5)
    warmup_steps = _env_int("WARMUP_STEPS", 5)
    log_interval = _env_int("LOG_INTERVAL", 10)
    max_length = _env_int("MAX_LENGTH", 128)
    wall_budget = _env_int("MAX_WALLCLOCK_SECONDS", 300)
    save_dir = Path(os.environ.get("MODEL_SAVE_DIR", "/tmp/hf-model"))

    try:
        import torch
        from datasets import load_dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
    except ImportError as exc:
        print(
            f"ERROR: missing dependency — install requirements.txt ({exc})",
            file=sys.stderr,
        )
        return 2

    start_time = time.monotonic()

    print(f"Loading {dataset_name} via datasets.load_dataset ...", flush=True)
    ds = load_dataset("glue", dataset_name)
    # SST-2 has train/validation/test; GLUE test labels are hidden, so we
    # use the validation split for val.
    train_split = ds["train"]
    val_split = ds["validation"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    text_key = "sentence" if "sentence" in train_split.column_names else "text"

    def _tokenize(examples):
        return tokenizer(
            examples[text_key],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    train_ds = train_split.map(_tokenize, batched=True)
    val_ds = val_split.map(_tokenize, batched=True)
    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")
    keep = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=keep)
    val_ds.set_format(type="torch", columns=keep)

    num_labels = len(set(int(x) for x in train_split["label"]))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    class CrucibleLogCallback(TrainerCallback):
        """Emit Crucible training-contract lines from Trainer's log stream."""

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            step = int(state.global_step)
            if "loss" in logs and step % log_interval == 0:
                print(
                    f"step:{step}/{max_steps} train_loss:{logs['loss']:.4f}",
                    flush=True,
                )

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if not metrics:
                return
            step = int(state.global_step)
            val_loss = metrics.get("eval_loss", float("nan"))
            val_acc = metrics.get("eval_accuracy")
            if val_acc is None:
                # Fall back to any accuracy-like metric
                for k, v in metrics.items():
                    if "accuracy" in k or k == "eval_matthews_correlation":
                        val_acc = v
                        break
            extras = f" val_acc:{val_acc:.4f}" if val_acc is not None else ""
            print(
                f"step:{step}/{max_steps} val_loss:{val_loss:.4f}{extras}",
                flush=True,
            )

    def _compute_metrics(eval_pred):
        import numpy as np
        preds, labels = eval_pred
        preds = np.argmax(preds, axis=-1)
        return {"accuracy": float((preds == labels).mean())}

    args = TrainingArguments(
        output_dir=str(save_dir),
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        logging_steps=max(1, log_interval // 2),
        eval_strategy="steps",
        eval_steps=max(max_steps // 4, 5),
        save_strategy="no",
        report_to=[],
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
        callbacks=[CrucibleLogCallback()],
    )

    print(f"Starting training: {max_steps} steps, lr={lr}, bs={batch_size}", flush=True)
    trainer.train()

    elapsed = time.monotonic() - start_time
    if elapsed > wall_budget:
        print(
            f"WARNING: wall-clock budget ({wall_budget}s) exceeded: {elapsed:.1f}s",
            file=sys.stderr,
        )

    # Final eval — guarantees we emit at least one val line for the leaderboard
    metrics = trainer.evaluate()
    final_step = int(trainer.state.global_step)
    final_loss = metrics.get("eval_loss", float("nan"))
    final_acc = metrics.get("eval_accuracy")
    extras = f" val_acc:{final_acc:.4f}" if final_acc is not None else ""
    print(
        f"step:{final_step}/{max_steps} val_loss:{final_loss:.4f}{extras}",
        flush=True,
    )

    # Save and report size — Crucible's parser picks up the byte count
    # for model_bytes / Pareto frontier analysis.
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_dir))
    total_bytes = sum(
        p.stat().st_size for p in save_dir.rglob("*") if p.is_file()
    )
    print(f"Serialized model {save_dir} {total_bytes} bytes", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
