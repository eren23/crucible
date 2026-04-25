"""SFT-driven Diff-XYZ smoke runner.

Loads a small Qwen base in 4-bit, attaches LoRA, fine-tunes for a handful of
steps on a CommitPackFT subset (deduped against Diff-XYZ test), then runs
post-training eval on a small Diff-XYZ slice. Written to fit a 24GB GPU
(RTX 4090 / 3090 / A5000) and finish in ~5 minutes.

Env vars (all optional with sane defaults):
    DIFFXYZ_BASE_MODEL      base HF id, default 'Qwen/Qwen2.5-Coder-1.5B-Instruct'
    DIFFXYZ_LIMIT_TRAIN     # commits, default 200
    DIFFXYZ_MAX_STEPS       SFT steps, default 100
    DIFFXYZ_BATCH           per-device batch, default 1
    DIFFXYZ_GRAD_ACCUM      grad accumulation, default 4
    DIFFXYZ_LR              learning rate, default 2e-5
    DIFFXYZ_LORA_R          LoRA rank, default 32
    DIFFXYZ_EVAL_LIMIT      Diff-XYZ samples to score, default 30
    DIFFXYZ_EVAL_LANG       eval language, default 'python'
    DIFFXYZ_FORMAT          training/eval format, default 'search-replace'
    DIFFXYZ_OUT             result JSON path, default '/workspace/project/result.json'
    DIFFXYZ_CKPT_DIR        adapter save path, default '/workspace/project/ckpts'

Required keys (forwarded by Crucible env_forward):
    HF_TOKEN          (for gated models like Qwen3-Coder family)
    WANDB_API_KEY     (optional logging)

Stdout: prints ``RESULT EM=<float> IoU=<float>`` on the last line for Crucible.
"""
from __future__ import annotations

import difflib
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path


def _setup_paths() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    for p in (repo_root, repo_root / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _build_search_replace(old: str, new: str) -> str:
    """Render a search-replace diff from two snippets (line-level matcher)."""
    old_lines = old.splitlines(keepends=False)
    new_lines = new.splitlines(keepends=False)
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    blocks: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        search = "\n".join(old_lines[i1:i2])
        replace = "\n".join(new_lines[j1:j2])
        blocks.append(
            f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE"
        )
    return "\n".join(blocks)


def main() -> int:
    base_model = os.environ.get("DIFFXYZ_BASE_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
    limit_train = int(os.environ.get("DIFFXYZ_LIMIT_TRAIN", "200"))
    max_steps = int(os.environ.get("DIFFXYZ_MAX_STEPS", "100"))
    per_device_batch = int(os.environ.get("DIFFXYZ_BATCH", "1"))
    grad_accum = int(os.environ.get("DIFFXYZ_GRAD_ACCUM", "4"))
    lr = float(os.environ.get("DIFFXYZ_LR", "2e-5"))
    lora_r = int(os.environ.get("DIFFXYZ_LORA_R", "32"))
    eval_limit = int(os.environ.get("DIFFXYZ_EVAL_LIMIT", "30"))
    eval_lang = os.environ.get("DIFFXYZ_EVAL_LANG", "python")
    fmt = os.environ.get("DIFFXYZ_FORMAT", "search-replace")
    out_path = Path(os.environ.get("DIFFXYZ_OUT", "/workspace/project/result.json"))
    ckpt_dir = Path(os.environ.get("DIFFXYZ_CKPT_DIR", "/workspace/project/ckpts"))
    seed = int(os.environ.get("DIFFXYZ_SEED", "42"))

    _setup_paths()
    print(f"[sft_smoke] base={base_model}  train_commits={limit_train}  "
          f"steps={max_steps}  fmt={fmt}  eval_lang={eval_lang}", flush=True)

    # ---- imports here (need torch + transformers + trl) ----
    import torch  # type: ignore[import-not-found]
    from datasets import Dataset, load_dataset  # type: ignore[import-not-found]
    from peft import (  # type: ignore[import-not-found]
        LoraConfig, get_peft_model, prepare_model_for_kbit_training,
    )
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer  # type: ignore[import-not-found]

    from evaluation.diff_xyz.dataset import DiffXYZSample, load_samples
    from evaluation.diff_xyz.harness import score_sample
    from evaluation.diff_xyz.models import HFBackend
    from evaluation.diff_xyz.prompts import system_prompt, user_prompt

    # ---- 1. Diff-XYZ test for both dedup and post-training eval ----
    print("[sft_smoke] loading Diff-XYZ test (for dedup)...", flush=True)
    dxyz = load_dataset("JetBrains-Research/diff-xyz", split="test")
    dxyz_keys: set[tuple[str, str, str]] = {
        (str(row["repo"]), str(row["commit"]), str(row["path"])) for row in dxyz
    }

    # ---- 2. Load base + LoRA in 4-bit ----
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    print(f"[sft_smoke] loading base {base_model} in 4-bit...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_r * 2, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ---- 3. Stream CommitPackFT-python, dedup, build SFT examples ----
    print("[sft_smoke] streaming CommitPackFT/python...", flush=True)
    raw = load_dataset("bigcode/commitpackft", "python", split="train", streaming=True)
    examples: list[dict] = []
    skipped = 0
    for row in raw:
        if len(examples) >= limit_train * 3:  # 3 tasks per commit
            break
        old = row.get("old_contents") or ""
        new = row.get("new_contents") or ""
        if not old or not new:
            continue
        key = (str(row.get("repos", "")), str(row.get("commit", "")),
               str(row.get("old_file", "")))
        if key in dxyz_keys:
            skipped += 1
            continue
        sr = _build_search_replace(old, new)
        if not sr:
            continue
        sample = DiffXYZSample(
            repo=key[0], commit=key[1], path=key[2], lang="python",
            old_code=old, new_code=new,
            udiff="", udiff_h="", udiff_l="", search_replace=sr,
            n_added=0, n_removed=0, n_hunks=0, change_kind="",
        )
        for task, target in (
            ("apply", new),
            ("anti_apply", old),
            ("diff_gen", sr),
        ):
            sys_p = system_prompt(fmt, "format")
            usr_p = user_prompt(task, fmt, sample)
            examples.append({"messages": [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": usr_p},
                {"role": "assistant", "content": target},
            ]})
    print(f"[sft_smoke] sft examples={len(examples)} (skipped {skipped} in test)", flush=True)

    if not examples:
        print("[sft_smoke] FAIL no training data", file=sys.stderr, flush=True)
        return 3

    ds = Dataset.from_list(examples)
    sft_cfg = SFTConfig(
        output_dir=str(ckpt_dir),
        max_steps=max_steps,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=0.03,
        max_seq_length=2048,
        bf16=True,
        logging_steps=10,
        save_steps=max(50, max_steps // 2),
        save_total_limit=1,
        seed=seed,
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else ["none"],
        run_name=os.environ.get("WANDB_RUN_NAME", f"sft-smoke-{base_model.split('/')[-1]}"),
        gradient_checkpointing=True,
        packing=False,
    )

    # ---- 4. Train ----
    t0 = time.time()
    trainer = SFTTrainer(model=model, args=sft_cfg, train_dataset=ds, processing_class=tokenizer)
    print(f"[sft_smoke] training {max_steps} steps...", flush=True)
    trainer.train()
    trainer.save_model(str(ckpt_dir))
    print(f"[sft_smoke] training done in {time.time() - t0:.1f}s", flush=True)

    # ---- 5. Post-training eval ----
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[sft_smoke] eval on {eval_limit} Diff-XYZ samples ({eval_lang})...", flush=True)
    eval_samples = load_samples(limit=eval_limit, langs=[eval_lang], seed=seed)
    backend = HFBackend(model_id=base_model)
    backend._model = model
    backend._tokenizer = tokenizer
    rows = []
    for i, sample in enumerate(eval_samples):
        r = score_sample(backend, sample, idx=i, task="apply", fmt=fmt,
                         sys_mode="format", max_tokens=2048, temperature=0.0)
        rows.append(r)
    em = statistics.fmean([r.em for r in rows]) if rows else 0.0
    iou = statistics.fmean([r.iou for r in rows]) if rows else 0.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "base_model": base_model,
        "limit_train": limit_train,
        "max_steps": max_steps,
        "fmt": fmt,
        "eval_lang": eval_lang,
        "eval_limit": eval_limit,
        "metrics": {"EM": em, "IoU": iou, "n": len(rows)},
        "per_sample": [
            {"idx": r.idx, "em": r.em, "iou": r.iou, "lang": r.lang,
             "elapsed_s": r.elapsed_s, "error": r.error}
            for r in rows
        ],
    }, indent=2), encoding="utf-8")
    print(f"[sft_smoke] wrote {out_path}", flush=True)
    print(f"RESULT EM={em:.4f} IoU={iou:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
