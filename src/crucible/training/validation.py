"""Tokenizer-agnostic validation and BPB evaluation.

It's common for small models to have a large fraction of their parameters be
embeddings, since the 2 * d_model * d_vocab vectors can be gigantic.  Instead
of locking the tokenizer, we let you bring your own and calculate our
validation metrics on the average compression of the validation set.  We
calculate BPB (bits-per-byte) instead of validation loss, so we need methods
to count the number of bits per token in the tokenizer.
"""
from __future__ import annotations

import glob
import math
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from crucible.training.data_loader import load_data_shard


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    if tokens.numel() - 1 < seq_len:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens


def validate_model(
    args,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    # Sliding-window validation: each token is scored with up to (seq_len - stride)
    # tokens of left context.  Target positions in the context region are masked with
    # -100 so F.cross_entropy(ignore_index=-100) skips them automatically.
    seq_len = args.train_seq_len
    stride = args.eval_stride if args.eval_stride > 0 else seq_len
    n_tokens = val_tokens.numel() - 1

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    # Build scoring windows and distribute across ranks.
    windows: list[tuple[int, int, int]] = []  # (ctx_start, score_start, score_end)
    for score_start in range(0, n_tokens, stride):
        score_end = min(score_start + stride, n_tokens)
        ctx_start = max(score_end - seq_len, 0)
        windows.append((ctx_start, score_start, score_end))
    my_windows = windows[rank::world_size]
    win_batch = max(1, args.val_batch_size // seq_len)

    model.train(False)
    with torch.inference_mode():
        for b in range(0, len(my_windows), win_batch):
            batch = my_windows[b : b + win_batch]
            bs = len(batch)
            x = torch.zeros(bs, seq_len, dtype=torch.int64)
            y = torch.full((bs, seq_len), -100, dtype=torch.int64)
            score_lens: list[int] = []
            for i, (cs, ss, se) in enumerate(batch):
                wl = se - cs
                x[i, :wl] = val_tokens[cs:se]
                y[i, :wl] = val_tokens[cs + 1 : se + 1]
                ctx_len = ss - cs
                if ctx_len > 0:
                    y[i, :ctx_len] = -100
                score_lens.append(se - ss)
            x = x.to(device=device, non_blocking=True)
            y = y.to(device=device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            scored = sum(score_lens)
            val_loss_sum += batch_loss.to(torch.float64) * scored
            val_token_count += scored
            for i, (cs, ss, se) in enumerate(batch):
                ctx_len = ss - cs
                sl = se - ss
                prev_ids = x[i, ctx_len : ctx_len + sl]
                tgt_ids = y[i, ctx_len : ctx_len + sl]
                token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
                token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
                val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)
