"""TTT (Test-Time Training) LoRA evaluation.

Evaluates a trained model with batched per-document LoRA adaptation at
test time.  This is an optional evaluation mode gated by TTT_ENABLED.
"""
from __future__ import annotations

import glob
import math
from pathlib import Path

import torch
import torch.distributed as dist
from torch import Tensor, nn

from crucible.training.data_loader import load_data_shard

# BatchedTTTLoRA lives in crucible.models.components.lora
from crucible.models.components.lora import BatchedTTTLoRA


BOS_ID = 1


def _find_docs(all_tokens: Tensor) -> list[tuple[int, int]]:
    """Return (start_offset, length) for each document delimited by BOS tokens."""
    bos_positions = (all_tokens == BOS_ID).nonzero(as_tuple=True)[0].numpy()
    docs = []
    for i in range(len(bos_positions)):
        start = int(bos_positions[i])
        end = int(bos_positions[i + 1]) + 1 if i + 1 < len(bos_positions) else all_tokens.numel()
        if end - start >= 2:
            docs.append((start, end - start))
    return docs


def _chunk_window(ci: int, pred_len: int, num_chunks: int, chunk_size: int, seq_len: int):
    chunk_start = ci * chunk_size
    chunk_end = pred_len if ci == num_chunks - 1 else (ci + 1) * chunk_size
    win_start = max(0, chunk_end - seq_len)
    return win_start, chunk_end - win_start, chunk_start - win_start, chunk_end - chunk_start


def ttt_lora_evaluate(
    args, base_model: nn.Module, rank: int, world_size: int,
    device: torch.device, base_bytes_lut: Tensor, has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    """Evaluate with batched LoRA test-time training. Returns (val_loss, val_bpb)."""
    files = sorted(glob.glob(args.val_files))
    all_tokens = torch.cat([load_data_shard(Path(f)) for f in files])
    docs = _find_docs(all_tokens)
    rank_docs = docs[(len(docs) * rank) // world_size : (len(docs) * (rank + 1)) // world_size]
    chunk_size, ttt_seq = args.ttt_chunk_size, args.ttt_eval_seq_len
    batch_size, lora_rank = args.ttt_batch_size, args.ttt_lora_rank
    rank_docs.sort(key=lambda d: (d[1] - 2) // chunk_size)

    base_model.train(False)
    for p in base_model.parameters():
        p.requires_grad_(False)
    lora = BatchedTTTLoRA(batch_size, base_model, lora_rank).to(device)
    opt = torch.optim.Adam(lora.parameters(), lr=args.ttt_lora_lr, betas=(args.beta1, args.beta2), eps=1e-10)
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    byte_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)

    for bi in range(0, len(rank_docs), batch_size):
        batch = rank_docs[bi:bi + batch_size]
        bsz = len(batch)
        if bsz == batch_size:
            cur_lora, cur_opt = lora, opt
            cur_lora.reset()
            for g in cur_opt.param_groups:
                for p in g['params']:
                    s = cur_opt.state.get(p)
                    if s:
                        s['exp_avg'].zero_(); s['exp_avg_sq'].zero_(); s['step'].fill_(0)
        else:
            cur_lora = BatchedTTTLoRA(bsz, base_model, lora_rank).to(device)
            cur_opt = torch.optim.Adam(cur_lora.parameters(), lr=args.ttt_lora_lr, betas=(args.beta1, args.beta2), eps=1e-10)
        pred_lens = [dl - 1 for _, dl in batch]
        num_chunks = [(pl + chunk_size - 1) // chunk_size for pl in pred_lens]
        max_nc = max(num_chunks)
        for ci in range(max_nc):
            active = [ci < nc for nc in num_chunks]
            needs_train = any(ci < nc - 1 for nc in num_chunks)
            cs0 = _chunk_window(ci, (ci + 1) * chunk_size, ci + 1, chunk_size, ttt_seq)
            context_size, chunk_offset = cs0[1], cs0[2]
            x = torch.zeros(bsz, context_size, dtype=torch.int64, device=device)
            y = torch.zeros(bsz, context_size, dtype=torch.int64, device=device)
            doc_info: list[tuple[int, int]] = []
            for b in range(bsz):
                if not active[b]:
                    doc_info.append((0, 0)); continue
                ds, dl = batch[b]
                ws, wl, co, cl = _chunk_window(ci, pred_lens[b], num_chunks[b], chunk_size, ttt_seq)
                chunk = all_tokens[ds + ws: ds + ws + wl + 1].to(dtype=torch.int64, device=device)
                x[b, :wl] = chunk[:-1]; y[b, :wl] = chunk[1:]
                doc_info.append((co, cl))
            if needs_train:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    ptl = base_model(x, y, lora=cur_lora)
            else:
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    ptl = base_model(x, y, lora=cur_lora)
            with torch.no_grad():
                for b in range(bsz):
                    if not active[b]:
                        continue
                    co, cl = doc_info[b]
                    lbl = ptl[b, co:co + cl].to(torch.float64)
                    prev, tgt = x[b, co:co + cl], y[b, co:co + cl]
                    tok_bytes = base_bytes_lut[tgt].to(torch.float64)
                    tok_bytes += has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]
                    loss_sum += lbl.sum(); byte_sum += tok_bytes.sum(); token_count += cl
            if needs_train:
                mask = torch.tensor([float(ci < num_chunks[b] - 1) for b in range(bsz)], device=device)
                per_doc = ptl[:, chunk_offset:chunk_offset + chunk_size].mean(dim=-1)
                cur_opt.zero_grad(); (per_doc * mask).sum().backward(); cur_opt.step()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
    val_loss = float(loss_sum.item() / token_count.item())
    val_bpb = float((loss_sum.item() / math.log(2.0)) / byte_sum.item())
    return val_loss, val_bpb
