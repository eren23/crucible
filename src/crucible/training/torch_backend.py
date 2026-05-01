"""PyTorch training backend — main training loop entry point.

Invoke directly (``python torch_backend.py``) or via the crucible runner / MCP tools.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Self-bootstrap: ensure src/ is on path when invoked directly
_src = str(Path(__file__).resolve().parent.parent.parent)
if _src not in sys.path:
    sys.path.insert(0, _src)

import copy
import io
import math
import os
import random
import signal
import subprocess
import time

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# Crucible training modules (siblings)
from crucible.training.hyperparams import Hyperparameters
from crucible.training.muon import zeropower_via_newtonschulz5
from crucible.training.data_loader import DistributedTokenLoader
from crucible.training.validation import validate_model, build_sentencepiece_luts, load_validation_tokens
from crucible.training.quantization import (
    CONTROL_TENSOR_NAME_PATTERNS,
    quantize_state_dict,
    dequantize_state_dict,
    compress_blob,
    decompress_blob,
    fake_int6_quant,
)
from crucible.training.ttt_eval import ttt_lora_evaluate

# Crucible model layer
from crucible.models.registry import build_model
from crucible.models.components.linear import CastedLinear

# Crucible runner utilities
from crucible.runner.tracker import RunTracker
from crucible.runner.wandb_logger import WandbLogger
from crucible.core.fingerprint import code_fingerprint
from crucible.core.io import collect_public_attrs

try:
    import zstandard as zstd
except ImportError:
    zstd = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


_zeropower_compiled = False


def main() -> None:
    global _zeropower_compiled
    import crucible.training.muon as _muon_mod

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    if not _zeropower_compiled:
        _muon_mod.zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
        _zeropower_compiled = True

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    # torchrun sets RANK, WORLD_SIZE, LOCAL_RANK automatically. Trust them.
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    grad_accum_steps_env = os.environ.get("GRAD_ACCUM_STEPS")
    if grad_accum_steps_env is not None:
        grad_accum_steps = int(grad_accum_steps_env)
        if grad_accum_steps <= 0:
            raise ValueError(f"GRAD_ACCUM_STEPS must be positive, got {grad_accum_steps}")
    elif world_size == 1:
        grad_accum_steps = 1
    else:
        if 8 % world_size != 0:
            raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
        grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0
    _graceful_shutdown = False
    def _handle_signal(signum, frame):
        nonlocal _graceful_shutdown
        _graceful_shutdown = True
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    tracker: RunTracker | None = None
    wandb: WandbLogger | None = None

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(args.multiscale_window > 0 or bool(args.block_pattern))

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        config = collect_public_attrs(args)
        fp = code_fingerprint(Path(__file__).resolve().parent.parent.parent.parent)
        config["code_fingerprint"] = fp["fingerprint"]
        config["code_files"] = fp["files"]
        run_tags = ["torch", args.model_family]
        if args.attention_variant != "standard":
            run_tags.append(f"attn:{args.attention_variant}")
        if args.residual_variant != "standard":
            run_tags.append(f"resid:{args.residual_variant}")
        if args.embed_bottleneck_dim > 0:
            run_tags.append("factorized_embed")
        if args.gpu_count > 1:
            run_tags.append(f"gpu:{args.gpu_count}")
        run_preset = os.environ.get("RUN_PRESET", "").strip()
        if run_preset:
            run_tags.append(run_preset)
        tracker = RunTracker(args.run_id, out_dir="logs", project_root=Path(__file__).resolve().parent.parent.parent.parent)
        tracker.write_manifest(
            backend="torch",
            script_path=Path(__file__),
            config=config,
            tags=run_tags,
            extra={
                "trainer": "torch_backend",
                "run_preset": run_preset or None,
                "parent_run_id": args.parent_run_id or None,
                "gpu_count": args.gpu_count,
            },
        )
        tracker.update(state="starting", phase="starting", backend="torch", config=config)
        wandb = WandbLogger.create(
            run_id=args.run_id,
            config=config,
            backend="torch",
            tracker=tracker,
            job_type=run_preset or None,
            tags=run_tags,
        )
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    base_model = build_model(args).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)

    # Int6 QAT: register forward pre-hooks that fake-quantize weight matrices.
    _qat_hooks: list = []
    if args.int6_qat:
        def _make_qat_hook(module: nn.Module):
            def hook(mod, inputs):
                mod.weight.data = fake_int6_quant(mod.weight.data)
            return hook
        for module in base_model.modules():
            if isinstance(module, CastedLinear):
                _qat_hooks.append(module.register_forward_pre_hook(_make_qat_hook(module)))

    # Discover and build callbacks BEFORE torch.compile so that on_model_ready
    # can register forward hooks that will be visible to the compiled graph.
    from crucible.core.plugin_discovery import discover_all_plugins
    from crucible.training.callbacks import CALLBACK_REGISTRY, build_callbacks
    _proj_root = Path(__file__).resolve().parent.parent.parent.parent
    discover_all_plugins(
        {"callbacks": CALLBACK_REGISTRY},
        project_root=_proj_root,
    )
    _callbacks_str = os.environ.get("CALLBACKS", "")
    _callbacks = build_callbacks(_callbacks_str) if _callbacks_str else []
    if _callbacks:
        log0(f"callbacks: {[type(cb).__name__ for cb in _callbacks]}")

    # on_model_ready: let callbacks register forward hooks BEFORE compile.
    _cb_state_early = {"model": base_model, "total_steps": args.iterations}
    for _cb in _callbacks:
        _cb.on_model_ready(_cb_state_early)

    # torch.compile gives meaningful throughput on long runs but takes 30-60s
    # to warm up and uses fullgraph=True (any graph break is fatal). Set
    # TORCH_COMPILE=0 to skip — useful for smoke iteration on plugins with
    # compile-incompatible ops (e.g. .item() in metric stashes) and for
    # variants whose compiled-graph time exceeds the smoke wallclock budget.
    if os.environ.get("TORCH_COMPILE", "1") != "0":
        compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    else:
        log0("torch.compile disabled by TORCH_COMPILE=0")
        compiled_model = base_model
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    token_param_names = base_model.token_parameter_names()
    head_param_names = {"lm_head.weight"} if base_model.lm_head is not None else set()
    token_params: list[Tensor] = []
    head_params: list[Tensor] = []
    matrix_params: list[Tensor] = []
    scalar_params: list[Tensor] = []
    for name, p in base_model.named_parameters():
        if not p.requires_grad:
            continue
        if name in token_param_names:
            token_params.append(p)
        elif name in head_param_names:
            head_params.append(p)
        elif p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS):
            matrix_params.append(p)
        else:
            scalar_params.append(p)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr

    # Discover custom optimizer plugins (callbacks already discovered before torch.compile).
    from crucible.training.optimizers import OPTIMIZER_REGISTRY, build_optimizer
    discover_all_plugins(
        {"optimizers": OPTIMIZER_REGISTRY},
        project_root=_proj_root,
    )

    # Pluggable per-group optimizers — env vars override defaults.
    _embed_opt = os.environ.get("EMBED_OPTIMIZER", "adam")
    _matrix_opt = os.environ.get("MATRIX_OPTIMIZER", "muon")
    _scalar_opt = os.environ.get("SCALAR_OPTIMIZER", "adamw")
    _head_opt = os.environ.get("HEAD_OPTIMIZER", "adam")

    # Adam-family kwargs — only forwarded when using adam/adamw to avoid
    # TypeError on optimizers that don't accept betas/eps/fused.
    _ADAM_FAMILY = {"adam", "adamw"}
    _adam_kw = dict(betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)

    optimizer_tok = build_optimizer(
        _embed_opt,
        [{"params": token_params, "lr": token_lr, "base_lr": token_lr}],
        **(_adam_kw if _embed_opt in _ADAM_FAMILY else {}),
    )
    optimizer_muon = build_optimizer(
        _matrix_opt,
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_weight_decay,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = build_optimizer(
        _scalar_opt,
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        weight_decay=args.adam_weight_decay,
        **(_adam_kw if _scalar_opt in _ADAM_FAMILY else {}),
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if head_params:
        optimizer_head = build_optimizer(
            _head_opt,
            [{"params": head_params, "lr": args.head_lr, "base_lr": args.head_lr}],
            **(_adam_kw if _head_opt in _ADAM_FAMILY else {}),
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(
        f"model_family:{args.model_family} attention_variant:{args.attention_variant} "
        f"residual_variant:{args.residual_variant} embed_bottleneck_dim:{args.embed_bottleneck_dim}"
    )
    log0(
        f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads} "
        f"share_blocks:{args.share_blocks} recurrence_steps:{args.recurrence_steps} state_dim:{args.state_dim}"
    )
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if head_params else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(
        f"lr_schedule:{args.lr_schedule} lr_decay_iters:{args.lr_decay_iters} "
        f"min_lr_scale:{args.min_lr_scale:.4f}"
    )
    log0(f"train_shard_limit:{args.train_shard_limit}")
    log0(f"seed:{args.seed}")

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(
        args.train_files,
        rank,
        world_size,
        device,
        shard_limit=args.train_shard_limit,
    )

    # Epoch-based training: resolve EPOCHS to iterations from dataset size
    if args.epochs > 0:
        from crucible.training.data_loader import count_shard_tokens
        total_tokens = count_shard_tokens(args.train_files, shard_limit=args.train_shard_limit)
        if total_tokens > 0:
            iterations = int(args.epochs * total_tokens / args.train_batch_tokens)
            log0(f"epoch_mode:epochs={args.epochs} total_tokens={total_tokens:,} "
                 f"tokens_per_step={args.train_batch_tokens} iterations={iterations}")
            args.iterations = iterations

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    _VAL_SAFETY_MS = 30_000.0  # reserve 30s for final validation + serialization
    if max_wallclock_ms is not None:
        max_wallclock_ms = max(max_wallclock_ms - _VAL_SAFETY_MS, 0.0)

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.lr_schedule == "cosine":
            warmup_steps = max(args.warmup_steps, 0)
            if warmup_steps > 0 and step < warmup_steps:
                return max(step, 1) / warmup_steps
            decay_iters = args.lr_decay_iters if args.lr_decay_iters > 0 else args.iterations
            if decay_iters <= warmup_steps:
                return args.min_lr_scale
            if step >= decay_iters:
                return args.min_lr_scale
            progress = (step - warmup_steps) / max(decay_iters - warmup_steps, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return args.min_lr_scale + (1.0 - args.min_lr_scale) * cosine
        if args.lr_schedule != "linear_warmdown":
            raise ValueError(f"Unsupported LR_SCHEDULE={args.lr_schedule!r}")
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
                if tracker is not None:
                    tracker.heartbeat("warming_up", warmup_step=warmup_step + 1, warmup_total=args.warmup_steps)
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(
            args.train_files,
            rank,
            world_size,
            device,
            shard_limit=args.train_shard_limit,
        )

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    _cb_state = {"model": base_model, "total_steps": args.iterations, "optimizers": optimizers}
    for _cb in _callbacks:
        _cb.on_train_begin(_cb_state)

    training_time_ms = 0.0
    stop_after_step: int | None = None
    swa_state: dict[str, Tensor] | None = None
    swa_count = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = validate_model(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            if tracker is not None:
                tracker.heartbeat(
                    "validating",
                    step=step,
                    total_steps=args.iterations,
                    latest_val_loss=val_loss,
                    latest_val_bpb=val_bpb,
                    train_time_ms=training_time_ms,
                )
            _val_metrics = {"val_loss": val_loss, "val_bpb": val_bpb}
            for _cb in _callbacks:
                _cb.on_validation_end(step, _val_metrics, _cb_state)
            if wandb is not None:
                _wandb_val = {
                    "run/phase": "validating",
                    "metrics/val_loss": val_loss,
                    "metrics/val_bpb": val_bpb,
                    "timing/train_time_ms": training_time_ms,
                    "timing/step_avg_ms": training_time_ms / max(step, 1),
                }
                for _mk, _mv in _val_metrics.items():
                    if _mk not in ("val_loss", "val_bpb") and isinstance(_mv, (int, float)):
                        _wandb_val[f"compression/{_mk}"] = _mv
                wandb.log(_wandb_val, step=step)
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        step_t0 = time.perf_counter()
        for _cb in _callbacks:
            _cb.on_step_begin(step, _cb_state)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps
        if torch.isnan(train_loss) or torch.isinf(train_loss):
            log0(f"FATAL: train_loss is {train_loss.item()} at step {step}. Halting.")
            if tracker is not None:
                tracker.finalize("failed", phase="nan_detected", step=step)
            if wandb is not None:
                wandb.finish(1)
            break

        for _cb in _callbacks:
            _cb.on_after_backward(step, _cb_state)

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        # SWA: accumulate fp32 weight average during warmdown phase.
        if args.swa_interval > 0 and scale < 1.0 and (step + 1) % args.swa_interval == 0:
            if swa_state is None:
                swa_state = {n: p.data.float().clone() for n, p in base_model.named_parameters()}
            else:
                for n, p in base_model.named_parameters():
                    swa_state[n].add_(p.data.float())
            swa_count += 1

        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        step += 1

        _step_metrics = {"train_loss": float(train_loss.item())}
        for _cb in _callbacks:
            _cb.on_step_end(step, _step_metrics, _cb_state)

        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        tok_s = args.train_batch_tokens / max(step_ms / 1000.0, 1e-9)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms tok_s:{tok_s:.0f}"
            )
            if tracker is not None:
                tracker.heartbeat(
                    "training",
                    step=step,
                    total_steps=args.iterations,
                    latest_train_loss=float(train_loss.item()),
                    train_time_ms=approx_training_time_ms,
                    step_avg_ms=approx_training_time_ms / step,
                    tok_s=tok_s,
                )
            if wandb is not None:
                _wandb_payload = {
                    "run/phase": "training",
                    "metrics/train_loss": float(train_loss.item()),
                    "timing/train_time_ms": approx_training_time_ms,
                    "timing/step_avg_ms": approx_training_time_ms / step,
                    "timing/tok_s": tok_s,
                }
                # Forward any extra metrics injected by callbacks
                for _mk, _mv in _step_metrics.items():
                    if _mk != "train_loss" and isinstance(_mv, (int, float)):
                        _wandb_payload[f"compression/{_mk}"] = _mv
                wandb.log(
                    _wandb_payload,
                    step=step,
                )

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step
        if stop_after_step is None and _graceful_shutdown:
            log0("graceful_shutdown: signal received, stopping after this step")
            stop_after_step = step

    for _cb in _callbacks:
        _cb.on_train_end(_cb_state)

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )
    if tracker is not None:
        tracker.heartbeat(
            "serializing",
            peak_memory_allocated_mib=torch.cuda.max_memory_allocated() // 1024 // 1024,
            peak_memory_reserved_mib=torch.cuda.max_memory_reserved() // 1024 // 1024,
        )

    # Remove QAT hooks before serialization.
    for h in _qat_hooks:
        h.remove()

    # Apply SWA averaged weights if collected.
    if swa_state is not None and swa_count > 0:
        log0(f"swa: applying averaged weights from {swa_count} snapshots")
        with torch.no_grad():
            for n, p in base_model.named_parameters():
                p.data.copy_((swa_state[n] / swa_count).to(dtype=p.dtype))
        del swa_state

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    # Save the raw state (useful for debugging/loading in PyTorch directly), then always produce
    # the compressed int8+zlib artifact and validate the round-tripped weights.

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    compress_mode = "zstd" if (args.quant_mode in ("int6", "int5_int6") and zstd is not None) else "zlib"
    quant_obj, quant_stats = quantize_state_dict(base_model.state_dict(), mode=args.quant_mode)
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = compress_blob(quant_raw, mode=compress_mode)
    quant_raw_bytes = len(quant_raw)
    artifact_name = f"final_model.{args.quant_mode}.ptz"
    if master_process:
        with open(artifact_name, "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize(artifact_name)
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model {args.quant_mode}+{compress_mode}: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size {args.quant_mode}+{compress_mode}: {quant_file_bytes + code_bytes} bytes")
        if tracker is not None:
            tracker.heartbeat(
                "serializing",
                final_model_path=str(Path(artifact_name).resolve()),
                model_bytes=quant_file_bytes,
            )

    if distributed:
        dist.barrier()
    with open(artifact_name, "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(decompress_blob(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = validate_model(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    q_eval_ms = 1000.0 * (time.perf_counter() - t_qeval)
    log0(
        f"final_{args.quant_mode}_{compress_mode}_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{q_eval_ms:.0f}ms"
    )
    log0(f"final_{args.quant_mode}_{compress_mode}_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
    if wandb is not None:
        wandb.log(
            {
                "run/phase": "final",
                "metrics/final_val_loss": q_val_loss,
                "metrics/final_val_bpb": q_val_bpb,
                "artifacts/model_bytes": quant_file_bytes if master_process else None,
                "timing/final_eval_ms": q_eval_ms,
            },
            step=step,
        )
    # LoRA test-time training evaluation (optional, env-var gated).
    ttt_val_loss, ttt_val_bpb = None, None
    if args.ttt_enabled:
        torch._dynamo.reset()
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        ttt_val_loss, ttt_val_bpb = ttt_lora_evaluate(
            args, base_model, rank, world_size, device,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        )
        torch.cuda.synchronize()
        log0(
            f"final_{args.quant_mode}_ttt_lora val_loss:{ttt_val_loss:.4f} val_bpb:{ttt_val_bpb:.4f} "
            f"eval_time:{1000.0 * (time.perf_counter() - t_ttt):.0f}ms"
        )
        if wandb is not None:
            wandb.log({"metrics/ttt_val_loss": ttt_val_loss, "metrics/ttt_val_bpb": ttt_val_bpb}, step=step)
    if wandb is not None:
        wandb.update_summary(
            {
                "final_val_loss": q_val_loss,
                "final_val_bpb": q_val_bpb,
                "ttt_val_bpb": ttt_val_bpb,
                "model_bytes": quant_file_bytes if master_process else None,
                "backend": "torch",
            }
        )
        wandb.finish(0)

    if tracker is not None:
        result_dict: dict = {
            "val_loss": q_val_loss,
            "val_bpb": q_val_bpb,
            "steps_completed": step,
            "train_time_ms": training_time_ms,
        }
        if ttt_val_bpb is not None:
            result_dict["ttt_val_loss"] = ttt_val_loss
            result_dict["ttt_val_bpb"] = ttt_val_bpb
        tracker.finalize(
            "completed",
            phase="completed",
            result=result_dict,
            model_bytes=quant_file_bytes if master_process else None,
        )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
