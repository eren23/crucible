"""Hyperparameters: environment-variable-driven training configuration.

All hyperparameters are read from environment variables with sensible defaults.
The class is instantiated once at script startup, after .env files are loaded.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from crucible.core.env import load_env_files

# Load .env files from the project root (two levels up from this file:
# src/crucible/training/ -> src/crucible/ -> src/ -> project root).
# Also try the direct parent chain for backward compatibility.
_training_dir = Path(__file__).resolve().parent
_project_root = _training_dir.parent.parent.parent
load_env_files(_project_root)


class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    lr_schedule = os.environ.get("LR_SCHEDULE", "linear_warmdown").strip().lower()
    lr_decay_iters = int(os.environ.get("LR_DECAY_ITERS", "0"))
    min_lr_scale = float(os.environ.get("MIN_LR_SCALE", "0.1"))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape.
    model_family = os.environ.get("MODEL_FAMILY", "baseline").strip().lower()
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    share_blocks = int(os.environ.get("SHARE_BLOCKS", 1))
    recurrence_steps = int(os.environ.get("RECURRENCE_STEPS", 0))
    state_dim = int(os.environ.get("STATE_DIM", 256))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    embed_bottleneck_dim = int(os.environ.get("EMBED_BOTTLENECK_DIM", "0"))
    attention_variant = os.environ.get("ATTENTION_VARIANT", "standard").strip().lower()
    residual_variant = os.environ.get("RESIDUAL_VARIANT", "standard").strip().lower()
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.03))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.02))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.02))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    muon_weight_decay = float(os.environ.get("MUON_WEIGHT_DECAY", 0.0))
    adam_weight_decay = float(os.environ.get("ADAM_WEIGHT_DECAY", 0.0))
    train_shard_limit = int(os.environ.get("TRAIN_SHARD_LIMIT", "0"))

    # Eval.
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))

    # Post-training.
    quant_mode = os.environ.get("QUANT_MODE", "int8").strip().lower()
    swa_interval = int(os.environ.get("SWA_INTERVAL", 0))
    int6_qat = bool(int(os.environ.get("INT6_QAT", "0")))

    # Model extras.
    smear_gate = bool(int(os.environ.get("SMEAR_GATE", "0")))
    bigram_hash = bool(int(os.environ.get("BIGRAM_HASH", "0")))
    bigram_hash_buckets = int(os.environ.get("BIGRAM_HASH_BUCKETS", 2048))
    bigram_hash_embed_dim = int(os.environ.get("BIGRAM_HASH_EMBED_DIM", "128"))
    ortho_init = bool(int(os.environ.get("ORTHO_INIT", "0")))
    spectral_embed_init = bool(int(os.environ.get("SPECTRAL_EMBED_INIT", "0")))
    trigram_hash = bool(int(os.environ.get("TRIGRAM_HASH", "0")))
    trigram_hash_buckets = int(os.environ.get("TRIGRAM_HASH_BUCKETS", "4096"))

    # Novel architecture features.
    conv_block = bool(int(os.environ.get("CONV_BLOCK", "0")))
    conv_kernel = int(os.environ.get("CONV_KERNEL", "3"))
    multiscale_window = int(os.environ.get("MULTISCALE_WINDOW", "0"))
    token_merge_layer = int(os.environ.get("TOKEN_MERGE_LAYER", "0"))
    token_merge_threshold = float(os.environ.get("TOKEN_MERGE_THRESHOLD", "0.9"))
    block_pattern = os.environ.get("BLOCK_PATTERN", "").strip()
    activation = os.environ.get("ACTIVATION", "relu_sq").strip().lower()

    # Mixture of Experts.
    use_moe = bool(int(os.environ.get("USE_MOE", "0")))
    moe_num_experts = int(os.environ.get("MOE_NUM_EXPERTS", "4"))
    moe_top_k = int(os.environ.get("MOE_TOP_K", "2"))

    # TTT LoRA.
    ttt_lora_rank = int(os.environ.get("TTT_LORA_RANK", 8))
    ttt_lora_lr = float(os.environ.get("TTT_LORA_LR", 0.01))
    ttt_chunk_size = int(os.environ.get("TTT_CHUNK_SIZE", 256))
    ttt_eval_seq_len = int(os.environ.get("TTT_EVAL_SEQ_LEN", 1024))
    ttt_batch_size = int(os.environ.get("TTT_BATCH_SIZE", 64))
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))

    # Lineage tracking.
    parent_run_id = os.environ.get("PARENT_RUN_ID", "")

    # Multi-GPU support.
    gpu_count = int(os.environ.get("GPU_COUNT", "1"))

    def __init__(self) -> None:
        """Validate critical hyperparameters on instantiation."""
        checks = {
            "vocab_size": (self.vocab_size, 1),
            "num_layers": (self.num_layers, 1),
            "model_dim": (self.model_dim, 1),
            "num_heads": (self.num_heads, 1),
            "num_kv_heads": (self.num_kv_heads, 1),
            "mlp_mult": (self.mlp_mult, 1),
        }
        for name, (value, minimum) in checks.items():
            if value < minimum:
                raise ValueError(f"{name.upper()} must be >= {minimum}, got {value}")
