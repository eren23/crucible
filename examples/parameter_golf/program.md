# Parameter Golf Research Program

## Objective
Minimize validation bits-per-byte (val_bpb) for a language model that fits in 16MB
(int8 + zlib compressed), trained in under 10 minutes on 8xH100 GPUs.

## Model Families
- **baseline**: Standard transformer (9 layers, 512 dim, 8 heads, 4 KV heads)
- **looped**: Weight-sharing recurrent transformer
- **convloop**: ConvGPT-inspired bottleneck + looped core
- **prefix_memory**: Bounded internal memory variant

## Key Hyperparameters
- MODEL_FAMILY, NUM_LAYERS, MODEL_DIM, NUM_HEADS, NUM_KV_HEADS
- RECURRENCE_STEPS, SHARE_BLOCKS (for looped)
- ATTENTION_VARIANT (standard, paired), RESIDUAL_VARIANT (default, gated)
- QUANT_MODE (int8, int6, int5), INT6_QAT
- MUON_WEIGHT_DECAY, SMEAR_GATE, BIGRAM_HASH, ORTHO_INIT

## Research Priorities
1. Quantization-aware training (int6 QAT shows promise)
2. Weight-sharing architectures (looped > baseline at same param budget)
3. Attention optimizations (paired attention, windowed)
4. Training dynamics (warmdown LR, MLP multiplier)

## Constraints
- 16MB artifact size (code + compressed model weights)
- 10-minute wallclock training limit
- FineWeb validation set, tokenizer-agnostic bits-per-byte metric
- No external teachers, no external memory at inference
