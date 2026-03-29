# Crucible Compression Toolkit + World Model Modality

**Date:** 2026-03-29
**Status:** Draft

## Context

Synapse (our Rust+Zig inference engine) already demonstrates post-training compression of LE-WM world models — INT8, Q4, ternary quantization, and Wanda pruning — achieving ~8MB from a 69MB f32 baseline. But post-training compression leaves quality on the table. Training-aware compression (QAT, gradual pruning, distillation) can produce models that are both **smaller and better** than post-training approaches.

Crucible's plugin system (3-tier registry, callback hooks, objective composition, tree search) is perfectly suited to systematically explore compression recipes. This spec defines:

1. **Generic compression plugins** — QAT, pruning, distillation, sensitivity analysis callbacks that work with any model
2. **World model modality** — LE-WM (ViT+DiT JEPA) as the first target, extensible to other world models

Both ship as a **tap repo** (`crucible-compression-tap`), installable via `crucible tap add`.

## Architecture

### Two Layers

```
Layer 1: Generic Compression Plugins (model-agnostic)
├── QAT callbacks (INT8, INT4, mixed-precision)
├── Pruning callbacks (magnitude, Wanda, layer removal, head pruning)
├── Knowledge distillation (callback + objective)
├── Sensitivity analysis (per-layer importance)
├── Compression metrics tracking
└── Search tree recipes (systematic exploration)

Layer 2: World Model Modality (LE-WM specific, future-proof)
├── Trajectory data adapter (HDF5)
├── SIGReg objective (Cramér-Wold regularizer)
├── ViT+DiT architecture plugin
└── CEM planner evaluation callback
```

Layer 1 composes with Layer 2 naturally:
```bash
CALLBACKS=sensitivity_analysis,qat_mixed,wanda_pruning,distillation,compression_metrics \
MODEL_FAMILY=lewm \
DATA_ADAPTER=trajectory_hdf5 \
TRAINING_OBJECTIVE=composite
```

### Tap Repo Layout

```
crucible-compression-tap/
├── plugin.yaml                          # tap manifest
├── callbacks/
│   ├── qat_int8/
│   │   ├── plugin.yaml
│   │   └── qat_int8.py
│   ├── qat_int4/
│   │   ├── plugin.yaml
│   │   └── qat_int4.py
│   ├── qat_mixed/
│   │   ├── plugin.yaml
│   │   └── qat_mixed.py
│   ├── pruning_magnitude/
│   │   ├── plugin.yaml
│   │   └── pruning_magnitude.py
│   ├── pruning_wanda/
│   │   ├── plugin.yaml
│   │   └── pruning_wanda.py
│   ├── pruning_layer_removal/
│   │   ├── plugin.yaml
│   │   └── pruning_layer_removal.py
│   ├── pruning_attention_head/
│   │   ├── plugin.yaml
│   │   └── pruning_attention_head.py
│   ├── sensitivity_analysis/
│   │   ├── plugin.yaml
│   │   └── sensitivity_analysis.py
│   ├── distillation/
│   │   ├── plugin.yaml
│   │   └── distillation.py
│   └── compression_metrics/
│       ├── plugin.yaml
│       └── compression_metrics.py
├── objectives/
│   ├── distillation/
│   │   ├── plugin.yaml
│   │   └── distillation.py
│   └── sigreg/
│       ├── plugin.yaml
│       └── sigreg.py
├── architectures/
│   └── lewm/
│       ├── plugin.yaml
│       └── lewm.py
├── data_adapters/
│   └── trajectory_hdf5/
│       ├── plugin.yaml
│       └── trajectory_hdf5.py
└── lib/
    └── compression_utils/
        ├── plugin.yaml                  # type: lib (not auto-discovered as plugin)
        └── compression_utils.py         # shared primitives, installed to ~/.crucible-hub/lib/
```

After `crucible tap add <url> && crucible tap install qat_int8 pruning_wanda ...`, plugins land in `~/.crucible-hub/plugins/{type}/` and auto-discover.

**Shared utilities note:** The `lib/compression_utils.py` module is installed to `~/.crucible-hub/lib/` (not a plugin directory — it's not auto-discovered). Compression plugins import it via `sys.path` manipulation or inline the needed functions. The simplest approach: each plugin that needs a utility function includes it directly. If the shared code grows large enough to warrant a separate package, it can become a pip-installable `crucible-compression-utils`.

## Layer 1: Compression Plugins

### Prerequisite: Wire Missing Callback Hooks

`torch_backend.py` defines `on_step_begin` and `on_validation_end` on the base class but never calls them. Two surgical additions (~4 lines):

- Call `on_step_begin(step, state)` before `zero_grad_all()` in the training loop
- Call `on_validation_end(step, val_metrics, state)` after validation metrics are computed

### Shared Utilities (`lib/compression_utils.py`)

Model-agnostic primitives imported by compression plugins:

```python
class CompressionMetrics:
    @staticmethod
    def sparsity(model) -> dict[str, float]        # per-param + overall
    @staticmethod
    def effective_bits(model, bit_map) -> float     # weighted avg bitwidth
    @staticmethod
    def model_size_bytes(model) -> int              # param memory

def iter_prunable_layers(model, include=(), exclude=()) -> Iterator[tuple[str, Module]]
def apply_weight_mask(module, mask) -> RemovableHandle   # forward pre-hook
def remove_all_masks(model) -> None                      # for serialization
```

### Callback Priority Contract

All compression callbacks compose via `CALLBACKS` env var. Priority controls execution order:

| Priority | Callback | Phase |
|----------|----------|-------|
| 3 | `sensitivity_analysis` | Diagnostic pre-pass |
| 5 | `qat_int8` / `qat_int4` / `qat_mixed` | Fake quantization hooks |
| 8 | `pruning_magnitude` / `pruning_wanda` | Weight pruning |
| 8 | `pruning_attention_head` | Head pruning |
| 8 | `pruning_layer_removal` | Layer deletion |
| 10 | `grad_clip` (existing builtin) | Gradient clipping |
| 15 | `distillation` | Teacher-student loss |
| 20 | `nan_detector` (existing builtin) | Safety check |
| 95 | `compression_metrics` | Metric reporting |

### State-Sharing Convention

Callbacks communicate through the `state` dict:
- `state["sensitivity"]` — written by `sensitivity_analysis`, read by pruning callbacks for non-uniform sparsity allocation
- `state["qat_bit_assignments"]` — written by QAT callbacks, read by `compression_metrics`
- `state["pruning_masks"]` — written by pruning callbacks, read by `compression_metrics`
- `state["teacher_model"]` — written by `distillation`, available for inspection

### 1. QAT INT8 Callback

**Env vars:** `CALLBACKS=qat_int8`, `QAT_INT8_WARMUP_STEPS=500`, `QAT_INT8_EXCLUDE_PATTERNS=tok_emb,embed_low`

Registers forward pre-hooks on eligible `nn.Linear` modules that apply fake INT8 quantization via straight-through estimator (STE). Hooks are disabled until warmup completes.

```python
def fake_int8_quant(w: Tensor) -> Tensor:
    """STE: gradient flows through w, forward sees quantized version."""
    if w.ndim != 2:
        return w
    with torch.no_grad():
        scale = w.abs().amax(dim=1, keepdim=True) / 127.0
        scale = scale.clamp_min(1.0 / 127.0)
    q = torch.clamp(torch.round(w / scale), -128, 127)
    return w + (q * scale - w).detach()
```

Follows the exact pattern of existing `fake_int6_quant` in `quantization.py:114-122`.

### 2. QAT INT4 Callback

**Env vars:** `CALLBACKS=qat_int4`, `QAT_INT4_WARMUP_STEPS=1000`, `QAT_INT4_GROUP_SIZE=128`

Same STE pattern but range=7 (INT4 = [-8, 7]). Per-group quantization (groups of 128 weights share a scale) for accuracy at low bitwidth.

### 3. QAT Mixed-Precision Callback

**Env vars:** `CALLBACKS=qat_mixed`, `QAT_MIXED_PATTERN=encoder:8,predictor:4,lm_head:8`, `QAT_MIXED_WARMUP_STEPS=500`

Parses the pattern string into a name-prefix→bitwidth map. Walks model modules and assigns the correct fake-quant hook per module based on name matching. Critical for JEPA models where encoder sensitivity differs from predictor.

### 4. Magnitude Pruning Callback

**Env vars:** `CALLBACKS=pruning_magnitude`, `PRUNE_SPARSITY=0.3`, `PRUNE_SCHEDULE=cubic`, `PRUNE_START_STEP=500`, `PRUNE_END_STEP=15000`, `PRUNE_FREQUENCY=100`, `PRUNE_EXCLUDE_PATTERNS=tok_emb,lm_head`

Gradual magnitude pruning with cubic schedule:
```
s(t) = target_sparsity * (1 - (1 - (t - start) / (end - start))^3)
```

At `PRUNE_FREQUENCY` intervals, sorts all eligible weights by absolute magnitude, computes threshold for current sparsity target, applies binary masks via forward pre-hooks. On `on_train_end`, makes masks permanent (zeros weights, removes hooks).

### 5. Wanda Pruning Callback

**Env vars:** `CALLBACKS=pruning_wanda`, `WANDA_SPARSITY=0.5`, `WANDA_CALIBRATION_STEPS=50`, `WANDA_APPLY_AT_STEP=1000`, `WANDA_STRUCTURED=0`, `WANDA_NM_RATIO=2:4`

One-shot pruning using the Wanda score: `importance[i,j] = |W[i,j]| * ||X[:,j]||_2`

During calibration steps, registers forward hooks to accumulate input activation norms per layer. At `WANDA_APPLY_AT_STEP`, computes scores, prunes, removes activation hooks, applies permanent weight masks. Optional N:M structured sparsity (e.g., 2:4 for GPU acceleration).

### 6. Layer Removal Callback (ShortGPT-style)

**Env vars:** `CALLBACKS=pruning_layer_removal`, `LAYER_REMOVE_COUNT=2`, `LAYER_REMOVE_METRIC=angular_distance`, `LAYER_REMOVE_AT_STEP=0`

Identifies redundant layers by angular distance between input and output (layers with small angular distance are effectively identity). Physically removes layers from `nn.ModuleList` and rebuilds optimizer parameter groups via `state["optimizers"]`.

### 7. Attention Head Pruning Callback

**Env vars:** `CALLBACKS=pruning_attention_head`, `HEAD_PRUNE_RATIO=0.25`, `HEAD_PRUNE_METRIC=taylor`, `HEAD_PRUNE_AT_STEP=500`

Scores heads by Taylor importance (`gradient * activation`). Zeros out Q/K/V projection slices for pruned heads via permanent masks.

### 8. Sensitivity Analysis Callback

**Env vars:** `CALLBACKS=sensitivity_analysis`, `SENSITIVITY_METHOD=fisher`, `SENSITIVITY_STEPS=100`, `SENSITIVITY_OUTPUT_PATH=sensitivity_report.json`

Runs as a diagnostic pre-pass. Accumulates per-layer Fisher information (`gradient^2`) during the first N training steps, then writes a JSON report with per-layer importance scores and suggested non-uniform sparsity allocation. Stores results in `state["sensitivity"]` so downstream pruning callbacks can use sensitivity-informed sparsity.

Output:
```json
{
  "method": "fisher",
  "layers": [
    {"name": "encoder.blocks.0.attn.qkv", "importance": 0.85, "rank": 1, "suggested_sparsity": 0.1},
    {"name": "predictor.blocks.2.mlp.fc", "importance": 0.12, "rank": 8, "suggested_sparsity": 0.7}
  ]
}
```

### 9. Knowledge Distillation

**Callback** (for `torch_backend`): Loads a frozen teacher model at `on_train_begin`, registers a forward hook on the student model that intercepts the loss, runs the teacher on the same input, and replaces the loss with `alpha * KL(student_soft || teacher_soft) + (1-alpha) * task_loss`.

**Env vars:** `CALLBACKS=distillation`, `DISTILL_TEACHER_PATH=teacher.pt`, `DISTILL_TEACHER_FAMILY=baseline`, `DISTILL_TEMPERATURE=4.0`, `DISTILL_ALPHA=0.5`

**Objective** (for `generic_backend`): A `DistillationObjective` that stores a teacher reference and computes the combined loss in `compute()`. Uses existing `CompositeObjective` pattern when combined with other losses.

### 10. Compression Metrics Callback

**Env vars:** `CALLBACKS=compression_metrics`, `COMPRESSION_METRICS_EVERY=100`

Always include when using compression plugins. Reports at every N steps:
- `sparsity` — fraction of zeros
- `effective_bits` — weighted average bitwidth (reads `state["qat_bit_assignments"]`)
- `compression_ratio` — original_size / effective_compressed_size
- `model_size_mb` — current parameter memory
- `nonzero_params` — count of non-zero parameters

Writes final `compression_report.json` at `on_train_end`.

### 11. Compression Search Tree

Uses existing `SearchTree` infrastructure — zero new code. The search space is:

| Dimension | Values |
|-----------|--------|
| `QAT_BITS` | none, 4, 6, 8 |
| `PRUNE_SPARSITY` | 0.0, 0.1, 0.2, 0.3, 0.5, 0.7 |
| `PRUNE_METHOD` | none, magnitude, wanda |
| `LAYER_REMOVE_COUNT` | 0, 1, 2, 3 |
| `DISTILL_ALPHA` | 0.0, 0.3, 0.5, 0.7 |

Root nodes are single techniques. Children explore combinations. UCB1 policy balances exploration vs exploitation. Each node's `config` dict is a complete set of env var overrides.

Example tree creation via MCP:
```
tree_create(name="compression-lewm", primary_metric="val_bpb",
            metric_direction="minimize", expansion_policy="ucb1")

# Root: single techniques
expand_node(root, children=[
    {name: "qat-int8", config: {CALLBACKS: "qat_int8,compression_metrics"}},
    {name: "wanda-30", config: {CALLBACKS: "pruning_wanda,compression_metrics", WANDA_SPARSITY: "0.3"}},
    {name: "distill-0.5", config: {CALLBACKS: "distillation,compression_metrics", DISTILL_ALPHA: "0.5"}},
])

# After results: expand promising nodes with combinations
expand_node("qat-int8", children=[
    {name: "qat8+wanda30", config: {CALLBACKS: "qat_int8,pruning_wanda,compression_metrics", ...}},
    {name: "qat8+distill", config: {CALLBACKS: "qat_int8,distillation,compression_metrics", ...}},
])
```

## Layer 2: World Model Modality

### 1. SIGReg Objective

Replaces the VICReg-style variance hinge in the existing `JEPAObjective` with the Cramér-Wold based Gaussian regularizer from LE-WM. Mathematically cleaner, single tunable hyperparameter (lambda).

**Algorithm:**
1. Sample M random unit projections on S^{d-1}
2. Project batch embeddings Z [B, D] onto each direction
3. For each 1D projection, compute Epps-Pulley test statistic against N(0,1)
4. Average across M projections

**Env vars:** `TRAINING_OBJECTIVE=composite` with SIGReg as a component. `SIGREG_WEIGHT=1.0`, `SIGREG_PROJECTIONS=128`

Composes with existing `CompositeObjective`:
```python
CompositeObjective([
    (1.0, MSEObjective()),           # prediction loss
    (lambda_val, SIGRegObjective()), # Gaussian regularizer
])
```

The model returns both `predictions["pred_embeddings"]` (for MSE) and `predictions["embeddings"]` (for SIGReg).

### 2. Trajectory Data Adapter (HDF5)

**Env vars:** `DATA_ADAPTER=trajectory_hdf5`, `HDF5_PATH=data/pusht.hdf5`, `ACTION_DIM=2`, `NUM_FRAMES=4`, `IMAGE_SIZE=224`

Returns:
```python
{
    "frames": Tensor[B, T, 3, 224, 224],    # normalized [-1,1]
    "actions": Tensor[B, T-1, action_dim],
    "goals": Tensor[B, 3, 224, 224],        # optional
}
```

Uses `h5py` (lazy-imported). Supports pre-loading to memory for datasets that fit in RAM. Modality tag: `"world_model"`.

### 3. ViT+DiT Architecture Plugin (LE-WM)

**Env vars:** `MODEL_FAMILY=lewm`, plus `MODEL_DIM=192`, `PATCH_SIZE=16`, `ENCODER_DEPTH=12`, `PREDICTOR_DEPTH=6`, `ACTION_DIM=2`, `IMAGE_SIZE=224`

Architecture:
```
ViTEncoder (~5M params):
  PatchEmbed(3, 192, patch_size=16) -> [B, N_patches, 192]
  CLS token + learnable positional embeddings
  12 × TransformerBlock(192, heads=12)
  Extract CLS token -> [B, 192]
  BatchNorm projection (critical for SIGReg)

DiTPredictor (~10M params):
  Input: z_t [B, 192] + a_t [B, action_dim]
  Action MLP -> condition vector
  6 × DiTBlock(hidden_dim, AdaLN conditioned on action)
  Final linear -> [B, 192] (predicted z_{t+1})
```

Key LE-WM insight: **no EMA target encoder**. Encoder outputs serve directly as targets (with stop-gradient on the target side). SIGReg prevents collapse instead of EMA.

Exposes `encode(frames)` and `predict_next(z, action)` methods for inference.

**Future extensibility:**
- `build_encoder(type)` factory — swap ViT-Tiny for DINOv2, MAE, CNN
- `build_predictor(type)` factory — swap DiT for RNN, SSM (Mamba), MLP
- Both are constructor arguments now, registries later if needed

### 4. CEM Planner Evaluation

**Env vars:** `CALLBACKS=cem_eval`, `CEM_EVAL_EVERY=0` (0=end only), `CEM_EPISODES=10`, `CEM_SAMPLES=300`, `CEM_ITERATIONS=30`, `CEM_HORIZON=5`

A callback that runs Cross-Entropy Method planning at validation time. Reports `cem_success_rate` as a metric. Expensive (~1s per plan), so runs infrequently.

Standalone `CEMPlanner` class for reuse:
```python
class CEMPlanner:
    def plan(self, z_current, z_goal) -> action_sequence
```

Extensible to MPPI, random shooting via the same interface.

## Implementation Phases

### Phase 1: Foundation (in Crucible core)
1. Wire `on_step_begin` + `on_validation_end` hooks in `torch_backend.py` (~4 lines)
2. Create `compression_utils.py` shared primitives

### Phase 2: Compression Callbacks (in tap repo, independent of each other)
3. `qat_int8.py`
4. `qat_int4.py`
5. `qat_mixed.py`
6. `pruning_magnitude.py`
7. `pruning_wanda.py`
8. `pruning_layer_removal.py`
9. `pruning_attention_head.py`

### Phase 3: Distillation + Metrics (in tap repo)
10. `distillation.py` (callback + objective)
11. `compression_metrics.py`
12. `sensitivity_analysis.py`

### Phase 4: World Model Modality (in tap repo)
13. `sigreg.py` objective
14. `trajectory_hdf5.py` data adapter
15. `lewm.py` architecture
16. `cem_eval.py` callback + planner

### Phase 5: Search Tree Recipes (no code, MCP workflow)
17. Create compression search trees via MCP tools
18. Define root recipes and expansion strategy

## Verification

### Unit Tests (per plugin)
- QAT: verify STE gradient flow, quantized output range, warmup behavior
- Pruning: verify sparsity targets, mask application, schedule correctness
- Distillation: verify teacher is frozen, loss composition, temperature scaling
- SIGReg: verify gradient flow, penalizes non-Gaussian distributions, passes for Gaussian
- Trajectory adapter: verify batch shapes, normalization, HDF5 reading

### Integration Test
```bash
# Install tap
crucible tap add ./crucible-compression-tap
crucible tap install qat_int8 pruning_wanda compression_metrics

# Run compressed training (smoke preset)
PYTHONPATH=src CALLBACKS=qat_int8,pruning_wanda,compression_metrics \
QAT_INT8_WARMUP_STEPS=50 WANDA_SPARSITY=0.3 WANDA_CALIBRATION_STEPS=10 WANDA_APPLY_AT_STEP=100 \
pytest tests/ -v -k compression

# Verify compression report
python -c "import json; r=json.load(open('compression_report.json')); assert r['sparsity'] > 0.25"
```

### End-to-End (LE-WM)
```bash
crucible tap install lewm sigreg trajectory_hdf5 cem_eval

PYTHONPATH=src MODEL_FAMILY=lewm DATA_ADAPTER=trajectory_hdf5 \
TRAINING_OBJECTIVE=composite SIGREG_WEIGHT=1.0 \
CALLBACKS=qat_mixed,pruning_wanda,compression_metrics,cem_eval \
QAT_ENCODER_BITS=8 QAT_PREDICTOR_BITS=4 WANDA_SPARSITY=0.3 \
python -m crucible.training.generic_backend
```

## Challenges & Mitigations

1. **torch.compile compatibility**: Register all hooks on `base_model` BEFORE `torch.compile` wraps it. The existing QAT pattern (torch_backend.py:239-246) already does this — follow the same order.

2. **Optimizer state after layer removal**: When layers are physically deleted, rebuild optimizer parameter groups via `state["optimizers"]`.

3. **Wanda memory**: Accumulate running statistics (`mean of ||X_j||_2`) rather than storing raw activations. Release hooks after calibration.

4. **Teacher model memory for distillation**: Teacher is frozen + eval + `torch.no_grad()` so no gradient memory. For 15M LE-WM, adds ~30MB. Support `DISTILL_TEACHER_DEVICE=cpu` for larger models.

5. **CompositeObjective key routing**: SIGReg needs `predictions["embeddings"]`, MSE needs `predictions["pred_embeddings"]`. Model must return both — natural for JEPA architectures.
