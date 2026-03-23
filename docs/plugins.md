---
layout: default
title: Architecture Plugins
---

# Architecture Plugins

Crucible ships with 4 built-in transformer architectures (baseline, looped, convloop, prefix_memory). Everything else is a **plugin** — created declaratively via YAML specs or as Python code.

## Two Ways to Create Architectures

### Option A: Declarative Composition (Recommended)

Compose from known components via YAML specs — no Python code needed. Uses the `model_compose` MCP tool.

**Available building blocks:**

| Stack Patterns | Block Types | Augmentations |
|---------------|-------------|---------------|
| `sequential` — linear pass | `attention_block` — standard transformer | `smear_gate` — previous-token gating |
| `encoder_decoder_skip` — U-Net skips (baseline) | `prefix_memory_block` — bounded memory | `bigram_hash` — token-pair embeddings |
| `looped` — weight-shared iteration | | `trigram_hash` — token-triple embeddings |
| `prefix_memory_stack` — sequential + step scales | | |

**MCP workflow:**

```
1. model_list_stack_patterns()     → see available wiring patterns
2. model_list_block_types()        → see available blocks
3. model_compose(name, spec)       → create .crucible/architectures/my_arch.yaml
4. design_enqueue_batch(...)       → run with MODEL_FAMILY: my_arch
```

**Example: Looped + Augmented (novel hybrid)**

The built-in `looped` architecture has no augmentations. Compose a hybrid that adds BigramHash + SmearGate:

```yaml
# .crucible/architectures/looped_augmented.yaml
name: looped_augmented
version: 1
base: tied_embedding_lm

embedding:
  vocab_size: "{VOCAB_SIZE:50304}"
  model_dim: "{MODEL_DIM:512}"
  tie_embeddings: "{TIE_EMBEDDINGS:true}"
  tied_embed_init_std: "{TIED_EMBED_INIT_STD:0.02}"
  logit_softcap: "{LOGIT_SOFTCAP:30.0}"

block:
  type: attention_block
  dim: "{MODEL_DIM:512}"
  params:
    num_heads: "{NUM_HEADS:8}"
    num_kv_heads: "{NUM_KV_HEADS:4}"
    mlp_mult: "{MLP_MULT:2}"
    activation: "{ACTIVATION:relu_sq}"

stack:
  pattern: looped
  logical_steps: "{RECURRENCE_STEPS:12}"
  unique_blocks: "{SHARE_BLOCKS:3}"

augmentations:
  smear_gate:
    enabled: "{SMEAR_GATE:true}"
    dim: "{MODEL_DIM:512}"
  bigram_hash:
    enabled: "{BIGRAM_HASH:true}"
    vocab_size: "{VOCAB_SIZE:50304}"
    num_buckets: "{BIGRAM_HASH_BUCKETS:4096}"
    embed_dim: "{BIGRAM_HASH_EMBED_DIM:128}"
    model_dim: "{MODEL_DIM:512}"
```

Template variables like `"{VOCAB_SIZE:50304}"` resolve from experiment config env vars at build time.

**Forking existing specs:**

Use `model_from_template` to fork an existing architecture and override specific fields:

```json
{
  "tool": "model_from_template",
  "arguments": {
    "name": "wide_baseline",
    "base": "baseline",
    "overrides": {
      "block": { "params": { "mlp_mult": "{MLP_MULT:4}" } }
    }
  }
}
```

### Option B: Python Plugin (for novel forward logic)

When you need custom forward passes that YAML can't express.

1. `model_generate_template(name="my_arch")` — get boilerplate
2. Edit the code to implement your architecture
3. `model_add_architecture(name="my_arch", code="...")` — saves to `.crucible/architectures/my_arch.py`

**The contract:** A Python plugin must call `register_model(name, factory_fn)` where the factory takes an `args` namespace and returns an `nn.Module`.

**Available components to reuse:**

```python
from crucible.models.components.attention import Block
from crucible.models.components.mlp import MLP
from crucible.models.components.norm import RMSNorm
from crucible.models.components.linear import CastedLinear
from crucible.models.components.conv import DepthwiseConv1D
from crucible.models.components.gate import SmearGate
from crucible.models.components.moe import MoELayer
from crucible.models.components.rotary import Rotary
from crucible.models.components.memory import CausalPrefixMemory
from crucible.models.components.hash_embed import BigramHash, TrigramHash
```

See `src/crucible/models/user_architectures/example_two_tower.py` for a complete working example.

## Plugin Discovery (3-tier precedence)

| Tier | Location | Precedence |
|------|----------|-----------|
| **Builtin** | `src/crucible/models/architectures/` | Lowest |
| **Global (hub)** | `~/.crucible-hub/architectures/plugins/` | Medium |
| **Local (project)** | `.crucible/architectures/` | Highest |

Both `.py` and `.yaml` files are auto-discovered. At the same scope, `.py` takes precedence over `.yaml`.

**Hub promotion:** Use `model_promote_architecture` to share a local plugin across projects.
**Hub import:** Use `model_import_architecture` to pull a hub plugin into your project.

## Important Notes

- **Plugins sync to pods automatically** — `.crucible/architectures/` is included in rsync
- **Higher precedence overrides lower** — a local plugin named `baseline` overrides the built-in
- **YAML specs need no torch** — they're interpreted at runtime by the `ComposedArchitecture` class
- **Python plugin errors propagate** — syntax errors surface as real tracebacks
