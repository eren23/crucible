---
layout: default
title: Architecture Plugins
---

# Architecture Plugins

Crucible ships with 4 built-in transformer architectures (baseline, looped, convloop, prefix_memory). Everything else is a **plugin** — a Python file you drop into `models/user_architectures/`.

## How It Works

1. Any `.py` file in `src/crucible/models/user_architectures/` is auto-discovered on import
2. Each plugin calls `register_model(name, factory_fn)` to register itself
3. The training script falls back to the Crucible registry for unknown `MODEL_FAMILY` values
4. Plugins get rsynced to pods automatically — they work on remote GPUs

## Writing a Plugin

### The Contract

Your plugin must export a **factory function** that:
- Takes an `args` namespace (with `vocab_size`, `model_dim`, `num_layers`, etc.)
- Returns an `nn.Module` that extends `TiedEmbeddingLM`

### Available Args

The training script provides these attributes on the `args` namespace:

| Arg | Type | Description |
|-----|------|-------------|
| `vocab_size` | int | Vocabulary size |
| `model_dim` | int | Model embedding dimension |
| `num_layers` | int | Number of transformer layers |
| `num_heads` | int | Number of attention heads |
| `num_kv_heads` | int | Number of key/value heads (for GQA) |
| `mlp_mult` | int | MLP expansion factor |
| `rope_base` | float | RoPE base frequency |
| `qk_gain_init` | float | QK initialization gain |
| `tie_embeddings` | bool | Whether to tie input/output embeddings |
| `tied_embed_init_std` | float | Embedding initialization std |
| `logit_softcap` | float | Logit soft-capping value |
| `embed_bottleneck_dim` | int | Embedding bottleneck (0 = none) |
| `attention_variant` | str | Attention variant name |
| `residual_variant` | str | Residual connection variant |
| `model_family` | str | The family name (your plugin's name) |
| `activation` | str | Activation function name |

Additional args may be available depending on the experiment config — use `getattr(args, 'my_param', default)` for custom parameters.

### Available Components

Reuse these from `crucible.models.components`:

```python
from crucible.models.components.attention import Block      # Full transformer block
from crucible.models.components.mlp import MLP              # Feed-forward network
from crucible.models.components.norm import RMSNorm          # RMS normalization
from crucible.models.components.linear import CastedLinear   # Float32-cast linear
from crucible.models.components.conv import DepthwiseConv1D  # Causal convolution
from crucible.models.components.gate import SmearGate        # Gated residual
from crucible.models.components.moe import MoELayer          # Mixture of Experts
from crucible.models.components.rotary import Rotary         # RoPE embeddings
from crucible.models.components.memory import CausalPrefixMemory  # Bounded memory
```

### Example: Two-Tower Architecture

See `src/crucible/models/user_architectures/example_two_tower.py` for a complete working example.

```python
"""Two parallel transformer towers with gated fusion."""
from __future__ import annotations
from typing import Any
import torch
from torch import Tensor, nn

from crucible.models.base import TiedEmbeddingLM
from crucible.models.registry import register_model
from crucible.models.components.attention import Block


class TwoTowerLM(TiedEmbeddingLM):
    def __init__(self, vocab_size, model_dim, num_layers, num_heads,
                 num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                 tie_embeddings, tied_embed_init_std, logit_softcap, **kw):
        super().__init__(vocab_size, model_dim, tie_embeddings,
                         tied_embed_init_std, logit_softcap)
        self.tower_a = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult,
                  rope_base, qk_gain_init)
            for _ in range(num_layers)
        ])
        self.tower_b = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult,
                  rope_base, qk_gain_init)
            for _ in range(num_layers)
        ])
        self.fusion_gate = nn.Parameter(torch.zeros(model_dim))

    def hidden(self, input_ids, lora=None):
        x = self.embed_tokens(input_ids)
        a, b = x, x
        for block in self.tower_a:
            a = block(a, x)
        for block in self.tower_b:
            b = block(b, x)
        gate = torch.sigmoid(self.fusion_gate).to(dtype=x.dtype)
        return gate * a + (1 - gate) * b


def _build(args):
    return TwoTowerLM(
        vocab_size=args.vocab_size, model_dim=args.model_dim,
        num_layers=args.num_layers, num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
    )


register_model("two_tower", _build)
```

## Using a Plugin in Experiments

Once your plugin is in `user_architectures/`, use it in experiment designs:

```yaml
# .crucible/designs/my-two-tower/current.yaml
config:
  MODEL_FAMILY: "two_tower"
  NUM_LAYERS: "6"
  MODEL_DIM: "512"
  NUM_HEADS: "8"
  NUM_KV_HEADS: "4"
```

Or via MCP:
```json
{
  "tool": "version_save_design",
  "arguments": {
    "name": "two-tower-exp",
    "config": {"MODEL_FAMILY": "two_tower", "NUM_LAYERS": "6"},
    "hypothesis": "Two parallel towers may capture different feature patterns"
  }
}
```

## MCP Tools for Plugins

- `model_generate_template(name="my_arch")` — Generate boilerplate for a new plugin
- `model_add_architecture(name="my_arch", code="...")` — Save code to user_architectures/ and register
- `model_list_families()` — See all registered families (built-in + plugins)
- `model_validate_config(family="my_arch", config={...})` — Validate config against schema

## Important Notes

- **Plugins sync to pods automatically** — `user_architectures/` is not in `sync_excludes`
- **Built-in families take priority** — If your plugin name conflicts with baseline/looped/convloop/prefix_memory, the built-in wins (they're hardcoded in the training script dispatch)
- **Plugin errors propagate** — Syntax errors or import failures surface as real tracebacks, not silent fallbacks
- **No core modifications needed** — Never add architectures to `models/architectures/`. Use `user_architectures/` exclusively.
