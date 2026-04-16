"""Declarative architecture composer.

Interprets YAML architecture specs and builds PyTorch models from them,
using the same components as hand-written architectures but without
requiring Python code.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.core.errors import ComposerError
from crucible.core.types import PluginFactory


# ---------------------------------------------------------------------------
# Template variable resolution
# ---------------------------------------------------------------------------

_VAR_PATTERN = re.compile(r"\{([A-Z_][A-Z0-9_]*)(?::([^}]*))?\}")


def _coerce_value(raw: str) -> bool | int | float | str:
    """Coerce a resolved string to the most specific Python type."""
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    try:
        return int(raw)
    except (ValueError, TypeError):
        pass
    try:
        return float(raw)
    except (ValueError, TypeError):
        pass
    return raw


def _resolve_value(value: Any, variables: dict[str, Any]) -> Any:
    """Resolve template ``{VAR:default}`` references in *value*.

    - Strings are scanned for ``{VAR}`` or ``{VAR:default}`` patterns.
    - If the entire string is a single template reference, the resolved
      value is type-coerced.  If mixed with other text, the result stays
      a string.
    - Dicts and lists are resolved recursively.
    - All other types pass through unchanged.
    """
    if isinstance(value, str):
        # Check if the entire string is exactly one template reference
        m = _VAR_PATTERN.fullmatch(value)
        if m:
            var_name, default = m.group(1), m.group(2)
            raw = variables.get(var_name)
            if raw is None:
                if default is not None:
                    return _coerce_value(default)
                raise ComposerError(f"Unresolved template variable {{{var_name}}} with no default")
            # If variable value is already typed (int, bool, etc.), use it directly
            if not isinstance(raw, str):
                return raw
            return _coerce_value(raw)
        # Mixed template: substitute inline
        def _sub(m: re.Match) -> str:
            var_name, default = m.group(1), m.group(2)
            raw = variables.get(var_name)
            if raw is None:
                if default is not None:
                    return default
                raise ComposerError(f"Unresolved template variable {{{var_name}}} with no default")
            return str(raw)
        return _VAR_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _resolve_value(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_value(item, variables) for item in value]
    return value


# ---------------------------------------------------------------------------
# Spec dataclass
# ---------------------------------------------------------------------------

@dataclass
class ArchitectureSpec:
    """Parsed YAML architecture specification."""
    name: str
    version: int = 1
    base: str = "tied_embedding_lm"
    embedding: dict = field(default_factory=dict)
    block: dict = field(default_factory=dict)
    stack: dict = field(default_factory=dict)
    transform: dict | None = None
    init: dict | None = None
    augmentations: dict | None = None

    @classmethod
    def from_dict(cls, d: dict) -> ArchitectureSpec:
        """Create an ArchitectureSpec from a parsed YAML dict."""
        if "name" not in d:
            raise ComposerError("Architecture spec must include a 'name' field")
        return cls(
            name=d["name"],
            version=d.get("version", 1),
            base=d.get("base", "tied_embedding_lm"),
            embedding=d.get("embedding", {}),
            block=d.get("block", {}),
            stack=d.get("stack", {}),
            transform=d.get("transform"),
            init=d.get("init"),
            augmentations=d.get("augmentations"),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ArchitectureSpec:
        """Load an ArchitectureSpec from a YAML file."""
        import yaml
        path = Path(path)
        if not path.exists():
            raise ComposerError(f"Spec file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ComposerError(f"Spec file must contain a YAML mapping, got {type(data).__name__}")
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Resolved spec
# ---------------------------------------------------------------------------

@dataclass
class ResolvedSpec:
    """Fully resolved architecture specification with concrete values."""
    name: str
    version: int
    base: str
    embedding: dict
    block: dict
    stack: dict
    transform: dict | None
    init: dict | None
    augmentations: dict | None


class SpecResolver:
    """Resolves template variables in an ArchitectureSpec against an args namespace."""

    def __init__(self, spec: ArchitectureSpec, args: Any):
        self.spec = spec
        self.args = args

    def _build_variables(self) -> dict[str, Any]:
        """Extract template variables from the args namespace.

        Converts attribute names to uppercase for matching against
        ``{VAR_NAME}`` patterns in specs.
        """
        variables: dict[str, Any] = {}
        for attr in dir(self.args):
            if attr.startswith("_"):
                continue
            val = getattr(self.args, attr, None)
            if callable(val):
                continue
            variables[attr.upper()] = val
        return variables

    def resolve(self) -> ResolvedSpec:
        """Resolve all template references and return a ResolvedSpec."""
        variables = self._build_variables()
        return ResolvedSpec(
            name=self.spec.name,
            version=self.spec.version,
            base=self.spec.base,
            embedding=_resolve_value(copy.deepcopy(self.spec.embedding), variables),
            block=_resolve_value(copy.deepcopy(self.spec.block), variables),
            stack=_resolve_value(copy.deepcopy(self.spec.stack), variables),
            transform=_resolve_value(copy.deepcopy(self.spec.transform), variables) if self.spec.transform else None,
            init=_resolve_value(copy.deepcopy(self.spec.init), variables) if self.spec.init else None,
            augmentations=_resolve_value(copy.deepcopy(self.spec.augmentations), variables) if self.spec.augmentations else None,
        )


# ---------------------------------------------------------------------------
# Block type registry (PluginRegistry-backed)
# ---------------------------------------------------------------------------

from crucible.core.plugin_registry import PluginRegistry

BLOCK_TYPE_REGISTRY = PluginRegistry("block_type")
BLOCK_TYPES: dict[str, PluginFactory] = BLOCK_TYPE_REGISTRY._registry  # convenience alias


def _ensure_block_types() -> None:
    """Lazily populate block types on first use (avoids top-level torch import)."""
    if BLOCK_TYPE_REGISTRY._registry:
        return
    from crucible.models.components.attention import Block
    from crucible.models.components.memory import PrefixMemoryBlock
    BLOCK_TYPE_REGISTRY.register("attention_block", Block, source="builtin")
    BLOCK_TYPE_REGISTRY.register("prefix_memory_block", PrefixMemoryBlock, source="builtin")


# ---------------------------------------------------------------------------
# Augmentation registry (PluginRegistry-backed)
# ---------------------------------------------------------------------------

AUGMENTATION_REGISTRY = PluginRegistry("augmentation")
AUGMENTATIONS: dict[str, PluginFactory] = AUGMENTATION_REGISTRY._registry  # convenience alias


def _ensure_augmentations() -> None:
    """Lazily populate augmentations on first use."""
    if AUGMENTATION_REGISTRY._registry:
        return
    from crucible.models.components.gate import SmearGate
    from crucible.models.components.hash_embed import BigramHash, TrigramHash
    AUGMENTATION_REGISTRY.register("smear_gate", SmearGate, source="builtin")
    AUGMENTATION_REGISTRY.register("bigram_hash", BigramHash, source="builtin")
    AUGMENTATION_REGISTRY.register("trigram_hash", TrigramHash, source="builtin")


# ---------------------------------------------------------------------------
# Stack patterns
# ---------------------------------------------------------------------------

class StackPattern:
    """Base class for stack wiring patterns."""

    def build(self, resolved: ResolvedSpec, args: Any) -> dict:
        """Return extra nn.Modules/Parameters the pattern needs.

        Returns a dict of name -> nn.Module/nn.Parameter to be registered
        on the parent model.
        """
        raise NotImplementedError

    def forward(self, x: Any, x0: Any, blocks: Any, extra: dict, lora: Any = None) -> Any:
        """Execute the stack forward pass.

        *x* is the current hidden state, *x0* is the initial embeddings,
        *blocks* is the nn.ModuleList of blocks, *extra* is the dict
        returned by ``build()``, and *lora* is an optional LoRA adapter.
        """
        raise NotImplementedError


class SequentialPattern(StackPattern):
    """Simple linear pass through all blocks."""

    def build(self, resolved: ResolvedSpec, args: Any) -> dict:
        return {}

    def forward(self, x: Any, x0: Any, blocks: Any, extra: dict, lora: Any = None) -> Any:
        for i, block in enumerate(blocks):
            qd = lora.q_loras[i] if lora else None
            vd = lora.v_loras[i] if lora else None
            x = block(x, x0, qd, vd)
        return x


class EncoderDecoderSkipPattern(StackPattern):
    """Encoder-decoder with learned skip connections (from baseline.py)."""

    def build(self, resolved: ResolvedSpec, args: Any) -> dict:
        import torch
        from torch import nn

        num_layers = len(resolved.stack.get("layers", [])) if "layers" in resolved.stack else resolved.stack.get("num_layers", getattr(args, "num_layers", 9))
        dim = resolved.block.get("dim", getattr(args, "model_dim", 512))
        num_encoder = num_layers // 2
        num_decoder = num_layers - num_encoder
        return {
            "skip_weights": nn.Parameter(torch.ones(min(num_encoder, num_decoder), dim, dtype=torch.float32)),
            "_num_encoder": num_encoder,
            "_num_decoder": num_decoder,
        }

    def forward(self, x: Any, x0: Any, blocks: Any, extra: dict, lora: Any = None) -> Any:
        num_encoder = extra["_num_encoder"]
        num_decoder = extra["_num_decoder"]
        skip_weights = extra["skip_weights"]
        token_merger = extra.get("_token_merger")
        token_merge_layer = extra.get("_token_merge_layer", 0)

        skips: list = []
        for i in range(num_encoder):
            qd = lora.q_loras[i] if lora else None
            vd = lora.v_loras[i] if lora else None
            x = blocks[i](x, x0, qd, vd)
            if token_merger is not None and i + 1 == token_merge_layer:
                x = token_merger(x)
            skips.append(x)
        for i in range(num_decoder):
            if skips:
                x = x + skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            bi = num_encoder + i
            qd = lora.q_loras[bi] if lora else None
            vd = lora.v_loras[bi] if lora else None
            x = blocks[bi](x, x0, qd, vd)
            if token_merger is not None and bi + 1 == token_merge_layer:
                x = token_merger(x)
        return x


class LoopedPattern(StackPattern):
    """Looped iteration over shared blocks with step scales (from looped.py)."""

    def build(self, resolved: ResolvedSpec, args: Any) -> dict:
        import torch
        from torch import nn

        stack = resolved.stack
        logical_steps = stack.get("logical_steps", 0)
        # Mirror Python builder: if recurrence_steps <= 0, fall back to num_layers
        if logical_steps <= 0:
            logical_steps = getattr(args, "num_layers", 9)
        dim = resolved.block.get("dim", getattr(args, "model_dim", 512))
        return {
            "step_scales": nn.Parameter(torch.ones(logical_steps, dim, dtype=torch.float32)),
            "_logical_steps": logical_steps,
        }

    def forward(self, x: Any, x0: Any, blocks: Any, extra: dict, lora: Any = None) -> Any:
        logical_steps = extra["_logical_steps"]
        step_scales = extra["step_scales"]
        num_blocks = len(blocks)
        for step in range(logical_steps):
            x_next = blocks[step % num_blocks](x, x0)
            x = x + step_scales[step].to(dtype=x.dtype)[None, None, :] * (x_next - x)
        return x


class PrefixMemoryStackPattern(StackPattern):
    """Sequential with step scales for PrefixMemoryBlock (from prefix_memory.py)."""

    def build(self, resolved: ResolvedSpec, args: Any) -> dict:
        import torch
        from torch import nn

        num_layers = resolved.stack.get("num_layers", getattr(args, "num_layers", 9))
        dim = resolved.block.get("dim", getattr(args, "model_dim", 512))
        return {
            "step_scales": nn.Parameter(torch.ones(num_layers, dim, dtype=torch.float32)),
            "_num_layers": num_layers,
        }

    def forward(self, x: Any, x0: Any, blocks: Any, extra: dict, lora: Any = None) -> Any:
        step_scales = extra["step_scales"]
        for step, block in enumerate(blocks):
            x_next = block(x, x0)
            x = x + step_scales[step].to(dtype=x.dtype)[None, None, :] * (x_next - x)
        return x


STACK_PATTERN_REGISTRY = PluginRegistry("stack_pattern")
STACK_PATTERN_REGISTRY.register("sequential", SequentialPattern(), source="builtin")
STACK_PATTERN_REGISTRY.register("encoder_decoder_skip", EncoderDecoderSkipPattern(), source="builtin")
STACK_PATTERN_REGISTRY.register("looped", LoopedPattern(), source="builtin")
STACK_PATTERN_REGISTRY.register("prefix_memory_stack", PrefixMemoryStackPattern(), source="builtin")
STACK_PATTERNS: dict[str, StackPattern] = STACK_PATTERN_REGISTRY._registry  # convenience alias


# ---------------------------------------------------------------------------
# ComposedArchitecture
# ---------------------------------------------------------------------------

class ComposedArchitecture:
    """nn.Module built from a resolved YAML spec.

    This is a factory function disguised as a class -- calling
    ``ComposedArchitecture(resolved_spec, args)`` returns a
    ``_ComposedModel`` instance (a ``TiedEmbeddingLM`` subclass).
    We use this indirection so the class can be referenced by name in
    docstrings while keeping the actual model as a proper ``nn.Module``.
    """

    def __new__(cls, resolved: ResolvedSpec, args: Any) -> Any:
        return _build_composed_model(resolved, args)


def _build_blocks(
    resolved_spec: ResolvedSpec,
    build_args: Any,
    pattern_name: str,
    model_dim: int,
) -> tuple[Any, int]:
    """Build a ModuleList of blocks from the resolved spec.

    Returns ``(blocks, num_blocks_built)`` where *blocks* is a
    ``torch.nn.ModuleList``.  Factored out so both ``TiedEmbeddingLM``
    and ``CrucibleModel`` composers can share it.
    """
    from torch import nn
    from crucible.models.components.attention import _parse_block_pattern

    block_cfg = resolved_spec.block
    block_type_name = block_cfg.get("type", "attention_block")
    if block_type_name not in BLOCK_TYPES:
        raise ComposerError(f"Unknown block type {block_type_name!r}; available: {list(BLOCK_TYPES)}")
    block_cls = BLOCK_TYPES[block_type_name]

    block_dim = block_cfg.get("dim", model_dim)
    block_params = block_cfg.get("params", {})
    stack_cfg = resolved_spec.stack

    # Determine number of blocks to instantiate
    if pattern_name in ("looped",):
        logical_steps = stack_cfg.get("logical_steps", 0)
        if logical_steps <= 0:
            logical_steps = getattr(build_args, "num_layers", 9)
        unique_blocks = stack_cfg.get("unique_blocks", logical_steps)
        unique_blocks = max(1, min(unique_blocks, logical_steps))
        num_blocks_to_build = unique_blocks
    elif pattern_name == "prefix_memory_stack":
        num_blocks_to_build = stack_cfg.get("num_layers", getattr(build_args, "num_layers", 9))
    elif pattern_name == "encoder_decoder_skip":
        num_blocks_to_build = stack_cfg.get("num_layers", getattr(build_args, "num_layers", 9))
    else:
        num_blocks_to_build = stack_cfg.get("num_layers", getattr(build_args, "num_layers", 9))

    if block_type_name == "attention_block":
        block_pattern = block_params.get("block_pattern", "")
        _lw = _parse_block_pattern(block_pattern, num_blocks_to_build) if block_pattern else [0] * num_blocks_to_build

        blocks = nn.ModuleList([
            block_cls(
                block_dim,
                block_params.get("num_heads", getattr(build_args, "num_heads", 8)),
                block_params.get("num_kv_heads", getattr(build_args, "num_kv_heads", 4)),
                block_params.get("mlp_mult", getattr(build_args, "mlp_mult", 2)),
                block_params.get("rope_base", getattr(build_args, "rope_base", 10000.0)),
                block_params.get("qk_gain_init", getattr(build_args, "qk_gain_init", 1.0)),
                block_params.get("attention_variant", getattr(build_args, "attention_variant", "standard")),
                block_params.get("residual_variant", getattr(build_args, "residual_variant", "standard")),
                use_conv=block_params.get("use_conv", False),
                conv_kernel=block_params.get("conv_kernel", 3),
                multiscale_window=block_params.get("multiscale_window", 0) if _lw[i] == 0 else 0,
                attention_window=_lw[i],
                activation=block_params.get("activation", getattr(build_args, "activation", "relu_sq")),
                use_moe=block_params.get("use_moe", False),
                moe_num_experts=block_params.get("moe_num_experts", 4),
                moe_top_k=block_params.get("moe_top_k", 2),
            )
            for i in range(num_blocks_to_build)
        ])
    elif block_type_name == "prefix_memory_block":
        state_dim = block_params.get("state_dim", getattr(build_args, "state_dim", model_dim))
        blocks = nn.ModuleList([
            block_cls(
                block_dim,
                state_dim,
                block_params.get("mlp_mult", getattr(build_args, "mlp_mult", 2)),
                block_params.get("residual_variant", getattr(build_args, "residual_variant", "standard")),
                activation=block_params.get("activation", getattr(build_args, "activation", "relu_sq")),
            )
            for _ in range(num_blocks_to_build)
        ])
    else:
        raise ComposerError(f"Don't know how to construct block type {block_type_name!r}")

    return blocks, num_blocks_to_build


def _build_stack_extras(
    model: Any,
    pattern: StackPattern,
    pattern_name: str,
    resolved_spec: ResolvedSpec,
    build_args: Any,
) -> dict:
    """Build and register stack pattern extras on *model*."""
    from torch import nn

    stack_extra = pattern.build(resolved_spec, build_args)
    for key, val in stack_extra.items():
        if key.startswith("_"):
            continue
        if isinstance(val, nn.Parameter):
            model.register_parameter(f"stack_{key}", val)
            stack_extra[key] = val
        elif isinstance(val, nn.Module):
            model.add_module(f"stack_{key}", val)
            stack_extra[key] = val
    return stack_extra


def _build_composed_model(resolved: ResolvedSpec, args: Any) -> Any:
    """Build a composed model from a resolved spec + args namespace."""
    import math
    import torch
    from torch import nn

    from crucible.models.base import TiedEmbeddingLM

    _ensure_block_types()
    _ensure_augmentations()

    # ----- Resolve stack pattern -----
    pattern_name = resolved.stack.get("pattern", "sequential")
    if pattern_name not in STACK_PATTERNS:
        raise ComposerError(f"Unknown stack pattern {pattern_name!r}; available: {list(STACK_PATTERNS)}")
    pattern = STACK_PATTERNS[pattern_name]

    base = resolved.base

    # ----- CrucibleModel path (no embedding/lm_head) -----
    if base == "crucible_model":
        from crucible.models.base import CrucibleModel
        from crucible.models.components.norm import RMSNorm

        class _ComposedGenericModel(CrucibleModel):
            def __init__(self, resolved_spec: ResolvedSpec, build_args: Any):
                super().__init__()
                self._resolved = resolved_spec
                self._pattern = pattern
                self._pattern_name = pattern_name

                block_cfg = resolved_spec.block
                model_dim = block_cfg.get("dim", getattr(build_args, "model_dim", 512))
                self._model_dim = model_dim

                self.blocks, num_blocks_built = _build_blocks(
                    resolved_spec, build_args, pattern_name, model_dim
                )
                self._stack_extra = _build_stack_extras(
                    self, pattern, pattern_name, resolved_spec, build_args
                )
                self.final_norm = RMSNorm()

                if resolved_spec.init:
                    if resolved_spec.init.get("ortho", False):
                        self._apply_ortho_init(num_blocks_built)

            def _apply_ortho_init(self, num_layers: int) -> None:
                for name, p in self.named_parameters():
                    if p.ndim == 2 and p.numel() > 256:
                        nn.init.orthogonal_(p)
                        if 'proj' in name:
                            p.data *= 1.0 / math.sqrt(2 * num_layers)

            def forward(self, **batch) -> dict:
                x = batch.get("input")
                if x is None:
                    raise ComposerError(
                        "CrucibleModel composed forward expects 'input' key in batch"
                    )
                x0 = x
                x = self._pattern.forward(x, x0, self.blocks, self._stack_extra)
                x = self.final_norm(x)
                output = {"output": x}
                if "target" in batch and "loss_fn" in batch:
                    output["loss"] = batch["loss_fn"](x, batch["target"])
                return output

        return _ComposedGenericModel(resolved, args)

    # ----- Default: TiedEmbeddingLM path -----
    class _ComposedModel(TiedEmbeddingLM):
        def __init__(self, resolved_spec: ResolvedSpec, build_args: Any):
            # Extract embedding params
            emb = resolved_spec.embedding
            vocab_size = emb.get("vocab_size", getattr(build_args, "vocab_size", 50304))
            model_dim = emb.get("model_dim", getattr(build_args, "model_dim", 512))
            tie_embeddings = emb.get("tie_embeddings", getattr(build_args, "tie_embeddings", True))
            tied_embed_init_std = emb.get("tied_embed_init_std", getattr(build_args, "tied_embed_init_std", 0.02))
            logit_softcap = emb.get("logit_softcap", getattr(build_args, "logit_softcap", 30.0))
            embed_bottleneck_dim = emb.get("embed_bottleneck_dim", getattr(build_args, "embed_bottleneck_dim", 0))
            spectral_embed_init = emb.get("spectral_embed_init", getattr(build_args, "spectral_embed_init", False))

            super().__init__(
                vocab_size, model_dim, tie_embeddings, tied_embed_init_std,
                logit_softcap, embed_bottleneck_dim, spectral_embed_init,
            )

            self._resolved = resolved_spec
            self._pattern = pattern
            self._pattern_name = pattern_name

            # ----- Build blocks -----
            block_cfg = resolved_spec.block
            block_dim = block_cfg.get("dim", model_dim)

            self.blocks, num_blocks_to_build = _build_blocks(
                resolved_spec, build_args, pattern_name, model_dim
            )

            # ----- Build stack pattern extras -----
            self._stack_extra = _build_stack_extras(
                self, pattern, pattern_name, resolved_spec, build_args
            )

            # ----- Transform (pre/post stack) -----
            self._has_transform = resolved_spec.transform is not None
            if self._has_transform:
                tc = resolved_spec.transform
                pre = tc.get("pre_stack", {})
                post = tc.get("post_stack", {})

                # Pre-stack: compress
                if pre.get("type") == "compress":
                    from crucible.models.components.conv import FeatureConvBottleneck
                    in_dim = pre.get("in_dim", model_dim)
                    out_dim = pre.get("out_dim", block_dim)
                    self.compress = FeatureConvBottleneck(in_dim, out_dim)

                # Post-stack: expand
                if post.get("type") == "expand":
                    from crucible.models.components.linear import CastedLinear
                    in_dim = post.get("in_dim", block_dim)
                    out_dim = post.get("out_dim", model_dim)
                    self.expand = CastedLinear(in_dim, out_dim, bias=False)

                self._pre_transform_type = pre.get("type")
                self._post_transform_type = post.get("type")
                self._post_residual = post.get("residual", False)
            else:
                self._pre_transform_type = None
                self._post_transform_type = None
                self._post_residual = False

            # ----- Augmentations -----
            self._aug_order: list[str] = []
            aug_cfg = resolved_spec.augmentations
            if aug_cfg:
                for aug_name, aug_params in aug_cfg.items():
                    if not aug_params.get("enabled", False):
                        continue
                    if aug_name not in AUGMENTATIONS:
                        raise ComposerError(f"Unknown augmentation {aug_name!r}; available: {list(AUGMENTATIONS)}")
                    aug_cls = AUGMENTATIONS[aug_name]
                    params = {k: v for k, v in aug_params.items() if k != "enabled"}
                    aug_mod = aug_cls(**params)
                    setattr(self, f"aug_{aug_name}", aug_mod)
                    self._aug_order.append(aug_name)

            # ----- Token merger (baseline-specific augmentation) -----
            if aug_cfg and "token_merge" in aug_cfg:
                tm_cfg = aug_cfg["token_merge"]
                if tm_cfg.get("enabled", False):
                    from crucible.models.components.merge import TokenMerger
                    self._token_merger = TokenMerger(tm_cfg.get("threshold", 0.9))
                    self._token_merge_layer = tm_cfg.get("layer", 0)
                    if pattern_name == "encoder_decoder_skip":
                        self._stack_extra["_token_merger"] = self._token_merger
                        self._stack_extra["_token_merge_layer"] = self._token_merge_layer
                else:
                    self._token_merger = None
                    self._token_merge_layer = 0
            else:
                self._token_merger = None
                self._token_merge_layer = 0

            # ----- Init -----
            if resolved_spec.init:
                init_cfg = resolved_spec.init
                if init_cfg.get("ortho", False):
                    self._apply_ortho_init(num_blocks_to_build)

        def _apply_ortho_init(self, num_layers: int) -> None:
            skip_patterns = ('tok_emb', 'embed_low', 'embed_proj', 'lm_head', 'bigram_hash', 'trigram_hash', 'smear_gate')
            for name, p in self.named_parameters():
                if p.ndim == 2 and p.numel() > 256 and not any(pat in name for pat in skip_patterns):
                    nn.init.orthogonal_(p)
                    if 'proj' in name:
                        p.data *= 1.0 / math.sqrt(2 * num_layers)

        def hidden(self, input_ids, lora=None):
            x = self.embed_tokens(input_ids)

            # Augmentations (pre-embedding transforms)
            for aug_name in self._aug_order:
                aug_mod = getattr(self, f"aug_{aug_name}")
                if aug_name == "smear_gate":
                    x = aug_mod(x)
                elif aug_name == "bigram_hash":
                    prev_ids = torch.cat([input_ids[:, :1], input_ids[:, :-1]], dim=1)
                    x = x + aug_mod(prev_ids, input_ids)
                elif aug_name == "trigram_hash":
                    x = x + aug_mod(input_ids)

            # Pre-stack transform (e.g., convloop compress)
            if self._has_transform and self._pre_transform_type == "compress":
                x_wide = x
                x = self.compress(x)

            x0 = x

            # Stack forward
            x = self._pattern.forward(x, x0, self.blocks, self._stack_extra, lora=lora)

            # Post-stack transform (e.g., convloop expand + residual)
            if self._has_transform and self._post_transform_type == "expand":
                x = self.expand(x)
                if self._post_residual:
                    x = x + x_wide

            return x

    model = _ComposedModel(resolved, args)
    return model


# ---------------------------------------------------------------------------
# Public API: register a spec as a model family
# ---------------------------------------------------------------------------

def register_from_spec(
    name: str,
    spec_path_or_dict: str | Path | dict,
    *,
    source: str = "local",
) -> None:
    """Load an architecture spec and register it as a model family.

    *spec_path_or_dict* can be a file path (str/Path) to a YAML file,
    or an already-parsed dict.

    The registered factory creates a ``ComposedArchitecture`` that
    mirrors the Python originals in parameter count and forward behavior.
    """
    from crucible.models.registry import register_model

    if isinstance(spec_path_or_dict, dict):
        spec = ArchitectureSpec.from_dict(spec_path_or_dict)
    else:
        spec = ArchitectureSpec.from_yaml(spec_path_or_dict)

    actual_name = name or spec.name

    def _factory(args: Any) -> Any:
        resolver = SpecResolver(spec, args)
        resolved = resolver.resolve()
        return ComposedArchitecture(resolved, args)

    register_model(actual_name, _factory, source=source)
