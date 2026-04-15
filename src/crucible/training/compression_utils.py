"""Shared compression primitives for QAT, pruning, and metrics callbacks.

Model-agnostic utilities used across compression plugins. All functions
operate on ``nn.Module`` instances without assuming architecture details.
"""
from __future__ import annotations

from typing import Iterator

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Prunable layer iteration
# ---------------------------------------------------------------------------

_DEFAULT_PRUNABLE = (nn.Linear,)
_DEFAULT_EXCLUDE_NAMES = ("tok_emb", "embed_low", "lm_head")


def iter_prunable_layers(
    model: nn.Module,
    include_patterns: tuple[str, ...] = (),
    exclude_patterns: tuple[str, ...] = _DEFAULT_EXCLUDE_NAMES,
) -> Iterator[tuple[str, nn.Module]]:
    """Yield ``(name, module)`` for layers eligible for pruning/quantization.

    By default includes all ``nn.Linear`` (and subclasses); excludes
    embeddings and output heads.
    """
    for name, module in model.named_modules():
        if not isinstance(module, _DEFAULT_PRUNABLE):
            continue
        if any(p in name for p in exclude_patterns):
            continue
        if include_patterns and not any(p in name for p in include_patterns):
            continue
        yield name, module


# ---------------------------------------------------------------------------
# Weight masking
# ---------------------------------------------------------------------------


def apply_weight_mask(module: nn.Module, mask: Tensor) -> torch.utils.hooks.RemovableHandle:
    """Register a forward pre-hook that zeros masked weights each forward pass.

    Returns a handle that can be removed to detach the mask.
    """
    def _hook(mod: nn.Module, inputs: tuple[Tensor, ...]) -> None:
        mod.weight.data.mul_(mask)

    return module.register_forward_pre_hook(_hook)


def make_masks_permanent(model: nn.Module, masks: dict[str, Tensor]) -> None:
    """Zero out masked weights permanently (for serialization)."""
    named = dict(model.named_modules())
    for name, mask in masks.items():
        if name in named and hasattr(named[name], "weight"):
            named[name].weight.data.mul_(mask)


def remove_all_masks(model: nn.Module) -> None:
    """Remove all forward pre-hooks that look like pruning masks."""
    for module in model.modules():
        hooks_to_remove = []
        for handle_id, hook in module._forward_pre_hooks.items():
            # Our mask hooks are closures referencing 'mask'
            if hasattr(hook, "__name__") and hook.__name__ == "_hook":
                hooks_to_remove.append(handle_id)
        for hid in hooks_to_remove:
            del module._forward_pre_hooks[hid]


# ---------------------------------------------------------------------------
# Compression metrics
# ---------------------------------------------------------------------------


class CompressionMetrics:
    """Compute compression statistics for any ``nn.Module``."""

    @staticmethod
    def sparsity(model: nn.Module) -> dict[str, float]:
        """Per-parameter and overall sparsity ratios.

        Returns dict with per-param keys ``"sparsity/{name}"`` plus
        ``"sparsity/overall"``.
        """
        total = 0
        total_zero = 0
        result: dict[str, float] = {}
        for name, param in model.named_parameters():
            n = param.numel()
            z = int((param.data == 0).sum().item())
            total += n
            total_zero += z
            if n > 0:
                result[f"sparsity/{name}"] = z / n
        result["sparsity/overall"] = total_zero / total if total > 0 else 0.0
        return result

    @staticmethod
    def effective_bits(
        model: nn.Module,
        bit_assignments: dict[str, int] | None = None,
        default_bits: int = 32,
    ) -> float:
        """Weighted-average effective bit-width across parameters.

        *bit_assignments* maps parameter name prefixes to their quantized
        bit-width. Unmatched parameters use *default_bits*.
        """
        if bit_assignments is None:
            bit_assignments = {}

        total_params = 0
        weighted_bits = 0.0
        for name, param in model.named_parameters():
            n = param.numel()
            bits = default_bits
            for prefix, b in bit_assignments.items():
                if prefix in name:
                    bits = b
                    break
            total_params += n
            weighted_bits += n * bits
        return weighted_bits / total_params if total_params > 0 else float(default_bits)

    @staticmethod
    def model_size_bytes(model: nn.Module) -> int:
        """Total parameter memory footprint in bytes."""
        return sum(p.numel() * p.element_size() for p in model.parameters())

    @staticmethod
    def nonzero_params(model: nn.Module) -> int:
        """Count of non-zero parameters."""
        return sum(int((p.data != 0).sum().item()) for p in model.parameters())

    @staticmethod
    def compression_ratio(
        original_size: int,
        model: nn.Module,
        bit_assignments: dict[str, int] | None = None,
        default_bits: int = 32,
    ) -> float:
        """Compression ratio accounting for both sparsity and quantization.

        ``ratio = original_size / effective_compressed_size``
        """
        eff_bits = CompressionMetrics.effective_bits(model, bit_assignments, default_bits)
        # Effective size = (non-zero params * effective bits) / 8
        nz = CompressionMetrics.nonzero_params(model)
        effective_bytes = (nz * eff_bits) / 8.0
        if effective_bytes == 0:
            return float("inf")
        return original_size / effective_bytes
