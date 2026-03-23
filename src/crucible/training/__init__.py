"""Crucible training module — extracted from train_gpt.py.

Contains the PyTorch training loop, hyperparameters, optimizers,
data loading, validation, quantization, and TTT evaluation.
"""
from __future__ import annotations

__all__ = [
    "Hyperparameters",
    "Muon",
    "DistributedTokenLoader",
    "TokenStream",
    "validate_model",
    "quantize_state_dict",
    "dequantize_state_dict",
    "compress_blob",
    "decompress_blob",
]
