from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from crucible.models.components.norm import RMSNorm
from crucible.models.components.linear import CastedLinear


class CrucibleModel(nn.Module, ABC):
    """Abstract base for all Crucible models, modality-agnostic.

    Every model that participates in the Crucible training contract should
    inherit from this class.  The generic training backend
    (``training.generic_backend``) relies on :meth:`training_step` and
    :meth:`validation_step`` to drive the loop.
    """

    @abstractmethod
    def forward(self, **batch) -> dict[str, Tensor]:
        """Forward pass. Returns dict with at least ``'loss'`` key."""
        ...

    def training_step(self, **batch) -> dict[str, Tensor]:
        """Training forward pass. Default: delegates to forward()."""
        return self.forward(**batch)

    def validation_step(self, **batch) -> dict[str, Tensor]:
        """Validation forward pass. Default: delegates to forward()."""
        return self.forward(**batch)

    def metric_names(self) -> list[str]:
        """Names of metrics this model reports beyond ``'loss'``."""
        return []

    def param_groups(self) -> list[dict]:
        """Optimizer parameter groups. Override for custom grouping."""
        return [{"params": list(self.parameters())}]

    @classmethod
    def modality(cls) -> str:
        """Modality tag: ``'lm'``, ``'vision'``, ``'diffusion'``, ``'rl'``, etc."""
        return "generic"


class TiedEmbeddingLM(CrucibleModel):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        embed_bottleneck_dim: int = 0,
        spectral_embed_init: bool = False,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        if embed_bottleneck_dim < 0:
            raise ValueError(f"EMBED_BOTTLENECK_DIM must be non-negative, got {embed_bottleneck_dim}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.embed_bottleneck_dim = embed_bottleneck_dim
        self.spectral_embed_init = spectral_embed_init
        if embed_bottleneck_dim > 0:
            if embed_bottleneck_dim >= model_dim:
                raise ValueError(f"EMBED_BOTTLENECK_DIM={embed_bottleneck_dim} must be smaller than MODEL_DIM={model_dim}")
            self.embed_low = nn.Embedding(vocab_size, embed_bottleneck_dim)
            self.embed_proj = CastedLinear(embed_bottleneck_dim, model_dim, bias=False)
        else:
            self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            emb = self.embed_low if self.embed_bottleneck_dim > 0 else self.tok_emb
            if self.spectral_embed_init:
                # SVD-based spectral init: orthogonal embedding vectors for better
                # gradient flow through tied input/output projection.
                W = torch.randn_like(emb.weight)
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                emb.weight.data = U * (self.tied_embed_init_std * math.sqrt(emb.weight.size(0)))
            else:
                nn.init.normal_(emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def token_parameter_names(self) -> set[str]:
        return {"embed_low.weight", "embed_proj.weight"} if self.embed_bottleneck_dim > 0 else {"tok_emb.weight"}

    def embed_tokens(self, input_ids: Tensor) -> Tensor:
        x = self.embed_proj(self.embed_low(input_ids)) if self.embed_bottleneck_dim > 0 else self.tok_emb(input_ids)
        return F.rms_norm(x, (x.size(-1),))

    def tied_logits(self, x: Tensor) -> Tensor:
        if self.embed_bottleneck_dim > 0:
            return F.linear(F.linear(x, self.embed_proj.weight.T.to(x.dtype)), self.embed_low.weight.to(x.dtype))
        return F.linear(x, self.tok_emb.weight)

    def compute_loss(self, hidden: Tensor, target_ids: Tensor, lora=None) -> Tensor:
        x = self.final_norm(hidden)
        flat_x = x.reshape(-1, hidden.size(-1))
        logits_proj = self.tied_logits(flat_x) if self.tie_embeddings else self.lm_head(flat_x)
        if lora is not None:
            logits_proj = logits_proj + lora.lm_head_lora(x).reshape(-1, logits_proj.size(-1))
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        if lora is not None:
            bsz, sl = hidden.shape[:2]
            return F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction="none").reshape(bsz, sl)
        return F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction="mean", ignore_index=-100)

    def per_token_loss(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        """Per-token cross-entropy (no reduction), shape [batch*seq]."""
        hidden = self.hidden(input_ids)
        x = self.final_norm(hidden).reshape(-1, hidden.size(-1))
        logits_proj = self.tied_logits(x) if self.tie_embeddings else self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction="none")

    def hidden(self, input_ids: Tensor, lora=None) -> Tensor:
        raise NotImplementedError

    def forward(self, input_ids: Tensor, target_ids: Tensor, lora=None) -> Tensor:  # type: ignore[override]
        return self.compute_loss(self.hidden(input_ids, lora=lora), target_ids, lora=lora)

    def training_step(self, **batch) -> dict[str, Tensor]:
        """Bridge to generic training contract: ``{'loss': scalar}``."""
        loss = self.compute_loss(
            self.hidden(batch["input_ids"], lora=batch.get("lora")),
            batch["target_ids"],
            lora=batch.get("lora"),
        )
        return {"loss": loss}

    @classmethod
    def modality(cls) -> str:
        return "lm"
