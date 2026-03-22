"""Mixture of Experts layer for parameter-efficient capacity scaling.

Replaces the standard MLP in transformer blocks with a top-k routed
mixture of smaller expert FFNs. Each token is routed to the top-k
experts (default 2 of 4), with load-balancing auxiliary loss.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from crucible.models.components.linear import CastedLinear
from crucible.models.components.mlp import ACTIVATIONS


class Expert(nn.Module):
    """Single expert FFN: dim -> hidden -> dim with configurable activation."""

    def __init__(self, dim: int, hidden: int, activation: str = "relu_sq"):
        super().__init__()
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation {activation!r}")
        self._activation = ACTIVATIONS[activation]

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self._activation(self.fc(x)))


class MoELayer(nn.Module):
    """Top-k Mixture of Experts layer with load-balancing loss.

    Args:
        dim: Model dimension.
        num_experts: Number of expert FFNs (default 4).
        top_k: How many experts each token is routed to (default 2).
        mlp_mult: MLP expansion factor per expert.
        activation: Activation function name.
        aux_loss_coeff: Coefficient for load-balancing auxiliary loss.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        mlp_mult: int = 2,
        activation: str = "relu_sq",
        aux_loss_coeff: float = 0.01,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coeff = aux_loss_coeff

        # Router: project to num_experts logits
        self.router = nn.Linear(dim, num_experts, bias=False)

        # Expert FFNs
        hidden = mlp_mult * dim
        self.experts = nn.ModuleList([
            Expert(dim, hidden, activation=activation)
            for _ in range(num_experts)
        ])

        # Store auxiliary loss for training
        self._aux_loss: Tensor | None = None

    @property
    def aux_loss(self) -> Tensor:
        """Load-balancing auxiliary loss from the last forward pass."""
        if self._aux_loss is None:
            return torch.tensor(0.0)
        return self._aux_loss

    def forward(self, x: Tensor) -> Tensor:
        """Route each token to top-k experts and combine outputs.

        Args:
            x: (batch, seq, dim)

        Returns:
            (batch, seq, dim) — weighted sum of expert outputs.
        """
        B, T, D = x.shape

        # Compute router logits and top-k selection
        logits = self.router(x)  # (B, T, num_experts)
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)  # (B, T, top_k)
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # (B, T, top_k)

        # Compute load-balancing auxiliary loss
        if self.training:
            router_probs = F.softmax(logits, dim=-1)  # (B, T, E)
            # Fraction of tokens routed to each expert
            tokens_per_expert = torch.zeros(self.num_experts, device=x.device)
            for k in range(self.top_k):
                tokens_per_expert.scatter_add_(
                    0, top_k_indices[:, :, k].reshape(-1),
                    torch.ones(B * T, device=x.device),
                )
            fraction_dispatched = tokens_per_expert / (B * T * self.top_k)
            # Average router probability per expert
            fraction_probs = router_probs.mean(dim=(0, 1))
            # Aux loss: encourage uniform distribution
            self._aux_loss = self.aux_loss_coeff * self.num_experts * (
                fraction_dispatched * fraction_probs
            ).sum()

        # Compute expert outputs — simple loop (efficient enough for 4 experts)
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, :, k]  # (B, T)
            weight = top_k_weights[:, :, k].unsqueeze(-1)  # (B, T, 1)
            for e in range(self.num_experts):
                mask = (expert_idx == e)  # (B, T)
                if not mask.any():
                    continue
                # Gather tokens for this expert
                expert_input = x[mask]  # (num_tokens, D)
                expert_output = self.experts[e](expert_input)  # (num_tokens, D)
                # Scatter back weighted output
                output[mask] += weight[mask] * expert_output

        return output
