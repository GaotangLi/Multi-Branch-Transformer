# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Causal token/state fusion for temporal Jacobi transformer passes."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def rms_normalize(x: Tensor, eps: float) -> Tensor:
    """Parameter-free RMS normalization with FP32 reduction statistics."""

    inverse_rms = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)
    return x * inverse_rms.to(dtype=x.dtype)


class TemporalJacobiMixer(nn.Module):
    """Fuse token embeddings with the preceding token's prior-pass state.

    Position ``t`` receives state from position ``t - 1``.  Position zero is a
    hard, bitwise identity bypass, so the first temporal pass can use the token
    embeddings directly and every later pass has an unambiguous sequence
    boundary.  The recurrent branch is state-mandatory: the previous state is
    the value and the current token embedding only gates that value.

    This first implementation deliberately supports tensor-parallel size one;
    the two dense projections are therefore ordinary replicated parameters.
    """

    def __init__(self, hidden_size: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}.")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError(f"eps must be finite and positive, got {eps}.")

        self.hidden_size = hidden_size
        self.eps = eps
        self.state_value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.token_gate = nn.Linear(hidden_size, hidden_size, bias=False)

        # Preserve the state direction at initialization.  The token-derived
        # gate starts non-saturated around 0.5, following fan-in scaling.
        nn.init.eye_(self.state_value.weight)
        nn.init.normal_(self.token_gate.weight, mean=0.0, std=hidden_size**-0.5)

    def forward(self, token_embeddings: Tensor, previous_state: Tensor) -> Tensor:
        if token_embeddings.ndim != 3:
            raise ValueError(
                "temporal Jacobi inputs must have shape [sequence, batch, hidden], "
                f"got {tuple(token_embeddings.shape)}."
            )
        if token_embeddings.shape != previous_state.shape:
            raise ValueError(
                "token embeddings and previous state must have identical shapes, got "
                f"{tuple(token_embeddings.shape)} and {tuple(previous_state.shape)}."
            )
        if token_embeddings.shape[-1] != self.hidden_size:
            raise ValueError(
                f"expected hidden size {self.hidden_size}, got {token_embeddings.shape[-1]}."
            )
        if token_embeddings.shape[0] < 1:
            raise ValueError("temporal Jacobi mixing requires a non-empty sequence.")

        # For output position t>=1, the recurrent value is state[t-1] while
        # the gate is conditioned on token_embedding[t].  Computing only the
        # tail makes the boundary bypass both exact and free of dead work.
        token_tail = token_embeddings[1:]
        state_tail = previous_state[:-1]
        normalized_token = rms_normalize(token_tail, self.eps)
        normalized_state = rms_normalize(state_tail, self.eps)

        state_value = self.state_value(normalized_state)
        gate = torch.sigmoid(self.token_gate(normalized_token).float()).to(
            dtype=state_value.dtype
        )
        fused = state_value * gate

        # Match the recurrent branch RMS to the corresponding token embedding.
        # This avoids an implicit scale change while keeping the branch wholly
        # dependent on the carried state.
        fused = rms_normalize(fused, self.eps)
        token_rms = torch.sqrt(
            token_tail.float().square().mean(dim=-1, keepdim=True) + self.eps
        ).to(dtype=fused.dtype)
        recurrent_tail = fused * token_rms

        return torch.cat((token_embeddings[:1], recurrent_tail), dim=0)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.eps}"
