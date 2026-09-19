from __future__ import annotations

import torch
from torch import Tensor, nn

from hermes.model.attention import CausalSelfAttention


class FeedForward(nn.Module):
    """Position-wise MLP with a 4x hidden expansion and GELU activation."""

    def __init__(self, d_model: int, expansion_ratio: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = d_model * expansion_ratio
        self.network = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.network(hidden_states)


class TransformerBlock(nn.Module):
    """Pre-normalized causal attention block with residual connections."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        context_length: int,
        dropout: float = 0.0,
        expansion_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            context_length=context_length,
            dropout=dropout,
        )
        self.feed_forward_norm = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(
            d_model=d_model,
            expansion_ratio=expansion_ratio,
            dropout=dropout,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = hidden_states + self.attention(self.attention_norm(hidden_states))
        hidden_states = hidden_states + self.feed_forward(self.feed_forward_norm(hidden_states))
        return hidden_states