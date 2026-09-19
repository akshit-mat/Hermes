from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class CausalSelfAttention(nn.Module):
    """Manual scaled dot-product multi-head causal self-attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        context_length: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must divide evenly by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.context_length = context_length
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        causal_mask = torch.tril(torch.ones(context_length, context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, context_length, context_length))

    def split_heads(self, tensor: Tensor) -> Tensor:
        batch_size, sequence_length, _ = tensor.shape
        return tensor.view(batch_size, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)

    def merge_heads(self, tensor: Tensor) -> Tensor:
        batch_size, _, sequence_length, _ = tensor.shape
        return tensor.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.d_model)

    def forward(self, hidden_states: Tensor) -> Tensor:
        batch_size, sequence_length, width = hidden_states.shape
        if width != self.d_model:
            raise ValueError(f"Expected hidden width {self.d_model}, got {width}")
        if sequence_length > self.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds context length {self.context_length}"
            )

        queries = self.split_heads(self.query(hidden_states))
        keys = self.split_heads(self.key(hidden_states))
        values = self.split_heads(self.value(hidden_states))
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            ~self.causal_mask[:, :, :sequence_length, :sequence_length],
            torch.finfo(scores.dtype).min,
        )
        weights = torch.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)
        attended = torch.matmul(weights, values)
        return self.residual_dropout(self.output(self.merge_heads(attended)))