from __future__ import annotations

import torch
from torch import Tensor, nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.embedding(token_ids)


class PositionalEmbedding(nn.Module):
    def __init__(self, context_length: int, d_model: int) -> None:
        super().__init__()
        self.context_length = context_length
        self.embedding = nn.Embedding(context_length, d_model)

    def forward(self, token_embeddings: Tensor) -> Tensor:
        _, sequence_length, _ = token_embeddings.shape
        if sequence_length > self.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds context length {self.context_length}"
            )
        positions = torch.arange(
            sequence_length, device=token_embeddings.device
        )
        return token_embeddings + self.embedding(positions).unsqueeze(0)