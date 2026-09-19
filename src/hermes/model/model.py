from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from hermes.model.embeddings import PositionalEmbedding, TokenEmbedding
from hermes.model.transformer import TransformerBlock


class HermesModel(nn.Module):
    """Full decoder-only Hermes language model assembled from Day 5 blocks."""

    def __init__(
        self,
        *,
        vocab_size: int,
        context_length: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        pad_token_id: int,
        dropout: float = 0.0,
        expansion_ratio: int = 4,
        tie_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.pad_token_id = pad_token_id
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_embedding = PositionalEmbedding(context_length, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    context_length=context_length,
                    dropout=dropout,
                    expansion_ratio=expansion_ratio,
                )
                for _ in range(n_layers)
            ]
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.token_embedding.embedding.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence_length]")
        if input_ids.shape[1] > self.context_length:
            raise ValueError(
                f"Sequence length {input_ids.shape[1]} exceeds context length {self.context_length}"
            )
        hidden_states = self.token_embedding(input_ids)
        hidden_states = self.positional_embedding(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.lm_head(hidden_states)

    def compute_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        if logits.ndim != 3 or labels.ndim != 2:
            raise ValueError("logits must be [batch, sequence, vocab] and labels [batch, sequence]")
        if logits.shape[:2] != labels.shape:
            raise ValueError("logits and labels must have matching batch and sequence dimensions")
        return nn.functional.cross_entropy(
            logits[:, :-1].contiguous().view(-1, self.vocab_size),
            labels[:, 1:].contiguous().view(-1),
            ignore_index=self.pad_token_id,
        )

    def loss(self, input_ids: Tensor, labels: Tensor | None = None) -> Tensor:
        target_labels = input_ids if labels is None else labels
        return self.compute_loss(self(input_ids), target_labels)

    def count_parameters(self, trainable_only: bool = True) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (parameter for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)


def build_model_from_config(
    config: dict[str, Any],
    *,
    pad_token_id: int,
    tie_embeddings: bool = True,
) -> HermesModel:
    model_config = config["model"]
    return HermesModel(
        vocab_size=int(model_config["vocab_size"]),
        context_length=int(model_config["context_length"]),
        d_model=int(model_config["d_model"]),
        n_layers=int(model_config["n_layers"]),
        n_heads=int(model_config["n_heads"]),
        dropout=float(model_config.get("dropout", 0.0)),
        pad_token_id=pad_token_id,
        tie_embeddings=tie_embeddings,
    )


def load_model_from_yaml(
    config_path: str | Path,
    *,
    pad_token_id: int,
    tie_embeddings: bool = True,
) -> HermesModel:
    import yaml

    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return build_model_from_config(
        config,
        pad_token_id=pad_token_id,
        tie_embeddings=tie_embeddings,
    )