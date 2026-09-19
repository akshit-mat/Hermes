from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


class TokenizedSplitDataset(Dataset[dict[str, torch.Tensor]]):
    """Load one Day 4 tokenized split without consulting another split.

    Uses a byte-offset index so the full file is never held in memory;
    each __getitem__ call seeks to the correct line and parses it on demand.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        # Build a lightweight index of byte offsets — one entry per non-empty line.
        # This avoids loading any JSON into RAM at init time.
        self._offsets: list[int] = []
        with self.path.open("rb") as handle:
            offset = 0
            for raw_line in handle:
                stripped = raw_line.strip()
                if stripped:
                    self._offsets.append(offset)
                offset += len(raw_line)

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        with self.path.open("rb") as handle:
            handle.seek(self._offsets[index])
            raw_line = handle.readline()
        record: dict[str, Any] = json.loads(raw_line)
        if not isinstance(record, dict):
            raise ValueError(f"Expected JSON object at offset {self._offsets[index]} in {self.path}")
        return {
            "input_ids": torch.tensor(record["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(record["attention_mask"], dtype=torch.long),
        }


def create_dataloader(
    path: str | Path,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader[dict[str, torch.Tensor]]:
    return DataLoader(
        TokenizedSplitDataset(path),
        batch_size=batch_size,
        shuffle=shuffle,
    )