from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class TokenizedSplitDataset(Dataset[dict[str, torch.Tensor]]):
    """Load one Day 4 tokenized split without consulting another split.

    Uses a byte-offset index so the full file is never held in memory;
    each __getitem__ call seeks to the correct line and parses it on demand.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

        # Build a lightweight index of byte offsets, one entry per non-empty
        # line. The JSON records themselves are not loaded into memory.
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
            raise ValueError(
                f"Expected JSON object at offset {self._offsets[index]} "
                f"in {self.path}"
            )

        return {
            "input_ids": torch.tensor(record["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(
                record["attention_mask"],
                dtype=torch.long,
            ),
        }


class EpochSeededRandomSampler(Sampler[int]):
    """Deterministic memory-bounded random sampler with explicit epochs.

    Each epoch gets its own deterministic permutation derived from the
    configured seed and epoch number. The sampler does not rely on PyTorch's
    global RNG state, so a fresh process can reconstruct the same ordering
    after loading a checkpoint.

    ``torch.randperm`` stores only the integer permutation tensor for the
    current epoch. No permutation is stored inside checkpoints.
    """

    def __init__(
        self,
        data_source: Dataset[Any],
        *,
        seed: int,
    ) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.data_source)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def state_dict(self) -> dict[str, int]:
        return {
            "seed": self.seed,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if "seed" not in state or "epoch" not in state:
            raise ValueError("Invalid EpochSeededRandomSampler state")

        self.seed = int(state["seed"])
        self.set_epoch(int(state["epoch"]))

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        # randperm uses O(N) tensor memory for the current epoch but does not
        # create a Python list containing millions of indices.
        permutation = torch.randperm(
            len(self.data_source),
            generator=generator,
        )

        for index in permutation:
            yield int(index.item())


def create_dataloader(
    path: str | Path,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int = 42,
) -> DataLoader[dict[str, torch.Tensor]]:
    dataset = TokenizedSplitDataset(path)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    if shuffle:
        sampler = EpochSeededRandomSampler(
            dataset,
            seed=seed,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            generator=loader_generator,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=loader_generator,
    )