from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


def split_records(records: list[dict[str, Any]], *, seed: int = 42, train_ratio: float = 0.90, validation_ratio: float = 0.05, test_ratio: float = 0.05) -> dict[str, list[dict[str, Any]]]:
    """Split records into deterministic train/validation/test splits while preserving language balance."""
    if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")

    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        language = str(record.get("language", "unknown"))
        by_language[language].append(record)

    result: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    rng = random.Random(seed)

    for language, language_records in sorted(by_language.items()):
        shuffled = list(language_records)
        rng.shuffle(shuffled)
        total = len(shuffled)
        if total == 0:
            continue
        train_count = int(round(total * train_ratio))
        validation_count = int(round(total * validation_ratio))
        test_count = total - train_count - validation_count
        if test_count < 0:
            test_count = 0
        if train_count + validation_count + test_count != total:
            # Preserve the required 90/5/5 structure as closely as possible.
            remaining = total - (train_count + validation_count + test_count)
            if remaining != 0:
                train_count += remaining
        split_1 = shuffled[:train_count]
        split_2 = shuffled[train_count:train_count + validation_count]
        split_3 = shuffled[train_count + validation_count:train_count + validation_count + test_count]
        result["train"].extend(split_1)
        result["validation"].extend(split_2)
        result["test"].extend(split_3)

    rng = random.Random(seed)
    for split_name in ("train", "validation", "test"):
        rng.shuffle(result[split_name])

    return result


__all__ = ["split_records"]
