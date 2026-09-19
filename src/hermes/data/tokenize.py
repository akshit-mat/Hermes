from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import yaml

from hermes.tokenizer.tokenizer import HermesTokenizer


SPLITS = ("train", "validation", "test")
CATEGORIES = ("en", "hi", "hinglish")


def iter_jsonl_records(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if isinstance(record, dict):
                yield record


def _sequence_from_tokens(
    tokens: list[int],
    *,
    context_length: int,
    bos_id: int,
    eos_id: int,
    pad_id: int,
) -> Iterator[tuple[list[int], list[int]]]:
    payload_length = context_length - 2
    if payload_length <= 0:
        raise ValueError("context_length must be at least 3")
    for start in range(0, len(tokens), payload_length):
        payload = tokens[start:start + payload_length]
        input_ids = [bos_id, *payload, eos_id]
        padding = context_length - len(input_ids)
        input_ids.extend([pad_id] * padding)
        attention_mask = [1] * (context_length - padding) + [0] * padding
        yield input_ids, attention_mask


def tokenize_split(
    *,
    input_path: str | Path,
    output_path: str | Path,
    tokenizer: HermesTokenizer,
    context_length: int,
) -> dict[str, Any]:
    pad_id = tokenizer.token_to_id("<pad>")
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    if pad_id is None or bos_id is None or eos_id is None:
        raise ValueError("Tokenizer must contain <pad>, <bos>, and <eos> tokens")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {
        "record_count": 0,
        "sequence_count": 0,
        "empty_records_skipped": 0,
        "category_sequence_counts": {category: 0 for category in CATEGORIES},
        "category_token_lengths": defaultdict(list),
    }
    with output.open("w", encoding="utf-8") as handle:
        for record_index, record in enumerate(iter_jsonl_records(input_path)):
            text = record.get("text")
            language = record.get("language", "unknown")
            if not isinstance(text, str) or not text:
                stats["empty_records_skipped"] += 1
                continue
            stats["record_count"] += 1
            tokens = tokenizer.encode(text)
            for chunk_index, (input_ids, attention_mask) in enumerate(
                _sequence_from_tokens(
                    tokens,
                    context_length=context_length,
                    bos_id=bos_id,
                    eos_id=eos_id,
                    pad_id=pad_id,
                )
            ):
                output_record = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "language": language,
                    "source_record_index": record_index,
                    "chunk_index": chunk_index,
                }
                handle.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                stats["sequence_count"] += 1
                if language in stats["category_sequence_counts"]:
                    stats["category_sequence_counts"][language] += 1
                    stats["category_token_lengths"][language].append(
                        sum(attention_mask)
                    )

    stats["category_token_lengths"] = dict(stats["category_token_lengths"])
    return stats


def tokenize_dataset(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    tokenizer_path: str | Path,
    statistics_path: str | Path,
    context_length: int,
) -> dict[str, Any]:
    tokenizer = HermesTokenizer.load(tokenizer_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    split_stats: dict[str, Any] = {}
    for split in SPLITS:
        split_stats[split] = tokenize_split(
            input_path=Path(input_dir) / f"{split}.jsonl",
            output_path=output_root / f"{split}.jsonl",
            tokenizer=tokenizer,
            context_length=context_length,
        )
    statistics = {
        "context_length": context_length,
        "chunking": "fixed non-overlapping chunks; each chunk is independently framed with BOS/EOS",
        "padding": "right padding with tokenizer <pad> ID",
        "splits": split_stats,
    }
    stats_path = Path(statistics_path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return statistics


def tokenize_from_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    tokenized_config = config["tokenized_data"]
    return tokenize_dataset(
        input_dir=tokenized_config["input_dir"],
        output_dir=tokenized_config["output_dir"],
        tokenizer_path=tokenized_config["tokenizer_path"],
        statistics_path=tokenized_config["statistics_path"],
        context_length=int(config["model"]["context_length"]),
    )