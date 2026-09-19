from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from hermes.data.preprocess import preprocess_records
from hermes.data.splits import split_records


def _approximate_token_count(text: str) -> int:
    payload = text.strip()
    if not payload:
        return 0
    return max(1, len(payload.split()))


def save_metadata(output_dir: str | Path, metadata: dict[str, Any]) -> Path:
    """Persist metadata in JSON format for later inspection and training."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata_path


def save_processed_dataset(records: list[dict[str, Any]] | dict[str, list[dict[str, Any]]], output_dir: str | Path, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write processed records as JSONL and emit metadata.json when provided."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if isinstance(records, dict):
        split_files = records
        for name, items in split_files.items():
            file_path = output_path / f"{name}.jsonl"
            with file_path.open("w", encoding="utf-8") as handle:
                for item in items:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        file_path = output_path / "processed.jsonl"
        with file_path.open("w", encoding="utf-8") as handle:
            for item in records:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    if metadata is not None:
        save_metadata(output_path, metadata)

    return {"output_dir": str(output_path), "record_count": len(records) if isinstance(records, list) else sum(len(items) for items in records.values())}


def load_processed_dataset(path: str | Path) -> Iterator[dict[str, Any]]:
    """Load JSONL records from a directory or an individual JSONL file."""
    dataset_path = Path(path)
    if dataset_path.is_dir():
        candidates = sorted(dataset_path.glob("*.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"No JSONL dataset files found in {dataset_path}")
        for candidate in candidates:
            yield from load_processed_dataset(candidate)
        return

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                yield item


def build_debug_records(max_examples_per_language: int = 20, seed: int = 42) -> list[dict[str, Any]]:
    """Create a tiny deterministic subset of synthetic examples that exercises the pipeline without a full dataset download."""
    languages = {
        "en": [
            "Hello world and welcome to the project.",
            "The quick brown fox jumps over the lazy dog.",
            "English text is easy to read and analyze.",
            "This is a short example of multilingual training data.",
            "We keep the pipeline deterministic and reproducible.",
        ],
        "hi": [
            "नमस्ते दुनिया, यह हिंदी भाषा का उदाहरण है।",
            "यह पाठ देवनागरी में लिखा गया है और साफ़ है।",
            "हिंदी में तकनीकी सामग्री भी अच्छी तरह से लिखी जाती है।",
            "इस उदाहरण में देवनागरी को सुरक्षित रखा गया है।",
            "भाषा और लिपि दोनों को सुरक्षित रखने की कोशिश की गई है।",
        ],
        "hinglish": [
            "kal movie dekhne chale? 😂",
            "yeh code-mix hai but still clear enough.",
            "arre bhai, aaj weather kitna hot hai?",
            "logon ko chill karna chahiye, no stress.",
            "movie ka trailer bahut awesome tha!",
        ],
    }

    records: list[dict[str, Any]] = []
    for language, examples in languages.items():
        for index in range(max_examples_per_language):
            text = examples[index % len(examples)]
            records.append({"text": f"{text} ({language} sample {index})", "language": language})
    return records


def _build_category_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    distribution = {category: 0 for category in ["en", "hi", "hinglish"]}
    approx_tokens = 0
    for record in records:
        language = str(record.get("language", "unknown"))
        if language in distribution:
            distribution[language] += 1
        text = str(record.get("text", ""))
        approx_tokens += _approximate_token_count(text)
    return {"distribution": distribution, "approximate_tokens": approx_tokens}


def run_debug_pipeline(*, output_dir: str | Path, seed: int = 42, max_examples_per_language: int = 20) -> dict[str, Any]:
    """Create a small, complete debug corpus that exercises each pipeline stage."""
    debug_min_length = 4
    debug_max_length = 10000
    records = build_debug_records(max_examples_per_language=max_examples_per_language, seed=seed)
    processed = preprocess_records(records, min_length=debug_min_length, max_length=debug_max_length)
    split = split_records(processed, seed=seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    category_distribution = {language: 0 for language in ["en", "hi", "hinglish"]}
    split_approx_tokens = {name: 0 for name in ["train", "validation", "test"]}
    for name, items in split.items():
        with (output_path / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for item in items:
                language = str(item.get("language", "unknown"))
                if language in category_distribution:
                    category_distribution[language] += 1
                split_approx_tokens[name] += _approximate_token_count(str(item.get("text", "")))
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    metadata = {
        "dataset_name": "Hermes Phase 1",
        "dataset_preparation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "debug",
        "source": "synthetic_debug_records",
        "source_identifier": "synthetic_debug_records",
        "source_category_details": {"en": {"dataset": "synthetic_debug_records", "language": "en"}, "hi": {"dataset": "synthetic_debug_records", "language": "hi"}, "hinglish": {"dataset": "synthetic_debug_records", "language": "hinglish"}},
        "seed": seed,
        "target_tokens_per_category": {"en": 0, "hi": 0, "hinglish": 0},
        "target_approximate_tokens_per_category": {"en": 0, "hi": 0, "hinglish": 0},
        "achieved_approximate_tokens_per_category": {"en": sum(_approximate_token_count(str(item.get("text", ""))) for item in processed if item.get("language") == "en"), "hi": sum(_approximate_token_count(str(item.get("text", ""))) for item in processed if item.get("language") == "hi"), "hinglish": sum(_approximate_token_count(str(item.get("text", ""))) for item in processed if item.get("language") == "hinglish")},
        "total_achieved_approximate_tokens": sum(_approximate_token_count(str(item.get("text", ""))) for item in processed),
        "raw_examples_seen": len(records),
        "retained_examples": len(processed),
        "removed_examples": max(0, len(records) - len(processed)),
        "duplicate_records_removed": max(0, len(records) - len(processed)),
        "category_distribution": category_distribution,
        "split_ratios": {"train": 0.90, "validation": 0.05, "test": 0.05},
        "train_example_count": len(split["train"]),
        "validation_example_count": len(split["validation"]),
        "test_example_count": len(split["test"]),
        "train_approximate_tokens": split_approx_tokens["train"],
        "validation_approximate_tokens": split_approx_tokens["validation"],
        "test_approximate_tokens": split_approx_tokens["test"],
        "output_dir": str(output_path),
        "output_format": "jsonl",
        "output_files": {"train": str(output_path / "train.jsonl"), "validation": str(output_path / "validation.jsonl"), "test": str(output_path / "test.jsonl")},
        "metadata_path": str(output_path / "metadata.json"),
        "preprocessing_configuration": {
            "min_text_length": debug_min_length,
            "max_text_length": debug_max_length,
            "normalize_line_endings": True,
            "normalize_whitespace": True,
            "preserve_devanagari": True,
            "preserve_latin": True,
            "preserve_punctuation": True,
            "allow_code_mixing": True,
            "transliteration_disabled": True,
        },
        "filtering_rules": {
            "min_text_length": debug_min_length,
            "max_text_length": debug_max_length,
            "empty_text_removed": True,
            "too_short_removed": True,
            "too_long_removed": True,
            "symbol_only_removed": True,
            "pathological_repetition_removed": True,
            "obvious_url_html_noise_removed": True,
            "social_media_hinglish_preserved": True,
        },
        "approximate_token_methodology": "Whitespace-tokenized word count before tokenizer training; not exact BPE token count.",
        "mode": "debug",
    }
    save_metadata(output_path, metadata)
    return {"output_dir": str(output_path), "metadata": metadata}


__all__ = [
    "build_debug_records",
    "load_processed_dataset",
    "save_metadata",
    "save_processed_dataset",
    "run_debug_pipeline",
]
