from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yaml

from hermes.data.dataset import run_debug_pipeline, save_metadata, save_processed_dataset
from hermes.data.download import download_source_records
from hermes.data.preprocess import preprocess_records
from hermes.data.splits import split_records


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _approximate_token_count(text: str) -> int:
    value = text.strip()
    if not value:
        return 0
    return max(1, len(value.split()))


def _deterministic_split_key(language: str, text: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}|{language}|{text}".encode("utf-8")).digest()
    return str(int.from_bytes(digest[:8], "big") % 100)


def _assign_record_to_split(language: str, text: str, seed: int, *, train_ratio: float = 0.90, validation_ratio: float = 0.05) -> str:
    bucket = int(_deterministic_split_key(language, text, seed))
    if bucket < int(train_ratio * 100):
        return "train"
    if bucket < int((train_ratio + validation_ratio) * 100):
        return "validation"
    return "test"


def _write_jsonl_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_data(config_path: str | Path, output_dir: str | Path, *, seed: int = 42, max_examples: int | None = None, debug: bool = False) -> dict:
    config = load_config(config_path)
    data_cfg = config.get("data", {})
    output_path = Path(output_dir)

    if debug:
        result = run_debug_pipeline(output_dir=output_path, seed=seed, max_examples_per_language=data_cfg.get("debug", {}).get("max_examples_per_language", 20))
        return result

    source_configs = data_cfg.get("sources", {})
    english_source_cfg = source_configs.get("english", {})
    hindi_source_cfg = source_configs.get("hindi", {})
    hinglish_source_cfg = source_configs.get("hinglish", {})
    hinglish_environment_path = os.environ.get("HERMES_HINGLISH_SOURCE_PATH")
    hinglish_configured_local_path = hinglish_source_cfg.get("local_path")
    hinglish_actual_source_path = hinglish_configured_local_path or hinglish_environment_path

    category_sources = {
        "en": {"dataset": "wikimedia/wikipedia", "language": "en", "source_id": english_source_cfg.get("dataset", "wikimedia/wikipedia"), "actual_source": english_source_cfg.get("local_path") or "wikimedia/wikipedia", "actual_source_path": english_source_cfg.get("local_path")},
        "hi": {"dataset": "wikimedia/wikipedia", "language": "hi", "source_id": hindi_source_cfg.get("dataset", "wikimedia/wikipedia"), "actual_source": hindi_source_cfg.get("local_path") or "wikimedia/wikipedia", "actual_source_path": hindi_source_cfg.get("local_path")},
        "hinglish": {"dataset": "L3Cube-HingCorpus", "language": "hinglish", "source_id": hinglish_source_cfg.get("source_url", "https://github.com/l3cube-pune/code-mixed-nlp"), "actual_source": hinglish_actual_source_path if hinglish_actual_source_path else hinglish_source_cfg.get("source_url"), "actual_source_path": hinglish_actual_source_path},
    }

    target_tokens = {category: int(data_cfg.get("target_tokens_per_category", 60000000)) for category in ["en", "hi", "hinglish"]}
    categories = ["en", "hi", "hinglish"]
    raw_examples_seen = {category: 0 for category in categories}
    removed_examples = {category: 0 for category in categories}
    duplicate_records_removed = {category: 0 for category in categories}
    retained_examples = {category: 0 for category in categories}
    category_seen: dict[str, set[str]] = {category: set() for category in categories}
    category_approx_tokens = {category: 0 for category in categories}
    category_reached_target = {category: False for category in categories}
    min_len = data_cfg.get("min_text_length", 4)
    max_len = data_cfg.get("max_text_length", 50000)
    split_counts = {"train": 0, "validation": 0, "test": 0}
    split_approx_tokens = {"train": 0, "validation": 0, "test": 0}
    split_ratios = {"train": float(data_cfg.get("train_ratio", 0.90)), "validation": float(data_cfg.get("validation_ratio", 0.05)), "test": float(data_cfg.get("test_ratio", 0.05))}

    output_path.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        split_handles = {
            split_name: stack.enter_context((output_path / f"{split_name}.jsonl").open("w", encoding="utf-8"))
            for split_name in ("train", "validation", "test")
        }

        for language_name in ("english", "hindi", "hinglish"):
            source_cfg = data_cfg.get("sources", {}).get(language_name, {})
            language = source_cfg.get("language", "en" if language_name == "english" else "hi" if language_name == "hindi" else "hinglish")
            local_path = source_cfg.get("local_path")
            if max_examples is not None:
                source_iter = download_source_records(language, max_examples=max_examples, seed=seed, source_path=local_path)
            else:
                source_iter = download_source_records(language, seed=seed, source_path=local_path)

            for record in source_iter:
                raw_examples_seen[language] += 1
                if category_reached_target[language]:
                    break

                text = str(record.get("text", "")).strip()
                normalized = text
                if not normalized or len(normalized) < min_len or len(normalized) > max_len:
                    removed_examples[language] += 1
                    continue

                processed = preprocess_records([{"text": normalized, "language": language}], min_length=min_len, max_length=max_len)
                if not processed:
                    removed_examples[language] += 1
                    continue
                processed_text = processed[0]["text"]
                if processed_text in category_seen[language]:
                    duplicate_records_removed[language] += 1
                    continue
                category_seen[language].add(processed_text)

                approx_tokens = _approximate_token_count(processed_text)
                category_approx_tokens[language] += approx_tokens
                retained_examples[language] += 1

                assigned_split = _assign_record_to_split(language, processed_text, seed, train_ratio=split_ratios["train"], validation_ratio=split_ratios["validation"])
                split_counts[assigned_split] += 1
                split_approx_tokens[assigned_split] += approx_tokens
                _write_jsonl_record(split_handles[assigned_split], {"text": processed_text, "language": language})

                if category_approx_tokens[language] >= target_tokens[language]:
                    category_reached_target[language] = True
                    break

    metadata = {
        "dataset_name": "Hermes Phase 1",
        "dataset_preparation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_category_details": category_sources,
        "source_identifiers": {
            "en": {"dataset": "wikimedia/wikipedia", "config_candidates": english_source_cfg.get("config_candidates", ["20231101.en", "20220301.en", "20210101.en"]), "actual_source": english_source_cfg.get("local_path") or "wikimedia/wikipedia", "actual_source_path": english_source_cfg.get("local_path")},
            "hi": {"dataset": "wikimedia/wikipedia", "config_candidates": hindi_source_cfg.get("config_candidates", ["20231101.hi", "20220301.hi", "20210101.hi"]), "actual_source": hindi_source_cfg.get("local_path") or "wikimedia/wikipedia", "actual_source_path": hindi_source_cfg.get("local_path")},
            "hinglish": {"dataset": "L3Cube-HingCorpus", "source_url": hinglish_source_cfg.get("source_url"), "local_fallback": hinglish_configured_local_path, "environment_source_path": hinglish_environment_path, "actual_source_path": hinglish_actual_source_path, "actual_source": hinglish_actual_source_path or hinglish_source_cfg.get("source_url")},
        },
        "seed": seed,
        "preprocessing_configuration": {
            "min_text_length": min_len,
            "max_text_length": max_len,
            "normalize_line_endings": True,
            "normalize_whitespace": True,
            "preserve_devanagari": True,
            "preserve_latin": True,
            "preserve_punctuation": True,
            "allow_code_mixing": True,
            "transliteration_disabled": True,
        },
        "filtering_rules": {
            "min_text_length": min_len,
            "max_text_length": max_len,
            "empty_text_removed": True,
            "too_short_removed": True,
            "too_long_removed": True,
            "symbol_only_removed": True,
            "pathological_repetition_removed": True,
            "obvious_url_html_noise_removed": True,
            "social_media_hinglish_preserved": True,
        },
        "random_seed": seed,
        "target_tokens_per_category": target_tokens,
        "target_approximate_tokens_per_category": target_tokens,
        "achieved_approximate_tokens_per_category": category_approx_tokens,
        "total_achieved_approximate_tokens": sum(category_approx_tokens.values()),
        "approximate_token_methodology": "Whitespace-tokenized word count before tokenizer training; not exact BPE token count.",
        "raw_examples_seen": raw_examples_seen,
        "retained_examples": retained_examples,
        "removed_examples": removed_examples,
        "duplicate_records_removed": duplicate_records_removed,
        "split_ratios": split_ratios,
        "train_example_count": split_counts["train"],
        "validation_example_count": split_counts["validation"],
        "test_example_count": split_counts["test"],
        "train_approximate_tokens": split_approx_tokens["train"],
        "validation_approximate_tokens": split_approx_tokens["validation"],
        "test_approximate_tokens": split_approx_tokens["test"],
        "category_distribution": {category: retained_examples[category] for category in categories},
        "output_format": "jsonl",
        "output_files": {"train": str(output_path / "train.jsonl"), "validation": str(output_path / "validation.jsonl"), "test": str(output_path / "test.jsonl")},
        "metadata_path": str(output_path / "metadata.json"),
        "target_reached_by_category": category_reached_target,
        "debug_run": False,
    }
    save_metadata(output_path, metadata)
    return {"output_dir": str(output_path), "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Hermes data for tokenizer training.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to the YAML config file.")
    parser.add_argument("--output-dir", default="data/processed", help="Where processed JSONL files and metadata will be written.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for deterministic splitting.")
    parser.add_argument("--max-examples", type=int, default=None, help="Optional cap on examples per language for a smaller debug or sample run.")
    parser.add_argument("--debug", action="store_true", help="Generate a tiny deterministic debug subset instead of the full dataset.")
    args = parser.parse_args()

    prepare_data(args.config, args.output_dir, seed=args.seed, max_examples=args.max_examples, debug=args.debug)


if __name__ == "__main__":
    main()
