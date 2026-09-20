from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Force UTF-8 stdout so Devanagari and other non-ASCII characters don't crash
# the Windows CP1252 console encoder when printed directly.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hermes.tokenizer.tokenizer import HermesTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect tokenization using train-split examples only.")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer_final/tokenizer.json")
    parser.add_argument("--train-data", default="data/processed_180m_rebuild/train.jsonl")
    parser.add_argument("--output", default="results/tokenization_inspection.txt")
    args = parser.parse_args()

    tokenizer = HermesTokenizer.load(args.tokenizer)
    examples: dict[str, str] = {}
    with Path(args.train_data).open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            language = record.get("language")
            if language in {"en", "hi", "hinglish"} and language not in examples:
                examples[language] = record["text"]
            if len(examples) == 3:
                break

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for language in ("en", "hi", "hinglish"):
            text = examples[language]
            ids = tokenizer.encode(text)
            decoded = tokenizer.decode(ids)
            handle.write(f"[{language}]\ntext: {text}\nids: {ids}\ndecoded: {decoded}\n\n")
            print(f"[{language}] {text}\n  ids={ids}\n  decoded={decoded}")
    print(f"Saved inspection output to {output_path}")


if __name__ == "__main__":
    main()