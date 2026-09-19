from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

from hermes.tokenizer.train import train_from_config


def main() -> None:
	parser = argparse.ArgumentParser(description="Train the shared Hermes BPE tokenizer from the train split.")
	parser.add_argument("--config", default="configs/base.yaml")
	args = parser.parse_args()
	tokenizer = train_from_config(args.config)
	print(f"Saved tokenizer with vocabulary size {tokenizer.vocab_size}")


if __name__ == "__main__":
	main()
