from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hermes.data.tokenize import tokenize_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize Hermes splits without crossing split boundaries.")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    statistics = tokenize_from_config(args.config)
    print(f"Tokenized splits at {statistics['context_length']} tokens")


if __name__ == "__main__":
    main()