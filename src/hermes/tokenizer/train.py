from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import yaml

from hermes.tokenizer.tokenizer import HermesTokenizer


def iter_train_texts(path: str | Path) -> Iterator[str]:
	"""Yield text from the train split only, without accumulating the split."""
	train_path = Path(path)
	with train_path.open("r", encoding="utf-8") as handle:
		for line in handle:
			if not line.strip():
				continue
			record = json.loads(line)
			text = record.get("text") if isinstance(record, dict) else None
			if isinstance(text, str) and text:
				yield text


def train_tokenizer(
	*,
	train_path: str | Path,
	output_path: str | Path,
	vocab_size: int,
	special_tokens: list[str],
) -> HermesTokenizer:
	import tempfile
	import os
	
	fd, tmp_path = tempfile.mkstemp(suffix=".txt")
	with os.fdopen(fd, "w", encoding="utf-8") as out_f:
		for text in iter_train_texts(train_path):
			out_f.write(text + "\n")
			
	try:
		tokenizer = HermesTokenizer().train_from_file(
			tmp_path,
			vocab_size=vocab_size,
			special_tokens=special_tokens,
		)
	finally:
		os.remove(tmp_path)
		
	tokenizer.save(output_path)
	return tokenizer


def train_from_config(config_path: str | Path) -> HermesTokenizer:
	with Path(config_path).open("r", encoding="utf-8") as handle:
		config: dict[str, Any] = yaml.safe_load(handle) or {}
	tokenizer_config = config["tokenizer"]
	return train_tokenizer(
		train_path=tokenizer_config["training_data"],
		output_path=tokenizer_config["output_path"],
		vocab_size=int(tokenizer_config["vocab_size"]),
		special_tokens=list(tokenizer_config["special_tokens"]),
	)
