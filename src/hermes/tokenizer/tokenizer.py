from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


class HermesTokenizer:
	"""Shared BPE tokenizer for English, Hindi, and Hinglish text."""

	def __init__(self, tokenizer: Tokenizer | None = None) -> None:
		self._tokenizer = tokenizer

	def train(
		self,
		texts: Iterable[str],
		*,
		vocab_size: int,
		special_tokens: Sequence[str],
	) -> "HermesTokenizer":
		model = BPE(unk_token="<unk>")
		tokenizer = Tokenizer(model)
		tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
		tokenizer.decoder = ByteLevelDecoder()
		trainer = BpeTrainer(
			vocab_size=vocab_size,
			special_tokens=list(special_tokens),
			limit_alphabet=1000,
		)
		tokenizer.train_from_iterator(texts, trainer=trainer)
		self._tokenizer = tokenizer
		return self

	def train_from_file(
		self,
		file_path: str,
		*,
		vocab_size: int,
		special_tokens: Sequence[str],
	) -> "HermesTokenizer":
		model = BPE(unk_token="<unk>")
		tokenizer = Tokenizer(model)
		tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
		tokenizer.decoder = ByteLevelDecoder()
		trainer = BpeTrainer(
			vocab_size=vocab_size,
			special_tokens=list(special_tokens),
			limit_alphabet=1000,
		)
		tokenizer.train([file_path], trainer=trainer)
		self._tokenizer = tokenizer
		return self
	def encode(self, text: str) -> list[int]:
		return self._require_tokenizer().encode(text).ids

	def decode(self, ids: Sequence[int]) -> str:
		return self._require_tokenizer().decode(list(ids), skip_special_tokens=False)

	def save(self, path: str | Path) -> Path:
		output_path = Path(path)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		self._require_tokenizer().save(str(output_path))
		return output_path

	@classmethod
	def load(cls, path: str | Path) -> "HermesTokenizer":
		return cls(Tokenizer.from_file(str(Path(path))))

	def token_to_id(self, token: str) -> int | None:
		return self._require_tokenizer().token_to_id(token)

	@property
	def vocab_size(self) -> int:
		return self._require_tokenizer().get_vocab_size()

	def _require_tokenizer(self) -> Tokenizer:
		if self._tokenizer is None:
			raise RuntimeError("Tokenizer has not been trained or loaded.")
		return self._tokenizer
