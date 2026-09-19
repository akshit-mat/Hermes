import json

from hermes.tokenizer.tokenizer import HermesTokenizer
from hermes.tokenizer.train import train_tokenizer


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def test_tokenizer_trains_saves_loads_encodes_and_decodes(tmp_path):
	train_path = tmp_path / "train.jsonl"
	records = [
		{"text": "Hello world", "language": "en"},
		{"text": "नमस्ते दुनिया", "language": "hi"},
		{"text": "kal movie dekhne chale", "language": "hinglish"},
	]
	train_path.write_text(
		"".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
		encoding="utf-8",
	)
	artifact_path = tmp_path / "tokenizer.json"

	tokenizer = train_tokenizer(
		train_path=train_path,
		output_path=artifact_path,
		vocab_size=64,
		special_tokens=SPECIAL_TOKENS,
	)
	loaded = HermesTokenizer.load(artifact_path)

	ids = loaded.encode("kal movie dekhne chale")
	assert ids
	assert loaded.decode(ids)
	assert artifact_path.exists()
	assert tokenizer.vocab_size == loaded.vocab_size
	assert [loaded.token_to_id(token) for token in SPECIAL_TOKENS] == [0, 1, 2, 3]
