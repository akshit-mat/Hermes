import json

from hermes.data.tokenize import tokenize_split
from hermes.tokenizer.tokenizer import HermesTokenizer


def test_tokenize_split_pads_and_frames_sequences(tmp_path):
    input_path = tmp_path / "train.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"text": "hello world", "language": "en"}),
                json.dumps({"text": "", "language": "en"}),
                json.dumps({"text": "नमस्ते kal friends", "language": "hinglish"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    tokenizer = HermesTokenizer().train(
        ["hello world", "नमस्ते kal friends"],
        vocab_size=64,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
    )
    output_path = tmp_path / "tokenized.jsonl"

    stats = tokenize_split(
        input_path=input_path,
        output_path=output_path,
        tokenizer=tokenizer,
        context_length=12,
    )
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    pad_id = tokenizer.token_to_id("<pad>")
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    assert stats["empty_records_skipped"] == 1
    assert records
    assert all(len(record["input_ids"]) == 12 for record in records)
    assert all(record["input_ids"][0] == bos_id for record in records)
    assert all(eos_id in record["input_ids"] for record in records)
    assert any(pad_id in record["input_ids"] for record in records)
    for record in records:
        first_pad = next((i for i, value in enumerate(record["input_ids"]) if value == pad_id), None)
        if first_pad is not None:
            assert all(value == pad_id for value in record["input_ids"][first_pad:])
            assert all(value == 0 for value in record["attention_mask"][first_pad:])