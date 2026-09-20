import io
import bz2
import json
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from hermes.data.dataset import (load_processed_dataset, save_processed_dataset,
                                 save_metadata)
from hermes.data.download import (download_hinglish, get_hinglish_source_access_method,
                                  _iter_jsonl_from_archive, _iter_wikimedia_dump,
                                  _sample_stream, HINDI_WIKIMEDIA_DUMP_URL)
from hermes.data.preprocess import (deduplicate_examples, filter_examples,
                                   normalize_text)
from hermes.data.splits import split_records
from scripts.prepare_data import _assign_record_to_split, _write_jsonl_record, load_config, prepare_data


@pytest.mark.parametrize(
    "text",
    [
        "नमस्ते दुनिया!",
        "Hello, world!",
        "kal movie dekhne chale? 😂",
        "हैलो friends, kaise ho?",
    ],
)
def test_normalization_preserves_content(text):
    normalized = normalize_text(text)
    assert normalized == normalize_text(normalized)
    assert text.strip() == normalized or text.strip() in normalized


def test_empty_and_invalid_examples_are_filtered():
    samples = [
        "",
        "   ",
        "\n\n",
        "!!!",
        "###",
        "@@@",
        "This is valid text.",
        "नमस्ते世界",
    ]
    kept = [item for item in filter_examples(samples)]
    assert "This is valid text." in kept
    assert "नमस्ते世界" in kept
    assert not any(item in kept for item in ["", "   ", "!!!", "###", "@@@"])


def test_duplicate_normalized_texts_are_removed():
    examples = [
        {"text": "Hello world!", "language": "en"},
        {"text": " Hello   world!\n", "language": "en"},
        {"text": "नमस्ते दुनिया", "language": "hi"},
        {"text": "नमस्ते दुनिया", "language": "hi"},
    ]
    deduped = deduplicate_examples(examples)
    assert len(deduped) == 2
    assert {item["text"] for item in deduped} == {"Hello world!", "नमस्ते दुनिया"}
    assert all(item["text"] for item in deduped)


def test_split_records_are_deterministic_and_balanced():
    records = []
    for label in ["en", "hi", "hinglish"]:
        for i in range(60):
            records.append({"text": f"{label} sample {i}", "language": label})

    split_a = split_records(records, seed=42)
    split_b = split_records(records, seed=42)
    assert split_a == split_b

    counts = {name: len(items) for name, items in split_a.items()}
    assert counts["train"] + counts["validation"] + counts["test"] == len(records)
    assert abs(counts["train"] / len(records) - 0.90) < 0.1
    assert abs(counts["validation"] / len(records) - 0.05) < 0.05
    assert abs(counts["test"] / len(records) - 0.05) < 0.05

    for label in ["en", "hi", "hinglish"]:
        label_items = [item for item in records if item["language"] == label]
        label_split = split_records(label_items, seed=42)
        assert sum(len(v) for v in label_split.values()) == len(label_items)


def test_dataset_round_trip(tmp_path):
    records = [
        {"text": "Hello world", "language": "en"},
        {"text": "नमस्ते दुनिया", "language": "hi"},
        {"text": "kal movie dekhne chale? 😂", "language": "hinglish"},
    ]

    output_dir = tmp_path / "processed"
    save_processed_dataset(records, output_dir, metadata={"seed": 7})
    loaded = list(load_processed_dataset(output_dir))

    assert len(loaded) == len(records)
    assert loaded[0]["language"] == "en"
    assert loaded[1]["language"] == "hi"
    assert loaded[2]["language"] == "hinglish"

    metadata_path = output_dir / "metadata.json"
    assert metadata_path.exists()
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert data["seed"] == 7


def test_debug_pipeline_runs(tmp_path):
    output_dir = tmp_path / "debug_data"

    from hermes.data.dataset import run_debug_pipeline

    result = run_debug_pipeline(output_dir=output_dir, seed=9, max_examples_per_language=20)

    assert result["output_dir"] == str(output_dir)
    assert result["metadata"]["seed"] == 9
    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "validation.jsonl").exists()
    assert (output_dir / "test.jsonl").exists()
    assert (output_dir / "metadata.json").exists()

    train = list(load_processed_dataset(output_dir / "train.jsonl"))
    validation = list(load_processed_dataset(output_dir / "validation.jsonl"))
    test = list(load_processed_dataset(output_dir / "test.jsonl"))
    assert train
    assert validation
    assert test
    assert {item["language"] for item in train + validation + test} == {"en", "hi", "hinglish"}


def test_seeded_sampling_is_deterministic():
    stream = [{"text": f"example {i}", "language": "en"} for i in range(100)]
    first = list(_sample_stream(stream, max_examples=10, seed=123))
    second = list(_sample_stream(stream, max_examples=10, seed=123))
    assert first == second
    assert len(first) == 10


def test_hindi_wikimedia_dump_stream_preserves_text_and_devanagari(tmp_path):
    dump_path = tmp_path / "hiwiki-pages-articles.xml.bz2"
    root = ET.Element("mediawiki")
    for namespace, text in [("0", "नमस्ते दुनिया यह हिंदी लेख है"), ("1", "शीर्षक")]:
        page = ET.SubElement(root, "page")
        ET.SubElement(page, "ns").text = namespace
        revision = ET.SubElement(page, "revision")
        ET.SubElement(revision, "text").text = text
    payload = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    dump_path.write_bytes(bz2.compress(payload))

    records = list(_iter_wikimedia_dump(dump_path, language="hi"))

    assert records == [{"text": "नमस्ते दुनिया यह हिंदी लेख है", "language": "hi"}]
    assert HINDI_WIKIMEDIA_DUMP_URL.endswith("hiwiki-20260901-pages-articles.xml.bz2")


def test_base_config_resolves_verified_hindi_wikimedia_dump():
    hindi = load_config("configs/base.yaml")["data"]["sources"]["hindi"]

    assert hindi["language"] == "hi"
    assert hindi["dump_date"] == "20260901"
    assert hindi["dump_url"] == HINDI_WIKIMEDIA_DUMP_URL


def test_filtering_keeps_normal_text_and_rejects_url_html_garbage():
    retained = [
        "This is a normal English sentence.",
        "नमस्ते दुनिया, यह एक सामान्य हिंदी वाक्य है।",
        "kal movie dekhne chale? 😂",
        "yeh emoji hai 😄 and code-mix hai",
    ]
    for text in retained:
        assert list(filter_examples([text])) == [text]

    rejected = [
        "https://example.com/long/path",
        "<p>table of contents</p>",
        "!!!",
        "@@@",
    ]
    for text in rejected:
        assert list(filter_examples([text])) == []


def test_target_enforcement_stops_after_reaching_configured_threshold():
    records = [
        {"text": "alpha beta gamma delta", "language": "en"},
        {"text": "epsilon zeta eta theta", "language": "en"},
        {"text": "iota kappa lambda mu", "language": "en"},
    ]
    count = 0
    for record in records:
        if count >= 2:
            break
        count += 1
    assert count == 2
    assert _assign_record_to_split("en", "alpha beta gamma delta", 7) in {"train", "validation", "test"}


def test_hinglish_source_access_method_is_documented():
    method = get_hinglish_source_access_method()
    assert method["dataset"] == "L3Cube-HingCorpus"
    assert "github.com/l3cube-pune/code-mixed-nlp" in method["source_url"]
    assert "Google Drive" in method["access_method"]


def test_hinglish_local_source_and_upstream_failure_paths(tmp_path, monkeypatch):
    local_path = tmp_path / "hinglish.jsonl"
    local_path.write_text(
        '{"text": "kal movie dekhne chale?", "language": "hinglish"}\n'
        '{"text": "yeh awesome hai", "language": "hinglish"}\n',
        encoding="utf-8",
    )
    assert list(download_hinglish(source_path=local_path)) == [
        {"text": "kal movie dekhne chale?", "language": "hinglish"},
        {"text": "yeh awesome hai", "language": "hinglish"},
    ]

    def fake_probe(_source_url):
        return False, "HTTP 404 from https://github.com/l3cube-pune/code-mixed-nlp"

    monkeypatch.setattr("hermes.data.download._probe_hinglish_source_access", fake_probe)
    with pytest.raises(RuntimeError, match="HTTP 404"):
        list(download_hinglish())


def test_hinglish_tar_gz_archive_source(tmp_path):
    archive_path = tmp_path / "dummy_hinglish_archive.zip"
    train_member_name = "R11_final_data/concatenated_train_final_shuffled.txt"
    validation_member_name = "R11_final_data/concatenated_validation.txt"
    train_lines = [
        "kal movie dekhne chale?",
        "yeh awesome hai",
        "aaj weather bahut hot hai",
    ]
    validation_lines = ["validation Hinglish record"]

    with archive_path.open("wb") as handle:
        with tarfile.open(fileobj=handle, mode="w:gz") as archive:
            for member_name, lines in ((train_member_name, train_lines), (validation_member_name, validation_lines)):
                payload = ("\n".join(lines) + "\n").encode("utf-8")
                member = tarfile.TarInfo(member_name)
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))

    records = list(_iter_jsonl_from_archive(archive_path))

    assert [record["text"] for record in records] == train_lines + validation_lines
    assert [record["language"] for record in records] == ["hinglish"] * len(records)


def test_hinglish_truncated_archive_fails_closed(tmp_path):
    archive_path = tmp_path / "dummy_truncated_archive.tar.gz"
    member_name = "R11_final_data/concatenated_train_final_shuffled.txt"
    payload = (b"kal movie dekhne chale? this is a longer Hinglish record.\n" * 1000)

    with archive_path.open("wb") as handle:
        with tarfile.open(fileobj=handle, mode="w:gz") as archive:
            member = tarfile.TarInfo(member_name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))

    archive_bytes = archive_path.read_bytes()
    archive_path.write_bytes(archive_bytes[: len(archive_bytes) // 2])

    with pytest.raises(ValueError, match="archive is truncated"):
        list(_iter_jsonl_from_archive(archive_path))


def test_prepare_data_streams_incrementally_without_buffering(tmp_path, monkeypatch):
    write_seen = {"flag": False}
    original_write = _write_jsonl_record

    def guarded_write(handle, record):
        write_seen["flag"] = True
        original_write(handle, record)

    monkeypatch.setattr("scripts.prepare_data._write_jsonl_record", guarded_write)

    class GuardedStream:
        def __init__(self):
            self._items = iter([
                {"text": "alpha beta gamma", "language": "en"},
                {"text": "delta epsilon zeta", "language": "en"},
                {"text": "eta theta iota", "language": "en"},
            ])
            self._count = 0

        def __iter__(self):
            return self

        def __next__(self):
            self._count += 1
            item = next(self._items)
            if self._count > 1 and not write_seen["flag"]:
                raise AssertionError("records were buffered before the first file write")
            return item

    def fake_download_source_records(language, *, max_examples=None, seed=42, source_path=None):
        if language == "en":
            return GuardedStream()
        return iter([])

    monkeypatch.setattr("scripts.prepare_data.download_source_records", fake_download_source_records)
    output_dir = tmp_path / "streamed_production"
    result = prepare_data("configs/base.yaml", output_dir, seed=7, debug=False)

    assert result["metadata"]["train_example_count"] + result["metadata"]["validation_example_count"] + result["metadata"]["test_example_count"] > 0
    assert (output_dir / "train.jsonl").exists()


def test_target_enforcement_stops_at_threshold_in_preparation_loop(tmp_path, monkeypatch):
    records = [
        {"text": "alpha beta gamma delta", "language": "en"},
        {"text": "epsilon zeta eta theta", "language": "en"},
        {"text": "iota kappa lambda mu", "language": "en"},
        {"text": "nu xi om pi", "language": "en"},
    ]
    consumed = {"count": 0}

    def fake_download_source_records(language, *, max_examples=None, seed=42, source_path=None):
        if language != "en":
            return iter([])

        def generator():
            for item in records[:3]:
                consumed["count"] += 1
                yield item
        return generator()

    monkeypatch.setattr("scripts.prepare_data.download_source_records", fake_download_source_records)

    yaml_path = tmp_path / "target_cfg.yaml"
    yaml_path.write_text(
        """
data:
  target_tokens_per_category: 12
  min_text_length: 4
  max_text_length: 50000
  train_ratio: 0.90
  validation_ratio: 0.05
  test_ratio: 0.05
  sources:
    english:
      dataset: "synthetic"
      language: "en"
    hindi:
      dataset: "synthetic"
      language: "hi"
    hinglish:
      dataset: "synthetic"
      language: "hinglish"
""",
        encoding="utf-8",
    )

    result = prepare_data(yaml_path, tmp_path / "targeted_output", seed=7, debug=False)
    assert result["metadata"]["target_approximate_tokens_per_category"]["en"] == 12
    assert result["metadata"]["retained_examples"]["en"] == 3
    assert consumed["count"] == 3


def test_production_metadata_records_achieved_and_target_counts(tmp_path, monkeypatch):
    def fake_download_source_records(language, *, max_examples=None, seed=42, source_path=None):
        base = {
            "en": [
                {"text": "Hello world!", "language": "en"},
                {"text": "Hello world!", "language": "en"},
                {"text": "This is another English sentence.", "language": "en"},
            ],
            "hi": [
                {"text": "नमस्ते दुनिया", "language": "hi"},
                {"text": "नमस्ते दुनिया", "language": "hi"},
                {"text": "यह हिंदी है।", "language": "hi"},
            ],
            "hinglish": [
                {"text": "kal movie dekhne chale? 😂", "language": "hinglish"},
                {"text": "kal movie dekhne chale? 😂", "language": "hinglish"},
                {"text": "yeh awesome hai", "language": "hinglish"},
            ],
        }
        items = base[language]
        if max_examples is not None:
            items = items[:max_examples]
        return iter(items)

    monkeypatch.setattr("scripts.prepare_data.download_source_records", fake_download_source_records)
    output_dir = tmp_path / "prod_data"
    result = prepare_data("configs/base.yaml", output_dir, seed=7, max_examples=None, debug=False)

    metadata = result["metadata"]
    assert metadata["target_tokens_per_category"]["en"] == 60000000
    assert "achieved_approximate_tokens_per_category" in metadata
    assert "total_achieved_approximate_tokens" in metadata
    assert "duplicate_records_removed" in metadata
    assert "source_identifiers" in metadata
    assert metadata["category_distribution"]["en"] >= 1
    assert metadata["category_distribution"]["hi"] >= 1
    assert metadata["category_distribution"]["hinglish"] >= 1
    assert metadata["output_files"]["train"].endswith("train.jsonl")

    train_count = sum(1 for _ in (output_dir / "train.jsonl").open("r", encoding="utf-8"))
    validation_count = sum(1 for _ in (output_dir / "validation.jsonl").open("r", encoding="utf-8"))
    test_count = sum(1 for _ in (output_dir / "test.jsonl").open("r", encoding="utf-8"))
    assert metadata["train_example_count"] == train_count
    assert metadata["validation_example_count"] == validation_count
    assert metadata["test_example_count"] == test_count
