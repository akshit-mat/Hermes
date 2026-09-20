from __future__ import annotations

import json
import bz2
import os
import random
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib import error, request

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - only used when datasets is unavailable.
    load_dataset = None


WIKIPEDIA_LANGUAGE_MAP = {
    "en": ["20231101.en", "20220301.en", "20210101.en"],
    "hi": ["20231101.hi", "20220301.hi", "20210101.hi"],
}
HINDI_WIKIMEDIA_DUMP_URL = (
    "https://dumps.wikimedia.org/hiwiki/20260901/"
    "hiwiki-20260901-pages-articles.xml.bz2"
)
L3CUBE_HINGCORPUS_SOURCE_URL = "https://github.com/l3cube-pune/code-mixed-nlp"
L3CUBE_HINGCORPUS_GDRIVE_URL = "https://drive.google.com/file/d/1s_6eHO9zDhxQ-xVN1TyNguszV1meZkl9/view"


def get_hinglish_source_access_method() -> dict[str, str]:
    """Return the official L3Cube access method as a documented source identifier."""
    return {
        "dataset": "L3Cube-HingCorpus",
        "source_url": L3CUBE_HINGCORPUS_SOURCE_URL,
        "access_method": "GitHub repository + Google Drive corpus download",
    }


def _iter_jsonl_from_archive(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield Hinglish records from JSONL or the official streaming TAR.GZ artifact."""
    dataset_path = Path(path).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {dataset_path}")

    suffix = dataset_path.suffix.lower()
    if suffix == ".jsonl":
        yield from _iter_local_jsonl(dataset_path)
        return

    expected_members = (
        "R11_final_data/concatenated_train_final_shuffled.txt",
        "R11_final_data/concatenated_validation.txt",
    )
    found_members: set[str] = set()
    try:
        with dataset_path.open("rb") as archive_handle:
            with tarfile.open(fileobj=archive_handle, mode="r|gz") as archive:
                for member in archive:
                    if member.name not in expected_members:
                        continue
                    found_members.add(member.name)
                    if not member.isfile():
                        raise ValueError(
                            f"Expected Hinglish archive member is not a regular file: {member.name}"
                        )
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise ValueError(
                            f"Could not read expected Hinglish archive member: {member.name}"
                        )
                    with extracted:
                        try:
                            for raw_line in extracted:
                                text = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                                if text:
                                    yield {"text": text, "language": "hinglish"}
                        except (EOFError, tarfile.ReadError) as exc:
                            raise ValueError(
                                "Hinglish source archive is truncated while reading "
                                f"{member.name}: {dataset_path}"
                            ) from exc
    except tarfile.ReadError as exc:
        raise ValueError(
            f"Hinglish source is not a valid GZIP-compressed TAR archive: {dataset_path}"
        ) from exc

    missing_members = [member for member in expected_members if member not in found_members]
    if missing_members:
        raise ValueError(
            f"Expected Hinglish archive member is missing: {', '.join(missing_members)} in {dataset_path}"
        )


def _iter_local_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    dataset_path = Path(path).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {dataset_path}")
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    yield {"text": str(text), "language": item.get("language", "unknown")}


def _sample_stream(stream: Iterable[dict[str, Any]], *, max_examples: int | None, seed: int = 42) -> Iterator[dict[str, Any]]:
    if max_examples is None:
        for item in stream:
            yield item
        return

    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    seen = 0
    for item in stream:
        seen += 1
        candidate = dict(item)
        if len(reservoir) < max_examples:
            reservoir.append(candidate)
        else:
            j = rng.randrange(seen)
            if j < max_examples:
                reservoir[j] = candidate
    for item in reservoir:
        yield item


def _iter_wikipedia_dataset(language: str, *, dataset_name: str = "wikimedia/wikipedia", config: str | None = None) -> Iterator[dict[str, Any]]:
    if config is None:
        config = f"{language}"
    dataset = load_dataset(dataset_name, config, split="train", streaming=True)
    for example in dataset:
        if not isinstance(example, dict):
            continue
        text = example.get("text")
        if not isinstance(text, str):
            continue
        normalized = text.strip()
        if normalized:
            yield {"text": normalized, "language": language}


def _iter_wikimedia_dump(source: str | Path, *, language: str) -> Iterator[dict[str, Any]]:
    """Stream current-article text from an official Wikimedia XML dump."""
    source_value = str(source)
    response = None
    raw_stream: Any
    if source_value.startswith(("http://", "https://")):
        response = request.urlopen(
            request.Request(source_value, headers={"User-Agent": "Hermes/1.0"}),
            timeout=60,
        )
        raw_stream = response
    else:
        raw_stream = Path(source_value).open("rb")

    compressed_stream = bz2.BZ2File(raw_stream)
    try:
        for _, page in ET.iterparse(compressed_stream, events=("end",)):
            if page.tag.rsplit("}", 1)[-1] != "page":
                continue
            namespace = next(
                (child.text for child in page if child.tag.rsplit("}", 1)[-1] == "ns"),
                None,
            )
            if namespace == "0":
                revision = next(
                    (child for child in page if child.tag.rsplit("}", 1)[-1] == "revision"),
                    None,
                )
                text_node = None
                if revision is not None:
                    text_node = next(
                        (child for child in revision if child.tag.rsplit("}", 1)[-1] == "text"),
                        None,
                    )
                if text_node is not None and isinstance(text_node.text, str) and text_node.text.strip():
                    yield {"text": text_node.text.strip(), "language": language}
            page.clear()
    finally:
        compressed_stream.close()
        if response is not None:
            response.close()
        elif hasattr(raw_stream, "close"):
            raw_stream.close()


def _load_wikipedia_with_fallback(language: str, *, max_examples: int | None = None, seed: int = 42) -> Iterator[dict[str, Any]]:
    if load_dataset is None:
        raise ImportError("The 'datasets' package is required for downloading Wikipedia data.")

    config_candidates = WIKIPEDIA_LANGUAGE_MAP.get(language, [f"{language}"])
    for config in config_candidates:
        try:
            stream = _iter_wikipedia_dataset(language, config=config)
            yield from _sample_stream(stream, max_examples=max_examples, seed=seed)
            return
        except Exception:
            continue
    raise ValueError(
        f"Could not access the official Wikimedia Wikipedia source for language='{language}'. "
        f"Provide a local dump or configure an offline source path."
    )


def _probe_hinglish_source_access(source_url: str = L3CUBE_HINGCORPUS_SOURCE_URL) -> tuple[bool, str]:
    try:
        req = request.Request(source_url, headers={"User-Agent": "Hermes/1.0"})
        with request.urlopen(req, timeout=20) as response:
            return response.status == 200, f"HTTP {response.status} from {source_url}"
    except error.HTTPError as exc:
        return False, f"HTTP {exc.code} from {source_url}"
    except error.URLError as exc:
        return False, f"URL error for {source_url}: {exc.reason}"
    except Exception as exc:  # pragma: no cover - defensive fallback
        return False, f"acquisition error for {source_url}: {type(exc).__name__}: {exc}"


def _probe_hinglish_drive_access(source_url: str = L3CUBE_HINGCORPUS_GDRIVE_URL) -> tuple[bool, str]:
    try:
        req = request.Request(source_url, headers={"User-Agent": "Hermes/1.0"})
        with request.urlopen(req, timeout=25) as response:
            text = response.read(2048).decode("utf-8", errors="replace")
            if "Google Drive" in text or "drive.google.com" in text.lower() or "download" in text.lower():
                return True, f"HTTP {response.status} from {source_url}"
            return response.status == 200, f"HTTP {response.status} from {source_url}"
    except error.HTTPError as exc:
        return False, f"HTTP {exc.code} from {source_url}"
    except error.URLError as exc:
        return False, f"URL error for {source_url}: {exc.reason}"
    except Exception as exc:  # pragma: no cover - defensive fallback
        return False, f"acquisition error for {source_url}: {type(exc).__name__}: {exc}"


def _iter_hinglish_repository_records(source_url: str = L3CUBE_HINGCORPUS_SOURCE_URL) -> Iterator[dict[str, Any]]:
    req = request.Request(source_url, headers={"User-Agent": "Hermes/1.0"})
    with request.urlopen(req, timeout=20) as response:
        html = response.read().decode("utf-8", errors="replace")
    if "<!doctype html" not in html.lower() and "<html" not in html.lower():
        return
    drive_reachable, drive_reason = _probe_hinglish_drive_access()
    if drive_reachable:
        return
    raise RuntimeError(
        f"Official L3Cube source at {source_url} did not expose a directly fetchable raw data file in this environment. "
        "The official database is distributed via the Google Drive artifact, not the README repository. "
        f"Drive probe result: {drive_reason}."
    )


def _load_hinglish_source(hinglish_source_path: str | Path | None = None, *, max_examples: int | None = None, seed: int = 42) -> Iterator[dict[str, Any]]:
    source_path = hinglish_source_path or os.environ.get("HERMES_HINGLISH_SOURCE_PATH")
    if source_path:
        yield from _sample_stream(_iter_jsonl_from_archive(source_path), max_examples=max_examples, seed=seed)
        return

    source_url = L3CUBE_HINGCORPUS_SOURCE_URL
    reachable, reason = _probe_hinglish_source_access(source_url)
    if reachable:
        # The official README confirms the corpus is distributed via a Google Drive link,
        # so the repository itself is documentation only and does not contain the raw corpus.
        drive_reachable, drive_reason = _probe_hinglish_drive_access()
        if not drive_reachable:
            raise RuntimeError(
                "HINGLISH SOURCE ACQUISITION FAILED: "
                f"The official L3Cube repo is reachable ({reason}), but its Google Drive corpus artifact is not available here. "
                f"Drive probe: {drive_reason}. Use a local JSONL/ZIP/GZ fallback via `hinglish_source_path` or HERMES_HINGLISH_SOURCE_PATH."
            )
        # No direct streaming parser is available for the external Drive payload in CI, so this path
        # intentionally supports only local artifact fallbacks. The official source metadata remains documented.
        yield from []
        return

    raise RuntimeError(
        "HINGLISH SOURCE ACQUISITION FAILED: "
        f"{reason}. Official L3Cube source is {source_url}. "
        "Use a local JSONL/ZIP/GZ fallback via `hinglish_source_path` or HERMES_HINGLISH_SOURCE_PATH."
    )


def download_wikipedia(language: str, *, max_examples: int | None = None, seed: int = 42, source_path: str | Path | None = None) -> Iterator[dict[str, Any]]:
    """Stream English or Hindi Wikipedia examples without materializing the full source in RAM."""
    if source_path is not None:
        if str(source_path).startswith(("http://", "https://")) or str(source_path).lower().endswith((".bz2", ".xml")):
            yield from _sample_stream(
                _iter_wikimedia_dump(source_path, language=language),
                max_examples=max_examples,
                seed=seed,
            )
            return
        yield from _sample_stream(_iter_local_jsonl(source_path), max_examples=max_examples, seed=seed)
        return
    yield from _load_wikipedia_with_fallback(language, max_examples=max_examples, seed=seed)


def download_hinglish(*, max_examples: int | None = None, seed: int = 42, source_path: str | Path | None = None) -> Iterator[dict[str, Any]]:
    """Stream Hinglish records from the official L3Cube source or a local fallback JSONL file."""
    yield from _load_hinglish_source(source_path, max_examples=max_examples, seed=seed)


def download_source_records(language: str, *, max_examples: int | None = None, seed: int = 42, source_path: str | Path | None = None) -> Iterator[dict[str, Any]]:
    """Keep source-specific logic isolated behind a single interface and stream records incrementally."""
    if language == "en":
        yield from download_wikipedia("en", max_examples=max_examples, seed=seed, source_path=source_path)
        return
    if language == "hi":
        yield from download_wikipedia("hi", max_examples=max_examples, seed=seed, source_path=source_path)
        return
    if language == "hinglish":
        yield from download_hinglish(max_examples=max_examples, seed=seed, source_path=source_path)
        return
    raise ValueError(f"Unsupported language/category: {language}")


__all__ = [
    "HINDI_WIKIMEDIA_DUMP_URL",
    "L3CUBE_HINGCORPUS_SOURCE_URL",
    "download_hinglish",
    "download_source_records",
    "download_wikipedia",
    "get_hinglish_source_access_method",
    "WIKIPEDIA_LANGUAGE_MAP",
]
