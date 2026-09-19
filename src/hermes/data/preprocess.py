from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from typing import Any


_WHITESPACE_RE = re.compile(r"[\t\r\f\v]+")
_MULTILINE_RE = re.compile(r"\n\s*\n+")
_ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\ufeff]")
_SYMBOL_BLOCK_RE = re.compile(r"^[^\w\s\u0900-\u097F\u0980-\u09FF]+$")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{8,}")
_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
_HTML_TAG_RE = re.compile(r"(?i)<\s*(?:br|p|div|a|img|script|html|body|table)[^>]*>")


def normalize_text(text: str) -> str:
    """Normalize obvious formatting issues while preserving scripts and punctuation."""
    if not isinstance(text, str):
        text = str(text)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = _ZERO_WIDTH_RE.sub("", text)
    text = text.replace("\t", " ").replace("\f", " ").replace("\v", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n +", "\n", text)
    text = re.sub(r" +\n", "\n", text)
    text = _MULTILINE_RE.sub("\n\n", text)
    text = text.strip()
    return text


def _is_symbol_heavy(text: str) -> bool:
    letters = 0
    for ch in text:
        if ch.isalnum():
            letters += 1
            continue
        if "\u0900" <= ch <= "\u097F" or "\u0980" <= ch <= "\u09FF":
            letters += 1
    if letters == 0:
        return True
    symbols = len(text) - letters
    return symbols > max(2, len(text) // 3)


def _contains_obvious_url_or_html_noise(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if _HTML_TAG_RE.search(stripped):
        return True
    if _URL_RE.search(stripped):
        if len(stripped) < 200 and stripped.count(" ") <= 4:
            return True
    return False


def filter_examples(examples: Iterable[str], min_length: int = 4, max_length: int = 50000) -> Iterator[str]:
    """Keep only usable records while being conservative about code-mixed text."""
    for example in examples:
        if example is None:
            continue
        text = normalize_text(str(example))
        if not text:
            continue
        if len(text) < min_length:
            continue
        if len(text) > max_length:
            continue
        if _SYMBOL_BLOCK_RE.match(text):
            continue
        if _REPEATED_CHAR_RE.search(text):
            continue
        if _is_symbol_heavy(text):
            continue
        if _contains_obvious_url_or_html_noise(text):
            continue
        yield text


def deduplicate_examples(records: Sequence[dict[str, Any]] | Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove exact duplicate normalized text records while retaining first occurrence."""
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        text = record.get("text")
        normalized_text = normalize_text(text) if isinstance(text, str) else ""
        if not normalized_text:
            continue
        key = normalized_text
        if key in seen:
            continue
        seen.add(key)
        deduped.append({**record, "text": normalized_text})
    return deduped


def preprocess_records(records: Iterable[dict[str, Any]], *, min_length: int = 4, max_length: int = 50000) -> list[dict[str, Any]]:
    """Normalize, filter, and deduplicate a list of records in a deterministic order."""
    cleaned: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        text = record.get("text")
        if not isinstance(text, str):
            continue
        normalized = normalize_text(text)
        if normalized and min_length <= len(normalized) <= max_length:
            cleaned.append({**record, "text": normalized})

    filtered = []
    for record in cleaned:
        text = record.get("text", "")
        if text and all(result for result in filter_examples([text], min_length=min_length, max_length=max_length)):
            filtered.append(record)

    return deduplicate_examples(filtered)


__all__ = [
    "normalize_text",
    "filter_examples",
    "deduplicate_examples",
    "preprocess_records",
]
