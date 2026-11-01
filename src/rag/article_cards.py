"""ARTICLE_CARDS_ENABLED (default false): the writer reads an article's card, made once from its text by
scripts/article_cards_bulk.py (ARTICLE_CARDS_FILE, default data/article_cards.jsonl; nothing in Neo4j).

An article over its budget reaches the writer as its card instead of a selection of its commi; the others as a short
text plus the card. Only the writer: the reviews judge against the official text (_source_block's official=True), never a
summary of it. A card is given only if every check passed and its text_hash is this row's text: a changed article, a
fragment found by a search, or a doubtful card leaves the official text as it was.
"""

import hashlib
import json
import logging
import os
import threading
from typing import Any, Dict, Optional, Tuple

from . import comma_material, feature_flags

logger = logging.getLogger(__name__)

FLAG = "ARTICLE_CARDS_ENABLED"
FILE_ENV = "ARTICLE_CARDS_FILE"
DEFAULT_FILE = os.path.join("data", "article_cards.jsonl")
SHORT_TEXT = 400          # the text beside the card, for an article within its budget
CARD_ROOM = 2             # a card may take this many times the article's budget
HEAD = "Scheda di sintesi (non è il testo ufficiale: non citarla tra virgolette):"


def enabled() -> bool:
    return feature_flags.enabled(FLAG)


def text_hash(text: Optional[str]) -> str:
    """Of the text's words: the same article, whatever its line breaks."""
    return hashlib.sha1(" ".join((text or "").split()).encode("utf-8")).hexdigest()[:16]


def usable(record: Dict[str, Any]) -> bool:
    return bool(record.get("card")) and not any((record.get("checks") or {}).values())


_lock = threading.Lock()
_cards: Dict[Tuple[str, str], Dict[str, Any]] = {}
_signature: Optional[tuple] = None


def _key(doc: Optional[str], number: Any) -> Tuple[str, str]:
    return (doc or "", str(number).strip().lower())


def _store() -> Dict[Tuple[str, str], Dict[str, Any]]:
    """{(doc, article): record}, the last line of each; read again when the file changes."""
    global _cards, _signature
    path = os.getenv(FILE_ENV) or DEFAULT_FILE
    if not os.path.isfile(path):
        return {}
    signature = (path, os.path.getsize(path), os.path.getmtime(path))
    with _lock:
        if signature != _signature:
            cards: Dict[Tuple[str, str], Dict[str, Any]] = {}
            try:
                with open(path, encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            record = json.loads(line)
                        except ValueError:
                            continue
                        if isinstance(record, dict):
                            cards[_key(record.get("doc"), record.get("article"))] = record
            except OSError as exc:
                logger.warning("article cards: %s not readable: %s", path, exc)
                return {}
            _cards, _signature = cards, signature
            logger.info("article cards: %d from %s", len(cards), path)
        return _cards


def material(row: Dict[str, Any], chars: int) -> Optional[str]:
    """The article's block with its card, within `chars` (a shrunk row's own limit first); None for the official text."""
    if not enabled() or not row.get("_article"):
        return None
    section = row.get("s") or {}
    whole = " ".join((section.get("plain_text") or "").split())
    record = _store().get(_key((row.get("d") or {}).get("name"), row["_article"][0]))
    if not whole or not record or not usable(record) or record.get("text_hash") != text_hash(whole):
        return None
    limit = min(chars, row.get("_comma_chars") or chars, len(row.get("_passage") or "") or chars)
    focus, named = row.get(comma_material.FOCUS_KEY) or "", row.get(comma_material.NAMED_KEY) or ()
    if len(whole) > chars:
        return f"{HEAD}\n{comma_material.trim(record['card'], limit * CARD_ROOM, focus)}"
    short = comma_material.trim(whole, min(SHORT_TEXT, limit), focus, named)
    return f"{short}\n{HEAD}\n{comma_material.trim(record['card'], limit, focus)}"
