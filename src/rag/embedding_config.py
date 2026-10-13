"""How V2 embeds questions and which vectors it searches.

Oct 2026: LangChain's OpenAIEmbeddings, with its default settings, tokenizes text
with OpenAI's cl100k_base and sends token IDs; the Qwen embedding server reads
them with its own vocabulary, so every question and every stored Section vector
was made from gibberish (the right section was in the real index's top 30 for 0
of 400 test questions; 94% with plain text and the Qwen query instruction, on a
sample). The fix is switched on during the rollout, together with the recomputed
vectors (the BE backfill). The defaults here reproduce the old behaviour exactly,
so this code can be deployed first.

Read once at import (after loading .env here, whatever imports this first); a
change needs a restart.
  EMBEDDING_QUERY_MODE    token_ids (default, old behaviour) | text
  EMBEDDING_INDEX_SUFFIX  "" (default) | e.g. "_v2": added to the migrated indexes only
  EMBEDDING_PROPERTY      embedding (default) | e.g. embedding_v2: read by the document-scoped search
  EMBEDDING_MIN_SCORE_SECTION / _COMMENTATO / _FISCALITA: text-mode thresholds (default 0.55)
In text mode, indexes that still hold token-ID vectors are not searched
(TEXT_MODE_SKIPPED_INDEXES).
"""

import logging
import os
import re
from typing import Any, Dict

from dotenv import load_dotenv

# Not left to whoever imports this first: lookups.py imports this module before
# ai_chat (which also loads .env), and a mode set only in .env would be ignored.
load_dotenv()

logger = logging.getLogger(__name__)

QUERY_MODES = ("token_ids", "text")

# Qwen3-Embedding wants an instruction on questions and none on documents.
QUERY_INSTRUCTION = (
    "Instruct: Given a legal question in Italian, retrieve the passages of laws, rulings and "
    "commentary that answer it\nQuery: "
)

# The indexes whose vectors the backfill recomputes. Every other index keeps
# its name and its token-ID thresholds.
MIGRATED_INDEXES = ("section_embeddings", "commentato_section_embeddings", "fiscalita_section_embeddings")

# Not re-embedded yet and still holding token-ID vectors (35 Documents): a
# plain-text question scores 0.07-0.35 against them at random, above their 0.25
# threshold, so in text mode they are not searched until they are re-embedded.
TEXT_MODE_SKIPPED_INDEXES = ("document_embeddings",)


def _choice(name: str, default: str, allowed: tuple) -> str:
    value = (os.getenv(name) or default).strip().lower()
    if value not in allowed:
        logger.warning("%s=%r is not one of %s: using %r", name, value, allowed, default)
        return default
    return value


def _identifier(name: str, default: str, pattern: str) -> str:
    """Index suffixes and property names are written into Cypher, so only
    plain identifier characters are accepted."""
    value = os.getenv(name)
    value = default if value is None else value.strip()
    if not re.fullmatch(pattern, value):
        logger.warning("%s=%r is not a valid name: using %r", name, value, default)
        return default
    return value


def _score(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        logger.warning("%s=%r is not a number: using %s", name, os.getenv(name), default)
        return default


QUERY_MODE = _choice("EMBEDDING_QUERY_MODE", "token_ids", QUERY_MODES)
INDEX_SUFFIX = _identifier("EMBEDDING_INDEX_SUFFIX", "", r"[A-Za-z0-9_]*")
VECTOR_PROPERTY = _identifier("EMBEDDING_PROPERTY", "embedding", r"[A-Za-z_][A-Za-z0-9_]*")

# Text-mode thresholds. Sections: 0.55 keeps ~90% of the right sections in the
# 2026-10-12 measurement; commentato and fiscalita are provisional until measured.
TEXT_MODE_INDEX_SETTINGS: Dict[str, Dict[str, Any]] = {
    "section_embeddings": {"k": 30, "min_score": _score("EMBEDDING_MIN_SCORE_SECTION", 0.55)},
    "commentato_section_embeddings": {"k": 3, "min_score": _score("EMBEDDING_MIN_SCORE_COMMENTATO", 0.55)},
    "fiscalita_section_embeddings": {"k": 3, "min_score": _score("EMBEDDING_MIN_SCORE_FISCALITA", 0.55)},
}


def text_mode() -> bool:
    return QUERY_MODE == "text"


def searched(base: str) -> bool:
    """False for an index that the current mode leaves out."""
    return not (text_mode() and base in TEXT_MODE_SKIPPED_INDEXES)


def vector_index_name(base: str) -> str:
    """The index actually queried: the suffix applies to the migrated indexes only."""
    return base + INDEX_SUFFIX if base in MIGRATED_INDEXES else base


def openai_embedding_kwargs() -> Dict[str, Any]:
    """Extra OpenAIEmbeddings settings, shared by the question client and V2's writer.

    Text mode sends plain text (check_embedding_ctx_length=False). Not
    tiktoken_enabled=False: that looks for a HuggingFace tokenizer called
    "qwen3-embedding-8b", which doesn't exist. truncate_prompt_tokens lets the
    server cut the rare text over its 8192-token limit instead of refusing it.
    """
    kwargs: Dict[str, Any] = {"timeout": 60}
    if text_mode():
        kwargs["check_embedding_ctx_length"] = False
        kwargs["model_kwargs"] = {"extra_body": {"truncate_prompt_tokens": 8000}}
    return kwargs
