"""Session language handling: Italian default, LLM detection on first long message, explicit switches."""

from __future__ import annotations

import logging
import re
from typing import Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from .ai_chat import _call_chat

logger = logging.getLogger(__name__)

SessionLang = Literal["it", "en", "es"]

DEFAULT_LANGUAGE: SessionLang = "it"

# Minimum length / words to run auto-detection on the first user message
_MIN_CHARS_FOR_AUTO_DETECT = 48
_MIN_WORDS_FOR_AUTO_DETECT = 8

_LANG_NAMES = {
    "it": "Italian",
    "en": "English",
    "es": "Spanish",
}


def language_display_name(code: str) -> str:
    return _LANG_NAMES.get(code, "Italian")


def normalize_lang(code: Optional[str]) -> SessionLang:
    if not code:
        return DEFAULT_LANGUAGE
    c = code.strip().lower()[:2]
    if c in ("it", "en", "es"):
        return c  # type: ignore[return-value]
    return DEFAULT_LANGUAGE


def should_auto_detect_language(text: str) -> bool:
    t = text.strip()
    if len(t) < _MIN_CHARS_FOR_AUTO_DETECT:
        return False
    words = re.findall(r"\w+", t)
    return len(words) >= _MIN_WORDS_FOR_AUTO_DETECT


def detect_language_llm(text: str) -> SessionLang:
    """Classify user text as it, en, or es (LLM)."""
    raw = _call_chat(
        [
            SystemMessage(
                content=(
                    "You classify the primary language of a user message for a legal assistant. "
                    "Allowed outputs exactly one token: it, en, or es "
                    "(Italian, English, Spanish). "
                    "If uncertain or mixed with no clear primary language, output it."
                )
            ),
            HumanMessage(
                content=f"Message:\n{text}\n\nReply with only: it, en, or es"
            ),
        ]
    )
    token = re.sub(r"\s+", "", (raw or "").lower())[:2]
    if token in ("it", "en", "es"):
        return token  # type: ignore[return-value]
    logger.warning("Language classifier returned %r; defaulting to it", raw)
    return DEFAULT_LANGUAGE


def detect_explicit_language_switch_llm(text: str, current: SessionLang) -> Optional[SessionLang]:
    """If the user explicitly asks to switch conversation language, return the new code."""
    raw = _call_chat(
        [
            SystemMessage(
                content=(
                    "The user may ask to continue in Italian, English, or Spanish. "
                    "If they explicitly request a language change for the conversation, "
                    "reply with exactly one token: it, en, or es. "
                    "If they do not request a language change, reply with exactly: no"
                )
            ),
            HumanMessage(
                content=(
                    f"Current conversation language code: {current}\n"
                    f"User message:\n{text}\n\n"
                    "Your reply (one of: it, en, es, no):"
                )
            ),
        ]
    )
    reply = (raw or "").strip().lower()
    if reply in ("no", "none", "n"):
        return None
    token = re.sub(r"[^a-z]", "", reply)[:2]
    if token in ("it", "en", "es") and token != current:
        return token  # type: ignore[return-value]
    return None
