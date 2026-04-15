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
    """Classify user text as it, en, or es (LLM, few-shot, max_tokens=5)."""
    raw = _call_chat(
        [
            SystemMessage(
                content=(
                    "Classify the primary language of the user message. "
                    "Reply with exactly one of: it  en  es\n\n"
                    "Examples:\n"
                    "Message: Qual è la pena prevista per frode fiscale?\nLanguage: it\n"
                    "Message: What penalty applies to tax fraud?\nLanguage: en\n"
                    "Message: ¿Cuál es la sanción por fraude fiscal?\nLanguage: es\n"
                    "Message: Dimmi le clausole del contratto n. 12.\nLanguage: it\n"
                    "Message: Show me the clauses of contract no. 12.\nLanguage: en"
                )
            ),
            HumanMessage(
                content=f"Message: {text}\nLanguage:"
            ),
        ],
        max_tokens=5,
    )
    token = re.sub(r"\s+", "", (raw or "").lower())[:2]
    if token in ("it", "en", "es"):
        return token  # type: ignore[return-value]
    logger.warning("Language classifier returned %r; defaulting to it", raw)
    return DEFAULT_LANGUAGE


# Regex patterns for explicit language switch requests (avoids LLM call in >95% of messages)
_SWITCH_PATTERNS: dict[str, re.Pattern[str]] = {
    "it": re.compile(
        r"\b(?:(?:switch|change|continue|respond|reply|answer|speak|write|talk)\s+(?:in|to)\s+italian"
        r"|(?:rispondi|continua|parla|scrivi)\s+in\s+italiano"
        r"|in\s+italiano\s*(?:per\s+favore|prego)?)\b",
        re.IGNORECASE,
    ),
    "en": re.compile(
        r"\b(?:(?:switch|change|continue|respond|reply|answer|speak|write|talk)\s+(?:in|to)\s+english"
        r"|(?:rispondi|continua|parla|scrivi)\s+in\s+inglese"
        r"|(?:responde|continúa|habla|escribe)\s+en\s+inglés"
        r"|in\s+english\s*please?)\b",
        re.IGNORECASE,
    ),
    "es": re.compile(
        r"\b(?:(?:switch|change|continue|respond|reply|answer|speak|write|talk)\s+(?:in|to)\s+spanish"
        r"|(?:rispondi|continua|parla|scrivi)\s+in\s+spagnolo"
        r"|(?:responde|continúa|habla|escribe)\s+en\s+español"
        r"|in\s+spanish\s*please?)\b",
        re.IGNORECASE,
    ),
}

# Quick pre-filter: if message doesn't contain any of these tokens, skip entirely
_SWITCH_HINT_TOKENS = frozenset({
    "switch", "change", "continue", "respond", "reply", "answer", "speak", "write", "talk",
    "italian", "english", "spanish", "italiano", "inglese", "spagnolo", "inglés", "español",
    "rispondi", "continua", "parla", "scrivi", "responde", "continúa", "habla", "escribe",
})


def detect_explicit_language_switch(text: str, current: SessionLang) -> Optional[SessionLang]:
    """Detect an explicit language-switch request using fast regex, falling back to LLM only if ambiguous."""
    lowered = text.lower()

    # Fast path: skip if no switch-related tokens are present
    words = set(re.findall(r"\w+", lowered))
    if not words & _SWITCH_HINT_TOKENS:
        return None

    # Regex path: check each language pattern
    for lang_code, pattern in _SWITCH_PATTERNS.items():
        if pattern.search(text):
            if lang_code != current:
                logger.info("Language switch detected via regex: %s → %s", current, lang_code)
                return lang_code  # type: ignore[return-value]
            return None

    # Ambiguous case: hint tokens present but no regex match — fall back to LLM
    logger.debug("Language switch hint tokens found but no regex match; falling back to LLM")
    return _detect_explicit_language_switch_llm(text, current)


def _detect_explicit_language_switch_llm(text: str, current: SessionLang) -> Optional[SessionLang]:
    """LLM fallback for ambiguous language switch detection."""
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
        ],
        max_tokens=10,
    )
    reply = (raw or "").strip().lower()
    if reply in ("no", "none", "n"):
        return None
    token = re.sub(r"[^a-z]", "", reply)[:2]
    if token in ("it", "en", "es") and token != current:
        return token  # type: ignore[return-value]
    return None
