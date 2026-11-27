"""v79.19.5 (item 7): the template picker's Italian stemmer (document_generation._stem_word), in a module with no model
import, so that posture.charged_by_name can read it (document_generation builds a chat client at import)."""

import re

STEM_SUFFIXES = ("zioni", "zione", "menti", "mento", "ali", "ale")


def stem_word(word: str) -> str:
    """Strip a common Italian noun/adjective ending, only if the remaining stem is > 3 chars."""
    for suffix in STEM_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) > 3:
            return word[: -len(suffix)]
    stem = re.sub(r"[oaie]s?$", "", word)
    if stem != word and len(stem) > 3:
        return stem
    return word
