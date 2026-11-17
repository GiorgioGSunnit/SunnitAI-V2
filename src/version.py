"""v77: the release this checkout runs -- the VERSION file at the repository root, «<semver>+<git short sha>», written by
scripts/release.sh (the sha is the code commit the release names; the release commit itself only writes VERSION). It
is read once per process: /api/version returns it, every case-first answer ends with its footer («SunnitAI <version>»),
the held-out records, the sims and the nightly carry it."""

import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "VERSION"
UNKNOWN = "unknown"
FOOTER_PREFIX = "SunnitAI "
_FOOTER_RE = re.compile(r"\n*^SunnitAI \S+[ \t]*\Z", re.M)


@lru_cache(maxsize=1)
def current() -> str:
    """The VERSION file's first line, «unknown» when there is none (a checkout before v77, a copy without the file)."""
    try:
        line = VERSION_FILE.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    except (OSError, IndexError):
        return UNKNOWN
    return line or UNKNOWN


def footer() -> str:
    return FOOTER_PREFIX + current()


def strip_footer(text: str) -> str:
    """The answer without its footer line (a document draft, a diff against the previous answer)."""
    return _FOOTER_RE.sub("", text or "")


def with_footer(text: str) -> str:
    """The answer ending with exactly one footer line -- an answer that already has one (any version) gets this one."""
    body = strip_footer(text).rstrip()
    return f"{body}\n\n{footer()}" if body else footer()


def has_footer(text: str, version: str = "") -> bool:
    """The answer's last line is a footer (this version's, when given)."""
    last = (text or "").rstrip().rsplit("\n", 1)[-1].strip()
    return last == FOOTER_PREFIX + version if version else bool(re.fullmatch(r"SunnitAI \S+", last))
