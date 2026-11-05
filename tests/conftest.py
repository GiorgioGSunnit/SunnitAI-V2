"""Every test starts with every feature flag unset (v48): the suite passes whatever flags the environment sets -- a
server's .env.v45 with them all on included -- and a test that needs a flag sets it (monkeypatch.setenv). The flags are
the names src/rag reads with feature_flags.enabled() or keeps in a *_FLAG constant."""
import pathlib
import re

import pytest

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "rag"
_READ = re.compile(r"""(?:enabled|_enabled)\(\s*["']([A-Z][A-Z0-9_]+)["']|^[A-Z_]*FLAG[A-Z_]*\s*=\s*["']([A-Z][A-Z0-9_]+)["']""",
                   re.MULTILINE)


def _flags() -> set:
    names = set()
    for path in _SRC.rglob("*.py"):
        for m in _READ.finditer(path.read_text(encoding="utf-8", errors="replace")):
            names.add(m.group(1) or m.group(2))
    return names


FLAGS = frozenset(_flags())


@pytest.fixture(autouse=True)
def _flags_unset(monkeypatch):
    for name in FLAGS:
        monkeypatch.delenv(name, raising=False)
