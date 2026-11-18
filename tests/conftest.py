"""Every test starts with every feature flag unset (v48): the suite passes whatever flags the environment sets -- a
server's .env.v45 with them all on included -- and a test that needs a flag sets it (monkeypatch.setenv). The flags are
the names src/rag reads with feature_flags.enabled() or keeps in a *_FLAG constant."""
import os
import pathlib
import re
import tempfile

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

# v77: no test writes the server's runtime state or held-out set (/opt/chatbot/state, /opt/chatbot/heldout) -- set before
# any module computes its paths at import
os.environ.setdefault("STATE_DIR", tempfile.mkdtemp(prefix="sunnit_state_"))
os.environ.setdefault("HELDOUT_DIR", tempfile.mkdtemp(prefix="sunnit_heldout_"))


@pytest.fixture(autouse=True)
def _flags_unset(monkeypatch):
    for name in FLAGS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _intake_without_model(monkeypatch):
    """v61: no test reaches a model through the intake (src/rag/intake.ask_model): the rules and the defaults decide,
    as with the model unavailable. A test of the model's part stubs ask_model itself."""
    from src.rag import intake
    monkeypatch.setattr(intake, "ask_model", lambda *a, **k: {})


@pytest.fixture(autouse=True)
def _corpus_names_unknown(monkeypatch):
    """v79.9 (B2): no test reads the graph's document names -- unknown, so no missing-laws line, unless a test sets them
    (corpus_laws.set_names, or missing_laws(ctx, known))."""
    from src.rag import corpus_laws
    monkeypatch.setitem(corpus_laws._NAMES, "names", None)
