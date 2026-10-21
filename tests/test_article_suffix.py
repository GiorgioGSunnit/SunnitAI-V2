"""article_suffix.py: every Latin suffix spelling normalised to the canonical form
the BE parser writes (SunnitAI-BE lex_package/parsing_utils/article_suffix.py,
an identical copy), and lookups that still find a legacy spelling."""

import os

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

import src.rag.defensive_generation as dg  # noqa: E402
from src.rag import issue_spotting  # noqa: E402
from src.rag.article_suffix import canonical_suffix, normalize_number, number_variants  # noqa: E402


def test_spellings_normalise_to_one_form():
    assert canonical_suffix("novies") == "nonies" and canonical_suffix("quattuordecies") == "quaterdecies"
    assert canonical_suffix("ter  decies") == "terdecies" and canonical_suffix("vicies semel") == "unvicies"
    assert normalize_number("2545 octodecies") == "2545-octiesdecies"
    assert normalize_number("452-terdecies") == "452-terdecies"
    assert number_variants("21-nonies") == ["21-nonies", "21-novies"]
    assert number_variants("2051") == ["2051"]


def test_cited_articles_read_long_suffixes_whole():
    """The old list stopped at "decies": 452-terdecies was read as 452-ter."""
    refs = dg._refs_in("art. 452-terdecies c.p., art. 21-novies c.p.a. e art. 2545-octodecies c.c.")
    assert refs == [("452-terdecies", "c.p."), ("21-nonies", "c.p.a."), ("2545-octiesdecies", "c.c.")]


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def data(self):
        return self._rows


class _Session:
    """Holds one Section, named with a legacy spelling."""
    def __init__(self, name):
        self.name, self.asked = name, []

    def run(self, query, **params):
        self.asked.append(params.get("num"))
        if "IN [$zero_zero, $num]" in query and self.name in (params["zero_zero"], params["num"]):
            return _Result([{"d": {"id": "doc"}, "s": {"name": self.name}, "name": self.name}])
        return _Result([])


def test_fetch_article_tries_the_canonical_spelling_then_legacy_ones():
    session = _Session("21-novies.0.0")
    rows = issue_spotting.fetch_article(session, "doc", "21-nonies")
    assert rows and rows[0]["s"]["name"] == "21-novies.0.0"
    assert session.asked == ["21-nonies", "21-nonies", "21-novies"]       # exact, fragments, then legacy
