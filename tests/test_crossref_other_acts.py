"""Cross-references to other acts are not looked up in the matched article's code.

broaden_article_context adds the articles a matched article's own text refers
to ("degli articoli 581 e 582"), from the same code. A reference to another act
inside that text ("articolo 186, comma 2, del decreto legislativo n. 285/1992",
"articoli 2 e 3 della legge n. 689/1981") was looked up in the same code too,
fetching unrelated articles that then reached the answer as protected rows.
"""

import os

import pytest

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

from src.rag.nodes import retrieval  # noqa: E402


class _Result:
    def data(self):
        return []


class _Session:
    def __init__(self, looked_up):
        self.looked_up = looked_up

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        self.looked_up.add(params["num"])
        return _Result()


class _Driver:
    def __init__(self):
        self.looked_up = set()

    def session(self, **config):
        return _Session(self.looked_up)


def _looked_up(text):
    driver = _Driver()
    state = {"raw_result": [{"d": {"id": "codice_penale"}, "s": {"name": "589.0.0", "plain_text": text}}]}
    retrieval.broaden_article_context(state, driver, "neo4j")
    return driver.looked_up


@pytest.mark.parametrize("text, expected", [
    ("La pena è aumentata nei casi previsti dagli articoli 581 e 582.", {"581", "582"}),
    ("Si applica l'art. 416-bis e l'articolo 61, n. 2, del presente codice.", {"416-bis", "61"}),
    ("Restano ferme le disposizioni dello stesso articolo 590 e dell'articolo 583.", {"590", "583"}),
])
def test_references_within_the_code_are_followed(text, expected):
    assert _looked_up(text) == expected


@pytest.mark.parametrize("text", [
    "con violazione dell'articolo 186, comma 2, lettera c), del decreto legislativo 30 aprile 1992, n. 285",
    "ai sensi degli articoli 2 e 3 della legge 24 novembre 1981, n. 689",
    "nei casi di cui all'art. 444 c.p.p.",
    "secondo l'articolo 2043 del codice civile",
    "nel rispetto dell'art. 27 della Costituzione",
    "come previsto dall'art. 7 del d.lgs. n. 231/2001",
    "ai sensi dell'art. 5 del d.P.R. n. 309/1990",
])
def test_references_to_another_act_are_skipped(text):
    assert _looked_up(text) == set()


def test_a_text_with_both_keeps_only_the_codes_own_articles():
    text = ("Si applicano gli articoli 581 e 582; per la guida in stato di ebbrezza l'articolo 186, "
            "comma 2, del decreto legislativo n. 285 del 1992.")
    assert _looked_up(text) == {"581", "582"}
