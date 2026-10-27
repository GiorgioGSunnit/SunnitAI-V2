"""Every article a chat answer cites carries its code or law (src/rag/article_codes.py). Invented rows."""
import os

os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")

import pytest  # noqa: E402

from src.rag import article_codes as ac  # noqa: E402


def _row(doc_name, section, doc_type="primary"):
    return {"d": {"name": doc_name, "document_type": doc_type}, "s": {"name": section}}


ROWS = [_row("Codice di procedura penale", "585.0.0"), _row("Codice di procedura penale", "593.0.0"),
        _row("Codice Penale", "52.0.0"), _row("Codice di procedura penale", "52.0.0"),
        _row("Legge 7 agosto 1990, n. 241", "21-nonies"), _row("Codice Civile", "1490.0.0"),
        _row("Cass. pen. 123/2024", "9", "interpretation"), _row("Manuale di procedura penale", "12.0.0", "special")]


@pytest.mark.parametrize("answer, expected", [
    ("Si propone entro il termine dell'art. 585.", "Si propone entro il termine dell'art. 585 c.p.p."),
    ("dell'art. 585, comma 1, lett. b)", "dell'art. 585 c.p.p., comma 1, lett. b)"),
    ("gli artt. 585 e 593 disciplinano", "gli artt. 585 e 593 c.p.p. disciplinano"),
    ("gli artt. 585 e 1490", "gli artt. 585 c.p.p. e 1490 c.c."),
    ("l'annullamento (art. 21-nonies)", "l'annullamento (art. 21-nonies l. 241/1990)"),
])
def test_a_bare_article_gets_the_one_code_the_rows_have_it_in(answer, expected):
    assert ac.complete_codes(answer, ROWS)[0] == expected


@pytest.mark.parametrize("answer", [
    "l'art. 585 c.p.p. e l'art. 1490 del codice civile",
    "Codice di procedura penale, art. 585",
    "(c.p.p., art. 593)",
    "l'art. 21-nonies della legge n. 241 del 1990",
    "l'art. 3 Cost.",
])
def test_an_article_with_its_code_is_left_alone(answer):
    out, report = ac.complete_codes(answer, ROWS)
    assert out == answer and report["bare"] == 0


def test_an_ambiguous_or_unknown_article_is_left_and_logged():
    out, report = ac.complete_codes("La legittima difesa (art. 52) e l'art. 9; l'art. 12.", ROWS)
    assert out == "La legittima difesa (art. 52) e l'art. 9; l'art. 12."
    assert report["ambiguous"] == ["52: c.p., c.p.p."] and report["unknown"] == ["9", "12"]   # rulings, books skipped


def test_an_earlier_articles_code_does_not_cover_a_later_bare_one():
    out, _ = ac.complete_codes("l'art. 1490 c.c. e l'art. 585 prevedono", ROWS)
    assert out == "l'art. 1490 c.c. e l'art. 585 c.p.p. prevedono"


def test_tagged_rows_and_law_labels():
    assert ac.codes_by_article([{"_article": ("640", "c.p."), "d": {}, "s": {}}]) == {"640": {"c.p."}}
    assert ac.law_label("D.Lgs. 36/2023 Codice dei contratti pubblici") == "d.lgs. 36/2023"
    assert ac.law_label("Testo unico enti locali 2026") == "(Testo unico enti locali)"


def test_the_synthesis_prompt_asks_for_the_code():
    from src.rag.prompts import synthesis_system_message
    assert "art. 585 c.p.p." in synthesis_system_message("it")
