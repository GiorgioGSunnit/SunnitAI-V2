"""CASE_ANALYSIS_ENABLED: _analyse_case_v2 (structured fact model + per-issue
categories and norms naming any code or law) and _select_articles_v2 (per-
issue exact fetch across any document + targeted search across every
primary legal act, not just the four/six codes the v1 path knows).

Fixes the gap behind misses c2 (Codice del Consumo artt. 128-135) and part of
c3 (L. 392/1978 art. 5) in the prod eval: _CODE_PREFIXES and
_CASE_ANALYSIS_SYSTEM's prompt only ever named six traditional codes, so a
law outside them could never be proposed, searched or fetched by the v1 path.
"""

import json
import os

import pytest

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

import src.rag.defensive_generation as dg  # noqa: E402


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def data(self):
        return self._rows


# --- _parse_norma ------------------------------------------------------------

@pytest.mark.parametrize("norma, expected", [
    # A trailing "." is stripped (it's as likely sentence punctuation as part
    # of the abbreviation); _code_of normalises "c.p" back to "c.p." anyway.
    ("art. 589 c.p.", ("589", "c.p")),
    ("art. 128 Codice del Consumo", ("128", "Codice del Consumo")),
    ("art. 5 l. 392/1978", ("5", "l. 392/1978")),
    ("articolo 584 c.p.", ("584", "c.p")),
    ("art. 415-bis c.p.p.", ("415-bis", "c.p.p")),
    ("art. 658 del codice di procedura civile", ("658", "codice di procedura civile")),
    ("", None),
    ("non un articolo", None),
])
def test_parse_norma(norma, expected):
    assert dg._parse_norma(norma) == expected


@pytest.mark.parametrize("remainder, abbr", [("c.p", "c.p."), ("c.p.p", "c.p.p."), ("c.p.", "c.p.")])
def test_code_of_normalises_parse_normas_stripped_dot(remainder, abbr):
    """_parse_norma's trailing-dot stripping doesn't break code recognition:
    _code_of treats the dot as optional either way."""
    assert dg._code_of(remainder) == abbr


# --- _analyse_case_v2 --------------------------------------------------------

def test_analyse_case_v2_keeps_the_v1_fields_and_adds_the_new_ones(monkeypatch):
    reply = {
        "parte_assistita": "Sempronio", "posizione": "indagato", "area": "penale",
        "richiesta": "strategia difensiva", "ricerca_fatti": ["schiaffo", "caduta", "marciapiede"],
        "fatti_strutturati": {
            "parti": "Sempronio (autore), Caio (vittima)",
            "condotta": "uno schiaffo al volto",
            "indizi_intenzionali": "nessuna intenzione di uccidere",
            "evento": "morte di Caio per trauma cranico",
            "nesso_causale": "schiaffo -> caduta -> trauma cranico -> morte",
            "condotta_successiva": "Sempronio presta soccorso e chiama l'ambulanza",
            "fase_procedurale": "indagini preliminari",
            "campo_non_previsto": "ignorato",
        },
        "questioni": [
            {"titolo": "Omicidio preterintenzionale", "categoria": "qualificazione",
             "norme": ["art. 584 c.p."], "parole_chiave": ["preterintenzione"]},
            {"titolo": "Concorso anomalo", "categoria": "non-una-categoria-valida",
             "norme": ["art. 116 c.p."], "parole_chiave": ["concorso"]},
        ],
        "fatti": [{"fatto": "Sempronio presta soccorso", "effetto": "favorevole", "perche": "condotta successiva"}],
    }
    monkeypatch.setattr(dg, "_call_chat", lambda *a, **k: json.dumps(reply))
    result = dg._analyse_case_v2("fatti del caso", "it")

    assert result["parte_assistita"] == "Sempronio"
    assert result["area"] == "penale"
    assert result["ricerca_fatti"] == ["schiaffo", "caduta", "marciapiede"]
    assert result["fatti_strutturati"]["nesso_causale"] == "schiaffo -> caduta -> trauma cranico -> morte"
    assert "campo_non_previsto" not in result["fatti_strutturati"]   # unknown key dropped
    assert result["questioni"][0]["categoria"] == "qualificazione"
    assert result["questioni"][1]["categoria"] == ""                  # invalid category -> cleared, not kept


def test_analyse_case_v2_fails_open(monkeypatch):
    monkeypatch.setattr(dg, "_call_chat", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("timeout")))
    result = dg._analyse_case_v2("fatti", "it")
    assert result == {"parte_assistita": "", "posizione": "", "area": "", "richiesta": "", "ricerca_fatti": [],
                      "questioni": [], "fatti": [], "fatti_strutturati": {}}


def test_analyse_case_v2_handles_a_non_dict_fatti_strutturati(monkeypatch):
    monkeypatch.setattr(dg, "_call_chat", lambda *a, **k: json.dumps({"fatti_strutturati": "non un oggetto"}))
    result = dg._analyse_case_v2("fatti", "it")
    assert result["fatti_strutturati"] == {}


# --- _fetch_any: any document, not just the six codes -----------------------

def test_fetch_any_uses_the_fast_path_for_a_standard_code(monkeypatch):
    calls = []

    def fake_fetch_articles(session, refs):
        calls.append(refs)
        return [{"d": {"name": "Codice Penale 2026"}, "s": {"name": "584.0.0"}, "_article": refs[0]}]

    monkeypatch.setattr(dg, "_fetch_articles", fake_fetch_articles)
    rows = dg._fetch_any(object(), [], [("584", "c.p.")])
    assert calls == [[("584", "c.p.")]]
    assert rows[0]["_article"] == ("584", "c.p.")


def test_fetch_any_resolves_a_non_standard_law_by_document_name(monkeypatch):
    documents = [{"id": "doc1", "name": "Codice del Consumo", "aliases": []}]
    monkeypatch.setattr(dg, "_fetch_articles", lambda session, refs: [])

    import src.rag.issue_spotting as issue_spotting
    monkeypatch.setattr(issue_spotting, "resolve_code",
                        lambda name, docs: documents[0] if "consumo" in name.lower() else None)
    monkeypatch.setattr(issue_spotting, "fetch_article",
                        lambda session, doc_id, number, *a, **k:
                        [{"d": {"id": doc_id, "name": "Codice del Consumo"}, "s": {"name": f"{number}.0.0"}}])

    rows = dg._fetch_any(object(), documents, [("128", "Codice del Consumo")])
    assert len(rows) == 1
    assert rows[0]["_article"] == ("128", "Codice del Consumo")


def test_fetch_any_skips_a_law_that_does_not_resolve(monkeypatch):
    monkeypatch.setattr(dg, "_fetch_articles", lambda session, refs: [])
    import src.rag.issue_spotting as issue_spotting
    monkeypatch.setattr(issue_spotting, "resolve_code", lambda name, docs: None)
    rows = dg._fetch_any(object(), [], [("5", "l. 999/1900")])
    assert rows == []


# --- _issue_article_refs: search across every primary act, not a prefix list

def test_issue_article_refs_finds_a_non_standard_document():
    class Session:
        def run(self, query, **params):
            return _Result([
                {"doc": "Codice del Consumo", "sec": "128.0.0", "score": 2.0},
                {"doc": "Codice Civile 2026", "sec": "1490.0.0", "score": 1.5},
            ])

    refs = dg._issue_article_refs(Session(), ["difetto", "vendita", "garanzia"])
    assert ("128", "Codice del Consumo") in refs
    assert ("1490", "c.c.") in refs


def test_issue_article_refs_empty_keywords_skips_the_query():
    class Session:
        def run(self, query, **params):
            raise AssertionError("should not query with no keywords")

    assert dg._issue_article_refs(Session(), []) == []


def test_issue_article_refs_fails_open_on_a_db_error():
    class Session:
        def run(self, query, **params):
            raise RuntimeError("db down")

    assert dg._issue_article_refs(Session(), ["qualcosa"]) == []


# --- _grouped_source_block ----------------------------------------------------

def _row(article, text, question=None, source=None):
    row = {"d": {"name": "Codice Penale 2026"}, "s": {"name": f"{article[0]}.0.0", "plain_text": text},
           "_article": article}
    if question is not None:
        row["_question"] = question
    if source:
        row["_source"] = source
    return row


def test_grouped_source_block_groups_by_question_with_headings():
    rows = [
        _row(("584", "c.p."), "testo 584", question=0),
        _row(("581", "c.p."), "testo 581", question=0),
        _row(("116", "c.p."), "testo 116", question=1),
    ]
    questioni = [{"titolo": "Omicidio preterintenzionale"}, {"titolo": "Concorso anomalo"}]
    out = dg._grouped_source_block(rows, questioni, 1500)
    assert out.index("### Omicidio preterintenzionale") < out.index("[art. 584 c.p.]") < out.index("### Concorso anomalo")
    assert "[art. 116 c.p.]" in out


def test_grouped_source_block_puts_unquestioned_rows_in_a_trailing_block():
    rows = [_row(("584", "c.p."), "testo 584", question=0), _row(("581", "c.p."), "testo 581")]
    out = dg._grouped_source_block(rows, [{"titolo": "Omicidio"}], 1500)
    assert "### Altri articoli" in out
    assert out.index("### Omicidio") < out.index("### Altri articoli")


def test_grouped_source_block_falls_back_to_flat_when_no_row_has_a_question():
    rows = [_row(("584", "c.p."), "testo 584"), _row(("581", "c.p."), "testo 581")]
    out = dg._grouped_source_block(rows, [{"titolo": "Omicidio"}], 1500)
    assert "###" not in out
    assert out == dg._source_block(rows, 1500)
