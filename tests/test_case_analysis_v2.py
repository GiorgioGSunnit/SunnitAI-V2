"""CASE_ANALYSIS_ENABLED: _analyse_case_v2 (structured fact model + per-issue
categories and institutions named in words, not numbers) and
_select_articles_v2 (per-issue exact fetch across any document + targeted
search across every primary legal act, not just the four/six codes the v1
path knows).

Fixes two gaps found while testing this on prod (16 Oct):
- misses c2 (Codice del Consumo artt. 128-135) and part of c3 (L. 392/1978
  art. 5): _CODE_PREFIXES and the v1 prompt only ever named six traditional
  codes, so a law outside them could never be proposed, searched or fetched.
- check_applicability.py on Sempronio showed the model's own article numbers
  are frequently wrong from memory ("elemento soggettivo" -> art. 12 c.p.
  instead of 43; "reati alternativi" -> 597, 598 c.p. instead of 589, 586;
  589 never proposed at all) — so the schema asks for the institution's name
  instead, resolved against the real rubriche (_resolve_institution); a
  model-given number is only accepted as a last resort, and only if the
  article it names is itself titled consistently with that name.
"""

import json
import os
import re

import pytest

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

import src.rag.defensive_generation as dg  # noqa: E402
import src.rag.issue_spotting as issue_spotting  # noqa: E402


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def data(self):
        return self._rows


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
             "istituti": [{"nome": "omicidio preterintenzionale", "codice": "c.p.", "numero_indicativo": "584"}],
             "parole_chiave": ["preterintenzione"]},
            {"titolo": "Concorso anomalo", "categoria": "non-una-categoria-valida",
             "istituti": [{"nome": "concorso anomalo", "codice": "c.p.", "numero_indicativo": ""},
                          "non un dict", {"nome": "", "codice": "c.p."}],
             "parole_chiave": ["concorso"]},
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
    assert result["questioni"][0]["istituti"] == [
        {"nome": "omicidio preterintenzionale", "codice": "c.p.", "numero_indicativo": "584"}
    ]
    assert result["questioni"][1]["categoria"] == ""                  # invalid category -> cleared, not kept
    # Non-dict entries and an institution with no name are dropped, not kept.
    assert result["questioni"][1]["istituti"] == [
        {"nome": "concorso anomalo", "codice": "c.p.", "numero_indicativo": ""}
    ]


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
    monkeypatch.setattr(issue_spotting, "resolve_code", lambda name, docs: None)
    rows = dg._fetch_any(object(), [], [("5", "l. 999/1900")])
    assert rows == []


# --- _resolve_institution: the Sempronio trace (43, 586, 589) ---------------
#
# Real rubriche (titles) of the three articles the model should reach:
#   art. 43 c.p.  "Elemento psicologico del reato"
#   art. 586 c.p. "Morte o lesione come conseguenza di altro delitto"
#   art. 589 c.p. "Omicidio colposo"
# The model named them, on prod, as "elemento soggettivo" (wrong number: 12),
# "reati alternativi" (wrong numbers: 597, 598) — 589 was never proposed.

_CP_DOC = {"id": "cp-doc", "name": "Codice Penale 2026", "aliases": []}
_CP_ARTICLES = {
    "43": {"title": "Elemento psicologico del reato",
           "text": "Il delitto è doloso, o secondo l'intenzione, quando l'evento dannoso o pericoloso, che è "
                   "il risultato dell'azione od omissione e da cui la legge fa dipendere l'esistenza del "
                   "delitto, è dal soggetto preveduto e voluto come conseguenza della propria azione od "
                   "omissione; è colposo, o contro l'intenzione, quando l'evento, anche se preveduto, non è "
                   "voluto dall'agente e si verifica a causa di negligenza, imprudenza o imperizia"},
    "586": {"title": "Morte o lesione come conseguenza di altro delitto",
            "text": "Quando da un fatto preveduto come delitto doloso deriva, quale conseguenza non voluta "
                    "dal colpevole, la morte o la lesione personale di qualcuno, si applicano le disposizioni "
                    "dell'articolo precedente, ma le pene stabilite in esse sono aumentate"},
    "589": {"title": "Omicidio colposo",
            "text": "Chiunque cagiona per colpa la morte di una persona è punito con la reclusione da sei "
                    "mesi a cinque anni"},
    # Unrelated to any institution name used below — real art. 12 c.p. (abrogato,
    # ex pena di morte), just needs a title whose stems don't overlap with
    # "elemento soggettivo del reato", to test a number hint being rejected
    # against an article that genuinely exists but doesn't match.
    "12": {"title": "Successione di leggi penali nel tempo",
           "text": "Qualora la legge del tempo in cui fu commesso il fatto e le leggi posteriori siano "
                   "diverse, si applica quella le cui disposizioni sono più favorevoli al colpevole"},
}


class _FakeCodicePenaleSession:
    """Simulates enough of Neo4j to run _resolve_institution's real calls
    into issue_spotting.article_by_title / fetch_article (not mocked — the
    point is to prove the actual resolution logic, not a stand-in for it)
    and this file's own _issue_article_refs, against the three articles
    above, dispatching on the distinguishing Cypher text each one sends."""

    def __init__(self, articles=_CP_ARTICLES):
        self.articles = articles

    def run(self, query, **params):
        q = " ".join(query.split())
        if "ALL(w IN $stems" in q:                           # article_by_title
            stems = set(params["stems"])
            rows = [{"name": f"{num}.0.0", "title": a["title"]}
                    for num, a in self.articles.items()
                    if all(w in a["title"].lower() for w in stems)]
            return _Result(rows)
        if "s.name IN [$zero_zero, $num]" in q:               # fetch_article, exact
            art = self.articles.get(params["num"])
            if not art:
                return _Result([])
            num = params["num"]
            return _Result([{"d": {"id": "cp-doc", "name": "Codice Penale 2026"},
                             "s": {"name": f"{num}.0.0", "title": art["title"], "plain_text": art["text"]}}])
        if "s.name STARTS WITH $num" in q:                    # fetch_article, comma-level fallback
            return _Result([])
        if "d.name STARTS WITH $prefix" in q:                 # _fetch_articles (standard-code fast path)
            art = self.articles.get(params["n"])
            if not art:
                return _Result([])
            num = params["n"]
            return _Result([{"d": {"id": "cp-doc", "name": "Codice Penale 2026"},
                             "s": {"name": f"{num}.0.0", "title": art["title"], "plain_text": art["text"]}}])
        if "db.index.fulltext.queryNodes" in q:                # _issue_article_refs
            # len > 3: a real fulltext index's analyzer drops short, common
            # words ("di", "la", "chi") rather than treat them as hits.
            terms = {w.rstrip("*").lower() for w in params["t"].split() if len(w.rstrip("*")) > 3}
            rows = []
            for num, a in self.articles.items():
                words = set(re.findall(r"\w+", a["text"].lower()))
                if terms & words:
                    rows.append({"doc": "Codice Penale 2026", "sec": f"{num}.0.0", "score": 1.0})
            return _Result(rows)
        if "s.name = $n + '.0.0'" in q:                        # _number_matches_institution
            art = self.articles.get(params["n"])
            return _Result([{"title": art["title"]}] if art else [])
        raise AssertionError(f"unexpected query: {q}")


def test_589_found_by_title_match_omicidio_colposo():
    """The institution's name IS the rubrica, verbatim — the most common
    case, and the one the model should usually manage even without a
    reliable number."""
    rows = dg._resolve_institution(_FakeCodicePenaleSession(), [_CP_DOC],
                                   {"nome": "omicidio colposo", "codice": "c.p.", "numero_indicativo": ""})
    assert [r["_article"] for r in rows] == [("589", "c.p.")]


def test_586_found_by_title_match_full_phrase():
    rows = dg._resolve_institution(
        _FakeCodicePenaleSession(), [_CP_DOC],
        {"nome": "morte o lesione come conseguenza di altro delitto", "codice": "c.p.", "numero_indicativo": ""})
    assert [r["_article"] for r in rows] == [("586", "c.p.")]


def test_43_falls_back_to_fulltext_when_the_name_does_not_match_the_title():
    """"elemento soggettivo" does not title-match "Elemento psicologico del
    reato" (different word: soggettivo vs psicologico) — this is exactly the
    case the model got wrong on prod (proposed art. 12). The issue's own
    parole_chiave (dolo, colpa) are what the operative text actually says,
    and is what finds it."""
    istituto = {"nome": "elemento soggettivo del reato", "codice": "c.p.", "numero_indicativo": "12"}
    session = _FakeCodicePenaleSession()

    # Without parole_chiave, nome alone doesn't appear in the text either —
    # confirms the fallback genuinely needs the search terms, not a free pass.
    assert dg._resolve_institution(session, [_CP_DOC], istituto) == []

    rows = dg._resolve_institution(session, [_CP_DOC], istituto, parole_chiave=["negligenza", "imprudenza"])
    assert [r["_article"] for r in rows] == [("43", "c.p.")]


def test_a_wrong_number_hint_is_rejected_without_textual_support():
    """The model's number (12, wrong) is never reached as a fallback here
    because full-text already finds 43 — but even if it were the only
    option, _number_matches_institution would reject 12 for this name: a
    wrong guess with no textual support must not be fetched."""
    session = _FakeCodicePenaleSession()
    assert dg._number_matches_institution(session, "cp-doc", "12", "elemento soggettivo del reato") is False
    assert dg._number_matches_institution(session, "cp-doc", "43", "elemento psicologico del reato") is True


def test_number_hint_accepted_when_nothing_else_resolves_and_it_matches():
    """A single-word institution name ("omicidio") fails article_by_title's
    own tolerance rule (it allows zero extra title words for a one-word
    name, and "Omicidio colposo" has one: "colposo") even though "omicidio"
    is genuinely part of that title, and the word itself never appears in
    589's operative text (only in its title) so full-text misses it too.
    _number_matches_institution has no such tolerance cap — it only checks
    containment — so the model's number is accepted here precisely because
    the article_by_title route was too strict for this generic a name."""
    istituto = {"nome": "omicidio", "codice": "c.p.", "numero_indicativo": "589"}
    session = _FakeCodicePenaleSession()
    assert issue_spotting.article_by_title(session, "cp-doc", "omicidio") is None   # too generic for article_by_title
    rows = dg._resolve_institution(session, [_CP_DOC], istituto)
    assert [r["_article"] for r in rows] == [("589", "c.p.")]


def test_number_hint_discarded_when_it_does_not_match_the_institution():
    istituto = {"nome": "concorso anomalo", "codice": "c.p.", "numero_indicativo": "43"}
    rows = dg._resolve_institution(_FakeCodicePenaleSession(), [_CP_DOC], istituto)
    assert rows == []


def test_no_name_returns_nothing():
    assert dg._resolve_institution(_FakeCodicePenaleSession(), [_CP_DOC],
                                   {"nome": "", "codice": "c.p.", "numero_indicativo": "589"}) == []


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
