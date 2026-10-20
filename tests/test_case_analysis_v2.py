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


# --- ISSUE_CHECKLISTS_ENABLED (round 10, 19 Oct) ----------------------------

def test_issue_checklists_are_appended_to_the_prompt_only_when_the_flag_is_on(monkeypatch):
    seen = {}

    def fake_call_chat(messages, **kwargs):
        seen["system"] = messages[0].content
        return "{}"
    monkeypatch.setattr(dg, "_call_chat", fake_call_chat)

    monkeypatch.delenv("ISSUE_CHECKLISTS_ENABLED", raising=False)
    dg._analyse_case_v2("fatti", "it")
    assert dg._ISSUE_CHECKLISTS_BLOCK not in seen["system"]

    monkeypatch.setenv("ISSUE_CHECKLISTS_ENABLED", "true")
    dg._analyse_case_v2("fatti", "it")
    assert dg._ISSUE_CHECKLISTS_BLOCK in seen["system"]


def test_issue_checklists_cover_the_five_areas_as_methods_not_article_numbers():
    """These must be methods to check for ("nesso causale", "accettazione
    tacita"...), never article numbers or a code sigle -- an institution
    the model invents just to satisfy a checklist item would be worse than
    missing it."""
    for area in ("penale", "obbligazioni e contratti", "diritti reali e possesso",
                "successioni", "diritto amministrativo"):
        assert f"- {area}:" in dg._ISSUE_CHECKLISTS_BLOCK
    assert not re.search(r"\bart(?:icol[oi])?\.?\s*\d", dg._ISSUE_CHECKLISTS_BLOCK, re.IGNORECASE)


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
    "584": {"title": "Omicidio preterintenzionale",
            "text": "Chiunque, con atti diretti a commettere uno dei delitti preveduti dagli articoli 581 e "
                    "582, cagiona la morte di un uomo, è punito con la reclusione non inferiore a dieci anni"},
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
    "581": {"title": "Percosse",
            "text": "Chiunque percuote alcuno, se dal fatto non deriva una malattia nel corpo o nella mente, "
                    "è punito, a querela della persona offesa, con la reclusione fino a sei mesi o con la "
                    "multa fino a euro 309"},
    "582": {"title": "Lesione personale",
            "text": "Chiunque cagiona ad alcuno una lesione personale, dalla quale deriva una malattia nel "
                    "corpo o nella mente, è punito con la reclusione da tre mesi a tre anni"},
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
        if "db.index.fulltext.queryNodes" in q:                # _issue_article_refs (scoped to doc_id)
            # len > 3: a real fulltext index's analyzer drops short, common
            # words ("di", "la", "chi") rather than treat them as hits.
            terms = {w.rstrip("*").lower() for w in params["t"].split() if len(w.rstrip("*")) > 3}
            rows = []
            for num, a in self.articles.items():
                words = set(re.findall(r"\w+", a["text"].lower()))
                if terms & words:
                    rows.append({"sec": f"{num}.0.0", "titolo_articolo": "", "title": a["title"],
                                "plain_text": a["text"], "score": 1.0})
            return _Result(rows)
        if "s.name = $n + '.0.0'" in q:                        # _number_matches_institution
            art = self.articles.get(params["n"])
            return _Result([{"title": art["title"], "plain_text": art["text"]}] if art else [])
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


def test_title_match_tags_heading_match_but_fulltext_and_number_hint_do_not():
    """Round 9, 19 Oct: the applicability check must not be able to drop a
    core qualification the article's own rubrica resolved -- only rows
    found this way (not the weaker fulltext fallback or number-hint paths)
    carry _heading_match, the tag _applicability_candidates exempts on."""
    title_rows = dg._resolve_institution(
        _FakeCodicePenaleSession(), [_CP_DOC],
        {"nome": "omicidio colposo", "codice": "c.p.", "numero_indicativo": ""})
    assert all(r.get("_heading_match") is True for r in title_rows)

    fulltext_rows = dg._resolve_institution(
        _FakeCodicePenaleSession(), [_CP_DOC],
        {"nome": "elemento soggettivo del reato", "codice": "c.p.", "numero_indicativo": ""},
        parole_chiave=["negligenza", "imprudenza"])
    assert fulltext_rows and all(not r.get("_heading_match") for r in fulltext_rows)

    number_hint_rows = dg._resolve_institution(
        _FakeCodicePenaleSession(), [_CP_DOC],
        {"nome": "omicidio", "codice": "c.p.", "numero_indicativo": "589"})
    assert number_hint_rows and all(not r.get("_heading_match") for r in number_hint_rows)


# --- HEADING_INDEX_ENABLED (round 10, 19 Oct): semantic match on the ------
# article's own rubrica, for an institution article_by_title's word/stem
# matching structurally rejects (a real rubrica, wrong only because of a
# word-count rule, not because it names the wrong article).

class _FakeHeadingIndexSession(_FakeCodicePenaleSession):
    """Extends _FakeCodicePenaleSession with a canned db.index.vector.
    queryNodes reply -- the embedding itself is mocked in each test, so
    this fake session doesn't need to interpret it, only return what a
    real ANN query against the heading index would have ranked first."""

    def __init__(self, articles=_CP_ARTICLES, vector_rows=()):
        super().__init__(articles)
        self.vector_rows = list(vector_rows)

    def run(self, query, **params):
        q = " ".join(query.split())
        if "db.index.vector.queryNodes" in q:
            return _Result(self.vector_rows)
        return super().run(query, **params)


@pytest.fixture(autouse=False)
def fake_embedding(monkeypatch):
    monkeypatch.setattr(dg, "embedding_model", type("M", (), {"embed_query": lambda self, text: [0.0]})())


def test_heading_index_match_accepts_a_candidate_with_enough_stem_overlap(fake_embedding):
    session = _FakeHeadingIndexSession(vector_rows=[
        {"name": "56.0.0", "titolo_articolo": "Delitto tentato", "score": 0.9},
    ])
    # "tentativo" (stem "tenta") overlaps the single content stem of
    # "Delitto tentato" ("tenta") -- needed = 1, found = 1.
    assert dg._heading_index_match(session, "cp-doc", "tentativo") == "56"


def test_heading_index_match_rejects_a_candidate_with_no_shared_words(fake_embedding):
    session = _FakeHeadingIndexSession(vector_rows=[
        {"name": "999.0.0", "titolo_articolo": "Tutt'altra materia", "score": 0.95},
    ])
    assert dg._heading_index_match(session, "cp-doc", "tentativo") is None


def test_heading_index_match_returns_none_when_the_index_does_not_exist(monkeypatch, fake_embedding):
    class _BrokenSession:
        def run(self, query, **params):
            raise Exception("no such index 'section_heading_embeddings'")
    assert dg._heading_index_match(_BrokenSession(), "cp-doc", "tentativo") is None


def test_heading_index_match_returns_none_without_usable_stems(fake_embedding):
    session = _FakeHeadingIndexSession(vector_rows=[{"name": "1.0.0", "titolo_articolo": "x", "score": 1.0}])
    assert dg._heading_index_match(session, "cp-doc", "e la") is None   # only stop-words


def test_resolve_institution_falls_back_to_the_heading_index_when_the_flag_is_on(monkeypatch, fake_embedding):
    """"tentativo" alone does not title-match "Delitto tentato" (article_by_
    title's own tolerance rule: one extra title word over a single-stem
    name) -- the heading index, enabled, resolves it anyway and still tags
    _heading_match."""
    articles = {**_CP_ARTICLES, "56": {"title": "Delitto tentato",
               "text": "Chi compie atti idonei diretti in modo non equivoco a commettere un delitto..."}}
    session = _FakeHeadingIndexSession(articles, vector_rows=[
        {"name": "56.0.0", "titolo_articolo": "Delitto tentato", "score": 0.9},
    ])
    monkeypatch.setenv("HEADING_INDEX_ENABLED", "true")

    rows = dg._resolve_institution(session, [_CP_DOC],
                                   {"nome": "tentativo", "codice": "c.p.", "numero_indicativo": ""})
    assert [r["_article"] for r in rows] == [("56", "c.p.")]
    assert all(r.get("_heading_match") is True for r in rows)


def test_resolve_institution_does_not_use_the_heading_index_when_the_flag_is_off(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the heading index must not be queried when the flag is off")
    monkeypatch.setattr(dg, "_heading_index_match", boom)
    monkeypatch.delenv("HEADING_INDEX_ENABLED", raising=False)

    rows = dg._resolve_institution(_FakeCodicePenaleSession(), [_CP_DOC],
                                   {"nome": "tentativo", "codice": "c.p.", "numero_indicativo": ""})
    assert rows == []   # falls through every other path too: "tentativo" matches nothing in _CP_ARTICLES


def test_resolve_institution_tries_the_heading_index_once_for_nome_not_per_parola_chiave(monkeypatch):
    """Bounded added latency: one embedding call per unresolved institution,
    not up to 1+len(parole_chiave) of them."""
    calls = []
    monkeypatch.setattr(dg, "_heading_index_match", lambda session, doc_id, term: calls.append(term) or None)
    monkeypatch.setenv("HEADING_INDEX_ENABLED", "true")

    dg._resolve_institution(_FakeCodicePenaleSession(), [_CP_DOC],
                            {"nome": "tentativo", "codice": "c.p.", "numero_indicativo": ""},
                            parole_chiave=["atti idonei", "univocità", "desistenza"])
    assert calls == ["tentativo"]


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


def test_institution_title_match_tries_parole_chiave_too_not_only_nome():
    """Round 3, 18 Oct (c1 diagnostic): "nesso causale"/"causa effettiva"
    does not title-match a rubrica worded "Rapporto di causalità" or
    "Concorso di cause" even though it is the same institution -- but one
    of the issue's own parole_chiave, the model's other words for it, can.
    Title-match is tried for nome first, then for each parola_chiave in
    turn, before ever falling back to full-text."""
    articles = {**_CP_ARTICLES,
               "40": {"title": "Rapporto di causalità",
                      "text": "Nessuno può essere punito se l'evento non è conseguenza della sua condotta."},
               "41": {"title": "Concorso di cause",
                      "text": "Più condotte indipendenti possono concorrere nel determinare lo stesso evento."}}
    session = _FakeCodicePenaleSession(articles)
    istituto = {"nome": "nesso causale", "codice": "c.p.", "numero_indicativo": ""}

    # nome alone matches neither title nor (being generic) the full text.
    assert dg._resolve_institution(session, [_CP_DOC], istituto) == []

    rows = dg._resolve_institution(session, [_CP_DOC], istituto,
                                   parole_chiave=["rapporto di causalità", "concorso di cause"])
    assert [r["_article"] for r in rows] == [("40", "c.p.")]


def test_a_wrong_number_hint_is_rejected_without_textual_support():
    """The model's number (12, wrong) is never reached as a fallback here
    because full-text already finds 43 — but even if it were the only
    option, _number_matches_institution would reject 12 for this name: a
    wrong guess with no textual support must not be fetched."""
    session = _FakeCodicePenaleSession()
    assert dg._number_matches_institution(session, "cp-doc", "12", "elemento soggettivo del reato") is False
    assert dg._number_matches_institution(session, "cp-doc", "43", "elemento psicologico del reato") is True


def test_number_hint_without_a_title_checks_the_articles_own_text():
    """Round 3, 18 Oct (c3 diagnostic): art. 1950 c.c.'s matching Section had
    no title, and the model's wrong numero_indicativo for a completely
    unrelated institution ("diritto di revoca del contratto di locazione")
    was accepted purely because there was nothing to check it against --
    _number_matches_institution used to fail open whenever the title was
    missing. Falling back to the article's own plain_text when the title
    is empty closes that gap."""
    articles = {**_CP_ARTICLES, "1950": {
        "title": "", "text": "Il fideiussore che ha pagato il debito ha regresso verso il debitore "
                             "principale, con gli interessi e le spese."}}
    session = _FakeCodicePenaleSession(articles)
    assert dg._number_matches_institution(
        session, "cp-doc", "1950", "diritto di revoca del contratto di locazione") is False
    assert dg._number_matches_institution(session, "cp-doc", "1950", "regresso del fideiussore") is True


def test_number_hint_for_an_unknown_article_fails_open():
    """Nothing to check at all (fetch_article will also find nothing, so
    there is nothing to wrongly trust either)."""
    session = _FakeCodicePenaleSession()
    assert dg._number_matches_institution(session, "cp-doc", "9999", "qualunque cosa") is True


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


# --- _issue_article_refs: scoped to the institution's own document, verified
# against its own words (round 8, 19 Oct) -----------------------------------
#
# This used to search across every primary act with no scope at all -- on
# prod, when an institution's "codice" didn't resolve to one document (a
# generic doctrinal name like "causa effettiva"), that unscoped search
# returned matches from unrelated codes/laws entirely (23 c.c., 46 L. sul
# diritto internazionale privato for a criminal-law institution), trusted
# just because a Lucene OR-of-words query ranked them. It is now always
# scoped to one document (never searched without one) and every hit is
# re-verified against the institution's own words.

def test_issue_article_refs_returns_nothing_without_a_doc_id():
    """No document resolved -> no corpus-wide search any more: unresolved
    is better than wrong."""
    class Session:
        def run(self, query, **params):
            raise AssertionError("must not query without a doc_id")

    assert dg._issue_article_refs(Session(), None, ["qualcosa"]) == []


def test_issue_article_refs_scoped_and_verified_finds_the_real_article():
    numbers = dg._issue_article_refs(_FakeCodicePenaleSession(), "cp-doc", ["negligenza", "imprudenza"])
    assert numbers == ["43"]


def test_issue_article_refs_rejects_a_hit_sharing_none_of_the_institutions_words():
    """A fulltext hit is dropped unless at least half the institution's own
    stems actually appear in its title or text -- a bare Lucene score is
    not proof of relevance, only that the query's words were ORed somewhere."""
    class Session:
        def run(self, query, **params):
            return _Result([{"sec": "999.0.0", "titolo_articolo": "", "title": "Tutt'altra materia",
                             "plain_text": "un testo che non condivide nessuna delle parole cercate"}])

    assert dg._issue_article_refs(Session(), "doc", ["negligenza", "imprudenza"]) == []


def test_issue_article_refs_empty_keywords_skips_the_query():
    class Session:
        def run(self, query, **params):
            raise AssertionError("should not query with no keywords")

    assert dg._issue_article_refs(Session(), "doc", []) == []


def test_issue_article_refs_fails_open_on_a_db_error():
    class Session:
        def run(self, query, **params):
            raise RuntimeError("db down")

    assert dg._issue_article_refs(Session(), "doc", ["qualcosa"]) == []


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


# --- End-to-end: Sempronio and sempronio_no_hint (mocked LLM output) -------
#
# Demonstrates the full chain for both eval_narrative_cases.json entries:
# _analyse_case_v2 (LLM call mocked with a plausible institutions-in-words
# reply — never a trusted number) -> _resolve_institution per issue -> the
# five articles step 1 is meant to reach: 43, 581, 582, 586, 589. Both cases
# get the SAME mocked reply, on purpose: the whole point of naming
# institutions instead of numbers is that it does not depend on whether the
# question happens to cite an article number — "sempronio_no_hint" has none
# at all, yet the institutions are identical, because the facts are.

_SEMPRONIO_ANALYSIS_REPLY = {
    "parte_assistita": "Sempronio", "posizione": "indagato", "area": "penale",
    "richiesta": "strategia difensiva", "ricerca_fatti": ["schiaffo", "caduta", "marciapiede", "soccorso"],
    "fatti_strutturati": {
        "parti": "Sempronio (autore), Caio (vittima)", "condotta": "un violento schiaffo al volto",
        "evento": "morte di Caio per emorragia cerebrale da trauma cranico",
        "nesso_causale": "schiaffo -> caduta -> trauma cranico -> emorragia cerebrale -> morte",
        "condotta_successiva": "Sempronio presta soccorso e chiama l'ambulanza",
    },
    "questioni": [
        {"titolo": "Omicidio preterintenzionale", "categoria": "qualificazione",
         "istituti": [{"nome": "omicidio preterintenzionale", "codice": "c.p.", "numero_indicativo": ""}],
         "parole_chiave": ["preterintenzione"]},
        {"titolo": "Percosse e lesione personale", "categoria": "qualificazione_alternativa",
         "istituti": [{"nome": "percosse", "codice": "c.p.", "numero_indicativo": ""},
                      {"nome": "lesione personale", "codice": "c.p.", "numero_indicativo": ""}],
         "parole_chiave": []},
        {"titolo": "Morte come conseguenza di altro delitto", "categoria": "qualificazione_alternativa",
         "istituti": [{"nome": "morte o lesione come conseguenza di altro delitto",
                      "codice": "c.p.", "numero_indicativo": ""}],
         "parole_chiave": []},
        {"titolo": "Omicidio colposo", "categoria": "qualificazione_alternativa",
         "istituti": [{"nome": "omicidio colposo", "codice": "c.p.", "numero_indicativo": ""}],
         "parole_chiave": []},
        {"titolo": "Elemento soggettivo", "categoria": "elemento_soggettivo",
         # "elemento soggettivo" does not title-match "Elemento psicologico del
         # reato" — this is deliberately the same wording that got art. 12 c.p.
         # proposed on prod; the wrong number is included here too, to prove
         # it's never the reason 43 is found.
         "istituti": [{"nome": "elemento soggettivo del reato", "codice": "c.p.", "numero_indicativo": "12"}],
         "parole_chiave": ["negligenza", "imprudenza"]},
    ],
    "fatti": [],
}


def _articles_found(analysis):
    """{article_number: match_reason} for every institution across every
    questione, resolved the same way _select_articles_v2 does."""
    session = _FakeCodicePenaleSession()
    found = {}
    for q in analysis["questioni"]:
        for istituto in q["istituti"]:
            rows = dg._resolve_institution(session, [_CP_DOC], istituto, q["parole_chiave"])
            for row in rows:
                found[row["_article"][0]] = istituto["nome"]
    return found


def test_sempronio_with_the_pm_hint_reaches_all_five_articles(monkeypatch):
    monkeypatch.setattr(dg, "_call_chat", lambda *a, **k: json.dumps(_SEMPRONIO_ANALYSIS_REPLY))
    analysis = dg._analyse_case_v2(
        "Sempronio ... Il PM contesta a Sempronio l'art 584 cp. ...", "it")
    found = _articles_found(analysis)
    assert {"43", "581", "582", "586", "589"} <= set(found)


def test_sempronio_no_hint_reaches_the_same_five_articles(monkeypatch):
    """No article number anywhere in the question — same mocked analysis
    reply, since the facts (and therefore the institutions) are the same;
    this is the property the redesign exists for."""
    monkeypatch.setattr(dg, "_call_chat", lambda *a, **k: json.dumps(_SEMPRONIO_ANALYSIS_REPLY))
    analysis = dg._analyse_case_v2(
        "Sempronio a seguito di un acceso diverbio con Caio gli sferra un violento schiaffo al volto ... "
        "nessun articolo citato nel testo", "it")
    found = _articles_found(analysis)
    assert {"43", "581", "582", "586", "589"} <= set(found)
    # Specifically: 43 via the parole_chiave fallback, not the wrong number 12.
    assert found["43"] == "elemento soggettivo del reato"


# --- _select_articles_v2: additive union, not a replacement ----------------
#
# The regression found on prod (17 Oct): v2 used to run its own from-scratch
# selection instead of _select_articles (v1), losing whatever v1's other
# paths (ruling agreement, crossref, the model's own candidate choice) had
# found — net avg 1.36 vs. 1.64 flag-off, despite genuine gains elsewhere.
# _select_articles is mocked here (not re-exercised — it has its own tests)
# specifically to isolate and prove the UNION property.

def test_select_articles_v2_adds_to_v1_instead_of_replacing_it(monkeypatch):
    v1_row = {"d": {"id": "cp-doc", "name": "Codice Penale 2026"},
             "s": {"name": "581.0.0", "plain_text": "testo 581 da v1 (ruling agreement)"},
             "_article": ("581", "c.p.")}
    base_result = {"first_rulings": [], "similar": [], "keep": [("581", "c.p.")],
                   "named": [], "candidates": [("581", "c.p.")], "chosen": [("581", "c.p.")],
                   "selected": [("581", "c.p.")], "article_rows": [v1_row], "spotted": []}
    monkeypatch.setattr(dg, "_select_articles", lambda *a, **k: base_result)

    import src.rag.doc_lookup as doc_lookup
    monkeypatch.setattr(doc_lookup, "_fetch_doc_names", lambda driver, database: [_CP_DOC])
    import src.rag.nodes.retrieval as retrieval_module
    monkeypatch.setattr(retrieval_module, "broaden_article_context",
                        lambda state, driver, database: {"raw_result": state["raw_result"]})

    analysis = {"questioni": [{"titolo": "Omicidio colposo", "istituti": [
        {"nome": "omicidio colposo", "codice": "c.p.", "numero_indicativo": ""}], "parole_chiave": []}]}
    picked = dg._select_articles_v2(_FakeCodicePenaleSession(), object(), "db", "facts", "", analysis)

    numbers = {r["_article"][0] for r in picked["article_rows"]}
    assert "581" in numbers    # v1's own row — preserved, not dropped
    assert "589" in numbers    # institution-resolved — added
    assert picked["institutions_resolved"] == [("589", "c.p.")]
    assert ("581", "c.p.") in picked["selected"] and ("589", "c.p.") in picked["selected"]


def test_select_articles_v2_does_not_duplicate_an_article_v1_already_has(monkeypatch):
    """An institution that resolves to the same article v1 already selected
    must not be fetched twice."""
    v1_row = {"d": {"id": "cp-doc", "name": "Codice Penale 2026"},
             "s": {"name": "589.0.0", "plain_text": "testo 589 da v1"},
             "_article": ("589", "c.p.")}
    base_result = {"first_rulings": [], "similar": [], "keep": [], "named": [],
                   "candidates": [], "chosen": [], "selected": [("589", "c.p.")],
                   "article_rows": [v1_row], "spotted": []}
    monkeypatch.setattr(dg, "_select_articles", lambda *a, **k: base_result)
    import src.rag.doc_lookup as doc_lookup
    monkeypatch.setattr(doc_lookup, "_fetch_doc_names", lambda driver, database: [_CP_DOC])

    analysis = {"questioni": [{"titolo": "Omicidio colposo", "istituti": [
        {"nome": "omicidio colposo", "codice": "c.p.", "numero_indicativo": ""}], "parole_chiave": []}]}
    picked = dg._select_articles_v2(_FakeCodicePenaleSession(), object(), "db", "facts", "", analysis)

    assert len(picked["article_rows"]) == 1   # not two copies of 589
    assert picked["institutions_resolved"] == []   # nothing new was added


def test_select_articles_v2_broadens_only_named_or_added_not_every_v1_candidate(monkeypatch):
    """Round 3, item 3 (18 Oct): broadening the full merged set re-expanded
    articles v1 had already selected and broadened once itself (116, 118 on
    Sempronio), pulling in generic cross-references a second time (1, 2, 11
    c.p.) with no connection to anything v2 added. The seed for
    broaden_article_context must be only the articles the user's own text
    names, plus whatever v2 itself resolved via an institution -- not v1's
    whole candidate pool."""
    v1_row = {"d": {"id": "cp-doc", "name": "Codice Penale 2026"},
             "s": {"name": "116.0.0", "plain_text": "v1's own candidate, not named in the case"},
             "_article": ("116", "c.p.")}
    base_result = {"first_rulings": [], "similar": [], "keep": [("116", "c.p.")],
                   "named": [], "candidates": [("116", "c.p.")], "chosen": [("116", "c.p.")],
                   "selected": [("116", "c.p.")], "article_rows": [v1_row], "spotted": []}
    monkeypatch.setattr(dg, "_select_articles", lambda *a, **k: base_result)
    import src.rag.doc_lookup as doc_lookup
    monkeypatch.setattr(doc_lookup, "_fetch_doc_names", lambda driver, database: [_CP_DOC])

    seeds_seen = []

    def fake_broaden(state, driver, database):
        seeds_seen.append({r.get("_article") for r in state["raw_result"]})
        return {"raw_result": state["raw_result"]}

    import src.rag.nodes.retrieval as retrieval_module
    monkeypatch.setattr(retrieval_module, "broaden_article_context", fake_broaden)

    analysis = {"questioni": [{"titolo": "Omicidio colposo", "istituti": [
        {"nome": "omicidio colposo", "codice": "c.p.", "numero_indicativo": ""}], "parole_chiave": []}]}
    picked = dg._select_articles_v2(_FakeCodicePenaleSession(), object(), "db", "facts", "", analysis)

    assert len(seeds_seen) == 1
    seeded = seeds_seen[0]
    assert ("116", "c.p.") not in seeded   # v1's own candidate, not named -- not re-broadened
    assert ("589", "c.p.") in seeded       # what v2 just added via the institution -- is
    # v1's own row is still in the final result, broaden or not.
    assert ("116", "c.p.") in {r["_article"] for r in picked["article_rows"]}


# --- _fetch_articles: one row per duplicate Section, not Neo4j's order -----

def test_section_quality_prefers_titolo_articolo_over_longer_text():
    """A short duplicate with the parser's own rubrica (BE issue 7.1) beats
    a longer duplicate without one -- length is only the final tie-break,
    not the first one. Round 3, 18 Oct: preferring length alone picked a
    stale duplicate for article 628 and lost it from the draft (3/3 -> 0/3,
    same code, same facts, flag-off)."""
    with_titolo = {"titolo_articolo": "Rapina", "plain_text": "breve"}
    longer_without = {"titolo_articolo": "", "plain_text": "molto più lungo " * 20}
    assert dg._section_quality(with_titolo) > dg._section_quality(longer_without)


def test_section_quality_falls_back_to_length_when_neither_has_titolo_articolo():
    shorter = {"titolo_articolo": "", "plain_text": "corto"}
    longer = {"titolo_articolo": "", "plain_text": "molto più lungo di questo"}
    assert dg._section_quality(longer) > dg._section_quality(shorter)


def test_dedupe_sections_keeps_the_best_duplicate_in_first_seen_order():
    rows = [
        {"d": {"id": "cp-doc"}, "s": {"name": "628.0.0", "titolo_articolo": "", "plain_text": "stale, lungo " * 10}},
        {"d": {"id": "cp-doc"}, "s": {"name": "628.0.0", "titolo_articolo": "Rapina", "plain_text": "attuale, corto"}},
        {"d": {"id": "cp-doc"}, "s": {"name": "629.0.0", "titolo_articolo": "", "plain_text": "altro articolo"}},
    ]
    deduped = dg._dedupe_sections(rows)
    assert [r["s"]["name"] for r in deduped] == ["628.0.0", "629.0.0"]
    assert deduped[0]["s"]["titolo_articolo"] == "Rapina"


# --- _prioritise_article_rows ------------------------------------------------

def test_article_priority_orders_named_crossref_institution_other():
    named = {("584", "c.p.")}

    def row(article, source=None):
        r = _row(article, "testo")
        if source:
            r["_source"] = source
        return r

    from src.rag.nodes.cypher import CROSSREF_SOURCE
    rows = [row(("584", "c.p.")), row(("581", "c.p."), CROSSREF_SOURCE),
            row(("589", "c.p."), dg.CASE_ANALYSIS_SOURCE), row(("116", "c.p."))]
    assert [dg._article_priority(r, named) for r in rows] == [0, 1, 2, 3]


def test_prioritise_article_rows_keeps_everything_under_the_limit():
    rows = [_row((str(n), "c.p."), "testo") for n in range(5)]
    assert dg._prioritise_article_rows(rows, set(), limit=10) == rows


def test_prioritise_article_rows_trims_lowest_priority_first():
    named = {("584", "c.p.")}
    named_row = _row(("584", "c.p."), "testo")
    other_rows = [_row((str(n), "c.p."), "testo") for n in range(5)]
    result = dg._prioritise_article_rows([*other_rows, named_row], named, limit=3)
    assert named_row in result   # never trimmed
    assert len(result) == 3


def test_a_match_on_the_llm_written_title_is_not_exempt_from_the_check(monkeypatch):
    """19 Oct, later: only titolo_articolo (the official heading) tags
    _heading_match. s.title is LLM-written on documents not yet re-ingested
    (BE, 16 Oct: "Estorsione" on art. 612), so a match on it is checked."""
    import src.rag.issue_spotting as isp
    monkeypatch.setattr(isp, "article_by_title",
                        lambda session, doc_id, term, visibility="", *, field="title", **kw:
                        "589" if field == "title" else None)
    monkeypatch.setattr(isp, "fetch_article",
                        lambda session, doc_id, number, visibility="", **kw:
                        [{"d": {"id": doc_id}, "s": {"name": f"{number}.0.0", "plain_text": "Chiunque cagiona..."}}])
    rows = dg._resolve_institution(None, [_CP_DOC], {"nome": "omicidio colposo", "codice": "c.p.",
                                                      "numero_indicativo": ""})
    assert [r["_article"] for r in rows] == [("589", "c.p.")]
    assert not any(r.get("_heading_match") for r in rows)
