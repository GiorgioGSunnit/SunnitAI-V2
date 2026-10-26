"""ANSWER_INVARIANTS_ENABLED (src/rag/answer_invariants.py): I1-I7 on the written answer, and I3 at resolution.
Invented cases only: never the boat/pool texts, h1-h11, n1-n9, nor the live narratives."""

import json
import os

import pytest

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

import src.rag.answer_invariants as ai  # noqa: E402
import src.rag.defensive_generation as dg  # noqa: E402
import src.rag.fact_signals as fs  # noqa: E402

DOCS = [{"id": "cc", "name": "Codice Civile 2026"}, {"id": "cpc", "name": "Codice di procedura civile 2026"}]


@pytest.fixture
def logged(monkeypatch):
    out = []
    monkeypatch.setattr(dg, "vlog", lambda event, data=None, *a, **k: out.append((event, data)))
    return out


def _row(number, heading, text="", grade=None, code="c.c."):
    row = {"d": {"id": "cc", "name": "Codice Civile 2026"}, "_article": (number, code),
           "s": {"name": f"{number}.0.0", "titolo_articolo": heading, "plain_text": text}}
    if grade is not None:
        row["_grade"] = grade
    return row


class _Result:
    def __init__(self, rows):
        self.rows = rows

    def data(self):
        return self.rows


class ChapterSession:
    """The articles of one chapter, as pick_in_chapter's query returns them."""

    def __init__(self, rows):
        self.rows, self.queries = rows, []

    def run(self, query, **params):
        self.queries.append(params)
        return _Result([{"num": n, "rubrica": r} for n, r in self.rows
                        if ai.in_ranges(n, params.get("ranges") or [[0, 99999]])])


# --- I3: an institution within its chapter --------------------------------------------------------------------------

@pytest.mark.parametrize("name,code,ranges", [
    ("usucapione di servitù apparente", "c.c.", [[1158, 1167], [1061, 1061]]),
    ("passaggio coattivo", "c.c.", [[1051, 1055]]),
    ("indennità per il passaggio coattivo", "c.c.", [[1051, 1055]]),
    ("servitù per destinazione del padre di famiglia", "c.c.", [[1027, 1099]]),
    ("estinzione della servitù per prescrizione", "c.c.", [[1027, 1099]]),
    ("azione negatoria", "c.c.", [[948, 951], [1079, 1079]]),
    ("azione di reintegrazione", "c.c.", [[1140, 1172]]),
    ("responsabilità extracontrattuale", "c.c.", [[2043, 2059]]),
    ("garanzia per i vizi della cosa venduta", "c.c.", [[1490, 1497]]),
    ("garanzia per le difformità e i vizi dell'opera", "c.c.", [[1655, 1677]]),
    ("accettazione dell'eredità", "c.c.", [[456, 768]]),
    ("comunione legale dei beni", "c.c.", [[177, 197]]),
    ("prescrizione del credito", "c.c.", [[2934, 2969]]),
    ("delitto tentato", "c.p.", [[39, 240]]),
    ("circostanze aggravanti del furto", "c.p.", None),
    ("omicidio preterintenzionale", "c.p.", None),
    ("usucapione", "c.p.", None),
    ("risarcimento del danno", "c.c.", None),
])
def test_chapters(name, code, ranges):
    assert ai.chapter(name, code) == ranges


def test_pick_in_chapter_by_rubrica_words_lowest_on_a_tie():
    session = ChapterSession([("1158", "Usucapione dei beni immobili"), ("1159", "Usucapione decennale"),
                              ("1061", "Servitù non apparenti"), ("1060", "Servitù volontaria")])
    assert ai.pick_in_chapter(session, "cc", "usucapione di servitù apparente", [[1158, 1167], [1061, 1061]]) == "1061"
    assert ai.pick_in_chapter(session, "cc", "usucapione del fondo", [[1158, 1167]]) == "1158"
    assert ai.pick_in_chapter(session, "cc", "acquisto del fondo", [[1158, 1167]]) is None
    assert ai.pick_in_chapter(session, "cc", "acquisto del fondo", [[1158, 1167]], ["usucapione decennale"]) == "1159"


def test_resolution_refuses_outside_the_chapter_and_picks_inside(monkeypatch, logged):
    from src.rag import issue_spotting
    headings = {"2689": "Usucapione", "1158": "Usucapione dei beni immobili", "1061": "Servitù non apparenti"}
    monkeypatch.setattr(issue_spotting, "article_by_title",
                        lambda session, doc, term, field="titolo_articolo": "2689")
    monkeypatch.setattr(issue_spotting, "fetch_article", lambda session, doc, n: [
        {"d": {"id": "cc"}, "s": {"name": f"{n}.0.0", "titolo_articolo": headings[n], "plain_text": "x"}}]
        if n in headings else [])
    monkeypatch.setattr(dg, "_issue_article_refs", lambda session, doc, terms: [])
    monkeypatch.setattr(dg, "_number_matches_institution", lambda session, doc, n, name: True)
    session = ChapterSession(list(headings.items()))
    istituto = {"nome": "usucapione di servitù apparente", "codice": "c.c.", "numero_indicativo": "2689"}
    assert [r["_article"] for r in dg._resolve_institution(session, DOCS, istituto)] == [("2689", "c.c.")]
    monkeypatch.setenv("ANSWER_INVARIANTS_ENABLED", "true")
    assert [r["_article"] for r in dg._resolve_institution(session, DOCS, istituto)] == [("1061", "c.c.")]
    picked = [d for e, d in logged if e == "case_analysis_institution_range"][-1]
    assert "2689" in picked["refused"] and picked["picked"] == "1061"
    assert any("(I3)" in d["why"] for e, d in logged if e == "case_analysis_institution_rejected")


def test_a_fact_rule_institution_is_picked_inside_its_ranges(monkeypatch, logged):
    from src.rag import issue_spotting
    monkeypatch.setattr(issue_spotting, "fetch_article", lambda session, doc, n: [
        {"d": {"id": doc}, "s": {"name": f"{n}.0.0", "titolo_articolo": "Passaggio coattivo"}}])
    session = ChapterSession([("1051", "Passaggio coattivo"), ("1053", "Indennità")])
    istituto = {"nome": "passaggio coattivo", "codice": "c.c.", "fonte": "regola fondo_intercluso",
                "ranges": [[1051, 1055]]}
    assert dg._pick_in_rule_ranges(session, DOCS, istituto) == []                          # flag off
    monkeypatch.setenv("ANSWER_INVARIANTS_ENABLED", "true")
    assert [r["_article"] for r in dg._pick_in_rule_ranges(session, DOCS, istituto)] == [("1051", "c.c.")]


# --- I5: internal tokens; the terms in plain Italian ----------------------------------------------------------------

def test_terms_reach_the_writer_in_plain_italian(monkeypatch):
    signals = {"terms": [{"text": "azione possessoria: termine annuale DECORSO (sbarra 2019, avvocato 2021)",
                          "ref": "1168 c.c."}, {"text": "usucapione: termine INCERTO", "ref": ""}]}
    assert "DECORSO" in fs.terms_block(signals, refs=True)
    monkeypatch.setenv("ANSWER_INVARIANTS_ENABLED", "true")
    assert fs.terms_block(signals, refs=True) == ("- azione possessoria: termine annuale già decorso (sbarra 2019, "
                                                  "avvocato 2021) (art. 1168 c.c.)\n- usucapione: termine incerto")


def test_internal_tokens_repaired():
    text = ("**4. Applicazione al caso**\n\nIl termine annuale è DECORSO [DA VERIFICARE]. Vedi snciv_s20_a2025_n01234"
            ".clean.pdf e la regola fondo_intercluso. L'art. 1158.0.0 c.c. e l'art. 2051.2.0 c.c. valgono per la la "
            "carrareccia. Il convenuto [nome del convenuto] va citato. Si veda [art. 1079 c.c.].\n**Termini: entro un "
            "anno.")
    assert ai.internal_tokens(text)
    out, log = ai.repair_tokens(text)
    assert ai.internal_tokens(out) == []
    assert "Il termine annuale è già decorso (da verificare)." in out
    assert "L'art. 1158 c.c. e l'art. 2051, comma 2 c.c. valgono per la carrareccia." in out
    assert "convenuto" not in out and "snciv" not in out and "fondo_intercluso" not in out
    assert "Si veda art. 1079 c.c." in out and "Termini: entro un anno." in out
    assert {e["kind"] for e in log} == {"status", "raw_id", "mangled", "placeholder", "bracket"}


# --- I1: citation closure -------------------------------------------------------------------------------------------

def test_allowed_material():
    rows = [_row("1079", "Accertamento", grade=3), _row("1062", "Destinazione", grade=2), _row("2043", "Illecito", grade=1)]
    signals = {"tools": [{"refs": [["614-bis", "c.p.c."]]}], "terms": [{"ref": "1168 c.c."}]}
    allowed = ai.allowed_refs(rows, [("1051", "c.c.")], signals, [("700", "c.p.c.")])
    assert allowed == [("1079", "c.c."), ("1062", "c.c."), ("1051", "c.c."), ("614-bis", "c.p.c."), ("700", "c.p.c."),
                       ("1168", "c.c.")]
    assert ai.allowed_refs([_row("2043", "Illecito")], [], {}, []) == [("2043", "c.c.")]   # nothing graded: the pool


def test_a_citation_outside_the_material_goes_with_its_sentence_conclusions_too():
    text = ("**4. Applicazione al caso**\n\nOttavia passa ai sensi dell'art. 1062 c.c. Il danno si risarcisce ex art. "
            "2043 c.c.\n\n**6. Conclusioni**\n\nOttavia ha usucapito ai sensi dell'art. 2689 c.c. Ottavia vince.")
    out, log = ai.i1_closure(text, [("1062", "c.c.")], DOCS, {("1062", "c.c."), ("2043", "c.c.")})
    assert out == ("**4. Applicazione al caso**\n\nOttavia passa ai sensi dell'art. 1062 c.c.\n\n**6. Conclusioni**\n\n"
                   "Ottavia vince.")
    assert (log["found"], log["fixed"], log["remaining"]) == (2, 2, 0)


# --- I2: claim <-> article ------------------------------------------------------------------------------------------

def test_a_mismatched_citation_goes_unless_graded_3_or_named():
    rows = {("2051", "c.c."): _row("2051", "Danno cagionato da cosa in custodia", "Ciascuno è responsabile..."),
            ("1079", "c.c."): _row("1079", "Accertamento della servitù", "Il titolare della servitù...", grade=3),
            ("1064", "c.c."): _row("1064", "Estensione del diritto di servitù", "Il diritto di servitù..."),
            ("1062", "c.c."): _row("1062", "Destinazione del padre di famiglia", "La destinazione...")}
    text = ("Bruno risponde della sbarra ai sensi dell'art. 2051 c.c. Ottavia può chiedere i danni per i raccolti persi "
            "ai sensi dell'art. 1079 c.c. L'art. 1064 c.c. obbliga Bruno a pagare. La servitù nasce dall'art. 1062 c.c.")
    asked = []

    def chat_json(messages, max_tokens=None, temperature=None):
        asked.append(messages[-1].content)
        return [{"articolo": "A0", "compatibile": False, "perche": "custodia, non servitù", "tipo": "materia"},
                {"articolo": "A1", "compatibile": False, "perche": "danni", "tipo": "materia"},
                {"articolo": "A2", "compatibile": False, "perche": "x", "tipo": "materia"},
                {"articolo": "A3", "compatibile": False, "perche": "x", "tipo": "formulazione"}]
    known = set(rows)
    out, log = ai.i2_consistency(text, rows, {("1079", "c.c."), ("1064", "c.c.")}, DOCS, known, chat_json)
    assert "[A0] art. 2051 c.c.\nRubrica: Danno cagionato da cosa in custodia" in asked[0]
    assert out == ("Ottavia può chiedere i danni per i raccolti persi. L'art. 1064 c.c. obbliga Bruno a pagare (da "
                   "verificare: l'art. 1064 c.c. riguarda «Estensione del diritto di servitù»). La servitù nasce "
                   "dall'art. 1062 c.c.")
    assert [e["how"] for e in log["entries"]] == ["sentence dropped", "clause dropped", "marked"]
    assert (log["found"], log["fixed"], log["remaining"]) == (3, 2, 1)


@pytest.mark.parametrize("sentence,number,out", [
    ("La sbarra va rimossa (art. 1079 c.c.) a spese di Bruno.", "1079", "La sbarra va rimossa a spese di Bruno."),
    ("Ottavia può far cessare le turbative, in virtù dell'art. 1079 c.c.", "1079",
     "Ottavia può far cessare le turbative."),
    ("L'art. 1079 c.c. consente di agire.", "1079", ""),
    ("Agisce ex art. 1079 c.c.", "1079", ""),                                   # too little left: marked instead
])
def test_clause_without(sentence, number, out):
    assert ai._clause_without(sentence, number) == out


# --- I4: ruling relevance -------------------------------------------------------------------------------------------

def _ruling(rid, number, **extra):
    return {"d": {"id": rid, "name": f"Cass. civ. n. {number}"}, "s": {"plain_text": "p"}, **extra}


def test_rulings_without_a_substantive_article_procedural_or_adverse_in_support_go():
    rows = [_ruling("r1", "1111/2025"), _ruling("r2", "2222/2025"), _ruling("r3", "3333/2025"),
            _ruling("r4", "4444/2025", _fit_adverse=True)]
    cites = {"r1": {("1079", "c.c."), ("360", "c.p.c.")}, "r2": {("360", "c.p.c."), ("1051", "c.c.")},
             "r3": {("345", "c.p.c.")}, "r4": {("1079", "c.c.")}}
    text = ("Ottavia agisce ex art. 1079 c.c., come Cass. civ. n. 1111/2025. Lo conferma Cass. civ. n. 2222/2025. "
            "Secondo Cass. civ. n. 3333/2025 le nuove domande in appello sono vietate dall'art. 345 c.p.c. La servitù "
            "esiste, come dice Cass. civ. n. 4444/2025.\nObiezione della controparte: Cass. civ. n. 4444/2025 nega "
            "il passaggio. Replica: lì il fondo aveva un accesso.")
    known = {("1079", "c.c."), ("345", "c.p.c.")}
    out, log = ai.i4_rulings(text, rows, cites, "Ottavia passa da trent'anni sulla carrareccia.", DOCS, known)
    assert out == ("Ottavia agisce ex art. 1079 c.c., come Cass. civ. n. 1111/2025.\nObiezione della controparte: Cass. "
                   "civ. n. 4444/2025 nega il passaggio. Replica: lì il fondo aveva un accesso.")
    assert [(e["ruling"], e["why"]) for e in log["entries"]] == [
        ("Cass. civ. n. 2222/2025", "no substantive article in common with the answer"),
        ("Cass. civ. n. 3333/2025", "procedural holding only"),
        ("Cass. civ. n. 4444/2025", "adverse, cited outside an objection or a reply")]
    kept, _ = ai.i4_rulings(text, rows[2:3], cites, "Bruno vuole proporre appello.", DOCS, known)
    assert "3333/2025" in kept                                       # a procedural question: the holding serves


# --- I6: readability ------------------------------------------------------------------------------------------------

def test_readability_rewrites_only_when_citations_stay():
    def call_chat(messages, max_tokens=None, temperature=None):
        return json.dumps([{"i": 1, "frase": "Ottavia può agire contro Bruno ai sensi dell'art. 1079 c.c."},
                           {"i": 2, "frase": "La sbarra viola l'art. 1064 c.c."}])
    sections = [("**4. Applicazione al caso**", "Ottavia agire può contro Bruno ai sensi art. 1079 c.c. La sbarra "
                                                "viola l'art. 1065 c.c. per la la servitù.")]
    out, log = ai.i6_readability(sections, call_chat)
    assert out[0][1].startswith("Ottavia può agire contro Bruno ai sensi dell'art. 1079 c.c. La sbarra")
    assert "1065" in out[0][1]                                       # the rewrite changing 1065 to 1064 refused
    assert (log["found"], log["fixed"], log["remaining"]) == (2, 1, 1)


# --- I7: terms ------------------------------------------------------------------------------------------------------

def test_durations_come_from_the_term_table_the_case_or_the_cited_text():
    assert ai.durations("entro otto giorni, 20 anni, da vent'anni, termine annuale, 3 mesi") == {
        (8, "giorn"), (20, "ann"), (1, "ann"), (3, "mes")}
    rows = {("1495", "c.c."): _row("1495", "Termini e condizioni", "Il compratore decade ... entro otto giorni")}
    signals = {"terms": [{"length": "1 anno", "text": "azione possessoria: termine annuale"}]}
    text = ("Il termine annuale è già decorso. La denuncia va fatta entro otto giorni (art. 1495 c.c.). Il diritto "
            "si prescrive in cinque anni. Ottavia passa da trent'anni.")
    out, log = ai.i7_terms(text, signals, "Ottavia passa da trent'anni sulla carrareccia.", rows, DOCS,
                           {("1495", "c.c.")})
    assert out == ("Il termine annuale è già decorso. La denuncia va fatta entro otto giorni (art. 1495 c.c.). "
                   "Ottavia passa da trent'anni.")
    assert log["entries"][0]["durations"] == ["5 ann"]


# --- the layer ------------------------------------------------------------------------------------------------------

def test_enforce_runs_every_invariant_and_logs_them(logged):
    draft = ("**1. Risposte ai quesiti**\n\nOttavia può passare ai sensi dell'art. 1062 c.c.\n\n"
             "**3. Norme e giurisprudenza**\n\nL'art. 2051 c.c. regola la custodia. Questa disposizione è pertinente. "
             "Il termine è DECORSO.\n\n**6. Conclusioni**\n\nOttavia vince.\n\n⚠️ Le sentenze [DA VERIFICARE].")
    rows = [_row("1062", "Destinazione del padre di famiglia", "La destinazione...", grade=3),
            _row("2051", "Danno cagionato da cosa in custodia", grade=1)]
    out, log = ai.enforce(draft, facts="Ottavia passa sulla carrareccia.", request="Ottavia può passare?",
                          analysis={}, article_rows=rows, ruling_rows=[], documents=DOCS, session=None,
                          chat_json=lambda *a, **k: [], call_chat=lambda *a, **k: "[]", cites={})
    assert out == ("**1. Risposte ai quesiti**\n\nOttavia può passare ai sensi dell'art. 1062 c.c.\n\n"
                   "**3. Norme e giurisprudenza**\n\nIl termine è già decorso.\n\n**6. Conclusioni**\n\nOttavia vince."
                   "\n\n⚠️ Le sentenze [DA VERIFICARE].")                    # the tail after ⚠️ untouched
    assert (log["I1"]["found"], log["I1"]["remaining"]) == (1, 0)
    assert log["I5"]["found"] == 1 and log["I5"]["remaining"] == 0
    assert log["guards"] == {"questions": 1, "questions_unanswered": [], "not_apply": 0, "orphans_removed": 1}
    assert [e for e, _ in logged] == ["case_analysis_invariants"]
