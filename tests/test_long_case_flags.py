"""The five flags from scripts/diagnose_long.py's run of 25 Oct (33 long cases), each off by default:
CHECKLIST_GATED, FRAME_SANITY, RESOLUTION_STRICT, CHECK_PRIORITY, PLAN_KEEP_CHECKED (all *_ENABLED).
Synthetic cases only."""

import json
import os
import re

import pytest

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

import src.rag.defensive_generation as dg  # noqa: E402
import src.rag.plan_then_write as ptw  # noqa: E402
from src.rag import issue_spotting, method_candidates, practical_checklist, vote_pool  # noqa: E402
from src.rag.nodes.cypher import CROSSREF_SOURCE  # noqa: E402

FLAGS = ("CHECKLIST_GATED_ENABLED", "FRAME_SANITY_ENABLED", "RESOLUTION_STRICT_ENABLED", "CHECK_PRIORITY_ENABLED",
         "PLAN_KEEP_CHECKED_ENABLED")


@pytest.fixture(autouse=True)
def flags_off(monkeypatch):
    for name in FLAGS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def logged(monkeypatch):
    events = []
    monkeypatch.setattr(dg, "vlog", lambda event, data=None, *a, **k: events.append((event, data)))
    return events


def _row(number, code, text, heading="", **tags):
    return {"d": {"id": f"doc-{code}", "name": "Codice Civile 2026" if code == "c.c." else code},
            "s": {"name": f"{number}.0.0", "plain_text": text, "titolo_articolo": heading},
            "_article": (number, code), **tags}


# --- F1 CHECKLIST_GATED -------------------------------------------------------------------------

def _analysis(areas, posture):
    return {"area": "", "frame": {"areas": areas, "posture_type": posture}}


def test_case_lists_unchanged_without_the_flag():
    assert method_candidates.case_lists(_analysis(["penale"], "precontenzioso")) == (["penale"], "precontenzioso")


def test_a_posture_of_another_family_is_dropped(monkeypatch):
    monkeypatch.setenv("CHECKLIST_GATED_ENABLED", "true")
    # a criminal case the frame called civil pre-litigation: no civil list
    assert method_candidates.case_lists(_analysis(["penale"], "precontenzioso")) == (["penale"], "")
    assert method_candidates.case_lists(_analysis(["contratti"], "condanna_primo_grado")) == (["contratti"], "")
    assert method_candidates.case_lists(_analysis(["contratti"], "altro")) == (["contratti"], "")
    # a fitting posture keeps only its family's areas
    assert method_candidates.case_lists(_analysis(["penale", "responsabilita_civile"], "condanna_primo_grado")) == \
        (["penale"], "condanna_primo_grado")
    assert method_candidates.case_lists(_analysis(["locazione"], "precontenzioso")) == (["locazione"], "precontenzioso")


def test_every_checklist_item_naming_an_article_is_read():
    items = [i for v in list(practical_checklist.POSTURE_CHECKS.values())
             + list(practical_checklist.AREA_CHECKS.values()) for i in v]
    for item in items:
        assert len(practical_checklist.item_refs(item)) >= len(re.findall(r"\bartt?\.", item)), item
    assert practical_checklist.item_refs(practical_checklist.POSTURE_CHECKS["giudicato_amministrativo_non_eseguito"][0]) \
        == [("112", "c.p.a."), ("113", "c.p.a."), ("114", "c.p.a.")]
    assert ("5", "d.lgs. 28/2010") in practical_checklist.item_refs(practical_checklist.AREA_CHECKS["locazione"][2])


def test_gated_checks_keep_an_item_only_with_its_articles():
    allowed = {("1453", "c.c."), ("1455", "c.c."), ("447-bis", "c.p.c."), ("657", "c.p.c."), ("658", "c.p.c."),
               ("5", "d.lgs. 28/2010")}
    checks = practical_checklist.gated_checks(["locazione"], "", lambda ref: ref in allowed)
    assert any("447-bis" in c for c in checks)                     # every article it names allowed
    assert not any("1575" in c for c in checks)                    # 1575/1587 not checked
    assert not any("Mediazione" in c for c in checks)              # a pre-litigation step, no such posture
    with_posture = practical_checklist.gated_checks(["locazione"], "precontenzioso", lambda ref: ref in allowed)
    assert any("Mediazione" in c for c in with_posture)
    assert any(c.startswith("Chi è la controparte") for c in with_posture)   # no article: kept


def test_the_plans_checklist_names_only_checked_articles(monkeypatch):
    analysis = _analysis(["responsabilita_civile"], "precontenzioso")
    rows = [_row("2043", "c.c.", "x", _applicability="applicabile"),
            _row("2947", "c.c.", "x"),                                       # not judged
            _row("1227", "c.c.", "x", _source=CROSSREF_SOURCE)]              # exempt: through
    assert any("2943" in c for c in ptw._checks(analysis, rows, "fatti", ""))          # flag off: as before
    monkeypatch.setenv("CHECKLIST_GATED_ENABLED", "true")
    checks = ptw._checks(analysis, rows, "fatti", "")
    assert not any(re.search(r"\b(2943|2946|2947|1219|633|696-bis)\b", c) for c in checks)
    assert any("Concorso del danneggiato (art. 1227" in c for c in checks)
    assert ptw._checks(_analysis(["penale"], "precontenzioso"), rows, "fatti", "") == \
        [c for c in practical_checklist.AREA_CHECKS["penale"] if not practical_checklist.item_refs(c)]


def test_the_method_lists_candidates_must_pass_the_check(monkeypatch, logged):
    monkeypatch.setenv("CHECKLIST_GATED_ENABLED", "true")
    monkeypatch.setattr(dg, "_APPLICABILITY_MAX_CHECKED", 1)
    rows = [_row("2043", "c.c.", "Qualunque fatto doloso o colposo..."),
            _row("2946", "c.c.", "Salvi i casi in cui la legge dispone diversamente...",
                 _source=dg.METHOD_CANDIDATES_SOURCE)]
    monkeypatch.setattr(dg, "_call_chat", lambda m, **k: json.dumps([{"article": "2043 c.c.", "verdict": "applicabile"}]))
    kept = dg._filter_applicable_articles("Tizio urta Caio.", "", rows)
    assert [r["_article"] for r in kept] == [("2043", "c.c.")]
    assert [d for e, d in logged if e == "case_analysis_applicability"][-1]["removed_unchecked"] == ["2946 c.c."]


# --- F2 FRAME_SANITY ----------------------------------------------------------------------------

def _frame(areas, posture):
    return {"facts": [], "procedural_posture": "", "posture_type": posture, "areas": areas,
            "tasks": ["?"], "central_issue_candidates": []}


@pytest.mark.parametrize("text, posture", [
    ("Al termine del giudizio Tizio è condannato per furto aggravato. Si chiede la strategia.", "condanna_primo_grado"),
    ("La Corte d'appello ha confermato la condanna di Tizio alla pena di tre anni.", "sentenza_penale_appello"),
    ("Caia sporge denuncia-querela contro il medico che l'ha operata.", "indagini_preliminari"),
    ("Assuma le vesti del difensore dell'imputato Tizio.", "giudizio_penale_primo_grado"),
    ("Mevio paga con un assegno scoperto. Quali reati sono configurabili?", "indagini_preliminari"),
])
def test_criminal_signals_set_penale_and_a_criminal_posture(text, posture, logged):
    frame = dg._sane_frame(_frame(["contratti"], "precontenzioso"), text)
    assert frame["areas"] == ["penale", "contratti"] and frame["posture_type"] == posture
    assert frame["areas_uncertain"] is True
    assert [e for e, _ in logged] == ["case_frame_sanity"]


def test_a_criminal_posture_already_there_is_kept(logged):
    frame = dg._sane_frame(_frame(["penale"], "indagini_preliminari"), "Tizio è indagato per lesioni.")
    assert frame["posture_type"] == "indagini_preliminari" and "areas_uncertain" not in frame


@pytest.mark.parametrize("text, areas, posture", [
    ("Il convenuto propone querela di falso contro la scrittura privata.", ["contratti"], "giudizio_civile_primo_grado"),
    ("Il Tribunale ha condannato la società al pagamento di 10.000 euro.", ["contratti"], "sentenza_civile_primo_grado"),
    ("Il Comune esclude Alfa dalla gara: è configurabile un grave illecito professionale.", ["amministrativo"],
     "provvedimento_amministrativo"),
])
def test_civil_and_administrative_cases_are_left_alone(text, areas, posture, logged):
    frame = _frame(areas, posture)
    assert dg._sane_frame(frame, text) is frame and not logged


def test_an_administrative_frame_is_left_as_it_is(logged):
    frame = _frame(["amministrativo"], "provvedimento_amministrativo")
    assert dg._sane_frame(frame, "Alfa è esclusa dalla gara per la condanna penale del suo amministratore per il "
                                 "reato di turbata libertà degli incanti.") is frame and not logged


def test_the_frame_is_corrected_only_with_the_flag(monkeypatch, logged):
    reply = {"facts": ["x"], "procedural_posture": "condanna", "posture_type": "sentenza_civile_primo_grado",
             "areas": ["responsabilita_civile"], "tasks": ["Quale impugnazione?"], "central_issue_candidates": []}
    monkeypatch.setattr(dg, "_call_chat", lambda m, **k: json.dumps(reply))
    text = "Tizio è stato condannato alla pena di due anni per lesioni colpose."
    assert dg._case_frame(text, "")["posture_type"] == "sentenza_civile_primo_grado"
    monkeypatch.setenv("FRAME_SANITY_ENABLED", "true")
    frame = dg._case_frame(text, "")
    assert frame["posture_type"] == "condanna_primo_grado" and frame["areas"] == ["penale", "responsabilita_civile"]


def test_uncertain_areas_switch_the_vote_filter_off(monkeypatch):
    analysis = {"frame": {"areas": ["penale", "responsabilita_civile"], "areas_uncertain": True}}
    assert vote_pool.filter_areas(analysis) == ["penale", "responsabilita_civile"]
    monkeypatch.setenv("FRAME_SANITY_ENABLED", "true")
    assert vote_pool.filter_areas(analysis) == []
    assert vote_pool.allowed_codes(vote_pool.filter_areas(analysis), set()) is None
    assert vote_pool.filter_areas({"frame": {"areas": ["penale"]}}) == ["penale"]


# --- F3 RESOLUTION_STRICT -----------------------------------------------------------------------

DOCS = [{"id": "doc-cc", "name": "Codice Civile 2026"}, {"id": "doc-cp", "name": "Codice Penale 2026"}]
HEADINGS = {"81": "Risarcimento dei danni", "2043": "Risarcimento per fatto illecito",
            "544-quinquies": "Divieto di combattimenti tra animali", "110": "Pena per coloro che concorrono nel reato"}


@pytest.fixture
def corpus(monkeypatch, logged):
    """article_by_title finds `titled[term]`; the full-text search finds `fulltext`."""
    state = {"titled": {}, "fulltext": []}
    monkeypatch.setattr(issue_spotting, "article_by_title",
                        lambda session, doc_id, term, field="title": state["titled"].get(term) if field == "titolo_articolo" else None)
    monkeypatch.setattr(issue_spotting, "fetch_article", lambda session, doc_id, number: [
        {"d": {"id": doc_id}, "s": {"name": f"{number}.0.0", "titolo_articolo": HEADINGS.get(number, ""),
                                    "plain_text": "testo"}}])
    monkeypatch.setattr(dg, "_issue_article_refs", lambda session, doc_id, terms: state["fulltext"])
    state["logged"] = logged
    return state


def _resolve(nome, codice, numero=""):
    rows = dg._resolve_institution(None, DOCS, {"nome": nome, "codice": codice, "numero_indicativo": numero})
    return [r["_article"][0] for r in rows]


def test_a_generic_rubrica_alone_resolves_nothing(monkeypatch, corpus):
    corpus["titled"] = {"risarcimento dei danni": "81"}
    corpus["fulltext"] = ["2043"]
    assert _resolve("risarcimento dei danni", "c.c.") == ["81"]                   # flag off: as before
    monkeypatch.setenv("RESOLUTION_STRICT_ENABLED", "true")
    assert _resolve("risarcimento dei danni", "c.c.") == ["2043"]                 # 81 refused, full text next
    assert _resolve("risarcimento dei danni", "c.c.", numero="81") == ["81"]      # the model's own number
    rejected = [d for e, d in corpus["logged"] if e == "case_analysis_institution_rejected"]
    assert rejected[0]["why"] == "generic heading alone" and rejected[0]["article"] == "81 c.c."


def test_a_full_text_hit_must_share_a_word_with_the_institution(monkeypatch, corpus):
    corpus["fulltext"] = ["544-quinquies", "110"]
    assert _resolve("concorso di persone", "c.p.") == ["544-quinquies", "110"]
    monkeypatch.setenv("RESOLUTION_STRICT_ENABLED", "true")
    assert _resolve("concorso di persone", "c.p.") == ["110"]                     # "concorrono" shares "conco"
    corpus["fulltext"] = ["544-quinquies"]
    assert _resolve("concorso di persone", "c.p.") == []


def test_generic_headings():
    assert dg._generic_heading("Risarcimento dei danni") and dg._generic_heading("Accessioni")
    assert dg._generic_heading("Prescrizione")
    assert not dg._generic_heading("Interruzione della prescrizione")
    assert not dg._generic_heading("Risarcimento per fatto illecito") and not dg._generic_heading("")


# --- F4 CHECK_PRIORITY --------------------------------------------------------------------------

CASE = "Tizio possiede il fondo da venticinque anni, in modo pacifico e pubblico. Caio rivendica il fondo."


class Check:
    """The applicability model: every article applicable but those in `incompatible` (label -> fatto)."""

    def __init__(self, incompatible=None):
        self.incompatible, self.seen, self.systems = dict(incompatible or {}), [], []

    def __call__(self, messages, max_tokens=None, **kwargs):
        self.systems.append(messages[0].content)
        labels = re.findall(r"^\[([^\]]+)\]$", messages[-1].content, re.MULTILINE)
        self.seen += labels
        return json.dumps([{"article": label, "verdict": "incompatibile", "requisito": "il possesso ventennale",
                            "fatto": self.incompatible[label], "reason": "x"} if label in self.incompatible
                           else {"article": label, "verdict": "applicabile"} for label in labels])


def _pool():
    return [_row("1140", "c.c.", "Il possesso è il potere sulla cosa..."),
            _row("1158", "c.c.", "La proprietà dei beni immobili si acquista in virtù del possesso continuato per "
                                 "venti anni.", _multi_signal=True),
            _row("948", "c.c.", "Il proprietario può rivendicare la cosa...", _vote=True),
            _row("1168", "c.c.", "Chi è stato violentemente spogliato...", _source=dg.METHOD_CANDIDATES_SOURCE),
            _row("1163", "c.c.", "Il possesso acquistato in modo violento o clandestino...")]


def test_the_checked_slots_go_by_priority_and_the_unchecked_are_removed(monkeypatch, logged):
    monkeypatch.setattr(dg, "_APPLICABILITY_MAX_CHECKED", 3)
    model = Check()
    monkeypatch.setattr(dg, "_call_chat", model)
    kept = dg._filter_applicable_articles(CASE, "", _pool())
    assert model.seen == ["1140 c.c.", "1158 c.c.", "948 c.c."] and len(kept) == 5       # flag off: as before
    monkeypatch.setenv("CHECK_PRIORITY_ENABLED", "true")
    model.seen = []
    kept = dg._filter_applicable_articles(CASE, "", _pool())
    assert model.seen == ["948 c.c.", "1158 c.c.", "1140 c.c."]      # voted, multi-signal, the rest; method last
    assert [r["_article"][0] for r in kept] == ["1140", "1158", "948"]
    log = [d for e, d in logged if e == "case_analysis_applicability"][-1]
    assert log["removed_unchecked"] == ["1163 c.c.", "1168 c.c."] and log["unchecked_labels"] == ["1163 c.c.", "1168 c.c."]


def test_a_rejection_drops_with_the_priority_flag_too(monkeypatch):
    # the rule "a rejection without a quote counts as applicable" is gone: it revived 46 wrong articles
    monkeypatch.setenv("CHECK_PRIORITY_ENABLED", "true")
    model = Check({"1158 c.c.": "il possesso dura da meno di venti anni"})
    monkeypatch.setattr(dg, "_call_chat", model)
    assert "1158" not in {r["_article"][0] for r in dg._filter_applicable_articles(CASE, "", _pool())}
    assert model.systems[0] == dg._APPLICABILITY_SYSTEM


# --- CHECK_GRADED -------------------------------------------------------------------------------

class Grades:
    """The graded model: grade per label (default 2), short keys as asked."""

    def __init__(self, grades):
        self.grades, self.humans, self.calls = dict(grades), [], []

    def __call__(self, messages, max_tokens=None, temperature=None, **kwargs):
        self.calls.append({"system": messages[0].content, "max_tokens": max_tokens, "temperature": temperature})
        self.humans.append(messages[-1].content)
        labels = re.findall(r"^\[([^\]]+)\]$", messages[-1].content, re.MULTILINE)
        return json.dumps([{"a": label, "g": self.grades.get(label, 2), "q": "usucapione", "r": "x"} for label in labels])


GRADED_ANALYSIS = {"questioni": [{"titolo": "Usucapione del fondo"}, {"titolo": "Rivendicazione"}],
                   "frame": {"tasks": ["Chi è proprietario del fondo?"],
                             "central_issue_candidates": [{"nome": "usucapione", "codice": "c.c."}]}}


def test_graded_check_keeps_by_grade_and_removes_grade_zero(monkeypatch, logged):
    monkeypatch.setenv("CHECK_GRADED_ENABLED", "true")
    model = Grades({"1158 c.c.": 3, "1140 c.c.": 1, "1163 c.c.": 0})
    monkeypatch.setattr(dg, "_call_chat", model)
    pool = _pool()
    pool[1]["s"]["titolo_articolo"] = "Usucapione dei beni immobili e dei diritti reali immobiliari"
    pool[1]["s"]["plain_text"] = "La proprietà dei beni immobili " + "x" * 2000
    kept = {r["_article"][0]: r for r in dg._filter_applicable_articles(CASE, "", pool, GRADED_ANALYSIS)}
    assert "1163" not in kept                                                       # grade 0: out of the pool
    assert kept["1158"]["_grade"] == 3 and kept["1158"]["_applicability"] == "applicabile"
    assert kept["1140"]["_grade"] == 1 and kept["1140"]["_applicability"] == "marginale"
    assert kept["948"]["_grade"] == 2 and kept["948"]["_grade_issue"] == "usucapione"
    human = model.humans[0]
    text = ("La proprietà dei beni immobili " + "x" * 2000)[:400]
    assert f"Rubrica: Usucapione dei beni immobili e dei diritti reali immobiliari\nTesto: {text}\n\n" in human
    assert "Questione centrale: usucapione" in human and "- Rivendicazione" in human and "- Chi è proprietario" in human
    assert model.calls[0]["temperature"] == 0 and model.calls[0]["max_tokens"] == 60 + 45 * 5
    assert "voto di pertinenza" in model.calls[0]["system"]
    log = [d for e, d in logged if e == "case_analysis_applicability"][-1]
    assert log["grades"]["1158 c.c."] == 3 and log["verdicts"]["1163 c.c."] == "grade 0"
    assert log["dropped"][0]["article"] == "1163 c.c." and log["dropped"][0]["grade"] == 0


def test_a_grade_one_article_does_not_pass_for_the_checklist(monkeypatch):
    rows = [dict(_row("2943", "c.c.", "x"), _applicability="marginale", _grade=1)]
    assert ptw._through_check(rows, "fatti", "") == []


def test_graded_plan_keeps_twelve_by_grade_then_fusion_rank(monkeypatch):
    monkeypatch.setenv("CHECK_GRADED_ENABLED", "true")
    rows = [dict(_row(str(n), "c.c.", "x"), _grade=2, _fused_rank=n) for n in range(20, 0, -1)]
    rows += [dict(_row("1158", "c.c.", "x"), _grade=3, _fused_rank=40), dict(_row("1140", "c.c.", "x"), _grade=1)]
    kept = ptw._checked(rows)
    assert kept[0] == ("1158", "c.c.") and len(kept) == 12 and ("1140", "c.c.") not in kept
    assert kept[1:] == [(str(n), "c.c.") for n in range(1, 12)]


def test_graded_plan_forces_no_catch_all():
    plan = {"questioni": [{"titolo": "Usucapione del fondo", "norme": [], "pronunce": [],
                           "argomento": "Il possesso ventennale fonda l'acquisto.", "obiezione": "", "replica": ""}]}
    headings = {("1158", "c.c."): "Usucapione dei beni immobili", ("1227", "c.c."): "Concorso del fatto colposo"}
    issue_of = {("1227", "c.c."): "risarcimento"}
    out, placed, grouped, added = ptw._group_checked(plan, [("1158", "c.c."), ("1227", "c.c.")], headings, 7,
                                                     issue_of, catch_all=False)
    assert grouped == {("1158", "c.c."): "Usucapione del fondo"} and added == []
    assert [q["titolo"] for q in out["questioni"]] == ["Usucapione del fondo"]


# --- F5 PLAN_KEEP_CHECKED -----------------------------------------------------------------------

def test_every_checked_article_is_placed_by_issue(monkeypatch, logged):
    plan = {"questioni": [{"titolo": "Usucapione del fondo", "norme": [("1158", "c.c.")], "pronunce": [],
                           "argomento": "Il possesso ventennale fonda l'acquisto.", "obiezione": "", "replica": ""},
                          {"titolo": "Rivendicazione", "norme": [], "pronunce": [],
                           "argomento": "Caio deve provare la proprietà.", "obiezione": "", "replica": ""}]}
    headings = {("1158", "c.c."): "Usucapione dei beni immobili", ("1140", "c.c."): "Possesso",
                ("948", "c.c."): "Azione di rivendicazione",
                ("1227", "c.c."): "Concorso del fatto colposo del creditore"}
    checked = [("1158", "c.c."), ("1140", "c.c."), ("948", "c.c."), ("1227", "c.c.")]
    out, placed, grouped, added = ptw._group_checked(plan, checked, headings, 7)
    assert placed == [("1158", "c.c.")]
    assert grouped == {("1140", "c.c."): "Usucapione del fondo", ("948", "c.c."): "Rivendicazione"}
    assert added == [("1227", "c.c.")] and out["questioni"][-1]["titolo"] == ptw.OTHER_NORMS_TITLE
    assert plan["questioni"][1]["norme"] == []                                           # the input untouched
    full, _, grouped, added = ptw._group_checked(plan, checked, headings, 2)
    assert len(full["questioni"]) == 2 and ("1227", "c.c.") in full["questioni"][-1]["norme"] and not added


def test_checked_means_judged_not_exempt():
    rows = [_row("1158", "c.c.", "x", _applicability="applicabile"), _row("1159", "c.c.", "x", _applicability="alternativa"),
            _row("1140", "c.c.", "x", _source=CROSSREF_SOURCE), _row("1163", "c.c.", "x")]
    assert ptw._checked(rows) == [("1158", "c.c."), ("1159", "c.c.")]


def test_method_candidates_follow_the_verified_lists(monkeypatch, logged):
    from src.rag import doc_lookup
    monkeypatch.setenv("METHOD_CANDIDATES_ENABLED", "true")
    monkeypatch.setenv("APPLICABILITY_CHECK_ENABLED", "true")
    monkeypatch.setattr(doc_lookup, "_fetch_doc_names", lambda driver, database: [])
    monkeypatch.setattr(dg, "_fetch_any", lambda session, documents, refs: [_row(n, c, "x") for n, c in refs])
    analysis = _analysis(["penale"], "precontenzioso")           # a criminal case called civil pre-litigation

    def added():
        return dg._add_method_candidates(None, None, "neo4j", analysis, {"article_rows": [], "selected": []})["method_candidates"]
    assert ("1219", "c.c.") in added() and ("56", "c.p.") in added()                  # flag off: both lists
    monkeypatch.setenv("CHECKLIST_GATED_ENABLED", "true")
    assert ("1219", "c.c.") not in added() and ("56", "c.p.") in added()
