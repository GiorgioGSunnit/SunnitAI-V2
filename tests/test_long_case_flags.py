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
         "PLAN_KEEP_CHECKED_ENABLED", "CHECK_GRADED_ENABLED", "GRADER_EVIDENCE_ENABLED", "FLAGGED_REMOVE_ENABLED",
         "GRADER_EXAMPLES_ENABLED", "WRITER_RULES_ENABLED", "FACT_REGIMES_ENABLED")


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
    """The graded model: grade per label (default 2), short keys and the ids sent, as asked. skip: labels
    it leaves out on the first n calls; cut_after: the first reply cut after that many entries; fail: the
    first n calls raise."""

    def __init__(self, grades, skip=(), skip_calls=1, cut_after=None, fail=0):
        self.grades, self.humans, self.calls = dict(grades), [], []
        self.skip, self.skip_calls, self.cut_after, self.fail = set(skip), skip_calls, cut_after, fail

    def __call__(self, messages, max_tokens=None, temperature=None, **kwargs):
        self.calls.append({"system": messages[0].content, "max_tokens": max_tokens, "temperature": temperature})
        self.humans.append(messages[-1].content)
        n = len(self.calls)
        if n <= self.fail:
            raise RuntimeError("upstream timeout")
        sent = re.findall(r"^\[(A\d+)\] (.+)$", messages[-1].content, re.MULTILINE)
        entries = [{"a": key, "g": self.grades.get(label, 2), "q": "usucapione", "r": "x"} for key, label in sent
                   if not (label in self.skip and n <= self.skip_calls)]
        reply = json.dumps(entries)
        if self.cut_after is not None and n == 1:
            reply = json.dumps(entries[:self.cut_after])[:-1] + ', {"a": "A9", "g'
        return reply


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
    assert (f"Rubrica: Usucapione dei beni immobili e dei diritti reali immobiliari\nTrovato da: più fonti\n"
            f"Testo: {text}\n\n") in human                                           # multi-signal (T1)
    assert "[A1] 1140 c.c.\nRubrica: -\nTesto:" in human                            # one channel: no mark
    assert "nel dubbio tra 1 e 2 scegli 2" in model.calls[0]["system"]
    assert "Questione centrale: usucapione" in human and "- Rivendicazione" in human and "- Chi è proprietario" in human
    assert model.calls[0]["temperature"] == 0 and model.calls[0]["max_tokens"] == 120 + 60 * 5
    assert "[A2] 1158 c.c.\nRubrica:" in human and "A1, A2, A3, A4, A5" in human
    assert "voto di pertinenza" in model.calls[0]["system"]
    log = [d for e, d in logged if e == "case_analysis_applicability"][-1]
    assert log["grades"]["1158 c.c."] == 3 and log["verdicts"]["1163 c.c."] == "grade 0"
    assert log["dropped"][0]["article"] == "1163 c.c." and log["dropped"][0]["grade"] == 0
    assert len(model.calls) == 1 and log["ungraded"] == []                     # nothing missing, no retry


def test_a_cut_reply_keeps_its_grades_and_the_rest_are_retried_in_a_smaller_batch(monkeypatch, logged):
    monkeypatch.setenv("CHECK_GRADED_ENABLED", "true")
    monkeypatch.setattr(dg, "_GRADED_RETRY_BATCH", 2)
    model = Grades({"1158 c.c.": 3}, cut_after=2)
    monkeypatch.setattr(dg, "_call_chat", model)
    kept = {r["_article"][0]: r for r in dg._filter_applicable_articles(CASE, "", _pool(), GRADED_ANALYSIS)}
    assert all(r.get("_grade") is not None for r in kept.values())               # every article graded in the end
    retried = [re.findall(r"^\[A\d+\] (.+)$", h, re.MULTILINE) for h in model.humans[1:]]
    assert retried == [["948 c.c.", "1168 c.c."], ["1163 c.c."]]                 # only the 3 lost to the cut
    batch_log = next(d for e, d in logged if e == "case_analysis_applicability" and "retried" in d)
    assert batch_log["cut_replies"] == 1 and batch_log["retried"] == 3 and batch_log["recovered"] == 3


def test_an_id_still_missing_after_the_retry_stays_ungraded(monkeypatch, logged):
    monkeypatch.setenv("CHECK_GRADED_ENABLED", "true")
    monkeypatch.setenv("CHECK_PRIORITY_ENABLED", "true")
    monkeypatch.setenv("CHECKLIST_GATED_ENABLED", "true")
    model = Grades({}, skip={"1158 c.c.", "1168 c.c."}, skip_calls=99)
    monkeypatch.setattr(dg, "_call_chat", model)
    kept = {r["_article"][0]: r for r in dg._filter_applicable_articles(CASE, "", _pool(), GRADED_ANALYSIS)}
    assert len(model.calls) == 2 and re.findall(r"^\[A\d+\] (.+)$", model.humans[1], re.MULTILINE) == \
        ["1158 c.c.", "1168 c.c."]
    # kept for the writer, method candidate included, but not passing the check nor kept by the plan
    assert kept["1158"]["_applicability"] == "non valutato" and "_grade" not in kept["1158"]
    assert kept["1168"]["_applicability"] == "non valutato"
    passed = ptw._through_check(list(kept.values()), "fatti", "")
    assert ("948", "c.c.") in passed and ("1158", "c.c.") not in passed and ("1168", "c.c.") not in passed
    log = [d for e, d in logged if e == "case_analysis_applicability"][-1]
    assert log["ungraded"] == ["1158 c.c.", "1168 c.c."] and log["verdicts"]["1158 c.c."] == "ungraded (kept)"
    assert log["removed_unchecked"] == []
    monkeypatch.setenv("PLAN_KEEP_CHECKED_ENABLED", "true")
    assert ("1158", "c.c.") not in ptw._checked(list(kept.values()))


def test_a_failed_graded_call_is_retried(monkeypatch, logged):
    monkeypatch.setenv("CHECK_GRADED_ENABLED", "true")
    monkeypatch.setenv("CHECK_PRIORITY_ENABLED", "true")
    model = Grades({"1158 c.c.": 3}, fail=1)
    monkeypatch.setattr(dg, "_call_chat", model)
    kept = {r["_article"][0]: r for r in dg._filter_applicable_articles(CASE, "", _pool(), GRADED_ANALYSIS)}
    assert len(kept) == 5 and kept["1158"]["_grade"] == 3
    batch_log = next(d for e, d in logged if e == "case_analysis_applicability" and "retried" in d)
    assert batch_log["failed_batches"] == 1 and batch_log["recovered"] == 5 and batch_log["error"].startswith("error:")


def test_a_reply_by_label_or_wrapped_still_counts():
    ids = {"A1": "640-ter c.p.", "A2": "1453 c.c."}
    wrapped = '```json\n{"articoli": [{"a": "Art. 640-ter c.p.", "g": "2"}, {"a": "[A2]", "g": 0}]}\n```'
    assert {k: v["grade"] for k, v in dg._graded_reply(wrapped, ids).items()} == {"640-ter c.p.": 2, "1453 c.c.": 0}
    assert dg._graded_reply('[{"a": "A1", "g": 5}, {"a": "A7", "g": 1}]', ids) == {}     # no grade, unknown id


def test_the_graded_check_has_twice_the_slots(monkeypatch, logged):
    monkeypatch.setenv("CHECK_GRADED_ENABLED", "true")
    monkeypatch.setattr(dg, "_GRADED_MAX_CHECKED", 4)
    monkeypatch.setattr(dg, "_call_chat", Grades({}))
    dg._filter_applicable_articles(CASE, "", _pool(), GRADED_ANALYSIS)
    log = [d for e, d in logged if e == "case_analysis_applicability"][-1]
    assert log["max_checked"] == 4 and len(log["unchecked_labels"]) == 1


def test_a_grade_one_article_does_not_pass_for_the_checklist(monkeypatch):
    rows = [dict(_row("2943", "c.c.", "x"), _applicability="marginale", _grade=1)]
    assert ptw._through_check(rows, "fatti", "") == []


def test_graded_plan_keeps_fourteen_by_grade_then_fusion_rank(monkeypatch):
    monkeypatch.setenv("CHECK_GRADED_ENABLED", "true")
    rows = [dict(_row(str(n), "c.c.", "x"), _grade=2, _fused_rank=n) for n in range(20, 0, -1)]
    rows += [dict(_row("1158", "c.c.", "x"), _grade=3, _fused_rank=40), dict(_row("1140", "c.c.", "x"), _grade=1)]
    kept = ptw._checked(rows)
    assert kept[0] == ("1158", "c.c.") and len(kept) == 14 and ("1140", "c.c.") not in kept
    assert kept[1:] == [(str(n), "c.c.") for n in range(1, 14)]


# --- T2: exempt articles linked to a forced one -----------------------------------------------------

def _sempronio_rows():
    return [dict(_row("584", "c.p.", "Chiunque, con atti diretti a commettere uno dei delitti preveduti dagli "
                                     "articoli 581 e 582, cagiona la morte di un uomo...", "Omicidio preterintenzionale"),
                 _heading_match=True, _central=True, _fused_rank=1),
            dict(_row("581", "c.p.", "Chiunque percuote taluno...", "Percosse"), _source=CROSSREF_SOURCE),
            dict(_row("582", "c.p.", "Chiunque cagiona ad alcuno una lesione personale...", "Lesione personale"),
                 _source=CROSSREF_SOURCE),
            dict(_row("1", "c.p.", "Nessuno può essere punito...", "Reati e pene"), _source=CROSSREF_SOURCE),
            dict(_row("589", "c.p.", "Chiunque cagiona per colpa la morte...", "Omicidio colposo"), _grade=2,
                 _fused_rank=2),
            dict(_row("41", "c.p.", "Il concorso di cause preesistenti...", "Concorso di cause"), _heading_match=True)]


def test_forced_holds_the_central_articles_and_their_cross_references(monkeypatch):
    monkeypatch.setenv("CHECK_GRADED_ENABLED", "true")
    forced = ptw._forced(_sempronio_rows())
    assert list(forced) == [("584", "c.p."), ("581", "c.p."), ("582", "c.p."), ("589", "c.p.")]
    assert forced[("584", "c.p.")] == "central issue" and forced[("581", "c.p.")] == "cross-reference of 584 c.p."
    # 1 c.p. is cross-referenced from nothing forced; 41 c.p. is heading-matched but not central
    assert ("1", "c.p.") not in forced and ("41", "c.p.") not in forced
    monkeypatch.setattr(ptw, "_GRADED_KEEP", 2)
    assert list(ptw._forced(_sempronio_rows())) == [("584", "c.p."), ("581", "c.p.")]


def test_a_cross_reference_goes_under_its_parents_issue_and_a_central_article_under_the_first():
    plan = {"questioni": [{"titolo": "Qualificazione del fatto", "norme": [("584", "c.p.")], "pronunce": [],
                           "argomento": "Manca il dolo di percosse.", "obiezione": "", "replica": ""},
                          {"titolo": "Pena", "norme": [], "pronunce": [], "argomento": "Le attenuanti.",
                           "obiezione": "", "replica": ""}]}
    reasons = {("584", "c.p."): "central issue", ("581", "c.p."): "cross-reference of 584 c.p.",
               ("43", "c.p."): "central issue"}
    out, placed, grouped, added = ptw._group_checked(plan, list(reasons), {("43", "c.p."): "Elemento psicologico"},
                                                     7, {}, catch_all=False, reasons=reasons)
    assert grouped == {("581", "c.p."): "Qualificazione del fatto", ("43", "c.p."): "Qualificazione del fatto"}
    assert out["questioni"][0]["norme"] == [("584", "c.p."), ("581", "c.p."), ("43", "c.p.")]


def test_central_questions_are_the_frames_central_issues():
    analysis = {"frame": {"central_issue_candidates": [{"nome": "Omicidio preterintenzionale", "codice": "c.p."}]},
                "questioni": [{"titolo": "Questione centrale: x", "voce": "questione_centrale", "istituti": []},
                              {"titolo": "Reato", "istituti": [{"nome": "omicidio  preterintenzionale"}]},
                              {"titolo": "Pena", "istituti": [{"nome": "attenuanti generiche"}]}]}
    assert dg._central_questions(analysis) == {0, 1}
    assert dg._central_questions({}) == set()


# --- T3: citations only from the section's material ---------------------------------------------------

class Rephrase:
    def __init__(self, reply):
        self.reply, self.humans = reply, []

    def __call__(self, messages, max_tokens=None, temperature=None, **kwargs):
        self.humans.append(messages[-1].content)
        return self.reply


MATERIAL = "TESTI DELLE NORME:\n[art. 2051 c.c.] (Danno cagionato da cosa in custodia)\nCiascuno è responsabile..."


def test_a_citation_the_material_lacks_is_rephrased_away(monkeypatch):
    model = Rephrase('[{"i": 1, "frase": "La penale è riducibile dal giudice."}]')
    monkeypatch.setattr(dg, "_call_chat", model)
    text = ("Il custode risponde ai sensi dell'art. 2051 c.c. La penale è riducibile ex art. 1384 c.c. dal giudice. "
            "Si veda Cass. civ., sez. III, n. 190/2026.")
    out, log = ptw._cite_only_material("Applicazione al caso", text, MATERIAL, [], [])
    assert out == ("Il custode risponde ai sensi dell'art. 2051 c.c. La penale è riducibile dal giudice. "
                   "Si veda Cass. civ., sez. III, n. 190/2026.")
    assert "1. La penale è riducibile ex art. 1384 c.c. dal giudice.\n   togli: art. 1384 c.c." == model.humans[0]
    assert log == [{"section": "Applicazione al caso", "removed": ["1384 c.c."], "how": "rephrased",
                    "sentence": "La penale è riducibile ex art. 1384 c.c. dal giudice.",
                    "now": "La penale è riducibile dal giudice."}]


def test_a_citation_the_case_names_stays_and_no_call_is_made(monkeypatch):
    model = Rephrase("[]")
    monkeypatch.setattr(dg, "_call_chat", model)
    text = "La penale è riducibile ex art. 1384 c.c., come chiede il cliente."
    assert ptw._cite_only_material("x", text, MATERIAL, [("1384", "c.c.")], []) == (text, [])
    assert model.humans == []


def test_a_rewrite_still_citing_from_memory_drops_the_sentence(monkeypatch):
    monkeypatch.setattr(dg, "_call_chat", Rephrase('[{"i": 1, "frase": "Vale l\'art. 156 c.p.p."}]'))
    text = "Prima frase.\nLa notifica è nulla ai sensi dell'art. 159 c.p.p. Seconda frase.\n- Terza: art. 33 Codice del Consumo."
    docs = [{"id": "x", "name": "Codice del Consumo 2026", "aliases": []}]
    out, log = ptw._cite_only_material("x", text, MATERIAL, [], docs)
    assert out == "Prima frase.\nSeconda frase."                      # both dropped, the bullet whole; a law by name too
    assert [e["how"] for e in log] == ["sentence dropped", "sentence dropped"]
    assert log[1]["removed"] == ["33 Codice del Consumo 2026"]



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


# --- FLAGGED_REMOVE ---------------------------------------------------------------------------------

def test_the_citation_context_is_the_sentence_citing_that_article():
    draft = ("Il proprietario risponde ex art. 2053 c.c. per la rovina dell'edificio, e non per altro.\n"
             "Ai sensi dell'art. 658 c.p.c. si intima lo sfratto per finita locazione al conduttore moroso.\n"
             "L'art. 658 Codice Civile prevede la nullità della donazione di beni futuri nel caso in esame.")
    assert dg._citation_context(draft, ("658", "c.c.")).startswith("Ai sensi")            # without documents: as before
    assert dg._citation_context(draft, ("658", "c.c."), []).startswith("L'art. 658 Codice Civile")
    assert dg._citation_context(draft, ("1982", "c.c."), []) == ""                        # cited nowhere: not scored


class Qualification:
    """The qualification check (every article incompatible, tipo per label) and the sentence rewrite."""

    def __init__(self, tipo, rewrite="[]"):
        self.tipo, self.rewrite, self.systems, self.humans = tipo, rewrite, [], []

    def __call__(self, messages, max_tokens=None, temperature=None, **kwargs):
        system = messages[0].content
        self.systems.append(system)
        self.humans.append(messages[-1].content)
        if "Riscrivi ogni frase" in system:
            return self.rewrite
        labels = re.findall(r"^\[([^\]]+)\]$", messages[-1].content, re.MULTILINE)
        return json.dumps([{"articolo": label, "compatibile": False, "perche": "x", "tipo": self.tipo.get(label, "materia")}
                           for label in labels])


H3_DRAFT = ("**Art. 757 c.c. – Effetti della divisione**\n\nIl coerede è reputato solo e immediato successore.\n\n"
            "La servitù di passaggio spetta ex art. 1054 c.c. al fondo intercluso dalla divisione. La clausola è "
            "riducibile ex art. 1384 c.c. dal giudice. Il reato di cui all'art. 600-bis c.p. non rileva qui. "
            "Il fondo di Mevio era uno solo.\n\nSul punto si veda anche l'art. 681 c.c.")


def _h3_rows():
    return [dict(_row("757", "c.c.", "Ogni coerede è reputato solo e immediato successore...",
                      "Retroattività della divisione"), _grade=1),
            dict(_row("1054", "c.c.", "Se, in conseguenza di alienazione o di divisione...",
                      "Interclusione per effetto di alienazione o di divisione"), _grade=2),
            dict(_row("681", "c.c.", "Le fronde degli alberi...", "Luci e vedute"))]


def test_flagged_citations_go_unless_the_pipeline_has_its_own_evidence(monkeypatch, logged):
    monkeypatch.setenv("FLAGGED_REMOVE_ENABLED", "true")
    monkeypatch.setattr(dg, "_fetch_any", lambda session, documents, refs: [
        _row("600-bis", "c.p.", "Chiunque recluta o induce alla prostituzione...", "Prostituzione minorile"),
        _row("1384", "c.c.", "La penale può essere diminuita...", "Riduzione della penale")])
    model = Qualification({"1054 c.c.": "formulazione"}, rewrite='[{"i": 1, "frase": ""}, {"i": 2, "frase": ""}]')
    monkeypatch.setattr(dg, "_call_chat", model)
    out = dg._check_qualifications(H3_DRAFT, {}, _h3_rows(), None, [], "Il cliente chiede dell'art. 1384 c.c.")
    assert "757" not in out and "Il coerede è reputato" not in out                   # heading with its paragraph
    assert "600-bis" not in out and "681" not in out                                   # single channel, ungraded
    assert "ex art. 1054 c.c. al fondo intercluso" in out and "DA VERIFICARE: l'art. 1054" not in out  # grade 2, wording
    assert "art. 1384 c.c. [DA VERIFICARE: l'art. 1384 c.c. riguarda «Riduzione della penale»]" in out  # named: kept, marked
    assert "Il fondo di Mevio era uno solo." in out
    assert '"tipo"' in model.systems[0]
    log = [d for e, d in logged if e == "case_analysis_qualification"][-1]
    assert sorted(log["removed"]) == ["600-bis c.p.", "681 c.c.", "757 c.c."]
    assert log["kept"] == {"1054 c.c.": "grade 2", "1384 c.c.": "named in the case"} and log["note_dropped"] == ["1054 c.c."]
    removal = [d for e, d in logged if e == "case_analysis_flagged_removed"][-1]
    assert [e["how"] for e in removal["removed"]] == ["heading and paragraph dropped", "sentence dropped", "sentence dropped"]


def test_flagged_citations_are_only_marked_without_the_flag(monkeypatch, logged):
    monkeypatch.setattr(dg, "_fetch_any", lambda session, documents, refs: [])
    monkeypatch.setattr(dg, "_call_chat", Qualification({}))
    out = dg._check_qualifications(H3_DRAFT, {}, _h3_rows(), None, [])
    assert "Art. 757 c.c. [DA VERIFICARE: l'art. 757 c.c. riguarda «Retroattività della divisione»]" in out
    assert "art. 1054 c.c. [DA VERIFICARE" in out and "681 c.c. [DA VERIFICARE" in out


class _Session:
    def __init__(self, types):
        self.types = types

    def run(self, query, **params):
        class _R:
            def __init__(self, rows):
                self.rows = rows

            def data(self):
                return self.rows
        return _R([{"name": n, "types": self.types.get(n, [])} for n in params.get("names", [])])


MAXIMS = "CORTE SUPREMA DI CASSAZIONE – PRINCIPÎ DI DIRITTO 2025"


def test_an_article_of_a_document_that_is_not_a_law_is_not_cited(monkeypatch, logged):
    monkeypatch.setattr(dg, "_call_chat", Qualification({}, rewrite='[{"i": 1, "frase": "La massima conferma la tesi."}]'))
    docs = [{"id": "m", "name": MAXIMS, "aliases": []}]
    draft = f"La servitù spetta ex art. 1054 c.c. al fondo. La massima di cui all'art. 563 {MAXIMS} conferma la tesi."
    out = dg._citation_hygiene(draft, _Session({MAXIMS: ["special"]}), docs, [], "Tizio chiede del suo fondo.")
    assert out == "La servitù spetta ex art. 1054 c.c. al fondo. La massima conferma la tesi."
    assert [d for e, d in logged if e == "case_analysis_citation_hygiene"][-1]["types"] == {MAXIMS: ["special"]}
    # a law stays; a CCNL stays when the case is about one; nothing named in the case is removed
    assert dg._citation_hygiene(draft, _Session({MAXIMS: ["primary"]}), docs, [], "x") == draft
    assert dg._citation_hygiene(draft, _Session({MAXIMS: ["ccnl"]}), docs, [], "Il CCNL applicato al rapporto") == draft
    assert dg._citation_hygiene(draft, _Session({MAXIMS: ["special"]}), docs, [], f"art. 563 {MAXIMS}") == draft


# --- GRADER_EVIDENCE ------------------------------------------------------------------------------

def test_evidence_lines_say_what_found_the_article():
    rows = [dict(_row("1218", "c.c.", "x"), _signals=["current", "frame"], _vote=True, _central=True,
                 _vote_stats={"rulings": 4, "similar": 30, "carded": 2, "cards": 9})]
    lines = dg._evidence_lines(rows, {"allowed": {"c.c.", "c.p.c."}})
    assert lines[0] == ("Trovato da: ricerca sull'analisi del caso; ricerca sui quesiti e sulle questioni; "
                        "sentenze su fatti simili; istituto di una questione centrale (più fonti)")
    assert lines[1] == "Sentenze simili: applicato da 4 delle 30 più simili; 2 delle 9 con scheda lo applicano"
    assert lines[2] == "Materia: del caso"
    alone = [dict(_row("2043", "c.c.", "x"), _signals=["baseline"])]
    assert dg._evidence_lines(alone, {"allowed": {"c.p.", "c.p.p."}}) == [
        "Trovato da: ricerca sul testo del caso", "Materia: un'altra rispetto alle aree del caso"]
    assert dg._evidence_lines(alone, {"allowed": None}) == ["Trovato da: ricerca sul testo del caso"]


def test_the_issues_are_numbered_and_a_q_reply_reads_as_the_issue():
    analysis = {"frame": {"tasks": ["Chi è proprietario del fondo?"],
                          "central_issue_candidates": [{"nome": "usucapione"}]},
                "questioni": [{"titolo": "Questione centrale: usucapione"}, {"titolo": "Rivendicazione"}]}
    issues = dg._graded_issues(analysis)
    assert issues == ["Chi è proprietario del fondo?", "usucapione", "Rivendicazione"]
    assert dg._graded_case("Fatti.", "", analysis, issues).endswith(
        "Questioni del caso:\nQ1. Chi è proprietario del fondo?\nQ2. usucapione\nQ3. Rivendicazione")
    read = dg._graded_reply('[{"a": "A1", "g": 3, "q": "Q2", "r": "x"}, {"a": "A2", "g": 0, "q": "", "r": "y"}]',
                            {"A1": "1158 c.c.", "A2": "2043 c.c."}, issues)
    assert read["1158 c.c."]["issue"] == "usucapione" and read["2043 c.c."]["issue"] == ""


def test_the_grader_versions_build_on_the_live_prompt():
    assert dg._graded_system() == dg._GRADED_SYSTEM
    full = dg._graded_system(evidence=True, issues=True, examples=True)
    assert "3 o più sentenze simili e della materia del caso vale almeno 2" in full and "ESEMPI" in full
    assert '"q": "Q2"' in full and "quanto direttamente la disciplina" in full
    assert "ESEMPI" not in dg._graded_system(evidence=True) and "voto di pertinenza" in dg._graded_system(evidence=True)


def test_the_examples_teach_no_held_out_article():
    cases = json.load(open(os.path.join(os.path.dirname(__file__), "eval_narrative_cases.json"), encoding="utf-8"))["cases"]
    held = {(str(a), e["code"]) for c in cases if re.match(r"^(h\d+|n\d+)$", c["id"])
            for e in c.get("expected") or [] for a in (e.get("any_of") or [e["article"]])}
    codes = {"c.p.": "cp", "c.c.": "cc", "c.p.p.": "cpp", "c.p.c.": "cpc", "c.p.a.": "cpa", "l. 241/1990": "l241"}
    cited = re.findall(r"[Aa]rt\. ([\w-]+) (c\.p\.p\.|c\.p\.a\.|c\.p\.c\.|c\.p\.|c\.c\.|l\. 241/1990)", dg._GRADED_EXAMPLES)
    assert len(cited) >= 20 and not {(n, codes[c]) for n, c in cited} & held


def test_with_evidence_the_check_grades_per_issue_without_examples(monkeypatch, logged):
    monkeypatch.setenv("CHECK_GRADED_ENABLED", "true")
    monkeypatch.setenv("GRADER_EVIDENCE_ENABLED", "true")
    model = Grades({})
    monkeypatch.setattr(dg, "_call_chat", model)
    dg._filter_applicable_articles(CASE, "", _pool(), GRADED_ANALYSIS)
    assert "ESEMPI" not in model.calls[0]["system"] and "Q1. Chi è proprietario" in model.humans[0]   # v2
    monkeypatch.setenv("GRADER_EXAMPLES_ENABLED", "true")
    model = Grades({"1158 c.c.": 3})
    monkeypatch.setattr(dg, "_call_chat", model)
    pool = _pool()
    pool[1]["_signals"] = ["current", "baseline"]
    pool[1]["_vote_stats"] = {"rulings": 5, "similar": 30, "carded": 0, "cards": 0}
    kept = {r["_article"][0]: r for r in dg._filter_applicable_articles(CASE, "", pool, GRADED_ANALYSIS)}
    human, system = model.humans[0], model.calls[0]["system"]
    assert "ESEMPI" in system and "Q1. Chi è proprietario del fondo?" in human
    assert ("[A2] 1158 c.c.\nRubrica: -\nTrovato da: ricerca sull'analisi del caso; ricerca sul testo del caso "
            "(più fonti)\nSentenze simili: applicato da 5 delle 30 più simili\n") in human
    assert kept["1158"]["_grade"] == 3


def test_vote_stats_count_the_rulings_applying_each_article():
    stats = vote_pool.vote_stats({"r1": [("1218", "c.c."), ("1218", "c.c.")], "r2": [("1218", "c.c.")], "r3": []},
                                 {"r2"})
    assert stats == {("1218", "c.c."): {"rulings": 2, "similar": 3, "carded": 1, "cards": 1}}


def test_with_evidence_the_plan_forces_eighteen_offered_by_issue(monkeypatch):
    monkeypatch.setenv("CHECK_GRADED_ENABLED", "true")
    rows = [dict(_row(str(n), "c.c.", "x"), _grade=2, _fused_rank=n, _grade_issue="Q" if n % 2 else "R")
            for n in range(1, 25)]
    assert len(ptw._forced(rows)) == 14
    monkeypatch.setenv("GRADER_EVIDENCE_ENABLED", "true")
    forced = ptw._forced(rows)
    assert len(forced) == 18
    groups = ptw._issue_groups(forced, {r["_article"]: r["_grade_issue"] for r in rows})
    assert list(groups) == ["Q", "R"] and len(groups["Q"]) == 9
    reasons = {("584", "c.p."): "central issue", ("581", "c.p."): "cross-reference of 584 c.p."}
    assert ptw._issue_groups(reasons, {("584", "c.p."): "omicidio"}) == {"omicidio": [("584", "c.p."), ("581", "c.p.")]}
    human = ptw._plan_human("Fatti.", "", {}, "NORME", "", "", [], applicable=list(forced), must_place="graded",
                            groups=groups)
    assert "NORME PIÙ PERTINENTI AL CASO, per la questione che disciplinano:\n- Q: [art. 1 c.c.], [art. 3 c.c.]" in human


def test_an_article_about_the_inverse_situation_goes_even_when_protected(monkeypatch, logged):
    """WRITER_RULES_ENABLED: the qualification check reads the article's opening text and rejects one about the
    inverse situation ("tipo": "direzione"), removed even when graded 2, kept only when the case names it."""
    monkeypatch.setenv("WRITER_RULES_ENABLED", "true")
    rows = [dict(_row("2049", "c.c.", "I padroni e i committenti sono responsabili per i danni arrecati dal fatto "
                      "illecito dei loro domestici e commessi...", "Responsabilità dei padroni e dei committenti"), _grade=2),
            dict(_row("1588", "c.c.", "Il conduttore risponde della perdita e del deterioramento della cosa...",
                      "Perdita e deterioramento della cosa locata"))]
    draft = ("Il lavoratore ferito può chiedere il risarcimento ex art. 2049 c.c. per il danno subito. Il locatore "
             "risponde dei vizi ex art. 1588 c.c.")
    model = Qualification({"2049 c.c.": "direzione", "1588 c.c.": "direzione"}, rewrite="[]")
    monkeypatch.setattr(dg, "_call_chat", model)
    out = dg._check_qualifications(draft, {}, rows, None, [], "Il cliente chiede dell'art. 1588 c.c.")
    assert "2049" not in out                                                  # graded 2, still removed
    assert "art. 1588 c.c. [DA VERIFICARE" in out                             # named in the case: only marked
    assert "situazione inversa" in model.systems[0] and "Testo: I padroni e i committenti" in model.humans[0]
    log = [d for e, d in logged if e == "case_analysis_qualification"][-1]
    assert log["inverse"] == ["2049 c.c."] and log["removed"] == ["2049 c.c."]
