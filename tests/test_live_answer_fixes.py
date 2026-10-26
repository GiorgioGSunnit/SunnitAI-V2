"""The live answer of 26 Oct (a long narrative on a locked road), six general defects, under the existing flags:
issue order for a right already acquired (REVIEW_FIXES), fact-rule ranges mandatory and generic rubrica words kept
in their chapter (FACT_SIGNALS / STRICT_ARTICLES), adverse rulings only as the opponent's argument (RULING_FIT),
whole-sentence citation removal and dangling fragments (FLAGGED_REMOVE, citation discipline, writer rules), every
explicit question answered and art. 614-bis c.p.c. for sanctions (REVIEW_FIXES, FACT_REGIMES), duplicate issues
merged and repeated evidence lists collapsed (REVIEW_FIXES).
Invented cases only: never the boat/pool texts, h1-h11, n1-n9, nor the live narrative itself."""

import json
import os

import pytest

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

import src.rag.defensive_generation as dg  # noqa: E402
import src.rag.fact_signals as fs  # noqa: E402
import src.rag.plan_then_write as ptw  # noqa: E402
from src.rag import ruling_cards  # noqa: E402


@pytest.fixture
def logged(monkeypatch):
    out = []
    monkeypatch.setattr(dg, "vlog", lambda event, data=None, *a, **k: out.append((event, data)))
    return out


def _issue(title, norme, argomento="Argomento.", pronunce=None):
    return {"titolo": title, "norme": norme, "pronunce": list(pronunce or []), "argomento": argomento,
            "obiezione": "", "replica": ""}


def _plan(issues, **extra):
    return {"parte_assistita": "Ottavia", "posizione": "proprietaria del fondo dominante", "obiettivo": "",
            "questione_centrale": extra.pop("centrale", "Se Ottavia può continuare a passare sulla carrareccia."),
            "risposte": extra.pop("risposte", []), "questioni": issues, "subordinate": extra.pop("subordinate", []),
            "punti_deboli": [], **extra}


ACQUIRED_ANALYSIS = {"parte_assistita": "Ottavia", "posizione": "proprietaria del fondo dominante",
                     "fact_signals": {"fired": [{"id": "destinazione_padre_famiglia", "acquired_right": True,
                                                 "institutions": ["servitù per destinazione del padre di famiglia"],
                                                 "why": "due fondi di un unico proprietario, separati con un'opera"}]}}


# --- 1. a right already acquired is argued first ----------------------------------------------------------------

def test_recognition_first_and_the_constitutive_remedy_as_subordinate():
    plan = _plan([_issue("Passaggio coattivo", [("1051", "c.c.")]),
                  _issue("Servitù della carrareccia", [("1062", "c.c.")]),
                  _issue("Danni per il blocco", [("2043", "c.c.")]),
                  _issue("Azione confessoria", [("1079", "c.c.")])])
    out, log = ptw.order_for_acquired(plan, ptw.acquired_rights(ACQUIRED_ANALYSIS))
    assert [q["titolo"] for q in out["questioni"]] == ["Servitù della carrareccia", "Azione confessoria",
                                                      "Danni per il blocco", "In via subordinata: Passaggio coattivo"]
    assert out["subordinate"] == ["Passaggio coattivo: solo se il diritto già acquisito non fosse riconosciuto"]
    assert "destinazione del padre di famiglia" in out["diritto_acquisito"]
    assert log["before"][0] == "Passaggio coattivo"
    text = ptw._plan_text(out)
    assert "Diritto già acquisito: servitù per destinazione del padre di famiglia" in text
    assert "il rimedio che costituisce il diritto ex novo solo in via subordinata" in text


def test_no_reordering_without_an_acquired_right_or_a_constitutive_issue():
    plan = _plan([_issue("Passaggio coattivo", [("1051", "c.c.")]), _issue("Azione confessoria", [("1079", "c.c.")])])
    assert ptw.order_for_acquired(plan, []) == (plan, {})
    only_recognition = _plan([_issue("Usucapione del sentiero", [("1158", "c.c.")])])
    assert ptw.order_for_acquired(only_recognition, ["usucapione"]) == (only_recognition, {})


def test_acquired_rights_from_the_analysis_too_but_never_a_constitutive_name():
    analysis = {"questioni": [{"titolo": "Acquisto della servitù per usucapione",
                               "istituti": [{"nome": "passaggio coattivo"}, {"nome": "usucapione"}]}]}
    assert ptw.acquired_rights(analysis) == ["Acquisto della servitù per usucapione", "usucapione"]


def test_plan_problems_name_the_missing_recognition_and_a_constitutive_central_question():
    plan = _plan([_issue("Passaggio coattivo", [("1051", "c.c.")])],
                 centrale="Se Ottavia ha diritto al passaggio coattivo.")
    problems = ptw._plan_problems(plan, ACQUIRED_ANALYSIS, [])
    assert problems[0].startswith("manca la questione del riconoscimento")
    assert problems[1].startswith("la questione centrale è un rimedio costitutivo")
    assert ptw._plan_problems(plan, {}, []) == []                  # no acquired right: nothing to fix


class PlanModel:
    """The plan (one reply per call), the stance review, the plan check."""

    def __init__(self, plans):
        self.plans, self.humans = list(plans), []

    def __call__(self, messages, max_tokens=None, temperature=None, **kwargs):
        system, human = messages[0].content, messages[-1].content
        if "PIANO, nell'interesse" in system:
            self.humans.append((system, human))
            return json.dumps(self.plans.pop(0) if len(self.plans) > 1 else self.plans[0], ensure_ascii=False)
        if "rivede la bozza" in system:
            return json.dumps({"parte_assistita": "Ottavia", "argomenti_contrari": []})
        if "controlli il piano" in system:
            return "[]"
        raise AssertionError(system[:80])


ROWS = [{"d": {"id": "cc", "name": "Codice Civile 2026"}, "_article": (n, "c.c."),
         "s": {"name": f"{n}.0.0", "plain_text": f"Testo dell'art. {n}.", "titolo_articolo": h}}
        for n, h in (("1051", "Passaggio coattivo"), ("1062", "Destinazione del padre di famiglia"),
                     ("1079", "Accertamento della servitù"))] + [
    {"d": {"id": "cpc", "name": "Codice di procedura civile 2026"}, "_article": ("614-bis", "c.p.c."),
     "s": {"name": "614-bis.0.0", "plain_text": "Testo.", "titolo_articolo": "Misure di coercizione indiretta"}}]


def _raw_plan(issues, risposte=None, central="Se Ottavia può passare."):
    return {"parte_assistita": "Ottavia", "posizione": "", "obiettivo": "", "questione_centrale": central,
            "risposte": risposte or [], "subordinate": [], "punti_deboli": [],
            "questioni": [{"titolo": t, "norme": n, "pronunce": [], "argomento": "Arg.", "obiezione": "", "replica": ""}
                          for t, n in issues]}


def test_the_plan_is_made_again_once_when_it_misses_the_recognition_and_a_question(monkeypatch, logged):
    monkeypatch.setenv("REVIEW_FIXES_ENABLED", "true")
    request = ("Assisti Ottavia. Può chiedere che il vicino paghi una somma per ogni giorno in cui tiene chiusa la "
               "sbarra?")
    first = _raw_plan([("Passaggio coattivo", ["art. 1051 c.c."]), ("Diritto di passaggio", ["art. 1051 c.c."])])
    second = _raw_plan([("Passaggio coattivo", ["art. 1051 c.c."]), ("Servitù per destinazione", ["art. 1062 c.c."]),
                        ("Somma per ogni giorno di chiusura della sbarra", ["art. 614-bis c.p.c."])],
                       risposte=[{"quesito": "Somma per ogni giorno di chiusura?",
                                  "risposta": "Sì: la misura coercitiva di una somma per ogni giorno in cui il vicino "
                                              "tiene chiusa la sbarra."}])
    model = PlanModel([first, second])
    monkeypatch.setattr(dg, "_call_chat", model)
    plan = ptw.make_plan("Ottavia passa da sempre sulla carrareccia.", request, ACQUIRED_ANALYSIS, "NORME", "", "",
                         ROWS, {})
    system, human = model.humans[0]
    assert ptw._ACQUIRED_RULE in system and ptw._QUESTIONS_RULE in system
    assert ptw._QUESTIONS_TITLE + ":\n1. Può chiedere che il vicino paghi" in human
    assert ptw._ACQUIRED_TITLE + ":\n- servitù per destinazione del padre di famiglia" in human
    retry = model.humans[1][1]
    assert "manca la questione del riconoscimento" in retry and "non ha la sua risposta" in retry
    assert [q["titolo"] for q in plan["questioni"]] == ["Servitù per destinazione",
                                                       "Somma per ogni giorno di chiusura della sbarra",
                                                       "In via subordinata: Passaggio coattivo"]
    assert plan["questioni"][1]["norme"] == [("614-bis", "c.p.c.")]
    fixes = [d for e, d in logged if e == "case_analysis_plan"][-1]["review_fixes"]
    assert fixes["unanswered"] == [] and fixes["order"]["after"][-1] == "In via subordinata: Passaggio coattivo"


def test_flag_off_no_rule_no_retry(monkeypatch, logged):
    first = _raw_plan([("Passaggio coattivo", ["art. 1051 c.c."])])
    model = PlanModel([first])
    monkeypatch.setattr(dg, "_call_chat", model)
    plan = ptw.make_plan("Fatti.", "Si può chiedere una somma per ogni giorno?", ACQUIRED_ANALYSIS, "NORME", "", "",
                         ROWS, {})
    assert len(model.humans) == 1 and ptw._ACQUIRED_RULE not in model.humans[0][0]
    assert [q["titolo"] for q in plan["questioni"]] == ["Passaggio coattivo"]


# --- 2. fact-rule ranges are mandatory; a generic rubrica word stays in its chapter -------------------------------

def test_every_rule_has_ranges_and_one_without_is_refused(tmp_path):
    assert all(rule["ranges"] for rule in fs.load_rules())
    bad = tmp_path / "bad.yaml"
    bad.write_text("- {id: x, area: civile, when: [{fact: {pattern: 'a'}}], institution: usucapione, codice: c.c., "
                   "why: y, status: experiment}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no ranges"):
        fs.load_rules(str(bad))


def test_usucapione_rules_resolve_only_in_their_chapter():
    rules = {r["id"]: r for r in fs.load_rules()}
    assert rules["usucapione_servitu_apparente"]["ranges"] == [[1158, 1167], [1061, 1061]]
    assert all(rules[i].get("acquired_right") for i in ("destinazione_padre_famiglia", "usucapione_servitu_apparente",
                                                         "usucapione_ordinaria"))
    row = {"_article": ("2689", "c.c."), "s": {"titolo_articolo": "Usucapione"}}
    istituto = {"nome": "usucapione di servitù apparente", "fonte": "regola usucapione_servitu_apparente",
                "ranges": rules["usucapione_servitu_apparente"]["ranges"]}
    assert fs.keep_resolved(istituto, [row]) == []
    assert fs.keep_resolved({**istituto, "ranges": []}, [row]) == []      # a rule without ranges resolves nowhere


@pytest.mark.parametrize("name,number,rubrica,out", [
    ("usucapione di servitù apparente", "2689", "Usucapione", "usuca"),
    ("usucapione di servitù apparente", "1158", "Usucapione dei beni immobili", ""),
    ("usucapione di servitù apparente", "1061", "Servitù non apparenti", ""),
    ("prescrizione del credito", "1073", "Estinzione per prescrizione", "presc"),
    ("estinzione della servitù per prescrizione", "1073", "Estinzione per prescrizione", ""),
    ("prescrizione del credito", "2946", "Prescrizione ordinaria", ""),
    ("accessioni", "1010", "Accessioni e miglioramenti", "acces"),
])
def test_generic_rubrica_words_outside_their_chapter(name, number, rubrica, out):
    assert fs.generic_out_of_range(name, "c.c.", number, rubrica) == out
    assert fs.generic_out_of_range(name, "c.p.", number, rubrica) == ""       # only the codes in the table


def test_resolution_skips_the_transcription_article(monkeypatch, logged):
    from src.rag import issue_spotting
    headings = {"2689": "Usucapione", "1158": "Usucapione dei beni immobili"}
    monkeypatch.setattr(issue_spotting, "article_by_title",
                        lambda session, doc, term, field="titolo_articolo": "2689" if "usucapione" in term else None)
    monkeypatch.setattr(issue_spotting, "fetch_article", lambda session, doc, n: [
        {"d": {"id": "cc", "name": "Codice Civile 2026"}, "s": {"name": f"{n}.0.0", "titolo_articolo": headings[n],
                                                                 "plain_text": "x"}}] if n in headings else [])
    monkeypatch.setattr(dg, "_issue_article_refs", lambda session, doc, terms: ["2689", "1158"])
    documents = [{"id": "cc", "name": "Codice Civile 2026"}]
    istituto = {"nome": "usucapione di servitù apparente", "codice": "c.c.", "numero_indicativo": "2689"}
    assert [r["_article"] for r in dg._resolve_institution(None, documents, istituto)] == [("2689", "c.c.")]
    monkeypatch.setenv("FACT_SIGNALS_ENABLED", "true")
    assert [r["_article"] for r in dg._resolve_institution(None, documents, istituto)] == [("1158", "c.c.")]
    rejected = [d for e, d in logged if e == "case_analysis_institution_rejected"]
    assert rejected and all(d["article"] == "2689 c.c." and "outside its chapter" in d["why"] for d in rejected)


# --- 3. an adverse ruling is only the opponent's argument --------------------------------------------------------

def _ruling(number, passage, **extra):
    return {"d": {"id": f"r{number}", "name": f"[CIVILE] Ordinanza n. {number}"}, "s": {"plain_text": passage},
            "_passage": passage, **extra}


def test_ruling_fit_marks_the_adverse_ruling_from_the_check_or_the_card(monkeypatch, logged):
    monkeypatch.setenv("RULING_FIT_ENABLED", "true")
    rows = [_ruling("1", "Il fondo non è intercluso se ha un accesso, anche scomodo, alla via pubblica."),
            _ruling("2", "Il titolare della servitù può farne accertare l'esistenza."),
            _ruling("3", "Passo con scheda.", _card_role="sfavorevole", _card_esito="rigetto")]
    asked = []

    def chat_json(messages, max_tokens=None, temperature=None):
        asked.append((messages[0].content, messages[-1].content))
        return [{"id": "S1", "uso": "regola", "verso": "contro", "perche": "nega l'interclusione"},
                {"id": "S2", "uso": "regola", "verso": "a favore"},
                {"id": "S3", "uso": "regola", "verso": "a favore"}]
    monkeypatch.setattr(dg, "_chat_json", chat_json)
    kept = dg._ruling_fit("Fatti.", "Richiesta.", rows, {"parte_assistita": "Ottavia", "posizione": "attrice"})
    assert [bool(r.get("_fit_adverse")) for r in kept] == [True, False, True]       # the card decides for S3
    assert '"verso"' in asked[0][0] and "POSIZIONE DELLA PARTE ASSISTITA: Ottavia (attrice)" in asked[0][1]
    log = [d for e, d in logged if e == "case_analysis_ruling_fit"][-1]
    assert log["adverse"] == [log["kept"][0]["ruling"], log["kept"][2]["ruling"]] and len(set(log["adverse"])) == 2
    assert [e.get("adverse_by") for e in log["kept"]] == ["check", None, "card"]
    noted = dg._with_fit_facts(kept)
    assert noted[0]["_passage"].startswith("ORIENTAMENTO SFAVOREVOLE") and "Uso nel parere" not in noted[0]["_passage"]
    assert noted[1]["_passage"].startswith("Uso nel parere: regola generale")


def test_card_rows_carry_their_role_outcome_and_party():
    card = {"id": "x", "_role": "sfavorevole", "esito": "rigetto", "parte_favorita": "il proprietario del fondo",
            "a_favore_di": "", "ricorrente": "", "citazione": ""}
    row = ruling_cards.annotate([{"d": {"id": "x"}, "_passage": "P."}], {"x": card})[0]
    assert (row["_card_role"], row["_card_esito"], row["_card_favoured"]) == ("sfavorevole", "rigetto",
                                                                              "il proprietario del fondo")


def test_adverse_rulings_leave_the_support_and_a_supporting_sentence_goes():
    references = {"Cass. civ., sez. II, ord. n. 5555/2025": [{"_fit_adverse": True}],
                  "Cass. civ., sez. II, ord. n. 6666/2025": [{}]}
    plan = _plan([_issue("Interclusione", [("1051", "c.c.")], pronunce=list(references))])
    out, moved = ptw.split_adverse(plan, references)
    assert out["questioni"][0]["pronunce"] == ["Cass. civ., sez. II, ord. n. 6666/2025"]
    assert out["questioni"][0]["pronunce_contrarie"] == ["Cass. civ., sez. II, ord. n. 5555/2025"] == moved
    assert "pronunce della controparte, solo nell'obiezione con la replica: Cass. civ., sez. II, ord. n. 5555/2025" \
        in ptw._plan_text(out)
    text = ("Il fondo di Ottavia è intercluso, come conferma Cass. n. 5555/2025. La controparte potrebbe invocare "
            "Cass. n. 5555/2025, ma lì il fondo aveva un accesso. Vale anche Cass. n. 6666/2025.")
    kept, log = ptw.adverse_as_support("Norme e giurisprudenza", text, out["pronunce_contrarie"])
    assert kept == ("La controparte potrebbe invocare Cass. n. 5555/2025, ma lì il fondo aveva un accesso. Vale anche "
                    "Cass. n. 6666/2025.")
    assert log[0]["ruling"] == ["5555/2025"] and log[0]["kind"] == "adverse_as_support"


# --- 4. a removed citation takes its whole sentence, and no fragment is left -------------------------------------

def test_removal_drops_the_sentence_and_the_back_reference_left_alone():
    text = ("#### Usucapione della servitù\n"
            "L'acquisto è trascritto ai sensi dell'art. 2689 c.c. Questa disposizione è pertinente al caso di Ottavia.\n"
            "\n#### Passaggio\nL'art. 1051 c.c. dà il passaggio al fondo intercluso. Questa disposizione è pertinente.")
    out, log = ptw.remove_citations(text, [("2689", "c.c.")], [], {("2689", "c.c."), ("1051", "c.c.")})
    assert out == ("#### Usucapione della servitù\n\n#### Passaggio\nL'art. 1051 c.c. dà il passaggio al fondo "
                   "intercluso. Questa disposizione è pertinente.")
    assert [e["how"] for e in log] == ["sentence dropped", "sentence dropped"]
    assert log[1]["kind"] == "dangling"


@pytest.mark.parametrize("text,out", [
    ("Ottavia ha diritto al passaggio. Inoltre, conferma che il fondo è intercluso.", "Ottavia ha diritto al passaggio."),
    ("Ottavia può chiedere i danni subiti, in conformità.", "Ottavia può chiedere i danni subiti."),
    ("Il danno va provato, come previsto dall'.", "Il danno va provato."),
    ("Tale principio vale anche qui.", ""),
    ("Secondo Cass. n. 7/2024 il passaggio va concesso. Tale principio vale anche qui.",
     "Secondo Cass. n. 7/2024 il passaggio va concesso. Tale principio vale anche qui."),
    ("L'art. 1079 c.c. tutela il titolare.\n\nQuesta norma consente la rimozione della sbarra.",
     "L'art. 1079 c.c. tutela il titolare.\n\nQuesta norma consente la rimozione della sbarra."),
    ("Ottavia passa ai sensi dell'art. 1062 c.c.", "Ottavia passa ai sensi dell'art. 1062 c.c."),
])
def test_drop_dangling(text, out):
    assert ptw.drop_dangling(text, "x")[0] == out


def test_section_checks_drop_fragments_with_any_citation_check(monkeypatch, logged):
    sections = [(t, "Testo.") for t in ptw.SECTION_TITLES]
    sections[5] = (ptw.SECTION_TITLES[5], "Ottavia vince. Inoltre, conferma che la sbarra va tolta.")
    plan = _plan([_issue("x", [])])
    assert ptw._after_checks(plan, sections, [], "", [])[5][1] == "Ottavia vince. Inoltre, conferma che la sbarra va tolta."
    monkeypatch.setenv("FLAGGED_REMOVE_ENABLED", "true")
    assert ptw._after_checks(plan, sections, [], "", [])[5][1] == "Ottavia vince."
    assert [d for e, d in logged if e == "case_analysis_dangling"][-1]["count"] == 1


# --- 5. every explicit question answered; sanctions get art. 614-bis c.p.c. ---------------------------------------

def test_explicit_questions_and_their_answers():
    request = ("Difendi Bianca. Può ottenere il risarcimento per i raccolti persi? E una sanzione per ogni giorno di "
               "ritardo nel togliere il cancello?")
    questions = ptw.explicit_questions(request)
    assert questions == ["Può ottenere il risarcimento per i raccolti persi?",
                         "E una sanzione per ogni giorno di ritardo nel togliere il cancello?"]
    assert ptw._answers(questions[0], "Il risarcimento dei raccolti persi spetta a Bianca.")
    assert not ptw._answers(questions[1], "Per ogni giorno di ritardo nel togliere il cancello si chiede il danno.")
    assert ptw._answers(questions[1], "Per ogni giorno di ritardo nel togliere il cancello: art. 614-bis c.p.c.")
    assert ptw.explicit_questions("Prepara un parere.", "Bianca chiede: posso passare dal fondo vicino?") == [
        "Bianca chiede: posso passare dal fondo vicino?"]


def test_a_sanction_request_brings_the_coercive_measures():
    facts = {"requests": ["una penalità per ogni giorno in cui il cancello resta chiuso"], "events": []}
    tools = fs.procedural_tools(facts, ["diritti_reali"])
    assert [t["id"] for t in tools] == ["misure_coercitive"] and tools[0]["refs"] == [["614-bis", "c.p.c."]]
    assert [t["id"] for t in fs.procedural_tools({"requests": []}, [], "Si può chiedere una sanzione al vicino?")] \
        == ["misure_coercitive"]
    assert fs.procedural_tools({"requests": ["il risarcimento"]}, []) == []
    assert fs.procedural_tools(facts, ["penale"]) == []


def test_the_answers_section_is_written_again_for_an_unanswered_question(monkeypatch, logged):
    monkeypatch.setenv("REVIEW_FIXES_ENABLED", "true")
    calls = []

    def chat(messages, max_tokens=None, **kwargs):
        calls.append(messages[0].content)
        return "Il risarcimento dei raccolti persi spetta. La sanzione per ogni giorno di ritardo è l'art. 614-bis c.p.c."
    monkeypatch.setattr(dg, "_call_chat", chat)
    questions = ["Può ottenere il risarcimento per i raccolti persi?",
                 "E una sanzione per ogni giorno di ritardo nel togliere il cancello?"]
    sections = [(t, "Testo.") for t in ptw.SECTION_TITLES]
    sections[0] = (ptw.SECTION_TITLES[0], "Il risarcimento dei raccolti persi spetta a Bianca.")
    jobs = [(t, "SEZIONE", "MATERIALE", 600) for t in ptw.SECTION_TITLES]
    out = ptw._after_checks(_plan([_issue("x", [])]), sections, jobs, "SISTEMA", questions)
    assert "614-bis" in out[0][1] and "non rispondeva a: E una sanzione" in calls[0]
    assert [d for e, d in logged if e == "case_analysis_questions"][-1]["rewritten"] is True


def test_section_one_gets_the_questions_and_the_tools(monkeypatch):
    monkeypatch.setenv("REVIEW_FIXES_ENABLED", "true")
    monkeypatch.setenv("FACT_REGIMES_ENABLED", "true")
    analysis = {"parte_assistita": "Bianca", "fact_signals": {"tools": [
        {"id": "misure_coercitive", "refs": [["614-bis", "c.p.c."]], "text": "Misure (art. 614-bis c.p.c.).",
         "why": "w", "matched": ["m"]}]}, "frame": {}}
    plan = ptw.clean_plan(_raw_plan([("Accertamento della servitù", ["art. 1079 c.c."])]), {("1079", "c.c.")}, {})[0]
    jobs = ptw.section_jobs(plan, "Fatti.", "Si può chiedere una sanzione per ogni giorno di ritardo?", analysis,
                            ROWS, {}, "")
    title, instruction, material, _ = jobs[0]
    assert "Ai QUESITI ESPLICITI rispondi uno per uno" in instruction
    assert "QUESITI ESPLICITI:\n1. Si può chiedere una sanzione" in material and "614-bis" in material


# --- 6. duplicate issues merged; a repeated evidence list collapsed ----------------------------------------------

def test_duplicate_issues_merge_by_governing_article_or_near_title():
    plan = _plan([_issue("Passaggio coattivo", [("1051", "c.c.")], "Primo argomento.", ["R1"]),
                  _issue("Diritto di passaggio", [("1051", "c.c."), ("1052", "c.c.")], "Doppione.", ["R2"]),
                  _issue("Accertamento della servitù", [("1079", "c.c.")]),
                  _issue("Servitù: accertamento", [("1062", "c.c.")]),
                  _issue("Danni", [("2043", "c.c.")])])
    out, merged = ptw.merge_duplicate_issues(plan)
    assert [q["titolo"] for q in out["questioni"]] == ["Passaggio coattivo", "Accertamento della servitù", "Danni"]
    first = out["questioni"][0]
    assert first["norme"] == [("1051", "c.c."), ("1052", "c.c.")] and first["pronunce"] == ["R1", "R2"]
    assert first["argomento"] == "Primo argomento."
    assert merged == [{"into": "Passaggio coattivo", "merged": "Diritto di passaggio"},
                      {"into": "Accertamento della servitù", "merged": "Servitù: accertamento"}]
    assert plan["questioni"][0]["norme"] == [("1051", "c.c.")]                     # the input untouched


def test_an_evidence_list_identical_to_an_earlier_one_is_not_repeated():
    text = ("#### Servitù\nArgomento.\n\nProve da acquisire:\n- Foto del cancello.\n- Testimoni dei vicini.\n\n"
            "#### Confessoria\nArgomento.\n\n**Prove da acquisire:**\n- testimoni dei vicini\n- foto del cancello\n\n"
            "Obiezione della controparte: nessuna.\n\n#### Danni\nProve da acquisire:\n- Fatture.")
    out, log = ptw.dedupe_evidence(text)
    assert out == ("#### Servitù\nArgomento.\n\nProve da acquisire:\n- Foto del cancello.\n- Testimoni dei vicini.\n\n"
                   "#### Confessoria\nArgomento.\n\nProve da acquisire: le stesse indicate per «Servitù».\n\n"
                   "Obiezione della controparte: nessuna.\n\n#### Danni\nProve da acquisire:\n- Fatture.")
    assert log[0]["under"] == "Confessoria" and log[0]["same_as"] == "Servitù"
    assert ptw.dedupe_evidence("Prove da acquisire:\n- A.\n\nProve da acquisire:\n- B.") == (
        "Prove da acquisire:\n- A.\n\nProve da acquisire:\n- B.", [])
