"""PLAN_THEN_WRITE_ENABLED: the parere from a plan, section by section (src/rag/plan_then_write.py).
Synthetic cases only (Flavio's handover, 22 Oct: never copy the held-out texts)."""

import json
import os
import re
import threading

import pytest

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

import src.rag.defensive_generation as dg  # noqa: E402
import src.rag.plan_then_write as ptw  # noqa: E402
from src.rag import method_candidates, practical_checklist  # noqa: E402


def _article(number, code, text, heading=""):
    return {"d": {"id": f"LEGAL_DOC::{code}", "name": "Codice Civile 2026"}, "_article": (number, code),
            "s": {"name": f"{number}.0.0", "plain_text": text, "titolo_articolo": heading}}


def _ruling(file_name, passage, name="[CIVILE] Ordinanza sul ricorso iscritto al n. 4321/2024 R.G."):
    return {"d": {"id": file_name, "name": name, "file_name": file_name}, "s": {"plain_text": passage},
            "_passage": passage}


ARTICLES = [_article("2048", "c.c.", "I precettori sono responsabili del danno cagionato dal fatto illecito dei "
                                     "loro allievi nel tempo in cui sono sotto la loro vigilanza."),
            _article("1218", "c.c.", "Il debitore che non esegue esattamente la prestazione dovuta è tenuto al "
                                     "risarcimento del danno, se non prova che l'inadempimento è stato determinato "
                                     "da impossibilità della prestazione derivante da causa a lui non imputabile.",
                     "Responsabilità del debitore"),
            _article("2051", "c.c.", "Ciascuno è responsabile del danno cagionato dalle cose che ha in custodia.",
                     "Danno cagionato da cosa in custodia"),
            _article("1917", "c.c.", "TESTO-NON-NEL-PIANO sull'assicurazione della responsabilità civile.")]
RULINGS = [_ruling("snciv_s30_a2026_n00190_tO.clean.pdf",
                   "La scuola risponde a titolo contrattuale delle lesioni che l'allievo si procura da solo."),
           _ruling("snpen@s50@a2026@n01234@tS.clean.pdf", "Passo penale non pertinente.",
                   name="[PENALE] Sentenza sul ricorso proposto da X")]
ANALYSIS = {"parte_assistita": "i genitori di Tizio", "posizione": "danneggiati", "area": "civile",
            "fatti": [{"fatto": "Tizio scivola in palestra e urta una panchina", "effetto": "favorevole"}],
            "questioni": [{"titolo": "Responsabilità della scuola"}],
            "frame": {"tasks": ["Prepara un parere per i genitori di Tizio."], "areas": ["responsabilita_civile"],
                      "posture_type": "precontenzioso", "procedural_posture": "Nessun giudizio ancora avviato."}}
FACTS = "Tizio, alunno, scivola durante l'ora di educazione fisica e urta una panchina di legno."
REQUEST = "Prepara un parere per i genitori di Tizio."

PLAN = {"parte_assistita": "i genitori di Tizio", "posizione": "danneggiati", "obiettivo": "il risarcimento",
        "questione_centrale": "Se la scuola risponde delle lesioni che Tizio si è procurato da solo.",
        "risposte": [{"quesito": REQUEST, "risposta": "Sì, a titolo contrattuale."}],
        "questioni": [{"titolo": "Responsabilità contrattuale della scuola",
                       "norme": ["art. 1218 c.c.", "art. 2052 c.c."],
                       "pronunce": ["Cass. civ., sez. III, ord. n. 190/2026", "Cass. civ., n. 9999/2020"],
                       "argomento": "La scuola deve provare che la caduta era imprevedibile.",
                       "obiezione": "L'insegnante aveva istruito gli alunni.", "replica": "Non basta."},
                      {"titolo": "Custodia della panchina", "norme": ["art. 2051 c.c."], "pronunce": [],
                       "argomento": "La panchina era nello spazio di gioco.", "obiezione": "", "replica": ""},
                      {"titolo": "Senza argomento", "norme": ["art. 1218 c.c."], "argomento": ""}],
        "subordinate": ["In subordine, il concorso ex art. 1227 c.c."], "punti_deboli": ["Il gioco era sorvegliato."]}


class FakeModel:
    """Answers the plan, the stance check and each section; records every call."""

    def __init__(self, plans=None, opposing=None, failing=None):
        self.plans = list(plans or [PLAN])
        self.opposing = list(opposing or [])
        self.failing = dict(failing or {})       # section title -> failures left
        self.check = {}                           # "1218 c.c." -> verdict fields for the plan check
        self.check_fails = False
        self.calls, self.lock = [], threading.Lock()

    def __call__(self, messages, max_tokens=None, temperature=None, **kwargs):
        system, human = messages[0].content, messages[-1].content
        with self.lock:
            self.calls.append({"system": system, "human": human, "max_tokens": max_tokens,
                               "temperature": temperature})
        if "PIANO, nell'interesse" in system:
            plan = self.plans.pop(0) if len(self.plans) > 1 else self.plans[0]
            return plan if isinstance(plan, str) else json.dumps(plan, ensure_ascii=False)
        if "rivede la bozza" in system:
            flagged = self.opposing.pop(0) if self.opposing else []
            return json.dumps({"parte_assistita": "i genitori di Tizio", "argomenti_contrari": flagged})
        if "controlli il piano" in system:
            if self.check_fails:
                raise RuntimeError("upstream timeout")
            labels = re.findall(r"^\[art\. ([^\]]+)\]", human, re.MULTILINE)
            return json.dumps([{"articolo": f"art. {label}", "pertinente": True, "contraddice": False,
                                "perche": "x", **self.check.get(label, {})} for label in labels])
        title = next(t for t in ptw.SECTION_TITLES if f"SEZIONE: {t}" in system)
        with self.lock:
            if self.failing.get(title):
                self.failing[title] -= 1
                raise RuntimeError("upstream timeout")
        if title == "Conclusioni":
            return f"**6. {title}**\nTesto della sezione {title}."           # a repeated heading
        return f"Testo della sezione {title}."

    def section_call(self, title):
        return next(c for c in self.calls if f"SEZIONE: {title}" in c["system"])


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setenv("PLAN_THEN_WRITE_ENABLED", "true")
    logged = []
    monkeypatch.setattr(dg, "vlog", lambda event, data=None, *a, **k: logged.append((event, data)))
    fake = FakeModel()
    monkeypatch.setattr(dg, "_call_chat", fake)
    fake.logged = logged
    return fake


def _write(articles="NORME", rulings="GIURISPRUDENZA"):
    return ptw.write_parere(FACTS, REQUEST, ANALYSIS, ARTICLES, RULINGS, articles, rulings, "", "it")


# --- the full reference ------------------------------------------------------------------------

def test_ruling_reference_from_italgiure_file_names():
    def ref(name):
        return dg._ruling_reference({"d": {"file_name": name}})
    assert ref("snpen@s50@a2026@n00190@tS.clean.pdf") == "Cass. pen., sez. V, sent. n. 190/2026"
    assert ref("snciv_s30_a2026_n12601_tO.clean.pdf") == "Cass. civ., sez. III, ord. n. 12601/2026"
    assert ref("snciv_sU0_a2026_n00007_tS") == "Cass. civ., Sez. U., sent. n. 7/2026"
    assert ref("snciv_s61_a2026_n00042_tO") == "Cass. civ., sez. VI-1, ord. n. 42/2026"
    assert ref("snciv_sL0_a2026_n00042_tI") == "Cass. civ., sez. L., ord. interl. n. 42/2026"
    assert ref("sentenza_tribunale_milano.pdf") == ""
    assert dg._ruling_reference({"d": {"name": "[PENALE] snpen@s10@a2026@n00005@tS"}}).endswith("n. 5/2026")


def test_ruling_label_is_the_full_reference_only_with_the_flag(monkeypatch):
    monkeypatch.delenv("PLAN_THEN_WRITE_ENABLED", raising=False)
    assert dg._ruling_label(RULINGS[0]) == "Ordinanza sul ricorso iscritto al n. 4321/2024 R.G."
    monkeypatch.setenv("PLAN_THEN_WRITE_ENABLED", "true")
    assert dg._ruling_label(RULINGS[0]) == "Cass. civ., sez. III, ord. n. 190/2026"
    assert dg._ruling_label({"d": {"name": "[CIVILE] Tribunale di Roma"}}) == "Tribunale di Roma"
    draft = "Come affermato da Cass. civ., sez. III, ord. n. 190/2026, la scuola risponde."
    assert dg._ruling_is_cited(dg._ruling_label(RULINGS[0]), draft.lower(), dg._decision_numbers(draft))


# --- the plan ------------------------------------------------------------------------------------

def test_the_plan_keeps_only_given_articles_and_rulings():
    known = {r["_article"] for r in ARTICLES}
    references = {"Cass. civ., sez. III, ord. n. 190/2026": [RULINGS[0]]}
    plan, removed = ptw.clean_plan(PLAN, known, references)
    first, second = plan["questioni"]                                  # the issue with no argument dropped
    assert first["norme"] == [("1218", "c.c.")] and first["pronunce"] == ["Cass. civ., sez. III, ord. n. 190/2026"]
    assert second["norme"] == [("2051", "c.c.")]
    assert removed == {"norme": ["2052 c.c."], "pronunce": ["Cass. civ., n. 9999/2020"]}
    assert ptw.clean_plan({"questioni": [{"titolo": "x", "argomento": ""}]}, known, references)[0] is None
    assert ptw.clean_plan("not a plan", known, references)[0] is None


def test_the_parere_is_written_from_the_plan_in_six_sections(model):
    draft = _write()
    headings = [line for line in draft.splitlines() if line.startswith("**")]
    assert headings == [f"**{i}. {t}**" for i, t in enumerate(ptw.SECTION_TITLES, 1)]
    assert draft.count("**6. Conclusioni**") == 1                       # the repeated heading removed
    assert "Testo della sezione Risposte ai quesiti." in draft          # a sentence naming it is kept
    plan_call = model.calls[0]
    assert plan_call["temperature"] == 0 and "PARTE ASSISTITA (dall'analisi del caso): i genitori di Tizio" \
        in plan_call["human"] and plan_call["human"].endswith("Prepara ora il piano.")
    applied = model.section_call("Applicazione al caso")
    assert "La scuola deve provare che la caduta era imprevedibile." in applied["human"]
    assert "Il debitore che non esegue" in applied["human"] and "TESTO-NON-NEL-PIANO" not in applied["human"]
    assert "[Cass. civ., sez. III, ord. n. 190/2026]" in applied["human"]
    assert "nell'interesse di i genitori di Tizio (danneggiati)" in applied["system"]
    procedural = model.section_call("Profili processuali e termini")
    assert "Prescrizione del diritto" in procedural["human"] and "Chi risponde" in procedural["human"]
    assert "CASO:" not in model.section_call("Conclusioni")["human"]
    assert "Il gioco era sorvegliato." in model.section_call("Conclusioni")["human"]
    plan_log = [data for event, data in model.logged if event == "case_analysis_plan"][-1]
    assert plan_log["issues"] == 2 and plan_log["replanned"] is False and plan_log["references_available"] == 2
    assert plan_log["removed"]["norme"] == ["2052 c.c."]


def test_the_planner_reads_headings_and_practical_checks(model):
    _write(articles=dg._source_block(ARTICLES, 1500))
    human = model.calls[0]["human"]
    assert "[art. 1218 c.c.]\nRubrica: Responsabilità del debitore\nIl debitore" in human
    assert "VERIFICHE PER LA FASE E LA MATERIA:\n- " in human and "Prescrizione del diritto" in human


def test_rulings_are_cited_only_where_their_passages_are(model):
    _write()
    for title in ("Risposte ai quesiti", "Inquadramento e questione centrale", "Profili processuali e termini",
                  "Conclusioni"):
        call = model.section_call(title)
        assert "Cass." not in call["human"] and "non citare pronunce" in call["system"]
    for title in ("Norme e giurisprudenza", "Applicazione al caso"):
        assert "[Cass. civ., sez. III, ord. n. 190/2026]" in model.section_call(title)["human"]
    assert "Rubrica: Responsabilità del debitore" in model.section_call("Applicazione al caso")["human"]


def _second_issue_titled(title):
    return dict(PLAN, questioni=[PLAN["questioni"][0], dict(PLAN["questioni"][1], titolo=title)])


def test_the_plan_check_drops_contradicted_issues_and_off_topic_articles(model):
    model.check = {"1218 c.c.": {"contraddice": True}, "2051 c.c.": {"pertinente": False}}
    model.plans = [_second_issue_titled("Posizione della panchina")]
    _write()
    applied = model.section_call("Applicazione al caso")["human"]
    assert "La scuola deve provare" not in applied and "La panchina era nello spazio di gioco." in applied
    assert "Posizione della panchina (norme: -;" in applied              # 2051 removed from its issue
    assert "Sì, a titolo contrattuale." not in model.section_call("Risposte ai quesiti")["human"]
    plan_log = [d for e, d in model.logged if e == "case_analysis_plan"][-1]
    assert plan_log["checked"] == {"off_topic": ["2051 c.c."], "contradicted": ["Responsabilità contrattuale della scuola"],
                                   "kept_by_heading": [], "kept_by_vote": []}
    assert len([c for c in model.calls if "controlli il piano" in c["system"]]) == 2   # one call per issue


def test_an_article_whose_heading_names_the_issue_is_never_removed(model):
    # Round 4: 624-bis c.p. ("Furto in abitazione...") judged off-topic for "tentativo di furto in
    # abitazione" in every burglary run. Here 2051 ("Danno cagionato da cosa in custodia") for "Custodia".
    model.check = {"2051 c.c.": {"pertinente": False}}
    _write()
    assert "Custodia della panchina (norme: art. 2051 c.c.;" in model.section_call("Applicazione al caso")["human"]
    plan_log = [d for e, d in model.logged if e == "case_analysis_plan"][-1]
    assert plan_log["checked"]["kept_by_heading"] == ["2051 c.c."] and plan_log["checked"]["off_topic"] == []
    assert ptw._heading_shares_title("Furto in abitazione e furto con strappo", "Tentativo di furto in abitazione")
    assert not ptw._heading_shares_title("Intimazione di licenza e di sfratto", "Sanzioni per il cancello chiuso")
    assert not ptw._heading_shares_title("Disposizioni generali sul reato", "Reato di stalking")


def test_answers_naming_a_removed_article_are_dropped(model):
    model.check = {"2051 c.c.": {"pertinente": False}}
    model.plans = [dict(_second_issue_titled("Posizione della panchina"),
                        risposte=[{"quesito": "a", "risposta": "Sì, a titolo contrattuale."},
                                  {"quesito": "b", "risposta": "Basta la custodia (art. 2051 c.c.)."}])]
    _write()
    answers = model.section_call("Risposte ai quesiti")["human"]
    assert "Sì, a titolo contrattuale." in answers and "Basta la custodia" not in answers


def test_a_plan_left_without_issues_means_the_one_call_parere(model):
    model.check = {"1218 c.c.": {"contraddice": True}, "2051 c.c.": {"contraddice": True}}
    assert _write() is None
    assert not any("SEZIONE:" in c["system"] for c in model.calls)


def test_a_failed_plan_check_keeps_the_plan(model):
    model.check_fails = True
    assert _write()
    assert [d for e, d in model.logged if e == "case_analysis_plan"][-1]["checked"] == \
        {"off_topic": [], "contradicted": [], "kept_by_heading": [], "kept_by_vote": []}


def test_a_plan_arguing_for_the_other_side_is_made_once_more(model):
    other_side = dict(PLAN, questioni=[dict(PLAN["questioni"][0], argomento="L'insegnante fu diligente.")])
    model.plans = [other_side, PLAN]
    model.opposing = [[{"argomento": "L'insegnante fu diligente.", "perche": "Giova alla scuola."}]]
    draft = _write()
    plans = [c for c in model.calls if "PIANO, nell'interesse" in c["system"]]
    assert len(plans) == 2 and "giovano alla controparte" in plans[1]["human"]
    assert "L'insegnante fu diligente." in plans[1]["human"]           # the first plan, shown to be fixed
    assert "La scuola deve provare" in model.section_call("Applicazione al caso")["human"]
    assert draft and [d for e, d in model.logged if e == "case_analysis_plan"][-1]["replanned"] is True


def test_no_usable_plan_means_the_one_call_parere(model):
    model.plans = ["Non so rispondere."]
    assert _write() is None
    assert not any("SEZIONE:" in c["system"] for c in model.calls)


def test_a_section_is_tried_once_more_then_the_one_call_parere(model):
    model.failing = {"Conclusioni": 1}
    assert _write().endswith("Testo della sezione Conclusioni.")
    model.failing = {"Norme e giurisprudenza": 2}
    assert _write() is None


# --- the practical checks ------------------------------------------------------------------------

def test_practical_checks_follow_the_method_lists_keys():
    assert set(practical_checklist.POSTURE_CHECKS) <= set(method_candidates.POSTURE_ARTICLES)
    assert set(practical_checklist.AREA_CHECKS) <= set(method_candidates.AREA_ARTICLES)
    checks = practical_checklist.checks_for(["penale", "penale"], "indagini_preliminari")
    assert checks[0].startswith("Avviso di conclusione delle indagini")       # posture first
    assert any("art. 116 c.p." in c for c in checks) and len(checks) == len(set(checks))
    assert practical_checklist.checks_for([], "") == []


# --- VOTE_POOL_ENABLED: voted articles are offered to the plan and never removed by its check ----

def _voted_articles():
    return [dict(a, _vote=True, _vote_rank=1) if a["_article"] == ("2051", "c.c.") else a for a in ARTICLES]


def test_voted_articles_are_offered_to_the_plan_and_survive_its_check(model):
    model.check = {"2051 c.c.": {"pertinente": False}, "1218 c.c.": {"pertinente": False}}
    model.plans = [_second_issue_titled("Posizione della panchina")]
    ptw.write_parere(FACTS, REQUEST, ANALYSIS, _voted_articles(), RULINGS,
                     "[art. 2051 c.c.]\nCiascuno è responsabile", "GIURISPRUDENZA", "", "it")
    plan_call = next(c for c in model.calls if "PIANO, nell'interesse" in c["system"])
    assert "SU FATTI SIMILI (sono in NORME): [art. 2051 c.c.]" in plan_call["human"]
    applied = model.section_call("Applicazione al caso")["human"]
    assert "Posizione della panchina (norme: art. 2051 c.c.;" in applied     # voted: kept although off-topic
    plan_log = [d for e, d in model.logged if e == "case_analysis_plan"][-1]
    assert plan_log["checked"]["off_topic"] == ["1218 c.c."]
    assert plan_log["checked"]["kept_by_vote"] == ["2051 c.c. (fuori tema)"]
    assert plan_log["voted_offered"] == ["2051 c.c."]


def test_a_voted_article_judged_contradicted_does_not_drop_its_issue(model):
    model.check = {"2051 c.c.": {"contraddice": True}}
    model.plans = [_second_issue_titled("Posizione della panchina")]
    ptw.write_parere(FACTS, REQUEST, ANALYSIS, _voted_articles(), RULINGS, "NORME", "GIURISPRUDENZA", "", "it")
    assert "La panchina era nello spazio di gioco." in model.section_call("Applicazione al caso")["human"]
    plan_log = [d for e, d in model.logged if e == "case_analysis_plan"][-1]
    assert plan_log["checked"]["contradicted"] == [] and plan_log["checked"]["kept_by_vote"] == ["2051 c.c. (contraddice)"]
    plan_call = next(c for c in model.calls if "PIANO, nell'interesse" in c["system"])
    assert "SU FATTI SIMILI" not in plan_call["human"]    # not in the NORME block given: not offered


# --- the plan reads back a law named in full; PLAN_KEEP_APPLICABLE_ENABLED keeps breadth ------------

APPALTI = "Codice degli Appalti 2026"


def test_a_law_named_in_full_is_read_back_not_removed():
    known = {("95", APPALTI), ("98", APPALTI), ("1218", "c.c.")}
    assert ptw._refs_of("art. 95 Codice degli Appalti 2026", known) == [("95", APPALTI)]
    assert ptw._refs_of("art. 1218 c.c.", known) == [("1218", "c.c.")]
    plan, removed = ptw.clean_plan({"questioni": [{"titolo": "Esclusione", "argomento": "x",
                                                  "norme": ["art. 98 Codice degli Appalti 2026", "art. 5 d.lgs. 9"]}]},
                                   known, {})
    assert plan["questioni"][0]["norme"] == [("98", APPALTI)] and removed["norme"] == ["art. 5 d.lgs. 9"]


def test_up_to_seven_issues_when_wide():
    assert "da due a sette" in ptw._plan_system(7) and "da due a cinque" in ptw._plan_system()
    raw = {"questioni": [{"titolo": f"Q{i}", "argomento": "x", "norme": []} for i in range(9)]}
    assert len(ptw.clean_plan(raw, set(), {}, 7)[0]["questioni"]) == 7


def _applicable_articles():
    rows = [dict(a, _applicability="applicabile") for a in ARTICLES]            # 2048, 1218, 2051, 1917 c.c.
    rows.append(dict(_article("624", "c.p.", "Furto."), _applicability="applicabile"))   # another area
    rows.append(dict(_article("2043", "c.c.", "Risarcimento per fatto illecito."), _applicability="incompatibile"))
    return rows


def test_applicable_articles_in_the_cases_areas():
    assert ptw._applicable(_applicable_articles(), FACTS, REQUEST, ANALYSIS) == \
        [("2048", "c.c."), ("1218", "c.c."), ("2051", "c.c."), ("1917", "c.c.")]
    # named in the text: kept even from another area
    assert ("624", "c.p.") in ptw._applicable(_applicable_articles(), FACTS + " Ipotesi di art. 624 c.p.",
                                              REQUEST, ANALYSIS)


def test_keep_applicable_places_protects_and_adds_the_rest(model, monkeypatch):
    monkeypatch.setenv("PLAN_KEEP_APPLICABLE_ENABLED", "true")
    model.check = {"2051 c.c.": {"pertinente": False}}
    model.plans = [_second_issue_titled("Posizione della panchina")]
    ptw.write_parere(FACTS, REQUEST, ANALYSIS, _applicable_articles(), RULINGS, "NORME", "GIURISPRUDENZA", "", "it")
    plan_call = next(c for c in model.calls if "PIANO, nell'interesse" in c["system"])
    assert "NORME APPLICABILI AI FATTI: [art. 2048 c.c.], [art. 1218 c.c.], [art. 2051 c.c.], [art. 1917 c.c.]" \
        in plan_call["human"]
    applied = model.section_call("Applicazione al caso")["human"]
    assert "Posizione della panchina (norme: art. 2051 c.c.;" in applied            # off-topic, but applicable
    assert f"{ptw.OTHER_NORMS_TITLE} (norme: art. 2048 c.c., art. 1917 c.c." in applied
    assert "TUTTE le norme del PIANO" in model.section_call("Norme e giurisprudenza")["system"]
    assert ptw.OTHER_NORMS_TITLE in model.section_call("Applicazione al caso")["system"]
    logged = [d for e, d in model.logged if e == "case_analysis_plan_applicable"][-1]
    assert logged["placed"] == ["1218 c.c.", "2051 c.c."] and logged["added_to_last_issue"] == ["2048 c.c.", "1917 c.c."]
    plan_log = [d for e, d in model.logged if e == "case_analysis_plan"][-1]
    assert plan_log["checked"]["kept_as_applicable"] == ["2051 c.c. (fuori tema)"]


def test_without_the_flag_nothing_is_added(model):
    model.check = {"2051 c.c.": {"pertinente": False}}
    model.plans = [_second_issue_titled("Posizione della panchina")]
    ptw.write_parere(FACTS, REQUEST, ANALYSIS, _applicable_articles(), RULINGS, "NORME", "GIURISPRUDENZA", "", "it")
    applied = model.section_call("Applicazione al caso")["human"]
    assert ptw.OTHER_NORMS_TITLE not in applied and "Posizione della panchina (norme: -;" in applied
    assert "TUTTE le norme del PIANO" not in model.section_call("Norme e giurisprudenza")["system"]
    assert not [d for e, d in model.logged if e == "case_analysis_plan_applicable"]


def test_keep_checked_places_every_judged_article(model, monkeypatch):
    """PLAN_KEEP_CHECKED_ENABLED: every article judged applicabile or alternativa is in the plan, under the
    issue its rubrica shares words with or in a last one; protected from the plan check; seven issues."""
    monkeypatch.setenv("PLAN_KEEP_CHECKED_ENABLED", "true")
    model.check = {"2051 c.c.": {"pertinente": False}}
    model.plans = [_second_issue_titled("Posizione della panchina")]
    ptw.write_parere(FACTS, REQUEST, ANALYSIS, _applicable_articles(), RULINGS, "NORME", "GIURISPRUDENZA", "", "it")
    plan_call = next(c for c in model.calls if "PIANO, nell'interesse" in c["system"])
    assert ("NORME GIUDICATE APPLICABILI AI FATTI: [art. 2048 c.c.], [art. 1218 c.c.], [art. 2051 c.c.], "
            "[art. 1917 c.c.], [art. 624 c.p.]. Ognuna deve comparire nel piano") in plan_call["human"]
    assert "da due a sette" in plan_call["system"]
    logged = [d for e, d in model.logged if e == "case_analysis_plan_applicable"][-1]
    assert logged["mode"] == "checked" and logged["placed"] == ["1218 c.c.", "2051 c.c."]
    assert logged["added_to_last_issue"] == ["2048 c.c.", "1917 c.c.", "624 c.p."]       # no rubrica to group by
    applied = model.section_call("Applicazione al caso")["human"]
    assert "Posizione della panchina (norme: art. 2051 c.c.;" in applied                   # off-topic, but checked
    assert f"{ptw.OTHER_NORMS_TITLE} (norme: art. 2048 c.c., art. 1917 c.c., art. 624 c.p." in applied
    assert "TUTTE le norme del PIANO" in model.section_call("Norme e giurisprudenza")["system"]


def test_graded_keep_checked_offers_the_best_and_forces_none(model, monkeypatch):
    """CHECK_GRADED_ENABLED with PLAN_KEEP_CHECKED_ENABLED: grades 2-3 only, offered to the plan; an
    article the plan does not place and whose rubrica fits no issue stays out (no catch-all issue)."""
    monkeypatch.setenv("PLAN_KEEP_CHECKED_ENABLED", "true")
    monkeypatch.setenv("CHECK_GRADED_ENABLED", "true")
    model.plans = [_second_issue_titled("Posizione della panchina")]
    rows = [dict(ARTICLES[0], _applicability="applicabile", _grade=2, _fused_rank=3),
            dict(ARTICLES[1], _applicability="applicabile", _grade=3, _fused_rank=1),
            dict(ARTICLES[2], _applicability="applicabile", _grade=2, _fused_rank=2),
            dict(ARTICLES[3], _applicability="marginale", _grade=1, _fused_rank=4)]
    ptw.write_parere(FACTS, REQUEST, ANALYSIS, rows, RULINGS, "NORME", "GIURISPRUDENZA", "", "it")
    plan_call = next(c for c in model.calls if "PIANO, nell'interesse" in c["system"])
    assert ("NORME PIÙ PERTINENTI AL CASO: [art. 1218 c.c.], [art. 2051 c.c.], [art. 2048 c.c.]. Metti ciascuna "
            "tra le norme della questione che disciplina; una che non disciplina nessuna questione del piano resta "
            "fuori.") in plan_call["human"]
    logged = [d for e, d in model.logged if e == "case_analysis_plan_applicable"][-1]
    assert logged["placed"] == ["1218 c.c.", "2051 c.c."] and logged["added_to_last_issue"] == []
    applied = model.section_call("Applicazione al caso")["human"]
    assert ptw.OTHER_NORMS_TITLE not in applied and "1917" not in applied
