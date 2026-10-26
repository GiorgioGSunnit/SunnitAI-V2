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
        self.texts, self.rephrase = {}, "[]"        # section title -> its text; the rewrite's reply
        self.review = {}                            # section title -> the review's reply (WRITER_RULES_ENABLED)

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
        if "Riscrivi ogni frase" in system:
            return self.rephrase
        if "Controlli una sezione" in system:               # WRITER_RULES_ENABLED's review
            title = re.search(r"SEZIONE «([^»]+)»", human).group(1)
            reply = self.review.get(title, "[]")
            if isinstance(reply, Exception):
                raise reply
            return reply
        title = next(t for t in ptw.SECTION_TITLES if f"SEZIONE: {t}" in system)
        if title in self.texts:
            text = self.texts[title]
            if isinstance(text, list):                      # one text per call, the last one repeated
                with self.lock:
                    return text.pop(0) if len(text) > 1 else text[0]
            return text
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
    # The boat case (the lawyer's review, 26 Oct): "garanzia" alone ties nothing; "vizi" still does.
    assert not ptw._heading_shares_title("Garanzia", "Garanzia per i vizi della barca")
    assert not ptw._heading_shares_title("Garanzia e rischi dei conferimenti", "Garanzia per i vizi della barca")
    assert ptw._heading_shares_title("Garanzia per i vizi della cosa venduta", "Garanzia per i vizi della barca")


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


def test_graded_sections_cite_forced_articles_in_their_issue_and_nothing_from_memory(model, monkeypatch):
    """T3/T4 (PLAN_KEEP_CHECKED_ENABLED with CHECK_GRADED_ENABLED): each forced article goes to the sections
    with its issue, the system prompt no longer shows a real article as its example, and a citation the
    section's material lacks goes with its whole sentence (logged)."""
    monkeypatch.setenv("PLAN_KEEP_CHECKED_ENABLED", "true")
    monkeypatch.setenv("CHECK_GRADED_ENABLED", "true")
    model.plans = [_second_issue_titled("Posizione della panchina")]
    model.texts["Applicazione al caso"] = ("La scuola risponde ex art. 2051 c.c. La clausola penale è riducibile "
                                           "(art. 1384 c.c.) dal giudice.")
    rows = [dict(ARTICLES[1], _applicability="applicabile", _grade=3, _fused_rank=1, _grade_issue="inadempimento"),
            dict(ARTICLES[2], _applicability="applicabile", _grade=2, _fused_rank=2, _grade_issue="custodia")]
    draft = ptw.write_parere(FACTS, REQUEST, ANALYSIS, rows, RULINGS, "NORME", "GIURISPRUDENZA", "", "it")
    applied = model.section_call("Applicazione al caso")
    assert "NORME E LORO QUESTIONE:\n- art. 1218 c.c. (inadempimento): solo in «" in applied["human"]
    assert "- art. 2051 c.c. (custodia): solo in «Posizione della panchina»" in applied["human"]
    assert "mai a sostegno di un'altra" in applied["system"] and "1384" not in applied["system"]
    assert "La scuola risponde ex art. 2051 c.c.\n" in draft and "riducibile" not in draft     # the sentence goes
    logged = [d for e, d in model.logged if e == "case_analysis_citation_discipline"][-1]
    assert logged["count"] == 1 and logged["dropped"] == 1 and logged["removed"][0]["removed"] == ["1384 c.c."]


# --- WRITER_RULES_ENABLED (the lawyer's blind review, 26 Oct) ----------------------------------------

def _job(jobs, title):
    return next(j for j in jobs if j[0] == title)


PLAN_CLEAN = ptw.clean_plan(PLAN, {r["_article"] for r in ARTICLES} | {("2052", "c.c.")}, {})[0]


def test_writer_rules_off_keeps_the_old_section_prompts(monkeypatch):
    monkeypatch.delenv(ptw.WRITER_RULES_FLAG, raising=False)
    jobs = ptw.section_jobs(PLAN_CLEAN, FACTS, REQUEST, ANALYSIS, ARTICLES, {}, "", keep_all=True)
    assert "come si ripartisce l'onere della prova" in _job(jobs, "Norme e giurisprudenza")[1]
    assert "TUTTE le norme del PIANO" in _job(jobs, "Norme e giurisprudenza")[1]
    assert "Al massimo" not in _job(jobs, "Conclusioni")[1] and _job(jobs, "Conclusioni")[3] == 500
    assert "non pertinente" not in ptw._section_system(PLAN_CLEAN, "it")


def test_writer_rules_section_prompts(monkeypatch):
    monkeypatch.setenv(ptw.WRITER_RULES_FLAG, "true")
    jobs = ptw.section_jobs(PLAN_CLEAN, FACTS, REQUEST, ANALYSIS, ARTICLES, {}, "", keep_all=True)
    norms = _job(jobs, "Norme e giurisprudenza")[1]
    assert "TUTTE le norme" not in norms and "come si ripartisce" not in norms
    assert "solo se la norma stessa pone una presunzione" in norms
    assert "Altre norme rilevanti" not in _job(jobs, "Applicazione al caso")[1]
    assert "fonda o sostiene la pretesa o la difesa di i genitori di Tizio" in _job(
        jobs, "Inquadramento e questione centrale")[1]
    title, instruction, _, tokens = _job(jobs, "Conclusioni")
    assert tokens == 900 and instruction.endswith("Al massimo 300 parole; chiudi sempre l'ultima frase.")
    assert _job(jobs, "Applicazione al caso")[3] == int(2200 * 1.3)
    system = ptw._section_system(PLAN_CLEAN, "it")
    assert "«non pertinente»" in system and "l'onere della prova ricade su" in system
    assert "I fatti di una pronuncia non sono mai fatti del caso" in system and "stesse norme e pronunce" not in system
    assert "Una pronuncia si cita solo dove i fatti del CASO ne integrano i presupposti" in system


SECTION_TEXT = (
    "L'art. 2051 c.c. pone la responsabilità del custode. L'onere della prova ricade sulla scuola (art. 2051 c.c.). "
    "Il debitore risponde se non prova la causa non imputabile (art. 1218 c.c.); l'onere della prova grava quindi "
    "sulla scuola. L'art. 1917 c.c. non è pertinente al caso. La scuola sapeva che la panchina era instabile. "
    "Va escluso che l'art. 2048 c.c. si applichi, perché Tizio si è fatto male da solo.")


def test_review_drops_does_not_apply_and_burden_templates_and_flags_facts(model, monkeypatch):
    monkeypatch.setenv(ptw.WRITER_RULES_FLAG, "true")
    model.review["Applicazione al caso"] = json.dumps([
        {"i": 4, "tipo": "non_applicabile", "perche": "solo per dire che non rileva"},
        {"i": 5, "tipo": "fatto_non_nel_caso", "perche": "il caso non dice che la scuola lo sapesse"},
        {"i": 6, "tipo": "non_applicabile", "perche": "esclude una norma"}])
    text, log = ptw.review_section("Applicazione al caso", SECTION_TEXT, FACTS, ARTICLES, [])
    assert "1917" not in text                                        # cited only to say it does not apply
    assert "L'onere della prova ricade sulla scuola" not in text     # 2051's text allocates nothing
    assert "se non prova la causa non imputabile" in text            # 1218's own text allocates the burden
    assert "La scuola sapeva che la panchina era instabile." in text      # FACT_MARK_INLINE_ENABLED off: only logged
    assert "Va escluso che l'art. 2048 c.c. si applichi" in text      # no "does not apply" wording: kept
    assert sorted(e["kind"] for e in log) == ["burden", "fact_not_in_case", "non_application"]
    assert next(e for e in log if e["kind"] == "fact_not_in_case")["how"] == "logged"
    monkeypatch.setenv(ptw.FACT_MARK_FLAG, "true")
    text, log = ptw.review_section("Applicazione al caso", SECTION_TEXT, FACTS, ARTICLES, [])
    assert "panchina era instabile [fatto non indicato nel caso: da verificare]." in text
    review = next(c for c in model.calls if "Controlli una sezione" in c["system"])
    assert FACTS in review["human"] and review["temperature"] == 0


def test_review_without_the_call_drops_only_remarks_on_relevance(model, monkeypatch):
    monkeypatch.setenv(ptw.WRITER_RULES_FLAG, "true")
    model.review["Applicazione al caso"] = RuntimeError("upstream timeout")
    text = ("L'art. 1917 c.c. non è pertinente al caso. L'aggravante dell'art. 61 c.p. non si applica, perché il "
            "fatto è avvenuto di giorno.")
    out, log = ptw.review_section("Applicazione al caso", text, FACTS, ARTICLES, [])
    assert out == "L'aggravante dell'art. 61 c.p. non si applica, perché il fatto è avvenuto di giorno."
    assert [e["kind"] for e in log] == ["non_application"]


def test_burden_without_a_norm_dropped_only_in_the_norms_section():
    text = "La norma regola la custodia. L'onere della prova grava sul danneggiato."
    spans = ptw._sentence_spans(text)
    assert ptw._burden_templates("Norme e giurisprudenza", text, spans, ARTICLES, [], set()) == {
        1: "burden stated without a norm"}
    assert ptw._burden_templates("Applicazione al caso", text, spans, ARTICLES, [], set()) == {}
    quoted = "L'onere della prova grava sul custode, che risponde «salvo che provi il caso fortuito» (art. 2051 c.c.)."
    assert ptw._burden_templates("Norme e giurisprudenza", quoted, ptw._sentence_spans(quoted), ARTICLES, [], set()) == {}


def test_cut_sections(monkeypatch):
    assert ptw._looks_cut("Il venditore risponde dei vizi della")
    assert not ptw._looks_cut("Il venditore risponde dei vizi.")
    assert not ptw._looks_cut("Prove da acquisire:\n- fotografie\n- perizia tecnica")
    assert ptw._looks_cut("Prove da acquisire:\n- fotografie dei")
    assert ptw._trim_to_sentence("Prima frase. Seconda frase tronc") == "Prima frase."
    assert ptw._trim_to_sentence("Prove:\n- fotografie\n- perizia del") == "Prove:\n- fotografie"


def test_a_cut_section_is_written_again_then_trimmed(model, monkeypatch):
    monkeypatch.setenv(ptw.WRITER_RULES_FLAG, "true")
    model.texts["Conclusioni"] = ["Il rischio è contenuto. La condotta consigliata è di",
                                  "Il rischio è contenuto. Si consiglia di"]
    model.texts["Risposte ai quesiti"] = ["Sì, la scuola risponde. Poi tronc", "Sì, la scuola risponde."]
    out = _write()
    assert out.endswith("**6. Conclusioni**\n\nIl rischio è contenuto.")
    assert "Sì, la scuola risponde." in out and "Poi tronc" not in out
    calls = [c for c in model.calls if "SEZIONE: Conclusioni" in c["system"]]
    assert [c["max_tokens"] for c in calls] == [900, 1350]
    writer = [d for e, d in model.logged if e == "case_analysis_writer_rules"][-1]
    assert writer["cut"] == 2 and {e["how"] for e in writer["entries"] if e["kind"] == "cut"} == {
        "rewritten with more tokens", "trimmed to the last sentence"}


def test_drop_spans_leaves_no_empty_list_item():
    text = "Prove:\n- L'art. 1917 c.c. non è pertinente.\n- fotografie"
    spans = ptw._sentence_spans(text)
    assert ptw._drop_spans(text, [s for s in spans if "1917" in text[s[0]:s[1]]]) == "Prove:\n- fotografie"


def test_regime_blocks_reach_the_plan_and_the_procedural_section(monkeypatch):
    monkeypatch.setenv("FACT_REGIMES_ENABLED", "true")
    signals = {"qualifications": [{"party": "Anna", "quality": "consumatore", "why": "persona fisica"}],
               "tools": [{"id": "provvedimento_urgenza", "refs": [["700", "c.p.c."]], "text": "Provvedimento "
                          "d'urgenza (art. 700 c.p.c.).", "why": "w", "matched": ["la caldaia perde gas"]}],
               "terms": [{"id": "denuncia_vizi_vendita", "text": "denuncia dei vizi: termine di 8 giorni — DA "
                          "VERIFICARE", "ref": "art. 1495 c.c.", "status": "DA VERIFICARE",
                          "sources": [{"ref": "art. 1495 c.c.", "text": "Il compratore decade dal diritto..."}]}]}
    analysis = {**ANALYSIS, "fact_signals": signals}
    rows = ARTICLES + [_article("700", "c.p.c.", "Fuori dei casi regolati nelle precedenti sezioni...")]
    jobs = ptw.section_jobs(PLAN_CLEAN, FACTS, REQUEST, analysis, rows, {}, "")
    title, instruction, material, _ = _job(jobs, "Profili processuali e termini")
    assert "senza ometterne nessuno: **Termini**: ogni termine" in instruction
    assert "**Strumenti processuali**" in instruction and "**Profilo penale**" not in instruction
    assert "Indica sempre ciascuno di questi termini" in instruction
    assert "Nel caso: la caldaia perde gas." in material and "[art. 1495 c.c.]" in material
    assert "Il compratore decade dal diritto" in material and "[art. 700 c.p.c.]\nFuori dei casi" in material
    assert "- Anna: consumatore (persona fisica)" in material
    assert "QUALIFICHE DELLE PARTI" in _job(jobs, "Applicazione al caso")[2]
    human = ptw._plan_human(FACTS, REQUEST, analysis, "NORME", "", "", [])
    assert "QUALIFICHE DELLE PARTI" in human and "STRUMENTI PROCESSUALI INDICATI DAI FATTI" in human
    monkeypatch.delenv("FACT_REGIMES_ENABLED")
    jobs = ptw.section_jobs(PLAN_CLEAN, FACTS, REQUEST, analysis, rows, {}, "")
    assert "STRUMENTI" not in _job(jobs, "Profili processuali e termini")[2]


def test_stance_and_roles_rules_in_the_plan_and_the_sections(monkeypatch):
    monkeypatch.delenv(ptw.WRITER_RULES_FLAG, raising=False)
    assert "contro la parte assistita o contro chi è assistito" not in ptw._plan_system()
    monkeypatch.setenv(ptw.WRITER_RULES_FLAG, "true")
    plan_system = ptw._plan_system(7)
    assert "concorso di colpa è solo una difesa della controparte" in plan_system and "da due a sette" in plan_system
    assert "per ciascun ruolo" in plan_system
    section = ptw._section_system(PLAN_CLEAN, "it")
    assert "contro i genitori di Tizio o contro chi è assistito insieme" in section


def test_the_plan_check_rejects_the_inverse_situation(model, monkeypatch):
    monkeypatch.setenv(ptw.WRITER_RULES_FLAG, "true")
    _write()
    check = next(c for c in model.calls if "controlli il piano" in c["system"])
    assert "situazione inversa" in check["system"]


def test_review_drops_a_sentence_against_the_client(model, monkeypatch):
    monkeypatch.setenv(ptw.WRITER_RULES_FLAG, "true")
    model.review["Profili processuali e termini"] = json.dumps([{"i": 2, "tipo": "contro_assistito", "perche": "x"}])
    text = ("Si propone la domanda di risarcimento contro la scuola. Si valuti anche una denuncia contro la madre di "
            "Tizio per omessa vigilanza.")
    out, log = ptw.review_section("Profili processuali e termini", text, FACTS, ARTICLES, [], "i genitori di Tizio")
    assert out == "Si propone la domanda di risarcimento contro la scuola."
    assert [e["kind"] for e in log] == ["against_client"]
    call = next(c for c in model.calls if "Controlli una sezione" in c["system"])
    assert call["human"].startswith("PARTE ASSISTITA: i genitori di Tizio") and "contro_assistito" in call["system"]


def test_damages_and_roles_blocks_reach_the_plan_and_the_sections(monkeypatch):
    monkeypatch.setenv("FACT_REGIMES_ENABLED", "true")
    signals = {"damages": [{"id": "danni_morte", "refs": [["2059", "c.c."]], "lines": ["Iure proprio: ..."]}],
               "multi_roles": [{"party": "Edil Gamma", "activities": ["ha costruito", "manutiene"]}]}
    analysis = {**ANALYSIS, "fact_signals": signals}
    human = ptw._plan_human(FACTS, REQUEST, analysis, "NORME", "", "", [])
    assert "VOCI DI DANNO DA CONSIDERARE:\n- Iure proprio" in human and "PARTI CON PIÙ RUOLI NEI FATTI" in human
    jobs = ptw.section_jobs(PLAN_CLEAN, FACTS, REQUEST, analysis, ARTICLES, {}, "")
    title, instruction, material, _ = _job(jobs, "Applicazione al caso")
    assert "la responsabilità per ciascun ruolo" in instruction and "- Edil Gamma: ha costruito; manutiene" in material
    assert "VOCI DI DANNO" in _job(jobs, "Profili processuali e termini")[2]
    assert ("2059", "c.c.") in ptw._tool_refs(analysis)


def test_fact_flags_on_arguments_quotes_sources_and_evidence_are_filtered():
    case = "Tizio, alunno, scivola durante l'ora di educazione fisica e urta una panchina di legno."
    assert ptw._fact_flag_excluded("I genitori possono sostenere che la scuola conosceva il pericolo.", case)
    assert ptw._fact_flag_excluded("- testimonianze dei compagni di classe", case) == "evidence to gather"
    assert ptw._fact_flag_excluded("La Cassazione ha ritenuto la scuola responsabile (Cass. n. 190/2026).", case)
    assert ptw._fact_flag_excluded("Tizio scivola durante l'ora di educazione fisica e urta la panchina.", case) == \
        "restates the case"
    assert ptw._fact_flag_excluded("Il caso dice: «scivola durante l'ora di educazione fisica».", case) == \
        "restates the case"
    assert ptw._fact_flag_excluded("La scuola sapeva che la panchina era instabile.", case) is None
    assert ptw._fact_flag_excluded("L'insegnante si era allontanata dalla palestra senza avvisare.", case) is None


def test_a_filtered_fact_flag_is_logged_as_such(model, monkeypatch):
    monkeypatch.setenv(ptw.WRITER_RULES_FLAG, "true")
    model.review["Applicazione al caso"] = json.dumps([{"i": 1, "tipo": "fatto_non_nel_caso", "perche": "x"}])
    text = "I genitori possono sostenere che la scuola conosceva il pericolo."
    out, log = ptw.review_section("Applicazione al caso", text, FACTS, ARTICLES, [])
    assert out == text and [(e["kind"], e["why"]) for e in log] == [("fact_flag_filtered", "argues or supposes")]
    assert ptw._writer_log(log)["facts_filtered"] == {"argues or supposes": 1}


SIGNALS_5 = {"tools": [{"id": "accertamento_preventivo", "refs": [["696", "c.p.c."], ["696-bis", "c.p.c."]],
                        "text": "ATP (art. 696 c.p.c.).", "why": "w", "matched": ["difetti"]},
                       {"id": "profilo_penale", "refs": [["589", "c.p."]], "text": "Profilo penale (art. 589 c.p.).",
                        "why": "w", "matched": ["morte"]}],
             "damages": [{"id": "danni_morte", "refs": [["2059", "c.c."]], "lines": ["Iure proprio: ..."]}],
             "terms": [{"id": "denuncia_vizi_vendita", "length": "8 giorni", "text": "denuncia: 8 giorni",
                        "ref": "art. 1495 c.c."},
                       {"id": "azione_conformita_consumo", "length": "26 mesi", "text": "azione: 26 mesi",
                        "ref": "Codice del Consumo"}]}


def test_material_use_reads_each_block_off_the_section(monkeypatch):
    monkeypatch.setenv("FACT_REGIMES_ENABLED", "true")
    analysis = {**ANALYSIS, "fact_signals": SIGNALS_5}
    text = ("**Termini**\nLa denuncia va fatta entro otto giorni dalla scoperta (art. 1495 c.c.).\n\n**Strumenti "
            "processuali**\nUn ATP (artt. 696 e 696-bis c.p.c.).\n\n**Voci di danno**\nIl danno da perdita del "
            "rapporto parentale e le assicurazioni delle parti.")
    use = ptw.material_use(analysis, text)
    assert use["termini"] == {"offered": ["denuncia_vizi_vendita", "azione_conformita_consumo"],
                              "written": ["denuncia_vizi_vendita"]}
    assert use["strumenti"]["written"] == ["accertamento_preventivo"] and use["profilo_penale"]["written"] == []
    assert use["voci_di_danno"]["written"] == ["iure proprio", "assicurazioni"]
    assert ptw._missing_blocks(use) == ["termini", "profilo_penale"]
    assert ptw._length_written("1 anno", "entro un anno dalla consegna") and not ptw._length_written("2 anni", "entro un anno")
    jobs = ptw.section_jobs(PLAN_CLEAN, FACTS, REQUEST, analysis, ARTICLES, {}, "")
    title, instruction, material, tokens = _job(jobs, "Profili processuali e termini")
    for block in ("**Termini**", "**Strumenti processuali**", "**Profilo penale**", "**Voci di danno**"):
        assert block in instruction
    assert tokens == 900 + 4 * 250 and "PROFILO PENALE DEL DANNO ALLA PERSONA:\n- Profilo penale" in material
    assert "Profilo penale" not in material.split("STRUMENTI PROCESSUALI INDICATI DAI FATTI:")[1].split("\n\n")[0]


def test_the_procedural_section_is_written_again_when_it_leaves_out_a_block(model, monkeypatch):
    monkeypatch.setenv("FACT_REGIMES_ENABLED", "true")
    model.texts["Profili processuali e termini"] = [
        "Un ATP (art. 696 c.p.c.).",
        "**Termini**\nOtto giorni per la denuncia; 26 mesi per l'azione.\n\n**Strumenti processuali**\nUn ATP (art. "
        "696 c.p.c.).\n\n**Profilo penale**\nOmicidio colposo (art. 589 c.p.).\n\n**Voci di danno**\nIure proprio "
        "e iure hereditatis; le assicurazioni."]
    out = ptw.write_parere(FACTS, REQUEST, {**ANALYSIS, "fact_signals": SIGNALS_5}, ARTICLES, RULINGS, "NORME",
                           "GIURISPRUDENZA", "", "it")
    assert "Omicidio colposo (art. 589 c.p.)" in out
    calls = [c for c in model.calls if "SEZIONE: Profili processuali" in c["system"]]
    assert len(calls) == 2 and "La stesura precedente ometteva: termini: azione_conformita_consumo" in calls[1]["system"]
    use = [d for e, d in model.logged if e == "case_analysis_material_use"][-1]
    assert use["missing"] == [] and use["written"]["profilo_penale"] == ["profilo_penale"]


def test_review_fixes_a_criminal_route_still_missing_gets_its_standard_paragraph(model, monkeypatch):
    """Round 3: 589 c.p. never named even after the rewrite."""
    monkeypatch.setenv("FACT_REGIMES_ENABLED", "true")
    without = ("**Termini**\nOtto giorni per la denuncia; 26 mesi per l'azione.\n\n**Strumenti processuali**\nUn ATP "
               "(art. 696 c.p.c.).\n\n**Voci di danno**\nIure proprio e iure hereditatis; le assicurazioni.")
    model.texts["Profili processuali e termini"] = [without, without]
    analysis = {**ANALYSIS, "fact_signals": SIGNALS_5}
    out = ptw.write_parere(FACTS, REQUEST, analysis, ARTICLES, RULINGS, "NORME", "GIURISPRUDENZA", "", "it")
    assert "**Profilo penale**" not in out                                       # flag off: as before
    monkeypatch.setenv(ptw.REVIEW_FIXES_FLAG, "true")
    model.texts["Profili processuali e termini"] = [without, without]
    out = ptw.write_parere(FACTS, REQUEST, analysis, ARTICLES, RULINGS, "NORME", "GIURISPRUDENZA", "", "it")
    assert "**Profilo penale**\n\nProfilo penale (art. 589 c.p.). Norme: art. 589 c.p." in out
    assert [d for e, d in model.logged if e == "case_analysis_standard_blocks"][-1] == {"appended": ["profilo_penale"]}
    use = [d for e, d in model.logged if e == "case_analysis_material_use"][-1]
    assert use["written"]["profilo_penale"] == ["profilo_penale"]


# --- REVIEW_FIXES_ENABLED (the lawyer's review of the boat and pool cases, 26 Oct) ------------------------------

def test_review_fixes_section_2_names_the_decisive_facts(monkeypatch):
    monkeypatch.delenv(ptw.REVIEW_FIXES_FLAG, raising=False)
    jobs = ptw.section_jobs(PLAN_CLEAN, FACTS, REQUEST, ANALYSIS, ARTICLES, {}, "")
    assert "ogni fatto del caso valutato dal punto di vista" in _job(jobs, "Inquadramento e questione centrale")[1]
    monkeypatch.setenv(ptw.REVIEW_FIXES_FLAG, "true")
    jobs = ptw.section_jobs(PLAN_CLEAN, FACTS, REQUEST, ANALYSIS, ARTICLES, {}, "")
    instruction = _job(jobs, "Inquadramento e questione centrale")[1]
    assert "i fatti che decidono l'esito" in instruction and "Non elencare né etichettare" in instruction
    assert "ogni fatto del caso valutato" not in instruction


def test_review_fixes_every_item_of_a_block_is_written(monkeypatch):
    use = {"voci_di_danno": {"offered": ["iure proprio", "iure hereditatis", "assicurazioni"],
                             "written": ["iure proprio", "assicurazioni"]},
           "strumenti": {"offered": ["provvedimento_urgenza"], "written": ["provvedimento_urgenza"]}}
    monkeypatch.delenv(ptw.REVIEW_FIXES_FLAG, raising=False)
    assert ptw._missing_blocks(use) == []
    monkeypatch.setenv(ptw.REVIEW_FIXES_FLAG, "true")
    assert ptw._missing_blocks(use) == ["voci_di_danno"]


BOAT = ("Tizio acquista dalla società Marex una barca. Il motore segnala allarmi. La società venditrice garantisce "
        "la riparazione. L'episodio si ripete in alto mare.")
CLAIMS = ("I vizi non sono stati denunciati entro il termine di 8 giorni previsto dall'art. 1495 c.c. "
          "Marex non ha dimostrato di ignorare i vizi senza colpa, come richiede l'art. 1494 c.c. "
          "L'azione si prescrive in un anno dalla consegna, entro il termine dell'art. 1495 c.c. "
          "Tizio ha acquistato la barca da Marex.")


def test_review_fixes_claims_the_case_does_not_make_become_conditions(monkeypatch):
    calls = []

    def fake(messages, max_tokens=None, **kwargs):
        calls.append({"human": messages[-1].content, **kwargs})
        return json.dumps([
            {"i": 1, "da_correggere": True, "frase": "Se i vizi non sono stati denunciati entro otto giorni dalla "
                                                    "scoperta, Tizio decade dalla garanzia (art. 1495 c.c.): il caso "
                                                    "non lo indica."},
            {"i": 2, "da_correggere": True, "frase": "Marex deve provare."},          # loses art. 1494: unusable
            {"i": 3, "da_correggere": False}])
    monkeypatch.setattr(ptw.dg, "_call_chat", fake)
    out, log = ptw.rewrite_unsupported("Risposte ai quesiti", CLAIMS, BOAT)
    assert out.startswith("Se i vizi non sono stati denunciati entro otto giorni dalla scoperta, Tizio decade")
    assert "Marex non ha dimostrato di ignorare i vizi senza colpa, come richiede l'art. 1494 c.c." in out
    assert "entro il termine dell'art. 1495 c.c." in out and out.endswith("Tizio ha acquistato la barca da Marex.")
    assert [e["how"] for e in log] == ["rewritten", "kept"]
    sent = calls[0]["human"].split("FRASI DEL PARERE:")[1]
    assert "3. L'azione si prescrive" in sent and "Tizio ha acquistato" not in sent
    assert calls[0]["temperature"] == 0
    assert ptw.rewrite_unsupported("x", "Tizio ha acquistato la barca da Marex.", BOAT) == (
        "Tizio ha acquistato la barca da Marex.", [])


CAMPER = ("Caio acquista dalla Delta s.r.l. un camper usato. Dopo un mese il frigorifero smette di funzionare. "
          "Delta si offre di ripararlo.")


def test_review_fixes_rules_conditions_and_article_text_are_never_rewritten(monkeypatch):
    """Round 3: a correct rule (art. 1495 co. 2) got a condition appended. Invented case."""
    calls = []
    monkeypatch.setattr(ptw.dg, "_call_chat", lambda messages, **k: calls.append(messages[-1].content) or "[]")
    rule = "La denuncia non è necessaria se il venditore ha riconosciuto l'esistenza del vizio (art. 1495 c.c.)."
    assert ptw.rewrite_unsupported("x", rule, CAMPER) == (rule, []) and calls == []
    rows = [{"_article": ("1495", "c.c."), "s": {"plain_text": "Il compratore decade dal diritto alla garanzia, "
                                                             "se non denunzia i vizi al venditore entro otto giorni "
                                                             "dalla scoperta."}}]
    verbatim = "Il compratore decade dal diritto alla garanzia quando non denunzia i vizi entro otto giorni (art. 1495 c.c.)."
    assert ptw._states_a_rule(verbatim, rows)
    assert ptw.rewrite_unsupported("x", verbatim, CAMPER, rows) == (verbatim, []) and calls == []
    claim = "I vizi non sono stati denunciati entro il termine di 8 giorni previsto dall'art. 1495 c.c."
    assert not ptw._states_a_rule(claim, rows)
    ptw.rewrite_unsupported("x", claim, CAMPER, rows)
    assert len(calls) == 1                                                       # a claim still goes to the check


def test_review_fixes_the_timing_of_a_death_is_a_claim_to_check():
    claim = "Poiché Tizietta è deceduta immediatamente, non si trasmette alcun danno da morte agli eredi."
    assert ptw._UNSUPPORTED_RE.search(claim)
    assert ptw._UNSUPPORTED_RE.search("La morte è sopraggiunta sul colpo.")
    assert not ptw._UNSUPPORTED_RE.search("I genitori chiedono il risarcimento per la morte della figlia.")
