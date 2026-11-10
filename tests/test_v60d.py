"""v60d: live round 1 on unseen cases (≈ 92 s, rows=11 and 5, questions from rules only) -- the fixes as mechanisms:
1 replies in the first person and relative dates; 2 the first read's institutions as searches, the area's code first;
3 the grading's breadth; 4 the first read's questions; 5 no keyword templates in §4; 6 the «non si applica» blocks;
7 the parties (the deceased, the relationships, a rule only in its area). The model and the graph are stubbed."""

import copy
import os
import sys
from pathlib import Path

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.rag import case_read, fact_signals, interaction_copy as C, render_computed, topic_groups  # noqa: E402
from src.rag import defensive_generation as dg  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
import test_interactive_flow as base  # noqa: E402

MONDAY = "2026-11-09"


# --- 1. replies: the client in the first person, relative dates ------------------------------------------------------

@pytest.mark.parametrize("text, name", [
    ("Assisto Mario Rossi.", "Mario Rossi"),
    ("assisto il sig. Bianchi, che ha subito il danno", "Bianchi"),
    ("Difendo Rossi nel procedimento penale", "Rossi"),
    ("Sono il legale di Anna Verdi", "Anna Verdi"),
    ("sono l'avvocato della signora Verdi", "Verdi"),
    ("Il mio cliente è Rossi.", "Rossi"),
    ("la mia cliente è Anna Verdi", "Anna Verdi"),
    ("Rappresento la società Alfa", "Alfa"),
    ("Seguo Giulia Neri", "Giulia Neri"),
    ("Assistiamo Carlo, il proprietario", "Carlo"),
    ("agisco per conto di Bianchi", "Bianchi"),
    ("Sono il difensore di Tizio", "Tizio"),
    ("Sei l'avvocato di Leonardo", "Leonardo"),
])
def test_the_client_the_user_names_in_the_first_person(text, name):
    assert flow.stated(text)["client"] == name


@pytest.mark.parametrize("text, role", [
    ("Difendo l'imputato", "imputato"), ("assisto la persona offesa", "persona offesa"),
    ("il mio cliente è il convenuto", "convenuto"), ("Sono il legale del locatore", "locatore"),
])
def test_the_client_the_user_names_by_its_role(text, role):
    said = flow.stated(text)
    assert said["client"] == "" and said["client_role"] == role


def test_no_client_without_the_words():
    for text in ("Il caso riguarda Rossi e Bianchi.", "Rossi ha chiuso il cancello", "Che cosa può fare Rossi?"):
        assert flow.stated(text)["client"] == "" and flow.stated(text)["client_role"] == ""


@pytest.mark.parametrize("text, written", [
    ("2 giorni fa", "7 novembre 2026"), ("due giorni fa", "7 novembre 2026"), ("ieri sera", "8 novembre 2026"),
    ("l'altro ieri", "7 novembre 2026"), ("stamattina", "9 novembre 2026"), ("una settimana fa", "2 novembre 2026"),
    ("la settimana scorsa", "settimana dal 2 novembre 2026 al 8 novembre 2026"), ("a marzo", "marzo 2026"),
    ("a dicembre", "dicembre 2025"), ("3 mesi fa", "agosto 2026"), ("il mese scorso", "ottobre 2026"),
    ("l'anno scorso", "2025"), ("due anni fa", "2024"), ("lunedì scorso", "2 novembre 2026"),
    ("sabato scorso", "7 novembre 2026"), ("il 3 marzo", "3 marzo 2026"), ("il 20 dicembre", "20 dicembre 2025"),
])
def test_a_relative_date_is_resolved_against_the_day_the_user_writes(monkeypatch, text, written):
    monkeypatch.setenv(fact_signals.TODAY_ENV, MONDAY)
    assert fact_signals.relative_date(text) == written
    assert fact_signals.parse_date(written)          # a date the computed terms read


def test_a_week_is_an_interval_and_an_absolute_date_is_no_relative_one(monkeypatch):
    monkeypatch.setenv(fact_signals.TODAY_ENV, MONDAY)
    lo, hi, iso = fact_signals.parse_date(fact_signals.relative_date("la settimana scorsa"))
    assert (lo.isoformat(), hi.isoformat(), iso) == ("2026-11-02", "2026-11-08", "2026-11-02")
    assert fact_signals.relative_date("nel 2021") == "" and fact_signals.parse_date("nel 2021")[2] == "2021"
    assert fact_signals.relative_date("il 3 marzo 2020") == "" and fact_signals.relative_date("nessuna data") == ""


def _date_state(monkeypatch):
    state = base._state(monkeypatch, base.NO_CLIENT)
    monkeypatch.setenv(fact_signals.TODAY_ENV, MONDAY)
    facts = flow._facts(flow._restored_analysis(state["analysis"]))
    q = flow._family_questions("data_fatto", facts, {})[0]
    state["questions"] = [q] + [x for x in state["questions"] if x["id"] != q["id"]]
    return state, q


def test_a_reply_with_a_relative_date_answers_the_date_of_the_fact_and_sets_the_present(monkeypatch):
    state, q = _date_state(monkeypatch)
    empty = {"topics": [], "confirm": False, "answers": {}, "angle": "", "corrections": [], "statements": [],
             "intents": [], "question": "", "how": "none"}
    parsed = flow.reply_stated("È successo 2 giorni fa.", state, empty)
    assert parsed["answers"] == {q["id"]: "7 novembre 2026"} and parsed["relative_dates"] == [q["id"]]
    flow.apply_reply(state, parsed)
    facts = flow._facts(flow._restored_analysis(state["analysis"]))
    assert state["answered"][q["id"]] == "7 novembre 2026" and facts["present_date"] == "9 novembre 2026"
    assert any(e.get("date") == "7 novembre 2026" for e in facts["events"])


def test_the_interpreters_relative_answer_is_resolved_too(monkeypatch):
    state, q = _date_state(monkeypatch)
    out = flow._clean_reply({"answers": {q["id"]: "ieri"}}, state)
    assert out["answers"] == {q["id"]: "8 novembre 2026"} and out["relative_dates"] == [q["id"]]


def test_a_reply_naming_the_client_settles_it_and_no_assumption_line_follows(monkeypatch):
    state = base._state(monkeypatch, base.NO_CLIENT)
    assert any(q["id"] == "parte_assistita" for q in state["questions"])
    empty = {"topics": [], "confirm": False, "answers": {}, "angle": "", "corrections": [], "statements": [],
             "intents": [], "question": "", "how": "none"}
    changes = flow.apply_reply(state, flow.reply_stated("Assisto Leonardo.", state, empty))
    assert state["answered"]["parte_assistita"] == "Leonardo" and "parte assistita: Leonardo" in changes
    analysis = flow._restored_analysis(state["analysis"])
    assert flow.settle_client(state, analysis) == ("Leonardo", "")


def test_the_interpreters_client_answer_is_kept_not_dropped_as_a_non_yes_no(monkeypatch):
    state = base._state(monkeypatch, base.NO_CLIENT)
    out = flow._clean_reply({"answers": {"parte_assistita": "Domitilla"}}, state)
    assert out["answers"] == {"parte_assistita": "Domitilla"}
    flow.apply_reply(state, out)
    assert state["answered"]["parte_assistita"] == "Domitilla"


def test_after_a_reply_that_answered_a_question_never_non_mi_hai_indicato(monkeypatch):
    facts = {**base.NO_CLIENT, "parties": [{"name": "Leonardo", "role": "attore"},
                                           {"name": "Domitilla", "role": "convenuta"}]}
    state = base._state(monkeypatch, facts)
    analysis = flow._restored_analysis(state["analysis"])
    assert flow.settle_client(state, copy.deepcopy(analysis))[1].startswith("Non mi hai indicato")
    state["answered_in_reply"] = True
    name, line = flow.settle_client(state, copy.deepcopy(analysis))
    assert name == "Domitilla" and line == ""


# --- 2. the first read's institutions as searches; the area's code first ------------------------------------------

def test_the_area_codes_follow_the_frame():
    assert flow.area_codes({"frame": {"areas": ["penale"]}}) == ["c.p.", "c.p.p."]
    assert flow.area_codes({"frame": {"areas": ["successioni", "famiglia"]}}) == ["c.c.", "c.p.c."]
    assert flow.area_codes({"frame": {"areas": ["amministrativo"]}}) == ["c.p.a."]
    assert flow.area_codes({}) == ["c.c.", "c.p.", "c.p.c.", "c.p.p."]


def test_each_institution_is_searched_in_each_area_code_as_an_ordinary_candidate():
    read = {"hypotheses": [{"line": "controparte", "institutions": ["Omicidio"]},
                           {"line": "principale", "institutions": ["legittima difesa", "Omicidio"]}]}
    assert [x["name"] for x in flow.read_institutions(read)] == ["legittima difesa", "Omicidio"]   # main line first, once
    asked = []

    def resolve(session, documents, ist):
        asked.append((ist["nome"], ist["codice"]))
        if ist["codice"] != "c.p.":
            return [], {}
        number = "52" if ist["nome"] == "legittima difesa" else "575"
        return [{"_article": (number, "c.p."), "_heading_match": True, "_fact_institution": ist["nome"], "_tier": 1,
                 "s": {"titolo_articolo": "x"}}], {}
    rows, log = flow.first_read_search(None, "db", [], flow.read_institutions(read), ["c.p.", "c.p.p."],
                                       session=object(), resolve=resolve)
    assert asked == [("legittima difesa", "c.p."), ("legittima difesa", "c.p.p."), ("Omicidio", "c.p."),
                     ("Omicidio", "c.p.p.")]
    assert [r["_article"] for r in rows] == [("52", "c.p."), ("575", "c.p.")]
    assert all(not r.get("_heading_match") and not r.get("_tier") and r["_first_read_code"] == "c.p." for r in rows)
    assert log["hits"] == {"legittima difesa": ["52 c.p."], "Omicidio": ["575 c.p."]}


def test_the_first_reads_finds_are_graded_outside_the_prefilter_cap(monkeypatch):
    texts = {f"{n} c.p.": "t" for n in range(1, 6)}
    by_label = {label: tuple(label.split(" ")) for label in texts}
    graded = []
    monkeypatch.setattr(dg, "_prompt_reach", lambda rows, grades, *a: {("1", "c.p.")})

    def grade_these(chosen):
        graded.extend(chosen)
        return {}, []
    with dg.grading_limits({"rounds": 2, "prefilter_keep": ["5 c.p."]}):
        dg._grade_prompt_rows(texts, by_label, [], grade_these, "f", "", {}, "it")
    assert graded == ["1 c.p.", "5 c.p."]          # admitted by the estimate, and the first read's own


def test_the_prefilter_leaves_the_first_reads_finds_out_of_its_ranking(monkeypatch):
    rows = [{"_article": (str(n), "c.p."), "s": {"plain_text": f"testo {n}", "titolo_articolo": f"r{n}"}}
            for n in range(1, 6)]
    seen = {}

    def fake_prompt_rows(texts, by_label, article_rows, grade_these, *a):
        seen["texts"] = list(texts)
        return {}, {}, []
    monkeypatch.setattr(dg, "_grade_prompt_rows", fake_prompt_rows)
    monkeypatch.setattr(dg, "_prefiltered", lambda texts, case, top, *a: (dict(list(texts.items())[:top]),
                                                                          list(texts)[top:]))
    with dg.grading_limits({"graded": True, "grade_all": True, "rounds": 2, "prefilter": 2,
                            "prefilter_keep": ["5 c.p."]}) as limits:
        dg._filter_applicable_articles("Il caso.", "", rows, {})
    assert seen["texts"] == ["1 c.p.", "2 c.p.", "5 c.p."] and limits["prefilter_kept"] == ["1 c.p.", "2 c.p."]
    assert limits["candidates_n"] == 5


def test_for_the_same_institution_the_areas_code_comes_first():
    civil = {"title": "Fatti illeciti", "code": "c.c.", "graded": 3, "best_grade": 3, "primary": True,
             "institutions": ["lesioni"], "hypotheses": []}
    other = {"title": "Altro", "code": "c.c.", "graded": 2, "best_grade": 2, "primary": True, "hypotheses": []}
    criminal = {"title": "Delitti contro la persona", "code": "c.p.", "graded": 1, "best_grade": 3, "primary": True,
                "institutions": [], "hypotheses": [{"line": "subordinata", "anchor_institution": "Lesioni"}]}
    topics = [civil, other, criminal]
    topic_groups.mark_area(topics, ["c.p.", "c.p.p."])
    topic_groups.order(topics)
    assert [t["title"] for t in topics] == ["Delitti contro la persona", "Fatti illeciti", "Altro"]
    plain = [dict(civil, in_area=None), dict(other), dict(criminal, in_area=None)]
    topic_groups.order(plain)                       # no area marked: v60c's order alone
    assert [t["title"] for t in plain] == ["Fatti illeciti", "Altro", "Delitti contro la persona"]


# --- 3. the grading's breadth -----------------------------------------------------------------------------------------

def test_the_breadth_says_when_the_prefilter_is_cutting():
    rows = [{"_article": (str(n), "c.p."), "_grade": 2} for n in range(1, 43)]
    limits = {"candidates_n": 120, "prefilter": 50, "prefilter_kept": [f"{n} c.p." for n in range(1, 51)],
              "prefilter_keep": ["99 c.p."], "graded_n": 51}
    out = flow.grading_breadth(rows, limits)
    assert (out["candidates"], out["prefilter_kept"], out["graded_2plus"], out["kept_graded_2plus"]) == (120, 50, 42, 42)
    assert out["cutting"] is True and out["outside_cap"] == 1
    assert flow.grading_breadth(rows[:11], limits)["cutting"] is False


# --- 4. the first read's questions --------------------------------------------------------------------------------

def test_the_first_reads_questions_keep_their_room_and_each_drop_has_its_reason():
    rules = [{"id": f"q{i}", "text": f"Domanda numero {i} sulla servitù?", "short": f"punto {i}"} for i in range(5)]
    log = []
    out = flow.with_read_questions(rules, ["se l'arma era detenuta legalmente", "", "punto numero",
                                           "chi ha chiamato i soccorsi", "quando è stata sporta denuncia"], {}, log)
    ids = [q["id"] for q in out]
    assert len(out) == flow.MAX_QUESTIONS and ids[:3] == ["q0", "q1", "q2"] and ids[3:] == ["read_1", "read_4"]
    reasons = {e["point"] if "point" in e else e["question"]: e["dropped"] for e in log}
    assert reasons[""] == "empty" and reasons["punto numero"] == "dedup" and reasons["quando è stata sporta denuncia"] == "cap"
    assert reasons["q3"].startswith("cap") and reasons["q4"].startswith("cap")


def test_the_first_read_takes_its_unknowns_however_the_model_writes_them():
    assert case_read._points("se c'era un testimone; quando è avvenuto") == ["se c'era un testimone", "quando è avvenuto"]
    assert case_read._points([{"fatto": "l'ora del fatto"}, "chi guidava"]) == ["l'ora del fatto", "chi guidava"]
    read = case_read.clean({"fatti": ["Tizio cade"], "non_detto": "se il pavimento era bagnato"})
    assert read["unknown"] == ["se il pavimento era bagnato"] and read["unknown_raw"] == {"type": "str", "n": 27}


def test_the_case_with_named_unknowns_shows_one_after_the_rules_questions(monkeypatch):
    import test_case_first as cf
    monkeypatch.setenv("INTERACTIVE_CASE_FIRST", "true")
    monkeypatch.setattr(dg, "_analyse_case_for_pipeline", lambda text, lang, request="": {"frame": {"areas": ["penale"]}})
    monkeypatch.setattr(case_read, "read", lambda text, request="", lang="it": case_read.clean(cf.READ, request))
    rows = [cf._graded("52", "c.p.", 3, "Difesa legittima")]
    built = topic_groups.build(rows, cf.UNITS)
    monkeypatch.setattr(flow, "_retrieve", lambda text, request, lang, analysis, pending=None: {
        "article_rows": rows, "regime_rows": [], "topics": built["topics"], "others": [], "topic_rows": rows,
        "rubriche": topic_groups.rubriche_of(rows), "grading": {"s": 1.0, "topics_s": 0.1}, "reserve": []})
    monkeypatch.setattr(flow, "_case_branches_early", lambda *a: __import__("contextlib").nullcontext())
    monkeypatch.setattr(render_computed, "render", lambda *a, **k: "")
    rule_qs = [{"id": f"q{i}", "family": "x", "text": f"Regola {i}?", "short": f"r{i}"} for i in range(5)]
    monkeypatch.setattr(flow, "questions", lambda *a, **k: list(rule_qs))
    logged = []
    monkeypatch.setattr(flow, "_log", lambda event, data, *a: logged.append((event, data)))
    state = flow.analyse_case("Tizio spara al ladro.", "")
    assert "read_1" in [q["id"] for q in state["questions"]]       # v60c's were overwritten by the rules' questions
    entry = next(d for e, d in logged if e == "first_read_questions")
    assert entry["raw"] == ["se l'arma era detenuta legalmente"] and entry["asked"] == ["read_1"]


# --- 5. no keyword templates in §4 ----------------------------------------------------------------------------------

def _harm_analysis(**extra):
    tool = {"id": fact_signals.CRIMINAL_TOOL, "refs": [("589", "c.p.")], "text": "Profilo penale: la morte...",
            "why": "morte"}
    return {"frame": {"areas": ["penale"]}, "fact_signals": {"tools": [tool], "damages": [{"id": "x", "refs": []}],
                                                             "facts": {"events": []}}, **extra}


def test_store_only_has_no_keyword_profile_and_no_heads_of_damage(monkeypatch):
    monkeypatch.setattr(fact_signals, "damages_block", lambda signals: "- danno biologico")
    monkeypatch.setattr(render_computed, "client_damaged", lambda *a: True)
    legacy = [t for t, _ in render_computed.blocks(_harm_analysis(), "Tizio")]
    assert "Profilo penale" in legacy and any(t.startswith("Voci di danno") for t in legacy)
    titles = [t for t, _ in render_computed.blocks(_harm_analysis(store_offences_only=True), "Tizio")]
    assert "Profilo penale" not in titles and not any(t.startswith("Voci di danno") for t in titles)


def test_store_only_procedure_lines_are_the_topics_offences_from_the_store():
    assert render_computed.criminal_procedure(_harm_analysis(store_offences_only=True)) == ""       # no offence, no line
    facts = {"582": {"procedibilita": "a querela della persona offesa (co. 2)", "querela": True,
                     "competenza": "giudice di pace", "pena": "reclusione"}}
    text = render_computed.criminal_procedure(_harm_analysis(store_offences_only=True, offences=[("582", "Lesione personale"),
                                                                                                 ("589", "Omicidio colposo")],
                                                             offence_facts=facts))
    assert "582" in text and "querela" in text
    assert "589" in text and C.PROCEDURE_UNKNOWN in text           # no store facts: «da verificare», never a default
    assert "procedibile d'ufficio" not in text


# --- 6. «non si applica» blocks ------------------------------------------------------------------------------------

@pytest.mark.parametrize("gloss, says", [
    ("La norma non si applica al caso, perché manca un contratto tra le parti.", True),
    ("Nel caso in esame la disposizione non trova applicazione: il danno non deriva da cose in custodia.", True),
    ("L'art. 2051 c.c. non è applicabile alla fattispecie.", True),
    ("Non si applica: il fatto non è avvenuto nell'esercizio di un'attività pericolosa.", True),
    ("La norma si applica al caso: il convenuto aveva la custodia del cancello.", False),
    ("Il comma 2 non si applica al caso, ma il comma 1 regola la responsabilità.", False),
    ("Se il fondo non è intercluso, la norma non si applica al caso.", False),
    ("«La disposizione non si applica ai beni demaniali» (co. 3): il fondo è privato e la norma regola il caso.", False),
])
def test_a_gloss_declaring_its_article_inapplicable(gloss, says):
    assert flow.declares_inapplicable(gloss) is says


def test_an_inapplicable_block_is_one_line_of_the_considered_norms():
    text = ("**Art. 2043 c.c. — Risarcimento per fatto illecito**\n\nIl fatto doloso o colposo obbliga al risarcimento: "
            "il convenuto ha chiuso il cancello.\n\n**Art. 2051 c.c. — Danno cagionato da cosa in custodia**\n\n"
            "La norma non si applica al caso, perché il danno non deriva dalla cosa.\n\n"
            "**Art. 1218 c.c. — Responsabilità del debitore**\n\nNon applicabile: tra le parti non c'è un contratto.\n\n"
            "Altre norme del tema non trattate: art. 2052 c.c.")
    log = []
    out = flow.norms_final(text, log)
    assert "**Art. 2043 c.c." in out and "**Art. 2051" not in out and "**Art. 1218" not in out
    assert "Norme considerate e non applicabili: art. 2051 c.c., art. 1218 c.c." in out
    assert out.rstrip().endswith("Altre norme del tema non trattate: art. 2052 c.c.")
    assert [e["not_applicable"] for e in log if "not_applicable" in e] == ["2051 c.c.", "1218 c.c."]
    assert flow.norms_final(out) == out                                   # stable on a second pass


# --- 7. the parties --------------------------------------------------------------------------------------------------

SUCCESSION = {"parties": [{"name": "Giuseppe", "role": "de cuius"}, {"name": "Marco", "role": "figlio, erede"},
                          {"name": "Lucia", "role": "coniuge"}, {"name": "Anna", "role": "convivente"}],
              "events": [], "_regimes": True}
CASE_SUCCESSION = ("Giuseppe è deceduto nel 2025 senza testamento. Marco è figlio di Giuseppe; Lucia era sua moglie, "
                   "separata dal 2019. Anna era la convivente di Giuseppe da dieci anni. La casa è stata divisa.")


def test_the_deceased_is_never_a_party_or_a_client_option():
    assert flow.deceased(SUCCESSION, CASE_SUCCESSION) == ["Giuseppe"]
    other = {"parties": [{"name": "Paolo", "role": "vittima"}, {"name": "Rino", "role": "imputato"}]}
    assert flow.deceased(other, "La morte di Paolo è avvenuta in ospedale.") == ["Paolo"]
    asked = flow.client_question(SUCCESSION, (), CASE_SUCCESSION)
    assert asked and "Giuseppe" not in asked[0]["names"] and set(asked[0]["names"]) == {"Marco", "Lucia", "Anna"}


def test_the_relationships_the_case_states_are_given_to_the_writer(monkeypatch):
    state = base._state(monkeypatch, SUCCESSION, text=CASE_SUCCESSION)
    block = flow.relationships_block(state, flow._restored_analysis(state["analysis"]))
    assert block.startswith(flow.RELATIONSHIPS_TITLE)
    assert "Marco è figlio di Giuseppe" in block and "Anna era la convivente di Giuseppe da dieci anni." in block
    assert "- Giuseppe: deceduto/a (non è una parte)" in block


def test_a_servitu_question_is_not_asked_on_a_succession(monkeypatch):
    assert not flow.family_fits("atto_separazione", ["successioni"], "713 c.c.")
    assert not flow.family_fits("atto_separazione", ["diritti_reali", "successioni"], "713 c.c.")   # not Libro III
    assert flow.family_fits("atto_separazione", ["diritti_reali"], "1062 c.c.")
    assert flow.family_fits("data_fatto", ["successioni"], "713 c.c.") and flow.family_fits("arma_detenuta", [], "")
    state = base._state(monkeypatch)
    analysis = flow._restored_analysis(state["analysis"])
    analysis["frame"] = {"areas": ["successioni"]}
    asked = {q["family"] for q in flow.questions(analysis, (), state["topics"], state["rows"])}
    assert not asked & {"opera_separazione", "atto_separazione", "possesso_inizio", "accesso_alternativo"}


def test_a_rule_fires_only_in_its_own_area(monkeypatch):
    monkeypatch.setenv("INTERACTIVE_CASE_FIRST", "true")
    analysis = {"frame": {"areas": ["successioni"]},
                "fact_signals": {"fired": [{"id": "destinazione_padre_famiglia", "area": "diritti_reali"},
                                           {"id": "x_succ", "area": "successioni"}]}}
    assert flow.rules_in_area(analysis) == ["destinazione_padre_famiglia"]
    assert [r["id"] for r in analysis["fact_signals"]["fired"]] == ["x_succ"]
    monkeypatch.setenv("INTERACTIVE_CASE_FIRST", "false")
    legacy = {"frame": {"areas": ["successioni"]}, "fact_signals": {"fired": [{"id": "a", "area": "diritti_reali"}]}}
    assert flow.rules_in_area(legacy) == [] and len(legacy["fact_signals"]["fired"]) == 1
