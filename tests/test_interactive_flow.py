"""INTERACTIVE_FLOW (phase 2): a narrative case answered in rounds -- topics, facts and questions first, then the frame,
then the answer on the chosen topic. Off, or for a "direct" request, the single-shot pipeline answers as before."""

import copy
import re
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.rag import fact_signals, interactive_flow as flow, rule_quote

HERE = Path(__file__).parent
TODAY = "2026-11-08"

FACTS = {
    "parties": [{"name": "Leonardo", "role": "cliente, proprietario"}, {"name": "Domitilla", "role": "controparte, proprietario"}],
    "client_side": "Leonardo",
    "property_history": [{"owner": "Mevio", "asset": "terreni", "together_with": "", "ended_by": ""},
                         {"owner": "Leonardo", "asset": "terreno", "how_acquired": ""},
                         {"owner": "Domitilla", "asset": "terreno", "how_acquired": "compravendita", "from": "2003",
                          "from_iso": "2003"}],
    "works_or_signs": [{"what": "strada sterrata", "visible": "sì", "before_separation": "",
                        "existing_since": "inizio del 1900"}],
    "possession": [{"who": "Leonardo", "asset": "strada sterrata", "use": "passaggio", "since": "", "how": "pacifico"}],
    "events": [{"date": "2016", "what": "Domitilla appone un cancello", "kind": "chiusura/ostruzione accesso",
                "date_iso": "2016"}],
    "access": [], "_regimes": True}
TOPICS = [
    {"title": "Servitù per destinazione del padre di famiglia", "anchor": "1062 c.c.", "how": "suffix",
     "rubrica": "Destinazione del padre di famiglia.", "articles": ["1062 c.c.", "1061 c.c."],
     "facts": ["da verificare: un'opera visibile e permanente già esistente quando i fondi furono separati"],
     "source": "regola da verificare destinazione_padre_famiglia", "merged": []},
    {"title": "Passaggio coattivo", "anchor": "1051 c.c.", "how": "exact", "rubrica": "Passaggio coattivo.",
     "articles": ["1051 c.c.", "1052 c.c.", "1054 c.c."], "facts": [], "source": "frame", "merged": []},
    {"title": "Usucapione ordinaria di beni immobili", "anchor": "1158 c.c.", "how": "term", "rubrica": "",
     "articles": ["1158 c.c.", "1159 c.c."], "facts": ["da verificare: usucapione ordinaria di beni immobili: termine da "
                                                       "verificare"], "source": "termine usucapione_ordinaria", "merged": []}]
ART_1062 = ("La destinazione del padre di famiglia ha luogo quando consta, mediante qualunque genere di prova, che due fondi, "
            "attualmente divisi, sono stati posseduti dallo stesso proprietario, e che questi ha posto o lasciato le cose "
            "nello stato dal quale risulta la servitù.")
ART_1158 = ("La proprietà dei beni immobili e gli altri diritti reali di godimento sui beni medesimi si acquistano in "
            "virtù del possesso continuato per venti anni.")
ART_1061 = ("Le servitù non apparenti non possono acquistarsi per usucapione o per destinazione del padre di famiglia.\n"
            "Non apparenti sono le servitù quando non si hanno opere visibili e permanenti destinate al loro esercizio.")
ART_1051 = ("Il proprietario, il cui fondo è circondato da fondi altrui, e che non ha uscita sulla via pubblica né può "
            "procurarsela senza eccessivo dispendio o disagio, ha diritto di ottenere il passaggio sul fondo vicino per la "
            "coltivazione e il conveniente uso del proprio fondo.\nSono esenti da questa servitù le case, i cortili, i "
            "giardini e le aie ad esse attinenti.")


def _row(number, text, code="c.c."):
    return {"_article": [number, code], "s": {"name": f"{number}.0.0", "plain_text": text, "titolo_articolo": ""},
            "d": {"id": "LEGAL_DOC::cc", "name": "Codice Civile 2026"}}


def _state(monkeypatch=None, facts=None, text=None):
    if monkeypatch is not None:
        monkeypatch.setenv("FACT_SIGNALS_TODAY", TODAY)
    facts = copy.deepcopy(facts or FACTS)
    terms = fact_signals.compute_terms(facts, ["diritti_reali"])
    case = text or "Leonardo, proprietario di un terreno, ... Domitilla ha chiuso il cancello nel 2016."
    analysis = {"parte_assistita": "Leonardo", "frame": {"areas": ["diritti_reali"]}, flow.CASE_KEY: case,
                "fact_signals": {"facts": facts, "fired": fact_signals.fire_rules(facts, terms, areas=["diritti_reali"]),
                                 "terms": terms, "areas": ["diritti_reali"], "tools": []}}
    state = {"schema": flow.SCHEMA, "version": 0, "state": "", "created": datetime.now(timezone.utc).isoformat(),
             "lang": "it", "text": case,
             "request": "", "additions": [], "analysis": flow._jsonable(analysis), "topics": copy.deepcopy(TOPICS),
             "rows": [_row("1062", ART_1062), _row("1061", ART_1061), _row("1051", ART_1051), _row("1158", ART_1158)],
             "tool_rows": [],
             "pool": [], "rendered": "", "answered": {}, "chosen": [], "angle": "", "changes": [], "answers": {},
             "timings": {}}
    state["questions"] = flow.questions(flow._restored_analysis(state["analysis"]), (), state["topics"], state["rows"])
    flow.touch(state, "test")
    return state


# --- safety ---------------------------------------------------------------------------------------------------------

def test_the_modules_import_and_the_response_has_the_interaction_field(monkeypatch):
    for name, value in (("LLM_API_KEY", "k"), ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "k")):
        monkeypatch.setenv(name, value)
    import src.chatbot.api as api
    spec = importlib.util.spec_from_file_location("diagnose_long", HERE.parent / "scripts" / "diagnose_long.py")
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)
    assert "interaction" in api.ChatResponse.model_fields
    assert api.ChatResponse(session_id="s", answer="a", original_query="q", resolved_query="q").interaction is None
    assert callable(dl.run_interactive_sim) and callable(flow.follow_up) and flow.FLAG == "INTERACTIVE_FLOW"


def test_a_failing_step_is_logged_with_its_traceback_and_the_turn_goes_on(monkeypatch):
    seen = []
    monkeypatch.setattr(flow, "_log", lambda event, data, ms=None: seen.append((event, data)))

    def broken():
        raise RuntimeError("boom")
    assert flow._isolated("x", broken, "default") == "default"
    event, data = seen[-1]
    assert event == "step_failed" and data["step"] == "x" and "RuntimeError: boom" in data["traceback"]


# --- the session state ----------------------------------------------------------------------------------------------

def test_the_state_expires_after_its_ttl_and_versions_only_grow(monkeypatch):
    state = _state(monkeypatch)
    anchors = flow.store({"articles": ["1062 c.c."]}, state)
    assert anchors["articles"] == ["1062 c.c."] and flow.active(anchors)["version"] == state["version"]
    anchors[flow.STATE_KEY]["expires"] = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    assert flow.active(anchors) is None
    newer = _state(monkeypatch)
    newer["created"] = "later"
    assert flow.store(anchors, newer)[flow.STATE_KEY]["version"] == anchors[flow.STATE_KEY]["version"] + 1
    assert flow.active({flow.STATE_KEY: {**state, "schema": 99}}) is None


# --- 2. round 1 -----------------------------------------------------------------------------------------------------

def test_questions_come_per_topic_from_its_anchors_unknown_requisites_on_the_clients_side(monkeypatch):
    state = _state(monkeypatch)
    by = {q["id"]: q for q in state["questions"]}
    # 1062: the work at the separation and the act that separated the plots; 1051: the access; 1158: the possession
    assert set(by) == {"opera_separazione_0", "atto_separazione", "possesso_inizio_0", "accesso_alternativo"}
    assert by["opera_separazione_0"]["topics"] == ["1062 c.c."] and by["atto_separazione"]["topics"] == ["1062 c.c."]
    assert by["accesso_alternativo"]["topics"] == ["1051 c.c."] and by["possesso_inizio_0"]["topics"] == ["1158 c.c."]
    assert by["atto_separazione"]["text"] == ("Con quale atto e in quale anno Mevio ha separato i fondi? Da questo dipende "
                                              "anche se il passaggio coattivo sarebbe dovuto senza indennità (art. 1054 c.c.).")
    assert by["opera_separazione_0"]["text"] == ("Per la destinazione del padre di famiglia: quando i fondi furono "
                                                 "separati, esistevano già opere visibili e permanenti (strada sterrata) "
                                                 "destinate al passaggio?")
    assert by["accesso_alternativo"]["text"] == ("Il fondo di Leonardo ha un altro accesso alla via pubblica, anche se "
                                                 "scomodo o insufficiente?")
    unlocked = [len(q["unlocks"]) for q in state["questions"]]
    assert unlocked == sorted(unlocked, reverse=True) and all(unlocked) and len(unlocked) <= flow.MAX_QUESTIONS


def test_the_other_sides_possession_is_not_asked_and_a_question_with_no_template_is_not_asked(monkeypatch):
    facts = copy.deepcopy(FACTS)
    facts["possession"] = [{"who": "Domitilla", "asset": "strada sterrata", "use": "passaggio", "since": ""}]
    state = _state(monkeypatch, facts)
    assert "possesso_inizio_0" not in {q["id"] for q in state["questions"]}
    monkeypatch.setitem(flow.C.QUESTIONS, "atto_separazione", None)
    analysis = flow._restored_analysis(state["analysis"])
    assert "atto_separazione" not in {q["id"] for q in flow.questions(analysis, (), state["topics"], state["rows"])}


H3_TEXT = ("I terreni di Leonardo e di Sempronio appartenevano a Mevio, che li possedeva insieme. Una strada sterrata, che "
           "esisteva già all'inizio del 1900, attraversa il fondo di Sempronio. Nel 2003 Sempronio vende il terreno a "
           "Domitilla; nel 2016 Domitilla appone un cancello chiuso a chiave.")


def test_1062_asks_both_questions_unless_the_case_text_states_the_answers(monkeypatch):
    # h3 live: the extraction filled before_separation (from «esisteva già») and Mevio's ended_by; the text states neither
    facts = copy.deepcopy(FACTS)
    facts["works_or_signs"][0]["before_separation"] = "sì"
    facts["property_history"][0]["ended_by"] = "divisione"
    state = _state(monkeypatch, facts, H3_TEXT)
    ids = {q["id"] for q in state["questions"]}
    assert {"opera_separazione_0", "atto_separazione"} <= ids
    analysis = flow._restored_analysis(state["analysis"])
    assert flow.requisite_status("che questi ha posto o lasciato le cose nello stato dal quale risulta la servitù",
                                 analysis) == "da provare"
    stated = H3_TEXT + (" Nel 1950 Mevio divise i terreni tra Leonardo e Sempronio, e la strada esisteva già al momento "
                        "della divisione.")
    state = _state(monkeypatch, facts, stated)
    ids = {q["id"] for q in state["questions"]}
    assert "opera_separazione_0" not in ids and "atto_separazione" not in ids
    analysis = flow._restored_analysis(state["analysis"])
    assert flow.requisite_status("che questi ha posto o lasciato le cose nello stato dal quale risulta la servitù",
                                 analysis) == "provato"


def test_a_requisite_a_field_denies_does_not_result(monkeypatch):
    facts = copy.deepcopy(FACTS)
    facts["access"] = [{"asset": "terreno di Leonardo", "landlocked": "no"}]
    analysis = flow._restored_analysis(_state(monkeypatch, facts)["analysis"])
    assert flow.requisite_status("il fondo è circondato da fondi altrui", analysis) == "non risulta"


def test_round_one_is_the_pos_copy_topics_facts_terms_questions(monkeypatch):
    state = _state(monkeypatch)
    text, field = flow.round1(state)
    assert text.startswith("Ho letto il caso di Leonardo. Prima di scrivere il parere, vorrei concordare con te")
    line = next(l for l in text.splitlines() if l.startswith("1. **"))
    assert line.startswith("1. **Servitù per destinazione del padre di famiglia** (artt. 1062, 1061 c.c.) — da verificare: "
                           "due fondi di un unico proprietario")              # the rule's own why, a rule to verify (v55)
    assert " Da chiarire: un'opera visibile" in line and "(domanda" in line
    assert "Ecco i fatti e i termini che ho ricavato dal caso:" in text
    assert "Nel 2016 Domitilla appone un cancello." in text
    assert "L'azione possessoria è ormai preclusa (termine annuale dal 2016)." in text
    assert "L'estinzione per non uso non è maturata (scadrebbe nel 2036)." in text
    assert text.count("L'usucapione va verificata: il caso non dice da quando dura il possesso.") == 1
    assert text.rstrip().endswith(flow.C.INSTRUCTION) and '"procedi"' not in text      # v60 (H): no string dictated
    assert field["state"] == flow.TOPICS_OFFERED and [t["n"] for t in field["topics"]] == [1, 2, 3]
    assert {q["family"] for q in field["questions"]} == {"opera_separazione", "atto_separazione", "possesso_inizio",
                                                        "accesso_alternativo"}


def test_the_access_question_is_asked_the_other_way_round_and_a_yes_is_an_access(monkeypatch):
    state = _state(monkeypatch)
    q = next(q for q in state["questions"] if q["family"] == "accesso_alternativo")
    facts = flow._with_answer(copy.deepcopy(FACTS), copy.deepcopy(q), "sì")
    assert facts["access"][-1]["landlocked"] == "no" and facts["access"][-1]["asset"] == "terreno di Leonardo"


def test_an_open_prescription_or_contract_term_asks_its_date(monkeypatch):
    facts = {"parties": [{"name": "Tizio", "role": "cliente"}, {"name": "Caio", "role": "controparte"}],
             "client_side": "Tizio", "events": [{"date": "", "what": "Caio aggredisce Tizio", "kind": "reato"},
                                                {"date": "", "what": "Tizio invia una diffida a Caio", "kind": "diffida"}],
             "_regimes": True}
    analysis = {"frame": {"areas": ["penale"]}, "fact_signals": {"facts": facts, "terms": []}}
    topics = [{"title": "Querela", "anchor": "124 c.p.", "articles": ["124 c.p."], "source": "termine querela"},
              {"title": "Prescrizione ordinaria", "anchor": "2946 c.c.", "articles": ["2946 c.c."],
               "source": "termine prescrizione_ordinaria"}]
    by = {q["id"]: q for q in flow.questions(analysis, (), topics, [])}
    assert by["data_fatto"]["text"] == "Quando è avvenuto il fatto? Serve per i termini di prescrizione e di querela."
    assert by["data_fatto"]["target"] == ["events", 0, ["date"]]
    assert by["data_messa_in_mora"]["text"] == "In quale data Caio è stata messa in mora o ha ricevuto la contestazione?"
    assert by["data_messa_in_mora"]["target"] == ["events", 1, ["date"]]


# --- 3. the reply ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("message,how,topics", [("2", "numbers", [2]), ("approfondisci 1", "numbers", [1]),
                                                ("1 e 2", "numbers", [1, 2]), ("procedi", "confirm", []),
                                                ("OK", "confirm", []), ("genera", "confirm", []),
                                                ("tutti", "all", [1, 2, 3]), ("7", "none", [])])
def test_numbers_confirmations_and_all_are_read_without_a_model(monkeypatch, message, how, topics):
    state = _state(monkeypatch)
    monkeypatch.setattr(flow, "_reply_call", lambda *a: None)          # "7" is out of range: the model, which abstains
    parsed = flow.parse_reply(message, state)
    assert (parsed["how"], parsed["topics"]) == (how, topics)


def test_free_text_goes_to_one_json_call_whose_reply_is_kept_only_where_it_holds(monkeypatch):
    state = _state(monkeypatch)
    calls = []

    def fake_json(messages, max_tokens, **kwargs):
        calls.append((messages, max_tokens))
        return {"topics": [2, 9], "confirm": False,
                "answers": {"possesso_inizio_0": "dal 1985", "opera_separazione_0": "sì, c'era già",
                            "accesso_alternativo": "forse", "nessuna": "sì"},
                "angle": "", "corrections": [{"kind": "date", "event": "cancello", "date": "marzo 2017"},
                                             {"kind": "date", "event": "x", "date": "ieri"}],
                "statements": ["Il cancello è chiuso a chiave."]}
    from src.rag import defensive_generation as dg
    monkeypatch.setattr(dg, "_chat_json", fake_json)
    parsed = flow.parse_reply("Il possesso è iniziato nel 1985, la strada c'era già; il cancello è del marzo 2017", state)
    assert len(calls) == 1 and calls[0][1] == flow._REPLY_TOKENS
    assert parsed["how"] == "model" and parsed["topics"] == [2]
    assert parsed["answers"] == {"possesso_inizio_0": "dal 1985", "opera_separazione_0": "sì"}
    assert parsed["corrections"] == [{"kind": "date", "event": "cancello", "date": "marzo 2017"}]


def test_answers_go_into_the_facts_and_the_terms_rules_and_questions_are_recomputed(monkeypatch):
    monkeypatch.setenv("FACT_SIGNALS_ENABLED", "true")
    state = _state(monkeypatch)
    version = state["version"]
    changes = flow.apply_reply(state, {"answers": {"possesso_inizio_0": "1985", "opera_separazione_0": "sì"},
                                       "corrections": [{"kind": "date", "event": "cancello", "date": "marzo 2017"}],
                                       "statements": []})
    analysis = flow._restored_analysis(state["analysis"])
    assert analysis["fact_signals"]["facts"]["possession"][0]["since_iso"] == "1985"
    assert "Usucapione ordinaria di beni immobili: ora DECORSO (prima DA VERIFICARE)" in changes
    assert "Servitù per destinazione del padre di famiglia: ora applicabile" in changes
    assert any("cancello" in c and "marzo 2017" in c for c in changes)
    assert {r["id"] for r in analysis["fact_signals"]["fired"]} >= {"destinazione_padre_famiglia", "usucapione_ordinaria"}
    assert {q["id"] for q in state["questions"]} == {"accesso_alternativo", "atto_separazione"}
    assert state["topics"][0]["source"] == "regola destinazione_padre_famiglia"
    assert not any(f.startswith("da verificare") for f in state["topics"][0]["facts"])
    assert state["version"] == version + 1 and state["changes"] == changes
    assert "PRECISAZIONI DEL CLIENTE" in flow.case_text(state) and "1985" in flow.case_text(state)


def test_an_angle_picks_the_topics_it_names(monkeypatch):
    state = _state(monkeypatch)
    assert flow.match_angle(state, "il passaggio coattivo sul fondo vicino") == [2]
    assert flow.match_angle(state, "") == []


# --- 4. round 2 and the turn ----------------------------------------------------------------------------------------

def test_a_topic_number_gives_round_two_and_a_confirmation_the_answer(monkeypatch):
    state = _state(monkeypatch)
    flow.round1(state)
    monkeypatch.setattr(flow, "write_answer", lambda st, numbers: (f"RISPOSTA su {numbers}", []))
    two = flow.follow_up("2", state)
    assert two["kind"] == "round2" and two["interaction"]["state"] == flow.FRAME_RESTATED
    assert two["answer"].startswith("Riassumo come imposterò il parere. Tratterò: Passaggio coattivo. Fatti su cui mi "
                                    "baso: Nel 2016 Domitilla appone un cancello")
    assert "Resta da chiarire: altro accesso alla via pubblica; dove non abbiamo la risposta" in two["answer"]
    assert "Parti da convenire" not in two["answer"]            # no defendant computed: the sentence is left out
    assert two["answer"].rstrip().endswith("Va bene così, o vuoi correggere qualcosa?")
    done = flow.follow_up("sì", two["state"])
    assert done["kind"] == "answer" and done["answer"] == "RISPOSTA su [2]" and done["state"]["state"] == flow.ANSWERED
    deeper = flow.follow_up("approfondisci 1", done["state"])
    assert deeper["kind"] == "deepen" and deeper["answer"] == "RISPOSTA su [1]"


def test_procedi_at_round_one_skips_round_two(monkeypatch):
    state = _state(monkeypatch)
    flow.round1(state)
    monkeypatch.setattr(flow, "write_answer", lambda st, numbers: (f"RISPOSTA su {numbers}", []))
    done = flow.follow_up("procedi", state)
    assert done["kind"] == "answer" and done["answer"] == "RISPOSTA su [1]"


def test_after_the_answer_a_new_fact_is_recomputed_and_the_answer_offered_again(monkeypatch):
    monkeypatch.setenv("FACT_SIGNALS_ENABLED", "true")
    state = _state(monkeypatch)
    state["state"], state["chosen"] = flow.ANSWERED, [1]
    from src.rag import defensive_generation as dg
    monkeypatch.setattr(dg, "_chat_json", lambda *a, **k: {"topics": [], "confirm": False,
                                                           "answers": {"opera_separazione_0": "sì"}, "angle": "",
                                                           "corrections": [], "statements": []})
    monkeypatch.setattr(flow, "write_answer", lambda st, numbers: (f"RISPOSTA su {numbers}", []))
    offer = flow.follow_up("aggiungi che la strada c'era già quando Mevio divise i terreni", state)
    assert offer["kind"] == "new_fact" and "riscriva la risposta su «Servitù per destinazione" in offer["answer"]
    assert offer["state"]["pending_regenerate"] == [1]
    again = flow.follow_up("sì", offer["state"])
    assert again["kind"] == "regenerate" and again["answer"] == "RISPOSTA su [1]"


def test_a_message_the_interaction_does_not_hold_gets_one_clarifying_question_and_a_new_case_the_old_path(monkeypatch):
    state = _state(monkeypatch)       # v60 (H): inside the interaction, never a fresh round 1 -- a new narrative restarts
    state["state"] = flow.ANSWERED
    monkeypatch.setattr(flow, "_reply_call", lambda *a: {"topics": [], "confirm": False, "answers": {}, "angle": "",
                                                          "corrections": [], "statements": []})
    turn = flow.follow_up("Che cos'è l'enfiteusi?", state)
    assert turn["kind"] == "unclear" and turn["answer"].endswith(flow.C.CLARIFY_ANSWERED)
    monkeypatch.setattr(flow, "routes_here", lambda message: True)
    assert flow.follow_up("Tizio e Caio " * 60, state) is None


# --- 5. the topic answer --------------------------------------------------------------------------------------------

def test_a_requisite_is_proved_only_by_the_fact_fields_it_names(monkeypatch):
    analysis = flow._restored_analysis(_state(monkeypatch)["analysis"])
    assert flow.requisite_status("non si hanno opere visibili e permanenti destinate al loro esercizio", analysis) == "provato"
    assert flow.requisite_status("il fondo è circondato da fondi altrui", analysis) == "da provare"
    assert flow.requisite_status("qualunque genere di prova", analysis) == "da provare"


def test_the_rules_block_marks_each_requisite_with_its_status_and_the_default_block_is_unchanged(monkeypatch):
    analysis = flow._restored_analysis(_state(monkeypatch)["analysis"])
    rows = flow._restored_rows([_row("1062", ART_1062)])
    plan = {"parte_assistita": "Leonardo", "questioni": [{"norme": [("1062", "c.c.")]}]}
    block = rule_quote.material_block(plan, rows, analysis, [("1062", "c.c.")],
                                      status=lambda r: flow.requisite_status(r, analysis), title=flow.RULES_TITLE,
                                      rule_text=flow.RULES_TEXT, force=True)
    assert block.startswith(flow.RULES_TITLE) and flow.RULES_TEXT in block
    assert rule_quote.material_block(plan, rows, analysis, [("1062", "c.c.")]) == ""         # RULE_QUOTE off: as before
    monkeypatch.setenv("RULE_QUOTE", "true")
    default = rule_quote.material_block(plan, rows, analysis, [("1062", "c.c.")])
    assert default.startswith(rule_quote.TITLE) and "— provato" not in default and "— da provare" not in default
    assert "dallo stesso proprietario» — da provare" in block   # "attualmente divisi": the act of division is not stated
    assert "risulta la servitù» — da provare" in block                          # the road at the separation: unknown


def test_the_extended_verbs_make_a_rule_sentence_only_where_asked():
    sentence = "L'art. 1051 c.c. esenta dalla servitù i cortili e i giardini."
    assert not rule_quote.rule_sentence(sentence, [])
    assert rule_quote.rule_sentence(sentence, [], rule_quote.EXTENDED_VERB_RE)
    assert rule_quote.rule_sentence("L'art. 1062 c.c. prevede la destinazione.", [], rule_quote.EXTENDED_VERB_RE)


def test_objections_are_the_copys_titles_for_the_open_questions_then_the_real_actions_defences(monkeypatch):
    state = _state(monkeypatch)
    one = flow.objections(state, [state["topics"][0]], [("1062", "c.c."), ("1061", "c.c.")])
    assert one == ["La controparte potrà obiettare che le opere non erano visibili al momento della separazione.",
                   "La controparte potrà obiettare che il passaggio avveniva per mera tolleranza.",
                   "La controparte potrà obiettare che il ritardo nell'agire dimostra l'abbandono del diritto."]
    # v58: only the chosen topic's (its rule, regime or term): a frame topic raises none
    assert flow.objections({**state, "questions": []}, [state["topics"][1]], [("1051", "c.c.")]) == []
    rule_topic = {**state["topics"][1], "source": "regola fondo_intercluso"}
    two = flow.objections({**state, "questions": []}, [rule_topic], [("1051", "c.c."), ("1052", "c.c.")])
    assert two[-1] == ("La controparte potrà obiettare che il passaggio inciderebbe su cortili o giardini esenti "
                       "(art. 1051 c.c.).") and len(two) == 3
    three = flow.objections(state, [state["topics"][2]], [("1158", "c.c.")])
    assert three[0] == "La controparte potrà obiettare che non è provato quando è iniziato il possesso."
    assert "La controparte potrà obiettare che le opere non erano visibili al momento della separazione." not in three
    water = {"title": "Attribuzioni dell'amministratore", "anchor": "1130 c.c.", "source": "regola amministratore_condominio"}
    assert flow.objections(state, [water], [("1117", "c.c."), ("1130", "c.c.")]) == []      # v57-l3: no servitù


def test_the_objection_titles_are_the_copys_an_objection_with_no_reply_is_dropped_never_a_placeholder():
    log = []
    text = flow.assemble_objections("[1] Il cancello è del 2016.\n\n[3] La strada esisteva già.", ["Prima", "Seconda",
                                                                                                  "Terza"], log)
    assert text == "**1) Prima**\n\nIl cancello è del 2016.\n\n**2) Terza**\n\nLa strada esisteva già."
    assert flow.assemble_objections("Testo senza segni.", ["A", "B"], log) == ""        # nothing to pair: no section


def test_section_2_prints_each_articles_heading_and_status_lines_before_the_writers_application(monkeypatch):
    state = _state(monkeypatch)
    analysis = flow._restored_analysis(state["analysis"])
    rows = flow._restored_rows(state["rows"])
    plan = {"parte_assistita": "Leonardo", "questioni": [{"norme": [("1062", "c.c."), ("1061", "c.c.")]}]}
    from src.rag import rule_quote
    rules = rule_quote.article_rules(plan, rows, analysis, [("1062", "c.c."), ("1061", "c.c.")])
    log = []
    text = flow.assemble_norms("[1] Leonardo prova la destinazione con i titoli.", rules, analysis, rows, log)
    blocks = text.split("\n\n")
    assert blocks[0].startswith("**Art. 1062 c.c.**\n- testo, requisito «consta, mediante qualunque genere di prova")
    assert blocks[0].rstrip().endswith("risulta la servitù»: da provare.")
    assert blocks[1] == "Leonardo prova la destinazione con i titoli."
    assert blocks[2] == "**Art. 1061 c.c.**\n- co. 1: provato.\n- co. 2: provato."     # its own heading; the road is visible
    assert flow.assemble_norms("", rules, analysis, rows, []).count("**Art.") == 2           # nothing applied: the statuses


def test_a_requisite_to_prove_is_never_called_proved_and_makes_its_sentences_conditional(monkeypatch):
    analysis = flow._restored_analysis(_state(monkeypatch)["analysis"])
    documents = [{"id": "LEGAL_DOC::cc", "name": "Codice Civile 2026"}]
    rules = [{"ref": ("1054", "c.c."), "selected": [1],
              "requisites": {1: ["il fondo è divenuto da ogni parte chiuso per effetto di alienazione a titolo oneroso",
                                 "il proprietario ha diritto di ottenere dall'altro contraente il passaggio"]}}]
    log = []
    out = flow.strict_status([("Norme applicate ai fatti", "Il requisito del fondo divenuto chiuso per effetto di "
                               "alienazione è provato.")], rules, analysis, log)
    assert "è da provare" in out[0][1] and log
    log = []
    out = flow.conditional_on_unknown([("Risposta e strategia", "Nessuna indennità è dovuta a Domitilla (art. 1054 c.c.). "
                                        "Se il fondo è intercluso, vale l'art. 1054 c.c.")], rules, analysis, documents,
                                      ["Leonardo"], log)
    # v58: the marker, never a prefix
    assert out[0][1].startswith("Nessuna indennità è dovuta a Domitilla (art. 1054 c.c.) (da verificare: atto e anno "
                                "della separazione dei fondi).")
    assert out[0][1].endswith("Se il fondo è intercluso, vale l'art. 1054 c.c.") and len(log) == 1


def test_the_condition_marks_a_sentence_once_at_its_end_never_a_prefix_never_inside(monkeypatch):
    # v57-h3 §1 and §5: «Se le opere risultano esistenti alla separazione,» three times, once as «…, Questi fatti»
    analysis = flow._restored_analysis(_state(monkeypatch, None, H3_TEXT)["analysis"])
    documents = [{"id": "LEGAL_DOC::cc", "name": "Codice Civile 2026"}]
    rules = [{"ref": ("1062", "c.c."), "selected": [None],
              "requisites": {None: ["che questi ha posto o lasciato le cose nello stato dal quale risulta la servitù"]}}]
    text = ("Leonardo può ottenere la servitù, subordinatamente all'esistenza dell'opera (art. 1062 c.c.). Il caso non dice "
            "quando avvenne. Questi fatti sono determinanti per l'art. 1062 c.c. La servitù è "
            "costituita (art. 1062 c.c.) (da verificare: opere visibili alla separazione dei fondi).")
    out = flow.conditional_on_unknown([("Conclusioni", text)], rules, analysis, documents, ["Leonardo"], [])[0][1]
    assert "Se le opere" not in out and ", Questi" not in out
    assert "subordinatamente all'esistenza dell'opera (art. 1062 c.c.)." in out                 # already conditional
    assert "Questi fatti sono determinanti per l'art. 1062 c.c. (da verificare: opere visibili alla separazione dei " \
           "fondi)." in out
    assert out.count("(da verificare:") == 2                                                      # never twice
    assert flow.marked("la servitù è costituita ex art. 1062 c.c", ["x"]) is None              # no stop of its own
    assert flow.conditional_on_unknown([("Conclusioni", "Tali fatti vanno verificati (art. 1062 c.c.).")], rules, analysis,
                                       documents, [], [])[0][1] == "Tali fatti vanno verificati (art. 1062 c.c.)."


def test_every_round_text_comes_from_the_copy(monkeypatch):
    state = _state(monkeypatch)
    monkeypatch.setattr(flow.C, "ROUND1_OPENING", "Caso di {client}.")
    monkeypatch.setitem(flow.C.QUESTIONS, "possesso_inizio", "Inizio del possesso di {who}?")
    monkeypatch.delitem(flow.C.QUESTIONS, "atto_separazione")
    analysis = flow._restored_analysis(state["analysis"])
    state["questions"] = flow.questions(analysis, (), state["topics"], state["rows"])
    text, _ = flow.round1(state)
    assert text.startswith("Caso di Leonardo.") and "Inizio del possesso di Leonardo?" in text
    assert "Mevio ha separato" not in text                                  # no template: not asked


def test_term_consistency_rewrites_a_sentence_against_a_computed_term(monkeypatch):
    from src.rag import plan_then_write as ptw
    analysis = flow._restored_analysis(_state(monkeypatch)["analysis"])
    context = ptw.statement_context(analysis)
    log = []
    out = flow.term_consistency([("Risposta e strategia", "La servitù si è già acquisita per usucapione ventennale. "
                                  "Il cancello è stato apposto nel 2016.")], context, analysis, log)
    text = out[0][1]
    assert "inizio del possesso è da accertare" in text and "Il cancello è stato apposto nel 2016." in text
    assert log and log[0]["how"] == "rewritten"


def test_an_article_no_fact_supports_goes_to_other_profiles_and_a_citation_outside_the_set_goes(monkeypatch):
    analysis = flow._restored_analysis(_state(monkeypatch)["analysis"])
    documents = [{"id": "LEGAL_DOC::cc", "name": "Codice Civile 2026"}]
    rules = [{"ref": ("1051", "c.c."), "selected": [1], "requisites": {1: ["il fondo è circondato da fondi altrui",
                                                                              "non ha uscita sulla via pubblica"]}}]
    sections = [("Risposta e strategia", "Leonardo può agire in confessoria. Può anche chiedere il passaggio coattivo "
                                         "ex art. 1051 c.c."),
                ("Norme applicate ai fatti", "L'art. 1062 c.c. fa nascere la servitù."),
                ("Obiezioni e repliche", "**La controparte** invocherà l'art. 1051 c.c."), ("Conclusioni", "Agire.")]
    log = []
    assert flow.unsupported_out(sections, rules, analysis, documents, [], keep=[("1051", "c.c.")]) == sections  # an anchor
    out = flow.unsupported_out(sections, rules, analysis, documents, log)
    titles = [t for t, _ in out]
    assert titles == ["Risposta e strategia", "Norme applicate ai fatti", "Obiezioni e repliche", flow.OTHER_PROFILES_TITLE,
                      "Conclusioni"]
    assert "1051" not in out[0][1] and "passaggio coattivo ex art. 1051 c.c." in out[3][1]
    assert "1051" in out[2][1]                         # an objection keeps its article
    log = []
    checked = flow.citation_check([("Norme applicate ai fatti", "L'art. 1062 c.c. fa nascere la servitù. Si applica anche "
                                    "l'art. 2043 c.c. al danno.")], [("1062", "c.c.")], documents, log)
    assert "2043" not in checked[0][1] and "1062" in checked[0][1] and log


def test_the_answer_has_the_five_sections_the_rendered_block_untouched_and_offers_the_other_topics(monkeypatch):
    from src.rag import plan_then_write as ptw, render_computed
    state = _state(monkeypatch)
    state["rendered"] = "**Termini**: azione possessoria: termine annuale DECORSO."
    monkeypatch.setattr(flow, "_documents", lambda: [{"id": "LEGAL_DOC::cc", "name": "Codice Civile 2026"}])
    plans = []

    def fake_plan(facts, request, analysis, articles, rulings, principles, rows, references):
        plans.append(sorted(tuple(r["_article"]) for r in rows))
        return {"parte_assistita": "Leonardo", "posizione": "", "obiettivo": "", "questione_centrale": "",
                "risposte": [], "subordinate": [], "punti_deboli": [],
                "questioni": [{"titolo": "T", "norme": [("1062", "c.c.")], "pronunce": [], "argomento": "",
                               "obiezione": "", "replica": ""}]}
    monkeypatch.setattr(ptw, "make_plan", fake_plan)
    monkeypatch.setattr(render_computed, "render", lambda analysis, party="": "RENDERED BLOCK")
    jobs_seen = []

    def fake_write(plan, jobs, lang, named, documents, rows, case, **kwargs):
        jobs_seen.extend(jobs)
        mark = {"Norme applicate ai fatti": "[1] ", "Obiezioni e repliche": "[1] "}
        return [(title, f"{mark.get(title, '')}Testo di {title}. L'art. 1062 c.c. si applica. L'art. 2043 c.c. no.")
                for title, *_ in jobs]
    monkeypatch.setattr(ptw, "write_sections", fake_write)
    text, citations = flow.write_answer(state, [1])
    assert plans == [[("1061", "c.c."), ("1062", "c.c.")]]                 # the topic's articles only
    assert [j[0] for j in jobs_seen] == ["Risposta e strategia", "Norme applicate ai fatti", "Obiezioni e repliche",
                                         "Conclusioni"]
    norme = next(j for j in jobs_seen if j[0] == "Norme applicate ai fatti")
    assert flow.RULES_TITLE in norme[2] and "— da provare" in norme[2]          # the road at the separation: unknown
    objections = next(j for j in jobs_seen if j[0] == "Obiezioni e repliche")
    assert "OBIEZIONI DELLA CONTROPARTE DA TRATTARE" in objections[2]
    heads = [line for line in text.splitlines() if re.match(r"^\*\*\d+\. ", line)]
    assert heads == ["**1. Risposta e strategia**", "**2. Norme applicate ai fatti**", "**3. Obiezioni e repliche**",
                     "**4. Profili processuali**", "**5. Conclusioni**"]
    assert "RENDERED BLOCK" in text and "2043" not in text and "[1]" not in text
    assert "**Art. 1062 c.c.**" in text and "**1) La controparte potrà obiettare" in text
    # v59: 1061's block, emptied by tidy after assembly, is no bare heading -- named in the closing line
    assert "**Art. 1061 c.c.**" not in text and "Altre norme del tema non trattate: art. 1061 c.c." in text
    assert text.rstrip().endswith("Se vuoi, posso approfondire anche: 2. Passaggio coattivo; 3. Usucapione ordinaria di "
                                  "beni immobili. Puoi anche darmi un fatto nuovo, e aggiorno il parere, o chiedermi il "
                                  "documento.")
    assert state["answers"]["1"]


# --- through /api/chat ----------------------------------------------------------------------------------------------

@pytest.fixture
def chat(monkeypatch, tmp_path):
    for name, value in (("LLM_API_KEY", "k"), ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "k")):
        monkeypatch.setenv(name, value)
    import src.chatbot.api as api
    from fastapi.testclient import TestClient
    from src.db.base import get_db
    monkeypatch.setattr(api.chatbot, "_sessions", {})
    monkeypatch.setattr(api.chatbot, "_sessions_file", str(tmp_path / "sessions.json"))
    monkeypatch.setattr(api, "_classify_legal_intent", lambda message, lang: "defensive_draft")
    monkeypatch.setattr(api, "_is_topic_question", lambda message: False)
    calls = {"single_shot": 0}

    def single_shot(*args, **kwargs):
        calls["single_shot"] += 1
        return {"draft": "PARERE IN UNA VOLTA", "citations": []}
    import src.rag.defensive_generation as dg
    monkeypatch.setattr(dg, "run_defensive_pipeline", single_shot)
    api.app.dependency_overrides[get_db] = lambda: None
    try:
        yield api, TestClient(api.app), calls
    finally:
        api.app.dependency_overrides.pop(get_db, None)


CASE = "Leonardo è proprietario di un terreno; Domitilla nel 2016 ha chiuso il cancello sulla strada. Come agire?"


def test_with_the_flag_off_the_single_shot_pipeline_answers(chat, monkeypatch):
    api, client, calls = chat
    monkeypatch.setattr(flow, "start", lambda *a, **k: pytest.fail("the interactive flow ran with its flag off"))
    body = client.post("/api/chat", json={"message": CASE}).json()
    assert calls["single_shot"] == 1 and "PARERE IN UNA VOLTA" in body["answer"] and body["interaction"] is None


def test_with_the_flag_on_intent_defensive_draft_alone_takes_round_one(chat, monkeypatch):
    api, client, calls = chat           # v60: a "direct" route the legal intent reads as defensive_draft -- no single-shot
    monkeypatch.setenv("INTERACTIVE_FLOW", "true")
    monkeypatch.setattr(flow, "routes_here", lambda message: False)
    state = _state(monkeypatch)
    monkeypatch.setattr(flow, "start", lambda message, lang="it", request="": (*flow.round1(state), state))
    body = client.post("/api/chat", json={"message": CASE}).json()
    assert calls["single_shot"] == 0 and body["status_messages"] == ["interactive_round1"]


def test_with_the_flag_on_a_narrative_case_runs_in_rounds_kept_in_the_session(chat, monkeypatch):
    api, client, calls = chat
    monkeypatch.setenv("INTERACTIVE_FLOW", "true")
    monkeypatch.setattr(flow, "routes_here", lambda message: True)
    state = _state(monkeypatch)

    def start(message, lang="it", request=""):
        reply, field = flow.round1(state)
        return reply, field, state
    monkeypatch.setattr(flow, "start", start)
    monkeypatch.setattr(flow, "write_answer", lambda st, numbers: ("**1. Risposta e strategia**\n\nOK", []))
    one = client.post("/api/chat", json={"message": CASE}).json()
    assert calls["single_shot"] == 0 and one["status_messages"] == ["interactive_round1"]
    assert one["interaction"]["state"] == flow.TOPICS_OFFERED and one["interaction"]["topics"][0]["n"] == 1
    session = api.chatbot._sessions[one["session_id"]]
    assert flow.active(session.anchors)["state"] == flow.TOPICS_OFFERED
    two = client.post("/api/chat", json={"message": "2", "session_id": one["session_id"]}).json()
    assert two["interaction"]["state"] == flow.FRAME_RESTATED and two["interaction"]["chosen"] == [2]
    three = client.post("/api/chat", json={"message": "sì", "session_id": one["session_id"]}).json()
    assert three["interaction"]["state"] == flow.ANSWERED and three["answer"].startswith("**1. Risposta e strategia**")
    assert three["interaction"]["version"] > one["interaction"]["version"]
    assert [m.metadata.get("type") for m in session.messages] == ["interactive"] * 6


def test_a_failing_round_one_falls_back_to_the_single_shot_pipeline(chat, monkeypatch):
    api, client, calls = chat
    monkeypatch.setenv("INTERACTIVE_FLOW", "true")
    monkeypatch.setattr(flow, "routes_here", lambda message: True)

    def broken(*a, **k):
        raise RuntimeError("no graph")
    monkeypatch.setattr(flow, "start", broken)
    body = client.post("/api/chat", json={"message": CASE}).json()
    assert calls["single_shot"] == 1 and body["interaction"] is None


def test_a_narrative_case_starts_round_one_ahead_of_every_route_and_logs_it(chat, monkeypatch):
    api, client, calls = chat
    monkeypatch.setenv("INTERACTIVE_FLOW", "true")
    monkeypatch.setattr(flow, "routes_here", lambda message: True)
    monkeypatch.setattr(api, "_classify_legal_intent", lambda *a: pytest.fail("the router ran before round 1"))
    monkeypatch.setattr(api, "_detect_document_intent", lambda *a: pytest.fail("the document route ran before round 1"))
    state = _state(monkeypatch)
    monkeypatch.setattr(flow, "start", lambda message, lang="it", request="": (*flow.round1(state), state))
    logged = []
    monkeypatch.setattr(api, "vlog", lambda event, data=None, *a: logged.append((event, data)))
    body = client.post("/api/chat", json={"message": CASE}).json()
    assert calls["single_shot"] == 0 and body["status_messages"] == ["interactive_round1"]
    assert any(e == "interactive" and d.get("round") == 1 for e, d in logged)


def test_the_document_after_a_topic_answer_wraps_that_answer_never_a_fresh_single_shot(chat, monkeypatch):
    api, client, calls = chat
    monkeypatch.setenv("INTERACTIVE_FLOW", "true")
    monkeypatch.setattr(flow, "routes_here", lambda message: True)
    state = _state(monkeypatch)
    monkeypatch.setattr(flow, "start", lambda message, lang="it", request="": (*flow.round1(state), state))
    monkeypatch.setattr(flow, "write_answer", lambda st, numbers: ("**1. Risposta e strategia**\n\nIL PARERE", []))
    one = client.post("/api/chat", json={"message": CASE}).json()
    sid = one["session_id"]
    assert api._interactive_document(api.chatbot._sessions[sid], sid, "genera il documento", None, None) is None
    client.post("/api/chat", json={"message": "1", "session_id": sid})
    client.post("/api/chat", json={"message": "procedi", "session_id": sid})
    monkeypatch.setattr(flow, "start", lambda *a, **k: pytest.fail("round 1 again"))
    doc = client.post("/api/chat", json={"message": "genera il documento", "session_id": sid}).json()
    assert doc["status_messages"] == ["interactive_document"] and calls["single_shot"] == 0
    assert "IL PARERE" in doc["answer"] and doc["answer"].startswith("⚖️ **BOZZA DOCUMENTO**")
    assert api._asks_interactive_document("bozza") and api._asks_interactive_document("Mi prepari la bozza?")
    assert not api._asks_interactive_document(CASE + " " * 400)


NO_CLIENT = {**FACTS, "parties": [{"name": "Leonardo", "role": "proprietario"}, {"name": "Domitilla", "role": "proprietario"}]}
ROLE_AND_QUESTION = "Sei l'avvocato di Leonardo. Qual è l'atto giudiziario più opportuno?"


def _analysed(monkeypatch, seen=None):
    """flow.analyse_case stubbed: the test state, its parties naming no client (round 1 would ask «Chi assisti»)."""
    def analyse_case(text, request="", lang="it"):
        if seen is not None:
            seen.append((text, request))
        state = _state(monkeypatch, NO_CLIENT)
        state["text"], state["request"] = text, request
        return state
    monkeypatch.setattr(flow, "analyse_case", analyse_case)
    monkeypatch.setattr(flow, "_angle_topic", lambda *a: None)          # no corpus read
    assert any(q["id"] == "parte_assistita" for q in _state(monkeypatch, NO_CLIENT)["questions"])


def test_a_case_with_the_role_and_a_question_takes_round_one_with_the_client_and_the_angle(chat, monkeypatch):
    api, client, calls = chat           # v60 (journal 19:12:19): case + «Sei l'avvocato di X» + a question
    monkeypatch.setenv("INTERACTIVE_FLOW", "true")
    monkeypatch.setattr(flow, "routes_here", lambda message: True)
    _analysed(monkeypatch)
    body = client.post("/api/chat", json={"message": f"{CASE} {ROLE_AND_QUESTION}"}).json()
    assert calls["single_shot"] == 0 and body["status_messages"] == ["interactive_round1"]
    assert not body.get("generated_document_id")
    assert "Chi assisti" not in body["answer"] and "Mi chiedi: «Qual è l'atto giudiziario più opportuno?»" in body["answer"]
    state = flow.active(api.chatbot._sessions[body["session_id"]].anchors)
    assert state["answered"]["parte_assistita"] == "Leonardo" and state["analysis"]["parte_assistita"] == "Leonardo"
    assert "parte_assistita" not in [q["id"] for q in state["questions"]]
    assert state["angle"] == "Qual è l'atto giudiziario più opportuno?"
    assert body["answer"].startswith("Ho letto il caso di Leonardo.")


def test_the_angle_matching_a_topic_lists_it_first_and_procedi_takes_it(monkeypatch):
    state = _state(monkeypatch, NO_CLIENT)
    flow.apply_stated(state, flow.stated("Assisti Leonardo: come ottengo il passaggio coattivo?"))
    assert [t["title"] for t in state["topics"]][:1] == ["Passaggio coattivo"] and len(state["topics"]) == 3
    assert state["chosen"] == [1] and state["answered"]["parte_assistita"] == "Leonardo"
    reply, _ = flow.round1(state)
    assert "1. **Passaggio coattivo**" in reply and "se mi chiedi il parere, parto da lì" in reply
    assert flow.stated("Il caso è questo. Nessuna domanda.") == {"client": "", "client_role": "", "angle": "", "date": ""}


def test_an_uploaded_narrative_with_a_one_line_role_and_question_is_the_case_of_round_one(chat, monkeypatch, tmp_path):
    import asyncio
    from types import SimpleNamespace
    api, client, calls = chat           # v60: the file the case, the message the angle; the upload's text analysed
    monkeypatch.setenv("INTERACTIVE_FLOW", "true")
    upload = tmp_path / "caso.txt"
    upload.write_text(CASE, encoding="utf-8")
    doc = SimpleNamespace(storage_path=str(upload), id="d1", original_filename="caso.txt", document_role="document")
    seen, logged = [], []
    _analysed(monkeypatch, seen)
    monkeypatch.setattr(flow, "routes_here", lambda text: "cancello" in text)        # the case is in the file only
    monkeypatch.setattr(api, "_classify_legal_intent", lambda *a: "rag")
    monkeypatch.setattr(api, "vlog", lambda event, data=None, *a: logged.append((event, data)))
    session = api.chatbot.create_session()
    run = lambda intent: asyncio.run(api._interactive_upload_case(session, session.session_id, ROLE_AND_QUESTION, "it",
                                                                  doc, intent, lambda path: open(path).read()))
    body = run("analyse")
    assert body is not None and body.status_messages == ["interactive_round1"] and calls["single_shot"] == 0
    assert seen == [(CASE, ROLE_AND_QUESTION)]                                       # the upload analysed, the message its request
    entry = next(d for e, d in logged if e == "interactive" and d.get("round") == 1)
    assert entry["case_source"] == "upload+message" and entry["case_chars"] == len(CASE) + len(ROLE_AND_QUESTION)
    state = flow.active(session.anchors)
    assert state["answered"]["parte_assistita"] == "Leonardo" and state["angle"].startswith("Qual è l'atto")
    assert "Chi assisti" not in body.answer and not body.generated_document_id
    assert session.messages[0].metadata["document_name"] == "caso.txt"
    # not a case (a template to fill): the document branch goes on
    upload.write_text("Contratto di locazione tra ____ e ____.", encoding="utf-8")
    seen.clear()
    assert run("generate") is None and seen == []
    assert run("defensive_draft") is not None and seen                               # the branch's own classifier


def test_a_narrative_message_naming_an_upload_is_left_to_the_document_branch_then_round_one(chat, monkeypatch):
    import asyncio
    api, client, calls = chat           # v60: the file is the case -- the document branch reads it first
    monkeypatch.setenv("INTERACTIVE_FLOW", "true")
    monkeypatch.setattr(flow, "routes_here", lambda message: True)
    order, logged = [], []
    detect = api._detect_document_intent
    monkeypatch.setattr(api, "_detect_document_intent", lambda m: order.append("document branch") or detect(m))
    state = _state(monkeypatch)
    monkeypatch.setattr(flow, "start", lambda message, lang="it", request="": order.append("round 1") or
                        (*flow.round1(state), state))
    monkeypatch.setattr(api, "vlog", lambda event, data=None, *a: logged.append((event, data)))
    message = CASE + " Ti ho allegato il documento con i dettagli."
    assert api._refers_to_upload(message) and api._refers_to_upload("vedi «perizia.pdf»") and not api._refers_to_upload(CASE)
    monkeypatch.setattr(api, "enforce_tenant_product_access", lambda *a: None)
    user = {"sub": "00000000-0000-0000-0000-000000000001", "tenant_id": "00000000-0000-0000-0000-000000000002"}
    body = asyncio.run(api._chat_impl(api.ChatRequest(message=message), user, None))
    assert any(e == "interactive_deferred" for e, _ in logged)
    assert "round 1" in order and "document branch" in order[:order.index("round 1")]     # no file found: the message
    assert body.status_messages == ["interactive_round1"] and calls["single_shot"] == 0


# --- 6. the simulation ----------------------------------------------------------------------------------------------

def test_the_simulated_user_picks_the_topics_covering_the_most_expected_norms():
    spec = importlib.util.spec_from_file_location("diagnose_long", HERE.parent / "scripts" / "diagnose_long.py")
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)
    ev = dl._eval_module()
    expected = [{"article": "1051", "code": "cc"}, {"article": "1054", "code": "cc"}, {"article": "1062", "code": "cc"}]
    oracle = dl.oracle_topics(TOPICS, expected, ev)
    assert oracle["chosen"] == [2] and oracle["covered"][2] == ["art. 1051 cc", "art. 1054 cc"]
    assert dl.oracle_topics(TOPICS, [{"article": "2043", "code": "cc"}], ev)["chosen"] == [1]
    scored = dl.mentioned_score(ev, {"expected": expected}, "Si applica l'art. 1051 c.c. e l'art. 1062 c.c.")
    assert (scored["mentioned"], scored["expected"]) == (2, 3) and scored["missing"] == ["art. 1054 cc"]


def test_the_possession_question_reads_as_the_pos_template_for_h3(monkeypatch):
    facts = copy.deepcopy(FACTS)
    facts["possession"] = [{"who": "incaricati di Leonardo", "asset": "strada sterrata", "use": "passaggio", "since": ""}]
    facts["property_history"][0]["asset"] = "terreni di Leonardo e di Sempronio"
    facts["access"] = [{"asset": "terreni di Leonardo", "landlocked": "sì", "how_reached": "strada sterrata"}]
    state = _state(monkeypatch, facts)
    q = next(q for q in state["questions"] if q["family"] == "possesso_inizio")
    assert q["text"] == ("Per valutare l'usucapione: da quando Leonardo o i suoi danti causa usano la strada per raggiungere "
                         "i terreni? Il caso non indica l'anno in cui è iniziato.")
    assert [flow._with_article(n) for n in ("strada", "terreni", "fondo", "sentiero", "accesso", "corte")] == [
        "la strada", "i terreni", "il fondo", "il sentiero", "l’accesso", "la corte"]


def test_round_two_prints_the_full_parties_line(monkeypatch):
    state = _state(monkeypatch)
    from src.rag import render_computed
    monkeypatch.setattr(render_computed, "blocks", lambda analysis, party="": [
        ("Parti da convenire", "Parti da convenire: Domitilla; da valutare la chiamata di Sempronio quali proprietari dei "
                               "fondi interessati (litisconsorzio necessario ex art. 102 c.p.c.).")])
    state["chosen"] = [1]
    text, _ = flow.round2(state)
    assert ("Parti da convenire: Domitilla; da valutare la chiamata di Sempronio quali proprietari dei fondi interessati "
            "(litisconsorzio necessario ex art. 102 c.p.c.).") in text


def test_events_at_the_start_of_a_year_mediation_wording_and_other_profiles_by_capo():
    facts = [{"kind": "date", "source": "event", "date": "inizio del 1900", "what": "esiste la strada"},
             {"kind": "date", "source": "event", "date": "2016", "what": "Domitilla appone un cancello"}]
    assert flow.events_sentence(facts) == "All'inizio del 1900 esiste la strada; nel 2016 Domitilla appone un cancello."
    rendered = "**Mediazione**\n\nMediazione obbligatoria: condizione di procedibilità.\n\n**Termini**\n\nX."
    out = flow.with_copy_mediation(rendered, {"fact_signals": {"mediation": {"required": True}}})
    assert "improcedibilità" in out and "**Termini**\n\nX." in out
    lines = flow.other_profile_lines([
        {"title": "Proprietà e uso", "articles": ["1102 c.c."], "capo": "CAPO II – DELLA COMUNIONE"},
        {"title": "Uso della cosa", "articles": ["1114 c.c."], "capo": "CAPO II – DELLA COMUNIONE"},
        {"title": "Abuso", "articles": ["2814 c.c."], "capo": ""}])
    assert lines == ["- Capo II – Della comunione (artt. 1102, 1114 c.c.)", "- Abuso (artt. 2814 c.c.)"]


# --- v55: a rule fired on an unstated field, the act's notification, a table's question --------------------------------

def test_a_rule_fired_on_fields_the_case_does_not_state_is_to_verify_and_its_questions_are_asked(monkeypatch):
    from src.rag import topic_step
    facts = copy.deepcopy(FACTS)
    facts["works_or_signs"][0]["before_separation"] = "sì"           # h3 live: filled by the extraction
    facts["property_history"][0]["ended_by"] = "divisione"
    state = _state(monkeypatch, facts, H3_TEXT)
    analysis = flow._restored_analysis(state["analysis"])
    assert "destinazione_padre_famiglia" in {r["id"] for r in analysis["fact_signals"]["fired"]}
    assert flow.mark_unstated(analysis) == ["destinazione_padre_famiglia"]
    rule = next(r for r in analysis["fact_signals"]["fired"] if r["id"] == "destinazione_padre_famiglia")
    assert "opera_separazione" in rule["unstated"] and rule["unstated_conditions"]
    signals = analysis["fact_signals"]
    assert "padre di famiglia (art. 1062" not in fact_signals.conclusions_block(signals, analysis)   # never fixed
    block = flow.to_verify_block(analysis)
    assert block.startswith(flow.CONCLUSIONS_TO_VERIFY) and "Da verificare:" in block and "1062" in block
    inst = next(i for i in topic_step.institutions(analysis) if "padre di famiglia" in i["name"])
    assert inst["source"] == "regola da verificare" and inst["to_verify"] == rule["unstated_conditions"]
    assert {"opera_separazione_0", "atto_separazione"} <= {q["id"] for q in state["questions"]}
    by = {q["id"]: q for q in state["questions"]}             # the client answers both: the rule is fixed again
    flow._apply_answer(signals["facts"], by["opera_separazione_0"], "sì")
    flow._apply_answer(signals["facts"], by["atto_separazione"], "divisione nel 1950")
    assert flow.mark_unstated(analysis) == []
    stated = H3_TEXT + (" Nel 1950 Mevio divise i terreni tra Leonardo e Sempronio, e la strada esisteva già al momento "
                        "della divisione.")
    analysis = flow._restored_analysis(_state(monkeypatch, facts, stated)["analysis"])
    assert flow.mark_unstated(analysis) == []                 # the case states both: a fixed conclusion


def test_a_topic_of_a_rule_to_verify_says_so_in_its_line(monkeypatch):
    state = _state(monkeypatch)
    topic = state["topics"][0]
    assert flow._decides(topic).startswith("da verificare: due fondi di un unico proprietario")
    assert not flow._decides({**topic, "source": "regola destinazione_padre_famiglia"}).startswith("da verificare")


ADMIN = {"parties": [{"name": "Tizio", "role": "cliente"}, {"name": "Comune di Siena", "role": "controparte, comune"}],
         "client_side": "Tizio", "events": [{"date": "", "what": "il Comune notifica a Tizio il provvedimento di diniego",
                                             "kind": "provvedimento"}], "_regimes": True}


def test_an_open_tar_term_asks_the_acts_notification_and_the_answer_runs_the_term(monkeypatch):
    monkeypatch.setenv("FACT_SIGNALS_TODAY", TODAY)
    facts = copy.deepcopy(ADMIN)
    terms = fact_signals.compute_terms(facts, ["amministrativo"])
    tar = next(t for t in terms if t["id"] == "ricorso_tar")
    assert tar["status"] in flow.OPEN_TERM_STATUS
    analysis = {"frame": {"areas": ["amministrativo"]}, "fact_signals": {"facts": facts, "terms": terms}}
    topics = [{"title": "Ricorso al TAR per l'annullamento", "anchor": "29 c.p.a.", "articles": ["29 c.p.a."],
               "source": "termine ricorso_tar"}]
    q = next(q for q in flow.questions(analysis, (), topics, []) if q["family"] == "atto_notifica")
    assert q["text"] == ("In quale data è stato notificato (o pubblicato) il provvedimento che intendi impugnare? Da "
                         "questa data decorre il termine per il ricorso al TAR.")
    assert q["target"] == ["events", 0, ["date"]]
    flow._apply_answer(facts, q, "20 ottobre 2026")
    after = next(t for t in fact_signals.compute_terms(facts, ["amministrativo"]) if t["id"] == "ricorso_tar")
    assert after["status"] == "NON DECORSO"
    closed = {"frame": {"areas": ["amministrativo"]}, "fact_signals": {"facts": facts, "terms": [after]}}
    assert not [q for q in flow.questions(closed, (), topics, []) if q["family"] == "atto_notifica"]   # no open term


def test_a_table_topics_question_family_is_asked_until_its_date_is_known(monkeypatch):
    monkeypatch.setenv("FACT_SIGNALS_TODAY", TODAY)
    facts = {"parties": [{"name": "Tizio", "role": "cliente, indagato"}], "client_side": "Tizio",
             "events": [{"date": "", "what": "eseguita l'ordinanza di custodia cautelare in carcere", "kind": "altro"}]}
    analysis = {"frame": {"areas": ["penale"]}, "fact_signals": {"facts": facts, "terms": []}}
    topics = [{"title": "Custodia cautelare in carcere", "anchor": "285 c.p.p.", "articles": ["285 c.p.p.", "309 c.p.p."],
               "source": "tabella misure_cautelari", "questions": ["data_misura"]}]
    q = next(q for q in flow.questions(analysis, (), topics, []) if q["family"] == "data_misura")
    assert q["text"] == flow.C.QUESTIONS["data_misura"] and q["target"] == ["events", 0, ["date"]]
    facts["events"][0]["date"] = "3 novembre 2026"
    assert not [q for q in flow.questions(analysis, (), topics, []) if q["family"] == "data_misura"]


# --- v56: §2 by the plan, the clean text, the gates, the unanswered questions, the new tables ------------------------

ART_52 = ("Non è punibile chi ha commesso il fatto, per esservi stato costretto dalla necessità di difendere un diritto "
          "proprio od altrui contro il pericolo attuale di un'offesa ingiusta, sempre che la difesa sia proporzionata "
          "all'offesa.\nNei casi previsti dall'articolo 614, primo e secondo comma, sussiste sempre il rapporto di "
          "proporzione di cui al primo comma del presente articolo se taluno legittimamente presente in uno dei luoghi "
          "ivi indicati usa un'arma legittimamente detenuta o altro mezzo idoneo al fine di difendere: a) la propria o la "
          "altrui incolumità; b) i beni propri o altrui, quando non vi è desistenza e vi è pericolo d'aggressione. [12]\n"
          "Note: comma aggiunto dalla l. 59/2006.")
ART_42 = ("Nessuno può essere punito per un'azione od omissione preveduta dalla legge come reato, se non l'ha commessa con "
          "coscienza e volontà.")


def test_every_planned_article_gets_its_block_and_each_paragraph_the_article_it_cites():
    from src.rag import rule_quote
    analysis = {"frame": {"areas": ["penale"]}, "fact_signals": {"facts": {"parties": []}, "terms": []}}
    rows = [_row("42", ART_42, "c.p."), _row("52", ART_52, "c.p.")]
    refs = [("42", "c.p."), ("52", "c.p."), ("56", "c.p.")]
    plan = {"questioni": [{"norme": [("42", "c.p."), ("52", "c.p."), ("56", "c.p.")]}]}
    assert flow.section_refs(plan, rows, refs) == [("42", "c.p."), ("52", "c.p."), ("56", "c.p.")]
    rules = rule_quote.article_rules(plan, rows, analysis, [("42", "c.p."), ("52", "c.p.")])
    writer = ("[1] Tizio ha agito con coscienza e volontà (art. 42 c.p.).\n\n"
              "La difesa era proporzionata ai sensi dell'art. 52 c.p.\n\n"
              "Non c'era «desistenza e vi è pericolo d'aggressione» nel momento della reazione.\n\n"
              "Il tentativo resta escluso: l'art. 56 c.p. richiede atti idonei.")
    log = []
    text = flow.assemble_norms(writer, rules, analysis, rows, log, flow.section_refs(plan, rows, refs),
                               {("56", "c.p."): "Delitto tentato"})
    at = {h: text.index(h) for h in ("**Art. 42 c.p.", "**Art. 52 c.p.")}
    assert at["**Art. 42 c.p."] < text.index("coscienza e volontà (art. 42") < at["**Art. 52 c.p."]
    assert at["**Art. 52 c.p."] < text.index("proporzionata ai sensi") < text.index("Non c'era «desistenza")
    # v58: 56 c.p. has no text read: no block, its paragraph dropped, one closing line
    assert "Il tentativo resta escluso" not in text and "**Art. 56" not in text
    assert text.endswith("Altre norme del tema non trattate: art. 56 c.p.")
    assert log[0]["unplaced"] == 0 and log[0]["dropped_no_text"] == {"56 c.p.": 1}


def test_the_article_text_loses_its_footnotes_and_notes_and_definitions_have_no_requisites():
    from src.rag import rule_quote
    assert rule_quote.clean_rubrica("Disposizioni generali sul giudizio di ottemperanza [196]") == \
        "Disposizioni generali sul giudizio di ottemperanza"
    assert rule_quote.clean_rubrica("Truffa (1) (2) (3)") == "Truffa"
    commi = rule_quote.row_commi(_row("52", ART_52, "c.p."))
    assert len(commi) == 2 and "[12]" not in commi[1][1] and "Note" not in commi[1][1]
    reqs = rule_quote.requisites(commi[1][1])
    assert any("legittimamente detenuta" in r for r in reqs) and any("desistenza" in r for r in reqs)
    nullity = ("È nullo il provvedimento amministrativo che manca degli elementi essenziali, che è viziato da difetto "
               "assoluto di attribuzione, che è stato adottato in violazione o elusione del giudicato; nonché negli altri "
               "casi espressamente previsti dalla legge.")
    assert rule_quote.requisites(nullity) == []                   # a definition: no quando / se / purché / salvo
    long_clause = "Il fatto è punito quando " + "l'agente agisce con modalità particolarmente gravi e reiterate " * 5 + "."
    assert rule_quote.requisites(long_clause) == []               # longer than REQUISITE_CHARS


def test_civil_tools_only_in_a_civil_main_area():
    from src.rag import render_computed
    tool = {"id": "accertamento_preventivo", "refs": [("696", "c.p.c.")], "text": "Accertamento tecnico preventivo.",
            "why": "", "matched": []}
    for areas, shown in ((["penale"], False), (["amministrativo", "contratti"], False), (["contratti"], True), ([], True)):
        titles = [t for t, _ in render_computed.blocks({"frame": {"areas": areas}, "fact_signals": {"tools": [tool]}})]
        assert ("Strumenti processuali" in titles) == shown, areas


def test_an_unanswered_question_is_not_known_to_the_writer_and_its_assertion_is_marked(monkeypatch):
    state = _state(monkeypatch, None, H3_TEXT)
    families = {q["family"] for q in flow.unanswered(state)}
    assert {"possesso_inizio", "atto_separazione"} <= families
    block = flow.unknown_block(state)
    assert block.startswith(flow.C.UNKNOWN_TITLE) and "inizio del possesso: non noto" in block
    state["answered"] = {q["id"]: "x" for q in state["questions"] if q["family"] == "accesso_alternativo"}
    sections = [("Risposta e strategia", "Leonardo usa la strada fin dal 1900. Il passaggio non sarebbe dovuto senza "
                                         "alcuna indennità. Se il possesso è iniziato nel 1990, l'usucapione è maturata."),
                ("Profili processuali", "Leonardo usa la strada fin dal 1900.")]
    log = []
    out = dict(flow.unanswered_out(sections, state, log))
    first = out["Risposta e strategia"]
    assert "fin dal 1900 (da verificare: inizio del possesso)." in first
    assert "senza alcuna indennità (da verificare: atto e anno della separazione dei fondi)." in first
    assert "Se il possesso è iniziato nel 1990, l'usucapione è maturata." in first          # already conditional
    assert out["Profili processuali"] == "Leonardo usa la strada fin dal 1900."            # the rendered section
    assert len(log) == 2


def test_legittima_difesa_asks_52s_requisites_and_an_answer_decides_them(monkeypatch):
    from src.rag import topic_tables
    monkeypatch.setenv("TOPIC_TABLES", "true")
    case = {"frame": {"central_issue_candidates": [{"nome": "Eccesso colposo di legittima difesa"}]}}
    assert "legittima_difesa" not in {t["id"] for t in topic_tables.matches(case, ["penale"])}     # v58: no aggression
    case["fact_signals"] = {"facts": {"events": [{"what": "un ladro si introduce in casa e Tizio spara"}]}}
    found = {t["id"]: t for t in topic_tables.matches(case, ["penale"])}
    assert found["legittima_difesa"]["refs"][:2] == [("55", "c.p."), ("52", "c.p.")]
    assert {("575", "c.p."), ("589", "c.p.")} <= set(found["legittima_difesa"]["refs"])
    facts = {"parties": [{"name": "Tizio", "role": "cliente, indagato"}], "client_side": "Tizio", "events": []}
    analysis = {"frame": {"areas": ["penale"]}, "fact_signals": {"facts": facts, "terms": []}}
    topic = {"title": "Legittima difesa", "anchor": "52 c.p.", "articles": ["52 c.p."], "source": "frame"}
    qs = {q["family"]: q for q in flow.questions(analysis, (), [topic], [_row("52", ART_52, "c.p.")])}
    assert {"arma_detenuta", "desistenza", "pericolo_aggressione"} <= set(qs)          # from 52 co. 2's own requisites
    assert qs["arma_detenuta"]["text"] == "L'arma (o il mezzo) usata per difendersi era legittimamente detenuta da Tizio?"
    flow._apply_answer(facts, qs["desistenza"], "sì")                  # the aggressor had desisted
    flow._apply_answer(facts, qs["pericolo_aggressione"], "sì")
    assert flow.requisite_status("non vi è desistenza e vi è pericolo d'aggressione", analysis) == "non risulta"


def test_ottemperanza_and_the_elusion_of_the_judgment_anchor_on_21_septies_and_ask_the_acts_date(monkeypatch):
    from src.rag import topic_tables
    found = {t["id"]: t for t in topic_tables.matches(
        {"frame": {"central_issue_candidates": [{"nome": "Elusione del giudicato"}]}}, ["amministrativo"])}
    t = found["ottemperanza"]
    assert t["refs"][0] == ("21-septies", "Legge sul Procedimento Amministrativo 2026")
    assert {("112", "c.p.a."), ("114", "c.p.a."), ("31", "c.p.a.")} <= set(t["refs"]) and t["questions"] == ["data_atto_elusivo", "data_giudicato"]
    facts = {"parties": [{"name": "Tizio", "role": "cliente"}], "client_side": "Tizio",
             "events": [{"date": "", "what": "il Comune adotta un nuovo provvedimento di diniego", "kind": "provvedimento"}]}
    analysis = {"frame": {"areas": ["amministrativo"]}, "fact_signals": {"facts": facts, "terms": []}}
    topic = {"title": "Ottemperanza", "anchor": "112 c.p.a.", "articles": ["112 c.p.a."], "source": "tabella ottemperanza",
             "questions": ["data_atto_elusivo"]}
    q = next(q for q in flow.questions(analysis, (), [topic], []) if q["family"] == "data_atto_elusivo")
    assert q["target"] == ["events", 0, ["date"]] and "180 giorni" in q["text"]


def test_no_lone_closing_and_a_party_who_sold_is_no_owner(monkeypatch):
    state = _state(monkeypatch)
    facts = flow._facts(flow._restored_analysis(state["analysis"]))
    facts["parties"].append({"name": "Sempronio", "role": "proprietario"})
    facts["events"].append({"date": "2003", "what": "Sempronio vende il terreno a Domitilla", "kind": "vendita"})
    lines = {f["name"]: f["text"] for f in flow.known_facts({"fact_signals": {"facts": facts, "terms": []}})
             if f["kind"] == "party"}
    assert lines["Sempronio"] == "Sempronio (ex proprietario)" and "Domitilla (controparte, proprietario)" == lines["Domitilla"]


def test_with_no_other_topic_the_answer_has_no_closing(monkeypatch):
    from src.rag import plan_then_write as ptw, render_computed
    state = _state(monkeypatch)
    state["topics"] = state["topics"][:1]
    monkeypatch.setattr(flow, "_documents", lambda: [{"id": "LEGAL_DOC::cc", "name": "Codice Civile 2026"}])
    monkeypatch.setattr(ptw, "make_plan", lambda *a, **k: {
        "parte_assistita": "Leonardo", "posizione": "", "obiettivo": "", "questione_centrale": "", "risposte": [],
        "subordinate": [], "punti_deboli": [], "questioni": [{"titolo": "T", "norme": [("1062", "c.c.")], "pronunce": [],
                                                              "argomento": "", "obiezione": "", "replica": ""}]})
    monkeypatch.setattr(render_computed, "render", lambda analysis, party="": "")
    monkeypatch.setattr(ptw, "write_sections", lambda plan, jobs, *a, **k: [(t, f"[1] Testo di {t}.") for t, *_ in jobs])
    text, _ = flow.write_answer(state, [1])
    assert "Oppure dimmi" not in text and "Se vuoi, posso approfondire" not in text


# --- v57: earned glosses, the cap across topics, the placement by words, requisites, scaffolding, roles, terms, §4 --

def test_only_the_matched_rule_or_named_articles_are_applied_and_the_others_get_their_heading(monkeypatch):
    from src.rag import rule_quote
    analysis = {"frame": {"areas": ["penale"]}, "fact_signals": {"facts": {"parties": []}, "terms": [], "fired": []}}
    rows = [_row("42", ART_42, "c.p."), _row("52", ART_52, "c.p.")]
    refs = [("42", "c.p."), ("52", "c.p.")]
    topics = [{"title": "Legittima difesa", "anchor": "52 c.p.", "matched": ["52 c.p."], "articles": ["52 c.p.", "42 c.p."]}]
    glossed = flow.earned_refs(topics, analysis, "Tizio spara al ladro.", refs)
    assert glossed == [("52", "c.p.")]                                   # 42: neither matched, nor a rule's, nor named
    assert flow.earned_refs(topics, analysis, "Si applica l'art. 42 c.p.?", refs) == refs
    plan = {"questioni": [{"norme": refs}]}
    rules = rule_quote.article_rules(plan, rows, analysis, refs)
    log = []
    text = flow.assemble_norms("[1] La difesa era proporzionata (art. 52 c.p.).\n\nTizio agì con coscienza (art. 42 c.p.).",
                               rules, analysis, rows, log, refs, {}, glossed)
    assert "La difesa era proporzionata" in text and "con coscienza" not in text and "**Art. 42 c.p." not in text
    assert text.endswith("Altre norme del tema non trattate: art. 42 c.p.")      # v58: never a bare heading
    assert log[0]["dropped_unearned"] == {"42 c.p.": 1}


def test_the_cap_holds_across_the_chosen_topics_matched_articles_first():
    topics = [{"anchor": f"{a} c.p.", "matched": [f"{a} c.p."], "articles": [f"{a + i} c.p." for i in range(8)]}
              for a in (100, 200, 300)]
    refs = flow.capped_refs(topics)
    assert len(refs) == 8 and refs[:3] == [("100", "c.p."), ("200", "c.p."), ("300", "c.p.")]
    assert refs[3:6] == [("101", "c.p."), ("201", "c.p."), ("301", "c.p.")]      # then in turn


def test_a_paragraph_citing_only_a_comma_goes_to_the_article_whose_comma_holds_its_words():
    from src.rag import rule_quote
    analysis = {"frame": {"areas": ["penale"]}, "fact_signals": {"facts": {"parties": []}, "terms": []}}
    rows = [_row("42", ART_42, "c.p."), _row("52", ART_52, "c.p.")]
    refs = [("42", "c.p."), ("52", "c.p.")]
    rules = rule_quote.article_rules({"questioni": [{"norme": refs}]}, rows, analysis, refs)
    writer = ("[1] Tizio agì con coscienza e volontà (art. 42 c.p.).\n\n"
              "Ai sensi del comma 2, taluno legittimamente presente usa un'arma legittimamente detenuta al fine di "
              "difendere la propria incolumità.")
    text = flow.assemble_norms(writer, rules, analysis, rows, [], refs, {}, refs)
    assert text.index("**Art. 52 c.p.") < text.index("Ai sensi del comma 2")         # never under art. 42


def test_requisites_skip_definitions_penalties_and_fragments_and_no_placeholder_line():
    from src.rag import rule_quote
    art_43 = ("Il delitto: è doloso, o secondo l'intenzione, quando l'evento dannoso o pericoloso è dall'agente preveduto "
              "e voluto; è colposo, o contro l'intenzione, quando l'evento non è voluto.")
    assert rule_quote.requisites(art_43) == []                          # a definition, though it says «quando»
    assert rule_quote.requisites("Il giudice condanna al pagamento, salvo che ciò sia manifestamente iniquo, e se "
                                 "l'amministrazione non provvede entro il termine fissato.") == \
        ["l'amministrazione non provvede entro il termine fissato"]       # «…iniquo, e»: cut on a conjunction
    assert rule_quote.requisites("Se il fatto è commesso con violazione delle norme, la pena è della reclusione da due a "
                                 "sette anni.") == []                                 # v58: an aggravante, never a requisite
    analysis = {"fact_signals": {"facts": {"parties": []}, "terms": []}}
    rule = {"ref": ("575", "c.p."), "commi": [(None, "Chiunque cagiona la morte di un uomo è punito con la reclusione "
                                                     "non inferiore ad anni ventuno.")], "selected": [None], "how": "all",
            "requisites": {}}
    assert flow.norm_blocks([rule], analysis, []) == ["**Art. 575 c.p.**"]       # no «- testo: da provare.»


def test_label_only_lines_and_rule_glosses_leave_the_answer():
    assert flow.scaffolding_out("[3] art. 589 c.p.\nL'omicidio fu colposo.") == "[3] \nL'omicidio fu colposo."
    assert flow.scaffolding_out("art. 21-septies Legge sul Procedimento Amministrativo 2026\nTesto.") == "Testo."
    assert flow.scaffolding_out("art. 52 c.p. esclude la punibilità.") == "art. 52 c.p. esclude la punibilità."


def test_one_client_never_the_other_side_and_two_candidates_ask_first():
    out = {"client_side": "Caio", "parties": [{"name": "Tizio", "role": "cliente, privato"},
                                              {"name": "Caio", "role": "cliente, controparte, privato"}]}
    fact_signals.one_client(out, "")
    assert [p["role"] for p in out["parties"]] == ["privato", "cliente, privato"]          # v56-h10
    out = {"client_side": "", "parties": [{"name": "Tizio", "role": "cliente"}, {"name": "Caio", "role": "cliente"}]}
    fact_signals.one_client(out, "Caio si rivolge all'avvocato. Tizio lo segue.")
    assert [p["role"] for p in out["parties"]] == ["privato", "cliente"]
    out = {"client_side": "", "parties": [{"name": "Tizio", "role": "cliente"}, {"name": "Caio", "role": "cliente"}]}
    fact_signals.one_client(out, "Tizio e Caio litigano.")
    assert out["client_ambiguous"] == ["Tizio", "Caio"]
    analysis = {"frame": {"areas": ["civile"]}, "fact_signals": {"facts": out, "terms": []}}
    q = flow.questions(analysis, (), [], [])[0]
    assert q["id"] == "parte_assistita" and q["text"] == "Chi assisti in questo caso: Tizio o Caio?"
    flow._apply_answer(out, q, "assisto Caio")
    assert [p["role"] for p in out["parties"]] == ["privato", "cliente, privato"] and "client_ambiguous" not in out


def test_a_comma_the_article_does_not_have_moves_to_the_article_holding_it():
    from src.rag import rule_quote
    analysis = {"frame": {"areas": ["amministrativo"]}, "fact_signals": {"facts": {"parties": []}, "terms": []}}
    art_21s = ("1. È nullo il provvedimento amministrativo che è stato adottato in violazione o elusione del giudicato.\n"
               "2. Le questioni inerenti alla nullità sono attribuite alla giurisdizione esclusiva del giudice amministrativo.")
    art_31 = ("1. Decorsi i termini, chi vi ha interesse può chiedere l'accertamento dell'obbligo di provvedere.\n2. Il "
              "giudice può pronunciare sulla fondatezza della pretesa.\n3. Si applica il rito.\n4. La domanda volta "
              "all'accertamento delle nullità previste dalla legge si propone entro il termine di decadenza di "
              "centottanta giorni.")
    rows = [_row("112", art_21s, "c.p.a."), _row("31", art_31, "c.p.a.")]       # two commi: never a «co. 4»
    refs = [("112", "c.p.a."), ("31", "c.p.a.")]
    rules = rule_quote.article_rules({"questioni": [{"norme": refs}]}, rows, analysis, refs)
    writer = ("[1] Il provvedimento è nullo (art. 112 c.p.a.).\n\n«co. 4» — Ai sensi dell'art. 112 c.p.a., la "
              "domanda volta all'accertamento delle nullità si propone entro il termine di decadenza di centottanta "
              "giorni.")
    log = []
    text = flow.assemble_norms(writer, rules, analysis, rows, log, refs, {}, refs)
    assert text.index("**Art. 31 c.p.a.") < text.index("«co. 4»") and log[0]["commi"][0]["to"] == "31 c.p.a."


def test_the_judgments_two_terms_are_computed(monkeypatch):
    monkeypatch.setenv("FACT_SIGNALS_TODAY", TODAY)
    facts = {"parties": [{"name": "Caio", "role": "cliente"}, {"name": "Comune di C.", "role": "pubblica amministrazione"}],
             "events": [{"date": "", "what": "la sentenza del TAR annulla la graduatoria", "kind": "altro"},
                        {"date": "", "what": "il Comune adotta un nuovo provvedimento che riconferma Tizio",
                         "kind": "provvedimento"}], "_regimes": True}
    terms = {t["id"]: t for t in fact_signals.compute_terms(facts, ["amministrativo"])}
    assert terms["nullita_giudicato"]["status"] == "DA VERIFICARE" and terms["nullita_giudicato"]["ref"] == "art. 31 c.p.a."
    assert terms["ottemperanza"]["status"] == "DA VERIFICARE" and terms["ottemperanza"]["length"] == "10 anni"
    facts["events"][1]["date"] = "1 ottobre 2026"
    after = {t["id"]: t for t in fact_signals.compute_terms(facts, ["amministrativo"])}
    assert after["nullita_giudicato"]["status"] == "NON DECORSO"           # 180 days from 1 October


def test_a_criminal_case_has_its_procedural_section_and_the_answer_its_cosmetics():
    from src.rag import render_computed
    analysis = {"frame": {"areas": ["penale"], "central_issue_candidates": [{"nome": "Legittima difesa"}]},
                "fact_signals": {"facts": {"events": [{"what": "il ladro fugge e Tizio spara", "date": ""}]}}}
    blocks = dict(render_computed.blocks(analysis))
    body = blocks[render_computed.PROCEDURE_TITLE]
    assert "Omicidio (art. 575 c.p.): procedibile d'ufficio; competenza: Corte d'assise (art. 5 c.p.p.)." in body
    assert "Omicidio colposo (art. 589 c.p.): procedibile d'ufficio." in body
    assert "Data del fatto: non nota. Da essa decorre la prescrizione del reato (art. 157 c.p.)." in body
    log = []
    out = flow.cosmetic_fixes([("Risposta e strategia", "Conviene agire in via subordinata, ma prioritariamente, per "
                                "l'accertamento."), ("Conclusioni", "I termini vanno verificati, come indicato nella "
                                "sezione «Profili processuali e termini».")], list(flow.ANSWER_TITLES), log)
    assert "agire in via principale, per" in out[0][1]
    # v60f (P9): no name repaired after writing -- the writer is told the real ones (test_v60f)
    assert out[1][1].endswith("nella sezione «Profili processuali e termini».")


# --- v58: from the v57 transcripts --------------------------------------------------------------------------------

ART_640 = ("Chiunque, con artifizi o raggiri, inducendo taluno in errore, procura a sé o ad altri un ingiusto profitto con "
           "altrui danno, è punito con la reclusione da sei mesi a tre anni e con la multa da euro 51 a euro 1.032.\n"
           "La pena è della reclusione da uno a cinque anni: 1) se il fatto è commesso a danno dello Stato (8); 2-bis) se "
           "il fatto è commesso in presenza della circostanza di cui all'articolo 61, numero 5] (6) 2-ter.")


def test_an_offence_s_requisites_are_its_first_comma_s_elements_never_its_aggravanti():
    from src.rag import rule_quote
    rules = rule_quote.article_rules({"questioni": [{"norme": [("640", "c.p.")]}]}, [_row("640", ART_640, "c.p.")],
                                     {"fact_signals": {}}, [("640", "c.p.")])
    reqs = [r for v in rules[0]["requisites"].values() for r in v]
    assert reqs == ["con artifizi o raggiri", "inducendo taluno in errore",
                    "procura a sé o ad altri un ingiusto profitto con altrui danno"]          # v57-p2
    assert rules[0]["selected"] == [rules[0]["commi"][0][0]]
    assert "(8)" not in rule_quote.clean_text(ART_640) and "]" not in rule_quote.clean_text(ART_640)


def test_a_sentence_citing_a_comma_the_article_does_not_have_is_dropped():
    rules = [{"ref": ("628", "c.p."), "commi": [(1, "a"), (2, "b"), (3, "c")]}]
    text = ("Non è provata la sottrazione. Né la violenza, come richiesto dal comma 9 dell'art. 628 c.p. Il co. 2 "
            "dell'art. 628 c.p. vale per la rapina impropria.")
    out = flow.phantom_commi_out([("Norme applicate ai fatti", text)], rules, [])[0][1]
    assert "comma 9" not in out and "Non è provata la sottrazione." in out and "Il co. 2" in out          # v57-h5


def test_with_no_client_tagged_the_first_question_asks_who_it_is():
    facts = {"parties": [{"name": "Signora Bianchi", "role": "proprietario"}, {"name": "Amministratore", "role": "privato"},
                         {"name": "Comune", "role": "pubblica amministrazione"}]}
    q = flow.questions({"frame": {"areas": ["civile"]}, "fact_signals": {"facts": facts, "terms": []}}, (), [], [])[0]
    assert q["id"] == "parte_assistita" and q["text"] == "Chi assisti in questo caso: Signora Bianchi o l'amministratore?"   # v60h (9): a role


def test_the_quoted_comma_beats_the_cited_numbers():
    from src.rag import rule_quote
    art_320 = "Le disposizioni degli articoli 318 e 319 si applicano anche all'incaricato di un pubblico servizio."
    art_318 = ("Il pubblico ufficiale che, per l'esercizio delle sue funzioni, indebitamente riceve denaro o ne accetta la "
               "promessa è punito con la reclusione da tre a otto anni.")
    rows = [_row("320", art_320, "c.p."), _row("318", art_318, "c.p.")]
    refs = [("320", "c.p."), ("318", "c.p.")]
    rules = rule_quote.article_rules({"questioni": [{"norme": refs}]}, rows, {"fact_signals": {}}, refs)
    writer = ("[2] «Le disposizioni degli articoli 318 e 319 si applicano anche all'incaricato di un pubblico servizio» "
              "(co. 1): la norma estende gli artt. 318 e 319 c.p. a Tizio.")
    text = flow.assemble_norms(writer, rules, {"fact_signals": {}}, rows, [], refs, {}, refs)
    assert text.index("**Art. 320 c.p.") < text.index("«Le disposizioni") < text.index("**Art. 318 c.p.")   # v57-h7


def test_a_rule_institutions_resolved_article_earns_no_gloss():
    analysis = {"fact_signals": {"fired": [{"id": "condominio_parti_comuni", "requires": [], "conclusion": ""}]},
                "rule_institutions": [{"nome": "uso delle parti comuni", "refs": [("1122-ter", "c.c.")]}]}
    topics = [{"anchor": "1130 c.c.", "matched": ["1130 c.c."]}]
    assert flow.earned_refs(topics, analysis, "", [("1130", "c.c."), ("1122-ter", "c.c.")]) == [("1130", "c.c.")]


def test_usucapione_only_with_a_possession_over_time():
    from src.rag import topic_step
    owner = {"parties": [{"name": "Signora Bianchi", "role": "proprietario"}],
             "possession": [{"who": "Signora Bianchi", "asset": "appartamento", "since": ""}]}
    assert not topic_step.possession_over_time(owner)                                   # v57-l3
    road = {"parties": [{"name": "Leonardo", "role": "cliente, proprietario"}],
            "possession": [{"who": "Leonardo", "asset": "strada sterrata", "since": ""}]}
    assert topic_step.possession_over_time(road)                                       # h3: a way over the neighbours'
    terms = [{"id": "usucapione_ordinaria", "label": "usucapione", "ref": "art. 1158 c.c.", "status": "DA VERIFICARE"}]
    assert topic_step.term_institutions({"facts": owner, "terms": terms}) == []
    assert topic_step.term_institutions({"facts": road, "terms": terms})


def test_a_criminal_answer_always_has_its_procedural_section():
    from src.rag import render_computed
    analysis = {"frame": {"areas": ["penale"]}, "fact_signals": {"facts": {"events": []}},
                "offences": [("320", "Corruzione di persona incaricata di un pubblico servizio (1)")]}
    body = dict(render_computed.blocks(analysis))[render_computed.PROCEDURE_TITLE]
    assert "- Corruzione di persona incaricata di un pubblico servizio (1) (art. 320 c.p.): procedibile d'ufficio; " \
           "competenza: tribunale." in body                                                     # v57-h7


def test_an_altro_profilo_needs_its_capo_and_the_case_to_share_its_words():
    from src.rag import topic_step
    links = topic_step.CsvLinks([("c.c.", "CAPO", "CAPO IV – DELLE IPOTECHE", "2833")], [],
                                {("2833", "c.c."): "Responsabilità per mancata iscrizione."})
    topic = {"title": "Responsabilità per danni all'immobile", "anchor": "2833 c.c.", "articles": ["2833 c.c."],
             "capo": "CAPO IV – DELLE IPOTECHE"}
    assert not topic_step.relevant_other(topic, links, {}, {"umidit", "colonn", "condom"})          # v57-l3
    ok = {"title": "Danno emergente e lucro cessante", "anchor": "1223 c.c.", "articles": ["1223 c.c."], "capo": ""}
    links = topic_step.CsvLinks([], [], {("1223", "c.c."): "Risarcimento del danno emergente e lucro cessante."})
    assert topic_step.relevant_other(ok, links, {}, topic_step._content_stems("il lucro cessante dell'affitto perso"))


def test_a_sentence_cut_at_a_removed_citation_is_dropped_whole():
    text = ("La signora Bianchi può agire. , che attribuisce al condominio la responsabilità per i danni.\n"
            "art. 1130 c.c. impone la conservazione.\n- a) il primo punto.")
    out = flow.headless_out([("Altri profili", text)], [])[0][1]                                   # v57-l3 §4
    assert ", che attribuisce" not in out and "La signora Bianchi può agire." in out
    assert "art. 1130 c.c. impone" in out and "a) il primo punto" in out
