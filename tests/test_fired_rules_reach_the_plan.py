"""Invented cases: a chain of owners read when the fields are empty, a fired rule's articles in the plan, the writer's rules."""
import os

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

from src.rag import fact_signals as fs  # noqa: E402
import src.rag.plan_then_write as ptw  # noqa: E402

CASE = ("I terreni di Bruno e Carla appartenevano a Dario. Una strada visibile li attraversava già. Bruno ha il terreno a "
        "nord, Carla quello a sud, venduto nel 2003 a Elsa.")


def history(*entries):
    return [dict(zip(("asset", "owner", "together_with", "ended_by", "how_acquired"), (*e, "", "", "")[:5])) for e in entries]


def infer(*entries):
    items = history(*entries)
    fs._infer_transfers(items)
    return items


def test_a_joint_entry_then_parts_of_different_owners_is_a_division():
    items = infer(("terreni di Bruno e Carla", "Dario"), ("terreno a nord", "Bruno"), ("terreno a sud", "Carla"),
                  ("terreno a sud", "Elsa"))
    assert items[0]["ended_by"] == "divisione" and items[0]["ended_by_how"] == fs.SEQUENCE_DEDUCED
    assert items[1]["how_acquired"] == "divisione" and items[2]["how_acquired"] == "divisione"
    assert items[2]["ended_by"] == "altro"                       # the south plot passed whole to Elsa
    assert fs._p_common_owner({"property_history": items}, [], True)


def test_an_asset_covering_later_ones_by_its_own_words():
    assert infer(("campo Alto e campo Basso", "Dario"), ("campo Alto", "Bruno"), ("campo Basso", "Carla"))[0]["ended_by"] == "divisione"
    assert infer(("poderi Alto e Basso", "Dario"), ("podere Basso", "Bruno"))[0]["ended_by"] == "divisione"          # one part sold


def test_what_the_model_wrote_stands():
    items = infer(("terreni di Bruno e Carla", "Dario", "", "successione"), ("terreno", "Bruno"), ("terreno", "Carla"))
    assert items[0]["ended_by"] == "successione" and "ended_by_how" not in items[0]


@pytest.mark.parametrize("entries", [
    [("fondo", "Dario"), ("fondo", "Bruno"), ("fondo", "Carla")],                           # only generic words: not the same land
    [("vigneto Alto", "Dario"), ("oliveto Basso", "Bruno"), ("frutteto Largo", "Carla")],   # three neighbours
    [("vigneto", "Dario")]])
def test_neighbours_are_not_a_division(entries):
    assert not infer(*entries)[0]["ended_by"]


def test_the_rule_fires_from_the_sequence_through_validation(monkeypatch):
    monkeypatch.setenv(fs.TODAY_ENV, "2026-10-27")
    reply = {"property_history": [{"asset": "terreni di Bruno e Carla", "owner": "Dario"},
                                  {"asset": "terreno a nord", "owner": "Bruno"}, {"asset": "terreno a sud", "owner": "Carla"}],
             "works_or_signs": [{"what": "strada", "visible": "sì", "before_separation": "sì"}]}
    facts = fs.validate_facts(reply, CASE)
    assert facts["property_history"][0]["ended_by"] == "divisione"
    assert "destinazione_padre_famiglia" in [r["id"] for r in fs.fire_rules(facts, fs.compute_terms(facts))]
    assert "«divisione»" in fs._EXTRACT_SYSTEM and "ended_by è obbligatorio" in fs._EXTRACT_SYSTEM


# --- split_into_parts: whatever word the model used ---------------------------------------------------------------

def reply(ended_by, *later):
    return {"property_history": [{"asset": "terreni di Bruno e Carla", "owner": "Dario", "ended_by": ended_by},
                                 {"asset": "terreno a nord", "owner": "Bruno"}, {"asset": "terreno a sud", "owner": "Carla"},
                                 {"asset": "terreno a sud", "owner": "Elsa", "how_acquired": "vendita"}, *later],
            "works_or_signs": [{"what": "strada", "visible": "sì", "before_separation": "sì"}]}


def fired(raw, case=CASE):
    facts = fs.validate_facts(raw, case)
    return facts, "destinazione_padre_famiglia" in [r["id"] for r in fs.fire_rules(facts, fs.compute_terms(facts))]


@pytest.mark.parametrize("word", ["", "vendita", "compravendita", "alienazione", "donazione", "successione", "divisione", "altro"])
def test_the_rule_fires_whatever_word_ended_the_joint_ownership(monkeypatch, word):
    monkeypatch.setenv(fs.TODAY_ENV, "2026-10-27")
    facts, fire = fired(reply(word))
    assert fire and facts["split_into_parts"][0]["owner"] == "Dario" and facts["split_into_parts"][0]["how"] == fs.SEQUENCE_DEDUCED


def test_the_model_s_word_stays_and_the_evidence_says_it_was_deduced(monkeypatch):
    monkeypatch.setenv(fs.TODAY_ENV, "2026-10-27")
    facts, _ = fired(reply("compravendita"))
    assert facts["property_history"][0]["ended_by"] == "compravendita" and "ended_by_how" not in facts["property_history"][0]
    condition = {"common_previous_owner": True}
    assert fs._evidence(condition, facts, []) == "unico proprietario Dario, poi alienato in parti (dedotto dalla sequenza)"


def neighbours(*words):
    return {"property_history": [{"asset": a, "owner": o, "ended_by": e, "how_acquired": h} for a, o, e, h in words],
            "works_or_signs": [{"what": "strada", "visible": "sì", "before_separation": "sì"}]}


@pytest.mark.parametrize("entries", [
    [("vigneto Alto", "Dario", "", ""), ("oliveto Basso", "Bruno", "", ""), ("frutteto Largo", "Carla", "", "")],
    [("fondo Alto", "Dario", "compravendita", ""), ("fondo Basso", "Bruno", "", "compravendita"), ("fondo Largo", "Carla", "", "donazione")],
    [("fondo", "Dario", "", ""), ("fondo", "Bruno", "", ""), ("fondo", "Carla", "", "")]])
def test_neighbours_plots_with_no_shared_owner_do_not_fire(monkeypatch, entries):
    monkeypatch.setenv(fs.TODAY_ENV, "2026-10-27")
    facts, fire = fired(neighbours(*entries))
    assert not fire and facts["split_into_parts"] == []


@pytest.mark.parametrize("entries", [
    [("podere Alto", "Dario", "compravendita", ""), ("podere Alto", "Bruno", "", "compravendita")],
    [("podere Alto", "Dario", "", ""), ("podere Alto", "Bruno", "", "")],
    [("fondo", "Dario", "vendita", ""), ("fondo", "Bruno", "", "")],
    [("podere Alto", "Dario", "vendita", ""), ("podere Alto", "Bruno", "vendita", "vendita"), ("podere Alto", "Carla", "", "vendita")],
    [("fondo", "Dario", "vendita", ""), ("fondo", "Bruno", "vendita", "vendita"), ("fondo", "Carla", "", "vendita")]])
def test_a_whole_sale_to_one_buyer_or_a_chain_of_them_does_not_fire(monkeypatch, entries):
    monkeypatch.setenv(fs.TODAY_ENV, "2026-10-27")
    facts, fire = fired(neighbours(*entries))
    assert not fire and facts["split_into_parts"] == []


def test_generic_assets_with_a_stated_transfer_and_two_final_owners_do_fire(monkeypatch):
    monkeypatch.setenv(fs.TODAY_ENV, "2026-10-27")
    facts, fire = fired(neighbours(("fondo", "Dario", "compravendita", ""), ("fondo", "Bruno", "", ""), ("fondo", "Carla", "", "")))
    assert fire and facts["split_into_parts"] == []


def test_the_common_ownership_s_end_dates_a_work_on_a_split(monkeypatch):
    monkeypatch.setenv(fs.TODAY_ENV, "2026-10-27")
    case = "I campi Alto e Basso erano di Dario fino al 1975. Bruno ha il campo Alto, Carla il campo Basso. La strada c'era già."
    raw = {"property_history": [{"asset": "campi Alto e Basso", "owner": "Dario", "to": "1975", "ended_by": "donazione"},
                                {"asset": "campo Alto", "owner": "Bruno"}, {"asset": "campo Basso", "owner": "Carla"}],
           "works_or_signs": [{"what": "strada", "visible": "sì", "before_separation": "sì"}]}
    assert fs.validate_facts(raw, case)["works_or_signs"][0]["existing_since"] == "1975"


# --- the plan ------------------------------------------------------------------------------------------------------

def issue(title, *norms):
    return {"titolo": title, "norme": list(norms), "pronunce": [], "argomento": f"arg {title}", "obiezione": "", "replica": ""}


def plan_of(*issues, subordinate=()):
    return {"questioni": list(issues), "subordinate": list(subordinate), "punti_deboli": [], "risposte": []}


USUCAPIONE, SERVITU, CONFESSORIA, COATTIVO = ("1158", "c.c."), ("1061", "c.c."), ("1079", "c.c."), ("1051", "c.c.")
RULE = {"nome": "usucapione di servitù apparente", "fatto": "strada usata da trent'anni", "refs": [USUCAPIONE, SERVITU]}
FIRED = {"fact_signals": {"fired": [{"institutions": ["usucapione di servitù apparente"], "acquired_right": True, "why": "uso"}]},
         "rule_institutions": [RULE], "rule_required": [CONFESSORIA]}


@pytest.fixture
def analysis(monkeypatch):
    monkeypatch.setattr(ptw, "_signals", lambda a: a.get("fact_signals"))
    return dict(FIRED)


def rows(*refs):
    return [{"_article": r} for r in refs]


def test_a_fired_rule_absent_from_the_plan_gets_its_issue_before_the_remedy(analysis):
    plan = plan_of(issue("Passaggio coattivo", COATTIVO))
    plan, log = ptw.place_rule_institutions(plan, analysis, rows(USUCAPIONE, SERVITU, CONFESSORIA, COATTIVO))
    assert log["placed"] == ["1158 c.c.", "1061 c.c.", "1079 c.c."] and log["issues_added"]
    plan, _ = ptw.order_for_acquired(plan, ptw.acquired_rights(analysis))
    assert plan["questioni"][0]["norme"] == [USUCAPIONE, SERVITU, CONFESSORIA]
    assert plan["questioni"][-1]["titolo"].startswith(ptw._SUBORDINATE_PREFIX) and plan["questioni"][-1]["norme"] == [COATTIVO]
    assert any("Passaggio coattivo" in s for s in plan["subordinate"])
    assert plan["diritto_acquisito"]


def test_a_rule_merged_into_the_issue_that_holds_one_of_its_articles(analysis):
    plan, log = ptw.place_rule_institutions(plan_of(issue("Acquisto della servitù", SERVITU)), analysis, rows(USUCAPIONE, SERVITU))
    assert len(plan["questioni"]) == 1 and USUCAPIONE in plan["questioni"][0]["norme"] and log["merged_into"]


def test_only_articles_in_the_material_are_placed(analysis):
    plan, log = ptw.place_rule_institutions(plan_of(issue("Altro", ("2043", "c.c."))), analysis, rows(("2043", "c.c.")))
    assert len(plan["questioni"]) == 1 and log["placed"] == []


def test_an_issue_mixing_recognition_and_remedy_is_split(analysis):
    plan = plan_of(issue("La servitù di passaggio", SERVITU, COATTIVO), issue("Danni", ("2043", "c.c.")))
    plan, _ = ptw.order_for_acquired(plan, ptw.acquired_rights(analysis))
    titles = [q["titolo"] for q in plan["questioni"]]
    assert plan["questioni"][0]["norme"] == [SERVITU] and titles[1] == "Danni" and titles[2].startswith(ptw._SUBORDINATE_PREFIX)
    assert plan["questioni"][2]["norme"] == [COATTIVO] and "indennità" in plan["questioni"][2]["argomento"]


def test_no_acquired_right_changes_nothing():
    plan = plan_of(issue("Passaggio coattivo", COATTIVO))
    assert ptw.order_for_acquired(plan, []) == (plan, {})


def test_the_other_side_s_defence_is_no_subordinate_request():
    plan, moved = ptw.without_defences(plan_of(subordinate=["Passaggio coattivo", "Concorso di colpa del cliente ex art. 1227 c.c."]))
    assert plan["subordinate"] == ["Passaggio coattivo"] and plan["punti_deboli"] == moved and len(moved) == 1


# --- the writer ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sentence", [
    "La servitù è stata acquistata ai sensi dell'art. 1051 c.c. già nel 2003.", "Il diritto di passaggio è sorto ex art. 1051 c.c.",
    "Ai sensi dell'art. 1051 c.c. il fondo ha acquisito la servitù.", "Dall'uso ventennale discende una presunzione di servitù."])
def test_wrong_statements_are_seen(sentence):
    assert ptw._wrong_statement(sentence)


@pytest.mark.parametrize("sentence", [
    "In via subordinata si chiede la costituzione del passaggio coattivo ex art. 1051 c.c., contro indennità.",
    "L'art. 1051 c.c. non è titolo di acquisto della servitù.", "Dall'uso non deriva alcuna presunzione di servitù.",
    "La servitù è stata acquistata per usucapione ai sensi dell'art. 1158 c.c."])
def test_right_statements_stay(sentence):
    assert ptw._wrong_statement(sentence) == ""


def test_the_review_drops_them(monkeypatch):
    monkeypatch.setattr(ptw, "_review_call", lambda *a, **k: [])
    text = "La strada esiste da trent'anni. La servitù è stata acquistata ai sensi dell'art. 1051 c.c. nel 2003. Il fondo è intercluso."
    out, log = ptw.review_section("Applicazione", text, "caso", [], [], "Bruno")
    assert "1051" not in out and "intercluso" in out and log[0]["kind"] == "wrong_statement"
