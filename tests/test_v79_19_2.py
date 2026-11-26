"""v79.19.2 -- Antonelo's review of 79.19.1, rows 67-74 (the model faked; synthetic texts, invented names):

1. row 74: the graded reply's reason read from «perce» too
2. row 67: the frame's article the case text never cites (the charge sentence names none, no «art. N» anywhere) is a
   hypothesis, never the verified charge (frame_article_uncited); a text-named article is unchanged
3. row 68: the frame's charge replaces a read hypothesis or a rubrica match, never the user's own article unless the
   frame's article is itself among the user's or the named ones; row 72: a procedure article named in a charge
   sentence is never the charge nor the exposure
4. row 71: step_aside_answer logs reason= from a closed vocabulary; what the method computed before stepping aside is
   saved (state["method_aside"], lawyer_method_aside); no rule change
"""

import os
import sys
from pathlib import Path

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.rag import case_frame as cf  # noqa: E402
from src.rag import defensive_generation as dg  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import posture, verbose_logger  # noqa: E402
from test_v79_10_5 import IMPUTATO, _penal_state  # noqa: E402


@pytest.fixture(autouse=True)
def _frame_on(monkeypatch):
    monkeypatch.setenv("CASE_FRAME", "true")


# --- 1. «perce» ------------------------------------------------------------------------------------------------------------

def test_the_graded_reason_is_read_from_perce_too():
    ids = {"A1": "640 c.p."}
    got = dg._graded_reply('[{"a": "A1", "voto": 2, "questione": "q", "perce": "la ragione"}]', ids)
    assert got["640 c.p."]["reason"] == "la ragione" and got["640 c.p."]["grade"] == 2
    assert dg._graded_reply('[{"a": "A1", "voto": 2, "perche": "p", "perce": "x"}]', ids)["640 c.p."]["reason"] == "p"


# --- 2. an article the text never cites ------------------------------------------------------------------------------------

APPALTO = ("Assisto Marta Galli: la Edilnova S.r.l. ha costruito la sua villetta e dopo sei mesi sono comparse crepe e "
           "infiltrazioni; Marta Galli vuole far valere la garanzia per i vizi. Che cosa può ottenere?")


def _civil_reply(article, frase="«vuole far valere la garanzia per i vizi»"):
    return {"parti": [{"nome": "Marta Galli", "ruolo_nel_fatto": "committente", "lato": "cliente"},
                      {"nome": "Edilnova S.r.l.", "ruolo_nel_fatto": "appaltatore", "lato": "controparte"}],
            "cliente": {"nome": "Marta Galli", "perche": "«Assisto Marta Galli»"},
            "ruolo": {"valore": "attore", "fonte": "testo", "perche": "«vuole far valere la garanzia»"},
            "rito": {"valore": "civile", "perche": "«garanzia per i vizi»"}, "stadio": None,
            "addebito_o_domanda": [{"articolo": article, "descrizione": "garanzia per i vizi", "frase": frase}],
            "istituto": None, "quesiti": []}


def test_an_article_the_text_never_cites_is_a_hypothesis():
    res = cf.verify(_civil_reply("1667 c.c."), APPALTO, "civile")
    a = res["frame"]["addebito_o_domanda"]
    assert a["articolo"] is None and a["ipotesi"] == "1667 c.c." and a["da_fatti"] is True
    assert res["counters"]["frame_article_uncited"] == 1 and "frame_article_uncited" in cf.COUNTERS
    state = {"case_frame": res}
    cf.settle(state)
    assert cf.bound(state, "addebito_o_domanda") is None and cf.alternative_charges(state) == ["1667 c.c."]
    assert cf.hypotheses(state) == [{"label": "1667 c.c.", "stato": "aperta"}]
    cf.apply_fit(state, [{"article": "1667 c.c.", "fit": "si", "verified": True}])
    assert cf.bound(state, "addebito_o_domanda") == "1667 c.c."           # the judgment made it the charge


@pytest.mark.parametrize("text,frase", [
    (APPALTO.replace("garanzia per i vizi.", "garanzia per i vizi ex art. 1667 c.c."), "«vuole far valere la garanzia per i vizi»"),
    (APPALTO.replace("garanzia per i vizi.", "garanzia per i vizi (art. 1667)."), "«vuole far valere la garanzia per i vizi»"),
    (APPALTO.replace("garanzia per i vizi.", "garanzia per i vizi ex art. 1667 c.c."),
     "«far valere la garanzia per i vizi ex art. 1667 c.c.»")])
def test_a_text_named_article_is_unchanged(text, frase):
    res = cf.verify(_civil_reply("1667 c.c.", frase), text, "civile")
    a = res["frame"]["addebito_o_domanda"]
    assert a["articolo"] == "1667 c.c." and "ipotesi" not in a and res["counters"]["frame_article_uncited"] == 0


def test_a_citation_with_another_code_is_not_this_article():
    text = APPALTO.replace("garanzia per i vizi.", "garanzia per i vizi (art. 1667 c.p.).")
    res = cf.verify(_civil_reply("1667 c.c."), text, "civile")
    assert res["frame"]["addebito_o_domanda"]["ipotesi"] == "1667 c.c."
    art = cf.parse_article("1667 c.c.")
    assert cf.cited_anywhere("vedi art. 1667 c.c.", art) and cf.cited_anywhere("articolo 1667, comma 2", art)
    assert not cf.cited_anywhere("art. 1667 c.p.", art) and not cf.cited_anywhere("il 1667 è l'anno", art)
    # a facts' article given beside an uncited one: the model's «articolo_dai_fatti» stands as the hypothesis
    reply = _civil_reply("1667 c.c.")
    reply["addebito_o_domanda"][0]["articolo_dai_fatti"] = "1669 c.c."
    a = cf.verify(reply, APPALTO, "civile")["frame"]["addebito_o_domanda"]
    assert a["ipotesi"] == "1669 c.c." and a["articolo"] is None


# --- 3. the frame's charge against the user's article; a procedure article never the charge ------------------------------

PENAL = "Il PM contesta ad Alfa l'art. 612 c.p. per le parole rivolte a Beta. Alfa è imputato."


def _penal_frame(article):
    return {"parti": [{"nome": "Alfa", "ruolo_nel_fatto": "imputato", "lato": "cliente"},
                      {"nome": "Beta", "ruolo_nel_fatto": "destinatario", "lato": "controparte"}],
            "cliente": {"nome": "Alfa", "perche": "«Alfa è imputato»"},
            "ruolo": {"valore": "imputato/indagato", "fonte": "testo", "perche": "«Alfa è imputato»"},
            "rito": {"valore": "penale", "perche": "«Il PM contesta»"}, "stadio": None,
            "addebito_o_domanda": [{"articolo": article, "descrizione": "le parole rivolte a Beta",
                                    "frase": "«Il PM contesta ad Alfa l'art. 612 c.p. per le parole rivolte a Beta»"}],
            "istituto": None, "quesiti": []}


def _events(monkeypatch):
    events = []
    monkeypatch.setattr(verbose_logger, "vlog", lambda e, d=None, ms=None: events.append((e, d)))
    return events


def test_the_frames_charge_never_replaces_the_users_article(monkeypatch):
    events = _events(monkeypatch)
    state = _penal_state(monkeypatch, PENAL)
    # the frame read 624 c.p. (a reading error): the user's 612 c.p. stands, logged
    res = cf.verify(_penal_frame("624 c.p."), PENAL, "penale")
    res["frame"]["addebito_o_domanda"]["articolo"] = "624 c.p."         # the read's correction undone for the test
    res["frame"]["addebito_o_domanda"].pop("corretto_da", None)
    state["case_frame"] = res
    cf.settle(state)
    assert cf.bound(state, "addebito_o_domanda") == "624 c.p."
    out = posture.mark_sides(state, IMPUTATO)
    assert state["charge"] == {"article": "612 c.p.", "rule": "user_article"} and out["charged"][0] == "612 c.p."
    assert any(e == "interactive_charge_frame_not_over_user" and d == {"user_article": "612 c.p.", "frame": "624 c.p."}
               for e, d in events)
    # the frame's article the user's own: the same charge, the user's rule
    state = _penal_state(monkeypatch, PENAL)
    state["case_frame"] = cf.verify(_penal_frame("612 c.p."), PENAL, "penale")
    cf.settle(state)
    posture.mark_sides(state, IMPUTATO)
    assert state["charge"] == {"article": "612 c.p.", "rule": "user_article"}


def test_the_frames_charge_still_replaces_a_read_hypothesis(monkeypatch):
    case = "Alfa prende la bicicletta di Beta dal cortile e se ne va. Alfa è imputato."
    state = _penal_state(monkeypatch, case)
    reply = _penal_frame("612 c.p.")
    reply["addebito_o_domanda"][0]["frase"] = "«Alfa prende la bicicletta di Beta dal cortile e se ne va»"
    reply["addebito_o_domanda"][0]["descrizione"] = "prende la bicicletta"
    reply["rito"]["perche"] = "«Alfa è imputato»"
    state["case_frame"] = cf.verify(reply, case, "penale")
    state["case_frame"]["frame"]["addebito_o_domanda"]["articolo"] = "612 c.p."      # held as text-named for the test
    state["case_frame"]["frame"]["addebito_o_domanda"].pop("ipotesi", None)
    state["case_frame"]["frame"]["addebito_o_domanda"].pop("da_fatti", None)
    cf.settle(state)
    assert cf.bound(state, "addebito_o_domanda") == "612 c.p."
    posture.mark_sides(state, IMPUTATO)
    assert state["charge"] == {"article": "612 c.p.", "rule": posture.FRAME_CHARGE_RULE}


def test_a_procedure_article_named_by_the_text_is_never_the_charge_nor_the_exposure(monkeypatch):
    assert posture.procedural_article("380 c.p.p.") and posture.procedural_article("700 c.p.c.")
    assert posture.procedural_article("116 c.p.a.") and not posture.procedural_article("640 c.p.")
    assert not posture.procedural_article("73 d.P.R. 309/1990") and not posture.procedural_article("2043 c.c.")
    case = "Alfa è stato arrestato in flagranza e processato per il furto della bicicletta di Beta."
    state = _penal_state(monkeypatch, case, hypotheses=[])
    state["records"]["380 c.p.p."] = {"rubrica": "Arresto obbligatorio in flagranza", "requisites": []}
    monkeypatch.setattr(posture, "charged_by_name", lambda c, r, rows=(): ["380 c.p.p.", "624 c.p."])
    out = posture.choose_charge(state, IMPUTATO, state["records"], case, rows=state["rows"])
    assert out["by_name"] == ["624 c.p."] and state["charge"]["article"] == "624 c.p."
    assert "380 c.p.p." not in out["order"] and out["exposure"] != "380 c.p.p."


# --- 4. the step-aside reason and the method's record --------------------------------------------------------------------

def test_step_aside_answer_logs_a_closed_reason(monkeypatch):
    events = _events(monkeypatch)
    assert flow.STEP_ASIDE_REASONS == ("no_action", "placeholder")
    draft = "**Sintesi**\n\nUn parere.\n\n**Norme applicabili**\n\nL'art. 1218 c.c. regola l'inadempimento."
    for given, want in (("no_action", "no_action"), ("placeholder", "placeholder"), ("", "placeholder"),
                        ("qualcosa", "placeholder")):
        state = {"records": {}, "method_refused_reason": given}
        flow.step_aside_answer(state, draft, [], "Il caso.")
        got = [d for e, d in events if e == "interactive_step_aside_answer"][-1]
        assert got["reason"] == want and "cited" in got and "rule_statements_removed" in got


def test_what_the_method_computed_is_saved_when_it_steps_aside():
    ctx = {"client": "Anna Pino", "charge": {"article": "1218 c.c.", "rule": "user_article"}, "no_action": "Nessuna norma attribuisce un'azione.",
           "records": {"1218 c.c.": {"requisites": [{"id": "R1", "status": "provato"}, {"id": "R2", "status": "da_provare"}],
                                     "charged": 1},
                       "1453 c.c.": {"requisites": [{"id": "R3", "status": "escluso"}], "dropped": True}}}
    plan = {"soluzioni": [{"tesi": "La domanda è fondata.", "article": "1218 c.c.", "argued": "model"},
                          {"tesi": "Nessuna linea.", "placeholder": True}], "principale": "La domanda è fondata."}
    refusal = {"tesi": "Nessuna linea.", "why": "nessuna norma attribuisce un'azione", "no_action": True, "lines": []}
    got = flow.method_aside_record("no_action", refusal, [("Anna Pino", ctx, plan)])
    assert got["reason"] == "no_action" and got["why"] == refusal["why"] and got["tesi"] == "Nessuna linea."
    c = got["clients"][0]
    assert c["client"] == "Anna Pino" and c["charge"] == {"article": "1218 c.c.", "rule": "user_article"}
    assert c["records"]["1218 c.c."]["status"]["provato"] == 1 and c["records"]["1218 c.c."]["status"]["da_provare"] == 1
    assert c["records"]["1453 c.c."]["dropped"] is True and c["records"]["1218 c.c."]["charged"] is True
    assert [x["placeholder"] for x in c["lines"]] == [False, True] and c["main"] == "La domanda è fondata."
    assert c["no_action"].startswith("Nessuna norma")
    assert flow.method_aside_record("altro", {}, [])["reason"] == "placeholder"      # outside the vocabulary
    assert flow.method_aside_record("placeholder", {"tesi": "t"}, [("", None, None)])["clients"][0]["records"] == {}
