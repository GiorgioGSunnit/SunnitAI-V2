"""v79.19.0 -- frame correctness (a branch from v79.15.3), the model faked, on synthetic texts (invented names and
facts; never a real, eval or tester case):

2. one source of truth: the corrected frame written once (state["frame"], case_frame.settle); a reading correction
   (the charge sentence cites another article than the reply's number: old -> new, frame_corrections_applied, logged
   frame_correction); the posture's correction pass once; a corrected charge reaches every consumer -- bound, the
   analysis step's input, the charge rule (state["charge"]), the fit's protected set, the method's charge view
3. several charges: a list of 1..3, each verified, the first the charge, the others alternative lines (charged,
   decomposed as «frame_alternative»); frame_pretese_count
4. a deduced side: «fonte»: «dedotto» -- the frame marks it, the banner says it, no question
5. the user's article against the facts: «articolo_dai_fatti» -- both to the fit (the disputed charge unprotected), the
   judgment decides (apply_fit), frame_article_disputed
6. «da determinare» never when a fitted article exists (analysis_step.fitted_fallback, the fit tiers from the
   judgment); the honesty line only when the retrieval returned nothing (lawyer_method.corpus_line)
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

from src.rag import analysis_step as A  # noqa: E402
from src.rag import case_frame as cf  # noqa: E402
from src.rag import intake  # noqa: E402
from src.rag import interaction_copy as C  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import lawyer_method as M  # noqa: E402
from src.rag import posture  # noqa: E402
from test_v79_14_case_frame import _model, _read  # noqa: E402
from test_v79_14_flow import _req, _run, _state  # noqa: E402

LITE = ("Assisto Piero Vinci, indagato: la Procura contesta l'art. 584 c.p. per la morte di Luca Ferri, avvenuta in "
        "ospedale tre giorni dopo una lite in cui Vinci lo ha colpito con un pugno. Cosa rischia Piero Vinci?")
DISPUTE = ("Assisto Piero Vinci, indagato per l'art. 582 c.p.: durante una lite ha colpito Luca Ferri con un pugno e "
           "Ferri è morto in ospedale tre giorni dopo. Cosa rischia Piero Vinci?")
LOAN = ("Anna Pino ha prestato 9.000 euro a Bruno Faggio con una scrittura privata; Faggio non ha restituito la somma "
        "alla scadenza e sostiene che si trattava di una donazione. Come può Anna Pino recuperare la somma?")
TWO = ("Assisto Carla Noce, indagata: la Procura le contesta l'art. 646 c.p. per le somme trattenute e l'art. 640 c.p. "
       "per le fatture inviate al cliente Dino Rovere. Quali sono i rischi per Carla Noce?")


def _penal_reply(text_article, facts_article=None, frase="«la Procura contesta l'art. 584 c.p. per la morte di Luca "
                                                          "Ferri»"):
    charge = {"articolo": text_article, "descrizione": "la morte di Luca Ferri", "frase": frase}
    if facts_article:
        charge["articolo_dai_fatti"] = facts_article
    return {"parti": [{"nome": "Piero Vinci", "ruolo_nel_fatto": "lo ha colpito con un pugno", "lato": "cliente"},
                      {"nome": "Luca Ferri", "ruolo_nel_fatto": "è morto in ospedale", "lato": "controparte"}],
            "cliente": {"nome": "Piero Vinci", "perche": "«Assisto Piero Vinci»"},
            "ruolo": {"valore": "imputato/indagato", "fonte": "testo", "perche": "«Assisto Piero Vinci, indagato»"},
            "rito": {"valore": "penale", "perche": "«indagato»"},
            "stadio": {"valore": "indagini", "perche": "«indagato»"},
            "addebito_o_domanda": [charge], "istituto": None, "quesiti": ["Cosa rischia Piero Vinci?"]}


def _events(monkeypatch):
    seen = []
    from src.rag import verbose_logger
    monkeypatch.setattr(verbose_logger, "vlog", lambda event, data=None, *a, **k: seen.append((event, data)))
    return seen


# --- 2. one source of truth ------------------------------------------------------------------------------------------

def test_the_reply_number_is_corrected_to_the_article_the_charge_sentence_cites(monkeypatch):
    events = _events(monkeypatch)
    res = _read(LITE, _penal_reply("541 c.p."))
    charge = res["frame"]["addebito_o_domanda"]
    assert charge["articolo"] == "584 c.p." and charge["corretto_da"] == "541 c.p."
    assert res["counters"]["frame_corrections_applied"] == 1
    assert res["corrections"] == [{"field": "addebito_o_domanda.articolo", "old": "541 c.p.", "new": "584 c.p.",
                                   "why": "il testo cita un altro articolo nella frase dell'addebito", "step": "read"}]
    assert [d for e, d in events if e == cf.CORRECTION_EVENT] == res["corrections"]
    # the same number: nothing corrected; a sentence citing no article: the reply's number stands
    assert _read(LITE, _penal_reply("584 c.p."))["counters"]["frame_corrections_applied"] == 0
    plain = _read(LITE, _penal_reply("541 c.p.", frase="«Cosa rischia Piero Vinci?»"))
    assert plain["frame"]["addebito_o_domanda"]["articolo"] == "541 c.p." and plain["corrections"] == []


def test_the_corrected_frame_is_written_once_and_every_consumer_reads_it(monkeypatch):
    monkeypatch.setenv("CASE_FRAME", "true")
    monkeypatch.setattr(cf, "_default_call", _model(_penal_reply("541 c.p.")))
    got = intake.request_intake(LITE)
    records = {"584 c.p.": {"article": "584 c.p.", "rubrica": "Omicidio preterintenzionale",
                            "requisites": [_req(1, "atti diretti a commettere un delitto", "da provare")]}}
    state = _state(LITE, records=records)
    state["analysis"]["frame"] = {"areas": ["penale"]}
    flow.take_intake(state, got, LITE)
    settled = state["frame"]
    assert settled[cf.SETTLED] is True and cf.held(state) is settled
    assert settled["frame"]["addebito_o_domanda"]["label"] == "584 c.p."
    assert cf.settle(state) is settled                                   # idempotent: written once
    # every consumer: the binding, the analysis step's input, the charge rule, the fit's protected set, the method
    assert cf.bound(state, "addebito_o_domanda") == "584 c.p." and cf.charges(state) == ["584 c.p."]
    state["sentences"] = []
    assert A.inputs(state)["charge"] == "584 c.p."
    p = posture.of(state, flow._restored_analysis(state["analysis"]))
    state["posture"] = flow._jsonable(p)
    posture.choose_charge(state, p, state["records"], LITE)
    assert state["charge"]["article"] == "584 c.p." and state["charge"]["rule"] == posture.FRAME_CHARGE_RULE
    assert "584 c.p." in flow.fit_protected(state, p)
    assert M.frame_charge(state) == "584 c.p."
    assert M.charge_view(state) == {"article": "584 c.p.", "rule": posture.FRAME_CHARGE_RULE, "alternatives": [],
                                    **{k: v for k, v in state["charge"].items() if k not in ("article", "rule")}}
    a = cf.audit(state)
    assert a["frame_corrections_applied"] == 1 and a["frame_audit"]["corrections"][0]["old"] == "541 c.p."
    assert a["frame_pretese_count"] == 1 and a["frame_article_disputed"] == 0 and a["frame_role_deduced"] == 0


def test_the_postures_correction_pass_runs_once_and_is_logged_old_to_new(monkeypatch):
    monkeypatch.setenv("CASE_FRAME", "true")
    text = "Marco Lince ha ottenuto con raggiri da Pia Merlo la firma di una vendita. Cosa rischia Marco Lince?"
    reply = {"parti": [{"nome": "Marco Lince", "lato": "cliente"}],
             "cliente": {"nome": "Marco Lince", "perche": "«Cosa rischia Marco Lince?»"},
             "addebito_o_domanda": {"articolo": "1439 c.c.", "descrizione": "raggiri",
                                    "frase": "«ha ottenuto con raggiri da Pia Merlo la firma di una vendita»"},
             "quesiti": ["Cosa rischia Marco Lince?"]}
    monkeypatch.setattr(cf, "_default_call", _model(reply))
    state = _state(text)
    flow.take_intake(state, intake.request_intake(text), text)
    assert cf.bound(state, "addebito_o_domanda") == "1439 c.c."         # no posture yet: the article stands
    penal = {"rito": "penale", "family": "penale", "role": "imputato"}
    assert cf.settle_posture(state, penal) == [{"field": "addebito_o_domanda.articolo", "old": "1439 c.c.", "new": "",
                                                "why": "in una posizione penale non è una norma penale",
                                                "step": "posture"}]
    assert cf.settle_posture(state, penal) == [] and state["frame_charge_refused"] == "1439 c.c."
    assert cf.bound(state, "addebito_o_domanda") is None
    assert cf.bound(state, "addebito_o_domanda", {"rito": "civile", "family": "civile"}) is None   # one frame
    assert len(cf.corrections(state)) == 1 and cf.audit(state)["charge_rito_dropped"] == 1


# --- 3. several charges ---------------------------------------------------------------------------------------------

def _two_reply():
    return {"parti": [{"nome": "Carla Noce", "ruolo_nel_fatto": "somme trattenute", "lato": "cliente"},
                      {"nome": "Dino Rovere", "lato": "controparte"}],
            "cliente": {"nome": "Carla Noce", "perche": "«Assisto Carla Noce»"},
            "ruolo": {"valore": "imputato/indagato", "perche": "«Assisto Carla Noce, indagata»"},
            "rito": {"valore": "penale", "perche": "«la Procura le contesta»"},
            "stadio": {"valore": "indagini", "perche": "«indagata»"},
            "addebito_o_domanda": [
                {"articolo": "646 c.p.", "descrizione": "le somme trattenute",
                 "frase": "«la Procura le contesta l'art. 646 c.p. per le somme trattenute»"},
                {"articolo": "640 c.p.", "descrizione": "le fatture inviate",
                 "frase": "«l'art. 640 c.p. per le fatture inviate al cliente Dino Rovere»"}],
            "quesiti": ["Quali sono i rischi per Carla Noce?"]}


def test_several_charges_are_read_verified_and_the_first_leads(monkeypatch):
    res = _read(TWO, _two_reply())
    assert res["counters"]["frame_pretese_count"] == 2
    assert res["frame"]["addebito_o_domanda"]["articolo"] == "646 c.p."
    assert [a["articolo"] for a in res["frame"]["addebiti"]] == ["646 c.p.", "640 c.p."]
    # one of them unverified: dropped alone; four: the fourth dropped and logged
    reply = _two_reply()
    reply["addebito_o_domanda"][1]["frase"] = "«per le fatture false»"
    res = _read(TWO, reply)
    assert res["counters"]["frame_pretese_count"] == 1 and res["counters"]["frame_unverified.addebito_o_domanda"] == 1
    reply = _two_reply()
    reply["addebito_o_domanda"] += [{"articolo": a, "descrizione": "i rischi", "frase": "«Quali sono i rischi per Carla Noce?»"}
                                    for a in ("648 c.p.", "649 c.p.")]
    res = _read(TWO, reply)
    assert res["counters"]["frame_pretese_count"] == 3
    assert any("più di 3 addebiti" in r for r in res["unverified"]["addebito_o_domanda"])


def test_the_other_charges_become_alternative_lines_charged_and_decomposed(monkeypatch):
    monkeypatch.setenv("CASE_FRAME", "true")
    monkeypatch.setattr(cf, "_default_call", _model(_two_reply()))
    records = {label: {"article": label, "rubrica": rubrica, "requisites": [_req(1, "x", "da provare")]}
               for label, rubrica in (("646 c.p.", "Appropriazione indebita"), ("640 c.p.", "Truffa"))}
    state = _state(TWO, records=records)
    state["analysis"]["frame"] = {"areas": ["penale"]}
    flow.take_intake(state, intake.request_intake(TWO), TWO)
    assert cf.charges(state) == ["646 c.p.", "640 c.p."] and cf.alternative_charges(state) == ["640 c.p."]
    p = posture.of(state, flow._restored_analysis(state["analysis"]))
    state["posture"] = flow._jsonable(p)
    choice = posture.choose_charge(state, p, state["records"], TWO)
    assert choice["article"] == "646 c.p." and choice["order"][:2] == ["646 c.p.", "640 c.p."]
    assert M.charge_view(state)["alternatives"] == ["640 c.p."]
    assert flow.DECOMPOSE_WHY["frame_alternative"] == "altro addebito o articolo letto dal quadro"
    assert cf.audit(state)["frame_pretese_count"] == 2


# --- 4. a deduced side ----------------------------------------------------------------------------------------------

def _loan_reply(fonte="dedotto"):
    return {"parti": [{"nome": "Anna Pino", "ruolo_nel_fatto": "ha prestato 9.000 euro", "lato": "cliente"},
                      {"nome": "Bruno Faggio", "ruolo_nel_fatto": "non ha restituito la somma", "lato": "controparte"}],
            "cliente": {"nome": "Anna Pino", "perche": "«Come può Anna Pino recuperare la somma?»"},
            "ruolo": {"valore": "attore", "fonte": fonte, "perche": "«Come può Anna Pino recuperare la somma?»"},
            "rito": {"valore": "civile", "perche": "«scrittura privata»"},
            "stadio": {"valore": "pre_filing", "perche": "«non ha restituito la somma alla scadenza»"},
            "addebito_o_domanda": [{"articolo": None, "descrizione": "recuperare la somma",
                                    "frase": "«Come può Anna Pino recuperare la somma?»"}],
            "istituto": None, "quesiti": ["Come può Anna Pino recuperare la somma?"]}


def test_a_deduced_side_is_marked_said_in_the_banner_and_asks_nothing(monkeypatch):
    res = _read(LOAN, _loan_reply())
    assert res["frame"]["ruolo"][cf.DEDUCED] is True and res["counters"]["frame_role_deduced"] == 1
    assert _read(LOAN, _loan_reply("testo"))["counters"]["frame_role_deduced"] == 0
    assert cf.DEDUCED not in _read(LOAN, _loan_reply("testo"))["frame"]["ruolo"]
    out = _run(monkeypatch, LOAN, _loan_reply(), asked=True)
    assert out["client"] == "Anna Pino" and out["posture"]["role"] == "attore"
    assert out["line"] == "Il parere è impostato per Anna Pino (attore), dedotto dai fatti: dimmi se assisti " \
                          "un'altra parte."
    assert out["line"] == C.fill(C.CLIENT_DEDUCED, parte="Anna Pino (attore)")
    assert "?" not in out["line"] and "Non mi hai indicato" not in out["line"]
    assert out["audit"]["frame_role_deduced"] == 1 and out["audit"]["role_flip"] == 0
    # stated in the text: the usual line, nothing deduced
    stated = _run(monkeypatch, LOAN, _loan_reply("testo"), asked=True)
    assert "dedotto dai fatti" not in stated["line"] and stated["audit"]["frame_role_deduced"] == 0


# --- 5. the user's article against the facts ----------------------------------------------------------------------------

def test_a_disputed_article_carries_both_to_the_fit_and_the_judgment_decides(monkeypatch):
    frase = "«indagato per l'art. 582 c.p.: durante una lite ha colpito Luca Ferri con un pugno»"
    res = _read(DISPUTE, _penal_reply("582 c.p.", "584 c.p.", frase=frase))
    charge = res["frame"]["addebito_o_domanda"]
    assert charge["articolo"] == "582 c.p." and charge["articolo_fatti"] == "584 c.p." and charge["disputed"] is True
    assert res["counters"]["frame_article_disputed"] == 1 and res["corrections"] == []
    # the same article twice: nothing disputed; no text article: the facts' one is the article
    same = _read(DISPUTE, _penal_reply("582 c.p.", "582 c.p.", frase=frase))["frame"]["addebito_o_domanda"]
    assert "disputed" not in same and same["articolo"] == "582 c.p."
    facts_only = _read(DISPUTE, _penal_reply(None, "584 c.p.", frase=frase))["frame"]["addebito_o_domanda"]
    # v79.19.1 (item 1): no article in the text -- the facts' one is a hypothesis («ipotesi»), never the verified charge
    assert facts_only["articolo"] is None and facts_only["ipotesi"] == "584 c.p." and facts_only["da_fatti"] is True
    assert "disputed" not in facts_only
    # a civil article as the facts' one in a penal rito: dropped, the charge not disputed
    civil = _read(DISPUTE, _penal_reply("582 c.p.", "2043 c.c.", frase=frase))
    assert "disputed" not in civil["frame"]["addebito_o_domanda"] and civil["counters"]["charge_rito_dropped"] == 1

    monkeypatch.setenv("CASE_FRAME", "true")
    monkeypatch.setattr(cf, "_default_call", _model(_penal_reply("582 c.p.", "584 c.p.", frase=frase)))
    records = {label: {"article": label, "rubrica": rubrica, "requisites": [_req(1, "x", "da provare")]}
               for label, rubrica in (("582 c.p.", "Lesione personale"), ("584 c.p.", "Omicidio preterintenzionale"))}
    state = _state(DISPUTE, records=records)
    state["analysis"]["frame"] = {"areas": ["penale"]}
    flow.take_intake(state, intake.request_intake(DISPUTE), DISPUTE)
    p = posture.of(state, flow._restored_analysis(state["analysis"]))
    state["posture"] = flow._jsonable(p)
    assert cf.disputed(state) and cf.bound(state, "addebito_o_domanda") == "582 c.p."
    assert cf.alternative_charges(state) == ["584 c.p."]              # both go to the fit, the facts' one decomposed too
    assert "582 c.p." not in flow.fit_protected(state, p)              # never picked by its number alone
    # the judgment: the text's article «no», the facts' one «si» -- the facts' article is the charge (a correction)
    made = cf.apply_fit(state, [{"article": "582 c.p.", "fit": "no", "verified": True},
                                {"article": "584 c.p.", "fit": "si", "verified": True}])
    assert made == [{"field": "addebito_o_domanda.articolo", "old": "582 c.p.", "new": "584 c.p.",
                     "why": "giudizio di applicabilità: l'articolo citato nel testo non governa i fatti («no»), quello "
                            "letto dai fatti «si»", "step": "fit"}]
    assert cf.bound(state, "addebito_o_domanda") == "584 c.p." and cf.charges(state) == ["584 c.p."]
    assert cf.alternative_charges(state) == ["582 c.p."] and not cf.disputed(state)
    assert cf.apply_fit(state, []) == []                                 # once
    assert "584 c.p." in flow.fit_protected(state, p)                   # settled: the charge is protected again
    posture.choose_charge(state, p, state["records"], DISPUTE)
    assert state["charge"]["article"] == "584 c.p." and M.charge_view(state)["alternatives"] == ["582 c.p."]
    a = cf.audit(state)
    assert a["frame_article_disputed"] == 1 and a["frame_corrections_applied"] == 1


def test_a_disputed_article_judged_applicable_stands_and_the_facts_one_is_an_alternative(monkeypatch):
    monkeypatch.setenv("CASE_FRAME", "true")
    frase = "«indagato per l'art. 582 c.p.: durante una lite ha colpito Luca Ferri con un pugno»"
    monkeypatch.setattr(cf, "_default_call", _model(_penal_reply("582 c.p.", "584 c.p.", frase=frase)))
    state = _state(DISPUTE)
    flow.take_intake(state, intake.request_intake(DISPUTE), DISPUTE)
    for verdicts in ([{"article": "582 c.p.", "fit": "si", "verified": True},
                      {"article": "584 c.p.", "fit": "si", "verified": True}],
                     [{"article": "582 c.p.", "fit": "no", "verified": False}]):     # unverified: nothing changes
        fresh = _state(DISPUTE)
        flow.take_intake(fresh, intake.request_intake(DISPUTE), DISPUTE)
        assert cf.apply_fit(fresh, verdicts) == []
        assert cf.bound(fresh, "addebito_o_domanda") == "582 c.p." and cf.alternative_charges(fresh) == ["584 c.p."]
        assert not cf.disputed(fresh) and cf.corrections(fresh) == []


# --- 6. never «da determinare» with a fitted article; the honesty line only on an empty retrieval -------------------

def test_the_fit_tiers_rank_the_fallback_and_the_judgments_reach_the_input():
    from test_v79_15_0_analysis import _civil
    inp = _civil()
    for n in inp["norms"]:
        n["fit"] = {"1385 c.c.": "", "1453 c.c.": "no", "1223 c.c.": "dipende"}.get(n["article"], "")
    inp["norms"].append({"article": "1456 c.c.", "rubrica": "Clausola risolutiva espressa", "text": "x",
                         "elements": [], "periods": [], "fit": "si", "remedy": False})
    line = A.fitted_fallback(inp)
    assert line["articolo"] == "1456 c.c." and line["stato"] == A.FIT_LINE and line["fit"] == "si"
    assert line["tesi"] == "Clausola risolutiva espressa: elementi da provare"
    assert A.line_text(line) == ("art. 1456 c.c. — Clausola risolutiva espressa: elementi da provare (norma "
                                 "applicabile ai fatti, linea non verificata)")
    inp["norms"] = [n for n in inp["norms"] if n["article"] != "1456 c.c."]
    assert A.fitted_fallback(inp)["articolo"] == "1385 c.c."            # unjudged before a «no»; a «no» never
    assert A.fitted_fallback({"norms": [n for n in inp["norms"] if n.get("fit") == "no"]}) is None
    state = {"applicability": {"judgments": [{"article": "640 c.p.", "fit": "si", "verified": True},
                                             {"article": "641 c.p.", "fit": "dipende", "verified": False}]}}
    got = A.fit_verdicts(state)
    assert got == {flow._key(("640", "c.p.")): "si"}


def test_the_honesty_line_prints_only_when_the_retrieval_returned_nothing(monkeypatch):
    from src.rag import corpus_laws
    from test_v79_14_1 import _posture, _rec, _req as _r
    monkeypatch.setitem(corpus_laws._NAMES, "names", ["Codice Civile 2026"])
    base = {"circumstances": {}, "posture": _posture("civile", "attore"), "case": "x", "assessment": {},
            "institutions": ["pacchetto turistico"], "client": "Tizio"}
    read = {**base, "records": {"1218 c.c.": _rec("Responsabilità del debitore", _r(1, "l'inadempimento",
                                                                                     "da provare"))}}
    assert M.corpus_line(read) == "" and read["corpus_line_skipped"] == "retrieval returned records for the pretesa"
    empty = {**base, "records": {}}
    assert M.corpus_line(empty) == M.CORPUS_LINE.format(materia="pacchetto turistico") + "."
    noise = {**base, "records": {"1218 c.c.": {**_rec("Responsabilità del debitore", _r(1, "x", "da provare")),
                                               "ranked_candidate": True}}}
    assert M.corpus_line(noise) == M.CORPUS_LINE.format(materia="pacchetto turistico") + "."     # a candidate is no record
