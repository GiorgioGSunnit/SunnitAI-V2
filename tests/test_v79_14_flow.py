"""v79.14.0 build B -- the case frame bound across the pipeline (CASE_FRAME), the model faked, on synthetic request texts
(invented names and facts; never a real, eval or tester case):

1. the brief's five cases through the pipeline -- intake (request_intake: the frame read before the rules), the state
   (take_intake), the stated client, settle_client, posture.of, choose_charge / mark_sides, clients_of and the method's
   header (lawyer_method.method_material's «POSIZIONE», the thesis call's material)
2. a later step disagreeing: logged frame_conflict with its name, the frame wins, the delivered answer unchanged
3. a failed call: every field to the rules, the output as with the flag off; the flag off: nothing runs, no frame
4. multi-client and the follow-up (the stored frame reused, no second call)
5. the counters on the delivered answer (case_frame.audit, case_counters) and the nightly's print and check
"""

import json
import os
import sys
from pathlib import Path

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

from src.rag import case_frame as cf  # noqa: E402
from src.rag import intake  # noqa: E402
from src.rag import interaction_copy as C  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import lawyer_method as M  # noqa: E402
from src.rag import posture  # noqa: E402
from test_v79_14_case_frame import (A_CARICO, DEBTOR, DIRECTOR, NO_CLIENT, TWO_SIDES, _a_carico_reply,  # noqa: E402
                                    _debtor_reply, _director_reply, _model, _two_sides_reply)


def _boom(messages, kwargs):
    raise RuntimeError("connection refused")


def _state(text, parties=(), asked=False, read=None, records=None, rows=None, extraction_client=""):
    """A synthetic case-first state as analyse_case leaves it (no graph): the analysis (its parties, its client), the
    first read, the records and rows."""
    analysis = {flow.CASE_KEY: text, "frame": {"areas": []}}
    if parties:
        analysis["fact_signals"] = {"facts": {"parties": [{"name": n, "role": r} for n, r in parties]}}
    if extraction_client:
        analysis["parte_assistita"] = extraction_client
    return {"text": text, "request": "", "lang": "it", "mode": "single_shot", "analysis": flow._jsonable(analysis),
            "topics": [], "questions": [{"id": "parte_assistita"}] if asked else [], "answered": {},
            "rows": list(rows or []), "records": dict(records or {}), "read": dict(read or {}), "timings": {},
            "additions": []}


def _run(monkeypatch, text, reply, on=True, **kw):
    """The pipeline from the request to the method's header: intake (the frame first), the state, the stated client,
    settle_client, the posture, the sides and the charge, the clients, the method's «POSIZIONE» line."""
    if on:
        monkeypatch.setenv("CASE_FRAME", "true")
    else:
        monkeypatch.delenv("CASE_FRAME", raising=False)
    model = reply if callable(reply) else _model(reply)
    monkeypatch.setattr(cf, "_default_call", model)
    got = intake.request_intake(text)
    state = _state(text, **kw)
    flow.take_intake(state, got, text)
    said = flow.bound_said(state, flow.stated(text))
    flow.apply_stated(state, said)
    analysis = flow._restored_analysis(state["analysis"])
    client, line = flow.settle_client(state, analysis)
    if client:
        flow.persist_client(state, client, line)
    p = posture.of(state, analysis)
    state["posture"] = flow._jsonable(p)
    if state["records"]:
        posture.mark_sides(state, p)
    clients = flow.clients_of(state)
    ctx = {"client": client, "posture_text": M.posture_text(p), "case": text, "assessment": {}, "records": {}}
    header = next(x for x in M.method_material(ctx).splitlines() if x.startswith("POSIZIONE"))
    return {"state": state, "intake": got, "posture": p, "client": client, "line": line, "clients": clients,
            "header": header, "audit": cf.audit(state), "model": model}


def _req(i, text, status):
    return {"id": f"R{i}", "comma": 1, "kind": "requisito", "status": status, "supporting": [], "text": text}


# --- 1. the brief's five cases ----------------------------------------------------------------------------------------

def test_case_1_the_debtor_is_convenuto_through_posture_and_the_methods_header_never_the_claimant(monkeypatch):
    out = _run(monkeypatch, DEBTOR, _debtor_reply())
    got, p = out["intake"], out["posture"]
    assert got["role"] == "convenuto" and got["confidence"]["role"] == intake.FRAME == cf.FRAME_HOW
    assert got["why"]["role"].startswith("quadro del caso: «Delfino pretende ora il pagamento")
    assert got["rito"] == "civile" and got["stage"] == "pre_filing" and got["client"]["name"] == "Orsa S.r.l."
    assert got["questions"] == ["Quali strategie difensive può adottare Orsa?", "La penale è riducibile?"]
    assert p["role"] == "convenuto" and p["rito"] == "civile" and p["family"] == "civile"
    assert out["client"] == "Orsa S.r.l." and out["state"]["client"]["name"] == "Orsa S.r.l."
    assert out["header"] == "POSIZIONE: Orsa S.r.l., rito civile, role convenuto, stage pre_filing"
    assert "attore" not in out["header"]
    a = out["audit"]
    assert a["frame_conflict"] == 0 and a["role_flip"] == 0 and a["charge_rito_mismatch"] == 0
    assert a["frame_fields_verified"] == 8 and not any(a[f"frame_fallback.{f}"] for f in cf.FIELDS)
    # «resistente» is the same side: convenuto in intake's words
    res = _run(monkeypatch, DEBTOR, _debtor_reply(ruolo={"valore": "resistente", "perche": "«Delfino pretende ora il "
                                                                                          "pagamento di una penale»"}))
    assert res["posture"]["role"] == "convenuto" and "role convenuto" in res["header"]


def test_case_1_flag_off_the_rules_read_the_claimant_the_frame_never_and_a_dropped_role_falls_back(monkeypatch):
    off = _run(monkeypatch, DEBTOR, _debtor_reply(), on=False)
    # v79.14.1 (item 1, review): the rules alone now read the debtor too («Delfino pretende ora il pagamento di una
    # penale»: a demand of no one else named -- the client resists it): convenuto, not 79.13.0's claimant reading
    assert off["posture"]["role"] == "convenuto" and off["intake"]["resisting"]
    on = _run(monkeypatch, DEBTOR, _debtor_reply())
    assert on["posture"]["role"] == "convenuto"             # the verified frame: the side that resists
    # an «attore» quoting a sentence the text does not hold: dropped, the rules fill the field -- counted, printed
    out = _run(monkeypatch, DEBTOR, _debtor_reply(ruolo={"valore": "attore", "perche": "«Orsa chiede i danni»"}))
    assert out["intake"]["confidence"]["role"] != intake.FRAME
    assert out["audit"]["frame_unverified.ruolo"] == 1 and out["audit"]["frame_fallback.ruolo"] == 1
    assert out["audit"]["frame_audit"]["filled"]["ruolo"] == out["posture"]["role"] == off["posture"]["role"]


def _director_world():
    """The records and the first read the rules would choose a charge from: the read's hypothesis «appropriazione
    indebita» (646 c.p., its rows found by the first-read search), and the frame's 2634 c.c."""
    records = {
        "2634 c.c.": {"article": "2634 c.c.", "rubrica": "Infedeltà patrimoniale",
                      "requisites": [_req(1, "amministratori in conflitto di interessi", "da provare"),
                                     _req(2, "atti di disposizione dei beni sociali", "provato")]},
        "646 c.p.": {"article": "646 c.p.", "rubrica": "Appropriazione indebita",
                     "requisites": [_req(1, "si appropria il denaro o la cosa mobile altrui", "provato"),
                                    _req(2, "di cui abbia il possesso", "provato")]},
        "1439 c.c.": {"article": "1439 c.c.", "rubrica": "Dolo",
                      "requisites": [_req(1, "raggiri usati da uno dei contraenti", "da provare")]}}
    rows = [{"_article": ["646", "c.p."], "_first_read": "appropriazione indebita",
             "s": {"plain_text": "Chiunque si appropria il denaro o la cosa mobile altrui", "titolo_articolo":
                   "Appropriazione indebita"}}]
    read = {"hypotheses": [{"line": "principale", "claim": "cessione dei beni sociali",
                            "institutions": ["appropriazione indebita"], "rubricated": ["Appropriazione indebita"]}]}
    return {"records": records, "rows": rows, "read": read}


def test_case_2_the_director_is_indagato_charged_with_a_penal_article_the_frames_charge_held(monkeypatch):
    out = _run(monkeypatch, DIRECTOR, _director_reply("2634 c.c."), **_director_world())
    p, st = out["posture"], out["state"]
    assert p["role"] == "imputato" and p["rito"] == "penale" and p["stage"] == "indagini"
    assert out["header"] == "POSIZIONE: Marco Lince, rito penale, role imputato, stage indagini"
    assert st["charge"] == {"article": "2634 c.c.", "rule": posture.FRAME_CHARGE_RULE}
    assert st["records"]["2634 c.c."]["charged"] == 1 and st["records"]["2634 c.c."]["side"] == "threat"
    assert "charged" not in st["records"]["1439 c.c."]
    # the rules would have taken the read's 646 c.p.: logged with the step's name, and it lost
    tried = [c for c in out["audit"]["frame_audit"]["conflicts"] if c["step"] == "posture.choose_charge"]
    assert tried and tried[0]["proposed"] == "646 c.p." and tried[0]["frame"] == "2634 c.c."
    a = out["audit"]
    assert a["frame_conflict"] == 0 and a["charge_rito_mismatch"] == 0 and a["role_flip"] == 0
    assert a["frame_conflict_attempts"] >= 1


def test_case_2_a_frame_giving_1439_is_dropped_and_counted_the_fallback_fills_never_1439(monkeypatch):
    out = _run(monkeypatch, DIRECTOR, _director_reply("1439 c.c."), **_director_world())
    st, a = out["state"], out["audit"]
    assert st["charge"]["article"] == "646 c.p." and st["charge"]["rule"] == "read_hypothesis"
    assert "charged" not in st["records"]["1439 c.c."]
    assert a["charge_rito_dropped"] == 1 and a["frame_unverified.addebito_o_domanda.articolo"] == 1
    assert a["frame_fallback.addebito_o_domanda"] == 1 and a["frame_audit"]["filled"]["addebito_o_domanda"] == "646 c.p."
    assert a["charge_rito_mismatch"] == 0 and a["frame_conflict"] == 0
    assert out["posture"]["role"] == "imputato"          # the other fields stand


def test_case_3_two_parties_on_opposite_sides_one_client(monkeypatch):
    out = _run(monkeypatch, TWO_SIDES, _two_sides_reply({"nome": "Anna Pino", "perche": "«Assisto Anna Pino»"}),
               extraction_client="Bruno Faggio")
    assert out["client"] == "Anna Pino" and out["posture"]["role"] == "attore"
    assert out["header"].startswith("POSIZIONE: Anna Pino, ")
    assert out["clients"] == []                          # one client: the single-client parere
    assert flow.client_of(out["state"]) == "Anna Pino"
    assert out["audit"]["role_flip"] == 0 and out["audit"]["frame_conflict"] == 0
    assert "Bruno Faggio" not in out["header"] and out["state"]["analysis"]["parte_assistita"] == "Anna Pino"


def test_case_4_questions_a_carico_di_x_x_is_the_counterpartys_subject_the_client_whom_they_serve(monkeypatch):
    out = _run(monkeypatch, A_CARICO, _a_carico_reply({"nome": "Carla Ulivo",
                                                      "perche": "«Carla Ulivo può costituirsi parte civile?»"}))
    assert out["client"] == "Carla Ulivo" and out["posture"]["role"] == "persona_offesa"
    assert out["header"] == "POSIZIONE: Carla Ulivo, rito penale, role persona_offesa, stage primo_grado"
    assert out["intake"]["questions"] == ["Quali reati sono configurabili a carico di Dario Cedro?",
                                          "Carla Ulivo può costituirsi parte civile?"]
    # the model naming Dario Cedro as the client: dropped; the rules' assumption never takes the counterparty.
    # v79.14.1 (deliberate change): the fallback for the dropped client is item 1's assumption -- the party on the
    # verified role's side (persona offesa: Carla Ulivo), not 79.14.0's honesty line (which followed from the old
    # assumption picking the party «a carico», the controparte); with no such party, the honesty line as before
    dario = {"nome": "Dario Cedro", "perche": "«Quali reati sono configurabili a carico di Dario Cedro?»"}
    out = _run(monkeypatch, A_CARICO, _a_carico_reply(dario), asked=True,
               parties=(("Carla Ulivo", "persona offesa"), ("Dario Cedro", "imputato")))
    assert out["client"] != "Dario Cedro" and "Dario Cedro" not in out["header"]
    assert out["client"] == "Carla Ulivo" and "Carla Ulivo" in out["line"] and "Dario" not in out["line"]
    assert out["posture"]["role"] == "persona_offesa"
    assert out["audit"]["role_flip"] == 0 and out["audit"]["frame_conflict"] == 0
    assert not out["audit"]["frame_audit"]["role"]["client_is_counterparty"]
    alone = _run(monkeypatch, A_CARICO, _a_carico_reply(dario), asked=True, parties=(("Dario Cedro", "imputato"),))
    assert alone["client"] == "" and alone["line"] == C.CLIENT_UNKNOWN
    assert "Dario Cedro" not in alone["header"] and flow.client_of(alone["state"]) == ""
    assert not alone["audit"]["frame_audit"]["role"]["client_is_counterparty"] and alone["audit"]["role_flip"] == 0


NO_ROLE_REPLY = {"parti": [{"nome": "Volpe S.r.l.", "lato": "terzo"}, {"nome": "Lupo S.r.l.", "lato": "terzo"}],
                 "cliente": None,
                 "ruolo": {"valore": "sconosciuto", "perche": "«Si chiede una valutazione della vicenda»"},
                 "rito": {"valore": "civile", "perche": "«controversia sul pagamento di alcune fatture»"},
                 "stadio": {"valore": "sconosciuto", "perche": "«Si chiede una valutazione della vicenda»"},
                 "quesiti": []}


def test_case_5_no_client_signal_sconosciuto_the_honesty_line_neither_sides_wording(monkeypatch):
    parties = (("Volpe S.r.l.", "convenuta"), ("Lupo S.r.l.", "attrice"))
    off = _run(monkeypatch, NO_CLIENT, NO_ROLE_REPLY, on=False, asked=True, parties=parties)
    # flag off: the rules assume a party -- v79.14.1 (item 1, deliberate change): the one on the resolved role's side
    # (resolve's reading «attore»: the attrice), never the other side (79.14.0 took the convenuta: a role flip)
    assert off["client"] == "Lupo S.r.l." and off["posture"]["role"] == "attore"
    out = _run(monkeypatch, NO_CLIENT, NO_ROLE_REPLY, asked=True, parties=parties)
    assert out["intake"]["role"] == "sconosciuto" and out["intake"]["confidence"]["role"] == intake.FRAME
    assert out["posture"]["role"] == "sconosciuto" and out["posture"]["entry"] is None
    assert out["client"] == "" and out["line"] == C.CLIENT_UNKNOWN          # the honesty line
    assert out["header"] == "POSIZIONE: la parte assistita, rito civile"   # no role, the stage the text's silence
    assert "sconosciuto" not in M.KNOWN_ROLES                               # the method: neither side's wording
    assert out["audit"]["role_flip"] == 0 and out["audit"]["frame_conflict"] == 0
    # resolve would have derived a role: it lost, logged
    assert any(c["step"] == "intake.resolve" for c in out["audit"]["frame_audit"]["conflicts"])


def test_case_5_a_client_named_with_no_role_gets_the_honesty_line_with_its_name(monkeypatch):
    text = "Assisto la società Volpe S.r.l. " + NO_CLIENT
    reply = {**NO_ROLE_REPLY, "parti": [{"nome": "Volpe S.r.l.", "lato": "cliente"},
                                        {"nome": "Lupo S.r.l.", "lato": "terzo"}],
             "cliente": {"nome": "Volpe S.r.l.", "perche": "«Assisto la società Volpe S.r.l.»"}}
    out = _run(monkeypatch, text, reply)
    assert out["client"] == "Volpe S.r.l." and out["posture"]["role"] == "sconosciuto"
    assert out["line"] == C.fill(C.ROLE_UNKNOWN_CLIENT, client="Volpe S.r.l.")
    assert "{client}" not in out["line"] and "Volpe S.r.l." in out["line"]


# --- 2. a later step disagreeing --------------------------------------------------------------------------------------

def test_a_later_step_disagreeing_logs_frame_conflict_and_the_frame_wins(monkeypatch):
    seen = []
    monkeypatch.setattr(cf, "_log", lambda event, data: seen.append((event, data)))
    real = intake._resolve

    def flips(result, analysis):                    # a later step that would make the debtor the claimant
        out = real(result, analysis)
        return {**out, "role": "attore", "rito": "lavoro", "client": {**(out.get("client") or {}), "side": "victim"}}
    monkeypatch.setattr(intake, "_resolve", flips)
    out = _run(monkeypatch, DEBTOR, _debtor_reply())
    assert out["posture"]["role"] == "convenuto" and out["posture"]["rito"] == "civile"
    assert out["state"]["intake"]["client"]["side"] == "accused"
    steps = {(d["field"], d["step"]) for e, d in seen if e == "frame_conflict"}
    assert ("ruolo", "intake.resolve") in steps and ("rito", "intake.resolve") in steps
    assert out["audit"]["frame_conflict"] == 0 and out["audit"]["frame_conflict_attempts"] >= 2
    # posture.of's own disagreement loses too
    state = out["state"]
    state["intake"] = {**state["intake"], "role": "attore", "confidence": {**state["intake"]["confidence"],
                                                                           "role": intake.MODEL}}
    p = posture.of(state, flow._restored_analysis(state["analysis"]))
    assert p["role"] == "convenuto"
    assert any(c["step"] == "posture.of" and c["field"] == "ruolo" for c in state["frame_conflicts"])


def test_the_audit_counts_a_delivered_change_and_a_role_flip(monkeypatch):
    out = _run(monkeypatch, DEBTOR, _debtor_reply())
    state = out["state"]
    state["posture"] = {**state["posture"], "role": "attore"}          # a path that bypassed the binding
    a = cf.audit(state)
    assert a["frame_conflict"] == 1 and a["frame_audit"]["changed"][0]["field"] == "ruolo"
    assert a["role_flip"] == 1
    state["posture"]["role"] = "convenuto"
    state["client"] = {"name": "Delfino S.p.A.", "how": "assunta"}       # the counterparty as the client
    a = cf.audit(state)
    assert a["role_flip"] == 1 and a["frame_audit"]["role"]["client_is_counterparty"]
    # a civil article held as a penal charge
    state["client"] = {"name": "Orsa S.r.l."}
    state["posture"] = {**state["posture"], "family": "penale", "rito": "penale"}
    state["case_frame"]["frame"]["rito"] = None
    state["charge"] = {"article": "1439 c.c.", "rule": "user_article"}
    assert cf.audit(state)["charge_rito_mismatch"] == 1


# --- 3. a failed call, the flag off -----------------------------------------------------------------------------------

def _view(out):
    got = {k: out["intake"].get(k) for k in ("rito", "role", "stage", "questions", "client", "resisting", "deliverable",
                                              "route")}
    p = {k: v for k, v in out["posture"].items()}
    return got, p, out["client"], out["line"], out["header"], out["state"].get("charge")


def test_a_failed_call_every_field_to_the_rules_the_output_as_with_the_flag_off(monkeypatch):
    for text, kw in ((DEBTOR, {}), (DIRECTOR, _director_world()), (NO_CLIENT, {"asked": True, "parties": (
            ("Volpe S.r.l.", "convenuta"), ("Lupo S.r.l.", "attrice"))})):
        off = _run(monkeypatch, text, _boom, on=False, **kw)
        on = _run(monkeypatch, text, _boom, **kw)
        assert on["state"]["case_frame"]["status"] == cf.FAILED
        assert _view(on) == _view(off)
        assert on["audit"]["frame_failed"] == 1 and on["audit"]["frame_fields_verified"] == 0
        assert on["audit"]["frame_conflict"] == 0 and on["audit"]["frame_conflict_attempts"] == 0


def test_flag_off_nothing_runs_no_frame_anywhere(monkeypatch):
    calls = []
    out = _run(monkeypatch, DEBTOR, lambda m, k: calls.append(1) or "{}", on=False)
    assert calls == [] and "case_frame" not in out["intake"] and "case_frame" not in out["state"]
    assert "case_frame" not in intake.log_view(out["intake"])
    assert cf.audit(out["state"]) == {} and cf.held(out["state"]) is None
    assert out["intake"] == intake.safe_detect(DEBTOR, "", [])       # the intake as 79.13.0's
    assert "case_frame_audit" not in str(flow.case_counters(out["state"]))
    assert not any(k.startswith("frame_") for k in flow.case_counters(out["state"]))


def test_write_answer_with_a_failed_frame_is_byte_identical_to_the_flag_off(monkeypatch):
    import test_v60f as v60f
    writer = lambda jobs: [(t, v60f.WRITER.get(t, f"Testo di {t}.")) for t, *_ in jobs]
    monkeypatch.delenv("CASE_FRAME", raising=False)
    off_state = v60f._graded_state(monkeypatch)
    off, _ = v60f._answer(monkeypatch, off_state, writer)
    monkeypatch.setenv("CASE_FRAME", "true")
    on_state = v60f._graded_state(monkeypatch)
    on_state["case_frame"] = cf.read(on_state["text"], call=_boom)
    on, _ = v60f._answer(monkeypatch, on_state, writer)
    assert on == off
    assert on_state["posture"] == off_state["posture"] and on_state.get("charge") == off_state.get("charge")


# --- 4. multi-client, the follow-up -----------------------------------------------------------------------------------

def test_multi_client_one_verified_client_is_one_client_a_dropped_one_leaves_the_rules(monkeypatch):
    two = [{"name": "Anna Pino", "role": ""}, {"name": "Bruno Faggio", "role": ""}]
    monkeypatch.setattr(flow, "_clients_of", lambda state: list(two))
    out = _run(monkeypatch, TWO_SIDES, _two_sides_reply({"nome": "Anna Pino", "perche": "«Assisto Anna Pino»"}))
    assert out["clients"] == []
    assert any(c["step"] == "clients_of" for c in out["audit"]["frame_audit"]["conflicts"])
    both = [{"nome": "Anna Pino", "perche": "«Assisto Anna Pino»"},
            {"nome": "Bruno Faggio", "perche": "«Faggio non ha restituito la somma»"}]
    reply = _two_sides_reply(both)
    reply["parti"] = [dict(p, lato="cliente") for p in reply["parti"]]     # neither listed on the other side
    out = _run(monkeypatch, TWO_SIDES, reply)                            # the frame's two clients: dropped
    assert out["clients"] == two                                         # the multi-client rules decide
    assert out["audit"]["frame_unverified.cliente"] == 1
    # (review) a rules' client the frame lists as controparte is never one of several clients
    out = _run(monkeypatch, TWO_SIDES, _two_sides_reply(both))
    assert out["clients"] == []
    assert any(c["step"] == "clients_of" for c in out["audit"]["frame_audit"]["conflicts"])


def test_a_follow_up_reuses_the_stored_frame_and_a_new_case_reads_its_own(monkeypatch):
    out = _run(monkeypatch, DEBTOR, _debtor_reply())
    state = out["state"]
    assert len(out["model"].seen) == 1
    monkeypatch.setattr(cf, "_default_call", _boom)                    # a second call would fail: none is made
    state["answered"] = {"parte_assistita": "Delfino S.p.A."}          # a reply naming the other party
    analysis = flow._restored_analysis(state["analysis"])
    client, _ = flow.settle_client(state, analysis)
    p = posture.of(state, analysis)
    assert client == "Orsa S.r.l." and p["role"] == "convenuto"
    assert state["case_frame"]["status"] == cf.OK
    assert any(c["step"] == "settle_client" and c["proposed"] == "Delfino S.p.A." for c in state["frame_conflicts"])
    # a new case is a new request: its own intake, its own frame
    monkeypatch.setattr(cf, "_default_call", _model(_director_reply()))
    fresh = intake.request_intake(DIRECTOR)
    assert fresh["case_frame"]["frame"]["cliente"]["nome"] == "Marco Lince" and fresh["role"] == "imputato"


def test_the_frames_institution_is_the_first_one_module_activation_reads(monkeypatch):
    out = _run(monkeypatch, DEBTOR, _debtor_reply(), read={"hypotheses": [
        {"line": "principale", "claim": "la penale", "institutions": ["clausola penale"], "rubricated": []}]})
    insts = flow.read_institutions(out["state"]["read"])
    assert insts[0]["name"] == "contratto di fornitura" and insts[0]["frame"] is True
    assert [x["name"] for x in insts] == ["contratto di fornitura", "clausola penale"]
    assert M.module_input("", insts)["institutions"][0]["name"] == "contratto di fornitura"
    assert out["audit"]["frame_conflict"] == 0
    # analyse_case's pending read carries it from the start (the retrieval stage's institutions)

    class Pending:
        ms, wait_ms, llm, timed_out = 5, 0, 1, False

        def result(self):
            return {"hypotheses": []}
    framed = flow._FramedRead(Pending(), flow.frame_institution(out["state"]["case_frame"]))
    assert framed.result()["frame_institution"]["testo"] == "contratto di fornitura" and framed.llm == 1
    monkeypatch.delenv("CASE_FRAME", raising=False)
    assert flow.read_institutions({"hypotheses": [{"line": "principale", "institutions": ["clausola penale"]}]}) == \
        [{"name": "clausola penale", "rubricated": "", "claim": ""}]


# --- 5. the counters, the nightly -------------------------------------------------------------------------------------

def test_the_counters_reach_case_counters_on_the_delivered_answer(monkeypatch):
    out = _run(monkeypatch, DIRECTOR, _director_reply("1439 c.c."), **_director_world())
    counters = flow.case_counters(out["state"])
    for key in ("frame_fields_verified", "frame_unverified.addebito_o_domanda", "frame_fallback.addebito_o_domanda",
                "frame_conflict", "frame_conflict_attempts", "role_flip", "charge_rito_mismatch", "charge_rito_dropped",
                "frame_failed", "frame_seconds"):
        assert key in counters, key
    assert counters["frame_conflict"] == 0 and counters["role_flip"] == 0 and counters["charge_rito_mismatch"] == 0
    assert "frame_audit" not in counters


def test_the_nightly_prints_each_field_with_its_quote_or_its_fallback_and_fails_a_flip(monkeypatch):
    import nightly_sim
    out = _run(monkeypatch, DIRECTOR, _director_reply("1439 c.c."), **_director_world())
    fr = nightly_sim.frame_of(out["state"], DIRECTOR)
    assert fr["source"] == "pipeline" and fr["counters"]["charge_rito_dropped"] == 1
    text = "\n".join(fr["lines"])
    assert "ruolo: imputato/indagato (intake imputato) — «la Procura indaga per infedeltà patrimoniale»" in text
    assert "istituto: — (fallback: appropriazione indebita)" in text           # the first read's, the rules'
    assert "addebito_o_domanda: articolo — (fallback: 646 c.p.); infedeltà patrimoniale" in text
    assert "[scartato: articolo: in un rito penale «1439 c.c.» non è una norma penale]" in text
    counters = flow.case_counters(out["state"])
    assert nightly_sim.frame_check(fr, counters)["ok"]
    assert nightly_sim.frame_check({"ran": False}, {})["why"] == "not run"
    bad = nightly_sim.frame_check(fr, {**counters, "role_flip": 1})
    assert not bad["ok"] and "role_flip 1" in bad["why"]
    lines = "\n".join(nightly_sim.frame_lines([{"id": "cell_d", "case_frame": fr, "counters": counters}]))
    assert "**cell_d**" in lines and "| cell_d | 7 | 0 | 1 | 0 | 0 | 0 | 0 |" in lines
    framed = _run(monkeypatch, DIRECTOR, _director_reply("2634 c.c."), **_director_world())
    lines = "\n".join(nightly_sim.frame_lines([{"id": "cell_e", "case_frame": nightly_sim.frame_of(
        framed["state"], DIRECTOR), "counters": {}}]))
    assert "- conflitto (posture.choose_charge): addebito_o_domanda «646 c.p.» → quadro «2634 c.c.»" in lines
    # a field the frame left: printed with the value the rules filled it with
    fb = _run(monkeypatch, DEBTOR, _debtor_reply(ruolo={"valore": "attore", "perche": "«Orsa chiede i danni»"}))
    fr = nightly_sim.frame_of(fb["state"], DEBTOR)
    assert any(x.startswith(f"ruolo: — (fallback: {fb['posture']['role']})") and "[scartato:" in x
               for x in fr["lines"])


def test_analyse_case_gets_the_frame_only_with_the_flag_and_a_frame(monkeypatch):
    monkeypatch.setenv("CASE_FRAME", "true")
    monkeypatch.setattr(cf, "_default_call", _model(_debtor_reply()))
    got = intake.request_intake(DEBTOR)
    assert flow.framed_args(got) == {"frame": got["case_frame"]}
    assert flow.framed_args(None) == {} and flow.framed_args({"rito": "civile"}) == {}
    monkeypatch.delenv("CASE_FRAME", raising=False)
    assert flow.framed_args(got) == {}                  # the call as 79.13.0's: no keyword at all


# --- 6. (review) the binding's holes closed ---------------------------------------------------------------------------

def test_review_a_frame_rito_sconosciuto_never_lets_1439_be_the_penal_charge(monkeypatch):
    reply = _director_reply("1439 c.c.")
    reply["rito"] = {"valore": "sconosciuto", "perche": "«ha ricevuto un avviso di garanzia»"}
    out = _run(monkeypatch, DIRECTOR, reply, **_director_world())
    st, a = out["state"], out["audit"]
    assert out["posture"]["rito"] == "penale"
    assert st["charge"]["article"] != "1439 c.c." and "charged" not in st["records"]["1439 c.c."]
    assert a["charge_rito_dropped"] == 1 and a["charge_rito_mismatch"] == 0 and a["frame_conflict"] == 0


PENAL_NO_RITO = ("Marco Lince ha ottenuto con raggiri da Pia Merlo la firma di una vendita a un prezzo inferiore al "
                 "valore. Cosa rischia Marco Lince?")


def test_review_a_civil_frame_charge_with_no_rito_to_check_it_is_refused_by_the_penal_posture(monkeypatch):
    monkeypatch.setenv("CASE_FRAME", "true")
    assert intake.rito_rule(PENAL_NO_RITO, "") == ("", "")             # no rito for the frame's own check
    reply = {"parti": [{"nome": "Marco Lince", "lato": "cliente"}],
             "cliente": {"nome": "Marco Lince", "perche": "«Cosa rischia Marco Lince?»"},
             "addebito_o_domanda": {"articolo": "1439 c.c.", "descrizione": "raggiri",
                                    "frase": "«ha ottenuto con raggiri da Pia Merlo la firma di una vendita»"},
             "quesiti": ["Cosa rischia Marco Lince?"]}
    monkeypatch.setattr(cf, "_default_call", _model(reply))
    got = intake.request_intake(PENAL_NO_RITO)
    assert got["case_frame"]["frame"]["addebito_o_domanda"]["articolo"] == "1439 c.c."     # verified: no rito here
    state = _state(PENAL_NO_RITO, records={"1439 c.c.": {"article": "1439 c.c.", "rubrica": "Dolo", "requisites": [
        _req(1, "raggiri usati da uno dei contraenti", "da provare")]}})
    state["analysis"]["frame"] = {"areas": ["penale"]}
    flow.take_intake(state, got, PENAL_NO_RITO)
    p = posture.of(state, flow._restored_analysis(state["analysis"]))
    state["posture"] = flow._jsonable(p)
    assert p["family"] == "penale"
    posture.choose_charge(state, p, state["records"], PENAL_NO_RITO)
    assert state["charge"]["article"] != "1439 c.c." and state["charge"]["rule"] != posture.FRAME_CHARGE_RULE
    assert cf.bound(state, "addebito_o_domanda", p) is None and state["frame_charge_refused"] == "1439 c.c."
    a = cf.audit(state)
    assert a["charge_rito_mismatch"] == 0 and a["charge_rito_dropped"] == 1 and a["frame_conflict"] == 0
    # v79.19.0 (item 2): the refusal is a correction written once into the one frame (state["frame"]) -- every later
    # read, whatever posture it passes, sees the corrected value; the correction is logged old -> new
    assert cf.bound({**state, "frame_charge_refused": None}, "addebito_o_domanda",
                    {"rito": "civile", "family": "civile"}) is None
    made = cf.corrections(state)
    assert [(m["field"], m["old"], m["new"], m["step"]) for m in made] == \
        [("addebito_o_domanda.articolo", "1439 c.c.", "", "posture")]
    assert a["frame_corrections_applied"] == 1
    # a civil posture on a fresh state keeps the same frame charge
    fresh = _state(PENAL_NO_RITO, records={})
    flow.take_intake(fresh, intake.request_intake(PENAL_NO_RITO), PENAL_NO_RITO)
    assert cf.bound(fresh, "addebito_o_domanda", {"rito": "civile", "family": "civile"}) == "1439 c.c."
    assert cf.corrections(fresh) == []


def test_review_no_frame_client_means_no_party_served_anywhere_not_the_extractions(monkeypatch):
    dario = {"nome": "Dario Cedro", "perche": "«Quali reati sono configurabili a carico di Dario Cedro?»"}
    # v79.14.1: with Carla Ulivo among the parties item 1's fallback assumes her (test_case_4_...): here the facts
    # hold the counterparty alone -- no party to assume, the extraction's client (Dario Cedro) refused
    out = _run(monkeypatch, A_CARICO, _a_carico_reply(dario), asked=True, extraction_client="Dario Cedro",
               parties=(("Dario Cedro", "imputato"),))
    st = out["state"]
    assert out["line"] == C.CLIENT_UNKNOWN and out["client"] == ""
    assert flow.client_of(st) == "" and st["analysis"]["parte_assistita"] == ""
    assert any(c["step"] == "settle_client" and c["proposed"] == "Dario Cedro"
               for c in out["audit"]["frame_audit"]["conflicts"])
    analysis = flow._restored_analysis(st["analysis"])
    plan = flow.frame_plan(st, flow._fallback_plan(st, analysis, [], []))
    assert plan["parte_assistita"] == "" and st["plan_client"] == ""
    model_plan = flow.frame_plan(st, {"parte_assistita": "Dario Cedro", "posizione": "imputato"})   # the model's
    assert model_plan["parte_assistita"] == ""
    assert "Dario Cedro" not in out["header"]
    assert out["audit"]["role_flip"] == 0
    # a reader still serving the counterparty is caught
    st["analysis"]["parte_assistita"] = "Dario Cedro"
    assert cf.audit(st)["role_flip"] == 1
    # the unknown role likewise: the extraction's client is no client
    parties = (("Volpe S.r.l.", "convenuta"), ("Lupo S.r.l.", "attrice"))
    out = _run(monkeypatch, NO_CLIENT, NO_ROLE_REPLY, asked=True, parties=parties, extraction_client="Volpe S.r.l.")
    assert out["line"] == C.CLIENT_UNKNOWN and flow.client_of(out["state"]) == ""
    assert out["header"] == "POSIZIONE: la parte assistita, rito civile"
    assert flow.clients_of(out["state"]) == []


def test_review_the_extractions_position_takes_the_frame_roles_side(monkeypatch):
    from src.rag import defensive_generation as dg, fact_signals

    def world(state):
        state["analysis"]["posizione"] = "attore, danneggiato dal ritardo"
    real = flow.take_intake

    def take(state, got, message, request=""):
        world(state)
        return real(state, got, message, request)
    monkeypatch.setattr(flow, "take_intake", take)
    out = _run(monkeypatch, DEBTOR, _debtor_reply())
    st = out["state"]
    assert st["analysis"]["posizione"] == "convenuto"
    assert fact_signals.client_side(dg._stance_party(st["analysis"])) == "accused"
    assert any(c["step"] == "analysis.posizione" for c in out["audit"]["frame_audit"]["conflicts"])
    plan = flow.frame_plan(st, {"parte_assistita": "Orsa S.r.l.", "posizione": "attore che chiede i danni"})
    assert plan["posizione"] == "convenuto" and st["plan_client"] == "Orsa S.r.l."
    agreeing = flow.frame_plan(st, {"parte_assistita": "Orsa S.r.l.", "posizione": "convenuta, debitrice della penale"})
    assert agreeing["posizione"] == "convenuta, debitrice della penale"       # the same side: its own words stand
    assert out["audit"]["frame_conflict"] == 0
    # flag off: the extraction's position untouched
    off = _run(monkeypatch, DEBTOR, _debtor_reply(), on=False)
    assert off["state"]["analysis"]["posizione"] == "attore, danneggiato dal ritardo"


def test_review_a_frame_rito_out_of_the_corpus_keeps_the_out_of_scope_copy(monkeypatch):
    text = ("Assisto la società Airone S.r.l., che ha ricevuto dall'Agenzia delle Entrate un avviso di accertamento per "
            "IVA non versata di 40.000 euro relativo al 2023. Redigi un parere: come può impugnare l'avviso di "
            "accertamento davanti alla corte di giustizia tributaria?")
    reply = {"parti": [{"nome": "Airone S.r.l.", "lato": "cliente"},
                       {"nome": "Agenzia delle Entrate", "lato": "controparte"}],
             "cliente": {"nome": "Airone S.r.l.", "perche": "«Assisto la società Airone S.r.l.»"},
             "rito": {"valore": "tributario", "perche": "«davanti alla corte di giustizia tributaria»"}}
    monkeypatch.setattr(cf, "_default_call", _model(reply))
    monkeypatch.delenv("CASE_FRAME", raising=False)
    off = intake.request_intake(text)
    monkeypatch.setenv("CASE_FRAME", "true")
    on = intake.request_intake(text)
    assert on["confidence"]["rito"] == intake.FRAME and on["rito"] == "tributario"
    assert on["route"] == off["route"] == intake.OUT_OF_SCOPE


def test_review_one_verified_client_among_the_rules_several_keeps_the_multi_client_parere(monkeypatch):
    text = ("Assisto Rossi e Bianchi, entrambi indagati per truffa aggravata ai danni di Pia Merlo per la vendita di "
            "un terreno. Quali difese hanno Rossi e Bianchi?")
    reply = {"parti": [{"nome": "Rossi", "lato": "cliente"}, {"nome": "Bianchi", "lato": "cliente"},
                       {"nome": "Pia Merlo", "lato": "controparte"}],
             "cliente": {"nome": "Rossi", "perche": "«Assisto Rossi»"},
             "ruolo": {"valore": "indagato", "perche": "«entrambi indagati per truffa aggravata»"}}
    off = _run(monkeypatch, text, reply, on=False)
    out = _run(monkeypatch, text, reply)
    assert [c["name"] for c in off["clients"]] == ["Rossi", "Bianchi"]
    assert out["clients"] == off["clients"]
    assert not any(c["step"] == "clients_of" for c in out["audit"]["frame_audit"]["conflicts"])


def test_review_the_rules_parte_civile_refines_the_frames_persona_offesa(monkeypatch):
    text = ("Sono il difensore di Carla Ulivo, persona offesa, che si è costituita parte civile nel processo penale "
            "contro Dario Cedro per lesioni personali. Quali danni può chiedere Carla Ulivo?")
    reply = {"parti": [{"nome": "Carla Ulivo", "lato": "cliente"}, {"nome": "Dario Cedro", "lato": "controparte"}],
             "cliente": {"nome": "Carla Ulivo", "perche": "«Sono il difensore di Carla Ulivo»"},
             "ruolo": {"valore": "persona_offesa", "perche": "«si è costituita parte civile»"},
             "rito": {"valore": "penale", "perche": "«nel processo penale»"}}
    off = _run(monkeypatch, text, reply, on=False)
    out = _run(monkeypatch, text, reply)
    assert off["intake"]["role"] == "parte_civile"
    assert out["intake"]["role"] == "parte_civile" and out["intake"]["confidence"]["role"] == intake.FRAME
    assert out["posture"]["role"] == "parte_civile" and cf.bound(out["state"], "ruolo") == "parte_civile"
    assert out["audit"]["frame_conflict"] == 0 and out["audit"]["role_flip"] == 0
    # another side is never a refinement: the frame's role stands
    reply["ruolo"] = {"valore": "imputato", "perche": "«nel processo penale»"}
    assert _run(monkeypatch, text, reply)["intake"]["role"] == "imputato"


def test_review_a_plain_question_with_no_facts_makes_no_frame_call(monkeypatch):
    calls = []
    out = _run(monkeypatch, DEBTOR, lambda m, k: calls.append(1) or json.dumps(_debtor_reply()))
    assert len(calls) == 1 and out["state"]["case_frame"]["status"] == cf.OK
    got = intake.request_intake("Cosa prevede l'art. 2043 c.c.?")
    assert len(calls) == 1 and "case_frame" not in got and got["route"] == intake.RAG
