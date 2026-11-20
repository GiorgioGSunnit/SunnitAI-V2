"""v79.14.1 merged onto 79.14.0 -- item 1's role rules under the case frame (CASE_FRAME), one role_flip, items 2 and 3
with the frame. Synthetic request texts only (invented names and facts; never a real, eval or tester case); the
frame's model faked; no network, no Neo4j.

(a) the verified frame wins over every item-1 rule (intake.resolve's resisting reading, role_for_client,
    apply_intake_client, client_on_intake_side, plan_client): those are the fallback for a field the frame left empty
    or that failed verification; a disagreement logs frame_conflict and loses. Flag off: item 1 as built.
(b) ONE role_flip: item 1's reasons (interactive_flow.role_flip_reasons) and case_frame.audit's own as one set -- the
    same flip read by both definitions counted once; the nightly fails a cell on it once.
(c) items 2 and 3 with the frame: the frame's institution feeds governing_law (the honesty line); a frame-verified
    «amministrativo» rito is never the administrative-act signal by itself.
"""

import importlib.util
import os
import sys
from pathlib import Path

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from src.rag import case_frame as cf  # noqa: E402
from src.rag import family_gate  # noqa: E402
from src.rag import intake  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import lawyer_method as M  # noqa: E402
from src.rag import posture  # noqa: E402
from test_v79_14_flow import _run  # noqa: E402


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# the debtor resisting a penale (invented): the rules read him «convenuto» (the demand against him)
DEBT = ("Assisto Carlo Merlo. La società Gufo S.r.l. chiede a Carlo Merlo il pagamento della penale di 9.000 euro per "
        "il ritardo nella posa del parquet. Carlo Merlo vuole sapere come contestare la penale.")
PARTIES = (("Carlo Merlo", "posatore"), ("Gufo S.r.l.", "committente"))


def _reply(**over):
    reply = {
        "parti": [{"nome": "Carlo Merlo", "ruolo_nel_fatto": "ha posato il parquet", "lato": "cliente"},
                  {"nome": "Gufo S.r.l.", "ruolo_nel_fatto": "chiede la penale", "lato": "controparte"}],
        "cliente": {"nome": "Carlo Merlo", "perche": "«Assisto Carlo Merlo»"},
        "ruolo": {"valore": "convenuto", "perche": "«chiede a Carlo Merlo il pagamento della penale»"},
        "rito": {"valore": "civile", "perche": "«il pagamento della penale di 9.000 euro»"},
        "stadio": {"valore": "pre_filing", "perche": "«vuole sapere come contestare la penale»"},
        "addebito_o_domanda": {"articolo": "1382 c.c.", "descrizione": "la penale per il ritardo",
                               "frase": "«chiede a Carlo Merlo il pagamento della penale»"},
        "istituto": {"testo": "penale", "frase": "«il pagamento della penale di 9.000 euro»"},
        "quesiti": []}
    reply.update(over)
    return reply


def _steps(out):
    return {c["step"] for c in out["audit"]["frame_audit"]["conflicts"]}


# --- (a) the frame wins; item 1's rules are the fallback ----------------------------------------------------------------

def test_a_the_debtor_with_the_frame_verified_takes_the_frames_role(monkeypatch):
    out = _run(monkeypatch, DEBT, _reply(), parties=PARTIES)
    assert out["intake"]["confidence"]["role"] == intake.FRAME
    assert out["posture"]["role"] == "convenuto" and out["posture"]["intake_role"] == "convenuto"
    assert out["client"] == "Carlo Merlo" and "role convenuto" in out["header"]
    a = out["audit"]
    assert a["frame_conflict"] == 0 and a["role_flip"] == 0 and a["frame_fallback.ruolo"] == 0


def test_a_a_verified_frame_attore_beats_resolves_resisting_rule_logged_as_losing(monkeypatch):
    # the frame verified «attore» (its quote is in the text) while item 1's resolve reads the resisting debtor: the
    # frame wins, the rule's «convenuto» logged frame_conflict and loses -- the delivered answer carries the frame's
    reply = _reply(ruolo={"valore": "attore", "perche": "«Carlo Merlo vuole sapere come contestare la penale»"})
    out = _run(monkeypatch, DEBT, reply, parties=PARTIES)
    assert out["intake"]["role"] == "attore" and out["intake"]["confidence"]["role"] == intake.FRAME
    assert out["posture"]["role"] == "attore" and out["posture"]["intake_role"] == "attore"
    a = out["audit"]
    assert a["frame_conflict"] == 0 and a["role_flip"] == 0          # what was delivered is the frame's
    assert a["frame_conflict_attempts"] >= 1                          # the losing rules, by name
    lost = [c for c in a["frame_audit"]["conflicts"] if c["field"] == "ruolo"]
    assert lost and all(c["frame"] == "attore" for c in lost) and {c["proposed"] for c in lost} == {"convenuto"}
    # the same intake through resolve directly: item 1's rule fires on the model «attore» it would own, and loses
    raw = {"rito": "civile", "role": "attore", "stage": "pre_filing", "confidence": {"role": intake.FRAME},
           "why": {"role": ""}, "client": {"name": "Carlo Merlo", "side": "victim"}}
    got = intake.resolve(raw, {flow.CASE_KEY: DEBT})
    assert got["role"] == "attore" and got["confidence"]["role"] == intake.FRAME and not got.get("resisting")
    assert [c["step"] for c in got["frame_conflicts"]] == ["intake.resolve"]
    as_model = intake.resolve({**raw, "confidence": {"role": intake.MODEL}}, {flow.CASE_KEY: DEBT})
    assert as_model["role"] == "convenuto" and not as_model.get("frame_conflicts")      # flag off / no frame: item 1


def test_a_with_the_frames_role_dropped_item_ones_fallback_gives_convenuto(monkeypatch):
    # the frame's role quoting a sentence the text does not hold: dropped -- the rules and item 1 decide: convenuto
    reply = _reply(ruolo={"valore": "attore", "perche": "«Carlo Merlo chiede i danni a Gufo»"})
    out = _run(monkeypatch, DEBT, reply, parties=PARTIES)
    assert out["intake"]["confidence"]["role"] != intake.FRAME
    assert out["posture"]["role"] == "convenuto" and out["posture"]["resisting"]
    a = out["audit"]
    assert a["frame_unverified.ruolo"] == 1 and a["frame_fallback.ruolo"] == 1
    assert a["frame_audit"]["filled"]["ruolo"] == "convenuto" and a["role_flip"] == 0 and a["frame_conflict"] == 0
    # flag off: the same role, item 1 as built
    off = _run(monkeypatch, DEBT, reply, on=False, parties=PARTIES)
    assert off["posture"]["role"] == "convenuto" and off["audit"] == {}


def test_a_role_for_client_gives_the_frames_client_the_frames_role(monkeypatch):
    out = _run(monkeypatch, DEBT, _reply(), parties=PARTIES)
    st, p = out["state"], out["posture"]
    # the frame's client: the frame's role («quadro»), whatever the first read's word
    assert flow.role_for_client(st, p, {"name": "Carlo Merlo", "role": "attore"}, DEBT) == ("convenuto",
                                                                                           flow.ROLE_FROM_FRAME)
    # another client: item 1's rule as built (the intake's role; no frame binding for it)
    assert flow.role_for_client(st, p, {"name": "Ivo Tasso", "role": "attore"}, DEBT) == ("convenuto",
                                                                                         flow.ROLE_FROM_INTAKE)
    # the request stating the frame client's role otherwise: logged frame_conflict «role_for_client», it loses
    text = DEBT + " Carlo Merlo ha citato in giudizio Gufo S.r.l."
    role, won = flow.role_for_client(st, p, {"name": "Carlo Merlo", "role": ""}, text)
    assert (role, won) == ("convenuto", flow.ROLE_FROM_FRAME)
    assert any(c["step"] == "role_for_client" and c["frame"] == "convenuto" for c in st["frame_conflicts"])
    # flag off: item 1's own rule decides
    monkeypatch.delenv("CASE_FRAME", raising=False)
    assert flow.role_for_client(st, p, {"name": "Carlo Merlo", "role": ""}, text)[1] == flow.ROLE_FROM_REQUEST


def test_a_client_on_intake_side_never_moves_the_frames_client(monkeypatch):
    # the extraction reads the frame's client on the claiming side (creditore) and a third party on the debtor's: item
    # 1 would move the client to it -- the frame's client stands, the move logged frame_conflict and lost
    parties = (("Carlo Merlo", "creditore"), ("Gufo S.r.l.", "debitore"), ("Ivo Tasso", "debitore"))
    out = _run(monkeypatch, DEBT, _reply(), parties=parties)
    st = out["state"]
    analysis = flow._restored_analysis(st["analysis"])
    assert flow.client_on_intake_side(st, analysis, "Carlo Merlo") is None
    check = st["role_check"]
    assert check["frame_client"] == "Carlo Merlo" and not check["stated_by_user"] and not check["moved"]
    assert any(c["step"] == "client_on_intake_side" and c["proposed"] == "Ivo Tasso" for c in st["frame_conflicts"])
    assert flow.client_of(st) == "Carlo Merlo"
    assert flow.role_counted(st, out["posture"])["role_flip"] == 0      # the frame verified client and role: no flip
    # never a move to a party the frame lists as controparte: without the third party nothing moves, nothing logged
    out = _run(monkeypatch, DEBT, _reply(), parties=parties[:2])
    st = out["state"]
    assert flow.client_on_intake_side(st, flow._restored_analysis(st["analysis"]), "Carlo Merlo") is None
    assert not any(c["step"] == "client_on_intake_side" for c in st.get("frame_conflicts") or [])
    # flag off: item 1 as built -- the client nobody named, read on the other side, moves to the debtor's side
    raw = {"rito": "civile", "role": "convenuto", "stage": "", "confidence": {"role": "rule"}, "why": {"role": ""},
           "client": {"name": "", "side": ""}}
    monkeypatch.delenv("CASE_FRAME", raising=False)
    plain = {"text": DEBT, "request": "", "intake": raw, "answered": {}, "questions": [], "block_counters": {},
             "analysis": {flow.CASE_KEY: DEBT, "fact_signals": {"facts": {"parties": [
                 {"name": n, "role": r} for n, r in parties]}}}}
    moved = flow.client_on_intake_side(plain, flow._restored_analysis(plain["analysis"]), "Carlo Merlo")
    assert moved and moved[0] == "Gufo S.r.l." and plain["role_check"]["moved"]


def test_a_apply_intake_client_and_plan_client_are_not_consulted_under_a_frame_client(monkeypatch):
    out = _run(monkeypatch, DEBT, _reply(), parties=PARTIES)
    st = out["state"]
    assert flow.case_frame_client(st) == "Carlo Merlo"
    assert flow.apply_intake_client(st, {"client": ""}) == ""
    calls = []
    monkeypatch.setattr(flow, "plan_client", lambda state, plan: calls.append(plan) or {**plan, "parte_assistita": "X"})
    plan = flow.plan_party(st, {"parte_assistita": "Gufo S.r.l.", "posizione": "attore"}, "")
    assert calls == [] and plan["parte_assistita"] == "Carlo Merlo" and st["method_client"] == "Carlo Merlo"
    assert "plan.parte_assistita" in _steps({"audit": cf.audit(st)})        # the model's party: logged, it loses
    # the frame's client dropped (no quote): plan_client is the fallback
    dropped = _run(monkeypatch, DEBT, _reply(cliente={"nome": "Carlo Merlo", "perche": "«Carlo Merlo è il mio "
                                                                                       "assistito»"}),
                   parties=PARTIES)
    assert flow.case_frame_client(dropped["state"]) == ""
    flow.plan_party(dropped["state"], {"parte_assistita": "Ivo Tasso"}, "")
    assert len(calls) == 1
    # a rules' client the frame lists as controparte is never applied (logged, it loses)
    st2 = dropped["state"]
    st2["intake"] = {**st2["intake"], "client": {"name": "Gufo S.r.l.", "side": ""}}
    st2["answered"] = {}
    assert flow.apply_intake_client(st2, {"client": ""}) == ""
    assert any(c["step"] == "apply_intake_client" for c in st2["frame_conflicts"])


# --- (b) one role_flip --------------------------------------------------------------------------------------------------

def test_b_the_same_flip_read_by_both_definitions_counts_once(monkeypatch):
    out = _run(monkeypatch, DEBT, _reply(), parties=PARTIES)
    st = out["state"]
    # a later step writing «attore» into the posture: item 1 (posture role ≠ intake role) and the audit (the frame's
    # role ≠ the method's) read the same flip -- one
    st["posture"] = {**st["posture"], "role": "attore", "intake_role": "convenuto"}
    flow.role_corpus_counted(st, "Testo.", [], st["posture"])
    assert st["role_flip_reasons"] == [flow.ROLE_FLIP_ROLE] and st["block_counters"]["role_flip"] == 1
    assert cf.audit(st)["role_flip"] == 1
    assert flow.case_counters(st)["role_flip"] == 1
    # the counterparty taken as client too: a second, distinct flip -- read by item 1 (the method writes for another
    # party than the one settled) and by the audit (a served client the frame lists as controparte): counted once
    st["method_client"] = "Gufo S.r.l."
    st["role_check"] = {"client": "Carlo Merlo"}
    flow.role_corpus_counted(st, "Testo.", [], st["posture"])
    st["plan_client"] = "Gufo S.r.l."
    assert sorted(st["role_flip_reasons"]) == sorted([flow.ROLE_FLIP_ROLE, flow.ROLE_FLIP_COUNTERPARTY])
    a = cf.audit(st)
    assert a["role_flip"] == 2 and a["frame_audit"]["role"]["client_is_counterparty"]
    counters = flow.case_counters(st)
    assert counters["role_flip"] == 2
    # the nightly fails the cell on it once: check case_frame, not zero_counters as well
    nightly = _script("nightly_sim")
    checks = [nightly.frame_check({"ran": True}, counters),
              nightly.zero_counters_check(counters, frame_ran=True)]
    failing = [c for c in checks if not c["ok"] and "role_flip" in c["why"]]
    assert len(failing) == 1 and failing[0]["check"] == "case_frame"


def test_b_flag_off_the_counter_is_item_ones_and_zero_counters_fails_it():
    st = {"text": DEBT, "request": "", "block_counters": {}, "role_check": {}, "analysis": {},
          "method_roles": [{"client": "Carlo Merlo", "role": "attore", "intake_role": "convenuto",
                            "from": flow.ROLE_FROM_INTAKE}]}
    flow.role_corpus_counted(st, "Testo.", [], {"role": "convenuto", "intake_role": "convenuto"})
    assert st["role_flip_reasons"] == [f"{flow.ROLE_FLIP_ROLE}:Carlo Merlo"]
    counters = flow.case_counters(st)
    assert counters["role_flip"] == 1 and "frame_conflict" not in counters
    nightly = _script("nightly_sim")
    assert nightly.frame_check({"ran": False}, counters)["ok"]
    bad = nightly.zero_counters_check(counters, frame_ran=False)
    assert not bad["ok"] and "role_flip=1" in bad["why"]


def test_b_a_clean_debtor_answer_counts_zero_in_both(monkeypatch):
    out = _run(monkeypatch, DEBT, _reply(), parties=PARTIES)
    st = out["state"]
    st["method_client"] = "Carlo Merlo"
    st["role_check"] = {"client": "Carlo Merlo", "frame_client": "Carlo Merlo"}
    flow.role_corpus_counted(st, "Testo.", [], st["posture"])
    assert st["block_counters"]["role_flip"] == 0 and cf.audit(st)["role_flip"] == 0
    assert flow.case_counters(st)["role_flip"] == 0


# --- (c) items 2 and 3 with the frame ----------------------------------------------------------------------------------

def test_c_the_frames_institution_feeds_the_governing_law(monkeypatch):
    from src.rag import corpus_laws
    monkeypatch.setitem(corpus_laws._NAMES, "names", ["Codice Civile 2026", "Codice Penale 2026"])
    code = flow.read_institutions({"frame_institution": {"testo": "legittima difesa", "frase": "x"}})
    assert code[0]["frame"] and code[0]["name"] == "legittima difesa"
    ctx = {"records": {}, "institutions": code}
    assert M.governing_law(code[0], ctx) == ("c.p.", "tabella") and M.corpus_line(ctx) == ""
    travel = flow.read_institutions({"frame_institution": {"testo": "contratto di viaggio", "frase": "x"}})
    ctx = {"records": {}, "institutions": travel}
    assert M.corpus_line(ctx).startswith("Il corpus non contiene ancora informazioni sufficienti su contratto di viaggio")


SALE = ("Elio Corvo ha acquistato da Nina Gallo un villino con rogito del 4 aprile 2025. Dopo il rogito ha scoperto "
        "difformità edilizie rispetto al permesso di costruire depositato in Comune. Elio Corvo vuole agire contro "
        "Nina Gallo per la riduzione del prezzo.")


def test_c_a_frame_verified_amministrativo_rito_is_no_administrative_act(monkeypatch):
    reply = {"parti": [{"nome": "Elio Corvo", "lato": "cliente"}, {"nome": "Nina Gallo", "lato": "controparte"}],
             "cliente": {"nome": "Elio Corvo", "perche": "«Elio Corvo vuole agire contro Nina Gallo»"},
             "ruolo": {"valore": "attore", "perche": "«Elio Corvo vuole agire contro Nina Gallo»"},
             "rito": {"valore": "amministrativo", "perche": "«permesso di costruire depositato in Comune»"},
             "quesiti": []}
    out = _run(monkeypatch, SALE, reply)
    assert out["posture"]["rito"] == "amministrativo"            # the frame's rito binds the posture ...
    assert family_gate.administrative_act(SALE) == []           # ... but the text states no act against the client
    ctx = family_gate.context(SALE)
    assert ctx["admin_act"] is False and "amministrativo" not in ctx["modules"]
    assert not family_gate.decide("c.p.a.", "", ctx, number="29")["ok"]
    assert "amministrativo" not in [m["id"] for m in M.active_modules({"case": SALE, "institutions": []})]
    # with an act in the text: admitted
    act = SALE + " Il Comune ha poi notificato a Elio Corvo un'ordinanza di demolizione delle opere difformi."
    assert family_gate.context(act)["admin_act"] and family_gate.decide("c.p.a.", "", family_gate.context(act),
                                                                        number="29")["ok"]
