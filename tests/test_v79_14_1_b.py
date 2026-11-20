"""v79.14.1 -- the surviving 79.12.8 items 6 and 8 (Build B), on synthetic texts, records and rulings only (no model,
no Neo4j, no network: every call is a fake):

6. objections with every element of the main record «da provare»: ONE model-written objection, on the strongest element
   (lawyer_method.strongest_open: the only element the main line turns on, else the one the most narrated facts touch,
   else the distinctive one, else the first), grounded as every rule statement (rule_statements.check removes nothing of
   it) -- or the line saying none is grounded; never the per-element template («La controparte negherà l'elemento …
   Risposta. Va provato: …») repeated, in any answer. Counters objection_template_repeated (must be 0),
   objection_grounded, objection_none. Written for the client's role (lawyer_method.objection_party: the imputato's
   defence objects to a victim client) and, with CASE_FRAME on, for the verified frame's client and role (a different
   posture role or single client logged frame_conflict «objections_plan», losing).
8. the «requisiti provati N su M» count never in prose under the method -- the step-aside path's status sentence prints
   the verdict alone and a last pass removes any count left (count_sentences_removed, count_sentences_left: must be 0);
   §5 relevance: a ruling whose quoted passage never names the record's institution is dropped
   (data/ruling_relevance.json; rulings_passage_irrelevant_dropped).
"""

import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

NO_REVIEW = lambda messages, **kw: []          # noqa: E731 -- the reply-binding review: no flags, no model


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for name, value in (("LLM_API_KEY", "k"), ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "k"), ("NEO4J_URI", "bolt://localhost:7687"), ("NEO4J_USER", "n"),
                        ("NEO4J_PASSWORD", "x")):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("THESIS_DETERMINISTIC", "")


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _req(i, text, status="da provare", **extra):
    return {"id": f"R{i}", "comma": 1, "kind": "requisito", "text": text, "status": status, **extra}


# --- 6. objections with every element «da provare» -------------------------------------------------------------------------

CIVIL_CASE = "Il muro del fondo di Caio è crollato sull'auto di Tizio parcheggiata nel cortile. Assisto Tizio."
CIVIL_SENTS = [{"id": "S1", "text": "Il muro del fondo di Caio è crollato sull'auto di Tizio.", "source": "caso",
                "certainty": "accertato"}]
PENAL_CASE = "Durante una lite Mevio spinge Sempronio, che cade. Il PM contesta a Mevio l'art. 582 c.p. Assisto Mevio."
PENAL_SENTS = [{"id": "S1", "text": "Mevio spinge Sempronio, che cade.", "source": "caso", "certainty": "accertato"}]


def _civil_records(*statuses):
    texts = ["il fatto doloso o colposo del danneggiante", "il danno ingiusto subito dal danneggiato",
             "il nesso causale tra il fatto e il danno"]
    reqs = [_req(i, texts[i - 1], st) for i, st in enumerate(statuses, 1)]
    reqs[1]["supporting"] = ["S1"]                          # the facts touch the damage
    return {"2043 c.c.": {"rubrica": "Risarcimento per fatto illecito", "side": "weapon", "requisites": reqs}}


def _penal_records(*statuses):
    texts = ["una condotta di percosse o violenza sulla persona", "una malattia nel corpo o nella mente della vittima",
             "il dolo di ledere"]
    reqs = [_req(i, texts[i - 1], st, distinctive=(i == 2)) for i, st in enumerate(statuses, 1)]
    return {"582 c.p.": {"rubrica": "Lesione personale", "side": "threat", "charged": 1, "requisites": reqs}}


def _ctx(records, family="civile"):
    from src.rag import lawyer_method as M, posture as P
    role, stage = ("attore", "pre_filing") if family == "civile" else ("imputato", "primo_grado")
    posture = {"rito": family, "family": family, "role": role, "stage": stage, "entry": P.entry_for(family, role, stage)}
    case, sents, client = (CIVIL_CASE, CIVIL_SENTS, "Tizio") if family == "civile" else (PENAL_CASE, PENAL_SENTS,
                                                                                           "Mevio")
    return M.prepare({"records": records, "sentences": sents}, posture, [], case, client, "", list(records), [], [], [],
                     [], link_call=lambda *a: {}, case_id_="t79131b")


def _plan(ctx):
    from src.rag import lawyer_method as M
    return M.clean_method({}, M.thesis_candidates(ctx))


def _objections(answer):
    from src.rag import lawyer_method as M
    return dict(M.answer_sections(answer))[M.title("obiezioni")]


def test_the_strongest_open_element_by_the_main_line_the_facts_the_distinctive_the_order():
    from src.rag import lawyer_method as M
    ctx = _ctx(_civil_records("da provare", "da provare", "da provare"))
    assert M.all_open(ctx)[0] == "2043 c.c."
    one = [{"tesi": "x", "contesta": ["R3"], "poggia_su": []}]
    assert M.strongest_open(ctx, one, 0) == {"id": "R3", "text": "il nesso causale tra il fatto e il danno",
                                             "article": "2043 c.c.", "why": "main_line"}
    rests = [{"tesi": "x", "contesta": [], "poggia_su": ["R1", "R2", "R3"]}]
    assert M.strongest_open(ctx, rests, 0)["id"] == "R2" and M.strongest_open(ctx, rests, 0)["why"] == "facts"
    assert M.strongest_open(ctx)["id"] == "R2"                    # no line: every live element, the facts decide
    penal = _ctx(_penal_records("da provare", "da provare", "da provare"), "penale")
    assert M.strongest_open(penal) == {"id": "R2", "text": "una malattia nel corpo o nella mente della vittima",
                                       "article": "582 c.p.", "why": "distinctive"}
    for r in penal["records"]["582 c.p."]["requisites"]:
        r.pop("distinctive", None)
    assert M.strongest_open(penal)["id"] == "R1" and M.strongest_open(penal)["why"] == "first"


def test_not_all_open_with_one_element_proven_or_excluded():
    from src.rag import lawyer_method as M
    assert M.all_open(_ctx(_civil_records("provato", "da provare", "da provare"))) == ("", [])
    assert M.strongest_open(_ctx(_civil_records("da provare", "da provare", "non risulta"))) == {}


def test_the_objections_call_asks_one_on_the_strongest_and_keeps_the_one_on_it():
    from src.rag import lawyer_method as M
    ctx = _ctx(_civil_records("da provare", "da provare", "da provare"))
    got = {}

    def call(messages):
        got["system"], got["human"] = messages[0].content, messages[1].content
        return {"obiezioni": [
            {"obiezione": "La controparte dirà che il muro era stabile", "risposta": "Lo dirà la perizia."},
            {"obiezione": "La controparte negherà il danno ingiusto: l'auto era già rovinata",
             "risposta": "Il danno ingiusto subito dal danneggiato risulta dalle fotografie del crollo."},
            {"obiezione": "La controparte contesterà il danno ingiusto anche per il fermo", "risposta": "No."}]}
    plan = M.objections_plan(ctx, _plan(ctx), call, NO_REVIEW)
    assert M.OBJECTIONS_ONE.format(who="la controparte") in got["system"]
    assert M.OBJECTIONS_THREE.format(who="la controparte") not in got["system"]
    assert M.STRONGEST_LINE.format(id="R2", el="il danno ingiusto subito dal danneggiato", art="2043 c.c.",
                                   why=M.STRONGEST_WHY["facts"]) in got["human"]
    assert [o["obiezione"] for o in plan["obiezioni"]] == ["La controparte negherà il danno ingiusto: l'auto era già "
                                                           "rovinata"]
    assert ctx["objection_all_open"]["written"] and ctx["objection_all_open"]["offered"] == 3


def test_with_every_element_open_compose_prints_one_grounded_objection_never_the_template():
    from src.rag import lawyer_method as M
    ctx = _ctx(_civil_records("da provare", "da provare", "da provare"))
    plan = M.objections_plan(ctx, _plan(ctx), lambda m: {"obiezioni": [
        {"obiezione": "La controparte negherà il danno ingiusto: l'auto era già rovinata",
         "risposta": "Il danno ingiusto subito dal danneggiato risulta dalle fotografie del crollo."}]}, NO_REVIEW)
    answer = M.compose([("Norme applicate ai fatti", "regola")], ctx, plan)
    section = _objections(answer)
    assert section.startswith(M.ALL_OPEN_ONE.format(who="della controparte",
                                                    el="il danno ingiusto subito dal danneggiato"))
    assert section.count("**Obiezione ") == 1 and not M.OPEN_TEMPLATE_RE.search(section)
    assert "Va provato:" not in section and ctx["objection_grounded"] == 1 and ctx["objection_none"] == 0
    checks = {c["check"]: c["ok"] for c in M.structure_checks(answer, {})}
    assert checks["objections_all_open"] and checks["objection_template_once"] and "three_objections" not in checks
    assert M.template_repeats(answer) == 0


def test_an_ungrounded_objection_with_every_element_open_gives_none():
    from src.rag import lawyer_method as M
    ctx = _ctx(_civil_records("da provare", "da provare", "da provare"))
    plan = M.objections_plan(ctx, _plan(ctx), lambda m: {"obiezioni": [
        {"obiezione": "La controparte negherà il danno ingiusto",
         "risposta": "La legge prevede la forma scritta ad substantiam per ogni donazione di beni mobili."}]},
        NO_REVIEW)
    assert len(plan["obiezioni"]) == 1                      # the call's: on the element, not yet checked
    answer = M.compose([("Norme applicate ai fatti", "regola")], ctx, plan)
    assert _objections(answer) == M.ALL_OPEN_NONE.format(who="della controparte")
    assert ctx["objection_grounded"] == 0 and ctx["objection_none"] == 1
    assert {c["check"]: c["ok"] for c in M.structure_checks(answer, {})}["objections_all_open"]


@pytest.mark.parametrize("family", ["civile", "penale"])
def test_a_failed_call_with_every_element_open_gives_none_never_the_template(family):
    from src.rag import lawyer_method as M
    records = _civil_records(*["da provare"] * 3) if family == "civile" else _penal_records(*["da provare"] * 3)
    ctx = _ctx(records, family)
    assert any(M.OPEN_TEMPLATE_RE.search(o["obiezione"]) for o in ctx["objections_fallback"])    # what it was

    def down(messages):
        raise RuntimeError("down")
    plan = _plan(ctx)
    sols, main, _ = M.selection(ctx, plan)
    assert not sols[main].get("placeholder")                # a line to object to, in both families
    plan = M.objections_plan(ctx, plan, down, NO_REVIEW)
    answer = M.compose([("Norme applicate ai fatti", "regola")], ctx, plan)
    who = M.ALL_OPEN_WHO[family]
    assert _objections(answer) == M.ALL_OPEN_NONE.format(who=who)
    assert not M.OPEN_TEMPLATE_RE.search(answer) and M.template_repeats(answer) == 0


def test_an_objection_on_another_element_is_not_the_one():
    from src.rag import lawyer_method as M
    ctx = _ctx(_civil_records("da provare", "da provare", "da provare"))
    plan = M.objections_plan(ctx, _plan(ctx), lambda m: {"obiezioni": [
        {"obiezione": "La controparte dirà che il muro era stabile", "risposta": "Lo dirà la perizia."}]}, NO_REVIEW)
    assert plan["obiezioni"] == [] and ctx["objection_all_open"]["written"] is False
    assert _objections(M.compose([("Norme applicate ai fatti", "x")], ctx, plan)).startswith(M.ALL_OPEN_PREFIX)


def test_with_some_element_proven_each_open_element_keeps_its_objection_and_three_stay():
    # review: item 6 is the all-open case only -- a mixed record keeps 79.14.0's fallback, one objection per open
    # element (each on its own element: no repeat), so a model offering fewer than three still gets three
    from src.rag import lawyer_method as M
    ctx = _ctx(_civil_records("provato", "da provare", "da provare"))
    opens = [o for o in ctx["objections_fallback"] if M.OPEN_TEMPLATE_RE.search(o["obiezione"])]
    assert len(opens) == 2 and len({M.OPEN_TEMPLATE_RE.search(o["obiezione"]).group(1) for o in opens}) == 2
    plan = M.objections_plan(ctx, _plan(ctx), lambda m: {"obiezioni": []}, NO_REVIEW)
    answer = M.compose([("Norme applicate ai fatti", "regola")], ctx, plan)
    section = _objections(answer)
    assert not section.startswith(M.ALL_OPEN_PREFIX) and len(M.OPEN_TEMPLATE_RE.findall(section)) == 2
    assert M.template_repeats(answer) == 0
    checks = {c["check"]: c["ok"] for c in M.structure_checks(answer, {})}
    assert checks["three_objections"] and checks["objection_template_once"]
    assert ctx["objection_grounded"] == ctx["objection_none"] == 0


def _compose_min():
    from src.rag import lawyer_method as M
    ctx = _ctx(_civil_records("provato", "da provare", "da provare"))
    return M.compose([("Norme applicate ai fatti", "regola")], ctx,
                     M.objections_plan(ctx, _plan(ctx), lambda m: {"obiezioni": []}, NO_REVIEW))


def test_template_repeats_reads_each_section_and_the_all_open_layout():
    from src.rag import lawyer_method as M
    answer = _compose_min()
    sec = _objections(answer)
    one, two = (f"**Obiezione {i}.** {M.OBJ_DENIED[0].format(el=e)}.\n**Risposta.** Va provato: {e}."
                for i, e in ((1, "alfa"), (2, "beta")))
    again = one.replace("Obiezione 1", "Obiezione 3")
    assert M.template_repeats(answer.replace(sec, one)) == 0                    # once: allowed
    assert M.template_repeats(answer.replace(sec, f"{one}\n\n{two}")) == 0      # two elements, each its own: allowed
    assert M.template_repeats(answer.replace(sec, f"{one}\n\n{two}\n\n{again}")) == 1     # the same element again
    assert M.template_repeats(answer.replace(sec, f"{M.ALL_OPEN_PREFIX}: x.\n\n{one}")) == 1     # all open: never


def test_objections_counted_on_the_delivered_answer():
    from src.rag import interactive_flow as flow, lawyer_method as M
    answer = _compose_min()
    state = {"block_counters": {}}
    flow.objections_counted(state, answer, [{"objection_grounded": 1}, {"objection_none": 1}])
    assert state["block_counters"] == {"objection_grounded": 1, "objection_none": 1, "objection_template_repeated": 0}
    sec = _objections(answer)
    doubled = answer.replace(sec, sec + "\n\n**Obiezione 9.** " + M.OBJ_DENIED[0].format(el="zeta") +
                             ".\n**Risposta.** Va provato: zeta." + "\n\n**Obiezione 10.** " +
                             M.OBJ_DENIED[0].format(el="zeta") + ".\n**Risposta.** Va provato: zeta.")
    flow.objections_counted(state, doubled, [])
    assert state["block_counters"]["objection_template_repeated"] >= 1


# --- 6, with the case frame (CASE_FRAME): the objection written for the frame's client and role -------------------------

VICTIM_CASE = ("Durante una lite Mevio spinge Sempronio, che cade e si frattura il polso. Sempronio vuole denunciare "
               "Mevio. Assisto Sempronio.")


def _frame_state(role="persona_offesa", client="Sempronio", **extra):
    frame = {"parti": [{"nome": "Sempronio", "lato": "cliente"}, {"nome": "Mevio", "lato": "controparte"}],
             "cliente": {"nome": client} if client else None,
             "ruolo": {"valore": role, "intake": role} if role else None}
    return {"records": _penal_records("da provare", "da provare", "da provare"), "sentences": PENAL_SENTS,
            "case_frame": {"frame": {k: v for k, v in frame.items() if v}}, **extra}


def _framed_ctx(state, client="Sempronio", role="imputato"):
    from src.rag import lawyer_method as M, posture as P
    posture = {"rito": "penale", "family": "penale", "role": role, "stage": "indagini",
               "entry": P.entry_for("penale", role, "indagini")}
    return M.prepare(state, posture, [], VICTIM_CASE, client, "", list(state["records"]), [], [], [], [],
                     link_call=lambda *a: {}, case_id_="t79141bf")


def _asked(ctx):
    from src.rag import lawyer_method as M
    got = {}
    sols, main, _ = M.selection(ctx, _plan(ctx))
    got["strong"] = el = M.strongest_open(ctx, sols, main)["text"]

    def call(messages):
        got["system"], got["human"] = messages[0].content, messages[1].content
        return {"obiezioni": [{"obiezione": f"Si dirà che l'elemento «{el}» non è provato",
                               "risposta": f"L'elemento «{el}» risulta dai fatti narrati."}]}
    plan = M.objections_plan(ctx, _plan(ctx), call, NO_REVIEW)
    return got, plan


def test_with_the_frame_the_objection_is_written_for_the_frames_client_and_role(monkeypatch):
    from src.rag import lawyer_method as M
    monkeypatch.setenv("CASE_FRAME", "true")
    state = _frame_state()
    ctx = _framed_ctx(state)                 # the posture says imputato: the frame's persona_offesa wins
    party = ctx["objection_party"]
    assert party == {"client": "Sempronio", "role": "persona_offesa", "frame": True, "side": "vittima"}
    assert {"field": "ruolo", "step": "objections_plan", "proposed": "imputato", "frame": "persona_offesa"} in \
        state["frame_conflicts"]
    got, plan = _asked(ctx)
    assert M.OBJECTIONS_ONE.format(who="la difesa dell'imputato") in got["system"]
    assert "l'accusa opporrà" not in got["system"]
    assert "POSIZIONE: Sempronio, rito penale, role persona_offesa" in got["human"]
    section = _objections(M.compose([("Norme applicate ai fatti", "regola")], ctx, plan))
    assert section.startswith(M.ALL_OPEN_ONE.format(who="della difesa dell'imputato", el=got["strong"]))
    assert ctx["objection_grounded"] == 1 and not M.OPEN_TEMPLATE_RE.search(section)


def test_with_the_frame_a_client_other_than_the_frames_loses_with_one_client(monkeypatch):
    monkeypatch.setenv("CASE_FRAME", "true")
    state = _frame_state()
    ctx = _framed_ctx(state, client="Mevio", role="persona_offesa")
    assert ctx["objection_party"]["client"] == "Sempronio"
    assert any(c["field"] == "cliente" and c["step"] == "objections_plan" and c["proposed"] == "Mevio"
               for c in state["frame_conflicts"])
    got, _ = _asked(ctx)
    assert "POSIZIONE: Sempronio," in got["human"]


def test_with_the_frame_another_client_of_a_multi_client_answer_keeps_its_own_role(monkeypatch):
    monkeypatch.setenv("CASE_FRAME", "true")
    state = _frame_state(method_roles=[{"client": "Caio", "role": "imputato", "intake_role": "imputato"}])
    ctx = _framed_ctx(state, client="Caio", role="imputato")
    assert ctx["objection_party"] == {"client": "Caio", "role": "imputato", "frame": False, "side": "penale"}
    assert not state.get("frame_conflicts")


def test_with_the_frame_a_verified_unknown_role_objects_from_neither_side(monkeypatch):
    from src.rag import lawyer_method as M
    monkeypatch.setenv("CASE_FRAME", "true")
    ctx = _framed_ctx(_frame_state(role="sconosciuto"))
    assert ctx["objection_party"]["side"] == "ignoto"
    got, plan = _asked(ctx)
    assert M.OBJECTIONS_ONE.format(who="l'altra parte") in got["system"]
    assert "role sconosciuto" not in got["human"] and "role imputato" not in got["human"]
    assert _objections(M.compose([("Norme applicate ai fatti", "x")], ctx, plan)).startswith(
        M.ALL_OPEN_PREFIX + ": l'obiezione più forte dell'altra parte")


def test_with_the_frame_off_the_posture_decides_and_nothing_is_logged(monkeypatch):
    monkeypatch.delenv("CASE_FRAME", raising=False)
    state = _frame_state()
    ctx = _framed_ctx(state)
    assert ctx["objection_party"] == {"client": "Sempronio", "role": "imputato", "frame": False, "side": "penale"}
    assert not state.get("frame_conflicts")
    got, _ = _asked(ctx)
    assert "POSIZIONE: Sempronio, rito penale, role imputato" in got["human"]
    victim = _framed_ctx(_frame_state(), role="persona_offesa")      # a victim client: the defence objects, flag off too
    assert victim["objection_party"]["side"] == "vittima"


# --- 8a. the count sentence never in prose -----------------------------------------------------------------------------------

def test_count_out_cuts_a_parenthetical_and_removes_a_sentence():
    from src.rag import lawyer_method as M
    found = []
    text = ("**Art. 2043 c.c.**\nEsito sui fatti del caso: da provare (requisiti provati: 0 su 3).\n\nIl danno è da "
            "provare. Per ora i requisiti provati 0 su 5 non bastano. Resta la perizia.")
    out = M.count_out(text, found)
    assert "Esito sui fatti del caso: da provare." in out and "Il danno è da provare. Resta la perizia." in out
    assert not M.count_left(out) and len(found) == 2
    assert M.count_out("Nessun conteggio qui.", []) == "Nessun conteggio qui."


def test_the_status_sentence_under_the_method_is_the_verdict_alone_on_the_step_aside(monkeypatch):
    from src.rag import interaction_copy as C, interactive_flow as flow
    rec = {"rubrica": "Furto", "verdict": "da provare", "requisites": [_req(1, "impossessamento", "da provare")]}
    state = {"mode": "single_shot", "records": {"624 c.p.": rec}, "method_refused": "x"}
    assert "requisiti provati: 0 su 1" in flow.status_sentence("624 c.p.", rec, state)      # the legacy prose (v62)
    monkeypatch.setenv("LAWYER_METHOD", "true")
    assert flow.status_sentence("624 c.p.", rec, state) == C.fill(C.ARTICLE_STATUS_VERDICT, verdict="da provare")
    assert state["count_sentences_source"] == 1
    state.pop("method_refused")
    assert flow.status_sentence("624 c.p.", rec, state) is None                              # the method's own


def test_the_step_aside_answer_prints_no_count_and_counts_what_it_removed(monkeypatch):
    import test_v60g
    from test_v60f import _graded_state
    from src.rag import interactive_flow as flow, lawyer_method as M
    state = _graded_state(monkeypatch)
    state["mode"] = "single_shot"
    monkeypatch.setenv("LAWYER_METHOD", "true")
    monkeypatch.setattr(M, "objections_plan", lambda ctx, plan: plan)
    monkeypatch.setattr(M, "method_plan", lambda ctx, call=None: M.clean_method({}, []))     # the step-aside
    writer = {flow.ANSWER_TITLES[1]: "La servitù va provata: requisiti provati 0 su 5 sui fatti narrati. "
                                     "Il passaggio era esercitato da anni."}
    text, _ = test_v60g._answer(monkeypatch, state, writer, None, {})
    assert state.get("method_refused") and M.not_applied_note(state["method_refused"]) in text
    assert not M.count_left(text) and "requisiti provati" not in text
    c = flow.case_counters(state)
    assert c["count_sentences_left"] == 0 and c["count_sentences_removed"] >= 1


def test_with_the_method_off_the_legacy_status_sentence_stays():
    from src.rag import interactive_flow as flow
    state = {"block_counters": {}}
    text = "Esito sui fatti del caso: da provare (requisiti provati: 0 su 3)."
    assert flow.count_sentences_out(state, text) == text
    assert state["block_counters"] == {"count_sentences_removed": 0, "count_sentences_left": 0}


# --- 8b. §5 relevance ----------------------------------------------------------------------------------------------------------

CARD = ("L'indebito utilizzo della carta di credito altrui integra il reato quando l'agente ne abbia il possesso per "
        "consegna del titolare.")
NAMED = ("L'appropriazione indebita richiede che l'agente abbia il possesso del denaro altrui e se ne appropri con "
         "un atto di interversione.")
CAUTELARE = ("La misura cautelare richiede gravi indizi della partecipazione al fatto e il pericolo di reiterazione, "
             "valutati in concreto dal giudice.")
RISSA = "La partecipazione alla rissa integra il reato anche senza lesioni, purché vi sia il pericolo per l'incolumità."


@pytest.mark.parametrize("passage, label, rubrica, named", [
    (CARD, "646 c.p.", "Appropriazione indebita", False),         # «indebito» alone: not the institution
    (NAMED, "646 c.p.", "Appropriazione indebita", True),         # its name
    ("Ai sensi dell'art. 646 c.p. il possesso deve preesistere.", "646 c.p.", "Appropriazione indebita", True),
    ("L'interversione del possesso si ha quando l'agente si comporta da proprietario.", "646 c.p.",
     "Appropriazione indebita", True),                            # by synonym (data aliases), never the name
    ("Il bancomat altrui usato oltre il mandato ricevuto integra il delitto di indebito utilizzo.", "646 c.p.",
     "Appropriazione indebita", False),                           # a credit-card misuse passage, without 646
    (CAUTELARE, "588 c.p.", "Rissa", False),
    ("La custodia cautelare in carcere non è adeguata quando basta l'obbligo di firma.", "588 c.p.", "Rissa", False),
    ("Per l'art. 588 c.p. occorrono almeno tre persone.", "588 c.p.", "Rissa", True),       # by article number
    (RISSA, "588 c.p.", "Rissa", True),
    ("La malattia nel corpo sussiste quando vi sia un'alterazione funzionale.", "582 c.p.", "Lesione personale",
     True),                                                       # its defining element's noun (the record's own words)
    ("Lo svolgimento di attività pericolose impone ogni misura idonea.", "2050 c.c.",
     "Responsabilità per l'esercizio di attività pericolose", True),          # two of its nouns together
    ("Qualunque passaggio.", "999 c.p.", "", None),                # nothing to test it on
])
def test_a_passage_names_the_records_institution(passage, label, rubrica, named):
    from src.rag import rulings
    assert rulings.passage_names_institution(passage, label, rubrica) is named


def test_the_relevance_table_is_data():
    data = json.loads((ROOT / "data" / "ruling_relevance.json").read_text(encoding="utf-8"))
    assert data["rubrica_stems_min"] == 2 and "646 c.p." in data["aliases"] and "_about" in data
    from src.rag import rulings
    assert rulings.relevance_table()["aliases"]["588 c.p."] == ["riss"]


def _ruling(i, abstract):
    return {"ruling": f"X{i}", "file_name": f"snpen@s20@a2025@n{i:05d}@tS.clean.pdf", "name": "",
            "document_number": "", "date_enacted": "2025-03-01", "linked": True, "section": f"S{i}",
            "abstract": abstract}


class _Session:
    def __init__(self, rows, texts):
        self.rows, self.text_of = rows, texts

    def run(self, query, **kw):
        from src.rag import rulings
        rows = self.rows if query == rulings.FETCH_QUERY else [
            {"section": i, "plain_text": self.text_of.get(i, "")} for i in kw["ids"]]

        class R:
            def data(self_inner):
                return rows
        return R()


@pytest.mark.parametrize("label, rubrica, element, off, on", [
    ("646 c.p.", "Appropriazione indebita", "appropriarsi del denaro altrui di cui abbia il possesso", CARD, NAMED),
    ("588 c.p.", "Rissa", "la partecipazione alla rissa con pericolo per l'incolumità", CAUTELARE, RISSA),
])
def test_section_5_drops_a_ruling_whose_passage_never_names_the_institution(label, rubrica, element, off, on):
    from src.rag import rulings
    number = label.split(" ")[0]
    rec = {"side": "threat", "rubrica": rubrica, "requisites": [_req(1, element)]}
    ctx = {"records": {label: rec}, "candidates": [{"id": "T1", "contesta": ["R1"]}], "case": "Un caso sintetico."}
    abstract = f"{rubrica}: sui presupposti"
    session = _Session([_ruling(1, abstract), _ruling(2, abstract)], {"S1": off, "S2": on})

    def call(messages, tokens):
        return {"P1": {"frase": "P1.1", "elemento": "R1", "pertinenza": 2, "verso": "a favore"},
                "P2": {"frase": "P2.1", "elemento": "R1", "pertinenza": 2, "verso": "a favore"}}
    record = rulings.build(ctx, [{"_article": (number, "c.p."), "d": {"id": "CP"}, "s": {"name": number}}], session,
                           call=call)
    kept = {r["id"]: r for r in record["kept"]}
    assert "X1" not in kept and kept["X2"]["passage"] == on
    dropped = record["log"]["passage_irrelevant_dropped"]
    assert [d["article"] for d in dropped] == [label] and dropped[0]["passage"].startswith(off[:40])
    assert record["log"]["articles"][label]["filters"]["off_institution"] == 1
    section = rulings.section_text(record)
    assert on[:40] in section and off[:40] not in section


def test_the_counter_reads_the_rulings_record(monkeypatch):
    from src.rag import interactive_flow as flow
    monkeypatch.setenv("RULINGS_RECORD", "true")
    state = {"block_counters": {}, "rulings_record": {"log": {"passage_irrelevant_dropped": [{"article": "646 c.p."}]}}}
    flow.text_counted(state, "Nessuna pronuncia.")
    assert state["block_counters"]["rulings_passage_irrelevant_dropped"] == 1
    monkeypatch.delenv("RULINGS_RECORD")
    flow.text_counted(state, "Nessuna pronuncia.")
    assert state["block_counters"]["rulings_passage_irrelevant_dropped"] == 0


# --- the counters, the nightly, the docs, the copy --------------------------------------------------------------------------

def test_the_nightly_fails_a_cell_with_a_repeated_template_or_a_count_left():
    nightly = _script("nightly_sim")
    for key in ("objection_template_repeated", "count_sentences_left"):
        got = nightly.zero_counters_check({key: 1})
        assert not got["ok"] and key in got["why"]
    assert nightly.zero_counters_check({"objection_grounded": 3, "objection_none": 1,
                                        "count_sentences_removed": 4,
                                        "rulings_passage_irrelevant_dropped": 2})["ok"]


def test_a_whole_answer_carries_the_new_counters(monkeypatch):
    import test_v60g
    from test_v60f import _graded_state
    from src.rag import interactive_flow as flow, lawyer_method as M
    state = _graded_state(monkeypatch)
    state["mode"] = "single_shot"
    monkeypatch.setenv("LAWYER_METHOD", "true")
    line = M.NO_ACTION_LINE.format(istituto="servitù")

    def plan(ctx, call=None):
        ctx["candidates"], ctx["no_action"], ctx["no_action_kind"] = [], line, "none"
        return M.clean_method({}, [])
    monkeypatch.setattr(M, "method_plan", plan)
    test_v60g._answer(monkeypatch, state, {}, None, {})
    counters = flow.case_counters(state)
    for key in ("objection_template_repeated", "objection_grounded", "objection_none", "count_sentences_removed",
                "count_sentences_left", "rulings_passage_irrelevant_dropped"):
        assert key in counters, key
    assert counters["objection_template_repeated"] == 0 and counters["count_sentences_left"] == 0


def test_the_docs_and_the_copy_name_build_b():
    rules = (ROOT / "docs" / "RULES.md").read_text(encoding="utf-8")
    for word in ("objection_template_repeated", "count_sentences_left", "rulings_passage_irrelevant_dropped",
                 "ruling_relevance.json"):
        assert word in rules, word
    copy = (ROOT / "docs" / "copy_review.md").read_text(encoding="utf-8")
    assert "objection_party" in rules and "«objections_plan»" in rules
    for key in ("ALL_OPEN_ONE", "ALL_OPEN_NONE", "ARTICLE_STATUS_VERDICT", "OBJECTOR", "della difesa dell'imputato"):
        assert key in copy, key
    nightly = (ROOT / "scripts" / "nightly_sim.py").read_text(encoding="utf-8")
    assert "objection_template_repeated" in nightly and "rulings_passage_irrelevant_dropped" in nightly


def test_the_all_open_lines_open_with_the_prefix_the_checks_read():
    from src.rag import lawyer_method as M
    assert M.ALL_OPEN_ONE.startswith(M.ALL_OPEN_PREFIX) and M.ALL_OPEN_NONE.startswith(M.ALL_OPEN_PREFIX)
    assert M.OBJECTIONS_THREE.format(who="x") in M.OBJECTIONS_SYSTEM.format(who="x")      # the sentence replaced
