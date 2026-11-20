"""v79.14.1 -- the surviving 79.12.8 items 1-3 (Build A), on synthetic texts, states and records only (no model, no
Neo4j, no network):

1. role flip: the method's role is the intake's resolved role unless the request itself states otherwise (logged: which
   wins and why), and the counterparty is never taken as the client -- (a) v79.10.2's resisting reading lived in the
   posture only (intake.resolve said «attore», the method «convenuto»): now in resolve; (b) with several clients each
   client's role came from the first read's word: now the intake's, or the request's own statement; (c) the party
   assumed «a carico» while the intake reads the client as the claimant; (d) the intake's rule-read client lost to the
   extraction's; (e) the party_threat path -- the demanding party that threatens to sue settled as the client. Counter
   role_flip (must be 0), checked by the nightly.
2. the honesty line «Il corpus non contiene ancora informazioni sufficienti su <materia>» only when the institution's
   governing law is outside the codes and missing from the archive (data/institution_laws.json); never for a code
   institution (legittima difesa: art. 52 c.p.); a no-claim answer keeps its ten sections with NO_CLAIM_SUMMARY in §1.
   Counters corpus_line, corpus_line_code (must be 0).
3. the amministrativo module and family, the c.p.a. articles and the administrative regime only with an administrative
   act against the client (data/administrative_act.json) -- never «Comune», «difformità edilizie», «permesso di
   costruire» alone in a civil sale. Counters amministrativo_refused_no_act, amministrativo_without_act (must be 0).
"""

import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for name, value in (("LLM_API_KEY", "k"), ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "k"), ("NEO4J_URI", "bolt://localhost:7687"), ("NEO4J_USER", "n"),
                        ("NEO4J_PASSWORD", "x")):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("THESIS_DETERMINISTIC", "")


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _req(i, text, status="provato"):
    return {"id": f"R{i}", "comma": 1, "kind": "requisito", "text": text, "status": status}


def _rec(rubrica, *reqs, side="weapon"):
    return {"rubrica": rubrica, "side": side, "requisites": list(reqs)}


def _posture(family, role, stage="pre_filing"):
    from src.rag import posture
    return {"rito": family, "family": family, "role": role, "stage": stage, "intake_role": role,
            "entry": posture.entry_for(family, role, stage)}


def _state(case, parties, intake=None, answered=None):
    from src.rag import interactive_flow as flow
    analysis = {flow.CASE_KEY: case, "fact_signals": {"facts": {"parties": parties}}}
    return {"text": case, "request": "", "analysis": analysis, "intake": intake or {}, "answered": answered or {},
            "questions": [], "block_counters": {}}


def _intake(role, how="rule", name="", rito="civile"):
    return {"rito": rito, "role": role, "stage": "pre_filing", "confidence": {"role": how}, "why": {"role": ""},
            "client": {"name": name, "side": ""}}


# --- 1. role flip -----------------------------------------------------------------------------------------------------

RESISTING = ("Beta S.r.l. chiede a Tizio il pagamento della penale di 8.000 euro per il ritardo nella posa del parquet. "
             "Tizio vuole sapere come contestare la penale.")


def test_a_resolve_and_the_posture_read_the_same_role_on_a_resisting_case():
    # cause (a): v79.10.2 (B1) turned a model's «attore» into «convenuto» in posture.of only -- the intake said one
    # role, the method used another
    from src.rag import intake, interactive_flow as flow, posture
    model = {**_intake("attore", "model", "Tizio")}
    analysis = {flow.CASE_KEY: RESISTING}
    resolved = intake.resolve(model, analysis)
    assert resolved["role"] == "convenuto" and resolved["resisting"] and "resiste" in resolved["role_why"]
    got = posture.of({"intake": model}, analysis)
    assert got["role"] == resolved["role"] == got["intake_role"] == "convenuto"
    assert flow.role_counted({}, got)["role_flip"] == 0
    # a rule's role (the user's word) is never changed, and the posture follows it
    stated = {**model, "confidence": {"role": "rule"}}
    assert intake.resolve(stated, analysis)["role"] == posture.of({"intake": stated}, analysis)["role"] == "attore"


def test_a_posture_with_no_case_in_the_analysis_reads_the_states_text():
    from src.rag import posture
    got = posture.of({"intake": _intake("attore", "model", "Tizio"), "text": RESISTING}, {})
    assert got["role"] == got["intake_role"] == "convenuto"


def test_b_several_clients_take_the_intakes_role_not_the_first_reads_word(monkeypatch):
    # cause (b): method_inputs_clients set each client's role from the first read's word («persona offesa» on a
    # client the intake reads as imputato)
    from src.rag import interactive_flow as flow
    case = "Tizio e Caio sono indagati per lesioni ai danni di Sempronio. Difendo entrambi."
    state = {"text": case, "request": ""}
    posture = _posture("penale", "imputato", "indagini")
    role, won = flow.role_for_client(state, posture, {"name": "Caio", "role": "persona offesa"}, case)
    assert (role, won) == ("imputato", flow.ROLE_FROM_INTAKE)
    # the request itself stating this client's role wins (logged), and is counted apart
    case2 = "Tizio e Caio sono persone offese. Caio si è costituito parte civile nel processo. Assisto entrambi."
    posture2 = _posture("penale", "persona_offesa", "primo_grado")
    role, won = flow.role_for_client({"text": case2}, posture2, {"name": "Caio", "role": "persona offesa"}, case2)
    assert (role, won) == ("parte_civile", flow.ROLE_FROM_REQUEST)
    counted = flow.role_counted({"method_roles": [{"client": "Caio", "role": "parte_civile", "intake_role":
                                                   "persona_offesa", "from": flow.ROLE_FROM_REQUEST}]}, posture2)
    assert counted == {"role_flip": 0, "role_request_wins": 1, "role_client_moved": 0}
    # the counter sees a method role the intake did not decide
    bad = {"method_roles": [{"client": "Caio", "role": "persona_offesa", "intake_role": "imputato",
                             "from": flow.ROLE_FROM_INTAKE}]}
    assert flow.role_counted(bad, posture)["role_flip"] == 1


def test_b_the_multi_client_method_sets_the_intakes_role(monkeypatch):
    import inspect
    from src.rag import interactive_flow as flow
    src = inspect.getsource(flow.method_inputs_clients)
    assert "method_role(state, posture, c, case)" in src and "_role_of_words(c.get" not in src
    assert "role_for_client(state, posture, client, case)" in inspect.getsource(flow.method_role)


CLAIM = ("Tizio ha prestato 6.000 euro a Caio, che non li ha restituiti alla scadenza pattuita. Vogliamo agire per "
         "recuperare la somma.")


def test_c_the_assumed_party_is_on_the_intakes_side_never_a_carico():
    # cause (c): with the client asked and unanswered the answer assumed the party «a carico» (convenuto) -- the
    # counterparty when the intake reads the client as the claimant
    from src.rag import interactive_flow as flow
    parties = [{"name": "Caio", "role": "convenuto"}, {"name": "Tizio", "role": "creditore"}]
    state = _state(CLAIM, parties, _intake("attore"))
    pick = flow._assumed_party(state, flow._restored_analysis(state["analysis"]))
    assert pick["name"] == "Tizio"
    # no party on the intake's side: never the one on the other side
    state = _state(CLAIM, [{"name": "Caio", "role": "convenuto"}, {"name": "Tizio", "role": ""}], _intake("attore"))
    assert flow._assumed_party(state, flow._restored_analysis(state["analysis"]))["name"] == "Tizio"
    state = _state(CLAIM, [{"name": "Caio", "role": "convenuto"}], _intake("attore"))
    assert flow._assumed_party(state, flow._restored_analysis(state["analysis"])) is None
    # no intake side: as before (the party a carico)
    state = _state(CLAIM, parties, {})
    assert flow._assumed_party(state, flow._restored_analysis(state["analysis"]))["name"] == "Caio"


def test_c_a_counterparty_settled_as_client_is_replaced_by_the_intakes_side(monkeypatch):
    from src.rag import interactive_flow as flow
    logged = []
    monkeypatch.setattr(flow, "_log", lambda event, data=None, *a, **k: logged.append((event, data)))
    parties = [{"name": "Caio", "role": "convenuto"}, {"name": "Tizio", "role": "creditore"}]
    state = _state(CLAIM, parties, _intake("attore"))
    analysis = flow._restored_analysis(state["analysis"])
    moved = flow.client_on_intake_side(state, analysis, "Caio", "Il parere è reso nell'interesse di Caio.")
    assert moved == ("Tizio", "Il parere è reso nell'interesse di Tizio.")
    assert state["client"]["name"] == "Tizio" and state["analysis"]["parte_assistita"] == "Tizio"
    assert state["role_check"]["moved"] and state["role_check"]["from"] == "Caio"
    decision = next(d for e, d in logged if e == "role_decision")
    assert decision["won"] == flow.ROLE_FROM_INTAKE and "Caio" in decision["why"]
    assert flow.role_counted(state, _posture("civile", "attore")) == {"role_flip": 0, "role_request_wins": 0,
                                                                       "role_client_moved": 1}


def test_c_the_users_own_client_wins_over_the_intakes_side():
    from src.rag import interactive_flow as flow
    parties = [{"name": "Caio", "role": "convenuto"}, {"name": "Tizio", "role": "creditore"}]
    state = _state(CLAIM, parties, _intake("attore"), answered={"parte_assistita": "Caio"})
    assert flow.client_on_intake_side(state, flow._restored_analysis(state["analysis"]), "Caio") is None
    assert state["role_check"]["stated_by_user"] and flow.role_counted(state, _posture("civile", "attore"))[
        "role_flip"] == 0


def test_c_the_counter_sees_a_counterparty_left_as_client():
    from src.rag import interactive_flow as flow
    state = {"role_check": {"client": "Caio", "client_side": "accused", "intake_side": "victim",
                            "stated_by_user": False}}
    assert flow.role_counted(state, _posture("civile", "attore"))["role_flip"] == 1


def test_d_the_intakes_rule_read_client_wins_over_the_extractions(monkeypatch):
    # cause (d): «il mio cliente, Rossi, …» -- the intake read Rossi by rule; the extraction named the counterparty
    from src.rag import intake, interactive_flow as flow
    case = ("Il mio cliente, Rossi, ha venduto a Bianchi un'auto usata per 9.000 euro; Bianchi non ha pagato il saldo "
            "di 3.000 euro. Come recuperare il saldo?")
    read = intake.detect(case, model=False)
    assert read["client"]["name"] == "Rossi" and flow.stated(case)["client"] == ""
    state = _state(case, [{"name": "Bianchi", "role": "cliente"}, {"name": "Rossi", "role": "venditore"}],
                   intake.resolve(read, {flow.CASE_KEY: case}))
    state["analysis"]["parte_assistita"] = "Bianchi"
    seen = []
    monkeypatch.setattr(flow, "apply_stated", lambda st, said, renumber=(): seen.append(said["client"]))
    assert flow.apply_intake_client(state, flow.stated(case)) == "Rossi" and seen == ["Rossi"]
    # the user's own words (stated) or an answered client: nothing to do; the same client: nothing either
    assert flow.apply_intake_client(state, {"client": "Bianchi"}) == ""
    state["analysis"]["fact_signals"]["facts"]["parties"] = [{"name": "Rossi", "role": "cliente"}]
    assert flow.apply_intake_client(state, flow.stated(case)) == ""


def test_d_single_shot_and_start_apply_the_intakes_client():
    import inspect
    from src.rag import interactive_flow as flow
    for fn in (flow.single_shot, flow.start):
        assert "apply_intake_client(state, said)" in inspect.getsource(fn)


THREAT = ("Alfa ha inviato a Beta una diffida per il pagamento di 7.200 euro di forniture e intende citare in giudizio "
          "se non paga entro dieci giorni.")


def test_e_the_threatening_party_is_never_the_client(monkeypatch):
    # cause (e): the party_threat path read the client as resisting (convenuto) while the extraction settled the
    # threatening party (Alfa) as the client -- the counterparty as client
    from src.rag import intake, interactive_flow as flow
    assert intake.threat_parties(THREAT) == ["Alfa"]
    role, why, _ = intake.derive_role(THREAT, "civile", "")
    assert role == "convenuto" and "minaccia" in why
    parties = [{"name": "Alfa", "role": "società fornitrice"}, {"name": "Beta", "role": "debitrice"}]
    state = _state(THREAT, parties, _intake("convenuto", "rule"))
    assert flow.party_side(parties[0], ["Alfa"]) == "victim" and flow.party_side(parties[1], ["Alfa"]) == "accused"
    moved = flow.client_on_intake_side(state, flow._restored_analysis(state["analysis"]), "Alfa")
    assert moved and moved[0] == "Beta"
    assert flow.role_counted(state, _posture("civile", "convenuto"))["role_flip"] == 0


def test_the_plans_model_written_party_never_replaces_the_client_the_facts_name(monkeypatch):
    from src.rag import interactive_flow as flow
    logged = []
    monkeypatch.setattr(flow, "_log", lambda event, data=None, *a, **k: logged.append((event, data)))
    state = {"client": {"name": "Tizio"}}
    plan = flow.plan_client(state, {"parte_assistita": "Caio", "obiettivo": "x"})
    assert plan["parte_assistita"] == "Tizio" and plan["obiettivo"] == "x"
    assert logged and logged[0][0] == "role_decision" and logged[0][1]["client_before"] == "Caio"
    same = {"parte_assistita": "Tizio (acquirente)"}
    assert flow.plan_client(state, same) is same and flow.plan_client({}, same) is same
    # the counter sees the method writing for another party than the one settled
    flips = flow.role_counted({"method_client": "Caio", "role_check": {"client": "Tizio"}}, _posture("civile",
                                                                                                    "attore"))
    assert flips["role_flip"] == 1


def test_a_whole_answer_sets_the_new_counters_at_zero(monkeypatch):
    import test_v60g
    from test_v60f import _graded_state
    from src.rag import interactive_flow as flow, lawyer_method as M
    state = _graded_state(monkeypatch)
    state["mode"] = "single_shot"
    monkeypatch.setenv("LAWYER_METHOD", "true")
    line = M.NO_ACTION_LINE.format(istituto="servitù")

    def plan(ctx, call=None):
        ctx["candidates"], ctx["no_action"], ctx["no_action_kind"] = [], line, "none"
        ctx["institutions"] = [{"name": "servitù", "rubricated": "", "claim": ""}]
        return M.clean_method({}, [])
    monkeypatch.setattr(M, "method_plan", plan)
    text, _ = test_v60g._answer(monkeypatch, state, {}, None, {})
    counters = flow.case_counters(state)
    for key in ("role_flip", "corpus_line_code", "amministrativo_without_act", "act_last_pass_stripped"):
        assert counters[key] == 0, key
    assert counters["corpus_line"] == 0 and "amministrativo_refused_no_act" in counters
    assert state["method_client"] == "Leonardo" and state["role_check"]["client"] == "Leonardo"


def test_the_party_side_words_are_data():
    data = json.loads((ROOT / "data" / "party_sides.json").read_text(encoding="utf-8"))
    assert set(data) >= {"accused", "victim"} and data["accused"] and data["victim"]
    from src.rag import interactive_flow as flow
    assert flow.party_side({"name": "X", "role": "imputato"}) == "accused"
    assert flow.party_side({"name": "X", "role": "persona offesa"}) == "victim"
    assert flow.party_side({"name": "X", "role": "vicino di casa"}) == ""


# --- 2. the honesty line ------------------------------------------------------------------------------------------------

def _penal_ctx(institutions, recs=None):
    return {"records": recs or {}, "circumstances": {}, "posture": _posture("penale", "imputato", "indagini"),
            "case": "Tizio, aggredito di notte nel suo negozio, ha colpito l'aggressore con una spranga.",
            "assessment": {}, "institutions": list(institutions), "client": "Tizio"}


@pytest.mark.parametrize("names", [None, [], ["Codice Penale 2026"]])
def test_never_the_line_on_legittima_difesa(monkeypatch, names):
    from src.rag import corpus_laws, lawyer_method as M
    monkeypatch.setitem(corpus_laws._NAMES, "names", names)
    for inst in ("legittima difesa", {"name": "eccesso colposo di legittima difesa", "rubricated": "Difesa legittima",
                                      "claim": "la difesa dall'aggressione"}):
        ctx = _penal_ctx([inst], {"52 c.p.": _rec("Difesa legittima", _req(1, "un pericolo attuale di un'offesa "
                                                                               "ingiusta"))})
        assert M.governing_law(inst, ctx) == ("c.p.", "tabella")
        assert M.corpus_line(ctx) == ""
        assert next(iter(ctx["corpus_gap_log"].values()))["code"] is True
    # the check sees the line printed on a code institution
    printed = M.CORPUS_LINE.format(materia="legittima difesa") + "."
    assert M.corpus_line_on_code(printed, _penal_ctx(["legittima difesa"])) == ["legittima difesa"]


def test_the_governing_law_from_the_table_a_module_a_rubrica_or_a_citation():
    from src.rag import lawyer_method as M
    assert M.governing_law("contratto di viaggio", {}) == ("d.lgs. 79/2011", "tabella")
    assert M.governing_law("comodato", {}) == ("c.c.", "modulo")                 # a code module
    assert M.governing_law("licenziamento", {}) == ("L. 604/1966", "modulo")    # a special-law module
    recs = {"2058 c.c.": _rec("Risarcimento in forma specifica")}
    assert M.governing_law({"name": "reintegrazione", "rubricated": "Risarcimento in forma specifica"},
                           {"records": recs}) == ("c.c.", "rubrica")
    assert M.governing_law("tutela ex l. 104/1992", {}) == ("l. 104/1992", "citazione")
    assert M.governing_law("istituto senza nome noto", {}) == ("", "")


def test_the_line_fires_only_outside_the_codes_and_the_archive(monkeypatch):
    from src.rag import corpus_laws, lawyer_method as M
    ctx = {"records": {}, "institutions": ["contratto di viaggio"]}
    monkeypatch.setitem(corpus_laws._NAMES, "names", None)          # the graph not read: never
    assert M.corpus_line(ctx) == ""
    monkeypatch.setitem(corpus_laws._NAMES, "names", ["Codice Civile 2026"])
    assert M.corpus_line(ctx) == ("Il corpus non contiene ancora informazioni sufficienti su contratto di viaggio: il "
                                  "parere si limita alle norme dei codici.")
    # a special law the archive holds: never
    road = {"records": {}, "institutions": ["violazione del codice della strada"]}
    assert M.corpus_line(road) and M.governing_law(road["institutions"][0], road)[0] == "d.lgs. 285/1992"
    monkeypatch.setitem(corpus_laws._NAMES, "names", ["Codice Civile 2026", "Codice della strada 2026"])
    assert M.corpus_line(road) == "" and road["corpus_gap_log"]["violazione del codice della strada"]["in_archive"]
    # unknown governing law: never
    assert M.corpus_line({"records": {}, "institutions": ["istituto senza nome noto"]}) == ""


def test_a_no_claim_answer_on_a_code_institution_keeps_ten_sections_with_the_summary_line(monkeypatch):
    from src.rag import corpus_laws, lawyer_method as M
    monkeypatch.setitem(corpus_laws._NAMES, "names", ["Codice Civile 2026"])
    lent = "Tizio ha dato in comodato la sua auto a Caio, che la restituisce graffiata."
    ctx = {"records": {"1476 c.c.": _rec("Obbligazioni principali del venditore", _req(1, "consegnare la cosa"))},
           "circumstances": {}, "posture": _posture("civile", "attore"), "case": lent, "assessment": {},
           "institutions": ["comodato"], "client": "Tizio"}
    ctx["candidates"] = M.thesis_candidates(ctx)
    plan = M.clean_method({}, ctx["candidates"])
    text = M.compose([("Norme applicate ai fatti", "Regola.")], ctx, plan)
    secs = dict(M.answer_sections(text))
    assert list(secs) == M.titles()
    assert secs[M.title("sintesi")] == M.NO_CLAIM_SUMMARY.format(materia="comodato", client="Tizio")
    assert "Il corpus non contiene" not in text and ctx["corpus_line_code"] == 0


def test_a_no_claim_answer_on_a_special_law_outside_the_archive_says_the_line(monkeypatch):
    from src.rag import corpus_laws, lawyer_method as M
    monkeypatch.setitem(corpus_laws._NAMES, "names", ["Codice Civile 2026"])
    trip = "Tizio ha acquistato un pacchetto turistico; l'albergo era diverso da quello descritto nel catalogo."
    ctx = {"records": {"1218 c.c.": _rec("Responsabilità del debitore", _req(1, "l'inadempimento", "da provare"))},
           "circumstances": {}, "posture": _posture("civile", "attore"), "case": trip, "assessment": {},
           "institutions": ["pacchetto turistico"], "client": "Tizio", "no_action": "x", "no_action_kind": "none"}
    assert M.no_claim_summary(ctx, "none") == M.CORPUS_LINE.format(materia="pacchetto turistico") + "."
    assert M.no_claim_summary({**ctx, "institutions": ["comodato"]}, "none") == M.NO_CLAIM_SUMMARY.format(
        materia="comodato", client="Tizio")


def test_the_institution_law_table_is_data_and_compiles():
    from src.rag import lawyer_method as M
    data = json.loads((ROOT / "data" / "institution_laws.json").read_text(encoding="utf-8"))
    assert data["institutions"] and all(e.get("law") and e.get("patterns") for e in data["institutions"])
    names = [e["name"] for e in M.institution_laws()]
    assert names.index("contratto di viaggio") < names.index("obbligazioni e contratti")     # the special laws first


# --- 3. the administrative act --------------------------------------------------------------------------------------------

SALE = ("Tizio ha acquistato da Caio un appartamento con rogito del 3 marzo 2025. Dopo il rogito ha scoperto difformità "
        "edilizie rispetto al permesso di costruire depositato in Comune; Caio non le aveva dichiarate. Tizio vuole "
        "agire contro Caio per la riduzione del prezzo.")
ACT = SALE + " Il Comune ha poi notificato a Tizio un'ordinanza di demolizione delle opere difformi."


def _row(number, code, doc=""):
    return {"_article": (number, code), "d": {"name": doc or code}}


@pytest.mark.parametrize("text, act", [
    (SALE, False), (ACT, True),
    ("Il Comune non ha adottato alcun provvedimento contro Tizio.", False),
    ("Tizio chiede l'annullamento del contratto per dolo del venditore.", False),
    ("Il locatore, nel Comune di Milano, ha opposto il diniego di rinnovo alla prima scadenza.", False),
    ("Il Comune ha respinto l'istanza di permesso di costruire presentata da Tizio.", True),
    ("Tizio vuole impugnare al TAR il diniego dell'autorizzazione.", True),
    ("Sull'istanza di Tizio si è formato il silenzio-rifiuto dell'amministrazione.", True),
])
def test_the_administrative_act(text, act):
    from src.rag import family_gate
    assert bool(family_gate.administrative_act(text)) is act


def test_a_civil_sale_naming_the_comune_and_the_permesso_admits_nothing_administrative():
    from src.rag import family_gate, lawyer_method as M, special_laws
    assert "amministrativo" not in [m["id"] for m in M.active_modules({"case": SALE, "institutions": []})]
    ctx = family_gate.context(SALE)
    assert ctx["admin_act"] is False and ctx["modules"] == [] and ctx["modules_refused_no_act"] == ["amministrativo"]
    assert not special_laws.signals(SALE)["legge_241_1990"]
    assert special_laws.family_signal("amministrativo", SALE, act=False)            # the signal, refused for the act
    kept, dropped, _ = family_gate.gate([_row("29", "c.p.a."), _row("21-nonies",
                                                                     "Legge sul Procedimento Amministrativo 2026")],
                                        ctx, set())
    assert kept == [] and {x["why"] for x in dropped} == {family_gate.NO_ACT_WHY}
    assert {x["family"] for x in dropped} == {"amministrativo"}


def test_with_the_act_the_module_and_the_cpa_are_admitted():
    from src.rag import family_gate, lawyer_method as M
    assert "amministrativo" in [m["id"] for m in M.active_modules({"case": ACT, "institutions": []})]
    ctx = family_gate.context(ACT)
    assert ctx["admin_act"] and "amministrativo" in ctx["modules"]
    kept, dropped, _ = family_gate.gate([_row("29", "c.p.a.")], ctx, set())
    assert len(kept) == 1 and not dropped


def test_the_user_naming_the_cpa_article_or_code_admits_it():
    from src.rag import family_gate
    ctx = family_gate.context(SALE + " Si applica l'art. 29 c.p.a.?")
    got = family_gate.decide("c.p.a.", "", ctx, number="29")
    assert got["ok"] and got["door"] == family_gate.DOOR_USER
    kept, _, _ = family_gate.gate([_row("30", "c.p.a.")], family_gate.context(SALE), {("30", "c.p.a.")})
    assert len(kept) == 1                                                       # the «articolo» door


def test_an_institution_naming_an_act_the_case_does_not_state_activates_nothing():
    # review: the act is read in the case's own words only -- a model-written institution name (the first read's, an
    # alternative's, the frame's) naming an act the case does not state never opens the module or the c.p.a.
    from src.rag import family_gate, lawyer_method as M
    contract = "Alfa ha venduto a Beta un capannone; Beta non ha pagato il saldo."
    assert "amministrativo" not in [m["id"] for m in M.active_modules(
        {"case": contract, "institutions": ["annullamento d'ufficio del permesso di costruire"]})]
    assert family_gate.act_text(contract, ["ricorso al TAR contro il diniego di sanatoria"]) == contract
    with_act = contract + " Il Comune ha poi annullato d'ufficio il permesso di costruire di Beta."
    assert "amministrativo" in [m["id"] for m in M.active_modules({"case": with_act, "institutions": []})]


def test_the_scope_check_drops_cpa_without_the_act():
    from src.rag import lawyer_method as M
    ctx = {"records": {"1490 c.c.": _rec("Garanzia per i vizi della cosa venduta")}, "case": SALE, "institutions": []}
    scope = M.article_scope(ctx, ["Il ricorso va proposto entro sessanta giorni (art. 29 c.p.a.)."])
    assert "29 c.p.a." not in scope and "1490 c.c." in scope and ctx["act_scope_refused"] == ["29 c.p.a."]
    text = "La garanzia opera (art. 1490 c.c.). Il ricorso al giudice amministrativo segue l'art. 29 c.p.a."
    assert M.articles_out(text, scope) == "La garanzia opera (art. 1490 c.c.)."
    with_act = {"records": {}, "case": ACT, "institutions": []}
    assert "29 c.p.a." in M.article_scope(with_act, ["art. 29 c.p.a."])


def test_the_administrative_regime_needs_the_act():
    from src.rag import fact_signals
    analysis = {"_case_text": SALE, "questioni": [{"titolo": "annullamento e provvedimento", "istituti": []}],
                "fact_signals": {"facts": {}}}
    log = []
    got = fact_signals.regime_candidates(analysis, ["amministrativo"], log=log)
    assert not [r for r in got if r["id"] == "procedimento_amministrativo"]
    assert {"regime": "procedimento_amministrativo", "why": "no_administrative_act"} in log
    analysis["_case_text"] = ACT
    got = fact_signals.regime_candidates(analysis, ["amministrativo"])
    assert [r for r in got if r["id"] == "procedimento_amministrativo"]


def test_the_last_pass_strips_cpa_and_counts(monkeypatch):
    from src.rag import family_gate, interactive_flow as flow
    state = {"text": SALE, "request": "", "family_gate": {"admin_act": [], "modules_refused_no_act": ["amministrativo"],
                                                          "dropped": [{"article": "29 c.p.a.", "why":
                                                                       family_gate.NO_ACT_WHY}]},
             "records": {"1490 c.c.": {}}, "intake": {}}
    draft = "La garanzia per i vizi opera. Il termine è di sessanta giorni (art. 29 c.p.a.). Si propone la riduzione."
    out = flow.admin_act_out(state, draft, [{"act_scope_refused": ["30 c.p.a."]}])
    assert "c.p.a." not in out and "riduzione" in out
    c = state["block_counters"]
    assert c["act_last_pass_stripped"] == 1 and c["amministrativo_refused_no_act"] == 4
    assert c["amministrativo_without_act"] == 0
    # a c.p.a. record or the module still active without the act: counted
    state2 = {**state, "records": {"29 c.p.a.": {}}, "block_counters": {"modules_active": {"amministrativo":
                                                                                         "istituto: «x»"}}}
    flow.admin_act_out(state2, "Testo.", [])
    assert state2["block_counters"]["amministrativo_without_act"] == 2
    # with the act: nothing stripped
    state3 = {**state, "family_gate": {"admin_act": ["ordinanza di demolizione"]}, "block_counters": {}}
    assert flow.admin_act_out(state3, draft) == draft and state3["block_counters"]["amministrativo_without_act"] == 0


def test_the_act_table_is_data():
    data = json.loads((ROOT / "data" / "administrative_act.json").read_text(encoding="utf-8"))
    assert data["patterns"] and data["modules"] == ["amministrativo"] and data["codes"] == ["c.p.a."]
    assert data["families"] == ["amministrativo"]


# --- counters, nightly, docs, copy ----------------------------------------------------------------------------------------

def test_the_nightly_fails_a_cell_with_a_must_be_zero_counter_above_zero():
    nightly = _script("nightly_sim")
    assert nightly.zero_counters_check({"role_flip": 0, "corpus_line_code": 0})["ok"]
    for key in ("role_flip", "corpus_line_code", "amministrativo_without_act"):
        got = nightly.zero_counters_check({key: 1})
        assert not got["ok"] and key in got["why"]
    # with check case_frame run on the cell (CASE_FRAME), role_flip is left to it: one failure per flip
    assert nightly.zero_counters_check({"role_flip": 1}, frame_ran=True)["ok"]
    assert not nightly.zero_counters_check({"role_flip": 1, "corpus_line_code": 1}, frame_ran=True)["ok"]
    assert not nightly.frame_check({"ran": True}, {"role_flip": 1})["ok"]
    import inspect
    src = inspect.getsource(nightly.run_cell)
    assert "zero_counters_check(out[\"counters\"], frame_ran=" in src and "frame_check(" in src


def test_case_counters_carry_the_new_counters():
    from src.rag import interactive_flow as flow
    state = {"block_counters": {"role_flip": 0, "role_request_wins": 0, "role_client_moved": 0, "corpus_line_code": 0,
                                "amministrativo_refused_no_act": 1, "amministrativo_without_act": 0}}
    got = flow.case_counters(state)
    for key in state["block_counters"]:
        assert key in got


def test_role_corpus_counted_reads_the_delivered_answer(monkeypatch):
    from src.rag import corpus_laws, interactive_flow as flow, lawyer_method as M
    monkeypatch.setitem(corpus_laws._NAMES, "names", ["Codice Penale 2026"])
    state = {"text": "x", "request": "", "block_counters": {}, "role_check": {}}
    draft = M.CORPUS_LINE.format(materia="legittima difesa") + ". Altro."
    flow.role_corpus_counted(state, draft, [{"institutions": ["legittima difesa"], "records": {}}],
                             _posture("penale", "imputato"))
    c = state["block_counters"]
    assert c["corpus_line_code"] == 1 and c["corpus_line"] == 1 and c["role_flip"] == 0


def test_the_docs_and_the_copy_name_the_release():
    rules = (ROOT / "docs" / "RULES.md").read_text(encoding="utf-8")
    assert "| v79.14.1" in rules and "role_flip" in rules and "amministrativo_without_act" in rules
    copy = (ROOT / "docs" / "copy_review.md").read_text(encoding="utf-8")
    assert "NO_CLAIM_SUMMARY" in copy
