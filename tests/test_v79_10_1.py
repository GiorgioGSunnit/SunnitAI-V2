"""v79.10.1 -- Block A of the 79.10 brief, the three regressions, on synthetic sentences and records only:

A1. no claim admitted for the client: the method steps aside to the prose answer as 79.6 did (the placeholder, v76's
    note) -- the no-action sentence is the note's why, once, never «Linea principale», never with a forecast; the
    missing-laws line kept beside it; every client read -- with several clients the method steps aside only when none
    has a line, else the clients with none are said in a note and the others keep their sections (review); counter
    no_action_prose; the nightly's check
A2. an article only a remedies.json matter's actions admit is a claim only inside the first read's institution family
    (the module whose actions hold it, else its matter; a module a matter lists among its «member_modules» -- custodia
    in the fatti illeciti -- is of the matter's family too); where the read's institution has a module, only that
    module's; with no first-read institution, as in v79.8; counter claims_refused_family
A3. with one sample the kind decides first: while a contest, a defence or a riqualificazione is eligible, no
    circumstance and no «elemento escluso» fragment is the main line; an attenuante never leads while any other line
    is eligible; with the charge not fully integrato, the containment line neither (review); counter main_demoted
"""

import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for name, value in (("LLM_API_KEY", "k"), ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "k"), ("NEO4J_URI", "bolt://localhost:7687"), ("NEO4J_USER", "n"),
                        ("NEO4J_PASSWORD", "x")):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("THESIS_DETERMINISTIC", "")
    from src.rag import plan_then_write
    monkeypatch.setattr(plan_then_write, "negation_review", lambda items, known, call=None: [])


def _script(name):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _req(i, text, status="provato"):
    return {"id": f"R{i}", "comma": 1, "kind": "requisito", "text": text, "status": status}


def _rec(rubrica, *reqs, side="weapon"):
    return {"rubrica": rubrica, "side": side, "requisites": list(reqs)}


def _posture(family, role, stage="primo_grado"):
    from src.rag import posture
    return {"rito": family, "family": family, "role": role, "stage": stage,
            "entry": posture.entry_for(family, role, stage)}


ATTORE = _posture("civile", "attore", "pre_filing")
IMPUTATO = _posture("penale", "imputato")


def _civil(recs, case, institutions=()):
    return {"records": recs, "circumstances": {}, "posture": ATTORE, "case": case, "assessment": {},
            "institutions": list(institutions), "client": "Tizio"}


# --- A1: the method steps aside --------------------------------------------------------------------------------------------

LENT = "Tizio ha dato in comodato la sua auto a Caio, che la restituisce graffiata."
NO_ACTION = "Tra le norme esaminate per comodato nessuna attribuisce un'azione al cliente"


def _no_action_ctx():
    return _civil({"1476 c.c.": _rec("Obbligazioni principali del venditore", _req(1, "consegnare la cosa")),
                   "1225 c.c.": _rec("Prevedibilità del danno", _req(2, "il danno prevedibile"))}, LENT, ["comodato"])


def test_no_claim_admitted_gives_no_candidate_and_the_method_steps_aside_with_the_sentence_once():
    from src.rag import lawyer_method as M
    ctx = _no_action_ctx()
    ctx["candidates"] = M.thesis_candidates(ctx)
    assert ctx["candidates"] == [] and ctx["no_action"] == NO_ACTION
    plan = M.clean_method({}, ctx["candidates"])
    assert M.compose([("Norme applicate ai fatti", "Regola.")], ctx, plan) == ""      # the standard body
    refusal = M.method_refusal([(ctx, plan)])
    assert refusal["no_action"] is True and refusal["lines"] == []
    note = M.not_applied_note(refusal["why"])
    assert note == ("Nota: il metodo di analisi per elementi non è stato applicato a questo parere: tra le norme "
                    "esaminate per comodato nessuna attribuisce un'azione al cliente. Il parere segue la struttura "
                    "ordinaria.")
    assert M.no_action_problems(note + "\n\nTesto del parere ordinario.") == []
    assert "nessuna_azione" not in M.ADMITTED_BY and not hasattr(M, "NO_ACTION_ARGUMENT")


def test_a_norm_of_action_with_an_element_excluded_keeps_v76s_why():
    from src.rag import lawyer_method as M
    ctx = _civil({"1809 c.c.": _rec("Restituzione", _req(1, "la scadenza del termine", "non risulta"))}, LENT,
                 ["comodato"])
    ctx["candidates"] = M.thesis_candidates(ctx)
    assert ctx["candidates"] == [] and "no_action" not in ctx
    refusal = M.method_refusal([(ctx, M.clean_method({}, []))])
    assert refusal["why"] == M.NOT_APPLIED_WHY["placeholder"] and refusal["no_action"] is False


def test_every_client_is_read_and_a_client_with_a_line_keeps_the_method():
    from src.rag import lawyer_method as M
    good = _civil({"1809 c.c.": _rec("Restituzione", _req(1, "la scadenza del termine"))}, LENT, ["comodato"])
    good["candidates"] = M.thesis_candidates(good)
    good_plan = M.clean_method({}, good["candidates"])
    assert M.method_refusal([(good, good_plan)]) == {} and M.client_refusal(good, good_plan) == {}
    bad = _no_action_ctx()
    bad["candidates"] = M.thesis_candidates(bad)
    bad_plan = M.clean_method({}, [])
    # review: one client with a line keeps the method -- the whole answer never steps aside for another client
    assert M.method_refusal([(good, good_plan), (bad, bad_plan)]) == {}
    refusal = M.client_refusal(bad, bad_plan)
    assert refusal["no_action"] and NO_ACTION[1:] in refusal["why"]
    notes = M.refused_clients_notes([("Sempronio", refusal), ("Mevio", refusal)])
    assert notes == ["Nota: il metodo di analisi per elementi non è stato applicato alla posizione di Sempronio e "
                     "Mevio: tra le norme esaminate per comodato nessuna attribuisce un'azione al cliente."]
    assert M.no_action_problems("\n\n".join(notes + ["- Linea principale: restituzione del bene."])) == []
    whole = M.method_refusal([(bad, bad_plan), (_no_action_ctx(), bad_plan)])       # no client has a line
    assert whole["no_action"] and whole["why"] == refusal["why"]


@pytest.mark.parametrize("answer, problems", [
    ("Nota: … tra le norme esaminate per comodato nessuna attribuisce un'azione al cliente. Il parere segue …", []),
    ("- Linea principale: Tra le norme esaminate per comodato nessuna attribuisce un'azione al cliente.",
     ["come «Linea principale»"]),
    ("Tra le norme esaminate per X nessuna attribuisce un'azione al cliente.\n**Esito probabile**: sfavorevole.",
     ["con una previsione"]),
    ("Tra le norme esaminate per X nessuna attribuisce un'azione al cliente.\n\nDi nuovo: tra le norme esaminate per X "
     "nessuna attribuisce un'azione al cliente.", ["2 volte"]),
])
def test_the_no_action_check(answer, problems):
    from src.rag import lawyer_method as M
    assert M.no_action_problems(answer) == problems
    check = _script("nightly_sim").no_action_check(answer)
    assert check["check"] == "no_action_prose" and check["ok"] is (not problems)


def test_the_flow_steps_aside_with_the_sentence_and_counts_it(monkeypatch):
    import test_v60g
    from test_v60f import _graded_state
    from src.rag import interactive_flow as flow, lawyer_method as M
    state = _graded_state(monkeypatch)
    state["mode"] = "single_shot"
    monkeypatch.setenv("LAWYER_METHOD", "true")
    monkeypatch.setattr(M, "objections_plan", lambda ctx, plan: plan)
    line = M.NO_ACTION_LINE.format(istituto="servitù")

    def plan(ctx, call=None):           # the record offers the claimant nothing: no norm confers an action
        ctx["candidates"], ctx["no_action"] = [], line
        return M.clean_method({}, [])
    monkeypatch.setattr(M, "method_plan", plan)
    events = []
    real = flow._log
    monkeypatch.setattr(flow, "_log", lambda e, d, ms=None: (events.append((e, d)), real(e, d, ms)))
    text, _ = test_v60g._answer(monkeypatch, state, {}, None, {})
    assert state["method_refused"] == "tra le norme esaminate per servitù nessuna attribuisce un'azione al cliente"
    assert ("lawyer_method_refused", {"why": "no_action", "tesi": "Tesi da costruire sugli elementi ancora da "
                                                                  "provare"}) in events
    assert text.lower().count(M.NO_ACTION_CORE) == 1 and "Linea principale" not in text
    assert "Esito probabile" not in text and M.no_action_problems(text) == []
    assert "**1. Sintesi della soluzione**" not in text
    assert state["block_counters"]["no_action_prose"] == 1
    assert flow.case_counters(state)["claims_refused_family"] == 0 and "no_action_lines" not in flow.case_counters(state)


# --- A2: the matter table inside the read's family -----------------------------------------------------------------------

PROFESSIONAL = ("Tizio aveva incaricato l'avvocato Caio di proporre appello; Caio ha lasciato scadere il termine e "
                "l'appello non è stato proposto.")
PROFESSIONAL_READ = ["responsabilità professionale dell'avvocato", "inadempimento del contratto d'opera intellettuale"]


def _professional_records():
    return {"1490 c.c.": _rec("Garanzia per i vizi della cosa venduta", _req(1, "la cosa venduta presenta vizi")),
            "2051 c.c.": _rec("Danno cagionato da cosa in custodia", _req(2, "il danno cagionato dalla cosa")),
            "1218 c.c.": _rec("Responsabilità del debitore", _req(3, "il debitore non esegue la prestazione dovuta"))}


def test_a_sale_and_a_custody_article_never_lead_a_professional_liability_case():
    from src.rag import lawyer_method as M
    ctx = _civil(_professional_records(), PROFESSIONAL, PROFESSIONAL_READ)
    ctx["candidates"] = M.thesis_candidates(ctx)
    assert ctx["claim_modules"] == [] and set(ctx["claim_families"]) == {"obbligazioni", "contratti"}
    assert [c["article"] for c in ctx["candidates"] if c["kind"] == "claim"] == ["1218 c.c."]
    assert ctx["claims_refused_family"] == ["1490 c.c.", "2051 c.c."]
    assert ctx["claims_refused"]["1490 c.c."].startswith(M.FAMILY_REFUSED_HEAD)
    assert "(vendita;" in ctx["claims_refused"]["1490 c.c."] and "(custodia;" in ctx["claims_refused"]["2051 c.c."]
    plan = M.clean_method({}, ctx["candidates"])
    counts = M.selection_counts(ctx, plan, {})
    assert counts["claims_refused_family"] == 2 and counts["main_admitted_by"] == "norma generale delle obbligazioni"


def test_with_only_out_of_family_articles_no_action_is_said():
    from src.rag import lawyer_method as M
    recs = {k: v for k, v in _professional_records().items() if k != "1218 c.c."}
    ctx = _civil(recs, PROFESSIONAL, PROFESSIONAL_READ)
    assert M.thesis_candidates(ctx) == [] and ctx["claims_any_admitted"] is False
    assert ctx["no_action"].startswith("Tra le norme esaminate per responsabilità professionale dell'avvocato")


def test_a_matter_article_of_the_read_family_is_a_claim():
    from src.rag import lawyer_method as M
    recs = {"1418 c.c.": _rec("Cause di nullità del contratto", _req(1, "il contratto è contrario a norme imperative"))}
    ctx = _civil(recs, "Tizio ha firmato con Caio una scrittura privata per la cessione di un credito inesistente.",
                 ["nullità del contratto"])
    cands = M.thesis_candidates(ctx)
    assert (cands[0]["kind"], cands[0]["article"]) == ("claim", "1418 c.c.")      # the matter's remedies follow it
    assert cands[0]["admitted_by"] == "remedies.json: contratti" and ctx["claims_refused_family"] == []


def test_where_the_read_institution_has_a_module_the_matter_admits_nothing_outside_it():
    from src.rag import lawyer_method as M
    recs = {"1418 c.c.": _rec("Cause di nullità del contratto", _req(1, "il contratto è contrario a norme imperative"))}
    # the consumer module is named by the read but not active: no professional counterparty among the certain facts
    ctx = _civil(recs, "Tizio ha comprato da Caio, un privato, un telefono che non funziona.",
                 ["garanzia legale del consumatore", "nullità del contratto"])
    assert M.thesis_candidates(ctx) == [] and ctx["claim_modules"] == []
    assert ctx["claim_families"][0] == "consumo" and ctx["claims_refused_family"] == ["1418 c.c."]


def test_with_a_module_active_the_v79_8_outside_rule_stands():
    from src.rag import lawyer_method as M
    recs = {"2051 c.c.": _rec("Danno cagionato da cosa in custodia", _req(1, "il danno cagionato dalla cosa"))}
    ctx = _civil(recs, LENT, ["comodato"])
    M.thesis_candidates(ctx)
    assert ctx["claims_refused"]["2051 c.c."] == M.CLAIM_REFUSED["outside"] and ctx["claims_refused_family"] == []


def test_with_no_first_read_institution_the_matter_table_admits_as_before():
    from src.rag import lawyer_method as M
    recs = {"2051 c.c.": _rec("Danno cagionato da cosa in custodia", _req(1, "il danno cagionato dalla cosa"))}
    ctx = _civil(recs, "Tizio è inciampato in una buca e si è fratturato il polso.")
    cands = M.thesis_candidates(ctx)
    assert ctx["claim_families"] is None and ("claim", "2051 c.c.") in [(c["kind"], c["article"]) for c in cands]


BITE = "Il cane di Caio ha morso Tizio al parco."


def test_a_custody_article_is_inside_the_tort_family_a_sale_article_outside_the_contract_one():
    # review: custodia is a member module of the fatti illeciti (2052 for a dog's bite under a tort read); a typed
    # contract's module is not of «contratti» (1490 never under a generic contract read)
    from src.rag import lawyer_method as M
    assert M.article_families("2052 c.c.") == ["custodia", "fatti_illeciti"]
    assert M.article_families("1490 c.c.") == ["vendita"] and M.article_families("1418 c.c.") == ["contratti"]
    recs = {"2052 c.c.": _rec("Danno cagionato da animali", _req(1, "il danno cagionato dall'animale"))}
    for read in (["responsabilità extracontrattuale (fatto illecito)"], ["danno cagionato da animali"]):
        ctx = _civil(recs, BITE, read)
        cands = M.thesis_candidates(ctx)
        assert ctx["claim_families"] == ["fatti_illeciti"] and ctx["claims_refused_family"] == []
        assert (cands[0]["kind"], cands[0]["article"], cands[0]["admitted_by"]) == (
            "claim", "2052 c.c.", "remedies.json: fatti_illeciti") and "no_action" not in ctx


def test_a_generic_civil_liability_read_names_no_tort_family():
    # review: «responsabilità civile» is the professional's as much as the tort's -- no custody article through it
    from src.rag import lawyer_method as M
    ctx = _civil(_professional_records(), PROFESSIONAL, ["responsabilità civile professionale dell'avvocato"])
    ctx["candidates"] = M.thesis_candidates(ctx)
    assert "fatti_illeciti" not in ctx["claim_families"] and "2051 c.c." in ctx["claims_refused_family"]


def test_every_matter_pattern_compiles_and_the_family_of_an_article():
    import re
    from src.rag import lawyer_method as M
    for m in M.remedies()["matters"]:
        assert m["patterns"] and all(re.compile(p) for p in m["patterns"])
    assert M.article_family("1490 c.c.") == "vendita" and M.article_family("2051 c.c.") == "custodia"
    assert M.article_family("1418 c.c.") == "contratti" and M.article_family("575 c.p.") == ""


# --- A3: the main line by kind first ----------------------------------------------------------------------------------------

def _charge(*statuses, all_integrated=False):
    texts = ("cagiona ad alcuno una lesione personale", "dalla quale deriva una malattia nel corpo o nella mente")
    recs = {"582 c.p.": _rec("Lesione personale", *[_req(i + 1, t, s) for i, (t, s) in enumerate(zip(texts, statuses))],
                             side="threat")}
    circ = {"62-bis c.p.": {"kind": "attenuante", "rubrica": "attenuante: Circostanze attenuanti generiche",
                            "requisites": [_req(30, "la condotta successiva al reato", "provato")]}}
    return {"records": recs, "circumstances": circ, "posture": IMPUTATO, "case": "Alfa colpisce Beta.",
            "assessment": {"article": "582 c.p.", "all_integrated": all_integrated}, "institutions": [],
            "client": "Alfa", "riti": []}


def _valued(cands, **by_kind):
    return {"valutazioni": {c["id"]: {"realismo": by_kind.get(c["kind"], "medio")} for c in cands}}


def test_a_charge_with_one_contestable_element_and_62_bis_yields_the_contest_as_principale():
    from src.rag import lawyer_method as M
    ctx = _charge("provato", "da provare")
    ctx["candidates"] = M.thesis_candidates(ctx)
    assert [c["kind"] for c in ctx["candidates"]] == ["contest_integrated", "contest_open", "attenuante",
                                                      "contenimento"]
    raw = _valued(ctx["candidates"], attenuante="alto", contest_open="medio", contenimento="medio")
    plan = M.clean_method(raw, ctx["candidates"])
    sols, main, sub = M.selection(ctx, plan)
    assert sols[main]["kind"] == "contest_open"
    att = next(t for t in sols if t["kind"] == "attenuante")
    assert att["main_demoted"] == M.MAIN_DEMOTED_WHY["circumstance"]
    assert sols[sub]["kind"] == "attenuante"                     # subordinate, never the main line
    counts = M.selection_counts(ctx, plan, raw)
    assert counts["chosen_kind"] == "contest_open" and counts["main_demoted"] == 2     # the attenuante, the containment


@pytest.mark.parametrize("contest, containment", [("basso", "medio"), ("basso", "alto"), ("medio", "alto")])
def test_the_brief_s_test_holds_whatever_the_one_sample_s_realism(contest, containment):
    # review: with the charge open the containment line is v79.7's subordinate line -- never over the contest
    from src.rag import lawyer_method as M
    ctx = _charge("provato", "da provare")
    ctx["candidates"] = M.thesis_candidates(ctx)
    raw = _valued(ctx["candidates"], attenuante="alto", contest_open=contest, contenimento=containment)
    sols, main, sub = M.selection(ctx, M.clean_method(raw, ctx["candidates"]))
    assert sols[main]["kind"] == "contest_open" and sols[sub]["kind"] != "contest_open"
    cont = next(t for t in sols if t["kind"] == "contenimento")
    assert cont["main_demoted"] == M.MAIN_DEMOTED_WHY["containment"]


def test_an_attenuante_is_always_subordinate_to_the_containment_line():
    from src.rag import lawyer_method as M
    ctx = _charge("provato", "provato", all_integrated=True)
    ctx["candidates"] = M.thesis_candidates(ctx)
    raw = _valued(ctx["candidates"], attenuante="alto", contenimento="medio")
    sols, main, _ = M.selection(ctx, M.clean_method(raw, ctx["candidates"]))
    assert sols[main]["kind"] == "contenimento"


def test_an_escluso_fragment_never_leads_over_a_real_contest():
    from src.rag import lawyer_method as M
    known = {"R1": {"text": "il debitore", "status": "non risulta"},
             "R2": {"text": "il danno cagionato dalla cosa", "status": "da provare"}}
    sols = [{"tesi": "Frammento", "candidate": "T1", "kind": "contest_excluded", "realismo": "alto", "contesta": ["R1"],
             "poggia_su": []},
            {"tesi": "Contesta R2", "candidate": "T2", "kind": "contest_open", "realismo": "medio", "contesta": ["R2"],
             "poggia_su": []}]
    assert M.never_main(sols, [0, 1], known) == {0: M.MAIN_DEMOTED_WHY["fragment"]}
    recs = {"2051 c.c.": _rec("Custodia", _req(1, "il debitore", "non risulta"), _req(2, "il danno cagionato dalla cosa",
                                                                                     "da provare"), side="threat")}
    got, main = M.coherent_solutions({"soluzioni": sols}, recs, {}, ())
    assert got[main]["tesi"] == "Contesta R2"


def test_without_a_contest_a_victim_s_aggravante_may_still_lead():
    from src.rag import lawyer_method as M
    sols = [{"tesi": "Aggravante", "candidate": "T1", "kind": "aggravante", "realismo": "alto", "contesta": [],
             "poggia_su": ["R1"]}]
    assert M.never_main(sols, [0], {}) == {}


def test_the_flow_keeps_the_method_for_a_client_with_a_line_and_says_the_other(monkeypatch):
    # review: several clients, one with no norm of action -- its note, the other's sections kept (never the whole
    # answer in prose for one client, never a client dropped silently)
    import test_v60g
    from test_v60f import _graded_state
    from src.rag import interactive_flow as flow, lawyer_method as M
    state = _graded_state(monkeypatch)
    state["mode"] = "single_shot"
    monkeypatch.setenv("LAWYER_METHOD", "true")
    monkeypatch.setattr(M, "objections_plan", lambda ctx, plan: plan)
    line = M.NO_ACTION_LINE.format(istituto="servitù")

    def plan(ctx, call=None):           # the record's own lines (v70's end-to-end test)
        ctx["candidates"] = cands = M.thesis_candidates(ctx)
        return M.clean_method({"valutazioni": {c["id"]: {"pro": "opere visibili", "contro": "prova della separazione",
                                                         "realismo": "medio"} for c in cands}}, cands)
    monkeypatch.setattr(M, "method_plan", plan)
    monkeypatch.setattr(flow, "clients_of", lambda state: [{"name": "Tizio"}, {"name": "Sempronio"}])

    def inputs(state_, posture, plan_, topics, refs, rows, case, clients):
        out = []
        for c in clients:               # Sempronio: no norm confers an action on him
            ctx, mplan = flow.method_inputs(state_, posture, {**plan_, "parte_assistita": c["name"]}, topics, refs,
                                            rows, case)
            if c["name"] == "Sempronio":
                ctx = {**ctx, "candidates": [], "no_action": line}
                mplan = M.clean_method({}, [])
            out.append((c["name"], ctx, mplan))
        return out
    monkeypatch.setattr(flow, "method_inputs_clients", inputs)
    events = []
    real = flow._log
    monkeypatch.setattr(flow, "_log", lambda e, d, ms=None: (events.append((e, d)), real(e, d, ms)))
    text, _ = test_v60g._answer(monkeypatch, state, {}, {"R1": {"sostiene": ["S1"]}}, {})
    assert not state.get("method_refused") and "non è stato applicato a questo parere" not in text
    assert "Nota: il metodo di analisi per elementi non è stato applicato alla posizione di Sempronio: tra le norme " \
           "esaminate per servitù nessuna attribuisce un'azione al cliente." in text
    assert text.lower().count(M.NO_ACTION_CORE) == 1 and M.no_action_problems(text) == []
    assert "**1. Sintesi della soluzione**" in text                 # Tizio's method sections kept
    assert state["block_counters"]["no_action_prose"] == 1
    assert [e for e, _ in events].count("lawyer_method_refused_clients") == 1
