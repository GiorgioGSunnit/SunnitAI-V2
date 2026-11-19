"""v79.12.5 (part B) -- items 2, 6, 10, 11 of the 79.12.4 brief, on synthetic texts and records only (no model, no
Neo4j, no network):

2. no admitted claim keeps the ten sections: §1 «Il corpus non contiene ancora informazioni sufficienti su <materia>: il
   parere si limita alle norme dei codici» (or, every claim barred by a DECORSO term, the honesty line naming the terms),
   §6 the articles read with no main line and no forecast, §5 and §8 kept; the prose step-aside only when the first read
   found no institution at all; the corpus line in §1 of any parere whose institution has no module and nothing graded
   in its family; counters no_claim_sections, corpus_line
6. role (row 54): «agire / citare in giudizio» whose subject is a named demanding party is that party's threat, never the
   client's claim; a payment demand with no client named is never the claimant by default -- the request's task
   decides, else the role stays unknown with neither side's wording; counters role_threat_not_claim,
   role_unknown_on_demand
10. the counterparty's qualification (impresa, società, ditta, S.r.l., S.p.A., professionista, azienda) a certain fact;
    the consumer code's general titles (1-38, 141 ff.) for any consumer-professional contract, the sale block (128-135)
    only for a sale; the scope gate admits the general clauses and the stage table's procedure articles by default; the
    ATP line kept in §8 of the civil attore cell; the nightly's penal cells at the indagini and the two-turn cell with
    followup_scope
11. (row 24) main_outside_institution accepts general obligations law when the institution's module yields nothing proven
"""

import importlib.util
import json
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


def _posture(family, role, stage="pre_filing"):
    from src.rag import posture
    return {"rito": family, "family": family, "role": role, "stage": stage,
            "entry": posture.entry_for(family, role, stage)}


ATTORE = _posture("civile", "attore")
LENT = "Tizio ha dato in comodato la sua auto a Caio, che la restituisce graffiata."


def _civil(recs, case, institutions=(), **extra):
    return {"records": recs, "circumstances": {}, "posture": ATTORE, "case": case, "assessment": {},
            "institutions": list(institutions), "client": "Tizio", **extra}


def _no_claim_ctx(institutions=("comodato",)):
    return _civil({"1476 c.c.": _rec("Obbligazioni principali del venditore", _req(1, "consegnare la cosa")),
                   "1225 c.c.": _rec("Prevedibilità del danno", _req(2, "il danno prevedibile"))}, LENT, institutions)


def _sections(text):
    from src.rag import lawyer_method as M
    return dict(M.answer_sections(text))


# --- 2. no admitted claim keeps the ten sections -----------------------------------------------------------------------

def test_no_admitted_claim_with_an_institution_keeps_the_ten_sections_with_no_line_and_no_forecast():
    from src.rag import lawyer_method as M
    ctx = _no_claim_ctx()
    ctx["candidates"] = M.thesis_candidates(ctx)
    plan = M.clean_method({}, ctx["candidates"])
    assert ctx["candidates"] == [] and ctx["no_action_kind"] == "none" and M.no_claim(ctx) == "none"
    assert M.client_refusal(ctx, plan) == {} and M.method_refusal([(ctx, plan)]) == {}
    log = {}
    text = M.compose([("Norme applicate ai fatti", "Regola."), ("Profili processuali", "Mediazione obbligatoria.")],
                     ctx, plan, log=log)
    secs = _sections(text)
    assert list(secs) == M.titles()                                      # the ten sections, in order
    assert secs[M.title("sintesi")] == ("Il corpus non contiene ancora informazioni sufficienti su comodato: il "
                                        "parere si limita alle norme dei codici.")
    sol = secs[M.title("soluzioni")]
    assert sol.startswith("Nessuna linea è proposta e nessuna previsione di esito: tra le norme esaminate per comodato "
                          "nessuna attribuisce un'azione al cliente.")
    assert "- Art. 1476 c.c. (Obbligazioni principali del venditore):" in sol and "- Art. 1225 c.c." in sol
    assert "| Tesi |" not in sol
    assert secs[M.title("orientamenti")] and "Mediazione obbligatoria." in secs[M.title("processuali")]  # §5, §8 kept
    assert secs[M.title("conclusioni")] == M.NO_CLAIM_CONCLUSION
    assert "Esito probabile" not in text and "Linea principale" not in text and "Tesi da costruire" not in text
    assert M.no_action_problems(text) == [] and text.lower().count(M.NO_ACTION_CORE) == 1
    assert log["no_claim"] == "none" and log["corpus_line"] is True
    assert ctx["no_claim_sections"] == "none" and ctx["corpus_line"] is True
    checks = {c["check"]: c["ok"] for c in M.structure_checks(text, {})}
    assert checks["sections"] and checks["no_claim_no_forecast"] and "solutions_table" not in checks


def test_with_no_institution_at_all_the_method_still_steps_aside():
    from src.rag import lawyer_method as M
    ctx = _no_claim_ctx(())
    ctx["candidates"] = M.thesis_candidates(ctx)
    plan = M.clean_method({}, ctx["candidates"])
    assert ctx["no_action"] and M.no_claim(ctx) == ""
    assert M.compose([("Norme applicate ai fatti", "Regola.")], ctx, plan) == ""
    assert M.method_refusal([(ctx, plan)])["no_action"] is True


def test_every_claim_barred_by_a_term_keeps_the_sections_with_the_honesty_line():
    from src.rag import lawyer_method as M
    recs = {"1168 c.c.": _rec("Azione di reintegrazione", _req(1, "lo spoglio violento o clandestino del possesso"))}
    case = "Tizio possedeva il cortile; il vicino Caio ha cambiato la serratura del cancello due anni fa."
    ctx = _civil(recs, case, terms=[{"id": "possessoria", "status": "DECORSO", "ref": "artt. 1168 e 1170 c.c."}])
    ctx["candidates"] = M.thesis_candidates(ctx)
    plan = M.clean_method({}, ctx["candidates"])
    assert ctx["no_action_kind"] == "expired" and M.no_claim(ctx) == "expired"       # no institution needed
    assert M.method_refusal([(ctx, plan)]) == {}
    text = M.compose([("Norme applicate ai fatti", "Regola.")], ctx, plan)
    secs = _sections(text)
    assert "preclusa dal termine decorso" in secs[M.title("sintesi")] and "1168" in secs[M.title("sintesi")]
    assert M.CORPUS_LINE.split(" su ")[0] not in text
    assert M.NO_CLAIM_EXPIRED_WHY in secs[M.title("soluzioni")]
    assert "- Art. 1168 c.c. (Azione di reintegrazione): azione preclusa dal termine decorso." in secs[M.title("soluzioni")]
    assert "Esito probabile" not in text and M.no_action_problems(text) == []


def test_the_corpus_line_fires_where_the_institution_has_no_module_and_nothing_of_its_family_was_graded():
    from src.rag import lawyer_method as M
    ctx = _civil({"1218 c.c.": _rec("Responsabilità del debitore", _req(1, "l'inadempimento"))},
                 "Tizio ha prestato a Caio una somma per un viaggio.", ["contratto di viaggio"])
    mods = {m["id"] for m in M.institution_modules()["modules"]}
    assert not set(M.read_families(ctx)) & mods and M.corpus_line(ctx) == (
        "Il corpus non contiene ancora informazioni sufficienti su contratto di viaggio: il parere si limita alle norme "
        "dei codici.")
    assert M.corpus_line({**ctx, "institutions": ["comodato"]}) == ""        # a module: never
    assert M.corpus_line({**ctx, "institutions": []}) == ""                  # no institution: never
    # a graded record of the institution's family (remedies.json's matter): never
    nul = _civil({"1418 c.c.": _rec("Cause di nullità del contratto", _req(1, "contrario a norme imperative"))},
                 "Tizio ha firmato con Caio una scrittura privata.", ["nullità del contratto"])
    assert M.read_families(nul) and M.corpus_line(nul) == ""
    # in §1 of a parere with a line, after the summary
    ctx["candidates"] = M.thesis_candidates(ctx)
    plan = M.clean_method({"valutazioni": {c["id"]: {"realismo": "alto"} for c in ctx["candidates"]}},
                          ctx["candidates"])
    log = {}
    text = M.compose([("Norme applicate ai fatti", "Regola.")], ctx, plan, log=log)
    summary = _sections(text)[M.title("sintesi")]
    assert "Linea principale" in summary and summary.endswith(M.corpus_line(ctx)) and log["corpus_line"] is True


def test_no_objections_call_with_no_line():
    from src.rag import lawyer_method as M
    ctx = _no_claim_ctx()
    ctx["candidates"] = M.thesis_candidates(ctx)
    plan = M.clean_method({}, [])

    def boom(*a, **k):
        raise AssertionError("no call with no line")
    assert M.objections_plan(ctx, plan, call=boom) is plan


def test_the_flow_keeps_the_sections_and_counts_them(monkeypatch):
    import test_v60g
    from test_v60f import _graded_state
    from src.rag import interactive_flow as flow, lawyer_method as M
    state = _graded_state(monkeypatch)
    state["mode"] = "single_shot"
    monkeypatch.setenv("LAWYER_METHOD", "true")
    line = M.NO_ACTION_LINE.format(istituto="servitù")

    def plan(ctx, call=None):           # the first read found the servitù; no norm confers an action on the client
        ctx["candidates"], ctx["no_action"], ctx["no_action_kind"] = [], line, "none"
        ctx["institutions"] = [{"name": "servitù", "rubricated": "", "claim": ""}]
        return M.clean_method({}, [])
    monkeypatch.setattr(M, "method_plan", plan)
    text, _ = test_v60g._answer(monkeypatch, state, {}, None, {})
    assert not state.get("method_refused") and "non è stato applicato" not in text
    assert "**1. Sintesi della soluzione**" in text and "**10. Punti da verificare**" in text
    assert "Il corpus non contiene ancora informazioni sufficienti su servitù" in text
    assert "Esito probabile" not in text and M.no_action_problems(text) == []
    counters = flow.case_counters(state)
    assert counters["no_claim_sections"] == 1 and counters["corpus_line"] == 1 and counters["no_action_prose"] == 0
    assert counters["articles_out_of_scope"] >= 0 and counters["closed_list_kept"] >= 0


# --- 6. role -------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "La società Alfa chiede il pagamento di 5.000 euro per la fornitura e minaccia di agire in giudizio.",
    "Alfa ha inviato una diffida per il pagamento di 7.200 euro e intende citare in giudizio.",
    "La banca chiede il rimborso del mutuo di 20.000 euro e vuole agire in giudizio.",
    "Alfa, con una lettera, chiede il saldo della fattura e vuole fare causa.",
])
def test_a_demanding_partys_suit_is_its_threat_never_the_clients_claim(text):
    from src.rag import intake
    words, spans = intake.party_threat(text)
    assert words and spans
    role, why, reading = intake.derive_role(text, "civile")
    assert role == "convenuto" and reading == ""


@pytest.mark.parametrize("text, role", [
    ("Tizio vuole agire in giudizio contro Alfa, che gli chiede il pagamento di una penale.", "attore"),
    ("Il cliente vuole citare in giudizio la banca, che gli ha chiesto il pagamento di 3.000 euro.", "attore"),
    ("Alfa vuole agire in giudizio contro Beta per il prezzo non pagato.", "attore"),
])
def test_the_clients_own_suit_or_a_party_demanding_nothing_stays_a_claim(text, role):
    from src.rag import intake
    assert intake.party_threat(text)[0] == ""
    assert intake.derive_role(text, "civile")[0] == role


def test_a_payment_demand_with_no_client_named_is_never_the_claimant_by_default():
    from src.rag import intake
    text = "Alfa chiede il pagamento di 5.000 euro a Beta."
    role, why, reading = intake.derive_role(text, "civile")
    assert (role, why, reading) == (intake.UNKNOWN, intake.UNKNOWN_DEMAND_WHY, "")
    # the request's task decides
    assert intake.derive_role(text + " Come possiamo contestare la richiesta?", "civile")[:2] == (
        "convenuto", intake.TASK_WHY["defending"])
    assert intake.derive_role(text + " Come recuperare la somma?", "civile")[0] == "attore"
    assert intake.task_side(text + " Come recuperare la somma?") == "claiming"
    assert intake.task_side(text + " Come possiamo contestare la richiesta?") == "defending"
    assert intake.task_side(text) == ""
    # a client named: the reading as before
    assert intake.derive_role(text, "civile", "Beta")[0] == "attore"
    # nothing demanded: the reading as before
    assert intake.derive_role("Alfa e Beta hanno firmato un contratto di fornitura.", "civile")[0] == "attore"


def test_resolve_leaves_the_role_unknown_with_neither_sides_wording(monkeypatch):
    from src.rag import interactive_flow as flow, intake, lawyer_method as M
    text = "Alfa chiede il pagamento di 5.000 euro a Beta."
    got = intake.resolve({"role": intake.UNKNOWN, "rito": "civile", "confidence": {}}, {flow.CASE_KEY: text})
    assert got["role"] == intake.UNKNOWN and got["role_reading"] == "" and flow.role_reading_line({"intake": got}) == ""
    assert flow.role_counters(got) == {"role_threat_not_claim": 0, "role_unknown_on_demand": 1}
    threat = intake.resolve({"role": intake.UNKNOWN, "rito": "civile", "confidence": {}},
                            {flow.CASE_KEY: "La società Alfa chiede il pagamento di 900 euro e minaccia di agire in "
                                            "giudizio."})
    assert threat["role"] == "convenuto"
    assert flow.role_counters(threat) == {"role_threat_not_claim": 1, "role_unknown_on_demand": 0}
    recs = {"2043 c.c.": _rec("Risarcimento per fatto illecito", _req(1, "il fatto doloso o colposo"))}
    cands = M.thesis_candidates(_civil(recs, text, posture=_posture("civile", intake.UNKNOWN)))
    assert not any(c["tesi"].startswith(("Fondare la domanda", "Domanda riconvenzionale")) for c in cands)


def test_detect_says_why_the_role_stayed_unknown(monkeypatch):
    from src.rag import intake
    monkeypatch.setattr(intake, "ask_model", lambda *a, **k: {"role": "sconosciuto", "rito": "civile",
                                                               "deliverable": "parere", "stage": "pre_filing"})
    got = intake.detect("Il 3 marzo 2026 Alfa ha consegnato la merce a Beta. Alfa chiede il pagamento di 5.000 euro a "
                        "Beta; Beta sostiene che la merce era difettosa. Redigi un parere.")
    assert got["role"] == intake.UNKNOWN and got["why"]["role"] == intake.UNKNOWN_DEMAND_WHY


# --- 10. counterparty, consumer scope, scope gate defaults, ATP, cells -------------------------------------------------

@pytest.mark.parametrize("case, professional", [
    ("Tizio ha comprato una lavatrice da Alfa. Alfa è una ditta di elettrodomestici.", True),
    ("Tizio si è rivolto ad Alfa S.p.A. per la fornitura.", True),
    ("La società Beta ha installato la caldaia di Tizio.", True),
    ("Tizio ha incaricato un professionista per il progetto.", True),
    ("L'azienda Gamma ha consegnato un mobile rotto.", True),
    ("Tizio sostiene che Alfa sia un'impresa; ha comprato da Alfa una lavatrice.", False),
    ("Tizio ha comprato da un venditore privato la lavatrice della ditta Alfa.", False),
    ("Tizio ha comprato una lavatrice usata da Caio.", False),
    ("Tizio ha urtato l'auto aziendale di Caio durante la dittatura del traffico.", False),     # whole words only
])
def test_the_qualifier_words_make_the_counterparty_professional(case, professional):
    from src.rag import lawyer_method as M
    assert M.professional_counterparty({"case": case, "certain_text": ""}) is professional


@pytest.mark.parametrize("number, sale, out", [
    ("33", False, ""), ("1", False, ""), ("38", False, ""), ("141", False, ""), ("144-bis", False, ""),
    ("128", True, ""), ("135-bis", True, ""), ("130", False, "sale"), ("135-ter", False, "sale"), ("52", False, ""),
])
def test_the_consumer_codes_scope(number, sale, out):
    from src.rag import lawyer_method as M
    assert M.consumer_scope(number, sale) == (M.CONSUMER_SALE_ONLY if out else "")


def test_the_family_gate_keeps_the_sale_block_out_of_a_service_contract():
    from src.rag import family_gate as F
    service = F.context("La società Alfa ha installato la caldaia di Tizio, che ora perde acqua.")
    sale = F.context("Tizio ha acquistato dalla ditta Alfa una lavatrice che non centrifuga.")
    assert service["professional"] and not service["sale"] and sale["professional"] and sale["sale"]
    assert F.label_admitted("33 d.lgs. 206/2005", service)[0] is True
    assert F.label_admitted("130 d.lgs. 206/2005", service)[0] is False
    assert F.label_admitted("130 d.lgs. 206/2005", sale)[0] is True
    rows = [{"_article": ("130", "d.lgs. 206/2005"), "d": {"name": "Codice del Consumo 2026"}},
            {"_article": ("141", "d.lgs. 206/2005"), "d": {"name": "Codice del Consumo 2026"}}]
    kept, dropped, _ = F.gate(rows, service, set())
    assert [r["_article"][0] for r in kept] == ["141"]
    from src.rag import interactive_flow as flow, lawyer_method as M
    assert dropped[0]["article"].startswith("130") and dropped[0]["why"] == M.CONSUMER_SALE_ONLY
    counters = flow.case_counters({"family_gate": {"dropped": dropped, "professional": True, "qualifier": "società"}})
    assert counters["consumer_sale_block_out"] == 1 and counters["counterparty_professional"] == 1
    assert counters["counterparty_by_qualifier"] == 1


def test_a_sale_block_claim_is_refused_outside_a_sale():
    from src.rag import lawyer_method as M
    rule, group, _ = M.claim_admission("130 d.lgs. 206/2005", [], consumer=True, sale=False)
    assert rule == "" and group == M.CLAIM_REFUSED["consumer_sale"]
    assert M.claim_admission("130 d.lgs. 206/2005", [], consumer=True, sale=True)[1] != M.CLAIM_REFUSED["consumer_sale"]


def test_the_scope_gate_admits_the_general_clauses_and_the_stage_articles_by_default():
    from src.rag import topic_groups as T
    defaults = T.default_admitted()
    for label in ("1175 c.c.", "1176 c.c.", "1218 c.c.", "1223 c.c.", "2043 c.c.", "2056 c.c."):
        assert defaults[label] == T.GENERAL_CLAUSE_WHY
    for label in ("696 c.p.c.", "696-bis c.p.c.", "700 c.p.c.", "585 c.p.p."):
        assert defaults[label] == T.STAGE_PROCEDURE_WHY
    # the general clauses and the procedural codes are never gated (topic_groups.CODE_GENERAL, no unit): the defaults
    # keep a gated unit's article that is one of them -- here a stand-in default for a c.p. article of another Capo
    rows = [{"_article": ("624", "c.p."), "s": {"titolo_articolo": "Furto"}},
            {"_article": ("640", "c.p."), "s": {"titolo_articolo": "Truffa"}},
            {"_article": ("646", "c.p."), "s": {"titolo_articolo": "Appropriazione indebita"}},
            {"_article": ("1176", "c.c."), "s": {"titolo_articolo": "Diligenza nell'adempimento"}}]
    log = {}
    kept = T.code_scope_gate(rows, ["furto"], log=log, defaults={"640 c.p.": "prova"})
    assert [r["_article"][0] for r in kept] == ["624", "640", "1176"]
    assert log["admitted_default"] == {"640 c.p.": "prova"} and [d["article"] for d in log["dropped"]] == ["646 c.p."]
    from src.rag import interactive_flow as flow
    counters = flow.case_counters({"code_scope": log})
    assert counters["code_articles_out_of_scope"] == 1 and counters["code_articles_admitted_default"] == 1
    log = {}
    T.code_scope_gate(rows, ["furto"], log=log)                       # the real defaults: none gated here
    assert {d["article"] for d in log["dropped"]} == {"640 c.p.", "646 c.p."} and log["admitted_default"] == {}
    # the compose scope (article_scope, the output check's) holds them by default
    from src.rag import lawyer_method as M
    scope = M.article_scope({"records": {}, "case": ""})
    assert {"1176 c.c.", "2056 c.c.", "696 c.p.c.", "700 c.p.c."} <= scope


def test_the_atp_line_stays_in_section_8_of_the_civil_attore_cell():
    from src.rag import lawyer_method as M, render_computed as RC
    cell = next(c for c in _script("nightly_sim").cells() if c["id"] == "civile_attore_pre_filing")
    posture = _posture("civile", "attore")
    rendered = "\n\n".join(f"**{t}**\n\n{b}" for t, b in RC.stage_blocks(posture))
    assert "696" in rendered
    recs = {"1667 c.c.": _rec("Difformità e vizi dell'opera", _req(1, "l'opera presenta vizi"))}
    ctx = _civil(recs, cell["text"], ["appalto"])
    ctx["candidates"] = M.thesis_candidates(ctx)
    plan = M.clean_method({"valutazioni": {c["id"]: {"realismo": "alto"} for c in ctx["candidates"]}},
                          ctx["candidates"])
    log = {}
    text = M.compose([("Norme applicate ai fatti", "Regola."), ("Profili processuali", rendered)], ctx, plan, log=log)
    proc = _sections(text)[M.title("processuali")]
    assert "accertamento tecnico preventivo (art. 696 c.p.c.)" in proc and "(art. 700 c.p.c.)" in proc
    assert not any("696" in x["sentence"] for x in log["rule_statements_removed"])
    assert ctx["closed_list_kept"] >= 2 and len(log["closed_list_kept"]) == ctx["closed_list_kept"]
    checks = {c["check"]: c for c in M.structure_checks(text, cell["cell"])}
    assert checks["pre_filing:atp"]["ok"] and checks["pre_filing:urgenza_700"]["ok"]
    assert not [s for s in M.output_check(text, ctx["output_sources"])["unsourced"] if "696" in s or "700" in s]
    # a writer's rule sentence on 696 with no source is still removed: only the closed list's lines are kept
    from src.rag import rule_statements
    written = "L'accertamento tecnico preventivo richiede l'urgenza di fissare lo stato dei luoghi (art. 696 c.p.c.)."
    assert rule_statements.check(written, {"elements": {}, "store": {}, "passages": {}}, [], M.stage_lines(posture)) == ""


def test_the_cells_two_penal_at_the_indagini_and_the_two_turn_cell():
    nightly = _script("nightly_sim")
    cells = {c["id"]: c for c in nightly.cells()}
    assert cells["penale_imputato_primo_grado"]["cell"]["stage"] == "indagini"
    assert cells["penale_offesa_pre_filing"]["cell"]["stage"] == "indagini"
    two = [c for c in cells.values() if c.get("followup")]
    assert [c["id"] for c in two] == ["civile_attore_followup"] and two[0]["followup"] == "approfondisci il punto 1"
    assert two[0]["cell"] == {"family": "civile", "rito": "civile", "role": "attore", "stage": "pre_filing"}


PARERE = """**1. Sintesi della soluzione**

Linea principale: garanzia per i vizi (art. 1490 c.c.).

**4. Norme e analisi per elementi**

Art. 1490 c.c., art. 1495 c.c.

**10. Punti da verificare**

1. La data della denuncia dei vizi al venditore, da cui decorre il termine di otto giorni.
2. Il prezzo pagato e lo scontrino di acquisto della lavatrice."""


@pytest.mark.parametrize("second, problems", [
    ("La denuncia dei vizi va fatta entro otto giorni dalla scoperta (art. 1495 c.c.): la data della segnalazione al "
     "venditore decide se il termine è rispettato.", []),
    ("Lo scontrino prova il prezzo pagato e la data di acquisto della lavatrice.", ["fuori tema", "anche il punto 2"]),
    ("La denuncia dei vizi al venditore decorre dalla scoperta; anche il prezzo pagato, lo scontrino di acquisto della "
     "lavatrice contano.", ["anche il punto 2"]),
    ("La denuncia dei vizi al venditore entro otto giorni; vale anche l'art. 2043 c.c.", ["articoli non richiesti"]),
    ("**1. Sintesi**\n\nLa denuncia dei vizi al venditore.\n\n**2. Fatti**\n\nx\n\n**3. Questione**\n\ny",
     ["il parere intero di nuovo"]),
])
def test_followup_scope_on_synthetic_answers(second, problems):
    from src.rag import lawyer_method as M
    got = M.followup_scope(PARERE, second, "approfondisci il punto 1")
    assert [p.split(":")[0].split(" (")[0] for p in got] == problems, got


def test_followup_scope_records_and_a_missing_point():
    from src.rag import lawyer_method as M
    second = "La data della denuncia dei vizi al venditore fa decorrere il termine di otto giorni."
    assert M.followup_scope(PARERE, second, "approfondisci il punto 1", ["1490 c.c.", "1495 c.c."],
                            ["1490 c.c.", "1495 c.c."]) == []
    assert M.followup_scope(PARERE, second, "approfondisci il punto 1", ["1490 c.c.", "1495 c.c."],
                            ["1490 c.c."]) == ["schede perse: 1495 c.c."]
    assert M.followup_scope(PARERE, second, "approfondisci il punto 7")[0].startswith("punto 7 assente")
    nightly = _script("nightly_sim")
    assert nightly.followup_check(PARERE, second, "approfondisci il punto 1") == {
        "check": "followup_scope", "ok": True, "why": ""}
    assert nightly.followup_check(PARERE, "", "approfondisci il punto 1")["ok"] is False


def test_the_offline_two_turn_cell_reads_its_followup_file(tmp_path):
    nightly = _script("nightly_sim")
    spec = next(c for c in nightly.cells() if c.get("followup"))
    (tmp_path / f"{spec['id']}.md").write_text(PARERE, encoding="utf-8")
    (tmp_path / f"{spec['id']}.followup.md").write_text(
        "La data della denuncia dei vizi al venditore fa decorrere il termine di otto giorni.", encoding="utf-8")
    out = nightly.offline_cell(spec, tmp_path)
    check = next(c for c in out["checks"] if c["check"] == "followup_scope")
    assert check["ok"] is True


def test_the_live_two_turn_cell_runs_the_followup(monkeypatch):
    nightly = _script("nightly_sim")
    from src.rag import interactive_flow as flow
    state = {"records": {"1490 c.c.": {}, "1495 c.c.": {}}}
    monkeypatch.setattr(flow, "follow_up", lambda message, st: {
        "answer": "La data della denuncia dei vizi al venditore fa decorrere il termine di otto giorni.", "state": st})
    out = {"checks": []}
    nightly.run_followup({"followup": "approfondisci il punto 1"}, PARERE, state, out)
    assert out["checks"] == [{"check": "followup_scope", "ok": True, "why": ""}] and out["followup_answer"]


# --- 11. main_outside_institution -----------------------------------------------------------------------------------------

PRELIMINARE = "Tizio ha firmato con Caio un contratto preliminare di vendita di un appartamento; Caio non si presenta al rogito."


def test_1453_on_a_preliminare_is_accepted_when_no_module_line_is_proven():
    from src.rag import lawyer_method as M
    recs = {"1453 c.c.": _rec("Risolubilità del contratto per inadempimento", _req(1, "l'inadempimento di un contraente"))}
    ctx = _civil(recs, PRELIMINARE, ["contratto preliminare"])
    ctx["candidates"] = M.thesis_candidates(ctx)
    plan = M.clean_method({"valutazioni": {c["id"]: {"realismo": "alto"} for c in ctx["candidates"]}},
                          ctx["candidates"])
    counts = M.selection_counts(ctx, plan, {})
    main = next(c for c in ctx["candidates"] if c["id"] == counts["chosen"])
    assert main["article"] == "1453 c.c." and M.general_obligations("1453 c.c.")
    assert M.module_proven_line(ctx) is False and counts["main_outside_institution"] is False


def test_a_general_line_leading_while_the_module_has_a_proven_line_is_still_counted():
    from src.rag import lawyer_method as M
    case = "Tizio ha affidato all'impresa Alfa il rifacimento del tetto. L'opera è stata consegnata con infiltrazioni."
    recs = {"1218 c.c.": _rec("Responsabilità del debitore", _req(1, "l'inadempimento della prestazione")),
            "1667 c.c.": _rec("Difformità e vizi dell'opera", _req(3, "l'opera presenta vizi"))}
    ctx = _civil(recs, case, ["appalto"])
    ctx["candidates"] = M.thesis_candidates(ctx)
    assert M.module_proven_line(ctx) is True
    # the general line (B2 subordinates it here) as the main line all the same: outside, counted
    ctx["candidates"].append({"id": "T9", "kind": "claim", "article": "1218 c.c.", "poggia_su": ["R1"],
                              "admitted_by": "norma generale delle obbligazioni"})
    assert M.main_outside(ctx, {"candidate": "T9"}) is True


# --- review fixes (79.12.5) ---------------------------------------------------------------------------------------------

# 10: a business word that names the client's own business, its employer, a negated qualification or a question is
# no professional counterparty
@pytest.mark.parametrize("case", [
    "Assisto la società Beta S.r.l., che ha acquistato da Mario Rossi, suo vicino di casa, un'auto usata difettosa.",
    "Il mio cliente, titolare di una ditta di traslochi, ha comprato un divano dal signor Verdi, che si è rotto.",
    "Il signor Neri lavora come dipendente presso un'azienda di trasporti; ha comprato una bicicletta dal vicino Gialli.",
    "Ho comprato una bici usata da Luca? Era una ditta?",
    "Ho comprato una bici usata da Luca. Luca non è una società né una ditta, è un pensionato.",
    "Sono titolare di una ditta individuale. Ho stipulato con Mario Rossi un contratto per la ristrutturazione.",
    "La mia ditta ha comprato un furgone usato dal signor Bianchi.",
])
def test_review_a_business_word_not_about_the_counterparty_makes_none(case):
    from src.rag import lawyer_method as M
    assert M.counterparty_qualifier({"case": case, "certain_text": ""}) == ""
    assert M.professional_counterparty({"case": case}) is False


@pytest.mark.parametrize("case", [
    "Il mio cliente ha acquistato un divano dalla ditta Alfa, che si è rotto.",
    "Alfa S.r.l. ha venduto al signor Neri una lavatrice difettosa.",
    "Il signor Neri ha affidato i lavori alla ditta Alfa.",
    "Assisto il signor Neri contro la società Alfa, che gli ha venduto un'auto difettosa.",
])
def test_review_the_counterpartys_business_word_still_counts(case):
    from src.rag import lawyer_method as M
    assert M.counterparty_qualifier({"case": case, "certain_text": ""}) and M.professional_counterparty({"case": case})


# 10: a service or a supply told with a sale's word is no sale; a purchase told with other words is one
@pytest.mark.parametrize("case, sale", [
    ("Il cliente ha acquistato un abbonamento in palestra presso Fit S.r.l., che ha chiuso senza rimborso.", False),
    ("Il cliente ha un contratto di fornitura di energia; il venditore di energia ha fatturato consumi non dovuti.", False),
    ("Il cliente ha affidato alla ditta Alfa la riparazione della caldaia.", False),
    ("Ho ordinato online un frigorifero dalla ditta Alfa S.r.l.; è arrivato difettoso e non funziona.", True),
    ("Ho preso uno smartphone nel negozio di Beta S.p.A.: dopo un mese lo schermo si è rotto.", True),
    ("Ho pagato 900 euro alla ditta Gamma per una lavatrice, consegnata con un difetto.", True),
    ("Tizio ha acquistato dalla ditta Alfa una lavatrice che non centrifuga.", True),
])
def test_review_a_sale_is_read_by_its_object(case, sale):
    from src.rag import lawyer_method as M
    assert M.sale_contract(case) is sale


def test_review_a_consumer_sale_told_as_an_order_keeps_its_guarantee_action():
    from src.rag import lawyer_method as M
    mods = [m for m in M.institution_modules()["modules"] if m["id"] == "consumo"]
    case = "Ho ordinato online un frigorifero dalla ditta Alfa S.r.l.; è arrivato difettoso e non funziona."
    rule, group, _ = M.claim_admission("135-bis Codice del Consumo 2026", mods, True, M.sale_contract(case))
    assert (rule, group) == ("modulo consumo", "module")


@pytest.mark.parametrize("number, out", [
    ("135-septies", "sale"), ("135-bis Codice del Consumo 2026", "sale"), ("135 d.lgs. 206/2005", "sale"),
    ("135-octies", ""), ("135-novies", ""), ("135-decies", ""), ("135-vicies ter", ""), ("135-duodecies", ""),
])
def test_review_capo_i_bis_digital_content_is_no_sale_block(number, out):
    from src.rag import lawyer_method as M
    assert M.consumer_scope(number, False) == (M.CONSUMER_SALE_ONLY if out else "")


# 6: a company's form never ends the clause, and a leading adverb is passed over for the party that demands
@pytest.mark.parametrize("text", [
    "Beta S.p.A. pretende il pagamento di 3.000 euro e ha minacciato di agire in giudizio.",
    "Alfa S.r.l. mi chiede il pagamento di 5.000 euro e minaccia di citarmi in giudizio. Cosa posso fare?",
    "Alfa s.n.c. chiede il saldo di 2.000 euro e minaccia le vie legali.",
    "Successivamente Beta ha chiesto il pagamento di 3.000 euro e ha minacciato di agire in giudizio.",
])
def test_review_a_companys_threat_with_its_form_or_after_an_adverb(text):
    from src.rag import intake
    words, spans = intake.party_threat(text)
    assert words and spans
    assert intake.derive_role(text, "civile")[0] == "convenuto"


def test_review_a_company_form_keeps_the_sentence_end_before_a_capital():
    from src.rag import intake
    blanked = intake._company_blanked("Ho venduto merce ad Alfa S.r.l. Alfa non paga.")
    assert len(blanked) == len("Ho venduto merce ad Alfa S.r.l. Alfa non paga.") and blanked.count(".") == 2
    assert intake.derive_role("Ho venduto merce ad Alfa S.r.l. Alfa non paga. Voglio agire in giudizio per "
                              "recuperare il credito.", "civile")[0] == "attore"


# 11: a module that yields only an open claim proves nothing -- 1453 leading there is no line outside
def test_review_an_open_module_claim_is_no_proven_line():
    from src.rag import lawyer_method as M
    open_claim = {"id": "c1", "kind": "claim", "article": "1665 c.c.", "admitted_by": "modulo appalto",
                  "poggia_su": ["E2", "E3"], "tesi": M.CANDIDATE_TEXTS["claim_open"].format(
                      art="1665 c.c.", rubrica="", open="«x» (E2)")}
    gen = {"id": "c2", "kind": "claim", "article": "1453 c.c.", "poggia_su": ["E1"],
           "admitted_by": "norma generale delle obbligazioni"}
    ctx = {"candidates": [open_claim, gen], "claim_modules": ["appalto"]}
    assert M.module_proven_line(ctx) is False and M.main_outside(ctx, {"candidate": "c2"}) is False
    proven = {**open_claim, "tesi": M.CANDIDATE_TEXTS["claim"].format(art="1665 c.c.", rubrica="")}
    assert M.module_proven_line({"candidates": [proven, gen]}) is True
    # thesis_candidates' own reading wins when it ran
    assert M.module_proven_line({"candidates": [proven, gen], "module_proven": False}) is False


def test_review_thesis_candidates_records_whether_the_module_proved_a_line():
    from src.rag import lawyer_method as M
    case = "Tizio ha affidato all'impresa Alfa il rifacimento del tetto. L'opera è stata consegnata con infiltrazioni."
    recs = {"1218 c.c.": _rec("Responsabilità del debitore", _req(1, "l'inadempimento della prestazione")),
            "1667 c.c.": _rec("Difformità e vizi dell'opera", _req(3, "l'opera presenta vizi", status="da provare"))}
    ctx = _civil(recs, case, ["appalto"])
    ctx["candidates"] = M.thesis_candidates(ctx)
    assert ctx["module_proven"] is False and M.module_proven_line(ctx) is False
