"""v79.12.4 (items 1, 3, 4, 5, 7 of the brief), on synthetic sentences, rows and records only:

1. the victim's charge (text path): for the persona offesa / parte civile the charge is the querela's offence and only
   it -- a civil article (2043 c.c.) is a remedy under it, never the charge; §5 (rulings_for_record) fetches for the
   offence. Counters charge_victim_remedies, victim_charge_civil (0).
3. the charge choice: choose_charge applies event_mismatch to the first read's hypotheses (590 never the charge where
   the victim died); among several charged articles the one with the highest maximum pena from the store sets the
   exposure and the riti, never the order; every hypothesis decomposed so the rule scores it. Counters
   charge_event_mismatch (0), charge_event_refused, charged_undecomposed (0).
4. a fragment never leads: escluso_fragment guards contest_open too; fragment_kind reads a clause opening on
   «ovvero / oppure» and a definition as non-elements; an event the certain facts narrate is never «da provare»; a
   contest of a threat record that is not the charge ranks after the charge's and never leads; the containment line
   counts as eligible. Counters main_fragment (0), main_other_threat (0), event_narrated_proven,
   event_narrated_unproven (0).
5. arts. 40 and 43 c.p. are never defence lines; a scriminante or defence with fewer than half its elements proven never
   leads while another line is eligible. Counters defence_lines_40_43 (0), main_weak_defence (0).
7. a claim on an article whose action a computed term marks DECORSO is refused (decadenza: CLAIM_REFUSED «termine
   decorso») or kept minoritaria (prescrizione); the terms table names the articles each term bars. Counters
   claims_refused_expired, claims_expired_minority, claims_expired_sustainable (0).
"""

import os

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

from src.rag import fact_signals  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import issue_records  # noqa: E402
from src.rag import lawyer_method as M  # noqa: E402
from src.rag import posture, rule_quote, rulings  # noqa: E402


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))


def _p(family, role, stage="primo_grado"):
    return {"rito": family, "family": family, "role": role, "stage": stage,
            "entry": posture.entry_for(family, role, stage)}


IMPUTATO = _p("penale", "imputato")
OFFESA = _p("penale", "persona_offesa", "indagini")
ATTORE = _p("civile", "attore", "pre_filing")


def _req(i, text, status="provato"):
    return {"id": f"R{i}", "comma": 1, "kind": "requisito", "text": text, "status": status}


def _rec(rubrica, *reqs, side="weapon", **extra):
    return {"rubrica": rubrica, "side": side, "requisites": list(reqs), **extra}


def _row(number, code, inst="", rubrica="", pena=None):
    row = {"_article": (number, code), "_first_read": inst, "d": {"id": code.upper()}, "s": {
        "name": number, "plain_text": "testo", "titolo_articolo": rubrica}}
    if pena is not None:
        row["_proc"] = {"version": 4, "pena": pena, "querela": False}
    return row


def _sent(i, text, certainty="accertato"):
    return {"id": f"S{i}", "text": text, "source": "caso", "certainty": certainty}


# --- 1. the victim's charge -------------------------------------------------------------------------------------------------

VICTIM_CASE = ("Beta ha sporto querela contro Alfa per lesione personale (art. 582 c.p.) e chiede il risarcimento dei "
               "danni ai sensi dell'art. 2043 c.c.")


def _victim_state(monkeypatch, case=VICTIM_CASE, hypotheses=()):
    monkeypatch.setattr(flow, "case_text", lambda s: case)
    monkeypatch.setattr(posture, "alternative_rows", lambda s, rows=(): [])
    return {"read": {"hypotheses": list(hypotheses)},
            "rows": [_row("582", "c.p.", "lesioni", "Lesione personale"),
                     _row("2043", "c.c.", "risarcimento", "Risarcimento per fatto illecito")],
            "reserve": [],
            "records": {"582 c.p.": _rec("Lesione personale", _req(1, "cagiona ad alcuno una lesione personale"),
                                         _req(2, "dalla quale deriva una malattia nel corpo o nella mente", "da provare")),
                        "2043 c.c.": _rec("Risarcimento per fatto illecito", _req(3, "un fatto doloso o colposo"),
                                          _req(4, "un danno ingiusto"))}}


def test_a_victim_cell_yields_the_offence_as_the_charge_never_2043(monkeypatch):
    state = _victim_state(monkeypatch)
    out = posture.mark_sides(state, OFFESA)
    assert state["charge"]["article"] == "582 c.p." and out["charged"] == ["582 c.p."]
    rec = state["records"]["2043 c.c."]
    assert not rec.get("charged") and rec["remedy_of"] == "582 c.p."          # a remedy under the offence
    assert out["remedies"] == ["2043 c.c."] and state["block_counters"]["charge_victim_remedies"] == 1
    a = M.assessment(state["records"], OFFESA)
    assert a["article"] == "582 c.p."
    # the same cell with 590 named: 590, never 2043
    case = VICTIM_CASE.replace("lesione personale (art. 582 c.p.)", "lesioni colpose (art. 590 c.p.)")
    state = _victim_state(monkeypatch, case)
    state["records"]["590 c.p."] = state["records"].pop("582 c.p.")
    state["records"]["590 c.p."]["rubrica"] = "Lesioni personali colpose"
    posture.mark_sides(state, OFFESA)
    assert state["charge"]["article"] == "590 c.p."


def test_a_victim_with_only_a_civil_article_named_takes_the_reads_offence(monkeypatch):
    case = "Beta chiede il risarcimento dei danni ex art. 2043 c.c. ad Alfa che lo ha colpito al volto."
    state = _victim_state(monkeypatch, case, [{"line": "principale", "claim": "lesioni", "institutions": ["lesioni"]},
                                              {"line": "subordinata", "claim": "danno", "institutions": ["risarcimento"]}])
    posture.mark_sides(state, OFFESA)
    assert state["charge"] == {"article": "582 c.p.", "rule": "read_hypothesis"}
    assert not state["records"]["2043 c.c."].get("charged")


def test_the_victims_claim_line_is_on_the_offence_even_with_the_civil_article_proven():
    records = {"2043 c.c.": _rec("Risarcimento per fatto illecito", _req(3, "un fatto doloso o colposo"),
                                 _req(4, "un danno ingiusto")),
               "582 c.p.": _rec("Lesione personale", _req(1, "cagiona ad alcuno una lesione personale"),
                                _req(2, "dalla quale deriva una malattia", "da provare"))}
    ctx = {"records": records, "circumstances": {}, "posture": OFFESA, "case": "Alfa colpisce Beta.",
           "assessment": M.assessment(records, OFFESA), "client": "Beta", "institutions": []}
    assert ctx["assessment"]["article"] == "582 c.p."
    cands = M.thesis_candidates(ctx)
    claim = next(c for c in cands if c["kind"] == "claim_offence")
    assert claim["article"] == "582 c.p."
    assert all(c.get("article") != "2043 c.c." for c in cands)


def test_no_pertinence_promotion_replaces_the_victims_line():
    sols = [{"tesi": "Querela", "kind": "claim_offence", "candidate": "T1", "realismo": "alto"},
            {"tesi": "Domanda civile", "kind": "claim", "candidate": "T2", "realismo": "medio", "article": "2051 c.c."}]
    ctx = {"posture": OFFESA, "candidates": [{"id": "T1", "kind": "claim_offence", "article": "582 c.p."},
                                             {"id": "T2", "kind": "claim", "article": "2051 c.c."}],
           "institutions": ["custodia"], "records": {}}
    assert M.promote_by_pertinence(sols, 0, [0, 1], {}, ctx) == 0


class _Session:
    """rulings.FETCH_QUERY per (doc, num): records each fetch."""

    def __init__(self):
        self.fetched = []

    def run(self, query, **kw):
        if query == rulings.FETCH_QUERY:
            self.fetched.append((kw["doc"], kw["num"]))

        class R:
            def data(self_inner):
                return []
        return R()


def test_section_5_fetches_the_rulings_for_the_offence(monkeypatch):
    state = _victim_state(monkeypatch)
    posture.mark_sides(state, OFFESA)
    records = state["records"]
    a = M.assessment(records, OFFESA)
    ctx = {"records": records, "case": VICTIM_CASE, "assessment": a,
           "candidates": [{"id": "T1", "kind": "claim_offence", "contesta": [], "poggia_su": ["R1"]}]}
    session = _Session()
    record = rulings.build(ctx, state["rows"], session, call=lambda m, t: {})
    assert ("C.P.", "582") in session.fetched                     # §5 fetches for the offence
    assert "582 c.p." in record["log"]["articles"] and a["article"] == "582 c.p."


def test_a_victims_civil_article_is_decomposed_as_a_remedy_never_decisive(monkeypatch):
    state = {"records": {}, "analysis": {}, "rows": [], "read": {"hypotheses": []}}
    monkeypatch.setattr(flow, "case_text", lambda s: VICTIM_CASE)
    monkeypatch.setattr(flow, "_restored_analysis", lambda a: {})
    monkeypatch.setattr(flow, "user_named", lambda s: [])
    monkeypatch.setattr(posture, "alternative_labels", lambda s, rows=(): [])
    monkeypatch.setattr(issue_records, "_requisites", lambda r, a, *_: [
        {"id": "", "comma": 1, "kind": "requisito", "text": "un elemento", "status": "da provare"}])
    monkeypatch.setattr(issue_records, "refresh", lambda *a, **k: None)
    row = _row("582", "c.p.", rubrica="Lesione personale")
    flow.decompose_records(state, OFFESA, [], set(), fetch=lambda keys: [row])      # 2043 unread
    assert state["records"]["582 c.p."]["decomposed_for"] == flow.DECOMPOSE_WHY["charge"]
    assert "method_refused" not in state                     # the remedy unread never refuses the method


# --- 3. the charge choice -----------------------------------------------------------------------------------------------------

def _death_state(monkeypatch, case, hypotheses):
    monkeypatch.setattr(flow, "case_text", lambda s: case)
    monkeypatch.setattr(posture, "alternative_rows", lambda s, rows=(): [])
    return {"read": {"hypotheses": hypotheses},
            "rows": [_row("590", "c.p.", "lesioni colpose", "Lesioni personali colpose"),
                     _row("589", "c.p.", "omicidio colposo", "Omicidio colposo")],
            "reserve": [],
            "records": {"590 c.p.": _rec("Lesioni personali colpose", _req(1, "cagiona per colpa una lesione"),
                                         _req(2, "la lesione personale"), _req(3, "per colpa")),
                        "589 c.p.": _rec("Omicidio colposo", _req(4, "cagiona per colpa la morte", "da provare"),
                                         _req(5, "per colpa"))}}


HYPS = [{"line": "principale", "claim": "lesioni", "institutions": ["lesioni colpose"]},
        {"line": "subordinata", "claim": "omicidio", "institutions": ["omicidio colposo"]}]


def test_590_is_never_the_charge_where_the_victim_died(monkeypatch):
    case = "Alfa, alla guida, investe Beta. Beta muore in ospedale due giorni dopo."
    state = _death_state(monkeypatch, case, HYPS)
    out = posture.mark_sides(state, IMPUTATO)
    # 590 holds more elements integrato, but the certain facts narrate a death: 589 is the charge
    assert state["charge"] == {"article": "589 c.p.", "rule": "read_hypothesis"}
    assert out["event_refused"] == ["590 c.p."] and not state["records"]["590 c.p."].get("charged")
    counters = state["block_counters"]
    assert counters["charge_event_refused"] == 1 and counters["charge_event_mismatch"] == 0


def test_without_a_death_the_most_integrato_hypothesis_still_wins(monkeypatch):
    state = _death_state(monkeypatch, "Alfa, alla guida, investe Beta, che si frattura il braccio.", HYPS)
    posture.mark_sides(state, IMPUTATO)
    assert state["charge"]["article"] == "590 c.p."
    assert state["block_counters"]["charge_event_refused"] == 1          # 589 does not fit injuries without a death


def test_the_certain_facts_decide_the_event_never_an_allegation(monkeypatch):
    state = _death_state(monkeypatch, "Alfa investe Beta.", HYPS)
    state["sentences"] = [_sent(1, "Alfa investe Beta."), _sent(2, "Forse Beta è morto.", "ipotizzato")]
    posture.mark_sides(state, IMPUTATO)
    assert state["charge"]["article"] == "590 c.p."


def _two_charges(monkeypatch, case):
    monkeypatch.setattr(flow, "case_text", lambda s: case)
    monkeypatch.setattr(posture, "alternative_rows", lambda s, rows=(): [])
    rows = [_row("582", "c.p.", rubrica="Lesione personale", pena="reclusione da sei mesi a tre anni (co. 1)"),
            _row("572", "c.p.", rubrica="Maltrattamenti contro familiari e conviventi",
                 pena="reclusione da tre a sette anni (co. 1)")]
    return {"read": {"hypotheses": []}, "rows": rows, "reserve": [],
            "records": {"582 c.p.": _rec("Lesione personale", _req(1, "cagiona ad alcuno una lesione personale")),
                        "572 c.p.": _rec("Maltrattamenti contro familiari e conviventi",
                                         _req(2, "maltratta una persona della famiglia", "da provare"))}}


@pytest.mark.parametrize("case", ["Il PM contesta ad Alfa gli artt. 582 e 572 c.p.",
                                  "Il PM contesta ad Alfa gli artt. 572 e 582 c.p."])
def test_the_highest_maximum_pena_sets_exposure_and_rites_never_the_order(monkeypatch, case):
    state = _two_charges(monkeypatch, case)
    out = posture.mark_sides(state, IMPUTATO)
    assert out["exposure"] == "572 c.p." and state["records"]["572 c.p."]["exposure"] is True
    assert M.main_threat(state["records"]) == "572 c.p."
    assert M.assessment(state["records"], IMPUTATO)["article"] == "572 c.p."
    rites = M.charge_rites(state["records"], state["rows"], IMPUTATO)
    assert rites and all(x["charge"] == "572 c.p." for x in rites)
    assert out["charged_undecomposed"] == [] and state["block_counters"]["charged_undecomposed"] == 0


def test_heaviest_ranks_the_ergastolo_first_and_an_unknown_pena_last():
    rows = [_row("582", "c.p.", pena="reclusione da sei mesi a tre anni"), _row("575", "c.p.", pena="ergastolo"),
            _row("612", "c.p.")]
    assert posture.heaviest(["612 c.p.", "582 c.p.", "575 c.p."], rows) == "575 c.p."
    assert posture.heaviest(["612 c.p.", "582 c.p."], rows) == "582 c.p."
    assert posture.heaviest(["612 c.p.", "2043 c.c."], rows) == "612 c.p."        # none known: the order
    assert posture.max_pena("575 c.p.", rows) == float("inf") and posture.max_pena("612 c.p.", rows) is None


def test_every_read_hypothesis_is_decomposed_but_one_the_event_does_not_fit(monkeypatch):
    case = "Alfa investe Beta con l'auto. Beta muore in ospedale."
    state = {"records": {}, "analysis": {}, "rows": [], "reserve": [], "read": {"hypotheses": HYPS + [
        {"line": "subordinata", "claim": "omissione", "institutions": ["omissione di soccorso"]}]}}
    rows = [_row("590", "c.p.", "lesioni colpose", "Lesioni personali colpose"),
            _row("589", "c.p.", "omicidio colposo", "Omicidio colposo"),
            _row("593", "c.p.", "omissione di soccorso", "Omissione di soccorso")]
    monkeypatch.setattr(flow, "case_text", lambda s: case)
    monkeypatch.setattr(flow, "_restored_analysis", lambda a: {})
    monkeypatch.setattr(flow, "user_named", lambda s: [])
    monkeypatch.setattr(posture, "alternative_labels", lambda s, rows=(): [])
    monkeypatch.setattr(issue_records, "_requisites", lambda r, a, *_: [
        {"id": "", "comma": 1, "kind": "requisito", "text": "un elemento", "status": "da provare"}])
    monkeypatch.setattr(issue_records, "refresh", lambda *a, **k: None)
    flow.decompose_records(state, IMPUTATO, rows, set(), fetch=lambda keys: [])
    recs = state["records"]
    assert recs["590 c.p."]["decomposed_for"] == flow.DECOMPOSE_WHY["hypothesis"]     # the read's main one
    assert recs["589 c.p."]["decomposed_for"] == posture.READ_CANDIDATE_WHY
    assert recs["593 c.p."]["decomposed_for"] == posture.READ_CANDIDATE_WHY
    # the charge rule then scores them; one not chosen is graded, never printed
    monkeypatch.setattr(posture, "alternative_rows", lambda s, rows=(): [])
    state["rows"] = rows
    posture.mark_sides(state, IMPUTATO)
    assert state["charge"]["article"] in ("589 c.p.", "593 c.p.")
    assert recs["593 c.p." if state["charge"]["article"] == "589 c.p." else "589 c.p."]["unprinted"] is True
    assert not recs[state["charge"]["article"]].get("unprinted")


# --- 4. a fragment never leads ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, kind", [
    ("ovvero cagiona un danno alle cose", "alternative"),
    ("Oppure ne fa uso in modo diverso", "alternative"),
    ("la violenza consiste nella coazione fisica della vittima", "definition"),
    ("si intende per abitazione ogni luogo di privata dimora", "definition"),
    ("cagiona ad alcuno una lesione personale", ""),
    ("minaccia ad altri un ingiusto danno, ovvero lo costringe a fare", "")])
def test_an_ovvero_clause_and_a_definition_are_no_elements(text, kind):
    assert rule_quote.fragment_kind(text) == kind
    assert rule_quote.fragment_kind(text, wider=False) == ""        # the schema checker's kinds are v79.9's
    assert bool(rule_quote.fragment_why(text)) == bool(kind)


def _sol(tesi, kind, realismo, contesta=(), poggia_su=(), **extra):
    return {"tesi": tesi, "candidate": tesi, "kind": kind, "realismo": realismo, "contesta": list(contesta),
            "poggia_su": list(poggia_su), **extra}


def test_an_open_fragment_contest_never_leads_over_a_real_contest():
    known = {"R1": {"article": "582 c.p.", "text": "ovvero ne fa uso", "status": "da provare", "side": "threat"},
             "R2": {"article": "582 c.p.", "text": "dalla quale deriva una malattia", "status": "da provare",
                    "side": "threat"}}
    sols = [_sol("Frammento", "contest_open", "alto", ["R1"]), _sol("Malattia", "contest_open", "medio", ["R2"])]
    assert M.escluso_fragment(sols[0], known) and not M.escluso_fragment(sols[1], known)
    assert M.never_main(sols, [0, 1], known) == {0: M.MAIN_DEMOTED_WHY["fragment"]}


def test_the_containment_line_counts_so_a_fragment_never_leads_alone():
    recs = {"582 c.p.": _rec("Lesione personale", _req(1, "il debitore", "non risulta"),
                             _req(2, "cagiona ad alcuno una lesione personale"), side="threat", charged=1)}
    sols = [_sol("Frammento", "contest_excluded", "alto", ["R1"]),
            _sol("Contenere la pena", "contenimento", "medio", fallback=True)]
    got, main = M.coherent_solutions({"soluzioni": sols}, recs, {"article": "582 c.p."}, ())
    assert got[main]["kind"] == "contenimento" and got[0]["main_demoted"] == M.MAIN_DEMOTED_WHY["fragment"]
    # truly alone (no other line at all): it is not demoted -- nothing else could lead
    assert M.never_main([sols[0]], [0], M.elements_by_id(recs)) == {}


def _two_threats():
    return {"582 c.p.": _rec("Lesione personale", _req(1, "cagiona ad alcuno una lesione personale"),
                             _req(2, "dalla quale deriva una malattia", "da provare"), side="threat", charged=1),
            "612 c.p.": _rec("Minaccia", _req(3, "minaccia ad altri un ingiusto danno", "da provare"),
                             side="threat")}


def test_a_contest_of_another_threat_never_leads_while_the_charge_has_a_line():
    recs = _two_threats()
    sols = [_sol("Contesta la minaccia", "contest_open", "alto", ["R3"]),
            _sol("Contesta la malattia", "contest_open", "medio", ["R2"]),
            _sol("Contenere la pena", "contenimento", "basso", fallback=True)]
    got, main = M.coherent_solutions({"soluzioni": sols}, recs, {"article": "582 c.p."}, ())
    assert got[main]["tesi"] == "Contesta la malattia"
    assert got[0]["main_demoted"] == M.MAIN_DEMOTED_WHY["other_threat"] and got[0]["other_threat"]
    # it ranks after the charge's lines: the subordinate is the containment line, never the other threat's contest
    assert got[M.subordinate_of(got, main)]["tesi"] == "Contenere la pena"


def test_the_threats_contest_is_judged_against_the_charge_only():
    known = M.elements_by_id(_two_threats())
    assert M.other_threat_contest(_sol("x", "contest_open", "alto", ["R3"]), known, ["582 c.p."])
    assert not M.other_threat_contest(_sol("x", "contest_open", "alto", ["R2"]), known, ["582 c.p."])
    assert not M.other_threat_contest(_sol("x", "contest_open", "alto", ["R3"]), known, [])     # no charge known


def test_an_event_the_certain_facts_narrate_is_never_da_provare():
    recs = {"589 c.p.": _rec("Omicidio colposo", _req(1, "la morte di una persona", "da provare"),
                             _req(2, "cagiona per colpa la morte", "da provare"), _req(3, "per colpa", "da provare"))}
    certain = "Alfa investe Beta con l'auto. Beta muore in ospedale."
    assert M.event_unproven(recs, certain) == 1
    got = M.narrated_event_proven(recs, certain)
    assert got == ["589 c.p.: R1"]
    els = recs["589 c.p."]["requisites"]
    assert els[0]["status"] == M.PROVATO and els[0]["event_narrated"] == "morte"
    assert els[1]["status"] == M.DA_PROVARE           # the causal link is never proven by the narration alone
    assert M.event_unproven(recs, certain) == 0
    # no death among the certain facts (a threat of death is none): nothing changes
    recs2 = {"589 c.p.": _rec("Omicidio colposo", _req(1, "la morte di una persona", "da provare"))}
    assert M.narrated_event_proven(recs2, "Alfa minaccia Beta di morte.") == []


def test_prepare_proves_the_narrated_event_from_the_certain_sentences_only():
    recs = {"589 c.p.": _rec("Omicidio colposo", _req(1, "la morte di una persona", "da provare"),
                             _req(2, "la violazione di una regola cautelare", "da provare"), side="threat", charged=1)}
    sents = [_sent(1, "Alfa investe Beta con l'auto."), _sent(2, "Beta muore in ospedale.")]
    ctx = M.prepare({"records": recs, "sentences": sents}, IMPUTATO, [], "Alfa investe Beta. Beta muore.", "Alfa", "",
                    list(recs), [], [], [], [], link_call=lambda *a: {}, case_id_="v79_12_4")
    assert ctx["event_proven"] == ["589 c.p.: R1"]
    recs = {"589 c.p.": _rec("Omicidio colposo", _req(1, "la morte di una persona", "da provare"), side="threat",
                             charged=1)}
    sents = [_sent(1, "Alfa investe Beta con l'auto."), _sent(2, "Beta forse è morto.", "ipotizzato")]
    ctx = M.prepare({"records": recs, "sentences": sents}, IMPUTATO, [], "Alfa investe Beta.", "Alfa", "",
                    list(recs), [], [], [], [], link_call=lambda *a: {}, case_id_="v79_12_4")
    assert ctx["event_proven"] == [] and recs["589 c.p."]["requisites"][0]["status"] == M.DA_PROVARE


# --- 5. 40 and 43 never defences; a half-proven scriminante never leads ---------------------------------------------------

def test_40_and_43_are_never_defence_lines():
    recs = {"582 c.p.": _rec("Lesione personale", _req(1, "cagiona ad alcuno una lesione personale"),
                             _req(2, "dalla quale deriva una malattia", "da provare"), side="threat", charged=1),
            "40 c.p.": _rec("Rapporto di causalità", _req(3, "l'evento è conseguenza dell'azione"),
                            decomposed_for=issue_records.DEFENCE_WHY),
            "43 c.p.": _rec("Elemento psicologico del reato", _req(4, "l'evento è preveduto e voluto"),
                            decomposed_for=issue_records.DEFENCE_WHY),
            "55 c.p.": _rec("Eccesso colposo", _req(5, "si eccedono colposamente i limiti"),
                            decomposed_for=issue_records.DEFENCE_WHY)}
    ctx = {"records": recs, "circumstances": {}, "posture": IMPUTATO, "case": "Alfa colpisce Beta.",
           "assessment": {"article": "582 c.p."}, "client": "Alfa", "institutions": [], "riti": []}
    cands = M.thesis_candidates(ctx)
    defences = [c["article"] for c in cands if c["kind"] in M.DEFENCE_KINDS]
    assert defences == ["55 c.p."]
    assert ctx["candidates_trace"]["never_defence"] == ["40 c.p.", "43 c.p."]
    assert M.never_defence("43 c.p.") and M.never_defence("40 c.p. co. 2") and not M.never_defence("41 c.p.")
    # a model's own line on them never leads either
    known = M.elements_by_id(recs)
    sols = [_sol("Causalità", "defence", "alto", poggia_su=["R3"], article="40 c.p."),
            _sol("Contesta R2", "contest_open", "medio", ["R2"])]
    assert M.never_main(sols, [0, 1], known)[0] == M.MAIN_DEMOTED_WHY["never_defence"]


def _scriminante(*statuses):
    texts = ("la necessità di difendere un diritto", "il pericolo attuale di un'offesa ingiusta",
             "la difesa proporzionata all'offesa")
    return {"582 c.p.": _rec("Lesione personale", _req(1, "cagiona ad alcuno una lesione personale"),
                             _req(2, "dalla quale deriva una malattia", "da provare"), side="threat", charged=1),
            "52 c.p.": _rec("Difesa legittima", *[_req(10 + i, t, s) for i, (t, s) in enumerate(zip(texts, statuses))])}


def test_a_scriminante_with_fewer_than_half_its_elements_proven_never_leads():
    recs = _scriminante("provato", "da provare", "da provare")
    sols = [_sol("Legittima difesa", "scriminante", "alto", poggia_su=["R10"], article="52 c.p."),
            _sol("Contesta R2", "contest_open", "medio", ["R2"])]
    got, main = M.coherent_solutions({"soluzioni": sols}, recs, {"article": "582 c.p."}, ())
    assert got[main]["tesi"] == "Contesta R2" and got[0]["main_demoted"] == M.MAIN_DEMOTED_WHY["weak_defence"]
    # half proven (2 of 3): it may lead on its realismo
    recs = _scriminante("provato", "provato", "da provare")
    got, main = M.coherent_solutions({"soluzioni": [dict(s) for s in sols]}, recs, {"article": "582 c.p."}, ())
    assert got[main]["tesi"] == "Legittima difesa"
    # alone it is not demoted: nothing else is eligible
    recs = _scriminante("provato", "da provare", "da provare")
    assert M.never_main([sols[0]], [0], M.elements_by_id(recs)) == {}


# --- 7. expired terms -------------------------------------------------------------------------------------------------------

def test_the_terms_table_names_the_articles_each_term_bars():
    assert fact_signals.term_bars("possessoria") == (("1168 c.c.", "1170 c.c."), "decadenza")
    assert fact_signals.term_bars("azione_garanzia_appalto") == (("1667 c.c.", "1668 c.c."), "prescrizione")
    assert fact_signals.term_bars("usucapione_ordinaria") == ((), "")
    for t in fact_signals.TERMS:
        assert (not t.bars) == (not t.bars_kind) and t.bars_kind in ("", "prescrizione", "decadenza")
    terms = [{"id": "possessoria", "status": "DECORSO", "ref": "artt. 1168 e 1170 c.c."},
             {"id": "prescrizione_ordinaria", "status": "NON DECORSO"}]
    assert fact_signals.barring_terms("1168 c.c.", terms) == [
        {"id": "possessoria", "ref": "artt. 1168 e 1170 c.c.", "kind": "decadenza", "firm": True}]
    assert fact_signals.barring_terms("1453 c.c.", terms) == []                 # not decorso
    assert fact_signals.barring_terms("1168 c.p.", terms) == []                 # another code


APPALTO = "Tizio ha affidato all'impresa Alfa il rifacimento del tetto; dopo le piogge compaiono infiltrazioni."


def _appalto(terms):
    recs = {"1218 c.c.": _rec("Responsabilità del debitore", _req(1, "l'inadempimento della prestazione")),
            "1453 c.c.": _rec("Risolubilità del contratto per inadempimento", _req(2, "l'inadempimento")),
            "1667 c.c.": _rec("Difformità e vizi dell'opera", _req(3, "l'opera presenta vizi"))}
    return {"records": recs, "circumstances": {}, "posture": ATTORE, "case": APPALTO, "assessment": {},
            "institutions": [], "terms": terms}


def _claims(cands):
    return [(c["article"], c["label"]) for c in cands if c["kind"] == "claim"]


def test_a_claim_whose_action_has_lapsed_is_refused():
    ctx = _appalto([{"id": "denuncia_difformita_appalto", "status": "DECORSO", "ref": "art. 1667 c.c."}])
    cands = M.thesis_candidates(ctx)
    assert "1667 c.c." not in [a for a, _ in _claims(cands)]
    assert ctx["claims_refused"]["1667 c.c."].startswith("termine decorso")
    assert ctx["claims_expired"] == ["1667 c.c."]
    # the module's line refused: general obligations law follows, no longer subordinate to it -- (review) minoritaria
    # under the module's terms, never sustainable: the general remedies for the same defect follow the special terms
    assert _claims(cands) == [("1218 c.c.", M.MINORITY), ("1453 c.c.", M.MINORITY)]
    assert all(c["expired"] == ["denuncia_difformita_appalto"] for c in cands if c["kind"] == "claim")


def test_a_prescribed_claim_is_kept_minoritaria():
    ctx = _appalto([{"id": "azione_garanzia_appalto", "status": "DECORSO", "ref": "art. 1667 c.c."}])
    cands = M.thesis_candidates(ctx)
    # (review) the module's prescribed line is no proven line: general obligations law is no longer refused as
    # subordinate to it -- it follows, minoritaria under the module's terms
    assert _claims(cands) == [("1667 c.c.", M.MINORITY), ("1218 c.c.", M.MINORITY), ("1453 c.c.", M.MINORITY)]
    claim = next(c for c in cands if c["kind"] == "claim")
    assert claim["expired"] == ["azione_garanzia_appalto"]
    assert ctx["claims_expired_minority"] == ["1667 c.c.", "1218 c.c.", "1453 c.c."]
    assert "1218 c.c." not in ctx["claims_refused"]


def test_a_term_not_decorso_changes_nothing():
    ctx = _appalto([{"id": "azione_garanzia_appalto", "status": "NON DECORSO"}])
    assert _claims(M.thesis_candidates(ctx)) == [("1667 c.c.", M.SUSTAINABLE)]


def test_prepare_reads_the_computed_terms():
    state = {"records": {}, "sentences": [], "analysis": {"fact_signals": {"terms": [
        {"id": "possessoria", "status": "DECORSO", "ref": "artt. 1168 e 1170 c.c."}]}}}
    assert M.computed_terms(state) == [{"id": "possessoria", "status": "DECORSO", "ref": "artt. 1168 e 1170 c.c."}]
    assert M.computed_terms({"analysis": "restored"}) == []


# --- the counters -------------------------------------------------------------------------------------------------------------

def _valued(cands, **by_kind):
    return {"valutazioni": {c["id"]: {"realismo": by_kind.get(c["kind"], "medio")} for c in cands}}


def test_the_selection_counts_carry_the_items_counters():
    ctx = _appalto([{"id": "denuncia_difformita_appalto", "status": "DECORSO", "ref": "art. 1667 c.c."}])
    ctx["candidates"] = M.thesis_candidates(ctx)
    raw = _valued(ctx["candidates"])
    counts = M.selection_counts(ctx, M.clean_method(raw, ctx["candidates"]), raw)
    assert counts["claims_refused_expired"] == 1 and counts["claims_expired_sustainable"] == 0
    assert counts["claims_expired_minority"] == 2                  # 1218, 1453 under the module's decadenza
    for k in ("victim_charge_civil", "main_fragment", "main_other_threat", "event_narrated_unproven",
              "defence_lines_40_43", "main_weak_defence", "event_narrated_proven", "event_narrated_causal"):
        assert counts[k] == 0
    state = {"thesis_counts": counts, "block_counters": {"charge_event_refused": 1, "charged_undecomposed": 0}}
    got = flow.case_counters(state)
    assert got["claims_refused_expired"] == 1 and got["charge_event_refused"] == 1
    assert got["claims_expired_sustainable"] == 0 and "charged_undecomposed" in got


def test_a_victims_civil_outcome_would_be_counted():
    ctx = {"records": {}, "candidates": [], "posture": OFFESA, "assessment": {"article": "2043 c.c."}}
    assert M.item_counts(ctx, [], 0, {})["victim_charge_civil"] == 1
    ctx["assessment"] = {"article": "582 c.p."}
    assert M.item_counts(ctx, [], 0, {})["victim_charge_civil"] == 0


# --- review fixes ---------------------------------------------------------------------------------------------------------------

def test_a_circumstance_is_never_the_exposure_the_heaviest_fattispecie_is(monkeypatch):
    case = "Il PM contesta ad Alfa gli artt. 612, 582 e 583 c.p."
    monkeypatch.setattr(flow, "case_text", lambda s: case)
    monkeypatch.setattr(posture, "alternative_rows", lambda s, rows=(): [])
    rows = [_row("612", "c.p.", rubrica="Minaccia", pena="reclusione fino a un anno"),
            _row("582", "c.p.", rubrica="Lesione personale", pena="reclusione da sei mesi a tre anni"),
            _row("583", "c.p.", rubrica="Circostanze aggravanti", pena="reclusione da tre a sette anni")]
    state = {"read": {"hypotheses": []}, "rows": rows, "reserve": [],
             "records": {"612 c.p.": _rec("Minaccia", _req(1, "minaccia ad altri un ingiusto danno")),
                         "582 c.p.": _rec("Lesione personale", _req(2, "cagiona ad alcuno una lesione personale")),
                         "583 c.p.": _rec("Circostanze aggravanti", _req(3, "una malattia per più di quaranta giorni"))}}
    assert not issue_records.counted("583 c.p.", "Circostanze aggravanti")
    out = posture.mark_sides(state, IMPUTATO)
    assert out["exposure"] == "582 c.p." and state["charge"]["exposure"] == "582 c.p."
    assert not state["records"]["583 c.p."].get("exposure")
    assert M.main_threat(state["records"]) == "582 c.p."
    assert M.assessment(state["records"], IMPUTATO)["article"] == "582 c.p."
    assert {x["charge"] for x in M.charge_rites(state["records"], rows, IMPUTATO)} == {"582 c.p."}


def test_an_attempt_is_never_refused_by_the_event_table(monkeypatch):
    case = "Alfa colpisce Beta con sei coltellate al torace. Beta viene ricoverato con ferite profonde e sopravvive."
    monkeypatch.setattr(flow, "case_text", lambda s: case)
    monkeypatch.setattr(posture, "alternative_rows", lambda s, rows=(): [])
    hyps = [{"line": "principale", "claim": "tentato omicidio", "institutions": ["omicidio volontario"]},
            {"line": "subordinata", "claim": "lesioni", "institutions": ["lesioni"]}]
    state = {"read": {"hypotheses": hyps}, "reserve": [],
             "rows": [_row("575", "c.p.", "omicidio volontario", "Omicidio"),
                      _row("582", "c.p.", "lesioni", "Lesione personale")],
             "records": {"575 c.p.": _rec("Omicidio", _req(1, "cagiona la morte di un uomo", "da provare"),
                                          _req(2, "con coscienza e volontà")),
                         "582 c.p.": _rec("Lesione personale",
                                          _req(3, "cagiona ad alcuno una lesione personale", "da provare"))}}
    out = posture.mark_sides(state, IMPUTATO)
    assert state["charge"] == {"article": "575 c.p.", "rule": "read_hypothesis"} and out["event_refused"] == []
    assert state["block_counters"]["charge_event_mismatch"] == 0
    assert posture.attempted(hyps[0]) and not posture.attempted(hyps[1])
    assert posture.event_refuses("575 c.p.", "lesioni", hyps[1]) and not posture.event_refuses("575 c.p.", "lesioni",
                                                                                                 hyps[0])


def test_a_comma_label_is_penal_and_the_victims_outcome_keeps_it():
    assert posture.penal_article("582 c.p. co. 2") and posture.penal_article("612-bis c.p. co. 2")
    assert not posture.penal_article("2043 c.c. co. 1")
    recs = {"582 c.p. co. 2": _rec("Lesione personale", _req(1, "cagiona ad alcuno una lesione personale"), charged=1),
            "2043 c.c.": _rec("Risarcimento per fatto illecito", _req(2, "un danno ingiusto"))}
    assert M.assessment(recs, OFFESA)["article"] == "582 c.p. co. 2"


def test_a_line_on_40_without_an_article_key_is_still_demoted():
    recs = {"582 c.p.": _rec("Lesione personale", _req(1, "x"), _req(2, "y", "da provare"), side="threat", charged=1),
            "40 c.p.": _rec("Rapporto di causalità", _req(3, "l'evento è conseguenza dell'azione"))}
    known = M.elements_by_id(recs)
    sols = [_sol("Causalità", "defence", "alto", poggia_su=["R3"]),          # as clean_method builds it: no article
            _sol("Contesta R2", "contest_open", "medio", ["R2"])]
    assert M.line_article(sols[0], known) == "40 c.p."
    assert M.never_main(sols, [0, 1], known) == {0: M.MAIN_DEMOTED_WHY["never_defence"]}


@pytest.mark.parametrize("text, event, how", [
    ("la morte di una persona", "morte", "bare"),
    ("la morte di un uomo", "morte", "bare"),
    ("una malattia nel corpo o nella mente", "lesioni", "bare"),
    ("una lesione personale", "lesioni", "bare"),
    ("l'evento lesivo", "lesioni", "bare"),
    ("cagiona la morte di un uomo", "morte", "causal"),
    ("cagiona per colpa la morte di una persona", "morte", "causal"),
    ("la morte come conseguenza degli atti", "morte", "causal"),
    ("cagiona ad alcuno una lesione personale, dalla quale deriva una malattia nel corpo o nella mente", "lesioni",
     "causal"),
    # a qualifier, a means, a conduct mentioning the victim's state: never the bare event
    ("lesioni personali gravi o gravissime", "lesioni", ""),
    ("una malattia o un'incapacità di attendere alle ordinarie occupazioni per un tempo superiore ai quaranta giorni "
     "ovvero lesioni gravissime", "lesioni", ""),
    ("lesione commessa con armi", "lesioni", ""),
    ("trovando una persona ferita o altrimenti in pericolo, omette di prestare l'assistenza occorrente", "lesioni", ""),
    ("morte cagionata con premeditazione", "morte", ""),
    ("la morte di una persona", "lesioni", "")])
def test_only_the_bare_event_is_proven_by_the_narration(text, event, how):
    assert M.event_element(text, event) == how


def test_gravity_and_conduct_elements_are_never_set_provato_by_the_narration():
    certain = "Alfa urta Beta con l'auto. Beta riporta lesioni guaribili in 10 giorni."
    recs = {"593 c.p.": _rec("Omissione di soccorso", _req(1, "trovando una persona ferita o altrimenti in pericolo, "
                                                              "omette di prestare l'assistenza occorrente", "da provare")),
            "590-bis c.p.": _rec("Lesioni stradali", _req(2, "lesioni personali gravi o gravissime", "da provare")),
            "585 c.p.": _rec("Circostanze aggravanti", _req(3, "lesione commessa con armi", "da provare")),
            "582 c.p.": _rec("Lesione personale", _req(4, "una malattia nel corpo o nella mente", "da provare"),
                             _req(5, "cagiona ad alcuno una lesione personale", "da provare"))}
    assert M.narrated_event_proven(recs, certain) == ["582 c.p.: R4"]
    assert all(r["status"] == M.DA_PROVARE for x in ("593 c.p.", "590-bis c.p.", "585 c.p.")
               for r in recs[x]["requisites"])
    causal = recs["582 c.p."]["requisites"][1]
    assert causal["status"] == M.DA_PROVARE and causal["event_narrated"] == "lesioni"     # the cause still to prove
    assert M.event_unproven(recs, certain) == 0 and M.event_causal_open(recs, certain) == 1


def _remedy_ctx(terms):
    recs = {"1667 c.c.": _rec("Difformità e vizi dell'opera", _req(1, "l'opera presenta vizi"))}
    return {"records": recs, "circumstances": {}, "posture": ATTORE, "case": APPALTO, "assessment": {},
            "institutions": [], "terms": terms}


def test_the_remedies_of_a_prescribed_claim_are_minoritaria_too():
    ctx = _remedy_ctx([{"id": "azione_garanzia_appalto", "status": "DECORSO", "ref": "art. 1667 c.c."}])
    ctx["candidates"] = M.thesis_candidates(ctx)
    remedies = [c for c in ctx["candidates"] if c["kind"] == "remedy"]
    assert remedies and all(c["minority"] and M.is_minority(c) for c in remedies)
    raw = _valued(ctx["candidates"], remedy="alto", claim="alto")
    plan = M.clean_method(raw, ctx["candidates"])
    assert all(s["minoritaria"] and s["realismo"] == M.BASSO for s in plan["soluzioni"] if s["kind"] == "remedy")
    counts = M.selection_counts(ctx, plan, raw)
    assert counts["claims_expired_sustainable"] == 0
    # not decorso: the remedies stay as they were
    ctx = _remedy_ctx([{"id": "azione_garanzia_appalto", "status": "NON DECORSO"}])
    assert not any(M.is_minority(c) for c in M.thesis_candidates(ctx) if c["kind"] == "remedy")


def _facts(**kw):
    base = {k: ([] if isinstance(v, list) else "") for k, v in fact_signals.SCHEMA.items()}
    base.update(kw)
    return base


APPALTO_CONTRACT = [{"type": "appalto per il rifacimento del tetto", "parties": "Tizio, impresa Alfa", "clauses": ""}]
DISCOVERY = {"what": "Tizio scopre infiltrazioni dal tetto", "date": "2024-10-10", "kind": "scoperta vizio"}


def test_a_denuncia_the_case_does_not_mention_never_refuses_the_claim():
    terms = fact_signals.compute_terms(_facts(contracts=APPALTO_CONTRACT, events=[
        DISCOVERY, {"what": "Tizio chiede un preventivo a un'altra impresa", "date": "2025-02-01", "kind": ""}]))
    term = next(t for t in terms if t["id"] == "denuncia_difformita_appalto")
    assert term["status"] == "DECORSO" and term["to_now"] is True
    assert fact_signals.barring_terms("1667 c.c.", terms)[0]["firm"] is False
    ctx = _appalto(terms)
    cands = M.thesis_candidates(ctx)
    assert ("1667 c.c.", M.MINORITY) in _claims(cands) and "1667 c.c." not in ctx["claims_refused"]
    assert ctx["claims_expired"] == []


def test_a_late_denuncia_the_case_states_refuses_the_claim():
    terms = fact_signals.compute_terms(_facts(contracts=APPALTO_CONTRACT, events=[
        DISCOVERY, {"what": "Tizio denuncia i vizi all'impresa Alfa", "date": "2025-02-01", "kind": "denuncia/reclamo"}]))
    term = next(t for t in terms if t["id"] == "denuncia_difformita_appalto")
    assert term["status"] == "DECORSO" and not term.get("to_now")
    ctx = _appalto(terms)
    M.thesis_candidates(ctx)
    assert ctx["claims_refused"]["1667 c.c."].startswith("termine decorso")
    # the contractor acknowledged the defect (art. 1667 co. 2 c.c.): no denuncia needed -- kept, minoritaria
    ctx = _appalto(terms)
    ctx["case"] = APPALTO + " L'impresa Alfa ha riconosciuto i vizi per iscritto."
    assert ("1667 c.c.", M.MINORITY) in _claims(M.thesis_candidates(ctx))


def test_a_possessoria_run_to_the_present_is_firm():
    terms = [{"id": "possessoria", "status": "DECORSO", "ref": "artt. 1168 e 1170 c.c.", "to_now": True}]
    assert fact_signals.barring_terms("1168 c.c.", terms)[0]["firm"] is True      # the client has not sued: firm
    assert fact_signals.barring_terms("1168 c.c. co. 2", terms) and fact_signals.barring_terms("1170 c.c. co. 3", terms)


def test_every_claim_barred_says_the_term_never_that_no_norm_confers_an_action():
    recs = {"1168 c.c.": _rec("Azione di reintegrazione", _req(1, "lo spoglio violento o clandestino del possesso"))}
    case = "Tizio possedeva il cortile; il vicino Caio ha cambiato la serratura del cancello due anni fa."
    ctx = {"records": recs, "circumstances": {}, "posture": ATTORE, "case": case, "assessment": {}, "institutions": [],
           "terms": [{"id": "possessoria", "status": "DECORSO", "ref": "artt. 1168 e 1170 c.c."}]}
    assert M.thesis_candidates(ctx) == []
    line = ctx["no_action"]
    assert M.NO_ACTION_CORE not in line and "preclusa dal termine decorso" in line
    assert "azione possessoria" in line and "1168" in line
    ctx["terms"] = []
    assert M.thesis_candidates(ctx) and not ctx.get("no_action")
