"""v79.10.3 -- Block C of the 79.10 brief, rites and alternatives that fit the charge, on synthetic sentences, records
and store entries and the nightly cells only:

C1. the riti under the main line only when the charge's pena from the procedural store allows them: patteggiamento
    (444 c.p.p.) only if the minimum reduced by a third can be within five years (v79.10.7: its best case, the
    attenuanti held counted, «possibile» -- tests/test_v79_10_7.py); messa alla prova (168-bis c.p.) only for a maximum
    within four years or a delitto of art. 550, co. 2, c.p.p. (data); abbreviato always (v79.10.7: but the ergastolo,
    art. 438, co. 1-bis, c.p.p.; «da verificare» on an unknown pena); the pena
    unknown, «da verificare sulla pena edittale», never as admitted; a precluded rito is §8's line, never the
    strategy's; remissione della querela and 162-ter only when the store says querela
C2. an alternative qualification fits the event of the certain facts: a death, never 590 (589, 584, 586 are); injuries
    and no death, never 575 / 589 (data/event_offences.json)
C3. the counters rites_unavailable_offered and requalify_event_mismatch (both 0), the nightly's check and table
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


def _script(name):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _req(i, text, status="provato", **kw):
    return {"id": f"R{i}", "comma": 1, "kind": "requisito", "text": text, "status": status, **kw}


def _posture(role="imputato", stage="primo_grado"):
    from src.rag import posture
    return {"rito": "penale", "family": "penale", "role": role, "stage": stage,
            "entry": posture.entry_for("penale", role, stage)}


IMPUTATO = _posture()


def _row(number, pena=None, text="", **facts):
    """A synthetic c.p. row: its procedural store entry (_proc) when a pena is given, else its text alone."""
    row = {"_article": (number, "c.p."), "s": {"plain_text": text or f"Chiunque commette il fatto ({number})."}}
    if pena is not None:
        row["_proc"] = {"version": 4, "pena": pena, "querela": False, **facts}
    return row


def _charge(label, rubrica="Reato contestato"):
    return {label: {"rubrica": rubrica, "side": "threat", "charged": True, "decomposed_for": "addebito",
                    "requisites": [_req(1, "la condotta descritta dalla norma")]}}


def _by_id(statuses):
    return {x["id"]: x for x in statuses}


# --- C1: the riti by the charge's pena ----------------------------------------------------------------------------------------

# v79.10.7: the patteggiamento «possibile» (never «ammesso») when its best case is within five years -- here with no
# attenuante held; the abbreviato «ammesso» unless the ergastolo (art. 438, co. 1-bis, c.p.p.), «da verificare» on an
# unknown pena like the others
@pytest.mark.parametrize("pena, number, want", [
    # the nightly cells' charges: maltrattamenti (572), furto (624), ricettazione (648, listed in art. 550, co. 2)
    ("reclusione da tre a sette anni (co. 1)", "572", {"abbreviato": "ammesso", "patteggiamento": "possibile",
                                                       "messa_alla_prova": "precluso"}),
    ("reclusione da sei mesi a tre anni e con la multa da euro 154 a euro 516 (co. 1)", "624",
     {"abbreviato": "ammesso", "patteggiamento": "possibile", "messa_alla_prova": "ammesso"}),
    ("reclusione da due a otto anni e con la multa da euro 516 a euro 10.329 (co. 1)", "648",
     {"abbreviato": "ammesso", "patteggiamento": "possibile", "messa_alla_prova": "ammesso"}),
    ("reclusione non inferiore ad anni ventuno (co. 1)", "575", {"abbreviato": "ammesso", "patteggiamento": "precluso",
                                                                 "messa_alla_prova": "precluso"}),
    ("reclusione da dodici a venti anni (co. 1)", "999", {"abbreviato": "ammesso", "patteggiamento": "precluso",
                                                          "messa_alla_prova": "precluso"}),
    # v79.10.7: no attenuante held -- 584's 10 years less the rite's third is 6 years and 8 months: precluded
    ("reclusione da dieci a diciotto anni (co. 1)", "584", {"abbreviato": "ammesso", "patteggiamento": "precluso",
                                                            "messa_alla_prova": "precluso"}),
    ("reclusione da otto a venti anni (co. 1)", "999", {"abbreviato": "ammesso", "patteggiamento": "precluso",
                                                        "messa_alla_prova": "precluso"}),
    ("reclusione da sette a quindici anni (co. 1)", "999", {"abbreviato": "ammesso", "patteggiamento": "possibile",
                                                            "messa_alla_prova": "precluso"}),
    ("ergastolo (co. 1)", "999", {"abbreviato": "precluso", "patteggiamento": "precluso",
                                  "messa_alla_prova": "precluso"}),
    ("multa fino a euro 309 (co. 1)", "999", {"abbreviato": "ammesso", "patteggiamento": "possibile",
                                              "messa_alla_prova": "ammesso"}),
    ("arresto fino a un anno (co. 1)", "999", {"abbreviato": "ammesso", "patteggiamento": "possibile",
                                               "messa_alla_prova": "ammesso"}),
    ("", "999", {"abbreviato": "da verificare", "patteggiamento": "da verificare", "messa_alla_prova": "da verificare"}),
    ("", "337", {"abbreviato": "da verificare", "patteggiamento": "da verificare",
                 "messa_alla_prova": "ammesso"}),                                   # listed: whatever its pena
])
def test_each_rito_is_gated_on_the_charges_pena(pena, number, want):
    from src.rag import lawyer_method as M
    got = _by_id(M.rite_statuses(pena, number))
    assert {k: got[k]["status"] for k in want} == want
    assert got["commisurazione"]["status"] == M.AVAILABLE                   # the sentencing line: no rito


def test_the_listed_offences_are_data():
    from src.rag import lawyer_method as M
    entry = next(r for r in M.data()["riti_alternativi"] if r["id"] == "messa_alla_prova")
    assert {"336", "337", "625", "648"} <= set(entry["listed_offences"]) and "550, co. 2" in entry["listed_source"]
    line = _by_id(M.rite_statuses("reclusione da due a otto anni (co. 1)", "648"))["messa_alla_prova"]["line"]
    assert "ammessa per il reato contestato" in line and "550, co. 2" in line
    out = _by_id(M.rite_statuses("reclusione da tre a sette anni (co. 1)", "572"))["messa_alla_prova"]["line"]
    assert "preclusa" in out and "supera 4 anni" in out and "550, co. 2" in out


def test_the_pena_unknown_is_never_admitted():
    from src.rag import lawyer_method as M
    lines = M.riti_lines(IMPUTATO, "")
    unknown = [x for x in lines if "444" in x or "168-bis" in x]
    assert len(unknown) == 2 and all(M.RITE_TO_VERIFY in x for x in unknown)
    assert not any(x.split(":", 1)[1].strip().startswith("ammess") for x in unknown)
    # v79.10.7: the abbreviato too (art. 438, co. 1-bis, c.p.p.: the ergastolo cannot be ruled out), never named
    abbreviato = next(x for x in lines if "abbreviato" in x)
    assert M.RITE_TO_VERIFY in abbreviato and "438, co. 1-bis, c.p.p.)" in abbreviato
    assert M.riti_names(lines, 3) == "Commisurazione della pena (art. 133 c.p.) e attenuanti generiche (art. 62-bis c.p.)"


def test_the_charges_pena_comes_from_the_procedural_store():
    from src.rag import lawyer_method as M
    recs = _charge("572 c.p.", "Maltrattamenti contro familiari e conviventi")
    got = _by_id(M.charge_rites(recs, [_row("572", "reclusione da tre a sette anni (co. 1)")], IMPUTATO))
    assert got["patteggiamento"]["status"] == M.POSSIBLE and got["messa_alla_prova"]["status"] == M.PRECLUDED
    assert got["patteggiamento"]["charge"] == "572 c.p." and got["patteggiamento"]["pena"].startswith("reclusione")
    # no store entry: computed from the row's text, as the store's own reader does
    text = "Chiunque maltratta una persona della famiglia è punito con la reclusione da tre a sette anni."
    assert _by_id(M.charge_rites(recs, [_row("572", text=text)], IMPUTATO))["messa_alla_prova"]["status"] == M.PRECLUDED
    # no row at all, or a special law's charge (the store reads the c.p. alone): the pena unknown
    assert _by_id(M.charge_rites(recs, [], IMPUTATO))["patteggiamento"]["status"] == M.TO_VERIFY_RITE
    special = _charge("73 d.P.R. 309/1990", "Produzione, traffico e detenzione illeciti di sostanze stupefacenti")
    assert _by_id(M.charge_rites(special, [], IMPUTATO))["messa_alla_prova"]["status"] == M.TO_VERIFY_RITE
    assert M.charge_rites(recs, [], {"rito": "civile", "family": "civile", "role": "convenuto"}) == []


def test_a_precluded_rito_is_never_under_the_strategy_and_is_said_in_8():
    from src.rag import lawyer_method as M
    heavy = "reclusione non inferiore ad anni ventuno (co. 1)"
    riti = M.riti_lines(IMPUTATO, heavy, "575")
    assert not any("444" in x or "168-bis" in x for x in riti)
    sols = [M.fallback_thesis({"obiettivo": M.CONTENIMENTO}, riti, IMPUTATO)]
    section = M.solutions_section(sols, 0, {}, riti, "x")
    assert "Giudizio abbreviato" in section and "444" not in section and "168-bis" not in section
    block = M.precluded_block(M.riti_precluded(IMPUTATO, heavy, "575"))
    assert block.startswith(M.RITI_PRECLUDED_HEAD) and "444" in block and "168-bis" in block
    said = "- Applicazione della pena su richiesta (art. 444 c.p.p.): preclusa, il minimo edittale ridotto di un terzo."
    assert "444" not in M.precluded_block(M.riti_precluded(IMPUTATO, heavy, "575"), said)     # never said twice


def test_the_containment_line_names_only_the_admitted_riti():
    from src.rag import lawyer_method as M
    unknown = M.fallback_thesis({"obiettivo": M.CONTENIMENTO}, M.riti_lines(IMPUTATO, ""), IMPUTATO)
    assert "abbreviato" not in unknown["pro"] and "444" not in unknown["pro"]          # v79.10.7: none admitted
    riti = M.riti_lines(IMPUTATO, "reclusione da dieci a diciotto anni (co. 1)", "584")
    thesis = M.fallback_thesis({"obiettivo": M.CONTENIMENTO}, riti, IMPUTATO)
    assert "Giudizio abbreviato" in thesis["pro"] and "444" not in thesis["pro"] and "168-bis" not in thesis["pro"]
    ctx = {"riti": riti, "posture": IMPUTATO, "records": {}, "circumstances": {}, "facts": {}}
    steps = M.steps_from_selection(ctx, [thesis], 0)
    assert steps and "444" not in steps[0]["scopo"] and "abbreviato" in steps[0]["scopo"]


def test_the_ritos_own_da_verificare_stays_on_its_line_and_its_point_is_in_10():
    from src.rag import lawyer_method as M
    line = M.riti_lines(IMPUTATO, "")[1]
    other = "La data della querela è da verificare."
    (_, text), = M.collect_points([("Soluzioni", f"- {line}\n{other}")])[0]
    assert M.RITE_TO_VERIFY in text and "v. punto 1" in text
    sections = "\n\n".join(f"**{i}. {t}**\n\n{'- ' + line if t == M.title('soluzioni') else 'testo'}"
                           for i, t in enumerate(M.titles(), 1))
    checks = {c["check"]: c["ok"] for c in M.structure_checks(sections)}
    assert checks["markers_collected"]
    assert M.RITE_TO_VERIFY in M.RITE_POINT.format(names="x")
    # (review) §10's point is never itself an offer the whole-answer sweep drops
    unknown = M.rite_statuses("", "")
    point = M.RITE_POINT.format(names=" e ".join(x["name"] for x in unknown if x["status"] == M.TO_VERIFY_RITE))
    assert "444" in point and "168-bis" in point and M.rites_offered_unavailable(f"1. {point}", unknown) == []


def test_a_written_sentence_offering_a_rito_not_admitted_is_dropped():
    from src.rag import lawyer_method as M
    statuses = M.rite_statuses("reclusione non inferiore ad anni ventuno (co. 1)", "575")
    text = ("Si potrà chiedere il patteggiamento per ottenere uno sconto. Il giudizio abbreviato riduce la pena.\n"
            "La messa alla prova non è ammessa per questo reato.\n"
            "| Contenimento | Il patteggiamento è la via più rapida. Riduce i tempi. | nulla |")
    assert len(M.rites_offered_unavailable(text, statuses)) == 2
    log = []
    out = M.rites_out(text, statuses, log)
    assert "patteggiamento" not in out and "abbreviato" in out and "non è ammessa" in out
    assert "| Contenimento | Riduce i tempi. | nulla |" in out and len(log) == 2
    assert M.rites_offered_unavailable(out, statuses) == []
    admitted = M.rite_statuses("reclusione da sei mesi a tre anni (co. 1)", "624")
    assert M.rites_out(text, admitted) == text                          # admitted: nothing to drop
    both = statuses + admitted                                          # several clients: one admitted, kept
    assert M.rites_out(text, both) == text


def test_the_plans_subordinate_lines_and_8s_block_follow_the_charge():
    from src.rag import lawyer_method as M, posture, render_computed
    heavy = [dict(x, charge="575 c.p.") for x in M.rite_statuses("reclusione non inferiore ad anni ventuno", "575")]
    plan = posture.apply_plan({"subordinate": []}, IMPUTATO, heavy)
    assert any("abbreviato" in x for x in plan["subordinate"]) and not any("444" in x for x in plan["subordinate"])
    unknown = M.rite_statuses("", "")
    plan = posture.apply_plan({"subordinate": []}, IMPUTATO, unknown)
    assert any("444" in x and M.RITE_TO_VERIFY in x for x in plan["subordinate"])
    assert posture.apply_plan({"subordinate": []}, IMPUTATO)["subordinate"] == IMPUTATO["entry"]["subordinate"]
    blocks = dict(render_computed.stage_blocks(IMPUTATO, rites=heavy))
    riti = blocks["Riti alternativi"]
    assert "preclusa" in riti and "Giudizio abbreviato" in riti and "Commisurazione" not in riti
    assert "L'imputato può chiedere" in dict(render_computed.stage_blocks(IMPUTATO))["Riti alternativi"]


def test_remissione_and_162_ter_only_when_the_store_says_querela():
    from src.rag import lawyer_method as M
    assert "162-ter" in M.stragiudiziale_line(IMPUTATO, "querela")
    for proc in ("ufficio", "", M.IRREVOCABLE):
        assert "162-ter" not in M.stragiudiziale_line(IMPUTATO, proc)
        assert "remissione" not in M.stragiudiziale_line(_posture("persona_offesa"), proc)
    assert "remissione" in M.stragiudiziale_line(_posture("persona_offesa"), "querela")
    text = "Un risarcimento integrale consentirebbe l'estinzione per condotte riparatorie (art. 162-ter c.p.). Altro."
    assert M.procedibility_out(text, {"582 c.p.": "querela"}, "582 c.p.", penal=True) == text
    for proc in ({"582 c.p.": "ufficio"}, {"582 c.p.": M.IRREVOCABLE}, {}):
        assert M.procedibility_out(text, proc, "582 c.p.", penal=True) == "Altro."
    civil = "La remissione del debito estingue l'obbligazione (art. 1236 c.c.)."
    assert M.procedibility_out(civil, {}, "1236 c.c.") == civil                 # outside a criminal matter: as before


def test_an_irrevocable_querela_is_read_from_the_store():
    from src.rag import lawyer_method as M
    recs = _charge("609-bis c.p.", "Violenza sessuale")
    row = _row("609-bis", "reclusione da sei a dodici anni (co. 1)", querela=True,
               remissione="querela irrevocabile (co. 4)")
    assert M.procedibility(recs, [row]) == {"609-bis c.p.": M.IRREVOCABLE}
    row["_proc"]["remissione"] = "remissibile (art. 152 c.p.)"
    assert M.procedibility(recs, [row]) == {"609-bis c.p.": "querela"}


# --- C2: the alternative fits the event -------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, want", [
    ("Tizio ha investito Caio, che è deceduto in ospedale.", "morte"),
    ("Caio è morto dopo la caduta; aveva riportato lesioni gravi.", "morte"),
    ("Caio ha riportato la frattura del perone ed è guarito dopo tre mesi.", "lesioni"),
    ("Tizio ha minacciato più volte di morte Caio.", ""),
    ("Tizio ha minacciato Caio di morte e lo ha ferito al braccio.", "lesioni"),
    ("Tizio ha sottratto una bicicletta dal cortile.", ""),
])
def test_the_event_of_the_certain_facts(text, want):
    from src.rag import lawyer_method as M
    assert M.event_of(text) == want


def test_the_event_table_is_data():
    from src.rag import lawyer_method as M
    table = json.loads((HERE.parent / "data" / "event_offences.json").read_text(encoding="utf-8"))
    assert [e["event"] for e in table["events"]] == ["morte", "lesioni"]
    assert M.event_mismatch("590 c.p.", "morte") and not M.event_mismatch("589 c.p.", "morte")
    assert not M.event_mismatch("584 c.p.", "morte") and not M.event_mismatch("586 c.p.", "morte")
    assert M.event_mismatch("575 c.p.", "lesioni") and M.event_mismatch("589 c.p.", "lesioni")
    assert not M.event_mismatch("590 c.p.", "lesioni") and not M.event_mismatch("586 c.p.", "lesioni")
    assert not M.event_mismatch("590 c.p.", "") and not M.event_mismatch("2043 c.c.", "morte")


def _alt(label, rubrica, text):
    return {label: {"rubrica": rubrica, "side": "weapon", "alternative": True, "decomposed_for": "x",
                    "requisites": [_req(5, text), _req(6, f"il fatto descritto dall'art. {label}")]}}


def _event_ctx(charge, alts, certain):
    recs = {charge[0]: {"rubrica": charge[1], "side": "threat", "decomposed_for": "x", "requisites": [
        _req(1, charge[2], distinctive=True), _req(2, "la condotta dell'imputato")]}}
    for a in alts:
        recs.update(_alt(*a))
    return {"records": recs, "circumstances": {}, "posture": _posture(stage="indagini"), "case": certain,
            "assessment": {"article": charge[0]}, "institutions": [], "certain_text": certain}


DEATH = "Il 3 marzo 2026 Tizio, alla guida, ha investito Caio, che è deceduto il giorno dopo."
INJURY = "Il 3 marzo 2026 Tizio ha colpito Caio, procurandogli la frattura del perone; Caio è guarito in tre mesi."


def test_with_a_death_590_is_never_offered_589_584_586_are():
    from src.rag import lawyer_method as M
    ctx = _event_ctx(("575 c.p.", "Omicidio", "con la volontà di uccidere"),
                     [("590 c.p.", "Lesioni personali colpose", "per colpa cagiona una lesione personale"),
                      ("589 c.p.", "Omicidio colposo", "per colpa cagiona la morte di una persona"),
                      ("584 c.p.", "Omicidio preterintenzionale", "con atti diretti a percuotere cagiona la morte"),
                      ("586 c.p.", "Morte o lesioni come conseguenza di altro delitto", "dal fatto deriva la morte")],
                     DEATH)
    cands = M.thesis_candidates(ctx)
    offered = [c["article"] for c in cands if c["kind"] == "requalify"]
    assert "590 c.p." not in offered and {"589 c.p.", "584 c.p.", "586 c.p."} <= set(offered)
    assert ctx["candidates_trace"]["requalify_refused_event"] == ["590 c.p."]
    assert ctx["candidates_trace"]["event"] == "morte"
    ctx["candidates"] = cands
    counts = M.selection_counts(ctx, {"soluzioni": []})
    assert counts["requalify_event_mismatch"] == 0 and counts["requalify_refused_event"] == 1


def test_with_injuries_and_no_death_575_and_589_are_not_offered():
    from src.rag import lawyer_method as M
    ctx = _event_ctx(("582 c.p.", "Lesione personale", "con la volontà di cagionare una lesione"),
                     [("575 c.p.", "Omicidio", "cagiona la morte di un uomo"),
                      ("589 c.p.", "Omicidio colposo", "per colpa cagiona la morte di una persona"),
                      ("590 c.p.", "Lesioni personali colpose", "per colpa cagiona una lesione personale")],
                     INJURY)
    offered = [c["article"] for c in M.thesis_candidates(ctx) if c["kind"] == "requalify"]
    assert offered == ["590 c.p."]
    assert sorted(ctx["candidates_trace"]["requalify_refused_event"]) == ["575 c.p.", "589 c.p."]


def test_no_event_no_gate_and_a_requalification_outside_the_families_stays():
    from src.rag import lawyer_method as M
    ctx = _event_ctx(("640 c.p.", "Truffa", "con artifizi o raggiri induce taluno in errore"),
                     [("646 c.p.", "Appropriazione indebita", "si appropria del denaro altrui")],
                     "Tizio ha venduto a Caio un'auto inesistente.")
    assert [c["article"] for c in M.thesis_candidates(ctx) if c["kind"] == "requalify"] == ["646 c.p."]
    assert "requalify_refused_event" not in ctx["candidates_trace"]


def test_the_victims_uncharged_offence_fits_the_event():
    from src.rag import lawyer_method as M
    recs = {"575 c.p.": {"rubrica": "Omicidio", "side": "weapon", "decomposed_for": "x",
                         "requisites": [_req(1, "cagiona la morte di un uomo")]},
            "582 c.p.": {"rubrica": "Lesione personale", "side": "weapon", "decomposed_for": "x",
                         "requisites": [_req(2, "cagiona a taluno una lesione personale")]}}
    ctx = {"records": recs, "circumstances": {}, "posture": _posture("persona_offesa", "pre_filing"),
           "case": INJURY, "assessment": {"article": ""}, "institutions": [], "certain_text": INJURY}
    claim = next(c for c in M.thesis_candidates(ctx) if c["kind"] == "claim_offence")
    assert claim["article"] == "582 c.p."


def test_the_prose_paths_alternatives_fit_the_event():
    from src.rag import interactive_flow as flow, issue_records
    state = {"sentences": issue_records.sentences(DEATH)}
    assert flow.certain_event(state) == "morte"


# --- C3: the counters and the nightly ---------------------------------------------------------------------------------------

def test_the_counters_reach_the_case_counters():
    from src.rag import interactive_flow as flow, lawyer_method as M
    state = {"thesis_counts": {"requalify_event_mismatch": 0, "requalify_refused_event": 1}}
    statuses = M.rite_statuses("reclusione non inferiore ad anni ventuno (co. 1)", "575")
    draft = "Si potrà chiedere il patteggiamento. Il giudizio abbreviato riduce la pena."
    out = flow.rites_checked(state, draft, statuses)
    assert out == "Il giudizio abbreviato riduce la pena."
    counters = flow.case_counters(state)
    assert counters["rites_unavailable_offered"] == 0 and counters["requalify_event_mismatch"] == 0
    assert counters["rites_precluded"] == 2 and counters["rites_to_verify"] == 0
    assert counters["requalify_refused_event"] == 1
    assert state["ledger"]["removed_by_step"].get("rites_out") == 1
    ctx = {"rites": M.rite_statuses("", "")}                       # the method's own statuses first
    flow.rites_checked(state, "Il patteggiamento è la via più rapida.", statuses, method_ctx=ctx)
    # v79.10.7: the abbreviato too is «da verificare» on an unknown pena
    assert state["block_counters"]["rites_to_verify"] == 3 and state["block_counters"]["rites_precluded"] == 0


def test_the_nightly_check_and_table_of_c():
    nightly = _script("nightly_sim")
    assert nightly.rites_fit_check({"rites_unavailable_offered": 0, "requalify_event_mismatch": 0})["ok"]
    assert nightly.rites_fit_check({})["ok"]
    bad = nightly.rites_fit_check({"rites_unavailable_offered": 1, "requalify_event_mismatch": 0})
    assert not bad["ok"] and bad["why"] == "rites_unavailable_offered=1"
    lines = nightly.counter_lines([{"id": "penale_imputato_primo_grado", "counters": {
        "rites_unavailable_offered": 0, "requalify_event_mismatch": 0}}])
    assert any(x.startswith("| rites_unavailable_offered | 0") for x in lines)
    assert any(x.startswith("| requalify_event_mismatch | 0") for x in lines)


def test_the_nightly_cells_charges_and_events():
    """The six cells: the penal ones' charges and events, read with synthetic store entries."""
    from src.rag import lawyer_method as M
    nightly = _script("nightly_sim")
    cells = {c["id"]: c for c in nightly.cells()}
    offesa = cells["penale_offesa_pre_filing"]["text"]
    assert M.event_of(offesa) == "lesioni"
    imputato = cells["penale_imputato_primo_grado"]["text"]
    assert M.event_of(imputato) == ""
    maltrattamenti = _by_id(M.rite_statuses("reclusione da tre a sette anni (co. 1)", "572"))
    assert maltrattamenti["messa_alla_prova"]["status"] == M.PRECLUDED
    ricettazione = _by_id(M.rite_statuses("reclusione da due a otto anni (co. 1)", "648"))
    assert ricettazione["messa_alla_prova"]["status"] == M.AVAILABLE
    assert ricettazione["messa_alla_prova"]["why"] == "listed"


# --- the review of 79.10.3 ------------------------------------------------------------------------------------------------

def test_the_written_sweep_is_the_imputatos_before_the_sentence_alone():
    """Only the imputato's answer before the sentence loses a sentence offering a rito; an appeal's, the execution's or
    the persona offesa's describes the riti freely."""
    from src.rag import interactive_flow as flow, lawyer_method as M
    text = ("Contro la sentenza di patteggiamento il ricorso per cassazione è ammesso solo per i motivi dell'art. 448, "
            "co. 2-bis, c.p.p.\nSe l'imputato chiede la messa alla prova, la persona offesa è sentita dal giudice "
            "(art. 464-quater c.p.p.).")
    for posture in (_posture(stage="impugnazione"), _posture(stage="esecuzione"), _posture("persona_offesa")):
        state = {}
        assert flow.rites_checked(state, text, M.charge_rites({}, [], posture), posture=posture) == text
        assert state["block_counters"]["rites_unavailable_offered"] == 0 and "ledger" not in state
    state = {}
    out = flow.rites_checked(state, "Si potrà chiedere il patteggiamento. Altro.", M.charge_rites({}, [], IMPUTATO),
                             posture=IMPUTATO)
    assert out == "Altro." and state["ledger"]["removed_by_step"].get("rites_out") == 1
    # the method's ctx for a client who is not the imputato before the sentence carries no statuses: nothing goes
    state = {}
    assert flow.rites_checked(state, text, M.charge_rites({}, [], IMPUTATO), method_ctx={"rites": []},
                              posture=IMPUTATO) == text


def test_a_background_death_is_no_event_and_the_victim_always_keeps_a_main_offence():
    from src.rag import lawyer_method as M
    background = ("Il padre di Mario, morto nel 2015, gli aveva lasciato la casa. Il 3 marzo 2026 Luca colpiva Mario con "
                  "un pugno, causandogli una frattura al naso con prognosi di 30 giorni.")
    assert M.event_of(background) == "lesioni"
    assert M.event_of("Il 3 marzo 2026 Tizio investiva Caio, morto nel 2026 dopo il ricovero.") == "morte"
    stab = ("Luca colpiva Mario con un coltello all'addome, causandogli lesioni con prognosi di 40 giorni. "
            "Mario veniva ricoverato.")
    assert M.event_of(stab) == "lesioni"
    recs = {"575 c.p.": {"rubrica": "Omicidio", "side": "weapon", "decomposed_for": "x",
                         "requisites": [_req(1, "atti idonei diretti in modo non equivoco a uccidere")]}}
    ctx = {"records": recs, "circumstances": {}, "posture": _posture("persona_offesa", "pre_filing"),
           "case": stab, "assessment": {"article": ""}, "institutions": [], "certain_text": stab}
    kinds = {c["kind"]: c.get("article") for c in M.thesis_candidates(ctx)}
    assert kinds.get("claim_offence") == "575 c.p." and "parte_civile" in kinds


def test_dropping_a_rito_leaves_no_orphan_number_nor_an_empty_cell():
    from src.rag import lawyer_method as M
    unknown = M.rite_statuses("reclusione non inferiore ad anni ventuno (co. 1)", "575")    # v79.10.7: abbreviato admitted
    text = ("1. Contestare il dolo.\n2. Patteggiamento (art. 444 c.p.p.) con pena contenuta.\n3. Giudizio abbreviato.\n\n"
            "| Opzione | Esito |\n|---|---|\n| Messa alla prova | ammessa |\n| Abbreviato | riduzione |")
    out = M.rites_out(text, unknown)
    assert out.split("\n") == ["1. Contestare il dolo.", "3. Giudizio abbreviato.", "", "| Opzione | Esito |",
                               "|---|---|", "| Abbreviato | riduzione |"]


def test_the_prose_paths_alternatives_section_never_prints_one_the_event_does_not_fit():
    from src.rag import interactive_flow as flow
    alternatives = [({"name": "lesioni personali colpose"}, {"_article": ("590", "c.p."),
                                                            "rubrica": "Lesioni personali colpose"}),
                    ({"name": "omicidio colposo"}, {"_article": ("589", "c.p."), "rubrica": "Omicidio colposo"})]
    written = ("[1] SÌ: Il fatto può essere riqualificato come lesioni personali colpose.\n"
               "[2] SÌ: La morte deriva dalla colpa del conducente.")
    log = {}
    text = flow.alternatives_text(written, alternatives, log, "morte")
    assert "590" not in text and "Art. 589 c.p." in text and log["refused_event"]["articles"] == ["590 c.p."]
    assert flow.alternatives_text(written, alternatives) .count("**Art.") == 2           # no event: as before
    rejected = "[1] NO: La vittima è deceduta, non si tratta di lesioni.\n[2] SÌ: La morte deriva dalla colpa."
    assert "Art. 590 c.p." in flow.alternatives_text(rejected, alternatives, {}, "morte")    # the writer's NO stays
    state = {}
    flow.alternatives_event_counted(state, text, written, alternatives, "morte")
    flow.alternatives_event_counted(state, flow.alternatives_text(rejected, alternatives, {}, "morte"), rejected,
                                    alternatives, "morte")
    assert state["block_counters"]["alternatives_event_mismatch"] == 0
    flow.alternatives_event_counted(state, flow.alternatives_text(written, alternatives), written, alternatives,
                                    "morte")
    assert flow.case_counters(state)["requalify_event_mismatch"] == 1            # an offer printed is counted


def test_the_outcome_names_only_an_alternative_the_event_fits():
    from src.rag import lawyer_method as M
    recs = {"575 c.p.": {"rubrica": "Omicidio", "side": "threat", "charged": True, "decomposed_for": "x",
                         "requisites": [_req(1, "cagiona la morte di un uomo")]},
            "590 c.p.": {"rubrica": "Lesioni personali colpose", "side": "weapon", "alternative": True,
                         "decomposed_for": "x", "requisites": [_req(2, "per colpa cagiona una lesione")]}}
    a = M.assessment(recs, IMPUTATO, event=M.event_of(DEATH))
    assert "590" not in a["why"] and not a.get("alternative") and a["obiettivo"] == M.CONTENIMENTO
    assert M.assessment(recs, IMPUTATO)["alternative"] == "590 c.p."               # no event: as before
    ctx = {"records": recs, "certain_text": DEATH, "candidates": [], "assessment": {"alternative": "590 c.p."}}
    assert M.selection_counts(ctx, {"soluzioni": []})["requalify_event_mismatch"] == 1


@pytest.mark.parametrize("sentence, offered", [
    ("Conviene chiedere il patteggiamento, che non può essere revocato dopo la ratifica.", True),
    ("Conviene chiedere il patteggiamento, il cui termine di proposizione è da verificare.", True),
    ("Conviene il patteggiamento, mentre la messa alla prova è esclusa.", True),
    ("Il patteggiamento, che la difesa potrebbe chiedere, è precluso dalla pena.", False),
    ("L'imputato non può chiedere il patteggiamento.", False),
    ("Il patteggiamento non è ammesso per questo reato.", False),
])
def test_only_a_negation_of_the_ritos_access_says_it_is_not_offered(sentence, offered):
    from src.rag import lawyer_method as M
    statuses = M.rite_statuses("reclusione non inferiore ad anni ventuno (co. 1)", "575")
    assert bool(M.rites_offered_unavailable(sentence, statuses)) is offered
    unknown = M.rite_statuses("", "")
    assert M.rites_offered_unavailable("Si può chiedere il patteggiamento, ma la data dell'udienza è da verificare.",
                                       unknown)
    assert not M.rites_offered_unavailable(f"Patteggiamento: {M.RITE_TO_VERIFY}.", unknown)


def test_remissione_needs_a_querela_also_with_no_charge_and_on_the_prose_path():
    from src.rag import lawyer_method as M, procedural_facts as pf
    text = "Un accordo può portare alla remissione della querela."
    assert M.procedibility_out(text, {}, "", penal=True) == ""
    assert M.procedibility_out(text, {"572 c.p.": "ufficio"}, "", penal=True) == ""
    assert M.procedibility_out(text, {}, "") == text                              # outside a criminal matter
    offer = "Il risarcimento integrale del danno può estinguere il reato per condotte riparatorie (art. 162-ter c.p.)."
    assert pf._conflict(offer, {"procedibilita": "d'ufficio", "querela": False})
    assert pf._conflict(offer, {"procedibilita": "a querela", "querela": True,
                                "remissione": "querela irrevocabile (co. 4)"})
    assert pf._conflict(offer, {"procedibilita": "a querela", "querela": True, "remissione": "remissibile"}) == ""
    denied = "La remissione della querela non è ammessa: il reato è procedibile d'ufficio."
    assert pf._conflict(denied, {"procedibilita": "d'ufficio", "querela": False}) == ""


def test_444_is_judged_on_the_best_case_of_the_pena_in_concreto():
    """v79.10.7 (the user's ruling): the minimum less each attenuante the record holds as integrato or da provare, then
    the rite's third; precluded only when even that exceeds five years, otherwise «possibile, dipende dalla pena in
    concreto» (never «ammesso»)."""
    from src.rag import lawyer_method as M
    heavy = _by_id(M.rite_statuses("reclusione da dodici a venti anni (co. 1)", "999", ["62-bis c.p."]))["patteggiamento"]
    assert heavy["status"] == M.PRECLUDED and "62-bis" in heavy["line"] and "oltre 5 anni" in heavy["line"]
    ok = _by_id(M.rite_statuses("reclusione da sette a quindici anni (co. 1)", "999"))["patteggiamento"]
    assert ok["status"] == M.POSSIBLE and "possibile, dipende dalla pena in concreto" in ok["line"]
    assert "4 anni e 8 mesi" in ok["line"] and "ammess" not in ok["line"]
