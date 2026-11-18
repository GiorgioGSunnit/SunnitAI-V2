"""v79.10.7 -- the user's rulings on 79.10.3 (C1) and 79.10.4 (D1), on synthetic sentences, records and store entries only:

1. patteggiamento (art. 444 c.p.p.): judged on the best case of the pena in concreto -- the edittale minimum less each
   attenuante the record holds as integrato or da provare (62-bis and the common ones up to a third, art. 65 n. 3 c.p.;
   one of special effect first, art. 63, co. 3; never below a quarter with two or more, art. 67, co. 2), then the
   rite's third; precluded only when even that exceeds five years, otherwise «possibile, dipende dalla pena in
   concreto», never «ammesso»; the pena unknown, «da verificare sulla pena edittale»
2. abbreviato: excluded for an offence punished with the ergastolo (art. 438, co. 1-bis, c.p.p.), one lookup on the
   store's pena; the pena unknown, «da verificare sulla pena edittale»
3. the 168-bis list (art. 550, co. 2, c.p.p.): a draft for the lawyer, so labelled in the data file, cited as provisional
4. D1: a span counted to today («tre mesi fa») dates an event, not the act the term governs -- with the act undated the
   term is «da provare», never «escluso»; both dated, graded as before
"""

import os

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import issue_records as I  # noqa: E402
from src.rag import lawyer_method as M  # noqa: E402
from src.rag import posture  # noqa: E402

PROVATO, DA_PROVARE, NON_RISULTA = flow.PROVATO, flow.DA_PROVARE, flow.NON_RISULTA


def _posture(role="imputato", stage="primo_grado"):
    return {"rito": "penale", "family": "penale", "role": role, "stage": stage,
            "entry": posture.entry_for("penale", role, stage)}


IMPUTATO = _posture()


def _row(number, pena):
    return {"_article": (number, "c.p."), "s": {"plain_text": f"Chiunque commette il fatto ({number})."},
            "_proc": {"version": 4, "pena": pena, "querela": False}}


def _charge(label, rubrica="Reato contestato"):
    return {label: {"rubrica": rubrica, "side": "threat", "charged": True, "decomposed_for": "addebito",
                    "requisites": [{"id": "R1", "comma": 1, "kind": "requisito", "text": "la condotta",
                                    "status": PROVATO}]}}


def _attenuante(label, status, sup=("S1",)):
    """A synthetic circumstance record (circumstance_records' shape) with every element at the status given, each
    supported by the sentence ids given (a fact of the case: the record holds it); sup=() an element no fact touches."""
    table = {c["label"]: c for c in M.data()["circumstances"]}
    rec = M.circumstance_records([table[label]])[label]
    for r in rec["requisites"]:
        r["status"] = status
        r["supporting"] = list(sup)
    return {label: rec}


def _by_id(statuses):
    return {x["id"]: x for x in statuses}


# --- 1. patteggiamento: the best case ------------------------------------------------------------------------------------

def test_584_stays_precluded_without_an_attenuante_held():
    """584 c.p. (10-18 years): with no attenuante held the best case is 10 years less the rite's third, 6 years and 8
    months -- precluded, as under the old rule (the minimum less a third)."""
    got = _by_id(M.charge_rites(_charge("584 c.p.", "Omicidio preterintenzionale"),
                                [_row("584", "reclusione da dieci a diciotto anni (co. 1)")], IMPUTATO))
    pat = got["patteggiamento"]
    assert pat["status"] == M.PRECLUDED and pat["why"] == "min" and pat["attenuanti"] == []
    assert "6 anni e 8 mesi" in pat["line"] and "senza attenuanti integrate o da provare" in pat["line"]
    assert "anche nel caso più favorevole" in pat["line"] and "oltre 5 anni" in pat["line"]


def test_an_attenuante_held_da_provare_turns_the_old_precluso_into_possibile():
    """8-20 years: the old rule (the minimum less a third, attenuanti ignored) gives 5 years and 4 months, precluded; the
    62-bis held «da provare» (a fact of the case supports it) brings the best case to 8 x 2/3 x 2/3 = 3 years, 6
    months and 20 days."""
    rows = [_row("999", "reclusione da otto a venti anni (co. 1)")]
    recs = _charge("999 c.p.")
    assert _by_id(M.charge_rites(recs, rows, IMPUTATO))["patteggiamento"]["status"] == M.PRECLUDED
    circ = _attenuante("62-bis c.p.", DA_PROVARE)
    assert M.held_attenuanti(circ) == ["62-bis c.p."]
    pat = _by_id(M.charge_rites(recs, rows, IMPUTATO, circumstances=circ))["patteggiamento"]
    assert pat["status"] == M.POSSIBLE and pat["status"] != M.AVAILABLE
    assert pat["line"].startswith("Applicazione della pena su richiesta (art. 444 c.p.p.): possibile, dipende dalla "
                                  "pena in concreto")
    assert "62-bis" in pat["line"] and "3 anni, 6 mesi e 20 giorni" in pat["line"]
    # integrato counts as well
    assert _by_id(M.charge_rites(recs, rows, IMPUTATO, circumstances=_attenuante("62-bis c.p.", PROVATO))
                  )["patteggiamento"]["status"] == M.POSSIBLE


def test_an_attenuante_held_as_non_risulta_is_not_counted():
    rows = [_row("999", "reclusione da otto a venti anni (co. 1)")]
    circ = _attenuante("62-bis c.p.", NON_RISULTA)
    assert M.held_attenuanti(circ) == []
    pat = _by_id(M.charge_rites(_charge("999 c.p."), rows, IMPUTATO, circumstances=circ))["patteggiamento"]
    assert pat["status"] == M.PRECLUDED and "senza attenuanti" in pat["line"]
    # 584 with the 62-bis escluso: still precluded
    pat = _by_id(M.charge_rites(_charge("584 c.p."), [_row("584", "reclusione da dieci a diciotto anni (co. 1)")],
                                IMPUTATO, circumstances=circ))["patteggiamento"]
    assert pat["status"] == M.PRECLUDED
    # one element «non risulta» excludes the circumstance: provocazione with its state of anger not found
    prov = _attenuante("62 n. 2 c.p.", DA_PROVARE)
    prov["62 n. 2 c.p."]["requisites"][0]["status"] = NON_RISULTA
    assert M.held_attenuanti(prov) == []
    # an aggravante is never an attenuante held
    agg = {"61 n. 1 c.p.": {**_attenuante("62-bis c.p.", PROVATO)["62-bis c.p."], "kind": "aggravante"}}
    assert M.held_attenuanti(agg) == []


def test_each_attenuante_held_is_applied_in_turn():
    """Two common attenuanti: each up to a third on the pena the previous left (art. 63 c.p.); 62-bis counts once."""
    got = M.best_case(12, ["62-bis c.p.", "62 n. 6 c.p.", "62-bis c.p."])
    assert got["applied"] == ["62-bis c.p.", "62 n. 6 c.p."]
    assert got["after_circumstances"] == pytest.approx(12 * 2 / 3 * 2 / 3)
    assert got["best"] == pytest.approx(12 * 2 / 3 * 2 / 3 * 2 / 3)
    circ = {**_attenuante("62-bis c.p.", DA_PROVARE), **_attenuante("62 n. 6 c.p.", PROVATO)}
    pat = _by_id(M.charge_rites(_charge("999 c.p."), [_row("999", "reclusione da dodici a venti anni (co. 1)")],
                                IMPUTATO, circumstances=circ))["patteggiamento"]
    assert pat["status"] == M.POSSIBLE and "le attenuanti degli artt. 62-bis e 62 n. 6 c.p." in pat["line"]


def test_a_special_effect_attenuante_goes_first_and_the_quarter_floor_holds(monkeypatch):
    """A synthetic table: an attenuante of special effect (half) applied before the common one (art. 63, co. 3, c.p.);
    with two or more, never below a quarter of the minimum (art. 67, co. 2, c.p.)."""
    table = {"circumstances": [{"label": "X c.p.", "kind": "attenuante", "max_reduction": "1/2", "effect": "speciale"},
                               {"label": "62-bis c.p.", "kind": "attenuante", "max_reduction": "1/3",
                                "effect": "comune"},
                               {"label": "Y c.p.", "kind": "attenuante", "max_reduction": "4/5",
                                "effect": "speciale"}]}
    monkeypatch.setattr(M, "data", lambda: table)
    assert [x["label"] for x in M.attenuante_measures(["62-bis c.p.", "X c.p."])] == ["X c.p.", "62-bis c.p."]
    assert M.best_case(9, ["62-bis c.p.", "X c.p."])["after_circumstances"] == pytest.approx(9 / 2 * 2 / 3)
    floored = M.best_case(20, ["Y c.p.", "62-bis c.p."])
    assert floored["after_circumstances"] == pytest.approx(5.0)                    # 20 x 1/5 x 2/3 < 20 / 4
    assert M.best_case(20, ["Y c.p."])["after_circumstances"] == pytest.approx(4.0)   # one alone: no floor


def test_the_unknown_pena_stays_da_verificare_and_the_ergastolo_precluded():
    unknown = _by_id(M.rite_statuses("", "999", ["62-bis c.p."]))["patteggiamento"]
    assert unknown["status"] == M.TO_VERIFY_RITE and M.RITE_TO_VERIFY in unknown["line"]
    life = _by_id(M.rite_statuses("ergastolo (co. 1)", "999", ["62-bis c.p."]))["patteggiamento"]
    assert life["status"] == M.PRECLUDED and "ergastolo" in life["line"]


def test_possibile_is_admitted_for_the_sweep_and_said_in_its_own_words():
    statuses = M.rite_statuses("reclusione da tre a sette anni (co. 1)", "572")
    assert _by_id(statuses)["patteggiamento"]["status"] == M.POSSIBLE
    text = "Si potrà chiedere il patteggiamento."
    assert M.rites_offered_unavailable(text, statuses) == [] and M.rites_out(text, statuses) == text
    assert any("444" in x for x in M.riti_lines(IMPUTATO, "reclusione da tre a sette anni (co. 1)", "572"))
    # the plan's subordinate line naming the patteggiamento is the rito's own line, never the bare offer
    plan = posture.apply_plan({"subordinate": []}, IMPUTATO, statuses)
    pat = [x for x in plan["subordinate"] if M.rite_of(x) == "patteggiamento"]
    assert pat and all("possibile, dipende dalla pena in concreto" in x for x in pat)


CASE_584 = "Tizio è imputato di omicidio preterintenzionale: ha spinto la vittima, che è caduta ed è morta."
ROW_584 = _row("584", "reclusione da dieci a diciotto anni (co. 1)")


def test_584_stays_precluded_on_the_imputatos_real_paths():
    """(review) The imputato's posture always lists the 62-bis as a candidate: ungraded, or graded with no fact of the
    case for it, the record does not hold it -- 584 c.p. precluded on the method's path and on the flow's."""
    recs = _charge("584 c.p.", "Omicidio preterintenzionale")
    circ = M.circumstance_records(M.circumstance_candidates(CASE_584, recs, IMPUTATO))
    assert list(circ) == ["62-bis c.p."] and M.held_attenuanti(circ) == []
    pat = _by_id(M.charge_rites(recs, [ROW_584], IMPUTATO, circumstances=circ))["patteggiamento"]
    assert pat["status"] == M.PRECLUDED and "6 anni e 8 mesi" in pat["line"]
    state = {"text": CASE_584, "request": "", "records": recs}
    flow_pat = _by_id(flow.charge_rites(state, IMPUTATO, [ROW_584]))["patteggiamento"]
    assert flow_pat["status"] == M.PRECLUDED and flow_pat["attenuanti"] == []
    plan = posture.apply_plan({"subordinate": []}, IMPUTATO, flow.charge_rites(state, IMPUTATO, [ROW_584]))
    assert not [x for x in plan["subordinate"] if M.rite_of(x) == "patteggiamento"]
    # a fact of the case for the generiche (graded «da provare» on it): the record holds it, the best case decides
    held = _attenuante("62-bis c.p.", DA_PROVARE)
    got = _by_id(M.charge_rites(recs, [ROW_584], IMPUTATO, circumstances=held))
    assert got["patteggiamento"]["status"] == M.POSSIBLE
    # graded with no supporting sentence: still not held
    bare = _attenuante("62-bis c.p.", DA_PROVARE, sup=())
    assert M.held_attenuanti(bare) == []
    bare["62-bis c.p."]["requisites"][0]["link"] = {"supporting": ["S2"], "contradicting": []}
    assert M.held_attenuanti(bare) == ["62-bis c.p."]


def test_the_plan_and_section_8_take_the_methods_graded_statuses():
    """(review) The plan's subordinate lines and §8 (computed before the method grades the circumstances) are replaced
    by the method's statuses -- either way: the method holding the generiche, or excluding them."""
    rows = [_row("999", "reclusione da otto a venti anni (co. 1)")]
    recs = _charge("999 c.p.")
    flow_rites = M.charge_rites(recs, rows, IMPUTATO)                                  # nothing held: precluded
    held = M.charge_rites(recs, rows, IMPUTATO, circumstances=_attenuante("62-bis c.p.", DA_PROVARE))
    plan = posture.apply_plan({"subordinate": ["Linea del modello."]}, IMPUTATO, flow_rites)
    assert not [x for x in plan["subordinate"] if M.rite_of(x) == "patteggiamento"]
    new_plan, rites, analysis_p = flow.method_rites(plan, IMPUTATO, flow_rites, {"rites": held}, {"rites": flow_rites})
    pat = [x for x in new_plan["subordinate"] if M.rite_of(x) == "patteggiamento"]
    assert pat and all("possibile, dipende dalla pena in concreto" in x for x in pat)
    assert new_plan["subordinate"][0] == "Linea del modello." and rites == held and analysis_p["rites"] == held
    # the reverse: the plan said «possibile», the method graded the 62-bis «non risulta» -- the line goes
    excluded = M.charge_rites(recs, rows, IMPUTATO, circumstances=_attenuante("62-bis c.p.", NON_RISULTA))
    back, _, back_p = flow.method_rites(new_plan, IMPUTATO, held, {"rites": excluded}, analysis_p)
    assert not [x for x in back["subordinate"] if M.rite_of(x) == "patteggiamento"]
    assert _by_id(back_p["rites"])["patteggiamento"]["status"] == M.PRECLUDED
    # the method computed none (not the imputato's answer) or the same: unchanged
    assert flow.method_rites(plan, IMPUTATO, flow_rites, {"rites": []}, {"rites": flow_rites})[0] is plan


def test_a_rito_only_possibile_is_never_written_as_simply_admitted():
    """(review) «possibile, dipende dalla pena in concreto» is never «ammesso»: a sentence saying so without a condition
    is dropped (and counted); one with the condition, or a plain offer, is kept."""
    statuses = M.rite_statuses("reclusione da otto a venti anni (co. 1)", "999", ["62-bis c.p."])
    assert _by_id(statuses)["patteggiamento"]["status"] == M.POSSIBLE
    bare = "In via subordinata, il patteggiamento (art. 444 c.p.p.) è ammesso e consente di definire il processo."
    assert M.rites_offered_unavailable(bare, statuses) and M.rites_out(bare + " Altro.", statuses) == "Altro."
    for kept in ("Il patteggiamento è ammesso se la pena in concreto, con le generiche, resta entro cinque anni.",
                 "Si potrà chiedere il patteggiamento.",
                 _by_id(statuses)["patteggiamento"]["line"]):
        assert M.rites_offered_unavailable(kept, statuses) == [] and M.rites_out(kept, statuses) == kept
    # a rito simply admitted (the abbreviato here): «ammesso» stays
    assert M.rites_out("Il giudizio abbreviato è ammesso.", statuses) == "Il giudizio abbreviato è ammesso."


# --- 2. abbreviato: art. 438, co. 1-bis, c.p.p. --------------------------------------------------------------------------

def test_the_abbreviato_is_excluded_for_an_offence_punished_with_the_ergastolo():
    life = _by_id(M.charge_rites(_charge("577 c.p.", "Omicidio aggravato"), [_row("577", "ergastolo (co. 1)")],
                                 IMPUTATO))["abbreviato"]
    assert life["status"] == M.PRECLUDED and life["why"] == "ergastolo"
    assert life["line"] == ("Giudizio abbreviato (artt. 438 ss. c.p.p.): precluso, il reato contestato è punito con "
                            "l'ergastolo (art. 438, co. 1-bis, c.p.p.).")
    assert life["line"] not in M.riti_lines(IMPUTATO, "ergastolo (co. 1)", "577")
    assert life["line"] in M.riti_precluded(IMPUTATO, "ergastolo (co. 1)", "577")
    # an offer of it is dropped from the written answer
    statuses = M.rite_statuses("ergastolo (co. 1)", "577")
    assert M.rites_out("Si può chiedere il giudizio abbreviato. Altro.", statuses) == "Altro."
    # a non-ergastolo omicidio: admitted
    plain = _by_id(M.charge_rites(_charge("575 c.p.", "Omicidio"),
                                  [_row("575", "reclusione non inferiore ad anni ventuno (co. 1)")], IMPUTATO))
    assert plain["abbreviato"]["status"] == M.AVAILABLE and "riduzione di un terzo" in plain["abbreviato"]["line"]


def test_the_abbreviato_on_an_unknown_pena_is_da_verificare():
    got = _by_id(M.charge_rites(_charge("73 d.P.R. 309/1990"), [], IMPUTATO))["abbreviato"]
    assert got["status"] == M.TO_VERIFY_RITE
    assert got["line"] == ("Giudizio abbreviato (artt. 438 ss. c.p.p.): da verificare sulla pena edittale del reato "
                           "contestato (non è ammesso per i delitti puniti con l'ergastolo, art. 438, co. 1-bis, "
                           "c.p.p.).")
    entry = next(r for r in M.data()["riti_alternativi"] if r["id"] == "abbreviato")
    assert entry["excluded_ergastolo"] is True and "always" not in entry


# --- 3. the 168-bis list: a draft for the lawyer ----------------------------------------------------------------------------

def test_the_listed_offences_are_labelled_a_draft_and_cited_as_provisional():
    entry = next(r for r in M.data()["riti_alternativi"] if r["id"] == "messa_alla_prova")
    assert entry["status"].startswith("bozza") and "avvocato" in entry["status"] and "Cartabia" in entry["status"]
    assert entry["listed_offences"] == ["336", "337", "343", "349", "588", "590-bis", "625", "648"]   # unchanged
    listed = _by_id(M.rite_statuses("reclusione da due a otto anni (co. 1)", "648"))["messa_alla_prova"]["line"]
    assert "bozza" in listed and "550, co. 2" in listed
    precluded = _by_id(M.rite_statuses("reclusione da tre a sette anni (co. 1)", "572"))["messa_alla_prova"]["line"]
    assert "preclusa" in precluded and "bozza" in precluded and "non figura" in precluded


# --- 4. D1: a span counted to today ----------------------------------------------------------------------------------------

TERM = "la denunzia va fatta entro sessanta giorni dalla scoperta"


def _settled(text, case, sup=()):
    sents = I.sentences(case)
    req = {"kind": "requisito", "text": text, "facts_status": DA_PROVARE,
           "link": {"supporting": list(sup), "contradicting": []}}
    I.settle(req, {s["id"]: s for s in sents})
    return req


def test_a_span_to_today_is_marked():
    assert [x["ago"] for x in I.spans_of("La scoperta è avvenuta tre mesi fa.")] == [True]
    assert [x["ago"] for x in I.spans_of("Il vizio è stato scoperto oltre un anno fa.")] == [True]
    assert [x["ago"] for x in I.spans_of("Ha denunciato i vizi dopo tre mesi dalla scoperta.")] == [False]


@pytest.mark.parametrize("case", [
    "La scoperta è avvenuta tre mesi fa.",
    "La scoperta è avvenuta tre mesi fa. Il compratore ha poi denunciato il vizio.",
    "Il vizio è stato scoperto oltre un anno fa.",
    "La scoperta del vizio è avvenuta tre mesi fa. Un mese fa il compratore è stato ricoverato.",
])
def test_the_discovery_dated_to_today_with_the_act_undated_is_da_provare(case):
    req = _settled(TERM, case, sup=["S1"])
    assert req["status"] == DA_PROVARE and req["why"] == I.TERMINE_OPEN_WHY and req["termine"]["verdict"] == ""


def test_a_span_to_today_never_grades_a_term_with_no_anchor():
    assert _settled("l'azione si prescrive entro un anno", "Il contratto è stato concluso tre anni fa.",
                    sup=["S1"])["status"] == DA_PROVARE


def test_both_the_start_and_the_act_dated_grade_normally():
    # calendar dates
    beyond = _settled(TERM, "Il 1 marzo 2024 il compratore ha scoperto il vizio. Il 20 giugno 2024 il compratore ha "
                            "denunciato il vizio.")
    assert beyond["status"] == NON_RISULTA and beyond["termine"]["span"] == "dal 01/03/2024 al 20/06/2024"
    inside = _settled(TERM, "Il 1 marzo 2024 il compratore ha scoperto il vizio. Il 20 marzo 2024 il compratore ha "
                            "denunciato il vizio.")
    assert inside["status"] == PROVATO
    # both counted to today: their difference is the duration
    late = _settled(TERM, "La scoperta del vizio è avvenuta tre mesi fa. Il compratore ha denunciato il vizio dieci "
                          "giorni fa.")
    assert late["status"] == NON_RISULTA and late["termine"]["span"] == "tre mesi fa – dieci giorni fa"
    on_time = _settled(TERM, "Il vizio è stato scoperto tre mesi fa e denunciato due mesi fa.")
    assert on_time["status"] == PROVATO and on_time["termine"]["verdict"] == I.INSIDE


def test_a_start_to_today_inside_the_term_with_the_act_done_is_inside():
    """The act stated as done by today lies within the span: within the term, it was on time (an upper bound, never
    «beyond»)."""
    req = _settled(TERM, "La scoperta è avvenuta cinquanta giorni fa. Il compratore ha poi denunciato il vizio.")
    assert req["status"] == PROVATO and req["termine"]["span"] == "cinquanta giorni fa"
    assert _settled(TERM, "La scoperta è avvenuta cinquanta giorni fa.")["status"] == DA_PROVARE    # act not named
    # the start's longest calendar reading beyond the term: nothing («due mesi» may be 62 days)
    assert _settled(TERM, "La scoperta è avvenuta due mesi fa. Il compratore ha denunciato il vizio.")["status"] \
        == DA_PROVARE


ELEMENT = "il compratore deve denunziare il vizio entro sessanta giorni dalla scoperta"


@pytest.mark.parametrize("case", [
    "La scoperta è avvenuta tre mesi fa. Il vizio è stato riparato dieci giorni fa.",
    "La scoperta è avvenuta tre mesi fa. Il vizio si è aggravato un mese fa.",
    "La scoperta è avvenuta tre mesi fa, il vizio è ricomparso venti giorni fa.",
])
def test_an_event_of_the_acts_object_is_no_act(case):
    """(review) The act is the verb («denunziare»), never its object: an event about the vizio dates no denunzia."""
    req = _settled(ELEMENT, case, sup=["S1"])
    assert req["status"] == DA_PROVARE and req["termine"]["verdict"] == ""


@pytest.mark.parametrize("element,case", [
    (ELEMENT, "La scoperta è avvenuta cinquanta giorni fa. Il vizio era evidente."),
    (TERM, "La scoperta è avvenuta cinquanta giorni fa. Il compratore non ha mai denunciato il vizio."),
    (TERM, "La scoperta è avvenuta cinquanta giorni fa. Il compratore non ha ancora denunciato il vizio."),
    (TERM, "La scoperta è avvenuta cinquanta giorni fa. Il compratore vuole denunciare il vizio."),
    (TERM, "La scoperta è avvenuta cinquanta giorni fa. La denuncia è pronta."),
])
def test_the_start_dated_with_no_act_done_is_da_provare(element, case):
    """(review) Only the start dated, the act not stated as done (absent, denied, not yet done, intended): no
    duration -- never «integrato» on the start's span alone."""
    req = _settled(element, case, sup=["S1"])
    assert req["status"] == DA_PROVARE and req["termine"]["verdict"] == ""


def test_the_act_done_named_by_its_verb_is_the_upper_bound():
    assert _settled("il vizio è denunziato entro sessanta giorni dalla scoperta",
                    "Mario ha scoperto il vizio cinquanta giorni fa e lo ha denunciato subito.")["status"] == PROVATO
    assert _settled(TERM, "La scoperta è avvenuta cinquanta giorni fa. Il compratore denunciò il vizio.")["status"] \
        == PROVATO
    assert I.act_stems("il compratore deve denunziare il vizio ") == {"denun"}
    assert I.act_stems("la denunzia va fatta ") == {"denun"}
    assert I.act_stems("il vizio deve essere denunziato ") == {"denun"}
    assert I.act_stems("il ricorso va proposto ", "notif") == {"ricor", "propo"}


@pytest.mark.parametrize("element,case", [
    (TERM, "La scoperta del vizio è avvenuta tre mesi fa. Il compratore ha denunciato il vizio un mese fa."),
    ("il ricorso va proposto entro sessanta giorni dalla notifica",
     "La notifica è avvenuta tre mesi fa. Il ricorso è stato depositato un mese fa."),
])
def test_two_spans_to_today_that_round_across_the_term_are_da_provare(element, case):
    """(review) «tre mesi fa – un mese fa» is 53 to 65 days on the calendar: against sixty days neither inside nor
    beyond -- «da provare», never «escluso» by a month's rounding."""
    req = _settled(element, case)
    assert req["status"] == DA_PROVARE and req["termine"]["verdict"] == ""
    # well past the term on every reading: beyond
    late = _settled(element, case.replace("un mese fa", "dieci giorni fa"))
    assert late["status"] == NON_RISULTA and late["termine"]["verdict"] == I.BEYOND


def test_dated_events_with_the_act_undated_never_exclude():
    case = "Il vizio è stato scoperto il 1 marzo 2024. Il 5 dicembre 2024 è arrivata una lettera."
    req = _settled("il diritto si esercita entro sessanta giorni dalla scoperta", case)
    assert req["status"] == DA_PROVARE and req["termine"]["verdict"] == ""


def test_dated_events_with_the_act_undated_never_integrate():
    """(review) An unrelated later date inside the term is no act either: «da provare», never «integrato»."""
    case = "Il vizio è stato scoperto il 1 marzo 2024. Il 10 marzo 2024 è arrivata una lettera."
    req = _settled("il diritto si esercita entro sessanta giorni dalla scoperta", case)
    assert req["status"] == DA_PROVARE and req["termine"]["verdict"] == ""
