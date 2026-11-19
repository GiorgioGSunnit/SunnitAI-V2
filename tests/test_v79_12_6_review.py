"""v79.12.7, the review's findings on items 1-7, on synthetic sentences, records, passages and article texts only:

1. a passage (or a note) saying the jurisprudential element is NOT required is no source of it.
2. the provocazione is triggered by a diverbio, never by «persona offesa» nor by «provocò la caduta»; 62 n. 6's step is
   the offender's (the parte civile's claim «da quantificare» is none); the «esclusa» prose check reads each clause
   (another institute's exclusion stays; two circumstances in one sentence, each its own), more ways of excluding, and
   an aggravante the record does not hold is never raised against the client.
3. the soccorso is the offender's: the place «pronto soccorso», help by passers-by or a neighbour, help refused -- 593
   stays and nothing favourable is printed; the flow's records lose the absorbed titles too, and the counter reads them.
4. a fact that denies the push («nega di aver dato alcuna spinta») or a motive («spinto dalla rabbia») bears on nothing.
5. a trigger fact the facts deny («non soffriva di alcuna patologia») lifts no cap; a capped causation line never leads
   while the containment line is eligible.
6. a written sentence on a computed rito: a figure the computation does not print is dropped and counted, a result
   asserted with no figure too; the code's own precluded computation is never an «offer»; the PM's consent said with
   the precluded computation for a listed offence; a contravvenzione's abbreviato halves.
7. (a) ranges, pairs and a parenthesis of codes, a stance heading a citation, a text with no known code; (b) the
   headless reference's «del», «udienza», «Rv. … - 01», «sent. n.» tails; (c) the iscrizione nel registro, the capo
   d'imputazione and the PM's charge certain, a hypothetical act («potrebbe contestare») not; (f) a cenno inverting or
   absolutising art. 41's synthetic text is not grounded.
"""

import os
import sys

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import issue_records as IR  # noqa: E402
from src.rag import lawyer_method as M  # noqa: E402
from src.rag import penal_lines as P  # noqa: E402
from src.rag import rulings  # noqa: E402
from src.rag import text_hygiene as T  # noqa: E402
import test_v79_12_6 as A  # noqa: E402
import test_v79_12_6_b as B  # noqa: E402


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("THESIS_DETERMINISTIC", "")
    from src.rag import plan_then_write
    monkeypatch.setattr(plan_then_write, "negation_review", lambda items, known, call=None: [])


def _s(*texts):
    return [{"id": f"S{i}", "text": t, "source": "caso", "certainty": "accertato"} for i, t in enumerate(texts, 1)]


RULE_593 = next(r for r in P.data()["absorption"] if r["mode"] == "post_fact")
JURIS = P.data()["jurisprudential_elements"][0]
# a synthetic art. 41-style text, two commi (no real source)
T41 = {"41 c.p.": "1. Il concorso di cause preesistenti o simultanee o sopravvenute, anche se indipendenti dall'azione od "
                  "omissione del colpevole, non esclude il rapporto di causalità fra l'azione od omissione e l'evento.\n"
                  "2. Le cause sopravvenute escludono il rapporto di causalità quando sono state da sole sufficienti "
                  "a determinare l'evento."}


# --- 1. a passage denying the element ------------------------------------------------------------------------------------

@pytest.mark.parametrize("passage", [
    "Ai fini dell'omicidio preterintenzionale non è richiesta la prevedibilità in concreto dell'evento morte, bastando "
    "il nesso causale con gli atti diretti a percuotere.",
    "La prevedibilità in concreto dell'evento morte non è necessaria per l'omicidio preterintenzionale.",
    "Il reato sussiste indipendentemente dalla prevedibilità in concreto dell'evento morte.",
])
def test_a_passage_denying_the_element_adds_nothing(passage):
    assert not P.states_element(JURIS, passage)
    ctx = A._ctx(A._records(), A.SLAP)
    ctx["rulings_record"] = A._record_with_passage(passage)
    assert M.ruling_elements(ctx) == []
    assert not any(r.get("jurisprudential") for r in ctx["records"]["584 c.p."]["requisites"])


def test_the_stating_passage_still_adds_it_and_a_denying_note_does_not():
    assert P.states_element(JURIS, A.PASSAGE)
    recs = A._records()
    recs["584 c.p."]["note_di_inquadramento"] = "Non occorre la prevedibilità in concreto dell'evento morte."
    assert P.jurisprudential_elements(recs) == []


# --- 2. the attenuanti's triggers and the «esclusa» check ------------------------------------------------------------------

def test_persona_offesa_and_a_caused_fall_trigger_no_provocazione():
    for text in ("Gamma colpisce Delta, persona offesa, con uno schiaffo.", "La persona offesa sporge querela.",
                 "Lo schiaffo provocò la caduta di Delta.", "Le parti offese chiedono il risarcimento."):
        assert "62 n. 2 c.p." not in P.circumstance_triggers(_s(text)), text
    for text in (B.QUARREL, "Gamma reagì a un'offesa di Delta.", "Gamma, provocato da Delta, lo colpì."):
        assert "62 n. 2 c.p." in P.circumstance_triggers(_s(text)), text
    ctx = B._ctx(["Gamma colpisce Delta, persona offesa, con uno schiaffo.", B.FALL])
    assert "62 n. 2 c.p." not in M.held_attenuanti(ctx["circumstances"])
    assert B._by_id(ctx["rites"])["patteggiamento"]["status"] == M.PRECLUDED


def test_the_parte_civile_claim_to_quantify_is_no_reparation_step():
    ctx = B._ctx(["Gamma colpisce Delta con uno schiaffo.", B.FALL, "Gamma è incensurato.",
                  "La parte civile chiede il risarcimento del danno, da quantificare in separata sede."])
    assert "62 n. 6 c.p." not in ctx["circumstance_triggers"]
    assert "62 n. 6 c.p." not in M.held_attenuanti(ctx["circumstances"])
    got = P.circumstance_triggers(_s("Il consulente della difesa ha quantificato il danno per offrirlo agli eredi."))
    assert got["62 n. 6 c.p."]["step"] == "la quantificazione del danno"


@pytest.mark.parametrize("text", [
    "L'attenuante della provocazione (art. 62, n. 2, c.p.) non spetta a Gamma.",
    "Le attenuanti generiche (art. 62-bis c.p.) vanno negate.",
    "L'attenuante della provocazione (art. 62, n. 2, c.p.) non può essere invocata.",
])
def test_more_ways_of_excluding_are_read(text):
    assert P.says_excluded(text)
    log = []
    out = M.circumstance_exclusions(text, {}, log)
    assert "resta aperta sui fatti del caso" in out and len(log) == 1


def test_the_exclusion_check_reads_each_clause():
    other = "La legittima difesa è esclusa, mentre l'attenuante della provocazione (art. 62, n. 2, c.p.) resta da valutare."
    assert M.circumstance_exclusions(other, {}, []) == other
    log = []
    both = ("L'attenuante della provocazione (art. 62, n. 2, c.p.) non è esclusa, mentre le attenuanti generiche "
            "(art. 62-bis c.p.) sono escluse.")
    out = M.circumstance_exclusions(both, {}, log)
    assert out.startswith("L'attenuante della provocazione (art. 62, n. 2, c.p.) non è esclusa.")
    assert "Attenuanti generiche (art. 62-bis c.p.): resta aperta" in out and [x["circumstance"] for x in log] == \
        ["62-bis c.p."]
    out = M.circumstance_exclusions("L'attenuante della provocazione, in assenza di un fatto ingiusto, è esclusa.", {}, [])
    assert out.startswith("Attenuante della provocazione (art. 62, n. 2, c.p.): resta aperta") and "In assenza" not in out


def test_the_defence_excluding_an_aggravante_the_record_does_not_hold_stands():
    text = "L'aggravante dei futili motivi (art. 61, n. 1, c.p.) è esclusa: il diverbio aveva una causa seria."
    log = []
    assert M.circumstance_exclusions(text, {}, log) == text and log == []


# --- 3. the soccorso is the offender's ------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Delta fu portato al pronto soccorso, dove morì il giorno dopo.",
    "Gamma si allontanò; furono i passanti a chiamare il 118.",
    "Gamma non chiamò il 118 e se ne andò; un vicino chiamò l'ambulanza.",
    "Alfa si allontana senza soccorrere Beta; un passante chiama il 118.",
    "Alfa rifiutò di chiamare i soccorsi.",
])
def test_help_not_the_offenders_is_no_soccorso(text):
    assert P.soccorso_narrated(_s(text), RULE_593) == []


def test_the_offenders_help_is_still_read():
    for text in (A.SLAP[3], B.HELP, "Gamma, che non conosceva Delta, chiamò il 118."):
        assert P.soccorso_narrated(_s(text), RULE_593), text


def test_593_stays_and_nothing_favourable_when_others_called_the_118():
    ctx = A._ctx(A._records(with_593=True), A.SLAP[:3] + [
        "Alfa si allontana di corsa lasciando Beta a terra; sono i passanti a chiamare il 118."])
    assert "593 c.p." in ctx["records"] and ctx["post_fact"] == []
    assert P.post_fact_lines(ctx) == []
    assert not any("passanti" in x for x in ctx["riti"])


def test_the_flow_records_lose_the_absorbed_titles_and_the_counter_reads_them():
    state = {"records": A._records(with_582=True), "sentences": _s(B.QUARREL, B.FALL)}
    ctx = M.prepare(state, A.IMPUTATO, [], " ".join([B.QUARREL, B.FALL]), "Gamma", "", list(state["records"]), [], [],
                    [], [], link_call=lambda *a: {}, case_id_="review3")
    assert "582 c.p." not in state["records"] and "582 c.p." not in ctx["records"]
    alt = state["records"]["589 c.p."]
    assert alt["alternative"] and not alt.get("side") and not alt.get("charged")
    assert P.absorbed_as_autonomous(ctx) == 0
    state["records"]["582 c.p."] = {"side": "threat", "charged": 1, "requisites": []}      # a path re-adding it
    assert P.absorbed_as_autonomous(ctx) == 1


# --- 4. the riqualificazione's facts ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Alfa nega di aver dato alcuna spinta: ammette solo lo schiaffo.",
    "Non vi fu alcuna spinta: Alfa colpì Beta con un pugno al volto.",
    "Spinto dalla rabbia, Alfa colpì Beta.",
])
def test_a_denied_push_or_a_motive_bears_on_nothing(text):
    rule = P.requalify_rule("584 c.p.", "589 c.p.")
    assert P.requalify_facts(rule, _s(text)) == []
    ctx = A._ctx(A._records(), A.SLAP + [text])
    assert not any(c["kind"] == "requalify" for c in M.thesis_candidates(ctx))


def test_the_impulse_still_bears_on_the_element():
    rule = P.requalify_rule("584 c.p.", "589 c.p.")
    assert [f["text"] for f in P.requalify_facts(rule, _s(A.IMPULSE))] == [A.IMPULSE]


# --- 5. the art. 41 cap ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("denial", [
    "Beta non soffriva di alcuna patologia pregressa.",
    "Il medico legale esclude errori dei sanitari.",
    "Nel decorso non vi fu nulla di anomalo.",
])
def test_a_trigger_the_facts_deny_lifts_no_cap(denial):
    got = P.causation_cap(" ".join(A.SLAP + [denial]))
    assert got["capped"] and got["triggers"] == {}


def test_a_trigger_narrated_still_lifts_it():
    got = P.causation_cap(" ".join(A.SLAP + ["Beta soffriva di una cardiopatia."]))
    assert not got["capped"] and "patologia" in got["triggers"]


def test_a_capped_causation_line_never_leads_over_the_containment_line():
    ctx = A._ctx(A._records(), A.SLAP)
    plan = M.method_plan(ctx, call=A._valued(ctx, "medio"))
    sols, main, _ = M.selection(ctx, plan)
    assert not sols[main].get("causation_capped") and sols[main]["kind"] == "contenimento"
    capped = [t for t in sols if t.get("causation_capped")]
    assert capped and all(t.get("realismo") == M.BASSO and t.get("main_demoted") for t in capped)


# --- 6. the riti in the written prose ----------------------------------------------------------------------------------------

def test_a_written_figure_the_computation_does_not_print_is_dropped_and_counted():
    statuses = M.rite_statuses(B.PENA_584, "584", ["62-bis c.p."])
    text = ("Il patteggiamento è possibile se il PM presta il consenso, con una pena finale di circa due anni.\n"
            "Con il giudizio abbreviato la pena finale scende sotto la soglia.\n"
            "Il giudizio abbreviato comporta comunque uno sconto.\n"
            "Il patteggiamento porta la pena a 4 anni, 5 mesi e 10 giorni, entro cinque anni.")
    log = []
    out = M.rites_out(text, statuses, log)
    assert "circa due anni" not in out and "scende sotto la soglia" not in out
    assert "comporta comunque uno sconto" in out and "4 anni, 5 mesi e 10 giorni" in out
    assert [x.split(":")[0] for x in log] == ["mismatch", "uncomputed"]
    left = []
    M.rite_prose_out(text, statuses, left, rewrite=False)
    assert len(left) == 2


def test_the_rites_checked_counters_reach_case_counters():
    statuses = M.rite_statuses(B.PENA_584, "584", ["62-bis c.p."])
    state = {}
    out = flow.rites_checked(state, "Il patteggiamento comporta una pena finale di circa due anni.", statuses)
    assert "due anni" not in out
    c = state["block_counters"]
    assert c["rites_prose_removed"] == 1 and c["rites_prose_mismatch"] == 0 and c["rites_prose_uncomputed"] == 0
    state["block_counters"]["rites_prose_mismatch"] = 1           # a path that left one
    assert flow.case_counters(state)["rites_computation_mismatch"] >= 1


def test_the_codes_own_precluded_computation_is_no_offer():
    statuses = M.rite_statuses(B.PENA_584, "584")
    text = "\n".join("- " + x["line"] for x in statuses)
    assert M.rites_offered_unavailable(text, statuses) == []
    assert M.rites_out(text, statuses) == text
    pat = B._by_id(statuses)["patteggiamento"]
    assert pat["status"] == M.PRECLUDED and "Calcolo:" in pat["line"] and "improbabile" in pat["line"]
    other = B._by_id(M.rite_statuses(B.PENA_584, "624"))["patteggiamento"]      # not a listed offence: no consent line
    assert "consenso" not in other["line"]


def test_a_contravvenzione_abbreviato_halves():
    ab = B._by_id(M.rite_statuses("arresto da tre mesi a un anno", ""))["abbreviato"]
    assert "meno la metà per il rito (art. 442, co. 2, c.p.p.): 1 mese e 15 giorni" in ab["line"]
    delitto = B._by_id(M.rite_statuses("reclusione da tre mesi a un anno", ""))["abbreviato"]
    assert "meno un terzo per il rito (art. 442, co. 2, c.p.p.): 2 mesi" in delitto["line"]


# --- 7. the residue -----------------------------------------------------------------------------------------------------------

KNOWN = {"R1": {"text": "atti diretti a percuotere"}, "R2": {"text": "cagiona la morte"},
         "R3": {"text": "evento non voluto"}}


@pytest.mark.parametrize("text, gone", [
    ("La difesa contesta gli elementi R1-R3 sul piano soggettivo.", "R1"),
    ("La difesa contesta gli elementi R1/R2 sul piano soggettivo.", "R2"),
    ("La difesa contesta (R1-R3) sul piano soggettivo.", "R3"),
    ("La pronuncia va letta (contro, Cass. pen., Sez. 5, n. 11111/2025) sul punto.", "contro"),
    ("La pronuncia va letta (Cass. pen., Sez. 5, n. 11111/2025; a favore) sul punto.", "a favore"),
])
def test_ranges_pairs_and_stances_with_content_go_and_are_counted(text, gone):
    log = {}
    out = T.residue_out(text, KNOWN, log)
    assert gone not in out and sum(len(v) for v in log.values()) >= 1
    assert sum(T.residue_left(out, KNOWN).values()) == 0
    assert sum(T.residue_left(text, KNOWN).values()) >= 1           # the counter sees the original


def test_with_no_known_code_an_element_code_still_goes():
    text = "La difesa contesta R2 (contro) sul piano soggettivo. Il resto resta."
    assert T.residue_left(text, {})["codes"] >= 1
    out = T.residue_out(text, {}, {})
    assert out == "Il resto resta." and sum(T.residue_left(out, {}).values()) == 0
    plain = "La patente R2 non esiste."                  # an R-code not used as an element: read as it is
    assert T.residue_out(plain, {}, {}) == plain


@pytest.mark.parametrize("ref", [
    "(Sez. 4, n. 11111/2025 del 11.04.2025)", "(Sez. 4, n. 11111/2025, Rv. 288123 - 01, 11.04.2025)",
    "(Sez. 4, n. 11111/2025, udienza 11.04.2025)", "(Sez. 4, sent. n. 11111/2025, 11.04.2025)",
])
def test_headless_tails_go_whole(ref):
    out = T.residue_out(f"La prevedibilità va provata {ref}.", {}, {})
    assert out == "La prevedibilità va provata." and rulings.printed_problems(out) == []
    headed = "La prevedibilità va provata (Cass. pen., Sez. 4, n. 11111/2025, 11.04.2025)."
    assert T.residue_out(headed, {}, {}) == headed


@pytest.mark.parametrize("text, want", [
    ("La Procura ha iscritto Gamma nel registro degli indagati per omicidio preterintenzionale, ipotizzando l'art. "
     "584 c.p.", IR.ACCERTATO),
    ("Il capo d'imputazione ipotizza l'omicidio preterintenzionale.", IR.ACCERTATO),
    ("Il PM ipotizza a carico di Gamma il reato di omicidio preterintenzionale.", IR.ACCERTATO),
    ("Il PM potrebbe contestare a Gamma anche l'aggravante.", IR.IPOTIZZATO),
    ("Il PM contesta a Gamma l'omicidio preterintenzionale.", IR.ACCERTATO),
])
def test_procedural_acts_and_a_hypothetical_one(text, want):
    assert IR.certainty(text) == want


@pytest.mark.parametrize("sentence, kept", [
    ("Cenni: l'art. 41 c.p. stabilisce che il concorso di cause preesistenti esclude il rapporto di causalità.", False),
    ("Cenni: l'art. 41 c.p. stabilisce che le cause sopravvenute escludono sempre il rapporto di causalità.", False),
    ("Cenni: l'art. 41 c.p. stabilisce che il concorso di cause preesistenti non esclude il rapporto di causalità.",
     True),
    ("Cenni: l'art. 41 c.p. stabilisce che le cause sopravvenute escludono il rapporto di causalità quando sono da sole "
     "sufficienti a determinare l'evento.", True),
])
def test_a_cenno_inverting_or_absolutising_the_article_is_not_grounded(sentence, kept):
    log = []
    out = T.cenni_check(sentence, T41, {}, log)
    assert (out == sentence) is kept and (not log) is kept
