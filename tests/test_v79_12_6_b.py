"""v79.12.7 (items 2 and 6 of the brief), on synthetic sentences, records and store entries only:

2. the attenuanti from the facts, not a fixed block (data/penal_lines.json «circumstance_triggers»): 62 n. 2 when a
   diverbio / provocazione / offesa is narrated, its «da verificare: natura del diverbio» point tied to it; 62 n. 6 tied
   to a concrete step (trattativa con gli eredi, quantificazione, risarcimento offerto), the step named; 62-bis with
   record support, «da verificare» without; one art. 69 c.p. line whenever an aggravante is in the record. Never
   «esclusa» for a circumstance the facts do not exclude -- a written sentence saying so is rewritten. Counters
   attenuanti_from_facts, attenuanti_to_verify, attenuanti_excluded_by_facts, bilanciamento_lines,
   attenuanti_fixed_block (0), circumstance_exclusions_rewritten, circumstance_excluded_unsupported (0).
6. the riti computed, not asserted: the patteggiamento's base (the edittale minimum), each diminution in order (each
   attenuante held, then the third), the result and the five-year test, the PM's consent (unlikely for the offences the
   data lists); the abbreviato's third and the 442 co. 2-bis further sixth in sequence. The statuses agree with the
   printed numbers. Counters rites_computed, rites_without_computation (0), rites_computation_mismatch (0).
"""

import os

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

from src.rag import lawyer_method as M  # noqa: E402
from src.rag import penal_lines as P  # noqa: E402
from src.rag import posture  # noqa: E402


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("THESIS_DETERMINISTIC", "")
    from src.rag import plan_then_write
    monkeypatch.setattr(plan_then_write, "negation_review", lambda items, known, call=None: [])


IMPUTATO = {"rito": "penale", "family": "penale", "role": "imputato", "stage": "primo_grado",
            "entry": posture.entry_for("penale", "imputato", "primo_grado")}
OFFESA = {"rito": "penale", "family": "penale", "role": "persona_offesa", "stage": "primo_grado",
          "entry": posture.entry_for("penale", "persona_offesa", "primo_grado")}

# a synthetic case: a quarrel, a slap, a fall, a death; the help called; a step towards the heirs
QUARREL = "Gamma, durante un diverbio davanti a un bar, colpisce Delta con uno schiaffo."
FALL = "Delta cade all'indietro e batte la testa sul selciato; muore tre giorni dopo."
HELP = "Gamma chiama subito il 118 e resta accanto a Delta fino all'arrivo dell'ambulanza."
HEIRS = "Il difensore di Gamma ha avviato una trattativa con gli eredi di Delta per il risarcimento del danno."
CLAIM_ONLY = "La parte civile chiede il risarcimento del danno."
PENA_584 = "reclusione da dieci a diciotto anni (co. 1)"


def _sents(texts):
    return [{"id": f"S{i}", "text": t, "source": "caso", "certainty": "accertato"} for i, t in enumerate(texts, 1)]


def _records():
    return {"584 c.p.": {"rubrica": "Omicidio preterintenzionale", "side": "threat", "charged": 1, "requisites": [
        {"id": "R1", "comma": 1, "kind": "requisito", "status": "provato", "supporting": ["S1"],
         "text": "con atti diretti a commettere uno dei delitti preveduti dagli articoli 581 e 582"},
        {"id": "R2", "comma": 1, "kind": "requisito", "status": "da provare", "supporting": ["S2"],
         "text": "cagiona la morte di un uomo"}]}}


def _row(number="584", pena=PENA_584):
    return {"_article": (number, "c.p."), "s": {"plain_text": f"Chiunque commette il fatto ({number})."},
            "_proc": {"version": 4, "pena": pena, "querela": False}}


def _ctx(texts, role=IMPUTATO, records=None, rows=None):
    recs = records if records is not None else _records()
    return M.prepare({"records": recs, "sentences": _sents(texts)}, role, list(rows if rows is not None else [_row()]),
                     " ".join(texts), "Gamma", "", list(recs), [], [], [], [], link_call=lambda *a: {},
                     case_id_="v79126b")


def _att(ctx):
    """The attenuanti lines of the printed riti (§6), by the article they name."""
    return {x.split(":")[0]: x for x in ctx["riti"] if M.rite_of(x) == "" and "art. 6" in x.split(":")[0]}


def _by_id(statuses):
    return {x["id"]: x for x in statuses}


# --- 2. the attenuanti from the facts -------------------------------------------------------------------------------------

def test_the_triggers_are_data():
    labels = [r["label"] for r in P.trigger_rules()]
    assert labels == ["62 n. 2 c.p.", "62 n. 6 c.p.", "62-bis c.p."]
    steps = [s["name"] for s in next(r for r in P.trigger_rules() if r["label"] == "62 n. 6 c.p.")["steps"]]
    assert steps == ["la trattativa con gli eredi", "la quantificazione del danno", "il risarcimento offerto"]
    assert (P.data().get("bilanciamento") or {}).get("article") == "art. 69 c.p."


def test_a_diverbio_triggers_the_provocazione_with_its_point_and_counts_in_the_best_case():
    ctx = _ctx([QUARREL, FALL])
    line = _att(ctx)["Attenuante della provocazione (art. 62, n. 2, c.p.)"]
    assert "«Gamma, durante un diverbio davanti a un bar, colpisce Delta con uno schiaffo»" in line
    assert "da verificare: natura del diverbio" in line
    assert ctx["circumstances"]["62 n. 2 c.p."]["triggered_by"] == ["S1"]
    assert "62 n. 2 c.p." in M.held_attenuanti(ctx["circumstances"])
    pat = _by_id(ctx["rites"])["patteggiamento"]
    assert "per l'attenuante dell'art. 62 n. 2 c.p.: 6 anni e 8 mesi" in pat["line"]


def test_a_negated_diverbio_triggers_nothing_and_nothing_is_esclusa():
    ctx = _ctx(["Gamma, senza alcun diverbio, colpisce Delta con uno schiaffo.", FALL])
    assert "62 n. 2 c.p." not in ctx["circumstance_triggers"]
    assert not any("provocazione" in x for x in ctx["riti"])
    assert not any("esclusa" in x for x in ctx["riti"] if M.rite_of(x) == "")


def test_62_6_needs_a_concrete_step_and_names_it():
    ctx = _ctx([QUARREL, FALL, HEIRS])
    line = _att(ctx)["Attenuante della riparazione del danno (art. 62, n. 6, c.p.)"]
    assert "il passo concreto narrato è la trattativa con gli eredi" in line and "prima del giudizio" in line
    assert ctx["circumstances"]["62 n. 6 c.p."]["trigger_step"] == "la trattativa con gli eredi"
    # a claim for damages alone is no step: no line, not counted
    bare = _ctx([QUARREL, FALL, CLAIM_ONLY])
    assert "62 n. 6 c.p." not in bare["circumstance_triggers"]
    assert not any("riparazione del danno" in x for x in bare["riti"])
    assert "62 n. 6 c.p." not in M.held_attenuanti(bare["circumstances"])
    for text, step in (("Il consulente ha completato la quantificazione del danno subito dagli eredi.",
                        "la quantificazione del danno"),
                       ("Gamma ha offerto ai familiari una somma a titolo di risarcimento.", "il risarcimento offerto")):
        got = P.circumstance_triggers(_sents([text]))
        assert got["62 n. 6 c.p."]["step"] == step


def test_62_bis_with_record_support_or_da_verificare_never_esclusa():
    helped = _ctx([QUARREL, FALL, HELP])
    line = _att(helped)["Attenuanti generiche (art. 62-bis c.p.)"]
    assert "la condotta successiva al fatto «Gamma chiama subito il 118" in line
    assert "62-bis c.p." in M.held_attenuanti(helped["circumstances"])
    alone = _ctx(["Gamma colpisce Delta con uno schiaffo.", FALL])
    line = _att(alone)["Attenuanti generiche (art. 62-bis c.p.)"]
    assert "da verificare" in line and "esclusa" not in line
    assert "62-bis c.p." not in M.held_attenuanti(alone["circumstances"])
    # without any support the 584 patteggiamento stays precluded, its computation shown
    pat = _by_id(alone["rites"])["patteggiamento"]
    assert pat["status"] == M.PRECLUDED and "senza attenuanti integrate o da provare" in pat["line"]


def test_a_circumstance_the_record_excludes_is_esclusa_with_its_sentence_and_not_held():
    texts = [QUARREL, FALL, "Delta non aveva rivolto a Gamma alcuna offesa né compiuto alcun fatto ingiusto."]
    ctx = _ctx(texts)
    rec = ctx["circumstances"]["62 n. 2 c.p."]
    rec["requisites"][1].update({"status": "non risulta", "contradicting": ["S3"]})
    lines = P.attenuanti_lines(P.circumstance_triggers(_sents(texts)), ctx["circumstances"], _sents(texts))
    line = next(x for x in lines if "provocazione" in x)
    assert "esclusa dai fatti" in line and "alcun fatto ingiusto" in line
    assert "62 n. 2 c.p." not in M.held_attenuanti(ctx["circumstances"])


def test_the_fixed_block_is_gone_and_the_sentencing_line_never_printed():
    ctx = _ctx([QUARREL, FALL, HELP])
    fixed = next(x["line"] for x in ctx["rites"] if not x["rite"])
    assert fixed not in ctx["riti"]
    counts = M.attenuanti_counts(ctx)
    assert counts["attenuanti_fixed_block"] == 0 and counts["attenuanti_from_facts"] == 2
    assert counts["attenuanti_to_verify"] == 0 and counts["bilanciamento_lines"] == 0


def test_the_victims_answer_reads_no_attenuanti_triggers():
    ctx = _ctx([QUARREL, FALL, HEIRS], role=OFFESA)
    assert ctx["circumstance_triggers"] == {} and not ctx["riti"]


def test_art_69_line_whenever_an_aggravante_is_in_the_record():
    table = {c["label"]: c for c in M.data()["circumstances"]}
    circ = M.circumstance_records([table["61 n. 1 c.p."], table["62-bis c.p."]])
    for r in circ["61 n. 1 c.p."]["requisites"]:
        r.update({"status": "da provare", "supporting": ["S1"]})
    log = {}
    lines = P.attenuanti_lines({}, circ, _sents([QUARREL]), log)
    b = next(x for x in lines if x.startswith("Bilanciamento delle circostanze (art. 69 c.p.)"))
    assert "l'aggravante dell'art. 61 n. 1 c.p." in b and "prevalenti" in b and log["aggravanti"] == ["61 n. 1 c.p."]
    # an aggravante no fact supports, or one the record excludes, is not in the record: no line
    for r in circ["61 n. 1 c.p."]["requisites"]:
        r.update({"supporting": []})
    assert not any("art. 69" in x for x in P.attenuanti_lines({}, circ, []))
    circ["61 n. 1 c.p."]["requisites"][0].update({"status": "non risulta", "contradicting": ["S1"], "supporting": ["S1"]})
    assert not any("art. 69" in x for x in P.attenuanti_lines({}, circ, _sents([QUARREL])))


def test_a_written_esclusa_on_a_circumstance_the_record_does_not_exclude_is_rewritten():
    table = {c["label"]: c for c in M.data()["circumstances"]}
    circ = M.circumstance_records([table["62 n. 2 c.p."]])
    log = []
    out = M.circumstance_exclusions("L'attenuante della provocazione è esclusa. Il resto vale.", circ, log)
    assert "esclusa" not in out and "Attenuante della provocazione (art. 62, n. 2, c.p.): resta aperta" in out
    assert "(da verificare: il reato è commesso in stato d'ira" in out and out.endswith("Il resto vale.")
    assert log and log[0]["circumstance"] == "62 n. 2 c.p."
    # «non può essere esclusa» says the contrary: unchanged
    same = "La provocazione non può essere esclusa."
    assert M.circumstance_exclusions(same, circ, []) == same
    # the record excludes it (a sentence of the case says the contrary): the sentence stands
    circ["62 n. 2 c.p."]["requisites"][1].update({"status": "non risulta", "contradicting": ["S3"]})
    kept = "L'attenuante della provocazione è esclusa dai fatti."
    assert M.circumstance_exclusions(kept, circ, []) == kept
    # (review) an aggravante the record does not hold: the defence's «esclusa» stands -- rewriting it would raise it
    aggr = "L'aggravante dei motivi abietti o futili va esclusa."
    assert M.circumstance_exclusions(aggr, {}, []) == aggr
    # one the record holds (an element with a supporting sentence) and does not exclude: its kind and name
    held = M.circumstance_records([table["61 n. 1 c.p."]])
    held["61 n. 1 c.p."]["requisites"][0]["supporting"] = ["S1"]
    out = M.circumstance_exclusions(aggr, held, [])
    assert out.startswith("Aggravante (motivi abietti o futili) (art. 61 n. 1 c.p.): resta aperta")


def test_compose_rewrites_and_counts_the_unfounded_esclusa(monkeypatch):
    ctx = _ctx([QUARREL, FALL, HELP])

    def call(messages):
        return {"valutazioni": {c["id"]: {"pro": "p", "contro": "c", "argomento": "a", "realismo": "medio"}
                                for c in ctx["candidates"]}}
    plan = M.method_plan(ctx, call=call)
    log = {}
    answer = M.compose([("Norme applicate ai fatti", "La provocazione è esclusa. La norma vale.")], ctx, plan, log=log)
    assert "La provocazione è esclusa" not in answer
    assert ctx["circumstance_exclusions_rewritten"] == 1 and ctx["circumstance_excluded_unsupported"] == 0
    # §6 prints the attenuanti from the facts, the provocazione's point in §10
    six = answer[answer.find("**6."):answer.find("**7.")]
    assert "Attenuante della provocazione (art. 62, n. 2, c.p.)" in six and "Attenuanti generiche (art. 62-bis c.p.)" in six
    assert "Commisurazione della pena (art. 133 c.p.) e attenuanti generiche" not in answer
    ten = answer[answer.find("**10."):]
    assert "natura del diverbio" in ten


# --- 6. the riti computed ------------------------------------------------------------------------------------------------

def test_584_without_attenuanti_shows_the_precluded_computation():
    pat = _by_id(M.rite_statuses(PENA_584, "584"))["patteggiamento"]
    assert pat["status"] == M.PRECLUDED and pat["why"] == "min"
    assert pat["line"] == ("Applicazione della pena su richiesta (art. 444 c.p.p.): preclusa, perché anche nel caso più "
                           "favorevole la pena supera 5 anni. Calcolo: minimo edittale 10 anni, senza attenuanti "
                           "integrate o da provare; meno un terzo per il rito (art. 444, co. 1, c.p.p.): 6 anni e 8 mesi, "
                           "oltre 5 anni. Richiede il consenso del pubblico ministero (art. 444, co. 1, c.p.p.), "
                           "improbabile per un delitto come quello contestato (art. 584 c.p.).")


def test_the_patteggiamento_steps_in_order_and_the_pm_consent_unlikely_for_584():
    pat = _by_id(M.rite_statuses(PENA_584, "584", ["62 n. 2 c.p.", "62-bis c.p."]))["patteggiamento"]
    assert pat["status"] == M.POSSIBLE
    line = pat["line"]
    steps = ["minimo edittale 10 anni", "per l'attenuante dell'art. 62 n. 2 c.p.: 6 anni e 8 mesi",
             "per l'attenuante dell'art. 62-bis c.p.: 4 anni, 5 mesi e 10 giorni",
             "meno un terzo per il rito (art. 444, co. 1, c.p.p.): 2 anni, 11 mesi e 17 giorni, entro 5 anni"]
    at = [line.index(s) for s in steps]
    assert at == sorted(at)
    assert line.endswith("Richiede il consenso del pubblico ministero (art. 444, co. 1, c.p.p.), improbabile per un "
                         "delitto come quello contestato (art. 584 c.p.).")
    # an offence the list does not hold: the consent is required, never called unlikely
    other = _by_id(M.rite_statuses("reclusione da due a sei anni (co. 1)", "999"))["patteggiamento"]
    assert other["line"].endswith("Richiede il consenso del pubblico ministero (art. 444, co. 1, c.p.p.).")
    assert "584" in M.data()["riti_alternativi"][1]["computation"]["consent_unlikely_offences"]


def test_the_abbreviato_third_then_the_further_sixth_in_sequence():
    abb = _by_id(M.rite_statuses(PENA_584, "584"))["abbreviato"]
    assert abb["status"] == M.AVAILABLE
    third = "meno un terzo per il rito (art. 442, co. 2, c.p.p.): 6 anni e 8 mesi"
    sixth = ("meno un ulteriore sesto se l'imputato non impugna la sentenza (art. 442, co. 2-bis, c.p.p.): 5 anni, 6 "
             "mesi e 20 giorni")
    assert third in abb["line"] and sixth in abb["line"] and abb["line"].index(third) < abb["line"].index(sixth)
    assert abb["line"].index("minimo edittale 10 anni") < abb["line"].index(third)


def test_the_floor_of_art_67_is_a_printed_step(monkeypatch):
    """A synthetic table: a special-effect attenuante (four fifths) first, then the 62-bis -- below a quarter of the
    minimum, the floor of art. 67, co. 2, c.p. is its own printed step."""
    rite = M.data()["riti_alternativi"][1]
    table = {"circumstances": [{"label": "Y c.p.", "kind": "attenuante", "max_reduction": "4/5", "effect": "speciale"},
                               {"label": "62-bis c.p.", "kind": "attenuante", "max_reduction": "1/3",
                                "effect": "comune"}]}
    monkeypatch.setattr(M, "data", lambda: table)
    got = M.rite_computation(rite, 10, ["62-bis c.p.", "Y c.p."])
    assert got["steps"][1].startswith("meno fino a 4/5 per l'attenuante dell'art. Y c.p.: 2 anni")
    assert got["steps"][3] == "non meno di un quarto del minimo (art. 67, co. 2, c.p.): 2 anni e 6 mesi"
    assert got["after_rite"] == pytest.approx(10 / 4 * 2 / 3)


@pytest.mark.parametrize("pena", ["reclusione da sette a quindici anni (co. 1)", "reclusione da otto a venti anni (co. 1)",
                                  "reclusione da sei a dodici anni (co. 1)", PENA_584])
@pytest.mark.parametrize("att", [[], ["62-bis c.p."], ["62-bis c.p.", "62 n. 6 c.p."]])
def test_the_statuses_agree_with_the_printed_numbers(pena, att):
    statuses = M.rite_statuses(pena, "999", att)
    pat = _by_id(statuses)["patteggiamento"]
    over = pat["result"] > 5 + 1e-9
    assert (pat["status"] == M.PRECLUDED) == over and ("oltre 5 anni" in pat["line"]) == over
    assert ("entro 5 anni" in pat["line"]) == (not over)
    assert M.rite_counts({"rites": statuses}) == {"rites_computed": 2, "rites_without_computation": 0,
                                                  "rites_computation_mismatch": 0}


def test_no_computation_where_the_pena_cannot_give_one():
    for pena in ("", "ergastolo (co. 1)", "multa fino a euro 309 (co. 1)"):
        statuses = M.rite_statuses(pena, "999")
        assert all(not x["computable"] and not x["computation"] for x in statuses)
        assert M.rite_counts({"rites": statuses})["rites_without_computation"] == 0


def test_the_counters_catch_a_rito_printed_without_its_computation_or_against_it():
    statuses = M.rite_statuses(PENA_584, "584")
    pat = _by_id(statuses)["patteggiamento"]
    bare = [{**x, "line": x["name"] + ": preclusa."} if x is pat else x for x in statuses]
    assert M.rite_counts({"rites": bare})["rites_without_computation"] == 1
    wrong = [{**x, "status": M.POSSIBLE, "why": "pena"} if x is pat else x for x in statuses]
    assert M.rite_counts({"rites": wrong})["rites_computation_mismatch"] == 1


def test_the_method_prints_the_computed_riti_and_counts_them():
    ctx = _ctx([QUARREL, FALL, HELP])
    pat = next(x for x in ctx["riti"] if M.rite_of(x) == "patteggiamento")
    assert "Calcolo nel caso più favorevole: minimo edittale 10 anni" in pat and "improbabile" in pat
    counts = M.rite_counts(ctx)
    assert counts == {"rites_computed": 2, "rites_without_computation": 0, "rites_computation_mismatch": 0}


def test_the_new_counters_reach_the_case_counters():
    from src.rag import interactive_flow as flow
    keys = ("attenuanti_from_facts", "attenuanti_to_verify", "attenuanti_excluded_by_facts", "bilanciamento_lines",
            "attenuanti_fixed_block", "rites_computed", "rites_without_computation", "rites_computation_mismatch")
    assert set(keys) <= set(M.PENAL_COUNT_KEYS)
    out = flow.case_counters({"thesis_counts": {k: 1 for k in keys},
                              "block_counters": {"circumstance_exclusions_rewritten": 2,
                                                 "circumstance_excluded_unsupported": 0}})
    assert all(out[k] == 1 for k in keys)
    assert out["circumstance_exclusions_rewritten"] == 2 and out["circumstance_excluded_unsupported"] == 0
