"""v74: general rules, no case-specific logic -- synthetic records and texts, one group per rule.

1. arguments bound to declarations: a model-argued thesis's sentence negating an element outside its «contesta» is
   dropped (the requirement check, element by element; its word test when the call fails); a thesis contesting
   nothing (containment, attenuanti, scriminanti, claims) is argued by the code
2. every record decomposed: the charge, the alternatives and the named articles split as the charge is; an article
   that cannot be decomposed gets no record -- no sussunzione row, no fatto mancante, no passo; a riqualificazione
   only for a decomposed, sound alternative
3. the non-fattispecie gates: procedural codes and c.p. Libro I never fattispecie records (a cause di giustificazione
   graded, never a fattispecie); d.lgs. 231/2001 only with its signal
4. facts named with the case's words: the act and the parties as the fatti certi name them
5. «Punti da verificare»: whole sentences, duplicates merged
"""

import os
import sys
from pathlib import Path

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.rag import fact_terms, issue_records, lawyer_method as M, plan_then_write as ptw  # noqa: E402
from src.rag import posture as posture_mod, record_check, rule_quote, special_laws  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402

ENTRY = posture_mod.entry_for("penale", "imputato", "primo_grado")
IMPUTATO = {"rito": "penale", "family": "penale", "role": "imputato", "stage": "primo_grado", "entry": ENTRY}
ATTORE = {"rito": "civile", "family": "civile", "role": "attore", "stage": "pre_filing"}
SENTS = [{"id": "S1", "text": "Sempronio colpisce Caio con uno schiaffo al volto.", "source": "caso",
          "certainty": "accertato"},
         {"id": "S2", "text": "Caio cade, batte la testa e muore.", "source": "caso", "certainty": "accertato"}]
CASE = "Sempronio colpisce Caio con uno schiaffo al volto. Caio cade, batte la testa e muore."
T584 = ("Chiunque, con atti diretti a commettere uno dei delitti preveduti dagli articoli 581 e 582, cagiona la morte "
        "di un uomo, è punito con la reclusione da dieci a diciotto anni.")
T589 = "Chiunque cagiona per colpa la morte di una persona è punito con la reclusione da sei mesi a cinque anni."
T832 = ("Il proprietario ha diritto di godere e disporre delle cose in modo pieno ed esclusivo, entro i limiti e con "
        "l'osservanza degli obblighi stabiliti dall'ordinamento giuridico.")


def _req(i, text, status, kind="requisito", **extra):
    return {"id": f"R{i}", "comma": 1, "kind": kind, "text": text, "status": status, **extra}


def _records(*statuses, alt=None, alt_kind="requisito"):
    texts = ["atti diretti a percuotere o ledere", "la morte come conseguenza degli atti", "il nesso causale"]
    recs = {"584 c.p.": {"rubrica": "Omicidio preterintenzionale", "side": "threat", "charged": 1,
                         "requisites": [_req(i, texts[i - 1], st, distinctive=(i == 1))
                                        for i, st in enumerate(statuses, 1)]}}
    if alt:
        recs["589 c.p."] = {"rubrica": "Omicidio colposo", "side": "weapon", "alternative": True,
                            "requisites": [_req(9, "per colpa", alt, kind=alt_kind),
                                           _req(10, "cagiona la morte di una persona", "provato", kind=alt_kind)]}
    return recs


def _ctx(recs, posture=IMPUTATO, case=CASE, circ_link=None):
    return M.prepare({"records": recs, "sentences": SENTS}, posture, [], case, "Sempronio", "", list(recs), [], [], [],
                     [], link_call=circ_link or (lambda *a: {}), case_id_="t74")


# --- 1. arguments bound to declarations ------------------------------------------------------------------------------------

def test_a_thesis_contesting_nothing_is_argued_by_the_code():
    ctx = _ctx(_records("provato", "provato", "provato"))
    cands = M.thesis_candidates(ctx)
    by_kind = {c["kind"]: c for c in cands}
    contain = by_kind["contenimento"]
    assert contain["argued"] == "code" and contain["code_argument"]["realismo"] == "alto"
    assert "Tutti gli elementi dell'art. 584 c.p. risultano integrati (R1, R2, R3)" in contain["code_argument"]["argomento"]
    assert ENTRY["goal"] in contain["code_argument"]["argomento"]
    assert all(c["argued"] == "model" for c in cands if c["kind"].startswith("contest_"))
    assert "[argomentata dal codice: indica solo il realismo]" in M.candidates_block(cands)
    log = {}
    raw = {"valutazioni": {contain["id"]: {"pro": "Del modello.", "argomento": "Il nesso non c'è.", "realismo": "basso"}}}
    sols = M.apply_valuations(raw, cands, log)
    mine = next(s for s in sols if s["kind"] == "contenimento")
    # v79.7: the text stays the code's, the realismo is the model's
    assert "Del modello" not in mine["pro"] and "nesso" not in mine["argomento"] and mine["realismo"] == "basso"
    assert log["code_argued_ignored"] == [contain["id"]]
    full = {"valutazioni": {c["id"]: {"realismo": "basso"} for c in cands}}        # v79.7: every candidate valued
    assert M.score_sample(full, cands, {})[0] == 10.0


def test_an_attenuante_and_a_claim_are_argued_from_their_records():
    circ = {"62 n. 2 c.p.": {"kind": "attenuante", "requisites": [_req(30, "lo stato d'ira", "provato"),
                                                                  _req(31, "il fatto ingiusto altrui", "da provare")]}}
    ctx = {"records": {}, "circumstances": circ, "posture": IMPUTATO, "assessment": {}}
    arg = M.code_argument({"kind": "attenuante", "article": "62 n. 2 c.p.", "poggia_su": ["R30"]}, ctx)
    assert arg["argomento"] == "L'attenuante dell'art. 62 n. 2 c.p. si regge sugli elementi provati R30: «lo stato d'ira»."
    assert arg["contro"] == "Restano da provare: «il fatto ingiusto altrui» (R31)." and arg["realismo"] == "medio"
    recs = {"2043 c.c.": {"rubrica": "Risarcimento per fatto illecito", "side": "weapon",
                          "requisites": [_req(1, "il fatto illecito", "provato"), _req(2, "il danno ingiusto", "provato")]}}
    claim = M.code_argument({"kind": "claim", "article": "2043 c.c."}, {"records": recs, "posture": ATTORE})
    assert claim["realismo"] == "alto" and claim["contro"] == "La controparte ne contesterà gli elementi."


def test_a_sentence_negating_an_element_outside_contesta_is_dropped_by_the_review():
    known = {"R1": {"article": "584 c.p.", "text": "atti diretti a ledere", "status": "provato"},
             "R3": {"article": "584 c.p.", "text": "il nesso causale", "status": "provato"}}
    sols = [{"candidate": "T1", "tesi": "Contestare R1", "contesta": ["R1"], "argued": "model",
             "pro": "Lo schiaffo non mirava a ledere.", "contro": "", "argomento":
             "Lo schiaffo non era diretto a ledere. Inoltre manca il nesso causale con la morte."},
            {"candidate": "T2", "tesi": "Contenimento", "contesta": [], "argued": "code",
             "argomento": "Manca il nesso causale."}]
    seen = {}

    def call(messages, **kw):
        seen["human"] = messages[1].content
        return [{"i": 3, "elemento": "R3", "perche": "nega il nesso"}, {"i": 1, "elemento": "R1", "perche": "contestato"}]
    log = M.bind_arguments(sols, known, call)
    assert sols[0]["argomento"] == "Lo schiaffo non era diretto a ledere." and sols[0]["pro"] == \
        "Lo schiaffo non mirava a ledere."                    # R1 is the one it contests: kept
    assert sols[1]["argomento"] == "Manca il nesso causale."  # the code's own text is not the model's
    assert log["how"] == "review" and [d["elemento"] for d in log["dropped"]] == ["R3"]
    assert log["dropped"][0]["campo"] == "argomento" and "[contesta: R1]" in seen["human"]
    assert "elemento_negato" in ptw.NEGATION_SYSTEM


def test_the_word_test_decides_when_the_review_call_fails():
    known = {"R2": {"text": "la morte come conseguenza degli atti", "status": "provato"},
             "R3": {"text": "il nesso causale", "status": "da provare"}}
    sols = [{"candidate": "T1", "tesi": "x", "contesta": ["R3"], "argued": "model", "pro": "",
             "contro": "La morte non è conseguenza degli atti.", "argomento": "Il nesso causale non è provato."}]

    def broken(messages, **kw):
        raise RuntimeError("down")
    log = M.bind_arguments(sols, known, broken)
    assert log["how"] == "word_test" and sols[0]["contro"] == "" and sols[0]["argomento"] == \
        "Il nesso causale non è provato."
    assert record_check.negated_elements("Il nesso causale non regge (R3).", known, ["R2"]) == ["R3"]
    assert record_check.negated_elements("Il nesso causale è provato.", known) == []


def test_the_conclusions_restate_the_bound_argument_only(monkeypatch, tmp_path):
    monkeypatch.setenv(M.THESIS_LOG_ENV, str(tmp_path))
    monkeypatch.setattr(ptw, "negation_review", lambda items, known, call=None: None)      # the word test decides
    ctx = _ctx(_records("provato", "da provare", "provato"))
    cands = M.thesis_candidates(ctx)
    open_ = next(c for c in cands if c["kind"] == "contest_open")
    reply = {"valutazioni": {c["id"]: {"realismo": "medio", "argomento": "La morte non è provata come conseguenza degli "
                                       "atti. Gli atti diretti a percuotere o ledere non ci sono."}
                             for c in cands if c["argued"] == "model"}}
    plan = M.method_plan(ctx, call=lambda messages: [reply])
    mine = next(s for s in plan["soluzioni"] if s["candidate"] == open_["id"])
    assert mine["argomento"] == "La morte non è provata come conseguenza degli atti."      # R1 is not its to negate
    answer = M.compose([("Norme applicate ai fatti", "regola")], ctx, plan)
    assert "Gli atti diretti a percuotere o ledere non ci sono" not in answer
    assert any(d["elemento"] == "R1" and d["tesi"] == open_["id"] for d in ctx["argument_binding"]["dropped"])


# --- 2. every record decomposed --------------------------------------------------------------------------------------------

def test_the_method_reasons_only_on_decomposed_fattispecie_records():
    recs = _records("provato", "da provare", "provato")
    recs["2043 c.c."] = {"rubrica": "Risarcimento", "requisites": [_req(40, "Qualunque fatto doloso…", "da provare",
                                                                         kind="comma")]}
    recs["41 c.p."] = {"rubrica": "Concorso di cause", "requisites": [_req(41, "le concause", "provato")]}
    recs["52 c.p."] = {"rubrica": "Difesa legittima", "side": "weapon",
                       "requisites": [_req(42, "la necessità di difendersi", "provato")]}
    recs["584 c.p."]["requisites"].append(_req(43, "Chiunque … è punito", "da provare", kind="comma"))
    log = {}
    view = M.method_records(recs, log)
    assert set(view) == {"584 c.p.", "52 c.p."} and log == {"gated": ["41 c.p."], "undecomposed": ["2043 c.c."]}
    assert [r["id"] for r in view["584 c.p."]["requisites"]] == ["R1", "R2", "R3"]      # the whole comma is no element
    ctx = _ctx(recs)
    assert "2043" not in ctx["subsumption"] and "41 c.p." not in ctx["subsumption"]
    assert all(m["article"] in ("584 c.p.", "52 c.p.") for m in ctx["facts"]["mancanti"])
    assert not any("R40" in s["scopo"] or "R43" in s["scopo"]
                   for s in M.steps_from_selection(ctx, [{"tesi": "x", "contesta": ["R40", "R43", "R2"],
                                                          "poggia_su": []}], 0))


def test_a_riqualificazione_exists_only_for_a_decomposed_sound_alternative():
    ctx = _ctx(_records("provato", "provato", "provato", alt="provato"))
    # v79.12.7 (item 4): 584 -> 589 is offered only with a fact on the differing element (a spinta, an impulse)
    ctx["sentences"] = list(ctx["sentences"]) + [{"id": "S9", "text": "Sempronio dice di avere reagito d'impulso, con una spinta.", "source": "caso", "certainty": "accertato"}]
    assert any(c["kind"] == "requalify" and c["article"] == "589 c.p." for c in M.thesis_candidates(ctx))
    recs = _records("provato", "provato", "provato", alt="provato", alt_kind="comma")
    assert not any(c["kind"] == "requalify" for c in M.thesis_candidates(_ctx(recs)))
    assert issue_records.decomposed(recs["584 c.p."]) and not issue_records.decomposed(recs["589 c.p."])


def test_a_one_clause_offence_is_split_as_the_charge_is():
    reqs = issue_records._requisites({"_article": ("589", "c.p."), "s": {"plain_text": T589}}, {})
    assert [(r["kind"], r["text"]) for r in reqs] == [("requisito", "cagiona la morte di una persona"),
                                                     ("requisito", "per colpa")]
    assert rule_quote.offence_elements(T584) == ["con atti diretti a commettere uno dei delitti preveduti dagli "
                                                 "articoli 581 e 582", "cagiona la morte di un uomo"]


def _row(number, code, text, rubrica):
    return {"_article": (number, code), "s": {"plain_text": text, "titolo_articolo": rubrica}, "d": {"id": "doc"}}


def test_the_charge_the_alternatives_and_the_named_articles_get_decomposed_records(monkeypatch):
    rows = [_row("584", "c.p.", T584, "Omicidio preterintenzionale"), _row("589", "c.p.", T589, "Omicidio colposo"),
            _row("832", "c.c.", T832, "Contenuto del diritto"), _row("41", "c.p.", "Il concorso di cause…",
                                                                     "Concorso di cause")]
    state = {"analysis": {}, "text": CASE + " Il PM contesta a Sempronio l'art. 584 c.p.; si richiamano gli artt. 41 "
             "c.p. e 832 c.c.", "request": "", "intake": {}, "sentences": SENTS, "questions": [],
             "read": {"alternatives": [{"name": "omicidio colposo", "rubricated": "Omicidio colposo"}]},
             "records": {"589 c.p.": {"rubrica": "Omicidio colposo", "requisites": [], "dropped": "altri profili"}}}
    link, fetched = flow.decompose_records(state, {"role": "imputato"}, rows, set(), fetch=lambda refs: [])
    recs = state["records"]
    assert link == ["584 c.p.", "589 c.p."] and fetched == []
    assert [r["text"] for r in recs["589 c.p."]["requisites"]] == ["cagiona la morte di una persona", "per colpa"]
    assert recs["584 c.p."]["decomposed_for"] == "addebito" and recs["589 c.p."]["decomposed_for"] == \
        "qualificazione alternativa"
    assert "41 c.p." not in recs and "832 c.c." not in recs          # gated; not decomposable: no record
    ids = [r["id"] for rec in recs.values() for r in rec["requisites"]]
    assert len(ids) == len(set(ids)) == 4

    linked = {}
    monkeypatch.setattr(issue_records, "link", lambda records, topics, labels, sents: (
        linked.update(topics=[t.get("title") for t in topics], labels=labels), {"topics": [], "total_ms": 0})[1])
    monkeypatch.setattr(flow, "mark_sides", lambda sents, st: None)
    flow.link_records(state, [], set(), link)
    assert linked["topics"] == [flow.DECOMPOSED_TOPIC] and linked["labels"] == [["584 c.p.", "589 c.p."]]


# --- 3. the non-fattispecie gates ------------------------------------------------------------------------------------------

def test_procedural_codes_and_libro_i_are_never_fattispecie_records(monkeypatch):
    monkeypatch.setattr(issue_records, "_requisites", lambda row, analysis, *_: [
        {"id": "", "comma": 1, "text": "un requisito", "kind": "requisito", "status": "da provare"}])
    topic = {"title": "T", "articles": ["575 c.p.", "41 c.p.", "52 c.p.", "405 c.p.p.", "700 c.p.c."]}
    rows = [_row(n, c, "testo", "") for n, c in (("575", "c.p."), ("41", "c.p."), ("52", "c.p."), ("405", "c.p.p."),
                                                  ("700", "c.p.c."))]
    recs = issue_records.build([topic], rows, {})
    assert recs["575 c.p."]["requisites"] and recs["52 c.p."]["requisites"]          # a fattispecie; a scriminante
    for label in ("41 c.p.", "405 c.p.p.", "700 c.p.c."):
        assert recs[label]["requisites"] == [] and recs[label]["not_fattispecie"] == issue_records.NOT_FATTISPECIE
    assert issue_records.justification("52 c.p.") and not issue_records.counted("52 c.p.")


def test_a_non_fattispecie_charge_is_never_the_threat_nor_a_candidate():
    recs = {"405 c.p.p.": {"rubrica": "Esercizio dell'azione penale", "side": "threat", "charged": 1,
                           "requisites": [_req(1, "la notizia di reato", "provato")]}}
    assert M.main_threat(recs) == ""
    ctx = {"records": recs, "circumstances": {}, "assessment": {"article": "405 c.p.p."}, "posture": IMPUTATO}
    assert [c for c in M.thesis_candidates(ctx) if c["kind"].startswith("contest_")] == []


def test_dlgs_231_reaches_grading_only_with_its_signal():
    law = special_laws.law_of("d.lgs. 231/2001")
    assert law and law["id"] == "dlgs_231_2001"
    row = {"_article": ("25-septies", "d.lgs. 231/2001")}
    kept, dropped = special_laws.gate([row], "Tizio ha colpito il vicino con uno schiaffo.", set())
    assert kept == [] and dropped and "231/2001" in dropped[0]
    for text in ("La società Alfa s.r.l. non aveva un modello organizzativo.", "Il datore di lavoro non ha formato "
                 "l'operaio.", "L'ente non ha vigilato."):
        assert special_laws.gate([row], text, set())[0] == [row], text


# --- 4. facts named with the case's words ----------------------------------------------------------------------------------

CERTI = ["Sempronio colpisce Caio con uno schiaffo al volto."]


@pytest.mark.parametrize("written, expected", [
    ("Sempronio ha sferrato un pugno a Caio.", "Sempronio ha sferrato uno schiaffo a Caio."),
    ("La forza del pugno non era tale da uccidere.", "La forza dello schiaffo non era tale da uccidere."),
    ("Il ceffone fu violento.", "Lo schiaffo fu violento."),
    ("Colpì a pugni.", ""),                                         # a plural: cannot be inflected -- dropped
    ("Il pugno e il calcio.", ""),                                  # two acts the case does not state -- dropped
    ("Lo schiaffo, non un pugno, causò la caduta.", "Lo schiaffo, non un pugno, causò la caduta."),   # a contrast
    ("L'aggressore non voleva la morte.", "Sempronio non voleva la morte."),
    ("La condotta dell'aggressore fu breve.", "La condotta di Sempronio fu breve."),
    ("Gli aggressori erano due.", ""),
])
def test_the_act_and_the_parties_are_named_with_the_fatti_certi(written, expected):
    log = []
    assert fact_terms.apply(written, CERTI, CERTI[0] + " Caio muore.", "Sempronio", True, log) == expected
    assert bool(log) == (written != expected or "non un pugno" in written)


def test_an_object_the_case_does_not_state_is_dropped_not_replaced():
    log = []
    out = fact_terms.apply("L'automobile urtò Caio. La bici era ferma.", ["Tizio guidava la sua auto."],
                           "Tizio guidava la sua auto.", "", False, log)
    assert out == "L'auto urtò Caio." and [x["how"] for x in log] == ["replaced", "dropped"]
    assert fact_terms.apply("L'aggressore fuggì.", CERTI, CERTI[0], "Caio", False) == "L'aggressore fuggì."


def test_the_check_runs_on_the_writer_and_on_the_methods_prose():
    state = {"sentences": SENTS + [{"id": "S3", "text": "Assisto Sempronio.", "source": "sistema",
                                    "certainty": "accertato"}],
             "client": {"name": "Sempronio"}, "posture": {"role": "imputato"}}
    certi, client, accused = flow.fact_wording_inputs(state)
    assert certi == [s["text"] for s in SENTS] and client == "Sempronio" and accused
    ctx = _ctx(_records("provato", "da provare", "provato"))
    # v79.12.7 (item 5, review): a pathology narrated lifts the art. 41 cap -- the causal contest still leads here
    ctx["sentences"] = list(ctx["sentences"]) + [{"id": "S9", "text": "Caio soffriva di una cardiopatia pregressa.",
                                                  "source": "caso", "certainty": "accertato"}]
    cands = M.thesis_candidates(ctx)
    plan = M.clean_method({"valutazioni": {c["id"]: {"realismo": "medio", "argomento": "Un pugno al volto non basta."}
                                           for c in cands}}, cands)
    log = {}
    answer = M.compose([("Norme applicate ai fatti", "regola")], ctx, plan, log=log)
    assert "pugno" not in answer and "Uno schiaffo al volto non basta" in answer
    assert log["fact_wording"] and log["fact_wording"][0]["from"] == "pugno"


# --- 5. «Punti da verificare»: whole sentences ------------------------------------------------------------------------------

def test_each_point_is_the_whole_sentence_that_carried_the_marker():
    sections = [("A", "La data della notizia è da verificare, e così il luogo."),
                ("B", "Il fatto è noto. Chi ha chiamato i soccorsi, e quando? (da verificare)"),
                ("C", "La querela è tempestiva (da verificare: la data della notizia del fatto)."),
                ("D", "- Querela tempestiva: la data della notizia del fatto (da verificare).")]
    body, points = M.collect_points(sections)
    assert points == ["La data della notizia è da verificare, e così il luogo.",
                      "Chi ha chiamato i soccorsi, e quando?",
                      "La querela è tempestiva (la data della notizia del fatto)."]       # C and D merged
    assert body[0][1] == "La data della notizia (v. punto 1), e così il luogo."
    assert body[3][1] == "- Querela tempestiva: la data della notizia del fatto (v. punto 3)."
    assert "da verificare" not in " ".join(x for _, x in body).lower()


def test_no_point_is_a_clause_fragment():
    text = ("Il termine di querela decorre dalla conoscenza del fatto, che il caso non data (da verificare), e la "
            "remissione resta possibile.")
    _, points = M.collect_points([("A", text)])
    assert points == ["Il termine di querela decorre dalla conoscenza del fatto, che il caso non data, e la remissione "
                      "resta possibile."]
    assert M.same_point("La querela è tempestiva.", "la querela è tempestiva") and \
        not M.same_point("La querela è tempestiva.", "La remissione è possibile.")
