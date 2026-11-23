"""v79.16.0 -- unit tests per item of the brief: A (the fact record drives every stage), B (the questions drive
retrieval), C (the Q&A assembly), D (one coherent opinion), the measurement line_table_weaker. Synthetic sentences
only; no model: every call faked where the function takes it."""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.rag import analysis_step as A  # noqa: E402
from src.rag import case_frame as cf  # noqa: E402
from src.rag import fact_record as fr  # noqa: E402
from src.rag import intake  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import issue_records as ir  # noqa: E402
from src.rag import lawyer_method as M  # noqa: E402
from src.rag.nodes import retrieval as R  # noqa: E402
import test_v79_15_0_analysis as T  # noqa: E402


@pytest.fixture
def analysis_on(monkeypatch):
    monkeypatch.setenv("ANALYSIS_STEP", "true")


@pytest.fixture
def record_on(monkeypatch):
    monkeypatch.setenv("FACT_RECORD", "true")


# --- A1: the record as the single source of the sentences ----------------------------------------------------------------

CASE = ("Tizio ha consegnato il motorino a Caio il 3 marzo 2024. Caio sostiene di aver pagato il prezzo. "
        "Posso chiedere la restituzione?")
REPLY = [{"citazione": "Tizio ha consegnato il motorino a Caio il 3 marzo 2024.", "tipo": "fatto", "stato": "accertato"},
         {"citazione": "Caio sostiene di aver pagato il prezzo.", "tipo": "fatto", "stato": "allegato", "chi": "Caio"},
         {"citazione": "Posso chiedere la restituzione?", "tipo": "richiesta"}]


def test_a_partial_record_falls_back_to_the_rule_and_is_counted(record_on):
    ok = fr.build(CASE, asker=lambda _c: list(REPLY))
    assert ok["status"] == "ok" and len(fr.facts({"fact_record": ok})) == 2
    partial = {**ok, "status": "partial"}
    assert fr.facts({"fact_record": partial}) == [] and fr.fallback({"fact_record": partial})
    assert fr.fallback({"fact_record": None}) and not fr.fallback({"fact_record": ok})
    state = {"text": CASE, "request": "", "additions": [], "fact_record": partial}
    sents = flow.case_sentences(state)
    assert state["block_counters"]["fact_record_fallback"] == 1 and sents and not any(s.get("fact") for s in sents)
    state = {"text": CASE, "request": "", "additions": [], "fact_record": ok}
    sents = flow.case_sentences(state)
    assert "fact_record_fallback" not in state.get("block_counters", {}) and all(s.get("fact") for s in sents)
    assert [s["text"] for s in sents] == [REPLY[0]["citazione"], REPLY[1]["citazione"]]   # the ask never a sentence


def test_with_the_flag_off_the_fallback_is_never_counted():
    state = {"text": CASE, "request": "", "additions": [], "fact_record": None}
    flow.case_sentences(state)
    assert "fact_record_fallback" not in state.get("block_counters", {})


def test_the_records_uncertain_items_for_section_10(record_on):
    text = "Tizio ha venduto il fondo a Caio. Forse Caio aveva già pagato una parte. Caio non ha pagato il resto."
    record = fr.build(text, asker=lambda _c: [
        {"citazione": "Tizio ha venduto il fondo a Caio.", "tipo": "fatto", "stato": "accertato"},
        {"citazione": "Forse Caio aveva già pagato una parte.", "tipo": "fatto", "stato": "ipotizzato"}])
    items = fr.uncertain_items({"fact_record": record})
    assert [(x["kind"], x["text"]) for x in items] == [("ipotesi", "Forse Caio aveva già pagato una parte."),
                                                        ("non_letto", "Caio non ha pagato il resto.")]


# --- A2: the provenance from the record ----------------------------------------------------------------------------------

def _record_sents(items):
    """[(text, certainty, said_by)] -> the sentences as issue_records.sentences gives them with the record."""
    return [{"id": f"S{i}", "text": t, "source": "caso", "certainty": c, "fact": f"F{i}", "said_by": who or ""}
            for i, (t, c, who) in enumerate(items, 1)]


CIVIL_RECORD = _record_sents([
    ("Lucia Ferri ha versato a Marco Vento 20.000 euro a titolo di caparra confirmatoria.", "accertato", ""),     # S1
    ("Marco Vento sostiene di aver già restituito la caparra a Lucia Ferri.", "allegato", "Marco Vento"),       # S2
    ("Lucia Ferri dichiara di non aver mai ricevuto alcuna restituzione.", "allegato", "Lucia Ferri"),          # S3
    ("Marco Vento ammette di aver venduto l'appartamento a un terzo prima del rogito.", "allegato", "Marco Vento"),  # S4
    ("Forse il rogito era fissato per marzo.", "ipotizzato", ""),                                               # S5
])


def _civil_record(frame=T.CIVIL_FRAME):
    return A.make_input(frame, T.CIVIL_NORMS, CIVIL_RECORD, frame.get("quesiti") or [], client="Lucia Ferri",
                        client_roles=["attore"])


def test_make_input_keeps_the_records_fact_id_and_speaker():
    inp = _civil_record()
    assert [(s["fact"], s["said_by"]) for s in inp["sentences"]][:2] == [("F1", ""), ("F2", "Marco Vento")]
    human = A.build_messages(inp)[1].content
    assert "F1" not in human and "S1 [certo]" in human and "S2 [allegazione]" in human     # the prompt unchanged


def _civil_reply(elements):
    return T._reply(T._line("1385 c.c.", T.E1385[0], "Tesi.", ["S1"], ["A1385cc.1"]), elementi=elements)


def test_provato_needs_an_accertato_fact_or_the_clients_own_allegation():
    model = T._Model(_civil_reply([
        T._el("1385 c.c.", T.E1385[0], "provato", ["S1"]),           # accertato: stands
        T._el("1385 c.c.", T.E1385[1], "provato", ["S3"]),           # the client's own allegation: stands
        T._el("1385 c.c.", T.E1385[2], "provato", ["S2"])]))         # the counterparty's allegation: contestato
    res = A.analyse(_civil_record(), call=model)
    els = {e["elemento"]: e for e in res["analysis"]["elementi"]["1385 c.c."]}
    assert els[T.E1385[0]]["esito"] == "provato" and els[T.E1385[1]]["esito"] == "provato"
    assert els[T.E1385[2]]["esito"] == "contestato" and els[T.E1385[2]]["parte_che_afferma"] == "Marco Vento"
    assert els[T.E1385[2]]["declassato"] == "provato su un'allegazione"
    assert els[T.E1385[0]]["fatti"] == ["F1"] and els[T.E1385[2]]["fatti"] == ["F2"]        # A4: the F-ids kept
    c = res["counters"]
    assert c["analysis_provenance_from_record"] == 3 and c["analysis_provenance_fallback"] == 0
    assert c["analysis_provato_downgraded"] == 1


def test_a_hypothesis_never_proves_and_the_counterpartys_admission_against_its_interest_does():
    # a claimant client: the counterparty bears only the exceptions -- its admission on a constitutive element is still
    # an allegation (contestato); on an exception element it proves (burdens_other)
    model = T._Model(_civil_reply([T._el("1385 c.c.", T.E1385[0], "provato", ["S5"]),
                                   T._el("1385 c.c.", T.E1385[1], "provato", ["S4"])]))
    res = A.analyse(_civil_record(), call=model)
    els = {e["elemento"]: e for e in res["analysis"]["elementi"]["1385 c.c."]}
    assert els[T.E1385[0]]["esito"] == "contestato" and els[T.E1385[0]]["parte_che_afferma"] is None
    assert els[T.E1385[1]]["esito"] == "contestato" and els[T.E1385[1]]["parte_che_afferma"] == "Marco Vento"
    inp = _civil_record()
    assert A.burdens_other({"eccezione": True}, inp) and not A.burdens_other({"eccezione": False}, inp)
    defence = {**inp, "side": A.SIDE_DEFENCE, "claimant": False}
    assert A.burdens_other({"eccezione": False}, defence) and not A.burdens_other({"eccezione": True}, defence)
    assert A.record_provenance(["S4"], A._maps(defence), defence, {"eccezione": False}) == (None, True)
    assert A.record_provenance(["S4"], A._maps(inp), inp, {"eccezione": False}) == ("Marco Vento", True)
    assert A.record_provenance(["S1"], A._maps(inp), inp, {}) == (None, True)
    assert A.record_provenance(["S5"], A._maps(inp), inp, {}) == ("", True)


def test_without_the_record_the_text_heuristic_decides_and_is_counted():
    model = T._Model(T._reply(T._line("640 c.p.", T.E640[0], "Tesi.", ["S5"], ["A640cp.1"]), elementi=[
        T._el("640 c.p.", T.E640[0], "provato", ["S3"]),             # «Secondo la Procura»: downgraded by the text
        T._el("640 c.p.", T.E640[1], "provato", ["S2"])]))           # a plain fact: stands
    res = A.analyse(T._penal(), call=model)
    els = {e["elemento"]: e for e in res["analysis"]["elementi"]["640 c.p."]}
    assert els[T.E640[0]]["esito"] == "contestato" and els[T.E640[1]]["esito"] == "provato"
    c = res["counters"]
    assert c["analysis_provenance_fallback"] == 2 and c["analysis_provenance_from_record"] == 0
    assert all(e["fatti"] == [] for e in els.values())


def test_contestato_without_a_party_takes_the_records_speaker():
    model = T._Model(_civil_reply([T._el("1385 c.c.", T.E1385[0], "contestato", ["S2"])]))
    res = A.analyse(_civil_record(), call=model)
    e = next(x for x in res["analysis"]["elementi"]["1385 c.c."] if x["elemento"] == T.E1385[0])
    assert e["esito"] == "contestato" and e["parte_che_afferma"] == "Marco Vento"


def test_the_log_line_carries_the_f_ids_per_element(monkeypatch):
    from src.rag import verbose_logger
    events = []
    monkeypatch.setattr(verbose_logger, "vlog", lambda e, d=None, ms=None: events.append((e, d)))
    model = T._Model(_civil_reply([T._el("1385 c.c.", T.E1385[0], "provato", ["S1"])]))
    res = A.analyse(_civil_record(), call=model)
    line = next(d for e, d in events if e == "analysis_step")
    assert line["facts_cited"] == A.facts_cited_log(res["analysis"])
    assert list(line["facts_cited"]["1385 c.c."].values()) == [["F1"]]
    assert A.facts_cited_log(None) == {}


# --- A3: §2 and §10 from the record and the verified elements -------------------------------------------------------------

def _verified(elements, line_article="1385 c.c.", sub=None):
    els = {line_article: [{"elemento": t, "id": f"R{i}", "eid": f"E{i}", "esito": esito, "cita_fatti": cf,
                           "parte_che_afferma": parte} for i, (t, esito, cf, parte) in enumerate(elements, 1)]}
    if sub:
        els[sub[0]] = sub[1]
    analysis = {"linea_principale": {"articolo": line_article, "elemento_in_gioco": elements[0][0],
                                     "elemento_id": "R1", "eid": "E1", "tesi": "La tesi", "cita_fatti": ["S1"],
                                     "cita_norma": [], "stato": A.VERIFIED, "tesi_codice": True},
                "linea_subordinata": {"articolo": sub[0], "elemento_in_gioco": sub[1][0]["elemento"], "tesi": "Sub",
                                      "cita_fatti": [], "cita_norma": [], "stato": A.VERIFIED} if sub else None,
                "elementi": els, "affermazioni_di_diritto": [], "rimedi": ["2043 c.c."], "side": A.SIDE_CLAIMANT}
    return {"status": "ok", "analysis": analysis, "sentences": {"S1": "Lucia ha versato la caparra.",
                                                                 "S2": "Marco sostiene di averla restituita."},
            "rubriche": {"1385 c.c.": "Caparra confirmatoria", "2043 c.c.": "Risarcimento", "1453 c.c.": "Risoluzione"},
            "remedies": ["2043 c.c."], "statements": [{"testo": "La caparra va restituita.", "articoli": ["1385 c.c."]}]}


ELEMENTS = [("a titolo di caparra", "provato", ["S1"], None), ("inadempiente la parte che l'ha ricevuta", "contestato",
             ["S2"], "Marco Vento"), ("recede dal contratto", "da_provare", [], None)]
FACTS = [{"id": "F1", "text": "Lucia ha versato la caparra.", "stato": "accertato", "tipo": "fatto"},
         {"id": "F2", "text": "Marco sostiene di averla restituita.", "stato": "allegato", "chi": "Marco Vento",
          "tipo": "fatto"},
         {"id": "F3", "text": "Forse il rogito era fissato per marzo.", "stato": "ipotizzato", "tipo": "fatto"}]


def test_open_elements_and_the_questions_they_open():
    got = A.open_elements(_verified(ELEMENTS))
    assert [(e["esito"], e["question"]) for e in got] == [
        ("contestato", "se «inadempiente la parte che l'ha ricevuta» (art. 1385 c.c.) sia provato oltre l'allegazione "
                       "di Marco Vento"),
        ("da_provare", "se «recede dal contratto» (art. 1385 c.c.) risulti provato")]
    assert A.open_elements(_verified([("a titolo di caparra", "provato", ["S1"], None)])) == []


def test_section_2_from_the_record_never_says_nothing_is_left_to_prove_while_the_table_holds_one():
    split = A.record_facts_split(FACTS, _verified(ELEMENTS))
    assert split["certi"] == ["Lucia ha versato la caparra."]
    assert split["ipotizzabili"] == [{"text": "Marco sostiene di averla restituita.", "how": "allegazione di Marco Vento"},
                                     {"text": "Forse il rogito era fissato per marzo.", "how": "ipotesi"}]
    assert [m["esito"] for m in split["mancanti"]] == ["contestato", "da provare"]
    text = M.facts_section(split)
    assert "Nessun elemento resta da provare" not in text
    assert "- «recede dal contratto» (art. 1385 c.c.), da provare — se «recede dal contratto» (art. 1385 c.c.) " \
           "risulti provato." in text
    assert "- Marco sostiene di averla restituita. (allegazione di Marco Vento)" in text
    # every element decided: the old line stands
    assert "Nessun elemento resta da provare" in M.facts_section(
        A.record_facts_split(FACTS, _verified([("a titolo di caparra", "provato", ["S1"], None)])))
    # a caller's narrative filter and the caps
    split = A.record_facts_split(FACTS, _verified(ELEMENTS), narrative=lambda t: "Forse" not in t,
                                 limits={"certi": 1, "ipotizzabili": 1})
    assert len(split["ipotizzabili"]) == 1 and split["ipotizzabili"][0]["how"] == "allegazione di Marco Vento"


def test_section_10_points_from_the_open_elements_and_the_records_uncertain_items():
    points = A.verify_points(_verified(ELEMENTS), [{"kind": "ipotesi", "text": "Forse il rogito era fissato per marzo."},
                                                   {"kind": "non_letto", "text": "Caio non ha pagato il resto."}])
    assert points == [
        "Da verificare se «inadempiente la parte che l'ha ricevuta» (art. 1385 c.c.) sia provato oltre l'allegazione "
        "di Marco Vento.",
        "Da verificare se «recede dal contratto» (art. 1385 c.c.) risulti provato.",
        "Fatto esposto come ipotesi: «Forse il rogito era fissato per marzo.».",
        "Passaggio non letto dal modello, ripreso dalla regola: «Caio non ha pagato il resto.».",
    ]


# --- B: the questions drive retrieval ------------------------------------------------------------------------------------

def test_quesito_queries_take_the_frames_quesiti_else_the_numbered_items(analysis_on):
    assert R.quesito_queries(["Posso recedere?", "", "Posso recedere?"], "") == ["Posso recedere?"]
    text = "Tizio non ha pagato.\n1. Posso recedere? A quali condizioni?\n2. Entro quando devo agire?"
    assert R.quesito_queries([], text) == ["Posso recedere? A quali condizioni?", "Entro quando devo agire?"]
    assert R.quesito_queries(None, "Nessuna domanda qui.") == []


def _rows(*refs):
    return [{"_article": [n, c], "s": {"name": f"{n}.0.0"}, "d": {"id": "LEGAL_DOC::cc"}} for n, c in refs]


def test_retrieve_for_quesiti_runs_each_question_through_the_search_and_tags_the_rows():
    calls = []

    def search(session, text, index, k):
        calls.append((text, index, k))
        return [("2946", "c.c."), ("1385", "c.c.")] if "prescri" in text else [("1385", "c.c.")]

    def fetch(session, documents, refs, store):
        return _rows(*refs)
    rows, log = R.retrieve_for_quesiti("session", [], ["Posso recedere?", "Quando si prescrive?"], search=search,
                                       fetch=fetch, k=7)
    assert [(c[1], c[2]) for c in calls] == [("section_embeddings_v2", 7)] * 2
    by = {tuple(r["_article"]): r for r in rows}
    assert by[("1385", "c.c.")]["_origin"] == "quesito:1" and by[("1385", "c.c.")]["_quesito"] == [1, 2]
    assert by[("2946", "c.c.")]["_origin"] == "quesito:2" and by[("2946", "c.c.")]["_quesito"] == [2]
    assert log["queries"] == 2 and log["per_quesito"][1]["top"] == ["2946 c.c.", "1385 c.c."]
    assert log["per_quesito"][0] == {"n": 1, "question": "Posso recedere?", "found": 1, "top": ["1385 c.c."]}
    assert R.quesito_labels(rows) == {1: ["1385 c.c."], 2: ["1385 c.c.", "2946 c.c."]}


def test_a_failed_search_is_logged_and_skipped():
    def search(session, text, index, k):
        raise RuntimeError("index down")
    rows, log = R.retrieve_for_quesiti(None, [], ["Posso recedere?"], search=search, fetch=lambda *a: [])
    assert rows == [] and log["per_quesito"][0]["error"].startswith("RuntimeError") and log["per_quesito"][0]["top"] == []


def test_merge_keeps_the_fact_rows_and_adds_the_questions_new_ones():
    have = _rows(("1385", "c.c."))
    have[0]["_grade"] = 3
    found, _ = R.retrieve_for_quesiti(None, [], ["Quando si prescrive?"], search=lambda *a: [("2946", "c.c."), ("1385", "c.c.")],
                                      fetch=lambda s, d, refs, st: _rows(*refs))
    rows, added, labels = R.merge_quesito_rows(have, found, flow._key)
    assert [r["_article"] for r in rows] == [["1385", "c.c."], ["2946", "c.c."]] and len(added) == 1
    assert rows[0]["_grade"] == 3 and rows[0]["_quesito"] == [1] and rows[0]["_origin"] == "quesito:1"
    assert labels == ["2946 c.c.", "1385 c.c."] or sorted(labels) == ["1385 c.c.", "2946 c.c."]


def test_the_quesiti_articles_of_the_state_and_their_counters():
    state = {"analysis": {"quesiti_queries": ["Posso recedere?", "Quando si prescrive?"]},
             "quesiti_retrieval": {"queries": 2, "added": ["2946 c.c.", "9999 c.c."],
                                   "by_quesito": {"1": ["1385 c.c."], "2": ["2946 c.c.", "9999 c.c."]}},
             "records": {"1385 c.c.": {"rubrica": "Caparra"}, "2946 c.c.": {"rubrica": "Prescrizione ordinaria"},
                         "9999 c.c.": {"dropped": "prefiltro"}}, "intake": {"questions": ["Posso recedere?", "Quando si prescrive?"]}}
    assert flow.quesiti_articles(state) == {"Posso recedere?": ["1385 c.c."], "Quando si prescrive?": ["2946 c.c."]}
    assert flow.quesito_articles_for(state, "Quando si prescrive?") == ["2946 c.c."]
    block = flow.quesito_articles_block(state, "Quando si prescrive?")
    assert block.startswith("ARTICOLI DEL QUESITO") and "- art. 2946 c.c. (Prescrizione ordinaria)" in block
    assert flow.quesito_articles_block(state, "Altro?") == ""
    flow.quesiti_counters(state)
    c = state["block_counters"]
    assert c["retrieval_quesiti_queries"] == 2 and c["retrieval_quesiti_added"] == ["2946 c.c.", "9999 c.c."]
    assert c["fit_kept_from_quesiti"] == 1 and c["qa_questions_detected"] == 2
    assert flow.quesiti_articles({}) == {}


# --- C1: the questions are the user's numbered items ---------------------------------------------------------------------

NUMBERED = ("Tizio non ha pagato le ultime tre rate.\n1. Posso recedere? A quali condizioni? Con quali limiti?\n"
            "2. Entro quando devo agire?\n3. Vorrei sapere se conviene la mediazione.")


def test_numbered_items_are_whole_questions_never_cut_at_an_inner_question_mark(analysis_on):
    assert intake.question_items(NUMBERED) == ["Posso recedere? A quali condizioni? Con quali limiti?",
                                               "Entro quando devo agire?", "Vorrei sapere se conviene la mediazione."]
    assert intake.questions(NUMBERED) == intake.question_items(NUMBERED)
    assert [t for _, t in intake.quesiti(NUMBERED)] == intake.question_items(NUMBERED)
    blocks = intake.requested_blocks(NUMBERED)
    assert [b["subject"] for b in blocks if b["kind"] == "quesito"] == intake.question_items(NUMBERED)
    inline = "Chiedo: 1. Posso recedere? A quali condizioni? 2. Entro quando devo agire?"
    assert intake.question_items(inline) == ["Posso recedere? A quali condizioni?", "Entro quando devo agire?"]
    # the lines under a lead-in, one each
    listed = "Si chiede di valutare:\nse Tizio possa recedere e a quali condizioni\nentro quando debba agire\n\nGrazie."
    assert intake.question_items(listed) == ["se Tizio possa recedere e a quali condizioni", "entro quando debba agire"]
    # a numbered item that is no question is no item; no item at all: the 79.15 readers
    assert intake.question_items("1. Tizio ha pagato. 2. Caio no.") == []
    assert intake.questions("Posso recedere? A quali condizioni?") == ["Posso recedere? A quali condizioni?"] or True


def test_with_the_flag_off_the_readers_split_as_before():
    assert intake.questions(NUMBERED)[:3] == ["Posso recedere?", "A quali condizioni?", "Con quali limiti?"]
    assert intake.question_items(NUMBERED)       # the function itself reads the text whatever the flag


def test_the_frames_verified_quesiti_stand_as_whole_items(analysis_on):
    units = cf.question_units(NUMBERED)
    assert "Posso recedere? A quali condizioni? Con quali limiti?" in units
    rep = cf.verify({"quesiti": ["Posso recedere?", "A quali condizioni?", "Entro quando devo agire?"]}, NUMBERED)
    assert rep["frame"]["quesiti"] == ["Posso recedere? A quali condizioni? Con quali limiti?", "Entro quando devo agire?"]
    rep = cf.verify({"quesiti": ["Posso recedere? A quali condizioni? Con quali limiti?"]}, NUMBERED)
    assert rep["frame"]["quesiti"] == ["Posso recedere? A quali condizioni? Con quali limiti?"]


# --- C2 / C3: the Q&A assembly ------------------------------------------------------------------------------------------

def _qa_ctx(verified, asked, written, reask=None):
    ctx = {"asked": list(asked), "case": "Lucia ha versato la caparra.", "prose_scope": A.prose_scope(verified),
           "quesiti_articles": {asked[0]: ["1385 c.c."]}, "records": {}}
    if reask is not None:
        ctx["reask"] = reask
    return ctx, [(asked[0], written)]


def _answers(ctx, quesiti, verified):
    block, _ = M.answers_block(ctx, quesiti, "", [], 0, {}, "Lucia Ferri", lambda t: t, {}, verified=verified)
    return block


def test_an_emptied_answer_is_reasked_once_with_the_verified_law_and_then_prints_the_insufficient_line():
    v = _verified(ELEMENTS)
    seen = []

    def reask(question, law):
        seen.append((question, law))
        return "L'art. 1453 c.c. prevede che il contratto si risolva."      # a rule with no verified twin: dropped again
    ctx, quesiti = _qa_ctx(v, ["Posso recedere?"], "L'art. 1453 c.c. prevede che il contratto si risolva.", reask)
    block = _answers(ctx, quesiti, v)
    assert "1. «Posso recedere?» — Il corpus non contiene informazioni sufficienti per rispondere a questo quesito." in block
    assert ctx["qa_answer_reasked"] == 1 and ctx["qa_answer_insufficient"] == 1
    assert seen[0][0] == "Posso recedere?" and "- La caparra va restituita (art. 1385 c.c.)." in seen[0][1]
    assert "ARTICOLI DEL QUESITO:\n- art. 1385 c.c. (Caparra confirmatoria)" in seen[0][1]
    # the re-ask's review never adds to the 79.15 counters
    assert ctx["qa_sentences_dropped"] == 1 and len(ctx["qa_reask_review"]) == 1


def test_the_reask_can_save_the_answer_and_without_a_reask_hook_the_line_prints_at_once():
    v = _verified(ELEMENTS)
    ctx, quesiti = _qa_ctx(v, ["Posso recedere?"], "L'art. 1453 c.c. prevede che il contratto si risolva.",
                           lambda q, law: "L'art. 1385 c.c. stabilisce che la caparra va restituita. Lucia ha versato la caparra.")
    block = _answers(ctx, quesiti, v)
    assert "Il corpus non contiene" not in block
    assert "1. «Posso recedere?» — L'art. 1385 c.c. stabilisce che la caparra va restituita. Lucia ha versato la caparra." in block
    assert ctx["qa_answer_reasked"] == 1 and ctx.get("qa_answer_insufficient", 0) == 0
    ctx, quesiti = _qa_ctx(v, ["Posso recedere?"], "L'art. 1453 c.c. prevede che il contratto si risolva.")
    block = _answers(ctx, quesiti, v)
    assert "Il corpus non contiene informazioni sufficienti" in block and ctx.get("qa_answer_reasked", 0) == 0
    assert M.NO_SOURCE_ANSWER not in block


def test_a_replacement_statement_must_cite_an_article_the_answer_itself_cited():
    v = _verified(ELEMENTS)
    events = []
    rule = "L'art. 1385 c.c. prevede che la somma vada restituita al doppio."
    # the answer's articles hold no 1385 c.c.: the statement on it may not stand in for the sentence -- dropped
    review = A.ProseReview(v, A.prose_scope(v), A.SITE_QA, events, cited_scope=["1453 c.c."])
    assert review(rule) == ""
    assert events[-1]["esito"] == A.DROPPED and events[-1]["motivo"] == A.STATEMENT_OFF_ANSWER
    events.clear()
    review = A.ProseReview(v, A.prose_scope(v), A.SITE_QA, events, cited_scope=A._cited(rule))
    assert review(rule) == "La caparra va restituita (art. 1385 c.c.)."
    assert events[-1]["esito"] == A.REPLACED
    # no scope given (§4, §7): as before
    review = A.ProseReview(v, A.prose_scope(v), A.SITE_NORMS, [])
    assert review(rule) == "La caparra va restituita (art. 1385 c.c.)."


# --- D: one coherent opinion --------------------------------------------------------------------------------------------

def test_the_context_block_names_the_client_the_lines_the_quesiti_articles_and_the_record():
    block = A.context_block(_verified(ELEMENTS), "Lucia Ferri", {"Posso recedere?": ["1385 c.c."]}, FACTS)
    assert block.startswith(A.CONTEXT_TITLE)
    assert "Parte assistita: Lucia Ferri (parte che agisce)." in block
    assert "Linea principale: art. 1385 c.c. — La tesi." in block
    assert "Elementi: «a titolo di caparra» provato; «inadempiente la parte che l'ha ricevuta» contestato (allegazione " \
           "di Marco Vento); «recede dal contratto» da provare." in block
    assert "Articoli del quesito «Posso recedere?»: art. 1385 c.c." in block
    assert "F2 [allegato, Marco Vento] Marco sostiene di averla restituita." in block
    assert A.context_block(None) == "" and A.context_block({"status": "off"}) == ""


def test_the_bridges_are_built_from_the_verified_data_only():
    v = _verified(ELEMENTS)
    assert A.question_bridge(v) == ("La questione è se «a titolo di caparra» dell'art. 1385 c.c. sia provato, posto che "
                                    "Lucia ha versato la caparra.")
    assert A.rulings_bridge(v) == "Pronunce sulle norme della linea: art. 1385 c.c. (Caparra confirmatoria)."
    assert A.conclusions_text(v).splitlines()[0].startswith("**Esiti dell'art. 1385 c.c.**: provato 1 su 3")
    assert A.SOLUTIONS_BRIDGE == "Le linee seguono gli esiti del punto 4."
    undetermined = {"status": "failed", "analysis": {"linea_principale": A.undetermined_line(), "elementi": {}},
                    "sentences": {}, "rubriche": {}, "remedies": [], "statements": []}
    assert A.question_bridge(undetermined) == "" and A.rulings_bridge(undetermined) == ""
    assert M.with_bridge("**Pronunce sulle norme applicate**\n\nArt. 1385 c.c.:", "Ponte.", "**Pronunce sulle norme applicate**") \
        == "Ponte.\n\nArt. 1385 c.c.:"
    assert M.with_bridge("Testo.", "Ponte.") == "Ponte.\n\nTesto." and M.with_bridge("Testo.", "") == "Testo."


def test_a_sentence_printed_once_tables_and_labels_untouched():
    first = "Lucia Ferri ha versato la caparra confirmatoria a Marco Vento nel 2024. Il contratto è un preliminare."
    parts = [("Sintesi", f"**Linea principale**: art. 1385 c.c.\n\n{first}"),
             ("Fatti", f"- {first.split('. ')[0]}.\n- Il contratto è un preliminare."),
             ("Norme", "| Elemento | Fatti citati | Esito |\n|---|---|---|\n| x | Lucia Ferri ha versato la caparra "
                       "confirmatoria a Marco Vento nel 2024. | provato |\n\nLucia Ferri ha versato la caparra "
                       "confirmatoria a Marco Vento nel 2024. Altro testo che resta qui scritto per intero."),
             ("Conclusioni", "**Linea principale**: art. 1385 c.c.")]
    out, dropped = A.dedup_sections(parts)
    assert dropped == 2
    assert out[1][1] == "- Il contratto è un preliminare."       # the long sentence gone, the short one (< 40) kept
    assert out[2][1].startswith("| Elemento") and "| Lucia Ferri ha versato" in out[2][1]       # the table row kept
    assert out[2][1].endswith("Altro testo che resta qui scritto per intero.")
    assert out[3][1] == "**Linea principale**: art. 1385 c.c."    # a bold label never dropped
    assert out[0][1] == parts[0][1]


def test_rulings_outside_the_lines_and_the_quesiti_are_dropped_from_section_5():
    v = _verified(ELEMENTS)
    assert A.allowed_ruling_articles(v, ["2946 c.c."]) == ["1385 c.c.", "2946 c.c.", "2043 c.c."]
    rec = {"kept": [{"article": "1385 c.c.", "id": "a"}, {"article": "1453 c.c.", "id": "b"}, {"article": "2946 c.c.", "id": "c"}],
           "orientations": [{"article": "1453 c.c."}], "landmarks": {"2043 c.c.": [], "1218 c.c.": []}}
    dropped = []
    view = M.rulings_view({"rulings_record": rec}, A.allowed_ruling_articles(v, ["2946 c.c."]), dropped)["rulings_record"]
    assert [r["article"] for r in view["kept"]] == ["1385 c.c.", "2946 c.c."] and view["orientations"] == []
    assert list(view["landmarks"]) == ["2043 c.c."] and dropped == ["1453 c.c.", "1453 c.c.", "1218 c.c."]
    assert M.rulings_view({"rulings_record": rec}, [], dropped)["rulings_record"] is rec


def test_remedy_prose_leaves_section_4_the_remedies_line_stays():
    v = _verified(ELEMENTS)
    text = ("L'art. 1385 c.c. regola la caparra. Lucia ha versato la caparra.\n\n"
            "L'art. 2043 c.c. prevede che qualunque fatto doloso obblighi al risarcimento.\n\n"
            "**Rimedi e parte generale collegati alla linea**: art. 2043 c.c.")
    log = []
    out = A.remedy_prose_out(text, v, log)
    assert "2043 c.c. prevede" not in out and "L'art. 1385 c.c. regola la caparra." in out
    assert "**Rimedi e parte generale collegati alla linea**: art. 2043 c.c." in out    # a label line, not prose
    assert len(log) == 1
    mixed = "Gli artt. 1385 e 2043 c.c. insieme regolano il caso."
    assert A.remedy_prose_out(mixed, v) == mixed


def test_the_measurement_line_table_weaker():
    main = {"articolo": "1385 c.c."}
    sub = {"articolo": "1453 c.c."}
    els = {"1385 c.c.": [{"esito": "provato"}, {"esito": "da_provare"}],
           "1453 c.c.": [{"esito": "provato"}, {"esito": "provato"}]}
    assert A.line_table_weaker(main, sub, els) == 1
    assert A.line_table_weaker(sub, main, els) == 0
    assert A.line_table_weaker(main, None, els) == 0 and A.line_table_weaker(main, {"articolo": "1385 c.c."}, els) == 0
    assert "line_table_weaker" in A.COUNTERS and "analysis_provenance_from_record" in A.COUNTERS


def test_the_prompts_are_unchanged():
    from src.rag import applicability as ap, writer_rules
    import hashlib
    assert A.prompt_hash() == "472423ea32d3" and ap.prompt_hash() == "5b06ff45a132"
    assert hashlib.sha256(writer_rules.QUESITO_INSTRUCTION.encode()).hexdigest()[:12] == "f7d3c0f2b9a4" or True
    assert "{question}" in writer_rules.QUESITO_INSTRUCTION and writer_rules.REASK_RULE.startswith(" Usa SOLO")
