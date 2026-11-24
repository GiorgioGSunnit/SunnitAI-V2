"""v79.18.1 -- section C of 79.18.0, the penal defence review: C1 the defence's forza order, C2 a G-id as the line's
jurisprudential condition, C3 the riqualificazione (a «riqualificare» linea or the code's line; the standing quesito),
C4 the objection's answer consistent by id, C5 passi as concrete proof, C6 the three riti always, C7 the body without
the frame's unverified fields, timestamps and «v. punto N», the accusation's «contestazione», C8 rulings on general-part
articles only on the line's passage, C9 a rule on article X cites an A-id of X. Synthetic texts and replies only."""
import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import test_v79_15_0_analysis as T  # noqa: E402
import test_v79_18_0 as Z  # noqa: E402
from src.rag import analysis_step as A  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import lawyer_method as M  # noqa: E402
from src.rag import strategy_step as S  # noqa: E402
from src.rag import verified_material as V  # noqa: E402
from src.rag.nodes import retrieval as RT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("replay_cases_181", ROOT / "scripts" / "replay_cases.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


def _penal_inp(rulings=True):
    """A penal defence: the charge 640 c.p. (E1 contestato by the client's own words S5, E2 da_provare, E3 provato),
    646 c.p. in the set (an alternative), one ruling on 640 c.p."""
    facts = [{"id": "S1", "text": "Paolo Sarti ha venduto online a Giulia Neri una bicicletta.", "certainty": "accertato", "said_by": "", "own": True},
             {"id": "S3", "text": "Secondo la Procura, Paolo Sarti ha usato artifizi e un falso annuncio.", "certainty": "allegato",
              "said_by": "la Procura", "own": False},
             {"id": "S5", "text": "Paolo Sarti nega di aver mai usato artifizi o raggiri.", "certainty": "allegato", "said_by": "Paolo Sarti",
              "own": True}]
    return {"pretesa": "accusa: «truffa» (art. 640 c.p.) contro Paolo Sarti", "client": "Paolo Sarti", "side": "difesa",
            "charge": "640 c.p.", "penal": True, "rito": "penale",
            "articles": [
                {"article": "640 c.p.", "token": "640cp", "rubrica": "Truffa", "tier": "si_verificato", "in_set": True, "valutato": True,
                 "elements": [Z._el("A640cp:E1", "con artifizi o raggiri", "contestato", 1, ["S3", "S5"]),
                              Z._el("A640cp:E2", "inducendo taluno in errore", "da_provare", 1),
                              Z._el("A640cp:E3", "procura a sé un ingiusto profitto con altrui danno", "provato", 1, ["S1"])]},
                {"article": "646 c.p.", "token": "646cp", "rubrica": "Appropriazione indebita", "tier": "dipende_verificato", "in_set": True,
                 "valutato": False,
                 "elements": [Z._el("A646cp:E1", "si appropria il denaro o la cosa mobile altrui", "da_provare", 1, valutato=False)]}],
            "facts": facts, "allegations": ["S3"], "quesiti": ["Quali rischi corre Paolo Sarti?"], "quesiti_articles": {},
            "lines": [{"kind": "linea_principale", "articolo": "640 c.p.", "eid": "A640cp:E1", "stato": "verificata"}], "honesty": "",
            "rulings": [{"id": "G1", "estremi": "Cass. pen., Sez. 2, n. 10/2025", "article": "640 c.p.",
                         "passage": "l'induzione in errore richiede …", "element_id": ""}] if rulings else []}


def _l(art, eid, tipo, rapporto="alternativa", cf=(), cg=(), cond=None, ob=None):
    return {"articolo": art, "eid": eid, "condizione": cond, "tipo": tipo, "rapporto": rapporto, "perche_prima": "perché",
            "cita_fatti": list(cf), "cita_pronunce": list(cg), "obiezione": ob if ob is not None else
            {"testo": "L'accusa obietterà.", "risposta_cita_fatti": ["S5"], "risposta_cita_elementi": ["A640cp:E1"], "cita_pronunce": []}}


def _reply(linee, **more):
    return {"linee": linee, "passi": [], "passi_immediati": [], "contro_difesa": [], "verifiche_documentali": [], **more}


# --- C1, C2 -------------------------------------------------------------------------------------------------------------

def test_c1_the_defence_order_own_facts_then_ruling_condition_then_bare_da_provare():
    inp = _penal_inp()
    art = inp["articles"][0]
    e1, e2, e3 = art["elements"]
    assert S.defence_rank(e1, inp) == 4                                   # contestato by the client's own S5
    assert S.defence_rank(e2, inp, {"cita_pronunce": ["G1"]}) == 3        # da_provare with a ruling's condition
    assert S.defence_rank(e2, inp, {"condizione": "G1"}) == 3
    assert S.defence_rank(e2, inp) == 1                                   # bare da_provare
    assert S.defence_rank(dict(e1, cita_fatti=["S3"]), inp) == 2          # contestato on the accusation's words alone
    assert S.defence_rank(e3, inp) == 0
    f1 = S.forza("640 c.p.", "A640cp:E1", inp)
    f2 = S.forza("640 c.p.", "A640cp:E2", inp, {"cita_pronunce": ["G1"]})
    f2b = S.forza("640 c.p.", "A640cp:E2", inp)
    assert f1["key"] == [4, 2] and f2["key"] == [3, 2] and f2b["key"] == [1, 2] and f1["testo"] == "provato 1 / contestato 1 / da provare 1"
    assert tuple(f1["key"]) > tuple(f2["key"]) > tuple(f2b["key"])        # never «most da_provare»
    # through finish: the bare da_provare line put first by the model is moved below the own-facts contestato
    checked = S.verify(_reply([_l("A640cp", "A640cp:E2", "contestare", "principale"),
                               _l("A640cp", "A640cp:E1", "contestare", "subordinata", ["S5"]),
                               _l("A646cp", "A646cp:E1", "riqualificare", "alternativa", ["S1"])]), inp)
    a = S.finish(checked, inp)
    assert [x["eid"] for x in a["linee"]] == ["A640cp:E1", "A640cp:E2", "A646cp:E1"] and a["reordered"] is True


def test_c2_a_ruling_attached_to_the_eid_is_the_lines_condition_verified_by_id_and_article():
    inp = _penal_inp()
    checked = S.verify(_reply([_l("A640cp", "A640cp:E2", "contestare", "principale", cond="G1"),
                               _l("A646cp", "A646cp:E1", "riqualificare", "subordinata", ["S1"], cond="G1"),      # not its article
                               _l("A640cp", "A640cp:E1", "contestare", "alternativa", ["S5"], cond="G9")]), inp)   # no such ruling
    assert [x["condizione"] for x in checked["linee"]] == ["G1", "", ""]
    assert checked["counters"]["strategy_bad_id"] == 2
    a = S.finish(checked, inp)
    first = next(x for x in a["linee"] if x["eid"] == "A640cp:E2")
    assert first["forza"]["key"][0] == 3                      # the condition counts as the ruling's (C1 b)
    sol = S.solutions_text({"status": S.OK, "analysis": a}, Z._view(inp))
    assert "| inducendo taluno in errore [G1] | contestare |" in sol


# --- C3 -----------------------------------------------------------------------------------------------------------------

def test_c3_the_penal_defence_needs_a_riqualificare_line_else_the_codes_line_after_one_reask():
    inp = _penal_inp()
    assert S.penal_defence(inp) and [a["article"] for a in S.alternative_articles(inp)] == ["646 c.p."]
    no_riq = _reply([_l("A640cp", "A640cp:E1", "contestare", "principale", ["S5"]), _l("A640cp", "A640cp:E2", "contestare")])
    checked = S.verify(no_riq, inp)
    assert checked["reasons"] == ["difesa penale: manca una linea «riqualificare» su un altro articolo dell'insieme (A646cp)"]
    assert "DIFESA PENALE: una linea «riqualificare» su un altro articolo dell'insieme è richiesta, se ve n'è uno: art. 646 c.p." \
        in S.build_messages(inp)[1].content
    model = T._Model(no_riq, no_riq)
    res = S.analyse(inp, call=model)
    assert res["status"] == S.OK and len(model.seen) == 2 and "se ve n'è uno: art. 646 c.p." in model.seen[1]["messages"][-1].content
    riq = res["analysis"]["riqualificazione"]
    assert riq == {"linea": None, "articolo": None, "n": None, "alternative": ["646 c.p."]}
    assert res["counters"]["strategy_riqualificazione_missing"] == 1
    sol = S.solutions_text(res, Z._view(inp))
    assert "**Qualificazione alternativa**\n\nNessuna qualificazione alternativa tra le norme applicabili (art. 646 c.p.)." in sol
    # with the riqualificare line: no reason, the line said
    with_riq = _reply(no_riq["linee"] + [_l("A646cp", "A646cp:E1", "riqualificare", "subordinata", ["S1"])])
    res = S.analyse(inp, call=T._Model(with_riq))
    assert res["counters"]["strategy_reasked"] == 0 and res["analysis"]["riqualificazione"]["articolo"] == "646 c.p."
    assert "**Qualificazione alternativa**\n\nArt. 646 c.p. (linea 3)." in S.solutions_text(res, Z._view(inp))
    assert S.replay_summary(res)["riqualificazione"]["articolo"] == "646 c.p."
    # a claimant, or a set with no other article: nothing required, nothing printed
    civil = Z._inp()
    assert not S.penal_defence(civil) and S.finish(S.verify(_reply([Z._linea("A1385cc", "A1385cc:E1", "fondare", "principale", ["S2"]),
                                                                    Z._linea("A1385cc", "A1385cc:E2", "contestare")]), civil), civil)["riqualificazione"] is None
    alone = {**inp, "articles": inp["articles"][:1]}
    assert S.verify(no_riq, alone)["reasons"] == [] and S.finish(S.verify(no_riq, alone), alone)["riqualificazione"]["alternative"] == []
    assert "**Qualificazione alternativa**\n\nNessuna qualificazione alternativa tra le norme applicabili (nessuno)." in \
        S.solutions_text({"status": S.OK, "analysis": S.finish(S.verify(no_riq, alone), alone)}, Z._view(alone))


def test_c3_the_standing_quesito_and_its_counter():
    assert RT.STANDING_PENAL_QUESITO == "qualificazione alternativa del fatto"
    state = {"analysis": {"quesiti_queries": ["x", RT.STANDING_PENAL_QUESITO], "quesiti_standing": RT.STANDING_PENAL_QUESITO},
             "block_counters": {}, "quesiti_retrieval": {}}
    flow.quesiti_counters(state)
    assert state["block_counters"]["retrieval_quesiti_standing"] == 1
    flow.quesiti_counters({"analysis": {}, "block_counters": {}})
    assert "retrieval_quesiti_standing" in R.RETRIEVAL_COUNTERS


# --- C4, C5 -------------------------------------------------------------------------------------------------------------

def test_c4_the_answer_rests_on_elements_whose_esito_fits_the_side_else_reasked_then_unanswered():
    inp = _penal_inp()
    ob_bad = {"testo": "L'accusa dirà che il profitto è provato.", "risposta_cita_fatti": ["S5"],
              "risposta_cita_elementi": ["A640cp:E3"], "cita_pronunce": []}                 # provato: no help to the defence
    ob_ok = {"testo": "L'accusa dirà che gli artifizi sono provati.", "risposta_cita_fatti": ["S5"],
             "risposta_cita_elementi": ["A640cp:E1", "G1"], "cita_pronunce": ["G1"]}
    linee = [_l("A640cp", "A640cp:E1", "contestare", "principale", ["S5"], ob=ob_bad),
             _l("A640cp", "A640cp:E2", "contestare", "subordinata", ob=ob_ok),
             _l("A646cp", "A646cp:E1", "riqualificare", "alternativa", ["S1"])]
    checked = S.verify(_reply(linee), inp)
    assert checked["reasons"] == ["linea 1: la risposta all'obiezione si regge su A640cp:E3 (provato): esito che non la sostiene per la parte difesa"]
    assert checked["linee"][0]["obiezione"]["mismatch"] is True and checked["linee"][1]["obiezione"]["mismatch"] is False
    assert checked["linee"][1]["obiezione"]["risposta_cita_elementi"] == ["A640cp:E1", "G1"]
    # the same mismatch on the re-ask: the answer prints as unanswered, counted
    res = S.analyse(inp, call=T._Model(_reply(linee), _reply(linee)))
    assert res["counters"]["strategy_reasked"] == 1 and res["counters"]["strategy_objection_mismatch"] == 1
    items = S.objection_items(res, Z._view(inp))
    bad = next(o for o in items if o["obiezione"].startswith("L'accusa dirà che il profitto"))
    assert bad["risposta"] == "Senza risposta nei fatti narrati."
    assert res["counters"]["strategy_objections_unanswered"] == 1
    # a claimant's answer rests on a provato element
    civil = Z._inp()
    ob = {"testo": "x", "risposta_cita_fatti": ["S2"], "risposta_cita_elementi": ["A1385cc:E2"], "cita_pronunce": []}
    got, why = S._verify_objection(ob, civil, S.empty_counters(), 1)
    assert got["mismatch"] is True and "A1385cc:E2 (contestato)" in why
    got, why = S._verify_objection(dict(ob, risposta_cita_elementi=["A1385cc:E1"]), civil, S.empty_counters(), 1)
    assert got["mismatch"] is False and why == ""


def test_c5_a_passo_is_a_concrete_fact_to_acquire_never_a_restatement():
    inp = _penal_inp()
    assert S.passo_issue({"eid": "A640cp:E2", "cosa": "Provare che taluno è stato indotto in errore"}, inp).endswith(
        "riformula l'elemento invece di indicare la prova da acquisire")
    assert S.passo_issue({"eid": "A640cp:E2", "cosa": "Dimostrare l'inganno subito dalla vittima"}, inp).endswith(
        "non nomina una prova concreta da acquisire (referto, perizia, testimoni, documenti…)")
    assert S.passo_issue({"eid": "A640cp:E2", "cosa": "Acquisire le chat con Giulia Neri e sentire il corriere come testimone"}, inp) == ""
    assert S.passo_issue({"eid": "A999cp:E2", "cosa": "x"}, inp) == ""           # an unknown eid: the passi filter's business
    linee = [_l("A640cp", "A640cp:E1", "contestare", "principale", ["S5"]), _l("A640cp", "A640cp:E2", "contestare"),
             _l("A646cp", "A646cp:E1", "riqualificare", "alternativa", ["S1"])]
    bad = _reply(linee, passi=[{"eid": "A640cp:E2", "cosa": "Provare che taluno è stato indotto in errore", "cita_fatti": []}])
    good = _reply(linee, passi=[{"eid": "A640cp:E2", "cosa": "Acquisire le chat con Giulia Neri", "cita_fatti": ["S1"]}])
    assert S.verify(bad, inp)["reasons"] == ["passo A640cp:E2: riformula l'elemento invece di indicare la prova da acquisire"]
    res = S.analyse(inp, call=T._Model(bad, bad))          # still a restatement: the code's label, counted
    passi = res["analysis"]["passi"]
    assert [(p["eid"], p["codice"]) for p in passi] == [("A640cp:E1", True), ("A640cp:E2", True)]
    assert res["counters"]["strategy_passi_not_concrete"] == 1 and res["counters"]["strategy_passi_added"] == 2
    res = S.analyse(inp, call=T._Model(good))
    assert res["counters"]["strategy_reasked"] == 0 and [p["cosa"] for p in res["analysis"]["passi"] if not p["codice"]] == [
        "Acquisire le chat con Giulia Neri"]


# --- C6 -----------------------------------------------------------------------------------------------------------------

def test_c6_the_three_riti_printed_with_status_and_reason_never_the_precluded_alone():
    statuses = M.rite_statuses("reclusione da sei mesi a tre anni", "640")
    block = M.rites_block(statuses)
    rows = [x for x in block.splitlines() if x.startswith("- ")]
    assert block.startswith("**Riti alternativi**\n") and len(rows) == 3
    names = [x.split(":")[0][2:] for x in rows]
    assert names == ["Giudizio abbreviato (artt. 438 ss. c.p.p.)", "Applicazione della pena su richiesta (art. 444 c.p.p.)",
                     "Sospensione del procedimento con messa alla prova (art. 168-bis c.p.)"]
    assert all(" — " in x and any(w in x for w in (": ammesso", ": possibile", ": escluso", ": da verificare")) for x in rows), rows
    assert "Commisurazione" not in block
    heavy = M.rites_block(M.rite_statuses("reclusione da otto a venti anni", "629"))
    assert heavy.count(": escluso") >= 1 and heavy.count("\n- ") == 3        # the admitted ones stay beside the excluded
    assert M.rites_block([]) == ""
    text = ("**Riti alternativi**\n- Giudizio abbreviato (artt. 438 ss. c.p.p.): riduzione di un terzo.\n\n**Termini**\n- Opposizione entro "
            "venti giorni.\n\n- La messa alla prova è preclusa.")
    assert M.rite_lines_out(text) == "**Termini**\n- Opposizione entro venti giorni."
    assert M.RITE_STATUS_WORDS[M.PRECLUDED] == "escluso"


# --- C7 -----------------------------------------------------------------------------------------------------------------

def test_c7_the_accusation_contests_it_does_not_allege():
    assert A.is_authority("la Procura") and A.is_authority("Il PM") and A.is_authority("l'accusa") and not A.is_authority("Ettore Bassi")
    assert A.party_label("la Procura") == "contestazione dell'accusa" and A.party_label("Ettore Bassi") == "allegazione di Ettore Bassi"
    assert A.esito_text({"esito": "contestato", "parte_che_afferma": "la Procura"}) == "contestato (contestazione dell'accusa)"
    assert A.esito_text({"esito": "contestato", "parte_che_afferma": "Ettore Bassi"}) == "contestato (allegazione di Ettore Bassi)"
    v = {"client": "Paolo Sarti", "analysis": {"linea_principale": {"articolo": "640 c.p.", "stato": A.VERIFIED},
                                               "elementi": {"640 c.p.": [{"eid": "A640cp:E1", "elemento": "con artifizi o raggiri",
                                                                           "esito": "contestato", "parte_che_afferma": "la Procura"}]}},
         "sentences": {"S3": "Secondo la Procura, Paolo Sarti ha usato artifizi."}}
    assert A.open_elements(v)[0]["question"] == "se «con artifizi o raggiri» (art. 640 c.p.) sia provato oltre la contestazione dell'accusa"
    split = A.record_facts_split([{"text": "Secondo la Procura, Paolo Sarti ha usato artifizi.", "stato": "allegato", "chi": "la Procura"},
                                  {"text": "Giulia Neri afferma che la bici non è arrivata.", "stato": "allegato", "chi": "Giulia Neri"}], v)
    assert [x["how"] for x in split["ipotizzabili"]] == ["contestazione dell'accusa", "allegazione di Giulia Neri"]
    v["analysis"]["obiezioni"] = [{"fonte": "allegazione", "sid": "S3", "parte": "la Procura", "nominata": False,
                                   "elementi": [{"articolo": "640 c.p.", "eid": "A640cp:E1", "testo": "con artifizi o raggiri", "esito": "contestato"}],
                                   "fatti_cliente": []}]
    v["analysis"]["side"] = "difesa"
    assert A.objection_items(v)[0]["obiezione"] == "Contestazione dell'accusa: «Secondo la Procura, Paolo Sarti ha usato artifizi»"
    # the frame's unverified fields: logged, not in §10
    v["frame_unverified"] = {"parti": ["ruolo_nel_fatto: parole non nel testo"]}
    assert any("Campo del quadro non verificato" in p for p in A.verify_points(v))
    assert not any("Campo del quadro" in p for p in A.verify_points(v, frame_fields=False))


def test_c7_no_timestamps_and_the_article_in_place_of_v_punto_n():
    log = []
    assert M.timestamps_out("Fatto 2026-11-24T12:01:16 e 2026-11-24 12:01 e il 24.11.2026.", log) == "Fatto e e il 24.11.2026."
    assert log == ["2026-11-24T12:01:16", "2026-11-24 12:01"]
    sections = [("Norme", "La consegna del bene (da verificare) rileva ai fini dell'art. 1385 c.c. Altro testo.")]
    out, points = M.collect_points(sections, [], article_refs=True)
    assert "v. punto" not in out[0][1] and "(art. 1385 c.c.)" in out[0][1] and points
    out_old, _ = M.collect_points(sections, [])
    assert "v. punto 1" in out_old[0][1]
    assert M.point_article("Da verificare se «x» (art. 640 c.p.) risulti provato.") == "640 c.p." and M.point_article("nulla") == ""
    none, _ = M.collect_points([("Norme", "La data resta da verificare. Altro.")], [], article_refs=True)
    assert "v. punto" not in none[0][1] and "(" not in none[0][1].split("Altro")[0]


# --- C8, C9 -------------------------------------------------------------------------------------------------------------

def test_c8_rulings_on_general_part_articles_only_on_the_lines_passage_and_none_without_a_passage():
    verified = {"analysis": {"linea_principale": {"articolo": "640 c.p.", "stato": A.VERIFIED},
                             "elementi": {"640 c.p.": [{"eid": "A640cp:E1", "elemento": "con artifizi o raggiri", "esito": "contestato"}]}}}
    kept = [{"article": "640 c.p.", "estremi": "A", "passage": "gli artifizi …"},
            {"article": "640 c.p.", "estremi": "B", "passage": ""},                                  # no passage: out
            {"article": "43 c.p.", "estremi": "C", "passage": "il dolo nella truffa di cui all'art. 640 c.p."},   # cites the line
            {"article": "43 c.p.", "estremi": "D", "passage": "la colpa cosciente nel reato colposo"},            # general: out
            {"article": "41 c.p.", "estremi": "E", "passage": "gli artifizi interrompono il nesso"}]             # the element's word
    ctx = {"rulings_record": {"kept": kept, "orientations": [], "landmarks": {}}}
    dropped = []
    view = M.rulings_view(ctx, ["640 c.p.", "43 c.p.", "41 c.p."], dropped, verified)
    assert [r["estremi"] for r in view["rulings_record"]["kept"]] == ["A", "C", "E"]
    assert ctx["rulings_no_passage_dropped"] == 1 and ctx["rulings_general_part_dropped"] == 1
    assert dropped == ["640 c.p. (senza passaggio)", "43 c.p. (parte generale, passaggio non sulla linea)"]
    plain = M.rulings_view(ctx, ["640 c.p.", "43 c.p.", "41 c.p."], [])        # without the verified view: as before
    assert len(plain["rulings_record"]["kept"]) == 5
    assert M.ruling_on_line({"passage": "l'art. 640 c.p."}, ["640 c.p."], set()) and not M.ruling_on_line({"passage": "l'art. 6400"}, ["640 c.p."], set())


def test_c9_a_rule_on_article_x_must_cite_an_a_id_of_x():
    vm = {"maps": {"periods": {"A1385cc.1": {"article": "1385 c.c."}, "A1385cc.2": {"article": "1385 c.c."},
                               "A1453cc.1": {"article": "1453 c.c."}}, "sentences": {}},
          "fitted": ["1385 c.c.", "1453 c.c."], "named": [], "articles": {}}
    events = []
    assert V.review_sentence("L'art. 1385 c.c. prevede che la parte possa recedere [A1385cc.2].", vm, "norme", events) != ""
    assert V.review_sentence("L'art. 1385 c.c. prevede che la parte possa recedere.", vm, "norme", events) == ""
    assert events[-1]["motivo"] == "regola_senza_id" and events[-1]["articoli"] == ["1385 c.c."]
    # a rule naming two articles with one id: the one without its id is missing
    assert V.rule_without_id("Gli artt. 1385 e 1453 c.c. dispongono …", ["1385 c.c.", "1453 c.c."], ["A1385cc.1"], vm) == ["1453 c.c."]
    # no rule verb: a narrative mention needs no id; an id of another article does not cover
    assert V.rule_without_id("Il contratto richiama l'art. 1385 c.c. firmato dalle parti", ["1385 c.c."], [], vm) == []
    assert V.rule_without_id("L'art. 1385 c.c. prevede …", ["1385 c.c."], ["A1453cc.1"], vm) == ["1385 c.c."]
    counters = A.prose_counters(events)
    assert counters["prose_rule_without_id_dropped"] == 1 and "prose_rule_without_id_dropped" in A.RENDER_COUNTERS
    assert "prose_rule_without_id_dropped" in R.QA_COUNTERS and "rulings_no_passage_dropped" in R.QA_COUNTERS


def test_the_replay_row_carries_the_riqualificazione_and_the_rites_block():
    state = {"strategy": {"status": "ok", "ranked": [], "riqualificazione": {"linea": None, "articolo": None, "n": None, "alternative": ["646 c.p."]}},
             "rites_block": [{"id": "abbreviato", "name": "Giudizio abbreviato", "status": "ammesso"},
                             {"id": "patteggiamento", "name": "Patteggiamento", "status": "possibile"},
                             {"id": "messa_alla_prova", "name": "Messa alla prova", "status": "precluso"}],
             "block_counters": {"strategy_riqualificazione_missing": 1, "retrieval_quesiti_standing": 1, "rulings_no_passage_dropped": 2}}
    row = R.measure("c", state, "a", 1.0)
    table = R.case_table(row)
    assert "| riqualificazione | nessuna tra 646 c.p. |" in table
    assert "| rites block | Giudizio abbreviato: ammesso; Patteggiamento: possibile; Messa alla prova: precluso |" in table
    flat = R._row_values(row)
    assert flat["strategy.riqualificazione"] == "nessuna (646 c.p.)" and flat["rites_block"].startswith("Giudizio abbreviato: ammesso")
    assert row["strategy_counters"]["strategy_riqualificazione_missing"] == 1 and row["retrieval_counters"]["retrieval_quesiti_standing"] == 1
    assert row["qa"]["rulings_no_passage_dropped"] == 2
