"""v79.18.0 -- strategy as a verified step; set tiers; set fallback (src/rag/strategy_step.py, data/strategy_step.json;
analysis_step.set_fallback / degree_grade / record_facts_split; interactive_flow.binding_pretesa «tiers»,
method_objections, strategy_state; scripts/replay_cases.py «strategy»; data/mediation_matters.json «appalto»).
Synthetic texts, names and replies only."""
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import test_v79_15_0_analysis as T  # noqa: E402
from src.rag import analysis_step as A  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import strategy_step as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("replay_cases_180", ROOT / "scripts" / "replay_cases.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

FACTS = [{"id": "S1", "text": "Anna Corsi ha firmato un preliminare con Ettore Bassi", "certainty": "accertato", "said_by": ""},
         {"id": "S2", "text": "Anna Corsi ha versato 15.000 euro a titolo di caparra", "certainty": "accertato", "said_by": ""},
         {"id": "S3", "text": "Ettore Bassi sostiene di aver venduto a un terzo perché Anna Corsi non si è presentata dal notaio",
          "certainty": "allegato", "said_by": "Ettore Bassi"}]


def _el(eid, testo, esito, comma=1, cita=(), valutato=True):
    return {"eid": eid, "testo": testo, "comma": comma, "esito": esito, "cita_fatti": list(cita), "valutato": valutato,
            "eccezione": False, "procedurale": False, "parte_che_afferma": None}


def _inp(side="attore", rulings=True):
    return {"pretesa": "domanda: «esigere il doppio della caparra» contro Ettore Bassi", "client": "Anna Corsi", "side": side,
            "charge": "1385 c.c.",
            "articles": [
                {"article": "1385 c.c.", "token": "1385cc", "rubrica": "Caparra confirmatoria", "tier": "si_verificato",
                 "in_set": True, "valutato": True,
                 "elements": [_el("A1385cc:E1", "una parte dà all'altra, a titolo di caparra, una somma", "provato", 1, ["S2"]),
                              _el("A1385cc:E2", "la parte che ha ricevuto la caparra è inadempiente", "contestato", 2, ["S3"]),
                              _el("A1385cc:E3", "l'altra recede dal contratto", "da_provare", 2)]},
                {"article": "2043 c.c.", "token": "2043cc", "rubrica": "Risarcimento per fatto illecito", "tier": "non_verificato",
                 "in_set": True, "valutato": False,
                 "elements": [_el("A2043cc:E1", "fatto doloso o colposo", "da_provare", 1, valutato=False),
                              _el("A2043cc:E2", "danno ingiusto", "da_provare", 1, valutato=False)]}],
            "facts": [dict(f) for f in FACTS], "allegations": ["S3"],
            "quesiti": ["Anna può esigere il doppio?", "Può chiedere il risarcimento?"],
            "quesiti_articles": {"Q1": ["1385 c.c."], "Q2": ["2043 c.c."]},
            "lines": [{"kind": "linea_principale", "articolo": "1385 c.c.", "eid": "A1385cc:E2", "stato": "verificata"}],
            "honesty": "",
            "rulings": [{"id": "G2", "estremi": "Cass. civ., Sez. 2, n. 100/2025", "article": "1385 c.c.",
                         "passage": "la caparra confirmatoria …", "element_id": ""}] if rulings else []}


def _linea(art, eid, tipo, rapporto="alternativa", cf=(), cg=(), ob=None, perche="perché"):
    return {"articolo": art, "eid": eid, "tipo": tipo, "rapporto": rapporto, "perche_prima": perche, "cita_fatti": list(cf),
            "cita_pronunce": list(cg), "obiezione": ob if ob is not None else
            {"testo": "La controparte obietterà.", "risposta_cita_fatti": ["S2"], "cita_pronunce": []}}


def _reply(linee, **more):
    return {"linee": linee, "passi": [], "passi_immediati": [], "contro_difesa": [], "verifiche_documentali": [], **more}


def _view(inp):
    return {"analysis": {"side": inp["side"], "linea_principale": {"articolo": "1385 c.c.", "stato": A.VERIFIED, "eid": "A1385cc:E2"}},
            "sentences": {f["id"]: f["text"] for f in inp["facts"]}, "pretesa_line": "Pretesa: domanda — «esigere il doppio»."}


# --- B. the strategy step: the prompt, the verification, forza and the ranking ------------------------------------------

def test_the_prompt_is_ids_and_verbatim_texts_only_and_its_hash_is_the_datas(monkeypatch):
    monkeypatch.setenv("ANALYSIS_STEP", "true")
    human = S.build_messages(_inp())[1].content
    assert human.startswith("PRETESA: domanda: «esigere il doppio della caparra» contro Ettore Bassi\nParte assistita: Anna Corsi (attore)")
    assert "- A1385cc: art. 1385 c.c. (Caparra confirmatoria) [si verificato]" in human
    assert "  - [A1385cc:E1] una parte dà all'altra, a titolo di caparra, una somma — provato — S2" in human
    assert "- A2043cc: art. 2043 c.c. (Risarcimento per fatto illecito) [non verificato]" in human
    assert "(esiti non valutati dall'analisi: ogni elemento da_provare)" in human
    assert "S3 [allegato; Ettore Bassi] Ettore Bassi sostiene" in human
    assert "ALLEGAZIONI DELLA CONTROPARTE (una voce di contro_difesa per ciascuna): S3" in human
    assert "- G2: Cass. civ., Sez. 2, n. 100/2025 — art. 1385 c.c. — «la caparra confirmatoria …»" in human
    assert "Q1: Anna può esigere il doppio?" in human and "- linea_principale: art. 1385 c.c. A1385cc:E2 (verificata)" in human
    assert S.prompt_hash() == __import__("hashlib").sha256(S.prompt_system().encode()).hexdigest()[:12]
    assert S.enabled() and S.FLAG == "ANALYSIS_STEP"
    assert S.request_kwargs()["max_tokens"] == 2500 and S.request_kwargs()["temperature"] == 0


def test_verification_ids_exist_the_article_is_in_the_set_the_eid_belongs_to_it_the_tipo_fits_the_esito():
    inp = _inp()
    bad = _reply([
        _linea("A1385cc", "A1385cc:E1", "fondare", "principale", ["S2"], ["G2"]),            # ok
        _linea("A1234cc", "A1234cc:E1", "fondare"),                                           # not in the set
        _linea("A1385cc", "A1385cc:E9", "contestare"),                                        # no such element
        _linea("A2043cc", "A2043cc:E1", "fondare"),                                           # fondare on da_provare
        _linea("A1385cc", "A1385cc:E1", "contestare"),                                        # contestare on provato
        _linea("A1385cc", "A1385cc:E2", "negare"),                                            # no such tipo
        _linea("A1385cc", "A1385cc:E2", "contestare", "alternativa", ["S3", "S9"], ["G9"]),   # bad ids dropped, kept
        _linea("A1385cc", "A1385cc:E2", "contestare")])                                       # a repeat
    checked = S.verify(bad, inp)
    assert [x["eid"] for x in checked["linee"]] == ["A1385cc:E1", "A1385cc:E2"]
    whys = [r["why"] for r in checked["rejected"]]
    assert whys[0].startswith("linea 2: articolo non nell'insieme: A1234cc; ammessi A1385cc, A2043cc")
    assert whys[1].startswith("linea 3: eid «A1385cc:E9» non è un elemento di art. 1385 c.c. (A1385cc:E1, A1385cc:E2, A1385cc:E3)")
    assert whys[2] == "linea 4: «fondare» su A2043cc:E1 con esito da_provare: ammesso solo su un elemento provato"
    assert whys[3] == "linea 5: «contestare» su A1385cc:E1 con esito provato: ammesso solo su un elemento contestato o da_provare"
    assert whys[4].startswith("linea 6: tipo «negare» non ammesso")
    assert whys[5] == "linea 8: ripete la linea su A1385cc:E2"
    assert checked["counters"]["strategy_line_rejected"] == 6 and checked["counters"]["strategy_bad_id"] == 2
    assert checked["linee"][1]["cita_fatti"] == ["S3"] and checked["linee"][1]["cita_pronunce"] == []
    assert checked["reasons"] == whys                       # two verified lines: no «almeno due» reason
    one = S.verify(_reply([_linea("A1385cc", "A1385cc:E1", "fondare")]), inp)
    assert one["reasons"] == ["servono almeno 2 linee verificate (ne restano 1)"]
    assert S.verify({}, inp)["reasons"][0] == "«linee» mancante o vuota"
    # the article resolved however spelt; a bare «E2» on the article
    assert S.article_of("art. 1385 c.c.", inp)["article"] == "1385 c.c." and S.article_of("1385cc", inp)["article"] == "1385 c.c."
    assert S.element_of(inp["articles"][0], "E2")["eid"] == "A1385cc:E2" and S.element_of(inp["articles"][0], "A2043cc:E1") is None


def test_forza_is_the_codes_by_side_and_item_10_ranks_a_claimants_line_by_its_elements_esito_and_comma():
    inp = _inp("attore")
    f1, f2, f3 = (S.forza("1385 c.c.", e, inp) for e in ("A1385cc:E1", "A1385cc:E2", "A1385cc:E3"))
    assert f1["testo"] == "provato 1 / contestato 1 / da provare 1" and f1["totale"] == 3 and f1["valutato"] is True
    # the eid's esito first (provato > contestato > da_provare), then minus the further conditions in the same comma
    assert f1["key"] == [3, 0, 1] and f2["key"] == [2, -1, 1] and f3["key"] == [1, -1, 1]
    assert tuple(f1["key"]) > tuple(f2["key"]) > tuple(f3["key"])
    g = S.forza("2043 c.c.", "A2043cc:E1", inp)
    assert g["key"] == [1, -1, 0] and g["valutato"] is False       # a remedy whose element is da_provare never above E1
    # the defence (v79.18.1, C1): the eid's rank (own-facts contestato > da_provare with a ruling > contestato > bare
    # da_provare > provato), then contestato + da_provare on the line's article
    d = _inp("difesa")
    assert S.forza("1385 c.c.", "A1385cc:E1", d)["key"] == [0, 2] and S.forza("2043 c.c.", "A2043cc:E1", d)["key"] == [1, 2]
    assert S.forza("1385 c.c.", "A1385cc:E3", d)["key"] == [1, 2]


def test_the_ranking_rule_reorders_by_forza_unless_the_weaker_line_cites_a_ruling():
    inp = _inp()
    weak = dict(_linea("1385 c.c.", "A1385cc:E2", "contestare"), elemento="x", esito="contestato")
    strong = dict(_linea("1385 c.c.", "A1385cc:E1", "fondare"), elemento="y", esito="provato")
    for x in (weak, strong):
        x["forza"] = S.forza(x["articolo"], x["eid"], inp)
    ranked, reordered = S.rank([weak, strong])
    assert [x["eid"] for x in ranked] == ["A1385cc:E1", "A1385cc:E2"] and reordered is True
    cited = dict(weak, cita_pronunce=["G2"])
    ranked, reordered = S.rank([cited, strong])
    assert [x["eid"] for x in ranked] == ["A1385cc:E2", "A1385cc:E1"] and reordered is False    # a G-id keeps it above
    tie = dict(strong, eid="A1385cc:E9", forza=dict(strong["forza"]))
    assert [x["eid"] for x in S.rank([strong, tie])[0]] == ["A1385cc:E1", "A1385cc:E9"]          # ties: the model's order
    # through finish: the first ranked is the principale, the model's «principale» elsewhere becomes subordinata
    checked = S.verify(_reply([_linea("A1385cc", "A1385cc:E2", "contestare", "principale", ["S3"]),
                               _linea("A1385cc", "A1385cc:E1", "fondare", "subordinata", ["S2"])]), inp)
    a = S.finish(checked, inp)
    assert a["final_order"] == ["1385 c.c. A1385cc:E1", "1385 c.c. A1385cc:E2"] and a["reordered"] is True
    assert a["model_order"] == ["1385 c.c. A1385cc:E2", "1385 c.c. A1385cc:E1"]
    assert [x["rapporto"] for x in a["linee"]] == ["principale", "subordinata"]
    assert checked["counters"]["strategy_rank_reordered"] == 1 and checked["counters"]["strategy_lines"] == 2


def test_the_result_objections_passi_contro_difesa_and_the_id_cited_entries():
    inp = _inp()
    reply = _reply([
        _linea("A1385cc", "A1385cc:E1", "fondare", "principale", ["S2"], ["G2"],
               ob={"testo": "Anna non si è presentata dal notaio.", "risposta_cita_fatti": ["S2", "S9"], "cita_pronunce": ["G2", "G7"]}),
        _linea("A1385cc", "A1385cc:E2", "contestare", "alternativa", ["S3"],
               ob={"testo": "Bassi era legittimato a vendere.", "risposta_cita_fatti": "nessuna risposta nei fatti", "cita_pronunce": []}),
        _linea("A2043cc", "A2043cc:E1", "riqualificare", "cumulativa", ["S1"], ob={"testo": "", "risposta_cita_fatti": []})],
        passi=[{"eid": "A1385cc:E2", "cosa": "Produrre la diffida", "cita_fatti": ["S1", "S2"]},
               {"eid": "A1385cc:E1", "cosa": "non è aperto", "cita_fatti": []},        # a provato element: dropped
               {"eid": "A2043cc:E1", "cosa": "altro articolo", "cita_fatti": []}],     # not the principale's: dropped
        passi_immediati=[{"testo": "Conservare la ricevuta della caparra", "cita_elementi": ["A1385cc:E1"], "cita_fatti": ["S2"]},
                         {"testo": "Senza id", "cita_elementi": [], "cita_fatti": []}],
        contro_difesa=[{"allegazione": "S3", "risposta": "La convocazione dal notaio non è provata", "cita_linea": "A1385cc:E2",
                        "cita_fatti": ["S2"]},
                       {"allegazione": "S1", "risposta": "non è un'allegazione", "cita_linea": None, "cita_fatti": []}],
        verifiche_documentali=[{"testo": "Verificare il termine per il rogito", "cita_fatti": ["S1"], "cita_elementi": []},
                               {"testo": "Nulla", "cita_fatti": ["S8"], "cita_elementi": ["A9cc:E1"]}])
    checked = S.verify(reply, inp)
    a = S.finish(checked, inp)
    c = checked["counters"]
    assert [x["eid"] for x in a["linee"]] == ["A1385cc:E1", "A1385cc:E2", "A2043cc:E1"] and a["reordered"] is False
    o1, o2 = a["linee"][0]["obiezione"], a["linee"][1]["obiezione"]
    assert o1 == {"testo": "Anna non si è presentata dal notaio.", "risposta_cita_fatti": ["S2"], "cita_pronunce": ["G2"],
                  "senza_risposta": False, "mismatch": False, "risposta_cita_elementi": []}
    assert o2["senza_risposta"] is True and a["linee"][2]["obiezione"] is None
    assert c["strategy_objections_unanswered"] == 1 and c["strategy_rulings_cited"] == 1 and a["rulings_cited"] == ["G2"]
    # the passi: the principale's open elements -- the model's for E2 (one S-id kept), the code's label for E3
    assert [(p["eid"], p["codice"], p["cita_fatti"]) for p in a["passi"]] == [("A1385cc:E2", False, ["S1"]), ("A1385cc:E3", True, [])]
    assert c["strategy_passi_added"] == 1
    # item 12: one contro_difesa per allegation (S3), S1 refused; the uncited entries dropped
    assert a["contro_difesa"] == [{"allegazione": "S3", "risposta": "La convocazione dal notaio non è provata",
                                   "cita_linea": "A1385cc:E2", "cita_fatti": ["S2"], "senza_risposta": False, "codice": False}]
    assert a["passi_immediati"] == [{"testo": "Conservare la ricevuta della caparra", "cita_fatti": ["S2"], "cita_elementi": ["A1385cc:E1"]}]
    assert a["verifiche_documentali"] == [{"testo": "Verificare il termine per il rogito", "cita_fatti": ["S1"], "cita_elementi": []}]
    assert c["strategy_entries_uncited"] == 2 and c["strategy_contro_difesa_added"] == 0
    # no entry for the allegation: the code's «senza risposta»
    b = S.finish(S.verify(_reply(reply["linee"][:2]), inp), inp)
    assert b["contro_difesa"] == [{"allegazione": "S3", "risposta": "", "cita_linea": None, "cita_fatti": [], "senza_risposta": True,
                                   "codice": True}]


def test_analyse_re_asks_once_merges_and_fails_loudly_with_no_verified_line(monkeypatch):
    monkeypatch.setenv("ANALYSIS_STEP", "true")
    inp = _inp()
    first = _reply([_linea("A2043cc", "A2043cc:E1", "fondare", "principale", ["S1"]),       # rejected
                    _linea("A1385cc", "A1385cc:E1", "fondare", "subordinata", ["S2"], ["G2"])])
    second = _reply([_linea("A1385cc", "A1385cc:E1", "fondare", "principale", ["S2"], ["G2"]),
                     _linea("A1385cc", "A1385cc:E2", "contestare", "alternativa", ["S3"])],
                    contro_difesa=[{"allegazione": "S3", "risposta": "senza risposta", "cita_linea": None, "cita_fatti": []}])
    model = T._Model(first, second)
    res = S.analyse(inp, call=model)
    assert res["status"] == S.OK and len(model.seen) == 2 and res["counters"]["strategy_calls"] == 2
    assert res["counters"]["strategy_reasked"] == 1 and res["counters"]["strategy_line_rejected"] == 1
    reask = model.seen[1]["messages"][-1].content
    assert reask.startswith("La risposta precedente è stata respinta per questi motivi:\n- linea 1: «fondare» su A2043cc:E1")
    assert "- A1385cc (art. 1385 c.c.): A1385cc:E1 (provato), A1385cc:E2 (contestato), A1385cc:E3 (da_provare)" in reask
    a = res["analysis"]
    assert a["final_order"] == ["1385 c.c. A1385cc:E1", "1385 c.c. A1385cc:E2"] and a["contro_difesa"][0]["senza_risposta"] is True
    assert res["prompt_hash"] == S.prompt_hash() and res["counters"]["strategy_lines"] == 2
    # nothing verified on either attempt: failed, no analysis, the counters say so
    res = S.analyse(inp, call=T._Model(_reply([_linea("A9cc", "A9cc:E1", "fondare")]), {}))
    assert res["status"] == S.FAILED and res["analysis"] is None and res["counters"]["strategy_failed"] == 1
    # the call raising: failed; no input: no_input, nothing called
    res = S.analyse(inp, call=T._Model(RuntimeError("down")))
    assert res["status"] == S.FAILED and res["counters"]["strategy_calls"] == 1
    assert S.analyse({"articles": [], "facts": []})["status"] == S.NO_INPUT
    # the step on a context: the registry's rulings on the set's articles (G-ids as the registry numbers them)
    ctx = {"strategy_input": {k: v for k, v in _inp(rulings=False).items() if k != "rulings"},
           "rulings_record": {"kept": [{"article": "640 c.p.", "estremi": "Cass. pen. 1/2025", "passage": "x"},
                                       {"article": "1385 c.c.", "estremi": "Cass. civ., Sez. 2, n. 100/2025", "passage": "la caparra"}]}}
    model = T._Model(second)
    out = S.run(ctx, call=model)
    assert "- G2: Cass. civ., Sez. 2, n. 100/2025 — art. 1385 c.c. — «la caparra»" in model.seen[0]["messages"][1].content
    assert "G1" not in model.seen[0]["messages"][1].content and ctx["strategy"] is out and ctx["timings"]["strategy_s"] >= 0
    assert S.run({}, call=model)["status"] == S.NO_INPUT
    monkeypatch.setenv("ANALYSIS_STEP", "false")
    assert S.run(ctx, call=model)["status"] == S.OFF


def test_the_rendering_sections_one_to_ten_from_the_verified_strategy(monkeypatch):
    monkeypatch.setenv("ANALYSIS_STEP", "true")
    inp = _inp()
    reply = _reply([
        _linea("A1385cc", "A1385cc:E2", "contestare", "principale", ["S3"],
               ob={"testo": "Bassi era legittimato a vendere.", "risposta_cita_fatti": [], "cita_pronunce": []}),
        _linea("A1385cc", "A1385cc:E1", "fondare", "subordinata", ["S2"], ["G2"], perche="la caparra è provata",
               ob={"testo": "Anna non si è presentata dal notaio.", "risposta_cita_fatti": ["S2"], "cita_pronunce": ["G2"]})],
        passi=[{"eid": "A1385cc:E2", "cosa": "Produrre la diffida", "cita_fatti": ["S1"]}],
        passi_immediati=[{"testo": "Conservare la ricevuta della caparra", "cita_elementi": ["A1385cc:E1"], "cita_fatti": ["S2"]}],
        contro_difesa=[{"allegazione": "S3", "risposta": "La convocazione dal notaio non è provata", "cita_linea": "A1385cc:E2",
                        "cita_fatti": ["S2"]}],
        verifiche_documentali=[{"testo": "Verificare il termine per il rogito", "cita_fatti": ["S1"], "cita_elementi": []}])
    res = S.analyse(inp, call=T._Model(reply))
    v = _view(inp)
    # §1: the ranked lines (the fondare line first: reordered), the tesi the analysis step's templates by side and esito
    sintesi = S.summary_text(v, res)
    assert sintesi == ("Pretesa: domanda — «esigere il doppio».\n\nLinea principale: art. 1385 c.c. — «una parte dà all'altra, a titolo "
                       "di caparra, una somma» risulta provato: «Anna Corsi ha versato 15.000 euro a titolo di caparra».\n\n"
                       "Linea subordinata: art. 1385 c.c. — «la parte che ha ricevuto la caparra è inadempiente» è da provare.")
    # §6 in item 12's order; the forza shown as the data prints it; the reordering said
    sol = S.solutions_text(res, v, inp["quesiti"], inp["quesiti_articles"])
    titles = ["**Passi immediati (prova e termini)**", "**Linee per forza**", "**Risposta ai quesiti**", "**Contro-difesa**",
              "**Verifiche documentali**"]
    assert [t for t in titles if t in sol] == titles and sorted(sol.index(t) for t in titles) == [sol.index(t) for t in titles]
    assert "- Conservare la ricevuta della caparra: «Anna Corsi ha versato 15.000 euro a titolo di caparra» [A1385cc:E1]." in sol
    assert ("| Linea 1 | art. 1385 c.c. | una parte dà all'altra, a titolo di caparra, una somma | fondare | principale | "
            "la caparra è provata | provato 1 / contestato 1 / da provare 1 | «Anna Corsi ha versato 15.000 euro a titolo di caparra» | (G2) |") in sol
    assert "| Linea 2 | art. 1385 c.c. | la parte che ha ricevuto la caparra è inadempiente | contestare | subordinata |" in sol
    assert "Ordine ricalcolato dal codice per forza (l'ordine del modello era: 1385 c.c. A1385cc:E2, 1385 c.c. A1385cc:E1)." in sol
    assert "- Q1 «Anna può esigere il doppio?»: art. 1385 c.c. (linea 1), art. 1385 c.c. (linea 2)" in sol
    assert "- Q2 «Può chiedere il risarcimento?»: nessuna linea verificata sul quesito" in sol
    assert S.quesiti_map(inp, {"Può chiedere il risarcimento?": ["2043 c.c."], "altro": ["1 c.c."]}) == {
        "Q1": ["1385 c.c."], "Q2": ["2043 c.c."]}                 # the fit's tags first, then the quesito's own search
    assert S.quesiti_map({"quesiti": ["x"], "quesiti_articles": {}}, {"x": ["9 c.c."]}) == {"Q1": ["9 c.c."]}
    assert S.passi_elements(res) == ["la parte che ha ricevuto la caparra è inadempiente", "l'altra recede dal contratto"]
    assert ("- Allegazione S3 «Ettore Bassi sostiene di aver venduto a un terzo perché Anna Corsi non si è presentata dal notaio»: "
            "La convocazione dal notaio non è provata: «Anna Corsi ha versato 15.000 euro a titolo di caparra» (linea: art. 1385 c.c., "
            "«la parte che ha ricevuto la caparra è inadempiente»).") in sol
    assert "- Verificare il termine per il rogito: «Anna Corsi ha firmato un preliminare con Ettore Bassi»." in sol
    # §7: the obiezioni with their verified answers -- the cited facts as written, the ruling by its label, or the data's line
    assert S.objection_items(res, v) == [
        {"obiezione": "Anna non si è presentata dal notaio", "risposta": "«Anna Corsi ha versato 15.000 euro a titolo di caparra» (G2).",
         "fonte": "strategia", "linea": "A1385cc:E1"},
        {"obiezione": "Bassi era legittimato a vendere", "risposta": "Senza risposta nei fatti narrati.", "fonte": "strategia",
         "linea": "A1385cc:E2"}]
    # §9: the risk from the principale by the data's rules (v79.18.3: a claimant's provato element with a G-cite -> favorevole)
    assert S.risk_word(res) == "favorevole" and res["analysis"]["risk"]["capped"] is False
    conclusions = A.conclusions_text({**v, "analysis": {**v["analysis"], "elementi": {"1385 c.c.": [
        {"elemento": "x", "esito": "provato"}]}, "linea_principale": {"articolo": "1385 c.c.", "stato": A.VERIFIED, "tesi": "t"}},
        "client": "Anna Corsi"}, risk=S.risk_word(res))
    assert "**Rischio per Anna Corsi**: favorevole." in conclusions
    assert A.conclusions_text({**v, "client": "Anna Corsi"}, risk="incerto", lines=S.summary_lines(res, v)).splitlines()[0] == \
        "**Linea principale**: art. 1385 c.c. — «una parte dà all'altra, a titolo di caparra, una somma» risulta provato: " \
        "«Anna Corsi ha versato 15.000 euro a titolo di caparra»."
    # §10: the passi -- the model's with its fact, the code's label for the open element it gave none for
    assert S.passi_points(res, v) == [
        "Passo: «la parte che ha ricevuto la caparra è inadempiente» (art. 1385 c.c.) — Produrre la diffida («Anna Corsi ha firmato "
        "un preliminare con Ettore Bassi»).",
        "«l'altra recede dal contratto» (art. 1385 c.c.): da provare, nessun passo indicato dal modello."]
    # the replay summary
    summary = S.replay_summary(res)
    assert summary["ranked"][0] == {"articolo": "1385 c.c.", "eid": "A1385cc:E1", "tipo": "fondare", "rapporto": "principale",
                                    "forza": "provato 1 / contestato 1 / da provare 1", "key": [3, 0, 1], "pronunce": ["G2"]}
    assert summary["reordered"] is True and summary["risk"] == "favorevole" and summary["prompt_hash"] == S.prompt_hash()
    assert summary["risk_why"] == "elemento provato, con pronuncia" and summary["risk_capped"] is False
    assert summary["objections"] == [{"linea": "A1385cc:E1", "answered": True}, {"linea": "A1385cc:E2", "answered": False}]


def test_the_risk_thresholds_by_side_from_the_data():
    # v79.18.3 (item 1 of 79.18.2): the rules of data «risk», by side
    def res(side, esiti, cg=(), own=False, tier="si_verificato"):
        els = [_el(f"A1385cc:E{i}", f"e{i}", e, 1, ["S2"] if (own and i == 1) else []) for i, e in enumerate(esiti, 1)]
        inp = {"side": side, "articles": [{"article": "1385 c.c.", "token": "1385cc", "tier": tier, "in_set": True,
                                           "valutato": True, "elements": els}],
               "facts": [{"id": "S2", "text": "x", "own": own}]}
        linea = dict(_linea("1385 c.c.", "A1385cc:E1", "fondare", cg=cg), forza=S.forza("1385 c.c.", "A1385cc:E1", inp,
                                                                                      {"cita_pronunce": list(cg), "cita_fatti": []}))
        return {"status": S.OK, "analysis": {"linee": [linea], "side": side, "set_non_verificato": tier == "non_verificato"}}
    # claimant: provato + a second provato -> favorevole; provato + a G-cite -> favorevole; provato alone -> incerto
    assert S.risk_word(res("attore", ["provato", "provato"])) == "favorevole"
    assert S.risk_word(res("attore", ["provato", "da_provare"], cg=["G2"])) == "favorevole"
    assert S.risk_word(res("attore", ["provato", "da_provare"])) == "incerto"
    assert S.risk_word(res("attore", ["contestato", "provato"])) == "incerto"
    assert S.risk_word(res("attore", ["da_provare", "contestato"])) == "sfavorevole"
    assert S.risk_word(res("attore", ["escluso", "provato"])) == "sfavorevole"
    assert S.risk_word(res("neutro", ["provato", "provato"])) == "favorevole"
    # defence: contestato by own facts + a G-cite -> favorevole; either alone -> incerto; every element provato -> sfavorevole
    assert S.risk_word(res("difesa", ["contestato", "provato"], cg=["G2"], own=True)) == "favorevole"
    assert S.risk_word(res("difesa", ["contestato", "provato"], own=True)) == "incerto"
    assert S.risk_word(res("difesa", ["contestato", "provato"], cg=["G2"])) == "incerto"
    assert S.risk_word(res("difesa", ["da_provare", "provato"], cg=["G2"])) == "incerto"
    assert S.risk_word(res("difesa", ["da_provare", "provato"])) == "incerto"
    assert S.risk_word(res("difesa", ["provato", "provato"])) == "sfavorevole"
    assert S.risk_word(res("difesa", ["provato", "escluso"])) == "incerto"            # not all provato, no own contest
    # the cap: an unverified-only set, or a one-element article, never above incerto
    assert S.risk_word(res("attore", ["provato", "provato"], tier="non_verificato")) == "incerto"
    assert S.risk_of(res("attore", ["provato", "provato"], tier="non_verificato")["analysis"])["capped"] is True
    assert S.risk_word(res("attore", ["provato"], cg=["G2"])) == "incerto"
    assert S.risk_of(res("attore", ["provato"], cg=["G2"])["analysis"]) == {
        "word": "incerto", "capped": True, "why": "elemento provato, con pronuncia; tetto incerto: articolo con un solo elemento"}
    assert S.risk_of(res("attore", ["da_provare"])["analysis"])["capped"] is False      # the cap lowers favorevole only
    assert S.risk_word(None) == "" and S.risk_word({"status": "failed"}) == "" and S.risk_of(None) == {}
    assert set(S.config()["risk"]) == {"attore", "neutro", "difesa", "cap"}


# --- B. the input from the state, the flow's hook, the counters ----------------------------------------------------------

def _state_for_input():
    verified = {"status": A.OK, "pretesa_line": "Pretesa: domanda — «esigere il doppio».", "client": "Anna Corsi",
                "analysis": {"side": "attore", "frame_charge": "1385 c.c.",
                             "linea_principale": {"articolo": "1385 c.c.", "eid": "A1385cc:E2", "stato": A.VERIFIED},
                             "linea_subordinata": {"articolo": "2946 c.c.", "eid": "A2946cc:E1", "stato": A.VERIFIED},
                             "elementi": {"1385 c.c.": [{"eid": "A1385cc:E1", "esito": "provato", "cita_fatti": ["S2"]},
                                                        {"eid": "A1385cc:E2", "esito": "contestato", "cita_fatti": ["S3"],
                                                         "parte_che_afferma": "Ettore Bassi"}]}}}
    norms = [{"article": "1385 c.c.", "rubrica": "Caparra confirmatoria", "remedy": False,
              "elements": [{"eid": "A1385cc:E1", "testo": "a titolo di caparra", "comma": 1},
                           {"eid": "A1385cc:E2", "testo": "ricevuta", "comma": 2}]},
             {"article": "2043 c.c.", "rubrica": "Fatto illecito", "remedy": False,
              "elements": [{"eid": "A2043cc:E1", "testo": "fatto doloso", "comma": 1}]},
             {"article": "1453 c.c.", "rubrica": "Risoluzione", "remedy": True, "elements": [{"eid": "A1453cc:E1", "testo": "x"}]},
             {"article": "2946 c.c.", "rubrica": "Prescrizione", "remedy": False,
              "elements": [{"eid": "A2946cc:E1", "testo": "decorso di dieci anni", "comma": 1, "schema_kind": "termine"}]}]
    state = {"analysis_norms": norms,
             "binding_pretesa": {"principale": ["1385 c.c.", "2043 c.c."], "tiers": {"1385 c.c.": "si_verificato", "2043 c.c.": "non_verificato"},
                                 "quesiti": {"Q1": ["1385 c.c."]}, "pretesa_article": "1385 c.c."},
             "analysis_maps": {"sentences": {f["id"]: {"text": f["text"], "certainty": f["certainty"], "said_by": f["said_by"]}
                                             for f in FACTS}},
             "case_frame": {"frame": {"parti": [{"nome": "Anna Corsi", "lato": "cliente"}, {"nome": "Ettore Bassi", "lato": "controparte"}],
                                      "cliente": {"nome": "Anna Corsi"}, "quesiti": ["Anna può esigere il doppio?"]},
                            "verified": ["parti", "cliente", "quesiti"]},
             "client": {"name": "Anna Corsi"}}
    return state, verified


def test_the_input_from_the_state_the_set_with_its_tiers_the_lines_articles_and_the_allegations(monkeypatch):
    monkeypatch.setenv("ANALYSIS_STEP", "true")
    monkeypatch.setenv("CASE_FRAME", "true")
    state, verified = _state_for_input()
    inp = S.state_input(state, verified)
    assert [(a["article"], a["tier"], a["in_set"], a["valutato"]) for a in inp["articles"]] == [
        ("1385 c.c.", "si_verificato", True, True), ("2043 c.c.", "non_verificato", True, False),
        ("2946 c.c.", "linea", False, False)]                 # the verified subordinate's article: in, not of the set
    e1, e2 = inp["articles"][0]["elements"]
    assert (e1["esito"], e1["cita_fatti"], e1["valutato"], e1["comma"]) == ("provato", ["S2"], True, 1)
    assert (e2["esito"], e2["parte_che_afferma"], e2["comma"]) == ("contestato", "Ettore Bassi", 2)
    assert inp["articles"][1]["elements"][0]["esito"] == "da_provare" and inp["articles"][2]["elements"][0]["procedurale"] is True
    assert inp["allegations"] == ["S3"] and [f["id"] for f in inp["facts"]] == ["S1", "S2", "S3"]
    assert inp["pretesa"] == "Pretesa: domanda — «esigere il doppio»." and inp["side"] == "attore" and inp["client"] == "Anna Corsi"
    assert inp["quesiti"] == ["Anna può esigere il doppio?"] and inp["quesiti_articles"] == {"Q1": ["1385 c.c."]}
    assert inp["lines"][0]["eid"] == "A1385cc:E2" and inp["charge"] == "1385 c.c." and inp["rulings"] == []
    # no set article with elements: no input; the flag off: none
    assert S.state_input({**state, "analysis_norms": [], "binding_pretesa": {}}, verified) is None
    monkeypatch.setenv("ANALYSIS_STEP", "false")
    assert S.state_input(state, verified) is None


def test_method_objections_runs_the_strategy_step_instead_of_the_objections_call_under_the_flag(monkeypatch):
    monkeypatch.setenv("ANALYSIS_STEP", "true")
    from src.rag import lawyer_method as M
    monkeypatch.setattr(M, "objections_plan", lambda ctx, mplan: {**mplan, "obiezioni": [{"obiezione": "old", "risposta": "x"}]})
    reply = _reply([_linea("A1385cc", "A1385cc:E1", "fondare", "principale", ["S2"]),
                    _linea("A1385cc", "A1385cc:E2", "contestare", "alternativa", ["S3"])])
    monkeypatch.setattr(S, "_default_call", T._Model(reply))
    ctx = {"strategy_input": {k: v for k, v in _inp(rulings=False).items() if k != "rulings"}, "rulings_record": {"kept": []}}
    mplan = flow.method_objections(ctx, {"sintesi": "s"})
    assert mplan == {"sintesi": "s"} and ctx["strategy"]["status"] == S.OK          # the objections call retired
    assert [x["eid"] for x in ctx["strategy"]["analysis"]["linee"]] == ["A1385cc:E1", "A1385cc:E2"]
    state = {"block_counters": {}}
    flow.strategy_state(state, [ctx, {"strategy": {"counters": {"strategy_calls": 1, "strategy_lines": 3, "strategy_seconds": 1.5}}}])
    c = state["block_counters"]
    assert c["strategy_calls"] == 2 and c["strategy_lines"] == 5 and c["strategy_rank_reordered"] == 0
    assert c["strategy_seconds"] == round(ctx["strategy"]["seconds"] + 1.5, 3) == state["timings"]["strategy_s"]
    assert state["strategy"]["ranked"][0]["eid"] == "A1385cc:E1" and state["strategy"]["status"] == "ok"
    # no strategy input: the old call as before
    assert flow.method_objections({}, {"sintesi": "s"})["obiezioni"] == [{"obiezione": "old", "risposta": "x"}]
    monkeypatch.setenv("ANALYSIS_STEP", "false")
    assert flow.method_objections(dict(ctx), {"sintesi": "s"})["obiezioni"][0]["obiezione"] == "old"


def test_the_replay_row_carries_the_strategy_line_and_its_counters():
    state = {"strategy": {"status": "ok", "ranked": [{"articolo": "1385 c.c.", "eid": "A1385cc:E1", "tipo": "fondare",
                                                        "rapporto": "principale", "forza": "provato 1 / contestato 1 / da provare 1",
                                                        "key": [3, 0, 1], "pronunce": ["G2"]}],
                          "reordered": True, "model_order": ["1385 c.c. A1385cc:E2", "1385 c.c. A1385cc:E1"],
                          "final_order": ["1385 c.c. A1385cc:E1", "1385 c.c. A1385cc:E2"], "risk": "incerto",
                          "objections": [{"linea": "A1385cc:E1", "answered": True}], "contro_difesa": [{"sid": "S3", "answered": False}],
                          "prompt_hash": "abc", "error": ""},
             "block_counters": {"strategy_calls": 1, "strategy_lines": 2, "strategy_rank_reordered": 1, "strategy_seconds": 3.2,
                                "binding_set_unverified": 1, "analysis_line_fallback_set": 0, "analysis_degree_downgraded": 1},
             "timings": {"strategy_s": 3.2, "applicability_s": 2.0}, "analysis_step": {"seconds": 4.0},
             "single_shot_timings": {"answer_s": 20.0, "total_s": 30.0}}
    row = R.measure("c", state, "a", 1.0)
    assert row["strategy"]["ranked"][0]["tipo"] == "fondare" and row["strategy_counters"]["strategy_rank_reordered"] == 1
    assert row["frame_counters"]["binding_set_unverified"] == 1 and row["analysis_counters"]["analysis_degree_downgraded"] == 1
    assert row["stages"]["strategy"] == 3.2 and row["stages"]["write"] == 10.8          # answer less fit, analysis, strategy
    table = R.case_table(row)
    assert ("| strategy | ok: 1. art. 1385 c.c. A1385cc:E1 fondare/principale [provato 1 / contestato 1 / da provare 1] cita G2; "
            "reordered 1 (model order 1385 c.c. A1385cc:E2, 1385 c.c. A1385cc:E1); risk incerto; objections answered 1/1; "
            "contro-difesa answered 0/1 |") in table
    assert "| strategy counters | strategy_calls 1, strategy_lines 2, strategy_rank_reordered 1, strategy_seconds 3.2 |" in table
    flat = R._row_values(row)
    assert flat["strategy.reordered"] is True and flat["strategy.risk"] == "incerto"
    assert flat["strategy.ranked"] == "1385 c.c. A1385cc:E1 fondare [provato 1 / contestato 1 / da provare 1]"
    assert "strategy" in R.STAGES and all(k in R.STRATEGY_COUNTERS for k in S.COUNTERS)
    assert R.measure("c", {}, "a", 1.0)["strategy"] == {}


# --- A. the set tiers, the set fallback, the degree words, the client's own fact -------------------------------------------

def test_the_set_fallback_leads_with_the_best_tier_article_every_element_da_provare_never_da_determinare():
    import test_v79_16_1 as X
    norms = X._fitted(list(T.PENAL_NORMS), {"640 c.p.": ("si", False), "641 c.p.": ("si", True)})
    binding = {"enforced": True, "principale": ["641 c.p.", "640 c.p."],
               "tiers": {"641 c.p.": "si_verificato", "640 c.p.": "non_verificato"}}
    inp = A.make_input(X.NO_CHARGE, norms, T._sents(T.PENAL_SENTENCES, sistema=(9,)), T.PENAL_FRAME["quesiti"],
                       client="Paolo Sarti", client_roles=["imputato"], binding=binding)
    assert inp["binding"]["tiers"] == binding["tiers"] and A.governs_sets(inp)["tiers"] == binding["tiers"]
    # the model's line fails twice (a bad eid both times): the set's verified «si» article leads, everything open
    bad = {"linea_principale": {"articolo": "640 c.p.", "elemento_in_gioco": "A640cp:E12", "tesi": "t", "cita_fatti": ["S1"],
                                "cita_norma": []}, "linea_subordinata": None, "elementi": [], "affermazioni_di_diritto": []}
    res = A.analyse(inp, call=T._Model(bad, bad))
    a, c = res["analysis"], res["counters"]
    assert c["analysis_reasked"] == 1 and c["analysis_line_rejected"] == 2 and c["analysis_bad_eid"] == 2
    assert a["linea_principale"]["articolo"] == "641 c.p." and a["linea_principale"]["stato"] == A.FALLBACK_SET
    assert a["linea_principale"]["tesi"] == "Insolvenza fraudolenta: elementi da provare dall'accusa"
    assert all(e["esito"] == A.DA_PROVARE and e["origine"] == "insieme" for e in a["elementi"]["641 c.p."])
    assert c["analysis_line_fallback_set"] == 1 and c["analysis_line_undetermined"] == 0 and a["honesty"] == ""
    assert a["fallback_skipped_reason"] == A.SKIP_SET_LINE == c[A.SKIPPED_REASON]
    v = A.view(res, inp)
    assert A.summary_text(v) == "Linea principale: art. 641 c.p. — Insolvenza fraudolenta: elementi da provare dall'accusa."
    assert "| Linea principale | art. 641 c.p. | — | Insolvenza fraudolenta: elementi da provare dall'accusa | — |" in A.solutions_text(v)
    # the set fallback alone: the first allowed article of line_tier's order (the verified «si» before the unverified)
    assert A.set_fallback(inp)["articolo"] == "641 c.p." and A.set_fallback(inp)["tier"] == "fit si verificato"
    # an empty set: the honesty line, as before
    empty = A.make_input(X.NO_CHARGE, norms, T._sents(T.PENAL_SENTENCES, sistema=(9,)), T.PENAL_FRAME["quesiti"],
                         client="Paolo Sarti", client_roles=["imputato"], binding={"enforced": True, "principale": []})
    res = A.analyse(empty, call=T._Model(bad, bad))
    assert res["analysis"]["honesty"] == A.HONESTY_LINE and res["counters"]["analysis_line_fallback_set"] == 0
    assert "analysis_line_fallback_set" in A.COUNTERS and "analysis_line_fallback_set" in R.ANALYSIS_COUNTERS


def test_a_qualified_term_in_the_element_needs_the_same_or_a_stronger_degree_in_the_cited_fact():
    assert A.degree_grade("l'inadempimento è del tutto imputabile") == 2 and A.degree_grade("un danno grave") == 1
    assert A.degree_grade("un danno") == 0 and A.degree_grade("in modo apprezzabile") == 1 and A.degree_grade("gravissimo") == 2
    assert A.degree_words_in("danno grave e irreparabile") == ["grave", "irreparabile"]
    frame = {k: v for k, v in T.CIVIL_FRAME.items() if k != "addebito_o_domanda"}
    norm = T._norm("1455 c.c.", "1. Il contratto non si può risolvere se l'inadempimento ha scarsa importanza.",
                   "Importanza dell'inadempimento", ["l'inadempimento della consegna è del tutto imputabile al venditore"], "V")
    sents = T._sents(["Lucia Ferri lamenta che l'inadempimento della consegna è imputabile a Marco Vento.",
                      "L'inadempimento della consegna è del tutto imputabile a Marco Vento, che non ha spedito nulla."])
    inp = A.make_input(frame, [norm], sents, [], client="Lucia Ferri", client_roles=["attore"])
    m = A._maps(inp)
    element = inp["norms"][0]["elements"][0]
    counters = A.empty_counters()
    got = A._verify_element({"esito": "provato", "cita_fatti": ["S1"]}, element, inp, m, counters)
    assert got["esito"] == A.DA_PROVARE and counters["analysis_degree_downgraded"] == 1
    assert got["declassato_grado"]["grado"] == 2 and got["declassato_grado"]["parole"] == ["del tutto"]
    got = A._verify_element({"esito": "provato", "cita_fatti": ["S2"]}, element, inp, m, counters)
    assert got["esito"] == A.PROVATO and counters["analysis_degree_downgraded"] == 1 and "declassato_grado" not in got
    assert A.degree_log({"elementi": {"1455 c.c.": [{"eid": "A1455cc:E1", "declassato_grado": {"grado": 2}}]}}) == [
        {"articolo": "1455 c.c.", "eid": "A1455cc:E1", "grado": 2}]
    assert "analysis_degree_downgraded" in A.COUNTERS and "analysis_degree_downgraded" in R.ANALYSIS_COUNTERS


def test_a_clients_own_stated_fact_is_certo_never_an_allegation_and_never_the_party_che_afferma():
    v = {"client": "Marta Serra", "analysis": {"linea_principale": {"articolo": None}, "elementi": {}}, "sentences": {}}
    facts = [{"text": "Marta Serra ha pagato 3.000 euro", "stato": "accertato"},
             {"text": "Secondo Marta Serra, Luca Prandi non ha consegnato il motorino", "stato": "allegato", "chi": "Marta Serra"},
             {"text": "Luca Prandi sostiene di aver spedito il motorino", "stato": "allegato", "chi": "Luca Prandi"},
             {"text": "Il corriere avrebbe perso il pacco", "stato": "allegato"}]
    split = A.record_facts_split(facts, v)
    assert split["certi"] == ["Marta Serra ha pagato 3.000 euro", "Secondo Marta Serra, Luca Prandi non ha consegnato il motorino"]
    assert [x["how"] for x in split["ipotizzabili"]] == ["allegazione di Luca Prandi", "allegazione"]
    assert split["cliente_certi"] == 1
    assert A.client_said("Serra", v) and A.client_said("la cliente", v) and not A.client_said("Luca Prandi", v)
    # the contestato's party «che afferma»: never the client
    inp = A.make_input(T.CIVIL_FRAME, list(T.CIVIL_NORMS), T._sents(T.CIVIL_SENTENCES), [], client="Lucia Ferri",
                       client_roles=["attore"])
    for s in inp["sentences"]:
        if s["id"] == "S5":
            s["said_by"], s["fact"], s["certainty"] = "Lucia Ferri", "F5", "allegato"
    m = A._maps(inp)
    element = inp["norms"][0]["elements"][1]
    got = A._verify_element({"esito": "contestato", "cita_fatti": ["S5"]}, element, inp, m, A.empty_counters())
    assert got["esito"] == A.CONTESTATO and got["parte_che_afferma"] is None


def test_the_mediation_matters_data_lists_the_appalto_to_confirm_and_the_facts_reading_uses_the_data():
    from src.rag import fact_signals, render_computed as RC
    matters = RC.matters_list()["mediazione"]["matters"]
    assert "appalto" in matters and "contratto d'opera" in matters and matters["appalto"].get("da_confermare")
    assert "appalto" not in RC.matters_list()["mediazione"]["facoltativa"]["contracts"]
    names = [x[0] for x in fact_signals.mediation_subjects()]
    assert names == list(matters) and dict((x[0], x[1]) for x in fact_signals.mediation_subjects())["appalto"] == "appalto"
    assert fact_signals.mediation({"contracts": [{"type": "appalto per il rifacimento del tetto"}]}, [])["subject"] == "appalto"
    # with the article's text in the corpus: the matter only when the text names it
    text = "La mediazione è condizione di procedibilità in materia di condominio, diritti reali, divisione, locazione."
    assert not fact_signals.mediation({"contracts": [{"type": "appalto per il rifacimento del tetto"}]}, [], text)["required"]
    assert fact_signals.mediation({"contracts": [{"type": "locazione di un negozio"}]}, [], text)["subject"] == "locazione"
    assert fact_signals.mediation({"contracts": [{"type": "contratto preliminare di compravendita"}]}, [])["optional"] is True
