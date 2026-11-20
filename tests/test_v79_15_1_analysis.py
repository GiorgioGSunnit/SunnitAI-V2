"""v79.15.1 build A -- the analysis step by element ids, the re-ask's allowed list and the rubrica fallback, the
claimant's constitutive line and §7 from the verified analysis (src/rag/analysis_step.py, data/analysis_step.json,
lawyer_method.objections_section). The model is faked at its source; every other model path and every outbound socket
refused (tests/test_v79_15_0_analysis.py's sealing, re-exported). Synthetic facts and invented names only; the article
texts are statutory (640, 641, 62-bis c.p.; 1385, 1453, 1667, 2043 c.c.).

1. elements by id: the prompt numbers each article's schema elements «A<token>:E<n>»; elemento_in_gioco and
   elementi[].elemento are ids, verified by existence only (analysis_bad_eid); the text printed is the schema's
2. the re-ask lists the allowed articles (the fitted norms minus the remedy set, the frame's charge kept) with their
   E-ids; after a failed re-ask the frame's charge article first (analysis_frame_charge_missing), else the frame's named
   charge / claim matched exactly to a fitted rubrica (analysis_line_fallback_rubrica), else «da determinare»
3. a claimant's line is a constitutive element (the schema's own kind, else the keyword list); one re-ask with the
   constitutive E-ids; §7: one objection per non-client allegation behind a «contestato» element, a claimant's raised
   exception elements, the «no objections» line only when none exists
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import test_v79_15_0_analysis as T  # noqa: E402  (its autouse sealing fixture is re-exported below)
from src.rag import analysis_step as A  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402

_sealed = T._sealed
_Model, _reply, _line, _el, _sents = T._Model, T._reply, T._line, T._el, T._sents


# --- the synthetic appalto case (a claimant: the committente) --------------------------------------------------------

ART_1667 = ("L'appaltatore è tenuto alla garanzia per le difformità e i vizi dell'opera. La garanzia non è dovuta se "
            "il committente ha accettato l'opera e le difformità o i vizi erano da lui conosciuti o erano "
            "riconoscibili, purché, in questo caso, non siano stati in mala fede taciuti dall'appaltatore.\n"
            "Il committente deve, a pena di decadenza, denunziare all'appaltatore le difformità o i vizi entro "
            "sessanta giorni dalla scoperta. La denunzia non è necessaria se l'appaltatore ha riconosciuto le "
            "difformità o i vizi o se li ha occultati.")
E1667 = ["l'appaltatore è tenuto alla garanzia per le difformità e i vizi dell'opera",
         "il committente ha accettato l'opera",
         "le difformità o i vizi erano da lui conosciuti o erano riconoscibili",
         "denunziare all'appaltatore le difformità o i vizi entro sessanta giorni dalla scoperta"]
T.ELEMENTS["1667 c.c."] = E1667
APPALTO_SENTENCES = [
    "Marta Galli ha affidato alla Edilnova S.r.l. il rifacimento del bagno con un contratto di appalto.",      # S1
    "Dopo la consegna Marta Galli ha scoperto infiltrazioni d'acqua sotto il pavimento.",                     # S2
    "La Edilnova S.r.l. sostiene che Marta Galli ha accettato l'opera senza riserve.",                          # S3
    "Marta Galli ha denunziato i vizi all'impresa con una raccomandata dieci giorni dopo la scoperta.",        # S4
]
APPALTO_FRAME = {
    "parti": [{"nome": "Marta Galli", "ruolo_nel_fatto": "committente", "lato": "cliente"},
              {"nome": "Edilnova S.r.l.", "ruolo_nel_fatto": "appaltatrice", "lato": "controparte"}],
    "cliente": {"nome": "Marta Galli", "perche": "«Assisto Marta Galli»"},
    "ruolo": {"valore": "attore", "intake": "attore", "perche": "«chiede la garanzia»"},
    "rito": {"valore": "civile", "intake": "civile", "perche": "«contratto di appalto»"},
    "addebito_o_domanda": {"articolo": "1667 c.c.", "descrizione": "garanzia per le difformità e i vizi",
                           "frase": "«garanzia»"},
    "quesiti": ["Marta Galli può ottenere la garanzia?"]}
APPALTO_NORMS = [T._norm("1667 c.c.", ART_1667, "Difformità e vizi dell'opera", E1667, "R"),
                 T._norm("1453 c.c.", T.ART_1453, "Risolubilità del contratto per inadempimento", T.E1453, "T")]


def _appalto(frame=APPALTO_FRAME, norms=APPALTO_NORMS):
    return A.make_input(frame, norms, _sents(APPALTO_SENTENCES), frame.get("quesiti") or [], client="Marta Galli",
                        client_roles=["attore"])


GOOD_APPALTO = _reply(
    _line("1667 c.c.", E1667[0], "Edilnova S.r.l. deve la garanzia per le infiltrazioni.", ["S2"], ["A1667cc.1"]),
    elementi=[_el("1667 c.c.", E1667[0], "provato", ["S2"]),
              _el("1667 c.c.", E1667[1], "provato", ["S3"]),          # the company's assertion only: contestato
              _el("1667 c.c.", E1667[2], "da_provare"),
              _el("1667 c.c.", E1667[3], "provato", ["S4"])])          # an exception element with a verdict


# --- 1. elements by id ------------------------------------------------------------------------------------------------

def test_the_prompt_numbers_every_articles_elements_by_id():
    inp = T._penal()
    human = A.build_messages(inp)[1].content
    assert "- [A640cp:E1] con artifizi o raggiri\n- [A640cp:E2] inducendo taluno in errore\n" in human
    assert "- [A641cp:E1] dissimulando il proprio stato d'insolvenza" in human
    assert A.element_id("62-bis c.p.", 2) == "A62biscp:E2" and A.element_id("1385 c.c.", 1) == "A1385cc:E1"
    assert [e["eid"] for e in inp["norms"][0]["elements"]] == ["A640cp:E1", "A640cp:E2", "A640cp:E3"]
    system = A.prompt_system()
    assert "A<articolo>:E<n>" in system and "copiato esattamente" not in system
    assert A.prompt_hash() == "472423ea32d3"
    assert "analysis_bad_eid" in A.COUNTERS and "analysis_element_not_verbatim" not in A.COUNTERS


def test_the_printed_element_is_the_schemas_never_the_models():
    model = _Model(_reply(dict(_line("640 c.p.", "A640cp:E2", "Tesi.", ["S5"], ["A640cp.1"]),
                               elemento_testo="un testo inventato"),
                          elementi=[{"articolo": "640 c.p.", "elemento": "A640cp:E1", "testo": "riscritto",
                                     "esito": "escluso", "cita_fatti": ["S5"]}]))
    res = A.analyse(T._penal(), call=model)
    an = res["analysis"]
    assert an["linea_principale"]["elemento_in_gioco"] == T.E640[1]
    first = an["elementi"]["640 c.p."][0]
    assert first["elemento"] == T.E640[0] and first["esito"] == "escluso"
    v = A.view(res, T._penal())
    assert "un testo inventato" not in A.summary_text(v) and "riscritto" not in A.norms_tables(v)


def test_an_unknown_e_id_of_a_line_voids_the_line_and_counts():
    model = _Model(_reply(_line("640 c.p.", "A640cp:E7", "Tesi.", ["S5"], ["A640cp.1"])), T.GOOD_PENAL)
    res = A.analyse(T._penal(), call=model)
    assert res["counters"]["analysis_bad_eid"] == 1 and res["counters"]["analysis_line_rejected"] == 1
    assert res["analysis"]["linea_principale"]["eid"] == "A640cp:E1"


# --- 2. the re-ask's allowed list, the fallbacks ----------------------------------------------------------------------

def test_the_re_ask_lists_the_allowed_articles_with_their_e_ids():
    remedy = _reply(_line("62-bis c.p.", "A62biscp:E1", "Attenuanti.", ["S5"]))
    model = _Model(remedy, T.GOOD_PENAL)
    A.analyse(T._penal(), call=model)
    reask = model.seen[1]["messages"][3].content
    assert "Articoli ammessi come linea (A<articolo>) con gli id dei loro elementi:" in reask
    assert "- A640cp (art. 640 c.p.): A640cp:E1, A640cp:E2, A640cp:E3" in reask
    assert "- A641cp (art. 641 c.p.): A641cp:E1, A641cp:E2" in reask
    assert "A62biscp" not in reask.split("con gli id dei loro elementi:")[1]          # a remedy: never listed
    assert A.config()["reask_claimant"] not in reask                                    # an imputato: no claimant line
    # the frame's own claim is kept even when it is in the remedy set; the other remedies are not listed
    claim = dict(T.CIVIL_FRAME, addebito_o_domanda={"articolo": "1453 c.c.", "descrizione": "risoluzione",
                                                    "frase": "«risoluzione»"})
    allowed = A.allowed_text(T._civil(frame=claim))
    assert "- A1453cc (art. 1453 c.c.): A1453cc:E1" in allowed and "A2043cc" not in allowed
    assert "- A1385cc (art. 1385 c.c.): A1385cc:E1, A1385cc:E2, A1385cc:E3" in allowed


def _no_charge(frame, **extra):
    out = {k: v for k, v in frame.items() if k != "addebito_o_domanda"}
    out.update(extra)
    return out


def test_after_a_failed_re_ask_the_frames_named_charge_finds_its_article_by_rubrica():
    frame = _no_charge(T.PENAL_FRAME, addebito_o_domanda={"articolo": None, "descrizione": "truffa",
                                                          "frase": "«procede per truffa»"})
    inp = T._penal(frame=frame)
    assert inp["charge"] == "" and inp["issues"] == ["truffa"]
    text_not_id = _reply(_line("640 c.p.", "gli artifizi e i raggiri", "Tesi.", ["S5"], ["A640cp.1"]))
    remedy = _reply(_line("62-bis c.p.", "A62biscp:E1", "Attenuanti.", ["S5"]))
    model = _Model(text_not_id, remedy)
    res = A.analyse(inp, call=model)
    lp = res["analysis"]["linea_principale"]
    assert len(model.seen) == 2 and lp["stato"] == A.FALLBACK_LINE == "rubrica_del_quadro"
    assert lp["articolo"] == "640 c.p." and lp["tesi"] == "Truffa: elementi da provare dall'accusa"
    assert lp["elemento_in_gioco"] is None and lp["nome"] == "truffa"
    els = res["analysis"]["elementi"]["640 c.p."]
    assert [(e["esito"], e["origine"]) for e in els] == [("da_provare", "rubrica")] * 3
    c = res["counters"]
    assert c["analysis_line_fallback_rubrica"] == 1 and c["analysis_line_undetermined"] == 0
    assert c["analysis_frame_charge_missing"] == 0 and c["analysis_bad_eid"] == 1 and c["analysis_line_rejected"] == 2
    v = A.view(res, inp)
    assert A.summary_text(v) == "Linea principale: art. 640 c.p. — Truffa: elementi da provare dall'accusa."
    assert "| Linea principale | art. 640 c.p. | — | Truffa: elementi da provare dall'accusa | — |" in \
        A.solutions_text(v)
    assert "**Art. 640 c.p.** — Truffa" in A.norms_tables(v) and "da provare 3 su 3" in A.norms_tables(v)
    assert A.conclusions_text(v).startswith("**Linea principale**: art. 640 c.p. — Truffa: elementi da provare")


def test_the_civil_fallback_tesi_and_the_istituto_as_the_named_claim():
    frame = _no_charge(T.CIVIL_FRAME)                    # istituto «caparra confirmatoria»
    remedy = _reply(_line("1453 c.c.", T.E1453[0], "Risoluzione.", ["S4"], ["A1453cc.1"]))
    res = A.analyse(T._civil(frame=frame), call=_Model(remedy, remedy))
    lp = res["analysis"]["linea_principale"]
    assert lp["articolo"] == "1385 c.c." and lp["tesi"] == "Caparra confirmatoria: elementi da provare"
    assert A.FALLBACK_TESI == {"penale": "{rubrica}: elementi da provare dall'accusa",
                               "civile": "{rubrica}: elementi da provare"}


def test_the_rubrica_match_is_exact_after_case_and_accents_only():
    assert A.rubrica_key("  Risolubilità del contratto per INADEMPIMENTO. ") == \
        A.rubrica_key("risolubilita del contratto per inadempimento")
    for name, hit in (("TRUFFA", True), ("Truffa.", True), ("truffe", False), ("truffa aggravata", False),
                      ("la truffa", False), ("Circostanze attenuanti generiche", False)):     # the last: a remedy
        frame = _no_charge(T.PENAL_FRAME, addebito_o_domanda={"articolo": None, "descrizione": name, "frase": "«x»"})
        got = A.rubrica_fallback(T._penal(frame=frame))
        assert bool(got) == hit, name
    # nothing matches: «da determinare»
    frame = _no_charge(T.PENAL_FRAME, addebito_o_domanda={"articolo": None, "descrizione": "estorsione",
                                                          "frase": "«x»"})
    remedy = _reply(_line("62-bis c.p.", "A62biscp:E1", "Attenuanti.", ["S5"]))
    res = A.analyse(T._penal(frame=frame), call=_Model(remedy, remedy))
    assert res["analysis"]["linea_principale"]["stato"] == "da_determinare"
    assert res["counters"]["analysis_line_undetermined"] == 1 and res["counters"]["analysis_line_fallback_rubrica"] == 0


def test_the_frames_charge_article_comes_before_the_rubrica_fallback():
    remedy = _reply(_line("62-bis c.p.", "A62biscp:E1", "Attenuanti.", ["S5"]))
    res = A.analyse(T._penal(), call=_Model(remedy, remedy))          # the frame names 640 c.p. and «truffa»
    assert res["analysis"]["linea_principale"]["stato"] == "addebito_del_quadro"
    c = res["counters"]
    assert c["analysis_frame_charge_missing"] == 1 and c["analysis_line_fallback_rubrica"] == 0


def test_a_failed_first_call_gives_nothing_not_the_rubrica_line():
    frame = _no_charge(T.PENAL_FRAME, addebito_o_domanda={"articolo": None, "descrizione": "truffa", "frase": "«x»"})
    res = A.analyse(T._penal(frame=frame), call=_Model(RuntimeError("down")))
    assert res["status"] == "failed" and res["analysis"] is None


def test_the_older_frames_central_issues_count_only_when_grounded_in_the_case():
    state = {"analysis": {"frame": {"central_issue_candidates": [{"nome": "Truffa"}, {"nome": "Estorsione"},
                                                                 "Insolvenza fraudolenta"]}}}
    sents = _sents(T.PENAL_SENTENCES, sistema=(9,))
    assert A.older_issues(state, sents) == ["Truffa"]
    frame = {k: v for k, v in T.PENAL_FRAME.items() if k != "addebito_o_domanda"}
    inp = A.make_input(frame, T.PENAL_NORMS, sents, [], client="Paolo Sarti", older_issues=["Truffa"])
    assert inp["issues"] == ["Truffa"] and A.rubrica_fallback(inp)["articolo"] == "640 c.p."
    # the 79.14 frame's own names first, each once
    assert A.named_issues({"addebito_o_domanda": {"descrizione": "truffa"}, "istituto": {"testo": "Truffa"}},
                          ["Insolvenza fraudolenta"]) == ["truffa", "Insolvenza fraudolenta"]


# --- 3. the claimant's line, §7 ---------------------------------------------------------------------------------------

def test_who_is_the_claimant():
    assert A.claimant(APPALTO_FRAME) and A.claimant(T.CIVIL_FRAME)
    assert not A.claimant(T.PENAL_FRAME) and not A.claimant({})
    assert not A.claimant(dict(APPALTO_FRAME, ruolo={"valore": "convenuto", "intake": "convenuto"}))
    assert A.claimant(dict(APPALTO_FRAME, ruolo={"valore": "ricorrente", "intake": "attore"}))
    no_role = {k: v for k, v in APPALTO_FRAME.items() if k != "ruolo"}
    assert A.claimant(no_role)                                          # «committente» in the client's ruolo_nel_fatto
    assert not A.claimant(dict(no_role, parti=[{"nome": "Marta Galli", "ruolo_nel_fatto": "appaltatrice",
                                                "lato": "cliente"}]))


def test_an_exception_element_by_the_keywords_and_by_the_schemas_own_kind():
    inp = _appalto()
    els = inp["norms"][0]["elements"]
    assert [(e["eccezione"], e["eccezione_fonte"]) for e in els] == [
        (False, "parole_chiave"), (True, "parole_chiave"), (True, "parole_chiave"), (True, "parole_chiave")]
    periods = A._ap().article_ids("1667 c.c.", ART_1667)
    assert A._clause_lead(E1667[1], periods) == "la garanzia non e dovuta se "
    # the schema's own kind decides when the element carries one -- the keywords not read
    assert A.exception_of({"testo": E1667[1], "schema_kind": "condizione"}, periods) == (False, "schema")
    assert A.exception_of({"testo": E1667[0], "schema_kind": "negazione"}, periods) == (True, "schema")
    assert A.exception_of({"testo": E1667[0], "schema_kind": "condotta", "special": "limite"}, periods) == \
        (True, "schema")
    # 1453's «salvo, in ogni caso, il risarcimento» comes after its element: never its clause
    assert A.exception_of({"testo": T.E1453[0]}, A._ap().article_ids("1453 c.c.", T.ART_1453)) == \
        (False, "parole_chiave")


def test_inputs_carry_the_schema_kind_of_the_records_requisites(monkeypatch):
    monkeypatch.setenv("ANALYSIS_STEP", "true")
    state = T._state()
    state["records"]["640 c.p."]["requisites"][1]["schema_kind"] = "negazione"
    els = A.inputs(state)["norms"][0]["elements"]
    assert els[1]["schema_kind"] == "negazione" and els[1]["eccezione"] and els[1]["eccezione_fonte"] == "schema"
    assert "schema_kind" not in els[0] and els[0]["eccezione_fonte"] == "parole_chiave"


def test_a_claimants_exception_line_is_refused_and_re_asked_with_the_constitutive_ids():
    first = _reply(_line("1667 c.c.", E1667[1], "Marta Galli non ha accettato l'opera.", ["S3"], ["A1667cc.2"]))
    model = _Model(first, GOOD_APPALTO)
    res = A.analyse(_appalto(), call=model)
    c = res["counters"]
    assert len(model.seen) == 2 and c["analysis_line_exception"] == 1 and c["analysis_line_rejected"] == 1
    why = res["rejected"][0]["why"]
    assert "A1667cc:E2 è un'eccezione o un limite (parole_chiave)" in why and "elementi costitutivi A1667cc:E1" in why
    reask = model.seen[1]["messages"][3].content
    assert "- A1667cc (art. 1667 c.c.): A1667cc:E1\n" in reask and A.config()["reask_claimant"] in reask
    an = res["analysis"]
    assert an["linea_principale"]["eid"] == "A1667cc:E1" and an["claimant"]
    assert [x["eid"] for x in an["eccezioni"]] == ["A1667cc:E2", "A1667cc:E3", "A1667cc:E4"]
    # the objections: the company's allegation on E2 (the raised exception folded into it), the exception E4 proven
    objs = an["obiezioni"]
    assert [(o["fonte"], o.get("sid"), o.get("eid")) for o in objs] == [("allegazione", "S3", None),
                                                                        ("eccezione", None, "A1667cc:E4")]
    assert objs[0]["parte"] == "Edilnova S.r.l."
    assert [e["eid"] for e in objs[0]["elementi"]] == ["A1667cc:E2"]
    assert c["objections_from_allegations"] == 1 and c["objections_from_exceptions"] == 1
    # an exception line twice: refused twice, the frame's charge then leads with every element open
    model = _Model(first, first)
    res = A.analyse(_appalto(), call=model)
    assert res["counters"]["analysis_line_exception"] == 2 and res["analysis"]["linea_principale"]["stato"] == \
        "addebito_del_quadro"
    assert [(o["fonte"], o["eid"]) for o in res["analysis"]["obiezioni"]] == [("eccezione", "A1667cc:E2")]
    assert res["counters"]["objections_from_exceptions"] == 1


def test_a_defendant_may_play_the_exception():
    frame = dict(APPALTO_FRAME, ruolo={"valore": "convenuto", "intake": "convenuto"},
                 parti=[{"nome": "Marta Galli", "ruolo_nel_fatto": "appaltatrice", "lato": "cliente"},
                        {"nome": "Edilnova S.r.l.", "ruolo_nel_fatto": "committente", "lato": "controparte"}])
    model = _Model(_reply(_line("1667 c.c.", E1667[1], "L'opera è stata accettata.", ["S1"], ["A1667cc.2"])))
    res = A.analyse(_appalto(frame=frame), call=model)
    assert len(model.seen) == 1 and res["analysis"]["linea_principale"]["eid"] == "A1667cc:E2"
    assert res["counters"]["analysis_line_exception"] == 0 and not res["analysis"]["claimant"]
    assert A.config()["reask_claimant"] not in A.reask_messages([], "", ["x"], _appalto(frame=frame))[-1].content


def test_the_objections_render_labels_and_the_cases_sentences_only():
    inp = _appalto()
    res = A.analyse(inp, call=_Model(GOOD_APPALTO))
    items = A.objection_items(A.view(res, inp))
    assert [i["fonte"] for i in items] == ["allegazione", "eccezione"]
    # v79.15.3 (item 3): one plain form -- the allegation's sentence as written; the answer the status in words (a
    # claimant: «è da provare»), then the client's own sentences
    assert items[0]["obiezione"] == "La Edilnova S.r.l. sostiene che Marta Galli ha accettato l'opera senza riserve"
    assert items[0]["risposta"] == "L'elemento «il committente ha accettato l'opera» (art. 1667 c.c.) è da provare."
    assert items[1]["obiezione"] == ("Eccezione dell'art. 1667 c.c.: «denunziare all'appaltatore le difformità o i vizi "
                                     "entro sessanta giorni dalla scoperta»")
    assert items[1]["risposta"] == ("L'elemento «denunziare all'appaltatore le difformità o i vizi entro sessanta giorni "
                                    "dalla scoperta» (art. 1667 c.c.) risulta provato: «Marta Galli ha denunziato i "
                                    "vizi all'impresa con una raccomandata dieci giorni dopo la scoperta».")
    assert not any("Esito verificato" in i["risposta"] or "Fatti del cliente" in i["risposta"] for i in items)
    # the step not ok: none
    assert A.objection_items(A.view(None, inp)) == []


def test_an_allegation_on_two_elements_is_one_objection_with_the_clients_own_sentences():
    reply = _reply(_line("640 c.p.", T.E640[0], "Gli artifizi non sono provati.", ["S3", "S5"], ["A640cp.1"]),
                   elementi=[_el("640 c.p.", T.E640[0], "provato", ["S3"]),
                             _el("640 c.p.", T.E640[1], "contestato", ["S3"]),
                             _el("640 c.p.", T.E640[2], "contestato", ["S4"])])
    inp = T._penal()
    res = A.analyse(inp, call=_Model(reply))
    objs = res["analysis"]["obiezioni"]
    assert [(o["sid"], o["parte"], [e["eid"] for e in o["elementi"]]) for o in objs] == [
        ("S3", "la Procura", ["A640cp:E1", "A640cp:E2"]), ("S4", "Giulia Neri", ["A640cp:E3"])]
    assert objs[0]["fatti_cliente"] == ["S5"]                 # the line's own fact of the client (the element in play)
    assert res["counters"]["objections_from_exceptions"] == 0   # an imputato: no exception objections


def test_section_7_prints_the_verified_objections_first_and_the_no_objections_line_only_without_any():
    from src.rag import lawyer_method as M
    none = "Il materiale non indica obiezioni della controparte sui fatti narrati."
    assert M.objections_section({"obiezioni": []}, [], verified=[]) == none
    v = [{"obiezione": f"Allegazione {i}", "risposta": "Esito verificato: contestato."} for i in range(4)]
    text = M.objections_section({"obiezioni": [{"obiezione": "Del piano", "risposta": "R."}]}, [], verified=v)
    assert none not in text and text.count("**Obiezione ") == 4 and "Del piano" not in text      # never cut
    text = M.objections_section({"obiezioni": [{"obiezione": "Del piano", "risposta": "R."}]}, [], verified=v[:1])
    assert text.startswith("**Obiezione 1.** Allegazione 0.\n**Risposta.** Esito verificato: contestato.")
    assert "**Obiezione 2.** Del piano." in text
    # the flag off: unchanged
    assert M.objections_section({"obiezioni": [{"obiezione": "A", "risposta": "B"}]}) == \
        "**Obiezione 1.** A.\n**Risposta.** B"


def test_the_step_logs_the_claimant_the_exceptions_and_the_objections(monkeypatch):
    events = []
    from src.rag import verbose_logger
    monkeypatch.setattr(verbose_logger, "vlog", lambda event, data=None, duration_ms=None: events.append((event, data)))
    A.analyse(_appalto(), call=_Model(GOOD_APPALTO))
    line = [d for e, d in events if e == "analysis_step"][0]
    assert line["claimant"] is True and [x["fonte"] for x in line["exceptions"]] == ["parole_chiave"] * 3
    assert [o["fonte"] for o in line["objections"]] == ["allegazione", "eccezione"]
    assert line["lines"][0]["eid"] == "A1667cc:E1"
    assert flow is not None


# --- review fixes: the first reply's verified line kept; a bare E-id ambiguous on two articles ----------------------

NO_ARTICLE_TRUFFA = dict(T.PENAL_FRAME, addebito_o_domanda={"articolo": None, "descrizione": "truffa",
                                                           "frase": "«truffa»"})
FIRST_OK_SUB_BAD = _reply(_line("640 c.p.", T.E640[0], "L'accusa deve provare gli artifizi.", ["S3"], ["A640cp.1"]),
                          subordinata={"articolo": "641 c.p.", "elemento_in_gioco": "A641cp:E9", "tesi": "x",
                                       "cita_fatti": ["S1"], "cita_norma": []},
                          elementi=[_el("640 c.p.", T.E640[2], "provato", ["S7"])])
SECOND_LINE_BAD = _reply({"articolo": "640 c.p.", "elemento_in_gioco": "con artifizi o raggiri", "tesi": "t",
                          "cita_fatti": ["S3"], "cita_norma": []})


def test_a_leading_line_verified_on_the_first_reply_survives_a_re_ask_that_rejects_it():
    """The re-ask was asked for the subordinate line (A641cp:E9 does not exist); its own leading line is rejected
    (a text, not an id): the first reply's verified leading line stays -- never the rubrica fallback."""
    r = A.analyse(T._penal(frame=NO_ARTICLE_TRUFFA), call=_Model(FIRST_OK_SUB_BAD, SECOND_LINE_BAD))
    a, c = r["analysis"], r["counters"]
    assert c["analysis_calls"] == 2 and c["analysis_line_rejected"] == 2
    assert a["linea_principale"]["stato"] == A.VERIFIED and a["linea_principale"]["eid"] == "A640cp:E1"
    assert a["linea_principale"]["tesi_modello"] == "L'accusa deve provare gli artifizi."      # v79.15.3: logged only
    assert c["analysis_line_kept_first"] == 1
    assert c["analysis_line_fallback_rubrica"] == 0 and c["analysis_line_undetermined"] == 0
    # the first reply's element verdict is read too (by id), the second reply having none for it
    got = {e["eid"]: e["esito"] for e in a["elementi"]["640 c.p."]}
    assert got["A640cp:E3"] == "provato"


def test_the_second_replys_verified_line_wins_over_the_firsts():
    second = _reply(_line("640 c.p.", T.E640[1], "Gli artifizi non indussero in errore.", ["S2"], ["A640cp.1"]),
                    elementi=[_el("640 c.p.", T.E640[1], "da_provare")])
    r = A.analyse(T._penal(frame=NO_ARTICLE_TRUFFA), call=_Model(FIRST_OK_SUB_BAD, second))
    a, c = r["analysis"], r["counters"]
    assert a["linea_principale"]["eid"] == "A640cp:E2" and c["analysis_line_kept_first"] == 0
    got = {e["eid"]: e["esito"] for e in a["elementi"]["640 c.p."]}
    assert got["A640cp:E2"] == "da_provare"          # the second reply's verdict first


def test_a_bare_e_id_is_ambiguous_on_two_line_articles_and_read_on_one():
    two = _reply(_line("640 c.p.", T.E640[0], "L'accusa deve provare.", ["S3"], ["A640cp.1"]),
                 subordinata=_line("641 c.p.", T.E641[0], "Sub.", ["S1"], ["A641cp.1"]),
                 elementi=[{"articolo": None, "elemento": "E2", "esito": "provato", "cita_fatti": ["S2"]}])
    r = A.analyse(T._penal(frame=NO_ARTICLE_TRUFFA), call=_Model(two))
    els = r["analysis"]["elementi"]
    assert all(e["esito"] == "da_provare" for x in ("640 c.p.", "641 c.p.") for e in els[x] if e["eid"].endswith(":E2"))
    assert r["counters"]["analysis_bad_eid"] == 1
    one = _reply(_line("640 c.p.", T.E640[0], "L'accusa deve provare.", ["S3"], ["A640cp.1"]),
                 elementi=[{"articolo": None, "elemento": "E2", "esito": "da_provare", "cita_fatti": []}])
    r = A.analyse(T._penal(frame=NO_ARTICLE_TRUFFA), call=_Model(one))
    e2 = next(e for e in r["analysis"]["elementi"]["640 c.p."] if e["eid"] == "A640cp:E2")
    assert e2["origine"] == "modello" and r["counters"]["analysis_bad_eid"] == 0
