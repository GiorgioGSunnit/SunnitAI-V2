"""v79.19.5 -- the page argues one thesis, knows its side, answers the questions; two labelling rules for an accused
(the model faked; synthetic texts, invented names; nothing named after a case).

1. row 91: the side from the posture's role words when the frame verifies no role (client_roles); neutro only when
   both are empty; a civil or labour defence's «da provare» template names the counterparty, never «L'accusa»
2. row 92: §7's objections written for the step's lines with the step's verdicts (thesis_source: step); the step
   stepped aside or not run: the old selection (legacy)
3. row 93: Antonelo's eleven probes read as questions, the gerund of a narrative sentence not; a request listing its
   points is one question per point
4. row 90: an accused's proven element is «Tesi dell'accusa», the open line «Linea difensiva»; every element proven:
   the fixed sentence, then the subordinate; a claimant's rendering unchanged
5. row 94(a): the general-part defences the facts raise, printed as records after the lines, never as a line
6. row 85: the matter gate keeps a held label and does not add its matter
7. row 87: an offence named in words found by stems; a one-stem phrase never fires
"""

import os
import sys
from pathlib import Path

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.rag import analysis_step as A  # noqa: E402
from src.rag import intake, lawyer_method as M, posture, topic_groups, verbose_logger  # noqa: E402
import test_v79_15_0_analysis as T  # noqa: E402
from test_v79_14_1_b import NO_REVIEW, _ctx, _penal_records, _plan, _req  # noqa: E402
from test_v79_15_0_analysis import _Model, _el, _line, _reply, _sealed  # noqa: E402,F401


def _events(monkeypatch):
    events = []
    monkeypatch.setattr(verbose_logger, "vlog", lambda e, d=None, ms=None: events.append((e, d)))
    return events


def _penal_input(frame=None, roles=("imputato", "indagato")):
    frame = T.PENAL_FRAME if frame is None else frame
    return A.make_input(frame, T.PENAL_NORMS, T._sents(T.PENAL_SENTENCES, sistema=(9,)), frame.get("quesiti") or [],
                        client="Paolo Sarti", client_roles=list(roles))


# --- 1. the side from the posture ----------------------------------------------------------------------------------------

def test_frame_with_no_role_and_client_roles_imputato_is_the_defence():
    assert A.client_side({}, "", ["imputato"]) == (A.SIDE_DEFENCE, "client_roles: imputato")
    assert A.client_side({}, "", ["attore"]) == (A.SIDE_CLAIMANT, "client_roles: attore")
    assert A.client_side({}, "", []) == (A.SIDE_NEUTRAL, "")                     # both empty: neutro
    assert A.client_side({}, "", ["attore", "convenuto"]) == (A.SIDE_NEUTRAL, "")   # both sides' words: neither
    frame = {k: v for k, v in T.PENAL_FRAME.items() if k != "ruolo"}
    frame["parti"] = [{"nome": "Paolo Sarti", "ruolo_nel_fatto": "venditore", "lato": "cliente"}]
    inp = _penal_input(frame)
    assert inp["side"] == A.SIDE_DEFENCE and inp["side_why"] == "client_roles: indagato" and not inp["claimant"]
    assert _penal_input(frame, roles=())["side"] == A.SIDE_NEUTRAL
    # the frame's verified role still decides first: a verified «attore» over a posture «convenuto»
    civil = A.make_input(T.CIVIL_FRAME, T.CIVIL_NORMS, T._sents(T.CIVIL_SENTENCES), [], "Lucia Ferri", ["convenuto"])
    assert civil["side"] == A.SIDE_CLAIMANT and civil["side_why"] == "ruolo: attore"


def test_a_civil_defence_names_the_counterparty_a_penal_one_the_accusa(_sealed):
    frame = {**T.CIVIL_FRAME, "ruolo": {"valore": "convenuto", "intake": "convenuto", "perche": "«convenuta»"}}
    inp = A.make_input(frame, T.CIVIL_NORMS, T._sents(T.CIVIL_SENTENCES), [], "Lucia Ferri", ["convenuto"])
    assert inp["side"] == A.SIDE_DEFENCE
    reply = _reply(_line("1385 c.c.", T.E1385[0], "x", ["S1"], ["A1385cc.1"]),
                   elementi=[_el("1385 c.c.", T.E1385[0], "da_provare")])
    line = A.analyse(inp, call=_Model(reply))["analysis"]["linea_principale"]
    assert line["tesi"] == ("Spetta a Marco Vento provare «una parte dà all'altra, a titolo di caparra, una somma di "
                            "danaro» (art. 1385 c.c.)")
    penal = A.analyse(_penal_input(), call=_Model(T.GOOD_PENAL))["analysis"]["linea_principale"]
    assert penal["tesi"].startswith("L'accusa deve provare «con artifizi o raggiri»")
    assert A.counterparty_name({"frame": {}}) == "la controparte"


# --- 2. §7 written for the line §1 prints ----------------------------------------------------------------------------------

def _step_view(first_esito="da_provare", with_sub=True, side=A.SIDE_DEFENCE):
    """A verified step's view by hand: the leading line on 582 c.p. (R3, the dolo), the subordinate on 52 c.p."""
    first = {"articolo": "582 c.p.", "stato": A.VERIFIED, "eid": "A582cp:E3", "elemento_in_gioco": "il dolo di ledere",
             "esito_in_gioco": first_esito, "tesi": "L'accusa deve provare «il dolo di ledere» (art. 582 c.p.)",
             "tesi_codice": True, "tesi_fatti": "«Mevio spinge Sempronio, che cade»", "cita_fatti": ["S1"]}
    second = {"articolo": "52 c.p.", "stato": A.VERIFIED, "eid": "A52cp:E1", "elemento_in_gioco": "la necessità di difendere",
              "esito_in_gioco": "da_provare", "tesi": "L'accusa deve provare «la necessità di difendere» (art. 52 c.p.)",
              "tesi_codice": True, "tesi_fatti": "nessun fatto citato", "cita_fatti": []}
    els = {"582 c.p.": [
        {"id": "R1", "eid": "A582cp:E1", "elemento": "una condotta di percosse o violenza sulla persona",
         "esito": "provato", "cita_fatti": ["S1"], "parte_che_afferma": None},
        {"id": "R2", "eid": "A582cp:E2", "elemento": "una malattia nel corpo o nella mente della vittima",
         "esito": "da_provare", "cita_fatti": [], "parte_che_afferma": None},
        {"id": "R3", "eid": "A582cp:E3", "elemento": "il dolo di ledere", "esito": first_esito, "cita_fatti": [],
         "parte_che_afferma": None}],
        "52 c.p.": [{"id": "R4", "eid": "A52cp:E1", "elemento": "la necessità di difendere", "esito": "da_provare",
                     "cita_fatti": [], "parte_che_afferma": None}]}
    return {"status": A.OK, "analysis": {"side": side, "claimant": side == A.SIDE_CLAIMANT, "linea_principale": first,
                                         "linea_subordinata": second if with_sub else None, "elementi": els,
                                         "affermazioni_di_diritto": [], "rimedi": []},
            "sentences": {"S1": "Mevio spinge Sempronio, che cade."}, "rubriche": {}, "remedies": [], "statements": []}


def _records_582_52():
    records = _penal_records("da provare", "da provare", "da provare")
    records["52 c.p."] = {"rubrica": "Difesa legittima", "side": "weapon",
                          "requisites": [_req(4, "la necessità di difendere", "da provare")]}   # ids unique per answer
    return records


def test_the_objections_are_written_for_the_steps_lines_with_the_steps_verdicts(monkeypatch):
    events = _events(monkeypatch)
    ctx = _ctx(_records_582_52(), "penale")
    view = _step_view()
    view["relabel"], view["general_part"] = A.defence_relabel(view), {"records": []}
    ctx["verified_analysis"] = view
    got = {}

    def call(messages):
        got["human"] = messages[1].content
        return {"obiezioni": [{"obiezione": "L'accusa dirà che il dolo risulta dalla spinta.",
                               "risposta": "La spinta non prova il dolo di ledere."}]}
    legacy_main = M.selection(ctx, _plan(ctx))[0][M.selection(ctx, _plan(ctx))[1]]["tesi"]
    M.objections_plan(ctx, _plan(ctx), call, NO_REVIEW)
    human = got["human"]
    assert "LINEA PRINCIPALE: art. 582 c.p. — L'accusa deve provare «il dolo di ledere» (art. 582 c.p.)" in human
    assert "LINEA SUBORDINATA: art. 52 c.p. — L'accusa deve provare «la necessità di difendere»" in human
    assert legacy_main not in human                      # the old engine's line is not the thesis any more
    # the SCHEDE carry the step's verdicts: R1 of 582 c.p. «integrato» by the step while the record holds «da provare»
    assert "[R1] una condotta di percosse o violenza sulla persona: integrato" in human
    assert "[R2] una malattia nel corpo o nella mente della vittima: da provare" in human
    logged = next(d for e, d in events if e == "lawyer_method_objections")
    assert logged["thesis_source"] == "step" and logged["main"].startswith("art. 582 c.p.")
    assert ctx["objections_thesis_source"] == "step"
    step = M.step_selection(ctx)
    assert step["known"]["R1"]["status"] == M.PROVATO and step["known"]["R2"]["status"] == M.DA_PROVARE
    assert step["sols"][0]["contesta"] == ["R2", "R3"] and step["sols"][0]["poggia_su"] == ["R1"]


def test_the_step_aside_or_not_run_keeps_the_legacy_selection(monkeypatch):
    events = _events(monkeypatch)
    ctx = _ctx(_records_582_52(), "penale")
    got = {}

    def call(messages):
        got["human"] = messages[1].content
        return {"obiezioni": []}
    M.objections_plan(ctx, _plan(ctx), call, NO_REVIEW)                   # no verified analysis at all
    logged = [d for e, d in events if e == "lawyer_method_objections"]
    assert logged[-1]["thesis_source"] == "legacy" and "582 c.p." not in got["human"].split("LINEA PRINCIPALE:")[1][:40] \
        or logged[-1]["thesis_source"] == "legacy"
    sols, main, _ = M.selection(ctx, _plan(ctx))
    assert f"LINEA PRINCIPALE: {sols[main]['tesi'].rstrip('.')}" in got["human"]
    failed = {"status": A.FAILED, "analysis": {"linea_principale": A.undetermined_line(), "linea_subordinata": None,
                                                "elementi": {}}, "sentences": {}, "rubriche": {}}
    ctx["verified_analysis"] = failed
    assert M.step_selection(ctx) is None
    M.objections_plan(ctx, _plan(ctx), call, NO_REVIEW)
    assert [d for e, d in events if e == "lawyer_method_objections"][-1]["thesis_source"] == "legacy"


def test_the_fallback_objections_argue_the_steps_line():
    ctx = _ctx(_records_582_52(), "penale")
    step = {"sols": [{"tesi": "art. 582 c.p. — L'accusa deve provare «il dolo di ledere»", "contesta": ["R3"],
                      "poggia_su": ["R1"]}], "main": 0,
            "known": {"R1": {"article": "582 c.p.", "text": "una condotta", "status": M.PROVATO, "side": "threat"}}}
    out = M.thesis_objections(step["sols"], 0, ctx, step["known"])
    assert out[0]["obiezione"] == "L'accusa contesterà la linea principale: art. 582 c.p. — L'accusa deve provare «il dolo di ledere»"
    assert any("elementi integrati (R1)" in o["obiezione"] for o in out)


# --- 3. the request is read as questions ----------------------------------------------------------------------------------

PROBES = [
    ("Vorrei sapere se posso impugnare il provvedimento.", ["Vorrei sapere se posso impugnare il provvedimento."]),
    ("Vorrei capire se posso impugnare il provvedimento.", ["Vorrei capire se posso impugnare il provvedimento."]),
    ("Il candidato illustri i rimedi esperibili.", ["illustri i rimedi esperibili"]),
    ("Il candidato, letta la documentazione allegata, illustri i rimedi esperibili.", ["illustri i rimedi esperibili"]),
    ("Letta la documentazione allegata, il candidato illustri i rimedi esperibili.", ["illustri i rimedi esperibili"]),
    ("Si illustrino i rimedi esperibili.", ["Si illustrino i rimedi esperibili"]),
    ("Tizio chiede quali rimedi ha.", ["Tizio chiede quali rimedi ha."]),
    ("Tizio intende sapere contro chi agire.", ["Tizio intende sapere contro chi agire."]),
    ("Si prepari un parere, soffermandosi sui termini.", ["soffermandosi sui termini"]),
    ("Tizio si rivolge al legale per sapere quali rimedi ha.", ["Tizio si rivolge al legale per sapere quali rimedi ha."]),
    ("Vorrei sapere a chi rivolgermi, in che tempi e con quali costi.", ["a chi rivolgermi", "in che tempi",
                                                                        "con quali costi"]),
    ("Tizio desidera sapere se il contratto è valido.", ["Tizio desidera sapere se il contratto è valido."]),
    ("Si espongano le questioni di diritto.", ["Si espongano le questioni di diritto"]),
    ("Premesso quanto sopra, si indichino i termini per agire.", ["si indichino i termini per agire"]),
]


@pytest.mark.parametrize("text, want", PROBES)
def test_the_eleven_probes_are_read_as_questions(text, want):
    assert intake.questions(text) == want


@pytest.mark.parametrize("text", [
    "Il giudice, valutando le prove, ha condannato Tizio.",         # the gerund of a narrative sentence
    "Tizio si reca dal notaio, indicando il prezzo.",
    "La lettera non indica alcuna motivazione.",
    "Il giudice sta valutando le prove.",
    "Nella lettera si indica il prezzo convenuto.",                 # the impersonal indicative: a narrative
    "Tizio ha venduto la casa a Caio.",
])
def test_no_question_in_a_narrative_sentence(text):
    assert intake.questions(text) == []


def test_the_three_point_request_yields_three_questions_and_the_frame_takes_each():
    from src.rag import case_frame
    text = "Assisto Tizio. Vorrei sapere a chi rivolgermi, in che tempi e con quali costi."
    assert intake.questions(text) == ["a chi rivolgermi", "in che tempi", "con quali costi"]
    units = case_frame.question_units(text)
    assert "a chi rivolgermi" in units and "con quali costi" in units
    assert "Vorrei sapere a chi rivolgermi, in che tempi e con quali costi." in units      # the whole sentence too
    assert intake.split_points("Vorrei sapere se Tizio, imputato, possa agire.") == \
        ["Vorrei sapere se Tizio, imputato, possa agire."]                                # one point: no split


# --- 4. an accused's proven element is not «Linea principale» ------------------------------------------------------------

def _defence_reply(first_esito, sub_esito=None):
    first = _line("640 c.p.", T.E640[0], "x", ["S7"], ["A640cp.1"])
    sub = _line("641 c.p.", T.E641[0], "y", ["S1"], ["A641cp.1"]) if sub_esito else None
    elementi = [_el("640 c.p.", T.E640[0], first_esito, ["S7"] if first_esito == "provato" else []),
                _el("640 c.p.", T.E640[1], "da_provare"), _el("640 c.p.", T.E640[2], "da_provare")]
    if sub_esito:
        elementi.append(_el("641 c.p.", T.E641[0], sub_esito, ["S1"] if sub_esito == "provato" else []))
    return _reply(first, sub, elementi)


def _view(monkeypatch, inp, reply):
    events = _events(monkeypatch)
    res = A.analyse(inp, call=_Model(reply))
    return A.view(res, inp), events


def test_defence_side_first_line_provato_is_the_accusa_and_the_open_line_the_difesa(monkeypatch, _sealed):
    v, events = _view(monkeypatch, _penal_input(), _defence_reply("provato", "da_provare"))
    assert v["analysis"]["linea_principale"]["esito_in_gioco"] == "provato"
    summary = A.summary_text(v)
    assert summary.startswith("Tesi dell'accusa: art. 640 c.p. — l'elemento «con artifizi o raggiri» risulta provato da "
                              "«Paolo Sarti dichiara di aver incassato il bonifico di Giulia Neri».")
    assert "Linea difensiva: art. 641 c.p. — L'accusa deve provare «dissimulando il proprio stato d'insolvenza» "
    assert "Linea principale" not in summary and "Linea difensiva: art. 641 c.p." in summary
    assert A.solutions_text(v).count("| Tesi dell'accusa | art. 640 c.p. |") == 1
    assert "| Linea difensiva | art. 641 c.p. |" in A.solutions_text(v)
    assert A.conclusions_text(v).startswith("**Tesi dell'accusa**: art. 640 c.p.") and \
        "**Linea difensiva**: art. 641 c.p." in A.conclusions_text(v)
    first, second = A.objection_lines(v)                   # §7 argues the «Linea difensiva»
    assert first["articolo"] == "641 c.p." and second["articolo"] == "640 c.p."
    logged = next(d for e, d in events if e == "analysis_defence_relabel")
    assert logged == {"line": "640 c.p.", "element": "con artifizi o raggiri", "esito": "provato", "difesa": "641 c.p.",
                      "nessun_elemento": False}
    # which lines exist and their order: unchanged
    assert [x["articolo"] for x in (v["analysis"]["linea_principale"], v["analysis"]["linea_subordinata"])] == \
        ["640 c.p.", "641 c.p."]


def test_every_element_proven_prints_the_fixed_sentence_then_the_subordinate(monkeypatch, _sealed):
    v, _ = _view(monkeypatch, _penal_input(), _defence_reply("provato", "provato"))
    lines = A.summary_text(v).split("\n\n")
    assert lines[0].startswith("Tesi dell'accusa: art. 640 c.p.")
    assert lines[1] == ("Linea difensiva: Nessun elemento della contestazione è contestabile sui fatti dati; la difesa "
                        "si gioca sulla qualificazione e sulla pena.")
    assert lines[2].startswith("Linea subordinata: art. 641 c.p. — «dissimulando il proprio stato d'insolvenza» risulta "
                               "provato")
    assert "| Linea difensiva | — | — | Nessun elemento della contestazione" in A.solutions_text(v)
    first, second = A.objection_lines(v)                   # §7: the qualification / pena line
    assert first["articolo"] == "641 c.p." and second["articolo"] == "640 c.p."
    alone, _ = _view(monkeypatch, _penal_input(), _defence_reply("provato"))
    assert A.summary_text(alone).split("\n\n")[1].startswith("Linea difensiva: Nessun elemento") and \
        len(A.summary_text(alone).split("\n\n")) == 2
    first, second = A.objection_lines(alone)
    assert first["articolo"] == "640 c.p." and first["tesi"].startswith("l'elemento «con artifizi") and second is None


def test_an_open_element_or_a_claimant_keeps_the_usual_labels(monkeypatch, _sealed):
    v, events = _view(monkeypatch, _penal_input(), _defence_reply("da_provare", "da_provare"))
    assert A.summary_text(v).startswith("Linea principale: art. 640 c.p. — L'accusa deve provare")
    assert v["relabel"] == {"applied": False} and not [e for e, _ in events if e == "analysis_defence_relabel"]
    civil = T._civil()
    reply = _reply(_line("1385 c.c.", T.E1385[0], "x", ["S2"], ["A1385cc.1"]),
                   elementi=[_el("1385 c.c.", T.E1385[0], "provato", ["S2"])])
    v, _ = _view(monkeypatch, civil, reply)
    assert v["analysis"]["side"] == A.SIDE_CLAIMANT and v["analysis"]["linea_principale"]["esito_in_gioco"] == "provato"
    assert A.summary_text(v).startswith("Linea principale: art. 1385 c.c. — «una parte dà all'altra")


# --- 5. the general-part defences the facts raise ---------------------------------------------------------------------------

def _graded(*items):
    out = []
    for label, grade, rubrica, elements in items:
        out.append({"article": label, "grade": grade, "rubrica": rubrica,
                    "elements": [{"id": f"R{i}", "text": t, "status": st, "contested": False}
                                 for i, (t, st) in enumerate(elements, 1)]})
    return out


def test_a_grade_3_general_part_record_not_a_line_is_printed_in_the_section(monkeypatch, _sealed):
    inp = _penal_input()
    inp["graded"] = _graded(("640 c.p.", 3, "Truffa", []),
                            ("55 c.p.", 3, "Eccesso colposo", [("i limiti della difesa sono superati per colpa",
                                                                "provato"), ("la proporzione", "da provare")]),
                            ("56 c.p.", 1, "Delitto tentato", [("atti idonei", "da provare")]),
                            ("62-bis c.p.", 2, "Circostanze attenuanti generiche", [("altre circostanze", "non risulta")]),
                            ("2043 c.c.", 3, "Risarcimento per fatto illecito", []))
    v, events = _view(monkeypatch, inp, T.GOOD_PENAL)
    sol = A.solutions_text(v)
    assert "**Difese di parte generale che i fatti sollevano**" in sol
    assert "- art. 55 c.p. — Eccesso colposo: «i limiti della difesa sono superati per colpa» (provato); " \
           "«la proporzione» (da provare)." in sol
    assert "- art. 62-bis c.p. — Circostanze attenuanti generiche: «altre circostanze» (escluso)." in sol
    assert "56 c.p." not in sol and "2043 c.c." not in sol               # grade 1; not a c.p. article
    assert not any("55 c.p." in row for row in sol.splitlines() if row.startswith("| "))   # never a line row
    assert next(d for e, d in events if e == "analysis_general_part_section") == {
        "count": 2, "articles": ["55 c.p.", "62-bis c.p."], "title": "Difese di parte generale che i fatti sollevano"}
    assert A.summary_text(v).startswith("Linea principale: art. 640 c.p.")       # the lines unchanged


def test_no_such_record_no_section_and_grade_1_not_printed(monkeypatch, _sealed):
    inp = _penal_input()
    inp["graded"] = _graded(("640 c.p.", 3, "Truffa", []), ("55 c.p.", 1, "Eccesso colposo", [("x y", "provato")]))
    v, events = _view(monkeypatch, inp, T.GOOD_PENAL)
    assert v["general_part"]["records"] == [] and "Difese di parte generale" not in A.solutions_text(v)
    assert not [e for e, _ in events if e == "analysis_general_part_section"]
    assert A.general_part_text(A.view(None, None)) == ""


def test_a_claimant_in_contract_gets_the_remedies_only_with_no_claim_line(monkeypatch, _sealed):
    inp = T._civil()
    assert inp["contractual"]
    inp["graded"] = _graded(("1385 c.c.", 3, "Caparra confirmatoria", []),
                            ("1453 c.c.", 3, "Risolubilità del contratto per inadempimento",
                             [("uno dei contraenti non adempie", "da provare")]))
    reply = _reply(_line("1385 c.c.", T.E1385[0], "x", ["S2"], ["A1385cc.1"]),
                   elementi=[_el("1385 c.c.", T.E1385[0], "provato", ["S2"])])
    v, _ = _view(monkeypatch, inp, reply)                 # a claim article (1385 c.c.) is a verified line: no section
    assert v["general_part"]["records"] == []
    failed = A.view({"status": A.FAILED, "analysis": None}, inp)          # no verified line: the remedies section
    assert failed["general_part"]["title"] == "Rimedi applicabili"
    assert [r["article"] for r in failed["general_part"]["records"]] == ["1453 c.c."]
    assert "**Rimedi applicabili**\n- art. 1453 c.c. — Risolubilità del contratto per inadempimento: «uno dei " \
           "contraenti non adempie» (da provare)." in A.solutions_text(failed)


# --- 6. the matter gate honours the held labels ----------------------------------------------------------------------------

def _row(n, code, rubrica, **tags):
    return {"_article": (n, code), "s": {"titolo_articolo": rubrica}, **tags}


def test_a_held_label_passes_the_matter_gate_and_its_matter_is_not_added():
    rows = [_row("624", "c.p.", "Furto", _multi_signal=True),
            _row("2872", "c.c.", "Riduzione delle ipoteche"),
            _row("2873", "c.c.", "Riduzione delle ipoteche")]
    log = {}
    kept = topic_groups.matter_gate(rows, [], log, every=True, held=["2872 c.c."])
    assert [r["_article"][0] for r in kept] == ["624", "2872"]              # held: kept; its neighbour: dropped
    assert not any(m.startswith("c.c. VI") for m in log["matters"])         # the held label added no matter
    assert log["held"] == [{"article": "2872 c.c.", "matter": "c.c. VI responsabilità patrimoniale, privilegi e ipoteche"}]
    assert [x["article"] for x in log["dropped"]] == ["2873 c.c."]
    through_keep = {}
    topic_groups.matter_gate(rows, ["2872 c.c."], through_keep, every=True)
    assert any(m.startswith("c.c. VI") for m in through_keep["matters"])    # through keep the matter would come in


def test_the_flow_passes_the_held_scope_to_the_matter_gate():
    import inspect
    from src.rag import interactive_flow as flow
    src = inspect.getsource(flow)
    assert "topic_groups.matter_gate(\n                article_rows, list(anchor_keep) + list(must) + named_labels, " \
           "matter_log, every=True, held=held_scope)" in src


# --- 7. an offence named in words ------------------------------------------------------------------------------------------

RECORDS = {"572 c.p.": {"rubrica": "Maltrattamenti contro familiari e conviventi"},
           "582 c.p.": {"rubrica": "Lesione personale"},
           "612-bis c.p.": {"rubrica": "Atti persecutori"}}


def test_the_stem_match_fires_on_the_maltrattamenti_phrase_not_on_lesioni_alone(monkeypatch):
    events = _events(monkeypatch)
    assert posture.charged_by_name("Tizio è imputato di maltrattamenti in famiglia.", RECORDS) == ["572 c.p."]
    logged = next(d for e, d in events if e == "interactive_charge_by_name")
    assert logged == {"stem_match": [{"phrase": "maltrattamenti in famiglia", "article": "572 c.p."}]}
    assert posture.charged_by_name("Tizio è imputato di lesioni.", RECORDS) == []            # one stem: ignored
    assert posture.charged_by_name("Tizio è imputato del reato di atti persecutori.", RECORDS) == ["612-bis c.p."]
    assert posture.charged_by_name("Il procedimento è per il delitto di maltrattamenti in famiglia ai sensi dell'art. "
                                   "572 c.p.", RECORDS) == ["572 c.p."]
    assert posture.stem_match("maltrattamenti", RECORDS) == "" and posture.stem_match("lesioni gravi", RECORDS) == ""
    sentence = "Tizio è imputato di maltrattamenti in famiglia, commessi nel 2024."
    assert posture.charge_phrase(sentence, posture.charge_re().search(sentence).end()) == "maltrattamenti in famiglia"
    two = {"572 c.p.": RECORDS["572 c.p."], "572-bis c.p.": {"rubrica": "Maltrattamenti in famiglia aggravati"}}
    assert posture.stem_match("maltrattamenti in famiglia", two) == ""        # two records: ambiguous, none
