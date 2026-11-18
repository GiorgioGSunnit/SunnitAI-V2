"""v79.11 block C -- element schemas in the pipeline (SPEC §6, §7), on synthetic texts, rows, records and sentences only:

C1. decomposition: flag off, _requisites is today's for a row that HAS a schema (element_schemas never consulted);
    flag on, the schema's elements, the path per record (rec["element_path"], rec["schema"], state["element_paths"],
    the element_path log), an article with no schema on the text path
C2. the general-part gate: a schema never decomposes past the hard invariants (c.p. 43 tagged sanction stays gated, a
    procedural code's article too); an unreviewed limiting schema never gates out, a reviewed one does; the three sites
C3. grading: element_rule (today's regexes when no schema key); a schema «elemento soggettivo» with a conduct fact
    stays «da provare» (79.9-A by construction) without the mental words; the regex still applies under kind condotta;
    a «condizione» joined by «e» half proven names the missing part; a «termine» graded on the facts' durations
    (v79.10.4 D1, replacing the stand-in); the kind
    never upgrades; grading_rule only on schema elements
C4. admission in shadow: schema_shadow never changes the candidates and reports agree / false_exclusions / extra /
    recall; main_admitted_by; active_modules from a schema rapporto only when the patterns find none
C5. the counters: neutral with the flag off; elements_per_article (+ by path), element_paths, schema_rejected,
    rules_selected, schema_shadow, main_lines_by_rule
"""

import copy
import json

import pytest


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for name, value in (("LLM_API_KEY", "k"), ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "k"), ("NEO4J_URI", "bolt://localhost:7687"), ("NEO4J_USER", "n"),
                        ("NEO4J_PASSWORD", "x")):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("ARTICLE_SCHEMAS_FILE", str(tmp_path / "schemas.jsonl"))
    monkeypatch.setenv("ARTICLE_SCHEMAS_REVIEWED_FILE", str(tmp_path / "reviewed.jsonl"))


def _on(monkeypatch):
    monkeypatch.setenv("ELEMENT_SCHEMAS", "true")


@pytest.fixture
def logs(monkeypatch):
    from src.rag import interactive_flow as flow, lawyer_method as M
    out = []
    monkeypatch.setattr(flow, "_log", lambda event, data, ms=None: out.append((event, data)))
    monkeypatch.setattr(M, "_log", lambda event, data, ms=None: out.append((f"lawyer_method_{event}", data)))
    return out


# synthetic article texts and their schemas
PENAL = ("Chiunque cagiona ad alcuno una lesione personale, dalla quale deriva una malattia nel corpo o nella mente, "
         "è punito con la reclusione da sei mesi a tre anni.")
COLPOSE = "Chiunque cagiona ad altri per colpa una lesione personale è punito con la reclusione fino a tre mesi."
DOC = "Codice Penale 2026"


def _el(eid, testo, kind="condotta", comma="1", grounding="explicit", **extra):
    out = {"id": eid, "testo": testo, "kind": kind, "kind_note": None, "comma": comma, "branches": [],
           "connective": None, "parti": [], "grounding": grounding, "fonte": {"comma": comma, "testo": testo},
           "spiegazione": None}
    out.update(extra)
    return out


def _penal(tipo=("sanction",), **over):
    soggettivo = _el("E3", "coscienza e volontà di cagionare la lesione", kind="elemento soggettivo",
                     grounding="inferred", spiegazione="delitto punito solo a titolo di dolo")
    soggettivo["fonte"] = {"comma": "1", "testo": None}
    schema = {"tipo": list(tipo), "destinatario": [{"ruolo": "chiunque", "condizione": None}],
              "legittimato": [{"ruolo": "persona offesa", "condizione": None}],
              "soggetto_passivo": [{"ruolo": "persona offesa", "condizione": None}], "rapporto": "lesioni personali",
              "elementi": [_el("E1", "cagiona ad alcuno una lesione personale"),
                           _el("E2", "dalla quale deriva una malattia nel corpo o nella mente", kind="evento",
                               branches=["malattia nel corpo", "malattia nella mente"], connective="o"),
                           soggettivo],
              "logic": {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}, {"ref": "E3"}]},
              "eccezioni": [], "conseguenza": {"testo": "è punito con la reclusione da sei mesi a tre anni",
                                               "tipo": "pena", "comma": "1"},
              "commi_speciali": [], "collegati": []}
    schema.update(over)
    return schema


def _write(path, lines):
    with open(path, "a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")


def _gen(number="582", code="c.p.", text=PENAL, schema=None, doc=DOC):
    from src.rag import element_schemas as E
    return E.make_line({"number": number, "code": code, "rubrica": "Lesione personale", "text": text}, "m-test",
                       _penal() if schema is None else schema, doc=doc)


def _schemas(tmp_path, *lines):
    _write(tmp_path / "schemas.jsonl", lines or [_gen()])


def _row(number="582", code="c.p.", text=PENAL, rubrica="Lesione personale", doc=DOC):
    return {"_article": (number, code), "d": {"name": doc}, "s": {"plain_text": text, "titolo_articolo": rubrica}}


# --- C1 decomposition ----------------------------------------------------------------------------------------------------

def test_flag_off_requisites_are_todays_for_a_row_with_a_schema(monkeypatch, tmp_path):
    from src.rag import element_schemas as E, issue_records
    _schemas(tmp_path)
    got = issue_records._requisites(_row(), {})
    monkeypatch.setattr(E, "for_row", lambda row: pytest.fail("a schema read with the flag off"))
    assert issue_records._requisites(_row(), {}) == got
    monkeypatch.setattr(issue_records, "_schema_requisites", lambda *a, **k: None)      # today's function, verbatim
    assert issue_records._requisites(_row(), {}) == got
    assert got and not any("schema_kind" in r or r.get("split") == "schema" for r in got)


def test_flag_on_the_schema_elements_replace_the_text_split(monkeypatch, tmp_path):
    from src.rag import issue_records
    _schemas(tmp_path)
    today = issue_records._requisites(_row(), {})
    _on(monkeypatch)
    frags = []
    reqs = issue_records._requisites(_row(), {}, frags)
    assert reqs != today and frags == []
    assert [r["text"] for r in reqs] == ["cagiona ad alcuno una lesione personale",
                                         "dalla quale deriva una malattia nel corpo o nella mente",
                                         "coscienza e volontà di cagionare la lesione"]
    assert all(r["split"] == "schema" and r["kind"] == "requisito" and r["comma"] == 1 for r in reqs)
    assert [r["schema_kind"] for r in reqs] == ["condotta", "evento", "elemento soggettivo"]
    assert reqs[1]["branches"] == ["malattia nel corpo", "malattia nella mente"]
    # an article with no schema, one whose text changed: the text path, unchanged
    assert issue_records._requisites(_row("590", text=COLPOSE), {}) == \
        _flag_off(monkeypatch, lambda: issue_records._requisites(_row("590", text=COLPOSE), {}))
    assert not any(r.get("split") == "schema" for r in issue_records._requisites(_row(text=PENAL + " Cambiato."), {}))


def _flag_off(monkeypatch, run):
    monkeypatch.delenv("ELEMENT_SCHEMAS", raising=False)
    try:
        return run()
    finally:
        monkeypatch.setenv("ELEMENT_SCHEMAS", "true")


def test_a_schema_that_yields_no_element_leaves_the_text_path(monkeypatch, tmp_path):
    from src.rag import issue_records
    only = _penal(elementi=[_el("E1", "è punito con la reclusione da sei mesi a tre anni", kind="quantificazione")],
                  logic={"ref": "E1"})
    _schemas(tmp_path, _gen(schema=only))
    today = issue_records._requisites(_row(), {})
    _on(monkeypatch)
    assert issue_records._requisites(_row(), {}) == today
    assert issue_records.element_path(_row(), today) == {"path": "text", "why": "schema_empty"}


def test_build_records_the_path_per_record_and_logs_it(monkeypatch, tmp_path, logs):
    from src.rag import issue_records
    _schemas(tmp_path)
    topic = {"title": "Lesioni", "articles": ["582 c.p.", "590 c.p."]}
    rows = [_row(), _row("590", text=COLPOSE, rubrica="Lesioni personali colpose")]
    off = issue_records.build([topic], rows, {})
    assert not any("element_path" in r or "schema" in r for r in off.values())
    assert not [e for e, _ in logs if e in ("element_path", "element_rule", "schema_gate_disagree")]
    _on(monkeypatch)
    recs = issue_records.build([topic], rows, {})
    assert recs["582 c.p."]["element_path"] == {"path": "schema", "why": "schema"}
    assert recs["590 c.p."]["element_path"] == {"path": "text", "why": "no_schema"}
    strip = lambda reqs: [{k: v for k, v in r.items() if k != "id"} for r in reqs]
    assert strip(recs["590 c.p."]["requisites"]) == strip(off["590 c.p."]["requisites"])   # no schema: today's path
    assert all(r["split"] == "schema" for r in recs["582 c.p."]["requisites"])
    assert recs["582 c.p."]["schema"]["tipo"] == ["sanction"] and recs["582 c.p."]["schema"]["source"] == "schema"
    assert "schema" not in recs["590 c.p."]
    paths = [d for e, d in logs if e == "element_path"]
    assert {"article": "582 c.p.", "path": "schema", "why": "schema"} in paths
    assert {"article": "590 c.p.", "path": "text", "why": "no_schema"} in paths


IMPUTATO = {"rito": "penale", "family": "penale", "role": "imputato", "stage": "primo_grado", "intake_rito": "penale"}


def _decompose(rows):
    from src.rag import interactive_flow as flow
    state = {"analysis": {}, "text": "Il PM contesta a Tizio l'art. 582 c.p.", "request": "", "intake": {},
             "records": {}, "sentences": [], "questions": []}
    flow.decompose_records(state, IMPUTATO, rows, set(), fetch=lambda refs: [])
    return state


def test_decompose_records_keeps_the_path_and_state_element_paths(monkeypatch, tmp_path, logs):
    _schemas(tmp_path)
    off = _decompose([_row()])
    assert "element_paths" not in off and "element_path" not in off["records"]["582 c.p."]
    _on(monkeypatch)
    state = _decompose([_row()])
    rec = state["records"]["582 c.p."]
    assert rec["element_path"] == {"path": "schema", "why": "schema"} and rec["schema"]["rapporto"] == "lesioni personali"
    assert state["element_paths"] == {"582 c.p.": {"path": "schema", "why": "schema"}}
    split = [d for e, d in logs if e == "decompose_records"][-1]["split"]["582 c.p."]
    assert split["by"] == "schema" and split["elements"] == 3
    assert ("element_path", {"article": "582 c.p.", "path": "schema", "why": "schema"}) in logs


def test_a_reviewed_overlay_is_the_reviewed_path(monkeypatch, tmp_path):
    from src.rag import element_schemas as E, issue_records
    _schemas(tmp_path)
    line = _gen()
    _write(tmp_path / "reviewed.jsonl", [{"label": "582 c.p.", "reviewed": True, "source_hash": line["source_hash"],
                                          "schema": line["schema"], "schema_version": 1}])
    _on(monkeypatch)
    reqs = issue_records._requisites(_row(), {})
    assert issue_records.element_path(_row(), reqs) == {"path": "reviewed", "why": "reviewed"}
    assert issue_records.schema_summary(E.for_row(_row())[0])["reviewed"] is True


# --- C2 the gate --------------------------------------------------------------------------------------------------------

def _line(tipo, reviewed=False):
    return {"schema": {"tipo": list(tipo)}, "reviewed": reviewed, "source": "reviewed" if reviewed else "schema"}


def test_a_sanction_schema_never_opens_the_hard_invariants(monkeypatch, logs):
    from src.rag import issue_records
    _on(monkeypatch)
    for reviewed in (False, True):
        assert not issue_records.decomposed_by_gate("43 c.p.", "Elemento psicologico del reato",
                                                    schema=_line(["sanction"], reviewed))
        assert not issue_records.decomposed_by_gate("405 c.p.p.", "", schema=_line(["claim-conferring"], reviewed))
    assert issue_records.decomposed_by_gate("43 c.p.", "", defence=True, schema=_line(["sanction"]))   # v79 D1 stays
    assert issue_records.decomposed_by_gate("52 c.p.", "Difesa legittima", schema=_line(["scriminante"]))
    assert any(e == "schema_gate_disagree" and d["article"] == "43 c.p." and d["by"] == "invariant" for e, d in logs)


def test_an_unreviewed_limiting_schema_never_gates_out_a_reviewed_one_does(monkeypatch, logs):
    from src.rag import issue_records
    _on(monkeypatch)
    assert issue_records.decomposed_by_gate("1453 c.c.", "Risolubilità del contratto", schema=_line(["limiting"]))
    assert not issue_records.decomposed_by_gate("1453 c.c.", "Risolubilità del contratto",
                                                schema=_line(["limiting"], reviewed=True))
    assert [d["by"] for e, d in logs if e == "schema_gate_disagree"] == ["current", "schema"]
    # a decomposable tipo opens a civil article the rubrica gate refuses
    assert not issue_records.decomposed_by_gate("1999 c.c.", "Nozione")
    assert issue_records.decomposed_by_gate("1999 c.c.", "Nozione", schema=_line(["claim-conferring"]))
    # a record's summary reads as its line (method_records' site)
    assert not issue_records.decomposed_by_gate("1453 c.c.", "", schema={"tipo": ["definitional"], "reviewed": True,
                                                                          "source": "reviewed"})


def test_the_gate_with_the_flag_off_ignores_any_schema(logs):
    from src.rag import issue_records
    assert issue_records.decomposed_by_gate("1453 c.c.", "", schema=_line(["limiting"], reviewed=True))
    assert not issue_records.decomposed_by_gate("43 c.p.", "", schema=_line(["sanction"], reviewed=True))
    assert not logs


def test_the_three_gate_sites(monkeypatch, tmp_path):
    from src.rag import interactive_flow as flow, issue_records, lawyer_method as M
    lim = _penal(tipo=("limiting",))
    line = _gen(schema=lim)
    _schemas(tmp_path, line)
    _on(monkeypatch)
    topic = {"title": "Lesioni", "articles": ["582 c.p."]}
    assert issue_records.build([topic], [_row()], {})["582 c.p."]["requisites"]       # unreviewed: still decomposed
    _write(tmp_path / "reviewed.jsonl", [{"label": "582 c.p.", "reviewed": True, "source_hash": line["source_hash"],
                                          "schema": lim, "schema_version": 1}])
    rec = issue_records.build([topic], [_row()], {})["582 c.p."]
    assert rec["requisites"] == [] and rec["not_fattispecie"] == issue_records.NOT_FATTISPECIE
    state = _decompose([_row()])
    assert "582 c.p." not in state["records"] and state["decompose_why"]["582 c.p."] == "gated"
    log = {}
    records = {"582 c.p.": {"rubrica": "Lesione personale", "schema": {"tipo": ["limiting"], "reviewed": True},
                            "requisites": [{"id": "R1", "comma": 1, "kind": "requisito", "text": "x", "status": "provato"}]}}
    assert M.method_records(records, log) == {} and log["gated"] == ["582 c.p."]
    monkeypatch.delenv("ELEMENT_SCHEMAS")
    assert "582 c.p." in M.method_records(records, {})
    assert flow.DECOMPOSE_WHY        # the module imported


# --- C3 grading --------------------------------------------------------------------------------------------------------

PUSH = "Tizio spinge Caio, che cade a terra e si frattura il polso."


def _settled(text, sents, sup=(), con=(), facts="da provare", **schema):
    from src.rag import issue_records
    by_id = {s["id"]: s for s in sents}
    req = {"kind": "requisito", "text": text, "facts_status": facts,
           "link": {"supporting": list(sup), "contradicting": list(con)}, **schema}
    issue_records.settle(req, by_id)
    return req


@pytest.mark.parametrize("req,rule", [
    ({"kind": "requisito", "text": "per colpa"}, ("mental", "regex")),
    ({"kind": "requisito", "text": "il venditore deve consegnare la cosa"}, ("duty", "regex")),
    ({"kind": "procedibilita", "text": "se il fatto è commesso con armi e la persona offesa è un minore"},
     ("condition", "regex")),
    ({"kind": "requisito", "text": "se il fatto è commesso con armi e la persona offesa è un minore"}, ("plain", "regex")),
    ({"kind": "requisito", "text": "la cosa", "branches": ["mobile", "immobile"]}, ("branches", "regex")),
    ({"kind": "requisito", "text": "cagiona una lesione"}, ("plain", "regex")),
    ({"kind": "requisito", "text": "coscienza e volontà", "schema_kind": "elemento soggettivo"}, ("mental", "schema")),
    ({"kind": "requisito", "text": "con dolo", "schema_kind": "elemento soggettivo"}, ("mental", "both")),
    ({"kind": "requisito", "text": "per colpa", "schema_kind": "condotta"}, ("mental", "regex")),
    ({"kind": "requisito", "text": "entro un anno", "schema_kind": "termine"}, ("termine", "schema")),
    ({"kind": "requisito", "text": "a e b", "schema_kind": "condizione", "schema_connective": "e",
      "schema_parti": ["la disdetta", "il preavviso"]}, ("condition", "schema")),
    ({"kind": "requisito", "text": "cagiona una lesione", "schema_kind": "condotta"}, ("plain", "schema")),
])
def test_element_rule(req, rule):
    from src.rag import issue_records
    assert issue_records.element_rule(req) == rule


def test_a_schema_mental_element_with_a_conduct_fact_stays_to_prove():
    from src.rag import issue_records
    sents = issue_records.sentences(PUSH)
    text = "coscienza e volontà di cagionare la lesione"
    assert not issue_records.mental_element(text)                              # no mental word in its text
    assert _settled(text, sents, ["S1"])["status"] == "provato"                # the regex alone: today's
    req = _settled(text, sents, ["S1"], schema_kind="elemento soggettivo")
    assert req["status"] == "da provare" and req["why"] == issue_records.MENTAL_CONDUCT_WHY and req["mental_conduct"]
    assert req["grading_rule"] == {"rule": "mental", "source": "schema", "before": "provato", "after": "da provare"}
    admitted = issue_records.sentences("Tizio ha ammesso di aver spinto Caio deliberatamente.")
    assert _settled(text, admitted, ["S1"], schema_kind="elemento soggettivo")["status"] == "provato"


def test_the_regex_mental_check_stays_under_a_condotta_kind():
    from src.rag import issue_records
    req = _settled("per colpa", issue_records.sentences(PUSH), ["S1"], schema_kind="condotta")
    assert req["status"] == "da provare" and req["grading_rule"]["source"] == "regex"
    assert issue_records.polarity_counters({"x": {"requisites": [
        {"text": "coscienza e volontà", "schema_kind": "elemento soggettivo", "status": "provato", "supporting": ["S1"]}]}},
        {s["id"]: s for s in issue_records.sentences(PUSH)}) == {"mental_integrated_by_conduct": 1,
                                                                 "duty_polarity_errors": 0}


DISDETTA = "Il conduttore ha ricevuto la disdetta il 3 marzo."
CONDIZIONE = "il conduttore abbia ricevuto la disdetta e sia decorso il preavviso pattuito"
PARTI = ["il conduttore abbia ricevuto la disdetta", "sia decorso il preavviso pattuito"]


def test_a_condition_joined_by_e_half_proven_names_the_missing_part():
    from src.rag import issue_records
    sents = issue_records.sentences(DISDETTA)
    assert sents[0]["certainty"] == "accertato"
    assert _settled(CONDIZIONE, sents, ["S1"])["status"] == "provato"           # a requisito: no C3 today
    req = _settled(CONDIZIONE, sents, ["S1"], schema_kind="condizione", schema_connective="e", schema_parti=PARTI)
    assert req["status"] == "da provare" and "sia decorso il preavviso pattuito" in req["why"]
    assert req["parts"] == {PARTI[0]: True, PARTI[1]: False}
    assert req["grading_rule"]["rule"] == "condition"
    both = issue_records.sentences(DISDETTA + " Il preavviso pattuito è decorso il 3 settembre.")
    assert _settled(CONDIZIONE, both, ["S1", "S2"], schema_kind="condizione", schema_connective="e",
                    schema_parti=PARTI)["status"] == "provato"
    # connective «o»: no per-part check
    assert _settled(CONDIZIONE, sents, ["S1"], schema_kind="condizione", schema_connective="o",
                    schema_parti=PARTI)["status"] == "provato"


def test_a_branch_found_never_overturns_the_per_part_check():
    """v79.11 review: a schema «condizione» joined by «e» that also carries branches -- the branch a certain fact
    speaks of does not raise the element back to «provato» while a part is missing."""
    from src.rag import issue_records
    sents = issue_records.sentences("Tizio ha agito con violenza contro Caio.")
    assert sents[0]["certainty"] == "accertato"
    parti = ["con violenza o minaccia", "in danno di un minore"]
    req = _settled("con violenza o minaccia in danno di un minore", sents, ["S1"], schema_kind="condizione",
                   schema_connective="e", schema_parti=parti, branches=["con violenza", "con minaccia"])
    assert req["branch_status"]["con violenza"] == "provato"
    assert req["status"] == "da provare" and "in danno di un minore" in req["why"]
    assert req["parts"] == {parti[0]: True, parti[1]: False}
    assert req["grading_rule"]["after"] == "da provare"
    # no schema key: the branch's v79.6 behaviour, as today
    plain = _settled("con violenza o minaccia in danno di un minore", sents, ["S1"],
                     branches=["con violenza", "con minaccia"])
    assert plain["status"] == "provato" and "parts" not in plain


def test_a_termine_is_graded_on_the_durations_of_the_facts():
    """v79.10.4 (D1, replacing v79.11's stand-in): a term is graded against the facts' durations, on the text path and
    on a schema «termine» alike -- inside «provato», beyond «non risulta», no duration «da provare» (a supporting sentence
    with no duration no longer proves it; a fact denying the act, with no duration, no longer excludes it)."""
    from src.rag import issue_records
    text = "ne chiede il risarcimento entro un anno dalla restituzione"
    plain = issue_records.sentences("Tizio chiede il risarcimento al custode.")
    assert plain[0]["certainty"] == "accertato"
    for kind in ({}, {"schema_kind": "termine"}):
        req = _settled(text, plain, ["S1"], **kind)
        assert req["status"] == "da provare" and req["why"] == issue_records.TERMINE_OPEN_WHY
    assert issue_records.element_rule({"kind": "requisito", "text": text}) == ("termine", "regex")
    inside = issue_records.sentences("La cosa è stata restituita il 3 marzo 2025. Tizio chiede il risarcimento al custode "
                                     "il 10 settembre 2025.")
    req = _settled(text, inside, ["S2"], schema_kind="termine")
    assert req["status"] == "provato" and req["termine"]["verdict"] == "inside"
    beyond = issue_records.sentences("La cosa è stata restituita il 3 marzo 2023. Tizio chiede il risarcimento al custode "
                                     "il 10 settembre 2025.")
    req = _settled(text, beyond, ["S2"], schema_kind="termine")
    assert req["status"] == "non risulta" and "S2" in req["contradicting"] and "S2" not in req["supporting"]
    no = issue_records.sentences("Tizio non ha mai chiesto il risarcimento al custode.")
    assert _settled(text, no, [], ["S1"], schema_kind="termine")["status"] == "da provare"
    # a schema «termine» whose text states no duration keeps the stand-in's check: a dated certain fact
    unread = "nel termine stabilito dal contratto"
    assert _settled(unread, plain, ["S1"], schema_kind="termine")["why"] == issue_records.TERMINE_UNREAD_WHY
    dated = issue_records.sentences("Tizio chiede il risarcimento al custode il 3 marzo 2025.")
    assert _settled(unread, dated, ["S1"], schema_kind="termine")["status"] == "provato"


@pytest.mark.parametrize("kind", ["condotta", "evento", "elemento soggettivo", "condizione", "termine",
                                  "qualità del soggetto", "oggetto", "nesso_causale", "negazione",
                                  "elemento_finalistico", "other"])
@pytest.mark.parametrize("facts", ["da provare", "non risulta"])
def test_a_kind_never_sets_a_status_by_itself(kind, facts):
    from src.rag import issue_records
    req = _settled("il fatto avviene", issue_records.sentences(PUSH), facts=facts, schema_kind=kind,
                   schema_connective="e", schema_parti=["il fatto", "avviene"])
    assert req["status"] == "da provare"
    assert req["grading_rule"]["after"] == "da provare"


def test_grading_rule_only_on_schema_elements(monkeypatch, logs):
    from src.rag import issue_records
    sents = issue_records.sentences(PUSH)
    plain = _settled("cagiona una lesione", sents, ["S1"])
    assert "grading_rule" not in plain
    schema = _settled("cagiona una lesione", sents, ["S1"], schema_kind="condotta")
    assert schema["grading_rule"] == {"rule": "plain", "source": "schema", "before": "provato", "after": "provato"}
    assert not [e for e, _ in logs if e == "element_rule"]                    # the flag off: no log
    _on(monkeypatch)
    _settled("cagiona una lesione", sents, ["S1"])
    _settled("cagiona una lesione", sents, ["S1"], schema_kind="condotta")
    rules = [d for e, d in logs if e == "element_rule"]
    assert len(rules) == 1 and rules[0]["rule"] == "plain" and rules[0]["status"] == "provato"


# --- C4 admission (shadow), main lines, modules ---------------------------------------------------------------------------

def _summary(tipo, legittimato=("attore",), rapporto="", reviewed=False):
    return {"source": "reviewed" if reviewed else "schema", "tipo": list(tipo), "reviewed": reviewed,
            "legittimato": [{"ruolo": r, "condizione": None} for r in legittimato], "rapporto": rapporto}


def _shadow_ctx():
    recs = {"2051 c.c.": {"side": "weapon", "schema": _summary(["claim-conferring"])},
            "1218 c.c.": {"side": "weapon", "schema": _summary(["limiting"])},
            "1490 c.c.": {"side": "weapon", "schema": _summary(["claim-conferring"])},
            "1667 c.c.": {"side": "weapon"}}
    cands = [{"id": "T1", "kind": "claim", "article": "2051 c.c.", "admitted_by": "modulo custodia"},
             {"id": "T2", "kind": "claim", "article": "1218 c.c.", "admitted_by": "norma generale delle obbligazioni"},
             {"id": "T3", "kind": "claim", "article": "1667 c.c.", "admitted_by": "modulo appalto"}]
    return {"records": recs, "candidates": cands, "posture": {"role": "attore"}}


def test_schema_shadow_reports_and_never_changes_the_candidates(logs):
    from src.rag import lawyer_method as M
    ctx = _shadow_ctx()
    before = copy.deepcopy(ctx)
    out = M.schema_shadow(ctx)
    assert ctx == before
    assert out["current"]["claim"] == ["2051 c.c.", "1218 c.c.", "1667 c.c."]
    assert out["schema"]["claim"] == ["2051 c.c.", "1490 c.c."]
    assert [x[:2] for x in out["false_exclusions"]] == [("1218 c.c.", "claim")]
    assert "non conferisce azione" in out["false_exclusions"][0][2]
    assert out["extra"] == [("1490 c.c.", "claim")]
    assert out["no_schema"] == ["1667 c.c."]
    assert out["recall_hit"] == 1 and out["recall_of"] == 2 and out["candidate_recall"] == 0.5
    assert out["agree"] == 3 * 3 - 2
    logged = next(d for e, d in logs if e == "lawyer_method_schema_admission")
    assert ("2051 c.c.", "claim", False) in logged["admitted"] and logged["false_exclusions"] == out["false_exclusions"]


SENTS = [{"id": "S1", "text": "Sempronio sferra uno schiaffo a Caio, che cade e si frattura il polso.", "source": "caso",
          "certainty": "accertato"}]


def _method_ctx():
    from src.rag import lawyer_method as M, posture as posture_mod
    posture = {**IMPUTATO, "entry": posture_mod.entry_for("penale", "imputato", "primo_grado")}
    reqs = [{"id": f"R{i}", "comma": 1, "kind": "requisito", "text": t, "status": "provato", "supporting": ["S1"]}
            for i, t in enumerate(["cagiona ad alcuno una lesione personale", "una malattia nel corpo"], 1)]
    recs = {"582 c.p.": {"rubrica": "Lesione personale", "side": "threat", "charged": 1, "requisites": reqs,
                         "schema": _summary(["sanction"], ("persona offesa",))}}
    return M.prepare({"records": recs, "sentences": SENTS}, posture, [], "Sempronio colpisce Caio.", "Sempronio", "",
                     list(recs), [], [], [], [], link_call=lambda *a: {}, case_id_="t79_11c")


def test_method_plan_shadow_leaves_the_candidates_as_they_are(monkeypatch):
    from src.rag import lawyer_method as M
    off = _method_ctx()
    M.method_plan(off, call=lambda messages: {"valutazioni": {}})
    assert "schema_shadow" not in off and "schema_shadow" not in off["selection_counts"]
    _on(monkeypatch)
    on = _method_ctx()
    M.method_plan(on, call=lambda messages: {"valutazioni": {}})
    assert on["candidates"] == off["candidates"]
    assert on["schema_shadow"]["current"]["contest"] == ["582 c.p."]
    assert on["schema_shadow"]["schema"]["contest"] == ["582 c.p."] and on["schema_shadow"]["candidate_recall"] == 1.0
    assert on["selection_counts"]["schema_shadow"]["recall_of"] == 1
    assert off["selection_counts"]["main_admitted_by"] == on["selection_counts"]["main_admitted_by"]
    main = next(c for c in off["candidates"] if c["id"] == off["selection_counts"]["chosen"])
    assert off["selection_counts"]["main_admitted_by"] == main["admitted_by"] != ""


def test_active_modules_from_a_schema_rapporto_only_when_patterns_find_none(monkeypatch, logs):
    from src.rag import lawyer_method as M
    recs = {"2051 c.c.": {"side": "weapon", "schema": _summary(["claim-conferring"], rapporto="custodia")},
            "1667 c.c.": {"side": "threat", "schema": _summary(["claim-conferring"], rapporto="appalto")}}
    quiet = {"records": recs, "case": "Tizio e Caio litigano per una somma.", "institutions": []}
    assert M.active_modules(quiet) == []
    _on(monkeypatch)
    assert M.active_modules(quiet) == []                     # an unreviewed (generated) schema never activates a module
    assert any(e == "lawyer_method_schema_modules_shadow" and d["modules"] == ["custodia"] for e, d in logs)
    quiet["records"] = {**recs, "2051 c.c.": {"side": "weapon",
                                              "schema": _summary(["claim-conferring"], rapporto="custodia",
                                                                 reviewed=True)},
                        "1667 c.c.": {"side": "threat", "schema": _summary(["claim-conferring"], rapporto="appalto",
                                                                           reviewed=True)}}
    assert [m["id"] for m in M.active_modules(quiet)] == ["custodia"]        # the threat's rapporto is not the client's
    assert M.active_modules({"case": quiet["case"], "institutions": []}) == []          # family_gate.context: no records
    named = {**quiet, "institutions": ["comodato"],
             "records": {**recs, "2051 c.c.": {"side": "weapon", "schema": _summary(["claim-conferring"],
                                                                                   rapporto="custodia")}}}
    assert [m["id"] for m in M.active_modules(named)] == ["comodato"]       # patterns hit: unchanged
    assert any(e == "lawyer_method_schema_module_disagree" and d["schema"] == {"custodia": ["2051 c.c."]}
               for e, d in logs)


def test_unreviewed_schema_module_never_refuses_a_claim(monkeypatch):
    """v79.11 review: a generated schema's rapporto would activate a module whose «outside» rule refuses the client's
    matter claim -- with the schema unreviewed the claim admission is the flag-off one."""
    from src.rag import lawyer_method as M
    recs = {"1575 c.c.": {"side": "weapon", "schema": _summary(["claim-conferring"], rapporto="locazione")}}
    ctx = {"records": recs, "case": "Tizio e Caio litigano per una somma.", "institutions": []}
    off = M.active_modules(ctx)
    _on(monkeypatch)
    assert M.active_modules(ctx) == off == []
    assert M.schema_rapporti(ctx) == {"locazione": ["1575 c.c."]} and M.schema_rapporti(ctx, True) == {}


# --- C5 counters --------------------------------------------------------------------------------------------------------

def _counted_state():
    rs = lambda n, **kw: [{"kind": "requisito", "text": f"e{i}", "status": "da provare", **kw} for i in range(n)]
    return {"sentences": [],
            "records": {"582 c.p.": {"requisites": rs(3, schema_kind="condotta",
                                                    grading_rule={"rule": "plain", "source": "schema"}),
                                     "element_path": {"path": "schema", "why": "schema"}},
                        "590 c.p.": {"requisites": rs(2), "element_path": {"path": "text", "why": "hash_mismatch"}},
                        "52 c.p.": {"requisites": rs(4, grading_rule={"rule": "mental", "source": "schema"}),
                                    "element_path": {"path": "reviewed", "why": "reviewed"}},
                        "1 c.c.": {"requisites": []}},
            "element_paths": {"582 c.p.": {"path": "schema", "why": "schema"},
                              "590 c.p.": {"path": "text", "why": "hash_mismatch"},
                              "52 c.p.": {"path": "reviewed", "why": "reviewed"},
                              "999 c.p.": {"path": "text", "why": "no_schema"}},
            "thesis_counts_by_client": {
                "Tizio": {"main_admitted_by": "modulo custodia",
                          "schema_shadow": {"agree": 4, "false_exclusions": 1, "extra": 0, "recall_hit": 1, "recall_of": 2}},
                "Caio": {"main_admitted_by": "", "schema_shadow": {"agree": 2, "false_exclusions": 0, "extra": 1,
                                                                    "recall_hit": 2, "recall_of": 2}}}}


def test_the_counters_are_neutral_with_the_flag_off():
    from src.rag import interactive_flow as flow
    c = flow.case_counters(_counted_state())
    assert c["elements_per_article"] is None and c["elements_per_article_by_path"] == {}
    assert c["element_paths"] == {} and c["schema_rejected"] == {} and c["rules_selected"] == {}
    assert c["schema_shadow"] == {"agree": 0, "false_exclusions": 0, "extra": 0, "candidate_recall": None}
    assert c["main_lines_by_rule"] == {}


def test_the_counters_with_the_flag_on(monkeypatch):
    from src.rag import interactive_flow as flow
    _on(monkeypatch)
    c = flow.case_counters(_counted_state())
    assert c["elements_per_article"] == 3.0
    assert c["elements_per_article_by_path"] == {"schema": 3.0, "reviewed": 4.0, "text": 2.0}
    assert c["element_paths"] == {"schema": 1, "text": 2, "reviewed": 1}
    assert c["schema_rejected"] == {"hash_mismatch": 1}
    assert c["rules_selected"] == {"plain": 3, "mental": 4}
    assert c["schema_shadow"] == {"agree": 6, "false_exclusions": 1, "extra": 1, "candidate_recall": 0.75}
    assert c["main_lines_by_rule"] == {"modulo custodia": 1, "": 1}
    assert flow.grading_counters(_counted_state())["rules_selected"] == {"plain": 3, "mental": 4}
