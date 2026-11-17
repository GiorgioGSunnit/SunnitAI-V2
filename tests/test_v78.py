"""v78 (RULINGS_RECORD): the Cassazione rulings into the record -- synthetic tests per point, no database, no model.

1. estremi from the file name only (archive, section code, year, number; the chamber from data/ruling_sections.json;
   the date from date_enacted, else «data n.d.»; document_number never over the file name)
2. the fetch per record article, grouped by ruling, only the linked ones; found / linked / kept logged per article
3. the abstract only ranks: Sezioni Unite first, then the score (overlap + the call's pertinenza), then the date;
   3 per article, 8 per answer; pertinenza 0 dropped
4. the quoted passage is our sentence, verbatim, the one the call names; none -> the estremi alone; never «massima»
5. the orientation computed per controverted point (>= 3 with a majority: prevalente / minoritario; else to verify),
   the fixed corpus line
6. the writer's rule flips; a ruling outside the registry is dropped from the prose; the thesis and objections calls
   receive the registry; a prevailing line against a contest lowers its realismo to basso
7. the lawyer's pronunce di riferimento, only when he wrote them
8. no case-law search and no primary-to-primary expansion into the pool
9. the nightly's check, the held-out capture, the structure check of §5
"""

import json
import sys
from datetime import date
from pathlib import Path

import pytest

from src.rag import landmark_rulings, lawyer_method, rulings

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))


@pytest.fixture(autouse=True)
def _model_env(monkeypatch):
    """The code helpers (topic_step._code) import the model client: dummy keys, as the flow's tests set."""
    for name, value in (("LLM_API_KEY", "k"), ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "k"), ("NEO4J_URI", "bolt://localhost:7687")):
        monkeypatch.setenv(name, value)


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv(rulings.FLAG, "true")


# --- 1. estremi ----------------------------------------------------------------------------------------------------------------

def test_the_estremi_come_from_the_file_name_in_every_separator():
    assert rulings.parse_estremi("snpen@s10@a2025@n00001@tS.clean.pdf") == {
        "archive": "penale", "section": "10", "year": "2025", "number": 1, "act": "S"}
    assert rulings.parse_estremi("snciv_sU0_a2026_n12726_tD.clean.pdf")["section"] == "U0"
    assert rulings.parse_estremi("[CIVILE] snciv s30 a2025 n00777 tO")["number"] == 777
    assert rulings.parse_estremi("Ordinanza Rossi.pdf") is None and rulings.parse_estremi(None) is None


def test_the_estremi_render_as_asked_with_the_date_or_data_nd():
    su = rulings.parse_estremi("snpen@sU0@a2025@n12345@tS")
    assert rulings.render_estremi(su, "2025-03-12") == "Cass. pen., Sez. Un., n. 12345/2025 (12.03.2025)"
    assert rulings.sezioni_unite(su)
    third = rulings.parse_estremi("snciv_s30_a2026_n00678_tO")
    assert rulings.render_estremi(third, None) == "Cass. civ., Sez. 3, n. 678/2026 (data n.d.)"
    assert rulings.render_estremi(third, "1999-01-01").endswith("(data n.d.)")         # a date far from the year: no
    assert rulings.render_estremi(third, "05/02/2026").endswith("(05.02.2026)")
    sub = rulings.parse_estremi("snciv_s61_a2026_n00001_tO")
    assert rulings.chamber(sub) == "Sez. 6-1" and not rulings.sezioni_unite(sub)
    unmapped = rulings.parse_estremi("snciv_sZ0_a2026_n00002_tO")
    assert rulings.render_estremi(unmapped) == "Cass. civ., n. 2/2026 (data n.d.)"     # no chamber, never a guess


def test_the_section_mapping_is_marked_provisional_until_antonelo_confirms_it():
    data = json.loads(Path(rulings.SECTIONS_FILE).read_text(encoding="utf-8"))
    assert data["confirmed"] is False and "Antonelo" in data["source"] and data["sezioni_unite"] == ["U"]


# --- a graph and a model, synthetic ---------------------------------------------------------------------------------------------

PASSAGE = ("In tema di omicidio preterintenzionale, l'elemento soggettivo è costituito dalla volontà di percuotere o "
           "ledere, senza che sia necessaria la prevedibilità dell'evento morte.")
TEXT = ("Il ricorrente lamenta la violazione di legge. " + PASSAGE + " Il ricorso va pertanto rigettato. "
        "Le spese seguono la soccombenza.")


class Session:
    """FETCH_QUERY: per (doc, num) the citing sections; TEXT_QUERY: their plain_text."""

    def __init__(self, by_article, texts):
        self.by_article, self.text_of, self.queries = by_article, texts, []

    def run(self, query, **kw):
        self.queries.append(query)
        if query == rulings.FETCH_QUERY:
            rows = self.by_article.get((kw["doc"], kw["num"]), [])
        else:
            rows = [{"section": i, "plain_text": self.text_of.get(i, "")} for i in kw["ids"]]

        class R:
            def data(self_inner):
                return rows
        return R()


def _ruling(i, section="50", year="2025", linked=True, abstract="omicidio preterintenzionale volontà percuotere",
            when="2025-01-%02d"):
    return {"ruling": f"R{i}", "file_name": f"snpen@s{section}@a{year}@n{i:05d}@tS.clean.pdf", "name": "",
            "document_number": "999999", "date_enacted": when % (i % 27 + 1) if "%" in when else when,
            "linked": linked, "section": f"S{i}", "abstract": abstract}


def _ctx():
    records = {"584 c.p.": {"side": "threat", "rubrica": "Omicidio preterintenzionale", "requisites": [
        {"id": "R1", "text": "atti diretti a percuotere o ledere", "status": "provato", "kind": "requisito"},
        {"id": "R2", "text": "volontà di percuotere, senza prevedibilità della morte", "status": "da provare",
         "kind": "requisito"}]}}
    return {"records": records, "case": "Sempronio schiaffeggia Caio, che cade e muore.",
            "candidates": [{"id": "T1", "kind": "contest_open", "tesi": "Contestare R2", "contesta": ["R2"],
                            "poggia_su": [], "label": "sostenibile"}],
            "assessment": {"article": "584 c.p."}}


ROWS = [{"_article": ("584", "c.p."), "d": {"id": "CP"}, "s": {"name": "584"}}]


def _call(verdicts):
    def call(messages, max_tokens):
        return verdicts
    return call


def _all_for(n, verso="a favore", pertinenza=2, frase=1):
    return {f"P{i}": {"frase": f"P{i}.{frase}", "elemento": "R2", "pertinenza": pertinenza, "verso": verso}
            for i in range(1, n + 1)}


# --- 2. the fetch ---------------------------------------------------------------------------------------------------------------

def test_only_linked_rulings_are_kept_grouped_by_document_and_counted_per_article():
    rows = [_ruling(1), dict(_ruling(1), section="S1b"), _ruling(2, linked=False), _ruling(3)]
    rows[1]["section"] = "S1b"
    session = Session({("CP", "584"): rows}, {"S1": TEXT, "S1b": "", "S3": TEXT})
    record = rulings.build(_ctx(), ROWS, session, call=_call(_all_for(2)))
    art = record["log"]["articles"]["584 c.p."]
    assert art["found"] == 3 and art["linked"] == 2 and art["kept"] == 2
    assert {r["id"] for r in record["kept"]} == {"R1", "R3"}
    assert record["log"]["estremi"] == {"parsed": 2, "unparsed": 0, "section_unmapped": 0}


def test_an_article_whose_row_is_not_read_is_logged_unread():
    record = rulings.build(_ctx(), [], Session({}, {}), call=_call({}))
    assert record["log"]["articles"]["584 c.p."].get("unread") and record["kept"] == []


# --- 3. ranking -----------------------------------------------------------------------------------------------------------------

def test_sezioni_unite_first_then_score_then_date_three_per_article():
    rows = [_ruling(1, abstract="omicidio preterintenzionale, nulla sul punto", when="2025-06-01"),
            _ruling(2, abstract="omicidio preterintenzionale: volontà percuotere prevedibilità morte", when="2025-02-01"),
            _ruling(3, section="U0", abstract="omicidio preterintenzionale", when="2025-01-01"),
            _ruling(4, abstract="omicidio preterintenzionale: volontà percuotere prevedibilità morte", when="2025-09-01"),
            _ruling(5, abstract="omicidio preterintenzionale", when="2025-03-01")]
    session = Session({("CP", "584"): rows}, {f"S{i}": TEXT for i in range(1, 6)})
    record = rulings.build(_ctx(), ROWS, session, call=_call(_all_for(5, pertinenza=1)))
    order = [r["id"] for r in record["kept"]]
    assert order[0] == "R3" and record["kept"][0]["su"]                 # Sezioni Unite first
    assert order[1:] == ["R4", "R2"] and len(order) == rulings.PER_ARTICLE      # then score, then the newer date


def test_the_unrelated_are_dropped_and_eight_per_answer_round_robin():
    ctx = _ctx()
    ctx["records"]["581 c.p."] = {"side": "threat", "requisites": [
        {"id": "R3", "text": "percuotere taluno", "status": "provato", "kind": "requisito"}]}
    ctx["records"]["582 c.p."] = {"side": "threat", "requisites": [
        {"id": "R4", "text": "cagionare una lesione", "status": "provato", "kind": "requisito"}]}
    rows = ROWS + [{"_article": ("581", "c.p."), "d": {"id": "CP"}, "s": {"name": "581"}},
                   {"_article": ("582", "c.p."), "d": {"id": "CP"}, "s": {"name": "582"}}]
    graph = {("CP", n): [_ruling(k + 10 * j) for k in range(1, 6)] for j, n in enumerate(("584", "581", "582"))}
    session = Session(graph, {f"S{k + 10 * j}": TEXT for j in range(3) for k in range(1, 6)})
    verdicts = {f"P{i}": {"frase": f"P{i}.1", "elemento": "", "pertinenza": 0 if i == 1 else 1, "verso": "neutro"}
                for i in range(1, 16)}
    record = rulings.build(ctx, rows, session, call=_call(verdicts))
    assert len(record["kept"]) == rulings.PER_ANSWER
    assert record["log"]["dropped_unrelated"] and record["kept"][0]["article"] == "584 c.p."
    assert [r["article"] for r in record["kept"][:3]] == ["584 c.p.", "581 c.p.", "582 c.p."]


# --- 4. the passage --------------------------------------------------------------------------------------------------------------

def test_the_printed_passage_is_our_sentence_verbatim_and_none_means_estremi_only():
    session = Session({("CP", "584"): [_ruling(1), _ruling(2)]}, {"S1": TEXT, "S2": TEXT})
    verdicts = {"P1": {"frase": "P1.1", "elemento": "R2", "pertinenza": 2, "verso": "a favore"},
                "P2": {"frase": "P2.99", "elemento": "R2", "pertinenza": 2, "verso": "a favore"}}   # an id never sent
    record = rulings.build(_ctx(), ROWS, session, call=_call(verdicts))
    quoted, bare = sorted(record["kept"], key=lambda r: r["passage"] == "")        # P1: the newer, R2
    assert quoted["passage"] == PASSAGE and bare["passage"] == "" and quoted["id"] == "R2"
    text = rulings.section_text(record)
    assert f"- {quoted['estremi']}, passaggio della motivazione: «{PASSAGE}»." in text
    assert f"- {bare['estremi']}." in text and "massima" not in text.lower()
    assert record["log"]["passages"] == {"quoted": 1, "none": 1}


def test_a_failed_call_keeps_the_estremi_without_quotes():
    def boom(messages, max_tokens):
        raise RuntimeError("model down")
    session = Session({("CP", "584"): [_ruling(1)]}, {"S1": TEXT})
    record = rulings.build(_ctx(), ROWS, session, call=boom)
    assert record["kept"] and record["kept"][0]["passage"] == "" and not record["log"]["link_call"]["ok"]
    assert record["orientations"][0]["status"] == "da_verificare"


def test_the_candidate_sentences_are_the_texts_own_near_the_element():
    els = [{"id": "R2", "text": "volontà di percuotere, prevedibilità della morte"}]
    got = rulings.candidate_sentences(TEXT, els, "584 c.p.")
    assert PASSAGE in got and "Le spese seguono la soccombenza." not in got


# --- 5. the orientation -----------------------------------------------------------------------------------------------------------

def test_three_with_a_majority_make_a_prevailing_and_a_minority_line():
    session = Session({("CP", "584"): [_ruling(i) for i in range(1, 4)]}, {f"S{i}": TEXT for i in range(1, 4)})
    verdicts = _all_for(3)
    verdicts["P3"]["verso"] = "contro"
    record = rulings.build(_ctx(), ROWS, session, call=_call(verdicts))
    o = record["orientations"][0]
    assert o["status"] == "prevalente" and o["side"] == "a favore" and (o["for"], o["against"]) == (2, 1)
    line = rulings.orientation_line(o)
    assert "orientamento prevalente: elemento integrato (2 pronunce su 3); orientamento minoritario: elemento non " \
           "integrato (1 su 3)" in line
    assert rulings.section_text(record).rstrip().endswith(rulings.CORPUS_LINE)


def test_fewer_than_three_is_an_orientation_to_verify_with_the_rulings_listed():
    session = Session({("CP", "584"): [_ruling(1), _ruling(2)]}, {"S1": TEXT, "S2": TEXT})
    record = rulings.build(_ctx(), ROWS, session, call=_call(_all_for(2)))
    o = record["orientations"][0]
    assert o["status"] == "da_verificare" and len(o["rulings"]) == 2
    points = []
    text = rulings.section_text(record, points)
    assert "orientamento da verificare (pronunce: Cass. pen., Sez. 5, n. 2/2025 (03.01.2025); Cass. pen., Sez. 5, " \
           "n. 1/2025 (02.01.2025))." in text
    assert points and points[0].startswith("Orientamento sul punto:")


# --- 6. the writer, the calls, the realism --------------------------------------------------------------------------------------

def _record():
    session = Session({("CP", "584"): [_ruling(i) for i in range(1, 4)]}, {f"S{i}": TEXT for i in range(1, 4)})
    return rulings.build(_ctx(), ROWS, session, call=_call(_all_for(3)))


def test_a_ruling_outside_the_registry_leaves_the_prose_and_the_registrys_stay():
    record = _record()
    allowed = rulings.citable(record)
    mine = record["kept"][0]["estremi"].split(" (")[0]
    text = f"Come affermato da {mine}, la volontà basta; v. anche Cass. pen., Sez. 4, n. 4444/2024."
    out = lawyer_method.unretrieved_rulings_out(text, allowed)
    assert mine in out and "4444/2024" not in out


def test_the_thesis_and_objections_calls_receive_the_registry_and_the_writer_rule_flips():
    record = _record()
    ctx = {**_ctx(), "rulings_record": record, "rulings": rulings.citable(record), "posture_text": "",
           "stragiudiziale": ""}
    material = lawyer_method.method_material(ctx)
    assert rulings.REGISTRY_TITLE in material and PASSAGE in material and "ORIENTAMENTI CALCOLATI" in material
    sols = [{"tesi": "Contestare R2", "contesta": ["R2"], "poggia_su": [], "realismo": "medio"}]
    assert rulings.REGISTRY_TITLE in lawyer_method.objections_material(ctx, sols, 0, None)
    assert "Non citare" not in rulings.WRITER_RULE and "estremi" in rulings.WRITER_RULE


def test_the_norms_writer_gets_the_registry_only_in_method_mode(monkeypatch):
    from src.rag import interactive_flow as flow
    record = _record()
    monkeypatch.setattr(flow, "method_mode", lambda state: True)
    assert rulings.REGISTRY_TITLE in flow.rulings_writer_block({"rulings_record": record})
    monkeypatch.setattr(flow, "method_mode", lambda state: False)
    assert flow.rulings_writer_block({"rulings_record": record}) == ""


def test_a_prevailing_line_for_integration_lowers_its_contest_to_basso():
    record = _record()
    assert record["orientations"][0]["status"] == "prevalente"
    plan = {"soluzioni": [{"tesi": "Contestare R2", "contesta": ["R2"], "realismo": "alto"},
                          {"tesi": "Contenimento", "contesta": [], "realismo": "medio"}]}
    log = []
    rulings.cap_realism(plan, record, log)
    assert [t["realismo"] for t in plan["soluzioni"]] == ["basso", "medio"] and log[0]["was"] == "alto"


# --- 7. the lawyer's pronunce di riferimento -----------------------------------------------------------------------------------

LANDMARKS = """# Pronunce di riferimento
<!-- ## art. 41 c.p.
- esempio — mai letto -->
## art. 40 c.p.
- Cass. pen., Sez. Un., n. 30328/2002 (Franzese) — il nesso causale si accerta con alta probabilità logica.
- una riga senza trattino lungo
## art. 5 c.p.
- Corte cost., n. 364/1988 — l'ignoranza inevitabile della legge penale scusa.
"""


def test_the_lawyers_file_is_parsed_and_its_comments_never_read(tmp_path):
    parsed = landmark_rulings.parse(LANDMARKS)
    assert set(parsed) == {"40 c.p.", "5 c.p."} and len(parsed["40 c.p."]) == 1
    assert parsed["5 c.p."][0] == {"estremi": "Corte cost., n. 364/1988",
                                   "principio": "l'ignoranza inevitabile della legge penale scusa."}


def test_the_store_first_then_the_file_and_nothing_when_he_wrote_nothing(tmp_path):
    path = tmp_path / "p.md"
    path.write_text(LANDMARKS, encoding="utf-8")
    store_rows = [{"_article": ("40", "c.p."), "_pronunce": [{"estremi": "dallo store", "principio": "p"}]}]
    assert landmark_rulings.for_label("40 c.p.", store_rows, str(path))[1] == "store"
    items, source = landmark_rulings.for_label("5 c.p.", [], str(path))
    assert source == "file" and items[0]["estremi"].startswith("Corte cost.")
    assert landmark_rulings.for_label("584 c.p.", [], str(path)) == ([], "")
    assert landmark_rulings.parse(Path(landmark_rulings.FILE).read_text(encoding="utf-8")) == {}     # the shipped file
    record = {"kept": [], "orientations": [], "landmarks": landmark_rulings.for_records(["5 c.p."], [], str(path))[0]}
    text = rulings.section_text(record)
    assert "**Pronunce di riferimento indicate dal consulente**" in text and "- art. 5 c.p.: Corte cost." in text
    assert "consulente" not in rulings.section_text({"kept": [], "orientations": [], "landmarks": {}})


def test_the_lawyers_rulings_are_citable(tmp_path):
    record = {"kept": [], "landmarks": {"40 c.p.": [{"estremi": "Cass. pen., Sez. Un., n. 30328/2002", "principio": "p"}]}}
    out = lawyer_method.unretrieved_rulings_out("Secondo Cass. pen., Sez. Un., n. 30328/2002 il nesso…",
                                                rulings.citable(record))
    assert "30328/2002" in out


# --- 8. the retrieval expansions off --------------------------------------------------------------------------------------------

def test_no_case_law_search_feeds_the_v1_pool(monkeypatch, on):
    from src.rag import defensive_generation as dg
    called = []

    class Stop(Exception):
        pass
    monkeypatch.setattr(dg, "_search_case_law", lambda *a, **k: called.append(1) or [])

    def stop(*a, **k):
        raise Stop()
    monkeypatch.setattr(dg, "_search_similar_facts", stop)
    with pytest.raises(Stop):
        dg._select_articles(None, None, "db", "fatti", "", {"route": ""})
    assert called == []
    monkeypatch.setenv(rulings.FLAG, "false")
    with pytest.raises(Stop):
        dg._select_articles(None, None, "db", "fatti", "", {"route": ""})
    assert called == [1]


def test_no_refers_to_neighbour_joins_a_topics_articles(monkeypatch):
    from src.rag import topic_step

    class Links:
        def unit(self, ref, kind):
            return []

        def refers_out(self, ref):
            return [(("581", "c.p."), 3), (("582", "c.p."), 4)]

        def rubriche(self, refs):
            return {}
    ref = ("584", "c.p.")
    monkeypatch.setattr(topic_step, "_book", lambda r, links: "")
    monkeypatch.setattr(topic_step, "_rubriche", lambda links, refs: {})
    with_expansion = topic_step.articles(ref, Links(), {}, None, "omicidio preterintenzionale")
    monkeypatch.setenv(rulings.FLAG, "true")
    without = topic_step.articles(ref, Links(), {}, None, "omicidio preterintenzionale")
    assert ("581", "c.p.") in with_expansion and without == [ref]


def test_edge_expand_never_runs_with_the_record(monkeypatch):
    from src.rag import edge_expand
    monkeypatch.setenv(edge_expand.FLAG, "true")
    assert edge_expand.enabled()
    monkeypatch.setenv(rulings.FLAG, "true")
    assert not edge_expand.enabled()


# --- 9. the nightly, the held-out set, the structure ----------------------------------------------------------------------------

def _answer(section5: str) -> str:
    titles = lawyer_method.titles()
    return "\n\n".join(f"**{i}. {t}**\n\n{section5 if t == lawyer_method.title('orientamenti') else 'x'}"
                       for i, t in enumerate(titles, 1))


def test_the_nightly_fails_a_penal_cell_without_a_quoted_passage_and_a_cc_civil_cell_too(on):
    import importlib.util
    spec = importlib.util.spec_from_file_location("nightly_sim", HERE.parent / "scripts" / "nightly_sim.py")
    nightly = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nightly)
    record = _record()
    good = _answer(rulings.section_text(record))
    bare = _answer(rulings.section_text({"kept": [], "orientations": []}))
    penal = {"family": "penale"}
    assert nightly.ruling_check(good, penal, record)["ok"]
    assert not nightly.ruling_check(bare, penal, {"kept": [], "charge": "584 c.p."})["ok"]
    assert nightly.ruling_check(bare, {"family": "civile"}, {"charge_cc_fattispecie": False}) is None
    assert not nightly.ruling_check(bare, {"family": "civile"}, {"charge_cc_fattispecie": True, "kept": []})["ok"]
    lines = "\n".join(nightly.rulings_lines([{"id": "penale_x", "cell": penal, "answer": good,
                                              "rulings": record["log"], "checks": []}]))
    assert "| penale_x | 584 c.p. | 3 | 3 | 3 | 3 |" in lines and "### §5 of penale_x" in lines and PASSAGE in lines


def test_the_structure_check_reads_the_registry_layout():
    record = _record()
    checks = {c["check"]: c for c in lawyer_method.structure_checks(_answer(rulings.section_text(record)))}
    assert checks["orientations_code_rendered"]["ok"] and checks["no_massima"]["ok"]
    assert checks["markers_collected"]["ok"]           # «orientamento da verificare» belongs to §5
    written = rulings.section_text(record) + "\n- La giurisprudenza ritiene in genere che basti lo schiaffo."
    bad = {c["check"]: c for c in lawyer_method.structure_checks(_answer(written))}
    assert not bad["orientations_code_rendered"]["ok"]


def test_the_held_out_record_carries_the_rulings_used(monkeypatch, tmp_path):
    from src.chatbot import heldout
    record = _record()
    rec = heldout.record("s", "p", {"answer": "x", "kind": "answer", "state": {"rulings_record": record}}, "single_shot")
    assert rec["rulings"]["kept"] == record["kept"] and rec["rulings"]["orientations"] == record["orientations"]
    assert "rulings" not in heldout.record("s", "p", {"answer": "x", "kind": "answer", "state": {}}, "single_shot")


def test_the_record_survives_the_sessions_json():
    record = _record()
    assert json.loads(json.dumps(record))["kept"][0]["date"] == record["kept"][0]["date"]


def test_the_parse_rate_selftest():
    import rulings_parse_rate
    rulings_parse_rate.selftest()


def test_the_dates_stay_dates():
    assert rulings.ruling_date("2025-03-12T00:00:00Z", "2025") == date(2025, 3, 12)
    assert rulings.ruling_date("31/02/2025", "2025") is None and rulings.ruling_date("", "2025") is None


# --- the wiring: the record before the thesis call, the cap before the objections, §5 from the registry -------------------

def test_method_inputs_builds_the_record_first_and_caps_before_the_objections(monkeypatch, on, tmp_path):
    from src.rag import interactive_flow as flow
    record = _record()
    seen = {}
    monkeypatch.setattr(flow, "main_line_topic", lambda topics: None)
    monkeypatch.setattr(flow, "user_named", lambda state: [])
    monkeypatch.setattr(flow, "objections", lambda *a: [])
    monkeypatch.setattr(flow, "unanswered", lambda state: [])
    monkeypatch.setattr(flow, "rulings_record", lambda ctx, rows: record)
    monkeypatch.setattr(lawyer_method, "prepare", lambda *a, **k: {**_ctx(), "rulings": [{"id": "old"}]})

    def samples(messages, call):        # v79: the real method_plan (its realism cap inside), the reply stubbed
        seen["material_has_registry"] = rulings.REGISTRY_TITLE in messages[1].content
        return [("{}", {"valutazioni": {"T1": {"realismo": "alto", "pro": "p"}}})]
    real_plan = lawyer_method.method_plan

    def plan(ctx):
        seen["citable"] = [r["label"] for r in ctx["rulings"]]
        return real_plan(ctx)

    def objections_plan(ctx, mplan):
        seen["realismo_at_objections"] = mplan["soluzioni"][0]["realismo"]
        return mplan
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(lawyer_method, "thesis_candidates", lambda ctx: [dict(c, argued="model")
                                                                         for c in _ctx()["candidates"]])
    monkeypatch.setattr(lawyer_method, "_samples", samples)
    monkeypatch.setattr(lawyer_method, "bind_arguments", lambda sols, known, call=None: {})
    monkeypatch.setattr(lawyer_method, "method_plan", plan)
    monkeypatch.setattr(lawyer_method, "objections_plan", objections_plan)
    state = {"records": _ctx()["records"]}
    ctx, mplan = flow.method_inputs(state, {}, {}, [], [], ROWS, "il caso")
    assert seen["material_has_registry"] and seen["citable"] == [r["estremi"] for r in record["kept"]]
    assert seen["realismo_at_objections"] == "basso" and state["rulings_record"] is record
    text = lawyer_method.orientations_text(ctx, {"orientamenti": [{"punto": "x", "pronuncia": ""}]}, ctx["rulings"])
    assert text == rulings.section_text(record) and "x: orientamento da verificare" not in text


def test_without_the_flag_nothing_changes(monkeypatch):
    from src.rag import interactive_flow as flow
    monkeypatch.setattr(flow, "main_line_topic", lambda topics: None)
    monkeypatch.setattr(flow, "user_named", lambda state: [])
    monkeypatch.setattr(flow, "objections", lambda *a: [])
    monkeypatch.setattr(flow, "unanswered", lambda state: [])
    monkeypatch.setattr(flow, "rulings_record", lambda ctx, rows: pytest.fail("the record built with the flag off"))
    monkeypatch.setattr(lawyer_method, "prepare", lambda *a, **k: _ctx())
    monkeypatch.setattr(lawyer_method, "method_plan", lambda ctx: {"soluzioni": [], "orientamenti": []})
    monkeypatch.setattr(lawyer_method, "objections_plan", lambda ctx, mplan: mplan)
    state = {"records": _ctx()["records"]}
    ctx, _ = flow.method_inputs(state, {}, {}, [], [], ROWS, "il caso")
    assert "rulings_record" not in state and lawyer_method.registry(ctx) == ""
    assert lawyer_method.orientations_text(ctx, {"orientamenti": []}, []) == lawyer_method.NO_POINTS_LINE


# --- six graded per article for the orientation, three printed -------------------------------------------------------------

def test_the_orientation_counts_six_graded_while_three_are_printed():
    session = Session({("CP", "584"): [_ruling(i) for i in range(1, 8)]}, {f"S{i}": TEXT for i in range(1, 8)})
    verdicts = _all_for(6)
    verdicts["P5"]["verso"] = "contro"
    verdicts["P6"]["pertinenza"] = 0                                   # graded unrelated: neither printed nor counted
    record = rulings.build(_ctx(), ROWS, session, call=_call(verdicts))
    art = record["log"]["articles"]["584 c.p."]
    assert (art["found"], art["graded"], art["kept"]) == (7, 5, 3) and len(record["kept"]) == 3
    o = record["orientations"][0]
    assert o["status"] == "prevalente" and (o["for"], o["against"]) == (4, 1)
    assert "elemento integrato (4 pronunce su 5); orientamento minoritario: elemento non integrato (1 su 5)" in \
        rulings.orientation_line(o)


def test_a_graded_ruling_listed_but_not_printed_is_citable():
    session = Session({("CP", "584"): [_ruling(i) for i in range(1, 6)]}, {f"S{i}": TEXT for i in range(1, 6)})
    verdicts = _all_for(5)
    for k in ("P2", "P4"):
        verdicts[k]["verso"] = "contro"
    verdicts["P5"]["verso"] = "neutro"                                 # 2 for, 2 against: no majority
    record = rulings.build(_ctx(), ROWS, session, call=_call(verdicts))
    o = record["orientations"][0]
    assert o["status"] == "da_verificare" and len(o["rulings"]) == 5 and len(record["kept"]) == 3
    unprinted = [x for x in o["rulings"] if x not in {r["estremi"] for r in record["kept"]}]
    assert len(unprinted) == 2
    labels = [c["label"] for c in rulings.citable(record)]
    assert all(x in labels for x in o["rulings"]) and len(labels) == len(set(labels))
