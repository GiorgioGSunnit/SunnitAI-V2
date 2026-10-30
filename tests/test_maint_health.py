"""scripts/maint_health.py: the maintenance agent's job 1, the nightly read-only health report (27 Oct)."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("maint_health", HERE.parent / "scripts" / "maint_health.py")
mh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mh)


def test_article_of_and_gaps():
    assert mh.article_of("1490.0.0") == "1490" and mh.article_of("1117.3.0_2") == "1117"
    assert mh.article_of("337-ter") == "337-ter" and mh.article_of("corte_cassazione.8") is None
    assert mh.numbering_gaps(["1", "2", "4", "4-bis", "7"]) == [3, 5, 6] and mh.numbering_gaps([]) == []


def test_date_only_articles_need_text_elsewhere():
    sections = [("128.0.0", 30, True), ("128", 4004, False),            # date only, text in the bare Section
                ("105.0.0", 29, True), ("105.1.0", 321, False),         # date only, text in a comma
                ("9.0.0", 28, True),                                    # date only, nothing else: not counted
                ("1490.0.0", 321, False), ("1490.1.0", 200, False)]     # real text in N.0.0
    assert mh.date_only_articles(sections) == ["105", "128"]


def test_suspected_duplicates():
    docs = [{"name": "Codice del Consumo 2026", "hash": "a"}, {"name": "Codice del Consumo 2025", "hash": "b"},
            {"name": "Testo Unico della Finanza 2026", "hash": "c"}, {"name": "Testo unico della finanza 2026", "hash": "d"},
            {"name": "Copia", "hash": "c"}, {"name": "Unico", "hash": None}]
    found = mh.suspected_duplicates(docs)
    assert found["same hash"] == [["Testo Unico della Finanza 2026", "Copia"]]
    assert found["same name"] == [["Testo Unico della Finanza 2026", "Testo unico della finanza 2026"]]
    assert found["same name but the year"] == [["Codice del Consumo 2026", "Codice del Consumo 2025"]]
    assert found["same ruling file"] == []


def test_rulings_compared_by_file_never_by_title():
    """The first report (28 Oct): 496 false alarms from generic or anonymised ruling titles."""
    r = lambda i, name, file, h=None: {"id": i, "name": name, "file": file, "hash": h, "type": "interpretation"}  # noqa: E731
    docs = [r("1", "Sentenza della Corte Suprema di Cassazione", "snpen@s10@a2026@n00190@tS.clean.pdf"),
            r("2", "Sentenza della Corte Suprema di Cassazione", "snpen@s10@a2026@n00191@tS.clean.pdf"),
            r("3", "Ordinanza sul ricorso 17360-2020", "snciv_s30_a2026_n00001_tO.pdf"),
            r("4", "Ordinanza sul ricorso 17360-2022", "snciv@s30@a2026@n00001@tO.clean.pdf", "h"),
            r("5", "Sentenza sul ricorso proposto da Rossi", "", "h")]
    found = mh.suspected_duplicates(docs)
    assert found["same name"] == [] and found["same name but the year"] == []
    assert found["same ruling file"] == [["3", "4"]]                                # the same file loaded twice
    assert found["same hash"] == [["snciv@s30@a2026@n00001@tO", "5"]]              # never the ruling's title


def test_short_first_pieces_are_not_dates():
    sections = [("564.0.0", 3, False), ("564.1.0", 300, False),     # "1." in N.0.0, the text in the comma
                ("128.0.0", 30, True), ("128", 4004, False),        # a date: counted elsewhere
                ("9.0.0", 10, False),                               # nothing else: not counted
                ("1490.0.0", 321, False)]
    assert mh.short_first_pieces(sections) == ["564"]


def test_closures_are_left_out_of_the_cards_to_do(tmp_path):
    links = tmp_path / "links.jsonl"
    links.write_text("\n".join(json.dumps(x) for x in [{"id": "c9", "branch": "civile", "outcome": "estinto"},
                                                        {"id": "c1", "branch": "civile", "outcome": "rigetto"},
                                                        {"id": "p1", "branch": "penale", "outcome": "estinto"}]),
                     encoding="utf-8")
    assert mh.closures(str(links)) == {"c9"} and mh.closures(str(tmp_path / "none.jsonl")) == set()


def test_branch_of():
    assert mh.branch_of("snpen@s10@a2026@n00190@tS.clean.pdf", "") == "penale"
    assert mh.branch_of("", "[CIVILE] snciv_s30_a2026_n12345_tO") == "civile"
    assert mh.branch_of("altro.pdf", "") == "unknown"


NUMBERS = {"sections_no_v2_with_text": 10, "sections_empty": 3, "docs_untyped": 2, "sections_detached": 0,
           "date_only_articles": 1100, "numbering_gaps": 40, "numbering_gaps_new": 0, "rulings_without_cites": 4254,
           "queue_waiting": 13000, "queue_failed": 5, "duplicates": 1, "rulings_without_card": None}


def test_summary_line_with_deltas_and_cards():
    line = mh.summary_line(NUMBERS, None)
    assert line.startswith("v2 missing 10 · empty sections 3 · untyped docs 2") and "card" not in line
    assert "suspected duplicates 1 (not touched)" in line
    prev = {**NUMBERS, "sections_no_v2_with_text": 4, "rulings_without_cites": 4300}
    line = mh.summary_line({**NUMBERS, "rulings_without_card": 5939}, {**prev, "rulings_without_card": 6000})
    assert "v2 missing 10 (+6)" in line and "rulings without CITES 4,254 (-46)" in line
    assert "rulings without card 5,939 (-61)" in line


GRAPH = {"documents": 6, "docs_by_type": {"interpretation": 3, "primary": 1, "(none)": 2},
         "docs_untyped_ids": ["id-a", "id-b"], "docs_unknown_type": {},
         "sections_by_type": {"interpretation": 30, "primary": 100}, "sections_empty_by_type": {"primary": 1},
         "sections_no_v2_by_type": {"primary": 2}, "sections_no_v2_with_text_by_type": {"primary": 1},
         "sections_detached": 0, "date_only_by_doc": {"Codice del Consumo 2026": 205},
         "short_first_by_doc": {"Codice di procedura penale 2026": 138},
         "numbering_gaps_by_code": {"Codice Civile 2026": [3, 5]},
         "rulings_by_branch": {"penale": {"rulings": 2, "without_cites": 1}, "civile": {"rulings": 1, "without_cites": 0}},
         "ruling_ids_by_branch": {"penale": ["p1", "p2"], "civile": ["c1"]},
         "duplicates": {"same hash": [], "same name": [], "same name but the year": []}}
QUEUE = {"waiting": 2, "by_kind": {"penale": 2}, "failed": 0, "service": "active", "queue_dir_found": True}


def test_build_report_first_night_and_new_gaps(tmp_path):
    md, state, line = mh.build_report(GRAPH, QUEUE, [{"log": "/var/log/x.log", "found": False}], (7, {"not_found": 7}),
                                      {"p1", "c1"}, None, 12.0, by_design={"c9"})
    assert "First report: no comparison." in md and "- Codice del Consumo 2026: 205" in md
    assert "| penale | 2 | 1 | 1 | 0 | 1 |" in md and "untyped docs 2" in line and "rulings without card 1" in line
    assert "- Codice di procedura penale 2026: 138" in md and "Untyped documents (ids" in md and "id-a; id-b" in md
    assert "| primary | 1 | 100 | 1 | 1 | 1 |" in md                    # 2 without the vector: 1 with text, 1 blank
    assert "CITES unresolved citations (linker log): 7 -- not_found 7" in md and "/var/log/x.log: not found" in md
    (tmp_path / "state_20261026.json").write_text(json.dumps(state), encoding="utf-8")
    prev = mh.previous_state(str(tmp_path), "20261027")
    gaps = {**GRAPH, "numbering_gaps_by_code": {"Codice Civile 2026": [3, 5, 9]}}
    md, state, line = mh.build_report(gaps, QUEUE, [], (None, {}), None, prev, 3.0)
    assert "CITES unresolved citations (linker log): log not found on this machine" in md
    assert "**NEW since the last report: 9**" in md and "numbering gaps 3 (new 1)" in line
    assert "no card file readable here" in md
    assert mh.previous_state(str(tmp_path), "20261026") is None                     # today's own state is not "previous"


def test_missing_articles_counts_only_citations_with_no_article(tmp_path):
    log = tmp_path / "unresolved.jsonl"
    rows = [{"citation": "410-bis c.p.p.", "reason": "article_not_in_code"}] * 3 + [
        {"citation": "16 Cost.", "reason": "article_not_in_code"}, {"citation": "5 c.p.a.", "reason": "code_not_on_be2"}]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n", encoding="utf-8")
    assert mh.missing_articles(str(log)) == [("410-bis c.p.p.", 3), ("16 Cost.", 1)]
    assert mh.missing_articles(str(tmp_path / "none.jsonl")) is None
    md, state, _ = mh.build_report(GRAPH, QUEUE, [], (5, {"article_not_in_code": 4}), None, None, 1.0,
                                   missing=mh.missing_articles(str(log)))
    assert "## Articles the rulings cite that our codes lack" in md and "- 410-bis c.p.p.: 3" in md
    assert state["missing_articles"] == {"410-bis c.p.p.": 3, "16 Cost.": 1}
    md, _, _ = mh.build_report(GRAPH, QUEUE, [], (None, {}), None, None, 1.0)
    assert "cite that our codes lack" not in md                                    # no linker log here: no section


class _Session:
    """Fake read session answering each of graph_checks' queries."""

    def run(self, query, **params):
        if query.startswith("MATCH (d:Document) RETURN d.id"):
            data = [{"id": "cc", "name": "Codice Civile 2026", "hash": "h1", "type": "primary", "file": ""},
                    {"id": "r1", "name": "[PENALE] x", "hash": "h2", "type": "interpretation",
                     "file": "snpen@s10@a2026@n00190@tS.clean.pdf"},
                    {"id": "r2", "name": "[CIVILE] y", "hash": "h3", "type": "interpretation",
                     "file": "snciv_s30_a2026_n00001_tO.pdf"},
                    {"id": "u", "name": "senza tipo", "hash": "h4", "type": None, "file": ""}]
        elif "embedding_v2 IS NULL" in query:
            data = [{"t": "primary", "n": 5, "empty": 1, "no_v2": 2, "no_v2_text": 1}] if "cc" in params["ids"] else []
        elif "NOT ()-[:CONTAINS|PART_OF]->(s)" in query:
            return type("R", (), {"single": lambda self_: {"n": 4}})()
        elif "left(coalesce(s.plain_text" in query:
            data = [{"name": "1.0.0", "n": 100, "head": "Il primo"}, {"name": "3.0.0", "n": 30,
                                                                       "head": "In vigore dal 1 gennaio 2020"},
                    {"name": "3.1.0", "n": 200, "head": "Testo"}]
        elif "[:CITES]" in query:
            data = [{"id": "r1"}]
        else:
            raise AssertionError(query)
        return type("R", (), {"data": lambda self_: data})()


def test_graph_checks_on_a_fake_graph():
    g = mh.graph_checks(_Session(), batch=2, pause=0)
    assert g["docs_by_type"] == {"primary": 1, "interpretation": 2, "(none)": 1} and g["docs_untyped_ids"] == ["u"]
    assert g["sections_no_v2_with_text_by_type"] == {"primary": 1} and g["sections_detached"] == 4
    assert g["date_only_by_doc"] == {"Codice Civile 2026": 1} and g["numbering_gaps_by_code"] == {"Codice Civile 2026": [2]}
    assert g["rulings_by_branch"] == {"penale": {"rulings": 1, "without_cites": 0}, "civile": {"rulings": 1, "without_cites": 1}}


def test_graph_checks_can_skip_the_heavy_stages():
    class Counting(_Session):
        def __init__(self):
            self.queries = []

        def run(self, query, **params):
            self.queries.append(query)
            return super().run(query, **params)
    session = Counting()
    g = mh.graph_checks(session, batch=2, pause=0, skip=("sections", "detached"))
    assert not any("embedding_v2" in q or "NOT ()-[:CONTAINS" in q for q in session.queries)
    assert g["skipped"] == ["detached", "sections"] and g["sections_detached"] == 0
    md, state, line = mh.build_report(g, QUEUE, [], (0, {}), None, None, 1.0)
    assert line.endswith("skipped: detached, sections")


def test_only_a_python_process_running_the_job_counts_as_busy():
    lines = ["  101 grep --color=auto backfill_embeddings_v2",
             "  102 watch -n 60 ps aux | grep backfill_embeddings_v2",
             "  103 less /opt/v2tools/backfill_embeddings_v2.py",
             "  104 /usr/bin/python3 /opt/v2tools/maint_health.py --env /opt/chatbot/.env",
             "  105 /bin/sh -c cd /opt/v2tools && /opt/chatbot/venv/bin/python link_ruling_citations.py --write"]
    assert mh.running_job(lines, me=104) is None
    assert mh.running_job(lines + ["  106 /opt/chatbot/venv/bin/python link_ruling_citations.py --write --out x"],
                          me=104) == "link_ruling_citations (pid 106)"
    assert mh.running_job(["  7 python3 -m scripts.backfill_embeddings_v2 --batch 64"], me=1) == \
        "backfill_embeddings_v2 (pid 7)"
