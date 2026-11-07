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
    assert g["rulings_by_year"] == {"2026": 2}


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


def test_year_of_reads_the_italgiure_file_name():
    assert mh.year_of("snpen@s10@a2025@n00001@tS.clean.pdf", "") == "2025"
    assert mh.year_of("", "[CIVILE] snciv s10 a2026 n00016 tO.clean") is None     # spaces: not the file name form
    assert mh.year_of("snciv_s30_a2026_n00001_tO.pdf", "") == "2026"
    assert mh.year_of("contratto.pdf", "Contratto di appalto") is None


def test_batch_status_counts_per_batch_and_today(tmp_path):
    manifest = tmp_path / "upload_batches.txt"
    manifest.write_text("2026-11-04 pdf2025 snpen@s10@a2025@n00002@tS.clean.pdf\n"
                        "2026-11-04 pdf2025 snpen@s10@a2025@n00003@tS.clean.pdf\n"
                        "2026-11-05 pdf2025 snpen@s10@a2025@n00005@tS.clean.pdf\n"
                        "2026-11-05 civ2025 snciv_s10_a2025_n00001_tO.pdf\n"
                        "garbage line\n", encoding="utf-8")
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "a.pdf").write_bytes(b"x")
    (staging / "a.name").write_bytes(b"x")
    b = mh.batch_status(str(manifest), str(staging), "2026-11-05")
    assert b["found"] and b["fed_today"] == 2 and b["staging_left"] == 1
    assert b["batches"]["pdf2025"] == {"fed": 3, "fed_today": 1, "first": "2026-11-04", "last": "2026-11-05"}
    assert b["batches"]["civ2025"]["fed"] == 1
    none = mh.batch_status(str(tmp_path / "missing.txt"), str(tmp_path / "nowhere"), "2026-11-05")
    assert none == {"found": False, "batches": {}, "staging_left": None, "fed_today": 0}


def test_report_shows_batches_and_rulings_by_year_with_the_nights_change():
    g = {**GRAPH, "rulings_by_year": {"2026": 3, "2025": 2}}
    batches = {"found": True, "fed_today": 60, "staging_left": 4947,
               "batches": {"pdf2025": {"fed": 360, "fed_today": 60, "first": "2026-11-04", "last": "2026-11-05"}}}
    prev = {"numbers": {"rulings_by_year": {"2026": 3, "2025": 1}}}
    md, state, line = mh.build_report(g, QUEUE, [], (0, {}), None, prev, 1.0, batches=batches)
    assert "| pdf2025 | 360 | 60 | 2026-11-04 | 2026-11-05 |" in md and "still staged for the feeder: 4,947" in md
    assert "| 2025 | 2 | +1 |" in md and "| 2026 | 3 | +0 |" in md
    assert line.endswith("fed to the queue today 60 · rulings 2026 3, 2025 2 (+1)")
    assert state["batches"] == batches["batches"] and state["numbers"]["rulings_by_year"] == {"2026": 3, "2025": 2}
    md, _, line = mh.build_report(GRAPH, QUEUE, [], (0, {}), None, None, 1.0)      # no manifest, no years: still fine
    assert "No upload manifest here" in md and "fed to the queue" not in line


def test_queue_status_counts_documents_not_their_sidecars(tmp_path):
    queue, failed = tmp_path / "upload", tmp_path / "failed"
    (queue / "sub").mkdir(parents=True)
    failed.mkdir()
    (queue / "snpen@s10@a2025@n00001@tS.clean.pdf").write_bytes(b"x")
    (queue / "snpen@s10@a2025@n00001@tS.clean.name").write_bytes(b"x")
    (queue / "sub" / "snciv_s10_a2025_n00001_tO.pdf").write_bytes(b"x")
    for stem in ("snpen@s10@a2025@n00007@tS.clean", "snpen@s10@a2025@n00009@tS.clean"):
        (failed / f"{stem}.pdf").write_bytes(b"x")
        (failed / f"{stem}.name").write_bytes(b"x")
    (failed / "altro.docx").write_bytes(b"x")
    q = mh.queue_status(str(queue), str(failed))
    assert q["waiting"] == 2 and q["by_kind"] == {"penale": 1, "civile": 1} and q["failed"] == 3
    none = mh.queue_status(str(tmp_path / "none"), str(tmp_path / "none"))
    assert {k: none[k] for k in ("waiting", "by_kind", "failed", "service", "queue_dir_found")} == {
        "waiting": 0, "by_kind": {}, "failed": 0, "service": q["service"], "queue_dir_found": False}
    assert none["per_queue"] == {"none": None} and none["services"] == {"sunnitai-ingest": q["service"]}


class _Out:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr


JOURNAL = """2026-11-05T13:14:38+0100 be2 python3[1]: 2026-11-05 13:14:38,250 INFO —   [4/4] done 2.3s — 65 nodes, 85 rels
2026-11-05T13:15:55+0100 be2 python3[1]: 2026-11-05 13:15:55,377 WARNING — Chunk 4/6 structured attempt 1/2 timed out after 15s
2026-11-05T13:15:56+0100 be2 python3[1]: 2026-11-05 13:15:56,000 WARNING — Chunk 4/6 structured attempt 2/2 failed: fallback to plain text
2026-11-05T13:15:57+0100 be2 python3[1]: 2026-11-05 13:15:57,885 INFO —   [1/4] done 18.0s
2026-11-05T13:16:07+0100 be2 python3[1]: 2026-11-05 13:16:07,118 INFO —   [4/4] done 2.1s — 16 nodes, 16 rels
2026-11-05T11:00:01+0100 be2 python3[1]: 2026-11-05 11:00:01,000 ERROR — LLM call failed: 404 The model `qwen2.5-32b-awq` does not exist
2026-11-05T11:00:02+0100 be2 python3[1]: 2026-11-05 11:00:02,000 ERROR — LLM call failed: 404 The model `qwen2.5-32b-awq` does not exist
2026-11-05T11:00:03+0100 be2 python3[1]: 2026-11-05 11:00:03,000 WARNING — Moved to failed: snciv@sL0@a2026@n00001@tO.clean.pdf
2026-11-05T11:00:04+0100 be2 python3[1]: 2026-11-05 11:00:04,000 INFO — [SKIP ANALYSIS] INGEST_SKIP_ANALYSIS set: 41 chunks kept raw, no LLM analysis
"""


def test_ingest_journal_counts_documents_written_and_the_model_the_server_lacks():
    calls = []

    def runner(cmd, **kw):
        calls.append(cmd)
        return _Out(JOURNAL)
    j = mh.ingest_journal("24 hours ago", runner)
    assert calls[0][:4] == ["journalctl", "-u", "sunnitai-ingest", "--since"] and "24 hours ago" in calls[0]
    assert j["found"] and j["written"] == 2 and j["last_written"] == "2026-11-05T13:16:07"
    assert j["errors"] == 2 and j["model_missing"] == {"qwen2.5-32b-awq": 2}
    assert j["chunk_timeouts"] == 1 and j["chunk_fallbacks"] == 1 and j["moved_to_failed"] == 1
    assert j["no_analysis"] == 1                                   # the cheap path, counted per document
    assert mh.ingest_journal(runner=lambda *a, **k: _Out("", 1, "No journal files were found.")) == {
        "found": False, "why": "No journal files were found."}

    def no_journalctl(*a, **k):
        raise FileNotFoundError("journalctl")
    assert mh.ingest_journal(runner=no_journalctl) == {"found": False, "why": "FileNotFoundError"}


def test_report_flags_the_missing_model_in_the_summary_and_the_page():
    ingest = mh.ingest_journal(runner=lambda *a, **k: _Out(JOURNAL))
    md, state, line = mh.build_report(GRAPH, QUEUE, [], (0, {}), None, None, 1.0, ingest=ingest)
    assert ("queue 2 waiting / 0 failed · ingest 24 h: 2 written (1 without analysis), 2 LLM model errors, "
            "1 chunk fallbacks · suspected duplicates") in line
    assert "2 documents written (1 without analysis: INGEST_SKIP_ANALYSIS)" in md
    assert "**1 chunk analyses fell back to plain text**" in md
    assert "- ingest journal (last 24 hours ago): 2 documents written (1 without analysis: INGEST_SKIP_ANALYSIS), last at 2026-11-05T13:16:07; errors 2; " \
           "chunk analyses timed out 1, fell back to plain text 1; moved to failed 1" in md
    assert "**LLM model not found: qwen2.5-32b-awq (2 calls)**" in md and state["ingest"] == ingest
    md, _, line = mh.build_report(GRAPH, QUEUE, [], (0, {}), None, None, 1.0)             # journal not read: still fine
    assert "ingest journal: not readable here (not read)" in md and "ingest 24 h" not in line
    md, _, line = mh.build_report(GRAPH, QUEUE, [], (0, {}), None, None, 1.0,
                                  ingest={"found": True, "since": "24 hours ago", "written": 0, "last_written": None,
                                          "errors": 0, "model_missing": {}, "chunk_timeouts": 0, "chunk_fallbacks": 0,
                                          "moved_to_failed": 0})
    assert "ingest 24 h: 0 written, 0 LLM model errors" in line and "LLM model not found" not in md



def test_queue_status_counts_several_queue_folders(tmp_path):
    q1, q2 = tmp_path / "document_upload", tmp_path / "document_upload_2"
    q1.mkdir(), q2.mkdir()
    for folder, names in ((q1, ("snpen_a.pdf", "snpen_b.pdf")), (q2, ("snciv_c.pdf",))):
        for n in names:
            (folder / n).write_bytes(b"x")
    q = mh.queue_status(f"{q1},{q2}", str(tmp_path / "failed"), units=("sunnitai-ingest", "sunnitai-ingest-2"))
    assert q["waiting"] == 3 and q["per_queue"] == {"document_upload": 2, "document_upload_2": 1}
    assert set(q["services"]) == {"sunnitai-ingest", "sunnitai-ingest-2"} and q["queue_dir_found"]


def test_ingest_journal_reads_every_unit_it_is_given():
    seen = {}

    def runner(cmd, **kw):
        seen["cmd"] = cmd
        return _Out(JOURNAL)
    j = mh.ingest_journal(runner=runner, units=("sunnitai-ingest", "sunnitai-ingest-2"))
    assert seen["cmd"][:5] == ["journalctl", "-u", "sunnitai-ingest", "-u", "sunnitai-ingest-2"]
    assert j["found"] and j["written"] == 2 and j["chunk_fallbacks"] == 1


def test_feeder_state_reads_the_feeder_json_or_says_it_is_missing(tmp_path):
    assert mh.feeder_state(str(tmp_path / "none.json")) == {"found": False}
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert mh.feeder_state(str(tmp_path / "bad.json")) == {"found": False, "why": "JSONDecodeError"}
    (tmp_path / "feed_state.json").write_text(
        '{"time": "2026-11-10T08:00:00Z", "batch": "cass_batch1", "paused": false, "fed": 40, "left": 34960, '
        '"queues": {"/opt/sunnitai-be/document_upload": 100}, "already_fed": 2, "failed": 11}', encoding="utf-8")
    st = mh.feeder_state(str(tmp_path / "feed_state.json"))
    assert st["found"] and st["left"] == 34960 and st["fed"] == 40 and st["already_fed"] == 2


def test_disk_status_gives_free_space_and_the_store_size(tmp_path):
    assert mh.disk_status(str(tmp_path / "none")) == {"found": False}
    d = mh.disk_status(str(tmp_path), runner=lambda cmd, **kw: _Out(f"{103 * 2 ** 30}\t/var/lib/neo4j/data\n"))
    assert d["found"] and d["free_gb"] > 0 and d["total_gb"] >= d["free_gb"] and d["store_gb"] == 103.0   # GiB, as du -sh
    d = mh.disk_status(str(tmp_path), runner=lambda cmd, **kw: (_ for _ in ()).throw(TimeoutError()))
    assert d["found"] and d["store_gb"] is None                               # du too slow: free space still there


def test_report_shows_disk_feeder_and_flags_a_stalled_ingest_and_a_full_disk():
    g = GRAPH
    queue = {"waiting": 150, "by_kind": {"penale": 150}, "failed": 0, "service": "active", "queue_dir_found": True,
             "per_queue": {"document_upload": 100, "document_upload_2": 50},
             "services": {"sunnitai-ingest": "active", "sunnitai-ingest-2": "failed"}}
    ingest = {"found": True, "since": "24 hours ago", "written": 0, "last_written": None, "errors": 0,
              "model_missing": {}, "chunk_timeouts": 0, "chunk_fallbacks": 0, "moved_to_failed": 0}
    feeder = {"found": True, "time": "2026-11-10T08:00:00Z", "batch": "cass_batch1", "paused": True, "fed": 0,
              "left": 34960, "queues": {}, "already_fed": 0, "failed": 11}
    disk = {"found": True, "free_gb": 31.5, "total_gb": 334.0, "store_gb": 140.2}
    md, state, line = mh.build_report(g, queue, [], (None, {}), None, None, 1.0, ingest=ingest, feeder=feeder,
                                      disk=disk, min_free_gb=40)
    assert "ingest 24 h: 0 written, 0 LLM model errors, 0 chunk fallbacks · disk free 32 GB (store 140 GB)" in line
    assert line.endswith("feeder: 34,960 left PAUSED")
    assert "(document_upload 100, document_upload_2 50); failed 0; service sunnitai-ingest active, service sunnitai-ingest-2 failed" in md
    assert "**nothing written in the last 24 hours ago while 150 files wait**" in md
    assert "**disk below 40 GB free**" in md
    assert "- feeder (maint_feed.py, last run 2026-11-10T08:00:00Z, batch cass_batch1): PAUSED; fed 0 at that run, 34960 left" in md
    assert state["feeder"]["left"] == 34960 and state["disk"]["free_gb"] == 31.5
    md_no_feeder, _, line_no_feeder = mh.build_report(g, queue, [], (None, {}), None, None, 1.0, ingest=ingest,
                                                      feeder={"found": False}, disk=disk, min_free_gb=40)
    assert "- feeder: no state file -- maint_feed.py has not run yet" in md_no_feeder and "feeder:" not in line_no_feeder
    # a healthy night: no bold flags
    ingest["written"], disk["free_gb"], feeder["paused"] = 1400, 160.0, False
    md, _, line = mh.build_report(g, queue, [], (None, {}), None, None, 1.0, ingest=ingest, feeder=feeder, disk=disk)
    assert "**" not in md.split("## Ingestion queue")[1].split("## Upload batches")[0]
    assert line.endswith("feeder: 34,960 left")
