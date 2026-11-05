"""EDGE_EXPAND: the anchors in order, the per-anchor picks (2 nearest in the Sezione, 2 specific REFERS_TO each way),
the cap, the tagging as one retrieval signal, the grading of the new and the never-graded rows, the reach log, and
the CSV-backed lookup on scripts/article_links.py's files. No model, no database."""
import csv
import os

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

import pytest  # noqa: E402

import src.rag.defensive_generation as dg  # noqa: E402
from src.rag import edge_expand as ee  # noqa: E402
import src.rag.plan_then_write as ptw  # noqa: E402


@pytest.fixture
def events(monkeypatch):
    seen = []
    monkeypatch.setattr(dg, "vlog", lambda name, data=None, **k: seen.append((name, data)))
    return seen


class Lookup:
    """h5 (593 c.p.p. cites 580) and h3 (1158 in 1159's Sezione) in small; inbound: how many articles cite each."""
    OUT = {("593", "c.p.p."): {("580", "c.p.p.")}}
    SEZIONE = {("1159", "c.c."): {("1158", "c.c."), ("1159", "c.c."), ("1160", "c.c.")}}
    INBOUND = {}

    def refers_out(self, ref):
        return self.OUT.get(tuple(ref), set())

    def refers_in(self, ref):
        return {k for k, targets in self.OUT.items() if tuple(ref) in targets}

    def sezione(self, ref):
        return self.SEZIONE.get(tuple(ref), set())

    def inbound(self, k):
        return self.INBOUND.get(tuple(k), sum(1 for targets in self.OUT.values() if tuple(k) in targets))


def row(n, code, grade=None, signals=()):
    return {"_article": (n, code), "_grade": grade, "_signals": list(signals), "d": {"id": f"doc-{code}"},
            "s": {"name": f"{n}.0.0", "plain_text": f"Testo {n}."}}


def test_the_anchors_are_the_prompt_s_articles_and_the_grade_3_rows_by_grade_then_signals():
    rows = [row("580", "c.p.p.", 1, ["current", "frame"]), row("1227", "c.c.", 3), row("1159", "c.c.", 1),
            row("1218", "c.c.", None), row("581", "c.p.p.", 1, ["current"])]
    prompt = "[art. 580 c.p.p.]\n[art. 1159 c.c.]\n[art. 581 c.p.p.]"
    got = [(f"{a['ref'][0]} {a['ref'][1]}", a["why"]) for a in ee.anchors(rows, prompt)]
    assert got == [("1227 c.c.", "grade 3"), ("580 c.p.p.", "in the prompt"), ("581 c.p.p.", "in the prompt"),
                   ("1159 c.c.", "in the prompt")]
    exempt = {**row("2043", "c.c."), "_heading_match": True}
    assert ee.anchors([row("580", "c.p.p.", 1), exempt], "[art. 580 c.p.p.]\n[art. 2043 c.c.]")[0]["ref"] == \
        ("2043", "c.c.")                                                  # exempt from the check: as grade 3
    assert [a["ref"][0] for a in ee.anchors(rows, prompt, min_grade=2)] == ["1227", "580", "581"]   # P4


class Many(Lookup):
    """1159's Sezione 1150-1165; 580 cites 590-599 (590 cited by 30 articles, 591-599 by fewer), 593 and 600 cite 580."""
    OUT = {("580", "c.p.p."): {(str(n), "c.p.p.") for n in range(590, 600)},
           ("593", "c.p.p."): {("580", "c.p.p.")}, ("600", "c.p.p."): {("580", "c.p.p."), ("581", "c.p.p.")}}
    SEZIONE = {("1159", "c.c."): {(str(n), "c.c.") for n in range(1150, 1166)}}
    INBOUND = {("590", "c.p.p."): 30, ("591", "c.p.p."): 1, ("592", "c.p.p."): 2, **{(str(n), "c.p.p."): 5
                                                                                     for n in range(593, 600)}}


def test_per_anchor_two_nearest_in_the_sezione_two_specific_citations_and_two_citing_it(monkeypatch):
    monkeypatch.setattr(ee, "REFERS_SOURCES", True)
    ranked, stopped = ee.neighbours([("1159", "c.c."), ("580", "c.p.p."), ("581", "c.p.p.")], Many())
    by = {f"{e['key'][0]} {e['key'][1]}": e for e in ranked}
    assert stopped is None and [e["rank"] for e in ranked] == list(range(1, len(ranked) + 1))
    assert [e["key"][0] for e in ranked[:2]] == ["1158", "1160"]                   # the nearest two, 1159 first
    assert by["591 c.p.p."]["channels"] == ["edge:refers_to"] and "592 c.p.p." in by and "590 c.p.p." not in by
    assert by["593 c.p.p."]["channels"] == ["edge:cited_by"]                        # h5: 593 cites 580
    assert by["600 c.p.p."]["anchors"] == ["580 c.p.p.", "581 c.p.p."]              # reached twice
    assert len(ranked) == 2 + 2 + 2
    monkeypatch.setattr(ee, "REFERS_SOURCES", False)
    assert "593 c.p.p." not in {f"{e['key'][0]} {e['key'][1]}" for e in ee.neighbours([("580", "c.p.p.")], Many())[0]}


def test_the_cap_stops_the_expansion_and_tagged_or_rejected_rows_do_not_count():
    status = {("1158", "c.c."): "tag", ("1160", "c.c."): "skip"}
    ranked, stopped = ee.neighbours([("1159", "c.c."), ("580", "c.p.p.")], Many(),
                                    lambda k: status.get(k, "grade"), cap=2)
    # 1159: 1158 (tagged, free) and 1157 (1160 skipped; 1157 and 1161 tie, the lower first); 580: 591, then 592 is
    # past the cap
    assert [(e["key"][0], e["action"]) for e in ranked] == [("1158", "tag"), ("1157", "grade"), ("591", "grade")]
    assert stopped == 1


def test_expand_tags_the_graded_grades_up_to_the_cap_and_signals_two_anchors(monkeypatch, events):
    monkeypatch.setattr(ee, "MAX_ADDED", 2)
    checked = [row("580", "c.p.p.", 2), row("581", "c.p.p.", 2), row("600", "c.p.p.", None, ["current"]),
               row("1159", "c.c.", 3), row("1158", "c.c.", 2, ["current"])]
    before = checked + [row("1160", "c.c.", 0)]                      # the check rejected 1160
    fetched = []

    def fetch(session, documents, refs, store=None):
        fetched.extend(refs)
        return [row(n, c) for n, c in refs]
    monkeypatch.setattr(dg, "_fetch_any", fetch)
    graded = []

    def grade(rows):
        graded.extend(f"{r['_article'][0]} {r['_article'][1]}" for r in rows)
        for r in rows:
            r["_grade"] = 2
        return rows
    prompt = "[art. 580 c.p.p.]\n[art. 581 c.p.p.]"
    rows, new = ee.expand(None, [], checked, before, prompt, grade, lookup=Many())
    by = {f"{r['_article'][0]} {r['_article'][1]}": r for r in rows}
    # anchors 1159 (grade 3), 580, 581: 1158 tagged, 1160 rejected, 1157 and 591 fill the cap of 2
    assert graded == ["1157 c.c.", "591 c.p.p."]
    assert fetched == [("1157", "c.c."), ("591", "c.p.p.")] and len(new) == 2
    assert "edge" in by["1158 c.c."]["_signals"] and by["1158 c.c."]["_grade"] == 2       # graded: tagged only
    assert by["591 c.p.p."]["_source"] == ee.SOURCE and by["591 c.p.p."]["_edge_rank"] == 3
    assert "600 c.p.p." in by and "edge" not in by["600 c.p.p."]["_signals"]              # past the cap
    log = next(d for n, d in events if n == "case_analysis_edge_expand")
    assert log["sent_to_grading"] == 2 and log["grading_calls"] == 1 and log["stopped_at"] == "580 c.p.p."
    assert all("rank" in g for g in log["graded"]) and log["rejected_before"] == ["1160 c.c."]
    assert [a["why"] for a in log["anchors"]] == ["grade 3", "in the prompt", "in the prompt"]


def test_two_anchors_make_a_second_signal(monkeypatch, events):
    monkeypatch.setattr(ee, "REFERS_SOURCES", True)
    monkeypatch.setattr(dg, "_fetch_any", lambda session, documents, refs, store=None: [row(n, c) for n, c in refs])
    checked = [row("580", "c.p.p.", 2), row("581", "c.p.p.", 2)]
    rows, new = ee.expand(None, [], checked, checked, "[art. 580 c.p.p.]\n[art. 581 c.p.p.]", lambda rs: rs,
                          lookup=Many())
    six = next(r for r in new if r["_article"] == ("600", "c.p.p."))
    assert six["_signals"] == ["edge", ee.SIGNAL_ANCHORS] and six["_multi_signal"]
    assert six["_edge_channels"] == ["edge:cited_by"]


def test_the_edge_is_one_signal_for_plan_rule_include(monkeypatch):
    monkeypatch.delenv("EDGE_EXPAND", raising=False)
    assert "edge" not in ptw._include_signals()                         # the flag off: as before
    monkeypatch.setenv("EDGE_EXPAND", "true")
    assert {"edge", ee.SIGNAL_ANCHORS} <= set(ptw._include_signals()) and "edge" in ptw._include_signals(True)
    plan = {"questioni": [{"titolo": "Usucapione", "argomento": "Usucapione", "norme": [("1159", "c.c.")]}]}
    rows = [{"_article": ("1158", "c.c."), "_grade": 2, "_signals": ["edge"],
             "s": {"plain_text": "Usucapione.", "titolo_articolo": "Usucapione"}}]
    _, inserted, skipped = ptw.include_by_rule(plan, rows, "[art. 1158 c.c.]", {}, {})
    assert [e["article"] for e in inserted] == ["1158 c.c."] and not skipped


def test_the_reach_log(events):
    rows = [{**row("593", "c.p.p.", 2, ["edge"]), "_edge_channels": ["edge:refers_to"]},
            {**row("1158", "c.c.", 2, ["edge"]), "_edge_channels": ["edge:sezione"], "_source": ee.SOURCE}]
    ee.note_plan({"questioni": [{"norme": [("593", "c.p.p.")]}]})
    out = ee.log_reach(rows, "[art. 593 c.p.p.]\nTesto", "L'appello si propone ai sensi dell'art. 593 c.p.p.")
    assert out["edge_rows"] == 2 and out["added"] == 1
    assert out["in_prompt"] == ["593 c.p.p."] and out["in_plan"] == ["593 c.p.p."] and out["in_answer"] == ["593 c.p.p."]
    assert ee.log_reach([row("1", "c.c.")], "", "") is None


def test_the_csv_lookup(tmp_path):
    units = tmp_path / "units.csv"
    with open(units, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["code", "level", "unit name", "article"])
        for n in ("1158", "1159"):
            w.writerows([["c.c.", "CAPO", "CAPO II – DELL'USUCAPIONE", n],
                         ["c.c.", "SEZIONE", "SEZIONE I – Dell'usucapione dei beni immobili", n]])
        w.writerow(["c.c.", "CAPO", "CAPO I – ALTRO", "1100"])
    refers = tmp_path / "refers_to.csv"
    with open(refers, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["from article", "to article", "pattern", "matched text"])
        w.writerows([["593 c.p.p.", "580 c.p.p.", "abbr", "art. 580"], ["593 c.p.p.", "580 c.p.p.", "abbr", "580"],
                     ["579 c.p.p.", "580 c.p.p.", "abbr", "art. 580"]])
    look = ee.CsvLookup(str(units), str(refers))
    assert look.sezione(("1159", "c.c.")) == {("1158", "c.c."), ("1159", "c.c.")}
    assert look.sezione(("1100", "c.c.")) == set()                                  # no Sezione
    assert look.refers_out(("593", "c.p.p.")) == {("580", "c.p.p.")}
    assert look.refers_in(("580", "c.p.p.")) == {("593", "c.p.p."), ("579", "c.p.p.")}
    assert look.inbound(("580", "c.p.p.")) == 2 and look.inbound(("593", "c.p.p.")) == 0   # distinct citing articles


def test_the_graph_queries_only_read():
    for query in (ee.QUERY_REFERS_OUT, ee.QUERY_REFERS_IN, ee.QUERY_SEZIONE):
        assert "MATCH" in query and not any(w in query for w in ("MERGE", "CREATE", "SET ", "DELETE"))
    assert "(s)-[:REFERS_TO]->(t" in ee.QUERY_REFERS_OUT and "(s)<-[:REFERS_TO]-(t" in ee.QUERY_REFERS_IN
    assert "AS inbound" in ee.QUERY_REFERS_OUT and "AS inbound" in ee.QUERY_REFERS_IN


def test_the_graph_lookup_keeps_each_article_s_highest_inbound(monkeypatch):
    class Session:
        def run(self, query, anchors):
            class R:
                def __init__(self, rows):
                    self.rows = rows

                def data(self):
                    return self.rows
            if query == ee.QUERY_REFERS_IN:
                return R([{"anchor": "580|c.p.p.", "doc": "Codice di Procedura Penale", "section": "593.1.0",
                           "inbound": 1},
                          {"anchor": "580|c.p.p.", "doc": "Codice di Procedura Penale", "section": "593", "inbound": 3}])
            return R([])
    monkeypatch.setattr(ee, "code_of_document", lambda name: "c.p.p.")
    look = ee.GraphLookup(Session(), [{"ref": ("580", "c.p.p."), "row": row("580", "c.p.p.")}])
    assert look.refers_in(("580", "c.p.p.")) == {("593", "c.p.p.")} and look.inbound(("593", "c.p.p.")) == 3
