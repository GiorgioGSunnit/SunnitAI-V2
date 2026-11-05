"""POOL_GATE_AREA / POOL_GATE_LINK (v50): dedup by (code, article), the area gate, the link gate (Capo, Titolo without Capi, REFERS_TO, hubs),
the pipeline's log and the rows given back to EDGE_EXPAND; and the simulation reading a report's traces. No model, no
database."""
import os

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

import pytest  # noqa: E402

import src.rag.defensive_generation as dg  # noqa: E402
from src.rag import feature_flags, pool_gates as pg  # noqa: E402
from src.rag.nodes.cypher import CROSSREF_SOURCE  # noqa: E402


def _row(n, code, signals=(), doc="cc", name=None, text=None, **tags):
    return {"_article": (n, code), "_signals": list(signals), "d": {"id": doc, "name": doc},
            "s": {"name": name or n, "plain_text": f"Testo {n}." if text is None else text}, **tags}


class Links:
    """capo / links / inbound from dicts."""

    def __init__(self, capo=None, links=None, inbound=None):
        self._capo, self._links, self._inbound = capo or {}, links or {}, inbound or {}

    def capo(self, k):
        return self._capo.get(k)

    def links(self, k):
        return self._links.get(k, set())

    def inbound(self, k):
        return self._inbound.get(k, 0)


def _cands(rows, named=()):
    return pg.merge([pg.candidate(r, named) for r in rows])


# --- dedup ------------------------------------------------------------------------------------------------------------

def test_dedup_keeps_one_document_per_article_and_its_fragments_merging_the_tags():
    rows = [_row("2051", "c.c.", ["frame"], name="2051"),
            _row("2051", "c.c.", name="2051.1", text="Secondo comma."),                  # a fragment: kept
            _row("2051", "c.c.", ["distill"], name="2051_1", _vote=True),                  # same text again: merged
            _row("21", "c.p.a.", ["frame"], doc="cpa"),
            _row("21", "Codice del processo amministrativo 2025", ["baseline"], doc="cpa25",
                 _source=CROSSREF_SOURCE)]                                                 # another document: merged
    kept, merged = pg.dedup(rows)
    assert [r["s"]["name"] for r in kept] == ["2051", "2051.1", "21"] and len(merged) == 2
    assert kept[0]["_vote"] and kept[0]["_signals"] == ["frame", "distill"]
    assert kept[2]["_source"] == CROSSREF_SOURCE and kept[2]["_signals"] == ["frame", "baseline"]


# --- area -------------------------------------------------------------------------------------------------------------

def test_the_area_gate_drops_a_main_code_the_areas_do_not_allow_unless_named_or_found_twice():
    rows = [_row("360", "c.p.c.", ["baseline"]), _row("163", "c.p.c."), _row("700", "c.p.c.", ["frame", "distill"]),
            _row("584", "c.p.", ["frame"]), _row("12", "Codice del Consumo 2026", ["frame"])]
    cands = _cands(rows, named=[("163", "c.p.c.")])
    assert pg.area_out(cands, {"c.p.", "c.p.p."}) == [("360", "c.p.c.")]      # a law outside the five: never
    assert pg.area_out(cands, None) == []                                       # areas unknown: no gate


# --- link -------------------------------------------------------------------------------------------------------------

def test_the_link_gate_keeps_a_single_channel_row_with_a_strong_one_in_its_capo_or_linked():
    strong = _row("1218", "c.c.", ["frame", "distill"])
    same_capo = _row("1223", "c.c.", ["baseline"])
    linked = _row("1176", "c.c.", ["baseline"])
    alone = _row("1350", "c.c.", ["baseline"])
    via_hub = _row("1", "c.c.", ["baseline"])
    hub_anchor = _row("3", "c.p.c.", ["frame", "distill"])
    to_hub = _row("99", "c.p.c.", ["baseline"])
    heading = _row("2697", "c.c.", ["baseline"], _heading_match=True)
    named = _row("2043", "c.c.")
    links = Links(capo={("1218", "c.c."): "capo III", ("1223", "c.c."): "capo III", ("1350", "c.c."): "capo IX"},
                  links={("1176", "c.c."): {("1218", "c.c.")}, ("1", "c.c."): {("1218", "c.c.")},
                         ("99", "c.p.c."): {("3", "c.p.c.")}},
                  inbound={("1", "c.c."): 500, ("3", "c.p.c."): 1269})
    cands = _cands([strong, same_capo, linked, alone, via_hub, hub_anchor, to_hub, heading, named],
                   named=[("2043", "c.c.")])
    assert sorted(pg.link_out(cands, links)) == [("1", "c.c."), ("1350", "c.c."), ("99", "c.p.c.")]


def test_a_titolo_without_capi_counts_as_the_capo_in_the_csv_lookup(tmp_path):
    units = tmp_path / "units.csv"
    units.write_text("code,level,unit name,article\n"
                     "c.c.,LIBRO,LIBRO IV,2043\nc.c.,TITOLO,TITOLO IX – DEI FATTI ILLECITI,2043\n"
                     "c.c.,LIBRO,LIBRO IV,2051\nc.c.,TITOLO,TITOLO IX – DEI FATTI ILLECITI,2051\n"
                     "c.c.,LIBRO,LIBRO IV,1218\nc.c.,TITOLO,TITOLO I,1218\nc.c.,CAPO,CAPO III,1218\n", encoding="utf-8")
    refers = tmp_path / "refers.csv"
    refers.write_text("from article,to article,pattern,matched text\n2051 c.c.,1218 c.c.,abbr,art. 1218\n",
                      encoding="utf-8")
    links = pg.CsvLinks(str(units), str(refers))
    assert links.capo(("2043", "c.c.")) == links.capo(("2051", "c.c.")) is not None
    assert links.capo(("1218", "c.c."))[-1] == "CAPO III"
    assert links.links(("1218", "c.c.")) == {("2051", "c.c.")} and links.inbound(("1218", "c.c.")) == 1


# --- the pipeline step ------------------------------------------------------------------------------------------------

def test_apply_runs_the_three_gates_logs_them_and_gives_back_the_rows_gated_out(monkeypatch):
    seen = []
    monkeypatch.setattr(dg, "vlog", lambda name, data=None, *a, **k: seen.append((name, data)))
    rows = [_row("584", "c.p.", ["frame", "distill"], doc="cp"), _row("43", "c.p.", ["frame"], doc="cp"),
            _row("43", "c.p.", ["frame"], doc="cp", name="43_1"),           # same text: merged
            _row("2043", "c.c.", ["baseline"]), _row("640", "c.p.", ["baseline"], doc="cp")]
    analysis = {"area": "penale", "frame": {"areas": ["penale"]}}
    kept, gone = pg.apply(None, rows, analysis, "Tizio colpisce Caio.",
                          lookup=Links(capo={("584", "c.p."): "capo I", ("43", "c.p."): "capo I"}), area=True, link=True)
    assert [(r["_article"][0], r["s"]["name"]) for r in kept] == [("584", "584"), ("43", "43")]
    event = next(d for n, d in seen if n == pg.EVENT)
    assert event["area"] == ["2043 c.c."] and event["dedup"] == {"rows": 1, "articles": ["43 c.p."]}
    assert "640 c.p." in event["link"] and event["rows_before"] == 5 and event["rows_after"] == len(kept)
    assert sorted(r["s"]["name"] for r in gone) == ["2043", "43_1", "640"] and event["allowed"] == ["c.p.", "c.p.p."]


def test_a_failed_link_lookup_skips_only_the_link_gate(monkeypatch):
    seen = []
    monkeypatch.setattr(dg, "vlog", lambda name, data=None, *a, **k: seen.append((name, data)))

    class Broken:
        def run(self, *a, **k):
            raise RuntimeError("no graph")
    rows = [_row("2043", "c.c.", ["baseline"]), _row("640", "c.p.", ["baseline"], doc="cp")]
    kept, gone = pg.apply(Broken(), rows, {"area": "penale", "frame": {"areas": ["penale"]}}, "x", area=True, link=True)
    event = next(d for n, d in seen if n == pg.EVENT)
    assert event["lookup"].startswith("failed") and event["area"] == ["2043 c.c."] and event["link"] == []
    assert [r["_article"] for r in kept] == [("640", "c.p.")]


def test_both_gates_are_off_by_default_and_each_flag_runs_only_its_own(monkeypatch):
    assert not pg.enabled() and not feature_flags.enabled(pg.AREA_FLAG) and not feature_flags.enabled(pg.LINK_FLAG)
    monkeypatch.setattr(dg, "vlog", lambda *a, **k: None)
    rows = [_row("584", "c.p.", ["frame", "distill"], doc="cp"), _row("2043", "c.c.", ["baseline"]),
            _row("640", "c.p.", ["baseline"], doc="cp")]
    analysis = {"area": "penale", "frame": {"areas": ["penale"]}}
    monkeypatch.setenv(pg.AREA_FLAG, "true")
    kept, _ = pg.apply(None, [dict(r) for r in rows], analysis, "x", lookup=Links())
    assert [r["_article"][0] for r in kept] == ["584", "640"]                     # area only
    monkeypatch.delenv(pg.AREA_FLAG)
    monkeypatch.setenv(pg.LINK_FLAG, "true")
    kept, _ = pg.apply(None, [dict(r) for r in rows], analysis, "x", lookup=Links())
    assert [r["_article"][0] for r in kept] == ["584"]                            # link only: 2043 and 640 unlinked


def test_the_link_gate_never_removes_the_understanding_s_regime_or_procedural_rows():
    rows = [_row("1218", "c.c.", ["frame", "distill"]), _row("1", "c.c.", ["frame"]), _row("2", "c.c.", ["distill"]),
            _row("3", "c.c.", _source="regime_candidate"), _row("4", "c.c.", _source="procedural_tool"),
            _row("5", "c.c.", ["baseline"]), _row("6", "c.c.", _source="method_candidates")]
    assert sorted(pg.link_out(_cands(rows), Links())) == [("5", "c.c."), ("6", "c.c.")]


def test_every_area_the_frame_or_the_analysis_emits_is_mapped():
    analysis_areas = ("civile", "penale", "amministrativo", "lavoro", "tributario", "altro")     # _ANALYSIS_SYSTEM
    assert set(dg._FRAME_AREAS) | set(analysis_areas) <= set(pg.AREA_CODES) | pg.NO_CODES
    assert pg.allowed_codes(pg.gate_areas({"frame": {"areas": ["lavoro", "amministrativo", "altro"]}}), []) == \
        {"c.c.", "c.p.c.", "c.p.a."}                                              # n3: c.p.a. only before
    assert pg.gate_areas({"frame": {"areas": []}, "area": "lavoro"}) == ["lavoro"]
    assert pg.allowed_codes(pg.gate_areas({"frame": {"areas": ["altro"]}}), []) is None    # nothing known: no gate
    assert pg.allowed_codes(["penale"], {"c.c."}) == {"c.p.", "c.p.p.", "c.c."}           # the user names c.c.


def test_labour_laws_and_ccnl_documents_are_never_area_gated():
    rows = [_row("18", "Statuto dei lavoratori"), _row("19", "Decreto legislativo 81/2015"),
            _row("10", "CCNL Commercio 2024"), _row("2094", "c.c.")]
    assert pg.area_out(_cands(rows), {"c.p.", "c.p.p."}) == [("2094", "c.c.")]


# --- the simulation ---------------------------------------------------------------------------------------------------

def test_the_simulation_reads_the_traces_channels_and_counts_the_expected_norms_lost():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import pool_gates_sim as sim

    def trace(label, channels, signals=(), hits=None, check="grade 2"):
        return {"label": label, "channels": list(channels), "signals": list(signals),
                "hits": len(signals) if hits is None else hits, "in_pool": True, "check": check}
    case = {"areas": ["penale"], "area": "penale",
            "pool_traces": [trace("584 c.p.", ["frame", "distill"], ["frame", "distill"]),
                            trace("40 c.p.", ["other_search"], [], hits=2),              # a fifth signal: hits only
                            trace("2043 c.c.", ["distill"], ["distill"]),
                            trace("640 c.p.", ["distill"], ["distill"]),
                            trace("5 c.p.", ["edge:sezione"], [])],                       # an EDGE_EXPAND addition
            "slots": [{"label": "art. 641 cp", "stage": "MENTIONED", "pool_label": "641 c.p."}],
            "unexpected": [{"label": "2043 c.c.", "in_pool": True}]}
    case["pool_traces"].append(trace("641 c.p.", ["vector"], ["baseline"]))
    out = sim.simulate(case, Links(capo={("584", "c.p."): "a"}), "Tizio.", 100)
    assert out["pool"] == 5 and out["area"]["removed"] == [("2043", "c.c.")]
    assert out["link"]["removed"] == [("641", "c.p.")]                  # alone; 640 and 2043: distill, never by link
    assert out["both"]["removed"] == [("2043", "c.c."), ("641", "c.p.")] and out["area"]["noise"] == 1
    assert out["link"]["lost"] == [("art. 641 cp", "MENTIONED")] and out["area"]["lost"] == []
