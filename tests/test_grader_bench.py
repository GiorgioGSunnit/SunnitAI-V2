"""scripts/grader_bench.py offline: a synthetic run folder, a fake grader, no database."""

import importlib.util
import json
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

import src.rag.defensive_generation as dg  # noqa: E402

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("grader_bench", HERE.parent / "scripts" / "grader_bench.py")
gb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gb)
dl = gb._diagnose_long()
_ev_spec = importlib.util.spec_from_file_location("eval_narrative", HERE / "eval_narrative.py")
ev = importlib.util.module_from_spec(_ev_spec)
_ev_spec.loader.exec_module(ev)

# The 1206-1415 dump format: the label itself in brackets, the multi-signal mark of T1.
DUMP = ("stage: applicability\nmax_tokens: 840\nstatus: ok\n\n===== SYSTEM =====\nun voto di pertinenza\n\n"
        "===== USER =====\nFatti del caso:\nTizio spinge Caio, che muore.\n\nArticoli candidati (testo reale dalla "
        "banca dati), 3: un oggetto per ciascuno degli identificativi A1, A2, A3.\n\n"
        "[A1] 584 c.p.\nRubrica: Omicidio preterintenzionale\nTrovato da: più fonti\nTesto: Chiunque, con atti "
        "diretti a commettere...\n\n[A2] 133 c.p.\nRubrica: Gravità del reato\nTesto: Nell'esercizio del potere...\n\n"
        "[A3] 2043 c.c.\nRubrica: -\nTesto: Qualunque fatto doloso...\n\n===== REPLY =====\n[]\n")
RECORD = {"id": "sempronio", "kind": "eval", "expected": [{"code": "cp", "article": "584"}],
          "frame": {"tasks": ["Quale reato?"], "central_issue_candidates": [{"nome": "omicidio preterintenzionale"}],
                    "areas": ["penale"]},
          "questioni": [{"titolo": "Questione centrale: omicidio preterintenzionale", "voce": "questione_centrale",
                         "istituti": [{"nome": "omicidio preterintenzionale"}]}],
          "area": "penale",
          "pool": [{"label": "584 c.p.", "sources": [], "multi_signal": True}, {"label": "133 c.p.", "sources": []},
                   {"label": "2043 c.c.", "sources": []}],
          "fusion": {"signals": {"current": ["584 c.p."], "baseline": ["584 c.p.", "2043 c.c."], "distill": ["133 c.p."]}},
          "institutions": [{"phase": "v2", "nome": "omicidio preterintenzionale", "articles": [{"label": "584 c.p."}]}],
          "check_result": {"grades": {"584 c.p.": 1, "133 c.p.": 1, "2043 c.c.": 0}}}


class Grader:
    """3 for 584 when it sees the evidence, else 1; 133 always 1; 2043 always 0."""

    def __init__(self):
        self.humans = []

    def __call__(self, messages, max_tokens=None, temperature=None, **kwargs):
        human = messages[-1].content
        self.humans.append(human)
        out = []
        for key, label in re.findall(r"^\[(A\d+)\] (.+)$", human, re.MULTILINE):
            block = human.split(f"[{key}] {label}", 1)[1]
            grade = {"133 c.p.": 1, "2043 c.c.": 0}.get(label, 3 if "Sentenze simili" in block or "(più fonti)" in block
                                                         and "istituto di una questione centrale" in block else 1)
            out.append({"a": key, "g": grade, "q": "Q1", "r": "x"})
        return json.dumps(out)


def test_inputs_come_from_the_dumps_in_both_formats():
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "05_applicability.txt"), "w", encoding="utf-8") as fh:
        fh.write(DUMP)
    old = DUMP.replace("[A1] 584 c.p.\n", "[584 c.p.]\n").replace("[A2] 133 c.p.\n", "[133 c.p.]\n") \
        .replace("[A3] 2043 c.c.\n", "[2043 c.c.]\n")
    with open(os.path.join(d, "09_applicability_retry.txt"), "w", encoding="utf-8") as fh:
        fh.write(old.replace("584 c.p.", "585 c.p."))
    inputs = gb.graded_inputs(d, dl._dump_parts)
    assert list(inputs) == ["584 c.p.", "133 c.p.", "2043 c.c.", "585 c.p."]
    assert inputs["584 c.p."] == ("Omicidio preterintenzionale", "Chiunque, con atti diretti a commettere...")
    assert inputs["2043 c.c."][0] == ""


def test_pool_tags_rebuild_the_evidence():
    tags = gb.pool_tags(RECORD, dg)
    assert tags["584 c.p."]["_signals"] == ["current", "baseline"] and tags["584 c.p."]["_central"]
    assert tags["584 c.p."]["_multi_signal"] and not tags["2043 c.c."].get("_central")


def test_the_bench_grades_every_version_and_reports(monkeypatch):
    monkeypatch.setattr(dg, "_call_chat", Grader())
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "05_applicability.txt"), "w", encoding="utf-8") as fh:
        fh.write(DUMP)
    inputs = gb.graded_inputs(d, dl._dump_parts)
    cases = {"sempronio": {"question": "Tizio spinge Caio, che cade e muore."}}
    results = gb.bench_case(RECORD, inputs, ["v0", "v3"], dg, dl._ekey_fn(ev), cases, vote=False)
    by = {r["label"]: r for r in results}
    assert by["584 c.p."]["expected"] and not by["133 c.p."]["expected"]
    assert by["584 c.p."]["grades"]["saved"]["grade"] == 1 and by["584 c.p."]["grades"]["v3"]["grade"] == 3
    assert by["584 c.p."]["grades"]["v3"]["issue"] == "Quale reato?"            # Q1 read as the first issue
    summary = gb.summarise(results)
    assert summary["v3"]["exp_pct"] == 100 and summary["saved"]["exp_pct"] == 0 and summary["v3"]["other_hi"] == 0
    text = gb.render("/runs/20261027-1415", results, ["v0", "v3"], 60, [])
    assert "| v3 | 1 | 1 (100%) | 2 | 0 (0%) | 0/0/0/1/0 | 1/1/0/0/0 |" in text
    assert "## saved: expected articles graded 1 or less (1)" in text and "| sempronio | 584 c.p. | 1 |" in text
