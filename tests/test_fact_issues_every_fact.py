"""FACT_ISSUES_EVERY_FACT_ENABLED (src/rag/fact_issues.py) and scripts/fact_loss_report.py. Every article here is
invented; no model, no database."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")

from src.rag import defensive_generation as dg  # noqa: E402
from src.rag import fact_issues as fi  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _reply(n_facts, per_fact):
    return {"istituti": [{"fatto": f"fatto {f}", "nome": f"istituto {chr(97 + f)}{'bis' * k}", "codice": "c.c.", "area": "contratti",
                          "ruolo_cliente": "acquirente"} for f in range(n_facts) for k in range(per_fact)]}


def test_off_the_cap_and_the_prompt_are_as_before(monkeypatch):
    monkeypatch.setenv(fi.FLAG, "true")
    monkeypatch.delenv(fi.EVERY_FACT_FLAG, raising=False)
    assert fi.cap("x" * 40000, {"facts": ["a"] * 20}) == fi.MAX_INSTITUTIONS and fi.system_for(16) == fi.SYSTEM
    names = [i["nome"] for i in fi.clean(_reply(6, 2), [])]
    assert names[:3] == ["istituto a", "istituto abis", "istituto b"] and len(names) == 8   # the model's order


def test_every_fact_named_first_and_the_cap_scales_with_the_case(monkeypatch):
    monkeypatch.setenv(fi.FLAG, "true")
    monkeypatch.setenv(fi.EVERY_FACT_FLAG, "true")
    assert fi.cap("x" * 3000, {"facts": ["a"] * 3}) == 8                 # a short case: as before
    assert fi.cap("x" * 3000, {"facts": ["a"] * 12}) == 12               # one per decisive fact
    assert fi.cap("x" * 20000, {"facts": ["a"] * 2}) == 12               # a long case: one more per 4000 characters
    assert fi.cap("x" * 90000, {"facts": ["a"] * 40}) == fi.MAX_WIDE
    assert "per OGNI fatto decisivo almeno un istituto" in fi.system_for(12) and "al massimo 12" in fi.system_for(12)
    names = [i["nome"] for i in fi.clean(_reply(6, 2), [], 8, by_fact=True)]
    assert names == [f"istituto {c}" for c in "abcdef"] + ["istituto abis", "istituto bbis"]   # no fact starved
    seen = {}
    monkeypatch.setattr(dg, "_chat_json", lambda messages, max_tokens, **k: seen.update(
        system=messages[0].content, max_tokens=max_tokens) or _reply(12, 1))
    added = fi.fact_issues("x" * 3000, {"facts": [f"fatto {i}" for i in range(12)]}, None)
    assert len(added) == 12 and "al massimo 12" in seen["system"] and seen["max_tokens"] == 1800


def test_the_loss_report_tells_named_from_never_named():
    root = tempfile.mkdtemp()
    run = os.path.join(root, "live")
    os.makedirs(os.path.join(run, "cases", "c1", "llm"))
    events = [["case_analysis_fact_issues", {"institutions": [
        {"nome": "garanzia per i vizi", "fatto": "la lavatrice si rompe dopo tre mesi"}] + [
        {"nome": f"istituto {k}", "fatto": f"altro evento numero {k}"} for k in range(7)]}],
        ["case_analysis_fact_institution", {"institution": "garanzia per i vizi", "fonte": "fatto",
                                            "kept": ["1470 c.c."], "rejected": ["1490 c.c."]}]]
    json.dump({"id": "c1", "events": events}, open(os.path.join(run, "cases", "c1.json"), "w"))
    lost = [{"article": "art. 1490 cc", "fact": "la lavatrice si rompe dopo tre mesi dall'acquisto",
             "captured": ["fatto: rottura"], "funnel": {"lost_at": "resolved"}, "capture_kind": "fact"},
            {"article": "art. 2236 cc", "fact": "infezione contratta dopo l'intervento chirurgico",
             "captured": ["fatto: infezione"], "funnel": {"lost_at": "resolved"}, "capture_kind": "fact"},
            {"article": "art. 1218 cc", "fact": "x", "captured": [], "funnel": {}, "capture_kind": None}]
    audit = os.path.join(root, "audit.json")
    json.dump({"cases": [{"run": "live", "id": "c1", "understanding": {"lost": lost}}]}, open(audit, "w"))
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "fact_loss_report.py"), audit, run], cwd=ROOT,
                         capture_output=True, text=True, env={**os.environ}).stdout
    assert "captured only as a fact: 2" in out and "yes 1, no 1" in out
    assert "| rejected by the rubrica filter | 1 | 50% |" in out and "| not named: cap reached | 1 | 50% |" in out
