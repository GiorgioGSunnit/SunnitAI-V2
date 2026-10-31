"""scripts/rerank_bench.py on a synthetic saved run: no model, no database."""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rb = _module("rerank_bench", ROOT / "scripts" / "rerank_bench.py")
dl = _module("diagnose_long_for_bench", ROOT / "scripts" / "diagnose_long.py")
gb = _module("grader_bench_for_bench", ROOT / "scripts" / "grader_bench.py")
ev = _module("eval_narrative_for_bench", ROOT / "tests" / "eval_narrative.py")
ekey = dl._ekey_fn(ev)


def _record():
    return {"id": "x1", "kind": "eval", "request": "", "area": "penale",
            "expected": [{"code": "cp", "article": "584"}, {"code": "cp", "any_of": ["582", "583"]},
                         {"code": "cp", "article": "43"}, {"code": "cp", "article": "999"}],
            "frame": {"areas": ["penale"], "tasks": ["Quale reato?"],
                      "central_issue_candidates": [{"nome": "omicidio preterintenzionale"}, {"nome": "quale reato?"}]},
            "questioni": [{"titolo": "Questione centrale: nesso causale"}],
            "pool": [{"label": lab, "rubrica": f"Rubrica {lab}"} for lab in
                     ("584 c.p.", "43 c.p.", "582 c.p.", "2043 c.c.", "81 c.p.", "1 c.p.")],
            "fusion": {"fused": ["2043 c.c.", "81 c.p.", "43 c.p.", "584 c.p."]},
            "check": {"input": ["584 c.p.", "43 c.p.", "582 c.p.", "2043 c.c.", "81 c.p.", "1 c.p."],
                      "exempt": {"584 c.p.": "user_text"}},
            "check_result": {"verdicts": {"43 c.p.": "grade 3", "582 c.p.": "grade 1", "2043 c.c.": "grade 0",
                                          "81 c.p.": "grade 2", "1 c.p.": "missing"},
                             "issues": {"43 c.p.": "elemento soggettivo"}},
            "events": [["case_analysis_applicability", {"batches": 1}, 2500.0]]}


def _case(rec=None):
    inputs = {"43 c.p.": ("Elemento psicologico", "Il delitto è doloso quando " + "x" * 600)}
    return rb.build_case(rec or _record(), "Il caso.", inputs, {"582 c.p.": "Chiunque cagiona una lesione"}, ekey,
                         {"c.p.", "c.p.p."}, {"c.p.", "c.p.p.", "c.c.", "c.p.c.", "c.p.a."}, 400)


def test_a_case_is_built_from_the_saved_data_only():
    case = _case()
    by = {c["label"]: c for c in case["candidates"]}
    assert by["43 c.p."]["source"] == "graded call" and len(by["43 c.p."]["doc"]) < 460
    assert by["582 c.p."]["source"] == "material.txt"
    assert by["584 c.p."]["source"] == "rubrica only" and by["584 c.p."]["exempt"]
    assert by["2043 c.c."]["other_area"] and not by["81 c.p."]["other_area"]
    assert by["2043 c.c."]["grade"] == 0 and by["1 c.p."]["grade"] is None
    assert case["issues"] == ["Quale reato?", "omicidio preterintenzionale", "nesso causale"]
    assert case["qwen_seconds"] == 2.5


def test_a_run_without_grades_or_key_is_skipped():
    rec = _record()
    rec["check_result"]["verdicts"] = {"43 c.p.": "applicabile"}
    assert "grades" in _case(rec)["skip"]
    rec = _record()
    rec["expected"] = []
    assert "key" in _case(rec)["skip"]


def test_orders():
    case = _case()
    labels = lambda order: [c["label"] for c in order]     # noqa: E731
    assert labels(rb.fusion_order(case["candidates"]))[:4] == ["2043 c.c.", "81 c.p.", "43 c.p.", "584 c.p."]
    assert labels(rb.qwen_order(case["candidates"])) == ["584 c.p.", "43 c.p.", "81 c.p.", "582 c.p.", "1 c.p.",
                                                         "2043 c.c."]
    scores = {"2043 c.c.": 9, "584 c.p.": 1, "43 c.p.": 5, "82 c.p.": 0}
    assert labels(rb.score_order(case["candidates"], scores))[:2] == ["2043 c.c.", "43 c.p."]
    assert labels(rb.score_order(case["candidates"], scores, pin=True))[:2] == ["584 c.p.", "2043 c.c."]


def test_metrics_count_slots_ceiling_and_losses_at_qwen_s_size():
    case = _case()
    qwen = rb.metrics(case, rb.qwen_order(case["candidates"]), 2)
    assert qwen["selected"] == 3 and qwen["slots"] == 4 and qwen["ceiling"] == 3   # 999 is not a candidate
    assert qwen["hit10"] == 3 and qwen["rr"] == 1.0 and qwen["lost"] == [] and qwen["gained"] == []
    fusion = rb.metrics(case, rb.fusion_order(case["candidates"]), 2)
    assert fusion["rr"] == 1 / 3
    assert fusion["lost"] == ["art. 584 cp"] and fusion["gained"] == []
    # 2043 c.c.: other area and grade 0; 1 c.p. never graded; the expected 582 graded 1 is not unrelated
    assert qwen["unrelated"] == 1 and qwen["other_area"] == 1 and qwen["never_graded"] == 1
    assert fusion["top"] == 6


def test_an_any_of_slot_counts_once_by_its_best_alternative():
    rec = _record()
    rec["check"]["input"].append("583 c.p.")
    rec["pool"].append({"label": "583 c.p.", "rubrica": "R"})
    case = _case(rec)
    out = rb.metrics(case, rb.qwen_order(case["candidates"]), 2)
    assert out["ceiling"] == 3


def test_issues_score_is_the_max_and_a_case_without_issues_uses_the_whole_text():
    class Fake:
        max_length = 8

        def predict(self, pairs):
            return [len(q) * (2 if "81" in d else 1) / 10 for q, d in pairs]

        def tokens(self, text):
            return len(text.split())
    case = _case()
    out = rb.score_case(Fake(), case)
    assert out["issues"]["81 c.p."] == 2 * len("omicidio preterintenzionale") / 10
    assert out["pairs"] == 6 * 4
    case["issues"] = []
    assert rb.score_case(Fake(), case)["issues"] == rb.score_case(Fake(), case)["whole"]


def test_examples_prefer_the_pinned_rows():
    case = _case()
    bad = {c["label"]: -i for i, c in enumerate(rb.qwen_order(case["candidates"]))}
    worst = {k: -v for k, v in bad.items()}
    ranked = {"x1": rb.rankings(case, {"bge": {"whole": worst, "issues": bad}})}
    picked = rb.examples([case], ranked, 2, 10)
    # 43 is lost with the exempt articles pinned; 584, exempt, only without the pin
    assert [(p["article"], p["pin"]) for p in picked] == [("art. 43 cp", True), ("art. 584 cp", False)]
    assert all("+pin" in name for name, _ in picked[0]["rankers"])
    assert rb.examples([case], ranked, 2, 0) == []


def test_material_texts_read_the_prompt_block_only(tmp_path):
    path = tmp_path / "material.txt"
    path.write_text("head\n========== ARTICOLI ==========\n[art. 584 c.p.]\nChiunque\n>> provenance: x\ncagiona\n\n"
                    "[art. 43 c.p.]\nIl delitto\n\n========== SENTENZE ==========\n[art. 1 c.p.]\nno\n", encoding="utf-8")
    assert rb.material_texts(str(path)) == {"584 c.p.": "Chiunque cagiona", "43 c.p.": "Il delitto"}
    assert rb.material_texts(str(tmp_path / "none.txt")) == {}


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def data(self):
        return self.rows


class _Session:
    """The two document lookups and article_text's sections query, from {doc name: {section name: text}}."""
    def __init__(self, documents):
        self.documents, self.reads = documents, 0

    def run(self, query, **params):
        if "RETURN d.id AS id" in query:
            found = [n for n in self.documents if n == params.get("name") or n.startswith(params.get("prefix") or "\0")]
            return _Rows([{"id": n} for n in found])
        self.reads += 1
        rows = []
        for item in params["items"]:
            for name, text in self.documents[item["doc"]].items():
                if name == item["num"] or name.startswith(item["num"] + "."):
                    rows.append({"doc": item["doc"], "num": item["num"], "d": {"id": item["doc"]},
                                 "s": {"name": name, "plain_text": text, "titolo_articolo": "Rubrica letta"}})
        return _Rows(rows)


def test_thin_candidates_get_the_shared_article_text_from_their_document():
    rec = _record()
    for row in rec["pool"]:
        if row["label"] == "584 c.p.":
            row["document"], row["rubrica"] = "Codice Penale 2026", ""
    case = _case(rec)
    text = "Chiunque, con atti diretti a commettere uno dei delitti, cagiona la morte di un uomo, è punito."
    session = _Session({"Codice Penale 2026": {"584.0.0": "In vigore dal 1 gennaio 2020", "584": text,
                                               "582.0.0": "x" * 70}})
    cache = {}
    counts = rb.apply_texts([case], session, cache, 400)
    by = {c["label"]: c for c in case["candidates"]}
    assert by["584 c.p."]["source"] == "neo4j" and text in by["584 c.p."]["doc"]     # the date-only N.0.0 is skipped
    assert by["584 c.p."]["rubrica"] == "Rubrica letta"
    assert by["43 c.p."]["source"] == "graded call"                                  # already its text: not read
    assert by["582 c.p."]["source"] == "neo4j"                                       # material.txt: read too, by prefix
    assert by["2043 c.c."]["source"] == "rubrica only"                               # nowhere: stays, and is counted
    assert counts["fetched"] == 2 and counts["thin"] == 5 and counts["not_found"] == 3
    fresh = _case(rec)
    assert rb.apply_texts([fresh], None, cache, 400)["reads"] == 0                   # the second time, from the cache
    assert {c["label"]: c["source"] for c in fresh["candidates"]}["584 c.p."] == "neo4j"


def test_a_short_or_cut_graded_text_is_thin():
    assert rb.is_thin({"source": "graded call", "text": "breve"}, 400)
    assert not rb.is_thin({"source": "graded call", "text": "x" * 400}, 400)
    assert rb.is_thin({"source": "graded call", "text": "x" * 400}, 800)
    assert rb.is_thin({"source": "material.txt", "text": "x" * 400}, 400)


def test_saved_scores_are_not_reused_for_another_text(tmp_path):
    case = _case()
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"inputs": rb.inputs_hash(case)}), encoding="utf-8")
    assert rb.load_scores(str(path), case)
    case["candidates"][0]["doc"] += " altro"
    assert rb.load_scores(str(path), case) is None


def test_bge_is_the_default_and_jina_is_opt_in(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["rerank_bench.py", "--help"])
    try:
        rb.main()
    except SystemExit:
        pass
    assert "default bge" in capsys.readouterr().out


def test_the_whole_run_resumes_and_reports(tmp_path, monkeypatch):
    run = tmp_path / "run"
    (run / "cases" / "c1" / "llm").mkdir(parents=True)
    rec = _record()
    rec["id"] = "c1"
    (run / "cases" / "c1.json").write_text(json.dumps(rec), encoding="utf-8")
    (run / "cases" / "c1.input.json").write_text("{}", encoding="utf-8")
    (run / "cases" / "c1" / "llm" / "01_applicability.txt").write_text(
        "stage: applicability\n\n===== SYSTEM =====\nun voto di pertinenza\n\n===== USER =====\nFatti\n\nArticoli "
        "candidati (testo)\n\n[A1] 43 c.p.\nRubrica: Elemento\nTrovato da: più fonti\nTesto: Il delitto\n\n"
        "===== REPLY =====\n[]\n", encoding="utf-8")
    calls = []

    class Fake:
        max_length = 512

        def __init__(self, *a):
            calls.append(a[0])

        def predict(self, pairs):
            return [float(len(d)) for _, d in pairs]

        def tokens(self, text):
            return 600
    monkeypatch.setattr(rb, "Scorer", Fake)
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(rb, "allowed_for", lambda rec, text: ({"c.p."}, {"c.p.", "c.c."}))
    monkeypatch.setattr(sys, "argv", ["rerank_bench.py", str(run), "--models", "bge,jina", "--only", "c1",
                                      "--no-fetch"])
    questions = {"cases": [{"id": "c1", "question": "Il caso"}]}
    real_open = open

    def fake_open(path, *a, **kw):
        if str(path).endswith("eval_narrative_cases.json"):
            target = tmp_path / "eval.json"
            target.write_text(json.dumps(questions), encoding="utf-8")
            return real_open(target, *a, **kw)
        return real_open(path, *a, **kw)
    monkeypatch.setattr("builtins.open", fake_open)
    rb.main()
    assert calls == ["bge", "jina"]
    rb.main()                             # the saved scores: no model loaded again
    assert calls == ["bge", "jina"]
    report = next(run.glob("rerank_bench_*.md")).read_text(encoding="utf-8")
    assert "## Query: whole" in report and "## Query: issues" in report and "| bge+pin whole |" in report
    assert "### jina issues" in report and "graded call 1" in report and "Trovato da" not in report
    assert "1 of 1 cases with a case text over 80%" in report
