"""v79.16.2: scripts/replay_cases.py (measurement only) -- the per-case measurement on the synthetic fixtures with the
model faked (the harness of 79.15.0), the diff against a baseline, the six synthetic cases of data/replay_cases. No
network, no model: the script's run_case (the real entry against a model) is exercised on the server by hand."""
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import _fixture_79_15_0 as H  # noqa: E402
import test_v79_16_0_fixture_civil as CIVIL  # noqa: E402
import test_v79_16_0_fixture_penal as PENAL  # noqa: E402
import test_v79_16_1_fixture_defence as DEFENCE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("replay_cases", ROOT / "scripts" / "replay_cases.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

FIXTURES = (("fixture_civil_claimant", CIVIL), ("fixture_penal_victim", PENAL), ("fixture_civil_defence", DEFENCE))


def _rows(monkeypatch, tmp_path):
    rows = []
    for name, mod in FIXTURES:
        run = H.run(monkeypatch, tmp_path / name, mod.case)
        rows.append(R.measure(name, run["state"], run["answer"], 0.0, run["by"]))
    return rows


def test_the_measurement_reads_every_stage_of_the_faked_fixtures(monkeypatch, tmp_path):
    rows = _rows(monkeypatch, tmp_path)
    by_id = {r["id"]: r for r in rows}
    civil, penal, defence = (by_id[n] for n, _ in FIXTURES)
    for r in rows:
        assert r["error"] == "" and r["answer_chars"] > 0
        assert r["analysis"]["status"] == "ok" and r["analysis"]["line_tier"] == "fit_si"
        assert r["analysis"]["fallback_skipped_reason"] == "line_verified"
        assert r["retrieval"]["source"] == "frame" and r["retrieval"]["reason"] == ""
        assert r["fit"]["fit_si_verified"] >= 2 and r["fit"]["fit_unverified_ids"] == 0
        assert r["analysis_counters"]["analysis_provato_no_pertinence"] == 0
        assert "quesiti" in r["frame"]["verified"] and r["frame"]["fields"]["quesiti"]
    assert civil["analysis"]["lines"][0]["articolo"] == "1385 c.c." and civil["retrieval"]["added"] == ["2946 c.c."]
    assert civil["retrieval"]["queries"] == 3 and civil["qa"]["qa_questions_detected"] == 3
    assert civil["qa"]["qa_answer_reasked"] == 1 and civil["qa"]["qa_answer_insufficient"] == 1
    assert penal["analysis"]["lines"][0]["articolo"] == "640 c.p." and penal["retrieval"]["added"] == ["74 c.p.p."]
    assert defence["analysis"]["lines"][0]["articolo"] == "1460 c.c."
    assert defence["analysis"]["rejected"][0]["articolo"] == "1454 c.c."
    assert defence["analysis_counters"]["analysis_line_procedural_rejected"] == 1
    assert dict(e for e in civil["analysis"]["esiti"]["1385 c.c."])["A1385cc:E1"] == "provato"
    json.dumps(rows, ensure_ascii=False)        # every value serialisable


def test_the_report_and_the_diff_say_what_moved(monkeypatch, tmp_path, capsys):
    rows = _rows(monkeypatch, tmp_path)
    run = {"version": "79.16.2+test", "model": "faked", "base_url": "", "flags": {k: "true" for k in R.FLAGS},
           "cases": rows}
    before = copy.deepcopy(rows)
    before[0]["analysis"]["lines"][0]["articolo"] = "1453 c.c."         # the baseline led elsewhere
    before[0]["fit"]["fit_si_verified"] = 1
    before[0]["seconds"] = 41.2
    before[1]["analysis"]["esiti"]["640 c.p."][0][1] = "da_provare"      # 79.16.1's pertinence downgrade
    before[1]["analysis_counters"]["analysis_provato_no_pertinence"] = 1
    before = [b for b in before if b["id"] != "fixture_civil_defence"] + [{"id": "gone_case", "seconds": 1.0}]
    diff = R.diff_rows(rows, before)
    by_id = {d["id"]: d for d in diff}
    civil = {c["key"]: c for c in by_id["fixture_civil_claimant"]["changes"]}
    assert civil["analysis.linea_principale"]["before"].startswith("1453 c.c.")
    assert civil["analysis.linea_principale"]["now"].startswith("1385 c.c.")
    assert civil["fit.fit_si_verified"] == {"key": "fit.fit_si_verified", "before": 1, "now": 4}
    assert civil["seconds"]["before"] == 41.2 and civil["seconds"]["now"] == 0.0
    penal = {c["key"]: c for c in by_id["fixture_penal_victim"]["changes"]}
    assert penal["analysis.esiti.640 c.p."]["before"].startswith("A640cp:E1=da_provare")
    assert penal["analysis_counters.analysis_provato_no_pertinence"] == {
        "key": "analysis_counters.analysis_provato_no_pertinence", "before": 1, "now": 0}
    assert by_id["fixture_civil_defence"]["new"] and by_id["gone_case"]["gone"]
    text = R.report(run, diff, "before.json")
    assert text.startswith("# replay_cases — version 79.16.2+test — model faked")
    assert "### fixture_civil_claimant" in text and "| retrieval_quesiti | queries 3, source frame, reason —, added [2946 c.c.] |" in text
    assert "- analysis.linea_principale: 1453 c.c." in text and "→ 1385 c.c. A1385cc:E4 verificata" in text
    assert "**fixture_civil_defence**: new (not in the baseline)" in text and "**gone_case**: in the baseline only" in text
    print("\n@@REPORT@@\n" + text)
    (tmp_path / "before.json").write_text(json.dumps({"cases": before}), encoding="utf-8")
    assert R.read_baseline(str(tmp_path / "before.json"))["cases"][0]["id"] == "fixture_civil_claimant"
    with pytest.raises(ValueError):
        (tmp_path / "bad.json").write_text("[1]", encoding="utf-8")
        R.read_baseline(str(tmp_path / "bad.json"))


def test_the_six_synthetic_cases_are_in_place():
    cases = R.load_cases(R.CASES_DIR)
    assert [c[0] for c in cases] == ["civ_attore_appalto_vizi_quesiti", "civ_attore_caparra_quesiti",
                                     "civ_convenuto_opposizione_decreto", "lav_lavoratore_licenziamento",
                                     "pen_indagato_appropriazione", "pen_persona_offesa_truffa"]
    assert all(len(text) > 400 for _, text in cases)
    assert [c[0] for c in R.load_cases(R.CASES_DIR, ["pen_indagato_appropriazione"])] == ["pen_indagato_appropriazione"]
    numbered = [t for _, t in cases if "1)" in t or "1." in t]
    assert len(numbered) >= 4                                   # the claimants', the defence's and the worker's quesiti
    assert "646 c.p." in dict(cases)["pen_indagato_appropriazione"]
    last = dict(cases)["pen_persona_offesa_truffa"].split("\n")[-1]
    assert last.count("?") >= 2 and last.index("?") < len(last.rstrip()) - 1       # a question with an inner «?»


def test_the_cli_refuses_an_empty_folder_and_a_bad_baseline(tmp_path, capsys):
    assert R.main(["--env", str(tmp_path / "none.env"), "--cases", str(tmp_path)]) == 1
    assert "no case in" in capsys.readouterr().err
    (tmp_path / "b.json").write_text("{}", encoding="utf-8")
    assert R.main(["--env", str(tmp_path / "none.env"), "--cases", str(tmp_path), "--baseline", str(tmp_path / "b.json")]) == 1
    assert "not a replay_cases JSON" in capsys.readouterr().err
