"""The acceptance-set runner's scoring (tests/eval_narrative.py), offline."""

import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("eval_narrative", HERE / "eval_narrative.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


@pytest.mark.parametrize("name, article", [
    ("1434", "1434"), ("1434.0.0", "1434"), ("1434.1", "1434"), ("1434_2", "1434"),
    ("624-bis", "624-bis"), ("624bis", "624-bis"), ("624-bis.0.0", "624-bis"),
    ("Art. 52", "52"), ("Articolo 5", "5"), ("art.1418", "1418"),
    ("14340", "14340"), ("2_3", "2"), ("Premessa", None), ("", None),
])
def test_section_names_map_to_their_article(name, article):
    assert ev.art_of(name) == article


def test_1434_never_matches_14340():
    case = {"expected": [{"code": "cc", "article": "1434"}]}
    citations = [{"document_name": "Codice Civile 2026", "sections": [{"name": "14340"}]}]
    assert ev.score(case, "", citations)["expected"][0]["status"] == "MISSING"


@pytest.mark.parametrize("document_name, code", [
    ("Codice Penale 2026", "cp"), ("Codice Penale Commentato ADE", None),
    ("Codice di procedura penale 2026", "cpp"), ("Codice di procedura civile Edizione 2026", "cpc"),
    ("Codice Civile 2026", "cc"), ("Codice del Consumo", "consumo"),
    ("Legge 27 luglio 1978, n. 392", "l392"), ("Ordinanza sul ricorso iscritto al n. 23908/2022 R.G.", None),
    ("Codice del processo amministrativo 2025", "cpa"),
    ("Legge 7 agosto 1990, n. 241 - Nuove norme in materia di procedimento amministrativo", "l241"),
    ("Legge sul Procedimento Amministrativo 2026", "l241"),
])
def test_citation_documents_map_to_codes(document_name, code):
    assert ev.code_of_document(document_name) == code


def test_cpa_is_not_mistaken_for_cp_in_the_answer_text():
    """"c.p.a." must resolve to cpa, not cp (held-out h10/h11, 19 Oct)."""
    answer = "L'autorità deve dare esecuzione ex art. 112 c.p.a. e art. 114 c.p.a."
    assert ev.mentions(answer) == [("112", "cpa"), ("114", "cpa")]


def test_articles_named_in_the_answer_keep_their_code():
    answer = ("Si applicano l'art. 584 c.p. e gli artt. 581 e 582 del codice penale; in subordine "
              "l'articolo 589 cp. Il locatore agisce ex art. 658 c.p.c. e art. 5 L. 392/1978; "
              "vedi anche l'art. 1453 c.c. e l'art. 43.")
    assert ev.mentions(answer) == [
        ("584", "cp"), ("581", "cp"), ("582", "cp"), ("589", "cp"),
        ("658", "cpc"), ("5", "l392"), ("1453", "cc"), ("43", None),
    ]


def test_each_expected_article_gets_a_status():
    case = {
        "expected": [{"code": "cp", "article": "584"}, {"code": "cp", "article": "581"},
                     {"code": "cp", "article": "43"}, {"code": "cp", "article": "589"},
                     {"code": "consumo", "any_of": ["128", "130"], "label": "Cod. Consumo 128-135"}],
        "forbidden_cited": [{"code": "cp", "article": "116", "why": "single agent"}],
        "forbidden_named": [{"pattern": "gelli", "why": "not in the corpus"}],
    }
    answer = "L'art. 584 c.p., l'art. 581 c.p. e l'art. 43; la legge Gelli-Bianco; art. 999 c.c."
    citations = [
        {"document_name": "Codice Penale 2026", "sections": [{"name": "584.0.0"}, {"name": "116.0.0"}]},
        {"document_name": "Codice del Consumo", "sections": [{"name": "130"}]},
        {"document_name": "Codice Penale Commentato ADE", "sections": [{"name": "589"}]},   # commentary, not c.p.
    ]
    result = ev.score(case, answer, citations)
    assert [(e["expected"], e["status"]) for e in result["expected"]] == [
        ("art. 584 cp", "CITED"), ("art. 581 cp", "NAMED-ONLY"), ("art. 43 cp", "NAMED-BARE"),
        ("art. 589 cp", "MISSING"), ("Cod. Consumo 128-135", "CITED"),
    ]
    assert result["named_not_cited"] == ["art. 581 cp", "art. 43 (no code)", "art. 999 cc"]
    assert result["failures"] == ["FAIL cited art. 116 cp (single agent)", "FLAG named /gelli/ (not in the corpus)"]


def test_expected_documents_are_checked():
    case = {"expected": [], "expected_docs": [{"doc": "commentat", "label": "penal commentary"}]}
    citations = [{"document_name": "Codice Penale Commentato ADE", "sections": [{"name": "x"}]}]
    assert ev.score(case, "", citations)["expected"] == [{"expected": "penal commentary", "status": "CITED"}]


def test_the_cases_file_is_complete():
    cases = json.loads((HERE / "eval_narrative_cases.json").read_text(encoding="utf-8"))["cases"]
    ids = [c["id"] for c in cases if not c.get("extra") and not c.get("heldout")]
    assert ids == ["sempronio", "p1", "p2", "p3", "c1", "c2", "c3", "l1", "l2", "l3", "l4", "cp_short"]
    for case in cases:
        assert case["question"].strip()
        for item in case["expected"]:
            assert item["code"] in ev.CODES


def test_the_held_out_set_is_complete_and_separate_from_the_tuning_set():
    """h1-h11: real exam-style cases from a practising lawyer, held out from
    the tuning set above -- provisional expected articles, never themselves
    used to tune behaviour (see the module docstring and each case's own
    "expected_status")."""
    cases = json.loads((HERE / "eval_narrative_cases.json").read_text(encoding="utf-8"))["cases"]
    heldout = [c for c in cases if c.get("heldout")]
    assert [c["id"] for c in heldout] == [f"h{n}" for n in range(1, 12)]
    for case in heldout:
        assert not case.get("extra")   # heldout and extra are different, non-overlapping concepts
        assert case.get("expected_status") == "draft — pending lawyer validation"
        assert case["question"].strip()
        assert case["expected"]
        for item in case["expected"]:
            assert item["code"] in ev.CODES


def test_runs_can_be_compared(tmp_path, capsys):
    def run(status):
        return {"results": [{"id": "c2", "score": {"expected": [{"expected": "art. 1490 cc", "status": status}],
                                                   "failures": []}}]}
    (tmp_path / "a.json").write_text(json.dumps(run("MISSING")), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(run("CITED")), encoding="utf-8")
    ev.compare(tmp_path / "a.json", tmp_path / "b.json")
    assert "art. 1490 cc: MISSING -> CITED" in capsys.readouterr().out


def test_aggregate_reports_cited_rate_per_expected_article():
    def run(repeat, status_584, status_581):
        return {"id": "sempronio", "repeat": repeat,
                "score": {"expected": [{"expected": "art. 584 cp", "status": status_584},
                                       {"expected": "art. 581 cp", "status": status_581}]}}
    results = [run(0, "CITED", "CITED"), run(1, "CITED", "MISSING"), run(2, "CITED", "CITED")]
    per_case, overall_avg = ev._aggregate(results, 3)
    assert per_case["sempronio"]["counts"] == {"art. 584 cp": 3, "art. 581 cp": 2}
    assert per_case["sempronio"]["n"] == 3
    assert overall_avg == pytest.approx((3 + 2) / 3)


def test_aggregate_excludes_errored_runs_from_the_count():
    results = [
        {"id": "c2", "score": {"expected": [{"expected": "art. 1490 cc", "status": "CITED"}]}},
        {"id": "c2", "error": "TimeoutError: ..."},
    ]
    per_case, _ = ev._aggregate(results, 2)
    assert per_case["c2"]["n"] == 1
    assert per_case["c2"]["errors"] == 1
    assert per_case["c2"]["counts"] == {"art. 1490 cc": 1}


def test_aggregate_lines_shows_counts_as_a_fraction():
    per_case = {"sempronio": {"counts": {"art. 584 cp": 3, "art. 581 cp": 2}, "n": 3, "errors": 0}}
    lines = "\n".join(ev._aggregate_lines(per_case, 2.5, 3))
    assert "3/3  art. 584 cp" in lines
    assert "2/3  art. 581 cp" in lines
    assert "OVERALL average CITED per case: 2.50" in lines


def test_aggregate_reports_average_seconds_per_case():
    def run(repeat, seconds):
        return {"id": "sempronio", "repeat": repeat, "seconds": seconds,
                "score": {"expected": [{"expected": "art. 584 cp", "status": "CITED"}]}}
    results = [run(0, 10.0), run(1, 20.0), run(2, 30.0)]
    per_case, _ = ev._aggregate(results, 3)
    assert per_case["sempronio"]["avg_seconds"] == pytest.approx(20.0)


def test_aggregate_lines_shows_average_seconds_and_an_overall_average():
    per_case = {
        "sempronio": {"counts": {"art. 584 cp": 3}, "n": 2, "errors": 0, "avg_seconds": 10.0},
        "c2": {"counts": {"art. 1490 cc": 1}, "n": 2, "errors": 0, "avg_seconds": 30.0},
    }
    lines = "\n".join(ev._aggregate_lines(per_case, 2.0, 3))
    assert "avg 10.0 s" in lines
    assert "avg 30.0 s" in lines
    assert "overall average 20.0 s per run" in lines


# --- Held-out reporting (19 Oct): its own section, its own average, never
# mixed into the tuning set's -------------------------------------------------

def _result(case_id, status="CITED", heldout=False, seconds=1.0):
    return {"id": case_id, "heldout": heldout, "seconds": seconds,
            "score": {"expected": [{"expected": f"art. 1 {case_id}", "status": status}],
                     "failures": [], "named_not_cited": []}}


def test_summary_lines_separates_heldout_into_its_own_section():
    results = [_result("sempronio"), _result("h1", heldout=True)]
    lines = "\n".join(ev._summary_lines(results))
    assert "## sempronio" in lines
    assert "HELD-OUT SET" in lines
    assert "## h1" in lines
    # The held-out case comes after the section header, not mixed into the tuning set.
    assert lines.index("## sempronio") < lines.index("HELD-OUT SET") < lines.index("## h1")


def test_summary_lines_reports_its_own_average_per_group():
    results = [_result("sempronio", status="CITED"), _result("c2", status="MISSING"),
              _result("h1", status="CITED", heldout=True)]
    lines = "\n".join(ev._summary_lines(results))
    # Tuning set: (1 + 0) / 2 = 0.50; held-out: 1/1 = 1.00 -- reported separately.
    assert "average CITED per case: 0.50 (across 2 case(s))" in lines
    assert "average CITED per case: 1.00 (across 1 case(s))" in lines


def test_summary_lines_has_no_heldout_section_when_there_is_none():
    lines = "\n".join(ev._summary_lines([_result("sempronio")]))
    assert "HELD-OUT" not in lines


def _http_error(code, detail, headers=None):
    import email.message
    import io
    import urllib.error
    msg = email.message.Message()
    for key, value in (headers or {}).items():
        msg[key] = value
    return urllib.error.HTTPError("http://x/api/auth/login", code, "Unauthorized", msg,
                                  io.BytesIO(json.dumps({"detail": detail}).encode()))


def test_login_asks_for_the_2fa_code_when_the_account_needs_it(monkeypatch):
    sent = []

    def fake_post(url, payload, token=None, timeout=60):
        sent.append(dict(payload))
        if "totp_code" not in payload:
            raise _http_error(401, "2FA code required", {"X-2FA-Required": "true"})
        return {"access_token": "jwt"}

    monkeypatch.setattr(ev, "_post", fake_post)
    monkeypatch.setenv("EVAL_EMAIL", "eval@example.com")
    monkeypatch.setenv("EVAL_PASSWORD", "pw")
    monkeypatch.delenv("EVAL_TOTP", raising=False)
    monkeypatch.setattr("builtins.input", lambda prompt: " 123456 ")
    assert ev.login("http://x") == "jwt"
    assert sent[-1]["totp_code"] == "123456"


def test_login_failure_shows_the_server_reason(monkeypatch):
    def fake_post(url, payload, token=None, timeout=60):
        raise _http_error(401, "Incorrect email or password")

    monkeypatch.setattr(ev, "_post", fake_post)
    monkeypatch.setenv("EVAL_EMAIL", "eval@example.com")
    monkeypatch.setenv("EVAL_PASSWORD", "wrong")
    with pytest.raises(SystemExit) as exit_info:
        ev.login("http://x")
    assert "Incorrect email or password" in str(exit_info.value)


def test_heldout_only_and_include_heldout_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        ev.main(["--heldout-only", "--include-heldout", "--base-url", "http://x"])


def test_heldout_only_selects_just_the_held_out_cases(monkeypatch, tmp_path):
    """--heldout-only runs h1-h11 and nothing else -- doesn't need a live
    server: every case errors immediately (fake login, no HTTP), so this
    only checks which cases were attempted, via the printed "... <id>" lines."""
    monkeypatch.setattr(ev, "login", lambda base_url: "jwt")

    def fake_ask(base_url, token, question, timeout):
        raise TimeoutError("no server in this test")
    monkeypatch.setattr(ev, "ask", fake_ask)

    printed = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: printed.append(" ".join(str(x) for x in a)))

    ev.main(["--heldout-only", "--base-url", "http://x", "--out", str(tmp_path)])
    attempted = {line.split()[1] for line in printed if line.startswith("... ")}
    assert attempted == {f"h{n}" for n in range(1, 12)}
