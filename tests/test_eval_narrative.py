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
])
def test_citation_documents_map_to_codes(document_name, code):
    assert ev.code_of_document(document_name) == code


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
    ids = [c["id"] for c in cases if not c.get("extra")]
    assert ids == ["sempronio", "p1", "p2", "p3", "c1", "c2", "c3", "cp_short"]
    for case in cases:
        assert case["question"].strip()
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
