"""scripts/trace_ruling.py: one ruling traced through the pipeline, read-only (the PM's tracing request, 4 Nov)."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("trace_ruling", HERE.parent / "scripts" / "trace_ruling.py")
tr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tr)


def test_parse_number_and_matching():
    assert tr.parse_number("14172/2026") == ("14172", "2026")
    assert tr.parse_number("n. 7075 del 2026") == ("7075", "2026")
    assert tr.parse_number("014172") == ("14172", None)
    assert tr.number_matches("14172", "14172") and tr.number_matches("014172", "14172")
    assert tr.number_matches("n. 14172/2026", "14172") and not tr.number_matches("114172", "14172")
    assert not tr.number_matches("1417", "14172") and not tr.number_matches(None, "14172")


def test_year_and_stem_from_the_file_name():
    assert tr.year_of({"file_name": "snpen@s10@a2025@n00001@tS.clean.pdf"}) == "2025"
    assert tr.year_of({"name": "[CIVILE] snciv s10 a2026 n00016 tO.clean"}) == "2026"
    assert tr.year_of({"date_enacted": "2026-01-21"}) == "2026" and tr.year_of({}) is None
    assert tr.stem_of("/x/snpen_s10_a2025_n00001_tS.clean.pdf") == "snpen@s10@a2025@n00001@tS"
    assert tr.stem_of("snpen@s10@a2025@n00001@tS") == "snpen@s10@a2025@n00001@tS"


def test_find_records_reads_only_the_documents_lines(tmp_path):
    path = tmp_path / "cards.jsonl"
    path.write_text(json.dumps({"id": "A", "esito": "rigetto"}) + "\n" + "not json A\n"
                    + json.dumps({"id": "AB", "esito": "x"}) + "\n" + json.dumps({"id": "A", "errore": "json"}) + "\n",
                    encoding="utf-8")
    recs = tr.find_records(str(path), "A")
    assert [r.get("esito", r.get("errore")) for r in recs] == ["rigetto", "json"]
    assert tr.find_records(str(tmp_path / "missing.jsonl"), "A") == []


def test_card_summary_never_carries_the_quote():
    card = {"id": "A", "prompt_version": "cards-v3", "modello": "qwen3-32b-awq", "esito": "annullamento con rinvio",
            "a_favore_di": "imputato", "ricorrente": "imputato", "articoli_applicati": ["612-bis c.p.", "131-bis c.p."],
            "parole_chiave_fatti": ["messaggi"], "citazione": "Il testo della sentenza che non va stampato",
            "citazione_verificata": True, "citazione_scartata": "", "completo": True}
    out = tr.card_summary(card)
    assert out["modello"] == "qwen3-32b-awq" and out["articoli_applicati"] == ["612-bis c.p.", "131-bis c.p."]
    assert out["citazione"] == f"{len(card['citazione'])} chars, verified True"
    assert "sentenza" not in json.dumps(out, ensure_ascii=False) and "citazione_scartata" not in out


def test_cites_lines_group_by_article_with_how_and_rule():
    rows = [{"code": "c.p.", "article": "612-bis", "how": "explicit", "rule": "past_lists", "law": "Codice Penale 2026", "n": 1},
            {"code": "c.p.", "article": "612-bis", "how": "explicit", "rule": "plain", "law": "Codice Penale 2026", "n": 2},
            {"code": "c.p.p.", "article": "380", "how": "explicit", "rule": "plain", "law": "Codice di procedura penale 2026", "n": 1}]
    lines = tr.cites_lines(rows)
    assert lines[0] == "art. 612-bis c.p. (Codice Penale 2026): 3 links, how explicit 3; rule plain 2, past_lists 1"
    assert lines[1].startswith("art. 380 c.p.p. (Codice di procedura penale 2026): 1 link, ")


def test_names_are_printed_without_the_parties():
    assert tr.safe_name("Sentenza della Corte di Cassazione sul ricorso proposto da Mario Rossi") ==         "Sentenza della Corte di Cassazione [...]"
    assert tr.safe_name("Ordinanza Civile Sez. 3 Num. 10586 Anno 2026") == "Ordinanza Civile Sez. 3 Num. 10586 Anno 2026"
    assert tr.safe_name("Sentenza nei confronti di Tizio Caio") == "Sentenza [...]" and tr.safe_name(None) == ""
    assert tr.branch_of({"file_name": "snpen@s10@a2026@n14172@tS.clean.pdf"}) == "penale"
    assert tr.branch_of({"name": "[CIVILE] snciv s10 a2026 n00016 tO.clean"}) == "civile" and tr.branch_of({}) == "?"

