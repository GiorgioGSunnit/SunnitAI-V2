"""v79.11 block D -- measurement in the nightly (scripts/nightly_sim.py), synthetic result dicts and files only:

D1. --element-schemas sets ELEMENT_SCHEMAS for the run and the mode says so; without it, nothing set, the mode as before
D2. run_cell's copies: state["element_paths"] and the shadow admission (state, thesis counts, per client)
D3. the report: the v79.11 headline («before → now» side by side), the element paths, the shadow admission (agree,
    false exclusions with reasons, extra, recall) -- a cell or a baseline without the new keys renders
D4. coverage by code from a tmp schema file against a tmp cards file; quality over a tmp reviewed overlay («nessuna
    revisione» when none)
"""

import importlib.util
import json
import os
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for name, value in (("LLM_API_KEY", "k"), ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "k"), ("NEO4J_URI", "bolt://localhost:7687"), ("NEO4J_USER", "n"),
                        ("NEO4J_PASSWORD", "x")):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("ARTICLE_SCHEMAS_FILE", str(tmp_path / "schemas.jsonl"))
    monkeypatch.setenv("ARTICLE_SCHEMAS_REVIEWED_FILE", str(tmp_path / "reviewed.jsonl"))
    monkeypatch.setenv("ARTICLE_CARDS_FILE", str(tmp_path / "cards.jsonl"))


def _nightly():
    spec = importlib.util.spec_from_file_location("nightly_sim", ROOT / "scripts" / "nightly_sim.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TEXT = "Chiunque cagiona ad alcuno una lesione personale, dalla quale deriva una malattia nel corpo o nella mente, è punito."


def _schema():
    return {"tipo": ["sanction"], "destinatario": [{"ruolo": "chiunque", "condizione": None}],
            "legittimato": [{"ruolo": "persona offesa", "condizione": None}], "rapporto": "lesioni personali",
            "elementi": [{"id": "E1", "testo": "cagiona ad alcuno una lesione personale", "kind": "condotta",
                          "comma": "1", "branches": [], "connective": None, "grounding": "explicit",
                          "fonte": {"comma": "1", "testo": "cagiona ad alcuno una lesione personale"}}],
            "logic": {"op": "all_of", "args": [{"ref": "E1"}]}, "eccezioni": [], "commi_speciali": [], "collegati": []}


def _write(path, lines):
    pathlib.Path(path).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines), encoding="utf-8")


# --- D1 -------------------------------------------------------------------------------------------------------------------

def _main(nightly, monkeypatch, tmp_path, *extra):
    monkeypatch.setattr(nightly, "cells", lambda: [])
    monkeypatch.setattr(nightly, "stage_check", lambda: None)
    out = tmp_path / "n.json"
    nightly.main(["--env", "/nonexistent", "--no-stages", "--json-out", str(out), *extra])
    return json.loads(out.read_text(encoding="utf-8"))


def test_the_switch_sets_the_flag_for_the_run_and_the_mode_says_so(monkeypatch, tmp_path):
    nightly = _nightly()
    monkeypatch.setenv("ELEMENT_SCHEMAS", "false")         # restored after the test, whatever main() sets
    got = _main(nightly, monkeypatch, tmp_path, "--element-schemas")
    assert os.environ["ELEMENT_SCHEMAS"] == "true"
    assert got["mode"] == "best-of-3; element schemas on (ELEMENT_SCHEMAS)"


def test_without_the_switch_nothing_is_set_and_the_mode_is_unchanged(monkeypatch, tmp_path):
    nightly = _nightly()
    got = _main(nightly, monkeypatch, tmp_path)
    assert "ELEMENT_SCHEMAS" not in os.environ and got["mode"] == "best-of-3"


def test_the_switch_adds_coverage_and_quality_to_the_report(monkeypatch, tmp_path, capsys):
    nightly = _nightly()
    monkeypatch.setenv("ELEMENT_SCHEMAS", "false")
    _main(nightly, monkeypatch, tmp_path, "--element-schemas")
    text = capsys.readouterr().out
    assert "## Element schemas: coverage (by code; v79.11)" in text and "nessuna revisione" in text


def test_a_failing_section_is_a_line_not_the_end_of_the_run(monkeypatch):
    nightly = _nightly()

    def boom(*a, **k):
        raise RuntimeError("broken file")
    monkeypatch.setattr(nightly, "schema_coverage_lines", boom)
    got = nightly.schema_sections()
    assert any("not computed (RuntimeError: broken file)" in x for x in got)
    assert any("Element schemas: quality" in x for x in got)


# --- D2 -------------------------------------------------------------------------------------------------------------------

SHADOW = {"items": [], "current": {"claim": ["2043 c.c."], "contest": ["582 c.p."]},
          "schema": {"claim": ["2043 c.c."], "defence": ["52 c.p."]}, "agree": 1,
          "false_exclusions": [["582 c.p.", "contest", ["contest: norma del cliente, non della controparte"]]],
          "extra": [["52 c.p.", "defence"]], "no_schema": ["1218 c.c."]}


def test_shadow_of_reads_the_state_then_the_thesis_counts_then_each_client():
    nightly = _nightly()
    assert nightly.shadow_of({}) == {}
    assert nightly.shadow_of({"schema_shadow": SHADOW}) == {"schema_shadow": SHADOW}
    assert nightly.shadow_of({"thesis_counts": {"schema_shadow": SHADOW}}) == {"schema_shadow": SHADOW}
    got = nightly.shadow_of({"thesis_counts_by_client": {"Tizio": {"schema_shadow": SHADOW}, "Caio": {}}})
    assert got == {"schema_shadow_by_client": {"Tizio": SHADOW}}


def test_run_cell_copies_the_element_paths_and_the_shadow(monkeypatch):
    nightly = _nightly()
    from src.rag import interactive_flow, lawyer_method
    paths = {"582 c.p.": {"path": "schema", "why": "schema"}, "2043 c.c.": {"path": "text", "why": "no_schema"}}
    state = {"element_paths": paths, "schema_shadow": SHADOW, "intake_source": "request"}
    monkeypatch.setattr(interactive_flow, "request_case_first", lambda *a, **k: {
        "intake": {"rito": "civile", "role": "attore", "stage": "pre_filing"}, "answer": "", "state": state,
        "api_route": "case_first"})
    monkeypatch.setattr(lawyer_method, "structure_checks", lambda *a, **k: [])
    out = nightly.run_cell({"id": "c1", "cell": {"family": "civile", "rito": "civile", "role": "attore",
                                                  "stage": "pre_filing"}, "text": "Un caso sintetico."})
    assert out["element_paths"] == paths and out["schema_shadow"] == SHADOW


# --- D3 -------------------------------------------------------------------------------------------------------------------

NOW = [{"id": "c1", "checks": [],
        "counters": {"elements_per_article": 3.5, "fragments_in_tables": 0, "mental_integrated_by_conduct": 0,
                     "duty_polarity_errors": 0, "unadmitted_candidates": 0, "main_lines_by_rule": {"remedy": 1},
                     "elements_per_article_by_path": {"schema": 4.0, "text": 3.0},
                     "schema_shadow": {"agree": 1, "false_exclusions": 1, "extra": 1, "candidate_recall": 0.5}},
        "element_paths": {"582 c.p.": {"path": "schema", "why": "schema"},
                          "2043 c.c.": {"path": "text", "why": "no_schema"},
                          "1218 c.c.": {"path": "text", "why": "hash_mismatch"}},
        "schema_shadow": SHADOW}]
BASE = {"version": "79.9.3", "cells": [{"id": "c1", "checks": [],
                                        "counters": {"fragments_in_tables": 2, "mental_integrated_by_conduct": 1,
                                                     "duty_polarity_errors": 0, "unadmitted_candidates": 0}}]}


def test_the_headline_puts_before_and_now_side_by_side():
    nightly = _nightly()
    text = "\n".join(nightly.headline_lines(NOW, BASE))
    assert "## v79.11 headline" in text
    assert "| c1 | — → 3.5 | 2 → 0 | 1 → 0 | 0 → 0 | 0 → 0 | — → remedy: 1 | — → 0.5 | — → 1 |" in text


def test_the_headline_without_a_baseline_shows_now_only():
    nightly = _nightly()
    text = "\n".join(nightly.headline_lines(NOW))
    assert "| c1 | 3.5 | 0 | 0 | 0 | 0 | remedy: 1 | 0.5 | 1 |" in text


def test_the_headline_falls_back_on_what_the_cell_kept():
    nightly = _nightly()
    cell = {"id": "c2", "checks": [], "counters": {"fragments_in_tables": 0},
            "thesis_counts": {"main_admitted_by": "claim_admission"}, "schema_shadow": SHADOW}
    assert nightly.headline_value(cell, "main_lines_by_rule") == {"claim_admission": 1}
    assert nightly.headline_value(cell, "false_exclusions") == 1
    # current: 2043 claim, 582 contest (both with a schema); the schema admits 2043 claim only
    assert nightly.headline_value(cell, "candidate_recall") == 0.5
    two = {"id": "c3", "thesis_counts_by_client": {"A": {"main_admitted_by": "remedy"}, "B": {"main_admitted_by": ""}}}
    assert nightly.headline_value(two, "main_lines_by_rule") == {"remedy": 1, "—": 1}


def test_cells_without_the_new_keys_render_and_add_no_section():
    nightly = _nightly()
    old = [{"id": "c1", "checks": [{"check": "version_footer", "ok": True, "why": ""}]}]
    assert nightly.headline_lines(old, BASE) == [] and nightly.schema_lines(old) == []
    text = nightly.report(old, None, [], "79.11.0", [], BASE, "best-of-3")
    assert "v79.11" not in text and "Element paths" not in text
    # a baseline cell without the new keys: «—» before the arrow, never an error
    assert "— → 3.5" in "\n".join(nightly.headline_lines(NOW, {"cells": [{"id": "c1"}]}))


def test_schema_lines_by_path_and_the_shadow_summary():
    nightly = _nightly()
    text = "\n".join(nightly.schema_lines(NOW))
    assert "| c1 | 1 | 0 | 2 | 4.0 · — · 3.0 | no_schema: 1, hash_mismatch: 1 |" in text
    assert "- c1: schema: 582 c.p.; text: 2043 c.c. (no_schema), 1218 c.c. (hash_mismatch)" in text
    assert "| c1 | 1 | 1 | 1 | 0.5 | 1 |" in text
    assert "582 c.p. (contest): contest: norma del cliente, non della controparte" in text
    assert "Extra (admitted by the schema only): c1: 52 c.p. (defence)" in text


def test_schema_lines_from_the_counters_alone_and_per_client():
    nightly = _nightly()
    cells = [{"id": "c1", "counters": {"element_paths": {"schema": 2, "text": 1}, "schema_rejected": {"no_schema": 1}}},
             {"id": "c2", "schema_shadow_by_client": {"Tizio": {"agree": [1, 2], "false_exclusions": [],
                                                                "candidate_recall": None}}}]
    text = "\n".join(nightly.schema_lines(cells))
    assert "| c1 | 2 | 0 | 1 | — · — · — | no_schema: 1 |" in text
    assert "| c2 / Tizio | 2 | 0 | 0 | — | 0 |" in text and "False exclusions" not in text


def test_schema_lines_on_the_selection_counts_shape():
    """v79.11 review: the thesis counts keep selection_counts' numbers (no lists) -- the shadow section renders them,
    the recall from recall_hit / recall_of, and the report does not fail."""
    nightly = _nightly()
    counts = {"agree": 5, "false_exclusions": 2, "extra": 1, "recall_hit": 3, "recall_of": 5}
    cell = {"id": "c1", "checks": [], **nightly.shadow_of({"thesis_counts": {"schema_shadow": counts},
                                             "thesis_counts_by_client": {"Tizio": {"schema_shadow": counts}}})}
    text = "\n".join(nightly.schema_lines([cell]))
    assert "| c1 | 5 | 2 | 1 | 0.6 | 0 |" in text and "| c1 / Tizio | 5 | 2 | 1 | 0.6 | 0 |" in text
    assert "False exclusions" not in text and "Extra" not in text           # their reasons are not in the counts
    assert nightly.headline_value(cell, "false_exclusions") == 2 and nightly.headline_value(cell, "candidate_recall") == 0.6
    assert nightly.shadow_recall({"recall_hit": 0, "recall_of": 0}) is None
    assert "## Shadow admission" in nightly.report([cell], None, [], "79.11.0", [], BASE, "best-of-3", [])


def test_the_state_keeps_the_whole_shadow_for_the_nightly(monkeypatch):
    """v79.11 review: lawyer_method.schema_shadow -> interactive_flow.shadow_state (the state, the clients') -> the
    nightly's false exclusions with their reasons."""
    nightly = _nightly()
    from src.rag import interactive_flow, lawyer_method
    monkeypatch.setenv("ELEMENT_SCHEMAS", "true")
    summary = {"source": "schema", "tipo": ["limiting"], "reviewed": False, "rapporto": "",
               "legittimato": [{"ruolo": "attore", "condizione": None}]}
    ctx = {"posture": {"role": "attore"}, "records": {"1218 c.c.": {"side": "weapon", "schema": summary}},
           "candidates": [{"id": "T1", "kind": "claim", "article": "1218 c.c.", "admitted_by": "generale"}]}
    shadow = lawyer_method.schema_shadow(ctx)
    state = interactive_flow.shadow_state(shadow)
    assert "items" not in state and state["false_exclusions"][0][:2] == ["1218 c.c.", "claim"]
    cell = {"id": "c1", **nightly.shadow_of({"schema_shadow": state, "thesis_counts": {"schema_shadow": {"agree": 0}},
                                             "schema_shadow_by_client": {"Tizio": state}})}
    assert cell["schema_shadow"] == state and cell["schema_shadow_by_client"] == {"Tizio": state}
    text = "\n".join(nightly.schema_lines([cell]))
    assert "c1: 1218 c.c. (claim): claim:" in text and "c1 / Tizio: 1218 c.c. (claim)" in text


def test_the_report_carries_the_new_sections_in_order():
    nightly = _nightly()
    text = nightly.report(NOW, None, [], "79.11.0", [], BASE, "best-of-3", ["", "## Element schemas: coverage"])
    assert text.index("## Counters") < text.index("## v79.11 headline") < text.index("## Element paths") \
        < text.index("## Shadow admission") < text.index("## Element schemas: coverage")


# --- D4 -------------------------------------------------------------------------------------------------------------------

def _line(doc, article, code, schema, **checks):
    from src.rag import element_schemas as es
    return {"doc": doc, "article": article, "code": code, "label": f"{article} {code}", "schema_version": 1,
            "source_hash": es.text_hash(TEXT), "schema": schema, "checks": {**es.empty_checks(), **checks}}


def test_coverage_by_code_from_tmp_files(tmp_path):
    nightly = _nightly()
    _write(tmp_path / "cards.jsonl", [{"doc": "Codice Penale", "article": "582", "card": "x"},
                                      {"doc": "Codice Penale", "article": "583", "card": "x"},
                                      {"doc": "Codice Penale", "article": "584", "card": "x"},
                                      {"doc": "Codice Civile", "article": "2043", "card": "x"},
                                      {"doc": "Codice Civile", "article": "2043", "card": "y"}])
    _write(tmp_path / "schemas.jsonl", [_line("Codice Penale", "582", "c.p.", _schema()),
                                        _line("Codice Penale", "583", "c.p.", _schema(),
                                              logic_errors=["logic.all_of[1]: ref E9 senza elemento"]),
                                        _line("Codice Civile", "1218", "c.c.", _schema())])
    _write(tmp_path / "reviewed.jsonl", [{"label": "582 c.p.", "reviewed": True, "schema": _schema()},
                                         {"label": "584 c.p.", "reviewed": False, "note_di_inquadramento": []}])
    text = "\n".join(nightly.schema_coverage_lines())
    assert "| c.c. | 1 | 0 | 1 | 0 |" in text          # 1218 generated (not carded), 2043 carded without
    assert "| c.p. | 1 | 1 | 1 | 1 |" in text          # 582 usable and reviewed, 583 unusable, 584 without
    assert "| all | 2 | 1 | 2 | 1 |" in text
    assert "Unusable by blocking check: logic_errors 1" in text      # v79.12.2: ungrounded no longer blocks


def test_coverage_with_no_files_is_an_empty_row(tmp_path):
    nightly = _nightly()
    text = "\n".join(nightly.schema_coverage_lines(str(tmp_path / "none.jsonl")))
    assert "| — | 0 | 0 | 0 | 0 |" in text


def test_quality_without_reviews_says_nessuna_revisione():
    nightly = _nightly()
    text = "\n".join(nightly.quality_lines())
    assert "nessuna revisione" in text
    for k in nightly.QUALITY_MEASURES:
        assert f"| {k} |" in text
    assert "| element_precision | — |" in text


def test_quality_over_a_tmp_reviewed_overlay(tmp_path):
    nightly = _nightly()
    gold = _schema()
    pred = _schema()
    pred["elementi"][0]["comma"] = "2"
    _write(tmp_path / "reviewed.jsonl", [{"label": "582 c.p.", "reviewed": True, "schema": gold,
                                          "generated_schema": pred, "corrections": [{"field": "E1"}]}])
    text = "\n".join(nightly.quality_lines())
    assert "nessuna revisione" not in text
    assert "| n | 1 |" in text and "| element_precision | 1.0 |" in text and "| comma_accuracy | 0.0 |" in text
    assert "| correction_rate | 1.0 |" in text
