"""v60 (the v59-all server run's lint): a table's own name over the sub-name its anchors give, a matched article the
writer skips keeps its comma line, chosen topics all in c.p./c.p.p. make §4, the lint names a run by its directory."""

import os
import sys
from pathlib import Path

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import answer_lint  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import render_computed, rule_quote, topic_step, topic_tables  # noqa: E402

ART_585 = ("Nei casi previsti dagli articoli 582, 583, 583-bis, 583-quinquies e 584, la pena è aumentata da un terzo alla "
           "metà, se concorre alcuna delle circostanze aggravanti prevedute dall'articolo 576.")


def test_a_table_topic_named_by_an_anchor_pattern_takes_the_tables_own_name():
    analysis = {"frame": {"areas": ["penale"], "central_issue_candidates": [{"nome": "Custodia cautelare in carcere"}]}}
    table = next(t for t in topic_tables.matches(analysis, ["penale"]) if t["id"] == "misure_cautelari")
    assert table["name"] == "custodia cautelare in carcere" and table["table_name"] == "misure cautelari personali"
    topic = {"title": "Custodia cautelare in carcere", "table_name": "Misure cautelari personali", "merged": [],
             "refs": [("285", "c.p.p."), ("274", "c.p.p."), ("273", "c.p.p."), ("275", "c.p.p."), ("309", "c.p.p."),
                      ("286", "c.p.p."), ("657", "c.p.p.")], "planned": 7,
             "own": [("285", "c.p.p."), ("274", "c.p.p."), ("273", "c.p.p."), ("275", "c.p.p."), ("309", "c.p.p.")]}
    topic_step.named_by_table(topic)                                                            # v59-n2: 5 of 7
    assert topic["title"] == "Misure cautelari personali" and topic["merged"] == ["Custodia cautelare in carcere"]


def test_a_table_merged_under_its_sub_name_still_names_the_topic_by_the_tables_name():
    topic = {"title": "Elemento psicologico del reato", "merged": [], "facts": [], "questions": [], "planned": 7,
             "refs": [("43", "c.p."), ("55", "c.p."), ("52", "c.p."), ("575", "c.p."), ("589", "c.p."), ("42", "c.p."),
                      ("56", "c.p.")]}
    table = {"title": "Eccesso colposo di legittima difesa", "table_name": "legittima difesa ed eccesso colposo",
             "source": "tabella legittima_difesa", "facts": [], "merged": [], "questions": [],
             "own": [("55", "c.p."), ("52", "c.p."), ("575", "c.p."), ("589", "c.p.")]}
    topic_step._merge(topic, table)
    topic_step.named_by_table(topic)                                                            # v59-p1: 4 of 7
    assert topic["title"] == "Legittima difesa ed eccesso colposo"


def test_a_matched_article_the_writer_skips_keeps_its_comma_line():
    rows = [{"_article": ("585", "c.p."), "_commi": [ART_585, "Le pene sono aumentate."],
             "s": {"titolo_articolo": "Circostanze aggravanti"}}]
    rules = [{**r, "referring": [], "requisites": {}} for r in rule_quote.article_rules({}, rows, {}, [("585", "c.p.")])]
    log = []
    plain = flow.assemble_norms("", rules, {}, rows, log, [("585", "c.p.")])
    assert plain.startswith("Altre norme del tema non trattate")                                # not matched: as before
    log = []
    kept = flow.assemble_norms("", rules, {}, rows, log, [("585", "c.p.")], matched=[("585", "c.p.")])   # v59-p3
    assert kept.startswith("**Art. 585 c.p. — Circostanze aggravanti**\n- co. 1: ") and "non trattate" not in kept


def test_chosen_topics_all_in_the_criminal_codes_make_the_procedural_section():
    body = render_computed.criminal_procedure({"frame": {"areas": ["contratti"]},
                                               "topic_refs": [("43", "c.p."), ("45", "c.p.")]})          # v59-n1
    assert "Data del fatto: non nota" in body
    assert render_computed.criminal_procedure({"frame": {"areas": ["contratti"]},
                                               "topic_refs": [("43", "c.p."), ("1815", "c.c.")]}) == ""


def test_the_lint_names_a_runs_interactive_md_by_its_directory(tmp_path):
    run = tmp_path / "v60-all"
    run.mkdir()
    assert answer_lint.file_label(str(run / "interactive.md")) == "v60-all"
    assert answer_lint.file_label(str(tmp_path / "v58-h7.md")) == "v58-h7.md"
    table = answer_lint.tables([], [str(run / "interactive.md")])
    assert "| v60-all |" in table and "interactive.md" not in table
