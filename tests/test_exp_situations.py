"""scripts/exp_situations.py's pure scoring helpers, offline. The live part
(embedding the case, querying situation_embeddings_v2) needs prod and is only
exercised by running the script there."""

import importlib.util
import os
from pathlib import Path

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("exp_situations", HERE.parent / "scripts" / "exp_situations.py")
es = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(es)

ROWS = [
    {"document": "Codice Penale 2026", "article": "584", "score": 0.81},
    {"document": "Codice Penale 2026", "article": "589", "score": 0.79},
    {"document": "Codice Penale 2026", "article": "589", "score": 0.77},
    {"document": "Codice Penale 2026", "article": "589", "score": 0.70},
    {"document": "Codice Civile 2026", "article": "2043", "score": 0.75},
    {"document": "Codice del Consumo 2026", "article": "130", "score": 0.60},
    {"document": "Codice Penale Commentato ADE", "article": "584", "score": 0.99},   # not a code: skipped
    {"document": "Codice Penale 2026", "article": "", "score": 0.98},                # no article: skipped
]


def test_articles_are_ranked_by_best_score_and_by_count():
    by_best, by_hits = es.rank_articles(ROWS)
    assert by_best == [("cp", "584"), ("cp", "589"), ("cc", "2043"), ("consumo", "130")]
    assert by_hits[0] == ("cp", "589")                       # three of its situations among the nearest


def test_expected_rank_and_recall_including_any_of():
    by_best, _ = es.rank_articles(ROWS)
    expected = [{"code": "cp", "article": "584"}, {"code": "cp", "article": "43"},
                {"code": "consumo", "any_of": ["128", "130", "135"], "label": "Codice del Consumo artt. 128-135 (any)"}]
    assert [es.expected_rank(by_best, e) for e in expected] == [1, None, 4]
    assert es.recall(by_best, expected, 3) == (1, 3)
    assert es.recall(by_best, expected, 10) == (2, 3)


def test_new_finds_are_articles_the_live_pipeline_mostly_misses():
    aggregate = {"per_case": {"sempronio_no_hint": {"n": 3, "counts": {"art. 584 cp": 3, "art. 589 cp": 1,
                                                                       "art. 43 cp": 0}}}}
    rates = es.reference_rates(aggregate)["sempronio_no_hint"]
    by_best, _ = es.rank_articles(ROWS)
    expected = [{"code": "cp", "article": "584"}, {"code": "cp", "article": "589"}, {"code": "cp", "article": "43"}]
    # 584: cited every run, not new; 589: cited 1/3 and ranked 2nd, new; 43: missed but not found either.
    assert es.new_finds(expected, by_best, rates) == [("art. 589 cp", 2, 1 / 3)]


def test_fact_texts_come_from_the_structured_facts():
    extracted = {"fatti": [{"fatto": " Tizio colpisce Caio "}, {"fatto": ""}, "not a dict", {"tipo": "evento"}]}
    assert es.fact_texts(extracted) == ["Tizio colpisce Caio"]
    assert es.fact_texts(None) == []
