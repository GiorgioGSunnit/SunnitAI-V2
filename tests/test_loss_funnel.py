"""scripts/loss_funnel.py's pure classification logic, offline. The script
itself calls the live pipeline once per case (a real draft generation) and
needs Neo4j + the chat model -- this file only exercises the part that
decides, from a captured fusion pool/drop list/citation set, which of the
five stages (NOT_IN_POOL / IN_POOL_CUT_BY_CAP / DROPPED_BY_CHECK /
SELECTED_NOT_CITED / CITED) an expected article reaches.
"""

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
_spec = importlib.util.spec_from_file_location("loss_funnel", HERE.parent / "scripts" / "loss_funnel.py")
lf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lf)


def test_to_eval_ref_maps_a_standard_code_tuple():
    assert lf._to_eval_ref(("584", "c.p.")) == ("cp", "584")


def test_to_eval_ref_maps_a_real_document_name():
    assert lf._to_eval_ref(("130", "Codice del Consumo 2026")) == ("consumo", "130")


def test_to_eval_ref_is_none_for_an_unmapped_document():
    assert lf._to_eval_ref(("1", "Manuale di diritto processuale penale")) is None


def test_ordered_eval_refs_keeps_first_occurrence_and_drops_unmapped():
    refs = lf._ordered_eval_refs([("584", "c.p."), ("1", "Manuale X"), ("584", "c.p."), ("589", "c.p.")])
    assert refs == [("cp", "584"), ("cp", "589")]


def test_parse_dropped_label_splits_number_and_document():
    assert lf._parse_dropped_label("624-bis c.p.") == ("624-bis", "c.p.")
    assert lf._parse_dropped_label("130 Codice del Consumo 2026") == ("130", "Codice del Consumo 2026")


def test_parse_dropped_label_handles_a_bare_number():
    assert lf._parse_dropped_label("584") == ("584", "")


# --- _classify_one: the furthest stage a single ref reaches -----------------

def test_classify_one_not_in_pool_when_absent_from_everything():
    stage, _ = lf._classify_one(("cp", "999"), set(), {}, {}, set())
    assert stage == "NOT_IN_POOL"


def test_classify_one_cut_by_cap_when_found_but_never_reaches_article_rows():
    """Ranked in the fused pool (fused_rank), but NOT in article_rows_eval
    -- i.e. it didn't survive the cap and nothing else (keep, cross-
    reference, issue spotting) rescued it either."""
    fused_rank = {("cp", "999"): lf.dg._FUSION_POOL_CAP + 3}
    stage, detail = lf._classify_one(("cp", "999"), set(), fused_rank, {}, set())
    assert stage == "IN_POOL_CUT_BY_CAP"
    assert f"rank {lf.dg._FUSION_POOL_CAP + 3}" in detail


def test_classify_one_selected_not_cited_when_in_article_rows_but_not_cited():
    article_rows_eval = {("cp", "584")}
    fused_rank = {("cp", "584"): 3}   # just for the detail string
    stage, detail = lf._classify_one(("cp", "584"), article_rows_eval, fused_rank, {}, set())
    assert stage == "SELECTED_NOT_CITED"
    assert "rank 3" in detail


def test_classify_one_cited_when_in_article_rows_and_cited():
    article_rows_eval = {("cp", "584")}
    stage, _ = lf._classify_one(("cp", "584"), article_rows_eval, {}, {}, {("cp", "584")})
    assert stage == "CITED"


def test_classify_one_reached_via_keep_or_crossref_is_never_cut_by_the_cap():
    """A ref in article_rows_eval but absent from fused_rank entirely --
    exactly what "keep", cross-reference broadening and issue spotting all
    look like, since none of them ever appear in the fused ranking -- still
    reaches CITED, not IN_POOL_CUT_BY_CAP or NOT_IN_POOL. Round 9, 19 Oct:
    this is the fix for the funnel's own blind spot that misreported
    crossref-added articles as NOT_IN_POOL."""
    article_rows_eval = {("cp", "584")}
    stage, detail = lf._classify_one(("cp", "584"), article_rows_eval, {}, {}, {("cp", "584")})
    assert stage == "CITED"
    assert "keep" in detail or "cross-reference" in detail


def test_classify_one_dropped_by_check_beats_cited():
    """A verdict that dropped the article takes priority even if, for some
    reason, a citation for it also exists (it shouldn't once dropped, but
    the check itself is what the stage should report)."""
    article_rows_eval = {("cp", "584")}
    dropped_eval = {("cp", "584"): "incompatibile: ..."}
    stage, detail = lf._classify_one(("cp", "584"), article_rows_eval, {}, dropped_eval, {("cp", "584")})
    assert stage == "DROPPED_BY_CHECK"
    assert detail == "incompatibile: ..."


# --- _classify: picks the best alternative for an any_of group -------------

def test_classify_plain_item_delegates_to_classify_one():
    item = {"code": "cp", "article": "584"}
    stage, _ = lf._classify(item, {("cp", "584")}, {("cp", "584"): 1}, {}, {("cp", "584")})
    assert stage == "CITED"


def test_classify_any_of_picks_the_furthest_alternative():
    item = {"code": "consumo", "any_of": ["128", "130"], "label": "any"}
    article_rows_eval = {("consumo", "130")}                        # only 130 was selected
    fused_rank = {("consumo", "128"): lf.dg._FUSION_POOL_CAP + 1}    # 128 was found but cut by cap
    cited_eval = {("consumo", "130")}                                # 130 was cited
    stage, _ = lf._classify(item, article_rows_eval, fused_rank, {}, cited_eval)
    assert stage == "CITED"


def test_classify_any_of_not_in_pool_when_no_alternative_appears_anywhere():
    item = {"code": "consumo", "any_of": ["128", "130"], "label": "any"}
    stage, _ = lf._classify(item, set(), {}, {}, set())
    assert stage == "NOT_IN_POOL"
