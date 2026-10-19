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


def test_classify_one_cut_by_cap_when_ranked_past_it():
    fused_rank = {("cp", "999"): lf.dg._FUSION_POOL_CAP + 3}
    stage, detail = lf._classify_one(("cp", "999"), set(), fused_rank, {}, set())
    assert stage == "IN_POOL_CUT_BY_CAP"
    assert f"rank {lf.dg._FUSION_POOL_CAP + 3}" in detail


def test_classify_one_selected_not_cited_when_ranked_within_the_cap_but_not_cited():
    fused_rank = {("cp", "584"): 3}
    stage, detail = lf._classify_one(("cp", "584"), set(), fused_rank, {}, set())
    assert stage == "SELECTED_NOT_CITED"
    assert "rank 3" in detail


def test_classify_one_cited_when_ranked_within_the_cap_and_cited():
    fused_rank = {("cp", "584"): 1}
    stage, _ = lf._classify_one(("cp", "584"), set(), fused_rank, {}, {("cp", "584")})
    assert stage == "CITED"


def test_classify_one_keep_is_never_cut_by_the_cap():
    """A "keep" ref (named by the user / ruling agreement) reaches at
    least "selected" regardless of its rank in the fused pool."""
    keep_eval = {("cp", "584")}
    stage, detail = lf._classify_one(("cp", "584"), keep_eval, {}, {}, {("cp", "584")})
    assert stage == "CITED"
    assert "keep" in detail


def test_classify_one_dropped_by_check_beats_cited():
    """A verdict that dropped the article takes priority even if, for some
    reason, a citation for it also exists (it shouldn't once dropped, but
    the check itself is what the stage should report)."""
    fused_rank = {("cp", "584"): 1}
    dropped_eval = {("cp", "584"): "incompatibile: ..."}
    stage, detail = lf._classify_one(("cp", "584"), set(), fused_rank, dropped_eval, {("cp", "584")})
    assert stage == "DROPPED_BY_CHECK"
    assert detail == "incompatibile: ..."


# --- _classify: picks the best alternative for an any_of group -------------

def test_classify_plain_item_delegates_to_classify_one():
    item = {"code": "cp", "article": "584"}
    stage, _ = lf._classify(item, set(), {("cp", "584"): 1}, {}, {("cp", "584")})
    assert stage == "CITED"


def test_classify_any_of_picks_the_furthest_alternative():
    item = {"code": "consumo", "any_of": ["128", "130"], "label": "any"}
    fused_rank = {("consumo", "128"): lf.dg._FUSION_POOL_CAP + 1,   # cut by cap
                  ("consumo", "130"): 1}                            # ranked within the cap
    cited_eval = {("consumo", "130")}                               # cited
    stage, _ = lf._classify(item, set(), fused_rank, {}, cited_eval)
    assert stage == "CITED"


def test_classify_any_of_not_in_pool_when_no_alternative_appears_anywhere():
    item = {"code": "consumo", "any_of": ["128", "130"], "label": "any"}
    stage, _ = lf._classify(item, set(), {}, {}, set())
    assert stage == "NOT_IN_POOL"
