"""The branch filter of the rulings the writer reads (defensive_generation._other_branch / _case_branch).

Prod tests, 23-24 Oct: when the case analysis failed, its fallback has area "", and every criminal
ruling of a stalking case was dropped as "the other branch": the parere was written from 1 ruling."""

import os

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

import src.rag.defensive_generation as dg  # noqa: E402

PENAL = ["Il ricorso è inammissibile: art. 572 c.p., art. 606 c.p.p., art. 133 c.p."]
CIVIL = ["La domanda di risarcimento ex art. 2043 c.c. e art. 2059 c.c. è accolta; art. 115 c.p.c."]


def test_an_unknown_branch_drops_nothing():
    assert dg._other_branch(PENAL, "") is False and dg._other_branch(CIVIL, "") is False


def test_a_known_branch_still_drops_the_other():
    assert dg._other_branch(CIVIL, "penale") and not dg._other_branch(PENAL, "penale")
    assert dg._other_branch(PENAL, "civile") and not dg._other_branch(CIVIL, "civile")


def test_the_branch_falls_back_on_the_case_frame():
    assert dg._case_branch({"area": "civile", "frame": {"areas": ["penale"]}}) == "civile"
    assert dg._case_branch({"area": "", "frame": {"areas": ["penale"]}}) == "penale"
    assert dg._case_branch({"frame": {"areas": ["responsabilita_civile", "contratti"]}}) == "civile"
    assert dg._case_branch({"frame": {"areas": ["amministrativo"]}}) == "amministrativo"
    assert dg._case_branch({}) == ""
