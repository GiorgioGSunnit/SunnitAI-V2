"""Simulation/production parity evidence for an AMBIGUOUS calculator match.

The dev simulation asks the user to choose among the tied candidates (a
structured clarification) on ANY ambiguous match. Production
(src/rag/calculation.py, exercised in tests/test_calculation_route.py) now
does the same, but only when every tied candidate would have auto-routed on
its own; a weak tie such as the one below still falls back to normal RAG on
purpose, since prompting there would turn any passing mention of a legal
topic into a menu. This test pins the simulation side; the production side
is pinned in tests/test_calculation_route.py. See docs/TO_VERIFY.md.
"""

from app.main import engine
from simulation.conversation import SimulatedConversation
from simulation.planner import plan_sentence

# Two penal-draft calculators tie on single-token evidence -> deterministic
# ambiguous routing (no phrase match, equal score). Two live calculators,
# not the withheld penal drafts: the ambiguity mechanism is
# calculator-agnostic and must stay under test in the default,
# drafts-disabled configuration.
AMBIGUOUS_QUERY = "imu e irpef"


def test_matcher_and_planner_report_ambiguous():
    plan = plan_sentence(AMBIGUOUS_QUERY, engine.registry.definitions())
    assert plan.status == "ambiguous"
    assert len(plan.candidates) >= 2


def test_simulation_asks_user_to_choose_on_ambiguous_match():
    reply = SimulatedConversation(engine).send(AMBIGUOUS_QUERY)
    assert reply.kind == "ambiguous"
    assert reply.plan is not None and reply.plan.status == "ambiguous"
    # every tied candidate is surfaced for the user to pick from
    for candidate in reply.plan.candidates:
        assert candidate.calculator_id in reply.text
