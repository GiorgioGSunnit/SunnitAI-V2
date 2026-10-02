"""The tone / standing / response-length settings: 3 levels since Oct 2026.

Testers who changed a slider went straight to its top level (4) and nobody used
level 3, so 3 and 4 were merged into a new 3 that keeps what 4 did. A 4 stored
in the database, or sent by a frontend that still shows four positions, must
keep that behaviour.

The answer rules used to contradict the length setting (a hard cap of 3
citations against "cite every relevant section"; a mandatory conclusive closing
against "always close with a follow-up suggestion"); these tests pin the fix.
"""

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from fastapi import HTTPException  # noqa: E402

from src.rag import prompts as P  # noqa: E402


def test_old_level_four_and_odd_values_map_onto_three_levels():
    assert [P.setting_level(v) for v in (1, 2, 3, 4)] == [1, 2, 3, 3]
    assert P.setting_level(None) == 2
    assert P.setting_level("x") == 2
    assert P.setting_level(0) == 1


def test_a_stored_four_keeps_what_level_four_did():
    """Directive tone, Latin maxims, the longest answers."""
    assert "imperative" in P._TONE[P.setting_level(4)]
    assert "Latin maxims" in P._STANDING[P.setting_level(4)]
    assert "4 to 6" in P.length_instruction(4)


@pytest.mark.parametrize("level", [1, 2, 3])
def test_each_length_level_says_how_many_sections_to_cite(level):
    assert "Cite" in P.length_instruction(level)


def test_answer_rules_no_longer_contradict_the_length_setting():
    rules = P.synthesis_system_message("it", length=3)
    assert "Never cite more than 3" not in rules
    assert "2 to 3 most directly relevant" not in rules
    assert "up to 5" in rules
    assert "Always close with either a follow-up suggestion" not in rules
    assert rules.count("CRITICAL TOPICALITY TEST") == 1


def test_not_in_the_documents_reply_stays_short_whatever_the_length():
    for build in (P.synthesis_empty_system, P.synthesis_error_system):
        rules = build("it", length=3)
        assert "RESPONSE LENGTH" not in rules
        assert "4 to 6" not in rules


def test_answer_ceiling_has_one_entry_per_level():
    from src.rag.nodes.synthesis import _ANSWER_TOKENS

    assert sorted(_ANSWER_TOKENS) == list(range(1, P.SETTING_LEVELS + 1))


# --- Saving the settings ----------------------------------------------------


@pytest.fixture
def auth_routes(monkeypatch):
    from src.chatbot.routes import auth

    saved = {}

    def fake_update(**kwargs):
        saved.update({k: kwargs[k] for k in ("tone", "standing", "response_length")})
        return SimpleNamespace(**saved)

    monkeypatch.setattr(auth.crud, "update_user_settings", fake_update)
    monkeypatch.setattr(auth.crud, "get_user_preferences", lambda db, user_id: None)
    return auth, saved


def test_a_four_from_the_older_frontend_is_saved_as_three(auth_routes):
    auth, saved = auth_routes
    request = auth.UpdateSettingsRequest(tone=4, standing=2, response_length=4)

    reply = auth.update_settings(request, current_user=SimpleNamespace(id="u1"), db=None)

    assert saved == {"tone": 3, "standing": 2, "response_length": 3}
    assert reply["settings"]["response_length"] == 3


def test_values_outside_the_sliders_are_still_refused(auth_routes):
    auth, _ = auth_routes
    request = auth.UpdateSettingsRequest(tone=5, standing=2, response_length=2)

    with pytest.raises(HTTPException) as refused:
        auth.update_settings(request, current_user=SimpleNamespace(id="u1"), db=None)

    assert refused.value.status_code == 400
