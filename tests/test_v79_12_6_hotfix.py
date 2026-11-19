"""Hotfix 79.12.6 -- element_schemas._map_ops lifting an «other» op whose child is a bare ref string («E3» in place of
{"ref": "E3"}): it raised AttributeError ('str' object has no attribute 'get') in _normalize_one -> _map_ops. A bare
ref string is now read as that ref node (logged as a mapping) wherever it stands in a node. Synthetic article only."""

import pytest


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for name, value in (("LLM_API_KEY", "k"), ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "k"), ("NEO4J_URI", "bolt://localhost:7687"), ("NEO4J_USER", "n"),
                        ("NEO4J_PASSWORD", "x")):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("ARTICLE_SCHEMAS_FILE", str(tmp_path / "schemas.jsonl"))
    monkeypatch.setenv("ARTICLE_SCHEMAS_REVIEWED_FILE", str(tmp_path / "reviewed.jsonl"))
    monkeypatch.delenv("ELEMENT_SCHEMAS", raising=False)


# a synthetic article (invented wording, no real article)
HIVES = ("1. Chiunque, senza il consenso del custode, asporta le arnie altrui è punito con la multa fino a mille "
         "euro.\n2. Se il fatto è commesso di notte, la pena è aumentata.")


def _el(eid, testo, kind="condotta", comma="1"):
    return {"id": eid, "testo": testo, "kind": kind, "kind_note": None, "comma": comma, "branches": [],
            "connective": None, "parti": [], "grounding": "explicit", "fonte": {"comma": comma, "testo": testo},
            "spiegazione": None}


def _schema(logic):
    return {"tipo": ["reato", "sanction"], "destinatario": [{"ruolo": "chiunque", "condizione": None}],
            "legittimato": [], "soggetto_passivo": [], "rapporto": "furto di arnie",
            "elementi": [_el("E1", "asporta le arnie altrui"),
                         _el("E2", "senza il consenso del custode", kind="negazione"),
                         _el("E3", "il fatto è commesso di notte", kind="condizione", comma="2")],
            "logic": logic, "eccezioni": [],
            "conseguenza": {"testo": "è punito con la multa fino a mille euro", "tipo": "pena", "comma": "1"},
            "commi_speciali": [], "collegati": []}


@pytest.mark.parametrize("logic,expected,bare", [
    # the server's shape: an «other» node with one bare ref string
    ({"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}, {"op": "other", "args": ["E3"]}]},
     {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}, {"ref": "E3"}]}, ["E3"]),
    ({"op": "other", "args": ["E3"]}, {"ref": "E3"}, ["E3"]),
    ({"op": "other", "args": ["E1", " E2 ", {"ref": "E3"}]},
     {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}, {"ref": "E3"}]}, ["E1", " E2 "]),
    ({"op": "all_of", "args": ["E1", "E2", {"op": "conditional", "if": "E3", "then": "E1"}]},
     {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"},
                               {"op": "conditional", "if": {"ref": "E3"}, "then": {"ref": "E1"}}]},
     ["E1", "E2", "E3", "E1"]),
])
def test_a_bare_ref_string_is_read_as_its_ref(logic, expected, bare):
    from src.rag import element_schemas as E
    s = _schema(logic)
    norm, mappings = E.normalize(s)              # raised AttributeError before the hotfix
    assert norm["logic"] == expected
    assert sorted(m["from"] for m in mappings if m["field"] == "ref") == sorted(bare)       # each one logged
    checks = E.check(s, HIVES)
    assert checks["invalid"] == [] and checks["logic_errors"] == []
    assert E.all_dangling_refs(E.repair(s, HIVES)[0]) == []


def test_the_other_lift_is_logged_with_its_target():
    from src.rag import element_schemas as E
    _, mappings = E.normalize(_schema({"op": "other", "args": ["E3"]}))
    assert [(m["from"], m["to"]) for m in mappings if m["field"] == "op"] == [("other", "ref")]


def test_a_bare_ref_to_no_element_is_a_logic_error_not_a_crash():
    from src.rag import element_schemas as E
    checks = E.check(_schema({"op": "other", "args": ["E9"]}), HIVES)
    assert checks["logic_errors"] and checks["invalid"] == []


def test_non_string_children_stay_invalid():
    from src.rag import element_schemas as E
    checks = E.check(_schema({"op": "other", "args": [7]}), HIVES)      # no crash; a malformed node stays invalid
    assert checks["invalid"]
    assert E.unknown_values(_schema({"op": "other", "args": [7]})) is not None
