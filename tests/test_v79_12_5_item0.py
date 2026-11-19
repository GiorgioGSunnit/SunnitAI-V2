"""v79.12.5 item 0 -- the element schemas' logic pruner and vocabulary; synthetic article texts only (no model, no
Neo4j, no network):

1. a conditional's if / then / else pruned exactly as an all_of's or any_of's args: a dropped element's ref removed, a
   branch left with one child collapsed into it, an emptied branch dropped, the node collapsed or dropped -- never kept
   with a branch missing, never a dangling ref (no then: «if not(if) then else», or nothing without an else; no if:
   then, else, or an any_of of the two)
2. kind «obbligo» -> «condotta»; a comma speciale's «limiting» -> «limite» (a special kind now); a logic op «other»
   lifted (its children in its place: one is the node, several an all_of, none removes it)
3. a schema with no tipo is tipo «other» with the note «tipo mancante», never invalid -- and no opinion for the gate
4. every value no vocabulary maps (kind, connective, op, tipo, role, special_kind) listed (unknown_values) and counted
   by scripts/schema_show.py --recheck (printed, logged, in its result and on the regeneration keys)
5. --recheck on a synthetic file of each shape: usable before -> after
"""

import importlib.util
import json
import logging
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


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
    monkeypatch.delenv("ELEMENT_SCHEMAS", raising=False)
    for name in ("SCHEMA_LLM_BASE_URL", "SCHEMA_LLM_MODEL", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def show():
    spec = importlib.util.spec_from_file_location("schema_show", ROOT / "scripts" / "schema_show.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# a synthetic article (invented wording, no real article)
HIVES = ("1. Chiunque, senza il consenso del custode, asporta le arnie altrui è punito con la multa fino a mille "
         "euro.\n2. Se il fatto è commesso di notte, la pena è aumentata.\n"
         "3. La pena resta comunque entro il doppio del massimo.")
DOC = "Codice Penale Sintetico 2026"


def _el(eid, testo, kind="condotta", comma="1", grounding="explicit", **extra):
    out = {"id": eid, "testo": testo, "kind": kind, "kind_note": None, "comma": comma, "branches": [],
           "connective": None, "parti": [], "grounding": grounding, "fonte": {"comma": comma, "testo": testo},
           "spiegazione": None}
    out.update(extra)
    return out


def _schema(**over):
    schema = {"tipo": ["reato", "sanction"], "destinatario": [{"ruolo": "chiunque", "condizione": None}],
              "legittimato": [], "soggetto_passivo": [], "rapporto": "furto di arnie",
              "elementi": [_el("E1", "asporta le arnie altrui"),
                           _el("E2", "senza il consenso del custode", kind="negazione")],
              "logic": {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}]}, "eccezioni": [],
              "conseguenza": {"testo": "è punito con la multa fino a mille euro", "tipo": "pena", "comma": "1"},
              "commi_speciali": [], "collegati": []}
    schema.update(over)
    return schema


NIGHT = _el("E3", "il fatto è commesso di notte", kind="condizione", comma="2")
DOUBLE = _el("E4", "la pena è raddoppiata dal giudice", kind="condizione", comma="2")    # ungrounded: dropped
LIMIT = _el("E5", "la pena resta entro il doppio del massimo", kind="quantificazione", comma="3")


def _blocking(checks):
    from src.rag import element_schemas as E
    return E.raised(checks)


# --- 1. the conditional's branches as args ------------------------------------------------------------------------------

COND = {"op": "conditional", "if": {"ref": "E1"}, "then": {"op": "all_of", "args": [{"ref": "E2"}, {"ref": "E3"}]},
        "else": {"ref": "E4"}}


@pytest.mark.parametrize("gone,expected", [
    (set(), COND),
    ({"E2"}, {"op": "conditional", "if": {"ref": "E1"}, "then": {"ref": "E3"}, "else": {"ref": "E4"}}),  # collapsed
    ({"E4"}, {"op": "conditional", "if": {"ref": "E1"}, "then": {"op": "all_of", "args": [{"ref": "E2"},
                                                                                       {"ref": "E3"}]}}),
    # no then: the else holds where the condition fails
    ({"E2", "E3"}, {"op": "conditional", "if": {"op": "not", "arg": {"ref": "E1"}}, "then": {"ref": "E4"}}),
    ({"E2", "E3", "E4"}, None),                 # a condition with no consequence requires nothing
    # no if: then and else are the alternatives, never read as both
    ({"E1"}, {"op": "any_of", "args": [{"op": "all_of", "args": [{"ref": "E2"}, {"ref": "E3"}]}, {"ref": "E4"}],
              "da": "conditional"}),
    ({"E1", "E4"}, {"op": "all_of", "args": [{"ref": "E2"}, {"ref": "E3"}]}),               # then alone: the then
    ({"E1", "E2", "E3"}, {"ref": "E4"}),                                                    # else alone: the else
    ({"E1", "E2", "E3", "E4"}, None),
])
def test_a_conditional_is_pruned_as_args(gone, expected):
    from src.rag import element_schemas as E
    got = E.prune_logic(COND, gone)
    assert got == expected
    refs = []
    E._every_ref(got, refs)
    assert not set(refs) & gone                 # never a ref to a dropped element


def test_a_conditional_the_model_wrote_with_a_branch_missing_reads_the_same():
    from src.rag import element_schemas as E
    assert E.prune_logic({"op": "conditional", "then": {"ref": "E2"}}) == {"ref": "E2"}
    assert E.prune_logic({"op": "conditional", "if": {"ref": "E1"}}) is None
    assert E.prune_logic({"op": "conditional", "if": {"op": "not", "arg": {"ref": "E1"}}, "else": {"ref": "E2"}}) == \
        {"op": "conditional", "if": {"ref": "E1"}, "then": {"ref": "E2"}}            # no double negation
    nested = {"op": "all_of", "args": [{"ref": "E1"}, {"op": "conditional", "if": {"ref": "E2"},
                                                       "then": {"ref": "E3"}}]}
    assert E.prune_logic(nested, {"E3"}) == {"ref": "E1"}           # the emptied conditional gone, the all_of collapsed
    assert E.prune_logic({"op": "conditional", "ref": "E9", "if": {"ref": "E1"}, "then": {"ref": "E2"}}, {"E1"}) == \
        {"op": "all_of", "args": [{"ref": "E9"}, {"ref": "E2"}]}    # a node's own live ref stays cited


def test_a_conditional_whose_then_element_is_dropped_never_dangles():
    """The shape: «se il fatto è commesso di notte, [an ungrounded consequence]» -- the repair drops the then's element;
    the conditional goes with it (no conditional left with its «if» alone), the all_of collapses, nothing dangles."""
    from src.rag import element_schemas as E
    s = _schema(elementi=[_el("E1", "asporta le arnie altrui"), NIGHT, DOUBLE],
                logic={"op": "all_of", "args": [{"ref": "E1"}, {"op": "conditional", "if": {"ref": "E3"},
                                                                "then": {"ref": "E4"}}]})
    checks = E.check(s, HIVES)
    assert not _blocking(checks) and checks["ungrounded"]
    fixed, actions = E.repair(s, HIVES)
    assert [e["id"] for e in fixed["elementi"]] == ["E1", "E3"] and fixed["logic"] == {"ref": "E1"}
    assert E.all_dangling_refs(fixed) == [] and ("", "E4", "ungrounded", "dropped") in [
        (a["scope"], a["id"], a["check"], a["action"]) for a in actions]
    again = E.check(fixed, HIVES)
    assert not _blocking(again) and not any("conditional" in m for m in again["logic_incomplete"])
    # with an else, the else holds where the condition fails
    s["elementi"].append(_el("E5", "la pena resta entro il doppio del massimo", comma="3"))
    s["logic"]["args"][1]["else"] = {"ref": "E5"}
    fixed, _ = E.repair(s, HIVES)
    assert fixed["logic"] == {"op": "all_of", "args": [{"ref": "E1"}, {"op": "conditional", "if": {
        "op": "not", "arg": {"ref": "E3"}}, "then": {"ref": "E5"}}]}


def test_an_if_dropped_leaves_then_or_else_never_both_required():
    """The repair's any_of over no «o» in the text is read as an all_of -- never the then / else a dropped condition
    leaves: one of them holds, never both."""
    from src.rag import element_schemas as E
    s = _schema(elementi=[_el("E1", "asporta le arnie altrui"), DOUBLE, NIGHT, LIMIT],
                logic={"op": "conditional", "if": {"ref": "E4"}, "then": {"ref": "E3"}, "else": {"ref": "E5"}})
    fixed, actions = E.repair(s, HIVES)
    assert fixed["logic"] == {"op": "any_of", "args": [{"ref": "E3"}, {"ref": "E5"}], "da": "conditional"}
    assert not any(a["check"] == "branches_without_alternative" for a in actions)


# --- 2. the vocabulary: obbligo, limiting, op other ---------------------------------------------------------------------

def test_kind_obbligo_is_a_condotta():
    from src.rag import element_schemas as E
    s = _schema(elementi=[_el("E1", "asporta le arnie altrui", kind="Obbligo"),
                          _el("E2", "senza il consenso del custode", kind="obblighi")])
    checks = E.check(s, HIVES)
    assert checks["invalid"] == [] and checks["vocabulary_mapped"] == ["E1: kind 'Obbligo' -> 'condotta'",
                                                                       "E2: kind 'obblighi' -> 'condotta'"]
    assert [e["kind"] for e in E.repair(s, HIVES)[0]["elementi"]] == ["condotta", "condotta"]


def test_a_special_comma_limiting_is_limite():
    from src.rag import element_schemas as E
    assert "limite" in E.SPECIAL_KINDS and '"limite"' in E.SCHEMA_SYSTEM
    s = _schema(commi_speciali=[{"comma": "3", "kind": "limiting", "schema": {
        "tipo": ["limiting"], "elementi": [LIMIT], "logic": {"ref": "E5"}, "eccezioni": []}}])
    checks = E.check(s, HIVES)
    assert checks["invalid"] == [] and "3/kind: comma_speciale 'limiting' -> 'limite'" in [
        m.replace("co.", "") for m in checks["vocabulary_mapped"]]
    assert E.repair(s, HIVES)[0]["commi_speciali"][0]["kind"] == "limite"


@pytest.mark.parametrize("logic,expected,to", [
    ({"op": "other", "args": [{"ref": "E1"}, {"ref": "E2"}]}, {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}]},
     "all_of"),
    ({"op": "all_of", "args": [{"ref": "E1"}, {"op": "Other", "args": [{"ref": "E2"}]}]},
     {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}]}, "ref"),
    ({"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}, {"op": "altro"}]},
     {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}]}, None),
    ({"op": "other", "ref": "E1", "arg": {"op": "any_of", "args": [{"ref": "E2"}, {"ref": "E1"}]}},
     {"op": "all_of", "args": [{"ref": "E1"}, {"op": "any_of", "args": [{"ref": "E2"}, {"ref": "E1"}]}]}, "all_of"),
])
def test_a_logic_op_other_is_lifted(logic, expected, to):
    """«other» is no relation: its children stand in its place -- one is the node, several an all_of (each still
    required and cited: the fattispecie's conjunction), none removes it. Each lift a mapping."""
    from src.rag import element_schemas as E
    s = _schema(logic=logic)
    checks = E.check(s, HIVES)
    assert checks["invalid"] == [] and checks["logic_errors"] == []
    norm, mappings = E.normalize(s)
    assert norm["logic"] == expected
    assert [(m["field"], m["to"]) for m in mappings if m["field"] == "op"] == [("op", to)]
    assert E.normalize({**s, "logic": {"op": "other"}})[0]["logic"] is None      # nothing under it: no logic


def test_an_unknown_op_is_lifted_and_counted_never_a_rejection():
    """Review (counts, not rejections): an op no vocabulary maps is lifted as «other» is (its children in its place,
    several as an all_of), logged as unknown and counted; a non-string op stays invalid (a malformed node)."""
    from src.rag import element_schemas as E
    s = _schema(logic={"op": "xor", "args": [{"ref": "E1"}, {"ref": "E2"}]})
    checks = E.check(s, HIVES)
    assert checks["invalid"] == [] and not _blocking(checks)
    assert "logic: op 'xor' -> 'all_of'" in checks["vocabulary_mapped"]
    assert E.normalize(s)[0]["logic"] == {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}]}
    assert [(u["field"], u["value"]) for u in E.unknown_values(s)] == [("op", "xor")]
    s = _schema(logic={"op": 7, "args": [{"ref": "E1"}, {"ref": "E2"}]})
    assert any("op sconosciuto 7" in m for m in E.check(s, HIVES)["invalid"])
    assert [(u["field"], u["value"]) for u in E.unknown_values(s)] == [("op", 7)]
    # only a lifted node with no child leaves: a null arg the model wrote stays (invalid, as before)
    s = _schema(logic={"op": "all_of", "args": [None, {"ref": "E1"}, {"ref": "E2"}]})
    assert E.normalize(s)[0]["logic"]["args"][0] is None
    assert "logic.all_of[0]: nodo non valido" in E.check(s, HIVES)["invalid"]


# --- 3. a missing tipo --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("over", [{"tipo": None}, {"tipo": []}, {"tipo": ""}, {"tipo": [" "]}, "absent"])
def test_a_schema_with_no_tipo_is_other_with_a_note(over):
    from src.rag import element_schemas as E
    s = _schema()
    if over == "absent":
        del s["tipo"]
    else:
        s.update(over)
    checks = E.check(s, HIVES)
    assert checks["invalid"] == [] and not _blocking(checks)
    assert checks["vocabulary_mapped"] == [f"tipo: tipo {s.get('tipo')!r} -> ['other']"]
    fixed, _ = E.repair(s, HIVES)
    assert fixed["tipo"] == ["other"] and fixed["tipo_note"] == "tipo mancante"
    # the note's «other» is no tipo: no gate opinion, no offence -- the model's own «other» still is one
    line = {"schema": fixed, "source": "reviewed"}
    assert E.tipi_of(fixed) == [] and E.opinion(line) is None and E.gate("990 c.p.", line) is None
    assert not E.is_offence_schema(fixed)
    assert E.tipi_of({"tipo": ["other"]}) == ["other"] and E.opinion({"schema": {"tipo": ["other"]}}) is False


def test_a_tipo_that_is_unknown_is_other_with_a_note_and_counted():
    """Review (counts, not rejections): an unknown tipo is left out (logged as unknown); none left -> «other» with
    tipo_note «tipo sconosciuto …», read as no tipo (no gate opinion), as a missing one; a non-string stays invalid."""
    from src.rag import element_schemas as E
    s = _schema(tipo="punitiva")
    checks = E.check(s, HIVES)
    assert checks["invalid"] == [] and not _blocking(checks)
    fixed, _ = E.repair(s, HIVES)
    assert fixed["tipo"] == ["other"] and fixed["tipo_note"] == "tipo sconosciuto 'punitiva'"
    assert E.tipi_of(fixed) == [] and E.opinion({"schema": fixed, "source": "reviewed"}) is None
    assert [(u["field"], u["value"]) for u in E.unknown_values(s)] == [("tipo", "punitiva")]
    mixed = E.normalize(_schema(tipo=["reato", "punitiva"]))[0]
    assert mixed["tipo"] == ["reato"] and "tipo_note" not in mixed
    assert any("tipo sconosciuto [5]" in m for m in E.check(_schema(tipo=[5]), HIVES)["invalid"])


@pytest.mark.parametrize("over,mapped", [
    ({"destinatario": [{"ruolo": "marziano", "condizione": None}]}, "destinatario: destinatario 'marziano' -> 'altro'"),
    ({"commi_speciali": [{"comma": "2", "kind": "bizzarro", "schema": {"tipo": ["sanction"], "elementi": [], "logic": None}}]},
     "co.2/kind: comma_speciale 'bizzarro' -> 'other'"),
])
def test_an_unknown_role_or_special_kind_is_its_neutral_value_and_counted(over, mapped):
    from src.rag import element_schemas as E
    s = _schema(**over)
    checks = E.check(s, HIVES)
    assert not any("sconosciuto" in m or "non valido" in m for m in checks["invalid"]), checks["invalid"]
    assert mapped in checks["vocabulary_mapped"] and len(E.unknown_values(s)) == 1


# --- 4. the unknown values ----------------------------------------------------------------------------------------------

def test_every_unknown_value_is_listed_by_field():
    from src.rag import element_schemas as E
    s = _schema(tipo=["reato", "boh"], destinatario=[{"ruolo": "marziano", "condizione": None}],
                elementi=[_el("E1", "asporta le arnie altrui", kind="Fattispecie  Strana"),
                          _el("E2", "senza il consenso del custode", kind="obbligo", connective="forse")],
                logic={"op": "all_of", "args": [{"ref": "E1"}, {"op": "xor", "args": [{"ref": "E2"}]}]},
                commi_speciali=[{"comma": "2", "kind": "bizzarro", "schema": {
                    "tipo": ["sanction"], "elementi": [_el("E1", "il fatto è commesso di notte", kind="nebbia",
                                                           comma="2")], "logic": {"ref": "E1"}}}])
    got = E.unknown_values(s)
    assert [(u["scope"], u["id"], u["field"], u["value"]) for u in got] == [
        ("", "tipo", "tipo", "boh"), ("", "destinatario", "role", "marziano"),
        ("", "E1", "kind", "Fattispecie  Strana"), ("", "E2", "connective", "forse"), ("", "logic", "op", "xor"),
        ("2", "kind", "special_kind", "bizzarro"), ("2", "E1", "kind", "nebbia")]
    assert {u["field"] for u in got} <= set(E.UNKNOWN_FIELDS)
    assert E.unknown_text(got[2]) == "kind 'fattispecie strana'"            # case and spaces folded for the count
    assert E.unknown_values(_schema()) == []                                # obbligo mapped: never unknown


# --- 5. --recheck on a synthetic file of each shape ---------------------------------------------------------------------

def _line(number, schema, invalid=()):
    """A line as 79.12.4's --recheck left it (rechecked_at; its checks as that checker recorded them)."""
    from src.rag import element_schemas as E
    h = E.text_hash(HIVES)
    checks = {**E.empty_checks(), "invalid": list(invalid)}
    return {"doc": DOC, "article": number, "code": "c.p.", "label": f"{number} c.p.", "schema_version": 1,
            "source_hash": h, "text_hash": h, "prompt_hash": E.prompt_hash(), "schema": schema, "text": HIVES,
            "checks": checks, "dropped": [], "rechecked_at": "2026-11-18T00:00:00+00:00", "elements_after": 2}


def _shapes():
    """One line per shape; `invalid` is what the 79.12.4 checker recorded for it (run on this file before the change)."""
    no_tipo = _schema()
    del no_tipo["tipo"]
    return [
        _line("901", _schema(elementi=[_el("E1", "asporta le arnie altrui"), NIGHT, DOUBLE], logic={
            "op": "all_of", "args": [{"ref": "E1"}, {"op": "conditional", "if": {"ref": "E3"},
                                                     "then": {"ref": "E4"}}]})),
        _line("902", _schema(elementi=[_el("E1", "asporta le arnie altrui", kind="obbligo"),
                                       _el("E2", "senza il consenso del custode", kind="negazione")]),
              ["E1: kind sconosciuto 'obbligo'"]),
        _line("903", _schema(commi_speciali=[{"comma": "3", "kind": "limiting", "schema": {
            "tipo": ["limiting"], "elementi": [LIMIT], "logic": {"ref": "E5"}, "eccezioni": []}}]),
              ["co.3: kind speciale 'limiting' non valido"]),
        _line("904", _schema(logic={"op": "other", "args": [{"ref": "E1"}, {"ref": "E2"}]}),
              ["logic: op sconosciuto 'other'"]),
        _line("905", no_tipo, ["tipo mancante"]),
        _line("906", _schema(elementi=[_el("E1", "asporta le arnie altrui", kind="nebbia"),
                                       _el("E2", "senza il consenso del custode", kind="negazione")]),
              ["E1: kind sconosciuto 'nebbia'"]),
        _line("907", _schema(elementi=[_el("E1", "asporta le arnie altrui", kind="nebbia"),
                                       _el("E2", "senza il consenso del custode", kind="Nebbia ")]),
              ["E1: kind sconosciuto 'nebbia'", "E2: kind sconosciuto 'Nebbia '"]),
    ]


def _write(path, lines):
    with open(path, "w", encoding="utf-8") as fh:
        for x in lines:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")


def test_recheck_usable_before_and_after_with_the_unknown_values_counted(tmp_path, show, caplog):
    _write(tmp_path / "schemas.jsonl", _shapes())
    regen_file = tmp_path / "regen.jsonl"
    with caplog.at_level(logging.WARNING, logger="schema_show"):
        r = show.recheck(now="2026-11-19T00:00:00+00:00", regen_list=str(regen_file))
    # review (counts, not rejections): the two «nebbia» lines are counted, read as «other», and usable: 1 -> 7
    assert (r["articles"], r["usable_before"], r["usable_after"]) == (7, 1, 7)
    assert r["by_code"] == {"c.p.": {"articles": 7, "schemas": 7, "before": 1, "after": 7}}
    assert r["unknown"] == {"kind 'nebbia'": 3} and r["unknown_total"] == 3 and r["unknown_schemas"] == 2
    assert r["unknown_by_field"] == {"kind": 3} and r["unusable_after"] == {}
    assert r["mapped"].get("kind 'nebbia' -> 'other'") == 2 and r["mapped"].get("kind 'Nebbia ' -> 'other'") == 1
    for mapping in ("kind 'obbligo' -> 'condotta'", "comma_speciale 'limiting' -> 'limite'", "op 'other' -> 'all_of'",
                    "tipo None -> ['other']"):
        assert r["mapped"].get(mapping) == 1, (mapping, r["mapped"])
    keys = [json.loads(x) for x in regen_file.read_text(encoding="utf-8").splitlines()] if regen_file.exists() else []
    assert keys == []                                   # nothing left to regenerate: a gap is a count, not a rejection
    assert "3 values no vocabulary maps in 2 schemas -- kind 'nebbia': 3" in caplog.text
    after = {x["article"]: x for x in map(json.loads, (tmp_path / "schemas.jsonl").read_text().splitlines())}
    assert after["901"]["checks"]["logic_incomplete"] == [] and after["901"]["elements_after"] == 2
    assert [a["id"] for a in after["901"]["dropped"] if a["action"] == "dropped"] == ["E4"]


def test_recheck_prints_the_unknown_values_with_their_counts(tmp_path, show, capsys):
    _write(tmp_path / "schemas.jsonl", _shapes())
    assert show.main(["--recheck", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Schemi usabili: prima 1, dopo 7 (+6)." in out
    assert "Valori fuori vocabolario: 3 in 2 schemi (per campo: kind 3):" in out
    assert "       3  kind 'nebbia'" in out
