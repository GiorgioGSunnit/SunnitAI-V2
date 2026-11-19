"""v79.12.3 -- the element schemas' checker vocabulary, drops that never dangle, the full vocabularies in the prompt,
--recheck's regeneration list, polarity prepositions in grounding; on synthetic article texts only (no model, no
Neo4j, no network):

1. kind and connective synonyms (data/schema_vocabulary.json) mapped before validation instead of marking the schema
   invalid -- case, accent and plural insensitive, each mapping logged; an unknown value with no mapping still invalid
2. an element dropped: every logic ref to it pruned, an all_of / any_of left with one child collapsed into it; the
   checker never outputs a dangling ref (DanglingRefError on a bug), one already in the input stays logic_errors
3. tipo «reato», destinatario «chiunque» (a bare string too); SCHEMA_SYSTEM lists every vocabulary, prompt_hash moved;
   article_cards_bulk.py --schemas keeps a line whose prompt alone changed (--regen-prompt), --only-keys FILE makes
   exactly the keys asked, a reviewed label never
4. the procedibilità (querela, d'ufficio, their conditions, the querela irrevocable) leaves the offence's elements and
   logic for the schema's «procedibilita» block; the store's procedibilità reads it (ELEMENT_SCHEMAS), its own value
   winning on conflict unless the schema is reviewed; the records' conditions and the querela remedies read it
5. scripts/schema_show.py --recheck on a synthetic file shaped like the server's failures (unknown kind «procedural»,
   connective «ovvero», dangling refs): usable before -> after per code, mappings, the regeneration list (--regen-list)
6. «con», «senza», «contro», «salvo», «tranne», «eccetto» are content words: «con il consenso» never grounds against
   «senza il consenso», nor the reverse
7. a surname alone after an article or an articled preposition («il Rossi», «del Bianchi», «dal Verdi», «la Russo») is
   redacted (rule «article», logged with its word) and counted by the nightly's own reading (names_left); a role noun
   (data/redaction_stoplist.json «role_nouns»), a gazetteer word or an opener after an article stays
"""

import importlib.util
import itertools
import json
import logging
import os
import sys
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


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def show():
    return _load("schema_show")


@pytest.fixture
def bulk():
    return _load("article_cards_bulk")


# synthetic article texts (invented wording, no real article)
HIVES = ("1. Chiunque, senza il consenso del custode, asporta le arnie altrui o ne sottrae i favi è punito con la "
         "multa fino a mille euro.\n"
         "2. Il delitto è punibile a querela della persona offesa.\n"
         "3. La pena è aumentata se il fatto è commesso di notte.")
DEPOSIT = ("1. Il depositario che lascia deteriorare la merce affidatagli è tenuto a indennizzare il danno al "
           "depositante.\n2. La domanda si propone davanti al giudice del luogo del deposito.")
PENAL_DOC, CIVIL_DOC = "Codice Penale Sintetico 2026", "Codice Civile Sintetico 2026"


def _el(eid, testo, kind="condotta", comma="1", grounding="explicit", fonte=None, **extra):
    out = {"id": eid, "testo": testo, "kind": kind, "kind_note": None, "comma": comma, "branches": [],
           "connective": None, "parti": [], "grounding": grounding,
           "fonte": {"comma": comma, "testo": testo if fonte is None else fonte}, "spiegazione": None}
    out.update(extra)
    return out


def _hives(**over):
    schema = {"tipo": ["reato", "sanction"], "destinatario": [{"ruolo": "chiunque", "condizione": None}],
              "legittimato": [{"ruolo": "persona offesa", "condizione": None}], "soggetto_passivo": [],
              "rapporto": "furto di arnie",
              "elementi": [_el("E1", "asporta le arnie altrui o ne sottrae i favi", connective="o",
                               branches=["asporta le arnie altrui", "ne sottrae i favi"]),
                           _el("E2", "senza il consenso del custode", kind="negazione"),
                           _el("E3", "punibile a querela della persona offesa", kind="procedibilità", comma="2")],
              "logic": {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}, {"ref": "E3"}]},
              "eccezioni": [], "conseguenza": {"testo": "è punito con la multa fino a mille euro", "tipo": "pena",
                                               "comma": "1"},
              "commi_speciali": [{"comma": "3", "kind": "aggravamento", "schema": {
                  "tipo": ["sanction"], "elementi": [_el("E1", "il fatto è commesso di notte", kind="condizione",
                                                         comma="3")],
                  "logic": {"ref": "E1"}, "eccezioni": [], "conseguenza": None}}],
              "collegati": []}
    schema.update(over)
    return schema


def _deposit(**over):
    schema = {"tipo": ["claim-conferring"], "destinatario": [{"ruolo": "altro", "condizione": "il depositario"}],
              "legittimato": [{"ruolo": "attore", "condizione": None}], "soggetto_passivo": [], "rapporto": "deposito",
              "elementi": [_el("E1", "lascia deteriorare la merce affidatagli"),
                           _el("E2", "la domanda si propone davanti al giudice del luogo del deposito",
                               kind="competenza", comma="2")],
              "logic": {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}]}, "eccezioni": [],
              "conseguenza": {"testo": "è tenuto a indennizzare il danno", "tipo": "risarcimento", "comma": "1"},
              "commi_speciali": [], "collegati": []}
    schema.update(over)
    return schema


def _write(path, lines):
    with open(path, "w", encoding="utf-8") as fh:
        for x in lines:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")


def _ids(schema):
    return [e["id"] for e in schema["elementi"]]


def _acts(actions):
    return [(a["scope"], a["id"], a["check"], a["action"]) for a in actions]


# --- 1. the vocabulary's synonyms --------------------------------------------------------------------------------------

@pytest.mark.parametrize("raw,testo,want", [
    ("procedural", "punibile a querela della persona offesa", "procedibilità"),
    ("Procedurale", "si procede d'ufficio", "procedibilità"),
    ("procedural", "la domanda si propone davanti al giudice del luogo del deposito", "competenza"),
    ("evento dannoso", None, "evento"),
    ("Eventi Dannosi", None, "evento"),                    # case and plural
    ("CONDOTTE", None, "condotta"),                        # a vocabulary value in the plural
    ("nessi causali", None, "nesso_causale"),
    ("qualita del soggetto", None, "qualità del soggetto"),   # accent
    ("dolo specifico", None, "elemento_finalistico"),
    ("azione", None, None),                                # no mapping: still unknown
    ("boh", None, None),
])
def test_kind_synonyms(raw, testo, want):
    from src.rag import element_schemas as E
    assert E.canon_kind(raw, testo) == want


@pytest.mark.parametrize("raw,want", [("ovvero", "o"), ("Oppure", "o"), ("od", "o"), ("OD", "o"), ("e/o", "o"),
                                      ("ed", "e"), ("Ed", "e"), ("nonché", "e"), ("e", "e"), ("o", "o"),
                                      (None, None), ("", None), ("null", None)])
def test_connective_synonyms(raw, want):
    from src.rag import element_schemas as E
    assert E.canon_connective(raw) == want


def test_an_unknown_connective_or_kind_is_counted_not_invalid():
    """v79.12.5 (item 0, review: «counts, not rejections»): before, an unknown connective or kind made the schema
    invalid; now each is read as its field's neutral value (connective none, kind «other» with a kind_note naming
    it), logged as a mapping and counted (unknown_values) -- the schema stays usable."""
    from src.rag import element_schemas as E
    assert E.canon_connective("forse") is E._UNKNOWN
    s = _hives()
    s["elementi"][0]["connective"] = "forse"
    s["elementi"][1]["kind"] = "azione"
    checks = E.check(s, HIVES)
    assert checks["invalid"] == [] and E.usable({"schema": s, "checks": checks})
    assert "E1: connective 'forse' -> None" in checks["vocabulary_mapped"]
    assert "E2: kind 'azione' -> 'other'" in checks["vocabulary_mapped"] and not checks["kind_other_unexplained"]
    assert [(u["field"], u["value"]) for u in E.unknown_values(s)] == [("connective", "forse"), ("kind", "azione")]


def test_the_vocabulary_file_maps_only_to_vocabulary_values_and_never_shadows_one():
    from src.rag import element_schemas as E
    with open(E.VOCABULARY_FILE, encoding="utf-8") as fh:
        raw = json.load(fh)
    vocabs = E._vocabularies()
    for field in E._VOCAB_FIELDS:
        own = {E.vkey(v) for v in vocabs[field] if v is not None}
        assert len(own) == len([v for v in vocabs[field] if v is not None]), field     # no two values share a key
        assert len({E.vkey(k) for k in raw[field]}) == len(raw[field]), field          # nor two synonyms
        for synonym, target in raw[field].items():
            assert E.vkey(synonym) not in own, (field, synonym)          # a synonym never stands for a value itself
            rules = target if isinstance(target, list) else [{"to": target}]
            assert all(r["to"] in vocabs[field] for r in rules), (field, synonym)
        assert len(E.vocabulary()[field]) == len(raw[field])              # every entry loaded


def test_the_checker_maps_synonyms_before_validating_and_logs_each(caplog):
    """The server's failure shapes: kind «procedural» and connective «ovvero» are no longer invalid; each mapping is
    reported (vocabulary_mapped), a «mapped» action of the repair (logged when the runtime reads the line) and the
    repaired schema carries the vocabulary's own values; the input is never touched."""
    from src.rag import element_schemas as E
    s = _hives(tipo=["Fattispecie incriminatrice", "sanction"], destinatario="Chiunque")
    s["elementi"][0]["connective"] = "ovvero"
    s["elementi"][1]["kind"] = "elemento negativo"
    s["elementi"][2]["kind"] = "procedural"
    s["commi_speciali"][0]["kind"] = "circostanza aggravante"
    s["commi_speciali"][0]["schema"]["elementi"][0]["connective"] = "ed"
    checks = E.check(s, HIVES)
    assert checks["invalid"] == [] and checks["logic_errors"] == [] and E.usable({"schema": s, "checks": checks})
    assert checks["vocabulary_mapped"] == [
        "tipo: tipo 'Fattispecie incriminatrice' -> 'reato'",
        "destinatario: destinatario 'Chiunque' -> ['Chiunque']",
        "destinatario: destinatario 'Chiunque' -> 'chiunque'",
        "E1: connective 'ovvero' -> 'o'", "E2: kind 'elemento negativo' -> 'negazione'",
        "E3: kind 'procedural' -> 'procedibilità'", "co.3/kind: comma_speciale 'circostanza aggravante' -> 'aggravamento'",
        "co.3/E1: connective 'ed' -> 'e'"]
    fixed, actions = E.repair(s, HIVES)
    mapped = [a for a in actions if a["action"] == "mapped"]
    assert len(mapped) == 8 and all(a["check"] == "vocabulary" for a in mapped)
    assert ("", "E3", "vocabulary", "mapped") in _acts(actions) and mapped[5]["from"] == "procedural" and \
        mapped[5]["to"] == "procedibilità"
    assert fixed["tipo"] == ["reato", "sanction"] and fixed["destinatario"] == ["chiunque"]
    # v79.12.3 (item 4): the mapped procedibilità element moved to the block, out of the offence's elements and logic
    assert [e["kind"] for e in fixed["elementi"]] == ["condotta", "negazione"]
    assert fixed["procedibilita"]["regime"] == "querela" and ("", "E3", "procedibilita", "moved") in _acts(actions)
    assert fixed["elementi"][0]["connective"] == "o" and fixed["commi_speciali"][0]["kind"] == "aggravamento"
    assert s["elementi"][2]["kind"] == "procedural" and s["elementi"][0]["connective"] == "ovvero"   # a copy
    assert E.check(fixed, HIVES)["vocabulary_mapped"] == [] and E.repair(fixed, HIVES)[1] == []      # idempotent
    with caplog.at_level(logging.INFO, logger="src.rag.element_schemas"):
        E._REPAIRED.clear()
        E._repaired(s, HIVES, "991 c.p.")
    assert "vocabulary mapped E3 (kind 'procedural' -> 'procedibilità')" in caplog.text


def test_a_mapped_schema_is_read_by_the_runtime(monkeypatch, tmp_path):
    from src.rag import element_schemas as E
    monkeypatch.setenv("ELEMENT_SCHEMAS", "true")
    s = _hives()
    s["elementi"][0]["connective"] = "ovvero"
    s["elementi"][2]["kind"] = "procedural"
    line = E.make_line({"number": "991", "code": "c.p.", "text": HIVES}, "m-test", s, doc=PENAL_DOC)
    assert line["checks"]["invalid"] == [] and E.usable(line)
    _write(tmp_path / "schemas.jsonl", [line])
    row = {"_article": ("991", "c.p."), "d": {"name": PENAL_DOC}, "s": {"plain_text": HIVES}}
    got, why = E.for_row(row)
    assert why == "schema"
    reqs = E.requisites(row, got, {})
    assert [r["text"] for r in reqs] == ["asporta le arnie altrui o ne sottrae i favi", "senza il consenso del custode",
                                         "il fatto è commesso di notte"]       # procedibilità is no requisite
    assert reqs[0]["schema_connective"] == "o"
    summary = E.summary(got)
    assert summary["altri_elementi"] == [] and summary["procedibilita"]["regime"] == "querela"   # v79.12.3 (item 4)
    assert sum(1 for a in summary["dropped"] if a["action"] == "mapped") == 2
    from src.rag import interactive_flow
    counters = interactive_flow.element_counters({"records": {"991 c.p.": {"schema": summary, "requisites": []}}})
    assert counters["schema_vocabulary_mapped"] == 2 and counters["schema_elements_dropped"] == 0


def test_a_reviewed_schema_is_read_with_the_vocabulary_but_never_rewritten(monkeypatch, tmp_path):
    from src.rag import element_schemas as E
    monkeypatch.setenv("ELEMENT_SCHEMAS", "true")
    s = _hives()
    s["elementi"][0]["connective"] = "ovvero"
    s["elementi"][2]["kind"] = "procedural"
    _write(tmp_path / "reviewed.jsonl", [{"label": "991 c.p.", "reviewed": True, "reviewer": "avv",
                                          "source_hash": E.text_hash(HIVES), "schema_version": 1, "schema": s}])
    row = {"_article": ("991", "c.p."), "d": {"name": PENAL_DOC}, "s": {"plain_text": HIVES}}
    before = (tmp_path / "reviewed.jsonl").read_bytes()
    got, why = E.for_row(row)
    # review: read as check() validated it -- mapped, each mapping logged and counted; the file never rewritten
    assert why == "reviewed" and len(got["schema"]["elementi"]) == 2      # v79.12.3 (item 4): E3 in the block
    assert got["schema"]["procedibilita"]["regime"] == "querela"
    assert got["schema"]["elementi"][0]["connective"] == "o"
    assert (tmp_path / "reviewed.jsonl").read_bytes() == before
    reqs = E.requisites(row, got, {})
    assert "punibile a querela della persona offesa" not in [r["text"] for r in reqs]
    assert reqs[0]["schema_connective"] == "o"
    summary = E.summary(got)
    assert summary["source"] == "reviewed" and [a["action"] for a in summary["dropped"]] == ["mapped", "mapped",
                                                                                              "moved"]
    from src.rag import interactive_flow
    counters = interactive_flow.element_counters({"records": {"991 c.p.": {"schema": summary, "requisites": []}}})
    assert counters["schema_vocabulary_mapped"] == 2 and counters["schema_elements_dropped"] == 0


def test_a_reviewed_ref_mapped_by_case_is_read_mapped_never_dangling(monkeypatch, tmp_path):
    """Review: check() maps «e1» / «E2 » to their elements, so the overlay passes STRUCTURAL; the schema read must hold
    the same mapped refs (79.12.2: unusable:logic_errors; never a dangling ref reaching the runtime)."""
    from src.rag import element_schemas as E
    monkeypatch.setenv("ELEMENT_SCHEMAS", "true")
    s = _hives(tipo=["fattispecie incriminatrice"],
               logic={"op": "all_of", "args": [{"ref": "e1"}, {"ref": "E2 "}, {"ref": "E3"}]})
    s["commi_speciali"][0]["kind"] = "aggravante"
    _write(tmp_path / "reviewed.jsonl", [{"label": "991 c.p.", "reviewed": True, "reviewer": "avv",
                                          "source_hash": E.text_hash(HIVES), "schema_version": 1, "schema": s}])
    got, why = E.for_label("991 c.p.", HIVES)
    assert why == "reviewed" and E.all_dangling_refs(got["schema"]) == []
    summary = E.summary(got)
    assert summary["logic"] == {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}]}     # E3: the block's
    assert summary["tipo"] == ["reato"] and summary["commi_speciali"] == [{"comma": "3", "kind": "aggravamento"}]
    assert [s for s, kind, _ in E._schemas(got)][1:] and E._schemas(got)[1][1] == "aggravamento"
    assert sum(1 for a in summary["dropped"] if a["action"] == "mapped") == 4
    # a generated line read without its text: normalized, its mappings carried and counted too
    E._REPAIRED.clear()
    monkeypatch.setattr(E, "reviewed", lambda: {})
    line = {**_line_of(s, "991"), "checks": E.empty_checks(), "text": ""}
    _write(tmp_path / "schemas.jsonl", [line])
    got, why = E.for_label("991 c.p.")
    assert why == "schema" and E.all_dangling_refs(got["schema"]) == []
    assert sum(1 for a in got["dropped"] if a["action"] == "mapped") == 4


def test_a_lawyers_edit_is_stored_mapped(monkeypatch, tmp_path, show):
    """Review: --edit stores the schema as the checker reads it (a ref «e1» as «E1»), each mapping a warning."""
    from src.rag import element_schemas as E
    _write(tmp_path / "schemas.jsonl", [{**_line_of(_hives(), "991"), "checks": E.empty_checks()}])
    line, warnings = show.edit({"article": "991 c.p.", "reviewer": "avv",
                                "logic": {"op": "all_of", "args": [{"ref": "e1"}, {"ref": "E2"}, {"ref": "E3"}]}},
                               today="2026-11-19")
    assert line["schema"]["logic"]["args"][0] == {"ref": "E1"} and E.all_dangling_refs(line["schema"]) == []
    assert "vocabolario: logic: ref 'e1' -> 'E1'" in warnings


# --- 2. drops never dangle ---------------------------------------------------------------------------------------------

def test_an_all_of_left_with_one_child_is_that_child():
    from src.rag import element_schemas as E
    assert E.prune_logic({"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}]}, {"E2"}) == {"ref": "E1"}
    tree = {"op": "all_of", "args": [{"ref": "E1"}, {"op": "any_of", "args": [{"ref": "E2"}, {"ref": "E3"}]},
                                     {"op": "not", "arg": {"op": "all_of", "args": [{"ref": "E4"}, {"ref": "E5"}]}}]}
    assert E.prune_logic(tree, {"E3", "E5"}) == {"op": "all_of", "args": [
        {"ref": "E1"}, {"ref": "E2"}, {"op": "not", "arg": {"ref": "E4"}}]}      # a not keeps its single child
    assert E.prune_logic(tree, {"E1", "E2", "E3"}) == {"op": "not", "arg": {"op": "all_of", "args": [
        {"ref": "E4"}, {"ref": "E5"}]}}
    assert E.prune_logic({"op": "all_of", "args": [{"ref": "E1"}]}) == {"op": "all_of", "args": [{"ref": "E1"}]}
    cond = {"op": "conditional", "if": {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}]}, "then": {"ref": "E3"}}
    assert E.prune_logic(cond, {"E2"}) == {"op": "conditional", "if": {"ref": "E1"}, "then": {"ref": "E3"}}
    assert E.prune_logic(cond, {"E3"}) is None      # v79.12.5 (item 0): a condition with no consequence requires none


def test_refs_to_a_dropped_element_go_wherever_they_stand():
    from src.rag import element_schemas as E
    odd = {"op": "xor", "args": [{"ref": "E1"}, {"ref": "E2"}], "arg": {"ref": "E2"}}
    assert E.prune_logic(odd, {"E2"}) == {"op": "xor", "args": [{"ref": "E1"}]}
    assert E.prune_logic({"op": "all_of", "ref": "E2", "args": [{"ref": "E1"}, {"ref": "E3"}]}, {"E2"}) == \
        {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E3"}]}
    assert E.prune_logic({"ref": " E2 "}, {"E2"}) is None


def _tree_schema(text_ok):
    """Four elements under every op; text_ok[i] False makes element i+1 ungrounded (dropped by the repair)."""
    words = ["asporta le arnie altrui", "ne sottrae i favi", "senza il consenso del custode",
             "il fatto è commesso di notte"]
    bad = ["asporta le vespe altrui", "ne sottrae le api", "senza il permesso del sindaco", "il fatto avviene al mare"]
    els = [_el(f"E{i + 1}", words[i] if ok else bad[i]) for i, ok in enumerate(text_ok)]
    logic = {"op": "all_of", "args": [
        {"op": "any_of", "args": [{"ref": "E1"}, {"ref": "E2"}]},
        {"op": "not", "arg": {"op": "all_of", "args": [{"ref": "E3"}, {"ref": "E4"}]}},
        {"op": "conditional", "if": {"ref": "E4"}, "then": {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E3"}]},
         "else": {"ref": "E2"}}]}
    return {"tipo": ["reato"], "elementi": els, "logic": logic, "eccezioni": [], "commi_speciali": [
        {"comma": "3", "kind": "aggravamento", "schema": {"tipo": ["sanction"], "elementi": [
            _el("E1", words[3] if text_ok[3] else bad[3], comma="3"), _el("E2", words[0] if text_ok[0] else bad[0])],
            "logic": {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}]}}}]}


def _one_child_nodes(node, out):
    if isinstance(node, dict):
        if node.get("op") in ("all_of", "any_of") and len(node.get("args") or []) == 1:
            out.append(node)
        for a in node.get("args") or []:
            _one_child_nodes(a, out)
        for k in ("arg", "if", "then", "else"):
            _one_child_nodes(node.get(k), out)


@pytest.mark.parametrize("text_ok", list(itertools.product((True, False), repeat=4)))
def test_the_checker_never_outputs_a_dangling_ref(text_ok):
    """Every combination of dropped elements: no dangling ref anywhere in the output, no logic_errors on it, no
    all_of / any_of the pruning left with one child."""
    from src.rag import element_schemas as E
    s = _tree_schema(text_ok)
    fixed, actions = E.repair(s, HIVES)
    dropped = {a["id"] for a in actions if a["action"] == "dropped" and not a["scope"]}
    assert dropped == {f"E{i + 1}" for i, ok in enumerate(text_ok) if not ok}
    for sc, _, _ in E._schemas({"schema": fixed}):
        assert E.dangling_refs(sc) == []
        nodes = []
        _one_child_nodes(sc.get("logic"), nodes)
        assert nodes == []
    assert E.check(fixed, HIVES)["logic_errors"] == []


def test_a_dangling_ref_left_by_the_checker_is_a_bug(monkeypatch, tmp_path, caplog):
    from src.rag import element_schemas as E
    s = _tree_schema((True, True, False, True))
    monkeypatch.setattr(E, "prune_logic", lambda node, dropped=(), depth=0: node)      # a broken pruning
    with pytest.raises(E.DanglingRefError, match="ref E3"):
        E.repair(s, HIVES)
    monkeypatch.setenv("ELEMENT_SCHEMAS", "true")
    _write(tmp_path / "schemas.jsonl", [{**_line_of(s, "992"), "checks": E.empty_checks()}])
    row = {"_article": ("992", "c.p."), "d": {"name": PENAL_DOC}, "s": {"plain_text": HIVES}}
    E._REPAIRED.clear()
    with caplog.at_level(logging.ERROR, logger="src.rag.element_schemas"):
        assert E.for_row(row) == (None, "unusable:logic_errors")
    assert "lasciati dagli elementi tolti" in caplog.text


def _line_of(schema, number, code="c.p.", doc=PENAL_DOC, text=HIVES):
    from src.rag import element_schemas as E
    h = E.text_hash(text)
    return {"doc": doc, "article": number, "code": code, "label": f"{number} {code}", "schema_version": 1,
            "source_hash": h, "text_hash": h, "prompt_hash": E.prompt_hash(), "schema": schema, "text": text}


def test_a_dangling_ref_in_the_input_stays_a_logic_error():
    from src.rag import element_schemas as E
    s = _hives(logic={"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}, {"ref": "E3"}, {"ref": "E9"}]})
    checks = E.check(s, HIVES)
    assert checks["logic_errors"] == ["logic.all_of[2]: ref E9 senza elemento"] and not E.usable(   # E3 moved out
        {"schema": s, "checks": checks})
    fixed, _ = E.repair(s, HIVES)                                     # no drop explains it: no error raised
    assert E.dangling_refs(fixed) == ["E9"]


def test_a_ref_differing_only_by_case_or_spaces_is_mapped():
    from src.rag import element_schemas as E
    s = _hives(logic={"op": "all_of", "args": [{"ref": "e1"}, {"ref": " E2"}, {"ref": "E3"}]})
    checks = E.check(s, HIVES)
    assert checks["logic_errors"] == [] and checks["vocabulary_mapped"] == ["logic: ref 'e1' -> 'E1'",
                                                                            "logic: ref ' E2' -> 'E2'"]
    assert E.repair(s, HIVES)[0]["logic"] == {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}]}   # E3 moved


# --- 3. tipo «reato», «chiunque», the prompt, what is made again --------------------------------------------------------

def test_tipo_reato_and_destinatario_chiunque():
    from src.rag import element_schemas as E
    assert "reato" in E.TIPI and "reato" in E.DECOMPOSABLE_TIPI and "procedibilità" in E.KINDS
    for dest in ("chiunque", ["Chiunque"], [{"ruolo": "qualunque persona", "condizione": None}],
                 [{"ruolo": "chiunque", "condizione": None}]):
        checks = E.check(_hives(destinatario=dest), HIVES)
        assert checks["invalid"] == [], dest
    assert E.check(_hives(tipo=["reato"]), HIVES)["invalid"] == []
    assert E.opinion({"schema": _hives(tipo=["reato"])}) is True
    assert E.roles_match("imputato", "chiunque") == (True, False)
    shadow = E.shadow_admission("991 c.p.", {"schema": _hives(tipo=["reato"])}, "persona_offesa", "threat")
    assert shadow["contest"] is True and shadow["tipo"] == ["reato"]


def test_the_prompt_lists_every_vocabulary_and_its_hash_moved():
    from src.rag import element_schemas as E
    p = E.SCHEMA_SYSTEM
    for values in (E.TIPI, E.ROLES, E.KINDS, E.SPECIAL_KINDS, E.LOGIC_OPS, E.GROUNDING):
        for v in values:
            assert f'"{v}"' in p, v
    for needle in ('"reato"', "fattispecie incriminatrice", '"chiunque"', '"procedibilità"', "«ovvero»", "«oppure»",
                   "«od»", "«ed»", "VOCABOLARI", "querela"):
        assert needle in p, needle
    assert E.prompt_hash() != "f20fc6a8a0e8"                       # 79.12.2's prompt


def _article(number, text=HIVES, code="c.p."):
    return {"number": number, "code": code, "rubrica": "Arnie sintetiche", "text": text}


def _fake_call(seen):
    def call(endpoint, user, system=None, usage=None):
        seen.append(user.split("\n")[0])
        if usage is not None:
            usage.update({"prompt_tokens": 1, "completion_tokens": 1})
        return json.dumps(_hives())
    return call


END = {"url": "http://model.invalid:8003/v1", "key": "k", "model": "qwen38-27b", "max_tokens": 6000, "extra": {}}


def test_a_new_prompt_never_remakes_the_file_unasked(monkeypatch, tmp_path, bulk):
    from src.rag import element_schemas as E
    seen = []
    monkeypatch.setattr(bulk.ac, "call", _fake_call(seen))
    path = os.environ["ARTICLE_SCHEMAS_FILE"]
    items = [(PENAL_DOC, _article(n)) for n in ("991", "992", "993")]
    bulk.run_schemas(items, END, path, {}, 1, bulk.Throttle(0))
    have = bulk.read_file(path)
    for key in have:                                              # made with 79.12.2's prompt
        have[key]["prompt_hash"] = "f20fc6a8a0e8"
        have[key]["generation_hash"] = E.generation_hash(have[key]["source_hash"], "qwen38-27b", "f20fc6a8a0e8")
    assert bulk.schema_due(items[0][1], PENAL_DOC, have, "qwen38-27b") == "prompt"
    assert not bulk.is_due("prompt") and bulk.is_due("prompt", prompt=True)
    out = bulk.run_schemas(items, END, path, have, 1, bulk.Throttle(0))
    assert out["due"] == 0 and out["prompt_kept"] == 3 and out["why"] == {} and len(seen) == 3
    assert bulk.schema_due(items[0][1], PENAL_DOC, have, "another") == "generation"   # a new model still remakes
    out = bulk.run_schemas(items, END, path, have, 1, bulk.Throttle(0), prompt=True)
    assert out["due"] == 3 and out["why"] == {"prompt": 3} and len(seen) == 6


def test_only_keys_makes_exactly_the_keys_asked_never_a_reviewed_label(monkeypatch, tmp_path, bulk):
    from src.rag import element_schemas as E
    seen = []
    monkeypatch.setattr(bulk.ac, "call", _fake_call(seen))
    path = os.environ["ARTICLE_SCHEMAS_FILE"]
    items = [(PENAL_DOC, _article(n)) for n in ("991", "992", "993")]
    bulk.run_schemas(items, END, path, {}, 1, bulk.Throttle(0))
    have = bulk.read_file(path)
    keys_file = tmp_path / "regen.jsonl"
    keys_file.write_text(json.dumps({"doc": PENAL_DOC, "article": "992", "reason": "invalid"}) + "\n# a note\n" +
                         f"{PENAL_DOC}\t993\n\n", encoding="utf-8")
    keys = bulk.read_keys(str(keys_file))
    assert keys == {(PENAL_DOC, "992"), (PENAL_DOC, "993")}
    _write(tmp_path / "reviewed.jsonl", [{"label": "993 c.p.", "reviewed": True, "schema": _hives(),
                                          "source_hash": E.text_hash(HIVES)}])
    asked = [(d, a) for d, a in items if (d, a["number"]) in keys]
    out = bulk.run_schemas(asked, END, path, have, 1, bulk.Throttle(0), reviewed=bulk.reviewed_hashes(), asked=True)
    assert out["due"] == 1 and out["why"] == {"asked": 1} and out["reviewed_kept"] == 1
    assert seen[-1].startswith("ARTICOLO: art. 992 c.p.") and len(seen) == 4


def test_the_bulk_parser_takes_only_keys_and_regen_prompt(monkeypatch, bulk):
    got = {}
    monkeypatch.setattr(bulk, "schemas_main", lambda args: got.update(vars(args)))
    monkeypatch.setattr(sys, "argv", ["article_cards_bulk.py", "--schemas", "--only-keys", "k.jsonl", "--regen-prompt"])
    bulk.main()
    assert got["only_keys"] == "k.jsonl" and got["regen_prompt"] is True


# --- 4. the procedibilità block ------------------------------------------------------------------------------------------

# a synthetic 582-like offence (invented wording, an invented article number in the c.p. Libro II range)
STING = ("1. Chiunque cagiona a un apicoltore una lesione da puntura è punito con la reclusione da tre mesi a due anni.\n"
         "2. Se la lesione guarisce entro dieci giorni, il delitto è punibile a querela della persona offesa, salvo "
         "che il fatto sia commesso con un'arnia incendiata.\n"
         "3. La querela proposta è irrevocabile.")
# the store's own reading misses these words: «eccetto quando» is no «salvo che», «non può essere ritirata» no
# «irrevocabile» -- the schema's block is then the only source of the d'ufficio condition and of the remissione
SMOKE = ("1. Chiunque affumica l'alveare altrui è punito con la reclusione fino a un anno.\n"
         "2. Il delitto è punibile a querela della persona offesa, eccetto quando il fatto è commesso di notte.\n"
         "3. La querela non può essere ritirata.")
STING_NO, SMOKE_NO = "649-sexies", "649-septies"


def _sting(**over):
    schema = {"tipo": ["reato", "sanction"], "destinatario": ["chiunque"], "legittimato": [], "soggetto_passivo": [],
              "rapporto": "lesione da puntura",
              "elementi": [_el("E1", "cagiona a un apicoltore una lesione da puntura"),
                           _el("E2", "la lesione guarisce entro dieci giorni", kind="procedural", comma="2"),
                           _el("E3", "punibile a querela della persona offesa, salvo che il fatto sia commesso con "
                                     "un'arnia incendiata", kind="procedibilità", comma="2"),
                           _el("E4", "La querela proposta è irrevocabile", kind="querela", comma="3")],
              "logic": {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"},
                                                 {"op": "any_of", "args": [{"ref": "E3"}, {"ref": "E4"}]}]},
              "eccezioni": [], "conseguenza": {"testo": "è punito con la reclusione da tre mesi a due anni",
                                               "tipo": "pena", "comma": "1"},
              "commi_speciali": [], "collegati": []}
    schema.update(over)
    return schema


def _smoke_block(**over):
    block = {"regime": "querela", "comma": "2", "testo": "punibile a querela della persona offesa",
             "grounding": "explicit", "fonte": {"comma": "2", "testo": "punibile a querela della persona offesa"},
             "condizioni": [{"testo": "il fatto è commesso di notte", "effetto": "ufficio", "comma": "2",
                             "grounding": "explicit", "fonte": {"comma": "2", "testo": "il fatto è commesso di notte"}}],
             "remissione": "irrevocabile", "remissione_testo": "La querela non può essere ritirata", "termine": None}
    block.update(over)
    return {"tipo": ["reato"], "destinatario": ["chiunque"], "legittimato": [], "soggetto_passivo": [],
            "rapporto": "affumicamento", "elementi": [_el("E1", "affumica l'alveare altrui")], "logic": {"ref": "E1"},
            "eccezioni": [], "conseguenza": None, "commi_speciali": [], "collegati": [], "procedibilita": block}


def _row(number, text):
    return {"_article": (number, "c.p."), "d": {"name": PENAL_DOC}, "s": {"plain_text": text}}


def test_procedibilita_elements_leave_the_offence_for_the_block():
    """582-like: «a querela della persona offesa, salvo …», the querela's own condition (a «procedural» element mapped
    by item 1 to «competenza», re-read by its comma) and the irrevocable querela leave the elements and the logic --
    the logic left with the fattispecie alone (the all_of of one child collapsed) -- into the block; nothing dangles."""
    from src.rag import element_schemas as E
    s = _sting()
    checks = E.check(s, STING)
    assert checks["invalid"] == [] and checks["logic_errors"] == [] and E.usable({"schema": s, "checks": checks})
    assert [x.split(":")[0] for x in checks["procedibilita_moved"]] == ["E2", "E2", "E3", "E4"]
    assert "'procedural' -> 'procedibilità' (dal comma 2" in checks["procedibilita_moved"][0]
    fixed, actions = E.repair(s, STING)
    assert _ids(fixed) == ["E1"] and fixed["logic"] == {"ref": "E1"} and E.all_dangling_refs(fixed) == []
    block = fixed["procedibilita"]
    assert (block["regime"], block["comma"], block["remissione"]) == ("querela", "2", "irrevocabile")
    assert [(c["id"], c["testo"], c["effetto"], c["da"]) for c in block["condizioni"]] == [
        ("P1", "la lesione guarisce entro dieci giorni", "querela", "E2"),
        ("P2", "il fatto sia commesso con un'arnia incendiata", "ufficio", "E3")]
    assert [(a["id"], a["action"]) for a in actions if a["check"] == "procedibilita"] == [
        ("E2", "mapped"), ("E2", "moved"), ("E3", "moved"), ("E4", "moved")]
    assert s["elementi"][2]["kind"] == "procedibilità" and len(s["elementi"]) == 4          # the input untouched
    again, more = E.repair(fixed, STING)                                                   # idempotent
    assert more == [] and again["procedibilita"] == block
    facts = E.procedibilita_facts({"schema": fixed})
    assert facts["querela"] is True and facts["querela_comma"] == "co. 2"
    assert facts["ufficio_condizioni"] == [{"comma": "co. 2", "opener": "se", "source": "schema",
                                            "text": "il fatto sia commesso con un'arnia incendiata"}]
    assert facts["querela_condizioni"][0]["text"] == "la lesione guarisce entro dieci giorni"
    assert facts["remissione"] == "querela irrevocabile (co. 3)"


def test_the_decomposition_never_grades_a_procedibilita_condition(monkeypatch, tmp_path):
    from src.rag import element_schemas as E
    monkeypatch.setenv("ELEMENT_SCHEMAS", "true")
    line = E.make_line({"number": STING_NO, "code": "c.p.", "text": STING}, "m-test", _sting(), doc=PENAL_DOC)
    assert E.usable(line) and line["checks"]["procedibilita_moved"]
    _write(tmp_path / "schemas.jsonl", [line])
    got, why = E.for_row(_row(STING_NO, STING))
    assert why == "schema"
    reqs = E.requisites(_row(STING_NO, STING), got, {})
    assert [r["text"] for r in reqs] == ["cagiona a un apicoltore una lesione da puntura"]
    summary = E.summary(got)
    assert summary["altri_elementi"] == [] and summary["logic"] == {"ref": "E1"}
    assert summary["procedibilita"]["regime"] == "querela" and len(summary["procedibilita"]["condizioni"]) == 2
    from src.rag import interactive_flow
    counters = interactive_flow.element_counters({"records": {f"{STING_NO} c.p.": {
        "schema": summary, "requisites": [], "procedibilita_source": "schema"}}})
    assert counters["schema_procedibilita_moved"] == 3 and counters["procedibilita_from_schema"] == 1
    assert counters["schema_vocabulary_mapped"] == 3          # E2's two mappings and E4's «querela»


def test_a_procedibilita_comma_speciale_goes_to_the_block_whole():
    from src.rag import element_schemas as E
    s = _sting(elementi=[_el("E1", "cagiona a un apicoltore una lesione da puntura")], logic={"ref": "E1"},
               commi_speciali=[{"comma": "2", "kind": "querela", "schema": {
                   "tipo": ["procedural"], "elementi": [_el("E1", "la lesione guarisce entro dieci giorni",
                                                            kind="condizione", comma="2")],
                   "logic": {"ref": "E1"}, "eccezioni": [], "conseguenza": None}}])
    checks = E.check(s, STING)
    assert checks["invalid"] == [] and E.usable({"schema": s, "checks": checks})
    fixed, actions = E.repair(s, STING)
    assert fixed["commi_speciali"] == [] and _ids(fixed) == ["E1"]
    assert [(c["testo"], c["effetto"], c["da"]) for c in fixed["procedibilita"]["condizioni"]] == [
        ("la lesione guarisce entro dieci giorni", "querela", "co2.E1")]
    assert fixed["procedibilita"]["regime"] == "querela"
    assert ("2", "comma speciale", "procedibilita", "merged") in _acts(actions)     # review: no element moved
    line = {"schema": fixed, "base_comma": "1"}
    assert [r["text"] for r in E.requisites({"_article": (STING_NO, "c.p.")}, line, {})] == [
        "cagiona a un apicoltore una lesione da puntura"]


def test_an_ungrounded_condition_of_the_block_is_dropped_like_an_eccezione():
    from src.rag import element_schemas as E
    s = _smoke_block(condizioni=[{"testo": "il fatto è commesso di giorno", "effetto": "d'ufficio", "comma": "2",
                                  "grounding": "explicit"}, "il fatto è commesso di notte"])
    checks = E.check(s, SMOKE)
    assert any(x.startswith("procedibilita/P1") for x in checks["ungrounded"]) and E.usable({"schema": s,
                                                                                              "checks": checks})
    fixed, actions = E.repair(s, SMOKE)
    assert [(c["id"], c["testo"], c["effetto"]) for c in fixed["procedibilita"]["condizioni"]] == [
        ("P2", "il fatto è commesso di notte", "ufficio")]
    assert ("", "procedibilita.P1", "ungrounded", "dropped") in _acts(actions)


def test_the_store_reads_the_block_where_its_own_text_reading_is_silent(monkeypatch, tmp_path):
    """The store (here computed from the text: the same extract) reads «a querela» but neither the d'ufficio condition
    nor the irrevocable querela; with ELEMENT_SCHEMAS the block fills both -- the records' procedibilità conditions,
    the querela remedies (procedibility: irrevocable, then d'ufficio once the condition is proved) and the penal line
    read it -- logged with its source; the flag off, the facts are the store's as before."""
    from src.rag import element_schemas as E, issue_records, lawyer_method, procedural_facts as pf
    line = E.make_line({"number": SMOKE_NO, "code": "c.p.", "text": SMOKE}, "m-test", _smoke_block(), doc=PENAL_DOC)
    assert E.usable(line)
    _write(tmp_path / "schemas.jsonl", [line])
    label = f"{SMOKE_NO} c.p."
    remission = "Valutare la remissione della querela."

    plain = pf.from_row(_row(SMOKE_NO, SMOKE))          # the flag off: as before
    assert plain["querela"] is True and plain["ufficio_condizioni"] == [] and plain["remissione"] == pf.REMISSIONE_DEFAULT
    assert pf.facts_of(_row(SMOKE_NO, SMOKE)) == plain
    rows = [_row(SMOKE_NO, SMOKE)]
    assert pf.attach(None, rows) == {"store": 0, "computed": 1} and "procedibilita_source" not in rows[0]["_proc"]
    assert issue_records._conditions(_row(SMOKE_NO, SMOKE)) == []
    assert lawyer_method.procedibility({label: {}}, [_row(SMOKE_NO, SMOKE)]) == {label: "querela"}

    monkeypatch.setenv("ELEMENT_SCHEMAS", "true")
    rows = [_row(SMOKE_NO, SMOKE)]
    assert pf.attach(None, rows) == {"store": 0, "computed": 1, "schema": 1}
    facts = rows[0]["_proc"]
    assert facts["procedibilita_source"] == "schema" and facts["source"] == "computed"
    assert facts["procedibilita_from_schema"] == ["ufficio_condizioni", "remissione"]
    assert facts["ufficio_condizioni"][0]["text"] == "il fatto è commesso di notte"
    assert facts["remissione"] == "querela irrevocabile (co. 2)"
    assert facts["procedibilita"] == plain["procedibilita"]          # the store's own text kept
    assert pf.facts_of(_row(SMOKE_NO, SMOKE)) == {k: v for k, v in facts.items() if k != "source"}   # not attached
    conds = issue_records._conditions(rows[0])
    assert [(c["kind"], c["text"]) for c in conds] == [("procedibilita", "il fatto è commesso di notte")]
    assert lawyer_method.procedibility({label: {}}, rows) == {label: lawyer_method.IRREVOCABLE}
    proved = {label: {"procedibilita": [{**conds[0], "status": "provato"}]}}
    assert lawyer_method.procedibility(proved, rows) == {label: "ufficio"}
    assert lawyer_method.procedibility_out(remission, lawyer_method.procedibility({label: {}}, rows), label,
                                           penal=True) == ""
    assert "d'ufficio se il fatto è commesso di notte" in pf.procedibilita_text(facts)


def test_the_stores_own_value_wins_unless_the_schema_is_reviewed(monkeypatch, tmp_path, caplog):
    from src.rag import element_schemas as E, procedural_facts as pf
    monkeypatch.setenv("ELEMENT_SCHEMAS", "true")
    # the store reads «salvo che …» (a condition of its own) and art. 124's term (its reading misses «si presenta
    # nei sei mesi»); the schema says otherwise
    text = ("1. Chiunque affumica l'alveare altrui è punito con la reclusione fino a un anno.\n"
            "2. Il delitto è punibile a querela della persona offesa, salvo che il fatto sia commesso di notte. La "
            "querela si presenta nei sei mesi dal fatto.")
    s = _smoke_block(condizioni=[{"testo": "il fatto sia commesso", "effetto": "ufficio", "comma": "2",
                                  "grounding": "explicit"}], remissione=None, termine="sei mesi",
                     termine_testo="La querela si presenta nei sei mesi dal fatto")
    _write(tmp_path / "schemas.jsonl", [E.make_line({"number": SMOKE_NO, "code": "c.p.", "text": text}, "m", s,
                                                    doc=PENAL_DOC)])
    with caplog.at_level(logging.INFO, logger="src.rag.procedural_facts"):
        facts = pf.facts_of(_row(SMOKE_NO, text))
    assert facts["ufficio_condizioni"][0]["text"] == "il fatto sia commesso di notte"          # the store's
    assert facts["procedibilita_conflicts"]["ufficio_condizioni"]["kept"] == "store"
    assert facts["querela_termine"] == "sei mesi (co. 2)" and facts["procedibilita_from_schema"] == ["querela_termine"]
    assert "procedibilità from schema" in caplog.text and "ufficio_condizioni kept store" in caplog.text
    # a regime the store contradicts: the store's facts whole
    ufficio = "1. Chiunque affumica l'alveare altrui è punito con la reclusione fino a un anno.\n2. Si procede d'ufficio."
    s2 = _smoke_block(testo=None, grounding="inferred", condizioni=[], remissione=None)
    _write(tmp_path / "schemas.jsonl", [E.make_line({"number": SMOKE_NO, "code": "c.p.", "text": ufficio}, "m", s2,
                                                    doc=PENAL_DOC)])
    facts = pf.facts_of(_row(SMOKE_NO, ufficio))
    assert facts["querela"] is False and facts["procedibilita_source"] == "computed"
    assert facts["procedibilita_conflicts"]["querela"]["kept"] == "store"
    # the lawyer's schema (reviewed) wins
    _write(tmp_path / "reviewed.jsonl", [{"label": f"{SMOKE_NO} c.p.", "reviewed": True, "reviewer": "avv",
                                          "source_hash": E.text_hash(ufficio), "schema_version": 1, "schema": s2}])
    E._REPAIRED.clear()
    facts = pf.facts_of(_row(SMOKE_NO, ufficio))
    assert facts["querela"] is True and facts["procedibilita_source"] == "reviewed"
    assert facts["procedibilita"].startswith("a querela della persona offesa (co. 2)")
    assert facts["procedibilita_conflicts"]["querela"]["kept"] == "schema"


def test_the_prompt_describes_the_block():
    from src.rag import element_schemas as E
    assert '"procedibilita": {...}}' in E.SCHEMA_SYSTEM and "12. procedibilita:" in E.SCHEMA_SYSTEM
    assert "FUORI dagli elementi e dalla logic" in E.SCHEMA_SYSTEM and '"effetto": "ufficio" | "querela"' in \
        E.SCHEMA_SYSTEM


def test_recheck_counts_the_moves_and_a_line_of_procedibilita_alone(tmp_path, show):
    from src.rag import element_schemas as E
    only = _sting(elementi=[_el("E3", "punibile a querela della persona offesa", kind="procedural", comma="2")],
                  logic={"ref": "E3"})
    _write(tmp_path / "schemas.jsonl", [{**_line_of(_sting(), STING_NO, text=STING), "checks": E.empty_checks()},
                                        {**_line_of(only, "649-octies", text=STING), "checks": E.empty_checks()}])
    r = show.recheck(write=False)
    assert r["procedibilita_moved"] == 4 and r["procedibilita_blocks"] == 2
    assert r["usable_after"] == 1 and r["unusable_after"] == {"no_elements": 1}
    assert any("Procedibilità: 4 elementi spostati nel blocco, 2 schemi con il blocco." == x
               for x in show.recheck_lines(r))


# --- 5. --recheck on the server's failure shapes -----------------------------------------------------------------------

def _stored(line, **checks):
    """A line as 79.12.2's --recheck left it: its checks by that rule, "dropped" and "rechecked_at" set."""
    from src.rag import element_schemas as E
    return {**line, "checks": {**E.empty_checks(), **checks}, "dropped": [], "rechecked_at": "2026-11-18T00:00:00"}


def _server_file(tmp_path):
    from src.rag import element_schemas as E
    procedural = _hives()
    procedural["elementi"][2]["kind"] = "procedural"
    ovvero = _hives()
    ovvero["elementi"][0]["connective"] = "ovvero"
    collapsing = _hives(logic={"op": "all_of", "args": [{"op": "any_of", "args": [{"ref": "E1"}, {"ref": "E4"}]},
                                                        {"ref": "E2"}, {"ref": "E3"}]})
    collapsing["elementi"].append(_el("E4", "ne sottrae le api"))                    # ungrounded: dropped
    dangling = _hives(logic={"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}, {"ref": "E9"}]})
    to_base = _hives()
    to_base["commi_speciali"][0]["schema"]["logic"] = {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}]}
    to_base["commi_speciali"][0]["schema"]["elementi"][0]["id"] = "E2"
    emptied = _deposit(elementi=[_el("E1", "lascia marcire la merce altrui")], logic={"ref": "E1"})
    unknown = _deposit()
    unknown["elementi"][0]["kind"] = "azione"
    lines = [_stored(_line_of(procedural, "801"), invalid=["E3: kind sconosciuto 'procedural'"]),
             _stored(_line_of(ovvero, "802"), invalid=["E1: connective 'ovvero' non valido"]),
             _stored(_line_of(collapsing, "803", "c.c.", CIVIL_DOC)),
             _stored(_line_of(dangling, "804"), logic_errors=["logic.all_of[2]: ref E9 senza elemento"]),
             _stored(_line_of(to_base, "805"), logic_errors=["co.3/logic.all_of[0]: ref E1 senza elemento"]),
             E.make_line({"number": "806", "code": "c.c.", "text": DEPOSIT}, "m", None, failed="HTTP 500",
                         doc=CIVIL_DOC),
             _stored(_line_of(emptied, "807", "c.c.", CIVIL_DOC, DEPOSIT)),
             _stored(_line_of(unknown, "808", "c.c.", CIVIL_DOC, DEPOSIT), invalid=["E1: kind sconosciuto 'azione'"])]
    _write(tmp_path / "schemas.jsonl", lines)
    _write(tmp_path / "reviewed.jsonl", [{"label": "808 c.c.", "reviewed": True, "reviewer": "avv",
                                          "source_hash": E.text_hash(DEPOSIT), "schema_version": 1,
                                          "schema": _deposit(), "text": DEPOSIT}])
    return lines


def test_recheck_on_the_servers_failure_shapes(tmp_path, show):
    lines = _server_file(tmp_path)
    regen_file = tmp_path / "out" / "regen.jsonl"
    r = show.recheck(now="2026-11-19T00:00:00+00:00", regen_list=str(regen_file))
    # 807: every element dropped -- the runtime reads it on the text path, so it is not usable after (review)
    # v79.12.5 (item 0, review: counts, not rejections): 808's unknown kind «azione» is read as «other» and counted,
    # so 808 is usable after (it was invalid, and on the regeneration list but for the lawyer's review)
    assert (r["articles"], r["schemas"], r["usable_before"], r["usable_after"]) == (8, 7, 2, 4)
    assert r["by_code"] == {"c.c.": {"articles": 4, "schemas": 3, "before": 2, "after": 2},
                            "c.p.": {"articles": 4, "schemas": 4, "before": 0, "after": 2}}
    assert r["mapped"] == {"kind 'procedural' -> 'procedibilità'": 1, "connective 'ovvero' -> 'o'": 1,
                           "kind 'azione' -> 'other'": 1}
    assert (r["mapped_total"], r["mapped_schemas"]) == (3, 3)
    assert r["unknown"] == {"kind 'azione'": 1} and r["unknown_schemas"] == 1
    assert r["dropped"] == {"ungrounded": 2} and r["fixed"] == {"logic_incomplete": 2}
    assert r["unusable_after"] == {"logic_errors": 2, "schema": 1, "no_elements": 1}
    assert r["regen"] == {"logic_errors": 2, "schema": 1, "no_elements": 1} and r["reviewed_skipped"] == 0
    keys = [json.loads(x) for x in regen_file.read_text(encoding="utf-8").splitlines()]
    assert [(k["article"], k["reason"]) for k in keys] == [("804", "logic_errors"), ("805", "logic_errors"),
                                                           ("806", "schema"), ("807", "no_elements")]
    assert keys[0]["detail"].startswith("ref_unknown: E9") and keys[1]["detail"].startswith("ref_to_base: co3.E1")
    assert keys[2]["detail"] == "HTTP 500" and keys[0]["doc"] == PENAL_DOC and keys[0]["label"] == "804 c.p."
    after = [json.loads(x) for x in (tmp_path / "schemas.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [x["schema"] for x in after] == [x["schema"] for x in lines]                     # no schema touched
    assert after[0]["checks"]["invalid"] == [] and after[0]["checks"]["vocabulary_mapped"] == [
        "E3: kind 'procedural' -> 'procedibilità'"]
    assert ("", "E3", "vocabulary", "mapped") in _acts(after[0]["dropped"])
    collapsed = [a for a in after[2]["dropped"] if a["check"] == "logic_incomplete"]
    assert collapsed and collapsed[0]["why"] == "nodi vuoti o riferimenti a elementi tolti rimossi"
    again = show.recheck()
    assert again["usable_before"] == again["usable_after"] == 4                 # a second run: nothing changes
    assert keys[3]["detail"] == "ogni elemento tolto dal controllo"
    bulk = _load("article_cards_bulk")
    assert bulk.read_keys(str(regen_file)) == {(PENAL_DOC, "804"), (PENAL_DOC, "805"), (CIVIL_DOC, "806"),
                                               (CIVIL_DOC, "807")}


def test_recheck_prints_before_after_per_code_mappings_and_the_regen_list(tmp_path, show, capsys):
    _server_file(tmp_path)
    before = (tmp_path / "schemas.jsonl").read_bytes()
    regen_file = tmp_path / "regen.jsonl"
    assert show.main(["--recheck", "--dry-run", "--regen-list", str(regen_file)]) == 0
    out = capsys.readouterr().out
    assert "Schemi usabili: prima 2, dopo 4 (+2)." in out          # v79.12.5 (item 0, review): 808 counted, usable
    assert "Per codice (usabili prima -> dopo, su articoli): c.c. 2 -> 2 su 4; c.p. 0 -> 2 su 4." in out
    assert "Vocabolario mappato: 3 valori in 3 schemi:" in out and "kind 'procedural' -> 'procedibilità'" in out
    assert "Da rigenerare ancora: 4 (logic_errors 2, no_elements 1, schema 1)." in out
    assert f"--only-keys {regen_file}" in out and "Nessun file scritto" in out
    assert (tmp_path / "schemas.jsonl").read_bytes() == before and len(regen_file.read_text().splitlines()) == 4


# --- 6. polarity prepositions --------------------------------------------------------------------------------------------

SENZA = "1. Chiunque, senza il consenso del custode, asporta le arnie è punito con la multa contro il reo."
CON = "1. Chiunque, con il consenso del custode, ritira le arnie, salvo che il fatto sia lieve, è punito."


def test_polarity_prepositions_are_content_words():
    from src.rag import element_schemas as E
    for p in ("con", "senza", "contro", "salvo", "tranne", "eccetto"):
        assert p in E.POLARITY_PREPOSITIONS and p not in E.FUNCTION_WORDS and E.grounding_words(f"{p} il bene") == \
            [p, "bene"]
    assert not {"non", "senza", "salvo", "piu", "meno", "dopo", "entro", "sopra", "sotto"} & E.FUNCTION_WORDS


def test_con_never_grounds_against_senza_both_ways():
    from src.rag import element_schemas as E
    # «con» stands in SENZA («punito con la multa») but never before «consenso»: the pair is missing
    assert E.missing_words("con il consenso del custode", SENZA) == ["con consenso"]
    assert E.missing_words("senza il consenso del custode", SENZA) == []
    assert E.missing_words("senza il consenso del custode", CON) == ["senza"]
    assert E.missing_words("con il consenso del custode", CON) == []
    assert E.missing_words("con il previo consenso del custode", CON) == ["previo"]
    assert E.missing_words("con il consenso", "Chi agisce con il previo e scritto consenso del custode") == []
    assert E.missing_words("contro il reo", SENZA) == []
    assert E.missing_words("contro il custode", SENZA) == ["contro custode"]
    assert E.missing_words("salvo che il fatto sia lieve", CON) == []
    assert E.missing_words("tranne che il fatto sia lieve", CON) == ["tranne"]
    assert E.missing_words("eccetto il custode", CON) == ["eccetto"]
    s = {"tipo": ["reato"], "elementi": [_el("E1", "asporta le arnie"),
                                         _el("E2", "con il consenso del custode", kind="condizione")],
         "logic": {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}]}, "eccezioni": [], "commi_speciali": []}
    fixed, actions = E.repair(s, SENZA)
    assert _ids(fixed) == ["E1"] and fixed["logic"] == {"ref": "E1"}             # dropped, the all_of collapsed
    assert ("", "E2", "ungrounded", "dropped") in _acts(actions)
    assert actions[0]["why"] == "parole assenti dal testo (con consenso)"


# --- review fixes ------------------------------------------------------------------------------------------------------

def test_contracted_and_agreeing_forms_are_the_preposition_itself():
    """«col», «coi», «coll'» are «con»; «salva», «salvi», «salve» are «salvo» -- never dropped as function words
    (polarity kept) nor missing against the other form (79.12.2 grounded them)."""
    from src.rag import element_schemas as E
    m = E.missing_words
    assert m("col consenso di lui", "Chiunque cagiona la morte di un uomo senza il consenso di lui") == ["con"]
    assert m("con il consenso di lui", "Chiunque cagiona la morte di un uomo, col consenso di lui") == []
    assert m("senza il consenso di lui", "Chiunque cagiona la morte di un uomo, col consenso di lui") == ["senza"]
    assert m("coi mezzi", "con i mezzi") == m("con i mezzi", "coi mezzi") == []
    assert m("con l'aiuto del custode", "coll'aiuto del custode") == []
    assert m("salvo i casi previsti dalla legge", "La domanda è ammessa, salvi i casi previsti dalla legge.") == []
    assert m("salvo diversa pattuizione", "salva diversa pattuizione delle parti") == []
    assert m("salve le eccezioni", "salvo le eccezioni di rito") == []
    s = {"tipo": ["claim-conferring"], "elementi": [_el("E1", "con il consenso del coniuge", kind="condizione")],
         "logic": {"ref": "E1"}, "eccezioni": [], "commi_speciali": []}
    fixed, actions = E.repair(s, "Il contratto è stipulato col consenso del coniuge.")
    assert _ids(fixed) == ["E1"] and not [a for a in actions if a["action"] == "dropped"]


def test_a_preposition_governs_its_coordinated_words():
    """«con dolo e consenso» never grounds against «con dolo e senza consenso»; a coordinated word the text governs by
    the same preposition (or by none) still grounds."""
    from src.rag import element_schemas as E
    m = E.missing_words
    assert m("con dolo e consenso", "Chi agisce con dolo e senza consenso.") == ["con consenso"]
    assert m("con violenza o minaccia", "con violenza, senza minaccia") == ["con minaccia"]
    assert m("con violenza o minaccia", "Chi con violenza o minaccia costringe altri") == []
    assert m("con violenza o con minaccia", "Chi con violenza o minaccia costringe altri") == []
    assert m("senza dolo né colpa", "Chi agisce senza dolo né colpa non risponde") == []
    assert m("senza dolo né colpa", "Chi agisce senza dolo, con colpa, risponde") == ["senza colpa"]
    # no coordination: a later word elsewhere governed by another preposition never fails the element
    assert m("senza giustificato motivo si allontana dal domicilio",
             "Chiunque si allontana, con qualsiasi mezzo, dal domicilio senza giustificato motivo") == []


def test_ovvero_sia_is_a_connective_synonym():
    from src.rag import element_schemas as E
    assert E.canon_connective("ovvero sia") == E.canon_connective("Ovvero  sia") == "o"
    s = _hives()
    s["elementi"][0]["connective"] = "ovvero sia"
    assert E.check(s, HIVES)["invalid"] == []


def test_a_models_dangling_ref_beside_a_dropped_one_is_no_checker_bug():
    """Review: a node with both ref and op -- a dangling ref in its args is the model's, seen before the pruning too:
    logic_errors as before, never DanglingRefError."""
    from src.rag import element_schemas as E
    text = "Chiunque cagiona ad alcuno una lesione personale è punito con la reclusione."
    s = {"tipo": ["reato"], "elementi": [_el("E1", "cagiona ad alcuno una lesione personale"),
                                         _el("E2", "volando sulla luna")],
         "logic": {"op": "all_of", "ref": "E2", "args": [{"ref": "E1"}, {"ref": "E9"}]},
         "eccezioni": [], "commi_speciali": []}
    assert E.dangling_refs(s) == ["E9"]                       # the model's, in the args of a node with a ref too
    assert E.check(s, text)["logic_errors"] == ["logic.all_of[1]: ref E9 senza elemento"]
    fixed, actions = E.repair(s, text)
    assert _ids(fixed) == ["E1"] and E.dangling_refs(fixed) == ["E9"]
    assert ("", "E2", "ungrounded", "dropped") in _acts(actions)
    assert E.dangling_refs({"elementi": [_el("E1", "x")],
                            "logic": {"op": "all_of", "ref": "E1", "args": [{"ref": "E7"}]}}) == ["E7"]


def test_recheck_a_schema_written_with_no_element_is_listed(tmp_path, show):
    """Review: a schema the model wrote with no element is never «usable after» and is listed for regeneration."""
    from src.rag import element_schemas as E
    text = "Chiunque cagiona ad altri un danno ingiusto è punito con la reclusione."
    empty = {"tipo": ["reato"], "destinatario": ["chiunque"], "elementi": [], "eccezioni": [], "commi_speciali": []}
    line = {**_line_of(empty, "811", text=text), "checks": E.empty_checks()}
    no_text = {**_line_of(empty, "812", text=text), "checks": E.empty_checks(), "text": ""}
    _write(tmp_path / "schemas.jsonl", [line, no_text])
    regen_file = tmp_path / "regen.jsonl"
    r = show.recheck(write=False, regen_list=str(regen_file))
    assert (r["usable_before"], r["usable_after"]) == (2, 0)
    assert r["regen"] == {"no_elements": 2} and r["unusable_after"] == {"no_elements": 2}
    keys = [json.loads(x) for x in regen_file.read_text(encoding="utf-8").splitlines()]
    assert [(k["article"], k["detail"]) for k in keys] == [("811", "nessun elemento nello schema"),
                                                           ("812", "nessun elemento nello schema")]


# --- review fixes: the procedibilità block (group 2) ---------------------------------------------------------------------

# a querela offence whose text says nothing of the remissione, the querela's term or a d'ufficio case
BARE_QUERELA = ("1. Chiunque affumica l'alveare altrui è punito con la reclusione fino a un anno.\n"
                "2. Il delitto è punibile a querela della persona offesa.")


def test_the_blocks_remissione_and_termine_need_the_texts_words(monkeypatch, tmp_path):
    """Review: a remissione or a term the model wrote with no passage of the text (or a passage the text lacks) is
    reported and dropped -- the store's defaults (art. 124, art. 152) never replaced by the model's word alone, the
    remission remedy never suppressed by it; with the passage in the text it stands."""
    from src.rag import element_schemas as E, lawyer_method, procedural_facts as pf
    made_up = _smoke_block(condizioni=[], remissione="irrevocabile", remissione_testo=None, termine="sei mesi")
    checks = E.check(made_up, BARE_QUERELA)
    assert {x.split(":")[0] for x in checks["ungrounded"]} == {"procedibilita/remissione", "procedibilita/termine"}
    assert E.usable({"schema": made_up, "checks": checks})
    fixed, actions = E.repair(made_up, BARE_QUERELA)
    assert (fixed["procedibilita"]["remissione"], fixed["procedibilita"]["termine"]) == (None, None)
    assert ("", "procedibilita.remissione", "ungrounded", "dropped") in _acts(actions)
    assert ("", "procedibilita.termine", "ungrounded", "dropped") in _acts(actions)
    cited = _smoke_block(condizioni=[], remissione_testo="La querela proposta è irrevocabile")   # not in the text
    assert E.repair(cited, BARE_QUERELA)[0]["procedibilita"]["remissione"] is None
    wrong = _smoke_block(condizioni=[], remissione=None, termine="sei mesi",
                         termine_testo="Il delitto è punibile a querela della persona offesa")   # no term in it
    assert E.repair(wrong, BARE_QUERELA)[0]["procedibilita"]["termine"] is None
    monkeypatch.setenv("ELEMENT_SCHEMAS", "true")
    _write(tmp_path / "schemas.jsonl", [E.make_line({"number": SMOKE_NO, "code": "c.p.", "text": BARE_QUERELA}, "m",
                                                    made_up, doc=PENAL_DOC)])
    facts = pf.facts_of(_row(SMOKE_NO, BARE_QUERELA))
    assert facts["remissione"] == pf.REMISSIONE_DEFAULT and facts["querela_termine"] == pf.QUERELA_DEFAULT
    assert "procedibilita_from_schema" not in facts
    label = f"{SMOKE_NO} c.p."
    assert lawyer_method.procedibility({label: {}}, [_row(SMOKE_NO, BARE_QUERELA)]) == {label: "querela"}
    # the text's own words: the statement of the article (no passage needed) or the passage cited
    said = BARE_QUERELA + "\n3. La querela proposta è irrevocabile."
    assert E.repair(made_up, said)[0]["procedibilita"]["remissione"] == "irrevocabile"


def test_an_inferred_condition_of_the_block_never_reaches_the_store(monkeypatch, tmp_path):
    """Review: a condition the model inferred (not the text's words) is dropped from the block -- the records never
    ask the user to prove it, the penal line never quotes it as the comma's own."""
    from src.rag import element_schemas as E, issue_records, procedural_facts as pf
    s = _smoke_block(condizioni=[{"testo": "il fatto è commesso con armi", "effetto": "ufficio", "comma": "2",
                                  "grounding": "inferred", "spiegazione": "di regola"}], remissione=None)
    checks = E.check(s, BARE_QUERELA)
    assert any(x.startswith("procedibilita/P1: condizione dedotta") for x in checks["ungrounded"])
    fixed, actions = E.repair(s, BARE_QUERELA)
    assert fixed["procedibilita"]["condizioni"] == []
    assert ("", "procedibilita.P1", "ungrounded", "dropped") in _acts(actions)
    monkeypatch.setenv("ELEMENT_SCHEMAS", "true")
    _write(tmp_path / "schemas.jsonl", [E.make_line({"number": SMOKE_NO, "code": "c.p.", "text": BARE_QUERELA}, "m",
                                                    s, doc=PENAL_DOC)])
    row = _row(SMOKE_NO, BARE_QUERELA)
    assert pf.facts_of(row)["ufficio_condizioni"] == [] and issue_records._conditions(row) == []
    assert "armi" not in pf.procedibilita_text(pf.facts_of(row))


def test_a_rule_on_the_querela_itself_keeps_its_elements():
    """Review: only an offence's schema (tipo «reato» / «sanction») sends its procedibilità to the block -- a rule on
    the querela itself keeps its requirements in its elements and logic; a block written in such a schema is dropped,
    logged."""
    from src.rag import element_schemas as E
    text = ("1. La querela è proposta mediante dichiarazione scritta presentata al pubblico ministero o a un ufficiale "
            "di polizia giudiziaria, entro il termine stabilito dalla legge.")
    s = {"tipo": ["procedural"], "destinatario": [], "legittimato": [], "soggetto_passivo": [], "rapporto": "querela",
         "elementi": [_el("E1", "La querela è proposta mediante dichiarazione scritta", kind="procedural"),
                      _el("E2", "presentata al pubblico ministero o a un ufficiale di polizia giudiziaria",
                          kind="procedural")],
         "logic": {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}]}, "eccezioni": [], "conseguenza": None,
         "commi_speciali": [], "collegati": []}
    assert not E.is_offence_schema(s) and E.is_offence_schema(_sting())
    checks = E.check(s, text)
    assert checks["procedibilita_moved"] == [] and checks["invalid"] == []
    fixed, actions = E.repair(s, text)
    assert _ids(fixed) == ["E1", "E2"] and fixed["logic"] == s["logic"] and "procedibilita" not in fixed
    assert not [a for a in actions if a["check"] == "procedibilita"]
    with_block = {**s, "procedibilita": {"regime": "querela", "condizioni": ["La querela è proposta"]}}
    fixed, actions = E.repair(with_block, text)
    assert "procedibilita" not in fixed and _ids(fixed) == ["E1", "E2"]
    assert ("", "procedibilita", "procedibilita", "dropped") in _acts(actions)


def test_a_procedibilita_passage_of_any_kind_leaves_the_offence():
    """Review: a passage that states the procedibilità leaves the offence whatever kind the model gave it
    («condizione», «other»), relabelled and logged; a «salvo che …» of the querela's sentence is the block's d'ufficio
    condition, never a fragment dropped; a «d'ufficio» of the conduct («i doveri d'ufficio») stays."""
    from src.rag import element_schemas as E
    text = ("1. Il custode che, violando i doveri d'ufficio, apre le arnie altrui è punito con la reclusione fino a un "
            "anno.\n2. Il delitto è punibile a querela della persona offesa, salvo che il fatto sia commesso con "
            "un'arnia incendiata.")
    conduct = _el("E1", "violando i doveri d'ufficio, apre le arnie altrui")
    for kind in ("condizione", "other"):
        s = _sting(elementi=[conduct, _el("E2", "Il delitto è punibile a querela della persona offesa", kind=kind,
                                          comma="2")],
                   logic={"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}]})
        checks = E.check(s, text)
        assert checks["procedibilita_moved"][0] == f"E2: kind '{kind}' -> 'procedibilità' (il suo testo dice la " \
                                                   "procedibilità)"
        fixed, _ = E.repair(s, text)
        assert _ids(fixed) == ["E1"] and fixed["logic"] == {"ref": "E1"}
        assert fixed["procedibilita"]["regime"] == "querela" and fixed["procedibilita"]["comma"] == "2"
        assert [r["text"] for r in E.requisites({"_article": (STING_NO, "c.p.")}, {"schema": fixed,
                                                                                  "base_comma": "1"}, {})] == [
            "violando i doveri d'ufficio, apre le arnie altrui"]
    s = _sting(elementi=[conduct, _el("E2", "salvo che il fatto sia commesso con un'arnia incendiata",
                                      kind="negazione", comma="2")],
               logic={"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}]})
    fixed, actions = E.repair(s, text)
    assert _ids(fixed) == ["E1"] and not [a for a in actions if a["check"] == "fragments"]
    assert [(c["testo"], c["effetto"]) for c in fixed["procedibilita"]["condizioni"]] == [
        ("il fatto sia commesso con un'arnia incendiata", "ufficio")]
    assert fixed["procedibilita"]["regime"] == "querela"
    assert not E.states_procedibilita("violando i doveri d'ufficio") and E.states_procedibilita(
        "si procede d'ufficio") and E.states_procedibilita("perseguibile, tuttavia, d'ufficio")


def test_the_if_of_a_conditional_whose_then_is_the_querela_goes_with_it():
    """Review: in the 582-shaped logic «if the lesion heals within ten days then a querela», the «if» is the
    querela's own condition: it leaves the offence with it (by the comma's opening clause, or by the conditional
    alone) -- no conditional left with only its «if», no requisite graded for it."""
    from src.rag import element_schemas as E
    lead = ("1. Chiunque cagiona a un apicoltore una lesione da puntura è punito con la reclusione da tre mesi a due "
            "anni.\n2. Se la lesione guarisce entro dieci giorni, il delitto è punibile a querela della persona "
            "offesa, salvo che il fatto sia commesso con un'arnia incendiata.")
    after = ("1. Chiunque cagiona a un apicoltore una lesione da puntura è punito con la reclusione da tre mesi a due "
             "anni.\n2. Il delitto è punibile a querela della persona offesa quando la lesione guarisce entro dieci "
             "giorni.")
    for text, why in ((lead, "condizione di procedibilità del comma 2"),
                      (after, "condizione di un conditional che porta alla procedibilità")):
        s = _sting(elementi=[_el("E1", "cagiona a un apicoltore una lesione da puntura"),
                             _el("E2", "la lesione guarisce entro dieci giorni", kind="condizione", comma="2"),
                             _el("E3", "il delitto è punibile a querela della persona offesa", kind="procedibilità",
                                 comma="2")],
                   logic={"op": "all_of", "args": [{"ref": "E1"}, {"op": "conditional", "if": {"ref": "E2"},
                                                                   "then": {"ref": "E3"}}]})
        checks = E.check(s, text)
        assert checks["logic_incomplete"] == [] and f"E2: kind 'condizione' -> 'procedibilità' ({why})" in \
            checks["procedibilita_moved"]
        fixed, _ = E.repair(s, text)
        assert _ids(fixed) == ["E1"] and fixed["logic"] == {"ref": "E1"} and E.all_dangling_refs(fixed) == []
        assert [(c["testo"], c["effetto"]) for c in fixed["procedibilita"]["condizioni"]] == [
            ("la lesione guarisce entro dieci giorni", "querela")]
        assert [r["text"] for r in E.requisites({"_article": (STING_NO, "c.p.")}, {"schema": fixed,
                                                                                  "base_comma": "1"}, {})] == [
            "cagiona a un apicoltore una lesione da puntura"]
    # an «if» whose «then» is the offence's own element stays
    s = _sting(elementi=[_el("E1", "cagiona a un apicoltore una lesione da puntura"),
                         _el("E2", "la lesione guarisce entro dieci giorni", kind="condizione", comma="2"),
                         _el("E3", "il delitto è punibile a querela della persona offesa", kind="procedibilità",
                             comma="2")],
               logic={"op": "all_of", "args": [{"op": "conditional", "if": {"ref": "E2"}, "then": {"ref": "E1"}},
                                               {"ref": "E3"}]})
    assert _ids(E.repair(s, after)[0]) == ["E1", "E2"]


def test_the_stores_general_rule_yields_to_a_querela_the_text_states(monkeypatch, tmp_path):
    """Review: the store's d'ufficio by the general rule alone (its reading found no «a querela» in «si procede su
    querela») is unstated -- a querela the schema reads in the text's own words is the article's (logged as a
    conflict the schema wins, counted), art. 124's term and art. 152's remissione the defaults; an inferred regime
    never replaces it."""
    from src.rag import element_schemas as E, interactive_flow, lawyer_method, procedural_facts as pf
    monkeypatch.setenv("ELEMENT_SCHEMAS", "true")
    text = ("1. Chiunque affumica l'alveare altrui è punito con la reclusione fino a un anno.\n"
            "2. Si procede su querela della persona offesa.")
    s = _sting(elementi=[_el("E1", "affumica l'alveare altrui"),
                         _el("E2", "Si procede su querela della persona offesa", kind="procedibilità", comma="2")],
               logic={"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}]})
    _write(tmp_path / "schemas.jsonl", [E.make_line({"number": SMOKE_NO, "code": "c.p.", "text": text}, "m", s,
                                                    doc=PENAL_DOC)])
    row = _row(SMOKE_NO, text)
    assert pf.from_row(row)["procedibilita"] == pf.UFFICIO_DEFAULT
    facts = pf.facts_of(row)
    assert facts["querela"] is True and facts["procedibilita"] == "a querela della persona offesa (co. 2)"
    assert facts["procedibilita_source"] == "schema" and "querela" in facts["procedibilita_from_schema"]
    assert facts["procedibilita_conflicts"]["querela"]["kept"] == "schema"
    assert (facts["querela_termine"], facts["remissione"]) == (pf.QUERELA_DEFAULT, pf.REMISSIONE_DEFAULT)
    label = f"{SMOKE_NO} c.p."
    assert lawyer_method.procedibility({label: {}}, [row]) == {label: "querela"}
    counters = interactive_flow.element_counters({"records": {label: {"schema": {}, "requisites": [],
                                                                      "procedibilita_source": "schema"}}})
    assert counters["procedibilita_from_schema"] == 1
    inferred = _smoke_block(testo=None, grounding="inferred", condizioni=[], remissione=None)
    _write(tmp_path / "schemas.jsonl", [E.make_line({"number": SMOKE_NO, "code": "c.p.", "text": text}, "m",
                                                    inferred, doc=PENAL_DOC)])
    facts = pf.facts_of(_row(SMOKE_NO, text))
    assert facts["querela"] is False and facts["procedibilita_conflicts"]["querela"]["kept"] == "store"


def test_the_counters_count_only_the_elements_moved(monkeypatch, tmp_path, show):
    """Review: a comma speciale moved whole and a sub-schema's block merged are «merged» -- logged, never counted as
    elements moved (nightly schema_procedibilita_moved, --recheck procedibilita_moved) nor as fixes."""
    from src.rag import element_schemas as E, interactive_flow
    monkeypatch.setenv("ELEMENT_SCHEMAS", "true")
    sub = {"tipo": ["procedural"], "elementi": [_el("E1", "la lesione guarisce entro dieci giorni", kind="condizione",
                                                    comma="2")],
           "logic": {"ref": "E1"}, "eccezioni": [], "conseguenza": None,
           "procedibilita": {"regime": "querela", "condizioni": []}}
    s = _sting(elementi=[_el("E1", "cagiona a un apicoltore una lesione da puntura")], logic={"ref": "E1"},
               commi_speciali=[{"comma": "2", "kind": "querela", "schema": sub}])
    line = E.make_line({"number": STING_NO, "code": "c.p.", "text": STING}, "m", s, doc=PENAL_DOC)
    _write(tmp_path / "schemas.jsonl", [line])
    got, why = E.for_row(_row(STING_NO, STING))
    assert why == "schema"
    summ = E.summary(got)
    assert [(d["id"], d["action"]) for d in summ["dropped"] if d["check"] == "procedibilita"] == [
        ("co2.procedibilita", "merged"), ("co2.E1", "moved"), ("co2.comma speciale", "merged")]
    counters = interactive_flow.element_counters({"records": {"x": {"schema": summ, "requisites": []}}})
    assert counters["schema_procedibilita_moved"] == 1
    r = show.recheck(write=False)
    assert r["procedibilita_moved"] == 1 and "procedibilita" not in r["fixed"]
