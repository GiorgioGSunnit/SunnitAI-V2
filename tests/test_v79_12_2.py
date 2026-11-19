"""v79.12.2 -- element schema checks per element, on synthetic article texts only (no model, no Neo4j, no network):

1. grounding on Italian stems (-e/-i/-o/-a, participles against the verb stem), function words ignored: an element
   whose content words all match by stem is explicit
2. ungrounded, fragments and branches_without_alternative fix or drop the offending element and keep the schema (the
   drop logged); only invalid and a logic tree with dangling refs make a schema unusable
3. a logic node with no args removed, recursively; an empty tree is no logic, not an error
4. the runtime reads the repaired schema (requisites, summary, the fragments log), a reviewed schema never repaired
5. scripts/schema_show.py --recheck on a synthetic file (usable before/after, drops by reason, the schemas and the
   reviewed overlay untouched, --dry-run) and --unusable N
"""

import importlib.util
import json
import os
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


@pytest.fixture
def show():
    spec = importlib.util.spec_from_file_location("schema_show", ROOT / "scripts" / "schema_show.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# synthetic article texts (invented wording, no real article)
TEXT = ("1. Il depositario che, per incuria o imperizia, lascia deteriorare la merce affidatagli è tenuto a "
        "indennizzare il danno al depositante, se questi ne fa richiesta entro sei mesi dalla riconsegna.\n"
        "2. Non è tenuto all'indennizzo il depositario che prova la forza maggiore.\n"
        "3. Il danno si presume se la merce è riconsegnata dopo il termine pattuito.")
OFFENCE = ("Chiunque danneggia le colture altrui, cagionando una perdita al raccolto, è punito con la multa fino a "
           "mille euro.")
DOC = "Codice Sintetico 2026"


def _el(eid, testo, kind="condotta", comma="1", grounding="explicit", fonte=None, **extra):
    out = {"id": eid, "testo": testo, "kind": kind, "kind_note": None, "comma": comma, "branches": [],
           "connective": None, "parti": [], "grounding": grounding,
           "fonte": {"comma": comma, "testo": testo if fonte is None else fonte}, "spiegazione": None}
    out.update(extra)
    return out


def _schema(**over):
    schema = {"tipo": ["claim-conferring"], "destinatario": [{"ruolo": "altro", "condizione": "il depositario"}],
              "legittimato": [{"ruolo": "attore", "condizione": None}], "soggetto_passivo": [],
              "rapporto": "deposito",
              "elementi": [_el("E1", "lascia deteriorare la merce affidatagli"),
                           _el("E2", "per incuria o imperizia", kind="elemento soggettivo",
                               branches=["incuria", "imperizia"], connective="o"),
                           _el("E3", "ne fa richiesta entro sei mesi dalla riconsegna", kind="termine")],
              "logic": {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}, {"ref": "E3"}]},
              "eccezioni": [{"testo": "prova la forza maggiore", "comma": "2", "grounding": "explicit",
                             "fonte": {"comma": "2", "testo": "prova la forza maggiore"}, "spiegazione": None}],
              "conseguenza": {"testo": "è tenuto a indennizzare il danno", "tipo": "risarcimento", "comma": "1"},
              "commi_speciali": [{"comma": "3", "kind": "presunzione", "schema": {
                  "tipo": ["claim-conferring"], "elementi": [
                      _el("E1", "la merce è riconsegnata dopo il termine pattuito", kind="condizione", comma="3")],
                  "logic": {"ref": "E1"}, "eccezioni": [], "conseguenza": None}}],
              "collegati": []}
    schema.update(over)
    return schema


def _ids(schema):
    return [e["id"] for e in schema["elementi"]]


def _acts(actions):
    return [(a["scope"], a["id"], a["check"], a["action"]) for a in actions]


# --- 1. stems and function words --------------------------------------------------------------------------------------

@pytest.mark.parametrize("words,same", [
    (("cagionata", "cagiona", "cagionare", "cagionate"), True),
    (("lesioni", "lesione"), True),
    (("deteriorato", "deteriorare", "deteriora"), True),
    (("punito", "punire", "punita"), True),
    (("affidatagli", "affidata"), False),            # an enclitic is no inflection
])
def test_stems(words, same):
    from src.rag import element_schemas as E
    assert (len({E.stem(w) for w in words}) == 1) is same


def test_stem_keeps_three_letters_and_an_accented_end():
    from src.rag import element_schemas as E
    assert E.stem("reato") == E.stem("reati") == "reat"           # «-ato» would leave two letters: the vowel only
    assert E.stem("sei") == "sei" and E.stem("citta") == "citt"
    assert E.stem("Città") == E.stem("città") == E.stem("citta", accented=True) == "citta"   # no inflection there
    assert E.stem("autorità") != E.stem("autore") and E.stem("qualità") != E.stem("quale")


def test_a_short_verbs_participle_meets_the_verb():
    """Review finding: the 3-letter guard kept «usata» from «usa» -- the participle meets its verb by verb_base, down
    to 2 letters, and a 3-letter word's base only against a verb form («uso» is never «usa» by that road alone)."""
    from src.rag import element_schemas as E
    assert {E.verb_base(w) for w in ("usata", "usato", "usare")} == {"us"} and E.verb_base("usa") == ""
    text = ("1. Chiunque usa violenza o minaccia contro la persona offesa e danneggia la cosa è punito con la "
            "reclusione.")
    for passage in ("violenza usata", "la violenza è usata", "persona minacciata", "cosa danneggiata"):
        assert E.missing_words(passage, text) == [], passage
    assert E.missing_words("usa violenza", "1. Chiunque, usata violenza, sottrae la cosa è punito.") == []
    assert E.missing_words("ora", "1. Chiunque sottrae oro è punito.") == ["ora"]          # no 2-letter stems
    s = {"tipo": ["sanction"], "elementi": [_el("E1", "violenza usata", fonte="usa violenza"),
                                            _el("E2", "danneggia la cosa")],
         "logic": {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}]}, "eccezioni": [], "commi_speciali": []}
    fixed, actions = E.repair(s, text)
    assert actions == [] and _ids(fixed) == ["E1", "E2"] and E.check(s, text)["ungrounded"] == []


def test_polarity_words_are_never_ignored():
    """Review finding: «non» and «senza» (and più/meno, dopo/entro, sopra/sotto, salvo) carry the element's polarity --
    an element saying the opposite of the text stays ungrounded and is dropped, never an explicit requisite."""
    from src.rag import element_schemas as E
    text = ("1. Chiunque, con il consenso del proprietario, asporta il bene altrui è punito con la multa entro sei "
            "mesi, se il valore è più di mille euro.")
    assert E.missing_words("senza il consenso del proprietario", text) == ["senza"]
    assert E.missing_words("non asporta il bene altrui", text) == ["non"]
    assert E.missing_words("dopo sei mesi", text) == ["dopo"]
    assert E.missing_words("il valore è meno di mille euro", text) == ["meno"]
    assert E.missing_words("con il consenso del proprietario", text) == []
    assert not {"non", "senza", "salvo", "piu", "meno", "dopo", "entro", "sopra", "sotto"} & E.FUNCTION_WORDS
    s = {"tipo": ["sanction"], "elementi": [_el("E1", "asporta il bene altrui"),
                                            _el("E2", "senza il consenso del proprietario", kind="negazione")],
         "logic": {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}]}, "eccezioni": [], "commi_speciali": []}
    fixed, actions = E.repair(s, text)
    assert _ids(fixed) == ["E1"] and ("", "E2", "ungrounded", "dropped") in _acts(actions)


def test_function_words_in_the_text_never_ground_by_stem():
    """Review finding: «dove» in the text is no «dovere», «quale» no «qualità»; an accented end («autorità», or
    «autorita'» as old texts spell it) is never stripped into «autore»."""
    from src.rag import element_schemas as E
    text = "1. Il depositario, dove sia il deposito e quale che sia l'autorita' competente, risponde del danno."
    assert E.missing_words("il dovere di deposito", text) == ["dovere"]
    assert E.missing_words("la qualità del deposito", text) == ["qualita"]
    assert E.missing_words("l'autore del danno", text) == ["autore"]
    assert E.missing_words("l'autorità competente", text) == []
    assert E.missing_words("i depositi", text) == []                                   # inflection still matches


def test_an_element_whose_content_words_all_match_by_stem_is_explicit():
    from src.rag import element_schemas as E
    s = _schema()
    s["elementi"][0]["testo"] = "la merce è stata lasciata deteriorata per incuria"   # inflected, an auxiliary
    s["elementi"][0]["fonte"]["testo"] = "lascia deteriorare la merce affidatagli"
    checks = E.check(s, TEXT)
    assert checks["ungrounded"] == [] and E.usable({"schema": s, "checks": checks})
    assert E.missing_words("le merci deteriorate", E.content_words(TEXT)) == []
    assert E.missing_words("che sono state riconsegnate", E.content_words(TEXT)) == []     # function words ignored
    assert E.missing_words("la merce distrutta", E.content_words(TEXT)) == ["distrutta"]


# --- 2. per element: fix or drop, keep the schema ---------------------------------------------------------------------

def test_an_ungrounded_element_is_dropped_and_the_schema_kept():
    from src.rag import element_schemas as E
    s = _schema()
    s["elementi"][2] = _el("E3", "ne fa richiesta con raccomandata entro sei mesi", kind="termine")
    checks = E.check(s, TEXT)
    assert checks["ungrounded"] == ["E3: parole assenti dal testo (con, raccomandata)"]
    assert E.raised(checks) == [] and E.usable({"schema": s, "checks": checks})
    fixed, actions = E.repair(s, TEXT)
    assert _ids(fixed) == ["E1", "E2"] and _ids(s) == ["E1", "E2", "E3"]          # a copy: the input as it was
    assert fixed["logic"] == {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}]}
    assert _acts(actions) == [("", "E3", "ungrounded", "dropped"), ("", "logic", "logic_incomplete", "fixed")]
    assert actions[0]["why"] == "parole assenti dal testo (con, raccomandata)" and actions[0]["comma"] == "1"
    again = E.check(fixed, TEXT)
    assert not any(again[k] for k in E.ELEMENT_CHECKS) and E.repair(fixed, TEXT)[1] == []   # idempotent


def test_a_fragment_is_dropped_and_the_schema_kept():
    from src.rag import element_schemas as E
    s = _schema()
    s["elementi"].append(_el("E4", "il debitore", kind="qualità del soggetto", grounding="inferred", fonte=None,
                             spiegazione="x"))
    s["logic"]["args"].append({"ref": "E4"})
    assert E.check(s, TEXT)["fragments"] == ["E4: soggetto senza condizione"]
    fixed, actions = E.repair(s, TEXT)
    assert _ids(fixed) == ["E1", "E2", "E3"] and ("", "E4", "fragments", "dropped") in _acts(actions)
    assert actions[0]["why"] == "soggetto senza condizione"


def test_branches_without_an_alternative_are_cleared_the_element_kept():
    from src.rag import element_schemas as E
    s = _schema()
    s["elementi"][0].update(branches=["lascia deteriorare", "la merce affidatagli"], connective="o")
    assert E.check(s, TEXT)["branches_without_alternative"] == ["E1: alternative senza «o» nel testo"]
    fixed, actions = E.repair(s, TEXT)
    assert fixed["elementi"][0]["branches"] == [] and fixed["elementi"][0]["connective"] is None
    assert _ids(fixed) == ["E1", "E2", "E3"] and fixed["elementi"][1]["branches"] == ["incuria", "imperizia"]
    assert _acts(actions) == [("", "E1", "branches_without_alternative", "fixed")]


def test_an_any_of_over_no_alternative_is_read_as_all_of():
    from src.rag import element_schemas as E
    s = {"tipo": ["sanction"], "elementi": [_el("E1", "danneggia le colture altrui"),
                                            _el("E2", "cagionando una perdita al raccolto", kind="evento")],
         "logic": {"op": "any_of", "args": [{"ref": "E1"}, {"ref": "E2"}]}}
    assert E.check(s, OFFENCE)["branches_without_alternative"] == ["any_of(E1, E2): nessuna alternativa nel testo"]
    fixed, actions = E.repair(s, OFFENCE)
    assert fixed["logic"] == {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}]}
    assert _acts(actions) == [("", "any_of(E1, E2)", "branches_without_alternative", "fixed")]
    assert E.check(fixed, OFFENCE)["branches_without_alternative"] == []


def test_an_unlocated_fonte_takes_the_elements_own_words():
    from src.rag import element_schemas as E
    s = _schema()
    s["elementi"][0]["fonte"]["testo"] = "lascia rovinare la merce"
    assert E.check(s, TEXT)["fonte_unlocated"] == ["E1: fonte.testo non trovato nel testo"]
    assert E.usable({"schema": s, "checks": E.check(s, TEXT)})
    fixed, actions = E.repair(s, TEXT)
    fonte = fixed["elementi"][0]["fonte"]
    assert fonte["testo"] == "lascia deteriorare la merce affidatagli" and TEXT[fonte["start"]:fonte["end"]] == \
        "lascia deteriorare la merce affidatagli"
    assert _acts(actions) == [("", "E1", "fonte_unlocated", "fixed")]


def test_an_ungrounded_eccezione_and_a_special_commas_element_are_dropped():
    from src.rag import element_schemas as E
    s = _schema()
    s["eccezioni"].append({"testo": "prova il dolo del depositante", "comma": "2", "grounding": "explicit",
                           "fonte": {"comma": "2", "testo": "prova il dolo"}, "spiegazione": None})
    s["commi_speciali"][0]["schema"]["elementi"][0]["testo"] = "la merce è consegnata in ritardo grave"
    fixed, actions = E.repair(s, TEXT)
    assert len(fixed["eccezioni"]) == 1 and fixed["commi_speciali"][0]["schema"]["elementi"] == []
    assert fixed["commi_speciali"][0]["schema"]["logic"] is None                  # the empty tree: no logic
    assert ("", "eccezione 2", "ungrounded", "dropped") in _acts(actions)
    assert ("3", "E1", "ungrounded", "dropped") in _acts(actions)
    assert not E.raised(E.check(fixed, TEXT))


@pytest.mark.parametrize("key,usable", [("invalid", False), ("logic_errors", False), ("failed", False), ("cut", False),
                                        ("ungrounded", True), ("fragments", True),
                                        ("branches_without_alternative", True), ("logic_incomplete", True),
                                        ("fonte_unlocated", True), ("comma_unknown", True)])
def test_only_invalid_and_dangling_refs_make_a_schema_unusable(key, usable):
    from src.rag import element_schemas as E
    checks = {**E.empty_checks(), key: ["x"] if key != "cut" else True}
    assert E.usable({"schema": _schema(), "schema_version": 1, "checks": checks}) is usable


def test_a_dangling_ref_blocks_a_dropped_elements_ref_does_not():
    from src.rag import element_schemas as E
    s = _schema(logic={"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}, {"ref": "E3"}, {"ref": "E8"}]})
    assert E.check(s, TEXT)["logic_errors"] == ["logic.all_of[3]: ref E8 senza elemento"]
    s = _schema()
    s["elementi"][2]["testo"] = "ne fa richiesta per iscritto"
    fixed, _ = E.repair(s, TEXT)
    assert E.check(fixed, TEXT)["logic_errors"] == [] and E.check(s, TEXT)["logic_errors"] == []


# --- 3. logic nodes with no args --------------------------------------------------------------------------------------

def test_nodes_with_no_args_are_removed_recursively():
    from src.rag import element_schemas as E
    tree = {"op": "all_of", "args": [{"ref": "E1"}, {"op": "any_of", "args": [{"op": "all_of", "args": []},
                                                                               {"op": "not"}]},
                                     {"op": "not", "arg": {"op": "all_of", "args": []}}, {"ref": "E2"}]}
    assert E.prune_logic(tree) == {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}]}
    assert E.prune_logic({"op": "all_of", "args": [{"op": "any_of", "args": []}]}) is None
    assert E.prune_logic({"op": "any_of", "args": [{"ref": "E1"}, {"ref": "E2"}]}, {"E2"}) == {"ref": "E1"}
    assert E.prune_logic({"op": "conditional", "if": {"ref": "E1"}, "then": {"ref": "E2"}}, {"E1", "E2"}) is None
    assert E.prune_logic({"op": "conditional", "if": {"ref": "E1"}, "then": {"ref": "E2"},
                          "else": {"op": "all_of"}}) == {"op": "conditional", "if": {"ref": "E1"}, "then": {"ref": "E2"}}


def test_an_empty_tree_is_no_logic_not_an_error():
    from src.rag import element_schemas as E
    s = _schema(logic={"op": "all_of", "args": [{"op": "any_of", "args": []}, {"op": "not"}]})
    checks = E.check(s, TEXT)
    assert checks["logic_errors"] == [] and E.usable({"schema": s, "checks": checks})
    assert any("any_of senza args" in m for m in checks["logic_incomplete"])
    fixed, actions = E.repair(s, TEXT)
    assert fixed["logic"] is None and ("", "logic", "logic_incomplete", "fixed") in _acts(actions)
    assert actions[-1]["why"] == "albero vuoto: nessuna logica"


def test_a_conditional_keeps_a_live_else():
    """Review finding: a conditional whose if and then prune away but whose else holds a live ref is no node without
    args -- the else's element still cited. v79.12.5 (item 0): its branches pruned exactly as args, so the node
    collapses into its live else (no conditional left with a branch missing); a then alone, once its if is dropped,
    is the then."""
    from src.rag import element_schemas as E
    tree = {"op": "conditional", "if": {"op": "all_of", "args": []}, "then": {"op": "all_of", "args": []},
            "else": {"ref": "E1"}}
    assert E.prune_logic(tree) == {"ref": "E1"}
    assert E.prune_logic({"op": "conditional", "if": {"ref": "E1"}, "then": {"ref": "E2"}}, {"E1"}) == {"ref": "E2"}
    assert E.prune_logic({"op": "conditional", "if": {"ref": "E1"}, "else": {"ref": "E2"}}, {"E1", "E2"}) is None
    s = {"tipo": ["sanction"], "elementi": [_el("E1", "danneggia le colture altrui")], "logic": tree, "eccezioni": [],
         "commi_speciali": []}
    fixed, actions = E.repair(s, OFFENCE)
    assert fixed["logic"] == {"ref": "E1"}
    assert actions[-1]["why"] == "nodi vuoti o riferimenti a elementi tolti rimossi"
    checks = E.check(fixed, OFFENCE)
    assert checks["logic_incomplete"] == [] and E.usable({"schema": fixed, "checks": checks})


# --- 4. the runtime ---------------------------------------------------------------------------------------------------

def _write(path, lines):
    with open(path, "w", encoding="utf-8") as fh:
        for x in lines:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")


def _gen(schema=None, number="901", text=TEXT, **over):
    from src.rag import element_schemas as E
    line = E.make_line({"number": number, "code": "c.c.", "rubrica": "Deposito sintetico", "text": text}, "m-test",
                       _schema() if schema is None else schema, doc=DOC)
    line.update(over)
    return line


def _row(number="901", text=TEXT):
    return {"_article": (number, "c.c."), "d": {"name": DOC}, "s": {"plain_text": text, "titolo_articolo": "x"}}


def test_the_runtime_reads_the_repaired_schema(monkeypatch, tmp_path):
    from src.rag import element_schemas as E
    monkeypatch.setenv("ELEMENT_SCHEMAS", "true")
    s = _schema()
    s["elementi"][2]["testo"] = "ne fa richiesta con raccomandata entro sei mesi"
    s["elementi"].append(_el("E4", "il debitore", kind="qualità del soggetto", grounding="inferred", fonte=None,
                             spiegazione="x"))
    s["logic"]["args"].append({"ref": "E4"})
    line = _gen(s)
    assert [a["id"] for a in line["dropped"] if a["action"] == "dropped"] == ["E3", "E4"]
    _write(tmp_path / "schemas.jsonl", [line])
    got, why = E.for_row(_row())
    assert why == "schema" and _ids(got["schema"]) == ["E1", "E2"]
    fragments = []
    reqs = E.requisites(_row(), got, {}, fragments)
    assert [r["schema_id"] for r in reqs if not r.get("special")] == ["E1", "E2"]
    assert fragments == [{"comma": "1", "text": "il debitore", "why": "soggetto senza condizione"}]
    summary = E.summary(got)
    assert {"id": "E3", "check": "ungrounded", "action": "dropped"} in summary["dropped"]
    assert summary["logic"] == {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}]}
    assert "dropped" not in E.summary({**got, "dropped": []})


def test_a_line_checked_by_the_older_rule_is_read_by_todays(monkeypatch, tmp_path):
    """A line whose recorded checks say «ungrounded» (v79.12.1: blocking) is used, its element dropped on reading;
    without the row's text (for_label with no text) the line's own text is re-checked by today's rule."""
    from src.rag import element_schemas as E
    monkeypatch.setenv("ELEMENT_SCHEMAS", "true")
    s = _schema()
    s["elementi"][0]["testo"] = "lascia marcire la merce affidatagli"
    old = _gen(s, checks={**E.empty_checks(), "ungrounded": ["E1: parole assenti dal testo (marcire)"],
                          "logic_errors": ["E9: mai citato nella logic"]})
    old.pop("dropped")
    _write(tmp_path / "schemas.jsonl", [old])
    got, why = E.for_row(_row())
    assert why == "schema" and _ids(got["schema"]) == ["E2", "E3"]
    got, why = E.for_label("901 c.c.")                  # no row text: the line's own text re-checked and repaired
    assert why == "schema" and _ids(got["schema"]) == ["E2", "E3"]
    got, why = E.for_label("901 c.c.", TEXT)
    assert why == "schema" and _ids(got["schema"]) == ["E2", "E3"]


def test_without_any_text_the_logic_checks_are_rerun(monkeypatch, tmp_path, show):
    """Review finding: a line with no text keeps the checks that read the text as recorded, but the logic and validity
    checks (which need none) are re-run by today's rules -- an older line's «mai citato» filed under logic_errors no
    longer blocks it, in the runtime, in --recheck and in --unusable alike; a real dangling ref still does."""
    from src.rag import element_schemas as E
    monkeypatch.setenv("ELEMENT_SCHEMAS", "true")
    uncited = _old(_gen(_schema(logic={"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}]}), number="907"),
                   logic_errors=["E3: mai citato nella logic"])
    dangling = _old(_gen(_schema(logic={"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E9"}]}), number="908"),
                    logic_errors=["logic.all_of[1]: ref E9 senza elemento"])
    for line in (uncited, dangling):
        line.pop("text")
    _write(tmp_path / "schemas.jsonl", [uncited, dangling])
    assert E.for_label("907 c.c.")[1] == "schema" and E.for_label("908 c.c.")[1] == "unusable:logic_errors"
    found, total = show.unusable(10)
    assert total == 1 and [x["label"] for x in found] == ["908 c.c."]
    assert found[0]["details"] == ["logic.all_of[1]: ref E9 senza elemento"]
    r = show.recheck(now="2026-11-18T00:00:00+00:00")
    assert (r["usable_before"], r["usable_after"], r["without_text"]) == (0, 1, 2)
    assert r["unusable_after"] == {"logic_errors": 1}
    after = [json.loads(x) for x in (tmp_path / "schemas.jsonl").read_text(encoding="utf-8").splitlines()]
    assert after[0]["checks"]["logic_errors"] == [] and after[0]["checks"]["logic_incomplete"] == [
        "E3: mai citato nella logic"] and "dropped" not in after[0]
    assert [x["schema"] for x in after] == [uncited["schema"], dangling["schema"]]


def test_a_dropped_base_element_never_makes_a_special_comma_the_base(monkeypatch, tmp_path):
    """Review finding: repair dropping the top schema's comma-1 element must not make the aggravating comma the base --
    its element keeps its special kind, never a plain constituent requisite."""
    from src.rag import element_schemas as E
    monkeypatch.setenv("ELEMENT_SCHEMAS", "true")
    text = ("1. Chiunque danneggia le colture altrui è punito con la multa.\n"
            "2. La pena è aumentata se il fatto è commesso di notte.")
    night = "il fatto è commesso di notte"
    s = {"tipo": ["sanction"], "destinatario": [], "legittimato": [], "soggetto_passivo": [], "rapporto": None,
         "elementi": [_el("E1", "danneggia le vigne altrui"), _el("E2", night, kind="condizione", comma="2")],
         "logic": {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}]}, "eccezioni": [],
         "conseguenza": {"testo": "è punito con la multa", "tipo": "pena", "comma": "1"},
         "commi_speciali": [{"comma": "2", "kind": "aggravamento", "schema": {
             "tipo": ["sanction"], "elementi": [_el("E1", night, kind="condizione", comma="2")],
             "logic": {"ref": "E1"}, "eccezioni": [], "conseguenza": None}}], "collegati": []}
    _write(tmp_path / "schemas.jsonl", [_gen(s, number="909", text=text)])
    row = _row("909", text)
    got, why = E.for_row(row)
    assert why == "schema" and _ids(got["schema"]) == ["E2"] and got["base_comma"] == "1"
    assert [(r["text"], r["comma"], r.get("special")) for r in E.requisites(row, got, {})] == \
        [(night, 2, "aggravamento")]
    s["elementi"][0] = _el("E1", "danneggia le colture altrui")                 # grounded: the same reading
    assert [(r["text"], r["comma"], r.get("special")) for r in E.requisites(row, {"schema": s}, {})] == \
        [("danneggia le colture altrui", 1, None), (night, 2, "aggravamento")]


def test_a_reviewed_schema_is_never_repaired(monkeypatch, tmp_path):
    from src.rag import element_schemas as E
    monkeypatch.setenv("ELEMENT_SCHEMAS", "true")
    s = _schema()
    s["elementi"][0]["testo"] = "lascia marcire la merce affidatagli"            # the lawyer's word
    _write(tmp_path / "schemas.jsonl", [_gen()])
    _write(tmp_path / "reviewed.jsonl", [{"label": "901 c.c.", "reviewed": True, "reviewer": "avv",
                                          "source_hash": E.text_hash(TEXT), "schema_version": 1, "schema": s}])
    got, why = E.for_row(_row())
    assert why == "reviewed" and _ids(got["schema"]) == ["E1", "E2", "E3"] and "dropped" not in got


# --- 5. schema_show --recheck and --unusable --------------------------------------------------------------------------

def _old(line, **checks):
    """A line as v79.12.1 wrote it: its checks by that rule, no "dropped"."""
    from src.rag import element_schemas as E
    line = dict(line)
    line.pop("dropped", None)
    line["checks"] = {**{k: v for k, v in E.empty_checks().items()
                         if k not in ("logic_incomplete", "fonte_unlocated")}, **checks}
    return line


def _synthetic_file(tmp_path):
    from src.rag import element_schemas as E
    ung = _schema()
    ung["elementi"][2]["testo"] = "ne fa richiesta con raccomandata entro sei mesi"
    uncited = _schema(logic={"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}]})
    dangling = _schema(logic={"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}, {"ref": "E3"}, {"ref": "E9"}]})
    notext = _old(_gen(number="906"), ungrounded=["E1: x"])
    notext.pop("text")
    lines = [_old(_gen(number="901")),                                                        # clean: usable both
             _old(_gen(ung, number="902"), ungrounded=["E3: parole assenti dal testo (con, raccomandata)"]),
             _old(_gen(uncited, number="903"), logic_errors=["E3: mai citato nella logic"]),
             _old(_gen(dangling, number="904"), logic_errors=["logic.all_of[3]: ref E9 senza elemento"]),
             E.make_line({"number": "905", "code": "c.c.", "text": TEXT}, "m-test", None, failed="HTTP 500", doc=DOC),
             notext]
    _write(tmp_path / "schemas.jsonl", lines)
    reviewed = _schema()
    reviewed["elementi"][0]["testo"] = "lascia marcire la merce affidatagli"
    _write(tmp_path / "reviewed.jsonl", [{"label": "901 c.c.", "reviewed": True, "reviewer": "avv", "text": TEXT,
                                          "source_hash": E.text_hash(TEXT), "schema_version": 1, "schema": reviewed,
                                          "checks": E.check(reviewed, TEXT)}])
    return lines


def test_recheck_on_a_synthetic_file(tmp_path, show):
    from src.rag import element_schemas as E
    lines = _synthetic_file(tmp_path)
    reviewed_before = (tmp_path / "reviewed.jsonl").read_bytes()
    r = show.recheck(now="2026-11-18T00:00:00+00:00")
    assert (r["lines"], r["articles"], r["schemas"]) == (6, 6, 5)
    assert (r["usable_before"], r["usable_after"]) == (1, 4)          # 902 and 903 now usable, 906 by today's rule
    assert r["dropped"] == {"ungrounded": 1} and r["fixed"] == {"logic_incomplete": 1}
    assert r["unusable_after"] == {"logic_errors": 1, "schema": 1} and r["without_text"] == 1
    assert (r["reviewed"], r["reviewed_sound_before"], r["reviewed_sound_after"]) == (1, 1, 1) and r["written"]
    after = [json.loads(x) for x in (tmp_path / "schemas.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [x["schema"] for x in after] == [x["schema"] for x in lines]                     # no schema touched
    assert after[1]["checks"]["ungrounded"] and [a["id"] for a in after[1]["dropped"]] == ["E3", "logic"]
    assert after[2]["checks"]["logic_errors"] == [] and after[2]["checks"]["logic_incomplete"]
    assert after[2]["rechecked_at"] == "2026-11-18T00:00:00+00:00"
    assert after[4] == lines[4]                                                    # the failed line as it was
    assert {k: v for k, v in after[5].items() if k not in ("rechecked_at", "checks", "elements_after")} == \
        {k: v for k, v in lines[5].items() if k != "checks"}                     # no text: marked, nothing else
    assert after[5]["checks"]["ungrounded"] == ["E1: x"]                           # the text checks as recorded
    assert {k: after[5]["checks"][k] for k in E.TEXT_FREE} == {k: [] for k in E.TEXT_FREE}   # logic re-run
    assert (tmp_path / "reviewed.jsonl").read_bytes() == reviewed_before          # reviewed: reported only
    again = show.recheck()
    assert again["usable_before"] == again["usable_after"] == 4                    # a second run: nothing changes


def test_recheck_dry_run_writes_nothing_and_main_prints_the_counts(tmp_path, show, capsys):
    _synthetic_file(tmp_path)
    before = (tmp_path / "schemas.jsonl").read_bytes()
    assert show.main(["--recheck", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Schemi usabili: prima 1, dopo 4 (+3)." in out and "Elementi tolti per controllo: ungrounded 1." in out
    assert "Nessun file scritto" in out and (tmp_path / "schemas.jsonl").read_bytes() == before
    assert show.main(["--recheck"]) == 0 and "File riscritto." in capsys.readouterr().out
    assert (tmp_path / "schemas.jsonl").read_bytes() != before


def test_unusable_lists_the_rejected_with_their_reasons(tmp_path, show, capsys):
    _synthetic_file(tmp_path)
    found, total = show.unusable(10)
    assert total == 2 and [(x["label"], x["reasons"]) for x in found] == [("904 c.c.", ["logic_errors"]),
                                                                          ("905 c.c.", ["schema"])]
    assert found[0]["details"] == ["logic.all_of[3]: ref E9 senza elemento"] and found[1]["details"] == ["HTTP 500"]
    assert len(show.unusable(1)[0]) == 1
    assert show.main(["--unusable", "1"]) == 0
    out = capsys.readouterr().out
    assert "2 schemi inutilizzabili; i primi 1:" in out and "904 c.c." in out and "905 c.c." not in out


def test_the_review_view_shows_what_is_dropped(tmp_path, show):
    s = _schema()
    s["elementi"][2]["testo"] = "ne fa richiesta con raccomandata entro sei mesi"
    _write(tmp_path / "schemas.jsonl", [_gen(s)])
    text = show.render(show.view("901 c.c."))
    assert "ungrounded (elemento tolto o corretto)" in text
    assert "ELEMENTI TOLTI O CORRETTI" in text and \
        "E3 tolto (ungrounded): parole assenti dal testo (con, raccomandata)" in text
    assert os.environ["ARTICLE_SCHEMAS_FILE"].endswith("schemas.jsonl")
