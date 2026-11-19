"""v79.11 block A -- element schemas: the module src/rag/element_schemas.py, on synthetic article texts only:

A1. check(): every check -- invalid, ungrounded (explicit only: words and fonte.testo), fragments, branches without an
    alternative, logic errors (nested any_of inside all_of / not / conditional, unreferenced ids), comma_unknown,
    kind_other_unexplained, inferred_unexplained; fonte offsets computed by the code (locate)
A2. the files: missing = empty, last line wins, re-read on change; for_row / for_label with every path reason (off,
    no_schema, hash_mismatch, unusable:<check>, schema, reviewed, reviewed_stale) and the lawyer's notes merged
A3. requisites(): issue_records._requisites' shape, "split": "schema", branches, special commi, the schema_* keys,
    fragments dropped, kinds that are no requisites left out (kept in summary), the cap
A4. roles_match (conditional roles), shadow_admission, gate / decide / hard_excluded
A5. quality with and without reviewed lines; the prompt, user_message, parse_reply, make_line
"""

import json

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


# synthetic texts: an offence in one comma, a civil article in three numbered commi
PENAL = ("Chiunque cagiona ad alcuno una lesione personale, dalla quale deriva una malattia nel corpo o nella mente, "
         "è punito con la reclusione da sei mesi a tre anni.")
CIVIL = ("1. Il custode che, per negligenza o imprudenza, lascia deteriorare il bene affidatogli è tenuto a risarcire il "
         "danno al proprietario, se questi ne chiede il risarcimento entro un anno dalla restituzione.\n"
         "2. Non è tenuto al risarcimento il custode che prova il caso fortuito.\n"
         "3. Il danno si presume se il bene è restituito oltre il termine convenuto.")
DOC = "Codice Penale 2026"


def _el(eid, testo, kind="condotta", comma="1", grounding="explicit", fonte=None, **extra):
    out = {"id": eid, "testo": testo, "kind": kind, "kind_note": None, "comma": comma, "branches": [],
           "connective": None, "parti": [], "grounding": grounding,
           "fonte": {"comma": comma, "testo": testo if fonte is None else fonte}, "spiegazione": None}
    out.update(extra)
    return out


def _penal(**over):
    schema = {"tipo": ["sanction"], "destinatario": [{"ruolo": "chiunque", "condizione": None}],
              "legittimato": [{"ruolo": "persona offesa", "condizione": None}],
              "soggetto_passivo": [{"ruolo": "persona offesa", "condizione": None}], "rapporto": "lesioni personali",
              "elementi": [_el("E1", "cagiona ad alcuno una lesione personale"),
                           _el("E2", "dalla quale deriva una malattia nel corpo o nella mente", kind="evento",
                               branches=["malattia nel corpo", "malattia nella mente"], connective="o"),
                           _el("E3", "coscienza e volontà di cagionare la lesione", kind="elemento soggettivo",
                               grounding="inferred", fonte=None, spiegazione="delitto punito solo a titolo di dolo")],
              "logic": {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}, {"ref": "E3"}]},
              "eccezioni": [], "conseguenza": {"testo": "è punito con la reclusione da sei mesi a tre anni",
                                               "tipo": "pena", "comma": "1"},
              "commi_speciali": [], "collegati": []}
    schema["elementi"][2]["fonte"] = {"comma": "1", "testo": None}
    schema.update(over)
    return schema


def _civil(**over):
    schema = {"tipo": ["claim-conferring"], "destinatario": [{"ruolo": "altro", "condizione": "il custode"}],
              "legittimato": [{"ruolo": "attore", "condizione": "se proprietario del bene"}],
              "soggetto_passivo": [], "rapporto": "custodia",
              "elementi": [_el("E1", "lascia deteriorare il bene affidatogli"),
                           _el("E2", "per negligenza o imprudenza", kind="elemento soggettivo",
                               branches=["negligenza", "imprudenza"], connective="o"),
                           _el("E3", "ne chiede il risarcimento entro un anno dalla restituzione", kind="termine"),
                           _el("E4", "risarcire il danno al proprietario", kind="quantificazione")],
              "logic": {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}, {"ref": "E3"}, {"ref": "E4"}]},
              "eccezioni": [{"testo": "prova il caso fortuito", "comma": "2", "grounding": "explicit",
                             "fonte": {"comma": "2", "testo": "prova il caso fortuito"}, "spiegazione": None}],
              "conseguenza": {"testo": "è tenuto a risarcire il danno", "tipo": "risarcimento", "comma": "1"},
              "commi_speciali": [{"comma": "3", "kind": "presunzione", "schema": {
                  "tipo": ["claim-conferring"], "destinatario": [], "legittimato": [], "soggetto_passivo": [],
                  "rapporto": "custodia",
                  "elementi": [_el("E1", "il bene è restituito oltre il termine convenuto", kind="condizione",
                                   comma="3")],
                  "logic": {"ref": "E1"}, "eccezioni": [], "conseguenza": None, "collegati": []}}],
              "collegati": ["1218 c.c."]}
    schema.update(over)
    return schema


def _blocking(checks):
    from src.rag import element_schemas as E
    return {k: checks[k] for k in E.BLOCKING if checks.get(k)}


# --- A1 checks --------------------------------------------------------------------------------------------------------

def test_a_clean_schemas_raise_nothing():
    from src.rag import element_schemas as E
    assert E.check(_penal(), PENAL) == E.empty_checks()
    assert E.check(_civil(), CIVIL) == E.empty_checks()


@pytest.mark.parametrize("over,needle", [
    ({"tipo": {"reato": True}}, "tipo mancante"),    # v79.12.5 (item 0): an empty tipo is «other» with a note
    # v79.12.5 (item 0, review): a word no vocabulary maps is counted, not invalid -- a non-string value still is
    ({"tipo": [5]}, "tipo sconosciuto"),                  # v79.12.3: «incriminatrice» maps to «reato»
    ({"legittimato": [{"ruolo": 7, "condizione": None}]}, "ruolo sconosciuto"),
    ({"collegati": "583 c.p."}, "collegati"),
    ({"commi_speciali": [{"comma": "1", "kind": 3, "schema": {"tipo": ["sanction"], "elementi": [],
                                                                    "logic": None}}]}, "kind speciale"),
])
def test_a_invalid_structure(over, needle):
    from src.rag import element_schemas as E
    checks = E.check(_penal(**over), PENAL)
    assert any(needle in m for m in checks["invalid"]), checks


def test_a_invalid_elements_kind_ids_grounding():
    from src.rag import element_schemas as E
    s = _penal()
    s["elementi"][0]["kind"] = None                    # v79.12.5 (item 0, review): «azione» is counted, not invalid
    s["elementi"][1]["id"] = "E1"                      # duplicate
    s["elementi"][2]["grounding"] = "supposto"
    s["elementi"].append(_el("", "cagiona ad alcuno una lesione personale"))
    s["logic"] = {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E3"}]}
    inv = E.check(s, PENAL)["invalid"]
    assert any("E1: kind sconosciuto" in m for m in inv)
    assert any("E1: id duplicato" in m for m in inv)
    assert any("E3: grounding" in m for m in inv)
    assert any("id mancante" in m for m in inv)
    assert not E.check({"tipo": ["sanction"]}, PENAL)["invalid"]
    assert E.check("not a dict", PENAL)["invalid"] == ["schema non è un oggetto"]


def test_a_kind_and_role_spellings_are_read_as_the_vocabulary():
    from src.rag import element_schemas as E
    s = _penal(legittimato=[{"ruolo": "persona_offesa", "condizione": None}])
    s["elementi"][0]["kind"] = "Nesso causale"
    assert E.check(s, PENAL)["invalid"] == []
    assert E.canon_kind("nesso causale") == "nesso_causale" and E.canon_role("pa") == "PA"


def test_a_explicit_grounding_words_and_passage():
    from src.rag import element_schemas as E
    s = _penal()
    s["elementi"][0] = _el("E1", "cagiona ad alcuno una lesione grave", fonte="cagiona ad alcuno una lesione personale")
    s["elementi"][1]["fonte"]["testo"] = "dalla quale deriva una malattia nel corpo e nella mente"
    checks = E.check(s, PENAL)
    assert checks["ungrounded"] == ["E1: parole assenti dal testo (grave)"]
    assert checks["fonte_unlocated"] == ["E2: fonte.testo non trovato nel testo"]    # v79.12.2: reported, E2's
    s = _penal()                                                                   # words are all in the text
    s["elementi"][0]["fonte"] = {"comma": "1", "testo": None}
    assert E.check(s, PENAL)["ungrounded"] == [] and E.check(s, PENAL)["fonte_unlocated"] == ["E1: fonte.testo mancante"]


def test_a_passage_found_whatever_its_case_spacing_and_end_punctuation():
    from src.rag import element_schemas as E
    s = _penal()
    s["elementi"][0]["fonte"]["testo"] = "  CAGIONA ad alcuno\n una lesione personale,"
    assert E.check(s, PENAL)["ungrounded"] == []


def test_a_inferred_never_rejected_for_wording_but_needs_a_spiegazione():
    from src.rag import element_schemas as E
    s = _penal()
    s["elementi"][2]["testo"] = "volontà consapevole di offendere l'integrità fisica altrui"   # no word of the text
    assert E.check(s, PENAL)["ungrounded"] == []
    s["elementi"][2]["spiegazione"] = None
    s["elementi"][2]["fonte"] = {"comma": None, "testo": None}
    checks = E.check(s, PENAL)
    assert checks["inferred_unexplained"] == ["E3: spiegazione mancante", "E3: fonte.comma mancante"]
    assert _blocking(checks) == {}                       # reported, not blocking
    assert E.usable({"schema": s, "checks": checks})


def test_a_offsets_are_computed_by_the_code_never_the_models():
    from src.rag import element_schemas as E
    assert E.locate(PENAL, "cagiona ad alcuno una lesione personale") == (9, 48)
    assert PENAL[9:48] == "cagiona ad alcuno una lesione personale"
    start, end = E.locate(CIVIL, "PROVA IL   caso fortuito")
    assert CIVIL[start:end] == "prova il caso fortuito"
    assert E.locate(PENAL, "lesione grave") == (None, None)
    assert E.locate(PENAL, "") == (None, None)
    s = _penal()
    s["elementi"][0]["fonte"].update({"start": 0, "end": 3})
    s["elementi"][1]["fonte"]["testo"] = "non nel testo"
    g = E.ground(s, PENAL)
    assert (g["elementi"][0]["fonte"]["start"], g["elementi"][0]["fonte"]["end"]) == (9, 48)
    assert (g["elementi"][1]["fonte"]["start"], g["elementi"][1]["fonte"]["end"]) == (None, None)
    assert s["elementi"][0]["fonte"]["start"] == 0         # the input left as it was


@pytest.mark.parametrize("testo,why", [("il debitore", "soggetto senza condizione"),
                                       ("salvo che il fatto costituisca più grave reato", "clausola di salvezza"),
                                       ("il danno", "oggetto senza condizione")])
def test_a_fragments(testo, why):
    from src.rag import element_schemas as E
    s = _penal()
    s["elementi"].append(_el("E4", testo, grounding="inferred", fonte=None, spiegazione="x"))
    s["logic"]["args"].append({"ref": "E4"})
    assert E.check(s, PENAL)["fragments"] == [f"E4: {why}"]


def test_a_branches_without_alternative():
    from src.rag import element_schemas as E
    s = _penal()
    s["elementi"][0]["branches"] = ["cagiona", "lesione personale"]
    assert E.check(s, PENAL)["branches_without_alternative"] == ["E1: alternative senza «o» nel testo"]
    s = _penal(logic={"op": "all_of", "args": [{"op": "any_of", "args": [{"ref": "E1"}, {"ref": "E3"}]},
                                               {"ref": "E2"}]})
    assert E.check(s, PENAL)["branches_without_alternative"] == []    # E3 inferred: comma 1 holds «o»
    one = "Chiunque cagiona ad alcuno una lesione personale è punito con la reclusione."
    s = {"tipo": ["sanction"], "elementi": [_el("E1", "cagiona ad alcuno"), _el("E2", "una lesione personale",
                                                                                    kind="evento")],
         "logic": {"op": "any_of", "args": [{"ref": "E1"}, {"ref": "E2"}]}}
    assert E.check(s, one)["branches_without_alternative"] == ["any_of(E1, E2): nessuna alternativa nel testo"]


@pytest.mark.parametrize("passage", ["la cosa o il denaro", "sia il venditore sia il compratore",
                                     "anche solo in parte", "il fatto ovvero l'omissione", "a) il bene; b) od il prezzo"])
def test_a_alternative_markers(passage):
    from src.rag import element_schemas as E
    assert E.has_alternative(passage)


@pytest.mark.parametrize("passage", ["la cosa e il denaro", "nei casi indicati alle lettere n) e o) del comma",
                                     "l'oggetto venduto"])
def test_a_no_alternative_marker(passage):
    from src.rag import element_schemas as E
    assert not E.has_alternative(passage)


def test_a_logic_nested_trees_are_valid():
    from src.rag import element_schemas as E
    s = _civil(logic={"op": "conditional",
                      "if": {"op": "all_of", "args": [{"ref": "E1"}, {"op": "any_of", "args": [
                          {"ref": "E2"}, {"op": "not", "arg": {"ref": "E3"}}]}]},
                      "then": {"ref": "E4"}})
    assert E.check(s, CIVIL)["logic_errors"] == []


@pytest.mark.parametrize("logic,key,needle", [
    ({"op": 7, "args": [{"ref": "E1"}]}, "invalid", "op sconosciuto"),     # v79.12.5 (review): «xor» lifted, counted
    ({"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}, {"ref": "E3"}, {"ref": "E9"}]}, "logic_errors",
     "ref E9 senza elemento"),
    ({"op": "all_of", "args": [{"op": "any_of", "args": [{"ref": "E1"}]}, {"ref": "E2"}, {"ref": "E3"}]},
     "logic_incomplete", "any_of con meno di due args"),
    ({"op": "all_of", "args": [{"op": "not"}, {"ref": "E1"}, {"ref": "E2"}, {"ref": "E3"}]}, "logic_incomplete",
     "not senza arg"),
    ({"op": "conditional", "if": {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}]}, "then": None,
      "else": {"ref": "E3"}}, "logic_incomplete", "conditional senza if/then"),
    ({"op": "all_of", "args": []}, "logic_incomplete", "all_of senza args"),
    ({"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}]}, "logic_incomplete", "E3: mai citato nella logic"),
    (None, "logic_incomplete", "logic mancante"),
])
def test_a_logic_errors(logic, key, needle):
    """v79.12.2: only a dangling ref is a logic error (blocking) and a node that is no node an invalid schema; empty
    nodes, a one-arg any_of, an incomplete conditional, an uncited element and no logic are reported."""
    from src.rag import element_schemas as E
    checks = E.check(_penal(logic=logic), PENAL)
    assert any(needle in m for m in checks[key]), checks
    assert bool(_blocking(checks)) == (key in ("invalid", "logic_errors"))


def test_a_unreferenced_id_inside_a_special_comma():
    from src.rag import element_schemas as E
    s = _civil()
    sub = s["commi_speciali"][0]["schema"]
    sub["elementi"].append(_el("E2", "il bene è restituito", kind="condizione", comma="3"))
    assert E.check(s, CIVIL)["logic_incomplete"] == ["co.3/E2: mai citato nella logic"]


def test_a_comma_unknown_is_reported_not_blocking():
    from src.rag import element_schemas as E
    s = _civil()
    s["elementi"][0]["comma"] = "4"
    s["eccezioni"][0]["comma"] = "co. 2"                  # read as "2"
    checks = E.check(s, CIVIL)
    assert checks["comma_unknown"] == ["E1: comma 4 non tra 1, 2, 3"]
    assert _blocking(checks) == {}
    s = _penal()
    s["elementi"][0]["comma"] = "2"                        # no numbered commi: only "1"
    assert E.check(s, PENAL)["comma_unknown"] == ["E1: comma 2 non tra 1"]
    assert list(E.commi("1. Il primo comma.\n1-bis. Il comma aggiunto.\n2. Il secondo.")) == ["1", "1-bis", "2"]
    assert E.norm_comma("1 bis") == E.norm_comma("1bis") == "1-bis" and E.norm_comma("comma 2") == "2"


def test_a_kind_other_needs_a_note():
    from src.rag import element_schemas as E
    s = _penal()
    s["elementi"][0]["kind"] = "other"
    assert E.check(s, PENAL)["kind_other_unexplained"] == ["E1: kind other senza kind_note"]
    s["elementi"][0]["kind_note"] = "modalità della condotta"
    assert E.check(s, PENAL)["kind_other_unexplained"] == []


def test_a_usable_and_blocking():
    from src.rag import element_schemas as E
    line = {"schema": _penal(), "schema_version": 1, "checks": E.empty_checks()}
    assert E.usable(line) and E.blocking(line) == []
    assert E.blocking({**line, "checks": {**E.empty_checks(), "cut": True}}) == ["cut"]
    assert E.blocking({**line, "checks": {**E.empty_checks(), "comma_unknown": ["x"]}}) == []
    assert E.blocking({**line, "schema": None}) == ["schema"]
    assert E.blocking({**line, "schema_version": 2}) == ["schema_version"]


# --- A2 files and lookup ----------------------------------------------------------------------------------------------

def _write(path, lines):
    with open(path, "a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")


def _gen(schema=None, text=PENAL, doc=DOC, number="582", code="c.p.", **over):
    from src.rag import element_schemas as E
    line = E.make_line({"number": number, "code": code, "rubrica": "Lesione personale", "text": text}, "m-test",
                       _penal() if schema is None else schema, doc=doc)
    line.update(over)
    return line


def _row(text=PENAL, doc=DOC, number="582", code="c.p."):
    return {"_article": (number, code), "d": {"name": doc}, "s": {"plain_text": text, "titolo_articolo": "x"}}


def _on(monkeypatch):
    monkeypatch.setenv("ELEMENT_SCHEMAS", "true")


def test_a_missing_files_are_empty(monkeypatch):
    from src.rag import element_schemas as E
    _on(monkeypatch)
    assert E.generated() == {} and E.reviewed() == {}
    assert E.for_row(_row()) == (None, "no_schema")
    assert E.quality_report()["note"] == "nessuna revisione"


def test_a_flag_off_reads_nothing(monkeypatch, tmp_path):
    from src.rag import element_schemas as E
    _write(tmp_path / "schemas.jsonl", [_gen()])
    assert not E.enabled()
    assert E.for_row(_row()) == (None, "off")
    assert E.for_label("582 c.p.") == (None, "off")
    assert E.gate("582 c.p.", _gen()) is None


def test_a_paths_from_the_env_at_call_time(monkeypatch, tmp_path):
    from src.rag import element_schemas as E
    assert E.schema_file() == str(tmp_path / "schemas.jsonl")
    assert E.reviewed_file() == str(tmp_path / "reviewed.jsonl")
    monkeypatch.delenv("ARTICLE_SCHEMAS_FILE")
    assert E.schema_file().endswith("data/article_schemas.jsonl")
    monkeypatch.setenv("ARTICLE_SCHEMAS_FILE", "data/x.jsonl")
    assert E.schema_file() == f"{E.ROOT}/data/x.jsonl"


def test_a_generated_schema_found_and_last_line_wins(monkeypatch, tmp_path):
    from src.rag import element_schemas as E
    _on(monkeypatch)
    path = tmp_path / "schemas.jsonl"
    _write(path, [_gen(rubrica="prima")])
    line, why = E.for_row(_row())
    assert why == "schema" and line["rubrica"] == "prima" and line["source"] == "schema"
    assert line["note_di_inquadramento"] == []
    _write(path, [_gen(rubrica="seconda, più lunga")])         # a longer file: re-read
    assert E.for_row(_row())[0]["rubrica"] == "seconda, più lunga"
    assert list(E.generated()) == [(DOC, "582")]


def test_a_for_row_by_label_when_the_doc_differs(monkeypatch, tmp_path):
    from src.rag import element_schemas as E
    _on(monkeypatch)
    _write(tmp_path / "schemas.jsonl", [_gen(doc="Codice Penale 2025")])
    assert E.for_row(_row())[1] == "schema"                 # same label, same text hash


def test_a_hash_mismatch_and_no_schema(monkeypatch, tmp_path):
    from src.rag import element_schemas as E
    _on(monkeypatch)
    _write(tmp_path / "schemas.jsonl", [_gen()])
    assert E.for_row(_row(text=PENAL + " Testo cambiato."))== (None, "hash_mismatch")
    assert E.for_row(_row(text="  " + PENAL.replace(" ", "\n  ")))[1] == "schema"      # whitespace only: same hash
    assert E.for_row(_row(number="583")) == (None, "no_schema")
    assert E.for_row(_row(text="")) == (None, "hash_mismatch")
    assert E.for_row({"d": {}, "s": {}}) == (None, "no_schema")


def test_a_unusable_reasons(monkeypatch, tmp_path):
    from src.rag import element_schemas as E
    _on(monkeypatch)
    bad = _penal()
    bad["elementi"][0]["testo"] = "cagiona ad alcuno una lesione gravissima"
    _write(tmp_path / "schemas.jsonl", [_gen(bad)])
    line, why = E.for_row(_row())                                 # v79.12.2: the element dropped, the schema used
    assert why == "schema" and [e["id"] for e in line["schema"]["elementi"]] == ["E2", "E3"]
    assert [(a["id"], a["check"], a["action"]) for a in line["dropped"]][0] == ("E1", "ungrounded", "dropped")
    dangling = _penal(logic={"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}, {"ref": "E3"}, {"ref": "E7"}]})
    _write(tmp_path / "schemas.jsonl", [_gen(dangling)])
    assert E.for_row(_row()) == (None, "unusable:logic_errors")
    _write(tmp_path / "schemas.jsonl", [_gen(checks={**E.empty_checks(), "failed": "timeout"})])
    assert E.for_row(_row()) == (None, "unusable:failed")
    no_schema = _gen()
    no_schema["schema"] = None
    _write(tmp_path / "schemas.jsonl", [no_schema])
    assert E.for_row(_row()) == (None, "unusable:schema")
    _write(tmp_path / "schemas.jsonl", [_gen(checks={**E.empty_checks(), "cut": True})])
    assert E.for_row(_row()) == (None, "unusable:cut")


def test_a_stored_checks_clean_but_the_check_rerun_on_reading(monkeypatch, tmp_path):
    from src.rag import element_schemas as E
    _on(monkeypatch)
    bad = _penal(logic={"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}, {"ref": "E3"}, {"ref": "E9"}]})
    _write(tmp_path / "schemas.jsonl", [_gen(bad, checks=E.empty_checks())])     # a line claiming clean checks
    assert E.for_row(_row()) == (None, "unusable:logic_errors")


def _overlay(schema=None, source_hash=None, reviewed=True, notes=None, label="582 c.p.", **over):
    from src.rag import element_schemas as E
    line = {"label": label, "doc": DOC, "article": "582", "code": "c.p.", "reviewed": reviewed, "reviewer": "avv",
            "review_date": "2026-11-18", "source_hash": source_hash or E.text_hash(PENAL), "schema_version": 1,
            "schema": schema if schema is not None else (_penal(rapporto="corretto") if reviewed else None),
            "generated_schema": _penal(), "corrections": [{"field": "rapporto", "from": "lesioni personali",
                                                          "to": "corretto", "by": "avv"}],
            "note_di_inquadramento": notes or []}
    line.update(over)
    return line


NOTE = {"testo": "il dolo è generico", "numero": None, "autore": "avvocato", "fonte": "docs/x.md",
        "importato": "2026-11-18"}


def test_a_reviewed_wins_while_its_source_is_the_articles(monkeypatch, tmp_path):
    from src.rag import element_schemas as E
    _on(monkeypatch)
    _write(tmp_path / "schemas.jsonl", [_gen()])
    _write(tmp_path / "reviewed.jsonl", [_overlay(notes=[NOTE])])
    line, why = E.for_row(_row())
    assert why == "reviewed" and line["schema"]["rapporto"] == "corretto" and line["source"] == "reviewed"
    assert line["note_di_inquadramento"] == [NOTE]
    assert E.for_label("582 c.p.")[1] == "reviewed"
    assert E.for_label("582 c.p. co. 2", PENAL)[1] == "reviewed"            # a special comma's label: its article's


def test_a_reviewed_without_any_generated_line(monkeypatch, tmp_path):
    from src.rag import element_schemas as E
    _on(monkeypatch)
    _write(tmp_path / "reviewed.jsonl", [_overlay()])
    assert E.for_row(_row())[1] == "reviewed"


def test_a_reviewed_stale_is_never_used(monkeypatch, tmp_path):
    from src.rag import element_schemas as E
    _on(monkeypatch)
    changed = PENAL + " La pena è aumentata."
    _write(tmp_path / "schemas.jsonl", [_gen(text=changed)])
    _write(tmp_path / "reviewed.jsonl", [_overlay()])
    assert E.for_row(_row(text=changed)) == (None, "reviewed_stale")       # not even the fresh generated one
    assert E.for_label("582 c.p.") == (None, "reviewed_stale")             # without a text: against the generated


def test_a_reviewed_but_structurally_broken_is_not_read(monkeypatch, tmp_path):
    from src.rag import element_schemas as E
    _on(monkeypatch)
    _write(tmp_path / "reviewed.jsonl", [_overlay(schema=_penal(tipo=[5]))])   # v79.12.5: «boh» is counted, read
    assert E.for_row(_row()) == (None, "unusable:invalid")


def test_a_notes_only_overlay_merged_into_the_generated_line(monkeypatch, tmp_path):
    from src.rag import element_schemas as E
    _on(monkeypatch)
    _write(tmp_path / "schemas.jsonl", [_gen()])
    _write(tmp_path / "reviewed.jsonl", [_overlay(reviewed=False, notes=[NOTE], label="582 c.p.")])
    line, why = E.for_row(_row())
    assert why == "schema" and line["note_di_inquadramento"] == [NOTE]
    assert E.reviewed()["582 c.p."]["reviewed"] is False
    assert E.for_row(_row(number="583")) == (None, "no_schema")


def test_a_labels_normalised():
    from src.rag import element_schemas as E
    assert E.label_of("582", "Codice Penale 2026") == "582 c.p."
    assert E.norm_label("62 n. 2 c.p.") == "62 c.p."
    assert E.norm_label("2051 Codice Civile co. 2") == "2051 c.c."
    assert E.norm_label("131-BIS c.p.") == "131-bis c.p."


# --- A3 requisites and summary ----------------------------------------------------------------------------------------

REQ_KEYS = {"id", "comma", "text", "kind", "facts_status", "status", "why", "supporting", "contradicting", "missing",
            "question", "split", "schema_id", "schema_kind", "schema_connective", "schema_parti", "schema_grounding"}


def test_a_requisites_shape():
    from src.rag import element_schemas as E
    reqs = E.requisites(_row(), {"schema": _penal()}, {})
    assert [r["text"] for r in reqs] == ["cagiona ad alcuno una lesione personale",
                                         "dalla quale deriva una malattia nel corpo o nella mente",
                                         "coscienza e volontà di cagionare la lesione"]
    for r in reqs:
        assert REQ_KEYS <= set(r) <= REQ_KEYS | {"branches", "special"}
        assert r["kind"] == "requisito" and r["comma"] == 1 and r["split"] == "schema"
        assert r["status"] == r["facts_status"] == "da provare" and r["id"] == ""
    assert reqs[1]["branches"] == ["malattia nel corpo", "malattia nella mente"] and "branches" not in reqs[0]
    assert (reqs[1]["schema_id"], reqs[1]["schema_kind"], reqs[1]["schema_connective"]) == ("E2", "evento", "o")
    assert reqs[2]["schema_grounding"] == "inferred" and reqs[2]["schema_kind"] == "elemento soggettivo"


def test_a_requisites_special_commi_and_left_out_kinds():
    from src.rag import element_schemas as E
    s = _civil()
    s["elementi"].append(_el("E5", "il danno si presume", kind="condizione", comma="3", grounding="inferred",
                             fonte=None, spiegazione="x"))
    reqs = E.requisites(_row(CIVIL, "Codice Civile", "9999", "c.c."), {"schema": s}, {})
    texts = [r["text"] for r in reqs]
    assert "risarcire il danno al proprietario" not in texts             # quantificazione: no requisite
    by = {r["text"]: r for r in reqs}
    assert "special" not in by["lascia deteriorare il bene affidatogli"]
    assert by["il danno si presume"]["special"] == "presunzione" and by["il danno si presume"]["comma"] == 3
    sub = by["il bene è restituito oltre il termine convenuto"]
    assert sub["special"] == "presunzione" and sub["schema_id"] == "co3.E1" and sub["comma"] == 3
    summ = E.summary({"schema": s, "schema_version": 1})
    assert summ["altri_elementi"] == [{"id": "E4", "testo": "risarcire il danno al proprietario",
                                       "kind": "quantificazione", "comma": "1"}]
    assert summ["commi_speciali"] == [{"comma": "3", "kind": "presunzione"}]


def test_a_requisites_never_special_on_the_base_comma_and_non_digit_comma():
    from src.rag import element_schemas as E
    s = _civil()
    s["commi_speciali"][0]["comma"] = "1"
    s["commi_speciali"][0]["schema"]["elementi"][0]["comma"] = "1-bis"
    reqs = E.requisites(_row(CIVIL), {"schema": s}, {})
    assert not any(r.get("special") for r in reqs)
    assert reqs[-1]["comma"] == 1                         # non-digit -> 1 (issue_records._comma)


def test_a_requisites_drop_fragments_and_cap(monkeypatch):
    from src.rag import element_schemas as E
    from src.rag import interactive_flow
    logged = []
    monkeypatch.setattr(interactive_flow, "_log", lambda event, data=None, ms=None: logged.append((event, data)))
    s = _penal()
    s["elementi"].append(_el("E4", "il debitore", grounding="inferred", fonte=None, spiegazione="x"))
    frags = []
    reqs = E.requisites(_row(), {"schema": s}, {}, frags)
    assert "il debitore" not in [r["text"] for r in reqs]
    assert frags == [{"comma": "1", "text": "il debitore", "why": "soggetto senza condizione"}]
    assert logged and logged[0][0] == "element_fragment"
    many = _penal(elementi=[_el(f"E{i}", f"elemento numero {i} della condotta") for i in range(20)])
    assert len(E.requisites(_row(), {"schema": many}, {})) == 12


def test_a_summary():
    from src.rag import element_schemas as E
    line = {"schema": _penal(), "schema_version": 1, "note_di_inquadramento": [NOTE]}
    summ = E.summary(line)
    assert summ["source"] == "schema" and not summ["reviewed"]
    assert summ["tipo"] == ["sanction"] and summ["rapporto"] == "lesioni personali"
    assert summ["legittimato"] == [{"ruolo": "persona offesa", "condizione": None}]
    assert summ["eccezioni"] == 0 and summ["note_di_inquadramento"] == [NOTE] and summ["logic"]["op"] == "all_of"
    assert E.summary(_overlay())["source"] == "reviewed"


def test_a_kind_rule():
    from src.rag import element_schemas as E
    assert E.kind_rule("elemento soggettivo") == "mental" and E.kind_rule("termine") == "termine"
    assert E.kind_rule("condizione") == "condition" and E.kind_rule("condotta") == "plain"
    assert E.kind_rule(None) == "plain"


# --- A4 roles, shadow admission, gate ---------------------------------------------------------------------------------

@pytest.mark.parametrize("role,entries,expected", [
    ("attore", [{"ruolo": "attore", "condizione": None}], (True, False)),
    ("attore", [{"ruolo": "attore", "condizione": "se proprietario"}], (True, True)),
    ("imputato", [{"ruolo": "indagato", "condizione": None}], (True, False)),
    ("persona_offesa", [{"ruolo": "parte civile", "condizione": None}], (True, False)),
    ("convenuto", [{"ruolo": "chiunque", "condizione": None}], (True, False)),
    ("attore", [{"ruolo": "creditore", "condizione": None}], (True, True)),
    ("convenuto", [{"ruolo": "attore", "condizione": None}], (False, False)),
    ("imputato", [{"ruolo": "persona offesa", "condizione": None}], (False, False)),
    ("sconosciuto", [{"ruolo": "attore", "condizione": None}], (True, True)),
    ("attore", [], (False, False)),
    ("persona offesa", ["persona offesa"], (True, False)),
    ("attore", [{"ruolo": "creditore", "condizione": None}, {"ruolo": "attore", "condizione": None}], (True, False)),
])
def test_a_roles_match(role, entries, expected):
    from src.rag import element_schemas as E
    assert E.roles_match(role, entries) == expected


def test_a_shadow_admission_claim_and_refusals():
    from src.rag import element_schemas as E
    line = {"schema": _civil()}
    out = E.shadow_admission("9999 c.c.", line, "attore", "weapon")
    assert out["schema"] and out["claim"] is True and out["conditional"] is True
    assert out["defence"] is False and out["contest"] is False
    assert "defence: tipo claim-conferring non è difesa né scriminante" in out["why"]
    out = E.shadow_admission("9999 c.c.", line, "convenuto", "weapon")
    assert out["claim"] is False and "claim: legittimato attore non comprende convenuto" in out["why"]
    out = E.shadow_admission("9999 c.c.", line, "convenuto", "threat")
    assert out["contest"] is True and out["claim"] is False
    out = E.shadow_admission("9999 c.c.", line, "attore", "threat")
    assert out["claim"] is False and "claim: norma della controparte" in out["why"]


def test_a_shadow_admission_defence_contest_and_no_schema():
    from src.rag import element_schemas as E
    scrim = {"schema": _penal(tipo=["scriminante"], legittimato=[{"ruolo": "chiunque", "condizione": None}])}
    assert E.shadow_admission("52 c.p.", scrim, "imputato", "weapon")["defence"] is True
    accused = {"schema": _penal(tipo=["defence"], legittimato=[{"ruolo": "convenuto", "condizione": None}])}
    assert E.shadow_admission("1460 c.c.", accused, "convenuto", "weapon")["defence"] is True
    limiting = {"schema": _penal(tipo=["limiting"])}
    assert E.shadow_admission("1227 c.c.", limiting, "convenuto", "weapon")["contest"] is True
    out = E.shadow_admission("1227 c.c.", limiting, "attore", "weapon")
    assert out["contest"] is False and "contest: norma limitativa, ma il cliente è attore, non convenuto" in out["why"]
    sanction = {"schema": _penal()}
    assert E.shadow_admission("582 c.p.", sanction, "imputato", "threat")["contest"] is True
    assert "contest: norma del cliente, non della controparte" in \
        E.shadow_admission("582 c.p.", sanction, "persona_offesa", "weapon")["why"]
    assert E.shadow_admission("582 c.p.", None, "imputato", "threat") == {
        "article": "582 c.p.", "schema": False, "tipo": [], "legittimato": [], "claim": None, "defence": None,
        "contest": None, "conditional": False, "why": ["no_schema"]}


def test_a_gate(monkeypatch):
    from src.rag import element_schemas as E
    sanction = {"schema": _penal()}
    limiting = {"schema": _penal(tipo=["limiting", "definitional"])}
    assert E.gate("582 c.p.", sanction) is None                       # flag off: no opinion
    _on(monkeypatch)
    assert E.gate("582 c.p.", sanction) is True
    assert E.gate("582 c.p.", {"schema": _penal(tipo=["procedural", "defence"])}) is True
    assert E.gate("1227 c.c.", limiting) is None                      # unreviewed: never gates out
    assert E.gate("1227 c.c.", {**limiting, "reviewed": True}) is False
    assert E.gate("1227 c.c.", {**limiting, "source": "reviewed"}) is False
    assert E.gate("1227 c.c.", None) is None


def test_a_decide_and_hard_invariants(monkeypatch):
    from src.rag import element_schemas as E
    _on(monkeypatch)
    sanction = {"schema": _penal()}
    limiting = {"schema": _penal(tipo=["limiting"])}
    assert E.hard_excluded("100 c.p.p.") and E.hard_excluded("40 c.p.") and E.hard_excluded("10 disp. prel.")
    assert not E.hard_excluded("52 c.p.") and not E.hard_excluded("582 c.p.") and not E.hard_excluded("2043 c.c.")
    assert E.decide("40 c.p.", sanction, False) == (False, "invariant", True)
    assert E.decide("40 c.p.", sanction, True) == (True, "invariant", False)          # the defence route's own
    assert E.decide("52 c.p.", {"schema": _penal(tipo=["scriminante"])}, True) == (True, "schema", False)
    assert E.decide("1227 c.c.", limiting, True) == (True, "current", True)          # unreviewed: logged, not applied
    assert E.decide("1227 c.c.", {**limiting, "reviewed": True}, True) == (False, "schema", True)
    assert E.decide("9999 c.c.", sanction, False) == (True, "schema", True)
    assert E.decide("9999 c.c.", None, False) == (False, "current", False)


# --- A5 quality, prompt, reply, line ----------------------------------------------------------------------------------

def test_a_quality_identical_and_corrected():
    from src.rag import element_schemas as E
    q = E.quality(_penal(), _penal())
    assert q["element_precision"] == q["element_recall"] == q["logic_accuracy"] == 1.0
    assert q["comma_accuracy"] == q["grounding_accuracy"] == 1.0 and q["exception_recall"] is None
    gold = _civil()
    pred = _civil(logic={"op": "all_of", "args": [{"ref": "E4"}, {"ref": "E3"}, {"ref": "E2"}, {"ref": "E1"}]})
    assert E.quality(pred, gold)["logic_accuracy"] == 1.0             # all_of's order is no difference
    pred = _civil()
    pred["elementi"][1]["grounding"] = "inferred"
    pred["elementi"][2]["comma"] = "2"
    pred["elementi"][3] = _el("E4", "somma liquidata dal giudice", kind="quantificazione")
    pred["eccezioni"] = []
    pred["logic"] = {"op": "any_of", "args": [{"ref": "E1"}, {"ref": "E2"}, {"ref": "E3"}, {"ref": "E4"}]}
    q = E.quality(pred, gold)
    assert (q["predicted"], q["gold"], q["matched"]) == (5, 5, 4)
    assert q["element_precision"] == q["element_recall"] == 0.8
    assert q["comma_accuracy"] == 0.75 and q["grounding_accuracy"] == 0.75
    assert q["exception_recall"] == 0.0 and q["logic_accuracy"] == 0.0


def test_a_quality_report_without_and_with_reviewed_lines(monkeypatch, tmp_path):
    from src.rag import element_schemas as E
    bad = _penal()
    bad["elementi"][0]["testo"] = "cagiona ad alcuno una lesione gravissima"
    gen = [_gen(), _gen(bad)]
    rep = E.quality_report(lines=[], generated_lines=gen)
    assert rep["n"] == 0 and rep["note"] == "nessuna revisione"
    assert all(rep[k] is None for k in ("element_precision", "element_recall", "logic_accuracy", "exception_recall",
                                        "comma_accuracy", "grounding_accuracy", "correction_rate"))
    assert rep["grounding_rate_auto"] == 0.75 and rep["explicit_elements"] == 4
    plain = _overlay(corrections=[], schema=_penal())
    rep = E.quality_report(lines=[_overlay(), plain, _overlay(reviewed=False)], generated_lines=gen)
    assert rep["n"] == 2 and rep["reviewed"] == 2 and rep["note"] == ""
    assert rep["correction_rate"] == 0.5 and rep["element_precision"] == 1.0 and rep["logic_accuracy"] == 1.0
    _write(tmp_path / "reviewed.jsonl", [_overlay()])
    _write(tmp_path / "schemas.jsonl", gen)
    assert E.quality_report()["n"] == 1 and E.quality_report()["generated_lines"] == 1      # one key: last wins


def test_a_prompt_says_what_the_spec_asks():
    from src.rag import element_schemas as E
    p = E.SCHEMA_SYSTEM
    for needle in ("UN SOLO oggetto JSON", "OGNI comma", "«o»", "«ovvero»", "COPIATO ALLA LETTERA", '"inferred"',
                   "spiegazione", "OGNI id", "clausola di salvezza", "abrogato", '"comodato"', '"condominio"',
                   '"claim-conferring"', "Più funzioni sono ammesse", '"nesso_causale"', '"procedibilità"'):
        assert needle in p, needle
    assert E.prompt_hash() == E.prompt_hash() and len(E.prompt_hash()) == 12
    msg = E.user_message({"number": "9999", "code": "c.c.", "rubrica": "Custodia  sintetica", "text": CIVIL},
                         note=E.JSON_NOTE)
    assert msg.startswith("ARTICOLO: art. 9999 c.c. - Custodia sintetica\nCOMMI DEL TESTO: 1, 2, 3\n\nTESTO UFFICIALE:\n1. ")
    assert msg.endswith("NOTA: " + E.JSON_NOTE)


@pytest.mark.parametrize("reply", [
    '{"tipo": ["sanction"]}',
    '```json\n{"tipo": ["sanction"]}\n```',
    'Ecco lo schema:\n```\n{"tipo": ["sanction"]}\n```\nSpero sia utile.',
    'Ecco lo schema: {"tipo": ["sanction"]} -- fine {nota}',
    '<think>ragiono {forse}</think>\n{"tipo": ["sanction"]}',
    '{prosa} {"tipo": ["sanction"]}',
])
def test_a_parse_reply_tolerates_fences_and_prose(reply):
    from src.rag import element_schemas as E
    assert E.parse_reply(reply) == {"tipo": ["sanction"]}


@pytest.mark.parametrize("reply", ["", "nessuno schema", '{"tipo": ["sanction"', "[1, 2]", "```json\n{tipo: x}\n```"])
def test_a_parse_reply_raises_on_invalid_json(reply):
    from src.rag import element_schemas as E
    with pytest.raises(ValueError):
        E.parse_reply(reply)


def test_a_make_line():
    from src.rag import element_schemas as E
    s = _penal()
    s["elementi"][0]["fonte"]["start"] = 99
    line = E.make_line({"number": "582", "code": "c.p.", "rubrica": " Lesione  personale", "text": PENAL}, "qwen-x",
                       s, seconds=12.34, usage={"prompt_tokens": 10, "completion_tokens": 5}, doc=DOC)
    assert list(line) == ["doc", "article", "code", "label", "rubrica", "schema_version", "source_hash", "text_hash",
                          "prompt_hash", "generation_hash", "model", "created_at", "seconds", "usage", "schema",
                          "checks", "dropped", "text"]
    assert (line["doc"], line["article"], line["code"], line["label"]) == (DOC, "582", "c.p.", "582 c.p.")
    assert line["rubrica"] == "Lesione personale" and line["text"] == PENAL and line["seconds"] == 12.3
    assert line["source_hash"] == line["text_hash"] == E.text_hash(PENAL)
    assert line["generation_hash"] == E.generation_hash(E.text_hash(PENAL), "qwen-x")
    assert line["generation_hash"] != E.generation_hash(E.text_hash(PENAL), "other-model")
    assert line["usage"] == {"prompt_tokens": 10, "completion_tokens": 5}
    assert line["schema"]["elementi"][0]["fonte"]["start"] == 9 and E.usable(line)
    assert line["checks"] == E.empty_checks() and line["dropped"] == []
    failed = E.make_line({"number": "582", "code": "c.p.", "text": PENAL}, "qwen-x", None, failed="json", cut=True)
    assert failed["schema"] is None and failed["checks"]["failed"] == "json" and failed["checks"]["cut"] is True
    assert not E.usable(failed)
    assert E.make_line({"number": "582", "code": "c.p.", "text": PENAL}, "m", None)["checks"]["failed"]


def test_a_flag_is_found_by_the_conftest_scan():
    import pathlib
    import re
    src = (pathlib.Path(__file__).parent.parent / "src" / "rag" / "element_schemas.py").read_text()
    rx = re.compile(r"""^[A-Z_]*FLAG[A-Z_]*\s*=\s*["']([A-Z][A-Z0-9_]+)["']""", re.M)
    assert rx.findall(src) == ["ELEMENT_SCHEMAS"]


def test_a_deploy_excludes_the_generated_file_only():
    import pathlib
    lines = {ln.strip() for ln in (pathlib.Path(__file__).parent.parent / "scripts" / "deploy_exclude.txt")
             .read_text().splitlines() if ln.strip() and not ln.startswith("#")}
    assert "article_schemas.jsonl" in lines
    assert not any("reviewed" in ln for ln in lines)
