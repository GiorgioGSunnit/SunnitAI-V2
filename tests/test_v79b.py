"""v79 block B (79.1) -- grounded rule statements, on synthetic sentences only:

B1. a rule statement (what a norm requires, permits, excludes, punishes or when it applies) stays only when it traces to
    a record element of the article it names, a store field (procedibilità, termini, pena, competenza) or a registry
    ruling's quoted passage -- in every written section, §6's cells and §7's objections and replies included; a removal
    is logged with the sentence and the reason
B2. the citation and scope checks on §6 and §7 as on the rest: an objection or a reply the checks empty is dropped, the
    standard fallback fills the section
B3. the nightly's output check: no rule statement without a source, no article outside the scope
"""

import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for name, value in (("LLM_API_KEY", "k"), ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "k"), ("NEO4J_URI", "bolt://localhost:7687"), ("NEO4J_USER", "n"),
                        ("NEO4J_PASSWORD", "x")):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))


def _script(name):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _req(i, text, status="provato"):
    return {"id": f"R{i}", "comma": 1, "kind": "requisito", "text": text, "status": status}


RECORDS = {"646 c.p.": {"rubrica": "Appropriazione indebita", "side": "threat", "charged": 1, "requisites": [
    _req(1, "il possesso del denaro o della cosa mobile altrui"),
    _req(2, "l'appropriazione per procurare a sé o ad altri un ingiusto profitto", "da provare")]}}
STORE = {"646 c.p.": ["competenza", "pena", "procedibilita", "termini"], "585 c.p.p.": ["termini"]}
KEPT = [{"article": "646 c.p.", "passage": "Il momento consumativo coincide con l'interversione del possesso, quando "
                                           "l'agente compie un atto di dominio sulla cosa."}]


def _src():
    from src.rag import rule_statements
    return rule_statements.sources({"records": RECORDS, "store_fields": STORE, "rulings_record": {"kept": KEPT}})


# --- B1: the definition -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sentence", [
    "L'art. 646 c.p. richiede il possesso della cosa altrui.",
    "Il reato si configura con l'appropriazione.",
    "L'elemento è integrato quando l'agente dispone della cosa.",
    "Non è necessario il dolo specifico.",
    "La norma prevede la reclusione.",
    "L'art. 646 c.p. punisce chi si appropria.",
    "La causa di non punibilità si applica ai fatti tenui.",
    "La scriminante esclude l'antigiuridicità.",
    "Il termine decorre dalla conoscenza del fatto.",
])
def test_the_rule_verbs(sentence):
    from src.rag import rule_statements
    assert rule_statements.is_rule(sentence)


@pytest.mark.parametrize("sentence", [
    "Alfa ha consegnato a Beta un'automobile.",
    "L'elemento «il possesso» (R1) dell'art. 646 c.p. è escluso dai fatti.",
    "Il cliente chiede come difendersi.",
])
def test_not_a_rule_statement(sentence):
    from src.rag import rule_statements
    assert not rule_statements.is_rule(sentence)


# --- B1: the sources ----------------------------------------------------------------------------------------------------

def test_a_rule_traced_to_a_record_element_stays():
    from src.rag import rule_statements
    s = "L'art. 646 c.p. richiede il possesso del denaro o della cosa mobile altrui."
    assert rule_statements.source_of(s, _src()) == "element:646 c.p."


def test_a_rule_traced_to_a_store_field_stays():
    from src.rag import rule_statements
    assert rule_statements.source_of("Per l'art. 646 c.p. si procede a querela della persona offesa, che è richiesta.",
                                     _src()) == "store:procedibilita:646 c.p."
    assert rule_statements.source_of("L'art. 646 c.p. punisce il fatto con la reclusione.", _src()) == \
        "store:pena:646 c.p."
    assert rule_statements.source_of("Il termine dell'art. 585 c.p.p. decorre dal deposito.", _src()) == \
        "store:termini:585 c.p.p."


def test_a_rule_traced_to_a_registry_passage_stays():
    from src.rag import rule_statements
    s = "Per l'art. 646 c.p. si configura con l'interversione del possesso, quando l'agente compie un atto di dominio."
    assert rule_statements.source_of(s, _src()).startswith(("element:", "passage:"))
    only_passage = "Il reato dell'art. 646 c.p. si configura con un atto di dominio dell'agente (interversione)."
    assert rule_statements.source_of(only_passage, _src()) == "passage:646 c.p."


def test_a_rule_with_no_source_is_removed_and_logged():
    from src.rag import rule_statements
    log = []
    text = ("Il cliente ha restituito l'auto. L'art. 2043 c.c. richiede un fatto doloso o colposo che cagiona un danno "
            "ingiusto. L'art. 646 c.p. richiede il possesso della cosa mobile altrui.")
    out = rule_statements.check(text, _src(), log)
    assert "2043" not in out and "restituito" in out and "possesso della cosa" in out
    assert log == [{"sentence": "L'art. 2043 c.c. richiede un fatto doloso o colposo che cagiona un danno ingiusto.",
                    "reason": rule_statements.reason("L'art. 2043 c.c.")}] and "2043 c.c." in log[0]["reason"]


def test_a_rule_naming_no_article_traces_to_any_source():
    from src.rag import rule_statements
    assert rule_statements.source_of("Il reato richiede il possesso della cosa mobile altrui.", _src())
    assert not rule_statements.source_of("Il reato richiede la violenza sulla persona offesa.", _src())


def test_a_rule_naming_an_article_traces_only_to_that_article():
    from src.rag import rule_statements
    assert not rule_statements.source_of("L'art. 624 c.p. richiede il possesso della cosa mobile altrui.", _src())


def test_tables_and_headings_are_left_as_they_are():
    from src.rag import rule_statements
    text = "| Tesi | Pro |\n|---|---|\n| La norma richiede la violenza | x |"
    assert rule_statements.check(text, _src(), []) == text


# --- B1/B2: compose -----------------------------------------------------------------------------------------------------

def _compose_ctx():
    from src.rag import lawyer_method as M, posture
    p = {"rito": "penale", "family": "penale", "role": "imputato", "stage": "primo_grado",
         "entry": posture.entry_for("penale", "imputato", "primo_grado")}
    ctx = M.prepare({"records": RECORDS, "sentences": [{"id": "S1", "text": "Alfa trattiene l'auto di Beta.",
                                                        "source": "caso", "certainty": "accertato"}]},
                    p, [], "Alfa trattiene l'auto di Beta.", "Alfa", "", list(RECORDS), [], [], [], [],
                    link_call=lambda *a: {}, case_id_="v79b")
    ctx["store_fields"] = STORE
    ctx["candidates"] = M.thesis_candidates(ctx)
    return ctx


def _plan(ctx, objections):
    from src.rag import lawyer_method as M
    vals = {c["id"]: {"pro": "Il fatto non è provato.", "contro": "L'art. 2043 c.c. richiede un danno ingiusto.",
                      "realismo": "medio", "argomento": "a"} for c in ctx["candidates"]}
    plan = M.clean_method({"valutazioni": vals}, ctx["candidates"])
    plan["obiezioni"] = objections
    return plan


def test_compose_removes_the_unsourced_rule_in_6_and_7_and_drops_an_emptied_reply():
    from src.rag import lawyer_method as M
    ctx = _compose_ctx()
    plan = _plan(ctx, [
        {"obiezione": "L'accusa sosterrà il possesso", "risposta": "La norma richiede la violenza sulla persona."},
        {"obiezione": "L'accusa sosterrà l'ingiusto profitto",
         "risposta": "Il fatto non è provato. La norma richiede la violenza sulla persona."},
        {"obiezione": "L'art. 2043 c.c. richiede un danno ingiusto.", "risposta": "Il danno non c'è."}])
    log = {}
    out = M.compose([("", "Testa.")], ctx, plan, (), log)
    assert "2043" not in out and "violenza sulla persona" not in out
    assert "L'accusa sosterrà il possesso" not in out                     # B2: its reply emptied, dropped
    assert "Il fatto non è provato" in out and "ingiusto profitto" in out
    removed = [x["sentence"] for x in log["rule_statements_removed"]]
    assert "La norma richiede la violenza sulla persona." in removed
    assert any("reply" in x for x in log["objections_dropped"]) and any("objection" in x for x in log["objections_dropped"])
    section7 = dict(M.answer_sections(out))[M.title("obiezioni")]
    assert section7.count(M.ANSWER_HEAD) == M.OBJECTIONS                  # the standard fallback fills it


def test_compose_keeps_the_sources_for_the_nightly_and_its_check_passes():
    from src.rag import lawyer_method as M
    ctx = _compose_ctx()
    out = M.compose([("", "Testa.")], ctx, _plan(ctx, []), (), {})
    got = M.output_check(out, ctx["output_sources"])
    assert got == {"unsourced": [], "out_of_scope": [], "wording": []}


def test_the_output_check_finds_what_the_checks_would_have_removed():
    from src.rag import lawyer_method as M
    ctx = _compose_ctx()
    out = M.compose([("", "Testa.")], ctx, _plan(ctx, []), (), {})
    bad = out.replace(f"**2. {M.title('fatti')}**", f"**2. {M.title('fatti')}**\n\nL'art. 2043 c.c. richiede un danno.")
    got = M.output_check(bad, ctx["output_sources"])
    assert got["unsourced"] == ["L'art. 2043 c.c. richiede un danno."] and got["out_of_scope"] == ["2043 c.c."]


def test_the_nightly_check_reads_the_sources_from_the_state():
    nightly = _script("nightly_sim")
    from src.rag import lawyer_method as M
    ctx = _compose_ctx()
    out = M.compose([("", "Testa.")], ctx, _plan(ctx, []), (), {})
    assert nightly.grounded_check(out, {"output_sources": ctx["output_sources"]})["ok"] is True
    missing = nightly.grounded_check(out, {})
    assert missing["ok"] is False and "not applied" in missing["why"]
