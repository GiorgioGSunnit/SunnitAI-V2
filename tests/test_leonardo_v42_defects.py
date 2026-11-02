"""Leonardo v42: the defects, from their exact strings, on invented cases."""
import os

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

from src.rag import fact_signals as fs, text_tidy  # noqa: E402
import src.rag.defensive_generation as dg  # noqa: E402
import src.rag.plan_then_write as ptw  # noqa: E402

A1158 = "La proprietà dei beni immobili e gli altri diritti reali di godimento sui beni medesimi si acquistano in virtù del possesso continuato per venti anni."
A614 = ("Con il provvedimento di condanna all'adempimento di obblighi diversi dal pagamento di una somma di denaro il giudice, salvo che ciò sia "
        "manifestamente iniquo, fissa, su richiesta di parte, la somma di denaro dovuta dall'obbligato per ogni violazione.")
A1079 = "Il titolare della servitù può farne riconoscere in giudizio l'esistenza contro chi ne contesta l'esercizio e può far cessare gli eventuali impedimenti e turbative."
ROWS = [{"_article": ("1158", "c.c."), "s": {"plain_text": A1158}}, {"_article": ("614-bis", "c.p.c."), "s": {"plain_text": A614}},
        {"_article": ("1079", "c.c."), "s": {"plain_text": A1079}}, {"_article": ("1062", "c.c."), "s": {"plain_text": "Testo breve."}}]


def clean(text):
    return text_tidy.without_copies(text, ROWS)


def test_a_copy_in_brackets_goes_the_citation_stays():
    out, log = clean(f"La servitù è sorta alla separazione (art. 1062 c.c. {A1158}) e va tutelata.")
    assert out == "La servitù è sorta alla separazione (art. 1062 c.c.) e va tutelata." and log[0]["kind"] == "copy out of its brackets"


def test_a_whole_comma_in_brackets_followed_by_its_parenthesis():
    out, _ = clean(f"Leonardo può chiedere una somma per il ritardo (art. 614-bis c.p.c. {A614}) nel giudizio.")
    assert out == "Leonardo può chiedere una somma per il ritardo (art. 614-bis c.p.c.) nel giudizio."


def test_a_whole_sentence_that_is_a_comma_becomes_a_quote():
    out, log = clean(f"La domanda è fondata. {A1158} Il termine va verificato.")
    assert f"«{A1158[:-1]}» (art. 1158 c.c.). Il termine" in out and log[0]["kind"] == "a whole comma quoted"


def test_a_quote_followed_by_its_citation_is_not_cited_twice():
    assert clean(f"{A1158} (art. 1158 c.c.).")[0] == f"«{A1158[:-1]}» (art. 1158 c.c.)."


def test_a_copy_after_its_own_citation_is_quoted():
    out, _ = clean(f"L'art. 1079 c.c. dispone che {A1079}")
    assert out == f"L'art. 1079 c.c. dispone che «{A1079[:-1]}»."


def test_a_copy_mid_sentence_drops_the_sentence():
    out, log = clean(f"Leonardo sostiene che {A1079[:-1]} e il vicino replica. La domanda è fondata.")
    assert out == "La domanda è fondata." and "sentence dropped" in log[0]["kind"]


def test_a_copy_after_another_article_s_citation_goes():
    assert clean(f"L'art. 1062 c.c. dispone che {A1079} La domanda è fondata.")[0] == "La domanda è fondata."


def test_a_quote_or_eleven_words_stay():
    for text in (f"Lo dice la norma: «{A1158}» (art. 1158 c.c.).", "La proprietà dei beni immobili e gli altri diritti reali restano fermi."):
        assert clean(text) == (text, [])


def test_the_pipeline_s_check_runs_it():
    assert "(art. 1062 c.c.)" in text_tidy.against_articles(f"La servitù è sorta (art. 1062 c.c. {A1158}).", ROWS)[0]


def test_the_cut_sentence_and_the_stray_colon():
    assert text_tidy.tidy("Le norme alternative o subordinate sono.\n\nLa domanda è fondata.")[0] == "La domanda è fondata."
    assert text_tidy.stop_comma("Mediazione obbligatoria:, la domanda è improcedibile.") == "Mediazione obbligatoria: la domanda è improcedibile."


def test_the_writer_is_told_to_paraphrase():
    plan = {"parte_assistita": "Leonardo", "posizione": ""}
    assert "solo tra «…»" in ptw._section_system(plan, "it")


# --- 2 ---------------------------------------------------------------------------------------------------------------

def row(number, grade=None, **extra):
    tags = {"_grade": grade} if grade is not None else {}
    return {"_article": (number, "c.c."), **tags, **extra, "s": {"plain_text": f"Testo dell'articolo {number}.", "titolo_articolo": f"Rubrica {number}"}}


def plan_of(*numbers):
    return {"parte_assistita": "Leonardo", "questioni": [{"titolo": "Passaggio", "argomento": "Il passaggio.", "norme": [(n, "c.c.") for n in numbers]}]}


SECTIONS = [("Applicazione al caso", "**Passaggio**\n\nTesto del parere.")]


@pytest.fixture
def events(monkeypatch):
    seen = []
    monkeypatch.setattr(dg, "vlog", lambda name, data=None, **k: seen.append((name, data)))
    for name in ("WRITER_RULES_ENABLED", "REVIEW_FIXES_ENABLED", "GROUNDED_REVIEW_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(ptw, "SECTION_TITLES", ptw.SECTION_TITLES[:3] + ("Applicazione al caso",) + ptw.SECTION_TITLES[4:])
    monkeypatch.setattr(dg, "_chat_json", lambda *a, **k: [{"articolo": "A0", "compatibile": True}])
    return seen


def test_an_ungraded_article_is_never_covered(monkeypatch, events):
    monkeypatch.setattr(dg, "_call_chat", lambda *a, **k: pytest.fail("called"))
    assert ptw.cover_plan_articles(plan_of("2160"), SECTIONS, [row("2160")], [], "Il caso.") == SECTIONS


def test_an_article_a_fired_rule_excludes_is_never_covered(monkeypatch, events):
    monkeypatch.setattr(dg, "_call_chat", lambda *a, **k: pytest.fail("called"))
    context = {"excluded": [("1052", "c.c."), ("1038", "c.c.")]}
    assert ptw.cover_plan_articles(plan_of("1052", "1038"), SECTIONS, [row("1052", 3), row("1038", 2)], [], "Il caso.", context) == SECTIONS


def test_the_landlocked_rule_excludes_them():
    rules = {r["id"]: r for r in fs.load_rules()}
    assert {"1052 c.c.", "1038 c.c."} == set(rules["fondo_intercluso"]["conclusion_excludes"])
    facts = {k: ([] if isinstance(v, list) else "") for k, v in fs.SCHEMA.items()}
    facts["events"] = [{"what": "il fondo è intercluso", "date": "", "who": "", "kind": ""}]
    signals = fs.signals_from(facts, rules=True)
    assert ("1052", "c.c.") in ptw.statement_context({"fact_signals": signals})["excluded"]


@pytest.mark.parametrize("reply", ["L'art. 2160 c.c. non si ricollega direttamente al passaggio di Leonardo sul fondo.",
                                   "L'art. 2160 c.c. riguarda una materia estranea al passaggio del caso in esame."])
def test_an_insertion_saying_it_does_not_apply_is_refused(monkeypatch, events, reply):
    monkeypatch.setattr(dg, "_call_chat", lambda *a, **k: reply)
    assert "2160" not in ptw.cover_plan_articles(plan_of("2160"), SECTIONS, [row("2160", 2)], [], "Il caso.")[0][1]


def test_an_insertion_denying_a_conclusion_is_refused(monkeypatch, events):
    monkeypatch.setattr(dg, "_call_chat", lambda *a, **k: "Il passaggio coattivo non spetta a Leonardo per l'art. 1056 c.c. nel caso concreto.")
    context = {"conclusions": [{"id": "fondo_intercluso", "conclusion_subject": "passaggio coattiv", "conclusion_unless": ""}]}
    assert "1056" not in ptw.cover_plan_articles(plan_of("1056"), SECTIONS, [row("1056", 3)], [], "Il caso.", context)[0][1]


def test_a_graded_article_is_still_covered(monkeypatch, events):
    monkeypatch.setattr(dg, "_call_chat", lambda *a, **k: "Il passaggio è tutelato dall'art. 1079 c.c. per i fatti del caso concreto.")
    assert "1079" in ptw.cover_plan_articles(plan_of("1079"), SECTIONS, [row("1079", 2)], [], "Il caso.")[0][1]


# --- 3 ---------------------------------------------------------------------------------------------------------------

TERM = {"terms": [{"id": "usucapione_ordinaria", "ref": "art. 1158 c.c.", "length": "20 anni", "status": "NON VERIFICABILE"}]}


@pytest.mark.parametrize("sentence", ["La servitù si è consolidata per usucapione ventennale.",
                                      "Il lungo uso finisce confermando l'acquisizione per usucapione.",
                                      "Il diritto è stato acquisito per usucapione ventennale.",
                                      "L'usucapione si è perfezionata.", "L'usucapione è maturata."])
def test_an_acquisition_stated_as_done_goes(sentence):
    assert "usucapione_ordinaria is NON VERIFICABILE" in ptw._wrong_statement(sentence, TERM)


@pytest.mark.parametrize("sentence", ["La giurisprudenza consolidata ammette l'usucapione della servitù apparente.",
                                      "L'usucapione sarebbe acquisita se il possesso risultasse ventennale.",
                                      "L'usucapione non è stata acquisita con certezza.",
                                      "L'usucapione ventennale è da verificare."])
def test_a_rule_a_condition_or_a_doubt_stays(sentence):
    assert ptw._wrong_statement(sentence, TERM) == ""


def test_a_settled_term_lets_it_stand():
    settled = {"terms": [{**TERM["terms"][0], "status": "DECORSO"}]}
    assert ptw._wrong_statement("La servitù si è consolidata per usucapione ventennale.", settled) == ""
