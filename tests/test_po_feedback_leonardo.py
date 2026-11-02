"""PO feedback on Leonardo (live v39): general fixes, on invented cases."""
import os

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

from src.rag import fact_signals as fs, text_tidy  # noqa: E402
import src.rag.plan_then_write as ptw  # noqa: E402


def facts(**kw):
    base = {k: ([] if isinstance(v, list) else "") for k, v in fs.SCHEMA.items()}
    base.update(kw)
    return base


def row(number, text, rubrica, code="c.c.", in_plan=True):
    return {"_article": (number, code), "_in_plan": in_plan, "s": {"plain_text": text, "titolo_articolo": rubrica}}


CLOSED = {"what": "Sempronia chiude il cancello sulla stradella", "date": "marzo 2016", "who": "Sempronia", "kind": "chiusura/ostruzione accesso"}
ROWS = [row("1061", "Le servitù non apparenti non possono acquistarsi per usucapione o per destinazione del padre di famiglia. "
                    "Non apparenti sono le servitù quando non si hanno opere visibili e permanenti destinate al loro esercizio.",
            "Servitù non apparenti"),
        row("1079", "Il titolare della servitù può farne riconoscere in giudizio l'esistenza contro chi ne contesta l'esercizio.",
            "Accertamento della servitù e altri provvedimenti di tutela")]
HEAD = "**3. Norme e giurisprudenza**\n\n"
TAIL = "\n\n**4. Applicazione al caso**\n\nTesto."


def norms(body):
    text, log = text_tidy.headed_norms(f"{HEAD}{body}{TAIL}", ROWS)
    return text[len(HEAD):-len(TAIL)].strip(), log


# --- 1 ---------------------------------------------------------------------------------------------------------------

def test_a_paragraph_that_lost_its_first_sentence_gets_its_article():
    out, log = norms("esclude che le servitù non apparenti possano acquistarsi per usucapione o per destinazione.")
    assert out.startswith("L'art. 1061 c.c. (Servitù non apparenti) esclude che le servitù") and log[0]["kind"] == "paragraph given its article"


def test_a_capitalised_verb_left_by_the_tidy_too():
    assert norms("Esclude che le servitù non apparenti possano acquistarsi per usucapione.")[0].startswith("L'art. 1061 c.c. (Servitù non apparenti) esclude")


def test_a_comma_reference_takes_the_article():
    out, _ = norms("Ai sensi del comma 1, le servitù non apparenti non possono acquistarsi per usucapione.")
    assert out.startswith("Ai sensi del comma 1 dell'art. 1061 c.c. (Servitù non apparenti), le servitù")


def test_the_article_cited_later_in_the_paragraph_wins():
    out, _ = norms("Il titolare agisce in giudizio. Lo prevede l'art. 1079 c.c.")
    assert out.startswith("L'art. 1079 c.c. (Accertamento della servitù e altri provvedimenti di tutela): Il titolare")


def test_a_paragraph_opening_with_its_article_a_ruling_or_a_list_stays():
    for body in ("L'art. 1061 c.c. esclude le servitù non apparenti.", "La Cass. n. 123/2020 afferma il principio generale.",
                 "- le servitù non apparenti non si usucapiscono"):
        assert norms(body) == (body, [])


def test_a_paragraph_under_its_article_s_heading_stays():
    body = "**Art. 1061 c.c.**\n\nLe servitù non apparenti non si acquistano."
    assert norms(body) == (body, [])


def test_a_paragraph_matching_no_article_joins_the_one_before():
    out, log = norms("L'art. 1061 c.c. esclude le servitù non apparenti.\n\nIl punto è discusso tra gli interpreti del diritto.")
    assert out == "L'art. 1061 c.c. esclude le servitù non apparenti. Il punto è discusso tra gli interpreti del diritto."
    assert log[0]["kind"] == "headless paragraph joined to the one before"


def test_other_sections_are_not_touched():
    text = "**4. Applicazione al caso**\n\nesclude che le servitù non apparenti."
    assert text_tidy.headed_norms(text, ROWS) == (text, [])


# --- 2 ---------------------------------------------------------------------------------------------------------------

RULES = {r["id"]: r for r in fs.load_rules()}
SPLIT = [{"asset": "fondo Alfa", "owner": "Mevia", "ended_by": "divisione"}, {"asset": "fondo Alfa nord", "owner": "Livia"},
         {"asset": "fondo Alfa sud", "owner": "Ottavia"}]


@pytest.mark.parametrize("rule, words", [
    ("destinazione_padre_famiglia", ("art. 1061", "art. 1062", "alla separazione dei fondi", "non per effetto dell'uso")),
    ("usucapione_servitu_apparente", ("usucapione ventennale", "1158", "termini calcolati")),
    ("fondo_intercluso", ("in via subordinata", "art. 1051", "art. 1053", "art. 1054", "{altro_contraente}")),
])
def test_the_conclusions(rule, words):
    assert all(w in RULES[rule]["conclusion"] for w in words)


def test_the_other_contractor_comes_from_the_split():
    f = facts(property_history=SPLIT, parties=[{"name": "Livia", "role": "cliente"}])
    f["split_into_parts"] = fs.splits(SPLIT)
    assert "l'altro contraente è Ottavia, condividente" in fs.other_contractor(f)
    sold = [{**SPLIT[0], "ended_by": "compravendita"}] + SPLIT[1:]
    f = facts(property_history=sold, split_into_parts=fs.splits(sold), parties=[{"name": "Livia", "role": "cliente"}])
    assert fs.other_contractor(f).startswith("l'altro contraente è Mevia, che ha alienato")
    assert "da individuare" in fs.other_contractor(facts())


def test_a_conclusion_reaches_the_writer_filled_in():
    f = facts(events=[{"what": "il fondo è intercluso", "date": "", "who": "", "kind": ""}])
    signals = fs.signals_from(f, rules=True)
    assert "art. 1054" in fs.conclusions_block(signals) and "{altro" not in fs.conclusions_block(signals)
    analysis = {"fact_signals": signals}
    assert "art. 1054" in ptw._conclusions(analysis)


CONTEXT = {"conclusions": [{"id": "usucapione_servitu_apparente", "conclusion_subject": "usucapi", "conclusion_unless": "abbreviat|\\b1159\\b"},
                           {"id": "fondo_intercluso", "conclusion_subject": "passaggio coattiv", "conclusion_unless": "subordinat"}]}


@pytest.mark.parametrize("sentence", [
    "L'usucapione non è applicabile al caso di specie.", "L'usucapione della servitù è esclusa dai fatti.",
    "Il passaggio coattivo non spetta a Leonardo.",
])
def test_a_sentence_denying_a_fired_conclusion_goes(sentence):
    assert "denies the conclusion" in ptw._wrong_statement(sentence, CONTEXT)


@pytest.mark.parametrize("sentence", [
    "L'usucapione abbreviata non è applicabile per mancanza di un titolo trascritto.",
    "La controparte potrebbe sostenere che l'usucapione è esclusa.",
    "L'usucapione sarebbe esclusa se il passaggio fosse avvenuto con tolleranza.",
    "L'usucapione non è esclusa dal cancello.",
    "Il passaggio coattivo, in via subordinata, non spetta se il diritto è già acquisito.",
])
def test_an_objection_a_condition_or_another_institution_stays(sentence):
    assert ptw._wrong_statement(sentence, CONTEXT) == ""


# --- 3 ---------------------------------------------------------------------------------------------------------------

USE = [{"who": "Leonardo", "asset": "stradella", "since": "1990", "until": "marzo 2016", "how": "pacifico", "use": "passaggio"}]
WAY = facts(possession=USE, events=[CLOSED], present_date="")


def test_the_real_actions_have_no_prescription():
    terms = {t["id"]: t for t in fs.signals_from(facts(possession=USE, events=[CLOSED, {"what": "contesta la servitù"}]), rules=True)["terms"]}
    for id, ref in (("azione_negatoria", "949"), ("azione_rivendicazione", "948")):
        assert terms[id]["status"] == "non soggetta a prescrizione" and ref in terms[id]["ref"]


def test_no_real_action_in_a_case_about_something_else():
    assert fs.no_prescription_terms(facts(events=[{"what": "tamponamento in autostrada"}]), []) == []


def test_non_use_runs_twenty_years_from_the_last_use():
    term = next(t for t in fs.compute_terms(WAY) if t["id"] == "non_uso_servitu")
    assert term["ref"] == "art. 1073 c.c." and term["length"] == "20 anni" and term["from"]["date"] == "marzo 2016"


def test_a_road_accident_has_no_non_use_term():
    assert not any(t["id"] == "non_uso_servitu" for t in fs.compute_terms(facts(events=[{"what": "incidente sulla strada", "date": "2020"}])))


TERMS = {"terms": [{"id": "azione_confessoria", "ref": "art. 1079 c.c.", "length": "", "status": "non soggetta a prescrizione"},
                   {"id": "non_uso_servitu", "ref": "art. 1073 c.c.", "length": "20 anni", "status": "DA VERIFICARE"},
                   {"id": "possessoria", "ref": "artt. 1168 e 1170 c.c.", "length": "1 anno", "status": "DECORSO"}]}


@pytest.mark.parametrize("sentence", ["L'azione confessoria si prescrive in 10 anni dalla chiusura.",
                                      "La confessoria va proposta entro il termine di dieci anni."])
def test_another_term_for_an_action_with_an_entry_goes(sentence):
    assert "azione_confessoria" in ptw._wrong_statement(sentence, TERMS)


@pytest.mark.parametrize("sentence", ["L'azione confessoria non si prescrive, mentre la servitù si estingue in 20 anni di non uso.",
                                      "La confessoria tutela il passaggio esercitato da 34 anni."])
def test_the_computed_term_or_a_duration_that_is_no_term_stays(sentence):
    assert ptw._wrong_statement(sentence, TERMS) == ""


def test_a_duration_of_the_case_stays():
    context = {**TERMS, "case_durations": [(34, "ann")]}
    assert ptw._wrong_statement("La confessoria, entro il termine, tutela l'uso di 34 anni.", context) == ""


def test_the_usucapione_s_twenty_years_are_not_the_non_use_term():
    assert ptw._wrong_statement("L'usucapione ventennale è maturata.", TERMS) == ""


def test_a_term_with_no_length_counts_as_written_by_its_article(monkeypatch):
    monkeypatch.setenv("FACT_REGIMES_ENABLED", "true")
    use = ptw.material_use({"fact_signals": {"terms": [{"id": "azione_confessoria", "ref": "art. 1079 c.c.", "length": ""}]}},
                           "La confessoria (art. 1079 c.c.) non si prescrive.")
    assert use["termini"]["written"] == ["azione_confessoria"]


# --- 4 ---------------------------------------------------------------------------------------------------------------

def test_an_ongoing_obstruction_brings_both_tools_with_their_reasons(monkeypatch):
    monkeypatch.setenv("FACT_REGIMES_ENABLED", "true")
    signals = fs.signals_from(facts(events=[CLOSED]), rules=True)
    assert {"provvedimento_urgenza", "accertamento_preventivo"} <= {t["id"] for t in signals["tools"]}
    block = fs.tools_block(signals)
    assert "art. 700 c.p.c." in block and "ancora in atto" in block
    assert "art. 696 c.p.c." in block and "lo stato dei luoghi va documentato" in block


def test_a_reopened_gate_brings_no_tool(monkeypatch):
    monkeypatch.setenv("FACT_REGIMES_ENABLED", "true")
    done = facts(events=[CLOSED, {"what": "il cancello viene riaperto", "date": "", "who": "", "kind": ""}])
    assert "ostruzione_in_corso" not in [r["id"] for r in fs.fire_rules(done, [])]


def test_a_tools_only_rule_adds_no_central_issue():
    assert fs.central_issues(fs.fire_rules(facts(events=[CLOSED]), [])) == []


def test_a_rule_with_an_unknown_tool_is_refused(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- {id: x, area: civile, when: [{fact: {pattern: 'a'}}], institution: [], ranges: [], codice: c.c., why: y, "
                   "status: experiment, tools: [{id: nessuno, why: z}]}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="a tool is"):
        fs.load_rules(str(bad))


# --- 5 ---------------------------------------------------------------------------------------------------------------

LAND = [{"asset": "particella 7", "owner": "Tizio"}, {"asset": "particella 8", "owner": "Caio"},
        {"asset": "particella 9", "owner": "Leonardo"}, {"asset": "particella 12", "owner": "Mario"}]
PARTIES = [{"name": "Leonardo", "role": "cliente"}]


def test_the_author_and_every_owner_of_the_land_crossed():
    f = facts(events=[CLOSED], property_history=LAND, parties=PARTIES,
              works_or_signs=[{"what": "stradella", "crosses": "particella 7, particella 8 e particella 9"}])
    sued = fs.parties_to_sue(f)
    assert [p["name"] for p in sued] == ["Sempronia", "Tizio", "Caio"]
    assert "autore dell'atto" in sued[0]["why"] and "particella 7" in sued[1]["why"]


def test_the_owners_are_read_where_the_road_is_described_too():
    f = facts(events=[CLOSED], property_history=LAND, parties=PARTIES,
              access=[{"asset": "fondo", "how_reached": "attraverso i fondi di Tizio e di Caio"}])
    assert [p["name"] for p in fs.parties_to_sue(f)] == ["Sempronia", "Tizio", "Caio"]


def test_a_former_owner_is_not_sued():
    history = [{"asset": "particella 7", "owner": "Tizio"}, {"asset": "particella 7", "owner": "Annibale"}]
    f = facts(events=[CLOSED], property_history=history, works_or_signs=[{"what": "stradella", "crosses": "particella 7"}])
    assert [p["name"] for p in fs.parties_to_sue(f)] == ["Sempronia", "Annibale"]


def test_no_obstruction_no_list_and_a_common_noun_is_no_party():
    assert fs.parties_to_sue(facts(property_history=LAND)) == []
    event = {**CLOSED, "who": "il vicino"}
    assert fs.parties_to_sue(facts(events=[event])) == []


def test_the_writer_gets_the_list_and_never_eventuali(monkeypatch):
    f = facts(events=[CLOSED], property_history=LAND, parties=PARTIES, works_or_signs=[{"what": "stradella", "crosses": "particella 7"}])
    signals = fs.signals_from(f, rules=True)
    assert "- Sempronia: autore dell'atto" in ptw._defendants({"fact_signals": signals})
    context = ptw.statement_context({"fact_signals": signals})
    assert context["defendants"]
    assert ptw._wrong_statement("Vanno convenuti eventuali altri proprietari dei fondi.", context) == "the parties to sue are named in the facts"
    assert ptw._wrong_statement("Vanno convenuti eventuali altri proprietari dei fondi.", {"defendants": False}) == ""


def test_the_section_jobs_carry_the_fixed_text(monkeypatch):
    analysis = {"fact_signals": {"fired": [{"conclusion": "Punto fermo di diritto."}], "defendants": [{"name": "Sempronia", "why": "autrice"}]}}
    plan = {"parte_assistita": "Leonardo", "posizione": "", "obiettivo": "", "questione_centrale": "", "questioni": [],
            "risposte": [], "subordinate": [], "punti_deboli": []}
    jobs = ptw.section_jobs(plan, "Fatti.", "Richiesta.", analysis, [], {}, "")
    assert all("Punto fermo di diritto." in job[2] for job in jobs)
    procedural = next(job for job in jobs if job[0] == ptw.SECTION_TITLES[4])
    assert "Sempronia: autrice" in procedural[2]
