"""v68: eight general rules from the review of the posture cells -- synthetic texts, no case of the sim.

1. the client's own article is a weapon: the article the client's question is about («se sussistano i presupposti per
   chiedere / recedere … ai sensi di») -- a threat only when the counterparty is the verb's subject; no marker on weapons
2. a narrative with exactly one question and no list is a quesito, an article named or not (a strategy, a course of
   action or the facts' qualification asked: still a parere)
3. the requested blocks first, in the asked order, then the body; five or more: the body short
4. a named article the writer skipped: a second writer call for it alone, never a bare heading
5. «Altri profili»: only in the code and Libro of a covered topic, or in the posture's table
6. the requirement check: every article with recorded elements, every section the writer writes
7. an institution named with an article whose rubrica does not match: the record's article instead, else no citation
8. the plain route (cenni, quesito): no article outside the retrieved sections, no case law without a ruling
"""

import inspect
import os
import sys
from pathlib import Path

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.rag import (answer_ledger, intake, issue_records, plain_grounding, posture, record_check,  # noqa: E402
                     writer_rules)
from src.rag import interaction_copy as C  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402

PROVATO, DA_PROVARE = flow.PROVATO, flow.DA_PROVARE
LEAD, NORMS, OBJECTIONS, PROCEDURE, CLOSING = flow.ANSWER_TITLES


def _rec(rubrica, *statuses, **extra):
    reqs = [{"kind": "requisito", "comma": 1, "text": f"elemento {i}", "status": st} for i, st in enumerate(statuses, 1)]
    rec = {"rubrica": rubrica, "requisites": reqs, **extra}
    rec["verdict"] = issue_records.verdict(rec)
    return rec


# --- 1. the client's own article: a weapon, no marker --------------------------------------------------------------------

@pytest.mark.parametrize("text, client, use", [
    ("Si chiede se sussistano i presupposti per chiedere la risoluzione del contratto ai sensi dell'art. 1453 c.c.",
     "Tizio", "weapon"),
    ("Tizio chiede se sussistano i presupposti per chiedere la risoluzione ai sensi dell'art. 1453 c.c.", "", "weapon"),
    ("Si chiede se sussistano i presupposti per recedere dal contratto ai sensi dell'art. 1385 c.c.", "Tizio", "weapon"),
    ("Si chiede se Caio possa chiedere la risoluzione del contratto ai sensi dell'art. 1453 c.c.", "Tizio", "threat"),
    ("La controparte chiede il risarcimento ai sensi dell'art. 2043 c.c.", "Tizio", "threat"),
    ("Tizio è stato convenuto in giudizio ai sensi dell'art. 2043 c.c.", "Tizio", "threat"),
    ("Al cliente si contesta il reato di cui all'art. 575 c.p.", "Tizio", "threat"),
])
def test_the_article_the_question_is_about_is_a_weapon_unless_the_counterparty_is_the_subject(text, client, use):
    (named,) = intake.named_articles(text, client)
    assert named["use"] == use


def test_a_weapon_is_never_marked_da_valutare_and_never_rendered_as_a_threat():
    weapon = _rec("Risoluzione per inadempimento", DA_PROVARE, DA_PROVARE)
    threat = _rec("Risarcimento per fatto illecito", DA_PROVARE)
    state = {"records": {"1453 c.c.": weapon, "2043 c.c.": threat},
             "intake": {"named_articles": [{"article": "1453 c.c.", "verb": "chiedere", "use": "weapon"},
                                           {"article": "2043 c.c.", "verb": "chiede", "use": "threat"}]}}
    posture.mark_sides(state, {"role": "attore", "family": "civile"})
    assert (weapon["side"], threat["side"]) == ("weapon", "threat")
    assert not flow.status_sentence("1453 c.c.", weapon, state).startswith("Per la difesa")
    for role in ("attore", "convenuto"):
        state["posture"] = {"role": role}
        draft = (f"**1. {LEAD}**\n\nSi chiede la risoluzione (art. 1453 c.c.). La controparte invoca l'art. 2043 c.c.\n\n"
                 f"**2. {NORMS}**\n\nX.")
        lead = dict(answer_ledger.sections_of(flow.asserted_only(draft, state)))[LEAD]
        assert "(art. 1453 c.c.)." in lead and "(da valutare: art. 1453" not in lead
        assert "(da valutare: art. 2043 c.c.)" in lead
    assert "{articles}" in writer_rules.WEAPON_TO_PROVE_RULE and "«da valutare»" in writer_rules.WEAPON_TO_PROVE_RULE


# --- 2. one question, no list: a quesito ----------------------------------------------------------------------------------

@pytest.mark.parametrize("text, deliverable", [
    ("Tizio ha citato in giudizio Caio per il pagamento di 8.000 euro per lavori mai pagati. La causa spetta al giudice "
     "di pace o al tribunale?", "quesito"),                                                   # no article: a quesito
    ("Il cane di Tizio ha morso Caio nel giardino. Chi risponde del danno?", "quesito"),
    ("Laura lavora da dodici anni per la società Alfa. Il datore di lavoro l'ha licenziata per giusta causa senza "
     "alcuna contestazione scritta. Come impugnare il licenziamento?", "parere"),                                                                               # a course of action
    ("Mevio acquista un'auto usata da un privato. La paga con un assegno scoperto e mente sulla sua identità. Quali "
     "reati sono configurabili?", "parere"),                                                                               # the facts' qualification
    ("Tizio è stato citato in giudizio da Caio per 20.000 euro. Il mio cliente Tizio non si è costituito. Cosa può "
     "eccepire?", "parere"),                                                                  # a defence asked
])
def test_one_question_and_no_list_is_a_quesito_whether_or_not_it_names_an_article(text, deliverable):
    got = intake.detect(text, model=False)
    assert (got["deliverable"] == "quesito") is (deliverable == "quesito"), got["why"]   # the others: never a quesito
    if deliverable == "quesito":
        assert got["route"] == intake.RAG and "una sola domanda di diritto" in got["why"]["deliverable"]


def test_two_questions_or_a_list_are_never_the_single_quesito():
    assert intake.point_of_law_question("Tizio ha venduto a Caio un'auto. Chi paga? E quando?") == ""
    assert intake.point_of_law_question("Tizio ha venduto a Caio un'auto. Si chiede: 1) chi paga; 2) entro quando "
                                        "agire?") == ""


# --- 3. the requested blocks first; five or more, the body short ----------------------------------------------------------

BODY = (f"**1. {LEAD}**\n\nLa risoluzione è fondata. Tizio può agire per la risoluzione del contratto entro dieci anni "
        f"dalla scadenza del termine.\n\n**2. {NORMS}**\n\n**Art. 1453 c.c. — Risoluzione**\n\nIl contratto si "
        f"risolve.\n\n**3. {OBJECTIONS}**\n\nObiezione.\n\n**4. {PROCEDURE}**\n\n- Termine.\n\n**5. {CLOSING}**\n\nEsito.")


def _write(jobs):
    return [(t, f"Tizio può agire per la risoluzione del contratto entro dieci anni dalla scadenza del termine. "
                f"Risposta a {t.split(':', 1)[1]}.") for t, *_ in jobs]


def test_the_questions_come_first_and_the_body_after_them_wherever_the_parere_is_asked():
    blocks = [{"kind": "parere", "subject": "parere"}, {"kind": "quesito", "subject": "se Tizio possa agire"},
              {"kind": "quesito", "subject": "entro quale termine"}]                           # the penal exam's order
    out = flow.requested_sections({}, BODY, blocks, _write, {}, "Il caso.")
    titles = [t for t, _ in answer_ledger.sections_of(out)]
    assert titles == ["Se Tizio possa agire", "Entro quale termine", LEAD, NORMS, OBJECTIONS, PROCEDURE, CLOSING]


def test_five_questions_and_the_body_is_a_summary_the_norms_and_the_procedure_with_nothing_said_twice():
    blocks = [{"kind": "parere"}] + [{"kind": "quesito", "subject": f"quesito numero {n}"} for n in range(1, 6)]
    out = flow.requested_sections({}, BODY, blocks, _write, {}, "Il caso.")
    sections = answer_ledger.sections_of(out)
    titles = [t for t, _ in sections]
    assert titles[5:] == [C.SUMMARY_SECTION, C.NORMS_SHORT_SECTION, PROCEDURE] and len(titles) == 8
    summary = dict(sections)[C.SUMMARY_SECTION]
    assert "La risoluzione è fondata." in summary and "entro dieci anni" not in summary       # said by the questions
    assert "**Art. 1453 c.c. — Risoluzione**" in dict(sections)[C.NORMS_SHORT_SECTION]
    state = {"intake": {"requested_blocks": blocks}}
    assert flow.collapsed(state) and not flow.collapsed({"intake": {"requested_blocks": blocks[:5]}})
    assert "{questions}" in writer_rules.COLLAPSED_LEAD_NOTE


# --- 4. a named article the writer skipped: a call for it alone -----------------------------------------------------------

def _named_state():
    return {"records": {"1385 c.c.": {**_rec("Caparra confirmatoria", DA_PROVARE), "named_due": True},
                        "1453 c.c.": _rec("Risoluzione per inadempimento", PROVATO)},
            "main_anchor": "1453 c.c."}


DRAFT_2 = f"**2. {NORMS}**\n\n**Art. 1453 c.c. — Risoluzione per inadempimento**\n\nIl contratto si risolve.\n\n**5. {CLOSING}**\n\nX."


def test_a_named_article_left_empty_gets_its_own_call_and_never_a_bare_heading():
    asked = []
    text, _ = flow.ledger_out(DRAFT_2, _named_state(), retry=lambda label: asked.append(label) or "La caparra è stata "
                                                                                                  "versata da Tizio.")
    norms = dict(answer_ledger.sections_of(text))[NORMS]
    assert asked == ["1385 c.c."] and "**Art. 1385 c.c. — Caparra confirmatoria**" in norms
    assert "La caparra è stata versata da Tizio." in norms
    log = {}
    text, rows = flow.ledger_out(DRAFT_2, _named_state(), log, retry=lambda label: "")
    assert "1385" not in dict(answer_ledger.sections_of(text))[NORMS]                    # empty again: a citation row
    assert log["named_retry"] == {"1385 c.c.": "empty"} and "1385 c.c." in log["bare"]
    assert any(r["article"] == "1385 c.c." and not r.get("in_body") for r in rows if r.get("requisite") is None)


def test_the_retry_is_one_job_on_the_article_alone():
    jobs = []

    def write(js):
        jobs.extend(js)
        return [(js[0][0], "[1] La caparra è stata versata.")]
    out = flow.named_retry(_named_state(), "1385 c.c.", {"1385 c.c.": "Se al momento della conclusione del contratto…"},
                           write)
    assert out == "La caparra è stata versata." and len(jobs) == 1
    title, instruction, material, _ = jobs[0]
    assert title == NORMS and "Tratta solo l'art. 1385 c.c." in instruction
    assert "1. art. 1385 c.c." in material and "Se al momento della conclusione" in material and "1453" not in material


# --- 5. «Altri profili»: the Libro of a covered topic, or the posture's table ---------------------------------------------

def _topic(title, articles, best=2):
    return {"title": title, "articles": articles, "source": "rubrica", "best_grade": best, "graded": 1}


def test_other_profiles_stay_in_the_book_of_a_covered_topic_or_in_the_posture_table():
    topics = [_topic("Risoluzione", ["1453 c.c."]), _topic("Mezzadria", ["2141 c.c."], 3),
              _topic("Soccida", ["2170 c.c."], 3), _topic("Promessa di matrimonio", ["79 c.c."], 3),
              _topic("Rito del lavoro", ["418 c.p.c."], 3), _topic("Procedimento sommario", ["702-bis c.p.c."], 3),
              _topic("Caparra", ["1385 c.c."]), _topic("Comparsa", ["167 c.p.c."])]
    state = {"topics": topics, "area_gate": {"families": []},
             "posture": {"entry": {"articles": ["166 c.p.c.", "167 c.p.c."]}}}
    lines = flow.single_profiles(state, [1], {})
    assert lines == ["- Caparra (artt. 1385 c.c.)", "- Comparsa (artt. 167 c.p.c.)"]
    assert flow.profile_book("2141 c.c.") == ("c.c.", 13) and flow.profile_book("702-bis c.p.c.") == ("c.p.c.", 4)  # v76:
    assert flow.profile_book("1470 c.c.") != flow.profile_book("2028 c.c.")     # Libro IV by Titolo


# --- 6. the requirement check: every recorded article, every section -------------------------------------------------------

def test_every_article_with_recorded_elements_is_checked_in_every_section_the_writer_writes():
    weapon = {**_rec("Risoluzione per inadempimento", PROVATO), "side": "weapon"}
    weapon["requisites"][0]["text"] = "inadempimento di non scarsa importanza della controparte"
    state = {"records": {"1453 c.c.": weapon, "40 c.p.": {"rubrica": "Rapporto di causalità", "requisites": [],
                                                          "applicazione": "cenni"}}}
    context = flow.requirement_context(state)
    assert list(context) == ["1453 c.c."]                                   # no element counted: outside the check
    assert NORMS in flow.REQUIREMENT_TITLES and PROCEDURE not in flow.REQUIREMENT_TITLES
    ctx = {"asserting": list(flow.REQUIREMENT_TITLES), "threats": context, "allowed": {"1453 c.c."}, "parties": []}
    sentence = "L'art. 1453 c.c. richiede una richiesta formale di risoluzione inviata per raccomandata."
    out, log = record_check.check([(NORMS, f"\n\n{sentence} Il contratto si risolve.\n\n")], ctx)
    assert sentence not in out[0][1] and log["rules"] == {"requisito_non_registrato": 1}
    kept = "L'art. 1453 c.c. richiede un inadempimento di non scarsa importanza della controparte."
    out, _ = record_check.check([(NORMS, f"\n\n{kept}\n\n")], ctx)
    assert kept in out[0][1]
    required = []
    text = flow.requirement_out(f"{sentence} Tizio può agire.", state, [], required)
    assert text == "Tizio può agire." and len(required) == 1                   # a quesito's section too


# --- 7. an institution with another article's number: the record's article, else no citation --------------------------------

def _ctx():
    return {"records": {"575 c.p.": _rec("Omicidio", DA_PROVARE), "582 c.p.": _rec("Lesione personale", DA_PROVARE)},
            "rubriche": {"575 c.p.": "Omicidio", "582 c.p.": "Lesione personale", "2043 c.c.": "Risarcimento per "
                         "fatto illecito"},
            "texts": {"575 c.p.": "Chiunque cagiona la morte di un uomo è punito con la reclusione.",
                      "582 c.p.": "Chiunque cagiona ad alcuno una lesione personale…",
                      "2043 c.c.": "Qualunque fatto doloso o colposo…"},
            "allowed": {"575 c.p.", "582 c.p.", "2043 c.c."}, "parties": []}


def test_an_institution_cited_with_another_articles_number_takes_the_records_article():
    out, log = record_check.check([(LEAD, "Le lesioni personali ai sensi dell'art. 575 c.p. sono procedibili a "
                                          "querela.")], _ctx())
    assert out[0][1] == "Le lesioni personali ai sensi dell'art. 582 c.p. sono procedibili a querela."
    assert log["rewritten"][0]["now_article"] == "582 c.p."
    sentence = "Tizio ha agito ai sensi dell'art. 2043 c.c. contro Caio."                 # no institution named
    out, _ = record_check.check([(LEAD, sentence)], _ctx())
    assert out[0][1] == sentence
    out, log = record_check.check([(LEAD, "La rapina impropria (art. 575 c.p.) è esclusa.")], _ctx())
    assert out[0][1] == "La rapina impropria è esclusa." and log["rewritten"][0]["now_article"] == ""


# --- 8. the plain route: grounded articles, no case law without a ruling ------------------------------------------------------

ROWS = [{"d": {"id": "cc", "name": "Codice Civile", "document_type": "primary"},
         "s": {"name": "2934.0.0", "plain_text": "Ogni diritto si estingue per prescrizione…"}}]
ANSWER = ("**Prescrizione**\n\nOgni diritto si estingue per prescrizione (art. 2934 c.c.). Il diritto al risarcimento "
          "si prescrive in cinque anni ai sensi dell'art. 2947 c.c. La giurisprudenza ha ribadito che il termine "
          "decorre dal danno.\n\n- Il termine dell'art. 2948 è di cinque anni.\n- La Cassazione ha chiarito il punto.")


def test_the_plain_answer_cites_only_retrieved_articles_and_no_case_law_without_a_ruling():
    log = {}
    out = plain_grounding.ground(ANSWER, ROWS, log)
    assert "(art. 2934 c.c.)" in out and "2947" not in out and "2948" not in out
    assert "Il diritto al risarcimento si prescrive in cinque anni." in out and "- Il termine è di cinque anni." in out
    assert "giurisprudenza" not in out.lower() and "Cassazione" not in out
    assert len(log["dropped_case_law"]) == 2 and log["rulings"] is False
    ruling = {"d": {"name": "Cass. civ., sez. III, n. 1234/2020", "document_type": "interpretation"}, "s": {"name": "x"}}
    kept = plain_grounding.ground("La giurisprudenza ha ribadito che il termine decorre dal danno.", ROWS + [ruling], {})
    assert kept.startswith("La giurisprudenza ha ribadito")
    assert plain_grounding.applies("cenni") and plain_grounding.applies("quesito") and not plain_grounding.applies("parere")


def test_the_deliverable_reaches_the_plain_answer():
    from src.rag import main
    from src.rag.nodes import synthesis
    assert "deliverable" in inspect.signature(main.run).parameters
    source = inspect.getsource(synthesis.synthesize_answer)
    assert "plain_grounding.ground(answer, data" in source and "plain_grounding.NO_RULINGS_NOTE" in source
    session = (HERE.parent / "src" / "chatbot" / "session.py").read_text(encoding="utf-8")
    assert '{"deliverable": str(intake["deliverable"])}' in session
