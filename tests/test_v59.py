"""v59: the output lint of the transcripts (scripts/answer_lint.py) and the code paths its rules point at -- the requisite
filter, the table's own articles, the terms the listed topics carry, the procedural section, the final pass."""

import os
import sys
from pathlib import Path

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import answer_lint  # noqa: E402
import render_replay  # noqa: E402
from src.rag import interaction_copy as C  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import render_computed, rule_quote, topic_step  # noqa: E402

ART_321 = ("Le pene stabilite nel primo comma dell'articolo 318, nell'articolo 319, nell'articolo 319-bis, nell'art. "
           "319-ter, e nell'articolo 320 in relazione alle suddette ipotesi degli articoli 318 e 319, si applicano anche a "
           "chi dà o promette al pubblico ufficiale o all'incaricato di un pubblico servizio il denaro od altra utilità.")
ART_322 = ("Chiunque offre o promette denaro od altra utilità non dovuti ad un pubblico ufficiale o ad un incaricato di un "
           "pubblico servizio, per l'esercizio delle sue funzioni o dei suoi poteri, soggiace, qualora l'offerta o la "
           "promessa non sia accettata, alla pena stabilita nel primo comma dell'articolo 318, ridotta di un terzo.")
ART_585 = ("Nei casi previsti dagli articoli 582, 583, 583-bis, 583-quinquies e 584, la pena è aumentata da un terzo alla "
           "metà, se concorre alcuna delle circostanze aggravanti prevedute dall'articolo 576.")


# --- the requisites (penalty_requisite) --------------------------------------------------------------------------------

def test_a_penalty_a_consequence_or_a_bare_reference_is_never_a_requisite():
    assert not rule_quote.requisite_kept("ridotta di un terzo")                                     # v58-h7 322
    assert not rule_quote.requisite_kept("alla pena stabilita nel primo comma dell'articolo 318")
    assert not rule_quote.requisite_kept("tale statuizione costituisce titolo esecutivo")           # v58-h10 114 c.p.a.
    assert not rule_quote.requisite_kept("è stabilita in misura pari agli interessi legali")
    assert not rule_quote.requisite_kept("nell'articolo 319")                                       # v58-h7 321
    assert not rule_quote.requisite_kept("e nell'articolo 320 in relazione alle suddette ipotesi degli articoli 318 e 319")
    assert rule_quote.requisite_kept("il fondo è circondato da fondi altrui")


def test_an_offence_extending_others_penalties_has_no_element_and_a_condition_in_its_penalty_clause_is_one():
    assert rule_quote.offence_elements(ART_321) == []                                               # v58-h7: six lines
    assert rule_quote.offence_elements(ART_322) == [
        "offre o promette denaro od altra utilità non dovuti ad un pubblico ufficiale o ad un incaricato di un pubblico "
        "servizio", "per l'esercizio delle sue funzioni o dei suoi poteri",
        "qualora l'offerta o la promessa non sia accettata"]
    assert rule_quote.refers_penalties(ART_321) and rule_quote.refers_penalties(ART_585)
    assert not rule_quote.refers_penalties(ART_322)


def test_an_article_referring_penalties_gets_its_comma_line_never_a_bare_heading():
    rows = [{"_article": ("321", "c.p."), "s": {"plain_text": f"(co. 1) {ART_321}", "titolo_articolo": "Pene per il corruttore"}},
            {"_article": ("585", "c.p."), "s": {"plain_text": f"(co. 1) {ART_585}", "titolo_articolo": "Circostanze aggravanti"}}]
    rules = rule_quote.article_rules({}, rows, {}, [("321", "c.p."), ("585", "c.p.")])
    assert all(len(r["referring"]) == 1 for r in rules) and all(not r["requisites"] for r in rules)
    blocks = flow.norm_blocks(rules, {}, rows, [("321", "c.p."), ("585", "c.p.")])
    assert blocks[0].startswith("**Art. 321 c.p. — Pene per il corruttore**\n- ") and blocks[0].endswith(": da provare.")
    assert len(blocks[1].split("\n")) == 2


# --- the table's own articles (unearned_topic_point) -------------------------------------------------------------------

def test_a_table_merged_in_makes_all_its_articles_the_topics_own_and_names_it_at_half():
    topic = {"title": "Elemento psicologico del reato", "refs": [("43", "c.p."), ("575", "c.p."), ("589", "c.p."),
                                                                  ("42", "c.p."), ("52", "c.p."), ("53", "c.p."),
                                                                  ("55", "c.p."), ("56", "c.p.")],
             "planned": 8, "merged": [], "facts": [], "questions": []}
    table = {"title": "Legittima difesa ed eccesso colposo", "source": "tabella legittima_difesa", "facts": [],
             "merged": [], "questions": [], "own": [("52", "c.p."), ("55", "c.p."), ("575", "c.p."), ("589", "c.p.")]}
    topic_step._merge(topic, table)                                                                 # v58-p1
    assert topic["own"] == table["own"] and topic["planned"] == 8
    assert topic["refs"][:5] == [("43", "c.p."), ("52", "c.p."), ("55", "c.p."), ("575", "c.p."), ("589", "c.p.")]
    topic_step.named_by_table(topic)                                                                # 4 of 8
    assert topic["title"] == "Legittima difesa ed eccesso colposo"
    assert "Elemento psicologico del reato" in topic["merged"]


# --- the terms (gated_term) --------------------------------------------------------------------------------------------

USUCAPIONE = {"id": "usucapione_ordinaria", "label": "usucapione ordinaria di beni immobili", "ref": "art. 1158 c.c."}
POSSESSORIA = {"id": "possessoria", "label": "azione possessoria (spoglio o turbativa)", "ref": "artt. 1168 e 1170 c.c."}
EXTRA = {"id": "prescrizione_extracontrattuale", "label": "prescrizione del risarcimento del danno extracontrattuale",
         "ref": "art. 2947 c.c."}


def test_a_term_is_shown_only_when_a_listed_topic_of_its_matter_carries_it():
    l3 = [{"title": "Danno cagionato da cosa in custodia", "articles": ["2051 c.c."]},
          {"title": "Attribuzioni dell'amministratore", "articles": ["1130 c.c.", "1117 c.c.", "1118 c.c."]}]
    assert not topic_step.carries_term(USUCAPIONE, l3)                                              # v58-l3
    h3 = [{"title": "Passaggio coattivo", "articles": ["1051 c.c.", "1052 c.c."]},
          {"title": "Usucapione ordinaria di beni immobili", "articles": ["1158 c.c."], "source": "termine usucapione"}]
    assert topic_step.carries_term(POSSESSORIA, h3) and topic_step.carries_term(USUCAPIONE, h3)     # diritti reali
    h10 = [{"title": "Prescrizione del risarcimento del danno extracontrattuale", "articles": ["2947 c.c."],
            "source": "termine prescrizione_extracontrattuale"},
           {"title": "Ottemperanza ed elusione del giudicato", "articles": ["112 c.p.a.", "114 c.p.a."]}]
    assert not topic_step.carries_term(EXTRA, h10)                                                  # v58-h10
    assert [t["title"] for t in topic_step.term_topics_out(h10, [EXTRA])] == ["Ottemperanza ed elusione del giudicato"]
    alone = h10[:1]
    assert topic_step.carries_term(EXTRA, alone) and topic_step.term_topics_out(alone, [EXTRA]) == alone


def test_the_date_of_the_fact_is_asked_for_the_querela_only_in_a_criminal_case():
    question = {"family": "data_fatto", "text": C.QUESTIONS["data_fatto"]}
    admin = [{"title": "Ottemperanza", "articles": ["112 c.p.a."]}]
    assert flow.question_wording(question, admin) == C.QUESTIONS["data_fatto_civile"]               # v58-h10 q.3
    assert flow.question_wording(question, [{"title": "Truffa", "articles": ["640 c.p."]}]) == C.QUESTIONS["data_fatto"]


# --- the procedural section (missing_section, duplicate_default) -------------------------------------------------------

def test_the_default_delitti_are_one_line_and_a_chosen_delitto_makes_the_section_whatever_the_frame():
    lines = render_computed.procedure_lines([], [("320", "Corruzione di persona incaricata di un pubblico servizio"),
                                                 ("318", "Corruzione per l'esercizio della funzione"),
                                                 ("640", "Truffa")], dated=False)
    assert lines[0].startswith("- Truffa (art. 640 c.p.): procedibile a querela")                    # its table entry
    assert lines[1] == ("- Corruzione di persona incaricata di un pubblico servizio (art. 320 c.p.); Corruzione per "
                        "l'esercizio della funzione (art. 318 c.p.): procedibili d'ufficio; competenza: tribunale.")
    assert len([l for l in lines if "d'ufficio; competenza: tribunale" in l]) == 1                   # v58-h7: five
    body = render_computed.criminal_procedure({"frame": {"areas": ["contratti"]}, "offences": [("640", "Truffa")]})
    assert "Truffa (art. 640 c.p.)" in body                                                          # v58-p2: no §4


def test_mediation_follows_the_chosen_topics_matter():
    assert render_computed.mediation_of({"topic_refs": [("1158", "c.c.")]})["subject"] == "diritti reali"   # v55-n9
    assert render_computed.mediation_of({"topic_refs": [("1130", "c.c.")]})["subject"] == "condominio"
    assert not render_computed.mediation_of({"topic_refs": [("2043", "c.c.")]}).get("required")


# --- the final pass (bare_heading, orphan_paragraph, repeated_marker, dangling_reference, scaffold_bullet) -------------

def test_a_bare_heading_goes_to_the_closing_line_with_those_already_there():
    norms = ("**Art. 628 c.p. — Rapina**\n- testo, requisito «mediante violenza»: da provare.\n\nTesto.\n\n"
             "**Art. 625 c.p. — Circostanze aggravanti**\n\n**Art. 52 c.p. — Difesa legittima**\n\n"
             "Altre norme del tema non trattate: art. 624-bis c.p.")
    out = flow.norms_final(norms)                                                                    # v58-h5
    assert "**Art. 625" not in out and "**Art. 52" not in out and "**Art. 628 c.p. — Rapina**" in out
    assert out.endswith("Altre norme del tema non trattate: art. 625 c.p., art. 52 c.p., art. 624-bis c.p.")


def test_the_final_pass_repairs_cut_lines_markers_references_and_bullets():
    draft = ("**1. Risposta e strategia**\n\nPrimo paragrafo.\n\n (relativo all'uso dell'arma) costituisce un vizio. "
             "La difesa ricorrerà.\n\n La responsabilità di Caio non è piena.\n\nSi veda (cfr. sezione *Profili "
             "processuali e termini*).\n\n**2. Norme applicate ai fatti**\n\n**Art. 1 c.c. — X**\n\nTesto.\n\n"
             "**3. Altri profili**\n\nPer le norme seguenti …\n\n- Tale consapevolezza (art. 321 c.p.\n-), non basta.\n"
             "- [3] art. 589 c.p.\n- Inoltre, la mancanza di prove dirette dell'art. 628 c.p.\n\n"
             "**4. Profili processuali**\n\n**Termini**\n\n- x.\n\n**5. Conclusioni**\n\n"
             "Se le opere risultano esistenti alla separazione, Leonardo vince. Se le opere risultano esistenti alla "
             "separazione, Questi fatti contano (da verificare: data) per lui.")
    out = flow.answer_final(draft)
    assert "(relativo" not in out and "La difesa ricorrerà." in out                                  # v58-h5 §1
    assert "\nLa responsabilità di Caio non è piena." in out                                         # v58-h7 §1
    # v60f (P9): the name is no longer repaired after writing -- the writer is told the answer's sections (test_v60f)
    assert "(cfr. sezione *Profili processuali e termini*)" in out                                   # v58-h10 §1
    assert "(art. 321 c.p.), non basta." in out and "[3] art. 589" not in out                        # v58-h7 §3, v55
    assert "Inoltre, la mancanza" not in out                                                         # v58-h5 §3
    assert out.count("Se le opere risultano esistenti alla separazione") == 1 and "Questi fatti contano per lui " \
        "(da verificare: data)." in out and "separazione, Leonardo vince." in out                    # v57-h3 §5


def test_a_comma_its_article_does_not_have_moves_even_inside_a_paragraph():
    refs = [("21-septies", "l241"), ("31", "c.p.a.")]
    by_key = {flow._key(refs[0]): {"ref": refs[0], "commi": [(1, "E' nullo il provvedimento"), (2, "Le questioni")]},
              flow._key(refs[1]): {"ref": refs[1], "commi": [(1, "Decorsi i termini"), (4, "La domanda volta "
                                                                    "all'accertamento delle nullità si propone entro")]}}
    placed = {1: ["«co. 1» E' nullo il provvedimento.\n«co. 4» La domanda volta all'accertamento delle nullità si "
                  "propone entro centottanta giorni."]}
    log = []
    out = flow.commi_moved(placed, refs, by_key, log)                                               # v56-h10
    assert out[1] == ["«co. 1» E' nullo il provvedimento."] and out[2][0].startswith("«co. 4» La domanda")
    assert log == [{"from": "21-septies l241", "comma": 4, "to": "31 c.p.a."}]


# --- the lint and the replay -------------------------------------------------------------------------------------------

TRANSCRIPT = """# Interactive flow, simulated (--interactive-sim)

## x1

### round 1 (round1, 1.0s) — user: (case)

**Argomenti individuati**

1. **Corruzione di persona incaricata di un pubblico servizio** (artt. 320, 321 c.p.).

Ecco i fatti e i termini che ho ricavato dal caso:

Parti: Tizio (incaricato di pubblico servizio); Caio (privato).

Dimmi su quale argomento vuoi che mi concentri (il numero, oppure descrivimi l'angolo che ti interessa), rispondi alle domande che puoi, oppure scrivi "procedi" e parto dalla tesi principale.

### round 2 (round2, 0.0s) — user: 1

Riassumo come imposterò il parere. Tratterò: Corruzione di persona incaricata di un pubblico servizio. Va bene così, o vuoi correggere qualcosa?

### round 3 (answer, 1.0s) — user: procedi

**1. Risposta e strategia**

Caio risponde.

 La responsabilità di Caio non è piena (cfr. sezione *Profili processuali e termini*).

**2. Norme applicate ai fatti**

**Art. 321 c.p. — Pene per il corruttore**
- testo, requisito «Le pene stabilite nel primo comma dell'articolo 318»: da provare.
- testo, requisito «si applicano anche a chi dà o promette denaro»: da provare.

**Art. 320 c.p. — Corruzione di persona incaricata di un pubblico servizio**

**3. Profili processuali**

**Procedibilità e competenza**

- Corruzione (art. 320 c.p.): procedibile d'ufficio; competenza: tribunale.
- Pene per il corruttore (art. 321 c.p.): procedibile d'ufficio; competenza: tribunale.

**4. Conclusioni**

Esito incerto.
"""


def test_the_lint_finds_each_defect_and_the_replay_leaves_none(tmp_path):
    path = tmp_path / "v58-x1.md"
    path.write_text(TRANSCRIPT, encoding="utf-8")
    rules = sorted({f[3] for f in answer_lint.lint_file(str(path))})
    assert rules == ["bare_heading", "dangling_reference", "duplicate_default", "no_client", "orphan_paragraph",
                     "penalty_requisite"]
    replayed = render_replay.replay_file(str(path))
    # v60f (P9): an old transcript's wrong section name stays (the repair pass is gone; a new answer's writer is told
    # the real names), so the lint still reports it -- the only finding left
    assert [f[3] for f in answer_lint.lint_file(str(path), replayed)] == ["dangling_reference"]
    case = replayed["x1"]
    assert "Chi assisti in questo caso: Tizio o Caio?" in case["round1"]
    assert "- testo: da provare." in case["answer"] and "Altre norme del tema non trattate: art. 320 c.p." in case["answer"]
