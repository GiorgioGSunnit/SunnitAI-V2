"""Leonardo v3 on Qwen3: the defects, from their exact strings, as general rules on invented cases."""
import importlib.util
import os
from pathlib import Path

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

from src.rag import anchored_writing, article_text, comma_material as cm, fact_signals as fs, text_tidy  # noqa: E402
import src.rag.defensive_generation as dg  # noqa: E402
import src.rag.plan_then_write as ptw  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
COMMA_1062 = ("La destinazione del padre di famiglia ha luogo quando consta, mediante qualunque genere di prova, che due fondi, "
              "attualmente divisi, sono stati posseduti dallo stesso proprietario, e che questi ha posto o lasciato le cose nello stato.")


# --- A1 ------------------------------------------------------------------------------------------------------------

def test_a_comma_is_never_shortened_inside_a_clause():
    for room in (30, 60, 100, 150):
        cut = article_text.clause_cut(COMMA_1062, room)
        kept = cut.replace(" […]", "").rstrip(",.")
        assert kept == COMMA_1062.rstrip(".") or cut.endswith("[…]") and COMMA_1062[len(kept)] in ",.;:"
    assert article_text.clause_cut("Breve.", 100) == "Breve."


def test_no_clause_end_in_reach_keeps_the_whole_comma():
    text = "Il giudice ordina la consegna del bene entro il termine fissato dalla sentenza di condanna al convenuto"
    assert article_text.clause_cut(text, 60) == text


def test_select_never_cuts_mid_clause():
    (_, kept), = cm.select([COMMA_1062], 60, "", [1])
    assert not kept.endswith("mediante […]") and not kept.endswith("mediante")


def test_the_exact_cut_quote_is_dropped_at_the_end():
    text = "La strada è visibile. Ai sensi dell'art. 1062 c.c., co. 1, la destinazione del padre di famiglia ha luogo quando consta mediante. Il fondo è diviso."
    out, log = text_tidy.tidy(text)
    assert "mediante" not in out and "Il fondo è diviso." in out and any(e["kind"] == "cut clause" for e in log)


def test_a_sentence_ending_on_a_full_word_stays():
    assert text_tidy.unreadable("La servitù si acquista per destinazione del padre di famiglia.") == ""


# --- A2 ------------------------------------------------------------------------------------------------------------

def test_the_exact_garbled_join():
    text = ("Il giudice dell'esecuzione, su ricorso dell'avente diritto, interviene dopo la notificazione., Leonardo può "
            "chiedere la misura coercitiva.")
    out, _ = text_tidy.tidy(text)
    assert "notificazione, Leonardo" in out


def test_a_removed_citation_leaves_no_stop_before_a_comma():
    out = anchored_writing.drop_citations("Il giudice interviene dopo la notificazione. (art. 614-bis c.p.c.), Leonardo può agire.",
                                          [("614-bis", "c.p.c.")])
    assert ".," not in out


def test_an_abbreviation_keeps_its_stop():
    assert text_tidy.stop_comma("lett., e art. 2 c.c., comma 1") == "lett., e art. 2 c.c., comma 1"


def test_a_repeated_paragraph_counts_once():
    para = "Con il provvedimento di condanna il giudice fissa la somma dovuta per ogni violazione successiva."
    other = "Il giudice determina l'ammontare della somma tenuto conto del valore della controversia."
    parts, how = cm.article_parts("\n".join([para, other] * 6))
    assert parts == [para, other] and max(cm.comma_numbers(parts, how)) == 2


# --- A3, A4 --------------------------------------------------------------------------------------------------------

def test_the_exact_lost_subject_is_dropped():
    sentence = "In via subordinata, prevede che il passaggio coattivo debba essere stabilito nella parte del fondo meno gravosa."
    assert text_tidy.unreadable(sentence) == "no subject"
    out, _ = text_tidy.tidy(f"La strada è antica. {sentence} Il fondo è intercluso.")
    assert "In via subordinata" not in out and "intercluso" in out
    assert text_tidy.unreadable("In via subordinata, Bruno chiede la costituzione del passaggio.") == ""


@pytest.mark.parametrize("orphan", ["Questo principio si applica al caso in esame, atteso che Bruno usa la strada.",
                                    "Tale norma si applica al fondo di Bruno.", "Ciò vale anche per il fondo di Bruno.",
                                    "Tale disposizione rileva nel caso."])
def test_a_removal_takes_the_sentence_leaning_on_it(orphan):
    text = f"La strada è antica. Secondo la Cassazione il passaggio è servitù. {orphan} Il fondo è intercluso."
    s = text.index("Secondo")
    assert orphan not in ptw._drop_spans(text, [(s, text.index(orphan))]) and "intercluso" in ptw._drop_spans(text, [(s, text.index(orphan))])


def test_an_orphan_opening_the_next_paragraph_goes_too():
    text = "La strada è antica. Secondo la Cassazione il passaggio è servitù.\n\nQuesto principio si applica al caso in esame.\n\nIl fondo è intercluso."
    out = ptw._drop_spans(text, [(text.index("Secondo"), text.index("\n\n"))])
    assert "Questo principio" not in out and "intercluso" in out


def test_three_in_a_row_and_one_with_its_own_norm():
    text = ("Prima frase qui. Da togliere qui. Questo principio vale. Tale norma vale. Tale regola ex art. 1062 c.c. vale. Fine qui.")
    out = ptw._drop_spans(text, [(text.index("Da togliere"), text.index("Questo"))])
    assert "Questo principio" not in out and "Tale norma" not in out and "1062" in out


def test_nothing_removed_nothing_orphaned():
    text = "La norma è chiara. Questo principio si applica."
    assert ptw._drop_spans(text, []) == text


# --- A5, A6 --------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("brackets", ["(,,)", "(, ,)", "( , )", "[,;]", "()", "(.)", "( ,, )"])
def test_brackets_of_only_punctuation_go(brackets):
    assert text_tidy.empty_brackets(f"Il diritto {brackets} resta.") == "Il diritto resta."


def test_brackets_with_words_stay():
    assert text_tidy.empty_brackets("Il diritto (art. 1) resta.") == "Il diritto (art. 1) resta."


TEMPLATE = ("Il fatto «decadenza dall'azione di reintegrazione e di manutenzione» pone decadenza dall'azione di reintegrazione e "
            "di manutenzione: che cosa prevedono art. 1168 c.c. e come incidono sulla posizione della parte assistita.")


def test_the_exact_template_never_reaches_the_answer():
    out, log = text_tidy.tidy(f"La strada è antica. {TEMPLATE} Il fondo è intercluso.")
    assert "pone decadenza" not in out and "intercluso" in out and any(e["kind"] == "issue template" for e in log)


def test_an_added_issue_argues_the_rule_s_reason_and_facts():
    analysis = {"fact_signals": {"fired": [{"institutions": ["decadenza dall'azione"], "why": "Oltre l'anno l'azione è preclusa.",
                                            "matched": ["termine decorso — «spoglio del 2010»"]}]}}
    argument = dg._rule_argument(analysis, {"nome": "decadenza dall'azione"})
    assert argument == "Oltre l'anno l'azione è preclusa. Nel caso: termine decorso — «spoglio del 2010»."
    plan = {"questioni": [], "subordinate": [], "punti_deboli": [], "risposte": []}
    placed, _ = ptw._place_institutions(plan, [{"nome": "decadenza dall'azione", "refs": [("1168", "c.c.")], "argomento": argument}],
                                        [{"_article": ("1168", "c.c.")}])
    assert placed["questioni"][0]["argomento"] == argument and not text_tidy.unreadable(argument)


def test_no_generic_text_survives_as_an_argument():
    argument = ptw.FACT_ISSUE_ARGUMENT.format(nome="n", norme="art. 1 c.c.", fatto="f")
    assert not text_tidy._TEMPLATE_RE.search(argument)


# --- A7 ------------------------------------------------------------------------------------------------------------

def test_answer_docx_writes_the_log_when_verbose():
    source = (ROOT / "scripts" / "answer_docx.py").read_text(encoding="utf-8")
    assert 'VERBOSE_LOGGING' in source and 'f"{args.out}.log"' in source and "done.stdout" in source


# --- B1 ------------------------------------------------------------------------------------------------------------

def test_the_rule_requires_its_articles():
    rules = {r["id"]: r for r in fs.load_rules()}
    for rule in ("destinazione_padre_famiglia", "usucapione_servitu_apparente"):
        assert rules[rule]["requires"] == ["1079 c.c.", "1067 c.c.", "2043 c.c."]


def test_a_required_article_retrieval_missed_is_read_and_only_that(monkeypatch):
    asked = []
    monkeypatch.setattr(dg, "_fetch_articles", lambda session, refs: asked.extend(refs) or [{"_article": r} for r in refs])
    analysis = {"fact_signals": {"fired": [{"requires": [("1079", "c.c."), ("1067", "c.c."), ("2043", "c.c.")]},
                                           {"requires": [("1079", "c.c.")]}, {"id": "other"}]}}
    rows = dg._required_rows(None, analysis, {("2043", "c.c.")})
    assert asked == [("1079", "c.c."), ("1067", "c.c.")] and len(rows) == 2 and len(analysis["rule_required"]) == 3


def test_required_articles_go_with_the_recognition():
    plan = {"questioni": [{"titolo": "Acquisto", "norme": [("1061", "c.c.")], "pronunce": [], "argomento": "a"}],
            "subordinate": [], "punti_deboli": [], "risposte": []}
    out, log = ptw.place_rule_institutions(plan, {"rule_required": [("1079", "c.c."), ("1067", "c.c.")]},
                                           [{"_article": ("1079", "c.c.")}, {"_article": ("1061", "c.c.")}])
    assert out["questioni"][0]["norme"] == [("1061", "c.c."), ("1079", "c.c.")] and log["placed"] == ["1079 c.c."]


def test_a_bad_requires_is_refused(tmp_path):
    path = tmp_path / "r.yaml"
    path.write_text('- {id: a, area: diritti_reali, when: [{common_previous_owner: true}], institution: x, codice: c.c., '
                    'ranges: [[1, 2]], why: y, status: experiment, requires: ["art. 1079"]}', encoding="utf-8")
    with pytest.raises(ValueError, match="requires"):
        fs.load_rules(str(path))


# --- B2 ------------------------------------------------------------------------------------------------------------

SERVITUDE = {"servitude": True, "contract": False, "terms": []}


@pytest.mark.parametrize("sentence", ["Bruno può esperire l'azione negatoria ex art. 949 c.c. contro Carla.",
                                      "Bruno proporrà la negatoria per far cessare le turbative."])
def test_the_holder_of_a_servitude_never_uses_the_negatoria(sentence):
    assert "confessoria" in ptw._wrong_statement(sentence, SERVITUDE)
    assert ptw._wrong_statement(sentence, {"servitude": False, "contract": False, "terms": []}) == ""


def test_the_confessoria_is_fine():
    assert ptw._wrong_statement("Bruno esperirà la confessoria ex art. 1079 c.c.", SERVITUDE) == ""


def test_1223_measures_it_does_not_found():
    assert "measures" in ptw._wrong_statement("Bruno ha diritto al risarcimento ai sensi dell'art. 1223 c.c.")
    assert ptw._wrong_statement("Il danno si misura secondo l'art. 1223 c.c.") == ""
    assert ptw._wrong_statement("Ex art. 2043 c.c. con i criteri dell'art. 1223 c.c. Bruno ha diritto.") == ""


def test_no_breach_without_a_contract():
    sentence = "L'inadempimento di Carla obbliga al risarcimento."
    assert "contract" in ptw._wrong_statement(sentence, SERVITUDE)
    assert ptw._wrong_statement(sentence, {"servitude": False, "contract": True, "terms": []}) == ""
    assert ptw._wrong_statement(sentence, {}) == ""                  # no typed facts: nothing known


def test_1159_needs_a_title_from_a_non_owner_transcribed():
    bought = {"property_history": [{"owner": "Bruno", "how_acquired": "compravendita con rogito"}]}
    assert not fs._title_from_non_owner(bought)
    assert fs._title_from_non_owner({"contracts": [{"type": "compravendita trascritta a non domino"}]})
    assert "non-owner" in ptw._wrong_statement("Il termine dell'art. 1159 c.c. è decorso.", SERVITUDE)
    entry = {"contract": False, "terms": [{"id": "usucapione_abbreviata", "ref": "art. 1159 c.c.", "length": "10 anni", "status": "DECORSO"}]}
    assert ptw._wrong_statement("Il termine dell'art. 1159 c.c. è decorso.", entry) == ""


def test_a_time_barred_possessoria_is_never_proposed():
    context = {"contract": False, "terms": [{"id": "possessoria", "ref": "", "length": "1 anno", "status": "DECORSO"}]}
    assert "precluded" in ptw._wrong_statement("La reintegrazione è proponibile in via subordinata.", context)
    assert ptw._wrong_statement("La reintegrazione è preclusa per decadenza.", context) == ""
    open_term = {"contract": False, "terms": [{"id": "possessoria", "ref": "", "length": "1 anno", "status": "NON DECORSO"}]}
    assert ptw._wrong_statement("La reintegrazione è proponibile.", open_term) == ""


def test_1062_arises_from_the_separation_not_from_use():
    assert "separation" in ptw._wrong_statement("La servitù ex art. 1062 c.c. è sorta dall'uso del sentiero.")
    assert ptw._wrong_statement("La servitù ex art. 1062 c.c. è sorta con la separazione dei fondi.") == ""


TWENTY = {"id": "usucapione_ordinaria", "ref": "art. 1158 c.c.", "length": "20 anni"}


@pytest.mark.parametrize("sentence", ["Il termine ventennale risulta soddisfatto.", "Il requisito dell'art. 1158 c.c. è maturato."])
@pytest.mark.parametrize("status", ["NON VERIFICABILE", "INCERTO", "DA VERIFICARE"])
def test_a_term_is_never_satisfied_against_its_computed_status(sentence, status):
    assert "computed term" in ptw._wrong_statement(sentence, {"terms": [{**TWENTY, "status": status}]})


def test_nor_the_reverse():
    context = {"terms": [{**TWENTY, "status": "DECORSO"}]}
    assert "has run" in ptw._wrong_statement("Il termine ventennale è non verificabile.", context)
    assert ptw._wrong_statement("Il termine ventennale è decorso.", context) == ""
    open_term = {"terms": [{**TWENTY, "status": "NON VERIFICABILE"}]}
    assert ptw._wrong_statement("Il termine ventennale è da verificare.", open_term) == ""


def test_the_case_s_context_comes_from_the_signals(monkeypatch):
    analysis = {"fact_signals": {"facts": {"contracts": []}, "terms": [{**TWENTY, "status": "INCERTO"}],
                                 "fired": [{"institutions": ["usucapione di servitù apparente"], "acquired_right": True, "why": "uso"}]}}
    context = ptw.statement_context(analysis)
    assert context["servitude"] and context["contract"] is False and context["terms"][0]["status"] == "INCERTO"
    assert ptw.statement_context({}) == {}


def test_the_writer_is_told_each_rule():
    rules = ptw._WRITER_RULES
    assert all(word in rules for word in ("949", "1079", "1223", "2043", "1159", "preclusa", "1062", "TERMINI CALCOLATI"))
