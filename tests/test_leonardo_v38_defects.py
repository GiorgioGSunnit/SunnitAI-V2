"""Leonardo v38: the text defects, from their exact strings."""
import importlib.util
import os
from pathlib import Path

import docx
import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

from src.rag import anchored_writing as aw, text_tidy  # noqa: E402
import src.rag.plan_then_write as ptw  # noqa: E402

ARTICLES = {
    "2043": "Qualunque fatto doloso o colposo, che cagiona ad altri un danno ingiusto, obbliga colui che ha commesso il fatto "
            "a risarcire il danno.",
    "1062": "La destinazione del padre di famiglia ha luogo quando consta, mediante qualunque genere di prova, che due fondi, "
            "attualmente divisi, sono stati posseduti dallo stesso proprietario, e che questi ha posto le cose nello stato dal "
            "quale risulta la servitù. Se i due fondi cessarono di appartenere allo stesso proprietario, la servitù s'intende "
            "stabilita.",
    "1051": "Il proprietario, il cui fondo è circondato da fondi altrui, e che non ha uscita sulla via pubblica né può procurarsela "
            "senza eccessivo dispendio o disagio, ha diritto di ottenere il passaggio sul fondo vicino.\n\n"
            "2. Il passaggio si deve stabilire in quella parte per cui l'accesso alla via pubblica è più breve.\n\n"
            "3. Essi possono essere estesi a chi ha bisogno di ampliare un passaggio.",
}
CPC = {"614-bis": "Con il provvedimento di condanna all'adempimento di obblighi diversi dal pagamento di una somma di denaro "
                  "il giudice, salvo che ciò sia manifestamente iniquo, fissa, su richiesta di parte, la somma di denaro dovuta.",
       "700": "Fuori dei casi precedenti, chi ha fondato motivo di temere che durante il tempo occorrente per far valere il suo "
              "diritto in via ordinaria, questo sia minacciato da un pregiudizio imminente e irreparabile, può chiedere con "
              "ricorso al giudice i provvedimenti d'urgenza."}
ROWS = [{"_article": (n, "c.c."), "s": {"plain_text": t}} for n, t in ARTICLES.items()] + \
       [{"_article": (n, "c.p.c."), "s": {"plain_text": t}} for n, t in CPC.items()]


def final(text):
    text, _ = text_tidy.against_articles(text, ROWS)
    return text_tidy.tidy(text)[0]


@pytest.mark.parametrize("sentence", [
    "Qualunque fatto doloso o colposo che cagiona ad altri un danno ingiusto, ai sensi dell'art. 2043 c.c., co. 1., è "
    "giustificato in quanto l'opera è abusiva.",
    "Con il provvedimento di condanna all'adempimento di obblighi diversi dal pagamento di una somma di denaro, ai sensi "
    "dell'art. 614-bis c.p.c., co. 1.",
    "La destinazione del padre di famiglia ha luogo quando consta mediante opera visibile e permanente, ai sensi dell'art. "
    "1062 c.c., co. 1., ha formato la servitù.",
    "Il proprietario, il cui fondo è circondato da fondi altrui, ai sensi dell'art. 1051 c.c., co. 3.",
    "Fuori dei casi precedenti, ai sensi dell'art. 700 c.p.c., chi ha fondato.",
])
def test_comma_words_around_a_citation_never_reach_the_answer(sentence):
    out = final(f"Leonardo agisce contro il vicino. {sentence} La domanda è fondata.")
    assert out == "Leonardo agisce contro il vicino. La domanda è fondata."


def test_the_same_words_with_no_citation_template_stay():
    text = "Leonardo ha subito un fatto colposo, ai sensi dell'art. 2043 c.c., e chiede il risarcimento del danno."
    assert final(text) == text


def test_a_cut_comma_alone_goes():
    assert final("Fuori dei casi precedenti, chi ha fondato. La domanda è fondata.") == "La domanda è fondata."


@pytest.mark.parametrize("old, new", [
    ("L'opera è illecita ex art. 2043 c.c., co. 1., e va rimossa.", "L'opera è illecita ex art. 2043 c.c., co. 1, e va rimossa."),
    ("L'opera è illecita (art. 2043 c.c., comma 1. e va rimossa).", "L'opera è illecita (art. 2043 c.c., comma 1 e va rimossa)."),
])
def test_no_stop_after_a_comma_s_number_inside_a_sentence(old, new):
    assert text_tidy.stop_comma(old) == new


def test_a_comma_number_ending_the_sentence_keeps_its_stop():
    assert text_tidy.stop_comma("Lo dice l'art. 2043 c.c., co. 1. Leonardo agisce.") == "Lo dice l'art. 2043 c.c., co. 1. Leonardo agisce."


def test_a_rewrite_copying_the_comma_is_refused_at_the_source():
    passage = ARTICLES["2043"]
    assert text_tidy.comma_words_cited("Qualunque fatto doloso o colposo che cagiona ad altri un danno ingiusto, ai sensi dell'art. "
                                       "2043 c.c., co. 1.", [passage])
    assert not text_tidy.comma_words_cited("Chi cagiona un danno ingiusto risponde, ai sensi dell'art. 2043 c.c.", [passage])
    assert "ai sensi dell'art" in aw.ANCHOR_RULE and "ai sensi dell'art" in aw.REWRITE_SYSTEM


def test_plan_cover_refuses_the_copied_comma():
    why = ptw._cover_refusal("Qualunque fatto doloso o colposo che cagiona ad altri un danno ingiusto, ai sensi dell'art. 2043 c.c.",
                             ("2043", "c.c."), [], ROWS)
    assert why == "copies the comma's words"


# --- 2 ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("old, new", [
    ("Per il possesso il termine è decorso, se il termine è effettivamente scaduto.", "Per il possesso il termine è decorso."),
    ("Il termine è decorso, se il termine è effettivamente scaduto, e l'azione è preclusa.",
     "Il termine è decorso, e l'azione è preclusa."),
    ("Il termine è decorso, se il termine è decorso.", "Il termine è decorso."),
    ("Il termine è scaduto, qualora il termine sia decorso davvero.", "Il termine è scaduto, qualora il termine sia decorso davvero."),
    ("L'azione è proponibile, se il termine è scaduto.", "L'azione è proponibile, se il termine è scaduto."),
])
def test_a_condition_repeating_its_clause_goes(old, new):
    assert text_tidy.self_condition(old) == new


def test_it_goes_every_time_in_the_answer():
    text = ("Il termine è decorso, se il termine è effettivamente scaduto. La domanda è fondata.\n\n"
            "Inoltre il termine è decorso, se il termine è effettivamente scaduto.")
    assert "effettivamente" not in text_tidy.tidy(text)[0]


# --- 3 ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sentence", [
    "potrebbe essere esercitata da Leonardo per far dichiarare l'inesistenza di qualsiasi diritto ostativo.",
    "Potrebbe essere esercitata da Leonardo per far dichiarare l'inesistenza di qualsiasi diritto ostativo.",
    "In via subordinata, potrebbe applicarsi.",
])
def test_a_sentence_that_lost_its_subject_goes(sentence):
    assert text_tidy.tidy(f"La domanda è fondata.\n\n{sentence}")[0] == "La domanda è fondata."


@pytest.mark.parametrize("sentence", [
    "Può essere proposta domanda di risarcimento dinanzi al giudice civile.",
    "L'azione negatoria potrebbe essere esercitata da Leonardo per far dichiarare l'inesistenza del diritto.",
    "In via subordinata, potrebbe applicarsi la disciplina della garanzia.",
])
def test_a_sentence_with_its_subject_stays(sentence):
    assert sentence in text_tidy.tidy(f"La domanda è fondata. {sentence}")[0]


# --- 4 ---------------------------------------------------------------------------------------------------------------

def _docx_script():
    spec = importlib.util.spec_from_file_location("answer_docx", Path(__file__).parent.parent / "scripts" / "answer_docx.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rulings_are_listed_by_their_reference_line_alone():
    passage = "Il ricorso è infondato perché la Corte d'appello ha correttamente ritenuto " * 20
    citations = [{"document_name": "Codice Civile 2026", "document_type": "primary",
                  "sections": [{"name": "2043.0.0", "title": "Risarcimento", "plain_text": "Qualunque fatto."}]},
                 {"document_name": "snciv@s20@a2026@n01234@tO.clean.pdf", "document_type": "interpretation",
                  "reference": "Cass. civ., sez. II, ord. n. 1234/2026",
                  "sections": [{"name": "p1", "title": passage[:100], "plain_text": passage}]},
                 {"document_name": "Sentenza senza riferimento", "document_type": "interpretation",
                  "sections": [{"name": "p1", "title": "t", "plain_text": passage}]}]
    doc = docx.Document()
    _docx_script().add_sources(doc, citations)
    lines = [p.text for p in doc.paragraphs]
    assert "Cass. civ., sez. II, ord. n. 1234/2026" in lines and "Sentenza senza riferimento" in lines
    assert not any("infondato" in line or "snciv" in line for line in lines)
    assert "Qualunque fatto." in lines and any(line.startswith("Art. 2043") for line in lines)
