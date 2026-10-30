"""src/rag/text_tidy.py and the cleanup in anchored_writing.strip_anchors / drop_citations: the defects left by the
edits after writing, on the examples seen in real answers. No model, no database."""
import os

os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")

import pytest  # noqa: E402

from src.rag import anchored_writing as aw  # noqa: E402
from src.rag import text_tidy  # noqa: E402


@pytest.mark.parametrize("before,after", [
    ("La cosa venduta deve essere idonea all'uso a cui è destinata ().",
     "La cosa venduta deve essere idonea all'uso a cui è destinata."),
    ("Il venditore è tenuto alla garanzia ( ) per i vizi.", "Il venditore è tenuto alla garanzia per i vizi."),
    ("Il compratore deve denunciare i vizi [] entro otto giorni.", "Il compratore deve denunciare i vizi entro otto giorni."),
])
def test_empty_brackets_go(before, after):
    text, log = text_tidy.tidy(before)
    assert text == after and log[0]["kind"] == "empty brackets"


def test_a_stripped_anchor_leaves_no_brackets():
    assert aw.strip_anchors("La cosa deve essere idonea all'uso a cui è destinata ([A3]).") == \
        "La cosa deve essere idonea all'uso a cui è destinata."
    assert aw.strip_anchors("Il custode risponde [A1, A2].") == "Il custode risponde."


def test_a_sentence_starting_with_a_fragment_is_repaired_or_dropped():
    text, log = text_tidy.tidy("Il contratto è valido. , comma 1, la cosa venduta deve essere esente da vizi.")
    assert text == "Il contratto è valido. La cosa venduta deve essere esente da vizi."
    assert log[0]["kind"] == "fragment repaired"
    text, _ = text_tidy.tidy("Il contratto è valido. comma 2 del Codice Civile, il termine decorre dalla scoperta.")
    assert text == "Il contratto è valido. Il termine decorre dalla scoperta."
    text, log = text_tidy.tidy("Il contratto è valido. , comma 1, c.c.\nIl compratore agisce in tempo.")
    assert text == "Il contratto è valido.\nIl compratore agisce in tempo." and log[0]["kind"] == "fragment dropped"
    text, _ = text_tidy.tidy("- , comma 3, il venditore risponde dei vizi occulti.")
    assert text == "- Il venditore risponde dei vizi occulti."
    plain = "Comma e articolo sono parole del testo, e la frase resta com'è."
    assert text_tidy.tidy(plain)[0] == plain


def test_dropping_a_citation_takes_its_comma_and_code_and_capitalises():
    assert aw.drop_citations("L'art. 1490 c.c., comma 1, prevede che la cosa sia esente da vizi.",
                             [("1490", "c.c.")]) == "Prevede che la cosa sia esente da vizi."
    assert aw.drop_citations("Il venditore risponde ai sensi dell'art. 1490, comma 1 del Codice Civile, dei vizi.",
                             [("1490", "c.c.")]) == "Il venditore risponde, dei vizi."
    assert aw.drop_citations("La cosa deve essere idonea all'uso a cui è destinata (art. 1490 c.c.).",
                             [("1490", "c.c.")]) == "La cosa deve essere idonea all'uso a cui è destinata."


def test_a_mid_sentence_unverified_note_moves_to_the_end_never_deleted():
    text, log = text_tidy.tidy("Il venditore [DA VERIFICARE: l'art. 1495 c.c. riguarda «Termini e condizioni per "
                               "l'azione»] risponde dei vizi entro un anno dalla consegna.")
    assert text == "Il venditore risponde dei vizi entro un anno dalla consegna (da verificare)."
    assert log[0]["kind"] == "note moved"
    text, _ = text_tidy.tidy("Il termine [DA VERIFICARE] decorre (da verificare: data incerta) dalla scoperta.")
    assert text == "Il termine decorre dalla scoperta (da verificare)."
    at_end = "Il termine decorre dalla scoperta [DA VERIFICARE]."
    assert text_tidy.tidy(at_end)[0] == at_end
    glued = "La pretesa rientra nell'art. 2051 c.c., non nell'art. 2044 c.c. [DA VERIFICARE] Né in altre norme."
    assert text_tidy.tidy(glued)[0] == glued        # a sentence's end the splitter does not see: left there
    tagged = ("La pretesa rientra nell'art. 2051 c.c., non nell'art. 2044 c.c. [DA VERIFICARE], né nell'articolo 50 "
              "del Codice Civile [DA VERIFICARE].")
    assert text_tidy.tidy(tagged)[0] == tagged      # a bare tag marking its citation stays with it
