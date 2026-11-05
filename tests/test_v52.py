"""v52: _citation_context, the text the qualification check (and its stamp) and diagnose_long's rubrica fits judge:
the sentence citing the article (with the one before it when under 60 characters), never a line cut at 400; a number
inside another article's quoted words is not a citation. The sentences are v51's (diagnose_reports/v51-candidate).
No model, no database."""
import importlib.util
import os
from pathlib import Path

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

import pytest  # noqa: E402

import src.rag.defensive_generation as dg  # noqa: E402

HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("diagnose_long", HERE.parent / "scripts" / "diagnose_long.py")
dl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dl)

# v51 n8: the line's first 400 characters (what was judged for 1114 c.c.) end before 1114 is cited
N8_LINE = ("L’art. 178 c.c. stabilisce che «i beni destinati all’esercizio dell’impresa di uno dei coniugi costituita dopo il "
           "matrimonio» sono «oggetto della comunione solo se sussistono al momento dello scioglimento di questa». Ciò "
           "implica che, se tali beni non sono presenti al momento dello scioglimento della comunione, non rientrano nel "
           "patrimonio comune e non sono soggetti alla divisione. L’art. 179 c.c., richiamato dalla controparte, riguarda "
           "i beni personali e non rileva nel caso. La divisione dei beni comuni segue l’art. 1114 c.c. per la divisione "
           "in natura. Va chiesta in giudizio.")
# v51 h8: 615-ter ... 635-quinquies only inside 629's words
H8_QUOTE = ("L'art. 629 c.p. punisce «con la reclusione da sei a dodici anni e con la multa da euro 5.000 a euro 10.000» "
            "«chiunque, mediante le condotte di cui agli articoli 615-ter, 617-quater, 617-sexies, 635-bis, 635-quater e "
            "635-quinquies ovvero con la minaccia di compierle», costringa «taluno a fare o ad omettere qualche cosa, "
            "procurando a sé o ad altri un ingiusto profitto con altrui danno».")


@pytest.mark.parametrize("documents", [None, []])
def test_a_short_sentence_past_the_400th_character_is_judged_itself(documents):
    assert N8_LINE.index("1114") > 400
    context = dg._citation_context(N8_LINE, ("1114", "c.c."), documents)
    assert context == "La divisione dei beni comuni segue l’art. 1114 c.c. per la divisione in natura."
    assert len(dg._citation_context(N8_LINE, ("178", "c.c."), documents)) > 0


@pytest.mark.parametrize("documents", [None, []])
def test_a_sentence_under_60_characters_comes_with_the_one_before(documents):
    draft = "Il fondo è intercluso dal lato della strada provinciale da anni.\nSi applica l'art. 1051 c.c.\n"
    assert dg._citation_context(draft, ("1051", "c.c."), documents) == \
        "Il fondo è intercluso dal lato della strada provinciale da anni. Si applica l'art. 1051 c.c."
    assert dg._citation_context("Si applica l'art. 1051 c.c.", ("1051", "c.c."), documents) == \
        "Si applica l'art. 1051 c.c."                                            # nothing before it


@pytest.mark.parametrize("documents", [None, []])
@pytest.mark.parametrize("number", ["615-ter", "617-quater", "617-sexies", "635-bis", "635-quater", "635-quinquies"])
def test_a_number_inside_another_articles_quoted_words_is_not_judged(number, documents):
    ref = (number, "c.p.")
    assert dg._citation_context(H8_QUOTE, ref, documents) == ""
    later = H8_QUOTE + f" Il reato di cui all'art. {number} c.p. resta distinto dall'estorsione."
    assert dg._citation_context(later, ref, documents) == \
        f"Il reato di cui all'art. {number} c.p. resta distinto dall'estorsione."
    assert dg._citation_context(H8_QUOTE, ("629", "c.p."), documents).startswith("L'art. 629 c.p. punisce")


def test_the_qualification_check_sends_the_sentence_and_skips_the_quoted(monkeypatch):
    sent = []
    monkeypatch.setattr(dg, "_chat_json", lambda messages, **k: sent.append(messages[-1].content) or [])
    rows = [{"_article": (n, code), "_grade": 2, "s": {"name": f"{n}.0.0", "plain_text": f"Testo {n}.",
                                                       "titolo_articolo": f"Rubrica {n}"}}
            for n, code in (("1114", "c.c."), ("615-ter", "c.p."), ("629", "c.p."))]
    dg._check_qualifications(N8_LINE + "\n" + H8_QUOTE, {}, rows, None, [], "")
    human = sent[0]
    assert "Frase: La divisione dei beni comuni segue l’art. 1114 c.c. per la divisione in natura." in human
    assert "[615-ter c.p.]" not in human and "[629 c.p.]" in human


def test_diagnose_rubrica_fits_scores_the_sentence_and_not_the_quoted(monkeypatch):
    asked = []
    monkeypatch.setattr(dg, "_chat_json", lambda messages, **k: asked.append(messages[-1].content) or [])
    rows = {(n, c): {"s": {"titolo_articolo": f"Rubrica {n}"}} for n, c in (("1114", "c.c."), ("615-ter", "c.p."))}
    fits = dl._rubrica_fits(dg, N8_LINE + "\n" + H8_QUOTE, list(rows), rows, [])
    assert fits["1114 c.c."]["context"].startswith("La divisione dei beni comuni")
    assert fits["615-ter c.p."]["context"] == "" and "[615-ter c.p.]" not in asked[0]
