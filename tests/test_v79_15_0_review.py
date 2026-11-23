"""v79.15.0 (review) -- the reviewer's findings on build A and B, on synthetic texts only (invented names, the build A
cases of tests/test_v79_15_0_analysis.py). No model: every model path and every outbound socket refused (that file's
sealing, re-exported below).

1. a sentence where the client speaks and another party or an authority asserts is that party's assertion: «provato»
   on it alone is «contestato» (asserting_party reads every asserting phrase, not the first)
2. the assertion verbs and markers: «ritengono», «racconta», «riporta che», «Per la Procura …» at the start, a
   passive's agent («accusato da …»); «per» in the middle of a sentence is no speaker
3. the client by his name only: a party sharing his surname is not the client, «la difesa di <controparte>» is the
   counterparty; a bare shared surname is never read as the client's own statement
4. an affermazione_di_diritto naming an article is verified only by that article's periods, and only when its words
   are grounded in the cited periods (law_statements_off_article, law_statements_ungrounded)
5. the Q&A / §4 filter reads a rule's other phrasings (commina, la pena, deve, reclusione) and «la norma» with no
   number
6. compose with the verified analysis: the old engine's no-claim verdict writes no §3 / §7 (only logged)
"""

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import test_v79_15_0_analysis as T  # noqa: E402
from src.rag import analysis_step as A  # noqa: E402

_sealed = T._sealed
LINE = T._line("640 c.p.", T.E640[0], "Tesi.", ["S5"], ["A640cp.1"])


def _with_sentence(text, frame=T.PENAL_FRAME):
    sents = T._sents(T.PENAL_SENTENCES, sistema=(9,))
    # v79.16.1 (item 4): the sentence under test keeps a lemma of the element it is cited for («artifizi»), so the
    # provenance rules, not the pertinence one, decide these cases
    sents[0] = {**sents[0], "text": f"{text} Gli artifizi sono il punto."}
    return A.make_input(frame, T.PENAL_NORMS, sents, [], client="Paolo Sarti", client_roles=["imputato", "indagato"])


def _provato_on_s1(inp):
    model = T._Model(T._reply(LINE, elementi=[T._el("640 c.p.", T.E640[0], "provato", ["S1"])]))
    return {e["elemento"]: e for e in A.analyse(inp, call=model)["analysis"]["elementi"]["640 c.p."]}[T.E640[0]]


# --- 1. the client speaks, another party asserts ----------------------------------------------------------------------

@pytest.mark.parametrize("text, who", [
    ("Paolo Sarti nega, ma la Procura sostiene che ha usato un falso annuncio.", "la Procura"),
    ("Paolo Sarti nega e Giulia Neri afferma che l'annuncio era falso.", "Giulia Neri"),
    ("Paolo Sarti dichiara di aver spedito la bicicletta, mentre Giulia Neri sostiene di non averla ricevuta.",
     "Giulia Neri"),
])
def test_a_sentence_with_the_clients_denial_and_another_partys_assertion_is_that_partys(text, who):
    inp = _with_sentence(text)
    assert A.asserting_party(text, inp) == who
    e = _provato_on_s1(inp)
    assert e["esito"] == "contestato" and e["parte_che_afferma"] == who


# --- 2. the verbs and the markers -------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, who", [
    ("Gli investigatori ritengono che Paolo Sarti abbia usato un falso annuncio.", "Gli investigatori"),
    ("Gli inquirenti ritenevano falso l'annuncio.", "Gli inquirenti"),
    ("Per la Procura Paolo Sarti ha usato un falso annuncio.", "la Procura"),
    ("Per Giulia Neri, la bicicletta non è mai arrivata.", "Giulia Neri"),
    ("Giulia Neri racconta che la bicicletta non è mai arrivata.", "Giulia Neri"),
    ("La querela di Giulia Neri riporta che l'annuncio era falso.", "Giulia Neri"),
    ("Paolo Sarti è accusato da Giulia Neri di aver usato un falso annuncio.", "Giulia Neri"),
    ("Paolo Sarti è stato denunciato dalla Procura per truffa.", "la Procura"),
    ("Nella versione di Giulia Neri l'annuncio era falso.", "Giulia Neri"),
])
def test_the_authorities_and_the_parties_by_every_assertion_form_are_downgraded(text, who):
    inp = _with_sentence(text)
    assert A.asserting_party(text, inp) == who
    assert _provato_on_s1(inp)["esito"] == "contestato"


@pytest.mark.parametrize("text", [
    "Paolo Sarti ha venduto la bicicletta per Giulia Neri.",                 # «per» mid-sentence: no speaker
    "Per procurare un profitto, Paolo Sarti ha venduto la bicicletta.",     # no party after «Per»
    "Paolo Sarti racconta di aver spedito la bicicletta il 3 marzo.",       # the client's own statement
    "Giulia Neri riporta lesioni lievi.",                                    # «riporta» with no «che»: no assertion
    "Il vicino ritiene che la bicicletta sia arrivata.",                     # no party of the frame
])
def test_what_is_no_other_partys_assertion_stays_provato(text):
    inp = _with_sentence(text)
    assert A.asserting_party(text, inp) == ""
    assert _provato_on_s1(inp)["esito"] == "provato"


# --- 3. the client by his name ----------------------------------------------------------------------------------------

SAME_SURNAME = {**T.PENAL_FRAME, "parti": [
    {"nome": "Paolo Sarti", "ruolo_nel_fatto": "venditore", "lato": "cliente"},
    {"nome": "Giulia Sarti", "ruolo_nel_fatto": "acquirente", "lato": "controparte"}]}


def test_a_party_sharing_the_clients_surname_is_not_the_client():
    text = "Giulia Sarti afferma che Paolo Sarti ha usato un falso annuncio."
    inp = _with_sentence(text, SAME_SURNAME)
    assert A.asserting_party(text, inp) == "Giulia Sarti"
    assert _provato_on_s1(inp)["esito"] == "contestato"
    assert A.asserting_party("Paolo Sarti afferma di aver spedito la bicicletta.", inp) == ""
    assert A.asserting_party("Sarti afferma di aver spedito la bicicletta.", inp) == "Giulia Sarti"   # a tie: never
    assert not A._is_client("Giulia Sarti", inp) and A._is_client("Paolo Sarti", inp)                  # the client's


def test_the_defence_of_the_counterparty_is_the_counterparty():
    text = "La difesa di Giulia Neri sostiene che l'annuncio era falso."
    inp = _with_sentence(text)
    assert A.asserting_party(text, inp) == "Giulia Neri"
    assert _provato_on_s1(inp)["esito"] == "contestato"
    for own in ("La difesa di Paolo Sarti sostiene che l'annuncio era vero.", "Il difensore sostiene che l'annuncio era "
                "vero.", "L'indagato dichiara di aver ricevuto il denaro."):
        assert A.asserting_party(own, inp) == "", own


def test_a_parte_che_afferma_sharing_the_surname_is_a_reported_assertion():
    inp = _with_sentence("Paolo Sarti ha venduto la bicicletta.", SAME_SURNAME)
    assert A.reported_by({"text": "Paolo Sarti ha venduto la bicicletta."}, inp, "Giulia Sarti") == "Giulia Sarti"
    assert A.reported_by({"text": "Paolo Sarti ha venduto la bicicletta."}, inp, "Paolo Sarti") == ""


# --- 4. a law statement verified only by its own article's text -------------------------------------------------------

def _statements(*items):
    model = T._Model(T._reply(LINE, affermazioni=list(items)))
    res = A.analyse(T._penal(), call=model)
    return res["analysis"]["affermazioni_di_diritto"], res["counters"]


def test_a_statement_on_one_article_citing_anothers_period_is_dropped():
    kept, c = _statements({"testo": "L'art. 62-bis c.p. impone al giudice di ridurre sempre la pena di metà.",
                           "cita_norma": ["A640cp.1"]})
    assert kept == [] and c["law_statements_dropped"] == 1 and c["law_statements_off_article"] == 1
    kept, c = _statements({"testo": "L'art. 646 c.p. punisce l'appropriazione indebita.", "cita_norma": ["A640cp.1"]})
    assert kept == [] and c["law_statements_off_article"] == 1                      # no fitted norm at all


def test_a_statement_its_cited_periods_do_not_ground_is_dropped():
    kept, c = _statements({"testo": "L'art. 62-bis c.p. impone al giudice di ridurre sempre la pena di metà.",
                           "cita_norma": ["A62biscp.1"]})
    assert kept == [] and c["law_statements_ungrounded"] == 1 and c["law_statements_dropped"] == 1


def test_a_grounded_statement_on_its_own_article_is_kept():
    good = {"testo": "L'art. 62-bis c.p. consente al giudice di prendere in considerazione altre circostanze diverse "
                     "che giustificano una diminuzione della pena.", "cita_norma": ["A62biscp.1"]}
    unnamed = {"testo": "Chi con artifizi o raggiri induce taluno in errore e procura a sé un ingiusto profitto con "
                        "altrui danno è punito con la reclusione.", "cita_norma": ["A640cp.1"]}
    kept, c = _statements(good, unnamed)
    assert [k["articoli"] for k in kept] == [["62-bis c.p."], ["640 c.p."]]
    assert c["law_statements_kept"] == 2 and c["law_statements_dropped"] == 0


# --- 5. the Q&A / §4 filter reads a rule's other phrasings ------------------------------------------------------------

VIEW_640 = {"statements": [{"testo": "Chi con artifizi o raggiri induce taluno in errore e procura a sé un ingiusto "
                                     "profitto con altrui danno è punito con la reclusione.", "articoli": ["640 c.p."],
                            "cita_norma": ["A640cp.1"]}]}


@pytest.mark.parametrize("text", [
    "L'art. 640 c.p. commina la reclusione fino a dieci anni.",
    "In base all'art. 640 c.p. la pena è la reclusione fino a dieci anni.",
    "Ai sensi dell'art. 640 c.p. chi induce in errore deve essere condannato all'ergastolo.",
    "La norma prevede la reclusione fino a dieci anni.",
    "L'art. 640 c.p. prevede la reclusione fino a dieci anni.",
])
def test_an_unverified_rule_in_any_phrasing_is_removed(text):
    log = []
    assert A.qa_filter([text], VIEW_640, log) == [] and len(log) == 1


def test_the_verified_rule_and_the_sentences_stating_none_are_kept():
    keep = ["Ai sensi dell'art. 640 c.p. chi con artifizi o raggiri induce taluno in errore e procura a sé un ingiusto "
            "profitto con altrui danno è punito.",
            "La norma punisce chi con artifizi o raggiri induce taluno in errore procurando un ingiusto profitto.",
            "Il caso riguarda l'art. 640 c.p.", "Paolo Sarti ha venduto la bicicletta il 3 marzo.",
            "La norma è stata letta sui fatti narrati."]
    assert A.qa_filter(keep, VIEW_640) == keep


# --- 6. compose: the old engine's no-claim verdict decides nothing with the verified analysis -------------------------

ART_1476 = ("Le obbligazioni principali del venditore sono: 1) quella di consegnare la cosa al compratore; 2) quella "
            "di fargli acquistare la proprietà della cosa o il diritto, se l'acquisto non è effetto immediato del "
            "contratto; 3) quella di garantire il compratore dall'evizione e dai vizi della cosa.")


def test_compose_with_the_verified_analysis_never_prints_the_no_claim_sections(monkeypatch, tmp_path):
    import test_v79_12_5_b as B
    from src.rag import lawyer_method as M, plan_then_write
    monkeypatch.setenv("THESIS_DETERMINISTIC", "")
    monkeypatch.setattr(plan_then_write, "negation_review", lambda items, known, call=None: [])
    ctx = B._no_claim_ctx()
    ctx["candidates"] = M.thesis_candidates(ctx)
    plan = M.clean_method({}, ctx["candidates"])
    assert M.no_claim(ctx) == "none"                                    # the old engine: no admitted claim
    frame = {"parti": [{"nome": "Tizio", "ruolo_nel_fatto": "comodante", "lato": "cliente"},
                       {"nome": "Caio", "ruolo_nel_fatto": "comodatario", "lato": "controparte"}],
             "cliente": {"nome": "Tizio"}, "rito": {"valore": "civile"}}
    norm = {"article": "1476 c.c.", "text": ART_1476, "rubrica": "Obbligazioni principali del venditore",
            "elements": [{"id": "R1", "testo": "consegnare la cosa"}]}
    inp = A.make_input(frame, [norm], T._sents(["Tizio ha consegnato la cosa a Caio."]), [], client="Tizio",
                       client_roles=["attore"])
    reply = T._reply(T._line("1476 c.c.", "A1476cc:E1", "Tizio chiede la consegna della cosa.", ["S1"],     # v79.15.1:
                             ["A1476cc.1"]), elementi=[T._el("1476 c.c.", "A1476cc:E1", "provato", ["S1"])])  # by id
    result = A.analyse(inp, call=T._Model(reply))
    assert result["status"] == "ok"
    ctx["verified_analysis"] = A.view(result, inp)
    log = {}
    text = M.compose([("Norme applicate ai fatti", "Regola.")], ctx, plan, log=log)
    secs = B._sections(text)
    # v79.15.3 (item 1): the code's tesi on the verified esito (no verified role: the neutral wording)
    assert secs[M.title("sintesi")].startswith("Linea principale: art. 1476 c.c. — «consegnare la cosa» risulta "
                                               "provato: «Tizio ha consegnato la cosa a Caio».")
    assert secs[M.title("questione")] != M.NO_CLAIM_QUESTION.format(client="Tizio", materia="comodato")
    assert secs[M.title("obiezioni")] != M.NO_CLAIM_OBJECTIONS
    assert M.NO_CLAIM_OBJECTIONS not in text and M.NO_CLAIM_CONCLUSION not in text
    assert log["no_claim_legacy"] == "none" and log["no_claim"] == "" and ctx["no_claim_sections"] == ""
