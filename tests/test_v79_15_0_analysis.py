"""v79.15.0 build A -- the verified analysis step (src/rag/analysis_step.py, data/analysis_step.json,
data/remedy_articles.json) and the fit by ids (src/rag/applicability.py, brief item 3). The model is faked at its
source; every other model path and every outbound socket refused (the 79.14.3 fixture's pattern). Synthetic facts and
invented names only; the article texts are statutory (640, 641, 62-bis c.p.; 1385, 1453, 2043 c.c.).

1. the input: the verified frame's fields, the fitted norms' schema elements and numbered periods (A<art>.<n>, the
   fit's own splitter), the sentences S1..Sn as the flow numbers them (the system's assumptions never sent), the
   questions; the prompt from data, thinking off, temperature 0
2. the verifier: a bad S-id, a bad A-id, an element that is no E-id (v79.15.1; was: a non-verbatim element),
   «provato» on another party's or an authority's
   assertion -> «contestato», the client's own statement never downgraded, «escluso» with no S-id, a remedy article as
   the line -> rejected, one re-ask, then «da determinare», the frame's charge missing -> one re-ask, then the frame's
   article with every element «da_provare», a law statement with no verifying A-id dropped; at most two calls
3. a deadline, a failed call, a reply with no JSON: nothing from the step
4. run(state): off with the flag off; the state's input; the «analysis_step» log line
5. the fit by ids: «si» cites the article's periods, «no» / «dipende» the facts; the frame's charge and a user-named
   article never removed (fit_protected_kept)
"""

import json
import os
import socket
import sys
import time
from pathlib import Path

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.rag import analysis_step as A  # noqa: E402
from src.rag import applicability as ap  # noqa: E402
from src.rag import case_frame as cf  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402


def _refused(*a, **k):
    raise AssertionError("no real model call or connection in a test")


@pytest.fixture(autouse=True)
def _sealed(monkeypatch, tmp_path):
    """Every model path refused at its source, and below them any outbound socket."""
    monkeypatch.setenv("ARTICLE_SCHEMAS_FILE", str(tmp_path / "schemas.jsonl"))
    monkeypatch.setenv("ARTICLE_SCHEMAS_REVIEWED_FILE", str(tmp_path / "reviewed.jsonl"))
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(socket.socket, "connect", _refused)
    monkeypatch.setattr(socket, "create_connection", _refused)
    from src.rag import ai_chat
    monkeypatch.setattr(ai_chat, "ChatOpenAI", _refused)
    for name, mod in list(sys.modules.items()):
        if name.startswith("src.rag.") and mod is not None:
            for attr in ("_call_chat", "_call_chat_with_tools", "embedding_model", "_embed_query_with_prefix"):
                if hasattr(mod, attr):
                    monkeypatch.setattr(mod, attr, _refused)
    monkeypatch.setattr(A, "_default_call", _refused)
    monkeypatch.setattr(ap, "_default_call", _refused)
    monkeypatch.setattr(cf, "_default_call", _refused)


# --- the synthetic cases ----------------------------------------------------------------------------------------------

ART_640 = ("Chiunque, con artifizi o raggiri, inducendo taluno in errore, procura a sé o ad altri un ingiusto profitto "
           "con altrui danno, è punito con la reclusione da sei mesi a tre anni e con la multa da euro 51 a euro 1.032.")
ART_641 = ("Chiunque, dissimulando il proprio stato d'insolvenza, contrae un'obbligazione col proposito di non "
           "adempierla è punito, a querela della persona offesa, qualora l'obbligazione non sia adempiuta, con la "
           "reclusione fino a due anni o con la multa fino a euro 516.")
ART_62BIS = ("Il giudice, indipendentemente dalle circostanze previste nell'articolo 62, può prendere in considerazione "
             "altre circostanze diverse, qualora le ritenga tali da giustificare una diminuzione della pena. Esse sono "
             "considerate in ogni caso, ai fini dell'applicazione di questo capo, come una sola circostanza, la quale "
             "può anche concorrere con una o più delle circostanze indicate nel predetto articolo 62.")
ART_1385 = ("1. Se al momento della conclusione del contratto una parte dà all'altra, a titolo di caparra, una somma di "
            "danaro o una quantità di altre cose fungibili, la caparra, in caso di adempimento, deve essere restituita "
            "o imputata alla prestazione dovuta.\n"
            "2. Se la parte che ha dato la caparra è inadempiente, l'altra può recedere dal contratto, ritenendo la "
            "caparra; se inadempiente è invece la parte che l'ha ricevuta, l'altra può recedere dal contratto ed "
            "esigere il doppio della caparra.\n"
            "3. Se però la parte che non è inadempiente preferisce domandare l'esecuzione o la risoluzione del "
            "contratto, il risarcimento del danno è regolato dalle norme generali.")
ART_1453 = ("1. Nei contratti con prestazioni corrispettive, quando uno dei contraenti non adempie le sue obbligazioni, "
            "l'altro può a sua scelta chiedere l'adempimento o la risoluzione del contratto, salvo, in ogni caso, il "
            "risarcimento del danno.\n"
            "2. La risoluzione può essere domandata anche quando il giudizio è stato promosso per ottenere "
            "l'adempimento; ma non può più chiedersi l'adempimento quando è stata domandata la risoluzione.")
ART_2043 = ("Qualunque fatto doloso o colposo, che cagiona ad altri un danno ingiusto, obbliga colui che ha commesso il "
            "fatto a risarcire il danno.")

E640 = ["con artifizi o raggiri", "inducendo taluno in errore",
        "procura a sé o ad altri un ingiusto profitto con altrui danno"]
E641 = ["dissimulando il proprio stato d'insolvenza", "contrae un'obbligazione col proposito di non adempierla"]
E1385 = ["una parte dà all'altra, a titolo di caparra, una somma di danaro",
         "inadempiente è invece la parte che l'ha ricevuta",
         "l'altra può recedere dal contratto ed esigere il doppio della caparra"]
E1453 = ["uno dei contraenti non adempie le sue obbligazioni"]
E2043 = ["fatto doloso o colposo", "danno ingiusto"]


def _norm(article, text, rubrica, elements, prefix):
    return {"article": article, "text": text, "rubrica": rubrica,
            "elements": [{"id": f"{prefix}{i}", "testo": t} for i, t in enumerate(elements, 1)]}


def _sents(texts, sistema=()):
    return [{"id": f"S{i}", "text": t, "source": "sistema" if i in sistema else "caso",
             "certainty": "assunto" if i in sistema else "accertato"} for i, t in enumerate(texts, 1)]


PENAL_SENTENCES = [
    "Paolo Sarti ha venduto online a Giulia Neri una bicicletta.",                                  # S1
    "Giulia Neri ha pagato 800 euro con un bonifico.",                                               # S2
    "Secondo la Procura, Paolo Sarti ha usato un falso annuncio per indurre in errore Giulia Neri.",   # S3
    "Giulia Neri afferma che la bicicletta non è mai arrivata.",                                     # S4
    "Paolo Sarti nega di aver mai pubblicato un falso annuncio.",                                    # S5
    "Il pubblico ministero procede per truffa ai sensi dell'art. 640 c.p.",                          # S6
    "Paolo Sarti dichiara di aver incassato il bonifico di Giulia Neri.",                            # S7
    "L'indagato dichiara di aver ricevuto il denaro.",                                               # S8
    "Parte assistita assunta: Paolo Sarti",                                                          # S9 (sistema)
]
PENAL_FRAME = {
    "parti": [{"nome": "Paolo Sarti", "ruolo_nel_fatto": "venditore", "lato": "cliente"},
              {"nome": "Giulia Neri", "ruolo_nel_fatto": "acquirente", "lato": "controparte"}],
    "cliente": {"nome": "Paolo Sarti", "perche": "«Assisto Paolo Sarti»"},
    "ruolo": {"valore": "imputato/indagato", "intake": "imputato", "perche": "«indagato per truffa»"},
    "rito": {"valore": "penale", "intake": "penale", "perche": "«Il pubblico ministero procede per truffa»"},
    "stadio": {"valore": "indagini", "intake": "indagini", "perche": "«indagini»"},
    "addebito_o_domanda": {"articolo": "640 c.p.", "descrizione": "truffa",
                           "frase": "«procede per truffa ai sensi dell'art. 640 c.p.»"},
    "quesiti": ["Quali rischi corre Paolo Sarti?"]}
PENAL_NORMS = [_norm("640 c.p.", ART_640, "Truffa", E640, "R"), _norm("641 c.p.", ART_641, "Insolvenza fraudolenta",
                                                                      E641, "Q"),
               _norm("62-bis c.p.", ART_62BIS, "Circostanze attenuanti generiche", [], "G")]

CIVIL_SENTENCES = [
    "Lucia Ferri ha firmato con Marco Vento un contratto preliminare di vendita di un appartamento.",   # S1
    "Lucia Ferri ha versato a Marco Vento 20.000 euro a titolo di caparra confirmatoria.",             # S2
    "Marco Vento sostiene di aver già restituito la caparra a Lucia Ferri.",                           # S3
    "Marco Vento ha venduto l'appartamento a un terzo prima del rogito.",                              # S4
    "Lucia Ferri dichiara di non aver mai ricevuto alcuna restituzione.",                              # S5
]
CIVIL_FRAME = {
    "parti": [{"nome": "Lucia Ferri", "ruolo_nel_fatto": "promissaria acquirente", "lato": "cliente"},
              {"nome": "Marco Vento", "ruolo_nel_fatto": "promittente venditore", "lato": "controparte"}],
    "cliente": {"nome": "Lucia Ferri", "perche": "«Assisto Lucia Ferri»"},
    "ruolo": {"valore": "attore", "intake": "attore", "perche": "«Assisto Lucia Ferri»"},
    "rito": {"valore": "civile", "intake": "civile", "perche": "«contratto preliminare»"},
    "addebito_o_domanda": {"articolo": "1385 c.c.", "descrizione": "recesso ed esigere il doppio della caparra",
                           "frase": "«caparra confirmatoria»"},
    "istituto": {"testo": "caparra confirmatoria", "frase": "«a titolo di caparra confirmatoria»"},
    "quesiti": ["Lucia Ferri può chiedere il doppio della caparra?"]}
CIVIL_NORMS = [_norm("1385 c.c.", ART_1385, "Caparra confirmatoria", E1385, "R"),
               _norm("1453 c.c.", ART_1453, "Risolubilità del contratto per inadempimento", E1453, "T"),
               _norm("2043 c.c.", ART_2043, "Risarcimento per fatto illecito", E2043, "U")]


def _penal(frame=PENAL_FRAME, norms=PENAL_NORMS):
    return A.make_input(frame, norms, _sents(PENAL_SENTENCES, sistema=(9,)), frame.get("quesiti") or [],
                        client="Paolo Sarti", client_roles=["imputato", "indagato"])


def _civil(frame=CIVIL_FRAME, norms=CIVIL_NORMS):
    return A.make_input(frame, norms, _sents(CIVIL_SENTENCES), frame.get("quesiti") or [], client="Lucia Ferri",
                        client_roles=["attore"])


ELEMENTS = {"640 c.p.": E640, "641 c.p.": E641, "1385 c.c.": E1385, "1453 c.c.": E1453, "2043 c.c.": E2043}


def eid(article, element):
    """v79.15.1 (brief item 1): the model cites an element by its E-id -- a schema element's text becomes its id
    («A640cp:E1»); any other text is passed as written (no E-id: rejected / not read, analysis_bad_eid)."""
    texts = ELEMENTS.get(article) or []
    if element in texts:
        return A.element_id(article, texts.index(element) + 1)
    return element


def _line(article, element, tesi="La linea.", cf_=("S1",), cn=None):
    return {"articolo": article, "elemento_in_gioco": eid(article, element), "tesi": tesi, "cita_fatti": list(cf_),
            "cita_norma": list(cn or [])}


def _el(article, element, esito, cf_=(), parte=None):
    return {"articolo": article, "elemento": eid(article, element) if article else element, "esito": esito,
            "cita_fatti": list(cf_), "parte_che_afferma": parte}


class _Model:
    """The model faked: the replies in order (a dict -> JSON; a string as is; an exception raised; a float -> sleep
    then the next); every call recorded."""

    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def __call__(self, messages, kwargs):
        self.seen.append({"messages": messages, "kwargs": kwargs})
        reply = self.replies.pop(0) if self.replies else {}
        while isinstance(reply, float):
            time.sleep(reply)
            reply = self.replies.pop(0) if self.replies else {}
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)


def _reply(principale, subordinata=None, elementi=(), affermazioni=()):
    return {"linea_principale": principale, "linea_subordinata": subordinata, "elementi": list(elementi),
            "affermazioni_di_diritto": list(affermazioni)}


GOOD_PENAL = _reply(_line("640 c.p.", E640[0], "L'accusa deve provare gli artifizi o raggiri.", ["S5"], ["A640cp.1"]),
                    elementi=[_el("640 c.p.", E640[0], "da_provare"), _el("640 c.p.", E640[1], "da_provare"),
                              _el("640 c.p.", E640[2], "da_provare")])


# --- 1. the input -----------------------------------------------------------------------------------------------------

def test_the_article_ids_are_the_fits_own_numbering_comma_then_period():
    got = ap.article_ids("1385 c.c.", ART_1385)
    assert [(p["id"], p["comma"]) for p in got] == [("A1385cc.1", "1"), ("A1385cc.2", "2"), ("A1385cc.3", "3")]
    assert [p["id"] for p in ap.article_ids("62-bis c.p.", ART_62BIS)] == ["A62biscp.1", "A62biscp.2"]
    assert ap.article_ids("62-bis c.p.", ART_62BIS)[1]["text"].startswith("Esse sono considerate")
    # an abbreviation never ends a period; two codes never share an id
    text = "Si applica l'art. 62 c.p. ai fatti. La pena è diminuita."
    assert [p["text"] for p in ap.article_ids("1 x", text)] == ["Si applica l'art. 62 c.p. ai fatti.",
                                                                 "La pena è diminuita."]
    assert ap.article_token("646 c.p.") != ap.article_token("646 c.c.")
    inp = _penal()
    fit_human = ap.build_messages([{"article": "640 c.p.", "text": ART_640, "rubrica": "Truffa"}],
                                  [{"id": "S1", "text": "x y z", "certainty": "accertato"}], "imputato")[1].content
    assert "[A640cp.1] Chiunque, con artifizi" in fit_human
    assert "[A640cp.1] Chiunque, con artifizi" in A.build_messages(inp)[1].content


def test_the_prompt_is_the_datas_with_the_verified_frame_the_facts_the_norms_and_the_questions():
    inp = _penal()
    system, human = [m.content for m in A.build_messages(inp)]
    assert system == A.prompt_system() and "linea_principale" in system and "affermazioni_di_diritto" in system
    assert len(A.prompt_hash()) == 12 and A.prompt_hash() == "b69eefcd0577"      # v79.15.1: elements by id
    assert human.startswith("QUADRO DEL CASO (verificato):\nCliente: Paolo Sarti\nRuolo del cliente: imputato/indagato")
    assert "Addebito o domanda: 640 c.p.; truffa" in human and "- Giulia Neri (controparte; acquirente)" in human
    assert "S3 [certo] Secondo la Procura" in human and "S8 [certo] L'indagato dichiara" in human
    assert "S9" not in human and "Parte assistita assunta" not in human          # the system's assumption: no fact
    assert "ARTICOLO: 62-bis c.p. (rimedio o parte generale: mai una linea)" in human
    assert "ARTICOLO: 640 c.p.\n" in human and "- [A640cp:E1] con artifizi o raggiri" in human
    assert "QUESITI DELL'UTENTE:\n- Quali rischi corre Paolo Sarti?" in human
    kwargs = A.request_kwargs()
    assert kwargs["temperature"] == 0 and kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    # a frame field that is not there (unverified: dropped by the frame's verification) is never sent
    no_rito = {k: v for k, v in PENAL_FRAME.items() if k != "rito"}
    assert "Rito:" not in A.build_messages(_penal(frame=no_rito))[1].content


def test_the_remedy_set_and_the_contractual_frame():
    for label in ("1218 c.c.", "1453 c.c.", "1458 c.c.", "40 c.p.", "43 c.p.", "59 c.p.", "61 c.p.", "62 c.p.",
                  "62-bis c.p.", "133 c.p."):
        assert A.is_remedy(label), label
    for label in ("62-ter c.p.", "63 c.p.", "640 c.p.", "1385 c.c.", "41 c.c.", "2043 c.c."):
        assert not A.is_remedy(label), label
    assert A.is_remedy("2043 c.c.", contractual=True)
    assert A.is_contractual(CIVIL_FRAME) and not A.is_contractual(PENAL_FRAME) and not A.is_contractual({})
    tort = {"rito": {"valore": "civile", "intake": "civile"},
            "addebito_o_domanda": {"articolo": "2043 c.c.", "descrizione": "danni da un incidente stradale",
                                   "frase": "«chiede i danni»"}}
    assert not A.is_contractual(tort)
    assert A.is_contractual(dict(tort, istituto={"testo": "inadempimento del conduttore"}))
    inp = _civil()
    assert inp["contractual"] and inp["charge"] == "1385 c.c."
    assert [n["article"] for n in inp["norms"] if n["remedy"]] == ["1453 c.c.", "2043 c.c."]
    # the frame's own verified article is never a remedy: the claim the line must carry
    claim = dict(CIVIL_FRAME, addebito_o_domanda={"articolo": "1453 c.c.", "descrizione": "risoluzione",
                                                  "frase": "«risoluzione»"})
    assert [n["article"] for n in _civil(frame=claim)["norms"] if n["remedy"]] == ["2043 c.c."]


# --- 2. the verifier --------------------------------------------------------------------------------------------------

def test_a_verified_civil_analysis_its_elements_and_its_law_statements():
    model = _Model(_reply(
        _line("1385 c.c.", E1385[1], "Marco Vento, che ha ricevuto la caparra, è inadempiente.", ["S2", "S4"],
              ["A1385cc.2"]),
        elementi=[_el("1385 c.c.", E1385[0], "provato", ["S2"]), _el("1385 c.c.", E1385[1], "provato", ["S4"])],
        affermazioni=[{"testo": "Se inadempiente è chi ha ricevuto la caparra, l'altra parte può esigerne il doppio.",
                       "cita_norma": ["A1385cc.2"]},
                      {"testo": "La caparra si restituisce sempre entro trenta giorni.", "cita_norma": []},
                      {"testo": "Il risarcimento è sempre dovuto.", "cita_norma": ["A1453cc.9"]}]))
    res = A.analyse(_civil(), call=model)
    assert res["status"] == "ok" and len(model.seen) == 1 and res["counters"]["analysis_calls"] == 1
    an = res["analysis"]
    lp = an["linea_principale"]
    assert lp["articolo"] == "1385 c.c." and lp["stato"] == "verificata" and lp["elemento_id"] == "R2"
    assert an["linea_subordinata"] is None and an["frame_charge"] == "1385 c.c." and not an["frame_charge_missing"]
    els = an["elementi"]["1385 c.c."]
    assert [(e["id"], e["esito"], e["origine"]) for e in els] == [("R1", "provato", "modello"),
                                                                   ("R2", "provato", "modello"),
                                                                   ("R3", "da_provare", "mancante")]
    assert [x["testo"] for x in an["affermazioni_di_diritto"]] == [
        "Se inadempiente è chi ha ricevuto la caparra, l'altra parte può esigerne il doppio."]
    assert an["affermazioni_di_diritto"][0]["articoli"] == ["1385 c.c."]
    c = res["counters"]
    assert c["law_statements_kept"] == 1 and c["law_statements_dropped"] == 2 and c["analysis_bad_aid"] == 1
    assert c["analysis_element_missing"] == 1 and c["analysis_reasked"] == 0
    assert an["rimedi"] == ["1453 c.c.", "2043 c.c."]


@pytest.mark.parametrize("bad,why", [
    (_line("640 c.p.", E640[0], "Tesi.", ["S1", "S12"], ["A640cp.1"]), "cita_fatti inesistenti: S12"),
    (_line("640 c.p.", E640[0], "Tesi.", ["S9"], ["A640cp.1"]), "cita_fatti inesistenti: S9"),     # sistema
    (_line("640 c.p.", E640[0], "Tesi.", ["S1"], ["A641cp.1"]), "cita_norma non dell'articolo: A641cp.1"),
    (_line("640 c.p.", E640[0], "Tesi.", ["S1"], ["A640cp.2"]), "cita_norma non dell'articolo: A640cp.2"),
    (_line("640 c.p.", "gli artifizi e i raggiri", "Tesi.", ["S1"], ["A640cp.1"]),      # v79.15.1: a text, no id
     "elemento_in_gioco «gli artifizi e i raggiri» non è l'id di un elemento"),
    (_line("640 c.p.", E640[0], "", ["S1"], ["A640cp.1"]), "tesi mancante"),
    (_line("640 c.p.", E640[0], "Tesi.", [], []), "nessun id citato"),
    (_line("999 c.p.", E640[0], "Tesi.", ["S1"]), "articolo non tra le norme applicabili"),
])
def test_a_line_with_a_bad_id_or_a_non_verbatim_element_is_rejected_and_re_asked_once(bad, why):
    model = _Model(_reply(bad), GOOD_PENAL)
    res = A.analyse(_penal(), call=model)
    assert res["status"] == "ok" and len(model.seen) == 2 and res["counters"]["analysis_reasked"] == 1
    assert res["rejected"][0]["kind"] == "linea_principale" and why in res["rejected"][0]["why"], res["rejected"]
    assert res["counters"]["analysis_line_rejected"] == 1
    assert res["analysis"]["linea_principale"]["articolo"] == "640 c.p."             # the re-ask's verified line
    reask = model.seen[1]["messages"]
    assert len(reask) == 4 and reask[2].content == json.dumps(_reply(bad), ensure_ascii=False)
    assert "respinta" in reask[3].content and why in reask[3].content


def test_an_element_is_read_by_its_id_only():
    """v79.15.1 (brief item 1, replaces the verbatim test): elemento_in_gioco and elemento are E-ids, verified by
    existence; a text -- even the element's own -- or an unknown id is never read (analysis_bad_eid); the printed text
    is the schema's."""
    assert A.parse_eid(" «A640cp:E2» ") == ("640cp", 2) and A.parse_eid("E3") == ("", 3)
    assert A.parse_eid("con artifizi o raggiri") is None and A.parse_eid("A640cp.1") is None
    model = _Model(_reply(_line("640 c.p.", "A640cp:E1", "Tesi.", ["S5"], ["A640cp.1"]), elementi=[
        _el("640 c.p.", "A640cp:E2", "escluso", ["S5"]),
        _el("640 c.p.", "induce in errore la vittima", "provato", ["S2"]),           # a paraphrase: never read
        _el("640 c.p.", E640[2], "provato", ["S2"]) | {"elemento": E640[2]},         # the text itself: no id
        _el(None, "A640cp:E9", "provato", ["S2"])]))                                  # past the last element
    res = A.analyse(_penal(), call=model)
    an = res["analysis"]
    assert an["linea_principale"]["elemento_in_gioco"] == E640[0] and an["linea_principale"]["eid"] == "A640cp:E1"
    assert [e["esito"] for e in an["elementi"]["640 c.p."]] == ["da_provare", "escluso", "da_provare"]
    assert [e["eid"] for e in an["elementi"]["640 c.p."]] == ["A640cp:E1", "A640cp:E2", "A640cp:E3"]
    assert res["counters"]["analysis_bad_eid"] == 3 and res["counters"]["analysis_element_missing"] == 2
    assert "analysis_element_not_verbatim" not in res["counters"]
    # a bare «E<n>» on the line's own article is read; another article's id never
    model = _Model(_reply(_line("640 c.p.", "E2", "Tesi.", ["S5"], ["A640cp.1"]),
                          elementi=[_el(None, "E3", "escluso", ["S5"]), _el(None, "A641cp:E1", "escluso", ["S5"])]))
    res = A.analyse(_penal(), call=model)
    assert res["analysis"]["linea_principale"]["elemento_in_gioco"] == E640[1]
    assert [e["esito"] for e in res["analysis"]["elementi"]["640 c.p."]] == ["da_provare", "da_provare", "escluso"]
    model = _Model(_reply(_line("640 c.p.", "A641cp:E1", "Tesi.", ["S5"], ["A640cp.1"])), GOOD_PENAL)
    res = A.analyse(_penal(), call=model)
    assert "non è l'id di un elemento dello schema di questo articolo (A640cp:E1…E3)" in res["rejected"][0]["why"]
    assert res["counters"]["analysis_bad_eid"] == 1 and len(model.seen) == 2


def test_provato_on_another_partys_or_an_authoritys_assertion_is_contestato_with_the_party():
    model = _Model(_reply(_line("640 c.p.", E640[0], "Tesi.", ["S5"], ["A640cp.1"]), elementi=[
        _el("640 c.p.", E640[0], "provato", ["S3"]),                                 # «Secondo la Procura, …»
        _el("640 c.p.", E640[1], "provato", ["S4"]),                                 # «Giulia Neri afferma …»
        _el("640 c.p.", E640[2], "provato", ["S2"], parte="Giulia Neri")]))          # the reply says who asserts it
    res = A.analyse(_penal(), call=model)
    els = {e["elemento"]: e for e in res["analysis"]["elementi"]["640 c.p."]}
    assert els[E640[0]]["esito"] == "contestato" and els[E640[0]]["parte_che_afferma"] == "la Procura"
    assert els[E640[1]]["esito"] == "contestato" and els[E640[1]]["parte_che_afferma"] == "Giulia Neri"
    assert els[E640[2]]["esito"] == "contestato" and els[E640[2]]["parte_che_afferma"] == "Giulia Neri"
    assert all(e["declassato"] == "provato su un'allegazione" for e in els.values())
    assert res["counters"]["analysis_provato_downgraded"] == 3
    # a certain fact among the support keeps it «provato»
    model = _Model(_reply(_line("640 c.p.", E640[0], "Tesi.", ["S5"], ["A640cp.1"]), elementi=[
        _el("640 c.p.", E640[2], "provato", ["S4", "S2"])]))
    els = {e["elemento"]: e for e in A.analyse(_penal(), call=model)["analysis"]["elementi"]["640 c.p."]}
    assert els[E640[2]]["esito"] == "provato"
    # an authority by its word: «L'accusa sostiene», «Il PM ritiene», «gli investigatori riferiscono»
    inp = _penal()
    for text, who in (("L'accusa sostiene che Paolo Sarti abbia mentito.", "L'accusa"),
                      ("Il PM ritiene che l'annuncio fosse falso.", "Il PM"),
                      ("Gli investigatori riferiscono che l'annuncio era falso.", "Gli investigatori"),
                      ("Giulia Neri, con una lettera del 3 marzo, sostiene di non aver ricevuto nulla.", "Giulia Neri"),
                      ("L'acquirente ha dichiarato di non aver ricevuto nulla.", "Giulia Neri")):
        assert A.asserting_party(text, inp) == who, text
    assert A.asserting_party("Giulia Neri ha pagato 800 euro con un bonifico.", inp) == ""
    assert A.asserting_party("Il vicino sostiene di aver visto tutto.", inp) == ""        # no party of the frame


def test_the_clients_own_statement_is_never_downgraded():
    model = _Model(_reply(_line("640 c.p.", E640[0], "Tesi.", ["S5"], ["A640cp.1"]), elementi=[
        _el("640 c.p.", E640[2], "provato", ["S7"]),                                 # «Paolo Sarti dichiara …»
        _el("640 c.p.", E640[1], "provato", ["S8"], parte="Paolo Sarti")]))           # «L'indagato dichiara …»
    res = A.analyse(_penal(), call=model)
    els = {e["elemento"]: e for e in res["analysis"]["elementi"]["640 c.p."]}
    assert els[E640[2]]["esito"] == "provato" and els[E640[1]]["esito"] == "provato"
    assert res["counters"]["analysis_provato_downgraded"] == 0
    inp = _penal()
    assert A.asserting_party(PENAL_SENTENCES[6], inp) == "" and A.asserting_party(PENAL_SENTENCES[7], inp) == ""
    assert A.asserting_party("Il cliente sostiene di aver spedito la bicicletta.", inp) == ""
    # 79.14.3's mark: a sentence alleging_party marked (alleged_by) is another party's assertion
    assert A.reported_by({"text": "Si dice che la bicicletta non è arrivata.", "alleged_by": "La vicina"}, inp) == \
        "La vicina"


def test_escluso_contestato_and_provato_need_an_existing_s_id():
    model = _Model(_reply(_line("640 c.p.", E640[0], "Tesi.", ["S5"], ["A640cp.1"]), elementi=[
        _el("640 c.p.", E640[0], "escluso", []),
        _el("640 c.p.", E640[1], "provato", []),
        _el("640 c.p.", E640[2], "contestato", ["S4", "S31"])]))
    res = A.analyse(_penal(), call=model)
    assert [e["esito"] for e in res["analysis"]["elementi"]["640 c.p."]] == ["da_provare"] * 3
    c = res["counters"]
    assert c["analysis_escluso_unsupported"] == 1 and c["analysis_provato_unsupported"] == 1
    assert c["analysis_bad_sid"] == 1
    model = _Model(_reply(_line("640 c.p.", E640[0], "Tesi.", ["S5"], ["A640cp.1"]), elementi=[
        _el("640 c.p.", E640[0], "escluso", ["S5"]), _el("640 c.p.", E640[1], "forse", ["S5"]),
        _el("640 c.p.", E640[2], "contestato", ["S4"])]))
    res = A.analyse(_penal(), call=model)
    els = res["analysis"]["elementi"]["640 c.p."]
    assert [e["esito"] for e in els] == ["escluso", "da_provare", "contestato"]
    assert els[2]["parte_che_afferma"] == "Giulia Neri" and res["counters"]["analysis_esito_invalid"] == 1


def test_a_remedy_article_as_the_line_is_rejected_re_asked_once_then_da_determinare():
    # v79.15.1: an istituto that is no rubrica -- «caparra confirmatoria» is 1385 c.c.'s (the rubrica fallback)
    no_claim = {k: v for k, v in CIVIL_FRAME.items() if k != "addebito_o_domanda"} | {
        "istituto": {"testo": "caparra del preliminare", "frase": "«caparra»"}}
    remedy = _reply(_line("1453 c.c.", E1453[0], "Si chiede la risoluzione.", ["S4"], ["A1453cc.1"]))
    model = _Model(remedy, remedy)
    res = A.analyse(_civil(frame=no_claim), call=model)
    assert res["status"] == "ok" and len(model.seen) == 2
    assert "rimedio o parte generale, mai una linea" in model.seen[1]["messages"][3].content
    lp = res["analysis"]["linea_principale"]
    # v79.19.0 (item 6): «da determinare» never while a fitted article exists -- the best-tier fitted norm (1385 c.c.,
    # the first outside the remedy set) leads with its elements da provare; no element in play, no tesi of the model
    assert lp["stato"] == A.FIT_LINE == "norma_applicabile" and lp["articolo"] == "1385 c.c."
    assert lp["tesi"] == "Caparra confirmatoria: elementi da provare" and lp["eid"] is None
    assert res["counters"]["analysis_line_fallback_fit"] == 1 and res["counters"]["analysis_line_undetermined"] == 0
    assert set(res["analysis"]["elementi"]) == {"1385 c.c."}       # its elements, every one «da_provare»
    assert all(e["esito"] == "da_provare" for e in res["analysis"]["elementi"]["1385 c.c."])
    c = res["counters"]
    assert c["analysis_line_rejected"] == 2 and c["analysis_line_undetermined"] == 0 and c["analysis_calls"] == 2
    # no fitted norm at all: «da determinare»
    bare = A.analyse(_civil(frame=no_claim, norms=[]), call=_Model(remedy, remedy))
    assert bare["status"] == "no_input"
    assert A.view(bare, _civil(frame=no_claim, norms=[]))["analysis"]["linea_principale"]["stato"] == "da_determinare"
    # 2043 c.c. on a contractual frame is a remedy too; the re-ask's own verified line is taken
    model = _Model(_reply(_line("2043 c.c.", E2043[1], "Danno.", ["S4"], ["A2043cc.1"])),
                   _reply(_line("1385 c.c.", E1385[1], "Inadempiente chi ha ricevuto.", ["S4"], ["A1385cc.2"])))
    res = A.analyse(_civil(frame=no_claim), call=model)
    assert res["analysis"]["linea_principale"]["articolo"] == "1385 c.c." and len(model.seen) == 2
    # a remedy as the subordinate line: after the re-ask, none
    sub = _reply(_line("1385 c.c.", E1385[1], "Tesi.", ["S4"], ["A1385cc.2"]),
                 _line("1453 c.c.", E1453[0], "Subordinata.", ["S4"], ["A1453cc.1"]))
    model = _Model(sub, sub)
    res = A.analyse(_civil(), call=model)
    assert res["analysis"]["linea_subordinata"] is None and res["counters"]["analysis_line_rejected"] == 2
    assert res["analysis"]["linea_principale"]["articolo"] == "1385 c.c."


def test_the_frames_charge_missing_is_re_asked_once_then_its_article_leads_with_every_element_da_provare():
    other = _reply(_line("641 c.p.", E641[0], "Non vi è dissimulazione.", ["S2"], ["A641cp.1"]), elementi=[
        _el("641 c.p.", E641[0], "escluso", ["S2"]), _el("640 c.p.", E640[0], "provato", ["S1"])])
    model = _Model(other, other)
    res = A.analyse(_penal(), call=model)
    assert len(model.seen) == 2 and "l'addebito o la domanda del quadro (art. 640 c.p.)" in \
        model.seen[1]["messages"][3].content
    an = res["analysis"]
    lp, ls = an["linea_principale"], an["linea_subordinata"]
    # v79.19.1 (item 1): the step's verified line leads; the frame's charge, missing but not refused, is the subordinate
    # with every element da provare (the «linea non verificata» label never beside a verified line)
    assert lp["articolo"] == "641 c.p." and lp["stato"] == "verificata"
    assert ls["articolo"] == "640 c.p." and ls["stato"] == "addebito_del_quadro" and ls["tesi"] == "" \
        and ls["posizione"] == "subordinata"
    assert [(e["esito"], e["origine"]) for e in an["elementi"]["640 c.p."]] == [("da_provare", "quadro")] * 3
    assert [e["esito"] for e in an["elementi"]["641 c.p."]] == ["escluso", "da_provare"]
    assert an["frame_charge_overridden"] and not an["frame_charge_missing"]
    assert res["counters"]["analysis_frame_charge_missing"] == 1 == res["counters"]["analysis_frame_charge_overridden"]
    assert res["counters"]["analysis_line_undetermined"] == 0
    assert A.LINE_UNVERIFIED_LABEL not in A.summary_text(A.view(res, _penal()))
    # the re-ask carrying both reasons at once -- still one re-ask, at most two calls
    both = _reply(_line("62-bis c.p.", "x", "Attenuanti.", ["S5"]))
    model = _Model(both, both, both)
    res = A.analyse(_penal(), call=model)
    reasons = model.seen[1]["messages"][3].content
    assert len(model.seen) == 2 and "62-bis c.p." in reasons and "art. 640 c.p." in reasons
    assert res["analysis"]["linea_principale"]["stato"] == "addebito_del_quadro"
    assert res["analysis"]["linea_subordinata"] is None and res["counters"]["analysis_calls"] == 2
    # the charge as the subordinate line is enough
    sub = _reply(_line("641 c.p.", E641[0], "Tesi.", ["S2"], ["A641cp.1"]),
                 _line("640 c.p.", E640[0], "Tesi.", ["S5"], ["A640cp.1"]))
    model = _Model(sub)
    res = A.analyse(_penal(), call=model)
    assert len(model.seen) == 1 and not res["analysis"]["frame_charge_missing"]


def test_a_charge_that_is_no_fitted_norm_is_not_re_asked_the_verified_line_leads_the_charge_is_the_subordinate():
    norms = [n for n in PENAL_NORMS if n["article"] != "640 c.p."]
    model = _Model(_reply(_line("641 c.p.", E641[0], "Tesi.", ["S2"], ["A641cp.1"])))
    res = A.analyse(_penal(norms=norms), call=model)
    an = res["analysis"]
    assert len(model.seen) == 1 and an["linea_principale"]["articolo"] == "641 c.p."     # v79.19.1 (item 1)
    assert an["linea_subordinata"]["articolo"] == "640 c.p." and an["linea_subordinata"]["stato"] == A.FRAME_LINE
    assert an["elementi"]["640 c.p."] == [] and res["counters"]["analysis_frame_charge_missing"] == 1
    assert res["counters"]["analysis_frame_charge_overridden"] == 1
    # no verified line at all: the text's charge leads as the frame's unverified line (79.15.2)
    model = _Model(_reply(_line("62-bis c.p.", "x", "Attenuanti.", ["S5"])))
    res = A.analyse(_penal(norms=norms), call=model)
    assert res["analysis"]["linea_principale"]["articolo"] == "640 c.p."
    assert res["analysis"]["linea_principale"]["stato"] == A.FRAME_LINE


# --- 3. nothing from a failed step ------------------------------------------------------------------------------------

@pytest.mark.parametrize("reply,status", [(RuntimeError("connection refused"), "failed"), ("non è JSON", "malformed"),
                                          ("[1, 2]", "malformed")])
def test_a_failed_call_or_no_json_object_gives_nothing(reply, status):
    # v79.19.8 (item 5, row 101): a malformed first reply is asked again once (the same input) -- malformed twice: nothing
    model = _Model(reply, reply)
    res = A.analyse(_penal(), call=model)
    calls = 2 if status == "malformed" else 1
    assert res["status"] == status and res["analysis"] is None and len(model.seen) == calls
    assert res["counters"]["analysis_failed"] == 1 and res["counters"]["analysis_calls"] == calls
    assert res["counters"]["analysis_retried"] == int(status == "malformed")


def test_the_deadline_cuts_the_call_and_nothing_is_printed(monkeypatch):
    cfg = dict(A.config(), deadline_s=0.3, deadline_per_article_s=0)      # v79.19.1 (item 2): the scaling off
    monkeypatch.setattr(A, "config", lambda: cfg)
    model = _Model(1.0, GOOD_PENAL)
    t0 = time.time()
    res = A.analyse(_penal(), call=model)
    assert time.time() - t0 < 0.9
    assert res["status"] == "timeout" and res["analysis"] is None
    assert res["counters"]["analysis_timeouts"] == 1 and res["counters"]["analysis_failed"] == 1


def test_a_failed_re_ask_keeps_the_first_replys_verification_with_its_fallbacks():
    no_claim = {k: v for k, v in CIVIL_FRAME.items() if k != "addebito_o_domanda"} | {     # v79.15.1: no rubrica
        "istituto": {"testo": "caparra del preliminare", "frase": "«caparra»"}}
    model = _Model(_reply(_line("1453 c.c.", E1453[0], "Risoluzione.", ["S4"], ["A1453cc.1"])),
                   RuntimeError("down"))
    res = A.analyse(_civil(frame=no_claim), call=model)
    assert res["status"] == "ok" and res["analysis"]["linea_principale"]["stato"] == A.FIT_LINE   # v79.19.0 (item 6)
    assert [c["status"] for c in res["calls"]] == ["ok", "failed"] and res["counters"]["analysis_calls"] == 2


def test_the_conftest_refuses_the_default_model_call():
    res = A.analyse(_penal())
    assert res["status"] == "failed" and res["analysis"] is None and "no real model call" in res["error"]


def test_no_norm_or_no_sentence_sends_nothing():
    model = _Model(GOOD_PENAL)
    assert A.analyse(_penal(norms=[]), call=model)["status"] == "no_input" and model.seen == []


# --- 4. run(state) ----------------------------------------------------------------------------------------------------

def _state():
    rows = [{"_article": ("640", "c.p."), "s": {"name": "640.0.0", "plain_text": ART_640, "titolo_articolo": "Truffa"},
             "d": {"id": "LEGAL_DOC::cp", "name": "Codice Penale 2026"}},
            {"_article": ("62-bis", "c.p."), "s": {"name": "62-bis.0.0", "plain_text": ART_62BIS,
                                                   "titolo_articolo": "Circostanze attenuanti generiche"},
             "d": {"id": "LEGAL_DOC::cp", "name": "Codice Penale 2026"}}]
    records = {"640 c.p.": {"article": "640 c.p.", "rubrica": "Truffa", "requisites": [
                   {"id": f"R{i}", "text": t, "status": "da provare"} for i, t in enumerate(E640, 1)]},
               "62-bis c.p.": {"article": "62-bis c.p.", "rubrica": "Circostanze", "requisites": []},
               "641 c.p.": {"article": "641 c.p.", "dropped": "non scrivibile", "requisites": []}}
    return {"records": records, "rows": [flow._compact_row(r) for r in rows],
            "sentences": _sents(PENAL_SENTENCES, sistema=(9,)), "client": {"name": "Paolo Sarti"},
            "intake": {"role": "imputato", "questions": ["Quali rischi corre Paolo Sarti?"]},
            "text": " ".join(PENAL_SENTENCES[:8]), "request": "", "analysis": {},
            "case_frame": {"frame": PENAL_FRAME, "status": "ok"}}


def test_run_with_the_flag_off_calls_nothing():
    model = _Model(GOOD_PENAL)
    res = A.run(_state(), call=model)
    assert res["status"] == "off" and res["analysis"] is None and model.seen == [] and not A.enabled()


def test_run_reads_the_state_and_logs_the_analysis_step_line(monkeypatch):
    monkeypatch.setenv("ANALYSIS_STEP", "true")
    monkeypatch.setenv("CASE_FRAME", "true")
    events = []
    from src.rag import verbose_logger
    monkeypatch.setattr(verbose_logger, "vlog", lambda event, data=None, duration_ms=None: events.append((event, data)))
    model = _Model(GOOD_PENAL)
    res = A.run(_state(), call=model)
    assert res["status"] == "ok" and res["analysis"]["linea_principale"]["articolo"] == "640 c.p."
    human = model.seen[0]["messages"][1].content
    assert "ARTICOLO: 640 c.p.\n" in human and "62-bis c.p. (rimedio o parte generale" in human
    assert "641 c.p." not in human and "[A640cp.1]" in human and "- [A640cp:E1] con artifizi o raggiri" in human
    assert "QUADRO DEL CASO (verificato):" in human and "S9" not in human
    line = [d for e, d in events if e == "analysis_step"]
    assert len(line) == 1 and line[0]["prompt_hash"] == A.prompt_hash() and line[0]["status"] == "ok"
    assert line[0]["lines"][0]["articolo"] == "640 c.p." and line[0]["counters"]["analysis_calls"] == 1
    assert set(A.COUNTERS) <= set(line[0]["counters"]) and len(line[0]["calls"]) == 1
    # CASE_FRAME off: no frame read -- the facts, the norms and the questions only
    monkeypatch.setenv("CASE_FRAME", "false")
    model = _Model(GOOD_PENAL)
    A.run(_state(), call=model)
    assert "QUADRO DEL CASO" not in model.seen[0]["messages"][1].content


def test_a_fit_removed_norm_is_no_input(monkeypatch):
    monkeypatch.setenv("APPLICABILITY_FIT", "true")
    state = _state()
    state["fit_removed"] = [{"article": "62-bis c.p."}]
    assert [n["article"] for n in A.inputs(state)["norms"]] == ["640 c.p."]


# --- 5. the fit by ids, the protected articles ------------------------------------------------------------------------

def _fit(item, norm=None, facts=None):
    norm = norm or {"article": "640 c.p.", "text": ART_640, "rubrica": "Truffa"}
    facts = facts or [{"id": s["id"], "text": s["text"], "certainty": s["certainty"]}
                      for s in _sents(PENAL_SENTENCES, sistema=(9,))]
    res = ap.judge([norm], facts, "imputato", call=lambda m, k: json.dumps({"giudizi": [item]}, ensure_ascii=False))
    return res, res["judgments"][0]


def test_the_fit_si_cites_the_articles_periods_and_the_no_the_facts():
    base = {"articolo": "640 c.p.", "ambito": "fattispecie", "condizione": ""}
    res, rec = _fit(dict(base, fit="si", perche="Paolo Sarti ha venduto la bicicletta (S1) e ha usato un falso "
                                                "annuncio (S3): artifizi o raggiri.", cita_norma=["A640cp.1"]))
    assert rec["verified"] and rec["passage"] == ART_640 and rec["facts"] == ["S1", "S3"]
    res, rec = _fit(dict(base, fit="si", perche="Paolo Sarti ha venduto la bicicletta (S1).", cita_norma=["A646cp.1"]))
    assert not rec["verified"] and res["counters"]["fit_unverified_ids"] == 1
    res, rec = _fit(dict(base, fit="no", perche="Giulia Neri ha pagato con un bonifico (S2).", cita_fatti=["S2"]))
    assert rec["verified"] and rec["fact_quotes"] == [PENAL_SENTENCES[1]] and rec["passage"] == ""
    res, rec = _fit(dict(base, fit="no", perche="Paolo Sarti è assistito (S9).", cita_fatti=["S9"]))
    assert not rec["verified"] and res["counters"]["fit_no_unverified_ids"] == 1      # the system's assumption
    res, rec = _fit(dict(base, fit="no", perche="Giulia Neri ha pagato (S2).", cita_fatti="S2"))   # one string
    assert rec["verified"] and rec["cita_fatti"] == ["S2"]


def test_no_string_matcher_is_left_in_the_fit():
    for name in ("fact_quote_at", "verify_no", "cited_fact", "FACT_FIELD", "FACT_QUOTE_MIN_WORDS", "_crosses_sentence",
                 "_rests_on_passage", "_comma_of"):
        assert not hasattr(ap, name), name
    assert not hasattr(flow, "fact_source")
    assert "fatto_citato" not in ap.prompt_system() and ap.prompt_hash() == "1c71b3e09bcd"


def test_the_frames_verified_charge_is_protected_from_a_fit_no(monkeypatch):
    import test_v79_14_2 as T
    monkeypatch.setenv("APPLICABILITY_FIT", "true")
    sents = [dict(f, source="caso") for f in T.FACTS]
    reply = {"giudizi": [T._item("1439 c.c.", "no", "«Il pubblico ministero procede per truffa» (S3).",
                                 cf=["S3"])]}
    monkeypatch.setattr(ap, "_default_call", lambda m, k: json.dumps(reply, ensure_ascii=False))
    rows = [T._row("1439", "c.c.", T.ART_1439, "Dolo")]

    def state():
        return T._flow_state(T.CASE_TEXT, [dict(s) for s in sents], {"1439 c.c.": {
            "article": "1439 c.c.", "rubrica": "Dolo", "requisites": [T._req(2, "i raggiri", "da provare")]}})
    plain = state()                                         # no frame: the verified «no» removes it
    assert flow.applicability_fit(plain, T.IMPUTATO, rows, set(), []) == {"1439 c.c."}
    assert plain["block_counters"]["fit_protected_kept"] == 0
    monkeypatch.setenv("CASE_FRAME", "true")
    framed = state()
    framed["case_frame"] = {"frame": {"rito": {"valore": "civile", "intake": "civile"}, "addebito_o_domanda": {
        "articolo": "1439 c.c.", "descrizione": "annullamento per dolo", "frase": "«dolo»"}}}
    assert "1439 c.c." in flow.fit_protected(framed, T.IMPUTATO)
    assert flow.applicability_fit(framed, T.IMPUTATO, rows, set(), []) == set()
    assert "1439 c.c." in framed["records"] and not framed.get("fit_removed")
    c = framed["block_counters"]
    assert c["fit_protected_kept"] == 1 and c["fit_removed"] == 0 and c["fit_no_verified"] == 1
    # a user-named article: protected as before (79.13.0), counted the same way
    named = state()
    named["text"] += " Si applica l'art. 1439 c.c.?"
    monkeypatch.setenv("CASE_FRAME", "false")
    assert flow.applicability_fit(named, T.IMPUTATO, rows, set(), []) == set()
    assert named["block_counters"]["fit_protected_kept"] == 1
