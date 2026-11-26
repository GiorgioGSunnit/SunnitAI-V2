"""v79.14.0 build A -- the case frame (CASE_FRAME, src/rag/case_frame.py), the model faked, on synthetic request texts
(invented names and facts; never a real, eval or tester case):

1. the call: one call, thinking off, temperature 0, the full text, the prompt from the data and hashed; a failed call,
   a reply that is no JSON object, a call over the deadline: every field to the fallback, never a crash
2. the verification of each field: the quotes found or not (stem tolerant; a quote altered in meaning fails as in
   79.13), a cliente or a ruolo with no quote never accepted, a cliente listed as controparte rejected, the penal
   article rule, the questions verbatim and in order
3. the brief's five cases at the module level
4. the counters, the fallback API, the frame as the 79.13 fit judgment's context, the flag, the nightly
"""

import json
import os
import sys
import time
from pathlib import Path

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

from src.rag import applicability as ap  # noqa: E402
from src.rag import case_frame as cf  # noqa: E402
from src.rag import intake  # noqa: E402


@pytest.fixture(autouse=True)
def _no_model(monkeypatch):
    def refuse(messages, kwargs):
        raise AssertionError("no real model call in a test")
    monkeypatch.setattr(cf, "_default_call", refuse)
    monkeypatch.setattr(ap, "_default_call", refuse)


def _model(reply):
    """A fake model answering `reply` (a dict: as JSON; a str: as it is), recording what it was sent."""
    seen = []

    def call(messages, kwargs):
        seen.append((messages, kwargs))
        return reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
    call.seen = seen
    return call


def _read(text, reply, **kw):
    return cf.read(text, call=_model(reply), **kw)


# --- synthetic request texts (invented) -------------------------------------------------------------------------------

DEBTOR = ("Assisto la società Orsa S.r.l., che ha stipulato con Delfino S.p.A. un contratto di fornitura di componenti. "
          "Delfino pretende ora il pagamento di una penale di 18.000 euro per un ritardo nella consegna e ha comunicato "
          "la risoluzione del contratto per inadempimento. Orsa sostiene di aver consegnato i componenti nei termini "
          "pattuiti. Quali strategie difensive può adottare Orsa? La penale è riducibile?")
DIRECTOR = ("Il mio cliente, Marco Lince, amministratore della società Gabbiano S.r.l., ha ricevuto un avviso di "
            "garanzia: la Procura indaga per infedeltà patrimoniale per una cessione di beni sociali a una società "
            "collegata a un prezzo inferiore al valore, che secondo il socio di minoranza è stata ottenuta con raggiri. "
            "Quali sono i rischi penali per il signor Lince?")
TWO_SIDES = ("Anna Pino ha prestato 9.000 euro a Bruno Faggio con una scrittura privata; Faggio non ha restituito la "
             "somma alla scadenza e sostiene che si trattava di una donazione. Assisto Anna Pino. Come può recuperare "
             "la somma?")
A_CARICO = ("Carla Ulivo ha subito un'aggressione davanti a un locale da parte di Dario Cedro, riportando lesioni "
            "guaribili in trenta giorni. Quali reati sono configurabili a carico di Dario Cedro? Carla Ulivo può "
            "costituirsi parte civile?")
NO_CLIENT = ("Tra la società Volpe S.r.l. e la società Lupo S.r.l. è sorta una controversia sul pagamento di alcune "
             "fatture per lavori di manutenzione. Si chiede una valutazione della vicenda.")


def _debtor_reply(**over):
    reply = {
        "parti": [{"nome": "Orsa S.r.l.", "ruolo_nel_fatto": "ha consegnato i componenti", "lato": "cliente"},
                  {"nome": "Delfino S.p.A.", "ruolo_nel_fatto": "pretende il pagamento di una penale",
                   "lato": "controparte"}],
        "cliente": {"nome": "Orsa S.r.l.", "perche": "«Assisto la società Orsa S.r.l.»"},
        "ruolo": {"valore": "convenuto", "perche": "«Delfino pretende ora il pagamento di una penale»"},
        "rito": {"valore": "civile", "perche": "«un contratto di fornitura di componenti»"},
        "stadio": {"valore": "pre_filing", "perche": "«ha comunicato la risoluzione del contratto per inadempimento»"},
        "addebito_o_domanda": {"articolo": "1382 c.c.", "descrizione": "pagamento della penale per il ritardo",
                               "frase": "«Delfino pretende ora il pagamento di una penale di 18.000 euro»"},
        "istituto": {"testo": "contratto di fornitura", "frase": "«stipulato con Delfino S.p.A. un contratto di "
                                                                 "fornitura»"},
        "quesiti": ["Quali strategie difensive può adottare Orsa?", "La penale è riducibile?"]}
    reply.update(over)
    return reply


def _director_reply(articolo="2634 c.c.", rito="penale"):
    return {
        "parti": [{"nome": "Marco Lince", "ruolo_nel_fatto": "amministratore", "lato": "cliente"},
                  {"nome": "Procura", "ruolo_nel_fatto": "indaga", "lato": "autorità"}],
        "cliente": {"nome": "Marco Lince", "perche": "«Il mio cliente, Marco Lince»"},
        "ruolo": {"valore": "indagato", "perche": "«la Procura indaga per infedeltà patrimoniale»"},
        "rito": {"valore": rito, "perche": "«ha ricevuto un avviso di garanzia»"},
        "stadio": {"valore": "indagini", "perche": "«la Procura indaga»"},
        "addebito_o_domanda": {"articolo": articolo, "descrizione": "infedeltà patrimoniale",
                               "frase": "«la Procura indaga per infedeltà patrimoniale»"},
        "istituto": None,
        "quesiti": ["Quali sono i rischi penali per il signor Lince?"]}


# --- 1. the call ------------------------------------------------------------------------------------------------------

def test_one_call_thinking_off_temperature_zero_the_full_text_the_prompt_from_the_data():
    model = _model(_debtor_reply())
    res = cf.read(DEBTOR, call=model)
    assert len(model.seen) == 1
    messages, kwargs = model.seen[0]
    assert kwargs["temperature"] == 0
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert messages[0].content == cf.prompt_system() == "\n".join(cf.config()["prompt_system"])
    assert DEBTOR in messages[1].content
    assert res["prompt_hash"] == cf.prompt_hash() and len(res["prompt_hash"]) == 12
    assert res["status"] == cf.OK and res["counters"]["frame_failed"] == 0


@pytest.mark.parametrize("reply", ["non è JSON", "[1, 2, 3]", "", "<think>{\"cliente\": 1}</think>"])
def test_a_reply_that_is_no_json_object_leaves_every_field_to_the_fallback(reply):
    res = _read(DEBTOR, reply)
    assert res["status"] == cf.MALFORMED
    assert res["fallback"] == list(cf.FIELDS) and res["verified"] == []
    assert res["counters"]["frame_failed"] == 1 and res["counters"]["frame_fields_verified"] == 0
    assert cf.intake_view(res)["role"] is None and cf.fit_context(res) == ""


def test_a_failed_call_never_crashes():
    def boom(messages, kwargs):
        raise RuntimeError("connection refused")
    res = cf.read(DEBTOR, call=boom)
    assert res["status"] == cf.FAILED and "connection refused" in res["error"]
    assert res["fallback"] == list(cf.FIELDS) and res["counters"]["frame_failed"] == 1


def test_a_call_over_the_deadline_is_not_waited_for(monkeypatch):
    monkeypatch.setitem(cf.config(), "deadline_s", 0.2)

    def slow(messages, kwargs):
        time.sleep(1.0)
        return json.dumps(_debtor_reply())
    t0 = time.time()
    res = cf.read(DEBTOR, call=slow)
    assert time.time() - t0 < 0.9
    assert res["status"] == cf.TIMEOUT and res["fallback"] == list(cf.FIELDS)
    assert res["counters"]["frame_failed"] == 1


def test_no_text_no_call():
    model = _model(_debtor_reply())
    res = cf.read("   ", call=model)
    assert model.seen == [] and res["status"] == cf.NO_TEXT and res["counters"]["frame_failed"] == 0


def test_a_field_of_the_wrong_shape_is_dropped_alone():
    res = _read(DEBTOR, _debtor_reply(parti="Orsa e Delfino", quesiti={"a": 1}, istituto=7))
    assert "parti" in res["unverified"] and "istituto" in res["unverified"]
    assert res["frame"]["parti"] == [] and res["frame"]["istituto"] is None
    assert res["frame"]["rito"]["valore"] == "civile"          # the other fields stand
    assert res["frame"]["cliente"] is None                      # no verified party: no client


# --- 2. the verification of each field --------------------------------------------------------------------------------

def test_the_whole_frame_verified_mapped_onto_intakes_vocabulary():
    res = _read(DEBTOR, _debtor_reply())
    f = res["frame"]
    assert res["verified"] == list(cf.FIELDS) and res["fallback"] == [] and res["unverified"] == {}
    assert f["cliente"]["nome"] == "Orsa S.r.l." and f["cliente"]["lato"] == "cliente"
    assert f["ruolo"] == {"valore": "convenuto", "intake": "convenuto",
                          "perche": "«Delfino pretende ora il pagamento di una penale»",
                          "citazione": ["Delfino pretende ora il pagamento di una penale"]}
    # v79.19.2 (item 2): the text names no article -- the reply's 1382 c.c. is a hypothesis, never the verified charge
    assert f["addebito_o_domanda"]["articolo"] is None and f["addebito_o_domanda"]["ipotesi"] == "1382 c.c."
    assert f["quesiti"] == ["Quali strategie difensive può adottare Orsa?", "La penale è riducibile?"]
    assert res["counters"]["frame_fields_verified"] == 8
    assert all(res["counters"][f"frame_unverified.{x}"] == 0 for x in cf.FIELDS)


def test_a_quote_is_found_stem_tolerant_as_the_fit_checker_finds_it():
    # «pretendere» for «pretende», «consegnati» for «consegnato», the text's article left out: the same words by stem
    reply = _debtor_reply(ruolo={"valore": "convenuto", "perche": "«Delfino pretendere ora pagamento di una penale»"})
    res = _read(DEBTOR, reply)
    assert res["frame"]["ruolo"]["valore"] == "convenuto"
    assert cf.found_at(DEBTOR, "Orsa sostiene di aver consegnati i componenti") is not None
    assert cf.found_at(DEBTOR, "Orsa sostiene di aver consegnato i componenti") == \
        ap.passage_at(DEBTOR, "Orsa sostiene di aver consegnato i componenti")


@pytest.mark.parametrize("quote", [
    "«Delfino pretende ora il pagamento di una penale di 19.000 euro»",      # a number changed
    "«Delfino rinuncia al pagamento di una penale»",                          # a word changed
    "«Delfino pretende ora il pagamento o la penale»",                        # «o» for «di una»
    "Delfino chiede i danni",                                                 # a paraphrase, no quote marks
])
def test_a_quote_altered_in_meaning_is_not_found_and_the_field_is_dropped(quote):
    res = _read(DEBTOR, _debtor_reply(ruolo={"valore": "convenuto", "perche": quote}))
    assert res["frame"]["ruolo"] is None and "ruolo" in res["fallback"]
    assert res["counters"]["frame_unverified.ruolo"] == 1
    assert res["log"][0]["field"] == "ruolo" and "citazione non nel testo" in res["log"][0]["reason"]


def test_a_quote_cut_after_a_negation_says_the_opposite_and_fails():
    text = "Assisto Elio Sasso. Il contratto non è stato risolto dal fornitore. Come devo procedere?"
    reply = {"parti": [{"nome": "Elio Sasso", "lato": "cliente"}],
             "cliente": {"nome": "Elio Sasso", "perche": "«Assisto Elio Sasso»"},
             "istituto": {"testo": "contratto", "frase": "«è stato risolto dal fornitore»"}}
    res = _read(text, reply)
    assert res["frame"]["istituto"] is None and res["counters"]["frame_unverified.istituto"] == 1


def test_a_cliente_without_a_quote_is_never_accepted():
    for why in ("", None, "il testo lo indica"):
        res = _read(DEBTOR, _debtor_reply(cliente={"nome": "Orsa S.r.l.", "perche": why}))
        assert res["frame"]["cliente"] is None, why
        assert res["counters"]["frame_unverified.cliente"] == 1
        assert "cliente" in res["fallback"]


def test_a_ruolo_without_a_quote_is_never_accepted():
    res = _read(DEBTOR, _debtor_reply(ruolo={"valore": "convenuto", "perche": ""}))
    assert res["frame"]["ruolo"] is None and res["counters"]["frame_unverified.ruolo"] == 1
    res = _read(DEBTOR, _debtor_reply(ruolo="convenuto"))
    assert res["frame"]["ruolo"] is None and res["counters"]["frame_unverified.ruolo"] == 1


def test_a_value_outside_the_vocabulary_is_dropped():
    res = _read(DEBTOR, _debtor_reply(rito={"valore": "commerciale", "perche": "«un contratto di fornitura»"}))
    assert res["frame"]["rito"] is None and "valore fuori schema" in res["unverified"]["rito"][0]


def test_a_cliente_listed_as_controparte_is_rejected():
    res = _read(DEBTOR, _debtor_reply(cliente={"nome": "Delfino S.p.A.", "perche": "«Delfino pretende ora il "
                                                                                     "pagamento di una penale»"}))
    assert res["frame"]["cliente"] is None
    assert "controparte" in res["unverified"]["cliente"][0]


def test_a_cliente_listed_twice_once_as_controparte_is_rejected():
    reply = _debtor_reply()
    reply["parti"].append({"nome": "Orsa", "lato": "controparte"})
    res = _read(DEBTOR, reply)
    assert res["frame"]["cliente"] is None and "controparte" in res["unverified"]["cliente"][0]


def test_a_cliente_not_among_the_parties_or_not_in_the_text_is_rejected():
    res = _read(DEBTOR, _debtor_reply(parti=[{"nome": "Delfino S.p.A.", "lato": "controparte"}]))
    assert res["frame"]["cliente"] is None and "non è tra le parti" in res["unverified"]["cliente"][0]
    res = _read(DEBTOR, _debtor_reply(cliente={"nome": "Volpe S.r.l.", "perche": "«Assisto la società Orsa S.r.l.»"}))
    assert res["frame"]["cliente"] is None and "nome non nel testo" in res["unverified"]["cliente"][0]


def test_a_party_whose_name_is_not_in_the_text_or_whose_lato_is_unknown_is_dropped_alone():
    reply = _debtor_reply()
    reply["parti"] += [{"nome": "Gru S.r.l.", "lato": "terzo"}, {"nome": "Delfino", "lato": "nemico"}]
    res = _read(DEBTOR, reply)
    assert [p["nome"] for p in res["frame"]["parti"]] == ["Orsa S.r.l.", "Delfino S.p.A."]
    assert res["counters"]["frame_unverified.parti"] == 2
    assert res["frame"]["cliente"]["nome"] == "Orsa S.r.l."


def test_a_partys_role_in_the_facts_in_words_not_in_the_text_is_blanked():
    reply = _debtor_reply()
    reply["parti"][1]["ruolo_nel_fatto"] = "creditore che ha agito in via monitoria"
    res = _read(DEBTOR, reply)
    assert res["frame"]["parti"][1]["ruolo_nel_fatto"] == ""
    assert res["counters"]["frame_unverified.parti.ruolo_nel_fatto"] == 1
    assert res["counters"]["frame_unverified.parti"] == 0 and len(res["frame"]["parti"]) == 2


def test_the_questions_verbatim_in_the_texts_order_once_each_an_invented_one_dropped():
    reply = _debtor_reply(quesiti=["La penale è riducibile?", "Quali strategie difensive può adottare Orsa?",
                                   "La penale è riducibile?", "Orsa può chiedere i danni a Delfino?"])
    res = _read(DEBTOR, reply)
    assert res["frame"]["quesiti"] == ["Quali strategie difensive può adottare Orsa?", "La penale è riducibile?"]
    assert res["counters"]["frame_unverified.quesiti"] == 1


def test_the_institution_in_the_lawyers_words_only():
    res = _read(DEBTOR, _debtor_reply(istituto={"testo": "appalto di servizi",
                                                "frase": "«un contratto di fornitura di componenti»"}))
    assert res["frame"]["istituto"] is None and "non con le parole del testo" in res["unverified"]["istituto"][0]


def test_a_charge_description_in_words_not_in_the_text_is_blanked_the_charge_kept():
    reply = _debtor_reply(addebito_o_domanda={"articolo": "1382 c.c.", "descrizione": "clausola vessatoria abusiva",
                                              "frase": "«Delfino pretende ora il pagamento di una penale»"})
    res = _read(DEBTOR, reply)
    assert res["frame"]["addebito_o_domanda"]["descrizione"] == ""
    assert res["frame"]["addebito_o_domanda"]["ipotesi"] == "1382 c.c."       # v79.19.2 (item 2): uncited
    assert res["counters"]["frame_unverified.addebito_o_domanda.descrizione"] == 1


@pytest.mark.parametrize("articolo,kept,mismatch", [
    ("1439 c.c.", None, 1),                         # the civil code's dolo: never a penal charge
    ("art. 1439 del codice civile", None, 1),
    ("art. 700 c.p.c.", None, 1),
    ("2634 c.c.", "2634 c.c.", 0),                  # the civil code's reati societari are penal provisions
    ("art. 640 c.p.", "640 c.p.", 0),
    ("640-bis c.p.", "640-bis c.p.", 0),
    ("73, comma 5, d.P.R. 309/1990", "73 d.P.R. 309/1990", 0),    # a penal special law (data/special_laws.json)
    ("art. 2 d.lgs. 74/2000", "2 d.lgs. 74/2000", 0),             # a penal law outside that table (the data's)
    ("art. 2634 c.c. e altri reati", "2634 c.c.", 0),             # the code alone, the sentence's words out
    ("art. 1439 c.c. per il raggiro", None, 1),
])
def test_in_a_penal_rito_the_charge_article_must_be_a_penal_provision(articolo, kept, mismatch):
    res = _read(DIRECTOR, _director_reply(articolo))
    charge = res["frame"]["addebito_o_domanda"]
    assert charge is not None and charge["frase"]               # the charge stands; only its article may go
    # v79.19.2 (item 2): the text cites no article -- a penal provision the reply names is a hypothesis («ipotesi»),
    # a non-penal one dropped as before; «articolo» is never set from the reply alone
    assert charge["articolo"] is None and charge.get("ipotesi") == kept
    assert res["counters"]["charge_rito_dropped"] == mismatch


def test_the_penal_rule_reads_the_fallbacks_rito_when_the_frame_has_none():
    reply = _director_reply("1439 c.c.")
    reply["rito"] = None
    assert _read(DIRECTOR, reply)["frame"]["addebito_o_domanda"]["ipotesi"] == "1439 c.c."    # v79.19.2: uncited
    res = _read(DIRECTOR, reply, rito_hint="penale")
    assert res["frame"]["addebito_o_domanda"]["articolo"] is None and res["counters"]["charge_rito_dropped"] == 1
    assert "ipotesi" not in res["frame"]["addebito_o_domanda"]


def test_a_civil_rito_keeps_a_civil_article():
    res = _read(DIRECTOR, _director_reply("1439 c.c.", rito="civile"))
    assert res["frame"]["addebito_o_domanda"]["ipotesi"] == "1439 c.c."        # v79.19.2 (item 2): uncited
    assert res["counters"]["charge_rito_dropped"] == 0


def test_a_bare_article_number_takes_the_code_the_text_cites_it_with():
    text = DIRECTOR + " Il PM ipotizza l'art. 2634 c.c. e altri reati societari."
    assert _read(text, _director_reply("2634"))["frame"]["addebito_o_domanda"]["articolo"] == "2634 c.c."
    res = _read(DIRECTOR, _director_reply("2634"))
    assert res["frame"]["addebito_o_domanda"]["articolo"] is None
    assert res["counters"]["frame_unverified.addebito_o_domanda.articolo"] == 1
    assert res["counters"]["charge_rito_dropped"] == 0


def test_the_vocabularies_map_inside_intakes_own():
    cfg = cf.config()
    assert set(cfg["ruoli"].values()) <= set(intake.ROLES)
    assert set(cfg["riti"].values()) <= set(intake.RITI) and set(cfg["riti"]) == set(intake.RITI)
    assert set(cfg["stadi"].values()) <= set(intake.STAGES)
    for table in ("ruoli", "riti", "stadi", "lati"):
        assert set(cfg[f"{table}_sinonimi"].values()) <= set(cfg[table]), table
    assert cf.canon("Imputato / Indagato", "ruoli") == "imputato/indagato"
    assert cf.canon("persona offesa", "ruoli") == "persona_offesa"
    assert cf.canon("appello", "stadi") == "appello" and cfg["stadi"]["appello"] == "impugnazione"
    assert cf.canon("autorita", "lati") == "autorità"
    from src.rag import special_laws
    ids = {law["id"] for law in special_laws.laws()}
    laws = cfg["penal_provisions"]["laws"]
    assert {x["special_law_id"] for x in laws if x.get("special_law_id")} <= ids
    assert all(x.get("expression") or x.get("special_law_id") for x in laws)
    assert len(cf._penal_laws()) == len(laws)          # every entry compiled


# --- 3. the brief's five cases ----------------------------------------------------------------------------------------

def test_case_1_the_debtor_resisting_a_penale_and_a_resolution_is_convenuto_never_the_claimant():
    res = _read(DEBTOR, _debtor_reply())
    assert cf.intake_view(res)["role"] == "convenuto"
    res = _read(DEBTOR, _debtor_reply(ruolo={"valore": "resistente", "perche": "«Delfino pretende ora il pagamento di "
                                                                               "una penale»"}))
    assert res["frame"]["ruolo"]["valore"] == "resistente" and cf.intake_view(res)["role"] == "convenuto"
    # an «attore» on a sentence the text does not hold never stands: the rules decide
    res = _read(DEBTOR, _debtor_reply(ruolo={"valore": "attore", "perche": "«Orsa chiede il risarcimento»"}))
    assert res["frame"]["ruolo"] is None and "ruolo" in cf.fallback_fields(res)


def test_case_2_a_director_under_investigation_is_indagato_charged_with_a_penal_article_never_1439():
    res = _read(DIRECTOR, _director_reply("2634 c.c."))
    view = cf.intake_view(res)
    assert res["frame"]["ruolo"]["valore"] == "imputato/indagato" and view["role"] == "imputato"
    assert view["rito"] == "penale" and view["stage"] == "indagini"
    assert view["charge"]["articolo"] is None and view["charge"]["ipotesi"] == "2634 c.c."    # v79.19.2: uncited
    res = _read(DIRECTOR, _director_reply("1439 c.c."))
    assert cf.intake_view(res)["charge"]["articolo"] is None
    assert res["counters"]["charge_rito_dropped"] == 1
    assert "1439" not in cf.fit_context(res)


def _two_sides_reply(cliente):
    return {"parti": [{"nome": "Anna Pino", "ruolo_nel_fatto": "ha prestato 9.000 euro", "lato": "cliente"},
                      {"nome": "Bruno Faggio", "ruolo_nel_fatto": "non ha restituito la somma", "lato": "controparte"}],
            "cliente": cliente,
            "ruolo": {"valore": "attore", "perche": "«Come può recuperare la somma?»"},
            "quesiti": ["Come può recuperare la somma?"]}


def test_case_3_two_parties_on_opposite_sides_one_client():
    res = _read(TWO_SIDES, _two_sides_reply({"nome": "Anna Pino", "perche": "«Assisto Anna Pino»"}))
    assert res["frame"]["cliente"]["nome"] == "Anna Pino" and cf.intake_view(res)["client"] == "Anna Pino"
    assert [p["lato"] for p in res["frame"]["parti"]] == ["cliente", "controparte"]
    both = [{"nome": "Anna Pino", "perche": "«Assisto Anna Pino»"},
            {"nome": "Bruno Faggio", "perche": "«Faggio non ha restituito la somma»"}]
    res = _read(TWO_SIDES, _two_sides_reply(both))
    assert res["frame"]["cliente"] is None and "più di un cliente" in res["unverified"]["cliente"][0]
    res = _read(TWO_SIDES, _two_sides_reply({"nome": "Bruno Faggio", "perche": "«Faggio non ha restituito la somma»"}))
    assert res["frame"]["cliente"] is None and "controparte" in res["unverified"]["cliente"][0]
    res = _read(TWO_SIDES, _two_sides_reply([{"nome": "Anna Pino", "perche": "«Assisto Anna Pino»"}]))
    assert res["frame"]["cliente"]["nome"] == "Anna Pino"


def _a_carico_reply(cliente, lati=("cliente", "controparte")):
    return {"parti": [{"nome": "Carla Ulivo", "ruolo_nel_fatto": "ha subito un'aggressione", "lato": lati[0]},
                      {"nome": "Dario Cedro", "ruolo_nel_fatto": "aggressione davanti a un locale", "lato": lati[1]}],
            "cliente": cliente,
            "ruolo": {"valore": "persona_offesa", "perche": "«Carla Ulivo può costituirsi parte civile?»"},
            "rito": {"valore": "penale", "perche": "«Quali reati sono configurabili»"},
            "quesiti": ["Quali reati sono configurabili a carico di Dario Cedro?",
                        "Carla Ulivo può costituirsi parte civile?"]}


def test_case_4_questions_a_carico_di_x_make_x_the_counterpartys_subject_the_client_is_whom_they_serve():
    res = _read(A_CARICO, _a_carico_reply({"nome": "Carla Ulivo",
                                           "perche": "«Carla Ulivo può costituirsi parte civile?»"}))
    assert res["frame"]["cliente"]["nome"] == "Carla Ulivo"
    assert cf.intake_view(res)["role"] == "persona_offesa"
    dario = {"nome": "Dario Cedro", "perche": "«Quali reati sono configurabili a carico di Dario Cedro?»"}
    res = _read(A_CARICO, _a_carico_reply(dario))                                  # listed as controparte
    assert res["frame"]["cliente"] is None
    res = _read(A_CARICO, _a_carico_reply(dario, lati=("terzo", "cliente")))       # even listed as the client
    assert res["frame"]["cliente"] is None and "a carico" in res["unverified"]["cliente"][0]
    assert "cliente" in cf.fallback_fields(res)


def test_case_5_no_client_signal_sconosciuto_and_the_honesty_line():
    reply = {"parti": [{"nome": "Volpe S.r.l.", "lato": "terzo"}, {"nome": "Lupo S.r.l.", "lato": "terzo"}],
             "cliente": None,
             "ruolo": {"valore": "sconosciuto", "perche": "«Si chiede una valutazione della vicenda»"},
             "rito": {"valore": "civile", "perche": "«controversia sul pagamento di alcune fatture»"},
             "stadio": {"valore": "sconosciuto", "perche": "«Si chiede una valutazione della vicenda»"},
             "quesiti": []}
    res = _read(NO_CLIENT, reply)
    assert res["frame"]["ruolo"]["valore"] == "sconosciuto" and "ruolo" in res["verified"]
    assert cf.role_unknown(res) and cf.intake_view(res)["role_unknown"]
    assert cf.intake_view(res)["role"] == "sconosciuto" and cf.intake_view(res)["client"] is None
    assert res["counters"]["frame_unverified.ruolo"] == 0 and res["counters"]["frame_unverified.cliente"] == 0
    assert "cliente" in res["empty"] and "cliente" in cf.fallback_fields(res)
    # «sconosciuto» with no quote: the field empty (the rules read the text's silence), never the honesty line
    reply["ruolo"] = {"valore": "sconosciuto", "perche": ""}
    res = _read(NO_CLIENT, reply)
    assert res["frame"]["ruolo"] is None and not cf.role_unknown(res)
    assert "ruolo" in res["empty"] and res["counters"]["frame_unverified.ruolo"] == 0


# --- 4. counters, the fallback API, the fit context, the flag, the nightly --------------------------------------------

def test_the_counters_and_the_fallback_api():
    counters = cf.empty_counters()
    assert set(cf.COUNTERS) == set(counters)
    res = _read(DEBTOR, _debtor_reply(istituto=None, ruolo={"valore": "convenuto", "perche": "inventato"}))
    assert cf.fallback_fields(res) == ["ruolo", "istituto"] == res["fallback"]
    assert res["empty"] == ["istituto"] and list(res["unverified"]) == ["ruolo"]
    assert res["counters"]["frame_fields_verified"] == 6
    cf.count_fallback(res["counters"], "ruolo")
    cf.count_fallback(res["counters"], "non_un_campo")
    assert res["counters"]["frame_fallback.ruolo"] == 1 and "frame_fallback.non_un_campo" not in res["counters"]
    assert cf.fallback_fields(None) == list(cf.FIELDS)
    assert res["counters"]["frame_seconds"] == res["seconds"] >= 0


def test_the_frame_as_the_fit_judgments_context_and_none_without_one():
    res = _read(DEBTOR, _debtor_reply())
    context = cf.fit_context(res)
    assert context.startswith(cf.config()["fit_context"]["title"])
    assert "Cliente: Orsa S.r.l." in context and "Ruolo del cliente: convenuto" in context
    norm = {"article": "1382 c.c.", "rubrica": "Effetti della clausola penale", "text": "La clausola penale ha "
            "l'effetto di limitare il risarcimento alla prestazione promessa."}
    facts = [{"id": "S1", "text": "Delfino pretende il pagamento di una penale.", "certainty": "accertato"}]
    plain = ap.build_messages([norm], facts, "convenuto")
    framed = ap.build_messages([norm], facts, "convenuto", context)
    assert framed[1].content == f"{context}\n\n{plain[1].content}" and framed[0].content == plain[0].content
    assert ap.build_messages([norm], facts, "convenuto", "")[1].content == plain[1].content
    assert cf.fit_context(None) == "" and cf.fit_context(_read(DEBTOR, "{}")) == ""
    sent = []
    ap.judge([norm], facts, "convenuto", call=lambda m, k: sent.append(m[1].content) or '{"giudizi": []}',
             frame_context=context)
    assert sent and sent[0].startswith(context)


def test_the_flow_passes_the_frame_to_the_fit_judgment_only_with_the_flag(monkeypatch):
    from src.rag import interactive_flow as flow
    got = []

    class Stop(Exception):
        pass

    def judge(norms, facts, role, **kw):
        got.append(kw.get("frame_context"))
        raise Stop
    monkeypatch.setattr(ap, "judge", judge)
    state = {"records": {}, "sentences": [], "text": DEBTOR, "request": "", "intake": {}, "analysis": {},
             "case_frame": _read(DEBTOR, _debtor_reply())}
    for flag, want in (("false", ""), ("true", cf.fit_context(state["case_frame"]))):
        monkeypatch.setenv("CASE_FRAME", flag)
        with pytest.raises(Stop):
            flow.applicability_fit(state, {"role": "convenuto"}, [], set(), [])
        assert got[-1] == want
    assert got[-1]


def test_the_flag_is_case_frame_default_off_and_in_the_suites_list(monkeypatch):
    import conftest
    assert cf.FLAG == "CASE_FRAME" and "CASE_FRAME" in conftest.FLAGS
    assert not cf.enabled()
    monkeypatch.setenv("CASE_FRAME", "true")
    assert cf.enabled()


def test_the_unverified_fields_are_logged(monkeypatch):
    from src.rag import verbose_logger
    events = []
    monkeypatch.setattr(verbose_logger, "vlog", lambda event, data=None, **kw: events.append((event, data)))
    _read(DEBTOR, _debtor_reply(ruolo={"valore": "convenuto", "perche": "inventato"}))
    names = [e for e, _ in events]
    assert "case_frame_unverified" in names and "case_frame_read" in names
    read = dict(events)["case_frame_read"]
    assert read["prompt_hash"] == cf.prompt_hash() and "ruolo" in read["fallback"]


def test_the_nightly_turns_the_flag_on_and_prints_the_frame_per_cell(monkeypatch, tmp_path):
    import nightly_sim
    monkeypatch.setattr(nightly_sim, "cells", lambda path=None: [])
    monkeypatch.setattr(nightly_sim, "stage_check", lambda: {"ok": True, "report": ""})
    nightly_sim.main(["--answers", str(tmp_path), "--no-stages"])
    assert os.environ["CASE_FRAME"] == "true"
    nightly_sim.main(["--answers", str(tmp_path), "--no-stages", "--no-case-frame"])
    assert os.environ["CASE_FRAME"] == "false"
    monkeypatch.delenv("CASE_FRAME", raising=False)
    monkeypatch.delenv("APPLICABILITY_FIT", raising=False)
    monkeypatch.delenv("FAMILY_GATE_SHADOW_GRADE", raising=False)
    # a cell's frame: the pipeline's when the state has one, else (flag on) the nightly's own read of the cell's text
    assert nightly_sim.frame_of({}, DEBTOR) == {"ran": False}
    piped = nightly_sim.frame_of({"case_frame": _read(DEBTOR, _debtor_reply())}, DEBTOR)
    assert piped["source"] == "pipeline" and "cliente: Orsa S.r.l. — «Assisto la società Orsa S.r.l.»" in piped["lines"]
    monkeypatch.setenv("CASE_FRAME", "true")
    monkeypatch.setattr(cf, "_default_call", _model(_director_reply("1439 c.c.")))
    own = nightly_sim.frame_of({}, DIRECTOR)
    assert own["source"] == "nightly" and own["counters"]["charge_rito_dropped"] == 1
    lines = nightly_sim.frame_lines([{"id": "cell_a", "case_frame": piped, "counters": {}},
                                     {"id": "cell_b", "case_frame": own, "counters": {"role_flip": 0}}])
    text = "\n".join(lines)
    assert "## Case frame (CASE_FRAME)" in text and "**cell_a**" in text and "**cell_b**" in text
    assert "| cell_b | " in text and "[scartato: articolo: in un rito penale" in text


# --- 5. (review) the verification's holes closed ----------------------------------------------------------------------

def test_review_a_verified_rito_sconosciuto_never_skips_the_penal_rule_the_fallbacks_rito_reads():
    reply = _director_reply("1439 c.c.")
    reply["rito"] = {"valore": "sconosciuto", "perche": "«Quali sono i rischi penali per il signor Lince?»"}
    res = _read(DIRECTOR, reply, rito_hint="penale")
    assert res["frame"]["rito"]["valore"] == "sconosciuto"                 # the frame's own value stands
    assert res["frame"]["addebito_o_domanda"]["articolo"] is None          # ...but the charge rule read «penale»
    assert res["counters"]["charge_rito_dropped"] == 1
    # no rito at all, no hint: nothing to check against here -- the binding checks it against the posture (flow test)
    reply["rito"] = None
    assert _read(DIRECTOR, reply)["frame"]["addebito_o_domanda"]["ipotesi"] == "1439 c.c."    # v79.19.2: uncited


@pytest.mark.parametrize("label,penal", [
    ("44 d.P.R. 380/2001", True), ("44, lett. b), d.P.R. 380/2001", True),     # edilizia: the penal article
    ("167 d.lgs. 196/2003", True), ("256 d.lgs. 152/2006", True), ("171-ter l. 633/1941", True),
    ("25-ter d.lgs. 231/2001", True), ("216 l. fall.", True), ("186 codice della strada", True),
    ("2634 c.c.", True), ("640 c.p.", True), ("73 d.P.R. 309/1990", True), ("2 d.lgs. 74/2000", True),
    ("67 l. fall.", False),                     # the revocatoria: civil
    ("117 d.lgs. 385/1993", False), ("117 TUB", False),             # the banking contract's form
    ("142 codice della strada", False),         # speeding: an administrative sanction
    ("2 CCII", False),                          # definitions
    ("1439 c.c. ex d.lgs. 74/2000", False),     # led by the civil code: the civil code's
    ("2043 c.c. legge armi", False), ("1439 c.c.", False), ("700 c.p.c.", False), ("5 d.lgs. 231/2007", False),
])
def test_review_a_penal_provision_is_a_penal_laws_penal_article_never_the_whole_law(label, penal):
    assert cf.is_penal(cf.parse_article(label)) is penal


def test_review_a_penal_special_laws_charge_is_kept_in_a_penal_rito():
    text = ("Assisto Luca Serra, indagato per aver costruito senza permesso di costruire un capannone. La Procura "
            "contesta l'art. 44, lett. b), d.P.R. 380/2001. Quali difese ha Luca Serra?")
    reply = {"parti": [{"nome": "Luca Serra", "lato": "cliente"}],
             "cliente": {"nome": "Luca Serra", "perche": "«Assisto Luca Serra»"},
             "rito": {"valore": "penale", "perche": "«indagato per aver costruito senza permesso di costruire»"},
             "addebito_o_domanda": {"articolo": "44 d.P.R. 380/2001", "descrizione": "",
                                    "frase": "«La Procura contesta l'art. 44, lett. b), d.P.R. 380/2001»"}}
    res = _read(text, reply)
    assert res["frame"]["addebito_o_domanda"]["articolo"] == "44 d.P.R. 380/2001"
    assert res["counters"]["charge_rito_dropped"] == 0


def test_review_a_defence_client_named_after_a_carico_di_is_kept_the_brief_case_4_still_refused():
    text = ("È stato aperto un procedimento penale a carico di Paolo Neri, che mi ha chiesto di difenderlo. Paolo Neri "
            "avrebbe sottratto merce dal magazzino. Quali difese ha?")
    reply = {"parti": [{"nome": "Paolo Neri", "lato": "cliente"}],
             "cliente": {"nome": "Paolo Neri",
                         "perche": "«procedimento penale a carico di Paolo Neri, che mi ha chiesto di difenderlo»"}}
    assert _read(text, reply)["frame"]["cliente"]["nome"] == "Paolo Neri"
    text = ("Il signor Marco Lince, che assisto, è indagato per infedeltà patrimoniale. Quali reati possono essere "
            "contestati a carico di Marco Lince e come difenderlo?")
    reply = {"parti": [{"nome": "Marco Lince", "lato": "cliente"}],
             "cliente": {"nome": "Marco Lince", "perche": "«Quali reati possono essere contestati a carico di Marco "
                                                          "Lince e come difenderlo?»"}}
    assert _read(text, reply)["frame"]["cliente"]["nome"] == "Marco Lince"
    # the brief's case 4: no signal of assistance for Dario Cedro -- the questions' subject, never the client
    dario = {"nome": "Dario Cedro", "perche": "«Quali reati sono configurabili a carico di Dario Cedro?»"}
    res = _read(A_CARICO, _a_carico_reply(dario, lati=("cliente", "cliente")))
    assert res["frame"]["cliente"] is None and "a carico" in res["unverified"]["cliente"][0]
    # «assisto Carla contro Dario»: the signal is Carla's, a marker stands between it and Dario
    assert cf._signalled_client("Assisto Carla Ulivo contro Dario Cedro.", "Carla Ulivo")
    assert not cf._signalled_client("Assisto Carla Ulivo contro Dario Cedro.", "Dario Cedro")


def test_review_names_match_exactly_never_by_stem_company_forms_and_titles_normalised():
    text = ("Assisto Luca Bianchi. La controparte è Maria Rossi, che chiede il pagamento della penale. Quali difese ha "
            "il mio cliente?")
    reply = {"parti": [{"nome": "Mario Rossi", "lato": "cliente"}, {"nome": "Luca Bianchi", "lato": "controparte"}],
             "cliente": {"nome": "Mario Rossi", "perche": "«Assisto Luca Bianchi»"}}
    res = _read(text, reply)
    assert res["frame"]["cliente"] is None
    assert "Mario Rossi" not in [p["nome"] for p in res["frame"]["parti"]]
    assert not cf.name_found(text, "Mario Rossi") and cf.name_found(text, "Maria Rossi")
    text = "Assisto Orsa Srl contro Delfino SpA, che chiede una penale. Quali difese ha Orsa?"
    reply = {"parti": [{"nome": "Orsa S.r.l.", "lato": "cliente"}, {"nome": "Delfino S.p.A.", "lato": "controparte"}],
             "cliente": {"nome": "Orsa S.r.l.", "perche": "«Assisto Orsa Srl»"}}
    res = _read(text, reply)
    assert res["frame"]["cliente"]["nome"] == "Orsa S.r.l." and len(res["frame"]["parti"]) == 2
    reply = {"parti": [{"nome": "Sig. Marco Lince", "lato": "cliente"}],
             "cliente": {"nome": "Sig. Marco Lince", "perche": "«Il mio cliente, Marco Lince»"}}
    assert _read(DIRECTOR, reply)["frame"]["cliente"]["nome"] == "Sig. Marco Lince"


ALFA_BETA = "Assisto la società Alfa S.r.l. La società Beta S.r.l. chiede la penale e la risoluzione del contratto."


def test_review_a_cliente_quote_must_show_that_party_its_role_goes_with_it():
    reply = {"parti": [{"nome": "Alfa S.r.l.", "lato": "controparte"}, {"nome": "Beta S.r.l.", "lato": "cliente"}],
             "cliente": {"nome": "Beta S.r.l.", "perche": "«Assisto la società Alfa S.r.l.»"},
             "ruolo": {"valore": "attore", "perche": "«Beta S.r.l. chiede la penale»"}}
    res = _read(ALFA_BETA, reply)
    assert res["frame"]["cliente"] is None and res["frame"]["ruolo"] is None
    assert "Alfa" in res["unverified"]["cliente"][0] and "cliente scartato" in res["unverified"]["ruolo"][0]
    # the parties right, the quote naming the client: kept
    reply = {"parti": [{"nome": "Alfa S.r.l.", "lato": "cliente"}, {"nome": "Beta S.r.l.", "lato": "controparte"}],
             "cliente": {"nome": "Alfa S.r.l.", "perche": "«Assisto la società Alfa S.r.l.»"},
             "ruolo": {"valore": "convenuto", "perche": "«Beta S.r.l. chiede la penale»"}}
    res = _read(ALFA_BETA, reply)
    assert res["frame"]["cliente"]["nome"] == "Alfa S.r.l." and res["frame"]["ruolo"]["valore"] == "convenuto"


def test_review_a_cliente_quoting_only_its_name_or_contradicting_assisto_x_is_refused():
    reply = {"parti": [{"nome": "Anna Pino", "lato": "controparte"}, {"nome": "Bruno Faggio", "lato": "cliente"}],
             "cliente": {"nome": "Bruno Faggio", "perche": "«Bruno Faggio»"},
             "ruolo": {"valore": "convenuto", "perche": "«Faggio non ha restituito la somma»"}}
    res = _read(TWO_SIDES, reply)
    assert res["frame"]["cliente"] is None and res["frame"]["ruolo"] is None
    reply["cliente"] = {"nome": "Bruno Faggio", "perche": "«Faggio non ha restituito la somma alla scadenza»"}
    res = _read(TWO_SIDES, reply)                       # a contentful quote, but the text says «Assisto Anna Pino»
    assert res["frame"]["cliente"] is None and "Anna Pino" in res["unverified"]["cliente"][0]
    assert res["frame"]["ruolo"] is None
    reply = {"parti": [{"nome": "Anna Pino", "lato": "cliente"}], "cliente": {"nome": "Anna Pino", "perche": "«Pino»"}}
    assert "solo il nome" in _read(TWO_SIDES, reply)["unverified"]["cliente"][0]


def test_review_quesiti_are_whole_questions_of_the_text_never_a_statement_or_a_question_cut_short():
    text = ("Assisto Alfa. Quali reati sono configurabili a carico di Rossi se il pagamento è avvenuto dopo la diffida? "
            "Redigi un parere: come può Alfa recuperare il credito?")
    res = _read(text, {"quesiti": ["Quali reati sono configurabili a carico di Rossi", "Assisto Alfa"]})
    assert res["frame"]["quesiti"] == [] and res["counters"]["frame_unverified.quesiti"] == 2
    whole = ["Quali reati sono configurabili a carico di Rossi se il pagamento è avvenuto dopo la diffida?",
             "come può Alfa recuperare il credito?"]
    assert _read(text, {"quesiti": whole})["frame"]["quesiti"] == whole
    assert _read(text, {"quesiti": ["Redigi un parere: come può Alfa recuperare il credito?"]})["frame"]["quesiti"]
    # a request phrased as a task (intake.questions reads it) is a question too
    task = "Assisto Alfa. Il candidato illustri i rimedi esperibili da Alfa."
    got = _read(task, {"quesiti": ["Il candidato illustri i rimedi esperibili da Alfa."]})["frame"]["quesiti"]
    assert got == ["Il candidato illustri i rimedi esperibili da Alfa."]
    assert _read(task, {"quesiti": ["Il candidato illustri i rimedi"]})["frame"]["quesiti"] == []


def test_review_an_off_schema_controparte_lato_still_bars_the_client_and_a_client_is_a_lato_cliente_party():
    text = ("Assisto la società Alfa S.r.l. La società Beta S.r.l. chiede la penale e la risoluzione del contratto. "
            "Quali difese ha Alfa?")
    reply = {"parti": [{"nome": "Beta S.r.l.", "lato": "cliente"}, {"nome": "Beta S.r.l.", "lato": "controparte "
                                                                                                   "(creditore)"},
                       {"nome": "Alfa S.r.l.", "lato": "terzo"}],
             "cliente": {"nome": "Beta S.r.l.", "perche": "«La società Beta S.r.l. chiede la penale»"}}
    res = _read(text, reply)
    assert res["frame"]["cliente"] is None and "controparte" in res["unverified"]["cliente"][0]
    reply = {"parti": [{"nome": "Beta S.r.l.", "lato": "autorità"}],
             "cliente": {"nome": "Beta S.r.l.", "perche": "«La società Beta S.r.l. chiede la penale»"}}
    res = _read(text, reply)
    assert res["frame"]["cliente"] is None and "lato cliente" in res["unverified"]["cliente"][0]
