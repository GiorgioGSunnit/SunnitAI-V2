"""v79.15.3 (hotfix) -- three rendering fixes in the analysis step (src/rag/analysis_step.py, data/analysis_step.json
«claimant», «rendering», «objection_answer_check»). No prompt change (prompt hash b69eefcd0577), no new logic in the
model call. The model is faked at its source; every other model path and every outbound socket refused
(tests/test_v79_15_0_analysis.py's sealing, re-exported; the end-to-end runs through tests/_fixture_79_15_0.py).
Synthetic facts and invented names only; statutory article texts.

1. the printed tesi of a verified linea_principale / linea_subordinata is built by the code from the verified esito of
   its elemento_in_gioco and the client's side (code_tesi, data «rendering.tesi»); the model's tesi only in the log
   («analysis_step» tesi_modello=); §1, §6, §9 print the code's sentence
2. the client's side from the verified frame only (client_side): the brief's claimant and defence sets on the
   verified ruolo, then on the client party's verified ruolo_nel_fatto; «lavoratore» only when the claim is theirs;
   unverified -> neither (claimant False, as before)
3. §7: one plain form for every entry (objection_items: the allegation's sentence / the exception element, then the
   status in words and the client's sentences); a §7 answer sentence asserting an element integrato / provato that the
   verified table holds «da_provare» / «contestato» dropped (asserted_unproven, objections_sentences_dropped)
Fixtures: the civil caparra, the penal indagato and the appalto claimant end to end; with FIXTURE_79_15_3_OUT set, each
one's §1 line, §7 block and «analysis_step» log line are written there.
"""

import os
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import test_v79_15_0_analysis as T  # noqa: E402  (its autouse sealing fixture is re-exported below)
import test_v79_15_1_analysis as T1  # noqa: E402
from src.rag import analysis_step as A  # noqa: E402

_sealed = T._sealed
_Model, _reply, _line, _el = T._Model, T._reply, T._line, T._el


def _events(monkeypatch):
    events = []
    from src.rag import verbose_logger
    monkeypatch.setattr(verbose_logger, "vlog", lambda event, data=None, duration_ms=None: events.append((event, data)))
    return events


def _penal_reply(esito, cf_=("S5",), tesi="Gli artifizi non sono provati."):
    return _reply(_line("640 c.p.", T.E640[0], tesi, ["S5"], ["A640cp.1"]),
                  elementi=[_el("640 c.p.", T.E640[0], esito, list(cf_)), _el("640 c.p.", T.E640[1], "da_provare"),
                            _el("640 c.p.", T.E640[2], "da_provare")])


def _civil_reply(esito, cf_=("S4",), tesi="Lucia Ferri può esigere il doppio della caparra."):
    return _reply(_line("1385 c.c.", T.E1385[0], tesi, ["S2"], ["A1385cc.1"]),
                  elementi=[_el("1385 c.c.", T.E1385[0], esito, list(cf_))])


# --- 1. the tesi rendered by the code ---------------------------------------------------------------------------------

def test_the_prompt_is_unchanged():
    assert A.prompt_hash() == "b69eefcd0577"


@pytest.mark.parametrize("esito,cf_,want", [
    ("da_provare", (), "L'accusa deve provare «con artifizi o raggiri» (art. 640 c.p.)"),
    ("contestato", ("S3",), "L'accusa deve provare «con artifizi o raggiri» (art. 640 c.p.)"),
    ("escluso", ("S5",), "«con artifizi o raggiri» è escluso: «Paolo Sarti nega di aver mai pubblicato un falso "
                         "annuncio»"),
    # the brief gives no defence + provato template: the claimant's provato wording, the element proven as it is
    ("provato", ("S7",), "«con artifizi o raggiri» risulta provato: «Paolo Sarti dichiara di aver incassato il bonifico "
                         "di Giulia Neri»"),
])
def test_the_defences_tesi_by_the_verified_esito(esito, cf_, want):
    inp = T._penal()
    assert inp["side"] == A.SIDE_DEFENCE and not inp["claimant"]
    line = A.analyse(inp, call=_Model(_penal_reply(esito, cf_)))["analysis"]["linea_principale"]
    assert line["tesi"] == want and line["tesi_modello"] == "Gli artifizi non sono provati."
    assert line["esito_in_gioco"] == esito and line["tesi_codice"] is True


@pytest.mark.parametrize("esito,cf_,want", [
    ("provato", ("S2",), "«una parte dà all'altra, a titolo di caparra, una somma di danaro» risulta provato: «Lucia "
                         "Ferri ha versato a Marco Vento 20.000 euro a titolo di caparra confirmatoria»"),
    ("da_provare", (), "«una parte dà all'altra, a titolo di caparra, una somma di danaro» è da provare"),
    ("contestato", ("S3",), "«una parte dà all'altra, a titolo di caparra, una somma di danaro» è da provare"),
    ("escluso", ("S4",), "«una parte dà all'altra, a titolo di caparra, una somma di danaro» è da provare"),
])
def test_the_claimants_tesi_by_the_verified_esito(esito, cf_, want):
    inp = T._civil()
    assert inp["side"] == A.SIDE_CLAIMANT and inp["claimant"]
    line = A.analyse(inp, call=_Model(_civil_reply(esito, cf_)))["analysis"]["linea_principale"]
    assert line["tesi"] == want and line["tesi_modello"] == "Lucia Ferri può esigere il doppio della caparra."


def test_a_provato_tesi_cites_only_the_sentences_no_other_party_asserts():
    """S7 the client's own statement, S3 the Procura's: the verdict stays provato (one usable sentence), the tesi
    quotes S7 only."""
    inp = T._penal()
    line = A.analyse(inp, call=_Model(_penal_reply("provato", ("S3", "S7"))))["analysis"]["linea_principale"]
    assert line["esito_in_gioco"] == "provato"
    assert line["tesi"] == ("«con artifizi o raggiri» risulta provato: «Paolo Sarti dichiara di aver incassato il "
                            "bonifico di Giulia Neri»")
    assert "Secondo la Procura" not in line["tesi"]


def test_with_no_verified_role_the_neutral_wording():
    frame = {k: v for k, v in T.PENAL_FRAME.items() if k != "ruolo"}
    frame["parti"] = [{"nome": "Paolo Sarti", "ruolo_nel_fatto": "venditore", "lato": "cliente"}]
    inp = T._penal(frame=frame)
    assert inp["side"] == A.SIDE_NEUTRAL and not inp["claimant"]
    for esito, cf_, want in (("da_provare", (), "«con artifizi o raggiri» è da provare"),
                             ("escluso", ("S5",), "«con artifizi o raggiri» è escluso: «Paolo Sarti nega di aver mai "
                                                  "pubblicato un falso annuncio»")):
        line = A.analyse(inp, call=_Model(_penal_reply(esito, cf_)))["analysis"]["linea_principale"]
        assert line["tesi"] == want


def test_the_subordinate_line_is_built_on_its_own_articles_verdict():
    reply = _reply(_line("640 c.p.", T.E640[0], "Prima.", ["S5"], ["A640cp.1"]),
                   subordinata=_line("641 c.p.", T.E641[0], "Seconda.", ["S1"], ["A641cp.1"]),
                   elementi=[_el("640 c.p.", T.E640[0], "da_provare"), _el("641 c.p.", T.E641[0], "escluso", ["S2"])])
    an = A.analyse(T._penal(), call=_Model(reply))["analysis"]
    assert an["linea_subordinata"]["tesi"] == ("«dissimulando il proprio stato d'insolvenza» è escluso: «Giulia Neri ha "
                                               "pagato 800 euro con un bonifico»")
    assert an["linea_subordinata"]["tesi_modello"] == "Seconda."


def test_section_1_6_9_print_the_codes_tesi_and_the_model_tesi_goes_to_the_log(monkeypatch):
    events = _events(monkeypatch)
    inp = T._penal()
    res = A.analyse(inp, call=_Model(_penal_reply("contestato", ("S3",), tesi="Un testo del modello.")))
    v = A.view(res, inp)
    want = "L'accusa deve provare «con artifizi o raggiri» (art. 640 c.p.)"
    assert A.summary_text(v) == f"Linea principale: art. 640 c.p. — {want}."
    assert "elemento in gioco" not in A.summary_text(v)                 # the element named by the tesi itself
    assert f"| Linea principale | art. 640 c.p. | con artifizi o raggiri | {want} |" in A.solutions_text(v)
    assert A.conclusions_text(v).startswith(f"**Linea principale**: art. 640 c.p. — {want}.")
    assert A.answer_strategy(v) == [f"La linea principale è: art. 640 c.p. — {want}."]
    for text in (A.summary_text(v), A.solutions_text(v), A.conclusions_text(v), A.answer_strategy(v)[0]):
        assert "Un testo del modello" not in text
    line = [d for e, d in events if e == "analysis_step"][0]
    assert line["tesi_modello"] == {"linea_principale": "Un testo del modello."}
    assert line["side"] == A.SIDE_DEFENCE and line["side_why"] == "ruolo: indagato"
    assert line["lines"][0]["esito_in_gioco"] == "contestato"


def test_the_frame_charge_and_the_rubrica_fallback_lines_are_unchanged():
    # v79.19.1 (item 1): the model's verified line on 641 leads; the frame's charge, missing, is the subordinate «art. X
    # (addebito del quadro, elementi da provare)» -- no «linea non verificata» beside a verified line
    reply = _reply(_line("641 c.p.", T.E641[0], "Tesi.", ["S1"], ["A641cp.1"]))
    inp = T._penal()
    res = A.analyse(inp, call=_Model(reply, reply))
    v = A.view(res, inp)
    assert A.summary_text(v).endswith("Linea subordinata: art. 640 c.p. (addebito del quadro, elementi da provare).")
    assert A.summary_text(v).startswith("Linea principale: art. 641 c.p. — ")
    assert "tesi_modello" not in res["analysis"]["linea_subordinata"]
    assert res["analysis"]["linea_principale"]["tesi_modello"] == "Tesi."      # the model's line leads
    # no verified line at all: the frame's charge leads «art. X (addebito del quadro, linea non verificata)»
    none = _reply(_line("62-bis c.p.", "x", "Attenuanti.", ["S1"]))
    res = A.analyse(inp, call=_Model(none, none))
    assert A.summary_text(A.view(res, inp)).startswith("Linea principale: art. 640 c.p. (addebito del quadro, linea non "
                                                       "verificata).")
    # the rubrica fallback: its own code-written «<rubrica>: elementi da provare dall'accusa», no element in play
    bad = _reply({"articolo": "640 c.p.", "elemento_in_gioco": "un testo", "tesi": "t", "cita_fatti": ["S1"],
                  "cita_norma": []})
    inp = T._penal(frame=T1.NO_ARTICLE_TRUFFA)
    res = A.analyse(inp, call=_Model(bad, bad))
    assert A.summary_text(A.view(res, inp)) == "Linea principale: art. 640 c.p. — Truffa: elementi da provare dall'accusa."
    assert "tesi_modello" not in res["analysis"]["linea_principale"]


# --- 2. the client's side from the verified frame ---------------------------------------------------------------------

def _party(role, lato="cliente", nome="Marta Galli"):
    return {"nome": nome, "ruolo_nel_fatto": role, "lato": lato}


@pytest.mark.parametrize("role", ["attore", "ricorrente", "committente", "creditore", "promissario acquirente",
                                  "danneggiato", "locatore", "conduttore", "creditrice", "promissaria acquirente",
                                  "la committente dei lavori"])
def test_the_claimant_set_on_the_client_partys_verified_role(role):
    frame = {"parti": [_party(role), _party("appaltatrice", "controparte", "Edilnova S.r.l.")]}
    assert A.client_side(frame, "Marta Galli")[0] == A.SIDE_CLAIMANT and A.claimant(frame, "Marta Galli")


@pytest.mark.parametrize("role", ["indagato", "imputato", "convenuto", "resistente", "debitore", "debitrice"])
def test_the_defence_set_on_the_client_partys_verified_role(role):
    frame = {"parti": [_party(role)]}
    assert A.client_side(frame)[0] == A.SIDE_DEFENCE and not A.claimant(frame)


@pytest.mark.parametrize("valore,intake,side", [
    ("attore", "attore", A.SIDE_CLAIMANT), ("ricorrente", "attore", A.SIDE_CLAIMANT),
    ("convenuto", "convenuto", A.SIDE_DEFENCE), ("resistente", "convenuto", A.SIDE_DEFENCE),
    ("imputato/indagato", "imputato", A.SIDE_DEFENCE), ("persona_offesa", "persona_offesa", A.SIDE_NEUTRAL),
    ("sconosciuto", "sconosciuto", A.SIDE_NEUTRAL)])
def test_the_verified_ruolo_decides_first(valore, intake, side):
    assert A.client_side({"ruolo": {"valore": valore, "intake": intake}})[0] == side


def test_the_procedural_role_beats_the_partys_role_and_the_party_reads_only_without_one():
    committente = {"parti": [_party("committente")]}
    assert A.client_side(dict(committente, ruolo={"valore": "convenuto", "intake": "convenuto"})) == \
        (A.SIDE_DEFENCE, "ruolo: convenuto")
    assert A.client_side(committente) == (A.SIDE_CLAIMANT, "ruolo_nel_fatto: committente")
    # a role that is no side (persona offesa) leaves the party's word to decide
    assert A.client_side(dict(committente, ruolo={"valore": "persona_offesa", "intake": "persona_offesa"}))[0] == \
        A.SIDE_CLAIMANT


def test_unverified_or_ambiguous_is_neither():
    assert A.client_side({}) == (A.SIDE_NEUTRAL, "")
    assert A.client_side(None) == (A.SIDE_NEUTRAL, "")
    assert A.client_side({"parti": [_party("")]})[0] == A.SIDE_NEUTRAL       # case_frame voided the ungrounded role
    assert A.client_side({"parti": [_party("tesoriere")]})[0] == A.SIDE_NEUTRAL
    assert A.client_side({"parti": [_party("creditore e debitore")]})[0] == A.SIDE_NEUTRAL       # both sides' words
    # the counterparty's role never decides the client's side
    assert A.client_side({"parti": [_party("committente", "controparte", "Edilnova S.r.l.")]})[0] == A.SIDE_NEUTRAL
    # a word inside another word is no role («attoreo», «sub-committente» read as whole words only)
    assert A.client_side({"parti": [_party("subcommittente")]})[0] == A.SIDE_NEUTRAL


def test_the_worker_is_the_claimant_only_when_the_claim_is_theirs():
    worker = {"parti": [_party("lavoratrice")]}
    assert A.client_side(worker)[0] == A.SIDE_NEUTRAL                                  # no rito lavoro, no domanda
    assert A.client_side(dict(worker, rito={"valore": "lavoro", "intake": "lavoro"}))[0] == A.SIDE_NEUTRAL
    theirs = dict(worker, rito={"valore": "lavoro", "intake": "lavoro"},
                  addebito_o_domanda={"articolo": None, "descrizione": "differenze retributive"})
    assert A.client_side(theirs) == (A.SIDE_CLAIMANT, "ruolo_nel_fatto: lavoratrice")
    assert A.client_side(dict(theirs, rito={"valore": "civile", "intake": "civile"}))[0] == A.SIDE_NEUTRAL
    assert A.client_side(dict(theirs, ruolo={"valore": "convenuto", "intake": "convenuto"}))[0] == A.SIDE_DEFENCE


def test_the_committente_cause_the_frame_never_holds_it_as_a_ruolo_value():
    """79.15.1's claimant() read «claimant.roles» (attore, ricorrente) on the verified ruolo and the party words only
    when no defendant ruolo was verified. «committente» is no value of the frame's ruolo (data/case_frame.json
    «ruoli»): a model writing it there fails «valore fuori schema» and leaves the ruolo unverified -- the side then
    comes from the client party's verified ruolo_nel_fatto, or is neither."""
    from src.rag import case_frame
    assert case_frame.canon("committente", "ruoli") is None
    assert A.client_side({"ruolo": {"valore": "committente", "intake": "committente"}})[0] == A.SIDE_CLAIMANT
    assert A.client_side({"ruolo": None, "parti": [_party("committente")]})[0] == A.SIDE_CLAIMANT


def test_the_input_carries_the_side_and_the_model_call_is_unchanged():
    inp = T1._appalto(frame={k: v for k, v in T1.APPALTO_FRAME.items() if k != "ruolo"})
    assert inp["side"] == A.SIDE_CLAIMANT and inp["claimant"] and inp["side_why"] == "ruolo_nel_fatto: committente"
    reask = A.reask_messages([], "", ["x"], inp)[-1].content
    assert A.config()["reask_claimant"] in reask                   # the claimant's existing re-ask rule, no new one
    neutral = T._penal(frame={k: v for k, v in T.PENAL_FRAME.items() if k not in ("ruolo", "parti")})
    assert neutral["side"] == A.SIDE_NEUTRAL and not neutral["claimant"]
    assert A.config()["reask_claimant"] not in A.reask_messages([], "", ["x"], neutral)[-1].content


# --- 3. §7: one plain form; an answer asserting an open element dropped ----------------------------------------------

def test_every_objection_has_the_same_plain_form():
    inp = T1._appalto()
    res = A.analyse(inp, call=_Model(T1.GOOD_APPALTO))
    items = A.objection_items(A.view(res, inp))
    assert items[0] == {"obiezione": "La Edilnova S.r.l. sostiene che Marta Galli ha accettato l'opera senza riserve",
                        "risposta": "L'elemento «il committente ha accettato l'opera» (art. 1667 c.c.) è da provare.",
                        "fonte": "allegazione"}
    assert items[1]["obiezione"].startswith("Eccezione dell'art. 1667 c.c.: «denunziare all'appaltatore")
    assert items[1]["risposta"].endswith("risulta provato: «Marta Galli ha denunziato i vizi all'impresa con una "
                                         "raccomandata dieci giorni dopo la scoperta».")
    for i in items:
        assert "Esito verificato" not in i["risposta"] and "Fatti del cliente" not in i["risposta"]


def test_the_defences_objection_on_two_elements_with_the_clients_sentence():
    reply = _reply(_line("640 c.p.", T.E640[0], "Tesi.", ["S3", "S5"], ["A640cp.1"]),
                   elementi=[_el("640 c.p.", T.E640[0], "contestato", ["S3"]),
                             _el("640 c.p.", T.E640[1], "contestato", ["S3"])])
    inp = T._penal()
    items = A.objection_items(A.view(A.analyse(inp, call=_Model(reply)), inp))
    assert items == [{"obiezione": "Secondo la Procura, Paolo Sarti ha usato un falso annuncio per indurre in errore "
                                   "Giulia Neri",
                      "risposta": "L'elemento «con artifizi o raggiri» (art. 640 c.p.) non è provato; l'elemento "
                                  "«inducendo taluno in errore» (art. 640 c.p.) non è provato: «Paolo Sarti nega di aver "
                                  "mai pubblicato un falso annuncio».", "fonte": "allegazione"}]


def test_an_allegation_naming_no_speaker_is_attributed():
    """A sentence marked as another party's allegation (79.14.3's alleged_by) that names no speaker itself."""
    sents = T._sents(T.PENAL_SENTENCES, sistema=(9,))
    sents[3] = dict(sents[3], text="La bicicletta non è mai arrivata.", alleged_by="Giulia Neri")
    inp = A.make_input(T.PENAL_FRAME, T.PENAL_NORMS, sents, [], client="Paolo Sarti",
                       client_roles=["imputato", "indagato"])
    reply = _reply(_line("640 c.p.", T.E640[0], "Tesi.", ["S5"], ["A640cp.1"]),
                   elementi=[_el("640 c.p.", T.E640[2], "contestato", ["S4"])])
    res = A.analyse(inp, call=_Model(reply))
    obj = res["analysis"]["obiezioni"][0]
    assert obj["sid"] == "S4" and obj["nominata"] is False
    item = A.objection_items(A.view(res, inp))[0]
    assert item["obiezione"] == "Allegazione di Giulia Neri: «La bicicletta non è mai arrivata»"
    assert item["risposta"] == ("L'elemento «procura a sé o ad altri un ingiusto profitto con altrui danno» (art. 640 "
                                "c.p.) non è provato.")


def _table():
    inp = T._penal()
    reply = _reply(_line("640 c.p.", T.E640[0], "Tesi.", ["S5"], ["A640cp.1"]),
                   elementi=[_el("640 c.p.", T.E640[0], "contestato", ["S3"]),
                             _el("640 c.p.", T.E640[1], "provato", ["S7"]),
                             _el("640 c.p.", T.E640[2], "da_provare")])
    return A.view(A.analyse(inp, call=_Model(reply)), inp)


@pytest.mark.parametrize("sentence,eid", [
    ("Gli artifizi o raggiri sono integrati.", "A640cp:E1"),                       # every token of the element
    ("L'elemento con artifizi o raggiri risulta provato.", "A640cp:E1"),           # the schema text
    ("L'elemento A640cp:E1 è integrato.", "A640cp:E1"),                            # the E-id
    ("L'elemento E3 sussiste.", "A640cp:E3"),                                      # «E<n>» on the line article
    ("Il profitto ingiusto con altrui danno è dimostrato.", "A640cp:E3"),          # the overlap (3 of 4 tokens)
])
def test_an_answer_asserting_an_open_or_contested_element_is_caught(sentence, eid):
    got = A.asserted_unproven(sentence, _table())
    assert got is not None and got["eid"] == eid


@pytest.mark.parametrize("sentence", [
    "Gli artifizi o raggiri non sono integrati.",                  # a negation
    "Gli artifizi o raggiri vanno provati dall'accusa.",           # a modal: nothing asserted
    "Resta da provare che gli artifizi o raggiri siano integrati.",
    "Se gli artifizi o raggiri fossero provati, la pena cambierebbe.",
    "L'induzione di taluno in errore è provata.",                  # E2 is «provato» in the table: true
    "Il bonifico è integrato.",                                    # no element named
    "Gli artifizi o raggiri sono gravi.",                          # no assertion verb
])
def test_an_answer_that_asserts_nothing_against_the_table_is_kept(sentence):
    assert A.asserted_unproven(sentence, _table()) is None


def test_the_review_drops_the_answers_sentence_and_its_follower_and_counts_it():
    v = _table()
    events = []
    review = A.ProseReview(v, A.prose_scope(v), A.SITE_OBJECTIONS, events)          # an answer (replace=True)
    out = [review(x) for x in ("Gli artifizi o raggiri sono integrati.", "Inoltre Giulia Neri ha pagato.",
                               "Paolo Sarti nega ogni addebito.")]
    assert out == ["", "", "Paolo Sarti nega ogni addebito."]
    assert [(e["esito"], e.get("motivo"), e.get("elemento")) for e in events] == [
        (A.DROPPED, A.UNPROVEN_ASSERTED, "A640cp:E1"), (A.DEPENDENT, None, None)]
    c = A.prose_counters(events)
    assert c["objections_sentences_dropped"] == 1 and c["qa_sentences_dropped_dependent"] == 1
    # the objection itself (replace=False: the counterparty's claim) and the Q&A are not this check's
    for site, replace in ((A.SITE_OBJECTIONS, False), (A.SITE_QA, True)):
        assert A.ProseReview(v, A.prose_scope(v), site, [], replace=replace)(
            "Gli artifizi o raggiri sono integrati.") == "Gli artifizi o raggiri sono integrati."


# --- (review) the «asserts» check clause by clause ------------------------------------------------------------------

@pytest.mark.parametrize("sentence,eid", [
    # a word that only starts like a negation («restituzione», «contesto») negates nothing
    ("Gli artifizi o raggiri sono integrati: la restituzione del denaro lo conferma.", "A640cp:E1"),
    ("Gli artifizi o raggiri sono provati, come risulta dal contesto dei messaggi.", "A640cp:E1"),
    # an emphatic «senza alcun dubbio» is no negation, before or after the verb
    ("Gli artifizi o raggiri sono integrati senza alcun dubbio.", "A640cp:E1"),
    ("Senza alcun dubbio gli artifizi o raggiri sono integrati.", "A640cp:E1"),
    # a negation in one clause does not cover the next: E3 (da_provare) asserted provato
    ("Gli artifizi o raggiri non sono integrati, mentre l'ingiusto profitto con altrui danno è provato.", "A640cp:E3"),
    # the element named by a verbless clause before the verb's
    ("Quanto agli artifizi o raggiri, sono integrati.", "A640cp:E1"),
    # a negation after the verb («senza» a later qualifier) does not negate it
    ("Gli artifizi o raggiri sono provati, senza che la difesa abbia replicato.", "A640cp:E1"),
])
def test_review_an_assertion_the_old_sentence_wide_negation_missed_is_caught(sentence, eid):
    got = A.asserted_unproven(sentence, _table())
    assert got is not None and got["eid"] == eid
    v, events = _table(), []
    assert A.ProseReview(v, A.prose_scope(v), A.SITE_OBJECTIONS, events)(sentence) == ""
    assert A.prose_counters(events)["objections_sentences_dropped"] == 1


@pytest.mark.parametrize("sentence", [
    "La Procura afferma che gli artifizi o raggiri sono integrati, ma Paolo Sarti lo nega.",
    "Per l'accusa gli artifizi o raggiri sono provati dall'annuncio; la difesa lo smentisce.",
    "Secondo la Procura, gli artifizi o raggiri sono integrati; Paolo Sarti nega di aver pubblicato l'annuncio.",
    "La controparte sostiene che l'elemento E3 sussiste, ma non lo prova.",
    "Gli artifizi o raggiri non sono integrati, mentre l'induzione di taluno in errore è provata.",   # E2 provato
])
def test_review_a_reported_claim_or_a_negated_clause_asserts_nothing(sentence):
    assert A.asserted_unproven(sentence, _table()) is None
    v, events = _table(), []
    assert A.ProseReview(v, A.prose_scope(v), A.SITE_OBJECTIONS, events)(sentence) == sentence
    assert not [e for e in events if e.get("motivo") == A.UNPROVEN_ASSERTED]


def test_review_the_check_lists_are_explicit_word_forms():
    cfg = A.config()["objection_answer_check"]
    assert "rest\\w*" not in cfg["negations"] and "contest\\w*" not in cfg["negations"]
    assert "resta" in cfg["negations"] and "contesta" in cfg["negations"]
    neg = A._answer_check()["neg"]
    for word in ("restituzione", "contesto", "difetto", "devoluzione", "mancia"):
        assert not neg.search(word), word
    for word in ("resta", "contesta", "deve", "manca", "non", "va"):
        assert neg.search(word), word


# --- (review) the tesi never quotes a counterparty's allegation as its fact ---------------------------------------

def test_review_an_escluso_tesi_never_quotes_the_procuras_allegation():
    inp = T._penal()
    only = A.analyse(inp, call=_Model(_penal_reply("escluso", ("S3",))))["analysis"]["linea_principale"]
    assert only["esito_in_gioco"] == "escluso"
    assert "Secondo la Procura" not in only["tesi"]
    # the verdict cites only the Procura's sentence; the line cites S5 (the client's denial): the line's sentence
    assert only["tesi"] == ("«con artifizi o raggiri» è escluso: «Paolo Sarti nega di aver mai pubblicato un falso "
                            "annuncio»")
    both = A.analyse(inp, call=_Model(_penal_reply("escluso", ("S3", "S5"))))["analysis"]["linea_principale"]
    assert both["tesi"] == only["tesi"]


def test_review_with_no_client_sentence_at_all_the_tesi_says_so():
    reply = _reply(_line("640 c.p.", T.E640[0], "Tesi.", ["S3"], ["A640cp.1"]),
                   elementi=[_el("640 c.p.", T.E640[0], "escluso", ["S3"]), _el("640 c.p.", T.E640[1], "da_provare"),
                             _el("640 c.p.", T.E640[2], "da_provare")])
    line = A.analyse(T._penal(), call=_Model(reply))["analysis"]["linea_principale"]
    assert line["stato"] == A.VERIFIED and line["tesi_codice"] is True
    assert line["tesi"] == "«con artifizi o raggiri» è escluso: " + A.rendering()["tesi"]["nessun_fatto"]


# --- (review) the side sets in the plural -----------------------------------------------------------------------------

@pytest.mark.parametrize("role,side", [("committenti", A.SIDE_CLAIMANT), ("i danneggiati", A.SIDE_CLAIMANT),
                                       ("promissari acquirenti", A.SIDE_CLAIMANT), ("indagati", A.SIDE_DEFENCE),
                                       ("convenute", A.SIDE_DEFENCE)])
def test_review_the_plural_roles_read_their_side(role, side):
    assert A.client_side({"parti": [_party(role)]})[0] == side


def test_review_role_words_outside_the_verified_frame_are_no_side():
    """The brief: claimant detection from the frame's verified role, unverified -> neither. The pipeline's role words
    (inp client_roles: intake, posture, extracted parties) are not the verified frame: a voided ruolo_nel_fatto with
    «committente» only there stays neither."""
    inp = A.make_input({"cliente": {"nome": "Marta Galli"},
                        "parti": [{"nome": "Marta Galli", "lato": "cliente", "ruolo_nel_fatto": ""},
                                  {"nome": "Edilnova S.r.l.", "lato": "controparte", "ruolo_nel_fatto": "appaltatore"}],
                        "rito": {"valore": "civile", "intake": "civile"}},
                       [], [{"id": "S1", "text": "Marta Galli ha commissionato i lavori."}], [], "Marta Galli",
                       ["committente"])
    assert inp["side"] == A.SIDE_NEUTRAL and inp["claimant"] is False


# --- the fixtures end to end ------------------------------------------------------------------------------------------

def _fixture_modules():
    import _fixture_79_15_0 as H
    import test_v79_15_1_fixtures as X
    return H, X


def _report(run, title):
    target = os.environ.get("FIXTURE_79_15_3_OUT")
    if not target:
        return
    H, _ = _fixture_modules()
    import test_v79_14_2_fixture as F
    step = [F._line(e, d, ms) for e, d, ms in run["events"] if e == "analysis_step"]
    mode = "a" if os.path.exists(target) and os.path.getsize(target) else "w"
    with open(target, mode, encoding="utf-8") as f:
        f.write(f"# {title}\n\n## §1 as delivered\n\n{H.section(run['answer'], 'sintesi')}\n\n"
                f"## §7 as delivered\n\n{H.section(run['answer'], 'obiezioni')}\n\n"
                f"## the analysis_step log line (production format)\n\n" + "\n".join(step) + "\n\n")


def test_fixture_civil_caparra(monkeypatch, tmp_path):
    H, X = _fixture_modules()
    run = H.run(monkeypatch, tmp_path, X.CIVIL_QA)
    _report(run, "v79.15.3 civil caparra fixture (tests/test_v79_15_1_fixtures.py CIVIL_QA: Anna Corsi, attore)")
    X._no_failed_steps(run)
    answer, step = run["answer"], run["by"]["analysis_step"][0][0]
    assert H.section(answer, "sintesi") == ("Linea principale: art. 1385 c.c. — «inadempiente è invece la parte che "
                                            "l'ha ricevuta» è da provare.")
    assert step["side"] == A.SIDE_CLAIMANT and step["claimant"] is True and step["side_why"] == "ruolo: attore"
    assert set(step["tesi_modello"]) == {"linea_principale"} and step["tesi_modello"]["linea_principale"]
    assert step["tesi_modello"]["linea_principale"] not in answer
    obiezioni = H.section(answer, "obiezioni")
    assert obiezioni == (f"**Obiezione 1.** {X.CIVIL.BASSI}.\n**Risposta.** L'elemento «inadempiente è invece la parte "
                         "che l'ha ricevuta» (art. 1385 c.c.) è da provare: «Anna Corsi ha versato a Ettore Bassi "
                         "15.000 euro a titolo di caparra confirmatoria».")


def test_fixture_penal_indagato(monkeypatch, tmp_path):
    H, X = _fixture_modules()
    run = H.run(monkeypatch, tmp_path, X.PENAL_ID)
    _report(run, "v79.15.3 penal indagato fixture (tests/test_v79_15_0_fixture_penal.py, the reply in the id form: "
                 "Dario Melis, indagato)")
    X._no_failed_steps(run)
    answer, step = run["answer"], run["by"]["analysis_step"][0][0]
    assert H.section(answer, "sintesi") == ("Linea principale: art. 646 c.p. — L'accusa deve provare «per procurare a "
                                            "sé o ad altri un ingiusto profitto» (art. 646 c.p.).")
    assert step["side"] == A.SIDE_DEFENCE and step["claimant"] is False
    assert step["tesi_modello"]["linea_principale"] not in answer
    obiezioni = H.section(answer, "obiezioni")
    assert obiezioni.startswith(f"**Obiezione 1.** {X.PENAL.PROCURA}.\n**Risposta.** L'elemento «per procurare a sé o "
                                "ad altri un ingiusto profitto» (art. 646 c.p.) non è provato: «Dario Melis nega di "
                                "aver trattenuto il denaro».")
    assert f"**Obiezione 2.** {X.PENAL.SOCIETA}.\n**Risposta.**" in obiezioni


def test_fixture_appalto_claimant(monkeypatch, tmp_path):
    H, X = _fixture_modules()
    run = H.run(monkeypatch, tmp_path, X.APPALTO)
    _report(run, "v79.15.3 appalto claimant fixture (tests/test_v79_15_1_fixtures.py APPALTO: Marta Galli, "
                 "committente, ruolo not verified)")
    X._no_failed_steps(run)
    answer, step = run["answer"], run["by"]["analysis_step"][0][0]
    assert run["state"]["case_frame"]["frame"].get("ruolo") in (None, {})
    assert step["side"] == A.SIDE_CLAIMANT and step["side_why"] == "ruolo_nel_fatto: committente"
    assert H.section(answer, "sintesi") == ("Linea principale: art. 1667 c.c. — «L'appaltatore è tenuto alla garanzia "
                                            "per le difformità e i vizi dell'opera» è da provare.")
    obiezioni = H.section(answer, "obiezioni")
    assert obiezioni.count("**Obiezione ") == 3 and "Esito verificato" not in obiezioni


def test_fixture_a_written_answer_asserting_the_contested_element_is_dropped(monkeypatch, tmp_path):
    """The civil fixture with the plan's objection answered «… l'elemento è integrato» while the verified table holds
    it «contestato»: that sentence dropped (objections_sentences_dropped), the answer's other sentence kept."""
    H, X = _fixture_modules()
    case = types.SimpleNamespace(**{k: getattr(X.CIVIL_QA, k) for k in dir(X.CIVIL_QA) if not k.startswith("__")})
    case.objections = [{"obiezione": "Ettore Bassi sosterrà di non dover restituire nulla ad Anna Corsi",
                        "risposta": "Inadempiente è invece la parte che l'ha ricevuta: l'elemento è integrato. Anna "
                                    "Corsi ha versato la caparra a Ettore Bassi."}]
    run = H.run(monkeypatch, tmp_path, case)
    X._no_failed_steps(run)
    c, answer = run["state"]["block_counters"], run["answer"]
    obiezioni = H.section(answer, "obiezioni")
    assert "l'elemento è integrato" not in answer
    assert "**Obiezione 2.** Ettore Bassi sosterrà di non dover restituire nulla ad Anna Corsi." in obiezioni
    assert "Anna Corsi ha versato la caparra a Ettore Bassi." in obiezioni
    events = [e for d, _ in run["by"].get("lawyer_method_prose_review") or [] for e in d["events"]]
    assert [e["elemento"] for e in events if e.get("motivo") == A.UNPROVEN_ASSERTED] == ["A1385cc:E4"]
    assert c["objections_sentences_dropped"] == 1
