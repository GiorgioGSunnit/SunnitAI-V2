"""FACT_RECORD and USER_TEXT_UNCAPPED (22 Nov 2026): the main query's fact record, verified by code, and the test switch
that lifts the cuts on the user's text. No model here: the reply is a stand-in (build's `asker`)."""
import json
import time

import pytest

from src.rag import fact_record as fr
from src.rag import issue_records as ir
from src.rag import user_limits

LONG = ("Mario Rossi ha affidato alla società Alfa la ristrutturazione della casa nel marzo 2020, i lavori sono finiti "
        "nel dicembre 2020, nel gennaio 2026 sono comparse crepe nella scala e la società Alfa, interpellata subito, "
        "ha risposto che la garanzia è prescritta e che in ogni caso i lavori riguardavano un edificio già esistente.")
CASE = ("Risposta del chatbot: la linea principale è fondare la domanda sull'art. 1667 c.c. "
        "Non sono d'accordo con questa risposta. " + LONG + " Riformula la strategia tenendo conto di questo.")
REPLY = [
    {"citazione": "Risposta del chatbot: la linea principale è fondare la domanda sull'art. 1667 c.c.", "tipo": "estraneo"},
    {"citazione": "Non sono d'accordo con questa risposta.", "tipo": "estraneo"},
    {"citazione": "Mario Rossi ha affidato alla società Alfa la ristrutturazione della casa nel marzo 2020",
     "tipo": "fatto", "stato": "accertato"},
    {"citazione": "i lavori sono finiti nel dicembre 2020", "tipo": "fatto", "stato": "accertato"},
    {"citazione": "nel gennaio 2026 sono comparse crepe nella scala", "tipo": "fatto", "stato": "accertato"},
    {"citazione": "la società Alfa, interpellata subito, ha risposto che la garanzia è prescritta",
     "tipo": "fatto", "stato": "allegato", "chi": "società Alfa"},
    {"citazione": "in ogni caso i lavori riguardavano un edificio già esistente",
     "tipo": "fatto", "stato": "allegato", "chi": "società Alfa"},
    {"citazione": "Riformula la strategia tenendo conto di questo.", "tipo": "istruzione"},
]


def careful(_chunk):
    return list(REPLY)


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv(fr.FLAG, "true")


def _facts(record):
    return [f for f in record["facts"] if f["tipo"] == fr.FACT]


def test_the_record_separates_the_facts_and_keeps_the_user_words():
    record = fr.build(CASE, asker=careful)
    assert record["status"] == "ok" and record["version"] == fr.VERSION
    assert [f["id"] for f in record["facts"]] == [f"F{n}" for n in range(1, len(record["facts"]) + 1)]
    facts = _facts(record)
    assert len(facts) == 5                                      # one long sentence: five facts, none cut
    assert all(f["text"] in CASE and CASE[f["start"]:f["end"]] == f["text"] for f in record["facts"])
    assert not any("Risposta del chatbot" in f["text"] or "Riformula" in f["text"] for f in facts)
    last = facts[-1]                                            # the other party's second defence: its own fact
    assert last["text"] == "in ogni caso i lavori riguardavano un edificio già esistente"
    assert last["stato"] == ir.ALLEGATO and last["chi"] == "società Alfa" and last["source"] == fr.MODEL
    assert {f["tipo"] for f in record["facts"]} == {fr.FACT, fr.OTHER, fr.INSTRUCTION}
    json.dumps(record)                                          # a plain JSON: saved with the session's state


def test_the_numbered_sentences_come_from_the_record(on):
    state = {"fact_record": fr.build(CASE, asker=careful)}
    sents = ir.sentences(CASE, facts=fr.facts(state))
    assert [s["id"] for s in sents] == ["S1", "S2", "S3", "S4", "S5"]
    assert all(s["source"] == "caso" and s["verbatim"] for s in sents)
    assert [s["certainty"] for s in sents] == [ir.ACCERTATO] * 3 + [ir.ALLEGATO] * 2
    assert sents[3]["said_by"] == "società Alfa" and sents[0]["fact"] == "F3" and "said_by" not in sents[0]
    assert max(len(s["text"]) for s in sents) < ir.SENTENCE_CHARS       # no fact needs the cut


def test_with_the_flag_off_nothing_changes():
    state = {"fact_record": fr.build(CASE, asker=careful)}
    assert fr.facts(state) == [] and fr.start(CASE) is None
    rule = ir.sentences(CASE)
    assert rule == ir.sentences(CASE, facts=fr.facts(state))
    # today: three pieces («c.c.» ends no sentence) -- the pasted line is a fact, the long sentence is cut
    assert len(rule) == 3 and rule[0]["text"].startswith("Risposta del chatbot")
    assert len(rule[1]["text"]) == ir.SENTENCE_CHARS and rule[1]["certainty"] == ir.ACCERTATO


def test_a_quote_that_is_not_the_text_is_dropped_and_the_rule_gives_the_sentence_back():
    text = "Caio ha consegnato il motorino a Tizio il 3 marzo 2024. Tizio non ha pagato il prezzo."
    record = fr.build(text, asker=lambda _c: [
        {"citazione": "Caio consegnava un motorino a Tizio nel marzo del 2024", "tipo": "fatto", "stato": "accertato"},
        {"citazione": "Tizio non ha pagato il prezzo.", "tipo": "fatto", "stato": "accertato"}])
    assert record["counters"]["quote_not_found"] == 1 and record["counters"]["rule_whole"] == 1
    assert [(f["text"], f["source"]) for f in record["facts"]] == [
        ("Caio ha consegnato il motorino a Tizio il 3 marzo 2024.", fr.RULE), ("Tizio non ha pagato il prezzo.", fr.MODEL)]


def test_the_model_can_make_a_fact_less_certain_never_more():
    text = "Caio sostiene che il muro era già crepato. Il muro è crollato il 5 maggio."
    assert ir.certainty("Caio sostiene che il muro era già crepato.") == ir.ALLEGATO
    record = fr.build(text, asker=lambda _c: [
        {"citazione": "Caio sostiene che il muro era già crepato.", "tipo": "fatto", "stato": "accertato"},
        {"citazione": "Il muro è crollato il 5 maggio.", "tipo": "fatto", "stato": "ipotizzato"}])
    assert [f["stato"] for f in record["facts"]] == [ir.ALLEGATO, ir.IPOTIZZATO]
    assert record["counters"]["status_kept_cautious"] == 1


def test_values_outside_the_closed_lists_fall_back_to_the_rule():
    text = "Tizio ha venduto il fondo a Caio nel 2019. Caio afferma di avere pagato tutto."
    record = fr.build(text, asker=lambda _c: [
        {"citazione": "Tizio ha venduto il fondo a Caio nel 2019.", "tipo": "evento", "stato": "certissimo", "chi": "Tizio"},
        {"citazione": "Caio afferma di avere pagato tutto.", "tipo": "fatto", "stato": "allegato", "chi": "Sempronio"}])
    first, second = record["facts"]
    assert first["tipo"] == fr.FACT and first["stato"] == ir.ACCERTATO and "chi" not in first    # no speaker for a fact
    assert second["stato"] == ir.ALLEGATO and "chi" not in second                               # a name not in the text
    assert record["counters"]["kind_unknown"] == 1 and record["counters"]["status_unknown"] == 1
    assert record["counters"]["speaker_dropped"] == 2


def test_a_stretch_left_out_comes_back_from_eight_words_on_and_connectives_do_not():
    record = fr.build(LONG, asker=lambda _c: [
        {"citazione": "Mario Rossi ha affidato alla società Alfa la ristrutturazione della casa nel marzo 2020",
         "tipo": "fatto", "stato": "accertato"},
        {"citazione": "la società Alfa, interpellata subito, ha risposto che la garanzia è prescritta",
         "tipo": "fatto", "stato": "allegato", "chi": "società Alfa"},
        {"citazione": "in ogni caso i lavori riguardavano un edificio già esistente", "tipo": "fatto", "stato": "allegato"}])
    texts = [f["text"] for f in record["facts"]]
    assert "i lavori sono finiti nel dicembre 2020, nel gennaio 2026 sono comparse crepe nella scala" in texts
    assert not any(t in ("e che", "e che in ogni caso") for t in texts)         # the join between two facts: not a fact
    assert record["counters"]["rule_stretch"] == 1 and record["counters"]["left_out_chars"] > 0


def test_spaces_and_typographic_quotes_do_not_break_a_quote():
    text = "L’impresa  ha consegnato\nl’opera «finita» nel 2020. Poi nulla."
    record = fr.build(text, asker=lambda _c: [
        {"citazione": "L'impresa ha consegnato l'opera \"finita\" nel 2020.", "tipo": "fatto", "stato": "accertato"},
        {"citazione": "Poi nulla.", "tipo": "fatto", "stato": "accertato"}])     # too short to be a sentence
    assert [f["text"] for f in record["facts"]] == ["L’impresa ha consegnato l’opera «finita» nel 2020."]
    assert record["counters"]["quote_too_short"] == 1


def test_no_reply_no_record_and_a_piece_that_fails_is_read_by_the_rule(monkeypatch):
    def broken(_chunk):
        raise ValueError("no JSON")
    assert fr.build(CASE, asker=broken) is None and fr.build("", asker=careful) is None

    def slow(_chunk):
        time.sleep(0.3)
        return list(REPLY)
    assert fr.build(CASE, asker=slow, deadline=0.0) is None                     # late: no record

    monkeypatch.setattr(fr, "config", lambda: {**json.load(open(fr.DATA_FILE, encoding="utf-8")), "chunk_chars": 100})
    seen = []

    def half(chunk):
        seen.append(chunk)
        if chunk.startswith("Mario Rossi"):
            raise ValueError("no JSON")
        return [x for x in REPLY if x["citazione"] in chunk]
    record = fr.build(CASE, asker=half)
    assert len(seen) > 1 and record["status"] == "partial" and record["counters"]["chunks_failed"] == 1
    assert any(f["source"] == fr.RULE and f["text"] == LONG for f in record["facts"])     # the failed piece: whole


def test_start_runs_beside_and_gives_the_record(on, monkeypatch):
    monkeypatch.setattr(fr, "ask", careful)
    pending = fr.start(CASE)
    assert pending is not None and len(_facts(pending.result())) == 5
    monkeypatch.setattr(fr, "build", lambda case: (_ for _ in ()).throw(RuntimeError("down")))
    assert fr.start(CASE).result() is None                                       # any failure: no record


def test_the_record_speaker_names_a_party_of_the_case():
    from src.rag import interactive_flow as flow
    names = ["Mario Rossi", "Alfa S.r.l."]
    assert flow.said_by_party({"said_by": "società Alfa"}, names) == "Alfa S.r.l."
    assert flow.said_by_party({"said_by": "il vicino"}, names) == "" and flow.said_by_party({}, names) == ""


# --- USER_TEXT_UNCAPPED -------------------------------------------------------------------------------------------

def test_the_cuts_on_the_user_text_stand_until_the_switch_is_on(monkeypatch):
    from src.rag import defensive_generation as dg
    assert user_limits.cap(260) == 260 and user_limits.most(80) == 80
    many = " ".join(f"Il giorno {n} Tizio ha pagato la rata numero {n} a Caio." for n in range(1, 101))
    huge = "Tizio ha consegnato la merce. " * 600                                # 18,000 characters
    assert len(ir.sentences(LONG)[0]["text"]) == ir.SENTENCE_CHARS
    assert len(ir.sentences(many)) == ir.SENTENCES_MAX
    assert dg._case_text(huge, "").endswith("[... documento troncato ...]")
    monkeypatch.setenv(user_limits.FLAG, "true")
    assert user_limits.cap(260) is None and user_limits.most(80) > 10 ** 9
    assert ir.sentences(LONG)[0]["text"] == LONG
    assert len(ir.sentences(many)) == 100
    assert dg._case_text(huge, "") == huge
