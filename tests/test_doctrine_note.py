"""The doctrine note and the Fonti line under a chat answer (Oct 2026).

A test of 21 answers showed: a 200-word note under "brief" answers; notes
written from commentary on another area of law (truffa contrattuale under a
civil question) or saying only that the extracts contain nothing relevant; and
a Fonti line listing every commentary hit, with section "names" that are
internal ids or text fragments ('1386.0.0', '2011) , . Difforme, Cass. pen.').
"""

import io
import os

import pytest

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

import src.rag.nodes.synthesis as S  # noqa: E402


# --- The rules on their own -------------------------------------------------

def test_a_note_that_says_nothing_is_recognised():
    assert S._note_is_empty("NESSUNA_NOTA")
    assert S._note_is_empty("")
    assert S._note_is_empty(
        "Gli estratti forniti non contengono informazioni specifiche riguardanti i termini "
        "per l'impugnazione di un licenziamento."
    )
    assert S._note_is_empty("La dottrina rileva che i temi trattati non sono pertinenti alla domanda.")


def test_a_real_note_is_kept_even_when_it_says_non_contengono():
    assert not S._note_is_empty(
        "Secondo il Codice Civile Commentato, le norme sulla caparra non contengono un elenco "
        "tassativo delle ipotesi di recesso."
    )


def test_note_size_grows_with_the_length_setting():
    assert sorted(S._NOTE_LENGTH) == sorted(S._NOTE_TOKENS) == [1, 2, 3]
    assert S._NOTE_TOKENS[1] < S._NOTE_TOKENS[2] < S._NOTE_TOKENS[3]


def test_only_the_works_the_note_names_are_listed():
    cites = [{"document_name": "Codice Penale Commentato ADE"},
             {"document_name": "La Fiscalità delle Società IAS IFRS"}]
    note = "Secondo il Codice Penale Commentato, la truffa contrattuale presuppone..."
    assert [c["document_name"] for c in S._sources_named_in(note, cites)] == ["Codice Penale Commentato ADE"]
    # A note that names no work keeps them all: it was written from these extracts.
    assert S._sources_named_in("La dottrina rileva che...", cites) == cites


def test_fonti_shows_articles_for_codes_and_hides_internal_ids():
    code = {"document_name": "Codice Civile 2026",
            "sections": [{"name": "1386.0.0"}, {"name": "1385.0.0"}, {"name": "1386.2_1"}]}
    ruling = {"document_name": "Ordinanza sul ricorso 28041-2019",
              "sections": [{"name": "corte_cassazione.6"}, {"name": "9.23.0"}]}
    commentary = {"document_name": "Codice Penale Commentato ADE",
                  "sections": [{"name": "2011) , . Difforme, Cass. pen., sez. V"}]}
    assert S._fonti_entry(code) == "Codice Civile 2026 sezioni: art. 1386, art. 1385"
    assert S._fonti_entry(ruling) == "Ordinanza sul ricorso 28041-2019"
    assert S._fonti_entry(commentary) == "Codice Penale Commentato ADE"
    assert S._fonti_entry({"document_name": "Codice Penale 2026",
                           "sections": [{"name": "371-ter.0.0"}]}) == "Codice Penale 2026 sezioni: art. 371-ter"


# --- Through the answer step ------------------------------------------------

LAW = "La caparra confirmatoria ... in caso di inadempimento l'altra parte può recedere dal contratto."
MAIN_ROWS = [{
    "d": {"id": "LEGAL_DOC::cc", "name": "Codice Civile 2026"},
    "s": {"id": "DOCUMENT_SECTION::cc::1385", "name": "1385.0.0", "plain_text": LAW},
}]
COMMENTARY_ROWS = [
    {"d": {"id": "LEGAL_DOC::cpc", "name": "Codice Penale Commentato ADE"},
     "s": {"id": "DOCUMENT_SECTION::cpc::x1", "name": "In tema di truffa contrattuale, l'ingiusto profi",
           "plain_text": "In tema di truffa contrattuale, l'ingiusto profitto e il danno correlativo ..."}},
    {"d": {"id": "LEGAL_DOC::ias", "name": "La Fiscalità delle Società IAS IFRS"},
     "s": {"id": "DOCUMENT_SECTION::ias::x2", "name": "r. risoluzione n. 232/E del 22 agosto 2007",
           "plain_text": "La risoluzione n. 232/E del 22 agosto 2007 dell'Agenzia delle Entrate ..."}},
]
MAIN_ANSWER = ("La caparra confirmatoria consente di recedere dal contratto (Fonte: Codice Civile 2026, "
               "sezione: 1385.0.0).\n\nIn definitiva, la caparra confirmatoria rafforza il vincolo.")


@pytest.fixture
def answer_step(monkeypatch):
    calls = []

    def run(note_reply, length=1, raw_result=MAIN_ROWS):
        def fake_chat(messages, max_tokens=None, stop=None):
            system = messages[0].content
            is_note = "dottrina giuridica" in system
            calls.append({"note": is_note, "system": system, "max_tokens": max_tokens})
            return note_reply if is_note else MAIN_ANSWER

        monkeypatch.setattr(S, "_call_chat", fake_chat)
        monkeypatch.setattr(S, "rerank_results", lambda query, rows: rows)
        monkeypatch.setattr(S, "log_cypher_event", lambda *a, **k: None)
        monkeypatch.setattr(S, "vlog", lambda *a, **k: None)
        monkeypatch.setattr(S, "open", lambda *a, **k: io.StringIO(), raising=False)
        return S.synthesize_answer({
            "query": "Che differenza c'è tra caparra confirmatoria e caparra penitenziale?",
            "session_language": "it", "tone": 2, "standing": 2, "response_length": length,
            "raw_result": raw_result, "special_dottrina_rows": COMMENTARY_ROWS,
        })

    run.calls = calls
    return run


def test_brief_answers_get_a_short_note(answer_step):
    out = answer_step("Secondo il Codice Penale Commentato, la caparra ha natura reale.", length=1)
    note_call = [c for c in answer_step.calls if c["note"]][0]
    assert note_call["max_tokens"] == S._NOTE_TOKENS[1]
    assert S._NOTE_LENGTH[1] in note_call["system"]
    assert "**Nota dottrinale:**" in out["answer"]


def test_an_irrelevant_note_is_dropped_with_its_sources(answer_step):
    out = answer_step("NESSUNA_NOTA", length=2)
    assert "Nota dottrinale" not in out["answer"]
    assert "NESSUNA_NOTA" not in out["answer"]
    names = {c["document_name"] for c in out["citations"]}
    assert names == {"Codice Civile 2026"}
    assert "Commentato" not in out["answer"] and "IAS" not in out["answer"]


def test_fonti_and_side_panel_list_the_same_documents(answer_step):
    out = answer_step("Secondo il Codice Penale Commentato, la caparra ha natura reale.", length=3)
    names = [c["document_name"] for c in out["citations"]]
    assert names == ["Codice Civile 2026", "Codice Penale Commentato ADE"]   # not the IAS text
    fonti = out["answer"].split("Fonti: ", 1)[1].split("\n")[0]
    assert fonti == "Codice Civile 2026 sezioni: art. 1385, Codice Penale Commentato ADE"


def test_commentary_alone_and_irrelevant_gives_the_not_found_reply(answer_step):
    out = answer_step("NESSUNA_NOTA", length=2, raw_result=[])
    assert out["answer"] == S._NOTHING_FOUND["it"]
    assert out["citations"] == []
