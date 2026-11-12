"""v60j (the v69 sim): the third condition of an earned §2 block -- a missing fact an open round-1 question asks (1);
the institution named for an article cleared by its rubrica, the code's institutions index, its text or the record's
names before it is marked, the source logged (2). The model and the graph are stubbed."""

import os
import sys
from pathlib import Path

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.rag import answer_ledger, issue_records, record_check, writer_rules  # noqa: E402
from src.rag import interaction_copy as C  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from test_v60f import _graded_state  # noqa: E402
from test_v60g import _answer, _ctx  # noqa: E402

NORMS = flow.ANSWER_TITLES[1]
NT_PREFIX = C.OTHER_NORMS_LINE.split("{")[0]


# --- 1. an open round-1 question asking a missing fact earns the block ---------------------------------------------------

def _asked(question="atto_separazione"):
    return {"requisites": [{"kind": "requisito", "comma": 1, "text": "fondi divisi", "status": C.STATUS_DA_PROVARE,
                            "supporting": [], "question": question}]}


def test_a_missing_fact_an_open_question_asks_earns_the_block():
    rec = _asked()
    assert issue_records.asked(rec, ["atto_separazione"]) == ["atto_separazione"]
    assert issue_records.block_due(rec, open_questions=["atto_separazione"])
    assert issue_records.line_reason(rec, open_questions=["atto_separazione"]) is None
    assert not issue_records.block_due(rec, open_questions=[])                        # answered: no longer asked
    assert not issue_records.block_due(_asked(""), open_questions=["atto_separazione"])  # nothing asked for it
    not_applicable = {"requisites": [{**_asked()["requisites"][0], "status": C.STATUS_NON_RISULTA}]}
    assert not issue_records.block_due(not_applicable, open_questions=["atto_separazione"])
    assert flow.open_questions({"questions": [{"id": "a"}, {"id": "b"}], "answered": {"b": "x"}}) == ["a"]
    assert "con un fatto mancante che una domanda aperta chiede" in writer_rules.system_rules("Leonardo")


def test_the_answer_writes_the_article_the_lawyer_asked_about_and_names_its_question(monkeypatch):
    state = _graded_state(monkeypatch)
    seen = {}
    writer = {flow.ANSWER_TITLES[0]: "Leonardo può far valere la servitù.",
              NORMS: "[1] L'art. 1062 c.c. fa sorgere la servitù per destinazione.\n\n"
                     "[2] L'art. 1061 c.c. vuole opere visibili e permanenti: da verificare quali opere esistevano."}
    text, _ = _answer(monkeypatch, state, writer, seen=seen)          # no sentence linked: only the open questions
    norms = dict(answer_ledger.sections_of(text))[NORMS]
    assert "**Art. 1062 c.c." in norms and "**Art. 1061 c.c." in norms and "da accertare quali opere" in norms  # v60q
    assert not any(line.startswith(NT_PREFIX) and ("1062" in line or "1061" in line) for line in norms.split("\n"))
    schede = " ".join(str(j) for j in seen["jobs"])
    assert "domanda aperta" in schede                                   # the writer knows which question asks it
    assert state["ledger"]["unaccounted"] == [] and state["ledger"]["unreasoned"] == []


# --- 2. the institution named for an article: rubrica, index, text, record -----------------------------------------------

RUBRICA_1079 = "Accertamento della servitù e altri provvedimenti di tutela"
TEXT_1079 = ("Il titolare della servitù può farne riconoscere in giudizio l'esistenza contro chi ne contesta l'esercizio "
             "e può far cessare gli eventuali impedimenti e turbative.")
UNITS = [{"name": "Delle azioni a difesa delle servitù", "rubricated": "", "level": "CAPO", "articles": ["1079"]},
         {"name": "Delle servitù prediali", "rubricated": "", "level": "TITOLO", "articles": ["1027", "1079"]}]
CONFESSORIA = "L'azione confessoria ex art. 1079 c.c. tutela il passaggio di Leonardo."


def _ctx_1079(index):
    return _ctx(rubriche={"1079 c.c.": RUBRICA_1079}, texts={"1079 c.c.": TEXT_1079}, allowed={"1079 c.c."},
                index=index)


def test_the_index_clears_the_azione_confessoria_and_the_source_is_logged():
    out, log = record_check.check([("Conclusioni", CONFESSORIA)], _ctx_1079(lambda code: UNITS if code == "c.c." else None))
    assert "(da verificare" not in out[0][1] and log["marked"] == []
    assert log["cleared"] == [{"article": "1079 c.c.", "institution": "l'azione confessoria", "by": "indice"}]


def test_an_institution_all_sources_miss_is_marked_with_what_was_tried():
    sentence = "La responsabilità del vettore marittimo ex art. 1079 c.c. copre il carico."
    out, log = record_check.check([("Conclusioni", sentence)], _ctx_1079(lambda code: UNITS))
    assert "(da verificare: art. 1079 c.c., rubrica «" in out[0][1]
    assert log["marked"][0]["missed"] == list(record_check.CLEAR_SOURCES) and log["marked"][0]["index"] is True
    out, log = record_check.check([("Conclusioni", CONFESSORIA)], _ctx_1079(lambda code: None))   # no index built
    assert "(da verificare: art. 1079 c.c." in out[0][1]
    assert log["marked"][0]["index"] is False and "indice" not in log["marked"][0]["missed"]
    other = [{"name": "Delle azioni a difesa della proprietà", "level": "CAPO", "articles": ["948", "949"]}]
    out, _ = record_check.check([("Conclusioni", CONFESSORIA)], _ctx_1079(lambda code: other))    # 1079 in no unit
    assert "(da verificare: art. 1079 c.c." in out[0][1]


def test_each_source_clears_in_its_order():
    none = lambda code: None                                                                     # noqa: E731
    cases = (("L'accertamento della servitù ex art. 1079 c.c. va chiesto subito.", none, "rubrica"),
             ("La turbativa del passaggio ex art. 1079 c.c. va fatta cessare.", none, "testo"),
             ("L'azione confessoria ex art. 1079 c.c. tutela il passaggio.", lambda code: UNITS, "indice"))
    for sentence, index, by in cases:
        _, log = record_check.check([("Conclusioni", sentence)], _ctx_1079(index))
        assert [c["by"] for c in log["cleared"]] == [by], sentence
    ctx = {**_ctx_1079(none), "names": {"1079 c.c.": "Azione confessoria"}}                    # the record's own name
    _, log = record_check.check([("Conclusioni", CONFESSORIA)], ctx)
    assert [c["by"] for c in log["cleared"]] == ["record"] and log["marked"] == []
