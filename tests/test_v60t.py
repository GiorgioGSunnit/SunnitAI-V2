"""v60t: «provato» grounded in the case's own words, v60h's nouns condition gone -- the truffa topic's 640 c.p. linked to
sentences that narrate it in plain words now 3/3; the client's account (narrated or attributed) proves, a hedged or
hypothetical sentence does not, the counterparty's goes to the contradiction path (1); single shot, a body block needs a proven element or an open
question, the main line's anchor always written with its honest status (2); the «(da valutare)» marker never on the
procedure's lines nor on the main line's anchor (3). The model and the graph are stubbed."""

import os
import sys
from pathlib import Path

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.rag import answer_ledger, issue_records  # noqa: E402
from src.rag import interaction_copy as C  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from test_v60k import WRITER, _single  # noqa: E402

NORMS = flow.ANSWER_TITLES[1]
LEAD, PROC, END = flow.ANSWER_TITLES[0], flow.ANSWER_TITLES[3], flow.ANSWER_TITLES[4]
PROVATO, DA_PROVARE, NON_RISULTA = C.STATUS_PROVATO, C.STATUS_DA_PROVARE, C.STATUS_NON_RISULTA

# --- 1. «provato» grounded in the case's own words ------------------------------------------------------------------------

TRUFFA = ("Chiunque, con artifizi o raggiri, inducendo taluno in errore, procura a sé o ad altri un ingiusto profitto con "
          "altrui danno, è punito con la reclusione da sei mesi a tre anni e con la multa da euro 51 a euro 1.032.")
CASE = ("Sempronio si è presentato a Caio come consulente finanziario e gli ha mostrato falsi estratti conto. "
        "Convinto dell'affare, Caio gli ha consegnato 20.000 euro. Sempronio ha trattenuto il denaro e non si è più "
        "fatto trovare.")


def test_the_truffa_linked_in_plain_words_is_proven_three_of_three():
    reqs = issue_records._requisites({"_article": ("640", "c.p."), "s": {"plain_text": TRUFFA}}, {})
    assert [(issue_records._comma(r), r["kind"]) for r in reqs] == [(1, "requisito")] * 3    # three elements, co. 1
    sents = issue_records.sentences(CASE)
    assert all(s["verbatim"] and s["certainty"] == "accertato" for s in sents)
    by_id = {s["id"]: s for s in sents}
    for r in reqs:
        r["supporting"] = [s["id"] for s in sents]
        issue_records.settle(r, by_id)
    assert [r["status"] for r in reqs] == [PROVATO] * 3                    # v60h's nouns condition would give 0/3
    assert reqs[0]["why"] == "lo dice il caso (S1, accertato)"
    assert issue_records.verdict({"requisites": reqs}) == issue_records.APPLICABLE


def test_grounding_is_the_cases_own_words_and_a_contradiction_still_excludes():
    sents = issue_records.sentences("Caio ha consegnato il denaro a Sempronio.", ["Il denaro era in contanti."],
                                    ["Parte assistita assunta: Caio"])
    by_id = {s["id"]: s for s in sents}
    assert [(s["source"], issue_records.grounded(s)) for s in sents] == [("caso", True), ("precisazione", True),
                                                                          ("sistema", False)]
    assert not issue_records.verbatim("Caio ha consegnato i gioielli.", "Caio ha consegnato il denaro a Sempronio.")
    req = {"kind": "requisito", "text": "consegna", "facts_status": DA_PROVARE, "supporting": ["S3"], "contradicting": []}
    issue_records.settle(req, by_id)
    assert req["status"] == DA_PROVARE and "solo sistema" in req["why"]
    req = {**req, "supporting": ["S1"], "contradicting": ["S1"]}
    issue_records.settle(req, by_id)
    assert req["status"] == NON_RISULTA                                     # the contradiction narrated as fact wins


# --- the client's account proves; a hypothesis does not; the counterparty's goes to the contradiction path ----------------

PARTIES = {"analysis": {"fact_signals": {"facts": {"parties": [{"name": "Caio"}, {"name": "Sempronio Rossi"}]}}},
           "client": {"name": "Caio"}}


def test_hedged_and_hypothetical_sentences_are_ipotizzato_even_when_attributed():
    c = issue_records.certainty
    assert c("Caio sostiene che Sempronio gli ha mostrato falsi estratti conto.") == "allegato"
    assert c("Forse Sempronio sapeva della truffa.") == c("Sempronio potrebbe aver mentito.") == "ipotizzato"
    assert c("Se fosse vero, Sempronio avrebbe incassato il denaro.") == "ipotizzato"
    assert c("Caio sostiene che forse Sempronio sapeva.") == "ipotizzato"   # an attributed hypothesis: still one


def test_each_attributed_sentence_knows_its_speaker_and_side():
    names = ["Caio", "Sempronio Rossi"]
    assert flow.attributed_to("Caio sostiene che Sempronio lo ha ingannato.", names) == "Caio"
    assert flow.attributed_to("Sempronio Rossi afferma che il denaro era un prestito.", names) == "Sempronio Rossi"
    assert flow.attributed_to("Secondo Sempronio il denaro era un prestito.", names) == "Sempronio Rossi"
    assert flow.attributed_to("Il denaro sarebbe stato restituito.", names) == ""
    sents = issue_records.sentences("Caio sostiene che Sempronio gli ha mostrato falsi estratti conto. Sempronio "
                                    "afferma che il denaro era un prestito. Il denaro sarebbe stato restituito.")
    flow.mark_sides(sents, PARTIES)
    assert [(s["by"], s["side"]) for s in sents] == [("Caio", "cliente"), ("Sempronio Rossi", "controparte"), ("", "")]
    flow.mark_sides(sents, {**PARTIES, "client": {}, "analysis": {"fact_signals": {"facts": {
        "parties": PARTIES["analysis"]["fact_signals"]["facts"]["parties"]}}}})
    assert [s["side"] for s in sents] == ["", "", ""]                       # no client known: no side


def test_the_three_outcomes_of_a_linked_element():
    sents = issue_records.sentences("Caio sostiene che Sempronio gli ha mostrato falsi estratti conto. Sempronio "
                                    "afferma di avergli mostrato solo bilanci veri. Forse Sempronio sapeva.")
    flow.mark_sides(sents, PARTIES)
    by_id = {s["id"]: s for s in sents}

    def settled(sup, con=()):
        req = {"kind": "requisito", "text": "artifizi o raggiri", "facts_status": DA_PROVARE,
               "link": {"supporting": list(sup), "contradicting": list(con)}}
        issue_records.settle(req, by_id)
        return req
    assert settled(["S1"])["status"] == PROVATO                              # the client's account
    assert settled(["S3"])["status"] == DA_PROVARE and "solo ipotizzato" in settled(["S3"])["why"]
    theirs = settled(["S2"])                                                 # the counterparty's version, linked as support
    assert theirs["status"] == DA_PROVARE and theirs["supporting"] == [] and theirs["contradicting"] == ["S2"]
    assert settled(["S1", "S2"])["status"] == PROVATO                        # the client's account stands
    flow.mark_sides(sents, {**PARTIES, "client": {"name": "Sempronio Rossi"}})   # the other side's parere
    assert settled(["S2"])["status"] == PROVATO and settled(["S1"])["contradicting"] == ["S1"]


# --- 2. single shot: a block needs a proven element; the main line's anchor always written --------------------------------

def _rec(**kw):
    return {"verdict": DA_PROVARE, "requisites": [{"kind": "requisito", "comma": 1, "text": "x", "status": DA_PROVARE,
                                                   "supporting": ["S1"], "question": ""}], **kw}


def test_a_supporting_sentence_alone_no_longer_earns_a_block_in_single_shot():
    assert issue_records.block_due(_rec())                                  # the rounds: a supporting sentence
    assert not issue_records.block_due(_rec(needs_proof=True))              # single shot: no element proven
    proven = _rec(needs_proof=True)
    proven["requisites"].append({"kind": "requisito", "comma": 1, "text": "y", "status": PROVATO})
    assert issue_records.proven(proven) == 1 and issue_records.block_due(proven)
    asked = _rec(needs_proof=True)
    asked["requisites"][0]["question"] = "q1"
    assert issue_records.block_due(asked, open_questions=["q1"])            # a missing fact a question asks


def test_the_main_line_is_the_principale_topic_else_the_first():
    topics = [{"title": "A", "anchor": "1 c.c."}, {"title": "B", "anchor": "2 c.c.",
                                                   "hypotheses": [{"line": "principale"}]}]
    assert flow.main_line_topic(topics)["title"] == "B" and flow.main_line_topic(topics[:1])["title"] == "A"
    assert flow.main_line_topic([]) is None


def test_single_shot_writes_the_main_anchor_and_no_unproven_block(monkeypatch):
    state, seen = _single(monkeypatch, writer={**WRITER, NORMS: "[1] L'art. 1062 c.c. fa sorgere la servitù.\n\n"
                                                              "[2] L'art. 1158 c.c. vuole vent'anni.\n\n"
                                                              "[3] L'art. 1051 c.c. dà il passaggio."})
    state["questions"] = []                                                 # no open question: no block by it
    flow.refresh_records(state)
    text = flow.single_shot("Il caso.")["answer"]
    norms = dict(answer_ledger.sections_of(text))[NORMS]
    assert state["main_anchor"] == "1062 c.c."
    assert norms.strip().startswith("**Art. 1062 c.c.**\nEsito sui fatti del caso: da provare (requisiti provati: 0 su 2).")
    assert "**Art. 1158 c.c.**" not in norms and "**Art. 1051 c.c.**" not in norms      # 0 proven, none asked
    heads = {r["article"]: r for r in state["ledger_rows"] if r["requisite"] is None and r["comma"] is None}
    assert heads["1062 c.c."]["in_body"] is True
    assert heads["1158 c.c."]["in_body"] is False and heads["1051 c.c."]["in_body"] is False
    assert all(r.get("needs_proof") for r in state["records"].values())
    assert {label for label, r in state["records"].items() if r.get("anchor_due")} == {"1062 c.c."}


def test_the_main_anchor_comes_back_when_the_body_lost_it():
    draft = (f"**2. {NORMS}**\n\n**Art. 1158 c.c.**\n\nIl possesso dura.\n\n"
             "Norme considerate e non applicabili: art. 1062 c.c. (non risulta «fondi divisi»).\n\n**5. Conclusioni**\n\nX.")
    state = {"main_anchor": "1062 c.c.",
             "records": {"1062 c.c.": {"verdict": issue_records.NOT_APPLICABLE, "rubrica": "Destinazione del padre di "
                                       "famiglia", "requisites": [{"comma": 1, "text": "fondi divisi",
                                                                   "status": NON_RISULTA}]},
                         "1158 c.c.": {"verdict": DA_PROVARE, "requisites": []}}}
    log = {}
    text, rows = flow.ledger_out(draft, state, log)
    norms = dict(answer_ledger.sections_of(text))[NORMS]
    assert norms.strip().startswith("**Art. 1062 c.c. — Destinazione del padre di famiglia**\n"
                                    "Esito sui fatti del caso: non applicabile (requisiti provati: 0 su 1).\n\n"
                                    "**Art. 1158 c.c.**")
    assert log["main_anchor_added"] == "1062 c.c." and rows[0]["article"] == "1062 c.c." and rows[0]["in_body"]


# --- 3. the marker: never on the procedure's lines, never on the main line's anchor ----------------------------------------

def test_the_marker_reads_the_opening_and_the_conclusions_never_the_procedure_or_the_main_anchor():
    records = {"1061 c.c.": {"verdict": DA_PROVARE, "requisites": [{"status": DA_PROVARE}]},
               "1062 c.c.": {"verdict": DA_PROVARE, "requisites": [{"status": DA_PROVARE}]}}
    draft = (f"**1. {LEAD}**\n\nLeonardo invoca l'art. 1062 c.c. e l'art. 1061 c.c.\n\n"
             f"**3. {PROC}**\n\n- Termine (art. 1061 c.c.): decorso.\n\n"
             f"**4. {END}**\n\nL'esito dipende dall'art. 1061 c.c. e dall'art. 1062 c.c.")
    out = dict(answer_ledger.sections_of(flow.asserted_only(draft, {"records": records, "main_anchor": "1062 c.c."})))
    assert out[LEAD].strip() == "Leonardo invoca l'art. 1062 c.c. e l'art. 1061 c.c. (da valutare: art. 1061 c.c.)."
    assert out[PROC].strip() == "- Termine (art. 1061 c.c.): decorso."                # «Profili processuali»: never
    assert out[END].strip() == ("L'esito dipende dall'art. 1061 c.c. e dall'art. 1062 c.c. "
                                "(da valutare: art. 1061 c.c.).")
    assert flow.ASSERTING_TITLES == (LEAD, END)
