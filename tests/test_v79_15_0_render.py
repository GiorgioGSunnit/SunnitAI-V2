"""v79.15.0 build B -- the rendering pieces with ANALYSIS_STEP on (src/rag/analysis_step.py: view, summary_text,
norms_tables, solutions_text, conclusions_text, answer_statements, the Q&A filter, the legacy comparison), on
synthetic analyses (invented names; the build A test cases). No model: every model path and every outbound socket
refused (tests/test_v79_15_0_analysis.py's sealing).

1. «da determinare»: a step not ok gives the leading line «da determinare (linea non verificata)» in §1, §6 and §9,
   no element table, no count -- never another line in its place
2. §4: one table per line article from the verified verdicts; a remedy / general-part article named, no table, no
   count; the counts only the line article's
3. the Q&A filter: a sentence stating what a norm provides is kept only with a verified affermazione_di_diritto on the
   same article number and at least 60% of the sentence's tokens (the denominator: the sentence's tokens)
4. the legacy comparison: legacy_disagrees_line, legacy_disagrees_esito
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import test_v79_15_0_analysis as T  # noqa: E402  (its autouse sealing fixture is re-exported below)
from src.rag import analysis_step as A  # noqa: E402

_sealed = T._sealed


def _verified_penal():
    inp = T._penal()
    reply = T._reply(T._line("640 c.p.", T.E640[0], "Gli artifizi non sono provati.", ("S3", "S5"), ["A640cp.1"]),
                     elementi=[T._el("640 c.p.", T.E640[0], "provato", ("S3",)),
                               T._el("640 c.p.", T.E640[1], "provato", ("S7",)),
                               T._el("640 c.p.", T.E640[2], "escluso", ("S5",))],
                     affermazioni=[{"testo": "L'art. 640 c.p. punisce chi, con artifizi o raggiri, inducendo taluno in "
                                             "errore, procura a sé o ad altri un ingiusto profitto con altrui danno.",
                                    "cita_norma": ["A640cp.1"]}])
    result = A.analyse(inp, call=T._Model(reply))
    return A.view(result, inp), result


# --- 1. «da determinare» ----------------------------------------------------------------------------------------------

def test_a_step_not_ok_is_the_undetermined_line_everywhere():
    inp = T._penal()
    for result in (None, {"status": "failed", "analysis": None}, {"status": "timeout"}):
        v = A.view(result, inp)
        assert A.summary_text(v) == "Linea principale: da determinare (linea non verificata)."
        assert "da determinare (linea non verificata)" in A.solutions_text(v)
        assert A.solutions_text(v).count("\n") == 2               # the head, the rule, the one undetermined row
        assert A.conclusions_text(v) == "**Linea principale**: da determinare (linea non verificata)."
        tables = A.norms_tables(v)
        assert tables.startswith(A.LABEL_NO_TABLE) and " su " not in tables
        assert "art. 62-bis c.p." in tables                       # the fitted general-part norm, named only
        assert A.answer_strategy(v) == ["La linea principale è: da determinare (linea non verificata)."]
        assert A.statements_block(v) == ""


def test_an_undetermined_line_from_the_step_prints_the_label_not_the_rejected_line():
    inp = T._civil()
    bad = T._reply(T._line("1453 c.c.", T.E1453[0], "La risoluzione.", ("S4",), ["A1453cc.1"]))
    result = A.analyse(inp, call=T._Model(bad, bad))
    assert result["counters"]["analysis_frame_charge_missing"] == 1      # the frame's 1385 c.c. kept, no tesi
    v = A.view(result, inp)
    assert A.summary_text(v) == "Linea principale: art. 1385 c.c. (addebito del quadro, linea non verificata)."
    assert "1453" not in A.summary_text(v) and "1453" not in A.solutions_text(v)
    assert "1453" not in A.conclusions_text(v)
    tables = A.norms_tables(v)
    assert "**Art. 1385 c.c.**" in tables and "**Art. 1453 c.c.**" not in tables
    assert "da provare 3 su 3" in tables


# --- 2. §4 -------------------------------------------------------------------------------------------------------------

def test_the_tables_are_the_verified_verdicts_and_the_remedies_are_named_only():
    v, result = _verified_penal()
    assert result["counters"]["analysis_provato_downgraded"] == 1         # S3: «Secondo la Procura, …»
    tables = A.norms_tables(v)
    assert "**Art. 640 c.p.** — Truffa" in tables and "| Elemento | Fatti citati | Esito |" in tables
    assert "| con artifizi o raggiri | Secondo la Procura" in tables and "| contestato (allegazione " in tables
    assert "| inducendo taluno in errore | Paolo Sarti dichiara di aver incassato" in tables and "| provato |" in tables
    assert "| escluso |" in tables
    assert "Esiti dell'art. 640 c.p.: provato 1 su 3; contestato 1 su 3; da provare 0 su 3; escluso 1 su 3." in tables
    assert "**Rimedi e parte generale collegati alla linea**: art. 62-bis c.p. (Circostanze attenuanti generiche)." \
        in tables
    assert "**Art. 62-bis c.p.**" not in tables and "**Art. 641 c.p.**" not in tables
    counts = [x for x in tables.splitlines() if " su " in x]
    assert counts and all("640 c.p." in x for x in counts)


def test_the_summary_solutions_and_conclusions_carry_the_verified_line_only():
    v, _ = _verified_penal()
    assert A.summary_text(v) == ("Linea principale: art. 640 c.p. — Gli artifizi non sono provati (elemento in gioco: "
                                 "«con artifizi o raggiri»).")
    sol = A.solutions_text(v)
    assert "| Linea principale | art. 640 c.p. | con artifizi o raggiri | Gli artifizi non sono provati |" in sol
    assert "Contenimento" not in sol and "Ammettere" not in sol
    concl = A.conclusions_text(v)
    assert concl.splitlines()[0].startswith("**Linea principale**: art. 640 c.p.")
    assert "**Esiti dell'art. 640 c.p.**: provato 1 su 3" in concl
    assert "**Contestati**: «con artifizi o raggiri» (contestato (allegazione " in concl


def test_the_counts_never_trip_the_count_sentence_rule():
    from src.rag import lawyer_method as M
    v, _ = _verified_penal()
    text = A.norms_tables(v) + "\n" + A.conclusions_text(v)
    assert M.count_left(text) == []


# --- 3. the Q&A filter ------------------------------------------------------------------------------------------------

STATEMENT = {"testo": "Se inadempiente è la parte che ha ricevuto la caparra, l'altra può recedere dal contratto ed "
                      "esigere il doppio della caparra.", "articoli": ["1385 c.c."], "cita_norma": ["A1385cc.2"]}
VIEW = {"statements": [STATEMENT]}


def test_the_tokens_and_the_overlap_denominator():
    assert A.tokens("L'art. 1385 c.c. consente all'altra parte di recedere.") == {"consente", "altra", "parte",
                                                                                   "recedere"}
    # 3 of the sentence's 4 tokens in the statement: 0.75 -- the denominator the sentence's own tokens
    assert A.overlap("La parte può recedere oggi.", STATEMENT["testo"]) == 0.75
    assert A.overlap("", STATEMENT["testo"]) == 0.0


def test_a_statement_with_a_verified_twin_is_kept():
    s = ("L'art. 1385 c.c. consente alla parte non inadempiente di recedere dal contratto ed esigere il doppio della "
         "caparra ricevuta dall'altra parte.")
    log = []
    assert A.qa_filter([s], VIEW, log) == [s] and log == []


def test_a_statement_with_no_verified_twin_is_removed_and_logged():
    s = ("L'art. 1385 c.c. prevede che, se la parte non inadempiente preferisce domandare l'esecuzione o la risoluzione "
         "del contratto, il risarcimento del danno è regolato dalle norme generali.")
    log = []
    assert A.qa_filter([s], VIEW, log) == []
    assert log and log[0]["articoli"] == ["1385 c.c."] and log[0]["overlap"] < 0.6


def test_the_same_words_on_another_article_are_removed():
    s = ("L'art. 1453 c.c. consente alla parte non inadempiente di recedere dal contratto ed esigere il doppio della "
         "caparra ricevuta dall'altra parte.")
    assert A.qa_filter([s], VIEW) == []


def test_a_sentence_citing_no_article_or_stating_no_rule_is_kept():
    plain = "Anna Corsi ha versato la caparra il 3 marzo."
    named = "Il caso riguarda l'art. 1453 c.c."
    assert A.qa_filter([plain, named], {"statements": []}) == [plain, named]


def test_the_statements_block_prints_each_verified_statement_with_its_article():
    assert A.statements_block(VIEW) == ("**Affermazioni di diritto verificate**\n\n- Se inadempiente è la parte che ha "
                                        "ricevuto la caparra, l'altra può recedere dal contratto ed esigere il doppio "
                                        "della caparra (art. 1385 c.c.).")


# --- 4. the legacy comparison -----------------------------------------------------------------------------------------

def test_the_legacy_comparison_counts_the_line_and_the_verdicts():
    v, _ = _verified_penal()
    records = {"640 c.p.": {"requisites": [{"id": "R1", "text": T.E640[0], "status": "provato"},
                                           {"id": "R2", "text": T.E640[1], "status": "provato"},
                                           {"id": "R3", "text": T.E640[2], "status": "da provare"}]}}
    same = A.compare(v, {"linea_principale": {"articolo": "640 c.p."}}, records)
    assert same["legacy_disagrees_line"] == 0
    assert same["legacy_disagrees_esito"] == 2          # R1: provato vs contestato; R3: da provare vs escluso
    other = A.compare(v, {"linea_principale": {"articolo": "62-bis c.p."}}, records)
    assert other["legacy_disagrees_line"] == 1
    undetermined = A.compare(A.view(None, T._penal()), {"linea_principale": {"articolo": "640 c.p."}}, records)
    assert undetermined["legacy_disagrees_line"] == 1 and undetermined["legacy_disagrees_esito"] == 0
