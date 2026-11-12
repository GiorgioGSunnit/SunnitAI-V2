"""v60i: noise down the line -- a §2 block only where the record earns one, every other article one line with its reason
(1); role-aware verdicts, the counterparty's failing article as the client's argument (2); the c.p.a. only in
administrative frames (3); the user's own message never a trigger (4); the usucapione's and the trascrizione's own
triggering facts (5); the «non si applica» detector's new phrasings (6); articled prepositions in party references (7);
role phrases that set the client (8); the labour dispute's procedural facts as a closed list (9). The model and the
graph are stubbed."""

import os
import sys
from pathlib import Path

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.rag import (answer_ledger, fact_signals, issue_records, party_names, record_check,  # noqa: E402
                     render_computed, writer_rules)
from src.rag import interaction_copy as C  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
import test_interactive_flow as base  # noqa: E402
from test_v60f import _graded_state  # noqa: E402
from test_v60g import _answer  # noqa: E402

NORMS = flow.ANSWER_TITLES[1]
NA_PREFIX = C.NOT_APPLICABLE_LINE.split("{")[0]
NT_PREFIX = C.OTHER_NORMS_LINE.split("{")[0]


def _rec(statuses, supporting=(), relied=""):
    reqs = [{"id": f"R{i}", "kind": "requisito", "comma": 1, "text": f"elemento numero {i} della norma",
             "status": s, "supporting": list(supporting) if i == 1 else []} for i, s in enumerate(statuses, 1)]
    rec = {"article": "1 c.c.", "requisites": reqs}
    if relied:
        rec["relied_by"] = relied
    return rec


# --- 1. a §2 block only where the record earns one; every other article one line with its reason ----------------------

def test_a_block_is_due_only_to_an_applicable_article_or_one_to_prove_with_a_sentence_for_it():
    assert issue_records.block_due(_rec([C.STATUS_PROVATO, C.STATUS_PROVATO]))                 # applicabile
    assert issue_records.block_due(_rec([C.STATUS_DA_PROVARE], supporting=["S1"]))             # da provare, a sentence
    assert not issue_records.block_due(_rec([C.STATUS_DA_PROVARE]))                            # da provare, none
    assert not issue_records.block_due(_rec([C.STATUS_NON_RISULTA]))                           # non applicabile
    system = {"S9": {"id": "S9", "source": "sistema"}}
    assert not issue_records.block_due(_rec([C.STATUS_DA_PROVARE], supporting=["S9"]), system)   # an assumption: none
    assert issue_records.line_reason(_rec([C.STATUS_DA_PROVARE])) == (issue_records.LINE_NOT_TREATED,
                                                                       C.LINE_REASON_NO_SUPPORT)
    assert issue_records.line_reason({"requisites": []}) == (issue_records.LINE_NOT_TREATED, C.LINE_REASON_NO_TEXT)
    kind, reason = issue_records.line_reason(_rec([C.STATUS_NON_RISULTA]))
    assert kind == issue_records.LINE_NOT_APPLICABLE and reason == "non risulta «elemento numero 1 della norma»"
    assert issue_records.line_reason(_rec([C.STATUS_PROVATO])) is None


def test_every_article_without_a_block_is_one_line_with_its_reason():
    text = ("**Art. 1062 c.c.**\n- co. 1, requisito «fondi divisi»: non risulta.\n\n"
            "**Art. 1061 c.c.**\n- testo: da provare.\n\n"
            "**Art. 1158 c.c.**\n- testo: da provare.\n\nIl possesso dura da vent'anni.\n\n"
            + C.fill(C.OTHER_NORMS_LINE, articles="art. 1159 c.c."))
    verdicts = {"1062 c.c.": issue_records.NOT_APPLICABLE, "1061 c.c.": "da provare", "1158 c.c.": "da provare"}
    lines = {"1062 c.c.": (issue_records.LINE_NOT_APPLICABLE, "non risulta «due fondi, ora divisi»"),
             "1061 c.c.": (issue_records.LINE_NOT_TREATED, C.LINE_REASON_NO_SUPPORT),
             "1159 c.c.": (issue_records.LINE_NOT_TREATED, C.LINE_REASON_NO_TEXT)}
    log = []
    out = flow.norms_final(text, log, verdicts, lines)
    assert "**Art. 1158 c.c.**" in out and "**Art. 1062" not in out and "**Art. 1061" not in out
    assert NA_PREFIX + "art. 1062 c.c. (non risulta «due fondi, ora divisi»)." in out      # a comma inside the reason
    assert NT_PREFIX + "art. 1061 c.c. (nessun fatto del caso a sostegno), art. 1159 c.c. (testo non letto)." in out
    assert flow.norms_final(out, [], verdicts, lines) == out                               # stable on a second pass
    assert {e.get("one_line") for e in log} >= {"1062 c.c.", "1061 c.c."}
    prose = text.replace("- testo: da provare.\n\n**Art. 1158", "- testo: da provare.\n\nLe opere sono visibili.\n\n"
                                                                 "**Art. 1158")
    assert "**Art. 1061 c.c.**" in flow.norms_final(prose, [], verdicts, lines)           # the check removes it first


def test_the_answer_gives_a_block_to_the_supported_article_and_a_line_to_the_other(monkeypatch):
    state = _graded_state(monkeypatch)
    state["questions"] = [q for q in state["questions"] if q["id"] != "opera_separazione_0"]    # 1061's, closed (v60j)
    writer = {flow.ANSWER_TITLES[0]: "Leonardo può far valere la servitù.",
              NORMS: "[1] L'art. 1062 c.c. fa sorgere la servitù per destinazione.\n\n"
                     "[2] L'art. 1061 c.c. vuole opere visibili e permanenti, qui presenti sul fondo."}
    text, _ = _answer(monkeypatch, state, writer, {"R1": {"sostiene": ["S1"]}})
    norms = dict(answer_ledger.sections_of(text))[NORMS]
    assert "**Art. 1062 c.c." in norms and "**Art. 1061" not in norms
    assert "art. 1061 c.c. (nessun fatto del caso a sostegno)" in norms
    ledger = state["ledger"]
    assert ledger["unaccounted"] == [] and ledger["unreasoned"] == []                   # no deletion without a reason
    removed = [e for e in ledger["entries"] if e["kind"] == "removed"]
    assert all(e["step"] == "record_check" and e["rule"] in record_check.REMOVE_RULES for e in removed)
    buckets = {label: r["answer"] for label, r in state["records"].items() if r.get("answer")}
    assert buckets["1062 c.c."]["bucket"] == flow.BUCKET_WRITTEN
    assert buckets["1061 c.c."]["bucket"] == flow.BUCKET_NOT_TREATED                   # one bucket per article


# --- 2. role-aware verdicts -------------------------------------------------------------------------------------------

def test_a_failing_article_the_counterparty_relies_on_is_the_clients_argument():
    rec = _rec([C.STATUS_NON_RISULTA], relied="Domitilla invoca la servitù per destinazione")
    assert issue_records.verdict(rec) == issue_records.CLIENT_ARGUMENT
    assert issue_records.block_due(rec) and issue_records.line_reason(rec) is None       # a block, never a line
    assert issue_records.verdict(_rec([C.STATUS_NON_RISULTA])) == issue_records.NOT_APPLICABLE
    assert issue_records.verdict(_rec([C.STATUS_PROVATO], relied="x")) == issue_records.APPLICABLE
    block = issue_records.writer_block({"1062 c.c.": {**rec, "verdict": issue_records.verdict(rec)}},
                                       [("1062", "c.c.")], [])
    assert "esito: argomento del cliente (la controparte vi si fonda — Domitilla invoca" in block


def test_the_counterpartys_articles_are_read_from_the_lines_of_the_other_side():
    state = {"analysis": {"parte_assistita": "Leonardo", "action_gate": {"949 c.c.": "x"}}, "topics": [
        {"anchor": "1062 c.c.", "hypotheses": [{"line": "controparte", "party": "Leonardo", "claim": "la servitù c'è",
                                                "anchored_by": "art. 1062 c.c.: Servitù per destinazione"}]},
        {"anchor": "1158 c.c.", "hypotheses": [{"line": "principale", "party": "Domitilla", "claim": "ha usucapito"}]},
        {"anchor": "1051 c.c.", "hypotheses": [{"line": "principale", "party": "Leonardo", "claim": "passaggio"},
                                               {"line": "principale", "party": "Domitilla", "claim": "passaggio"}]}]}
    got = flow.counterparty_articles(state)
    assert set(got) == {"1062 c.c.", "1158 c.c.", "949 c.c."}       # 1051: the client's own line rests on it too
    assert got["1062 c.c."] == "la servitù c'è"
    assert flow.counterparty_articles({"analysis": {}, "topics": state["topics"]}) == {}      # no client: none


def test_the_check_keeps_the_clients_argument_and_removes_its_contradiction():
    rec = {**_rec([C.STATUS_NON_RISULTA], relied="la controparte invoca la servitù"),
           "verdict": issue_records.CLIENT_ARGUMENT}
    records = {"1062 c.c.": rec}
    assert record_check._verdict_contradicted("L'art. 1062 c.c. non si applica al caso: i fondi non erano di un solo "
                                              "proprietario.", "1062 c.c.", records) == ""
    assert record_check._verdict_contradicted("La controparte invoca l'art. 1062 c.c., che a suo dire si applica.",
                                              "1062 c.c.", records) == ""
    assert record_check._verdict_contradicted("L'art. 1062 c.c. si applica al caso.", "1062 c.c.", records)


def test_the_answer_writes_the_counterpartys_failing_article_as_the_clients_argument(monkeypatch):
    state = _graded_state(monkeypatch)
    state["records"]["1062 c.c."]["requisites"][0]["distinctive"] = True
    state["topics"][0]["hypotheses"] = [{"line": "controparte", "party": "Leonardo",
                                         "claim": "Domitilla invoca la servitù per destinazione",
                                         "anchored_by": "art. 1062 c.c.: Servitù per destinazione del padre di famiglia"}]
    writer = {flow.ANSWER_TITLES[0]: "Leonardo resiste alla pretesa.",
              NORMS: "[1] La controparte invoca l'art. 1062 c.c., ma i due fondi non erano dello stesso proprietario: "
                     "l'elemento non risulta."}
    text, _ = _answer(monkeypatch, state, writer, {"R1": {"contraddice": ["S2"]}})
    assert state["records"]["1062 c.c."]["verdict"] == issue_records.CLIENT_ARGUMENT
    norms = dict(answer_ledger.sections_of(text))[NORMS]
    assert "**Art. 1062 c.c." in norms and "non erano dello stesso proprietario" in norms
    assert not any(line.startswith(NA_PREFIX) and "1062" in line for line in norms.split("\n"))
    answer = state["records"]["1062 c.c."]["answer"]
    assert answer["bucket"] == flow.BUCKET_WRITTEN and answer["reason"].startswith("argomento del cliente")
    assert state["ledger"]["unaccounted"] == [] and state["ledger"]["unreasoned"] == []


def test_the_writer_is_told_how_to_write_the_clients_argument():
    rules = writer_rules.system_rules("Leonardo")
    assert "«argomento del cliente»" in rules and "mai come una norma «non applicabile»" in rules
    assert "con il motivo" in rules and "«argomento del cliente», prima la pretesa della controparte" in \
        writer_rules.NORMS_INSTRUCTION


# --- 3. the c.p.a. only in administrative frames ----------------------------------------------------------------------

def test_the_cpa_is_kept_out_of_a_non_administrative_frame_unless_named():
    # v60q: the general code-area gate (area_gate) -- outside the frame's area, «altri profili» unless named
    rows = [{"_article": ("29", "c.p.a.")}, {"_article": ("30", "Codice del processo amministrativo 2025")},
            {"_article": ("1158", "c.c.")}]
    civil = {"frame": {"areas": ["diritti_reali"]}}
    log = {}
    kept, outside = flow.area_gate(rows, civil, "Leonardo usa il passaggio da anni.", {}, log)
    assert [r["_article"][0] for r in kept] == ["1158"] and log["altri_profili"] == ["29 c.p.a.", "30 c.p.a."]
    assert [r["_article"][0] for r in outside] == ["29", "30"]
    assert len(flow.area_gate(rows, {"frame": {"areas": ["amministrativo"]}}, "")[0]) == 2       # 1158 c.c.: outside
    assert len(flow.area_gate(rows, {"area": "amministrativo"}, "")[0]) == 2
    log = {}
    named = flow.area_gate(rows, civil, "Si applica l'art. 29 c.p.a.?", {}, log)[0]
    assert [r["_article"][0] for r in named] == ["29", "1158"]
    assert log["admitted"] == [{"article": "29 c.p.a.", "why": "nominato dall'utente"}]
    log = {}
    assert len(flow.area_gate(rows, {}, "", {}, log)[0]) == 3 and log["off"] == "areas unknown"   # no gate


# --- 4. the user's own message is never a trigger ----------------------------------------------------------------------

CALCIATORE = ("Sei l'avvocato di Tizio, Qual è l'atto giudiziario più opportuno per le problematiche da affrontare? "
              "Durante una partita Sempronio ha aggredito Tizio, fratturandogli il perone.")


def test_the_users_message_is_no_case_sentence():
    assert fact_signals.case_sentences(CALCIATORE) == ["Durante una partita Sempronio ha aggredito Tizio, "
                                                       "fratturandogli il perone."]
    for said in ("Assisto il proprietario.", "Indica la strategia.", "Si chiede di valutare l'usucapione.",
                 "Valuta anche i termini.", "Posso agire per usucapione?"):
        assert fact_signals.user_message(said), said
    for fact in ("Tizio usa il sentiero da anni.", "Come ogni mattina, Tizio passava sul fondo.",
                 "Il legale del proprietario ha inviato una diffida."):
        assert not fact_signals.user_message(fact), fact
    upload = "RICHIESTA DELL'UTENTE: Valuta l'usucapione del sentiero.\n\nDOCUMENTO CARICATO:\nIl muro crolla."
    assert fact_signals.case_only(upload).strip() == "Il muro crolla."


def test_no_term_tool_or_fact_trigger_reads_the_users_message():
    asked = "Leonardo passa sul fondo. Può far valere l'usucapione, visto che lo usa da vent'anni?"
    assert fact_signals.trigger_sentence("usucapione_ordinaria", asked) == ""                       # term
    assert render_computed.tool_trigger({"id": "trascrizione_domanda"},
                                        "Il fondo è di Domitilla. Posso rivendicare la proprietà del terreno?") == ""  # tool
    sents = issue_records.sentences(CALCIATORE, ["Difendo Tizio.", "Il fatto è del 3 marzo 2026."])
    assert [s["text"] for s in sents] == ["Durante una partita Sempronio ha aggredito Tizio, fratturandogli il perone.",
                                          "Il fatto è del 3 marzo 2026."]                             # fact
    facts = {"parties": [{"name": "Leonardo", "role": "cliente"}], "events": [], "requests": []}
    request = "Voglio una somma per ogni giorno di ritardo."
    assert any(t["id"] == "misure_coercitive" for t in fact_signals.procedural_tools(facts, ["civile"], request))
    monkey_regimes = fact_signals.regimes_enabled
    try:
        fact_signals.regimes_enabled = lambda: True
        got = fact_signals.signals_from(facts, areas=["civile"], request=request, case="Leonardo è il proprietario.")
    finally:
        fact_signals.regimes_enabled = monkey_regimes
    assert not any(t["id"] == "misure_coercitive" for t in got.get("tools") or [])


# --- 5. the usucapione's and the trascrizione's own triggering facts --------------------------------------------------

def test_the_usucapione_needs_the_possession_ad_usucapionem_stated():
    for said in ("Tizio possiede il terreno dal 1990 come proprietario.", "Caio sostiene di aver usucapito la striscia.",
                 "La famiglia transita sulla strada da più di trent'anni.", "Uso il sentiero da anni."):
        assert fact_signals.trigger_sentence("usucapione_ordinaria", said), said
    for said in ("La causa riguarda l'usucapione della servitù.", "Leonardo passa sul fondo di Domitilla.",
                 "Tizio usa il passaggio da anni con il permesso del vicino.", "Abita in affitto da dieci anni."):
        assert fact_signals.trigger_sentence("usucapione_ordinaria", said) == "", said


def test_the_trascrizione_needs_a_domanda_art_2653_lists(monkeypatch):
    tool = {"id": "trascrizione_domanda", "refs": [["2653", "c.c."]], "text": "Trascrizione della domanda giudiziale.",
            "matched": ["trascrizione della domanda giudiziale (azione confessoria) ai sensi dell'art. 2653, n. 1, c.c."]}
    for said in ("Leonardo vuole far accertare la servitù di passaggio sul fondo di Domitilla.",
                 "Domitilla rivendica la proprietà del terreno.", "Leonardo intende proporre azione confessoria.",
                 "Leonardo ha notificato un atto per interrompere l'usucapione del cortile."):
        assert render_computed.tool_trigger(tool, said), said
    for said in ("Il caso riguarda una servitù di passaggio sul fondo.", "Domitilla rivendica la proprietà dell'auto."):
        assert render_computed.tool_trigger(tool, said) == "", said
    analysis = {"store_offences_only": True, flow.CASE_KEY: "Leonardo vuole far accertare la servitù sul fondo.",
                "fact_signals": {"tools": [tool]}}
    blocks = dict(render_computed.blocks(analysis))
    assert blocks[render_computed.TRASCRIZIONE_TITLE] == "Va trascritta la domanda giudiziale (azione confessoria)."
    analysis[flow.CASE_KEY] = "La lite riguarda la servitù di passaggio sul fondo."
    assert render_computed.TRASCRIZIONE_TITLE not in dict(render_computed.blocks(analysis))


# --- 6. the «non si applica» detector ---------------------------------------------------------------------------------

def test_the_detector_reads_the_pos_phrasings():
    for said in ("La norma non è rilevante.", "Il termine di tre mesi non è scaduto.",
                 "L'art. 336 c.p.p. non costituisce una norma sostanziale.", "La norma non si applica in questo caso.",
                 "Il termine è ancora in corso."):
        assert flow.declares_inapplicable(said), said
    for said in ("Il requisito è provato; la norma si applica.", "Il termine non è scaduto, ma va verificato.",
                 # the replay's art. 1783 c.c.: one comma's sub-point, never the article (no split after «n.»)
                 "L'albergatore è responsabile. - Il secondo profilo (comma 2, n. 2) non è rilevante, in quanto non "
                 "risulta la custodia."):
        assert not flow.declares_inapplicable(said), said


# --- 7. articled prepositions in party references ----------------------------------------------------------------------

def test_party_references_take_the_articled_preposition():
    names = ["Inquilino", "Acquirente", "Società"]
    for raw, want in (("la posizione dell'Inquilino", "la posizione dell'inquilino"),
                      ("la posizione di l'inquilino", "la posizione dell'inquilino"),
                      ("scrive a l’acquirente", "scrive all’acquirente"),
                      ("la tutela dell’Acquirente", "la tutela dell’acquirente"),
                      ("consegna a Acquirente", "consegna all'acquirente"),
                      ("il contratto di la società", "il contratto della società"),
                      ("alla Società Alfa", "alla Società Alfa")):
        assert party_names.lower_roles(raw, names) == want, raw
    assert party_names.contract("il caso di l’inquilino e a il venditore") == "il caso dell’inquilino e al venditore"


# --- 8. role phrases set the client -------------------------------------------------------------------------------------

def test_role_phrases_name_the_client_by_its_role():
    for said, role in (("Sono l'avvocato dell'inquilino.", "inquilino"), ("l'avvocato dell'inquilino", "inquilino"),
                       ("difendo il proprietario", "proprietario"), ("Assisto la società.", "società"),
                       ("In qualità di difensore del conduttore, chiedo un parere.", "conduttore")):
        assert flow.stated(said)["client_role"] == role, said
    assert flow.stated("Il legale del proprietario ha inviato una diffida.")["client_role"] == ""   # the other side's
    parties = [{"name": "Mario Rossi", "role": "conduttore"}, {"name": "Immobiliare Beta S.r.l.", "role": "locatore"}]
    assert flow.role_party("inquilino", parties) == "Mario Rossi"
    assert flow.role_party("società", parties) == "Immobiliare Beta S.r.l."
    assert flow.role_party("proprietario", [{"name": "A", "role": "proprietario"}, {"name": "B", "role": "proprietario"}]) \
        == ""                                                                                         # two: ask


def test_a_role_phrase_sets_the_client_of_the_case(monkeypatch):
    lease = {**base.FACTS, "parties": [{"name": "Mario Rossi", "role": "conduttore"},
                                       {"name": "Immobiliare Beta S.r.l.", "role": "locatore, proprietario"}]}
    state = base._state(monkeypatch, facts=lease)
    flow.apply_stated(state, flow.stated("Sono l'avvocato dell'inquilino."))
    assert state["answered"]["parte_assistita"] == "Mario Rossi"
    assert flow._restored_analysis(state["analysis"])["parte_assistita"] == "Mario Rossi"


# --- 9. the labour dispute's procedural facts, a closed list ----------------------------------------------------------

def test_the_labour_lines_come_from_the_closed_list_for_a_labour_topic():
    analysis = {"store_offences_only": True, "topic_refs": [["2119", "c.c."]],
                flow.CASE_KEY: "Mario è stato licenziato il 3 marzo con una lettera. Sei l'avvocato di Mario?"}
    got = render_computed.labour_of(analysis)
    assert got["matter"] == "2119 c.c." and len(got["lines"]) == 3 and got["untriggered"] == []
    assert "entro 60 giorni" in got["lines"][0] and "180 giorni" in got["lines"][0] and "l. 604/1966" in got["lines"][0]
    assert "Rito del lavoro" in got["lines"][1] and "facoltativo" in got["lines"][2]
    title = render_computed.labour_list()["title"]
    assert dict(render_computed.blocks(analysis))[title].startswith("- Impugnazione del licenziamento")
    asked = {**analysis, flow.CASE_KEY: "Mario lavora da dieci anni. Posso impugnare il licenziamento?"}
    assert render_computed.labour_of(asked)["untriggered"] == ["impugnazione_licenziamento"]       # the question: no
    assert render_computed.labour_of({**analysis, "topic_refs": [["1158", "c.c."]]})["lines"] == []
    assert title not in dict(render_computed.blocks({**analysis, "store_offences_only": False}))


# --- 10. the rubric on the live transcripts ------------------------------------------------------------------------------

def _rubric():
    import importlib.util
    spec = importlib.util.spec_from_file_location("rubric_score", HERE.parent / "scripts" / "rubric_score.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_rubric_reads_the_live_transcripts_and_leaves_the_po_a_column(tmp_path):
    rubric = _rubric()
    live = rubric.from_live(str(HERE.parent / "diagnose_reports" / "live"))
    assert set(live) == {"test_calciatore", "test_stalking"}
    assert live["test_stalking"][1].startswith("1. Risposta e strategia")              # round 1 left out
    assert live["test_calciatore"][1].startswith("⚖️ BOZZA")
    (tmp_path / "a.txt").write_text("Ho letto il caso.\n\n1. Risposta e strategia\n\nTesto.", encoding="utf-8")
    (tmp_path / "a.case.txt").write_text("Il caso.", encoding="utf-8")
    assert rubric.from_live(str(tmp_path)) == {"a": ("Il caso.", "1. Risposta e strategia\n\nTesto.")}
    scored = {"a": rubric.score("Il caso.", "Testo.", caller=lambda system, human: {k: {"score": 0.5, "why": "w"}
                                                                                   for k, _, _ in rubric.ROWS})}
    out = rubric.po_tables(scored, ["a"])
    assert "| row | ours | PO |" in out and "| strumento/atto | 0.5 |  |" in out and "model's guess" in out
