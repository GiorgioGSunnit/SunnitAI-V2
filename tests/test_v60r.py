"""v60r: single-shot render and record fixes -- the record's rows never in the body, one status sentence per article,
the rows in the article citations (1); the single-shot «Altri profili»: inside the unit it is named by, past the
code-area gate, by score and cut (2); every article «non applicabile» only from its anchor comma (its distinctive
element's, else its first element's), any other unmet comma «non pertinente» (3); consecutive «(da verificare…)» markers merged (4); the party's label from the frame (5); an element no fact touches
«da provare» in the payload, never narrated as missing (6). Synthetic records; the model and the graph are stubbed."""

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

NORMS = flow.ANSWER_TITLES[1]
PROVATO, DA_PROVARE, NON_RISULTA = C.STATUS_PROVATO, C.STATUS_DA_PROVARE, C.STATUS_NON_RISULTA


def _req(comma, text, status, **kw):
    return {"kind": "requisito", "comma": comma, "text": text, "status": status, "facts_status": DA_PROVARE,
            "supporting": [], "contradicting": [], "question": "", **kw}


# --- 1. the record's rows never in the body --------------------------------------------------------------------------------

DRAFT = ("**1. Risposta e strategia**\n\nLeonardo agisce.\n\n**2. Norme applicate ai fatti**\n\n"
         "**Art. 1062 c.c. — Destinazione del padre di famiglia**\n"
         "- testo, requisito «due fondi posseduti dallo stesso proprietario»: provato.\n"
         "- testo, requisito «cose lasciate nello stato della servitù»: da provare.\n"
         "- Sanzione: art. 1070 c.c. — Abbandono del fondo servente.\n\n"
         "Mevio possedeva entrambi i fondi.\n\n"
         "**Art. 1061 c.c.**\n- co. 1: da provare.\n\n"
         "Altre norme del tema non trattate: art. 1052 c.c. (nessun fatto del caso a sostegno).\n"
         "Norme considerate e non applicabili: art. 1051 c.c. (non risulta «fondo intercluso»).\n\n"
         "**5. Conclusioni**\n\nEsito favorevole.")


def _ledger_state():
    return {"sentences": [{"id": "S1", "text": "All'inizio del 1900 i fondi appartenevano a Mevio."},
                          {"id": "S2", "text": "Leonardo ha un altro accesso alla via."}],
            "records": {
                "1062 c.c.": {"verdict": DA_PROVARE, "requisites": [
                    _req(None, "due fondi posseduti dallo stesso proprietario", PROVATO, supporting=["S1"]),
                    _req(None, "cose lasciate nello stato della servitù", DA_PROVARE)]},
                "1061 c.c.": {"verdict": DA_PROVARE, "requisites": [_req(1, "servitù non apparenti", DA_PROVARE),
                                                                   _req(2, "opere visibili", NON_RISULTA,
                                                                        contradicting=["S2"])]},
                "1051 c.c.": {"verdict": issue_records.NOT_APPLICABLE,
                              "requisites": [_req(1, "fondo intercluso", NON_RISULTA, contradicting=["S2"])]}}}


def test_the_body_keeps_one_status_sentence_per_article_and_no_row():
    log = {}
    text, rows = flow.ledger_out(DRAFT, _ledger_state(), log)
    norms = dict(answer_ledger.sections_of(text))[NORMS]
    assert norms.strip() == ("**Art. 1062 c.c. — Destinazione del padre di famiglia**\n"
                             "Esito sui fatti del caso: da provare (requisiti provati: 1 su 2).\n\n"
                             "Mevio possedeva entrambi i fondi.")
    assert "Leonardo agisce." in text and "Esito favorevole." in text                    # the other sections as they were
    assert log["bare"] == ["1061 c.c."] and log["status_lines"] == {"1062 c.c.": 3}
    assert list(log["articles"].items()) == [("1062 c.c.", "body"), ("1070 c.c.", "sanzione"),
                                             ("1061 c.c.", "non trattata"), ("1052 c.c.", "non trattata"),
                                             ("1051 c.c.", "non applicabile")]


def test_every_row_goes_to_the_citations_with_its_status_and_sentence():
    _, rows = flow.ledger_out(DRAFT, _ledger_state())
    heads = {r["article"]: r for r in rows if r["requisite"] is None and r["comma"] is None}
    assert heads["1062 c.c."] == {"article": "1062 c.c.", "comma": None, "requisite": None, "status": DA_PROVARE,
                                  "reason": "", "in_body": True}
    assert heads["1061 c.c."]["status"] == "non trattata" and heads["1061 c.c."]["reason"] == C.LINE_REASON_NOT_WRITTEN
    assert heads["1052 c.c."]["reason"] == "nessun fatto del caso a sostegno"
    assert heads["1051 c.c."]["status"] == "non applicabile" and heads["1051 c.c."]["reason"] == \
        "non risulta «fondo intercluso»"
    assert heads["1070 c.c."]["status"] == "sanzione" and heads["1070 c.c."]["reason"] == "sanziona l'art. 1062 c.c."
    elements = [r for r in rows if r["requisite"]]
    first = next(r for r in elements if r["requisite"].startswith("due fondi"))
    assert first["status"] == PROVATO and first["supporting_sentence"].startswith("All'inizio del 1900")
    other = next(r for r in elements if r["requisite"] == "opere visibili")
    assert other["status"] == issue_records.NOT_RELEVANT and other["contradicting_sentence"].startswith("Leonardo ha")
    assert next(r for r in elements if r["requisite"] == "fondo intercluso")["status"] == NON_RISULTA


def test_the_citations_list_every_article_the_rows_hold(monkeypatch):
    monkeypatch.setattr(flow, "_log", lambda *a, **k: None)
    state = {**_ledger_state(), "rows": [], "tool_rows": []}
    _, state["ledger_rows"] = flow.ledger_out(DRAFT, state)
    out = flow.article_citations(state, "Leonardo agisce.")
    assert [a["id"] for a in out] == ["1062 c.c.", "1070 c.c.", "1061 c.c.", "1052 c.c.", "1051 c.c."]   # §2's order
    gone = next(a for a in out if a["id"] == "1051 c.c.")
    assert gone["status"] == "non applicabile" and gone["in_body"] is False and gone["commi"] == []
    assert gone["rows"] == [{"article": "1051 c.c.", "comma": 1, "requisite": "fondo intercluso", "status": NON_RISULTA,
                             "supporting_sentence": None,
                             "contradicting_sentence": "Leonardo ha un altro accesso alla via."}]


# --- 2. the single-shot «Altri profili» ---------------------------------------------------------------------------------------

def _topic(title, articles, unit="", source="sezione", best=2, graded=1):
    return {"title": title, "articles": articles, "unit": unit, "source": source, "best_grade": best, "graded": graded}


def test_the_profiles_are_inside_their_unit_past_the_area_gate_by_score_and_cut(monkeypatch):
    logged = []
    monkeypatch.setattr(flow, "_log", lambda event, data, *a: logged.append((event, data)))
    topics = [_topic("Coperto", ["1 c.c."]),
              _topic("Passaggio coattivo", ["1051 c.c.", "1158 c.c."], "SEZIONE I – PASSAGGIO", best=2, graded=2),
              _topic("Fatti illeciti", ["2043 c.c."], "CAPO II – FATTI ILLECITI", source="capo", best=3),
              _topic("Servitù apparenti", ["1062 c.c.", "1061 c.c."], "SEZIONE III – APPARENTI", best=3),
              _topic("Senza unità", ["640 c.p."], source="rubrica", best=1)]
    topics += [_topic(f"T{n}", [f"{n} c.p."], source="rubrica", best=2, graded=n) for n in (600, 601, 602)]
    units = {"1051 c.c.": ["SEZIONE I – PASSAGGIO"], "1158 c.c.": ["CAPO II – USUCAPIONE"],
             "2043 c.c.": ["CAPO II – FATTI ILLECITI"], "1062 c.c.": ["SEZIONE II – ALTRA"],
             "1061 c.c.": ["SEZIONE III – APPARENTI"]}
    state = {"topics": topics, "units": units,
             "area_gate": {"families": ["penale"], "admitted": [{"article": "1051 c.c.", "why": "regola «x»"},
                                                                 {"article": "1061 c.c.", "why": "regola «y»"}]}}
    lines = flow.single_profiles(state, [1], {})
    assert lines == ["- Servitù apparenti (artt. 1061 c.c.)", "- T602 (artt. 602 c.p.)", "- T601 (artt. 601 c.p.)",
                     "- T600 (artt. 600 c.p.)", "- Passaggio coattivo (artt. 1051 c.c.)"]
    log = dict(logged)["single_shot_profiles"]
    assert log["outside_unit"] == {"Passaggio coattivo": ["1158 c.c."], "Servitù apparenti": ["1062 c.c."]}
    assert log["outside_area"] == {"Fatti illeciti": ["2043 c.c."]} and log["dropped"] == ["Fatti illeciti"]
    assert log["reanchored"] == {"Servitù apparenti": "1061 c.c."} and log["cut"] == ["Senza unità"]
    assert flow.OTHER_PROFILES_MAX == 5


# --- 3. «non applicabile» only from the anchor comma, every article --------------------------------------------------------

def test_the_anchor_comma_is_co_1_unless_the_topic_names_one():
    # v60s (2): co. 1 -- never the first element's, never the distinctive element's -- unless the topic names a comma
    assert issue_records.anchor_comma({"requisites": [_req(2, "a", PROVATO), _req(1, "b", PROVATO)]}) == 1
    assert issue_records.anchor_comma({"requisites": [_req(1, "a", PROVATO), _req(2, "b", PROVATO, distinctive=True)]}) == 1
    assert issue_records.anchor_comma({"requisites": [], "anchor_comma": 2}) == 2
    assert issue_records.anchor_comma({"requisites": []}) == 1
    topics = [{"title": "Legittima difesa domiciliare (art. 52, co. 2, c.p.)", "anchor": "52 c.p.",
               "articles": ["52 c.p.", "55 c.p."], "planned": 2}]
    records = issue_records.build(topics, [], {}, [], [], {})
    assert records["52 c.p."]["anchor_comma"] == 2 and "anchor_comma" not in records["55 c.p."]


def test_comma_1_contradicted_comma_2_proven_turns_on_which_is_the_anchor_comma():
    first = _req(1, "fondo intercluso", NON_RISULTA)
    second = _req(2, "case e cortili esenti", PROVATO)
    rec = {"requisites": [second, first]}                               # any article; the order of its elements: none
    assert issue_records.verdict(rec) == issue_records.NOT_APPLICABLE   # co. 1, the anchor comma, contradicted
    assert issue_records.element_status(rec, first) == NON_RISULTA
    assert issue_records.verdict_reason(rec) == "requisito «fondo intercluso» non risulta"
    named = {**rec, "anchor_comma": 2}                                  # the topic names co. 2
    assert issue_records.verdict(named) == issue_records.APPLICABLE     # co. 1 «non pertinente», co. 2 proven
    assert issue_records.element_status(named, first) == issue_records.NOT_RELEVANT
    single = {"requisites": [_req(None, "unico comma", NON_RISULTA)]}   # a text with no numbered commi: its co. 1
    assert issue_records.verdict(single) == issue_records.NOT_APPLICABLE


def test_a_non_anchor_comma_never_excludes_and_never_proves():
    anchor = _req(1, "uscita sulla via pubblica", DA_PROVARE)
    rec = {"requisites": [anchor, _req(2, "case esenti", NON_RISULTA), _req(3, "indennità", NON_RISULTA)]}
    assert issue_records.verdict(rec) == DA_PROVARE                     # the anchor comma open: «da provare»
    assert not issue_records.block_due(rec) and issue_records.line_reason(rec) == (
        issue_records.LINE_NOT_TREATED, C.LINE_REASON_NO_SUPPORT)        # never «non applicabile»
    anchor["status"] = PROVATO
    assert issue_records.verdict(rec) == issue_records.APPLICABLE       # proven on its anchor comma alone
    rec["requisites"].append(_req(4, "altro", DA_PROVARE))
    assert issue_records.verdict(rec) == DA_PROVARE                     # a live comma still open: the existing rule
    assert [issue_records.element_status(rec, r) for r in rec["requisites"]] == [
        PROVATO, issue_records.NOT_RELEVANT, issue_records.NOT_RELEVANT, DA_PROVARE]
    assert issue_records.verdict({**rec, "relied_by": "Domitilla", "requisites": [_req(1, "a", NON_RISULTA)]}) == \
        issue_records.CLIENT_ARGUMENT


# --- 4. consecutive markers ---------------------------------------------------------------------------------------------------

def test_consecutive_markers_are_one():
    text = ("Il passaggio è dovuto (da verificare: data del rogito) (da verificare: data del rogito)"
            " (da verificare: opere visibili). Poi (da verificare: altro).")
    log = []
    assert flow.merged_markers(text, log) == ("Il passaggio è dovuto (da verificare: data del rogito; opere visibili). "
                                              "Poi (da verificare: altro).")
    assert len(log) == 1 and flow.merged_markers("x (da verificare) (da verificare).") == "x (da verificare)."


# --- 5. the party's label from the frame ---------------------------------------------------------------------------------------

def test_the_partys_label_comes_from_the_frame():
    penal = {"frame": {"areas": ["penale"]}, "fact_signals": {"facts": {}}}
    assert flow.frame_party_label(penal, "Tizio ha colpito Caio.") == "indagato"
    assert flow.frame_party_label(penal, "Caio ha sporto querela contro Tizio.") == "querelato"
    assert flow.frame_party_label({"frame": {"areas": ["diritti_reali"]}}, "") == "convenuto"
    assert flow.frame_party_label({"frame": {"areas": ["diritti_reali"]}}, "", "convenuta, proprietaria") == "convenuta"
    assert flow.frame_party_label({"frame": {"areas": ["amministrativo"]}}, "") == ""
    assert flow.frame_party_label({}, "") == ""


# --- 6. an element no fact touches -----------------------------------------------------------------------------------------------

def test_an_element_no_fact_touches_is_da_provare_and_never_narrated_as_missing():
    quiet = _req(1, "fondo intercluso", DA_PROVARE, missing="se il fondo ha un'uscita")
    asked = _req(1, "opere visibili", DA_PROVARE, question="opera_separazione", missing="le opere")
    assert issue_records.no_signal(quiet) and not issue_records.no_signal(asked)
    assert not issue_records.no_signal(_req(1, "x", DA_PROVARE, supporting=["S1"]))
    block = issue_records.writer_block({"1051 c.c.": {"verdict": DA_PROVARE, "requisites": [quiet, asked]}},
                                       [("1051", "c.c.")], [], [{"id": "opera_separazione", "text": "Le opere?"}])
    assert f"requisito «fondo intercluso»: da provare ({issue_records.NO_SIGNAL_NOTE})" in block
    assert "MANCA: se il fondo" not in block and "MANCA: le opere" in block and "domanda aperta: Le opere?" in block
    from src.rag import writer_rules
    assert writer_rules.NO_SIGNAL_RULE in writer_rules.system_rules("Leonardo")
