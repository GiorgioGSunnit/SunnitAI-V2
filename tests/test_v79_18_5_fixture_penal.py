"""v79.18.5 -- the synthetic PENAL fixture of 79.15.0 (tests/test_v79_15_0_fixture_penal.py: Dario Melis, indagato for
appropriazione indebita, 646 c.p. and the general-part 62-bis c.p.; invented names and facts) run end to end with the
strategy step's model faked too (strategy_step._default_call). The faked strategy writes its free text as the Q&A's
prose must: the facts by their S-ids, one objection with its accusa_forte, one objection with no S-id at all (dropped), a
perche_prima naming the client with no id (dropped), a passo that asks a legal conclusion (the code's label) and one
naming the means with its element, a contro-difesa answer citing the denial. Expected: the page carries the facts'
quotes and words only -- no id, no label --, §7 prints the accusa_forte's facts, the counters say what was dropped."""

import json
import os
import re
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _fixture_79_15_0 as H  # noqa: E402
import test_v79_15_0_fixture_penal as PENAL  # noqa: E402
from src.rag import strategy_step as S  # noqa: E402

OBJ_GROUNDED = "L'accusa sosterrà che il profitto risulta dagli incassi trattenuti"
OBJ_UNGROUNDED = "L'accusa sosterrà che la cassa era affidata a Dario Melis."
PERCHE_NO_ID = "Dario Melis nega e la Procura non ha altro"
PASSO_OK = "Acquisire gli estratti conto della cassa e sentire i testimoni del negozio"
PASSO_CONCLUSION = "Dimostrare che Dario Melis non è responsabile dell'appropriazione"
CONTRO = "La negazione di Dario Melis contraddice l'assunto"


def strategy_call(messages, kwargs):
    human = messages[1].content           # the re-ask (the passo asking a conclusion) gets the same reply: its label stands
    els = re.findall(r"^  - \[(A646cp:E\d+)\] (.*?) — (\w+) — (.*)$", human, re.MULTILINE)
    contested = [eid for eid, _t, esito, _c in els if esito == "contestato"]
    provato = next(eid for eid, _t, esito, _c in els if esito == "provato")
    sids = dict(re.findall(r"^(S\d+) \[[^\]]*\] (.*)$", human, re.MULTILINE))
    procura = next(i for i, t in sids.items() if t.startswith("Secondo la Procura"))
    denies = next(i for i, t in sids.items() if t.startswith("Dario Melis nega"))
    cash = next(i for i, t in sids.items() if t.startswith("Dario Melis lavorava"))
    alleg = re.search(r"^ALLEGAZIONI DELLA CONTROPARTE[^:]*: (.*)$", human, re.MULTILINE).group(1).split(", ")
    strategy_call.seen.append({"contested": contested, "provato": provato, "procura": procura, "denies": denies, "alleg": alleg})
    dolo, other = contested[0], contested[1]
    reply = {"linee": [
        {"articolo": "A646cp", "eid": dolo, "tipo": "contestare", "rapporto": "principale",
         "perche_prima": f"Il fine di profitto lo afferma solo la Procura [{procura}] e Dario Melis lo nega [{denies}].",
         "cita_fatti": [denies], "cita_pronunce": [],
         "obiezione": {"testo": f"{OBJ_GROUNDED} [{procura}].",
                       "accusa_forte": {"testo": f"Gli incassi trattenuti [{procura}].", "cita_fatti": [procura]},
                       "risposta_cita_fatti": [denies], "risposta_cita_elementi": [dolo], "cita_pronunce": []}},
        {"articolo": "A646cp", "eid": other, "tipo": "contestare", "rapporto": "subordinata",
         "perche_prima": PERCHE_NO_ID, "cita_fatti": [], "cita_pronunce": [],
         "obiezione": {"testo": OBJ_UNGROUNDED, "risposta_cita_fatti": [denies], "risposta_cita_elementi": [other],
                       "cita_pronunce": []}}],
        "passi": [{"eid": dolo, "cosa": PASSO_OK, "cita_fatti": [cash]},
                  {"eid": other, "cosa": PASSO_CONCLUSION, "cita_fatti": []}],
        "passi_immediati": [],
        "contro_difesa": [{"allegazione": alleg[0], "risposta": f"{CONTRO} [{denies}].", "cita_linea": dolo, "cita_fatti": [denies]}],
        "verifiche_documentali": []}
    return json.dumps(reply, ensure_ascii=False)


strategy_call.seen = []

case = types.SimpleNamespace(**{k: getattr(PENAL, k) for k in dir(PENAL) if not k.startswith("_") and not k.startswith("test_")})
case.CASE_ID = "v79185_penal"
case.strategy_call = strategy_call


def test_the_strategy_text_is_verified_as_the_qa_prose_on_the_penal_fixture(monkeypatch, tmp_path):
    strategy_call.seen.clear()
    run = H.run(monkeypatch, tmp_path, case)
    state, answer, by = run["state"], run["answer"], run["by"]
    c = state["block_counters"]
    target = os.environ.get("FIXTURE_79_18_5_OUT")
    if target:
        H.write_report(target, "v79.18.5 synthetic PENAL fixture with the strategy step (tests/test_v79_18_5_fixture_penal.py)", run, "w")
    assert [x for x in run["failed"] if x.split(":")[0] in ("analysis_step", "strategy_step", "strategy_input",
                                                             "lawyer_method_compose", "lawyer_method_plan")] == []
    assert len(strategy_call.seen) == 2 and len(strategy_call.seen[0]["contested"]) == 2
    step = by["strategy_step"][0][0]
    assert step["status"] == "ok" and step["set"] == ["646 c.p."] and step["side"] == "difesa"
    assert c["strategy_lines"] == 2 and c["strategy_calls"] == 2 and c["strategy_reasked"] == 1   # the passo re-asked once
    # item 1: the perche naming the client with no id is dropped; the cited ones are rendered
    assert c["strategy_sentences_dropped_no_fact"] >= 1 and c["strategy_ids_rendered"] >= 3
    # item 5: one objection grounded by its accusa_forte, the other without any S-id dropped
    assert c["strategy_objection_strongest"] == 1 and c["strategy_objection_dropped"] == 1
    # item 4: the passo asking a legal conclusion is the code's label, the other stands
    assert c["strategy_passi_not_concrete"] == 1 and c["strategy_passi_added"] >= 1
    # item 3: no id, no label on the page
    sol = H.section(answer, "soluzioni")
    assert not S._RESIDUE_RE.search(sol) and c["residues_dropped"] == 0
    assert "| Prima linea | art. 646 c.p. |" in sol and "| contestare l'elemento | in via principale |" in sol
    assert "| Seconda linea |" in sol and "| in subordine |" in sol
    assert " fondare |" not in sol and "| contestare |" not in sol and "Linea 1" not in sol and "Linea 2" not in sol
    assert f"- Allegazione «{PENAL.SOCIETA}»: {CONTRO} «Dario Melis nega di aver trattenuto il denaro»" in sol
    assert c["strategy_contro_difesa_added"] == 0
    assert "Allegazione S" not in sol and "accusa S" not in sol
    # §7: the grounded objection with its accusa_forte and the denial as the answer; the ungrounded one absent
    obiezioni = H.section(answer, "obiezioni")
    assert f"**Obiezione 1.** {OBJ_GROUNDED} «{PENAL.PROCURA}». La controparte farà leva su: Gli incassi trattenuti «{PENAL.PROCURA}»." in obiezioni
    assert "**Risposta.** «Dario Melis nega di aver trattenuto il denaro»." in obiezioni
    assert OBJ_UNGROUNDED not in obiezioni and "Obiezione 2." not in obiezioni
    # §10: the passo with its means, the code's label for the conclusion-asking one
    verificare = H.section(answer, "verificare")
    assert f"— {PASSO_OK} («Dario Melis lavorava" in verificare and PASSO_CONCLUSION not in verificare
    assert "da provare, nessun passo indicato dal modello" in verificare
    # the whole page: no raw id anywhere the strategy wrote
    for key in ("sintesi", "soluzioni", "obiezioni", "verificare", "conclusioni"):
        assert not re.search(r"\bA646cp[:.]|\bS\d{1,2}\b|\bG\d{1,2}\b", H.section(answer, key)), key
    # the replay row
    import importlib.util
    spec = importlib.util.spec_from_file_location("replay_cases_185f", HERE.parent / "scripts" / "replay_cases.py")
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    row = R.measure("c", state, answer, 1.0)
    assert row["strategy_counters"]["strategy_objection_strongest"] == 1 and row["strategy_counters"]["residues_dropped"] == 0
