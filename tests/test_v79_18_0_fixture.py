"""v79.18.0 (B) -- the synthetic CIVIL fixture of 79.16.0 (tests/test_v79_16_0_fixture_civil.py: Anna Corsi, the caparra
confirmatoria, three numbered quesiti; invented names and facts, statutory texts) run end to end with the strategy step's
model faked too (strategy_step._default_call): the objections call is retired, §1's lines, §6, §7, §9's risk word and
§10's passi come from the verified strategy. The faked strategy puts the «contestare» line first and the «fondare» line
(on the provato element, citing its fact) second: the code reorders them by forza (strategy_rank_reordered), the
fondare line's objection has no answer in the facts (printed as such), one contro-difesa entry answers the
counterparty's allegation, the passi name the principale's open elements."""

import json
import os
import re
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _fixture_79_15_0 as H  # noqa: E402
import test_v79_16_0_fixture_civil as CIVIL  # noqa: E402
from src.rag import strategy_step as S  # noqa: E402

OBJ_FONDARE = "La caparra è stata restituita ad Anna Corsi."
OBJ_CONTESTARE = "Anna Corsi non si è presentata dal notaio."
PASSO = "Produrre la diffida ad adempiere inviata a Ettore Bassi"
IMMEDIATO = "Conservare la ricevuta del versamento della caparra"
CONTRO = "La mancata presentazione dal notaio non risulta dai fatti"
VERIFICA = "Verificare nel preliminare il termine fissato per il rogito"


def strategy_call(messages, kwargs):
    """The strategy model faked: the ids read back from the prompt (the 1385 c.c. elements with their esito, the S-ids,
    the counterparty's allegations)."""
    human = messages[-1].content
    assert len(messages) == 2, "no re-ask expected"
    els = re.findall(r"^  - \[(A1385cc:E\d+)\] (.*?) — (\w+) — (.*)$", human, re.MULTILINE)
    provato = next(eid for eid, _t, esito, _c in els if esito == "provato")
    contestato = next(eid for eid, _t, esito, _c in els if esito == "contestato")
    sids = dict(re.findall(r"^(S\d+) \[[^\]]*\] (.*)$", human, re.MULTILINE))
    paid = next(i for i, t in sids.items() if t.startswith("Anna Corsi ha versato"))
    signed = next(i for i, t in sids.items() if t.startswith("Anna Corsi ha firmato"))
    alleg = re.search(r"^ALLEGAZIONI DELLA CONTROPARTE[^:]*: (.*)$", human, re.MULTILINE).group(1).split(", ")
    strategy_call.seen.append({"provato": provato, "contestato": contestato, "paid": paid, "alleg": alleg})
    reply = {"linee": [
        {"articolo": "A1385cc", "eid": contestato, "tipo": "contestare", "rapporto": "principale",
         "perche_prima": "l'inadempimento di Ettore Bassi è il punto conteso", "cita_fatti": [alleg[0]], "cita_pronunce": [],
         "obiezione": {"testo": OBJ_CONTESTARE, "risposta_cita_fatti": [paid], "cita_pronunce": []}},
        {"articolo": "A1385cc", "eid": provato, "tipo": "fondare", "rapporto": "subordinata",
         "perche_prima": "la caparra versata è provata", "cita_fatti": [paid], "cita_pronunce": [],
         "obiezione": {"testo": OBJ_FONDARE, "risposta_cita_fatti": "nessuna risposta nei fatti", "cita_pronunce": []}}],
        "passi": [{"eid": contestato, "cosa": PASSO, "cita_fatti": [signed]}],
        "passi_immediati": [{"testo": IMMEDIATO, "cita_elementi": [provato], "cita_fatti": [paid]}],
        "contro_difesa": [{"allegazione": alleg[0], "risposta": CONTRO, "cita_linea": contestato, "cita_fatti": [paid]}],
        "verifiche_documentali": [{"testo": VERIFICA, "cita_fatti": [signed], "cita_elementi": []}]}
    return json.dumps(reply, ensure_ascii=False)


strategy_call.seen = []

case = types.SimpleNamespace(**{k: getattr(CIVIL, k) for k in dir(CIVIL) if not k.startswith("_") and k != "test_the_civil_fixture_end_to_end"})
case.CASE_ID = "v79180_civil"
case.strategy_call = strategy_call


def test_the_strategy_step_end_to_end_on_the_civil_fixture(monkeypatch, tmp_path):
    strategy_call.seen.clear()
    run = H.run(monkeypatch, tmp_path, case)
    state, answer, by = run["state"], run["answer"], run["by"]
    c = state["block_counters"]
    target = os.environ.get("FIXTURE_79_18_0_OUT")
    if target:
        H.write_report(target, "v79.18.0 synthetic CIVIL fixture with the strategy step (tests/test_v79_18_0_fixture.py)", run, "w")
    assert [x for x in run["failed"] if x.split(":")[0] in ("analysis_step", "strategy_step", "strategy_input",
                                                             "lawyer_method_compose", "lawyer_method_plan")] == []
    # the step ran once after the analysis step, on the set's article with its verified esiti; the objections call retired
    assert len(strategy_call.seen) == 1 and strategy_call.seen[0]["alleg"] and strategy_call.seen[0]["paid"].startswith("S")
    step = by["strategy_step"][0][0]
    assert step["status"] == "ok" and step["prompt_hash"] == S.prompt_hash() and step["set"] == ["1385 c.c."]
    assert step["reordered"] is True and len(step["final_order"]) == 2 and step["final_order"][0].endswith(strategy_call.seen[0]["provato"])
    assert "lawyer_method_objections" not in by
    assert c["strategy_calls"] == 1 and c["strategy_reasked"] == 0 and c["strategy_lines"] == 2
    assert c["strategy_rank_reordered"] == 1 and c["strategy_line_rejected"] == 0
    assert c["strategy_objections_unanswered"] == 1 and c["strategy_contro_difesa_added"] == 0 and c["strategy_passi_added"] >= 1
    assert c["strategy_rulings_cited"] == 0 and c["strategy_seconds"] >= 0 and state["timings"]["strategy_s"] == c["strategy_seconds"]
    assert state["strategy"]["ranked"][0]["tipo"] == "fondare" and state["strategy"]["ranked"][1]["tipo"] == "contestare"
    assert state["strategy"]["risk"] == "incerto"
    # §1: the ranked lines -- the fondare line (provato) first, the contestare one second
    sintesi = H.section(answer, "sintesi")
    assert sintesi.startswith("Pretesa: domanda — «esigere il doppio della caparra ai sensi dell'art. 1385 c.c.» (art. 1385 c.c.)."
                              "\n\nLinea principale: art. 1385 c.c. — «")
    assert "a titolo di caparra" in sintesi.split("Linea principale")[1].split("\n\n")[0]
    assert "risulta provato: «Anna Corsi ha versato a Ettore Bassi 15.000 euro a titolo di caparra confirmatoria»." in sintesi
    assert "Linea subordinata: art. 1385 c.c. — «" in sintesi and "» è da provare." in sintesi
    # §6: item 12's order, the table with the forza the code computed, the reordering said, the quesiti answered by line
    sol = H.section(answer, "soluzioni")
    assert sol.startswith("Le linee seguono gli esiti del punto 4.")
    titles = ["**Passi immediati (prova e termini)**", "**Linee per forza**", "**Risposta ai quesiti**", "**Contro-difesa**",
              "**Verifiche documentali**"]
    assert sorted(sol.index(t) for t in titles) == [sol.index(t) for t in titles]
    assert f"- {IMMEDIATO}: «Anna Corsi ha versato a Ettore Bassi 15.000 euro a titolo di caparra confirmatoria» [" in sol
    assert "| Linea 1 | art. 1385 c.c. |" in sol and "| fondare | principale | la caparra versata è provata | provato 1 / contestato 1 / da provare" in sol
    assert "| Linea 2 | art. 1385 c.c. |" in sol and "| contestare | subordinata |" in sol
    assert "Ordine ricalcolato dal codice per forza" in sol
    assert "- Q1 «Anna Corsi può esigere il doppio della caparra?»: art. 1385 c.c. (linea 1), art. 1385 c.c. (linea 2)" in sol
    # the fit tagged no quesito here: each row from the quesito's own search (the third brought 2946 and 1385 c.c.)
    assert "- Q3 «Entro quale termine si prescrive l'azione di Anna Corsi?»: art. 1385 c.c. (linea 1), art. 1385 c.c. (linea 2)" in sol
    assert S.quesiti_map(state["strategy_input"], {"Entro quale termine si prescrive l'azione di Anna Corsi?": ["2946 c.c.", "1385 c.c."]}
                         )["Q3"] == ["2946 c.c.", "1385 c.c."]
    assert f"»: {CONTRO}: «Anna Corsi ha versato a Ettore Bassi 15.000 euro a titolo di caparra confirmatoria» (linea: art. 1385 c.c., «" in sol
    assert f"- {VERIFICA}: «Anna Corsi ha firmato con Ettore Bassi un contratto preliminare di compravendita di un appartamento»." in sol
    # §7: the strategy's obiezioni, the verified answers -- a fact as written, or the data's «senza risposta» line
    obiezioni = H.section(answer, "obiezioni")
    assert obiezioni.startswith(f"**Obiezione 1.** {OBJ_FONDARE}\n**Risposta.** Senza risposta nei fatti narrati.")
    assert (f"**Obiezione 2.** {OBJ_CONTESTARE}\n**Risposta.** «Anna Corsi ha versato a Ettore Bassi 15.000 euro a titolo di caparra "
            "confirmatoria».") in obiezioni
    assert "Obiezione 3." not in obiezioni
    # §9: the risk word from the principale's forza; §10: the passi
    conclusioni = H.section(answer, "conclusioni")
    assert conclusioni.startswith("**Esiti dell'art. 1385 c.c.**") and "**Rischio per Anna Corsi**: incerto." in conclusioni
    assert conclusioni.splitlines()[1].startswith("**Linea principale**: art. 1385 c.c. — «") and "a titolo di caparra" in \
        conclusioni.splitlines()[1]                 # §9's lines are §1's: the ranked principale
    # §10 names each open element once: the passo where there is one, the analysis's question otherwise
    verificare_ = H.section(answer, "verificare")
    assert verificare_.count("inadempiente è invece la parte che l'ha ricevuta") == 1
    verificare = H.section(answer, "verificare")
    assert f"(art. 1385 c.c.) — {PASSO} («Anna Corsi ha firmato con Ettore Bassi un contratto preliminare" in verificare
    assert "(art. 1385 c.c.): da provare, nessun passo indicato dal modello." in verificare
    # the replay row carries the strategy line
    import importlib.util
    spec = importlib.util.spec_from_file_location("replay_cases_180f", HERE.parent / "scripts" / "replay_cases.py")
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    row = R.measure("c", state, answer, 1.0)
    assert row["strategy"]["reordered"] is True and row["strategy_counters"]["strategy_lines"] == 2
    assert "| strategy | ok: 1. art. 1385 c.c. " in R.case_table(row)
    # the 79.16.0 invariants that the strategy does not touch
    assert H.section(answer, "questione").startswith("La questione è se la domanda «esigere il doppio della caparra")
    assert c["qa_answer_insufficient"] == 0 and c["qa_answer_fragment"] == 0
