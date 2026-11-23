"""v79.15.0 build B -- the synthetic PENAL fixture end to end (invented names and facts; statutory text of 646 and
62-bis c.p.) through the real entry, intake.request_intake -> interactive_flow.single_shot -> write_answer ->
_write_answer, with ANALYSIS_STEP, CASE_FRAME, APPLICABILITY_FIT and LAWYER_METHOD on; every model faked at its source,
every other call and every outbound socket refused (tests/_fixture_79_15_0.py).

The client, Dario Melis, is indagato for appropriazione indebita (646 c.p.) and denies; the Procura asserts the profit
(«Secondo la Procura, …») and the company the appropriation («La società sostiene che …»); the only certain fact is his
custody of the till. 62-bis c.p. is a fitted general-part norm. The faked analysis offers 646 c.p. as the line on its
dolo element, «provato» on the Procura's and the company's assertions (the code downgrades both to «contestato»), the
possession «provato» on the certain fact, one law statement citing its period and one citing none (dropped).

Expected: §1 «Linea principale: art. 646 c.p. — …» on the dolo element; §4's table from the verified verdicts --
«contestato (allegazione …)» for the two assertions, «provato» only for the possession; 62-bis c.p. named among the
remedies, no table, no count; §6 and §9 the verified lines only, never «Ammettere la condotta» nor «Contenimento»; the
old engine logged «legacy_plan» and printed nowhere. With FIXTURE_79_15_0_OUT set, the report is appended there.
"""

import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _fixture_79_15_0 as H  # noqa: E402
import test_v79_14_2_fixture as F  # noqa: E402

CASE_ID = "v79150_penal"
CLIENT = "Dario Melis"
PROCURA = "Secondo la Procura, Dario Melis ha trattenuto 4.000 euro degli incassi per procurarsi un profitto"
SOCIETA = "La società sostiene che Dario Melis si è appropriato del denaro della cassa"
TEXT = ("Assisto Dario Melis, indagato per appropriazione indebita. Dario Melis lavorava come cassiere presso la "
        f"società Ferramenta Lunardi e custodiva la cassa del negozio. {PROCURA}. {SOCIETA}. Dario Melis nega di aver "
        "trattenuto il denaro. Il pubblico ministero procede per appropriazione indebita ai sensi dell'art. 646 c.p. e "
        "le indagini preliminari sono in corso. Quali rischi corre Dario Melis?")
ART_62BIS = ("Il giudice, indipendentemente dalle circostanze previste nell'articolo 62, può prendere in considerazione "
             "altre circostanze diverse, qualora le ritenga tali da giustificare una diminuzione della pena. Esse sono "
             "considerate in ogni caso, ai fini dell'applicazione di questo capo, come una sola circostanza, la quale "
             "può anche concorrere con una o più delle circostanze indicate nel predetto articolo 62.")
ARTICLES = (("646", "c.p.", F.ART_646, "Appropriazione indebita", "LEGAL_DOC::cp", "Codice Penale 2026"),
            ("62-bis", "c.p.", ART_62BIS, "Circostanze attenuanti generiche", "LEGAL_DOC::cp", "Codice Penale 2026"))
DOCUMENTS = ({"id": "LEGAL_DOC::cp", "name": "Codice Penale 2026"},)
TITLE, ANCHOR, RUBRICA = "Appropriazione degli incassi", "646 c.p.", "Appropriazione indebita."
FRAME_REPLY = {
    "parti": [{"nome": "Dario Melis", "ruolo_nel_fatto": "cassiere", "lato": "cliente"},
              {"nome": "Ferramenta Lunardi", "ruolo_nel_fatto": "società", "lato": "controparte"}],
    "cliente": {"nome": "Dario Melis", "perche": "«Assisto Dario Melis, indagato per appropriazione indebita»"},
    "ruolo": {"valore": "indagato", "perche": "«Assisto Dario Melis, indagato per appropriazione indebita»"},
    "rito": {"valore": "penale", "perche": "«Il pubblico ministero procede per appropriazione indebita»"},
    "stadio": {"valore": "indagini", "perche": "«le indagini preliminari sono in corso»"},
    "addebito_o_domanda": {"articolo": "646 c.p.", "descrizione": "appropriazione indebita",
                           "frase": "«procede per appropriazione indebita ai sensi dell'art. 646 c.p.»"},
    "istituto": None,
    "quesiti": ["Quali rischi corre Dario Melis?"]}


def _sid(parsed_or_human, start):
    facts = parsed_or_human if isinstance(parsed_or_human, dict) else {
        i: t for i, t in re.findall(r"^(S\d+) \[[^\]]+\] (.*)$", parsed_or_human, re.MULTILINE)}
    return next(i for i, t in facts.items() if t.startswith(start))


def fit_reply(messages, kwargs):
    """The fit faked: both norms «si», each citing its first period."""
    human = messages[-1].content
    cash = _sid(human, "Dario Melis lavorava")
    items = {"646 c.p.": F._fit_item("646 c.p.", "si", f"Dario Melis custodiva la cassa del negozio ({cash}): «di cui "
                                                     "abbia, a qualsiasi titolo, il possesso»."),
             "62-bis c.p.": F._fit_item("62-bis c.p.", "si", "Il giudice «può prendere in considerazione altre "
                                                           "circostanze diverse».", ambito="circostanza")}
    labels = [line.split(": ", 1)[1] for line in human.splitlines() if line.startswith("ARTICOLO: ")]
    return json.dumps({"giudizi": [items[x] for x in labels if x in items]}, ensure_ascii=False)


def link(topic, reqs, sents):
    """The element link faked: the profit on the Procura's assertion, the appropriation on the company's, the possession
    on the certain fact of the till."""
    sid = {s["text"]: s["id"] for s in sents}
    procura = next(i for t, i in sid.items() if t.startswith("Secondo la Procura"))
    societa = next(i for t, i in sid.items() if t.startswith("La società sostiene"))
    cash = next(i for t, i in sid.items() if t.startswith("Dario Melis lavorava"))
    out = {}
    for label, r in reqs:
        text = str(r.get("text") or "").lower()
        if label.startswith("646") and "profitto" in text:
            out[r["id"]] = {"sostiene": [procura], "contraddice": [], "manca": ""}
        elif label.startswith("646") and "appropria" in text:
            out[r["id"]] = {"sostiene": [societa], "contraddice": [], "manca": ""}
        elif label.startswith("646") and "possesso" in text:
            out[r["id"]] = {"sostiene": [cash], "contraddice": [], "manca": ""}
    return out


def _pick(elements, word):
    return next(e for e in elements if word in e.lower())


def analysis_reply(parsed, attempt):
    """The analysis faked (one call): the line on 646's dolo element; «provato» on the Procura's and the company's
    assertions (downgraded by the code), the possession on the certain fact; a law statement citing its period, one
    citing none."""
    facts, norms = parsed["facts"], parsed["norms"]
    el = norms["646 c.p."]["elements"]
    procura, societa = _sid(facts, "Secondo la Procura"), _sid(facts, "La società sostiene")
    cash, denies = _sid(facts, "Dario Melis lavorava"), _sid(facts, "Dario Melis nega")
    dolo = _pick(el, "profitto")

    def ref(e):                 # v79.15.1: the element cited by its id
        return H.eid(parsed, "646 c.p.", e)
    elementi = [{"articolo": "646 c.p.", "elemento": ref(dolo), "esito": "provato", "cita_fatti": [procura],
                 "parte_che_afferma": None}]
    for e in el:
        if e == dolo:
            continue
        if "appropria" in e.lower():
            elementi.append({"articolo": "646 c.p.", "elemento": ref(e), "esito": "provato",
                             "cita_fatti": [societa],
                             "parte_che_afferma": None})
        elif "possesso" in e.lower():
            elementi.append({"articolo": "646 c.p.", "elemento": ref(e), "esito": "provato",
                             "cita_fatti": [cash],
                             "parte_che_afferma": None})
        else:
            elementi.append({"articolo": "646 c.p.", "elemento": ref(e), "esito": "da_provare",
                             "cita_fatti": [],
                             "parte_che_afferma": None})
    return {"linea_principale": {"articolo": "646 c.p.", "elemento_in_gioco": ref(dolo),
                                 "tesi": "Il fine di profitto di Dario Melis non è provato: lo afferma solo la Procura, "
                                         "e Dario Melis nega di aver trattenuto il denaro.",
                                 "cita_fatti": [procura, denies], "cita_norma": [norms["646 c.p."]["periods"][0]]},
            "linea_subordinata": None, "elementi": elementi,
            "affermazioni_di_diritto": [
                {"testo": "L'art. 646 c.p. punisce chi, per procurare a sé o ad altri un ingiusto profitto, si "
                          "appropria il denaro o la cosa mobile altrui di cui abbia il possesso.",
                 "cita_norma": [norms["646 c.p."]["periods"][0]]},
                {"testo": "L'art. 62-bis c.p. riduce sempre la pena di un terzo.", "cita_norma": []}]}


def writer_line(label):
    if label.startswith("646"):
        return ("L'art. 646 c.p. punisce chi, per procurare a sé o ad altri un ingiusto profitto, si appropria il "
                "denaro o la cosa mobile altrui di cui abbia il possesso.")
    return "L'art. 62-bis c.p. prevede che la pena sia sempre dimezzata per chi è incensurato."


case = sys.modules[__name__]


def test_the_penal_fixture_end_to_end(monkeypatch, tmp_path):
    run = H.run(monkeypatch, tmp_path, case)
    state, answer, by = run["state"], run["answer"], run["by"]
    c = state["block_counters"]
    target = os.environ.get("FIXTURE_79_15_0_OUT")
    if target:
        H.write_report(target, "v79.15.0 synthetic PENAL fixture (tests/test_v79_15_0_fixture_penal.py)", run, "a")
    assert [x for x in run["failed"] if x.split(":")[0] in ("analysis_step", "applicability_fit", "legacy_plan",
                                                             "lawyer_method_compose", "lawyer_method_plan")] == []
    # the step: one call, the frame's charge the line, nothing rejected
    step = by["analysis_step"][0][0]
    assert step["status"] == "ok" and step["counters"]["analysis_calls"] == 1 == len(run["model"].calls)
    assert c["analysis_line_rejected"] == 0 and c["analysis_frame_charge_missing"] == 0
    # provato on the Procura's and on the company's assertion -> contestato; the possession stays provato
    assert c["analysis_provato_downgraded"] == 2
    assert c["law_statements_kept"] == 1 and c["law_statements_dropped"] == 1
    # §1: the line on the charge's dolo element
    sintesi = H.section(answer, "sintesi")
    # v79.15.3 (item 1): the tesi is the code's -- the defence (ruolo indagato), the element in play contested
    assert sintesi.startswith("Linea principale: art. 646 c.p. — L'accusa deve provare «per procurare a sé o ad altri "
                              "un ingiusto profitto» (art. 646 c.p.).")
    assert "Il fine di profitto di Dario Melis non è provato" not in answer      # the model's tesi: the log only
    # §4: the verified verdicts only; no allegation provato; 62-bis a remedy -- no table, no count
    col = H.esito_column(answer)
    assert col and all(x.startswith("646 c.p.") for x in col), col
    provati = [x for x in col if x.endswith("| provato")]
    assert len(provati) == 1 and "possesso" in provati[0], col
    assert sum(1 for x in col if "| contestato (allegazione" in x) == 2, col
    norme = H.section(answer, "norme")
    assert "**Rimedi e parte generale collegati alla linea**: art. 62-bis c.p." in norme
    assert "**Circostanze**" not in norme and "62-bis c.p.**" not in norme
    counts = [x for x in answer.splitlines() if re.search(r"\b\d+ su \d+\b", x)]
    assert counts and all("646 c.p." in x for x in counts), counts
    # §6 / §9: the verified lines only; never the containment or the admission
    sol = H.section(answer, "soluzioni")
    assert sol.count("| Linea ") == 2 and "art. 646 c.p." in sol        # the head and the leading line
    assert "Contenimento" not in answer and "mmettere la condotta" not in answer
    concl = H.section(answer, "conclusioni")
    assert concl.splitlines()[1].startswith("**Linea principale**: art. 646 c.p.") and "Esito probabile" not in answer  # v79.16.0 (D2)
    # the Q&A: the verified statement printed; the unverified 62-bis one dropped wherever the Q&A carried it
    assert "**Affermazioni di diritto verificate**" in answer and "riduce sempre la pena di un terzo" not in answer
    # the old engine: logged, compared, printed nowhere
    legacy = by["legacy_plan"][0][0]
    assert legacy["rendered"] is False and legacy["linea_principale"]["articolo"]
    assert legacy["linea_principale"]["tesi"] not in answer
    assert "legacy_disagrees_line" in c and "legacy_disagrees_esito" in c


def test_a_failed_step_prints_the_undetermined_line_never_the_old_engines(monkeypatch, tmp_path):
    """The analysis call failing (the model raising): status failed, §1 «da determinare (linea non verificata)», no
    element table, no count, §6 and §9 the undetermined line -- the old engine's line logged only."""
    import types

    def broken(parsed, attempt):
        raise RuntimeError("the analysis model is down")
    failing = types.SimpleNamespace(**{k: getattr(case, k) for k in dir(case) if not k.startswith("__")})
    failing.analysis_reply = broken
    failing.CASE_ID = "v79150_penal_failed"
    run = H.run(monkeypatch, tmp_path, failing)
    state, answer, by = run["state"], run["answer"], run["by"]
    c = state["block_counters"]
    assert by["analysis_step"][0][0]["status"] == "failed" and c["analysis_failed"] == 1 and c["analysis_calls"] == 1
    assert H.section(answer, "sintesi") == "Linea principale: da determinare (linea non verificata)."
    assert H.esito_column(answer) == [] and "Nessuna linea verificata" in H.section(answer, "norme")
    assert not re.search(r"\b\d+ su \d+\b", answer)
    assert "da determinare (linea non verificata)" in H.section(answer, "soluzioni")
    assert H.section(answer, "conclusioni") == "**Linea principale**: da determinare (linea non verificata)."
    legacy = by["legacy_plan"][0][0]
    assert legacy["legacy_disagrees_line"] == 1 and legacy["linea_principale"]["tesi"] not in answer
    assert "L'accusa deve provare" not in answer and "Esito probabile" not in answer
