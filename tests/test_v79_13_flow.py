"""v79.13.0 build B -- the applicability judgment in the flow (APPLICABILITY_FIT), the model faked, on synthetic facts:

1. the brief's cases end to end through lawyer_method.compose: a rissa between two people (si, no invented rule, no
   removal); a loan with no instalments (1819 c.c. removed from the records, the candidates and §4, «rateale» quoted in
   the log); an advance payment (646 c.p. judged on «altrui», its effect only after verification); an inventing
   judgment (no effect, counted); a user-named or charged article given «no» (kept, printed as the defence's contest)
2. dipende: the touched element forced «da provare» (never «provato» by a later grading), the question in §10
3. flag off: nothing runs, the output byte-identical; flag on with every judgment «si»: the same answer
4. a removed norm never comes back: the decomposition's routes, a later answer, the scope check, the last pass
5. the nightly: the counters, the removed-norms list, the check

The article texts are statutory (public law: 588 c.p., 646 c.p., 1813 and 1819 c.c.); the facts are synthetic.
"""

import json
import os
import sys
from pathlib import Path

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

from src.rag import applicability as ap  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import issue_records as IR  # noqa: E402
from src.rag import lawyer_method as M  # noqa: E402
from src.rag import posture  # noqa: E402


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("THESIS_DETERMINISTIC", "")
    monkeypatch.setenv("ARTICLE_SCHEMAS_FILE", str(tmp_path / "schemas.jsonl"))
    monkeypatch.setenv("ARTICLE_SCHEMAS_REVIEWED_FILE", str(tmp_path / "reviewed.jsonl"))
    from src.rag import plan_then_write
    monkeypatch.setattr(plan_then_write, "negation_review", lambda items, known, call=None: [])
    monkeypatch.setattr(ap, "_default_call", _no_model)


def _no_model(messages, kwargs):
    raise AssertionError("no real model call in a test")


ART_588 = ("1. Chiunque partecipa a una rissa è punito con la multa fino a euro 2.000.\n"
           "2. Se nella rissa taluno rimane ucciso, o riporta lesione personale, la pena, per il solo fatto della "
           "partecipazione alla rissa, è della reclusione da sei mesi a sei anni. La stessa pena si applica se "
           "l'uccisione, o la lesione personale, avviene immediatamente dopo la rissa e in conseguenza di essa.")
ART_646 = ("1. Chiunque, per procurare a sé o ad altri un ingiusto profitto, si appropria il denaro o la cosa mobile "
           "altrui di cui abbia, a qualsiasi titolo, il possesso, è punito, a querela della persona offesa, con la "
           "reclusione da due a cinque anni e con la multa da euro 1.000 a euro 3.000.\n"
           "2. Se il fatto è commesso su cose possedute a titolo di deposito necessario, la pena è aumentata.")
ART_1813 = ("Il mutuo è il contratto col quale una parte consegna all'altra una determinata quantità di danaro o di "
            "altre cose fungibili, e l'altra si obbliga a restituire altrettante cose della stessa specie e qualità.")
ART_1819 = ("Se è stata convenuta la restituzione rateale delle cose mutuate e il mutuatario non adempie l'obbligo del "
            "pagamento anche di una sola rata, il mutuante può chiedere, secondo le circostanze, l'immediata "
            "restituzione dell'intero.")

IMPUTATO = {"rito": "penale", "family": "penale", "role": "imputato", "stage": "primo_grado",
            "entry": posture.entry_for("penale", "imputato", "primo_grado")}
CONVENUTO = {"rito": "civile", "family": "civile", "role": "convenuto", "stage": "primo_grado",
             "entry": posture.entry_for("civile", "convenuto", "primo_grado")}
OFFESA = {"rito": "penale", "family": "penale", "role": "persona_offesa", "stage": "pre_filing",
          "entry": posture.entry_for("penale", "persona_offesa", "pre_filing")}


def _sents(texts, alleged=()):
    return [{"id": f"S{i}", "text": t, "source": "caso", "certainty": "allegato" if i in alleged else "accertato"}
            for i, t in enumerate(texts, 1)]


def _req(i, text, status, supporting=(), comma=1):
    return {"id": f"R{i}", "comma": comma, "kind": "requisito", "status": status, "supporting": list(supporting),
            "text": text}


def _row(number, code, text, rubrica, **proc):
    return {"_article": (number, code), "s": {"plain_text": text, "titolo_articolo": rubrica},
            **({"_proc": {"version": 4, **proc}} if proc else {})}


def _item(article, fit, perche, ambito="fattispecie", condizione="", cf=None, cn=None):
    """v79.15.0 (item 3): a judgment cites ids -- the facts (cf: «cita_fatti», a «no» / «dipende») and the article's
    numbered periods (cn: «cita_norma», applicability.article_ids; a «si»); None: no such key."""
    out = {"articolo": article, "fit": fit, "perche": perche, "condizione": condizione, "ambito": ambito}
    if cf is not None:
        out["cita_fatti"] = cf
    if cn is not None:
        out["cita_norma"] = cn
    return out


def _model(monkeypatch, *items, seen=None):
    """The model faked: every call answers the items whose article the call carries."""
    def call(messages, kwargs):
        human = messages[-1].content
        labels = [line.split(": ", 1)[1] for line in human.splitlines() if line.startswith("ARTICOLO: ")]
        if seen is not None:
            seen.append({"labels": labels, "kwargs": kwargs})
        return json.dumps({"giudizi": [i for i in items if i["articolo"] in labels]}, ensure_ascii=False)
    monkeypatch.setattr(ap, "_default_call", call)


# --- the cases --------------------------------------------------------------------------------------------------------

RISSA_FACTS = ["Tizio e Caio, due persone, si sono colpiti a vicenda davanti a un bar.",
               "Caio sostiene che Tizio lo abbia colpito per primo e di aver riportato una ferita.",
               "Il pubblico ministero ha contestato a Tizio il reato di rissa (art. 588 c.p.)."]


def _rissa():
    recs = {"588 c.p.": {"article": "588 c.p.", "rubrica": "Rissa", "side": "threat", "charged": 1, "requisites": [
        _req(1, "partecipa a una rissa", "provato", ["S1"]),
        _req(2, "taluno rimane ucciso, o riporta lesione personale", "provato", ["S2"], comma=2)]}}
    rows = [_row("588", "c.p.", ART_588, "Rissa", pena="multa fino a euro 2.000 (co. 1)", querela=False)]
    return _case(recs, _sents(RISSA_FACTS, alleged=(2,)), rows, IMPUTATO, "Tizio")


LOAN_FACTS = ["Tizio ha prestato a Caio 5.000 euro da restituire in un'unica soluzione entro il 30 giugno.",
              "Caio non ha restituito la somma alla scadenza."]


def _loan(text_extra=""):
    recs = {"1813 c.c.": {"article": "1813 c.c.", "rubrica": "Nozione", "side": "threat", "requisites": [
                _req(1, "una parte consegna all'altra una determinata quantità di danaro", "provato", ["S1"]),
                _req(2, "l'altra si obbliga a restituire altrettante cose della stessa specie e qualità", "provato",
                     ["S1"])]},
            "1819 c.c.": {"article": "1819 c.c.", "rubrica": "Restituzione rateale", "side": "threat", "requisites": [
                _req(3, "è stata convenuta la restituzione rateale delle cose mutuate", "da provare"),
                _req(4, "il mutuatario non adempie l'obbligo del pagamento anche di una sola rata", "provato",
                     ["S2"])]}}
    rows = [_row("1813", "c.c.", ART_1813, "Nozione"), _row("1819", "c.c.", ART_1819, "Restituzione rateale")]
    topics = [{"title": "Mutuo", "articles": ["1813 c.c.", "1819 c.c."], "anchor": "1819 c.c.",
               "matched": ["1819 c.c."], "grades": {"1813 c.c.": 3, "1819 c.c.": 2}, "planned": 2}]
    return _case(recs, _sents(LOAN_FACTS), rows, CONVENUTO, "Caio", topics=topics, extra=text_extra)


ADVANCE_FACTS = ["Il cliente ha versato a Beta 3.000 euro come acconto sul prezzo dei lavori.",
                 "Il cliente afferma che Beta non ha eseguito i lavori e trattiene l'acconto."]


def _advance(charged=False):
    texts = ADVANCE_FACTS + (["Il pubblico ministero contesta a Beta l'art. 646 c.p."] if charged else [])
    rec = {"article": "646 c.p.", "rubrica": "Appropriazione indebita", "side": "threat", "requisites": [
        _req(1, "si appropria il denaro o la cosa mobile altrui", "da provare"),
        _req(2, "di cui abbia, a qualsiasi titolo, il possesso", "provato", ["S1"])]}
    if charged:
        rec["charged"] = 1
    rows = [_row("646", "c.p.", ART_646, "Appropriazione indebita", pena="reclusione da due a cinque anni (co. 1)",
                 querela=True)]
    return _case({"646 c.p.": rec}, _sents(texts, alleged=(2,)), rows, IMPUTATO if charged else OFFESA,
                 "Beta" if charged else "il cliente")


def _case(recs, sents, rows, posture_, client, topics=None, extra=""):
    text = " ".join(s["text"] for s in sents) + (f" {extra}" if extra else "")
    state = {"records": recs, "sentences": sents, "text": text, "request": "", "intake": {}, "analysis": {},
             "topics": topics or [{"title": "Tema", "articles": list(recs), "anchor": list(recs)[0],
                                   "matched": list(recs), "planned": len(recs)}]}
    return {"state": state, "rows": rows, "posture": posture_, "client": client}


def _flow(case, judge=True):
    """The answer stage as _write_answer runs it, from the records on: the judgment (APPLICABILITY_FIT on) -- before
    the decomposition --, the «dipende» marks, the link's settle, then the method's context, plan and compose."""
    state, rows, posture_ = case["state"], case["rows"], case["posture"]
    dead: set = set()
    if judge and ap.enabled():
        dead |= flow.fit_removed_keys(state)
        flow.applicability_fit(state, posture_, rows, dead, [])
        flow.fit_force(state)
    by_id = {s["id"]: s for s in state["sentences"]}
    for rec in state["records"].values():         # the link's settle (link_records), every element again
        for r in rec.get("requisites") or []:
            IR.settle(r, by_id)
    live = [r for r in rows if flow._key(r["_article"]) not in dead]
    ctx = M.prepare(state, posture_, live, state["text"], case["client"], "", list(state["records"]), [], [], [], [],
                    link_call=lambda *a: {}, case_id_="v7913")
    ctx["asked"] = []

    def call(messages):
        return {"valutazioni": {c["id"]: {"pro": "p", "contro": "c", "argomento": "a", "realismo": "medio"}
                                for c in ctx["candidates"]},
                "obiezioni": [{"obiezione": "L'altra parte insisterà sui fatti.", "risposta": "Vanno provati."}]}
    plan = M.method_plan(ctx, call=call)
    norms = "\n\n".join(f"L'art. {x} riguarda i fatti narrati." for x in state["records"])     # the writer's §4
    answer = M.compose([("Norme applicate ai fatti", norms)], ctx, plan, log={})
    return {"state": state, "ctx": ctx, "answer": answer, "dead": dead}


def _section(answer, key):
    return dict(M.answer_sections(answer)).get(M.title(key)) or ""


# --- 1. the brief's cases ---------------------------------------------------------------------------------------------

def test_rissa_between_two_people_si_on_the_quoted_text_no_rule_invented_nothing_removed(monkeypatch):
    monkeypatch.setenv("APPLICABILITY_FIT", "true")
    _model(monkeypatch, _item("588 c.p.", "si", "Tizio e Caio si sono colpiti a vicenda davanti a un bar (S1): "
                                                "«Chiunque partecipa a una rissa è punito».", cf=["S1"], cn=["A588cp.1"]))
    got = _flow(_rissa())
    c = got["state"]["block_counters"]
    assert c["fit_judged"] == 1 and c["fit_si"] == 1 and c["fit_si_verified"] == 1 and c["norms_removed_by_fit"] == 0
    assert c["fit_added_content"] == 0 and c["fit_unverified"] == 0
    assert "588 c.p." in got["state"]["records"] and not got["state"].get("fit_removed")
    off = _flow(_rissa(), judge=False)
    assert got["answer"] == off["answer"]                       # a «si» changes nothing


def test_rissa_a_rule_the_text_does_not_hold_is_discarded_and_has_no_effect(monkeypatch):
    monkeypatch.setenv("APPLICABILITY_FIT", "true")
    _model(monkeypatch, _item("588 c.p.", "no", "La rissa richiede almeno tre persone, mentre i fatti descrivono "
                                                "«Tizio e Caio, due persone» (S1).", cf=["S1"]))
    got = _flow(_rissa())
    c = got["state"]["block_counters"]
    # v79.16.1 (item 2): a «no» citing its fact is verified whatever rule its perché adds -- applied: the charged
    # article is kept as the defence's contest (fit_kept_named), an unprotected one would be removed
    assert c["fit_no"] == 1 and c["fit_no_verified"] == 1 and c["fit_unverified"] == 0 and c["fit_added_content"] == 0
    assert c["norms_removed_by_fit"] + c["fit_kept_named"] == 1


def test_loan_without_instalments_1819_removed_from_records_candidates_and_section_4(monkeypatch):
    monkeypatch.setenv("APPLICABILITY_FIT", "true")
    events = []
    real_log = flow._log
    monkeypatch.setattr(flow, "_log", lambda event, data, ms=None: events.append((event, data)) or real_log(
        event, data, ms))
    _model(monkeypatch,
           _item("1813 c.c.", "si", "Tizio ha prestato a Caio 5.000 euro (S1): «una parte consegna all'altra una "
                                    "determinata quantità di danaro».", cf=["S1"], cn=["A1813cc.1"]),
           _item("1819 c.c.", "no", "Il prestito è «da restituire in un'unica soluzione» (S1), non «rateale».",
                 ambito="rimedio", cf=["S1"], cn=["A1819cc.1"]))
    before = _flow(_loan(), judge=False)
    assert "1819" in before["answer"] and any(c.get("article") == "1819 c.c." for c in before["ctx"]["candidates"])
    got = _flow(_loan())
    state = got["state"]
    assert "1819 c.c." not in state["records"] and ("1819", "c.c.") in got["dead"]
    assert state["topics"][0]["articles"] == ["1813 c.c."] and state["topics"][0]["anchor"] == "1813 c.c."
    assert not any(c.get("article") == "1819 c.c." for c in got["ctx"]["candidates"])
    assert "1819" not in got["answer"] and "1819" not in _section(got["answer"], "norme")
    keys = ("article", "passage", "facts", "perche", "ambito")
    assert [{k: e[k] for k in keys} for e in state["fit_removed"]] == [
        {"article": "1819 c.c.", "passage": ART_1819, "facts": ["S1"],
         "perche": "Il prestito è «da restituire in un'unica soluzione» (S1), non «rateale».", "ambito": "rimedio"}]
    assert state["fit_removed"][0]["record"]["rubrica"] == "Restituzione rateale"      # (review) kept to put back
    logged = dict(events)["applicability_effects"]
    assert logged["removed"][0]["article"] == "1819 c.c." and "«rateale»" in logged["removed"][0]["perche"]
    c = state["block_counters"]
    assert c["fit_no"] == 1 and c["fit_si"] == 1 and c["norms_removed_by_fit"] == 1 and c["fit_kept_named"] == 0


def test_advance_payment_646_judged_on_altrui_its_effect_only_after_verification(monkeypatch):
    monkeypatch.setenv("APPLICABILITY_FIT", "true")
    # the reasoning adds a rule the passage, the facts and the note do not hold: no effect
    _model(monkeypatch, _item("646 c.p.", "no", "L'acconto «sul prezzo dei lavori» (S1) non è «altrui» perché la "
                                                "proprietà del denaro passa a chi lo riceve.",
                              cf=["S1"], cn=["A646cp.1"]))
    first = _flow(_advance())
    # v79.16.1 (item 2): the perché's words are never read -- the «no» citing its fact and its period is verified
    assert "646 c.p." not in first["state"]["records"] and first["state"].get("fit_removed")
    c = first["state"]["block_counters"]
    assert c["fit_added_content"] == 0 and c["norms_removed_by_fit"] == 1
    assert first["state"]["applicability"]["judgments"][0]["reason"] == ""
    # the same «no», its reasoning on the quoted passage and the fact only: verified, applied
    _model(monkeypatch, _item("646 c.p.", "no", "L'acconto versato a Beta «sul prezzo dei lavori» (S1) non è «il "
                                                "denaro o la cosa mobile altrui»: è di Beta.",
                              cf=["S1"], cn=["A646cp.1"]))
    second = _flow(_advance())
    assert "646 c.p." not in second["state"]["records"]
    assert "il denaro o la cosa mobile altrui" in second["state"]["fit_removed"][0]["passage"]     # its period
    assert second["state"]["block_counters"]["norms_removed_by_fit"] == 1
    assert "646" not in second["answer"]


def test_an_inventing_judgment_has_no_effect_and_is_counted(monkeypatch):
    monkeypatch.setenv("APPLICABILITY_FIT", "true")
    _model(monkeypatch, _item("588 c.p.", "dipende", "Tizio e Caio si sono colpiti a vicenda (S1): «Chiunque partecipa "
                                                     "a una rissa».",
                              condizione="Alla rissa hanno partecipato almeno tre persone?",
                              cf=["S1"], cn=["A588cp.1"]))
    got = _flow(_rissa())
    c = got["state"]["block_counters"]
    # v79.16.1 (item 2): the condizione is recorded, not verified -- the «dipende» with its ids is verified and open
    assert c["fit_dipende"] == 1 and c["fit_dipende_verified"] == 1 and c["fit_unverified"] == 0
    assert c["fit_added_content"] == 0
    assert [e["article"] for e in got["state"]["applicability"]["open"]] == ["588 c.p."]


def test_a_charged_article_given_no_is_kept_and_printed_as_the_defences_contest(monkeypatch):
    monkeypatch.setenv("APPLICABILITY_FIT", "true")
    _model(monkeypatch, _item("646 c.p.", "no", "L'acconto versato a Beta «sul prezzo dei lavori» (S1) non è «il "
                                                "denaro o la cosa mobile altrui»: è di Beta.",
                              cf=["S1"], cn=["A646cp.1"]))
    got = _flow(_advance(charged=True))
    state = got["state"]
    assert "646 c.p." in state["records"] and not state.get("fit_removed")
    assert state["block_counters"]["fit_kept_named"] == 1 and state["block_counters"]["norms_removed_by_fit"] == 0
    assert state["applicability"]["kept_named"][0]["article"] == "646 c.p."
    line = next(c for c in got["ctx"]["candidates"] if c["kind"] == "contest_fit")
    assert line["tesi"] == ("L'art. 646 c.p. non governa i fatti narrati: l'acconto versato a Beta «sul prezzo dei "
                            "lavori» non è «il denaro o la cosa mobile altrui»: è di Beta")
    assert line["argued"] == "code" and line["article"] == "646 c.p."
    solutions = _section(got["answer"], "soluzioni")
    assert "non governa i fatti narrati" in solutions and "«il denaro o la cosa mobile altrui»" in solutions
    assert "(S1)" not in solutions


def test_a_user_named_article_given_no_is_never_removed(monkeypatch):
    monkeypatch.setenv("APPLICABILITY_FIT", "true")
    _model(monkeypatch,
           _item("1813 c.c.", "si", "Tizio ha prestato a Caio 5.000 euro (S1): «una parte consegna all'altra una "
                                    "determinata quantità di danaro».", cf=["S1"], cn=["A1813cc.1"]),
           _item("1819 c.c.", "no", "Il prestito è «da restituire in un'unica soluzione» (S1), non «rateale».",
                 ambito="rimedio", cf=["S1"], cn=["A1819cc.1"]))
    case = _loan(text_extra="Il cliente chiede se si applica l'art. 1819 c.c.")
    assert "1819 c.c." in flow.fit_protected(case["state"], case["posture"])
    got = _flow(case)
    assert "1819 c.c." in got["state"]["records"] and not got["state"].get("fit_removed")
    assert got["state"]["block_counters"]["fit_kept_named"] == 1
    assert any(c["kind"] == "contest_fit" and c["article"] == "1819 c.c." for c in got["ctx"]["candidates"])
    assert "1819" in got["answer"]


def test_the_client_own_named_norm_given_no_is_the_other_sides_objection(monkeypatch):
    monkeypatch.setenv("APPLICABILITY_FIT", "true")
    _model(monkeypatch, _item("1819 c.c.", "no", "Il prestito è «da restituire in un'unica soluzione» (S1), non "
                                                 "«rateale».", ambito="rimedio",
                              cf=["S1"], cn=["A1819cc.1"]))
    case = _loan(text_extra="Il cliente chiede se si applica l'art. 1819 c.c.")
    for rec in case["state"]["records"].values():
        rec["side"] = "weapon"                  # the client's own norm, not the claim against him
    got = _flow(case)
    first = got["ctx"]["objections_fallback"][0]
    assert first["obiezione"] == ("La controparte potrà sostenere che l'art. 1819 c.c. non governa i fatti narrati: "
                                  "il prestito è «da restituire in un'unica soluzione», non «rateale»")
    assert f"«{ART_1819}»" in first["risposta"]          # v79.15.0 (item 3): the cited period, the article's words


# --- 2. dipende -------------------------------------------------------------------------------------------------------

def test_dipende_forces_the_touched_element_da_provare_and_asks_the_question_in_section_10(monkeypatch):
    monkeypatch.setenv("APPLICABILITY_FIT", "true")
    _model(monkeypatch, _item("588 c.p.", "dipende", "Caio sostiene di aver riportato una ferita (S2): dipende se "
                                                     "taluno «riporta lesione personale».",
                              condizione="Caio ha riportato una lesione personale?",
                              cf=["S2"], cn=["A588cp.2"]))
    case = _rissa()
    case["state"]["sentences"][1]["certainty"] = "accertato"      # a certain fact that would prove the element
    got = _flow(case)
    reqs = {r["id"]: r for r in got["state"]["records"]["588 c.p."]["requisites"]}
    assert reqs["R2"]["status"] == "da provare" and reqs["R2"]["fit_open"] == "Caio ha riportato una lesione personale?"
    assert reqs["R2"]["why"] == "resta da provare: dipende dalla risposta a «Caio ha riportato una lesione personale?»"
    assert reqs["R1"]["status"] == "provato" and not reqs["R1"].get("fit_open")
    assert got["state"]["applicability"]["forced"] == {"588 c.p.": ["R2"]}
    by_id = {s["id"]: s for s in got["state"]["sentences"]}
    IR.settle(reqs["R2"], by_id)                                  # any later grading: still «da provare»
    assert reqs["R2"]["status"] == "da provare"
    M.narrated_event_proven(got["state"]["records"], "Caio rimane ucciso dopo la rissa.")     # the event narrated
    assert reqs["R2"]["status"] == "da provare"
    points = _section(got["answer"], "verificare")
    assert "Art. 588 c.p.: Caio ha riportato una lesione personale?" in points
    assert "dipende dalla risposta a: Caio ha riportato una lesione personale?" in _section(got["answer"], "norme")
    assert got["state"]["block_counters"]["fit_dipende"] == 1


def test_a_new_answer_clears_the_old_marks_before_its_own():
    recs = {"588 c.p.": {"requisites": [dict(_req(1, "partecipa a una rissa", "provato"), fit_open="vecchia?")]}}
    assert ap.force_open(recs, []) == {} and "fit_open" not in recs["588 c.p."]["requisites"][0]


def test_printable_drops_the_fact_ids():
    assert ap.printable("Il prestito (S1) è da restituire, non «rateale» (S1, S2).", ["S1", "S2"]) == \
        "Il prestito è da restituire, non «rateale»."
    assert ap.printable("S2 dice che c'è una ferita?", ["S2"]) == "il fatto narrato dice che c'è una ferita?"
    assert ap.clause("Il prestito è da restituire.") == "il prestito è da restituire"
    assert ap.clause("Beta trattiene l'acconto.") == "Beta trattiene l'acconto"


# --- 3. flag off ------------------------------------------------------------------------------------------------------

def test_flag_off_nothing_runs_and_the_output_is_byte_identical(monkeypatch):
    monkeypatch.delenv("APPLICABILITY_FIT", raising=False)
    monkeypatch.setattr(ap, "judge", lambda *a, **k: (_ for _ in ()).throw(AssertionError("judged with the flag off")))
    case = _loan()
    got = _flow(case)                                   # the step's own gate: enabled() off
    assert not any(k.startswith("fit_") or k == "norms_removed_by_fit" for k in got["state"].get("block_counters", {}))
    assert "applicability" not in got["state"] and "fit_removed" not in got["state"]
    assert "fit_contests" not in got["ctx"] and "fit_removed" not in got["ctx"]
    assert got["answer"] == _flow(_loan(), judge=False)["answer"]


def test_decompose_records_with_no_hook_is_unchanged_and_with_one_judges_before_splitting(monkeypatch):
    row = _row("1819", "c.c.", ART_1819, "Restituzione rateale")
    # v79.14.2 (A): the case text holds the facts a «no» quotes (as the flow's sentences come from it)
    base = {"analysis": {}, "text": " ".join(LOAN_FACTS) + " Tizio chiede a Caio la restituzione dell'intero prestito.",
            "request": "",
            "intake": {}, "records": {}, "sentences": _sents(LOAN_FACTS), "questions": [],
            "ranked": ["1819 c.c."]}
    plain = json.loads(json.dumps(base))
    flow.decompose_records(plain, CONVENUTO, [row], set(), fetch=lambda refs: [])
    assert "1819 c.c." in plain["records"] and plain["decompose_why"] == {}
    monkeypatch.setenv("APPLICABILITY_FIT", "true")
    _model(monkeypatch, _item("1819 c.c.", "no", "Il prestito è «da restituire in un'unica soluzione» (S1), non "
                                                 "«rateale».", ambito="rimedio",
                              cf=["S1"], cn=["A1819cc.1"]))
    judged = json.loads(json.dumps(base))
    dead: set = set()
    calls = []

    def hook(wanted, have):
        calls.append(list(wanted))
        return flow.applicability_fit(judged, CONVENUTO, [row], dead, wanted, have)
    flow.decompose_records(judged, CONVENUTO, [row], dead, fetch=lambda refs: [], fit=hook)
    assert calls == [["1819 c.c."]]                          # the ranked route's candidate, judged before its split
    assert "1819 c.c." not in judged["records"] and judged["decompose_why"] == {"1819 c.c.": "fit_removed"}
    assert ("1819", "c.c.") in dead


# --- 4. never back ----------------------------------------------------------------------------------------------------

def test_a_removed_norm_never_comes_back(monkeypatch):
    monkeypatch.setenv("APPLICABILITY_FIT", "true")
    _model(monkeypatch,
           _item("1813 c.c.", "si", "Tizio ha prestato a Caio 5.000 euro (S1): «una parte consegna all'altra una "
                                    "determinata quantità di danaro».", cf=["S1"], cn=["A1813cc.1"]),
           _item("1819 c.c.", "no", "Il prestito è «da restituire in un'unica soluzione» (S1), non «rateale».",
                 ambito="rimedio", cf=["S1"], cn=["A1819cc.1"]))
    got = _flow(_loan())
    state = got["state"]
    # a later answer: the decomposition's routes (the ranked, an alternative) never split it again
    state.update(analysis={}, questions=[], ranked=["1819 c.c."], retrieval_top=[{"rank": 1, "article": "1819 c.c."}])
    dead = flow.fit_removed_keys(state)
    flow.decompose_records(state, CONVENUTO, got["ctx"].get("rows") or [_row("1819", "c.c.", ART_1819, "R")], dead,
                           fetch=lambda refs: [])
    assert "1819 c.c." not in state["records"] and state["decompose_why"].get("1819 c.c.") == "fit_removed"
    assert flow.ranked_trace(state)[0]["why"] == flow.RANKED_WHY["fit_removed"]
    # the scope check: a plan sentence citing it removed
    assert "1819 c.c." not in M.article_scope({**got["ctx"], "case": "art. 1819 c.c."})
    # the last pass: a writer's sentence citing it, dropped and counted
    out = flow.fit_removed_out(state, "Primo punto sul prestito. L'art. 1819 c.c. dà l'intero. L'art. 1813 c.c. resta.")
    assert out == "Primo punto sul prestito. L'art. 1813 c.c. resta."
    assert state["block_counters"]["fit_removed_stripped"] == 1 and state["block_counters"]["fit_removed_left"] == 0
    # a second answer's judgment: the removed norm is not judged again, never restored
    seen = []
    _model(monkeypatch, _item("1813 c.c.", "si", "Tizio ha prestato a Caio 5.000 euro (S1): «una parte consegna "
                                                 "all'altra una determinata quantità di danaro»."), seen=seen)
    flow.applicability_fit(state, CONVENUTO, [_row("1813", "c.c.", ART_1813, "Nozione"),
                                              _row("1819", "c.c.", ART_1819, "R")], dead, ["1819 c.c."])
    assert seen[0]["labels"] == ["1813 c.c."] and "1819 c.c." not in state["records"]
    assert [e["article"] for e in state["fit_removed"]] == ["1819 c.c."]


def test_write_answer_flag_off_and_all_si_byte_identical_and_a_no_removes(monkeypatch):
    import test_v60f as v60f
    writer = lambda jobs: [(t, v60f.WRITER.get(t, f"Testo di {t}.")) for t, *_ in jobs]
    monkeypatch.delenv("APPLICABILITY_FIT", raising=False)
    off_state = v60f._graded_state(monkeypatch)
    off, _ = v60f._answer(monkeypatch, off_state, writer)
    assert "applicability" not in off_state and not any(k.startswith("fit_") for k in off_state["block_counters"])

    def every_si(messages, kwargs):                     # each norm «si» on its own first words and S1
        human = messages[-1].content
        items, label = [], None
        for line in human.splitlines():
            if line.startswith("ARTICOLO: "):
                label = line.split(": ", 1)[1]
            elif label and line.startswith("[") and "] " in line:
                words = line.split("] ", 1)[1].split()[:4]       # v79.15.0: the period it cites, by its id
                items.append(_item(label, "si", f"«{' '.join(words).rstrip(',.;:')}» (S1).", cf=["S1"],
                                   cn=[line[1:line.index("]")]]))     # v79.16.1 (item 2): a «si» cites its facts
                label = None
        return json.dumps({"giudizi": items}, ensure_ascii=False)
    monkeypatch.setenv("APPLICABILITY_FIT", "true")
    monkeypatch.setattr(ap, "_default_call", every_si)
    on_state = v60f._graded_state(monkeypatch)
    on, _ = v60f._answer(monkeypatch, on_state, writer)
    c = on_state["block_counters"]
    assert c["fit_judged"] >= 2 and c["fit_si"] == c["fit_judged"] and c["norms_removed_by_fit"] == 0
    assert on == off                                    # every judgment «si»: the same answer, byte for byte

    def no_1061(messages, kwargs):
        reply = json.loads(every_si(messages, kwargs))
        human = messages[-1].content
        for it in reply["giudizi"]:
            if it["articolo"] == "1061 c.c.":       # (review) a «no» says what the facts lack, on the quoted text
                # v79.15.0 (item 3): a «no» cites the fact it rests on by its id (S1's first words in the perché)
                s1 = next(x.split("] ", 1)[1] for x in human.splitlines() if x.startswith("S1 ["))
                quote = ' '.join(s1.split()[:4]).rstrip(',.;:')
                it.update(fit="no", perche=f"«{quote}» (S1): non è tra le «servitù non apparenti».", cita_fatti=["S1"])
        return json.dumps(reply, ensure_ascii=False)
    monkeypatch.setattr(ap, "_default_call", no_1061)
    cut_state = v60f._graded_state(monkeypatch)
    cut, _ = v60f._answer(monkeypatch, cut_state, writer)
    assert "1061 c.c." not in cut_state["records"] and "1061" not in cut
    assert [e["article"] for e in cut_state["fit_removed"]] == ["1061 c.c."]
    assert cut_state["block_counters"]["norms_removed_by_fit"] == 1
    assert cut_state["block_counters"]["fit_removed_left"] == 0
    assert all("1061 c.c." not in (t.get("articles") or []) for t in cut_state["topics"])


# --- 5. the nightly ---------------------------------------------------------------------------------------------------

def test_the_nightly_lists_every_removed_norm_and_checks_the_answer(monkeypatch):
    import nightly_sim
    monkeypatch.setenv("APPLICABILITY_FIT", "true")
    _model(monkeypatch,
           _item("1813 c.c.", "si", "Tizio ha prestato a Caio 5.000 euro (S1): «una parte consegna all'altra una "
                                    "determinata quantità di danaro».", cf=["S1"], cn=["A1813cc.1"]),
           _item("1819 c.c.", "no", "Il prestito è «da restituire in un'unica soluzione» (S1), non «rateale».",
                 ambito="rimedio", cf=["S1"], cn=["A1819cc.1"]))
    got = _flow(_loan())
    state = got["state"]
    fit = nightly_sim.fit_of(state)
    assert fit["ran"] and fit["removed"] == [{"article": "1819 c.c.", "passage": ART_1819,
                                              "perche": "Il prestito è «da restituire in un'unica soluzione» (S1), "
                                                        "non «rateale»."}]
    ok = nightly_sim.fit_check(got["answer"], state)
    assert ok["ok"] and ok["check"] == "applicability_fit"
    bad = nightly_sim.fit_check(got["answer"] + "\n\nL'art. 1819 c.c. torna qui.", state)
    assert not bad["ok"] and bad["cited"] == ["1819 c.c."]
    cell = {"id": "civile_convenuto", "counters": flow.case_counters(state, 1.0), "applicability": fit}
    lines = "\n".join(nightly_sim.fit_lines([cell]))
    assert "## Applicability judgment (APPLICABILITY_FIT)" in lines
    assert f"- removed 1819 c.c.: «{ART_1819}» — Il prestito è «da restituire in un'unica soluzione» (S1), non " \
        "«rateale»." in lines
    assert cell["counters"]["norms_removed_by_fit"] == 1 and cell["counters"]["fit_added_content_in_effects"] == 0
    assert nightly_sim.fit_check("", {})["why"] == "not run" and nightly_sim.fit_lines([{"id": "x"}]) == []


def test_the_nightly_turns_the_flag_on_for_its_cells(monkeypatch, tmp_path):
    import nightly_sim
    monkeypatch.delenv("APPLICABILITY_FIT", raising=False)
    monkeypatch.setattr(nightly_sim, "cells", lambda path=None: [])
    monkeypatch.setattr(nightly_sim, "stage_check", lambda: {"ok": True, "report": ""})
    nightly_sim.main(["--answers", str(tmp_path), "--no-stages"])
    assert os.environ["APPLICABILITY_FIT"] == "true"
    nightly_sim.main(["--answers", str(tmp_path), "--no-stages", "--no-applicability-fit"])
    assert os.environ["APPLICABILITY_FIT"] == "false"
    monkeypatch.delenv("APPLICABILITY_FIT", raising=False)


def test_the_budget_counts_from_the_flow_steps_start():
    import time

    def slow(messages, kwargs):
        time.sleep(0.6)
        return json.dumps({"giudizi": []})
    norm = {"article": "588 c.p.", "rubrica": "Rissa", "text": ART_588}
    facts = [{"id": "S1", "text": RISSA_FACTS[0], "certainty": "accertato"}]
    late = ap.judge([norm], facts, "imputato", call=slow, started=time.time() - (ap.config()["deadline_s"] - 0.2))
    assert late["counters"]["fit_timeouts"] == 1 and late["judgments"][0]["reason"] == "chiamata oltre il tempo"
    on_time = ap.judge([norm], facts, "imputato", call=slow)
    assert on_time["counters"]["fit_timeouts"] == 0
