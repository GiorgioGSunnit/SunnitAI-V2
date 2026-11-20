"""v79.14.2 -- the two verified model steps wired into the pipeline (the model faked; synthetic texts and facts, invented
names; statutory text of 640 c.p., 1439, 2043 and 1218 c.c.):

A. the fit verification by verdict type (src/rag/applicability.py) -- v79.15.0 (item 3): by ids, the string matcher
   removed: a «si» cites the article's numbered periods (cita_norma), a «no» and a «dipende» the facts (cita_fatti),
   no article period required for a «no»; an unverified «no» keeps the norm (fit_no_unverified_ids); a verified «no»
   removes it (fit_removed, the removal's «applicability_fit_discard» line with the reason «removed: fit no
   (verified)»); the prompt's id fields
B. the verified frame binds the charge set (src/rag/lawyer_method.py): penal rito + imputato/indagato -- a c.c.
   candidate excluded from the charge, the candidates, «Linea principale» and «Esito probabile», kept as context;
   civil rito -- a c.p. article excluded; a frame not verified on the rito or the role -- no binding; the charge set
   emptied -- no fallback to the excluded, the corpus line in §1
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

from src.rag import applicability as ap  # noqa: E402
from src.rag import case_frame as cf  # noqa: E402
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

    def refuse(messages, kwargs):
        raise AssertionError("no real model call in a test")
    monkeypatch.setattr(ap, "_default_call", refuse)
    monkeypatch.setattr(cf, "_default_call", refuse)


ART_640 = ("Chiunque, con artifizi o raggiri, inducendo taluno in errore, procura a sé o ad altri un ingiusto profitto "
           "con altrui danno, è punito con la reclusione da sei mesi a tre anni e con la multa da euro 51 a euro 1.032.")
ART_1439 = ("Il dolo è causa di annullamento del contratto quando i raggiri usati da uno dei contraenti sono stati tali "
            "che, senza di essi, l'altra parte non avrebbe contrattato. Quando i raggiri sono stati usati da un terzo, "
            "il contratto è annullabile se essi erano noti al contraente che ne ha tratto vantaggio.")
ART_2043 = ("Qualunque fatto doloso o colposo, che cagiona ad altri un danno ingiusto, obbliga colui che ha commesso il "
            "fatto a risarcire il danno.")

FACTS = [{"id": "S1", "text": "Bruno Gelso ha venduto a Carla Tasso un orologio dichiarandolo d'oro.",
          "certainty": "accertato"},
         {"id": "S2", "text": "L'orologio era placcato e Carla Tasso ha pagato 4.000 euro con un bonifico.",
          "certainty": "accertato"},
         {"id": "S3", "text": "Il pubblico ministero procede per truffa.", "certainty": "accertato"}]
CASE_TEXT = " ".join(f["text"] for f in FACTS)
CONTRATTO = {"article": "1439 c.c.", "rubrica": "Dolo", "text": ART_1439}
TRUFFA = {"article": "640 c.p.", "rubrica": "Truffa", "text": ART_640}


def _item(article, fit, perche, ambito="fattispecie", condizione="", cf=None, cn=None):
    """v79.15.0 (item 3): a judgment cites ids -- the facts (cf: «cita_fatti», a «no» / «dipende») and the article's
    numbered periods (cn: «cita_norma», applicability.article_ids; a «si»); None: no such key."""
    out = {"articolo": article, "fit": fit, "perche": perche, "condizione": condizione, "ambito": ambito}
    if cf is not None:
        out["cita_fatti"] = cf
    if cn is not None:
        out["cita_norma"] = cn
    return out


PM = "Il pubblico ministero procede per truffa"


def _judge(norm, item, facts=FACTS):
    res = ap.judge([norm], facts, "imputato", call=lambda m, k: json.dumps({"giudizi": [item]}, ensure_ascii=False))
    return res, res["judgments"][0]


# --- A. the verification by verdict type ------------------------------------------------------------------------------

def test_a_verified_no_cites_the_fact_it_rests_on_no_article_period_needed():
    res, rec = _judge(CONTRATTO, _item("1439 c.c.", "no", "«Il pubblico ministero procede per truffa» (S3).", cf=["S3"]))
    assert rec["verified"] and rec["fit"] == "no" and rec["passage"] == ""
    assert rec["fact_quotes"] == ["Il pubblico ministero procede per truffa."] and rec["facts"] == ["S3"]
    c = res["counters"]
    assert c["fit_no"] == 1 and c["fit_no_verified"] == 1 and c["fit_no_unverified_ids"] == 0
    # a period of the article beside the fact: its text the printed passage
    rec = _judge(CONTRATTO, _item("1439 c.c.", "no", "«Carla Tasso ha pagato 4.000 euro con un bonifico» (S2), non "
                                                     "«annullamento del contratto».", cf=["S2"], cn=["A1439cc.1"]))[1]
    assert rec["verified"] and rec["passage"].startswith("Il dolo è causa di annullamento del contratto")
    assert rec["facts"] == ["S2"] and rec["comma"] == "1"


@pytest.mark.parametrize("perche,ids,why", [
    # v79.15.0 (item 3): the facts by id in «cita_fatti» -- the perché is no citation
    ("Il contratto non è stato annullato (S1).", None, "cita_fatti mancante"),                    # no field at all
    ("La norma riguarda il «contratto» (S1).", [], "cita_fatti mancante"),                         # empty
    ("«Il pubblico ministero procede per truffa» (S3).", None, "cita_fatti mancante"),            # named in perché only
    ("Il fatto (S1) non riguarda la norma.", ["S9"], "cita_fatti: fatto inesistente: S9"),         # never given
    ("Il fatto (S2) non riguarda la norma.", ["S2", "S5"], "cita_fatti: fatto inesistente: S5"),   # one of them
    ("Il fatto (S1) non riguarda la norma.", ["s1"], "cita_fatti: fatto inesistente: s1"),         # as written
])
def test_an_unverified_no_keeps_the_norm_and_is_counted_as_an_id_failure(perche, ids, why):
    res, rec = _judge(CONTRATTO, _item("1439 c.c.", "no", perche, cf=ids))
    assert not rec["verified"] and rec["ids_failed"] and rec["reason"] == why, rec["reason"]
    c = res["counters"]
    assert c["fit_no"] == 0 and c["fit_no_verified"] == 0 and c["fit_no_unverified_ids"] == 1
    assert c["fit_unverified_ids"] == 1 and c["fit_unverified"] == 1 and ap.effects(res, [])["removed"] == []


def test_a_no_citing_a_fact_but_adding_a_rule_is_still_discarded():
    res, rec = _judge(CONTRATTO, _item("1439 c.c.", "no", "«Il pubblico ministero procede per truffa» (S3): la norma "
                                                          "vale solo per i contratti scritti.", cf=["S3"]))
    assert not rec["verified"] and rec["added_content"] and not rec["ids_failed"]
    assert res["counters"]["fit_added_content"] == 1 and res["counters"]["fit_no_unverified_ids"] == 0


def test_a_si_citing_no_period_of_the_article_stays_unverified():
    res, rec = _judge(TRUFFA, _item("640 c.p.", "si", "Bruno Gelso ha venduto un orologio placcato (S1, S2)."))
    assert not rec["verified"] and rec["reason"] == "cita_norma mancante" and rec["ids_failed"]
    # a fact id is no period of the article for a «si»
    rec = _judge(TRUFFA, _item("640 c.p.", "si", "Bruno Gelso ha venduto a Carla Tasso un orologio (S1).",
                               cn=["S1"]))[1]
    assert not rec["verified"] and rec["reason"] == "cita_norma: periodo non dell'articolo: S1"
    res, rec = _judge(TRUFFA, _item("640 c.p.", "si", "Bruno Gelso ha venduto a Carla Tasso un orologio (S1): «con artifizi "
                                                      "o raggiri».", cn=["A640cp.1"]))
    assert rec["verified"] and res["counters"]["fit_si"] == 1 and res["counters"]["fit_no_verified"] == 0
    assert rec["passage"] == ART_640


@pytest.mark.parametrize("perche", [
    # (review) a fact cited for show, the «no» resting on a claim in the article's words the case never states
    "«Il pubblico ministero procede per truffa» e i raggiri non sono stati usati da uno dei contraenti (S3).",
    "«Il pubblico ministero procede per truffa», non un contratto (S3).",
])
def test_a_no_whose_words_are_the_articles_uncited_not_the_facts_is_discarded(perche):
    res, rec = _judge(CONTRATTO, _item("1439 c.c.", "no", perche, cf=["S3"]))
    assert not rec["verified"] and rec["added_content"] and not rec["ids_failed"], rec["reason"]
    assert res["counters"]["fit_no_verified"] == 0 and ap.effects(res, [])["removed"] == []
    # the article's words with their period cited may carry the «no»; the rubrica's word may be said
    rec = _judge(CONTRATTO, _item("1439 c.c.", "no", "«Il pubblico ministero procede per truffa» (S3), non "
                                                     "«annullamento del contratto».", cf=["S3"], cn=["A1439cc.1"]))[1]
    assert rec["verified"] and "annullamento del contratto" in rec["passage"]
    rec = _judge(CONTRATTO, _item("1439 c.c.", "no", "«Il pubblico ministero procede per truffa» (S3), non dolo.",
                                  cf=["S3"]))[1]
    assert rec["verified"], rec["reason"]


def test_the_system_assumption_is_no_fact_to_cite():
    facts = FACTS + [{"id": "S4", "text": "Parte assistita assunta: Bruno Gelso", "certainty": "assunto"}]
    res, rec = _judge(CONTRATTO, _item("1439 c.c.", "no", "«Parte assistita assunta: Bruno Gelso» (S4).", cf=["S4"]),
                      facts)
    assert not rec["verified"] and rec["ids_failed"] and rec["reason"].startswith("cita_fatti: assunzione del sistema")
    assert res["counters"]["fit_no_unverified_ids"] == 1
    res = ap.judge([CONTRATTO], FACTS, "imputato", call=lambda m, k: json.dumps(
        {"giudizi": [_item("1439 c.c.", "no", "«Carla Tasso ha pagato 4.000 euro» (S2).", cf=["S2"])]}))
    assert res["counters"]["fit_no_verified"] == 1


def test_the_prompt_carries_the_id_fields_and_no_fact_quote():
    # v79.15.0 (item 3): «fatto_citato» and its line removed -- the schema's «cita_norma» and «cita_fatti» instead
    lines = ap.config()["prompt_system"]
    assert [x for x in lines if "fatto_citato" in x] == [] and [x for x in lines if x.startswith('- con "no"')] == []
    norma = [x for x in lines if x.startswith('- "cita_norma"')]
    fatti = [x for x in lines if x.startswith('- "cita_fatti"')]
    assert len(norma) == 1 and len(fatti) == 1
    assert lines[lines.index(norma[0]) - 1].startswith('- "perche": UNA sola frase')
    assert lines.index(fatti[0]) == lines.index(norma[0]) + 1
    assert '"cita_norma": ["..."]' in lines[-1] and '"cita_fatti": ["..."]' in lines[-1]
    assert len(lines) == 19


def _flow_state(text, sentences, records, topics=None):
    return {"records": records, "sentences": sentences, "text": text, "request": "", "intake": {}, "analysis": {},
            "topics": topics or [{"title": "Tema", "articles": list(records), "anchor": list(records)[0],
                                  "matched": list(records), "planned": len(records)}]}


def _row(number, code, text, rubrica):
    return {"_article": (number, code), "s": {"plain_text": text, "titolo_articolo": rubrica}}


def _req(i, text, status, supporting=()):
    return {"id": f"R{i}", "comma": 1, "kind": "requisito", "status": status, "supporting": list(supporting),
            "text": text}


IMPUTATO = {"rito": "penale", "family": "penale", "role": "imputato", "stage": "indagini",
            "entry": posture.entry_for("penale", "imputato", "indagini")}


def test_a_verified_no_removes_the_norm_logs_the_removal_and_counts_it(monkeypatch):
    monkeypatch.setenv("APPLICABILITY_FIT", "true")
    events = []
    from src.rag import verbose_logger
    monkeypatch.setattr(verbose_logger, "vlog", lambda event, data=None, ms=None: events.append((event, data)))
    sents = [dict(f, source="caso") for f in FACTS]
    records = {"640 c.p.": {"article": "640 c.p.", "rubrica": "Truffa", "side": "threat", "charged": 1,
                            "requisites": [_req(1, "con artifizi o raggiri", "da provare")]},
               "1439 c.c.": {"article": "1439 c.c.", "rubrica": "Dolo", "requisites": [
                   _req(2, "i raggiri usati da uno dei contraenti", "da provare")]}}
    state = _flow_state(CASE_TEXT, sents, records)
    rows = [_row("640", "c.p.", ART_640, "Truffa"), _row("1439", "c.c.", ART_1439, "Dolo")]
    reply = {"giudizi": [_item("640 c.p.", "si", "Bruno Gelso ha venduto a Carla Tasso un orologio (S1): «con artifizi o "
                                                 "raggiri».", cn=["A640cp.1"]),
                         _item("1439 c.c.", "no", "«Il pubblico ministero procede per truffa» (S3).", cf=["S3"])]}
    monkeypatch.setattr(ap, "_default_call", lambda m, k: json.dumps(reply, ensure_ascii=False))
    removed = flow.applicability_fit(state, IMPUTATO, rows, set(), [])
    assert removed == {"1439 c.c."} and "1439 c.c." not in state["records"] and "640 c.p." in state["records"]
    c = state["block_counters"]
    assert c["fit_removed"] == c["norms_removed_by_fit"] == 1 and c["fit_no_verified"] == 1
    assert c["fit_no_unverified_ids"] == 0
    discards = [d for e, d in events if e == "applicability_fit_discard"]
    assert discards == [{"article": "1439 c.c.", "reason": "removed: fit no (verified)", "fit": "no",
                         "perche": "«Il pubblico ministero procede per truffa» (S3)."}]
    assert ap.REMOVED_REASON == "removed: fit no (verified)"
    logged = next(d for e, d in events if e == "interactive_applicability_effects")["counters"]
    assert {k: logged[k] for k in ("fit_no_verified", "fit_no_unverified_ids", "fit_removed")} == \
        {"fit_no_verified": 1, "fit_no_unverified_ids": 0, "fit_removed": 1}
    fit_line = next(d for e, d in events if e == "applicability_fit")["counters"]
    assert fit_line["fit_no_verified"] == 1 and fit_line["fit_no_unverified_ids"] == 0


def test_a_kept_named_no_quoting_only_facts_prints_the_facts_not_an_empty_passage():
    item = {"article": "1439 c.c.", "passage": "", "fact_quotes": ["Il pubblico ministero procede per truffa"]}
    assert ap.contest_pro(item) == ("Si fonda sui fatti narrati («Il pubblico ministero procede per truffa») letti sul "
                                    "testo della norma")
    assert ap.objection_answer(item) == ("Il punto si decide sui fatti narrati («Il pubblico ministero procede per "
                                         "truffa») e su quelli che restano da provare")
    old = {"passage": "rateale"}
    assert ap.contest_pro(old) == "Si fonda sul testo della norma («rateale») letto sui fatti narrati"
    assert ap.objection_answer(old).startswith("Il punto si decide sul testo della norma («rateale»)")


# --- B. the frame binds the charge set ----------------------------------------------------------------------------------

PENAL_TEXT = ("Assisto Bruno Gelso, indagato per truffa. Bruno Gelso ha venduto a Carla Tasso un orologio dichiarandolo "
              "d'oro. L'orologio era placcato e Carla Tasso ha pagato 4.000 euro con un bonifico. Il pubblico ministero "
              "procede per truffa e le indagini preliminari sono in corso. Carla Tasso chiede anche il risarcimento del "
              "danno. Quali rischi corre Bruno Gelso?")
CIVIL_TEXT = ("Assisto Dora Pesco, convenuta da Elio Rovo. Elio Rovo chiede a Dora Pesco il pagamento di 6.000 euro per "
              "una fornitura di legname non pagata. Elio Rovo sostiene che Dora Pesco lo abbia raggirato. Il giudizio "
              "civile è in primo grado. Dora Pesco deve pagare?")


def _penal_reply(**over):
    reply = {"parti": [{"nome": "Bruno Gelso", "ruolo_nel_fatto": "venditore", "lato": "cliente"},
                       {"nome": "Carla Tasso", "ruolo_nel_fatto": "acquirente", "lato": "controparte"}],
             "cliente": {"nome": "Bruno Gelso", "perche": "«Assisto Bruno Gelso, indagato per truffa»"},
             "ruolo": {"valore": "indagato", "perche": "«Assisto Bruno Gelso, indagato per truffa»"},
             "rito": {"valore": "penale", "perche": "«Il pubblico ministero procede per truffa»"},
             "stadio": {"valore": "indagini", "perche": "«le indagini preliminari sono in corso»"},
             "addebito_o_domanda": {"articolo": None, "descrizione": "truffa",
                                    "frase": "«Il pubblico ministero procede per truffa»"},
             "istituto": None, "quesiti": ["Quali rischi corre Bruno Gelso?"]}
    reply.update(over)
    return reply


def _civil_reply(**over):
    reply = {"parti": [{"nome": "Dora Pesco", "ruolo_nel_fatto": "acquirente", "lato": "cliente"},
                       {"nome": "Elio Rovo", "ruolo_nel_fatto": "fornitore", "lato": "controparte"}],
             "cliente": {"nome": "Dora Pesco", "perche": "«Assisto Dora Pesco, convenuta da Elio Rovo»"},
             "ruolo": {"valore": "convenuto", "perche": "«Assisto Dora Pesco, convenuta da Elio Rovo»"},
             "rito": {"valore": "civile", "perche": "«Il giudizio civile è in primo grado»"},
             "stadio": {"valore": "primo_grado", "perche": "«Il giudizio civile è in primo grado»"},
             "addebito_o_domanda": {"articolo": None, "descrizione": "pagamento di 6.000 euro",
                                    "frase": "«Elio Rovo chiede a Dora Pesco il pagamento di 6.000 euro»"},
             "istituto": None, "quesiti": ["Dora Pesco deve pagare?"]}
    reply.update(over)
    return reply


def _sentences(text):
    parts = [p.strip() for p in text.replace("?", "?.").split(". ") if p.strip() and not p.strip().endswith("?.")]
    return [{"id": f"S{i}", "text": p.rstrip(".") + ".", "source": "caso", "certainty": "accertato"}
            for i, p in enumerate(parts, 1)]


def _method(monkeypatch, text, reply, records, posture_, client, frame_on=True, extra=None):
    """The answer stage from the records on, the frame held as intake leaves it: prepare, the thesis call (faked),
    compose -- {"ctx", "plan", "answer", "state", "events"}."""
    if frame_on:
        monkeypatch.setenv("CASE_FRAME", "true")
    else:
        monkeypatch.delenv("CASE_FRAME", raising=False)
    events = []
    real = M._log
    monkeypatch.setattr(M, "_log", lambda event, payload, ms=None: events.append((event, payload)) or real(
        event, payload, ms))
    state = _flow_state(text, _sentences(text), records)
    state["case_frame"] = cf.verify(reply, text) if reply is not None else None
    state.update(extra or {})
    by_id = {s["id"]: s for s in state["sentences"]}
    for rec in records.values():
        for r in rec.get("requisites") or []:
            IR.settle(r, by_id)
    ctx = M.prepare(state, posture_, [], text, client, "", list(records), [], [], [], [], link_call=lambda *a: {},
                    case_id_="v79142")
    ctx["asked"] = []

    def call(messages):
        return {"valutazioni": {c["id"]: {"pro": "p", "contro": "c", "argomento": "a", "realismo": "medio"}
                                for c in ctx["candidates"]},
                "obiezioni": [{"obiezione": "L'altra parte insisterà sui fatti.", "risposta": "Vanno provati."}]}
    plan = M.method_plan(ctx, call=call)
    answer = M.compose([("Norme applicate ai fatti", "\n\n".join(f"L'art. {x} riguarda i fatti narrati."
                                                                 for x in records))], ctx, plan, log={})
    return {"ctx": ctx, "plan": plan, "answer": answer, "state": state, "events": events}


def _section(answer, key):
    return dict(M.answer_sections(answer)).get(M.title(key)) or ""


def _penal_records(with_cp=True):
    out = {}
    if with_cp:
        out["640 c.p."] = {"article": "640 c.p.", "rubrica": "Truffa", "side": "threat", "charged": 1, "requisites": [
            _req(1, "con artifizi o raggiri", "provato", ["S2"]),
            _req(2, "inducendo taluno in errore", "da provare"),
            _req(3, "procura a sé un ingiusto profitto con altrui danno", "da provare")]}
    out["2043 c.c."] = {"article": "2043 c.c.", "rubrica": "Risarcimento per fatto illecito", "side": "threat",
                        "relied_by": "Carla Tasso", "requisites": [
                            _req(4, "fatto doloso o colposo", "da provare"),
                            _req(5, "danno ingiusto", "provato", ["S3"])]}
    return out


def _arts(ctx):
    owner = {r["id"]: x for x, rec in ctx["records"].items() for r in rec.get("requisites") or []}
    out = set()
    for c in ctx["candidates"]:
        out |= {c["article"]} if c.get("article") else {owner.get(i) for i in c.get("contesta") or []}
    return out - {None, ""}


def test_penal_frame_with_an_indagato_excludes_a_civil_code_candidate_from_the_charge(monkeypatch):
    got = _method(monkeypatch, PENAL_TEXT, _penal_reply(), _penal_records(), IMPUTATO, "Bruno Gelso")
    ctx, answer = got["ctx"], got["answer"]
    assert M.binding_mode(got["state"]) == "penale"
    b = ctx["frame_binding"]
    assert b["applied"] and b["excluded"] == ["2043 c.c."] and b["kept"] == ["640 c.p."] and not b["empty"]
    assert "2043 c.c." not in _arts(ctx) and "640 c.p." in _arts(ctx)
    assert ctx["assessment"]["article"] == "640 c.p."
    assert "2043 c.c." in ctx["records"]                                  # marked, never printed
    schede = M.records_block(ctx["records"])
    # v79.14.3 (item 4): excluded means excluded -- no scheda, no §4 table, no «Fatti mancanti» row
    assert "2043 c.c." not in schede and "2043" not in _section(answer, "norme")
    assert "2043" not in _section(answer, "fatti")
    assert "art. 640 c.p. — Truffa [addebito o domanda contro il cliente]" in schede
    esito = next(x for x in answer.splitlines() if "**Esito probabile**" in x)
    assert "640 c.p." in esito and "2043" not in esito
    assert "2043" not in _section(answer, "soluzioni")
    c = got["state"]["block_counters"]
    assert c["frame_binding_applied"] == 1 and c["frame_binding_excluded"] == ["2043 c.c."]
    assert c["frame_binding_empty"] == 0
    plan_log = next(p for e, p in got["events"] if e == "plan")
    assert plan_log["frame_binding"]["excluded"] == ["2043 c.c."]
    assert "2043 c.c." not in plan_log["candidate_articles"]
    assert next(p for e, p in got["events"] if e == "frame_binding")["mode"] == "penale"


def test_a_kept_named_civil_contest_is_never_offered_under_a_penal_binding(monkeypatch):
    # (review) 79.13.0 keeps a charged or named norm the judgment says «no» to as the defence's contest; under a penal
    # binding that norm is context and its contest is not offered (the brief: never «Linea principale») -- the norm
    # itself stays in the records; with the frame off the 79.13.0 contest is offered as before
    monkeypatch.setenv("APPLICABILITY_FIT", "true")
    fit = {"applicability": {"kept_named": [{"article": "2043 c.c.", "fit": "no", "passage": "",
                                             "perche": "«Il pubblico ministero procede per truffa» (S3).",
                                             "fact_quotes": ["Il pubblico ministero procede per truffa"],
                                             "facts": ["S3"]}],
                             "charged": ["2043 c.c."], "facts": []}}
    off = _method(monkeypatch, PENAL_TEXT, _penal_reply(), _penal_records(), IMPUTATO, "Bruno Gelso",
                  frame_on=False, extra=fit)
    assert ("contest_fit", "2043 c.c.") in [(c["kind"], c.get("article")) for c in off["ctx"]["candidates"]]
    on = _method(monkeypatch, PENAL_TEXT, _penal_reply(), _penal_records(), IMPUTATO, "Bruno Gelso", extra=fit)
    assert "contest_fit" not in [c["kind"] for c in on["ctx"]["candidates"]]
    assert "2043 c.c." in on["ctx"]["records"] and on["ctx"]["frame_binding"]["excluded"] == ["2043 c.c."]
    assert "2043" not in next(x for x in on["answer"].splitlines() if "**Esito probabile**" in x)


def test_civil_frame_excludes_a_penal_code_article_from_the_claim(monkeypatch):
    records = {"1218 c.c.": {"article": "1218 c.c.", "rubrica": "Responsabilità del debitore", "side": "threat",
                             "charged": 1, "requisites": [_req(1, "il debitore non esegue esattamente la prestazione",
                                                               "provato", ["S2"])]},
               "640 c.p.": {"article": "640 c.p.", "rubrica": "Truffa", "side": "threat", "relied_by": "Elio Rovo",
                            "requisites": [_req(2, "con artifizi o raggiri", "da provare")]}}
    convenuto = {"rito": "civile", "family": "civile", "role": "convenuto", "stage": "primo_grado",
                 "entry": posture.entry_for("civile", "convenuto", "primo_grado")}
    got = _method(monkeypatch, CIVIL_TEXT, _civil_reply(), records, convenuto, "Dora Pesco")
    ctx = got["ctx"]
    assert ctx["frame_binding"]["mode"] == "civile" and ctx["frame_binding"]["excluded"] == ["640 c.p."]
    assert "640 c.p." not in _arts(ctx) and ctx["assessment"]["article"] == "1218 c.c."
    assert "640" not in next(x for x in got["answer"].splitlines() if "**Esito probabile**" in x)
    assert M.civil_excluded("640 c.p.") and not M.civil_excluded("1218 c.c.") and not M.civil_excluded("405 c.p.p.")


@pytest.mark.parametrize("over", [
    {"ruolo": {"valore": "indagato", "perche": "«Bruno Gelso è stato arrestato»"}},     # its quote not in the text
    {"ruolo": None},                                                                     # no role
    {"rito": None, "stadio": None},                                                      # no rito, no penal stage
    {"ruolo": {"valore": "sconosciuto", "perche": "«Assisto Bruno Gelso»"}},           # the honesty line's role
])
def test_a_frame_not_verified_on_the_rito_or_the_role_binds_nothing(monkeypatch, over):
    got = _method(monkeypatch, PENAL_TEXT, _penal_reply(**over), _penal_records(), IMPUTATO, "Bruno Gelso")
    ctx = got["ctx"]
    assert M.binding_mode(got["state"]) == "" and not ctx["frame_binding"]["applied"]
    assert not any(M.frame_excluded(r) for r in ctx["records"].values())
    assert "frame_binding_applied" not in (got["state"].get("block_counters") or {})
    off = _method(monkeypatch, PENAL_TEXT, _penal_reply(), _penal_records(), IMPUTATO, "Bruno Gelso", frame_on=False)
    assert not off["ctx"]["frame_binding"]["applied"]
    assert _arts(ctx) == _arts(off["ctx"]) and "2043 c.c." in _arts(off["ctx"])      # the 79.14.1 path


def test_a_stage_of_investigation_binds_when_no_rito_is_verified(monkeypatch):
    got = _method(monkeypatch, PENAL_TEXT, _penal_reply(rito=None), _penal_records(), IMPUTATO, "Bruno Gelso")
    assert got["ctx"]["frame_binding"]["mode"] == "penale" and got["ctx"]["frame_binding"]["excluded"] == ["2043 c.c."]


def test_a_charge_set_the_binding_empties_takes_the_corpus_line_never_the_excluded(monkeypatch):
    got = _method(monkeypatch, PENAL_TEXT, _penal_reply(), _penal_records(with_cp=False), IMPUTATO, "Bruno Gelso")
    ctx, answer = got["ctx"], got["answer"]
    b = ctx["frame_binding"]
    assert b["empty"] and b["excluded"] == ["2043 c.c."] and b["kept"] == [] and b["materia"] == "truffa"
    assert ctx["assessment"]["article"] == ""
    assert [c["kind"] for c in ctx["candidates"]] == ["contenimento"]        # no line on the excluded article
    assert not ctx["candidates"][0]["tesi"].startswith("Ammettere")          # v79.14.3 (item 3): never under a frame
    summary = _section(answer, "sintesi")
    esito = next(x for x in answer.splitlines() if "**Esito probabile**" in x)
    assert "2043" not in esito
    c = got["state"]["block_counters"]
    assert c["frame_binding_empty"] == 1 and c["frame_binding_excluded"] == ["2043 c.c."]
    # v79.14.3 (item 3): «truffa» is a code institution (640 c.p.) -- the corpus line never prints on it (79.14.2
    # printed it and counted corpus_line_code 1); nor is it the forecast's why
    assert b["charge_in_corpus"] and "corpus non contiene" not in summary and "corpus non contiene" not in esito
    assert ctx["corpus_line_code"] == 0 == len(M.corpus_line_on_code(summary, ctx))
    assert "corpus" not in str(ctx["assessment"].get("why") or "")
    M.refresh_after_elements(ctx)
    assert "corpus" not in str(ctx["assessment"].get("why") or "")
    # the frame names no charge: the data's words
    reply = _penal_reply(addebito_o_domanda=None)
    assert M.binding_materia({"case_frame": cf.verify(reply, PENAL_TEXT)}, "penale") == "questo addebito"


def test_penal_sources_are_the_codes_the_penal_provisions_and_the_flagged_laws():
    for label in ("640 c.p.", "582 c.p. co. 2", "405 c.p.p.", "2621 c.c.", "73 d.P.R. 309/1990",
                  "25 d.lgs. 231/2001"):
        assert M.penal_source(label), label
    for label in ("1439 c.c.", "2043 c.c.", "5 d.P.R. 309/1990", "117 d.lgs. 385/1993", "10 l. 99/2099"):
        assert not M.penal_source(label), label
    flagged = [e["name"] for e in M.institution_laws() if e.get("penal")]
    assert "procedura penale" in flagged and "stupefacenti" in flagged and "obbligazioni e contratti" not in flagged
