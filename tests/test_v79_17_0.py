"""v79.17.0 -- the pretesa anchors every rito (case_frame: item 1), the frame asks instead of guessing (interactive_flow
.ask_side), the fit judges against the pretesa and the quesiti (applicability «governa»: item 3), the binding on the
pretesa (interactive_flow.binding_pretesa, analysis_step.governs_sets / line_tier / _verify_line: item 2), the prose
from verified material with one rule (verified_material: item 4), the sections from one object (analysis_step
.attach_object, summary_text, question_bridge, risk_word, verify_points, allowed_ruling_articles; lawyer_method
.objections_section: item 5), the law sets' families (data/institution_laws.json). Synthetic texts and replies only."""
import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import test_v79_14_case_frame as CF  # noqa: E402
import test_v79_15_0_analysis as T  # noqa: E402
import test_v79_16_1 as X  # noqa: E402
from src.rag import analysis_step as A  # noqa: E402
from src.rag import applicability as ap  # noqa: E402
from src.rag import case_frame as cf  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import lawyer_method as LM  # noqa: E402
from src.rag import verified_material as VM  # noqa: E402

PRETESA_REPLY = dict(CF._debtor_reply(), pretesa={"tipo": "domanda", "contenuto": "pagamento di una penale",
                                                   "contro": "Orsa S.r.l.", "articolo": None,
                                                   "frase": "«Delfino pretende ora il pagamento di una penale»"})
PRETESA_REPLY.pop("addebito_o_domanda", None)


@pytest.fixture
def frame_on(monkeypatch):
    monkeypatch.setenv("CASE_FRAME", "true")


# --- 1. the frame: the pretesa of every rito ------------------------------------------------------------------------------

def test_the_prompt_asks_the_pretesa_and_its_hash_moved():
    text = cf.prompt_system()
    assert '"pretesa": {"tipo": "accusa | domanda | ricorso | impugnazione | opposizione | istanza"' in text
    assert "- pretesa: che cosa è chiesto o contestato nel caso, in ogni rito" in text and '"addebito_o_domanda"' not in text
    assert cf.prompt_hash() != "01439962c7c9"           # 79.16.x's frame prompt; the new hash is the report's


def test_a_pretesa_is_verified_by_its_frase_tipo_contenuto_and_contro():
    res = CF._read(CF.DEBTOR, PRETESA_REPLY)
    p = res["frame"]["addebito_o_domanda"]
    assert "addebito_o_domanda" in res["verified"] and res["counters"]["frame_pretesa_verified"] == 1
    assert res["counters"]["frame_pretesa_derived"] == 0 and not res.get("derived")
    assert p["tipo"] == "domanda" and p["contenuto"] == "pagamento di una penale" == p["descrizione"]
    assert p["contro"] == "Orsa S.r.l." and p["rito"] == "civile" and p["fonte"] == cf.FONTE_TESTO and p["derivata"] is False
    assert cf.pretesa(res) == p and cf.pretesa_text(res) == "domanda: «pagamento di una penale» contro Orsa S.r.l."
    assert cf.pretesa_line(res) == "Pretesa: domanda — «pagamento di una penale», contro Orsa S.r.l."
    # an off-schema tipo: derived from the text (a resisting client, no opposizione word: «domanda»), logged, kept
    res = CF._read(CF.DEBTOR, dict(PRETESA_REPLY, pretesa=dict(PRETESA_REPLY["pretesa"], tipo="richiesta di pagamento")))
    assert res["frame"]["addebito_o_domanda"]["tipo"] == "domanda" and "addebito_o_domanda" in res["verified"]
    assert any(e["field"] == "addebito_o_domanda.tipo" for e in res["log"])
    # a contro the text does not name: dropped, the pretesa stands
    res = CF._read(CF.DEBTOR, dict(PRETESA_REPLY, pretesa=dict(PRETESA_REPLY["pretesa"], contro="Gabbiano S.r.l.")))
    assert res["frame"]["addebito_o_domanda"]["contro"] == "" and "addebito_o_domanda" in res["verified"]
    # a frase not in the text: the whole pretesa unverified, then derived from the quesiti
    res = CF._read(CF.DEBTOR, dict(PRETESA_REPLY, pretesa=dict(PRETESA_REPLY["pretesa"], frase="«parole inventate»")))
    assert "addebito_o_domanda" not in res["verified"] and res["counters"]["frame_unverified.addebito_o_domanda"] == 1
    assert res["frame"]["addebito_o_domanda"]["derivata"] is True and res["derived"] == ["addebito_o_domanda"]


def test_the_older_addebito_key_is_still_read_as_a_pretesa():
    res = CF._read(CF.DEBTOR, CF._debtor_reply())
    p = res["frame"]["addebito_o_domanda"]
    assert "addebito_o_domanda" in res["verified"] and p["tipo"] == "domanda" and p["fonte"] == cf.FONTE_TESTO
    assert p["descrizione"] == p["contenuto"]


def test_the_pretesa_is_derived_from_the_quesiti_then_from_the_last_request():
    reply = {k: v for k, v in CF._debtor_reply().items() if k != "addebito_o_domanda"}
    res = CF._read(CF.DEBTOR, reply)
    p = res["frame"]["addebito_o_domanda"]
    assert p["derivata"] is True and p["fonte"] == cf.FONTE_QUESITI and p["contenuto"] == reply["quesiti"][0]
    assert p["descrizione"] == "" and p["contro"] == "Delfino S.p.A." and p["tipo"] == "domanda"
    assert res["derived"] == ["addebito_o_domanda"] and "addebito_o_domanda" not in res["verified"]
    assert res["counters"]["frame_pretesa_derived"] == 1 and res["counters"]["frame_pretesa_verified"] == 0
    assert res["counters"]["frame_fields_verified"] == len(res["verified"])
    assert cf.pretesa_line(res).startswith(f"Pretesa dedotta dai quesiti: «{reply['quesiti'][0].rstrip('?')}».\n\nPretesa: domanda — «")
    assert "Pretesa (dedotta dai quesiti): domanda: «" in cf.fit_context(res) and "Addebito o domanda" not in cf.fit_context(res)
    # no quesiti either: the text's last request
    res = CF._read(CF.DEBTOR, dict(reply, quesiti=[]))
    p = res["frame"]["addebito_o_domanda"]
    assert p["derivata"] and p["fonte"] == cf.FONTE_LAST and p["contenuto"].startswith("La penale è riducibile")
    assert cf.pretesa_line(res).startswith("Pretesa dedotta dall'ultima richiesta: «La penale è riducibile».")
    # a failed frame derives nothing (every field to the rules)
    failed = cf.verify(None, CF.DEBTOR)
    assert failed["frame"]["addebito_o_domanda"] is None and not failed.get("derived")


def test_derive_tipo_follows_the_text_then_the_rito():
    assert cf.derive_tipo("civile", "convenuto", "Alfa intende proporre opposizione al decreto ingiuntivo") == "opposizione"
    assert cf.derive_tipo("lavoro", "attore", "Il lavoratore impugna il licenziamento") == "impugnazione"
    assert cf.derive_tipo("penale", "imputato", "La Procura procede per truffa") == "accusa"
    assert cf.derive_tipo("lavoro", "attore", "Chiede il pagamento delle differenze retributive") == "ricorso"
    assert cf.derive_tipo("amministrativo", "", "Il Comune ha annullato il permesso") == "ricorso"
    assert cf.derive_tipo("civile", "ricorrente", "x") == "ricorso"
    assert cf.derive_tipo("civile", "attore", "Chiede un sequestro conservativo in via cautelare") == "istanza"
    assert cf.derive_tipo("civile", "attore", "Chiede la risoluzione del contratto") == "domanda"
    assert cf.canon_tipo("Opposizione a decreto ingiuntivo") == "opposizione" and cf.canon_tipo("reato") == "accusa"
    assert cf.canon_tipo("qualcosa") is None


def test_must_ask_only_when_a_field_is_missing_and_the_text_names_no_side():
    text = ("Delfino S.p.A. pretende da Orsa S.r.l. il pagamento di una penale di 18.000 euro per un ritardo nella "
            "consegna. Orsa sostiene di aver consegnato nei termini. La penale è riducibile?")
    reply = {"parti": [{"nome": "Orsa S.r.l.", "ruolo_nel_fatto": "consegnato", "lato": "terzo"},
                       {"nome": "Delfino S.p.A.", "ruolo_nel_fatto": "pretende il pagamento di una penale", "lato": "terzo"}],
             "cliente": None, "ruolo": None, "rito": {"valore": "civile", "perche": "«penale»"}, "stadio": None,
             "pretesa": None, "istituto": None, "quesiti": ["La penale è riducibile?"]}
    res = CF._read(text, reply)
    assert cf.must_ask(res, text) == ["cliente", "ruolo", "pretesa"]      # a derived pretesa counts as missing
    assert cf.side_named(res, text) is False and cf.ask_side_question() == "Per chi è reso il parere e che cosa chiede?"
    assert cf.must_ask(res, "Assisto Orsa S.r.l. " + text) == []          # the text names the side: no question
    assert cf.must_ask(res, "Il mio cliente, convenuto, " + text) == []
    res = CF._read(CF.DEBTOR, PRETESA_REPLY)                                # everything verified: nothing asked
    assert cf.must_ask(res, CF.DEBTOR) == []
    assert cf.must_ask(cf.verify(None, text), text) == []                   # a failed frame: the rules' path
    assert cf.must_ask(None, text) == []


def test_the_flow_asks_its_one_question_and_the_next_message_answers(monkeypatch, frame_on):
    text = ("Delfino S.p.A. pretende da Orsa S.r.l. il pagamento di una penale di 18.000 euro per un ritardo nella "
            "consegna. La penale è riducibile?")
    reply = {"parti": [], "cliente": None, "ruolo": None, "rito": {"valore": "civile", "perche": "«penale»"},
             "stadio": None, "pretesa": None, "istituto": None, "quesiti": ["La penale è riducibile?"]}
    held = CF._read(text, reply)
    state = {"case_frame": held, "text": text, "request": "", "topics": [], "block_counters": {}}
    got = flow.ask_side(state, text, "")
    assert got and got["kind"] == "ask_side" and got["answer"] == cf.ask_side_question() and got["asked"] == ["cliente", "ruolo", "pretesa"]
    assert state["state"] == flow.ASKED_SIDE and state["block_counters"]["frame_asked_side"] == 1
    assert state["asked_side"]["message"] == text
    # the next message: follow_up hands it to answer_after_ask (the case run again as one text)
    seen = []
    monkeypatch.setattr(flow, "answer_after_ask", lambda message, st: seen.append((message, st)) or {"kind": "answer"})
    assert flow.follow_up("Assisto Orsa S.r.l., chiede la riduzione della penale.", state) == {"kind": "answer"}
    assert seen and seen[0][0].startswith("Assisto Orsa")
    # a frame that holds the side: nothing asked
    assert flow.ask_side({"case_frame": CF._read(CF.DEBTOR, PRETESA_REPLY), "text": CF.DEBTOR}, CF.DEBTOR) is None
    assert flow.ask_side({"text": text}, text) is None


# --- 3. the fit judges against the pretesa and the quesiti -------------------------------------------------------------------

NORM = {"article": "1382 c.c.", "rubrica": "Effetti della clausola penale",
        "text": "La clausola penale ha l'effetto di limitare il risarcimento alla prestazione promessa. La penale è dovuta "
                "indipendentemente dalla prova del danno."}
FACTS = [{"id": "S1", "text": "Delfino pretende il pagamento di una penale.", "certainty": "accertato"},
         {"id": "S2", "text": "Orsa ha consegnato nei termini.", "certainty": "allegato"}]


def test_the_fit_prompt_carries_the_pretesa_and_the_numbered_quesiti_and_its_hash_moved():
    assert '"governa"' in ap.prompt_system() and ap.prompt_hash() == "c60a2f78634c"
    plain = ap.build_messages([NORM], FACTS, "convenuto")
    asked = ap.build_messages([NORM], FACTS, "convenuto", "", "domanda: «pagamento di una penale»", ["La penale è riducibile?"])
    human = asked[1].content
    assert human.startswith("PRETESA (ciò che è chiesto o contestato nel caso):\ndomanda: «pagamento di una penale»\n\nQUESITI:\nQ1 La penale è riducibile?\n\n")
    assert human.endswith(plain[1].content) and asked[0].content == plain[0].content
    assert ap.build_messages([NORM], FACTS, "convenuto")[1].content == plain[1].content
    assert ap.pretesa_quesiti_block("", []) == ""


def test_governs_tokens_are_read_by_schema_and_an_absent_governa_says_nothing():
    assert ap.governs_of(["pretesa", "Q1", "q3", 2, "Q9", "altro"], 3) == (["pretesa", "Q1", "Q3", "Q2"], ["Q9", "altro"])
    assert ap.governs_of([], 3) == ([], []) and ap.governs_of("pretesa", 0) == (["pretesa"], [])
    assert ap.governs_of(None, 3) == (None, [])
    item = {"articolo": "1382 c.c.", "fit": "si", "perche": "x", "cita_norma": ["A1382cc.1"], "cita_fatti": ["S1"],
            "condizione": "", "ambito": "fattispecie", "governa": ["pretesa", "Q1"]}
    rec = ap.verify(item, NORM, FACTS, quesiti_n=1, governs_asked=True)
    assert rec["verified"] and rec["governs"] == ["pretesa", "Q1"] and rec["governs_asked"] is True
    rec = ap.verify({k: v for k, v in item.items() if k != "governa"}, NORM, FACTS, quesiti_n=1, governs_asked=True)
    assert rec["verified"] and rec["governs"] is None
    rec = ap.verify(item, NORM, FACTS)                      # the fit not asked: nothing recorded
    assert rec["governs"] is None and rec["governs_asked"] is False


def test_the_judge_counts_what_the_verified_norms_govern():
    norms = [NORM, dict(NORM, article="1384 c.c.", rubrica="Riduzione della penale",
                        text="La penale può essere diminuita equamente dal giudice. Il giudice valuta l'interesse."),
             dict(NORM, article="1218 c.c.", rubrica="Responsabilità del debitore", text="Il debitore è tenuto al risarcimento.")]

    def call(messages, kwargs):
        assert "PRETESA" in messages[1].content and "Q1 La penale è riducibile?" in messages[1].content
        return json.dumps({"giudizi": [
            {"articolo": "1382 c.c.", "fit": "si", "perche": "x", "cita_norma": ["A1382cc.1"], "cita_fatti": ["S1"],
             "condizione": "", "ambito": "fattispecie", "governa": ["pretesa"]},
            {"articolo": "1384 c.c.", "fit": "si", "perche": "y", "cita_norma": ["A1384cc.1"], "cita_fatti": ["S1"],
             "condizione": "", "ambito": "rimedio", "governa": ["Q1"]},
            {"articolo": "1218 c.c.", "fit": "si", "perche": "z", "cita_norma": ["A1218cc.1"], "cita_fatti": ["S1"],
             "condizione": "", "ambito": "fattispecie", "governa": []}]})
    res = ap.judge(norms, FACTS, "convenuto", call=call, pretesa="domanda: «pagamento di una penale»",
                   quesiti=["La penale è riducibile?"])
    c = res["counters"]
    assert res["governs_asked"] is True and res["quesiti_n"] == 1
    assert c["fit_si_verified"] == 3 and c["fit_governs_pretesa"] == 1 and c["fit_governs_quesiti"] == 1 and c["fit_context_only"] == 1
    assert res["by_article"]["1218 c.c."]["governs"] == [] and res["by_article"]["1384 c.c."]["governs"] == ["Q1"]
    assert ap.effects(res, [])["removed"] == [] and all("governs" in e for e in ap.effects(res, []).get("open") or [])
    plain = ap.judge(norms, FACTS, "convenuto", call=lambda m, k: call(m, k) if "PRETESA" in m[1].content else json.dumps(
        {"giudizi": [{"articolo": "1382 c.c.", "fit": "si", "perche": "x", "cita_norma": ["A1382cc.1"], "cita_fatti": ["S1"],
                      "condizione": "", "ambito": "fattispecie"}]}))
    assert plain["governs_asked"] is False and plain["counters"]["fit_governs_pretesa"] == 0
    for name in ("fit_governs_pretesa", "fit_governs_quesiti", "fit_context_only"):
        assert name in ap.COUNTERS and name in ap.FLOW_COUNTERS


# --- 2. the binding on the pretesa --------------------------------------------------------------------------------------------

def _fit_state(governs):
    judgments = [{"article": a, "fit": "si", "verified": True, "governs": g, "cita_norma": [f"A{ap.article_token(a)}.1"],
                  "cita_fatti": ["S1"], "substantive": not A.is_remedy(a)} for a, g in governs.items()]   # v79.17.3
    return {"applicability": {"judgments": judgments, "governs_asked": True, "quesiti": ["Q uno?", "Q due?"]},
            "block_counters": {}}


def test_binding_pretesa_reads_the_sets_from_the_verdicts_and_counts():
    state = _fit_state({"640 c.p.": ["pretesa"], "641 c.p.": ["Q2"], "62-bis c.p.": [], "646 c.p.": None})
    b = flow.binding_pretesa(state)
    # v79.17.3 (item 1): every substantive verified «si» is in the set, the tag first; 62-bis c.p. (remedy) stays out
    assert b["applied"] is True and b["principale"] == ["640 c.p.", "641 c.p.", "646 c.p."] and b["quesiti"] == {"Q2": ["641 c.p."]}
    assert b["reasons"] == {"640 c.p.": ["tag", "fit_si"], "641 c.p.": ["fit_si"], "646 c.p.": ["fit_si"]}
    assert b["context_only"] == ["62-bis c.p."] and b["asked"] is True
    assert state["block_counters"]["binding_pretesa_applied"] == 1
    assert state["block_counters"]["binding_pretesa_excluded"] == ["62-bis c.p."]
    state = _fit_state({"640 c.p.": [], "641 c.p.": ["Q1"]})           # no tag: the verified «si» still make the set
    b = flow.binding_pretesa(state)
    assert b["applied"] is True and b["principale"] == ["640 c.p.", "641 c.p."]
    assert state["block_counters"]["binding_pretesa_applied"] == 1 and state["block_counters"]["binding_pretesa_excluded"] == []
    assert flow.binding_pretesa({"applicability": {"judgments": [], "governs_asked": False}})["asked"] is False


def test_the_quesiti_articles_are_filtered_by_the_verdict():
    state = {"quesiti_retrieval": {"by_quesito": {"1": ["2946 c.c.", "1385 c.c."], "2": ["74 c.p.p."]}},
             "analysis": {"quesiti_queries": ["Entro quale termine?", "Parte civile?"]},
             "records": {"2946 c.c.": {}, "1385 c.c.": {}, "74 c.p.p.": {}},
             "applicability": {"governs_asked": True, "judgments": [
                 {"article": "2946 c.c.", "fit": "si", "verified": True, "governs": ["Q1"]},
                 {"article": "1385 c.c.", "fit": "si", "verified": True, "governs": ["pretesa"]},
                 {"article": "74 c.p.p.", "fit": "si", "verified": True, "governs": []}]}}
    assert flow.governs_by_label(state) == {"2946 c.c.": ["Q1"], "1385 c.c.": ["pretesa"], "74 c.p.p.": []}
    assert flow.quesiti_articles(state) == {"Entro quale termine?": ["2946 c.c.", "1385 c.c."]}   # 74 c.p.p. answers no Q
    block = flow.quesito_articles_block(state, "Entro quale termine?")
    assert "- art. 2946 c.c. [governa: Q1]" in block and "- art. 1385 c.c. [governa: pretesa]" in block
    state["applicability"]["governs_asked"] = False                     # the fit not asked: 79.16.0's mapping
    assert flow.quesiti_articles(state) == {"Entro quale termine?": ["2946 c.c.", "1385 c.c."], "Parte civile?": ["74 c.p.p."]}


def _penal_governs(governs, fits=None):
    fits = fits or {a: ("si", True) for a in governs}
    norms = []
    for n in X._fitted(T.PENAL_NORMS, fits):
        n = dict(n)
        if n["article"] in governs:
            n["governs"] = governs[n["article"]]
        norms.append(n)
    return A.make_input(X.NO_CHARGE, norms, T._sents(T.PENAL_SENTENCES, sistema=(9,)), T.PENAL_FRAME["quesiti"],
                        client="Paolo Sarti", client_roles=["imputato", "indagato"])


def test_the_lines_candidate_sets_follow_the_governs():
    inp = _penal_governs({"640 c.p.": ["pretesa"], "641 c.p.": ["Q1"]})
    gs = A.governs_sets(inp)
    assert gs == {"applied": True, "pretesa": ["640 c.p."], "quesiti": ["641 c.p."], "context": [], "enforced": False,
                  "tiers": {}}                                                       # v79.18.0 (A1): the set's tiers
    assert A.line_tier(inp) == (A.TIER_SI, ["640 c.p."])                               # the leading line's set
    assert A.line_tier(inp, "linea_subordinata") == (A.TIER_SI, ["640 c.p.", "641 c.p."])   # the subordinate's
    assert "- A641cp (art. 641 c.p.): A641cp:E1, A641cp:E2 [fit si verificato; governa: Q1] [solo linea subordinata]" in A.allowed_text(inp)
    human = A.build_messages(inp)[1].content
    assert "ARTICOLO: 640 c.p.\nRubrica: Truffa\nfit: si (verificato) — perché di 640 c.p.\n" in human
    assert "governa: pretesa\n" in human and "governa: Q1\n" in human
    assert A.prompt_hash() == "472423ea32d3"
    inp = _penal_governs({"640 c.p.": ["pretesa"], "641 c.p.": []})
    assert A.governs_sets(inp)["context"] == ["641 c.p."] and A.line_tier(inp, "linea_subordinata")[1] == ["640 c.p."]
    assert "(solo contesto: né la pretesa né un quesito)" in A.build_messages(inp)[1].content
    assert A.governs_sets(_penal_governs({}))["applied"] is False           # no governs: 79.16.2's tiers
    assert A.governs_sets(_penal_governs({"640 c.p.": [], "641 c.p.": ["Q1"]}))["applied"] is False


def test_a_context_only_norm_is_never_a_line_and_a_quesitos_only_the_subordinate():
    inp = _penal_governs({"640 c.p.": ["pretesa"], "641 c.p.": []})
    model = T._Model(T._reply(T._line("641 c.p.", T.E641[0], "Tesi.", ["S2"], ["A641cp.1"])),
                     T._reply(T._line("640 c.p.", T.E640[0], "Tesi.", ["S3"], ["A640cp.1"])))
    res = A.analyse(inp, call=model)
    assert res["rejected"][0]["why"].startswith("art. 641 c.p.: solo contesto (il fit non lo riferisce né alla pretesa né a un quesito), mai una linea: ammessi art. 640 c.p.")
    assert res["counters"]["analysis_line_context_rejected"] == 1 and res["counters"]["analysis_reasked"] == 1
    assert res["analysis"]["linea_principale"]["articolo"] == "640 c.p."
    inp = _penal_governs({"640 c.p.": ["pretesa"], "641 c.p.": ["Q1"]})
    model = T._Model(T._reply(T._line("641 c.p.", T.E641[0], "Tesi.", ["S2"], ["A641cp.1"])),
                     T._reply(T._line("640 c.p.", T.E640[0], "Tesi.", ["S3"], ["A640cp.1"]),
                              subordinata=T._line("641 c.p.", T.E641[0], "Sub.", ["S2"], ["A641cp.1"])))
    res = A.analyse(inp, call=model)
    assert res["rejected"][0]["why"].startswith("art. 641 c.p.: governa solo un quesito, ammesso come linea subordinata, non come principale: principale tra art. 640 c.p.")
    assert res["counters"]["analysis_line_quesito_only"] == 1
    assert res["analysis"]["linea_principale"]["articolo"] == "640 c.p." and res["analysis"]["linea_subordinata"]["articolo"] == "641 c.p."
    assert res["counters"]["analysis_line_context_rejected"] == 0
    for name in ("analysis_line_context_rejected", "analysis_line_quesito_only"):
        assert name in A.COUNTERS


# --- 4. the prose from verified material: one rule ------------------------------------------------------------------------------

PRETESA = {"tipo": "accusa", "contenuto": "«truffa»", "contro": "Paolo Sarti", "articolo": "640 c.p.", "derivata": False}


def _vm(binding_applied=True):
    inp = _penal_governs({"640 c.p.": ["pretesa", "Q1"], "641 c.p.": ["Q1"]})
    res = A.analyse(inp, call=T._Model(T._reply(T._line("640 c.p.", T.E640[0], "Tesi.", ["S3"], ["A640cp.1"]),
                                                elementi=[T._el("640 c.p.", T.E640[0], "provato", ["S3"])])))
    v = A.view(res, inp)
    judgments = [{"article": "640 c.p.", "cita_norma": ["A640cp.1"], "governs": ["pretesa", "Q1"]},
                 {"article": "641 c.p.", "cita_norma": ["A641cp.1"], "governs": ["Q1"]}]
    binding = {"applied": binding_applied, "principale": ["640 c.p."], "quesiti": {"Q1": ["640 c.p.", "641 c.p."]}, "context_only": []}
    vm = VM.build(v, A.id_maps(inp), PRETESA, "Paolo Sarti", judgments, binding, [], T.PENAL_FRAME["quesiti"], named=["1218 c.c."])
    return vm, v, inp


def test_the_material_block_carries_the_pretesa_the_lines_the_tagged_articles_and_the_facts_by_id():
    vm, v, inp = _vm()
    assert vm["pretesa"] == "accusa: «truffa» (art. 640 c.p.), contro Paolo Sarti" and vm["fitted"] == ["640 c.p.", "641 c.p.", "62-bis c.p."]
    assert vm["articles"]["640 c.p."]["passages"] == [("A640cp.1", inp["norms"][0]["periods"][0]["text"])]
    assert vm["articles"]["640 c.p."]["governs"] == ["pretesa", "Q1"] and vm["articles"]["62-bis c.p."]["governs"] is None
    assert vm["facts"][0][0] == "S1" and vm["facts"][0][2] == T.PENAL_SENTENCES[0]
    q = T.PENAL_FRAME["quesiti"][0]
    block = VM.block(vm, VM.SITE_QA, q, 1)
    assert block.startswith(VM.TITLE + "\nPretesa: accusa: «truffa» (art. 640 c.p.), contro Paolo Sarti.\nParte assistita: Paolo Sarti (parte che resiste).\n")
    assert f"Quesito Q1: «{q}»" in block and "Linea principale: art. 640 c.p. — " in block
    assert "Esiti dell'art. 640 c.p.: «con artifizi o raggiri» contestato;" in block     # S3: an allegation (the record)
    assert "- art. 640 c.p. (Truffa) [governa: pretesa, Q1]\n  [A640cp.1] " in block and "- art. 641 c.p. (Insolvenza fraudolenta) [governa: Q1]" in block
    assert "62-bis" not in block and "FATTI DEL CASO (da citare con il loro id):\n[S1] [accertato] " in block
    assert "CASO:" not in block and "SCHEDE" not in block
    norms_block = VM.block(vm, VM.SITE_NORMS)
    assert "NORME NELL'ORDINE DEI PARAGRAFI:\n1. art. 640 c.p." in norms_block and "641 c.p." not in norms_block.split("NORME NELL")[1]
    assert VM.block(vm, VM.SITE_NORMS, only=["641 c.p."]).count("- art. ") == 1
    assert VM.articles_for(_vm(False)[0], VM.SITE_NORMS) == ["640 c.p.", "641 c.p."]      # not applied: the fitted non-remedy norms
    assert VM.block({}, VM.SITE_QA) == ""


def test_the_one_rule_drops_an_unknown_id_or_an_outside_article_and_renders_the_ids():
    vm, v, inp = _vm()
    events = []
    kept = VM.review_sentence("Paolo Sarti ha usato artifizi [A640cp.1] e lo dice [S3].", vm, VM.SITE_QA, events)
    assert kept == f"Paolo Sarti ha usato artifizi art. 640 c.p. e lo dice «{T.PENAL_SENTENCES[2].rstrip('.')}»." and events == []
    assert VM.review_sentence("L'art. 640 c.p. [A640cp.1] punisce la truffa.", vm, VM.SITE_QA, events) == "L'art. 640 c.p. punisce la truffa."
    assert VM.review_sentence("La norma [A999cp.1] non esiste.", vm, VM.SITE_QA, events) == ""
    assert events[-1]["esito"] == A.DROPPED and events[-1]["motivo"] == VM.UNKNOWN_ID and events[-1]["id"] == ["A999cp.1"]
    assert VM.review_sentence("Si applica anche l'art. 2043 c.c. al caso.", vm, VM.SITE_QA, events) == ""
    assert events[-1]["esito"] == A.ARTICLE_UNFITTED and events[-1]["articoli"] == ["2043 c.c."]
    assert VM.review_sentence("L'art. 1218 c.c. è stato nominato dall'utente.", vm, VM.SITE_QA, events) != ""   # named: kept
    assert VM.review_sentence("Vale l'art. 646 c.p. Pertanto si applica.", vm, VM.SITE_QA, events, excluded=["646 c.p."]) == ""
    assert events[-1]["esito"] == A.ARTICLE_EXCLUDED
    text = "Prima frase sul fatto [S1]. L'art. 2043 c.c. non c'entra. Pertanto, resta la truffa ai sensi dell'art. 640 c.p. Fine."
    kept, dropped = VM.review_sentences(text, vm, VM.SITE_QA, [])
    assert dropped == 1 and kept[1].startswith("Pertanto, resta la truffa") and kept[2] == "Fine."     # no dependent drop
    done, n = VM.review_text("**Art. 640 c.p.**\n[1] Il fatto [S1]. Si applica l'art. 2043 c.c.\n\n| a | b |", vm, VM.SITE_NORMS, [])
    assert n == 1 and done.startswith("**Art. 640 c.p.**\n[1] Il fatto «") and "| a | b |" in done and "2043" not in done
    counters = A.prose_counters(events)
    assert counters["prose_unknown_id_dropped"] == 1 and counters["qa_sentences_dropped"] == 1
    assert counters["prose_unfitted_article_dropped"] == 1 and counters["prose_excluded_article_dropped"] == 1


def test_a_fragment_or_an_articles_text_is_never_the_answer():
    vm, v, inp = _vm()
    assert VM.fragment_or_copy(["Sì."], vm) == "frammento" and VM.fragment_or_copy(["senza punto finale di sei parole"], vm) == "frammento"
    assert VM.fragment_or_copy([inp["norms"][0]["periods"][0]["text"]], vm) == "copia"
    assert VM.fragment_or_copy(["Paolo Sarti risponde di truffa perché ha usato artifizi e raggiri verso Giulia Neri."], vm) == ""
    assert VM.sentences_of("Vale l'art. 1439 c.c. Pertanto si applica. Vedi art. 2 n. 3.") == \
        ["Vale l'art. 1439 c.c.", "Pertanto si applica.", "Vedi art. 2 n. 3."]


# --- 5. one object, each section one source ---------------------------------------------------------------------------------------

def test_the_object_on_the_view_and_the_sections_from_it():
    vm, v, inp = _vm()
    A.attach_object(v, pretesa=PRETESA, pretesa_line="Pretesa: accusa — «truffa» (art. 640 c.p.), contro Paolo Sarti.",
                    client="Paolo Sarti", frame_unverified={"stadio": ["parole non nel testo: udienza"]},
                    governs={"applied": True, "principale": ["640 c.p."], "quesiti": {"Q1": ["641 c.p."]}, "context_only": ["62-bis c.p."]},
                    record_facts=[{"id": "F1", "text": "x", "stato": "accertato", "chi": ""}],
                    record_uncertain=[{"kind": "ipotesi", "text": "forse"}])
    o = v["object"]
    assert o["pretesa"]["tipo"] == "accusa" and o["client"] == "Paolo Sarti" and o["side"] == A.SIDE_DEFENCE
    assert o["lines"][0]["articolo"] == "640 c.p." and o["esiti"]["640 c.p."][0]["esito"] == "contestato"
    assert o["articles_by_governs"] == {"pretesa": ["640 c.p."], "quesiti": {"Q1": ["641 c.p."]}, "context": ["62-bis c.p."], "applied": True}
    assert o["facts"] == [{"id": "F1", "text": "x", "stato": "accertato", "chi": ""}]
    assert [i["kind"] for i in o["open_items"]] == ["elemento", "elemento", "elemento", "quadro", "ipotesi"]
    assert A.summary_text(v).startswith("Pretesa: accusa — «truffa» (art. 640 c.p.), contro Paolo Sarti.\n\nLinea principale: art. 640 c.p. — ")
    assert A.question_bridge(v).startswith("La questione è se l'accusa «truffa» sia fondata sull'art. 640 c.p., posto che ")
    assert A.allowed_ruling_articles(v, ["2946 c.c."]) == ["640 c.p.", "641 c.p.", "62-bis c.p."]    # the governs, not the retrieval
    points = A.verify_points(v, [{"kind": "ipotesi", "text": "forse"}])
    assert "Campo del quadro non verificato: stadio (parole non nel testo: udienza)." in points and points[-1] == "Fatto esposto come ipotesi: «forse»."
    concl = A.conclusions_text(v)
    assert "**Rischio per Paolo Sarti**: " in concl
    # the risk in words: none provato, one contestato, two open -> the accusa infondata -> favorevole for the client resisting it
    assert A.risk_word(v) == "favorevole"
    plain = A.view(None, None)
    assert A.summary_text(plain) == "Linea principale: da determinare (linea non verificata)." and A.question_bridge(plain) == ""
    assert A.risk_word(plain) == ""


def test_the_risk_word_from_the_leading_lines_esiti():
    def v_with(esiti, side):
        els = [{"eid": f"A640cp:E{i}", "elemento": f"e{i}", "esito": e} for i, e in enumerate(esiti, 1)]
        return {"analysis": {"linea_principale": {"articolo": "640 c.p.", "stato": A.VERIFIED, "eid": "A640cp:E1"},
                             "linea_subordinata": None, "elementi": {"640 c.p.": els}, "side": side}}
    assert A.risk_word(v_with(["provato", "provato"], A.SIDE_CLAIMANT)) == "favorevole"
    assert A.risk_word(v_with(["provato", "provato"], A.SIDE_DEFENCE)) == "sfavorevole"
    assert A.risk_word(v_with(["provato", "da_provare"], A.SIDE_CLAIMANT)) == "incerto"
    assert A.risk_word(v_with(["provato", "da_provare"], A.SIDE_DEFENCE)) == "incerto"
    assert A.risk_word(v_with(["da_provare", "contestato"], A.SIDE_CLAIMANT)) == "sfavorevole"
    assert A.risk_word(v_with(["da_provare", "contestato"], A.SIDE_DEFENCE)) == "favorevole"
    assert A.risk_word(v_with(["provato", "escluso"], A.SIDE_CLAIMANT)) == "sfavorevole"
    assert A.risk_word(v_with([], A.SIDE_CLAIMANT)) == ""


def test_no_objections_prints_only_when_the_record_holds_no_non_client_allegation():
    assert LM.objections_section({"obiezioni": []}, allegations=False) == LM.NO_OBJECTIONS
    assert LM.objections_section({"obiezioni": []}, allegations=True) == LM.OBJECTIONS_FROM_ALLEGATIONS
    assert LM.objections_section({"obiezioni": []}) == LM.NO_OBJECTIONS
    ctx = {"client": "Anna Corsi", "fact_record_facts": [{"stato": "allegato", "chi": "Ettore Bassi", "text": "x"}]}
    assert LM.record_allegations(ctx) is True
    assert LM.record_allegations({"client": "Anna Corsi", "fact_record_facts": [{"stato": "allegato", "chi": "Anna Corsi"}]}) is False
    assert LM.record_allegations({"client": "Anna Corsi", "fact_record_facts": [{"stato": "accertato", "chi": ""}]}) is False


def test_the_statements_block_is_the_logs_only():
    import inspect
    src = inspect.getsource(LM.answers_block)
    assert 'log["statements_block"] = statements' in src and 'block += "\\n\\n" + statements' not in src


# --- the data and the counters -----------------------------------------------------------------------------------------------------

def test_every_law_set_has_a_family_and_the_missing_families_are_there():
    laws = LM.institution_laws()
    families = {e.get("family") for e in laws}
    assert families == {"civile", "penale", "lavoro", "amministrativo", "tributario", "famiglia", "fallimentare"}
    assert all(e.get("tags") for e in laws) and len(laws) == 36
    by_family = {f: sorted(e["law"] for e in laws if e["family"] == f) for f in families}
    assert "L. 604/1966" in by_family["lavoro"] and "L. 300/1970" in by_family["lavoro"] and "d.lgs. 23/2015" in by_family["lavoro"]
    assert "d.lgs. 104/2010" in by_family["amministrativo"] and "L. 241/1990" in by_family["amministrativo"]
    assert "d.lgs. 546/1992" in by_family["tributario"] and "L. 898/1970" in by_family["famiglia"] and "d.lgs. 14/2019" in by_family["fallimentare"]
    text = "Il lavoratore impugna il licenziamento per giustificato motivo oggettivo e chiede il TFR."
    hit = [e["law"] for e in laws if any(rx.search(text) for rx in e["_rx"])]
    assert "L. 604/1966" in hit and "c.c." in hit


def test_the_new_counters_are_declared():
    for name in ("frame_pretesa_verified", "frame_pretesa_derived", "frame_asked_side"):
        assert name in cf.COUNTERS and name in cf.empty_counters()
    assert cf.PRETESA_TIPI == ("accusa", "domanda", "ricorso", "impugnazione", "opposizione", "istanza")
    assert A.prose_counters([])["prose_unknown_id_dropped"] == 0
    import importlib.util
    spec = importlib.util.spec_from_file_location("replay_cases", os.path.join(os.path.dirname(__file__), "..", "scripts", "replay_cases.py"))
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    assert "binding_pretesa_applied" in R.FRAME_COUNTERS and "fit_governs_pretesa" in R.FIT_COUNTERS
    assert [c[0] for c in R.load_cases(R.CASES_DIR / "batch_2")] == [
        "amm_ricorrente_annullamento_permesso", "civ_convenuto_risarcimento_infiltrazioni", "fam_madre_mantenimento_affidamento",
        "lav_lavoratrice_licenziamento_disciplinare", "pen_indagato_lesioni_colpose", "trib_contribuente_avviso_accertamento"]
    assert all(len(t) > 500 for _, t in R.load_cases(R.CASES_DIR / "batch_2"))
    assert not set(c[0] for c in R.load_cases(R.CASES_DIR)) & set(c[0] for c in R.load_cases(R.CASES_DIR / "batch_2"))
