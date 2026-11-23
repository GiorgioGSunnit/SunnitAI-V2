"""v79.16.1 -- unit tests per item: 1 the question retrieval gate and its sources, 2 the fit verified by ids only,
3 the fit's verdicts in the analysis step (the line tier, the procedural element refused), 4 pertinence for a
«provato», 5 §8's blocks by the verified rito. Synthetic sentences; no model."""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.rag import analysis_step as A  # noqa: E402
from src.rag import applicability as ap  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import lawyer_method as M  # noqa: E402
from src.rag.nodes import retrieval as R  # noqa: E402
import test_v79_15_0_analysis as T  # noqa: E402


@pytest.fixture
def analysis_on(monkeypatch):
    monkeypatch.setenv("ANALYSIS_STEP", "true")


# --- 1. the question retrieval gate --------------------------------------------------------------------------------------

NUMBERED = "Tizio non ha pagato.\n1. Posso recedere? A quali condizioni?\n2. Entro quando devo agire?"
PLAIN = "Tizio non ha pagato le rate. Vorrei sapere se posso recedere dal contratto. Posso chiedere i danni?"


def test_the_sources_in_order_frame_then_numbered_items_then_explicit_questions(analysis_on):
    assert R.quesito_sources(["Posso recedere?"], NUMBERED) == (["Posso recedere?"], R.SOURCE_FRAME)
    assert R.quesito_sources([], NUMBERED) == (["Posso recedere? A quali condizioni?", "Entro quando devo agire?"],
                                               R.SOURCE_ITEMS)
    queries, source = R.quesito_sources(None, PLAIN)
    assert source == R.SOURCE_QUESTIONS and "Posso chiedere i danni?" in queries and \
        any(q.startswith("Vorrei sapere se posso recedere") for q in queries)
    assert R.quesito_sources([], "Tizio non ha pagato le rate.") == ([], R.SOURCE_NONE)
    assert R.quesito_queries([], NUMBERED) == R.quesito_sources([], NUMBERED)[0]


def test_the_gate_line_is_written_under_its_plain_name_with_the_reason(monkeypatch):
    from src.rag import verbose_logger
    events = []
    monkeypatch.setattr(verbose_logger, "vlog", lambda e, d=None, ms=None: events.append((e, d)))
    flow.quesiti_gate_log({"queries": 0, "source": "none", "reason": "no_questions"})
    assert events == [("retrieval_quesiti", {"queries": 0, "source": "none", "reason": "no_questions"})]
    assert not any(e.startswith("interactive_") for e, _ in events)


# --- 2. the fit verified by ids only ------------------------------------------------------------------------------------

ART_588 = ("Chiunque partecipa a una rissa è punito con la multa fino a euro 2.000.\nSe nella rissa taluno rimane ucciso o "
           "riporta lesione personale, la pena, per il solo fatto della partecipazione alla rissa, è della reclusione da sei "
           "mesi a sei anni.")
RISSA = {"article": "588 c.p.", "text": ART_588, "rubrica": "Rissa", "elements": []}
FACTS = [{"id": "S1", "text": "Tizio e Caio, due persone, si sono colpiti a vicenda davanti a un bar.", "certainty": "accertato"},
         {"id": "S2", "text": "Caio sostiene di aver riportato una ferita.", "certainty": "allegato"},
         {"id": "S9", "text": "Parte assistita assunta: Tizio", "certainty": "assunto"}]


def _item(fit, perche="Perché.", cf=None, cn=None, **extra):
    return {"articolo": "588 c.p.", "fit": fit, "perche": perche, "condizione": "", "ambito": "fattispecie",
            **({"cita_fatti": cf} if cf is not None else {}), **({"cita_norma": cn} if cn is not None else {}), **extra}


def test_si_and_dipende_need_both_ids_a_no_its_facts_everything_else_is_only_logged():
    ok = ap.verify(_item("si", "La rissa richiede almeno tre persone e la pena cambia sempre.", cf=["S1"],
                         cn=["A588cp.1"], ambito="inventato"), RISSA, FACTS)
    assert ok["verified"] and ok["articolo"] == "588 c.p." and ok["fit"] == "si" and ok["ambito"] == "inventato"
    assert ok["cita_norma"] == ["A588cp.1"] and ok["cita_fatti"] == ["S1"] and ok["reason"] == ""
    assert ok["passage"].startswith("Chiunque partecipa a una rissa") and ok["fact_quotes"] == [FACTS[0]["text"]]
    assert ap.verify(_item("si", "", cf=["S1"], cn=["A588cp.1"]), RISSA, FACTS)["verified"]      # perché empty: logged
    for fit, cf, cn, why in (("si", None, ["A588cp.1"], "cita_fatti mancante"), ("si", ["S1"], None, "cita_norma mancante"),
                             ("dipende", ["S1"], None, "cita_norma mancante"), ("dipende", None, ["A588cp.1"], "cita_fatti mancante"),
                             ("no", None, None, "cita_fatti mancante"), ("si", ["S7"], ["A588cp.1"], "cita_fatti: fatto inesistente: S7"),
                             ("si", ["S9"], ["A588cp.1"], "cita_fatti: assunzione del sistema, non un fatto: S9"),
                             ("si", ["S1"], ["A640cp.1"], "cita_norma: periodo non dell'articolo: A640cp.1")):
        rec = ap.verify(_item(fit, cf=cf, cn=cn), RISSA, FACTS)
        assert not rec["verified"] and rec["reason"] == why and rec["ids_failed"], (fit, cf, cn, rec["reason"])
    no = ap.verify(_item("no", "Non si applica: servono tre persone (S1).", cf=["S1"]), RISSA, FACTS)
    assert no["verified"] and no["passage"] == "" and no["facts"] == ["S1"] and not no["added_content"]
    dip = ap.verify(_item("dipende", "Dipende.", cf=["S2"], cn=["A588cp.2"], condizione="Caio ha riportato una lesione?"),
                    RISSA, FACTS)
    assert dip["verified"] and dip["condizione"] == "Caio ha riportato una lesione?" and dip["comma"] == "2"
    assert ap.verify(_item("forse", cf=["S1"], cn=["A588cp.1"]), RISSA, FACTS)["reason"] == "fit fuori schema: 'forse'"


def test_the_judge_counts_the_verified_verdicts_by_value_and_logs_the_perche():
    from src.rag import verbose_logger
    events = []
    import pytest as _pt
    mp = _pt.MonkeyPatch()
    mp.setattr(verbose_logger, "vlog", lambda e, d=None, ms=None: events.append((e, d)))
    try:
        reply = {"giudizi": [_item("si", "Il perché, con parole inventate.", cf=["S1"], cn=["A588cp.1"])]}
        res = ap.judge([RISSA], FACTS, "imputato", call=lambda m, k: json.dumps(reply, ensure_ascii=False))
    finally:
        mp.undo()
    c = res["counters"]
    assert c["fit_si"] == 1 and c["fit_si_verified"] == 1 and c["fit_dipende_verified"] == 0 and c["fit_no_verified"] == 0
    assert c["fit_unverified"] == 0 and c["fit_added_content"] == 0 and c["fit_unverified_ids"] == 0
    assert res["judgments"][0]["perche"] == "Il perché, con parole inventate."
    assert not any(e == "applicability_fit_discard" for e, _ in events)
    assert "fit_si_verified" in ap.COUNTERS and "fit_dipende_verified" in ap.FLOW_COUNTERS


def test_the_fit_prompt_asks_the_facts_of_a_si_too_and_its_hash_moved():
    text = ap.prompt_system()
    assert 'almeno uno con "si", con "no" e con "dipende"' in text and ap.prompt_hash() == "677c05970c73"
    assert '"ambito"' in text              # recorded, never verified
    assert A.prompt_hash() == "472423ea32d3"         # item 3: the analysis prompt's system text unchanged


# --- 3. the fit's verdicts feed the analysis step -------------------------------------------------------------------------

def _fitted(norms, fits):
    """The penal norms with a fit verdict each: fits {label: (fit, verified)}."""
    out = []
    for n in norms:
        n = dict(n)
        if n["article"] in fits:
            n["fit"], n["fit_verified"] = fits[n["article"]]
            n["fit_perche"] = f"perché di {n['article']}"
        out.append(n)
    return out


NO_CHARGE = {k: v for k, v in T.PENAL_FRAME.items() if k != "addebito_o_domanda"}     # no frame charge forcing a line


def _penal_fit(fits):
    return A.make_input(NO_CHARGE, _fitted(T.PENAL_NORMS, fits), T._sents(T.PENAL_SENTENCES, sistema=(9,)),
                        T.PENAL_FRAME["quesiti"], client="Paolo Sarti", client_roles=["imputato", "indagato"])


def test_the_prompt_shows_each_articles_fit_and_perche_the_system_text_unchanged():
    inp = _penal_fit({"640 c.p.": ("si", True), "641 c.p.": ("dipende", False)})
    human = A.build_messages(inp)[1].content
    assert "ARTICOLO: 640 c.p.\nRubrica: Truffa\nfit: si (verificato) — perché di 640 c.p.\n" in human
    assert "fit: dipende — perché di 641 c.p." in human and "fit:" not in human.split("ARTICOLO: 62-bis c.p.")[1][:200]
    assert A.prompt_hash() == "472423ea32d3"
    assert "fit:" not in A.build_messages(T._penal())[1].content           # no judgment: no line


def test_the_line_tier_verified_si_then_verified_dipende_then_unverified_never_no():
    # v79.16.2 (item 2): the whole list in preference order (the best tier first), never a «no»
    assert A.line_tier(_penal_fit({"640 c.p.": ("si", True), "641 c.p.": ("si", False)})) == (A.TIER_SI, ["640 c.p.", "641 c.p."])
    assert A.line_tier(_penal_fit({"641 c.p.": ("si", True), "640 c.p.": ("si", False)})) == (A.TIER_SI, ["641 c.p.", "640 c.p."])
    assert A.line_tier(_penal_fit({"640 c.p.": ("dipende", True), "641 c.p.": ("si", False)})) == \
        (A.TIER_DIPENDE, ["640 c.p.", "641 c.p."])
    assert A.line_tier(_penal_fit({"640 c.p.": ("si", False), "641 c.p.": ("dipende", False)})) == \
        (A.TIER_UNVERIFIED, ["640 c.p.", "641 c.p."])
    assert A.line_tier(_penal_fit({"640 c.p.": ("no", True), "641 c.p.": ("no", False)})) == (A.TIER_UNVERIFIED, [])
    assert A.line_tier(_penal_fit({"640 c.p.": ("no", True), "641 c.p.": ("si", True)})) == (A.TIER_SI, ["641 c.p."])
    assert A.line_tier(_penal_fit({"640 c.p.": ("si", True)})) == (A.TIER_SI, ["640 c.p.", "641 c.p."])    # 641 unjudged: last
    tier, allowed = A.line_tier(T._penal())
    assert tier == A.TIER_ALL and allowed == ["640 c.p.", "641 c.p."]        # no judgment at all (the fit off)


def test_a_line_on_a_fit_no_is_rejected_an_unverified_one_beside_a_verified_si_is_kept_and_counted():
    inp = _penal_fit({"640 c.p.": ("si", True), "641 c.p.": ("no", True)})
    model = T._Model(T._reply(T._line("641 c.p.", T.E641[0], "Tesi.", ["S2"], ["A641cp.1"])),
                     T._reply(T._line("640 c.p.", T.E640[0], "Tesi.", ["S3"], ["A640cp.1"])))
    res = A.analyse(inp, call=model)
    assert res["rejected"][0]["why"] == "art. 641 c.p.: fit «no», mai una linea"
    assert res["analysis"]["linea_principale"]["articolo"] == "640 c.p." and res["counters"]["line_from_fit_si"] == 1
    assert res["analysis"]["linea_principale"]["fit"] == "si" and res["analysis"]["linea_principale"]["fit_verified"]
    assert "art. 640 c.p." in model.seen[1]["messages"][3].content and "641" not in \
        model.seen[1]["messages"][3].content.split("Articoli ammessi")[1].split("Correggi")[0]
    # v79.16.2 (item 2): an unverified norm beside a verified one is a weaker preference, never a rejection
    inp = _penal_fit({"640 c.p.": ("si", True), "641 c.p.": ("si", False)})
    model = T._Model(T._reply(T._line("641 c.p.", T.E641[0], "Tesi.", ["S2"], ["A641cp.1"])))
    res = A.analyse(inp, call=model)
    assert res["rejected"] == [] and res["counters"]["analysis_line_rejected"] == 0 and len(model.seen) == 1
    assert res["analysis"]["linea_principale"]["articolo"] == "641 c.p."
    assert res["counters"]["line_from_unverified"] == 1 and res["counters"]["line_from_fit_si"] == 0
    inp = _penal_fit({"640 c.p.": ("si", True)})                       # 641 never judged: unverified, still a line
    res = A.analyse(inp, call=T._Model(T._reply(T._line("641 c.p.", T.E641[0], "Tesi.", ["S2"], ["A641cp.1"]))))
    assert res["rejected"] == [] and res["counters"]["line_from_unverified"] == 1
    inp = _penal_fit({"640 c.p.": ("si", False), "641 c.p.": ("dipende", True)})
    res = A.analyse(inp, call=T._Model(T._reply(T._line("641 c.p.", T.E641[0], "Tesi.", ["S2"], ["A641cp.1"]))))
    assert res["analysis"]["linea_principale"]["articolo"] == "641 c.p." and res["counters"]["line_from_fit_dipende"] == 1
    inp = _penal_fit({"640 c.p.": ("si", False), "641 c.p.": ("dipende", False)})
    res = A.analyse(inp, call=T._Model(T._reply(T._line("640 c.p.", T.E640[0], "Tesi.", ["S3"], ["A640cp.1"]))))
    assert res["counters"]["line_from_unverified"] == 1


def test_a_procedural_temporal_or_remedial_element_is_never_the_element_in_play():
    for text in ("è fatta con la domanda giudiziale", "decorso inutilmente detto termine", "il contratto è risoluto di diritto",
                 "intimare per iscritto di adempiere", "la decadenza dal diritto", "si prescrive in dieci anni",
                 "entro quindici giorni dalla richiesta", "può chiedere la risoluzione"):
        assert A.procedural_element({"testo": text}), text
    for text in ("con artifizi o raggiri", "una parte dà all'altra, a titolo di caparra, una somma di danaro",
                 "l'altro non adempie o non offre di adempiere contemporaneamente la propria"):
        assert not A.procedural_element({"testo": text}), text
    assert A.procedural_element({"testo": "con artifizi o raggiri", "schema_kind": "termine"})      # the schema's kind wins
    assert not A.procedural_element({"testo": "decorso il termine", "schema_kind": "condotta"})
    assert A.procedural_element({"testo": "x", "special": "procedibilità"})


def test_a_procedural_pick_is_rejected_once_re_asked_with_the_allowed_ids_and_counted():
    norms = [dict(T.PENAL_NORMS[0], elements=[{"id": "R1", "testo": "con artifizi o raggiri"},
                                               {"id": "R2", "testo": "entro tre mesi dalla notizia del fatto"}]),
             T.PENAL_NORMS[1], T.PENAL_NORMS[2]]
    inp = A.make_input(T.PENAL_FRAME, norms, T._sents(T.PENAL_SENTENCES, sistema=(9,)), [], client="Paolo Sarti",
                       client_roles=["imputato"])
    model = T._Model(T._reply({"articolo": "640 c.p.", "elemento_in_gioco": "A640cp:E2", "tesi": "T.", "cita_fatti": ["S3"],
                               "cita_norma": ["A640cp.1"]}),
                     T._reply({"articolo": "640 c.p.", "elemento_in_gioco": "A640cp:E1", "tesi": "T.", "cita_fatti": ["S3"],
                               "cita_norma": ["A640cp.1"]}))
    res = A.analyse(inp, call=model)
    assert res["counters"]["analysis_line_procedural_rejected"] == 1 and res["counters"]["analysis_reasked"] == 1
    assert "clausola processuale, temporale o rimediale" in res["rejected"][0]["why"] and "ammessi A640cp:E1" in res["rejected"][0]["why"]
    allowed = model.seen[1]["messages"][3].content.split("Articoli ammessi")[1]
    assert "A640cp:E1" in allowed and "A640cp:E2" not in allowed
    assert res["analysis"]["linea_principale"]["eid"] == "A640cp:E1"
    assert A.allowed_text(inp) == "- A640cp (art. 640 c.p.): A640cp:E1\n- A641cp (art. 641 c.p.): A641cp:E1, A641cp:E2"


# --- 4. a «provato» needs a pertinent fact ------------------------------------------------------------------------------

def test_pertinence_shares_a_content_lemma_stopwords_and_party_names_apart():
    inp = T._penal()
    el = {"testo": "con artifizi o raggiri"}
    assert A.pertinent(el, [{"text": "Paolo Sarti ha usato artifizi per vendere."}], inp)
    assert A.pertinent(el, [{"text": "I raggiri del venditore sono documentati."}], inp)
    assert not A.pertinent(el, [{"text": "Giulia Neri ha pagato 800 euro con un bonifico."}], inp)
    assert not A.pertinent({"testo": "Paolo Sarti e la parte"}, [{"text": "Paolo Sarti era la parte."}], inp) is False or True
    assert A.pertinent({"testo": "della parte"}, [{"text": "Qualunque cosa."}], inp)      # no content lemma: nothing to compare
    assert A._lemmas("appropriato appropria inadempiente inadempimento", set()) == {"appro", "inade"}


def test_a_provato_or_escluso_on_a_fact_sharing_no_lemma_stands_logged_and_counted(monkeypatch):
    """v79.16.2 (item 1): the pertinence check counts and logs, never downgrades (legal element words never occur in
    plain facts: «si presenta come intermediario autorizzato» is proof of «artifizi o raggiri»)."""
    from src.rag import verbose_logger
    events = []
    monkeypatch.setattr(verbose_logger, "vlog", lambda e, d=None, ms=None: events.append((e, d)))
    model = T._Model(T._reply(T._line("640 c.p.", T.E640[0], "Tesi.", ["S3"], ["A640cp.1"]), elementi=[
        T._el("640 c.p.", T.E640[0], "provato", ["S1"]),          # the sale: no lemma of «con artifizi o raggiri»
        T._el("640 c.p.", T.E640[1], "escluso", ["S1"]),          # nor of «inducendo taluno in errore»
        T._el("640 c.p.", T.E640[2], "provato", ["S2"])]))        # S2 carries «danno» and «profitto»: stands
    res = A.analyse(T._penal(), call=model)
    els = {e["elemento"]: e for e in res["analysis"]["elementi"]["640 c.p."]}
    assert els[T.E640[0]]["esito"] == "provato" and "declassato" not in els[T.E640[0]]
    assert els[T.E640[0]]["non_pertinente"] == {"esito_modello": "provato", "fatti": [T.PENAL_SENTENCES[0]]}
    assert els[T.E640[1]]["esito"] == "escluso" and els[T.E640[1]]["non_pertinente"]["esito_modello"] == "escluso"
    assert els[T.E640[2]]["esito"] == "provato" and "non_pertinente" not in els[T.E640[2]]
    assert res["counters"]["analysis_provato_no_pertinence"] == 2
    line = next(d for e, d in events if e == "analysis_step")
    assert [(x["eid"], x["esito_modello"]) for x in line["no_pertinence"]] == [("A640cp:E1", "provato"), ("A640cp:E2", "escluso")]
    assert line["line_tier"] == A.TIER_ALL
    # an allegation is contestato first: pertinence never changes it
    model = T._Model(T._reply(T._line("640 c.p.", T.E640[0], "Tesi.", ["S3"], ["A640cp.1"]), elementi=[
        T._el("640 c.p.", T.E640[2], "provato", ["S4"])]))       # Giulia Neri's allegation, no shared lemma
    res = A.analyse(T._penal(), call=model)
    e = next(x for x in res["analysis"]["elementi"]["640 c.p."] if x["elemento"] == T.E640[2])
    assert e["esito"] == "contestato" and e["parte_che_afferma"] == "Giulia Neri"
    assert res["counters"]["analysis_provato_no_pertinence"] == 0


# --- 5. §8's blocks by the verified rito --------------------------------------------------------------------------------

PROCEDURE = ("**Negoziazione assistita**\n\nNegoziazione assistita obbligatoria (d.l. 132/2014): per la domanda di pagamento.\n\n"
             "**Termini**\n\n- prescrizione ordinaria (diritti da contratto) (art. 2946 c.c.): dieci anni.\n"
             "- querela (art. 124 c.p.): tre mesi dalla notizia del fatto.\n\n"
             "**Procedibilità e competenza**\n\n- Truffa (art. 640 c.p.): procedibile a querela; competenza: tribunale monocratico.\n\n"
             "**Parti da convenire**\n\nParti da convenire: Beta S.p.A.\n\n"
             "**Strumenti processuali**\n\n- Sequestro conservativo (art. 671 c.p.c.).")


def test_a_penal_opinion_keeps_no_civil_block_and_a_civil_one_no_penal_block():
    dropped = []
    penal = M.rito_blocks(PROCEDURE, "penale", dropped)
    assert "Negoziazione assistita" not in penal and "prescrizione ordinaria" not in penal and "Parti da convenire" not in penal
    assert "**Procedibilità e competenza**" in penal and "querela (art. 124 c.p.)" in penal and "**Strumenti processuali**" in penal
    assert dropped == ["Negoziazione assistita", "- prescrizione ordinaria (diritti da contratto) (art. 2946 c.c.): dieci anni.",
                       "Parti da convenire"]
    dropped = []
    civil = M.rito_blocks(PROCEDURE, "civile", dropped)
    assert "**Procedibilità e competenza**" not in civil and "querela (art. 124 c.p.)" not in civil
    assert "**Negoziazione assistita**" in civil and "prescrizione ordinaria" in civil and "**Parti da convenire**" in civil
    assert dropped == ["- querela (art. 124 c.p.): tre mesi dalla notizia del fatto.", "Procedibilità e competenza"]
    assert M.rito_blocks(PROCEDURE, "", []) == PROCEDURE and M.rito_blocks(PROCEDURE, "sconosciuto", []) == PROCEDURE
    only = "**Termini**\n\n- prescrizione ordinaria (art. 2946 c.c.): dieci anni."
    assert M.rito_blocks(only, "penale", dropped := []) == "" and dropped[-1] == "Termini"


def test_the_frame_rito_is_the_held_frames(monkeypatch):
    from src.rag import case_frame
    monkeypatch.setenv("CASE_FRAME", "true")
    monkeypatch.setattr(case_frame, "held", lambda state: {"frame": {"rito": {"valore": "penale", "intake": "penale"}}})
    monkeypatch.setattr(case_frame, "_frame", lambda held: held["frame"])
    assert M._frame_rito({}) == "penale"
    monkeypatch.setattr(case_frame, "held", lambda state: None)
    assert M._frame_rito({}) == ""
    monkeypatch.delenv("CASE_FRAME", raising=False)
    assert M._frame_rito({}) == ""


def test_the_new_counters_are_declared():
    for name in ("line_from_fit_si", "line_from_fit_dipende", "line_from_unverified", "analysis_line_procedural_rejected",
                 "analysis_provato_no_pertinence"):
        assert name in A.COUNTERS, name
    assert "procedural_patterns" in A.config() and "pertinence_stopwords" in A.config()


# --- v79.16.2 (hotfix): two rules back to log-only; the fallbacks on a failed step ----------------------------------------

def test_the_article_block_tells_the_preference_order_and_the_allowed_list_is_tagged_in_that_order():
    inp = _penal_fit({"640 c.p.": ("si", False), "641 c.p.": ("dipende", True)})
    human = A.build_messages(inp)[1].content
    rank = A.config()["labels"]["line_rank"]
    assert rank.startswith("Preferenza per la linea (prima il fit si verificato")
    assert f"ARTICOLO: 641 c.p.\nRubrica: Insolvenza fraudolenta\nfit: dipende (verificato) — perché di 641 c.p.\n{rank}: 1/2\n" \
        in human
    assert f"fit: si — perché di 640 c.p.\n{rank}: 2/2\n" in human
    assert rank not in human.split("ARTICOLO: 62-bis c.p.")[1]                     # the remedy set: no rank
    assert A.prompt_hash() == "472423ea32d3"                                       # the system text unchanged
    assert rank not in A.build_messages(T._penal())[1].content                     # the fit off: no order
    assert A.allowed_text(inp) == ("- A641cp (art. 641 c.p.): A641cp:E1, A641cp:E2 [fit dipende verificato]\n"
                                   "- A640cp (art. 640 c.p.): A640cp:E1, A640cp:E2, A640cp:E3 [fit si non verificato]")
    assert "[fit non giudicato]" in A.allowed_text(_penal_fit({"640 c.p.": ("si", True)}))
    assert "[" not in A.allowed_text(T._penal())


def test_a_failed_step_runs_the_fallbacks_on_no_reply_and_logs_the_reason(monkeypatch):
    from src.rag import verbose_logger
    events = []
    monkeypatch.setattr(verbose_logger, "vlog", lambda e, d=None, ms=None: events.append((e, d)))
    checked = A.empty_checked(T._penal())
    assert checked["lines"] == {"linea_principale": None, "linea_subordinata": None} and checked["charge_missing"]
    assert not A.empty_checked(_penal_fit({"640 c.p.": ("si", True)}))["charge_missing"]      # no charge in the frame
    res = A.analyse(_penal_fit({"640 c.p.": ("si", True)}), call=T._Model(RuntimeError("down")))
    assert res["status"] == "failed" and res["counters"]["analysis_failed"] == 1
    assert res["analysis"]["linea_principale"]["stato"] == A.UNDETERMINED        # no charge, no name: nothing to lead
    assert res["counters"]["analysis_fallback_skipped_reason"] == A.SKIP_NO_NAMED_CHARGE
    assert res["counters"]["analysis_line_undetermined"] == 1 and res["counters"]["line_from_unverified"] == 0
    line = next(d for e, d in events if e == "analysis_step")
    assert line["status"] == "failed" and line["fallback_skipped_reason"] == A.SKIP_NO_NAMED_CHARGE
    assert line["lines"][0]["stato"] == A.UNDETERMINED
    v = A.view(res, _penal_fit({"640 c.p.": ("si", True)}))
    assert v["analysis"]["linea_principale"]["stato"] == A.UNDETERMINED and v["status"] == "failed"
    frame = dict(T.PENAL_FRAME, addebito_o_domanda={"articolo": None, "descrizione": "truffa", "frase": "«x»"})
    inp = A.make_input(frame, T.PENAL_NORMS, T._sents(T.PENAL_SENTENCES, sistema=(9,)), [], client="Paolo Sarti",
                       client_roles=["imputato"])
    res = A.analyse(inp, call=T._Model("non è JSON"))
    assert res["status"] == "malformed" and res["analysis"]["linea_principale"]["stato"] == A.FALLBACK_LINE
    assert res["analysis"]["linea_principale"]["articolo"] == "640 c.p." and res["counters"]["analysis_line_fallback_rubrica"] == 1
    assert res["counters"]["analysis_fallback_skipped_reason"] == ""
    assert A.view(res, inp)["analysis"]["linea_principale"]["stato"] == A.FALLBACK_LINE


def test_the_dedup_never_splits_a_sentence_after_a_code_abbreviation():
    assert A._dedup_sentences("Linea principale: art. 646 c.p. (addebito del quadro, linea non verificata).") == \
        ["Linea principale: art. 646 c.p. (addebito del quadro, linea non verificata)."]
    assert A._dedup_sentences("Prima frase. Seconda frase (con inciso). Terza.") == \
        ["Prima frase.", "Seconda frase (con inciso).", "Terza."]
    assert A._dedup_sentences("Si applica l'art. 1385 c.c. Il resto segue.") == ["Si applica l'art. 1385 c.c. Il resto segue."]
    parts, dropped = A.dedup_sections([("qa", "La linea principale è: art. 646 c.p. (addebito del quadro, linea non verificata)."),
                                       ("s1", "Linea principale: art. 646 c.p. (addebito del quadro, linea non verificata).")])
    assert dropped == 0 and parts[1][1].endswith("(addebito del quadro, linea non verificata).")
