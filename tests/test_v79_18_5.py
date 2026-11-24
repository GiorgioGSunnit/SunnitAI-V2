"""v79.18.5 -- verification parity for the strategy's text: item 1 every free-text field reviewed as the Q&A's prose (a
factual sentence with no S-id dropped); item 2 a rule sentence with its A-id printed as the verbatim passage (§4, §7,
the strategy); item 3 ids and labels never on the page (words, ordinals, element texts, estremi; the sweep); item 4 a
passo asking a legal conclusion is no passo; item 5 the objection's accusa_forte by S-ids, an ungrounded objection
dropped; item 6 a general-part ruling kept when its article is attached to an element of the line; item 7 the charge's
capo as alternative candidates (deterministic). Synthetic inputs only."""
import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import test_v79_17_3 as T173  # noqa: E402
import test_v79_18_0 as Z  # noqa: E402
import test_v79_18_1 as C  # noqa: E402
from src.rag import analysis_step as A  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import lawyer_method as M  # noqa: E402
from src.rag import strategy_step as S  # noqa: E402
from src.rag import topic_groups as TG  # noqa: E402
from src.rag import verified_material as V  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("replay_cases_185", ROOT / "scripts" / "replay_cases.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

FACT2 = "«Anna Corsi ha versato 15.000 euro a titolo di caparra»"


def _inp():
    inp = Z._inp()
    inp["parties"] = ["Anna Corsi", "Ettore Bassi"]
    return inp


# --- item 1 ---------------------------------------------------------------------------------------------------------------

def test_item1_a_factual_sentence_without_an_s_id_is_dropped_and_a_cited_one_is_rendered():
    inp = _inp()
    assert S.factual("Anna Corsi non si è presentata dal notaio.", inp) is True        # a party named
    assert S.factual("Il versamento è del 3 marzo.", inp) is True                       # a figure
    assert S.factual("La caparra è il punto conteso.", inp) is False
    assert S.factual("La società S.r.l. ha pagato.", {"parties": ["Alfa S.r.l."]}) is False   # a company form is no name
    c = S.empty_counters()
    # a factual sentence with no id: dropped; a non-factual one stays; a cited one is rendered with the fact's quote
    text = "Anna Corsi non si è presentata dal notaio. La caparra è il punto conteso. Anna Corsi ha versato la caparra [S2]."
    got = S.review_free_text(text, inp, None, c)
    assert got == f"La caparra è il punto conteso. Anna Corsi ha versato la caparra {FACT2}."
    assert c["strategy_sentences_dropped_no_fact"] == 1 and c["strategy_ids_rendered"] == 1 and c["strategy_sentences_dropped"] == 0
    # the field's own cita_fatti grounds a one-sentence field
    c = S.empty_counters()
    assert S.review_free_text("Anna Corsi non si è presentata dal notaio.", inp, None, c, listed=["S3"]) == \
        "Anna Corsi non si è presentata dal notaio."
    assert c["strategy_sentences_dropped_no_fact"] == 0
    # two sentences: the list grounds neither
    c = S.empty_counters()
    assert S.review_free_text("Anna Corsi non si è presentata. Ettore Bassi ha venduto.", inp, None, c, listed=["S3"]) == ""
    assert c["strategy_sentences_dropped_no_fact"] == 2
    # the Q&A's reviewer: an unknown id drops the sentence (strategy_sentences_dropped)
    c = S.empty_counters()
    assert S.review_free_text("La caparra risulta versata [S9].", inp, None, c) == "" and c["strategy_sentences_dropped"] == 1
    assert S.review_free_text("", inp, None, S.empty_counters()) == ""
    # the material: compose's when it has maps, else the input's facts
    assert S.material(inp, None)["maps"]["sentences"]["S2"]["text"].startswith("Anna Corsi ha versato")
    vm = {"maps": {"periods": {}, "sentences": {"S2": {"text": "x"}}}, "fitted": []}
    assert S.material(inp, vm) is vm


def test_item1_the_fields_are_reviewed_in_finish_and_an_ungrounded_objection_goes():
    inp = _inp()
    reply = Z._reply([
        Z._linea("A1385cc", "A1385cc:E1", "fondare", "principale", ["S2"], perche="Ettore Bassi ha ricevuto la caparra e tace",
                 ob={"testo": "Anna non si è presentata dal notaio.", "risposta_cita_fatti": ["S2"], "cita_pronunce": []}),
        Z._linea("A1385cc", "A1385cc:E2", "contestare", "subordinata", ["S3"], perche="l'inadempimento è conteso",
                 ob={"testo": "Bassi era legittimato a vendere [S3].", "risposta_cita_fatti": [], "cita_pronunce": []})],
        passi=[{"eid": "A1385cc:E2", "cosa": "Produrre la diffida inviata a Ettore Bassi", "cita_fatti": []}],
        passi_immediati=[{"testo": "Conservare la ricevuta di Anna Corsi", "cita_elementi": ["A1385cc:E1"], "cita_fatti": []},
                         {"testo": "Conservare la ricevuta [S2]", "cita_elementi": [], "cita_fatti": ["S2"]}],
        contro_difesa=[{"allegazione": "S3", "risposta": "Ettore Bassi non ha mai convocato Anna Corsi", "cita_linea": "A1385cc:E2",
                        "cita_fatti": []}],
        verifiche_documentali=[{"testo": "Verificare il termine per il rogito", "cita_fatti": ["S1"], "cita_elementi": []}])
    checked = S.verify(reply, inp)
    a = S.finish(checked, inp)
    c = checked["counters"]
    # the perche of the first line is one sentence grounded by its cita_fatti list; the objection without any S-id is gone
    first = next(x for x in a["linee"] if x["eid"] == "A1385cc:E1")
    assert first["perche_prima"] == "Ettore Bassi ha ricevuto la caparra e tace" and first["obiezione"] is None
    second = next(x for x in a["linee"] if x["eid"] == "A1385cc:E2")
    assert second["obiezione"]["testo"].startswith("Bassi era legittimato a vendere «Ettore Bassi sostiene")
    assert c["strategy_objection_dropped"] == 1 and c["strategy_sentences_dropped_no_fact"] >= 3
    # the passo naming a party with no id: the code's label; the immediate step naming a party with no id: gone
    assert [(p["eid"], p["codice"]) for p in a["passi"]] == [("A1385cc:E2", True), ("A1385cc:E3", True)]
    assert a["passi_immediati"] == [{"testo": f"Conservare la ricevuta {FACT2}", "cita_fatti": ["S2"], "cita_elementi": []}]
    # the contro-difesa's answer naming parties with no id: emptied, the line still answers it
    assert a["contro_difesa"][0]["risposta"] == "" and a["contro_difesa"][0]["senza_risposta"] is False
    assert a["verifiche_documentali"][0]["testo"] == "Verificare il termine per il rogito"
    assert any(e["motivo"] == "fatto_senza_id" for e in a["review_events"])


# --- item 2 ---------------------------------------------------------------------------------------------------------------

def _vm():
    return {"maps": {"periods": {"A1385cc.1": {"article": "1385 c.c.", "text": "Se al momento della conclusione del contratto una parte dà all'altra una somma di denaro a titolo di caparra, la caparra deve essere restituita."},
                                 "A1385cc.2": {"article": "1385 c.c.", "text": "Se la parte che ha dato la caparra è inadempiente, l'altra può recedere dal contratto."}},
                     "sentences": {"S2": {"text": "Anna Corsi ha versato 15.000 euro a titolo di caparra"}}},
            "fitted": ["1385 c.c."], "named": []}


def test_item2_a_rule_sentence_with_its_a_id_prints_as_the_verbatim_passage_in_norme_obiezioni_and_the_strategy():
    vm = _vm()
    sentence = "L'art. 1385 c.c. [A1385cc.2] prevede che chi ha dato la caparra e non adempie la perda."
    for site in (A.SITE_NORMS, A.SITE_OBJECTIONS, S.SITE_STRATEGY):
        events = []
        assert V.review_sentence(sentence, vm, site, events) == \
            "art. 1385 c.c. («Se la parte che ha dato la caparra è inadempiente, l'altra può recedere dal contratto»)."
        assert events == [{"sede": site, "esito": "legge_verbatim", "testo": sentence[:200], "articoli": ["1385 c.c."]}]
    # the Q&A keeps the writer's wording (its ids rendered)
    events = []
    assert V.review_sentence(sentence, vm, A.SITE_QA, events).startswith("L'art. 1385 c.c. prevede che") and events == []
    # two periods of the same article: both quoted; no rule verb or no A-id: nothing replaced
    both = "L'art. 1385 c.c. [A1385cc.1] [A1385cc.2] dispone la restituzione e il recesso."
    assert V.law_verbatim(both, ["A1385cc.1", "A1385cc.2"], vm).count("«") == 2
    assert V.law_verbatim("La caparra è stata versata [S2].", ["S2"], vm) == ""
    assert V.law_verbatim("L'art. 1385 c.c. [A1385cc.2] è la norma della caparra.", ["A1385cc.2"], vm) == ""
    assert A.prose_counters(events + [{"sede": "norme", "esito": "legge_verbatim"}])["law_quoted_verbatim"] == 1
    assert "law_quoted_verbatim" in A.PROSE_COUNTERS and "law_quoted_verbatim" in R.QA_COUNTERS
    # the strategy's free text: the passage in place of the model's wording
    inp = _inp()
    c = S.empty_counters()
    assert S.review_free_text(sentence, inp, vm, c) == \
        "art. 1385 c.c. («Se la parte che ha dato la caparra è inadempiente, l'altra può recedere dal contratto»)."


# --- item 3 ---------------------------------------------------------------------------------------------------------------

def test_item3_words_ordinals_texts_and_estremi_on_the_page_and_the_sweep_counts_residues():
    assert S._word("tipo_words", "fondare") == "fondare la pretesa sull'elemento" and S._word("tipo_words", "cumulativa") == "cumulativa"
    assert S._word("rapporto_words", "cumulativa") == "in via cumulativa" and S._word("rapporto_words", "principale") == "in via principale"
    assert S.ordinal(1) == "prima linea" and S.ordinal(3) == "terza linea" and S.ordinal(7) == "7ª linea"
    c = {}
    assert S.swept("la linea su A1385cc:E1 cita G2 e [S3] e A1385cc.2.", c) == "la linea su cita e e."
    assert c["residues_dropped"] == 4
    assert S.swept("Il bonifico del 3 marzo 2024 (art. 1385 c.c.).", c) == "Il bonifico del 3 marzo 2024 (art. 1385 c.c.)." and c["residues_dropped"] == 4
    inp = _inp()
    reply = Z._reply([Z._linea("A1385cc", "A1385cc:E1", "fondare", "principale", ["S2"], ["G2"], perche="la caparra è provata",
                               ob={"testo": "La controparte negherà il versamento [S3].", "risposta_cita_fatti": ["S2"], "cita_pronunce": ["G2"]}),
                      Z._linea("A1385cc", "A1385cc:E2", "contestare", "subordinata", ["S3"]),
                      Z._linea("A1385cc", "A1385cc:E3", "eccepire", "cumulativa", ["S1"])],
                     passi_immediati=[{"testo": "Conservare la ricevuta", "cita_elementi": ["A1385cc:E1"], "cita_fatti": ["S2"]}])
    res = S.analyse(inp, call=C.T._Model(reply))
    v = Z._view(inp)
    counters = {"strategy_ids_rendered": 0, "residues_dropped": 0}
    sol = S.solutions_text(res, v, inp["quesiti"], inp["quesiti_articles"], counters=counters)
    assert "| Prima linea |" in sol and "| Terza linea |" in sol and "| eccepire | in via cumulativa |" in sol
    assert "Cass. civ., Sez. 2, n. 100/2025" in sol and "(G2)" not in sol and "[A1385cc:E1]" not in sol
    assert "(«una parte dà all'altra, a titolo di caparra, una somma»)" in sol
    assert not S._RESIDUE_RE.search(sol) and counters["residues_dropped"] == 0
    items = S.objection_items(res, v, counters)
    assert items[0]["risposta"].endswith("(Cass. civ., Sez. 2, n. 100/2025).") and "G2" not in items[0]["risposta"]
    assert [label for label, _ in S.summary_lines(res, v)][2] == "Terza linea"
    assert "strategy_ids_rendered" in R.STRATEGY_COUNTERS and "residues_dropped" in R.STRATEGY_COUNTERS


# --- item 4 ---------------------------------------------------------------------------------------------------------------

def test_item4_a_passo_asking_a_legal_conclusion_is_no_passo():
    inp = _inp()
    assert S.passo_issue({"eid": "A1385cc:E2", "cosa": "Produrre la diffida ad adempiere e la PEC di risposta"}, inp) == ""
    assert "conclusione giuridica" in S.passo_issue({"eid": "A1385cc:E2", "cosa": "Acquisire i documenti per accertare la responsabilità di Bassi"}, inp)
    assert "conclusione giuridica" in S.passo_issue({"eid": "A1385cc:E2", "cosa": "Produrre il contratto per dimostrare che Bassi è inadempiente"}, inp)
    assert "prova concreta" in S.passo_issue({"eid": "A1385cc:E2", "cosa": "Valutare la posizione"}, inp)
    assert "riformula" in S.passo_issue({"eid": "A1385cc:E2", "cosa": "la parte che ha ricevuto la caparra è inadempiente"}, inp)
    reply = Z._reply([Z._linea("A1385cc", "A1385cc:E2", "contestare", "principale", ["S3"]),
                      Z._linea("A1385cc", "A1385cc:E1", "fondare", "subordinata", ["S2"])],
                     passi=[{"eid": "A1385cc:E2", "cosa": "Dimostrare che la controparte è inadempiente", "cita_fatti": []}])
    checked = S.verify(reply, inp)
    assert any(r.startswith("passo A1385cc:E2") for r in checked["reasons"])       # no means named: re-asked
    a = S.finish(checked, inp)
    assert next(p for p in a["passi"] if p["eid"] == "A1385cc:E2")["codice"] is True
    assert checked["counters"]["strategy_passi_not_concrete"] == 1
    assert S.config()["passi_conclusion_words"]


# --- item 5 ---------------------------------------------------------------------------------------------------------------

def test_item5_the_accusa_forte_by_s_ids_grounds_the_objection_and_prints_the_facts():
    inp = _inp()
    c = S.empty_counters()
    forte = {"testo": "Farà leva sulla vendita a un terzo [S3].", "cita_fatti": ["S3"]}
    got, _ = S._verify_objection({"testo": "La controparte negherà l'inadempimento.", "accusa_forte": forte,
                                  "risposta_cita_fatti": ["S2"], "cita_pronunce": []}, inp, c, 1)
    assert got["grounded"] is True and got["accusa_forte"] == {"testo": forte["testo"], "cita_fatti": ["S3"]}
    got, _ = S._verify_objection({"testo": "La controparte negherà l'inadempimento.", "risposta_cita_fatti": ["S2"], "cita_pronunce": []}, inp, c, 1)
    assert got["grounded"] is False and got["accusa_forte"] is None
    got, _ = S._verify_objection({"testo": "La controparte negherà il versamento [S2].", "risposta_cita_fatti": [], "cita_pronunce": []}, inp, c, 1)
    assert got["grounded"] is True
    got, _ = S._verify_objection({"testo": "x", "accusa_forte": {"testo": "y", "cita_fatti": ["S9"]}, "risposta_cita_fatti": [], "cita_pronunce": []}, inp, c, 1)
    assert got["grounded"] is False and got["accusa_forte"] is None            # an unknown id grounds nothing
    reply = Z._reply([Z._linea("A1385cc", "A1385cc:E1", "fondare", "principale", ["S2"],
                               ob={"testo": "La controparte negherà l'inadempimento.", "accusa_forte": forte,
                                   "risposta_cita_fatti": ["S2"], "cita_pronunce": []}),
                      Z._linea("A1385cc", "A1385cc:E2", "contestare", "subordinata", ["S3"],
                               ob={"testo": "La controparte dirà che il notaio fu convocato.", "risposta_cita_fatti": "nessuna risposta nei fatti", "cita_pronunce": []})])
    checked = S.verify(reply, inp)
    a = S.finish(checked, inp)
    assert checked["counters"]["strategy_objection_strongest"] == 1 and checked["counters"]["strategy_objection_dropped"] == 1
    items = S.objection_items({"status": S.OK, "analysis": a}, Z._view(inp))
    assert len(items) == 1
    assert items[0]["obiezione"] == ("La controparte negherà l'inadempimento. La controparte farà leva su: Farà leva sulla vendita a un "
                                     "terzo «Ettore Bassi sostiene di aver venduto a un terzo perché Anna Corsi non si è presentata dal notaio»")
    assert items[0]["risposta"] == f"{FACT2}."
    assert "accusa_forte" in S.prompt_system() and "strategy_objection_strongest" in R.STRATEGY_COUNTERS


# --- item 6 ---------------------------------------------------------------------------------------------------------------

def test_item6_a_general_part_ruling_is_kept_when_its_article_is_attached_to_an_element_of_the_line():
    els = [{"eid": "A590cp:E1", "elemento": "cagiona ad altri per colpa una lesione personale", "esito": "da_provare"}]
    assert A.general_part_attached("43 c.p.", els) is True          # the mental element
    assert A.general_part_attached("40 c.p.", els) is True          # causation («cagiona»)
    assert A.general_part_attached("56 c.p.", els) is False
    assert A.general_part_attached("43 c.p.", [{"elemento": "si appropria del denaro"}]) is False
    assert A.general_part_attached("9999 c.p.", els) is False
    verified = {"analysis": {"linea_principale": {"articolo": "590 c.p.", "stato": A.VERIFIED, "eid": "A590cp:E1"},
                             "elementi": {"590 c.p.": els}}}
    ruling = lambda art, passage: {"article": art, "estremi": f"Cass. su {art}", "passage": passage}      # noqa: E731
    ctx = {"rulings_record": {"kept": [ruling("590 c.p.", "la lesione colposa …"), ruling("43 c.p.", "la colpa generica …"),
                                       ruling("56 c.p.", "il tentativo …"), ruling("41 c.p.", "le concause …")],
                              "orientations": [], "landmarks": {}}}
    dropped = []
    out = M.rulings_view(ctx, ["590 c.p.", "43 c.p.", "56 c.p.", "41 c.p."], dropped, verified)
    kept = [r["article"] for r in out["rulings_record"]["kept"]]
    assert kept == ["590 c.p.", "43 c.p.", "41 c.p."]              # 43 and 41 attached (colpa, cagiona); 56 unattached, off the line
    assert ctx["rulings_general_part_kept_attached"] == 2 and ctx["rulings_general_part_dropped"] == 1
    assert dropped == ["56 c.p. (parte generale, passaggio non sulla linea)"]
    assert "rulings_general_part_kept_attached" in R.QA_COUNTERS


# --- item 7 ---------------------------------------------------------------------------------------------------------------

def test_item7_the_charges_capo_gives_the_alternative_candidates_deterministically():
    got = TG.capo_articles(("584", "c.p."))
    assert got[:4] == ["583 c.p.", "585 c.p.", "582 c.p.", "586 c.p."] and "584 c.p." not in got and len(got) == 18
    assert all(575 <= int(x.split()[0]) <= 593 for x in got)
    assert TG.capo_articles(("640", "c.p.")) == ["641 c.p.", "642 c.p.", "643 c.p.", "644 c.p.", "645 c.p.", "646 c.p.", "647 c.p.", "648 c.p.", "649 c.p."]
    assert len(TG.capo_articles(("660", "c.p."))) == TG.CAPO_ALTERNATIVES_MAX          # a long capo: the nearest 24
    assert TG.capo_articles(("43", "c.p.")) == [] and TG.capo_articles(("2043", "c.c.")) == []       # general parts
    assert TG.capo_articles(("1470", "c.c."))[:2] == ["1471 c.c.", "1472 c.c."]                       # the contracts' Capo
    assert TG.capo_articles(("x", "c.p.")) == [] and TG.capo_articles(()) == []
    assert flow.DECOMPOSE_WHY["capo"] == "alternativa dallo stesso capo del codice"


def test_item7_the_capo_rows_are_fetched_for_a_penal_charge_only_and_bind_as_alternatives(monkeypatch):
    from src.rag import defensive_generation as dg
    fetched = []

    def fake_fetch(session, refs, store):
        fetched.append(refs)
        return [{"_article": (n, c), "s": {"name": n, "plain_text": "x"}} for n, c in refs[:3]]
    monkeypatch.setattr(dg, "_fetch_articles", fake_fetch)
    got = flow.capo_alternatives(None, {"frame_rito": "penale", "frame_charge": "584 c.p."}, [{"_article": ("583", "c.p.")}], None)
    assert got["charge"] == "584 c.p." and got["articles"] == ["583 c.p.", "585 c.p.", "582 c.p."] and got["already_in_pool"] == ["583 c.p."]
    assert all(r["_capo_alternative"] == "584 c.p." for r in got["rows"]) and len(fetched[0]) == 18
    assert flow.capo_alternatives(None, {"frame_rito": "civile", "frame_charge": "1470 c.c."}, [], None) == {}
    assert flow.capo_alternatives(None, {"frame_rito": "penale", "frame_charge": ""}, [], None) == {}
    assert flow.capo_alternatives(None, {"frame_rito": "penale", "frame_charge": "73 d.P.R. 309/1990"}, [], None) == {}
    # the binding: a capo article is an «alternativa» whatever its verdict, never in the set; the charge itself is not
    monkeypatch.setenv("CASE_FRAME", "true")
    state = T173._state([T173._j("584 c.p.", governs=["pretesa"], has_elements=True, capo_alternative=True),
                         T173._j("583 c.p.", governs=["pretesa"], has_elements=True, capo_alternative=True),
                         T173._j("582 c.p.", fit="no", governs=[], has_elements=True, capo_alternative=True, alternative="c.p. … Capo I")],
                        frame_article="584 c.p.")
    b = flow.binding_pretesa(state)
    assert b["principale"] == ["584 c.p."] and b["alternative"] == ["583 c.p.", "582 c.p."]
    assert b["tiers"] == {"584 c.p.": "si_verificato", "582 c.p.": "alternativa", "583 c.p.": "alternativa"}
    assert "alternatives_from_capo" in R.FIT_COUNTERS
