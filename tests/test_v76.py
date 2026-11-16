"""v76: general rules -- synthetic records and texts, one group per rule.

1. a status needs a sentence
2. the convenuto's claim is what the citazione pleads
3. containment is penal-only; the convenuto's fallback from the remedies table; no penal line under a civil rito
4. candidates only from remedy-conferring fattispecie; named articles always candidates when decomposed; 1490/1497
5. selection by realism, then proven count
6. the method never steps aside silently: the general penal splitter, then the note
7. concepts for the protection asks
8. the leftovers: d'ufficio conditions on decomposed records, the model's added theses, Altri profili by Titolo, the
   giudice di pace by the exact article
"""

import os
import sys
from pathlib import Path

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.rag import cenni, issue_records, lawyer_method as M, posture as posture_mod, procedural_facts  # noqa: E402
from src.rag import rule_quote  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402

CONVENUTO = {"rito": "civile", "family": "civile", "role": "convenuto", "stage": "primo_grado", "intake_rito": "civile"}
ATTORE = {**CONVENUTO, "role": "attore", "stage": "pre_filing"}
IMPUTATO = {"rito": "penale", "family": "penale", "role": "imputato", "stage": "primo_grado", "intake_rito": "penale",
            "entry": posture_mod.entry_for("penale", "imputato", "primo_grado")}
SENTS = [{"id": "S1", "text": "Tizio vende a Caio un'auto che si rivela difettosa.", "source": "caso",
          "certainty": "accertato"}]
T1490 = ("Il venditore è tenuto a garantire che la cosa venduta sia immune da vizi che la rendano inidonea all'uso a cui "
         "è destinata o ne diminuiscano in modo apprezzabile il valore.")
T1497 = ("Quando la cosa venduta non ha le qualità promesse ovvero quelle essenziali per l'uso a cui è destinata, il "
         "compratore ha diritto di ottenere la risoluzione del contratto secondo le disposizioni generali sulla "
         "risoluzione per inadempimento, purché il difetto di qualità ecceda i limiti di tolleranza stabiliti dagli usi.")
T612BIS = ("Salvo che il fatto costituisca più grave reato, è punito con la reclusione da un anno a sei anni e sei mesi "
           "chiunque, con condotte reiterate, minaccia o molesta taluno in modo da cagionare un perdurante e grave stato "
           "di ansia o di paura ovvero da ingenerare un fondato timore per l'incolumità propria.")


def _req(i, text, status, **extra):
    return {"id": f"R{i}", "comma": 1, "kind": "requisito", "text": text, "status": status, **extra}


def _ctx(recs, posture, circ_link=None):
    return M.prepare({"records": recs, "sentences": SENTS}, posture, [], "Tizio vende a Caio un'auto difettosa.", "Caio",
                     "", list(recs), [], [], [], [], link_call=circ_link or (lambda *a: {}), case_id_="t76")


# --- 1. a status needs a sentence ---------------------------------------------------------------------------------------------

def test_a_status_needs_a_sentence():
    by_id = {"S1": {"id": "S1", "source": "caso", "certainty": "accertato", "text": "Domitilla ha chiuso il cancello."},
             "S2": {"id": "S2", "source": "caso", "certainty": "accertato",
                    "text": "Mevio separò i due fondi con un atto di divisione."}}
    req = lambda **kw: {"kind": "requisito", "supporting": [], "contradicting": [], **kw}
    alone = req(facts_status="provato", text="l'apparenza delle opere visibili")
    issue_records.settle(alone, by_id)
    assert alone["status"] == "da provare" and "nessuna frase" in alone["why"]
    triggered = req(facts_status="provato", text="due fondi attualmente divisi")
    issue_records.settle(triggered, by_id)
    assert triggered["status"] == "provato" and triggered["supporting"] == ["S2"]
    excluded = req(facts_status="non risulta", text="l'apparenza delle opere visibili")
    issue_records.settle(excluded, by_id)
    assert excluded["status"] == "da provare"
    comma = req(kind="comma", facts_status="provato", text="Chiunque …")
    issue_records.settle(comma, by_id)
    assert comma["status"] == "da provare"


# --- 2. the pleaded claim ---------------------------------------------------------------------------------------------------

READ = {"hypotheses": [
    {"line": "controparte", "claim": "Sempronio chiede il risarcimento per fatto illecito",
     "institutions": ["fatto illecito"]},
    {"line": "subordinata", "claim": "Garanzia per i vizi della cosa venduta e risoluzione della vendita",
     "institutions": ["garanzia per i vizi"]}]}
ROWS = [{"_article": ("2043", "c.c."), "_first_read": "fatto illecito", "s": {"titolo_articolo": "Fatto illecito"}},
        {"_article": ("1490", "c.c."), "_first_read": "garanzia per i vizi",
         "s": {"titolo_articolo": "Garanzia per i vizi della cosa venduta"}}]


def test_the_convenutos_claim_is_the_pleaded_one():
    state = {"text": "Con atto di citazione Sempronio chiedeva al Tribunale di dichiarare la risoluzione della vendita "
                     "per i vizi della cosa venduta e di condannare Caio alla restituzione del prezzo.", "read": READ}
    log = {}
    assert posture_mod.claim_from_read(state, CONVENUTO, ROWS, log) == ["1490 c.c."]
    assert log["pleaded"].startswith("Con atto di citazione") and "fatto illecito" in log["unmatched_hypothesis"]
    nothing = {"text": "Sempronio chiedeva al Tribunale di accertare la nullità del testamento.", "read": READ}
    assert posture_mod.claim_from_read(nothing, CONVENUTO, ROWS) == []         # no hypothesis matches: no claim
    unpleaded = {"text": "Sempronio lamenta un danno.", "read": READ}
    assert posture_mod.claim_from_read(unpleaded, CONVENUTO, ROWS) == ["2043 c.c."]     # no citazione: the read's line


# --- 3. containment is penal-only -------------------------------------------------------------------------------------------

def test_the_civil_convenutos_fallback_is_never_the_penal_containment():
    a = {"all_integrated": True, "obiettivo": M.CONTENIMENTO, "article": "2043 c.c."}
    t = M.fallback_thesis(a, [], CONVENUTO)
    assert t["tesi"] == "Contestare la domanda o, in alternativa, transigere" and "pena" not in t["tesi"]
    assert M.fallback_thesis(a, [], IMPUTATO)["tesi"] != t["tesi"]


def test_no_penal_candidate_under_a_civil_intake_rito():
    civil_imputato = {**IMPUTATO, "intake_rito": "civile"}         # an «imputato» read off the penal area
    recs = {"584 c.p.": {"rubrica": "Omicidio preterintenzionale", "side": "threat", "charged": 1,
                         "requisites": [_req(1, "atti diretti a ledere", "provato"), _req(2, "la morte", "provato")]}}
    circ = {"62 n. 6 c.p.": {"kind": "attenuante", "requisites": [_req(30, "la riparazione", "provato")]}}
    ctx = {"records": recs, "circumstances": circ, "posture": civil_imputato,
           "assessment": {"article": "584 c.p.", "all_integrated": True}}
    kinds = [c["kind"] for c in M.thesis_candidates(ctx)]
    assert not set(kinds) & set(M.PENAL_KINDS)
    assert "attenuante" in [c["kind"] for c in M.thesis_candidates({**ctx, "posture": IMPUTATO})]


# --- 4. remedy-conferring fattispecie only ----------------------------------------------------------------------------------

def test_time_limit_and_proof_articles_are_never_fattispecie():
    for label, rubrica in (("1495 c.c.", "Termini e condizioni per l'azione"), ("2697 c.c.", "Onere della prova"),
                           ("2946 c.c.", "Prescrizione ordinaria"), ("2964 c.c.", "Decadenza")):
        assert not issue_records.counted(label, rubrica), label
    assert issue_records.counted("1490 c.c.", "Garanzia per i vizi della cosa venduta")


def test_the_sale_guarantees_decompose_without_cut_letters():
    assert rule_quote.civil_elements(T1490) == [
        "la cosa venduta sia immune da vizi",
        "la rendano inidonea all'uso a cui è destinata o ne diminuiscano in modo apprezzabile il valore"]
    assert [r["text"] for r in issue_records._requisites({"_article": ("1497", "c.c."), "s": {"plain_text": T1497}},
                                                        {})] == [
        "la cosa venduta non ha le qualità promesse ovvero quelle essenziali per l'uso a cui è destinata",
        "il difetto di qualità ecceda i limiti di tolleranza stabiliti dagli usi"]
    assert rule_quote._SUBJECT_HEAD_RE.sub("", "imputato che agisce") == "imputato che agisce"
    assert rule_quote._SUBJECT_HEAD_RE.sub("", "lesione grave") == "lesione grave"
    assert rule_quote._SUBJECT_HEAD_RE.sub("", "il debitore") == "debitore"


def test_a_named_decomposed_article_is_always_a_candidate():
    recs = {"1490 c.c.": {"rubrica": "Garanzia per i vizi della cosa venduta", "side": "weapon", "named_due": True,
                          "requisites": [_req(1, "la cosa venduta sia immune da vizi", "da provare")]}}
    for posture, kind in ((CONVENUTO, "riconvenzionale"), (ATTORE, "claim")):
        cands = M.thesis_candidates({"records": recs, "circumstances": {}, "posture": posture, "assessment": {}})
        assert cands and cands[0]["kind"] == kind and "provando «la cosa venduta sia immune da vizi» (R1)" in \
            cands[0]["tesi"], posture["role"]


# --- 5. selection by realism, then proven count ------------------------------------------------------------------------------

def test_a_zero_proven_line_never_leads_over_an_alto_one():
    recs = {"1490 c.c.": {"rubrica": "Garanzia", "side": "weapon",
                          "requisites": [_req(1, "a", "provato"), _req(2, "b", "provato"), _req(3, "c", "da provare")]}}
    sols = [{"tesi": "T1", "candidate": "T1", "realismo": "medio", "poggia_su": ["R3"], "contesta": []},
            {"tesi": "T2", "candidate": "T2", "realismo": "alto", "poggia_su": ["R1"], "contesta": []},
            {"tesi": "T3", "candidate": "T3", "realismo": "alto", "poggia_su": ["R1", "R2"], "contesta": []}]
    got, main = M.coherent_solutions({"soluzioni": sols}, recs, {}, ())
    assert got[main]["tesi"] == "T3"                         # alto first, then the more proven


# --- 6. never stepping aside silently --------------------------------------------------------------------------------------

def test_the_general_penal_splitter_takes_the_clause_after_chiunque():
    assert rule_quote.offence_elements(T612BIS) == []          # v76b: a clausola di riserva is no element
    reqs = issue_records._requisites({"_article": ("612-bis", "c.p."), "s": {"plain_text": T612BIS}}, {})
    assert [r["text"] for r in reqs] == ["con condotte reiterate", "minaccia o molesta taluno",
                                         "in modo da cagionare un perdurante e grave stato di ansia o di paura ovvero da "
                                         "ingenerare un fondato timore per l'incolumità propria"]
    assert all(r["kind"] == "requisito" and r["split"] == "clausole penali" for r in reqs)
    one = issue_records._requisites({"_article": ("612", "c.p."), "s": {"plain_text": "Chiunque minaccia ad altri un "
                                     "ingiusto danno è punito, a querela della persona offesa, con la multa."}}, {})
    assert [r["text"] for r in one] == ["minaccia ad altri un ingiusto danno"]


def test_an_undecomposable_charge_refuses_the_method_and_the_answer_says_so():
    row = {"_article": ("999", "c.p."), "s": {"plain_text": "Le pene stabilite nell'articolo precedente si applicano "
                                              "anche a chi …", "titolo_articolo": "Pena"}}
    state = {"analysis": {}, "text": "Il PM contesta a Tizio l'art. 999 c.p.", "request": "", "intake": {},
             "records": {}, "sentences": [], "questions": []}
    flow.decompose_records(state, IMPUTATO, [row], set(), fetch=lambda refs: [])
    assert state["method_refused"] == M.NOT_APPLIED_WHY["undecomposed"].format(art="999 c.p.")
    note = M.not_applied_note(state["method_refused"])
    assert note.startswith("Nota: il metodo di analisi per elementi non è stato applicato a questo parere: l'art. 999")
    unread = {**state, "records": {}}
    unread.pop("method_refused")
    flow.decompose_records(unread, IMPUTATO, [], set(), fetch=lambda refs: [])
    assert unread["method_refused"] == M.NOT_APPLIED_WHY["unread"].format(art="999 c.p.")


# --- 7. the protection asks -------------------------------------------------------------------------------------------------

def test_protection_asks_resolve_through_the_concepts_table():
    assert cenni.procedural_refs("divieto di avvicinamento alla persona offesa")[1] == [("282-ter", "c.p.p.")]
    assert cenni.procedural_refs("allontanamento d'urgenza dalla casa familiare")[1] == [("384-bis", "c.p.p.")]
    assert cenni.procedural_refs("misure cautelari")[1] == [(str(n), "c.p.p.") for n in (272, 273, 274, 275)]
    assert cenni.procedural_refs("ammonimento del questore")[1] == [("8", "d.l. 11/2009")]


# --- 8. the leftovers -------------------------------------------------------------------------------------------------------

def test_a_decomposed_charge_carries_its_dufficio_conditions():
    t612 = ("Chiunque minaccia ad altri un ingiusto danno è punito, a querela della persona offesa, con la multa fino a "
            "euro 1.032.\nSe la minaccia è grave, o è fatta in uno dei modi indicati nell'articolo 339, la pena è della "
            "reclusione fino a un anno.\nSi procede d'ufficio se la minaccia è fatta in uno dei modi indicati "
            "nell'articolo 339.")
    row = {"_article": ("612", "c.p."), "s": {"plain_text": t612, "titolo_articolo": "Minaccia"},
           "_proc": {"querela": True, "ufficio_condizioni": [{"comma": 3, "text": "la minaccia è fatta in uno dei "
                                                                                  "modi indicati nell'articolo 339"}]}}
    state = {"analysis": {}, "text": "Il PM contesta a Tizio l'art. 612 c.p.", "request": "", "intake": {},
             "records": {}, "sentences": [], "questions": []}
    flow.decompose_records(state, IMPUTATO, [row], set(), fetch=lambda refs: [])
    conds = state["records"]["612 c.p."]["procedibilita"]
    assert conds and conds[0]["kind"] == "procedibilita" and conds[0]["status"] == "da provare" and conds[0]["id"]


def test_the_models_added_theses_are_never_displayed():
    recs = {"584 c.p.": {"rubrica": "Omicidio preterintenzionale", "side": "threat", "charged": 1,
                         "requisites": [_req(1, "atti diretti a ledere", "provato"), _req(2, "la morte", "da provare")]}}
    ctx = _ctx(recs, IMPUTATO)
    plan = M.clean_method({"stragiudiziale": "Proporre al PM una tesi nuova sulla legittima difesa."},
                          M.thesis_candidates(ctx))
    plan["soluzioni"].append({"tesi": "Legittima difesa (tesi del modello)", "pro": "", "contro": "",
                              "realismo": "basso", "contesta": ["R2"], "poggia_su": []})
    log = {}
    answer = M.compose([("Norme applicate ai fatti", "regola")], ctx, plan, log=log)
    assert "Legittima difesa" not in answer and "tesi nuova" not in answer
    assert log["added_theses_dropped"] == ["Legittima difesa (tesi del modello)"]
    assert log["stragiudiziale_model_dropped"].startswith("Proporre al PM")


def test_other_profiles_split_libro_iv_by_titolo():
    assert flow.profile_book("1470 c.c.") == ("c.c.", 6) and flow.profile_book("2028 c.c.") == ("c.c.", 9)
    assert flow.profile_book("2043 c.c.") == ("c.c.", 12) and flow.profile_book("2170 c.c.") == ("c.c.", 13)


def test_the_giudice_di_pace_by_the_exact_article():
    assert procedural_facts.competenza("612", 1, False).startswith("giudice di pace")
    assert not procedural_facts.competenza("612-bis", 6.5, False).startswith("giudice di pace")
    assert not procedural_facts.competenza("582-bis", 3, False).startswith("giudice di pace")
