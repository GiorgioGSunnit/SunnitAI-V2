"""v75: general rules from the live cases -- synthetic records and texts, one group per rule.

1. a civil claim naming no article: the claim record is the first read's principal hypothesis for the attore's claim
   (threat for a convenuto, weapon for an attore); a placeholder thesis is a failure state, never printed (the
   standard body, logged)
2. ratione temporis, stragiudiziale, mediazione and riti lines from the intake rito only, never the detected area
3. the ask and the requested blocks are never facts; abbreviations («Sig.») never end a sentence
4. a regulation needs its own domain signal (DORA: ICT risk, not «banca»); d.lgs. 11/2010 on the tier-1 list
5. candidate sets by role: persona offesa / parte civile, attore, convenuto, imputato
6. procedural asks resolve through the concepts table, never through a single-word rubrica hit
"""

import json
import os
import sys
from pathlib import Path

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from src.rag import cenni, intake, issue_records, lawyer_method as M, posture as posture_mod  # noqa: E402
from src.rag import render_computed, sentence_split, special_laws  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402

PO = {"rito": "penale", "family": "penale", "role": "persona_offesa", "stage": "indagini", "intake_rito": "penale"}
IMPUTATO = {"rito": "penale", "family": "penale", "role": "imputato", "stage": "primo_grado",
            "entry": posture_mod.entry_for("penale", "imputato", "primo_grado")}
ATTORE = {"rito": "civile", "family": "civile", "role": "attore", "stage": "pre_filing", "intake_rito": "civile"}
CONVENUTO = {"rito": "civile", "family": "civile", "role": "convenuto", "stage": "primo_grado", "intake_rito": "civile"}
SENTS = [{"id": "S1", "text": "Tizio ha trattenuto il denaro ricevuto da Caio.", "source": "caso",
          "certainty": "accertato"}]


def _req(i, text, status, kind="requisito"):
    return {"id": f"R{i}", "comma": 1, "kind": kind, "text": text, "status": status}


def _ctx(recs, posture, case="Tizio ha trattenuto il denaro di Caio.", circ_link=None):
    return M.prepare({"records": recs, "sentences": SENTS}, posture, [], case, "Caio", "", list(recs), [], [], [], [],
                     link_call=circ_link or (lambda *a: {}), case_id_="t75")


# --- 1. the civil claim from the first read; the placeholder never printed ----------------------------------------------

def _row(number, code, inst, rubrica=""):
    return {"_article": (number, code), "_first_read": inst, "s": {"plain_text": "testo", "titolo_articolo": rubrica}}


READ = {"hypotheses": [
    {"line": "principale", "claim": "Caio chiede il risarcimento", "institutions": ["inadempimento contrattuale",
                                                                                    "fatto illecito"]},
    {"line": "controparte", "claim": "Tizio chiede il risarcimento dei danni", "institutions": ["fatto illecito"]}]}


def test_a_civil_claim_naming_no_article_is_the_first_reads_principal_hypothesis():
    rows = [_row("1218", "c.c.", "inadempimento contrattuale", "Responsabilità del debitore"),
            _row("2043", "c.c.", "fatto illecito", "Risarcimento per fatto illecito"),
            _row("163", "c.p.c.", "inadempimento contrattuale", "Contenuto della citazione")]
    state = {"read": READ}
    assert posture_mod.claim_from_read(state, ATTORE, rows) == ["1218 c.c.", "2043 c.c."]     # the procedural row out
    assert posture_mod.claim_from_read(state, CONVENUTO, rows) == ["2043 c.c."]                 # the other side's line
    assert posture_mod.claim_from_read(state, PO, rows) == []
    assert posture_mod.claim_from_read({"read": {"hypotheses": []}}, ATTORE, rows) == []


@pytest.mark.parametrize("posture, side", [(CONVENUTO, "threat"), (ATTORE, "weapon")])
def test_the_claim_is_marked_threat_for_a_convenuto_and_weapon_for_an_attore(posture, side):
    state = {"text": "Tizio chiedeva il risarcimento dei danni.", "read": READ,
             "rows": [_row("2043", "c.c.", "fatto illecito", "Risarcimento per fatto illecito"),
                      _row("1218", "c.c.", "inadempimento contrattuale", "Responsabilità del debitore")],
             "records": {"2043 c.c.": {"rubrica": "Risarcimento per fatto illecito",
                                       "requisites": [_req(1, "il fatto doloso o colposo", "provato")]},
                         "1218 c.c.": {"rubrica": "Responsabilità del debitore",
                                       "requisites": [_req(2, "l'inadempimento", "provato")]}}}
    log = posture_mod.mark_sides(state, posture)
    first = log["charge_from"]["first_read"][0]
    assert state["records"][first]["charged"] == 1 and state["records"][first]["side"] == side
    assert log["charge_from"]["case"] == [] and log["charge_from"]["intake"] == []


def test_the_decomposition_takes_the_claim_from_the_first_read():
    t2043 = ("Qualunque fatto doloso o colposo, che cagiona ad altri un danno ingiusto, obbliga colui che ha commesso il "
             "fatto a risarcire il danno.")
    row = {"_article": ("2043", "c.c."), "_first_read": "fatto illecito",
           "s": {"plain_text": t2043, "titolo_articolo": "Risarcimento per fatto illecito"}}
    state = {"analysis": {}, "text": "Tizio chiedeva il risarcimento dei danni.", "request": "", "intake": {},
             "read": READ, "records": {}, "sentences": [], "questions": []}
    link, _ = flow.decompose_records(state, CONVENUTO, [row], set(), fetch=lambda refs: [])
    assert link == ["2043 c.c."] and state["records"]["2043 c.c."]["decomposed_for"] == flow.DECOMPOSE_WHY["claim"]
    assert issue_records.decomposed(state["records"]["2043 c.c."])


def test_a_one_sentence_civil_fattispecie_is_decomposed():
    from src.rag import rule_quote
    assert rule_quote.civil_elements("Qualunque fatto doloso o colposo, che cagiona ad altri un danno ingiusto, obbliga "
                                     "colui che ha commesso il fatto a risarcire il danno.") == \
        ["fatto doloso o colposo", "cagiona ad altri un danno ingiusto"]
    assert rule_quote.civil_elements("Il debitore che non esegue esattamente la prestazione dovuta è tenuto al "
                                     "risarcimento del danno, se non prova che l'inadempimento …") == \
        ["il debitore", "non esegue esattamente la prestazione dovuta", "il danno"]     # v76b: the subject's quality
    assert rule_quote.civil_elements("Ciascuno è responsabile del danno cagionato dalle cose che ha in custodia, salvo "
                                     "che provi il caso fortuito.") == ["danno cagionato dalle cose", "ha in custodia"]
    assert rule_quote.civil_elements("Il proprietario ha diritto di godere e disporre delle cose in modo pieno.") == []


def test_a_placeholder_main_line_is_never_printed():
    recs = {"1062 c.c.": {"rubrica": "Destinazione del padre di famiglia", "side": "weapon",
                          "requisites": [_req(1, "le opere visibili", "non risulta")]}}      # v75b: excluded -- no line
    ctx = _ctx(recs, ATTORE)
    plan = M.clean_method({}, M.thesis_candidates(ctx))
    assert M.placeholder_main(ctx, plan) == "Tesi da costruire sugli elementi ancora da provare"
    log = {}
    assert M.compose([("Norme applicate ai fatti", "regola")], ctx, plan, log=log) == ""
    assert log["refused"]["why"] == "placeholder"


def test_the_flow_writes_the_standard_body_when_the_method_refuses(monkeypatch):
    import test_v60g
    from test_v60f import _graded_state
    state = _graded_state(monkeypatch)
    state["mode"] = "single_shot"
    monkeypatch.setenv("LAWYER_METHOD", "true")
    monkeypatch.setattr(M, "objections_plan", lambda ctx, plan: plan)
    # v79.6 (item 5): the role is never left unknown (a reading gives the attore's «provando …» line): the
    # placeholder path is kept by a record with no candidate at all
    monkeypatch.setattr(M, "method_plan", lambda ctx, call=None: M.clean_method({}, []))
    events = []
    real = flow._log
    monkeypatch.setattr(flow, "_log", lambda e, d, ms=None: (events.append((e, d)), real(e, d, ms)))
    seen = {}
    text, _ = test_v60g._answer(monkeypatch, state, {}, None, seen)        # nothing proven: no line for the attore
    assert state["method_refused"] == M.NOT_APPLIED_WHY["placeholder"]                 # v76: the reason, said
    assert ("lawyer_method_refused", {"why": "placeholder", "tesi": "Tesi da costruire sugli elementi ancora da "
                                                                     "provare"}) in events
    assert "Tesi da costruire" not in text and "**1. Sintesi della soluzione**" not in text
    assert M.not_applied_note(state["method_refused"]) in text
    assert len(seen["jobs"][0]) > 1 and M.current_rules() == ""            # the standard jobs, his rules off


# --- 2. the tables by the intake rito ---------------------------------------------------------------------------------------

def test_the_tables_follow_the_intake_rito_not_the_detected_area():
    civil = {"rito": "civile", "family": "civile", "role": "imputato", "stage": "pre_filing", "intake_rito": "civile"}
    line, _ = M.temporal("Il 3 marzo 2025 Caio riceveva una mail di phishing.", civil)
    assert "Cartabia penale" not in line and "Codice rosso" not in line and "penale" not in line.lower()
    assert "querela" not in M.stragiudiziale_line(civil) and M.stragiudiziale_line(civil) == \
        M.data()["stragiudiziale"]["civile"]
    assert M.riti_lines(civil) == []                         # an «imputato» read off the penal area: no riti
    unknown = {**civil, "intake_rito": ""}
    assert M.stragiudiziale_line(unknown) == "" and posture_mod.intake_family(unknown) == ""
    assert render_computed.mediation_rito(ATTORE) and not render_computed.mediation_rito(PO)
    assert not render_computed.mediation_rito({**ATTORE, "intake_rito": "penale"})


def test_the_posture_keeps_the_rito_intake_read(monkeypatch):
    monkeypatch.setattr(flow, "frame_families", lambda analysis: ["penale"])
    read = {"rito": "civile", "role": "attore", "stage": "pre_filing", "confidence": {}}
    out = posture_mod.of({"intake": read, "text": "phishing"}, {})
    assert out["intake_rito"] == "civile" and posture_mod.intake_family(out) == "civile"
    out = posture_mod.of({"intake": {"rito": "sconosciuto", "role": "attore", "stage": "", "confidence": {}}}, {})
    assert out["intake_rito"] == ""


# --- 3. the ask is never a fact ---------------------------------------------------------------------------------------------

ASKED = ("Caio riceveva una mail dal Sig. Tizio, che si qualificava come operatore della banca. Tizio chiedeva il "
         "risarcimento dei danni. Assistiamo il Sig. Caio nella predisposizione della difesa. Si chiede: 1) Indicare le "
         "azioni esperibili; 2) valutare la responsabilità della banca.\n3) Quali sono i termini?")


def test_abbreviations_never_end_a_sentence():
    texts = [s["text"] for s in issue_records.sentences(ASKED)]
    assert texts == ["Caio riceveva una mail dal Sig. Tizio, che si qualificava come operatore della banca.",
                     "Tizio chiedeva il risarcimento dei danni."]
    assert intake.sentences("Il Dott. Rossi e l'Avv. Bianchi. Alfa S.p.A. Il giorno dopo.") == \
        ["Il Dott. Rossi e l'Avv. Bianchi.", "Alfa S.p.A. Il giorno dopo."]
    assert sentence_split.rejoin(["Il Sig.", "Caio paga.", "Fine."]) == ["Il Sig. Caio paga.", "Fine."]


def test_the_requested_blocks_are_never_facts():
    text = "Caio ha versato 5.000 euro a Tizio. Si chiede in particolare:\n- la responsabilità della banca;\n- i termini."
    asks = [b["subject"] for b in intake.requested_blocks(text) if b["kind"] == "quesito"]
    got = [s["text"] for s in issue_records.sentences(text, asks=asks)]
    assert got == ["Caio ha versato 5.000 euro a Tizio."]
    state = {"text": ASKED, "request": "", "analysis": {}, "intake": {"requested_blocks": []}}
    facts = M.facts_split(flow.case_sentences(state), {}, False)
    assert all("Assistiamo" not in x and "Indicare" not in x and "Caio nella predisposizione" not in x
               for x in facts["certi"])


# --- 4. a regulation needs its own signal -----------------------------------------------------------------------------------

def test_a_regulation_needs_its_own_domain_signal():
    dora = {"_article": ("143", "Reg. 2022/2554"), "d": {"name": "Regolamento (UE) 2022/2554 (DORA)"}}
    bank = "Caio ha ricevuto una mail che sembrava della sua banca e ha disposto un bonifico."
    assert special_laws.gate([dora], bank, set())[0] == []                            # the sector's word is not its own
    assert special_laws.gate([dora], "La banca non ha gestito il rischio ICT del fornitore.", set())[0] == [dora]
    other = {"_article": ("5", "Reg. 2019/1150"), "d": {"name": "Regolamento (UE) 2019/1150"}}
    kept, dropped = special_laws.gate([other], bank, set())
    assert kept == [] and special_laws.UNLISTED_REGULATION in dropped[0]
    assert special_laws.gate([other], bank, {("5", "Reg. 2019/1150")})[0] == [other]   # the user names it
    gdpr = {"_article": ("82", "Reg. 2016/679"), "d": {"name": "GDPR"}}
    assert special_laws.gate([gdpr], "Il trattamento dei dati personali di Caio.", set())[0] == [gdpr]
    psd = {"_article": ("12", "d.lgs. 11/2010")}
    assert special_laws.law_of("d.lgs. 11/2010")["id"] == "dlgs_11_2010"
    assert special_laws.gate([psd], bank + " Era phishing.", set())[0] == [psd]
    tier1 = json.loads((ROOT / "data" / "ingestion_tier1.json").read_text(encoding="utf-8"))
    assert [d["id"] for d in tier1["documents"]][0] == "dlgs_11_2010"


# --- 5. candidate sets by role ----------------------------------------------------------------------------------------------

def _646(status="provato"):
    return {"646 c.p.": {"rubrica": "Appropriazione indebita", "side": "weapon", "charged": 1,
                         "requisites": [_req(i, t, status) for i, t in enumerate(
                             ["il possesso del denaro altrui", "l'appropriazione", "il fine di profitto",
                              "l'ingiusto profitto"], 1)]}}


def test_the_victim_founds_the_claim_on_the_charge_and_never_gets_the_offenders_attenuanti():
    def link(topic, reqs, sents):
        return {r["id"]: {"sostiene": ["S1"]} for _, r in reqs}
    ctx = _ctx(_646(), PO, case="Tizio ha trattenuto il denaro di Caio e lo ha risarcito in parte; l'ha fatto con "
                                "abuso di relazioni d'ufficio.", circ_link=link)
    cands = M.thesis_candidates(ctx)
    kinds = [c["kind"] for c in cands]
    assert kinds[:3] == ["claim_offence", "parte_civile", "sequestro"]
    assert "attenuante" not in kinds and "scriminante" not in kinds and "contenimento" not in kinds
    assert cands[0]["tesi"] == "Fondare la domanda sul reato contestato: art. 646 c.p. (Appropriazione indebita)"
    assert all(c["argued"] == "code" for c in cands)
    plan = M.clean_method({}, cands)
    sols, main, _ = M.selection(ctx, plan)
    assert sols[main]["kind"] == "claim_offence"
    assert "La difesa dell'imputato" in cands[0]["code_argument"]["contro"] or \
        cands[0]["code_argument"]["contro"].startswith("Restano")


def test_the_victims_aggravanti_are_its_weapons():
    circ = {"61 n. 11 c.p.": {"kind": "aggravante", "requisites": [_req(30, "l'abuso di relazioni", "provato")]},
            "62 n. 6 c.p.": {"kind": "attenuante", "requisites": [_req(31, "la riparazione del danno", "provato")]}}
    ctx = {"records": _646(), "circumstances": circ, "posture": PO, "assessment": {}, "case": ""}
    kinds = [c["kind"] for c in M.thesis_candidates(ctx)]
    assert "aggravante" in kinds and "attenuante" not in kinds
    imputato = {**ctx, "posture": IMPUTATO, "records": {"646 c.p.": {**_646()["646 c.p."], "side": "threat"}},
                "assessment": {"article": "646 c.p."}}
    assert "attenuante" in [c["kind"] for c in M.thesis_candidates(imputato)]         # the imputato's, as before


def test_the_attore_gets_its_articles_and_the_matters_remedies():
    recs = {"2043 c.c.": {"rubrica": "Risarcimento per fatto illecito", "side": "weapon",
                          "requisites": [_req(1, "il fatto colposo", "provato"), _req(2, "il danno", "da provare")]}}
    cands = M.thesis_candidates({"records": recs, "circumstances": {}, "posture": ATTORE, "assessment": {}})
    assert [c["kind"] for c in cands] == ["claim", "remedy", "remedy"]
    assert cands[1]["tesi"] == "Chiedere il risarcimento del danno patrimoniale e non patrimoniale (artt. 2043, 2056 " \
                               "e 2059 c.c.)"
    arg = cands[1]["code_argument"]
    assert arg["argomento"].startswith("Il risarcimento del danno patrimoniale") and arg["realismo"] == "medio"


def test_the_convenuto_contests_the_claim_and_gets_the_eccezioni_and_the_riconvenzionale():
    recs = {"1218 c.c.": {"rubrica": "Responsabilità del debitore", "side": "threat", "charged": 1,
                          "requisites": [_req(1, "l'inadempimento", "da provare"), _req(2, "il danno", "provato")]},
            "1460 c.c.": {"rubrica": "Eccezione d'inadempimento", "side": "weapon",
                          "requisites": [_req(3, "l'inadempimento della controparte", "provato")]},
            # v79.8 (B1): the riconvenzionale on a norm conferring an action (1460 is an eccezione, never a claim)
            "1453 c.c.": {"rubrica": "Risolubilità del contratto per inadempimento", "side": "weapon",
                          "requisites": [_req(4, "l'inadempimento di non scarsa importanza", "provato")]}}
    cands = M.thesis_candidates({"records": recs, "circumstances": {}, "posture": CONVENUTO,
                                 "assessment": {"article": "1218 c.c."}})
    kinds = [c["kind"] for c in cands]
    assert kinds[0] == "contest_open" and "eccezione" in kinds and "riconvenzionale" in kinds
    assert "claim" not in kinds and "remedy" not in kinds and "transazione" in kinds
    assert [c["article"] for c in cands if c["kind"] == "riconvenzionale"] == ["1453 c.c."]
    ecc = next(c for c in cands if c["kind"] == "eccezione")
    assert ecc["code_argument"]["realismo"] == "basso" and "richiede, nei fatti" in ecc["code_argument"]["argomento"]


# --- 6. procedural asks through the concepts table --------------------------------------------------------------------------

def test_procedural_asks_resolve_through_the_concepts_table():
    assert cenni.procedural_refs("sequestro") == ("sequestro", [("253", "c.p.p."), ("321", "c.p.p."),
                                                                ("262", "c.p.p.")])
    assert cenni.procedural_refs("costituirsi parte civile")[1] == [(str(n), "c.p.p.") for n in range(74, 80)]
    assert cenni.procedural_refs("sequestro di persona") == ("", [])
    assert cenni.procedural_refs("appropriazione indebita") == ("", [])


def test_the_first_read_search_never_takes_a_procedural_ask_through_a_rubrica_hit():
    asked = []

    def resolve(own, documents, query):
        asked.append(query["nome"])
        hit = {"sequestro": ("474-bis", "c.p."), "costituirsi parte civile": ("597", "c.p."),
               "appropriazione indebita": ("646", "c.p.")}.get(query["nome"])
        return [{"_article": hit, "s": {}}] if hit and query["codice"] == "c.p." else []

    def fetch(refs):
        return [{"_article": ref, "s": {"plain_text": "testo"}} for ref in refs]
    rows, log = flow.first_read_search(None, "db", [], ["appropriazione indebita", "sequestro",
                                                        "costituirsi parte civile"], ["c.p."], session=object(),
                                       resolve=resolve, text_search=lambda *a: [], index_search=lambda *a: [],
                                       concept_fetch=fetch)
    labels = {flow._label(flow._key(r["_article"])): r for r in rows}
    assert "474-bis c.p." not in labels and "597 c.p." not in labels and "646 c.p." in labels
    assert {"253 c.p.p.", "321 c.p.p.", "262 c.p.p.", "74 c.p.p.", "79 c.p.p."} <= set(labels)
    assert labels["253 c.p.p."]["_first_read_channel"] == "concetto" and "sequestro" not in asked
    assert log["concepts"]["sequestro"] == ["253 c.p.p.", "321 c.p.p.", "262 c.p.p."]
