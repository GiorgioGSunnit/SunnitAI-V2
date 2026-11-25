"""v79 block C (79.2) -- charge and role, on synthetic sentences only:

C1. mark_sides: the first charged article stays the charge even when the first read lists it among the alternatives; a
    second charged article that is also the client's alternative keeps the old behaviour (the client's line)
C2. the charge wordings, from data/charge_wordings.json: «arrestato (in flagranza) per», «giudicato responsabile»,
    «condannato per», «rinviato a giudizio per», «imputato del delitto di», «tratto in arresto per»
C3. a charge under a special law of data/special_laws.json (art. 73 d.P.R. 309/1990) is a charge like a code article
C4. a criminal case with no stated charge: the first read's main hypothesis, labelled «ipotesi di reato (addebito non
    indicato nel quesito)» in §1 and §9; the «registro non contiene l'addebito» sentence never printed
C5. the model's «convenuto» only when the text shows someone suing, serving or prosecuting the client, else «attore»;
    «Assisto un/una/il/la/l' <noun>»; no «Non mi hai indicato chi assisti» before a parere written anyway
"""

import pytest


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for name, value in (("LLM_API_KEY", "k"), ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "k"), ("NEO4J_URI", "bolt://localhost:7687"), ("NEO4J_USER", "n"),
                        ("NEO4J_PASSWORD", "x")):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))


def _req(i, text, status="provato"):
    return {"id": f"R{i}", "comma": 1, "kind": "requisito", "text": text, "status": status}


# --- C2 ------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sentence", [
    "Ieri Alfa è stato arrestato in flagranza per l'art. 624 c.p.",
    "Alfa è stato arrestato per l'art. 624 c.p. mentre usciva dal negozio.",
    "Il tribunale ha giudicato responsabile Alfa dell'art. 624 c.p.",
    "Alfa è stato condannato per l'art. 624 c.p. in primo grado.",
    "Alfa è stato rinviato a giudizio per l'art. 624 c.p.",
    "Alfa è imputato del delitto di cui all'art. 624 c.p.",
    "Alfa è stato tratto in arresto per l'art. 624 c.p.",
])
def test_each_charge_wording_marks_the_charge(sentence):
    from src.rag import posture
    assert posture.charge_labels(f"Alfa vive a Roma. {sentence} Beta è un suo vicino.") == ["624 c.p."]


def test_the_wordings_are_in_data_not_code():
    import json
    from pathlib import Path
    from src.rag import posture
    data = json.loads((Path(posture.__file__).resolve().parents[2] / "data" / "charge_wordings.json").read_text())
    assert len(data["wordings"]) >= 15 and posture.charge_re().pattern.count("|") >= len(data["wordings"]) - 1
    assert posture.charge_labels("Alfa vive a Roma. Alfa ha letto l'art. 624 c.p. su un giornale.") == []


# --- C3 ------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sentence, label", [
    ("Alfa è imputato per l'art. 73 d.P.R. 309/1990.", "73 d.P.R. 309/1990"),
    ("Alfa è imputato del reato di cui all'art. 73, comma 5, del d.P.R. n. 309 del 1990.", "73 d.P.R. 309/1990"),
    ("Il PM contesta ad Alfa l'art. 55 del d.lgs. 81/2008.", "55 d.lgs. 81/2008"),
])
def test_a_special_law_charge_is_read(sentence, label):
    from src.rag import posture
    assert posture.charge_labels(sentence) == [label]


def test_an_act_outside_the_special_laws_table_is_not_a_charge():
    from src.rag import posture
    assert posture.charge_labels("Il PM contesta ad Alfa l'art. 3 della legge 999/2031.") == []


def test_a_special_law_charge_gets_a_record_under_the_stores_name(monkeypatch):
    from src.rag import interactive_flow as flow, issue_records, posture
    store_name = "Testo Unico sugli Stupefacenti 2026"             # v79.10: the archive's exact name
    row = {"_article": ("73", store_name), "d": {"id": "TU", "name": store_name},
           "s": {"name": "73", "rubrica": "Produzione, traffico e detenzione illeciti di sostanze stupefacenti",
                 "plain_text": "Chiunque, senza l'autorizzazione, detiene sostanze stupefacenti è punito con la "
                               "reclusione da sei a venti anni."}}
    state = {"records": {}, "analysis": {}, "case": "Alfa è imputato per l'art. 73 d.P.R. 309/1990.", "rows": []}
    monkeypatch.setattr(flow, "case_text", lambda s: s["case"])
    monkeypatch.setattr(flow, "_restored_analysis", lambda a: {})
    monkeypatch.setattr(flow, "user_named", lambda s: [])
    monkeypatch.setattr(posture, "alternative_labels", lambda s, rows=(): [])
    monkeypatch.setattr(issue_records, "_requisites", lambda r, a, *_: [
        {"id": "", "comma": 1, "kind": "requisito", "text": "detiene sostanze stupefacenti", "status": "da provare"}])
    monkeypatch.setattr(issue_records, "refresh", lambda *a, **k: None)
    seen = {}

    def fetch(keys):
        seen["keys"] = keys
        return [row]
    p = {"rito": "penale", "family": "penale", "role": "imputato", "stage": "primo_grado"}
    link, _ = flow.decompose_records(state, p, [], set(), fetch=fetch)
    canonical = flow._label(flow._key(row["_article"]))
    assert seen["keys"][0] == ("73", "d.P.R. 309/1990") and link == [canonical] and canonical in state["records"]
    assert set(seen["keys"][1:]) == {("40", "c.p."), ("41", "c.p."), ("43", "c.p.")}      # v79.7: with any charge
    assert state["charge_aliases"] == {"73 d.P.R. 309/1990": canonical}
    state["records"][canonical]["requisites"][0]["id"] = "R1"
    out = posture.mark_sides(state, p)
    assert out["charged"] == [canonical] and state["records"][canonical]["side"] == "threat"


# --- C1 ------------------------------------------------------------------------------------------------------------------

def _sides_state(monkeypatch, case, alternatives):
    from src.rag import interactive_flow as flow, posture
    records = {"624 c.p.": {"rubrica": "Furto", "requisites": [_req(1, "sottrae la cosa mobile altrui")]},
               "646 c.p.": {"rubrica": "Appropriazione indebita", "requisites": [_req(2, "si appropria")]}}
    monkeypatch.setattr(flow, "case_text", lambda s: case)
    monkeypatch.setattr(posture, "alternative_rows", lambda s, rows=(): [
        ({"name": x, "facts": "sì"}, {"_article": (x.split()[0], "c.p.")}) for x in alternatives])
    return {"records": records, "read": {"alternatives": [{"name": x} for x in alternatives]}}


IMPUTATO = {"rito": "penale", "family": "penale", "role": "imputato", "stage": "primo_grado"}


def test_the_first_charged_article_stays_the_charge_when_the_read_lists_it_as_an_alternative(monkeypatch):
    from src.rag import posture
    state = _sides_state(monkeypatch, "Il PM contesta ad Alfa l'art. 624 c.p.", ["624 c.p."])
    out = posture.mark_sides(state, IMPUTATO)
    rec = state["records"]["624 c.p."]
    assert out["charged"] == ["624 c.p."] and rec["side"] == "threat" and rec["charged"] == 1
    assert not rec.get("alternative")


def test_a_second_charged_article_that_is_the_clients_alternative_keeps_the_old_behaviour(monkeypatch):
    from src.rag import posture
    state = _sides_state(monkeypatch, "Il PM contesta ad Alfa gli artt. 624 e 646 c.p.", ["646 c.p."])
    posture.mark_sides(state, IMPUTATO)
    assert state["records"]["624 c.p."]["side"] == "threat"
    assert state["records"]["646 c.p."]["side"] == "weapon" and state["records"]["646 c.p."]["alternative"]


# --- C4 ------------------------------------------------------------------------------------------------------------------

def _hypothesis_state():
    return {"read": {"hypotheses": [{"line": "subordinata", "claim": "minaccia", "institutions": ["minaccia"]},
                                    {"line": "principale", "claim": "furto", "institutions": ["furto"]}]},
            "rows": [], "reserve": []}


def test_a_criminal_case_with_no_charge_takes_the_first_reads_main_hypothesis(monkeypatch):
    from src.rag import interactive_flow as flow, posture
    rows = [{"_article": ("612", "c.p."), "_first_read": "minaccia", "s": {"rubrica": "Minaccia"}},
            {"_article": ("624", "c.p."), "_first_read": "furto", "s": {"rubrica": "Furto"}}]
    monkeypatch.setattr(flow, "case_text", lambda s: "Alfa prende la bici di Beta dal cortile.")
    assert posture.claim_from_read(_hypothesis_state(), IMPUTATO, rows) == ["624 c.p."]
    assert posture.claim_from_read(_hypothesis_state(), {**IMPUTATO, "role": "persona_offesa"}, rows) == ["624 c.p."]


def test_the_hypothesis_is_labelled_and_the_registro_sentence_is_never_printed():
    from src.rag import lawyer_method as M
    recs = {"624 c.p.": {"rubrica": "Furto", "side": "threat", "charged": 1, "charge_hypothesis": True,
                         "requisites": [_req(1, "sottrae la cosa mobile altrui", "provato"),
                                        _req(2, "al fine di trarne profitto", "da provare")]}}
    a = M.assessment(recs, IMPUTATO)
    assert M.HYPOTHESIS_LABEL in a["why"] and "registro" not in a["why"]
    assert M.HYPOTHESIS_LABEL == "ipotesi di reato (addebito non indicato nel quesito)"
    assert M.HYPOTHESIS_LABEL in M.summary_section({}, [{"tesi": "Linea"}], 0, a, "Alfa")
    assert M.HYPOTHESIS_LABEL in M.assessment_text(a, "Alfa")
    none = M.assessment({}, IMPUTATO)
    assert "registro" not in none["why"]


# --- C5 ------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Il 3 marzo 2026 Alfa ha consegnato a Beta un quadro d'autore pattuendo un prezzo di 9.000 euro. Beta ha ritirato "
    "il quadro e da allora non ha pagato nulla, nonostante tre solleciti scritti. Quali azioni può intraprendere Alfa "
    "contro Beta?",
    "Nel gennaio 2026 il vicino Beta ha costruito un muro alto tre metri a ridosso del confine del giardino di Alfa, "
    "togliendogli luce e veduta. Alfa ha scritto a Beta senza ottenere risposta. Cosa può chiedere Alfa e davanti a chi?",
    "Il 10 febbraio 2026 Alfa ha prestato 5.000 euro a Beta con una scrittura privata che prevedeva la restituzione "
    "entro giugno. Beta non ha restituito nulla e non risponde più al telefono. Come può agire Alfa per riavere la somma?",
])
def test_a_claimant_is_never_the_models_convenuto(monkeypatch, text):
    from src.rag import intake
    monkeypatch.setattr(intake, "ask_model", lambda *a, **k: {"role": "convenuto", "rito": "civile",
                                                               "deliverable": "parere", "stage": "pre_filing"})
    monkeypatch.setattr(intake, "client_rule", lambda *a, **k: {"role": "", "why": "", "name": ""})
    got = intake.detect(text)
    assert got["role"] == "attore" and "azione" in got["why"]["role"]


@pytest.mark.parametrize("text", [
    "Il 2 ottobre 2026 Beta ha notificato ad Alfa un atto di citazione chiedendo 20.000 euro per le infiltrazioni "
    "provenienti dal suo appartamento. Alfa sostiene che l'acqua arrivi dal tetto comune. Come si difende Alfa?",
    "Nel settembre 2026 Beta ha fatto causa ad Alfa davanti al Tribunale di Roma per i danni causati dal crollo di un "
    "muro di confine. Alfa ritiene che il muro fosse di proprietà comune. Quale difesa impostare?",
    "Il 5 novembre 2026 ad Alfa è stato notificato un decreto ingiuntivo di 8.000 euro per forniture che Alfa sostiene "
    "di non aver mai ricevuto. Il termine per opporsi sta per scadere. Come rispondere?",
])
def test_a_defendant_keeps_the_models_convenuto(monkeypatch, text):
    from src.rag import intake
    monkeypatch.setattr(intake, "ask_model", lambda *a, **k: {"role": "convenuto", "rito": "civile",
                                                               "deliverable": "parere", "stage": "pre_filing"})
    monkeypatch.setattr(intake, "client_rule", lambda *a, **k: {"role": "", "why": "", "name": ""})
    assert intake.detect(text)["role"] == "convenuto"


@pytest.mark.parametrize("text, role", [
    ("Assisto un imprenditore edile che non è stato pagato.", "imprenditore"),
    ("Assisto una ristoratrice a cui è stato chiuso il locale.", "ristoratrice"),
    ("Assisto il condominio di via Roma.", "condominio"),
    ("Assisto la società che gestisce il cinema del paese.", "società"),
    ("Assisto l'acquirente di un'auto difettosa.", "acquirente"),
])
def test_assisto_with_an_article_and_a_noun_names_the_clients_role(text, role):
    from src.rag import interactive_flow as flow
    assert flow.stated(text)["client_role"] == role


def test_the_method_writes_from_the_derived_role_and_says_so_in_one_line(monkeypatch):
    from src.rag import interaction_copy as C, interactive_flow as flow
    state = {"questions": [{"id": "parte_assistita"}], "answered": {}, "intake": {"role": "attore"}}
    monkeypatch.setattr(flow, "single_mode", lambda s: True)
    monkeypatch.setattr(flow, "lawyer_method_on", lambda: True)
    monkeypatch.setattr(flow, "_assumed_party", lambda s, a: None)
    client, line = flow.settle_client(state, {"facts": {}})
    # v79.19.0 (item 4): the one sentence for a side read from the facts -- the brief's banner, no question
    assert "Non mi hai indicato chi assisti" not in line and line == C.fill(C.CLIENT_DEDUCED, parte="la parte attore")
    assert line == "Il parere è impostato per la parte attore, dedotto dai fatti: dimmi se assisti un'altra parte."
    assert line.count(".") == 1
