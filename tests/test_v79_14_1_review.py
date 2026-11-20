"""v79.14.1 -- the review's fixes on the merged release (items 1-3, 6, 8 and the merge rules). Synthetic request texts
and passages only (invented names and facts; never a real, eval or tester case); every model faked; no network, no
Neo4j.

1   the debtor's other wordings («decreto ingiuntivo contro Tizio», «gli ha notificato un atto di citazione», «Beta
    pretende ora il pagamento …») resolve to convenuto -- the fallback item 1 gives when the frame is off or dropped.
(b) role_flip once: a posture flip carried to the clients of a multi-client answer, the frame's own client, a verified
    «sconosciuto» role, the counterparty read by both definitions.
2   the honesty line's governing law: no special law's pattern takes a code institution («fallimento delle
    trattative», «furto tentato fallito», «vita privata»); a module's leading action decides.
3   the administrative act: never a private party's «diniego», «revoca», «delibera»; never an act that grants; never
    one negated after it; the Questura's revoca, the licenza commerciale; read in the case's words only.
6   a mixed record keeps one objection per open element; objection_party in a multi-client answer with no frame client.
8   the count in its other wordings; §5's relevance (generic roots, overlapping roots, adjective forms, «612 bis»).
"""

import os
import sys
from pathlib import Path

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.rag import case_frame as cf  # noqa: E402
from src.rag import family_gate  # noqa: E402
from src.rag import intake  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import lawyer_method as M  # noqa: E402
from src.rag import rulings  # noqa: E402
from test_v79_14_1_merge import DEBT, PARTIES, _reply  # noqa: E402
from test_v79_14_flow import _run  # noqa: E402


# --- 1. the debtor's other wordings ------------------------------------------------------------------------------------

def _resolved(text, role="attore", how=None):
    raw = {"rito": "civile", "role": role, "confidence": {"role": how or intake.MODEL}, "client": {"name": "Tizio"}}
    return intake.resolve(raw, {flow.CASE_KEY: text})


@pytest.mark.parametrize("text", [
    "Assisto Tizio. Beta S.r.l. ha ottenuto un decreto ingiuntivo contro Tizio per 8.000 euro.",
    "Assisto Tizio. Beta S.r.l. gli ha notificato un atto di citazione chiedendo 8.000 euro di penale.",
    "Assisto Tizio. Beta S.r.l. pretende ora il pagamento di una penale di 8.000 euro.",
])
def test_the_debtors_other_wordings_resolve_to_convenuto(text):
    assert intake.resisting(text, "Tizio")
    got = _resolved(text)
    assert got["role"] == "convenuto" and got["resisting"] and got["role_why"].startswith("«")


@pytest.mark.parametrize("text", [
    "Assisto Tizio. Tizio pretende ora il pagamento di una penale di 8.000 euro da Beta.",
    "Il mio cliente Tizio ha fornito merce a Beta. Beta non ha pagato; Tizio pretende il pagamento del saldo.",
    "Assisto Tizio. Beta S.r.l. chiede il pagamento della penale a Gamma.",
    "Assisto Tizio. Tizio ha ottenuto un decreto ingiuntivo contro Beta.",
    "Assisto Tizio. Se Beta pretendesse il pagamento della penale, Tizio potrebbe opporsi.",
])
def test_the_claimant_and_another_partys_demand_stay_attore(text):
    assert not intake.resisting(text, "Tizio")
    assert _resolved(text)["role"] == "attore"


def test_a_verified_frame_attore_still_wins_over_the_new_wordings():
    text = "Assisto Tizio. Beta S.r.l. pretende ora il pagamento di una penale di 8.000 euro."
    got = _resolved(text, how=intake.FRAME)
    assert got["role"] == "attore" and [c["step"] for c in got["frame_conflicts"]] == ["intake.resolve"]


# --- (b) one role_flip --------------------------------------------------------------------------------------------------

def _roles(st, p, names, case=DEBT):
    return [flow.method_role(st, p, {"name": n, "role": ""}, case) for n in names]


def test_a_posture_flip_carried_to_a_multi_client_answer_counts_once(monkeypatch):
    parties = PARTIES + (("Ivo Tasso", "posatore"),)
    out = _run(monkeypatch, DEBT, _reply(), parties=parties)
    st = out["state"]
    clean = _roles(st, st["posture"], ["Carlo Merlo", "Ivo Tasso"])
    assert [(m["role"], m["intake_role"], m["from"]) for m in clean] == [
        ("convenuto", "convenuto", flow.ROLE_FROM_FRAME), ("convenuto", "convenuto", flow.ROLE_FROM_INTAKE)]
    st["method_roles"] = clean
    flow.role_corpus_counted(st, "Testo.", [], st["posture"])
    assert st["role_flip_reasons"] == [] and cf.audit(st)["role_flip"] == 0
    # a later step writing «attore» into the posture: the frame's client keeps the frame's role (no flip of its own),
    # Ivo inherits the posture's flip -- one flip, «ruolo», never also «ruolo:Carlo Merlo» or «ruolo:Ivo Tasso»
    st["posture"] = {**st["posture"], "role": "attore", "intake_role": "convenuto"}
    st["method_roles"] = _roles(st, st["posture"], ["Carlo Merlo", "Ivo Tasso"])
    assert [m["intake_role"] for m in st["method_roles"]] == ["convenuto", "convenuto"]
    flow.role_corpus_counted(st, "Testo.", [], st["posture"])
    assert st["role_flip_reasons"] == [flow.ROLE_FLIP_ROLE] and st["block_counters"]["role_flip"] == 1
    assert cf.audit(st)["role_flip"] == 1 and flow.case_counters(st)["role_flip"] == 1


def test_a_verified_unknown_role_gives_no_client_a_side_and_no_flip(monkeypatch):
    reply = _reply(ruolo={"valore": "sconosciuto", "perche": "«Assisto Carlo Merlo»"})
    out = _run(monkeypatch, DEBT, reply, parties=PARTIES + (("Ivo Tasso", "posatore"),))
    st, p = out["state"], out["posture"]
    assert p["role"] == "sconosciuto"
    got = [flow.method_role(st, p, {"name": "Carlo Merlo", "role": "attore"}, DEBT),
           flow.method_role(st, p, {"name": "Ivo Tasso", "role": "convenuto"}, DEBT)]
    assert [m["role"] for m in got] == ["sconosciuto", "sconosciuto"]
    st["method_roles"] = got
    flow.role_corpus_counted(st, "Testo.", [], st["posture"])
    assert st["role_flip_reasons"] == [] and cf.audit(st)["role_flip"] == 0
    # flag off with no resolved role: the read's word, as before -- nothing to disagree with, no flip
    monkeypatch.delenv("CASE_FRAME", raising=False)
    plain = {"block_counters": {}, "role_check": {}, "analysis": {}, "text": DEBT, "request": "",
             "method_roles": [{"client": "Ivo Tasso", "role": "convenuto", "intake_role": "sconosciuto",
                               "from": flow.ROLE_FROM_INTAKE}]}
    assert flow.role_flip_reasons(plain, {"role": "sconosciuto"})[0] == []


def test_the_counterparty_served_counts_once_per_client_and_audit(monkeypatch):
    out = _run(monkeypatch, DEBT, _reply(), parties=PARTIES)
    st = out["state"]
    st["method_roles"] = [{"client": "Carlo Merlo", "role": "convenuto", "intake_role": "convenuto",
                           "from": flow.ROLE_FROM_FRAME},
                          {"client": "Gufo S.r.l.", "role": "convenuto", "intake_role": "convenuto",
                           "from": flow.ROLE_FROM_INTAKE}]
    st["role_flip_reasons"] = [f"{flow.ROLE_FLIP_COUNTERPARTY}:Gufo S.r.l."]
    st["plan_client"] = "Gufo S.r.l."
    a = cf.audit(st)
    assert a["frame_audit"]["role"]["client_is_counterparty"]
    assert a["role_flip"] == 1 and a["frame_audit"]["role"]["role_flip_reasons"] == [
        f"{flow.ROLE_FLIP_COUNTERPARTY}:Gufo S.r.l."]
    # without item 1's per-client reason the audit's own counts it, once
    st["role_flip_reasons"] = []
    assert cf.audit(st)["role_flip"] == 1


# --- 2. the governing law ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name, law", [
    ("fallimento delle trattative", "c.c."),
    ("responsabilità precontrattuale per fallimento delle trattative", "c.c."),
    ("furto tentato fallito", "c.p."),
    ("interferenze illecite nella vita privata (privacy)", "c.p."),
    ("sostituzione di persona con dati personali altrui", "c.p."),
    ("anatocismo", "c.c."),
    ("dichiarazione di fallimento della società", "d.lgs. 14/2019"),
    ("bancarotta fraudolenta", "d.lgs. 14/2019"),
    ("illecito trattamento dei dati personali", "Reg. UE 2016/679"),
    ("responsabilità medica", "L. 24/2017"),
    ("appalto pubblico", "d.lgs. 36/2023"),
    ("guida in stato di ebbrezza", "d.lgs. 285/1992"),
    ("anatocismo bancario", "d.lgs. 385/1993"),
])
def test_the_governing_law_of_code_and_special_institutions(name, law):
    assert M.governing_law({"name": name}, {"records": {}})[0] == law


def test_the_honesty_line_never_fires_on_a_code_institution_a_special_word_names(monkeypatch):
    from src.rag import corpus_laws
    monkeypatch.setitem(corpus_laws._NAMES, "names", ["Codice Civile 2026", "Codice Penale 2026"])
    for name in ("fallimento delle trattative", "furto tentato fallito",
                 "interferenze illecite nella vita privata (privacy)"):
        assert M.corpus_line({"records": {}, "institutions": [{"name": name}]}) == ""
    # a special-law institution missing from the archive: the line, as built
    line = M.corpus_line({"records": {}, "institutions": [{"name": "responsabilità medica"}]})
    assert line.startswith("Il corpus non contiene ancora informazioni sufficienti su responsabilità medica")


def test_a_modules_leading_action_decides_its_law():
    assert M.governing_law({"name": "locazione abitativa"}, {"records": {}}) == ("c.c.", "modulo")
    assert M.governing_law({"name": "licenziamento per giusta causa"}, {"records": {}})[0] == "L. 604/1966"


# --- 3. the administrative act ---------------------------------------------------------------------------------------

SALE = ("Elio Corvo ha acquistato da Nina Gallo un villino con rogito del 4 aprile 2025. Dopo il rogito ha scoperto "
        "difformità edilizie rispetto al permesso di costruire depositato in Comune. Elio Corvo vuole agire contro "
        "Nina Gallo per la riduzione del prezzo.")


@pytest.mark.parametrize("text", [
    SALE + " Nina Gallo aveva ottenuto dal Comune un provvedimento di sanatoria nel 2019.",        # grants the seller
    SALE + " La banca ha comunicato a Elio Corvo il diniego del mutuo.",                           # a private body
    SALE + " Il Comune aveva rilasciato a Nina Gallo il permesso di costruire con provvedimento dirigenziale del 2018.",
    SALE + " Il notaio ha verificato che il TAR non è mai stato adito.",                           # negated after
    "Voglio l'annullamento della delibera assembleare del condominio per vizi di convocazione.",
    "La banca ha opposto un diniego al mutuo.",
    "L'amministrazione condominiale ha emesso un provvedimento contro il condomino moroso.",
    "Il franchisor ha disposto la revoca della licenza del marchio.",
    "Il produttore ha comunicato la revoca della concessione di vendita in esclusiva.",
    "Il datore di lavoro mi ha escluso dalla procedura di selezione interna.",
    "Il locatore, nel Comune di Milano, ha opposto il diniego di rinnovo alla prima scadenza.",
    "Tizio vuole impugnare il permesso di costruire rilasciato dal Comune al vicino.",
])
def test_no_administrative_act(text):
    assert family_gate.administrative_act(text) == []


@pytest.mark.parametrize("text", [
    "La Questura mi ha revocato il porto d'armi per uso sportivo dopo una lite condominiale. Voglio impugnare la revoca.",
    "Mi hanno revocato la licenza commerciale del bar.",
    "La Prefettura ha sospeso la patente di guida di Tizio per sei mesi.",
    "Il Comune ha opposto un diniego alla richiesta di Tizio.",
    "Il Comune aveva rilasciato il permesso a Tizio, ma poi lo ha annullato in autotutela.",
    "Il ricorso al TAR non è stato ancora deciso.",
    "Tizio vuole impugnare davanti al TAR il permesso di costruire rilasciato dal Comune al vicino.",
])
def test_an_administrative_act(text):
    assert family_gate.administrative_act(text)


def test_the_last_pass_on_a_condominium_and_on_the_questura():
    draft = "La delibera è impugnabile (art. 1137 c.c.). Il termine è di sessanta giorni (art. 29 c.p.a.)."
    condo = {"text": "Sono condomino. L'assemblea ha approvato lavori straordinari senza quorum; voglio "
                     "l'annullamento della delibera.", "request": "", "records": {}, "intake": {}}
    out = flow.admin_act_out(condo, draft)
    assert "c.p.a." not in out and "1137 c.c." in out and condo["block_counters"]["act_last_pass_stripped"] == 1
    questura = {"text": "La Questura mi ha revocato il porto d'armi per uso sportivo dopo una lite condominiale. "
                        "Voglio impugnare la revoca.", "request": "", "records": {}, "intake": {}}
    assert flow.admin_act_out(questura, draft) == draft
    assert questura["block_counters"]["act_last_pass_stripped"] == 0


def test_the_act_is_read_in_the_cases_words_never_in_an_institution_name(monkeypatch):
    case = ("Elio Corvo ha acquistato da Nina Gallo un villino. Il Comune non ha emesso alcun provvedimento contro "
            "Elio Corvo, ma Elio Corvo ha scoperto difformità edilizie. Elio Corvo vuole agire contro Nina Gallo per "
            "la riduzione del prezzo.")
    framed = flow.read_institutions({"frame_institution": {
        "testo": "provvedimento del Comune", "frase": "«Il Comune non ha emesso alcun provvedimento contro Elio Corvo»"}})
    assert family_gate.administrative_act(family_gate.act_text(case, framed)) == []
    alternatives = [{"name": "vizi della cosa venduta"}, {"name": "ricorso al TAR contro il diniego di sanatoria"}]
    assert family_gate.administrative_act(family_gate.act_text(SALE, alternatives)) == []
    ctx = family_gate.context(SALE, institutions=alternatives)
    assert ctx["admin_act"] is False and not family_gate.decide("c.p.a.", "", ctx, number="29")["ok"]
    assert M.module_act({"case": SALE, "institutions": alternatives}) == []


# --- 6. objections ------------------------------------------------------------------------------------------------------

def test_objection_party_in_a_multi_client_answer_with_no_frame_client_keeps_each_role(monkeypatch):
    monkeypatch.setenv("CASE_FRAME", "true")
    state = {"case_frame": {"frame": {"parti": [{"nome": "Sempronio", "lato": "cliente"},
                                                {"nome": "Tizio", "lato": "cliente"},
                                                {"nome": "Mevio", "lato": "controparte"}],
                                      "ruolo": {"valore": "persona_offesa", "intake": "persona_offesa"}}},
             "method_roles": [{"client": "Sempronio", "role": "persona_offesa"}, {"client": "Tizio", "role": "imputato"}]}
    got = M.objection_party(state, "Tizio", {"rito": "penale", "family": "penale", "role": "imputato"})
    assert got == {"client": "Tizio", "role": "imputato", "frame": False, "side": "penale"}
    assert not state.get("frame_conflicts")
    # one client and no frame client: the frame's verified role still binds (79.14.1 as built)
    one = {"case_frame": state["case_frame"]}
    assert M.objection_party(one, "Sempronio", {"rito": "penale", "family": "penale", "role": "imputato"})["role"] == \
        "persona_offesa"


# --- 8. the count and §5's relevance -------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, left", [
    ("Esito sui fatti: da provare, requisiti provati 0/5.", ""),
    ("Risultano provati 0 requisiti su 5. La linea regge.", "La linea regge."),
    ("Sono provati 0 su 5 requisiti.", ""),
    ("Dei requisiti, provati 0 su 5.", ""),
    ("Sui fatti narrati i requisiti provati sono 0 su 5, quindi la tesi è debole.", ""),
    ("Requisiti provati: 0/5.", ""),
    ("Sono provati 2 dei 5 requisiti dell'art. 646 c.p.", ""),
    ("Esito: da provare (requisiti provati 0 su 5).", "Esito: da provare."),
])
def test_the_count_in_its_other_wordings_is_removed(text, left):
    assert M.count_left(text)
    assert M.count_out(text).strip() == left and not M.count_left(M.count_out(text))


def test_section_9s_reason_and_other_parentheses_stay():
    reason = "3 elementi su 5 dell'art. 646 c.p. (appropriazione) risultano provati; gli altri restano da provare."
    other = "La somma (euro 5.000) è dovuta entro 10 giorni (art. 1183 c.c.)."
    for text in (reason, other):
        assert not M.count_left(text) and M.count_out(text) == text


@pytest.mark.parametrize("passage, label, rubrica, named", [
    # generic aliases and roots: dropped
    ("Il risarcimento del danno da inadempimento contrattuale richiede la prova del nesso causale.", "2043 c.c.",
     "Risarcimento per fatto illecito", False),
    ("La garanzia fideiussoria prestata dal terzo si estingue con il pagamento del debito principale.", "1490 c.c.",
     "Garanzia per i vizi della cosa venduta", False),
    ("Gli artifici del mago nello spettacolo non sono rilevanti ai fini della misura cautelare.", "640 c.p.", "Truffa",
     False),
    ("In tema di misure cautelari reali, il sequestro dei beni immobili richiede il fumus.", "1158 c.c.",
     "Usucapione dei beni immobili e dei diritti reali immobiliari", False),
    ("In tema di misure cautelari, l'esercizio del potere del giudice è discrezionale.", "2050 c.c.",
     "Responsabilità per l'esercizio di attività pericolose", False),
    ("Il danno cagionato dalla circolazione di veicoli è risarcito dall'assicuratore.", "2051 c.c.",
     "Danno cagionato da cosa in custodia", False),
    # on the institution: kept
    ("Ai sensi dell'art. 612 bis c.p. il reato è abituale.", "612-bis c.p.", "Atti persecutori", True),
    ("La condotta estorsiva richiede la violenza o la minaccia per costringere la vittima a procurare un ingiusto "
     "profitto.", "629 c.p.", "Estorsione", True),
    ("Il committente risponde del fatto illecito del preposto commesso nell'esercizio delle incombenze cui è adibito.",
     "2049 c.c.", "Responsabilità dei padroni e dei committenti", True),
    ("La portata diffamatoria dello scritto va valutata nel contesto.", "595 c.p.", "Diffamazione", True),
    ("Chi è tenuto alla sorveglianza dell'incapace risponde del fatto da questo commesso.", "2047 c.c.",
     "Danno cagionato dall'incapace", True),
    ("Il conducente risponde se non prova di aver fatto il possibile.", "2054 c.c.", "Circolazione di veicoli", True),
    ("Gli artifici e raggiri inducono in errore la vittima.", "640 c.p.", "Truffa", True),
    ("La cosa in custodia che cagiona il danno obbliga il custode.", "2051 c.c.", "Danno cagionato da cosa in custodia",
     True),
])
def test_section_5_relevance_after_the_review(passage, label, rubrica, named):
    assert rulings.passage_names_institution(passage, label, rubrica) is named


def test_overlapping_roots_are_one_and_generic_roots_name_nothing():
    stems = rulings.institution_stems("Usucapione dei beni immobili e dei diritti reali immobiliari")
    assert stems.count("immobil") == 1 and not any(x.startswith("immobil") and x != "immobil" for x in stems)
    assert rulings.relevance_stems("Usucapione dei beni immobili e dei diritti reali immobiliari") == ["usucapio"]
    assert rulings.relevance_stems("Responsabilità per l'esercizio di attività pericolose") == ["pericolo"]
    assert rulings.relevance_stem("estorsione") == "estors" and rulings.relevance_stem("diffamazione") == "diffama"
