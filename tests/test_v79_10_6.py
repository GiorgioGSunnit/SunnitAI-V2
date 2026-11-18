"""v79.10.6 (Block E's review), on synthetic sentences, rows and records only -- each finding's failure scenario:

E3 the ledger: a removal is checked and struck only where its rule says the sentence is absent (answer_ledger.misplaced:
   the one check's section, a repetition's occurrences after the first, the client's own text); a copy outside it is
   never struck and the entry is never relabelled to make the nightly read 0
E1 the charge the user states in words is the user's article (first tier), never labelled «ipotesi di reato»; the
   thesis call may not replace it -- in §3 and in §1's summary -- while a defence (an exception, a limiting norm, a
   riqualificazione on the record, a scriminante) passes through
E2 names: the gazetteer splits a run (the name after an office is redacted), titles take surnames with particles, the
   institutions stay; the nightly's count reads the passages literally, apart from the redactor
"""

import os

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

from src.rag import answer_ledger as L  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import lawyer_method as M  # noqa: E402
from src.rag import posture, rulings  # noqa: E402

IMPUTATO = {"rito": "penale", "family": "penale", "role": "imputato", "stage": "primo_grado"}
DUP = "Il venditore ha consegnato la merce con un ritardo di tre mesi."


def _check(sentence, section, rule):
    return {"step": "record_check", "kind": "removed", "section": section, "sentence": sentence, "rule": rule,
            "articles": []}


# --- E3 (findings 1, 4, 10) ---------------------------------------------------------------------------------------------

def test_a_repetitions_kept_original_is_never_struck_nor_relabelled():
    draft = f"**1. Sintesi**\n\n{DUP} La domanda appare fondata.\n\n**2. Fatti**\n\nAltro testo dei fatti."
    entries = [_check(DUP, "Sintesi", "ripetizione")]
    out, report = L.settle(draft, entries)
    assert out == draft and report == {"struck": [], "kept": [], "present": []}
    assert entries[0]["kind"] == "removed" and "kept_why" not in entries[0]
    assert L.misplaced(draft, entries[0]) == [] and L.removed_present(draft, entries) == []
    # a copy of the repeated sentence in another section is that section's: untouched
    other = draft.replace("Altro testo dei fatti.", f"{DUP} Altro testo dei fatti.")
    assert L.settle(other, entries)[0] == other and entries[0]["kind"] == "removed"


def test_a_repetition_reprinted_twice_in_its_section_loses_only_the_copies():
    text = f"**1. Sintesi**\n\n{DUP} Primo. {DUP} Secondo. {DUP}\n\n**2. Fatti**\n\nAltro."
    entries = [_check(DUP, "Sintesi", "ripetizione")]
    assert len(L.misplaced(text, entries[0])) == 2
    out, report = L.settle(text, entries)
    assert out.count(DUP) == 1 and out.startswith(f"**1. Sintesi**\n\n{DUP} Primo.")
    assert entries[0]["settled"] == "struck" and entries[0]["kind"] == "removed" and entries[0]["struck"] == 2


@pytest.mark.parametrize("rule", ["alternativa_senza_fatti", "requisito_non_registrato"])
def test_a_section_rules_removal_never_strikes_the_copies_in_other_sections(rule):
    text = f"**1. Sintesi**\n\nIn apertura.\n\n**4. Norme**\n\n{DUP}\n\n**5. Soluzioni**\n\n{DUP} Altro."
    entries = [_check(DUP, "Sintesi", rule)]
    out, report = L.settle(text, entries)
    assert out == text and report["present"] == [] and entries[0]["kind"] == "removed"
    # reprinted in its own section: struck there only
    again = text.replace("In apertura.", f"In apertura. {DUP}")
    out, _ = L.settle(again, entries)
    assert out == text and entries[0]["settled"] == "struck"


def test_a_clients_removal_is_checked_in_its_own_section_never_relabelled():
    own = "La clausola penale non risulta manifestamente eccessiva nel caso."
    entry = {"step": "compose:rules", "kind": "removed", "sentence": own, "client": "Alfa", "articles": []}
    # still in Alfa's own text (one paragraph of the section kept it): present, struck -- never «kept» as «client»
    text = (f"**5. Norme**\n\n**Nell'interesse di Alfa**\n\nPrimo paragrafo.\n\n{own}\n\n"
            f"**Nell'interesse di Beta**\n\nTesto di Beta.")
    assert L.removed_present(text, [entry]) == [entry]
    out, _ = L.settle(text, [entry])
    assert own not in out and entry["settled"] == "struck" and entry["kind"] == "removed"
    # only in Beta's block: no presence for Alfa's removal, the entry stays «removed»
    beta = {"step": "compose:rules", "kind": "removed", "sentence": own, "client": "Alfa", "articles": []}
    text = f"**5. Norme**\n\n**Nell'interesse di Alfa**\n\nPrimo.\n\n**Nell'interesse di Beta**\n\n{own}"
    assert L.removed_present(text, [beta]) == []
    assert L.settle(text, [beta])[0] == text
    assert beta["kind"] == "removed" and "kept_why" not in beta and "settled" not in beta


def test_the_flow_counts_no_relabelled_entry_and_reads_zero_honestly():
    import scripts.nightly_sim as nightly
    own = "La condotta di Alfa integra pienamente il reato contestato."
    entries = [{"step": "compose", "kind": "removed", "sentence": own, "client": "Alfa", "articles": []},
               _check(DUP, "Sintesi", "ripetizione")]
    draft = (f"**1. Sintesi**\n\n{DUP} Testo.\n\n**2. Fatti**\n\n**Nell'interesse di Alfa**\n\nAltro.\n\n"
             f"**Nell'interesse di Beta**\n\n{own}")
    state = {"ledger": {"entries": entries, "removed_by_step": {"compose": 1, "record_check": 1}}}
    out = flow.ledger_settled(state, draft)
    assert out == draft and state["block_counters"] == {"ledger_struck": 0, "ledger_kept": 0,
                                                        "ledger_removed_present": 0}
    assert all(e["kind"] == "removed" for e in entries)
    assert state["ledger"]["removed_by_step"] == {"compose": 1, "record_check": 1}
    assert nightly.ledger_check(out, state)["count"] == 0
    # the same sentence reprinted where the removal holds is counted before settling
    bad = draft.replace("Altro.", f"Altro. {own}")
    assert nightly.ledger_check(bad, state)["count"] == 1


# --- E1 (findings 2, 9: the order) --------------------------------------------------------------------------------------

def _req(i, text, status="provato"):
    return {"id": f"R{i}", "comma": 1, "kind": "requisito", "text": text, "status": status}


def _row(number, code, inst, rubrica):
    return {"_article": (number, code), "_first_read": inst, "s": {"plain_text": "testo", "titolo_articolo": rubrica}}


def _state(monkeypatch, case, minaccia, furto):
    monkeypatch.setattr(flow, "case_text", lambda s: case)
    monkeypatch.setattr(posture, "alternative_rows", lambda s, rows=(): [])
    return {"read": {"hypotheses": [{"line": "principale", "claim": "minaccia", "institutions": ["minaccia"]},
                                    {"line": "subordinata", "claim": "furto", "institutions": ["furto"]}]},
            "rows": [_row("612", "c.p.", "minaccia", "Minaccia"), _row("624", "c.p.", "furto", "Furto")],
            "reserve": [],
            "records": {"612 c.p.": {"rubrica": "Minaccia", "requisites": minaccia},
                        "624 c.p.": {"rubrica": "Furto", "requisites": furto}}}


def test_the_charge_in_words_beats_a_read_with_more_elements_integrato(monkeypatch):
    # finding 2: minaccia named, furto has two elements integrato
    state = _state(monkeypatch, "Alfa è imputato di minaccia nei confronti di Beta, cui avrebbe preso la bicicletta.",
                   [_req(1, "minaccia ad altri un ingiusto danno", "da provare")],
                   [_req(2, "si impossessa della cosa"), _req(3, "sottraendola a chi la detiene")])
    out = posture.mark_sides(state, IMPUTATO)
    assert state["charge"] == {"article": "612 c.p.", "rule": "user_article"} and out["charged"] == ["612 c.p."]
    assert state["records"]["612 c.p."]["side"] == "threat" and state["records"]["624 c.p."]["side"] == "weapon"
    assert not any(r.get("charge_hypothesis") for r in state["records"].values())
    q = M.question_section({}, {}, state["records"], "Alfa", "612 c.p.")
    assert M.HYPOTHESIS_LABEL not in q and "contestato a Alfa" in q
    # finding 9: furto named, minaccia has two elements integrato -- the same facts twice, the same charge
    for _ in range(2):
        state = _state(monkeypatch, "Alfa è imputato di furto della bicicletta di Beta.",
                       [_req(1, "minaccia ad altri"), _req(4, "un ingiusto danno")],
                       [_req(2, "si impossessa della cosa")])
        out = posture.mark_sides(state, IMPUTATO)
        assert state["charge"] == {"article": "624 c.p.", "rule": "user_article"} and out["charged"] == ["624 c.p."]
        assert not state["records"]["624 c.p."].get("charge_hypothesis")


# --- E1 (findings 3, 8: the thesis may not replace the charge) ----------------------------------------------------------

def test_the_exception_question_passes_without_sides_on_the_records():
    # finding 3, verbatim records: no side marked on either
    recs = {"1453 c.c.": {"rubrica": "Risolubilità del contratto per inadempimento", "requisites": []},
            "1460 c.c.": {"rubrica": "Eccezione d'inadempimento", "requisites": []}}
    plan, log = {"questione_giuridica": "Se il convenuto possa opporre l'eccezione di inadempimento ex art. 1460 c.c. "
                                        "alla domanda di risoluzione."}, {}
    assert M.question_section(plan, {}, recs, "Rossi", "1453 c.c.", log) == plan["questione_giuridica"] and not log


def test_a_limiting_norm_and_a_scriminante_are_no_replacement():
    recs = {"1218 c.c.": {"rubrica": "Responsabilità del debitore", "side": "weapon", "charged": 1, "requisites": []},
            "1227 c.c.": {"rubrica": "Concorso del fatto colposo del creditore", "side": "weapon", "requisites": []}}
    plan, log = {"questione_giuridica": "Se il risarcimento vada ridotto ex art. 1227 c.c."}, {}
    assert M.question_section(plan, {}, recs, "Rossi", "1218 c.c.", log) == plan["questione_giuridica"] and not log
    penal = {"582 c.p.": {"rubrica": "Lesione personale", "side": "threat", "charged": 1, "requisites": []},
             "581 c.p.": {"rubrica": "Percosse", "side": "weapon", "alternative_facts": "nessuna malattia",
                          "requisites": []}}
    for text in ("Se la condotta di Alfa sia scriminata dalla legittima difesa ex art. 52 c.p.",
                 "Se la condotta di Alfa sia riqualificabile come percosse ex art. 581 c.p."):
        log = {}
        assert M.question_section({"questione_giuridica": text}, {}, penal, "Alfa", "582 c.p.", log) == text
        assert not log


def test_a_summary_replacing_the_charge_is_refused_and_counted():
    recs = {"646 c.p.": {"rubrica": "Appropriazione indebita", "side": "threat", "charged": 1, "requisites": []},
            "640 c.p.": {"rubrica": "Truffa", "side": "weapon", "requisites": []}}
    a = {"esito": M.INCERTO, "why": "elementi da provare", "obiettivo": "assoluzione"}
    sols = [{"tesi": "Contestare l'appropriazione"}]
    log = {}
    out = M.summary_section({"sintesi": "Alfa risponde di truffa ex art. 640 c.p."}, sols, 0, a, "Alfa", "646 c.p.",
                            recs, log)
    assert "640" not in out and out.startswith("Linea principale:")
    assert log["charge_refused_summary"]["named"] == ["640 c.p."]
    on_charge = {"sintesi": "Alfa risponde di appropriazione indebita: la difesa contesta il possesso."}
    assert M.summary_section(on_charge, sols, 0, a, "Alfa", "646 c.p.", recs) == on_charge["sintesi"]
    # no charge held: as before
    assert M.summary_section({"sintesi": "Truffa ex art. 640 c.p."}, sols, 0, a, "Alfa") == "Truffa ex art. 640 c.p."


# --- E2 (findings 5, 6, 7) ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,gone", [
    ("Il Giudice Mario Rossi ha rigettato l'istanza.", "Mario Rossi"),
    ("La sentenza del Presidente Luca Verdi è nulla.", "Luca Verdi"),
    ("Il Procuratore Generale Paolo Neri ha chiesto la condanna.", "Paolo Neri"),
    ("La Banca Rossi Bianchi ha agito.", "Rossi Bianchi"),
    ("La prof.ssa della Valle ha deposto.", "della Valle"),
    ("L'ing. van Basten ha firmato.", "van Basten"),
    ("La sig.ra de Santis ha denunciato.", "de Santis"),
])
def test_a_name_beside_an_office_or_after_a_title_with_a_particle_is_redacted(text, gone):
    out = rulings.redact_names(text)
    assert gone.split()[-1] not in out and rulings.REDACTED in out
    assert rulings.names_left(text) and rulings.names_left(out) == []


@pytest.mark.parametrize("text", [
    "Gazzetta Ufficiale n. 12.",
    "Pretura Circondariale di Milano aveva condannato.",
    "PQM Rigetta il ricorso.",
    "Ministero della Giustizia (Dipartimento Affari Penitenziari)",
    "Il Tribunale Amministrativo Regionale del Lazio ha deciso.",
    "Accoglie il ricorso e Cassa la sentenza.",
    "Il dott. di turno ha visitato il paziente.",
])
def test_institutions_are_neither_redacted_nor_counted(text):
    assert rulings.redact_names(text) == text and rulings.names_left(text) == []


def test_the_nightly_count_sees_a_redactors_miss(monkeypatch):
    import scripts.nightly_sim as nightly
    passage = "La diagnosi del dott. di Pietro fu tardiva; Mario Rossi lo ha confermato."
    raw = f"- Cass. pen., Sez. 4, n. 7/2025 (data n.d.), {rulings.PASSAGE_LABEL}: «{passage}»."
    assert nightly.names_check(raw)["count"] == 2
    # the redactor printing them (its reading missing both) does not blind the count
    monkeypatch.setattr(rulings, "name_spans", lambda text: ([], []))
    assert rulings.redact_names(passage) == passage
    got = nightly.names_check(raw)
    assert got["ok"] is False and got["count"] == 2 and "di Pietro" in got["why"] and "Mario Rossi" in got["why"]


# --- the verifiers' follow-ups (v79.10.6, second pass) -------------------------------------------------------------------

NEAR = "Il venditore ha consegnato tutta la merce con un ritardo di tre mesi."


def test_a_near_repetition_reprinted_is_misplaced_and_struck_not_hidden():
    # record_check removes a near-match: the logged sentence is not the one it repeated, so no copy of it is kept
    draft = f"**1. Sintesi**\n\n{DUP} {NEAR} La domanda appare fondata.\n\n**2. Fatti**\n\nAltro.\n"
    entries = [_check(NEAR, "Sintesi", "ripetizione")]
    assert len(L.removed_present(draft, entries)) == 1
    out, report = L.settle(draft, entries)
    assert NEAR not in out and DUP in out and report["struck"] == [NEAR]
    assert entries[0]["kind"] == "removed" and entries[0]["settled"] == "struck"
    assert L.removed_present(out, entries) == []


def test_an_exact_repetition_keeps_its_first_copy():
    draft = f"**1. Sintesi**\n\n{DUP} La domanda appare fondata. {DUP}\n\n**2. Fatti**\n\nAltro.\n"
    entries = [_check(DUP, "Sintesi", "ripetizione")]
    out, _ = L.settle(draft, entries)
    assert out.count(DUP) == 1 and entries[0]["settled"] == "struck"


def test_a_client_whose_name_holds_e_is_found_in_its_own_block():
    s = "La clausola penale non risulta manifestamente eccessiva nel caso."
    draft = ("**5. Norme**\n\n**Nell'interesse di Rossi e Figli S.n.c.**\n\n" + s + " Altro testo.\n\n"
             "**Nell'interesse di Beta S.p.A.**\n\nTesto di Beta.\n\n**6. Conclusioni**\n\nFine.\n")
    entries = [{"step": "compose:rules", "kind": "removed", "sentence": s, "client": "Rossi e Figli S.n.c."}]
    assert len(L.removed_present(draft, entries)) == 1
    out, report = L.settle(draft, entries)
    assert s not in out and report["struck"] == [s] and entries[0]["kind"] == "removed"
    # the same sentence in Beta's own block is Beta's: not this client's removal
    other = [{"step": "compose:rules", "kind": "removed", "sentence": "Testo di Beta lungo abbastanza da contare qui.",
              "client": "Rossi e Figli S.n.c."}]
    text = draft.replace("Testo di Beta.", other[0]["sentence"])
    assert L.removed_present(text, other) == []


def test_the_read_alternative_named_alone_as_the_question_is_refused():
    recs = {"646 c.p.": {"rubrica": "Appropriazione indebita", "side": "threat", "charged": 1, "requisites": []},
            "640 c.p.": {"rubrica": "Truffa", "side": "weapon", "alternative": True, "requisites": []}}
    log = {}
    plan = {"questione_giuridica": "Se i fatti integrino il reato di truffa aggravata e con quali conseguenze per Alfa."}
    out = M.question_section(plan, {"article": "640 c.p."}, recs, "Alfa", "646 c.p.", log)
    assert out != plan["questione_giuridica"] and "646 c.p." in out
    assert log["charge_refused"]["named"] == ["640 c.p."]
    # put as a riqualificazione, the same alternative passes
    log = {}
    text = "Se la condotta di Alfa sia qualificabile come truffa anziché come appropriazione."
    assert M.question_section({"questione_giuridica": text}, {}, recs, "Alfa", "646 c.p.", log) == text and not log
    log = {}
    text = "Se i fatti vadano derubricati nel reato di truffa ex art. 640 c.p."
    assert M.question_section({"questione_giuridica": text}, {}, recs, "Alfa", "646 c.p.", log) == text and not log


@pytest.mark.parametrize("text,gone", [
    ("Dei sigg. Rossi e de Luca si dice.", ("Rossi", "Luca")),
    ("Il Giudice Anna Maria Rossi-Bianchi ha deciso.", ("Anna", "Rossi", "Bianchi")),
    ("La teste Mario de Luca ha confermato il fatto.", ("Mario", "Luca")),
    ("Lo ha riferito Luca di Pietro in udienza.", ("Luca", "Pietro")),
    ("Il dr. Rossi ha visitato il paziente.", ("Rossi",)),
    ("La dr.ssa Bianchi ha deposto.", ("Bianchi",)),
    ("Lo conferma Anna dell'Orto, vicina di casa.", ("Anna", "Orto")),
    ("Anna Maria de Luca ha deposto.", ("Anna", "Maria", "Luca")),
    ("All'udienza Mario Rossi era presente.", ("Mario", "Rossi")),
])
def test_particles_hyphens_and_dr_are_redacted_and_counted_on_the_raw_text(text, gone):
    out = rulings.redact_names(text)
    assert not any(w in out for w in gone) and rulings.REDACTED in out
    assert rulings.names_left(text) and rulings.names_left(out) == []


@pytest.mark.parametrize("text", [
    "L'appellante Rossi ha agito.",
    "L'imputato ha agito con dolo.",
    "Si applica la Tabella A Allegata al decreto, non il d.lgs. n. 81.",
    "Ai sensi della lettera B Bis il termine decorre.",
    "In base al D Lgs n. 81 del 2008 si applica.",
    "La polizza UnipolSai Assicurazioni copre il danno.",
    "Il Tribunale di Milano ha deciso.",
    "Il Consiglio di Stato ha deciso.",
    "Secondo l'Ordine degli Avvocati la condotta è illecita.",
    "Lo Statuto dei Lavoratori si applica.",
])
def test_elided_articles_single_letters_and_institutions_stay_and_count_zero(text):
    assert rulings.redact_names(text) == text and rulings.names_left(text) == []


def test_the_nightly_count_sees_what_a_redaction_left():
    assert rulings.names_left("Il Giudice [omissis]-Bianchi ha deciso.") == ["[omissis]-Bianchi"]
    assert rulings.names_left("Dei sigg. [omissis] e de Luca si dice.") == ["de Luca"]


def test_a_surname_with_an_elided_capital_particle_is_one_token():
    out = rulings.redact_names("Lo ha detto Dell'Orto Mario in udienza.")
    assert "Orto" not in out and "Mario" not in out and rulings.names_left(out) == []
    assert rulings.redact_names("Dell'Ufficio Tecnico si dice poco.").startswith("Dell'Ufficio")
