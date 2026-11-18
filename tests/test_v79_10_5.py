"""v79.10.5 (Block E: charge stability, names, ledger), on synthetic sentences, rows and records only:

E1. the charge chosen by rule and held across the pipeline: the user's article, else the first read's hypothesis with
    most elements integrato, else the store's rubrica match; logged with its rule (state["charge"], the counter
    charge_rule); the thesis call's legal question may not replace it; the same facts twice give the same charge
E2. name redaction covers the surnames in quoted passages -- the capitalised tokens after «sig.», «dott.», «avv.», and
    two adjacent capitalised tokens outside the gazetteer, at a sentence's start too; the redaction log carries each
    rule; the nightly counts the names left in the passages (names_redacted, must be 0)
E3. the removal ledger: removal and logging in one function (answer_ledger.settle / strike) -- every sentence logged as
    removed is absent from the final text; the nightly's ledger_removed_absent reads 0
"""

import copy
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
ATTORE = {"rito": "ordinario", "family": "civile", "role": "attore", "stage": "pre_filing"}


def _req(i, text, status="provato"):
    return {"id": f"R{i}", "comma": 1, "kind": "requisito", "text": text, "status": status}


def _row(number, code, inst, rubrica):
    return {"_article": (number, code), "_first_read": inst, "s": {"plain_text": "testo", "titolo_articolo": rubrica}}


# --- E1 ------------------------------------------------------------------------------------------------------------------

def _penal_state(monkeypatch, case, hypotheses=None):
    monkeypatch.setattr(flow, "case_text", lambda s: case)
    monkeypatch.setattr(posture, "alternative_rows", lambda s, rows=(): [])
    hyps = hypotheses if hypotheses is not None else [
        {"line": "principale", "claim": "minaccia", "institutions": ["minaccia"]},
        {"line": "subordinata", "claim": "furto", "institutions": ["furto"]}]
    return {"read": {"hypotheses": hyps},
            "rows": [_row("612", "c.p.", "minaccia", "Minaccia"), _row("624", "c.p.", "furto", "Furto")],
            "reserve": [],
            "records": {"612 c.p.": {"rubrica": "Minaccia",
                                     "requisites": [_req(1, "minaccia ad altri un ingiusto danno", "da provare")]},
                        "624 c.p.": {"rubrica": "Furto",
                                     "requisites": [_req(2, "si impossessa della cosa mobile altrui"),
                                                    _req(3, "sottraendola a chi la detiene"),
                                                    _req(4, "al fine di trarne profitto", "da provare")]}}}


def test_the_users_article_is_the_charge_whatever_the_read_says(monkeypatch):
    state = _penal_state(monkeypatch, "Il PM contesta ad Alfa l'art. 612 c.p. per le parole rivolte a Beta.")
    out = posture.mark_sides(state, IMPUTATO)
    assert out["charged"][0] == "612 c.p." and state["records"]["612 c.p."]["charged"] == 1
    assert state["charge"] == {"article": "612 c.p.", "rule": "user_article"}
    assert state["block_counters"]["charge_rule"] == "user_article"
    assert out["charge_rule"]["rule"] == "user_article"
    assert not state["records"]["612 c.p."].get("charge_hypothesis")


def test_with_no_article_the_hypothesis_with_most_elements_integrato_is_the_charge(monkeypatch):
    state = _penal_state(monkeypatch, "Alfa prende la bicicletta di Beta dal cortile e se ne va.")
    out = posture.mark_sides(state, IMPUTATO)
    # the read's principale (612, nothing integrato) loses to its subordinata (624, two elements integrato)
    assert out["charged"][0] == "624 c.p." and state["charge"]["rule"] == "read_hypothesis"
    rec = state["records"]["624 c.p."]
    assert rec["charged"] == 1 and rec["side"] == "threat" and rec["charge_hypothesis"] is True
    assert out["charge_rule"]["scores"]["624 c.p."] == [2, 0, 3]


def test_the_same_facts_twice_give_the_same_charge(monkeypatch):
    first = _penal_state(monkeypatch, "Alfa prende la bicicletta di Beta dal cortile e se ne va.")
    second = copy.deepcopy(first)
    again = copy.deepcopy(first)
    again["read"]["hypotheses"] = list(reversed(again["read"]["hypotheses"]))      # the read in another order
    got = [posture.mark_sides(s, IMPUTATO)["charged"][0] for s in (first, second, again)]
    assert got == ["624 c.p."] * 3
    assert first["charge"] == second["charge"] == again["charge"] == {"article": "624 c.p.", "rule": "read_hypothesis"}


def test_a_tie_goes_to_the_principal_line_then_the_label_never_the_reads_order(monkeypatch):
    hyps = [{"line": "subordinata", "claim": "furto", "institutions": ["furto"]},
            {"line": "principale", "claim": "minaccia", "institutions": ["minaccia"]}]
    state = _penal_state(monkeypatch, "Alfa e Beta litigano nel cortile.", hyps)
    state["records"]["624 c.p."]["requisites"] = [_req(2, "si impossessa della cosa mobile altrui", "da provare")]
    posture.mark_sides(state, IMPUTATO)
    assert state["charge"] == {"article": "612 c.p.", "rule": "read_hypothesis"}


def test_the_charge_the_user_names_in_words_is_the_users_article(monkeypatch):
    state = _penal_state(monkeypatch, "Alfa è imputato di furto della bicicletta di Beta.", hypotheses=[])
    out = posture.mark_sides(state, IMPUTATO)
    assert state["charge"] == {"article": "624 c.p.", "rule": "user_article"} and out["charged"][0] == "624 c.p."
    assert not state["records"]["624 c.p."].get("charge_hypothesis")        # the user named it: no hypothesis label


@pytest.mark.parametrize("case,named", [
    ("Alfa è imputato di minaccia nei confronti di Beta, cui avrebbe preso la bicicletta.", "612 c.p."),
    ("Alfa è imputato di furto della bicicletta di Beta.", "624 c.p."),
])
def test_the_read_never_displaces_the_charge_the_user_names_in_words(monkeypatch, case, named):
    # the read's subordinata (624) has the most elements integrato; the user's words decide all the same
    state = _penal_state(monkeypatch, case)
    if named == "624 c.p.":         # here the read's best is minaccia: two elements integrato against furto's one
        state["records"]["612 c.p."]["requisites"] = [_req(1, "minaccia ad altri"), _req(9, "un ingiusto danno")]
    out = posture.mark_sides(state, IMPUTATO)
    assert state["charge"] == {"article": named, "rule": "user_article"} and out["charged"] == [named]
    assert not any(r.get("charge_hypothesis") for r in state["records"].values())
    assert state["records"][named]["side"] == "threat"


def test_a_hypothesis_rubricated_name_matches_the_stores_rubrica_when_no_row_holds_it(monkeypatch):
    hyps = [{"line": "principale", "claim": "appropriazione", "institutions": ["appropriazione"],
             "rubricated": ["Appropriazione indebita"]}]
    state = _penal_state(monkeypatch, "Alfa trattiene il denaro che Beta gli aveva affidato.", hyps)
    state["records"]["646 c.p."] = {"rubrica": "Appropriazione indebita",
                                    "requisites": [_req(5, "si appropria del denaro altrui")]}
    posture.mark_sides(state, IMPUTATO)
    assert state["charge"] == {"article": "646 c.p.", "rule": "rubrica_match"}


def test_no_rule_no_charge(monkeypatch):
    state = _penal_state(monkeypatch, "Alfa e Beta litigano nel cortile.", hypotheses=[])
    out = posture.mark_sides(state, IMPUTATO)
    assert out["charged"] == [] and state["charge"] == {"article": "", "rule": "none"}


def test_the_civil_claim_from_the_read_follows_the_same_rule(monkeypatch):
    monkeypatch.setattr(flow, "case_text", lambda s: "Caio non ha consegnato la merce pagata da Tizio.")
    monkeypatch.setattr(posture, "alternative_rows", lambda s, rows=(): [])
    read = {"hypotheses": [{"line": "principale", "claim": "danno", "institutions": ["fatto illecito"]},
                           {"line": "principale", "claim": "inadempimento", "institutions": ["inadempimento"]},
                           {"line": "controparte", "claim": "eccezione", "institutions": ["eccezione"]}]}
    state = {"read": read, "reserve": [],
             "rows": [_row("2043", "c.c.", "fatto illecito", "Risarcimento per fatto illecito"),
                      _row("1218", "c.c.", "inadempimento", "Responsabilità del debitore"),
                      _row("1460", "c.c.", "eccezione", "Eccezione d'inadempimento")],
             "records": {"2043 c.c.": {"rubrica": "Risarcimento per fatto illecito",
                                       "requisites": [_req(1, "il fatto doloso o colposo", "da provare")]},
                         "1218 c.c.": {"rubrica": "Responsabilità del debitore",
                                       "requisites": [_req(2, "l'inadempimento")]},
                         "1460 c.c.": {"rubrica": "Eccezione d'inadempimento",
                                       "requisites": [_req(3, "l'inadempimento dell'altra parte"),
                                                      _req(4, "la buona fede")]}}}
    posture.mark_sides(state, ATTORE)
    # the attore's line only (principale): 1218 has an element integrato, 2043 none; the other side's 1460 is no claim
    assert state["charge"] == {"article": "1218 c.c.", "rule": "read_hypothesis"}


def test_the_thesis_calls_question_never_replaces_the_charge():
    recs = {"646 c.p.": {"rubrica": "Appropriazione indebita", "side": "threat", "charged": 1,
                         "requisites": [_req(1, "si appropria del denaro altrui")]},
            "640 c.p.": {"rubrica": "Truffa", "side": "weapon",
                         "requisites": [_req(2, "con artifizi o raggiri", "da provare")]}}
    log = {}
    plan = {"questione_giuridica": "Se i fatti integrino la truffa dell'art. 640 c.p. e con quali conseguenze."}
    a = {"article": "640 c.p."}            # the forecast moved to the main line's article
    q = M.question_section(plan, a, recs, "Alfa", "646 c.p.", log)
    assert "art. 646 c.p." in q and "contestato a Alfa" in q and "640" not in q
    assert log["charge_refused"]["named"] == ["640 c.p."] and log["charge_refused"]["charge"] == "646 c.p."
    # named in words only, by the record's rubrica: refused alike
    words, log = {"questione_giuridica": "Se i fatti integrino il reato di truffa aggravata e con quali conseguenze "
                                         "per Alfa."}, {}
    assert "art. 646 c.p." in M.question_section(words, a, recs, "Alfa", "646 c.p.", log)
    assert log["charge_refused"]["named"] == ["640 c.p."]
    on_charge = {"questione_giuridica": "Se l'appropriazione indebita sia integrata dalla condotta di Alfa."}
    assert M.question_section(on_charge, a, recs, "Alfa", "646 c.p.") == on_charge["questione_giuridica"]
    both = {"questione_giuridica": "Se i fatti integrino l'art. 646 c.p. contestato o la truffa dell'art. 640 c.p."}
    assert M.question_section(both, a, recs, "Alfa", "646 c.p.") == both["questione_giuridica"]
    bare = {"questione_giuridica": "Se la condotta di Alfa sia punibile."}
    assert M.question_section(bare, a, recs, "Alfa", "646 c.p.") == bare["questione_giuridica"]
    # the code's own question is on the held charge, never on the article the forecast moved to
    assert "art. 646 c.p." in M.question_section({}, a, recs, "Alfa", "646 c.p.")
    assert "art. 640 c.p." in M.question_section({}, a, recs, "Alfa")       # no charge held: as before


def test_a_defence_question_is_no_replacement_of_the_charge():
    # the read's riqualificazione (an alternative): the imputato's own line
    recs = {"582 c.p.": {"rubrica": "Lesione personale", "side": "threat", "charged": 1, "requisites": []},
            "581 c.p.": {"rubrica": "Percosse", "side": "weapon", "alternative": True, "requisites": []}}
    plan, log = {"questione_giuridica": "Se la condotta di Alfa sia qualificabile come percosse ex art. 581 c.p."}, {}
    assert M.question_section(plan, {}, recs, "Alfa", "582 c.p.", log) == plan["questione_giuridica"] and not log
    # the convenuto's exception to the claim, and the other side's exception to the attore's claim
    civil = {"1453 c.c.": {"rubrica": "Risolubilità del contratto per inadempimento", "side": "threat", "charged": 1,
                           "requisites": []},
             "1460 c.c.": {"rubrica": "Eccezione d'inadempimento", "side": "weapon", "requisites": []}}
    plan = {"questione_giuridica": "Se il convenuto possa opporre l'eccezione di inadempimento ex art. 1460 c.c."}
    assert M.question_section(plan, {}, civil, "Rossi", "1453 c.c.", log) == plan["questione_giuridica"] and not log
    civil["1453 c.c."]["side"], civil["1460 c.c."]["side"] = "weapon", "threat"
    assert M.question_section(plan, {}, civil, "Rossi", "1453 c.c.", log) == plan["questione_giuridica"] and not log
    # another claim on the client's own side is a replacement
    civil["2043 c.c."] = {"rubrica": "Risarcimento per fatto illecito", "side": "weapon", "requisites": []}
    other = {"questione_giuridica": "Se spetti a Rossi il risarcimento ex art. 2043 c.c."}
    assert "art. 1453 c.c." in M.question_section(other, {}, civil, "Rossi", "1453 c.c.", log)
    assert log["charge_refused"]["named"] == ["2043 c.c."]


# --- E2 ------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,gone", [
    ("Il reato sussiste quando il sig. Mario Rossi dispone della cosa.", ["Mario", "Rossi"]),
    ("La responsabilità della dott.ssa De Luca è esclusa.", ["De Luca"]),
    ("Come ritenuto dall'Avv. ROSSI nel ricorso.", ["ROSSI"]),
    ("Il dott. Bianchi Giuseppe ha omesso la diagnosi.", ["Bianchi", "Giuseppe"]),
    ("I sigg. Rossi e Bianchi hanno agito in concorso.", ["Rossi", "Bianchi"]),
    ("Mario Rossi ha agito con dolo.", ["Mario Rossi"]),
    ("La condotta di «Mario Rossi» integra il reato.", ["Mario Rossi"]),
    ("Pertanto Luca Verdi risponde del danno.", ["Luca Verdi"]),
    ("Il Giudice Mario Rossi ha rigettato l'istanza.", ["Mario Rossi"]),
    ("La sentenza del Presidente Luca Verdi è nulla.", ["Luca Verdi"]),
    ("Il Consigliere Relatore Anna Neri ha esposto i fatti.", ["Anna Neri"]),
    ("La diagnosi del dott. di Pietro fu tardiva.", ["Pietro"]),
    ("Il sig. dell'Orto ha agito.", ["Orto"]),
    ("Il ricorso dell'avv. Mario de Luca è infondato.", ["Mario", "Luca"]),
])
def test_the_surnames_in_a_quoted_passage_are_redacted(text, gone):
    out = rulings.redact_names(text)
    assert all(g not in out for g in gone) and rulings.REDACTED in out


@pytest.mark.parametrize("text", [
    "La Corte Costituzionale e le Sezioni Unite lo escludono.",
    "Nella Repubblica Italiana il principio è pacifico.",
    "Il Tribunale di Roma ha accolto la domanda.",
    "Il ricorso dell'Avvocato Generale dello Stato è infondato.",
    "Gazzetta Ufficiale n. 12.",
    "Pretura Circondariale di Milano aveva condannato.",
    "PQM Rigetta il ricorso.",
    "Ministero della Giustizia (Dipartimento Affari Penitenziari)",
    "Il Tribunale Amministrativo Regionale del Lazio ha deciso.",
    "RITENUTO IN FATTO",
    "Il dott. di turno ha visitato il paziente.",
])
def test_offices_courts_and_openers_are_never_names(text):
    assert rulings.redact_names(text) == text


def test_the_redaction_log_carries_each_rule_and_what_the_gazetteer_kept(monkeypatch):
    seen = []
    monkeypatch.setattr(rulings, "_log", lambda event, data, *a: seen.append((event, data)))
    rulings.redact_names("Il sig. Mario Rossi e Luca Verdi hanno agito; la Corte Costituzionale lo esclude.")
    entry = next(d for e, d in seen if e == "redaction")
    assert entry["items"] == [{"words": "Mario Rossi", "rule": "titled"}, {"words": "Luca Verdi", "rule": "pair"}]
    assert entry["rules"] == {"titled": 1, "pair": 1} and entry["kept"] == ["Corte Costituzionale"]
    assert entry["replaced"] == ["Mario Rossi", "Luca Verdi"]


def test_the_nightly_counts_the_names_left_in_the_passages(monkeypatch):
    import scripts.nightly_sim as nightly
    passage = "Il reato sussiste quando il sig. Mario Rossi dispone della cosa altrui come propria."
    raw = f"- Cass. pen., Sez. 2, n. 5/2025 (data n.d.), {rulings.PASSAGE_LABEL}: «{passage}»."
    got = nightly.names_check(raw)
    assert got["ok"] is False and got["count"] == 1 and "Mario Rossi" in got["why"]
    line = rulings.ruling_line({"estremi": "Cass. pen., Sez. 2, n. 5/2025 (data n.d.)", "passage": passage})
    assert nightly.names_check(line) == {"check": "names_redacted", "ok": True, "count": 0, "why": ""}
    state = {}
    flow.names_counted(state, line)
    assert state["block_counters"] == {"names_redacted": 1, "names_unredacted": 0}
    flow.names_counted(state, raw + "\n\nMario Rossi, fuori dai passaggi, non conta.")
    assert state["block_counters"]["names_unredacted"] == 1
    monkeypatch.setenv("RULINGS_PARTY_NAMES", "keep")
    assert nightly.names_check(raw)["ok"] is True          # kept by choice


# --- E3 ------------------------------------------------------------------------------------------------------------------

SENT = "Il venditore deve garantire la cosa venduta dai vizi occulti."


@pytest.mark.parametrize("text,want", [
    (f"Prima frase del testo. {SENT} Ultima frase del testo.", "Prima frase del testo. Ultima frase del testo."),
    (f"- {SENT}\n- Altra voce dell'elenco qui.", "- Altra voce dell'elenco qui."),
    (f"| T1 | {SENT} | alto |", "| T1 | — | alto |"),
    (f"**1. Norme**\n\n{SENT}\n\nAltro paragrafo qui.", "**1. Norme**\n\nAltro paragrafo qui."),
    (f"{SENT[:-1]} (da verificare: la data).\n\nAltro.", "Altro."),
])
def test_strike_removes_every_whole_sentence_occurrence(text, want):
    out, n, held = L.strike(text, SENT)
    assert out == want and n == 1 and held == []


@pytest.mark.parametrize("text", [
    "Inoltre, il venditore deve garantire la cosa venduta dai vizi occulti.",
    "Il venditore deve garantire la cosa venduta dai vizi occulti (art. 1490 c.c.).",
    f"**{SENT}**",
])
def test_strike_never_cuts_a_longer_sentence_or_a_heading(text):
    out, n, held = L.strike(text, SENT)
    assert out == text and n == 0 and held


def test_every_sentence_logged_as_removed_is_absent_from_the_final_text():
    import scripts.nightly_sim as nightly
    code_line = "La pretesa si fonda sulla garanzia per i vizi della cosa venduta."
    entries = [{"step": "record_check", "kind": "removed", "sentence": SENT, "articles": []},           # reprinted
               {"step": "compose:prose", "kind": "removed", "sentence": code_line, "articles": []},     # the code's
               {"step": "compose:wording", "kind": "removed", "client": "Beta", "articles": [],         # B's keeps it
                "sentence": "L'acquirente ha denunciato i vizi entro otto giorni dalla scoperta."},
               {"step": "compose:scope", "kind": "removed", "articles": [],                             # inside another
                "sentence": "Il compratore può chiedere la risoluzione del contratto."},
               {"step": "compose:scope", "kind": "removed", "sentence": "Una frase che non c'è più nel testo.",
                "articles": []},
               {"step": "rites_out", "kind": "proposed", "sentence": "Frase proposta e rimasta nel testo finale.",
                "articles": []}]
    draft = (f"**1. Sintesi**\n\nTesto iniziale del parere. {SENT}\n\n{code_line}\n\n"
             "L'acquirente ha denunciato i vizi entro otto giorni dalla scoperta.\n\n"
             "Inoltre, il compratore può chiedere la risoluzione del contratto.\n\n"
             "Frase proposta e rimasta nel testo finale.")
    draft = draft.replace("L'acquirente ha denunciato i vizi entro otto giorni dalla scoperta.\n\n", "") + (
        "\n\n**2. Fatti**\n\n**Nell'interesse di Alfa**\n\n"
        "L'acquirente ha denunciato i vizi entro otto giorni dalla scoperta.\n\n"
        "**Nell'interesse di Beta**\n\nIl testo di Beta per questa sezione.")
    state = {"ledger": {"entries": entries, "removed_by_step": {"record_check": 1, "compose:prose": 1,
                                                                "compose:wording": 1, "compose:scope": 2}},
             "output_sources": {"code": [code_line]}}
    before = nightly.ledger_check(draft, state)
    assert before["ok"] is False and before["count"] == 3          # Beta's removal: only Alfa's block holds it
    out = flow.ledger_settled(state, draft)
    assert SENT not in out and code_line in out and "Frase proposta e rimasta nel testo finale." in out
    assert "Inoltre, il compratore può chiedere la risoluzione del contratto." in out
    assert nightly.ledger_check(out, state) == {"check": "ledger_removed_absent", "ok": True, "count": 0, "why": ""}
    assert state["block_counters"] == {"ledger_struck": 1, "ledger_kept": 2, "ledger_removed_present": 0}
    assert entries[0]["kind"] == "removed" and entries[0]["settled"] == "struck"
    assert [e.get("kept_why") for e in (entries[1], entries[3])] == ["code", "contained"]
    assert all(e["kind"] == "kept" and e["was"] == "removed" for e in (entries[1], entries[3]))
    # v79.10.6 (E3): Beta's removal, in Alfa's block only, is no presence -- never relabelled «kept»
    assert entries[2]["kind"] == "removed" and "kept_why" not in entries[2] and "settled" not in entries[2]
    assert entries[3]["now"].startswith("Inoltre, il compratore")
    assert entries[4]["kind"] == "removed" and entries[5]["kind"] == "proposed"
    assert state["ledger"]["removed_by_step"] == {"record_check": 1, "compose:prose": 0, "compose:wording": 1,
                                                  "compose:scope": 1}


def test_a_ledger_with_nothing_present_leaves_the_text_alone():
    state = {"ledger": {"entries": [{"step": "x", "kind": "removed", "sentence": SENT}]}}
    assert flow.ledger_settled(state, "Altro testo del parere.") == "Altro testo del parere."
    assert state["block_counters"] == {"ledger_struck": 0, "ledger_kept": 0, "ledger_removed_present": 0}
    empty = {}
    assert flow.ledger_settled(empty, "Testo.") == "Testo." and empty["block_counters"]["ledger_removed_present"] == 0


def test_removed_present_is_the_nightlys_test():
    entries = [{"kind": "removed", "sentence": SENT}, {"kind": "removed", "sentence": "Breve."},
               {"kind": "rewritten", "sentence": "Il venditore risponde dei vizi della cosa venduta."}]
    text = f"**{SENT}** e «Breve.» Il venditore risponde dei vizi della cosa venduta."
    assert L.removed_present(text, entries) == [entries[0]]          # markup aside; under 25 characters unchecked


# --- E3: a removal holds where it was made -------------------------------------------------------------------------------

DUP = "Il venditore ha consegnato la merce con un ritardo di tre mesi."


def _check(sentence, section, rule):
    return {"step": "record_check", "kind": "removed", "section": section, "sentence": sentence, "rule": rule,
            "articles": []}


def test_a_repetitions_removal_keeps_the_sentence_it_repeated():
    draft = f"**1. Sintesi**\n\n{DUP} La domanda appare fondata.\n\n**2. Fatti**\n\nAltro testo dei fatti."
    entries = [_check(DUP, "Sintesi", "ripetizione")]
    out, report = L.settle(draft, entries)
    assert out == draft and report == {"struck": [], "kept": [], "present": []}
    assert entries[0]["kind"] == "removed" and L.removed_present(out, entries) == []
    # reprinted by a later step: the repeat goes again, the first stays
    again = f"**1. Sintesi**\n\n{DUP} La domanda appare fondata. {DUP}\n\n**2. Fatti**\n\nAltro testo dei fatti."
    entries = [_check(DUP, "Sintesi", "ripetizione")]
    out, _ = L.settle(again, entries)
    assert out == draft and entries[0]["struck"] == 1 and entries[0]["settled"] == "struck"
    assert entries[0]["kind"] == "removed"


def test_the_one_checks_removal_is_its_sections_only():
    draft = (f"**1. Sintesi**\n\nIn apertura. {DUP}\n\n**2. Fatti**\n\n{DUP} Altro testo dei fatti.")
    entries = [_check(DUP, "Sintesi", "alternativa_senza_fatti")]
    out, report = L.settle(draft, entries)
    assert out == "**1. Sintesi**\n\nIn apertura.\n\n**2. Fatti**\n\n" + f"{DUP} Altro testo dei fatti."
    assert entries[0]["kind"] == "removed" and entries[0]["settled"] == "struck" and entries[0]["struck"] == 1
    # its section gone (compose's ten sections): a per-section rule's copy elsewhere is kept, another rule's struck
    gone = f"**1. Premessa**\n\n{DUP} Altro testo dei fatti."
    entries = [_check(DUP, "Sintesi", "requisito_non_registrato"), ]
    assert L.settle(gone, entries)[0] == gone and entries[0]["kind"] == "removed" and "kept_why" not in entries[0]
    entries = [_check(DUP, "Sintesi", "frammento")]
    assert L.settle(gone, entries)[0] == "**1. Premessa**\n\nAltro testo dei fatti."
    assert entries[0]["settled"] == "struck"


def test_a_clients_removal_is_struck_in_its_own_text_kept_in_another_clients():
    own = "La condotta di Alfa integra pienamente il reato contestato."
    draft = (f"**1. Sintesi**\n\n**Nell'interesse di Alfa**\n\n{own} Altro testo resta qui.\n\n"
             f"**Nell'interesse di Beta**\n\n{own} Testo di Beta.\n\n**2. Norme**\n\nTesto comune.")
    entries = [{"step": "compose", "kind": "removed", "sentence": own, "client": "Alfa", "articles": []}]
    out, _ = L.settle(draft, entries)
    assert out == (f"**1. Sintesi**\n\n**Nell'interesse di Alfa**\n\nAltro testo resta qui.\n\n"
                   f"**Nell'interesse di Beta**\n\n{own} Testo di Beta.\n\n**2. Norme**\n\nTesto comune.")
    assert entries[0]["kind"] == "removed" and entries[0]["settled"] == "struck" and entries[0]["struck"] == 1
    # a section shared by every client (no client's head) is the client's own text too
    shared = f"**1. Norme**\n\nTesto comune. {own}"
    entries = [{"step": "compose", "kind": "removed", "sentence": own, "client": "Alfa", "articles": []}]
    assert L.settle(shared, entries)[0] == "**1. Norme**\n\nTesto comune." and entries[0]["settled"] == "struck"
    # a block grouping Alfa with Beta («Nell'interesse di Alfa e Beta») is Alfa's text
    grouped = f"**1. Fatti**\n\n**Nell'interesse di Alfa e Beta**\n\n{own} Altro."
    entries = [{"step": "compose", "kind": "removed", "sentence": own, "client": "Alfa", "articles": []}]
    assert own not in L.settle(grouped, entries)[0] and entries[0]["kind"] == "removed"
