"""v65: general rules from five posture cells -- intake (listed questions, a single point-of-law question, the current
status first and never a hypothetical mention, the named articles with their verb), plan and sides (a section per
question, a named article always written with its side, the counterparty's label, the persona offesa's and the
convenuto's closed lists, the defence's applications, the alternatives' verdicts), retrieval gates (the rubrica
channel, the special laws), checks and render (the requirement check in the writer review, phantom letters and
numbers, no «non verificabile», the store's numbers) and the store (each term with its subject). Synthetic texts and
records; the model and the graph are stubbed."""

import os
import sys
from pathlib import Path

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

from src.rag import (answer_invariants, answer_ledger, case_read, fact_resolver, fact_signals, intake,  # noqa: E402
                     issue_records, posture, procedural_facts, record_check, render_computed, special_laws,
                     writer_rules)
from src.rag import interaction_copy as C  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import plan_then_write as ptw  # noqa: E402

PROVATO, DA_PROVARE, NON_RISULTA = flow.PROVATO, flow.DA_PROVARE, flow.NON_RISULTA
LEAD, NORMS, CLOSING = flow.ANSWER_TITLES[0], flow.ANSWER_TITLES[1], flow.ANSWER_TITLES[4]


def _rec(rubrica, *statuses, **extra):
    reqs = [{"kind": "requisito", "comma": 1, "text": f"elemento {i}", "status": st} for i, st in enumerate(statuses, 1)]
    rec = {"rubrica": rubrica, "requisites": reqs, **extra}
    rec["verdict"] = issue_records.verdict(rec)
    return rec


# --- 1. listed questions; one point-of-law question -----------------------------------------------------------------------

def test_listed_questions_are_quesito_blocks_in_order():
    read = intake.detect("Tizio ha venduto un'auto a Caio, che non ha pagato. Si chiede di valutare: 1. se Tizio possa "
                         "agire per la risoluzione; 2. quali termini di prescrizione si applichino; 3. se convenga un "
                         "decreto ingiuntivo.", model=False)
    assert read["requested_blocks"] == [
        {"kind": "quesito", "subject": "se Tizio possa agire per la risoluzione"},
        {"kind": "quesito", "subject": "quali termini di prescrizione si applichino"},
        {"kind": "quesito", "subject": "se convenga un decreto ingiuntivo"}]
    lettered = intake.requested_blocks("Caia viene investita. In particolare: a) chi risponde del danno? b) entro "
                                       "quando agire?")
    assert [b["subject"] for b in lettered] == ["chi risponde del danno?", "entro quando agire?"]
    clauses = intake.requested_blocks("Il candidato rediga parere, in particolare: se il credito sia prescritto; se la "
                                      "prova testimoniale sia ammessa.")
    assert [b["kind"] for b in clauses] == ["parere", "quesito", "quesito"]


def test_a_narrative_ending_in_one_question_on_an_article_is_a_quesito():
    one = intake.detect("Il cane di Tizio ha morso Caio nel giardino condominiale. È applicabile l'art. 2052 c.c.?",
                        model=False)
    assert one["deliverable"] == "quesito" and one["route"] == intake.RAG
    two = intake.detect("Il cane di Tizio ha morso Caio. È applicabile l'art. 2052 c.c.? E l'art. 2043 c.c.?",
                        model=False)
    assert two["deliverable"] != "quesito" or "una sola domanda" not in two["why"]["deliverable"]
    assert intake.point_of_law_question("Il cane di Tizio ha morso Caio. Chi paga?") == "Chi paga?"   # v68: no article


# --- 2. the current status first; never a hypothetical mention ------------------------------------------------------------

def test_role_and_stage_come_from_the_current_status_and_never_from_a_hypothesis():
    read = intake.detect("Caia ha sporto querela contro Tizio per lesioni. Caia, attualmente persona offesa, potrà "
                         "costituirsi parte civile nel processo. Il procedimento è nella fase delle indagini preliminari. "
                         "Sono l'avvocato di Caia: quali iniziative?", model=False)
    assert (read["role"], read["stage"]) == ("persona_offesa", "indagini")
    assert "stato attuale" in read["why"]["role"] and "stato attuale" in read["why"]["stage"]
    sued = intake.detect("Tizio è stato citato in giudizio da Caio per il pagamento di 20.000 euro. Il mio cliente "
                         "Tizio non ha ancora depositato la comparsa di costituzione. Cosa può eccepire?", model=False)
    assert (sued["role"], sued["stage"]) == ("convenuto", "primo_grado")
    assert intake.hypothetical("Tizio potrà costituirsi parte civile", 12, 36)
    assert intake.hypothetical("Tizio si costituirà parte civile", 9, 32)
    assert not intake.hypothetical("Tizio si è costituito parte civile", 9, 34)
    future = intake.client_rule("Caia ha sporto querela contro Tizio. Caia potrà costituirsi parte civile. Sono "
                                "l'avvocato di Caia.", penal=True)
    assert future["role"] == "persona_offesa"                            # never parte_civile from «potrà»


# --- 3. the named articles and their verb ---------------------------------------------------------------------------------

@pytest.mark.parametrize("text, client, expected", [
    ("Il mio cliente intende invocare l'art. 2051 c.c. contro il Comune.", "Tizio", ("2051 c.c.", "invocare", "weapon")),
    ("È possibile esperire l'azione ex art. 1168 c.c.?", "", ("1168 c.c.", "esperire", "weapon")),
    ("Caio chiede la condanna di Tizio al risarcimento ai sensi dell'art. 2043 c.c.", "Tizio",
     ("2043 c.c.", "chiede", "threat")),
    ("Tizio chiede la condanna di Caio al pagamento ai sensi dell'art. 1218 c.c.", "Tizio", ("1218 c.c.", "chiede", "weapon")),
    ("Caio eccepisce l'inadempimento ex art. 1460 c.c.", "Tizio", ("1460 c.c.", "eccepisce", "threat")),
    ("Il fatto è disciplinato dall'art. 2052 c.c.", "Tizio", ("2052 c.c.", "", "")),
])
def test_a_named_article_carries_its_verb_and_its_use(text, client, expected):
    got = intake.named_articles(text, client)
    assert [(a["article"], a["verb"], a["use"]) for a in got] == [expected]
    assert "named_articles" in intake.detect(text, model=False)


# --- 4. a section per question, from the record and the store ------------------------------------------------------------

BODY = f"**1. {LEAD}**\n\nLa linea.\n\n**2. {NORMS}**\n\nLe norme.\n\n**3. {CLOSING}**\n\nL'esito."


def test_each_question_is_a_section_headed_by_it_answered_from_the_record_and_the_store():
    jobs_seen = []

    def write(jobs):
        jobs_seen.extend(jobs)
        return [(t, f"Risposta a {t.split(':', 1)[1]}") for t, *_ in jobs]
    blocks = [{"kind": "quesito", "subject": "se Tizio possa agire per la risoluzione"},
              {"kind": "parere", "subject": "parere"},
              {"kind": "quesito", "subject": "quali termini si applichino"}]
    out = flow.requested_sections({}, BODY, blocks, write, {}, "Il caso.", quesito_material="SCHEDE: X\n\nSTORE: Y")
    titles = [t for t, _ in answer_ledger.sections_of(out)]
    assert titles == ["Se Tizio possa agire per la risoluzione", "Quali termini si applichino", LEAD, NORMS, CLOSING]
    # v68: the questions first, in the asked order, the body after them wherever the parere is asked
    assert all("SCHEDE: X" in material and "STORE: Y" in material for _, _, material, _ in jobs_seen)
    assert "se Tizio possa agire per la risoluzione" in jobs_seen[0][1]
    only = flow.requested_sections({}, BODY, blocks[:1], write, {}, "Il caso.")
    assert [t for t, _ in answer_ledger.sections_of(only)][0] == "Se Tizio possa agire per la risoluzione"


def test_the_quesito_material_holds_the_schede_and_the_stores_terms():
    rows = [{"_article": ("641", "c.p.c."), "s": {"plain_text": "L'opposizione si propone entro il termine di "
                                                                 "quaranta giorni dalla notificazione."}}]
    state = {"records": {"641 c.p.c.": _rec("Accoglimento della domanda", PROVATO)}, "sentences": []}
    text = flow.quesito_material(state, [("641", "c.p.c.")], rows, "- Termini: X")
    assert "[1] art. 641 c.p.c." in text and "PROFILI PROCESSUALI" in text
    assert "- art. 641 c.p.c.: Opposizione: quaranta giorni dalla notificazione (art. 641, testo, c.p.c.)" in text


# --- 5. a named article: always written, its side by its verb; the counterparty's label ------------------------------------

def test_a_named_articles_side_is_its_verbs_whatever_the_role():
    records = {"2043 c.c.": _rec("Risarcimento per fatto illecito", PROVATO),
               "1227 c.c.": _rec("Concorso del fatto colposo del creditore", DA_PROVARE)}
    state = {"records": records, "text": "Il caso.", "analysis": {flow.CASE_KEY: "Il caso."},
             "intake": {"named_articles": [{"article": "2043 c.c.", "verb": "chiede", "use": "threat"},
                                           {"article": "1227 c.c.", "verb": "eccepire", "use": "weapon"}]}}
    posture.mark_sides(state, {"role": "attore"})
    assert records["2043 c.c."]["side"] == "threat" and "chiede" in records["2043 c.c."]["side_why"]
    assert records["1227 c.c."]["side"] == "weapon"


def test_a_named_article_is_always_written_with_its_status():
    state = {"records": {"2051 c.c.": {**_rec("Danno cagionato da cosa in custodia", DA_PROVARE), "named_due": True},
                         "2043 c.c.": _rec("Risarcimento per fatto illecito", PROVATO)}}
    draft = f"**2. {NORMS}**\n\n**Art. 2043 c.c. — Risarcimento per fatto illecito**\n\nIl fatto illecito.\n\n**5. {CLOSING}**\n\nX."
    text, rows = flow.ledger_out(draft, state, retry=lambda label: "La cosa era in custodia del Comune.")
    norms = dict(answer_ledger.sections_of(text))[NORMS]
    assert ("**Art. 2051 c.c. — Danno cagionato da cosa in custodia**\nEsito sui fatti del caso: da provare (requisiti "
            "provati: 0 su 1).\n\nLa cosa era in custodia del Comune.") in norms      # v68: a second call, never bare
    assert issue_records.block_due(state["records"]["2051 c.c."])
    roads = flow.general_rows({"records": {}, "intake": {"named_articles": [{"article": "2051 c.c."}]}, "text": ""},
                              {}, {}, {}, [], fetch=lambda refs: [{"_article": r, "s": {}} for r in refs])
    assert [flow._label(flow._key(r["_article"])) for r in roads[C.APPLICATION_VIA_NAMED]] == ["2051 c.c."]


@pytest.mark.parametrize("role, stage, civil, mine, other", [
    ("attore", "pre_filing", False, "attore", "convenuto"), ("convenuto", "primo_grado", False, "convenuto", "attore"),
    ("imputato", "primo_grado", False, "imputato", "persona offesa"),
    ("imputato", "primo_grado", True, "imputato", "parte civile"),
    ("persona_offesa", "indagini", False, "persona offesa", "indagato"),
    ("parte_civile", "impugnazione", False, "parte civile", "imputato"), ("terzo", "esecuzione", False, "", "")])
def test_the_counterpartys_label_is_the_opposite_of_the_clients_role(role, stage, civil, mine, other):
    assert posture.role_label(role, stage) == mine and posture.counterparty_label(role, stage, civil) == other


def test_the_client_line_names_the_counterparty_by_the_opposite_role():
    analysis = {flow.CASE_KEY: "Leonardo chiede il passaggio a Domitilla.", "fact_signals": {"facts": {"parties": [
        {"name": "Leonardo", "role": "cliente"}, {"name": "Domitilla", "role": "controparte"}]}}}
    state = {"intake": {"role": "attore", "stage": "pre_filing"}, "text": analysis[flow.CASE_KEY]}
    line = flow.known_client_line(state, analysis, "Leonardo")
    assert line == "Il parere è reso nell'interesse di Leonardo; controparte: Domitilla (convenuto)."


# --- 6. the persona offesa's and the convenuto's closed lists ------------------------------------------------------------------

def _post(rito, role, stage):
    return {"rito": rito, "family": posture.family(rito), "role": role, "stage": stage,
            "entry": posture.entry_for(rito, role, stage), "prior_decisions": [], "querela_stated": True}


def test_the_persona_offesa_in_the_indagini_and_the_convenuto_at_trial_get_their_lists():
    titles = [t for t, _ in render_computed.stage_blocks(_post("penale", "persona_offesa", "indagini"))]
    assert titles == ["Diritti e facoltà della persona offesa", "Accertamenti tecnici non ripetibili",
                      "Richiesta di archiviazione e opposizione", "Costituzione di parte civile (fase successiva)"]
    text = "\n".join(b for _, b in render_computed.stage_blocks(_post("penale", "persona_offesa", "indagini")))
    for ref in ("art. 90 c.p.p.", "art. 90-bis c.p.p.", "art. 335, co. 3, c.p.p.", "art. 360 c.p.p.", "art. 367 c.p.p.",
                "artt. 408, co. 2-3, e 410 c.p.p.", "artt. 74-79 c.p.p."):
        assert ref in text, ref
    assert render_computed.stage_blocks(_post("penale", "imputato", "indagini")) == []
    civil = render_computed.stage_blocks(_post("civile", "convenuto", "primo_grado"))
    assert [t for t, _ in civil] == ["Costituzione del convenuto", "Domanda riconvenzionale", "Preclusioni",
                                     "Istruzione preventiva"]
    body = "\n".join(b for _, b in civil)
    for ref in ("art. 166 c.p.c.", "art. 167 c.p.c.", "art. 36 c.p.c.", "art. 171-ter c.p.c.", "art. 183 c.p.c.",
                "art. 696 c.p.c.", "art. 696-bis c.p.c."):
        assert ref in body, ref
    analysis = {"frame": {"areas": ["penale"]}, "posture": _post("penale", "persona_offesa", "indagini")}
    assert "Richiesta di archiviazione e opposizione" in [t for t, _ in render_computed.blocks(analysis)]


# --- 7. the defence's applications --------------------------------------------------------------------------------------------

def test_a_defending_clients_application_reads_from_the_defence():
    records = {"62 c.p.": {"rubrica": "Circostanze attenuanti comuni", "requisites": [], "applicazione": "x"}}
    block = issue_records.writer_block(records, [("62", "c.p.")], [], defending=True)
    assert f"[1] art. 62 c.p. — Circostanze attenuanti comuni — {writer_rules.APPLICATION_MARK_DEFENCE}" in block
    assert writer_rules.APPLICATION_RULE_DEFENCE in block and writer_rules.APPLICATION_RULE not in block
    assert writer_rules.APPLICATION_MARK in issue_records.writer_block(records, [("62", "c.p.")], [])


# --- 8. the alternatives: the first read's «no facts», a sound «SÌ», never in §1 ---------------------------------------------

ALTS = [{"name": "lesioni personali", "rubricated": "Lesione personale", "facts": "si"},
        {"name": "lesioni colpose", "rubricated": "Lesioni personali colpose", "facts": "no"},
        {"name": "percosse", "rubricated": "Percosse", "facts": "si"}]
ROWS = [{"_article": ("582", "c.p."), "s": {"titolo_articolo": "Lesione personale"}},
        {"_article": ("590", "c.p."), "s": {"titolo_articolo": "Lesioni personali colpose"}},
        {"_article": ("581", "c.p."), "s": {"titolo_articolo": "Percosse"}}]


def test_the_first_read_says_whether_a_fact_holds_each_alternative():
    assert [case_read.facts_verdict(x) for x in ("sì", "si", "no", "No, nessuno", "", None)] == ["si", "si", "no", "no",
                                                                                                 "", ""]
    read = case_read.clean({"fatti": ["Tizio colpisce Caio"],
                            "qualificazioni_alternative": [{"istituto": "lesioni colpose", "rubricato": "Lesioni "
                                                            "personali colpose", "perche": "x", "fatti": "no"}]})
    assert read["alternatives"][0]["facts"] == "no" and '"no" se nessun fatto' in case_read._SYSTEM


def test_a_no_facts_alternative_is_a_no_and_a_yes_needs_a_sound_record():
    contradicted = _rec("Percosse", NON_RISULTA)
    state = {"rows": ROWS, "read": {"alternatives": ALTS},
             "records": {"582 c.p.": _rec("Lesione personale", PROVATO), "590 c.p.": _rec("Lesioni personali colpose",
                                                                                          PROVATO),
                         "581 c.p.": contradicted}}
    written = "[1] SÌ: lo schiaffo mirava a ledere.\n[2] SÌ: una colpa.\n[3] SÌ: un colpo."
    items, _ = flow.parse_alternatives(written, posture.alternative_rows(state))
    assert [i["verdict"] for i in items] == [flow.FITS, flow.REJECTED, flow.FITS]          # 590: the read's «no»
    done = flow.write_alternatives(state, [], "Il caso.", lambda jobs: [(t, written) for t, *_ in jobs])
    assert done["fits"] == ["582 c.p."] and issue_records.contradicted(contradicted)          # 581: contradicted
    jobs, _ = flow.requested_jobs(state, [{"kind": "alternative"}], {}, "Il caso.", posture.alternative_rows(state))
    assert f"[2] art. 590 c.p. — Lesioni personali colpose (lesioni colpose) — {C.ALTERNATIVE_NO_FACTS}" in jobs[0][2]


def test_a_no_facts_alternative_stays_out_of_the_lead():
    records = {"584 c.p.": _rec("Omicidio preterintenzionale", PROVATO, charged=1),
               "590 c.p.": _rec("Lesioni personali colpose", PROVATO, alternative=True, alternative_facts="no"),
               "582 c.p.": _rec("Lesione personale", PROVATO, alternative=True, alternative_facts="si")}
    offences = [("584", "x"), ("590", "y"), ("582", "z")]
    assert posture.procedure_offences(records, offences)[0] == [("584", "x"), ("582", "z")]
    ctx = {"lead": LEAD, "unfounded": ["590 c.p."], "allowed": {"590 c.p.", "582 c.p.", "584 c.p."}}
    sections, log = record_check.check([(LEAD, "\n\nIn subordine, lesioni colpose (art. 590 c.p.). In subordine, lesioni "
                                               "dolose (art. 582 c.p.).\n\n")], ctx)
    assert "590" not in sections[0][1] and "582 c.p." in sections[0][1]
    assert log["rules"] == {"alternativa_senza_fatti": 1}
    kept, _ = record_check.check([(NORMS, "\n\nLe lesioni colpose (art. 590 c.p.) non ricorrono.\n\n")], ctx)
    assert "590" in kept[0][1]                                                           # only §1


# --- 9. the rubrica channel's gate ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name, heading, user, admitted", [
    ("servitù di passaggio", "Servitù di passaggio coattivo", None, True),         # two words
    ("truffa", "Truffa", None, False),                                             # one word, not the user's
    ("truffa", "Truffa", "Mi hanno truffato: è una truffa?", True),               # one word the user names
    ("termine di impugnazione", "Termini per le impugnazioni", "il termine", False),  # generic words alone
    ("decreto ingiuntivo", "Decreto di citazione a giudizio", None, False),        # «decreto» alone
    ("decreto ingiuntivo", "Decreto di ingiunzione", None, True)])
def test_a_rubrica_match_needs_two_words_of_the_name_or_the_users_own(name, heading, user, admitted):
    assert (fact_resolver.rubrica_admits(name, heading, user) == "") is admitted


def test_the_first_read_search_gates_its_rubrica_channel_and_the_index_ignores_generic_words():
    seen = []

    def resolve(own, documents, istituto):
        seen.append(istituto)
        return [], {}
    flow.first_read_search(None, "db", [], ["truffa"], ["c.p."], session=object(), resolve=resolve,
                           text_search=lambda *a: [], index_search=lambda *a: [], user_text="Il caso.")
    assert seen and all(i.get("testo_utente") == "Il caso." for i in seen)
    rows, log = fact_resolver.index_channel(None, [], "termine di impugnazione", "c.p.c.", index=[
        {"name": "Delle impugnazioni", "articles": ["323"]}])
    assert rows == [] and log["why"] == "solo parole generiche"


# --- 10. the special laws: only with their domain signal --------------------------------------------------------------------

def test_a_special_law_needs_its_domain_signal_in_the_first_read():
    cds = {"_article": ("141", "c.d.s."), "s": {}}              # v79.10: the labels the pipeline builds for the
    consumo = {"_article": ("130", "Codice del Consumo 2026"), "s": {}}     # archive's documents (their exact names)
    cc = {"_article": ("2043", "c.c."), "s": {}}
    kept, dropped = special_laws.gate([cds, consumo, cc], "Tizio scivola sul pavimento di un negozio.", set())
    assert kept == [cc] and len(dropped) == 2
    kept, _ = special_laws.gate([cds, consumo], "Tizio, alla guida della sua auto, urta Caio.", set())
    assert kept == [cds]
    kept, _ = special_laws.gate([consumo], "Caio, privato consumatore, acquista una lavatrice in un negozio.", set())
    assert kept == [consumo]
    kept, _ = special_laws.gate([consumo], "Il negozio vende a un'altra società.", {("130", "Codice del Consumo 2026")})
    assert kept == [consumo]                                                            # the user names it
    read = {"facts": ["Caio acquista online da un rivenditore"], "parties": [{"name": "Caio", "role": "consumatore"}]}
    assert special_laws.signals(special_laws.read_text(read))["codice_del_consumo"]


def test_the_consumer_codes_term_lines_need_its_signal():
    term = {"id": "denuncia_conformita_consumo", "ref": "art. 132 Codice del Consumo", "text": "denuncia: 2 mesi"}
    found = special_laws.signals("Tizio vende la sua auto a Caio.")
    assert not special_laws.term_allowed(term, found) and special_laws.term_allowed(term, None)
    assert special_laws.term_allowed({"ref": "art. 1495 c.c."}, found)


# --- 11. the requirement check in the writer review --------------------------------------------------------------------------

THREATS = {"584 c.p.": ["con atti diretti a commettere uno dei delitti preveduti dagli articoli 581 e 582",
                        "cagiona la morte di un uomo"]}
TEXT = ("L'art. 584 c.p. richiede la prevedibilità della morte come conseguenza dello schiaffo. Il colpo fu "
        "uno schiaffo, non un pugno diretto a uccidere. L'art. 584 c.p. presuppone il dolo specifico rispetto all'evento.")


def test_the_writer_review_flags_an_unrecorded_requirement_and_drops_the_whole_sentence(monkeypatch):
    seen = {}

    def chat(messages, **k):
        seen["system"], seen["human"] = messages[0].content, messages[1].content
        return [{"i": 3, "tipo": ptw.REVIEW_REQUIREMENT, "perche": "il dolo specifico non è un elemento registrato"}]
    monkeypatch.setattr(ptw.dg, "_chat_json", chat)
    extra = {"threats": THREATS, "sections": [LEAD]}
    text, log = ptw.review_section(LEAD, TEXT, "Il caso.", extra=extra)
    assert ptw.REVIEW_REQUIREMENT_TYPE in seen["system"] and "- art. 584 c.p.: «con atti diretti" in seen["human"]
    assert "dolo specifico" not in text and "prevedibilità della morte" in text
    assert extra["ran"] == [LEAD] and len(extra["flagged"]) == 1
    other = {"threats": THREATS, "sections": [LEAD]}
    ptw.review_section(NORMS, TEXT, "Il caso.", extra=other)
    assert ptw.REVIEW_REQUIREMENT_TYPE not in seen["system"] and not other.get("ran")      # only §1 and §7

    def fail(*a, **k):
        raise RuntimeError("down")
    monkeypatch.setattr(ptw.dg, "_chat_json", fail)
    failed = {"threats": THREATS, "sections": [LEAD]}
    ptw.review_section(LEAD, TEXT, "Il caso.", extra=failed)
    assert failed["failed"] == [LEAD]


def test_record_check_takes_the_reviews_verdict_and_the_word_test_only_when_it_failed():
    threats = {"584 c.p.": {"rubrica": "Omicidio preterintenzionale", "elements": THREATS["584 c.p."]}}
    wordy = "L'art. 584 c.p. richiede il dolo specifico per la morte."
    base = {"asserting": [LEAD, CLOSING], "threats": threats, "allowed": {"584 c.p."}}
    ran = record_check.check([(LEAD, f"\n\n{wordy} Altro.\n\n")], {**base, "requirement_review": {"ran": [LEAD],
                                                                                                 "flagged": []}})[0]
    assert wordy in ran[0][1]                                          # the review answered and flagged nothing
    flagged = record_check.check([(LEAD, f"\n\n{wordy} Altro.\n\n")], {**base, "requirement_review": {
        "ran": [LEAD], "flagged": [wordy]}})
    assert wordy not in flagged[0][0][1] and flagged[1]["rules"] == {"requisito_non_registrato": 1}
    fallback = record_check.check([(LEAD, f"\n\n{wordy} Altro.\n\n")], {**base, "requirement_review": {
        "failed": [LEAD]}})[0]
    assert wordy not in fallback[0][1]                                 # the call failed: the word test


# --- 12. phantom letters and numbers ------------------------------------------------------------------------------------------

T61 = ("Aggravano il reato le circostanze seguenti: 1) l'avere agito per motivi abietti o futili; 2) l'avere adoperato "
       "sevizie; 3) l'avere, nei delitti colposi, agito nonostante la previsione dell'evento; 4) l'avere adoperato "
       "sevizie o agito con crudeltà verso le persone.")


def test_a_letter_or_a_number_the_article_lacks_is_a_phantom():
    items = {"61 c.p.": record_check.article_items(T61)}
    assert record_check.phantom_item("Ricorre l'aggravante dell'art. 61, n. 4 c.p.", [], items) == ""
    assert "il numero 22)" in record_check.phantom_item("Ricorre l'art. 61, n. 22 c.p.", [], items)
    assert "la lettera b)" in record_check.phantom_item("La lettera b) dell'art. 61 c.p. non rileva.", [], items)
    cut = {"61 c.p.": record_check.article_items(T61, cut=len(T61) + 10)}
    assert record_check.phantom_item("Ricorre l'art. 61, n. 22 c.p.", [], cut) == ""          # the text may be cut
    rules = [{"ref": ("61", "c.p."), "commi": [(None, T61)]}]
    log = []
    out = flow.phantom_commi_out([(LEAD, "Ricorre l'art. 61, n. 22 c.p. Ricorre l'art. 61, n. 1 c.p.")], rules, log)
    assert "n. 22" not in out[0][1] and "n. 1 c.p." in out[0][1] and len(log) == 1


# --- 13. never «non verificabile» ---------------------------------------------------------------------------------------------

def test_no_term_line_or_sentence_says_non_verificabile(monkeypatch):
    monkeypatch.delenv(fact_signals.TODAY_ENV, raising=False)
    facts = {"events": [{"what": "Caio chiude il cancello", "date": "marzo 2016"}]}
    term = fact_signals.compute_terms(facts)[0]
    assert term["status"] == "NON VERIFICABILE" and "VERIFICABILE" not in term["text"]
    assert fact_signals.TERM_DECORRENZA_FLAG in term["text"] and "con decorrenza da" in term["text"]
    assert answer_invariants.plain_status("termine NON VERIFICABILE") == "termine da verificare"
    log = []
    draft = flow.banned_words_out("Il termine è non verificabile con le date del caso. È NON VERIFICABILE.", log)
    assert "verificabil" not in draft.lower() and len(log) == 2
    assert render_computed._term_line({"ref": "Codice del Consumo", "text": "denuncia: 2 mesi"}) == ""
    assert render_computed._term_line({"ref": "art. 1168 c.c.", "text": "spoglio: un anno"}).startswith("- ")


# --- 14. the store's numbers ---------------------------------------------------------------------------------------------------

def test_a_term_or_an_amount_in_the_lead_is_the_stores():
    rows = [{"_article": ("641", "c.p.c."), "s": {"plain_text": "L'opposizione si propone entro il termine di quaranta "
                                                                 "giorni dalla notificazione."}},
            {"_article": ("650", "c.p."), "s": {"plain_text": "Chiunque non osserva un provvedimento legalmente dato "
                                                               "dall'autorità è punito con l'arresto fino a tre mesi o "
                                                               "con l'ammenda fino a euro 206."}}]
    store = flow.store_numbers(rows)
    assert store["641 c.p.c."]["terms"] == [(40, "giorni", "opposizione")]
    draft = (f"**1. {LEAD}**\n\nL'opposizione va proposta entro trenta giorni (art. 641 c.p.c.).\n\n"
             f"**2. {NORMS}**\n\nEntro trenta giorni (art. 641 c.p.c.).\n\n"
             f"**3. {CLOSING}**\n\nL'opposizione entro 30 giorni (art. 641 c.p.c.).")
    log = []
    out = flow.store_numbers_out(draft, store, [], log)
    sections = dict(answer_ledger.sections_of(out))
    assert "entro 40 giorni (art. 641" in sections[LEAD] and "entro 40 giorni" in sections[CLOSING]
    assert "trenta giorni" in sections[NORMS]                                      # §1 and §7 only
    assert [x["now"] for x in log] == ["40 giorni", "40 giorni"]
    two = {"700 c.p.c.": {"terms": [(10, "giorni", "ricorso"), (30, "giorni", "reclamo")], "amounts": []}}
    left = []
    flow.store_numbers_out(f"**1. {LEAD}**\n\nSi agisce entro 20 giorni (art. 700 c.p.c.).", two, [], left)
    assert left and "now" not in left[0]                                           # ambiguous: logged, left


# --- 15. the store: each term with its subject ---------------------------------------------------------------------------------

def test_each_term_is_read_per_comma_with_its_subject():
    commi = [(1, "L'opposizione si propone con atto di citazione davanti al giudice che ha emesso il decreto, entro il "
                 "termine di quaranta giorni dalla notificazione. Il termine per la costituzione è di dieci giorni."),
             (2, "Entro sessanta giorni dalla notificazione della sentenza può essere proposto appello; la denuncia "
                 "deve essere fatta entro otto giorni."),
             (3, "Entro dieci giorni dalla esecuzione o notificazione del provvedimento, l'imputato può proporre "
                 "richiesta di riesame.")]
    assert [procedural_facts.term_text(t) for t in procedural_facts.termini(commi)] == [      # v66: with their «dal …»
        "opposizione: quaranta giorni dalla notificazione (co. 1)", "costituzione: dieci giorni (co. 1)",
        "appello: sessanta giorni dalla notificazione della sentenza (co. 2)", "denuncia: otto giorni (co. 2)",
        "richiesta di riesame: dieci giorni dalla esecuzione o notificazione del provvedimento (co. 3)"]
    facts = procedural_facts.extract("645", "c.p.c.", commi)
    assert facts["version"] == 4 and len(facts["termini"]) == 5 and facts["termine"] == "quaranta giorni (co. 1)"   # v67: 4
    v2 = {"_article": ("641", "c.p.c."), "_proc": {"version": 2, "termine": "quaranta giorni (co. 1)"},
          "s": {"plain_text": commi[0][1]}}
    assert procedural_facts.by_number([v2], "c.p.c.")["641"]["termini"][0]["subject"] == "opposizione"


def test_the_store_script_shows_any_code_and_writes_nothing_while_showing():
    import fill_procedural_store as fps
    assert fps.shown("641 c.p.c.,585 c.p.p.,582", "c.p.c.") == {"641", "582"}
    assert fps.shown("641 c.p.c.,585 c.p.p.,582", "c.p.p.") == {"585", "582"}
    source = (HERE.parent / "scripts" / "fill_procedural_store.py").read_text(encoding="utf-8")
    assert "if args.write and not args.show:" in source and "| {t.get('subject') or '—'} |" in source
