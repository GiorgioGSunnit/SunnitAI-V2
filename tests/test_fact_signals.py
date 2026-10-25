"""FACT_SIGNALS_ENABLED (src/rag/fact_signals.py, fact_signal_rules.yaml): date arithmetic, typed facts,
one invented case per rule. Synthetic facts only -- never the held-out texts."""

import datetime as dt
import json
import os

import pytest

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

import src.rag.defensive_generation as dg  # noqa: E402
import src.rag.plan_then_write as ptw  # noqa: E402
from src.rag import fact_signals as fs  # noqa: E402

TODAY = "2026-10-27"


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    monkeypatch.setenv(fs.TODAY_ENV, TODAY)
    monkeypatch.delenv(fs.FACT_SIGNALS_FLAG, raising=False)


def F(**kw):
    base = {k: ([] if isinstance(v, list) else "") for k, v in fs.SCHEMA.items()}
    base.update(kw)
    return base


def ev(what, date=""):
    return {"date": date, "what": what, "who": ""}


def party(name, role, age=""):
    return {"name": name, "role": role, "age": age}


def contract(type_, clauses=""):
    return {"type": type_, "parties": "", "clauses": clauses}


def proc(stage, decision="", date="", authority=""):
    return {"stage": stage, "authority": authority, "decision": decision, "date": date}


# --- dates and terms -------------------------------------------------------------------------------

def test_dates_as_written_with_their_precision():
    assert fs.parse_date("il 12/03/2016") == (dt.date(2016, 3, 12), dt.date(2016, 3, 12), "2016-03-12")
    assert fs.parse_date("1° marzo 2016")[2] == "2016-03-01"
    assert fs.parse_date("nel marzo del 2016") == (dt.date(2016, 3, 1), dt.date(2016, 3, 31), "2016-03")
    assert fs.parse_date("nel 2016") == (dt.date(2016, 1, 1), dt.date(2016, 12, 31), "2016")
    assert fs.parse_date("2016-03-12")[2] == "2016-03-12"
    assert fs.parse_date("tre mesi dopo") is None and fs.parse_date("31/02/2016")[2] == "2016"   # no such day: its year


def test_periods_and_status_respect_the_precision():
    assert fs.add_period(dt.date(2024, 2, 29), years=1) == dt.date(2025, 2, 28)
    assert fs.add_period(dt.date(2026, 1, 31), months=1) == dt.date(2026, 2, 28)
    year_2016, year_2018 = fs.parse_date("2016")[:2], fs.parse_date("2018")[:2]
    assert fs.term_status(year_2016, year_2018, years=1) == "DECORSO"         # even 31 Dec 2016 -> 1 Jan 2018
    assert fs.term_status(year_2016, fs.parse_date("2017")[:2], years=1) == "INCERTO"
    day = fs.parse_date("1 marzo 2026")[:2]
    assert fs.term_status(day, fs.parse_date("20 marzo 2026")[:2], months=3) == "NON DECORSO"


def test_the_possessory_term_reads_as_the_lawyer_writes_it():
    terms = fs.compute_terms(F(events=[ev("Caio chiude il cancello del passaggio", "2016"),
                                       ev("Tizio propone ricorso possessorio", "2018")]))
    t = next(t for t in terms if t["id"] == "possessoria")
    assert t["status"] == "DECORSO" and t["ref"] == "artt. 1168 e 1170 c.c."
    assert t["text"] == ("azione possessoria (spoglio o turbativa): termine annuale DECORSO (Caio chiude il "
                         "cancello del passaggio 2016, Tizio propone ricorso possessorio 2018)")


def test_a_term_with_no_act_runs_to_today():
    terms = fs.compute_terms(F(events=[ev("Caio occupa il cortile", "1 settembre 2026")]))
    t = next(t for t in terms if t["id"] == "possessoria")
    assert t["status"] == "NON DECORSO" and t["to"] == {"what": "oggi", "date": TODAY}


def test_criminal_appeal_terms_follow_the_deposit_of_the_reasons():
    sentence = proc("primo grado", "sentenza di condanna", "10 settembre 2026")
    t = next(t for t in fs.compute_terms(F(proceedings=[sentence])) if t["id"] == "impugnazione_penale")
    assert t["status"] == "INCERTO" and "dipende da quando è depositata la motivazione" in t["text"]
    deposited = F(proceedings=[sentence], events=[ev("deposito della motivazione", "1 ottobre 2026")])
    assert next(t for t in fs.compute_terms(deposited) if t["id"] == "impugnazione_penale")["status"] == "NON DECORSO"
    late = F(proceedings=[proc("primo grado", "sentenza di condanna", "10 gennaio 2026")])
    assert next(t for t in fs.compute_terms(late) if t["id"] == "impugnazione_penale")["status"] == "DECORSO"


def test_usucapione_and_tender_terms():
    held = F(possession=[{"who": "Tizio", "asset": "fondo di Caio", "since": "1995", "how": "pacifico",
                          "still_ongoing": "sì"}])
    assert next(t for t in fs.compute_terms(held) if t["id"] == "usucapione_ordinaria")["status"] == "DECORSO"
    car = F(possession=[{"who": "Tizio", "asset": "auto di Caio", "since": "2020", "how": "", "still_ongoing": "sì"}])
    assert [t["id"] for t in fs.compute_terms(car)] == ["usucapione_mobili"]
    tender = F(proceedings=[proc("gara", "esclusione dalla gara", "1 ottobre 2026")])
    ids = {t["id"]: t["status"] for t in fs.compute_terms(tender)}
    assert ids == {"ricorso_tar_appalti": "NON DECORSO"}


# --- typed facts ---------------------------------------------------------------------------------

def test_a_date_the_text_does_not_hold_is_dropped():
    case = "Nel 2016 Caio chiuse il cancello. Tizio agì due anni dopo."
    raw = {"events": [{"date": "2016", "what": "Caio chiude il cancello"}, {"date": "2018", "what": "Tizio agisce"}],
           "parties": [{"name": "Tizio", "role": "cliente", "age": 40}], "extra": "dropped", "requests": "Che fare?"}
    facts = fs.validate_facts(raw, case)
    assert facts["events"][0]["date"] == "2016" and facts["events"][0]["date_iso"] == "2016"
    assert facts["events"][1]["date"] == "" and facts["events"][1]["date_iso"] == ""      # 2018 is not in the text
    assert facts["parties"][0]["age"] == "40" and facts["requests"] == ["Che fare?"] and "extra" not in facts
    assert fs.validate_facts(["not", "a", "dict"], case) is None and fs.validate_facts({"foo": 1}, case) is None


def test_extraction_is_one_call_retried_once(monkeypatch):
    replies = iter(["non è JSON", json.dumps({"events": [{"date": "", "what": "x"}]})])
    calls = []

    def chat(messages, max_tokens=None, temperature=None, **kw):
        calls.append((messages[0].content, temperature))
        return next(replies)
    monkeypatch.setattr(fs, "_call_chat", chat)
    facts, problem = fs.extract_facts("Un caso.")
    assert facts["events"][0]["what"] == "x" and problem == "" and len(calls) == 2
    assert calls[0][1] == 0 and calls[1][0].endswith(fs._RETRY_SUFFIX)


# --- the rules: one invented case each -------------------------------------------------------------

FIXTURES = {
    "destinazione_padre_famiglia": F(property_history=[{"asset": "fondo A", "owner": "Mevio"}, {"asset": "fondo B", "owner": "Mevio"}],
                                     works_or_signs=[{"what": "strada sterrata", "visible": "sì", "existing_since": ""}]),
    "usucapione_servitu_apparente": F(works_or_signs=[{"what": "sentiero", "visible": "sì", "existing_since": "1990"}],
                                      possession=[{"who": "Tizio", "asset": "sentiero", "since": "1990", "how": "pacifico",
                                                   "still_ongoing": "sì"}]),
    "usucapione_ordinaria": F(possession=[{"who": "Tizio", "asset": "orto", "since": "1995", "how": "pacifico e pubblico",
                                           "still_ongoing": "sì"}]),
    "possesso_vizioso": F(possession=[{"who": "Tizio", "asset": "orto", "since": "2010", "how": "clandestino",
                                       "still_ongoing": "sì"}]),
    "possessoria_preclusa": F(events=[ev("Caio chiude il cancello del passaggio", "marzo 2016"),
                                      ev("Tizio propone ricorso possessorio", "2018")]),
    "possessoria_in_termine": F(events=[ev("Caio chiude il cancello", "1 settembre 2026")]),
    "fondo_intercluso": F(events=[ev("il fondo di Tizio è intercluso")]),
    "immissioni": F(events=[ev("odori e fumi dal ristorante vicino")]),
    "distanze_vedute": F(events=[ev("Caio costruisce un muro a un metro dal confine")]),
    "condominio_parti_comuni": F(events=[ev("l'assemblea del condominio vieta l'uso del cortile")]),
    "inadempimento_mora": F(contracts=[contract("fornitura")], events=[ev("il fornitore non ha consegnato la merce")]),
    "clausola_penale": F(contracts=[contract("appalto", "penale per ritardo")], events=[ev("ritardo di tre mesi")]),
    "caparra": F(contracts=[contract("preliminare di vendita", "caparra confirmatoria")]),
    "risoluzione_contratto": F(contracts=[contract("vendita")], requests=["Può risolvere il contratto?"]),
    "vizi_vendita": F(contracts=[contract("compravendita")], events=[ev("la caldaia ha difetti")]),
    "conformita_consumatore": F(parties=[party("Tizio", "consumatore"), party("Alfa srl", "impresa")],
                                events=[ev("lo smartphone non funziona")]),
    "clausole_vessatorie": F(parties=[party("Tizio", "consumatore")],
                             contracts=[contract("abbonamento", "foro esclusivo dell'impresa")]),
    "usura": F(money=[{"amount": "10000", "rate": "25", "period": "annuo", "threshold": "16"}]),
    "contatto_sociale_scuola": F(parties=[party("Marco", "minore", "8")],
                                 events=[ev("durante la ricreazione a scuola Marco cade")],
                                 injuries=[{"who": "Marco", "days_to_heal": "20", "permanent": "no"}]),
    "responsabilita_sanitaria_civile": F(parties=[party("dott. Rossi", "medico")],
                                         injuries=[{"who": "Tizio", "days_to_heal": "90", "permanent": "sì"}]),
    "responsabilita_sanitaria_penale": F(parties=[party("dott. Rossi", "medico")],
                                         injuries=[{"who": "Tizio", "days_to_heal": "90", "permanent": "sì"}],
                                         proceedings=[proc("indagini preliminari", authority="Procura")]),
    "albergatore": F(events=[ev("dalla camera d'albergo sono stati rubati i gioielli")]),
    "difformita_appalto": F(contracts=[contract("appalto di ristrutturazione")], events=[ev("infiltrazioni nel bagno")]),
    "rovina_gravi_difetti": F(contracts=[contract("appalto")], events=[ev("lavori ultimati", "2015"),
                                                                       ev("compaiono crepe nei muri portanti", "2025")]),
    "danno_cose_custodia": F(events=[ev("Tizio inciampa in una buca del marciapiede")]),
    "circolazione_veicoli": F(events=[ev("incidente stradale all'incrocio")]),
    "danno_non_patrimoniale": F(injuries=[{"who": "Tizio", "days_to_heal": "60", "permanent": "no"}]),
    "prescrizione_decorsa": F(events=[ev("incidente stradale", "2015")]),
    "tentativo": F(events=[ev("Tizio si è procurato un piede di porco e ha fatto un sopralluogo")]),
    "concorso_persone": F(parties=[party("Tizio", "indagato/imputato"), party("Caio", "indagato/imputato")],
                          proceedings=[proc("indagini", authority="Procura")]),
    "concorso_extraneus": F(parties=[party("Rossi", "pubblico ufficiale, indagato/imputato"),
                                     party("Bianchi", "indagato/imputato")], proceedings=[proc("indagini")]),
    "morte_o_lesioni_da_altro_delitto": F(events=[ev("Tizio spinge Caio durante una lite")],
                                          injuries=[{"who": "Caio", "days_to_heal": "", "permanent": "morte"}]),
    "medico_e_latitante": F(parties=[party("dott. Verdi", "medico")],
                            events=[ev("il medico cura un uomo ricercato, ferito da colpo di arma da fuoco"),
                                    ev("il medico non ha avvisato le autorità")]),
    "querela_tardiva": F(events=[ev("Caio minaccia Tizio", "gennaio 2026")]),
    "impugnazione_penale": F(proceedings=[proc("primo grado", "sentenza di condanna", "10 settembre 2026")]),
    "legittima_difesa": F(events=[ev("Tizio, aggredito da Caio, ha reagito colpendolo")]),
    "ottemperanza": F(proceedings=[proc("TAR", "sentenza di annullamento del diniego")],
                      events=[ev("il Comune non ha dato esecuzione alla sentenza")]),
    "riadozione_senza_contraddittorio": F(events=[ev("il provvedimento annullato è stato riadottato con motivazione "
                                                     "per relationem")]),
    "esclusione_gara": F(events=[ev("la società è stata esclusa dalla gara")]),
    "ricorso_tar_decorso": F(parties=[party("Comune di X", "pubblica amministrazione")],
                             proceedings=[proc("provvedimento", "diniego del permesso", "1 marzo 2026")]),
    "silenzio_amministrazione": F(parties=[party("Comune di X", "pubblica amministrazione")],
                                  events=[ev("il Comune non ha risposto all'istanza")]),
    "comunione_de_residuo": F(events=[ev("durante il matrimonio la moglie acquista un locale per la sua impresa")]),
    "mantenimento": F(events=[ev("separazione dei coniugi; il padre non versa il mantenimento")]),
    "accettazione_tacita": F(parties=[party("Anna", "erede")],
                             events=[ev("Anna riscuote i canoni dell'appartamento del defunto")]),
    "lesione_legittima": F(events=[ev("con testamento il padre lascia tutto al figlio maggiore, escludendo la figlia")]),
    "morosita_conduttore": F(contracts=[contract("locazione abitativa")], events=[ev("il conduttore non paga i canoni")]),
    "finita_locazione": F(contracts=[contract("locazione")],
                          events=[ev("alla scadenza il conduttore non vuole lasciare l'immobile")]),
    "licenziamento": F(parties=[party("Alfa srl", "datore di lavoro"), party("Tizio", "lavoratore")],
                       events=[ev("il lavoratore viene licenziato")]),
    "differenze_retributive": F(parties=[party("Alfa srl", "datore di lavoro"), party("Tizio", "lavoratore")],
                                events=[ev("gli straordinari non sono stati pagati")]),
}


def _fired(facts):
    return {r["id"] for r in fs.fire_rules(facts, fs.compute_terms(facts))}


def test_every_rule_has_an_invented_case():
    assert {r["id"] for r in fs.load_rules()} == set(FIXTURES)


@pytest.mark.parametrize("rule_id", sorted(FIXTURES))
def test_each_rule_fires_on_its_case(rule_id):
    assert rule_id in _fired(FIXTURES[rule_id])


def test_rules_that_must_not_fire():
    violent = F(possession=[{"who": "Tizio", "asset": "orto", "since": "1995", "how": "violento", "still_ongoing": "sì"}])
    assert "usucapione_ordinaria" not in _fired(violent) and "possesso_vizioso" in _fired(violent)
    recent = F(possession=[{"who": "Tizio", "asset": "orto", "since": "2015", "how": "pacifico", "still_ongoing": "sì"}])
    assert "usucapione_ordinaria" not in _fired(recent)
    assert "possessoria_in_termine" not in _fired(FIXTURES["possessoria_preclusa"])
    assert "responsabilita_sanitaria_civile" not in _fired(FIXTURES["responsabilita_sanitaria_penale"])
    assert "danno_cose_custodia" not in _fired(F(contracts=[contract("noleggio")], events=[ev("una buca")]))
    criminal = {r["area"] for r in fs.fire_rules(FIXTURES["concorso_persone"], [])}
    assert criminal == {"penale"}
    assert _fired(F()) == set() and fs.fire_rules(None, []) == []


def test_rules_name_institutions_never_articles(tmp_path):
    for rule in fs.load_rules():
        assert rule["status"] == "experiment" and rule["codice"]
    bad = tmp_path / "bad.yaml"
    bad.write_text("- {id: x, area: penale, when: [{fact: {pattern: 'a'}}], institution: 'art. 56 c.p.', "
                   "codice: c.p., why: y, status: experiment}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="names an article"):
        fs.load_rules(str(bad))
    bad.write_text("- {id: x, area: penale, when: [{guess: 1}], institution: tentativo, codice: c.p., why: y, "
                   "status: experiment}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown condition"):
        fs.load_rules(str(bad))


def test_central_issues_once_each_and_capped():
    fired = fs.fire_rules(FIXTURES["possessoria_preclusa"], fs.compute_terms(FIXTURES["possessoria_preclusa"]))
    central = fs.central_issues(fired + fired)
    assert central[0] == {"nome": "decadenza dall'azione di reintegrazione e di manutenzione", "codice": "c.c.",
                          "fonte": "regola possessoria_preclusa"}
    assert len(central) == len({c["nome"] for c in central}) <= fs.MAX_CENTRAL


# --- integration ---------------------------------------------------------------------------------

def test_fired_institutions_join_the_frames_central_issues(monkeypatch):
    monkeypatch.setenv("CASE_FRAME_ENABLED", "true")
    monkeypatch.setattr(dg, "_case_frame", lambda facts, request: {
        "tasks": ["Che cosa può fare Tizio?"], "central_issue_candidates": [{"nome": "usucapione", "codice": "c.c."}]})
    signals = {"facts": FIXTURES["usucapione_ordinaria"], "terms": [], "fired": fs.fire_rules(
        FIXTURES["usucapione_ordinaria"], []), "problem": ""}
    monkeypatch.setattr(fs, "analyse", lambda text: signals)
    frame, got = dg._frame_and_fact_signals("Tizio coltiva l'orto di Caio dal 1995.", "")
    assert got is signals
    assert [c["nome"] for c in frame["central_issue_candidates"]] == ["usucapione"]      # already there: once
    monkeypatch.setattr(fs, "analyse", lambda text: {**signals, "fired": fs.fire_rules(FIXTURES["fondo_intercluso"], [])})
    frame, _ = dg._frame_and_fact_signals("x", "")
    assert [c["nome"] for c in frame["central_issue_candidates"]] == [
        "usucapione", "passaggio coattivo", "indennità per il passaggio coattivo"]


def test_computed_terms_reach_the_plan_and_the_procedural_section():
    terms = fs.compute_terms(FIXTURES["possessoria_preclusa"])
    analysis = {"fact_signals": {"terms": terms}}
    human = ptw._plan_human("Fatti.", "", analysis, "NORME", "", "", [])
    assert "TERMINI CALCOLATI DALLE DATE DEL CASO:\n- azione possessoria (spoglio o turbativa): termine annuale DECORSO" in human
    assert "non va proposta come praticabile" in human
    plan = {"parte_assistita": "Tizio", "posizione": "", "obiettivo": "", "questione_centrale": "", "risposte": [],
            "questioni": [{"titolo": "Passaggio", "norme": [], "pronunce": [], "argomento": "x", "obiezione": "",
                           "replica": ""}], "subordinate": [], "punti_deboli": []}
    jobs = ptw.section_jobs(plan, "Fatti.", "", analysis, [], {}, "")
    title, instruction, material, _ = jobs[4]
    assert title == "Profili processuali e termini" and "TERMINI CALCOLATI DALLE DATE DEL CASO:\n- azione possessoria" in material
    assert "non va proposta come praticabile" in instruction
    assert "TERMINI CALCOLATI" not in ptw._plan_human("Fatti.", "", {}, "NORME", "", "", [])          # flag off: nothing


def _script(name):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent.parent / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_lawyers_table_lists_every_rule_in_italian():
    text = _script("export_fact_rules").markdown(fs)
    assert text.count("| in sperimentazione (approvata per l'esperimento) |") == len(fs.load_rules())
    assert ("| fondo_intercluso | Diritti reali e possesso | un fondo senza accesso alla via pubblica | passaggio "
            "coattivo; indennità per il passaggio coattivo | c.c. |") in text
    assert "| azione possessoria (spoglio o turbativa) | 1 anno | artt. 1168 e 1170 c.c. |" in text


def test_the_bench_counts_lost_articles_a_rule_reaches():
    bench = _script("fact_signals_bench")
    case = {"id": "x1", "slots": [{"label": "art. 1051 cc", "pool_label": "1051 c.c.", "stage": "LOST_AT_UNDERSTANDING"}]}
    facts = FIXTURES["fondo_intercluso"]
    signals = {"facts": facts, "terms": [], "fired": fs.fire_rules(facts, []) + fs.fire_rules(FIXTURES["caparra"], []),
               "problem": ""}
    resolved = {"passaggio coattivo": ["1051 c.c."], "indennità per il passaggio coattivo": ["1053 c.c."],
                "caparra confirmatoria": ["1385 c.c."], "recesso per inadempimento": []}
    result = {"case": "x1", "signals": signals, "resolved": resolved, "lost": case["slots"]}
    ekey = lambda label: (label.split()[0], {"c.c.": "cc"}.get(label.split(" ", 1)[1], label.split(" ", 1)[1]))  # noqa: E731
    text = bench.render("/runs/r", [result], ekey, {"x1": {("1051", "cc")}}, 60, [])
    assert "LOST_AT_UNDERSTANDING in the run: 1; reached by a fired rule: 1 (x1 art. 1051 cc ← fondo_intercluso)" in text
    assert "possible false triggers): 1 (x1 caparra)" in text and "**1051 c.c.**, 1053 c.c." in text
