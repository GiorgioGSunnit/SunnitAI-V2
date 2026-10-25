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
    "clausola_penale": F(contracts=[contract("appalto", "penale per ritardo")], events=[ev("ritardo di tre mesi")]),
    "caparra": F(contracts=[contract("preliminare di vendita", "caparra confirmatoria")]),
    "risoluzione_contratto": F(contracts=[contract("vendita")], requests=["Può risolvere il contratto?"]),
    "vizi_vendita": F(contracts=[contract("compravendita")], events=[ev("la caldaia ha difetti")]),
    "conformita_consumatore": F(parties=[party("Tizio", "consumatore"), party("Alfa srl", "impresa")],
                                contracts=[contract("compravendita di uno smartphone")],
                                events=[ev("lo smartphone non funziona")]),
    "clausole_vessatorie": F(parties=[party("Tizio", "consumatore")],
                             contracts=[contract("abbonamento", "foro esclusivo dell'impresa")]),
    "usura": F(money=[{"amount": "10000", "rate": "25", "period": "annuo", "threshold": "16"}]),
    "contatto_sociale_scuola": F(parties=[party("Marco", "minore", "8")],
                                 events=[ev("durante la ricreazione a scuola Marco cade")],
                                 injuries=[{"who": "Marco", "days_to_heal": "20", "permanent": "no"}]),
    "responsabilita_sanitaria_civile": F(parties=[party("dott. Rossi", "medico")],
                                         events=[ev("intervento chirurgico al ginocchio")],
                                         injuries=[{"who": "Tizio", "days_to_heal": "90", "permanent": "sì"}]),
    "responsabilita_sanitaria_penale": F(parties=[party("dott. Rossi", "medico")],
                                         injuries=[{"who": "Tizio", "days_to_heal": "90", "permanent": "sì"}],
                                         proceedings=[proc("indagini preliminari", authority="Procura")]),
    "albergatore": F(events=[ev("dalla camera d'albergo sono stati rubati i gioielli")]),
    "difformita_appalto": F(contracts=[contract("appalto di ristrutturazione")], events=[ev("infiltrazioni nel bagno")]),
    "rovina_gravi_difetti": F(contracts=[contract("appalto")], events=[ev("lavori ultimati", "2015"),
                                                                       ev("compaiono crepe nei muri portanti", "2025")]),
    "circolazione_veicoli": F(events=[ev("incidente stradale all'incrocio")]),
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
    "ottemperanza": F(proceedings=[proc("ricorso", "sentenza di annullamento del diniego", authority="TAR")],
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
    assert "condominio_parti_comuni" not in _fired(F(events=[ev("l'assemblea dei soci approva il bilancio")]))
    assert "ottemperanza" not in _fired(F(proceedings=[proc("primo grado", "sentenza di condanna", authority="Tribunale")],
                                          events=[ev("non ha eseguito la sentenza")]))
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
                          "fonte": "regola possessoria_preclusa", "ranges": [[1140, 1172], [948, 951], [1079, 1079]]}
    assert len(central) == len({c["nome"] for c in central}) <= fs.MAX_CENTRAL


# --- integration ---------------------------------------------------------------------------------

def test_fired_institutions_join_the_frames_central_issues(monkeypatch):
    monkeypatch.setenv("CASE_FRAME_ENABLED", "true")
    monkeypatch.setattr(dg, "_case_frame", lambda facts, request: {
        "tasks": ["Che cosa può fare Tizio?"], "central_issue_candidates": [{"nome": "usucapione", "codice": "c.c."}]})
    monkeypatch.setattr(fs, "extract_facts", lambda text: (FIXTURES["usucapione_ordinaria"], ""))
    frame, got = dg._frame_and_fact_signals("Tizio coltiva l'orto di Caio dal 1995.", "")
    assert [r["id"] for r in got["fired"]] == ["usucapione_ordinaria"]
    assert [c["nome"] for c in frame["central_issue_candidates"]] == ["usucapione"]      # already there: once
    monkeypatch.setattr(fs, "extract_facts", lambda text: (FIXTURES["fondo_intercluso"], ""))
    frame, _ = dg._frame_and_fact_signals("x", "")
    assert [c["nome"] for c in frame["central_issue_candidates"]] == [
        "usucapione", "passaggio coattivo", "indennità per il passaggio coattivo"]
    assert frame["central_issue_candidates"][1]["ranges"] == [[1051, 1055]]
    analysis = dg._with_central_questions({"questioni": []}, frame)
    assert analysis["questioni"][1]["istituti"][0] == {"nome": "passaggio coattivo", "codice": "c.c.", "numero_indicativo": "",
                                                       "fonte": "regola fondo_intercluso", "ranges": [[1051, 1055]]}


def test_rules_fire_only_within_the_case_areas(monkeypatch):
    monkeypatch.setenv("CASE_FRAME_ENABLED", "true")
    facts = FIXTURES["legittima_difesa"]
    assert "legittima_difesa" in {r["id"] for r in fs.fire_rules(facts, [], areas=["penale"])}
    assert fs.fire_rules(facts, [], areas=["contratti", "responsabilita_civile"]) == []
    assert "legittima_difesa" in {r["id"] for r in fs.fire_rules(facts, [], areas=["altro"])}         # unknown: no gate
    usura = FIXTURES["usura"]
    assert {r["id"] for r in fs.fire_rules(usura, [], areas=["penale"])} == {"usura"}                 # both families
    monkeypatch.setattr(dg, "_case_frame", lambda facts, request: {"areas": ["contratti"], "central_issue_candidates": []})
    monkeypatch.setattr(fs, "extract_facts", lambda text: (facts, ""))
    frame, got = dg._frame_and_fact_signals("x", "")
    assert got["fired"] == [] and got["areas"] == ["contratti"]


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
    assert "    - perché: un fondo senza accesso alla via pubblica — «il fondo di tizio è intercluso»" in text


# --- the case's present ------------------------------------------------------------------------------

def test_without_a_present_date_no_term_runs_to_the_real_today(monkeypatch):
    monkeypatch.delenv(fs.TODAY_ENV)
    lonely = F(events=[ev("Caio chiude il cancello", "marzo 2016")])
    t = fs.compute_terms(lonely)[0]
    assert t["status"] == "NON VERIFICABILE" and "il caso non indica la data attuale" in t["text"]
    assert "possessoria_preclusa" not in _fired(lonely) and "possessoria_in_termine" not in _fired(lonely)
    later = F(events=[ev("Caio chiude il cancello", "marzo 2016"), ev("Tizio scrive al vicino", "giugno 2018")])
    t = fs.compute_terms(later)[0]
    assert t["status"] == "DECORSO" and "già al fatto più recente" in t["text"]      # run even by the latest fact
    dated = F(events=[ev("Caio chiude il cancello", "marzo 2016")], present_date="10 aprile 2016")
    assert fs.compute_terms(dated)[0]["status"] == "NON DECORSO"
    assert "usucapione_ordinaria" not in _fired(F(possession=[{"who": "T", "asset": "orto", "since": "2000", "how": "",
                                                                "still_ongoing": "sì"}]))


def test_a_present_date_is_kept_only_as_the_text_writes_it():
    facts = fs.validate_facts({"present_date": "oggi, 10 marzo 2019", "events": []}, "Siamo a oggi, 10 marzo 2019.")
    assert facts["present_date"] == "oggi, 10 marzo 2019"
    assert fs.validate_facts({"present_date": "1 gennaio 2030", "events": []}, "Nessuna data.")["present_date"] == ""


def test_a_cut_reply_keeps_its_complete_values(monkeypatch):
    cut = '{"parties": [{"name": "Tizio", "role": "cliente"}], "events": [{"date": "2016", "what": "a \\"b\\""}, {"date": "20'
    assert fs._parse_or_repair(cut) == {"parties": [{"name": "Tizio", "role": "cliente"}],
                                        "events": [{"date": "2016", "what": 'a "b"'}]}
    assert fs._parse_or_repair('```json\n{"events": []}\n```') == {"events": []}
    monkeypatch.setattr(fs, "_call_chat", lambda messages, **kw: cut)
    facts, problem = fs.extract_facts("Tizio, nel 2016, a.")
    assert facts["parties"][0]["name"] == "Tizio" and problem == ""


# --- why, and which articles ---------------------------------------------------------------------------

def test_a_fired_rule_says_why():
    rule = next(r for r in fs.fire_rules(FIXTURES["fondo_intercluso"], []) if r["id"] == "fondo_intercluso")
    assert rule["matched"] == ["un fondo senza accesso alla via pubblica — «il fondo di tizio è intercluso»"]
    rule = next(r for r in fs.fire_rules(FIXTURES["medico_e_latitante"], []) if r["id"] == "medico_e_latitante")
    assert rule["matched"][0] == "una parte è medico o professionista o struttura sanitaria — «dott. Verdi (medico)»"


def test_a_rule_institution_keeps_only_articles_in_range_whose_rubrica_fits():
    def row(number, rubrica):
        return {"_article": (number, "c.c."), "s": {"titolo_articolo": rubrica}}
    istituto = {"nome": "passaggio coattivo", "codice": "c.c.", "fonte": "regola fondo_intercluso", "ranges": [[1051, 1055]]}
    rows = [row("1051", "Passaggio coattivo"), row("2634", "Infedeltà patrimoniale"), row("1052", "Passaggio coattivo a favore di fondo non intercluso"),
            row("1053", "Indennità")]
    assert [r["_article"][0] for r in fs.keep_resolved(istituto, rows)] == ["1051", "1052"]
    out_of_range = {"nome": "danno non patrimoniale", "fonte": "regola x", "ranges": [[2043, 2059]]}
    assert fs.keep_resolved(out_of_range, [row("2634", "Danno non patrimoniale della società")]) == []
    assert fs.keep_resolved({"nome": "passaggio coattivo"}, rows) == rows                # not a fact rule's: unchanged
    assert fs.keep_resolved({"nome": "accettazione tacita dell'eredità", "fonte": "r"}, [row("476-bis", "Accettazione tacita")])


# --- answers to .docx, blind A/B --------------------------------------------------------------------------

def test_an_answer_and_its_sources_go_to_a_docx(tmp_path):
    from docx import Document
    docx_script = _script("answer_docx")
    record = {"status": "ok", "request": "Che cosa può fare Tizio?",
              "answer": "**1. Risposte ai quesiti**\n\nTizio può agire ex **art. 1051 c.c.**\n- prova del passaggio",
              "citations": [{"document_name": "Codice Civile 2026", "document_type": "primary",
                             "sections": [{"name": "1051.0.0", "title": "Passaggio coattivo", "plain_text": "Il proprietario..."}]},
                            {"document_name": "Cass. civ., sez. II, n. 1/2026", "document_type": "interpretation",
                             "sections": [{"name": "1", "title": "", "plain_text": "Il fondo intercluso ha diritto..."}]}]}
    out = str(tmp_path / "r.docx")
    docx_script.build(record, out, "Risposta 1", record["request"], None)
    text = "\n".join(p.text for p in Document(out).paragraphs)
    for part in ("Risposta 1", "Che cosa può fare Tizio?", "1. Risposte ai quesiti", "Tizio può agire ex art. 1051 c.c.",
                 "prova del passaggio", "Fonti citate", "Norme", "Codice Civile 2026", "Art. 1051 — Passaggio coattivo",
                 "Giurisprudenza", "Il fondo intercluso ha diritto..."):
        assert part in text
    assert "Configurazione" not in text                                          # blind
    docx_script.build(record, out, "Risposta", "", ["FACT_SIGNALS_ENABLED=true"])
    assert "FACT_SIGNALS_ENABLED=true" in "\n".join(p.text for p in Document(out).paragraphs)


def test_the_ab_order_is_random_and_the_settings_parse():
    ab = _script("ab_answers")
    assert ab.settings("A=1, B=2,") == ["A=1", "B=2"] and ab.settings("") == []
    assert ab.assign(1) == ab.assign(1) and sorted(ab.assign(1).values()) == ["A", "B"]
    assert {tuple(ab.assign(seed).values()) for seed in range(20)} == {("A", "B"), ("B", "A")}
