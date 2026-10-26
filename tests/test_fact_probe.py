"""Typed facts (27 Oct): roles from a list with qualified roles only when stated, owners and common ownership
with their period, pre-existing works, event kinds that drive the terms, possession with its end; every term
and rule explained (scripts/fact_probe.py); the facts whole in the vlog and the dry run's record.
Invented narratives only."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

from src.rag import fact_signals as fs  # noqa: E402

HERE = Path(__file__).resolve().parent


def _script(name):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CASE = ("Le particelle 10 e 11 appartenevano a Rossi fino al 1960, quando le divise tra i figli Anna e Bruno. "
        "La stradina che dalla particella 10 porta alla provinciale attraversa la particella 11 ed esisteva già "
        "allora. Anna, insegnante, e poi i suoi eredi l'hanno usata pacificamente dal 1960 fino al 2014. "
        "Nel marzo 2015 Bruno mette una catena all'ingresso della stradina. Anna si rivolge al giudice nel 2017. "
        "Il comune nel 2015 ha emesso un'ordinanza sulla viabilità. Che azioni ha Anna?")
REPLY = {
    "parties": [{"name": "Anna", "role": "parte assistita, incaricato di pubblico servizio", "age": ""},
                {"name": "Bruno", "role": "proprietario del fondo servente", "age": ""},
                {"name": "Rossi", "role": "dante causa", "age": ""}],
    "property_history": [{"asset": "particella 10", "owner": "Rossi", "period": "fino al 1960",
                          "together_with": "particella 11", "how_acquired": "", "ended_by": "divise tra i figli"}],
    "works_or_signs": [{"what": "stradina", "visible": "sì", "existing_since": "", "before_separation": "sì",
                        "serves": "particella 10"}],
    "possession": [{"who": "Anna", "asset": "stradina", "since": "1960", "until": "2014", "how": "pacificamente",
                    "use": "passaggio", "still_ongoing": "no"}],
    "events": [{"date": "marzo 2015", "what": "Bruno mette una catena all'ingresso della stradina", "who": "Bruno",
                "kind": "chiusura/ostruzione accesso"},
               {"date": "2017", "what": "Anna si rivolge al giudice", "who": "Anna", "kind": "azione giudiziaria"},
               {"date": "2015", "what": "il comune chiude la strada al passaggio con un'ordinanza", "who": "comune",
                "kind": "provvedimento"},
               {"date": "", "what": "Bruno sfratta Anna", "who": "Bruno", "kind": "sfratto"}],
    "requests": ["Che azioni ha Anna?"],
}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv(fs.TODAY_ENV, raising=False)
    monkeypatch.delenv(fs.FACT_SIGNALS_FLAG, raising=False)
    monkeypatch.delenv(fs.REGIMES_FLAG, raising=False)


def facts():
    return fs.validate_facts(json.loads(json.dumps(REPLY)), CASE)


# --- roles -----------------------------------------------------------------------------------------------

def test_a_qualified_role_needs_the_text_to_state_it_and_the_default_is_privato():
    roles = {p["name"]: p["role"] for p in facts()["parties"]}
    assert roles["Anna"] == "cliente, privato"                       # "incaricato di pubblico servizio" dropped
    assert roles["Bruno"] == "proprietario"
    assert roles["Rossi"] == "privato"                               # outside the list
    case = "La dottoressa Neri, medico del pronto soccorso, visita Carlo. L'ospedale San Luca lo dimette."
    reply = {"parties": [{"name": "Neri", "role": "medico"}, {"name": "Carlo", "role": "vittima, medico"},
                         {"name": "Ospedale San Luca", "role": "struttura sanitaria"}]}
    roles = {p["name"]: p["role"] for p in fs.validate_facts(reply, case)["parties"]}
    assert roles == {"Neri": "medico", "Carlo": "persona offesa", "Ospedale San Luca": "struttura sanitaria"}
    case = "Marco, agente della polizia locale, ferma Luca davanti al negozio."
    reply = {"parties": [{"name": "Marco", "role": "pubblico ufficiale"}, {"name": "Luca", "role": "pubblico ufficiale"},
                         {"name": "il vicino", "role": "incaricato di pubblico servizio"}]}
    roles = [p["role"] for p in fs.validate_facts(reply, case)["parties"]]
    assert roles == ["pubblico ufficiale", "privato", "privato"]       # not in the text: never qualified


# --- owners, works, events, possession -------------------------------------------------------------------

def test_owners_with_their_period_and_common_ownership_are_typed():
    item = facts()["property_history"][0]
    assert item["period"] == "fino al 1960" and item["period_iso"] == "1960"
    assert item["ended_by"] == "divisione" and item["together_with"] == "particella 11"
    assert fs.holds({"common_previous_owner": True}, facts(), [])
    sold = {"property_history": [{"asset": "villa", "owner": "Gallo", "how_acquired": "acquistata con rogito"}]}
    assert fs.validate_facts(sold, "Gallo ha la villa.")["property_history"][0]["how_acquired"] == "compravendita"


def test_a_work_existing_before_the_separation_makes_the_father_of_the_family_rule_fire():
    f = facts()
    fired = [r["id"] for r in fs.fire_rules(f, fs.compute_terms(f))]
    assert "destinazione_padre_famiglia" in fired
    later = json.loads(json.dumps(REPLY))
    later["works_or_signs"][0]["before_separation"] = "no"
    f = fs.validate_facts(later, CASE)
    assert "destinazione_padre_famiglia" not in [r["id"] for r in fs.fire_rules(f, fs.compute_terms(f))]


def test_event_kinds_drive_the_terms_not_their_words():
    f = facts()
    assert [e["kind"] for e in f["events"]] == ["chiusura/ostruzione accesso", "azione giudiziaria",
                                                "provvedimento", "altro"]
    t = next(t for t in fs.compute_terms(f) if t["id"] == "possessoria")
    # "una catena" and "si rivolge al giudice" match neither old pattern; the ordinance "chiude la strada" (a
    # provvedimento) is not a spoglio although its words would be.
    assert t["status"] == "DECORSO" and t["from"] == {"what": "Bruno mette una catena all'ingresso della", "date": "marzo 2015"}
    assert t["to"]["date"] == "2017"
    assert fs.holds({"event": {"kind": ["chiusura/ostruzione accesso"]}}, f, [])
    assert not fs.holds({"event": {"kind": "spoglio"}}, f, [])
    untyped = fs.validate_facts({"events": [{"date": "2016", "what": "Caio chiude il cancello del passaggio"}]},
                                "Nel 2016 Caio chiude il cancello.")
    assert any(t["id"] == "possessoria" for t in fs.compute_terms(untyped))   # no kind: read by its words


def test_possession_with_its_end_runs_to_its_end():
    f = facts()
    t = next(t for t in fs.compute_terms(f) if t["id"] == "usucapione_ordinaria")
    assert t["status"] == "DECORSO" and t["to"] == {"what": "fine del possesso", "date": "2014"}
    assert fs.holds({"possession": {"min_years": 20, "how_not": ["violento", "clandestino"]}}, f, [])
    item = f["possession"][0]
    assert item["use"] == "passaggio" and item["until_iso"] == "2014"
    short = json.loads(json.dumps(REPLY))
    short["possession"][0].update(since="2014", until="2015")
    f = fs.validate_facts(short, CASE + " Dal 2014 al 2015.")
    assert next(t for t in fs.compute_terms(f) if t["id"] == "usucapione_ordinaria")["status"] == "NON DECORSO"


def test_the_prompt_lists_the_roles_and_the_kinds():
    assert "chiusura/ostruzione accesso" in fs._EXTRACT_SYSTEM and "«privato»" in fs._EXTRACT_SYSTEM
    assert "before_separation" in fs._EXTRACT_SYSTEM and "together_with" in fs._EXTRACT_SYSTEM


# --- explained ---------------------------------------------------------------------------------------------

def test_every_term_and_rule_is_explained_condition_by_condition():
    f = facts()
    terms = fs.explain_terms(f)
    pos = next(t for t in terms if t["id"] == "possessoria")
    assert "eventi del tipo spoglio / turbativa / chiusura/ostruzione accesso" in pos["why"][0]
    assert "stato: DECORSO" in pos["why"]
    assert next(t for t in terms if t["id"] == "querela")["entry"] is None
    rules = {r["id"]: r for r in fs.explain_rules(f, fs.compute_terms(f), areas=["diritti_reali"])}
    dest = rules["destinazione_padre_famiglia"]
    assert dest["fired"] and all(c["matched"] for c in dest["conditions"])
    assert "particella 10 di Rossi fino al 1960 insieme a particella 11" in dest["conditions"][0]["fact"]
    assert rules["legittima_difesa"]["skipped"].startswith("famiglia penale")
    assert not rules["vizi_vendita"]["fired"] and not rules["vizi_vendita"]["conditions"][0]["matched"]


def test_the_probe_prints_the_reply_the_full_facts_terms_and_rules(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(fs, "_call_chat", lambda *a, **k: json.dumps(REPLY))
    probe = _script("fact_probe")
    out_file = tmp_path / "probe.json"
    probe.main(["--text", CASE, "--areas", "diritti_reali", "--env", str(tmp_path / "none.env"),
                "--save", str(out_file)])
    out = capsys.readouterr().out
    assert "## Typed facts (full)" in out and '"before_separation": "sì"' in out
    assert "- possessoria: DECORSO" in out and "- destinazione_padre_famiglia: FIRED" in out
    assert "MATCHED   un'opera visibile" in out and "unmatched" in out
    assert "one extraction call" in out
    saved = json.loads(out_file.read_text(encoding="utf-8"))
    assert saved["facts"]["events"][0]["kind"] == "chiusura/ostruzione accesso" and saved["raw_replies"]


# --- the facts whole ---------------------------------------------------------------------------------------

def test_the_vlog_prints_the_facts_whole(monkeypatch, capsys):
    from src.rag import verbose_logger
    monkeypatch.setattr(verbose_logger, "VERBOSE", True)
    long_facts = {"events": [{"what": "x" * 300}]}
    verbose_logger.vlog("case_analysis_fact_signals", {"facts": long_facts, "terms": ["y" * 300]})
    line = capsys.readouterr().out
    assert "x" * 300 in line and "y" * 300 not in line


def test_the_dry_run_record_carries_the_facts():
    dryrun = _script("dryrun_strategy")
    f = facts()
    events = [("case_analysis_fact_signals", {"facts": f, "terms": [{"id": "possessoria"}], "fired": []}, None)]
    record = dryrun.saved_run("code", "req", "ok", 1.0, [], events, {"draft": "d"})
    assert record["facts"] == f and record["fact_terms"] == [{"id": "possessoria"}] and record["fact_rules_fired"] == []


# --- 27 Oct, second probe: work dates, tolerance, landlocked plots, the client ------------------------------

def test_a_work_said_to_exist_already_at_a_date_takes_that_date():
    case = ("Il mulino e il prato erano di Ferri. La mulattiera che sale al prato esisteva già all'inizio del 1800. "
            "Nel 1990 Ferri vende il prato a Gori.")
    reply = {"works_or_signs": [{"what": "mulattiera", "visible": "sì", "existing_since": ""}]}
    work = fs.validate_facts(reply, case)["works_or_signs"][0]
    assert work["existing_since"] == "all'inizio del 1800" and work["existing_since_iso"] == "1800"
    assert work["existing_since_how"] == "il testo: esisteva già"
    other = fs.validate_facts({"works_or_signs": [{"what": "pozzo", "visible": "sì", "existing_since": ""}]}, case)
    assert other["works_or_signs"][0]["existing_since"] == ""                # not the mulattiera's date


def test_a_work_there_before_the_division_dates_from_the_end_of_the_common_ownership(monkeypatch):
    monkeypatch.setenv(fs.TODAY_ENV, "2026-10-27")
    case = "I due orti appartenevano a Sala fino al 1975, quando li divise. Il sentiero c'era già allora."
    reply = {"property_history": [{"asset": "orto nord", "owner": "Sala", "to": "1975", "together_with": "orto sud",
                                   "ended_by": "divisione"}],
             "works_or_signs": [{"what": "sentiero", "visible": "sì", "existing_since": "", "before_separation": "sì"}],
             "possession": [{"who": "Pace", "asset": "sentiero", "since": "1975", "how": "pacifico",
                             "still_ongoing": "sì"}]}
    facts = fs.validate_facts(reply, case + " Pace lo usa dal 1975.")
    work = facts["works_or_signs"][0]
    assert work["existing_since"] == "1975" and work["existing_since_how"].startswith("la fine della proprietà comune")
    fired = [r["id"] for r in fs.fire_rules(facts, fs.compute_terms(facts))]
    assert "usucapione_servitu_apparente" in fired and "destinazione_padre_famiglia" in fired


def test_tolerance_only_when_the_text_says_it():
    reply = {"possession": [{"who": "Lena", "asset": "cortile", "since": "2001", "how": "pacifico, con tolleranza"}]}
    silent = fs.validate_facts(reply, "Lena attraversa il cortile dal 2001.")["possession"][0]
    assert silent["how"] == "pacifico" and silent["how_dropped"] == "con tolleranza"
    for said in ("Lena passa per tolleranza del vicino dal 2001.", "Dal 2001 Lena passa con il permesso di Ugo.",
                 "Ugo la lascia passare per cortesia dal 2001."):
        assert fs.validate_facts(json.loads(json.dumps(reply)), said)["possession"][0]["how"] == "pacifico, con tolleranza"


@pytest.mark.parametrize("sentence", [
    "Il fondo di Ivo è intercluso.",
    "Il terreno di Ivo è senza accesso alla via pubblica.",
    "La casa di Ivo non è raggiungibile se non attraverso il fondo di Ada.",
    "Il podere di Ivo è raggiungibile solo passando dal cortile di Ada.",
])
def test_an_explicit_landlocked_statement_sets_access_and_fires_the_rule(sentence):
    facts = fs.validate_facts({"events": [{"what": "Ada chiude il cortile"}]}, sentence + " Ada chiude il cortile.")
    assert facts["access"][0]["landlocked"] == "sì" and facts["access"][0]["how_reached"] == sentence
    assert "fondo_intercluso" in [r["id"] for r in fs.fire_rules(facts, [])]


def test_a_plot_not_landlocked_or_with_its_own_access_does_not_fire():
    facts = fs.validate_facts({"events": [{"what": "Ada pianta una siepe"}]},
                              "Il fondo di Ivo non è intercluso: ha un accesso sulla provinciale. Ada pianta una siepe.")
    assert facts["access"] == [] and "fondo_intercluso" not in [r["id"] for r in fs.fire_rules(facts, [])]
    own = fs.validate_facts({"access": [{"asset": "fondo di Ivo", "landlocked": "no"}],
                             "events": [{"what": "il fondo di Ivo è intercluso secondo Ada"}]}, "Testo.")
    assert not fs.holds({"landlocked": {}}, own, [])                     # typed: the words do not count


def test_the_client_comes_from_the_request_then_the_text_never_the_lawyer():
    case = ("RICHIESTA DELL'UTENTE: Assumi la difesa di Bianca Riva e indica le azioni.\n\nDOCUMENTO CARICATO:\n"
            "L'avv. Conti scrive per conto di Dario Pino. Bianca Riva e Dario Pino litigano sul confine.")
    reply = {"client_side": "avv. Conti", "parties": [{"name": "Bianca Riva", "role": "vicino"},
                                                      {"name": "Dario Pino", "role": "vicino"}]}
    facts = fs.validate_facts(reply, case)
    assert facts["client_side"] == "Bianca Riva (cliente, vicino)" and facts["client_side_how"] == "la richiesta"
    assert facts["parties"][1]["role"] == "vicino"
    text_only = fs.validate_facts({"client_side": "", "parties": []},
                                  "Il candidato, legale di Sara Neri, rediga parere. Sara Neri compra un'auto.")
    assert text_only["client_side"] == "Sara Neri" and text_only["client_side_how"] == "il testo"
    lawyer = fs.validate_facts({"client_side": "l'avvocato dello studio"}, "Un caso senza cliente indicato.")
    assert lawyer["client_side"] == "" and "avvocato" in lawyer["client_side_how"]
    kept = fs.validate_facts({"client_side": "Ugo, proprietario"}, "Ugo possiede un orto.")
    assert kept["client_side"] == "Ugo, proprietario"


def test_client_from_reads_the_usual_phrasings():
    assert fs.client_from("Assunte le vesti del legale di Caio, rediga parere") == "Caio"
    assert fs.client_from("il difensore della società Alfa srl chiede") == "Alfa srl"
    assert fs.client_from("Agisci nell'interesse di Mara Sole.") == "Mara Sole"
    assert fs.client_from("Una lite tra vicini.") == ""
