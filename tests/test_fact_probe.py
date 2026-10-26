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
