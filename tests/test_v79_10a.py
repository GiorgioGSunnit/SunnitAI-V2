"""v79.10 addendum A4-A5 -- the amministrativo family and module, the family gate's safety valve and its alarm, on
synthetic sentences and rows only:

A4. the amministrativo module (L. 241/1990, c.p.a., TU edilizia d.P.R. 380/2001) active on one strong signal or two weak
    ones -- a 2051 custodia case naming «il Comune» activates nothing administrative and never admits the TU edilizia; a
    SCIA / annullamento d'ufficio case does; data/corpus_expected.json holds the module's laws; the special-law table's
    family signal (strong / weak) admits L. 241/1990 and, beside its own signal, the TU edilizia
A5. the override doors (istituto, utente, articolo, nome intero), each logged with its door and signal, a row the family
    rules admit anyway never one; label_admitted honours them; the dropped rows graded in shadow and recorded only (a
    proxy with FAMILY_GATE_SHADOW_GRADE off); the counters; the nightly's alarm section first, «nessuna» when none, a
    baseline without the keys rendering
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for name, value in (("LLM_API_KEY", "k"), ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "k"), ("NEO4J_URI", "bolt://localhost:7687"), ("NEO4J_USER", "n"),
                        ("NEO4J_PASSWORD", "x")):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))


def _row(number, code, doc="", **extra):
    return {"_article": (number, code), "d": {"name": doc or code}, **extra}


CUSTODIA = ("Tizio è inciampato in una buca della strada comunale e si è fratturato il polso; il Comune, custode della "
            "strada, non l'aveva segnalata né transennata.")
SCIA = ("Alfa s.r.l. ha presentato una SCIA per ampliare il capannone; dopo venti mesi il Comune ne ha disposto "
        "l'annullamento d'ufficio.")
CONTRACT = ("Alfa s.r.l. ha venduto a Beta s.r.l. un macchinario industriale per 50.000 euro; Beta non ha pagato il "
            "saldo alla scadenza.")
TU_EDILIZIA = _row("31", "Testo unico in materia edilizia Edizione 2025")      # v79.10: the archive's exact names
L241 = _row("21-nonies", "Legge sul Procedimento Amministrativo 2026")
ADOZIONE = _row("3", "Legge 4 maggio 1983, n. 184")


def _ids(case, institutions=()):
    from src.rag import lawyer_method as M
    return [m["id"] for m in M.active_modules({"case": case, "institutions": list(institutions)})]


# --- A4 -------------------------------------------------------------------------------------------------------------------

def test_a_custodia_case_naming_the_comune_activates_nothing_administrative():
    from src.rag import family_gate, special_laws
    assert "amministrativo" not in _ids(CUSTODIA) and "custodia" in _ids(CUSTODIA)
    ctx = family_gate.context(CUSTODIA)
    assert "amministrativo" not in ctx["modules"]
    found = special_laws.signals(CUSTODIA)
    assert not found["testo_unico_edilizia"] and not found["legge_241_1990"]
    kept, dropped, _ = family_gate.gate([TU_EDILIZIA, L241], ctx, set())
    assert kept == [] and {x["family"] for x in dropped} == {"edilizia", "amministrativo"}


@pytest.mark.parametrize("case", [
    "Il Comune ha notificato a Tizio un ricorso per decreto ingiuntivo per i canoni non pagati.",
    "Il Comune chiede la revoca del provvedimento del giudice che ha sospeso l'esecuzione.",
    "Caio è caduto sulle scale della scuola gestita dalla pubblica amministrazione; chiede i danni.",
])
def test_weak_signals_alone_or_civil_uses_do_not_activate(case):
    from src.rag import special_laws
    assert "amministrativo" not in _ids(case)
    assert not special_laws.signals(case)["legge_241_1990"]


@pytest.mark.parametrize("case", [
    SCIA,
    "Il Comune ha respinto la domanda di permesso di costruire senza motivare.",
    "Tizio vuole impugnare al TAR il diniego dell'autorizzazione.",
    "Sull'istanza di Caio si è formato il silenzio-assenso, ma il Comune ora lo contesta.",
    "Il Comune ha emesso un provvedimento di chiusura del locale di Tizio.",          # two weak: Comune + provvedimento
])
def test_a_strong_signal_or_two_weak_ones_activate(case):
    from src.rag import special_laws
    assert "amministrativo" in _ids(case)
    assert special_laws.signals(case)["legge_241_1990"]


def test_an_institution_activates_the_module():
    # v79.14.1 (item 3, review): the amministrativo module needs the administrative act in the case's own words -- an
    # institution naming one (a model's words) no longer opens it alone; the case stating the act does
    assert "amministrativo" not in _ids(CONTRACT, ["annullamento d'ufficio del permesso"])
    assert "amministrativo" in _ids(CONTRACT + " Il Comune ha poi annullato d'ufficio il permesso di costruire.",
                                    ["annullamento d'ufficio del permesso"])


def test_the_scia_case_admits_l241_and_the_tu_edilizia():
    from src.rag import family_gate
    ctx = family_gate.context(SCIA)
    assert "amministrativo" in ctx["modules"]
    assert ctx["module_laws"] == ["L. 241/1990", "d.P.R. 380/2001"]
    kept, dropped, outside = family_gate.gate([TU_EDILIZIA, L241, _row("29", "c.p.a.")], ctx, set())
    assert len(kept) == 3 and not dropped
    assert {x["family"] for x in outside} == {"edilizia", "amministrativo"}


def test_module_hit_defaults_to_any_pattern_and_honours_weak_min():
    from src.rag import lawyer_method as M
    assert M.module_hit({"patterns": [r"\bcomodat"]}, "un comodato") == ["comodat"]
    weak = {"patterns": [], "weak_patterns": [r"\balfa\b", r"\bbeta\b"], "weak_min": 2}
    assert M.module_hit(weak, "alfa") == [] and M.module_hit(weak, "alfa e beta") == ["alfa", "beta"]
    assert M.module_hit({**weak, "weak_min": 1}, "alfa") == ["alfa"]


def test_the_module_its_actions_and_the_corpus_expected_laws():
    from src.rag import corpus_laws, lawyer_method as M
    mod = next(m for m in M.institution_modules()["modules"] if m["id"] == "amministrativo")
    assert mod["draft"] is True and mod["laws"] == ["L. 241/1990", "d.P.R. 380/2001"] and len(mod["actions"]) == 16
    assert mod["actions"][:5] == ["7 l. 241/1990", "10-bis l. 241/1990", "21-quinquies l. 241/1990",
                                  "21-octies l. 241/1990", "21-nonies l. 241/1990"]
    exp = json.loads((ROOT / "data" / "corpus_expected.json").read_text(encoding="utf-8"))
    assert exp["modules"]["amministrativo"] == mod["laws"]
    assert "testo unico dell'edilizia" in exp["laws"]["d.P.R. 380/2001"]["aliases"]
    # v79.10 (archive names): the exact document names only
    assert corpus_laws.present("L. 241/1990", ["Legge sul Procedimento Amministrativo 2026"])
    assert not corpus_laws.present("L. 241/1990", ["Legge 7 agosto 1990, n. 241"])
    assert corpus_laws.present("d.P.R. 380/2001", ["Testo unico in materia edilizia Edizione 2025"])
    assert corpus_laws.gaps(["Legge sul Procedimento Amministrativo 2026"])["modules"]["amministrativo"] == [
        "d.P.R. 380/2001"]
    # the labels the pipeline builds for these documents match the module's specs; a c.p.a. article by its sigla
    assert M.claim_admission("21-nonies Legge sul Procedimento Amministrativo 2026", [mod])[1] == "module"
    assert M.claim_admission("31 Testo unico in materia edilizia Edizione 2025", [mod])[1] == "module"
    assert M.claim_admission("29 Codice del processo amministrativo 2025", [mod])[1] == "module"
    assert M.claim_admission("29 c.p.a.", [mod])[1] == "module"


def test_the_special_law_table_family():
    from src.rag import family_gate, special_laws
    assert family_gate.classify("Legge sul Procedimento Amministrativo 2026") == ("amministrativo", "legge_241_1990")
    assert family_gate.classify("Testo unico in materia edilizia Edizione 2025") == (
        "edilizia", "testo_unico_edilizia")
    assert family_gate.classify("Legge 7 agosto 1990, n. 241") == (family_gate.OTHER, "")     # v79.10: no number
    assert special_laws.family_signal("amministrativo", "il Comune") == []
    assert special_laws.family_signal("amministrativo", "il Comune e il suo provvedimento") == [
        "il Comune", "Comune e il suo provvedimento"]               # (review) the act attributed to the ente
    assert special_laws.family_signal("amministrativo", "l'annullamento d'ufficio") == ["annullamento d'ufficio"]
    # the TU edilizia keeps its own signal (private building works) and takes the family's too
    assert special_laws.signals("lavori di ristrutturazione dell'appartamento")["testo_unico_edilizia"]
    assert special_laws.signals("il ricorso al TAR")["testo_unico_edilizia"]
    ok, family, by = family_gate.admitted("Legge sul Procedimento Amministrativo 2026", "",
                                          {**family_gate.context(CONTRACT), "signals": special_laws.signals(SCIA),
                                           "words": SCIA})
    assert ok and family == "amministrativo" and by.startswith("segnale «")


# --- A5: the override doors -----------------------------------------------------------------------------------------------

def test_door_institution_by_number_and_by_alias():
    from src.rag import family_gate
    ctx = family_gate.context(CONTRACT, "", [{"name": "adozione in casi particolari", "rubricated": "",
                                              "claim": "l'adozione ex l. 184/1983"}])
    assert ctx["institutions"] == ["adozione in casi particolari", "l'adozione ex l. 184/1983"]
    overrides = []
    kept, dropped, outside = family_gate.gate([ADOZIONE], ctx, set(), "retrieval", overrides)
    assert kept == [ADOZIONE] and not dropped
    assert overrides == [{"doc": "Legge 4 maggio 1983, n. 184", "article": "3 Legge 4 maggio 1983, n. 184",
                          "door": "istituto", "signal": "«184/1983» in «l'adozione ex l. 184/1983»",
                          "source": "retrieval"}]
    assert outside[0]["by"].startswith("override: ")
    fornero = _row("1", "Riforma del Lavoro Fornero 2026")         # v79.10: a corpus_expected law, by its archive name
    ctx = family_gate.context(CONTRACT, "", ["il rito della riforma Fornero"])
    overrides = []
    assert family_gate.gate([fornero], ctx, set(), "retrieval", overrides)[0] == [fornero]
    assert overrides[0]["door"] == "istituto" and "riforma fornero" in overrides[0]["signal"]
    assert family_gate.gate([fornero], family_gate.context(CONTRACT), set())[0] == []
    statuto = _row("18", "Legge 20 maggio 1970, n. 300")           # no archive name of the L. 300/1970: never its alias
    ctx = family_gate.context(CONTRACT, "", ["tutela reale dello statuto dei lavoratori"])
    assert family_gate.gate([statuto], ctx, set())[0] == []


def test_door_user_logged_only_when_the_rules_would_drop():
    from src.rag import family_gate
    overrides = []
    case = CONTRACT + " Si applica la legge 184/1983?"
    kept, _, outside = family_gate.gate([ADOZIONE], family_gate.context(case), set(), "retrieval", overrides)
    assert kept and outside[0]["by"] == "nominata dall'utente"
    assert overrides == [{"doc": "Legge 4 maggio 1983, n. 184", "article": "3 Legge 4 maggio 1983, n. 184",
                          "door": "utente", "signal": "184/1983", "source": "retrieval"}]
    dismissal = "Laura, dipendente della società Alfa, è stata licenziata; invoca la legge 92/2012."
    overrides = []
    kept, _, outside = family_gate.gate([_row("1", "Riforma del Lavoro Fornero 2026")], family_gate.context(dismissal),
                                        set(), "retrieval", overrides)
    assert kept and outside[0]["by"] == "nominata dall'utente" and overrides == []      # admitted by the module anyway


def test_door_article_and_door_whole_name():
    from src.rag import family_gate
    overrides = []
    kept, _, _ = family_gate.gate([ADOZIONE], family_gate.context(CONTRACT), {("3", "Legge 4 maggio 1983, n. 184")},
                                  "retrieval", overrides)
    assert kept and overrides[0]["door"] == "articolo" and overrides[0]["signal"] == "«3 Legge 4 maggio 1983, n. 184»"
    whole = {**ADOZIONE, "_first_read": "adozione in casi particolari",
             "s": {"titolo_articolo": "Adozione in casi particolari"}}
    keyword = {**ADOZIONE, "_first_read": "adozione in casi particolari", "s": {"titolo_articolo": "Affidamento"}}
    overrides = []
    kept, dropped, outside = family_gate.gate([whole, keyword], family_gate.context(CONTRACT), set(), "retrieval",
                                              overrides)
    assert kept == [whole] and len(dropped) == 1
    assert overrides == [{"doc": "Legge 4 maggio 1983, n. 184", "article": "3 Legge 4 maggio 1983, n. 184",
                          "door": "nome intero", "signal": "«adozione in casi particolari»", "source": "retrieval"}]
    assert outside[0]["by"] == "override: «adozione in casi particolari»"


def test_a_row_admitted_anyway_is_no_override():
    from src.rag import family_gate
    overrides = []
    rows = [_row("1453", "c.c.", _first_read="risoluzione", s={"titolo_articolo": "Risoluzione"}),
            _row("5", "D.lgs. 28/2010", "Mediazione civile (Decreto Legislativo n. 28/2010) Edizione 2025")]   # A4
    ctx = family_gate.context(CONTRACT, "", ["risoluzione per inadempimento (d.lgs. 28/2010)"])
    kept, _, _ = family_gate.gate(rows, ctx, {("1453", "c.c.")}, "retrieval", overrides)
    assert kept == rows and overrides == []


def test_label_admitted_honours_the_institution_and_user_doors():
    from src.rag import family_gate, lawyer_method as M
    plain = family_gate.context(CONTRACT)
    assert not family_gate.label_admitted("3 l. 184/1983", plain)[0]
    fam = family_gate.context(CONTRACT, "", ["adozione ex l. 184/1983"])
    ok, _, by = family_gate.label_admitted("3 l. 184/1983", fam)
    assert ok and by == "override: «184/1983» in «adozione ex l. 184/1983»"
    assert M.family_gate_ok("3 l. 184/1983", fam)
    ctx = {"case": CONTRACT, "institutions": ["adozione ex l. 184/1983"]}
    assert M.family_gate_ok("3 l. 184/1983", M.family_context(ctx))         # the candidates' context: same door
    assert family_gate.label_admitted("3 l. 184/1983", family_gate.context(CONTRACT + " Vale la l. 184/1983?"))[0]


# --- A5: the shadow grading and the alarm ---------------------------------------------------------------------------------

def _dropped(n, doc="Legge 4 maggio 1983, n. 184"):
    return [_row(str(i), doc) for i in range(1, n + 1)]


def test_the_shadow_grading_is_recorded_only(monkeypatch):
    from src.rag import defensive_generation as dg, interactive_flow as flow
    monkeypatch.setenv("FAMILY_GATE_SHADOW_GRADE", "true")
    seen = {}

    def grader(text, request, rows, analysis=None):
        seen["n"] = len(rows)
        seen["limits"] = dict(dg._GRADING_LIMITS.get() or {})
        return [{**r, "_grade": 3 if int(r["_article"][0]) <= 6 else 1} for r in rows]
    monkeypatch.setattr(dg, "_filter_applicable_articles", grader)
    rows = _dropped(45) + _dropped(1)                          # 45 distinct, one repeated
    before = json.dumps(rows, sort_keys=True)
    got = flow.family_shadow_grade("caso", "", rows, {})
    assert json.dumps(rows, sort_keys=True) == before        # no row of the pipeline touched
    assert seen["n"] == flow.FAMILY_SHADOW_MAX == 40
    assert seen["limits"]["graded"] and seen["limits"]["grade_kept"] and len(seen["limits"]["prefilter_keep"]) == 40
    doc = got["Legge 4 maggio 1983, n. 184"]
    assert doc["graded_relevant"] == 6 and doc["articles"][:2] == ["1 Legge 4 maggio 1983, n. 184",
                                                                   "2 Legge 4 maggio 1983, n. 184"]
    from src.rag import family_gate
    alarm = family_gate.alarm(got)
    assert alarm and alarm[0]["n"] == 6 and alarm[0]["proxy"] is False


def test_with_the_flag_off_a_proxy_and_no_grading(monkeypatch):
    from src.rag import defensive_generation as dg, family_gate, interactive_flow as flow

    def grader(*a, **k):
        raise AssertionError("never graded with the flag off")
    monkeypatch.setattr(dg, "_filter_applicable_articles", grader)
    got = flow.family_shadow_grade("caso", "", _dropped(5) + _dropped(2, "Circolare n. 285 della Banca d'Italia"), {})
    assert got["Legge 4 maggio 1983, n. 184"] == {"dropped_articles": 5, "proxy": True,
                                                  "articles": [f"{i} Legge 4 maggio 1983, n. 184" for i in range(1, 6)]}
    alarm = family_gate.alarm(got)
    assert [a["doc"] for a in alarm] == ["Legge 4 maggio 1983, n. 184"] and alarm[0]["proxy"] is True
    assert flow.family_shadow_grade("caso", "", [], {}) == {}


def test_the_grader_failing_falls_back_to_the_proxy(monkeypatch):
    from src.rag import defensive_generation as dg, interactive_flow as flow
    monkeypatch.setenv("FAMILY_GATE_SHADOW_GRADE", "true")

    def grader(*a, **k):
        raise RuntimeError("model down")
    monkeypatch.setattr(dg, "_filter_applicable_articles", grader)
    got = flow.family_shadow_grade("caso", "", _dropped(2), {})
    assert got["Legge 4 maggio 1983, n. 184"]["proxy"] is True


def test_the_flag_is_declared_for_the_suite():
    import conftest
    assert "FAMILY_GATE_SHADOW_GRADE" in conftest.FLAGS


def test_the_counters():
    from src.rag import interactive_flow as flow
    graded = {"Legge 4 maggio 1983, n. 184": {"graded_relevant": 5, "articles": [f"{i} l. 184/1983" for i in range(5)]},
              "Circolare n. 285": {"graded_relevant": 1, "articles": ["1 Circolare"]}}
    state = {"family_gate": {"dropped": [], "admitted": [], "dropped_graded": graded,
                             "overrides": [{"doc": "x", "article": "1 x", "door": "utente", "signal": "1/2000"}]}}
    c = flow.case_counters(state)
    assert c["family_overrides"] == 1 and c["family_alarm"] == 1
    c = flow.case_counters({"family_gate": {"dropped": [], "admitted": []}})
    assert c["family_overrides"] == 0 and c["family_alarm"] == 0


# --- A5: the nightly -------------------------------------------------------------------------------------------------------

def _nightly():
    spec = importlib.util.spec_from_file_location("nightly_sim", ROOT / "scripts" / "nightly_sim.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ALARMED = {"id": "c1", "checks": [{"check": "version_footer", "ok": True, "why": ""}],
           "family_gate": {"dropped": [], "admitted": [],
                           "overrides": [{"doc": "Legge 4 maggio 1983, n. 184", "article": "3 Legge 4 maggio 1983, n. 184",
                                          "door": "istituto", "signal": "«184/1983» in «adozione»"}],
                           "dropped_graded": {"Decreto legislativo 1 settembre 1993, n. 385": {
                               "graded_relevant": 5, "articles": ["117 D.lgs. 385/1993", "118 D.lgs. 385/1993",
                                                                  "119 D.lgs. 385/1993", "120 D.lgs. 385/1993",
                                                                  "125-bis D.lgs. 385/1993"]}}}}
QUIET = {"id": "c2", "checks": [{"check": "version_footer", "ok": True, "why": ""}],
         "family_gate": {"dropped": [], "admitted": [], "dropped_graded": {"Circolare": {"graded_relevant": 4,
                                                                                         "articles": []}}}}


def test_the_alarm_section_comes_first_and_the_header_says_it():
    nightly = _nightly()
    text = nightly.report([ALARMED, QUIET], None, [], "79.10.0", [], None, "best-of-3")
    lines = text.splitlines()
    assert lines[0].endswith("— ALLARME presente")
    title = lines.index(f"## {nightly.ALARM_TITLE}")
    assert title < next(i for i, x in enumerate(lines) if x.startswith("| cell |"))
    assert not any(x.startswith("## ") for x in lines[:title])
    assert ("- c1: Decreto legislativo 1 settembre 1993, n. 385 — 5 articoli rilevanti (grado ≥ 2): 117 D.lgs. "
            "385/1993, 118 D.lgs. 385/1993, 119 D.lgs. 385/1993, 120 D.lgs. 385/1993, 125-bis D.lgs. 385/1993") in lines
    assert "- c2: nessuna" in lines
    assert ("c1: 3 Legge 4 maggio 1983, n. 184 — Legge 4 maggio 1983, n. 184 (porta istituto, segnale «184/1983» in "
            "«adozione»)") in text


def test_none_says_nessuna_and_a_proxy_is_labelled():
    nightly = _nightly()
    text = nightly.report([QUIET], None, [], "79.10.0", [], None, "best-of-3")
    assert "ALLARME presente" not in text.splitlines()[0] and "- c2: nessuna" in text
    proxy = {**QUIET, "family_gate": {"dropped_graded": {"Circolare": {"dropped_articles": 7, "proxy": True,
                                                                       "articles": ["1 Circolare"]}}}}
    text = nightly.report([proxy], None, [], "79.10.0", [], None, "best-of-3")
    assert text.splitlines()[0].endswith("ALLARME presente (stima, non graduati)")
    assert "- c2: Circolare — 7 articoli scartati, stima (non graduati): 1 Circolare" in text
    assert "- c3: nessun dato" in "\n".join(nightly.family_alarm_lines([{"id": "c3", "checks": []}]))


def test_a_baseline_without_the_keys_renders():
    nightly = _nightly()
    base = {"version": "79.9.3", "cells": [{"id": "c1", "checks": [{"check": "version_footer", "ok": True, "why": ""}],
                                            "counters": {"graded_articles": 3}}]}
    now = {**ALARMED, "counters": {"graded_articles": 3, "family_alarm": 1, "family_overrides": 1}}
    text = nightly.report([now], None, [], "79.10.0", [], base, "best-of-3")
    assert "(baseline 79.9.3)" in text and nightly.ALARM_TITLE in text
    assert nightly.report([{"id": "c1", "checks": []}], None, [], "79.10.0", [], base, "best-of-3")


def test_main_sets_the_shadow_grading_unless_told_not_to(monkeypatch, tmp_path):
    nightly = _nightly()
    monkeypatch.setattr(nightly, "cells", lambda: [])
    monkeypatch.setattr(nightly, "stage_check", lambda: None)
    monkeypatch.setenv("FAMILY_GATE_SHADOW_GRADE", "false")
    nightly.main(["--env", "/nonexistent", "--no-stages", "--json-out", str(tmp_path / "n.json")])
    assert os.environ["FAMILY_GATE_SHADOW_GRADE"] == "true"
    nightly.main(["--env", "/nonexistent", "--no-stages", "--no-family-shadow-grade"])
    assert os.environ["FAMILY_GATE_SHADOW_GRADE"] == "false"


# --- review: the weak pair and «autotutela» tightened; the doors reach the table's laws and the candidates; the shadow
# grading off the request's meter ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("case", [
    "Tizio è inciampato in una buca su una strada del Comune di Roma. Il Comune ha opposto un diniego alla richiesta di "
    "risarcimento.",
    "Tizio, residente a Palermo (PA), ha ricevuto un provvedimento di ingiunzione.",
    "La società con sede nel Comune di Bergamo ha adottato un provvedimento disciplinare.",
    "Il giudice ha emesso il provvedimento; il Comune è parte del giudizio.",
    "L'assicurazione ha opposto un diniego di indennizzo per la caduta sulla strada del Comune.",
    "Il locatore, nel Comune di Milano, ha opposto il diniego di rinnovo alla prima scadenza.",
    "Il condominio nel Comune di Napoli ha espresso un diniego.",
    "Il Comune è custode del marciapiede. Il giudice ha emesso il provvedimento.",
    "Il dipendente della PA ha subito un provvedimento disciplinare.",
    "Il Comune ha chiesto al tribunale un provvedimento d'urgenza.",
    "La società fornitrice ha sospeso le consegne in autotutela, opponendo al cliente l'eccezione di inadempimento.",
    "Il conduttore ha sospeso in autotutela il pagamento dei canoni per i vizi dell'immobile locato.",
])
def test_civil_wording_activates_nothing_administrative(case):
    from src.rag import special_laws
    assert "amministrativo" not in _ids(case)
    found = special_laws.signals(case)
    assert not found["legge_241_1990"] and not found["testo_unico_edilizia"]


def test_an_institution_naming_the_civil_autotutela_activates_nothing():
    assert "amministrativo" not in _ids(CONTRACT, ["eccezione di inadempimento come forma di autotutela contrattuale"])


@pytest.mark.parametrize("case", [
    "Il Comune ha annullato in autotutela il permesso rilasciato a Tizio.",
    "La PA ha adottato un provvedimento di revoca della concessione.",
    "Il Comune ha opposto il diniego del permesso.",
    "Il provvedimento del Comune è illegittimo, dice l'ufficio tecnico.",
    "Il Comune di Roma ha emesso un'ordinanza sindacale contro Tizio.",
])
def test_the_administrative_forms_still_activate(case):
    from src.rag import special_laws
    assert "amministrativo" in _ids(case)
    assert special_laws.signals(case)["legge_241_1990"]


def test_a_2051_claim_survives_the_comune_s_diniego():
    from src.rag import lawyer_method as M, posture
    attore = {"rito": "civile", "family": "civile", "role": "attore", "stage": "pre_filing",
              "entry": posture.entry_for("civile", "attore", "pre_filing")}
    req = lambda i, text: {"id": f"R{i}", "comma": 1, "kind": "requisito", "text": text, "status": "provato"}
    recs = {"2051 c.c.": {"rubrica": "Danno cagionato da cosa in custodia", "side": "weapon",
                          "requisites": [req(1, "la cosa in custodia"), req(2, "il danno cagionato dalla cosa")]}}
    case = ("Tizio è inciampato in una buca su una strada del Comune di Roma e si è fratturato il polso. Il Comune ha "
            "opposto un diniego alla richiesta di risarcimento.")
    ctx = {"records": recs, "circumstances": {}, "posture": attore, "case": case, "assessment": {}, "institutions": []}
    cands = M.thesis_candidates(ctx)
    assert "amministrativo" not in ctx["claim_modules"]
    assert ("claim", "2051 c.c.") in [(c["kind"], c["article"]) for c in cands]
    assert "2051 c.c." not in (ctx.get("claims_refused") or {})


ACCESSO = _row("22", "Legge sul Procedimento Amministrativo 2026")


def test_the_doors_reach_a_table_law_the_special_law_gate_would_drop():
    from src.rag import family_gate, special_laws
    words = "Il Comune non ha risposto alla richiesta di accesso agli atti."      # one weak signal: no family signal
    assert not special_laws.signals(words)["legge_241_1990"]
    ctx = family_gate.context(words, words, [{"name": "Accesso ai documenti amministrativi (l. 241/1990)",
                                              "rubricated": "", "claim": ""}])
    assert special_laws.gate([ACCESSO], words, set())[0] == []                  # v79.9: dropped before the family gate
    overrides = []
    keep = lambda r: family_gate.rescued(r, ctx, set(), "special_law_gate", overrides)
    kept, dropped = special_laws.gate([ACCESSO], words, set(), keep)
    assert kept == [ACCESSO] and dropped == []
    assert overrides and overrides[0]["door"] == "istituto" and overrides[0]["source"] == "special_law_gate"
    assert "241/1990" in overrides[0]["signal"]
    # no door: the special-law gate's rule stands (and a row only the family rules would admit is never rescued)
    plain = family_gate.context(words, words)
    gone = []
    assert special_laws.gate([ACCESSO], words, set(), lambda r: family_gate.rescued(r, plain, set(), "x", gone)) == (
        [], ["22 Legge sul Procedimento Amministrativo 2026 (L. 241/1990 (procedimento amministrativo))"]) and gone == []


def test_the_retrieval_wires_the_doors_into_the_special_law_gate():
    import inspect
    from src.rag import interactive_flow as flow
    src = inspect.getsource(flow._retrieve)
    assert src.index("family_gate.context(") < src.index("special_laws.gate(article_rows")
    assert "special_door)" in src and "fam_dropped_rows: List[Dict[str, Any]] = list(special_dropped_rows)" in src


def test_a_candidate_on_a_document_the_retrieval_admitted_by_a_door_stays():
    from src.rag import family_gate, lawyer_method as M
    whole = {**ADOZIONE, "_first_read": "adozione in casi particolari",
             "s": {"titolo_articolo": "Adozione in casi particolari"}}
    overrides = []
    kept, _, _ = family_gate.gate([whole], family_gate.context(CONTRACT), set(), "retrieval", overrides)
    assert kept == [whole] and overrides[0]["door"] == "nome intero"
    bare = {"case": CONTRACT, "institutions": ["Affidamento condiviso"]}
    assert not M.family_gate_ok("3 Legge 4 maggio 1983, n. 184", M.family_context(dict(bare)))
    ctx = {**bare, "family_overrides": overrides}                       # prepare: state["family_gate"]["overrides"]
    fam = M.family_context(ctx)
    assert M.family_gate_ok("3 Legge 4 maggio 1983, n. 184", fam) and M.family_gate_ok("44 l. 184/1983", fam)
    assert family_gate.label_admitted("3 l. 184/1983", fam)[2] == "override: «adozione in casi particolari»"
    assert not M.family_gate_ok("117 d.lgs. 385/1993", fam)                 # another document: still out of family
    art = [{"doc": "Legge 4 maggio 1983, n. 184", "article": "3 Legge 4 maggio 1983, n. 184", "door": "articolo",
            "signal": "«3»", "source": "retrieval"}]
    fam = M.family_context({**bare, "family_overrides": art})
    assert M.family_gate_ok("3 l. 184/1983", fam) and not M.family_gate_ok("44 l. 184/1983", fam)


def test_prepare_carries_the_gate_s_overrides():
    import inspect
    from src.rag import lawyer_method as M
    assert '"family_overrides": list((state.get("family_gate") or {}).get("overrides") or [])' in inspect.getsource(
        M.prepare)


def test_the_shadow_grading_counts_on_its_own_meter():
    from src.rag import stage_meter
    request = stage_meter.Meter()
    with request.active():
        request.mark("grading")
        stage_meter.current().count("llm")
        with stage_meter.side() as shadow:
            stage_meter.current().count("llm", 7)
            assert stage_meter._LAST[0] is shadow
        assert stage_meter._LAST[0] is request and stage_meter.current() is request
        stage_meter.current().count("llm")
    request.stop()
    assert request.total("llm") == 2 and shadow.total("llm") == 7


def test_the_counters_keep_the_shadow_cost_apart():
    from src.rag import interactive_flow as flow
    state = {"family_gate": {"dropped": [], "admitted": [], "shadow_grading": {"calls": 6, "s": 12.5}},
             "timings": {"round1_llm": 20}}
    c = flow.case_counters(state, 100.0)
    assert c["llm_calls"] == 20 and c["family_shadow_llm"] == 6 and c["family_shadow_s"] == 12.5
    assert c["total_s"] == 87.5
    c = flow.case_counters({"family_gate": {}}, 100.0)
    assert c["total_s"] == 100.0 and c["family_shadow_llm"] == 0
    import inspect
    src = inspect.getsource(flow._retrieve)
    assert "stage_meter.mark(FAMILY_SHADOW_STAGE)" in src and "with stage_meter.side() as shadow_meter:" in src
