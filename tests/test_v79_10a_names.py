"""v79.10 (A4, archive names) -- the archive's document names carry no law numbers: each corpus_expected law maps to the
exact names of its archive documents («archive_names»), matched by whitespace-normalised equality only, on synthetic
names and sentences:

  exact match only («Legge 7 agosto 1990, n. 241» is not the L. 241/1990), whitespace normalised, case and substrings
  not; the mappings are the user's; a module action («7 l. 241/1990») matches the label the pipeline builds for an
  article of the archive document («7 Legge sul Procedimento Amministrativo 2026»), a c.p.a. one by its sigla; an
  unmapped law is reported missing; the Fornero reform present with the missing-laws line naming licenziamento's four;
  the user naming «L. 241/1990» still user_named; the special-law table's exact entries (d.lgs. 81/2008, 231/2001);
  corpus_gaps.py's selftest and its exit code
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

L241_DOC = "Legge sul Procedimento Amministrativo 2026"
TU_DOC = "Testo unico in materia edilizia Edizione 2025"
CPA_DOC = "Codice del processo amministrativo 2025"
FORNERO_DOC = "Riforma del Lavoro Fornero 2026"
SCIA = ("Alfa s.r.l. ha presentato una SCIA per ampliare il capannone; dopo venti mesi il Comune ne ha disposto "
        "l'annullamento d'ufficio.")
CONTRACT = ("Alfa s.r.l. ha venduto a Beta s.r.l. un macchinario industriale per 50.000 euro; Beta non ha pagato il "
            "saldo alla scadenza.")
DISMISSAL = "Laura, dipendente della società Alfa da otto anni, è stata licenziata con una lettera senza motivazione."

# the user's mappings (BRIEF 79.10), verbatim
MAPPED = {"L. 241/1990": [L241_DOC], "d.P.R. 380/2001": [TU_DOC],
          "d.lgs. 28/2010": ["Mediazione civile (Decreto Legislativo n. 28/2010) Edizione 2025"],
          "d.P.R. 309/1990": ["Testo Unico sugli Stupefacenti 2026"], "d.lgs. 206/2005": ["Codice del Consumo 2026"],
          "d.lgs. 285/1992": ["Codice della strada 2026"], "L. 92/2012": [FORNERO_DOC],
          "L. 898/1970": ["Legge sul divorzio 2024"], "L. 431/1998": ["Legge sulle locazioni abitative 2023"]}
UNMAPPED = ["L. 604/1966", "L. 300/1970", "d.lgs. 23/2015", "d.lgs. 81/2015", "L. 392/1978", "L. 24/2017"]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for name, value in (("LLM_API_KEY", "k"), ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "k"), ("NEO4J_URI", "bolt://localhost:7687"), ("NEO4J_USER", "n"),
                        ("NEO4J_PASSWORD", "x")):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))
    from src.rag import plan_then_write
    monkeypatch.setattr(plan_then_write, "negation_review", lambda items, known, call=None: [])


def _row(number, doc):
    return {"_article": (number, doc), "d": {"name": doc}}


def _req(i, text, status="provato"):
    return {"id": f"R{i}", "comma": 1, "kind": "requisito", "text": text, "status": status}


# --- the data -------------------------------------------------------------------------------------------------------------

def test_the_mappings_are_the_users_and_every_other_law_is_unmapped():
    exp = json.loads((ROOT / "data" / "corpus_expected.json").read_text(encoding="utf-8"))
    assert {k: v["archive_names"] for k, v in exp["laws"].items() if v["archive_names"]} == MAPPED
    assert sorted(k for k, v in exp["laws"].items() if v["archive_names"] == []) == sorted(UNMAPPED)
    assert all(isinstance(v["archive_names"], list) for v in exp["laws"].values())
    assert exp["codes"] == {"c.p.a.": {"name": "Codice del processo amministrativo", "archive_names": [CPA_DOC]}}
    assert exp["modules"]["licenziamento"] == UNMAPPED[:4] + ["L. 92/2012"]
    table = {x["id"]: x for x in json.loads((ROOT / "data" / "special_laws.json").read_text(encoding="utf-8"))["laws"]}
    assert table["testo_unico_sicurezza_lavoro"]["archive_names"] == ["Testo Unico sulla Sicurezza del Lavoro 2026"]
    assert table["dlgs_231_2001"]["archive_names"] == ["Responsabilità Amministrativa di Società ed Enti 2026"]
    assert all(x["expected"] in exp["laws"] for x in table.values() if "expected" in x)


# --- exact, whitespace-normalised -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("name, there", [
    (L241_DOC, True), ("  Legge sul  Procedimento\nAmministrativo\t2026 ", True),
    ("Legge 7 agosto 1990, n. 241", False), ("L. 241/1990", False), ("Legge sul procedimento amministrativo 2026", False),
    ("Legge sul Procedimento Amministrativo", False), ("Legge sul Procedimento Amministrativo 2026 (testo vigente)", False)])
def test_a_law_is_present_by_its_exact_archive_name_only(name, there):
    from src.rag import corpus_laws
    assert corpus_laws.present("L. 241/1990", [name]) is there
    assert (corpus_laws.law_of_name(name) == "L. 241/1990") is there


def test_a_numbered_name_is_no_corpus_law_anywhere():
    from src.rag import corpus_laws, family_gate, special_laws
    numbered = "Legge 7 agosto 1990, n. 241"
    assert family_gate.classify(numbered) == (family_gate.OTHER, "") and family_gate.expected_law(numbered) == ""
    assert special_laws.law_of(numbered) is None and corpus_laws.law_for(numbered) == ""
    assert family_gate.classify(L241_DOC) == ("amministrativo", "legge_241_1990")
    assert family_gate.classify(FORNERO_DOC) == ("modulo", "L. 92/2012")
    assert family_gate.classify("Mediazione civile (Decreto Legislativo n. 28/2010) Edizione 2025") == (
        "mediazione", "d.lgs. 28/2010")
    assert family_gate.classify("Legge sulle locazioni abitative 2023") == ("modulo", "L. 431/1998")
    # a label's code that is the law's own citation, or the sigla its document is labelled with
    assert corpus_laws.law_for("l. 241/1990") == "L. 241/1990" and corpus_laws.law_for("c.d.s.") == "d.lgs. 285/1992"


def test_the_special_law_tables_exact_entries():
    from src.rag import family_gate, special_laws
    assert family_gate.classify("Testo Unico sulla Sicurezza del Lavoro 2026") == (
        "sicurezza sul lavoro", "testo_unico_sicurezza_lavoro")
    assert family_gate.classify("Responsabilità Amministrativa di Società ed Enti 2026") == (
        "responsabilità degli enti", "dlgs_231_2001")
    assert special_laws.law_of("Decreto legislativo 9 aprile 2008, n. 81") is None
    assert family_gate.classify("Decreto legislativo 9 aprile 2008, n. 81") == (family_gate.OTHER, "")
    # the user's words are still read by the expression (a computed term's reference too)
    assert special_laws.signals("infortunio in cantiere per violazione delle norme di sicurezza sul lavoro")
    assert special_laws.law_named("art. 132 Codice del Consumo")["id"] == "codice_del_consumo"
    assert not special_laws.term_allowed({"ref": "art. 132 Codice del Consumo"}, {"codice_del_consumo": False})
    # a row of the archive's document is gated by its signal (special_laws.gate reads the exact name)
    loan = "Tizio ha prestato un libro a Caio, che non lo restituisce."
    row = _row("74", "Testo Unico sulla Sicurezza del Lavoro 2026")
    assert special_laws.gate([row], loan, set())[0] == [] and special_laws.gate([row], "un infortunio", set())[0] == [row]


def test_law_module_reads_the_tables_expected_law():
    from src.rag import family_gate
    ctx = family_gate.context(SCIA)
    assert family_gate.law_module("testo_unico_edilizia", ctx) and family_gate.law_module("legge_241_1990", ctx)
    assert not family_gate.law_module("testo_unico_sicurezza_lavoro", ctx)
    kept, dropped, _ = family_gate.gate([_row("31", TU_DOC), _row("21-nonies", L241_DOC),
                                         _row("21-nonies", "Legge 7 agosto 1990, n. 241")], ctx, set())
    assert [r["_article"][1] for r in kept] == [TU_DOC, L241_DOC] and dropped[0]["family"] == family_gate.OTHER


# --- module actions <-> the label the pipeline builds for the archive document --------------------------------------------

def test_the_pipelines_labels_for_the_archive_documents():
    from src.rag import interactive_flow as flow, issue_records, topic_step
    assert issue_records._label(("7", L241_DOC)) == f"7 {L241_DOC}"
    assert flow._label(flow._key(("7", L241_DOC))) == f"7 {L241_DOC}" and topic_step._code(L241_DOC) == L241_DOC
    assert issue_records._label(("29", CPA_DOC)) == "29 c.p.a."


@pytest.mark.parametrize("module_id", ["amministrativo", "consumo", "famiglia"])
def test_every_mapped_action_matches_its_archive_documents_label(module_id):
    from src.rag import corpus_laws, issue_records, lawyer_method as M
    mod = next(m for m in M.institution_modules()["modules"] if m["id"] == module_id)
    seen = 0
    for spec in mod["actions"]:
        number, _, code = spec.partition(" ")
        law = corpus_laws.law_for(code)
        names = corpus_laws.archive_names(law) if law else corpus_laws.archive_names(code)
        if not names:                                       # a code the pipeline labels by its sigla (c.c.)
            assert code in ("c.c.", "c.p.c.") and M.same_article(spec, spec)
            continue
        for name in names:
            label = issue_records._label((number, name))
            assert M.same_article(label, spec), (label, spec)
            assert M.claim_admission(label, [mod])[1] == "module"
            seen += 1
    assert seen >= 2


@pytest.mark.parametrize("label, spec, same", [
    (f"7 {L241_DOC}", "7 l. 241/1990", True), ("7 L. 241/1990", "7 l. 241/1990", True),
    (f"10-bis {L241_DOC}", "7 l. 241/1990", False), ("7 Legge 7 agosto 1990, n. 241", "7 l. 241/1990", False),
    (f"7 {TU_DOC}", "7 l. 241/1990", False), ("2 Legge 15 luglio 1966, n. 604", "2 l. 604/1966", False),
    ("2 L. 604/1966", "2 l. 604/1966", True), (f"29 {CPA_DOC}", "29 c.p.a.", True), ("29 c.p.a.", "29 c.p.a.", True),
    ("2043 c.c.", "2043 c.c.", True)])
def test_same_article_by_archive_name(label, spec, same):
    from src.rag import lawyer_method as M
    assert M.same_article(label, spec) is same


def test_a_claim_on_the_archive_documents_article_is_the_modules():
    from src.rag import lawyer_method as M, posture
    attore = {"rito": "civile", "family": "civile", "role": "attore", "stage": "pre_filing",
              "entry": posture.entry_for("civile", "attore", "pre_filing")}
    label = f"21-nonies {L241_DOC}"
    recs = {label: {"rubrica": "Annullamento d'ufficio", "side": "weapon",
                    "requisites": [_req(1, "il provvedimento illegittimo"), _req(2, "entro un termine ragionevole")]}}
    ctx = {"records": recs, "circumstances": {}, "posture": attore, "case": SCIA, "assessment": {}, "institutions": []}
    cands = M.thesis_candidates(ctx)
    assert "amministrativo" in ctx["claim_modules"]
    assert ("claim", label) in [(c["kind"], c["article"]) for c in cands]
    numbered = "21-nonies Legge 7 agosto 1990, n. 241"
    assert M.claim_admission(numbered, M.active_modules(ctx))[1] != "module"


def test_the_consumer_code_by_its_archive_name():
    from src.rag import lawyer_method as M
    assert M.consumer_code("135-bis Codice del Consumo 2026") and M.consumer_code("135-bis d.lgs. 206/2005")
    assert not M.consumer_code("135-bis Codice del consumo") and not M.consumer_code("1490 c.c.")


# --- missing laws -------------------------------------------------------------------------------------------------------

def test_an_unmapped_law_is_reported_missing_whatever_the_graph_holds():
    from src.rag import corpus_laws
    names = ["Legge 15 luglio 1966, n. 604", "Legge 604/1966", "L. 604/1966", L241_DOC]
    assert corpus_laws.missing(["L. 604/1966", "L. 241/1990"], names) == ["L. 604/1966"]
    assert corpus_laws.status("L. 604/1966", names) == corpus_laws.UNMAPPED
    assert corpus_laws.status("d.P.R. 380/2001", names) == corpus_laws.NOT_IN_ARCHIVE
    assert corpus_laws.status("L. 241/1990", names) == corpus_laws.PRESENT
    assert corpus_laws.missing(["L. 604/1966"], None) is None or corpus_laws.names() is None


def test_fornero_present_and_the_line_for_licenziamentos_four():
    from src.rag import lawyer_method as M
    ctx = {"records": {}, "circumstances": {}, "case": DISMISSAL, "assessment": {}, "institutions": ["licenziamento"]}
    gone = M.missing_laws(ctx, known=[FORNERO_DOC, "Codice Civile 2026"])
    assert gone["module"] == "licenziamento" and "L. 92/2012" not in gone["leggi"]
    assert gone["leggi"] == ["L. 604/1966", "L. 300/1970", "d.lgs. 23/2015", "d.lgs. 81/2015"]
    ctx["missing_laws"] = gone
    assert M.missing_laws_line(ctx) == (
        "La disciplina speciale di licenziamento (L. 604/1966, L. 300/1970, d.lgs. 23/2015, d.lgs. 81/2015) non è "
        "nell'archivio: il parere si limita alle norme del codice.")
    assert "L. 92/2012" in M.missing_laws(ctx, known=["Codice Civile 2026"])["leggi"]


# --- the user's words ---------------------------------------------------------------------------------------------------

def test_the_user_naming_l241_is_still_user_named():
    from src.rag import family_gate
    case = CONTRACT + " Il cliente invoca la L. 241/1990 sull'accesso."
    ctx = family_gate.context(case)
    assert family_gate.user_named("amministrativo", "legge_241_1990", "", {**ctx, "_doc": L241_DOC})
    d = family_gate.decide(L241_DOC, L241_DOC, ctx)
    assert d["ok"] and d["by"] == "nominata dall'utente" and d["door"] == family_gate.DOOR_USER
    assert not family_gate.decide("Legge 7 agosto 1990, n. 241", "Legge 7 agosto 1990, n. 241", ctx)["ok"]
    fornero = family_gate.decide(FORNERO_DOC, FORNERO_DOC, family_gate.context(CONTRACT + " Vale la legge Fornero?"))
    assert not fornero["ok"]                                    # an alias is no user naming (the «istituto» door only)
    named = family_gate.decide(FORNERO_DOC, FORNERO_DOC, family_gate.context(CONTRACT + " Vale la l. 92/2012?"))
    assert named["ok"] and named["by"] == "nominata dall'utente"


def test_a_carried_override_by_the_archive_name():
    from src.rag import family_gate
    ctx = {**family_gate.context(CONTRACT),
           "overrides": [{"doc": L241_DOC, "article": f"22 {L241_DOC}", "door": "nome intero", "signal": "«accesso»",
                          "source": "retrieval"}]}
    assert family_gate.label_admitted(f"10-bis {L241_DOC}", ctx)[0]
    assert family_gate.label_admitted("10-bis l. 241/1990", ctx)[0]
    assert not family_gate.label_admitted("10-bis Legge 7 agosto 1990, n. 241", ctx)[0]
    numbered = {**family_gate.context(CONTRACT),
                "overrides": [{"doc": "Legge 7 agosto 1990, n. 241", "article": "22 Legge 7 agosto 1990, n. 241",
                               "door": "nome intero", "signal": "«accesso»", "source": "retrieval"}]}
    assert not family_gate.label_admitted("10-bis l. 241/1990", numbered)[0]


# --- corpus_gaps.py -----------------------------------------------------------------------------------------------------

def test_corpus_gaps_selftest_and_report():
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "corpus_gaps.py"), "--selftest"], capture_output=True,
                         text=True, cwd=ROOT)
    assert out.returncode == 0 and "selftest ok" in out.stdout, out.stderr
    assert "(expected 87)" in out.stdout and "## module licenziamento" in out.stdout
    assert f"| L. 92/2012 | «{FORNERO_DOC}» | present |" in out.stdout
    assert "| L. 604/1966 | — | UNMAPPED |" in out.stdout and "NOT IN ARCHIVE" in out.stdout
    assert "## archive names no law maps" in out.stdout and "- «Codice Civile 2026»" in out.stdout


def test_corpus_gaps_reads_the_names_only_and_exits_1_on_an_unmatched_module_law():
    import importlib.util
    from src.rag import corpus_laws
    spec = importlib.util.spec_from_file_location("corpus_gaps", ROOT / "scripts" / "corpus_gaps.py")
    gaps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gaps)

    class Session:
        def run(self, query):
            assert query == corpus_laws.NAMES_QUERY and "RETURN d.name AS name" in query
            return [{"name": "Codice Civile 2026"}, {"name": " Codice  Civile 2026"}, {"name": FORNERO_DOC},
                    {"name": None}]
    names = corpus_laws.fetch_names(Session())
    assert names == ["Codice Civile 2026", FORNERO_DOC]
    got = gaps.run(names)
    assert gaps.exit_code(got) == 1 and "L. 604/1966" in got["unmatched_modules"]
    assert "L. 92/2012" not in got["unmatched_modules"] and got["names_unmapped"] == ["Codice Civile 2026"]
    assert gaps.exit_code({"unmatched_modules": []}) == 0
    assert "Testo Unico sulla Sicurezza del Lavoro 2026" not in gaps.run(["Testo Unico sulla Sicurezza del Lavoro 2026"])[
        "names_unmapped"]                                        # a table entry's archive name is mapped


# --- v79.10 (A4, review): a document's name is never read as a citation or a sigla -------------------------------------

@pytest.mark.parametrize("name", ["Legge 604/1966", "Legge n. 604/1966", "Decreto legislativo n. 23/2015"])
def test_a_document_named_like_a_citation_is_not_the_law(name):
    from src.rag import corpus_laws, family_gate, lawyer_method as M
    ctx = family_gate.context(DISMISSAL, institutions=["licenziamento"])
    assert ctx["modules"] == ["licenziamento"] and "L. 604/1966" in ctx["module_laws"]
    assert family_gate.classify(name, name) == (family_gate.OTHER, "") and family_gate.classify(name)[1] == ""
    assert not family_gate.decide(name, name, ctx)["ok"]
    kept, dropped, _ = family_gate.gate([_row("2", name)], ctx, set())
    assert kept == [] and dropped[0]["family"] == family_gate.OTHER
    assert not M.same_article(f"2 {name}", "2 l. 604/1966") and corpus_laws.law_for(name) == ""
    assert corpus_laws.status("L. 604/1966", [name]) == corpus_laws.UNMAPPED
    # a module action and the pipeline's own citations are still read («l.», «d.lgs.», «d.P.R.»)
    assert corpus_laws.law_for("l. 604/1966") == "L. 604/1966" and corpus_laws.law_for("d.lgs. 23/2015") == "d.lgs. 23/2015"
    assert family_gate.classify("Legge 241/1990", "Legge 241/1990") == (family_gate.OTHER, "")


def test_a_cached_document_name_is_never_a_citation(monkeypatch):
    from src.rag import corpus_laws, family_gate, lawyer_method as M, special_laws
    monkeypatch.setattr(corpus_laws, "_NAMES", {})
    monkeypatch.setattr(corpus_laws, "_FOR", {})
    assert M.same_article("2 L. 604/1966", "2 l. 604/1966")          # a label's citation (no such document read)
    corpus_laws.set_names(["L. 604/1966", "D.lgs. 81/2008", L241_DOC])
    assert corpus_laws.cited("L. 604/1966") == ("", "") and corpus_laws.law_for("L. 604/1966") == ""
    assert not M.same_article("2 L. 604/1966", "2 l. 604/1966") and family_gate.classify("L. 604/1966")[1] == ""
    assert special_laws.law_of("D.lgs. 81/2008") is None and special_laws.law_of("d.lgs. 81/2008") is not None
    assert M.same_article(f"7 {L241_DOC}", "7 l. 241/1990") and corpus_laws.law_for("l. 241/1990") == "L. 241/1990"


def test_a_rows_document_decides_not_the_code_beside_it():
    from src.rag import family_gate, special_laws
    numbered = "Decreto legislativo 4 marzo 2010, n. 28"
    assert family_gate.classify("D.lgs. 28/2010", numbered) == (family_gate.OTHER, "")
    assert family_gate.classify("D.lgs. 28/2010") == ("mediazione", "d.lgs. 28/2010")         # a bare label's citation
    assert special_laws.law_of_row("d.lgs. 81/2008", "Decreto legislativo 9 aprile 2008, n. 81") is None
    assert special_laws.law_of_row("d.lgs. 81/2008", "Testo Unico sulla Sicurezza del Lavoro 2026")["id"] == (
        "testo_unico_sicurezza_lavoro")
    assert special_laws.law_of("D.lgs. 81/2008", name_only=True) is None


@pytest.mark.parametrize("doc", ["Regolamento di esecuzione del codice della strada", "Codice della strada commentato 2024",
                                 "Nuovo codice della strada annotato"])
def test_the_codice_della_strada_by_its_archive_name_only(doc):
    from src.rag import family_gate, lawyer_method as M, special_laws
    road = "Tizio guidava l'auto e ha investito un pedone sulle strisce."
    assert family_gate.classify(doc, doc) == (family_gate.OTHER, "") and family_gate.classify(doc)[1] == ""
    assert family_gate.classify("c.d.s.", doc) == (family_gate.OTHER, "")
    assert special_laws.law_of_row("c.d.s.", doc) is None and special_laws.law_of_row(doc, doc) is None
    kept, dropped, _ = family_gate.gate([_row("1", doc)], family_gate.context(road), set())
    assert kept == [] and dropped[0]["family"] == family_gate.OTHER
    # the archive's document, its sigla as a bare label's code
    exact = "Codice della strada 2026"
    assert family_gate.classify(exact, exact) == family_gate.classify("c.d.s.", exact) == ("strada", "codice_della_strada")
    assert family_gate.classify("c.d.s.") == ("strada", "codice_della_strada") and M.family_of_label("186 c.d.s.") == "strada"
    assert special_laws.law_of_row("c.d.s.", exact)["id"] == "codice_della_strada"


def test_an_override_on_a_document_named_like_a_citation_carries_nothing():
    from src.rag import family_gate
    ctx = {**family_gate.context(CONTRACT),
           "overrides": [{"doc": "Legge 604/1966", "article": "2 Legge 604/1966", "door": "nome intero",
                          "signal": "«licenziamento»", "source": "retrieval"}]}
    assert not family_gate.label_admitted("3 l. 604/1966", ctx)[0]
    assert family_gate.label_admitted("3 Legge 604/1966", ctx)[0]          # the same document's label: its own name
