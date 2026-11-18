"""v79.9 block B -- modules with stage and known laws, on synthetic sentences and records only:

B1. a module's stage (the certain facts' words, else the events the terms read) puts its stage's articles first --
    appalto in corso: 1662, 1453; consegnata: 1667, 1668, 1669; vendita adds 1439 on a certain false description; each
    module names the special laws that govern it
B2. a module whose laws the graph lacks: one fixed line in §1 and §9; 2118 / 2119 never a claim for a dismissed worker
B3. the consumer qualification needs a professional counterparty as a certain fact; a private seller gates the consumer
    code out
B4. data/corpus_expected.json -- the same laws as the modules'; corpus_gaps.py prints the missing ones
"""

import json
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
    from src.rag import plan_then_write
    monkeypatch.setattr(plan_then_write, "negation_review", lambda items, known, call=None: [])


def _req(i, text, status="provato"):
    return {"id": f"R{i}", "comma": 1, "kind": "requisito", "text": text, "status": status}


def _rec(rubrica, *reqs, side="weapon"):
    return {"rubrica": rubrica, "side": side, "requisites": list(reqs)}


def _posture(family, role, stage="pre_filing"):
    from src.rag import posture
    return {"rito": family, "family": family, "role": role, "stage": stage,
            "entry": posture.entry_for(family, role, stage)}


def _ctx(recs, case, institutions=(), **kw):
    return {"records": recs, "circumstances": {}, "posture": _posture("civile", "attore"), "case": case,
            "assessment": {}, "institutions": list(institutions), **kw}


def _claims(cands):
    return [c["article"] for c in cands if c["kind"] == "claim"]


APPALTO = {"1662 c.c.": _rec("Verifica nel corso di esecuzione dell'opera", _req(1, "l'appaltatore non procede secondo "
                                                                                    "le condizioni stabilite")),
           "1453 c.c.": _rec("Risolubilità del contratto per inadempimento", _req(2, "uno dei contraenti non adempie")),
           "1667 c.c.": _rec("Difformità e vizi dell'opera", _req(3, "l'opera presenta difformità o vizi")),
           "1668 c.c.": _rec("Contenuto della garanzia per difetti dell'opera", _req(4, "le difformità o i vizi"))}


# --- B1 -----------------------------------------------------------------------------------------------------------------------

def test_appalto_in_corso_puts_1662_and_1453_first():
    from src.rag import lawyer_method as M
    case = ("Tizio ha affidato all'impresa Alfa il rifacimento del tetto. I lavori sono iniziati a marzo e dopo sei "
            "mesi risultano incompleti.")
    ctx = _ctx(APPALTO, case)
    got = _claims(M.thesis_candidates(ctx))
    assert ctx["module_stages"] == {"appalto": "in_corso"}
    assert got[:2] == ["1662 c.c.", "1453 c.c."] and set(got[2:]) == {"1667 c.c.", "1668 c.c."}


def test_appalto_consegnata_puts_1667_and_1668_first():
    from src.rag import lawyer_method as M
    case = ("Tizio ha affidato all'impresa Alfa il rifacimento del tetto. L'opera è stata consegnata a maggio; a "
            "giugno compaiono infiltrazioni.")
    ctx = _ctx(APPALTO, case)
    got = _claims(M.thesis_candidates(ctx))
    assert ctx["module_stages"] == {"appalto": "consegnata"}
    assert got[:2] == ["1667 c.c.", "1668 c.c."] and "1453 c.c." not in got[:3]
    assert got.index("1667 c.c.") < got.index("1662 c.c.")


def test_the_stage_from_the_events_the_terms_read():
    from src.rag import lawyer_method as M
    mod = next(m for m in M.institution_modules()["modules"] if m["id"] == "appalto")
    ctx = {"case": "Tizio ha affidato all'impresa Alfa il rifacimento del tetto.", "events": ["fine lavori"]}
    assert M.module_stage(mod, ctx) == "consegnata"
    assert M.module_stage(mod, {**ctx, "events": []}) == ""


def test_vendita_adds_1439_only_on_a_certain_false_description():
    from src.rag import lawyer_method as M
    recs = {"1490 c.c.": _rec("Garanzia per i vizi della cosa venduta", _req(1, "la cosa venduta ha vizi")),
            "1439 c.c.": _rec("Dolo", _req(2, "i raggiri usati da uno dei contraenti"))}
    told = "Beta ha venduto a Tizio un'auto usata, descritta nell'annuncio come mai incidentata; in realtà era incidentata."
    assert _claims(M.thesis_candidates(_ctx(recs, told)))[:2] == ["1490 c.c.", "1439 c.c."]
    said = "Beta ha venduto a Tizio un'auto usata. Tizio sostiene che Beta l'aveva descritta come mai incidentata."
    ctx = _ctx(recs, said)
    assert _claims(M.thesis_candidates(ctx)) == ["1490 c.c."]
    assert ctx["claims_refused"]["1439 c.c."] == M.CLAIM_REFUSED["outside"]


def test_every_module_with_special_laws_names_them():
    from src.rag import lawyer_method as M
    mods = {m["id"]: m for m in M.institution_modules()["modules"]}
    assert mods["licenziamento"]["laws"] == ["L. 604/1966", "L. 300/1970", "d.lgs. 23/2015", "d.lgs. 81/2015"]
    assert mods["responsabilita_sanitaria"]["laws"] == ["L. 24/2017"]
    assert {"1218 c.c.", "1228 c.c.", "2043 c.c."} <= set(mods["responsabilita_sanitaria"]["actions"])
    assert mods["consumo"]["laws"] == ["d.lgs. 206/2005"] and mods["consumo"]["requires"] == "professional_counterparty"


# --- B2 -----------------------------------------------------------------------------------------------------------------------

DISMISSAL = "Laura, dipendente della società Alfa da otto anni, è stata licenziata con una lettera senza motivazione."


def _dismissal_ctx():
    recs = {"2119 c.c.": _rec("Recesso per giusta causa", _req(1, "una causa che non consenta la prosecuzione")),
            "2118 c.c.": _rec("Recesso dal contratto a tempo indeterminato", _req(2, "il contratto a tempo indeterminato"))}
    return _ctx(recs, DISMISSAL, ["licenziamento"], client="Laura")


def test_a_dismissal_with_the_laws_absent_gets_the_line_and_no_2119_main_line():
    from src.rag import lawyer_method as M
    ctx = _dismissal_ctx()
    ctx["missing_laws"] = M.missing_laws(ctx, known=["Codice civile"])
    assert ctx["missing_laws"]["leggi"] == ["L. 604/1966", "L. 300/1970", "d.lgs. 23/2015", "d.lgs. 81/2015"]
    ctx["candidates"] = M.thesis_candidates(ctx)
    assert not [c for c in ctx["candidates"] if c.get("article") in ("2119 c.c.", "2118 c.c.")]
    assert "non conferisce un'azione al cliente" in ctx["claims_refused"]["2119 c.c."]
    plan = M.clean_method({"valutazioni": {c["id"]: {"realismo": "basso"} for c in ctx["candidates"]}}, ctx["candidates"])
    out = M.compose([("Norme applicate ai fatti", "Regola.")], ctx, plan)
    by = dict(M.answer_sections(out))
    line = ("La disciplina speciale di licenziamento (L. 604/1966, L. 300/1970, d.lgs. 23/2015, d.lgs. 81/2015) non è "
            "nell'archivio: il parere si limita alle norme del codice")
    assert line in by[M.title("sintesi")] and line in by[M.title("conclusioni")]
    sols, main, _ = M.selection(ctx, plan)
    assert "2119" not in sols[main]["tesi"] and "2118" not in sols[main]["tesi"]


def test_with_the_laws_present_or_unknown_no_line():
    from src.rag import corpus_laws, lawyer_method as M
    ctx = _dismissal_ctx()
    names = ["Legge 15 luglio 1966, n. 604", "Legge 20 maggio 1970, n. 300", "Decreto legislativo 4 marzo 2015, n. 23",
             "Decreto legislativo 15 giugno 2015, n. 81"]
    assert M.missing_laws(ctx, known=names) == {}
    assert M.missing_laws(ctx) == {}                         # the graph's names unknown: nothing said missing
    corpus_laws.set_names(["Codice civile"])
    try:
        assert M.missing_laws(ctx)["module"] == "licenziamento"
    finally:
        corpus_laws.set_names(None)
    assert M.missing_laws_line({}) == ""


def test_the_counter_from_method_inputs(monkeypatch):
    from src.rag import interactive_flow as flow, lawyer_method as M
    state = {"block_counters": {}}
    monkeypatch.setattr(M, "missing_laws", lambda ctx, known=None: {"istituto": "licenziamento", "leggi": ["x"]})
    ctx = {}
    ctx["missing_laws"] = flow._isolated("missing_laws", lambda: M.missing_laws(ctx), {}) or {}
    assert ctx["missing_laws"]["leggi"] == ["x"]
    assert "missing_laws_lines" in flow.case_counters({**state, "block_counters": {"missing_laws_lines": 1}})


# --- B3 -----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("case, professional", [
    ("Tizio ha comprato una lavatrice nel negozio di elettrodomestici Alfa s.r.l.; dopo un mese si è guastata.", True),
    ("Tizio ha comprato una lavatrice usata da un venditore privato tramite un annuncio; dopo un mese si è guastata.",
     False),
    ("Tizio ha comprato una lavatrice usata da Caio; dopo un mese si è guastata.", False),
    ("Tizio sostiene che Alfa sia un'impresa; ha comprato da Alfa una lavatrice.", False),
])
def test_the_professional_counterparty_is_a_certain_fact(case, professional):
    from src.rag import lawyer_method as M
    assert M.professional_counterparty({"case": case}) is professional


def test_a_private_seller_gets_no_consumer_code_candidate():
    from src.rag import lawyer_method as M
    recs = {"135-bis d.lgs. 206/2005": _rec("Diritti del consumatore", _req(1, "il difetto di conformità del bene")),
            "1490 c.c.": _rec("Garanzia per i vizi della cosa venduta", _req(2, "la cosa venduta ha vizi"))}
    case = ("Il consumatore Tizio ha comprato una lavatrice usata da un venditore privato tramite un annuncio; dopo un "
            "mese si è guastata.")
    ctx = _ctx(recs, case)
    cands = M.thesis_candidates(ctx)
    assert "consumo" not in ctx["claim_modules"]
    assert not any("206/2005" in str(c.get("article")) for c in cands) and _claims(cands) == ["1490 c.c."]
    shop = case.replace("da un venditore privato tramite un annuncio", "nel negozio Alfa s.r.l.")
    ctx = _ctx(recs, shop)
    assert "135-bis d.lgs. 206/2005" in _claims(M.thesis_candidates(ctx)) and "consumo" in ctx["claim_modules"]


# --- B4 -----------------------------------------------------------------------------------------------------------------------

def test_corpus_expected_is_the_modules_laws():
    mods = json.loads((ROOT / "data" / "institution_modules.json").read_text(encoding="utf-8"))["modules"]
    exp = json.loads((ROOT / "data" / "corpus_expected.json").read_text(encoding="utf-8"))
    assert exp["modules"] == {m["id"]: m["laws"] for m in mods if m.get("laws")}
    assert all(law in exp["laws"] for laws in exp["modules"].values() for law in laws)
    assert all(law in exp["laws"] for law in exp["defaults"])


@pytest.mark.parametrize("law, name, there", [
    ("L. 604/1966", "Legge 15 luglio 1966, n. 604", True), ("L. 604/1966", "Legge 604/1966", True),
    ("L. 24/2017", "Decreto 24 marzo 2017, n. 50", False), ("L. 24/2017", "Legge 8 marzo 2017, n. 24", True),
    ("d.lgs. 206/2005", "Codice del Consumo", True), ("L. 300/1970", "Legge 20 maggio 1970, n. 3000", False)])
def test_a_law_is_matched_by_number_and_year_or_alias(law, name, there):
    from src.rag import corpus_laws
    assert corpus_laws.present(law, [name]) is there


def test_corpus_gaps_selftest():
    import subprocess
    import sys
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "corpus_gaps.py"), "--selftest"], capture_output=True,
                         text=True, cwd=ROOT)
    assert out.returncode == 0 and "selftest ok" in out.stdout and "Missing, for ingestion" in out.stdout
