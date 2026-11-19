"""v79.8 block B -- claims that confer an action, on synthetic records only:

B1. a claim candidate only on a norm conferring an action (an institution module's, a remedies.json matter's, general
    obligations law); never a limiting rule, the other party's duty, a definition; every candidate records its rule
B2. the first read's institution first; 1218 / 1453 / 2043 subordinate -- offered only when the module proves nothing;
    nothing outside the institution's matter
B3. a defendant never gets a claim: contests, eccezioni, riconvenzionale, transazione
B4. the counters: candidates no rule admits (0), a main line outside the read's institution
"""

from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for name, value in (("LLM_API_KEY", "k"), ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "k"), ("NEO4J_URI", "bolt://localhost:7687"), ("NEO4J_USER", "n"),
                        ("NEO4J_PASSWORD", "x")):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))


def _req(i, text, status="provato"):
    return {"id": f"R{i}", "comma": 1, "kind": "requisito", "text": text, "status": status}


def _posture(family, role, stage="pre_filing"):
    from src.rag import posture
    return {"rito": family, "family": family, "role": role, "stage": stage,
            "entry": posture.entry_for(family, role, stage)}


ATTORE = _posture("civile", "attore")
CONVENUTO = _posture("civile", "convenuto", "primo_grado")


def _ctx(recs, case, posture=ATTORE, institutions=()):
    return {"records": recs, "circumstances": {}, "posture": posture, "case": case, "assessment": {},
            "institutions": list(institutions)}


def _rec(rubrica, *reqs, side="weapon"):
    return {"rubrica": rubrica, "side": side, "requisites": list(reqs)}


def _claims(cands):
    return [(c["kind"], c["article"]) for c in cands if c["kind"] in ("claim", "riconvenzionale")]


# --- B5: comodato -----------------------------------------------------------------------------------------------------------

def test_comodato_the_sellers_duty_and_a_limiting_rule_are_never_a_claim():
    from src.rag import lawyer_method as M
    case = "Tizio ha dato in comodato la sua auto a Caio, che a scadenza si rifiuta di restituirla."
    recs = {"1809 c.c.": _rec("Restituzione", _req(1, "la scadenza del termine convenuto")),
            "1476 c.c.": _rec("Obbligazioni principali del venditore", _req(2, "consegnare la cosa al compratore")),
            "1225 c.c.": _rec("Prevedibilità del danno", _req(3, "il danno prevedibile al tempo dell'obbligazione")),
            "1490 c.c.": _rec("Garanzia per i vizi della cosa venduta", _req(4, "la cosa venduta ha vizi"))}
    ctx = _ctx(recs, case, institutions=["comodato"])
    cands = M.thesis_candidates(ctx)
    assert _claims(cands) == [("claim", "1809 c.c.")]
    claim = next(c for c in cands if c["kind"] == "claim")
    assert claim["admitted_by"] == "modulo comodato"
    refused = ctx["claims_refused"]
    assert "non conferisce" in refused["1476 c.c."] and "prevedibilità" in refused["1225 c.c."]
    assert refused["1490 c.c."] == M.CLAIM_REFUSED["outside"]            # another institution's action
    assert all(c["admitted_by"] for c in cands)


# --- B5: appalto -------------------------------------------------------------------------------------------------------------

def test_appalto_the_module_comes_before_1218_and_1453_follow_only_without_a_proven_module_line():
    from src.rag import lawyer_method as M
    case = "Tizio ha affidato all'impresa Alfa il rifacimento del tetto; dopo le piogge compaiono infiltrazioni."
    recs = {"1218 c.c.": _rec("Responsabilità del debitore", _req(1, "l'inadempimento della prestazione")),
            "1453 c.c.": _rec("Risolubilità del contratto per inadempimento", _req(2, "l'inadempimento")),
            "1667 c.c.": _rec("Difformità e vizi dell'opera", _req(3, "l'opera presenta vizi", "da provare"))}
    ctx = _ctx(recs, case)
    cands = M.thesis_candidates(ctx)
    assert ctx["claim_modules"] == ["appalto"]
    assert _claims(cands) == [("claim", "1667 c.c."), ("claim", "1218 c.c."), ("claim", "1453 c.c.")]
    recs["1667 c.c."]["requisites"][0]["status"] = "provato"
    ctx = _ctx(recs, case)
    cands = M.thesis_candidates(ctx)
    assert _claims(cands) == [("claim", "1667 c.c.")]
    assert ctx["claims_refused"]["1218 c.c."] == M.CLAIM_REFUSED["subordinate"]


# --- B5: custodia ------------------------------------------------------------------------------------------------------------

def test_custodia_a_falling_building_part_puts_the_ruin_norm_above_the_custody_norm():
    from src.rag import lawyer_method as M
    case = "Il cornicione dell'edificio di Beta è crollato sull'auto di Tizio parcheggiata in strada."
    recs = {"2051 c.c.": _rec("Danno cagionato da cosa in custodia", _req(1, "la cosa in custodia del convenuto")),
            "2053 c.c.": _rec("Rovina di edificio", _req(2, "la rovina dell'edificio o di una sua parte")),
            "2043 c.c.": _rec("Risarcimento per fatto illecito", _req(3, "un fatto doloso o colposo"))}
    ctx = _ctx(recs, case)
    cands = M.thesis_candidates(ctx)
    assert _claims(cands) == [("claim", "2053 c.c."), ("claim", "2051 c.c.")]
    assert ctx["claims_refused"]["2043 c.c."] == M.CLAIM_REFUSED["subordinate"]


# --- B3 ----------------------------------------------------------------------------------------------------------------------

def test_a_defendant_never_gets_a_claim_and_has_the_transazione():
    from src.rag import lawyer_method as M
    recs = {"1218 c.c.": _rec("Responsabilità del debitore", _req(1, "l'inadempimento", "da provare"), side="threat"),
            "2043 c.c.": _rec("Risarcimento per fatto illecito", _req(2, "un fatto colposo dell'attore"))}
    recs["1218 c.c."]["charged"] = 1
    cands = M.thesis_candidates(_ctx(recs, "Beta cita Alfa per l'inadempimento.", CONVENUTO))
    kinds = [c["kind"] for c in cands]
    assert "claim" not in kinds and "remedy" not in kinds
    assert set(kinds) <= {"contest_open", "contest_excluded", "contest_integrated", "eccezione", "riconvenzionale",
                          "transazione"}
    tx = next(c for c in cands if c["kind"] == "transazione")
    assert tx["argued"] == "code" and tx["code_argument"]["realismo"] == "medio" and "1965" in tx["tesi"]


# --- B1 / B4 -------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("label, spec, same", [("2 L. 604/1966", "2 l. 604/1966", True),
                                               ("2 L. 15 luglio 1966, n. 604/1966", "2 l. 604/1966", False),
                                               ("2 d.lgs. 23/2015", "2 l. 604/1966", False),
                                               ("1809 c.c.", "1809 c.c.", True), ("1809 c.c.", "1810 c.c.", False)])
def test_a_special_laws_article_is_matched_by_number_and_law(label, spec, same):
    from src.rag import lawyer_method as M
    assert M.same_article(label, spec) is same


def test_the_counters_say_no_candidate_is_unadmitted_and_flag_a_main_line_outside_the_institution():
    from src.rag import lawyer_method as M
    case = "Tizio ha affidato all'impresa Alfa il rifacimento del tetto."
    recs = {"1218 c.c.": _rec("Responsabilità del debitore", _req(1, "l'inadempimento della prestazione")),
            "1667 c.c.": _rec("Difformità e vizi dell'opera", _req(3, "l'opera presenta vizi", "da provare"))}
    ctx = _ctx(recs, case)
    ctx["candidates"] = M.thesis_candidates(ctx)
    general = next(c for c in ctx["candidates"] if c.get("article") == "1218 c.c." and c["kind"] == "claim")
    plan = M.clean_method({"valutazioni": {c["id"]: {"realismo": "alto" if c is general else "basso"}
                                           for c in ctx["candidates"]}}, ctx["candidates"])
    counts = M.selection_counts(ctx, plan, {})
    # v79.12.1 (row 16): pertinence before realismo -- the module's line (1667) is promoted over the general one
    module = next(c for c in ctx["candidates"] if c.get("article") == "1667 c.c." and c["kind"] == "claim")
    assert counts["unadmitted"] == 0 and counts["chosen"] == module["id"] != general["id"]
    assert counts["promoted_by_pertinence"] == 1 and counts["main_outside_institution"] is False
    # with no line inside the institution eligible, the general one leads -- v79.12.5 (item 11): accepted, the
    # module yields nothing proven (the line is right: the counter no longer flags it)
    recs["1667 c.c."]["requisites"][0]["status"] = "non risulta"
    ctx = _ctx(recs, case)
    ctx["candidates"] = M.thesis_candidates(ctx)
    general = next(c for c in ctx["candidates"] if c.get("article") == "1218 c.c." and c["kind"] == "claim")
    plan = M.clean_method({"valutazioni": {c["id"]: {"realismo": "alto"} for c in ctx["candidates"]}},
                          ctx["candidates"])
    counts = M.selection_counts(ctx, plan, {})
    assert counts["chosen"] == general["id"] and counts["main_outside_institution"] is False
