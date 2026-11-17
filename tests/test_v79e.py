"""v79 block E (79.4) -- forecast, wording, ledger, on synthetic records only:

E1. the forecast (assessment) on the chosen main line's article; a victim's or a parte civile's on the querela's offence
E2. no internal wording reaches the reader: element codes only inside tables; «l'accusa» and «aggravante» only in a
    criminal matter; the «registro» sentence never -- and the nightly's output check counts none
E3. compose's removals enter the ledger as «removed» (it is the last check); the nightly checks every sentence logged
    as removed is absent
E4. prescription and decadenza (c.c. 2934-2969, rubriche on them) are never claims or candidates
"""

import importlib.util
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


def _script(name):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _req(i, text, status="provato"):
    return {"id": f"R{i}", "comma": 1, "kind": "requisito", "text": text, "status": status}


IMPUTATO = {"rito": "penale", "family": "penale", "role": "imputato", "stage": "primo_grado"}
CONVENUTO = {"rito": "civile", "family": "civile", "role": "convenuto", "stage": "primo_grado"}
OFFESA = {"rito": "penale", "family": "penale", "role": "persona_offesa", "stage": "pre_filing"}


def _ctx(recs, posture=IMPUTATO):
    from src.rag import lawyer_method as M
    return M.prepare({"records": recs, "sentences": [{"id": "S1", "text": "Alfa trattiene il denaro di Beta.",
                                                      "source": "caso", "certainty": "accertato"}]}, posture, [],
                     "Alfa trattiene il denaro di Beta.", "Alfa", "", list(recs), [], [], [], [],
                     link_call=lambda *a: {}, case_id_="v79e")


def _penal_records():
    return {"640 c.p.": {"rubrica": "Truffa", "side": "threat", "charged": 1,
                         "requisites": [_req(1, "artifizi o raggiri"), _req(2, "induce taluno in errore", "da provare")]},
            "646 c.p.": {"rubrica": "Appropriazione indebita", "side": "weapon", "alternative": True,
                         "requisites": [_req(3, "il possesso del denaro altrui"),
                                        _req(4, "si appropria del denaro")]}}


# --- E1 ------------------------------------------------------------------------------------------------------------------

def test_the_forecast_follows_the_main_lines_article():
    from src.rag import lawyer_method as M
    ctx = _ctx(_penal_records())
    assert ctx["assessment"]["article"] == "640 c.p."
    line = {"tesi": "Riqualificazione nell'art. 646 c.p.", "article": "646 c.p.", "contesta": ["R1"],
            "poggia_su": ["R3", "R4"], "candidate": "T2", "kind": "requalify"}
    log = {}
    a = M.forecast(ctx, line, ctx["assessment"], log)
    assert a["article"] == "646 c.p." and log["forecast"] == {"was": "640 c.p.", "now": "646 c.p.", "esito": a["esito"]}


def test_a_line_on_no_fattispecie_keeps_the_charges_forecast():
    from src.rag import lawyer_method as M
    ctx = _ctx(_penal_records())
    fallback = {"tesi": "Contenimento", "fallback": True, "contesta": [], "poggia_su": []}
    defence = {"tesi": "Difesa sull'art. 55 c.p.", "article": "55 c.p.", "poggia_su": ["R9"], "kind": "defence"}
    assert M.forecast(ctx, fallback, ctx["assessment"]) is ctx["assessment"]
    assert M.forecast(ctx, defence, ctx["assessment"]) is ctx["assessment"]


def test_a_victims_forecast_stays_on_the_querelas_offence():
    from src.rag import lawyer_method as M
    recs = {"582 c.p.": {"rubrica": "Lesione personale", "side": "weapon", "charged": 1,
                         "requisites": [_req(1, "cagiona una lesione")]},
            "2043 c.c.": {"rubrica": "Risarcimento per fatto illecito", "side": "weapon",
                          "requisites": [_req(2, "fatto doloso o colposo")]}}
    ctx = _ctx(recs, OFFESA)
    line = {"tesi": "Domanda", "article": "2043 c.c.", "poggia_su": ["R2"], "kind": "claim"}
    assert M.forecast(ctx, line, ctx["assessment"])["article"] == "582 c.p."


def test_compose_prints_the_forecast_of_the_selected_line():
    from src.rag import lawyer_method as M
    ctx = _ctx(_penal_records())
    ctx["candidates"] = [{"id": "T1", "kind": "requalify", "tesi": "Riqualificazione nell'art. 646 c.p.",
                          "contesta": [], "poggia_su": ["R3", "R4"], "label": M.SUSTAINABLE, "argued": "model",
                          "article": "646 c.p."}]
    plan = M.clean_method({"valutazioni": {"T1": {"realismo": "alto", "pro": "p", "contro": "c", "argomento": "a"}}},
                          ctx["candidates"])
    log = {}
    out = M.compose([("", "Testa.")], ctx, plan, (), log)
    assert log["forecast"]["now"] == "646 c.p." and "dell'art. 646 c.p." in dict(M.answer_sections(out))[
        M.title("conclusioni")].split("\n")[0]


# --- E2 ------------------------------------------------------------------------------------------------------------------

KNOWN = {"R1": {"text": "artifizi o raggiri"}, "R2": {"text": "induce taluno in errore"}}


def test_element_codes_leave_the_prose_and_stay_in_the_tables():
    from src.rag import lawyer_method as M
    text = ("L'elemento «artifizi o raggiri» (R1) è provato; dipende dagli elementi R1, R2.\n"
            "| Elemento | Esito |\n| R1 artifizi | provato |")
    out = M.reader_wording(text, KNOWN, True)
    assert "(R1)" not in out and "dipende dagli elementi «artifizi o raggiri», «induce taluno in errore»." in out
    assert "| R1 artifizi | provato |" in out
    assert M.wording_hits(out, list(KNOWN), True) == []
    assert M.wording_hits(text, list(KNOWN), True) == ["codice R1", "codice R1", "codice R2"]


def test_an_unknown_code_is_left_alone():
    from src.rag import lawyer_method as M
    assert M.reader_wording("La categoria C3 del contratto.", KNOWN, False) == "La categoria C3 del contratto."


def test_accusa_and_aggravante_only_in_a_criminal_matter():
    from src.rag import lawyer_method as M
    text = "L'accusa sosterrà il danno. Nessuna aggravante risulta integrata. Il danno va provato da l'accusa."
    civil = M.reader_wording(text, {}, False)
    assert civil == "La controparte sosterrà il danno. Il danno va provato da la controparte."
    assert M.reader_wording(text, {}, True) == text
    assert M.wording_hits(text, [], False) == ["«L'accusa»", "«l'accusa»", "«aggravant…»"]
    assert not M.criminal({"family": "civile", "role": "convenuto"}) and M.criminal({"family": "penale"})
    assert M.criminal({"family": "", "role": "imputato"}) and not M.criminal({"family": "", "role": "attore"})


def test_the_registro_sentence_never_reaches_the_reader():
    from src.rag import lawyer_method as M
    text = "Il fatto è provato. Il registro non contiene l'addebito con i suoi elementi. Si procede."
    assert M.reader_wording(text, {}, True) == "Il fatto è provato. Si procede."
    for key in ("placeholder",):
        assert "registro" not in M.NOT_APPLIED_WHY[key]
    assert "registro" not in M.CODE_ARGUMENTS["contenimento"]["pro"]


def test_a_civil_parere_carries_no_criminal_wording_and_no_codes():
    from src.rag import lawyer_method as M
    recs = {"2051 c.c.": {"rubrica": "Danno cagionato da cosa in custodia", "side": "threat", "charged": 1,
                          "requisites": [_req(1, "la cosa è in custodia"), _req(2, "il danno deriva dalla cosa",
                                                                                   "da provare")]}}
    ctx = _ctx(recs, CONVENUTO)
    ctx["candidates"] = M.thesis_candidates(ctx)
    plan = M.clean_method({}, ctx["candidates"])
    out = M.compose([("", "Testa.")], ctx, plan, (), {})
    got = M.output_check(out, ctx["output_sources"])
    assert got["wording"] == [] and "aggravant" not in out.lower()
    assert "l'accusa" not in out.lower()


# --- E3 ------------------------------------------------------------------------------------------------------------------

def test_composes_removals_are_removed_entries_and_merged_into_the_ledger():
    from src.rag import interactive_flow as flow, lawyer_method as M
    ctx = _ctx(_penal_records())
    ctx["candidates"] = M.thesis_candidates(ctx)
    vals = {c["id"]: {"pro": "Il fatto non è provato. L'art. 2043 c.c. richiede un danno ingiusto.", "contro": "c",
                      "realismo": "medio", "argomento": "a"} for c in ctx["candidates"]}
    plan = M.clean_method({"valutazioni": vals}, ctx["candidates"])
    log = {}
    out = M.compose([("", "Testa.")], ctx, plan, (), log)
    removed = [e for e in log["ledger"] if e["kind"] == "removed"]
    assert any("2043" in e["sentence"] and e["step"].startswith("compose:") for e in removed)
    state = {"ledger": {"entries": [{"step": "record_check", "kind": "removed", "sentence": "x"}],
                        "removed_by_step": {"record_check": 1}}}
    flow.merge_removals(state, log["ledger"])
    assert state["ledger"]["removed_by_step"]["record_check"] == 1
    assert sum(v for k, v in state["ledger"]["removed_by_step"].items() if k.startswith("compose:")) == len(removed)
    nightly = _script("nightly_sim")
    assert nightly.ledger_check(out, state)["ok"] is True


def test_the_nightly_fails_a_removed_sentence_still_in_the_answer():
    nightly = _script("nightly_sim")
    state = {"ledger": {"entries": [{"step": "compose:scope", "kind": "removed",
                                     "sentence": "L'art. 2043 c.c. richiede un danno ingiusto."},
                                    {"step": "x", "kind": "proposed", "sentence": "Il fatto non è provato del tutto."}]}}
    got = nightly.ledger_check("Testo. L'art. 2043 c.c. richiede un danno ingiusto. Il fatto non è provato del tutto.",
                               state)
    assert got["ok"] is False and "compose:scope" in got["why"]
    assert nightly.ledger_check("Testo.", state)["ok"] is True


def test_the_nightly_wording_check_reads_the_output_check():
    nightly = _script("nightly_sim")
    assert nightly.wording_check({"wording": []}, {"output_sources": {"x": 1}})["ok"] is True
    bad = nightly.wording_check({"wording": ["codice R1"]}, {"output_sources": {"x": 1}})
    assert bad["ok"] is False and "R1" in bad["why"]
    assert nightly.wording_check({}, {})["ok"] is False


# --- E4 ------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("label, rubrica", [("2934 c.c.", "Estinzione dei diritti"), ("2943 c.c.", "Interruzione da "
                                            "parte del titolare"), ("2969 c.c.", "Rilevabilità d'ufficio"),
                                            ("1495 c.c.", "Termini e condizioni per l'azione: decadenza"),
                                            ("2935 c.c.", "Decorrenza della prescrizione")])
def test_prescription_and_decadenza_are_never_claims(label, rubrica):
    from src.rag import issue_records
    assert issue_records.prescription(label, rubrica) and not issue_records.counted(label, rubrica)


def test_an_ordinary_article_still_counts():
    from src.rag import issue_records
    assert issue_records.counted("2933 c.c.", "Rinunzia alla prescrizione") is False       # the rubrica decides
    assert issue_records.counted("2970 c.c.", "Ipoteca") and issue_records.counted("2043 c.c.", "Risarcimento")


def test_no_prescription_candidate_for_a_claimant_or_a_convenuto():
    from src.rag import lawyer_method as M
    recs = {"2947 c.c.": {"rubrica": "Prescrizione del diritto al risarcimento del danno", "side": "weapon",
                          "requisites": [_req(1, "il decorso di cinque anni")]},
            "2043 c.c.": {"rubrica": "Risarcimento per fatto illecito", "side": "weapon",
                          "requisites": [_req(2, "il fatto doloso o colposo")]}}
    attore = {"rito": "civile", "family": "civile", "role": "attore", "stage": "pre_filing"}
    cands = M.thesis_candidates(_ctx(recs, attore))
    assert cands and not any("2947" in c["tesi"] or c.get("article") == "2947 c.c." for c in cands)
    claim = {"2051 c.c.": {"rubrica": "Danno cagionato da cosa in custodia", "side": "threat", "charged": 1,
                           "requisites": [_req(3, "la cosa è in custodia")]}}
    eccezioni = [c for c in M.thesis_candidates(_ctx(claim, CONVENUTO)) if c["kind"] == "eccezione"]
    # v79.6: E4 applies to claims only -- the convenuto's eccezione di prescrizione stays a candidate
    assert any(M.prescription_cited(str(c.get("articles"))) for c in eccezioni)
