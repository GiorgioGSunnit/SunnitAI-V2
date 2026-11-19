"""v79.8.5 -- the PO's review of 79.8, on synthetic sentences and records only:

C1 restated: only an attributed sentence (reported speech, a party's version, a claim letter) is an allegazione; a fact
    narrated in the quesito stays certain, whatever word it uses («lamenta forti dolori», «ha sporto denuncia», «il
    medico ha dichiarato una prognosi», «secondo il contratto») -- a lesione's elements proven by the narrative
B1's hole: a claimant whose facts give no norm conferring an action gets the explicit main line «Nessuna norma d'azione
    individuata nel registro per <istituto>» (v79.8.6: «Tra le norme esaminate per <istituto> nessuna attribuisce
    un'azione al cliente») -- never an empty main line, never the standard body or another fallback. v79.10.1 (A1)
    reverses it: never a candidate nor «Linea principale» -- the method steps aside to the prose answer (v76's note,
    whose why is the sentence; tests/test_v79_10_1.py)
data: the draft modules (famiglia, consumo, lavoro, successioni, condominio), for the lawyer's review
"""

import pytest


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for name, value in (("LLM_API_KEY", "k"), ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "k"), ("NEO4J_URI", "bolt://localhost:7687"), ("NEO4J_USER", "n"),
                        ("NEO4J_PASSWORD", "x")):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))
    from src.rag import plan_then_write
    monkeypatch.setattr(plan_then_write, "negation_review", lambda items, known, call=None: [])


# --- C1 restated ------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sentence", [
    "Tizio spinge Caio, che cade a terra e si frattura il polso.",
    "Caio lamenta forti dolori al braccio.",
    "Caio ha sporto denuncia il giorno dopo.",
    "Il medico del pronto soccorso ha dichiarato una prognosi di trenta giorni.",
    "Secondo il contratto il pagamento era dovuto a marzo.",
    "Caio accusa un malore e viene portato in ospedale.",
    "Alfa contesta la multa davanti al giudice di pace.",
])
def test_a_narrated_fact_stays_certain(sentence):
    from src.rag import issue_records
    assert issue_records.certainty(sentence) == "accertato"


@pytest.mark.parametrize("sentence", [
    "Tizio sostiene di essere stato provocato.",
    "Caio afferma che Tizio lo ha colpito per primo.",
    "Secondo Caio, Tizio era ubriaco.",
    "Secondo la controparte la merce era difettosa.",
    "Caio lamenta che Tizio lo abbia insultato.",
    # v79.12.7 (item 7): the PM's contestazione is a procedural act, a certain fact (test_v79_12_6_c); a party's
    # «contesta a» stays an allegazione
    "Caio contesta a Tizio di averlo spinto.",
    "Beta aveva negato la consegna.",
    "Caio accusa Tizio di averlo spinto.",
    "Caio ha dichiarato che Tizio era ubriaco.",
    "Con diffida del 5 aprile Alfa chiedeva il pagamento.",
])
def test_an_attributed_sentence_is_an_allegazione(sentence):
    from src.rag import issue_records
    assert issue_records.certainty(sentence) == "allegato"


def test_a_lesione_narrated_in_the_quesito_is_proven_and_only_the_attributed_part_stays_open():
    from src.rag import issue_records
    text = ("Tizio spinge Caio, che cade a terra e si frattura il polso. Caio lamenta forti dolori al braccio. Il medico "
            "del pronto soccorso ha dichiarato una prognosi di trenta giorni. Caio sostiene che Tizio lo abbia colpito "
            "anche con un pugno al volto.")
    sents = issue_records.sentences(text)
    by_id = {s["id"]: s for s in sents}

    def settled(text_, sup):
        req = {"kind": "requisito", "text": text_, "facts_status": "da provare",
               "link": {"supporting": list(sup), "contradicting": []}}
        issue_records.settle(req, by_id)
        return req["status"]
    assert settled("cagiona ad alcuno una lesione personale", ["S1"]) == "provato"
    assert settled("dalla quale deriva una malattia nel corpo o nella mente", ["S2", "S3"]) == "provato"
    assert settled("il colpo al volto", ["S4"]) == "da provare"


# --- B1: no norm confers an action -----------------------------------------------------------------------------------------

def _req(i, text, status="provato"):
    return {"id": f"R{i}", "comma": 1, "kind": "requisito", "text": text, "status": status}


def _posture(family, role, stage="pre_filing"):
    from src.rag import posture
    return {"rito": family, "family": family, "role": role, "stage": stage,
            "entry": posture.entry_for(family, role, stage)}


CASE = "Tizio ha dato in comodato la sua auto a Caio, che la restituisce graffiata."


def _ctx():
    recs = {"1476 c.c.": {"rubrica": "Obbligazioni principali del venditore", "side": "weapon",
                          "requisites": [_req(1, "consegnare la cosa al compratore")]},
            "1225 c.c.": {"rubrica": "Prevedibilità del danno", "side": "weapon",
                          "requisites": [_req(2, "il danno prevedibile al tempo dell'obbligazione")]}}
    return {"records": recs, "circumstances": {}, "posture": _posture("civile", "attore"), "case": CASE,
            "assessment": {}, "institutions": ["comodato"], "client": "Tizio"}


def test_a_claimant_with_no_norm_of_action_gets_the_explicit_line():
    from src.rag import lawyer_method as M
    ctx = _ctx()
    cands = M.thesis_candidates(ctx)
    assert cands == []                          # v79.10.1 (A1): never a candidate -- the sentence, for the prose's note
    assert ctx["no_action"] == "Tra le norme esaminate per comodato nessuna attribuisce un'azione al cliente"


def test_the_line_is_the_main_line_and_survives_the_wording_pass():
    # v79.10.1 (A1): reversed -- the method steps aside (the placeholder, the standard body) and v76's note says the
    # sentence once; it is never the main line
    from src.rag import lawyer_method as M
    ctx = _ctx()
    ctx["candidates"] = M.thesis_candidates(ctx)
    plan = M.clean_method({}, ctx["candidates"])
    counts = M.selection_counts(ctx, plan, {})
    assert counts["main_is_fallback"] is True and "no_action" not in counts
    assert M.placeholder_main(ctx, plan)                            # the standard body
    # v79.12.5 (item 2): the first read's institution (comodato) keeps the ten sections; with none, the step-aside
    assert M.method_refusal([(ctx, plan)]) == {}
    ctx = {**_ctx(), "institutions": []}
    ctx["candidates"] = M.thesis_candidates(ctx)
    assert M.compose([("Norme applicate ai fatti", "Regola.")], ctx, plan) == ""
    note = M.not_applied_note(M.method_refusal([(ctx, plan)])["why"])
    assert "tra le norme esaminate per comodato nessuna attribuisce un'azione al cliente" in note
    assert M.registro_hits(note) == [] and not any("registro" in x for x in M.wording_hits(note, [], False))


def test_with_no_institution_named_the_line_names_the_matter_or_the_facts():
    from src.rag import lawyer_method as M
    ctx = {**_ctx(), "institutions": [], "case": "Tizio chiede un consiglio."}
    assert M.thesis_candidates(ctx) == []
    assert ctx["no_action"] == "Tra le norme esaminate per contratti nessuna attribuisce un'azione al cliente"
    ctx = {**ctx, "records": {}}
    M.thesis_candidates(ctx)
    assert ctx["no_action"].startswith("Tra le norme esaminate per i fatti narrati")


def test_a_norm_of_action_with_an_element_excluded_is_not_the_no_action_line():
    from src.rag import lawyer_method as M
    ctx = {**_ctx(), "records": {"1809 c.c.": {"rubrica": "Restituzione", "side": "weapon",
                                               "requisites": [_req(1, "la scadenza del termine", "non risulta")]}}}
    assert M.thesis_candidates(ctx) == [] and ctx["claims_any_admitted"] is True       # v76's note, as before


def test_a_defendant_never_gets_the_line():
    from src.rag import lawyer_method as M
    ctx = {**_ctx(), "posture": _posture("civile", "convenuto", "primo_grado")}
    assert "nessuna_azione" not in [c["kind"] for c in M.thesis_candidates(ctx)] and not ctx.get("no_action")


# --- the draft modules ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("case, module, article", [
    ("Tizio impugna la delibera dell'assemblea condominiale.", "condominio", "1137 c.c."),
    ("Laura chiede le differenze retributive per il lavoro straordinario.", "lavoro", "2099 c.c."),
    ("Tizio, coerede, chiede la divisione dell'eredità.", "successioni", "713 c.c."),
    ("Il consumatore chiede al negozio Alfa s.r.l. il ripristino del bene in garanzia legale.", "consumo",
     "135-bis d.lgs. 206/2005"),                        # v79.9 (B3): a professional counterparty, a certain fact
    ("Tizia chiede l'assegno di mantenimento dopo la separazione.", "famiglia", "156 c.c."),
])
def test_the_draft_modules_admit_their_actions(case, module, article):
    from src.rag import lawyer_method as M
    mods = M.active_modules({"case": case})
    assert module in [m["id"] for m in mods] and all(m.get("draft") for m in mods if m["id"] == module)
    rule, group, _ = M.claim_admission(article, mods)
    assert rule == f"modulo {module}" and group == "module"


# --- v79.8.6: the E2 rule absolute ------------------------------------------------------------------------------------------

def test_no_sentence_on_the_registro_is_exempt_and_the_fixed_lines_never_say_it():
    from src.rag import lawyer_method as M
    old = "Nessuna norma d'azione individuata nel registro per comodato."
    assert M.reader_wording(old, {}, False) == ""                  # the 79.8.5 exemption is gone
    assert M.wording_hits(old, [], False) == ["«nel registro»"]
    for line in (M.NO_ACTION_LINE.format(istituto="comodato"), M.NO_SOURCE_ANSWER):
        assert "registr" not in line.lower() and M.reader_wording(line, {}, False) == line
    assert M.NO_SOURCE_ANSWER == "Non risulta dagli atti né dalle norme esaminate."


def test_the_nightly_fails_any_registro_in_an_answer():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("nightly_sim", Path(__file__).resolve().parents[1] / "scripts" /
                                                  "nightly_sim.py")
    nightly = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nightly)
    assert nightly.registro_check("Testo pulito.")["ok"] is True
    for text in ("Il registro non dà altro.", "| a | Registri delle imprese |", "come da registro"):
        bad = nightly.registro_check(text)
        assert bad["ok"] is False and bad["check"] == "no_registro"


def test_the_stage_lines_never_say_registro():
    import json
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "data" / "stage_procedure.json").read_text(encoding="utf-8")
    assert "registro" not in text.lower() and "art. 335, co. 3, c.p.p." in text
    json.loads(text)
