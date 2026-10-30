"""scripts/fetch_audit.py: the four mechanisms over saved runs, on run folders written here (invented texts),
a fake judge and fake commi -- no model, no database."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")
os.environ.setdefault("LLM_API_KEY", "test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest  # noqa: E402

import fetch_audit as fa  # noqa: E402

COMMA_1 = "Il debitore che non esegue esattamente la prestazione dovuta è tenuto al risarcimento del danno."
COMMA_2 = ("Se la prestazione diventa impossibile per causa non imputabile al debitore, l'obbligazione si estingue "
           "senza alcun obbligo di risarcire.")
ARTICLE = {("9999", "c.c."): [COMMA_1, COMMA_2]}
MATERIAL = (f"CASO:\nTizio non ha consegnato il bene a Caio entro la data pattuita.\n\nTESTI DELLE NORME:\n"
            f"[art. 9999 c.c.]\nRubrica: Responsabilità immaginaria\n{COMMA_1[:80]}\n\n"
            f"[art. 9998 Legge Inventata 2099]\nIl termine per la diffida è di quindici giorni.")
SECTION_REPLY = (
    "Tizio risponde del danno perché non ha eseguito esattamente la prestazione dovuta (art. 9999 c.c.). "
    "Tuttavia, se la prestazione è divenuta impossibile per causa a lui non imputabile, l'obbligazione si "
    "estingue (art. 9999 c.c.). La diffida va inviata entro quindici giorni (art. 9998 Legge Inventata 2099). "
    "La domanda si propone con ricorso monitorio davanti al giudice di pace del luogo del debitore. "
    "Il cliente, imputato nel procedimento, può costituirsi parte civile per ottenere il risarcimento.")


def _dump(path, n, stage, system, user, reply):
    path.write_text(f"stage: {stage}\ncall: {n:02d} (thread x)\nmax_tokens: 900\n\n===== SYSTEM =====\n{system}\n\n"
                    f"===== USER =====\n{user}\n\n===== REPLY =====\n{reply}\n", encoding="utf-8")


@pytest.fixture
def runs(tmp_path):
    diag = tmp_path / "20990101-0000"
    llm = diag / "cases" / "zz1" / "llm"
    llm.mkdir(parents=True)
    _dump(llm / "01_frame.txt", 1, "frame", "frame", "CASO:\nTizio non ha consegnato il bene.", '{"facts": []}')
    _dump(llm / "02_case_analysis_v2.txt", 2, "case_analysis_v2", "analisi", "CASO:\nTizio...",
          '{"parte_assistita": "Tizio", "posizione": "imputato", "fatti": [{"fatto": "mancata consegna"}]}')
    _dump(llm / "03_section_applicazione_al_caso.txt", 3, "section_applicazione_al_caso",
          "Scrivi UNA sezione di un parere motivato nell'interesse di Tizio (imputato), seguendo il PIANO già deciso",
          MATERIAL, SECTION_REPLY)
    record = {"id": "zz1", "draft": "**4. Applicazione al caso**\n" + SECTION_REPLY + "\n\n⚠️ Verificare tutto.",
              "frame": {"facts": ["Tizio non consegna il bene"], "tasks": [], "central_issue_candidates": []},
              "questioni": [{"titolo": "Inadempimento", "istituti": [], "parole_chiave": []}], "institutions": []}
    (diag / "cases" / "zz1.json").write_text(json.dumps(record), encoding="utf-8")
    ev = tmp_path / "direct-20990101-0000-off"
    ev.mkdir()
    (ev / "d1.json").write_text(json.dumps({
        "id": "d1", "question": "Il convenuto può chiedere la chiamata in causa del terzo?",
        "answer": "Il convenuto può chiamare in causa il terzo entro la prima udienza secondo la regola di legge.",
        "citations": [{"document_name": "Codice di procedura civile", "sections": [
            {"name": "269.0.0", "plain_text": "Il convenuto che intende chiamare un terzo deve farne dichiarazione."}]}]}),
        encoding="utf-8")
    (ev / "summary.json").write_text("{}", encoding="utf-8")
    dry = tmp_path / "c2.json"
    dry.write_text(json.dumps({"request": "Strategia per Caio", "draft": "Caio deve agire entro un anno dal fatto.",
                               "citations": []}), encoding="utf-8")
    return diag, ev, dry


class FakeCommi:
    def get(self, number, code):
        return (ARTICLE.get((number, code)) or [], "sections")


def fake_judge(system, human, max_tokens):
    """Supported when the material holds the sentence's key words; the rest by fixed answers."""
    if system == fa.PROVENANCE_SYSTEM:
        lines = [ln for ln in human.split("FRASI:\n", 1)[1].splitlines() if ln.strip()]
        out = []
        for ln in lines:
            n, sentence = ln.split(". ", 1)
            if "risponde del danno" in sentence:
                out.append({"i": int(n), "tipo": "regola", "esito": "supportata", "prova": COMMA_1[:40]})
            elif "impossibile" in sentence:
                out.append({"i": int(n), "tipo": "regola", "esito": "assente"})
            elif "monitorio" in sentence:
                out.append({"i": int(n), "tipo": "procedura", "esito": "assente"})
            elif "prima udienza" in sentence:
                out.append({"i": int(n), "tipo": "procedura", "esito": "contraddetta", "prova": "deve farne"})
            else:
                out.append({"i": int(n), "tipo": "altro", "esito": None})
        return json.dumps(out)
    if system == fa.ROLE_SYSTEM:
        lines = human.split("FRASI:\n", 1)[1].splitlines()
        hits = [{"i": i, "perche": "rimedio della persona offesa"} for i, ln in enumerate(lines, 1) if "parte civile" in ln]
        return json.dumps({"parte": "Tizio", "ruolo": "imputato", "contrasti": hits})
    if system == fa.UNDERSTANDING_SYSTEM:
        return json.dumps({"fatto": "consegna entro la data pattuita", "tipo": "date_termini", "catturato_da": [],
                           "perche": "il termine essenziale"})
    raise AssertionError(system[:40])


class Args:
    batch, material_chars = 12, 14000


# --- pure pieces --------------------------------------------------------------------------------------------

def test_citations_material_blocks_and_numbers():
    assert fa.norm_number("415 BIS") == "415-bis" and fa.norm_number("2051") == "2051"
    labels = fa.labels_of(MATERIAL)
    assert labels == [("9999", "c.c."), ("9998", "Legge Inventata 2099")]
    cites = fa.citations_in("Si applica l'art. 9999, comma 2, c.c. e l'art. 9998 Legge Inventata 2099.", labels)
    assert [(c["number"], c["code"], c["comma"]) for c in cites] == [("9999", "c.c.", 2),
                                                                      ("9998", "Legge Inventata 2099", None)]
    blocks = fa.material_blocks(MATERIAL)
    assert blocks[("9999", "c.c.")] == COMMA_1[:80] and "quindici giorni" in blocks[("9998", "Legge Inventata 2099")]


def test_granularity_tells_the_comma_given_from_the_one_cut_off():
    block = fa.material_blocks(MATERIAL)[("9999", "c.c.")]
    commi = ARTICLE[("9999", "c.c.")]
    first = fa.granularity_claim("Tizio risponde del danno perché non ha eseguito esattamente la prestazione dovuta "
                                 "(art. 9999 c.c.).", {"number": "9999", "code": "c.c.", "comma": None}, block, commi)
    second = fa.granularity_claim("Se la prestazione è divenuta impossibile per causa non imputabile, "
                                  "l'obbligazione si estingue (art. 9999 c.c.).",
                                  {"number": "9999", "code": "c.c.", "comma": None}, block, commi)
    other = fa.granularity_claim("x", {"number": "1", "code": "c.c.", "comma": None}, None, [])
    assert (first["status"], first["comma"]) == ("in_material", 1)
    assert (second["status"], second["comma"]) == ("beyond_truncation", 2)
    assert other["status"] == "article_not_given"
    # a block holding a later comma only: the comma relied on was not given, but nothing was cut
    assert fa.granularity_claim(second["sentence"], {"number": "9999", "code": "c.c.", "comma": 1},
                                COMMA_2, commi)["status"] == "comma_not_given"


def test_a_json_object_holding_a_list_stays_an_object():
    assert fa.parse_json('Ecco: {"contrasti": [{"i": 1}]}') == {"contrasti": [{"i": 1}]}
    assert fa.parse_json('[{"i": 1, "x": {"a": 2}}]') == [{"i": 1, "x": {"a": 2}}]
    assert fa.parse_json("nessun json") is None


def test_split_commi_from_sections_or_text():
    assert fa.split_commi([(2, "B."), (1, "A.")]) == (["A.", "B."], "sections")
    assert fa.split_commi([], "Primo comma.\nSecondo comma.")[1] == "paragraphs"
    assert fa.split_commi([], "")[1] == "none"


def test_answer_sentences_skip_headings_notes_and_fragments():
    text = "**4. Applicazione al caso**\nUna frase abbastanza lunga da essere valutata qui.\n⚠️ Verificare tutto.\nBreve."
    assert fa.answer_sentences(text) == ["Una frase abbastanza lunga da essere valutata qui."]


def test_the_dumps_give_the_writer_material_case_text_and_role(runs):
    diag, _, _ = runs
    case = fa.load([str(diag)])[0]
    assert case["role"] == {"party": "Tizio", "position": "imputato", "from": "section prompt"}
    assert case["case_text"].startswith("Tizio non ha consegnato il bene")
    mapped = fa.sentence_map(case)
    assert {m["stage"] for m in mapped} == {"section_applicazione_al_caso"}
    assert all("TESTI DELLE NORME" in m["material"] for m in mapped)


# --- the mechanisms -----------------------------------------------------------------------------------------

def test_the_four_mechanisms_on_a_diagnose_case(runs):
    diag, _, _ = runs
    case = fa.load([str(diag)])[0]
    out = fa.audit_case(case, fa.Judge(None, call=fake_judge), FakeCommi(), None, Args)
    g = out["granularity"]["counts"]
    assert g == {"in_material": 1, "beyond_truncation": 1, "no_text": 1}
    assert out["granularity"]["defects"] == 1
    p = out["provenance"]
    assert p["rule_sentences"] == 3 and p["counts"] == {"supportata": 1, "assente": 2}
    # the "impossibile" sentence is not in the material because its comma was cut: counted once, as granularity
    assert p["explained_by_granularity"] == 1 and p["defects"] == 1
    r = out["role"]
    assert r["defects"] == 1 and r["contradictions"][0]["pattern"] and r["contradictions"][0]["judge"]
    assert out["understanding"] == {"lost": [], "available": False, "defects": 0}


def test_understanding_reports_the_trigger_and_whether_anything_captured_it(runs):
    diag, _, _ = runs
    case = fa.load([str(diag)])[0]
    analysed = {"slots": [{"label": "art. 9997 cc", "stage": "LOST_AT_UNDERSTANDING", "in_expected_rubrica": "Termine"},
                          {"label": "art. 9999 cc", "stage": "MENTIONED"}]}
    u = fa.audit_understanding(case, fa.Judge(None, call=fake_judge), analysed)
    assert u["defects"] == 1 and u["lost"][0]["trigger"] == "date_termini" and u["lost"][0]["captured"] == []


def test_eval_and_dryrun_runs_use_the_returned_sources_as_a_proxy(runs):
    _, ev, dry = runs
    cases = fa.load([str(ev), str(dry)])
    assert [c["key"] for c in cases] == ["direct-20990101-0000-off/d1", "c2.json/c2"]
    out = fa.audit_case(cases[0], fa.Judge(None, call=fake_judge), FakeCommi(), None, Args)
    assert out["proxy"] and out["granularity"]["skipped"]
    assert out["provenance"]["counts"] == {"contraddetta": 1}


def test_the_summary_ranks_and_the_report_proposes_fixes_for_the_top_two_only(runs):
    diag, ev, dry = runs
    judge = fa.Judge(None, call=fake_judge)
    done = [fa.audit_case(c, judge, FakeCommi(), None, Args) for c in fa.load([str(diag), str(ev), str(dry)])]
    s = fa.summary(done)
    assert s["totals"] == {"granularity": 1, "provenance": 2, "understanding": 0, "role": 1}
    assert s["ranked"][0] == "provenance" and abs(sum(s["share"].values()) - 1) < 1e-9
    report = fa.render(done, {"runs": ["x"], "judge": "fake", "commi": "fake", "when": "now"})
    for title in fa.TITLES.values():
        assert f"## {title}" in report
    fixes = report.split("## Fix directions for the top two mechanisms", 1)[1]
    assert fixes.count("### ") == 2 and "2 Provenance" in fixes
    assert "4 Role" not in fixes or "1 Granularity" not in fixes


def test_main_writes_the_report_and_reuses_the_judge_cache(runs, tmp_path, monkeypatch):
    diag, ev, dry = runs
    calls = []
    monkeypatch.setattr(fa.Judge, "_chat", staticmethod(lambda s, h, m: calls.append(1) or fake_judge(s, h, m)))
    out = tmp_path / "audit"
    argv = [str(diag), str(ev), str(dry), "--out", str(out), "--no-db", "--env", str(tmp_path / "none.env")]
    assert fa.main(argv) == 0
    first = len(calls)
    assert first > 0 and (out / "report.md").exists() and (out / "judge_cache.jsonl").exists()
    data = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    # --no-db: no commi, so the sentence on the cut comma is not granularity's and stays provenance's
    assert data["summary"]["totals"]["provenance"] == 3 and data["summary"]["totals"]["granularity"] == 0
    assert fa.main(argv) == 0 and len(calls) == first          # all from the cache
    assert fa.main(["--report-only", str(out)]) == 0


def test_no_judge_keeps_granularity_and_the_role_patterns(runs):
    diag, _, _ = runs
    case = fa.load([str(diag)])[0]
    out = fa.audit_case(case, fa.Judge(None, enabled=False), FakeCommi(), None, Args)
    assert out["granularity"]["defects"] == 1 and out["provenance"]["rule_sentences"] == 0
    assert out["role"]["defects"] == 1 and not out["role"]["contradictions"][0]["judge"]


def test_understanding_reads_the_lost_stages_from_diagnose_long_itself(tmp_path):
    """The real analyse_case on diagnose_long's own synthetic record: 61 c.p. LOST_AT_UNDERSTANDING, 2043 c.c.
    NOT_IN_PLAN, both judged."""
    import test_diagnose_long as tdl
    run = tmp_path / "run"
    (run / "cases").mkdir(parents=True)
    (run / "cases" / "x1.json").write_text(json.dumps(tdl._record()), encoding="utf-8")
    case = fa.load([str(run)])[0]
    out = fa.audit_case(case, fa.Judge(None, call=fake_judge), FakeCommi(), tdl.ev, Args)
    lost = {x["article"]: x["stage"] for x in out["understanding"]["lost"]}
    assert lost == {"art. 61 cp": "LOST_AT_UNDERSTANDING", "art. 2043 cc": "NOT_IN_PLAN"}
    assert out["understanding"]["defects"] == 2
    assert fa.summary([out])["triggers"] == {"date_termini": 2}


def test_dict_shaped_fact_signals_and_other_field_shapes_are_read():
    """Prod, run 0812: FACT_SIGNALS' facts are a dict of typed fields, not a list (KeyError: slice)."""
    rec = {"frame": {"facts": {"main": ["Tizio vende a Caio"]}, "tasks": "un solo compito",
                     "central_issue_candidates": [{"nome": "servitù di passaggio", "codice": "c.c."}]},
           "questioni": [{"titolo": "Servitù", "istituti": ["servitù"]}, "una questione come testo"],
           "institutions": [{"nome": "usucapione", "articles": [{"label": "1158 c.c."}, "1061 c.c."]}, "possesso"],
           "fact_signals": {"facts": {"parties": [{"name": "Tizio", "role": "privato"}],
                                      "events": [{"kind": "vendita", "date": "2010"}],
                                      "property_history": [{"owner": "Sempronio", "until": "1990"}],
                                      "notes": "un testo", "money": [], "questions": None},
                            "fired": [{"id": "usucapione_servitu_apparente", "institution": "usucapione"}]}}
    calls = [{"stage": "case_analysis_v2", "reply": '{"fatti": {"principali": [{"fatto": "passaggio dal 1995"}]}}'}]
    items = fa._extracted(rec, calls)
    joined = "\n".join(items)
    assert "frame facts main: Tizio vende a Caio" in joined and "frame tasks: un solo compito" in joined
    assert "frame central_issue_candidates: servitù di passaggio" in joined
    assert "fatto principali: passaggio dal 1995" in joined
    assert "questione: una questione come testo" in joined and "istituto: possesso" in joined
    assert "istituto: usucapione -> 1158 c.c., 1061 c.c." in joined
    assert 'fatto tipizzato parties: {"name": "Tizio", "role": "privato"}' in joined
    assert "fatto tipizzato property_history" in joined and "fatto tipizzato notes: un testo" in joined
    assert not any("money" in i or "questions" in i for i in items)
    assert "regola scattata: usucapione" in joined
    # a list shape still works, and nothing at all gives nothing
    assert fa._extracted({"fact_signals": {"facts": ["un fatto"]}}) == ["fatto tipizzato: un fatto"]
    assert fa._extracted({"frame": None, "fact_signals": None, "questioni": None}) == []


def test_understanding_with_dict_shaped_facts_and_a_string_captured_list(tmp_path):
    import test_diagnose_long as tdl
    rec = {**tdl._record(), "fact_signals": {"facts": {"events": [{"kind": "aggressione", "date": "2099-01-01"}]}}}
    run = tmp_path / "run"
    (run / "cases").mkdir(parents=True)
    (run / "cases" / "x1.json").write_text(json.dumps(rec), encoding="utf-8")
    case = fa.load([str(run)])[0]

    def judge(system, human, max_tokens):
        return json.dumps({"fatto": "aggressione", "tipo": "gravita_aggravante", "catturato_da": "1, 2"})
    out = fa.audit_case(case, fa.Judge(None, call=judge), FakeCommi(), tdl.ev, Args)
    assert [len(x["captured"]) for x in out["understanding"]["lost"]] == [2, 2]
    assert out["understanding"]["defects"] == 0
