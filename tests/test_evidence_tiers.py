"""Evidence tiers (src/rag/evidence_tiers.py): T1/T2/T3 on every candidate, evidence before rank at every cut-off,
removal by agreement for evidenced candidates, a fixed order among equals. Invented cases only."""

import json
import os

import pytest

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

import src.rag.answer_invariants as ai  # noqa: E402
import src.rag.defensive_generation as dg  # noqa: E402
import src.rag.evidence_tiers as et  # noqa: E402
import src.rag.plan_then_write as ptw  # noqa: E402

CASE = "Ottavia passa da trent'anni sulla carrareccia di Bruno, che nel 2024 ha messo una sbarra chiusa a chiave."


@pytest.fixture
def logged(monkeypatch):
    out = []
    monkeypatch.setattr(dg, "vlog", lambda event, data=None, *a, **k: out.append((event, data)))
    monkeypatch.setattr(ptw.dg, "vlog", lambda event, data=None, *a, **k: out.append((event, data)))
    return out


def _row(number, code="c.c.", text="Testo.", heading="", **extra):
    return {"d": {"id": f"doc-{code}", "name": "Codice Civile 2026"}, "_article": (number, code),
            "s": {"name": f"{number}.0.0", "plain_text": text, "titolo_articolo": heading}, **extra}


def _events(logged, name):
    return [d for e, d in logged if e == name]


# --- the tiers ------------------------------------------------------------------------------------------------------

def test_tiers_from_what_found_each_article():
    rows = [_row("1079"), _row("1062"), _row("1051", _heading_exact=True), _row("700", "c.p.c.", _regime=True),
            _row("1061", _vote_stats={"rulings": 3}), _row("1064", _multi_signal=True),
            _row("1066", _signals=["baseline", "map"]), _row("1027", _signals=["baseline"]),
            _row("1031", _vote_stats={"rulings": 2}), _row("1062", _signals=["distill"])]
    log = et.annotate(rows, named=[("1079", "c.c.")], fact_refs=[("1062", "c.c.")])
    assert [r["_tier"] for r in rows] == [1, 1, 1, 1, 2, 2, 2, 3, 3, 1]          # 1062's second row: its best tier
    assert log["T1"] == ["1051 c.c.", "1062 c.c.", "1079 c.c.", "700 c.p.c."]
    assert log["T2"] == ["1061 c.c.", "1064 c.c.", "1066 c.c."] and log["T3"] == 2
    assert log["why"]["1061 c.c."] == "applied by 3 similar rulings"
    assert et.tier({}) is None and not et.evidenced({"_tier": 3}) and et.evidenced({"_tier": 2})


def test_fixed_orders():
    assert sorted([("614-bis", "c.p.c."), ("614", "c.p.c."), ("61", "c.p.c.")], key=et.ref_key) == [
        ("61", "c.p.c."), ("614", "c.p.c."), ("614-bis", "c.p.c.")]
    rows = [_row("5", _tier=3, _grade=3, _fused_rank=1), _row("4", _tier=1, _grade=1, _fused_rank=9),
            _row("3", _tier=1, _grade=2, _fused_rank=9), _row("2", _tier=1, _grade=2, _fused_rank=9)]
    assert [r["_article"][0] for r in sorted(rows, key=et.cut_key)] == ["2", "3", "4", "5"]   # tier first
    assert [r["_article"][0] for r in sorted(rows, key=lambda r: et.forced_key(r, r["_grade"]))] == ["5", "2", "3", "4"]


def test_an_exact_heading_match_of_the_institution_is_marked(monkeypatch):
    from src.rag import issue_spotting
    monkeypatch.setattr(issue_spotting, "article_by_title",
                        lambda session, doc, term, field="titolo_articolo": "1051" if term == "passaggio coattivo"
                        else "1052" if term == "passaggio" and field == "titolo_articolo" else None)
    monkeypatch.setattr(issue_spotting, "fetch_article", lambda session, doc, n: [
        {"d": {"id": "cc"}, "s": {"name": f"{n}.0.0", "titolo_articolo": "Passaggio coattivo", "plain_text": "x"}}])
    docs = [{"id": "cc", "name": "Codice Civile 2026"}]
    exact = dg._resolve_institution(None, docs, {"nome": "passaggio coattivo", "codice": "c.c."})
    assert exact[0]["_heading_exact"] is True
    by_keyword = dg._resolve_institution(None, docs, {"nome": "fondo chiuso", "codice": "c.c."}, ["passaggio"])
    assert by_keyword[0]["_article"] == ("1052", "c.c.") and not by_keyword[0].get("_heading_exact")


# --- Rule 1 at the fusion cap ---------------------------------------------------------------------------------------

def test_the_fusion_cap_never_cuts_a_fact_rule_or_a_well_voted_article(monkeypatch, logged):
    from src.rag import vote_pool
    import src.rag.doc_lookup as doc_lookup
    import src.rag.nodes.retrieval as retrieval_module
    many = [(str(100 + i), "c.p.") for i in range(30)]
    v2 = {"first_rulings": [], "similar": [], "keep": [], "named": [("575", "c.p.")], "candidates": [], "chosen": [],
          "selected": [], "article_rows": [], "spotted": [], "institutions_resolved": [], "heading_matched": [],
          "fact_resolved": [("129", "c.p.")], "heading_exact": []}
    monkeypatch.setattr(dg, "_select_articles_v2", lambda *a, **k: v2)
    monkeypatch.setattr(dg, "_fusion_distill_bullets", lambda case: [])
    monkeypatch.setattr(dg, "_fusion_text_search", lambda *a, **k: many)
    monkeypatch.setattr(doc_lookup, "_fetch_doc_names", lambda driver, database: [])
    monkeypatch.setattr(dg, "_fetch_any", lambda session, documents, refs: [
        {"d": {"id": "doc"}, "s": {"name": f"{n}.0.0", "plain_text": "t"}, "_article": (n, c)} for n, c in refs])
    monkeypatch.setattr(retrieval_module, "broaden_article_context", lambda state, d, db: {"raw_result": state["raw_result"]})

    def vote(session, driver, database, case, index, analysis, stats):
        stats.update({("640", "c.p."): {"rulings": 4}, ("641", "c.p."): {"rulings": 1}})
        return [("641", "c.p."), ("640", "c.p.")]
    monkeypatch.setattr(vote_pool, "ruling_vote", vote)
    monkeypatch.setenv("VOTE_POOL_ENABLED", "true")
    monkeypatch.setenv("VOTE_POOL_MODE", "append:1")
    picked = dg._select_articles_fusion(object(), object(), "db", "facts", "", {"questioni": []})
    assert ("129", "c.p.") in picked["selected"]                     # a fact rule's article, 30th by rank
    assert ("640", "c.p.") in picked["selected"] and ("641", "c.p.") in picked["selected"]
    tiers = {r["_article"]: r["_tier"] for r in picked["article_rows"]}
    assert tiers[("129", "c.p.")] == 1 and tiers[("640", "c.p.")] == 2 and tiers[("641", "c.p.")] == 3
    cut = _events(logged, "evidence_cutoff")[0]
    assert cut["stage"] == "fusion_cap" and cut["t12_dropped"] == [] and all(i["tier"] == 3 for i in cut["cut"])
    assert picked["tiers"] is True and _events(logged, "evidence_tiers")


# --- Rules 1 and 2 at the check -------------------------------------------------------------------------------------

def _graded_env(monkeypatch, grades, second=None):
    monkeypatch.setenv("CHECK_GRADED_ENABLED", "true")
    monkeypatch.setattr(dg, "_graded_verdicts", lambda case, texts, *a, **k: (
        {dg._applicability_key(label): {"article": label, "grade": grades.get(label.split()[0], 2), "issue": "q",
                                        "reason": "r"} for label in texts}, []))
    calls = []

    def yes_no(facts, request, texts):
        calls.append(sorted(texts))
        return {dg._applicability_key(label): {"article": label, **(second or {}).get(label.split()[0], {})}
                for label in texts}
    monkeypatch.setattr(dg, "_applicability_verdicts", yes_no)
    return calls


def test_grade_zero_removes_an_evidenced_article_only_when_a_second_judgement_agrees(monkeypatch, logged):
    rows = [_row("1062", _tier=1), _row("1061", _tier=2), _row("1027", _tier=3), _row("1064", _tier=2)]
    calls = _graded_env(monkeypatch, {"1062": 0, "1061": 0, "1027": 0, "1064": 3},
                        {"1062": {"verdict": "incompatibile", "requisito": "due fondi dello stesso proprietario",
                                  "fatto": "la carrareccia di Bruno"},
                         "1061": {"verdict": "applicabile"}})
    out = dg._filter_applicable_articles(CASE, "", rows, {})
    assert calls == [["1061 c.c.", "1062 c.c."]]                   # one second call, the evidenced zeros only
    kept = {r["_article"][0]: r for r in out}
    assert set(kept) == {"1061", "1064"}                           # 1062: two agree; 1027: T3 as before
    assert kept["1061"]["_demoted"] and kept["1061"]["_applicability"] == "marginale"
    entries = _events(logged, "evidence_removal")[-1]["entries"]
    assert [(e["article"], e["how"], e["judgements"]) for e in entries] == [
        ("1062 c.c.", "removed", 2), ("1061 c.c.", "demoted", 1), ("1027 c.c.", "removed", 1)]
    assert entries[0]["verified_fact"] is True                     # "la carrareccia di Bruno" is in the case


def test_incompatibile_removes_an_evidenced_article_only_with_a_fact_of_the_case(monkeypatch, logged):
    rows = [_row("1158", _tier=2), _row("1159", _tier=2), _row("1160", _tier=3)]
    verdict = {"verdict": "incompatibile", "requisito": "il possesso ventennale"}
    monkeypatch.setattr(dg, "_applicability_verdicts", lambda facts, request, texts: {
        dg._applicability_key("1158 c.c."): {**verdict, "article": "1158 c.c.", "fatto": "Ottavia passa da trent'anni"},
        dg._applicability_key("1159 c.c."): {**verdict, "article": "1159 c.c.", "fatto": "un atto di acquisto trascritto"},
        dg._applicability_key("1160 c.c."): {**verdict, "article": "1160 c.c.", "fatto": "un atto di acquisto trascritto"}})
    out = dg._filter_applicable_articles(CASE, "", rows, {})
    assert [r["_article"][0] for r in out] == ["1159"] and out[0]["_demoted"]
    entries = _events(logged, "evidence_removal")[-1]["entries"]
    assert [(e["article"], e["how"], e["verified_fact"]) for e in entries] == [
        ("1158 c.c.", "removed", True), ("1159 c.c.", "demoted", False), ("1160 c.c.", "removed", False)]


def test_past_the_checked_slots_an_evidenced_article_stays_ungraded(monkeypatch, logged):
    monkeypatch.setattr(dg, "_GRADED_MAX_CHECKED", 2)
    monkeypatch.setenv("CHECK_PRIORITY_ENABLED", "true")           # unchecked articles are removed ...
    _graded_env(monkeypatch, {})
    rows = [_row("2", _tier=3, _fused_rank=1), _row("3", _tier=3, _fused_rank=2), _row("9", _tier=1, _fused_rank=50),
            _row("4", _tier=3, _fused_rank=3), _row("8", _tier=2, _fused_rank=40)]
    out = {r["_article"][0]: r for r in dg._filter_applicable_articles(CASE, "", rows, {})}
    assert set(out) == {"9", "8"}                                   # ... T3 ones past the slots go
    assert out["9"]["_grade"] == 2 and out["8"]["_grade"] == 2       # the slots went to T1 and T2 first
    cut = _events(logged, "evidence_cutoff")[-1]
    assert cut["stage"] == "checked_slots" and [i["article"] for i in cut["cut"]] == ["2 c.c.", "3 c.c.", "4 c.c."]
    assert cut["t12_dropped"] == []


def test_an_evidenced_article_past_the_slots_is_never_removed_as_unchecked(monkeypatch, logged):
    monkeypatch.setattr(dg, "_GRADED_MAX_CHECKED", 1)
    monkeypatch.setenv("CHECK_PRIORITY_ENABLED", "true")
    _graded_env(monkeypatch, {})
    rows = [_row("1", _tier=1, _fused_rank=1), _row("7", _tier=2, _fused_rank=7)]
    out = {r["_article"][0]: r for r in dg._filter_applicable_articles(CASE, "", rows, {})}
    assert out["7"]["_applicability"] == "non valutato"
    cut = _events(logged, "evidence_cutoff")[-1]
    assert cut["kept_aside"] == [{"article": "7 c.c.", "tier": 2}] and cut["t12_dropped"] == []


def test_graded_max_checked_sets_the_slots(monkeypatch, logged):
    monkeypatch.setattr(dg, "_GRADED_MAX_CHECKED", 2)
    monkeypatch.setenv("CHECK_PRIORITY_ENABLED", "true")
    _graded_env(monkeypatch, {})
    rows = [_row("2", _tier=3, _fused_rank=1), _row("3", _tier=3, _fused_rank=2), _row("4", _tier=3, _fused_rank=3)]
    monkeypatch.setenv("GRADED_MAX_CHECKED", "3")                   # one more slot: the third is graded, not removed
    out = {r["_article"][0]: r for r in dg._filter_applicable_articles(CASE, "", rows, {})}
    assert set(out) == {"2", "3", "4"} and out["4"]["_grade"] == 2
    assert _events(logged, "case_analysis_applicability")[-1]["max_checked"] == 3
    for value in ("", "0", "-5", "many"):                            # not a positive number: the default
        monkeypatch.setenv("GRADED_MAX_CHECKED", value)
        assert dg._graded_max_checked() == 2


def test_untiered_rows_behave_as_before(monkeypatch, logged):
    _graded_env(monkeypatch, {"1062": 0})
    out = dg._filter_applicable_articles(CASE, "", [_row("1062"), _row("1061")], {})
    assert [r["_article"][0] for r in out] == ["1061"]
    assert not _events(logged, "evidence_removal") and not _events(logged, "evidence_cutoff")


# --- Rule 3 and Rule 1 at the plan's forced list --------------------------------------------------------------------

def test_the_forced_list_has_a_fixed_order_skips_the_demoted_and_logs_its_cut(monkeypatch, logged):
    monkeypatch.setenv("CHECK_GRADED_ENABLED", "true")
    rows = [_row(str(n), _grade=2, _fused_rank=5, _tier=3) for n in range(30, 10, -1)]
    rows += [_row("7", _grade=2, _fused_rank=5, _tier=1), _row("8", _grade=3, _fused_rank=99, _tier=3),
             _row("9", _grade=2, _fused_rank=1, _tier=2, _demoted="grade 0 by one judgement"),
             _row("6", _grade=1, _tier=1, _central=True)]
    forced = list(ptw._forced(rows))
    assert len(forced) == 18 and ptw._GRADED_KEEP == 18
    assert forced[:4] == [("6", "c.c."), ("8", "c.c."), ("7", "c.c."), ("11", "c.c.")]   # grade, tier, rank, id
    assert ("9", "c.c.") not in forced                                                     # demoted: never forced
    assert forced == list(ptw._forced(list(reversed(rows))))                               # the input order is moot
    cut = _events(logged, "evidence_cutoff")[-1]
    assert cut["stage"] == "plan_forced" and cut["kept_aside"] == [] and len(cut["cut"]) == 23 - 18


# --- Rule 1 at the prompt -------------------------------------------------------------------------------------------

def test_the_prompt_keeps_evidenced_articles_and_refills_by_tier(monkeypatch, logged):
    rows = [_row(str(n), text="x" * 2500, _tier=3, _fused_rank=n) for n in range(1, 13)]
    rows += [_row("50", text="y" * 2500, _tier=1, _fused_rank=50), _row("51", text="z" * 2500, _tier=2, _fused_rank=51)]
    articles, _, _ = dg._fit_parere_prompt("SYS", CASE, {"questioni": []}, rows, [], "", "", budget=2600)
    assert "y" * 150 in articles and "z" * 150 in articles          # T1 and T2 kept, at least their opening
    cut = _events(logged, "evidence_cutoff")[-1]
    assert cut["stage"] == "prompt" and cut["t12_dropped"] == [] and cut["cut"]
    assert all(i["tier"] == 3 for i in cut["cut"])                 # the cut reaches T3 only
    order = dg._refill_order([_row("3", _tier=3, _fused_rank=1), _row("2", _tier=2, _fused_rank=9),
                              _row("1", _tier=3, _fused_rank=1, _grade=2)])
    assert [r["_article"][0] for r in order] == ["2", "1", "3"]
    assert dg._protected_article_row(_row("5", _tier=2), set()) and not dg._protected_article_row(_row("5", _tier=3), set())


# --- Rule 2 at the qualification check ------------------------------------------------------------------------------

class _Qualifier:
    def __init__(self, governs=None):
        self.governs = governs

    def __call__(self, messages, max_tokens=None, temperature=None, **kw):
        system = messages[0].content
        if "Decidi se" in system and "testo integrale" in system:          # second opinion
            return json.dumps({"disciplina": self.governs, "perche": "x"})
        labels = [line.strip("[]") for line in messages[-1].content.split("\n") if line.startswith("[")]
        return json.dumps([{"articolo": label, "compatibile": False, "perche": "altro", "tipo": "materia"}
                           for label in labels])


def test_a_single_mismatch_demotes_an_evidenced_citation_and_two_remove_it(monkeypatch, logged):
    monkeypatch.setenv("QUALIFICATION_CHECK_ENABLED", "true")
    monkeypatch.setenv("FLAGGED_REMOVE_ENABLED", "true")
    rows = [_row("1064", heading="Estensione del diritto di servitù", _tier=2),
            _row("1027", heading="Contenuto del diritto", _tier=3)]
    draft = "Bruno deve aprire la sbarra (art. 1064 c.c.). La servitù è un peso (art. 1027 c.c.)."
    monkeypatch.setattr(dg, "_call_chat", _Qualifier())
    out = dg._check_qualifications(draft, {}, rows, None, [], CASE)
    assert "1064" in out and "1027" not in out
    entries = _events(logged, "evidence_removal")[-1]["entries"]
    assert [(e["article"], e["how"], e["judgements"]) for e in entries] == [
        ("1064 c.c.", "demoted", 1), ("1027 c.c.", "removed", 1)]
    monkeypatch.setenv("STRICT_ARTICLES_ENABLED", "true")
    monkeypatch.setattr(dg, "_call_chat", _Qualifier(governs=False))
    out = dg._check_qualifications(draft, {}, rows, None, [], CASE)
    assert "1064" not in out
    assert ("1064 c.c.", "removed", 2) in [(e["article"], e["how"], e["judgements"])
                                           for e in _events(logged, "evidence_removal")[-1]["entries"]]


# --- the invariants respect the tiers -------------------------------------------------------------------------------

def test_invariants_treat_evidenced_articles_as_checked_and_protected():
    rows = [_row("1062", _grade=0, _tier=1, _demoted="x"), _row("1027", _grade=1, _tier=3)]
    assert ai.allowed_refs(rows, [], {}, []) == [("1062", "c.c.")]


# --- Rule 3: the plan and the grading at temperature 0 ---------------------------------------------------------------

def test_grading_and_plan_calls_at_temperature_zero(monkeypatch):
    seen = []
    monkeypatch.setattr(dg, "_chat", lambda messages, max_tokens=None, **kw: seen.append(kw.get("temperature")) or "[]")
    dg._graded_batch("Fatti.", {"1062 c.c.": "Testo."})
    monkeypatch.setattr(dg, "_chat_json", lambda messages, max_tokens=None, **kw: seen.append(kw.get("temperature"))
                        or {"questioni": []})
    ptw.make_plan("Fatti.", "", {}, "NORME", "", "", [], {})
    assert seen and set(seen) == {0}
