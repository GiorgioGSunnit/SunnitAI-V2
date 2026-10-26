"""scripts/diagnose_long.py's analysis and report, offline: from a synthetic capture, no pipeline."""

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("diagnose_long", HERE.parent / "scripts" / "diagnose_long.py")
dl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dl)
_ev_spec = importlib.util.spec_from_file_location("eval_narrative", HERE / "eval_narrative.py")
ev = importlib.util.module_from_spec(_ev_spec)
_ev_spec.loader.exec_module(ev)


def _alt(article, code, **kw):
    return {"article": article, "code": code, "rubrica": kw.get("rubrica", "R"), "field": "titolo_articolo",
            "in_corpus": True, "resolved_by_institution": kw.get("resolved", []), "named_by": kw.get("named", []),
            "area_allows": kw.get("allows", True)}


def _record():
    expected = [{"code": "cp", "article": "584"}, {"code": "cp", "article": "43"}, {"code": "cp", "article": "56"},
                {"code": "cp", "article": "61"}, {"code": "cc", "article": "2043"},
                {"code": "cp", "any_of": ["582", "583"]}]
    pool = [{"label": l, "sources": s, "rubrica": r} for l, s, r in (
        ("584 c.p.", ["case_analysis"], "Omicidio preterintenzionale"), ("43 c.p.", [], "Elemento psicologico"),
        ("2043 c.c.", [], "Risarcimento per fatto illecito"), ("81 c.p.", [], "Concorso formale"),
        ("582 c.p.", ["crossref"], "Lesione personale"))]
    return {
        "id": "x1", "kind": "eval", "status": "ok", "seconds": 120.0, "expected": expected, "draft": "...",
        "frame": {"areas": ["penale"], "posture_type": "indagini_preliminari", "tasks": ["quale reato"],
                  "central_issue_candidates": [{"nome": "omicidio preterintenzionale", "codice": "c.p."}]},
        "institutions": [{"phase": "v2", "nome": "omicidio preterintenzionale", "codice": "c.p.", "method": "heading",
                          "reason": "titolo_articolo", "parole_chiave": [],
                          "articles": [{"label": "584 c.p.", "rubrica": "Omicidio preterintenzionale",
                                        "field": "titolo_articolo", "fits_name": True, "fits_keyword": False,
                                        "generic": False}]},
                         {"phase": "v2", "nome": "colpa", "codice": "c.p.", "method": "fulltext", "reason": "fulltext",
                          "parole_chiave": [], "articles": [{"label": "81 c.p.", "rubrica": "Concorso formale",
                                                             "field": "titolo_articolo", "fits_name": False,
                                                             "fits_keyword": False, "generic": False}]}],
        "v1": {"keep": [], "named": [], "chosen": ["2043 c.c."], "selected": ["2043 c.c."], "spotted": []},
        "v1_sources": {"code_search": ["2043 c.c."]},
        "v2": {"selected": ["2043 c.c.", "584 c.p.", "81 c.p."], "heading_matched": ["584 c.p."]},
        "fusion": {"fused": ["584 c.p.", "43 c.p.", "2043 c.c.", "81 c.p.", "56 c.p."], "keep": [], "named": [],
                   "selected": ["584 c.p.", "43 c.p.", "2043 c.c.", "81 c.p."],
                   "exempt_from_cap": {"584 c.p.": "multi-signal"}, "cap": 2,
                   "fill_order": ["43 c.p.", "2043 c.c.", "81 c.p.", "56 c.p."],
                   "hits": {"584 c.p.": 2}, "signals": {"current": ["584 c.p."], "baseline": ["584 c.p."]}},
        "searches": [{"kind": "baseline", "channel": "vector", "refs": ["584 c.p.", "81 c.p.", "56 c.p."]},
                     {"kind": "distill", "channel": "fulltext", "refs": ["43 c.p."]}],
        "vote": {"cut": {"tagged": [], "added": []}, "voted": [], "info": {"rulings": 60}},
        "pool": pool, "method_candidates": [], "crossref_source": "crossref",
        "check": {"input": [p["label"] for p in pool], "exempt": {"584 c.p.": "heading", "582 c.p.": "crossref"},
                  "no_text": [], "unchecked": [], "output": {}},
        "check_result": {"verdicts": {"43 c.p.": "incompatibile", "2043 c.c.": "applicabile", "81 c.p.": "applicabile"},
                         "dropped": [{"article": "43 c.p.", "requisito": "dolo", "fatto": "colpa"}]},
        "prompt": {"labels": ["584 c.p.", "2043 c.c.", "81 c.p.", "582 c.p."]},
        "plan": {"issues": 2, "norme": ["584 c.p.", "81 c.p.", "582 c.p."],
                 "checked": {"kept_as_applicable": ["81 c.p. (fuori tema)"]}},
        "answer": {"mentioned": ["584 c.p.", "81 c.p.", "582 c.p."],
                   "mentioned_eval": [["584", "cp"], ["81", "cp"], ["582", "cp"]]},
        "fits": {"81 c.p.": {"rubrica": "Concorso formale", "compatibile": False, "perche": "non c'è concorso"}},
        "feature_flags_seen": {"fusion": True, "check": True, "plan": True},
        "expected_info": [{"alts": [_alt("584", "cp", resolved=["omicidio preterintenzionale (heading)"])]},
                          {"alts": [_alt("43", "cp")]}, {"alts": [_alt("56", "cp")]},
                          {"alts": [_alt("61", "cp")]}, {"alts": [_alt("2043", "cc", allows=False)]},
                          {"alts": [_alt("582", "cp"), _alt("583", "cp")]}],
    }


def test_stages_channels_and_wrong_articles():
    case = dl.analyse_case(_record(), ev)
    stages = {s["label"]: s["stage"] for s in case["slots"]}
    assert stages == {"art. 584 cp": "MENTIONED", "art. 43 cp": "REJECTED_BY_CHECK", "art. 56 cp": "BELOW_CAP",
                      "art. 61 cp": "LOST_AT_UNDERSTANDING", "art. 2043 cc": "NOT_IN_PLAN",
                      "art. 582/583 cp": "MENTIONED"}
    first = case["slots"][0]
    assert first["channels"] == ["institution", "vector"] and first["check"] == "exempt: heading"
    assert case["slots"][2]["cap"] == "fill #4 (cut, cap 2)" and case["slots"][2]["rank"] == 5
    assert case["slots"][4]["channels"] == ["v1_choice"] and case["slots"][4]["details"] == ["v1:code_search"]
    assert case["slots"][5]["channels"] == ["crossref"]          # any_of: its best alternative
    [wrong] = case["unexpected"]
    assert wrong["label"] == "81 c.p." and wrong["origin"] == ["institution", "vector"]
    assert wrong["check"] == "applicabile" and wrong["plan_notes"] == ["kept_as_applicable"]
    assert wrong["fit"]["compatibile"] is False
    assert {t["label"]: t["check"] for t in case["pool_traces"]}["43 c.p."] == "rejected"


def test_report_sections():
    case = dl.analyse_case(_record(), ev)
    unkeyed = dl.analyse_case({**_record(), "id": "leonardo", "kind": "file", "expected": None,
                               "expected_info": None}, ev)
    assert len(unkeyed["unexpected"]) == 3 and not unkeyed["slots"]
    text = dl.render([case, unkeyed], ["Flags: x"])
    assert "| MENTIONED | 2 | 33% |" in text and "| LOST_AT_UNDERSTANDING | 1 | 17% |" in text
    assert "1 of 6 slots with a main code have a code the frame's areas do not allow" in text
    assert "| 81 c.p. * | 1 (x1) | institution: 1, vector: 1 | applicabile: 1 |" in text
    assert "rubrica sharing no word with the institution or its keywords 1 (50%)" in text
    assert "## leonardo (no answer key)" in text
    assert "| penale | 1 | 6 | 2 (33%) |" in text




def test_llm_stage_names_the_calling_step():
    import sys as _sys

    def _applicability_batch():
        return dl._llm_stage(_sys._getframe(0))

    def _opposing_arguments():
        return dl._llm_stage(_sys._getframe(0))

    def make_plan():
        return _opposing_arguments()

    def helper():
        return dl._llm_stage(_sys._getframe(0))
    assert _applicability_batch() == "applicability"
    assert _opposing_arguments() == "stance" and make_plan() == "plan_stance"
    assert helper() == "other_helper"


def test_dump_llm_saves_each_call(tmp_path=None):
    import sys as _sys
    import tempfile
    import types
    tmp = str(tmp_path or tempfile.mkdtemp())

    class Msg:
        def __init__(self, kind, content):
            self.type, self.content = kind, content

    def real(messages, max_tokens=None, stop=None, temperature=None):
        return "[]"
    names = ("src", "src.rag", "src.rag.ai_chat", "src.rag.fake_user")
    saved = {n: _sys.modules.get(n) for n in names}
    ai_chat = types.ModuleType("src.rag.ai_chat")
    ai_chat._call_chat = real
    user = types.ModuleType("src.rag.fake_user")
    user._call_chat = real
    rag = types.ModuleType("src.rag")
    rag.ai_chat = ai_chat
    _sys.modules.update({"src": types.ModuleType("src"), "src.rag": rag, "src.rag.ai_chat": ai_chat,
                         "src.rag.fake_user": user})
    dg = types.SimpleNamespace(_JSON_RETRY_SUFFIX=" SOLO JSON", _estimate_tokens=lambda t: len(t) // 3 + 1)
    try:
        dl._dump_llm(dg, tmp, 0.0)

        def _case_frame():
            return user._call_chat([Msg("system", "SYS"), Msg("human", "FATTI " * 5)], max_tokens=1500, temperature=0)

        def _fusion_distill_bullets():
            return user._call_chat([Msg("system", "S SOLO JSON"), Msg("human", "x")], 700)
        assert _case_frame() == "[]" and _fusion_distill_bullets() == "[]"
    finally:
        for n, m in saved.items():
            if m is None:
                _sys.modules.pop(n, None)
            else:
                _sys.modules[n] = m
    import os as _os
    assert sorted(_os.listdir(tmp)) == ["01_frame.txt", "02_distill_json_retry.txt"]
    text = open(_os.path.join(tmp, "01_frame.txt"), encoding="utf-8").read()
    assert "max_tokens: 1500" in text and "temperature: 0\n" in text and "===== SYSTEM =====\nSYS" in text
    assert "===== USER =====\nFATTI FATTI" in text and text.endswith("===== REPLY =====\n[]\n")
    assert "max_tokens: 700" in open(_os.path.join(tmp, "02_distill_json_retry.txt"), encoding="utf-8").read()


def test_material_puts_provenance_under_each_article():
    import os as _os
    import tempfile
    path = _os.path.join(tempfile.mkdtemp(), "material.txt")
    record = {**_record(), "prompt": {"labels": ["584 c.p.", "81 c.p."],
                                      "tokens": {"total": 9000, "budget": 13000, "step": 2,
                                                 "articles_left_out": ["2043 c.c."]}}}
    dl._write_material(path, record, ("### Questione 1\n[art. 584 c.p.]\nChiunque...\n\n[art. 81 c.p.]\nÈ punito...",
                                      "[Cass. pen. 123/2020]\nmassima"), ev)
    text = open(path, encoding="utf-8").read()
    assert "[art. 584 c.p.]\n>> provenance: institution+vector" in text and "| EXPECTED" in text
    assert "[art. 81 c.p.]\n>> provenance: institution+vector" in text and "kept_as_applicable" in text
    assert "Chiunque..." in text and "[Cass. pen. 123/2020]\nmassima" in text
    assert "LEFT OUT OF THE PROMPT BY ITS TRIMMING (1)" in text and "43 c.p.: requisito «dolo»" in text


def test_compare_puts_tables_a_and_b_side_by_side():
    import copy
    import json as _json
    import os as _os
    import tempfile
    old_rec = _record()
    new_rec = copy.deepcopy(old_rec)
    new_rec["check_result"]["dropped"] = []                       # 43 c.p. no longer rejected...
    new_rec["answer"]["mentioned"] = ["584 c.p.", "43 c.p.", "582 c.p."]   # ...and now cited; 81 c.p. gone
    new_rec["answer"]["mentioned_eval"] = [["584", "cp"], ["43", "cp"], ["582", "cp"]]
    new_rec["flags"] = {"CHECK_PRIORITY_ENABLED": "true"}
    new_rec["discipline"] = {"removed": [{"section": "s", "removed": ["1384 c.c.", "56 c.p."], "how": "rephrased"}]}
    new_rec["qualification"] = {"flagged": ["757 c.c.", "56 c.p.", "1054 c.c."], "removed": ["757 c.c.", "56 c.p."],
                                "kept": {"1054 c.c.": "grade 2"}, "note_dropped": ["1054 c.c."]}
    new_rec["flagged_removed"] = {"count": 2}
    new_rec["hygiene"] = {"removed": ["563 Massime"]}
    root = tempfile.mkdtemp()
    dirs = []
    for name, rec in (("20261025-0940", old_rec), ("20261026-1000", new_rec)):
        _os.makedirs(_os.path.join(root, name, "cases"))
        with open(_os.path.join(root, name, "cases", "x1.json"), "w", encoding="utf-8") as fh:
            _json.dump(rec, fh)
        dirs.append(_os.path.join(root, name))
    text = dl.compare(dirs, ev)
    assert "1 cases in every run" in text and "CHECK_PRIORITY_ENABLED None / true" in text
    assert "| REJECTED_BY_CHECK | 1 (17%) | 0 (0%) | -1 |" in text
    assert "| MENTIONED | 2 (33%) | 3 (50%) | +1 |" in text
    assert "| not expected per answer | 1.0 | 0.0 |" in text and "| MENTIONED | 2 / 6 | 3 / 6 |" in text
    assert "| rubrica-incompatible | 1 | 0 |" in text
    assert "| citations taken out of sections | 0 | 2 |" in text and "| expected ones among them | 0 | 1 |" in text
    report = dl.render([dl.analyse_case(new_rec, ev)], [])
    assert "2 citations in 1 sentences: 1 rephrased, 0 dropped" in report and "x1 56 c.p." in report
    assert "| removed after the draft (flagged + not a law) | 0 | 3 |" in text
    assert "| expected ones removed after the draft | 0 | 1 |" in text
    assert "Flagged by the qualification check and removed: 2" in report and "Not a law, removed: 1" in report
    assert "mark dropped as a rubrica-wording mismatch: 1" in report
    assert "| institution | 1 (100%) | 0 (-) | -1 |" in text
    assert "| x1 | 2 | 3 | 6 | 1 | 0 |" in text
    three = dl.compare([dirs[0], dirs[1], dirs[1]], ev)                 # the newest last, two earlier ones
    assert "| stage | 20261025-0940 | 20261026-1000 | 20261026-1000 | change vs 20261025-0940 | change vs 20261026-1000 |" \
        in three and "| MENTIONED | 2 (33%) | 3 (50%) | 3 (50%) | +1 | +0 |" in three


def test_audit_grades_finds_cut_replies_and_echoed_ids():
    import json as _json
    import os as _os
    import tempfile
    rec = _record()
    rec["check_result"] = {"verdicts": {"43 c.p.": "missing", "2043 c.c.": "grade 2", "81 c.p.": "missing"},
                           "dropped": []}
    run = tempfile.mkdtemp()
    llm = _os.path.join(run, "cases", "x1", "llm")
    _os.makedirs(llm)
    with open(_os.path.join(run, "cases", "x1.json"), "w", encoding="utf-8") as fh:
        _json.dump(rec, fh)
    head = "stage: applicability\nmax_tokens: {mt}\nstatus: ok\n\n===== SYSTEM =====\nun voto di pertinenza\n\n"
    calls = [(150, "[2043 c.c.]\nRubrica: R\n\n[43 c.p.]\nRubrica: R",
              '[{"a": "2043 c.c.", "g": 2, "q": "danno", "r": "fatto illecito"}, {"a": "43 c'),
             (105, "[81 c.p.]\nRubrica: R", '[{"a": "art. 81 cp", "g": 1}]')]
    for i, (mt, user, reply) in enumerate(calls, 1):
        with open(_os.path.join(llm, f"0{i}_applicability.txt"), "w", encoding="utf-8") as fh:
            fh.write(head.format(mt=mt) + f"===== USER =====\nFatti\n\n{user}\n\n===== REPLY =====\n{reply}\n")
    text = dl.audit_grades(run, ev)
    assert "| cut | 1 |" in text and "| list | 1 |" in text
    assert "| graded | 1 |" in text and "| id mismatch | 1 |" in text and "| lost: reply cut | 1 |" in text
    assert "| x1 | art. 43 cp | lost: reply cut |" in text
    assert "| x1 | 81 c.p. | art. 81 cp |" in text


def test_fact_signals_reached_and_false_triggers():
    rec = _record()
    rec["fact_signals"] = {"fired": [{"id": "morte_o_lesioni", "area": "penale", "institutions": ["omicidio preterintenzionale"]},
                                     {"id": "caparra", "area": "contratti", "institutions": ["caparra confirmatoria"]}],
                           "terms": [{"text": "querela: termine di 3 mesi DECORSO (minaccia 2026, oggi 2026-10-27)"}]}
    case = dl.analyse_case(rec, ev)
    f = case["fact_signals"]
    assert f["reached"] == [{"rule": "morte_o_lesioni", "article": "art. 584 cp", "stage": "MENTIONED"}]
    assert f["false_triggers"] == [{"rule": "caparra", "area": "contratti", "articles": [], "matched": [],
                                    "other_area": True}]
    report = dl.render([case], [])
    assert "Expected articles reached through a fired rule's institution: 1 (1 of them MENTIONED)" in report
    assert "x1 caparra [other area] → nothing resolved" in report


if __name__ == "__main__":
    for _name, _test in list(globals().items()):
        if _name.startswith("test_"):
            _test()
    print("ok")


def test_form_trace_reads_the_lawyers_review_points_off_the_answer():
    draft = ("**3. Norme e giurisprudenza**\n\nL'art. 2043 c.p. non è pertinente al caso. L'onere della prova ricade "
             "sull'attore. Il termine per la denuncia è di 8 giorni.\n\n**4. Applicazione al caso**\n\nTizio sapeva "
             "del rischio [fatto non indicato nel caso: da verificare]. Si chiede un accertamento (art. 696 c.p.c.)."
             "\n\n**6. Conclusioni**\n\nIl rischio è\n\nVuoi approfondire in particolare [x](deepdive:x)?")
    rec = {**_record(), "draft": draft,
           "fact_signals": {"terms": [{"length": "8 giorni", "status": "DA VERIFICARE"}, {"length": "26 mesi"}],
                            "tools": [{"refs": [["696", "c.p.c."], ["696-bis", "c.p.c."]]}]},
           "fact_regimes": {"qualifications": [{"party": "Anna", "quality": "consumatore"}], "regimes": ["foro"]},
           "writer_rules": {"cut": 1, "non_application": 2, "burden": 1, "facts_flagged": 1,
                            "entries": [{"how": "sentence dropped", "sentence": "L'art. 43 c.p. non si applica."}]}}
    form = dl.analyse_case(rec, ev)["form"]
    assert (form["not_apply"], form["burden"], form["deep_dive"], form["cut"], form["facts_marked"]) == (1, 1, True, 1, 1)
    assert (form["terms"], form["terms_stated"], form["terms_undated"]) == (2, 1, 1)
    assert (form["tools_offered"], form["tools_cited"]) == (2, 1)
    assert form["qualifications"] == ["Anna: consumatore"] and form["expected_dropped"] == ["43 c.p."]
    report = dl.render([dl.analyse_case(rec, ev)], [])
    assert "## Form and writing" in report and "Parties qualified: x1 Anna: consumatore." in report
    rec = {**rec, "qualification": {"inverse": ["2049 c.c."]}, "claimant_frame": {"changed": ["posture text"]},
           "draft": draft.replace("(art. 696 c.p.c.).", "(art. 696 c.p.c.). Si valuti la querela (art. 590 c.p.)."),
           "fact_signals": {**rec["fact_signals"], "tools": [{"id": "profilo_penale", "refs": [["590", "c.p."]]}],
                            "damages": [{"id": "danni_morte"}]},
           "writer_rules": {**rec["writer_rules"], "against_client": 1}}
    form = dl.analyse_case(rec, ev)["form"]
    assert (form["against_client"], form["inverse"], form["posture_fixed"]) == (1, ["2049 c.c."], ["posture text"])
    assert form["criminal_offered"] and form["criminal_cited"] and form["damages_offered"]
    assert "Articles about the inverse situation removed: x1 2049 c.c." in dl.render([dl.analyse_case(rec, ev)], [])
