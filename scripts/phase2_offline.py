"""INTERACTIVE_FLOW, offline: round 1 replayed on a diagnose_long run's records (no database, no model call).

Per case: the topics (topic_step, as scripts/phase1_offline.py rebuilds them from the record's pool and article_links'
CSVs), the questions (interactive_flow.questions, from the record's extracted facts), the simulated user's choice
(diagnose_long.oracle_topics) and how many expected norms the chosen topics' anchor articles hold -- the most the topic
answer can name from its own set. The round-1 text of the --show cases. Written to <run>/phase2_offline.md.

    python scripts/phase2_offline.py ~/SunnitAI-V2/diagnose_reports/v52-candidate [--show h3,p1]
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


_DL: dict = {}


def case_text(record: dict) -> str:
    """The eval case's text (tests/eval_narrative_cases.json "question"), else the frame's facts."""
    import phase1_offline as p1
    if "dl" not in _DL:
        _DL["dl"] = p1._dl()
    text = str((_DL["dl"]._eval_case(record.get("id")) or {}).get("question") or "")
    return text or " ".join(map(str, (record.get("frame") or {}).get("facts") or []))


def topics_of(record: dict, links) -> tuple:
    """(topics, others) as round 1 builds them from the record: phase1_offline.replay_c's pool and resolutions, the run's
    action gate (its events) for the titles. v55: the fired rules marked unstated first (interactive_flow.mark_unstated,
    on the eval case's text)."""
    from src.rag import interactive_flow as flow, topic_step
    fused = (record.get("fusion") or {}).get("fused") or []
    rank = {lab: i for i, lab in enumerate(fused, 1)}
    rows = []
    for p in record.get("pool") or []:
        number, _, code = p["label"].partition(" ")
        rows.append({"_article": (number, code), "_multi_signal": bool(p.get("multi_signal")),
                     "_regime_candidate": "regime_candidate" in (p.get("sources") or []),
                     "_fused_rank": rank.get(p["label"]), "s": {"titolo_articolo": p.get("rubrica") or ""}})
    gate = {d["article"]: d["reason"] for e, d, _ in record.get("events") or []
            if e == "case_analysis_action_gate" and isinstance(d, dict) and not d.get("fits")}
    analysis = {"frame": record.get("frame") or {}, "questioni": record.get("questioni") or [],
                "fact_signals": record.get("fact_signals") or {}, "area": record.get("area"), "action_gate": gate,
                "institution_refs": {}}
    for inst in record.get("institutions") or []:
        if isinstance(inst, dict) and inst.get("nome"):
            analysis["institution_refs"].setdefault(inst["nome"], []).extend(a["label"] for a in inst.get("articles") or [])
    others = []
    analysis[flow.CASE_KEY] = case_text(record)
    flow.mark_unstated(analysis)
    frame_text = " ".join(map(str, (record.get("frame") or {}).get("facts") or []))
    return topic_step.build(analysis, rows, links, frame_text, others=others), others, gate


def main() -> None:
    import phase1_offline as p1
    from src.rag import interactive_flow as flow

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run")
    parser.add_argument("--show", default="h3,p1", help="cases whose round-1 text is printed")
    parser.add_argument("--links", help="article_links' output folder (default: <run>/../article_links_out)")
    parser.add_argument("--out")
    args = parser.parse_args()
    run = os.path.abspath(args.run)
    cases_dir = os.path.join(run, "cases")
    links_dir = args.links or os.path.join(os.path.dirname(run), "article_links_out")
    links = p1.links_for(links_dir, os.path.dirname(run)) if os.path.isfile(os.path.join(links_dir, "units.csv")) else None
    dl = p1._dl()
    ev = dl._eval_module()
    shows = set(args.show.split(","))
    rows = ["| case | topics | altri profili | questions | asked (by what they change) | oracle | anchors hold | "
            "recall (top 3) |", "|---|---|---|---|---|---|---|---|"]
    detail = []
    total = {"cases": 0, "q": 0, "hold": 0, "exp": 0, "rec": 0, "rec3": 0}
    for name in sorted(os.listdir(cases_dir)):
        if not name.endswith(".json"):
            continue
        record = json.load(open(os.path.join(cases_dir, name), encoding="utf-8"))
        if record.get("kind") not in ("eval", None):
            continue
        cid = record["id"]
        p1._flags(record, p1._dumps(os.path.join(cases_dir, cid)))
        os.environ.setdefault("FACT_SIGNALS_TODAY", "2026-11-08")
        try:
            topics, others, gate = topics_of(record, links)
            analysis = flow._restored_analysis(json.loads(json.dumps(
                {"frame": record.get("frame") or {}, "fact_signals": record.get("fact_signals") or {},
                 "parte_assistita": "", "questioni": record.get("questioni") or [], "action_gate": gate,
                 flow.CASE_KEY: case_text(record)}, default=str)))
            questions = flow.questions(analysis, (), topics, [])
        except Exception as exc:
            rows.append(f"| {cid} | error: {type(exc).__name__}: {exc} | | | | | | |")
            continue
        expected = record.get("expected") or []
        oracle = dl.oracle_topics(topics, expected, ev)
        held = {lab for n in oracle["chosen"] for lab in oracle["covered"].get(n) or []}
        recall = dl.topic_recall(topics, expected, ev) if expected else {}
        total["cases"] += 1
        total["q"] += len(questions)
        total["hold"] += len(held)
        total["exp"] += len(expected)
        total["rec"] += recall.get("in_topics") or 0
        total["rec3"] += recall.get("in_top3") or 0
        rows.append(f"| {cid} | {len(topics)} | {len(others)} | {len(questions)} | "
                    f"{'; '.join(f'{q['id']} ({len(q['unlocks'])})' for q in questions) or '-'} | "
                    f"{', '.join(map(str, oracle['chosen'])) or '-'} | {len(held)}/{len(expected)} | "
                    f"{recall.get('in_topics', '-')}/{recall.get('expected', '-')} ({recall.get('in_top3', '-')}) |")
        if cid in shows:
            state = {"schema": flow.SCHEMA, "version": 1, "topics": topics, "others": others, "analysis": analysis,
                     "questions": questions}
            text, _ = flow.round1(state)
            detail += ["", f"## {cid}: round 1", "", "```", text, "```", "",
                       f"Oracle: {oracle['why']}; covered: "
                       + "; ".join(f"{k}: {', '.join(v) or '-'}" for k, v in oracle["covered"].items()), ""]
            detail += [f"- {q['id']} (topics {', '.join(q['topics'])}): " + "; ".join(q["unlocks"]) for q in questions]
            detail += [f"- altro profilo: {t['title']} ({t.get('why')})" for t in others]
    if total["cases"]:
        rows.append(f"| **all** | | | {total['q']} ({total['q'] / total['cases']:.1f}/case) | | | "
                    f"{total['hold']}/{total['exp']} ({total['hold'] / max(1, total['exp']):.3f}) | "
                    f"{total['rec']}/{total['exp']} ({total['rec3']}) |")
    out = [f"# Phase 2 offline: round 1 replayed on {run}", "",
           "No database, no model: topics from the record's pool and article_links' CSVs, questions from the record's "
           "extracted facts. «anchors hold»: the expected norms among the oracle-chosen topics' anchor articles (the most "
           "the topic answer can name from its own set).", ""] + rows + detail
    path = args.out or os.path.join(run, "phase2_offline.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nWritten to {path}")


if __name__ == "__main__":
    main()
