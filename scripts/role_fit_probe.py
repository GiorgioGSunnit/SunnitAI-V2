"""ROLE_FIT on a diagnose_long case dump, with the real model (v48): the plan's substantive articles graded >= 2 asked
whom they work for, from the dump's facts and the article texts its "Norme e giurisprudenza" prompt held. Read-only.

    cd /root/v38 && venv/bin/python scripts/role_fit_probe.py diagnose_long_runs/v47-candidate/cases/h3.json --env .env.v45

Prints one line per article (answer and why) and the adverse ones; h3 (Leonardo): 949 c.c. must come out "controparte".
"""

import argparse
import glob
import json
import os
import re
import sys


def _ref(label: str) -> tuple:
    m = re.match(r"(\d+(?:-\w+)?)\s+(.+)", label)
    return (m.group(1), m.group(2)) if m else (label, "")


def _keep(texts: dict, label: str, rubrica: str, body: str) -> None:
    """The longest text seen for an article, with the first rubrica seen."""
    label, body = " ".join(label.split()), " ".join(re.sub(r"\(co\. \d+\) ", "", body or "").split())
    old_rubrica, old_body = texts.get(label, ("", ""))
    texts[label] = (old_rubrica or rubrica.strip(), body if len(body) > len(old_body) else old_body)


def article_texts(case_dir: str) -> dict:
    """{label: (rubrica, text)} from every place a case dump holds article texts: material.txt (the draft prompt's
    articles), the applicability prompts ("[An] label / Rubrica / Testo"), and the section prompts' TESTI DELLE NORME
    (v48: the Norme prompt alone left art. 949 c.c. without its text on h3)."""
    texts: dict = {}
    material = os.path.join(case_dir, "material.txt")
    if os.path.isfile(material):
        body = open(material, encoding="utf-8").read()
        for m in re.finditer(r"^\[art\. ([^\]]+)\]\n(?:>>[^\n]*\n)?(.*?)(?=\n\n|\Z)", body, re.M | re.S):
            _keep(texts, m.group(1), "", m.group(2))
    for path in sorted(glob.glob(os.path.join(case_dir, "llm", "*_applicability*.txt"))):
        body = open(path, encoding="utf-8").read()
        for m in re.finditer(r"^\[A\d+\] ([^\n]+)\nRubrica: ([^\n]*)\n(?:[^\n]*\n)*?Testo: ([^\n]*)", body, re.M):
            _keep(texts, m.group(1), "" if m.group(2).strip() == "-" else m.group(2), m.group(3))
    for path in sorted(glob.glob(os.path.join(case_dir, "llm", "*_section_*.txt"))):
        body = open(path, encoding="utf-8").read()
        for m in re.finditer(r"\[art\. ([^\]]+)\]\nRubrica: ([^\n]*)\n(.*?)(?=\n\n\[art\.|\n\n[A-Z ]{4,}:\n|\Z)",
                             body, re.S):
            _keep(texts, m.group(1), m.group(2), m.group(3))
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("case", help="a case dump: <run>/cases/<id>.json")
    parser.add_argument("--env", default=os.path.join(os.getcwd(), ".env"))
    args = parser.parse_args()
    sys.path.insert(0, os.getcwd())
    from dotenv import load_dotenv
    load_dotenv(args.env)
    from src.rag import role_fit
    rec = json.load(open(args.case, encoding="utf-8"))
    events = rec.get("events") or []
    final = [d for e, d, *_ in events if e == "case_analysis_plan_final"][-1]
    plan = {"parte_assistita": "", "questioni": [{"titolo": q["titolo"], "norme": [_ref(n) for n in q["norme"]]}
                                                for q in final.get("issues") or []]}
    verdicts = (rec.get("check_result") or {}).get("verdicts") or {}
    texts = article_texts(os.path.splitext(args.case)[0])
    rows = []
    for q in plan["questioni"]:
        for ref in q["norme"]:
            label = f"{ref[0]} {ref[1]}"
            grade = re.match(r"grade (\d)", str(verdicts.get(label) or ""))
            rubrica, body = texts.get(label, ("", ""))
            rows.append({"_article": ref, "_grade": int(grade.group(1)) if grade else None,
                         "s": {"plain_text": body, "titolo_articolo": rubrica}})
    spec = os.path.join(os.path.splitext(args.case)[0], "input.json")          # v49: cases/<id>/input.json
    if not os.path.isfile(spec):
        spec = os.path.splitext(args.case)[0] + ".input.json"                  # runs before v49
    case = json.load(open(spec, encoding="utf-8")).get("question", "") if os.path.isfile(spec) else ""
    out, log = role_fit.check(plan, rows, rec.get("request") or case, rec.get("fact_signals"))
    for e in log:
        print(f"{e['article']:<40} {e['answer']:<12} {e['why']}")
    print("adverse:", ", ".join(out.get("norme_contrarie") or []) or "none")


if __name__ == "__main__":
    main()
