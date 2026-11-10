#!/usr/bin/env python3
"""v60 (F): the lawyer's rubric on final answers -- ten rows, each scored 0-1 by the model on the answer and the case
text, then a table. For unseen cases the PO supplies; the benchmark cases (tests/eval_narrative_cases.json) are marked
«regression only» and left out of the acceptance mean (they shaped the tables: they are no yardstick).

Inputs, either:
  --cases DIR          one case per subfolder DIR/<id>/ with case.txt (the case as the user wrote it, the upload's text
                       included) and answer.md (the final answer); or files DIR/<id>.case.txt + DIR/<id>.answer.md
  --transcript FILE    an interactive transcript (diagnose_long --interactive-sim's interactive.md, or the server's run
                       directory): each case's answer, with --case-texts JSON {id: case text}

  python scripts/rubric_score.py --cases ~/po_cases --out rubric.md
  python scripts/rubric_score.py --transcript diagnose_reports/runs/v60-all/interactive.md --case-texts cases.json

Needs the model (the server's .env): one call per case.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# (key, the table's column, what 1 means -- the model reads it)
ROWS = (
    ("strumento", "strumento/atto", "indica lo strumento o l'atto giudiziario giusto per l'obiettivo del cliente "
                                    "(azione, ricorso, querela, opposizione, difesa…)"),
    ("qualificazione", "qualificazione", "qualifica correttamente i fatti (l'istituto o il reato giusto)"),
    ("procedibilita", "procedibilità e termini", "indica correttamente procedibilità, termini e decadenze rilevanti "
                                                 "(o dice che vanno verificati quando il caso non li permette)"),
    ("norme_fuori_luogo", "norme fuori luogo", "nessuna norma citata fuori luogo (1 = nessuna; 0 = più d'una "
                                               "palesemente estranea)"),
    ("fatti_inventati", "fatti inventati", "nessun fatto che il caso non dice è affermato come vero (1 = nessuno)"),
    ("strategia", "strategia principale/subordinata", "propone una linea principale e una subordinata coerenti"),
    ("controparte", "controparte", "anticipa la linea probabile della controparte e la contrasta"),
    ("prova", "prova", "indica i fatti da provare e le prove da raccogliere"),
    ("contraddizioni", "contraddizioni", "nessuna contraddizione interna (1 = nessuna)"),
    ("ruolo_cliente", "ruolo del cliente", "scrive dal punto di vista della parte giusta (il cliente indicato o, se "
                                           "non indicato, lo dichiara)"),
)
SYSTEM = (
    "Sei un avvocato senior che valuta il parere scritto da un collega su un caso. Per ciascuna voce della griglia dai un "
    "punteggio da 0 a 1 (0, 0.25, 0.5, 0.75 o 1) e una motivazione di una frase. Valuta solo ciò che il parere scrive, "
    "sul caso dato. Rispondi SOLO con un JSON {chiave: {\"score\": numero, \"why\": frase}} con queste chiavi:\n"
    + "\n".join(f"- {key}: 1 = {meaning}" for key, _, meaning in ROWS))
SCORE_TOKENS = 1400


def benchmark_ids() -> set:
    try:
        data = json.loads((ROOT / "tests" / "eval_narrative_cases.json").read_text(encoding="utf-8"))
        return {str(c.get("id")) for c in data.get("cases") or [] if c.get("id")}
    except (OSError, ValueError):
        return set()


def from_dir(folder: str) -> Dict[str, Tuple[str, str]]:
    """{id: (case text, answer)}."""
    out = {}
    base = Path(folder)
    for sub in sorted(p for p in base.iterdir() if p.is_dir()):
        case, answer = sub / "case.txt", sub / "answer.md"
        if case.is_file() and answer.is_file():
            out[sub.name] = (case.read_text(encoding="utf-8"), answer.read_text(encoding="utf-8"))
    for case in sorted(base.glob("*.case.txt")):
        cid = case.name[:-len(".case.txt")]
        answer = base / f"{cid}.answer.md"
        if answer.is_file():
            out[cid] = (case.read_text(encoding="utf-8"), answer.read_text(encoding="utf-8"))
    return out


def from_transcript(path: str, texts: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
    import answer_lint
    out = {}
    for cid, case in answer_lint.read_cases(path).items():
        if case.get("answer"):
            out[cid] = (texts.get(cid, ""), case["answer"])
    return out


def clean(raw: Any) -> Dict[str, Dict[str, Any]]:
    """The model's reply as {key: {"score": 0-1 or None, "why"}} for every row."""
    out = {}
    for key, _, _ in ROWS:
        item = raw.get(key) if isinstance(raw, dict) else None
        score = None
        why = ""
        if isinstance(item, dict):
            try:
                score = min(1.0, max(0.0, float(item.get("score"))))
            except (TypeError, ValueError):
                score = None
            why = " ".join(str(item.get("why") or "").split())[:200]
        out[key] = {"score": score, "why": why}
    return out


def score(case_text: str, answer: str, caller=None) -> Dict[str, Dict[str, Any]]:
    """One model call (caller: a stand-in taking (system, human) and returning the parsed JSON, for tests)."""
    human = f"CASO:\n{case_text.strip() or '(testo del caso non fornito)'}\n\nPARERE:\n{answer.strip()}"
    if caller is None:
        from langchain_core.messages import HumanMessage, SystemMessage
        from src.rag import defensive_generation as dg
        raw = dg._chat_json([SystemMessage(content=SYSTEM), HumanMessage(content=human)], max_tokens=SCORE_TOKENS,
                            temperature=0)
    else:
        raw = caller(SYSTEM, human)
    return clean(raw)


def table(scored: Dict[str, Dict[str, Dict[str, Any]]], bench: set) -> str:
    head = "| case | " + " | ".join(col for _, col, _ in ROWS) + " | total |"
    lines = [head, "|---" * (len(ROWS) + 2) + "|"]
    accepted = []
    for cid, rows in scored.items():
        values = [rows[k]["score"] for k, _, _ in ROWS]
        total = sum(v for v in values if v is not None)
        tag = f"{cid} (benchmark: regression only)" if cid in bench else cid
        lines.append(f"| {tag} | " + " | ".join("—" if v is None else f"{v:g}" for v in values) + f" | {total:g}/10 |")
        if cid not in bench:
            accepted.append(rows)
    if accepted:
        means = []
        for k, _, _ in ROWS:
            got = [r[k]["score"] for r in accepted if r[k]["score"] is not None]
            means.append(f"{sum(got) / len(got):.2f}" if got else "—")
        total = sum(float(m) for m in means if m != "—")
        lines.append(f"| **mean, unseen cases ({len(accepted)})** | " + " | ".join(means) + f" | {total:.2f}/10 |")
    notes = ["", "## Why", ""]
    for cid, rows in scored.items():
        notes.append(f"**{cid}**")
        notes += [f"- {col}: {rows[k]['why'] or '—'}" for k, col, _ in ROWS]
        notes.append("")
    return "\n".join(lines + notes)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cases", help="a folder of cases: <id>/case.txt + <id>/answer.md")
    parser.add_argument("--transcript", help="an interactive transcript (.md, or a run directory holding interactive.md)")
    parser.add_argument("--case-texts", help="JSON {id: case text} for --transcript")
    parser.add_argument("--out", help="write the table here (default: stdout)")
    args = parser.parse_args(argv)
    if args.cases:
        cases = from_dir(os.path.expanduser(args.cases))
    elif args.transcript:
        path = os.path.expanduser(args.transcript)
        path = os.path.join(path, "interactive.md") if os.path.isdir(path) else path
        texts = json.loads(Path(os.path.expanduser(args.case_texts)).read_text(encoding="utf-8")) if args.case_texts else {}
        cases = from_transcript(path, texts)
    else:
        parser.error("--cases or --transcript")
    bench = benchmark_ids()
    if bench & set(cases):
        print(f"note: {len(bench & set(cases))} benchmark case(s) scored for regression only, out of the mean",
              file=sys.stderr)
    scored = {}
    for cid, (text, answer) in cases.items():
        try:
            scored[cid] = score(text, answer)
        except Exception as exc:
            print(f"{cid}: scoring failed: {exc}", file=sys.stderr)
            scored[cid] = clean({})
    out = table(scored, bench)
    if args.out:
        Path(args.out).write_text(out + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(out)


if __name__ == "__main__":
    main()
