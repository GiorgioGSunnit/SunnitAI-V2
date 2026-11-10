#!/usr/bin/env python3
"""Issue preservation, measured (read only): for each stored interactive case, every article from round 1's topics to
the final answer, and the step where each one that is not written stops.

Inputs (any mix):
  RUN_DIR            a --interactive-sim run: cases/<id>.json (kind "interactive") and <id>/llm/NN_<stage>.txt dumps
  sessions.json      the chat sessions file (serve_test): anchors["interactive"] and the messages
  transcript.md      an interactive.md transcript (no grades, no dumps: the steps they need are "n/a")

  python scripts/measure_preservation.py /opt/chatbot/diagnose_long_runs/v6*-* --sessions /path/sessions.json --out f.md

Steps per article (each a set of article keys):
  T  in a listed topic graded >= 2 (topic["grades"]; a transcript: the topic line's articles)
  C  a writable article of a chosen topic (articles[:planned])
  P  planned: interactive_flow.capped_refs(chosen topics) -- TOPIC_ARTICLES across them
  G  in the §2 writer's numbered list «NORME NELL'ORDINE DEI PARAGRAFI» (earned_refs: an application is asked)
  R  the writer's raw §2 reply has a paragraph for it (its [N] mark, or a citation of it)
  W  the final §2 block has a paragraph beyond its heading and status lines
  NT listed «Altre norme del tema non trattate»;  NA listed «Norme considerate e non applicabili»
Reason for a C article not in W (the first step it fails): cap (C, not P) / unearned (P, not G: heading and status lines
only) / writer skipped (G, not R) / removed after writing (R, not W) / non si applica (NA) / heading lost (P, no heading,
not NT/NA). Introduced: an article the final answer cites that is not in P, by section.

v60f: a case whose record holds the per-article records (records[label]["answer"]: bucket and reason, from the answer's
ledger) is measured exactly from them -- «exact» in the table; the others are inferred as above («inferred»).
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, OrderedDict

ROOT = os.environ.get("SUNNIT_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

HEADING_RE = re.compile(r"^\*\*Art\.\s+(\S+)\s+(.+?)(?:\s+—\s+(.*?))?\*\*\s*$")
SECTION_RE = re.compile(r"(?m)^\*\*\d+\. ([^*\n]+)\*\*\s*$")
NORMS = "Norme applicate ai fatti"
NT_RE = re.compile(r"(?m)^Altre norme del tema non trattate:\s*(.*)$")
NA_RE = re.compile(r"(?m)^Norme considerate e non applicabili:\s*(.*)$")
ITEM_RE = re.compile(r"art\.\s*(\S+)\s+([^,]+?)(?=,\s*art\.|\.?\s*$)")
NUMBERED_RE = re.compile(r"NORME NELL'ORDINE DEI PARAGRAFI:\s*\n((?:\s*\d+\.\s*art\.[^\n]*\n?)+)")
MARK_RE = re.compile(r"(?m)^\s*\[(\d{1,2})\]")
PREFIXES = {"codice civile": "c.c.", "codice di procedura civile": "c.p.c.", "codice penale": "c.p.",
            "codice di procedura penale": "c.p.p.", "codice del processo amministrativo": "c.p.a.",
            "codice della strada": "c.d.s."}


def code_key(code):
    low = " ".join(str(code or "").lower().split())
    for prefix, abbr in PREFIXES.items():
        if low.startswith(prefix):
            return abbr
    low = re.sub(r"\s+", "", low)
    return {"cc": "c.c.", "cp": "c.p.", "cpc": "c.p.c.", "cpp": "c.p.p.", "cpa": "c.p.a."}.get(low.replace(".", ""), low)


def key(number, code):
    return f"{str(number).strip().lower()} {code_key(code)}"


def label_key(label):
    number, _, code = str(label).partition(" ")
    return key(number, code)


def cited_keys(text):
    """Every article the text cites with its code (the pipeline's own reading when importable)."""
    try:
        from src.rag import citations
        return [key(n, c) for n, c in citations.refs_in(text)]
    except Exception:
        from src.rag import defensive_generation as dg
        return [key(n, c) for n, c in dg._refs_in(text)]


def sections(answer):
    parts = SECTION_RE.split(answer or "")
    return OrderedDict((parts[i].strip(), parts[i + 1]) for i in range(1, len(parts) - 1, 2))


def norm_blocks(norms):
    """{key: gloss text beyond the heading and the status lines}, the headings in order."""
    out, current = OrderedDict(), None
    for line in (norms or "").splitlines():
        h = HEADING_RE.match(line.strip())
        if h:
            current = key(h.group(1), h.group(2))
            out.setdefault(current, "")
            continue
        if NT_RE.match(line) or NA_RE.match(line):
            current = None
            continue
        if current and line.strip() and not line.lstrip().startswith("- "):
            out[current] += line + "\n"
    return out


def listed(regex, text):
    m = regex.search(text or "")
    return {key(n, c) for n, c in ITEM_RE.findall(m.group(1).strip())} if m else set()


def writer_dump(case_dir):
    """(numbered keys, raw reply) of the last §2 writer call in the case's llm dumps, else (None, None)."""
    files = sorted(glob.glob(os.path.join(case_dir, "llm", "*_section_norme_applicate_ai_fatti*.txt")))
    if not files:
        return None, None
    text = open(files[-1], encoding="utf-8").read()
    prompt, _, reply = text.partition("===== REPLY =====")
    m = NUMBERED_RE.search(prompt)
    numbered = []
    if m:
        for line in m.group(1).strip().splitlines():
            lm = re.match(r"\s*\d+\.\s*art\.\s*(\S+)\s+(.+?)\s*$", line)
            if lm:
                numbered.append(key(lm.group(1), lm.group(2)))
    return numbered, reply.split("===== RAW REPLY =====")[0]


def raw_paragraphs(numbered, reply):
    """The keys the raw reply writes a paragraph for: its [N] marks, and the articles it cites."""
    out = set()
    for n in MARK_RE.findall(reply or ""):
        if 0 < int(n) <= len(numbered):
            out.add(numbered[int(n) - 1])
    out |= set(cited_keys(reply or ""))
    return out


def topic_keys(t, graded_only=True):
    grades = t.get("grades") or {}
    if grades:
        return [label_key(a) for a, g in grades.items() if isinstance(g, int) and g >= 2] if graded_only else \
            [label_key(a) for a in grades]
    return [label_key(a) for a in (t.get("articles") or [])[:int(t.get("planned") or len(t.get("articles") or []))]]


def planned_keys(chosen_topics):
    try:
        from src.rag import interactive_flow
        return [key(n, c) for n, c in interactive_flow.capped_refs(chosen_topics)]
    except Exception:
        out = []
        for t in chosen_topics:
            out += [k for k in topic_keys(t, False) if k not in out]
        return out[:8]


REASONS_EXACT = {"scritto": None, "non applicabile": "non si applica", "oltre il limite": "cap"}


def measure_exact(cid, topics, chosen, answer, records, ledger=None, grading=None):
    """From the records' answer buckets (v60f): every planned article's bucket and recorded reason; v60g: the answer's
    ledger (the one check's removals by rule, the steps' proposals, the invariants' counts) and round 1's demotions."""
    chosen_topics = [topics[n - 1] for n in chosen if 0 < n <= len(topics)]
    T = {k for t in topics for k in topic_keys(t)}
    C = []
    for t in chosen_topics:
        C += [label_key(a) for a in (t.get("articles") or [])[:int(t.get("planned") or len(t.get("articles") or []))]
              if label_key(a) not in C]
    answered = {label_key(lab): rec["answer"] for lab, rec in records.items() if (rec or {}).get("answer")}
    P = [k for k, a in answered.items() if a.get("bucket") != "oltre il limite"]
    rows = []
    for k, a in answered.items():
        if a.get("bucket") == "scritto":
            continue
        why = REASONS_EXACT.get(a.get("bucket")) or a.get("reason") or a.get("bucket")
        rows.append((k, why, a.get("where") or a.get("bucket")))
    secs = sections(answer)
    introduced = []
    for title, text in secs.items():
        for k in dict.fromkeys(cited_keys(text)):
            if k not in P and k not in C:
                introduced.append((title, k))
    count = lambda b: sum(1 for a in answered.values() if a.get("bucket") == b)
    led = ledger or {}
    check = {"removed": dict(led.get("removed_by_reason") or {}), "proposed": dict(led.get("proposed_by_step") or {}),
             "unaccounted": len(led.get("unaccounted") or []), "unreasoned": len(led.get("unreasoned") or []),
             "defer": led.get("defer"),
             "demoted": list(((grading or {}).get("distinctive") or {}).get("demoted") or [])} if led or grading else None
    return {"case": cid + " (exact)", "T": len(T), "C": len(C), "P": len(P), "G": None, "R": None,
            "W": count("scritto"), "NT": count("non trattato"), "NA": count("non applicabile"), "not_written": rows,
            "introduced": introduced, "T_not_chosen": sorted(T - set(C)), "check": check}


def measure(cid, topics, chosen, answer, case_dir=None):
    chosen_topics = [topics[n - 1] for n in chosen if 0 < n <= len(topics)]
    T = {k for t in topics for k in topic_keys(t)}
    C = []
    for t in chosen_topics:
        C += [label_key(a) for a in (t.get("articles") or [])[:int(t.get("planned") or len(t.get("articles") or []))]
              if label_key(a) not in C]
    P = planned_keys(chosen_topics)
    numbered, reply = writer_dump(case_dir) if case_dir else (None, None)
    G = set(numbered) if numbered is not None else None
    R = raw_paragraphs(numbered or [], reply) if reply is not None else None
    secs = sections(answer)
    norms = secs.get(NORMS, "")
    blocks = norm_blocks(norms)
    W = {k for k, gloss in blocks.items() if gloss.strip()}
    NT, NA = listed(NT_RE, norms), listed(NA_RE, norms)
    rows = []
    for k in C:
        if k in W:
            continue
        if k in NA:
            why = "non si applica"
        elif k not in P:
            why = "cap"
        elif G is not None and k not in G:
            why = "unearned"
        elif R is not None and k not in R:
            why = "writer skipped"
        elif R is not None and k in R:
            why = "removed after writing"
        elif k not in blocks and k not in NT:
            why = "heading lost"
        else:
            why = "not written (no dump: writer or check)"
        where = "non trattate" if k in NT else "status lines only" if k in blocks else "absent"
        rows.append((k, why, where))
    introduced = []
    for title, text in secs.items():
        for k in dict.fromkeys(cited_keys(text)):
            if k not in P and k not in C:
                introduced.append((title, k))
    return {"case": cid, "T": len(T), "C": len(C), "P": len(P), "G": None if G is None else len(G & set(P)),
            "R": None if R is None else len(R & set(P)), "W": len(W & set(C)), "NT": len(NT & set(C)),
            "NA": len(NA & set(C)), "not_written": rows, "introduced": introduced,
            "T_not_chosen": sorted(T - set(C))}


def from_run(run_dir):
    for path in sorted(glob.glob(os.path.join(run_dir, "cases", "*.json"))):
        try:
            r = json.load(open(path, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if r.get("kind") != "interactive":
            continue
        chosen = r.get("chosen") or (r.get("oracle") or {}).get("chosen") or [1]
        cid = f"{os.path.basename(run_dir.rstrip('/'))}/{r['id']}"
        if any((rec or {}).get("answer") for rec in (r.get("records") or {}).values()):
            yield measure_exact(cid, r.get("topics") or [], chosen, r.get("answer") or "", r["records"],
                                r.get("ledger"), r.get("grading"))
        else:
            yield measure(cid, r.get("topics") or [], chosen, r.get("answer") or "", os.path.join(run_dir, r["id"]))


def from_sessions(path):
    data = json.load(open(path, encoding="utf-8"))
    sessions = data.get("sessions") or data
    for sid, s in (sessions.items() if isinstance(sessions, dict) else enumerate(sessions)):
        state = ((s or {}).get("anchors") or {}).get("interactive") or {}
        if not state.get("topics"):
            continue
        answers = [m.get("content") or "" for m in s.get("messages") or [] if m.get("role") == "assistant"
                   and SECTION_RE.search(m.get("content") or "")]
        if not answers:
            continue
        cid = f"session/{str(sid)[:8]}"
        if any((rec or {}).get("answer") for rec in (state.get("records") or {}).values()):
            yield measure_exact(cid, state["topics"], state.get("chosen") or [1], answers[-1], state["records"],
                                state.get("ledger"), state.get("grading"))
        else:
            yield measure(cid, state["topics"], state.get("chosen") or [1], answers[-1])


def from_transcript(path):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import answer_lint as L
    for cid, case in L.read_cases(path).items():
        listed_topics = L.topics(case["round1"])
        topics = [{"title": t["title"], "articles": [f"{n} {c}" for n, c in t["refs"]],
                   "anchor": (f"{t['refs'][0][0]} {t['refs'][0][1]}" if t["refs"] else "")} for t in listed_topics]
        chosen = [t["n"] for t in L.chosen(case)] or [1]
        yield measure(f"{os.path.basename(path)}/{cid}", topics, chosen, case["answer"])


def report(results):
    lines = ["# Issue preservation, round 1 -> answer", "",
             "T: graded >= 2 in the listed topics; C: writable articles of the chosen topics; P: planned (cap); G: in the "
             "§2 writer's list; R: raw §2 paragraph; W: final §2 paragraph; NT: «non trattate»; NA: «non applicabili»; "
             "intro: cited in the answer, not planned.", "",
             "| case | T | C | P | G | R | W | NT | NA | not written | intro |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    reasons, intro_by = Counter(), Counter()
    for r in results:
        na = lambda v: "n/a" if v is None else v
        lines.append(f"| {r['case']} | {r['T']} | {r['C']} | {r['P']} | {na(r['G'])} | {na(r['R'])} | {r['W']} | "
                     f"{r['NT']} | {r['NA']} | {len(r['not_written'])} | {len(r['introduced'])} |")
        reasons.update(why for _, why, _ in r["not_written"])
        intro_by.update(t for t, _ in r["introduced"])
    lines += ["", "## Reasons, all cases", "", "| reason | articles |", "|---|---|"]
    lines += [f"| {why} | {n} |" for why, n in reasons.most_common()]
    lines += ["", "## Introduced in the answer (not planned), by section", "", "| section | citations |", "|---|---|"]
    lines += [f"| {t} | {n} |" for t, n in intro_by.most_common()]
    checked = [r for r in results if r.get("check")]
    if checked:             # v60g: the one check against the record, and round 1's demotions
        lines += ["", "## The check against the record (v60g)", "",
                  "| case | removed, by rule | proposed, by step | unaccounted | unreasoned | demoted in round 1 |",
                  "|---|---|---|---|---|---|"]
        fmt = lambda d: "; ".join(f"{k} {v}" for k, v in sorted(d.items(), key=lambda x: -x[1])) or "—"
        for r in checked:
            c = r["check"]
            lines.append(f"| {r['case']} | {fmt(c['removed'])} | {fmt(c['proposed'])} | {c['unaccounted']} | "
                         f"{c['unreasoned']} | {', '.join(c['demoted']) or '—'} |")
    lines += ["", "## Per article", ""]
    for r in results:
        if r["not_written"] or r["introduced"]:
            lines.append(f"### {r['case']}")
            lines += [f"- {k}: {why} ({where})" for k, why, where in r["not_written"]]
            if r["introduced"]:
                lines.append("- introduced: " + "; ".join(f"{k} ({t})" for t, k in r["introduced"]))
            lines.append("")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inputs", nargs="+", help="run dirs and/or transcript .md files")
    ap.add_argument("--sessions", action="append", default=[])
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    results = []
    for item in args.inputs:
        if os.path.isdir(item):
            results += list(from_run(item))
        elif item.endswith(".md"):
            results += list(from_transcript(item))
    for path in args.sessions:
        results += list(from_sessions(path))
    text = report(results)
    if args.out:
        open(args.out, "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
