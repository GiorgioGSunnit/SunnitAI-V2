"""Read-only audit over SAVED runs: which of four general failure mechanisms explains the answers' defects.

Instead of fixing cases one by one, it reads runs already made -- scripts/diagnose_long.py run folders (with
their LLM dumps), tests/eval_narrative.py run folders (narrative, --direct, --ccnl) and scripts/dryrun_strategy.py
--save records -- and measures, for every answer:

  1 GRANULARITY   for each article cited, the comma the citing sentence relies on (the one it names, else the
                  comma of the article's full text, read read-only from Neo4j, that shares most content words
                  with it) against the text the writer actually received for that article in that call (its
                  block in the section's or the draft's material: whole, its first N characters, which commi).
                  Counted: claims whose supporting comma was NOT in the material ("comma not given"; "beyond the
                  truncation" when the block given is the article's start, cut before that comma), articles cited
                  that the call was not given at all.
  2 PROVENANCE    every sentence stating a legal rule, an institution, a procedural step or a ruling's holding,
                  checked against the material of the writer call that wrote it (article texts, ruling passages,
                  terms / tools blocks): one judge call per batch, temperature 0, verdict supported /
                  not-in-material / contradicted. Reported: the share of rule-sentences written from memory.
  3 UNDERSTANDING for every expected article lost at LOST_AT_UNDERSTANDING or NOT_IN_PLAN (diagnose_long's
                  stages): one judge call -- which fact of the case makes it relevant, its kind of trigger
                  (gravity/aggravation, dates/terms, ownership history, role, amount, procedure stage, other),
                  and whether any extracted fact / issue / institution captured it.
  4 ROLE          the client's role (from the writer prompts, the plan or the analysis; for an eval run without
                  them, read by the judge from the question), and every sentence whose perspective contradicts it
                  (advising the accused to become parte civile, a remedy reserved to the other side, an argument
                  for the opponent): one judge call per batch, plus a few fixed patterns as a second signal.

  3 also locates the loss of a lost article whose trigger WAS captured: the first step it fails after the
                  understanding (article resolved or found -> in the pool -> past the applicability check -> in the
                  draft prompt -> in the plan), and, when it was never resolved, what captured it (an institution
                  resolved elsewhere or to nothing, an issue, only a fact).
  4 drops the judge's flags on a sentence that reports the other side's objection or move as such, or advises an
                  act of the client's own role (counted apart as "not kept"); the fixed patterns skip the former.

A run made with GROUNDED_REVIEW_ENABLED: a sentence its rewrite call wrote is attributed to that call (material: the
section's, with the articles it read), an article it read for a section counts as given to that section, and its
own logged counts are reported.

Defects: (1) comma not given / beyond the truncation; (2) not-in-material or contradicted rule-sentences that (1)
does not already explain (a claim on a comma the writer never had is counted once, as granularity); (3) lost
expected articles whose trigger fact nothing extracted; (4) role-contradicting sentences. The ranked summary
gives each mechanism's share of all defects, and the fix directions of the top two.

What each kind of run gives: a diagnose_long run all four (its llm/NN_section_*.txt and NN_draft.txt dumps are the
writer's exact material); an eval run or a dryrun record (no dumps) only 2, with the answer's returned sources as a
PROXY for the material (marked so in the report), and 4.

Run from the app folder (its .env: the judge is the app's chat model; Neo4j read-only, for the commi):
    cd /opt/chatbot && venv/bin/python scripts/fetch_audit.py diagnose_long_runs/20261027-0812
    ... diagnose_long_runs/20261027-0812 diagnose_long_runs/protected-off diagnose_long_runs/protected-on
    ... eval_runs/20261028-2210-ptw-on  eval_runs/direct-20261028-2300-off  /root/dry/c1.json
    ... --only h3,n2 --batch 12 --parallel 4 --max-examples 10 --out fetch_audit_runs/x
    ... --no-judge        # mechanisms 1 and the role patterns only, no model call
    ... --no-db           # no Neo4j: granularity counts only articles not given and explicit commi
    ... --report-only fetch_audit_runs/x   # report again from x/audit.json

Several runs are audited together (each case is "<run>/<case>", repeats "h3@r2" kept apart). The judge's replies
are cached in <out>/judge_cache.jsonl (by prompt hash), so a second run, or one after a crash, makes no repeated
call. Writes only under --out: audit.json, report.md, judge_cache.jsonl. Nothing else is written; every Neo4j
session is read-only.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

MECHANISMS = ("granularity", "provenance", "understanding", "role")
TITLES = {"granularity": "1 Granularity (the supporting comma vs the text given)",
          "provenance": "2 Provenance (memory vs material)",
          "understanding": "3 Understanding (the fact that should have triggered a lost article)",
          "role": "4 Role orientation"}
LOST_STAGES = ("LOST_AT_UNDERSTANDING", "NOT_IN_PLAN")
TRIGGERS = ("gravita_aggravante", "date_termini", "storia_proprieta", "ruolo_qualita", "importo_valore",
            "fase_procedimento", "altro")
TRIGGER_NAMES = {"gravita_aggravante": "gravity / aggravation", "date_termini": "dates / terms",
                 "storia_proprieta": "ownership / possession history", "ruolo_qualita": "role / quality of a party",
                 "importo_valore": "amount / value", "fase_procedimento": "procedure stage", "altro": "other"}
WRITER_STAGE_RE = re.compile(r"^(?:section_.+|draft|grounded_rewrite)$")
# GROUNDED_REVIEW_ENABLED's rewrite call: its material is what precedes the sentences it was asked to rewrite.
REWRITE_MARK = "\n\nFRASI DA RISCRIVERE:"
MIN_WORDS = 6

# Fix directions, printed for the two mechanisms that explain most defects only.
FIX_DIRECTIONS = {
    "granularity": [
        "Give the writer the comma, not the article's first N characters: build each article's block from its "
        "commi (article_text's \"N.k.0\" Sections) and keep, within the per-article budget, the commi whose words "
        "match the issue the plan places it under (the same overlap this audit uses), instead of a prefix cut.",
        "When a sentence names a comma (\"art. 1384, comma 2\") that is not in its material, fetch that comma "
        "before the section is written (the plan already names the articles): one read, no model call.",
    ],
    "provenance": [
        "A sentence stating a rule must point to a passage of its own material: after each section, the same "
        "judge (temperature 0) on its rule-sentences; a not-in-material one is rewritten from the material once, "
        "else dropped -- as the protected-source post-check does for copies.",
        "Put the material first and the rule \"only what the TESTI say\" next to it in each section prompt, and "
        "give a section only the texts its plan items need (fewer, fuller texts), so the model has no gap to fill.",
    ],
    "understanding": [
        "A fixed trigger grid in the fact extraction (gravity/aggravation, dates and terms, ownership history, "
        "roles and qualities, amounts, procedure stage) filled from the case before the issues are written, each "
        "filled slot searched as a query of its own.",
        "Feed the triggers the lost articles needed (the audit's examples) to the issue checklist as kinds of fact "
        "to look for, never as article numbers.",
    ],
    "role": [
        "Every remedy and procedural step carries the party that can take it; the plan and the sections drop "
        "those not available to the client's role (a role table per posture: accused, offended party, claimant, "
        "defendant).",
        "One check after the sections, against the role: a sentence advising the client what only the other side "
        "can do is rewritten as the other side's move or dropped.",
    ],
}


# =================================================================================================
# Text helpers (pure)
# =================================================================================================

def fold(text: Optional[str]) -> str:
    text = unicodedata.normalize("NFKD", (text or "").lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


_STOP = set("""il lo la i gli le un uno una di a da in con su per tra fra e o ed che chi cui non si se del dello
della dei degli delle al allo alla ai agli alle dal dallo dalla dai dagli dalle nel nello nella nei negli nelle sul
sullo sulla sui sugli sulle come anche ma piu essere sono stato stata viene puo deve sua suo suoi sue loro questo
questa quale quali ogni altro altra tale tali caso articolo art comma codice norma legge ovvero nonche quando""".split())


def stems(text: str) -> set:
    return {w[:6] for w in fold(text).split() if len(w) > 3 and w not in _STOP and not w.isdigit()}


# A full stop splits only before a capital: "art. 16", "n. 190/2026" and "c.c. e" stay whole.
_SPLIT_RE = re.compile(r"((?<=[.!?;])\s+(?=[A-ZÀ-Ü«\"“*(\-])|\n)")


def sentences(text: str) -> List[str]:
    parts = _SPLIT_RE.split(text or "")
    return [p.strip() for p in parts[0::2] if p and p.strip()]


_HEADING_RE = re.compile(r"^\s*(?:#+\s|\*\*\d+\.|\d+\.\s*\*\*|\*\*[^*]{0,80}\*\*\s*:?\s*$)")
_SKIP_RE = re.compile(r"^\s*(?:⚠️|💡|Vuoi approfondire|Fonti\s*:)", re.IGNORECASE)


def answer_sentences(answer: str) -> List[str]:
    """The answer's sentences worth judging: no headings, notes, Fonti lines or fragments under MIN_WORDS words."""
    out = []
    for s in sentences(answer):
        clean = re.sub(r"^[-*•\s]+", "", s).strip()
        if _HEADING_RE.match(s) or _SKIP_RE.match(clean) or len(fold(clean).split()) < MIN_WORDS:
            continue
        out.append(clean)
    return out


def _grams(words: Sequence[str], n: int) -> set:
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def contained(part: str, whole: str, n: int = 5, share: float = 0.5) -> bool:
    """`part`'s n-grams are at least `share` in `whole` (a short part: as a substring)."""
    pw, ww = fold(part).split(), fold(whole)
    if not pw:
        return False
    if len(pw) < n:
        return " ".join(pw) in ww
    grams = _grams(pw, n)
    return len(grams & _grams(ww.split(), n)) / len(grams) >= share


# =================================================================================================
# Citations and material blocks (pure)
# =================================================================================================

CODES = [   # longest first; the abbreviations the strategy writes
    (r"c\.?\s*p\.?\s*c\.?(?!\w)|cod\.\s*proc\.\s*civ\.?|codice di procedura civile", "c.p.c."),
    (r"c\.?\s*p\.?\s*p\.?(?!\w)|cod\.\s*proc\.\s*pen\.?|codice di procedura penale", "c.p.p."),
    (r"c\.?\s*p\.?\s*a\.?(?!\w)|codice del processo amministrativo", "c.p.a."),
    (r"c\.?\s*d\.?\s*s\.?(?!\w)|codice della strada", "c.d.s."),
    (r"c\.?\s*c\.?(?!\w)|cod\.\s*civ\.?|codice civile", "c.c."),
    (r"c\.?\s*p\.?(?!\w)|cod\.\s*pen\.?|codice penale", "c.p."),
    (r"cost\.|costituzione", "Cost."),
]
_NUM = r"(\d+(?:\s*-?\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|nonies|decies))?)"
_CITE_RE = re.compile(
    r"\b(?:artt?\.|articol[oi])\s*" + _NUM + r"(?:\s*,?\s*(?:co\.|comma)\s*(\d+))?\s*,?\s*(?:del(?:la|lo)?\s+)?("
    + "|".join(p for p, _ in CODES) + r")", re.IGNORECASE)
_LABEL_RE = re.compile(r"^\[art\. ([^\]\s]+) ([^\]]+?)\](?: \[FONTE RISERVATA[^\]]*\])?\s*$", re.MULTILINE)


def norm_number(raw: str) -> str:
    """"415 bis" / "415-BIS" -> "415-bis"; "2051" unchanged."""
    return re.sub(r"^(\d+)\s*-?\s*([a-z]+)$", r"\1-\2", raw.strip().lower())


def code_of(text: str) -> Optional[str]:
    for pattern, abbr in CODES:
        if re.fullmatch(pattern, text.strip(), re.IGNORECASE):
            return abbr
    return None


def citations_in(sentence: str, labels: Sequence[Tuple[str, str]] = ()) -> List[Dict[str, Any]]:
    """[{"number", "code", "comma"}] the sentence cites: a code by its abbreviation, or a law by the name a
    material label gives it ("art. 95 Codice degli Appalti 2026")."""
    out, seen = [], set()
    for m in _CITE_RE.finditer(sentence):
        number, code = norm_number(m.group(1)), code_of(m.group(3))
        if code and (number, code) not in seen:
            seen.add((number, code))
            out.append({"number": number, "code": code, "comma": int(m.group(2)) if m.group(2) else None})
    low = sentence.lower()
    for number, code in labels:
        if (number, code) in seen or code_of(code):
            continue
        if re.search(rf"\bart(?:t)?\.?\s*{re.escape(number)}\b[^.;]{{0,12}}{re.escape(code.lower())}", low):
            seen.add((number, code))
            out.append({"number": number, "code": code, "comma": None})
    return out


def material_blocks(material: str) -> Dict[Tuple[str, str], str]:
    """{(number, code): its text} for every "[art. N code]" block of a writer call's material (its Rubrica line
    dropped), as _source_block wrote it."""
    blocks: Dict[Tuple[str, str], str] = {}
    marks = list(_LABEL_RE.finditer(material or ""))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(material)
        body = material[m.end():end]
        body = re.split(r"\n\s*\n(?=[A-ZÀ-Ü ]{4,}:|\[)", body, maxsplit=1)[0]
        body = re.sub(r"^\s*Rubrica:[^\n]*\n?", "", body.lstrip("\n"))
        key = (norm_number(m.group(1)), code_of(m.group(2)) or m.group(2).strip())
        blocks[key] = (blocks.get(key, "") + " " + body.strip()).strip()
    return blocks


def labels_of(material: str) -> List[Tuple[str, str]]:
    return [(norm_number(m.group(1)), code_of(m.group(2)) or m.group(2).strip()) for m in _LABEL_RE.finditer(material or "")]


def split_commi(rows: List[Tuple[str, str]], whole: str = "") -> Tuple[List[str], str]:
    """(the article's commi, how): its "N.k.0" Sections in order, else its text split by numbered paragraphs or
    lines, else by sentences."""
    commi = [t for _, t in sorted(rows, key=lambda x: x[0]) if (t or "").strip()]
    if commi:
        return commi, "sections"
    text = (whole or "").strip()
    if not text:
        return [], "none"
    parts = [p.strip() for p in re.split(r"\n+|(?<=[.;:])\s+(?=\d{1,2}\.\s+[A-ZÀ-Ü])", text) if p.strip()]
    if len(parts) > 1:
        return parts, "paragraphs"
    return sentences(text) or [text], "sentences"


def supporting_comma(sentence: str, commi: Sequence[str], named: Optional[int]) -> Tuple[Optional[int], int]:
    """(index of the comma the sentence relies on, shared content words): the comma it names, else the one
    sharing most content stems (at least 2); None when none aligns."""
    if named and 1 <= named <= len(commi):
        return named - 1, -1
    words = stems(re.sub(_CITE_RE, " ", sentence))
    best, score = None, 1
    for i, comma in enumerate(commi):
        shared = len(words & stems(comma))
        if shared > score:
            best, score = i, shared
    return best, (score if best is not None else 0)


def granularity_claim(sentence: str, cite: Dict[str, Any], block: Optional[str], commi: Sequence[str]) -> Dict[str, Any]:
    """One cited claim's status: in_material, beyond_truncation, comma_not_given, article_not_given, unaligned
    (no comma shares its words) or no_text (the article's commi unknown)."""
    out = {"ref": f"{cite['number']} {cite['code']}", "sentence": sentence, "commi": len(commi),
           "given_chars": len(block or ""), "comma": None}
    if block is None:
        out["status"] = "article_not_given"
        return out
    if not commi:
        out["status"] = "no_text" if not cite.get("comma") else "unaligned"
        return out
    i, _ = supporting_comma(sentence, commi, cite.get("comma"))
    if i is None:
        out["status"] = "unaligned"
        return out
    out["comma"] = i + 1
    if contained(commi[i], block):
        out["status"] = "in_material"
        return out
    # The block is the article's start, cut: was it cut before this comma?
    starts = contained(commi[0], block[: max(200, len(commi[0]) + 50)]) if commi else False
    offset = len(" ".join(commi[:i]))
    out["status"] = "beyond_truncation" if starts and offset >= len(block) * 0.8 else "comma_not_given"
    return out


# =================================================================================================
# Loading saved runs
# =================================================================================================

def parse_dump(path: str) -> Dict[str, Any]:
    """One llm/NN_<stage>.txt: {"n", "stage", "system", "user", "reply"}."""
    text = open(path, encoding="utf-8").read()
    head = text.split("\n\n=====", 1)[0]
    stage = (re.search(r"^stage: (.+)$", head, re.M) or [None, os.path.basename(path)[3:-4]])[1].strip()
    parts = re.split(r"^===== (SYSTEM|USER|ASSISTANT|MESSAGE|REPLY) =====\n", text, flags=re.M)
    out = {"n": int(os.path.basename(path)[:2]) if os.path.basename(path)[:2].isdigit() else 0, "stage": stage,
           "system": "", "user": "", "reply": "", "path": path}
    for kind, body in zip(parts[1::2], parts[2::2]):
        key = {"SYSTEM": "system", "USER": "user", "REPLY": "reply"}.get(kind)
        if key:
            out[key] = (out[key] + "\n\n" + body.strip()).strip()
    return out


def _case_text_from_calls(calls: List[Dict[str, Any]]) -> str:
    for call in calls:
        m = re.search(r"CASO:\n(.+?)(?:\n\n[A-ZÀ-Ü][A-ZÀ-Ü ]{3,}:|\Z)", call["user"], re.S)
        if m:
            return m.group(1).strip()
    first = next((c for c in calls if c["stage"] in ("frame", "case_analysis_v2", "case_analysis_v1")), None)
    return first["user"][:6000] if first else ""


def _role_from_calls(calls: List[Dict[str, Any]]) -> Dict[str, str]:
    for call in calls:
        m = re.search(r"nell'interesse di (.+?)(?: \(([^)]+)\))?, seguendo il PIANO", call["system"])
        if m:
            return {"party": m.group(1).strip(), "position": (m.group(2) or "").strip(), "from": "section prompt"}
    for stage in ("plan", "case_analysis_v2", "case_analysis_structured", "case_analysis_v1"):
        for call in calls:
            if call["stage"].startswith(stage):
                m = re.search(r'"parte_assistita"\s*:\s*"([^"]*)"(?:.*?"posizione"\s*:\s*"([^"]*)")?', call["reply"], re.S)
                if m and m.group(1).strip():
                    return {"party": m.group(1).strip(), "position": (m.group(2) or "").strip(), "from": stage}
    return {}


def _eval_cases() -> Dict[str, Dict[str, Any]]:
    try:
        return {c["id"]: c for c in json.load(open(os.path.join(ROOT, "tests", "eval_narrative_cases.json"),
                                                   encoding="utf-8"))["cases"]}
    except (OSError, ValueError, KeyError):
        return {}


def load_diagnose_run(run_dir: str, only: Optional[set] = None) -> List[Dict[str, Any]]:
    """Each case of a diagnose_long run: its record, its writer calls (llm dumps), its case text and role."""
    cases_dir = os.path.join(run_dir, "cases")
    known = _eval_cases()
    out = []
    for name in sorted(os.listdir(cases_dir)):
        if not name.endswith(".json"):
            continue
        rec = json.load(open(os.path.join(cases_dir, name), encoding="utf-8"))
        base = re.sub(r"@r\d+$", "", str(rec.get("id")))
        if only and rec.get("id") not in only and base not in only:
            continue
        llm_dir = os.path.join(cases_dir, str(rec.get("id")), "llm")
        calls = [parse_dump(os.path.join(llm_dir, f)) for f in sorted(os.listdir(llm_dir))
                 if f.endswith(".txt")] if os.path.isdir(llm_dir) else []
        case_text = (known.get(base) or {}).get("question") or _case_text_from_calls(calls)
        out.append({"run": os.path.basename(os.path.normpath(run_dir)), "id": rec.get("id"), "kind": "diagnose",
                    "answer": rec.get("draft") or "", "record": rec, "calls": calls, "case_text": case_text,
                    "role": _role_from_calls(calls), "proxy": not calls, "citations": None})
    return out


def load_eval_run(run_dir: str, only: Optional[set] = None) -> List[Dict[str, Any]]:
    """Each case file of a tests/eval_narrative.py run (narrative, direct-*, ccnl-*): question, answer, citations."""
    out = []
    for name in sorted(os.listdir(run_dir)):
        if not name.endswith(".json") or name.startswith(("summary", "aggregate")):
            continue
        rec = json.load(open(os.path.join(run_dir, name), encoding="utf-8"))
        if not isinstance(rec, dict) or "answer" not in rec:
            continue
        cid = name[:-5]
        if only and cid not in only and rec.get("id") not in only:
            continue
        out.append({"run": os.path.basename(os.path.normpath(run_dir)), "id": cid, "kind": "eval",
                    "answer": rec.get("answer") or "", "record": rec, "calls": [], "proxy": True,
                    "case_text": rec.get("question") or "", "role": {}, "citations": rec.get("citations") or []})
    return out


def load_dryrun(path: str) -> List[Dict[str, Any]]:
    rec = json.load(open(path, encoding="utf-8"))
    return [{"run": os.path.basename(path), "id": os.path.splitext(os.path.basename(path))[0], "kind": "dryrun",
             "answer": rec.get("draft") or rec.get("answer") or "", "record": rec, "calls": [], "proxy": True,
             "case_text": rec.get("request") or "", "role": {}, "citations": rec.get("citations") or []}]


def load(paths: Sequence[str], only: Optional[set] = None) -> List[Dict[str, Any]]:
    cases = []
    for path in paths:
        if os.path.isfile(path):
            cases += load_dryrun(path)
        elif os.path.isdir(os.path.join(path, "cases")):
            cases += load_diagnose_run(path, only)
        elif os.path.isdir(path):
            cases += load_eval_run(path, only)
        else:
            sys.exit(f"no run at {path!r}")
    for c in cases:
        c["key"] = f"{c['run']}/{c['id']}"
    return cases


# =================================================================================================
# Writer calls: which call wrote each sentence of the final answer, with what material
# =================================================================================================

def section_stage(title: str) -> str:
    """A plan section's dump stage, as diagnose_long._llm_stage names it."""
    return "section_" + re.sub(r"[^a-z0-9]+", "_", str(title).lower()).strip("_")[:40]


def grounded_events(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The run's grounded_review logs (GROUNDED_REVIEW_ENABLED), from the record's events."""
    events = (case.get("record") or {}).get("events") or []
    return [e[1] for e in events if isinstance(e, (list, tuple)) and len(e) >= 2 and e[0] == "grounded_review"
            and isinstance(e[1], dict)]


def fetched_for_sections(case: Dict[str, Any]) -> Dict[str, str]:
    """{section stage: the "[art. N code]" blocks the grounded review read for that section's sentences}: an article
    read for a section counts as given to it."""
    out: Dict[str, List[str]] = defaultdict(list)
    for log in grounded_events(case):
        blocks = log.get("fetched") or {}
        for entry in log.get("entries") or []:
            for ref in entry.get("fetched") or []:
                block = blocks.get(ref)
                stage = section_stage(entry.get("section") or "")
                if block and block not in out[stage]:
                    out[stage].append(block)
    return {stage: "\n\n".join(blocks) for stage, blocks in out.items()}


def writer_calls(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    calls = [c for c in case["calls"] if WRITER_STAGE_RE.match(c["stage"])]
    if calls or not case["proxy"]:
        fetched = fetched_for_sections(case)
        out = []
        for c in calls:
            material = f"{c['system']}\n\n{c['user']}"
            if c["stage"] == "grounded_rewrite":
                material = material.split(REWRITE_MARK, 1)[0]
            elif fetched.get(c["stage"]):
                material = f"{fetched[c['stage']]}\n\n{material}"
            out.append({"stage": c["stage"], "material": material, "reply": c["reply"]})
        return out
    # No dumps: the answer's returned sources stand in for the material (a proxy).
    lines = []
    for _, cit in as_items(case.get("citations")):
        if not isinstance(cit, dict):
            continue
        for _, s in as_items(cit.get("sections")):
            if not isinstance(s, dict):
                continue
            text = s.get("plain_text") or ""
            if not isinstance(text, str):
                text = _plain(text)
            number = re.match(r"^(\d+(?:-[a-z]+)?)", str(s.get("name") or ""))
            code = next((abbr for abbr, prefix in _CODE_NAMES.items()
                         if (cit.get("document_name") or "").lower().startswith(prefix.lower())), None)
            label = f"[art. {number.group(1)} {code}]" if number and code else f"[{cit.get('document_name')}]"
            if text:
                lines.append(f"{label}\n{text}")
    return [{"stage": "returned_sources (proxy)", "material": "\n\n".join(lines), "reply": case["answer"]}]


_CODE_NAMES = {"c.c.": "Codice Civile", "c.p.c.": "Codice di procedura civile", "c.p.": "Codice Penale",
               "c.p.p.": "Codice di procedura penale", "c.p.a.": "Codice del processo amministrativo",
               "c.d.s.": "Codice della strada"}


def assign(sentence: str, calls: List[Dict[str, Any]]) -> Optional[int]:
    """The writer call whose reply holds the sentence (most of its 4-grams), else None (written by code)."""
    best, score = None, 0.0
    words = fold(sentence).split()
    if len(words) < 4:
        return None
    grams = _grams(words, 4)
    for i, call in enumerate(calls):
        share = len(grams & _grams(fold(call["reply"]).split(), 4)) / len(grams)
        if share > score:
            best, score = i, share
    return best if score >= 0.5 else None


def sentence_map(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    """[{"sentence", "call" (index or None), "material"}] for the final answer's sentences; a sentence no call
    wrote gets every writer call's material."""
    calls = writer_calls(case)
    union = "\n\n".join(c["material"] for c in calls)
    out = []
    for s in answer_sentences(case["answer"]):
        i = assign(s, calls) if calls else None
        out.append({"sentence": s, "call": i, "stage": calls[i]["stage"] if i is not None else "code",
                    "material": calls[i]["material"] if i is not None else union})
    return out


# =================================================================================================
# The judge (one call per batch, temperature 0, cached)
# =================================================================================================

class Judge:
    """The app's chat model at temperature 0, its replies cached by prompt hash in a JSONL file. `call` may be
    replaced (tests); None (--no-judge) answers nothing."""

    def __init__(self, cache_path: Optional[str], call: Optional[Callable[[str, str, int], str]] = None,
                 enabled: bool = True):
        self.cache_path, self.enabled, self.lock = cache_path, enabled, threading.Lock()
        self.cache: Dict[str, str] = {}
        self.calls = self.hits = self.failures = 0
        if cache_path and os.path.exists(cache_path):
            for line in open(cache_path, encoding="utf-8"):
                try:
                    item = json.loads(line)
                    self.cache[item["key"]] = item["reply"]
                except (ValueError, KeyError):
                    pass
        self._call = call or self._chat

    @staticmethod
    def _chat(system: str, human: str, max_tokens: int) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        from src.rag.ai_chat import _call_chat
        return _call_chat([SystemMessage(content=system), HumanMessage(content=human)], max_tokens=max_tokens,
                          temperature=0)

    def ask(self, system: str, human: str, max_tokens: int = 1500) -> Optional[Any]:
        """The reply parsed as JSON (a list or an object), None on no judge, a failure or an unreadable reply."""
        if not self.enabled:
            return None
        key = hashlib.sha256(f"{system}\n\x00\n{human}".encode("utf-8")).hexdigest()
        with self.lock:
            reply = self.cache.get(key)
        if reply is None:
            try:
                reply = self._call(system, human, max_tokens)
            except Exception as exc:
                with self.lock:
                    self.failures += 1
                print(f"judge call failed: {type(exc).__name__}: {str(exc)[:160]}", file=sys.stderr)
                return None
            with self.lock:
                self.calls += 1
                self.cache[key] = reply
                if self.cache_path:
                    with open(self.cache_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"key": key, "reply": reply}, ensure_ascii=False) + "\n")
        else:
            with self.lock:
                self.hits += 1
        return parse_json(reply)


def parse_json(reply: Optional[str]) -> Optional[Any]:
    text = re.sub(r"```(?:json)?\s*", "", reply or "").strip().rstrip("`").strip()
    pairs = sorted((("[", "]"), ("{", "}")), key=lambda p: (text.find(p[0]) == -1, text.find(p[0])))
    for opener, closer in pairs:            # the outermost first: an object holding a list stays an object
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except ValueError:
                continue
    return None


def _trim_material(material: str, sentences_: Sequence[str], limit: int) -> str:
    """The material within `limit` characters: the article blocks the sentences cite first, then the rest."""
    if len(material) <= limit:
        return material
    wanted = {(c["number"], c["code"]) for s in sentences_ for c in citations_in(s, labels_of(material))}
    blocks = re.split(r"\n(?=\[)", material)
    first = [b for b in blocks if any(f"[art. {n} {c}]" in b for n, c in wanted)]
    rest = [b for b in blocks if b not in first]
    out, size = [], 0
    for b in first + rest:
        if size + len(b) > limit:
            b = b[: max(0, limit - size)]
        if b:
            out.append(b)
            size += len(b)
        if size >= limit:
            break
    return "\n".join(out)


PROVENANCE_SYSTEM = (
    "Sei un revisore legale. Ricevi il MATERIALE che è stato dato a chi ha scritto un parere (testi di norme, "
    "passi di sentenze, elenchi di termini e strumenti) e alcune FRASI di quel parere, numerate. Per ogni frase "
    "decidi prima il tipo: \"regola\" (enuncia una regola giuridica o il contenuto di una norma), \"istituto\" "
    "(definisce o qualifica un istituto), \"procedura\" (un passaggio, un termine o un atto processuale), "
    "\"massima\" (ciò che una sentenza ha deciso o affermato), \"altro\" (fatti del caso, valutazioni, "
    "consigli pratici senza una regola). Per le frasi non \"altro\" decidi l'esito confrontandole SOLO con il "
    "MATERIALE, non con ciò che sai: \"supportata\" se il materiale contiene quella regola o quel contenuto (anche "
    "con altre parole), \"assente\" se il materiale non la contiene (anche se è vera), \"contraddetta\" se il "
    "materiale dice il contrario. Rispondi SOLO con un array JSON: [{\"i\": numero, \"tipo\": ..., \"esito\": "
    "\"supportata\"|\"assente\"|\"contraddetta\"|null, \"prova\": \"fino a 15 parole del materiale, o vuoto\"}].")

UNDERSTANDING_SYSTEM = (
    "Sei un avvocato italiano. Ricevi un CASO, un ARTICOLO che un esperto ritiene rilevante per il caso, e gli "
    "ELEMENTI che un sistema ha estratto dal caso (fatti, questioni, istituti), numerati. Indica quale fatto del "
    "caso rende rilevante l'articolo, citandolo con le parole del caso; il tipo di fatto che lo fa scattare, uno "
    "tra " + ", ".join(f'"{t}"' for t in TRIGGERS) + "; e quali ELEMENTI estratti catturano quel fatto in modo "
    "sufficiente a portare all'articolo (lista di numeri, vuota se nessuno). Rispondi SOLO con un oggetto JSON: "
    "{\"fatto\": \"...\", \"tipo\": \"...\", \"catturato_da\": [numeri], \"perche\": \"una frase\"}.")

ROLE_SYSTEM = (
    "Sei un avvocato italiano che rivede un parere. Ricevi il CASO (o la domanda), la PARTE ASSISTITA con la sua "
    "posizione (se manca, ricavala dal caso) e alcune FRASI numerate del parere. Elenca SOLO le frasi la cui "
    "prospettiva contraddice il ruolo della parte assistita: consigli o rimedi che solo un'altra parte può usare "
    "(per esempio costituirsi parte civile per l'imputato, chiedere il patteggiamento per la persona offesa, "
    "eccezioni del convenuto per l'attore), argomenti che giovano alla controparte presentati come propri, "
    "azioni contro la parte assistita stessa. NON vanno elencate: una frase che presenta un'obiezione, una mossa o "
    "un diritto della controparte come tale (\"la persona offesa potrà costituirsi parte civile\", \"la "
    "controparte eccepirà la prescrizione\", \"l'accusa sostiene che\"), anche senza replica; un consiglio che "
    "spetta proprio al ruolo della parte assistita (per la persona offesa o i suoi congiunti: querela, denuncia, "
    "costituzione di parte civile, risarcimento; per l'imputato o l'indagato: riti alternativi, difese, eccezioni; "
    "per l'attore: la domanda; per il convenuto: le eccezioni); la descrizione neutra di una norma o di un "
    "istituto. Nel dubbio non elencare. Rispondi SOLO con un oggetto JSON: {\"parte\": \"...\", \"ruolo\": \"attore|convenuto|"
    "imputato|indagato|persona offesa|parte civile|ricorrente|resistente|altro\", \"contrasti\": [{\"i\": numero, "
    "\"perche\": \"una frase\"}]}.")


def _batches(items: Sequence[Any], size: int) -> List[List[Any]]:
    return [list(items[i:i + size]) for i in range(0, len(items), size)]


# =================================================================================================
# The four mechanisms
# =================================================================================================

class Commi:
    """Each article's commi from Neo4j (read-only), cached; nothing without a driver."""

    def __init__(self, driver=None, database: Optional[str] = None):
        self.driver, self.database, self.cache, self.lock = driver, database, {}, threading.Lock()

    def get(self, number: str, code: str) -> Tuple[List[str], str]:
        key = (number, code)
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        found: Tuple[List[str], str] = ([], "no db")
        if self.driver is not None:
            try:
                found = self._read(number, code)
            except Exception as exc:
                print(f"commi of {number} {code}: {type(exc).__name__}: {exc}", file=sys.stderr)
                found = ([], "error")
        with self.lock:
            self.cache[key] = found
        return found

    def _read(self, number: str, code: str) -> Tuple[List[str], str]:
        import neo4j
        prefix = _CODE_NAMES.get(code)
        with self.driver.session(database=self.database, default_access_mode=neo4j.READ_ACCESS) as session:
            where = "d.name STARTS WITH $name" if prefix else "d.name = $name"
            docs = session.run(f"MATCH (d:Document) WHERE {where} AND coalesce(d.document_type, 'primary') = "
                               "'primary' RETURN d.id AS id LIMIT 5", name=prefix or code).data()
            for doc in docs:
                rows = session.run("MATCH (d:Document {id: $doc})-[:CONTAINS]->(s:Section) WHERE s.name = $num OR "
                                   "s.name STARTS WITH $num + '.' RETURN s.name AS name, s.plain_text AS text",
                                   doc=doc["id"], num=number).data()
                if not rows:
                    continue
                # The commi exactly as COMMA_MATERIAL_ENABLED gives them (src/rag/comma_material.py), so the audit
                # measures the same units the writer receives.
                from src.rag import comma_material
                return comma_material.commi_from_sections(number, {r["name"] or "": r["text"] or "" for r in rows})
        return [], "not found"


def audit_granularity(case: Dict[str, Any], mapped: List[Dict[str, Any]], commi: Commi) -> Dict[str, Any]:
    claims = []
    for item in mapped:
        labels = labels_of(item["material"])
        blocks = material_blocks(item["material"])
        for cite in citations_in(item["sentence"], labels):
            key = (cite["number"], cite["code"])
            block = blocks.get(key)
            parts, how = commi.get(*key) if block is not None or cite.get("comma") else ([], "skipped")
            claim = granularity_claim(item["sentence"], cite, block, parts)
            claim.update(stage=item["stage"], split=how)
            claims.append(claim)
    counts = Counter(c["status"] for c in claims)
    return {"claims": claims, "counts": dict(counts), "cited": len(claims),
            "defects": counts["beyond_truncation"] + counts["comma_not_given"],
            "proxy": case["proxy"]}


def audit_provenance(case: Dict[str, Any], mapped: List[Dict[str, Any]], judge: Judge, batch: int,
                     material_chars: int) -> Dict[str, Any]:
    by_material: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    for i, item in enumerate(mapped):
        by_material[(item["stage"], item["call"] if item["call"] is not None else -1)].append(i)
    verdicts: Dict[int, Dict[str, Any]] = {}
    jobs = []
    for _, idx in by_material.items():
        for chunk in _batches(idx, batch):
            jobs.append(chunk)
    for chunk in jobs:
        listed = [mapped[i]["sentence"] for i in chunk]
        material = _trim_material(mapped[chunk[0]]["material"], listed, material_chars)
        human = (f"MATERIALE:\n{material}\n\nFRASI:\n" + "\n".join(f"{n}. {s}" for n, s in enumerate(listed, 1)))
        reply = judge.ask(PROVENANCE_SYSTEM, human, max_tokens=120 + 70 * len(chunk))
        for item in reply if isinstance(reply, list) else []:
            try:
                n = int(item.get("i")) - 1
            except (TypeError, ValueError, AttributeError):
                continue
            if 0 <= n < len(chunk):
                verdicts[chunk[n]] = item
    sentences_out = []
    for i, item in enumerate(mapped):
        v = verdicts.get(i) or {}
        kind = str(v.get("tipo") or "").strip().lower() or None
        verdict = (str(v.get("esito") or "").strip().lower() or None) if kind and kind != "altro" else None
        if verdict not in (None, "supportata", "assente", "contraddetta"):
            verdict = None
        sentences_out.append({"sentence": item["sentence"], "stage": item["stage"], "kind": kind,
                              "verdict": verdict, "evidence": v.get("prova") or ""})
    rule = [s for s in sentences_out if s["kind"] and s["kind"] != "altro"]
    counts = Counter(s["verdict"] or "unjudged" for s in rule)
    return {"sentences": sentences_out, "rule_sentences": len(rule), "counts": dict(counts),
            "judged": sum(1 for s in sentences_out if s["kind"]), "total": len(sentences_out),
            "memory_share": (counts["assente"] / len(rule)) if rule else None, "proxy": case["proxy"]}


def _plain(value: Any) -> str:
    """A field's value as one line: a string as is, anything else as compact JSON."""
    if isinstance(value, str):
        return " ".join(value.split())
    return json.dumps(value, ensure_ascii=False, default=str)


def as_items(value: Any, prefix: str = "") -> List[Tuple[str, Any]]:
    """(label, item) for every item of a field whatever its shape: a list gives its items, a dict of typed
    fields ({"parties": [...], "events": [...], "notes": "..."}) each field's items labelled by its key, a scalar
    itself; None and empty values nothing."""
    if value is None or value == "" or value == [] or value == {}:
        return []
    if isinstance(value, (list, tuple, set)):
        return [(prefix, v) for v in value if v not in (None, "", [], {})]
    if isinstance(value, dict):
        out = []
        for key, v in value.items():
            label = f"{prefix} {key}".strip()
            out += as_items(v, label) if isinstance(v, (list, tuple, set)) else (
                [(label, v)] if v not in (None, "", [], {}) else [])
        return out
    return [(prefix, value)]


def _field(item: Any, *keys: str) -> Any:
    """The first of `keys` an item has (a dict), else the item itself (a string, a number)."""
    if isinstance(item, dict):
        return next((item[k] for k in keys if item.get(k) not in (None, "", [], {})), None)
    return item


def _extracted(rec: Dict[str, Any], calls: Sequence[Dict[str, Any]] = ()) -> List[str]:
    """What the understanding extracted: the frame's facts, tasks and central issues, the analysis' facts (from
    its dump's reply), its questions, the institutions with what they resolved to, the typed facts and the rules
    they fired. Every field read in either shape, a list or a dict of typed fields (FACT_SIGNALS' facts are the
    latter: parties, events, property_history, ...)."""
    items: List[str] = []
    frame = rec.get("frame") if isinstance(rec.get("frame"), dict) else {}
    for key in ("facts", "tasks", "central_issue_candidates"):
        for label, x in as_items(frame.get(key), key):
            items.append(f"frame {label}: {_plain(_field(x, 'nome', 'fatto', 'text') or x)}")
    if frame.get("procedural_posture"):
        items.append(f"frame fase: {_plain(frame['procedural_posture'])}")
    analysis = next((parse_json(c.get("reply")) for c in calls if str(c.get("stage", "")).startswith("case_analysis")),
                    None)
    if isinstance(analysis, dict):
        for label, f in as_items(analysis.get("fatti"), "fatto")[:30]:
            items.append(f"{label}: {_plain(_field(f, 'fatto', 'testo', 'text') or f)}")
    for _, q in as_items(rec.get("questioni"))[:20]:
        if isinstance(q, dict):
            items.append(f"questione: {_plain(q.get('titolo') or '')} — istituti {_plain(q.get('istituti') or [])}; "
                         f"parole {_plain(q.get('parole_chiave') or [])}")
        else:
            items.append(f"questione: {_plain(q)}")
    for _, inst in as_items(rec.get("institutions"))[:30]:
        if not isinstance(inst, dict):
            items.append(f"istituto: {_plain(inst)}")
            continue
        articles = [_field(a, "label") for _, a in as_items(inst.get("articles"))]
        items.append(f"istituto: {_plain(inst.get('nome') or '')} -> "
                     f"{', '.join(_plain(a) for a in articles if a) or 'non risolto'}")
    signals = rec.get("fact_signals") if isinstance(rec.get("fact_signals"), dict) else {}
    for label, f in as_items(signals.get("facts"))[:40]:
        items.append(f"fatto tipizzato{' ' + label if label else ''}: {_plain(f)}")
    for _, rule in as_items(signals.get("fired"))[:10]:
        items.append(f"regola scattata: {_plain(_field(rule, 'institution', 'id') or rule)}")
    return items[:80]


# A lost article whose trigger fact WAS captured: the first step after the understanding that it fails.
FUNNEL = ("resolved", "in_pool", "graded", "in_prompt", "in_plan")
FUNNEL_NAMES = {"resolved": "article resolved / found (issue -> article)", "in_pool": "in the pool",
                "graded": "passed the applicability check", "in_prompt": "in the draft prompt",
                "in_plan": "in the plan"}
CAPTURE_KINDS = {"institution_elsewhere": "an institution captured it, resolved to other articles",
                 "institution_unresolved": "an institution captured it, resolved to nothing",
                 "issue": "an issue / central question captured it, no institution",
                 "fact": "only a fact captured it"}


def funnel(slot: Dict[str, Any]) -> Dict[str, Any]:
    """{"steps": {step: passed}, "lost_at": the first failed step or None, "check", "channels"} of an expected
    article's diagnose_long slot: resolved (an institution resolved to it, or a channel found it: fusion rank, pool,
    vote top 20), in the pool, past the applicability check, in the draft prompt, in the plan."""
    info = slot.get("understanding") or {}
    vote = slot.get("vote_rank")
    conditions = {
        "resolved": bool(info.get("resolved_by_institution") or slot.get("institutions") or slot.get("rank")
                         or slot.get("in_pool") or (vote is not None and vote <= 20)),
        "in_pool": bool(slot.get("in_pool")),
        "graded": slot.get("check") != "rejected",
        "in_prompt": slot.get("in_prompt") is not False,
        "in_plan": slot.get("in_plan") is not False,
    }
    steps, ok, lost_at = {}, True, None
    for step in FUNNEL:
        ok = ok and conditions[step]
        steps[step] = ok
        if not ok and lost_at is None:
            lost_at = step
    return {"steps": steps, "lost_at": lost_at, "check": slot.get("check"), "channels": slot.get("channels") or []}


def capture_kind(captured: Sequence[str]) -> str:
    """How the understanding held a captured trigger (CAPTURE_KINDS), from the extracted elements that captured it."""
    insts = [c for c in captured if c.startswith("istituto:")]
    if any(not c.endswith("non risolto") for c in insts):
        return "institution_elsewhere"
    if insts:
        return "institution_unresolved"
    if any(c.startswith(("questione:", "frame central_issue", "frame tasks", "regola scattata")) for c in captured):
        return "issue"
    return "fact"


def audit_understanding(case: Dict[str, Any], judge: Judge, analysed: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not analysed:
        return {"lost": [], "available": False}
    extracted = _extracted(case["record"], case["calls"])
    lost = []
    for slot in analysed.get("slots") or []:
        if slot.get("stage") not in LOST_STAGES:
            continue
        rubrica = slot.get("in_expected_rubrica") or slot.get("rubrica") or ""
        human = (f"CASO:\n{case['case_text'][:6000]}\n\nARTICOLO: art. {slot['label'].replace('art. ', '')}"
                 + (f" — {rubrica}" if rubrica else "") + "\n\nELEMENTI ESTRATTI:\n"
                 + ("\n".join(f"{i}. {x}" for i, x in enumerate(extracted, 1)) or "(nessuno)"))
        reply = judge.ask(UNDERSTANDING_SYSTEM, human, max_tokens=300)
        reply = reply if isinstance(reply, dict) else {}
        trigger = str(reply.get("tipo") or "").strip().lower()
        raw = reply.get("catturato_da")
        raw = re.findall(r"\d+", raw) if isinstance(raw, str) else [x for _, x in as_items(raw)]
        captured = sorted({int(x) for x in raw if str(x).strip().isdigit() and 0 < int(x) <= len(extracted)})
        held = [extracted[i - 1] for i in captured]
        lost.append({"article": slot["label"], "stage": slot["stage"], "rubrica": rubrica,
                     "fact": reply.get("fatto") or "", "trigger": trigger if trigger in TRIGGERS else
                     ("altro" if reply else None), "captured": held,
                     "why": reply.get("perche") or "", "judged": bool(reply),
                     "funnel": funnel(slot), "capture_kind": capture_kind(held) if held else None})
    return {"lost": lost, "available": True,
            "defects": sum(1 for x in lost if x["judged"] and not x["captured"])}


_ROLE_PATTERNS = [   # (role words, sentence pattern, why): a second, fixed signal next to the judge
    (r"imputat|indagat|accusat", r"costitu\w+\s+parte\s+civile|\bquerela\b(?!.{0,40}contro di)", "accused advised a victim's act"),
    (r"persona offesa|parte civile|vittima|querelant", r"patteggiament|rito abbreviato|messa alla prova|oblazione", "victim advised a defendant's rite"),
    (r"convenut|resistente", r"atto di citazione|proporre (?:la )?domanda giudiziale|notificare la citazione", "defendant advised to sue"),
    (r"attore|ricorrente|creditore", r"eccepire la prescrizione|eccezione di incompetenza|chiamata in causa del terzo", "claimant advised a defendant's exception"),
]


# A sentence whose subject is the other side, or that reports its move: the other side's act described as such.
_OTHER_SIDE_RE = re.compile(
    r"controparte|\bavversari|\bobiezion|\bobiett|\beccepir\w*\s+(?:che|la|il)|l['’]accusa|pubblico ministero|"
    r"\bil pm\b|persona offesa\s+(?:potr|può|potrebbe|ha|avrà|intend|vorrà)|\b(?:la|le)\s+vittim\w*\s+(?:potr|può|"
    r"potrebbe)|\b(?:i|gli)\s+(?:familiari|congiunti|eredi)\s+(?:della vittima\s+)?(?:potr|possono|potrebbero)|"
    r"\bl['’]imputato\s+(?:potr|può|potrebbe)|\bil convenuto\s+(?:potr|può|potrebbe)|\bl['’]attore\s+(?:potr|può|"
    r"potrebbe)", re.IGNORECASE)
# The client's own acts by role: a judge flag on a sentence advising one of them is not kept.
_OWN_ACTS = [
    (r"persona offesa|parte civile|vittim|querelant|familiar|congiunt|\bered", r"\bquerel|denunci|costitu\w+\s+parte\s+"
     r"civile|risarciment|opposizione all['’]archiviazione"),
    (r"imputat|indagat|accusat", r"patteggiament|rito abbreviato|messa alla prova|oblazione|\bdifes|memori|"
     r"proscioglimento|assoluzion|attenuant|tenuità"),
    (r"attore|ricorrente|creditore", r"atto di citazione|domanda giudiziale|ricorso|risarciment|decreto ingiuntivo"),
    (r"convenut|resistente", r"eccepir|eccezion|chiamata in causa|domanda riconvenzionale"),
]


def _who(role: Dict[str, str]) -> str:
    return f"{role.get('party', '')} {role.get('position', '')} {role.get('judged_role') or ''}".lower()


def role_pattern_hits(role: Dict[str, str], sentence: str) -> Optional[str]:
    """The fixed pattern a sentence hits against the client's role, unless it reports the other side's move."""
    who = _who(role)
    if _OTHER_SIDE_RE.search(sentence):
        return None
    for words, pattern, why in _ROLE_PATTERNS:
        if re.search(words, who) and re.search(pattern, sentence, re.IGNORECASE):
            return why
    return None


def role_flag_filtered(role: Dict[str, str], sentence: str) -> Optional[str]:
    """Why a judge's role flag on the sentence is not kept, or None: it reports the other side's objection or move,
    or advises an act of the client's own role (and no fixed pattern says otherwise)."""
    if _OTHER_SIDE_RE.search(sentence):
        return "the other side's move, as such"
    who = _who(role)
    if role_pattern_hits(role, sentence):
        return None
    for words, acts in _OWN_ACTS:
        if re.search(words, who) and re.search(acts, sentence, re.IGNORECASE):
            return "an act of the client's own role"
    return None


def audit_role(case: Dict[str, Any], mapped: List[Dict[str, Any]], judge: Judge, batch: int) -> Dict[str, Any]:
    role = dict(case.get("role") or {})
    listed = [m["sentence"] for m in mapped]
    flagged: Dict[int, Dict[str, Any]] = {}
    for start in range(0, len(listed), batch):
        chunk = listed[start:start + batch]
        human = (f"CASO:\n{case['case_text'][:5000]}\n\nPARTE ASSISTITA: "
                 + (f"{role.get('party')} ({role.get('position')})" if role.get("party") else "(da ricavare dal caso)")
                 + "\n\nFRASI:\n" + "\n".join(f"{n}. {s}" for n, s in enumerate(chunk, 1)))
        reply = judge.ask(ROLE_SYSTEM, human, max_tokens=400)
        if not isinstance(reply, dict):
            continue
        if not role.get("party") and reply.get("parte"):
            role.update(party=str(reply.get("parte")), position=str(reply.get("ruolo") or ""), **{"from": "judge"})
        role.setdefault("judged_role", reply.get("ruolo"))
        for item in reply.get("contrasti") or []:
            try:
                n = int(item.get("i")) - 1
            except (TypeError, ValueError, AttributeError):
                continue
            if 0 <= n < len(chunk):
                flagged[start + n] = {"sentence": chunk[n], "why": item.get("perche") or "", "judge": True}
    filtered = []
    for i in list(flagged):
        why = role_flag_filtered(role, flagged[i]["sentence"])
        if why:
            filtered.append({**flagged.pop(i), "filtered": why})
    for i, s in enumerate(listed):
        hit = role_pattern_hits(role, s)
        if hit:
            flagged.setdefault(i, {"sentence": s, "why": hit, "judge": False})["pattern"] = hit
    items = [flagged[i] for i in sorted(flagged)]
    return {"role": role, "contradictions": items, "sentences": len(listed), "filtered": filtered,
            "defects": sum(1 for x in items if x.get("judge") or x.get("pattern"))}


def audit_case(case: Dict[str, Any], judge: Judge, commi: Commi, ev, args) -> Dict[str, Any]:
    mapped = sentence_map(case)
    analysed = None
    if case["kind"] == "diagnose" and case["record"].get("expected") and ev is not None:
        try:
            import diagnose_long
            analysed = diagnose_long.analyse_case(case["record"], ev)
        except Exception as exc:
            print(f"{case['key']}: diagnose_long.analyse_case failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    out = {"key": case["key"], "run": case["run"], "id": case["id"], "kind": case["kind"], "proxy": case["proxy"],
           "sentences": len(mapped), "writer_calls": len(writer_calls(case)),
           "unassigned": sum(1 for m in mapped if m["call"] is None)}
    out["granularity"] = audit_granularity(case, mapped, commi) if not case["proxy"] else \
        {"claims": [], "counts": {}, "cited": 0, "defects": 0, "proxy": True, "skipped": "no writer material"}
    out["provenance"] = audit_provenance(case, mapped, judge, args.batch, args.material_chars)
    out["understanding"] = audit_understanding(case, judge, analysed)
    out["role"] = audit_role(case, mapped, judge, args.batch)
    out["grounded"] = grounded_case_totals(case)
    attribute(out)
    return out


def grounded_case_totals(case: Dict[str, Any]) -> Dict[str, int]:
    """The case's grounded_review counts (summed over its logs), {} for a run without the review."""
    out: Counter = Counter()
    for log in grounded_events(case):
        out.update({k: int(v or 0) for k, v in list((log.get("totals") or {}).items())
                    + list((log.get("counts") or {}).items())})
        out["fetched"] += len(log.get("fetched") or {})
        out["not_found"] += len(log.get("not_found") or [])
    return dict(out)


def attribute(case_out: Dict[str, Any]) -> None:
    """A provenance defect on a sentence granularity already explains is counted once, as granularity."""
    gran = {c["sentence"] for c in case_out["granularity"]["claims"]
            if c["status"] in ("beyond_truncation", "comma_not_given")}
    prov = [s for s in case_out["provenance"]["sentences"] if s["verdict"] in ("assente", "contraddetta")]
    case_out["provenance"]["defects_raw"] = len(prov)
    case_out["provenance"]["explained_by_granularity"] = sum(1 for s in prov if s["sentence"] in gran)
    case_out["provenance"]["defects"] = len(prov) - case_out["provenance"]["explained_by_granularity"]
    case_out["understanding"].setdefault("defects", 0)


# =================================================================================================
# Report
# =================================================================================================

def summary(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    totals = {m: sum(c[m].get("defects", 0) for c in cases) for m in MECHANISMS}
    all_defects = sum(totals.values())
    ranked = sorted(MECHANISMS, key=lambda m: -totals[m])
    gran = Counter()
    for c in cases:
        gran.update(c["granularity"]["counts"])
    prov = Counter()
    for c in cases:
        prov.update(c["provenance"]["counts"])
    rule = sum(c["provenance"]["rule_sentences"] for c in cases)
    triggers = Counter(x["trigger"] or "unjudged" for c in cases for x in c["understanding"]["lost"])
    missed = Counter(x["trigger"] or "unjudged" for c in cases for x in c["understanding"]["lost"]
                     if x["judged"] and not x["captured"])
    held = [x for c in cases for x in c["understanding"]["lost"] if x["captured"]]
    lost_at = Counter((x.get("funnel") or {}).get("lost_at") or "none" for x in held)
    kinds = Counter(x.get("capture_kind") or "fact" for x in held
                    if (x.get("funnel") or {}).get("lost_at") == "resolved")
    lead = Counter(max(MECHANISMS, key=lambda m: c[m].get("defects", 0)) for c in cases
                   if any(c[m].get("defects", 0) for m in MECHANISMS))
    return {"totals": totals, "all": all_defects, "ranked": ranked,
            "share": {m: (totals[m] / all_defects if all_defects else 0.0) for m in MECHANISMS},
            "granularity": dict(gran), "cited": sum(c["granularity"]["cited"] for c in cases),
            "provenance": dict(prov), "rule_sentences": rule,
            "memory_share": (prov["assente"] / rule) if rule else None,
            "contradicted_share": (prov["contraddetta"] / rule) if rule else None,
            "explained_by_granularity": sum(c["provenance"].get("explained_by_granularity", 0) for c in cases),
            "triggers": dict(triggers), "triggers_missed": dict(missed),
            "lost": sum(len(c["understanding"]["lost"]) for c in cases),
            "captured_lost": len(held), "captured_lost_at": dict(lost_at), "unresolved_capture": dict(kinds),
            "role_filtered": sum(len(c["role"].get("filtered") or []) for c in cases),
            "grounded": grounded_totals(cases),
            "role_defects": totals["role"], "cases_led": dict(lead), "cases": len(cases),
            "proxy_cases": sum(1 for c in cases if c["proxy"])}


def grounded_totals(cases: List[Dict[str, Any]]) -> Dict[str, int]:
    """GROUNDED_REVIEW_ENABLED's own counts over the cases (audit_case keeps them per case)."""
    out: Counter = Counter()
    for c in cases:
        out.update(c.get("grounded") or {})
    return dict(out)


def _pct(x: Optional[float]) -> str:
    return "-" if x is None else f"{100 * x:.0f}%"


def _cut(text: str, n: int = 220) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def render(cases: List[Dict[str, Any]], meta: Dict[str, Any], max_examples: int = 10) -> str:
    s = summary(cases)
    L = [f"# fetch_audit: {', '.join(meta['runs'])}", "",
         f"{s['cases']} answers ({s['proxy_cases']} with the returned sources as a proxy for the material: eval runs "
         f"and dryrun records have no LLM dumps). Judge: {meta['judge']}; commi: {meta['commi']}. "
         f"Generated {meta['when']}.", "",
         "## Ranked summary: share of answer defects explained by each mechanism", "",
         "| rank | mechanism | defects | share | answers where it leads |", "|---:|---|---:|---:|---:|"]
    for rank, m in enumerate(s["ranked"], 1):
        L.append(f"| {rank} | {TITLES[m]} | {s['totals'][m]} | {_pct(s['share'][m])} | {s['cases_led'].get(m, 0)} |")
    L += ["", f"Defects in all: {s['all']}. A rule-sentence on a comma the writer never had is counted once, as "
              f"granularity ({s['explained_by_granularity']} such). Units differ: claims (1), rule-sentences (2), lost "
              f"expected articles (3), sentences (4).", ""]

    # --- 1
    L += [f"## {TITLES['granularity']}", "",
          "| answer | cited claims | in material | comma not given | beyond truncation | article not given | unaligned / no text |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for c in cases:
        g = c["granularity"]
        if g.get("skipped"):
            L.append(f"| {c['key']} | n/a ({g['skipped']}) | | | | | |")
            continue
        k = g["counts"]
        L.append(f"| {c['key']} | {g['cited']} | {k.get('in_material', 0)} | {k.get('comma_not_given', 0)} | "
                 f"{k.get('beyond_truncation', 0)} | {k.get('article_not_given', 0)} | "
                 f"{k.get('unaligned', 0) + k.get('no_text', 0)} |")
    k = s["granularity"]
    L += [f"| **total** | {s['cited']} | {k.get('in_material', 0)} | {k.get('comma_not_given', 0)} | "
          f"{k.get('beyond_truncation', 0)} | {k.get('article_not_given', 0)} | {k.get('unaligned', 0) + k.get('no_text', 0)} |",
          "", "Examples:"]
    ex = [(c["key"], x) for c in cases for x in c["granularity"]["claims"]
          if x["status"] in ("beyond_truncation", "comma_not_given")]
    L += [f"- {key}: art. {x['ref']}, comma {x['comma']}/{x['commi']} ({x['status'].replace('_', ' ')}; "
          f"{x['given_chars']} chars given, {x['stage']}): «{_cut(x['sentence'])}»" for key, x in ex[:max_examples]] or ["- none"]

    # --- 2
    L += ["", f"## {TITLES['provenance']}", "",
          "| answer | sentences | rule-sentences | supported | not in material | contradicted | unjudged | from memory |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in cases:
        p = c["provenance"]
        k = p["counts"]
        L.append(f"| {c['key']}{' (proxy)' if c['proxy'] else ''} | {p['total']} | {p['rule_sentences']} | "
                 f"{k.get('supportata', 0)} | {k.get('assente', 0)} | {k.get('contraddetta', 0)} | "
                 f"{k.get('unjudged', 0)} | {_pct(p['memory_share'])} |")
    k = s["provenance"]
    L += [f"| **total** | {sum(c['provenance']['total'] for c in cases)} | {s['rule_sentences']} | "
          f"{k.get('supportata', 0)} | {k.get('assente', 0)} | {k.get('contraddetta', 0)} | {k.get('unjudged', 0)} | "
          f"{_pct(s['memory_share'])} |", "",
          f"Share of rule-sentences written from memory: {_pct(s['memory_share'])}; contradicting the material: "
          f"{_pct(s['contradicted_share'])}.", "", "Examples (not in material, then contradicted):"]
    ex = sorted(((c["key"], x) for c in cases for x in c["provenance"]["sentences"]
                 if x["verdict"] in ("assente", "contraddetta")), key=lambda kx: kx[1]["verdict"] != "contraddetta")
    L += [f"- {key} [{x['verdict']}, {x['kind']}, {x['stage']}]: «{_cut(x['sentence'])}»"
          + (f" — material: «{_cut(x['evidence'], 120)}»" if x["evidence"] else "") for key, x in ex[:max_examples]] or ["- none"]

    # --- 3
    L += ["", f"## {TITLES['understanding']}", "",
          "| answer | lost (understanding / not in plan) | trigger not captured | captured but lost |", "|---|---:|---:|---:|"]
    for c in cases:
        u = c["understanding"]
        if not u.get("available"):
            L.append(f"| {c['key']} | n/a (no expected stages) | | |")
            continue
        lost = u["lost"]
        L.append(f"| {c['key']} | {len(lost)} ({sum(x['stage'] == LOST_STAGES[0] for x in lost)} / "
                 f"{sum(x['stage'] == LOST_STAGES[1] for x in lost)}) | {u['defects']} | "
                 f"{sum(1 for x in lost if x['captured'])} |")
    L += [f"| **total** | {s['lost']} | {s['totals']['understanding']} | "
          f"{sum(1 for c in cases for x in c['understanding']['lost'] if x['captured'])} |", "",
          "| trigger | lost | not captured |", "|---|---:|---:|"]
    for t in TRIGGERS + ("unjudged",):
        if s["triggers"].get(t):
            L.append(f"| {TRIGGER_NAMES.get(t, t)} | {s['triggers'][t]} | {s['triggers_missed'].get(t, 0)} |")
    if s["captured_lost"]:
        L += ["", f"### Captured but lost: the first step each of the {s['captured_lost']} fails", "",
              "| step | lost there | share |", "|---|---:|---:|"]
        for step in FUNNEL + ("none",):
            n = s["captured_lost_at"].get(step, 0)
            if n:
                L.append(f"| {FUNNEL_NAMES.get(step, 'passed every step (lost in the writing)')} | {n} | "
                         f"{_pct(n / s['captured_lost'])} |")
        if s["unresolved_capture"]:
            L += ["", "Not resolved, by what captured the trigger:", "", "| captured by | lost |", "|---|---:|"]
            for kind, n in sorted(s["unresolved_capture"].items(), key=lambda kv: -kv[1]):
                L.append(f"| {CAPTURE_KINDS.get(kind, kind)} | {n} |")
        L += ["", "Examples (captured, then lost):"]
        ex = [(c["key"], x) for c in cases for x in c["understanding"]["lost"] if x["captured"]]
        L += [f"- {key}: {x['article']} lost at «{FUNNEL_NAMES.get((x.get('funnel') or {}).get('lost_at'), '-')}» "
              f"(check: {(x.get('funnel') or {}).get('check') or '-'}; channels: "
              f"{', '.join((x.get('funnel') or {}).get('channels') or []) or 'none'}); captured by «{_cut(x['captured'][0], 140)}»"
              for key, x in ex[:max_examples]]
    L += ["", "Examples (trigger not captured):"]
    ex = [(c["key"], x) for c in cases for x in c["understanding"]["lost"] if x["judged"] and not x["captured"]]
    L += [f"- {key}: {x['article']} ({x['stage']}; {TRIGGER_NAMES.get(x['trigger'], x['trigger'])}): fact «{_cut(x['fact'], 160)}»"
          + (f" — {_cut(x['why'], 120)}" if x["why"] else "") for key, x in ex[:max_examples]] or ["- none"]

    # --- 4
    L += ["", f"## {TITLES['role']}", "", "| answer | client (role, from) | sentences | contradicting (judge / pattern) |",
          "|---|---|---:|---:|"]
    for c in cases:
        r = c["role"]
        role = r["role"]
        who = f"{role.get('party') or '?'} ({role.get('position') or role.get('judged_role') or '?'}, {role.get('from') or '-'})"
        items = r["contradictions"]
        L.append(f"| {c['key']} | {_cut(who, 70)} | {r['sentences']} | {r['defects']} "
                 f"({sum(1 for x in items if x.get('judge'))} / {sum(1 for x in items if x.get('pattern'))}) |")
    L += [f"| **total** | | {sum(c['role']['sentences'] for c in cases)} | {s['totals']['role']} |", "",
          f"Judge flags not kept: {s['role_filtered']} (the other side's objection or move presented as such, or an "
          f"act of the client's own role).", "", "Examples:"]
    ex = [(c["key"], x) for c in cases for x in c["role"]["contradictions"]]
    L += [f"- {key}: «{_cut(x['sentence'])}» — {_cut(x['why'], 120)}" for key, x in ex[:max_examples]] or ["- none"]

    # --- GROUNDED_REVIEW_ENABLED
    g = s.get("grounded") or {}
    if g:
        L += ["", "## Grounded review (GROUNDED_REVIEW_ENABLED, as logged by the run)", "",
              f"Sentences judged {g.get('judged', 0)} of {g.get('sentences', 0)}; rule-sentences {g.get('rule', 0)}: "
              f"supported {g.get('supportata', 0)}, not in material {g.get('assente', 0)}, contradicted "
              f"{g.get('contraddetta', 0)}. Done: kept after reading the cited article {g.get('kept_fetched', 0)}, "
              f"rewritten from the material {g.get('rewritten', 0)}, pointers {g.get('pointer', 0)}, dropped "
              f"{g.get('dropped', 0)}, kept unchanged {g.get('kept', 0)}. Articles read {g.get('fetched', 0)}, not "
              f"found {g.get('not_found', 0)}."]

    # --- fixes
    top = [m for m in s["ranked"] if s["totals"][m]][:2]
    L += ["", "## Fix directions for the top two mechanisms", ""]
    if not top:
        L.append("No defect found: nothing to fix.")
    for m in top:
        L += [f"### {TITLES[m]} ({_pct(s['share'][m])} of the defects)", ""] + [f"- {d}" for d in FIX_DIRECTIONS[m]] + [""]
    return "\n".join(L) + "\n"


# =================================================================================================
# Main
# =================================================================================================

def _driver(no_db: bool):
    if no_db:
        return None, None
    try:
        import neo4j
        uri = os.getenv("NEO4J_URI")
        if not uri:
            return None, None
        return (neo4j.GraphDatabase.driver(uri, auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))),
                os.getenv("NEO4J_DATABASE", "neo4j"))
    except Exception as exc:
        print(f"no Neo4j, granularity without the commi: {exc}", file=sys.stderr)
        return None, None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("runs", nargs="*", help="diagnose_long run folders, eval run folders, dryrun --save files")
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--out", help="output folder (default ./fetch_audit_runs/<date-time>)")
    parser.add_argument("--env", default=os.path.join(os.getcwd(), ".env"))
    parser.add_argument("--batch", type=int, default=12, help="sentences per judge call")
    parser.add_argument("--material-chars", type=int, default=14000, help="material per judge call, at most")
    parser.add_argument("--parallel", type=int, default=4, help="answers audited at once")
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--no-judge", action="store_true", help="no model call: granularity and role patterns only")
    parser.add_argument("--no-db", action="store_true", help="no Neo4j: no commi")
    parser.add_argument("--report-only", metavar="DIR", help="report again from DIR/audit.json")
    args = parser.parse_args(argv)

    if args.report_only:
        data = json.load(open(os.path.join(args.report_only, "audit.json"), encoding="utf-8"))
        report = render(data["cases"], data["meta"], args.max_examples)
        open(os.path.join(args.report_only, "report.md"), "w", encoding="utf-8").write(report)
        print(report)
        return 0
    if not args.runs:
        parser.error("give at least one run (or --report-only DIR)")
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, HERE)
    try:
        from dotenv import load_dotenv
        load_dotenv(args.env)
    except ImportError:
        pass
    only = {x.strip() for x in args.only.split(",")} if args.only else None
    cases = load(args.runs, only)
    if not cases:
        parser.error("no case in those runs")
    out_dir = args.out or os.path.join("fetch_audit_runs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    judge = Judge(os.path.join(out_dir, "judge_cache.jsonl"), enabled=not args.no_judge)
    driver, database = _driver(args.no_db)
    commi = Commi(driver, database)
    ev = None
    if any(c["kind"] == "diagnose" for c in cases):
        try:
            import diagnose_long
            ev = diagnose_long._eval_module()
        except Exception as exc:
            print(f"no eval module, understanding skipped: {exc}", file=sys.stderr)
    t0 = time.time()
    results: List[Optional[Dict[str, Any]]] = [None] * len(cases)

    def run(i):
        print(f"... {cases[i]['key']}", file=sys.stderr, flush=True)
        results[i] = audit_case(cases[i], judge, commi, ev, args)
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        list(pool.map(run, range(len(cases))))
    if driver is not None:
        driver.close()
    meta = {"runs": [os.path.basename(os.path.normpath(r)) for r in args.runs],
            "judge": ("off" if args.no_judge else f"the app's chat model, temperature 0 ({judge.calls} calls, "
                      f"{judge.hits} cached, {judge.failures} failed)"),
            "commi": "Neo4j (read-only)" if driver is not None else "none (--no-db or no NEO4J_URI)",
            "when": datetime.now().strftime("%Y-%m-%d %H:%M"), "seconds": round(time.time() - t0)}
    done = [r for r in results if r]
    json.dump({"meta": meta, "cases": done, "summary": summary(done)},
              open(os.path.join(out_dir, "audit.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    report = render(done, meta, args.max_examples)
    open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8").write(report)
    print(report)
    print(f"Written to {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
