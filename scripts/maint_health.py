"""Maintenance agent, job 1: the nightly health report of the knowledge graph (Flavio's brief, 27 Oct). READ-ONLY.

Runs on be2 (the database, the ingestion queue and the night jobs' logs are there), after the 02:00 vectors and 04:00
CITES jobs. Writes, in --dir (default /var/log/maint):
  health_YYYYMMDD.md     the report
  summary_YYYYMMDD.txt   one line, for the morning message
  state_YYYYMMDD.json    the numbers, so the next night reports what changed (new numbering gaps above all)

What it counts:
  - Sections without embedding_v2 (by document type, the empty ones apart) and Sections with no text property
    (neither plain_text nor abstract; tested for presence, never read)
  - Documents with no document_type or an unknown one
  - Sections detached from any Document (no incoming CONTAINS / PART_OF)
  - articles whose "N.0.0" holds only "In vigore dal ..." while another Section of theirs has the text, per code
  - gaps in the article numbering of the main codes (missing N in 1..max): a NEW gap is a parser regression
  - rulings without CITES edges, by branch; the CITES linker's unresolved citations, by reason, and the articles
    the rulings cite most that our codes lack (the map of what to load)
  - ruling cards coverage by branch (only if --cards names readable card files: they live on prod today)
  - suspected duplicate Documents (same hash, same name, same name but the year): REPORTED, never touched
  - the ingestion queue (files waiting, failed) and the last run of each night job

Safeguards: read-only session; skipped (with a summary line saying why) when MemAvailable < --min-gb or another heavy
job is running; documents read in batches with a pause; prints counts and document names only, no section text.

    python /opt/v2tools/maint_health.py --env /opt/chatbot/.env            # writes /var/log/maint/...
    python maint_health.py --skip sections,detached        # the quick version: no per-Section stages
    python maint_health.py --dir /tmp/maint --cards /root/ruling_cards/cards_pen_v3.jsonl,/root/ruling_cards/cards_civ_v1.jsonl
"""

import argparse
import datetime as dt
import glob
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

KNOWN_TYPES = {"interpretation", "ccnl", "primary", "special", "legislation", "quarantine"}
MAIN_CODES = ("Codice Civile", "Codice Penale", "Codice di procedura civile", "Codice di procedura penale",
              "Costituzione")
HEAVY_JOBS = ("backfill_embeddings_v2", "link_ruling_citations", "exp_ruling_cards", "exp_ruling_links",
              "reembed", "maint_health.py --write")
IN_FORCE = re.compile(r"^\s*in vigore dal\s+\d{1,2}\s+\w+\s+\d{4}\.?\s*$", re.IGNORECASE)
ARTICLE_NAME = re.compile(r"^(\d+)(?:-([a-z]+))?$", re.IGNORECASE)
ITALGIURE = re.compile(r"(snpen|snciv)[@_]s\w\w[@_]a\d{4}[@_]n\d+[@_]t[a-z]", re.IGNORECASE)
YEAR_SUFFIX = re.compile(r"^(?P<prefix>.*\S)\s+(?:19|20)\d{2}$")


# --- pure helpers (tested) -----------------------------------------------------------------------

def article_of(section_name: str) -> Optional[str]:
    """'1490.0.0' / '1117.3.0_2' / '337-ter' -> '1490' / '1117' / '337-ter'; None for anything else."""
    base = str(section_name or "").split(".")[0].split("_")[0].lower()
    return base if ARTICLE_NAME.match(base) else None


def numbering_gaps(articles: Iterable[str]) -> List[int]:
    """The plain numbers missing between 1 and the highest plain number present."""
    numbers = {int(ARTICLE_NAME.match(a).group(1)) for a in articles if a and ARTICLE_NAME.match(a)}
    return sorted(set(range(1, max(numbers) + 1)) - numbers) if numbers else []


def date_only_articles(sections: List[Tuple[str, int, bool]]) -> List[str]:
    """[(section name, text length, text is only "In vigore dal ...")] of one document -> the articles whose N.0.0 is
    date-only while another Section of theirs has real text (60+ chars)."""
    zero_date_only, has_text = set(), set()
    for name, length, date_only in sections:
        art = article_of(name)
        if not art:
            continue
        if name == f"{art}.0.0" and date_only:
            zero_date_only.add(art)
        elif name != f"{art}.0.0" and length >= 60:
            has_text.add(art)
    return sorted(zero_date_only & has_text, key=lambda a: (int(ARTICLE_NAME.match(a).group(1)), a))


def short_first_pieces(sections: List[Tuple[str, int, bool]]) -> List[str]:
    """The articles whose N.0.0 is short (under 60 chars) but NOT a date in force, while another Section of theirs has
    the text (60+ chars): the shared read rule (article_text.choose_text) takes such an N.0.0 as the text."""
    short_zero, has_text = set(), set()
    for name, length, date_only in sections:
        art = article_of(name)
        if not art:
            continue
        if name == f"{art}.0.0" and length < 60 and not date_only:
            short_zero.add(art)
        elif name != f"{art}.0.0" and length >= 60:
            has_text.add(art)
    return sorted(short_zero & has_text, key=lambda a: (int(ARTICLE_NAME.match(a).group(1)), a))


def normalize_name(name: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", (name or "").lower()).split())


def suspected_duplicates(docs: List[dict]) -> Dict[str, List[List[str]]]:
    """{reason: [[documents], ...]}: same hash, same normalized name, same name but the year -- report only. Rulings
    (interpretation) are compared by their Italgiure file stem, never by title: their titles are generic or anonymised
    ("Sentenza della Corte Suprema di Cassazione", "... proposto da XXXX"), and "ricorso 17360-2020" / "-2022" are two
    cases (the first report, 28 Oct: 496 such false alarms). A ruling is shown by its stem or id, never its title."""
    by_hash, by_name, by_prefix, by_stem = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for d in docs:
        ruling = d.get("type") == "interpretation"
        stem = ITALGIURE.search(d.get("file") or "") or ITALGIURE.search(d.get("name") or "")
        label = (stem.group(0) if stem else d.get("id") or "?") if ruling else (d.get("name") or d.get("id") or "?")
        if d.get("hash"):
            by_hash[d["hash"]].append(label)
        if ruling:
            if stem:
                by_stem[stem.group(0).lower().replace("_", "@")].append(d.get("id") or "?")
            continue
        name = d.get("name") or ""
        norm = normalize_name(name)
        if norm:
            by_name[norm].append(name)
            m = YEAR_SUFFIX.match(norm)
            if m:
                by_prefix[m.group("prefix")].append(name)
    out = {"same hash": [v for v in by_hash.values() if len(v) > 1],
           "same name": [v for v in by_name.values() if len(v) > 1]}
    out["same name but the year"] = [v for v in by_prefix.values() if len(set(v)) > 1
                                     and not any(set(v) <= set(g) for g in out["same name"])]
    out["same ruling file"] = [v for v in by_stem.values() if len(v) > 1]
    return out


def branch_of(file_name: str, name: str) -> str:
    m = ITALGIURE.search(file_name or "") or ITALGIURE.search(name or "")
    return ("penale" if m.group(1).lower() == "snpen" else "civile") if m else "unknown"


def summary_line(n: dict, prev: Optional[dict]) -> str:
    """The morning line; '(+N)' against the previous night where it means something."""
    def delta(key):
        if not prev or key not in prev:
            return ""
        d = n[key] - prev[key]
        return f" ({'+' if d >= 0 else ''}{d})" if d else ""
    parts = [f"v2 missing {n['sections_no_v2_with_text']:,}{delta('sections_no_v2_with_text')}",
             f"empty sections {n['sections_empty']:,}",
             f"untyped docs {n['docs_untyped']:,}",
             f"detached sections {n['sections_detached']:,}{delta('sections_detached')}",
             f"date-only articles {n['date_only_articles']:,}",
             f"numbering gaps {n['numbering_gaps']:,} (new {n['numbering_gaps_new']:,})",
             f"rulings without CITES {n['rulings_without_cites']:,}{delta('rulings_without_cites')}",
             f"queue {n['queue_waiting']:,} waiting / {n['queue_failed']:,} failed",
             f"suspected duplicates {n['duplicates']:,} (not touched)"]
    if n.get("rulings_without_card") is not None:
        parts.insert(7, f"rulings without card {n['rulings_without_card']:,}{delta('rulings_without_card')}")
    return " · ".join(parts)


# --- machine checks ------------------------------------------------------------------------------

def mem_available_gb(path: str = "/proc/meminfo") -> Optional[float]:
    try:
        with open(path) as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        return None
    return None


def running_job(ps_lines: Iterable[str], me: int) -> Optional[str]:
    """In `ps -eo pid=,args=` lines: the first heavy job a Python process is running, "job (pid N)", or None. A job
    named by something else (a grep, a watcher, an editor) does not count: on be2 one kept every run skipped (29 Oct)."""
    for line in ps_lines:
        pid, _, args = line.strip().partition(" ")
        argv = args.split()
        if pid == str(me) or not argv or "python" not in os.path.basename(argv[0]) or "maint_health.py" in args:
            continue
        for job in HEAVY_JOBS:
            if any(job in a for a in argv[1:4]):
                return f"{job} (pid {pid})"
    return None


def busy_job() -> Optional[str]:
    """The first heavy job running now, or None."""
    try:
        out = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    return running_job(out.splitlines(), os.getpid())


def queue_status(queue_dir: str, failed_dir: str) -> dict:
    files = glob.glob(os.path.join(queue_dir, "**", "*.pdf"), recursive=True) if os.path.isdir(queue_dir) else []
    failed = glob.glob(os.path.join(failed_dir, "**", "*"), recursive=True) if os.path.isdir(failed_dir) else []
    kinds = Counter("penale" if "snpen" in f else "civile" if "snciv" in f else "other" for f in files)
    try:
        service = subprocess.run(["systemctl", "is-active", "sunnitai-ingest"], capture_output=True, text=True,
                                 timeout=10).stdout.strip() or "unknown"
    except Exception:
        service = "unknown"
    return {"waiting": len(files), "by_kind": dict(kinds), "failed": sum(1 for f in failed if os.path.isfile(f)),
            "service": service, "queue_dir_found": os.path.isdir(queue_dir)}


def log_status(paths: List[str]) -> List[dict]:
    out = []
    for p in paths:
        if not os.path.isfile(p):
            out.append({"log": p, "found": False})
            continue
        age = (time.time() - os.path.getmtime(p)) / 3600
        last = ""
        try:
            with open(p, "rb") as fh:
                fh.seek(max(0, os.path.getsize(p) - 4000))
                lines = [l.strip() for l in fh.read().decode("utf-8", "replace").splitlines() if l.strip()]
                last = lines[-1][:200] if lines else ""
        except OSError:
            pass
        out.append({"log": p, "found": True, "hours_since_write": round(age, 1), "last_line": last})
    return out


def unresolved_reasons(path: str) -> Tuple[Optional[int], Dict[str, int]]:
    if not os.path.isfile(path):
        return None, {}
    reasons, n = Counter(), 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            n += 1
            try:
                reasons[json.loads(line).get("reason") or "?"] += 1
            except ValueError:
                reasons["unparseable"] += 1
    return n, dict(reasons.most_common(8))


def missing_articles(path: str, top: int = 40) -> Optional[List[Tuple[str, int]]]:
    """The citations the linker finds no article for ("410-bis c.p.p."), most cited first; None without the log."""
    if not os.path.isfile(path):
        return None
    counts = Counter()
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("reason") == "article_not_in_code" and row.get("citation"):
                counts[row["citation"]] += 1
    return counts.most_common(top)


def closures(path: str) -> Set[str]:
    """The civil rulings step 1 reads as closures ("estinto"): no card by design (exp_ruling_cards.pick)."""
    out = set()
    if not path or not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("branch") == "civile" and r.get("outcome") == "estinto" and r.get("id"):
                out.add(r["id"])
    return out


def card_ids(paths: List[str]) -> Optional[Set[str]]:
    paths = [p for p in paths if p and os.path.isfile(p)]
    if not paths:
        return None
    ids = set()
    for p in paths:
        with open(p, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    c = json.loads(line)
                except ValueError:
                    continue
                if isinstance(c, dict) and c.get("id") and not c.get("errore"):
                    ids.add(c["id"])
    return ids


# --- graph checks (read-only) --------------------------------------------------------------------

def _progress(t0: float, message: str) -> None:
    print(f"[{time.time() - t0:5.0f}s] {message}", file=sys.stderr, flush=True)


def graph_checks(session, batch: int, pause: float, skip: Iterable[str] = ()) -> dict:
    """The graph's numbers. `skip` leaves out the heavy stages: "sections" (every Section's vector and text, about 20
    minutes on prod's 450k Sections, 28 Oct), "detached" (one count over all Sections)."""
    skip, t0 = set(skip), time.time()
    out: dict = {"skipped": sorted(skip)}
    docs = session.run("MATCH (d:Document) RETURN d.id AS id, d.name AS name, d.hash AS hash, "
                       "d.document_type AS type, coalesce(d.file_name, '') AS file").data()
    out["documents"] = len(docs)
    out["docs_by_type"] = dict(Counter(d["type"] or "(none)" for d in docs).most_common())
    out["docs_untyped_ids"] = sorted(d["id"] for d in docs if not d["type"])[:30]
    out["docs_unknown_type"] = {t: k for t, k in out["docs_by_type"].items() if t != "(none)" and t not in KNOWN_TYPES}

    _progress(t0, f"{len(docs):,} documents read")
    # sections without embedding_v2 / without text, by document type, a batch of documents at a time. The text is
    # only tested for presence (IS NULL), never read: reading every Section's text made this stage take 20+ minutes.
    no_v2, no_v2_text, empty, sections = Counter(), Counter(), Counter(), Counter()
    ids = [d["id"] for d in docs] if "sections" not in skip else []
    for start in range(0, len(ids), batch):
        if start:
            time.sleep(pause)
            if (start // batch) % 20 == 0:
                _progress(t0, f"sections: {start:,}/{len(ids):,} documents")
        for r in session.run(
                "MATCH (d:Document) WHERE d.id IN $ids MATCH (d)-[:CONTAINS]->(s:Section) "
                "WITH d.document_type AS t, s, (s.plain_text IS NULL AND s.abstract IS NULL) AS empty "
                "RETURN t, count(s) AS n, sum(CASE WHEN empty THEN 1 ELSE 0 END) AS empty, "
                "sum(CASE WHEN s.embedding_v2 IS NULL THEN 1 ELSE 0 END) AS no_v2, "
                "sum(CASE WHEN s.embedding_v2 IS NULL AND NOT empty THEN 1 ELSE 0 END) AS no_v2_text",
                ids=ids[start:start + batch]).data():
            t = r["t"] or "(none)"
            sections[t] += r["n"]
            empty[t] += r["empty"]
            no_v2[t] += r["no_v2"]
            no_v2_text[t] += r["no_v2_text"]
    out.update({"sections_by_type": dict(sections), "sections_empty_by_type": dict(empty),
                "sections_no_v2_by_type": dict(no_v2), "sections_no_v2_with_text_by_type": dict(no_v2_text)})

    _progress(t0, "sections done" if "sections" not in skip else "sections skipped")
    out["sections_detached"] = 0 if "detached" in skip else session.run(
        "MATCH (s:Section) WHERE NOT ()-[:CONTAINS|PART_OF]->(s) RETURN count(s) AS n").single()["n"]
    _progress(t0, "detached count done" if "detached" not in skip else "detached count skipped")

    # per primary document: date-only articles, and numbering gaps for the main codes
    primary = [d for d in docs if d["type"] == "primary"]
    main = {}
    for prefix in MAIN_CODES:
        hits = [d for d in primary if (d["name"] or "").startswith(prefix)
                and not any((d["name"] or "").startswith(p) for p in MAIN_CODES if p != prefix and len(p) > len(prefix))]
        if hits:
            main[prefix] = min(hits, key=lambda d: len(d["name"] or ""))["id"]
    date_only, gaps, short_first = {}, {}, {}
    for d in primary:
        time.sleep(pause / 4)
        rows = session.run("MATCH (d:Document {id: $id})-[:CONTAINS]->(s:Section) "
                           "RETURN s.name AS name, size(coalesce(s.plain_text, '')) AS n, "
                           "left(coalesce(s.plain_text, ''), 45) AS head", id=d["id"]).data()
        sections_ = [(r["name"] or "", r["n"], bool(IN_FORCE.match(r["head"] or "")) and r["n"] <= 45) for r in rows]
        found = date_only_articles(sections_)
        if found:
            date_only[d["name"]] = len(found)
        short = short_first_pieces(sections_)
        if short:
            short_first[d["name"]] = len(short)
        if d["id"] in main.values():
            gaps[d["name"]] = numbering_gaps({article_of(n) for n, _, _ in sections_} - {None})
    out["date_only_by_doc"] = dict(sorted(date_only.items(), key=lambda x: -x[1]))
    out["short_first_by_doc"] = dict(sorted(short_first.items(), key=lambda x: -x[1]))
    _progress(t0, f"{len(primary)} codes and laws done")
    out["numbering_gaps_by_code"] = gaps

    # rulings without CITES, by branch
    rulings = [d for d in docs if d["type"] == "interpretation"]
    cited = {r["id"] for r in session.run(
        "MATCH (d:Document {document_type:'interpretation'})-[:CONTAINS]->(:Section)-[:CITES]->() "
        "RETURN DISTINCT d.id AS id").data()}
    by_branch = defaultdict(lambda: [0, 0])
    for d in rulings:
        b = branch_of(d["file"], d["name"])
        by_branch[b][0] += 1
        by_branch[b][1] += d["id"] not in cited
    out["rulings_by_branch"] = {b: {"rulings": v[0], "without_cites": v[1]} for b, v in by_branch.items()}
    out["ruling_ids_by_branch"] = {b: [d["id"] for d in rulings if branch_of(d["file"], d["name"]) == b]
                                   for b in by_branch}
    out["duplicates"] = suspected_duplicates(docs)
    _progress(t0, "rulings and duplicates done")
    return out


# --- report --------------------------------------------------------------------------------------

def previous_state(directory: str, today: str) -> Optional[dict]:
    files = sorted(f for f in glob.glob(os.path.join(directory, "state_*.json")) if not f.endswith(f"_{today}.json"))
    if not files:
        return None
    try:
        return json.load(open(files[-1], encoding="utf-8"))
    except (OSError, ValueError):
        return None


def build_report(g: dict, queue: dict, logs: List[dict], unresolved: Tuple[Optional[int], Dict[str, int]],
                 cards: Optional[Set[str]], prev: Optional[dict], took: float,
                 by_design: Optional[Set[str]] = None,
                 missing: Optional[List[Tuple[str, int]]] = None) -> Tuple[str, dict, str]:
    """by_design: the rulings left without a card on purpose (civil closures, "estinto" in step 1's links).
    missing: missing_articles(), the citations with no article in our codes."""
    by_design = by_design or set()
    prev_gaps = (prev or {}).get("gaps", {})
    gaps_new = {code: sorted(set(g_) - set(prev_gaps.get(code, g_))) for code, g_ in g["numbering_gaps_by_code"].items()}
    n = {"sections_no_v2_with_text": sum(g["sections_no_v2_with_text_by_type"].values()),
         "sections_empty": sum(g["sections_empty_by_type"].values()),
         "docs_untyped": g["docs_by_type"].get("(none)", 0),
         "sections_detached": g["sections_detached"],
         "date_only_articles": sum(g["date_only_by_doc"].values()),
         "numbering_gaps": sum(len(v) for v in g["numbering_gaps_by_code"].values()),
         "numbering_gaps_new": sum(len(v) for v in gaps_new.values()),
         "rulings_without_cites": sum(v["without_cites"] for v in g["rulings_by_branch"].values()),
         "queue_waiting": queue["waiting"], "queue_failed": queue["failed"],
         "duplicates": sum(len(v) for v in g["duplicates"].values()),
         "rulings_without_card": None}
    card_rows = {}
    if cards is not None:
        for b, ids in g["ruling_ids_by_branch"].items():
            card_rows[b] = {"rulings": len(ids), "with_card": sum(i in cards for i in ids),
                            "skipped_by_design": sum(i in by_design and i not in cards for i in ids)}
        n["rulings_without_card"] = sum(v["rulings"] - v["with_card"] - v["skipped_by_design"]
                                        for b, v in card_rows.items() if b != "unknown")
    line = summary_line(n, (prev or {}).get("numbers"))
    if g.get("skipped"):
        line += f" · skipped: {', '.join(g['skipped'])}"
    md = [f"# Graph health, {dt.date.today().isoformat()}", "", f"**Summary:** {line}", "",
          f"Read-only. {g['documents']:,} documents. Took {took:.0f}s." + ("" if prev else " First report: no comparison."),
          "", "## Documents by type", "", "| type | documents | sections | empty | no embedding_v2 (with text) |",
          "|---|---|---|---|---|"]
    for t, k in g["docs_by_type"].items():
        md.append(f"| {t} | {k:,} | {g['sections_by_type'].get(t, 0):,} | {g['sections_empty_by_type'].get(t, 0):,} | "
                  f"{g['sections_no_v2_with_text_by_type'].get(t, 0):,} |")
    if g["docs_untyped_ids"]:
        md += ["", "Untyped documents (ids; scripts/document_types.py proposes a type): " + "; ".join(g["docs_untyped_ids"])]
    if g["docs_unknown_type"]:
        md += ["", "Unknown types: " + ", ".join(f"{t} {k}" for t, k in g["docs_unknown_type"].items())]
    md += ["", f"Sections detached from any document: {g['sections_detached']:,}", "",
           "## Articles holding only their date in force (N.0.0 = \"In vigore dal ...\")", ""]
    md += [f"- {doc}: {k:,}" for doc, k in g["date_only_by_doc"].items()] or ["- none"]
    md += ["", "## Articles whose first piece (N.0.0) is short but not a date", "",
           "The shared read rule takes such a piece as the article's text: check what it holds.", ""]
    md += [f"- {doc}: {k:,}" for doc, k in (g.get("short_first_by_doc") or {}).items()] or ["- none"]
    md += ["", "## Article numbering of the main codes", ""]
    for code, gl in g["numbering_gaps_by_code"].items():
        new = gaps_new.get(code) or []
        md.append(f"- {code}: {len(gl):,} missing numbers" + (f" -- **NEW since the last report: {', '.join(map(str, new[:40]))}**"
                                                              if new and prev else "")
                  + (f" (first: {', '.join(map(str, gl[:25]))})" if gl else ""))
    md += ["", "## Rulings", "", "| branch | rulings | without CITES |"
           + (" with a card | skipped by design (closures) | still to card |" if cards is not None else ""),
           "|---|---|---|" + ("---|---|---|" if cards is not None else "")]
    for b, v in g["rulings_by_branch"].items():
        row = card_rows.get(b) or {}
        md.append(f"| {b} | {v['rulings']:,} | {v['without_cites']:,} |"
                  + (f" {row.get('with_card', 0):,} | {row.get('skipped_by_design', 0):,} | "
                     f"{row.get('rulings', 0) - row.get('with_card', 0) - row.get('skipped_by_design', 0):,} |"
                     if cards is not None else ""))
    if cards is None:
        md += ["", "Ruling cards: no card file readable here (they live on prod in /root/ruling_cards today)."]
    md += ["", "CITES unresolved citations (linker log): " + ("log not found on this machine" if unresolved[0] is None
                                                             else f"{unresolved[0]:,}")
           + (" -- " + ", ".join(f"{r} {k:,}" for r, k in unresolved[1].items()) if unresolved[1] else "")]
    if missing is not None:
        md += ["", "## Articles the rulings cite that our codes lack", "",
               "Most cited first. The linker finds no such article in the code named: missing from our text, stored "
               "under another name, or the ruling's own wrong code (\"131-bis c.p.p.\" for c.p.).", ""]
        md += [f"- {c}: {k:,}" for c, k in missing] or ["- none"]
    md += ["", "## Suspected duplicate documents (REPORTED ONLY, never touched)", ""]
    for reason, groups in g["duplicates"].items():
        md.append(f"- {reason}: {len(groups)}")
        md += [f"  - {' | '.join(names[:4])}" for names in groups[:15]]
    md += ["", "## Ingestion queue and night jobs", "",
           f"- queue: {queue['waiting']:,} files waiting " + (str(queue['by_kind']) if queue['by_kind'] else "")
           + f"; failed {queue['failed']:,}; service sunnitai-ingest {queue['service']}"
           + ("" if queue["queue_dir_found"] else " (queue folder not found on this machine)")]
    for l in logs:
        md.append(f"- {l['log']}: " + (f"written {l['hours_since_write']} h ago; last line: {l['last_line']}"
                                        if l["found"] else "not found"))
    state = {"date": dt.date.today().isoformat(), "numbers": n, "gaps": g["numbering_gaps_by_code"],
             "date_only_by_doc": g["date_only_by_doc"], "rulings_by_branch": g["rulings_by_branch"],
             "cards": card_rows, "queue": queue, "missing_articles": dict(missing or [])}
    return "\n".join(md) + "\n", state, line


def main() -> int:
    p = argparse.ArgumentParser(description="Maintenance agent, job 1: nightly graph health report (read-only)")
    p.add_argument("--env", default="/opt/chatbot/.env", help="the .env with NEO4J_URI / USER / PASSWORD / DATABASE")
    p.add_argument("--dir", default="/var/log/maint")
    p.add_argument("--min-gb", type=float, default=2.0, help="skip when MemAvailable is below this")
    p.add_argument("--batch", type=int, default=300, help="documents per query")
    p.add_argument("--pause", type=float, default=0.3, help="seconds between queries")
    p.add_argument("--queue-dir", default="/opt/sunnitai-be/document_upload")
    p.add_argument("--failed-dir", default="/opt/sunnitai-be/document_failed")
    p.add_argument("--logs", default="/var/log/backfill_v2_nightly.log,/var/log/link_rulings_nightly.log")
    p.add_argument("--unresolved", default="/var/log/ruling_citations_unresolved.jsonl")
    p.add_argument("--cards", default="", help="comma-separated card files (optional)")
    p.add_argument("--links", default="/root/ruling_cards/ruling_links.jsonl",
                   help="step 1's links: the civil closures ('estinto') are left without a card by design")
    p.add_argument("--force", action="store_true", help="run even if memory is low or a heavy job is running")
    p.add_argument("--skip", default="", help="heavy stages to leave out: sections,detached")
    args = p.parse_args()
    os.makedirs(args.dir, exist_ok=True)
    today = dt.date.today().strftime("%Y%m%d")
    summary_path = os.path.join(args.dir, f"summary_{today}.txt")

    mem, busy = mem_available_gb(), busy_job()
    if not args.force and ((mem is not None and mem < args.min_gb) or busy):
        why = f"memory available {mem:.1f} GB" if mem is not None and mem < args.min_gb else f"{busy} running"
        with open(summary_path, "w", encoding="utf-8") as fh:
            fh.write(f"health report SKIPPED: {why}\n")
        print(f"skipped: {why}")
        return 0

    try:
        from dotenv import load_dotenv
        load_dotenv(args.env)
    except ImportError:
        pass
    from neo4j import GraphDatabase, READ_ACCESS
    t0 = time.time()
    driver = GraphDatabase.driver(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")))
    try:
        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j"), default_access_mode=READ_ACCESS) as session:
            g = graph_checks(session, args.batch, args.pause, [x for x in args.skip.split(",") if x])
    finally:
        driver.close()
    md, state, line = build_report(g, queue_status(args.queue_dir, args.failed_dir),
                                   log_status([x for x in args.logs.split(",") if x]), unresolved_reasons(args.unresolved),
                                   card_ids([x for x in args.cards.split(",") if x]),
                                   previous_state(args.dir, today), time.time() - t0, closures(args.links),
                                   missing_articles(args.unresolved))
    for name, content in ((f"health_{today}.md", md), (f"summary_{today}.txt", line + "\n")):
        with open(os.path.join(args.dir, name), "w", encoding="utf-8") as fh:
            fh.write(content)
    with open(os.path.join(args.dir, f"state_{today}.json"), "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, indent=1)
    print(line)
    print(f"report: {os.path.join(args.dir, f'health_{today}.md')} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
