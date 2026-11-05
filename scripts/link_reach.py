"""Offline: are the expected norms a run misses one hop away from an article the case's prompt held? (v48, before the
link builder writes anything.)

    python scripts/link_reach.py [DIR] [--runs v46-candidate v47/v47-candidate ...] [--stable-runs ...]

DIR (default ./diagnose_reports) holds article_links_out/ (units.csv and refers_to.csv, scripts/article_links.py's dry
run on the live graph) and the runs' report.json. For each run (default: v46-candidate, v47/v47-candidate and
v48-candidate when there) and each expected norm whose stage is not MENTIONED: the anchors are the articles in that case's
prompt (in_prompt true in its slots or pool_traces); a miss is reachable by
  same Sezione  an anchor of the same code in the same Sezione (its whole path: Libro, Parte, Titolo, Capo, Sezione),
  same Capo     an anchor of the same code in the same Capo (its path down to the Capo),
  REFERS_TO     an edge between the miss and an anchor, in either direction (any code).
Prints per run the misses reachable by each hop and in total, the list (each with its anchors by hop), and the same for
the stable misses: those not MENTIONED in every run of --stable-runs (default v37-cards-off, v45-candidate,
v46-candidate). Nothing is written.

Limits of the inputs: units.csv has one row per article number and code (scripts/article_links.py keeps an article's
first Section), so two articles sharing a number in one document (c.p.c.'s 1-4) count as one; it holds only the
documents whose text carries heading markers; an expected norm of another document can still be reached by REFERS_TO.
"""

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict

DEFAULT_RUNS = ("v46-candidate", "v47/v47-candidate", "v48-candidate")
DEFAULT_STABLE = ("v37-cards-off", "v45-candidate", "v46-candidate")
LEVELS = ("LIBRO", "PARTE", "TITOLO", "CAPO", "SEZIONE")
HOPS = ("same Sezione", "same Capo", "REFERS_TO")
# the answer keys' code words -> the graph's code labels (scripts/article_links.py's code_of; other documents by name)
EVAL_CODES = {"cc": "c.c.", "cp": "c.p.", "cpc": "c.p.c.", "cpp": "c.p.p.", "cpa": "c.p.a.", "cds": "c.d.s.",
              "l241": "Legge sul Procedimento Amministrativo 2026", "dlgs36": "Codice degli Appalti 2026",
              "consumo": "Codice del Consumo 2026", "tuel": "Testo unico degli enti locali 2026",
              "privacy": "Codice della Privacy 2026"}
_LONG_CODES = {"codice civile": "c.c.", "codice penale": "c.p.", "codice di procedura civile": "c.p.c.",
               "codice di procedura penale": "c.p.p.", "codice del processo amministrativo": "c.p.a.",
               "codice della strada": "c.d.s."}


_SUFFIXES = ("bis|ter|quater|quinquies|sexies|septies|octies|novies|nonies|decies|undecies|duodecies|terdecies|"
             "quaterdecies|quinquiesdecies|sexiesdecies|septiesdecies|octiesdecies")


def norm_number(n: str) -> str:
    return re.sub(r"[\s-]+", "-", str(n).strip().lower())


def code_label(code: str) -> str:
    low = (code or "").strip().lower()
    for prefix, short in _LONG_CODES.items():
        if low.startswith(prefix):
            return short
    return EVAL_CODES.get(low, (code or "").strip())


def parse_label(label: str):
    """"584 c.p." / "229 Codice degli Appalti 2026" -> ("584", "c.p.")."""
    m = re.match(rf"^(?:artt?\.\s*)?(\d+(?:[\s-]*(?:{_SUFFIXES})\b)?)\s+(.+)$", (label or "").strip(), re.IGNORECASE)
    return (norm_number(m.group(1)), code_label(m.group(2))) if m else None


def slot_articles(slot: dict):
    """An expected norm's articles: its pool label when it reached the pool, else its key ("art. 582/583 cp")."""
    if slot.get("pool_label"):
        ref = parse_label(slot["pool_label"])
        if ref:
            return [ref]
    m = re.match(r"^art\.\s*([\w/-]+)\s+(\S+)$", slot.get("label") or "")
    if not m:
        return []
    return [(norm_number(n), code_label(m.group(2))) for n in m.group(1).split("/") if n]


# --- the links --------------------------------------------------------------------------------------------------------

def load_units(path: str):
    """{(article, code): {level: unit name}}."""
    units = defaultdict(dict)
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            units[(norm_number(row["article"]), row["code"])][row["level"]] = row["unit name"]
    return dict(units)


def unit_key(units: dict, ref, level: str):
    """The article's path down to `level` (that level included), None when it has no unit at that level."""
    path = units.get(ref) or {}
    if level not in path:
        return None
    return (ref[1],) + tuple(path.get(l, "") for l in LEVELS[:LEVELS.index(level) + 1])


def load_edges(path: str):
    """{(article, code): set of (article, code)} REFERS_TO, both directions."""
    edges = defaultdict(set)
    with open(path, encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            a, b = parse_label(row[0]), parse_label(row[1])
            if a and b and a != b:
                edges[a].add(b)
                edges[b].add(a)
    return edges


# --- the runs ---------------------------------------------------------------------------------------------------------

def load_run(root: str, name: str):
    path = os.path.join(root, name, "report.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return {c["id"]: c for c in json.load(fh) if isinstance(c, dict) and c.get("id")}


def anchors(case: dict) -> set:
    out = set()
    for item in (case.get("pool_traces") or []) + (case.get("slots") or []):
        if item.get("in_prompt"):
            ref = parse_label(item.get("pool_label") or item.get("label"))
            if ref:
                out.add(ref)
    return out


def reach(ref, anchor_set: set, units: dict, edges: dict) -> dict:
    """{hop: sorted anchors} for one missed article."""
    out = {}
    for hop, level in (("same Sezione", "SEZIONE"), ("same Capo", "CAPO")):
        key = unit_key(units, ref, level)
        if key:
            found = sorted(f"{a[0]} {a[1]}" for a in anchor_set if a != ref and unit_key(units, a, level) == key)
            if found:
                out[hop] = found
    linked = sorted(f"{a[0]} {a[1]}" for a in edges.get(ref, set()) & anchor_set if a != ref)
    if linked:
        out["REFERS_TO"] = linked
    return out


def misses(run: dict, units: dict, edges: dict, only=None) -> list:
    """[{case, label, stage, refs, placed, hops}] of the run's expected norms not MENTIONED (only: those keys)."""
    out = []
    for cid, case in run.items():
        held = anchors(case)
        for slot in case.get("slots") or []:
            if slot.get("stage") == "MENTIONED" or (only is not None and (cid, slot.get("label")) not in only):
                continue
            refs = slot_articles(slot)
            hops: dict = {}
            for ref in refs:
                for hop, found in reach(ref, held, units, edges).items():
                    hops.setdefault(hop, [])
                    hops[hop] += [a for a in found if a not in hops[hop]]
            out.append({"case": cid, "label": slot.get("label"), "stage": slot.get("stage"), "refs": refs,
                        "placed": any(r in units for r in refs), "hops": hops, "anchors": len(held)})
    return out


def report(title: str, found: list) -> list:
    by_hop = Counter(h for m in found for h in m["hops"])
    reachable = [m for m in found if m["hops"]]
    lines = [f"### {title}", "",
             f"{len(found)} misses; reachable in one hop: {len(reachable)} "
             f"({', '.join(f'{h} {by_hop[h]}' for h in HOPS)}; a miss may count under several hops). "
             f"With a Libro/Titolo/Capo/Sezione path in units.csv: {sum(1 for m in found if m['placed'])}.", ""]
    if reachable:
        lines += ["| case | missed norm | stage | hop | anchors in the prompt |", "|---|---|---|---|---|"]
        for m in reachable:
            for hop in HOPS:
                if hop in m["hops"]:
                    shown = m["hops"][hop]
                    lines.append(f"| {m['case']} | {m['label']} | {m['stage']} | {hop} | "
                                 + ", ".join(shown[:6]) + (f" (+{len(shown) - 6})" if len(shown) > 6 else "") + " |")
    unreached = [m for m in found if not m["hops"]]
    if unreached:
        lines += ["", "Not reachable: " + "; ".join(
            f"{m['case']} {m['label']} ({m['stage']}{', no unit path' if not m['placed'] else ''})" for m in unreached)]
    return lines + [""]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dir", nargs="?", default="diagnose_reports")
    parser.add_argument("--links", default=None, help="default: DIR/article_links_out")
    parser.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    parser.add_argument("--stable-runs", nargs="+", default=list(DEFAULT_STABLE))
    args = parser.parse_args()
    links = args.links or os.path.join(args.dir, "article_links_out")
    units = load_units(os.path.join(links, "units.csv"))
    edges = load_edges(os.path.join(links, "refers_to.csv"))
    print(f"# Link reach of the missed expected norms\n\nunits.csv: {len(units)} articles with a unit path; "
          f"refers_to.csv: {sum(len(v) for v in edges.values()) // 2} distinct links between {len(edges)} articles.\n")
    stable_runs = {name: load_run(args.dir, name) for name in args.stable_runs}
    missing = [n for n, r in stable_runs.items() if r is None]
    stable = None
    if not missing:
        sets = [{(cid, s.get("label")) for cid, c in r.items() for s in c.get("slots") or [] if s.get("stage") != "MENTIONED"}
                for r in stable_runs.values()]
        stable = set.intersection(*sets)
    print(f"Stable misses (not MENTIONED in every one of {', '.join(args.stable_runs)}): "
          + (f"{len(stable)}." if stable is not None else f"not computed, missing {', '.join(missing)}.") + "\n")
    for name in args.runs:
        run = load_run(args.dir, name)
        if run is None:
            print(f"## {name}\n\n(no report.json yet)\n")
            continue
        print(f"## {name}\n")
        print("\n".join(report("All misses", misses(run, units, edges))))
        if stable is not None:
            print("\n".join(report(f"The {len(stable)} stable misses", misses(run, units, edges, only=stable))))


if __name__ == "__main__":
    main()
