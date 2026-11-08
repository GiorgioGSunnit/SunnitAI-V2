"""Offline: POOL_GATE_AREA and POOL_GATE_LINK simulated on each case's pool (pool_traces of the runs' report.json),
each gate on its own -- per case the rows it would remove and the expected norms lost -- and the citations from outside
the material (v50).

    python scripts/pool_gates_sim.py [DIR] [--runs v46-candidate v49-candidate] [--texts DIR/v47/v47-candidate]
                                     [--hub 100] [--links DIR/article_links_out] [--before]

DIR (default ./diagnose_reports) holds article_links_out/ (units.csv, refers_to.csv) and each run's report.json. Per
case, the candidates are the pool's articles (in_pool, not an EDGE_EXPAND addition), each with
  channels  the fusion signals it carries (the trace's four, and a fifth from its hit count), the vote,
            cross-reference, method list, regime candidate, issue spotting (pool_gates.channels' rule);
  named     the user_text channel; exempt (link gate) a heading match, a cross-reference, a regime candidate (a
            procedural tool is not in the pool the traces show), and pool_gates.UNDERSTANDING's channels.
The gates are src/rag/pool_gates' own: area by pool_gates.allowed_codes(gate_areas, the codes the case text names --
read from the run's or --texts' input specs, cases/<id>.input.json or cases/<id>/input.json); link with
pool_gates.CsvLinks on the CSVs, --hub the hub threshold. Each gate is applied to the whole pool alone; "both" is the
area gate then the link gate on what it leaves. --before: v50's first rules (vote_pool's area mapping; the link gate's
exemptions only heading and cross-reference). Dedup: the traces are one per article already (diagnose_long keys them),
so dedup counts the pool's articles sharing a (code, article) under two labels.
An expected norm is lost when it was in the pool and every one of its pool articles is gated out; "of them MENTIONED"
counts those the run's answer cites. Removed rows the answer cites unexpected are the noise the gate would have kept
out of the answer. Last, per run: the articles cited from outside the material (unexpected, not in the pool), those
closure verified and those it read as offered material (v49), and the rest."""

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

for _k, _v in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
               ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(_k, _v)

import link_reach  # noqa: E402
from src.rag import pool_gates as pg, vote_pool  # noqa: E402

TRACE_CHANNELS = {"vote": "vote", "crossref": "crossref", "method_list": "method_list",
                  "regime_candidate": "regime_candidate", "issue_spotting": "issue_spotting"}


def trace_candidate(t: dict, before: bool = False):
    """(key, the gates' candidate) of a pool trace, or None (no article, an EDGE_EXPAND addition)."""
    ref = link_reach.parse_label(t.get("label"))
    chans = list(t.get("channels") or [])
    if not ref or not t.get("in_pool") or (chans and all(c.startswith("edge:") for c in chans)):
        return None
    channels = {s for s in t.get("signals") or [] if not str(s).startswith("edge")}
    # the trace names four fusion signals; a fifth (other_search) shows in hits only
    channels |= {f"signal {i}" for i in range(len(channels), int(t.get("hits") or 0))}
    channels |= {name for c, name in TRACE_CHANNELS.items() if c in chans}
    check = str(t.get("check") or "")
    k = pg.key(ref)
    exempt = (check.startswith("exempt: heading") or check.startswith("exempt: crossref") or "crossref" in chans
              or (not before and "regime_candidate" in chans))
    return k, {"key": k, "channels": channels, "named": "user_text" in chans, "exempt": exempt}


def case_text(dirs, cid: str) -> str:
    for d in dirs:
        for path in (os.path.join(d, "cases", cid, "input.json"), os.path.join(d, "cases", f"{cid}.input.json")):
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as fh:
                    spec = json.load(fh)
                return " ".join(str(spec.get(f) or "") for f in ("question", "request"))
    return ""


def _verdict(check: str) -> str:
    return ("rejected" if check == "rejected" or check.startswith("grade 0") else
            "unchecked" if check.startswith("unchecked") else
            "graded 2-3" if check.startswith(("grade 2", "grade 3")) else
            "grade 1" if check.startswith("grade 1") else "exempt or other")


def simulate(case: dict, lookup, text: str, hub: int, before: bool = False) -> dict:
    """Per gate ("area", "link" alone, "both"): {"removed", "lost", "noise", "verdicts"}; and the pool, dedup, areas."""
    rows, labels, verdict = [], Counter(), {}
    for t in case.get("pool_traces") or []:
        got = trace_candidate(t, before)
        if got:
            labels[got[0]] += 1
            rows.append(got[1])
            verdict[got[0]] = _verdict(str(t.get("check") or ""))
    cands = pg.merge(rows)
    frame = {"frame": {"areas": case.get("areas") or []}, "area": case.get("area") or ""}
    named_codes = vote_pool.named_codes(text) if text else {k[1] for k, c in cands.items() if c["named"]}
    if before:
        areas = vote_pool.case_areas(frame)
        allowed = vote_pool.allowed_codes(areas, named_codes)
    else:
        areas = pg.gate_areas(frame)
        allowed = pg.allowed_codes(areas, named_codes)
    saved = set(pg.UNDERSTANDING)
    if before:
        pg.UNDERSTANDING.clear()
    try:
        both = pg.plan_gates(cands, allowed, lookup, hub)
        gone = {"area": set(both["area"]), "link": set(pg.plan_gates(cands, allowed, lookup, hub, area=False)["link"]),
                "both": set(both["area"]) | set(both["link"])}
    finally:
        pg.UNDERSTANDING.update(saved)
    cited = {pg.key(ref) for u in case.get("unexpected") or [] if u.get("in_pool")
             for ref in [link_reach.parse_label(u.get("label"))] if ref}
    out = {"pool": len(cands), "dedup": sum(n - 1 for n in labels.values() if n > 1), "allowed": allowed,
           "areas": areas, "text": bool(text)}
    for gate, removed in gone.items():
        lost = []
        for slot in case.get("slots") or []:
            held = [k for k in (pg.key(r) for r in link_reach.slot_articles(slot)) if k in cands]
            if held and all(k in removed for k in held):
                lost.append((slot.get("label"), slot.get("stage")))
        out[gate] = {"removed": sorted(removed), "lost": lost, "noise": len(removed & cited),
                     "verdicts": Counter(verdict[k] for k in removed)}
    return out


def outside(run: dict) -> dict:
    """The unexpected citations with no pool row: offered material (a tool, a term, a procedural list, the case's own:
    diagnose_long's "material", v50; a report before it reads none), closure verified, offered rows read, rest."""
    def k(label):
        ref = link_reach.parse_label(label)
        return pg.key(ref) if ref else label
    total, verified, offered, material, bare = 0, 0, 0, 0, []
    for cid, case in run.items():
        cl = case.get("closure") or {}
        ver = {k(x) for x in cl.get("verified") or []}
        off = {k(x) for x in cl.get("offered_rows") or []}
        for u in case.get("unexpected") or []:
            if u.get("in_pool"):
                continue
            total += 1
            if u.get("material") in ("tool", "term", "procedural", "named"):
                material += 1
            elif k(u["label"]) in off:
                offered += 1
            elif k(u["label"]) in ver:
                verified += 1
            else:
                bare.append(f"{cid} {u['label']}")
    return {"total": total, "verified": verified, "offered": offered, "material": material, "bare": bare,
            "material_logged": any("material" in u for c in run.values() for u in c.get("unexpected") or []),
            "offered_logged": any("offered_rows" in (c.get("closure") or {}) for c in run.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dir", nargs="?", default="diagnose_reports")
    parser.add_argument("--runs", nargs="+", default=["v46-candidate", "v49-candidate"])
    parser.add_argument("--texts", nargs="*", default=None, help="run folders with the input specs (default: each run's "
                                                                 "own, then DIR/v47/v47-candidate)")
    parser.add_argument("--hub", type=int, default=pg.HUB_INBOUND)
    parser.add_argument("--links", default=None, help="folder of units.csv and refers_to.csv (default: "
                                                      "DIR/article_links_out)")
    parser.add_argument("--before", action="store_true", help="v50's first rules, for comparison")
    args = parser.parse_args()
    links = args.links or os.path.join(args.dir, "article_links_out")
    lookup = pg.CsvLinks(os.path.join(links, "units.csv"), os.path.join(links, "refers_to.csv"))
    texts = args.texts if args.texts is not None else [os.path.join(args.dir, "v47", "v47-candidate")]
    print(f"# POOL_GATE_AREA / POOL_GATE_LINK simulated (hub: {args.hub}+ citing articles"
          f"{'; v50 first rules' if args.before else ''})\n")
    totals = []
    for name in args.runs:
        run = link_reach.load_run(args.dir, name)
        if run is None:
            print(f"## {name}\n\n(no report.json)\n")
            continue
        sims = {cid: simulate(case, lookup, case_text([os.path.join(args.dir, name)] + texts, cid), args.hub,
                              args.before) for cid, case in run.items()}
        print(f"## {name}\n")
        print("| case | areas -> codes allowed | pool | dedup | area | link (alone) | both | % | expected lost: area | "
              "link | removed, cited unexpected (area / link) |\n|---|---|---:|---:|---:|---:|---:|---:|---|---|---|")
        for cid, s in sims.items():
            codes = "all" if s["allowed"] is None else "/".join(sorted(s["allowed"]))
            lost = {g: "; ".join(f"{lab} [{st}]" for lab, st in s[g]["lost"]) or "-" for g in ("area", "link")}
            print(f"| {cid} | {'/'.join(run[cid].get('areas') or []) or '-'} -> {codes}{'' if s['text'] else ' (no text)'}"
                  f" | {s['pool']} | {s['dedup']} | {len(s['area']['removed'])} | {len(s['link']['removed'])} | "
                  f"{len(s['both']['removed'])} | {100 * len(s['both']['removed']) / max(1, s['pool']):.0f} | "
                  f"{lost['area']} | {lost['link']} | {s['area']['noise']} / {s['link']['noise']} |")
        pool = sum(s["pool"] for s in sims.values())
        cited = sum(1 for c in run.values() for u in c.get("unexpected") or [] if u.get("in_pool"))
        row = {"name": name, "pool": pool, "cited": cited}
        print()
        for gate in ("area", "link", "both"):
            removed = sum(len(s[gate]["removed"]) for s in sims.values())
            lost = [(cid, lab, st) for cid, s in sims.items() for lab, st in s[gate]["lost"]]
            lost_m = [x for x in lost if x[2] == "MENTIONED"]
            noise = sum(s[gate]["noise"] for s in sims.values())
            verdicts = sum((s[gate]["verdicts"] for s in sims.values()), Counter())
            row[gate] = (removed, len(lost), len(lost_m), noise)
            print(f"**{name}, {gate}{' (alone)' if gate == 'link' else ''}**: removed {removed} of {pool} "
                  f"({100 * removed / max(1, pool):.0f}%, {removed / max(1, len(sims)):.1f} a case); expected lost "
                  f"{len(lost)}" + (": " + "; ".join(f"{cid} {lab} [{st}]" for cid, lab, st in lost) if lost else "")
                  + f"; of them MENTIONED {len(lost_m)}; unexpected cited from the pool gated out {noise} of {cited}; "
                  "what the check did with the removed: " + (", ".join(f"{v} {n}" for v, n in verdicts.most_common())
                                                             or "-") + ".\n")
        print(f"Dedup: {sum(s['dedup'] for s in sims.values())}.\n")
        o = outside(run)
        print(f"Cited with no pool row (unexpected): {o['total']}; offered material (tools, terms, procedural lists, "
              f"the case's own) {o['material']}"
              + ("" if o["material_logged"] else " (not classified: rescore the run with diagnose_long --rescore)")
              + f"; closure verified {o['verified']}, read as offered material {o['offered']}"
              + ("" if o["offered_logged"] else " (offered_rows not logged: a run before v49)")
              + f"; outside the material {len(o['bare'])}" + (": " + "; ".join(o["bare"]) if o["bare"] else "") + ".\n")
        totals.append(row)
    if totals:
        print("| run | pool | area: removed / lost / MENTIONED | link alone: removed / lost / MENTIONED | "
              "both: removed / lost / MENTIONED | cited unexpected gated out (both) |\n|---|---:|---|---|---|---:|")
        for r in totals:
            cells = [f"{r[g][0]} ({100 * r[g][0] / max(1, r['pool']):.0f}%) / {r[g][1]} / {r[g][2]}"
                     for g in ("area", "link", "both")]
            print(f"| {r['name']} | {r['pool']} | " + " | ".join(cells) + f" | {r['both'][3]} / {r['cited']} |")


if __name__ == "__main__":
    main()
