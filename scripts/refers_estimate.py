"""Offline, no model call (9 Nov 2026): would the explicit cross-references (REFERS_TO) and the Sezione neighbours
of scripts/article_links.py help as "norme collegate" next to the articles an answer cites?

Inputs: the builder's dry-run CSVs (units.csv, refers_to.csv), the eval cases (expected articles per case) and,
for the plan-seed estimate, the 28 Oct civil runs (<case>_A.json: case_analysis_plan.norme = the plan's articles,
case_analysis_retrieval.article_refs = the pool).

Three estimates, counts and article numbers only (public law):
  A. reach   for every expected article of a case: is it one hop from ANOTHER expected article of the same case
             (REFERS_TO either way / same Sezione / same Capo)? The ceiling of what any one-hop list can give.
  B. oracle  seeds = the case's expected articles; per seed the top-k neighbours by channel, ranked by specificity
             (fewest citing articles first, hubs cut); precision = other expected articles reached / offered.
  C. plan    seeds = the plan's articles of the 17 civil runs; gained = expected articles not in the plan that the
             lists offer; precision = gained / offered. Comparable to the 3 Nov estimate on the ruling relations.

    python -I refers_estimate.py --links C:/tplstage/al_out --cases <repo>/tests/eval_narrative_cases.json \
        --runs C:/tplstage/relations_v0_out_x/out
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

ORDER = ("bis", "ter", "quater", "quinquies", "sexies", "septies", "octies", "novies", "decies")
LABEL = re.compile(r"^(\d+(?:[\s-]*(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies)\b)?)\s+(.+?)(?:\s+\[(.+)\])?$",
                   re.IGNORECASE)
EVAL_CODES = {"cc": "c.c.", "cp": "c.p.", "cpc": "c.p.c.", "cpp": "c.p.p.", "cpa": "c.p.a.", "cds": "c.d.s.",
              "l241": "Legge sul Procedimento Amministrativo 2026", "dlgs36": "Codice degli Appalti 2026",
              "consumo": "Codice del Consumo 2026", "tuel": "Testo unico degli enti locali 2026",
              "privacy": "Codice della Privacy 2026"}
EXTRA = {"barca": ["1490", "1491", "1492", "1494", "1495", "1497", "1512"],
         "piscina": ["2043", "2050", "2051", "2055", "2056", "2059", "2947"]}
LEVELS = ("LIBRO", "PARTE", "TITOLO", "CAPO", "SEZIONE")


def num(article):
    a = re.sub(r"[\s-]+", "-", str(article).strip().lower())
    return a


def number(article):
    m = re.match(r"(\d+)(?:-(\w+))?", article)
    if not m:
        return float("inf")
    return int(m.group(1)) + ((ORDER.index(m.group(2)) + 1) / 10 if m.group(2) in ORDER else 0)


def ref_of(label):
    """'4 c.c. [disposizioni di attuazione]' -> ('c.c.', '4', 'disposizioni di attuazione'); '1158 c.c.' -> ('c.c.', '1158', '')."""
    m = LABEL.match((label or "").strip())
    if not m:
        return None
    return (m.group(2).strip(), num(m.group(1)), (m.group(3) or "").strip())


class Links:
    def __init__(self, folder):
        self.path = {}                       # ref -> {level: unit name}
        with open(os.path.join(folder, "units.csv"), encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                m = re.match(r"^(\S+)(?:\s+\[(.+)\])?$", row["article"].strip())
                if not m:
                    continue
                ref = (row["code"].strip(), num(m.group(1)), (m.group(2) or "").strip())
                self.path.setdefault(ref, {})[row["level"]] = row["unit name"]
        self.out, self.into = defaultdict(set), defaultdict(set)
        with open(os.path.join(folder, "refers_to.csv"), encoding="utf-8") as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                a, b = ref_of(row[0]), ref_of(row[1])
                if a and b and a != b:
                    self.out[a].add(b)
                    self.into[b].add(a)
        self.units = defaultdict(set)        # (level, code, run, path tuple) -> refs
        for ref, p in self.path.items():
            for level in ("CAPO", "SEZIONE"):
                u = self.unit(ref, level)
                if u:
                    self.units[u].add(ref)
        self.known = set(self.path) | set(self.out) | set(self.into)

    def unit(self, ref, level):
        p = self.path.get(ref) or {}
        if level not in p:
            return None
        return (level, ref[0], ref[2]) + tuple(p.get(l, "") for l in LEVELS[:LEVELS.index(level) + 1])

    def inbound(self, ref):
        return len(self.into.get(ref, ()))

    def neighbours(self, ref, channel, hub):
        """Ranked: REFERS_TO by specificity (fewest citing articles first, hubs out), the Sezione by number distance."""
        if channel == "out":
            cands = [r for r in self.out.get(ref, ()) if self.inbound(r) < hub]
            return sorted(cands, key=lambda r: (self.inbound(r), number(r[1])))
        if channel == "in":
            cands = [r for r in self.into.get(ref, ()) if self.inbound(r) < hub and self.inbound(ref) < hub]
            return sorted(cands, key=lambda r: (self.inbound(r), number(r[1])))
        level = "SEZIONE" if channel == "sez" else "CAPO"
        u = self.unit(ref, level)
        cands = [r for r in self.units.get(u, ()) if r != ref] if u else []
        return sorted(cands, key=lambda r: abs(number(r[1]) - number(ref[1])))


def load_cases(path):
    data = json.load(open(path, encoding="utf-8"))
    out = {}
    for c in data["cases"]:
        exp = [(EVAL_CODES.get(e["code"], e["code"]), num(e["article"]), "") for e in c.get("expected") or []
               if isinstance(e, dict) and e.get("article")]
        group = c.get("set") or ("heldout" if c.get("heldout") else "tuning")
        out[c["id"]] = (group, exp)
    for cid, arts in EXTRA.items():
        out[cid] = ("extra", [("c.c.", num(a), "") for a in arts])
    return out


def load_runs(folder):
    runs = {}
    for f in sorted(glob.glob(os.path.join(folder, "*_A.json"))):
        cid = os.path.basename(f)[:-7]
        d = json.load(open(f, encoding="utf-8"))
        ev = d.get("events") or []
        plan = [e.get("data") for e in ev if e.get("event") == "case_analysis_plan"]
        pool = [e.get("data") for e in ev if e.get("event") == "case_analysis_retrieval"]
        norme = [ref_of(x) for x in ((plan[-1] or {}).get("norme") or [])] if plan else []
        refs = [ref_of(x) for x in ((pool[-1] or {}).get("article_refs") or [])] if pool else []
        runs[cid] = ([r for r in norme if r], {r for r in refs if r})
    return runs


def offer(links, seeds, channels, k, hub, exclude):
    """The union of each seed's top-k per channel, seeds and `exclude` left out; also which seed reached what."""
    offered, by = [], defaultdict(set)
    for s in seeds:
        for ch in channels:
            for r in links.neighbours(s, ch, hub)[:k]:
                if r in seeds or r in exclude:
                    continue
                if r not in by:
                    offered.append(r)
                by[r].add(s)
    return offered, by


def pct(a, b):
    return f"{100 * a / b:.0f}%" if b else "-"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--links", required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--runs")
    p.add_argument("--hub", type=int, default=100)
    args = p.parse_args()
    L = Links(args.links)
    cases = load_cases(args.cases)
    print(f"links: {len(L.path):,} placed articles, {sum(len(v) for v in L.out.values()):,} distinct REFERS_TO pairs, "
          f"hubs (>= {args.hub} citing articles): {sum(1 for r in L.into if L.inbound(r) >= args.hub)}")

    # coverage of the expected articles
    all_exp = [(cid, g, e) for cid, (g, exp) in cases.items() for e in exp]
    unknown = [(cid, e) for cid, g, e in all_exp if e not in L.known]
    print(f"expected articles: {len(all_exp)} in {len(cases)} cases; unknown to the link data: {len(unknown)} "
          f"({Counter(e[0] for _, e in unknown).most_common(6)})")

    # A. reach
    print("\nA. reach: expected articles one hop from another expected article of the same case")
    print(f"{'set':9} {'cases':>5} {'expected':>8} {'REFERS_TO':>9} {'Sezione':>8} {'Capo':>6} {'any':>6}")
    for group in ("tuning", "heldout", "heldout2", "seen2", "extra"):
        ids = [cid for cid, (g, exp) in cases.items() if g == group and len(exp) >= 2]
        n = ref_hit = sez_hit = capo_hit = any_hit = 0
        for cid in ids:
            exp = cases[cid][1]
            for e in exp:
                others = [o for o in exp if o != e]
                n += 1
                r = any(o in L.out.get(e, ()) or o in L.into.get(e, ()) for o in others)
                s = bool(L.unit(e, "SEZIONE")) and any(L.unit(o, "SEZIONE") == L.unit(e, "SEZIONE") for o in others)
                c = bool(L.unit(e, "CAPO")) and any(L.unit(o, "CAPO") == L.unit(e, "CAPO") for o in others)
                ref_hit += r; sez_hit += s; capo_hit += c; any_hit += (r or s or c)
        if ids:
            print(f"{group:9} {len(ids):>5} {n:>8} {pct(ref_hit, n):>9} {pct(sez_hit, n):>8} {pct(capo_hit, n):>6} {pct(any_hit, n):>6}")

    # B. oracle precision
    print(f"\nB. oracle: seeds = the case's expected articles; offered = top-k per seed and channel (hub cut {args.hub})")
    print(f"{'channels':14} {'k':>2} {'cases':>5} {'offered':>7} {'per seed':>8} {'gained':>6} {'precision':>9}")
    ids = [cid for cid, (g, exp) in cases.items() if len(exp) >= 2]
    for channels in (("out",), ("out", "in"), ("out", "in", "sez"), ("sez",)):
        for k in (1, 2, 3):
            offered_n = gained = seeds_n = 0
            for cid in ids:                      # per seed: an expected article reached from another one of the case
                exp = cases[cid][1]
                seeds_n += len(exp)
                for s in exp:
                    offered, _ = offer(L, [s], channels, k, args.hub, set())
                    offered_n += len(offered)
                    gained += len([r for r in offered if r in exp])
            print(f"{'+'.join(channels):14} {k:>2} {len(ids):>5} {offered_n:>7} {offered_n / seeds_n if seeds_n else 0:>8.1f} "
                  f"{gained:>6} {pct(gained, offered_n):>9}")

    # C. plan seeds
    if args.runs:
        runs = load_runs(args.runs)
        rows = [(cid, cases[cid][1], plan, pool) for cid, (plan, pool) in runs.items() if cid in cases and cases[cid][1]]
        n_exp = sum(len(e) for _, e, _, _ in rows)
        n_in_plan = sum(len([x for x in e if x in plan]) for _, e, plan, _ in rows)
        print(f"\nC. plan seeds: {len(rows)} civil runs, expected {n_exp}, already in the plan {n_in_plan}: "
              f"at most {n_exp - n_in_plan} to gain")
        print(f"{'channels':14} {'k':>2} {'offered':>7} {'per case':>8} {'gained':>6} {'(in pool)':>9} {'precision':>9}  gained where")
        for channels in (("out",), ("out", "in"), ("out", "in", "sez"), ("sez",)):
            for k in (1, 2, 3):
                offered_n = gained = rescued = off2 = gained2 = 0
                where = []
                for cid, exp, plan, pool in rows:
                    offered, by = offer(L, plan, channels, k, args.hub, set())
                    g = [r for r in offered if r in exp]
                    offered_n += len(offered); gained += len(g); rescued += len([r for r in g if r in pool])
                    two = [r for r in offered if len(by[r]) >= 2]          # reached by two or more plan articles
                    off2 += len(two); gained2 += len([r for r in two if r in exp])
                    if g:
                        where.append(f"{cid}:{','.join(r[1] for r in g)}")
                print(f"{'+'.join(channels):14} {k:>2} {offered_n:>7} {offered_n / len(rows):>8.1f} {gained:>6} {rescued:>9} "
                      f"{pct(gained, offered_n):>9}  {' '.join(where)}"
                      f"   | reached by 2+ plan articles: offered {off2}, gained {gained2}")
        print("\nper case: expected | in plan | plan articles (c.c. and others)")
        for cid, exp, plan, pool in rows:
            print(f"  {cid:9} {','.join(e[1] + ('' if e[0] == 'c.c.' else ' ' + e[0]) for e in exp):40} "
                  f"{','.join(e[1] for e in exp if e in plan) or '-':18} {len(plan)} articles")
    return 0


if __name__ == "__main__":
    sys.exit(main())
