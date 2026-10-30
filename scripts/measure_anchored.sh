#!/usr/bin/env bash
# A candidate configuration against the live one, in one go (run on the server, from the repo):
#
#   nohup scripts/measure_anchored.sh [TAG [--set NAME=VALUE ...]] > /dev/null 2>&1 &
#   LIVE_RUN=diagnose_long_runs/<an earlier live run> nohup scripts/measure_anchored.sh TAG --set ... &
#
#   live       diagnose_long from the live .env, no overrides (FACT_ISSUES, COMMA_MATERIAL, PROTECTED_PARAPHRASE,
#              CCNL_PATH expected on: checked below, a warning if not); LIVE_RUN: that run, not run again
#   candidate  the same with the --set overrides after TAG (default: ANCHORED_WRITING_ENABLED=true
#              ROLE_BLOCKS_ENABLED=true)
#   then       diagnose_long --compare (headline + loss funnel), fetch_audit --mechanisms provenance,role on both,
#              and a summary block: MENTIONED, rubrica-incompatible, extra per answer, law / case-law sentences from
#              memory (%), role flags per 100 judged, abstentions per answer, expected articles lost in removed /
#              rewritten sentences.
#
# Everything goes to measure_runs/<TAG>/ (log.txt, compare.md, summary.md); DONE (or FAILED) written there last.
set -euo pipefail

cd "$(dirname "$0")/.."
TAG="${1:-aw-$(date +%Y%m%d-%H%M)}"
shift $(( $# > 0 ? 1 : 0 ))
SETS=("$@")
[ ${#SETS[@]} -gt 0 ] || SETS=(--set ANCHORED_WRITING_ENABLED=true --set ROLE_BLOCKS_ENABLED=true)
PY="${PY:-venv/bin/python}"
OUT="measure_runs/$TAG"
LIVE_DIR="${LIVE_RUN:-diagnose_long_runs/${TAG}-live}"
LIVE="$(basename "$LIVE_DIR")"
CAND="${TAG}-candidate"
CAND_DIR="diagnose_long_runs/$CAND"
AUDIT_DIR="fetch_audit_runs/$TAG"
mkdir -p "$OUT"
LOG="$OUT/log.txt"
exec > >(tee -a "$LOG") 2>&1
trap 'echo "failed at line $LINENO (exit $?)"; date > "$OUT/FAILED"' ERR

echo "== $TAG, $(date), code $(git rev-parse --short HEAD)"
for flag in FACT_ISSUES_ENABLED COMMA_MATERIAL_ENABLED PROTECTED_PARAPHRASE_ENABLED; do
    grep -Eqi "^\s*${flag}\s*=\s*\"?(true|1|yes|on)\"?\s*$" .env || echo "WARNING: $flag is not on in .env"
done
grep -Eq '^\s*CCNL_PATH\s*=\s*\S' .env || echo "WARNING: CCNL_PATH is not set in .env"
echo "candidate overrides: ${SETS[*]}"
for pair in "${SETS[@]}"; do
    [ "$pair" = "--set" ] && continue
    grep -Eqi "^\s*${pair%%=*}\s*=\s*\"?${pair#*=}\"?\s*$" .env && echo "WARNING: ${pair%%=*} is already ${pair#*=} in .env"
done

if [ -n "${LIVE_RUN:-}" ]; then
    [ -d "$LIVE_DIR/cases" ] || { echo "LIVE_RUN $LIVE_DIR has no cases/"; false; }
    echo "== live: reusing $LIVE_DIR"
else
    echo "== live: $(date)"
    "$PY" scripts/diagnose_long.py --out "$LIVE_DIR"
fi
echo "== candidate: $(date)"
"$PY" scripts/diagnose_long.py --out "$CAND_DIR" "${SETS[@]}"

echo "== compare: $(date)"
"$PY" scripts/diagnose_long.py --compare "$LIVE_DIR" "$CAND_DIR" --compare-only > /dev/null
cp "$CAND_DIR/compare_${LIVE}.md" "$OUT/compare.md"
# the headline and the loss funnel (A), up to table B
awk '/^## Headline/{on=1} /^## B\./{on=0} on' "$OUT/compare.md"

echo "== fetch_audit provenance,role: $(date)"
"$PY" scripts/fetch_audit.py --mechanisms provenance,role "$LIVE_DIR" "$CAND_DIR" --out "$AUDIT_DIR"
cp "$AUDIT_DIR/report.md" "$OUT/fetch_audit.md"

echo "== summary: $(date)"
"$PY" - "$LIVE_DIR" "$CAND_DIR" "$AUDIT_DIR" > "$OUT/summary.md" <<'PY'
import json
import os
import sys
from collections import Counter

sys.path.insert(0, "scripts")
import diagnose_long as dl   # noqa: E402
import fetch_audit as fa     # noqa: E402

run_dirs, audit_dir = sys.argv[1:3], sys.argv[3]
names = [os.path.basename(os.path.normpath(d)) for d in run_dirs]
ev = dl._eval_module()
runs = [dl._by_repeat(dl._load_cases(d, ev)[1])[0] for d in run_dirs]
common = [cid for cid in runs[-1] if all(cid in r for r in runs)]
aggs = [dl.aggregate([r[c] for c in common]) for r in runs]
audit = json.load(open(os.path.join(audit_dir, "audit.json"), encoding="utf-8"))
role = fa.role_by_run(audit["cases"])
prov = fa.provenance_by_section(audit["cases"])


def memory(run, kinds):
    total = Counter()
    for row in (prov.get(run) or {}).values():
        total.update(row)
    n = sum(total[k] for k in kinds)
    k = sum(total[f"{k}_assente"] for k in kinds)
    return f"{100 * k / n:.0f}% ({k}/{n})" if n else "-"


def lost(a):
    items = a.get("anchored_lost") or []
    if not a.get("anchored"):
        return "off"
    by = Counter(how for _, _, how in items)
    return f"{len(items)} (" + ", ".join(f"{h} {by[h]}" for h in dl.ANCHORED_LOSSES) + ")"


def flags(run):
    r = role.get(run) or {}
    return (f"{fa._rate(r.get('rate'))} ({r.get('judge', 0)}/{r.get('judged', 0)}, failed batches "
            f"{r.get('failed', 0)})") if r else "-"


rows = [("MENTIONED", lambda a, n: f"{a['stage_counts']['MENTIONED']} / {len(a['slots'])} "
                                  f"({100 * a['stage_counts']['MENTIONED'] / max(1, len(a['slots'])):.0f}%)"),
        ("rubrica-incompatible", lambda a, n: a["fit"]["incompatible"]),
        ("extra (not expected) per answer", lambda a, n: round(len(a["wrong"]) / max(1, len(common)), 1)),
        ("law sentences from memory (rule, institution, procedure)",
         lambda a, n: memory(n, ("regola", "istituto", "procedura"))),
        ("case-law sentences from memory", lambda a, n: memory(n, ("massima",))),
        ("all rule-sentences from memory (target <10%)", lambda a, n: memory(n, fa.KINDS)),
        ("role flags per 100 judged", lambda a, n: flags(n)),
        ("abstentions per answer", lambda a, n: round(a.get("abstentions", 0) / max(1, a.get("answered", 0)), 1)),
        ("fact-institution articles placed / reached (expected)",
         lambda a, n: (lambda f: f"{f.get('placed', 0)} / {f.get('reached', 0)} ({f.get('expected_placed', 0)} / "
                                 f"{f.get('expected_reached', 0)})" if a.get("fact_placed_logged") else "off")(
             a.get("fact_placed") or {})),
        ("expected articles lost in removed / rewritten / citation-dropped sentences", lambda a, n: lost(a))]
print(f"# Summary: {' vs '.join(names)}\n\n{len(common)} cases in both runs, with an answer key and an answer.\n")
print("| | " + " | ".join(names) + " |\n|---|" + "---:|" * len(names))
for title, value in rows:
    print(f"| {title} | " + " | ".join(str(value(a, n)) for a, n in zip(aggs, names)) + " |")
an = aggs[-1].get("anchored") or {}
if an:
    print(f"\nAnchored writing ({names[-1]}): rewritten {an.get('rewritten', 0)}, kept (checked article) "
          f"{an.get('kept_checked', 0)}, kept (other) {an.get('kept_failing', 0)}, removed {an.get('removed', 0)} "
          f"(case law without a ruling {an.get('removed_case_law', 0)}, no sentence left without its citations "
          f"{an.get('removed_without_citation', 0)}), citations dropped {an.get('citations_dropped', 0)}.")
    print("Overlap of the overlap failures: " + (", ".join(f"{b} {v}" for b, v in sorted(
        (aggs[-1].get("overlap_failures") or {}).items(), key=lambda x: float(x[0].strip(">=%").split("-")[0])))
        or "none") + ".")
    print("Expected articles lost: " + ("; ".join(f"{cid} {label} ({how})" for cid, label, how in
                                                aggs[-1].get("anchored_lost") or []) or "none") + ".")
print(f"\nFiles: {run_dirs[-1]}/compare_{names[0]}.md, {audit_dir}/report.md")
PY
cat "$OUT/summary.md"
date > "$OUT/DONE"
echo "== done: $(date) -> $OUT/DONE"
