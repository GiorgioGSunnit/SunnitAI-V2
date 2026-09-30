"""Pick the selection rule's thresholds on test set v2 (testset_v2.json).

For every combination of floor (below -> "we don't have it"), margin (best minus
second at least this -> auto-select) and window (picker = templates within this
of the best, 2 to 5 of them) it counts, over the requests that HAVE a right
template: auto right / picker containing a right one / wrongly told "we don't
have it" / picker without a right one / WRONG auto-select; and over the requests
the catalog cannot serve: correctly rejected / offered something anyway.

    python calibrate_v2.py [--dims 1024] [--show 0.60,0.12,0.05]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import calibrate as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, default=1024)
    ap.add_argument("--show", default="")
    args = ap.parse_args()

    catalog = json.load(open(C.CATALOG, encoding="utf-8"))
    idx = {e["filename"]: i for i, e in enumerate(catalog)}
    cases = json.load(open(os.path.join(HERE, "testset_v2.json"), encoding="utf-8"))
    M = C.normalise(np.load(os.path.join(C.STORE, "template_vectors.npy")), args.dims)
    cache = json.load(open(os.path.join(C.STORE, "test_queries.json"), encoding="utf-8"))
    Q = C.normalise(np.array([cache[c["q"]] for c in cases], dtype=np.float32), args.dims)
    S = Q @ M.T
    acc = [{idx[f] for f in c["acc"]} for c in cases]
    found = [i for i, c in enumerate(cases) if c["kind"] == "found"]
    absent = [i for i, c in enumerate(cases) if c["kind"] == "absent"]

    top1 = sum(1 for i in found if int(np.argmax(S[i])) in acc[i])
    top5 = sum(1 for i in found if set(int(j) for j in np.argsort(-S[i])[:5]) & acc[i])
    best_absent = sorted(float(S[i].max()) for i in absent)
    worst_found = sorted(float(max(S[i][j] for j in acc[i])) for i in found)
    print("requests: %d with a right template, %d without" % (len(found), len(absent)))
    print("right template 1st: %d/%d, in top 5: %d/%d" % (top1, len(found), top5, len(found)))
    print("best score of the 'absent' requests (highest 5):", ["%.3f" % x for x in best_absent[-5:]])
    print("score of the right template, lowest 5 found:   ", ["%.3f" % x for x in worst_found[:5]])
    print()
    print("floor margin window | auto  picker  false-no  bad-offer  WRONG-auto | absent: rejected offered | avg picker")
    rows = []
    for floor in (0.56, 0.58, 0.60, 0.62):
        for margin in (0.08, 0.10, 0.12):
            for window in (0.03, 0.05, 0.08):
                cnt = dict.fromkeys(("ok", "ok_picker", "false_reject", "bad_offer", "WRONG_AUTO"), 0)
                sizes, rej = [], 0
                for i in found:
                    a, p = C.decide(S[i], floor, margin, window)
                    cnt[C.outcome(a, p, acc[i])] += 1
                    if a == "picker":
                        sizes.append(len(p))
                for i in absent:
                    a, _ = C.decide(S[i], floor, margin, window)
                    rej += a == "reject"
                rows.append((floor, margin, window, cnt, rej, np.mean(sizes) if sizes else 0))
                print(" %.2f  %.2f   %.2f  | %4d  %5d   %5d     %5d      %5d     |          %2d       %2d    |  %.1f" % (
                    floor, margin, window, cnt["ok"], cnt["ok_picker"], cnt["false_reject"], cnt["bad_offer"],
                    cnt["WRONG_AUTO"], rej, len(absent) - rej, np.mean(sizes) if sizes else 0))

    if args.show:
        floor, margin, window = (float(x) for x in args.show.split(","))
        print("\nat floor %.2f, margin %.2f, window %.2f - every case that is not right:" % (floor, margin, window))
        for i, c in enumerate(cases):
            a, p = C.decide(S[i], floor, margin, window)
            o = C.outcome(a, p, acc[i])
            if o in ("ok", "ok_picker"):
                continue
            print("  [%s/%s -> %s] %s" % (c["kind"], o, a, c["q"][:90]))
            for j in np.argsort(-S[i])[:3]:
                print("      %s %.3f  %s" % ("*" if int(j) in acc[i] else " ", S[i][j], catalog[j]["label"][:95]))


if __name__ == "__main__":
    main()
