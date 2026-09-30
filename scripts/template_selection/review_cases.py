"""List the test requests whose expected template is not ranked first on the new
catalog (and the 'nothing fits' ones), with their top candidates, for relabelling.
Read-only; uses the vectors and query embeddings calibrate.py cached."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import calibrate as C  # noqa: E402
import selection_testset as T  # noqa: E402

catalog = json.load(open(C.CATALOG, encoding="utf-8"))
M = C.normalise(np.load(os.path.join(C.STORE, "template_vectors.npy")), 1024)
by_fn = {e["filename"]: i for i, e in enumerate(catalog)}
by_label = {}
for i, e in enumerate(catalog):
    by_label.setdefault(C.key(e.get("label", "")), i)
old = json.load(open(C.OLD, encoding="utf-8"))
pilot = sorted(f[:-5] for f in os.listdir(C.PILOT) if f.endswith(".docx") and not f.startswith("~$"))


def old_idx(i):
    return by_fn.get(old[i]["filename"], by_label.get(C.key(old[i]["label"])))


cases = [(q, {old_idx(i) for i in acc} - {None}) for q, acc in T.FOUND]
cases += [(q, ({old_idx(i) for i in acc} | {by_label.get(C.key(pilot[n]))}) - {None}) for q, acc, n in T.NEW]
cases += [(q, set()) for q in T.UNRELATED]
cache = json.load(open(os.path.join(C.STORE, "test_queries.json"), encoding="utf-8"))
Q = C.normalise(np.array([cache[q] for q, _ in cases], dtype=np.float32), 1024)
S = Q @ M.T
for n, ((q, acc), s) in enumerate(zip(cases, S)):
    order = np.argsort(-s)
    if acc and int(order[0]) in acc:
        continue
    print("#%d  %s" % (n, q))
    for j in order[:6]:
        print("     %s %.3f  %-105s %s" % ("*" if int(j) in acc else " ", s[j], catalog[j]["label"][:105], catalog[j]["filename"]))
    if acc and not set(int(x) for x in order[:6]) & acc:
        for j in sorted(acc, key=lambda j: -s[j])[:2]:
            print("     expected %.3f (rank %d)  %s" % (s[j], int(np.where(order == j)[0][0]) + 1, catalog[j]["label"][:90]))
