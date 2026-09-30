"""Re-check the embedding selection rule against the full new catalog.

Rule (Sept 2026 prototype): best score below FLOOR -> "we don't have it";
best minus second >= MARGIN -> auto-select; otherwise a picker of the templates
within WINDOW of the best (min 2, max 5).

The 89 labelled requests in selection_testset.py name the right templates by
their index in the OLD 484-entry catalog and in the sorted list of the 103 pilot
templates; both are translated to the new catalog here. After the upload, the
pilot template is the expected answer for the "near" and "missing" requests.

    python calibrate.py                 # grid + details at the chosen values
    python calibrate.py --dims 1024     # the same on shortened vectors
"""
import argparse
import json
import os
import re
import sys
import unicodedata

import httpx
import numpy as np
from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.environ.get("DEJURE_DATA", r"C:\Users\anton\Downloads\catalog_enrich")
STORE = os.path.join(DATA, "selection")
CATALOG = r"C:\tplstage\catalog_enriched.json"
OLD = r"C:\Users\anton\Downloads\catalog_enriched_server.json"
PILOT = r"C:\Users\anton\Downloads\downloads\downloads\converted"
INSTRUCT = ("Instruct: Data la richiesta di un avvocato, trova il modello di atto giuridico "
            "italiano che corrisponde al documento richiesto\nQuery: ")

sys.path.insert(0, HERE)
import selection_testset as T  # noqa: E402

load_dotenv(os.path.join(REPO, ".env"))
BASE = (os.getenv("EMBEDDING_BASE_URL") or "").rstrip("/")
MODEL = os.getenv("EMBEDDING_MODEL")
KEY = os.getenv("EMBEDDING_API_KEY", os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY")))


def key(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def normalise(m, dims):
    m = m[:, :dims] if dims else m
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def decide(sims, floor, margin, window=0.05, k=5):
    order = np.argsort(-sims)
    s1, s2 = sims[order[0]], sims[order[1]]
    if s1 < floor:
        return "reject", []
    if s1 - s2 >= margin:
        return "auto", [int(order[0])]
    picked = [int(i) for i in order[:k] if sims[i] >= s1 - window]
    return "picker", picked if len(picked) >= 2 else [int(i) for i in order[:2]]


def outcome(action, picked, acc):
    if not acc:
        return {"reject": "ok", "picker": "bad_offer", "auto": "WRONG_AUTO"}[action]
    if action == "reject":
        return "false_reject"
    hit = any(p in acc for p in picked)
    if action == "auto":
        return "ok" if hit else "WRONG_AUTO"
    return "ok_picker" if hit else "bad_offer"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, default=0)
    ap.add_argument("--floor", type=float, default=0.58)
    ap.add_argument("--margin", type=float, default=0.12)
    args = ap.parse_args()

    catalog = json.load(open(CATALOG, encoding="utf-8"))
    index = json.load(open(os.path.join(STORE, "template_index.json"), encoding="utf-8"))
    assert [e["filename"] for e in catalog] == [x["filename"] for x in index], "vectors out of date: re-run embed_catalog.py"
    M = normalise(np.load(os.path.join(STORE, "template_vectors.npy")), args.dims)

    by_fn = {e["filename"]: i for i, e in enumerate(catalog)}
    by_label = {}
    for i, e in enumerate(catalog):
        by_label.setdefault(key(e.get("label", "")), i)
    old = json.load(open(OLD, encoding="utf-8"))
    pilot = sorted(f[:-5] for f in os.listdir(PILOT) if f.endswith(".docx") and not f.startswith("~$"))

    def old_idx(i):   # an old entry, or the DEJURE version that replaced it
        return by_fn.get(old[i]["filename"], by_label.get(key(old[i]["label"])))

    cases = [{"q": q, "acc": {old_idx(i) for i in acc} - {None}, "kind": "found"} for q, acc in T.FOUND]
    for q, acc, n in T.NEW:
        a = {old_idx(i) for i in acc} | {by_label.get(key(pilot[n]))}
        cases.append({"q": q, "acc": a - {None}, "kind": "near" if acc else "missing"})
    cases += [{"q": q, "acc": set(), "kind": "unrelated"} for q in T.UNRELATED]

    qfile = os.path.join(STORE, "test_queries.json")
    cache = json.load(open(qfile, encoding="utf-8")) if os.path.exists(qfile) else {}
    todo = [c["q"] for c in cases if c["q"] not in cache]
    if todo:
        r = httpx.post(BASE + "/embeddings", json={"model": MODEL, "input": [INSTRUCT + q for q in todo]},
                       headers={"Authorization": "Bearer " + KEY} if KEY else {}, timeout=300)
        r.raise_for_status()
        cache.update({q: d["embedding"] for q, d in zip(todo, r.json()["data"])})
        json.dump(cache, open(qfile, "w", encoding="utf-8"))
    Q = normalise(np.array([cache[c["q"]] for c in cases], dtype=np.float32), args.dims)
    S = Q @ M.T

    answerable = [i for i, c in enumerate(cases) if c["acc"]]
    top1 = sum(1 for i in answerable if int(np.argmax(S[i])) in cases[i]["acc"])
    top5 = sum(1 for i in answerable if set(np.argsort(-S[i])[:5]) & cases[i]["acc"])
    print("vectors: %d x %d | requests: %d (%d with a right template, %d unrelated)" % (
        M.shape[0], M.shape[1], len(cases), len(answerable), len(cases) - len(answerable)))
    print("right template ranked 1st: %d/%d | in the top 5: %d/%d\n" % (top1, len(answerable), top5, len(answerable)))

    print("floor  margin | auto ok  picker ok  false 'no'  bad offer  WRONG AUTO | unrelated rejected")
    for floor in (0.52, 0.54, 0.56, 0.58, 0.60):
        for margin in (0.06, 0.08, 0.10, 0.12, 0.14):
            cnt = {}
            for i, c in enumerate(cases):
                a, p = decide(S[i], floor, margin)
                o = outcome(a, p, c["acc"])
                cnt[o] = cnt.get(o, 0) + 1
            unrel_ok = sum(1 for i, c in enumerate(cases) if not c["acc"] and decide(S[i], floor, margin)[0] == "reject")
            print(" %.2f   %.2f  |  %4d      %4d       %4d       %4d       %4d     |  %d/%d" % (
                floor, margin, cnt.get("ok", 0) - unrel_ok, cnt.get("ok_picker", 0), cnt.get("false_reject", 0),
                cnt.get("bad_offer", 0), cnt.get("WRONG_AUTO", 0), unrel_ok, len(cases) - len(answerable)))

    print("\nat floor %.2f, margin %.2f - every case that is not right:" % (args.floor, args.margin))
    for i, c in enumerate(cases):
        a, p = decide(S[i], args.floor, args.margin)
        o = outcome(a, p, c["acc"])
        if o in ("ok", "ok_picker"):
            continue
        order = np.argsort(-S[i])[:3]
        print("  [%s/%s] %s" % (c["kind"], o, c["q"][:80]))
        for j in order:
            mark = "*" if int(j) in c["acc"] else " "
            print("      %s %.3f  %s" % (mark, S[i][j], catalog[j]["label"][:95]))
        if c["acc"] and not any(int(j) in c["acc"] for j in order):
            best = max(c["acc"], key=lambda j: S[i][j])
            print("      expected: %.3f  %s (rank %d)" % (S[i][best], catalog[best]["label"][:80],
                                                        int(np.where(np.argsort(-S[i]) == best)[0][0]) + 1))


if __name__ == "__main__":
    main()
