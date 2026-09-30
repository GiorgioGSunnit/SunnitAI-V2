"""How far do live (one-at-a-time) query embeddings drift from the batched ones
the calibration used, and which decisions sit close enough to a threshold to flip?"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
from dotenv import load_dotenv  # noqa: E402
load_dotenv(os.path.join(REPO, ".env"))
import src.rag.template_selection as TS  # noqa: E402

SEL = r"C:\Users\anton\Downloads\catalog_enrich\selection"
cases = json.load(open(os.path.join(HERE, "testset_v2.json"), encoding="utf-8"))
cache = json.load(open(os.path.join(SEL, "test_queries.json"), encoding="utf-8"))
M = np.load(r"C:\tplstage\template_vectors.npy").astype(np.float32)
M /= np.linalg.norm(M, axis=1, keepdims=True)

drift, near, failed = [], [], 0
for c in cases:
    b = np.asarray(cache[c["q"]], dtype=np.float32)[:1024]
    b /= np.linalg.norm(b)
    live = TS._embed(c["q"], 1024)
    if live is None:                                   # one retry
        live = TS._embed(c["q"], 1024)
    if live is None:
        failed += 1
        continue
    drift.append(float(1 - b @ live))
    sb, sl = np.sort(M @ b)[::-1], np.sort(M @ live)[::-1]
    for name, vb, vl, t in (("best vs floor", sb[0], sl[0], TS.FLOOR), ("lead vs margin", sb[0] - sb[1], sl[0] - sl[1], TS.MARGIN)):
        if (vb >= t) != (vl >= t) or abs(vl - t) < 0.01:
            near.append("%-15s batched %.4f  live %.4f  (threshold %.2f)  %s" % (name, vb, vl, t, c["q"][:70]))
print("requests: %d, embedding calls that failed twice: %d" % (len(cases), failed))
print("cosine distance live vs batched: median %.2e, max %.2e" % (np.median(drift), max(drift)))
print("decisions within 0.01 of a threshold, or flipped:")
for line in near:
    print("   " + line)
