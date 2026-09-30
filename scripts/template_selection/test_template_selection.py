"""Tests for src/rag/template_selection.py and its use in classify_system_template.

  A. the rule and the fallbacks, on fake data (no network)
  B. end to end: all test_v2 requests through the production module, with the
     shipped vector file (C:\\tplstage) and the real embedding server
  C. classify_system_template: result shape, and the keyword fallback

    python test_template_selection.py
"""
import importlib
import json
import os
import sys
import tempfile
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
STAGE = r"C:\tplstage"
sys.path.insert(0, REPO)
from dotenv import load_dotenv  # noqa: E402
load_dotenv(os.path.join(REPO, ".env"))

failures = 0


def check(name, ok, detail=""):
    global failures
    print("%s  %s%s" % ("PASS" if ok else "FAIL", name, ("   " + detail) if detail else ""))
    failures += not ok


def fresh_module(**env):
    for k, v in env.items():
        os.environ[k] = v
    import src.rag.template_selection as TS
    return importlib.reload(TS)


# ---------------------------------------------------------------- A
TS = fresh_module()
check("A decide: below the floor -> nothing", TS.decide(np.array([0.5, 0.4]), 0.62) == [])
check("A decide: clear winner -> auto", [i for i, _ in TS.decide(np.array([0.9, 0.7, 0.6]), 0.62, 0.10)] == [0])
p = TS.decide(np.array([0.80, 0.79, 0.78, 0.77, 0.76, 0.755, 0.70]), 0.62, 0.10, 0.05)
check("A decide: close variants -> picker of at most 5", [i for i, _ in p] == [0, 1, 2, 3, 4], str(p))
p = TS.decide(np.array([0.80, 0.72, 0.60]), 0.62, 0.10, 0.05)
check("A decide: picker always offers at least 2", [i for i, _ in p] == [0, 1], str(p))

tmp = tempfile.mkdtemp()
cat = [{"filename": "a.docx", "codice": "Codice di procedura civile"},
       {"filename": "b.docx", "codice": "Codice di procedura penale"},
       {"filename": "c.docx", "codice": "Codice di procedura civile"}]
vec = np.eye(3, 4, dtype=np.float16)
np.save(os.path.join(tmp, "v.npy"), vec)
json.dump([{"filename": e["filename"]} for e in cat], open(os.path.join(tmp, "i.json"), "w"))

TS = fresh_module(TEMPLATE_VECTORS_PATH=os.path.join(tmp, "missing.npy"), TEMPLATE_VECTORS_INDEX=os.path.join(tmp, "i.json"))
check("A no vector file -> None (keyword fallback)", TS.select_templates("x", cat) is None)

json.dump([{"filename": f} for f in ("a.docx", "x.docx", "y.docx")], open(os.path.join(tmp, "bad.json"), "w"))
TS = fresh_module(TEMPLATE_VECTORS_PATH=os.path.join(tmp, "v.npy"), TEMPLATE_VECTORS_INDEX=os.path.join(tmp, "bad.json"))
check("A vectors out of step with the catalog -> None", TS.select_templates("x", cat) is None)

TS = fresh_module(TEMPLATE_VECTORS_PATH=os.path.join(tmp, "v.npy"), TEMPLATE_VECTORS_INDEX=os.path.join(tmp, "i.json"))
TS._embed = lambda text, dims: None
check("A embedding server down -> None", TS.select_templates("x", cat) is None)

TS._embed = lambda text, dims: np.array([0.70, 0.72, 0.0, 0.0], dtype=np.float32)[:dims] / np.linalg.norm([0.70, 0.72])
got = TS.select_templates("x", cat)
check("A no filter: the penal template (b) is offered", got is not None and 1 in [i for i, _ in got], str(got))
got = TS.select_templates("x", cat, exclude_codici={"Codice di procedura penale"})
check("A c.p.c. request: the penal template is dropped", got == [(0, got[0][1])] if got else False, str(got))
TS._embed = lambda text, dims: np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)[:dims]
got = TS.select_templates("x", cat, exclude_codici={"Codice di procedura penale"})
check("A filter skipped when it would leave nothing above the floor", got and got[0][0] == 1, str(got))

# ---------------------------------------------------------------- B
TS = fresh_module(TEMPLATE_VECTORS_PATH=os.path.join(STAGE, "template_vectors.npy"),
                  TEMPLATE_VECTORS_INDEX=os.path.join(STAGE, "template_index.json"))
catalog = json.load(open(os.path.join(STAGE, "catalog_enriched.json"), encoding="utf-8"))
cases = json.load(open(os.path.join(HERE, "testset_v2.json"), encoding="utf-8"))
idx = {e["filename"]: i for i, e in enumerate(catalog)}
count = dict.fromkeys(("auto_ok", "picker_ok", "false_no", "bad_offer", "WRONG_AUTO", "absent_ok", "absent_offered"), 0)
for c in cases:
    got = TS.select_templates(c["q"], catalog)
    if got is None:
        count["WRONG_AUTO"] += 1000          # must never happen here
        continue
    picked = {i for i, _ in got}
    acc = {idx[f] for f in c["acc"]}
    if c["kind"] == "absent":
        count["absent_ok" if not got else "absent_offered"] += 1
    elif not got:
        count["false_no"] += 1
    elif len(got) == 1:
        count["auto_ok" if picked & acc else "WRONG_AUTO"] += 1
    else:
        count["picker_ok" if picked & acc else "bad_offer"] += 1
print("   B outcomes:", count)
check("B no wrong auto-select", count["WRONG_AUTO"] == 0)
check("B all absent requests refused", count["absent_offered"] == 0)
# Live, one-at-a-time query embeddings drift ~0.002 from the batched ones the
# calibration used, so a request whose lead sits right at the margin can move
# between auto-select and a picker (both with the right template). What must
# hold: the same 107 right answers, the same 2 misses, no wrong auto-select.
check("B same as the calibration (107 right, 1 false no, 1 bad offer; auto/picker split may shift)",
      (count["auto_ok"] + count["picker_ok"], count["false_no"], count["bad_offer"]) == (107, 1, 1),
      "auto %d + picker %d" % (count["auto_ok"], count["picker_ok"]))

# ---------------------------------------------------------------- C
for name in ("langchain_openai",):
    m = types.ModuleType(name)

    class _C:
        def __init__(self, **kw): pass
        def bind(self, **kw): return self
        def with_structured_output(self, *a, **kw): return self
    m.ChatOpenAI = _C
    m.OpenAIEmbeddings = _C
    sys.modules[name] = m
fa = types.ModuleType("fastapi")


class HTTPException(Exception):
    def __init__(self, status_code=500, detail=""):
        super().__init__(detail)


fa.HTTPException = HTTPException
sys.modules["fastapi"] = fa
os.environ["SYSTEM_TEMPLATES_CATALOG_PATH"] = os.path.join(STAGE, "catalog_enriched.json")
import src.rag.document_generation as G  # noqa: E402
G.select_templates = TS.select_templates      # the module configured on the staged files
res = G.classify_system_template("Mi serve l'atto costitutivo di una srl", "it", top_k=5)
check("C result shape", isinstance(res, list) and res and set(res[0]) == {"key", "label", "codice", "sublabel", "score"}, str(res[:1]))
check("C srl request -> atto costitutivo offered", any("atto-costitutivo-di-s-r-l" in r["key"] for r in res), str([r["key"] for r in res]))
res = G.classify_system_template("Scrivimi una poesia per il compleanno di mia madre", "it", top_k=5)
check("C nothing in the catalog -> []", res == [], str(res))
G.select_templates = lambda *a, **k: None     # selection by meaning unavailable
G._call_chat = lambda *a, **k: "[0, 1, 2]"    # the keyword path's LLM ranking, stubbed
res = G.classify_system_template("Mi serve l'atto costitutivo di una srl", "it", top_k=5)
check("C fallback: keyword path answers when selection by meaning cannot", isinstance(res, list) and len(res) >= 1, "%d results" % len(res))

print("\n%d failure(s)" % failures)
sys.exit(1 if failures else 0)
