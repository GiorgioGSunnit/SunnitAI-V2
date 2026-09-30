"""Offline prototype: embedding-based template selection with a floor and a margin.

No production code is touched. Embeddings come from the same server and model
production uses (EMBEDDING_BASE_URL / EMBEDDING_MODEL in .env) and are cached in
emb_cache.json, so every text is sent once.
"""
import os, sys, json, hashlib, re
import numpy as np
import httpx
from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = r"C:\Users\anton\OneDrive\Documents\GitHub\SunnitAI-V2"
CATALOG = r"C:\Users\anton\Downloads\catalog_enriched_server.json"
CONVERTED = r"C:\Users\anton\Downloads\downloads\downloads\converted"
CACHE = os.path.join(HERE, "emb_cache.json")

load_dotenv(os.path.join(REPO, ".env"))
BASE = os.getenv("EMBEDDING_BASE_URL").rstrip("/")
MODEL = os.getenv("EMBEDDING_MODEL")
KEY = os.getenv("EMBEDDING_API_KEY", os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY")))

sys.path.insert(0, HERE)
import selection_testset as T

cat = json.load(open(CATALOG, encoding="utf-8"))
new_titles = sorted(f[:-5] for f in os.listdir(CONVERTED) if f.endswith(".docx") and not f.startswith("~$"))

_cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}


def embed(texts):
    """Embed with cache; batches of 32 to the production embedding server."""
    keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
    todo = [(k, t) for k, t in zip(keys, texts) if k not in _cache]
    for i in range(0, len(todo), 32):
        batch = todo[i:i + 32]
        r = httpx.post(BASE + "/embeddings", json={"model": MODEL, "input": [t for _, t in batch]},
                       headers={"Authorization": f"Bearer {KEY}"} if KEY else {}, timeout=300)
        r.raise_for_status()
        for (k, _), d in zip(batch, r.json()["data"]):
            _cache[k] = d["embedding"]
        json.dump(_cache, open(CACHE, "w", encoding="utf-8"))
        print("  embedded %d/%d" % (min(i + 32, len(todo)), len(todo)), file=sys.stderr)
    m = np.array([_cache[k] for k in keys], dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


# --- catalog text variants ---------------------------------------------------
def v_label(e):
    return e["label"]


def v_desc(e):
    parts = [e["label"]]
    if e["tipo_atto"].strip().lower() != e["label"].strip().lower():
        parts.append(e["tipo_atto"])
    if not e["description"].startswith("Template per:"):
        parts.append(e["description"])
    return ". ".join(parts)


def v_full(e):
    cats = [c for c in (e.get("categorie") or []) + (e.get("sottocategorie") or []) if c]
    return "%s. Area: %s. Categoria: %s." % (v_desc(e), e["codice"], ", ".join(cats))


VARIANTS = {"label": v_label, "label+desc": v_desc, "label+desc+area": v_full}
INSTRUCT = ("Instruct: Data la richiesta di un avvocato, trova il modello di atto giuridico "
            "italiano che corrisponde al documento richiesto\nQuery: ")


def cases():
    out = []
    for q, acc in T.FOUND:
        out.append({"q": q, "acc": set(acc), "kind": "found", "new": None})
    for q, acc, n in T.NEW:
        out.append({"q": q, "acc": set(acc), "kind": "near" if acc else "missing", "new": n})
    for q in T.UNRELATED:
        out.append({"q": q, "acc": set(), "kind": "unrelated", "new": None})
    return out


def decide(sims, floor, margin, window=0.03, k=5):
    """Product rule: below floor -> reject; clear winner -> auto; else picker."""
    order = np.argsort(-sims)
    s1, s2 = sims[order[0]], sims[order[1]]
    if s1 < floor:
        return "reject", []
    if s1 - s2 >= margin:
        return "auto", [int(order[0])]
    return "picker", [int(i) for i in order[:k] if sims[i] >= s1 - window] or [int(order[0])]


def outcome(case, action, picked, acc):
    if not acc:
        return {"reject": "ok", "picker": "bad_offer", "auto": "WRONG_AUTO"}[action]
    if action == "reject":
        return "false_reject"
    hit = any(p in acc for p in picked)
    if action == "auto":
        return "ok" if hit else "WRONG_AUTO"
    return "ok_picker" if hit else "bad_offer"
