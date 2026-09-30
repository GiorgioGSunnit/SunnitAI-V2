"""Meaning vectors for every catalog template, for embedding-based selection.

Each template is embedded as "label. tipo_atto. description" (the variant that
won the Sept 2026 prototype), on the production embedding server
(EMBEDDING_BASE_URL / EMBEDDING_MODEL in .env), in batches of 32.

Resumable: each batch is saved as a shard in <DATA>/selection/shards as soon as
it returns; a re-run embeds only what is missing. At the end the vectors are
assembled in catalog order:
    <DATA>/selection/template_vectors.npy    float32, one row per catalog entry
    <DATA>/selection/template_index.json     filename / label / text hash per row

    python embed_catalog.py [--catalog C:\\tplstage\\catalog_enriched.json]
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time

import httpx
import numpy as np
from dotenv import load_dotenv

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.environ.get("DEJURE_DATA", r"C:\Users\anton\Downloads\catalog_enrich")
STORE = os.path.join(DATA, "selection")
SHARDS = os.path.join(STORE, "shards")
BATCH = 32

load_dotenv(os.path.join(REPO, ".env"))
BASE = (os.getenv("EMBEDDING_BASE_URL") or "").rstrip("/")
MODEL = os.getenv("EMBEDDING_MODEL")
KEY = os.getenv("EMBEDDING_API_KEY", os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY")))


def template_text(e):
    """label + tipo_atto (when different) + description (unless a placeholder)."""
    parts = [e.get("label", "")]
    if (e.get("tipo_atto") or "").strip().lower() != parts[0].strip().lower():
        parts.append(e["tipo_atto"])
    desc = (e.get("description") or "").strip()
    if desc and not desc.startswith("Template per:"):
        parts.append(desc)
    return ". ".join(p for p in parts if p)


def sha(t):
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


def load_done():
    done = {}
    for jf in sorted(glob.glob(os.path.join(SHARDS, "*.json"))):
        keys = json.load(open(jf, encoding="utf-8"))
        vecs = np.load(jf[:-5] + ".npy")
        done.update(zip(keys, vecs))
    return done


def embed_batch(texts):
    for wait in (0, 10, 60, 300):
        time.sleep(wait)
        try:
            r = httpx.post(BASE + "/embeddings", json={"model": MODEL, "input": texts},
                           headers={"Authorization": "Bearer " + KEY} if KEY else {}, timeout=300)
            r.raise_for_status()
            return np.array([d["embedding"] for d in r.json()["data"]], dtype=np.float32)
        except Exception as exc:          # network / server busy: wait and retry
            print("  retrying after error: %s" % str(exc)[:120], flush=True)
    raise SystemExit("embedding server unreachable - re-run later, finished batches are kept")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=r"C:\tplstage\catalog_enriched.json")
    args = ap.parse_args()
    catalog = json.load(open(args.catalog, encoding="utf-8"))
    texts = [template_text(e) for e in catalog]
    keys = [sha(t) for t in texts]

    os.makedirs(SHARDS, exist_ok=True)
    done = load_done()
    todo = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in done]
    print("%d templates, %d already embedded, %d to do (%d batches)" % (
        len(catalog), len(set(keys)) - len(todo), len(todo), -(-len(todo) // BATCH)), flush=True)

    t0 = time.time()
    n_shard = len(glob.glob(os.path.join(SHARDS, "*.json")))
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        vecs = embed_batch([t for _, t in batch])
        n_shard += 1
        base = os.path.join(SHARDS, "shard_%05d" % n_shard)
        np.save(base + ".npy", vecs)
        json.dump([k for k, _ in batch], open(base + ".json", "w", encoding="utf-8"))
        done.update(zip([k for k, _ in batch], vecs))
        el = time.time() - t0
        n = i // BATCH + 1
        total = -(-len(todo) // BATCH)
        print("  batch %d/%d  %.1fs each  eta %.0f min" % (n, total, el / n, el / n * (total - n) / 60), flush=True)

    matrix = np.stack([done[k] for k in keys]).astype(np.float32)
    np.save(os.path.join(STORE, "template_vectors.npy"), matrix)
    json.dump([{"filename": e["filename"], "label": e.get("label", ""), "sha1": k}
               for e, k in zip(catalog, keys)],
              open(os.path.join(STORE, "template_index.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("done: %d x %d vectors -> %s" % (matrix.shape[0], matrix.shape[1], STORE), flush=True)


if __name__ == "__main__":
    sys.exit(main())
