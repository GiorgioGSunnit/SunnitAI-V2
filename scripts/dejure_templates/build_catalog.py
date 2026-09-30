"""Step 3 of 3 - build the new catalog and stage everything for upload.

new catalog = production catalog (as downloaded)
              + the 25-field fix for the pilot entries (never uploaded)
              + corrected pilot documents (glued words, headings) with their sections rebuilt
              - the 2 original entries replaced by DEJURE versions
              + the 4,559 new entries (inputs + model results)

Stages into C:\\tplstage (short path: filenames reach 209 chars):
  docs\\<filename>.docx     new templates + changed pilot documents
  catalog_enriched.json    the full new catalog
and writes a review sheet in Downloads\\catalog_enrich.

    python build_catalog.py [--cap 25]
"""
import argparse
import collections
import csv
import json
import os
import re
import shutil
import statistics
import sys
import unicodedata as ud

BASE = r"C:\Users\anton\Downloads\downloads\downloads"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DEJURE_DATA", r"C:\Users\anton\Downloads\catalog_enrich")   # working files, outside the repo
WORK = r"C:\Users\anton\Downloads\catalog_enrich"
STAGE = r"C:\tplstage"
sys.path.insert(0, HERE)
from enrich_dejure_templates import ascii_field, match_category, trim_sections  # noqa: E402
from prepare_catalog_inputs import sections_of as _sections_of  # noqa: E402

SIGNATURE = re.compile(r"^(firma|sottoscrizione|sottoscritto_firma)(_|$)|_(firma|sottoscrizione)$")


def key(s):
    s = ud.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", re.sub(r"\.docx$", "", s)).strip()


def clean_fields(raw, cap, stats):
    out, seen = [], set()
    for f in raw or []:
        f = ascii_field(f)
        if f and f not in seen:
            seen.add(f)
            out.append(f)
    if len(out) > cap:                       # signatures are not typed-in data
        kept = [f for f in out if not SIGNATURE.search(f)]
        stats["signature fields dropped"] += len(out) - len(kept)
        out = kept
    stats["after signature drop"].append(len(out))
    if len(out) > cap:
        stats["templates cut to the cap"] += 1
        stats["fields cut"] += len(out) - cap
        out = out[:cap]
    return out


def sections_of(path):
    import prepare_catalog_inputs as P
    old = P.FINAL
    P.FINAL = os.path.dirname(path)
    try:
        return _sections_of(path)
    finally:
        P.FINAL = old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=25)
    args = ap.parse_args()

    prod = json.load(open(os.path.join(WORK, "catalog_production.json"), encoding="utf-8"))
    staged = {e["filename"]: e for e in json.load(open(os.path.join(DATA, "catalog_fields_fix.json"), encoding="utf-8"))}
    inputs = json.load(open(os.path.join(DATA, "enrich_inputs.json"), encoding="utf-8"))
    results = {}
    for line in open(os.path.join(WORK, "results.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if r.get("ok"):
            results[r["id"]] = r

    stats = collections.Counter()
    stats["after signature drop"] = []
    catalog = []
    replaced = {e["replaces"] for e in inputs["entries"] if e.get("replaces")}

    # --- production entries: fields fix, corrected pilot documents ---------------
    pilot_dir = os.path.join(BASE, "converted_pilot_final")
    pilot_src = os.path.join(BASE, "converted")
    pilot_by_key = {key(n): n for n in os.listdir(pilot_dir) if n.endswith(".docx")}
    W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    from docx import Document

    def text_of(p):
        return [("".join(t.text or "" for t in x.iter(W + "t"))) for x in Document(p).element.body.iter(W + "p")]

    changed_pilot = []
    for e in prod:
        if e["filename"] in replaced:
            stats["original entries replaced"] += 1
            continue
        e = dict(e)
        s = staged.get(e["filename"])
        if s and s.get("fields") != e.get("fields"):
            e["fields"] = s["fields"]
            stats["pilot entries: fields fix applied"] += 1
        n = pilot_by_key.get(key(e.get("label", "")))
        if n and text_of(os.path.join(pilot_dir, n)) != text_of(os.path.join(pilot_src, n)):
            e["sections"], _ = trim_sections(sections_of(os.path.join(pilot_dir, n)))
            changed_pilot.append((os.path.join(pilot_dir, n), e["filename"]))
        catalog.append(e)
    stats["pilot documents corrected"] = len(changed_pilot)

    # --- new entries ------------------------------------------------------------
    known = collections.defaultdict(list)
    for e in catalog:
        for c in e.get("categorie") or []:
            if c and c not in known[e.get("codice", "")]:
                known[e.get("codice", "")].append(c)
    new, review = [], []
    for inp in inputs["entries"]:
        r = results.get(inp["id"])
        if r is None:
            stats["new: no model result"] += 1
            continue
        d = r["data"]
        cat = str(d.get("categoria") or "").strip()
        cat = match_category(cat, known[inp["codice"]]) if cat else ""
        if cat and cat not in known[inp["codice"]]:
            known[inp["codice"]].append(cat)
        sub = str(d.get("sottocategoria") or "").strip()
        entry = {
            "filename": inp["filename"],
            "codice": inp["codice"],
            "categorie": [cat] if cat else [],
            "sottocategorie": [sub] if sub else [],
            "tipo_atto": inp["label"],
            "label": inp["label"],
            "description": str(d.get("description") or "").strip(),
            "fields": clean_fields(d.get("fields"), args.cap, stats),
            "flagged_issues": list(inp["flagged_issues"]),
            "sections": inp["sections"],
        }
        if not entry["description"]:
            entry["flagged_issues"].append("descrizione mancante")
        if not entry["categorie"]:
            entry["flagged_issues"].append("categoria mancante")
        new.append((entry, inp["source_docx"]))
        review.append([entry["label"], entry["codice"], cat, sub, entry["description"], str(len(entry["fields"])),
                       ", ".join(entry["fields"]), "; ".join(entry["flagged_issues"]), entry["filename"]])
    catalog += [e for e, _ in new]

    # --- stage -------------------------------------------------------------------
    docs = os.path.join(STAGE, "docs")
    if os.path.exists(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(docs)
    for e, src in new:
        shutil.copy2(os.path.join(BASE, "converted_all_final", src), os.path.join(docs, e["filename"]))
    for src, fn in changed_pilot:
        shutil.copy2(src, os.path.join(docs, fn))
    json.dump(catalog, open(os.path.join(STAGE, "catalog_enriched.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(os.path.join(WORK, "review_nuovi_modelli.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["titolo", "area", "categoria", "sottocategoria", "descrizione", "n. campi", "campi", "problemi", "filename"])
        w.writerows(review)

    # --- checks ------------------------------------------------------------------
    fns = [e["filename"] for e in catalog]
    dup = [k for k, v in collections.Counter(fns).items() if v > 1]
    missing = [e["filename"] for e, _ in new if not os.path.exists(os.path.join(docs, e["filename"]))]
    after = stats.pop("after signature drop")
    print("catalog: %d entries (production %d, minus %d replaced, plus %d new)" % (
        len(catalog), len(prod), stats["original entries replaced"], len(new)))
    for k, v in sorted(stats.items()):
        print("  %-40s %d" % (k, v))
    print("fields per new template after dropping signatures: median %d | over 25: %d | over 30: %d | max %d" % (
        statistics.median(after), sum(1 for x in after if x > 25), sum(1 for x in after if x > 30), max(after)))
    print("categories: %d distinct across the catalog" % len({c for e in catalog for c in e.get("categorie") or []}))
    print("duplicate filenames: %d | staged docs: %d (missing: %d)" % (len(dup), len(os.listdir(docs)), len(missing)))
    print("catalog file: %.1f MB -> %s" % (os.path.getsize(os.path.join(STAGE, "catalog_enriched.json")) / 1e6, STAGE))


if __name__ == "__main__":
    main()
