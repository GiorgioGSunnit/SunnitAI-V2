"""Classify the 59 PDFs whose header title differs from their filename. Read-only.

  code-header : header is a code name ("Codice Civile") - different layout, not a mismatch
  swapped     : this file holds formulario Y, and the file named Y holds this one
  same-as     : identical text to another file (the same document saved under several names)
  contains    : the document also contains a heading matching its own filename
                (a multi-form document that includes the named form)
  other       : content belongs to the header title only - the filename is wrong
"""
import collections
import csv
import hashlib
import os
import re
import unicodedata

import fitz

BASE = r"C:\Users\anton\Downloads\downloads\downloads"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DEJURE_DATA", r"C:\Users\anton\Downloads\catalog_enrich")   # working files, outside the repo


def key(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


rows = list(csv.DictReader(open(os.path.join(DATA, "pdf_title_mismatch.csv"), encoding="utf-8-sig")))
by_name = {key(os.path.splitext(n)[0]): n for n in os.listdir(BASE) if n.lower().endswith(".pdf")}
# repaired names too
for r in rows:
    by_name.setdefault(key(r["name_title"]), r["pdf"])

text, digest = {}, {}
for r in rows:
    with fitz.open(os.path.join(BASE, r["pdf"])) as d:
        t = " ".join(p.get_text() for p in d)
    text[r["pdf"]] = key(t)
    digest[r["pdf"]] = hashlib.md5(key(t).encode()).hexdigest()
dupes = collections.Counter(digest.values())

out = collections.defaultdict(list)
for r in rows:
    h, n = key(r["header_title"]), key(r["name_title"])
    if h.startswith("codice"):
        out["code-header"].append(r)
        continue
    other = by_name.get(h)
    swapped = False
    if other and other != r["pdf"]:
        with fitz.open(os.path.join(BASE, other)) as d:
            lines = [l.strip() for l in d[0].get_text().splitlines() if l.strip()]
        swapped = bool(lines) and key(lines[0]).startswith(n[:40])
    if swapped:
        out["swapped"].append(r)
    elif dupes[digest[r["pdf"]]] > 1:
        out["same-as"].append(r)
    elif n[:40] in text[r["pdf"]][len(h):]:
        out["contains"].append(r)
    else:
        out["other"].append(r)

for cat in ("code-header", "swapped", "same-as", "contains", "other"):
    items = out[cat]
    print("%-12s %d" % (cat, len(items)))
    for r in items[:12]:
        print("     name: %-75s | header: %s" % (r["name_title"][:75], r["header_title"][:70]))
with open(os.path.join(DATA, "pdf_title_mismatch_classified.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["category", "pdf", "name_title", "header_title"])
    for cat, items in out.items():
        for r in items:
            w.writerow([cat, r["pdf"], r["name_title"], r["header_title"]])
