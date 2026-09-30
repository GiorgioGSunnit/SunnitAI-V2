"""Words glued without an accent ("FALLIMENTARECON"): a DOCX word absent from the
PDF that splits into two words the PDF has side by side. Read-only.

    python glued_any.py [folder]   (default converted_all_final)
"""
import collections
import csv
import multiprocessing as mp
import os
import re
import sys
import unicodedata

BASE = r"C:\Users\anton\Downloads\downloads\downloads"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DEJURE_DATA", r"C:\Users\anton\Downloads\catalog_enrich")   # working files, outside the repo
FOLDER = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "converted_all_final")
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
WORD = re.compile(r"[^\W\d_]+")


def check(job):
    name, pdf_name = job
    import fitz
    from docx import Document
    pdf = os.path.join(BASE, os.path.splitext(pdf_name)[0] + ".pdf")
    if not os.path.exists(pdf):
        return name, []
    with fitz.open(pdf) as d:
        ptxt = unicodedata.normalize("NFC", " ".join(p.get_text() for p in d)).lower()
    pw = WORD.findall(ptxt)
    words, bigrams = set(pw), set(zip(pw, pw[1:]))
    text = " ".join("".join(t.text or "" for t in p.iter(W + "t"))
                    for p in Document(os.path.join(FOLDER, name)).element.body.iter(W + "p"))
    found = []
    for w in WORD.findall(unicodedata.normalize("NFC", text)):
        lw = w.lower()
        if len(lw) < 5 or lw in words:
            continue
        for i in range(2, len(lw) - 1):
            if (lw[:i], lw[i:]) in bigrams:
                found.append(w)
                break
    return name, found


if __name__ == "__main__":
    pdf_for = {r["new_name"]: r["file"] for r in csv.DictReader(open(os.path.join(DATA, "fix_converted_all.csv"), encoding="utf-8-sig"))}
    names = sorted(n for n in os.listdir(FOLDER) if n.endswith(".docx"))
    with mp.Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        res = pool.map(check, [(n, pdf_for.get(n, n)) for n in names], chunksize=8)
    allw = collections.Counter(w for _, f in res for w in f)
    files = [(n, f) for n, f in res if f]
    print("glued words (any letter): %d in %d of %d files" % (sum(allw.values()), len(files), len(res)))
    print("most common:", ", ".join("%s x%d" % (w, c) for w, c in allw.most_common(30)))
    with open(os.path.join(DATA, "glued_any.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "count", "words"])
        for n, fl in sorted(files, key=lambda x: -len(x[1])):
            w.writerow([n, len(fl), " ".join(fl)])
