"""Does each PDF's own header title (the lines before 'Autori:') match its filename?
A mismatch means the file holds a different formulario than its name says. Read-only."""
import csv
import multiprocessing as mp
import os
import re
import unicodedata

BASE = r"C:\Users\anton\Downloads\downloads\downloads"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DEJURE_DATA", r"C:\Users\anton\Downloads\catalog_enrich")   # working files, outside the repo
AUTORI = re.compile(r"^\s*Autori\s*:", re.I)


def key(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def repair(n):
    def fix(m):
        try:
            return m.group(0).encode("cp437").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return m.group(0)
    return unicodedata.normalize("NFC", re.sub(r"[^\x00-\x7f]+", fix, n))


def check(pdf_name):
    import fitz
    with fitz.open(os.path.join(BASE, pdf_name)) as d:
        lines = [l.strip() for l in d[0].get_text().splitlines() if l.strip()]
    a = next((i for i, l in enumerate(lines[:8]) if AUTORI.match(l)), None)
    header = " ".join(lines[:a]) if a else (lines[0] if lines else "")
    name = repair(os.path.splitext(pdf_name)[0])
    kh, kn = key(header), key(name)
    same = kh == kn or kh.startswith(kn) or kn.startswith(kh) or kh.replace(" ", "")[:40] == kn.replace(" ", "")[:40]
    return {"pdf": pdf_name, "name_title": name, "header_title": header[:200], "has_autori": int(a is not None),
            "match": int(same)}


if __name__ == "__main__":
    pdfs = sorted(n for n in os.listdir(BASE) if n.lower().endswith(".pdf"))
    with mp.Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        rows = pool.map(check, pdfs, chunksize=16)
    bad = [r for r in rows if not r["match"]]
    print("PDFs: %d | header title matches the filename: %d | DIFFERENT: %d" % (len(rows), len(rows) - len(bad), len(bad)))
    with open(os.path.join(DATA, "pdf_title_mismatch.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(bad)
    for r in bad[:25]:
        print("  name:   %s\n  header: %s\n" % (r["name_title"][:100], r["header_title"][:100]))
