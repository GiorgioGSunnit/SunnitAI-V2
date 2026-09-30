"""Map each pilot DOCX to its source PDF (test folder first, then the main folder,
matching titles after repairing zip-garbled names). Writes the maps stage 1 and
stage 2 read."""
import csv
import json
import os
import re
import unicodedata

BASE = r"C:\Users\anton\Downloads\downloads\downloads"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DEJURE_DATA", r"C:\Users\anton\Downloads\catalog_enrich")   # working files, outside the repo


def repair(n):
    def fix(m):
        try:
            return m.group(0).encode("cp437").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return m.group(0)
    return unicodedata.normalize("NFC", re.sub(r"[^\x00-\x7f]+", fix, n))


def key(s):
    s = unicodedata.normalize("NFKD", repair(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", re.sub(r"\.(docx|pdf)$", "", s)).strip()


pdfs = {}
for folder in (os.path.join(BASE, "test folder"), BASE):
    for n in os.listdir(folder):
        if n.lower().endswith(".pdf"):
            pdfs.setdefault(key(n), os.path.join(folder, n))
pilot = sorted(n for n in os.listdir(os.path.join(BASE, "converted")) if n.endswith(".docx"))
mapping = {n: pdfs[key(n)] for n in pilot if key(n) in pdfs}
json.dump(mapping, open(os.path.join(DATA, "pilot_pdfmap.json"), "w", encoding="utf-8"), ensure_ascii=False)
with open(os.path.join(DATA, "pilot_pdfmap.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["new_name", "file"])
    for n, p in mapping.items():
        w.writerow([n, p])
print("pilot files: %d | source PDF found: %d | missing: %s" % (
    len(pilot), len(mapping), [n for n in pilot if n not in mapping]))
