"""List the forms that were named but never actually downloaded, with their DEJURE
links from the download log, for the colleague to re-fetch."""
import csv
import json
import os
import re
import unicodedata

BASE = r"C:\Users\anton\Downloads\downloads\downloads"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DEJURE_DATA", r"C:\Users\anton\Downloads\catalog_enrich")   # working files, outside the repo
OUT = r"C:\Users\anton\Downloads\formulari_da_riscaricare.csv"


def key(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


status = json.load(open(os.path.join(BASE, "_download_status.json"), encoding="utf-8"))["items"]
url_for = {}
for it in status:
    url_for.setdefault(key(it["name"]), it["url"])

plan = list(csv.DictReader(open(os.path.join(DATA, "mismatch_plan.csv"), encoding="utf-8-sig")))
todo = [p for p in plan if p["category"] not in ("code-header", "contains", "swapped")]
found = 0
with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f, delimiter=";")
    w.writerow(["titolo del formulario", "link DEJURE", "il file scaricato contiene invece"])
    for p in todo:
        k = key(p["named_title"])
        url = url_for.get(k) or next((u for kk, u in url_for.items() if kk.startswith(k[:50])), "")
        found += bool(url)
        w.writerow([p["named_title"], url, p["real_content"]])
print("written %s: %d forms, %d with their DEJURE link" % (OUT, len(todo), found))
