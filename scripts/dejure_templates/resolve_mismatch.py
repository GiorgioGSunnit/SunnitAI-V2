"""Decide, for each mislabelled file, whether to rename it (its real form exists
nowhere else) or drop it (its real form already exists under the right name).
Also lists the named forms that were never downloaded, and any overlap with the
pilot files on production. Writes mismatch_plan.csv. Read-only on the templates."""
import csv
import os
import re
import unicodedata

BASE = r"C:\Users\anton\Downloads\downloads\downloads"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DEJURE_DATA", r"C:\Users\anton\Downloads\catalog_enrich")   # working files, outside the repo
PILOT = os.path.join(BASE, "converted")


def key(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", re.sub(r"\.(pdf|docx)$", "", s)).strip()


rows = list(csv.DictReader(open(os.path.join(DATA, "pdf_title_mismatch_classified.csv"), encoding="utf-8-sig")))
fixlog = list(csv.DictReader(open(os.path.join(DATA, "fix_converted_all.csv"), encoding="utf-8-sig")))
docx_for_pdf = {os.path.splitext(r["file"])[0]: r["new_name"] for r in fixlog}
all_titles = {key(r["new_name"]) for r in fixlog}
pilot = {key(n) for n in os.listdir(PILOT) if n.endswith(".docx")}

def safe_name(title):
    return re.sub(r'[\\/:*?"<>|]', "_", title.strip()).rstrip(". ")


mislabelled = {key(r["name_title"]) for r in rows if r["category"] not in ("code-header", "contains")}
correct = all_titles - mislabelled        # files whose name matches their content

plan, never_downloaded = [], []
taken = set()
# swapped pairs first: each takes the other's name
order = sorted(rows, key=lambda r: {"swapped": 0}.get(r["category"], 1))
for r in order:
    cat = r["category"]
    docx = docx_for_pdf.get(os.path.splitext(r["pdf"])[0], "")
    hk = key(r["header_title"])
    if cat in ("code-header", "contains"):
        action, target = "keep", docx
    elif hk in correct or hk in taken:
        action, target = "drop", ""       # its real form is already there, correctly named
    else:
        action, target = "rename", safe_name(r["header_title"]) + ".docx"
        taken.add(hk)
    if cat not in ("code-header", "contains", "swapped"):
        never_downloaded.append(r["name_title"])
    plan.append({"category": cat, "action": action, "docx": docx, "rename_to": target,
                 "named_title": r["name_title"], "real_content": r["header_title"],
                 "pilot_on_production": int(key(r["name_title"]) in pilot)})

with open(os.path.join(DATA, "mismatch_plan.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(plan[0]))
    w.writeheader()
    w.writerows(plan)

from collections import Counter
print(Counter((p["category"], p["action"]) for p in plan))
print("\nrenames:")
for p in plan:
    if p["action"] == "rename":
        print("   %-60s -> %s" % (p["named_title"][:60], p["rename_to"][:70]))
print("\ndrops (their real content already exists under the right name):")
for p in plan:
    if p["action"] == "drop":
        print("   %-60s  (is really: %s)" % (p["named_title"][:60], p["real_content"][:55]))
print("\nforms named but never actually downloaded: %d" % len(never_downloaded))
print("mismatched files that are pilot templates already on production: %d" % sum(p["pilot_on_production"] for p in plan))
for p in plan:
    if p["pilot_on_production"]:
        print("   ", p["named_title"], "|", p["category"], "|", p["real_content"][:60])
