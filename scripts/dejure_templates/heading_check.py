"""How often did the colleague's converter drop the act's own heading?

For each file: does the PDF contain the document title written as an UPPERCASE
heading (the act's own heading, not DEJURE's title-case page header)? If so,
does the converted DOCX still contain it in uppercase? Also: is the first DOCX
paragraph the title (the line the new converter adds at the top)? Read-only.
"""
import csv
import multiprocessing as mp
import os
import re
import unicodedata

BASE = r"C:\Users\anton\Downloads\downloads\downloads"
FIXED = os.path.join(BASE, os.environ.get("HEADING_DIR", "converted_all_fixed"))
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DEJURE_DATA", r"C:\Users\anton\Downloads\catalog_enrich")   # working files, outside the repo
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def norm(s):
    s = unicodedata.normalize("NFC", s)
    s = s.replace("’", "'").replace("_", "/")
    return re.sub(r"\s+", " ", s).strip()


def check(pair):
    new_name, orig = pair
    import fitz
    from docx import Document
    title = norm(os.path.splitext(new_name)[0])
    if len(title) < 12:
        return None
    pdf_path = os.path.join(BASE, os.path.splitext(orig)[0] + ".pdf")
    if not os.path.exists(pdf_path):
        return None
    with fitz.open(pdf_path) as d:
        pdf = norm(" ".join(p.get_text() for p in d))
    paras = [norm("".join(t.text or "" for t in p.iter(W + "t"))) for p in Document(os.path.join(FIXED, new_name)).element.body.iter(W + "p")]
    paras = [p for p in paras if p]
    docx = " ".join(paras)
    probe = title.upper()[:60]            # long titles: the heading may wrap or be abridged
    return {
        "file": new_name,
        "pdf_has_upper_heading": int(probe in pdf),
        "docx_has_upper_heading": int(probe in docx),
        "docx_starts_with_title": int(bool(paras) and paras[0].lower()[:60] == title.lower()[:60]),
    }


def main():
    log = list(csv.DictReader(open(os.path.join(DATA, "fix_converted_all.csv"), encoding="utf-8-sig")))
    with mp.Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        rows = [r for r in pool.imap_unordered(check, [(r["new_name"], r["file"]) for r in log], chunksize=8) if r]
    has = [r for r in rows if r["pdf_has_upper_heading"]]
    lost = [r for r in has if not r["docx_has_upper_heading"]]
    print("files checked: %d" % len(rows))
    print("converted file starts with the title line: %d" % sum(r["docx_starts_with_title"] for r in rows))
    print("PDF has the title as an UPPERCASE act heading: %d" % len(has))
    print("   ...and the converted file no longer has it: %d" % len(lost))
    with open(os.path.join(DATA, "heading_lost.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(lost, key=lambda r: r["file"]))
    for r in sorted(lost, key=lambda r: r["file"])[:10]:
        print("     ", r["file"][:90])


if __name__ == "__main__":
    main()
