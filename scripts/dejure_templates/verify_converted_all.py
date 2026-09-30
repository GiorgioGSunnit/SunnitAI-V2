"""Batch check of the full DEJURE conversion (converted_all) against the pilot's standard.

Per file: opens, share of the PDF's words present in the DOCX in the same order,
DEJURE header/footer leftovers, logo images, text in Word's page header/footer,
"...." blanks not converted outside notes, [DA COMPILARE] count, tables.
Read-only. Writes verify_converted_all.csv next to this script and prints a summary.

    python verify_converted_all.py            # all files
    python verify_converted_all.py 150        # random sample, for timing
"""
import collections
import csv
import difflib
import multiprocessing as mp
import os
import random
import re
import sys
import time
import unicodedata

BASE = r"C:\Users\anton\Downloads\downloads\downloads"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DEJURE_DATA", r"C:\Users\anton\Downloads\catalog_enrich")   # working files, outside the repo
# VERIFY_FIXED=1 checks the corrected copies; their names were repaired, so the
# source PDF is found through the fix log. Set via env so pool workers see it.
FIXED = os.environ.get("VERIFY_FIXED") in ("1", "final")
STAGE = {"1": "converted_all_fixed", "final": "converted_all_final"}.get(os.environ.get("VERIFY_FIXED"), "converted_all")
DOCX_DIR = os.path.join(BASE, STAGE)
OUT = os.path.join(DATA, "verify_%s.csv" % STAGE)
ACCENT_OF = {"a": "à", "e": "è", "i": "ì", "o": "ò", "u": "ù"}
PDF_STEM = {}
if FIXED:
    with open(os.path.join(DATA, "fix_converted_all.csv"), encoding="utf-8-sig") as _f:
        PDF_STEM = {r["new_name"]: os.path.splitext(r["file"])[0] for r in csv.DictReader(_f)}
GLUED = re.compile(r"[^\W\d_]+[àèìòùÀÈÌÒÙ][^\W\d_]{2,}")

# Same patterns as convert_dejure_templates.py
AUTORI = re.compile(r"^\s*Autori\s*:", re.I)
COLLECTION = re.compile(r"^\s*FORMULARIO\b", re.I)
FOOTER = re.compile(r"(astrea@sunnit\.ai|©\s*Copyright|Copyright\s+Giu)", re.I)
BLANK = re.compile(r"\[\s*(?:…[.…]*|\.{3,})\s*\]|…[.…]*|\.{3,}")
NOTE_START = re.compile(r"^\s*\[\d{1,3}\]")
WORD = re.compile(r"\w+", re.UNICODE)

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def paragraphs(part_element):
    for p in part_element.iter(W + "p"):
        yield "".join(t.text or "" for t in p.iter(W + "t"))


def pdf_words(path):
    import fitz
    words = []
    with fitz.open(path) as doc:
        for page in doc:
            for line in page.get_text("text").splitlines():
                if FOOTER.search(line) or AUTORI.match(line) or COLLECTION.match(line):
                    continue            # removed on purpose by the converter
                words += WORD.findall(unicodedata.normalize("NFC", line).lower())
    return words


def check(name):
    from docx import Document
    row = {"file": name, "opens": 1}
    try:
        doc = Document(os.path.join(DOCX_DIR, name))
    except Exception as exc:
        row.update(opens=0, error=str(exc)[:120])
        return row
    body = doc.element.body
    raw_texts = [t.strip() for t in paragraphs(body) if t.strip()]
    # Decomposed accents ("a" + U+0300 instead of "à"): look the same in Word,
    # but are different strings to search, keyword matching and file paths.
    row["nfd_in_name"] = int(name != unicodedata.normalize("NFC", name))
    row["nfd_chars_in_text"] = sum(1 for t in raw_texts for ch in t if unicodedata.combining(ch))
    texts = [unicodedata.normalize("NFC", t) for t in raw_texts]
    row["paragraphs"] = len(texts)
    row["chars"] = sum(len(t) for t in texts)
    row["da_compilare"] = sum(t.count("[DA COMPILARE]") for t in texts)
    row["blanks_left"] = sum(len(BLANK.findall(t)) for t in texts if not NOTE_START.match(t))
    row["header_footer_text"] = sum(1 for t in texts if FOOTER.search(t) or AUTORI.match(t) or COLLECTION.match(t))
    row["images"] = sum(1 for _ in body.iter(W + "drawing")) + sum(
        1 for el in body.iter() if isinstance(el.tag, str) and el.tag.endswith("}imagedata"))
    row["tables"] = sum(1 for _ in body.iter(W + "tbl"))
    hf = 0
    for s in doc.sections:
        for part in (s.header, s.footer):
            try:
                hf += sum(1 for t in paragraphs(part._element) if t.strip())
            except Exception:
                pass
    row["word_header_footer_lines"] = hf

    row["glued_words"] = len(GLUED.findall(" ".join(texts)))
    pdf = os.path.join(BASE, PDF_STEM.get(name, os.path.splitext(name)[0]) + ".pdf")
    if os.path.exists(pdf):
        try:
            src = pdf_words(pdf)
            out = WORD.findall(" ".join(texts).lower())
            sm = difflib.SequenceMatcher(None, src, out, autojunk=False)
            kept = sum(b.size for b in sm.get_matching_blocks())
            row["pdf_words"] = len(src)
            row["words_kept_pct"] = round(100 * kept / max(1, len(src)), 2)
            # Present anywhere, regardless of position: blocks the PDF reader
            # orders differently (alternative clauses, tables) are not losses.
            missing = collections.Counter(src) - collections.Counter(out)
            lost = sum(missing.values())
            row["words_present_pct"] = round(100 * (len(src) - lost) / max(1, len(src)), 2)
            # final accent dropped ("Lunedi" where the PDF has "Lunedì")
            src_set = set(src)
            row["accent_lost"] = sum(1 for w in set(out) if len(w) >= 4 and w not in src_set
                                     and w[-1] in ACCENT_OF and (w[:-1] + ACCENT_OF[w[-1]]) in src_set)
            row["missing_words"] = lost
            row["missing_sample"] = " ".join(w for w, _ in missing.most_common(15))
        except Exception as exc:
            row["error"] = ("pdf: " + str(exc))[:120]
    return row


def main():
    names = sorted(f for f in os.listdir(DOCX_DIR) if f.lower().endswith(".docx"))
    if len(sys.argv) > 1:
        random.seed(7)
        names = random.sample(names, int(sys.argv[1]))
    t0 = time.time()
    rows = []
    with mp.Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for i, row in enumerate(pool.imap_unordered(check, names, chunksize=8), 1):
            rows.append(row)
            if i % 250 == 0:
                print("  %d/%d  (%.0fs)" % (i, len(names), time.time() - t0), flush=True)

    fields = ["file", "opens", "glued_words", "accent_lost", "words_present_pct", "missing_words", "missing_sample",
              "nfd_in_name", "nfd_chars_in_text",
              "words_kept_pct", "pdf_words", "paragraphs", "chars", "da_compilare",
              "blanks_left", "header_footer_text", "images", "word_header_footer_lines", "tables", "error"]
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["file"]):
            w.writerow(r)

    ok = [r for r in rows if r.get("opens")]
    print("\nfiles checked: %d in %.0fs" % (len(rows), time.time() - t0))
    print("do not open: %d" % (len(rows) - len(ok)))
    pres = [r for r in ok if "words_present_pct" in r]
    if pres:
        lost = sorted(r["missing_words"] for r in pres)
        print("PDF words missing from the DOCX (any position): median %d, max %d"
              % (lost[len(lost) // 2], lost[-1]))
        for lim in (0, 5, 20, 50):
            print("   files missing more than %d words: %d" % (lim, sum(1 for n in lost if n > lim)))
    print("%-45s %d words in %d files" % ("glued after an accent:", sum(r.get("glued_words", 0) for r in ok),
                                           sum(1 for r in ok if r.get("glued_words"))))
    print("%-45s %d files" % ("garbled filename:", sum(1 for r in ok if re.search(r"[─-╿Γ]", r["file"]))))
    print("%-45s %d words in %d files" % ("final accent dropped (Lunedi/Lunedì):", sum(r.get("accent_lost", 0) for r in ok),
                                           sum(1 for r in ok if r.get("accent_lost"))))
    print("%-45s %d files" % ("decomposed accents in the FILENAME:", sum(1 for r in ok if r.get("nfd_in_name"))))
    print("%-45s %d files (%d marks)" % ("decomposed accents in the TEXT:",
          sum(1 for r in ok if r.get("nfd_chars_in_text")), sum(r.get("nfd_chars_in_text", 0) for r in ok)))
    for key, label in (("header_footer_text", "DEJURE header/footer text left"),
                       ("images", "images (logo?) left"),
                       ("word_header_footer_lines", "text in Word page header/footer"),
                       ("blanks_left", "'....' blanks not converted (outside notes)")):
        n = sum(1 for r in ok if r.get(key))
        print("%-45s %d files" % (label + ":", n))
    print("%-45s %d files" % ("with tables:", sum(1 for r in ok if r.get("tables"))))
    print("%-45s %d files" % ("no [DA COMPILARE] at all:", sum(1 for r in ok if not r.get("da_compilare"))))
    print("pdf errors: %d" % sum(1 for r in rows if str(r.get("error", "")).startswith("pdf")))
    print("\n-> %s" % OUT)


if __name__ == "__main__":
    main()
