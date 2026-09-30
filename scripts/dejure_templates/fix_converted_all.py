"""Fix the converted DEJURE templates before upload. Writes corrected copies; never
touches the source folder.

1. Words glued after an accented letter ("puòessere" -> "può essere"). pdf2docx
   drops the space after some accented glyphs. A space is added only when the PDF
   text has one there, or (grave accents only) when the PDF does not have the glued
   form either: Italian words never carry a grave accent mid-word. Acute "é" is
   split only when the PDF confirms it ("perchéil"), since French names use it
   mid-word ("Société").
2. Filenames garbled by the zip (UTF-8 read as cp437: "╠Ç" = combining accent,
   "ΓÇ£" = “ ...) are decoded back.
3. Text normalised to NFC (accent + separate mark -> one character).
4. Empty "Autori:" lines left by the converter are removed.

Only w:t text nodes are edited, so run formatting, tabs, breaks, fields and
images are untouched. Every change adds a space or normalises a character:
the check at the end compares each paragraph with spaces removed.

    python fix_converted_all.py [sample_size]
"""
import csv
import multiprocessing as mp
import os
import random
import re
import sys
import time
import unicodedata

BASE = r"C:\Users\anton\Downloads\downloads\downloads"
DATA = os.environ.get("DEJURE_DATA", r"C:\Users\anton\Downloads\catalog_enrich")   # working files, outside the repo
# Defaults are the full batch; FIX_SRC/FIX_DST/FIX_LOG/FIX_PDFMAP (a JSON of
# docx name -> PDF path) point it at another set, e.g. the pilot files.
SRC = os.environ.get("FIX_SRC", os.path.join(BASE, "converted_all"))
DST = os.environ.get("FIX_DST", os.path.join(BASE, "converted_all_fixed"))
LOG = os.environ.get("FIX_LOG", os.path.join(DATA, "fix_converted_all.csv"))
PDF_FOR = {}
if os.environ.get("FIX_PDFMAP"):
    import json as _json
    PDF_FOR = _json.load(open(os.environ["FIX_PDFMAP"], encoding="utf-8"))

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"

GRAVE = set("àèìòùÀÈÌÒÙ")
ACUTE = set("éÉ")
LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
AUTORI_EMPTY = re.compile(r"^\s*Autori\s*:\s*$", re.I)
# "conformità̀": an accented letter followed by the same accent again.
DOUBLE_ACCENT = re.compile("([àèìòùÀÈÌÒÙ])̀+|([áéíóúÁÉÍÓÚ])́+")


def nfc(s):
    return DOUBLE_ACCENT.sub(lambda m: m.group(1) or m.group(2), unicodedata.normalize("NFC", s))


def long_path(p):
    return "\\\\?\\" + os.path.abspath(p) if not p.startswith("\\\\?\\") else p


def repair_name(name):
    """'imparzialita╠Ç' -> 'imparzialità', 'ΓÇ£x ΓÇ¥' -> '“x ”'."""
    def fix(m):
        s = m.group(0)
        try:
            return s.encode("cp437").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return s
    return unicodedata.normalize("NFC", re.sub(r"[^\x00-\x7f]+", fix, name))


def pdf_text(path):
    import fitz
    with fitz.open(path) as d:
        t = "\n".join(p.get_text() for p in d)
    t = re.sub(r"(\w)-[ \t]*\n\s*(\w)", r"\1\2", unicodedata.normalize("NFC", t))   # "qua-/ter" -> "quater"
    return re.sub(r"\s+", " ", t).lower()


PLAIN_WORD = re.compile(r"[^\W\d_]+")


def pdf_vocabulary(pdf):
    """Words of the PDF and the pairs that stand side by side in it."""
    ws = PLAIN_WORD.findall(pdf)
    return set(ws), set(zip(ws, ws[1:]))


def plain_split(left, right, vocab):
    """Glued with no accent involved ("FALLIMENTARE|CON", "pro|tempore"): split
    only if the PDF has the two words side by side and never the glued form."""
    if not vocab or len(left) < 2 or len(right) < 2:
        return False
    words, pairs = vocab
    glued = (left + right).lower()
    return glued not in words and glued not in ACCENTED_WORDS and (left.lower(), right.lower()) in pairs


def is_letter(ch):
    return bool(ch) and bool(LETTER.match(ch))


def word_left(s, end):
    i = end
    while i > 0 and is_letter(s[i - 1]):
        i -= 1
    return s[i:end]


def word_right(s, start):
    i = start
    while i < len(s) and is_letter(s[i]):
        i += 1
    return s[start:i]


# Real words written with a mid-word accent to tell them apart ("il giudice
# adìto" vs "àdito", "subìto" vs "sùbito"). Never split, whatever the PDF shows.
ACCENTED_WORDS = {
    "adìto", "adìta", "adìti", "adìte", "àdito", "subìto", "subìta", "subìti", "subìte",
    "sùbito", "àncora", "ancòra", "princìpi", "prìncipi", "sèguito", "sùbiti",
}


def should_split(left, right, pdf, stats):
    """left ends with the accented letter, right starts the next word."""
    if len(left) < 2 or not right:
        return False
    if (left + right).lower() in ACCENTED_WORDS:
        stats["kept_as_in_pdf"] += 1
        return False
    lo, ro = left[-8:].lower(), right[:8].lower()
    spaced, glued = lo + " " + ro, lo + ro
    if pdf is not None and spaced in pdf:
        return True
    if left[-1] in ACUTE:
        return False                     # only with PDF confirmation
    if pdf is not None and glued in pdf:
        stats["kept_as_in_pdf"] += 1     # the PDF itself has it glued
        return False
    return True


def fix_paragraph(p, pdf, stats, vocab=None):
    nodes = [t for t in p.iter(W + "t")]
    if not nodes:
        return
    # An accent mark that opens a node belongs to the letter closing the node
    # before it ("a" | "̀"): move it across so NFC can join them.
    for a, b in zip(nodes, nodes[1:]):
        sb = b.text or ""
        k = 0
        while k < len(sb) and unicodedata.combining(sb[k]):
            k += 1
        if k and a.text:
            a.text, b.text = a.text + sb[:k], sb[k:]
            a.set(XML_SPACE, "preserve")
            b.set(XML_SPACE, "preserve")
            stats["nfc_nodes"] += 1
    # NFC per node
    for t in nodes:
        if t.text and t.text != nfc(t.text):
            t.text = nfc(t.text)
            stats["nfc_nodes"] += 1
    # inside one node
    for t in nodes:
        s = t.text or ""
        if not any(ch in GRAVE or ch in ACUTE for ch in s):
            continue
        out, last = [], 0
        for i in range(1, len(s) - 1):
            if (s[i] in GRAVE or s[i] in ACUTE) and is_letter(s[i - 1]) and is_letter(s[i + 1]):
                left, right = word_left(s, i + 1), word_right(s, i + 1)
                if should_split(left, right, pdf, stats):
                    out.append(s[last:i + 1] + " ")
                    last = i + 1
                    stats["split_inside"] += 1
        if last:
            out.append(s[last:])
            t.text = "".join(out)
            t.set(XML_SPACE, "preserve")
    # Across consecutive nodes, judged on the whole paragraph: the converter often
    # gives an accented letter a run of its own ("modalit" | "à" | "di"), so the
    # word on either side of a boundary can span several nodes.
    texts = [t.text or "" for t in nodes]
    full = "".join(texts)
    off = 0
    for k in range(len(nodes) - 1):
        off += len(texts[k])
        if off == 0 or off >= len(full) or not (is_letter(full[off - 1]) and is_letter(full[off])):
            continue
        left, right = word_left(full, off), word_right(full, off)
        if full[off - 1] in GRAVE or full[off - 1] in ACUTE:        # "può" | "essere"
            if len(left) >= 2 and should_split(left, right, pdf, stats):
                kind = "split_across"
            else:
                continue
        elif plain_split(left, right, vocab):                      # "pro" | "tempore", "in" | "équipe"
            kind = "split_plain"
        else:
            continue
        nodes[k].text = (nodes[k].text or "") + " "
        nodes[k].set(XML_SPACE, "preserve")
        stats[kind] += 1
    # glued with no accent, inside one node ("DIRISCHIO")
    for t in nodes:
        s = t.text or ""
        out, last = [], 0
        for m in PLAIN_WORD.finditer(s):
            w = m.group(0)
            if len(w) < 5 or not vocab or w.lower() in vocab[0]:
                continue
            for i in range(2, len(w) - 1):
                if plain_split(w[:i], w[i:], vocab):
                    out.append(s[last:m.start() + i] + " ")
                    last = m.start() + i
                    stats["split_plain"] += 1
                    break
        if last:
            out.append(s[last:])
            t.text = "".join(out)
            t.set(XML_SPACE, "preserve")


def para_text(p):
    return "".join(t.text or "" for t in p.iter(W + "t"))


def process(name):
    from docx import Document
    stats = {"file": name, "new_name": repair_name(name), "split_inside": 0, "split_across": 0,
             "split_plain": 0, "kept_as_in_pdf": 0, "nfc_nodes": 0, "autori_removed": 0, "text_check": "ok"}
    try:
        doc = Document(long_path(os.path.join(SRC, name)))
        pdf_path = PDF_FOR.get(name) or os.path.join(BASE, os.path.splitext(name)[0] + ".pdf")
        pdf = pdf_text(long_path(pdf_path)) if os.path.exists(long_path(pdf_path)) else None
        vocab = pdf_vocabulary(pdf) if pdf is not None else None
        stats["has_pdf"] = int(pdf is not None)
        body = doc.element.body
        before = [unicodedata.normalize("NFC", para_text(p)) for p in body.iter(W + "p")]
        for p in list(body.iter(W + "p")):
            if AUTORI_EMPTY.match(para_text(p)):
                p.getparent().remove(p)
                stats["autori_removed"] += 1
                continue
            fix_paragraph(p, pdf, stats, vocab)
        after = [para_text(p) for p in body.iter(W + "p")]
        stats["accents_still_split"] = sum(1 for x in after for ch in x if unicodedata.combining(ch))
        # Only spaces may have been added (after dropping removed Autori lines).
        squash = lambda xs: [re.sub(r"\s+", "", nfc(x))
                             for x in xs if not AUTORI_EMPTY.match(x)]
        if squash(before) != squash(after):
            stats["text_check"] = "CHANGED"
        doc.save(long_path(os.path.join(DST, stats["new_name"])))
    except Exception as exc:
        stats["text_check"] = "ERROR: %s" % str(exc)[:150]
    return stats


def main():
    os.makedirs(long_path(DST), exist_ok=True)
    names = sorted(n for n in os.listdir(long_path(SRC)) if n.lower().endswith(".docx"))
    new_names = [repair_name(n) for n in names]
    dupes = {n for n in new_names if new_names.count(n) > 1}
    if dupes:
        sys.exit("repaired names collide: %s" % sorted(dupes)[:5])
    if len(sys.argv) > 1:
        random.seed(11)
        names = random.sample(names, int(sys.argv[1]))
    t0 = time.time()
    rows = []
    with mp.Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for i, r in enumerate(pool.imap_unordered(process, names, chunksize=8), 1):
            rows.append(r)
            if i % 500 == 0:
                print("  %d/%d (%.0fs)" % (i, len(names), time.time() - t0), flush=True)
    fields = ["file", "new_name", "split_inside", "split_across", "split_plain", "kept_as_in_pdf",
              "nfc_nodes", "accents_still_split", "autori_removed", "has_pdf", "text_check"]
    with open(LOG, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["file"]))
    print("\nfiles: %d in %.0fs  ->  %s" % (len(rows), time.time() - t0, DST))
    print("renamed (garbled names repaired): %d" % sum(1 for r in rows if r["new_name"] != r["file"]))
    print("spaces added: %d inside runs, %d between runs, in %d files" % (
        sum(r["split_inside"] for r in rows), sum(r["split_across"] for r in rows),
        sum(1 for r in rows if r["split_inside"] or r["split_across"])))
    print("spaces added where no accent is involved: %d" % sum(r["split_plain"] for r in rows))
    print("left glued because the PDF has it glued: %d" % sum(r["kept_as_in_pdf"] for r in rows))
    print("NFC-normalised text nodes: %d in %d files" % (
        sum(r["nfc_nodes"] for r in rows), sum(1 for r in rows if r["nfc_nodes"])))
    print("empty 'Autori:' lines removed: %d" % sum(r["autori_removed"] for r in rows))
    print("accent marks still split after the fix: %d" % sum(r.get("accents_still_split", 0) for r in rows))
    bad = [r for r in rows if r["text_check"] != "ok"]
    print("text check (only spaces added): %d ok, %d NOT ok" % (len(rows) - len(bad), len(bad)))
    for r in bad[:10]:
        print("   ", r["file"][:70], r["text_check"])


if __name__ == "__main__":
    main()
