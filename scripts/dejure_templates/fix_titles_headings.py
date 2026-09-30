"""Stage 2 of the template fixes: DEJURE title line out, act heading back in.

The colleague's converter keeps the DEJURE page-header title as line 1 (against
the agreed rule: the header goes) and, where the act's own heading repeats that
title, deletes the heading as a duplicate. This undoes both:

1. Remove paragraph 1 when it is the document title written in normal case (the
   header). A title in CAPITALS is the act's own heading and stays.
2. Find the act heading in the PDF: a run of lines after the "Autori:" block
   whose text equals the title. If the DOCX no longer has it, insert it before
   the paragraph matching the PDF line that follows it (or after the one
   matching the line before it), copying the paragraph and run formatting of the
   neighbouring centred heading.

Reads converted_all_fixed (stage 1), writes converted_all_final. Every file is
checked: the only differences allowed are the removed title line and the
inserted headings.

    python fix_titles_headings.py [--src DIR --dst DIR --pdfmap CSV] [sample]
"""
import copy
import csv
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
ARGS = sys.argv[1:]


def arg(name, default):
    return ARGS[ARGS.index(name) + 1] if name in ARGS else default


SRC = arg("--src", os.path.join(BASE, "converted_all_fixed"))
DST = arg("--dst", os.path.join(BASE, "converted_all_final"))
PDFMAP = arg("--pdfmap", os.path.join(DATA, "fix_converted_all.csv"))   # new_name -> original name
LOG = arg("--log", os.path.join(DATA, "fix_titles_headings.csv"))

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
AUTORI = re.compile(r"^\s*Autori\s*:", re.I)
BLANKS = re.compile(r"\[DA COMPILARE\]|\[\s*(?:…[.…]*|\.{3,})\s*\]|…[.…]*|\.{3,}|_{3,}")


def long_path(p):
    return "\\\\?\\" + os.path.abspath(p) if not p.startswith("\\\\?\\") else p


def key(s):
    """Comparison form: case, accents' encoding, quotes, blanks and spacing ignored."""
    s = unicodedata.normalize("NFC", s).replace("’", "'").replace("_", "/")
    s = BLANKS.sub(" ", s)
    return re.sub(r"[^\w]+", " ", s.lower()).strip()


def ptext(p_el):
    return "".join(t.text or "" for t in p_el.iter(W + "t"))


def body_paragraphs(doc):
    return [p for p in doc.element.body.iter(W + "p") if ptext(p).strip()]


def pdf_lines(path):
    import fitz
    with fitz.open(long_path(path)) as d:
        return [l.strip() for pg in d for l in pg.get_text().splitlines() if l.strip()]


def find_act_heading(lines, title_key):
    """(heading_lines, prev_line, next_line, following_lines, first_in_act) for the
    act's own heading, or None. Searches after the 'Autori:' block so the
    page-header title is skipped; first_in_act is True when nothing but the
    author line(s) precede it."""
    autori = next((i for i, l in enumerate(lines[:12]) if AUTORI.match(l)), 0)
    start = autori + 1
    for i in range(start, len(lines)):
        acc = ""
        for j in range(i, min(i + 4, len(lines))):
            acc = (acc + " " + key(lines[j])).strip()
            if acc == title_key:
                first = i <= autori + 3        # "Autori:" + one or two author lines
                return (lines[i:j + 1], (lines[i - 1] if i > 0 else None),
                        (lines[j + 1] if j + 1 < len(lines) else None), lines[j + 2:j + 5], first)
            if not title_key.startswith(acc):
                break
    return None


def header_style(p_el):
    """Bold, 15 pt or larger: how the DEJURE page-header title comes out."""
    runs = [r for r in p_el.iter(W + "r") if "".join(t.text or "" for t in r.iter(W + "t")).strip()]
    if not runs:
        return False
    def bold(r):
        b = r.find(W + "rPr/" + W + "b")
        return b is not None and b.get(W + "val") not in ("0", "false")
    def size(r):
        sz = r.find(W + "rPr/" + W + "sz")
        return int(sz.get(W + "val")) / 2 if sz is not None and (sz.get(W + "val") or "").isdigit() else 0
    return all(bold(r) for r in runs) and max(size(r) for r in runs) >= 15


def centred_model(paras, anchor):
    """A paragraph whose formatting the restored heading copies: the anchor if it
    is centred, else the nearest centred short paragraph, else the anchor."""
    def centred(p):
        jc = p.find(W + "pPr/" + W + "jc")
        return jc is not None and jc.get(W + "val") in ("center", "centre")
    if anchor is not None and centred(anchor):
        return anchor
    idx = paras.index(anchor) if anchor in paras else 0
    for dist in range(1, len(paras)):
        for k in (idx - dist, idx + dist):
            if 0 <= k < len(paras) and centred(paras[k]) and len(ptext(paras[k])) < 120:
                return paras[k]
    return anchor


def make_heading(model, text):
    new = copy.deepcopy(model)
    for child in list(new):
        if child.tag != W + "pPr":
            new.remove(child)
    first_run = model.find(W + "r")
    run = copy.deepcopy(first_run) if first_run is not None else None
    if run is None:
        from lxml import etree
        run = etree.SubElement(new, W + "r")
    else:
        for child in list(run):
            if child.tag != W + "rPr":
                run.remove(child)
        new.append(run)
    from lxml import etree
    t = etree.SubElement(run, W + "t")
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    pPr = new.find(W + "pPr")
    if pPr is not None and pPr.find(W + "jc") is None:
        jc = etree.SubElement(pPr, W + "jc")
        jc.set(W + "val", "center")
    return new


def process(job):
    name, pdf_name = job
    from docx import Document
    st = {"file": name, "title_removed": 0, "heading_restored": 0, "heading_where": "",
          "heading_unplaced": 0, "check": "ok"}
    try:
        doc = Document(long_path(os.path.join(SRC, name)))
        title = os.path.splitext(name)[0]
        tkey = key(title)
        paras = body_paragraphs(doc)
        before = [ptext(p) for p in paras]
        removed_text = None
        # 1. the header title line: the title in normal case, or in the header's
        # own style (bold, 15 pt or more) when DEJURE wrote the title in capitals
        if paras and key(ptext(paras[0])) == tkey and (
                ptext(paras[0]) != ptext(paras[0]).upper() or header_style(paras[0])):
            removed_text = ptext(paras[0])
            paras[0].getparent().remove(paras[0])
            paras = paras[1:]
            st["title_removed"] = 1
        # 2. the act heading
        inserted = None
        pdf_path = pdf_name if os.path.isabs(pdf_name) else os.path.join(BASE, os.path.splitext(pdf_name)[0] + ".pdf")
        if os.path.exists(long_path(pdf_path)):
            found = find_act_heading(pdf_lines(pdf_path), tkey)
            if found:
                heading_lines, prev_line, next_line, after_lines, first_in_act = found
                heading_text = " ".join(heading_lines)
                squashed = tkey.replace(" ", "")
                present = any(key(ptext(p)) == tkey or key(ptext(p)).startswith(tkey) for p in paras)
                glued = not present and any(key(ptext(p)).replace(" ", "").startswith(squashed) for p in paras)
                if glued:
                    st["heading_where"] = "present but glued - fix by hand"
                elif not present:
                    anchor, where = None, ""
                    # the line after the heading, or the next ones when that line
                    # was removed on purpose (e.g. the "FORMULARIO ..." label)
                    for nxt in [next_line] + after_lines:
                        nk = key(nxt or "")
                        if nk:
                            anchor = next((p for p in paras if key(ptext(p)).startswith(nk[:60])), None)
                            if anchor is not None:
                                where = "before"
                                break
                    if anchor is None and prev_line and key(prev_line) and not first_in_act:
                        pk = key(prev_line)
                        anchor = next((p for p in paras if key(ptext(p)).endswith(pk[-60:])), None)
                        where = "after"
                    if anchor is None and first_in_act and paras:
                        anchor, where = paras[0], "before"     # first line of the act: top
                        next_line = ptext(paras[0])
                    if anchor is None:
                        st["heading_unplaced"] = 1
                    else:
                        new = make_heading(centred_model(paras, anchor), heading_text)
                        if where == "before":
                            anchor.addprevious(new)
                        else:
                            anchor.addnext(new)
                        inserted = heading_text
                        st["heading_restored"] = 1
                        st["heading_where"] = "%s %r" % (where, (next_line if where == "before" else prev_line)[:50])
        # 3. check: before, minus the removed title, plus the inserted heading
        after = [ptext(p) for p in body_paragraphs(doc)]
        exp = list(before)
        if removed_text is not None:
            exp.remove(removed_text)
        got = list(after)
        if inserted is not None:
            got.remove(inserted)
        if exp != got:
            st["check"] = "CHANGED"
        doc.save(long_path(os.path.join(DST, name)))
    except Exception as exc:
        st["check"] = "ERROR: %s" % str(exc)[:150]
    return st


def main():
    os.makedirs(long_path(DST), exist_ok=True)
    pdf_for = {}
    if os.path.exists(PDFMAP):
        pdf_for = {r["new_name"]: r["file"] for r in csv.DictReader(open(PDFMAP, encoding="utf-8-sig"))}
    names = sorted(n for n in os.listdir(long_path(SRC)) if n.lower().endswith(".docx"))
    positional = [a for a in ARGS if a.isdigit()]
    if positional:
        random.seed(5)
        names = random.sample(names, int(positional[0]))
    jobs = [(n, pdf_for.get(n, n)) for n in names]
    t0 = time.time()
    with mp.Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        rows = pool.map(process, jobs, chunksize=8)
    fields = ["file", "title_removed", "heading_restored", "heading_unplaced", "heading_where", "check"]
    with open(LOG, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("files: %d in %.0fs -> %s" % (len(rows), time.time() - t0, DST))
    print("header title line removed: %d" % sum(r["title_removed"] for r in rows))
    print("act heading restored:      %d" % sum(r["heading_restored"] for r in rows))
    print("act heading found in the PDF but no place to put it: %d" % sum(r["heading_unplaced"] for r in rows))
    bad = [r for r in rows if r["check"] != "ok"]
    print("check (only the title removed / heading inserted): %d ok, %d NOT ok" % (len(rows) - len(bad), len(bad)))
    for r in bad[:8]:
        print("   ", r["file"][:70], r["check"])
    for r in [r for r in rows if r["heading_unplaced"]][:8]:
        print("    unplaced:", r["file"][:80])
    for r in [r for r in rows if "glued" in r["heading_where"]]:
        print("    heading present but glued (fix by hand):", r["file"][:80])


if __name__ == "__main__":
    main()
