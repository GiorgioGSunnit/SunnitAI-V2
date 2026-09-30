"""PDF formulario -> fill-ready DOCX, keeping the PDF as it is.

pdf2docx keeps bold/size/font/alignment. On top of it: the DEJURE header and
footer are removed (label, authors, repeated title, logo, copyright line), the
"...." blanks become the [DA COMPILARE] marker the fill pipeline recognises, and
pdf2docx's layout artefacts are repaired. Everything else - notes with their [n]
numbers, alternative clauses - stays where the PDF has it.
"""
import os, re, sys, glob, time, warnings, logging, copy, collections
import unicodedata as ud
warnings.filterwarnings("ignore")
logging.disable(logging.INFO)
from pdf2docx import Converter
from docx import Document
from docx.shared import Cm, Pt
import fitz
from lxml import etree

SRC = r"C:\Users\anton\Downloads\downloads\downloads\test folder"
OUT = r"C:\Users\anton\Downloads\downloads\downloads\converted"
TMP = os.path.join(os.environ.get("TEMP", "."), "_p2d_tmp.docx")
PLACEHOLDER = "[DA COMPILARE]"

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
AUTORI = re.compile(r"^\s*Autori\s*:", re.I)
# Labels vary: "FORMULARIO DEGLI ATTI PENALI", "FORMULARIO DELLA
# RESPONSABILITA CIVILE", but also bare "FORMULARIO CONTRATTI" / "APPALTI".
# No act body text begins with the word, so the prefix alone is safe.
COLLECTION = re.compile(r"^\s*FORMULARIO\b", re.I)
FOOTER = re.compile(r"(astrea@sunnit\.ai|©\s*Copyright|Copyright\s+Giu)", re.I)
FOOTNOTE = re.compile(r"\[\d{1,3}\]")
DOTS = re.compile(r"\.{3,}")
# Every way the forms mark a blank: "....", the ellipsis character in runs
# ("…………", "…….", "L'anno…il giorno") and bracketed ("nato a […], il […]").
# A bracketed blank becomes ONE marker, never "[[DA COMPILARE]]".
BLANK = re.compile(r"\[\s*(?:…[.…]*|\.{3,})\s*\]|…[.…]*|\.{3,}")
LIG = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "ft", "ﬆ": "st"}


BRACKET_START = re.compile(r"^\s*\[\d{1,3}\]")
DOT_NUMBERED = re.compile(r"^\d{1,3}\.\s")
SIGNATURE = re.compile(r"\b(Luogo e data|Firma|Notaio|Sottoscrizione)\b", re.I)


def _is_act_boundary(text):
    """A line that belongs to the act itself, never to a footnote: a short
    signature line ("Luogo e data ....", "Notaio ....") or an all-caps heading."""
    t = text.strip()
    # headings run long: "ART. 3 – POTERE DI DISPOSIZIONE DELLA BANCA IN CASO
    # DI PEGNO “IRREGOLARE”" is 75 chars. Notes are never set in capitals.
    return (len(t) < 60 and SIGNATURE.search(t)) or (t.isupper() and len(t) < 120)


def find_note_blocks(paras):
    """Which paragraphs are footnote bodies. Must run BEFORE [n] markers are
    stripped. Returns (indices that start a note, [block member indices]).

    The first version inferred footnotes from a number prefix after markers were
    removed, and treated "1)" as a footnote style. In documents whose footnotes
    read "[1]Le disposizioni...", stripping the marker erased the only thing
    identifying them, the search latched onto the body clause "1) Di sostituire i
    liquidatori", and everything from there to the end - clauses, signature and
    notes - was deleted.

    A note starts with a paragraph that BEGINS with [n] (one that merely contains
    [n] is body text citing a note), or with "N." after the last signature line
    (never "N)", which is clause numbering). It runs until a signature line or
    heading, where the act resumes.

    A block is only removed if it is a real notes block: it runs to the end of
    the document, or it holds 2+ notes (notes printed at the foot of a page, in
    mid-document). A lone [n] paragraph mid-document is act text that opens with
    its own marker - "[2]Le parti convengono che la Banca..." in Anticipazione
    bancaria - and is kept. Leaving a stray note is the safer failure.
    """
    if not paras:
        return set(), []

    starts = {i for i, p in enumerate(paras) if BRACKET_START.match(p.text)}
    sigs = [i for i, p in enumerate(paras)
            if SIGNATURE.search(p.text) and len(p.text.strip()) < 60]
    if sigs:
        starts |= {i for i in range(sigs[-1] + 1, len(paras))
                   if DOT_NUMBERED.match(paras[i].text)}
    blocks, current = [], None
    for i in range(min(starts, default=len(paras)), len(paras)):
        if i in starts:
            if current is None:
                current = []
        elif _is_act_boundary(paras[i].text):
            if current:
                blocks.append((current, False))
            current = None
        if current is not None:
            current.append(i)
    if current:
        blocks.append((current, True))      # ran to the end of the document

    found = [m for m, trailing in blocks
             if trailing or sum(1 for i in m if i in starts) >= 2]
    if found:
        return starts, found

    # Rule C - no signature line to anchor on (a stand-alone clause, a judge's
    # order, minutes signed "il Presidente"). Accept a numbered tail "1." ...
    # "k." after the last [n] marker only if it numbers EXACTLY the markers used
    # in the text: 13 markers and notes 1-13 in Quesito accertamenti autoptici.
    # A closing list of attachments will not line up with the markers like that.
    marker_nums, last_marker = set(), -1
    for i, p in enumerate(paras):
        nums = [int(n) for n in re.findall(r"\[(\d{1,3})\]", p.text)]
        if nums:
            marker_nums.update(nums)
            last_marker = i
    tail = [i for i in range(last_marker + 1, len(paras)) if DOT_NUMBERED.match(paras[i].text)]
    numbers = [int(paras[i].text.split(".", 1)[0]) for i in tail]
    if (len(numbers) >= 2 and numbers == list(range(1, len(numbers) + 1))
            and set(numbers) == marker_nums):
        return set(tail), [list(range(tail[0], len(paras)))]
    return set(), []


# --- notes stay as plain text, where the PDF has them -----------------------
# DEJURE's notes (case law, drafting instructions) are kept with their [n]
# numbers exactly as in the PDF - usually at the end of the document. They are
# recognised only so that the "...." inside them, which are ellipses in
# quotations ("dei fatti [...] costituenti"), do not become [DA COMPILARE].

XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"
SENTENCE_END = re.compile(r"[.:;!?»”\")]\s*$")


def _merge_wrapped_lines(elements):
    """Rejoin a note's lines where the PDF merely wrapped them. Splitting on
    line breaks made "...per le deliberazioni dell'assemblea" and
    "straordinaria necessitano..." two paragraphs of one note. A line that does
    not end a sentence continues on the next; one that does keeps its break."""
    out = []
    for el in elements:
        if out:
            prev = out[-1]
            prev_text = "".join(t.text or "" for t in prev.iter(W + "t"))
            if prev_text.strip() and not SENTENCE_END.search(prev_text):
                text = "".join(t.text or "" for t in el.iter(W + "t"))
                if not prev_text.endswith(" ") and not text.startswith(" "):
                    t = etree.SubElement(etree.SubElement(prev, W + "r"), W + "t")
                    t.text = " "
                    t.set(XML_SPACE, "preserve")
                for run in el.findall(W + "r"):
                    prev.append(run)
                if el.getparent() is not None:
                    el.getparent().remove(el)
                continue
        out.append(el)
    return out


def mark_notes(doc, stats):
    """Find the note paragraphs, rejoin their wrapped lines in place, and return
    the note paragraph elements. Nothing is moved or removed."""
    paras = [p for p in doc.paragraphs if p.text.strip()]
    starts, blocks = find_note_blocks(paras)
    kept = []
    for members in blocks:
        notes = []
        for i in members:
            if i in starts or not notes:
                notes.append([])
            notes[-1].append(paras[i]._element)
        for note in notes:
            kept.extend(_merge_wrapped_lines(note))
        stats["notes"] += len(notes)
    return kept


def _run_kind(run):
    """'tab' for a run holding only tab(s), 'text' otherwise, None if empty."""
    has_text = any((t.text or "").strip() for t in run.iter(W + "t"))
    if has_text or run.find(W + "footnoteReference") is not None:
        return "text"
    return "tab" if run.findall(W + "tab") else None


def fix_tab_layout(doc, stats):
    """Replace pdf2docx's tab-faked layout with real alignment.

    pdf2docx positions text with TAB runs against tab stops rather than with
    alignment. Two shapes cause visible damage once Word re-wraps the text:
      - "...arbitrale.<TAB><TAB><TAB>Luogo e data": a right-aligned signature line
        glued onto the end of the previous sentence
      - "<TAB>ATTO DI DIFFIDA": a centred heading faked with leading tabs
    """
    for par in list(doc.paragraphs):
        runs = [c for c in par._element if c.tag.endswith("}r")]
        kinds = [_run_kind(r) for r in runs]

        # trailing short segment after 2+ tabs -> its own right-aligned paragraph
        j = len(kinds)
        while j > 0 and kinds[j - 1] == "text":
            j -= 1
        tabs_end = j
        while j > 0 and kinds[j - 1] == "tab":
            j -= 1
        n_tabs = tabs_end - j
        tail = runs[tabs_end:]
        tail_text = "".join((t.text or "") for r in tail for t in r.iter(W + "t")).strip()
        head_text = "".join((t.text or "") for r in runs[:j] for t in r.iter(W + "t")).strip()
        if n_tabs >= 2 and tail and head_text and 0 < len(tail_text) < 60:
            new_p = copy.deepcopy(par._element)
            for c in list(new_p):
                if c.tag.endswith("}r"):
                    new_p.remove(c)
            for r in runs[j:tabs_end]:
                par._element.remove(r)
            for r in tail:
                par._element.remove(r)
                new_p.append(r)
            ppr = new_p.find(W + "pPr")
            if ppr is None:
                ppr = new_p.makeelement(W + "pPr", {})
                new_p.insert(0, ppr)
            for tag in ("jc", "tabs", "ind"):
                for e in ppr.findall(W + tag):
                    ppr.remove(e)
            for sp in ppr.findall(W + "spacing"):
                sp.set(W + "before", "0")
            # An all-caps tail is a heading ("DIFFIDA") that belongs centred;
            # mixed case is a signature line ("Luogo e data", "Firma") that
            # belongs right-aligned.
            align = "center" if tail_text.isupper() else "right"
            ppr.append(ppr.makeelement(W + "jc", {W + "val": align}))
            par._element.addnext(new_p)
            stats["tab_splits"] += 1
            continue

        # leading tabs + short all-caps heading -> genuinely centred
        lead = 0
        while lead < len(kinds) and kinds[lead] == "tab":
            lead += 1
        rest = "".join((t.text or "") for r in runs[lead:] for t in r.iter(W + "t")).strip()
        if lead >= 1 and rest and len(rest) < 70:
            for r in runs[:lead]:
                par._element.remove(r)
            ppr = par._element.find(W + "pPr")
            if ppr is None:
                ppr = par._element.makeelement(W + "pPr", {})
                par._element.insert(0, ppr)
            for tag in ("jc", "tabs", "ind"):
                for e in ppr.findall(W + tag):
                    ppr.remove(e)
            # all-caps = heading (centre); mixed case = signature line (right)
            ppr.append(ppr.makeelement(W + "jc", {W + "val": "center" if rest.isupper() else "right"}))
            stats["centred"] += 1


def _line_key(text):
    """Comparable form of a line in either format: the PDF has "....", the DOCX
    has [DA COMPILARE], and note markers may or may not still be present."""
    text = FOOTNOTE.sub(" ", DOTS.sub(" ", text.replace(PLACEHOLDER, " ")))
    return " ".join(re.sub(r"[^\w]+", " ", text.lower()).split())


def centre_from_pdf(doc, pdf_path, stats):
    """Restore centring pdf2docx faked with indents or lost outright.

    Headings like PREMESSO, CHIEDE or "Art. 16 (Clausola risolutiva espressa)"
    come out left-aligned with an 8cm indent, or with no indent at all. Neither
    text nor case says which lines are headings ("premesso" is centred in one
    form, "chiede" is left-aligned in another), so ask the PDF: a line whose
    midpoint is the page's midpoint and which stops well short of both margins
    is centred. DOCX paragraphs are matched to PDF lines in document order, so a
    left-aligned "chiede" consumes its own line before a later centred one.
    """
    lines = []
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            width = page.rect.width
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    key = _line_key("".join(s["text"] for s in line["spans"]))
                    if not key:
                        continue
                    x0, _, x1, _ = line["bbox"]
                    centred = (abs((x0 + x1) / 2 - width / 2) < 20
                               and x0 > 90 and x1 < width - 90)
                    lines.append((key, centred))

    cursor = 0
    for par in doc.paragraphs:
        key = _line_key(par.text)
        if not key or len(par.text.strip()) > 100:
            continue
        match = next((i for i in range(cursor, len(lines)) if lines[i][0] == key), None)
        if match is None:
            continue
        cursor = match + 1
        if not lines[match][1] or par.alignment in (1, 2):
            continue
        ppr = par._element.find(W + "pPr")
        if ppr is None:
            ppr = par._element.makeelement(W + "pPr", {})
            par._element.insert(0, ppr)
        for tag in ("jc", "ind", "tabs"):
            for e in ppr.findall(W + tag):
                ppr.remove(e)
        ppr.append(ppr.makeelement(W + "jc", {W + "val": "center"}))
        stats["centred_pdf"] += 1


def _pdf_lines(pdf_path):
    """Every text line of the PDF in reading order, with its horizontal place:
    (key, x0, x1, left edge of text on that page, right edge, page width)."""
    out = []
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            page_lines = []
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    key = _line_key("".join(s["text"] for s in line["spans"]))
                    if key:
                        page_lines.append((key, line["bbox"][0], line["bbox"][2]))
            if not page_lines:
                continue
            lo = min(x0 for _, x0, _ in page_lines)
            hi = max(x1 for _, _, x1 in page_lines)
            out += [(k, x0, x1, lo, hi, page.rect.width) for k, x0, x1 in page_lines]
    return out


def split_merged_lines(doc, pdf_path, stats):
    """Split a short paragraph pdf2docx assembled from separate PDF lines.

    "-attore-" (right-aligned) and "CONTRO" (centred on the line below) come out
    as one paragraph of two runs with no break between them, and render side by
    side as "-attore-CONTRO". A wrapped paragraph ALSO has one run per PDF line,
    so a split requires every line but the last to be a short line set away
    from the left margin - true of positioned headings, never of wrapped prose,
    whose lines span the text width. Each piece takes its alignment from where
    the PDF puts it.
    """
    lines = _pdf_lines(pdf_path)
    cursor = 0

    def run_text(r):
        return "".join(t.text or "" for t in r.iter(W + "t"))

    for par in list(doc.paragraphs):
        full = _line_key(par.text)
        if not full or len(par.text.strip()) > 120:
            continue
        same = next((i for i in range(cursor, len(lines)) if lines[i][0] == full), None)
        if same is not None:                       # already exactly one PDF line
            cursor = same + 1
            continue
        runs = [c for c in par._element if c.tag.endswith("}r")]
        found = None
        for j in range(cursor, len(lines)):
            if not full.startswith(lines[j][0]):
                continue
            groups, start, li, acc = [], 0, j, ""
            for ri, r in enumerate(runs):
                acc += run_text(r)
                k = _line_key(acc)
                if k and li < len(lines) and k == lines[li][0]:
                    groups.append((start, ri + 1, li))
                    start, li, acc = ri + 1, li + 1, ""
                elif k and li < len(lines) and not lines[li][0].startswith(k):
                    break
            if len(groups) < 2 or _line_key("".join(run_text(r) for r in runs[start:])):
                continue
            positioned = all(
                lines[g[2]][1] - lines[g[2]][3] > 36 and
                (lines[g[2]][2] - lines[g[2]][1]) < 0.6 * (lines[g[2]][4] - lines[g[2]][3])
                for g in groups[:-1])
            if positioned:
                found = groups
            break
        if not found:
            continue

        # trailing empty/tab runs stay with the last piece
        found[-1] = (found[-1][0], len(runs), found[-1][2])
        anchor = par._element
        for n, (a, b, li) in enumerate(found):
            if n == 0:
                el = par._element
                for r in runs[b:]:
                    el.remove(r)
            else:
                el = copy.deepcopy(par._element)
                for c in list(el):
                    if c.tag.endswith("}r"):
                        el.remove(c)
                for r in runs[a:b]:
                    el.append(r)
                anchor.addnext(el)
                anchor = el
            # leading tabs faked the position; alignment replaces them
            for r in [c for c in el if c.tag.endswith("}r")]:
                if run_text(r).strip():
                    break
                el.remove(r)
            ppr = el.find(W + "pPr")
            if ppr is None:
                ppr = el.makeelement(W + "pPr", {})
                el.insert(0, ppr)
            for tag in ("jc", "ind", "tabs"):
                for e in ppr.findall(W + tag):
                    ppr.remove(e)
            if n:
                for sp in ppr.findall(W + "spacing"):
                    sp.set(W + "before", "0")
            _, x0, x1, lo, hi, width = lines[li]
            if abs((x0 + x1) / 2 - width / 2) < 20 and x0 - lo > 36:
                align = "center"
            elif hi - x1 < 20 and x0 - lo > 36:
                align = "right"
            else:
                align = "left"
            ppr.append(ppr.makeelement(W + "jc", {W + "val": align}))
        cursor = found[-1][2] + 1
        stats["merged_lines"] += 1


def release_narrow_boxes(doc, stats):
    """Widen paragraphs pdf2docx boxed to the width of their PDF text.

    pdf2docx sets indents so each paragraph is exactly as wide as its line was:
    "2. [DA COMPILARE];" gets a 15.7cm right indent and a 1.3cm column. It looks
    right until the placeholder is filled in, then the value wraps one word per
    line. Where a paragraph has less than 7cm to live in, keep where it sits on
    the page but express it with alignment instead of a box:
      - centred in a symmetric box        -> centred, no indents
      - pushed to the right ("Con Osservanza", address lines) -> right-aligned
      - boxed on the left                 -> left, right indent dropped
      - a positioned fragment glued to a normal line by a big first-line indent
        ("00 .... ROMA" + "Oggetto: istanza") -> split into two paragraphs
    """
    sect = doc.sections[-1]
    column = int((sect.page_width - sect.left_margin - sect.right_margin) / 635)
    roomy = 3969                                   # 7 cm in twips

    def set_jc(ppr, val):
        for e in ppr.findall(W + "jc"):
            ppr.remove(e)
        ppr.append(ppr.makeelement(W + "jc", {W + "val": val}))

    def zero(ind):
        for a in ("left", "right", "firstLine", "hanging"):
            if ind.get(W + a) is not None:
                ind.set(W + a, "0")

    for par in list(doc.paragraphs):
        if not par.text.strip():
            continue
        ppr = par._element.find(W + "pPr")
        ind = ppr.find(W + "ind") if ppr is not None else None
        if ind is None:
            continue
        left, right, first = (int(ind.get(W + a) or 0) for a in ("left", "right", "firstLine"))

        if par.alignment == 1 and left and abs(left - right) < 300:
            zero(ind)
            stats["boxes"] += 1
            continue
        if column - left - right >= roomy and column - left - right - first >= roomy:
            continue

        runs = [c for c in par._element if c.tag.endswith("}r")]
        text_idx = [i for i, r in enumerate(runs)
                    if "".join(t.text or "" for t in r.iter(W + "t")).strip()]
        if first >= roomy and len(text_idx) >= 2:
            tail = copy.deepcopy(par._element)
            for c in list(tail):
                if c.tag.endswith("}r"):
                    tail.remove(c)
            for r in runs[text_idx[1]:]:
                par._element.remove(r)
                tail.append(r)
            tail_ppr = tail.find(W + "pPr")
            zero(tail_ppr.find(W + "ind"))
            set_jc(tail_ppr, "left")
            for sp in tail_ppr.findall(W + "spacing"):
                sp.set(W + "before", "0")
            par._element.addnext(tail)

        if left > right or left + first > column / 2:
            set_jc(ppr, "right")
        elif par.alignment != 1 and first >= roomy:
            set_jc(ppr, "center")
        zero(ind)
        stats["boxes"] += 1


SPACE_BEFORE_PUNCT = re.compile(r"(?<=\S) +([.,;:])(?=\s|$)")
LEADING_PUNCT = re.compile(r"^ *[.,;:](?=\s|$)")


def tidy_punctuation(paragraphs, stats):
    """Drop the space before . , ; : - "[DA COMPILARE] ." in the source forms,
    "aziendale  ." where a note marker was removed. Filled in, both read
    "Mario Rossi ." The space and the mark can sit in different runs."""
    for par in paragraphs:
        runs = [r for r in par.runs if r.text]
        for i, run in enumerate(runs):
            txt = SPACE_BEFORE_PUNCT.sub(r"\1", run.text)
            if i and LEADING_PUNCT.match(txt):
                prev = runs[i - 1]
                if txt.startswith(" ") or prev.text.endswith(" "):
                    if prev.text.rstrip(" ") and not prev.text.rstrip(" ").endswith("\t"):
                        txt = txt.lstrip(" ")
                        prev.text = prev.text.rstrip(" ")
            if txt != run.text:
                run.text = txt
                stats["punct"] += 1


def normalise_spacing(doc, stats):
    """Cap vertical gaps inherited from absolute PDF positioning.

    pdf2docx reproduces each line's distance from the one above it on the source
    page. Once the logo and header block are removed, the first paragraphs keep
    offsets measured against content that no longer exists - 66pt above a line
    that should sit a normal paragraph gap below its neighbour.
    """
    befores = collections.Counter()
    for par in doc.paragraphs:
        sb = par.paragraph_format.space_before
        if par.text.strip() and sb is not None:
            befores[round(sb.pt, 1)] += 1
    typical = befores.most_common(1)[0][0] if befores else 6.0
    for par in doc.paragraphs:
        sb = par.paragraph_format.space_before
        if sb is not None and sb.pt > max(18.0, typical * 2.5):
            par.paragraph_format.space_before = Pt(typical)
            stats["spacing"] += 1


def flatten_sections(doc, stats):
    """Collapse the one-section-per-PDF-page layout into a single flow.

    pdf2docx recreates every source page as its own NEW_PAGE section. A section
    break forces a page break, so once cleaning shortens the text the remainder
    of the page stays blank - which is the large empty gap seen mid-document.
    Page boundaries from the source PDF carry no meaning once this is an
    editable template, so only the final section properties are kept.
    """
    for par in list(doc.paragraphs):
        for sect in par._element.findall(".//" + W + "sectPr"):
            sect.getparent().remove(sect)
            stats["sections"] += 1
        # the paragraph existed only to carry the break
        if not par.text.strip() and not par._element.findall(".//" + W + "drawing"):
            drop(par)

    # collapse runs of blank paragraphs left behind by removals
    blank = 0
    for par in list(doc.paragraphs):
        if par.text.strip():
            blank = 0
            continue
        blank += 1
        if blank > 1:
            drop(par)

    # pdf2docx mirrors PDF geometry with ~1 cm margins, which is far too tight
    # for a document a lawyer will edit and print. Its indents are measured from
    # those margins, so widening the margins alone shifts every indented line
    # further in ("Con Osservanza" ended up 15cm across a 17cm text column).
    # Take the margin increase back off the indents to keep text where it was.
    last = doc.sections[-1]
    d_left = Cm(2) - (last.left_margin or 0)
    d_right = Cm(2) - (last.right_margin or 0)
    to_twips = lambda emu: int(emu / 635)
    for par in doc.paragraphs:
        ind = par._element.find(W + "pPr/" + W + "ind")
        if ind is None:
            continue
        left = int(ind.get(W + "left") or 0)
        right = int(ind.get(W + "right") or 0)
        first = int(ind.get(W + "firstLine") or 0)
        shift = to_twips(d_left)
        if shift > 0:
            new_left = max(0, left - shift)
            # what the left indent could not absorb comes off the first line
            first = max(0, first - (shift - (left - new_left)))
            ind.set(W + "left", str(new_left))
            if ind.get(W + "firstLine") is not None:
                ind.set(W + "firstLine", str(first))
        if to_twips(d_right) > 0:
            ind.set(W + "right", str(max(0, right - to_twips(d_right))))
        stats["indents"] += 1

    for sect in doc.sections:
        sect.top_margin = sect.bottom_margin = Cm(2)
        sect.left_margin = sect.right_margin = Cm(2)


def repair_name(name):
    """Undo UTF-8-read-as-codepage mojibake in the source filenames.

    The exports arrived with names like "responsabilita" + box-drawing junk:
    UTF-8 bytes decoded as cp437, so a-grave became "a" + chr(0x2560) + chr(0xC7)
    and the curly apostrophe became three Greek/Latin letters. Re-encoding
    recovers the original; NFC then keeps accents as single code points so the
    names compare equal to anything typed by hand.
    """
    for enc in ("cp437", "cp850", "cp1252"):
        try:
            return ud.normalize("NFC", name.encode(enc).decode("utf-8"))
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return ud.normalize("NFC", name)


def body_size(doc):
    """Most common run size in the document - i.e. the body text size."""
    sizes = collections.Counter()
    for par in doc.paragraphs:
        for r in par.runs:
            if r.text.strip() and r.font.size is not None:
                sizes[r.font.size.pt] += len(r.text)
    return sizes.most_common(1)[0][0] if sizes else None


def _is_wrap(before, after, pdf_lines):
    """Is this line break just the PDF wrapping a sentence onto the next line?

    "...in cui versa il Sig. ... giusta" / "documentazione già prodotta" must
    read as one paragraph; "PREMESSO" / "che in data ..." must stay two. Across
    the 103 pilot files, 213 breaks had a full-width line before them (>= 75% of
    the text width, starting at the left margin), no sentence end and a
    lowercase word after - every one a wrap. Headings are short lines, so they
    never qualify. Anything unclear stays split, as before.
    """
    if not pdf_lines or not before.strip() or not after.strip():
        return False
    if SENTENCE_END.search(before) or before.strip().isupper():
        return False
    if not re.match(r"\s*[a-zàáèéìíòóùú]", after):
        return False
    key = _line_key(before)
    if len(key) < 15:
        return False
    line = next((l for l in pdf_lines
                 if l[0] == key or (len(l[0]) > 15 and key.endswith(l[0]))), None)
    if line is None:
        return False
    _, x0, x1, lo, hi, _ = line
    return x1 - x0 >= 0.75 * (hi - lo) and x0 - lo <= 30


def _join_wrapped_breaks(par_el, pdf_lines, stats):
    """Replace the line breaks that are only PDF wraps with a space."""
    seq = []
    for run in par_el:
        if not run.tag.endswith("}r"):
            continue
        for ch in run:
            if ch.tag == W + "t":
                seq.append(ch.text or "")
            elif ch.tag == W + "tab":
                seq.append(" ")
            elif ch.tag == W + "br":
                seq.append(ch)
    breaks = [i for i, s in enumerate(seq) if not isinstance(s, str)]
    for n, i in enumerate(breaks):
        start = breaks[n - 1] + 1 if n else 0
        end = breaks[n + 1] if n + 1 < len(breaks) else len(seq)
        before, after = "".join(seq[start:i]), "".join(seq[i + 1:end])
        if not _is_wrap(before, after, pdf_lines):
            continue
        br = seq[i]
        if not before.endswith(" ") and not after.startswith(" "):
            t = br.makeelement(W + "t", {})
            t.text = " "
            t.set(XML_SPACE, "preserve")
            br.addprevious(t)
        br.getparent().remove(br)
        stats["wraps_joined"] += 1


def split_soft_breaks(doc, stats, pdf_lines=None):
    """Turn <w:br/> inside a paragraph into real paragraph boundaries.

    pdf2docx merges a heading and the sentence after it into one paragraph with
    a soft break, so the fill pipeline sees a heading and a fillable line as a
    single element. Splitting keeps formatting: each half keeps its own runs.
    Breaks that only wrap a sentence are joined instead (see _is_wrap).
    """
    for par in list(doc.paragraphs):
        if pdf_lines:
            _join_wrapped_breaks(par._element, pdf_lines, stats)
        brs = par._element.findall(".//" + W + "br")
        if not brs:
            continue
        groups, current = [], []
        for child in list(par._element):
            if not child.tag.endswith("}r"):
                continue
            if child.findall(W + "br"):
                # a run carrying a break closes the current paragraph
                for br in child.findall(W + "br"):
                    child.remove(br)
                if "".join(t.text or "" for t in child.iter(W + "t")).strip():
                    current.append(child)
                groups.append(current)
                current = []
            else:
                current.append(child)
        if current:
            groups.append(current)
        groups = [g for g in groups if any(
            "".join(t.text or "" for t in r.iter(W + "t")).strip() for r in g)]
        if len(groups) < 2:
            continue
        anchor = par._element
        for g in groups[1:]:
            new_p = copy.deepcopy(par._element)
            for child in list(new_p):
                if child.tag.endswith("}r"):
                    new_p.remove(child)
            for run in g:
                par._element.remove(run)
                new_p.append(run)
            # deepcopy carried the parent's space_before onto every fragment,
            # so a 66pt gap above the original paragraph was repeated above
            # each piece. Only the first piece should keep it.
            ppr = new_p.find(W + "pPr")
            if ppr is not None:
                for sp in ppr.findall(W + "spacing"):
                    sp.set(W + "before", "0")
                # Same for the first-line indent: pdf2docx centres a heading by
                # indenting its first line ~8cm, and every body line split off
                # from it started 8cm across the page.
                for ind in ppr.findall(W + "ind"):
                    for attr in ("firstLine", "hanging"):
                        if ind.get(W + attr) is not None:
                            ind.set(W + attr, "0")
            anchor.addnext(new_p)
            anchor = new_p
            stats["split"] += 1


NUMBERED_LINE = re.compile(r"^\s*\d{1,3}\.\s")


def split_numbered_lines(doc, pdf_path, stats):
    """Put a numbered line glued onto the previous paragraph back on its own.

    pdf2docx sometimes joins "...disponendo che vi proceda il CTU da solo." and
    the note "1. Formula esemplificativa..." - or list items like "(doc. 2);"
    and "3. in data ..." - into one paragraph with no break at all. Split only
    where the previous text ends a sentence and the PDF really starts a line
    with exactly that text (6 cases in the 103 pilot files, all confirmed).
    """
    line_starts = set()
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    text = " ".join("".join(s["text"] for s in line["spans"]).split())
                    if text:
                        line_starts.add(text)

    for par in list(doc.paragraphs):
        el = par._element
        while True:
            runs = [c for c in el if c.tag.endswith("}r")]
            before, cut = "", None
            for k, run in enumerate(runs):
                text = "".join(t.text or "" for t in run.iter(W + "t"))
                if k and NUMBERED_LINE.match(text) and re.search(r"[.;:]\s*$", before):
                    head = " ".join(text.split())[:20]
                    if len(head) >= 4 and any(s.startswith(head) for s in line_starts):
                        cut = k
                        break
                before += text
            if cut is None:
                break
            new_p = copy.deepcopy(el)
            for c in list(new_p):
                if c.tag.endswith("}r"):
                    new_p.remove(c)
            for run in runs[cut:]:
                el.remove(run)
                new_p.append(run)
            ppr = new_p.find(W + "pPr")
            if ppr is not None:
                for sp in ppr.findall(W + "spacing"):
                    sp.set(W + "before", "0")
                for ind in ppr.findall(W + "ind"):
                    for attr in ("firstLine", "hanging"):
                        if ind.get(W + attr) is not None:
                            ind.set(W + attr, "0")
            el.addnext(new_p)
            stats["numbered_split"] += 1
            el = new_p


def drop(par):
    par._element.getparent().remove(par._element)


def clean(path, stem, stats, pdf_path=None):
    doc = Document(path)

    # --- the DEJURE logo: an image, invisible to any text-based cleaning ---
    for drawing in list(doc.element.body.iter(W + "drawing")):
        run = drawing.getparent()
        while run is not None and not run.tag.endswith("}r"):
            run = run.getparent()
        if run is None:
            continue
        par = run.getparent()
        run.getparent().remove(run)
        stats["images"] += 1
        # the logo sits alone in its paragraph; drop the empty shell it leaves
        if par is not None and not "".join(
            e.text or "" for e in par.iter(W + "t")
        ).strip():
            if par.getparent() is not None:
                par.getparent().remove(par)

    # --- header/footer boxes rebuilt as tables, often NESTED ---------------
    # doc.tables only exposes top-level tables and cell.text does not recurse,
    # so the copyright box (body/tbl/tc/tbl/tc/p) was invisible to it. Walk the
    # XML instead, innermost first, so removing a child never takes a parent
    # that still holds real content with it.
    def table_text(tbl):
        return " ".join((t.text or "") for t in tbl.iter(W + "t")).strip()

    tbls = list(doc.element.body.iter(W + "tbl"))
    tbls.sort(key=lambda e: len(list(e.iterancestors())), reverse=True)
    for tbl in tbls:
        if tbl.getparent() is None:
            continue
        txt = table_text(tbl)
        if not txt:
            continue
        # length guard: never drop a large table that merely mentions the label
        if len(txt) < 250 and (COLLECTION.search(txt) or FOOTER.search(txt)):
            tbl.getparent().remove(tbl)
            stats["header"] += 1

    # tables left empty once their contents were removed
    for tbl in list(doc.element.body.iter(W + "tbl")):
        if tbl.getparent() is not None and not table_text(tbl):
            tbl.getparent().remove(tbl)

    # --- header/footer text, removed wherever it appears -------------------
    # Not gated to the top of the document: "Autori:" survived in 22 of 103
    # files because a body line preceded it and closed the window early.
    first_content = True
    for p in list(doc.paragraphs):
        t = p.text.strip()
        if not t:
            continue
        if FOOTER.search(t):
            drop(p); stats["footer"] += 1; continue
        if AUTORI.match(t) or COLLECTION.match(t):
            drop(p); stats["header"] += 1; continue
        # the repeated title line is position-dependent - it could legitimately
        # appear later in the body - so only strip it at the very start
        if first_content and t.lower().rstrip(".") == stem.lower().rstrip("."):
            drop(p); stats["header"] += 1; continue
        first_content = False

    split_soft_breaks(doc, stats, _pdf_lines(pdf_path) if pdf_path else None)
    # a note glued to the signature line ("(Firma Avv. ....) [1]Indicare...")
    # only starts with [n] once split off
    if pdf_path:
        split_merged_lines(doc, pdf_path, stats)
        split_numbered_lines(doc, pdf_path, stats)

    # --- notes: left in place; recognised only to protect their "...." ------
    # (the list keeps the elements alive, so their ids stay valid below)
    note_elements = mark_notes(doc, stats)
    note_ids = {id(el) for el in note_elements}

    def in_note(p):
        return id(p._element) in note_ids

    # --- run-level text cleanup (keeps bold / size / font) -----------------
    def scrub(runs, blanks=True):
        for r in runs:
            txt = r.text
            if not txt:
                continue
            for k, v in LIG.items():
                txt = txt.replace(k, v)
            txt = txt.replace("\xa0", " ").replace("\u2002", " ").replace("\u2003", " ")
            d = len(BLANK.findall(txt)) if blanks else 0
            if d:
                stats["placeholders"] += d
                txt = BLANK.sub(PLACEHOLDER, txt)
                # A space before punctuation in the source forms (".... , nato a")
                # reads badly once filled in: "Mario Rossi , nato a".
                txt = re.sub(r" +([,;:])", r"\1", txt)
            if txt != r.text:
                r.text = txt

    for p in doc.paragraphs:
        scrub(p.runs, blanks=not in_note(p))
    for t in doc.tables:
        for row in t.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    scrub(p.runs)

    # placeholders split across runs are invisible to the per-run pass
    stats["cross_run_missed"] += sum(len(BLANK.findall(p.text)) for p in doc.paragraphs
                                     if not in_note(p))

    tidy_punctuation((p for p in doc.paragraphs if not in_note(p)), stats)
    tidy_punctuation((p for t in doc.tables for row in t.rows
                      for cell in row.cells for p in cell.paragraphs), stats)

    fix_tab_layout(doc, stats)
    if pdf_path:
        centre_from_pdf(doc, pdf_path, stats)
    flatten_sections(doc, stats)
    release_narrow_boxes(doc, stats)
    normalise_spacing(doc, stats)

    os.makedirs(OUT, exist_ok=True)
    dest = os.path.join(OUT, stem + ".docx")
    doc.save(dest)
    els = sum(1 for p in doc.paragraphs if p.text.strip())
    els += sum(1 for t in doc.tables for r in t.rows for c in r.cells
               for p in c.paragraphs if p.text.strip())
    return els



def run_batch():
    files = sorted(glob.glob(os.path.join(SRC, "*.pdf")))
    stats = {"footer": 0, "header": 0, "images": 0, "notes": 0, "split": 0, "sections": 0, "tab_splits": 0, "centred": 0, "spacing": 0, "placeholders": 0, "cross_run_missed": 0, "centred_pdf": 0, "indents": 0, "punct": 0, "boxes": 0, "merged_lines": 0, "wraps_joined": 0, "numbered_split": 0}
    elements, failed, t0 = [], [], time.time()

    for i, f in enumerate(files, 1):
        stem = repair_name(os.path.splitext(os.path.basename(f))[0])
        try:
            c = Converter(f); c.convert(TMP); c.close()
            elements.append(clean(TMP, stem, stats, f))
        except Exception as e:
            failed.append((stem, str(e)[:70]))
        if i % 20 == 0:
            print("  ...%d/%d" % (i, len(files)), file=sys.stderr)

    el = elements or [0]
    print("converted %d/%d files in %.0fs -> %s" % (len(elements), len(files), time.time() - t0, OUT))
    print()
    print("elements per file: min %d  max %d  avg %d   (generation cap 600; over: %d)"
          % (min(el), max(el), sum(el) // len(el), sum(1 for e in el if e > 600)))
    print()
    print("cleaned:")
    print("  header lines/tables     :", stats["header"])
    print("  logo images removed     :", stats["images"])
    print("  notes kept in place     :", stats["notes"])
    print("  wrapped lines rejoined  :", stats["wraps_joined"])
    print("  paragraphs split on <br>:", stats["split"])
    print("  numbered lines split off:", stats["numbered_split"])
    print("  section breaks removed  :", stats["sections"])
    print("  signature lines split   :", stats["tab_splits"])
    print("  headings centred        :", stats["centred"])
    print("  oversized gaps capped   :", stats["spacing"])
    print("  centred from PDF layout :", stats["centred_pdf"])
    print("  indents re-based       :", stats["indents"])
    print("  space before punct     :", stats["punct"])
    print("  narrow boxes widened   :", stats["boxes"])
    print("  merged lines split     :", stats["merged_lines"])
    print("  footer lines removed   :", stats["footer"], "(pdf2docx drops most itself)")
    print("  placeholders converted :", stats["placeholders"])
    print("  dots left (split runs) :", stats["cross_run_missed"])
    if failed:
        print()
        print("FAILED (%d):" % len(failed))
        for n, e in failed[:10]:
            print("  ", n[:60], "|", e)


if __name__ == "__main__":
    run_batch()
