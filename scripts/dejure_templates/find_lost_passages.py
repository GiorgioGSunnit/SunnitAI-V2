"""For suspect converted files, show PDF passages (6+ words) that are truly absent
from the fixed DOCX - not just reordered. Read-only."""
import csv
import difflib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("VERIFY_FIXED", "1")
import verify_converted_all as V  # noqa: E402
from docx import Document  # noqa: E402

TARGETS = sys.argv[1:] or [
    "Polizza per la r.c.a", "Il Piano genitoriale", "Attestazione di conformit",
    "Richiesta di computo", "Decreto di computo",
    "Ricorso al Presidente del Tribunale per la ricusazione",
    "Richiesta di deindicizzazione", "Contratto per la prestazione dei servizi di investiment",
]

rows = list(csv.DictReader(open(V.OUT, encoding="utf-8-sig")))
ALL = "--all" in sys.argv
if ALL:   # every file missing more than N words; print only real losses
    limit = int(sys.argv[sys.argv.index("--all") + 1]) if len(sys.argv) > sys.argv.index("--all") + 1 else 20
    selected = [r for r in rows if int(r["missing_words"] or 0) > limit]
    if "--tables" in sys.argv:
        selected = [r for r in rows if int(r["tables"] or 0) > 0]
else:
    selected = [next((r for r in rows if r["file"].startswith(t)), None) for t in TARGETS]
flagged = 0
for r in selected:
    if r is None:
        continue
    doc = Document(os.path.join(V.DOCX_DIR, r["file"]))
    texts = [x.strip() for x in V.paragraphs(doc.element.body) if x.strip()]
    # the blank markers we inserted would break up the PDF's word sequence
    out = V.WORD.findall(" ".join(texts).replace("[DA COMPILARE]", " ").lower())
    src = V.pdf_words(os.path.join(V.BASE, V.PDF_STEM[r["file"]] + ".pdf"))
    title_words = V.WORD.findall(os.path.splitext(r["file"])[0].lower())
    outs = " ".join(out)
    sm = difflib.SequenceMatcher(None, src, out, autojunk=False)
    lost = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("delete", "replace") and i2 - i1 >= 6:
            chunk = src[i1:i2]
            # the DEJURE header (title + authors), removed by design
            if chunk[:5] == title_words[:5] or " ".join(title_words[-4:]) in " ".join(chunk[:len(title_words) + 2]):
                continue
            grams = [" ".join(chunk[k:k + 4]) for k in range(len(chunk) - 3)]
            present = sum(1 for g in grams if g in outs) / max(1, len(grams))
            if present < 0.5:
                lost.append((i2 - i1, " ".join(chunk)))
    lost.sort(reverse=True)
    if ALL and not lost:
        continue
    flagged += bool(lost)
    print("=" * 100)
    print("%s  (missing %s words, tables %s)" % (r["file"][:80], r["missing_words"], r["tables"]))
    for n, c in lost[:3]:
        print("   GONE %3d words: %s" % (n, c[:230]))
    if not lost:
        print("   no run of 6+ words is gone: only scattered words (titles, names, reordering)")
if ALL:
    print("\n%d of %d files checked have a passage of 6+ words truly gone" % (flagged, len(selected)))
