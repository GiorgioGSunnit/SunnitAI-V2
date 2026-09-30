"""Stage 3: targeted repairs the general stages cannot do. Runs on
converted_all_final in place, after fix_titles_headings.py (re-run it after any
re-run of stage 2).

1. Il Piano genitoriale: rebuild the parents' signature section as in the PDF
   (pages 3-4): FIRME DEI GENITORI, then per parent a 6x2 contact table (blank
   row above each label), Data, Firma. The converter kept only "Data",
   "FIRMA DELLA MADRE", "FIRMA DEL PADRE".
2. Il Piano genitoriale: weekday accents lost in the weekly table (Lunedi -> Lunedì).
3. Polizza per la r.c.a.: the act heading "POLIZZA PER LA R.C.A." before "TRA".
"""
import copy
import os

from docx import Document
from lxml import etree

FINAL = r"C:\Users\anton\Downloads\downloads\downloads\converted_all_final"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"
PH = "[DA COMPILARE]"


def ptext(el):
    return "".join(t.text or "" for t in el.iter(W + "t"))


def para_like(model, text):
    """A paragraph with model's paragraph and first-run formatting, holding text."""
    new = copy.deepcopy(model)
    for child in list(new):
        if child.tag != W + "pPr":
            new.remove(child)
    run = copy.deepcopy(model.find(W + "r"))
    for child in list(run):
        if child.tag != W + "rPr":
            run.remove(child)
    t = etree.SubElement(run, W + "t")
    t.text = text
    t.set(XML_SPACE, "preserve")
    new.append(run)
    return new


def left_aligned(model):
    p = copy.deepcopy(model)
    jc = p.find(W + "pPr/" + W + "jc")
    if jc is not None:
        jc.getparent().remove(jc)
    return p


def contact_table(doc, model_tbl, last_label):
    rows = [(PH, PH), ("NOME E COGNOME", "TELEFONO DI CASA"), (PH, PH),
            ("INDIRIZZO", "TELEFONO DEL LAVORO"), (PH, PH), ("CITTÀ, CAP", last_label)]
    t = doc.add_table(rows=len(rows), cols=2)
    tbl = t._tbl
    old_pr = tbl.find(W + "tblPr")
    tbl.replace(old_pr, copy.deepcopy(model_tbl.find(W + "tblPr")))       # borders, width
    cell_run = model_tbl.find(".//" + W + "tc//" + W + "r")
    for (a, b), row in zip(rows, t.rows):
        for text, cell in zip((a, b), row.cells):
            p = cell.paragraphs[0]._p
            run = copy.deepcopy(cell_run)
            for child in list(run):
                if child.tag != W + "rPr":
                    run.remove(child)
            tt = etree.SubElement(run, W + "t")
            tt.text = text
            tt.set(XML_SPACE, "preserve")
            p.append(run)
    tbl.getparent().remove(tbl)            # add_table appended it at the end
    return tbl


def piano_genitoriale():
    path = os.path.join(FINAL, "Il Piano genitoriale.docx")
    doc = Document(path)
    body = doc.element.body
    paras = [p for p in body.iter(W + "p") if ptext(p).strip()]
    if any(ptext(p).strip() == "FIRME DEI GENITORI" for p in paras):
        print("Il Piano genitoriale: signature section already rebuilt")
        return
    data = next(p for p in paras if ptext(p).strip().startswith("Data:"))
    madre = next(p for p in paras if ptext(p).strip() == "FIRMA DELLA MADRE")
    padre = next(p for p in paras if ptext(p).strip() == "FIRMA DEL PADRE")
    model_tbl = next(t for t in body.iter(W + "tbl") if "Località" in ptext(t))
    heading_model = next(p for p in paras if ptext(p).strip() == "7. CALENDARIZZAZIONE")
    line_model = left_aligned(data)

    block = [para_like(heading_model, "FIRME DEI GENITORI"),
             para_like(line_model, "Madre"),
             contact_table(doc, model_tbl, "CELL."),
             para_like(line_model, "Data: " + PH),
             para_like(line_model, "Firma della madre " + PH),
             para_like(line_model, "Padre"),
             contact_table(doc, model_tbl, "ALTRI NUMERI DI TELEFONO"),
             para_like(line_model, "Data: " + PH),
             para_like(line_model, "Firma del padre " + PH)]
    anchor = data
    for el in block:
        anchor.addnext(el)
        anchor = el
    for p in (data, madre, padre):
        p.getparent().remove(p)

    fixed = 0
    for t in body.iter(W + "t"):
        for plain, accented in (("Lunedi", "Lunedì"), ("Martedi", "Martedì"), ("Mercoledi", "Mercoledì"),
                                ("Giovedi", "Giovedì"), ("Venerdi", "Venerdì")):
            if t.text and t.text.strip() == plain:
                t.text = t.text.replace(plain, accented)
                fixed += 1
    doc.save(path)
    print("Il Piano genitoriale: signature section rebuilt (2 contact tables), %d weekday accents restored" % fixed)


def polizza():
    path = os.path.join(FINAL, "Polizza per la r.c.a.docx")
    doc = Document(path)
    paras = [p for p in doc.element.body.iter(W + "p") if ptext(p).strip()]
    if any(ptext(p).strip() == "POLIZZA PER LA R.C.A." for p in paras):
        print("Polizza per la r.c.a.: heading already there")
        return
    tra = next(p for p in paras if ptext(p).strip() == "TRA")
    tra.addprevious(para_like(tra, "POLIZZA PER LA R.C.A."))
    doc.save(path)
    print("Polizza per la r.c.a.: act heading restored before TRA")


def ricorso_decreto_ingiuntivo():
    """The agrarian 'Ricorso per decreto ingiuntivo': heading after the section line.
    Stage 2 took the small '- ricorso per decreto ingiuntivo.' line for it."""
    path = os.path.join(FINAL, "Ricorso per decreto ingiuntivo.docx")
    doc = Document(path)
    paras = [p for p in doc.element.body.iter(W + "p") if ptext(p).strip()]
    if any(ptext(p).strip() == "RICORSO PER DECRETO INGIUNTIVO" for p in paras):
        print("Ricorso per decreto ingiuntivo: heading already there")
        return
    sezione = next(p for p in paras if ptext(p).strip() == "Sezione Specializzata Agraria")
    sezione.addnext(para_like(sezione, "RICORSO PER DECRETO INGIUNTIVO"))
    doc.save(path)
    print("Ricorso per decreto ingiuntivo: act heading restored after 'Sezione Specializzata Agraria'")


if __name__ == "__main__":
    piano_genitoriale()
    polizza()
    ricorso_decreto_ingiuntivo()
