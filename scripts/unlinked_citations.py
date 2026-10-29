"""Read-only (28 Oct): what the CITES linker leaves unlinked, sorted by kind, to size each fix before building it.

scripts/link_ruling_citations.py links an article mention only when a main code follows it ("art. 1490 c.c."); last
night it counted about 278,000 mentions with no code after them. Using the linker's own rule (exp_ruling_links'
_code_after on the 120 characters after the mention), each such mention is sorted into:
  special law            "art. 5 d.lgs. 231/2001", "art. 73 d.P.R. 9 ottobre 1990, n. 309"  -> by law (type, number, year)
  named act              "art. 2 TUIR", "art. 186 c.d.s.", "art. 41-bis ord. pen."          -> by act
  implementing rules     "art. 165 disp. att. c.p.p."
  EU / international     "art. 6 CEDU", "art. 267 TFUE", a directive or regulation
  same act, named before "art. 3 dello stesso decreto", "art. 12 cit."
  code or law named earlier in the sentence  "artt. 2043 c.c. e 2059" (the 2059)
  unknown
and the precedents the rulings cite ("Cass. Sez. U, n. 18672/2019", "Sez. 3, n. 12345 del 15/02/2019", "Cass. 7 aprile
2014, n. 8053") are counted by year and matched with our rulings (by the number and year in their Italgiure file stem):
which years we would need. A criminal citation's date is usually the hearing's, so a ruling deposited the next year
is counted as missing (the criminal share we have is a little low). Each special law and named act
is looked up among our documents: in the database, or not (the list of laws worth acquiring).

Prints counts, law references and document names only, never ruling text (an unknown mention's 30 characters after
the number, lower-cased and with digits masked, at most 20 of them, to show what the rules miss).

    cd /opt/chatbot && venv/bin/python /root/ruling_cards/unlinked_citations.py --sample 6000   # about 2 minutes
    cd /opt/chatbot && venv/bin/python /root/ruling_cards/unlinked_citations.py                 # every ruling
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

# --- classification (pure, tested) ---------------------------------------------------------------

_LEAD = re.compile(r"^\s*,?\s*(?:(?:co\.|comma|commi|lett\.|lettera|n\.|num\.|par\.|paragrafo|punto)\s*[\w\-)]+\s*,?\s*){0,4}"
                   r"(?:del(?:la|lo|le)?\s+|dell['’]\s*|degli\s+|dei\s+|in\s+)?", re.IGNORECASE)
_TYPE = (r"(d\.?\s?lgs\.?|lgs\.|d\.?\s?leg\.?vo|decreto\s+legislativo|d\.?\s?p\.?\s?r\.?|decreto\s+del\s+presidente\s+della\s+"
         r"repubblica|d\.?\s?l\.(?!\s*gs)|decreto[-\s]legge|r\.?\s?d\.?|regio\s+decreto|d\.?\s?p\.?\s?c\.?\s?m\.?|"
         r"d\.?\s?m\.?|decreto\s+ministeriale|l\.|legge)")
# "231/2001", "n. 231 del 2001", "n. 231 dell'8 giugno 2001"
_NUM_YEAR = re.compile(_TYPE + r"\s*(?:n\.?|nr\.?|num\.?)?\s*(\d{1,5})\s*(?:/|-|del\s+|dell['’]\s*|,\s*)\s*"
                       r"(?:\d{1,2}\s+\w+\s+)?((?:19|20)?\d{2})\b", re.IGNORECASE)
# "29 settembre 1973, n. 600"
_DATE_NUM = re.compile(_TYPE + r"\s*(?:\d{1,2}°?\s+\w+\s+)?((?:19|20)\d{2})\s*,?\s*(?:n\.?|nr\.?|num\.?)\s*(\d{1,5})\b",
                       re.IGNORECASE)
_NAMED = [
    (r"t\.?\s?u\.?\s?i\.?\s?r\.?(?!\w)|tuir\b", "TUIR (d.P.R. 917/1986)"),
    (r"t\.?\s?u\.?\s?e\.?\s?l\.?(?!\w)|tuel\b", "TUEL (d.lgs. 267/2000)"),
    (r"t\.?\s?u\.?\s?b\.?(?!\w)|tub\b", "TUB (d.lgs. 385/1993)"),
    (r"t\.?\s?u\.?\s?f\.?(?!\w)|tuf\b", "TUF (d.lgs. 58/1998)"),
    (r"c\.?\s?d\.?\s?s\.?(?!\w)|codice\s+della\s+strada|cod\.\s*strada|c\.\s*strada", "Codice della strada (d.lgs. 285/1992)"),
    (r"l\.\s*fall\.?|legge\s+fallimentare", "Legge fallimentare (r.d. 267/1942)"),
    (r"statuto\s+(?:dei\s+diritti\s+)?del\s+contribuente", "Statuto del contribuente (l. 212/2000)"),
    (r"cod(?:ice|\.)\s*(?:del\s+)?cons(?:umo|\.)", "Codice del consumo (d.lgs. 206/2005)"),
    (r"codice\s+(?:dei\s+contratti|degli\s+appalti)", "Codice dei contratti pubblici"),
    (r"codice\s+(?:in\s+materia\s+di\s+protezione\s+dei\s+dati|della\s+privacy)", "Codice privacy (d.lgs. 196/2003)"),
    (r"codice\s+antimafia|cod\.\s*antimafia", "Codice antimafia (d.lgs. 159/2011)"),
    (r"ord(?:inamento)?\.?\s*pen(?:it(?:enziario)?)?\.?(?!\w)", "Ordinamento penitenziario (l. 354/1975)"),
    (r"t\.?\s?u\.?\s*(?:imm|immigrazione)|testo\s+unico\s+(?:sull['’]\s*|dell['’]\s*)?immigrazione",
     "TU immigrazione (d.lgs. 286/1998)"),
    (r"t\.?\s?u\.?\s*stup|testo\s+unico\s+(?:degli\s+|sugli\s+)?stupefacenti", "TU stupefacenti (d.P.R. 309/1990)"),
    (r"codice\s+della\s+crisi", "Codice della crisi d'impresa (d.lgs. 14/2019)"),
    (r"codice\s+del\s+processo\s+tributario|proc\.\s*trib", "Processo tributario (d.lgs. 546/1992)"),
    (r"testo\s+unico|t\.\s?u\.(?!\w)", "testo unico (not named)"),
]
_NAMED_RE = [(re.compile(p, re.IGNORECASE), label) for p, label in _NAMED]
_IMPLEMENTING = re.compile(r"(?:disp(?:osizioni)?\.?\s*(?:di\s+)?att(?:uazione|\.)?|norme\s+di\s+attuazione|"
                           r"disp\.?\s*(?:prel|trans)\w*\.?)\s*(?:del(?:la|lo)?\s+|al\s+)?\s*"
                           r"(c\.?\s?p\.?\s?p\.?|c\.?\s?p\.?\s?c\.?|c\.?\s?c\.?|c\.?\s?p\.?|codice\s+\w+(?:\s+\w+)?)?", re.IGNORECASE)
_EU = re.compile(r"cedu\b|conv(?:enzione)?\.?\s+(?:europea|e\.?d\.?u\.?)|tfue\b|t\.?f\.?u\.?e\.|\btue\b|carta\s+dei\s+diritti|"
                 r"carta\s+di\s+nizza|direttiva|reg(?:olamento)?\.?\s*(?:\(?\s*(?:ue|ce|cee)\b)|protocollo\s+(?:addizionale|n\.)",
                 re.IGNORECASE)
_SAME_ACT = re.compile(r"(?:(?:dello?|della|del|nello?|nella)\s+)?(?:stess[oa]|medesim[oa]|citat[oa]|predett[oa]|suddett[oa]|"
                       r"richiamat[oa])\b|cit\.(?!\w)|sopra\s+(?:citat|richiamat)", re.IGNORECASE)
_EARLIER_CODE = re.compile(r"c\.\s?p\.\s?p\.|c\.\s?p\.\s?c\.|c\.\s?c\.|c\.\s?p\.(?!\s?[pc]\.)|cod\.\s*(?:civ|pen|proc)|"
                           r"codice\s+(?:civile|penale|di\s+procedura)|cost\.(?!\w)|costituzione", re.IGNORECASE)
_EARLIER_LAW = re.compile(_TYPE + r"\s*(?:n\.?\s*)?\d", re.IGNORECASE)
_SENTENCE_END = re.compile(r"[.;:]\s+(?=[A-ZÀ-Ý(«\"])|\n")
# A Cassation ruling cited, then its number and year in one of three forms. Run 2 (28 Oct) knew only "Cass. ... n. X/Y"
# and "n. X del Y" after "Cass." or "Sez. U": the criminal form "Sez. 3, n. 12345 del 15/02/2019" (no "Cass", the
# date as dd/mm/yyyy) and the civil "Cass. 7 aprile 2014, n. 8053" (the date first) were never counted.
_YEAR = r"(?:19|20)\d{2}"
_DATE = r"(?:\d{1,2}°?\s+[a-zà-ù]+\s+|\d{1,2}\s*[/.-]\s*\d{1,2}\s*[/.-]\s*)"
_PRECEDENT = re.compile(
    r"\b(?P<court>cass\b\.?(?:\s*(?:civ|pen|lav|trib)\.?)?|cassazione|s\.\s?u\.|ss\.\s?uu\.|"
    r"sez(?:ion[ei]|\.)?\s*(?:un(?:ite)?\b\.?|u\b\.?|\d{1,2}\b|[ivx]{1,4}\b|lav\b|trib\b|f(?:er(?:iale)?)?\b))"
    r"(?P<pre>[^;\n]{0,90}?)(?:"
    rf"(?<![\d/])(?:\bn\.?\s*)?(?P<n1>\d{{1,6}})\s*/\s*(?P<y1>{_YEAR})\b"                 # n. 18672/2019, 18672/2019
    rf"|\bn\.?\s*(?P<n2>\d{{1,6}})\s*,?\s*del(?:l['’]\s*|\s+){_DATE}?(?P<y2>{_YEAR})\b"    # n. 5501 del (15/02/)2020
    rf"|{_DATE}(?P<y3>{_YEAR})\s*,?\s*n\.?\s*(?P<n3>\d{{1,6}})\b"                        # 7 aprile 2014, n. 8053
    r")", re.IGNORECASE)
# "Cass. ... d.P.R. n. 115/2002": a law's number, not a ruling's (run 1 counted n. 115/2002, 103/2017, 546/1992).
_LAW_BEFORE_NUMBER = re.compile(
    r"\b(?:d\.?\s?p\.?\s?r\.?|d\.?\s?lgs\.?|lgs\.|d\.?\s?l\.|legge|\bl\.|r\.?\s?d\.?|d\.?\s?m\.?|"
    r"decreto(?:[\s-]+(?:legislativo|legge|ministeriale|del\s+presidente\s+della\s+repubblica))?|"
    r"reg(?:olamento|\.)?(?:\s*\(\s*(?:ue|ce|cee)\s*\))?|direttiva)\s*$", re.IGNORECASE)
# "... ricorso per cassazione ... (Corte cost. n. 186 del 2000)": the Constitutional Court's, not a Cassation ruling
# (run 2's 4th most cited "precedent", from the inadmissibility sentence).
_OTHER_COURT = re.compile(r"\bcost\b|costituzional|\bc\.\s?cost|europea|\bcgue\b|\bcedu\b", re.IGNORECASE)
# A "Sez." with no "Cass" can be another court's: "Cons. Stato, Sez. IV", "Trib. Milano, sez. II", "Corte conti".
_OTHER_COURT_BEFORE = re.compile(
    r"(?:cons(?:iglio|\.)?\s*(?:di\s+)?st(?:ato|\.)|\bt\.?\s?a\.?\s?r\.?\b|corte\s+(?:dei\s+)?conti|c\.\s?conti|"
    r"corte\s+(?:d['’]\s*)?app(?:ello|\.)|\bapp\.\s|trib(?:unale|\.)\s+[a-z]|comm(?:issione|\.)\s*trib|corte\s+giust|"
    r"grande\s+sezione|\bcgue\b|\bcedu\b|corte\s+e\.?d\.?u|corte\s+cost|\bc\.\s?cost)[^;\n]{0,30}$", re.IGNORECASE)
# Our rulings' Italgiure file stems carry their number and year: snpen@s10@a2026@n01234@tS.
_ITALGIURE = re.compile(r"(snpen|snciv)[@_]s\w\w[@_]a(\d{4})[@_]n0*(\d+)", re.IGNORECASE)


def _type_of(raw: str) -> str:
    t = re.sub(r"[\s.]", "", raw.lower())
    if t.startswith(("dlgs", "lgs", "dlegvo", "decretolegislativo")):
        return "d.lgs."
    if t.startswith(("dpr", "decretodelpresidente")):
        return "d.P.R."
    if t.startswith(("dpcm",)):
        return "d.P.C.M."
    if t in ("dl",) or t.startswith("decreto-legge") or t.startswith("decretolegge"):
        return "d.l."
    if t.startswith(("rd", "regiodecreto")):
        return "r.d."
    if t.startswith(("dm", "decretoministeriale")):
        return "d.m."
    return "l."


def _year(raw: str) -> str:
    if len(raw) == 4:
        return raw
    n = int(raw)
    return str(2000 + n if n <= 30 else 1900 + n)


def law_of(rest: str) -> Optional[str]:
    """'d.lgs. 231/2001' from the start of the text after a mention (its lead stripped), or None."""
    for pattern, number_first in ((_NUM_YEAR, True), (_DATE_NUM, False)):
        m = pattern.match(rest)
        if m:
            number, year = (m.group(2), m.group(3)) if number_first else (m.group(3), m.group(2))
            return f"{_type_of(m.group(1))} {int(number)}/{_year(year)}"
    return None


# Run 1 (28 Oct, 6,000 rulings): 44% "unknown", led by main-code citations today's rule misses -- a page footer between
# the number and the code, a list ("606, comma 1, lett. b) e 125, comma 3, cod. proc. pen."), "1° co.", "comma 1 quater".
_FOOTER = re.compile(r"corte\s+di\s+cassazione\s*[-–—]\s*copia\s+non\s+ufficiale(?:\s+\d+)?", re.IGNORECASE)
_SUFF = "bis|ter|quater|quinquies|sexies|septies|octies|novies|nonies|decies"
_PART = re.compile(r"^\s*[,;-]?\s*(?:"
                   r"\d+\s*[°ºª]\s*(?:co\.|comma)"
                   r"|(?:co\.|comma|commi|lett\.|lettera|lettere|nn?\.|num\.|par\.|paragrafo|punto|periodo)\s*[\w\-)]+"
                   rf"(?:[\s-]+(?:{_SUFF})\b)?"
                   r"|(?:e|ed)\s+[a-z]\)"
                   rf"|(?:e|ed)\s+\d+(?:[\s-]*(?:{_SUFF})\b)?"
                   r"|(?:primo|secondo|terzo|quarto|quinto|sesto|settimo|ultimo|penultimo)\s+(?:comma|periodo)"
                   rf"|(?:{_SUFF})\b"
                   r")\s*,?", re.IGNORECASE)
# The court fee of every Cassation ruling's closing (art. 13, comma 1-quater, d.P.R. 115/2002, as l. 228/2012 amended):
# procedure, never a link worth making nor a law worth acquiring.
BOILERPLATE_LAWS = {"d.P.R. 115/2002", "l. 228/2012"}
_SAME_BEFORE = re.compile(r"(?:stess|medesim|citat|predett|suddett)\w*\s*$", re.IGNORECASE)


def skip_parts(text: str) -> str:
    """The text after a mention without its commi, letters, numbers and list items, a page footer removed."""
    text = _FOOTER.sub(" ", text)
    for _ in range(10):
        m = _PART.match(text)
        if not m or not m.end():
            break
        text = text[m.end():]
    return text


def classify(tail: str, head: str, code_after=None) -> Tuple[str, str]:
    """(kind, detail) of an article mention with no code after it, from the 120 characters after the mention and the
    text before it. code_after (the linker's _code_after): a code found once the footer and the list parts are skipped
    is a main-code citation today's rule misses ("code missed by today's rule", the code)."""
    skipped = skip_parts(tail)
    if code_after:
        code = code_after(skipped)
        if code:
            return "code missed by today's rule", code
    rest = skipped[_LEAD.match(skipped).end():]
    law = law_of(rest)
    if law:
        return ("court-fee boilerplate", law) if law in BOILERPLATE_LAWS else ("special law", law)
    if _SAME_BEFORE.search(head[-40:] if head else ""):
        return "same act, named before", ""
    imp = _IMPLEMENTING.match(rest)
    if imp and imp.group(0).strip():
        code = re.sub(r"[\s.]", "", (imp.group(1) or "").lower())
        return "implementing rules", {"cpp": "disp. att. c.p.p.", "cpc": "disp. att. c.p.c.", "cc": "disp. att. c.c.",
                                      "cp": "disp. att. c.p."}.get(code, "disp. att. (other)")
    for pattern, label in _NAMED_RE:
        if pattern.match(rest):
            return "named act", label
    if _EU.match(rest):
        return "EU / international", ""
    if _SAME_ACT.match(rest):
        return "same act, named before", ""
    sentence = _SENTENCE_END.split(head)[-1] if head else ""
    if _EARLIER_CODE.search(sentence):
        return "code named earlier in the sentence", ""
    if _EARLIER_LAW.search(sentence):
        return "law named earlier in the sentence", ""
    return "unknown", re.sub(r"\d", "#", " ".join((rest if rest.strip() else tail)[:30].lower().split()))


_OWN_CASE = re.compile(r"ricors|r\.\s?g\.|\brgn\b|reg\.\s*gen|iscritt|ruolo", re.IGNORECASE)
FORMS = {"n1": "n. X/year", "n2": "n. X del (date) year", "n3": "date, n. X"}


def precedents(text: str, own: Optional[Tuple[str, str]] = None, forms: Optional[Counter] = None) -> List[Tuple[str, str]]:
    """(number, year) of every Cassation ruling the text cites -- not the case's own docket number ("sul ricorso
    iscritto al n. 15966/2025"), nor the ruling's own (number, year) `own` from its header. `forms` counts the form
    each one was written in."""
    text = _FOOTER.sub(" ", text or "")
    out = []
    for m in _PRECEDENT.finditer(text):
        if _OWN_CASE.search(m.group(0)) or _LAW_BEFORE_NUMBER.search(m.group("pre")[-30:]) \
                or _OTHER_COURT.search(m.group("pre")):
            continue
        if m.group("court").lower().startswith("sez") and _OTHER_COURT_BEFORE.search(text[max(0, m.start() - 60):m.start()]):
            continue
        key = next(k for k in FORMS if m.group(k))
        found = (str(int(m.group(key))), m.group("y" + key[1]))
        if found == own:
            continue
        out.append(found)
        if forms is not None:
            forms[FORMS[key]] += 1
    return out


def our_ruling(file_and_name: str) -> Optional[Tuple[str, str, str]]:
    """(branch, number, year) of one of our rulings, from its Italgiure file stem; None without one."""
    m = _ITALGIURE.search(file_and_name or "")
    return ("penale" if m.group(1).lower() == "snpen" else "civile", str(int(m.group(3))), m.group(2)) if m else None


# What the laws are, for the report to the PM (the most cited, by hand).
LAW_NAMES = {
    "d.lgs. 546/1992": "Processo tributario", "d.P.R. 600/1973": "Accertamento delle imposte sui redditi",
    "r.d. 267/1942": "Legge fallimentare", "d.P.R. 633/1972": "IVA",
    "d.lgs. 472/1997": "Sanzioni amministrative tributarie (principi generali)",
    "d.P.R. 602/1973": "Riscossione delle imposte sui redditi", "d.lgs. 152/2006": "Codice dell'ambiente",
    "l. 689/1981": "Depenalizzazione e sanzioni amministrative", "d.lgs. 165/2001": "Testo unico del pubblico impiego",
    "d.lgs. 150/2022": "Riforma Cartabia (processo penale)", "d.lgs. 149/2022": "Riforma Cartabia (processo civile)",
    "l. 134/2021": "Legge delega Cartabia (penale)", "l. 206/2021": "Legge delega Cartabia (civile)",
    "d.lgs. 504/1992": "ICI e finanza degli enti territoriali", "d.P.R. 131/1986": "Testo unico dell'imposta di registro",
    "d.l. 4/2019": "Reddito di cittadinanza", "l. 197/2022": "Legge di bilancio 2023 (tregua fiscale)",
    "d.m. 55/2014": "Parametri dei compensi forensi", "d.l. 83/2012": "Decreto Sviluppo 2012",
    "l. 103/2017": "Riforma Orlando (codice e processo penale)", "l. 190/2014": "Legge di stabilità 2015",
    "d.lgs. 276/2003": "Riforma Biagi (mercato del lavoro)", "d.l. 119/2018": "Decreto fiscale 2019 (pace fiscale)",
    "d.lgs. 507/1993": "Tributi locali (pubblicità, suolo pubblico, rifiuti)", "d.lgs. 446/1997": "IRAP",
    "d.l. 137/2020": "Decreto Ristori (Covid)", "l. 110/1975": "Legge sulle armi",
    "d.lgs. 471/1997": "Sanzioni tributarie non penali", "l. 296/2006": "Legge finanziaria 2007",
    "l. 69/2005": "Mandato d'arresto europeo", "d.P.R. 115/2002": "Testo unico spese di giustizia",
    "l. 228/2012": "Legge di stabilità 2013", "l. 392/1978": "Locazioni di immobili urbani (equo canone)",
    "l. 300/1970": "Statuto dei lavoratori", "d.lgs. 23/2015": "Jobs Act (tutele crescenti)",
    "l. 92/2012": "Riforma Fornero del lavoro", "l. 104/1992": "Legge sull'handicap",
    "d.P.R. 445/2000": "Testo unico documentazione amministrativa", "l. 47/1985": "Condono edilizio",
    "d.l. 201/2011": "Decreto Salva Italia", "l. 223/1991": "Licenziamenti collettivi e cassa integrazione",
    "l. 604/1966": "Licenziamenti individuali", "l. 89/2001": "Legge Pinto (equa riparazione)",
    "l. 67/2014": "Messa alla prova", "d.lgs. 271/1989": "Disposizioni di attuazione del c.p.p.",
    "l. 3/2019": "Spazzacorrotti", "l. 69/2019": "Codice rosso", "r.d. 1736/1933": "Legge sull'assegno",
    "r.d. 1669/1933": "Legge cambiaria", "l. 47/1948": "Legge sulla stampa",
    "d.P.R. 1124/1965": "Testo unico infortuni sul lavoro", "l. 335/1995": "Riforma Dini delle pensioni",
    "d.lgs. 28/2015": "Particolare tenuità del fatto", "l. 401/1989": "Scommesse clandestine e frode sportiva",
    "l. 157/1992": "Caccia", "d.lgs. 109/2006": "Illeciti disciplinari dei magistrati", "l. 184/1983": "Adozione",
    "l. 54/2006": "Affidamento condiviso", "d.l. 18/2020": "Decreto Cura Italia", "d.l. 34/2020": "Decreto Rilancio",
    "d.lgs. 241/1997": "Versamenti unificati e compensazione (F24)", "d.lgs. 218/1997": "Accertamento con adesione",
    "d.lgs. 158/2015": "Revisione delle sanzioni tributarie 2015", "l. 130/2022": "Riforma della giustizia tributaria",
    "d.lgs. 175/2024": "Testo unico della giustizia tributaria", "d.lgs. 173/2024": "Testo unico sanzioni tributarie",
    "l. 266/2005": "Legge finanziaria 2006", "l. 244/2007": "Legge finanziaria 2008", "l. 388/2000": "Legge finanziaria 2001",
}


# The laws whose corpus document's name does not carry their number: (type number/year) -> the name it starts with.
# Checked against our document names at run time; to review once by hand (Flavio's brief, job 3).
KNOWN_DOCUMENTS = {
    "d.P.R. 917/1986": "Testo unico delle imposte sui redditi", "d.lgs. 267/2000": "Testo unico degli enti locali",
    "d.P.R. 309/1990": "Testo Unico sugli Stupefacenti", "d.lgs. 74/2000": "Legge sui Reati Tributari",
    "d.lgs. 231/2001": "Responsabilità Amministrativa di Società ed Enti", "l. 241/1990": "Legge sul Procedimento Amministrativo",
    "d.lgs. 81/2008": "Testo Unico sulla Sicurezza del Lavoro", "d.P.R. 380/2001": "Testo unico in materia edilizia",
    "l. 898/1970": "Legge sul divorzio", "l. 431/1998": "Legge sulle locazioni abitative",
    "d.lgs. 206/2005": "Codice del Consumo", "d.lgs. 36/2023": "Codice dei contratti pubblici",
    "d.lgs. 196/2003": "Codice della Privacy", "d.lgs. 286/1998": "Testo unico sull'immigrazione",
    "d.lgs. 285/1992": "Codice della strada", "l. 212/2000": "Statuto dei Diritti del Contribuente",
    "d.lgs. 58/1998": "Testo Unico della Finanza", "d.lgs. 209/2005": "Codice delle Assicurazioni Private",
    "d.lgs. 14/2019": "Codice della crisi d'impresa", "d.lgs. 42/2004": "Codice dei beni culturali",
    "l. 354/1975": "Legge sull'ordinamento penitenziario", "d.lgs. 159/2011": "Codice antimafia",
    "d.lgs. 117/2017": "Codice del Terzo Settore", "d.lgs. 28/2010": "Mediazione civile",
    "d.lgs. 175/2016": "Testo unico delle società partecipate", "d.lgs. 151/2001": "Testo unico sulla maternità",
    "l. 247/2012": "Legge Professionale Forense", "d.lgs. 171/2005": "Codice della nautica da diporto",
    "d.lgs. 219/2006": "Codice dei medicinali", "d.lgs. 259/2003": "Codice delle Comunicazioni Elettroniche",
    "d.lgs. 82/2005": "Codice dell'amministrazione digitale", "d.lgs. 174/2016": "Codice di giustizia contabile",
    "d.lgs. 252/2005": "Testo Unico della Previdenza Complementare", "d.lgs. 198/2006": "Codice delle Pari Opportunità",
    "l. 218/1995": "Legge sul diritto internazionale privato", "d.lgs. 34/2018": "Testo Unico in materia di foreste",
    "d.l. 14/2017": "Sicurezza Urbana", "d.lgs. 297/1994": "Testo Unico delle Disposizioni Legislative Vigenti in Materia di Istruzione",
    "d.lgs. 79/2011": "Codice del Turismo", "d.lgs. 385/1993": "Testo unico bancario",
}
_NAMED_LAW = {"TUIR (d.P.R. 917/1986)": "d.P.R. 917/1986", "TUEL (d.lgs. 267/2000)": "d.lgs. 267/2000",
              "TUB (d.lgs. 385/1993)": "d.lgs. 385/1993", "TUF (d.lgs. 58/1998)": "d.lgs. 58/1998",
              "Codice della strada (d.lgs. 285/1992)": "d.lgs. 285/1992", "Legge fallimentare (r.d. 267/1942)": "r.d. 267/1942",
              "Statuto del contribuente (l. 212/2000)": "l. 212/2000", "Codice del consumo (d.lgs. 206/2005)": "d.lgs. 206/2005",
              "Codice privacy (d.lgs. 196/2003)": "d.lgs. 196/2003", "Codice antimafia (d.lgs. 159/2011)": "d.lgs. 159/2011",
              "Ordinamento penitenziario (l. 354/1975)": "l. 354/1975", "TU immigrazione (d.lgs. 286/1998)": "d.lgs. 286/1998",
              "TU stupefacenti (d.P.R. 309/1990)": "d.P.R. 309/1990", "Codice della crisi d'impresa (d.lgs. 14/2019)": "d.lgs. 14/2019",
              "Processo tributario (d.lgs. 546/1992)": "d.lgs. 546/1992"}


def find_document(law: str, docs: List[dict], resolve=None) -> Optional[str]:
    """The name of our document for 'd.lgs. 231/2001': by the table above, else by the number and year in a document's
    name or aliases (the app's resolve_code when given), else None."""
    prefix = KNOWN_DOCUMENTS.get(law)
    if prefix:
        hit = next((d for d in docs if (d.get("name") or "").lower().startswith(prefix.lower())), None)
        if hit:
            return hit["name"]
    number, year = law.split(" ", 1)[1].split("/")
    for d in docs:
        text = " ".join([d.get("name") or ""] + list(d.get("aliases") or []))
        if re.search(rf"\b{number}\b", text) and re.search(rf"\b{year}\b", text):
            return f"{d['name']} (by number and year: to check)"
    if resolve:
        try:
            hit = resolve(law, docs)
            return f"{hit['name']} (by name: to check)" if hit else None
        except Exception:
            return None
    return None


# --- the run --------------------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="What the CITES linker leaves unlinked, by kind (read-only)")
    p.add_argument("--sample", type=int, default=0, help="read this many random rulings (0: all)")
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--pause", type=float, default=0.5)
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--out", default="/tmp/unlinked_citations.json")
    args = p.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import exp_ruling_links as rl                      # the linker's own rule; sets up the app's path and .env
    rl.read_only()
    from src.rag.defensive_generation import _ARTICLE_RUN_RE, _RUN_NUMBER_RE
    from src.rag.issue_spotting import resolve_code
    from src.rag.main import NEO4J_DATABASE, driver

    t0 = time.time()
    kinds, laws, named, implementing, unknown = Counter(), Counter(), Counter(), Counter(), Counter()
    missed_codes, boilerplate = Counter(), Counter()
    law_rulings, law_branch = defaultdict(set), defaultdict(Counter)
    linked = bare = 0
    years, cited_precedents, precedent_rulings, forms = Counter(), Counter(), 0, Counter()
    cited_by, have, missing_years = Counter(), Counter(), defaultdict(Counter)
    with driver.session(database=NEO4J_DATABASE) as session:
        rulings = session.run("MATCH (d:Document {document_type:'interpretation'}) RETURN d.id AS id, "
                              "coalesce(d.file_name, '') + ' ' + coalesce(d.name, '') AS f").data()
        ours = {r for r in (our_ruling(x["f"]) for x in rulings) if r}       # every ruling we have, before any sample
        our_years = defaultdict(Counter)
        for b, _, y in ours:
            our_years[b][y] += 1
        if args.sample and args.sample < len(rulings):
            rulings = random.Random(28).sample(rulings, args.sample)
        branch = {r["id"]: ("penale" if "snpen" in r["f"] else "civile" if "snciv" in r["f"] else "altro") for r in rulings}
        own_of = {r["id"]: (our_ruling(r["f"]) or ("", "", ""))[1:] for r in rulings}     # (number, year)
        docs = session.run("MATCH (d:Document) WHERE d.document_type IN ['primary', 'special', 'legislation'] "
                           "RETURN d.id AS id, d.name AS name, d.aliases AS aliases").data()
        ids = [r["id"] for r in rulings]
        for start in range(0, len(ids), args.batch):
            if start:
                time.sleep(args.pause)
            texts = defaultdict(list)
            for row in session.run("MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE d.id IN $ids "
                                   "RETURN d.id AS doc, s.plain_text AS text", ids=ids[start:start + args.batch]).data():
                if row["text"]:
                    texts[row["doc"]].append(row["text"])
            for doc, parts in texts.items():
                text = rl._PQM_RE.sub(" ; ", " ".join(parts))
                found_precedent = False
                b = branch[doc]
                for number, year in precedents(text, own_of.get(doc), forms):
                    years[year] += 1
                    cited_precedents[(number, year)] += 1
                    found_precedent = True
                    cited_by[b] += 1
                    if (b, number, year) in ours:            # matched in the citing ruling's own branch
                        have[b] += 1
                    else:
                        missing_years[b][year] += 1
                precedent_rulings += found_precedent
                for m in _ARTICLE_RUN_RE.finditer(text):
                    n = len(_RUN_NUMBER_RE.findall(m.group(1)))
                    tail = text[m.end():m.end() + 120]
                    if rl._code_after(tail):
                        linked += n
                        continue
                    bare += n
                    kind, detail = classify(tail, text[max(0, m.start() - 250):m.start()], rl._code_after)
                    kinds[kind] += n
                    if kind == "code missed by today's rule":
                        missed_codes[detail] += n
                    elif kind == "court-fee boilerplate":
                        boilerplate[detail] += n
                    elif kind == "special law":
                        laws[detail] += n
                        law_rulings[detail].add(doc)
                        law_branch[detail][branch[doc]] += n
                    elif kind == "named act":
                        named[detail] += n
                        key = _NAMED_LAW.get(detail)
                        if key:
                            law_rulings[key].add(doc)
                            law_branch[key][branch[doc]] += n
                    elif kind == "implementing rules":
                        implementing[detail] += n
                    elif kind == "unknown":
                        unknown[detail] += n
            if (start // args.batch) % 40 == 0:
                print(f"  {min(start + args.batch, len(ids)):,}/{len(ids):,} rulings, {time.time() - t0:.0f}s", file=sys.stderr)
    driver.close()

    def pct(n, d):
        return f"{100 * n / d:.0f}%" if d else "-"
    print(f"\nRulings read: {len(ids):,}" + (f" (a random sample of {args.sample:,})" if args.sample else "")
          + f"; article mentions: linked by the code rule {linked:,}, with no code after them {bare:,}")
    print("\nMentions with no code after them, by kind:")
    for kind, n in kinds.most_common():
        print(f"  {n:>9,}  {pct(n, bare):>4}  {kind}")
    if missed_codes:
        print("\nMain-code citations today's rule misses (a footer, a list, '1° co.', 'comma 1 quater' between number "
              "and code): " + ", ".join(f"{c} {n:,}" for c, n in missed_codes.most_common()))
    if boilerplate:
        print("Court-fee boilerplate, not to link: " + ", ".join(f"{law} {n:,}" for law, n in boilerplate.most_common()))

    merged = Counter(laws)
    for label, n in named.items():
        if label in _NAMED_LAW:
            merged[_NAMED_LAW[label]] += n
    in_db = {law: find_document(law, docs, resolve_code) for law, _ in merged.most_common(300)}
    covered = sum(n for law, n in merged.items() if in_db.get(law))
    print(f"\nSpecial laws and named acts: {sum(merged.values()):,} mentions of {len(merged):,} laws; "
          f"in our database: {covered:,} ({pct(covered, sum(merged.values()))} of the mentions, top 300 laws checked)")

    def named_law(law):
        return f"{law} ({LAW_NAMES[law]})" if law in LAW_NAMES else law
    print(f"{'mentions':>9}  {'rulings':>7}  {'penale':>7}  {'civile':>7}  law -> our document")
    for law, n in merged.most_common(args.top):
        b = law_branch[law]
        print(f"{n:>9,}  {len(law_rulings[law]):>7,}  {b['penale']:>7,}  {b['civile']:>7,}  {named_law(law)} -> "
              f"{in_db.get(law) or 'NOT IN THE DATABASE'}")
    missing = [(law, n) for law, n in merged.most_common(300) if not in_db.get(law) and law not in BOILERPLATE_LAWS]
    print(f"\nMost cited laws NOT in the database (worth acquiring), {min(args.top, len(missing))} of {len(missing)}:")
    print(f"{'mentions':>9}  {'rulings':>7}  {'penale':>7}  {'civile':>7}  law")
    for law, n in missing[:args.top]:
        b = law_branch[law]
        print(f"{n:>9,}  {len(law_rulings[law]):>7,}  {b['penale']:>7,}  {b['civile']:>7,}  {named_law(law)}")
    if implementing:
        print("\nImplementing rules: " + ", ".join(f"{k} {n:,}" for k, n in implementing.most_common()))
    if named.get("testo unico (not named)"):
        print(f"'Testo unico' with no name: {named['testo unico (not named)']:,}")
    print(f"\nPrecedents cited: {sum(cited_precedents.values()):,} citations of {len(cited_precedents):,} rulings, in "
          f"{precedent_rulings:,} of the rulings read ({pct(precedent_rulings, len(ids))})")
    print("  written as: " + ", ".join(f"{f} {n:,}" for f, n in forms.most_common()))
    print("  by year: " + ", ".join(f"{y} {n:,}" for y, n in sorted(years.items(), key=lambda x: -int(x[0]))[:20]))
    decades = Counter(f"{y[:3]}0s" for y in years.elements())
    print("  by decade: " + ", ".join(f"{d} {n:,}" for d, n in sorted(decades.items(), reverse=True)))
    print("  most cited: " + ", ".join(f"n. {num}/{y} ({n})" for (num, y), n in cited_precedents.most_common(15)))
    print("  matched with our rulings by number and year, in the citing ruling's branch:")
    for b in ("penale", "civile"):
        if not cited_by[b]:
            continue
        print(f"    cited by {b} rulings: {cited_by[b]:,}, in our database {have[b]:,} ({pct(have[b], cited_by[b])})")
        print(f"      missing, by year: " + ", ".join(
            f"{y} {n:,}" for y, n in sorted(missing_years[b].items(), key=lambda x: -int(x[0]))[:20]))
        print(f"      our {b} rulings, by year: " + ", ".join(
            f"{y} {n:,}" for y, n in sorted(our_years[b].items(), key=lambda x: -int(x[0]))))
    print("\nUnknown, the most frequent next words (digits as #): "
          + "; ".join(f"'{k}' {n:,}" for k, n in unknown.most_common(20)))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"rulings": len(ids), "sample": args.sample, "linked": linked, "bare": bare, "kinds": kinds,
                   "missed_codes": missed_codes, "boilerplate": boilerplate,
                   "laws": {law: {"name": LAW_NAMES.get(law), "mentions": n, "rulings": len(law_rulings[law]),
                                  "branch": law_branch[law], "document": in_db.get(law)}
                            for law, n in merged.most_common(300)},
                   "our_law_documents": sorted(d["name"] for d in docs if d.get("name")),
                   "implementing": implementing, "precedent_years": years, "precedent_forms": forms,
                   "precedents_by_branch": {b: {"cited": cited_by[b], "in_our_database": have[b],
                                                "missing_by_year": missing_years[b], "ours_by_year": our_years[b]}
                                            for b in ("penale", "civile")},
                   "precedents_top": [[f"{num}/{y}", n] for (num, y), n in cited_precedents.most_common(100)]},
                  fh, ensure_ascii=False, indent=1)
    print(f"\nreport: {args.out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
