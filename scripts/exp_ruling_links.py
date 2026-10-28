"""Read-only: what the rulings (document_type 'interpretation') cite and decide.

Step 1 of the ruling-cards plan (22 Oct 2026). Today nothing links a ruling to the
articles it applies. The old LEGAL_CONCEPT / LEGAL_ACTION tags are procedural
boilerplate ("ricorso", "spese processuali"), and only about half the ruling titles
carry a number or a court. This script reads every ruling's own text, without any
model call, and reports:

  - the articles it cites, with their code ("art. 584 c.p.", "artt. 581 e 584 cod.
    pen.", "art. 41, secondo comma, c.p."). It uses the app's citation patterns,
    plus the comma forms rulings use ("secondo comma", "cpv.", "lett. e)");
  - its branch and court: from the Italgiure file name when it is one (every ruling,
    on 22 Oct: "snpen@s10@a2026@n00190@tS" = Cassazione penale, Sez. I, n. 190/2026,
    sentenza); else the section its opening names ("Sezione Quarta Penale") or the
    codes it cites most, and the court its opening names;
  - whether a full reference can be built (court + number + date_enacted);
  - the outcome its P.Q.M. states: accolto (annulla / cassa / accoglie), in parte
    accolto, rigettato, inammissibile, estinto. "Senza rinvio", the acquittal
    formulas ("perché il fatto non sussiste") and prescription are counted apart;
  - in criminal rulings, who appealed (the defendant, the prosecutor, the civil
    party), and from that which side the ruling favours.

--focus "584 c.p." adds a section on the rulings that cite that article: the
outcome by who appealed, the articles cited with it, and how many contain each of
--words.

Only counts are printed, never ruling text or names. --out gets one JSON line per
ruling (its id, number, date, the labels above and the articles; no text), for
step 2's pilot.

Run from the app folder (prod), so it uses the app's code and .env:
    cd /opt/chatbot && venv/bin/python /tmp/exp_ruling_links.py --limit 1000
    ... --focus "584 c.p." --words "schiaffo,pugno,cadut,trauma cranico,nesso causale,prevedibil"

Nothing is written to the database: every Neo4j session is read-only. The labels
come from patterns, not from reading the ruling. They are counts for deciding what
step 2 should extract, not facts to show a user.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.getcwd()):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.getcwd(), ".env"))

import neo4j  # noqa: E402

_open_session = neo4j.Driver.session


def _read_only_session(self, **config):
    config["default_access_mode"] = neo4j.READ_ACCESS
    return _open_session(self, **config)


def read_only() -> None:
    """Every session this process opens from now on is read-only. Called by main(),
    not on import: link_ruling_citations.py imports the parser and writes."""
    neo4j.Driver.session = _read_only_session


logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

from src.rag.defensive_generation import (  # noqa: E402
    _ARTICLE_RUN_RE, _CIVIL_CODES, _CODE_NAMES, _CODE_SPELLINGS, _PENAL_CODES, _RUN_NUMBER_RE,
    _article_num, _ruling_sections,
)

LONG_CHARS = 60000   # about 20k tokens: over the model's context, so step 2 must read it in parts

# --- Articles cited -----------------------------------------------------------

_ORD = r"(?:primo|secondo|terzo|quarto|quinto|sesto|settimo|ottavo|nono|decimo|ultimo|penultimo)"
# One piece of what sits between a cited number and its code in a ruling: "comma 2",
# "secondo comma", "cpv.", "co. 1", "n. 3", "lett. e)".
_PART = (r"(?:(?:co\.|comma|commi|cpv\.?|capoverso|" + _ORD + r"\s+comma)(?:\s*(?:\d+|" + _ORD + r"))?"
         r"|(?:n\.|num\.|numero|lett\.|lettera)\s*\w+\s*\)?)")
_LEAD_RE = re.compile(r"^\s*,?\s*(?:" + _PART + r"\s*,?\s*){0,4}(?:del(?:la|lo|l['’])?\s+|d\.\s*)?",
                      re.IGNORECASE)
_EXTRA_CODES = [(r"cost\.?(?!\w)|costituzione", "Cost."),
                # without the last dot, "cod. proc. pen, non ravvisando" (unlinked_citations.py, 28 Oct)
                (r"cod\.\s*proc\.\s*pen(?!\w)", "c.p.p."), (r"cod\.\s*proc\.\s*civ(?!\w)", "c.p.c."),
                (r"cod\.\s*pen(?!\w)", "c.p."), (r"cod\.\s*civ(?!\w)", "c.c.")]

# What _LEAD_RE does not pass, found by unlinked_citations.py (28 Oct, 6,000 rulings: 4,617 main-code mentions, about
# 7% of those with no code after them): a page footer ("Corte di Cassazione - copia non ufficiale 3"), a list
# ("lett. b) ed e)", "e 125, comma 3"), "1° co.", "comma 1 quater", "ultima parte" between the number and the code.
_FOOTER = re.compile(r"corte\s+di\s+cassazione\s*[-–—]\s*copia\s+non\s+ufficiale(?:\s+\d+)?", re.IGNORECASE)
_SUFF = "bis|ter|quater|quinquies|sexies|septies|octies|novies|nonies|decies"
_LIST_PART = re.compile(
    r"^\s*[,;-]?\s*(?:"
    r"\d*\s*[°ºª]\s*(?:co\.|comma)?"
    r"|(?:co\.|comma|commi|lett\.|lettera|lettere|nn?\.|num\.|par\.|paragrafo|punto|periodo)\s*[\w\-)]+"
    rf"(?:[\s-]+(?:{_SUFF})\b)?"
    r"|(?:e|ed)\s+[a-z]\)"
    rf"|(?:e|ed)\s+\d+(?:[\s-]*(?:{_SUFF})\b)?"
    r"|(?:primo|secondo|terzo|quarto|quinto|sesto|settimo|ultimo|penultimo)\s+(?:comma|periodo)"
    r"|(?:prima|seconda|ultima)\s+parte"
    rf"|(?:{_SUFF})\b"
    r")\s*,?", re.IGNORECASE)
# "art. 360, 1° co.": the run reads "360, 1", but the 1 is the comma's.
_ORDINAL_TAIL = re.compile(r"^\s*[°ºª]")


def skip_parts(text: str) -> str:
    """The text after a cited number without its commi, letters, numbers and list items, a page footer removed."""
    text = _FOOTER.sub(" ", text)
    for _ in range(10):
        m = _LIST_PART.match(text)
        if not m or not m.end():
            break
        text = text[m.end():]
    return text


def _code_after(tail: str):
    """The code a citation's tail names ("cod. pen.", "c.p.p.", "del codice civile"),
    or None: "art. 165 disp. att. c.p.p." and "art. 5 d.lgs. 231/2001" name none."""
    rest = tail[_LEAD_RE.match(tail).end():]
    for pattern, abbr in _CODE_SPELLINGS + _EXTRA_CODES:
        if re.match(rf"(?:{pattern})", rest, re.IGNORECASE):
            return abbr
    low = rest.lower()
    for name, abbr in _CODE_NAMES:
        if low.startswith(name):
            return abbr
    return None


def code_of_run(numbers: list, tail: str):
    """(numbers, code) for one run of cited numbers ("artt. 110 e 117") and the 120 characters after it; code None when
    no code follows. Past what _code_after reads, the code may follow a list or a page footer ("art. 606, comma 1,
    lett. b) ed e), cod. proc. pen."); in "art. 360, 1° co., n. 3, cod. proc. civ." the run's 1 is dropped."""
    code = _code_after(tail)
    if code:
        return numbers, code
    if _ORDINAL_TAIL.match(tail) and len(numbers) > 1:
        numbers = numbers[:-1]
    return numbers, _code_after(skip_parts(tail))


def cited_articles(text: str, past_lists: bool = True) -> list:
    """Every article the text cites with its code, as "584 c.p.", in order, each once.
    A number with no code after it is skipped: in a ruling it is often a special law's.
    P.Q.M. is blanked first: the codes' patterns allow spaces between the letters, so
    "art. 41 c.p. P.Q.M." would read as c.p.p. past_lists=False: the rule before 29 Oct
    (no code found past a list, an ordinal or a page footer)."""
    found = {}
    text = _PQM_RE.sub(" ; ", text or "")
    for m in _ARTICLE_RUN_RE.finditer(text):
        numbers, tail = _RUN_NUMBER_RE.findall(m.group(1)), text[m.end():m.end() + 120]
        numbers, code = code_of_run(numbers, tail) if past_lists else (numbers, _code_after(tail))
        if code:
            for n in numbers:
                found.setdefault(f"{_article_num(n)} {code}", None)
    return list(found)


# --- Court, branch ------------------------------------------------------------

_COURTS = [
    (r"corte\s+(?:suprema\s+)?di\s+cassazione|\bcassazione\b|\bcass\.", "Cassazione"),
    (r"consiglio\s+di\s+stato", "Consiglio di Stato"),
    (r"tribunale\s+amministrativo\s+regionale|\bT\.\s?A\.\s?R\.|\bTAR\b", "TAR"),
    (r"corte\s+costituzionale", "Corte costituzionale"),
    (r"corte\s+dei\s+conti", "Corte dei conti"),
    (r"corte\s+d['’]\s*appello|corte\s+di\s+appello", "Corte d'appello"),
    (r"\btribunale\b", "Tribunale"),
    (r"giudice\s+di\s+pace", "Giudice di pace"),
]
_SECTION_RE = re.compile(
    r"\bsezion[ei]\s+(?:unit[ei]\s+)?(?:(?:prima|seconda|terza|quarta|quinta|sesta|settima|feriale|[ivx]+|\d)\s+)?"
    r"(penal[ei]|civil[ei]|lavoro|tributaria)"
    r"|\bsez\.\s*(?:un\.?\s*|[ivx]+\s*|\d\s*)?(pen|civ|lav|trib)", re.IGNORECASE)


def court_of(header: str):
    """The court the opening names first (the issuing court; the lower court's
    "avverso la sentenza della Corte d'appello" comes after it), or None."""
    best = None
    for pattern, name in _COURTS:
        m = re.search(pattern, header or "", re.IGNORECASE)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), name)
    return best[1] if best else None


def branch_of(header: str, articles: list) -> str:
    """'penale' / 'civile' by the section the opening names, else by the codes the
    ruling cites (most of c.p. + c.p.p. against c.c. + c.p.c.); '?' on a tie."""
    m = _SECTION_RE.search(header or "")
    if m:
        return "penale" if (m.group(1) or m.group(2)).lower().startswith("pen") else "civile"
    codes = [a.split(" ", 1)[1] for a in articles]
    penal = sum(c in _PENAL_CODES for c in codes)
    civil = sum(c in _CIVIL_CODES for c in codes)
    return "penale" if penal > civil else "civile" if civil > penal else "?"


# --- Outcome --------------------------------------------------------------------

# Any case: "P.Q.M.", "p.q.m.", "Per Questi Motivi" (a 500-ruling trial, capitals only, found 67% vs 81%).
_PQM_RE = re.compile(r"\bp\.\s*q\.\s*m\b\.?|\bpqm\b|\bper\s+questi\s+motivi", re.IGNORECASE)
_CLOSING_RE = re.compile(r"cos[iì]\s+(?:è\s+)?decis|decis[oa]\s+in\s+roma|depositat[ao]\s+in\s+cancelleria",
                         re.IGNORECASE)
_GRANTED_RE = re.compile(r"\bannull|\bcassa\b(?!\s+delle\s+ammende)|\baccogli", re.IGNORECASE)
# The appeal or its grounds rejected -- not "rigetta la domanda", the claim, after a civil cassa.
_REJECTED_RE = re.compile(r"\brigett\w*[^.;]{0,40}?(?:ricors|motiv|resto)", re.IGNORECASE)
_INADMISSIBLE_RE = re.compile(r"inammissibil\w*[^.;]{0,60}?(?:ricors|motiv)|(?:ricors|motiv)\w*[^.;]{0,30}?inammissibil",
                              re.IGNORECASE)
_EXTINCT_RE = re.compile(r"\bestint|\bestinzione", re.IGNORECASE)
_ACQUITTAL_RE = re.compile(
    r"\bassolv|il\s+fatto\s+non\s+sussiste|non\s+costituisce\s+reato|non\s+aver\s+commesso\s+il\s+fatto"
    r"|non\s+è\s+previsto\s+dalla\s+legge\s+come\s+reato", re.IGNORECASE)
_PRESCRIPTION_RE = re.compile(r"prescrizion", re.IGNORECASE)
_NO_REMAND_RE = re.compile(r"senza\s+rinvio", re.IGNORECASE)


def dispositivo(text: str):
    """(the court's own operative part, anchored). That is the text after the last
    P.Q.M. that a closing ("Così deciso in Roma...") follows, else after the last
    P.Q.M. A ruling may quote the lower court's P.Q.M. in its facts."""
    text = text or ""
    marks = list(_PQM_RE.finditer(text))
    if not marks:
        return "", False
    anchored = [m for m in marks if _CLOSING_RE.search(text, m.end(), m.end() + 2500)]
    m = (anchored or marks)[-1]
    part = text[m.end():m.end() + 800].lstrip(" .:;,-")
    closing = _CLOSING_RE.search(part)
    return (part[:closing.start()] if closing else part), bool(anchored)


def outcome_of(part: str) -> str:
    granted = bool(_GRANTED_RE.search(part))
    refused = bool(_REJECTED_RE.search(part) or _INADMISSIBLE_RE.search(part))
    if granted and refused:
        return "in parte accolto"
    if granted:
        return "accolto"
    if _REJECTED_RE.search(part):
        return "rigettato"
    if _INADMISSIBLE_RE.search(part):
        return "inammissibile"
    if _EXTINCT_RE.search(part):
        return "estinto"
    return "?"


# --- Who appealed (criminal rulings) -------------------------------------------

_APPEAL_START_RE = re.compile(r"\bsu[il]?\s+ricors[oi]\b|\bricors[oi]\s+propost[oi]\b", re.IGNORECASE)
# Where the list of appellants ends: the decision appealed, or the defendant named as
# the one the prosecutor's appeal is against.
_APPEAL_END_RE = re.compile(r"\bavverso\b|nel\s+procedimento\s+a\s+carico|nei\s+confronti\s+d|\ba\s+carico\s+d",
                            re.IGNORECASE)
_DEFENDANT_RE = re.compile(r"\bnat[oa]\s+(?:a|il|in)\b|\bimputat|\bindagat", re.IGNORECASE)
_PROSECUTOR_RE = re.compile(r"(?i:procurat\w*\s+(?:generale|della\s+repubblica)|pubblico\s+ministero)|\bP\.\s*[GM]\.")
_CIVIL_PARTY_RE = re.compile(r"part[ei]\s+civil[ei]", re.IGNORECASE)
_GRANTED_OUTCOMES = ("accolto", "in parte accolto")
_REFUSED_OUTCOMES = ("rigettato", "inammissibile")


# Without the "sul ricorso proposto da" heading, the facts say it: "ha proposto ricorso per
# cassazione l'imputato", "il Procuratore generale ricorre per cassazione".
_APPEAL_VERB_RE = re.compile(r"\bpropost[oi]\s+ricors\w*|\bpropon\w*\s+ricors\w*|\bricorr\w*\s+per\s+cassazione",
                             re.IGNORECASE)
# A sentence end: not the dot of an abbreviation in capitals ("P.G. ha proposto").
_SENTENCE_END_RE = re.compile(r"(?<![A-Z])[.;]\s+(?=[A-Z]|$)")


def _appeal_span(header: str) -> str:
    m = _APPEAL_START_RE.search(header)
    if m:
        end = _APPEAL_END_RE.search(header, m.end(), m.end() + 1500)
        return header[m.start():end.start() if end else m.end() + 600]
    m = _APPEAL_VERB_RE.search(header)
    if not m:
        return ""
    before = _SENTENCE_END_RE.split(header[max(0, m.start() - 120):m.start()])[-1]
    after = header[m.end():m.end() + 150]
    stops = [x.start() for x in (_APPEAL_END_RE.search(after), _SENTENCE_END_RE.search(after)) if x]
    return before + m.group(0) + after[:min(stops) if stops else len(after)]


def appellant_of(header: str) -> str:
    """Who brought the appeal: 'imputato', 'pm', 'parte civile', joined by '+' when
    several did; '?' when the text does not say. Read from the opening's "sul ricorso
    proposto da ...", else from the sentence that says who appealed."""
    span = _appeal_span(header or "")
    if not span:
        return "?"
    who = [name for name, rx in (("imputato", _DEFENDANT_RE), ("pm", _PROSECUTOR_RE),
                                 ("parte civile", _CIVIL_PARTY_RE)) if rx.search(span)]
    return "+".join(who) or "?"


def favours_of(appellant: str, outcome: str) -> str:
    """'imputato' or "contro l'imputato" when one side appealed and the outcome is clear, else '?'."""
    if appellant == "imputato":
        side = 1
    elif appellant in ("pm", "parte civile", "pm+parte civile"):
        side = -1
    else:
        return "?"
    if outcome in _GRANTED_OUTCOMES:
        return "imputato" if side > 0 else "contro l'imputato"
    if outcome in _REFUSED_OUTCOMES:
        return "contro l'imputato" if side > 0 else "imputato"
    return "?"


# --- One ruling, and the reports --------------------------------------------------

def _plain(value):
    return None if value is None else value if isinstance(value, (int, float)) else str(value)


# --- Italgiure file names ------------------------------------------------------------
# Every ruling in the collection is a Cassazione decision downloaded from Italgiure (22 Oct
# 2026: 22,612 of 22,612). The file says it all: "snpen@s10@a2026@n00190@tS.clean.pdf"
# (criminal) or "snciv_s50_a2026_n01978_tO.clean.pdf" (civil) = archive, section, year,
# number, type. That is surer than any pattern on the text: the first run read the lower
# court the opening names ("avverso la sentenza della Corte d'appello") as the court for
# 40% of the rulings, and left 365 criminal ones without a branch.
_ITALGIURE_RE = re.compile(r"(snpen|snciv)[@_]s(\w)(\w)[@_]a(\d{4})[@_]n(\d+)[@_]t([a-z])", re.IGNORECASE)
_ROMAN = {"1": "I", "2": "II", "3": "III", "4": "IV", "5": "V", "6": "VI", "7": "VII"}
_SECTION_LETTERS = {"U": "U.", "F": "fer.", "L": "L.", "T": "T."}


def _section_label(section: str, sub: str) -> str:
    """'1','0' -> 'I'; 'U','0' -> 'U.'; 'L','0' -> 'L.'; '6','1' -> 'VI-1' (the old sub-sections)."""
    section = section.upper()
    label = _ROMAN.get(section) or _SECTION_LETTERS.get(section) or section + "."
    return label + (f"-{sub.upper()}" if sub != "0" else "")


def italgiure_of(*names) -> dict:
    """{kind, section, year, number, type} from the first name that is an Italgiure id, else None."""
    for name in names:
        m = _ITALGIURE_RE.search(str(name or ""))
        if m:
            kind, section, sub, year, number, kind_of_act = m.groups()
            return {"kind": kind.lower(), "section": _section_label(section, sub), "year": year,
                    "number": str(int(number)), "type": kind_of_act.upper()}
    return None


def read_ruling(doc_id: str, meta: dict, text: str, words=()) -> dict:
    """The labels for one ruling (no text in the result)."""
    header = (text or "")[:5000]
    articles = cited_articles(text)
    italgiure = italgiure_of(meta.get("file_name"), meta.get("name"))
    if italgiure:
        court, branch = "Cassazione", ("penale" if italgiure["kind"] == "snpen" else "civile")
    else:
        court = court_of(header) or court_of(meta.get("description") or "")
        branch = "amministrativo" if court in ("Consiglio di Stato", "TAR") else branch_of(header, articles)
    part, anchored = dispositivo(text)
    outcome = outcome_of(part) if part else "nessun P.Q.M."
    appellant = appellant_of(header) if branch == "penale" else "-"
    low = (text or "").lower()
    return {
        "id": doc_id, "number": _plain(meta.get("number")), "date": _plain(meta.get("date")),
        "italgiure": italgiure, "court": court, "branch": branch, "outcome": outcome, "pqm_anchored": anchored,
        "senza_rinvio": bool(_NO_REMAND_RE.search(part)), "acquittal_formula": bool(_ACQUITTAL_RE.search(part)),
        "prescription": bool(_PRESCRIPTION_RE.search(part)),
        "appellant": appellant, "favours": favours_of(appellant, outcome) if branch == "penale" else "-",
        "chars": len(text or ""), "articles": articles, "words": [w for w in words if w in low],
    }


def shape(value) -> str:
    """'12345/2019' -> '99999/9999', 'Rossi' -> 'aaaaa': a field's format without its content."""
    s = re.sub(r"\d", "9", "" if value is None else str(value))
    return re.sub(r"[^\W\d_]", "a", s)[:24] or "(empty)"


def full_reference(r: dict) -> bool:
    number = r.get("number") or (r.get("italgiure") or {}).get("number")
    return bool(r["court"] and re.search(r"\d", str(number or "")) and r["date"])


def focus_report(records: list, focus: str, words=()) -> dict:
    """The rulings citing `focus` ("584 c.p."): who appealed against the outcome,
    which side they favour, the articles cited with it, and the --words they contain."""
    hits = [r for r in records if focus in r["articles"]]
    penal = [r for r in hits if r["branch"] == "penale"]
    pro = [r for r in penal if r["favours"] == "imputato"]
    return {
        "n": len(hits),
        "branch": Counter(r["branch"] for r in hits),
        "appeal_outcome": Counter((r["appellant"], r["outcome"]) for r in penal),
        "favours": Counter(r["favours"] for r in penal),
        "pro": len(pro),
        "pro_flags": {"senza rinvio": sum(r["senza_rinvio"] for r in pro),
                      "formula assolutoria": sum(r["acquittal_formula"] for r in pro),
                      "prescrizione": sum(r["prescription"] for r in pro)},
        "with": Counter(a for r in hits for a in r["articles"] if a != focus),
        "words": {w: (sum(w in r["words"] for r in hits), sum(w in r["words"] for r in pro)) for w in words},
        "pro_two_words": sum(1 for r in pro if len(r["words"]) >= 2),
    }


def _median(values):
    values = sorted(values)
    return values[len(values) // 2] if values else 0


def _pct(n, total):
    return f"{n} ({n / total:.0%})" if total else "0"


def _row(counter, total=None, top=None):
    items = counter.most_common(top)
    return ", ".join(f"{k}: {_pct(v, total) if total else v}" for k, v in items) or "-"


def report(records: list, focus=None, words=(), top: int = 25) -> list:
    n = len(records)
    out = [f"Rulings read: {n}  (with text: {sum(1 for r in records if r['chars'])})",
           f"Text length: median {_median([r['chars'] for r in records])} chars; "
           f"over {LONG_CHARS} chars (over the model's context): {_pct(sum(r['chars'] > LONG_CHARS for r in records), n)}",
           ""]
    with_articles = [r for r in records if r["articles"]]
    out.append(f"Cite at least one article with its code: {_pct(len(with_articles), n)}; "
               f"median distinct articles per ruling: {_median([len(r['articles']) for r in with_articles])}")
    codes = Counter(c for r in records for c in {a.split(' ', 1)[1] for a in r["articles"]})
    out.append(f"Rulings citing each code: {_row(codes, n)}")
    articles = Counter(a for r in records for a in r["articles"])
    for title, keep in (("substantive (c.p., c.c., Cost.)", ("c.p.", "c.c.", "Cost.")),
                        ("procedure (c.p.p., c.p.c., c.p.a.)", ("c.p.p.", "c.p.c.", "c.p.a."))):
        part = Counter({a: k for a, k in articles.items() if a.split(" ", 1)[1] in keep})
        out.append(f"Most cited, {title}: {_row(part, top=top if 'substantive' in title else 10)}")
    out += ["", f"Branch: {_row(Counter(r['branch'] for r in records), n)}",
            f"Court named in the opening: {_row(Counter(r['court'] or '?' for r in records), n)}",
            f"Italgiure files (court, branch, section and number from the file name): "
            f"{_pct(sum(bool(r.get('italgiure')) for r in records), n)}",
            f"Full reference possible (court + number + date): {_pct(sum(full_reference(r) for r in records), n)}",
            f"document_number format: {_row(Counter(shape(r['number']) for r in records), top=8)}",
            f"date_enacted format: {_row(Counter(shape(r['date']) for r in records), top=5)}",
            ""]
    found = [r for r in records if r["outcome"] != "nessun P.Q.M."]
    out.append(f"P.Q.M. found: {_pct(len(found), n)}; anchored to the closing: {_pct(sum(r['pqm_anchored'] for r in found), n)}")
    for branch in ("penale", "civile", "amministrativo", "?"):
        group = [r for r in records if r["branch"] == branch]
        if group:
            out.append(f"Outcome, {branch} ({len(group)}): {_row(Counter(r['outcome'] for r in group), len(group))}")
    penal = [r for r in records if r["branch"] == "penale"]
    if penal:
        out += [f"Criminal: who appealed: {_row(Counter(r['appellant'] for r in penal), len(penal))}",
                f"Criminal: in favour of: {_row(Counter(r['favours'] for r in penal), len(penal))}",
                f"Criminal: senza rinvio {sum(r['senza_rinvio'] for r in penal)}, "
                f"formula assolutoria {sum(r['acquittal_formula'] for r in penal)}, "
                f"prescrizione {sum(r['prescription'] for r in penal)}"]
    if focus:
        f = focus_report(records, focus, words)
        out += ["", f"## Rulings citing {focus}: {f['n']}  (branch: {_row(f['branch'])})"]
        if f["appeal_outcome"]:
            out.append("   who appealed -> outcome: " + _row(Counter({f'{a} -> {o}': k for (a, o), k in f['appeal_outcome'].items()})))
        out.append(f"   in favour of: {_row(f['favours'])}")
        out.append(f"   in the defendant's favour: {f['pro']}  (" + ", ".join(f"{k} {v}" for k, v in f["pro_flags"].items()) + ")")
        out.append(f"   cited with it: {_row(f['with'], top=20)}")
        if words:
            out.append("   words (all citing / in the defendant's favour): "
                       + ", ".join(f"{w}: {a} / {b}" for w, (a, b) in f["words"].items()))
            out.append(f"   in the defendant's favour AND at least 2 of the words: {f['pro_two_words']}")
    return out


def main() -> None:
    read_only()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, help="read only the first N rulings (by id)")
    parser.add_argument("--batch", type=int, default=50, help="rulings per query (default 50)")
    parser.add_argument("--focus", help='an article to report on, e.g. "584 c.p."')
    parser.add_argument("--words", default="", help="comma-separated words or stems to count in the focus rulings")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--out", default="/tmp/ruling_links.jsonl", help="one JSON line per ruling, no text")
    args = parser.parse_args()

    focus = None
    if args.focus:
        parsed = cited_articles("art. " + args.focus.strip())
        if not parsed:
            parser.error(f"--focus {args.focus!r}: write it as \"584 c.p.\"")
        focus = parsed[0]
    words = [w.strip().lower() for w in args.words.split(",") if w.strip()]

    from src.rag.main import NEO4J_DATABASE, driver

    records, skipped, t0 = [], 0, time.time()
    with driver.session(database=NEO4J_DATABASE) as session, open(args.out, "w", encoding="utf-8") as out:
        query = "MATCH (d:Document {document_type:'interpretation'}) RETURN d.id AS id ORDER BY id"
        ids = [r["id"] for r in session.run(query + (" LIMIT $n" if args.limit else ""), n=args.limit)]
        print(f"Reading {len(ids)} rulings, {args.batch} per query ...", file=sys.stderr)
        for start in range(0, len(ids), args.batch):
            batch = ids[start:start + args.batch]
            for attempt in (1, 2):
                try:
                    meta = {r["id"]: r for r in session.run(
                        "MATCH (d:Document) WHERE d.id IN $ids RETURN d.id AS id, d.document_number AS number, "
                        "d.date_enacted AS date, d.description AS description, d.file_name AS file_name, "
                        "d.name AS name", ids=batch).data()}
                    sections = _ruling_sections(session, batch)
                    break
                except Exception as exc:
                    if attempt == 2:
                        print(f"   batch at {start} skipped: {type(exc).__name__}", file=sys.stderr)
                        meta, sections = None, None
                    else:
                        time.sleep(5)
            if meta is None:
                skipped += len(batch)
                continue
            for doc_id in batch:
                r = read_ruling(doc_id, meta.get(doc_id) or {}, " ".join(sections.get(doc_id) or []), words)
                records.append(r)
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
            done = start + len(batch)
            if done % 2000 < args.batch or done == len(ids):
                print(f"   {done}/{len(ids)} read, {time.time() - t0:.0f}s", file=sys.stderr)

    print("\n".join(report(records, focus, words, args.top)))
    if skipped:
        print(f"\nSkipped (query failed twice): {skipped}")
    print(f"\nDone. Nothing was written to the database. Labels per ruling (no text): {args.out}")


if __name__ == "__main__":
    main()
