"""The output lint of an --interactive-sim transcript (v59): what the user reads, checked line by line -- no database, no
model. One finding per defect: (file, case, section, rule, excerpt).

Rules (each with the v58 line that showed it):
  bare_heading          a «**Art. N code — rubrica**» line in §2 followed by nothing (no comma line, no requisite, no
                        paragraph) before the next heading or section.
  penalty_requisite     a «- … requisito «…»» line whose requisite holds a penalty or a consequence (pena, punito,
                        reclusione, multa, si applica, ridotta, aumentata, titolo esecutivo, interessi legali …) or is
                        only an article reference.
  orphan_paragraph      a paragraph or a line starting with a space, a lower-case word or a closing mark; a bullet
                        starting «-,» or «-)».
  dangling_reference    «cfr./vedi sezione …», «sezione «…»» naming a title the answer does not have.
  no_client             round 1's «Parti:» with no «cliente» and no «Chi assisti» question.
  repeated_marker       the same «(da verificare: …)» or «Se …,» clause twice in a paragraph; a marker mid-sentence.
  missing_section       a criminal answer with no «Procedibilità e competenza»; a civil one with no «Mediazione» where
                        the chosen topics' matter requires it (d.lgs. 28/2010, art. 5); any answer with no Conclusioni.
  gated_term            a term (round-1 facts, §4 «Termini», a term-only topic) no listed topic carries, or of a code
                        no other listed topic is in; the querela wording in a case with no criminal topic.
  unearned_topic_point  a table's articles (topic_tables.TABLES) in «non trattate» or bare under a topic the table
                        drives; a topic with at least half its articles from one table not named after it.
  duplicate_default     «procedibile d'ufficio; competenza: tribunale» on more than one line.
  scaffold_bullet       a bullet that is only «[N] art. X» or a one-line gloss with no verb.
  misattributed_comma   a «co. N» under an article with no comma N, or «comma N dell'art. X» past X's last comma
                        (the commi known locally: the v52 material files, COMMI_SOURCES).
"""

import glob
import os
import re
import sys
from collections import Counter, OrderedDict
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RULES = ["bare_heading", "penalty_requisite", "orphan_paragraph", "dangling_reference", "no_client", "repeated_marker",
         "missing_section", "gated_term", "unearned_topic_point", "duplicate_default", "scaffold_bullet",
         "misattributed_comma"]

SECTION_TITLES = ["Risposta e strategia", "Norme applicate ai fatti", "Obiezioni e repliche", "Profili processuali",
                  "Conclusioni", "Altri profili"]
NORMS = "Norme applicate ai fatti"
PROCEDURAL = "Profili processuali"
COMMI_SOURCES = [os.path.expanduser("~/SunnitAI-V2/diagnose_reports/v52-candidate/cases/*/material.txt")]

_HEADING_RE = re.compile(r"^\*\*Art\.\s+(\S+)\s+(.+?)(?:\s+—\s+(.*?))?\*\*\s*$")
_REQUISITE_RE = re.compile(r"^-\s+(?:(?:co\.\s*\d+|testo),\s+requisito|Requisito)\s+«(.*)»\s*:\s*[^:]*$")
_PENALTY_RE = re.compile(r"\b(?:pena|pene|punit\w*|punibil\w*|reclusione|multa|ammenda|si\s+applica(?:no)?|ridott[aoie]|"
                         r"aumentat[aoie]|titolo\s+esecutivo|interessi\s+legali|in\s+misura\s+pari|penalit\w*|sanzion\w*)\b",
                         re.IGNORECASE)
_ARTICLE_REF_RE = re.compile(r"\b(?:articol[oi]|artt?\.)\s*\d+(?:-[a-z]+)?(?:\s*(?:,|e)\s*\d+(?:-[a-z]+)?)*", re.IGNORECASE)
_REFERENCE_WORDS = {"e", "o", "nel", "nell", "nella", "negli", "nei", "del", "dell", "della", "degli", "dei", "al", "all",
                    "alla", "agli", "ai", "in", "di", "a", "da", "primo", "secondo", "terzo", "quarto", "quinto", "comma",
                    "commi", "relazione", "suddette", "suddetti", "suddetta", "suddetto", "ipotesi", "previste",
                    "previsti", "prevista", "previsto", "stabilite", "stabiliti", "indicate", "indicati", "medesime",
                    "stesse", "cui", "le", "la", "il", "lo", "i", "gli", "l", "alle", "allo", "dello", "delle", "dai",
                    "dagli", "dal", "dalla", "nelle", "nello", "sensi", "casi", "caso", "stabilita"}
_FINITE_VERB_RE = re.compile(
    r"\b(?:è|e'|sono|ha|hanno|può|possono|deve|devono|va|vanno|sta|fa|dà|era|erano|fu|sarà|saranno|sarebbe|sarebbero|"
    r"viola|violano|include|includono|presenta|presentano|integrano|integrino|indica|indicano|contesta|sostiene|"
    r"rileva|evidenzia|mostra|determina|implica|produce|ricorre|accerta|attesta|vieta|obbliga|autorizza|precisa|"
    r"chiarisce|ritiene|osserva|afferma|nega|limita|esclude|aggrava|attenua|condiziona|"
    r"risulta|risultano|sussiste|sussistono|richiede|richiedono|prevede|prevedono|dispone|consente|esclude|comporta|"
    r"applica|applicano|rientra|rientrano|costituisce|costituiscono|configura|spetta|impone|attribuisce|stabilisce|"
    r"deriva|dipende|manca|mancano|occorre|serve|basta|emerge|emergono|appare|appaiono|resta|restano|rimane|integra|"
    r"punisce|tutela|regola|disciplina|estende|riguarda|permette|presuppone|esige|giustifica|legittima|opera|"
    r"trova|trovano|risponde|rispondono|incide|grava|consiste|vale|valgono|segue|conferma|confermano|dimostra|"
    r"dimostrano|prova|provano|qualifica|chiede|chiedono|ottiene|nega|esclud\w+|\w+(?:ano|ono|ava|eva|iva|avano|evano|"
    r"rebbe|rebbero|erà|irà|ato|ata|ati|ate|ito|ita|iti|ite|uto|uta|uti|ute|ndo))\b", re.IGNORECASE)
_CONCLUSIONS = "Conclusioni"
# d.lgs. 28/2010 art. 5 (as amended 2022): the c.c. matters whose actions need mediation first, by article range
MEDIATION_RANGES = [(456, 809), (832, 1172), (1559, 1570), (1571, 1614), (1803, 1812), (1834, 1860), (1882, 1932),
                    (2222, 2238), (2247, 2324)]


# --- parsing -----------------------------------------------------------------------------------------------------------

def read_cases(path: str) -> "OrderedDict[str, dict]":
    """{case: {"round1", "round2", "answer", "choice": the user's round-2 message}} of a transcript (one case or many)."""
    text = open(path, encoding="utf-8").read()
    out: "OrderedDict[str, dict]" = OrderedDict()
    for part in re.split(r"(?m)^## ", text)[1:]:
        cid, _, body = part.partition("\n")
        case = {"round1": "", "round2": "", "answer": "", "choice": ""}
        for r in re.split(r"(?m)^### ", body)[1:]:
            title, _, rest = r.partition("\n")
            kind = re.search(r"\((\w+),", title)
            key = {"round1": "round1", "round2": "round2", "answer": "answer"}.get(kind.group(1) if kind else "")
            if key:
                case[key] = rest.strip("\n")
            if key == "round2":
                user = re.search(r"— user:\s*(.*)$", title)
                case["choice"] = user.group(1).strip() if user else ""
        if case["round1"] or case["answer"]:
            out[cid.strip()] = case
    return out


def _refs_of(text: str) -> List[Tuple[str, str]]:
    """«43, 575 c.p.; 3, 7 Legge …» -> [("43", "c.p."), …]."""
    out = []
    for group in re.split(r";", text or ""):
        m = re.match(r"\s*(?:artt?\.\s*)?((?:\d+(?:-[a-z]+)?(?:,\s*|\s+e\s+))*\d+(?:-[a-z]+)?)\s+(.+?)\s*$", group)
        if m:
            code = m.group(2).strip()
            out += [(n, code) for n in re.split(r",\s*|\s+e\s+", m.group(1)) if n]
    return out


def topics(round1: str) -> List[dict]:
    """Round 1's listed topics: [{"n", "title", "refs", "line"}]."""
    out = []
    for line in (round1 or "").splitlines():
        m = re.match(r"^(\d+)\.\s+\*\*(.+?)\*\*\s*\(([^)]*)\)(.*)$", line)
        if m:
            out.append({"n": int(m.group(1)), "title": m.group(2).strip(), "refs": _refs_of(m.group(3)),
                        "line": line, "rest": m.group(4)})
    return out


def chosen(case: dict) -> List[dict]:
    listed = topics(case["round1"])
    numbers = [int(x) for x in re.findall(r"\d+", case.get("choice") or "")]
    picked = [t for t in listed if t["n"] in numbers]
    if not picked:
        m = re.search(r"Tratterò:\s*([^.]+)\.", case.get("round2") or "")
        names = [x.strip() for x in m.group(1).split(";")] if m else []
        picked = [t for t in listed if t["title"] in names]
    return picked


def sections(answer: str) -> List[Tuple[str, str]]:
    """[(title, text)] of the answer, the closing offer («Se vuoi, posso approfondire») left out."""
    answer = re.sub(r"(?ms)^Se vuoi, posso approfondire:.*\Z", "", answer or "")
    marks = [m for m in re.finditer(r"(?m)^\*\*(\d+)\.\s+([^*\n]+)\*\*\s*$", answer)
             if m.group(2).strip() in SECTION_TITLES]
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(answer)
        out.append((m.group(2).strip(), answer[m.end():end].strip("\n")))
    return out


def facts_block(round1: str) -> str:
    m = re.search(r"Ecco i fatti e i termini che ho ricavato dal caso:\s*\n(.*?)(?=\n\*\*Da chiarire\*\*|\nDimmi su quale|\Z)",
                  round1 or "", re.DOTALL)
    return m.group(1) if m else ""


def question_lines(round1: str) -> List[str]:
    m = re.search(r"\*\*Da chiarire\*\*\s*\n(.*?)(?=\nDimmi su quale|\Z)", round1 or "", re.DOTALL)
    return [x.strip() for x in (m.group(1) if m else "").splitlines() if re.match(r"\s*\d+\.\s", x)]


def _excerpt(text: str, n: int = 110) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n - 1] + "…"


def _code(code: str) -> str:
    c = code.lower().replace(" ", "")
    if c.startswith("legge") or "procedimentoamministrativo" in c or c == "l241":
        return "l241"
    return c


def _criminal(refs: List[Tuple[str, str]]) -> bool:
    return bool(refs) and all(_code(c) in ("c.p.", "c.p.p.") for _, c in refs)


def _number(n: str) -> int:
    m = re.match(r"\d+", n)
    return int(m.group(0)) if m else -1


# --- commi known locally -----------------------------------------------------------------------------------------------

_COMMI: Optional[Dict[Tuple[str, str], int]] = None


# articles the local material does not number (its text runs the commi together): their last comma, from the code
COMMI_KNOWN = {("628", "c.p."): 5}


def commi_index() -> Dict[Tuple[str, str], int]:
    """{(number, code): last comma} of the articles whose commi the local material numbers («(co. N)», «commi non
    riportati: …»); an article with no numbered comma is unknown (absent)."""
    global _COMMI
    if _COMMI is not None:
        return _COMMI
    found: Dict[Tuple[str, str], int] = {}
    for pattern in COMMI_SOURCES:
        for path in glob.glob(pattern):
            text = open(path, encoding="utf-8", errors="replace").read()
            for m in re.finditer(r"(?m)^\[art\.\s+(\S+)\s+([^\]]+)\]\n((?:>>[^\n]*\n)*)([^\n]*)", text):
                body = m.group(4)
                numbers = [int(x) for x in re.findall(r"\(co\.\s*(\d+)\)", body)]
                hidden = re.search(r"commi non riportati:\s*([\d,\s]+)", body)
                numbers += [int(x) for x in re.findall(r"\d+", hidden.group(1))] if hidden else []
                if numbers:
                    key = (m.group(1), _code(m.group(2)))
                    found[key] = max(found.get(key, 0), max(numbers))
    _COMMI = {**found, **COMMI_KNOWN}
    return _COMMI


# --- the rules ---------------------------------------------------------------------------------------------------------

def _paragraphs(text: str) -> List[str]:
    return [p for p in re.split(r"\n[ \t]*\n", text or "") if p.strip()]


def _sentences(text: str) -> List[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-Ü«(\[])", text or "") if s.strip()]


def lint_bare(norms: str) -> List[str]:
    out = []
    lines = (norms or "").splitlines()
    for i, line in enumerate(lines):
        if not _HEADING_RE.match(line.strip()):
            continue
        rest = []
        for nxt in lines[i + 1:]:
            if _HEADING_RE.match(nxt.strip()) or nxt.startswith("Altre norme del tema non trattate"):
                break
            rest.append(nxt)
        if not "".join(rest).strip():
            out.append(line.strip())
    return out


def _only_reference(requisite: str) -> bool:
    if not _ARTICLE_REF_RE.search(requisite):
        return False
    rest = _ARTICLE_REF_RE.sub(" ", requisite.lower())
    words = [w for w in re.findall(r"[a-zà-ù]+", rest) if w not in _REFERENCE_WORDS]
    return not words


def lint_penalty(norms: str) -> List[str]:
    out = []
    for line in (norms or "").splitlines():
        m = _REQUISITE_RE.match(line.strip())
        if m and (_PENALTY_RE.search(m.group(1)) or _only_reference(m.group(1))):
            out.append(line.strip())
    return out


def lint_orphan(text: str) -> List[str]:
    out = []
    for p in _paragraphs(text):
        first = p.split("\n", 1)[0]
        if re.match(r"^[ \t]+\S", first) and not re.match(r"^[ \t]+(?:[-*•]|\d+\.)\s", first):
            out.append(p)
            continue
        body = first.lstrip()
        if re.match(r"^[a-zà-ù]", body) and not re.match(r"^(?:[a-z]\)|artt?\.\s*\d|ex\s+art|n\.\s*\d|co\.\s*\d)", body):
            out.append(p)
            continue
        if re.match(r"^[,;:)\]]", body):
            out.append(p)
            continue
        for line in p.split("\n")[1:]:
            if re.match(r"^[ \t]+[^\s\-*•\d]", line) or re.match(r"^-\s*[,;:)\]]", line):
                out.append(line)
        if re.match(r"^-\s*[,;:)\]]", first):
            out.append(first)
    return out


def lint_dangling(text: str, titles: List[str], headings: List[str]) -> List[str]:
    out = []
    present = {t.lower() for t in titles + headings}
    for m in re.finditer(r"(?:\b(?:cfr\.|vedi|v\.)\s+(?:la\s+)?sezione\s*[*«\"“]?([^*»\"”)\n]{3,80})[*»\"”]?|"
                         r"\bsezione\s+[«*\"“]([^»*\"”\n]{3,80})[»*\"”])", text or "", re.IGNORECASE):
        name = (m.group(1) or m.group(2) or "").strip(" .,;:")
        if name and name.lower() not in present and not re.match(r"^(?:successiv|precedent|seguent|\d)", name.lower()):
            out.append(m.group(0))
    return out


def lint_no_client(round1: str) -> List[str]:
    parties = next((l for l in (round1 or "").splitlines() if l.startswith("Parti:")), None)
    if parties is None or "cliente" in parties.lower():
        return []
    if any("Chi assisti" in q for q in question_lines(round1)):
        return []
    return [parties]


def lint_markers(text: str) -> List[str]:
    out = []
    for p in _paragraphs(text):
        marks = re.findall(r"\(da verificare:\s*([^)]*)\)", p)
        for mark, n in Counter(m.strip().lower() for m in marks).items():
            if n > 1:
                out.append(f"(da verificare: {mark}) ×{n}")
        for m in re.finditer(r"\(da verificare:[^)]*\)", p):
            after = p[m.end():]
            if not re.match(r"^\s*(?:[.!?;]|$)", after) and not re.match(r"^[»”\"]?\s*[.!?;]", after):
                out.append(_excerpt(p[max(0, m.start() - 40):m.end() + 40]))
        clauses = [c.strip().lower() for c in re.findall(r"(?:^|(?<=[.!?]\s)|(?<=\n))Se\s+([^,]{5,120}),", p)]
        for clause, n in Counter(clauses).items():
            if n > 1:
                out.append(f"Se {clause}, ×{n}")
        for m in re.finditer(r",\s+Se\s+[^,]{5,120},|[a-zà-ù],\s+(?:Questi|Questo|Questa|Queste|Tale|Tali|Il|La|Le|Lo|Gli|I)\s",
                             p):
            out.append(_excerpt(p[max(0, m.start() - 40):m.end() + 40]))
    return out


def lint_missing(case: dict, secs: List[Tuple[str, str]]) -> List[str]:
    if not case["answer"].strip():
        return []
    out = []
    titles = [t for t, _ in secs]
    if _CONCLUSIONS not in titles:
        out.append("no Conclusioni")
    picked = chosen(case)
    refs = [r for t in picked for r in t["refs"]]
    proc = dict(secs).get(PROCEDURAL, "")
    if _criminal(refs) and "**Procedibilità e competenza**" not in proc:
        out.append("criminal answer with no «Procedibilità e competenza»")
    firsts = [t["refs"][0] for t in picked if t["refs"]]
    if any(_code(c) == "c.c." and any(lo <= _number(n) <= hi for lo, hi in MEDIATION_RANGES) for n, c in firsts) \
            and "**Mediazione**" not in proc:
        out.append("civil answer in a mediation matter with no «Mediazione»")
    return out


def _term_table():
    from src.rag import fact_signals
    out = []
    for t in fact_signals.TERMS:
        refs = _refs_of(re.sub(r"\(.*?\)", "", t.ref))
        out.append((re.sub(r"\s*\(.*?\)", "", t.label).lower(), refs, t.id))
    return out


_IMPRESCRIPTIBLE = {"azione negatoria": [("949", "c.c.")], "azione di rivendicazione": [("948", "c.c.")],
                    "azione confessoria": [("1079", "c.c.")]}


def _terms_in(text: str) -> List[Tuple[str, List[Tuple[str, str]], str]]:
    """[(the term's sentence, its articles, its name)] of a facts block or a «Termini» list."""
    table = _term_table()
    out = []
    for line in (text or "").splitlines():
        line = re.sub(r"^\s*-\s*", "", line.strip())
        for sentence in re.split(r"(?<=\.)\s+(?=[A-ZÀ-Ü])", line):
            low = sentence.lower()
            hit = next(((name, refs) for name, refs, _ in table if low.startswith(name)), None)
            if hit:
                out.append((sentence, hit[1], hit[0]))
            elif re.match(r"l['’](?:usucapione|azione possessoria|estinzione per non uso)", low):
                tid = ("usucapione_ordinaria" if "usucapione" in low[:14] else
                       "possessoria" if "possessoria" in low[:22] else "non_uso_servitu")
                name, refs = next((n, r) for n, r, i in table if i == tid)
                out.append((sentence, refs, name))
            else:
                name = next((k for k in _IMPRESCRIPTIBLE if low.startswith(k)), None)
                if name and "prescrizione" in low:
                    out.append((sentence, _IMPRESCRIPTIBLE[name], name))
    return out


def _stems(text: str) -> set:
    from src.rag import topic_step
    return {w[:6] for w in re.findall(r"[a-zà-ù]{4,}", (text or "").lower())} - {g[:6] for g in topic_step.GENERIC_STEMS}


# a term's matter (its regime) by c.c. article; a term of another code: its code
CIVIL_MATTERS = (("diritti reali", ((810, 1116), (1140, 1172))), ("condominio", ((1117, 1139),)),
                 ("contratti", ((1173, 2042),)), ("fatto illecito", ((2043, 2059),)))
TERM_MATTERS = {"prescrizione del risarcimento del danno extracontrattuale": "fatto illecito",
                "prescrizione del risarcimento da circolazione di veicoli": "fatto illecito",
                "prescrizione ordinaria": "contratti", "denuncia del difetto di conformità": "contratti",
                "responsabilità del venditore per il difetto di conformità": "contratti",
                "prescrizione dell'azione per il difetto di conformità non dolosamente occultato": "contratti"}


def _matter(n: str, code: str) -> str:
    c = _code(code)
    if c == "c.c." and _number(n) >= 0:
        return next((name for name, spans in CIVIL_MATTERS for lo, hi in spans if lo <= _number(n) <= hi), c)
    return c


def _term_carried(refs, name, listed) -> bool:
    """A listed topic -- not the term's own term-only topic -- holds one of the term's articles or is of its matter."""
    keys = {(n, _code(c)) for n, c in refs}
    matters = {TERM_MATTERS[name]} if name in TERM_MATTERS else {_matter(n, c) for n, c in keys}
    own = [t for t in listed if t["refs"] and {(n, _code(c)) for n, c in t["refs"]} <= keys
           and (_stems(name) & _stems(t["title"]))]
    others = [t for t in listed if t not in own]
    if not others:
        return bool(own)            # its own topic alone: the answer's
    return any(keys & {(n, _code(c)) for n, c in t["refs"]} or matters & {_matter(n, c) for n, c in t["refs"]}
               for t in others)


def lint_gated(case: dict, secs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """[(section, excerpt)]."""
    listed = topics(case["round1"])
    out = []
    for sentence, refs, name in _terms_in(facts_block(case["round1"])):
        if not _term_carried(refs, name, listed):
            out.append(("round 1 facts", sentence))
    table = _term_table()
    for t in listed:
        own = {(n, _code(c)) for n, c in t["refs"]}
        term = next(((name, refs) for name, refs, _ in table if own and own <= {(n, _code(c)) for n, c in refs}
                     and _stems(name) & _stems(t["title"])), None)
        if term and not _term_carried(term[1], term[0], listed):
            out.append(("round 1 topics", t["line"]))
    proc = dict(secs).get(PROCEDURAL, "")
    terms_part = re.search(r"\*\*Termini\*\*\s*\n(.*?)(?=\n\*\*|\Z)", proc, re.DOTALL)
    for sentence, refs, name in _terms_in(terms_part.group(1) if terms_part else ""):
        if not _term_carried(refs, name, listed):
            out.append((PROCEDURAL, sentence))
    if not any(_code(c) in ("c.p.", "c.p.p.") for t in listed for _, c in t["refs"]):
        for q in question_lines(case["round1"]):
            if "querela" in q.lower():
                out.append(("round 1 questions", q))
    return out


def lint_unearned(case: dict, secs: List[Tuple[str, str]]) -> List[str]:
    from src.rag import topic_tables
    out = []
    norms = dict(secs).get(NORMS, "")
    untreated = set()
    m = re.search(r"Altre norme del tema non trattate:\s*(.*)", norms)
    if m:
        untreated = {(n, _code(c)) for n, c in re.findall(r"art\.\s*(\S+)\s+([^,]+?)(?=,|\.?$)", m.group(1).strip())}
    bare = set()
    for line in lint_bare(norms):
        h = _HEADING_RE.match(line)
        bare.add((h.group(1), _code(h.group(2))))
    from src.rag import interaction_copy as C
    families = Counter(q for table in topic_tables.TABLES for q in table.get("questions") or [])
    for t in topics(case["round1"]):
        mine = {(n, _code(c)) for n, c in t["refs"]}
        for table in topic_tables.TABLES:
            own = [(n, _code(c)) for n, c in table["refs"]]
            share = len(mine & set(own))
            shorts = [C.QUESTION_SHORT.get(q) for q in table.get("questions") or [] if families[q] == 1]
            drives = t["title"].lower() == table["name"].lower() or any(s and s in t["rest"] for s in shorts)
            if not share or not drives:
                continue
            if t in chosen(case):
                for ref in own:
                    if ref in untreated or ref in bare:
                        out.append(f"topic {t['n']} «{t['title']}»: art. {ref[0]} {ref[1]} of «{table['name']}» "
                                   + ("not treated" if ref in untreated else "bare"))
            if share * 2 >= len(mine) and t["title"].lower() != table["name"].lower():
                out.append(f"topic {t['n']} «{t['title']}»: {share}/{len(mine)} articles of «{table['name']}», not its name")
    return out


def lint_default(proc: str) -> List[str]:
    lines = [l for l in (proc or "").splitlines() if "procedibile d'ufficio; competenza: tribunale" in l]
    return lines[1:] if len(lines) > 1 else []


def lint_scaffold(text: str, glosses: bool = True) -> List[str]:
    out = []
    for line in (text or "").splitlines():
        m = re.match(r"^\s*[-•*]\s+(.*)$", line)
        if not m:
            continue
        body = m.group(1).strip()
        if re.match(r"^(?:\[\d+\]\s*)?artt?\.\s*\d+(?:-[a-z]+)?(?:\s+[\w.]+){0,4}\.?$", body, re.IGNORECASE):
            out.append(line.strip())
            continue
        if re.search(r"«[^»]*»\s*:\s*(?:da provare|provato|non risulta)", body) or re.match(r"^(?:co\.\s*\d+|testo)\b", body):
            continue        # a status line of the code (§2)
        plain = re.sub(r"«[^»]*»|\([^)]*\)", " ", body)
        if glosses and not _FINITE_VERB_RE.search(plain) and not re.search(
                r"\bnon\s+(?:si\s+|lo\s+|la\s+|ne\s+|vi\s+|ci\s+)?[a-zà-ù]{2,}|\w+isc(?:e|ono)\b", plain):
            out.append(line.strip())
    return out


def lint_commi(norms: str, text_all: str) -> List[str]:
    index = commi_index()
    out = []
    current = None
    for line in (norms or "").splitlines():
        h = _HEADING_RE.match(line.strip())
        if h:
            current = (h.group(1), _code(h.group(2)))
            continue
        if current is None or current not in index:
            continue
        for k in re.findall(r"«\s*co\.\s*(\d+)[^»]{0,12}»|^\s*\(co\.\s*(\d+)\)", line):
            n = int(k[0] or k[1])
            if n > index[current]:
                out.append(f"art. {current[0]} {current[1]} (last comma {index[current]}): {_excerpt(line, 80)}")
    pattern = re.compile(r"\b(?:comma|co\.)\s*(\d{1,2})\s*,?\s*(?:dell['’]\s*|dello\s+|del\s+)?art(?:icolo|\.)\s*"
                         r"(\d+(?:-[a-z]+)?)\s*([a-z.]{2,8})", re.IGNORECASE)
    for m in pattern.finditer(text_all or ""):
        key = (m.group(2), _code(m.group(3)))
        if key in index and int(m.group(1)) > index[key]:
            out.append(f"art. {key[0]} {key[1]} (last comma {index[key]}): {_excerpt(m.group(0), 80)}")
    return out


def lint_case(case: dict) -> List[Tuple[str, str, str]]:
    """[(section, rule, excerpt)] of one case."""
    found: List[Tuple[str, str, str]] = []
    secs = sections(case["answer"])
    titles = [t for t, _ in secs]
    norms = dict(secs).get(NORMS, "")
    headings = re.findall(r"(?m)^\*\*([^*\n]+)\*\*\s*$", case["answer"] or "")
    found += [(NORMS, "bare_heading", x) for x in lint_bare(norms)]
    found += [(NORMS, "penalty_requisite", x) for x in lint_penalty(norms)]
    for title, text in secs:
        found += [(title, "orphan_paragraph", x) for x in lint_orphan(text)]
        found += [(title, "dangling_reference", x) for x in lint_dangling(text, titles, headings)]
        found += [(title, "repeated_marker", x) for x in lint_markers(text)]
        if title not in (NORMS, PROCEDURAL):        # §2's status lines and §4 are the code's
            found += [(title, "scaffold_bullet", x) for x in lint_scaffold(text, title in ("Altri profili",
                                                                                           "Obiezioni e repliche"))]
    found += [("round 1", "no_client", x) for x in lint_no_client(case["round1"])]
    found += [("answer", "missing_section", x) for x in lint_missing(case, secs)]
    found += [(s, "gated_term", x) for s, x in lint_gated(case, secs)]
    found += [("round 1", "unearned_topic_point", x) for x in lint_unearned(case, secs)]
    found += [(PROCEDURAL, "duplicate_default", x) for x in lint_default(dict(secs).get(PROCEDURAL, ""))]
    found += [(NORMS, "misattributed_comma", x) for x in lint_commi(norms, case["answer"])]
    return [(s, r, _excerpt(x)) for s, r, x in found]


def file_label(path: str) -> str:
    """The name a finding carries: the run directory for a run's interactive.md (every run writes the same file
    name), else the file name."""
    base = os.path.basename(path)
    if base == "interactive.md":
        return os.path.basename(os.path.dirname(os.path.abspath(path))) or base
    return base


def lint_file(path: str, cases: Optional["OrderedDict[str, dict]"] = None) -> List[Tuple[str, str, str, str, str]]:
    """[(file, case, section, rule, excerpt)]."""
    name = file_label(path)
    out = []
    for cid, case in (cases if cases is not None else read_cases(path)).items():
        out += [(name, cid, s, r, x) for s, r, x in lint_case(case)]
    return out


def tables(findings: List[tuple], files: List[str]) -> str:
    """The per-rule totals and the per-file × rule counts, markdown."""
    names = [file_label(f) for f in files]
    per_rule = Counter(f[3] for f in findings)
    lines = ["| rule | findings |", "|---|---|"] + [f"| {r} | {per_rule.get(r, 0)} |" for r in RULES]
    lines.append(f"| **all** | **{len(findings)}** |")
    lines += ["", "| file | " + " | ".join(RULES) + " | all |", "|---" * (len(RULES) + 2) + "|"]
    for n in names:
        c = Counter(f[3] for f in findings if f[0] == n)
        lines.append(f"| {n} | " + " | ".join(str(c.get(r, 0) or "") for r in RULES) + f" | {sum(c.values())} |")
    return "\n".join(lines)
