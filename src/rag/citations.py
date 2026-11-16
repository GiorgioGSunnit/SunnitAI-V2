"""v60e: one reading of an answer's citations, for every step of the interactive flow that reads them back -- the §2
paragraphs' assignment to their article (interactive_flow._paragraph_article), the «non trattate» line, the read-back
(readback) and the lint (scripts/answer_lint.py).

Live runs cited «art. 10-bis Legge sul Procedimento Amministrativo 2026» and «art. 148 Codice delle Assicurazioni
Private 2026»: the steps that read only the sigle (c.c., c.p., c.p.c., c.p.p., c.p.a.) lost the code, so a paragraph
went under the wrong heading and its article was listed «non trattata» while its text was written. Here every form is
read by plan_then_write.read_citations, the pipeline's own reading: a sigla or a code in full, else the name of a
document of the store (issue_spotting.cited_document over `documents`, the store's names: doc_lookup._fetch_doc_names).
With no store at hand (a replay, the lint of a file), the documents the text's own §2 headings name stand in
(heading_documents): «**Art. 3 Legge sul Procedimento Amministrativo 2026 — …**» names its law.
"""

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

# a §2 heading: «**Art. N <code> — <rubrica>**»; the code a sigla or a store name, spaces and all
HEADING_RE = re.compile(r"^\*\*Art\.\s+(\S+)\s+(.+?)(?:\s+—\s+(.*?))?\*\*\s*$")
_SIGLA_RE = re.compile(r"^(?:c\.\s?c\.|c\.\s?p\.(?:\s?[cpa]\.)?|c\.\s?d\.\s?s\.)$", re.IGNORECASE)


# v60h: a document name that names a normative act (a code, a law, a decree, a regulation, a directive, a collective
# agreement …) -- scripts/ingestion_issues.py lists the others, scripts/article_links.py --unitless lists these
NORMATIVE_NAME_RE = re.compile(
    r"\b(?:codice|costituzione|legge|leggi|l\.\s?\d|decreto|d\.\s?lgs|d\.\s?l\.|d\.\s?p\.\s?r|d\.\s?m\.|d\.\s?p\.\s?c\.\s?m|"
    r"r\.\s?d\.|regio|regolament\w*|direttiva|trattato|testo unico|t\.\s?u\.|statuto|ccnl|contratto collettivo|"
    r"accordo collettivo|convenzione|carta (?:dei|europea)|disposizioni|norme|ordinamento|disciplina|preleggi|"
    r"c\.\s?c\.|c\.\s?p\.|c\.\s?p\.\s?c\.|c\.\s?p\.\s?p\.|c\.\s?p\.\s?a\.|tuir|tub|tuf|gdpr)\b", re.IGNORECASE)


def normative_name(name: Any) -> bool:
    return bool(NORMATIVE_NAME_RE.search(" ".join(str(name or "").split())))


def code_key(code: Any) -> str:
    """A code as one key: its sigla («Codice Civile 2026» -> «c.c.»), else its name with single spaces."""
    from .plan_then_write import _code_key
    return _code_key(" ".join(str(code or "").split()))


def label(ref: Sequence[str]) -> str:
    return f"{ref[0]} {code_key(ref[1])}"


def compact(ref: Sequence[str]) -> str:
    """The read-back's key: «1218 c.c.», «3 leggesulprocedimentoamministrativo2026» (no spaces, lower case)."""
    return f"{ref[0]} {re.sub(r'\s+', '', code_key(ref[1]).lower())}"


def heading(line: str) -> Optional[Tuple[str, str, str]]:
    """(number, code, rubrica) of a §2 heading line, or None."""
    m = HEADING_RE.match((line or "").strip())
    return (m.group(1), code_key(m.group(2)), (m.group(3) or "").strip()) if m else None


def heading_documents(text: str) -> List[Dict[str, Any]]:
    """The documents the text's §2 headings name, sigle aside: [{"id", "name"}]."""
    names = []
    for line in (text or "").splitlines():
        h = heading(line)
        if h and not _SIGLA_RE.match(h[1]) and h[1] not in names:
            names.append(h[1])
    return [{"id": name, "name": name, "aliases": []} for name in names]


def _documents(text: str, documents: Optional[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    docs = [d for d in documents or [] if isinstance(d, dict) and d.get("name")]
    have = {d["name"] for d in docs}
    return docs + [d for d in heading_documents(text) if d["name"] not in have]


def cites(text: str, documents: Optional[Sequence[Dict[str, Any]]] = None
          ) -> List[Tuple[int, int, Tuple[str, str]]]:
    """(start, end, (number, code)) of every article text cites, in order: read_citations' explicit and nearest
    readings (a bare number takes the code of the nearest citation in its sentence), each code as code_key."""
    from .plan_then_write import read_citations
    out = []
    for start, end, _, refs, _ in read_citations(text or "", _documents(text, documents)):
        out += [(start, end, (str(r[0]), code_key(r[1]))) for r in refs if r]
    return out


def refs_in(text: str, documents: Optional[Sequence[Dict[str, Any]]] = None) -> List[Tuple[str, str]]:
    """(number, code) of every article text cites, in order, repeats kept."""
    return [ref for _, _, ref in cites(text, documents)]


# v72: one tolerant reading of article citations for intake, the charge, the named articles and the rubrica check --
# whatever defensive_generation._refs_in reads («art 584 cp», «art.584 c.p.», «articolo 584 del codice penale», the
# «artt.» lists), and also the bare «584 c.p.» and «art. 62 n. 6 c.p.» / «art. 62, co. 1, n. 2, c.p.» (the article,
# its numero dropped). A bare number right after «comma», «co.», «n.», «numero», «lett.» is never an article.
_NUMBERED_RE = None
_BARE_RE = None
_NOT_ARTICLE_BEFORE_RE = re.compile(r"(?:\bco\.|\bcomma|\bcommi|\bn\.|\bnn\.|\bnumer[oi]|\bnum\.|\blett\.|\blettera|\bart(?:t|icol[oi])?\.?|"
                                    r"[\d/.,-])\s*,?\s*$", re.IGNORECASE)


def _patterns():
    global _NUMBERED_RE, _BARE_RE
    if _NUMBERED_RE is None:
        from . import defensive_generation as dg
        codes = "|".join(p for p, _ in dg._CODE_SPELLINGS)
        _NUMBERED_RE = re.compile(
            r"\b(?:art(?:t)?\.?|articol[oi])\s*" + dg._ARTICLE_NUM
            + r"(?:\s*,?\s*(?:co\.|comma)\s*\d+)?\s*,?\s*(?:n\.|num\.|numero)\s*\d+\s*,?\s*(?:del\s+)?(" + codes + r")",
            re.IGNORECASE)
        _BARE_RE = re.compile(r"(?<![\w])" + dg._ARTICLE_NUM + r"\s*,?\s*(?:del\s+)?(" + codes + r")", re.IGNORECASE)
    return _NUMBERED_RE, _BARE_RE


def article_refs(text: str) -> List[Tuple[str, str]]:
    """(number, code) of every article the text cites, in order, repeats kept -- defensive_generation._refs_in's
    readings first (the same results on what it reads), then the numbered and the bare forms outside them."""
    return [ref for _, ref in article_cites(text)]


def article_cites(text: str) -> List[Tuple[int, Tuple[str, str]]]:
    """article_refs with each citation's position in the text."""
    from . import defensive_generation as dg
    text = text or ""
    numbered, bare = _patterns()
    found: List[Tuple[int, Tuple[str, str]]] = []
    spans: List[Tuple[int, int]] = []
    for start, end, refs in dg._article_lists(text):
        found += [(start, r) for r in refs]
        spans.append((start, end))

    def free(a: int, b: int) -> bool:
        return not any(a < e and s < b for s, e in spans)
    for m in dg._CITED_ARTICLE_RE.finditer(text):
        if not free(*m.span()):
            continue
        code = dg._code_of(m.group(2))
        if code:
            found.append((m.start(), (dg._article_num(m.group(1)), code)))
            spans.append(m.span())
    for m in numbered.finditer(text):
        code = dg._code_of(m.group(2))
        if code and free(*m.span()):
            found.append((m.start(), (dg._article_num(m.group(1)), code)))
            spans.append(m.span())
    for m in bare.finditer(text):
        code = dg._code_of(m.group(2))
        if not code or not free(*m.span()) or _NOT_ARTICLE_BEFORE_RE.search(text[max(0, m.start() - 14):m.start()]):
            continue
        found.append((m.start(), (dg._article_num(m.group(1)), code)))
        spans.append(m.span())
    return sorted(found, key=lambda x: x[0])


def store_names(documents: Sequence[Dict[str, Any]]) -> List[str]:
    return [str(d["name"]) for d in documents or [] if isinstance(d, dict) and d.get("name")]


def self_check(documents: Sequence[Dict[str, Any]], number: str = "12") -> List[Dict[str, Any]]:
    """Each store name read back from the citation forms the answers use; the ones that fail (scripts/
    citation_names_check.py on the server): [{"name", "form", "read"}]."""
    failures = []
    for name in store_names(documents):
        forms = (f"Ai sensi dell'art. {number} {name}, il termine decorre.",
                 f"L'art. {number}, comma 2, {name} prevede il termine.",
                 f"Si applicano gli artt. {number} e 13 {name}.",
                 f"**Art. {number} {name} — Rubrica**")
        for form in forms:
            if form.startswith("**"):
                got = heading(form)
                ok = bool(got) and got[0] == number and got[1] == code_key(name)
            else:
                got = refs_in(form, documents)
                ok = (number, code_key(name)) in got
            if not ok:
                failures.append({"name": name, "form": form, "read": got})
    return failures
