"""CCNL_PATH_ENABLED (28 Oct): a question about a collective agreement (the router's "ccnl" route) answered from
that agreement, in the version in force at the case's date, with the labour law around it. Built on the audit of
scripts/audit_ccnl.py: 3,683 documents (Document+LEGAL_DOC, document_type 'ccnl'), sector on all of them,
valid_from on 3,640, valid_until on 1,990, about one Testo Consolidato per sector, no links between versions;
Sections are fragments (no titolo_articolo, names that are pieces of headings, 3.6% article-shaped), embedding_v2
on all of them, the provider's watermark ("IlCCNL | S.I.A. Srl | Riproduzione Riservata") inside the texts.

  1 contract  the sector index (the distinct d.sector values, with their document counts): the question's words
              against each sector's own words (stems) and a table of common names (metalmeccanici, commercio,
              turismo, studi professionali, edili, colf/badanti, trasporti, bancari ...); a variant of one family
              (industria / artigianato / cooperative / P.I.-Confapi / Cisal-Anpit; convivente / non convivente)
              the text does not settle is a real ambiguity: the most likely one is used, the answer states the
              assumption and asks to confirm. No match: one constrained call (calculation._classify_sector_llm);
              still none, the path gives up and the chat's graph answers.
  2 version   at the case's date (the latest date the text gives, else today): the sector's Testo Consolidato
              when its validity covers it, else the latest CCNL or Testo Consolidato starting on or before it
              (an expired one is still applied until renewed, and the answer says so); then the accordi di
              rinnovo, accordi, verbali and newer CCNL starting after it and on or before the date, by
              valid_from, a signed agreement over an "ipotesi" of the same date. Logged (ccnl_version).
  3 clauses   inside the chosen documents only: v2 cosine over their Sections plus a lexical score on the
              topic's words, fused (RRF); each hit grown to its neighbours (the Sections' order: see order_key,
              logged) to rebuild the clause; cited by the "Art. N - Title" its text carries, else by its title
              text. Pay questions get the rows of the tables (level -> amounts, decorrenza) read from the text,
              never inferred. The watermark is stripped everywhere (clean).
  4 framework the primary labour law for the topic (c.c. 2094-2134, l. 300/1970, d.lgs. 81/2015, d.lgs.
              66/2003, l. 604/1966, d.lgs. 23/2015 ...), fetched from primary documents only; then the rulings
              citing those articles (source_tiers.support_rulings: CITES edges exist for the codes only).

The answer: the contract and the version used (a fixed header written here, with the assumption if any), the
clauses paraphrased with their reference (no CCNL sentence reproduced: a passage of VERBATIM_WORDS words shared
with the clauses is rewritten once, then dropped), the legal framework, the practical answer, what to verify. An
amount the material does not contain is removed with its sentence. On any failure the chat's graph answers.
"""

import logging
import os
import re
import threading
import time
import unicodedata
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import feature_flags
from .verbose_logger import vlog

logger = logging.getLogger(__name__)

FLAG = "CCNL_PATH_ENABLED"
ORDER_ENV = "CCNL_SECTION_ORDER"
TYPE = "ccnl"


def enabled() -> bool:
    return feature_flags.enabled(FLAG)


# --- cleaning ------------------------------------------------------------------------------------------

_WATERMARK_RE = re.compile(r"[-–—]?\s*Il\s?CCNL\s*\|\s*S\.?\s?I\.?\s?A\.?\s*S\.?r\.?l\.?\s*\|\s*Riproduzione\s+"
                           r"Riservata\s*[-–—]?", re.IGNORECASE)
_BOILERPLATE_RES = [
    _WATERMARK_RE,
    re.compile(r"\bwww\.ilccnl\.it\S*", re.IGNORECASE),
    re.compile(r"©\s*S\.?I\.?A\.?\s*S\.?r\.?l\.?[^\n]{0,60}", re.IGNORECASE),
    re.compile(r"\bRiproduzione\s+Riservata\b", re.IGNORECASE),
]


def clean(text: Optional[str]) -> str:
    """The text without the provider's watermark and boilerplate, whitespace runs kept to one (newlines kept)."""
    out = text or ""
    for pattern in _BOILERPLATE_RES:
        out = pattern.sub(" ", out)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r" *\n *", "\n", out)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def _fold(text: str) -> str:
    """Lowercase, accents removed, anything but letters and digits a space."""
    text = unicodedata.normalize("NFKD", (text or "").lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


# --- documents ------------------------------------------------------------------------------------------

TC, CCNL, RINNOVO, IPOTESI, ACCORDO, VERBALE, PROTOCOLLO, OTHER = (
    "testo consolidato", "ccnl", "accordo di rinnovo", "ipotesi", "accordo", "verbale", "protocollo", "other")
_KIND_RES = [
    (IPOTESI, re.compile(r"\bipotesi\b", re.I)),
    (TC, re.compile(r"\btesto (?:unico |coordinato )?consolidat\w*|\btesto coordinato\b", re.I)),
    (RINNOVO, re.compile(r"\baccordo\b.{0,30}\brinnov\w*|\brinnovo\b", re.I)),
    (CCNL, re.compile(r"^\s*(?:c\.?c\.?n\.?l\.?|contratto collettivo)\b", re.I)),
    (VERBALE, re.compile(r"\bverbal\w*\b", re.I)),
    (PROTOCOLLO, re.compile(r"\bprotocoll\w*\b", re.I)),
    (ACCORDO, re.compile(r"\baccord\w*\b|\bintesa\b", re.I)),
]
# Kinds that may be the base text, and those that amend it.
BASE_KINDS = (TC, CCNL)
AMENDING_KINDS = (CCNL, RINNOVO, ACCORDO, VERBALE, IPOTESI)


def doc_kind(doc: Dict[str, Any]) -> str:
    """The agreement's kind, from its document_subtype ("Testo Consolidato - ...", "Accordo di rinnovo", "Ipotesi
    di Accordo", "CCNL", "Verbale Integrativo"), else the part of its name after the dash."""
    subtype = str(doc.get("document_subtype") or "").strip()
    if not subtype:
        name = str(doc.get("name") or "")
        subtype = re.split(r"\s+[—–-]\s+", re.sub(r"^\s*\[CCNL\]\s*", "", name), maxsplit=1)[-1]
    for kind, pattern in _KIND_RES:
        if pattern.search(subtype):
            return kind
    return OTHER


def to_date(value: Any) -> Optional[date]:
    """A date from "2024-01-01", a datetime/date, or a Neo4j Date; None otherwise."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if all(hasattr(value, k) for k in ("year", "month", "day")):
        try:
            return date(int(value.year), int(value.month), int(value.day))
        except (TypeError, ValueError):
            return None
    m = re.match(r"\s*((?:19|20)\d{2})-(\d{1,2})-(\d{1,2})", str(value))
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


_NAME_DATE_RE = re.compile(r"\bdel\s+(\d{2})(\d{2})((?:19|20)\d{2})\b")


def name_date(name: str) -> Optional[date]:
    """The signing date in a name: "... Accordo di rinnovo del 10052018" -> 2018-05-10."""
    m = _NAME_DATE_RE.search(name or "")
    if not m:
        return None
    try:
        return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    except ValueError:
        return None


def _doc_view(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {**doc, "kind": doc_kind(doc), "from": to_date(doc.get("valid_from")) or name_date(doc.get("name") or ""),
            "until": to_date(doc.get("valid_until")), "signed": name_date(doc.get("name") or "")}


def short_name(doc: Dict[str, Any]) -> str:
    """"[CCNL] Metalmeccanica - Industria — Testo Consolidato - ..." -> "Metalmeccanica - Industria — Testo ..."."""
    return re.sub(r"^\s*\[CCNL\]\s*", "", str(doc.get("name") or doc.get("id") or "")).strip()


def _fmt(d: Optional[date]) -> str:
    return d.strftime("%d/%m/%Y") if d else "n.d."


# --- step 1: the contract -------------------------------------------------------------------------------

# Common names of a sector family: (label, the question's words, the sector names' words). Both sides are
# matched on folded text (_fold).
CONCEPTS: List[Tuple[str, str, str]] = [
    ("metalmeccanici", r"metalmeccanic\w*|officina meccanica|carpenteria metallica|siderurgic\w*|fonderi\w*",
     r"metalmeccanic"),
    ("commercio", r"commerci(?:o|ale)?\b|terziario|negozi\w*|commess[oaie]|supermercat\w*|grande distribuzione|"
                  r"vendita al dettaglio|cassier[ea]", r"terziario|\bcommercio\b|distribuzione"),
    ("turismo", r"turism\w*|turistic\w*|alberg\w*|hotel|pubblici esercizi|ristorant\w*|\bbar\b|camerier\w*|"
                r"\bcuoc[oa]\b|barista|pizzeri\w*|stabiliment\w* balnear\w*|campeggi\w*",
     r"turismo|pubblici esercizi|alberghi|ristorazione"),
    ("studi professionali", r"studi\w* professional\w*|studio legale|studio (?:di )?commercialist\w*|"
                            r"studio notarile|studio tecnico|studio medico|praticant\w*",
     r"studi professionali"),
    ("edilizia", r"\bedil\w*|cantier\w*|murator\w*|costruzion\w* edil\w*|impresa di costruzion\w*", r"edil"),
    ("colf", r"\bcolf\b|badant\w*|domestic\w*|baby ?sitter|assistenza familiare|lavoro domestico",
     r"\bcolf\b|domestic"),
    ("trasporti", r"trasport\w*|autotrasport\w*|camionist\w*|logistic\w*|spedizion\w*|corrier\w*|"
                  r"magazzinier\w*", r"trasport|logistic|spedizion"),
    ("credito", r"bancari\w*|\bbanca\b|\bbanche\b|\bcredito\b|istituto di credito", r"credito|banch|bancari"),
    ("chimici", r"chimic\w*|farmaceutic\w*", r"chimic|farmaceutic"),
    ("tessili", r"tessil\w*|abbigliament\w*", r"tessil|abbigliament"),
    ("alimentari", r"alimentar\w*|panifici\w*|pasticcer\w*", r"alimentar|panific"),
    ("pulizie", r"pulizi\w*|multiserviz\w*", r"pulizi|multiserviz"),
    ("vigilanza", r"vigilanz\w*|guardi\w* giurat\w*|servizi fiduciari", r"vigilanz|servizi fiduciari"),
    ("sanita privata", r"cas[ae] di cura|clinic\w* privat\w*|sanita privata|\brsa\b", r"case di cura|sanita|\brsa\b"),
    ("cooperative sociali", r"cooperativ\w* social\w*", r"cooperative sociali"),
    ("agricoltura", r"agricol\w*|bracciant\w*|florovivai\w*", r"agricol"),
    ("telecomunicazioni", r"telecomunicazion\w*|call center|telefonia", r"telecomunicazion"),
    ("scuola privata", r"scuol\w* privat\w*|istitut\w* scolastic\w* privat\w*", r"scuol|istruzione"),
    ("gomma plastica", r"\bgomma\b|plastic\w*", r"gomma|plastic"),
    ("legno arredo", r"\blegno\b|arredament\w*|mobilifici\w*", r"legno|arred"),
    ("grafici editoria", r"grafic\w*|editori\w*|tipografi\w*", r"grafic|editori"),
    ("igiene ambientale", r"igiene ambientale|raccolta (?:dei )?rifiuti|nettezza urbana", r"igiene ambientale|ambiental"),
]
# Variants of one family, by group: within a group, a variant the question names favours the sectors with it
# and counts against the sectors with another one. Most specific first (the first match is the variant).
VARIANT_GROUPS: Dict[str, List[Tuple[str, str]]] = {
    "forma": [("confapi", r"confapi|\bp i\b|piccola e media industria|\bpmi\b"),
              ("piccola industria", r"piccola industria"),
              ("artigianato", r"artigian\w*"),
              ("cooperative", r"cooperativ\w*"),
              ("industria", r"\bindustri\w*")],
    "convivenza": [("non convivente", r"\bnon convivent\w*"), ("convivente", r"(?<!non )\bconvivent\w*")],
    "sindacato": [("cisal-anpit", r"\bcisal\b|\banpit\b"), ("ugl", r"\bugl\b"), ("confsal", r"\bconfsal\b"),
                  ("confcommercio", r"confcommercio"), ("confesercenti", r"confesercenti")],
}
_STOP = {"del", "della", "dello", "dei", "delle", "degli", "di", "da", "dal", "dalla", "per", "con", "e", "ed",
         "il", "la", "lo", "le", "gli", "i", "un", "una", "uno", "in", "su", "tra", "fra", "a", "al", "alla", "ai",
         "ccnl", "contratto", "collettivo", "nazionale", "lavoro", "settore", "personale", "dipendenti", "addetti",
         "aziende", "imprese"}
# A sector's name ends with a date qualifier ("(dal 01.07.04)") that is no part of its words.
_DATE_QUALIFIER_RE = re.compile(r"\(\s*dal\s+[\d.\s/]+\)", re.I)


def _stem(word: str) -> str:
    return word.rstrip("aeiou") or word


def _words(text: str) -> List[str]:
    return [w for w in _fold(text).split() if w not in _STOP and len(w) >= 4 and not w.isdigit()]


def _variant(text_folded: str, group: str) -> Optional[str]:
    for label, pattern in VARIANT_GROUPS[group]:
        if re.search(pattern, text_folded):
            return label
    return None


def sector_head(sector: str) -> str:
    """The family part of a sector's name: before its first " - " or "(" ("Metalmeccanica P.I. - Confapi" ->
    "Metalmeccanica P.I.")."""
    name = _DATE_QUALIFIER_RE.sub(" ", sector or "")
    return re.split(r"\s+[-–—]\s+|\(", name, maxsplit=1)[0].strip()


def score_sectors(text: str, sectors: Dict[str, int]) -> List[Dict[str, Any]]:
    """Every sector the text points to, best first: {"sector", "score", "why", "variants", "docs"}. The score:
    the whole name written in the text +10; a common name (CONCEPTS) of its family +3; every word of its family
    part written (as a stem) +3; each other word of its name written +1; per variant group, the variant the text
    names +2, another variant of that group -2."""
    folded = _fold(text)
    stems = {_stem(w) for w in _words(text)}
    asked = {group: _variant(folded, group) for group in VARIANT_GROUPS}
    concepts = [(label, side) for label, words, side in CONCEPTS if re.search(words, folded)]
    out = []
    for sector, docs in sectors.items():
        name = _fold(_DATE_QUALIFIER_RE.sub(" ", sector))
        if not name:
            continue
        score, why = 0, []
        if len(name) >= 6 and re.search(rf"\b{re.escape(name)}\b", folded):
            score += 10
            why.append("named")
        for label, side in concepts:
            if re.search(side, name):
                score += 3
                why.append(f"concept {label}")
        head = [_stem(w) for w in _words(sector_head(sector))]
        if head and all(h in stems for h in head):
            score += 3
            why.append("family words")
        rest = [_stem(w) for w in _words(sector)][len(head):]
        extra = [w for w in rest if w in stems]
        if extra and score:
            score += len(extra)
            why.append(f"words {', '.join(extra)}")
        if not score:
            continue
        variants = {group: _variant(name, group) for group in VARIANT_GROUPS}
        for group, wanted in asked.items():
            if not wanted or not variants[group]:
                continue
            if variants[group] == wanted:
                score += 2
                why.append(f"variant {wanted}")
            else:
                score -= 2
                why.append(f"other variant {variants[group]} (asked {wanted})")
        out.append({"sector": sector, "score": score, "why": why, "variants": variants, "docs": docs,
                    "unmatched": len([w for w in _words(sector) if _stem(w) not in stems])})
    out.sort(key=lambda c: (-c["score"], -_preference(c), c["unmatched"], -c["docs"], c["sector"]))
    return out


def _preference(candidate: Dict[str, Any]) -> int:
    """Between variants the text does not settle: the industry's own contract, then a name with no variant,
    then the others; no union qualifier before one."""
    variants = candidate.get("variants") or {}
    form = variants.get("forma")
    rank = 2 if form == "industria" else 1 if form is None else 0
    return rank * 2 + (0 if variants.get("sindacato") else 1)


def identify_contract(text: str, sectors: Dict[str, int], llm: bool = True) -> Dict[str, Any]:
    """{"sector", "how", "ambiguous", "alternatives", "candidates", "assumption"}; sector None when nothing
    matches. Ambiguous: two or more sectors share the top score -- the text did not settle the variant; the
    first by _preference is used and the others are the alternatives to confirm."""
    candidates = score_sectors(text, sectors)
    info: Dict[str, Any] = {"sector": None, "how": None, "ambiguous": False, "alternatives": [],
                            "candidates": [{k: c[k] for k in ("sector", "score", "why", "docs")} for c in candidates[:8]]}
    if candidates:
        top = candidates[0]["score"]
        tied = [c for c in candidates if c["score"] == top]
        info["sector"] = candidates[0]["sector"]
        info["how"] = "; ".join(candidates[0]["why"])
        if len(tied) > 1:
            info["ambiguous"] = True
            info["alternatives"] = [c["sector"] for c in tied[1:5]]
    elif llm and sectors:
        from .calculation import _classify_sector_llm
        picked = _classify_sector_llm(text, sorted(sectors))
        if picked:
            info.update(sector=picked, how="llm pick from the sector list", ambiguous=True)
    info["assumption"] = bool(info["sector"] and info["ambiguous"])
    return info


# --- step 2: the version --------------------------------------------------------------------------------

_MONTHS = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto", "settembre", "ottobre",
           "novembre", "dicembre"]
_FULL_NUMERIC_RE = re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-]((?:19|20)\d{2})\b")
_FULL_WORDS_RE = re.compile(rf"\b(\d{{1,2}})(?:°|º)?\s+({'|'.join(_MONTHS)})\s+((?:19|20)\d{{2}})\b", re.I)
_MONTH_YEAR_RE = re.compile(rf"\b({'|'.join(_MONTHS)})\s+((?:19|20)\d{{2}})\b", re.I)
_YEAR_RE = re.compile(r"(?<![/\d])((?:19|20)\d{2})\b(?!\s*/)")


def _month_end(year: int, month: int) -> date:
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    return date.fromordinal(nxt.toordinal() - 1)


def relevant_date(text: str, today: Optional[date] = None) -> Tuple[date, str]:
    """(the date the version must be in force at, how it was read): the latest date the text gives -- a full
    date as written, a month or a year at its last day (not past today) -- else today. A law's year ("l.
    300/1970") is no date."""
    today = today or date.today()
    found: List[Tuple[date, str]] = []
    spans = []
    for m in _FULL_NUMERIC_RE.finditer(text or ""):
        try:
            found.append((date(int(m.group(3)), int(m.group(2)), int(m.group(1))), m.group(0)))
            spans.append(m.span())
        except ValueError:
            pass
    for m in _FULL_WORDS_RE.finditer(text or ""):
        try:
            found.append((date(int(m.group(3)), _MONTHS.index(m.group(2).lower()) + 1, int(m.group(1))), m.group(0)))
            spans.append(m.span())
        except ValueError:
            pass
    for m in _MONTH_YEAR_RE.finditer(text or ""):
        if any(a <= m.start() < b for a, b in spans):
            continue
        found.append((min(_month_end(int(m.group(2)), _MONTHS.index(m.group(1).lower()) + 1), today), m.group(0)))
        spans.append(m.span())
    for m in _YEAR_RE.finditer(text or ""):
        if any(a <= m.start() < b for a, b in spans):
            continue
        found.append((min(date(int(m.group(1)), 12, 31), today), m.group(0)))
    if not found:
        return today, "today (no date in the text)"
    when, written = max(found)
    return when, f"from the text ({written})"


def choose_version(docs: Sequence[Dict[str, Any]], when: date) -> Dict[str, Any]:
    """{"base", "amendments", "dropped", "reason", "expired"} for the sector's documents at `when`:
      base        the Testo Consolidato whose validity covers the date; else the CCNL or Testo Consolidato with
                  the latest valid_from on or before it; else the latest accordo di rinnovo starting on or before
                  it; else (the date precedes them all) the earliest CCNL or Testo Consolidato; else any document
                  starting on or before it
      amendments  CCNL, accordi di rinnovo, accordi, verbali and ipotesi starting after the base and on or before
                  the date, by valid_from -- an ipotesi dropped when a signed agreement has its date
      expired     the base's valid_until is before the date (still applied until renewed)."""
    items = [_doc_view(d) for d in docs]
    out: Dict[str, Any] = {"base": None, "amendments": [], "dropped": [], "reason": "", "expired": False}
    if not items:
        out["reason"] = "no documents"
        return out
    tcs = [d for d in items if d["kind"] == TC]
    covering = [d for d in tcs if (d["from"] is None or d["from"] <= when) and (d["until"] is None or when <= d["until"])]
    started = [d for d in items if d["kind"] in BASE_KINDS and d["from"] and d["from"] <= when] \
        or [d for d in items if d["kind"] == RINNOVO and d["from"] and d["from"] <= when]
    if covering:
        base = max(covering, key=lambda d: (d["from"] or date.min, d.get("id") or ""))
        out["reason"] = "the Testo Consolidato covers the date"
    elif started:
        base = max(started, key=lambda d: (d["from"], d["kind"] == TC, d.get("id") or ""))
        out["reason"] = f"the latest {base['kind']} starting on or before the date"
    else:
        bases = [d for d in items if d["kind"] in BASE_KINDS and d["from"]]
        anyone = [d for d in items if d["from"] and d["from"] <= when]
        if bases:
            base = min(bases, key=lambda d: (d["from"], d.get("id") or ""))
            out["reason"] = "no CCNL in force at the date: the earliest one"
        elif anyone:
            base = max(anyone, key=lambda d: (d["from"], d.get("id") or ""))
            out["reason"] = "no CCNL or Testo Consolidato: the latest document starting on or before the date"
        else:
            base = max(items, key=lambda d: (d["from"] or date.min, d.get("id") or ""))
            out["reason"] = "no dated document: the latest one"
    out["base"] = base
    out["expired"] = bool(base["until"] and base["until"] < when)
    after = [d for d in items if d is not base and d["kind"] in AMENDING_KINDS and d["from"]
             and d["from"] <= when and (base["from"] is None or d["from"] > base["from"])]
    signed_dates = set()
    for d in after:
        if d["kind"] != IPOTESI:
            signed_dates.update(x for x in (d["from"], d["signed"]) if x)
    kept = []
    for d in after:
        if d["kind"] == IPOTESI and ({d["from"], d["signed"]} - {None}) & signed_dates:
            out["dropped"].append({"name": short_name(d), "why": "an ipotesi with a signed agreement of the same date"})
            continue
        kept.append(d)
    out["amendments"] = sorted(kept, key=lambda d: (d["from"], d.get("id") or ""))
    return out


# --- step 3: the clauses --------------------------------------------------------------------------------

# Topic -> the question's words, the words a clause on it uses, the primary labour law around it (article,
# "c.c." or the law's label as article_codes.law_label writes it).
TOPICS: Dict[str, Dict[str, Any]] = {
    "retribuzione": {"re": r"retribu\w*|stipend\w*|\bpaga\b|minim[io] (?:tabellar|contrattual|retributiv)\w*|"
                           r"tabell\w* retributiv\w*|busta paga|quanto (?:prende|guadagna|spetta)|lordo|"
                           r"superminimo|aument\w* (?:retributiv|contrattual)\w*",
                     "keywords": ["minimi tabellari", "retribuzione tabellare", "minimi", "tabellare", "paga base",
                                  "minimo contrattuale", "conglobato", "aumento", "decorrenza", "livello"],
                     "pay": True, "framework": [("2099", "c.c."), ("36", "Cost.")]},
    "ferie": {"re": r"\bferie\b", "keywords": ["ferie", "periodo feriale", "settimane", "giorni lavorativi",
                                               "fruizione", "godimento"],
              "framework": [("2109", "c.c."), ("10", "d.lgs. 66/2003")]},
    "preavviso": {"re": r"preavviso|dimission\w*|dimett\w*",
                  "keywords": ["preavviso", "termini di preavviso", "dimissioni", "indennità sostitutiva",
                               "licenziamento"],
                  "framework": [("2118", "c.c."), ("2119", "c.c.")]},
    "prova": {"re": r"periodo di prova|\bin prova\b|patto di prova",
              "keywords": ["periodo di prova", "prova", "durata", "giorni di lavoro effettivo", "mesi"],
              "framework": [("2096", "c.c.")]},
    "malattia": {"re": r"malatti\w*|comporto|infortuni\w*",
                 "keywords": ["malattia", "comporto", "conservazione del posto", "trattamento economico",
                              "certificato medico", "infortunio"],
                 "framework": [("2110", "c.c.")]},
    "disciplinare": {"re": r"disciplinar\w*|contestazion\w*|sanzion\w*|richiamo|\bmulta\b|sospension\w* dal "
                           r"servizio|provvediment\w*",
                     "keywords": ["provvedimenti disciplinari", "disciplinari", "contestazione", "rimprovero",
                                  "richiamo", "multa", "sospensione", "giustificazioni"],
                     "framework": [("7", "l. 300/1970"), ("2106", "c.c."), ("2104", "c.c.")]},
    "licenziamento": {"re": r"licenzi\w*",
                      "keywords": ["licenziamento", "giusta causa", "giustificato motivo", "recesso"],
                      "framework": [("2118", "c.c."), ("2119", "c.c."), ("1", "l. 604/1966"), ("3", "l. 604/1966"),
                                    ("3", "d.lgs. 23/2015")]},
    "orario": {"re": r"orario|straordinari\w*|lavoro notturno|\bturni\b|riposo settimanale",
               "keywords": ["orario di lavoro", "orario normale", "straordinario", "maggiorazione",
                            "ore settimanali", "lavoro notturno"],
               "framework": [("3", "d.lgs. 66/2003"), ("5", "d.lgs. 66/2003")]},
    "tempo determinato": {"re": r"tempo determinato|contratto a termine|\bproroga\b",
                          "keywords": ["tempo determinato", "contratto a termine", "proroga", "durata massima"],
                          "framework": [("19", "d.lgs. 81/2015"), ("21", "d.lgs. 81/2015")]},
    "part-time": {"re": r"part[- ]?time|tempo parziale",
                  "keywords": ["tempo parziale", "part-time", "clausole elastiche", "lavoro supplementare"],
                  "framework": [("4", "d.lgs. 81/2015"), ("6", "d.lgs. 81/2015")]},
    "apprendistato": {"re": r"apprendist\w*",
                      "keywords": ["apprendistato", "apprendista", "piano formativo", "durata"],
                      "framework": [("41", "d.lgs. 81/2015"), ("42", "d.lgs. 81/2015")]},
    "inquadramento": {"re": r"mansion\w*|inquadrament\w*|livell\w* di inquadramento|qualific\w*|declaratori\w*",
                      "keywords": ["inquadramento", "classificazione", "livello", "mansioni", "declaratoria"],
                      "framework": [("2103", "c.c."), ("2095", "c.c.")]},
    "tfr": {"re": r"\btfr\b|trattamento di fine rapporto|liquidazione",
            "keywords": ["trattamento di fine rapporto", "tfr", "anticipazione"], "framework": [("2120", "c.c.")]},
    "mensilita": {"re": r"tredicesim\w*|quattordicesim\w*|mensilit\w* aggiuntiv\w*",
                  "keywords": ["tredicesima", "quattordicesima", "mensilità aggiuntiva", "gratifica natalizia"],
                  "framework": [("2099", "c.c.")]},
    "permessi": {"re": r"permess\w*|\brol\b|ex festivit\w*",
                 "keywords": ["permessi retribuiti", "riduzione dell'orario", "rol", "ex festività", "permessi"],
                 "framework": []},
}


def topics_of(text: str) -> List[str]:
    folded = (text or "").lower()
    return [name for name, topic in TOPICS.items() if re.search(topic["re"], folded)]


def topic_keywords(text: str, topics: Sequence[str]) -> List[str]:
    """The words a clause on the topics would use; with no topic, the question's own content words."""
    words: List[str] = []
    for name in topics:
        words += [w for w in TOPICS[name]["keywords"] if w not in words]
    if not words:
        words = [w for w in dict.fromkeys(_words(text)) if len(w) >= 5][:8]
    return words


_HEADING_RE = re.compile(
    r"\b(Art(?:icolo|\.)?)\s*(\d{1,3}(?:\s*-?\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?)"
    r"\b\s*(?:[-–—.:)]\s*|(?=\n))", re.IGNORECASE)
_TITLE_STOP = {"Al", "Il", "La", "Lo", "Le", "Gli", "I", "Ai", "Per", "Nel", "Nella", "Nei", "In", "Fermo", "Ferma",
               "Salvo", "Ogni", "Dal", "Dalla", "È", "A", "Fatto", "Fatti", "Con", "Qualora", "Ove", "Tra", "Fra",
               "Sono", "Si", "Ad", "Alla", "Alle", "Agli", "Un", "Una", "Ciascun", "Ciascuna"}


def _title_after(text: str) -> str:
    """The heading's title: the words after "Art. N -" up to the clause's first sentence (a capitalised word
    that is no acronym, a comma number "1.", a newline or a full stop), at most 12 words."""
    line = (text or "").split("\n", 1)[0]
    words: List[str] = []
    for i, word in enumerate(line.split()):
        bare = word.strip("()[]«»\"',;")
        if i and (re.match(r"^\d+[.)]$", word) or (bare[:1].isupper() and not (bare.isupper() and len(bare) <= 6))
                  or bare in _TITLE_STOP):
            break
        if not i and (bare in _TITLE_STOP or re.match(r"^\d+[.)]$", word)):
            break
        words.append(word)
        if word.endswith((".", ";", ":")) or len(words) >= 12:
            break
    return " ".join(words).rstrip(".;:,- ").strip()


def article_headings(text: str) -> List[Tuple[int, str, str]]:
    """[(position, "Art. 16", "Diritti e rapporti sindacali")] for every article heading in the text; the title
    is "" when the heading has none on its line ("Articolo 24 -")."""
    out = []
    for m in _HEADING_RE.finditer(text or ""):
        number = re.sub(r"\s+", "", m.group(2))
        number = re.sub(r"(\d)-?(bis|ter|quater|quinquies|sexies|septies|octies|novies|decies)", r"\1-\2", number, flags=re.I)
        prefix = (text or "")[max(0, m.start() - 6):m.start()].lower()
        if re.search(r"(?:dell'|all'|nell'|dall'|sull'|l'|ex)\s*$", prefix):
            continue                                   # "dell'art. 7": a reference, not a heading
        out.append((m.start(), f"Art. {number}", _title_after(text[m.end():m.end() + 160])))
    return out


def opens_with_heading(text: str) -> bool:
    heads = article_headings(text or "")
    return bool(heads) and heads[0][0] <= 3


ORDER_CANDIDATES = ("rel.order", "rel.position", "rel.index", "rel.seq", "s.order", "s.position", "s.index",
                    "s.seq", "s.chunk_index", "id_suffix", "element_id")


def order_value(row: Dict[str, Any], key: str) -> Optional[int]:
    """A Section row's value for an order candidate: a property of its CONTAINS edge ("rel.*") or of the
    Section ("s.*"), the number ending its id ("...::12"), or the number ending its elementId ("4:uuid:123",
    creation order)."""
    if key.startswith("rel."):
        value = (row.get("rel") or {}).get(key[4:])
    elif key.startswith("s."):
        value = (row.get("props") or {}).get(key[2:])
    elif key == "id_suffix":
        m = re.search(r"(?:::|_|-)(\d+)$", str(row.get("id") or ""))
        value = m.group(1) if m else None
    elif key == "element_id":
        m = re.search(r"(\d+)$", str(row.get("element_id") or ""))
        value = m.group(1) if m else None
    else:
        value = None
    try:
        return int(value) if value is not None and str(value).strip() != "" else None
    except (TypeError, ValueError):
        return None


def order_key(rows: Sequence[Dict[str, Any]], forced: Optional[str] = None) -> Optional[str]:
    """The first candidate (ORDER_CANDIDATES, or CCNL_SECTION_ORDER when set to one of them) set to a distinct
    integer on 90% or more of a document's Sections; None when none is (the database's order is kept)."""
    if not rows:
        return None
    candidates = [forced] if forced in ORDER_CANDIDATES else list(ORDER_CANDIDATES)
    for key in candidates:
        values = [order_value(r, key) for r in rows]
        present = [v for v in values if v is not None]
        if len(present) >= 0.9 * len(rows) and len(set(present)) >= 0.9 * len(present):
            return key
    return None


def ordered(rows: List[Dict[str, Any]], forced: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    key = order_key(rows, forced)
    if key is None:
        return list(rows), None
    big = 1 << 62
    return sorted(rows, key=lambda r: (order_value(r, key) if order_value(r, key) is not None else big)), key


def lexical_score(text: str, keywords: Sequence[str]) -> float:
    """Each keyword's occurrences (at most 3) in the folded text, a phrase counting twice, the opening 150
    characters (where a heading sits) twice again."""
    folded = _fold(text)
    head = folded[:150]
    score = 0.0
    for keyword in keywords:
        k = _fold(keyword)
        if not k:
            continue
        hits = min(3, len(re.findall(rf"\b{re.escape(k)}", folded)))
        if hits:
            weight = 2.0 if " " in k else 1.0
            score += weight * hits + (weight * 2 if re.search(rf"\b{re.escape(k)}", head) else 0)
    return score


MAX_CLAUSES = 5
MAX_BASE_CLAUSES = 3
MAX_AMENDMENT_CLAUSES = 3
CLAUSE_CHARS = 4000
RRF_K = 60


def rank_sections(rows: Sequence[Dict[str, Any]], keywords: Sequence[str]) -> List[Dict[str, Any]]:
    """The rows worth a look, best first, each with "_lex", "_rrf": the cosine ranking (rows with a "sim") and
    the lexical one fused by reciprocal rank; a row needs a lexical hit or a top-10 cosine."""
    for r in rows:
        r["_lex"] = lexical_score(r.get("text") or "", keywords)
    by_sim = sorted((r for r in rows if r.get("sim") is not None), key=lambda r: -float(r["sim"]))
    by_lex = sorted((r for r in rows if r["_lex"] > 0), key=lambda r: -r["_lex"])
    sim_rank = {id(r): i for i, r in enumerate(by_sim)}
    lex_rank = {id(r): i for i, r in enumerate(by_lex)}
    out = []
    for r in rows:
        s, lx = sim_rank.get(id(r)), lex_rank.get(id(r))
        if lx is None and (s is None or s >= 10):
            continue
        r["_rrf"] = (1 / (RRF_K + s) if s is not None else 0) + (1 / (RRF_K + lx) if lx is not None else 0)
        out.append(r)
    return sorted(out, key=lambda r: (-r["_rrf"], str(r.get("id"))))


def rebuild_clause(doc_rows: List[Dict[str, Any]], at: int, max_chars: int = CLAUSE_CHARS) -> Dict[str, Any]:
    """The clause around the hit doc_rows[at] (the document's Sections in order): from the section holding its
    heading (the hit itself, else up to 3 before) -- or the one before -- to the sections after it that do not
    open a new article, while under max_chars. {"start", "end", "text", "ref", "title", "hit"}."""
    start = at
    heading_at = None
    for back in range(0, 4):
        i = at - back
        if i < 0:
            break
        if article_headings(doc_rows[i].get("text") or ""):
            heading_at = i
            break
    if heading_at is not None and at - heading_at <= 2:
        start = heading_at
    elif at > 0 and not opens_with_heading(doc_rows[at].get("text") or ""):
        start = at - 1
    end = at
    size = sum(len(doc_rows[i].get("text") or "") for i in range(start, at + 1))
    while end + 1 < len(doc_rows) and end - at < 2:
        nxt = doc_rows[end + 1].get("text") or ""
        if opens_with_heading(nxt) or size + len(nxt) > max_chars:
            break
        end += 1
        size += len(nxt)
    parts = [doc_rows[i].get("text") or "" for i in range(start, end + 1)]
    text = "\n".join(p for p in parts if p.strip())
    # The heading that governs the hit: the last one before the hit's own text, else the first one in it near
    # its start, else one up to 3 sections before.
    hit_offset = sum(len(p) + 1 for p in parts[:at - start])
    heads = article_headings(text)
    before = [h for h in heads if h[0] <= hit_offset + 3]
    ref, title = (before[-1][1], before[-1][2]) if before else (None, "")
    if ref is None and heads and heads[0][0] <= hit_offset + 200:
        ref, title = heads[0][1], heads[0][2]
    if ref is None and heading_at is not None:
        found = article_headings(doc_rows[heading_at].get("text") or "")
        ref, title = found[-1][1], found[-1][2]
    if not title:
        title = _title_from_name(doc_rows[at].get("name") or "")
    return {"start": start, "end": end, "text": text[:max_chars], "ref": ref, "title": title,
            "hit": doc_rows[at].get("id")}


def _title_from_name(name: str) -> str:
    """A Section name usable as a clause's title: words, not a number, an id or an amount."""
    name = (name or "").strip(" -–—,.")
    if len(re.findall(r"[A-Za-zÀ-ÿ]{3,}", name)) >= 2 and not re.search(r"::|\d{1,3}(?:\.\d{3})*,\d{2}", name):
        return name[:90]
    return ""


def clause_reference(clause: Dict[str, Any]) -> str:
    """"Art. 16 - Diritti e rapporti sindacali", "Art. 24", or «title» when the text carries no article."""
    if clause.get("ref"):
        return f"{clause['ref']} - {clause['title']}" if clause.get("title") else clause["ref"]
    if clause.get("title"):
        return f"clausola «{clause['title']}»"
    return "clausola senza titolo"


# Pay tables, as the texts lay them out: "1 € 2.089,63 € 46,14 € 2.135,77 2 € 1.959,16 ...", markdown rows,
# "Livello 3 1.743,75".
_AMOUNT = r"(?:€\s*)?\d{1,3}(?:\.\d{3})*,\d{2}"
_AMOUNT_RE = re.compile(_AMOUNT)
_LEVEL = (r"(?:(?i:livello|liv\.?|categoria|cat\.?)\s*)?(?:Q(?:uadr[oi])?|[1-9](?:°|º|o)?(?:\s?S(?:uper)?)?|"
          r"A[1-3S]?|B[1-3S]?|C[1-3]?(?:\s?Super)?|D[1-3]?(?:\s?Super)?|E[1-3]?|F)")
_ROW_RE = re.compile(rf"(?:(?<=\s)|^|(?<=\|))(?<!€ )\s*({_LEVEL})(?:\s+|\s*\|\s*)((?:{_AMOUNT}\s*\|?\s*)+)")
_DECORRENZA_RE = re.compile(r"(?:dal|decorrenza(?: dal)?|a decorrere dal|al)\s+(\d{1,2}(?:°|º)?[/.\s-]\s*(?:\d{1,2}|"
                            rf"{'|'.join(_MONTHS)})[/.\s-]\s*(?:19|20)\d{{2}})", re.IGNORECASE)


def pay_rows(text: str) -> Dict[str, Any]:
    """{"rows": [(level, [amounts])], "dates": [decorrenze written], "header": the words before the first row}
    -- only what the text writes, never computed."""
    text = clean(text)
    rows: List[Tuple[str, List[str]]] = []
    first = None
    for m in _ROW_RE.finditer(text):
        amounts = [a.replace(" ", "") for a in _AMOUNT_RE.findall(m.group(2))]
        level = " ".join(m.group(1).split())
        if not amounts or re.fullmatch(r"\d{4}", level):
            continue
        if first is None:
            first = m.start()
        rows.append((level, amounts))
    dates = list(dict.fromkeys(" ".join(d.split()) for d in _DECORRENZA_RE.findall(text)))
    header = " ".join(text[:first].split())[-300:] if first else ""
    return {"rows": rows, "dates": dates, "header": header}


def _rows_block(rows: Dict[str, Any]) -> str:
    lines = []
    if rows.get("header"):
        lines.append(f"intestazione: {rows['header']}")
    if rows.get("dates"):
        lines.append(f"decorrenze scritte: {', '.join(rows['dates'])}")
    lines += [f"{level}: {' | '.join(amounts)}" for level, amounts in rows.get("rows") or []]
    return "\n".join(lines)


# --- step 4: the legal framework --------------------------------------------------------------------------

def framework_refs(topics: Sequence[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for name in topics:
        for ref in TOPICS[name]["framework"]:
            if ref not in out:
                out.append(ref)
    return out


def _law_parts(label: str) -> Optional[Tuple[str, str]]:
    m = re.search(r"(\d{1,4})/((?:19|20)\d{2})$", label)
    return (m.group(1), m.group(2)) if m else None


def fetch_framework(session, refs: Sequence[Tuple[str, str]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """([{"d", "s", "_article": (number, label)}] from primary documents only, [refs not found]). A code by its
    document's name (defensive_generation._CODE_PREFIXES), a law by article_codes.law_label of its name."""
    from .article_codes import law_label
    from .article_text import fetch_article
    from .defensive_generation import _CODE_PREFIXES
    prefixes = {**_CODE_PREFIXES, "Cost.": "Costituzione"}
    rows, missing = [], []
    docs_for: Dict[str, List[Dict[str, Any]]] = {}
    for number, label in refs:
        if label not in docs_for:
            if label in prefixes:
                found = session.run(
                    "MATCH (d:Document) WHERE d.document_type = 'primary' AND d.name STARTS WITH $prefix "
                    "RETURN d.id AS id, d.name AS name", prefix=prefixes[label]).data()
                found = [d for d in found if "commentat" not in str(d["name"]).lower()]
            else:
                parts = _law_parts(label)
                found = session.run(
                    "MATCH (d:Document) WHERE d.document_type = 'primary' AND d.name CONTAINS $number "
                    "AND d.name CONTAINS $year RETURN d.id AS id, d.name AS name",
                    number=parts[0], year=parts[1]).data() if parts else []
                found = [d for d in found if law_label(d["name"]) == label]
            docs_for[label] = found
        got = None
        for doc in docs_for[label]:
            hit = fetch_article(session, doc["id"], number)
            if hit:
                got = hit[0]
                break
        if got:
            d = dict(got.get("d") or {})
            d.setdefault("document_type", "primary")
            rows.append({"d": d, "s": dict(got.get("s") or {}), "_article": (number, label),
                         "_source": "ccnl_framework"})
        else:
            missing.append(f"{number} {label}")
    return rows, missing


# --- the answer ---------------------------------------------------------------------------------------------

VERBATIM_WORDS = 10
SYSTEM = (
    "Sei un consulente del lavoro italiano. Rispondi in italiano alla domanda usando SOLO il materiale fornito: "
    "le clausole del CCNL individuato (nella versione indicata), le tabelle estratte, gli articoli di legge e le "
    "pronunce. L'intestazione con il contratto applicato è già scritta: non ripeterla.\n"
    "Scrivi queste sezioni, con questi titoli in grassetto:\n"
    "**Cosa prevede il CCNL** - spiega con parole tue cosa stabiliscono le clausole pertinenti, ciascuna con il "
    "suo riferimento tra parentesi (es. «art. 16 - Diritti e rapporti sindacali», o il titolo della clausola) e "
    "il documento da cui viene. Se un accordo successivo modifica la clausola, indica la regola aggiornata e da "
    "quando vale. NON copiare frasi del contratto: riformula sempre; puoi riportare solo cifre brevi (importi, "
    "giorni, mesi, percentuali). Gli importi solo se scritti nelle tabelle o nel testo fornito, con livello e "
    "decorrenza: mai calcolarli, stimarli o dedurli. Se le clausole non rispondono alla domanda, dillo.\n"
    "**Quadro normativo** - gli articoli di legge forniti, citati con il codice o la legge (es. art. 2109 c.c., "
    "art. 7 l. 300/1970), e come il CCNL vi si inserisce (il contratto collettivo può migliorare, non peggiorare, "
    "le tutele inderogabili). Cita solo gli articoli forniti.\n"
    "**Giurisprudenza** - solo le pronunce fornite che riguardano il punto, in una o due frasi ciascuna; ometti la "
    "sezione se nessuna è pertinente.\n"
    "**Risposta pratica** - la risposta concreta alla domanda.\n"
    "**Da verificare** - un elenco breve: il livello di inquadramento e l'anzianità del lavoratore, eventuali "
    "accordi territoriali o aziendali, ciò che il materiale non chiarisce.")


def _clause_block(i: int, clause: Dict[str, Any], with_rows: bool) -> str:
    doc = clause["doc"]
    head = (f"[CLAUSOLA {i}] {short_name(doc)} ({doc['kind']}, vigenza dal {_fmt(doc.get('from'))}"
            f"{' al ' + _fmt(doc.get('until')) if doc.get('until') else ''}) - riferimento: {clause_reference(clause)}")
    body = clean(clause["text"])
    block = f"{head}\n{body}"
    if with_rows and clause.get("rows", {}).get("rows"):
        block += f"\n[TABELLA ESTRATTA DALLA CLAUSOLA {i}]\n{_rows_block(clause['rows'])}"
    return block


def header(contract: Dict[str, Any], version: Dict[str, Any], when: date, how: str) -> str:
    """The fixed opening: the contract and version used, the amendments, an expired base, the assumption."""
    base = version["base"]
    lines = [f"**Contratto applicato**: CCNL {contract['sector']} - {short_name(base).split(' — ')[-1]} "
             f"(vigenza dal {_fmt(base.get('from'))} al "
             f"{_fmt(base.get('until')) if base.get('until') else 'scadenza non indicata'})."]
    if version["amendments"]:
        lines.append("Integrato da: " + "; ".join(
            f"{short_name(d).split(' — ')[-1]} (dal {_fmt(d.get('from'))})" for d in version["amendments"][-6:]) + ".")
    lines.append(f"Data di riferimento: {_fmt(when)}" + (" (oggi)." if how.startswith("today") else "."))
    if version["expired"]:
        lines.append(f"Nota: la vigenza indicata è scaduta il {_fmt(base.get('until'))}; il contratto continua di "
                     f"regola ad applicarsi fino al rinnovo, ma va verificato se ne è stato firmato uno nuovo.")
    if contract.get("assumption"):
        others = ", ".join(contract.get("alternatives") or []) or "altri CCNL dello stesso settore"
        lines.append(f"**Ipotesi**: dalla domanda non è chiaro quale CCNL applichi il datore di lavoro; ho "
                     f"considerato il più probabile, {contract['sector']}. Esistono anche: {others}. Conferma il "
                     f"CCNL indicato nella lettera di assunzione o in busta paga, così da verificare la risposta.")
    return "\n".join(lines)


_SPLIT_RE = re.compile(r"((?<=[.!?])\s+(?=[A-ZÀ-Ü«\"*(\-])|\n)")


def _sentences(text: str) -> List[str]:
    """The text's sentences, each with the space or newline after it (joined back they give the text); a full
    stop splits only before a capital, so "2.089,63", "art. 16" and "c.c. e" stay whole."""
    parts = _SPLIT_RE.split(text or "")
    return [parts[i] + (parts[i + 1] if i + 1 < len(parts) else "") for i in range(0, len(parts), 2)
            if parts[i] or (i + 1 < len(parts) and parts[i + 1])]


def _norm_amount(a: str) -> str:
    return re.sub(r"[€\s]", "", a)


def check_amounts(answer: str, material: str) -> Tuple[str, List[str]]:
    """(the answer without the sentences stating an amount the material does not contain, those amounts)."""
    known = {_norm_amount(a) for a in _AMOUNT_RE.findall(material or "")}
    removed: List[str] = []
    out = []
    for sentence in _sentences(answer or ""):
        unknown = [a for a in _AMOUNT_RE.findall(sentence) if _norm_amount(a) not in known]
        if unknown:
            removed += unknown
            out.append("\n" if sentence.endswith("\n") else "")
            continue
        out.append(sentence)
    return "".join(out), removed


def _shingles(text: str, n: int) -> set:
    words = _fold(text).split()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def verbatim_sentences(answer: str, sources: Iterable[str], n: int = VERBATIM_WORDS) -> List[str]:
    """The answer's sentences sharing a run of n or more words with the sources."""
    grams = set()
    for source in sources:
        grams |= _shingles(source, n)
    if not grams:
        return []
    return [s.strip() for s in _sentences(answer or "") if s.strip() and _shingles(s, n) & grams]


_REWRITE_SYSTEM = (
    "Riscrivi le frasi indicate dell'analisi senza riprodurre il testo del contratto collettivo: stessa "
    "sostanza, parole tue, stessi riferimenti e cifre. Restituisci l'intera analisi, invariata nel resto.")


def _drop_sentences(answer: str, sentences: Sequence[str]) -> str:
    for sentence in sentences:
        answer = answer.replace(sentence, "")
    return re.sub(r"[ \t]{2,}", " ", answer)


# --- the session ------------------------------------------------------------------------------------------

_SECTOR_CACHE: Dict[str, Any] = {"sectors": None, "ts": 0.0}
_SECTOR_LOCK = threading.Lock()
_SECTOR_TTL = 600


def sector_index(session) -> Dict[str, int]:
    """{sector: its document count} over the CCNL documents, cached for ten minutes."""
    now = time.time()
    with _SECTOR_LOCK:
        if _SECTOR_CACHE["sectors"] is not None and now - _SECTOR_CACHE["ts"] < _SECTOR_TTL:
            return _SECTOR_CACHE["sectors"]
    rows = session.run("MATCH (d:Document) WHERE d.document_type = $t AND d.sector IS NOT NULL "
                       "RETURN d.sector AS sector, count(*) AS n", t=TYPE).data()
    sectors = {r["sector"]: int(r["n"]) for r in rows if str(r.get("sector") or "").strip()}
    with _SECTOR_LOCK:
        _SECTOR_CACHE.update(sectors=sectors, ts=now)
    return sectors


def sector_documents(session, sector: str) -> List[Dict[str, Any]]:
    return session.run(
        "MATCH (d:Document) WHERE d.document_type = $t AND d.sector = $sector "
        "RETURN d.id AS id, d.name AS name, d.document_subtype AS document_subtype, d.valid_from AS valid_from, "
        "d.valid_until AS valid_until, d.sector AS sector", t=TYPE, sector=sector).data()


def document_sections(session, doc_ids: Sequence[str], embedding: Optional[Sequence[float]]) -> List[Dict[str, Any]]:
    """Every Section of the documents with what can order it (its CONTAINS edge's properties, its own scalar
    properties, its id and elementId) and its cosine to the question when an embedding is given."""
    from . import embedding_config
    prop = embedding_config.VECTOR_PROPERTY
    sim = f"vector.similarity.cosine(s.`{prop}`, $embedding)" if embedding is not None else "null"
    return session.run(
        "UNWIND $ids AS doc_id MATCH (d:Document {id: doc_id})-[r:CONTAINS]->(s:Section) "
        "RETURN d.id AS doc, s.id AS id, s.name AS name, s.plain_text AS text, elementId(s) AS element_id, "
        "properties(r) AS rel, "
        "{order: s.order, position: s.position, index: s.index, seq: s.seq, chunk_index: s.chunk_index} AS props, "
        f"CASE WHEN s.`{prop}` IS NULL THEN null ELSE {sim} END AS sim",
        ids=list(doc_ids), embedding=list(embedding) if embedding is not None else None).data()


def asks_ccnl(text: str) -> Dict[str, Any]:
    """The router's decision on the text (no tie-break: a ccnl route never needs one), not logged twice."""
    from . import query_router
    return query_router.classify(text or "", tie_break=False, log=False)


# A worker named by their trade ("faccio il cameriere", "una colf", "sono commessa"): with a CCNL topic, an
# employment question the router's employer-and-worker signal misses.
_TRADE_ROLE_RE = re.compile(
    r"\b(?:lavorat\w*|dipendent\w*|impiegat\w*|operai\w*|apprendist\w*|colf|badant\w*|commess[oaie]|"
    r"camerier[aei]|cuoc[oahi]+|barist[aei]|murator[ei]|carpentier[ei]|segretari[oae]|autist[aei]|"
    r"magazzinier[ei]|cassier[ae]|infermier[ei]|educator[ei]|educatric[ei]|receptionist|portier[ei]|"
    r"guardi[ae] giurat[ae]|addett[oaie] (?:alle?|ai|agli|del|della)\b|bracciant[ei]|praticant[ei])\b", re.I)


def employment_question(text: str) -> bool:
    """A worker or trade role and a CCNL topic in the text (topics_of)."""
    return bool(_TRADE_ROLE_RE.search(text or "")) and bool(topics_of(text))


def should_run(text: str, *, pending_calculation: Any = None, awaiting: bool = False,
               skip_calculation: bool = False) -> Tuple[bool, str]:
    """(run the CCNL path, why): the flag on, no pending turn, no calculation asked (the calculation path keeps
    precedence unless the caller skips it), and the router's route "ccnl" -- or "direct" with a worker's role and
    a CCNL topic (employment_question), which the router's own signals miss; a narrative never."""
    if not enabled():
        return False, "flag off"
    if pending_calculation or awaiting:
        return False, "a pending calculation, clarification or summary"
    from . import query_router
    if not skip_calculation and query_router.asks_calculation(text or ""):
        return False, "a calculation asked: the calculation path keeps precedence"
    route = asks_ccnl(text)
    if route["route"] == "direct" and employment_question(text):
        return True, "route direct, but a worker's role and a CCNL topic (" + ", ".join(topics_of(text)) + ")"
    if route["route"] != "ccnl":
        return False, f"route {route['route']}"
    return True, "; ".join(route.get("reasons") or [])


def _citations(clauses: List[Dict[str, Any]], framework: List[Dict[str, Any]],
               rulings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The answer's sources: each CCNL document with its clauses' references (no CCNL text: it is reserved),
    each law with its articles' text, each ruling with its section."""
    by_doc: Dict[str, Dict[str, Any]] = {}
    for clause in clauses:
        doc = clause["doc"]
        entry = by_doc.setdefault(doc["id"], {"document_name": short_name(doc), "document_id": doc["id"],
                                              "document_type": TYPE, "sections": []})
        entry["sections"].append({"name": clause_reference(clause), "title": clause.get("title") or None,
                                  "plain_text": "", "score": None})
    out = list(by_doc.values())
    for row in framework + rulings:
        d, s = row.get("d") or {}, row.get("s") or {}
        if not d.get("id"):
            continue
        entry = next((c for c in out if c["document_id"] == d["id"]), None)
        if entry is None:
            entry = {"document_name": _doc_name(d),
                     "document_id": d["id"], "document_type": d.get("document_type"), "sections": []}
            out.append(entry)
        entry["sections"].append({"name": s.get("name"), "title": s.get("titolo_articolo") or None,
                                  "plain_text": clean(s.get("plain_text"))[:2000], "score": None})
    return out


def answer(question: str, driver, database: str, today: Optional[date] = None, llm: bool = True
           ) -> Optional[Dict[str, Any]]:
    """The CCNL path's result ({"answer", "citations", "raw_result", "status_messages", "ccnl_sector", "ccnl",
    "route"}), or None when no contract is identified (the chat's graph answers then). Raises on a failure the
    caller falls back from."""
    import neo4j
    from langchain_core.messages import HumanMessage, SystemMessage
    from .ai_chat import _call_chat
    from . import source_tiers
    t0 = time.time()
    info: Dict[str, Any] = {"question": question}
    with driver.session(database=database, default_access_mode=neo4j.READ_ACCESS) as session:
        # 1 the contract
        sectors = sector_index(session)
        contract = identify_contract(question, sectors, llm=llm)
        info["contract"] = contract
        vlog("ccnl_contract", contract)
        if not contract["sector"]:
            info["outcome"] = "no contract identified"
            vlog("ccnl_path", info, duration_ms=(time.time() - t0) * 1000)
            logger.info("ccnl path: no contract identified, the graph answers")
            return None
        # 2 the version
        when, how = relevant_date(question, today)
        docs = sector_documents(session, contract["sector"])
        version = choose_version(docs, when)
        if version["base"] is None:
            info["outcome"] = "no documents for the sector"
            vlog("ccnl_path", info, duration_ms=(time.time() - t0) * 1000)
            return None
        chosen = [version["base"]] + version["amendments"]
        info["version"] = {"date": when.isoformat(), "date_from": how, "reason": version["reason"],
                           "expired": version["expired"],
                           "base": _doc_log(version["base"]), "amendments": [_doc_log(d) for d in version["amendments"]],
                           "dropped": version["dropped"],
                           "sector_documents": [_doc_log(_doc_view(d)) for d in docs]}
        vlog("ccnl_version", info["version"])
        logger.info("ccnl path: %s at %s -> %s + %d amendments (%s)", contract["sector"], when,
                    short_name(version["base"]), len(version["amendments"]), version["reason"])
        # 3 the clauses
        topics = topics_of(question)
        keywords = topic_keywords(question, topics)
        embedding = None
        try:
            from .lookups import _cached_embed_query
            embedding = list(_cached_embed_query(f"{question}\n{' '.join(keywords)}"))
        except Exception as exc:
            logger.warning("ccnl path: no embedding, lexical only: %s", exc)
        rows = document_sections(session, [d["id"] for d in chosen], embedding)
        for r in rows:
            r["text"] = clean(r.get("text"))
        clauses, order_used = select_clauses(rows, chosen, keywords, pay=any(TOPICS[t].get("pay") for t in topics))
        info["clauses"] = {"topics": topics, "keywords": keywords, "sections": len(rows), "order_by": order_used,
                           "found": [{"doc": short_name(c["doc"]), "ref": clause_reference(c),
                                      "sections": c["span"], "rows": len((c.get("rows") or {}).get("rows") or []),
                                      "text": c["text"][:400]} for c in clauses]}
        vlog("ccnl_clauses", info["clauses"])
        # 4 the framework and the rulings
        refs = framework_refs(topics)
        framework, missing = fetch_framework(session, refs) if refs else ([], [])
        code_refs = [(n, label) for n, label in refs if label in ("c.c.", "Cost.")
                     and f"{n} {label}" not in missing]
        rulings = source_tiers.support_rulings(session, code_refs, question, 3, stage="ccnl") if code_refs else []
        info["framework"] = {"wanted": [f"{n} {label}" for n, label in refs], "missing": missing,
                             "found": [f"{r['_article'][0]} {r['_article'][1]}" for r in framework],
                             "rulings": [(r.get("d") or {}).get("name") for r in rulings]}
        vlog("ccnl_framework", info["framework"])
    material_blocks = [_clause_block(i, c, True) for i, c in enumerate(clauses, 1)]
    law_blocks = [f"[LEGGE] art. {r['_article'][0]} {r['_article'][1]}"
                  f"{' - ' + r['s'].get('titolo_articolo') if r['s'].get('titolo_articolo') else ''}\n"
                  f"{clean(r['s'].get('plain_text'))[:1500]}" for r in framework]
    ruling_blocks = [f"[PRONUNCIA] {_doc_name(r.get('d') or {})}\n"
                     f"{clean((r.get('s') or {}).get('plain_text'))[:900]}" for r in rulings]
    head = header(contract, version, when, how)
    human = (f"Domanda: {question}\n\nContratto: CCNL {contract['sector']}; data di riferimento {_fmt(when)}.\n"
             f"Le clausole di accordi successivi prevalgono su quelle del testo base sullo stesso punto.\n\n"
             + ("\n\n".join(material_blocks) or "[nessuna clausola pertinente trovata nel CCNL]")
             + "\n\n" + ("\n\n".join(law_blocks) or "[nessun articolo di legge reperito]")
             + ("\n\n" + "\n\n".join(ruling_blocks) if ruling_blocks else ""))
    body = _call_chat([SystemMessage(content=SYSTEM), HumanMessage(content=human)], max_tokens=1800) if llm else ""
    body = clean(body)
    clause_texts = [c["text"] for c in clauses]
    copied = verbatim_sentences(body, clause_texts)
    rewritten = False
    if copied and llm:
        listed = "\n".join(f"- {s}" for s in copied)
        body = clean(_call_chat([SystemMessage(content=_REWRITE_SYSTEM),
                                 HumanMessage(content=f"FRASI DA RISCRIVERE:\n{listed}\n\nANALISI:\n{body}")],
                                max_tokens=1800))
        rewritten = True
    still = verbatim_sentences(body, clause_texts)
    if still:
        body = _drop_sentences(body, still)
    material = "\n".join(clause_texts + [b for b in law_blocks])
    body, amounts = check_amounts(body, material)
    final = f"{head}\n\n{body.strip()}".strip()
    info["answer"] = {"verbatim_found": len(copied), "rewritten": rewritten, "verbatim_dropped": len(still),
                      "amounts_removed": amounts, "chars": len(final)}
    info["outcome"] = "answered"
    vlog("ccnl_path", {k: v for k, v in info.items() if k != "question"}, duration_ms=(time.time() - t0) * 1000)
    ccnl_rows = [{"d": {"id": c["doc"]["id"], "name": c["doc"].get("name"), "document_type": TYPE,
                        "sector": contract["sector"]},
                  "s": {"id": c["hit"], "name": clause_reference(c), "plain_text": ""}, "_source": "ccnl_clause"}
                 for c in clauses]
    return {"answer": final, "citations": _citations(clauses, framework, rulings),
            "raw_result": ccnl_rows + framework + rulings, "status_messages": [], "ccnl_sector": contract["sector"],
            "ccnl": info, "route": "ccnl"}


def _doc_name(d: Dict[str, Any]) -> str:
    return re.sub(r"^\s*\[[A-Z]+\]\s*", "", str(d.get("name") or d.get("id") or ""))


def _doc_log(d: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": d.get("id"), "name": short_name(d), "kind": d.get("kind"),
            "valid_from": d["from"].isoformat() if d.get("from") else None,
            "valid_until": d["until"].isoformat() if d.get("until") else None}


def select_clauses(rows: List[Dict[str, Any]], chosen: Sequence[Dict[str, Any]], keywords: Sequence[str],
                   pay: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Optional[str]]]:
    """(the clauses, rebuilt from the best hits: at most MAX_BASE_CLAUSES from the base, MAX_AMENDMENT_CLAUSES
    from the amendments -- the latest first on a tie --, MAX_CLAUSES in all, a hit inside a clause already taken
    skipped; {document: the order key used}). On a pay question each clause gets its table rows."""
    forced = os.getenv(ORDER_ENV) or None
    by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_doc.setdefault(r["doc"], []).append(r)
    order_used: Dict[str, Optional[str]] = {}
    position: Dict[Any, Tuple[str, int]] = {}
    for doc_id, doc_rows in by_doc.items():
        by_doc[doc_id], order_used[doc_id] = ordered(doc_rows, forced)
        for i, r in enumerate(by_doc[doc_id]):
            position[id(r)] = (doc_id, i)
    docs = {d["id"]: d for d in chosen}
    base_id = chosen[0]["id"] if chosen else None
    rank = {d["id"]: i for i, d in enumerate(chosen)}
    hits = rank_sections(rows, keywords)
    hits.sort(key=lambda r: (-round(r["_rrf"], 6), -rank.get(r["doc"], 0), str(r.get("id"))))
    clauses: List[Dict[str, Any]] = []
    taken: Dict[str, List[Tuple[int, int]]] = {}
    counts = {"base": 0, "amendments": 0}
    for r in hits:
        if len(clauses) >= MAX_CLAUSES:
            break
        doc_id, i = position[id(r)]
        side = "base" if doc_id == base_id else "amendments"
        if counts[side] >= (MAX_BASE_CLAUSES if side == "base" else MAX_AMENDMENT_CLAUSES):
            continue
        if any(a <= i <= b for a, b in taken.get(doc_id, [])):
            continue
        clause = rebuild_clause(by_doc[doc_id], i)
        taken.setdefault(doc_id, []).append((clause["start"], clause["end"]))
        clause["doc"] = docs.get(doc_id) or {"id": doc_id, "name": doc_id, "kind": OTHER}
        clause["span"] = [by_doc[doc_id][k].get("name") for k in range(clause["start"], clause["end"] + 1)]
        if pay:
            clause["rows"] = pay_rows(clause["text"])
        counts[side] += 1
        clauses.append(clause)
    clauses.sort(key=lambda c: (rank.get(c["doc"]["id"], 0), c["start"]))
    return clauses, {short_name(docs.get(k) or {"id": k}): v for k, v in order_used.items()}
