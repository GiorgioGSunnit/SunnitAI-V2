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
# matched on folded text (_fold). Tourism is split by activity: its sectors are named "Turismo <association> -
# <activity>" (pubblici esercizi, aziende alberghiere, campeggi, agenzie di viaggi, stabilimenti balneari).
CONCEPTS: List[Tuple[str, str, str]] = [
    ("metalmeccanici", r"metalmeccanic\w*|officina meccanica|carpenteria metallica|siderurgic\w*|fonderi\w*",
     r"metalmeccanic"),
    ("commercio", r"commerci(?:o|ale)?\b|terziario|negozi\w*|commess[oaie]|supermercat\w*|ipermercat\w*|"
                  r"grande distribuzione|vendita al dettaglio|cassier[ea]", r"\bterziario|\bcommercio\b|\bdistribuzione"),
    ("turismo", r"turism\w*|turistic\w*", r"turismo"),
    ("pubblici esercizi", r"pubblici esercizi|ristorant\w*|\bbar\b|camerier\w*|\bcuoc[oa]\b|barista|pizzeri\w*|"
                          r"trattori\w*|\bpub\b|somministrazione di (?:alimenti|bevande)", r"pubblici esercizi|\bp esercizi"),
    ("alberghi", r"alberg\w*|hotel|\bresort\b", r"\balberg\w*\b(?! diurn)"),     # "Alberghi diurni": another trade
    ("campeggi", r"campeggi\w*|villaggi\w* turistic\w*", r"campegg"),
    ("agenzie di viaggi", r"agenzi\w* (?:di )?viagg\w*|tour operator", r"agenzie (?:di )?viagg|ag viaggi|imprese di viaggi"),
    ("stabilimenti balneari", r"stabiliment\w* balnear\w*|\blido\b|bagnin\w*", r"balnear"),
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
}
# The employers' associations a sector's name may carry outside brackets ("Commercio - Confcommercio",
# "Turismo Confesercenti - Alberghi"); the ones in brackets are read from the brackets.
_ASSOCIATIONS_RE = re.compile(r"\b(?:confcommercio|confesercenti|confindustria|federdistribuzione|confapi|"
                              r"confartigianato|casartigiani|cna|legacoop|confcooperative|federterziario)\b")
# Categories of workers with a contract of their own: a sector naming one fits only a question that does.
SPECIAL_CATEGORIES = [("dirigenti", r"\bdirigent\w*"),
                      ("agenti e rappresentanti", r"\bagenti e rappresentant\w*|\bagent\w* di commercio|"
                                                  r"\brappresentant\w* di commercio"),
                      ("personale artistico", r"\bpersonale artistic\w*|\bartist\w*")]
# A size or a scope inside one contract ("minori", "fino a 15 dip."): never the first choice the text does not ask.
_SIZE_RE = re.compile(r"\bminori\b|\bmaggiori\b|\bfino a \d+|\boltre \d+|\bda \d+ (?:a \d+ )?dip")
_STOP = {"del", "della", "dello", "dei", "delle", "degli", "di", "da", "dal", "dalla", "per", "con", "e", "ed",
         "il", "la", "lo", "le", "gli", "i", "un", "una", "uno", "in", "su", "tra", "fra", "a", "al", "alla", "ai",
         "ccnl", "contratto", "collettivo", "nazionale", "lavoro", "settore", "personale", "dipendenti", "addetti",
         "aziende", "imprese"}
# A sector's name ends with a date qualifier ("(dal 01.07.04)") that is no part of its words.
_DATE_QUALIFIER_RE = re.compile(r"\(\s*dal\s+[\d.\s/]+\)", re.I)
_BRACKETS_RE = re.compile(r"\(([^)]*)\)")
MIN_STEM = 5


def _stem(word: str) -> str:
    return word.rstrip("aeiou") or word


def _words(text: str) -> List[str]:
    return [w for w in _fold(text).split() if w not in _STOP and len(w) >= 4 and not w.isdigit()]


def _written(word: str, words: set, stems: set) -> bool:
    """A sector's word written in the text: by its stem when that has MIN_STEM letters or more, else as the word
    itself ("posto" is not "Poste")."""
    stem = _stem(word)
    return stem in stems if len(stem) >= MIN_STEM else word in words


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


def signatory(sector: str) -> Tuple[str, bool]:
    """(the folded words of who signed the sector's contract -- its brackets, except a date, and any employers'
    association its name carries --, whether they come from brackets: a minor signatory)."""
    name = _DATE_QUALIFIER_RE.sub(" ", sector or "")
    bracketed = " ".join(_fold(m) for m in _BRACKETS_RE.findall(name))
    named = " ".join(_ASSOCIATIONS_RE.findall(_fold(_BRACKETS_RE.sub(" ", name))))
    return " ".join(x for x in (bracketed, named) if x), bool(bracketed.strip())


def score_sectors(text: str, sectors: Dict[str, int]) -> List[Dict[str, Any]]:
    """Every sector the text points to, best first: {"sector", "score", "why", "variants", "docs"}. The score:
    the whole name written in the text +10; a common name (CONCEPTS) of its family +3; every word of its family
    part written +3; each other word of its name written +1; per variant group, the variant the text names +2,
    another variant of that group -2; a signatory the text names +2, another signatory -2 (when the text names
    one); a special category of workers (dirigenti ...) the text does not ask -3."""
    folded = _fold(text)
    words = set(_words(text)) | set(folded.split())
    stems = {_stem(w) for w in _words(text)}
    asked = {group: _variant(folded, group) for group in VARIANT_GROUPS}
    concepts = [(label, side) for label, phrase, side in CONCEPTS if re.search(phrase, folded)]
    signatories = {sector: signatory(sector) for sector in sectors}
    vocabulary = {w for sig, _ in signatories.values() for w in sig.split() if len(w) >= 3 and w not in _STOP}
    asked_signatory = {w for w in folded.split() if w in vocabulary}
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
        head = _words(sector_head(sector))
        if head and all(_written(w, words, stems) for w in head):
            score += 3
            why.append("family words")
        rest = _words(sector)[len(head):]
        extra = [w for w in rest if _written(w, words, stems)]
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
        sig, minor = signatories[sector]
        if asked_signatory and sig:
            if set(sig.split()) & asked_signatory:
                score += 2
                why.append(f"signatory {sig}")
            else:
                score -= 2
                why.append(f"other signatory {sig} (asked {', '.join(sorted(asked_signatory))})")
        for label, pattern in SPECIAL_CATEGORIES:
            if re.search(pattern, name) and not re.search(pattern, folded):
                score -= 3
                why.append(f"{label} not asked")
        variants["signatory"] = sig if minor else None
        variants["size"] = bool(_SIZE_RE.search(name))
        out.append({"sector": sector, "score": score, "why": why, "variants": variants, "docs": docs,
                    "unmatched": len([w for w in _words(sector) if not _written(w, words, stems)])})
    out.sort(key=lambda c: (-c["score"], -_preference(c), c["unmatched"], -c["docs"], c["sector"]))
    return out


def _preference(candidate: Dict[str, Any]) -> int:
    """Between sectors the text does not tell apart: no minor (bracketed) signatory first, then no size or
    scope ("minori", "fino a 15 dip."), then the industry's own form, then a name with no form."""
    variants = candidate.get("variants") or {}
    form = variants.get("forma")
    rank = 2 if form == "industria" else 1 if form is None else 0
    return (0 if variants.get("signatory") else 8) + (0 if variants.get("size") else 4) + rank


def identify_contract(text: str, sectors: Dict[str, int], llm: bool = True) -> Dict[str, Any]:
    """{"sector", "how", "ambiguous", "alternatives", "candidates", "assumption"}; sector None when nothing
    matches. Ambiguous: two or more sectors share the top score -- the text did not settle the variant; the
    first by _preference is used and the others are the alternatives to confirm (settle_by_documents then drops
    those filed with the very same documents)."""
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
            info["alternatives"] = [c["sector"] for c in tied[1:9]]
    elif llm and sectors:
        from .calculation import _classify_sector_llm
        picked = _classify_sector_llm(text, sorted(sectors))
        if picked:
            info.update(sector=picked, how="llm pick from the sector list", ambiguous=True)
    info["assumption"] = bool(info["sector"] and info["ambiguous"])
    return info


def documents_signature(docs: Iterable[Dict[str, Any]]) -> frozenset:
    """A sector's documents as (kind, valid_from, valid_until): two sectors with the same signature hold one
    contract filed twice ("Pubblici esercizi" and "Pubblici esercizi minori")."""
    return frozenset((doc_kind(d), str(to_date(d.get("valid_from"))), str(to_date(d.get("valid_until"))))
                     for d in docs)


def settle_by_documents(contract: Dict[str, Any], chosen_docs: Sequence[Dict[str, Any]],
                        docs_of) -> Dict[str, Any]:
    """The contract without the alternatives filed with the chosen sector's very documents, or with an earlier
    alternative's (docs_of(sector) -> its documents); no alternative left, no ambiguity. At most 4 kept."""
    if not contract.get("ambiguous") or not contract.get("alternatives") or contract.get("how", "").startswith("llm"):
        return contract
    seen = {documents_signature(chosen_docs)}
    kept, same = [], []
    for sector in contract["alternatives"]:
        if len(kept) >= 4:
            break
        signature = documents_signature(docs_of(sector))
        (same if signature in seen else kept).append(sector)
        seen.add(signature)
    out = {**contract, "alternatives": kept, "same_documents": same}
    if not kept:
        out.update(ambiguous=False, assumption=False)
    return out


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

# Topic -> the question's words ("re"), the words a clause on it uses ("keywords": specific ones only -- a
# generic "durata" or "mesi" matches half a contract), the phrases its heading carries ("headings": "Art. 23 -
# Periodo di prova", or a Section named after its heading's tail, "di prova"), the primary labour law around it
# ("framework": article, "c.c." or the law's label as article_codes.law_label writes it).
TOPICS: Dict[str, Dict[str, Any]] = {
    "retribuzione": {"re": r"retribu\w*|stipend\w*|\bpaga\b|minim[io] (?:tabellar|contrattual|retributiv)\w*|"
                           r"tabell\w* retributiv\w*|busta paga|quanto (?:prende|guadagna|spetta)|lordo|"
                           r"superminimo|aument\w* (?:retributiv|contrattual)\w*",
                     "keywords": ["minimi tabellari", "retribuzione tabellare", "minimi retributivi", "minimi",
                                  "tabellare", "paga base", "minimo contrattuale", "conglobato", "stipendio base",
                                  "tabella retributiva", "importi mensili"],
                     "headings": ["minimi tabellari", "minimi retributivi", "minimi contrattuali", "tabelle retributive",
                                  "tabella retributiva", "retribuzione", "trattamento economico", "aumenti retributivi"],
                     "pay": True, "framework": [("2099", "c.c."), ("36", "Cost.")]},
    "ferie": {"re": r"\bferie\b",
              "keywords": ["ferie", "periodo feriale", "periodo di ferie", "ferie annuali", "godimento delle ferie",
                           "fruizione delle ferie"],
              "headings": ["ferie"],
              "framework": [("2109", "c.c."), ("10", "d.lgs. 66/2003")]},
    "preavviso": {"re": r"preavviso|dimission\w*|dimett\w*",
                  "keywords": ["preavviso", "termini di preavviso", "periodo di preavviso", "preavviso di licenziamento",
                               "preavviso di dimissioni", "indennità sostitutiva del preavviso", "indennità di mancato "
                               "preavviso", "dimissioni"],
                  "headings": ["preavviso", "termini di preavviso", "preavviso di licenziamento e dimissioni",
                               "risoluzione del rapporto", "dimissioni"],
                  "framework": [("2118", "c.c."), ("2119", "c.c.")]},
    "prova": {"re": r"periodo di prova|\bin prova\b|patto di prova|\bla prova\b",
              "keywords": ["periodo di prova", "patto di prova", "assunzione in prova", "durata della prova",
                           "superamento della prova", "in prova", "conferma in servizio"],
              "headings": ["periodo di prova", "assunzione in prova", "patto di prova", "prova"],
              "framework": [("2096", "c.c.")]},
    "malattia": {"re": r"malatti\w*|comporto|infortuni\w*",
                 "keywords": ["malattia", "comporto", "periodo di comporto", "conservazione del posto",
                              "certificato medico", "infortunio", "assenze per malattia"],
                 "headings": ["malattia", "malattia e infortunio", "trattamento di malattia", "comporto",
                              "infortunio"],
                 "framework": [("2110", "c.c.")]},
    "disciplinare": {"re": r"disciplinar\w*|contestazion\w*|sanzion\w*|richiamo|\bmulta\b|sospension\w* dal "
                           r"servizio|provvediment\w*",
                     "keywords": ["provvedimenti disciplinari", "sanzioni disciplinari", "codice disciplinare",
                                  "norme disciplinari", "procedura disciplinare", "rimprovero verbale",
                                  "rimprovero scritto", "ammonizione", "multa", "sospensione dal lavoro",
                                  "sospensione dal servizio", "licenziamento disciplinare", "contestazione",
                                  "giustificazioni", "recidiva", "disciplinari"],
                     "headings": ["provvedimenti disciplinari", "sanzioni disciplinari", "codice disciplinare",
                                  "norme disciplinari", "procedura disciplinare", "provvedimenti e sanzioni"],
                     "framework": [("7", "l. 300/1970"), ("2106", "c.c."), ("2104", "c.c.")]},
    "licenziamento": {"re": r"licenzi\w*",
                      "keywords": ["licenziamento", "licenziamenti", "giusta causa", "giustificato motivo", "recesso"],
                      "headings": ["licenziamento", "licenziamenti", "risoluzione del rapporto", "recesso"],
                      "framework": [("2118", "c.c."), ("2119", "c.c."), ("1", "l. 604/1966"), ("3", "l. 604/1966"),
                                    ("3", "d.lgs. 23/2015")]},
    "orario": {"re": r"orario|straordinari\w*|lavoro notturno|\bturni\b|riposo settimanale",
               "keywords": ["orario di lavoro", "orario normale", "lavoro straordinario", "straordinario",
                            "maggiorazione", "ore settimanali", "lavoro notturno"],
               "headings": ["orario di lavoro", "orario", "lavoro straordinario", "straordinario", "lavoro notturno"],
               "framework": [("3", "d.lgs. 66/2003"), ("5", "d.lgs. 66/2003")]},
    "tempo determinato": {"re": r"tempo determinato|contratto a termine|\bproroga\b",
                          "keywords": ["tempo determinato", "contratto a termine", "proroga", "durata massima"],
                          "headings": ["contratto a tempo determinato", "tempo determinato", "contratto a termine"],
                          "framework": [("19", "d.lgs. 81/2015"), ("21", "d.lgs. 81/2015")]},
    "part-time": {"re": r"part[- ]?time|tempo parziale",
                  "keywords": ["tempo parziale", "part-time", "clausole elastiche", "lavoro supplementare"],
                  "headings": ["tempo parziale", "part-time", "part time"],
                  "framework": [("4", "d.lgs. 81/2015"), ("6", "d.lgs. 81/2015")]},
    "apprendistato": {"re": r"apprendist\w*",
                      "keywords": ["apprendistato", "apprendista", "piano formativo", "apprendistato professionalizzante"],
                      "headings": ["apprendistato"],
                      "framework": [("41", "d.lgs. 81/2015"), ("42", "d.lgs. 81/2015")]},
    "inquadramento": {"re": r"mansion\w*|inquadrament\w*|livell\w* di inquadramento|qualific\w*|declaratori\w*",
                      "keywords": ["inquadramento", "classificazione", "mansioni", "declaratoria", "profili professionali"],
                      "headings": ["classificazione del personale", "inquadramento", "classificazione", "mansioni"],
                      "framework": [("2103", "c.c."), ("2095", "c.c.")]},
    "tfr": {"re": r"\btfr\b|trattamento di fine rapporto|liquidazione",
            "keywords": ["trattamento di fine rapporto", "tfr", "anticipazione"],
            "headings": ["trattamento di fine rapporto"], "framework": [("2120", "c.c.")]},
    "mensilita": {"re": r"tredicesim\w*|quattordicesim\w*|mensilit\w* aggiuntiv\w*",
                  "keywords": ["tredicesima", "quattordicesima", "mensilità aggiuntiva", "gratifica natalizia"],
                  "headings": ["tredicesima", "quattordicesima", "mensilità"],
                  "framework": [("2099", "c.c.")]},
    "permessi": {"re": r"permess\w*|\brol\b|ex festivit\w*",
                 "keywords": ["permessi retribuiti", "riduzione dell'orario", "rol", "ex festività", "permessi"],
                 "headings": ["permessi", "riduzione dell'orario", "permessi retribuiti"],
                 "framework": []},
}


def topic_headings(topics: Sequence[str]) -> List[str]:
    out: List[str] = []
    for name in topics:
        out += [h for h in TOPICS[name].get("headings") or [] if h not in out]
    return out


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


# A regime with rules of its own: a heading inside one (the apprentices' "Art. 7 - Periodo di prova", a
# fixed-term title's) answers only a question about it.
SPECIAL_CONTEXTS = [r"apprendist\w*", r"tempo determinato|contratt\w* a termine", r"dirigent\w*",
                    r"somministra\w*", r"tempo parziale|part[- ]?time", r"telelavor\w*|lavoro agile"]
# Words a heading's tail must hold more than ("del rapporto" is the tail of many headings).
_GENERIC_TAIL_WORDS = {"rapporto", "lavoro", "contratto", "personale", "trattamento", "disciplina", "periodo",
                       "durata", "norme", "lavoratori", "lavoratore", "generale", "generali"}


def _special(context: str, question: str) -> bool:
    folded_q = _fold(question)
    return any(re.search(p, context) and not re.search(p, folded_q) for p in SPECIAL_CONTEXTS)


def heading_match(row: Dict[str, Any], phrases: Sequence[str], question: str = "") -> Dict[str, Any]:
    """How surely the Section holds a clause on the topic, and where it starts: {"score", "via", "anchor",
    "ref", "title"}. Score 4: an "Art. N - <title>" in its text whose title is a phrase (via "heading", anchor its
    offset), or a name that is a phrase or its tail of two words or more holding a specific word ("di prova" of
    "periodo di prova": the heading cut by the ingestion; via "name"); 3: a heading title holding a phrase; 2: a
    name holding one; 1: a text opening with one; 0: none, an empty text, a table of contents (four or more
    headings with little text between). A match inside a special regime the question does not ask about (the
    apprentices', a fixed-term title) loses 3."""
    text = row.get("text") or ""
    none = {"score": 0, "via": None, "anchor": None, "ref": None, "title": None}
    folded_phrases = [_fold(p) for p in phrases if _fold(p)]
    if not folded_phrases or not text.strip():
        return none
    heads = article_headings(text)
    if len(heads) >= 4 and len(text) / len(heads) < 160:
        return none
    best = dict(none)
    for pos, ref, title in heads:
        t = _fold(title)
        if not t:
            continue
        if any(t == p or (t.startswith(p) and len(t) <= len(p) + 4) for p in folded_phrases):
            score = 4
        elif any(re.search(rf"\b{re.escape(p)}\b", t) for p in folded_phrases):
            score = 3
        else:
            continue
        if _special(_fold(text[max(0, pos - 200):pos + 300]), question):
            score = max(0, score - 3)
        if score > best["score"]:
            best = {"score": score, "via": "heading", "anchor": pos, "ref": ref, "title": title}
    name = _fold(row.get("name") or "")
    if name:
        tail = (len(name) >= 5 and " " in name and set(name.split()) - _GENERIC_TAIL_WORDS - _STOP
                and any(p.endswith(" " + name) for p in folded_phrases))
        if name in folded_phrases or tail:
            score = 4
        elif any(re.search(rf"\b{re.escape(p)}\b", name) for p in folded_phrases):
            score = 2
        else:
            score = 0
        if score and _special(_fold(name + " " + text[:300]), question):
            score = max(0, score - 3)
        if score > best["score"]:
            best = {"score": score, "via": "name", "anchor": None, "ref": None, "title": None}
    if not best["score"]:
        opening = _fold(text[:80])
        if any(opening.startswith(p) for p in folded_phrases):
            best = {"score": 1, "via": "opening", "anchor": None, "ref": None, "title": None}
    return best


def heading_score(row: Dict[str, Any], phrases: Sequence[str], question: str = "") -> int:
    return heading_match(row, phrases, question)["score"]


def rank_sections(rows: Sequence[Dict[str, Any]], keywords: Sequence[str],
                  headings: Sequence[str] = (), question: str = "") -> List[Dict[str, Any]]:
    """The rows worth a look, best first, each with "_lex", "_head", "_rrf": a Section whose heading names the
    topic (heading_score) first, then the cosine ranking (rows with a "sim") and the lexical one fused by
    reciprocal rank; a row needs a heading match, a lexical hit or a top-10 cosine."""
    for r in rows:
        r["_lex"] = lexical_score(r.get("text") or "", keywords)
        match = heading_match(r, headings, question)
        r["_head"], r["_via"], r["_anchor"] = match["score"], match["via"], match["anchor"]
    by_sim = sorted((r for r in rows if r.get("sim") is not None), key=lambda r: -float(r["sim"]))
    by_lex = sorted((r for r in rows if r["_lex"] > 0), key=lambda r: -r["_lex"])
    sim_rank = {id(r): i for i, r in enumerate(by_sim)}
    lex_rank = {id(r): i for i, r in enumerate(by_lex)}
    out = []
    for r in rows:
        s, lx = sim_rank.get(id(r)), lex_rank.get(id(r))
        if lx is None and (s is None or s >= 10) and not r["_head"]:
            continue
        r["_rrf"] = (1 / (RRF_K + s) if s is not None else 0) + (1 / (RRF_K + lx) if lx is not None else 0)
        out.append(r)
    return sorted(out, key=lambda r: (-r["_head"], -r["_rrf"], str(r.get("id"))))


HEADING_ONLY_CHARS = 150


def _body_chars(text: str) -> int:
    """The characters of a Section's text after its last article heading and that heading's title."""
    heads = article_headings(text or "")
    if not heads:
        return len((text or "").strip())
    pos, ref, title = heads[-1]
    return max(0, len((text or "")[pos:].strip()) - len(ref) - len(title) - 4)


def _extend(doc_rows: List[Dict[str, Any]], at: int, size: int, after: int, max_chars: int) -> Tuple[List[str], int]:
    """The sections after doc_rows[at] that continue its clause: up to `after` of them, while under max_chars,
    stopping at one that opens a new article and cutting one at the first heading it holds."""
    parts, end = [], at
    while end + 1 < len(doc_rows) and end - at < after and size < max_chars:
        nxt = doc_rows[end + 1].get("text") or ""
        heads = article_headings(nxt)
        cut = heads[0][0] if heads else None
        if cut is not None and cut <= 3:
            break
        piece = nxt[:cut] if cut is not None else nxt
        parts.append(piece)
        end += 1
        size += len(piece)
        if cut is not None:
            break
    return parts, end


def rebuild_clause(doc_rows: List[Dict[str, Any]], at: int, max_chars: int = CLAUSE_CHARS,
                   anchor: Optional[int] = None, name_headed: bool = False) -> Dict[str, Any]:
    """The clause around the hit doc_rows[at] (the document's Sections in order), {"start", "end", "text", "ref",
    "title", "hit"}:
      anchor       (a heading matched at that offset of the hit's text) from that heading to the next one, in
                   the hit or in the sections after it -- a Section often holds several articles;
      name_headed  (the hit is named after its heading's tail, "di prova") from the hit, its article number
                   only from a heading ending the section before ("... Art. 109 -"), its title that heading's
                   title and the name;
      otherwise    from the section holding its heading (the hit itself, else up to 3 before) -- or the one
                   before -- to the sections after it that do not open a new article.
    5 sections after a heading-only hit, else 2."""
    hit_text = doc_rows[at].get("text") or ""
    after = 5 if _body_chars(hit_text) < HEADING_ONLY_CHARS else 2
    if anchor is not None and 0 <= anchor < len(hit_text):
        segment = hit_text[anchor:]
        heads = article_headings(segment)
        ref, title = (heads[0][1], heads[0][2]) if heads else (None, "")
        later = [h for h in heads if h[0] > 5]
        if later:
            text, end = segment[:later[0][0]], at
        else:
            parts, end = _extend(doc_rows, at, len(segment), 5 if len(segment) < 600 else after, max_chars)
            text = "\n".join([segment] + [p for p in parts if p.strip()])
        return {"start": at, "end": end, "text": text.strip()[:max_chars], "ref": ref,
                "title": title or _title_from_name(doc_rows[at].get("name") or ""), "hit": doc_rows[at].get("id")}
    if name_headed:
        ref, head_title = None, ""
        if at > 0:
            prev = doc_rows[at - 1].get("text") or ""
            heads = article_headings(prev)
            if heads and len(prev[heads[-1][0]:].strip()) <= len(heads[-1][1]) + len(heads[-1][2]) + 6:
                ref, head_title = heads[-1][1], heads[-1][2]          # the heading ends the section
        name = (doc_rows[at].get("name") or "").strip(" -–—,.")
        title = f"{head_title} {name}".strip() if head_title else (name[:1].upper() + name[1:])[:90]
        parts, end = _extend(doc_rows, at, len(hit_text), after, max_chars)
        text = "\n".join([hit_text] + [p for p in parts if p.strip()])
        return {"start": at, "end": end, "text": text.strip()[:max_chars], "ref": ref, "title": title,
                "hit": doc_rows[at].get("id")}
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
    elif at > 0 and not opens_with_heading(hit_text):
        start = at - 1
    size = sum(len(doc_rows[i].get("text") or "") for i in range(start, at + 1))
    more, end = _extend(doc_rows, at, size, after, max_chars)
    parts = [doc_rows[i].get("text") or "" for i in range(start, at + 1)] + more
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
_LEVEL = (r"(?i:operai[oe]?\s+(?:specializzat[oi]|qualificat[oi]|comun[ei])(?:\s+super)?|"
          r"impiegat[oi]\s+(?:di\s+)?[1-9](?:°|º|ª|a)?\s*(?:categoria|livello)(?:\s+super)?|"
          r"[1-9](?:°|º|ª)?\s*(?:livello|categoria|liv\.)|quadr[oi]|apprendist[ai])|"
          r"(?:(?i:livello|liv\.?|categoria|cat\.?)\s*)?(?:Q(?:uadr[oi])?|[1-9](?:°|º|ª|o)?(?:\s?S(?:uper)?)?|"
          r"A[1-3S]?|B[1-3S]?|C[1-3]?(?:\s?Super)?|D[1-3]?(?:\s?Super)?|E[1-3]?|F)")
_ROW_RE = re.compile(rf"(?:(?<=\s)|^|(?<=\|))(?<!€ )\s*({_LEVEL})(?:\s+|\s*\|\s*)((?:{_AMOUNT}\s*\|?\s*)+)")
_DECORRENZA_RE = re.compile(r"(?:dal|decorrenza(?: dal)?|a decorrere dal)\s+(\d{1,2}(?:°|º)?[/.\s-]\s*(?:\d{1,2}|"
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


# The standard labour norms the corpus may lack (no primary document for d.lgs. 66/2003 or l. 300/1970 on 28 Oct):
# cited by reference with their rubrica, never with a text.
FRAMEWORK_REFERENCES = {
    ("7", "l. 300/1970"): "Statuto dei lavoratori - Sanzioni disciplinari",
    ("18", "l. 300/1970"): "Statuto dei lavoratori - Tutela del lavoratore in caso di licenziamento illegittimo",
    ("10", "d.lgs. 66/2003"): "Ferie annuali",
    ("3", "d.lgs. 66/2003"): "Orario normale di lavoro",
    ("5", "d.lgs. 66/2003"): "Lavoro straordinario",
    ("3", "d.lgs. 23/2015"): "Licenziamento per giustificato motivo e giusta causa",
    ("19", "d.lgs. 81/2015"): "Apposizione del termine e durata massima",
    ("21", "d.lgs. 81/2015"): "Proroghe e rinnovi",
    ("42", "d.lgs. 81/2015"): "Disciplina generale dell'apprendistato",
}


# A law some corpora name by its title, not its number: looked up by these words when the number finds none.
LAW_ALIASES = {"l. 300/1970": ["statuto dei lavoratori"], "d.lgs. 23/2015": ["tutele crescenti"],
               "d.lgs. 81/2015": ["disciplina organica dei contratti di lavoro"],
               "d.lgs. 66/2003": ["organizzazione dell'orario di lavoro"]}


def fetch_framework(session, refs: Sequence[Tuple[str, str]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """([{"d", "s", "_article": (number, label)}] from primary documents only, [refs not found]). A code by its
    document's name (defensive_generation._CODE_PREFIXES), a law by article_codes.law_label of its name, else by
    its title (LAW_ALIASES)."""
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
                if not found and LAW_ALIASES.get(label):
                    found = session.run(
                        "MATCH (d:Document) WHERE d.document_type = 'primary' AND "
                        "any(a IN $aliases WHERE toLower(d.name) CONTAINS a) RETURN d.id AS id, d.name AS name",
                        aliases=LAW_ALIASES[label]).data()
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
_SYSTEM_HEAD = (
    "Sei un consulente del lavoro italiano. Rispondi in italiano alla domanda usando SOLO il materiale fornito: "
    "le clausole del CCNL individuato (nella versione indicata), le tabelle, gli articoli di legge e le pronunce. "
    "L'intestazione con il contratto applicato (e la fonte degli importi) è già scritta: non ripeterla.\n"
    "Scrivi queste sezioni, con questi titoli in grassetto:\n"
    "**Cosa prevede il CCNL** - spiega con parole tue cosa stabiliscono le clausole pertinenti, ciascuna con il "
    "suo riferimento tra parentesi (es. «art. 16 - Diritti e rapporti sindacali», o il titolo della clausola) e "
    "il documento da cui viene. Se un accordo successivo modifica la clausola, indica la regola aggiornata e da "
    "quando vale. NON copiare frasi del contratto: riformula sempre; puoi riportare solo cifre brevi (importi, "
    "giorni, mesi, percentuali). Se le clausole non rispondono alla domanda, dillo.\n")
_SYSTEM_PAY = (
    "Importi: usa SOLO le righe delle TABELLE IMPORTI (sono quelle in vigore alla data di riferimento), con livello "
    "e decorrenza; mai calcolarli, stimarli o dedurli, mai prenderli da altre clausole. Se non ci sono TABELLE "
    "IMPORTI, non indicare importi.\n")
_SYSTEM_NO_PAY = "Importi solo se scritti nel materiale, con livello e decorrenza: mai calcolarli, stimarli o dedurli.\n"
_SYSTEM_LAW = (
    "**Quadro normativo** - ciascun articolo di legge fornito che riguarda la domanda, citato con il codice o la "
    "legge (es. art. 2109 c.c., art. 7 l. 300/1970), e come il CCNL vi si inserisce (il contratto collettivo può "
    "migliorare, non peggiorare, le tutele inderogabili). Cita solo gli articoli forniti; quelli segnati «testo "
    "non disponibile» citali come norma di riferimento con la loro rubrica, senza attribuire loro contenuti che il "
    "materiale non riporta.\n")
_SYSTEM_RULINGS = (
    "**Giurisprudenza** - solo le pronunce fornite che decidono il punto della domanda, in una o due frasi "
    "ciascuna. Se nessuna lo decide, NON scrivere questa sezione e non dire che le pronunce non sono pertinenti.\n")
_SYSTEM_TAIL = (
    "**Risposta pratica** - la risposta concreta alla domanda.\n"
    "**Da verificare** - un elenco breve: il livello di inquadramento e l'anzianità del lavoratore, eventuali "
    "accordi territoriali o aziendali, ciò che il materiale non chiarisce.")


def system_prompt(pay: bool, rulings: bool) -> str:
    """The writer's instructions: the amounts rule of a pay question, the case-law section only with rulings."""
    return (_SYSTEM_HEAD + (_SYSTEM_PAY if pay else _SYSTEM_NO_PAY) + _SYSTEM_LAW
            + (_SYSTEM_RULINGS if rulings else "") + _SYSTEM_TAIL)


SYSTEM = system_prompt(False, True)
_CASELAW_HEAD_RE = re.compile(r"^\s*\*\*\s*Giurisprudenza\s*\*\*.*$", re.I | re.M)
_NEXT_HEAD_RE = re.compile(r"^\s*\*\*[^*\n]{3,60}\*\*", re.M)
_NONE_RELEVANT_RE = re.compile(r"\b(?:non|nessun\w*)\b[^.\n]{0,90}\b(?:pertinent\w*|rilevant\w*|riguard\w*|"
                               r"attinent\w*|specific\w*|applicabil\w*|decid\w*|affront\w*|tratt\w*)", re.I)
_RULING_REF_RE = re.compile(r"\b(?:Cass|Corte|Trib|App\.|sent|ord|sez)\w*\.?|\bn\.\s*\d+", re.I)


def drop_empty_caselaw(body: str) -> Tuple[str, str]:
    """(the answer without a case-law section that only says no ruling is relevant -- or without such sentences
    in one that also discusses a ruling --, what was done: "", "section", "sentences")."""
    m = _CASELAW_HEAD_RE.search(body or "")
    if not m:
        return body, ""
    nxt = _NEXT_HEAD_RE.search(body, m.end())
    end = nxt.start() if nxt else len(body)
    section = body[m.end():end]
    if not _NONE_RELEVANT_RE.search(section):
        return body, ""
    kept = [x for x in _sentences(section) if not _NONE_RELEVANT_RE.search(x)]
    if not any(_RULING_REF_RE.search(x) for x in kept):
        return (body[:m.start()].rstrip() + "\n\n" + body[end:].lstrip()).strip(), "section"
    return body[:m.end()] + "".join(kept) + body[end:], "sentences"


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
        contract = settle_by_documents(contract, docs, lambda sector: sector_documents(session, sector))
        info["contract"] = contract
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
        is_pay = any(TOPICS[t].get("pay") for t in topics)
        clauses, order_used = select_clauses(rows, chosen, keywords, pay=is_pay, headings=topic_headings(topics),
                                             question=question)
        # A clause the topic's words or headings found; the rest (cosine alone) only when nothing else is.
        strong = [c for c in clauses if c.get("_head", 0) >= 2 or c.get("_lex", 0) >= 4]
        later = None
        if strong:
            clauses = [c for c in clauses if c.get("_head", 0) or c.get("_lex", 0)]
        elif version["base"]["kind"] not in BASE_KINDS:
            # The base is a renewal (the full text in force at the date is not in the database): the clause from
            # the earliest full text after the date, said to be later.
            later = later_full_text(docs, when)
            if later is not None:
                later_rows = document_sections(session, [later["id"]], embedding)
                for r in later_rows:
                    r["text"] = clean(r.get("text"))
                found, _ = select_clauses(later_rows, [later], keywords, headings=topic_headings(topics),
                                          question=question)
                found = [c for c in found if c.get("_head", 0) >= 2 or c.get("_lex", 0) >= 4]
                if found:
                    for c in found:
                        c["later"] = True
                    clauses = found
                else:
                    later = None
        pay = pay_tables(group_ordered(rows)[0], chosen, when) if is_pay else None
        if pay is not None:
            info["pay"] = {"candidates": pay["candidates"],
                           "source": short_name(pay["source"]) if pay["source"] else None,
                           "decorrenza": pay["decorrenza"].isoformat() if pay["decorrenza"] else None,
                           "tables": [{"sections": t["sections"], "rows": len(t["rows"]["rows"])} for t in pay["tables"]],
                           "superseded_by": {"doc": short_name(pay["superseded_by"]["doc"]),
                                             "decorrenza": pay["superseded_by"]["decorrenza"].isoformat()}
                           if pay["superseded_by"] else None}
            vlog("ccnl_pay", info["pay"])
        info["clauses"] = {"topics": topics, "keywords": keywords, "sections": len(rows), "order_by": order_used,
                           "later_text": _doc_log(later) if later else None,
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
                             "by_reference": [f"{n} {label}" for n, label in refs if f"{n} {label}" in missing
                                              and (n, label) in FRAMEWORK_REFERENCES],
                             "rulings": [(r.get("d") or {}).get("name") for r in rulings]}
        vlog("ccnl_framework", info["framework"])
    material_blocks = [_clause_block(i, c, pay is None) for i, c in enumerate(clauses, 1)]
    if pay is not None and pay["tables"]:
        material_blocks.append(pay_block(pay))
    law_blocks = [f"[LEGGE] art. {r['_article'][0]} {r['_article'][1]}"
                  f"{' - ' + r['s'].get('titolo_articolo') if r['s'].get('titolo_articolo') else ''}\n"
                  f"{clean(r['s'].get('plain_text'))[:1500]}" for r in framework]
    law_blocks += [f"[LEGGE - testo non disponibile] art. {n} {label} - {FRAMEWORK_REFERENCES[(n, label)]}"
                   for n, label in refs if f"{n} {label}" in missing and (n, label) in FRAMEWORK_REFERENCES]
    ruling_blocks = [f"[PRONUNCIA] {_doc_name(r.get('d') or {})}\n"
                     f"{clean((r.get('s') or {}).get('plain_text'))[:900]}" for r in rulings]
    head = header(contract, version, when, how) + (f"\n{pay_line(pay)}" if pay is not None else "")
    if later is not None:
        head += (f"\nNota: il testo integrale del CCNL in vigore al {_fmt(when)} non è nella banca dati (c'è "
                 f"{short_name(version['base']).split(' — ')[-1]}, che non regola questo punto); la clausola citata è "
                 f"tratta da {short_name(later).split(' — ')[-1]} (in vigore dal {_fmt(later.get('from'))}), successivo "
                 f"alla data: verificare che la regola fosse già la stessa.")
    human = (f"Domanda: {question}\n\nContratto: CCNL {contract['sector']}; data di riferimento {_fmt(when)}.\n"
             f"Le clausole di accordi successivi prevalgono su quelle del testo base sullo stesso punto.\n"
             + (f"ATTENZIONE: le clausole vengono da {short_name(later)}, in vigore dal {_fmt(later.get('from'))}, "
                f"successivo alla data: presentale come la regola del testo successivo, da verificare per la data.\n"
                if later is not None else "") + "\n"
             + ("\n\n".join(material_blocks) or "[nessuna clausola pertinente trovata nel CCNL]")
             + "\n\n" + ("\n\n".join(law_blocks) or "[nessun articolo di legge reperito]")
             + ("\n\n" + "\n\n".join(ruling_blocks) if ruling_blocks else ""))
    body = _call_chat([SystemMessage(content=system_prompt(pay is not None, bool(rulings))),
                       HumanMessage(content=human)], max_tokens=1800) if llm else ""
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
    # A pay question's amounts only from the tables in force; any other question's from its material.
    amount_texts = [t["text"] for t in pay["tables"]] if pay is not None else clause_texts
    body, amounts = check_amounts(body, "\n".join(amount_texts + law_blocks))
    body, caselaw = drop_empty_caselaw(body)
    final = f"{head}\n\n{body.strip()}".strip()
    info["answer"] = {"verbatim_found": len(copied), "rewritten": rewritten, "verbatim_dropped": len(still),
                      "amounts_removed": amounts, "caselaw_dropped": caselaw, "chars": len(final)}
    info["outcome"] = "answered"
    vlog("ccnl_path", {k: v for k, v in info.items() if k != "question"}, duration_ms=(time.time() - t0) * 1000)
    ccnl_rows = [{"d": {"id": c["doc"]["id"], "name": c["doc"].get("name"), "document_type": TYPE,
                        "sector": contract["sector"]},
                  "s": {"id": c["hit"], "name": clause_reference(c), "plain_text": ""}, "_source": "ccnl_clause"}
                 for c in clauses]
    return {"answer": final, "citations": _citations(clauses, framework, rulings),
            "raw_result": ccnl_rows + framework + rulings, "status_messages": [], "ccnl_sector": contract["sector"],
            "ccnl": info, "route": "ccnl"}


def later_full_text(docs: Sequence[Dict[str, Any]], when: date) -> Optional[Dict[str, Any]]:
    """The sector's earliest CCNL or Testo Consolidato starting after `when` (a CCNL before a Testo Consolidato
    of the same date: the text closest to the date's), None without one."""
    full = [d for d in (_doc_view(x) for x in docs) if d["kind"] in BASE_KINDS and d["from"] and d["from"] > when]
    return min(full, key=lambda d: (d["from"], d["kind"] != CCNL, d.get("id") or "")) if full else None


def _doc_name(d: Dict[str, Any]) -> str:
    return re.sub(r"^\s*\[[A-Z]+\]\s*", "", str(d.get("name") or d.get("id") or ""))


def _doc_log(d: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": d.get("id"), "name": short_name(d), "kind": d.get("kind"),
            "valid_from": d["from"].isoformat() if d.get("from") else None,
            "valid_until": d["until"].isoformat() if d.get("until") else None}


def group_ordered(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Optional[str]]]:
    """({document: its Sections in order}, {document: the order key used}); CCNL_SECTION_ORDER forces a key."""
    forced = os.getenv(ORDER_ENV) or None
    by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_doc.setdefault(r["doc"], []).append(r)
    order_used: Dict[str, Optional[str]] = {}
    for doc_id, doc_rows in by_doc.items():
        by_doc[doc_id], order_used[doc_id] = ordered(doc_rows, forced)
    return by_doc, order_used


def select_clauses(rows: List[Dict[str, Any]], chosen: Sequence[Dict[str, Any]], keywords: Sequence[str],
                   pay: bool = False, headings: Sequence[str] = (), question: str = ""
                   ) -> Tuple[List[Dict[str, Any]], Dict[str, Optional[str]]]:
    """(the clauses, rebuilt from the best hits -- a heading naming the topic first (rank_sections) --: at most
    MAX_BASE_CLAUSES from the base, MAX_AMENDMENT_CLAUSES from the amendments -- the latest first on a tie --,
    MAX_CLAUSES in all, a hit inside a clause already taken skipped; {document: the order key used}). On a pay
    question each clause gets its table rows."""
    by_doc, order_used = group_ordered(rows)
    position: Dict[Any, Tuple[str, int]] = {}
    for doc_id, doc_rows in by_doc.items():
        for i, r in enumerate(doc_rows):
            position[id(r)] = (doc_id, i)
    docs = {d["id"]: d for d in chosen}
    base_id = chosen[0]["id"] if chosen else None
    rank = {d["id"]: i for i, d in enumerate(chosen)}
    hits = rank_sections(rows, keywords, headings, question)
    hits.sort(key=lambda r: (-r["_head"], -round(r["_rrf"], 6), -rank.get(r["doc"], 0), str(r.get("id"))))
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
        clause = rebuild_clause(by_doc[doc_id], i, anchor=r.get("_anchor") if r.get("_via") == "heading" else None,
                                name_headed=r.get("_via") == "name" and r.get("_head", 0) >= 2)
        clause["_head"], clause["_lex"] = r.get("_head", 0), r.get("_lex", 0)
        taken.setdefault(doc_id, []).append((clause["start"], clause["end"]))
        clause["doc"] = docs.get(doc_id) or {"id": doc_id, "name": doc_id, "kind": OTHER}
        clause["span"] = [by_doc[doc_id][k].get("name") for k in range(clause["start"], clause["end"] + 1)]
        if pay:
            clause["rows"] = pay_rows(clause["text"])
        counts[side] += 1
        clauses.append(clause)
    clauses.sort(key=lambda c: (rank.get(c["doc"]["id"], 0), c["start"]))
    return clauses, {short_name(docs.get(k) or {"id": k}): v for k, v in order_used.items()}


# --- pay: the table in force at the date ---------------------------------------------------------------------

# A pay table names what it pays (minimi, paga base, stipendio...) in its text or in the header before it;
# a contribution table (fondi, casse, enti bilaterali) does not.
_PAY_WORDS_RE = re.compile(r"minim[io]\b|minimi|paga base|paga oraria|tabellar\w*|stipendi\w*|conglobat\w*|"
                           r"retribuzion\w* (?:base|mensil\w*|tabellar\w*|minim\w*|contrattual\w*)|importi mensili",
                           re.I)
# Agreements about something else than pay, whose tables are contributions or quotas.
_NOT_PAY_DOC_RE = re.compile(r"previdenz|assistenz|sanitari|welfare|formazion|bilateral|sicurezz|\bfondo\b", re.I)
_LOOSE_DATE_RE = re.compile(rf"(\d{{1,2}})(?:°|º)?\s*[/.\s-]\s*(\d{{1,2}}|{'|'.join(_MONTHS)})\s*[/.\s-]\s*((?:19|20)\d{{2}})",
                            re.I)
MAX_PAY_TABLES = 3


def loose_date(text: str) -> Optional[date]:
    """"01/09/2022", "1-1-2024", "1º giugno 2024", "01.07.2021" -> a date; None otherwise."""
    m = _LOOSE_DATE_RE.search(text or "")
    if not m:
        return None
    month = m.group(2)
    month = int(month) if month.isdigit() else _MONTHS.index(month.lower()) + 1
    try:
        return date(int(m.group(3)), month, int(m.group(1)))
    except ValueError:
        return None


def pay_tables(by_doc: Dict[str, List[Dict[str, Any]]], chosen: Sequence[Dict[str, Any]], when: date
               ) -> Dict[str, Any]:
    """The pay table in force at `when`, across the base and every amending document (not only the clauses
    found): each Section with pay words (in it or in the header ending the section before: minimi, paga base,
    stipendio, retribuzione tabellare ...) and three or more amounts or two table rows -- never in an agreement on
    pensions, health, welfare, training or bilateral bodies --, joined to the table rows that follow it; its decorrenza the latest date it writes on or before `when` (the date before it included,
    where a header sits), else its document's valid_from; a table whose every date is after `when` is not yet
    in force. Readable: two or more rows read (pay_rows).
      {"tables": up to MAX_PAY_TABLES readable ones of the best (document, decorrenza), "source", "decorrenza",
       "superseded_by": {"doc", "decorrenza"} when a later table is unreadable, "candidates": n}"""
    rank = {d["id"]: i for i, d in enumerate(chosen)}
    docs = {d["id"]: d for d in chosen}
    candidates = []
    for doc_id, rows in by_doc.items():
        doc = docs.get(doc_id)
        if doc is None or _NOT_PAY_DOC_RE.search(str(doc.get("document_subtype") or short_name(doc).split(" — ")[-1])):
            continue
        i = 0
        while i < len(rows):
            text = rows[i].get("text") or ""
            header = ((rows[i - 1].get("text") or "")[-300:] + "\n" if i else "") + text
            parsed = pay_rows(text)
            if not _PAY_WORDS_RE.search(header) or (len(_AMOUNT_RE.findall(text)) < 3 and len(parsed["rows"]) < 2):
                i += 1
                continue
            j = i
            while j + 1 < len(rows) and j - i < 4 and pay_rows(rows[j + 1].get("text") or "")["rows"] \
                    and not opens_with_heading(rows[j + 1].get("text") or ""):
                j += 1
            table = "\n".join(rows[k].get("text") or "" for k in range(i, j + 1))
            context = (rows[i - 1].get("text") or "")[-300:] + "\n" + table if i else table
            dates = [d for d in (loose_date(x) for x in _DECORRENZA_RE.findall(context)) if d]
            dates += [d for d in (loose_date(m.group(0)) for m in _LOOSE_DATE_RE.finditer(context[:400])) if d]
            in_force = [d for d in dates if d <= when]
            if dates and not in_force:
                i = j + 1
                continue
            decorrenza = max(in_force) if in_force else (doc.get("from") if doc.get("from") and doc["from"] <= when
                                                         else None)
            if decorrenza is not None:
                rows_read = pay_rows(table)
                candidates.append({"doc": doc, "decorrenza": decorrenza, "readable": len(rows_read["rows"]) >= 2,
                                   "rows": rows_read, "text": table, "position": i,
                                   "sections": [rows[k].get("name") for k in range(i, j + 1)]})
            i = j + 1
    out: Dict[str, Any] = {"tables": [], "source": None, "decorrenza": None, "superseded_by": None,
                           "candidates": len(candidates)}
    if not candidates:
        return out

    def key(c):
        return (c["decorrenza"], rank.get(c["doc"]["id"], 0))
    latest = max(candidates, key=key)
    readable = [c for c in candidates if c["readable"]]
    best = max(readable, key=key) if readable else None
    if best is None or key(best) < key(latest):
        out["superseded_by"] = {"doc": latest["doc"], "decorrenza": latest["decorrenza"]}
    if best is not None:
        same = [c for c in readable if c["doc"]["id"] == best["doc"]["id"] and c["decorrenza"] == best["decorrenza"]]
        out["tables"] = sorted(same, key=lambda c: c["position"])[:MAX_PAY_TABLES]
        out["source"], out["decorrenza"] = best["doc"], best["decorrenza"]
    return out


def pay_line(pay: Dict[str, Any]) -> str:
    """The header's line on the amounts: the source document and decorrenza, or why none is given."""
    if pay.get("source"):
        line = (f"**Importi**: dalla tabella di {short_name(pay['source']).split(' — ')[-1]} con decorrenza dal "
                f"{_fmt(pay['decorrenza'])}.")
        later = pay.get("superseded_by")
        if later:
            line += (f" Potrebbe essere stata superata da {short_name(later['doc']).split(' — ')[-1]} (dal "
                     f"{_fmt(later['decorrenza'])}), la cui tabella non è leggibile: verificare gli importi aggiornati.")
        return line
    if pay.get("superseded_by"):
        later = pay["superseded_by"]
        return (f"**Importi**: la tabella in vigore ({short_name(later['doc']).split(' — ')[-1]}, dal "
                f"{_fmt(later['decorrenza'])}) non è leggibile: non indico importi, da verificare sul testo.")
    return "**Importi**: nei documenti in vigore alla data non c'è una tabella retributiva leggibile: non indico importi."


def pay_block(pay: Dict[str, Any]) -> str:
    blocks = []
    for n, t in enumerate(pay.get("tables") or [], 1):
        blocks.append(f"[TABELLA IMPORTI {n}] fonte: {short_name(t['doc'])}, decorrenza dal {_fmt(t['decorrenza'])}\n"
                      f"{_rows_block(t['rows'])}")
    return "\n\n".join(blocks)
