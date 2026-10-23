"""RULING_CARDS_ENABLED: the strategy's rulings chosen from the precomputed ruling cards.

A card (scripts/exp_ruling_cards.py) is one Cassazione ruling read once by the model: the
articles it applies (from those it cites), who appealed, the outcome and whom it favours, its
decisive argument and keyword phrases, a verbatim quote checked against the text, its full
reference. They are read from RULING_CARDS_FILES (comma-separated JSONL paths), again when a
file changes, so cards added by a night run are used without a restart.

For a case, every card sharing one of its central articles is scored as scripts/
exp_ruling_search.py measured it (23 Oct, criminal tuning cases, the five rulings the writer
reads: on topic 6% -> about 40%, held-out 0% -> 24%; full reference about 95%; for the
client's side 8% -> 44%): rarity-weighted central articles (named in the case or resolved from
its legal issues 1, the model's chosen pool 0.5, the generic method candidates left out; an
article the ruling only cites x0.3), keyword phrases found in the case, and a bonus when the
ruling favours the client. The best _CARD_SLOTS lead the writer's reading order, and their
passage starts with the outcome, whom it favours and the verified quote. Any failure leaves
the rulings as the pipeline found them.
"""

import json
import logging
import math
import os
import re
import threading
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from . import feature_flags
from .article_suffix import normalize_number
from .verbose_logger import vlog

logger = logging.getLogger(__name__)

RULING_CARDS_FLAG = "RULING_CARDS_ENABLED"
FILES_ENV = "RULING_CARDS_FILES"

_CARD_SLOTS = 3
W_ARTICLE, W_KEYWORD, W_SIDE = 3, 1, 2
CENTRAL, CHOSEN, CITED_ONLY = 1.0, 0.5, 0.3

_DEFENDANT_RE = re.compile(r"imputat|indagat|accusat|condannat|difes|arrestat|autore\s+del", re.IGNORECASE)
_VICTIM_RE = re.compile(r"persona\s+offesa|parte\s+civile|vittim|danneggiat|denunciant|querelant|famil", re.IGNORECASE)


# --- the cards ---------------------------------------------------------------------------------

def _article_key(number: Any, code: str) -> str:
    """('584', 'c.p.') -> '584 c.p.': the cards' own form (defensive_generation._article_num, as
    scripts/exp_ruling_links.py wrote them)."""
    return f"{normalize_number(str(number).lower().split('.')[0])} {code}"


def _trim(card: Dict[str, Any]) -> Dict[str, Any]:
    """What ranking and the writer need of a card."""
    keywords = [str(k) for k in (card.get("argomento_decisivo") or []) + (card.get("parole_chiave_fatti") or [])
                + (card.get("parole_chiave_questioni") or []) if str(k).strip()]
    return {"id": card["id"], "applied": set(card.get("articoli_applicati") or []),
            "cited": set(card.get("articoli_citati") or []), "keywords": list(dict.fromkeys(keywords)),
            "esito": str(card.get("esito") or ""), "ricorrente": str(card.get("ricorrente") or ""),
            "a_favore_di": str(card.get("a_favore_di") or ""), "parte_favorita": str(card.get("parte_favorita") or ""),
            "citazione": str(card.get("citazione") or "") if card.get("citazione_verificata") else "",
            "riferimento": str(card.get("riferimento") or ""), "completo": bool(card.get("riferimento_completo")),
            "branch": str(card.get("branch") or "")}


class CardStore:
    """The cards of some JSONL files, with each article's cards and the rarity of articles and keywords."""

    def __init__(self, cards: List[Dict[str, Any]]):
        self.cards = cards
        self.by_article: Dict[str, List[int]] = {}
        for i, c in enumerate(cards):
            for article in c["applied"] | c["cited"]:
                self.by_article.setdefault(article, []).append(i)
        n = max(len(cards), 2)
        self.article_rarity = _rarity(Counter(a for c in cards for a in c["applied"]), n)
        scale = math.log(n)
        self.keyword_rarity = {k: v / scale for k, v in
                               _rarity(Counter(k for c in cards for k in set(c["keywords"])), n).items()}

    @classmethod
    def from_files(cls, paths: List[str]) -> "CardStore":
        cards, seen = [], set()
        for path in paths:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        card = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(card, dict) and card.get("id") and not card.get("errore") \
                            and card["id"] not in seen:
                        seen.add(card["id"])
                        cards.append(_trim(card))
        return cls(cards)


def _rarity(counts: Counter, total: int) -> Dict[str, float]:
    """item -> log(total / count): an article or keyword on every card weighs ~0."""
    return {item: math.log(total / n) for item, n in counts.items() if n}


_store: Optional[CardStore] = None
_signature: Optional[tuple] = None
_lock = threading.Lock()


def _paths() -> List[str]:
    return [p.strip() for p in os.getenv(FILES_ENV, "").split(",") if p.strip()]


def store() -> Optional[CardStore]:
    """The cards of RULING_CARDS_FILES, loaded again when a file's size or time changed; None
    when no file is set or readable."""
    global _store, _signature
    paths = [p for p in _paths() if os.path.isfile(p)]
    if not paths:
        return None
    signature = tuple((p, os.path.getsize(p), os.path.getmtime(p)) for p in paths)
    with _lock:
        if signature != _signature:
            _store, _signature = CardStore.from_files(paths), signature
            logger.info("ruling cards: %d loaded from %s", len(_store.cards), ", ".join(paths))
        return _store


# --- ranking -----------------------------------------------------------------------------------

def client_side(posizione: str) -> str:
    """'imputato' / 'accusa' from the client's position in a criminal case, '' when unclear."""
    defendant, victim = bool(_DEFENDANT_RE.search(posizione or "")), bool(_VICTIM_RE.search(posizione or ""))
    return "imputato" if defendant and not victim else "accusa" if victim and not defendant else ""


def _stems(text: str) -> List[str]:
    return [w[:5] for w in re.findall(r"\w{4,}", (text or "").lower())]


def favours_client(card: Dict[str, Any], side: str, party: str) -> bool:
    """A criminal card for the client's side; a civil card whose favoured party is the client's role."""
    if card["branch"] == "civile":
        stems = _stems(card["parte_favorita"])
        return bool(stems) and all(s in party.lower() for s in stems)
    return bool(side) and card["a_favore_di"] == side


def central_weights(picked: Dict[str, Any], resolved: List[tuple]) -> Dict[str, float]:
    """The case's central articles and their weight, from the pipeline's own picks: named in the
    case or resolved from its legal issues CENTRAL, the model's chosen pool CHOSEN; the method
    candidates (generic articles by area and posture) left out."""
    method = {_article_key(*r) for r in picked.get("method_candidates") or []}
    weights = {}
    for ref in picked.get("chosen") or []:
        key = _article_key(*ref)
        if key not in method:
            weights[key] = CHOSEN
    for ref in list(picked.get("named") or []) + list(picked.get("institutions_resolved") or []) + list(resolved):
        weights[_article_key(*ref)] = CENTRAL
    return weights


def _hit(keyword: str, low: str) -> bool:
    stems = _stems(keyword)
    return bool(stems) and all(s in low for s in stems)


def score_card(card: Dict[str, Any], central: Dict[str, float], low_terms: str, favoured: bool,
               cards: CardStore) -> float:
    """W_ARTICLE x weight x rarity per central article the card applies (x CITED_ONLY when the
    ruling only cites it), W_KEYWORD x rarity per keyword phrase found in the case, W_SIDE when
    it favours the client; 0 when it shares no central article."""
    shared = [a for a in card["applied"] if a in central]
    cited_only = [a for a in card["cited"] - card["applied"] if a in central]
    if not shared and not cited_only:
        return 0.0
    score = W_ARTICLE * sum(central[a] * cards.article_rarity.get(a, 1.0) for a in shared)
    score += W_ARTICLE * CITED_ONLY * sum(central[a] * cards.article_rarity.get(a, 1.0) for a in cited_only)
    score += W_KEYWORD * sum(cards.keyword_rarity.get(k, 1.0) for k in card["keywords"] if _hit(k, low_terms))
    return score + (W_SIDE if favoured else 0)


def rank(cards: CardStore, central: Dict[str, float], case_terms: str, side: str, party: str,
         top: int = _CARD_SLOTS) -> List[Tuple[Dict[str, Any], float]]:
    """The `top` best-scoring cards with their score; ties broken by full reference, then id."""
    low = " ".join((case_terms or "").lower().split())
    candidates = {i for article in central for i in cards.by_article.get(article, ())}
    scored = []
    for i in candidates:
        card = cards.cards[i]
        s = score_card(card, central, low, favours_client(card, side, party), cards)
        if s > 0:
            scored.append((-s, not card["completo"], card["id"], i))
    return [(cards.cards[i], -s) for s, _, _, i in sorted(scored)[:top]]


# --- the pipeline's side -----------------------------------------------------------------------

def _resolved_institutions(session, driver, database: str, analysis: Dict[str, Any]) -> List[tuple]:
    """Every article the analysis' institutions resolve to (database only, no model call): the
    selection's own "institutions_resolved" keeps only those no earlier step had found."""
    from . import defensive_generation as dg
    from .doc_lookup import _fetch_doc_names
    documents = _fetch_doc_names(driver, database)
    found = []
    for q in analysis.get("questioni") or []:
        for istituto in q.get("istituti") or []:
            try:
                rows = dg._resolve_institution(session, documents, istituto, q.get("parole_chiave"))
            except Exception:
                continue
            found += [row["_article"] for row in rows if row.get("_article")]
    return list(dict.fromkeys(found))


def _case_terms(case: str, analysis: Dict[str, Any]) -> str:
    parts = [case] + list(analysis.get("ricerca_fatti") or [])
    for q in analysis.get("questioni") or []:
        parts += [q.get("titolo") or ""] + [w for w in (q.get("parole_chiave") or []) if isinstance(w, str)]
    return " ".join(str(p) for p in parts)


def card_rulings(session, driver, database: str, case: str, analysis: Dict[str, Any],
                 picked: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """(ruling rows for the best cards, in rank order, as the search returns rulings; {id: card}).
    ([], {}) with no cards or on any failure."""
    try:
        cards = store()
        if cards is None:
            logger.warning("%s is on but %s names no readable file: no card rulings", RULING_CARDS_FLAG, FILES_ENV)
            return [], {}
        central = central_weights(picked, _resolved_institutions(session, driver, database, analysis))
        party = f"{analysis.get('parte_assistita') or ''} {analysis.get('posizione') or ''}"
        best = rank(cards, central, _case_terms(case, analysis), client_side(analysis.get("posizione") or ""), party)
        ids = [card["id"] for card, _ in best]
        found = {row["d"]["id"]: row["d"] for row in session.run(
            "MATCH (d:Document) WHERE d.id IN $ids RETURN d", ids=ids).data() if row.get("d")} if ids else {}
        vlog("case_analysis_ruling_cards", {"cards": len(cards.cards), "central": len(central),
                                            "chosen": [{"riferimento": c["riferimento"], "score": round(s, 2),
                                                        "a_favore_di": c["a_favore_di"] or c["parte_favorita"]}
                                                       for c, s in best],
                                            "missing_in_db": [i for i in ids if i not in found]})
        return [{"d": found[i]} for i in ids if i in found], {c["id"]: c for c, _ in best if c["id"] in found}
    except Exception as exc:
        logger.warning("ruling cards failed, the rulings are the search's own: %s", exc)
        return [], {}


def lead_with(card_rows: List[Dict[str, Any]], reading: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The card rulings first, then the search's own without repeats."""
    ids = {(r.get("d") or {}).get("id") for r in card_rows}
    return card_rows + [r for r in reading if (r.get("d") or {}).get("id") not in ids]


def card_summary(card: Dict[str, Any]) -> str:
    """The card's outcome, whom it favours and its verified quote, as the writer reads them."""
    parts = [f"Esito (dalla scheda): {card['esito'] or 'non indicato'}"]
    if card["ricorrente"] and card["ricorrente"] != "altro":
        parts.append(f"ricorso di: {card['ricorrente']}")
    favoured = card["parte_favorita"] or card["a_favore_di"]
    if favoured and favoured != "non chiaro":
        parts.append(f"favorevole a: {favoured}")
    text = "; ".join(parts) + "."
    return text + (f" Principio decisivo: «{card['citazione']}»." if card["citazione"] else "")


def annotate(ruling_rows: List[Dict[str, Any]], carded: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The rows of carded rulings with the card summary before their passage."""
    out = []
    for row in ruling_rows:
        card = carded.get((row.get("d") or {}).get("id"))
        if card:
            passage = row.get("_passage") or (row.get("s") or {}).get("plain_text") or ""
            row = {**row, "_passage": f"{card_summary(card)} Passo: {passage}".strip()}
        out.append(row)
    return out


def enabled() -> bool:
    return feature_flags.enabled(RULING_CARDS_FLAG)
