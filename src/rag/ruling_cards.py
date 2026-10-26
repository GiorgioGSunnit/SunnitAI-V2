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
ruling favours the client.

Which cards lead the writer's reading order is chosen by side, not by score alone (round 5 of
the PM's cases, 23 Oct: 9 of the 12 cards ranked for two defendants favoured the prosecution,
the side bonus being small against 40-53 central articles): up to _FAVOURABLE_SLOTS cards that
favour the client and score at least _FAVOURABLE_FLOOR of the best, then _UNFAVOURABLE_SLOTS
of the others, the strongest adverse ruling, labelled for the writer as the other side's case
law to rebut. Their passage starts with that label, the outcome, whom the ruling favours and
the verified quote. Any failure leaves the rulings as the pipeline found them.
"""

import functools
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

# Rulings of the execution of a sentence (artt. 656-678 c.p.p.: suspension of the order, incidente
# di esecuzione, continuation in executivis, the supervision court) are about a stage no strategy
# case of the PM's has reached: round 6 (23 Oct) cited them, as favourable, for stalking and burglary
# defences. A card applying one of them is left out unless the case's own posture is that stage.
_EXECUTION_ARTICLES = {f"{n} c.p.p." for n in range(656, 679)}
_EXECUTION_POSTURE_RE = re.compile(r"esecuzion|sorveglianz", re.IGNORECASE)

# An article applied by more than this share of the cards (and more than _GENERIC_MIN of them)
# is generic. Round 7 (24 Oct, all 8,936 criminal cards): with 45-65 central articles, mostly the
# model's broad pool, a ruling on the weapons of mafia associations ranked as favourable for a
# stalking defence. A card now needs an anchor: an article it applies that is not generic and is
# central to the case, named in it or resolved from its issues; or in the model's pool, with one
# of the card's keyword phrases also found in the case.
_GENERIC_SHARE, _GENERIC_MIN = 0.02, 50

# Favourable cards first, then the strongest adverse one (0 to bring only favourable rulings).
_FAVOURABLE_SLOTS, _UNFAVOURABLE_SLOTS = 2, 1
_FAVOURABLE_FLOOR = 0.5            # a favourable card must still be on topic: half the best score

# Civil cases (round 10, 25 Oct): the cards' parties are procedural (ricorrente / controricorrente),
# so every civil pick is "neutro" and the slots are only a number; the civil search is already
# fairly on topic (about 40%). Both settable from the .env, measured with exp_ruling_search.py
# --combined --civil-slots N [--strong-only]: RULING_CARDS_CIVIL_SLOTS (default 3, as criminal) and
# RULING_CARDS_CIVIL_STRONG_ONLY (default false: pool anchors with a keyword hit allowed).
CIVIL_SLOTS_ENV, CIVIL_STRONG_ENV = "RULING_CARDS_CIVIL_SLOTS", "RULING_CARDS_CIVIL_STRONG_ONLY"


def civil_settings() -> Tuple[int, bool]:
    """(card slots, strong anchors only) for a civil case, from the .env."""
    try:
        slots = int(os.getenv(CIVIL_SLOTS_ENV, "") or _FAVOURABLE_SLOTS + _UNFAVOURABLE_SLOTS)
    except ValueError:
        slots = _FAVOURABLE_SLOTS + _UNFAVOURABLE_SLOTS
    return max(0, min(slots, 5)), feature_flags.enabled(CIVIL_STRONG_ENV)


def _slot_split(slots: int) -> Tuple[int, int]:
    """(favourable, unfavourable) for `slots` card rulings: one adverse slot, the rest favourable."""
    return (max(slots - _UNFAVOURABLE_SLOTS, 0), min(_UNFAVOURABLE_SLOTS, slots)) if slots else (0, 0)
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
            "branch": str(card.get("branch") or ""),
            # The issues it decides: the narrative map's issue queries (HYBRID_NARRATIVE_ENABLED, source_tiers).
            "issues": [str(k).strip() for k in card.get("parole_chiave_questioni") or [] if str(k).strip()]}


class CardStore:
    """The cards of some JSONL files, with each article's cards and the rarity of articles and keywords."""

    def __init__(self, cards: List[Dict[str, Any]]):
        self.cards = cards
        self.by_article: Dict[str, List[int]] = {}
        for i, c in enumerate(cards):
            for article in c["applied"] | c["cited"]:
                self.by_article.setdefault(article, []).append(i)
        n = max(len(cards), 2)
        applied = Counter(a for c in cards for a in c["applied"])
        self.article_rarity = _rarity(applied, n)
        # Articles too common to say what a ruling is about (24 Oct, 8,936 criminal cards: 606 c.p.p.
        # in 12%, 133 c.p. 7%, 81 c.p. 5%, 416-bis 3%): never an anchor (anchors()).
        self.generic = {a for a, k in applied.items() if k > max(_GENERIC_SHARE * n, _GENERIC_MIN)}
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


@functools.lru_cache(maxsize=1)
def _method_keys() -> frozenset:
    """The general-part and procedural method articles of every area and posture (method_candidates.py):
    specific by frequency but not by meaning (43 c.p., the mental element; 369-bis or 597 c.p.p.)."""
    from . import method_candidates
    refs = [r for _, rs in list(method_candidates.AREA_ARTICLES.values())
            + list(method_candidates.POSTURE_ARTICLES.values()) for r in rs]
    return frozenset(_article_key(*r) for r in refs)


def anchors(card: Dict[str, Any], central: Dict[str, float], low_terms: str, cards: CardStore,
            strong_only: bool = False) -> List[str]:
    """The non-generic central articles the card applies that tie it to the case: named in the
    case or resolved from its issues (CENTRAL); from the model's pool (CHOSEN) only when one of
    the card's keyword phrases is also found in the case, and never with `strong_only`. Never a
    method article (_method_keys): round 8 (24 Oct) anchored a stalking case's cards on 43 c.p.
    and on 369-bis and 597 c.p.p."""
    method = _method_keys()
    specific = [a for a in card["applied"] if a in central and a not in cards.generic and a not in method]
    strong = [a for a in specific if central[a] >= CENTRAL]
    if strong or strong_only:
        return sorted(strong)
    return sorted(specific) if specific and any(_hit(k, low_terms) for k in card["keywords"]) else []


def score_card(card: Dict[str, Any], central: Dict[str, float], low_terms: str, favoured: bool,
               cards: CardStore, strong_only: bool = False) -> float:
    """W_ARTICLE x weight x rarity per central article the card applies (x CITED_ONLY when the
    ruling only cites it), W_KEYWORD x rarity per keyword phrase found in the case, W_SIDE when
    it favours the client; 0 without an anchor (anchors())."""
    if not anchors(card, central, low_terms, cards, strong_only):
        return 0.0
    shared = [a for a in card["applied"] if a in central]
    cited_only = [a for a in card["cited"] - card["applied"] if a in central]
    score = W_ARTICLE * sum(central[a] * cards.article_rarity.get(a, 1.0) for a in shared)
    score += W_ARTICLE * CITED_ONLY * sum(central[a] * cards.article_rarity.get(a, 1.0) for a in cited_only)
    score += W_KEYWORD * sum(cards.keyword_rarity.get(k, 1.0) for k in card["keywords"] if _hit(k, low_terms))
    return score + (W_SIDE if favoured else 0)


def execution_stage(card: Dict[str, Any]) -> bool:
    """The ruling applies an article of the execution of a sentence (_EXECUTION_ARTICLES)."""
    return bool(card["applied"] & _EXECUTION_ARTICLES)


def rank(cards: CardStore, central: Dict[str, float], case_terms: str, side: str, party: str,
         top: Optional[int] = None, execution: bool = False,
         strong_only: bool = False) -> List[Tuple[Dict[str, Any], float]]:
    """The scoring cards (the `top` best, else all) with their score, best first; ties broken by
    full reference, then id. Execution-stage rulings only when `execution` (the case is at that
    stage); with `strong_only`, anchors only on articles named in the case or resolved from it."""
    low = " ".join((case_terms or "").lower().split())
    candidates = {i for article in central for i in cards.by_article.get(article, ())}
    scored = []
    for i in candidates:
        card = cards.cards[i]
        if not execution and execution_stage(card):
            continue
        s = score_card(card, central, low, favours_client(card, side, party), cards, strong_only)
        if s > 0:
            scored.append((-s, not card["completo"], card["id"], i))
    return [(cards.cards[i], -s) for s, _, _, i in sorted(scored)[:top]]


def role_of(card: Dict[str, Any], side: str, party: str) -> str:
    """"favorevole" to the client, "sfavorevole" (a criminal ruling for the other side), else
    "neutro": side unclear, the client's side unknown, or a civil ruling for another party (a
    different dispute, not necessarily against the client)."""
    if favours_client(card, side, party):
        return "favorevole"
    if card["branch"] != "civile" and side and card["a_favore_di"] in ("imputato", "accusa"):
        return "sfavorevole"
    return "neutro"


def choose(ranked: List[Tuple[Dict[str, Any], float]], side: str, party: str,
           favourable: int = _FAVOURABLE_SLOTS, unfavourable: int = _UNFAVOURABLE_SLOTS) -> List[tuple]:
    """[(card, score, role)]: up to `favourable` cards for the client scoring at least
    _FAVOURABLE_FLOOR of the best; then up to `unfavourable` of the others, an adverse ruling
    before an unclear one; the slots left go to unclear ones, never to more adverse rulings."""
    if not ranked:
        return []
    floor = _FAVOURABLE_FLOOR * ranked[0][1]
    roles = [(card, score, role_of(card, side, party)) for card, score in ranked]
    picks = [r for r in roles if r[2] == "favorevole" and r[1] >= floor][:favourable]
    others = sorted((r for r in roles if r[2] != "favorevole"), key=lambda r: (r[2] != "sfavorevole", -r[1]))
    picks += others[:unfavourable]
    left = favourable + unfavourable - len(picks)
    picks += [r for r in roles if r[2] == "neutro" and r not in picks][:max(left, 0)]
    return picks


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
        side = client_side(analysis.get("posizione") or "")
        frame = analysis.get("frame") or {}
        execution = bool(_EXECUTION_POSTURE_RE.search(f"{frame.get('posture_type') or ''} "
                                                      f"{frame.get('procedural_posture') or ''}"))
        terms = _case_terms(case, analysis)
        from .defensive_generation import _case_branch
        civil = _case_branch(analysis) == "civile"
        slots, strong_only = civil_settings() if civil else (_FAVOURABLE_SLOTS + _UNFAVOURABLE_SLOTS, False)
        favourable, unfavourable = _slot_split(slots)
        best = choose(rank(cards, central, terms, side, party, execution=execution, strong_only=strong_only),
                      side, party, favourable, unfavourable)
        low = " ".join(terms.lower().split())
        ids = [card["id"] for card, _, _ in best]
        found = {row["d"]["id"]: row["d"] for row in session.run(
            "MATCH (d:Document) WHERE d.id IN $ids RETURN d", ids=ids).data() if row.get("d")} if ids else {}
        vlog("case_analysis_ruling_cards", {"cards": len(cards.cards), "central": len(central), "client_side": side,
                                            "execution_stage_allowed": execution,
                                            "chosen": [{"riferimento": c["riferimento"], "score": round(s, 2),
                                                        "a_favore_di": c["a_favore_di"] or c["parte_favorita"],
                                                        "role": role, "anchors": anchors(c, central, low, cards, strong_only)}
                                                       for c, s, role in best],
                                            "missing_in_db": [i for i in ids if i not in found]})
        return ([{"d": found[i]} for i in ids if i in found],
                {c["id"]: {**c, "_role": role} for c, _, role in best if c["id"] in found})
    except Exception as exc:
        logger.warning("ruling cards failed, the rulings are the search's own: %s", exc)
        return [], {}


def lead_with(card_rows: List[Dict[str, Any]], reading: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The card rulings first, then the search's own without repeats."""
    ids = {(r.get("d") or {}).get("id") for r in card_rows}
    return card_rows + [r for r in reading if (r.get("d") or {}).get("id") not in ids]


_ROLE_LABELS = {
    "favorevole": "Orientamento favorevole alla parte assistita.",
    "sfavorevole": "ORIENTAMENTO SFAVOREVOLE alla parte assistita: è la giurisprudenza della controparte, da "
                   "presentare come obiezione e superare con la replica, non come argomento.",
}


def card_summary(card: Dict[str, Any]) -> str:
    """The card's role for the client, outcome, whom it favours and verified quote, as the writer reads them."""
    label = _ROLE_LABELS.get(card.get("_role") or "")
    parts = [f"Esito (dalla scheda): {card['esito'] or 'non indicato'}"]
    if card["ricorrente"] and card["ricorrente"] != "altro":
        parts.append(f"ricorso di: {card['ricorrente']}")
    favoured = card["parte_favorita"] or card["a_favore_di"]
    if favoured and favoured != "non chiaro":
        parts.append(f"favorevole a: {favoured}")
    text = (f"{label} " if label else "") + "; ".join(parts) + "."
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
