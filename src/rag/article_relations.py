"""Article relations (relations agent v0, 28 Oct): which Codice Civile articles go together, and which articles a
case's text points to, learnt from the Cassazione rulings -- read from a file the agent builds
(scripts/maint_relations.py), never from the graph yet.

The file (ARTICLE_RELATIONS_FILE, JSON):
  related   {"1490": [["1492", 0.031], ["1495", 0.030], ...]}  the top related articles of each article, by the
            reciprocal-rank fusion of three relations: applied together by the same rulings (the ruling cards),
            cited together (the CITES edges), named by the article's own text (scripts/exp_relations.py: from one
            expected article, 43% of a case's other expected articles in its top 10, against 29% for the articles
            numbered around it; 27 Oct, 13 civil test cases)
  via       {"1490": {"1492": "applied,cited,text"}}  which relations link the pair (for the log)
  profiles  {"rulings": {"idf": {...}, "vectors": {...}}, "text": {...}}  per article, TF-IDF of the keywords of
            the rulings applying it and of its own text and rubrica; a case's text ranked against both (fused) found
            70% of the expected articles in its top 50, 51% in its top 20

ARTICLE_RELATIONS_ENABLED (default off): the fusion pool (defensive_generation._select_articles_fusion) appends,
after its cap, up to ADD_RELATED articles related to its strongest candidates and, in a civil case, up to
ADD_PROFILE articles whose profiles best match the case. Never a signal (not multi-signal, no exemption): every
one is graded by the applicability check like any candidate. Logged as article_relations_added.
"""

import json
import logging
import math
import os
import re
import threading
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

from . import feature_flags

logger = logging.getLogger(__name__)

ARTICLE_RELATIONS_FLAG = "ARTICLE_RELATIONS_ENABLED"
FILE_ENV = "ARTICLE_RELATIONS_FILE"
CODE = "c.c."
SEED_TOP = 5          # the fused pool's first candidates used as seeds, with the named and resolved articles
ADD_RELATED = 6       # related articles appended at most
ADD_PROFILE = 4       # profile matches appended at most (civil cases only)
RRF_K = 60

_STOP = set("""
alla alle allo anche ancora avere aveva base caso cioè come con contro cosa così dalla dalle dallo dati degli della
delle dello dell detto dopo dove essere fatto fatti fine gli hanno il infatti invece loro mentre nella nelle nello
nonché oltre ogni ovvero parte perché però poiché presente prima quale quali quando quanto quella quelle quello questa
queste questo sono stata stati stato sulla sulle tale tali tanto tutte tutti una uno viene articolo articoli comma
codice civile legge sentenza corte ricorso ricorrente controricorrente cassazione giudice motivo motivi tizio caio
sempronio della signor signora proprio propri propria quindi""".split())


def enabled() -> bool:
    return feature_flags.enabled(ARTICLE_RELATIONS_FLAG)


def stems(text: str) -> List[str]:
    """Content words cut to six letters (the same in the builder and here)."""
    return [w[:6] for w in re.findall(r"[a-zà-ù]{4,}", (text or "").lower()) if w not in _STOP]


def fuse(rankings: Iterable[List[Tuple[str, float]]], k: int = RRF_K) -> List[Tuple[str, float]]:
    """Reciprocal-rank fusion of ranked lists of (article, score)."""
    scores: Dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (article, _) in enumerate(ranking, 1):
            scores[article] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def profile_vectors(docs: Dict[str, Counter], keep: int = 80) -> Dict[str, dict]:
    """{"idf": {stem: idf}, "vectors": {article: {stem: weight}}}: TF-IDF, normalised, the top `keep` stems each."""
    docs = {a: c for a, c in docs.items() if c}
    df = Counter(s for c in docs.values() for s in c)
    n = max(len(docs), 1)
    idf = {s: math.log((n + 1) / (k + 1)) + 1 for s, k in df.items()}
    vectors = {}
    for a, c in docs.items():
        v = sorted(((s, (1 + math.log(tf)) * idf[s]) for s, tf in c.items()), key=lambda x: -x[1])[:keep]
        norm = math.sqrt(sum(x * x for _, x in v)) or 1.0
        vectors[a] = {s: round(x / norm, 5) for s, x in v}
    used = {s for v in vectors.values() for s in v}
    return {"idf": {s: round(w, 4) for s, w in idf.items() if s in used}, "vectors": vectors}


def rank_profiles(profile: dict, text: str) -> List[Tuple[str, float]]:
    """The articles of one profile set ranked by cosine with the text."""
    idf = profile.get("idf") or {}
    q = Counter(s for s in stems(text) if s in idf)
    qv = {s: (1 + math.log(tf)) * idf[s] for s, tf in q.items()}
    norm = math.sqrt(sum(x * x for x in qv.values())) or 1.0
    scores = []
    for article, vec in (profile.get("vectors") or {}).items():
        score = sum(w * vec.get(s, 0.0) for s, w in qv.items()) / norm
        if score > 0:
            scores.append((article, score))
    return sorted(scores, key=lambda x: (-x[1], x[0]))


class Relations:
    def __init__(self, data: dict):
        self.related: Dict[str, List[Tuple[str, float]]] = {a: [(b, float(s)) for b, s in v]
                                                             for a, v in (data.get("related") or {}).items()}
        self.via: Dict[str, Dict[str, str]] = data.get("via") or {}
        self.profiles: Dict[str, dict] = data.get("profiles") or {}
        self.built_at = data.get("built_at", "")

    def neighbours(self, seeds: List[str], exclude: Set[str]) -> List[Tuple[str, str]]:
        """(article, seed it relates to), the related articles of the seeds fused by rank, the earlier seeds first."""
        rankings = [[(b, s) for b, s in self.related.get(seed, []) if b not in exclude] for seed in seeds]
        scores = dict(fuse(rankings))
        first: Dict[str, Tuple[int, int, str]] = {}
        for i, (seed, ranking) in enumerate(zip(seeds, rankings)):
            for rank, (b, _) in enumerate(ranking):
                first.setdefault(b, (i, rank, seed))
        ordered = sorted(scores, key=lambda a: (-scores[a], first[a][0], first[a][1], a))
        return [(a, first[a][2]) for a in ordered]

    def profile_matches(self, text: str, exclude: Set[str]) -> List[str]:
        rankings = [rank_profiles(p, text) for p in self.profiles.values() if p]
        return [a for a, _ in fuse(rankings) if a not in exclude]


_store: Optional[Relations] = None
_signature: Optional[tuple] = None
_lock = threading.Lock()


def store() -> Optional[Relations]:
    """The relations of ARTICLE_RELATIONS_FILE, read again when the file changes; None without a readable file."""
    global _store, _signature
    path = os.getenv(FILE_ENV, "")
    if not path or not os.path.isfile(path):
        return None
    signature = (path, os.path.getsize(path), os.path.getmtime(path))
    with _lock:
        if signature != _signature:
            try:
                with open(path, encoding="utf-8") as fh:
                    _store, _signature = Relations(json.load(fh)), signature
                logger.info("article relations: %d articles from %s", len(_store.related), path)
            except (OSError, ValueError) as exc:
                logger.warning("article relations: %s not readable: %s", path, exc)
                return None
        return _store


def candidates(seeds: List[tuple], case_text: str, have: Set[tuple], civil: bool,
               relations: Optional[Relations] = None) -> Tuple[List[tuple], Dict[tuple, str]]:
    """(refs to append, {ref: why}): up to ADD_RELATED c.c. articles related to the c.c. seeds, then, in a civil case,
    up to ADD_PROFILE whose profiles best match the case's text -- none already in `have`. ([], {}) without
    relations or on any failure."""
    rel = relations or store()
    if rel is None:
        return [], {}
    try:
        have_cc = {n for n, c in have if c == CODE}
        seed_cc = list(dict.fromkeys(n for n, c in seeds if c == CODE))
        added: List[tuple] = []
        why: Dict[tuple, str] = {}
        for article, seed in rel.neighbours(seed_cc, have_cc | set(seed_cc))[:ADD_RELATED]:
            ref = (article, CODE)
            added.append(ref)
            via = (rel.via.get(seed) or {}).get(article, "")
            why[ref] = f"related to {seed} c.c." + (f" ({via})" if via else "")
        if civil:
            taken = have_cc | set(seed_cc) | {n for n, _ in added}
            for article in rel.profile_matches(case_text, taken)[:ADD_PROFILE]:
                ref = (article, CODE)
                added.append(ref)
                why[ref] = "rulings' profile matches the case"
        return added, why
    except Exception as exc:
        logger.warning("article relations: no candidates: %s", exc)
        return [], {}
