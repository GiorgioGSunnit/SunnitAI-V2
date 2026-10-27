"""Read-only prototype of the relations agent, Codice Civile first (27 Oct): which relations the data we already have
can build between rulings, articles and cases, and whether they connect the right things. No model call, no write.

Relations built (all deterministic):
  APPLIES     ruling -> article   the c.c. articles a ruling's card says it applies (cards_civ_v1.jsonl)
  CITES       ruling -> article   Flavio's nightly (Section)-[:CITES {code, article}]->(Section) edges, code c.c.
  CO_APPLIED  article <-> article applied together in the same rulings (from APPLIES)
  CO_CITED    article <-> article cited together in the same rulings (from CITES)
  REFERS_TO   article -> article  the article's own text names the other ("nei casi indicati dall'articolo 1490")
  NEAR        baseline: the article numbers around it
  COMBINED    reciprocal-rank fusion of CO_APPLIED, CO_CITED and REFERS_TO

Tests:
  T1 connection: for each test case with two or more expected c.c. articles, leave one out -- from one expected
     article, how many of the others come up in each relation's top 5 / 10.
  T2 case -> articles: a profile per article (the keywords of the rulings applying it; the article's own text and
     rubrica), the case's text scored against every profile, and where the expected articles rank.

Prints numbers only (article numbers, counts, scores): no ruling text, no keywords. A JSON report goes to --out.

    cd /opt/chatbot && venv/bin/python /tmp/relations/exp_relations.py \\
        --cards /root/ruling_cards/cards_civ_v1.jsonl --cases /tmp/relations/eval_narrative_cases.json
    (--no-db: cards only, no CITES, no article texts; --show 1490,2051: each relation's neighbours of those articles)
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

SUFFIXES = ("bis", "ter", "quater", "quinquies", "sexies", "septies", "octies", "novies", "nonies", "decies",
            "undecies", "duodecies", "terdecies", "quaterdecies", "quinquiesdecies", "sexiesdecies",
            "septiesdecies", "octiesdecies")
_SUFFIX_RE = "|".join(sorted(SUFFIXES, key=len, reverse=True))
_KEY_RE = re.compile(rf"^(\d+)(?:\s*-?\s*({_SUFFIX_RE}))?$", re.IGNORECASE)
# Articles every kind of civil ruling applies (burden of proof, res judicata): never a neighbour (vote_pool.BOILERPLATE)
BOILERPLATE = {"2697", "2909"}
# An article applied by more than this share of the rulings says nothing about what a ruling is about (--generic-share;
# the cosine and lift rankings already weigh common articles down, so the cut is only for the very common ones).
GENERIC_SHARE = 0.10
EXTRA_CASES = {
    # The lawyer's two cases (26 Oct): the c.c. articles a correct answer rests on (our reading of the review).
    "barca": ["1490", "1491", "1492", "1494", "1495", "1497", "1512"],
    "piscina": ["2043", "2050", "2051", "2055", "2056", "2059", "2947"],
}


def key(number) -> Optional[str]:
    """'1490' / '337 ter' / '337-TER' / '1490.0.0' -> '1490' / '337-ter'; None for anything else."""
    text = str(number or "").strip().lower().split(".")[0].split("_")[0]
    m = _KEY_RE.match(text)
    if not m:
        return None
    return m.group(1).lstrip("0") + (f"-{m.group(2).lower()}" if m.group(2) else "") if m.group(1).strip("0") else None


def number_of(article: str) -> int:
    return int(re.match(r"\d+", article).group(0))


# --- relations ---------------------------------------------------------------------------------

def card_articles(card: dict, code: str = "c.c.") -> Set[str]:
    """The card's applied articles of `code`, as keys."""
    out = set()
    for ref in card.get("articoli_applicati") or []:
        ref = " ".join(str(ref).split())
        if ref.endswith(" " + code):
            k = key(ref[:-len(code) - 1])
            if k:
                out.add(k)
    return out


class CoRelation:
    """Articles that occur together in the same rulings: pair counts and three ways to rank neighbours."""

    def __init__(self, groups: Iterable[Set[str]], exclude: Set[str] = frozenset()):
        self.n = 0
        self.single: Counter = Counter()
        self.pair: Dict[str, Counter] = defaultdict(Counter)
        for group in groups:
            group = {a for a in group if a not in exclude}
            if not group:
                continue
            self.n += 1
            for a in group:
                self.single[a] += 1
                for b in group:
                    if a != b:
                        self.pair[a][b] += 1

    def neighbours(self, article: str, how: str = "cosine", min_support: int = 2) -> List[Tuple[str, float]]:
        out = []
        na = self.single.get(article, 0)
        for b, nab in self.pair.get(article, {}).items():
            if nab < min_support:
                continue
            nb = self.single[b]
            if how == "count":
                score = float(nab)
            elif how == "lift":
                score = nab * self.n / (na * nb)
            else:
                score = nab / math.sqrt(na * nb)
            out.append((b, score))
        return sorted(out, key=lambda x: (-x[1], number_of(x[0]), x[0]))


_REF_RE = re.compile(
    rf"\b(?:articol[oi]|artt?\.)\s+((?:da\s+)?\d+(?:\s*-?\s*(?:{_SUFFIX_RE}))?(?:\s*(?:,|e|ed|a|-)\s*\d+(?:\s*-?\s*(?:{_SUFFIX_RE}))?)*)",
    re.IGNORECASE)
_OTHER_CODE_RE = re.compile(
    r"^[\s,;.]{0,3}(?:del(?:la|l')?\s+)?(?:c\.?\s?p\.?\s?c|codice di procedura|procedura civile|c\.?\s?p\b|codice penale|"
    r"legge|l\.\s*\d|d\.\s?lgs|decreto|r\.\s?d\.|d\.\s?p\.\s?r|costituzione|disp(?:osizioni)?\.?\s+(?:di\s+)?att|"
    r"regolamento|codice della navigazione|codice del consumo|testo unico|t\.\s?u\.)", re.IGNORECASE)


def text_refs(text: str, own: str) -> Set[str]:
    """The c.c. articles an article's text names ("articoli 1491 e 1492", "artt. da 1490 a 1495"), not those of
    another code or law named right after them, never itself."""
    out = set()
    for m in _REF_RE.finditer(text or ""):
        if _OTHER_CODE_RE.match(text[m.end():m.end() + 45]):
            continue
        group = m.group(1).strip()
        span = re.match(r"(?:da\s+(\d+)\s+a\s+(\d+)|(\d+)\s*-\s*(\d+))$", group, re.IGNORECASE)
        lo, hi = ((int(span.group(1) or span.group(3)), int(span.group(2) or span.group(4))) if span else (0, 0))
        if span and 0 < hi - lo <= 30:
            found = [str(n) for n in range(lo, hi + 1)]
        else:
            # "a" or "-" that is not a range of articles ("articolo 2 a 5 anni"): only what comes before it
            group = re.split(r"\s+a\s+|\s*-\s*(?=\d)", group, maxsplit=1, flags=re.IGNORECASE)[0]
            found = re.findall(rf"\d+(?:\s*-?\s*(?:{_SUFFIX_RE}))?", group, re.IGNORECASE)
        for raw in found:
            k = key(raw)
            if k and k != own:
                out.add(k)
    return out


class RefersTo:
    """REFERS_TO edges among the code's articles: neighbours are the articles it names (2) and those naming it (1)."""

    def __init__(self, refs: Dict[str, Set[str]], articles: Set[str]):
        self.out = {a: {b for b in bs if b in articles} for a, bs in refs.items()}
        self.into: Dict[str, Set[str]] = defaultdict(set)
        for a, bs in self.out.items():
            for b in bs:
                self.into[b].add(a)

    def neighbours(self, article: str) -> List[Tuple[str, float]]:
        scores: Dict[str, float] = {}
        for b in self.out.get(article, ()):
            scores[b] = scores.get(b, 0) + 2
        for b in self.into.get(article, ()):
            scores[b] = scores.get(b, 0) + 1
        seed = number_of(article)
        return sorted(scores.items(), key=lambda x: (-x[1], abs(number_of(x[0]) - seed), x[0]))

    def edges(self) -> int:
        return sum(len(v) for v in self.out.values())


def near(article: str, articles: Iterable[str], k: int = 50) -> List[Tuple[str, float]]:
    seed = number_of(article)
    ranked = sorted((a for a in articles if a != article), key=lambda a: (abs(number_of(a) - seed), a))
    return [(a, 1.0 / (1 + abs(number_of(a) - seed))) for a in ranked[:k]]


def fuse(rankings: List[List[Tuple[str, float]]], k: int = 60) -> List[Tuple[str, float]]:
    """Reciprocal-rank fusion of several neighbour lists."""
    scores: Dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (article, _) in enumerate(ranking, 1):
            scores[article] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], number_of(x[0]), x[0]))


# --- T1: connection test -------------------------------------------------------------------------

def connection_test(cases: Dict[str, List[str]], relations: Dict[str, callable], ks=(5, 10)) -> Dict[str, dict]:
    """For each relation: the mean share of a case's other expected articles found in the top k neighbours of one
    expected article (leave one out, averaged over the seeds, then over the cases), the share of seeds with at least
    one found in the top 10, and the share of seeds with any neighbour at all."""
    report = {}
    for name, neighbours in relations.items():
        per_case, any10, covered, seeds = [], 0, 0, 0
        for case, expected in cases.items():
            shares = {k: [] for k in ks}
            for seed in expected:
                others = set(expected) - {seed}
                ranked = [a for a, _ in neighbours(seed)]
                seeds += 1
                covered += bool(ranked)
                for k in ks:
                    shares[k].append(len(others & set(ranked[:k])) / len(others))
                any10 += bool(others & set(ranked[:10]))
            per_case.append({k: sum(v) / len(v) for k, v in shares.items()})
        report[name] = {**{f"hit@{k}": round(100 * sum(c[k] for c in per_case) / len(per_case), 1) for k in ks},
                        "seeds_with_a_hit@10": round(100 * any10 / seeds, 1),
                        "seeds_with_neighbours": round(100 * covered / seeds, 1)}
    return report


def connection_by_case(cases: Dict[str, List[str]], neighbours, k: int = 10) -> Dict[str, str]:
    """Per case, for one relation: 'found/possible' pairs in the top k (each ordered pair seed -> other)."""
    out = {}
    for case, expected in cases.items():
        found = sum(len((set(expected) - {s}) & {a for a, _ in neighbours(s)[:k]}) for s in expected)
        out[case] = f"{found}/{len(expected) * (len(expected) - 1)}"
    return out


# --- T2: case -> articles ------------------------------------------------------------------------

_STOP = set("""
alla alle allo anche ancora avere aveva base caso cioè come con contro cosa così dalla dalle dallo dati degli della
delle dello dell detto dopo dove essere fatto fatti fine gli hanno il infatti invece loro mentre nella nelle nello
nonché oltre ogni ovvero parte perché però poiché presente prima quale quali quando quanto quella quelle quello questa
queste questo sono stata stati stato sulla sulle tale tali tanto tutte tutti una uno viene articolo articoli comma
codice civile legge sentenza corte ricorso ricorrente controricorrente cassazione giudice motivo motivi tizio caio
sempronio della signor signora proprio propri propria quindi""".split())


def stems(text: str) -> List[str]:
    return [w[:6] for w in re.findall(r"[a-zà-ù]{4,}", (text or "").lower()) if w not in _STOP]


class Profiles:
    """TF-IDF profiles of articles; a text scored against all of them by cosine."""

    def __init__(self, docs: Dict[str, Counter]):
        self.docs = {a: c for a, c in docs.items() if c}
        df = Counter(s for c in self.docs.values() for s in c)
        n = max(len(self.docs), 1)
        self.idf = {s: math.log((n + 1) / (k + 1)) + 1 for s, k in df.items()}
        self.vec = {}
        for a, c in self.docs.items():
            v = {s: (1 + math.log(tf)) * self.idf[s] for s, tf in c.items()}
            norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
            self.vec[a] = {s: x / norm for s, x in v.items()}

    def rank(self, text: str) -> List[Tuple[str, float]]:
        q = Counter(stems(text))
        qv = {s: (1 + math.log(tf)) * self.idf[s] for s, tf in q.items() if s in self.idf}
        norm = math.sqrt(sum(x * x for x in qv.values())) or 1.0
        scores = [(a, sum(qv[s] * v.get(s, 0.0) for s in qv) / norm) for a, v in self.vec.items()]
        return sorted([x for x in scores if x[1] > 0], key=lambda x: (-x[1], x[0]))


def retrieval_test(questions: Dict[str, str], cases: Dict[str, List[str]], rankers: Dict[str, callable],
                   ks=(10, 20, 50)) -> Dict[str, dict]:
    """For each ranker: the share of expected articles in the top k for the case's text, and the mean reciprocal rank."""
    report = {}
    for name, ranker in rankers.items():
        hits = {k: 0 for k in ks}
        rr, total, per_case = 0.0, 0, {}
        for case, expected in cases.items():
            if case not in questions:
                continue
            ranked = [a for a, _ in ranker(questions[case])]
            pos = {a: i + 1 for i, a in enumerate(ranked)}
            ranks = [pos.get(a) for a in expected]
            per_case[case] = [r if r else "-" for r in ranks]
            for r in ranks:
                total += 1
                rr += 1.0 / r if r else 0.0
                for k in ks:
                    hits[k] += bool(r and r <= k)
        report[name] = {**{f"recall@{k}": round(100 * hits[k] / max(total, 1), 1) for k in ks},
                        "mrr": round(rr / max(total, 1), 3), "ranks": per_case}
    return report


# --- data ----------------------------------------------------------------------------------------

def read_cards(path: str, branch: str = "civile") -> List[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                card = json.loads(line)
            except ValueError:
                continue
            if not card.get("errore") and (card.get("branch") or branch) == branch:
                out.append(card)
    return out


def read_cases(path: str) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """{case: [expected c.c. articles]} for the cases with two or more, and {case: question}."""
    data = json.load(open(path, encoding="utf-8"))
    items = data if isinstance(data, list) else (data.get("cases") or list(data.values()))
    cases, questions = {}, {}
    for c in items:
        if not isinstance(c, dict):
            continue
        arts = [key(e.get("article")) for e in c.get("expected") or [] if isinstance(e, dict) and e.get("code") == "cc"
                and e.get("article")]
        arts = list(dict.fromkeys(a for a in arts if a))
        questions[c["id"]] = c.get("question") or ""
        if len(arts) >= 2:
            cases[c["id"]] = arts
    return cases, questions


def code_articles(session, prefix: str = "Codice Civile") -> Tuple[Dict[str, str], Dict[str, str]]:
    """({article: text}, {article: rubrica}) of the code's primary document, the text as the app reads it: "N.0.0"
    unless it holds only its date in force, else the bare "N" without its date, else the commi "N.k.0" in order."""
    docs = session.run("MATCH (d:Document) WHERE d.document_type = 'primary' AND d.name STARTS WITH $p "
                       "RETURN d.id AS id, d.name AS name", p=prefix).data()
    if not docs:
        return {}, {}
    doc = min(docs, key=lambda d: len(d["name"] or ""))
    rows = session.run("MATCH (d:Document {id: $id})-[:CONTAINS]->(s:Section) RETURN s.name AS name, "
                       "s.plain_text AS text, coalesce(s.titolo_articolo, s.title, '') AS heading", id=doc["id"]).data()
    by_article: Dict[str, Dict[str, str]] = defaultdict(dict)
    headings: Dict[str, str] = {}
    for r in rows:
        name = str(r["name"] or "")
        k = key(name)
        if not k:
            continue
        text = " ".join(str(r["text"] or "").split())
        if len(text) > len(by_article[k].get(name, "")):
            by_article[k][name] = text
        if r["heading"] and k not in headings:
            headings[k] = " ".join(str(r["heading"]).split())
    in_force = re.compile(r"^\s*in vigore dal\s+\d{1,2}\s+\w+\s+\d{4}\.?\s*", re.IGNORECASE)
    texts = {}
    for k, parts in by_article.items():
        zero = parts.get(f"{k}.0.0", "")
        if zero and not in_force.fullmatch(zero):
            texts[k] = zero
            continue
        bare = in_force.sub("", parts.get(k, ""), count=1).strip()
        if bare:
            texts[k] = bare
            continue
        commi = sorted((int(m.group(1)), t) for n, t in parts.items() for m in [re.match(rf"^{re.escape(k)}\.(\d+)\.0$", n)]
                       if m and int(m.group(1)) > 0)
        texts[k] = "\n".join(t for _, t in commi) or " ".join(parts.values())
    return texts, headings


def cites_groups(session, code: str = "c.c.", batch: int = 2000, pause: float = 0.5) -> Dict[str, Set[str]]:
    """{ruling id: c.c. articles its CITES edges point to}, read in batches of rulings (light on be2)."""
    ids = [r["id"] for r in session.run("MATCH (d:Document {document_type:'interpretation'}) RETURN d.id AS id").data()]
    out: Dict[str, Set[str]] = {}
    for start in range(0, len(ids), batch):
        if start:
            time.sleep(pause)
        for r in session.run("MATCH (d:Document)-[:CONTAINS]->(:Section)-[c:CITES]->() WHERE d.id IN $ids AND c.code = $code "
                             "RETURN d.id AS id, collect(DISTINCT c.article) AS arts", ids=ids[start:start + batch],
                             code=code).data():
            arts = {k for k in (key(a) for a in r["arts"]) if k}
            if arts:
                out[r["id"]] = arts
    return out


# --- main ----------------------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cards", default="/root/ruling_cards/cards_civ_v1.jsonl")
    p.add_argument("--cases", default="tests/eval_narrative_cases.json")
    p.add_argument("--env", default="/opt/chatbot/.env")
    p.add_argument("--no-db", action="store_true", help="cards only: no CITES, no article texts")
    p.add_argument("--show", default="1490,2051,1218,2043,1453,1117", help="articles whose neighbours to print")
    p.add_argument("--out", default="/tmp/relations_report.json")
    p.add_argument("--generic-share", type=float, default=GENERIC_SHARE,
                   help="articles applied by more than this share of the rulings are left out (default 0.10)")
    args = p.parse_args()
    t0 = time.time()

    cards = read_cards(args.cards)
    applied = [card_articles(c) for c in cards]
    with_cc = [a for a in applied if a]
    freq = Counter(a for g in with_cc for a in g)
    generic = {a for a, k in freq.items() if k > args.generic_share * len(with_cc)} | BOILERPLATE
    print(f"civil cards: {len(cards):,}; applying at least one c.c. article: {len(with_cc):,} "
          f"({100 * len(with_cc) / max(len(cards), 1):.0f}%); APPLIES edges: {sum(len(g) for g in with_cc):,}; "
          f"distinct c.c. articles applied: {len(freq):,}")
    print("most applied: " + ", ".join(f"{a} {k:,}" for a, k in freq.most_common(15)))
    print(f"generic (left out of the co-relations): {', '.join(sorted(generic, key=number_of))}")

    co_applied = CoRelation(with_cc, exclude=generic)
    relations = {"NEAR": None, "CO_APPLIED count": lambda a: co_applied.neighbours(a, "count"),
                 "CO_APPLIED lift": lambda a: co_applied.neighbours(a, "lift"),
                 "CO_APPLIED cosine": lambda a: co_applied.neighbours(a, "cosine")}
    texts, headings, co_cited, refers = {}, {}, None, None
    if not args.no_db:
        from dotenv import load_dotenv
        from neo4j import GraphDatabase, READ_ACCESS
        load_dotenv(args.env)
        driver = GraphDatabase.driver(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")))
        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j"), default_access_mode=READ_ACCESS) as session:
            texts, headings = code_articles(session)
            print(f"Codice Civile articles read: {len(texts):,} ({time.time() - t0:.0f}s)")
            cited = cites_groups(session)
            print(f"rulings with c.c. CITES: {len(cited):,} ({time.time() - t0:.0f}s)")
        driver.close()
        co_cited = CoRelation(cited.values(), exclude=generic)
        refers = RefersTo({a: text_refs(t, a) for a, t in texts.items()}, set(texts))
        print(f"REFERS_TO edges among c.c. articles: {refers.edges():,}; articles naming another: "
              f"{sum(1 for v in refers.out.values() if v):,}")
        relations["CO_CITED cosine"] = lambda a: co_cited.neighbours(a, "cosine")
        relations["REFERS_TO"] = refers.neighbours
        relations["COMBINED"] = lambda a: fuse([co_applied.neighbours(a, "cosine"), co_cited.neighbours(a, "cosine"),
                                                refers.neighbours(a)])
    else:
        relations["COMBINED"] = lambda a: co_applied.neighbours(a, "cosine")
    universe = set(texts) or set(freq)
    relations["NEAR"] = lambda a: near(a, universe)

    cases, questions = read_cases(args.cases)
    extra = {k: v for k, v in EXTRA_CASES.items()}
    report = {"cards": len(cards), "cards_with_cc": len(with_cc), "generic": sorted(generic, key=number_of)}

    print(f"\n== T1 connection: {len(cases)} test cases with 2+ expected c.c. articles "
          f"({', '.join(cases)}), leave one out")
    t1 = connection_test(cases, relations)
    _table(t1)
    report["t1_tests"] = t1
    print(f"\n== T1 on the lawyer's two cases ({', '.join(extra)}; expected = our reading of the review)")
    t1x = connection_test(extra, relations)
    _table(t1x)
    report["t1_lawyer"] = t1x
    best = "COMBINED"
    per = connection_by_case({**cases, **extra}, relations[best])
    print(f"\n{best}, pairs found in the top 10 per case: " + ", ".join(f"{c} {v}" for c, v in per.items()))
    report["t1_by_case_combined"] = per

    rankers = {}
    ruling_docs: Dict[str, Counter] = defaultdict(Counter)
    for card, arts in zip(cards, applied):
        words = " ".join(str(k) for k in (card.get("argomento_decisivo") or []) + (card.get("parole_chiave_questioni") or [])
                         + (card.get("parole_chiave_fatti") or []))
        for a in arts - generic:
            ruling_docs[a].update(stems(words))
    by_rulings = Profiles(ruling_docs)
    rankers["profile from rulings"] = by_rulings.rank
    if texts:
        by_text = Profiles({a: Counter(stems(f"{headings.get(a, '')} {headings.get(a, '')} {t}")) for a, t in texts.items()})
        rankers["profile from article text"] = by_text.rank
        rankers["both"] = lambda q: fuse([by_rulings.rank(q), by_text.rank(q)])
    t2_cases = {c: [a for a in arts] for c, arts in cases.items()}
    t2_cases.update({c: [a] for c, a in (_single_expected(args.cases) or {}).items() if c not in t2_cases})
    print(f"\n== T2 case -> articles: {len(t2_cases)} cases with expected c.c. articles, the case's text ranked "
          f"against every article profile (rulings: {len(by_rulings.vec):,} articles"
          + (f", text: {len(texts):,})" if texts else ")"))
    t2 = retrieval_test(questions, t2_cases, rankers)
    for name, r in t2.items():
        print(f"  {name:28} " + "  ".join(f"{k} {v}" for k, v in r.items() if k != "ranks"))
    for name, r in t2.items():
        print(f"  ranks, {name}: " + "; ".join(f"{c} {'/'.join(map(str, v))}" for c, v in r["ranks"].items()))
    report["t2"] = t2

    shown = {}
    for a in [k for k in (key(x) for x in args.show.split(",")) if k]:
        shown[a] = {}
        print(f"\n-- neighbours of {a} c.c. (top 8)")
        for name, fn in relations.items():
            top = [n for n, _ in fn(a)[:8]]
            shown[a][name] = top
            print(f"  {name:20} {', '.join(top) or '-'}")
    report["shown"] = shown
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print(f"\nreport: {args.out} ({time.time() - t0:.0f}s)")


def _single_expected(path: str) -> Dict[str, str]:
    """Cases with exactly one expected c.c. article (T2 only)."""
    data = json.load(open(path, encoding="utf-8"))
    items = data if isinstance(data, list) else (data.get("cases") or list(data.values()))
    out = {}
    for c in items:
        arts = [key(e.get("article")) for e in c.get("expected") or [] if isinstance(e, dict) and e.get("code") == "cc"
                and e.get("article")]
        arts = [a for a in arts if a]
        if len(arts) == 1:
            out[c["id"]] = arts[0]
    return out


def _table(report: Dict[str, dict]) -> None:
    for name, r in report.items():
        print(f"  {name:20} " + "  ".join(f"{k} {v}%" for k, v in r.items()))


if __name__ == "__main__":
    sys.exit(main())
