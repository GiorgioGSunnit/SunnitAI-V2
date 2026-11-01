"""Article cards in bulk: Qwen3 without thinking, into a JSONL file. Read-only on Neo4j; run from the app folder:

    venv/bin/python scripts/article_cards_bulk.py --dry-run                                # the order, no model call
    nohup venv/bin/python scripts/article_cards_bulk.py --doc "Codice del Consumo 2026" --max-per-hour 600 \
        > article_cards_bulk.log 2>&1 &
    venv/bin/python scripts/article_cards_bulk.py --sample 40                              # N cards by code, for review

ORDER    (a) the --top (500) articles most often in the pools and plans of diagnose_long_runs/*/cases/*.json; (b) every
         article of c.c., c.p., c.p.c., c.p.p. (each code's primary document with most headed Sections); (c) each --doc.
RUNNING  --concurrency parallel calls (6), at most --max-per-hour started. Stop it at any time: each card is appended as it
         is made, and the next run goes on. A card is made again only when its text_hash changes, or if it failed (--force:
         all). A text that is only "(abrogato)" gets no card.
STORE    --file (data/article_cards.jsonl): doc, article, card, model, text_hash, created_at, checks; one line per article
         once the run ends (until then the last line of an article wins, for the reader too).
CHECKS   numbers_not_in_text, abrogated_as_live (a card line, without the word "abrogato", repeating 6 words in a row of
         bracketed or "(abrogato)" text), renumbered (an item or comma label the text does not have), empty_fields, cut,
         failed. finish_reason=length: once more, shorter, then failed. Prompts to look, not verdicts; but the chatbot gives
         the writer only cards with none (src/rag/article_cards.py), the official text otherwise. Summed up at the end.
SAMPLE   --sample N: N random cards, shared out by code, each beside its text, in markdown (--out).
"""

import argparse
import glob
import importlib.util
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("article_cards_prompt", os.path.join(HERE, "article_cards.py"))
ac = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ac)

DEFAULT_FILE = os.path.join("data", "article_cards.jsonl")
TOP = 500
SHORTER = "\n\nLa risposta precedente è stata troncata: sii più compatto, senza omettere commi o voci."
_ARTICLE_NAME_RE = re.compile(r"^(\d+(?:-[a-z]+)?)(?:[._]|$)", re.IGNORECASE)
_LABEL_RE = re.compile(r"^(?:art\.?\s*)?(\d+(?:-[a-z]+)?)\s+(.+)$", re.IGNORECASE)
_ITEM_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:-[a-z]+)?|[a-z](?:-[a-z]+)?)\)")
_COMMA_HEAD_RE = re.compile(r"(?:^|\s)(\d{1,3}(?:-[a-z]+)?)\.\s+[A-ZÀ-Ü]")
_CARD_COMMA_RE = re.compile(r"(?:\bcomma|\bco\.)\s*(\d{1,3}(?:-[a-z]+)?)\b(?!\s+(?:dell|del\b|della))", re.IGNORECASE)


def text_hash(text: str) -> str:
    from src.rag.article_cards import text_hash as shared
    return shared(text)


# --- the order ------------------------------------------------------------------------------------------------------

def frequency(paths: list) -> Counter:
    """{(doc or None, number, code label): how often}: each case's pool and plan count once each. The plan's articles take
    the document their label has in that case's pool."""
    counts = Counter()
    for path in paths:
        try:
            with open(path, encoding="utf-8") as fh:
                rec = json.load(fh)
        except (OSError, ValueError):
            continue
        docs = {}
        for row in rec.get("pool") or []:
            match = _LABEL_RE.match(str(row.get("label") or "").strip())
            if match:
                key = (match.group(1).lower(), match.group(2).strip())
                docs[key] = row.get("document") or None
                counts[(docs[key], *key)] += 1
        for label in (rec.get("plan") or {}).get("norme") or []:
            match = _LABEL_RE.match(str(label).strip())
            if match:
                key = (match.group(1).lower(), match.group(2).strip())
                counts[(docs.get(key), *key)] += 1
    return counts


def order_refs(top: list, codes: dict, specials: dict) -> list:
    """[(doc, number)]: the top ones, then the codes', then the --doc ones, each once."""
    seen, out = set(), []
    for doc, number in list(top) + [(d, n) for group in (codes, specials) for d, numbers in group.items() for n in numbers]:
        if (doc, number) not in seen:
            seen.add((doc, number))
            out.append((doc, number))
    return out


def number_key(number: str) -> tuple:
    match = re.match(r"(\d+)(.*)", number)
    return (int(match.group(1)), match.group(2)) if match else (0, number)


# --- the checks -----------------------------------------------------------------------------------------------------

def abrogated_segments(text: str) -> list:
    return re.findall(r"\[[^\[\]]{30,}\]", text) + [s for s in re.findall(r"[^.;\n]{30,}\(abrogato\)", text, re.IGNORECASE)]


def abrogated_as_live(card: str, text: str) -> list:
    grams = set()
    for segment in abrogated_segments(text):
        words = re.findall(r"\w+", segment.lower())
        grams |= {tuple(words[i:i + 6]) for i in range(len(words) - 5)}
    out = []
    for line in card.splitlines():
        words = re.findall(r"\w+", line.lower())
        if "abrogat" not in line.lower() and any(tuple(words[i:i + 6]) in grams for i in range(len(words) - 5)):
            out.append(line.strip()[:160])
    return out


def renumbered(card: str, text: str) -> list:
    """Labels in the card the text does not have: items always; commi when the text numbers its own."""
    out = set(_ITEM_RE.findall(card)) - set(_ITEM_RE.findall(text))
    commi = set(_COMMA_HEAD_RE.findall(text))
    if commi:
        out |= set(_CARD_COMMA_RE.findall(card)) - commi - set(_ITEM_RE.findall(text))
    return sorted(out)


def empty_fields(card: str) -> list:
    out = []
    for i, heading in enumerate(ac.HEADINGS):
        match = re.search(rf"\*\*{re.escape(heading)}\s*:?\*\*:?", card)
        if not match:
            out.append(heading)
            continue
        ahead = re.search(r"\*\*(?:" + "|".join(map(re.escape, ac.HEADINGS[i + 1:])) + r")\s*:?\*\*", card[match.end():]) \
            if i + 1 < len(ac.HEADINGS) else None
        if not card[match.end():match.end() + ahead.start() if ahead else len(card)].strip(" \n:-*"):
            out.append(heading)
    return out


def checks(card: str, article: dict) -> dict:
    source = f"{article['number']} {article['code']} {article['rubrica']} {article['text']}"
    have = set(re.findall(r"\d+", source))
    return {"numbers_not_in_text": list(dict.fromkeys(n for n in re.findall(r"\d+", card) if n not in have)),
            "abrogated_as_live": abrogated_as_live(card, article["text"]),
            "renumbered": renumbered(card, article["text"]), "empty_fields": empty_fields(card)}


# --- the run --------------------------------------------------------------------------------------------------------

class Throttle:
    """At most per_hour starts an hour, evenly spaced; 0: no limit."""

    def __init__(self, per_hour: int, clock=time.monotonic, sleep=time.sleep):
        self.gap, self.clock, self.sleep, self.lock, self.next = 3600 / per_hour if per_hour else 0, clock, sleep, threading.Lock(), 0.0

    def wait(self) -> None:
        if not self.gap:
            return
        with self.lock:
            now = self.clock()
            start = max(now, self.next)
            self.next = start + self.gap
        if start > now:
            self.sleep(start - now)


def read_file(path: str) -> dict:
    """{(doc, article): record}, the last line of each."""
    out = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                    out[(record["doc"], record["article"])] = record
                except (ValueError, KeyError, TypeError):
                    continue
    return out


def compact(path: str, records: dict) -> None:
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        for record in records.values():
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(path + ".tmp", path)


def due(article: dict, doc: str, have: dict, force: bool = False) -> bool:
    record = have.get((doc, article["number"]))
    return force or not record or record.get("text_hash") != text_hash(article["text"]) \
        or bool((record.get("checks") or {}).get("failed"))


def make(endpoint: dict, doc: str, article: dict) -> dict:
    """The file's line for the article. Cut: once more, shorter; then failed. Never raises for a bad reply."""
    user = ac.user_message(article)
    found = {"numbers_not_in_text": [], "abrogated_as_live": [], "renumbered": [], "empty_fields": [], "cut": False, "failed": ""}
    card = ""
    try:
        try:
            card = ac.call(endpoint, user)
        except ValueError as exc:
            if "cut at max_tokens" not in str(exc):
                raise
            card = ac.call(endpoint, user + SHORTER)
    except ValueError as exc:
        found["cut"] = "cut at max_tokens" in str(exc)
        found["failed"] = str(exc)[:300]
    if card:
        found.update(checks(card, article))
    return {"doc": doc, "article": article["number"], "card": card, "model": endpoint["model"],
            "text_hash": text_hash(article["text"]), "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "checks": found}


def summary(records: list) -> list:
    counts = Counter(k for r in records for k, v in (r.get("checks") or {}).items() if v)
    clean = sum(1 for r in records if not any((r.get("checks") or {}).values()))
    L = [f"{len(records):,} cards in the file: {clean:,} with no check raised ({clean * 100 // max(len(records), 1)}%), the only "
         "ones the chatbot would use"]
    L += [f"  {name:<22}{counts.get(name, 0):>7,}" for name in
          ("numbers_not_in_text", "abrogated_as_live", "renumbered", "empty_fields", "cut", "failed")]
    return L


def run(items: list, endpoint: dict, path: str, have: dict, concurrency: int, throttle: Throttle, force: bool = False) -> dict:
    """items: [(doc, article)]. Appends each line as it is made; {counts}."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    todo = [(doc, a) for doc, a in items if due(a, doc, have, force)]
    lock = threading.Lock()
    done = Counter()
    started = time.time()

    def one(item):
        doc, article = item
        throttle.wait()
        record = make(endpoint, doc, article)
        with lock:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            have[(doc, article["number"])] = record
            done["failed" if record["checks"]["failed"] else "made"] += 1
            total = done["made"] + done["failed"]
            if total % 25 == 0 or total == len(todo):
                print(f"{total:,}/{len(todo):,} ({done['failed']} failed), {total * 3600 / max(time.time() - started, 1):,.0f} an "
                      f"hour, last: art. {article['number']} {article['code']}", flush=True)
    with ThreadPoolExecutor(max_workers=max(concurrency, 1)) as pool:
        list(pool.map(one, todo))
    return {"due": len(todo), **done}


# --- the corpus -----------------------------------------------------------------------------------------------------

def code_document(session, code: str) -> str:
    rows = session.run("MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE d.document_type = 'primary' AND d.name STARTS WITH "
                       "$prefix AND coalesce(s.titolo_articolo, '') <> '' RETURN d.name AS name, count(s) AS n ORDER BY n DESC "
                       "LIMIT 1", prefix=ac.CODE_PREFIXES[code]).data()
    return rows[0]["name"] if rows else ""


def read_document(session, name: str, code: str) -> dict:
    """{number: {"number", "code", "rubrica", "text"}} for a document, all its Sections read once."""
    from src.rag import article_text
    rows = session.run("MATCH (d:Document {name: $name})-[:CONTAINS]->(s:Section) OPTIONAL MATCH (s)-[:NEXT]->(after:Section) "
                       "RETURN s {.name, .type, .plain_text, .titolo_articolo} AS s, after.name AS next, d.id AS doc",
                       name=name).data()
    grouped = defaultdict(list)
    for row in rows:
        match = _ARTICLE_NAME_RE.match(row["s"].get("name") or "")
        if match:
            grouped[match.group(1).lower()].append({"d": {"id": row["doc"]}, "s": row["s"], "next": row["next"]})
    out = {}
    for number, mine in grouped.items():
        found = article_text.article_rows(number, mine)
        section = (found[0].get("s") or {}) if found else {}
        text = "\n".join(((r.get("s") or {}).get("plain_text") or "").strip() for r in found).strip()
        if text:
            out[number] = {"number": number, "code": code, "text": text,
                           "rubrica": (section.get("titolo_articolo") or section.get("title") or "").strip()}
    return out


def has_body(article: dict) -> bool:
    return " ".join(article["text"].split()).strip(" .").lower() not in ("(abrogato)", "abrogato")


def sample(records: list, n: int, seed: int) -> list:
    """About n records, shared out by document in proportion to its cards, at least one each."""
    by_doc = defaultdict(list)
    for record in records:
        by_doc[record["doc"]].append(record)
    rng = random.Random(seed)
    out = []
    for doc in sorted(by_doc):
        mine = by_doc[doc]
        out += rng.sample(mine, min(len(mine), max(1, round(n * len(mine) / len(records)))))
    rng.shuffle(out)
    return out[:n]


def render_sample(chosen: list, texts: dict) -> str:
    L = ["# Schede: campione per la revisione", "", f"{len(chosen)} schede casuali, ripartite per documento.", ""]
    for i, record in enumerate(chosen, 1):
        article = texts.get((record["doc"], record["article"])) or {}
        stale = article and text_hash(article["text"]) != record["text_hash"]
        raised = {k: v for k, v in (record.get("checks") or {}).items() if v}
        L += ["---", "", f"## {i}. Art. {record['article']} - {record['doc']}" + (f" - {article['rubrica']}" if article.get("rubrica") else ""),
              "", "### Testo ufficiale", ""]
        L += [f"> {line}" if line.strip() else ">" for line in (article.get("text") or "(non trovato)").split("\n")]
        if stale:
            L += ["", "*Attenzione: il testo è cambiato dopo la scheda.*"]
        L += ["", f"### Scheda ({record['model']})", "", re.sub(r"\n+(?=\*\*)", "\n\n", record.get("card") or "*Scheda non generata.*"), "",
              "*Controlli: " + ("; ".join(f"{k}: {v}" for k, v in raised.items()) if raised else "nessuna segnalazione") + ".*", ""]
    return "\n".join(L)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--file", default=DEFAULT_FILE)
    parser.add_argument("--runs", default=os.path.join("diagnose_long_runs", "*", "cases", "*.json"))
    parser.add_argument("--top", type=int, default=TOP)
    parser.add_argument("--doc", action="append", default=[], help="a special law, by its exact document name (repeatable)")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--max-per-hour", type=int, default=0, help="0: no limit")
    parser.add_argument("--limit", type=int, default=0, help="only the first N articles of the order")
    parser.add_argument("--force", action="store_true", help="make every card again")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default=os.path.join("data", "article_cards_sample.md"))
    parser.add_argument("--url", help="Qwen3's base URL (default: LLM_BASE_URL on port 8010)")
    parser.add_argument("--model", default="qwen3-32b-awq")
    parser.add_argument("--env", default=os.path.join(os.getcwd(), ".env"))
    args = parser.parse_args()
    sys.path.insert(0, os.getcwd())
    from dotenv import load_dotenv
    load_dotenv(args.env)
    uri = os.getenv("NEO4J_URI")
    if not uri:
        sys.exit(f"NEO4J_URI is not set ({args.env})")
    have = read_file(args.file)
    import neo4j
    with neo4j.GraphDatabase.driver(uri, auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))) as driver:
        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j"), default_access_mode=neo4j.READ_ACCESS) as session:
            code_of = {code_document(session, code): code for code in ac.CODE_PREFIXES}
            code_of.pop("", None)
            if args.sample:
                chosen = sample(list(have.values()), args.sample, args.seed)
                texts = {(doc, n): a for doc in {r["doc"] for r in chosen}
                         for n, a in read_document(session, doc, code_of.get(doc, doc)).items()}
                os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
                with open(args.out, "w", encoding="utf-8") as fh:
                    fh.write(render_sample(chosen, texts))
                print(f"File: {os.path.abspath(args.out)}")
                return
            documents = {doc: read_document(session, doc, code) for doc, code in code_of.items()}
            by_code = {code: doc for doc, code in code_of.items()}
            top, counted = [], frequency(glob.glob(args.runs))
            for (doc, number, label), _ in counted.most_common():
                doc = doc or by_code.get(label)
                if doc and doc not in documents:
                    documents[doc] = read_document(session, doc, label)
                if doc and number in documents[doc] and (doc, number) not in top:
                    top.append((doc, number))
                if len(top) >= args.top:
                    break
            for doc in args.doc:
                if doc not in documents:
                    documents[doc] = read_document(session, doc, doc)
                if not documents[doc]:
                    sys.exit(f"--doc: no articles in {doc!r} (the exact document name)")
    codes = {doc: sorted(documents[doc], key=number_key) for doc in code_of}
    specials = {doc: sorted(documents[doc], key=number_key) for doc in args.doc}
    refs = order_refs(top, codes, specials)
    items = [(doc, documents[doc][number]) for doc, number in refs]
    bare = [item for item in items if not has_body(item[1])]
    items = [item for item in items if has_body(item[1])]
    if args.limit:
        items = items[:args.limit]
    print(f"{len(items):,} articles: top {len(top):,}, then the codes, then {len(args.doc)} --doc; {len(bare):,} only "
          f"\"(abrogato)\", no card; {sum(due(a, d, have, args.force) for d, a in items):,} to make", flush=True)
    if args.dry_run:
        for doc, article in items[:30]:
            print(f"  art. {article['number']} {doc}")
        print("\n".join(summary(list(have.values()))))
        return
    base, key = os.getenv("LLM_BASE_URL") or "", os.getenv("LLM_API_KEY") or ""
    url = args.url or (re.sub(r":\d+(?=/|$)", ":8010", base, count=1) if re.search(r":\d+(?:/|$)", base) else "")
    if not url or not key:
        sys.exit("LLM_API_KEY must be set, and LLM_BASE_URL have a port to change to 8010, or give --url")
    endpoint = {"url": url, "key": key, "model": args.model, "max_tokens": ac.A_TOKENS,
                "extra": {"chat_template_kwargs": {"enable_thinking": False}}}
    print(f"{args.model} at {url}, {args.concurrency} at once" +
          (f", at most {args.max_per_hour} an hour" if args.max_per_hour else ""), flush=True)
    outcome = run(items, endpoint, args.file, have, args.concurrency, Throttle(args.max_per_hour), args.force)
    compact(args.file, have)
    print(f"This run: {outcome}")
    print("\n".join(summary(list(have.values()))))


if __name__ == "__main__":
    main()
