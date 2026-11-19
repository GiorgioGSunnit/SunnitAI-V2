"""Article cards in bulk: the live model without thinking, into a JSONL file. Read-only on Neo4j; run from the app folder:

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

v79.11 SCHEMAS  --schemas: the element schemas (src/rag/element_schemas.py: SCHEMA_SYSTEM, user_message, parse_reply,
         make_line) into --schema-file (default element_schemas.schema_file(), data/article_schemas.jsonl); the card mode
         above is untouched. Order: (a) every article of the cards file (--file), in its order; (b) the rest of c.c., c.p.,
         c.p.c., c.p.p. by number; (c) each --doc. A text that is only "(abrogato)" gets no schema; each line carries the
         raw article text ("text") for scripts/schema_show.py. Endpoint: --url (SCHEMA_LLM_BASE_URL, else LLM_BASE_URL on
         port 8003), --model (SCHEMA_LLM_MODEL, else qwen38-27b), thinking off, --json-mode (on; --no-json-mode) asks
         response_format json_object, max_tokens SCHEMA_TOKENS. Same --concurrency, --max-per-hour (Throttle). A schema is
         made again when its source_hash or generation_hash (schema version, prompt, model) changed or it failed
         (--force: all) -- NEVER for a label the lawyer reviewed (data/article_schemas_reviewed.jsonl, reviewed: true):
         "reviewed, kept"; one whose text changed after the review is counted "reviewed_stale" (the runtime takes the
         text path) and its generated line records the text now (current_source_hash, current_text) so that
         scripts/schema_show.py --stale shows it; the overlay is never written. Cut: once more with element_schemas.SHORTER_NOTE, then failed; invalid JSON: once more with
         JSON_NOTE, then failed. Accounting every 25 and at the end (made, failed, unusable by check, tokens, wall time,
         articles an hour, seconds a call); the end: coverage by code (usable / unusable / without), the checks, the cost.
         --dry-run: the order and the coverage, no call.
         v79.12.3: a line whose prompt alone changed (its generation_hash is its own prompt_hash's) is kept, counted
         "prompt" (--regen-prompt: made again); --only-keys FILE (schema_show.py --recheck --regen-list's JSONL, or
         «doc<TAB>article» lines): only those articles, each made again ("asked") -- never a reviewed label.

    venv/bin/python scripts/article_cards_bulk.py --schemas --dry-run                      # the order and the coverage
    nohup venv/bin/python scripts/article_cards_bulk.py --schemas > article_schemas_bulk.log 2>&1 &
    venv/bin/python scripts/article_cards_bulk.py --schemas --only-keys regen.jsonl     # v79.12.3: what --recheck lists
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

DEFAULT_FILE = os.getenv("ARTICLE_CARDS_FILE") or os.path.join(   # v77: the runtime state folder (src/state_paths.py)
    os.getenv("STATE_DIR") or ("/opt/chatbot/state" if os.path.isdir("/opt/chatbot") else "state"), "article_cards.jsonl")
TOP = 500
SCHEMA_TOKENS = 6000                      # v79.11: --schemas' max_tokens
SCHEMA_MODEL, SCHEMA_PORT = "qwen38-27b", 8003
SCHEMA_CODES = ("c.c.", "c.p.", "c.p.c.", "c.p.p.")
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


# --- v79.11: element schemas (--schemas) -----------------------------------------------------------------------------

def _es():
    from src.rag import element_schemas
    return element_schemas


def schema_model(args) -> str:
    return getattr(args, "model", None) or os.getenv("SCHEMA_LLM_MODEL") or SCHEMA_MODEL


def schema_endpoint(args) -> dict:
    """v79.11: the schema model -- --url, else SCHEMA_LLM_BASE_URL, else LLM_BASE_URL on port 8003; --model, else
    SCHEMA_LLM_MODEL, else qwen38-27b; thinking off; response_format json_object unless --no-json-mode; SCHEMA_TOKENS."""
    base = os.getenv("LLM_BASE_URL") or ""
    url = getattr(args, "url", None) or os.getenv("SCHEMA_LLM_BASE_URL") or (
        re.sub(r":\d+(?=/|$)", f":{SCHEMA_PORT}", base, count=1) if re.search(r":\d+(?:/|$)", base) else "")
    if not url:
        sys.exit(f"LLM_BASE_URL has no port to change to {SCHEMA_PORT}: give --url or SCHEMA_LLM_BASE_URL ({base})")
    key = os.getenv("LLM_API_KEY") or ""
    if not key:
        sys.exit("LLM_API_KEY must be set")
    extra = {"chat_template_kwargs": {"enable_thinking": False}}
    if getattr(args, "json_mode", True):
        extra["response_format"] = {"type": "json_object"}
    return {"url": url, "key": key, "model": schema_model(args), "max_tokens": SCHEMA_TOKENS, "extra": extra}


def schema_label(article: dict) -> str:
    es = _es()
    return es.norm_label(es.label_of(article["number"], article["code"]))


def reviewed_labels() -> set:
    """The labels the lawyer reviewed (the overlay's reviewed: true lines): never generated again."""
    return {label for label, line in _es().reviewed().items() if line.get("reviewed") is True}


def reviewed_hashes() -> dict:
    """v79.11: {label: the source_hash the lawyer reviewed against} of the overlay's reviewed: true lines -- the labels
    never generated again (reviewed_labels), each with the hash schema_due checks for a stale review."""
    return {label: line.get("source_hash") for label, line in _es().reviewed().items() if line.get("reviewed") is True}


def schema_due(article: dict, doc: str, have: dict, model: str, reviewed=frozenset(), force: bool = False,
               asked: bool = False) -> str:
    """Why the article's schema is to be made: "new", "source" (its text changed), "generation" (schema version or
    model changed), "failed", "force", "asked" (v79.12.3: a key of --only-keys); "" when its line is current;
    "reviewed" for a label the lawyer reviewed -- never made again, --force, --only-keys or not; "reviewed_stale" for
    one whose review's source_hash is no longer the text's (when reviewed is reviewed_hashes()' {label: hash}; a set of
    labels gives "reviewed"), never made again either. v79.12.3: "prompt" when only the prompt changed (the line's
    generation_hash is its own prompt_hash's) -- due only with --regen-prompt (is_due), so a new prompt never remakes
    the whole file unasked."""
    es = _es()
    label = schema_label(article)
    if label in reviewed:
        base = reviewed.get(label) if isinstance(reviewed, dict) else None
        return "reviewed_stale" if base and base != es.text_hash(article["text"]) else "reviewed"
    record = have.get((doc, article["number"]))
    if not record:
        return "new"
    source = es.text_hash(article["text"])
    if record.get("source_hash") != source:
        return "source"
    prompt_only = False
    if record.get("generation_hash") != es.generation_hash(source, model):
        own = record.get("prompt_hash")
        if own and own != es.prompt_hash() and record.get("generation_hash") == es.generation_hash(source, model, own):
            prompt_only = True
        else:
            return "generation"
    if (record.get("checks") or {}).get("failed"):
        return "failed"
    if force:
        return "force"
    if asked:
        return "asked"
    return "prompt" if prompt_only else ""


def is_due(why: str, prompt: bool = False) -> bool:
    """v79.12.3: "prompt" (only the prompt changed) is due only with --regen-prompt (prompt=True)."""
    return why not in ("", "reviewed", "reviewed_stale") and (why != "prompt" or prompt)


def read_keys(path: str) -> set:
    """v79.12.3: the (doc, article) keys of --only-keys FILE -- schema_show.py --recheck --regen-list's JSONL ({"doc",
    "article", ...} a line), or «doc<TAB>article» lines; blank and «#» lines skipped. The article lowercased, as
    read_document numbers it."""
    keys = set()
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            if raw.startswith("{"):
                try:
                    rec = json.loads(raw)
                except ValueError:
                    continue
                doc, number = rec.get("doc"), rec.get("article")
            else:
                doc, _, number = raw.partition("\t")
            if doc is not None and str(number or "").strip():
                keys.add((str(doc), str(number).strip().lower()))
    return keys


def text_now(record: dict, article: dict):
    """v79.11: a reviewed label's generated line with the article's text now recorded beside it (current_source_hash,
    current_text: what schema_show --stale and --edit read) -- its schema, source_hash and text (what the model read)
    untouched, so the runtime still never takes it for the new text; None when the line already records it."""
    es = _es()
    now = es.text_hash(article["text"])
    if (record.get("current_source_hash") or record.get("source_hash")) == now:
        return None
    line = {k: v for k, v in record.items() if k not in ("current_source_hash", "current_text")}
    if line.get("source_hash") != now:
        line.update(current_source_hash=now, current_text=article["text"])
    return line


def make_schema(endpoint: dict, doc: str, article: dict, stats: dict = None) -> dict:
    """v79.11: the schema file's line for the article (element_schemas.make_line). Cut (finish_reason length): once more
    with SHORTER_NOTE; invalid JSON: once more with JSON_NOTE; then failed. Never raises for a bad reply. stats, when
    given, gains "calls" and "call_seconds"."""
    es = _es()
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    notes, retried, schema, failed, cut = [], set(), None, "", False
    started = time.time()
    while True:
        got, problem, reply = {}, None, ""
        t0 = time.time()
        try:
            reply = ac.call(endpoint, es.user_message(article, " ".join(notes)), system=es.SCHEMA_SYSTEM, usage=got)
        except ValueError as exc:
            problem = exc
        for k in usage:
            usage[k] += int(got.get(k) or 0)
        if stats is not None:
            stats["calls"] = stats.get("calls", 0) + 1
            stats["call_seconds"] = stats.get("call_seconds", 0.0) + time.time() - t0
        if problem is not None:
            if "cut at max_tokens" in str(problem) and "cut" not in retried:
                retried.add("cut")
                notes.append(es.SHORTER_NOTE)
                continue
            cut = "cut at max_tokens" in str(problem)
            failed = str(problem)[:300]
            break
        try:
            schema = es.parse_reply(reply)
            break
        except ValueError as exc:
            if "json" not in retried:
                retried.add("json")
                notes.append(es.JSON_NOTE)
                continue
            failed = f"invalid JSON: {exc}"[:300]
            break
    try:
        return es.make_line(article, endpoint["model"], schema, failed=failed, cut=cut, seconds=time.time() - started,
                            usage=usage, doc=doc)
    except Exception as exc:  # noqa: BLE001 -- a reply the checks cannot read is a failed line, never a stopped run
        return es.make_line(article, endpoint["model"], None, failed=f"unreadable schema: {exc}"[:300],
                            seconds=time.time() - started, usage=usage, doc=doc)


def unusable_checks(line: dict) -> list:
    """The blocking checks a made (not failed) line raised."""
    es = _es()
    return [k for k in es.raised(line.get("checks")) if k not in ("failed", "cut")]


def accounting(acc: dict, done: int, total: int, started: float) -> str:
    wall = max(time.time() - started, 1e-9)
    unusable = ", ".join(f"{k} {v}" for k, v in sorted(acc["unusable"].items())) or "none"
    return (f"{done:,}/{total:,}: {acc['made']:,} made, {acc['failed']:,} failed, unusable by check: {unusable}; tokens "
            f"{acc['prompt_tokens']:,} prompt + {acc['completion_tokens']:,} completion; {wall:,.0f}s, "
            f"{done * 3600 / wall:,.0f} articles an hour, {acc['call_seconds'] / max(acc['calls'], 1):.1f}s a call "
            f"({acc['calls']:,} calls)")


def run_schemas(items: list, endpoint: dict, path: str, have: dict, concurrency: int, throttle: Throttle,
                force: bool = False, reviewed=frozenset(), every: int = 25, asked: bool = False,
                prompt: bool = False) -> dict:
    """items: [(doc, article)]. Appends each line as it is made (the last line of an article wins); a reviewed label is
    never made ("reviewed_kept"). The accounting printed every `every` and at the end; returned with the counts.
    v79.12.3: asked -- the items are --only-keys' (each made, "asked", unless reviewed); prompt -- --regen-prompt (a
    line whose prompt alone changed made too; else kept, counted "prompt")."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    reasons, todo = Counter(), []
    for doc, article in items:
        why = schema_due(article, doc, have, endpoint["model"], reviewed, force, asked)
        reasons[why or "current"] += 1
        if is_due(why, prompt):
            todo.append((doc, article))
        elif why in ("reviewed", "reviewed_stale") and have.get((doc, article["number"])):
            line = text_now(have[(doc, article["number"])], article)        # the text now beside a reviewed label's
            if line is not None:                                            # line; the overlay never written
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(line, ensure_ascii=False) + "\n")
                have[(doc, article["number"])] = line
    lock = threading.Lock()
    acc = {"made": 0, "failed": 0, "unusable": Counter(), "prompt_tokens": 0, "completion_tokens": 0, "calls": 0,
           "call_seconds": 0.0}
    started = time.time()

    def one(item):
        doc, article = item
        throttle.wait()
        stats = {}
        line = make_schema(endpoint, doc, article, stats)
        with lock:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
            have[(doc, article["number"])] = line
            acc["failed" if line["checks"]["failed"] else "made"] += 1
            if not line["checks"]["failed"]:
                acc["unusable"].update(unusable_checks(line))
            acc["prompt_tokens"] += line["usage"]["prompt_tokens"]
            acc["completion_tokens"] += line["usage"]["completion_tokens"]
            acc["calls"] += stats.get("calls", 0)
            acc["call_seconds"] += stats.get("call_seconds", 0.0)
            done = acc["made"] + acc["failed"]
            if done % every == 0 or done == len(todo):
                print(accounting(acc, done, len(todo), started) + f", last: art. {article['number']} {article['code']}",
                      flush=True)
    with ThreadPoolExecutor(max_workers=max(concurrency, 1)) as pool:
        list(pool.map(one, todo))
    wall = time.time() - started
    done = acc["made"] + acc["failed"]
    return {"due": len(todo), "reviewed_kept": reasons["reviewed"] + reasons["reviewed_stale"],
            "reviewed_stale": reasons["reviewed_stale"], "current": reasons["current"],
            "prompt_kept": 0 if prompt else reasons["prompt"],
            "why": {k: v for k, v in reasons.items() if k not in ("current", "reviewed", "reviewed_stale")
                    and (k != "prompt" or prompt)},
            "made": acc["made"], "failed": acc["failed"],
            "unusable": dict(acc["unusable"]), "prompt_tokens": acc["prompt_tokens"],
            "completion_tokens": acc["completion_tokens"], "calls": acc["calls"], "seconds": round(wall, 1),
            "per_hour": round(done * 3600 / wall) if wall > 0 and done else 0,
            "mean_call_seconds": round(acc["call_seconds"] / acc["calls"], 1) if acc["calls"] else 0.0}


def schema_coverage(items: list, have: dict, reviewed=frozenset()) -> dict:
    """{code: Counter(usable, unusable, without, reviewed)} over the order's articles: a line with a usable schema, one
    with an unusable schema, none (or a failed one); reviewed counts the lawyer's labels besides."""
    es = _es()
    out = defaultdict(Counter)
    for doc, article in items:
        counts = out[article["code"]]
        line = have.get((doc, article["number"]))
        if not line or not isinstance(line.get("schema"), dict):
            counts["without"] += 1
        else:
            counts["usable" if es.usable(line) else "unusable"] += 1
        if schema_label(article) in reviewed:
            counts["reviewed"] += 1
    return dict(out)


def schema_summary(lines: list, coverage: dict) -> list:
    """The end of a run: coverage by code, the checks raised over the file's lines, the cost."""
    es = _es()
    L = ["Coverage of the order (with a usable schema / an unusable one / without; reviewed by the lawyer):",
         f"  {'code':<28}{'usable':>9}{'unusable':>10}{'without':>9}{'reviewed':>10}"]
    total = Counter()
    for code in sorted(coverage, key=lambda c: (c not in SCHEMA_CODES, SCHEMA_CODES.index(c) if c in SCHEMA_CODES else 0, c)):
        c = coverage[code]
        total.update(c)
        L.append(f"  {code[:27]:<28}{c['usable']:>9,}{c['unusable']:>10,}{c['without']:>9,}{c['reviewed']:>10,}")
    L.append(f"  {'all':<28}{total['usable']:>9,}{total['unusable']:>10,}{total['without']:>9,}{total['reviewed']:>10,}")
    counts = Counter(k for line in lines for k in es.BLOCKING + es.NON_BLOCKING if (line.get("checks") or {}).get(k))
    L.append(f"{len(lines):,} lines in the file, {sum(1 for x in lines if es.usable(x)):,} usable; checks raised "
             "(blocking, then reported):")
    L += [f"  {name:<32}{counts.get(name, 0):>7,}" for name in es.BLOCKING + es.NON_BLOCKING]
    prompt = sum(int((x.get("usage") or {}).get("prompt_tokens") or 0) for x in lines)
    completion = sum(int((x.get("usage") or {}).get("completion_tokens") or 0) for x in lines)
    seconds = sum(float(x.get("seconds") or 0) for x in lines)
    L.append(f"Cost: {prompt:,} prompt + {completion:,} completion tokens, {seconds:,.0f}s of calls "
             f"({seconds / max(len(lines), 1):.1f}s an article)")
    return L


def schema_items(cards: list, documents: dict, code_docs: list, specials: list = ()) -> tuple:
    """v79.11: (items [(doc, article)], bare, carded) -- the order of --schemas: (a) the cards file's articles in its
    order (those still in their document), (b) the codes' documents by number, (c) the --doc ones; each once; "bare" the
    ones whose text is only "(abrogato)" (no schema), left out of items."""
    carded = [(d, str(n).lower()) for d, n in cards if str(n).lower() in documents.get(d, {})]
    codes = {doc: sorted(documents[doc], key=number_key) for doc in code_docs}
    extra = {doc: sorted(documents[doc], key=number_key) for doc in specials}
    every = [(doc, documents[doc][number]) for doc, number in order_refs(carded, codes, extra)]
    return [x for x in every if has_body(x[1])], [x for x in every if not has_body(x[1])], carded


def schemas_main(args) -> None:
    """v79.11: --schemas (see the module's docstring)."""
    sys.path.insert(0, os.getcwd())
    from dotenv import load_dotenv
    load_dotenv(args.env)
    uri = os.getenv("NEO4J_URI")
    if not uri:
        sys.exit(f"NEO4J_URI is not set ({args.env})")
    es = _es()
    path = args.schema_file or es.schema_file()
    have = read_file(path)
    cards = list(read_file(args.file))            # (doc, article), in the cards file's order
    reviewed = reviewed_hashes()                  # v79.11: the labels never made again, with their review's hash
    import neo4j
    with neo4j.GraphDatabase.driver(uri, auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))) as driver:
        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j"), default_access_mode=neo4j.READ_ACCESS) as session:
            code_of = {code_document(session, code): code for code in SCHEMA_CODES}
            code_of.pop("", None)
            documents = {doc: read_document(session, doc, code) for doc, code in code_of.items()}
            for doc in dict.fromkeys(d for d, _ in cards):
                if doc not in documents:
                    documents[doc] = read_document(session, doc, code_of.get(doc, doc))
            for doc in args.doc:
                if doc not in documents:
                    documents[doc] = read_document(session, doc, doc)
                if not documents[doc]:
                    sys.exit(f"--doc: no articles in {doc!r} (the exact document name)")
    items, bare, carded = schema_items(cards, documents, list(code_of), args.doc)
    asked = bool(getattr(args, "only_keys", None))
    if asked:                                   # v79.12.3: only the keys asked (schema_show --recheck --regen-list)
        keys = read_keys(args.only_keys)
        items = [(d, a) for d, a in items if (str(d), str(a["number"]).lower()) in keys]
        print(f"--only-keys: {len(keys):,} keys, {len(items):,} of them in the order", flush=True)
    if args.limit:
        items = items[:args.limit]
    model = schema_model(args)
    prompt = bool(getattr(args, "regen_prompt", False))
    whys = Counter(schema_due(a, d, have, model, reviewed, args.force, asked) for d, a in items)
    print(f"{len(items):,} articles: {len(carded):,} carded ({len(cards) - len(carded):,} of the cards file not found), "
          f"then the codes, then {len(args.doc)} --doc; {len(bare):,} only \"(abrogato)\", no schema; "
          f"{sum(v for k, v in whys.items() if is_due(k, prompt)):,} to make ({model}), "
          f"{whys['reviewed'] + whys['reviewed_stale']:,} reviewed, kept ({whys['reviewed_stale']:,} stale: the text "
          "changed after the review, the text path runs until the lawyer reviews it again)" +
          ("" if prompt or not whys["prompt"] else f"; {whys['prompt']:,} made with an earlier prompt, kept "
           "(--regen-prompt to make them again)"), flush=True)
    if args.dry_run:
        for doc, article in items[:30]:
            print(f"  art. {article['number']} {doc}")
        print("\n".join(schema_summary(list(have.values()), schema_coverage(items, have, reviewed))))
        return
    endpoint = schema_endpoint(args)
    print(f"{endpoint['model']} at {endpoint['url']}, {args.concurrency} at once, json mode "
          f"{'on' if 'response_format' in endpoint['extra'] else 'off'}" +
          (f", at most {args.max_per_hour} an hour" if args.max_per_hour else ""), flush=True)
    outcome = run_schemas(items, endpoint, path, have, args.concurrency, Throttle(args.max_per_hour), args.force,
                          reviewed, asked=asked, prompt=prompt)
    compact(path, have)
    print(f"This run: {outcome}")
    print("\n".join(schema_summary(list(have.values()), schema_coverage(items, have, reviewed))))


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
    parser.add_argument("--url", help="the model's base URL (default: LLM_BASE_URL)")
    parser.add_argument("--model", help="default: LLM_MODEL")
    parser.add_argument("--env", default=os.path.join(os.getcwd(), ".env"))
    parser.add_argument("--schemas", action="store_true", help="v79.11: element schemas instead of cards")
    parser.add_argument("--schema-file", help="--schemas: default element_schemas.schema_file()")
    parser.add_argument("--json-mode", dest="json_mode", action="store_true", default=True,
                        help="--schemas: response_format json_object (default)")
    parser.add_argument("--no-json-mode", dest="json_mode", action="store_false")
    parser.add_argument("--only-keys", metavar="FILE", help="--schemas (v79.12.3): only these (doc, article) keys, "
                        "each made again unless reviewed -- schema_show.py --recheck --regen-list's file")
    parser.add_argument("--regen-prompt", action="store_true", help="--schemas (v79.12.3): also make again the "
                        "lines whose prompt alone changed (kept by default)")
    args = parser.parse_args()
    if args.schemas:
        schemas_main(args)
        return
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
    url, model = args.url or base, args.model or os.getenv("LLM_MODEL") or ""
    if not url or not key or not model:
        sys.exit("LLM_BASE_URL (or --url), LLM_MODEL (or --model) and LLM_API_KEY must be set")
    endpoint = {"url": url, "key": key, "model": model, "max_tokens": ac.A_TOKENS,
                "extra": {"chat_template_kwargs": {"enable_thinking": False}}}
    print(f"{model} at {url}, {args.concurrency} at once" +
          (f", at most {args.max_per_hour} an hour" if args.max_per_hour else ""), flush=True)
    outcome = run(items, endpoint, args.file, have, args.concurrency, Throttle(args.max_per_hour), args.force)
    compact(args.file, have)
    print(f"This run: {outcome}")
    print("\n".join(summary(list(have.values()))))


if __name__ == "__main__":
    main()
