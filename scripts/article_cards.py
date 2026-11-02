"""Prototype of article cards: 25 articles, each carded from its own text only, for a lawyer to review.

Read-only on Neo4j; nothing in the chatbot changes or reads the cards. Run from the app folder with its venv and .env:

    venv/bin/python scripts/article_cards.py --select-only      # the 25 and why: no LLM call
    venv/bin/python scripts/article_cards.py                    # the cards: <out>/article_cards.md

1 SELECTION  by code (--quota, default 9 c.c., 8 c.p., 4 c.p.c., 4 c.p.p.), one from each list in turn: most trouble in
             the audits (each fetch_audit audit.json: a sentence citing the article judged "contraddetta" weighs 3, one
             "assente" 1, the article lost in a case 3 -- the understanding mechanism's list, the one fact_loss_report
             reads) and most often expected (tests/eval_narrative_cases.json). A sentence citing two articles counts
             for both. A code short of articles leaves its slots to the others, and the report says so; an article with
             no text in the corpus is passed over. --articles "2043 c.c.,43 c.p." chooses by hand. --audit defaults to
             the newest fetch_audit_runs/*/audit.json. --articles-from <earlier article_cards.md> repeats its articles.
2 TEXT       article_text.fetch_article, the pipeline's own reading, from the code's primary document (the fullest copy).
             The Versione line is written here, from the "In vigore dal ..." the article's Sections hold, never by the
             model; without one it says so.
3 CARDS      temperature 0, from the text only, in a fixed format; "Articoli collegati" lists only what the text names.
             A = LLM_MODEL at LLM_BASE_URL, no thinking. With --with-b also B = --b-model (qwen3-32b-awq) at the same
             address on port 8010 (--b-url), no thinking either unless --b-thinking, its reasoning left out.
4 EXPORT     <out>/article_cards.md: the selection, then per article the official text, its card(s), and what the code
             checked: a heading missing, a number in the card that the text does not hold (a term in words written in
             figures is flagged too: it is a prompt to look, not a verdict).

Each card is saved under <out>/cards/ as it is made, so a stopped run resumes; a card that failed is shown as failed in
the export and tried again by the next run.
"""

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime

CODES = {"cc": "c.c.", "cp": "c.p.", "cpc": "c.p.c.", "cpp": "c.p.p."}
CODE_PREFIXES = {"c.c.": "Codice Civile", "c.p.c.": "Codice di procedura civile", "c.p.": "Codice Penale",
                 "c.p.p.": "Codice di procedura penale"}      # defensive_generation._CODE_PREFIXES
QUOTA = {"c.c.": 9, "c.p.": 8, "c.p.c.": 4, "c.p.p.": 4}
WEIGHT = {"contraddetta": 3, "assente": 1, "lost": 3}
HEADINGS = ("Regola principale", "Presupposti/condizioni", "Eccezioni e limiti", "Pene", "Termini", "Articoli collegati")
A_TOKENS, THINKING_TOKENS = 4000, 8000       # review3: 61 c.p., 309 c.p.p. cut at 2000; with thinking, B reasons within its limit
TRIES = 3
TIMEOUT = 600

SYSTEM = (
    "Sei un giurista italiano e redigi la scheda di un articolo di legge. Ricevi il TESTO UFFICIALE dell'articolo. "
    "Scrivi la scheda SOLO da quel testo: non usare ciò che sai, non aggiungere "
    "giurisprudenza, interpretazioni, esempi, termini, importi o altri articoli che il testo non contiene. Riporta "
    "fedelmente condizioni, eccezioni, numeri e termini; se il testo non dice una cosa, non dirla. Scrivi in italiano, "
    "in modo breve e preciso.\n"
    "Non tralasciare nulla: ogni comma va coperto e ogni voce di un elenco (lettere a), b)...; numeri 1), 2)...) va "
    "riportata, in forma compatta, una per una, anche se sono molte.\n"
    "Il testo tra parentesi quadre [ ] o segnato «(abrogato)» non è più in vigore: non presentarlo mai come regola o "
    "condizione vigente; indica quel comma o quella voce solo come «abrogato».\n"
    "Conserva la numerazione ufficiale di commi e voci così come è nel testo (3-bis, 11-quater, lett. c)): non "
    "rinumerare mai.\n"
    "Rispondi SOLO con queste sei voci, in quest'ordine, ciascuna col suo titolo esatto in grassetto seguito da due "
    "punti:\n"
    "**Regola principale:** una riga breve per OGNI comma dell'articolo, nell'ordine del testo, nessuno escluso.\n"
    "**Presupposti/condizioni:** ciò che serve perché la regola si applichi, in elenco; ogni voce enumerata del testo "
    "ha la sua riga.\n"
    "**Eccezioni e limiti:** eccezioni, limiti e deroghe che il testo prevede.\n"
    "**Pene:** le pene e le sanzioni, con le forme aggravate e attenuate e i loro importi e durate.\n"
    "**Termini:** termini e scadenze che il testo indica, ciascuno col suo decorso; non le durate delle pene.\n"
    "**Articoli collegati:** solo gli articoli o le leggi che il testo dell'articolo nomina, mai quelli di una nota.\n"
    "Per una voce vuota scrivi \"Nessuno nel testo.\"; non scrivere altro, né una riga sulla versione.")


def _eval_module():
    spec = importlib.util.spec_from_file_location("eval_narrative", os.path.join("tests", "eval_narrative.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- pure: the selection --------------------------------------------------------------------------------------------

def tally(audits: list, cases: list, mentions) -> dict:
    """{(number, code key): Counter(contraddetta, assente, lost, expected) + its first example}; only the four codes."""
    out = defaultdict(Counter)
    examples = {}
    seen = set()
    for audit in audits:
        for case in audit.get("cases") or []:
            base = re.sub(r"@r\d+$", "", str(case.get("id")))
            for item in (case.get("provenance") or {}).get("sentences") or []:
                verdict = item.get("verdict")
                if verdict not in ("contraddetta", "assente") or item.get("kind") in (None, "altro", "applicazione"):
                    continue
                sentence = item.get("sentence") or ""
                if (base, sentence) in seen:
                    continue
                seen.add((base, sentence))
                for number, code in dict.fromkeys(mentions(sentence)):
                    if code in CODES:
                        out[(number, code)][verdict] += 1
                        if verdict == "contraddetta" or (number, code) not in examples:
                            examples[(number, code)] = sentence
            for lost in (case.get("understanding") or {}).get("lost") or []:
                label = str(lost.get("article") or "").lower()
                code = re.sub(r"[^a-z]", "", label.rsplit(" ", 1)[-1])
                for number in re.findall(r"\d+(?:-[a-z]+)?", label):
                    if code in CODES and (base, number, code) not in seen:
                        seen.add((base, number, code))
                        out[(number, code)]["lost"] += 1
    for case in cases:
        counted = set()
        for item in case.get("expected") or []:
            for article in item.get("any_of") or [item["article"]]:
                key = (str(article).lower(), item["code"])
                if item["code"] in CODES and key not in counted:
                    counted.add(key)
                    out[key]["expected"] += 1
    return {key: {"counts": counts, "example": examples.get(key, "")} for key, counts in out.items()}


def audit_score(counts: Counter) -> int:
    return sum(counts[k] * w for k, w in WEIGHT.items())


def reason(item: dict) -> str:
    counts = item["counts"]
    parts = [f"{what}: {n}" for n, what in ((counts["contraddetta"], "frasi contraddette"),
                                            (counts["assente"], "frasi non nel materiale"),
                                            (counts["lost"], "casi in cui è perso"),
                                            (counts["expected"], "casi in cui è atteso")) if n]
    return ", ".join(parts) or "nessun segnale"


def choose(table: dict, quota: dict, has_text) -> tuple:
    """([(number, code label)] up to the quotas, [notes]): per code one from the audits' list and one from the expected
    list in turn; has_text(ref) false passes an article over; slots a code cannot fill go to the others, in turn."""
    lists = {}
    for key, code in CODES.items():
        mine = [k for k in table if k[1] == key]
        by_audit = sorted((k for k in mine if audit_score(table[k]["counts"])),
                          key=lambda k: (-audit_score(table[k]["counts"]), -table[k]["counts"]["expected"], k))
        by_expected = sorted((k for k in mine if table[k]["counts"]["expected"]),
                             key=lambda k: (-table[k]["counts"]["expected"], -audit_score(table[k]["counts"]), k))
        turns = [k for pair in zip(by_audit + [None] * len(mine), by_expected + [None] * len(mine)) for k in pair if k]
        lists[code] = list(dict.fromkeys(turns))
    picked, notes = [], []

    def take(code):
        while lists[code]:
            key = lists[code].pop(0)
            if has_text((key[0], code)):
                picked.append(key)
                return True
            notes.append(f"art. {key[0]} {code}: nessun testo nel corpus, saltato")
        return False
    short = 0
    for code, n in quota.items():
        got = sum(take(code) for _ in range(n))
        if got < n:
            notes.append(f"{code}: solo {got} articoli su {n} con segnali")
            short += n - got
    while short and any(lists[code] for code in quota):
        for code in quota:
            if short and take(code):
                short -= 1
    return picked, notes


# --- the text -------------------------------------------------------------------------------------------------------

def document_ids(session, code: str) -> list:
    return [(r["id"], r["name"]) for r in session.run(
        "MATCH (d:Document) WHERE d.document_type = 'primary' AND d.name STARTS WITH $prefix "
        "RETURN d.id AS id, d.name AS name", prefix=CODE_PREFIXES[code]).data()]


def in_force_date(session, doc_id: str, number: str) -> str:
    """"11 dicembre 2021" from the article's own Sections ("N.0.0", "N", then its commi); "" when none says."""
    rows = session.run("MATCH (d:Document {id: $doc})-[:CONTAINS]->(s:Section) WHERE s.name = $n OR s.name STARTS WITH "
                       "$n + '.' RETURN s.name AS name, s.plain_text AS text", doc=doc_id, n=number).data()
    rank = {f"{number}.0.0": 0, number: 1}
    for row in sorted(rows, key=lambda r: (rank.get(r["name"], 2), r["name"] or "")):
        m = re.match(r"\s*in vigore dal\s+(\d{1,2}\s+\w+\s+\d{4})", row["text"] or "", re.IGNORECASE)
        if m:
            return m.group(1)
    return ""


def read_article(session, number: str, code: str) -> dict:
    """{"number", "code", "rubrica", "text", "document", "date", "fragment"}, the fullest copy in the code; {} if none."""
    from src.rag import article_text
    best = {}
    for doc_id, name in document_ids(session, code):
        rows = article_text.fetch_article(session, doc_id, number)
        if not rows:
            continue
        section = rows[0].get("s") or {}
        text = "\n".join(((r.get("s") or {}).get("plain_text") or "").strip() for r in rows).strip()
        if not text:
            continue
        quality = (article_text.section_quality(section)[0], len(text))
        if not best or quality > best["quality"]:
            spelling = article_text.article_number(section.get("name") or "") or number
            best = {"quality": quality, "number": number, "code": code, "text": text, "document": name,
                    "rubrica": (section.get("titolo_articolo") or section.get("title") or "").strip(),
                    "date": in_force_date(session, doc_id, spelling),
                    "fragment": (section.get("name") or "") not in (spelling, f"{spelling}.0.0")}
    best.pop("quality", None)
    return best


# --- the cards ------------------------------------------------------------------------------------------------------

def user_message(article: dict) -> str:
    return (f"ARTICOLO: art. {article['number']} {article['code']}"
            + (f" - {article['rubrica']}" if article["rubrica"] else "")
            + f"\n\nTESTO UFFICIALE:\n{article['text']}")


def clean_reply(content: str) -> str:
    """The card without the model's reasoning, when it comes inside the reply."""
    text = re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    return text.strip()


def call(endpoint: dict, user: str) -> str:
    """The card; ValueError when the reply is unusable. Network and 5xx errors are tried TRIES times."""
    payload = {"model": endpoint["model"], "temperature": 0, "max_tokens": endpoint["max_tokens"],
               "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
               **endpoint.get("extra", {})}
    request = urllib.request.Request(endpoint["url"].rstrip("/") + "/chat/completions", method="POST",
                                     data=json.dumps(payload).encode("utf-8"),
                                     headers={"Authorization": f"Bearer {endpoint['key']}",
                                              "Content-Type": "application/json"})
    problem = ""
    for attempt in range(1, TRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as reply:
                choice = json.load(reply)["choices"][0]
            text = clean_reply((choice.get("message") or {}).get("content") or "")
            if choice.get("finish_reason") == "length":
                raise ValueError(f"reply cut at max_tokens {endpoint['max_tokens']}")
            if not text:
                raise ValueError("empty reply")
            return text
        except urllib.error.HTTPError as exc:
            problem = f"HTTP {exc.code}: {exc.read(300).decode('utf-8', 'replace')}"
            if exc.code < 500 and exc.code != 429:
                break
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError) as exc:
            problem = f"{type(exc).__name__}: {exc}"
        except ValueError as exc:
            raise ValueError(str(exc))
        if attempt < TRIES:
            time.sleep(10 * attempt)
    raise ValueError(problem)


def card(endpoint: dict, side: str, article: dict, folder: str) -> dict:
    """{"text"} or {"error"}: saved when good, so a rerun only tries what failed."""
    user = user_message(article)
    digest = hashlib.sha1(f"{SYSTEM}\n{user}\n{endpoint['model']}".encode("utf-8")).hexdigest()[:12]
    path = os.path.join(folder, f"{article['number']}_{article['code'].replace('.', '')}_{side}_{digest}.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    t0 = time.time()
    try:
        out = {"text": call(endpoint, user), "seconds": round(time.time() - t0, 1)}
    except ValueError as exc:
        return {"error": str(exc)}
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False)
    os.replace(path + ".tmp", path)
    return out


def checks(text: str, article: dict) -> list:
    """What the code can see: a heading missing; numbers in the card that the source does not hold."""
    missing = [h for h in HEADINGS if not re.search(rf"\*\*{re.escape(h)}\s*:?\*\*", text)]
    source = f"{article['number']} {article['code']} {article['rubrica']} {article['text']}"
    have = set(re.findall(r"\d+", source))
    extra = list(dict.fromkeys(n for n in re.findall(r"\d+", text) if n not in have))
    out = [f"voci mancanti: {', '.join(missing)}"] if missing else []
    return out + ([f"numeri non presenti nel testo: {', '.join(extra)}"] if extra else [])


def version_line(article: dict) -> str:
    date = f"in vigore dal {article['date']}" if article["date"] else "data di vigore non presente nel DB"
    return f"**Versione:** {date} ({article['document']})"


# --- the export -----------------------------------------------------------------------------------------------------

def label_of(key: tuple) -> str:
    return f"art. {key[0]} {CODES[key[1]]}"


def render(items: list, notes: list, names: dict, audits: list) -> str:
    models = f"Scheda A: {names['A']}." + (f" Scheda B: {names['B']}." if "B" in names else "")
    L = ["# Schede articolo: prototipo per la revisione", "",
         f"Generate il {datetime.now().strftime('%d/%m/%Y %H:%M')}. {models} Temperatura 0, dal solo testo dell'articolo.",
         f"Audit usati: {', '.join(audits) or 'nessuno'}.", "",
         "La riga Versione è scritta dal codice, non dal modello. «Da controllare» segnala ciò che il codice vede "
         "(voci mancanti, numeri che il testo non contiene): non è un giudizio sulla scheda.", "",
         "## Articoli scelti", "", "| # | articolo | rubrica | perché |", "|---:|---|---|---|"]
    for i, item in enumerate(items, 1):
        a = item["article"]
        L.append(f"| {i} | art. {a['number']} {a['code']} | {a['rubrica'] or '-'} | {item['why']} |")
    if notes:
        L += ["", "Note: " + "; ".join(notes) + "."]
    for i, item in enumerate(items, 1):
        a = item["article"]
        L += ["", "---", "", f"## {i}. Art. {a['number']} {a['code']}" + (f" - {a['rubrica']}" if a["rubrica"] else ""), "",
              f"*Perché è stato scelto: {item['why']}.*"]
        if item.get("example"):
            L += ["", f"*Esempio dall'audit: «{item['example'][:300]}»*"]
        L += ["", "### Testo ufficiale", ""]
        if a["fragment"]:
            L += ["*Attenzione: nel corpus l'articolo è solo in frammenti; il testo può essere incompleto.*", ""]
        L += ["> " + line if line.strip() else ">" for line in a["text"].split("\n")]
        L += ["", version_line(a)]
        for side in names:
            got = item["cards"][side]
            L += ["", f"### Scheda {side} ({names[side]})", ""]
            if got.get("error"):
                L.append(f"*Scheda non generata: {got['error'][:300]}*")
                continue
            L += [re.sub(r"\n+(?=\*\*)", "\n\n", got["text"]), "", version_line(a)]
            found = checks(got["text"], a)
            if found:
                L += ["", "*Da controllare: " + "; ".join(found) + ".*"]
    return "\n".join(L) + "\n"


def endpoints(args) -> dict:
    base, key, model = os.getenv("LLM_BASE_URL") or "", os.getenv("LLM_API_KEY") or "", os.getenv("LLM_MODEL") or ""
    if not base or not key or not model:
        sys.exit("LLM_BASE_URL, LLM_API_KEY and LLM_MODEL must be set (--env)")
    ends = {"A": {"url": base, "key": key, "model": model, "max_tokens": A_TOKENS}}
    if not args.with_b:
        return ends
    b_url = args.b_url or (re.sub(r":\d+(?=/|$)", ":8010", base, count=1) if re.search(r":\d+(?:/|$)", base) else "")
    if not b_url or b_url == base:
        sys.exit(f"LLM_BASE_URL has no port to change to 8010: give --b-url ({base})")
    ends["B"] = {"url": b_url, "key": key, "model": args.b_model, "max_tokens": THINKING_TOKENS if args.b_thinking else A_TOKENS,
                 "extra": {"chat_template_kwargs": {"enable_thinking": bool(args.b_thinking)}}}
    return ends


def articles_from(path: str) -> str:
    """"2043 c.c.,43 c.p." from the selection table of an earlier export; "" without a path."""
    if not path:
        return ""
    with open(path, encoding="utf-8") as fh:
        found = re.findall(r"^\| \d+ \| art\. (\S+) (c\.[a-z.]+) \|", fh.read(), re.MULTILINE)
    if not found:
        sys.exit(f"--articles-from: no articles in {path}")
    return ",".join(f"{number} {code}" for number, code in found)


def parse_quota(text: str) -> dict:
    out = {}
    for part in text.split(","):
        code, _, n = part.strip().partition("=")
        if code not in QUOTA or not n.isdigit():
            sys.exit(f"--quota: {part!r} (e.g. c.c.=9,c.p.=8,c.p.c.=4,c.p.p.=4)")
        out[code] = int(n)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--audit", action="append", default=[], help="a fetch_audit audit.json (repeatable; default: "
                                                                     "the newest under fetch_audit_runs/)")
    parser.add_argument("--articles", help='by hand instead: "2043 c.c.,43 c.p."')
    parser.add_argument("--quota", default=",".join(f"{c}={n}" for c, n in QUOTA.items()))
    parser.add_argument("--articles-from", help="the articles of an earlier article_cards.md, in its order")
    parser.add_argument("--with-b", action="store_true", help="also a second card from Qwen3")
    parser.add_argument("--b-thinking", action="store_true", help="B with thinking on")
    parser.add_argument("--b-url", help="Qwen3's base URL (default: LLM_BASE_URL on port 8010)")
    parser.add_argument("--b-model", default="qwen3-32b-awq")
    parser.add_argument("--select-only", action="store_true", help="the list and the reasons, no LLM call")
    parser.add_argument("--env", default=os.path.join(os.getcwd(), ".env"))
    parser.add_argument("--out", help="default article_cards/<date-time>")
    args = parser.parse_args()

    sys.path.insert(0, os.getcwd())
    from dotenv import load_dotenv
    load_dotenv(args.env)
    uri = os.getenv("NEO4J_URI")
    if not uri:
        sys.exit(f"NEO4J_URI is not set ({args.env})")
    quota = parse_quota(args.quota)
    paths = args.audit or sorted(glob.glob(os.path.join("fetch_audit_runs", "*", "audit.json")), key=os.path.getmtime)[-1:]
    audits = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            audits.append(json.load(fh))
    if not audits and not args.articles:
        print("WARNING: no audit.json: chosen from the eval keys alone", file=sys.stderr)
    with open(os.path.join("tests", "eval_narrative_cases.json"), encoding="utf-8") as fh:
        eval_cases = json.load(fh)["cases"]
    table = tally(audits, eval_cases, _eval_module().mentions)

    import neo4j
    with neo4j.GraphDatabase.driver(uri, auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))) as driver:
        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j"),
                            default_access_mode=neo4j.READ_ACCESS) as session:
            read = {}

            def has_text(ref):
                if ref not in read:
                    read[ref] = read_article(session, ref[0], ref[1])
                return bool(read[ref])
            by_hand = args.articles or articles_from(args.articles_from)
            if by_hand:
                abbr_key = {v: k for k, v in CODES.items()}
                keys = []
                for part in by_hand.split(","):
                    number, _, code = part.strip().partition(" ")
                    if code not in abbr_key or not has_text((number, code)):
                        sys.exit(f"--articles: {part.strip()!r} not found")
                    keys.append((number, abbr_key[code]))
                notes = []
            else:
                keys, notes = choose(table, quota, has_text)
            items = []
            for key in keys:
                entry = table.get(key) or {"counts": Counter(), "example": ""}
                article = read[(key[0], CODES[key[1]])]
                items.append({"key": key, "article": article, "why": reason(entry) if not by_hand or table.get(key)
                              else "scelto a mano", "example": entry["example"]})
    print(f"{len(items)} articoli:")
    for i, item in enumerate(items, 1):
        print(f"{i:>3}. {label_of(item['key']):<20} {item['article']['rubrica'][:50]:<52} {item['why']}")
    for note in notes:
        print(f"nota: {note}")
    if args.select_only:
        return

    ends = endpoints(args)
    out_dir = args.out or os.path.join("article_cards", datetime.now().strftime("%Y%m%d-%H%M"))
    folder = os.path.join(out_dir, "cards")
    os.makedirs(folder, exist_ok=True)
    for side, end in ends.items():
        print(f"{side}: {end['model']} at {end['url']}", flush=True)
    for i, item in enumerate(items, 1):
        item["cards"] = {}
        for side, end in ends.items():
            got = card(end, side, item["article"], folder)
            item["cards"][side] = got
            print(f"  {i}/{len(items)} {label_of(item['key'])} {side}: "
                  f"{'FAILED ' + got['error'][:120] if got.get('error') else str(got.get('seconds', 0)) + 's'}", flush=True)
    path = os.path.join(out_dir, "article_cards.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render(items, notes, {s: e["model"] for s, e in ends.items()},
                        [os.path.relpath(p) for p in paths]))
    failed = sum(1 for it in items for c in it["cards"].values() if c.get("error"))
    print(f"File: {os.path.abspath(path)}" + (f" ({failed} schede fallite: rilancia con --out {out_dir})" if failed else ""))


if __name__ == "__main__":
    main()
