"""Pilot: one label card per criminal ruling, made by the model, checked by code.

Step 2 of the ruling-cards plan (22 Oct 2026). Step 1 (exp_ruling_links.py) found the
ruling collection 81% civil, with only about 2,100 criminal rulings. That step labelled
the side a criminal ruling favours for only half of them, and links no ruling to the
articles it applies. Keywords instead of a summary is the PM's idea. Each card holds:

  - articoli_applicati: the articles the court applies to decide, chosen from the ones
    the ruling cites (step 1's list; anything else is dropped);
  - ricorrente, esito, a_favore_di (imputato / accusa / non chiaro);
  - argomento_decisivo, parole_chiave_fatti, parole_chiave_questioni: keywords;
  - citazione: one or two sentences copied from the ruling, kept only when the code
    finds them in the text;
  - riferimento: court, section, number and date. The court comes from step 1, the
    section from the opening, and the number and date from the document's own fields.

Reads step 1's labels (--links, default /tmp/ruling_links.jsonl) to pick the rulings,
criminal by default. Rulings citing a c.p. article come first. Appends one card per
ruling to --out and skips rulings already carded, so a run can stop and resume.
--stop-at 06:30 stops taking new rulings at that server time. --workers sets the
parallel model calls (the model is shared with the live chatbot: 1 by day, 3 at night).

Run from the app folder (prod), with exp_ruling_links.py next to this script:
    cd /opt/chatbot && venv/bin/python /tmp/exp_ruling_cards.py --limit 20
    ... --workers 3 --stop-at 06:30          (the night run)
    ... --report                              (only the report on the cards so far)

Nothing is written to the database: every Neo4j session is read-only. Only counts and
keywords are printed. The cards, with their quotes, stay in --out on the server.
"""

import argparse
import datetime as dt
import importlib.util
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("exp_ruling_links", os.path.join(_HERE, "exp_ruling_links.py"))
rl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rl)   # also makes every Neo4j session read-only, and loads the app

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

import src.rag.defensive_generation as dg  # noqa: E402

PROMPT_VERSION = "cards-v1"
MAX_CHARS = 40000        # about 13k tokens: with the prompt and the answer, inside the model's 20k
HEAD_CHARS = 8000        # a longer ruling is read as its opening + its last part (reasons, P.Q.M.)
QUOTE_CHARS = 300
_RICORRENTI = ("imputato", "pubblico ministero", "parte civile", "più parti", "altro")
_ESITI = ("rigetto", "inammissibilità", "annullamento con rinvio", "annullamento senza rinvio",
          "accoglimento parziale", "condanna", "assoluzione", "altro")
_LATI = ("imputato", "accusa", "non chiaro")

SYSTEM = (
    "Sei un avvocato penalista italiano. Ti vengono dati una sentenza penale e l'elenco degli articoli "
    "che cita. Compila la scheda della sentenza usando solo ciò che la sentenza dice. Rispondi solo con "
    "questo JSON:\n"
    '{"articoli_applicati": [], "ricorrente": "", "esito": "", "a_favore_di": "", '
    '"argomento_decisivo": [], "parole_chiave_fatti": [], "parole_chiave_questioni": [], "citazione": ""}\n'
    "- articoli_applicati: solo gli articoli dell'elenco che il giudice applica o interpreta per decidere "
    "il caso, non quelli citati di passaggio (spese, forma del ricorso), scritti come nell'elenco.\n"
    "- ricorrente: chi ha impugnato: \"imputato\", \"pubblico ministero\", \"parte civile\", \"più parti\" "
    "o \"altro\".\n"
    "- esito: \"rigetto\", \"inammissibilità\", \"annullamento con rinvio\", \"annullamento senza rinvio\", "
    "\"accoglimento parziale\", \"condanna\", \"assoluzione\" o \"altro\".\n"
    "- a_favore_di: \"imputato\" se la decisione gli è favorevole, \"accusa\" se gli è sfavorevole, "
    "\"non chiaro\" altrimenti.\n"
    "- argomento_decisivo: da 2 a 5 parole chiave sulla ragione per cui il giudice decide così.\n"
    "- parole_chiave_fatti: da 3 a 8 parole chiave sui fatti concreti (per esempio \"schiaffo\", "
    "\"caduta\", \"trauma cranico\"), senza nomi di persone o di luoghi.\n"
    "- parole_chiave_questioni: da 3 a 8 parole chiave sulle questioni giuridiche (per esempio \"nesso "
    "causale\", \"prevedibilità dell'evento\").\n"
    "- citazione: una o due frasi copiate parola per parola dalla sentenza, in cui il giudice afferma il "
    f"principio decisivo; al massimo {QUOTE_CHARS} caratteri; senza nomi di persone."
)

# --- Prompt and checks (pure) --------------------------------------------------------

def ruling_text_for_prompt(text: str):
    """(text, read in part): a ruling over MAX_CHARS is cut to its opening and its last
    part, where the reasons and the P.Q.M. are."""
    if len(text) <= MAX_CHARS:
        return text, False
    return text[:HEAD_CHARS] + " [...] " + text[-(MAX_CHARS - HEAD_CHARS):], True


def human_message(text: str, articles: list) -> str:
    listed = ", ".join(articles) or "(nessuno)"
    return f"Articoli citati dalla sentenza: {listed}\n\nSENTENZA:\n{text}"


def _norm(text: str) -> str:
    text = (text or "").lower().replace("’", "'").replace("“", '"').replace("”", '"').replace("«", '"').replace("»", '"')
    return " ".join(text.split())


def quote_in_text(quote: str, text: str) -> bool:
    """The quote is in the ruling, up to case, spacing and quote marks; a quote cut with
    "..." counts when every piece of 20+ characters is there."""
    q, t = _norm(quote).strip(' "'), _norm(text)
    if len(q) < 20:
        return False
    if q in t:
        return True
    pieces = [p.strip(' ."') for p in re.split(r"\.\.\.|…|\[\.\.\.\]", q)]
    pieces = [p for p in pieces if p]
    return len(pieces) > 1 and all(len(p) >= 20 and p in t for p in pieces)


def _choice(value, allowed, default):
    v = _norm(str(value or "")).strip(' ."')
    return v if v in allowed else default


def _keywords(value, most: int) -> list:
    out = []
    for item in value if isinstance(value, list) else []:
        word = _norm(str(item)).strip(' ."')
        if word and len(word) <= 60 and word not in out:
            out.append(word)
    return out[:most]


def _article_key(raw: str):
    """'art. 584 c.p.' / '584 c.p.' / '584 cod. pen.' -> '584 c.p.', else None."""
    raw = str(raw or "").strip()
    found = rl.cited_articles(raw if re.match(r"(?i)art", raw) else "art. " + raw)
    return found[0] if found else None


def check_card(raw: dict, text: str, articles: list) -> dict:
    """The model's card, kept to the allowed values: articles only from the ruling's own
    list, the quote only when it is in the text."""
    raw = raw if isinstance(raw, dict) else {}
    proposed = [k for k in (_article_key(a) for a in raw.get("articoli_applicati") or []) if k]
    applied = [a for a in dict.fromkeys(proposed) if a in articles]
    quote = " ".join(str(raw.get("citazione") or "").split())[:QUOTE_CHARS + 50]
    verified = quote_in_text(quote, text)
    return {
        "articoli_applicati": applied,
        "articoli_scartati": len(set(proposed)) - len(applied),
        "ricorrente": _choice(raw.get("ricorrente"), _RICORRENTI, "altro"),
        "esito": _choice(raw.get("esito"), _ESITI, "altro"),
        "a_favore_di": _choice(raw.get("a_favore_di"), _LATI, "non chiaro"),
        "argomento_decisivo": _keywords(raw.get("argomento_decisivo"), 5),
        "parole_chiave_fatti": _keywords(raw.get("parole_chiave_fatti"), 8),
        "parole_chiave_questioni": _keywords(raw.get("parole_chiave_questioni"), 8),
        "citazione": quote if verified else "",
        "citazione_verificata": verified,
    }


_SEZ_RE = re.compile(r"\bsezion[ei]\s+(unit[ei]|prima|seconda|terza|quarta|quinta|sesta|settima|feriale|[ivx]+\b|[1-7]\b)",
                     re.IGNORECASE)
_SEZ = {"prima": "I", "seconda": "II", "terza": "III", "quarta": "IV", "quinta": "V", "sesta": "VI",
        "settima": "VII", "1": "I", "2": "II", "3": "III", "4": "IV", "5": "V", "6": "VI", "7": "VII"}
_COURT_ABBR = {"Cassazione": "Cass. pen.", "Corte d'appello": "App.", "Tribunale": "Trib.",
               "Giudice di pace": "G.d.P.", "Corte costituzionale": "Corte cost."}


def sezione_of(header: str) -> str:
    m = _SEZ_RE.search(header or "")
    if not m:
        return ""
    word = m.group(1).lower()
    if word.startswith("unit"):
        return "Sez. U."
    if word == "feriale":
        return "Sez. fer."
    return "Sez. " + _SEZ.get(word, word.upper())


def reference_of(link: dict, sezione: str) -> str:
    """'Cass. pen., Sez. IV, n. 12345, 12/03/2019': what is known of it, in that order."""
    parts = [_COURT_ABBR.get(link.get("court") or "", link.get("court") or ""), sezione,
             f"n. {link['number']}" if link.get("number") not in (None, "") else "", str(link.get("date") or "")]
    return ", ".join(p for p in parts if p)


def pick(links: list, branch: str, done: set) -> list:
    """The rulings still to card in `branch`, those citing a c.p. article first."""
    todo = [r for r in links if r["branch"] == branch and r["id"] not in done and r["chars"]]
    return sorted(todo, key=lambda r: (not any(a.endswith(" c.p.") for a in r["articles"]), r["id"]))


def _rules_side(favours: str) -> str:
    return {"imputato": "imputato", "contro l'imputato": "accusa"}.get(favours, "non chiaro")


# The card's outcome and appellant in step 1's coarser classes, so the two can be compared:
# step 1 read them from the P.Q.M. and the opening with fixed patterns, no model.
_OUTCOME_CLASS = {"rigetto": "rigettato", "inammissibilità": "inammissibile", "annullamento con rinvio": "accolto",
                  "annullamento senza rinvio": "accolto", "accoglimento parziale": "accolto"}
_RULES_OUTCOME_CLASS = {"rigettato": "rigettato", "inammissibile": "inammissibile", "accolto": "accolto",
                        "in parte accolto": "accolto"}
_RULES_APPELLANT = {"imputato": "imputato", "pm": "pubblico ministero", "parte civile": "parte civile"}


def rules_appellant(appellant: str) -> str:
    if not appellant or appellant in ("?", "-"):
        return ""
    return "più parti" if "+" in appellant else _RULES_APPELLANT.get(appellant, "")


def grounded(keywords: list, text: str) -> int:
    """How many keywords the ruling's text contains: every word of 4+ letters, by its
    first 5 letters ("prevedibilità" -> "preve"), so a keyword the model made up shows."""
    low = _norm(text)
    count = 0
    for kw in keywords:
        stems = [w[:5] for w in re.findall(r"\w{4,}", kw)]
        count += bool(stems) and all(s in low for s in stems)
    return count


def agreement(cards: list, field: str, rules_field: str, mapping=None) -> tuple:
    """(same, compared): the card's field against step 1's, where both give an answer.
    With `mapping`, a value it does not map ("condanna", a trial court's outcome) is skipped."""
    same = compared = 0
    for c in cards:
        mine = c.get(field) if mapping is None else mapping.get(c.get(field))
        theirs = c.get(rules_field)
        if mine and theirs and mine not in ("altro", "non chiaro"):
            compared += 1
            same += mine == theirs
    return same, compared


def card_report(cards: list, remaining: int = 0, workers: int = 1) -> list:
    ok = [c for c in cards if not c.get("errore")]
    out = [f"Cards: {len(ok)}  (failed: {len(cards) - len(ok)}: "
           f"{dict(Counter(c['errore'] for c in cards if c.get('errore')))})"]
    if not ok:
        return out
    n = len(ok)
    seconds = sorted(c["secondi"] for c in ok)
    out += [f"Seconds per ruling: median {seconds[n // 2]:.0f}, max {seconds[-1]:.0f}; read in part: "
            f"{sum(c['letta_in_parte'] for c in ok)}",
            f"Quote found word for word in the ruling: {rl._pct(sum(c['citazione_verificata'] for c in ok), n)}",
            f"Articles applied: median {sorted(len(c['articoli_applicati']) for c in ok)[n // 2]} per ruling; "
            f"none: {sum(not c['articoli_applicati'] for c in ok)}; proposed but not in the ruling's list: "
            f"{sum(c['articoli_scartati'] for c in ok)}",
            f"Full reference: {rl._pct(sum(c['riferimento_completo'] for c in ok), n)}",
            f"Who appealed: {rl._row(Counter(c['ricorrente'] for c in ok), n)}",
            f"Outcome: {rl._row(Counter(c['esito'] for c in ok), n)}",
            f"In favour of: {rl._row(Counter(c['a_favore_di'] for c in ok), n)}"]
    out.append("Model vs step 1's fixed patterns (no model), where both give an answer:")
    for label, field, rules_field, mapping in (
            ("in favour of", "a_favore_di", "a_favore_di_regole", None),
            ("outcome", "esito", "esito_regole", _OUTCOME_CLASS),
            ("who appealed", "ricorrente", "ricorrente_regole", None)):
        compared = [c for c in ok if c.get(rules_field) and c[rules_field] != "non chiaro"]
        same, n_both = agreement(compared, field, rules_field, mapping)
        out.append(f"   {label}: same {same} / {n_both}" + (f" ({same / n_both:.0%})" if n_both else ""))
    words = sum(c.get("parole_totali", 0) for c in ok)
    if words:
        out.append(f"Keywords found in the ruling's text: {rl._pct(sum(c.get('parole_nel_testo', 0) for c in ok), words)}")
    out += [f"Most used fact keywords: {rl._row(Counter(k for c in ok for k in c['parole_chiave_fatti']), top=15)}",
            f"Most used issue keywords: {rl._row(Counter(k for c in ok for k in c['parole_chiave_questioni']), top=15)}",
            f"Most applied articles: {rl._row(Counter(a for c in ok for a in c['articoli_applicati']), top=15)}"]
    if remaining:
        hours = remaining * seconds[n // 2] / max(workers, 1) / 3600
        out.append(f"Still to card: {remaining}; at this speed with {workers} worker(s): about {hours:.1f} h "
                   "(parallel calls are rarely fully parallel: take it as a lower bound)")
    return out


# --- The run ---------------------------------------------------------------------------

def _latest(path: str) -> list:
    """The cards in `path`, one per ruling: a ruling that failed and was carded later counts once."""
    if not os.path.exists(path):
        return []
    latest = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            card = json.loads(line)
            if not (card.get("errore") and card["id"] in latest and not latest[card["id"]].get("errore")):
                latest[card["id"]] = card
    return list(latest.values())


def _deadline(stop_at: str):
    if not stop_at:
        return None
    hour, minute = (int(x) for x in stop_at.split(":"))
    now = dt.datetime.now()
    when = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return when if when > now else when + dt.timedelta(days=1)


def _card(link: dict, text: str) -> dict:
    t0 = time.time()
    prompt_text, in_part = ruling_text_for_prompt(text)
    messages = [SystemMessage(content=SYSTEM), HumanMessage(content=human_message(prompt_text, link["articles"]))]
    raw, parsed = None, None
    for attempt in (1, 2):
        try:
            raw = dg._chat(messages, max_tokens=700, temperature=0)
        except Exception as exc:
            return {"id": link["id"], "errore": f"chiamata: {type(exc).__name__}"}
        try:
            parsed = dg.parse_json_reply(raw)
            break
        except ValueError:
            messages = [SystemMessage(content=SYSTEM + dg._JSON_RETRY_SUFFIX), messages[1]]
    if not isinstance(parsed, dict):
        return {"id": link["id"], "errore": "json"}
    sezione = sezione_of(text[:5000])
    card = {"id": link["id"], "prompt_version": PROMPT_VERSION, "riferimento": reference_of(link, sezione),
            "riferimento_completo": rl.full_reference(link), "court": link.get("court"), "sezione": sezione,
            "number": link.get("number"), "date": link.get("date"), "branch": link["branch"],
            "articoli_citati": link["articles"], "a_favore_di_regole": _rules_side(link.get("favours")),
            "esito_regole": _RULES_OUTCOME_CLASS.get(link.get("outcome"), ""),
            "ricorrente_regole": rules_appellant(link.get("appellant")),
            "chars": len(text), "letta_in_parte": in_part}
    card.update(check_card(parsed, text, link["articles"]))
    keywords = card["argomento_decisivo"] + card["parole_chiave_fatti"] + card["parole_chiave_questioni"]
    card["parole_totali"], card["parole_nel_testo"] = len(keywords), grounded(keywords, text)
    card["secondi"] = round(time.time() - t0, 1)
    return card


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--links", default="/tmp/ruling_links.jsonl", help="step 1's labels")
    parser.add_argument("--out", default="/tmp/ruling_cards.jsonl")
    parser.add_argument("--branch", default="penale")
    parser.add_argument("--limit", type=int, help="card at most N rulings in this run")
    parser.add_argument("--workers", type=int, default=1, help="parallel model calls (1 by day, 3 at night)")
    parser.add_argument("--stop-at", help="HH:MM server time: take no new rulings after it")
    parser.add_argument("--report", action="store_true", help="only print the report on the cards so far")
    args = parser.parse_args()

    links = [json.loads(line) for line in open(args.links, encoding="utf-8")]
    cards = _latest(args.out)
    done = {c["id"] for c in cards if not c.get("errore")}
    todo = pick(links, args.branch, done)
    if args.report:
        print("\n".join(card_report(cards, len(todo), args.workers)))
        return
    batch = todo[:args.limit] if args.limit else todo
    deadline = _deadline(args.stop_at)
    print(f"{len(done)} cards already; carding {len(batch)} of the {len(todo)} {args.branch} rulings left"
          + (f", no new ones after {deadline:%d/%m %H:%M}" if deadline else "") + f", {args.workers} worker(s)",
          file=sys.stderr)

    from src.rag.main import NEO4J_DATABASE, driver

    lock, new, t0 = threading.Lock(), [], time.time()

    def work(link):
        if deadline and dt.datetime.now() >= deadline:
            return None
        with driver.session(database=NEO4J_DATABASE) as session:
            text = " ".join(rl._ruling_sections(session, [link["id"]]).get(link["id"]) or [])
        return _card(link, text) if text else {"id": link["id"], "errore": "senza testo"}

    with open(args.out, "a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, link) for link in batch]
        for future in as_completed(futures):
            card = future.result()
            if card is None:
                continue
            with lock:
                out.write(json.dumps(card, ensure_ascii=False) + "\n")
                out.flush()
                new.append(card)
                if len(new) % 25 == 0 or len(new) == len(batch):
                    print(f"   {len(new)}/{len(batch)} carded, {time.time() - t0:.0f}s", file=sys.stderr)

    cards = _latest(args.out)
    done = {c["id"] for c in cards if not c.get("errore")}
    print("\n".join(card_report(cards, len(pick(links, args.branch, done)), args.workers)))
    print(f"\nThis run: {len(new)} rulings in {time.time() - t0:.0f}s. Nothing was written to the database. "
          f"Cards: {args.out}")


if __name__ == "__main__":
    main()
