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

# v2 (22 Oct): "più parti" only for different kinds of parties -- the trial's model used it for
# several defendants, and "who appealed" agreed with step 1 in 9 of 14.
PROMPT_VERSION = "cards-v2"
MAX_CHARS = 40000        # about 13k tokens: with the prompt and the answer, inside the model's 20k
HEAD_CHARS = 8000        # a longer ruling is read as its opening + its last part (reasons, P.Q.M.)
QUOTE_CHARS = 300
# Costs and the fine on a failed criminal appeal: applied in most rulings, but they say nothing
# about the case (the trial, despite the prompt, carded 616 c.p.p. as applied 3 times in 20).
COST_ARTICLES = {"616 c.p.p.", "592 c.p.p.", "535 c.p.p."}
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
    "- ricorrente: chi ha impugnato: \"imputato\" (anche quando gli imputati sono più d'uno), \"pubblico "
    "ministero\", \"parte civile\", \"più parti\" solo se hanno impugnato parti di tipo diverso (per esempio "
    "imputato e pubblico ministero), o \"altro\".\n"
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
# Asked once more when the card's quote is not in the ruling (the trial: 16 of 20 found).
QUOTE_SYSTEM = (
    "Copia dalla sentenza, parola per parola, una o due frasi consecutive in cui il giudice afferma il "
    "principio decisivo. Non cambiare, aggiungere o togliere nessuna parola. Al massimo "
    f"{QUOTE_CHARS} caratteri, senza nomi di persone. Rispondi solo con le frasi copiate, senza virgolette."
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
    # PDF text: a word split at the line end ("respon- sabilità"), soft hyphens. The model
    # copies the word whole (trial 3: 16 of 20 quotes found, 1 more at the second ask).
    text = re.sub(r"([^\W\d_])-\s+([^\W\d_])", r"\1\2", text.replace("­", ""))
    return " ".join(text.split())


def _words(text: str) -> str:
    return " " + " ".join(re.sub(r"[^\w]+", " ", _norm(text)).split()) + " "


def quote_in_text(quote: str, text: str) -> bool:
    """The quote's words are in the ruling, in order, up to case, spacing and punctuation
    (the 20-ruling trial found 16 quotes character for character; a copy that only moves
    a comma or spaces "c.p." should still count, a changed word should not). A quote cut
    with "..." counts when every piece of 20+ characters is there."""
    q = _norm(quote).strip(' "')
    if len(q) < 20:
        return False
    t = _words(text)
    pieces = [_words(p) for p in re.split(r"\.\.\.|…|\[\.\.\.\]", q)]
    pieces = [p for p in pieces if p.strip()]
    if len(pieces) == 1:
        return pieces[0] in t
    return bool(pieces) and all(len(p) >= 22 and p in t for p in pieces)


def quote_overlap(quote: str, text: str) -> float:
    """The longest run of the quote's words found in the ruling in a row, as a share of its
    words. Near 1: a copy with small differences (a dropped parenthesis, a PDF artefact);
    low: the model wrote the sentence itself. Only for the report on dropped quotes."""
    words, t = _words(quote).split(), _words(text)
    best = 0
    for i in range(len(words)):
        if len(words) - i <= best:
            break
        j = i + 1
        while j <= len(words) and f" {' '.join(words[i:j])} " in t:
            j += 1
        best = max(best, j - 1 - i)
    return best / len(words) if words else 0.0


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
    applied = [a for a in dict.fromkeys(proposed) if a in articles and a not in COST_ARTICLES]
    quote = " ".join(str(raw.get("citazione") or "").split())[:QUOTE_CHARS + 50]
    verified = quote_in_text(quote, text)
    return {
        "articoli_applicati": applied,
        "articoli_scartati": len({a for a in proposed if a not in articles}),
        "ricorrente": _choice(raw.get("ricorrente"), _RICORRENTI, "altro"),
        "esito": _choice(raw.get("esito"), _ESITI, "altro"),
        "a_favore_di": _choice(raw.get("a_favore_di"), _LATI, "non chiaro"),
        "argomento_decisivo": _keywords(raw.get("argomento_decisivo"), 5),
        "parole_chiave_fatti": _keywords(raw.get("parole_chiave_fatti"), 8),
        "parole_chiave_questioni": _keywords(raw.get("parole_chiave_questioni"), 8),
        "citazione": quote if verified else "",
        "citazione_verificata": verified,
        # kept on the server for the report on dropped quotes; never used as a quote
        "citazione_scartata": "" if verified else quote,
        "citazione_somiglianza": None if verified or not quote else round(quote_overlap(quote, text), 2),
    }


_SEZ_RE = re.compile(r"\bsezion[ei]\s+(unit[ei]|prima|seconda|terza|quarta|quinta|sesta|settima|feriale|[ivx]+\b|[1-7]\b)",
                     re.IGNORECASE)
_SEZ = {"prima": "I", "seconda": "II", "terza": "III", "quarta": "IV", "quinta": "V", "sesta": "VI",
        "settima": "VII", "1": "I", "2": "II", "3": "III", "4": "IV", "5": "V", "6": "VI", "7": "VII"}
_COURT_ABBR = {"Cassazione": "Cass. pen.", "Corte d'appello": "App.", "Tribunale": "Trib.",
               "Giudice di pace": "G.d.P.", "Corte costituzionale": "Corte cost.",
               "Corte d'assise": "Ass.", "Corte d'assise d'appello": "Ass. App."}
_ASSISE_RE = re.compile(r"corte\s+d(?:['’]\s*|i\s+)assise(\s+d(?:['’]\s*|i\s+)appello)?", re.IGNORECASE)


def court_for_card(link_court, header: str):
    """Step 1's court, unless the opening names a Corte d'assise (where homicide is tried),
    which step 1 does not know: there "Corte d'assise d'appello" reads as Corte d'appello."""
    m = _ASSISE_RE.search((header or "")[:600])
    if m:
        return "Corte d'assise d'appello" if m.group(1) else "Corte d'assise"
    return link_court


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


_LOWER_COURTS = ("Tribunale", "Corte d'appello", "Giudice di pace", "Corte d'assise", "Corte d'assise d'appello")


_ACT_TYPES = {"S": "sent.", "O": "ord.", "I": "ord. interl.", "D": "decr."}


def reference_of(link: dict, sezione: str, city: str = "") -> str:
    """'Cass. pen., Sez. IV, sent. n. 1234/2026, 12/03/2026' / 'Trib. Milano, n. 1234/2021,
    12/03/2021': what is known of it, in that order. `link` may carry "branch" (civil
    Cassazione is "Cass. civ.") and "type" (the Italgiure act type: S, O, I, D)."""
    court = _COURT_ABBR.get(link.get("court") or "", link.get("court") or "")
    if link.get("court") == "Cassazione" and link.get("branch") == "civile":
        court = "Cass. civ."
    number = ""
    if link.get("number") not in (None, ""):
        number = f"{_ACT_TYPES.get(link.get('type') or '', '')} n. {link['number']}".strip()
    parts = [f"{court} {city}".strip(), sezione, number, str(link.get("date") or "")]
    return ", ".join(p for p in parts if p)


def reference_complete(court, city, number, date) -> bool:
    """Court, number and date; for a lower court also its city ("Trib. Milano")."""
    if not (court and re.search(r"\d", str(number or "")) and date):
        return False
    return court not in _LOWER_COURTS or bool(city)


# About 890 of the 2,118 criminal rulings have no document_number / date_enacted (step 1, 22 Oct;
# about as many as the Tribunale rulings, 892). The ruling's own text usually has both.
_SENTENZA_N_RE = re.compile(r"\bsent(?:enza|\.)\s+n(?:\.|um\.|umero)?\s*(\d{1,6})\s*/\s*((?:19|20)?\d{2})\b",
                            re.IGNORECASE)
# The lower courts' heading: "N. 1234/2021 SENT.", "N. 1234/21 Reg. Sent." (next to the R.G. numbers).
_N_SENT_RE = re.compile(r"\bn(?:\.|um\.|umero)?\s*(\d{1,6})\s*/\s*((?:19|20)?\d{2})\s*(?:reg\.?\s*)?sent\b",
                        re.IGNORECASE)
_MONTHS = {m: i for i, m in enumerate(("gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio",
                                       "agosto", "settembre", "ottobre", "novembre", "dicembre"), 1)}
_DATE_RE = re.compile(r"(\d{1,2})\s*[/.\-]\s*(\d{1,2})\s*[/.\-]\s*((?:19|20)\d{2})"
                      r"|(\d{1,2})\s*[°º]?\s+(" + "|".join(_MONTHS) + r")\s+((?:19|20)\d{2})", re.IGNORECASE)
# The decision date first (how rulings are cited), then the hearing, then the filing.
_DATE_ANCHORS = (re.compile(r"cos[iì]\s+(?:è\s+)?decis\w*|decis[oa]\s+in\b", re.IGNORECASE),
                 re.compile(r"udienza\s+(?:pubblica\s+|camerale\s+)?del|pronunciat\w*\s+(?:in\s+data|il)\b",
                            re.IGNORECASE),
                 re.compile(r"depositat\w*", re.IGNORECASE))
_CITY_RE = re.compile(
    r"\b(?:tribunale(?:\s+ordinario|\s+di\s+sorveglianza|\s+del\s+riesame|\s+per\s+i\s+minorenni)?"
    r"|corte\s+d['’]\s*appello|corte\s+di\s+appello|corte\s+d(?:['’]\s*|i\s+)assise(?:\s+d(?:['’]\s*|i\s+)appello)?|giudice\s+di\s+pace)"
    r"\s+di\s+([a-zà-ù'’]+)(?:\s+([a-zà-ù'’]+))?", re.IGNORECASE)
_TWO_WORD_CITIES = {"reggio", "busto", "torre", "nocera", "lamezia", "castellammare", "barcellona", "termini",
                    "ascoli", "vibo", "santa", "sesto"}


def number_from_text(text: str) -> str:
    """'1234/2021' from "Sentenza n. 1234/2021" in the opening or the closing ('' if none).
    Only after "sentenza": the register numbers (R.G., R.G.N.R.) are not the ruling's."""
    for part in (text[:5000], text[-3000:]):
        m = _SENTENZA_N_RE.search(part) or _N_SENT_RE.search(part)
        if m:
            year = m.group(2)
            year = year if len(year) == 4 else ("20" if int(year) < 50 else "19") + year
            return f"{m.group(1)}/{year}"
    return ""


def date_from_text(text: str) -> str:
    """'12/03/2021': the first date within 120 characters after "Così deciso", else after
    "udienza del", else after "depositata" ('' if none)."""
    for anchor in _DATE_ANCHORS:
        for a in anchor.finditer(text):
            m = _DATE_RE.search(text, a.end(), a.end() + 120)
            if not m:
                continue
            if m.group(1):
                day, month, year = int(m.group(1)), int(m.group(2)), m.group(3)
            else:
                day, month, year = int(m.group(4)), _MONTHS[m.group(5).lower()], m.group(6)
            if 1 <= day <= 31 and 1 <= month <= 12:
                return f"{day:02d}/{month:02d}/{year}"
    return ""


def city_of(header: str) -> str:
    """'Milano' from "TRIBUNALE DI MILANO" ('Reggio Calabria' for the two-word ones), '' if none."""
    m = _CITY_RE.search(header or "")
    if not m:
        return ""
    words = [m.group(1)] + ([m.group(2)] if m.group(2) and m.group(1).lower() in _TWO_WORD_CITIES else [])
    return " ".join(w[:1].upper() + w[1:].lower() for w in words)


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


def reference_gaps(cards: list) -> str:
    """What the incomplete references lack: 'number 3, date 2, city 4' (a card may lack several)."""
    gaps = Counter()
    for c in cards:
        if c.get("riferimento_completo"):
            continue
        gaps["court"] += not c.get("court")
        gaps["number"] += not re.search(r"\d", str(c.get("number") or ""))
        gaps["date"] += not c.get("date")
        gaps["city"] += c.get("court") in _LOWER_COURTS and not c.get("citta")
    return ", ".join(f"{k} {v}" for k, v in gaps.items() if v)


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
            f"Quote found word for word in the ruling: {rl._pct(sum(c['citazione_verificata'] for c in ok), n)} "
            f"(at the second ask: {sum(c.get('citazione_ritentata', False) for c in ok)})",
            f"Articles applied: median {sorted(len(c['articoli_applicati']) for c in ok)[n // 2]} per ruling; "
            f"none: {sum(not c['articoli_applicati'] for c in ok)}; proposed but not in the ruling's list: "
            f"{sum(c['articoli_scartati'] for c in ok)}",
            f"Full reference: {rl._pct(sum(c['riferimento_completo'] for c in ok), n)} "
            f"(number or date read from the ruling's text: {sum(c.get('riferimento_dal_testo', False) for c in ok)}); "
            f"the others miss: {reference_gaps(ok) or '-'}",
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
    dropped = [c for c in ok if not c["citazione_verificata"]]
    if dropped:
        sims = [c.get("citazione_somiglianza") for c in dropped]
        out.append(f"Dropped quotes: {len(dropped)}; their longest word-for-word run in the ruling: "
                   f"90%+ of the quote {sum(s is not None and s >= 0.9 for s in sims)}, "
                   f"50-90% {sum(s is not None and 0.5 <= s < 0.9 for s in sims)}, "
                   f"under 50% {sum(s is not None and s < 0.5 for s in sims)}, no quote given {sims.count(None)}")
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
    italgiure = link.get("italgiure")
    if italgiure:   # an Italgiure file: Cassazione, its section, number and year in the file name
        court, city, has_number = "Cassazione", "", True
        sezione = "Sez. " + italgiure["section"]
        number = f"{italgiure['number']}/{italgiure['year']}"
    else:
        sezione = sezione_of(text[:5000])
        court = court_for_card(link.get("court"), text[:3000])
        city = city_of(text[:3000]) if court in _LOWER_COURTS else ""
        has_number = re.search(r"\d", str(link.get("number") or ""))
        number = link.get("number") if has_number else number_from_text(text)
    date = link.get("date") or date_from_text(text)
    ref = {"court": court, "number": number, "date": date, "branch": link["branch"],
           "type": (italgiure or {}).get("type")}
    card = {"id": link["id"], "prompt_version": PROMPT_VERSION, "riferimento": reference_of(ref, sezione, city),
            "riferimento_completo": reference_complete(court, city, number, date),
            "riferimento_dal_testo": bool((number and not has_number) or (date and not link.get("date"))),
            "court": court, "citta": city, "sezione": sezione,
            "number": number, "date": date, "branch": link["branch"],
            "articoli_citati": link["articles"], "a_favore_di_regole": _rules_side(link.get("favours")),
            "esito_regole": _RULES_OUTCOME_CLASS.get(link.get("outcome"), ""),
            "ricorrente_regole": rules_appellant(link.get("appellant")),
            "chars": len(text), "letta_in_parte": in_part}
    card.update(check_card(parsed, text, link["articles"]))
    card["citazione_ritentata"] = False
    if not card["citazione_verificata"]:
        try:
            again = dg._chat([SystemMessage(content=QUOTE_SYSTEM), HumanMessage(content="SENTENZA:\n" + prompt_text)],
                             max_tokens=200, temperature=0)
        except Exception:
            again = ""
        quote = " ".join(str(again or "").split()).strip(' "«»“”')[:QUOTE_CHARS + 50]
        if quote_in_text(quote, text):
            card.update(citazione=quote, citazione_verificata=True, citazione_ritentata=True,
                        citazione_scartata="", citazione_somiglianza=None)
    keywords =card["argomento_decisivo"] + card["parole_chiave_fatti"] + card["parole_chiave_questioni"]
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
