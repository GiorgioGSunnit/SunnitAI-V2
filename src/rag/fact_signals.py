"""FACT_SIGNALS_ENABLED: the case's decisive facts, typed, turned into legal questions.

diagnose_long run 1415 lost 8-11 of 122 expected articles at LOST_AT_UNDERSTANDING (1062 c.c., 1176 c.c.,
365 c.p., 120 c.p.a.): the facts were in the text but never became a legal question. Three steps:

  1. extract_facts: ONE call at temperature 0 with a fixed JSON schema (parties and their roles, the history
     of the property, possession, visible works, dated events, injuries, money, contracts, proceedings, the
     questions asked). The reply is validated, retried once, and a date is kept only as the text writes it.
  2. compute_terms: date arithmetic in code, never by the model -- the time between dated facts against a
     table of legal terms (TERMS), each with its article only as the institution's reference:
     "azione possessoria: termine annuale DECORSO (spoglio 2016, azione 2018)".
  3. fire_rules: general rules in fact_signal_rules.yaml, each {id, area, when, institution(s), codice,
     why, status}. A rule names INSTITUTIONS in words, never an article: the case analysis takes them as
     central issues, and _resolve_institution finds their articles as for any other.

Integration (defensive_generation._analyse_case_for_pipeline): the fired institutions join the frame's
central issues (so they are resolved, tagged _central and forced into the plan after grading); the terms go
to the plan and to the procedural section as facts (plan_then_write), so an expired action is never
proposed as open. scripts/export_fact_rules.py prints the rules for the lawyer; scripts/fact_signals_bench.py
measures them on a diagnose_long run's cases.
"""

import calendar
import datetime as dt
import json
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from . import feature_flags
from .ai_chat import _call_chat, parse_json_reply
from .verbose_logger import vlog

logger = logging.getLogger(__name__)

FACT_SIGNALS_FLAG = "FACT_SIGNALS_ENABLED"
# FACT_REGIMES_ENABLED (the lawyer's blind review of two new cases, 26 Oct): the parties' qualifications inferred
# from the facts (a natural person buying for private use from a business is a consumer; a public body; a public
# official), the special regimes they bring in as central issues, every applicable legal term of TERMS stated
# even when the case gives no dates, and the procedural tools the facts call for (PROCEDURAL_TOOLS). Runs the
# extraction also with FACT_SIGNALS_ENABLED off; the rules of fact_signal_rules.yaml fire only with it on.
REGIMES_FLAG = "FACT_REGIMES_ENABLED"
RULES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fact_signal_rules.yaml")
# YYYY-MM-DD: the case's present for a case that states none (tests, benches). Never the real today: an exam
# case set years ago would have every term "DECORSO" (fact bench, run 1415).
TODAY_ENV = "FACT_SIGNALS_TODAY"
MAX_CENTRAL = 6                         # institutions the fired rules add as central issues, in the rules' order
_CASE_CHARS = 12000
_EXTRACT_TOKENS = 3500                  # run 1415's longest cases (n2, n9) came back cut at 1800


def enabled() -> bool:
    return feature_flags.enabled(FACT_SIGNALS_FLAG)


def regimes_enabled() -> bool:
    return feature_flags.enabled(REGIMES_FLAG)


# --- 1. typed facts ------------------------------------------------------------------------------

ROLES = ("cliente", "controparte", "terzo", "indagato/imputato", "persona offesa", "pubblico ufficiale",
         "incaricato di pubblico servizio", "medico", "struttura sanitaria", "professionista", "datore di lavoro",
         "lavoratore", "consumatore", "impresa", "pubblica amministrazione", "minore", "erede", "coniuge",
         "locatore", "conduttore", "albergatore", "committente", "appaltatore")

SCHEMA: Dict[str, Any] = {
    "parties": [{"name": "", "role": "", "age": ""}],
    "client_side": "",
    "property_history": [{"asset": "", "owner": "", "from": "", "to": "", "how_acquired": ""}],
    "possession": [{"who": "", "asset": "", "since": "", "how": "", "still_ongoing": ""}],
    "works_or_signs": [{"what": "", "visible": "", "existing_since": ""}],
    "events": [{"date": "", "what": "", "who": ""}],
    "injuries": [{"who": "", "days_to_heal": "", "permanent": ""}],
    "money": [{"amount": "", "rate": "", "period": "", "threshold": ""}],
    "contracts": [{"type": "", "parties": "", "clauses": ""}],
    "proceedings": [{"stage": "", "authority": "", "decision": "", "date": ""}],
    "requests": [""],
    "present_date": "",
}
# FACT_REGIMES_ENABLED: what the parties' qualifications and the procedural tools are inferred from.
SCHEMA_REGIMES: Dict[str, Any] = {
    "parties": [{"name": "", "role": "", "age": "", "kind": "", "occupation": "", "purpose": "", "activities": ""}],
    "risks": [{"what": "", "kind": ""}],
}


def schema(regimes: Optional[bool] = None) -> Dict[str, Any]:
    """SCHEMA, with SCHEMA_REGIMES' fields when FACT_REGIMES_ENABLED (or regimes) is on."""
    return {**SCHEMA, **SCHEMA_REGIMES} if (regimes_enabled() if regimes is None else regimes) else SCHEMA

_EXTRACT_SYSTEM = (
    "Sei un giurista. Leggi un caso e riporta i suoi fatti in un JSON con ESATTAMENTE queste chiavi (lascia vuoto "
    "ciò che il testo non dice; non dedurre, non completare):\n"
    "- parties: le persone e gli enti, ciascuno {name, role, age}; role tra: " + ", ".join(ROLES)
    + " (più ruoli separati da virgola); age solo se il testo la dice.\n"
    "- client_side: chi assiste il parere (nome e ruolo), se il testo lo dice.\n"
    "- property_history: i passaggi di proprietà {asset (quale bene), owner (chi ne era proprietario), from, to "
    "(date come scritte), how_acquired (come lo ha acquistato: compravendita, donazione, successione, divisione...)}.\n"
    "- possession: chi possiede o usa un bene altrui {who, asset, since (da quando, come scritto), how (pacifico, "
    "pubblico, violento, clandestino, con tolleranza...), still_ongoing (sì/no)}.\n"
    "- works_or_signs: opere e segni visibili {what (strada, sentiero, finestra, condotta, muro...), visible (sì/no), "
    "existing_since (da quando, come scritto)}.\n"
    "- events: i fatti datati e non {date (ESATTAMENTE come scritta nel testo, o vuota), what (in una frase, "
    "neutra), who}.\n"
    "- injuries: lesioni e morte {who, days_to_heal (numero, se detto), permanent (sì/no, o \"morte\")}.\n"
    "- money: somme, interessi e soglie {amount, rate (tasso in %, se detto), period, threshold (soglia, se detta)}.\n"
    "- contracts: i contratti {type, parties, clauses (le clausole notate: penale, termine, caparra, esonero...)}.\n"
    "- proceedings: procedimenti e atti di autorità {stage, authority, decision, date (come scritta)}.\n"
    "- requests: le domande poste dal caso, una per voce.\n"
    "- present_date: la data che il caso presenta come attuale (\"oggi\", \"alla data odierna\", la data della "
    "consultazione), come scritta; vuota se il caso non la dice.\n"
    "Sii conciso: what in al massimo 20 parole, al massimo 25 eventi.\n"
    "Non inventare date: una data compare solo se è scritta nel testo, e come vi è scritta. Rispondi SOLO con il JSON."
)
_EXTRACT_REGIMES = (
    "\nInoltre:\n"
    "- per ogni parte anche kind (persona fisica, società, ente pubblico, impresa individuale), occupation (il "
    "mestiere, la professione o la funzione pubblica, se detti) e purpose (per che cosa ha contratto: «personale» "
    "se il testo mostra che contrae per sé o per la famiglia, fuori da un'attività d'impresa o professionale; "
    "«professionale» se per la sua attività; vuoto se il testo non permette di dirlo), activities (ciò che la "
    "parte ha fatto o fa nei fatti rispetto al bene o all'attività da cui nasce la vicenda: ha costruito, "
    "installato, venduto, progettato, gestisce, custodisce, manutiene, sorveglia..., separati da virgola).\n"
    "- risks: i pericoli e le urgenze che il testo riferisce {what (in una frase), kind: sicurezza (rischio per "
    "persone o cose), aggravamento (un danno che continua o peggiora), prova (lo stato dei luoghi o della cosa "
    "può cambiare o andare perduto: riparazioni, smaltimento, lavori), urgenza (un evento o una scadenza "
    "imminente)}; lista vuota se il testo non ne riferisce.\n"
)
_RETRY_SUFFIX = ("\n\nLa risposta precedente non era un JSON valido con quelle chiavi. Rispondi SOLO con l'oggetto "
                 "JSON, dal primo carattere.")


def _clean_text(value: Any) -> str:
    if isinstance(value, bool):
        return "sì" if value else "no"
    if isinstance(value, (int, float)):
        return str(value)
    return " ".join(str(value or "").split()) if isinstance(value, str) else ""


def validate_facts(raw: Any, case_text: str, regimes: Optional[bool] = None) -> Optional[Dict[str, Any]]:
    """The facts in SCHEMA's shape (schema(regimes); unknown keys dropped, every value a string or a list
    of string dicts), or None when raw is not a JSON object with at least one of SCHEMA's keys. A date the
    case text does not contain is emptied (never invented); each dated item gains date_iso, from code."""
    if not isinstance(raw, dict) or not set(raw) & set(SCHEMA):
        return None
    norm_case = " ".join((case_text or "").lower().split())
    out: Dict[str, Any] = {}
    for key, shape in schema(regimes).items():
        value = raw.get(key)
        if isinstance(shape, str):
            out[key] = _clean_text(value)
        elif shape == [""]:
            out[key] = [t for t in (_clean_text(v) for v in (value if isinstance(value, list) else [value])) if t]
        else:
            fields = list(shape[0])
            items = []
            for item in value if isinstance(value, list) else []:
                if not isinstance(item, dict):
                    continue
                clean = {f: _clean_text(item.get(f)) for f in fields}
                if any(clean.values()):
                    items.append(clean)
            out[key] = items
    present = out.get("present_date", "")
    if present and " ".join(present.lower().split()) not in norm_case:
        out["present_date"] = ""
    for key, field in (("events", "date"), ("proceedings", "date"), ("possession", "since"),
                       ("works_or_signs", "existing_since"), ("property_history", "from"), ("property_history", "to")):
        for item in out[key]:
            written = item.get(field, "")
            if written and " ".join(written.lower().split()) not in norm_case:
                item[field] = ""                          # not in the text as written: never kept
            parsed = parse_date(item.get(field, ""))
            item[f"{field}_iso"] = parsed[2] if parsed else ""
    return out


def extract_facts(case_text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """(facts or None, problem): one call at temperature 0, its reply validated (validate_facts) and,
    when it is not valid, asked once more."""
    human = f"CASO:\n{(case_text or '')[:_CASE_CHARS]}"
    base = _EXTRACT_SYSTEM.replace("Sii conciso:", _EXTRACT_REGIMES.lstrip("\n") + "Sii conciso:") \
        if regimes_enabled() else _EXTRACT_SYSTEM
    system, problem = base, ""
    for attempt in (1, 2):
        try:
            raw = _call_chat([SystemMessage(content=system), HumanMessage(content=human)],
                             max_tokens=_EXTRACT_TOKENS, temperature=0)
            facts = validate_facts(_parse_or_repair(raw), case_text)
        except Exception as exc:
            facts, problem = None, f"{type(exc).__name__}: {str(exc)[:160]}"
        if facts is not None:
            return facts, ""
        problem = problem or "reply is not a JSON object with the schema's keys"
        system = base + _RETRY_SUFFIX
    return None, problem


def _parse_or_repair(raw: str) -> Any:
    """The reply's JSON; a reply cut short (a long case past the token budget) is read up to its last complete
    value, its open brackets closed."""
    try:
        value = parse_json_reply(raw)
    except ValueError:
        value = None
    # parse_json_reply reads the first complete object it finds: in a cut reply, an inner event.
    if isinstance(value, dict) and set(value) & set(SCHEMA):
        return value
    try:
        return _repaired(raw)
    except ValueError:
        if value is None:
            raise
        return value


def _repaired(raw: str) -> Any:
    text = raw or ""
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in the reply")
    stack, in_string, escape, safe = [], False, False, None
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack:
                stack.pop()
            safe = (i + 1, list(stack))
            if not stack:
                break
    if not safe:
        raise ValueError("the reply is cut before any complete value")
    end, still_open = safe
    return json.loads(text[start:end] + "".join(reversed(still_open)))


# --- 2. dates and terms ----------------------------------------------------------------------------

_MONTHS = {m: i for i, m in enumerate(("gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio",
                                       "agosto", "settembre", "ottobre", "novembre", "dicembre"), 1)}
_MONTH_RE = "|".join(_MONTHS)
_DATE_PATTERNS = (
    (re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-]((?:19|20)\d{2})\b"), "dmy"),
    (re.compile(r"\b((?:19|20)\d{2})-(\d{1,2})-(\d{1,2})\b"), "ymd"),
    (re.compile(rf"\b(\d{{1,2}})(?:°|º)?\s+({_MONTH_RE})\s+((?:19|20)\d{{2}})\b", re.IGNORECASE), "d month y"),
    (re.compile(rf"\b({_MONTH_RE})\s+(?:del\s+|dell'\s*)?((?:19|20)\d{{2}})\b", re.IGNORECASE), "month y"),
    (re.compile(r"\b((?:19|20)\d{2})\b"), "y"),
)


def parse_date(text: str) -> Optional[Tuple[dt.date, dt.date, str]]:
    """(earliest, latest, iso) of the first date written in text: a day, a month ("marzo 2016") or a year
    ("2016", "nel 2016"); None when it holds none. Relative dates ("tre mesi dopo") are not dates."""
    for pattern, kind in _DATE_PATTERNS:
        m = pattern.search(text or "")
        if not m:
            continue
        try:
            if kind == "dmy":
                d = dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
                return d, d, d.isoformat()
            if kind == "ymd":
                d = dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                return d, d, d.isoformat()
            if kind == "d month y":
                d = dt.date(int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1)))
                return d, d, d.isoformat()
            if kind == "month y":
                year, month = int(m.group(2)), _MONTHS[m.group(1).lower()]
                last = calendar.monthrange(year, month)[1]
                return dt.date(year, month, 1), dt.date(year, month, last), f"{year}-{month:02d}"
            year = int(m.group(1))
            return dt.date(year, 1, 1), dt.date(year, 12, 31), str(year)
        except ValueError:
            continue
    return None


def add_period(d: dt.date, years: int = 0, months: int = 0, days: int = 0) -> dt.date:
    """d plus a period, calendar-wise (29 Feb + 1 year = 28 Feb)."""
    month = d.month - 1 + months + 12 * years
    year, month = d.year + month // 12, month % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return dt.date(year, month, day) + dt.timedelta(days=days)


def term_status(start: Tuple[dt.date, dt.date], end: Tuple[dt.date, dt.date], years: int = 0, months: int = 0,
                days: int = 0) -> str:
    """"DECORSO" when the period has surely run between start and end (even from the latest possible start
    to the earliest possible end), "NON DECORSO" when it surely has not, else "INCERTO"."""
    if end[0] > add_period(start[1], years, months, days):
        return "DECORSO"
    if end[1] <= add_period(start[0], years, months, days):
        return "NON DECORSO"
    return "INCERTO"


def present(facts: Optional[Dict[str, Any]]) -> Optional[Tuple[str, Tuple[dt.date, dt.date]]]:
    """(what, interval) of the case's present: its own present_date, else TODAY_ENV (tests, benches), else
    None -- the real today is never assumed."""
    parsed = parse_date((facts or {}).get("present_date", ""))
    if parsed:
        return "data attuale del caso", (parsed[0], parsed[1])
    value = (os.getenv(TODAY_ENV) or "").strip()
    try:
        return ("oggi", (dt.date.fromisoformat(value),) * 2) if value else None
    except ValueError:
        return None


def latest_fact(facts: Optional[Dict[str, Any]]) -> Optional[Tuple[str, Tuple[dt.date, dt.date], str]]:
    """(what, interval, written) of the latest dated fact: the case's present is no earlier than it."""
    items = _events(facts or {}, r".") + [s for s in _possession_starts(facts or {})]
    return max(((i[1], i[0], i[2]) for i in items), key=lambda x: x[1][0], default=None)


def _status_to_now(facts, start, years=0, months=0, days=0) -> Tuple[str, str, str]:
    """(status, what, written) of a term running to the case's present: against present() when known; else
    DECORSO only if it had already run by the latest dated fact, otherwise NON VERIFICABILE."""
    now = present(facts)
    if now:
        return term_status(start, now[1], years, months, days), now[0], now[1][0].isoformat()
    last = latest_fact(facts)
    if last and term_status(start, last[1], years, months, days) == "DECORSO":
        return "DECORSO", f"già al fatto più recente ({last[0]})", last[2]
    return "NON VERIFICABILE", "il caso non indica la data attuale", ""


def _low(text: Any) -> str:
    return " ".join(str(text or "").lower().split())


def _dated(items: List[Dict[str, str]], pattern: str, fields: Tuple[str, ...], date_field: str = "date") -> List[tuple]:
    """[(interval, what, written date)] of the items whose fields match pattern and whose date parses, earliest first."""
    out = []
    for item in items or []:
        text = " ".join(_low(item.get(f)) for f in fields)
        parsed = parse_date(item.get(date_field, ""))
        if parsed and re.search(pattern, text):
            out.append(((parsed[0], parsed[1]), " ".join(str(item.get(fields[0]) or "").split()[:6]),
                        item.get(date_field, "")))
    return sorted(out, key=lambda x: x[0][0])


def _events(facts: Dict[str, Any], pattern: str) -> List[tuple]:
    return sorted(_dated(facts.get("events"), pattern, ("what", "who"))
                  + _dated(facts.get("proceedings"), pattern, ("stage", "decision", "authority")),
                  key=lambda x: x[0][0])


class Term:
    """A legal term: its label, length, the article it belongs to (the institution's reference only), the
    facts it runs from and to (patterns over the dated events and proceedings; to today when nothing ends
    it), and when it applies at all."""

    def __init__(self, id: str, label: str, ref: str, start: str, end: str, years: int = 0, months: int = 0,
                 days: int = 0, applies: Optional[Callable[[Dict[str, Any]], bool]] = None, word: str = "",
                 start_items: Optional[Callable[[Dict[str, Any]], List[tuple]]] = None, since: str = "",
                 family: str = "civile", regimes_only: bool = False):
        self.id, self.label, self.ref, self.start, self.end = id, label, ref, start, end
        self.years, self.months, self.days, self.applies, self.word = years, months, days, applies, word
        self.start_items = start_items
        # FACT_REGIMES_ENABLED: what the term runs from, in words, for a case that gives no dates; the family of
        # the cases it belongs to; a term only listed with the flag on.
        self.since, self.family, self.regimes_only = since, family, regimes_only

    def length(self) -> str:
        parts = [(self.years, "anno", "anni"), (self.months, "mese", "mesi"), (self.days, "giorno", "giorni")]
        return " e ".join(f"{n} {one if n == 1 else many}" for n, one, many in parts if n)


def _has_contract(facts, pattern):
    return any(re.search(pattern, _low(c.get("type")) + " " + _low(c.get("clauses"))) for c in facts.get("contracts") or [])


def _has_role(facts, pattern):
    return any(re.search(pattern, _low(p.get("role"))) for p in facts.get("parties") or [])


def _possession_starts(facts):
    return [((p[0], p[1]), " ".join(str(item.get("asset") or "il bene").split()[:6]), item.get("since", ""))
            for item in facts.get("possession") or [] if _low(item.get("still_ongoing")) not in ("no", "false")
            for p in [parse_date(item.get("since", ""))] if p]


_SPOGLIO = (r"spogli|turbativ|molest|impedi|sbarr|chiu[ds]\w* (?:il |l'|la |lo )?(?:passaggio|accesso|cancello|strada|varco)"
            r"|recinz|lucchett|occupa")
_ACTION = r"\bazion|ricors|citazion|reintegr|manutenzion|domanda giudiziale|causa\b"
_DAMAGE = r"incident|sinistr|cadut|caduta|lesion|ferit|danneggi|investit|crollo|infortun|tamponament"
_ROAD = r"stradal|investit|tamponament|circolazion|veicol|auto\b|moto\b"
_INTERRUPT = r"diffid|messa in mora|richiesta di risarciment|raccomandat|lettera|citazion|ricors|pec\b"
_OFFENCE = r"aggred|percoss|lesion|minacc|ingiuri|diffam|truff|furt|danneggi|molest|atti persecut|violenz"
_SENTENCE = r"sentenz|condann|assolu"
_APPEAL = r"appell|impugn|ricorso per cassazione"
_ACT = r"notific|provvediment|diniego|esclusion|aggiudicazion|ordinanz|decreto|revoca|annullament"
_TENDER = r"gara|appalt|aggiudicaz|affidament"
_TAR = r"\btar\b|ricorso al tribunale amministrativo|ricors\w* giurisdizional|ricorso amministrativo"
_DISCOVERY = r"scopr|scopert|si accorg|accort|manifest|emers|compars\w* (?:di |delle |dei )?(?:crep|infiltraz|vizi|difett)"
_DEFECT = r"vizi|difett|guast|malfunzion|non funzion|crep|infiltraz|cedim|distacc|umidit"
_DENOUNCE = r"denunci|contestaz|reclam|segnal|comunica"
_DELIVERY = r"consegn|acquist|compr|ritir"
_WORKS_DONE = r"ultimat|consegn|complet|fine dei lavori|terminat|collaud"
_MOVABLE = r"\bauto|veicol|\bmoto|gioiell|quadr|orolog|bicicl|barca|opera d'arte|arred"


TERMS: List[Term] = [
    Term("possessoria", "azione possessoria (spoglio o turbativa)", "artt. 1168 e 1170 c.c.", _SPOGLIO, _ACTION,
         years=1, word="termine annuale", since="dallo spoglio o dalla turbativa"),
    Term("usucapione_ordinaria", "usucapione ordinaria di beni immobili", "art. 1158 c.c.", "", "", years=20,
         word="ventennio di possesso", start_items=lambda f: [s for s in _possession_starts(f) if not re.search(_MOVABLE, _low(s[1]))],
         since="di possesso continuato"),
    Term("usucapione_abbreviata", "usucapione abbreviata (solo con titolo idoneo trascritto e buona fede)",
         "art. 1159 c.c.", "", "", years=10, word="decennio di possesso",
         applies=lambda f: any(re.search(r"trascri|rogito|atto pubblico|compravendit|donazion", _low(p.get("how_acquired")))
                               for p in f.get("property_history") or []),
         start_items=lambda f: [s for s in _possession_starts(f) if not re.search(_MOVABLE, _low(s[1]))],
         since="dalla trascrizione del titolo"),
    Term("usucapione_mobili", "usucapione di beni mobili (10 anni in buona fede; 20 altrimenti)", "art. 1161 c.c.",
         "", "", years=10, word="decennio di possesso",
         start_items=lambda f: [s for s in _possession_starts(f) if re.search(_MOVABLE, _low(s[1]))],
         since="di possesso continuato"),
    Term("prescrizione_circolazione", "prescrizione del risarcimento da circolazione di veicoli", "art. 2947 c.c.",
         _ROAD, _INTERRUPT, years=2, applies=lambda f: not f.get("contracts"), since="dal fatto"),
    Term("prescrizione_extracontrattuale", "prescrizione del risarcimento del danno extracontrattuale",
         "art. 2947 c.c.", _DAMAGE, _INTERRUPT, years=5, applies=lambda f: not f.get("contracts"), since="dal fatto"),
    Term("prescrizione_ordinaria", "prescrizione ordinaria (diritti da contratto)", "art. 2946 c.c.",
         r"inadempi|non (?:ha )?pagat|scadenz|ritard|mancat\w* (?:pagament|consegn)", _INTERRUPT, years=10,
         applies=lambda f: bool(f.get("contracts")), since="dal giorno in cui il diritto può essere fatto valere"),
    Term("querela", "querela", "art. 124 c.p.", _OFFENCE, r"querel|denunci", months=3,
         since="dalla notizia del fatto che costituisce il reato", family="penale"),
    Term("ricorso_tar_appalti", "ricorso al TAR in materia di appalti pubblici (rito speciale)", "art. 120 c.p.a.",
         _ACT, _TAR, days=30,
         applies=lambda f: any(re.search(_TENDER, _low(" ".join(map(str, e.values()))))
                               for e in (f.get("events") or []) + (f.get("proceedings") or [])),
         since="dalla comunicazione o conoscenza dell'atto", family="amministrativo"),
    Term("ricorso_tar", "ricorso al TAR per l'annullamento", "artt. 29 e 41 c.p.a.", _ACT, _TAR, days=60,
         applies=lambda f: not any(re.search(_TENDER, _low(" ".join(map(str, e.values()))))
                                   for e in (f.get("events") or []) + (f.get("proceedings") or []))
         and _has_role(f, r"pubblica amministrazione|comune|ente"),
         since="dalla notificazione, comunicazione o piena conoscenza dell'atto", family="amministrativo"),
    Term("denuncia_vizi_vendita", "denuncia dei vizi della cosa venduta", "art. 1495 c.c.", _DISCOVERY + "|" + _DEFECT,
         _DENOUNCE, days=8, applies=lambda f: _has_contract(f, r"vendit|compravendit|acquist")
         and (not _has_role(f, "consumatore") or bool(f.get("_regimes"))), since="dalla scoperta del vizio"),
    Term("azione_vizi_vendita", "azione di garanzia per i vizi della cosa venduta", "art. 1495 c.c.", _DELIVERY,
         _ACTION, years=1, applies=lambda f: _has_contract(f, r"vendit|compravendit|acquist")
         and (not _has_role(f, "consumatore") or bool(f.get("_regimes"))), since="dalla consegna"),
    Term("denuncia_conformita_consumo", "denuncia del difetto di conformità (consumatore)", "Codice del Consumo",
         _DISCOVERY + "|" + _DEFECT, _DENOUNCE, months=2,
         applies=lambda f: _has_role(f, "consumatore") and _has_contract(f, r"vendit|compravendit|acquist"),
         since="dalla scoperta del difetto"),
    Term("responsabilita_conformita_consumo", "responsabilità del venditore per il difetto di conformità "
         "(consumatore)", "Codice del Consumo", _DELIVERY, _DISCOVERY + "|" + _DEFECT,
         years=2, word="manifestazione del difetto entro 2 anni", applies=lambda f: _has_role(f, "consumatore") and _has_contract(f, r"vendit|compravendit|acquist"),
         since="dalla consegna del bene", regimes_only=True),
    Term("azione_conformita_consumo", "prescrizione dell'azione per il difetto di conformità non dolosamente "
         "occultato (consumatore)", "Codice del Consumo", _DELIVERY, _ACTION, months=26,
         applies=lambda f: _has_role(f, "consumatore") and _has_contract(f, r"vendit|compravendit|acquist"),
         since="dalla consegna del bene", regimes_only=True),
    Term("denuncia_difformita_appalto", "denuncia delle difformità e dei vizi dell'opera (appalto)", "art. 1667 c.c.",
         _DISCOVERY + "|" + _DEFECT, _DENOUNCE, days=60,
         applies=lambda f: _has_contract(f, r"appalt|lavori|ristruttur|costruz"), since="dalla scoperta"),
    Term("azione_garanzia_appalto", "azione di garanzia nell'appalto", "art. 1667 c.c.", _WORKS_DONE, _ACTION, years=2,
         applies=lambda f: _has_contract(f, r"appalt|lavori|ristruttur|costruz"), since="dalla consegna dell'opera"),
    Term("manifestazione_gravi_difetti", "manifestazione dei gravi difetti entro dieci anni dal compimento dell'opera",
         "art. 1669 c.c.", _WORKS_DONE, _DISCOVERY + "|" + _DEFECT, years=10, word="decennio dal compimento",
         applies=lambda f: _has_contract(f, r"appalt|lavori|ristruttur|costruz"), since="dal compimento dell'opera"),
    Term("denuncia_gravi_difetti", "denuncia dei gravi difetti dell'immobile", "art. 1669 c.c.",
         _DISCOVERY + "|" + _DEFECT, _DENOUNCE, years=1,
         applies=lambda f: _has_contract(f, r"appalt|lavori|ristruttur|costruz"), since="dalla scoperta"),
]

# Criminal appeals run from the deposit of the reasons (art. 585 c.p.p.): 15 days with the reasons read in
# court, 30 when deposited within 15 days, 45 within a longer term (at most 90 days).
_APPEAL_REF = "art. 585 c.p.p."


def _when(items: List[tuple], after: dt.date) -> Optional[tuple]:
    return next((i for i in items if i[0][1] >= after), None)


def _short(written: str) -> str:
    return written or "data non indicata"


def compute_terms(facts: Optional[Dict[str, Any]], areas: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Every term of TERMS the facts date: its status between the fact it runs from and the act that
    ends it, or the case's present when none does (_status_to_now); one line of text each. FACT_REGIMES_
    ENABLED (facts carrying "_regimes"): also every term that applies to facts the case states without a
    date, as DA VERIFICARE (_undated_entry), within the families of the case's areas."""
    if not facts:
        return []
    undated = bool(facts.get("_regimes"))
    families = {family(a) for a in areas or []} - {""}
    out = []
    for term in TERMS:
        if term.regimes_only and not undated:
            continue
        if term.applies and not term.applies(facts):
            continue
        starts = term.start_items(facts) if term.start_items else _events(facts, term.start)
        if not starts:
            if undated and (not families or term.family in families) and _mentioned(facts, term):
                out.append(_undated_entry(term))
            continue
        start = starts[0]
        ends = _events(facts, term.end) if term.end else []
        end = _when([e for e in ends if e[1] != start[1] or e[0] != start[0]], start[0][0])
        if end:
            status, end_what, end_written = term_status(start[0], end[0], term.years, term.months, term.days), end[1], end[2]
        else:
            status, end_what, end_written = _status_to_now(facts, start[0], term.years, term.months, term.days)
        out.append(_term_entry(term.id, term.label, term.ref, term.word or f"termine di {term.length()}",
                               term.length(), status, start, end_what, end_written))
    sentence = _events(facts, _SENTENCE)
    if sentence:
        start = sentence[-1]
        deposit = _events(facts, r"deposit\w* (?:delle |della )?motivazion")
        appeal = _when(_events(facts, _APPEAL), start[0][0])
        if deposit:
            base, sure, latest = deposit[-1], 30, 45
        else:
            base, sure, latest = start, 15, 90 + 45
        now = present(facts)
        if appeal or now:
            end_iv, end_what, end_written = ((appeal[0], appeal[1], appeal[2]) if appeal
                                             else (now[1], now[0], now[1][0].isoformat()))
            status = ("DECORSO" if end_iv[0] > add_period(base[0][1], days=latest) else
                      "NON DECORSO" if end_iv[1] <= add_period(base[0][0], days=sure) else "INCERTO")
        else:
            status, end_what, end_written = _status_to_now(facts, base[0], days=latest)
        length = "15, 30 o 45 giorni dal deposito della motivazione"
        out.append(_term_entry("impugnazione_penale", "impugnazione della sentenza penale", _APPEAL_REF,
                               f"termine di {length}", length, status, base, end_what, end_written,
                               note="" if deposit else "dipende da quando è depositata la motivazione"))
    return out


def _mentioned(facts: Dict[str, Any], term: Term) -> bool:
    """The case states, without a usable date, the fact a term runs from."""
    if term.start_items:
        movable = term.id == "usucapione_mobili"
        return any(bool(re.search(_MOVABLE, _low(p.get("asset")))) == movable
                   for p in facts.get("possession") or [] if _low(p.get("still_ongoing")) not in ("no", "false"))
    return bool(term.start) and bool(re.search(term.start, _texts(facts, ("events", "proceedings", "contracts",
                                                                             "works_or_signs", "injuries", "risks"))))


def _undated_entry(term: Term) -> Dict[str, Any]:
    word = term.word or f"termine di {term.length()}"
    text = (f"{term.label}: {word} {term.since}".rstrip() + " — DA VERIFICARE: il caso non indica le date da cui "
            "decorre")
    return {"id": term.id, "label": term.label, "ref": term.ref, "length": term.length(), "status": "DA VERIFICARE",
            "from": {"what": term.since, "date": ""}, "to": {"what": "", "date": ""}, "text": text, "undated": True}


def _term_entry(id, label, ref, word, length, status, start, end_what, end_written, note=""):
    text = (f"{label}: {word} {status} ({start[1] or 'fatto'} {_short(start[2])}, {end_what} "
            f"{end_written})".replace(" )", ")") + (f"; {note}" if note and status == "INCERTO" else ""))
    return {"id": id, "label": label, "ref": ref, "length": length, "status": status,
            "from": {"what": start[1], "date": start[2]}, "to": {"what": end_what, "date": end_written}, "text": text}


# --- 3. rules --------------------------------------------------------------------------------------

def _texts(facts: Dict[str, Any], keys: Tuple[str, ...] = ()) -> str:
    """Every string of the facts (or of keys), lower case, one per line."""
    out = []
    for key, value in facts.items():
        if keys and key not in keys:
            continue
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, dict):
                out += [_low(v) for k, v in item.items() if not k.endswith("_iso") and v]
            elif item:
                out.append(_low(item))
    return "\n".join(out)


def _number(text: Any) -> Optional[float]:
    m = re.search(r"\d+(?:[.,]\d+)?", str(text or ""))
    return float(m.group(0).replace(",", ".")) if m else None


def _yes(text: Any) -> bool:
    return _low(text) in ("sì", "si", "true", "yes", "vero", "visibile") or _low(text).startswith("sì")


def _p_fact(facts, terms, arg):
    return bool(re.search(arg["pattern"], _texts(facts, ("events", "works_or_signs", "injuries", "contracts",
                                                          "proceedings", "property_history", "possession", "money",
                                                          "client_side"))))


def _p_request(facts, terms, arg):
    return bool(re.search(arg["pattern"], _texts(facts, ("requests",))))


def _p_role(facts, terms, arg):
    return any(any(r in _low(p.get("role")) for r in arg) for p in facts.get("parties") or [])


def _p_role_count(facts, terms, arg):
    return sum(1 for p in facts.get("parties") or [] if any(r in _low(p.get("role")) for r in arg["roles"])) >= arg["min"]


def _p_age_under(facts, terms, arg):
    return any((_number(p.get("age")) or 999) < arg for p in facts.get("parties") or [])


def _p_contract(facts, terms, arg):
    for c in facts.get("contracts") or []:
        if arg.get("type") and not re.search(arg["type"], _low(c.get("type"))):
            continue
        if arg.get("clause") and not re.search(arg["clause"], _low(c.get("clauses"))):
            continue
        return True
    return False


def _p_proceeding(facts, terms, arg):
    for p in facts.get("proceedings") or []:
        text = " ".join(_low(p.get(f)) for f in ("stage", "decision", "authority"))
        if re.search(arg["pattern"], text) and (not arg.get("authority") or re.search(arg["authority"], _low(p.get("authority")))):
            return True
    return False


def _p_injury(facts, terms, arg):
    for i in facts.get("injuries") or []:
        days, permanent = _number(i.get("days_to_heal")), _low(i.get("permanent"))
        if arg.get("death") and "mort" not in permanent:
            continue
        if arg.get("min_days") and not ((days or 0) > arg["min_days"] or _yes(permanent) or "mort" in permanent):
            continue
        return True
    return False


def _p_money_over_threshold(facts, terms, arg):
    for m in facts.get("money") or []:
        rate, threshold = _number(m.get("rate")), _number(m.get("threshold"))
        if rate is not None and ((threshold is not None and rate > threshold) or
                                 (arg.get("rate_over") is not None and rate > arg["rate_over"])):
            return True
    return False


def _held_for(facts, since: str, years: int) -> bool:
    """A state that began at since lasted at least years by the case's present (_status_to_now)."""
    parsed = parse_date(since)
    return bool(parsed) and _status_to_now(facts, (parsed[0], parsed[1]), years)[0] == "DECORSO"


def _p_possession(facts, terms, arg):
    for p in facts.get("possession") or []:
        how = _low(p.get("how"))
        if arg.get("how") and not any(h in how for h in arg["how"]):
            continue
        if arg.get("how_not") and any(h in how for h in arg["how_not"]):
            continue
        if arg.get("min_years") and not _held_for(facts, p.get("since", ""), arg["min_years"]):
            continue
        return True
    return False


def _p_work(facts, terms, arg):
    for w in facts.get("works_or_signs") or []:
        if arg.get("visible") and not _yes(w.get("visible")):
            continue
        if arg.get("pattern") and not re.search(arg["pattern"], _low(w.get("what"))):
            continue
        if arg.get("min_years") and not _held_for(facts, w.get("existing_since", ""), arg["min_years"]):
            continue
        return True
    return False


def _p_common_owner(facts, terms, arg):
    by_owner: Dict[str, set] = {}
    for p in facts.get("property_history") or []:
        owner = _low(p.get("owner"))
        if owner:
            by_owner.setdefault(owner, set()).add(_low(p.get("asset")))
        if owner and re.search(r"divisi|frazionat|separat|lottizz", _low(p.get("how_acquired")) + " " + _low(p.get("asset"))):
            return True
    return any(len(assets) >= 2 for assets in by_owner.values())


def _p_term(facts, terms, arg):
    wanted = arg["status"] if isinstance(arg.get("status"), list) else [arg.get("status", "DECORSO")]
    ids = arg["id"] if isinstance(arg["id"], list) else [arg["id"]]
    return any(t["id"] in ids and t["status"] in wanted for t in terms)


def _p_not(facts, terms, arg):
    return not holds(arg, facts, terms)


def _p_any(facts, terms, arg):
    return any(holds(c, facts, terms) for c in arg)


def _d_roles(arg):
    return " o ".join(arg)


PREDICATES: Dict[str, Tuple[Callable, Callable]] = {
    "fact": (_p_fact, lambda a: a.get("label") or f"nei fatti: «{a['pattern']}»"),
    "request": (_p_request, lambda a: a.get("label") or f"tra le domande: «{a['pattern']}»"),
    "role": (_p_role, lambda a: f"una parte è {_d_roles(a)}"),
    "role_count": (_p_role_count, lambda a: f"almeno {a['min']} parti sono {_d_roles(a['roles'])}"),
    "age_under": (_p_age_under, lambda a: f"una parte ha meno di {a} anni"),
    "contract": (_p_contract, lambda a: a.get("label") or "un contratto"),
    "proceeding": (_p_proceeding, lambda a: a.get("label") or f"un procedimento: «{a['pattern']}»"),
    "injury": (_p_injury, lambda a: "la morte di una persona" if a.get("death") else
               f"lesioni oltre {a['min_days']} giorni o permanenti" if a.get("min_days") else "lesioni o morte"),
    "money_over_threshold": (_p_money_over_threshold, lambda a: "un tasso di interesse oltre la soglia indicata"
                             + (f" (o oltre il {a['rate_over']}%)" if a.get("rate_over") is not None else "")),
    "possession": (_p_possession, lambda a: "il possesso di un bene altrui"
                   + (f" da almeno {a['min_years']} anni" if a.get("min_years") else "")
                   + (f", {' o '.join(a['how'])}" if a.get("how") else "")
                   + (f", non {' né '.join(a['how_not'])}" if a.get("how_not") else "")),
    "work": (_p_work, lambda a: a.get("label") or ("un'opera visibile" if a.get("visible") else "un'opera")
             + (f" esistente da almeno {a['min_years']} anni" if a.get("min_years") else "")),
    "common_previous_owner": (_p_common_owner, lambda a: "i beni appartenevano a un unico proprietario, poi diviso o "
                              "alienato in parti"),
    "term": (_p_term, lambda a: f"termine calcolato «{a['id'] if isinstance(a['id'], str) else ' o '.join(a['id'])}» "
             f"{a.get('status', 'DECORSO') if isinstance(a.get('status', 'DECORSO'), str) else ' o '.join(a['status'])}"),
    "not": (_p_not, lambda a: f"NON ({describe([a])})"),
    "any": (_p_any, lambda a: "(" + " oppure ".join(describe([c]) for c in a) + ")"),
}


def holds(condition: Dict[str, Any], facts: Dict[str, Any], terms: List[Dict[str, Any]]) -> bool:
    """One condition ({predicate: argument}) on the facts and the computed terms."""
    (name, arg), = condition.items()
    return PREDICATES[name][0](facts, terms, arg)


def describe(conditions: List[Dict[str, Any]]) -> str:
    """The conditions in Italian, for the lawyer (scripts/export_fact_rules.py)."""
    out = []
    for condition in conditions:
        (name, arg), = condition.items()
        out.append(PREDICATES[name][1](arg))
    return " e ".join(out)


def load_rules(path: str = RULES_FILE) -> List[Dict[str, Any]]:
    """The rules of path, checked: each with id, area, when (known predicates), institution(s) and codice."""
    import yaml
    with open(path, encoding="utf-8") as fh:
        rules = yaml.safe_load(fh) or []
    seen = set()
    for rule in rules:
        missing = [k for k in ("id", "area", "when", "institution", "codice", "why", "status") if k not in rule]
        if missing:
            raise ValueError(f"rule {rule.get('id')!r}: missing {', '.join(missing)}")
        for lo_hi in rule.get("ranges") or []:
            if not (isinstance(lo_hi, list) and len(lo_hi) == 2 and all(isinstance(n, int) for n in lo_hi)):
                raise ValueError(f"rule {rule['id']!r}: a range is [first, last] article number")
        if rule["id"] in seen:
            raise ValueError(f"rule {rule['id']!r} twice")
        seen.add(rule["id"])
        _check_conditions(rule["id"], rule["when"])
        if re.search(r"\bart(?:icol[oi]|t)?\.?\s*\d", " ".join(_institutions(rule)), re.IGNORECASE):
            raise ValueError(f"rule {rule['id']!r}: an institution names an article; rules name institutions only")
    return rules


def _check_conditions(rule_id: str, conditions: Any) -> None:
    if not isinstance(conditions, list) or not conditions:
        raise ValueError(f"rule {rule_id!r}: when is not a list of conditions")
    for condition in conditions:
        if not isinstance(condition, dict) or len(condition) != 1 or next(iter(condition)) not in PREDICATES:
            raise ValueError(f"rule {rule_id!r}: unknown condition {condition!r}")
        (name, arg), = condition.items()
        if name == "any":
            _check_conditions(rule_id, arg)
        elif name == "not":
            _check_conditions(rule_id, [arg])
        elif isinstance(arg, dict) and "pattern" in arg:
            re.compile(arg["pattern"])


def _institutions(rule: Dict[str, Any]) -> List[str]:
    value = rule.get("institution")
    return [value] if isinstance(value, str) else list(value or [])


def family(area: Any) -> str:
    """An area as criminal, administrative or civil ("" for none or "altro")."""
    area = str(area or "").strip().lower()
    if not area or area == "altro":
        return ""
    return area if area in ("penale", "amministrativo") else "civile"


def rule_families(rule: Dict[str, Any]) -> set:
    """The families a rule belongs to: its own "families", else its area's."""
    return set(rule.get("families") or [family(rule["area"])])


def fire_rules(facts: Optional[Dict[str, Any]], terms: List[Dict[str, Any]],
               rules: Optional[List[Dict[str, Any]]] = None, areas: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """The rules whose conditions all hold, in the rules' order: {id, area, institutions, codice, ranges,
    why, matched (each condition with the fact it matched)}. With the case's areas (the frame's), a rule of
    another family (criminal, administrative, civil) does not fire (fact bench, run 1415: civil damages in
    criminal cases)."""
    if not facts:
        return []
    families = {family(a) for a in areas or []} - {""}
    fired = []
    for rule in rules if rules is not None else load_rules():
        if families and not rule_families(rule) & families:
            continue
        try:
            ok = all(holds(c, facts, terms) for c in rule["when"])
        except Exception as exc:
            logger.warning("fact signals: rule %s failed: %s", rule.get("id"), exc)
            ok = False
        if ok:
            fired.append({"id": rule["id"], "area": rule["area"], "institutions": _institutions(rule),
                          "codice": rule["codice"], "ranges": rule.get("ranges") or [], "why": rule["why"],
                          "status": rule["status"], "matched": why_fired(rule, facts, terms)})
    return fired


def why_fired(rule: Dict[str, Any], facts: Dict[str, Any], terms: List[Dict[str, Any]]) -> List[str]:
    """Each condition of a fired rule in words, with the fact that met it: "un fondo senza accesso alla via
    pubblica — «il fondo di tizio è intercluso»"."""
    out = []
    for condition in rule["when"]:
        found = _evidence(condition, facts, terms)
        out.append(describe([condition]) + (f" — «{found}»" if found else ""))
    return out


def _evidence(condition: Dict[str, Any], facts: Dict[str, Any], terms: List[Dict[str, Any]]) -> str:
    (name, arg), = condition.items()

    def line(text, pattern):
        return next((ln[:140] for ln in text.split("\n") if re.search(pattern, ln)), "")
    if name == "fact":
        return line(_texts(facts, ("events", "works_or_signs", "injuries", "contracts", "proceedings",
                                   "property_history", "possession", "money", "client_side")), arg["pattern"])
    if name == "request":
        return line(_texts(facts, ("requests",)), arg["pattern"])
    if name == "proceeding":
        return line(_texts(facts, ("proceedings",)), arg["pattern"])
    if name in ("role", "role_count"):
        roles = arg if name == "role" else arg["roles"]
        return "; ".join(f"{p.get('name')} ({p.get('role')})" for p in facts.get("parties") or []
                         if any(r in _low(p.get("role")) for r in roles))[:140]
    if name == "age_under":
        return "; ".join(f"{p.get('name')}, {p.get('age')} anni" for p in facts.get("parties") or [] if p.get("age"))
    if name == "contract":
        return "; ".join(f"{c.get('type')}" + (f" ({c.get('clauses')})" if c.get("clauses") else "")
                         for c in facts.get("contracts") or [])[:140]
    if name == "injury":
        return "; ".join(f"{i.get('who')}: {i.get('days_to_heal') or '-'} giorni, {i.get('permanent') or '-'}"
                         for i in facts.get("injuries") or [])[:140]
    if name == "money_over_threshold":
        return "; ".join(f"tasso {m.get('rate')}, soglia {m.get('threshold') or '-'}" for m in facts.get("money") or [])
    if name == "possession":
        return "; ".join(f"{p.get('who')} possiede {p.get('asset')} dal {p.get('since') or '?'} ({p.get('how') or '-'})"
                         for p in facts.get("possession") or [])[:140]
    if name == "work":
        return "; ".join(f"{w.get('what')}, visibile {w.get('visible') or '?'}, dal {w.get('existing_since') or '?'}"
                         for w in facts.get("works_or_signs") or [])[:140]
    if name == "common_previous_owner":
        return "; ".join(f"{p.get('asset')} di {p.get('owner')}" for p in facts.get("property_history") or [])[:140]
    if name == "term":
        ids = arg["id"] if isinstance(arg["id"], list) else [arg["id"]]
        return next((t["text"] for t in terms if t["id"] in ids), "")
    if name == "any":
        held = next((c for c in arg if holds(c, facts, terms)), None)
        return _evidence(held, facts, terms) if held else ""
    return ""


# --- the institution's articles ----------------------------------------------------------------------

_STOP = {"della", "delle", "dello", "degli", "dell", "nella", "nelle", "nello", "negli", "dalla", "dalle", "dallo",
         "dagli", "alla", "alle", "allo", "agli", "come", "sono", "altro", "altra", "altri", "diritto", "codice", "norme",
         "norma", "articolo", "caso", "cose", "parte", "parti", "ogni", "suoi", "loro"}


def _stems(text: str) -> set:
    return {w[:5] for w in re.findall(r"[a-zà-ù]{4,}", (text or "").lower()) if w not in _STOP}


def _in_ranges(number: str, ranges: List[List[int]]) -> bool:
    m = re.match(r"\d+", str(number or ""))
    return bool(m) and any(lo <= int(m.group(0)) <= hi for lo, hi in ranges)


def keep_resolved(istituto: Dict[str, Any], rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The rows a fact rule's institution (its "fonte") may resolve to: within the rule's ranges of article
    numbers when it has them (danno non patrimoniale resolved to art. 2634 c.c., company law: run 1415's fact
    bench), and only an article whose rubrica shares a content word with the institution. Any other
    institution's rows unchanged."""
    if not istituto.get("fonte"):
        return rows
    ranges, words = istituto.get("ranges") or [], _stems(istituto.get("nome", ""))
    kept = []
    for row in rows:
        number = (row.get("_article") or ("",))[0]
        s = row.get("s") or {}
        rubrica = " ".join(str(s.get("titolo_articolo") or s.get("title") or "").split())
        if ranges and not _in_ranges(number, ranges):
            continue
        if not words & _stems(rubrica):
            continue
        kept.append(row)
    return kept


def central_issues(fired: List[Dict[str, Any]], limit: int = MAX_CENTRAL) -> List[Dict[str, Any]]:
    """The fired rules' institutions as the frame's central issues ({nome, codice, fonte, ranges}), at most
    limit, each once; fonte and ranges travel to the analysis' institution (keep_resolved)."""
    out, seen = [], set()
    for rule in fired:
        for name in rule["institutions"]:
            if name.lower() not in seen and len(out) < limit:
                seen.add(name.lower())
                out.append({"nome": name, "codice": rule["codice"], "fonte": f"regola {rule['id']}",
                            "ranges": rule.get("ranges") or []})
    return out


def signals_from(facts: Optional[Dict[str, Any]], problem: str = "",
                 areas: Optional[List[str]] = None, rules: Optional[bool] = None) -> Dict[str, Any]:
    """{"facts", "terms", "fired", "problem", ...} from the extracted facts, the rules fired within the case's
    areas (only with FACT_SIGNALS_ENABLED, or rules); FACT_REGIMES_ENABLED first qualifies the parties
    (qualify_parties) and adds "qualifications", "regimes" and "tools". Logged as case_analysis_fact_signals."""
    regimes = regimes_enabled()
    qualifications: List[Dict[str, Any]] = []
    if regimes and facts:
        facts, qualifications = qualify_parties(facts)
    terms = compute_terms(facts, areas)
    fired: List[Dict[str, Any]] = []
    if enabled() if rules is None else rules:
        try:
            fired = fire_rules(facts, terms, areas=areas)
        except Exception as exc:
            logger.warning("fact signals: rules not applied: %s", exc)
            problem = problem or f"rules: {exc}"
    out = {"facts": facts, "terms": terms, "fired": fired, "problem": problem, "areas": list(areas or [])}
    if regimes:
        out.update({"qualifications": qualifications, "regimes": regime_issues(facts, qualifications, areas),
                    "tools": procedural_tools(facts, areas), "damages": damage_heads(facts),
                    "multi_roles": multiple_roles(facts)})
    vlog("case_analysis_fact_signals", {**out, "central": all_central(out)})
    return out


def all_central(signals: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The central issues the signals add to the frame: the regimes of the parties' qualifications first, then
    the fired rules' institutions (central_issues), each once, at most MAX_CENTRAL."""
    signals = signals or {}
    out, seen = [], set()
    for item in list(signals.get("regimes") or []) + central_issues(signals.get("fired") or [], limit=MAX_CENTRAL):
        if item["nome"].lower() not in seen and len(out) < MAX_CENTRAL:
            seen.add(item["nome"].lower())
            out.append({k: item[k] for k in ("nome", "codice", "fonte", "ranges")})
    return out


def analyse(case_text: str, areas: Optional[List[str]] = None) -> Dict[str, Any]:
    """extract_facts then signals_from with the rules, in one go (the benches)."""
    facts, problem = extract_facts(case_text)
    return signals_from(facts, problem, areas, rules=True)


def terms_block(signals: Optional[Dict[str, Any]], refs: bool = False) -> str:
    """The computed terms as facts for the plan and the procedural section ("" without any); with refs, each
    with the article it belongs to (FACT_REGIMES_ENABLED)."""
    terms = (signals or {}).get("terms") or []
    return "\n".join(f"- {t['text']}" + (f" [{t['ref']}]" if refs and t.get("ref") else "") for t in terms)


# --- 4. FACT_REGIMES_ENABLED: qualifications, regimes, procedural tools ------------------------------
# The lawyer's blind review (26 Oct): a natural person who bought for private use from a business was never
# treated as a consumer, so the Codice del Consumo's guarantee and the consumer's forum were missed; the legal
# terms were left out when the case gave no dates; the procedural tools for defects and urgency (ATP, art. 700
# c.p.c.) were gone. General inferences from the typed facts, never from a case's text.

_BUSINESS = (r"impres|societ|\bs\.?\s?r\.?\s?l\b|\bs\.?\s?p\.?\s?a\b|\bs\.?\s?n\.?\s?c\b|\bs\.?\s?a\.?\s?s\b|ditta|negozi|"
             r"concessionari|rivendit|commerciant|professionist|aziend|cantier|costrutt|produttor|fornitor|artigian|"
             r"albergator|struttura sanitaria|esercizio commerciale|venditore professionale|\bspa\b|\bsrl\b")
_PUBLIC_BODY = (r"\bcomune\b|\bregione\b|\bprovincia\b|ministero|prefettur|questur|\basl\b|\bats\b|\basp\b|"
                r"azienda sanitaria|azienda ospedalier|agenzia delle entrate|\binps\b|\binail\b|ente pubblico|"
                r"pubblica amministrazione|amministrazione (?:comunale|regionale|provinciale|statale|pubblica)|"
                r"scuola (?:statale|pubblica)|istituto comprensivo|universit[àa] statale|camera di commercio")
_PUBLIC_FUNCTION = (r"agente di polizia|poliziott|carabinier|vigil[ei] urban|polizia (?:locale|municipale|di stato)|"
                    r"guardia di finanza|finanzier|funzionari\w* (?:pubblic|comunal|regional|ministerial)|"
                    r"dipendente (?:pubblic|comunal|regional|ministerial)|dirigente (?:pubblic|comunal|scolastic)|"
                    r"\bsindaco|assessor|\bnotai|\bgiudic|magistrat|cancellier|ufficiale giudiziari|"
                    r"ufficiale di stato civile|insegnant\w* (?:di scuola )?(?:statal|pubblic)|"
                    r"docent\w* (?:di scuola )?(?:statal|pubblic)|medico (?:di|del) (?:un )?(?:ospedale pubblico|"
                    r"servizio sanitario|pronto soccorso pubblico)|ispettor\w* (?:del lavoro|asl|comunal)")
_PRIVATE_USE = r"personal|privat|famigli|domestic"
_CONSUMER_CONTRACT = (r"vendit|compravendit|acquist|fornitur|servizi|appalt|riparaz|manutenz|abbonament|viagg|"
                      r"noleggi|assicuraz|finanziament|credit|prestito|mutuo|conto|utenz|trasport")
_SALE = r"vendit|compravendit|acquist|fornitur"
CONSUMER, PROFESSIONAL = "consumatore", "professionista"
PUBLIC_BODY = "pubblica amministrazione"
PUBLIC_OFFICIAL = "pubblico ufficiale o incaricato di pubblico servizio"


def _party_text(p: Dict[str, Any]) -> str:
    return " ".join(_low(p.get(k)) for k in ("name", "role", "kind", "occupation"))


def _natural(p: Dict[str, Any]) -> bool:
    kind = _low(p.get("kind"))
    if kind:
        return "fisic" in kind
    return not re.search(_BUSINESS + "|" + _PUBLIC_BODY, _party_text(p))


def _business(p: Dict[str, Any]) -> bool:
    kind = _low(p.get("kind"))
    return (bool(re.search(r"societ|impres|ditta", kind)) or bool(re.search(_BUSINESS, _party_text(p)))
            or (bool(_low(p.get("occupation"))) and _low(p.get("purpose")).startswith("profession")))


def _with_role(p: Dict[str, Any], role: str) -> Dict[str, Any]:
    have = _low(p.get("role"))
    return p if role in have else {**p, "role": f"{p.get('role')}, {role}" if p.get("role") else role}


def qualify_parties(facts: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """(facts, qualifications): the facts with each inferred qualification added to the party's role (so the
    terms and the rules read it), marked "_regimes"; [{party, quality, why}]. A natural person who contracts for
    private use with a business is a consumer, and the business a professional; a public body is a public
    administration; a party with a public function a public official or a person in charge of a public service
    (which of the two depends on the function exercised)."""
    parties = [dict(p) for p in facts.get("parties") or []]
    contracts = [c for c in facts.get("contracts") or [] if re.search(_CONSUMER_CONTRACT, _low(c.get("type")))]
    out: List[Dict[str, str]] = []
    businesses = [i for i, p in enumerate(parties) if _business(p) and not re.search(_PUBLIC_BODY, _party_text(p))]
    for i, p in enumerate(parties):
        name = p.get("name") or "una parte"
        consumer_role = CONSUMER in _low(p.get("role"))
        private = re.search(_PRIVATE_USE, _low(p.get("purpose")))
        if contracts and businesses and i not in businesses and _natural(p) and (private or consumer_role):
            if not consumer_role:
                parties[i] = _with_role(p, CONSUMER)
            seller = parties[businesses[0]].get("name") or "un'impresa"
            out.append({"party": name, "quality": CONSUMER,
                        "why": f"persona fisica che contrae per uso personale o familiare con {seller}, che agisce "
                               f"nell'esercizio della sua attività ({_low(contracts[0].get('type'))})"})
            for j in businesses:
                if PROFESSIONAL not in _low(parties[j].get("role")):
                    parties[j] = _with_role(parties[j], PROFESSIONAL)
                    out.append({"party": parties[j].get("name") or "l'impresa", "quality": PROFESSIONAL,
                                "why": "impresa o professionista che contratta con un consumatore"})
        text = _party_text(parties[i])
        if re.search(_PUBLIC_BODY, text) and PUBLIC_BODY not in _low(parties[i].get("role")):
            parties[i] = _with_role(parties[i], PUBLIC_BODY)
            out.append({"party": name, "quality": PUBLIC_BODY, "why": "ente pubblico"})
        elif (re.search(_PUBLIC_FUNCTION, text) and "pubblico ufficiale" not in _low(parties[i].get("role"))
              and "incaricato di pubblico servizio" not in _low(parties[i].get("role"))):
            parties[i] = _with_role(parties[i], PUBLIC_OFFICIAL)
            out.append({"party": name, "quality": PUBLIC_OFFICIAL,
                        "why": f"esercita una funzione pubblica ({_low(p.get('occupation')) or _low(p.get('role'))}); "
                               "la qualifica dipende dalla funzione esercitata nel fatto"})
    return {**facts, "parties": parties, "_regimes": True}, out


# quality -> the regimes it brings in, as central issues (institutions in words, never an article; ranges as the
# rules' own: fact_signal_rules.yaml); "when": a condition of holds() the facts must also meet.
REGIMES: List[Dict[str, Any]] = [
    {"id": "consumatore_garanzia", "quality": CONSUMER, "institution": "garanzia legale di conformità",
     "codice": "Codice del Consumo", "ranges": [[128, 135]], "families": ["civile"],
     "when": [{"contract": {"type": _SALE}}]},
    {"id": "consumatore_foro", "quality": CONSUMER, "institution": "foro del consumatore",
     "codice": "Codice del Consumo", "ranges": [], "families": ["civile"], "when": []},
    {"id": "consumatore_clausole", "quality": CONSUMER,
     "institution": "clausole vessatorie nel contratto tra professionista e consumatore",
     "codice": "Codice del Consumo", "ranges": [[33, 38]], "families": ["civile"],
     "when": [{"contract": {"clause": r"\w"}}]},
    {"id": "nozione_pubblico_ufficiale", "quality": PUBLIC_OFFICIAL,
     "institution": "nozione di pubblico ufficiale e di incaricato di pubblico servizio", "codice": "c.p.",
     "ranges": [[357, 360]], "families": ["penale"], "when": []},
]


def regime_issues(facts: Optional[Dict[str, Any]], qualifications: List[Dict[str, str]],
                  areas: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """The REGIMES of the qualifications found, within the families of the case's areas, as central issues
    ({nome, codice, fonte, ranges}, as central_issues)."""
    qualities = {q["quality"] for q in qualifications} | {CONSUMER for p in (facts or {}).get("parties") or []
                                                          if CONSUMER in _low(p.get("role"))}
    families = {family(a) for a in areas or []} - {""}
    out = []
    for regime in REGIMES:
        if regime["quality"] not in qualities or (families and not set(regime["families"]) & families):
            continue
        if not all(holds(c, facts or {}, []) for c in regime["when"]):
            continue
        out.append({"nome": regime["institution"], "codice": regime["codice"], "fonte": f"qualifica {regime['id']}",
                    "ranges": regime["ranges"]})
    return out


_DEFECTS = _DEFECT + r"|non conform|danneggi|rott|allag|incendi|crollo"
_LOSABLE = r"\bprov|dispers|perd|ripar|ripristin|smalt|demol|modific|alter|cancell|deterior|sostitu"
_URGENT = (r"pericol|sicurezz|incolumit|rischio di (?:crollo|caduta|incendio|allagamento)|crollo|urgen|imminent|"
           r"irreparabil|aggrav|peggior|sta per")

# The procedural tools the facts call for, with the articles they rest on (a code table, like
# practical_checklist.py): offered to the plan and to the procedural section with their texts.
PROCEDURAL_TOOLS: List[Dict[str, Any]] = [
    {"id": "accertamento_preventivo", "refs": [("696", "c.p.c."), ("696-bis", "c.p.c.")],
     "text": "Accertamento tecnico preventivo (art. 696 c.p.c.), per fissare lo stato dei luoghi, della cosa o dei "
             "danni prima che cambi, o consulenza tecnica preventiva ai fini della composizione della lite (art. "
             "696-bis c.p.c.), che accerta anche le cause e l'entità dei danni e tenta la conciliazione.",
     "why": "difetti, danni a cose o una prova che può andare perduta"},
    {"id": "provvedimento_urgenza", "refs": [("700", "c.p.c.")],
     "text": "Provvedimento d'urgenza (art. 700 c.p.c.), se nel tempo necessario per la causa il diritto è "
             "minacciato da un pregiudizio imminente e irreparabile e nessun altro provvedimento cautelare lo tutela.",
     "why": "un pericolo per la sicurezza, un danno che si aggrava o un'urgenza"},
    # The criminal route of a personal harm caused by negligence, also when the frame's area is only civil.
    {"id": "profilo_penale", "refs": [("589", "c.p."), ("590", "c.p."), ("76", "c.p.p."), ("2947", "c.c.")],
     "text": "Profilo penale: la morte o le lesioni di una persona causate per colpa possono costituire reato "
             "(omicidio colposo, art. 589 c.p., procedibile d'ufficio; lesioni personali colpose, art. 590 c.p., "
             "di regola procedibili a querela): valutare la denuncia o la querela e i suoi termini, la costituzione "
             "di parte civile nel processo penale (art. 76 c.p.p.), le prove che il procedimento penale acquisisce "
             "(accertamenti, consulenze, testimonianze) e la prescrizione più lunga dell'azione civile quando il "
             "fatto è considerato dalla legge come reato (art. 2947, terzo comma, c.c.).",
     "why": "la morte o lesioni di una persona causate per colpa"},
]


def procedural_tools(facts: Optional[Dict[str, Any]], areas: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """The PROCEDURAL_TOOLS the facts call for, in civil cases (none when the case's areas are all criminal or
    administrative): {id, refs, text, why, matched}. The accertamento preventivo for defects, damage to things
    or evidence that may be lost; the provvedimento d'urgenza for a safety risk, a harm that grows or urgency."""
    if not facts:
        return []
    families = {family(a) for a in areas or []} - {""}
    if families and "civile" not in families:
        return []
    risks = facts.get("risks") or []
    texts = _texts(facts, ("events", "works_or_signs", "contracts", "injuries", "risks", "proceedings"))

    def line(pattern):
        return next((ln[:140] for ln in texts.split("\n") if re.search(pattern, ln)), "")
    found = {}
    defect = line(_DEFECTS)
    evidence = next((_low(r.get("what"))[:140] for r in risks if "prov" in _low(r.get("kind"))
                     or re.search(_LOSABLE, _low(r.get("what")))), "")
    if defect or evidence:
        found["accertamento_preventivo"] = [m for m in (defect, evidence) if m]
    urgent = next((_low(r.get("what"))[:140] for r in risks if re.search(r"sicurezz|aggrav|urgen", _low(r.get("kind")))), "")
    urgent = urgent or line(_URGENT)
    if urgent:
        found["provvedimento_urgenza"] = [urgent]
    harm = _personal_harm(facts)
    if harm and not re.search(_INTENT, texts):
        found["profilo_penale"] = [harm]
    out = []
    for t in PROCEDURAL_TOOLS:
        if t["id"] not in found:
            continue
        refs = [list(r) for r in t["refs"]]
        if t["id"] == "profilo_penale":         # the offence the harm fits: death, injury, or both
            death = _death(facts)
            drop = ([["590", "c.p."]] if death and not _injured(facts) else []) + ([] if death else [["589", "c.p."]])
            refs = [r for r in refs if r not in drop]
        out.append({"id": t["id"], "refs": refs, "text": t["text"], "why": t["why"], "matched": found[t["id"]]})
    return out


_DEATH_RE = r"\bmort[eoi]\b|\bdeced|\bdecess|perso la vita|\bannegat|\bucciso|privo di vita|\bsenza vita"
_INTENT = r"\bvolontariament|intenzional|\bdi proposito|\baggred|\bpicchi|\bpercoss|\bcon dolo\b|\ballo scopo di"


def _death(facts: Dict[str, Any]) -> bool:
    return any("mort" in _low(i.get("permanent")) for i in facts.get("injuries") or []) or bool(
        re.search(_DEATH_RE, _texts(facts, ("events", "injuries"))))


def _injured(facts: Dict[str, Any]) -> bool:
    return any("mort" not in _low(i.get("permanent")) and (i.get("who") or i.get("days_to_heal"))
               for i in facts.get("injuries") or [])


def _personal_harm(facts: Dict[str, Any]) -> str:
    """The fact of a person's death or injury, or "" (the injuries the extraction found, else a death in the events)."""
    for i in facts.get("injuries") or []:
        if i.get("who") or i.get("days_to_heal") or i.get("permanent"):
            return "; ".join(str(v) for v in (i.get("who"), i.get("permanent") or i.get("days_to_heal")) if v)[:140]
    return next((ln[:140] for ln in _texts(facts, ("events",)).split("\n") if re.search(_DEATH_RE, ln)), "")


# The heads of damage for the death of a person, a general rule for the plan and the sections (the lawyer's review):
# what the relatives claim in their own right, what they claim as heirs, and who pays.
DAMAGES_DEATH = {
    "id": "danni_morte", "refs": [("2059", "c.c."), ("1917", "c.c.")],
    "lines": [
        "Iure proprio, per ciascun familiare: il danno non patrimoniale da perdita del rapporto parentale o familiare "
        "(art. 2059 c.c., letto alla luce dei diritti inviolabili della Costituzione), da provare anche per "
        "presunzioni dal legame e dalla convivenza; il danno patrimoniale: spese funerarie, mediche e di assistenza "
        "psicologica, perdita del contributo economico della vittima.",
        "Iure hereditatis, per gli eredi: il danno terminale o catastrofale se la vittima è rimasta in vita e cosciente "
        "per un tempo apprezzabile tra la lesione e la morte; altrimenti nessun danno da morte si trasmette agli eredi.",
        "Chi paga: le assicurazioni della responsabilità civile delle parti responsabili (art. 1917 c.c.), da "
        "individuare e chiamare in causa, con l'azione diretta solo dove la legge la prevede.",
    ]}


def damage_heads(facts: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """[DAMAGES_DEATH] when the facts tell of a person's death, else []."""
    if not facts or not _death(facts):
        return []
    return [{"id": DAMAGES_DEATH["id"], "refs": [list(r) for r in DAMAGES_DEATH["refs"]],
             "lines": list(DAMAGES_DEATH["lines"]), "matched": [_personal_harm(facts)]}]


def multiple_roles(facts: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The parties with two or more activities in the facts (built and maintains, sold and installed): each role is a
    ground of liability of its own. [{party, activities}]."""
    out = []
    for p in (facts or {}).get("parties") or []:
        acts = [a.strip() for a in re.split(r"[,;]|\be\b", str(p.get("activities") or "")) if len(a.strip()) > 2]
        acts = list(dict.fromkeys(acts))
        if len(acts) >= 2:
            out.append({"party": p.get("name") or "una parte", "activities": acts})
    return out


# Who claims: a client who seeks damages or brings an action is a claimant; the "respond to a claim received"
# scaffolding only when the case says a claim or an act was received (the lawyer's review).
_CLAIMANT_RE = re.compile(
    r"\b(?:ottenere|chiedere|richiedere|domandare|esigere|far valere|agire|citare in giudizio|fare causa|"
    r"denunciare|querelare)\b[^.?\n]{0,60}?(?:risarciment|indennizz|ristor|rimbors|restituzion|responsabil|giudizio|"
    r"causa|danni|danno)|\b(?:vogliono|vuole|intendono|intende|possono|può)\s+(?:\w+\s+){0,2}(?:agire|citare|"
    r"denunciare|querelare|chiedere il risarcimento|ottenere il risarcimento|fare causa)|\brisarcimento\s+(?:dei|del)\s+"
    r"dann\w*\s+(?:subit|patit)", re.IGNORECASE)
_RECEIVED_RE = re.compile(
    r"\b(?:ricevut\w*|ricev\w+|pervenut\w*|recapitat\w*|notificat\w*|inviat\w*\s+(?:a|al|alla|ai))\b[^.\n]{0,80}?"
    r"(?:lettera|diffida|messa in mora|richiesta|pretesa|atto di citazione|citazione|ricorso|decreto ingiuntivo|"
    r"precetto|reclamo|contestazione|avviso|verbale|sanzione|pec|raccomandata)|\b(?:lettera|diffida|messa in mora|"
    r"richiesta di risarcimento|atto di citazione|citazione|ricorso|decreto ingiuntivo|precetto|contestazione|avviso|"
    r"raccomandata|pec)\b[^.\n]{0,60}?\b(?:ricevut\w*|pervenut\w*|notificat\w*|recapitat\w*)", re.IGNORECASE)


def claim_posture(case_text: str, facts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """{"claimant", "received", "why"}: the client seeks damages or an action (the case's text or the extracted
    requests), and the case says a claim or an act was received."""
    text = " ".join([case_text or ""] + list((facts or {}).get("requests") or []))
    claim, received = _CLAIMANT_RE.search(text), _RECEIVED_RE.search(case_text or "")
    return {"claimant": bool(claim), "received": bool(received),
            "why": "; ".join(m.group(0)[:100] for m in (claim, received) if m)}


def damages_block(signals: Optional[Dict[str, Any]]) -> str:
    return "\n".join(f"- {line}" for d in (signals or {}).get("damages") or [] for line in d["lines"])


def roles_block(signals: Optional[Dict[str, Any]]) -> str:
    return "\n".join(f"- {r['party']}: {'; '.join(r['activities'])} — la responsabilità va valutata per ciascun ruolo"
                     for r in (signals or {}).get("multi_roles") or [])


def regime_refs(signals: Optional[Dict[str, Any]]) -> Dict[Tuple[str, str], str]:
    """{article: what calls for it} of the procedural tools and the heads of damage, for their texts."""
    out: Dict[Tuple[str, str], str] = {}
    for item in list((signals or {}).get("tools") or []) + list((signals or {}).get("damages") or []):
        for r in item["refs"]:
            out.setdefault(tuple(r), item["id"])
    return out


def qualifications_block(signals: Optional[Dict[str, Any]]) -> str:
    """The parties' inferred qualifications, one line each ("" without any)."""
    return "\n".join(f"- {q['party']}: {q['quality']} ({q['why']})" for q in (signals or {}).get("qualifications") or [])


def tools_block(signals: Optional[Dict[str, Any]]) -> str:
    """The procedural tools the facts call for, one line each, with the facts that call for them."""
    return "\n".join(f"- {t['text']} Nel caso: {'; '.join(t['matched'])}." for t in (signals or {}).get("tools") or [])


def tool_refs(signals: Optional[Dict[str, Any]]) -> List[Tuple[str, str]]:
    return list(regime_refs(signals))


# Terms whose article is found in the corpus by its words (the Codice del Consumo's numbering changed in 2021):
# term id -> (code, phrases its article's text holds, tried in order). Not found: the term keeps the code alone.
_LOCATE = {
    "denuncia_conformita_consumo": ("Codice del Consumo", ["due mesi dalla data in cui ha scoperto",
                                                           "due mesi dalla scoperta"]),
    "responsabilita_conformita_consumo": ("Codice del Consumo", ["si manifesta entro due anni dalla consegna",
                                                                 "entro il termine di due anni dalla consegna",
                                                                 "entro due anni dalla consegna"]),
    "azione_conformita_consumo": ("Codice del Consumo", ["ventisei mesi dalla consegna", "ventisei mesi"]),
}
_REF_RE = re.compile(r"\bartt?\.\s*(\d+(?:-[a-z]+)?(?:\s*(?:,|e)\s*\d+(?:-[a-z]+)?)*)\s+(c\.c\.|c\.p\.c\.|c\.p\.p\.|"
                     r"c\.p\.a\.|c\.p\.)")


def attach_term_sources(session, documents: List[Dict[str, Any]], signals: Optional[Dict[str, Any]],
                        chars: int = 700) -> None:
    """In place: each term's article read from the corpus -- a _LOCATE term's found by the words of its text
    (its "ref" then names it), any other's by its ref -- with the first chars of its text as "sources"
    [{ref, text}], for the procedural section to quote. A failure leaves the terms as they are."""
    from . import defensive_generation as dg
    from .issue_spotting import resolve_code
    for term in (signals or {}).get("terms") or []:
        try:
            if term["id"] in _LOCATE:
                code, phrases = _LOCATE[term["id"]]
                doc = resolve_code(code, documents)
                row = _section_with(session, doc["id"], phrases) if doc else None
                if row:
                    number = str((row.get("s") or {}).get("name") or "").split(".")[0]
                    term["ref"] = f"art. {number} {doc['name']}"
                    term["sources"] = [{"ref": term["ref"], "text": _section_text(row)[:chars]}]
                continue
            refs = [(dg._article_num(n), code) for m in _REF_RE.finditer(term.get("ref") or "")
                    for n in re.split(r"\s*(?:,|\be\b)\s*", m.group(1)) for code in [m.group(2)] if n]
            rows = dg._fetch_any(session, documents, refs) if refs else []
            term["sources"] = [{"ref": f"art. {r['_article'][0]} {r['_article'][1]}", "text": _section_text(r)[:chars]}
                               for r in rows if r.get("_article") and _section_text(r)]
        except Exception as exc:
            logger.warning("fact regimes: no text for the term %s: %s", term.get("id"), exc)


def _section_text(row: Dict[str, Any]) -> str:
    s = row.get("s") or {}
    return " ".join(str(s.get("plain_text") or s.get("abstract") or "").split())


def _section_with(session, doc_id: str, phrases: List[str]) -> Optional[Dict[str, Any]]:
    """The first section of the document whose text holds one of phrases (in their order), lowest number first."""
    for phrase in phrases:
        rows = session.run(
            "MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE d.id = $doc_id "
            "AND toLower(coalesce(s.plain_text, s.abstract, '')) CONTAINS $phrase RETURN d, s LIMIT 5",
            doc_id=doc_id, phrase=phrase.lower()).data()
        if rows:
            return min(rows, key=lambda r: _number(str((r.get("s") or {}).get("name") or "")) or 9999)
    return None


def as_json(signals: Dict[str, Any]) -> str:
    return json.dumps(signals, ensure_ascii=False, default=str)
