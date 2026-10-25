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
RULES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fact_signal_rules.yaml")
TODAY_ENV = "FACT_SIGNALS_TODAY"        # YYYY-MM-DD: the date terms run to when no action is dated (tests, benches)
MAX_CENTRAL = 6                         # institutions the fired rules add as central issues, in the rules' order
_CASE_CHARS = 12000
_EXTRACT_TOKENS = 1800


def enabled() -> bool:
    return feature_flags.enabled(FACT_SIGNALS_FLAG)


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
}

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
    "Non inventare date: una data compare solo se è scritta nel testo, e come vi è scritta. Rispondi SOLO con il JSON."
)
_RETRY_SUFFIX = ("\n\nLa risposta precedente non era un JSON valido con quelle chiavi. Rispondi SOLO con l'oggetto "
                 "JSON, dal primo carattere.")


def _clean_text(value: Any) -> str:
    if isinstance(value, bool):
        return "sì" if value else "no"
    if isinstance(value, (int, float)):
        return str(value)
    return " ".join(str(value or "").split()) if isinstance(value, str) else ""


def validate_facts(raw: Any, case_text: str) -> Optional[Dict[str, Any]]:
    """The facts in SCHEMA's shape (unknown keys dropped, every value a string or a list of string
    dicts), or None when raw is not a JSON object with at least one of SCHEMA's keys. A date the case
    text does not contain is emptied (never invented); each dated item gains date_iso, from code."""
    if not isinstance(raw, dict) or not set(raw) & set(SCHEMA):
        return None
    norm_case = " ".join((case_text or "").lower().split())
    out: Dict[str, Any] = {}
    for key, shape in SCHEMA.items():
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
    system, problem = _EXTRACT_SYSTEM, ""
    for attempt in (1, 2):
        try:
            raw = _call_chat([SystemMessage(content=system), HumanMessage(content=human)],
                             max_tokens=_EXTRACT_TOKENS, temperature=0)
            facts = validate_facts(parse_json_reply(raw), case_text)
        except Exception as exc:
            facts, problem = None, f"{type(exc).__name__}: {str(exc)[:160]}"
        if facts is not None:
            return facts, ""
        problem = problem or "reply is not a JSON object with the schema's keys"
        system = _EXTRACT_SYSTEM + _RETRY_SUFFIX
    return None, problem


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


def today() -> dt.date:
    value = (os.getenv(TODAY_ENV) or "").strip()
    try:
        return dt.date.fromisoformat(value) if value else dt.date.today()
    except ValueError:
        return dt.date.today()


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
                 start_items: Optional[Callable[[Dict[str, Any]], List[tuple]]] = None):
        self.id, self.label, self.ref, self.start, self.end = id, label, ref, start, end
        self.years, self.months, self.days, self.applies, self.word = years, months, days, applies, word
        self.start_items = start_items

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
         years=1, word="termine annuale"),
    Term("usucapione_ordinaria", "usucapione ordinaria di beni immobili", "art. 1158 c.c.", "", "", years=20,
         word="ventennio di possesso", start_items=lambda f: [s for s in _possession_starts(f) if not re.search(_MOVABLE, _low(s[1]))]),
    Term("usucapione_abbreviata", "usucapione abbreviata (solo con titolo idoneo trascritto e buona fede)",
         "art. 1159 c.c.", "", "", years=10, word="decennio di possesso",
         applies=lambda f: any(re.search(r"trascri|rogito|atto pubblico|compravendit|donazion", _low(p.get("how_acquired")))
                               for p in f.get("property_history") or []),
         start_items=lambda f: [s for s in _possession_starts(f) if not re.search(_MOVABLE, _low(s[1]))]),
    Term("usucapione_mobili", "usucapione di beni mobili (10 anni in buona fede; 20 altrimenti)", "art. 1161 c.c.",
         "", "", years=10, word="decennio di possesso",
         start_items=lambda f: [s for s in _possession_starts(f) if re.search(_MOVABLE, _low(s[1]))]),
    Term("prescrizione_circolazione", "prescrizione del risarcimento da circolazione di veicoli", "art. 2947 c.c.",
         _ROAD, _INTERRUPT, years=2, applies=lambda f: not f.get("contracts")),
    Term("prescrizione_extracontrattuale", "prescrizione del risarcimento del danno extracontrattuale",
         "art. 2947 c.c.", _DAMAGE, _INTERRUPT, years=5, applies=lambda f: not f.get("contracts")),
    Term("prescrizione_ordinaria", "prescrizione ordinaria (diritti da contratto)", "art. 2946 c.c.",
         r"inadempi|non (?:ha )?pagat|scadenz|ritard|mancat\w* (?:pagament|consegn)", _INTERRUPT, years=10,
         applies=lambda f: bool(f.get("contracts"))),
    Term("querela", "querela", "art. 124 c.p.", _OFFENCE, r"querel|denunci", months=3),
    Term("ricorso_tar_appalti", "ricorso al TAR in materia di appalti pubblici (rito speciale)", "art. 120 c.p.a.",
         _ACT, _TAR, days=30,
         applies=lambda f: any(re.search(_TENDER, _low(" ".join(map(str, e.values()))))
                               for e in (f.get("events") or []) + (f.get("proceedings") or []))),
    Term("ricorso_tar", "ricorso al TAR per l'annullamento", "artt. 29 e 41 c.p.a.", _ACT, _TAR, days=60,
         applies=lambda f: not any(re.search(_TENDER, _low(" ".join(map(str, e.values()))))
                                   for e in (f.get("events") or []) + (f.get("proceedings") or []))
         and _has_role(f, r"pubblica amministrazione|comune|ente")),
    Term("denuncia_vizi_vendita", "denuncia dei vizi della cosa venduta", "art. 1495 c.c.", _DISCOVERY + "|" + _DEFECT,
         _DENOUNCE, days=8, applies=lambda f: _has_contract(f, r"vendit|compravendit|acquist")
         and not _has_role(f, "consumatore")),
    Term("azione_vizi_vendita", "azione di garanzia per i vizi della cosa venduta", "art. 1495 c.c.", _DELIVERY,
         _ACTION, years=1, applies=lambda f: _has_contract(f, r"vendit|compravendit|acquist")
         and not _has_role(f, "consumatore")),
    Term("denuncia_conformita_consumo", "denuncia del difetto di conformità (consumatore)", "Codice del Consumo",
         _DISCOVERY + "|" + _DEFECT, _DENOUNCE, months=2,
         applies=lambda f: _has_role(f, "consumatore") and _has_contract(f, r"vendit|compravendit|acquist")),
    Term("denuncia_difformita_appalto", "denuncia delle difformità e dei vizi dell'opera (appalto)", "art. 1667 c.c.",
         _DISCOVERY + "|" + _DEFECT, _DENOUNCE, days=60,
         applies=lambda f: _has_contract(f, r"appalt|lavori|ristruttur|costruz")),
    Term("azione_garanzia_appalto", "azione di garanzia nell'appalto", "art. 1667 c.c.", _WORKS_DONE, _ACTION, years=2,
         applies=lambda f: _has_contract(f, r"appalt|lavori|ristruttur|costruz")),
    Term("manifestazione_gravi_difetti", "manifestazione dei gravi difetti entro dieci anni dal compimento dell'opera",
         "art. 1669 c.c.", _WORKS_DONE, _DISCOVERY + "|" + _DEFECT, years=10, word="decennio dal compimento",
         applies=lambda f: _has_contract(f, r"appalt|lavori|ristruttur|costruz")),
    Term("denuncia_gravi_difetti", "denuncia dei gravi difetti dell'immobile", "art. 1669 c.c.",
         _DISCOVERY + "|" + _DEFECT, _DENOUNCE, years=1,
         applies=lambda f: _has_contract(f, r"appalt|lavori|ristruttur|costruz")),
]

# Criminal appeals run from the deposit of the reasons (art. 585 c.p.p.): 15 days with the reasons read in
# court, 30 when deposited within 15 days, 45 within a longer term (at most 90 days).
_APPEAL_REF = "art. 585 c.p.p."


def _when(items: List[tuple], after: dt.date) -> Optional[tuple]:
    return next((i for i in items if i[0][1] >= after), None)


def _short(written: str) -> str:
    return written or "data non indicata"


def compute_terms(facts: Optional[Dict[str, Any]], on: Optional[dt.date] = None) -> List[Dict[str, Any]]:
    """Every term of TERMS the facts date: its status between the fact it runs from and the act that
    ends it, or today when none does (TODAY_ENV in tests); one line of text each."""
    if not facts:
        return []
    now = on or today()
    out = []
    for term in TERMS:
        if term.applies and not term.applies(facts):
            continue
        starts = term.start_items(facts) if term.start_items else _events(facts, term.start)
        if not starts:
            continue
        start = starts[0]
        ends = _events(facts, term.end) if term.end else []
        end = _when([e for e in ends if e[1] != start[1] or e[0] != start[0]], start[0][0])
        end_iv, end_what, end_written = (end[0], end[1], end[2]) if end else ((now, now), "oggi", now.isoformat())
        status = term_status(start[0], end_iv, term.years, term.months, term.days)
        out.append(_term_entry(term.id, term.label, term.ref, term.word or f"termine di {term.length()}",
                               term.length(), status, start, end_what, end_written))
    sentence = _events(facts, _SENTENCE)
    if sentence:
        start = sentence[-1]
        deposit = _events(facts, r"deposit\w* (?:delle |della )?motivazion")
        appeal = _when(_events(facts, _APPEAL), start[0][0])
        end_iv, end_what, end_written = (appeal[0], appeal[1], appeal[2]) if appeal else ((now, now), "oggi", now.isoformat())
        if deposit:
            base, sure, latest = deposit[-1], 30, 45
        else:
            base, sure, latest = start, 15, 90 + 45
        status = ("DECORSO" if end_iv[0] > add_period(base[0][1], days=latest) else
                  "NON DECORSO" if end_iv[1] <= add_period(base[0][0], days=sure) else "INCERTO")
        length = "15, 30 o 45 giorni dal deposito della motivazione"
        out.append(_term_entry("impugnazione_penale", "impugnazione della sentenza penale", _APPEAL_REF,
                               f"termine di {length}", length, status, base, end_what, end_written,
                               note="" if deposit else "dipende da quando è depositata la motivazione"))
    return out


def _term_entry(id, label, ref, word, length, status, start, end_what, end_written, note=""):
    text = (f"{label}: {word} {status} ({start[1] or 'fatto'} {_short(start[2])}, {end_what} "
            f"{_short(end_written)})" + (f"; {note}" if note and status == "INCERTO" else ""))
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
    return bool(re.search(arg["pattern"], _texts(facts, ("proceedings",))))


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


def _years_held(since: str, now: dt.date) -> Optional[Tuple[dt.date, dt.date]]:
    parsed = parse_date(since)
    return (parsed[0], parsed[1]) if parsed else None


def _p_possession(facts, terms, arg):
    now = today()
    for p in facts.get("possession") or []:
        how = _low(p.get("how"))
        if arg.get("how") and not any(h in how for h in arg["how"]):
            continue
        if arg.get("how_not") and any(h in how for h in arg["how_not"]):
            continue
        if arg.get("min_years"):
            since = _years_held(p.get("since", ""), now)
            if not since or term_status(since, (now, now), years=arg["min_years"]) != "DECORSO":
                continue
        return True
    return False


def _p_work(facts, terms, arg):
    now = today()
    for w in facts.get("works_or_signs") or []:
        if arg.get("visible") and not _yes(w.get("visible")):
            continue
        if arg.get("pattern") and not re.search(arg["pattern"], _low(w.get("what"))):
            continue
        if arg.get("min_years"):
            since = _years_held(w.get("existing_since", ""), now)
            if not since or term_status(since, (now, now), years=arg["min_years"]) != "DECORSO":
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


def fire_rules(facts: Optional[Dict[str, Any]], terms: List[Dict[str, Any]],
               rules: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """The rules whose conditions all hold, in the rules' order: {id, area, institutions, codice, why}."""
    if not facts:
        return []
    fired = []
    for rule in rules if rules is not None else load_rules():
        try:
            ok = all(holds(c, facts, terms) for c in rule["when"])
        except Exception as exc:
            logger.warning("fact signals: rule %s failed: %s", rule.get("id"), exc)
            ok = False
        if ok:
            fired.append({"id": rule["id"], "area": rule["area"], "institutions": _institutions(rule),
                          "codice": rule["codice"], "why": rule["why"], "status": rule["status"]})
    return fired


def central_issues(fired: List[Dict[str, Any]], limit: int = MAX_CENTRAL) -> List[Dict[str, str]]:
    """The fired rules' institutions as the frame's central issues ({nome, codice, fonte}), at most limit,
    each once."""
    out, seen = [], set()
    for rule in fired:
        for name in rule["institutions"]:
            if name.lower() not in seen and len(out) < limit:
                seen.add(name.lower())
                out.append({"nome": name, "codice": rule["codice"], "fonte": f"regola {rule['id']}"})
    return out


def analyse(case_text: str) -> Dict[str, Any]:
    """{"facts", "terms", "fired", "problem"} for a case; logged as case_analysis_fact_signals."""
    facts, problem = extract_facts(case_text)
    terms = compute_terms(facts)
    try:
        fired = fire_rules(facts, terms)
    except Exception as exc:
        logger.warning("fact signals: rules not applied: %s", exc)
        fired, problem = [], problem or f"rules: {exc}"
    out = {"facts": facts, "terms": terms, "fired": fired, "problem": problem}
    vlog("case_analysis_fact_signals", {**out, "central": central_issues(fired)})
    return out


def terms_block(signals: Optional[Dict[str, Any]]) -> str:
    """The computed terms as facts for the plan and the procedural section ("" without any)."""
    terms = (signals or {}).get("terms") or []
    return "\n".join(f"- {t['text']}" for t in terms)


def as_json(signals: Dict[str, Any]) -> str:
    return json.dumps(signals, ensure_ascii=False, default=str)
