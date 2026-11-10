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

Typed fields (27 Oct, a prod dry run that fired nothing on a dated narrative): roles only from ROLES, a qualified
one only when the text states it of that party, else "privato"; owners with their period, the assets they held
together and how their ownership ended; works with whether they existed before the separation; events with a kind
from EVENT_KINDS, which the terms read before any words; possession with its end and use. scripts/fact_probe.py
prints one extraction whole, with every term and rule explained (explain_terms, explain_rules).

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
import copy
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from . import feature_flags
from .ai_chat import _call_chat, last_reply, parse_json_reply
from . import article_text
from .llm_reply import salvage_json
from .verbose_logger import exception_data, vlog


def _log_exception(stage: str, exc: BaseException, **detail: Any) -> None:
    """Every exception caught here, logged with its full traceback (pipeline_exception), whatever VERBOSE_LOGGING says."""
    vlog("pipeline_exception", exception_data(f"fact_signals.{stage}", exc, **detail))


logger = logging.getLogger(__name__)

FACT_SIGNALS_FLAG = "FACT_SIGNALS_ENABLED"
# FACT_REGIMES_ENABLED (the lawyer's blind review of two new cases, 26 Oct): the parties' qualifications inferred
# from the facts (a natural person buying for private use from a business is a consumer; a public body; a public
# official), the special regimes they bring in as central issues, every applicable legal term of TERMS stated
# even when the case gives no dates, and the procedural tools the facts call for (PROCEDURAL_TOOLS). Runs the
# extraction also with FACT_SIGNALS_ENABLED off; the rules of fact_signal_rules.yaml fire only with it on.
REGIMES_FLAG = "FACT_REGIMES_ENABLED"
# TERM_EXPIRY_TEXT (v45): the non-use term's year written as its end ("non decorso, scade nel 2036").
TERM_EXPIRY_TEXT_FLAG = "TERM_EXPIRY_TEXT"
RULES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fact_signal_rules.yaml")
# YYYY-MM-DD: the case's present for a case that states none (tests, benches). Never the real today: an exam
# case set years ago would have every term "DECORSO" (fact bench, run 1415).
TODAY_ENV = "FACT_SIGNALS_TODAY"
MAX_CENTRAL = 6                         # institutions the fired rules add as central issues, in the rules' order
_CASE_CHARS = 12000
_EXTRACT_TOKENS = 5000                  # run 1415's longest cases (n2, n9) came back cut at 1800


def enabled() -> bool:
    return feature_flags.enabled(FACT_SIGNALS_FLAG)


def regimes_enabled() -> bool:
    return feature_flags.enabled(REGIMES_FLAG)


# --- 1. typed facts ------------------------------------------------------------------------------

ROLES = ("cliente", "controparte", "terzo", "privato", "proprietario", "possessore", "vicino", "venditore",
         "acquirente", "indagato/imputato", "persona offesa", "pubblico ufficiale", "incaricato di pubblico servizio",
         "medico", "struttura sanitaria", "professionista", "datore di lavoro", "lavoratore", "consumatore", "impresa",
         "pubblica amministrazione", "minore", "erede", "coniuge", "locatore", "conduttore", "albergatore",
         "committente", "appaltatore")
# 27 Oct (a prod dry run): the extraction gave a landowner the role "incaricato di pubblico servizio". A qualified
# role is kept only when the text states it: a cue in the party's own name or in the words around it
# (_clean_roles); a party left with no role, or only its side in the case, is "privato".
QUALIFIED_ROLE_CUES = {
    "pubblico ufficiale": r"pubblic\w* ufficial|agent[ei]|poliziott|polizia|carabinier|vigil[ei]|sindaco|assessor|"
                          r"funzionari|giudic|magistrat|notai|ufficiale|militar|guardia di finanza",
    "incaricato di pubblico servizio": r"pubblic\w* servizi|incaricat\w* di (?:un )?pubblico|dipendente (?:pubblic|"
                                       r"comunal|regional)|bidell|postin|controllor|autista (?:di |del )?(?:autobus|"
                                       r"bus|mezzi pubblici)|ausiliari",
    "medico": r"medic|dottor|chirurg|anestesist|ginecolog|pediatr|ortoped|cardiolog|sanitari",
    "struttura sanitaria": r"ospedal|clinic|struttura sanitaria|casa di cura|\basl\b|azienda sanitaria|"
                           r"pronto soccorso|policlinic",
    "professionista": r"avvocat|notai|commercialist|architett|ingegner|geometr|medic|professionist|consulent|"
                      r"perit|progettist|direttore dei lavori",
    "pubblica amministrazione": r"\bcomune\b|regione|provincia|ministero|amministrazion|ente pubblico|\basl\b|"
                                r"agenzia|prefettur|questur|\binps\b|\binail\b|universit|scuola statale",
    "albergatore": r"albergo|hotel|albergator|struttura ricettiva|pensione|b&b|bed and breakfast",
    "impresa": r"societ|impres|ditta|s\.?r\.?l|s\.?p\.?a|s\.?n\.?c|s\.?a\.?s|aziend|negozi|concessionari|"
               r"\bspa\b|\bsrl\b|commerciant|costrutt|produttor|cantier",
}
_SIDE_ROLES = {"cliente", "controparte", "terzo"}
# The roles as models write them outside the list, to the list's word.
_ROLE_SYNONYMS = ((r"parte assistita|assistit", "cliente"), (r"convenut|resistent|controinteressat", "controparte"),
                  (r"vittim|offes", "persona offesa"), (r"imputat|indagat", "indagato/imputato"),
                  (r"propriet", "proprietario"), (r"possessor", "possessore"), (r"vicin", "vicino"),
                  (r"vendit", "venditore"), (r"acquirent|compratore", "acquirente"), (r"inquilin", "conduttore"),
                  (r"dipendent", "lavoratore"), (r"\beredi\b", "erede"), (r"moglie|marito", "coniuge"),
                  (r"minorenn", "minore"))
# The kind of each event, from a fixed list, so the terms read what happened by its kind (a gate locked is a
# "chiusura/ostruzione accesso", which starts the possessory term), not by its words.
EVENT_KINDS = ("vendita", "divisione", "successione", "donazione", "consegna", "fine lavori", "costruzione/opera",
               "uso/passaggio", "spoglio", "turbativa", "chiusura/ostruzione accesso", "scoperta vizio",
               "denuncia/reclamo", "querela", "diffida", "azione giudiziaria", "impugnazione", "sentenza", "notifica",
               "provvedimento", "inadempimento", "pagamento", "incidente", "lesione", "decesso", "danneggiamento",
               "reato", "altro")
# How a property was acquired, and how an owner's ownership ended.
TRANSFER_KINDS = ("compravendita", "donazione", "successione", "divisione", "usucapione", "costituzione", "altro")
_TRANSFER_SYNONYMS = ((r"vend|compr|acquist", "compravendita"), (r"don", "donazione"),
                      (r"succe|eredit|testament", "successione"), (r"divis|frazion|separa|lottizz", "divisione"),
                      (r"usucap", "usucapione"), (r"alien", "compravendita"))

SCHEMA: Dict[str, Any] = {
    "parties": [{"name": "", "role": "", "age": ""}],
    "client_side": "",
    "property_history": [{"asset": "", "owner": "", "from": "", "to": "", "period": "", "together_with": "",
                          "how_acquired": "", "ended_by": ""}],
    "possession": [{"who": "", "asset": "", "since": "", "until": "", "how": "", "use": "", "still_ongoing": ""}],
    "works_or_signs": [{"what": "", "visible": "", "existing_since": "", "before_separation": "", "serves": "", "crosses": ""}],
    "events": [{"date": "", "what": "", "who": "", "kind": ""}],
    "access": [{"asset": "", "landlocked": "", "how_reached": ""}],
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
    "- parties: le persone e gli enti, ciascuno {name, role, age}; role SOLO tra: " + ", ".join(ROLES)
    + " (più ruoli separati da virgola). Un ruolo qualificato (pubblico ufficiale, incaricato di pubblico servizio, "
    "medico, struttura sanitaria, professionista, pubblica amministrazione, albergatore, impresa) SOLO se il testo "
    "lo dice di quella parte: essere proprietario, vicino o usare una strada non lo è; in mancanza «privato». "
    "age solo se il testo la dice.\n"
    "- client_side: la parte per cui si scrive il parere, cioè il cliente (nome e ruolo): dalla richiesta "
    "(\"assumi la difesa di X\", \"il legale di X\", \"per conto di X\"), altrimenti dal testo; mai l'avvocato.\n"
    "- property_history: i proprietari, anche precedenti, di ciascun bene (\"appartenevano a X\", \"erano di X\", "
    "\"di proprietà di X fino a...\") {asset (quale bene), owner (chi ne era proprietario), from, to (date come "
    "scritte), period (il periodo come scritto, per esempio \"all'inizio del 1900\", \"fino al 1950\"), together_with "
    "(gli altri beni che lo stesso proprietario aveva insieme a questo in quel periodo, o vuoto), how_acquired e "
    "ended_by (come lo ha acquistato e come ha smesso di esserne proprietario, tra: " + ", ".join(TRANSFER_KINDS)
    + "; ended_by è obbligatorio quando una voce successiva mostra quel bene diviso, frazionato o passato ad altri: "
    "per un bene unico poi in mano a proprietari diversi, «divisione»)}.\n"
    "- possession: chi possiede o usa un bene o un passaggio {who, asset, since e until (da quando e fino a quando, "
    "come scritto), how (come scritto: pacifico, continuo, pubblico, violento, clandestino, con tolleranza...), use "
    "(che uso: passaggio, coltivazione, abitazione, parcheggio...), still_ongoing (sì/no)}; \"con tolleranza\" o "
    "\"con permesso\" solo se il testo dice tolleranza, permesso o cortesia.\n"
    "- works_or_signs: opere e segni, anche preesistenti (strade, sentieri, passaggi, finestre, condotte, muri) "
    "{what, visible (sì se si vedono: una strada o un sentiero tracciati, una finestra, una condotta in vista; no se "
    "interrati o nascosti), existing_since (da quando, come scritto: anche \"esisteva già all'inizio del 1900\" dà "
    "\"all'inizio del 1900\"), before_separation (sì se esisteva già quando i "
    "beni sono stati divisi o venduti, \"esisteva già\", \"c'era già\"; no se è stata fatta dopo; vuoto se il testo "
    "non lo dice), serves (il bene che l'opera serve), crosses (i fondi che l'opera attraversa, con i loro proprietari "
    "se il testo li dice)}.\n"
    "- access: per ogni bene di cui il testo dice come si raggiunge {asset, landlocked (sì se il testo lo dice "
    "intercluso, senza accesso alla via pubblica o raggiungibile solo attraverso fondi altrui; no se ha un accesso "
    "suo), how_reached (come scritto)}.\n"
    "- events: i fatti datati e non {date (ESATTAMENTE come scritta nel testo, o vuota), what (in una frase, "
    "neutra), who, kind (SOLO tra: " + ", ".join(EVENT_KINDS) + "; chi chiude un cancello, sbarra o ostruisce un "
    "passaggio o un accesso: chiusura/ostruzione accesso)}.\n"
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
_COMPACT_SUFFIX = ("\n\nLa risposta precedente è stata troncata: rispondi di nuovo con lo stesso JSON, completo, più compatto: "
                   "valori brevi, nessuna ripetizione, senza omettere chiavi.")
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
    for party in out.get("parties") or []:
        party["role"] = _clean_roles(party.get("role", ""), party.get("name", ""), case_text)
    for item in out.get("events") or []:
        item["kind"] = _enum(item.get("kind", ""), EVENT_KINDS)
    if "possession" in out:
        _stated_tolerance(out, case_text)
    if "access" in out:
        _stated_landlocked(out, case_text)
    if "client_side" in out:
        _client(out, case_text)
    if out.get("parties"):
        one_client(out, case_text)
    for item in out.get("property_history") or []:
        for field in ("how_acquired", "ended_by"):
            item[field] = _enum(item.get(field, ""), TRANSFER_KINDS, _TRANSFER_SYNONYMS)
    out["split_into_parts"] = splits(out.get("property_history") or [])
    _infer_transfers(out.get("property_history") or [])
    for key, field in (("events", "date"), ("proceedings", "date"), ("possession", "since"), ("possession", "until"),
                       ("works_or_signs", "existing_since"), ("property_history", "from"), ("property_history", "to"),
                       ("property_history", "period")):
        for item in out[key]:
            written = item.get(field, "")
            if written and " ".join(written.lower().split()) not in norm_case:
                item[field] = ""                          # not in the text as written: never kept
            parsed = parse_date(item.get(field, ""))
            item[f"{field}_iso"] = parsed[2] if parsed else ""
    if "works_or_signs" in out:
        _work_since(out, case_text)
    return out


# --- what the case states, read where the model left a field empty or filled it without the text ----------------

# 27 Oct (fact_probe on a prod narrative): a road that "esisteva già all'inizio del 1900" came out with no date, so
# the servitù by usucapione could not be weighed; "con tolleranza" given though the text never says so; a plot the
# text calls "intercluso" left the landlocked rule unfired; the client taken from the wrong place.
_ALREADY_RE = re.compile(r"esistev\w* già|c'er\w* già|er\w* già (?:present|esistent|tracciat|in uso|aperta|aperto)|"
                         r"già esistent|preesistent|già allora|già al(?:l')? ?(?:tempo|epoca)|risal\w* a", re.IGNORECASE)
_DATE_PREFIX_RE = re.compile(r"(?:(?:all'inizio|alla fine|alla metà|ai primi|agli inizi) (?:del|degli anni|dell')\s*|"
                             r"(?:negli|dagli) anni\s+|(?:sin|fin) dal\s+|dal\s+|nel\s+|intorno al\s+)$", re.IGNORECASE)


def _written_date(sentence: str) -> str:
    """The first date of the sentence as written, with its leading words ("all'inizio del 1900", "nel 1980")."""
    for pattern, _ in _DATE_PATTERNS:
        m = pattern.search(sentence or "")
        if m:
            prefix = _DATE_PREFIX_RE.search(sentence[:m.start()])
            return sentence[prefix.start() if prefix else m.start():m.end()].strip()
    return ""


def _sentences_of(text: str) -> List[str]:
    return [x.strip() for x in re.split(r"(?<=[.;!?])\s+|\n+", text or "") if x.strip()]


# Words that name any plot: two assets sharing only these are not shown to be the same land.
_ASSET_GENERIC = frozenset("terreno terreni fondo fondi podere poderi particella particelle appezzamento appezzamenti lotto "
                           "lotti immobile immobili proprieta proprietà bene beni parte parti area aree confinante "
                           "confinanti della delle dello degli".split())
SEQUENCE_DEDUCED = "dedotto dalla sequenza dei proprietari"


def _asset_words(text: Any) -> set:
    return {w[:7] for w in re.findall(r"[a-zà-ù0-9]+", _low(text)) if w not in _ASSET_GENERIC
            and w not in ("di", "del", "dei", "e", "ed", "il", "la", "lo", "i", "le", "a", "al", "o", "su", "in", "con")}


def _named_in(owner: str, text: str) -> bool:
    return bool(owner) and re.search(rf"\b{re.escape(owner)}\b", text) is not None


def _parts_of(entry: Dict[str, Any], later: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], bool]:
    """(the later entries holding this asset or a part of it, whether it was split): other owners whose asset it names,
    shares its own words with, or lists in together_with. Split: two named owners, two distinct parts, or an owner holding
    only a part. The same asset passing on, from one owner to the next, is no split however long the chain."""
    owner = _low(entry.get("owner"))
    text = f"{_low(entry.get('asset'))} {_low(entry.get('together_with'))}"
    words = _asset_words(entry.get("asset")) | _asset_words(entry.get("together_with"))
    parts = [h for h in later if _low(h.get("owner")) not in ("", owner)
             and (_named_in(_low(h.get("owner")), text) or words & _asset_words(h.get("asset")))]
    named = {_low(h.get("owner")) for h in parts if _named_in(_low(h.get("owner")), text)}
    assets = {frozenset(_asset_words(h.get("asset"))) for h in parts} - {frozenset()}
    return parts, len(named) >= 2 or len(assets) >= 2 or any(a < words for a in assets)


def splits(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The assets one owner held and others later held in parts, by the sequence alone: whatever ended_by says, which is a
    word the model may vary from one run to the next (art. 1062 c.c. covers division and alienation in parts alike)."""
    out = []
    for i, entry in enumerate(history):
        owner = str(entry.get("owner") or "").strip()
        parts, split = _parts_of(entry, history[i + 1:])
        if owner and split:
            out.append({"asset": entry.get("asset") or "", "owner": owner, "how": SEQUENCE_DEDUCED,
                        "parts": ", ".join(dict.fromkeys(str(h.get("owner")).strip() for h in parts))})
    return out


def _infer_transfers(history: List[Dict[str, Any]]) -> None:
    """A field the model left empty, from the sequence, in place, for the terms and dates reading it: "divisione" for a
    split, the parts' how_acquired too; "altro" for an asset passing whole, the sequence not saying how. Never over a
    field the model filled in. ended_by_how says it was deduced."""
    for i, entry in enumerate(history):
        if not _low(entry.get("owner")) or _low(entry.get("ended_by")):
            continue
        parts, split = _parts_of(entry, history[i + 1:])
        if not parts:
            continue
        entry["ended_by"], entry["ended_by_how"] = "divisione" if split else "altro", SEQUENCE_DEDUCED
        for later in parts if split else ():
            if not _low(later.get("how_acquired")):
                later["how_acquired"], later["how_acquired_how"] = "divisione", SEQUENCE_DEDUCED


def _work_since(out: Dict[str, Any], case_text: str) -> None:
    """A work with no existing_since: the date of a sentence saying the work already existed then ("la strada
    esisteva già all'inizio del 1900"); else, for a work that existed before the separation, the date the common
    ownership ended (property_history's to or period). existing_since_how says which."""
    sentences = _sentences_of(case_text)
    for work in out.get("works_or_signs") or []:
        if work.get("existing_since"):
            continue
        stems = {w[:5] for w in re.findall(r"[a-zà-ù]{4,}", _low(work.get("what")))}
        written = next((d for x in sentences if _ALREADY_RE.search(x) and stems & {w[:5] for w in re.findall(
            r"[a-zà-ù]{4,}", _low(x))} for d in [_written_date(x)] if d), "")
        how = "il testo: esisteva già"
        if not written and _yes(work.get("before_separation")):
            split_assets = {(s["asset"], s["owner"]) for s in out.get("split_into_parts") or []}
            written = next((h.get(f) for h in out.get("property_history") or [] for f in ("to", "period")
                            if h.get(f) and parse_date(h[f]) and ((h.get("asset"), h.get("owner")) in split_assets
                                                                  or _low(h.get("together_with"))
                                                                  or h.get("ended_by") in ("divisione", "compravendita"))), "")
            how = "la fine della proprietà comune (esisteva già alla divisione o vendita)"
        if written:
            parsed = parse_date(written)
            work.update(existing_since=written, existing_since_iso=parsed[2] if parsed else "", existing_since_how=how)


_TOLERANCE_RE = r"toller|permess|cortesi|concess|precari|autorizz|consens|benevol|favore"
_TOLERANCE_STATED_RE = (r"tolleranz|tollerat|permesso|per cortesia|cortesia|in concessione|concession|precari|"
                        r"autorizzazion|autorizzat|consenso|acconsent|benevolenz|per favore|a titolo di favore")


def _stated_tolerance(out: Dict[str, Any], case_text: str) -> None:
    """Tolerance or permission in a possession's "how" only when the text states it; otherwise those words go
    (how_dropped keeps them)."""
    if re.search(_TOLERANCE_STATED_RE, _low(case_text)):
        return
    for item in out.get("possession") or []:
        parts = [x.strip() for x in re.split(r",|;|/|\be\b", item.get("how", "")) if x.strip()]
        dropped = [x for x in parts if re.search(_TOLERANCE_RE, _low(x))]
        if dropped:
            item["how"] = ", ".join(x for x in parts if x not in dropped)
            item["how_dropped"] = ", ".join(dropped)


_LANDLOCKED_RE = re.compile(
    r"(?<!non è )(?<!non era )(?<!non sia )\binterclus\w*|senza (?:alcun |nessun |un )?(?:accesso|sbocco|uscita) "
    r"(?:diretto |autonomo |propri\w )?(?:alla|sulla|verso la|alle|sulle) (?:via|strada|vie|strade) pubblic\w*|"
    r"non (?:ha|abbia|aveva|hanno) (?:alcun |altro |un )?(?:accesso|sbocco|uscita) (?:diretto |autonomo )?"
    r"(?:alla|sulla|alle) (?:via|strada|vie) pubblic\w*|privo di (?:ogni |qualsiasi |alcun )?(?:accesso|sbocco)|"
    r"non (?:è|era|sono|erano) raggiungibil\w* se non|raggiungibil\w* (?:solo|soltanto|unicamente) (?:attraverso|"
    r"passando|tramite)|unic\w* (?:accesso|sbocco|via d'accesso)|circondat\w* da fondi altrui", re.IGNORECASE)


def _stated_landlocked(out: Dict[str, Any], case_text: str) -> None:
    """A plot the text explicitly calls landlocked gets access.landlocked "sì" (the fondo_intercluso rule reads it),
    with the sentence that says so, when the model left it out."""
    if any(_yes(a.get("landlocked")) for a in out.get("access") or []):
        return
    said = next((x for x in _sentences_of(case_text) if _LANDLOCKED_RE.search(x)), "")
    if said:
        out.setdefault("access", []).append({"asset": "", "landlocked": "sì", "how_reached": " ".join(said.split())[:200]})


_NAME = r"((?:[A-Z][\w'’.-]*)(?:\s+(?:[A-Z][\w'’.-]*|s\.?r\.?l\.?|s\.?p\.?a\.?|srl|spa))*)"
_THE = r"(?:(?:il|la|lo|l'|i|gli|le) )?(?:(?:sig\.?(?:ra)?|signor[ae]?|dott\.?(?:ssa)?|società|ditta) )?"
_CLIENT_RES = [re.compile(r"(?i:assum\w*|prend\w*)\s+(?i:la\s+|le\s+)?(?i:difesa|patrocinio|vesti del (?:legale|"
                          r"difensore|avvocato))\s+(?i:di|del|della|dello|dell')\s*" + _THE + _NAME),
               re.compile(r"(?i:legale|difensore|avvocato|patrocinatore)\s+(?i:di|del|della|dello|dell')\s*" + _THE + _NAME),
               re.compile(r"(?i:per conto|nell'interesse)\s+(?i:di|del|della|dello|dell')\s*" + _THE + _NAME)]
_LAWYER_RE = r"\bavvocat|\blegale\b|difensor|candidat|studio legale|\bscrivente\b|\bconsulente\b|\bio\b"
_NOT_A_CLIENT = {"Il", "La", "Lo", "Si", "Che", "Codice", "Corte", "Tribunale", "Cassazione", "Stato"}


def client_from(text: str) -> str:
    """The client a text names ("assumi la difesa di X", "il legale di X", "per conto di X"), or ""."""
    for pattern in _CLIENT_RES:
        for m in pattern.finditer(text or ""):
            name = m.group(1).strip(" .,'’")
            if name and name.split()[0] not in _NOT_A_CLIENT:
                return name
    return ""


def _client(out: Dict[str, Any], case_text: str) -> None:
    """client_side from the user's request, else from the text, else the model's own when it names no lawyer
    ("" then); the party it names gets the role "cliente". client_side_how says where it came from."""
    text = case_text or ""
    request = text.split("DOCUMENTO CARICATO:", 1)[0] if text.startswith("RICHIESTA DELL'UTENTE:") else ""
    model = out.get("client_side", "")
    name, how = client_from(request), "la richiesta"
    if not name:
        name, how = client_from(text), "il testo"
    if not name:
        name, how = client_from(model), "la risposta del modello"
    if not name:
        if model and re.search(_LAWYER_RE, _low(model)):
            out["client_side"], out["client_side_how"] = "", f"scartato: «{model}» è l'avvocato"
        return
    party = next((p for p in out.get("parties") or [] if p.get("name") and (
        _low(name) in _low(p["name"]) or _low(p["name"]) in _low(name))), None)
    if party is not None and "cliente" not in _low(party.get("role")):
        party["role"] = ", ".join(["cliente"] + [r for r in (party.get("role") or "").split(", ") if r and r != "cliente"])
    out["client_side"] = name + (f" ({party['role']})" if party and party.get("role") else "")
    out["client_side_how"] = how


_ASKS_RE = r"si rivolg\w*|chied\w*|vuol\w*|intend\w*|incaric\w*|domand\w*|desider\w*"


def _drop_role(party: Dict[str, Any], role: str) -> None:
    roles = [r for r in str(party.get("role") or "").split(", ") if r and r != role]
    party["role"] = ", ".join(roles) or "privato"


def one_client(out: Dict[str, Any], case_text: str) -> None:
    """v57 (v56-h10: "Tizio (cliente); Caio (cliente, controparte)"): exactly one party is the client, and the client is
    never the other side. The client is the party client_side names; else, of several, the one the case says turns to
    the lawyer or asks (_ASKS_RE in a sentence naming it, alone); else none: out["client_ambiguous"] lists them and the
    interactive flow asks it first (interaction_copy.QUESTIONS["parte_assistita"])."""
    parties = [p for p in out.get("parties") or [] if isinstance(p, dict) and p.get("name")]
    clients = [p for p in parties if "cliente" in _low(p.get("role")).split(", ")]
    side = _low(out.get("client_side")).split("(")[0].strip()
    named = [p for p in parties if side and (_low(p["name"]) in side or side in _low(p["name"]))]
    winner = named[0] if len(named) == 1 else None
    if winner is None and len(clients) > 1:
        sentences = [x for x in re.split(r"(?<=[.;!?])\s+", case_text or "") if re.search(_ASKS_RE, _low(x))]
        asking = [p for p in clients if any(re.search(rf"\b{re.escape(_low(p['name']))}\b", _low(x)) for x in sentences)]
        winner = asking[0] if len(asking) == 1 else None
        if winner is None:
            out["client_ambiguous"] = [p["name"] for p in clients]
            for p in clients:
                _drop_role(p, "cliente")
            return
    if winner is None:
        winner = clients[0] if len(clients) == 1 else None
    if winner is None:
        return
    for p in parties:
        if p is winner:
            if "cliente" not in _low(p.get("role")).split(", "):
                p["role"] = ", ".join(["cliente"] + [r for r in str(p.get("role") or "").split(", ") if r])
            _drop_role(p, "controparte")
        elif "cliente" in _low(p.get("role")).split(", "):
            _drop_role(p, "cliente")


def _enum(value: str, allowed: Tuple[str, ...], synonyms: Tuple[Tuple[str, str], ...] = ()) -> str:
    """value as one of allowed (the first it names, else by synonyms), "altro" for anything else, "" when empty."""
    low = _low(value)
    if not low:
        return ""
    found = next((a for a in allowed if a != "altro" and a in low), None) \
        or next((a for pattern, a in synonyms if re.search(pattern, low)), None)
    return found or "altro"


def _party_context(name: str, case_text: str) -> str:
    """What the case says of the party itself: at each mention of its name (a word of 4+ letters), the two words
    before it within its clause ("la dottoressa Neri") and the rest of its sentence after it, up to 100 characters
    ("Neri, medico del pronto soccorso"); "" when the name is not in the text. A cue elsewhere in the sentence is
    another party's."""
    words = [w for w in re.findall(r"[a-zà-ù]{4,}", _low(name)) if w not in ("della", "dello", "degli", "delle")]
    low = _low(case_text)
    out = []
    for word in words:
        for m in re.finditer(rf"\b{re.escape(word)}\b", low):
            before = " ".join(re.split(r"[,.;:!?()\n]", low[:m.start()])[-1].split()[-2:])
            after = re.split(r"[.;!?\n]", low[m.end():m.end() + 100])[0]
            out.append(f"{before} {word}{after}")
    return " ".join(out)


# v57 (v55 l3: "Inquilino (acquirente)"): a party the case names by its role keeps it, the roles it excludes removed
NAMED_ROLES = ((r"\binquilin|\bconduttor|\baffittuari|\blocatari", "conduttore", {"acquirente", "venditore", "proprietario"}),)


def _clean_roles(written: str, name: str, case_text: str) -> str:
    """The party's roles from ROLES only (synonyms mapped); a qualified one (QUALIFIED_ROLE_CUES) only if the
    party's name or the words around it in the case state it (_party_context); "privato" for a party left with
    no role but its side."""
    low = _low(written)
    roles = [r for r in ROLES if r in low]
    roles += [r for pattern, r in _ROLE_SYNONYMS if re.search(pattern, low) and r not in roles]
    where = _low(name) + " " + _party_context(name, case_text)
    roles = [r for r in roles if r not in QUALIFIED_ROLE_CUES or re.search(QUALIFIED_ROLE_CUES[r], where)]
    for pattern, role, excluded in NAMED_ROLES:
        if re.search(pattern, _low(name)):
            roles = [r for r in roles if r not in excluded] + ([role] if role not in roles else [])
    if not set(roles) - _SIDE_ROLES:
        roles.append("privato")
    return ", ".join(dict.fromkeys(roles))


def extract_facts(case_text: str, raw_replies: Optional[List[str]] = None) -> Tuple[Optional[Dict[str, Any]], str]:
    """(facts or None, problem): one call at temperature 0, its reply validated (validate_facts) and,
    when it is not valid, asked once more. raw_replies, when given, receives each reply as it came
    (scripts/fact_probe.py)."""
    human = f"CASO:\n{(case_text or '')[:_CASE_CHARS]}"
    base = _EXTRACT_SYSTEM.replace("Sii conciso:", _EXTRACT_REGIMES.lstrip("\n") + "Sii conciso:") \
        if regimes_enabled() else _EXTRACT_SYSTEM
    system, problem = base, ""
    salvaged = None
    for attempt in (1, 2):
        last_reply.finish_reason = None
        cut = False
        try:
            raw = _call_chat([SystemMessage(content=system), HumanMessage(content=human)],
                             max_tokens=_EXTRACT_TOKENS, temperature=0)
            if raw_replies is not None:
                raw_replies.append(raw)
            cut = getattr(last_reply, "finish_reason", None) == "length"
            facts = validate_facts(_parse_or_repair(raw), case_text)
        except Exception as exc:
            _log_exception("extract_facts", exc)
            facts, problem = None, f"{type(exc).__name__}: {str(exc)[:160]}"
        if cut:
            # what follows the cut is lost, and a rule may need it: the complete items are kept, and it is asked once
            # more, compact
            logger.warning("fact extraction: reply cut at %d tokens (attempt %d), %s", _EXTRACT_TOKENS, attempt,
                           "complete items kept" if facts is not None else "nothing complete")
            vlog("llm_reply_cut", {"call": "extract_facts", "max_tokens": _EXTRACT_TOKENS, "attempt": attempt,
                                   "salvaged": facts is not None})
            salvaged = facts if facts is not None else salvaged
            system = base + _COMPACT_SUFFIX
            continue
        if facts is not None:
            return facts, ""
        problem = problem or "reply is not a JSON object with the schema's keys"
        system = base + _RETRY_SUFFIX
    return (salvaged, "") if salvaged is not None else (None, problem or "reply cut at the token limit twice")


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
        return salvage_json(raw)
    except ValueError:
        if value is None:
            raise
        return value


# --- 2. dates and terms ----------------------------------------------------------------------------

_MONTHS = {m: i for i, m in enumerate(("gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio",
                                       "agosto", "settembre", "ottobre", "novembre", "dicembre"), 1)}
_MONTH_RE = "|".join(_MONTHS)
_DATE_PATTERNS = (
    # v60d: a relative week resolved (relative_date): the interval, its first day as the iso
    (re.compile(rf"\b(?:settimana|periodo) dal (\d{{1,2}}) ({_MONTH_RE}) ((?:18|19|20)\d{{2}}) al (\d{{1,2}}) "
                rf"({_MONTH_RE}) ((?:18|19|20)\d{{2}})\b", re.IGNORECASE), "week"),     # v60g: a relative period too
    (re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-]((?:18|19|20)\d{2})\b"), "dmy"),
    (re.compile(r"\b((?:18|19|20)\d{2})-(\d{1,2})-(\d{1,2})\b"), "ymd"),
    (re.compile(rf"\b(\d{{1,2}})(?:°|º)?\s+({_MONTH_RE})\s+((?:18|19|20)\d{{2}})\b", re.IGNORECASE), "d month y"),
    (re.compile(rf"\b({_MONTH_RE})\s+(?:del\s+|dell'\s*)?((?:18|19|20)\d{{2}})\b", re.IGNORECASE), "month y"),
    (re.compile(r"\b((?:18|19|20)\d{2})\b"), "y"),
)


def parse_date(text: str) -> Optional[Tuple[dt.date, dt.date, str]]:
    """(earliest, latest, iso) of the first date written in text: a day, a month ("marzo 2016") or a year
    ("2016", "nel 2016"); None when it holds none. Relative dates ("tre mesi dopo") are not dates."""
    for pattern, kind in _DATE_PATTERNS:
        m = pattern.search(text or "")
        if not m:
            continue
        try:
            if kind == "week":
                a = dt.date(int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1)))
                b = dt.date(int(m.group(6)), _MONTHS[m.group(5).lower()], int(m.group(4)))
                return a, b, a.isoformat()
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


# v60d: a date the user writes relative to the day they write it («2 giorni fa», «ieri», «la settimana scorsa», «a
# marzo», «lunedì scorso»), resolved against that day (reply_today) to a date parse_date reads. A case's own text is
# never read this way (its present is not today: present()); only the user's words in the conversation are.
_NUMBER_WORDS = {"un": 1, "uno": 1, "una": 1, "due": 2, "tre": 3, "quattro": 4, "cinque": 5, "sei": 6, "sette": 7,
                 "otto": 8, "nove": 9, "dieci": 10, "undici": 11, "dodici": 12, "quindici": 15, "venti": 20,
                 "trenta": 30}
_WEEKDAYS = {"lunedì": 0, "lunedi": 0, "martedì": 1, "martedi": 1, "mercoledì": 2, "mercoledi": 2, "giovedì": 3,
             "giovedi": 3, "venerdì": 4, "venerdi": 4, "sabato": 5, "domenica": 6}
_COUNT_RE = r"(\d{1,3}|" + "|".join(_NUMBER_WORDS) + r")"
_WEEKDAY_RE = "|".join(_WEEKDAYS)


def reply_today() -> dt.date:
    """The day the user writes: TODAY_ENV when set (tests, simulations), else the real today."""
    value = (os.getenv(TODAY_ENV) or "").strip()
    try:
        return dt.date.fromisoformat(value) if value else dt.date.today()
    except ValueError:
        return dt.date.today()


def _written_day(d: dt.date) -> str:
    return f"{d.day} {list(_MONTHS)[d.month - 1]} {d.year}"


def _written_month(year: int, month: int) -> str:
    return f"{list(_MONTHS)[month - 1]} {year}"


def _period(a: dt.date, b: dt.date) -> str:
    a, b = min(a, b), max(a, b)
    return f"periodo dal {_written_day(a)} al {_written_day(b)}"


def _back(today: dt.date, n: float, unit: str) -> dt.date:
    """today less n units (giorni, settimane, mesi, anni; a fraction of a month or a year in days)."""
    if unit == "gio":
        return today - dt.timedelta(days=round(n))
    if unit == "set":
        return today - dt.timedelta(days=round(7 * n))
    if unit == "mes":
        whole = int(n)
        return add_period(today, months=-whole) - dt.timedelta(days=round((n - whole) * 30))
    whole = int(n)
    return add_period(today, years=-whole) - dt.timedelta(days=round((n - whole) * 365))


_UNIT_RE = r"(giorn[oi]|settiman[ae]|mes[ei]|ann[oi])"
_APOSTROPHES_RE = re.compile(r"[’‘`´ʼ′]")


def relative_text(text: Any) -> str:
    """v60h (7): the words a relative date is matched on -- lower case, accents off («fà», «più», «metà», «lunedì»),
    every apostrophe one («’», «`», «´») with no space after it («un' ora» -> «un'ora»), an elided «un'» a count («un'ora»
    -> «un ora»), «vent'anni» / «trent'anni» spelled out."""
    import unicodedata
    low = _APOSTROPHES_RE.sub("'", str(text or "").lower())
    low = "".join(c for c in unicodedata.normalize("NFKD", low) if not unicodedata.combining(c))
    low = re.sub(r"'\s+", "'", " ".join(low.split()))
    low = re.sub(r"\bun'(?=[a-z])", "un ", low)
    return re.sub(r"\b(vent|trent)'anni\b", lambda m: {"vent": "venti", "trent": "trenta"}[m.group(1)] + " anni", low)
_SEASONS = {"primavera": (3, 5), "estate": (6, 8), "autunno": (9, 11), "inverno": (12, 2)}
_FAR_YEARS = 10             # «più di N … fa»: the earliest start at least this far back (an open bound, never a guess)
_MONTH_PARTS = {"inizio": (1, 10), "metà": (10, 20), "meta": (10, 20), "fine": (20, 31)}
_YEAR_PARTS = {"inizio": (1, 3), "metà": (5, 8), "meta": (5, 8), "fine": (10, 12)}


def relative_period(text: str, today: Optional[dt.date] = None) -> str:
    """v60g (6): a vague period relative to today, written as the range parse_date reads («periodo dal D M Y al D M
    Y»; "" when none): «circa N mesi fa» (N less and more a quarter of N, one unit at least), «qualche/alcuni/pochi
    giorni|settimane|mesi|anni fa» (2 to 6 of them), «più di / oltre N … fa» (from N to a bound far back: 3N, or
    _FAR_YEARS years), «meno di N … fa» and «negli ultimi N …» (N ago to today), «da (circa / più di) N …» (a start that
    long ago), «l'estate / l'inverno scorso» (the last such season ended before today), «a inizio / metà / fine del mese
    o dell'anno scorso», «tra marzo e aprile» (the latest such span not after today)."""
    today = today or reply_today()
    low = relative_text(text)
    if not low:
        return ""

    def count(word: str) -> int:
        return int(word) if word.isdigit() else _NUMBER_WORDS.get(word, 0)
    m = re.search(rf"\b(?:circa|quasi|piu o meno|all'incirca|grosso modo|intorno a) {_COUNT_RE} {_UNIT_RE} fa\b", low)
    if m and count(m.group(1)):
        n, unit = count(m.group(1)), m.group(2)[:3]
        d = max(1.0, n / 4)
        return _period(_back(today, n + d, unit), _back(today, max(0.0, n - d), unit))
    m = re.search(rf"\b(?:qualche|alcuni|alcune|pochi|poche|diversi|diverse) {_UNIT_RE} fa\b", low)
    if m:
        unit = m.group(1)[:3]
        return _period(_back(today, 6, unit), _back(today, 2, unit))
    m = re.search(rf"\b(?:piu di|oltre|da piu di|da oltre) {_COUNT_RE} {_UNIT_RE}(?: fa)?\b", low)
    if m and count(m.group(1)):
        n, unit = count(m.group(1)), m.group(2)[:3]
        far = min(_back(today, 3 * n, unit), add_period(today, years=-_FAR_YEARS))
        return _period(far, _back(today, n, unit))
    m = re.search(rf"\b(?:meno di|non piu di|neanche|nemmeno) {_COUNT_RE} {_UNIT_RE} fa\b", low) \
        or re.search(rf"\bnegli ultimi {_COUNT_RE} {_UNIT_RE}\b", low)
    if m and count(m.group(1)):
        return _period(_back(today, count(m.group(1)), m.group(2)[:3]), today)
    m = re.search(r"\bnell'ultim[oa] (settimana|mese|anno)\b", low)
    if m:
        return _period(_back(today, 1, m.group(1)[:3]), today)
    m = re.search(rf"\bda (?:circa |quasi |almeno )?{_COUNT_RE} {_UNIT_RE}\b", low)
    if m and count(m.group(1)) and not re.search(r"\bfa\b", low[m.end():m.end() + 4]):
        n, unit = count(m.group(1)), m.group(2)[:3]
        d = max(1.0, n / 4) if re.search(r"\bda (?:circa|quasi) ", m.group(0)) else 0.0
        return _period(_back(today, n + d, unit), _back(today, max(0.0, n - d), unit)) if d else \
            _written_day(_back(today, n, unit))
    m = re.search(r"\b(primavera|estate|autunno|inverno) (?:scors[oa]|passat[oa])\b", low) or \
        re.search(r"\b(?:scors[oa]|passat[oa]) (primavera|estate|autunno|inverno)\b", low)
    if m:
        first, last = _SEASONS[m.group(1)]
        for year in (today.year, today.year - 1, today.year - 2):
            start = dt.date(year - (1 if first > last else 0), first, 1)
            end_month_year = year
            end = dt.date(end_month_year, last, calendar.monthrange(end_month_year, last)[1])
            if end < today:
                return _period(start, end)
        return ""
    m = re.search(r"\b(?:a |ad |all'|alla )?(inizio|meta|fine) (?:del |dello |dell')?(mese|anno) (?:scorso|passato)\b",
                  low)
    if m:
        part, what = m.group(1), m.group(2)
        if what == "mese":
            ref = add_period(today.replace(day=1), months=-1)
            last = calendar.monthrange(ref.year, ref.month)[1]
            a, b = _MONTH_PARTS[part]
            return _period(ref.replace(day=a), ref.replace(day=min(b, last)))
        a, b = _YEAR_PARTS[part]
        year = today.year - 1
        return _period(dt.date(year, a, 1), dt.date(year, b, calendar.monthrange(year, b)[1]))
    m = re.search(rf"\btra (?:il mese di )?({_MONTH_RE}) e ({_MONTH_RE})\b", low)
    if m and not parse_date(low):
        first, last = _MONTHS[m.group(1)], _MONTHS[m.group(2)]
        for year in (today.year, today.year - 1):
            start = dt.date(year - (1 if first > last else 0), first, 1)
            end = dt.date(year, last, calendar.monthrange(year, last)[1])
            if start <= today:
                return _period(start, min(end, today))
    return ""


def relative_date(text: str, today: Optional[dt.date] = None) -> str:
    """The first relative date in text written as a date parse_date reads ("" when none): a day («ieri», «2 giorni
    fa», «lunedì scorso», «il 3 marzo»), a week («la settimana scorsa»), a month («a marzo», «2 mesi fa», «il mese
    scorso») or a year («l'anno scorso», «3 anni fa»). A month or a day without its year is the latest one not after
    today. v60g (6): first a vague period (relative_period: «circa tre mesi fa», «qualche settimana fa», «da più di un
    anno», «l'estate scorsa») as its range."""
    today = today or reply_today()
    low = relative_text(text)
    if not low:
        return ""
    period = relative_period(low, today)
    if period:
        return period

    def count(word: str) -> int:
        return int(word) if word.isdigit() else _NUMBER_WORDS.get(word, 0)
    m = re.search(rf"\b(?:{_COUNT_RE}|qualche|poche|alcune) or[ae] fa\b", low)
    if m:                       # v60h (7): «un'ora fa», «2 ore fa»: today
        return _written_day(today)
    m = re.search(rf"\b{_COUNT_RE} (giorn[oi]|settiman[ae]|mes[ei]|ann[oi]) fa\b", low)
    if m and count(m.group(1)):
        n, unit = count(m.group(1)), m.group(2)[:3]
        if unit == "gio":
            return _written_day(today - dt.timedelta(days=n))
        if unit == "set":
            return _written_day(today - dt.timedelta(days=7 * n))
        if unit == "mes":
            d = add_period(today, months=-n)
            return _written_month(d.year, d.month)
        return str(today.year - n)
    if re.search(r"\b(?:l'altro ieri|l'altroieri|altroieri|avantieri)\b", low):
        return _written_day(today - dt.timedelta(days=2))
    if re.search(r"\bieri\b", low):
        return _written_day(today - dt.timedelta(days=1))
    if re.search(r"\b(?:oggi|stamattina|stamane|stasera|stanotte|poco fa|poche ore fa)\b", low):
        return _written_day(today)
    if re.search(r"\b(?:la |nella )?(?:settimana scorsa|scorsa settimana|settimana passata)\b", low):
        start = today - dt.timedelta(days=today.weekday() + 7)
        end = start + dt.timedelta(days=6)
        return f"settimana dal {_written_day(start)} al {_written_day(end)}"
    if re.search(r"\b(?:il |nel |lo )?(?:mese scorso|scorso mese|mese passato)\b", low):
        d = add_period(today.replace(day=1), months=-1)
        return _written_month(d.year, d.month)
    if re.search(r"\b(?:l'anno scorso|lo scorso anno|nell'anno passato|l'anno passato)\b", low):
        return str(today.year - 1)
    m = re.search(rf"\b(?:(?:lo |la )?scors[oa] ({_WEEKDAY_RE})|({_WEEKDAY_RE}) scors[oa])\b", low)
    if m:
        wanted = _WEEKDAYS[m.group(1) or m.group(2)]
        back = (today.weekday() - wanted) % 7 or 7
        return _written_day(today - dt.timedelta(days=back))
    if parse_date(low):         # an absolute date (with its year) is no relative one
        return ""
    m = re.search(rf"\b(?:il |lo scorso |l'|lo )?(\d{{1,2}})(?:°|º)? ({_MONTH_RE})\b", low)
    if m:
        month, day = _MONTHS[m.group(2)], int(m.group(1))
        for year in (today.year, today.year - 1):
            try:
                d = dt.date(year, month, day)
            except ValueError:
                continue
            if d <= today:
                return _written_day(d)
        return ""
    m = re.search(rf"\b(?:a|ad|in|nel|nel mese di|lo scorso|questo|a fine|a inizio|ai primi di)\s+({_MONTH_RE})\b", low) \
        or re.search(rf"\b({_MONTH_RE}) scorso\b", low)
    if m:
        month = _MONTHS[m.group(1)]
        year = today.year if month <= today.month else today.year - 1
        return _written_month(year, month)
    return ""


RELATIVE_FIELDS = (("events", "date"), ("proceedings", "date"), ("possession", "since"), ("possession", "until"),
                   ("works_or_signs", "existing_since"))


def resolve_relative(facts: Optional[Dict[str, Any]], today: Optional[dt.date] = None) -> List[Dict[str, str]]:
    """v60g (6): in place, every date field of the facts written relative to the case's present («tre mesi fa», «da
    circa due anni», «l'estate scorsa») as the date or the range it stands for (relative_date, against the case's own
    present_date, else the day the user writes: reply_today), its words kept in <field>_relative; the case's present
    set to that day when it has none. The resolved ones."""
    if not facts:
        return []
    present = parse_date(str(facts.get("present_date") or ""))
    today = today or (present[0] if present else reply_today())
    done = []
    for key, field in RELATIVE_FIELDS:
        for item in facts.get(key) or []:
            if not isinstance(item, dict):
                continue
            written = str(item.get(field) or "").strip()
            if not written or parse_date(written):
                continue
            resolved = relative_date(written, today)
            if not resolved or not parse_date(resolved):
                continue
            item[f"{field}_relative"] = written
            item[field] = resolved
            item[f"{field}_iso"] = parse_date(resolved)[2]
            done.append({"field": f"{key}.{field}", "written": written, "now": resolved})
    if done and not present:
        facts["present_date"] = _written_day(today)
    return done


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
    starts = _possession_starts(facts or {})
    items = [(i[1], i[0], i[2]) for i in _events(facts or {}, r".") + starts] \
        + [(i[1], i[3], i[4]) for i in starts if i[3]]
    return max(items, key=lambda x: x[1][0], default=None)


_NON_OWNER_RE = r"a non domino|non (?:era|fosse|è) (?:il |la )?(?:\w+ )?proprietari|non proprietari|da chi non (?:era|fosse)"


def _title_from_non_owner(facts: Dict[str, Any]) -> bool:
    """Art. 1159 c.c. needs a title from one who was not the owner, transcribed: an ordinary purchase is neither."""
    text = _texts(facts, ("events", "contracts", "property_history", "possession"))
    return bool(re.search(_NON_OWNER_RE, text) and re.search(r"trascri", text))


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


def _typed_kind(item: Dict[str, Any]) -> str:
    kind = _low(item.get("kind"))
    return kind if kind in EVENT_KINDS and kind != "altro" else ""


def _matches(item: Dict[str, Any], pattern: str, fields: Tuple[str, ...], kinds: Tuple[str, ...] = (),
             both: bool = False) -> bool:
    """An event by its typed kind when it has one and kinds are asked (no keyword match on its words then; with
    both, its words must match too); by pattern over its fields otherwise."""
    kind = _typed_kind(item)
    if kinds and kind:
        return kind in kinds and (not both or bool(re.search(pattern, " ".join(_low(item.get(f)) for f in fields))))
    return bool(pattern) and bool(re.search(pattern, " ".join(_low(item.get(f)) for f in fields)))


def _dated(items: List[Dict[str, str]], pattern: str, fields: Tuple[str, ...], date_field: str = "date",
           kinds: Tuple[str, ...] = (), both: bool = False) -> List[tuple]:
    """[(interval, what, written date)] of the items that match (_matches) and whose date parses, earliest first."""
    out = []
    for item in items or []:
        parsed = parse_date(item.get(date_field, ""))
        if parsed and _matches(item, pattern, fields, kinds, both):
            out.append(((parsed[0], parsed[1]), " ".join(str(item.get(fields[0]) or "").split()[:6]),
                        item.get(date_field, "")))
    return sorted(out, key=lambda x: x[0][0])


def _events(facts: Dict[str, Any], pattern: str, kinds: Tuple[str, ...] = (), both: bool = False) -> List[tuple]:
    return sorted(_dated(facts.get("events"), pattern, ("what", "who"), kinds=kinds, both=both)
                  + _dated(facts.get("proceedings"), pattern, ("stage", "decision", "authority")),
                  key=lambda x: x[0][0])


class Term:
    """A legal term: its label, length, the article it belongs to (the institution's reference only), the
    facts it runs from and to (patterns over the dated events and proceedings; to today when nothing ends
    it), and when it applies at all."""

    def __init__(self, id: str, label: str, ref: str, start: str, end: str, years: int = 0, months: int = 0,
                 days: int = 0, applies: Optional[Callable[[Dict[str, Any]], bool]] = None, word: str = "",
                 start_items: Optional[Callable[[Dict[str, Any]], List[tuple]]] = None, since: str = "",
                 family: str = "civile", regimes_only: bool = False, start_kinds: Tuple[str, ...] = (),
                 end_kinds: Tuple[str, ...] = (), both: bool = False):
        self.id, self.label, self.ref, self.start, self.end = id, label, ref, start, end
        # The kinds of event (EVENT_KINDS) it runs from and to: a typed event is read by its kind, an untyped one by
        # the patterns; both: a typed one must match the start pattern too (a road accident among accidents).
        self.start_kinds, self.end_kinds, self.both = start_kinds, end_kinds, both
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
    """[(interval, what, since as written, until interval or None, until as written)] of the possessions with a
    date they began: an ongoing one runs to the case's present, an ended one to its "until" (one ended with no
    date it ended is left out)."""
    out = []
    for item in facts.get("possession") or []:
        start, until = parse_date(item.get("since", "")), parse_date(item.get("until", ""))
        ended = _low(item.get("still_ongoing")) in ("no", "false") or bool(until)
        if start and (not ended or until):
            out.append(((start[0], start[1]), " ".join(str(item.get("asset") or "il bene").split()[:6]),
                        item.get("since", ""), (until[0], until[1]) if until else None, item.get("until", "")))
    return out


# FACT_REGIMES_ENABLED, v48 (the PO's red on Leonardo: the ventennio counted from 2003, Sempronio's sale to Domitilla; in
# v47 from 2016, Domitilla's gate): the start of a possession for usucapione is never a sale's or a transfer's date, the
# date someone turned to a lawyer or a court, or a closure's; a possession that is itself a closure is none. No start
# left with a possession in the facts: the term says "inizio del possesso da accertare" (_undated_entry, start_unknown).
_NOT_A_START_KINDS = ("vendita", "divisione", "successione", "donazione", "diffida", "azione giudiziaria", "notifica",
                      "spoglio", "turbativa", "chiusura/ostruzione accesso")
_CONTACT_RE = r"avvocat|legale|consulen|parere|diffid|citazion|ricors|causa\b|giudizi"
_CLOSURE_USE_RE = r"chiusur|cancell|sbarr|ostru|impedi|recinz|lucchett"
START_UNKNOWN = "inizio del possesso da accertare"


def _not_start_years(facts: Dict[str, Any]) -> set:
    """The years of the facts' sales and transfers, contacts with a lawyer or a court, and closures."""
    years: set = set()
    for e in facts.get("events") or []:
        if _typed_kind(e) in _NOT_A_START_KINDS or re.search(_CONTACT_RE + "|" + _SPOGLIO, _low(e.get("what"))) \
                or re.search(r"vend|acquist|cession|donazion|eredit", _low(e.get("what"))):
            years.update(re.findall(r"\b(1[89]\d\d|20\d\d)\b", str(e.get("date") or "")))
    for h in facts.get("property_history") or []:
        for k in ("from", "to"):
            years.update(re.findall(r"\b(1[89]\d\d|20\d\d)\b", str(h.get(k) or "")))
    for c in facts.get("contracts") or []:
        years.update(re.findall(r"\b(1[89]\d\d|20\d\d)\b", str(c.get("date") or "")))
    return years


def _usucapione_starts(facts: Dict[str, Any], movable: bool = False) -> List[tuple]:
    """_possession_starts of immovables (movable: of movables); FACT_REGIMES_ENABLED (facts with "_regimes"): none
    from a sale, a contact or a closure date, none of a possession that is a closure."""
    starts = [s for s in _possession_starts(facts) if bool(re.search(_MOVABLE, _low(s[1]))) == movable]
    if not facts.get("_regimes"):
        return starts
    banned = _not_start_years(facts)
    closures = {_low(p.get("since")) for p in facts.get("possession") or [] if re.search(_CLOSURE_USE_RE, _low(p.get("use")))}
    return [s for s in starts if not (set(re.findall(r"\b(1[89]\d\d|20\d\d)\b", str(s[2]))) & banned)
            and _low(s[2]) not in closures]


_DURATION_RE = r"\b(?:da|per)\s+(?:più di |oltre |almeno |circa |ben )?(?:\d+|un|due|tre|dieci|venti|trenta|quaranta|cinquanta)\s+anni"


def _duration_stated(facts: Dict[str, Any]) -> bool:
    """A possession whose length the facts state without a date ("da più di 30 anni"): its start is not unknown."""
    return any(re.search(_DURATION_RE, _low(p.get("since")) + " " + _low(p.get("how"))) for p in facts.get("possession") or [])


def _possession_without_start(facts: Dict[str, Any], movable: bool = False) -> bool:
    """A possession in the facts, not a closure, whose start the facts do not give (or gave only as a sale, a contact or
    a closure date)."""
    return any(bool(re.search(_MOVABLE, _low(p.get("asset")))) == movable and not re.search(_CLOSURE_USE_RE, _low(p.get("use")))
               for p in facts.get("possession") or [])


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
_JUDGMENT = r"giudicat|sentenz\w*[^.]{0,40}(?:\btar\b|tribunale amministrativ|consiglio di stato|annull)|ottemperanz"
_REISSUED = (r"nuov\w* (?:provvediment|graduatori|delibera|att)|riadott|riconferm|conferm\w* (?:la|il|l['’])|"
             r"riapprov|elus\w*|in contrasto con (?:la sentenza|il giudicato)")
_TAR = r"\btar\b|ricorso al tribunale amministrativo|ricors\w* giurisdizional|ricorso amministrativo"
_DISCOVERY = r"scopr|scopert|si accorg|accort|manifest|emers|compars\w* (?:di |delle |dei )?(?:crep|infiltraz|vizi|difett)"
# A machine's fault is told by its symptoms as often as by "guasto" (the boat case, 26 Oct: alarms, the engine in
# protected mode -- no ATP offered, no 8-day denuncia term).
_DEFECT = (r"vizi|difett|guast|malfunzion|non funzion|crep|infiltraz|cedim|distacc|umidit|allarm|avari|anomal|"
           r"modalit\w* protett|in protezione|in panne")
_DENOUNCE = r"denunci|contestaz|reclam|segnal|comunica"
_DELIVERY = r"consegn|acquist|compr|ritir"
_WORKS_DONE = r"ultimat|consegn|complet|fine dei lavori|terminat|collaud"
_MOVABLE = r"\bauto|veicol|\bmoto|gioiell|quadr|orolog|bicicl|barca|opera d'arte|arred"


_WAY = r"passagg|transit|sentier|strad|viottol|varco|cancell|accesso"


def _right_of_way(facts: Dict[str, Any]) -> bool:
    """A right exercised over a way: the facts say servitude, someone uses a way, or a way was closed. A road accident
    names a road and no more."""
    if "servit" in _texts(facts, ("possession", "works_or_signs", "events")):
        return True
    if any(re.search(_WAY, f"{_low(p.get('use'))} {_low(p.get('asset'))}") for p in facts.get("possession") or []):
        return True
    closed = any(_typed_kind(e) in _OBSTRUCTIONS for e in facts.get("events") or []) or bool(re.search(_SPOGLIO, _texts(facts, ("events",))))
    return closed and bool(re.search(_WAY, _texts(facts, ("works_or_signs", "events", "access"))))


def _last_use(facts: Dict[str, Any]) -> List[tuple]:
    """The latest date the right was used: a possession's end, else the closure that stopped it."""
    items = [(*e, None, "") for e in _events(facts, _SPOGLIO, _OBSTRUCTIONS + ("uso/passaggio",))]
    for p in facts.get("possession") or []:
        parsed = parse_date(p.get("until") or "")
        if parsed:
            items.append(((parsed[0], parsed[1]), f"ultimo uso ({p.get('use') or 'passaggio'})", p.get("until"), None, ""))
    return sorted(items, key=lambda item: item[0][1], reverse=True)[:1]


TERMS: List[Term] = [
    Term("possessoria", "azione possessoria (spoglio o turbativa)", "artt. 1168 e 1170 c.c.", _SPOGLIO, _ACTION,
         years=1, word="termine annuale", since="dallo spoglio o dalla turbativa",
         start_kinds=("spoglio", "turbativa", "chiusura/ostruzione accesso"), end_kinds=("azione giudiziaria",)),
    Term("non_uso_servitu", "estinzione della servitù per non uso (non è il termine dell'usucapione)", "art. 1073 c.c.", "", _ACTION, years=20,
         word="ventennio di non uso", start_items=_last_use, applies=_right_of_way, since="dall'ultimo uso",
         end_kinds=("azione giudiziaria",)),
    Term("usucapione_ordinaria", "usucapione ordinaria di beni immobili", "art. 1158 c.c.", "", "", years=20,
         word="ventennio di possesso", start_items=lambda f: _usucapione_starts(f),
         since="di possesso continuato"),
    Term("usucapione_abbreviata", "usucapione abbreviata (solo con titolo idoneo trascritto e buona fede)",
         "art. 1159 c.c.", "", "", years=10, word="decennio di possesso",
         applies=lambda f: _title_from_non_owner(f),
         start_items=lambda f: _usucapione_starts(f),
         since="dalla trascrizione del titolo"),
    Term("usucapione_mobili", "usucapione di beni mobili (10 anni in buona fede; 20 altrimenti)", "art. 1161 c.c.",
         "", "", years=10, word="decennio di possesso",
         start_items=lambda f: _usucapione_starts(f, movable=True),
         since="di possesso continuato"),
    Term("prescrizione_circolazione", "prescrizione del risarcimento da circolazione di veicoli", "art. 2947 c.c.",
         _ROAD, _INTERRUPT, years=2, applies=lambda f: not f.get("contracts"), since="dal fatto",
         start_kinds=("incidente",), end_kinds=("diffida", "azione giudiziaria"), both=True),
    Term("prescrizione_extracontrattuale", "prescrizione del risarcimento del danno extracontrattuale",
         "art. 2947 c.c.", _DAMAGE, _INTERRUPT, years=5, applies=lambda f: not f.get("contracts"), since="dal fatto",
         start_kinds=("incidente", "lesione", "decesso", "danneggiamento"), end_kinds=("diffida", "azione giudiziaria")),
    Term("prescrizione_ordinaria", "prescrizione ordinaria (diritti da contratto)", "art. 2946 c.c.",
         r"inadempi|non (?:ha )?pagat|scadenz|ritard|mancat\w* (?:pagament|consegn)", _INTERRUPT, years=10,
         applies=lambda f: bool(f.get("contracts")), since="dal giorno in cui il diritto può essere fatto valere",
         start_kinds=("inadempimento",), end_kinds=("diffida", "azione giudiziaria")),
    Term("querela", "querela", "art. 124 c.p.", _OFFENCE, r"querel|denunci", months=3,
         since="dalla notizia del fatto che costituisce il reato", family="penale",
         start_kinds=("reato", "lesione", "danneggiamento"), end_kinds=("querela", "denuncia/reclamo")),
    Term("ricorso_tar_appalti", "ricorso al TAR in materia di appalti pubblici (rito speciale)", "art. 120 c.p.a.",
         _ACT, _TAR, days=30,
         applies=lambda f: any(re.search(_TENDER, _low(" ".join(map(str, e.values()))))
                               for e in (f.get("events") or []) + (f.get("proceedings") or [])),
         since="dalla comunicazione o conoscenza dell'atto", family="amministrativo",
         start_kinds=("notifica", "provvedimento"), end_kinds=("azione giudiziaria",)),
    Term("ricorso_tar", "ricorso al TAR per l'annullamento", "artt. 29 e 41 c.p.a.", _ACT, _TAR, days=60,
         applies=lambda f: not any(re.search(_TENDER, _low(" ".join(map(str, e.values()))))
                                   for e in (f.get("events") or []) + (f.get("proceedings") or []))
         and _has_role(f, r"pubblica amministrazione|comune|ente"),
         since="dalla notificazione, comunicazione o piena conoscenza dell'atto", family="amministrativo",
         start_kinds=("notifica", "provvedimento"), end_kinds=("azione giudiziaria",)),
    # v57 (I: v56-h10 §1 said 60 days, §5 ten years): the judgment's two terms -- the nullity of an act violating or
    # eluding it, 180 days from that act (art. 31 co. 4 c.p.a.); the ottemperanza, ten years from the judgment (art.
    # 114 co. 1 c.p.a.). The articles' commi are in the labels (a ref's numbers are its articles: term_institutions).
    Term("nullita_giudicato", "azione di nullità per violazione o elusione del giudicato (art. 31, co. 4)",
         "art. 31 c.p.a.", _REISSUED, _TAR, days=180,
         applies=lambda f: bool(re.search(_JUDGMENT, _texts(f, ("events", "proceedings"))))
         and _has_role(f, r"pubblica amministrazione|comune|ente"),
         since="dall'adozione o dalla conoscenza del provvedimento in contrasto con il giudicato", family="amministrativo",
         end_kinds=("azione giudiziaria",)),
    Term("ottemperanza", "azione di ottemperanza (art. 114, co. 1)", "art. 114 c.p.a.", _JUDGMENT,
         r"ottemperanz\w*[^.]{0,30}(?:propost|depositat|notificat)|ricors\w* (?:per|in) ottemperanz", years=10,
         applies=lambda f: bool(re.search(_JUDGMENT, _texts(f, ("events", "proceedings"))))
         and _has_role(f, r"pubblica amministrazione|comune|ente"),
         since="dal passaggio in giudicato della sentenza", family="amministrativo", end_kinds=("azione giudiziaria",)),
    Term("denuncia_vizi_vendita", "denuncia dei vizi della cosa venduta", "art. 1495 c.c.", _DISCOVERY + "|" + _DEFECT,
         _DENOUNCE, days=8, applies=lambda f: _has_contract(f, r"vendit|compravendit|acquist")
         and (not _has_role(f, "consumatore") or bool(f.get("_regimes"))), since="dalla scoperta del vizio",
         start_kinds=("scoperta vizio",), end_kinds=("denuncia/reclamo", "diffida")),
    Term("azione_vizi_vendita", "azione di garanzia per i vizi della cosa venduta", "art. 1495 c.c.", _DELIVERY,
         _ACTION, years=1, applies=lambda f: _has_contract(f, r"vendit|compravendit|acquist")
         and (not _has_role(f, "consumatore") or bool(f.get("_regimes"))), since="dalla consegna",
         start_kinds=("consegna", "vendita"), end_kinds=("azione giudiziaria",)),
    Term("denuncia_conformita_consumo", "denuncia del difetto di conformità (consumatore)", "Codice del Consumo",
         _DISCOVERY + "|" + _DEFECT, _DENOUNCE, months=2,
         applies=lambda f: _has_role(f, "consumatore") and _has_contract(f, r"vendit|compravendit|acquist"),
         since="dalla scoperta del difetto", start_kinds=("scoperta vizio",), end_kinds=("denuncia/reclamo", "diffida")),
    Term("responsabilita_conformita_consumo", "responsabilità del venditore per il difetto di conformità "
         "(consumatore)", "Codice del Consumo", _DELIVERY, _DISCOVERY + "|" + _DEFECT,
         years=2, word="manifestazione del difetto entro 2 anni", applies=lambda f: _has_role(f, "consumatore") and _has_contract(f, r"vendit|compravendit|acquist"),
         since="dalla consegna del bene", regimes_only=True, start_kinds=("consegna", "vendita"),
         end_kinds=("scoperta vizio",)),
    Term("azione_conformita_consumo", "prescrizione dell'azione per il difetto di conformità non dolosamente "
         "occultato (consumatore)", "Codice del Consumo", _DELIVERY, _ACTION, months=26,
         applies=lambda f: _has_role(f, "consumatore") and _has_contract(f, r"vendit|compravendit|acquist"),
         since="dalla consegna del bene", regimes_only=True, start_kinds=("consegna", "vendita"), end_kinds=("azione giudiziaria",)),
    Term("denuncia_difformita_appalto", "denuncia delle difformità e dei vizi dell'opera (appalto)", "art. 1667 c.c.",
         _DISCOVERY + "|" + _DEFECT, _DENOUNCE, days=60,
         applies=lambda f: _has_contract(f, r"appalt|lavori|ristruttur|costruz"), since="dalla scoperta",
         start_kinds=("scoperta vizio",), end_kinds=("denuncia/reclamo", "diffida")),
    Term("azione_garanzia_appalto", "azione di garanzia nell'appalto", "art. 1667 c.c.", _WORKS_DONE, _ACTION, years=2,
         applies=lambda f: _has_contract(f, r"appalt|lavori|ristruttur|costruz"), since="dalla consegna dell'opera",
         start_kinds=("fine lavori", "consegna"), end_kinds=("azione giudiziaria",)),
    Term("manifestazione_gravi_difetti", "manifestazione dei gravi difetti entro dieci anni dal compimento dell'opera",
         "art. 1669 c.c.", _WORKS_DONE, _DISCOVERY + "|" + _DEFECT, years=10, word="decennio dal compimento",
         applies=lambda f: _has_contract(f, r"appalt|lavori|ristruttur|costruz"), since="dal compimento dell'opera",
         start_kinds=("fine lavori", "consegna"), end_kinds=("scoperta vizio",)),
    Term("denuncia_gravi_difetti", "denuncia dei gravi difetti dell'immobile", "art. 1669 c.c.",
         _DISCOVERY + "|" + _DEFECT, _DENOUNCE, years=1,
         applies=lambda f: _has_contract(f, r"appalt|lavori|ristruttur|costruz"), since="dalla scoperta",
         start_kinds=("scoperta vizio",), end_kinds=("denuncia/reclamo", "diffida")),
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
    return [t["entry"] for t in explain_terms(facts, areas) if t["entry"]]


def explain_terms(facts: Optional[Dict[str, Any]], areas: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """For every term (TERMS, then the criminal appeal): {"id", "entry" (compute_terms' item, or None), "why"
    (each step in words: applies or not, the facts it runs from and to and how they were matched)}
    -- scripts/fact_probe.py prints it."""
    if not facts:
        return []
    undated = bool(facts.get("_regimes"))
    families = {family(a) for a in areas or []} - {""}
    out = [_term_trace(term, facts, families, undated) for term in TERMS]
    out.append(_appeal_trace(facts))
    return out


def _how(term: Term, facts: Dict[str, Any], kinds: Tuple[str, ...], pattern: str) -> str:
    typed = any(_typed_kind(e) for e in facts.get("events") or [])
    return (f"eventi del tipo {' / '.join(kinds)}" + (" (e per parole: «" + pattern + "»)" if term.both else "")
            + ("; eventi senza tipo per parole" if pattern else "")) if kinds and typed else f"per parole: «{pattern}»"


def _term_trace(term: Term, facts: Dict[str, Any], families: set, undated: bool) -> Dict[str, Any]:
    why: List[str] = []
    trace = {"id": term.id, "entry": None, "why": why}
    if term.regimes_only and not undated:
        why.append("solo con FACT_REGIMES_ENABLED: non calcolato")
        return trace
    if term.applies and not term.applies(facts):
        why.append("non si applica ai fatti (contratto, ruoli o materia richiesti assenti)")
        return trace
    if term.start_items:
        starts = term.start_items(facts)
        why.append(f"decorre dal possesso: {len(starts)} possessi con data di inizio"
                   + "".join(f"; «{x[1]}» dal {x[2]}" + (f" al {x[4]}" if x[3] else "") for x in starts))
    else:
        starts = _events(facts, term.start, term.start_kinds, term.both)
        why.append(f"inizio ({_how(term, facts, term.start_kinds, term.start)}): "
                   + ("; ".join(f"«{x[1]}» {x[2]}" for x in starts) or "nessun fatto datato"))
    if not starts:
        possessed = term.id.startswith("usucapione") and _possession_without_start(facts, term.id == "usucapione_mobili")
        if undated and (not families or term.family in families) and (_mentioned(facts, term) or possessed):
            unknown = bool(term.start_items) and not _duration_stated(facts)
            trace["entry"] = _undated_entry(term, start_unknown=unknown)
            why.append("fatto citato senza data: DA VERIFICARE" + (f" ({START_UNKNOWN})" if unknown else ""))
        elif undated and families and term.family not in families:
            why.append(f"materia {term.family} fuori dalle aree del caso")
        return trace
    start = starts[0]
    if len(start) > 3 and start[3]:
        status = term_status(start[0], start[3], term.years, term.months, term.days)
        end_what, end_written = "fine del possesso", start[4]
        why.append(f"fine: il possesso cessa {start[4]}")
    else:
        ends = _events(facts, term.end, term.end_kinds) if (term.end or term.end_kinds) else []
        end = _when([e for e in ends if e[1] != start[1] or e[0] != start[0]], start[0][0])
        if end:
            status, end_what, end_written = term_status(start[0], end[0], term.years, term.months, term.days), \
                end[1], end[2]
            why.append(f"fine ({_how(term, facts, term.end_kinds, term.end)}): «{end[1]}» {end[2]}")
        else:
            status, end_what, end_written = _status_to_now(facts, start[0], term.years, term.months, term.days)
            why.append(f"nessun atto che lo interrompa: fino a {end_what} {end_written}".rstrip())
    trace["entry"] = _term_entry(term.id, term.label, term.ref, term.word or f"termine di {term.length()}",
                                 term.length(), status, start, end_what, end_written)
    if term.id == "non_uso_servitu":
        expiry = start[0][1].year + term.years
        if feature_flags.enabled(TERM_EXPIRY_TEXT_FLAG):
            # the year said as the term's end, never as its start ("decorre nel 2036", open item 2)
            when = {"NON DECORSO": f"non decorso, scade nel {expiry}",
                    "DECORSO": f"già decorso, scaduto nel {expiry}"}.get(status, f"scade nel {expiry}")
            text = (f"estinzione per non uso: {when} ({term.years} anni dall'ultimo uso, {_short(start[2])}); "
                    "non è il termine dell'usucapione")
        else:
            text = (f"estinzione per non uso: {expiry} ({term.years} anni dall'ultimo uso, {_short(start[2])})"
                    + (", già decorso" if status == "DECORSO" else "") + "; non è il termine dell'usucapione")
        trace["entry"].update(expires=str(expiry), text=text)
    why.append(f"stato: {status}")
    return trace


def _appeal_trace(facts: Dict[str, Any]) -> Dict[str, Any]:
    why: List[str] = []
    trace = {"id": "impugnazione_penale", "entry": None, "why": why}
    sentence = _events(facts, _SENTENCE, ("sentenza",))
    if not sentence:
        why.append("nessuna sentenza datata")
        return trace
    start = sentence[-1]
    deposit = _events(facts, r"deposit\w* (?:delle |della )?motivazion")
    appeal = _when(_events(facts, _APPEAL, ("impugnazione",)), start[0][0])
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
    trace["entry"] = _term_entry("impugnazione_penale", "impugnazione della sentenza penale", _APPEAL_REF,
                                 f"termine di {length}", length, status, base, end_what, end_written,
                                 note="" if deposit else "dipende da quando è depositata la motivazione")
    why += [f"sentenza: «{start[1]}» {start[2]}", f"stato: {status}"]
    return trace


def _mentioned(facts: Dict[str, Any], term: Term) -> bool:
    """The case states, without a usable date, the fact a term runs from."""
    if term.start_items:
        movable = term.id == "usucapione_mobili"
        return any(bool(re.search(_MOVABLE, _low(p.get("asset")))) == movable
                   for p in facts.get("possession") or [] if _low(p.get("still_ongoing")) not in ("no", "false"))
    if term.start_kinds and any(_typed_kind(e) in term.start_kinds for e in facts.get("events") or []):
        return True
    return bool(term.start) and bool(re.search(term.start, _texts(facts, ("events", "proceedings", "contracts",
                                                                             "works_or_signs", "injuries", "risks"))))


def _undated_entry(term: Term, start_unknown: bool = False) -> Dict[str, Any]:
    word = term.word or f"termine di {term.length()}"
    if start_unknown:          # a possession's term: its start is what the case leaves open
        text = (f"{term.label}: {word} — DA VERIFICARE, {START_UNKNOWN}: il caso non indica quando il possesso è "
                "iniziato (non lo è una vendita, un contatto con l'avvocato o una chiusura), il termine non può dirsi "
                "maturato")
    else:
        text = (f"{term.label}: {word} {term.since}".rstrip() + " — DA VERIFICARE: il caso non indica le date da cui "
                "decorre")
    return {"id": term.id, "label": term.label, "ref": term.ref, "length": term.length(), "status": "DA VERIFICARE",
            "from": {"what": START_UNKNOWN if start_unknown else term.since, "date": ""}, "to": {"what": "", "date": ""},
            "text": text, "undated": True, **({"start_unknown": True} if start_unknown else {})}


def _term_entry(id, label, ref, word, length, status, start, end_what, end_written, note=""):
    text = (f"{label}: {word} {status} ({start[1] or 'fatto'} {_short(start[2])}, {end_what} "
            f"{end_written})".replace(" )", ")") + (f"; {note}" if note and status == "INCERTO" else ""))
    return {"id": id, "label": label, "ref": ref, "length": length, "status": status,
            "from": {"what": start[1], "date": start[2]}, "to": {"what": end_what, "date": end_written}, "text": text}


# The real actions: no prescription, whatever the dates. An entry of the same shape, with no length.
NO_PRESCRIPTION_STATUS = "non soggetta a prescrizione"
NO_PRESCRIPTION: Tuple[Tuple[str, str, str], ...] = (
    ("azione_confessoria", "azione confessoria", "art. 1079 c.c."),
    ("azione_negatoria", "azione negatoria", "art. 949 c.c."),
    ("azione_rivendicazione", "azione di rivendicazione", "art. 948 c.c."),
)
_REAL_ACTION = r"confessori|negatori|rivendic|servit"


def _non_use_for_a_servitude(facts: Optional[Dict[str, Any]], terms: List[Dict[str, Any]], fired: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """A fired servitude rule settles that there is a right to lose, whatever words the facts use for the way."""
    if not facts or not _claims_servitude(fired) or any(t["id"] == "non_uso_servitu" for t in terms):
        return []
    term = copy.copy(next(t for t in TERMS if t.id == "non_uso_servitu"))
    term.applies = None
    entry = _term_trace(term, facts, set(), True)["entry"]
    return [entry] if entry else []


def _claims_servitude(fired: List[Dict[str, Any]]) -> bool:
    return any(r.get("acquired_right") and "servit" in " ".join(r.get("institutions") or []).lower() for r in fired)


def no_prescription_terms(facts: Optional[Dict[str, Any]], fired: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The real actions' entries, when a rule on real rights fired or the facts speak of one. Who claims a servitude has the
    confessoria alone: the negatoria and the rivendicazione are the owner's, against a right someone else claims."""
    if not facts or not (any(r.get("area") == "diritti_reali" for r in fired) or re.search(_REAL_ACTION, _texts(facts))):
        return []
    wanted = ("azione_confessoria",) if _claims_servitude(fired) else ("azione_negatoria", "azione_rivendicazione")
    if re.search("confessori", _texts(facts)):
        wanted += ("azione_confessoria",)
    return [{"id": id, "label": label, "ref": ref, "length": "", "status": NO_PRESCRIPTION_STATUS, "from": {"what": "", "date": ""},
             "to": {"what": "", "date": ""}, "text": f"{label}: {NO_PRESCRIPTION_STATUS}"} for id, label, ref in NO_PRESCRIPTION
            if id in wanted]


# v60h (3): the fact that triggers each term -- a sentence of the case (or of the client's precisazioni) must state it,
# not negated («il telefono non è mai arrivato» delivers no good), or the term is neither computed nor rendered
# (signals_from(case=...); logged as terms_untriggered). {term id: (pattern, the fact in words)}.
_DELIVERED = (r"consegnat|consegn\w* (?:il|la|lo|l'|i|gli|le|del|della|dell')|ricevut|arrivat|recapitat|ritirat|"
              r"installat|montat|in (?:mio|suo|loro|nostro) possesso", "un bene consegnato",
              r"\b(?:e-?mail|mail|sms|messagg|whatsapp|telefonat|chiamat|link|bonific|pagament|codic|password|"
              r"credenzial|notific|comunicazion|lettera|raccomandat|pec|denaro|soldi|somm[ae]|rimbors)\w*")
_WORKS_DONE_FACT = (r"ultimat|consegnat|completat|fine dei lavori|lavori (?:finit|terminat|conclus)|terminat|collaud|"
                    r"realizzat|eseguit", "un'opera compiuta o consegnata")
_NOTIFIED_ACT = (r"(?:notific|comunicat|pubblicat|ricevut)\w*[^.;]{0,80}(?:provvediment|dinieg|ordinanz|decret|"
                 r"delibera|revoca|annullament|esclusion|aggiudicaz|graduatori|atto)|(?:provvediment|dinieg|ordinanz|decret|"
                 r"delibera|revoca|annullament|esclusion|aggiudicaz|graduatori)\w*[^.;]{0,80}(?:notific|comunicat|"
                 r"pubblicat|ricevut)", "un atto notificato o comunicato")
_JUDGMENT_FACT = (r"sentenz|giudicat", "una sentenza")
# v60h: possession is also the use of a way or a thing over time -- «usa», «utilizza», «passa», «transita», «da anni»,
# «da sempre», «da tempo» -- not only the word «possesso»
_POSSESSION_FACT = (r"possed|possess|usucap|coltiv|utilizz|occup|abitat|abit[ao]|godut|godiment|recint|"
                    r"\bus(?:a|ano|ava|avano|ato|ando)\b|\bpass(?:a|ano|ava|avano|ando)\b|\btransit\w*|"
                    r"\bda (?:molti |tanti |diversi |numerosi |oltre |piu di )?(?:anni|decenni)\b|\bda sempre\b|"
                    r"\bda (?:molto |lungo )?tempo\b", "il possesso del bene (anche l'uso di un passaggio o di un bene nel "
                    "tempo)")
_REAL_RIGHT_FACT = (r"servit|passagg|propriet|fondo|fondi|terren|immobil|casa|appartament|confin",
                    "un diritto reale su un bene")
TERM_TRIGGERS: Dict[str, Tuple[str, ...]] = {        # (pattern, the fact in words[, a sentence that is not it])
    "possessoria": (r"spogli|turbativ|molest|sbarr|chius|cancell|lucchett|recinz|ostru|impedi|occupat",
                    "lo spoglio o la turbativa del possesso"),
    "non_uso_servitu": (r"servit|passagg|transit|attravers", "l'esercizio di una servitù o di un passaggio"),
    "usucapione_ordinaria": _POSSESSION_FACT, "usucapione_abbreviata": _POSSESSION_FACT,
    "usucapione_mobili": _POSSESSION_FACT,
    "prescrizione_circolazione": (r"stradal|tamponament|investit|circolazion|(?:auto|moto|veicol|scooter|ciclomotor|camion)"
                                  r"\w*[^.;]{0,60}(?:urt|scontr|collision|invest|tampon|incident|sinistr)",
                                  "un sinistro stradale"),
    "prescrizione_extracontrattuale": (r"incident|sinistr|cadut|cadd|lesion|ferit|infortun|danneggi|investit|crollat|"
                                       r"tamponament|\bdann[oi]\b", "un fatto dannoso"),
    "prescrizione_ordinaria": (r"contratt|accordo|incaric|ordinat|ordine|fattur|prestit|mutuo|vend|acquist|comprat|"
                               r"appalt|locaz|affitt|assunt|rapporto di lavoro|dipendent|abbonament",
                               "un contratto"),
    "querela": (r"aggred|percoss|picchi|lesion|ferit|minacc|ingiuri|diffam|truff|raggir|frod|furt|rubat|sottratt|"
                r"danneggi|molest|perseguit|violenz|infortun|insult|ingann|sedicent|spacciat|phishing|clonat|"
                r"addebit\w* non autorizzat|fint[oa] (?:venditor|operator|funzionari|impiegat|sito)", "un fatto di reato"),
    "ricorso_tar_appalti": _NOTIFIED_ACT, "ricorso_tar": _NOTIFIED_ACT,
    "nullita_giudicato": _JUDGMENT_FACT, "ottemperanza": _JUDGMENT_FACT,
    "denuncia_vizi_vendita": _DELIVERED, "azione_vizi_vendita": _DELIVERED,
    "denuncia_conformita_consumo": _DELIVERED, "responsabilita_conformita_consumo": _DELIVERED,
    "azione_conformita_consumo": _DELIVERED,
    "denuncia_difformita_appalto": _WORKS_DONE_FACT, "azione_garanzia_appalto": _WORKS_DONE_FACT,
    "manifestazione_gravi_difetti": _WORKS_DONE_FACT, "denuncia_gravi_difetti": _WORKS_DONE_FACT,
    "impugnazione_penale": (r"sentenz|condann|assolt|assoluz", "una sentenza pronunciata"),
    "azione_confessoria": _REAL_RIGHT_FACT, "azione_negatoria": _REAL_RIGHT_FACT,
    "azione_rivendicazione": _REAL_RIGHT_FACT,
}
_NEGATED_BEFORE_RE = re.compile(r"\b(?:non|mai|nessun[oa]?|senza|né)\b[^.;,]{0,24}$", re.IGNORECASE)


def case_sentences(case: str) -> List[str]:
    return [" ".join(x.split()) for x in re.split(r"(?<=[.;!?])\s+|\n+", case or "") if x.strip()]


def trigger_sentence(term_id: str, case: str) -> Optional[str]:
    """The case sentence stating the term's triggering fact (TERM_TRIGGERS), not negated; "" when none; None for a term
    with no trigger rule."""
    rule = TERM_TRIGGERS.get(str(term_id or ""))
    if rule is None:
        return None
    pattern = re.compile(rule[0], re.IGNORECASE)
    exclude = re.compile(rule[2], re.IGNORECASE) if len(rule) > 2 else None
    for sentence in case_sentences(case):
        low = relative_text(sentence)
        if exclude is not None and exclude.search(low):
            continue
        for m in pattern.finditer(low):
            if not _NEGATED_BEFORE_RE.search(low[:m.start()]):
                return sentence[:200]
    return ""


def untriggered(terms: List[Dict[str, Any]], case: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """(the terms whose triggering fact a case sentence states, or with no trigger rule; the others as {id, fact})."""
    kept, dropped = [], []
    for t in terms or []:
        found = trigger_sentence(t.get("id"), case)
        if found is None or found:
            kept.append({**t, "trigger": found} if found else t)
        else:
            dropped.append({"id": str(t.get("id")), "fact": TERM_TRIGGERS[str(t.get("id"))][1]})
    return kept, dropped


# Art. 5, comma 1, d.lgs. 28/2010: the subjects where mediation is a condition of the action. (name, the word that
# finds it in the article's text, the rule areas or ids that make it the case's subject, words in the contracts or
# requests.) With the article's text in the corpus only the subjects it holds count.
MEDIATION_REF = "art. 5 d.lgs. 28/2010"
_MEDIATION_SUBJECTS: Tuple[Tuple[str, str, Tuple[str, ...], str], ...] = (
    ("diritti reali", "diritti reali", ("diritti_reali",), r"servit|usucapi|rivendic"),
    ("condominio", "condominio", ("condominio_parti_comuni",), r"condomin"),
    ("divisione", "divisione", (), r"scioglimento della comunione|domanda di divisione"),
    ("successioni ereditarie", "successioni", ("successioni",), r"successione ereditaria|eredit"),
    ("patti di famiglia", "patti di famiglia", (), r"patto di famiglia"),
    ("locazione", "locazione", ("locazione",), r"locazion"),
    ("comodato", "comodato", (), r"comodato"),
    ("affitto di aziende", "affitto di aziende", (), r"affitto d\w* aziend"),
    ("responsabilità medica e sanitaria", "responsabilità medica", ("responsabilita_sanitaria_civile",), r"responsabilità (?:medica|sanitaria)"),
    ("contratti assicurativi, bancari e finanziari", "contratti assicurativi", (), r"assicurativ|bancari|finanziari|polizza|mutuo"),
    ("associazione in partecipazione", "associazione in partecipazione", (), r"associazione in partecipazione"),
    ("consorzio", "consorzio", (), r"consorzio"),
    ("franchising", "franchising", (), r"franchising"),
    ("contratto d'opera", "opera", (), r"contratto d'opera"),
    ("contratto di rete", "rete", (), r"contratto di rete"),
    ("somministrazione", "somministrazione", (), r"somministrazione"),
    ("società di persone", "società di persone", (), r"società di persone|\bs\.?n\.?c\b|\bs\.?a\.?s\b"),
    ("subfornitura", "subfornitura", (), r"subfornitura"),
)


def mediation(facts: Optional[Dict[str, Any]], fired: List[Dict[str, Any]], article: str = "") -> Optional[Dict[str, Any]]:
    """{"required", "subject", "text"}: is mediation a condition of the action here? None with no facts."""
    if not facts:
        return None
    marks = {str(r.get("area")) for r in fired} | {str(r.get("id")) for r in fired}
    text, low = _texts(facts, ("contracts", "requests")), _low(article)
    for name, word, rules, pattern in _MEDIATION_SUBJECTS:
        if (marks & set(rules) or re.search(pattern, text)) and (not low or word in low):
            return {"required": True, "subject": name,
                    "text": f"Mediazione obbligatoria ({MEDIATION_REF}): la materia è {name}, il suo esperimento è condizione "
                            "di procedibilità della domanda."}
    return {"required": False, "subject": "", "text": ""}


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


def _held_for(facts, since: str, years: int, until: str = "") -> bool:
    """A state that began at since lasted at least years: by its end (until) when written, else by the case's
    present (_status_to_now)."""
    parsed, end = parse_date(since), parse_date(until)
    if not parsed:
        return False
    if end:
        return term_status((parsed[0], parsed[1]), (end[0], end[1]), years) == "DECORSO"
    return _status_to_now(facts, (parsed[0], parsed[1]), years)[0] == "DECORSO"


def _p_possession(facts, terms, arg):
    for p in facts.get("possession") or []:
        how = _low(p.get("how"))
        if arg.get("how") and not any(h in how for h in arg["how"]):
            continue
        if arg.get("how_not") and any(h in how for h in arg["how_not"]):
            continue
        if arg.get("min_years") and not _held_for(facts, p.get("since", ""), arg["min_years"], p.get("until", "")):
            continue
        return True
    return False


def _p_work(facts, terms, arg):
    for w in facts.get("works_or_signs") or []:
        if arg.get("visible") and not _yes(w.get("visible")):
            continue
        if arg.get("before_separation") and not _yes(w.get("before_separation")):
            continue
        if arg.get("pattern") and not re.search(arg["pattern"], _low(w.get("what"))):
            continue
        if arg.get("min_years") and not _held_for(facts, w.get("existing_since", ""), arg["min_years"]):
            continue
        return True
    return False


# What ended an ownership, or began the next, when the parts end with different owners.
_SEPARATING = frozenset(("divisione", "vendita", "compravendita", "alienazione", "donazione", "successione"))


def _stated_separation(history: List[Dict[str, Any]]) -> bool:
    """The model states a transfer and the asset ends with two different owners, the assets too generic for the sequence
    to link ("fondo"), though not two plots each named as its own. An owner who passed it on is not where it ended."""
    for i, entry in enumerate(history):
        owner, words = _low(entry.get("owner")), _asset_words(entry.get("asset"))
        later = [h for h in history[i + 1:] if _low(h.get("owner")) not in ("", owner)
                 and (not words or not _asset_words(h.get("asset")) or words & _asset_words(h.get("asset")))]
        stated = _low(entry.get("ended_by")) in _SEPARATING or any(_low(h.get("how_acquired")) in _SEPARATING for h in later)
        ended_with = {_low(h.get("owner")) for h in later if _low(h.get("ended_by")) not in _SEPARATING}
        if owner and stated and len(ended_with) >= 2:
            return True
    return False


def _p_common_owner(facts, terms, arg):
    history = facts.get("property_history") or []
    if facts.get("split_into_parts") or splits(history) or _stated_separation(history):
        return True
    by_owner: Dict[str, set] = {}
    for p in facts.get("property_history") or []:
        owner = _low(p.get("owner"))
        # Typed: the owner held another asset with it, or the ownership ended by a division.
        if owner and (_low(p.get("together_with")) or _low(p.get("ended_by")) == "divisione"):
            return True
        if owner:
            by_owner.setdefault(owner, set()).add(_low(p.get("asset")))
        if owner and re.search(r"divisi|frazionat|separat|lottizz", _low(p.get("how_acquired")) + " " + _low(p.get("asset"))):
            return True
    return any(len(assets) >= 2 for assets in by_owner.values())


_LANDLOCKED_WORDS = (r"intercluso|senza (?:alcun )?accesso|non ha (?:alcun )?accesso|nessun accesso|privo di accesso|"
                     r"accesso alla (?:via|strada) pubblica")


def _p_landlocked(facts, terms, arg):
    """A plot the text calls landlocked: access.landlocked when the facts say how a plot is reached (typed; the
    validation sets it from an explicit statement, _stated_landlocked); by the facts' words only when they say
    nothing of access."""
    access = facts.get("access") or []
    if access:
        return any(_yes(a.get("landlocked")) for a in access)
    return _p_fact(facts, terms, {"pattern": _LANDLOCKED_WORDS})


def _p_event(facts, terms, arg):
    """An event of one of the kinds (EVENT_KINDS) -- typed, never by its words."""
    kinds = arg["kind"] if isinstance(arg.get("kind"), list) else [arg.get("kind")]
    return any(_typed_kind(e) in kinds for e in facts.get("events") or [])


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
             + (" già esistente quando i beni furono divisi o venduti" if a.get("before_separation") else "")
             + (f" esistente da almeno {a['min_years']} anni" if a.get("min_years") else "")),
    "landlocked": (_p_landlocked, lambda a: a.get("label") or "un fondo senza accesso alla via pubblica"),
    "event": (_p_event, lambda a: a.get("label") or "un evento del tipo "
              + " o ".join(a["kind"] if isinstance(a.get("kind"), list) else [str(a.get("kind"))])),
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
        if not rule.get("ranges") and _institutions(rule):
            raise ValueError(f"rule {rule['id']!r}: no ranges (the article numbers its institutions may resolve to)")
        for lo_hi in rule["ranges"]:
            if not (isinstance(lo_hi, list) and len(lo_hi) == 2 and all(isinstance(n, int) for n in lo_hi)):
                raise ValueError(f"rule {rule['id']!r}: a range is [first, last] article number")
        for ref in rule.get("requires") or []:
            if not re.fullmatch(r"\d+(?:-[a-z]+)? \S.*", str(ref)):
                raise ValueError(f"rule {rule['id']!r}: requires {ref!r}: \"N code\", as \"1079 c.c.\"")
        for tool in rule.get("tools") or []:
            if tool.get("id") not in {t["id"] for t in PROCEDURAL_TOOLS} or not tool.get("why"):
                raise ValueError(f"rule {rule['id']!r}: a tool is {{id of PROCEDURAL_TOOLS, why}}: {tool!r}")
        if rule.get("conclusion") and not rule.get("conclusion_subject"):
            raise ValueError(f"rule {rule['id']!r}: a conclusion needs its conclusion_subject")
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
            _log_exception("fire_rules", exc)
            logger.warning("fact signals: rule %s failed: %s", rule.get("id"), exc)
            ok = False
        if ok:
            fired.append({"id": rule["id"], "area": rule["area"], "institutions": _institutions(rule),
                          "codice": rule["codice"], "ranges": rule.get("ranges") or [], "why": rule["why"],
                          "status": rule["status"], "acquired_right": bool(rule.get("acquired_right")),
                          "requires": [tuple(r.split(" ", 1)) for r in rule.get("requires") or []],
                          "conclusion": str(rule.get("conclusion") or "").replace("{altro_contraente}", other_contractor(facts)),
                          "conclusion_subject": rule.get("conclusion_subject") or "",
                          "conclusion_unless": rule.get("conclusion_unless") or "", "tools": rule.get("tools") or [],
                          "excludes": [tuple(r.split(" ", 1)) for r in rule.get("conclusion_excludes") or []],
                          "matched": why_fired(rule, facts, terms)})
    return fired


def _clients(facts: Dict[str, Any]) -> set:
    names = {_low(p.get("name")) for p in facts.get("parties") or [] if "cliente" in _low(p.get("role"))}
    side = _low(facts.get("client_side"))
    return {n for n in names if n} | ({side} if side else set())


def _is_client(name: str, clients: set) -> bool:
    low = _low(name)
    return any(low and (low in c or c in low) for c in clients)


def _the_split(facts: Dict[str, Any]) -> Dict[str, Any]:
    """The split that made the land what it is: the earliest of the sequence, never a later sale of one of its parts."""
    return next(iter(facts.get("split_into_parts") or []), {})


def other_contractor(facts: Dict[str, Any]) -> str:
    """Who art. 1054 c.c. calls "l'altro contraente", from the property sequence: in the split that caused the interclusion,
    the seller of a sale, the sharers of a division other than the client's own part."""
    split, clients = _the_split(facts), _clients(facts)
    owner, history = str(split.get("owner") or ""), facts.get("property_history") or []
    if not owner:
        return "l'altro contraente è chi ha alienato o condiviso il fondo, da individuare dai titoli"
    entry = next((h for h in history if _low(h.get("owner")) == _low(owner)), {})
    if _low(entry.get("ended_by")) == "compravendita":
        return f"l'altro contraente è {owner}, che ha alienato il fondo poi frazionato"
    mine = [_asset_words(h.get("asset")) for h in history if _is_client(str(h.get("owner") or ""), clients)]
    held = {_low(h.get("owner")) for h in history if _asset_words(h.get("asset")) in mine}      # the client's own part, by whoever held it
    others = [n.strip() for n in str(split.get("parts") or "").split(",") if n.strip() and not _is_client(n, clients) and _low(n) not in held]
    who = f"l'altro contraente è {', '.join(others)}, condividente" if others else "l'altro contraente è chi ha avuto l'altra parte"
    return f"{who} nella divisione del fondo di {owner}"


_OBSTRUCTIONS = ("spoglio", "turbativa", "chiusura/ostruzione accesso")


def _names(text: Any) -> List[str]:
    """The proper names in a field ("Domitilla e Tizio"), none for "il vicino"."""
    return [n.strip() for n in re.split(r",|;|\be\b", str(text or "")) if re.match(r"\s*[A-ZÀ-Ü]", n)]


def parties_to_sue(facts: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    """[{name, why}], from the facts alone: who did the act obstructing the right, then every present owner of the land
    the contested work crosses (the work's own field, else the owners named where the work, the possession or the access
    are described). Never the client. [] with no such act."""
    if not facts:
        return []
    clients = _clients(facts)
    out: Dict[str, str] = {}
    acts = [e for e in facts.get("events") or [] if _typed_kind(e) in _OBSTRUCTIONS]
    for event in acts:
        for name in _names(event.get("who")):
            out.setdefault(name, f"autore dell'atto: {_clean_text(event.get('what'))}")
    if not acts:
        return []
    history = facts.get("property_history") or []
    crossed = " ".join(str(w.get("crosses") or "") for w in facts.get("works_or_signs") or [])
    around = " ".join(str(x.get(k) or "") for key, keys in (("works_or_signs", ("what", "crosses")), ("possession", ("asset",)),
                                                            ("access", ("how_reached",))) for x in facts.get(key) or [] for k in keys)
    words = _asset_words(crossed or around)
    for i, entry in enumerate(history):
        owner = str(entry.get("owner") or "").strip()
        if not owner or _parts_of(entry, history[i + 1:])[0]:
            continue                                # not its present owner: the land passed on
        if words & _asset_words(entry.get("asset")) or _named_in(_low(owner), _low(crossed or around)):
            out.setdefault(owner, f"proprietario del fondo attraversato ({entry.get('asset') or 'fondo'})")
    return [{"name": n, "why": w} for n, w in out.items() if not _is_client(n, clients)]


def parties_block(signals: Optional[Dict[str, Any]]) -> str:
    return "\n".join(f"- {p['name']}: {p['why']}" for p in (signals or {}).get("defendants") or [])


# PARTIES_ENFORCED (v45): with more than one party to sue, the conclusions block names them all, with what parties_to_sue
# found for each -- no conclusion of its own beyond that.
PARTIES_ENFORCED_FLAG = "PARTIES_ENFORCED"


REAL_ACTION_REGIME = "azione_reale_immobiliare"


_PREVIOUS_ROLE = r"precedent|originari|dante causa|\bex\b|già proprietari|antecedent|avente causa"
_PAST_OWNERSHIP = r"appartenev|apparteneva|era di propriet|erano di propriet|un tempo|in origine|originariamente"


def _items_naming(facts: Dict[str, Any], name: str, keys: Tuple[str, ...]) -> List[str]:
    """Each item of facts' keys naming name, its values joined, lower case."""
    low = _low(name)
    out = []
    for key in keys:
        items = facts.get(key)
        if not isinstance(items, list):          # a flag or a string of the facts ("_regimes": true), not a list of facts
            continue
        for item in items:
            text = " ".join(_low(v) for k, v in item.items() if not k.endswith("_iso") and v) if isinstance(item, dict) \
                else _low(item)
            if _named_in(low, text):
                out.append(text)
    return out


_TRANSFER_VERBS = r"(?:ha\s+)?(?:vend\w*|ced\w*|don\w*|alien\w*|trasferi\w*)"


def _transferred(facts: Dict[str, Any], name: str, history: List[Dict[str, Any]]) -> bool:
    """name's sale or transfer is in the facts (Leonardo v52: "Sempronio vende il terreno a Domitilla", 2003, with no
    property_history entry of Sempronio's own): an event with name as the subject of a transfer verb, or another owner's
    property_history entry for "<asset> di <name>" acquired by sale, gift, division or succession."""
    low = _low(name)
    if not low:
        return False
    subject = re.compile(rf"(?:^|[\s,;(])(?:il\s+|la\s+|i\s+|gli\s+)?{re.escape(low)}\s+{_TRANSFER_VERBS}\b")
    for event in facts.get("events") or []:
        if isinstance(event, dict) and subject.search(_low(event.get("what"))):
            return True
    for entry in history:
        if _low(entry.get("owner")) in ("", low) or _low(entry.get("how_acquired")) not in _SEPARATING | {"compravendita"}:
            continue
        whose = re.search(r"\b(?:di|del|della|dei|degli)\s+(.+)$", _low(entry.get("asset")))
        if whose and _named_in(low, whose.group(1)) and not re.search(r"\be\b|,", whose.group(1)):
            return True
    return False


def _previous_owner(facts: Dict[str, Any], name: str, history: List[Dict[str, Any]]) -> bool:
    """name is in the facts only as a previous owner in the chain of title: every property_history entry of its own
    passed on (a later owner holds the asset or a part of it, or the entry ended), none it still holds; or, holding no
    entry it has not passed on, its sale or transfer is in the facts (_transferred: "Sempronio vende il terreno a
    Domitilla"); or its role, or every fact naming it, puts its ownership in the past ("appartenevano a Mevio")."""
    low = _low(name)
    own = [(i, e) for i, e in enumerate(history) if _low(e.get("owner")) == low]
    others = {_low(e.get("owner")) for e in history if _low(e.get("owner")) not in ("", low)}

    def passed_on(i: int, entry: Dict[str, Any]) -> bool:
        ended = _low(entry.get("ended_by")) and _low(entry.get("ended_by_how")) != SEQUENCE_DEDUCED
        # its asset is another owner's land by name ("terreni di Leonardo e Sempronio", Mevio's), wherever in the
        # history -- not land described by its neighbour ("terreno tra due fondi di Leonardo", Annibale's)
        whose = re.match(r"(?:(?:il|i|la|le|lo|gli|l')\s*)?[\wà-ù]+\s+(?:di|del|della|dei|degli)\s+(.+)$", _low(entry.get("asset")))
        named = bool(whose) and any(_named_in(o, whose.group(1)) for o in others)
        return bool(_parts_of(entry, history[i + 1:])[0] or ended or _low(entry.get("to")) or named)
    if own and all(passed_on(i, e) for i, e in own):
        return True
    if (not own or all(passed_on(i, e) for i, e in own)) and _transferred(facts, name, history):
        return True
    roles = " ".join(_low(p.get("role")) for p in facts.get("parties") or [] if _low(p.get("name")) == low)
    if re.search(_PREVIOUS_ROLE, roles):
        return True
    named = _items_naming(facts, name, tuple(k for k, v in facts.items() if isinstance(v, list) and k != "parties"))
    return bool(named) and all(re.search(_PAST_OWNERSHIP, t) for t in named)


def real_action_defendants(facts: Optional[Dict[str, Any]]) -> Tuple[List[str], List[str], str]:
    """(authors, owners, how) for a real action on land: who did the act obstructing the right (as parties_to_sue),
    then every current owner other than the client -- a party with role proprietario, or the owner of a
    property_history entry -- never one in the facts only as a previous owner in the chain of title (_previous_owner:
    who sold, who held the land before). v51 (Leonardo v50: Annibale, whose land lies between two of Leonardo's, left
    out of the answer): no filter by where the facts place the owner's land -- the works, the access, the path -- the
    litisconsorzio is weighed for each of them (how "current owners")."""
    if not facts:
        return [], [], ""
    clients = _clients(facts)
    authors: List[str] = []
    for event in facts.get("events") or []:
        if _typed_kind(event) in _OBSTRUCTIONS:
            authors += [n for n in _names(event.get("who")) if not _is_client(n, clients) and n not in authors]
    history = facts.get("property_history") or []
    candidates = [str(p.get("name") or "").strip() for p in facts.get("parties") or []
                  if "proprietari" in _low(p.get("role"))]
    candidates += [str(e.get("owner") or "").strip() for e in history]
    owners: List[str] = []
    for name in candidates:
        if not name or _is_client(name, clients) or name in authors or name in owners \
                or _previous_owner(facts, name, history):
            continue
        owners.append(name)
    return authors, owners, "current owners" if owners else ""


def _real_action(signals: Optional[Dict[str, Any]], analysis: Optional[Dict[str, Any]]) -> bool:
    """The regime azione_reale_immobiliare (REGIME_CANDIDATES' matching) fires on the case."""
    if analysis is None:
        return False
    areas = (analysis.get("frame") or {}).get("areas") or (signals or {}).get("areas") or []
    return any(r["id"] == REAL_ACTION_REGIME for r in regime_candidates(analysis, areas))


def parties_line(signals: Optional[Dict[str, Any]], analysis: Optional[Dict[str, Any]] = None) -> str:
    """With PARTIES_ENFORCED: on a real action on land (_real_action), "Parti da convenire: <authors>; da valutare la
    chiamata di <owners> quali proprietari dei fondi interessati (litisconsorzio necessario ex art. 102 c.p.c. se ...)"
    from real_action_defendants, with two or more of them; otherwise, with more than one party to sue, "Parti da
    convenire in giudizio: X, Y, Z (litisconsorzio necessario ex art. 102 c.p.c. se ...)". Else ""."""
    if not feature_flags.enabled(PARTIES_ENFORCED_FLAG):
        return ""
    if _real_action(signals, analysis):
        authors, owners, _ = real_action_defendants((signals or {}).get("facts"))
        if len(authors) + len(owners) < 2:
            return ""
        line = "Parti da convenire: " + (", ".join(authors) + ("; " if owners else "") if authors else "")
        if owners:
            line += (f"da valutare la chiamata di {', '.join(owners)} quali proprietari dei fondi interessati "
                     "(litisconsorzio necessario ex art. 102 c.p.c. se la decisione non può pronunciarsi che in "
                     "confronto di tutte)")
        return line
    parties = (signals or {}).get("defendants") or []
    if len(parties) < 2:
        return ""
    basis = "; ".join(f"{p['name']}: {p['why']}" for p in parties)
    return (f"Parti da convenire in giudizio: {', '.join(p['name'] for p in parties)} (litisconsorzio necessario ex "
            f"art. 102 c.p.c. se la decisione non può pronunciarsi che in confronto di tutte; {basis})")


def conclusions_block(signals: Optional[Dict[str, Any]], analysis: Optional[Dict[str, Any]] = None) -> str:
    # "unstated" (interactive_flow.mark_unstated): fired on a field the case does not state, never a fixed conclusion
    lines = [f"- {r['conclusion']}" for r in (signals or {}).get("fired") or [] if r.get("conclusion")
             and not r.get("unstated")]
    mediated = (signals or {}).get("mediation") or {}
    try:            # PARTIES_ENFORCED: logged and left out on any failure, the block as before
        parties = parties_line(signals, analysis)
    except Exception as exc:
        _log_exception("conclusions_block.parties_line", exc)
        parties = ""
    return "\n".join(lines + ([f"- {mediated['text']}"] if mediated.get("required") else [])
                     + ([f"- {parties}"] if parties else []))


def rule_tools(tools: List[Dict[str, Any]], fired: List[Dict[str, Any]],
               facts: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """The tools with those the fired rules list: always there, with the rule's reason. With the facts, the
    accertamento preventivo cites art. 696-bis c.p.c. only for a damages or credit claim (procedural_tools)."""
    out = list(tools)
    for rule in fired:
        for wanted in rule.get("tools") or []:
            table = next(t for t in PROCEDURAL_TOOLS if t["id"] == wanted["id"])
            have = next((t for t in out if t["id"] == wanted["id"]), None)
            if have:
                have["matched"] = list(dict.fromkeys([wanted["why"]] + list(have.get("matched") or [])))
            else:
                out.append({"id": table["id"], "refs": [list(r) for r in table["refs"]], "text": table["text"],
                            "why": wanted["why"], "matched": [wanted["why"]]})
    if facts is not None and not _credit_claim(facts):
        for t in out:
            if t["id"] == "accertamento_preventivo" and any(r[0] == "696-bis" for r in t["refs"]):
                t["refs"] = [r for r in t["refs"] if r[0] != "696-bis"]
                t["text"] = next(x["text_places"] for x in PROCEDURAL_TOOLS if x["id"] == "accertamento_preventivo")
    return out


def explain_rules(facts: Optional[Dict[str, Any]], terms: List[Dict[str, Any]],
                  rules: Optional[List[Dict[str, Any]]] = None, areas: Optional[List[str]] = None
                  ) -> List[Dict[str, Any]]:
    """For every rule: {"id", "fired", "skipped" (why it was not evaluated: another family), "conditions": [{"text",
    "matched", "fact"}]} -- every condition evaluated, with the fact it matched or the facts it looked at
    (scripts/fact_probe.py)."""
    families = {family(a) for a in areas or []} - {""}
    out = []
    for rule in rules if rules is not None else load_rules():
        item = {"id": rule["id"], "fired": False, "skipped": "", "conditions": []}
        if families and not rule_families(rule) & families:
            item["skipped"] = f"famiglia {'/'.join(sorted(rule_families(rule)))} fuori dalle aree del caso"
        for condition in rule["when"]:
            try:
                ok = holds(condition, facts or {}, terms)
                fact = _evidence(condition, facts or {}, terms)
            except Exception as exc:
                _log_exception("explain_rules", exc)
                ok, fact = False, f"errore: {exc}"
            item["conditions"].append({"text": describe([condition]), "matched": ok, "fact": fact})
        item["fired"] = not item["skipped"] and all(c["matched"] for c in item["conditions"])
        out.append(item)
    return out


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
                         + (f", prima della divisione {w.get('before_separation')}" if w.get("before_separation") else "")
                         for w in facts.get("works_or_signs") or [])[:200]
    if name == "common_previous_owner":
        found = facts.get("split_into_parts") or splits(facts.get("property_history") or [])
        if found:
            return "; ".join(f"unico proprietario {s['owner']}, poi alienato in parti (dedotto dalla sequenza)" for s in found)[:200]
        return "; ".join(f"{p.get('asset')} di {p.get('owner')}" + (f" {p.get('period')}" if p.get("period") else "")
                         + (f" insieme a {p.get('together_with')}" if p.get("together_with") else "")
                         + (f", finita per {p.get('ended_by')}" + (" (dedotto)" if p.get("ended_by_how") else "") if p.get("ended_by") else "")
                         for p in facts.get("property_history") or [])[:200]
    if name == "landlocked":
        if not facts.get("access"):
            return _evidence({"fact": {"pattern": _LANDLOCKED_WORDS}}, facts, terms)
        return "; ".join(f"{a.get('asset') or 'il fondo'}: {a.get('how_reached') or a.get('landlocked')}"
                         for a in facts.get("access") or [] if _yes(a.get("landlocked")))[:200]
    if name == "event":
        kinds = arg["kind"] if isinstance(arg.get("kind"), list) else [arg.get("kind")]
        return "; ".join(f"{e.get('what')} ({e.get('kind')})" for e in facts.get("events") or []
                         if _typed_kind(e) in kinds)[:200]
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
    if not ranges and str(istituto["fonte"]).startswith("regola"):
        return []                      # a rule's institution resolves only inside its ranges (load_rules: mandatory)
    kept = []
    for row in rows:
        number = (row.get("_article") or ("",))[0]
        s = row.get("s") or {}
        rubrica = article_text.heading(s)
        if ranges and not _in_ranges(number, ranges):
            continue
        if not words & _stems(rubrica):
            continue
        kept.append(row)
    return kept


# A rubrica word every part of a code uses: an institution matching an article on this word alone resolves only
# inside the word's own chapter (the live answer of 26 Oct: "usucapione di servitù apparente" -> art. 2689 c.c.,
# "Usucapione", the transcription of the judgment). Per code sigla: (stem, ranges).
GENERIC_RUBRICA_RANGES = {
    "c.c.": {"usuca": [[1158, 1167], [1061, 1061]], "presc": [[2934, 2969]], "acces": [[934, 947]]},
}


def generic_out_of_range(name: str, code: str, number: str, rubrica: str) -> str:
    """The generic rubrica word (its stem) when `name` and `rubrica` share only that word and article `number` of
    `code` lies outside its ranges (GENERIC_RUBRICA_RANGES), else "": then the institution does not resolve there."""
    table = GENERIC_RUBRICA_RANGES.get(code) or {}
    shared = _stems(name) & _stems(rubrica)
    if not shared or not shared <= set(table):
        return ""
    return next((stem for stem in sorted(shared) if not _in_ranges(number, table[stem])), "")


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
                 areas: Optional[List[str]] = None, rules: Optional[bool] = None, request: str = "",
                 case: Optional[str] = None) -> Dict[str, Any]:
    """{"facts", "terms", "fired", "problem", ...} from the extracted facts, the rules fired within the case's
    areas (only with FACT_SIGNALS_ENABLED, or rules); FACT_REGIMES_ENABLED first qualifies the parties
    (qualify_parties) and adds "qualifications", "regimes" and "tools". Logged as case_analysis_fact_signals. v60h (3):
    case given (the interactive flow), a term whose triggering fact no case sentence states (TERM_TRIGGERS) is not
    computed -- before the rules read the terms -- and listed in "terms_untriggered"."""
    regimes = regimes_enabled()
    qualifications: List[Dict[str, Any]] = []
    if regimes and facts:
        facts, qualifications = qualify_parties(facts)
    terms = compute_terms(facts, areas)
    dropped: List[Dict[str, str]] = []
    if case is not None:
        terms, dropped = untriggered(terms, case)
    fired: List[Dict[str, Any]] = []
    if enabled() if rules is None else rules:
        try:
            fired = fire_rules(facts, terms, areas=areas)
        except Exception as exc:
            _log_exception("signals_from", exc)
            logger.warning("fact signals: rules not applied: %s", exc)
            problem = problem or f"rules: {exc}"
    late = _non_use_for_a_servitude(facts, terms, fired) + no_prescription_terms(facts, fired)
    if case is not None:
        late, more = untriggered(late, case)
        dropped += more
    terms = terms + late
    out = {"facts": facts, "terms": terms, "fired": fired, "problem": problem, "areas": list(areas or []),
           "defendants": parties_to_sue(facts), "mediation": mediation(facts, fired)}
    if case is not None:
        out["terms_untriggered"] = dropped
    if regimes:
        out.update({"qualifications": qualifications, "regimes": regime_issues(facts, qualifications, areas),
                    "tools": rule_tools(procedural_tools(facts, areas, request), fired, facts), "damages": damage_heads(facts),
                    "multi_roles": multiple_roles(facts)})
    vlog("case_analysis_fact_signals", {**out, "central": all_central(out)})
    if facts:
        vlog("fact_rules", {"rules": [
            f"{'FIRED' if r['fired'] else 'not fired'} {r['id']}" + (f" ({r['skipped']})" if r["skipped"] else "") + ": "
            + "; ".join(f"{'sì' if c['matched'] else 'NO'} {c['text']}" + (f" — «{c['fact']}»" if c["fact"] else "")
                        for c in r["conditions"]) for r in explain_rules(facts, terms, areas=areas)]})
    return out


def all_central(signals: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The central issues the signals add to the frame: the regimes of the parties' qualifications first, then
    the fired rules' institutions (central_issues), each once, at most MAX_CENTRAL."""
    signals = signals or {}
    out, seen = [], set()
    # REVIEW_FIXES_ENABLED: the seller's repair as an issue of its own (round 1, 26 Oct: as a note on the terms
    # only, the boat case's answer listed it among the checks and never argued it).
    recognition = [{"nome": RECOGNITION_ISSUE, "codice": "c.c.", "ranges": [], "articolo": "1495"}] \
        if feature_flags.enabled(REVIEW_FIXES_FLAG) and _recognition_note(signals) else []
    for item in (list(signals.get("regimes") or []) + recognition
                 + central_issues(signals.get("fired") or [], limit=MAX_CENTRAL)):
        if item["nome"].lower() not in seen and len(out) < MAX_CENTRAL:
            seen.add(item["nome"].lower())
            out.append({k: item[k] for k in ("nome", "codice", "fonte", "ranges", "articolo") if k in item})
    return out


def analyse(case_text: str, areas: Optional[List[str]] = None) -> Dict[str, Any]:
    """extract_facts then signals_from with the rules, in one go (the benches)."""
    facts, problem = extract_facts(case_text)
    return signals_from(facts, problem, areas, rules=True)


def terms_block(signals: Optional[Dict[str, Any]], refs: bool = False) -> str:
    """The computed terms as facts for the plan and the procedural section ("" without any); with refs, each
    with the article it belongs to (FACT_REGIMES_ENABLED); with REVIEW_FIXES_ENABLED, then _recognition_note."""
    terms = (signals or {}).get("terms") or []
    from . import answer_invariants
    plain = answer_invariants.enabled()          # ANSWER_INVARIANTS_ENABLED (I5): statuses in plain Italian

    def line(t):
        text = answer_invariants.plain_status(t["text"]) if plain else t["text"]
        ref = (f" (art. {t['ref']})" if plain else f" [{t['ref']}]") if refs and t.get("ref") else ""
        return f"- {text}{ref}"
    lines = [line(t) for t in terms]
    note = _recognition_note(signals) if lines and feature_flags.enabled(REVIEW_FIXES_FLAG) else ""
    return "\n".join(lines + ([note] if note else []))


REVIEW_FIXES_FLAG = "REVIEW_FIXES_ENABLED"
RECOGNITION_ISSUE = "riconoscimento del vizio da parte del venditore"
_SALE_TERMS = {"denuncia_vizi_vendita", "azione_vizi_vendita"}
_REPAIR = r"ripar|messa a punto|sostitu|aggiust|sistemat|eliminare i vizi|eliminazione dei vizi"


def _recognition_note(signals: Optional[Dict[str, Any]]) -> str:
    """REVIEW_FIXES_ENABLED: a sale whose seller stepped in to repair the thing has its defects recognised -- the
    point the boat case's answers missed (26 Oct), saying instead that Tizio had not reported them in time.
    "" when no sale term applies or the facts tell of no repair."""
    signals = signals or {}
    if not {t.get("id") for t in signals.get("terms") or []} & _SALE_TERMS:
        return ""
    if not re.search(_REPAIR, _texts(signals.get("facts") or {}, ("events", "contracts", "works_or_signs"))):
        return ""
    return ("- Il venditore che interviene per riparare la cosa o si impegna a eliminarne i vizi ne riconosce "
            "l'esistenza: la denuncia non è necessaria (art. 1495, secondo comma, c.c.) e il riconoscimento "
            "interrompe la prescrizione dell'azione (art. 2944 c.c.); va verificato quando e come è avvenuto "
            "l'intervento.")


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
# REVIEW_FIXES_ENABLED (round 3, 26 Oct): the extraction fills "purpose" differently run to run, and the boat case
# lost its consumer once ("da diporto", "propri ospiti" are private use): these words count too, and with no
# purpose at all they are read in the events and the contracts of a natural person whose occupation has nothing
# to do with the purchase (_private_cues).
_PRIVATE_USE_WIDE = _PRIVATE_USE + r"|diport|svago|tempo libero|vacanz|hobby|ricreativ|ospiti"
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


def _private_cues(p: Dict[str, Any], facts: Dict[str, Any], contracts: List[Dict[str, Any]]) -> bool:
    """REVIEW_FIXES_ENABLED, for a party with no purpose extracted: the case's events or contracts name a private
    use (_PRIVATE_USE_WIDE) and the party's occupation, if any, shares no word with the contracts (a fisherman
    buying a boat is not a consumer for it; a lawyer buying one for leisure is)."""
    where = " ".join(_low(e.get(k)) for e in facts.get("events") or [] if isinstance(e, dict) for k in ("what", "who"))
    where += " " + " ".join(_low(c.get(k)) for c in facts.get("contracts") or [] if isinstance(c, dict)
                            for k in ("type", "clauses", "object"))
    if not re.search(_PRIVATE_USE_WIDE, where):
        return False
    occupation = {w[:5] for w in re.findall(r"[a-zàèéìòù]{4,}", _low(p.get("occupation")))}
    dealt = {w[:5] for c in contracts for w in re.findall(r"[a-zàèéìòù]{4,}", " ".join(
        _low(c.get(k)) for k in ("type", "clauses", "object")))}
    return not (occupation & dealt)


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
    wide = feature_flags.enabled(REVIEW_FIXES_FLAG)
    for i, p in enumerate(parties):
        name = p.get("name") or "una parte"
        consumer_role = CONSUMER in _low(p.get("role"))
        private = re.search(_PRIVATE_USE_WIDE if wide else _PRIVATE_USE, _low(p.get("purpose")))
        if not private and wide and not _low(p.get("purpose")):
            private = _private_cues(p, facts, contracts)
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
    # STRICT_ARTICLES_ENABLED: "articolo" is the institution's article (the resolver takes it as it is: no rubrica
    # names the consumer's forum, so the answers cited art. 1374 or 1469-bis c.c. for it, 26 Oct) and "nota"
    # what the writer reads about it (the provision is one letter of a long article, past its text's cut).
    {"id": "consumatore_foro", "quality": CONSUMER, "institution": "foro del consumatore",
     "codice": "Codice del Consumo", "ranges": [], "families": ["civile"], "when": [], "articolo": "33",
     "nota": "foro del consumatore: si presume vessatoria la clausola che stabilisce come sede del foro competente "
             "una località diversa da quella di residenza o domicilio elettivo del consumatore (art. 33, comma 2, "
             "lettera u), Codice del Consumo); la giurisprudenza ne ricava la competenza esclusiva del giudice del "
             "luogo di residenza o domicilio del consumatore."},
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
                    "ranges": regime["ranges"], **{k: regime[k] for k in ("articolo", "nota") if regime.get(k)}})
    return out


# REGIME_CANDIDATES (v45, a draft for the lawyer to validate): the articles a regime of the case's issues calls for, as
# candidates only -- read like any article, in the pool over and above its cap, graded like every other row; no
# conclusion and no fixed statement come with them. A regime applies, as regime_issues, within the families of the
# case's areas, when one of its "triggers" is in the name of one of the case's issues or institutions
# (regime_candidates).
REGIME_CANDIDATES_FLAG = "REGIME_CANDIDATES"
_L241 = "Legge sul Procedimento Amministrativo 2026"        # method_candidates.L241: the document by its own name
#
# v48 (the PO's red on Leonardo: art. 949 c.c., the owner's negatoria, argued for the client who claims the servitude):
# every entry, and every tool of PROCEDURAL_TOOLS, has a "client_role" -- the client's position it is for, one of
# CLIENT_ROLES or a list of them (any of) -- and a regime's article may have its own in "ref_roles"; an entry or an
# article whose condition fails on the case's facts (client_roles) is not added, logged with its role. "ref_commi": the
# comma of an article the material must hold ("last": its last comma, anchored_writing.build).
CLIENT_ROLES = ("any", "claims_servitude", "resists_servitude", "claims_ownership", "possessor", "landlocked_owner")
REGIME_CANDIDATES: List[Dict[str, Any]] = [
    {"id": "azione_reale_immobiliare", "families": ["civile"], "draft_pending_lawyer": True, "client_role": "any",
     "triggers": [r"servit\w*", r"usucapi\w*", r"(?:azione )?confessori\w*", r"(?:azione )?negatori\w*", r"rivendic\w*",
                  r"passaggio coattivo"],
     "refs": [("1079", "c.c."), ("948", "c.c."), ("949", "c.c.")],      # 2653 c.c., 21 and 102 c.p.c.: its tools
     # the negatoria only for the owner resisting a claimed servitude, the rivendicazione only for who claims ownership
     "ref_roles": {("949", "c.c."): "resists_servitude", ("948", "c.c."): "claims_ownership"}},
    # v48: the possession's own articles (accessione e successione nel possesso, interruzione per perdita)
    {"id": "usucapione", "families": ["civile"], "draft_pending_lawyer": True, "client_role": "any",
     "triggers": [r"usucapi\w*"],
     "refs": [("1146", "c.c."), ("1167", "c.c.")],
     "ref_roles": {("1146", "c.c."): "possessor",
                   ("1167", "c.c."): ["possessor", "claims_ownership", "resists_servitude"]}},
    # v48: the indemnity and the exemption of houses, courtyards, gardens and threshing floors (art. 1051, last comma)
    {"id": "passaggio_coattivo", "families": ["civile"], "draft_pending_lawyer": True, "client_role": "any",
     "triggers": [r"passaggio coattivo", r"interclus\w*"],
     "refs": [("1053", "c.c."), ("1051", "c.c.")],
     "ref_roles": {("1053", "c.c."): ["landlocked_owner", "resists_servitude"],
                   ("1051", "c.c."): ["landlocked_owner", "resists_servitude"]},
     "ref_commi": {("1051", "c.c."): "last"}},
    {"id": "responsabilita_contrattuale", "families": ["civile"], "draft_pending_lawyer": True, "client_role": "any",
     "triggers": [r"inadempi\w*", r"responsabilit\w* contrattual\w*", r"contatto sociale"],
     "refs": [("1218", "c.c."), ("1176", "c.c."), ("1223", "c.c."), ("1225", "c.c."), ("2236", "c.c.")]},
    {"id": "impossibilita_sopravvenuta", "families": ["civile"], "draft_pending_lawyer": True, "client_role": "any",
     "triggers": [r"impossibilit\w* sopravvenut\w*"],
     "refs": [("1256", "c.c."), ("1463", "c.c.")]},
    {"id": "procedimento_amministrativo", "families": ["amministrativo"], "draft_pending_lawyer": True, "client_role": "any",
     "triggers": [r"provvediment\w*", r"annullament\w*", r"autotutela", r"silenzio"],
     "refs": [("3", _L241), ("7", _L241), ("10-bis", _L241), ("21-octies", _L241), ("21-nonies", _L241),
              ("21-septies", _L241), ("29", "c.p.a."), ("31", "c.p.a.")]},
    {"id": "ottemperanza", "families": ["amministrativo"], "draft_pending_lawyer": True, "client_role": "any",
     "triggers": [r"ottemperanz\w*"],
     "refs": [("112", "c.p.a."), ("114", "c.p.a.")]},
]


def real_action_name(signals: Optional[Dict[str, Any]], regime: Dict[str, Any]) -> str:
    """The real action of the case: "azione confessoria" when the client claims a servitude (a fired rule) or the
    issues name the confessoria alone, "azione negatoria" when they name the negatoria alone, else both."""
    words = " ".join(_low(t) for t, _ in regime.get("matched") or [])
    confessoria = "confessori" in words or _claims_servitude((signals or {}).get("fired") or [])
    negatoria = "negatori" in words
    if confessoria != negatoria:
        return "azione confessoria" if confessoria else "azione negatoria"
    return "azione confessoria/negatoria"


def regime_tools(signals: Optional[Dict[str, Any]], analysis: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """REGIME_CANDIDATES: the PROCEDURAL_TOOLS keyed to a regime that fires on the case (regime_candidates), in civil
    cases only, shaped as procedural_tools' ({id, refs, text, why, matched, regime}); matched, the triggers and the
    issues they are in -- for litisconsorzio, the names of the PARTIES_ENFORCED line, which must hold two or more."""
    if not regime_candidates_enabled() or analysis is None:
        return []
    areas = (analysis.get("frame") or {}).get("areas") or (signals or {}).get("areas") or []
    families = {family(a) for a in areas} - {""}
    if families and "civile" not in families:
        return []
    fired = {r["id"]: r for r in regime_candidates(analysis, areas)}
    out = []
    roles = client_roles((signals or {}).get("facts"), (signals or {}).get("fired") or [])
    for tool in PROCEDURAL_TOOLS:
        regime = fired.get(tool.get("regime") or "")
        if not regime or not role_fits(tool.get("client_role", "any"), roles):
            continue
        matched = [f"{term} («{name}»)" for term, name in regime["matched"][:3]]
        if tool["id"] == "trascrizione_domanda":
            # the computed fact: the action whose demand is transcribed, never the servitude itself
            matched = [f"trascrizione della domanda giudiziale ({real_action_name(signals, regime)}) ai sensi "
                       "dell'art. 2653, n. 1, c.c."]
        if tool.get("parties"):
            if not parties_line(signals, analysis) or tool["regime"] != REAL_ACTION_REGIME:
                continue
            authors, owners, _ = real_action_defendants((signals or {}).get("facts"))
            matched = [f"parti da convenire: {', '.join(authors)}" if authors else "",
                       f"da valutare: {', '.join(owners)}" if owners else ""]
            matched = [m for m in matched if m]
        out.append({"id": tool["id"], "refs": [list(r) for r in tool["refs"]], "text": tool["text"],
                    "why": tool["why"], "matched": matched, "regime": tool["regime"]})
    return out


def regime_candidates_enabled() -> bool:
    return feature_flags.enabled(REGIME_CANDIDATES_FLAG)


def case_issue_names(analysis: Optional[Dict[str, Any]]) -> List[str]:
    """The names of the case's issues and institutions: the frame's questions and central issues, the analysis' issues
    and their institutions, the fired rules' institutions and the qualifications' regimes."""
    analysis = analysis or {}
    frame, signals = analysis.get("frame") or {}, analysis.get("fact_signals") or {}
    names = [str(t) for t in frame.get("tasks") or []]
    names += [str(c.get("nome")) for c in frame.get("central_issue_candidates") or [] if isinstance(c, dict)]
    for q in analysis.get("questioni") or []:
        if isinstance(q, dict):
            names.append(str(q.get("titolo") or ""))
            names += [str(i.get("nome")) for i in q.get("istituti") or [] if isinstance(i, dict)]
    names += [str(n) for r in signals.get("fired") or [] for n in r.get("institutions") or []]
    names += [str(r.get("nome")) for r in signals.get("regimes") or []]
    return [n for n in (" ".join(x.split()) for x in names) if n and n != "None"]


def regime_candidates(analysis: Optional[Dict[str, Any]], areas: Optional[List[str]] = None,
                      log: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """[{id, refs, matched: [(trigger, the issue it is in)], draft_pending_lawyer, commi}] of the REGIME_CANDIDATES the
    case's issues and institutions call for, within the families of its areas (none known: every family); an entry or
    an article whose client_role the client's roles (client_roles) do not meet left out, into `log`."""
    families = {family(a) for a in areas or []} - {""}
    names = case_issue_names(analysis)
    signals = (analysis or {}).get("fact_signals") or {}
    roles = client_roles(signals.get("facts"), signals.get("fired") or [])
    excluded: List[Dict[str, Any]] = []
    out = []
    for regime in REGIME_CANDIDATES:
        if families and not set(regime["families"]) & families:
            continue
        matched = [(m.group(0), name) for name in names for t in regime["triggers"]
                   for m in [re.search(rf"\b{t}", _low(name))] if m]
        if not matched:
            continue
        if not role_fits(regime.get("client_role", "any"), roles):
            excluded.append({"regime": regime["id"], "role": regime.get("client_role"), "roles": sorted(roles)})
            continue
        refs, left = [], []
        for r in regime["refs"]:
            want = (regime.get("ref_roles") or {}).get(tuple(r), "any")
            (refs if role_fits(want, roles) else left).append(tuple(r))
        for r in left:
            excluded.append({"regime": regime["id"], "article": f"{r[0]} {r[1]}",
                             "role": (regime.get("ref_roles") or {}).get(r), "roles": sorted(roles)})
        if refs:
            out.append({"id": regime["id"], "refs": refs, "matched": list(dict.fromkeys(matched)),
                        "draft_pending_lawyer": regime["draft_pending_lawyer"],
                        "commi": {f"{n} {c}": v for (n, c), v in (regime.get("ref_commi") or {}).items() if (n, c) in refs}})
    if excluded and log is not None:
        log.extend(excluded)
    return out


# REGIME_CANDIDATES (v48): the actions whose own article an institution may resolve to, and the client's role each is for
# -- the negatoria and the rivendicazione the owner's of the burdened land, the confessoria who claims the servitude, the
# possessory actions the possessor's. Another client's: the article is the counterparty's (action_gate).
ACTION_ARTICLES = {("949", "c.c."): "negatoria", ("948", "c.c."): "rivendicazione", ("1079", "c.c."): "confessoria",
                   ("1168", "c.c."): "possessoria", ("1170", "c.c."): "possessoria"}
ACTION_WORDS = {"negatoria": r"negatori", "rivendicazione": r"rivendic", "confessoria": r"confessori",
                "possessoria": r"possessori|reintegra|manutenzion|spogli"}
ACTION_ROLES = {"negatoria": ["resists_servitude", "claims_ownership"],
                "rivendicazione": ["resists_servitude", "claims_ownership"],
                "confessoria": ["claims_servitude"], "possessoria": ["possessor"]}


def action_gate(institution: str, ref: Any, roles: set) -> Optional[Dict[str, Any]]:
    """For an institution naming an action resolved to that action's article: {"institution", "article", "action",
    "roles", "fits", "reason"}; None for any other institution or article."""
    action = ACTION_ARTICLES.get(tuple(ref) if ref else ())
    if not action or not re.search(ACTION_WORDS[action], _low(institution)):
        return None
    fits = role_fits(ACTION_ROLES[action], roles)
    return {"institution": institution, "article": f"{ref[0]} {ref[1]}", "action": action,
            "roles": sorted(set(roles) - {"any"}), "fits": fits,
            "reason": (f"azione {action}: per il cliente" if fits else
                       f"azione {action}: spetta a chi " + " o ".join(ACTION_ROLES[action]) + ", non al cliente")}


def deterministic_side(ref: Any, roles: set) -> Optional[str]:
    """Whose an article is by the role conditions alone, "cliente" or "controparte": an action's article
    (ACTION_ARTICLES) by its action's roles, a regime candidate with its own role (ref_roles) by that; None for any
    other article (ROLE_FIT's shadow agreement)."""
    ref = tuple(ref)
    action = ACTION_ARTICLES.get(ref)
    if action:
        return "cliente" if role_fits(ACTION_ROLES[action], roles) else "controparte"
    for regime in REGIME_CANDIDATES:
        want = (regime.get("ref_roles") or {}).get(ref)
        if want is not None:
            return "cliente" if role_fits(want, roles) else "controparte"
    return None


def role_fits(condition: Any, roles: set) -> bool:
    """A client_role condition (a CLIENT_ROLES name or a list of them, any of) against the client's roles."""
    wanted = condition if isinstance(condition, (list, tuple)) else [condition or "any"]
    return "any" in wanted or bool(set(wanted) & set(roles))


_RESISTS_RE = (r"negatori|neg\w* (?:l['’]esistenza|la servit|il diritto)|contest\w* (?:la servit|il (?:diritto di )?passagg)|"
               r"pretes\w* (?:servit|diritto di passaggio)|vant\w* (?:una |la )?servit|nessun diritto di passaggio")
_OWNERSHIP_RE = (r"rivendic|restitu\w*|riottener\w*|rilascio (?:del|della|dell['’])|accertament\w* della propriet|"
                 r"riconosciment\w* della propriet")


def client_roles(facts: Optional[Dict[str, Any]], fired: Sequence[Dict[str, Any]] = ()) -> set:
    """The client's positions (CLIENT_ROLES) the facts show: claims_servitude (a fired rule acquiring a servitude, or a
    possession of a way by the client), resists_servitude (the client contests a right someone claims, and claims none),
    claims_ownership (asks for the thing back or for its ownership, or possesses another's), possessor (a possession
    by the client), landlocked_owner (the client owns a plot without access)."""
    roles = {"any"}
    if not facts:
        return roles
    clients = _clients(facts)
    mine = [p for p in facts.get("possession") or [] if _is_client(str(p.get("who") or ""), clients)]
    words = _low(" ".join(str(r) for r in facts.get("requests") or [])) + " " + _texts(facts, ("events", "proceedings"))
    if mine:
        roles.add("possessor")
    if _claims_servitude(list(fired)) or any(re.search(_WAY, f"{_low(p.get('use'))} {_low(p.get('asset'))}") for p in mine):
        roles.add("claims_servitude")
    if "claims_servitude" not in roles and re.search(_RESISTS_RE, words):
        roles.add("resists_servitude")
    owned = any(_is_client(str(h.get("owner") or ""), clients) for h in facts.get("property_history") or [])
    if re.search(_OWNERSHIP_RE, words) or (mine and not owned and any(
            "propriet" in _low(p.get("use")) or re.search(r"usucap", _low(p.get("how"))) for p in mine)):
        roles.add("claims_ownership")
    client_owner = any(_is_client(str(p.get("name") or ""), clients) and "propriet" in _low(p.get("role"))
                       for p in facts.get("parties") or [])
    if client_owner and _p_landlocked(facts, [], {}):
        roles.add("landlocked_owner")
    return roles


_DEFECTS = _DEFECT + r"|non conform|danneggi|rott|allag|incendi|crollo"
_LOSABLE = r"\bprov|dispers|perd|ripar|ripristin|smalt|demol|modific|alter|cancell|deterior|sostitu"
_URGENT = (r"pericol|sicurezz|incolumit|rischio di (?:crollo|caduta|incendio|allagamento)|crollo|urgen|imminent|"
           r"irreparabil|aggrav|peggior|sta per")

# The procedural tools the facts call for, with the articles they rest on (a code table, like
# practical_checklist.py): offered to the plan and to the procedural section with their texts.
PROCEDURAL_TOOLS: List[Dict[str, Any]] = [
    # v48: art. 696 c.p.c. for the state of places or things; art. 696-bis c.p.c. only with a damages or credit claim
    # among the computed facts (procedural_tools; Leonardo: 696-bis offered to fix a gate, no damages claimed)
    {"id": "accertamento_preventivo", "refs": [("696", "c.p.c."), ("696-bis", "c.p.c.")], "client_role": "any",
     "text": "Accertamento tecnico preventivo (art. 696 c.p.c.), per fissare lo stato dei luoghi, della cosa o dei "
             "danni prima che cambi, o consulenza tecnica preventiva ai fini della composizione della lite (art. "
             "696-bis c.p.c.), che accerta anche le cause e l'entità dei danni e tenta la conciliazione.",
     "text_places": "Accertamento tecnico preventivo (art. 696 c.p.c.), per fissare lo stato dei luoghi o della cosa "
                    "prima che cambi.",
     "why": "difetti, danni a cose o una prova che può andare perduta"},
    {"id": "provvedimento_urgenza", "refs": [("700", "c.p.c.")], "client_role": "any",
     "text": "Provvedimento d'urgenza (art. 700 c.p.c.), se nel tempo necessario per la causa il diritto è "
             "minacciato da un pregiudizio imminente e irreparabile e nessun altro provvedimento cautelare lo tutela.",
     "why": "un pericolo per la sicurezza, un danno che si aggrava o un'urgenza"},
    # The criminal route of a personal harm caused by negligence, also when the frame's area is only civil.
    {"id": "profilo_penale", "refs": [("589", "c.p."), ("590", "c.p."), ("76", "c.p.p."), ("2947", "c.c.")], "client_role": "any",
     "text": "Profilo penale: la morte o le lesioni di una persona causate per colpa possono costituire reato "
             "(omicidio colposo, art. 589 c.p., procedibile d'ufficio; lesioni personali colpose, art. 590 c.p., "
             "di regola procedibili a querela): valutare la denuncia o la querela e i suoi termini, la costituzione "
             "di parte civile nel processo penale (art. 76 c.p.p.), le prove che il procedimento penale acquisisce "
             "(accertamenti, consulenze, testimonianze) e la prescrizione più lunga dell'azione civile quando il "
             "fatto è considerato dalla legge come reato (art. 2947, terzo comma, c.c.).",
     "why": "la morte o lesioni di una persona causate per colpa"},
    # The live answer of 26 Oct: asked whether "una richiesta anche sanzionatoria" was possible against the neighbour
    # who locked the road, the answer offered only damages. The tool for a sum per violation or day of delay.
    {"id": "misure_coercitive", "refs": [("614-bis", "c.p.c.")], "client_role": "any",
     "text": "Misure di coercizione indiretta (art. 614-bis c.p.c.): con il provvedimento di condanna all'adempimento "
             "di obblighi diversi dal pagamento di somme di denaro (rimuovere un ostacolo, consentire un passaggio, "
             "astenersi da una condotta) il giudice, su richiesta di parte e salvo che sia manifestamente iniquo, "
             "fissa la somma dovuta dall'obbligato per ogni violazione o inosservanza successiva o per ogni ritardo "
             "nell'esecuzione del provvedimento; la richiesta va fatta con la domanda di condanna.",
     "why": "la richiesta di una sanzione o di una penalità per chi non adempie o non rispetta l'ordine del giudice"},
]

# REGIME_CANDIDATES (v45, a draft pending the lawyer's validation): the procedural tools of a regime of the case's
# issues (regime_tools), civil cases only. Each tool's text is its article's own rubrica, read from the corpus with the
# other tools' rows (defensive_generation._regime_rows); "text" here is only what stands in when it was not read.
# litisconsorzio only with the PARTIES_ENFORCED line naming two or more parties, those names as its facts.
PROCEDURAL_TOOLS += [
    {"id": "trascrizione_domanda", "regime": "azione_reale_immobiliare", "refs": [("2653", "c.c.")], "client_role": "any",
     "text": "Trascrizione della domanda giudiziale (art. 2653 c.c.).", "why": "un'azione reale su un immobile"},
    {"id": "competenza_territoriale", "regime": "azione_reale_immobiliare", "refs": [("21", "c.p.c.")], "client_role": "any",
     "text": "Foro per le cause relative a diritti reali su immobili (art. 21 c.p.c.).",
     "why": "un'azione reale su un immobile"},
    {"id": "litisconsorzio", "regime": "azione_reale_immobiliare", "refs": [("102", "c.p.c.")], "parties": True, "client_role": "any",
     "text": "Litisconsorzio necessario (art. 102 c.p.c.).", "why": "più parti da convenire"},
]

# A request for a sanction or a penalty against who does not comply (FACT_REGIMES_ENABLED: art. 614-bis c.p.c.).
_SANCTION = (r"sanzion|penalit|astreinte|coerci|somma (?:di denaro )?per ogni (?:giorno|violazione|ritardo)|"
             r"per ogni giorno di ritardo|costringer\w* (?:a|ad) (?:rispettar|adempier|eseguir|rimuover)")


def asks_sanction(text: str) -> str:
    """The words of `text` asking for a sanction or a penalty for non-compliance, or ""."""
    m = re.search(_SANCTION, _low(text))
    return m.group(0) if m else ""


def procedural_tools(facts: Optional[Dict[str, Any]], areas: Optional[List[str]] = None,
                     request: str = "") -> List[Dict[str, Any]]:
    """The PROCEDURAL_TOOLS the facts call for, in civil cases (none when the case's areas are all criminal or
    administrative): {id, refs, text, why, matched}. The accertamento preventivo for defects, damage to things
    or evidence that may be lost; the provvedimento d'urgenza for a safety risk, a harm that grows or urgency;
    the measures of art. 614-bis c.p.c. when the client's requests (or `request`, the user's message) ask for a
    sanction or a penalty against who does not comply."""
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
    # REVIEW_FIXES_ENABLED: after a death the client's right is to damages, which no danger in the meantime makes
    # irreparable (the pool case, 26 Oct: art. 700 c.p.c. offered "to make the pool safe" for the parents).
    if urgent and not (feature_flags.enabled(REVIEW_FIXES_FLAG) and _death(facts)):
        found["provvedimento_urgenza"] = [urgent]
    harm = _personal_harm(facts)
    if harm and not re.search(_INTENT, texts):
        found["profilo_penale"] = [harm]
    sanction = next((str(r)[:140] for r in facts.get("requests") or [] if r and asks_sanction(str(r))), "")
    if sanction or (request and asks_sanction(request)):
        found["misure_coercitive"] = [sanction or " ".join(request.split())[:140]]
    out = []
    roles = client_roles(facts)
    for t in PROCEDURAL_TOOLS:
        if t["id"] not in found or not role_fits(t.get("client_role", "any"), roles):
            continue
        refs = [list(r) for r in t["refs"]]
        text = t["text"]
        if t["id"] == "accertamento_preventivo" and not _credit_claim(facts):
            refs, text = [r for r in refs if r[0] != "696-bis"], t["text_places"]
        if t["id"] == "profilo_penale":         # the offence the harm fits: death, injury, or both
            death = _death(facts)
            drop = ([["590", "c.p."]] if death and not _injured(facts) else []) + ([] if death else [["589", "c.p."]])
            refs = [r for r in refs if r not in drop]
        out.append({"id": t["id"], "refs": refs, "text": text, "why": t["why"], "matched": found[t["id"]]})
    return out


_CREDIT_RE = r"risarc|danni\b|danno\b|credit|pagament|somm[ae]\b|indennizz|rimbors|corrispettiv|inadempi"


def _credit_claim(facts: Dict[str, Any]) -> bool:
    """A damages or credit claim among the computed facts: a damage head, money, an injury, a contract, or a request
    for one."""
    if damage_heads(facts) or facts.get("money") or facts.get("injuries") or facts.get("contracts"):
        return True             # a contract: its inexact performance is a credit (art. 696-bis c.p.c.)
    return any(re.search(_CREDIT_RE, _low(r)) for r in facts.get("requests") or [])


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
    """The parties' inferred qualifications, one line each ("" without any); with STRICT_ARTICLES_ENABLED, then
    the "nota" of each regime they bring in."""
    lines = [f"- {q['party']}: {q['quality']} ({q['why']})" for q in (signals or {}).get("qualifications") or []]
    if lines and feature_flags.enabled("STRICT_ARTICLES_ENABLED"):
        lines += [f"- {r['nota']}" for r in (signals or {}).get("regimes") or [] if r.get("nota")]
    return "\n".join(lines)


CRIMINAL_TOOL = "profilo_penale"
ROLE_BLOCKS_FLAG = "ROLE_BLOCKS_ENABLED"

# ROLE_BLOCKS_ENABLED (fetch_audit on run 0812): the criminal profile was written for the offended party whatever the
# client's side, so an accused client was advised to file a querela and become parte civile. The profile by side.
CRIMINAL_BY_SIDE: Dict[str, Dict[str, Any]] = {
    "accused": {
        "refs": [("589", "c.p."), ("590", "c.p."), ("74", "c.p.p."), ("2947", "c.c."), ("2044", "c.c.")],
        "text": "Profilo penale per la parte assistita, alla quale il fatto può essere addebitato: la morte o le "
                "lesioni di una persona causate per colpa possono costituire reato (omicidio colposo, art. 589 c.p., "
                "procedibile d'ufficio; lesioni personali colpose, art. 590 c.p., di regola procedibili a querela "
                "della persona offesa: verificare se la querela è stata proposta, il suo termine e la possibile "
                "remissione). Le difese: l'assenza di colpa o del nesso causale, il fatto della vittima o di un "
                "terzo, la qualificazione del fatto. Nel processo penale la persona offesa o i suoi eredi possono "
                "costituirsi parte civile contro la parte assistita (art. 74 c.p.p.): la loro domanda di "
                "risarcimento va contrastata anche lì. L'esposizione civile: il risarcimento chiesto dal "
                "danneggiato, con la prescrizione più lunga quando il fatto è considerato dalla legge come reato "
                "(art. 2947, terzo comma, c.c.); se i fatti mostrano che la parte assistita ha agito per "
                "difendersi, la legittima difesa esclude anche la responsabilità civile (art. 2044 c.c.).",
        "what": "il reato configurabile con la sua norma, la procedibilità, le difese, la costituzione di parte civile "
                "della persona offesa contro la parte assistita e la sua esposizione civile"},
    "victim": {
        "refs": [("589", "c.p."), ("590", "c.p."), ("336", "c.p.p."), ("76", "c.p.p."), ("2947", "c.c.")],
        "text": "Profilo penale a tutela della parte assistita, persona offesa o suoi congiunti: la morte o le "
                "lesioni di una persona causate per colpa possono costituire reato (omicidio colposo, art. 589 c.p., "
                "procedibile d'ufficio: basta la denuncia; lesioni personali colpose, art. 590 c.p., di regola "
                "procedibili a querela, da proporre nel termine di legge: art. 336 c.p.p.). Nel processo penale la "
                "parte assistita può costituirsi parte civile per il risarcimento (art. 76 c.p.p.) e si giova delle "
                "prove che il procedimento acquisisce (accertamenti, consulenze, testimonianze); l'azione civile ha "
                "la prescrizione più lunga quando il fatto è considerato dalla legge come reato (art. 2947, terzo "
                "comma, c.c.).",
        "what": "il reato configurabile con la sua norma, la procedibilità e la querela, la costituzione di parte "
                "civile della parte assistita (art. 76 c.p.p.), le prove del procedimento penale e la prescrizione "
                "dell'azione civile"},
}
# The client's side from its name and position, by the first role word in it ("convenuto dai familiari della vittima"
# is accused, "familiari della vittima" the offended side); in a harm case who is sued is who the fact is charged to.
_ACCUSED_CRIMINAL = r"imputat|indagat|accusat|querelat|sottopost\w* a (?:procedimento|indagin)"
_VICTIM_CRIMINAL = r"persona offesa|\boffes[oaie]\b|vittim|parte civile|querelant|denunciant"
_ACCUSED_CIVIL = r"convenut|resistent|danneggiant|responsabil|autore del (?:fatto|danno|sinistro)"
_VICTIM_CIVIL = r"danneggiat|\battor[ei]\b|ricorrent|familiar|congiunt|\bered[ei]\b|genitor"


def client_side(party: str) -> str:
    """"accused", "victim" or "" (no role word) from the client's name and position."""
    text = _low(party)
    first = []
    for side, pattern in (("accused", _ACCUSED_CRIMINAL), ("victim", _VICTIM_CRIMINAL), ("accused", _ACCUSED_CIVIL),
                          ("victim", _VICTIM_CIVIL)):
        m = re.search(pattern, text)
        if m:
            first.append((m.start(), side))
    return min(first)[1] if first else ""


def fit_to_role(signals: Optional[Dict[str, Any]], party: str) -> str:
    """ROLE_BLOCKS_ENABLED: in place, the criminal profile among the signals' tools written for the client's side
    (CRIMINAL_BY_SIDE: text and refs, 589/590 c.p. kept as the harm chose them); the side, "" when unknown or the
    flag is off (the profile then unchanged)."""
    if not feature_flags.enabled(ROLE_BLOCKS_FLAG) or not signals:
        return ""
    side = client_side(party)
    if not side:
        return ""
    for tool in signals.get("tools") or []:
        if tool.get("id") != CRIMINAL_TOOL:
            continue
        offences = [list(r) for r in tool.get("refs") or [] if tuple(r) in (("589", "c.p."), ("590", "c.p."))]
        fitted = CRIMINAL_BY_SIDE[side]
        tool.update(text=fitted["text"], side=side, what=fitted["what"],
                    refs=offences + [list(r) for r in fitted["refs"] if r[0] not in ("589", "590")])
    signals["side"] = side
    return side


def tools_block(signals: Optional[Dict[str, Any]], criminal: bool = False) -> str:
    """The procedural tools the facts call for, one line each, with the facts that call for them; the criminal route
    of a personal harm alone with criminal, never among the others."""
    return "\n".join(f"- {t['text']} Nel caso: {'; '.join(t['matched'])}." for t in (signals or {}).get("tools") or []
                     if (t["id"] == CRIMINAL_TOOL) == criminal)


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
            _log_exception("attach_term_sources", exc)
            logger.warning("fact regimes: no text for the term %s: %s", term.get("id"), exc)
    try:
        rows = dg._fetch_any(session, documents, [("5", "d.lgs. 28/2010")]) if (signals or {}).get("facts") else []
        if rows:
            signals["mediation"] = mediation(signals["facts"], signals.get("fired") or [], _section_text(rows[0]))
    except Exception as exc:
        _log_exception("attach_term_sources", exc)
        logger.warning("fact regimes: art. 5 d.lgs. 28/2010 not read, the table's list stands: %s", exc)


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
