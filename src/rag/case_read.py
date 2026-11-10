"""INTERACTIVE_CASE_FIRST (A): the first read of a case -- one bounded model call on the case text and the user's message,
before any anchoring: the client and the goal, the facts, what the case does not say, and the hypotheses (the main line,
the subordinate line, the other side's likely line), each naming the legal institutions it rests on (never article
numbers) with the facts it needs proven and the evidence to collect.

It runs beside the analysis and retrieval (start(): a thread), so it adds no wall time; its answer is taken when
retrieval ends, waiting at most READ_SECONDS from the start, else none. Every list is bounded and every string cut
(clean()); a reply that does not parse is no read. The user's request is never one of the facts (not_request()).
"""

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

READ_SECONDS = 20.0
READ_TOKENS = 2200              # v60e: the lines of up to three parties when the client is not known; v60g: four, and
                                # the alternative qualifications
LINES = ("principale", "subordinata", "controparte")
MAX_FACTS, MAX_UNKNOWN, MAX_HYPOTHESES, MAX_ITEMS = 12, 6, 12, 4
MAX_PARTIES, MAX_ALTERNATIVES = 6, 4
TOPUP_MIN_SECONDS = 6.0         # v60g (4): the top-up for the parties left without lines, only with this much budget left
TEXT_CHARS = 220

_SYSTEM = (
    "Sei un avvocato esperto. Leggi il caso e la richiesta dell'utente PRIMA di cercare qualsiasi norma, e rispondi con "
    "un JSON con ESATTAMENTE queste chiavi:\n"
    "- cliente: {nome, ruolo} -- la parte che l'utente assiste se il testo lo dice, altrimenti nome \"\";\n"
    "- obiettivo: che cosa vuole ottenere il cliente, in una frase;\n"
    "- fatti: i fatti del caso, uno per voce, come il testo li dice (al massimo 12; mai la domanda o la richiesta "
    "dell'utente);\n"
    "- non_detto: i fatti rilevanti che il caso NON dice (al massimo 6);\n"
    "- parti: ogni persona, società o ente che il caso nomina come parte, {nome, ruolo}, compresa la persona offesa "
    "(la vittima del reato o del danno) anche se non ha un nome (\"la persona offesa\"); mai una persona deceduta;\n"
    "- ipotesi: al massimo 12 voci {parte, linea, tesi, istituti, rubricati, da_provare, prove}: parte è il nome della "
    "parte di cui è la linea; se il testo dice chi l'utente assiste, tutte le linee sono di quella parte; se NON lo "
    "dice, dai le linee di OGNI parte di \"parti\" che l'utente potrebbe assistere, compresa la persona offesa (al "
    "massimo quattro parti, mai una persona deceduta), ognuna con le sue; linea è \"principale\" (la tesi più forte "
    "per quella parte), \"subordinata\" (se la principale cade) o \"controparte\" (la linea che l'altra parte "
    "probabilmente sosterrà contro di lei); tesi in una frase; istituti: gli istituti giuridici su cui si regge, per nome (es. "
    "\"legittima difesa\", \"servitù di passaggio\", \"risoluzione per inadempimento\"), MAI numeri di articolo; "
    "rubricati: per ciascun istituto, nello stesso ordine, il suo nome come rubricato nel codice (la rubrica "
    "dell'articolo che lo disciplina, es. \"Difesa legittima\", \"Servitù coattive\", \"Risoluzione per "
    "inadempimento\"), MAI numeri di articolo; "
    "da_provare: i fatti che vanno provati; prove: le prove da raccogliere.\n"
    "- qualificazioni_alternative: al massimo 4 voci {istituto, rubricato, perche}: le ALTRE qualificazioni giuridiche "
    "degli stessi fatti, diverse da quelle delle ipotesi (es. truffa o insolvenza fraudolenta; furto o appropriazione "
    "indebita; responsabilità contrattuale o extracontrattuale), ciascuna con il nome come rubricato nel codice e in una "
    "frase il fatto che la farebbe preferire; MAI numeri di articolo.\n"
    "Non inventare fatti. Rispondi SOLO con il JSON.")
TOPUP = ("PARTI SENZA LINEE: {parties}. Dai SOLO le ipotesi (principale, subordinata, controparte) di queste parti, "
         "con \"parte\" uguale al loro nome; le altre chiavi possono restare vuote.")


def _cut(value: Any, chars: int = TEXT_CHARS) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= chars else text[:chars].rsplit(" ", 1)[0] + "…"


def _list(value: Any, top: int) -> List[str]:
    items = value if isinstance(value, list) else []
    return [x for x in (_cut(v) for v in items) if x][:top]


_ARTICLE_RE = re.compile(r"\bartt?\.\s*\d+|\b\d+(?:-[a-z]+)?\s+c\.\s*[cp]\.", re.IGNORECASE)


def _words(text: str) -> set:
    return {w[:6] for w in re.findall(r"[a-zà-ü]{4,}", (text or "").lower())}


def not_request(fact: str, request: str) -> bool:
    """A fact that is not the user's request: no question, and not most of the request's words."""
    if not fact or fact.rstrip().endswith("?"):
        return False
    asked = _words(request)
    mine = _words(fact)
    return not (asked and mine and len(mine & asked) >= max(2, int(0.7 * len(mine))))


def _points(value: Any) -> List[str]:
    """v60d: the «non_detto» items however the model writes them -- a list of strings, of objects (their values
    joined), or one string (split at semicolons and new lines)."""
    if isinstance(value, str):
        value = [p for p in re.split(r"\s*(?:;|\n|\s-\s)\s*", value) if p.strip()]
    out = []
    for v in value if isinstance(value, list) else []:
        if isinstance(v, dict):
            v = " ".join(str(x) for x in v.values() if isinstance(x, (str, int, float)) and str(x).strip())
        out.append(v)
    return out


def clean(raw: Any, request: str = "") -> Dict[str, Any]:
    """The model's reply as the read: bounded lists, cut strings, the lines among LINES, institutions without article
    numbers, the request out of the facts. {} when nothing usable."""
    if not isinstance(raw, dict):
        return {}
    client = raw.get("cliente") if isinstance(raw.get("cliente"), dict) else {}
    hypotheses = []
    for h in raw.get("ipotesi") or [] if isinstance(raw.get("ipotesi"), list) else []:
        if not isinstance(h, dict):
            continue
        line = str(h.get("linea") or "").strip().lower()
        claim = _cut(h.get("tesi"))
        if line not in LINES or not claim:
            continue
        institutions = [i for i in _list(h.get("istituti"), MAX_ITEMS) if not _ARTICLE_RE.search(i)]
        rubricated = [i for i in _list(h.get("rubricati"), MAX_ITEMS) if not _ARTICLE_RE.search(i)]    # v60e
        party = _cut(h.get("parte"), 80) or _cut(client.get("nome"), 80)       # v60e: whose line it is
        hypotheses.append({"line": line, "claim": claim, "institutions": institutions, "rubricated": rubricated,
                           "party": party,
                           "to_prove": _list(h.get("da_provare"), MAX_ITEMS), "evidence": _list(h.get("prove"), MAX_ITEMS)})
    parties = []
    for p in raw.get("parti") or [] if isinstance(raw.get("parti"), list) else []:
        name = _cut(p.get("nome"), 80) if isinstance(p, dict) else _cut(p, 80)
        if name and not any(same_party(name, x["name"]) for x in parties):
            parties.append({"name": name, "role": _cut(p.get("ruolo"), 80) if isinstance(p, dict) else ""})
    alternatives = []
    for a in raw.get("qualificazioni_alternative") or [] if isinstance(raw.get("qualificazioni_alternative"), list) else []:
        if not isinstance(a, dict):
            continue
        name, rubricated = _cut(a.get("istituto"), 80), _cut(a.get("rubricato"), 80)
        if name and not _ARTICLE_RE.search(name) and not _ARTICLE_RE.search(rubricated or ""):
            alternatives.append({"name": name, "rubricated": rubricated, "why": _cut(a.get("perche") or a.get("perché"))})
    out = {"client": {"name": _cut(client.get("nome"), 80), "role": _cut(client.get("ruolo"), 80)},
           "parties": parties[:MAX_PARTIES], "alternatives": alternatives[:MAX_ALTERNATIVES],
           "goal": _cut(raw.get("obiettivo")),
           "facts": [f for f in _list(raw.get("fatti"), MAX_FACTS) if not_request(f, request)],
           "unknown": _list(_points(raw.get("non_detto")), MAX_UNKNOWN),
           "unknown_raw": {"type": type(raw.get("non_detto")).__name__,
                           "n": len(raw.get("non_detto")) if isinstance(raw.get("non_detto"), (list, str)) else 0},
           "hypotheses": hypotheses[:MAX_HYPOTHESES]}
    return out if out["hypotheses"] or out["facts"] else {}


CLIENT_SET = ("IL CLIENTE È: {client}. Tutte le linee sono di questa parte: la principale e la subordinata a suo favore, "
              "la controparte quella che l'altra parte sosterrà contro di lei.")


def read(text: str, request: str = "", lang: str = "it", client: str = "", parties: Sequence[str] = ()) -> Dict[str, Any]:
    """The first read, one model call (READ_TOKENS); {} on any failure (logged). client (v60e): the party settled, all
    the lines its own (the re-read for the settled client). parties (v60g 4): only these parties' lines (the top-up)."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from . import defensive_generation as dg
    case = dg._case_text(text, request)
    human = f"CASO:\n{case}" + (f"\n\n{CLIENT_SET.format(client=client)}" if client else "") \
        + (f"\n\n{TOPUP.format(parties='; '.join(parties))}" if parties and not client else "")
    try:
        raw = dg._chat_json([SystemMessage(content=_SYSTEM), HumanMessage(content=human)],
                            max_tokens=READ_TOKENS, temperature=0)
    except Exception as exc:
        logger.warning("case_read: the first read failed: %s", exc)
        return {}
    out = clean(raw, request)
    if client and out:
        out["client"] = {**(out.get("client") or {}), "name": out.get("client", {}).get("name") or client}
        for h in out.get("hypotheses") or []:
            h["party"] = client
    return out


def same_party(a: Any, b: Any) -> bool:
    """v60e: two ways of naming one party («Mario Rossi» and «Rossi», «la compagnia Alfa» and «Alfa»)."""
    x, y = " ".join(str(a or "").lower().split()), " ".join(str(b or "").lower().split())
    if not x or not y:
        return False
    if x in y or y in x:
        return True
    words = lambda t: {w for w in re.findall(r"[a-zà-ü]{4,}", t)} - {"della", "dello", "delle", "degli", "società"}
    return bool(words(x) & words(y))


def side(hypotheses: List[Dict[str, Any]], client: str) -> List[Dict[str, Any]]:
    """v60e: the settled client's lines -- those whose party is the client; none when no line names its party (whose
    they are is not known: the caller re-reads for the client). No client: every line."""
    if not client:
        return list(hypotheses or [])
    return [h for h in hypotheses or [] if h.get("party") and same_party(h["party"], client)]


def missing_parties(read: Dict[str, Any]) -> List[str]:
    """v60g (4): with the client not known, the named parties (read["parties"]) no line is of."""
    if not read or (read.get("client") or {}).get("name"):
        return []
    hyps = read.get("hypotheses") or []
    return [p["name"] for p in read.get("parties") or []
            if p.get("name") and not any(same_party(h.get("party"), p["name"]) for h in hyps)]


def topped_up(read: Dict[str, Any], more: Dict[str, Any], parties: Sequence[str]) -> Dict[str, Any]:
    """The read with the top-up's lines of the parties asked for (each line's party one of them)."""
    if not more:
        return read
    out = dict(read)
    lines = [h for h in more.get("hypotheses") or [] if any(same_party(h.get("party"), p) for p in parties)]
    out["hypotheses"] = list(read.get("hypotheses") or []) + lines
    out["topped_up"] = {"parties": list(parties), "lines": len(lines)}
    return out


def topup_read(text: str, request: str, lang: str, read_: Dict[str, Any], seconds_left: float,
               reader=None) -> Dict[str, Any]:
    """v60g (4): the parties left without lines asked once more (one call), when TOPUP_MIN_SECONDS are left."""
    parties = missing_parties(read_)
    if not parties or seconds_left < TOPUP_MIN_SECONDS:
        if parties:
            return {**read_, "topped_up": {"parties": parties, "lines": 0, "skipped": f"{seconds_left:.0f}s left"}}
        return read_
    reader = reader or (lambda t, r, l: read(t, r, l, parties=parties))
    return topped_up(read_, reader(text, request, lang), parties)


class Pending:
    """The first read running beside the rest of round 1. result() waits at most what is left of READ_SECONDS from
    the start (0 when the rest took longer): the join is bounded at READ_SECONDS. ms is the call itself (its own thread,
    start to end; None while it runs), wait_ms what result() waited, llm the chat calls it made (its own stage meter)."""

    def __init__(self, text: str, request: str = "", lang: str = "it", reader=None, topup=None):
        from . import stage_meter
        self.t0 = time.time()
        self.ms: Optional[float] = None
        self.wait_ms: Optional[float] = None
        self.timed_out = False
        self.first: Dict[str, Any] = {}         # v60g: the first call's read, while a top-up runs
        self.meter = stage_meter.Meter()
        reader = reader or read

        def run():
            started = time.time()
            try:
                with stage_meter.own(self.meter):
                    self.meter.mark("first_read")
                    out = reader(text, request, lang)
                    if topup is not None and out:
                        self.first = out
                        self.meter.mark("first_read_topup")
                        out = topup(text, request, lang, out, READ_SECONDS - (time.time() - self.t0))
                    return out
            finally:
                self.ms = (time.time() - started) * 1000
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="case_read")
        self._future = self._pool.submit(run)

    @property
    def llm(self) -> int:
        return int(self.meter.total("llm"))

    def result(self) -> Dict[str, Any]:
        """The read (cached: v60d takes it before grading, for the first-read search, and again after retrieval)."""
        if getattr(self, "_out", None) is not None:
            return self._out
        left = max(0.0, READ_SECONDS - (time.time() - self.t0))
        t_wait = time.time()
        try:
            out = self._future.result(timeout=left) or {}
        except FutureTimeout:
            self.timed_out = True
            out = dict(self.first)          # v60g: the top-up late, the first call's read stands
            if out:
                out["topped_up"] = {"parties": missing_parties(out), "lines": 0, "skipped": "timeout"}
        except Exception as exc:
            logger.warning("case_read: the first read failed: %s", exc)
            out = {}
        self.wait_ms = (time.time() - t_wait) * 1000
        self._pool.shutdown(wait=False)
        self._out = out
        return out


def start(text: str, request: str = "", lang: str = "it", reader=None, topup=None) -> Pending:
    """The first read beside round 1; with the default reader, the top-up for the parties left without lines (v60g 4)."""
    if reader is None and topup is None:
        topup = topup_read
    return Pending(text, request, lang, reader, topup)
