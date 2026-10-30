"""FACT_ISSUES_ENABLED (default false): the decisive facts of a case name the legal institutions they raise.

scripts/fetch_audit.py on gr-off / gr-on (32 cases): of 298 lost expected articles whose trigger fact the
understanding captured, 88% were lost at "article resolved", and 258 were captured ONLY as a fact -- no issue or
institution was ever named from it ("infezione dopo l'intervento" and no medical liability issue, so no 1218 / 2236
c.c.; "la lavatrice si rompe dopo tre mesi" and no sale-defects issue, so no 1490 / 1495 c.c. or Codice del Consumo).

With the flag, after the frame and the typed facts (fact_signals) and before the analysis, one call: for each
decisive fact, the legal institutions it raises -- named in Italian with the words of the code's or the law's
headings, with the code, the area and the client's role in it; never an article number. At most MAX_INSTITUTIONS,
each not already a central issue, join the frame's central issues with their fact attached ("fatto", "fonte":
"fatto"). From there the existing path takes them: the analysis sees them in the frame block, _with_central_questions
makes each a question, _resolve_institution resolves it (rubrica and title matches, the heading index, full text,
STRICT_ARTICLES' rules and generic-word chapter ranges), and fact_signals.keep_resolved keeps only an article whose
rubrica shares a word with the institution. Logged as case_analysis_fact_issues (fact -> institutions) and, per
institution, case_analysis_fact_institution (-> the articles kept / rejected).

FACT_ISSUES_RESOLVER_V2_ENABLED (default false; fact_loss_report on the live run: of 119 articles lost at "article
resolved", 106 had an institution named from their fact, resolved to other articles (46%) or to nothing (43%), named
generically or in Latin -- "Responsabilità medica", "Risarcimento danni", "Decisione amministrativa", "Dolus
eventualis", "Diligentia"): the call names each institution as precisely as a code heading, in the code's own Italian
terms (NAMING_RULE); a name that is generic (generic_name: no specific word, or a generic head with fewer than two
specific words) or Latin is sent back once to be renamed, and dropped if the new name is still one; the institutions
then resolve through fact_resolver.resolve (src/rag/fact_resolver.py).

FACT_ISSUES_EVERY_FACT_ENABLED (default false; live run aw-20261030-1821: of 142 lost expected articles 85% failed at
"article resolved", 119 with their trigger captured only as a fact -- no institution named from it): the call names
at least one institution for EVERY decisive fact (at most two each), the cap scales with the case (cap(): the frame's
facts, one more per 4000 characters past the first, between MAX_INSTITUTIONS and MAX_WIDE), and the cut keeps each
fact's first institution before any fact's second (clean(by_fact=True)), so a long case's later facts are not starved.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from . import feature_flags
from .verbose_logger import vlog

logger = logging.getLogger(__name__)

FLAG = "FACT_ISSUES_ENABLED"
EVERY_FACT_FLAG = "FACT_ISSUES_EVERY_FACT_ENABLED"
RESOLVER_V2_FLAG = "FACT_ISSUES_RESOLVER_V2_ENABLED"
MAX_INSTITUTIONS = 8
MAX_WIDE = 16              # FACT_ISSUES_EVERY_FACT_ENABLED: the cap of a long case, at most
CHARS_PER_EXTRA = 4000     # one more institution per this many characters of case past the first
FONTE = "fatto"
_AREAS = ("penale", "contratti", "responsabilita_civile", "locazione", "successioni", "diritti_reali", "famiglia",
          "consumo", "amministrativo", "lavoro", "tributario", "altro")
_NUMBER_RE = re.compile(r"\bartt?\.|\barticol|\b\d{2,4}(?:-\w+)?\b")

SYSTEM = (
    "Sei un avvocato italiano. Ricevi un CASO e i FATTI che ne sono stati estratti. Per ciascun fatto che decide "
    "l'esito (un danno, un difetto, un ritardo, un intervento, una qualità di una parte, un atto ricevuto, una data, "
    "un importo), indica gli istituti giuridici che quel fatto pone, anche se il caso non li nomina: per esempio "
    "un'infezione dopo un intervento pone la responsabilità della struttura sanitaria e del medico; un "
    "elettrodomestico che si guasta dopo pochi mesi pone la garanzia per i vizi della cosa venduta e il difetto di "
    "conformità del bene di consumo. Ogni istituto con il nome con cui è intitolato nel codice o nella legge, il "
    "codice (c.c., c.p.c., c.p., c.p.p., c.p.a., c.d.s., o il nome per intero di un'altra legge), l'area, tra "
    + ", ".join(f'"{a}"' for a in _AREAS) + ", e il ruolo che la parte assistita ha in esso (per esempio "
    "\"danneggiato\", \"acquirente consumatore\", \"imputato\"). Mai un numero di articolo; nessun istituto che i "
    "fatti non pongono; dal fatto più decisivo al meno, al massimo otto istituti in tutto. Rispondi SOLO con un "
    "oggetto JSON: {\"istituti\": [{\"fatto\": \"il fatto, con le parole del caso\", \"nome\": \"...\", "
    "\"codice\": \"...\", \"area\": \"...\", \"ruolo_cliente\": \"...\"}]}.")


EVERY_FACT_RULE = ("per OGNI fatto decisivo almeno un istituto (al massimo due per fatto), dal fatto più decisivo "
                   "al meno, al massimo {cap} istituti in tutto")
_ONE_CAP_RULE = "dal fatto più decisivo al meno, al massimo otto istituti in tutto"
assert _ONE_CAP_RULE in SYSTEM


NAMING_RULE = (
    "\nNOMI DEGLI ISTITUTI: ogni nome è un istituto specifico del diritto italiano, preciso come la rubrica di un "
    "articolo e con i termini del codice o della legge (per esempio \"inadempimento del conduttore\", \"risoluzione "
    "della locazione per morosità\", \"responsabilità contrattuale della struttura sanitaria\", \"giudizio di "
    "ottemperanza\", \"garanzia per i vizi della cosa venduta\"). Vietati i nomi generici (\"risarcimento danni\", "
    "\"responsabilità\", \"responsabilità medica\", \"decisione amministrativa\", \"tutela\") e il latino "
    "(\"dolus eventualis\", \"diligentia\"): nomina l'istituto preciso che il fatto pone, nella lingua del codice.")
RETRY_TITLE = "NOMI DA RIFARE"
# A name opening on one of these needs a specific word after it that is not a broad qualifier (BROAD_QUALIFIERS):
# "responsabilità del precettore", "responsabilità contrattuale della struttura sanitaria" -- not "responsabilità
# medica", "decisione amministrativa".
GENERIC_HEADS = ("risarcimento", "responsabilità", "responsabilita", "danno", "danni", "decisione", "provvedimento",
                 "tutela", "diritto", "azione", "obbligo", "obbligazione", "violazione", "illecito", "diligenza",
                 "procedimento", "garanzia", "contratto", "rapporto", "atto")
BROAD_QUALIFIERS = ("medica medico amministrativa amministrativo civile penale contrattuale extracontrattuale "
                    "professionale giuridica giuridico legale generale diretta indiretta pubblica pubblico privata "
                    "privato")
_LATIN_RE = re.compile(r"\b(?:[a-z]+(?:us|um|ae|entia|antia|alis|ilis|ionis)|in re ipsa|culpa|neminem|laedere|"
                       r"actio|ex contractu|ex delicto|una tantum)\b", re.IGNORECASE)


def enabled() -> bool:
    return feature_flags.enabled(FLAG)


def resolver_v2() -> bool:
    return enabled() and feature_flags.enabled(RESOLVER_V2_FLAG)


def generic_name(name: str) -> Optional[str]:
    """Why a name is not a specific institution ("latin", "generic"), or None."""
    from . import fact_resolver
    low = " ".join((name or "").lower().split())
    if _LATIN_RE.search(low):
        return "latin"
    specific = fact_resolver.specific_words(low)
    head = (re.findall(r"[a-zà-ù]+", low) or [""])[0]
    if not specific or (head in GENERIC_HEADS and not specific - fact_resolver._stems(BROAD_QUALIFIERS)):
        return "generic"
    return None


def every_fact() -> bool:
    return enabled() and feature_flags.enabled(EVERY_FACT_FLAG)


def cap(case_text: str, frame: Optional[Dict[str, Any]]) -> int:
    """How many institutions the call may add: MAX_INSTITUTIONS; FACT_ISSUES_EVERY_FACT_ENABLED: one per frame fact
    or one more per CHARS_PER_EXTRA characters past the first, whichever is more, at most MAX_WIDE."""
    if not every_fact():
        return MAX_INSTITUTIONS
    facts = len([f for f in (frame or {}).get("facts") or [] if f])
    longer = MAX_INSTITUTIONS + max(0, len(case_text or "") - CHARS_PER_EXTRA) // CHARS_PER_EXTRA
    return max(MAX_INSTITUTIONS, min(MAX_WIDE, max(facts, longer)))


def system_for(limit: int) -> str:
    system = SYSTEM.replace(_ONE_CAP_RULE, EVERY_FACT_RULE.format(cap=limit)) if every_fact() else SYSTEM
    return system.replace(" Rispondi SOLO", NAMING_RULE + " Rispondi SOLO", 1) if resolver_v2() else system


def _plain(value: Any) -> str:
    return " ".join(value.split()) if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)


def fact_lines(frame: Optional[Dict[str, Any]], signals: Optional[Dict[str, Any]], limit: int = 40) -> List[str]:
    """The facts the call reads: the frame's, then the typed ones (fact_signals' facts: a dict of typed lists),
    each once."""
    lines = [f"- {_plain(f)}" for f in (frame or {}).get("facts") or [] if f]
    typed = (signals or {}).get("facts")
    if isinstance(typed, dict):
        for key, value in typed.items():
            for item in value if isinstance(value, list) else [value]:
                if item in (None, "", [], {}):
                    continue
                if isinstance(item, dict):
                    item = {k: v for k, v in item.items() if v not in (None, "", [], {}) and not str(k).endswith("_iso")}
                lines.append(f"- {key}: {_plain(item)}")
    return list(dict.fromkeys(lines))[:limit]


def clean(raw: Any, have: List[str], limit: int = MAX_INSTITUTIONS, by_fact: bool = False) -> List[Dict[str, Any]]:
    """The call's institutions as central issues: named, without an article number, not already among `have` (the
    frame's own names), each once, at most `limit`, in the model's order; by_fact: each fact's first institution
    before any fact's second (then the seconds, in the facts' order)."""
    items = raw.get("istituti") if isinstance(raw, dict) else raw
    seen = {" ".join(h.lower().split()) for h in have if h}
    out = []
    if by_fact:
        rounds: Dict[str, List[Any]] = {}
        for item in items if isinstance(items, list) else []:
            fact = " ".join(str(item.get("fatto") or "").lower().split()) if isinstance(item, dict) else ""
            rounds.setdefault(fact, []).append(item)
        depth = max((len(v) for v in rounds.values()), default=0)
        items = [group[k] for k in range(depth) for group in rounds.values() if k < len(group)]
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        nome = " ".join(str(item.get("nome") or "").split())
        key = nome.lower()
        if not nome or _NUMBER_RE.search(nome) or key in seen:
            continue
        seen.add(key)
        area = str(item.get("area") or "").strip().lower()
        out.append({"nome": nome, "codice": " ".join(str(item.get("codice") or "").split()), "fonte": FONTE,
                    "fatto": " ".join(str(item.get("fatto") or "").split())[:300],
                    "area": area if area in _AREAS else "altro",
                    "ruolo": " ".join(str(item.get("ruolo_cliente") or "").split())[:120]})
        if len(out) >= limit:
            break
    return out


def fact_issues(case_text: str, frame: Optional[Dict[str, Any]], signals: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The institutions the decisive facts raise (clean), [] on any failure."""
    from . import defensive_generation as dg
    facts = fact_lines(frame, signals)
    human = f"CASO:\n{case_text[:8000]}\n\nFATTI:\n" + ("\n".join(facts) or "(nessuno estratto: leggili dal caso)")
    limit = cap(case_text, frame)
    t0 = time.time()
    try:
        raw = dg._chat_json([SystemMessage(content=system_for(limit)), HumanMessage(content=human)],
                            max_tokens=max(1200, 150 * limit), temperature=0)
    except Exception as exc:
        logger.warning("fact issues failed, the frame is kept: %s", exc)
        return []
    finally:
        vlog("case_analysis_fact_issues_llm", duration_ms=(time.time() - t0) * 1000)
    have = [c.get("nome") or "" for c in (frame or {}).get("central_issue_candidates") or [] if isinstance(c, dict)]
    if resolver_v2():
        raw = _renamed(raw, human, limit)
    return clean(raw, have, limit, by_fact=every_fact())


def _items(raw: Any) -> List[Dict[str, Any]]:
    items = raw.get("istituti") if isinstance(raw, dict) else raw
    return [i for i in items if isinstance(i, dict)] if isinstance(items, list) else []


def _renamed(raw: Any, human: str, limit: int) -> Dict[str, Any]:
    """FACT_ISSUES_RESOLVER_V2_ENABLED: the reply with each generic or Latin name (generic_name) sent back once to be
    renamed, in its place; a name still generic after the retry, or with the retry failing, dropped. Logged as
    case_analysis_fact_issues_names."""
    from . import defensive_generation as dg
    items = _items(raw)
    bad = [(k, i, generic_name(str(i.get("nome") or ""))) for k, i in enumerate(items)]
    bad = [(k, i, why) for k, i, why in bad if why]
    if not bad:
        return {"istituti": items}
    listing = "\n".join(f"- {i.get('nome')} (fatto: {i.get('fatto')}): {'in latino' if why == 'latin' else 'generico'}"
                        for _, i, why in bad)
    retry = (f"{human}\n\nLa tua risposta:\n{json.dumps({'istituti': items}, ensure_ascii=False)}\n\n{RETRY_TITLE} "
             f"(generici o in latino):\n{listing}\n\nPer ciascuno di questi fatti nomina l'istituto preciso, con i "
             "termini della rubrica del codice o della legge. Rispondi SOLO con un oggetto JSON {\"istituti\": [...]} "
             "con i soli istituti rinominati, nello stesso formato.")
    renamed: List[Dict[str, Any]] = []
    try:
        renamed = _items(dg._chat_json([SystemMessage(content=system_for(limit)), HumanMessage(content=retry)],
                                       max_tokens=max(600, 150 * len(bad)), temperature=0))
    except Exception as exc:
        logger.warning("fact issues: renaming failed, the generic names dropped: %s", exc)
    good_new = [i for i in renamed if not generic_name(str(i.get("nome") or ""))]
    left = list(range(len(good_new)))           # the renamed ones not yet placed, in the reply's order

    def take(fact: str) -> Optional[Dict[str, Any]]:
        same = [n for n in left if " ".join(str(good_new[n].get("fatto") or "").lower().split()) == fact]
        pick = same[0] if same else (left[0] if left else None)
        if pick is None:
            return None
        left.remove(pick)
        return good_new[pick]
    bad_at = {k for k, _, _ in bad}
    out, dropped = [], []
    for k, item in enumerate(items):
        new = item if k not in bad_at else take(" ".join(str(item.get("fatto") or "").lower().split()))
        if new is not None:
            out.append(new)
        else:
            dropped.append({"nome": item.get("nome"), "fatto": item.get("fatto")})
    vlog("case_analysis_fact_issues_names", {
        "generic": [{"nome": i.get("nome"), "fatto": i.get("fatto"), "why": why} for _, i, why in bad],
        "dropped": dropped,
        "renamed": [i.get("nome") for i in good_new], "still_generic": [i.get("nome") for i in renamed
                                                                         if generic_name(str(i.get("nome") or ""))]})
    return {"istituti": out}


def add_to_frame(frame: Optional[Dict[str, Any]], case_text: str,
                 signals: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """FACT_ISSUES_ENABLED: the frame with the facts' institutions among its central issues (unchanged with the flag
    off, without a frame, or when the call names none). Logged as case_analysis_fact_issues."""
    if not enabled() or not frame:
        return frame
    added = fact_issues(case_text, frame, signals)
    vlog("case_analysis_fact_issues", {"institutions": added, "facts_read": len(fact_lines(frame, signals)),
                                       "cap": cap(case_text, frame), "every_fact": every_fact()})
    if not added:
        return frame
    return {**frame, "central_issue_candidates": list(frame.get("central_issue_candidates") or []) + added}
