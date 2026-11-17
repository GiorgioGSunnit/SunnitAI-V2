"""v70: the lawyer's method as protocol (LAWYER_METHOD, default false). On, the single-shot parere is written by his
method (docs/lawyer/metodo.md; data/lawyer_method.json holds his words verbatim: scripts/lawyer_method_import.py) and
laid out in the ten sections of his STRUTTURA DELL'OUTPUT, replacing ours (the keys are the code's):

  sintesi       «Sintesi della soluzione» (max 5 righe): v72, the selection's main line and the record's assessment
  fatti         facts_split() (his Fase 1): the case's sentences «certi» (narrated as fact) and «ipotizzabili» (alleged,
                hedged, assumed), the «mancanti» -- every element still to prove -- each with what it would change
  questione     his Fase 2: the method plan's, else the main record's
  norme         «Norme e analisi per elementi» (his Fasi 3-4): our «Norme applicate ai fatti» without the «requisiti
                provati N su M» lines, then per article element / fatto che lo integra / esito (subsumption_table), the
                circumstances as records too (circumstance_records, graded against the sentences), the aggravante check
                always run for the charged offence (NO_AGGRAVANTE when none)
  orientamenti  his Fase 5: a ruling only when it is in the retrieved context (ruling_ok), else «orientamento da
                verificare» -- never an invented citation
  soluzioni     his Fase 6: a table (tesi, pro, contro, realismo); v71, each thesis checked by the element ids it
                declares (coherent_solutions): none declared -- unverifiable; contesting an «integrato» element --
                allowed only declared minoritaria (realismo basso), else barred; a minority thesis leads only without a
                sustainable non-minority line (the containment line is one when the charge is fully integrato); the strategy: main line, subordinate
                (the imputato's riti alternativi and sentencing: riti_lines), the stragiudiziale option, always
  obiezioni     his Fase 6: the three strongest objections, each answered
  processuali   his Fase 8: our computed section, the ratione temporis line (temporal) and the litisconsorzio lines
  conclusioni   his Fasi 9-10: the record's assessment first (assessment(): esito probabile, grado di certezza and what it
                depends on, obiettivo realistico -- esito pieno / parziale / contenimento --, rischio residuo), then the
                next steps with their purpose; the coherence check (coherent_text) before: no sentence denies an
                element marked «provato», none promises an outcome the record does not give
  verificare    every «da verificare» marker taken out of the body and collected here (collect_points)

v72: the theses are enumerated by the code (thesis_candidates: per-element contests, riqualificazioni with a sound
record, scriminanti and attenuanti with facts, the containment line) with their ids and labels; the thesis call only
argues them (pro, contro, realismo, argomento), best of SAMPLES replies scored by the code (score_sample); the summary,
the conclusions and the next steps follow from the selection (conclusions_from_selection, steps_from_selection), never
from the model's plan; three objections always (thesis_objections the last resort); the lawyer's notes di inquadramento
(article_notes) reach the records, the norms writer and the thesis call.

Every writer call carries his REGOLE DI ONESTÀ, his FASE 7, his STILE and the modules of the case's ambiti
(writer_rules_text, through the contextvar begin() sets); the thesis selection (method_plan) reads his whole method.
His CONTROLLO FINALE is the coherence checklist: final_check() answers each item from the code's own checks (logged).
The thesis selection and the writer run with thinking on under LLM_THINKING=per_call; their two durations are logged.
structure_checks() is the nightly's yardstick on an answer's shape (scripts/nightly_sim.py).
"""

import contextlib
import contextvars
import json
import logging
import os
import re
import time
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

FLAG = "LAWYER_METHOD"
DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data",
                         "lawyer_method.json")
METHOD_TOKENS = 3200            # v72: room enough that a cut never loses a key (v73: the objections have their own call)
CIRCUMSTANCES_SECONDS = 30.0
FACTS_MAX = {"certi": 12, "ipotizzabili": 8, "mancanti": 8}
OBJECTIONS = 3
FACT_CHARS = 220

PROVATO, DA_PROVARE, NON_RISULTA = "provato", "da provare", "non risulta"
NOT_RELEVANT = "non pertinente"
FAVOREVOLE, INCERTO, SFAVOREVOLE = "favorevole", "incerto", "sfavorevole"
PIENO, PARZIALE, CONTENIMENTO = "esito pieno", "esito parziale", "contenimento"
ALTO, MEDIO, BASSO = "alto", "medio", "basso"
REALISMO = (ALTO, MEDIO, BASSO)
ACCUSED = ("imputato", "convenuto")

_DATA: Dict[str, Any] = {}


def enabled() -> bool:
    from . import feature_flags
    return feature_flags.enabled(FLAG)


def data() -> Dict[str, Any]:
    """data/lawyer_method.json (read once; {} when missing or unreadable, logged)."""
    if "d" not in _DATA:
        try:
            with open(DATA_FILE, encoding="utf-8") as f:
                _DATA["d"] = json.load(f)
        except (OSError, ValueError) as exc:
            logger.warning("lawyer_method: %s not read: %s", DATA_FILE, exc)
            _DATA["d"] = {}
    return _DATA["d"]


def titles() -> List[str]:
    return [str(s.get("title") or s.get("key")) for s in data().get("sections") or []]


def title(key: str) -> str:
    return next((str(s.get("title")) for s in data().get("sections") or [] if s.get("key") == key), key)


def _log(event: str, payload: Any, duration_ms: Optional[float] = None) -> None:
    try:
        from .verbose_logger import vlog
        vlog(f"lawyer_method_{event}", payload, duration_ms)
    except Exception:           # noqa: BLE001 -- logging never fails the answer
        pass


def _plain(text: Any) -> str:
    return " ".join(str(text or "").split())


def _short(text: Any, limit: int = FACT_CHARS) -> str:
    text = _plain(text)
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def _cell(text: Any) -> str:
    """A markdown table cell: one line, no pipe."""
    return _plain(text).replace("|", "/") or "—"


# --- (2) the honesty rules and the domain modules, in every writer call ---------------------------------------------------

# his headings, as his text gives them
HONESTY_TITLE = "REGOLE DI ONESTÀ"
STYLE_TITLE = "STILE"
METHOD_TITLE = "METODO (da seguire nell'ordine, in modo vincolante)"
STRUCTURE_TITLE = "STRUTTURA DELL'OUTPUT"
FINAL_CHECK_TITLE = "CONTROLLO FINALE (prima di rispondere)"
WRITER_FASI = (7,)                    # the Fasi every writer call carries: the confusions to avoid
ROLE_SLOT = "[diritto penale / civile / amministrativo / altro]"
ROLE_BY_RITO = {"penale": "diritto penale", "penale_gdp": "diritto penale", "civile": "diritto civile",
                "famiglia": "diritto civile", "esecuzione": "diritto civile", "lavoro": "diritto del lavoro",
                "amministrativo": "diritto amministrativo"}
_RULES: "contextvars.ContextVar[str]" = contextvars.ContextVar("lawyer_method_rules", default="")


def modules_for(rito: str = "", frame_areas: Sequence[str] = ()) -> List[str]:
    """The modules of the case's ambiti (his «Uso»: the general prompt plus every relevant module): the rito's, then
    the frame's areas'; a civil rito with no area naming its matter, the general civil module."""
    d = data()
    out = list((d.get("modules_by_rito") or {}).get(str(rito or "")) or [])
    by_area = d.get("modules_by_area") or {}
    for a in frame_areas or ():
        out += by_area.get(str(a), [])
    if rito == "civile" and not any(by_area.get(str(a)) for a in frame_areas or ()):
        out += d.get("civil_default") or []
    return [m for m in dict.fromkeys(out) if m in (d.get("modules") or {})]


def _bullets(lines: Sequence[str]) -> str:
    return "\n".join(x if x.endswith(":") else f"- {x}" for x in lines)


def _fase(f: Dict[str, Any]) -> str:
    return f"FASE {f['n']} - {f['title']}\n{_bullets(f.get('lines') or [])}"


def modules_text(rito: str = "", frame_areas: Sequence[str] = ()) -> str:
    modules = data().get("modules") or {}
    return "\n\n".join(f"{modules[k]['title']}\n{_bullets(modules[k].get('lines') or [])}"
                         for k in modules_for(rito, frame_areas))


def writer_rules_text(rito: str = "", frame_areas: Sequence[str] = ()) -> str:
    """His rules for every writer call: his REGOLE DI ONESTÀ, his FASE 7, his STILE, then the modules of the case's
    ambiti -- his words, as data/lawyer_method.json holds them."""
    d = data()
    parts = []
    if d.get("honesty_rules"):
        parts.append(f"{HONESTY_TITLE}\n{_bullets(d['honesty_rules'])}")
    parts += [_fase(f) for f in d.get("fasi") or [] if f.get("n") in WRITER_FASI]
    if d.get("stile"):
        parts.append(f"{STYLE_TITLE}\n{d['stile']}")
    modules = modules_text(rito, frame_areas)
    if modules:
        parts.append(modules)
    return "\n\n".join(parts)


def role_text(rito: str = "") -> str:
    """His RUOLO, the matter named for the posture's rito."""
    return str(data().get("ruolo") or "").replace(ROLE_SLOT, ROLE_BY_RITO.get(str(rito or ""), "diritto"))


def method_text(rito: str = "", frame_areas: Sequence[str] = ()) -> str:
    """His whole method for the thesis selection: RUOLO, METODO (Fasi 1-10), STRUTTURA DELL'OUTPUT, STILE, REGOLE DI
    ONESTÀ, CONTROLLO FINALE, the modules of the case's ambiti."""
    d = data()
    structure = "\n".join(f"{i}. {s['title']}" + (f" ({s['note']})" if s.get("note") else "")
                           for i, s in enumerate(d.get("sections") or [], 1))
    parts = [role_text(rito), METHOD_TITLE + "\n\n" + "\n\n".join(_fase(f) for f in d.get("fasi") or []),
             f"{STRUCTURE_TITLE}\n{structure}", f"{STYLE_TITLE}\n{d.get('stile') or ''}",
             f"{HONESTY_TITLE}\n{_bullets(d.get('honesty_rules') or [])}",
             f"{FINAL_CHECK_TITLE}\n" + "\n".join(f"[ ] {x}" for x in d.get("controllo_finale") or []),
             modules_text(rito, frame_areas)]
    return "\n\n".join(p for p in parts if p.strip())


@contextlib.contextmanager
def active(rito: str = "", frame_areas: Sequence[str] = ()):
    """While the answer is written: plan_then_write.write_sections adds current_rules() to every section's system."""
    token = _RULES.set(writer_rules_text(rito, frame_areas))
    try:
        yield
    finally:
        _RULES.reset(token)


def current_rules() -> str:
    return _RULES.get()


@contextlib.contextmanager
def scope():
    """Whatever begin() sets inside is undone on exit (interactive_flow.write_answer: the answer's whole run)."""
    token = _RULES.set(_RULES.get())
    try:
        yield
    finally:
        _RULES.reset(token)


def begin(rito: str = "", frame_areas: Sequence[str] = ()) -> None:
    """The rules on, from here to the end of the enclosing scope() -- once the posture is known."""
    _RULES.set(writer_rules_text(rito, frame_areas))


def end() -> None:
    """v75: the rules off -- the method refused the answer (placeholder_main): the standard body is written as before."""
    _RULES.set("")


# --- Fase 1: the facts ----------------------------------------------------------------------------------------------------

CERTAIN = ("accertato",)
HOW = {"allegato": "allegazione", "ipotizzato": "ipotesi", "assunto": "assunto"}


def _live(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The record's elements as the reader gets them: an unmet one outside the anchor comma («non pertinente»,
    issue_records.element_status) left out."""
    from . import issue_records
    return [r for r in rec.get("requisites") or [] if issue_records.element_status(rec, r) != NOT_RELEVANT]


def changes(label: str, rec: Dict[str, Any], req: Dict[str, Any], penal: bool) -> str:
    """What a missing fact would change, by the element's id: for the charge or the claim against the client, whether
    that element of the article stands; else whether the client's norm applies."""
    head = f"l'elemento {req.get('id')} dell'art. {label} («{_short(req.get('text'), 90)}»)"
    if rec.get("side") == "threat":
        return (f"{head}: se provato è integrato, se escluso l'addebito su questa norma cade" if penal else
                f"{head}: se provato la domanda avversaria si regge, se escluso cade")
    return f"{head}: decide se si applica la norma"


# v71: the facts are the narrative's -- never the ask (the user's request, the candidate's instructions, a question) nor
# an article's text
_ASK_RE = re.compile(r"\?\s*$|\bil candidato\b|\bsi chiede\b|\bsi richiede\b|\bdelinei?\b|\bredig\w*|"
                     r"\bpremess[oi]\b[^.]{0,40}\bcenni\b|\bcenni\b|\bassisto\b|\bassist\w+ (?:la|il) client\w*|"
                     r"\bsono (?:il|la) (?:difensor|legal|avvocat)\w*|\bassunte le vesti\b|\bstrategia difensiva\b|"
                     r"\bquali? (?:azion|strategi|rimed|tutel)\w*", re.I)
_ARTICLE_LEAD_RE = re.compile(r"^\W*(?:l['’]\s*)?art(?:icolo|\.)\s*\d+", re.I)


def narrative(sentence: Dict[str, Any], article_texts: Sequence[str] = ()) -> bool:
    """A sentence of the case's narrative: not the system's, not the ask (fact_signals.user_message, _ASK_RE), not an
    article's text (one opening with «art. N», or found in an article's text)."""
    from . import fact_signals
    text = _plain(sentence.get("text"))
    if not text or sentence.get("source") == "sistema":
        return False
    if fact_signals.user_message(text) or _ASK_RE.search(text) or _ARTICLE_LEAD_RE.match(text):
        return False
    low = text.lower().rstrip(".")
    return not (len(low) >= 40 and any(low in _plain(t).lower() for t in article_texts))


def facts_split(sentences: Sequence[Dict[str, Any]], records: Dict[str, Dict[str, Any]], penal: bool = True,
                questions: Sequence[Dict[str, Any]] = (), article_texts: Sequence[str] = (),
                log: Optional[Dict[str, Any]] = None) -> Dict[str, List[Any]]:
    """{"certi": [text], "ipotizzabili": [{"text", "how"}], "mancanti": [{"fatto", "cambierebbe", "elemento", "article"}]}:
    the narrative's sentences (narrative()) by their certainty; as missing, the elements «da provare» and the open
    questions linked to an element -- each naming the element id it would change; a missing fact with no element is
    left out (logged)."""
    certi, ipotesi, excluded = [], [], []
    for s_ in sentences or []:
        if not narrative(s_, article_texts):
            if _plain(s_.get("text")) and s_.get("source") != "sistema":
                excluded.append(_short(s_.get("text"), 80))
            continue
        if s_.get("certainty") in CERTAIN:
            certi.append(_short(s_["text"]))
        else:
            ipotesi.append({"text": _short(s_["text"]), "how": HOW.get(str(s_.get("certainty") or ""), "ipotesi")})
    by_question: Dict[str, List[Tuple[str, Dict[str, Any], Dict[str, Any]]]] = {}
    mancanti, seen = [], set()
    for label, rec in (records or {}).items():
        for r in _live(rec):
            if r.get("question"):
                by_question.setdefault(str(r["question"]), []).append((label, rec, r))
            if r.get("status") != DA_PROVARE or not r.get("id"):
                continue
            what = _plain(r.get("missing")) or _plain(r.get("text"))
            key = what.lower()[:80]
            if not what or key in seen:
                continue
            seen.add(key)
            mancanti.append({"fatto": _short(what, 160), "cambierebbe": changes(label, rec, r, penal),
                             "elemento": r["id"], "article": label})
    unlinked = []
    for q in questions or []:
        what = _plain(q.get("point") or q.get("text"))
        linked = by_question.get(str(q.get("id") or ""))
        if not what:
            continue
        if not linked:
            unlinked.append(_short(what, 80))
            continue
        label, rec, r = linked[0]
        key = what.lower()[:80]
        if key not in seen and not any(m["elemento"] == r.get("id") for m in mancanti):
            seen.add(key)
            mancanti.append({"fatto": _short(what, 160), "cambierebbe": changes(label, rec, r, penal),
                             "elemento": r.get("id"), "article": label})
    if log is not None:
        log.update({"excluded_sentences": excluded, "questions_without_element": unlinked})
    return {"certi": certi[:FACTS_MAX["certi"]], "ipotizzabili": ipotesi[:FACTS_MAX["ipotizzabili"]],
            "mancanti": mancanti[:FACTS_MAX["mancanti"]]}


def facts_section(split: Dict[str, List[Any]]) -> str:
    out = ["**Fatti certi**", ""]
    out += [f"- {x}" for x in split.get("certi") or []] or ["- Nessun fatto narrato come certo."]
    out += ["", "**Fatti ipotizzabili**", ""]
    out += [f"- {x['text']} ({x['how']})" for x in split.get("ipotizzabili") or []] or ["- Nessuno."]
    out += ["", "**Fatti mancanti**", ""]
    out += [f"- {x['fatto']} — cambierebbe: {x['cambierebbe']}." for x in split.get("mancanti") or []] or [
        "- Nessun elemento resta da provare sui fatti narrati."]
    return "\n".join(out)


# --- Fase 4: the subsumption, element by element --------------------------------------------------------------------------

ESITO = {PROVATO: "integrato", DA_PROVARE: "da provare", NON_RISULTA: "escluso"}


def _sentence_text(ids: Sequence[str], by_id: Dict[str, Dict[str, Any]]) -> str:
    texts = [_short(by_id[i].get("text"), 160) for i in ids or [] if i in by_id]
    return " / ".join(texts[:2])


def subsumption_rows(rec: Dict[str, Any], by_id: Dict[str, Dict[str, Any]]) -> List[Tuple[str, str, str]]:
    """[(elemento, fatto che lo integra, esito)] of the record's live elements."""
    out = []
    for r in _live(rec):
        status = r.get("status") or DA_PROVARE
        if status == PROVATO:
            fact = _sentence_text(r.get("supporting") or [], by_id) or "i fatti estratti dal caso"
        elif status == NON_RISULTA:
            fact = _sentence_text(r.get("contradicting") or [], by_id) or "i fatti estratti lo escludono"
        else:
            fact = "nessun fatto del caso" + (f": manca {_short(r.get('missing'), 120)}" if r.get("missing") else "")
        out.append((_short(r.get("text"), 160), fact, ESITO.get(status, status)))
    return out


CHARGED_PROSE = {   # my drafts (v75b): the charged article's paragraph, from its sussunzione rows only
    "head": "Sui fatti narrati, degli elementi dell'art. {art}",
    PROVATO: "risultano provati {items}",
    DA_PROVARE: "restano da provare {items}",
    NON_RISULTA: "risultano esclusi {items}",
    "tail": "(v. l'analisi per elementi)",
}


def charged_prose(records: Dict[str, Dict[str, Any]], by_id: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """v75b: {label: paragraph} of each charged fattispecie -- what its sussunzione rows say (each element with its
    id, by its status), nothing else: the writer's paragraph on the charged article is replaced by it
    (charged_blocks)."""
    out = {}
    for label in _charged(records):
        live = _live(records[label])
        if not live:
            continue
        parts = []
        for status in (PROVATO, DA_PROVARE, NON_RISULTA):
            items = [f"«{_short(r.get('text'), 90)}» ({r.get('id')})" for r in live if (r.get("status") or DA_PROVARE)
                     == status]
            if items:
                parts.append(CHARGED_PROSE[status].format(items=", ".join(items)))
        out[label] = f"{CHARGED_PROSE['head'].format(art=label)} {'; '.join(parts)} {CHARGED_PROSE['tail']}."
    return out


_ART_HEAD_RE = re.compile(r"^\s*\*\*\s*(?:Art(?:icolo)?\.?|art\.)\s")


def charged_blocks(norms: str, prose_by_label: Dict[str, str], log: Optional[List[str]] = None) -> str:
    """v75b: the writer's norms with each charged article's block (its «**Art. N …**» heading to the next bold line)
    holding only the code's paragraph from its sussunzione rows (charged_prose)."""
    from . import citations
    if not prose_by_label or not norms:
        return norms
    out: List[str] = []
    skipping = False
    for line in norms.split("\n"):
        if line.strip().startswith("**"):
            skipping = False
            refs = citations.article_refs(line) if _ART_HEAD_RE.match(line) else []
            label = f"{refs[0][0]} {refs[0][1]}" if refs else ""
            if label in prose_by_label:
                out += [line, "", prose_by_label[label]]
                skipping = True
                if log is not None:
                    log.append(label)
                continue
        if not skipping:
            out.append(line)
    return "\n".join(out)


def subsumption_table(label: str, rec: Dict[str, Any], by_id: Dict[str, Dict[str, Any]]) -> str:
    rows = subsumption_rows(rec, by_id)
    if not rows:
        return ""
    rubrica = _plain(rec.get("rubrica"))
    head = f"**Art. {label}**" + (f" — {rubrica}" if rubrica else "")
    lines = [head, "", "| Elemento | Fatto che lo integra | Esito |", "|---|---|---|"]
    lines += [f"| {_cell(e)} | {_cell(f)} | {_cell(x)} |" for e, f, x in rows]
    return "\n".join(lines)


def subsumption_section(records: Dict[str, Dict[str, Any]], order: Sequence[str], by_id: Dict[str, Dict[str, Any]],
                        circumstances: Optional[Dict[str, Dict[str, Any]]] = None, aggravante_note: str = "") -> str:
    """The tables of the counted records, in order (the §3 order, then the others), then the circumstances'."""
    from . import issue_records
    labels = [x for x in order if x in records] + [x for x in records if x not in order]
    tables = [subsumption_table(x, records[x], by_id) for x in labels
              if issue_records.counted(x, str(records[x].get("rubrica") or "")) and not records[x].get("applicazione")]
    out = [t for t in tables if t]
    circ = [subsumption_table(x, rec, by_id) for x, rec in (circumstances or {}).items()]
    circ = [t for t in circ if t]
    if circ or aggravante_note:
        out.append("**Circostanze**")
        out += circ
        if aggravante_note:
            out.append(aggravante_note)
    return "\n\n".join(out) if out else "Nessuna norma con elementi da sussumere nel materiale recuperato."


# --- circumstances as fattispecie -----------------------------------------------------------------------------------------

NO_AGGRAVANTE = "Nessuna aggravante risulta integrata dai fatti narrati."
AGGRAVANTI_CHECKED = "Aggravanti verificate per il reato contestato: {items}."
CIRC_PREFIX = "C"


def _charged(records: Dict[str, Dict[str, Any]]) -> List[str]:
    out = sorted(((rec.get("charged"), label) for label, rec in (records or {}).items() if rec.get("charged")),
                 key=lambda x: x[0])
    return [label for _, label in out]


def circumstance_candidates(case: str, records: Dict[str, Dict[str, Any]], posture: Dict[str, Any]
                            ) -> List[Dict[str, Any]]:
    """The table's circumstances to grade (penal postures only): every aggravante for the charged offence -- named by the
    case, matched by its patterns or tied to the charged article --, and the attenuanti the case's words point to or the
    imputato's posture always considers (62-bis)."""
    family = str((posture or {}).get("family") or "")
    if family and family != "penale":
        return []
    low = (case or "").lower()
    charged = set(_charged(records))
    out = []
    for c in data().get("circumstances") or []:
        hit = any(re.search(p, low) for p in c.get("patterns") or [])
        tied = bool(charged & set(c.get("offences") or []))
        if c.get("kind") == "aggravante" and (hit or tied) and (charged or hit):
            out.append(c)
        elif c.get("kind") == "attenuante" and (hit or (c["label"] == "62-bis c.p." and posture.get("role") == "imputato")):
            out.append(c)
    return out


def circumstance_records(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """{label: record} -- each circumstance a record whose requisites are its elements, all «da provare» until linked."""
    out, n = {}, 0
    for c in candidates:
        reqs = []
        for text in c.get("elements") or []:
            n += 1
            reqs.append({"id": f"{CIRC_PREFIX}{n}", "comma": 1, "text": text, "kind": "requisito", "status": DA_PROVARE,
                         "facts_status": DA_PROVARE, "supporting": [], "contradicting": [], "missing": ""})
        out[c["label"]] = {"article": c["label"], "rubrica": f"{c.get('kind')}: {c.get('name')}", "kind": c.get("kind"),
                           "requisites": reqs, "circumstance": True}
    return out


def grade_circumstances(records: Dict[str, Dict[str, Any]], sentences: Sequence[Dict[str, Any]], call=None,
                        seconds: float = CIRCUMSTANCES_SECONDS) -> Dict[str, Any]:
    """In place: the circumstances' elements linked to the case's sentences and settled (issue_records.link, one call)."""
    from . import issue_records
    if not records:
        return {"topics": [], "total_ms": 0}
    log = issue_records.link(records, [{"title": "circostanze del reato"}], [list(records)], list(sentences or []),
                             call=call, seconds=seconds)
    by_id = {s["id"]: s for s in sentences or []}
    for rec in records.values():
        for r in rec.get("requisites") or []:
            issue_records.settle(r, by_id)
        rec["verdict"] = issue_records.verdict(rec)
    return log


def integrated(rec: Dict[str, Any]) -> bool:
    live = _live(rec)
    return bool(live) and all(r.get("status") == PROVATO for r in live)


def aggravante_note(circ: Dict[str, Dict[str, Any]], records: Dict[str, Dict[str, Any]]) -> str:
    """The aggravante check's line, always for a charged offence: the integrated ones, else NO_AGGRAVANTE."""
    if not _charged(records):
        return ""
    aggr = {x: r for x, r in (circ or {}).items() if r.get("kind") == "aggravante"}
    done = [x for x, r in aggr.items() if integrated(r)]
    if not done:
        return NO_AGGRAVANTE
    return AGGRAVANTI_CHECKED.format(items="; ".join(
        f"art. {x}: integrata" if x in done else f"art. {x}: non integrata sui fatti" for x in aggr))


CLAIM_MARK = " (da verificare: i requisiti dell'art. {label}: {elements})"


def _claim_re(c: Dict[str, Any]) -> "re.Pattern":
    """A sentence invoking the circumstance: its article and number («62 n. 6», «art. 62, co. 1, n. 6», «62-bis»), or
    its name."""
    label = str(c["label"]).replace(" c.p.", "")
    base, _, num = label.partition(" n. ")
    base = re.sub(r"\s*co\.\s*\d+", "", base).strip()
    article = re.escape(base) + (r"\s*,?\s*(?:co(?:mma)?\.?\s*\d+\s*,?\s*)?(?:n\.|num\.|numero)\s*" + re.escape(num) + r"\b"
                                 if num else r"(?![\d-])")
    return re.compile(r"\b(?:art\.\s*)?" + article + "|" + re.escape(str(c.get("name") or "~")), re.I)


def circumstance_claims(text: str, circ: Dict[str, Dict[str, Any]], log: Optional[List[Dict[str, Any]]] = None) -> str:
    """A sentence invoking an attenuante or an aggravante whose elements the record does not all prove -- or that has
    no record -- gets its open requisites as a point to verify (CLAIM_MARK): a circumstance is never claimed without
    its conditions (62 n. 6 without the integral reparation before the trial, 62 n. 2 without the fatto ingiusto)."""
    table = {c["label"]: c for c in data().get("circumstances") or []}
    out = []
    for para in (text or "").split("\n"):
        sentences = []
        for sentence in _SENT_SPLIT.split(para):
            for label, c in table.items():
                if "da verificare" in sentence or not _claim_re(c).search(sentence):
                    continue
                rec = (circ or {}).get(label)
                if rec and integrated(rec):
                    continue
                open_ = [_short(r.get("text"), 90) for r in _live(rec) if r.get("status") != PROVATO] if rec \
                    else [_short(e, 90) for e in c.get("elements") or []]
                if not open_:
                    continue
                mark = CLAIM_MARK.format(label=label, elements="; ".join(open_))
                sentence = sentence.rstrip()
                if _ABBREV_END_RE.search(sentence):          # «… c.p.»: the code's dot stays
                    sentence = f"{sentence}{mark}."
                else:
                    sentence = re.sub(r"([.;:!?]?)$", lambda m: mark + (m.group(1) or "."), sentence, count=1)
                if log is not None:
                    log.append({"circumstance": label, "open": open_})
                break
            sentences.append(sentence)
        out.append(" ".join(sentences))
    return "\n".join(out)


# --- the probability, from the record (point 7) ---------------------------------------------------------------------------

def _counts(rec: Dict[str, Any]) -> Tuple[int, int, int]:
    live = _live(rec)
    return (sum(1 for r in live if r.get("status") == PROVATO), sum(1 for r in live if r.get("status") == NON_RISULTA),
            len(live))


def main_threat(records: Dict[str, Dict[str, Any]]) -> str:
    from . import issue_records
    charged = [x for x in _charged(records) if records[x].get("side") == "threat" and _live(records[x])
               and issue_records.counted(x, str(records[x].get("rubrica") or ""))]      # v74: a fattispecie only
    if charged:
        return charged[0]
    return next((x for x, r in records.items() if r.get("side") == "threat" and _live(r)
                 and issue_records.counted(x, str(r.get("rubrica") or ""))), "")


HYPOTHESIS_LABEL = "ipotesi di reato (addebito non indicato nel quesito)"     # v79 (C4), the PO's words
NO_CHARGE_WHY = {"penale": "l'addebito non è indicato nel quesito e i fatti narrati non ne individuano uno",
                 "civile": "la domanda avversaria non è indicata nel quesito e i fatti narrati non ne individuano una"}


def assessment(records: Dict[str, Dict[str, Any]], posture: Dict[str, Any], main: str = "", client: str = ""
               ) -> Dict[str, Any]:
    """v79 (C4): a charge the first read's hypothesis gave (rec["charge_hypothesis"]) is labelled HYPOTHESIS_LABEL in
    the why -- §1 and §9 print it --; no charge at all is said in the reader's words (NO_CHARGE_WHY), never «il
    registro non contiene …»."""
    a = _assessment(records, posture, main, client)
    rec = (records or {}).get(a.get("article") or "") or {}
    if rec.get("charge_hypothesis"):
        a = {**a, "hypothesis": True, "why": f"{HYPOTHESIS_LABEL}: art. {a['article']}; {a['why']}"}
    return a


def _assessment(records: Dict[str, Dict[str, Any]], posture: Dict[str, Any], main: str = "", client: str = ""
                ) -> Dict[str, Any]:
    """{"esito", "grado", "obiettivo", "why", "article"}: the probable outcome read from the record, not from the side
    we write for. Defending (imputato, convenuto): the charge's or the claim's main article -- every element proven:
    unfavourable, the goal is containing it; most proven: unfavourable, a partial outcome (another qualification,
    no aggravante, the attenuanti); an element excluded: favourable, the full outcome. Claiming: the client's main
    article -- applicable: favourable; partly proven: uncertain; nothing proven or excluded: unfavourable."""
    who = client or "la parte assistita"
    role = str((posture or {}).get("role") or "")
    records = records or {}
    if role in ACCUSED:
        label = main_threat(records)
        if not label:                   # v71: no charge on the record -- never «esito pieno»
            return {"esito": INCERTO, "grado": BASSO, "obiettivo": PARZIALE, "article": "",
                    "why": NO_CHARGE_WHY["civile" if (posture or {}).get("family") == "civile" else "penale"]}
        proved, excluded, total = _counts(records[label])
        weapons = [x for x, r in records.items() if r.get("side") == "weapon" and r.get("alternative")
                   and _counts(r)[0] and not _counts(r)[1]]
        if excluded:
            return {"esito": FAVOREVOLE, "grado": MEDIO, "obiettivo": PIENO, "article": label,
                    "why": f"un elemento dell'art. {label} è escluso dai fatti"}
        if total and proved == total:
            return {"esito": SFAVOREVOLE, "grado": ALTO, "obiettivo": PARZIALE if weapons else CONTENIMENTO,
                    "article": label, "all_integrated": True, "alternative": weapons[0] if weapons else "",
                    "why": f"tutti gli elementi dell'art. {label} risultano integrati"
                    + (f"; resta la qualificazione alternativa (art. {weapons[0]})" if weapons else "")}
        if total and proved * 2 >= total:
            return {"esito": SFAVOREVOLE, "grado": MEDIO, "obiettivo": PARZIALE, "article": label,
                    "why": f"{proved} elementi su {total} dell'art. {label} risultano provati, gli altri restano da "
                           "provare per l'accusa" if (posture or {}).get("family") != "civile" else
                           f"la maggior parte degli elementi dell'art. {label} risulta provata"}
        return {"esito": INCERTO, "grado": BASSO, "obiettivo": PIENO, "article": label,
                "why": f"la maggior parte degli elementi dell'art. {label} resta da provare"}
    # v78: a victim's outcome is the offence of the querela -- the charge (rec["charged"]), whatever the main topic
    victim_charge = next((x for x in _charged(records) if _live(records[x])), "") if role in VICTIM_ROLES else ""
    from . import issue_records
    label = victim_charge or (main if main in records else next((x for x, r in records.items()
                                                                if r.get("side") != "threat" and _live(r)
                                                                and not issue_records.defence_article(x)), ""))
    if not label:
        return {"esito": INCERTO, "grado": BASSO, "obiettivo": PARZIALE, "article": "",
                "why": f"i fatti narrati non indicano la norma su cui {who} fonda la pretesa"}
    proved, excluded, total = _counts(records[label])
    if excluded:
        return {"esito": SFAVOREVOLE, "grado": MEDIO, "obiettivo": PARZIALE, "article": label,
                "why": f"un elemento dell'art. {label} è escluso dai fatti"}
    if total and proved == total:
        return {"esito": FAVOREVOLE, "grado": ALTO if total > 1 else MEDIO, "obiettivo": PIENO, "article": label,
                "why": f"tutti gli elementi dell'art. {label} risultano provati"}
    if proved:
        return {"esito": INCERTO, "grado": MEDIO, "obiettivo": PIENO, "article": label,
                "why": f"{proved} elementi su {total} dell'art. {label} risultano provati; gli altri vanno provati"}
    return {"esito": INCERTO, "grado": BASSO, "obiettivo": PARZIALE, "article": label,
            "why": f"nessun elemento dell'art. {label} risulta ancora provato"}


def settle_objective(a: Dict[str, Any], sols: Sequence[Dict[str, Any]], records: Dict[str, Dict[str, Any]],
                     circ: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """v71: every element of the charge integrato -- «esito parziale» when a sustainable alternative exists (an
    alternative record the facts support, or a non-minority, declared thesis not barred that rests on another article's
    elements: a minority thesis is no realistic objective), else «contenimento»; any other assessment as it is."""
    if not a.get("all_integrated"):
        return a
    known = elements_by_id(records, circ)
    thesis = next((t for t in sols if not t.get("barred") and not t.get("fallback") and not t.get("minoritaria")
                   and not t.get("unverifiable")
                   and any((known.get(x) or {}).get("article") not in ("", a.get("article"))
                           for x in t.get("poggia_su") or [])), None)
    if a.get("alternative") or thesis:
        why = a["why"] if a.get("alternative") else f"{a['why']}; resta una tesi sostenibile ({thesis['tesi'][:80]})"
        return {**a, "obiettivo": PARZIALE, "why": why}
    return {**a, "obiettivo": CONTENIMENTO}


# his Fase 10: the most probable solution with its grado di certezza and what it depends on; the realistic objective
# and the residual risk (the labels are his words: alto / medio / basso, esito pieno / parziale / contenimento)
ASSESSMENT_LINES = ("**Esito probabile**: {esito_text}: {why}.", "**Grado di certezza**: {grado}{depends}.",
                    "**Obiettivo realistico**: {obiettivo}{goal}.")
RISK_LINE = "**Rischio residuo**: {risk}."
STEPS_TITLE = "**Passi successivi**"


def _tail(text: str) -> str:
    text = _plain(text).rstrip(".")
    return f" — {text[0].lower()}{text[1:]}" if text else ""


def assessment_text(a: Dict[str, Any], client: str = "", goal: str = "", depends: str = "", risk: str = "") -> str:
    who = client or "la parte assistita"
    esito_text = {FAVOREVOLE: f"favorevole a {who}", SFAVOREVOLE: f"sfavorevole a {who}", INCERTO: "incerto"}.get(
        a.get("esito"), str(a.get("esito")))
    values = {"esito_text": esito_text, "why": _plain(a.get("why")).rstrip("."), "grado": a.get("grado"),
              "depends": (" — dipende da: " + _plain(depends).rstrip(".")) if _plain(depends) else "",
              "obiettivo": a.get("obiettivo"), "goal": _tail(goal)}
    lines = [line.format(**values) for line in ASSESSMENT_LINES]
    if _plain(risk):
        lines.append(RISK_LINE.format(risk=_plain(risk).rstrip(".")))
    return "\n".join(lines)


def steps_text(steps: Sequence[Dict[str, str]]) -> str:
    """His Fase 10: the next steps and the proofs or documents to acquire, each with its purpose."""
    def bare(text: str) -> str:         # «… dell'art. 584 c.p.» keeps its abbreviation's dot
        return text if _ABBREV_END_RE.search(text) else text.rstrip(".")
    lines = [f"- {bare(x['passo'])}" + (f" — scopo: {x['scopo'].rstrip('.')}" if x.get("scopo") else "") + "."
             for x in steps or [] if x.get("passo")]
    return f"{STEPS_TITLE}\n\n" + "\n".join(lines) if lines else ""


# --- the imputato's subordinate line: riti alternativi and sentencing -----------------------------------------------------

_QTY = r"(\d+|uno|una|due|tre|quattro|cinque|sei|sette|otto|nove|dieci|undici|dodici|quindici|venti|ventiquattro|trenta)"
_NUM = {"uno": 1, "una": 1, "due": 2, "tre": 3, "quattro": 4, "cinque": 5, "sei": 6, "sette": 7, "otto": 8, "nove": 9,
        "dieci": 10, "undici": 11, "dodici": 12, "quindici": 15, "venti": 20, "ventiquattro": 24, "trenta": 30}
_RANGE_RE = re.compile(r"reclusione da " + _QTY + r"(?: (mesi|anni|giorni))? a " + _QTY + r" (mesi|anni)", re.I)
_UPTO_RE = re.compile(r"reclusione fino a " + _QTY + r" (mesi|anni)", re.I)


def _years(qty: str, unit: str) -> Optional[float]:
    n = int(qty) if qty.isdigit() else _NUM.get(qty.lower())
    if n is None:
        return None
    unit = (unit or "").lower()
    return float(n) if unit.startswith("ann") else n / 12 if unit.startswith("mes") else n / 365


def pena_range(text: str) -> Tuple[Optional[float], Optional[float]]:
    """(minimum, maximum) of the first reclusione the text states, in years; (None, None) when it states none."""
    m = _RANGE_RE.search(text or "")
    if m:
        return _years(m.group(1), m.group(2) or m.group(4)), _years(m.group(3), m.group(4))
    m = _UPTO_RE.search(text or "")
    if m:
        return None, _years(m.group(1), m.group(2))
    return None, None


def riti_lines(posture: Dict[str, Any], charged_text: str = "") -> List[str]:
    """The imputato's subordinate line (penal postures, imputato, before the sentence): every rito of the table, each
    gated on the charged offence's pena where the table says so -- precluded by the pena, a line saying so; the pena
    unknown, the line as the table gives it."""
    if (posture or {}).get("role") != "imputato" or (posture or {}).get("stage") in ("impugnazione", "esecuzione") \
            or _family_key(posture) != "penale":            # v75: the intake rito's, never the frame's area
        return []
    low, high = pena_range(charged_text)
    out = []
    for r in data().get("riti_alternativi") or []:
        line = str(r.get("line") or "")
        cap = r.get("max_edittale_years")
        if cap is not None and high is not None and high > cap:
            out.append(f"{line.split(':')[0]}: preclusa, il massimo edittale del reato contestato supera {int(cap)} "
                       "anni.")
            continue
        limit = r.get("max_years_after_reduction")
        if limit is not None and low is not None and low * 2 / 3 > limit:
            out.append(f"{line.split(':')[0]}: preclusa, il minimo edittale ridotto di un terzo supera {int(limit)} "
                       "anni.")
            continue
        out.append(line)
    return out


# --- ratione temporis -----------------------------------------------------------------------------------------------------

MONTHS = {"gennaio": 1, "febbraio": 2, "marzo": 3, "aprile": 4, "maggio": 5, "giugno": 6, "luglio": 7, "agosto": 8,
          "settembre": 9, "ottobre": 10, "novembre": 11, "dicembre": 12}
_DATE_RES = (
    (re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b"), "dmy"),
    (re.compile(r"\b(\d{1,2})(?:°|º)?\s+(" + "|".join(MONTHS) + r")\s+(?:del\s+)?(\d{4})\b", re.I), "d_month_y"),
    (re.compile(r"\b(?:nel(?:l['’])?\s*|a\s+|di\s+|del\s+|in\s+)?(" + "|".join(MONTHS) + r")\s+(?:del\s+)?(\d{4})\b",
                re.I), "month_y"),
    (re.compile(r"\bnel\s+(\d{4})\b", re.I), "y"),
)
TEMPORAL_LINE = "Disciplina applicabile alla data dei fatti ({when}): {what}."
TEMPORAL_UNKNOWN = ("Disciplina applicabile alla data dei fatti: il caso non indica quando sono avvenuti i fatti, e le "
                    "riforme della materia ({reforms}) ne dipendono.")
TEMPORAL_ABROGATED = " Norme abrogate nel testo vigente: {articles}."
TEMPORAL_POINTS = {"unknown": "La data dei fatti: decide quale disciplina si applica ({reforms})",
                   "after": "La disciplina applicabile ai fatti, anteriori a {reforms}",
                   "abrogated": "La disciplina vigente al tempo dei fatti per le norme abrogate ({articles})"}


def fact_date(case: str) -> Optional[Tuple[date, str]]:
    """The earliest date the case text states, with its words (a month or a year alone: its first day)."""
    found: List[Tuple[date, str]] = []
    for rx, how in _DATE_RES:
        for m in rx.finditer(case or ""):
            try:
                if how == "dmy":
                    d = date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
                elif how == "d_month_y":
                    d = date(int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1)))
                elif how == "month_y":
                    d = date(int(m.group(2)), MONTHS[m.group(1).lower()], 1)
                else:
                    d = date(int(m.group(1)), 1, 1)
            except (ValueError, KeyError):
                continue
            if 1940 <= d.year <= date.today().year + 1:
                words = m.group(0).strip()
                words = re.sub(r"^(?:nel(?:l['’])?\s*|a\s+|di\s+|del\s+|in\s+)", "", words, flags=re.I)
                found.append((d, words))
    return min(found, key=lambda x: x[0]) if found else None


def _family_key(posture: Dict[str, Any]) -> str:
    """v75: the intake rito's family (posture.intake_family) -- never the frame's area."""
    from . import posture as posture_mod
    return posture_mod.intake_family(posture)


def temporal(case: str, posture: Dict[str, Any], abrogated: Sequence[str] = ()) -> Tuple[str, List[str]]:
    """(«Disciplina applicabile alla data dei fatti (…)», its points for «Punti da verificare»): the family's reforms in
    force at the fact date, and those that came after -- which discipline governs is a point (favor rei, art. 2 c.p.;
    procedural acts by the date of the act); the date unknown, the date is the point. ("", []) for a family with no
    reform in the table."""
    family = _family_key(posture)
    reforms = [r for r in data().get("reforms") or [] if r.get("family") == family]
    if not reforms:
        return "", []
    found = fact_date(case)
    points = [TEMPORAL_POINTS["abrogated"].format(articles=", ".join(f"art. {x}" for x in abrogated))] if abrogated else []
    tail = TEMPORAL_ABROGATED.format(articles=", ".join(f"art. {x}" for x in abrogated)) if abrogated else ""
    if not found:
        names = "; ".join(r["name"] for r in reforms)
        return TEMPORAL_UNKNOWN.format(reforms=names) + tail, [TEMPORAL_POINTS["unknown"].format(reforms=names)] + points
    when, words = found
    before = [r for r in reforms if date.fromisoformat(r["from"]) <= when]
    after = [r for r in reforms if date.fromisoformat(r["from"]) > when]
    parts = []
    if before:
        parts.append("si applicano " + "; ".join(f"{r['name']}: {r['note']}" for r in before))
    if after:
        rule = ("secondo la legge più favorevole al reo (art. 2 c.p.) e, per gli atti del processo, la legge del tempo "
                "dell'atto" if family == "penale" else "secondo la disciplina transitoria di ciascuna")
        parts.append("i fatti sono anteriori a " + "; ".join(
            f"{r['name']}, dal {date.fromisoformat(r['from']).strftime('%d.%m.%Y')}" for r in after)
            + f": quale disciplina regoli il caso si stabilisce {rule}")
        points.insert(0, TEMPORAL_POINTS["after"].format(reforms="; ".join(r["name"] for r in after)))
    return TEMPORAL_LINE.format(when=words, what="; ".join(parts)) + tail, points


def temporal_line(case: str, posture: Dict[str, Any], abrogated: Sequence[str] = ()) -> str:
    return temporal(case, posture, abrogated)[0]


# --- litisconsorzio and the stragiudiziale option -------------------------------------------------------------------------

def litisconsorzio_lines(case: str, records: Dict[str, Dict[str, Any]], posture: Dict[str, Any]) -> List[str]:
    """The table's lines for the matters the case or the records name (civil claimant before filing, or unknown cell)."""
    posture = posture or {}
    if posture.get("family") not in ("", None, "civile") or posture.get("role") not in ("", None, "sconosciuto",
                                                                                       "attore", "terzo"):
        return []
    low = (case or "").lower()
    labels = set(records or {})
    out = []
    for m in data().get("litisconsorzio") or []:
        if any(re.search(p, low) for p in m.get("patterns") or []) or labels & set(m.get("articles") or []):
            out.append(str(m["line"]))
    return out


def stragiudiziale_line(posture: Dict[str, Any], charge_procedibilita: str = "") -> str:
    """The table's option for the family and role; v71: for an imputato charged with a d'ufficio offence, the variant
    with no remissione and no 162-ter («penale_imputato_ufficio»)."""
    table = data().get("stragiudiziale") or {}
    family, role = _family_key(posture), str((posture or {}).get("role") or "")
    if family == "penale":
        if role in ("imputato", "sconosciuto", ""):
            key = "penale_imputato_ufficio" if charge_procedibilita == "ufficio" else "penale_imputato"
            return table.get(key) or table.get("penale_imputato") or ""
        return table.get("penale_offesa") or ""
    if not family:              # v75: intake read no rito -- no line from another rito's table
        return ""
    return table.get(family) or (table.get("civile") if family in ("civile", "lavoro") else "") or ""


# --- rulings: only the retrieved ones -------------------------------------------------------------------------------------

_RULING_NO_RE = re.compile(r"(?<![\d/])(\d{1,6})\s*/\s*(\d{4})(?![\d/])")
_RULING_CITE_RE = re.compile(r"\b(?:Cass(?:azione)?\.?|C\.\s*cost\.|Corte\s+(?:cost\w*|di\s+cassazione|d['’]appello)|"
                             r"Cons\.\s*St(?:ato)?\.?|Trib(?:unale)?\.?|S\.U\.|Sez(?:ioni)?\.?\s*Un\w*)[^()\n;]{0,80}?"
                             r"(?:n\.\s*)?\d{1,6}\s*/\s*\d{4}", re.I)
ORIENTATION_TO_VERIFY = "orientamento da verificare"


def ruling_ok(pronuncia: str, allowed: Sequence[str]) -> bool:
    """The ruling is one of the retrieved (allowed): its number/year among theirs, or the same label."""
    text = _plain(pronuncia)
    if not text:
        return False
    allowed = [str(x.get("label") or "") if isinstance(x, dict) else str(x or "") for x in allowed or []]
    numbers = {(int(a), b) for x in allowed for a, b in _RULING_NO_RE.findall(x or "")}
    own = {(int(a), b) for a, b in _RULING_NO_RE.findall(text)}
    if own:
        return bool(own & numbers)
    return any(text.lower() == _plain(x).lower() for x in allowed)


def unretrieved_rulings_out(text: str, allowed: Sequence[str], log: Optional[List[str]] = None) -> str:
    """A ruling cited in prose that is not among the retrieved: replaced by «orientamento da verificare»."""
    def swap(m):
        if ruling_ok(m.group(0), allowed):
            return m.group(0)
        if log is not None:
            log.append(m.group(0))
        return ORIENTATION_TO_VERIFY
    return _RULING_CITE_RE.sub(swap, text or "")


# --- v78: the scope of what the plan's prose cites -------------------------------------------------------------------------
# §6 and §7 are written from the plan after record_check ran on the writer's draft: their sentences are reviewed here
# the same way -- an article outside the answer's scope removes its sentence (record_check's fuori_dal_record) --, and a
# registry ruling is cited only for the article it was fetched for and on the element its passage was graded on

def _norm_label(label: str) -> str:
    from . import citations
    number, _, code = str(label or "").partition(" ")
    return citations.label((number, code)) if code else str(label or "")


def article_scope(ctx: Dict[str, Any], texts: Sequence[str] = ()) -> set:
    """The articles the plan's prose may cite: the records, the circumstances, the notes', the lawyer's landmark
    rulings' articles, and every article the checked or code-written texts cite (the norms and the procedure, which
    record_check passed; the ratione temporis, the litisconsorzio, the riti, the stragiudiziale option; the case)."""
    from . import citations
    labels = list((ctx.get("records") or {})) + list((ctx.get("circumstances") or {})) + list(ctx.get("notes") or {})
    labels += list(((ctx.get("rulings_record") or {}).get("landmarks") or {}))
    labels += [c["article"] for c in ctx.get("candidates") or [] if c.get("article")]      # v79: the candidates'
    written = [ctx.get("case") or "", ctx.get("temporal") or "", ctx.get("stragiudiziale") or ""]
    written += [f"{c.get('tesi') or ''} {c.get('articles') or ''}" for c in ctx.get("candidates") or []]
    written += list(ctx.get("litisconsorzio") or []) + list(ctx.get("riti") or []) + list(texts or [])
    return {_norm_label(x) for x in labels} | {citations.label(r) for t in written for r in citations.article_refs(t)}


_CITATION_ABBREV_RE = re.compile(r"\b(?:Sez|Cass|n|nn|art|artt|co|pen|civ|Un|sent|ord|Rv|dep|cfr)\.$", re.IGNORECASE)


def _citation_rejoin(pieces: Sequence[str]) -> List[str]:
    """The splitter's pieces with a cut inside a citation («… Cass. pen., Sez.» + «5, n. 3/2025.») joined; a sentence
    ending on a code («… c.p.p.») stays a sentence."""
    out: List[str] = []
    for piece in pieces:
        if out and _CITATION_ABBREV_RE.search(out[-1].rstrip()):
            out[-1] = f"{out[-1]} {piece}"
        else:
            out.append(piece)
    return out


def _by_sentence(text: str, review: Callable[[str], str]) -> str:
    """review(sentence) -> the sentence, changed or "" (removed), line by line; the text unchanged when nothing is."""
    lines, changed = [], False
    for line in (text or "").split("\n"):
        parts = _citation_rejoin(_SENT_SPLIT.split(line)) if line.strip() else [line]
        done = [review(x) if x.strip() else x for x in parts]
        changed = changed or done != parts
        kept = [x for x in done if x.strip()]
        if kept or not line.strip():
            lines.append(" ".join(kept) if line.strip() else line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() if changed else (text or "")


def articles_out(text: str, allowed: set, log: Optional[List[str]] = None) -> str:
    """The sentences citing an article outside the scope, removed (logged with the articles)."""
    from . import citations
    if not text or not allowed:
        return text or ""

    def review(sentence: str) -> str:
        outside = [citations.label(r) for r in citations.article_refs(sentence) if citations.label(r) not in allowed]
        if outside and log is not None:
            log.append(f"{', '.join(dict.fromkeys(outside))}: {sentence[:120]}")
        return "" if outside else sentence
    return _by_sentence(text, review)


def _removed_note(objection: Dict[str, Any], what: str = "objection") -> str:
    """The log line of an objection the scope review emptied (v79: or whose reply it emptied, what="reply")."""
    return f"{what} removed: " + str(objection.get("obiezione") or "")[:120]


def _ruling_entry(cite: str, allowed: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return next((x for x in allowed if isinstance(x, dict) and "article" in x and ruling_ok(cite, [x])), None)


def rulings_scope_out(text: str, allowed: Sequence[Dict[str, Any]], log: Optional[List[str]] = None) -> str:
    """v78: a registry ruling cited in a sentence about another article, or not on the element its passage was graded on
    (its nouns absent; a ruling with no passage is on no element) -- the citation removed: its parenthesis, else its
    sentence. The lawyer's landmark rulings: only their article's sentences."""
    from . import citations, rulings
    if not text or not any(isinstance(x, dict) and "article" in x for x in allowed or []):
        return text or ""

    def review(sentence: str) -> str:
        for m in list(_RULING_CITE_RE.finditer(sentence)):
            entry = _ruling_entry(m.group(0), allowed)
            if entry is None:
                continue
            own = _norm_label(entry["article"])
            others = [citations.label(r) for r in citations.article_refs(sentence) if citations.label(r) != own]
            landmark = str(entry.get("id") or "").startswith("consulente:")
            on_element = landmark or (bool(entry.get("element"))
                                      and rulings.element_noun_hit(sentence, {"text": entry["element"]}))
            if not others and on_element:
                continue
            if log is not None:
                log.append(f"{m.group(0)} [{own}" + (f"; cita {', '.join(others)}" if others else "; fuori elemento")
                           + f"]: {sentence[:120]}")
            paren = re.search(r"\([^()]*" + re.escape(m.group(0)) + r"[^()]*\)", sentence)
            sentence = (sentence[:paren.start()].rstrip() + sentence[paren.end():]) if paren else ""
            if not sentence:
                break
        return sentence
    return _by_sentence(text, review)


# --- the method plan: thesis selection (thinking on) ----------------------------------------------------------------------

# the code's glue around his method (my draft), v72: the theses are the code's (thesis_candidates) -- the call argues
# them; the facts, the norms with the analysis by elements, the procedural profiles, the summary, the conclusions and the
# next steps are the code's too
METHOD_SYSTEM = (
    "Applica al CASO il METODO che segue. Ricevi anche: la POSIZIONE del cliente; le SCHEDE (per ogni norma gli "
    "elementi con il loro id e il loro esito sui fatti: «integrato», «da provare», «escluso»); le TESI CANDIDATE, "
    "enumerate dal codice dal registro, ciascuna con il suo id (T1, T2…), gli elementi che contesta o su cui si regge "
    "e la sua etichetta (sostenibile / minoritaria); la VALUTAZIONE DAL REGISTRO (esito probabile, grado di certezza e "
    "obiettivo realistico calcolati sugli elementi provati: non renderla più favorevole); le PRONUNCE DISPONIBILI, le "
    "sole che puoi citare (per ogni altro punto: «orientamento da verificare»); le NOTE DI INQUADRAMENTO dell'avvocato; "
    "le OPZIONI STRAGIUDIZIALI. Non aggiungere tesi e non cambiarle: per ciascuna TESI CANDIDATA da argomentare (non "
    "quelle segnate «argomentata dal codice») scrivi gli argomenti a favore e contrari, il realismo (alto, medio, "
    "basso; per una tesi minoritaria: basso) e l'argomento in due-tre frasi. Gli argomenti di una tesi possono negare "
    "solo gli elementi che essa contesta: mai un altro elemento delle SCHEDE. La sintesi, le conclusioni, i passi successivi li scrive il codice dalla tesi scelta, e le obiezioni una "
    "chiamata successiva che la riceve. Rispondi SOLO con un JSON, in quest'ordine:\n"
    "{\"valutazioni\": {\"T1\": {\"pro\": \"…\", \"contro\": \"…\", \"realismo\": \"alto|medio|basso\", "
    "\"argomento\": \"…\"}, …}, \"questione_giuridica\": \"…\", "
    "\"orientamenti\": [{\"punto\": \"il punto controverso, in poche parole\", \"pronuncia\": \"l'id di una PRONUNCIA "
    "DISPONIBILE che lo decide, oppure vuoto\"}], \"stragiudiziale\": \"…\", "
    "\"controllo_finale\": [{\"voce\": 1, \"esito\": \"sì|no\", \"nota\": \"…\"}]}\n"
    "orientamenti: la sezione la scrive il codice, citando solo le PRONUNCE DISPONIBILI con la loro massima; tu indichi soltanto i punti "
    "controversi e, se c'è, l'id della pronuncia: non attribuire orientamenti alla giurisprudenza in nessuna parte del "
    "testo. stragiudiziale: sempre, dalle OPZIONI STRAGIUDIZIALI. controllo_finale: una voce per ciascuna domanda del "
    "CONTROLLO FINALE, nell'ordine.")


# --- v72: the theses enumerated by the code -------------------------------------------------------------------------------

CANDIDATES_MAX = 8
SUSTAINABLE, MINORITY = "sostenibile", "minoritaria"
CANDIDATE_TEXTS = {   # my drafts: each candidate's line, from the record
    "excluded": "L'elemento «{el}» ({id}) dell'art. {art} è escluso dai fatti: l'addebito su questa norma non regge",
    "open": "L'elemento «{el}» ({id}) dell'art. {art} non è provato",
    "integrated": "Contestare l'elemento «{el}» ({id}) dell'art. {art}, che i fatti integrano",
    "requalify": "Riqualificazione nell'art. {art}{rubrica}",
    "attenuante": "Attenuante dell'art. {art}{rubrica}",
    "scriminante": "Scriminante dell'art. {art}{rubrica}",
    "claim": "Fondare la domanda sull'art. {art}{rubrica}",
    "claim_open": "Fondare la domanda sull'art. {art}{rubrica}, provando {open}",       # v75b: elements to prove
    "riconvenzionale_open": "Domanda riconvenzionale sull'art. {art}{rubrica}, provando {open}",         # v76
    # v75 (my drafts): the candidate sets by role
    "claim_offence": "Fondare la domanda sul reato contestato: art. {art}{rubrica}",
    "parte_civile": "Costituirsi parte civile per le restituzioni e il risarcimento del danno (artt. 74-79 c.p.p.)",
    "sequestro": "Chiedere il sequestro e la restituzione dei beni (artt. 253, 321 e 262 c.p.p.)",
    "aggravante": "Far valere l'aggravante dell'art. {art}{rubrica}",
    "remedy": "Chiedere {label} ({articles})",
    "eccezione": "Eccepire {label} ({articles})",
    "riconvenzionale": "Domanda riconvenzionale sull'art. {art}{rubrica}",
    "defence": "Difesa sull'art. {art}{rubrica}",          # v79 (D1): a general-part defence with its elements proven
}
VICTIM_ROLES = ("persona_offesa", "parte_civile")
PENAL_KINDS = ("contenimento", "requalify", "scriminante", "attenuante", "claim_offence", "parte_civile", "sequestro",
               "aggravante", "defence")
CIVIL_ROLES = ("attore", "convenuto")
REMEDIES_FILE = os.path.join(os.path.dirname(DATA_FILE), "remedies.json")
_REMEDIES: Dict[str, Any] = {}


def remedies() -> Dict[str, Any]:
    """data/remedies.json (read once; {} when missing, logged)."""
    if "d" not in _REMEDIES:
        try:
            with open(REMEDIES_FILE, encoding="utf-8") as f:
                _REMEDIES["d"] = json.load(f)
        except (OSError, ValueError) as exc:
            logger.warning("lawyer_method: %s not read: %s", REMEDIES_FILE, exc)
            _REMEDIES["d"] = {}
    return _REMEDIES["d"]


def _in_range(label: str, code: str, span: Sequence[int]) -> bool:
    from .plan_then_write import _code_key
    parts = str(label or "").split(" ", 1)
    m = re.match(r"\d+", parts[0]) if len(parts) == 2 else None
    return bool(m) and _code_key(parts[1]) == _code_key(code) and span[0] <= int(m.group(0)) <= span[1]


def matter_of(label: str) -> Dict[str, Any]:
    """v75: the claim's matter by its article's place in the code (remedies.json «matters»); {} when none holds it."""
    return next((m for m in remedies().get("matters") or [] if _in_range(label, m["code"], m["range"])), {})


MODEL_ARGUED = ("contest_excluded", "contest_open", "contest_integrated", "requalify")    # v74: the model argues these
CODE_ARGUMENTS = {   # my drafts (v74): the argument of a thesis that contests no element, written from the record
    "contenimento": {"pro": "Non nega gli elementi che il registro dà per integrati ({ids}): sposta la difesa sulla "
                            "qualificazione e sulla misura della pena",
                     "contro": "Non evita la condanna per l'art. {art}: ne contiene l'esito",
                     "argomento": "Tutti gli elementi dell'art. {art} risultano integrati ({ids})"},
    "attenuante": {"argomento": "L'attenuante dell'art. {art} si regge sugli elementi provati {ids}: {elements}",
                   "contro": "Incide sulla pena, non sulla responsabilità"},
    "scriminante": {"argomento": "La scriminante dell'art. {art} si regge sugli elementi provati {ids}: {elements}",
                    "contro": "{who} ne contesterà i presupposti"},
    "defence": {"argomento": "La difesa sull'art. {art} si regge sugli elementi provati {ids}: {elements}",     # v79
                "contro": "{who} ne contesterà i presupposti"},
    "claim": {"argomento": "La domanda sull'art. {art} si regge sugli elementi provati {ids}: {elements}",
              "contro": "{who} ne contesterà gli elementi"},
    # v75 (my drafts): the role sets' lines
    "claim_offence": {"argomento": "La domanda si fonda sul reato contestato (art. {art}): elementi provati {ids}: "
                                   "{elements}", "contro": "{who} ne contesterà gli elementi"},
    "parte_civile": {"argomento": "La costituzione di parte civile porta nel processo penale la domanda di restituzione "
                                  "e di risarcimento del danno da reato (artt. 74-79 c.p.p.), sugli elementi provati "
                                  "{ids} dell'art. {art}", "contro": "Va proposta nei termini dell'art. 79 c.p.p."},
    "sequestro": {"argomento": "Il sequestro (artt. 253 e 321 c.p.p.) e la restituzione (art. 262 c.p.p.) tutelano i "
                               "beni oggetto del reato (art. {art})",
                  "contro": "Li dispone l'autorità giudiziaria: la persona offesa può chiederli, non ottenerli da sé"},
    "aggravante": {"argomento": "L'aggravante dell'art. {art} si regge sugli elementi provati {ids}: {elements}",
                   "contro": "Incide sulla pena, non sulla responsabilità"},
    "remedy": {"argomento": "{label_cap} ({articles}): si regge sugli elementi provati {ids} dell'art. {art}",
               "contro": "{who} ne contesterà i presupposti"},
    "eccezione": {"argomento": "{label_cap} ({articles}) richiede, nei fatti: {requires}",
                  "contro": "Il caso non ne indica ancora i presupposti"},
    "riconvenzionale": {"argomento": "La domanda riconvenzionale sull'art. {art} si regge sugli elementi provati {ids}: "
                                     "{elements}", "contro": "{who} ne contesterà gli elementi"},
    "pro": "Gli elementi {ids} dell'art. {art} risultano provati dai fatti narrati",
    "pro_open": "L'art. {art} è la norma della domanda: i fatti narrati ne indicano gli elementi, che vanno provati",
    "argomento_open": "{head} richiede la prova di: {open}",          # v75b: nothing proven yet
    "open": "Restano da provare: {open}",
    "goal": "obiettivo: {goal}",
    "riti": "riti alternativi: {riti}",
    "attenuanti": "attenuanti con elementi provati: {items}",
}


def _sentence_join(parts: Sequence[str]) -> str:
    return " ".join(p.rstrip(".") + "." for p in parts if p)


def code_argument(c: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, str]:
    """v74: {pro, contro, argomento, realismo} of a candidate that contests no element (the containment line, an
    attenuante, a scriminante, a claim) -- from the posture table (the containment's goal, the riti) and the records
    (the circumstance's, the scriminante's, the claim's elements): nothing the model wrote, so nothing it can negate.
    Realismo: the containment line «alto» (the charge fully integrato); another «alto» with every element proven, else
    «medio»."""
    records = ctx.get("records") or {}
    circ = ctx.get("circumstances") or {}
    posture = ctx.get("posture") or {}
    who = "La controparte" if posture.get("family") == "civile" else (
        "La difesa dell'imputato" if posture.get("role") in VICTIM_ROLES else "L'accusa")
    art = c.get("article") or ""
    texts = CODE_ARGUMENTS[c["kind"]]
    if c["kind"] == "eccezione":                # v75: no element of its own -- what it needs, from the table
        label = str(c.get("label") or "")
        return {"pro": _sentence_join([f"Non nega i fatti narrati: ne trae un'eccezione ({c.get('articles')})"]),
                "contro": _sentence_join([texts["contro"]]),
                "argomento": _sentence_join([texts["argomento"].format(label_cap=label[:1].upper() + label[1:],
                                                                       articles=c.get("articles") or "",
                                                                       requires=c.get("requires") or "")]),
                "realismo": BASSO}
    if c["kind"] == "contenimento":
        art = (ctx.get("assessment") or {}).get("article") or ""
        ids = ", ".join(c.get("poggia_su") or [])
        goal = str((posture.get("entry") or {}).get("goal") or "")
        riti = "; ".join(x.split(":")[0] for x in (ctx.get("riti") or [])[:2])
        att = [x for x, r in circ.items() if r.get("kind") == "attenuante"
               and any(q.get("status") == PROVATO for q in _live(r))]
        arg = [texts["argomento"].format(art=art, ids=ids)]
        arg += [CODE_ARGUMENTS["goal"].format(goal=goal)] if goal else []
        arg += [CODE_ARGUMENTS["riti"].format(riti=riti)] if riti else []
        arg += [CODE_ARGUMENTS["attenuanti"].format(items=", ".join(f"art. {x}" for x in att))] if att else []
        return {"pro": _sentence_join([texts["pro"].format(ids=ids)]),
                "contro": _sentence_join([texts["contro"].format(art=art)]),
                "argomento": _sentence_join(arg), "realismo": ALTO}
    rec = records.get(art) or circ.get(art) or {}
    live = _live(rec)
    proven = [r for r in live if r.get("status") == PROVATO and r.get("id")]
    open_ = [r for r in live if r.get("status") == DA_PROVARE and r.get("id")]
    ids = ", ".join(r["id"] for r in proven)
    elements = "; ".join(f"«{_short(r.get('text'), 80)}»" for r in proven)
    contro = CODE_ARGUMENTS["open"].format(open="; ".join(f"«{_short(r.get('text'), 80)}» ({r['id']})" for r in open_)) \
        if open_ else texts["contro"].format(who=who)
    label = str(c.get("label") or "")
    real = MEDIO if open_ or c["kind"] == "sequestro" else ALTO
    if c["kind"] in ("claim", "remedy") and not proven:      # v75b: the attore's claim with nothing proven yet
        head = f"{label[:1].upper() + label[1:]} ({c.get('articles')})" if c["kind"] == "remedy" else \
            f"La domanda sull'art. {art}"
        listed = "; ".join(f"«{_short(r.get('text'), 80)}» ({r['id']})" for r in open_)
        return {"pro": _sentence_join([CODE_ARGUMENTS["pro_open"].format(art=art)]),
                "contro": _sentence_join([texts["contro"].format(who=who)]),
                "argomento": _sentence_join([CODE_ARGUMENTS["argomento_open"].format(head=head, open=listed)]),
                "realismo": BASSO}
    return {"pro": _sentence_join([CODE_ARGUMENTS["pro"].format(ids=ids, art=art)]),
            "contro": _sentence_join([contro]),
            "argomento": _sentence_join([texts["argomento"].format(
                art=art, ids=ids, elements=elements, label_cap=label[:1].upper() + label[1:],
                articles=c.get("articles") or "")]),
            "realismo": real}


def _rubrica_tail(rec: Dict[str, Any]) -> str:
    rubrica = _plain(rec.get("rubrica"))
    rubrica = re.sub(r"^(?:aggravante|attenuante):\s*", "", rubrica)
    return f" ({rubrica})" if rubrica else ""


def thesis_candidates(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The candidate lines, from the record (v72) -- each with its ids (contesta / poggia_su) and its label, the
    code's: for the charge or the claim against the client, each element contested (excluded: sustainable; not proven:
    sustainable; integrated: minoritaria); a riqualificazione per alternative with a sound record (assertable, nothing
    contradicted; contesting the charge's distinctive element when the record marks one); a scriminante or an
    attenuante with a proven element; the containment line when the charge is fully integrato (the posture's); for a
    claimant, each of the client's articles with a proven element and none excluded. At most CANDIDATES_MAX, in that
    order of priority -- the containment line always kept."""
    from . import issue_records
    records = ctx.get("records") or {}
    circ = ctx.get("circumstances") or {}
    a = ctx.get("assessment") or {}
    posture = ctx.get("posture") or {}
    charge = a.get("article") or ""
    out: List[Dict[str, Any]] = []

    def add(kind: str, tesi: str, contesta=(), poggia_su=(), label: str = SUSTAINABLE, **extra):
        out.append({"kind": kind, "tesi": tesi, "contesta": list(contesta), "poggia_su": list(poggia_su),
                    "label": label, **extra})
    fatt = lambda x: issue_records.counted(x, str((records.get(x) or {}).get("rubrica") or ""))   # v74: fattispecie only
    named = lambda rec: bool(rec.get("named_due") or rec.get("decomposed_for") == "nominata dall'utente")      # v76
    threats = [charge] if (records.get(charge) or {}).get("side") == "threat" and fatt(charge) else []
    threats += [x for x, r in records.items() if r.get("side") == "threat" and x not in threats and fatt(x)]
    contests = {"excluded": [], "open": [], "integrated": []}
    for label in threats:
        for r in _live(records[label]):
            if not r.get("id"):
                continue
            kind = {NON_RISULTA: "excluded", DA_PROVARE: "open", PROVATO: "integrated"}.get(r.get("status"), "open")
            contests[kind].append((label, r))
    distinctive = next((r["id"] for r in _live(records.get(charge) or {}) if r.get("distinctive") and r.get("id")), "")
    known = elements_by_id(records, circ)
    for label, r in contests["excluded"] + contests["open"]:
        kind = "excluded" if r.get("status") == NON_RISULTA else "open"
        add(f"contest_{kind}", CANDIDATE_TEXTS[kind].format(el=_short(r.get("text"), 80), id=r["id"], art=label),
            contesta=[r["id"]])
    role = str(posture.get("role") or "")
    victim, civil_role = role in VICTIM_ROLES, role in CIVIL_ROLES
    offender = not victim and not civil_role        # v75: the imputato's set (as before), the role unknown too
    if victim:                  # v75: the victim's set -- never the offender's circumstances
        _victim_candidates(ctx, add, fatt)
    if a.get("all_integrated") and offender:
        main = posture_fallback(posture)
        if main:
            add("contenimento", main[0].upper() + main[1:],
                poggia_su=[r["id"] for r in _live(records.get(charge) or {}) if r.get("id")], fallback=True)
    for label, rec in records.items():
        if not offender:
            break
        if label == charge or rec.get("side") == "threat" or not (rec.get("alternative") or rec.get("alternative_facts")
                                                                  or rec.get("ranked_candidate")):     # v79 (D2)
            continue
        if not issue_records.assertable(rec) or issue_records.contradicted(rec):    # v75b: the record decides, never
            continue                                                                 # the read's «sì / no»
        if not fatt(label) or not issue_records.decomposed(rec):    # v74: a decomposed, sound fattispecie record only
            continue
        proven = [r["id"] for r in _live(rec) if r.get("status") == PROVATO and r.get("id")]
        contesta = [distinctive] if distinctive else []
        minority = any((known.get(x) or {}).get("status") == PROVATO for x in contesta)
        add("requalify", CANDIDATE_TEXTS["requalify"].format(art=label, rubrica=_rubrica_tail(rec)), contesta=contesta,
            poggia_su=proven, label=MINORITY if minority else SUSTAINABLE, article=label)
    for label, rec in records.items():
        if not offender:
            break
        if issue_records.justification(label):
            proven = [r["id"] for r in _live(rec) if r.get("status") == PROVATO and r.get("id")]
            if proven and not any(r.get("status") == NON_RISULTA for r in _live(rec)):
                add("scriminante", CANDIDATE_TEXTS["scriminante"].format(art=label, rubrica=_rubrica_tail(rec)),
                    poggia_su=proven, article=label)
    for label, rec in records.items():     # v79 (D1): a general-part defence (causation, fault, excess, attempt,
        if not offender:                    # aberratio, tenuità) with its elements proven and none excluded
            break
        if issue_records.defence_article(label) and not issue_records.justification(label):
            proven = [r["id"] for r in _live(rec) if r.get("status") == PROVATO and r.get("id")]
            if proven and not any(r.get("status") == NON_RISULTA for r in _live(rec)):
                add("defence", CANDIDATE_TEXTS["defence"].format(art=label, rubrica=_rubrica_tail(rec)),
                    poggia_su=proven, article=label)
    for label, rec in circ.items():
        if rec.get("kind") != "attenuante" or not offender:      # v75: the offender's attenuanti, the offender's only
            continue
        proven = [r["id"] for r in _live(rec) if r.get("status") == PROVATO and r.get("id")]
        if proven and not any(r.get("status") == NON_RISULTA for r in _live(rec)):
            add("attenuante", CANDIDATE_TEXTS["attenuante"].format(art=label, rubrica=_rubrica_tail(rec)),
                poggia_su=proven, article=label)
    if not victim and (role == "convenuto" or role not in ACCUSED):
        own_kind = "riconvenzionale" if role == "convenuto" else "claim"      # v75: the convenuto's own claim
        claims = []
        for label, rec in records.items():
            if rec.get("side") == "threat" or (own_kind == "riconvenzionale" and rec.get("alternative")):
                continue
            live = _live(rec)
            proven = [r["id"] for r in live if r.get("status") == PROVATO and r.get("id")]
            open_ = [r for r in live if r.get("status") == DA_PROVARE and r.get("id")]
            if any(r.get("status") == NON_RISULTA for r in live) or not fatt(label):
                continue
            if proven:
                add(own_kind, CANDIDATE_TEXTS[own_kind].format(art=label, rubrica=_rubrica_tail(rec)),
                    poggia_su=proven, article=label)
                claims.append((label, proven))
            elif (role == "attore" or named(rec)) and open_ and not rec.get("ranked_candidate"):   # v75b; v76: named
                add(own_kind, CANDIDATE_TEXTS[f"{own_kind}_open"].format(      # article always, when decomposed
                    art=label, rubrica=_rubrica_tail(rec),
                    open=", ".join(f"«{_short(r.get('text'), 60)}» ({r['id']})" for r in open_[:3])),
                    poggia_su=[r["id"] for r in open_], article=label)
                claims.append((label, []))
        if role == "attore":    # v75: the remedies of the claim's matter
            for label, proven in claims[:1]:
                for r in (matter_of(label).get("remedies") or []):
                    add("remedy", CANDIDATE_TEXTS["remedy"].format(**r), poggia_su=proven, article=label, **r)
    if role == "convenuto":     # v75: the eccezioni of the claim's matter (the claim against the client)
        claim = next((x for x in threats if matter_of(x)), "")
        for r in (matter_of(claim).get("eccezioni") or []) if claim else []:
            add("eccezione", CANDIDATE_TEXTS["eccezione"].format(**r), article=claim, **r)
    for label, r in contests["integrated"]:
        add("contest_integrated", CANDIDATE_TEXTS["integrated"].format(el=_short(r.get("text"), 80), id=r["id"],
                                                                       art=label),
            contesta=[r["id"]], label=MINORITY)
    if _family_key(posture) not in ("penale", ""):       # v76: a penal line never under a civil intake rito
        dropped = [c["kind"] for c in out if c["kind"] in PENAL_KINDS]
        out[:] = [c for c in out if c["kind"] not in PENAL_KINDS]
        if dropped:
            _log("penal_candidates_dropped", {"rito": _family_key(posture), "kinds": dropped})
    keep = [c for c in out if c.get("fallback")]
    capped = keep + [c for c in out if not c.get("fallback")][:max(0, CANDIDATES_MAX - len(keep))]
    capped = [c for c in out if c in capped]
    for i, c in enumerate(capped, 1):
        c["id"] = f"T{i}"
        c["argued"] = "model" if c["kind"] in MODEL_ARGUED else "code"      # v74: the model argues contests only
        if c["argued"] == "code":
            c["code_argument"] = code_argument(c, ctx)
    return capped


def candidate_articles(ctx: Dict[str, Any]) -> List[str]:
    """v79 (D3): the articles the thesis candidates rest on or cite -- the citation checks never remove them."""
    from . import citations
    out: List[str] = [c["article"] for c in ctx.get("candidates") or [] if c.get("article")]
    for c in ctx.get("candidates") or []:
        out += [citations.label(r) for r in citations.article_refs(f"{c.get('tesi') or ''} {c.get('articles') or ''}")]
    return list(dict.fromkeys(_norm_label(x) for x in out))


def _victim_candidates(ctx: Dict[str, Any], add: Callable[..., None], fatt: Callable[[str], bool]) -> None:
    """v75: persona offesa / parte civile -- the claim on the charged offence (the main line), the costituzione di parte
    civile (artt. 74-79 c.p.p.; not on appeal nor in execution), sequestro and restituzione (artt. 253, 321, 262 c.p.p.)
    when the offence is against property or the case asks for them, the aggravanti with a proven element as the
    victim's weapons. Never the offender's attenuanti, scriminanti or riti."""
    records = ctx.get("records") or {}
    circ = ctx.get("circumstances") or {}
    posture = ctx.get("posture") or {}
    charged = [x for x in _charged(records) if fatt(x) and _live(records[x])]
    main = charged[0] if charged else next((x for x, r in records.items() if fatt(x) and r.get("side") != "threat"
                                            and any(q.get("status") == PROVATO for q in _live(r))), "")
    if not main:
        return
    rec = records[main]
    proven = [r["id"] for r in _live(rec) if r.get("status") == PROVATO and r.get("id")]
    add("claim_offence", CANDIDATE_TEXTS["claim_offence"].format(art=main, rubrica=_rubrica_tail(rec)),
        poggia_su=proven, article=main)
    if posture.get("stage") not in ("impugnazione", "esecuzione"):
        add("parte_civile", CANDIDATE_TEXTS["parte_civile"], poggia_su=proven, article=main)
    prop = remedies().get("penal_property") or {}
    if prop and _in_range(main, prop["code"], prop["range"]) or \
            re.search(r"\bsequestr[a-z]*\b(?!\s+di\s+persona)|\brestituzion\w*|\bdissequestr", ctx.get("case") or "", re.I):
        add("sequestro", CANDIDATE_TEXTS["sequestro"], poggia_su=proven, article=main)
    for label, c in circ.items():
        live = _live(c)
        own = [r["id"] for r in live if r.get("status") == PROVATO and r.get("id")]
        if c.get("kind") == "aggravante" and own and not any(r.get("status") == NON_RISULTA for r in live):
            add("aggravante", CANDIDATE_TEXTS["aggravante"].format(art=label, rubrica=_rubrica_tail(c)), poggia_su=own,
                article=label)


def candidates_block(cands: Sequence[Dict[str, Any]]) -> str:
    lines = []
    for c in cands:
        ids = []
        if c.get("contesta"):
            ids.append("contesta " + ", ".join(c["contesta"]))
        if c.get("poggia_su"):
            ids.append("si regge su " + ", ".join(c["poggia_su"]))
        lines.append(f"- [{c['id']}] ({c['label']}) {c['tesi']}" + (f" — {'; '.join(ids)}" if ids else "")
                     + (" [argomentata dal codice: non valutarla]" if c.get("argued") == "code" else ""))
    return "\n".join(lines)


def records_block(records: Dict[str, Dict[str, Any]], circ: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
    lines = []
    for label, rec in list((records or {}).items()) + list((circ or {}).items()):
        live = _live(rec)
        if not live:
            continue
        side = {"threat": "addebito o domanda contro il cliente", "weapon": "norma del cliente"}.get(
            str(rec.get("side") or ""), "circostanza" if rec.get("circumstance") else "")
        lines.append(f"- art. {label}" + (f" — {_plain(rec.get('rubrica'))}" if rec.get("rubrica") else "")
                     + (f" [{side}]" if side else ""))
        lines += [f"  · [{r.get('id')}] {_short(r.get('text'), 140)}: {ESITO.get(r.get('status'), r.get('status'))}"
                  for r in live]
    return "\n".join(lines)


def registry(ctx: Dict[str, Any]) -> str:
    """v78 (RULINGS_RECORD): the answer's rulings registry -- the kept passages, the computed orientations, the lawyer's
    pronunce di riferimento -- as the calls receive it; "" without one."""
    if not ctx.get("rulings_record"):
        return ""
    from . import rulings
    return rulings.registry_block(ctx["rulings_record"])


def method_material(ctx: Dict[str, Any]) -> str:
    a = ctx.get("assessment") or {}
    parts = [f"CASO:\n{ctx.get('case') or ''}",
             f"POSIZIONE: {ctx.get('client') or 'la parte assistita'}, {ctx.get('posture_text') or 'non indicata'}",
             f"SCHEDE:\n{records_block(ctx.get('records') or {}, ctx.get('circumstances')) or '(nessuna)'}",
             "VALUTAZIONE DAL REGISTRO: " + f"esito probabile {a.get('esito')}; grado di certezza {a.get('grado')}; "
             f"obiettivo realistico {a.get('obiettivo')} ({a.get('why')})",
             "PRONUNCE DISPONIBILI:\n" + (registry(ctx) or "\n".join(
                 f"- [{r.get('id')}] {r.get('label')}: {_short(r.get('massima'), 300)}" if isinstance(r, dict) else f"- {r}"
                 for r in ctx.get("rulings") or []) or "(nessuna)"),
             f"OPZIONI STRAGIUDIZIALI: {ctx.get('stragiudiziale') or '(nessuna)'}"]
    if ctx.get("candidates"):
        parts.insert(3, "TESI CANDIDATE:\n" + candidates_block(ctx["candidates"]))
    if ctx.get("notes"):
        parts.append(f"{NOTES_TITLE}:\n" + "\n".join(f"- art. {x}: {n}" for x, n in ctx["notes"].items()))
    if ctx.get("riti"):
        parts.append("RITI ALTERNATIVI:\n" + "\n".join(f"- {x}" for x in ctx["riti"]))
    if ctx.get("lines"):
        parts.append("LINEE DELLA PRIMA LETTURA:\n" + "\n".join(f"- {x}" for x in ctx["lines"]))
    return "\n\n".join(parts)


def _texts(value: Any) -> List[str]:
    return [_plain(x) for x in value if _plain(x)] if isinstance(value, list) else []


def clean_objections(value: Any) -> List[Dict[str, str]]:
    """v72: the objections in any shape the model gives them -- {obiezione, risposta} (or {testo/domanda, replica}),
    or a plain string (its answer empty)."""
    out = []
    for o in value if isinstance(value, list) else []:
        if isinstance(o, str) and _plain(o):
            out.append({"obiezione": _plain(o), "risposta": ""})
        elif isinstance(o, dict):
            q = _plain(o.get("obiezione") or o.get("testo") or o.get("domanda"))
            if q:
                out.append({"obiezione": q, "risposta": _plain(o.get("risposta") or o.get("replica"))})
    return out


_CANDIDATE_KEY_RE = re.compile(r"^\W*T\s*(\d{1,2})\b", re.I)


def candidate_key(key: Any) -> str:
    """v79: a valuation's candidate id as the reply wrote it -- «T1», «t1», «[T1]», «T1 - Contestare …» -> «T1»; any
    other key upper-cased as it is (so a thesis the model adds is still seen, and dropped)."""
    m = _CANDIDATE_KEY_RE.match(str(key or ""))
    return f"T{int(m.group(1))}" if m else str(key or "").strip().upper()


def valuations(raw: Any) -> Dict[str, Dict[str, Any]]:
    """v79 (A1): the reply's «valutazioni» keyed by candidate id, in either shape the model gives them -- an object keyed
    by id, or a list of objects carrying their id («id», «tesi» or «candidata»); a value that is not an object dropped.
    Before v79 the list shape and a decorated key («[T1]») were read as no valuation at all."""
    vals = (raw or {}).get("valutazioni") if isinstance(raw, dict) else None
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(vals, dict):
        for k, v in vals.items():
            if isinstance(v, dict):
                out.setdefault(candidate_key(k), v)
    elif isinstance(vals, list):
        for v in vals:
            if isinstance(v, dict):
                key = next((v[f] for f in ("id", "tesi", "candidata") if v.get(f)), "")
                if key:
                    out.setdefault(candidate_key(key), v)
    return out


def apply_valuations(raw: Any, candidates: Sequence[Dict[str, Any]], log: Optional[Dict[str, Any]] = None
                     ) -> List[Dict[str, Any]]:
    """v72: the soluzioni -- the code's candidates, each with the model's pro, contro, realismo and argument (a
    minority one: realismo basso); a thesis the model adds (a key no candidate has) dropped, into log["dropped"]."""
    raw = raw if isinstance(raw, dict) else {}
    vals = valuations(raw)
    ids = {c["id"] for c in candidates}
    dropped = [k for k in vals if k not in ids]
    if raw.get("soluzioni"):
        dropped.append("soluzioni")
    out = []
    ignored = [c["id"] for c in candidates if c.get("code_argument") and c["id"] in vals]
    for c in candidates:
        v = c.get("code_argument") or vals.get(c["id"]) or {}       # v74: a code-argued candidate never the model's
        real = _plain(v.get("realismo")).lower()
        real = real if real in REALISMO else (BASSO if c["label"] == MINORITY else MEDIO)
        if c["label"] == MINORITY:
            real = BASSO
        out.append({"tesi": c["tesi"], "pro": _plain(v.get("pro")), "contro": _plain(v.get("contro")), "realismo": real,
                    "argomento": _plain(v.get("argomento")), "minoritaria": c["label"] == MINORITY,
                    "contesta": list(c["contesta"]), "poggia_su": list(c["poggia_su"]), "candidate": c["id"],
                    "kind": c["kind"], "argued": c.get("argued") or "model",
                    **({"fallback": True} if c.get("fallback") else {})})
    if log is not None:
        log["dropped"] = dropped
        log["code_argued_ignored"] = ignored
    return out


def score_sample(raw: Any, candidates: Sequence[Dict[str, Any]], known: Dict[str, Dict[str, Any]]
                 ) -> Tuple[float, List[str]]:
    """v72: one reply of the thesis call, scored by the code -- every candidate valued and none added (valid ids), no
    candidate resting on an excluded element rated above «basso», no minority one above «basso»; an unreadable reply:
    -inf. (v73: the objections are no longer the thesis call's -- objections_plan writes them after the selection.)"""
    if not isinstance(raw, dict) or not raw:
        return float("-inf"), ["reply not readable"]
    vals = valuations(raw)
    ids = {c["id"] for c in candidates if c.get("argued", "model") == "model"}    # v74: the code argues the rest
    keys = set(vals) - {c["id"] for c in candidates if c.get("argued") == "code"}
    score, why = 0.0, []
    covered = len(keys & ids)
    score += 10.0 * covered / max(1, len(ids))
    if covered < len(ids):
        why.append(f"{len(ids) - covered} candidates not valued")
    added = keys - ids
    if added:
        score -= 2.0 * len(added)
        why.append(f"added {sorted(added)}")
    by = {c["id"]: c for c in candidates}
    for k, v in vals.items():
        c = by.get(k)
        if not c or not isinstance(v, dict):
            continue
        real = _plain(v.get("realismo")).lower()
        if real not in REALISMO:
            score -= 1.0
            why.append(f"{c['id']}: no realismo")
            continue
        if real != BASSO and c["label"] == MINORITY:
            score -= 2.0
            why.append(f"{c['id']}: minority rated {real}")
        if real != BASSO and any((known.get(x) or {}).get("status") == NON_RISULTA for x in c.get("poggia_su") or []):
            score -= 3.0
            why.append(f"{c['id']}: rests on an excluded element, rated {real}")
    return score, why


def clean_method(raw: Any, candidates: Optional[Sequence[Dict[str, Any]]] = None,
                 log: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The method plan as the layout reads it: every key, typed, empty when the model gave nothing. v72: with the code's
    candidates, the soluzioni are theirs (apply_valuations) -- never the model's own."""
    raw = raw if isinstance(raw, dict) else {}
    sols = []
    for s in raw.get("soluzioni") or []:
        if not isinstance(s, dict) or not _plain(s.get("tesi")):
            continue
        real = _plain(s.get("realismo")).lower()
        ids = lambda v: [str(x).strip().upper() for x in (v if isinstance(v, list) else []) if str(x).strip()]
        sols.append({"tesi": _plain(s.get("tesi")), "pro": _plain(s.get("pro")), "contro": _plain(s.get("contro")),
                     "realismo": real if real in REALISMO else MEDIO, "minoritaria": bool(s.get("minoritaria")),
                     "contesta": ids(s.get("contesta")), "poggia_su": ids(s.get("poggia_su"))})
    try:
        main = int(raw.get("tesi_principale") or 1)
    except (TypeError, ValueError):
        main = 1
    objections = clean_objections(raw.get("obiezioni") or raw.get("obiezioni_e_risposte") or [])
    orient = [{"punto": _plain(o.get("punto") or o.get("tema")), "pronuncia": _plain(o.get("pronuncia"))}
              for o in raw.get("orientamenti") or [] if isinstance(o, dict) and _plain(o.get("punto") or o.get("tema"))]
    steps = [{"passo": _plain(x.get("passo")), "scopo": _plain(x.get("scopo"))}
             for x in raw.get("passi") or [] if isinstance(x, dict) and _plain(x.get("passo"))]
    checks = [{"voce": x.get("voce"), "esito": _plain(x.get("esito")).lower(), "nota": _plain(x.get("nota"))}
              for x in raw.get("controllo_finale") or [] if isinstance(x, dict)]
    if candidates is not None:
        sols, main = apply_valuations(raw, candidates, log), 0
    return {"questione_giuridica": _plain(raw.get("questione_giuridica")), "orientamenti": orient, "soluzioni": sols,
            "tesi_principale": (main if 1 <= main <= len(sols) else 1) if candidates is None else 0,
            "subordinata": _plain(raw.get("subordinata")),
            "stragiudiziale": _plain(raw.get("stragiudiziale")), "obiezioni": objections,
            "obiettivo": _plain(raw.get("obiettivo")), "conclusioni": str(raw.get("conclusioni") or "").strip(),
            "sintesi": str(raw.get("sintesi") or "").strip(), "dipende_da": _plain(raw.get("dipende_da")),
            "rischio_residuo": _plain(raw.get("rischio_residuo")), "passi": steps, "controllo_finale": checks}


THESIS_LOG_ENV = "THESIS_LOG_DIR"
THESIS_LOG_DIR = None          # v77: the runtime state folder's logs/ (src/state_paths.py; was <checkout>/logs)


def case_id(ctx: Dict[str, Any]) -> str:
    """The case's id for the thesis log: the run's own (single_shot's case_id), else a hash of the case text."""
    import hashlib
    given = re.sub(r"[^\w.-]+", "_", str(ctx.get("case_id") or "")).strip("_")
    return given or "h" + hashlib.sha1(str(ctx.get("case") or "").encode("utf-8")).hexdigest()[:10]


def thesis_log(ctx: Dict[str, Any], messages: Sequence[Any], raw_reply: str, parsed: Any, plan: Dict[str, Any],
               seconds: float, error: str = "", samples: Sequence[Dict[str, Any]] = (), best: Optional[int] = None
               ) -> Optional[str]:
    """v71: logs/thesis_<case>.json -- the thesis call's exact input (system and human messages, max_tokens, thinking)
    and output (the raw reply, the parsed JSON, the cleaned plan), with the record's elements and the assessment it was
    checked against: an A/B with another model replays the same input (scripts/thesis_replay.py). Never fails the
    answer."""
    try:
        from . import ai_chat
        from .. import state_paths
        folder = os.getenv(THESIS_LOG_ENV) or THESIS_LOG_DIR or state_paths.logs_dir()
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"thesis_{case_id(ctx)}.json")
        entry = {"case_id": case_id(ctx), "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "model": ai_chat._chat_kwargs.get("model"), "llm_thinking": os.getenv("LLM_THINKING") or "",
                 "thinking": thinking_supported() and ai_chat.per_call(),
                 "reasoning_effort": ai_chat.reasoning_effort() if thinking_supported() and ai_chat.per_call() else "",
                 "max_tokens": METHOD_TOKENS,
                 "messages": [{"role": "system" if i == 0 else "user", "content": getattr(m, "content", str(m))}
                              for i, m in enumerate(messages)],
                 "raw_reply": raw_reply, "parsed": parsed if isinstance(parsed, (dict, list)) else None,
                 "plan": plan, "seconds": seconds, "error": error,
                 "candidates": ctx.get("candidates") or [], "samples": list(samples), "best": best,
                 "argument_binding": ctx.get("argument_binding") or {},
                 "n": sampling()[0], "temperature": sampling()[1], "deterministic": deterministic(),
                 "choice": ctx.get("thesis_choice") or {}, "counts": ctx.get("selection_counts") or {},
                 "elements": elements_by_id(ctx.get("records") or {}, ctx.get("circumstances")),
                 "assessment": ctx.get("assessment") or {}, "posture": ctx.get("posture") or {}}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=1, default=str)
        return path
    except Exception as exc:          # noqa: BLE001 -- the log never fails the answer
        logger.warning("lawyer_method: thesis log not written: %s", exc)
        return None


SAMPLES = 3                    # v72: best-of-3 on the thesis call
SAMPLE_TEMPERATURE = 0.7
DETERMINISTIC_ENV = "THESIS_DETERMINISTIC"     # v79 (A3): one sample at temperature 0 (nightly / replay --deterministic)


def deterministic() -> bool:
    return (os.getenv(DETERMINISTIC_ENV) or "").strip().lower() in ("1", "true", "yes", "on")


def sampling() -> Tuple[int, float]:
    """(replies, temperature) of the thesis call: best-of-SAMPLES at SAMPLE_TEMPERATURE; v79 (A3): one at 0 under
    THESIS_DETERMINISTIC, for comparing two versions on the same input."""
    return (1, 0.0) if deterministic() else (SAMPLES, SAMPLE_TEMPERATURE)


def reply_digest(parsed: Any) -> str:
    """v79 (A3): the fixed tie-break -- the sha1 of the parsed reply, keys sorted: the same replies, the same choice,
    whatever order the server returns them in."""
    import hashlib
    return hashlib.sha1(json.dumps(parsed if isinstance(parsed, (dict, list)) else {}, ensure_ascii=False,
                                   sort_keys=True, default=str).encode("utf-8")).hexdigest()


def choose(scored: Sequence[Dict[str, Any]]) -> Tuple[Optional[int], Dict[str, Any]]:
    """v79 (A3): the reply kept -- the highest score (an unreadable one last), ties broken by the smallest digest of the
    parsed reply, never by the order the replies arrived in. (index, {by, tie_break, scores, digests, tied})."""
    if not scored:
        return None, {"by": "score", "tie_break": "sha1 of the parsed reply, smallest", "scores": [], "tied": []}
    value = lambda i: scored[i]["score"] if scored[i]["score"] is not None else -1e9
    digests = [reply_digest(x.get("parsed")) for x in scored]
    top = max(value(i) for i in range(len(scored)))
    tied = [i for i in range(len(scored)) if value(i) == top]
    best = min(tied, key=lambda i: (digests[i], i))
    return best, {"by": "score", "tie_break": "sha1 of the parsed reply, smallest", "scores": [x["score"] for x in scored],
                  "digests": [d[:12] for d in digests], "tied": tied if len(tied) > 1 else []}


def _samples(messages: Sequence[Any], call: Optional[Callable]) -> List[Tuple[str, Any]]:
    """[(raw reply, parsed)] of the thesis call's replies (one request, n from sampling()); call (tests, the replay): a
    stand-in taking the messages and returning one parsed reply or a list of them."""
    from . import ai_chat
    from .llm_reply import parse_json_reply
    if call is not None:
        got = call(messages)
        got = got if isinstance(got, list) else [got]
        return [(json.dumps(g, ensure_ascii=False) if not isinstance(g, str) else g, g) for g in got]
    n, temperature = sampling()
    texts = ai_chat.call_chat_n(list(messages), n, max_tokens=METHOD_TOKENS, temperature=temperature,
                                thinking=True if thinking_supported() else None)
    out = []
    for text in texts:
        try:
            out.append((text, parse_json_reply(text)))
        except ValueError:
            out.append((text, {}))
    return out


BIND_FIELDS = ("pro", "contro", "argomento")


def negation_flags(items: Sequence[Dict[str, Any]], known: Dict[str, Dict[str, Any]], call: Optional[Callable] = None
                   ) -> Tuple[str, Dict[int, Tuple[str, str]]]:
    """v74/v75b: (how, {item index: (element, why)}) -- the items [{text, may_negate}] that negate an element they may
    not: the review call (plan_then_write.negation_review), its word test when the call fails."""
    from . import plan_then_write as ptw, record_check
    flags = ptw.negation_review(list(items), known, call)
    bad: Dict[int, Tuple[str, str]] = {}
    if flags is not None:
        for f in flags:
            bad.setdefault(f["i"], (f["elemento"], f["perche"]))
        return "review", bad
    for i, it in enumerate(items):
        hit = record_check.negated_elements(it["text"], known, it["may_negate"])
        if hit:
            bad[i] = (hit[0], "il testo nega l'elemento (test delle parole)")
    return "word_test", bad


def bind_replies(objections: List[Dict[str, str]], main: Dict[str, Any], known: Dict[str, Dict[str, Any]],
                 call: Optional[Callable] = None) -> Dict[str, Any]:
    """v75b: in place, every sentence of an objection's reply that negates an element the main line does not contest
    dropped (negation_flags) -- the replies argue the selected line, never past its «contesta». Logged."""
    items, where = [], []
    split = []
    for oi, o in enumerate(objections):
        parts = [x.strip() for x in _SENT_SPLIT.split(o.get("risposta") or "") if x.strip()]
        split.append(parts)
        for k, sentence in enumerate(parts):
            items.append({"text": sentence, "may_negate": list(main.get("contesta") or [])})
            where.append((oi, k))
    if not items:
        return {"how": "", "checked": 0, "dropped": []}
    how, bad = negation_flags(items, known, call)
    dropped = []
    for i, (eid, why) in sorted(bad.items()):
        oi, k = where[i]
        dropped.append({"obiezione": objections[oi]["obiezione"][:120], "frase": split[oi][k][:300], "elemento": eid,
                        "perche": why})
        split[oi][k] = ""
    for oi, parts in enumerate(split):
        objections[oi]["risposta"] = " ".join(x for x in parts if x)
    out = {"how": how, "checked": len(items), "dropped": dropped}
    _log("reply_binding", out)
    return out


def bind_arguments(sols: List[Dict[str, Any]], known: Dict[str, Dict[str, Any]], call: Optional[Callable] = None
                   ) -> Dict[str, Any]:
    """v74: arguments bound to declarations -- in place, every sentence of a model-argued thesis's pro, contro and
    argomento that negates an element the thesis does not contest (its «contesta») dropped: the requirement check
    against the record, element by element (plan_then_write.negation_review, one call for every thesis), its word test
    when the call fails (record_check.negated_elements). The summary and the conclusions restate the cleaned argument.
    The log: {"how": review | word_test, "checked", "dropped": [{tesi, campo, frase, elemento, perche}]}."""
    items: List[Dict[str, Any]] = []
    where: List[Tuple[int, str, int]] = []
    split: Dict[Tuple[int, str], List[str]] = {}
    for si, t in enumerate(sols):
        if t.get("argued", "model") != "model":
            continue
        for field in BIND_FIELDS:
            parts = [x.strip() for x in _SENT_SPLIT.split(t.get(field) or "") if x.strip()]
            split[(si, field)] = parts
            for k, sentence in enumerate(parts):
                items.append({"text": sentence, "may_negate": list(t.get("contesta") or [])})
                where.append((si, field, k))
    if not items:
        return {"how": "", "checked": 0, "dropped": []}
    how, bad = negation_flags(items, known, call)
    dropped = []
    for i, (eid, why) in sorted(bad.items()):
        si, field, k = where[i]
        dropped.append({"tesi": sols[si].get("candidate") or sols[si]["tesi"][:80], "campo": field,
                        "frase": split[(si, field)][k][:300], "elemento": eid, "perche": why})
        split[(si, field)][k] = ""
    for (si, field), parts in split.items():
        sols[si][field] = " ".join(x for x in parts if x)
    out = {"how": how, "checked": len(items), "dropped": dropped}
    _log("argument_binding", out)
    return out


def method_plan(ctx: Dict[str, Any], call: Optional[Callable] = None) -> Dict[str, Any]:
    """The thesis call (v72): the code's candidates (thesis_candidates, into ctx["candidates"]) argued by the model --
    SAMPLES replies in one request with thinking on (LLM_THINKING=per_call), each scored by the code (score_sample), the
    best kept; the soluzioni are the candidates with its valuations, a thesis it adds dropped and logged. Timed into
    ctx["timings"], its input and every reply logged (thesis_log)."""
    from langchain_core.messages import HumanMessage, SystemMessage
    ctx["candidates"] = thesis_candidates(ctx)
    system = METHOD_SYSTEM + "\n\n" + method_text(ctx.get("rito") or "", ctx.get("areas") or ())
    if ctx.get("rulings_record"):       # v78: a passage on a contested element can lower a thesis' realism
        from . import rulings
        system += rulings.REALISM_RULE
    messages = [SystemMessage(content=system), HumanMessage(content=method_material(ctx))]
    known = elements_by_id(ctx.get("records") or {}, ctx.get("circumstances"))
    t0 = time.time()
    error = ""
    try:
        samples = _samples(messages, call)
    except Exception as exc:          # noqa: BLE001 -- no reply: the candidates stand, unargued
        logger.warning("lawyer_method: the thesis call failed: %s", exc)
        error = f"{type(exc).__name__}: {exc}"[:300]
        _log("plan_failed", {"error": error})
        samples = []
    scored = []
    for text, parsed in samples:
        value, why = score_sample(parsed, ctx["candidates"], known)
        scored.append({"raw_reply": text, "parsed": parsed if isinstance(parsed, (dict, list)) else None,
                       "score": value if value != float("-inf") else None, "why": why})
    best, ctx["thesis_choice"] = choose(scored)          # v79 (A3): score, then a fixed tie-break
    raw = scored[best]["parsed"] if best is not None else {}
    seconds = round(time.time() - t0, 1)
    ctx.setdefault("timings", {})["thesis_selection_s"] = seconds
    clean_log: Dict[str, Any] = {}
    plan = clean_method(raw, ctx["candidates"], clean_log)
    ctx["argument_binding"] = bind_arguments(plan["soluzioni"], known)       # v74: no negation beyond «contesta»
    if ctx.get("rulings_record"):       # v78: a prevailing line for an element's integration: its contest «basso»
        from . import rulings           # (v79: here, so the counts below are the selection compose makes)
        capped: List[Dict[str, Any]] = []
        rulings.cap_realism(plan, ctx["rulings_record"], capped)
        if capped:
            _log("rulings_realism", capped)
    ctx["selection_counts"] = selection_counts(ctx, plan, raw)
    ctx["thesis_log"] = thesis_log(ctx, messages, scored[best]["raw_reply"] if best is not None else "", raw, plan,
                                   seconds, error, scored, best)
    _log("plan", {"candidates": len(ctx["candidates"]), "samples": [x["score"] for x in scored], "best": best,
                  "choice": ctx["thesis_choice"], "counts": ctx["selection_counts"],
                  "dropped": clean_log.get("dropped"),
                  "orientamenti": len(plan["orientamenti"]), "seconds": seconds, "log": ctx["thesis_log"]},
         seconds * 1000)
    return plan


COUNT_KEYS = ("offered", "parsed", "valued", "barred", "unverifiable", "minoritaria", "chosen", "main_is_fallback")


def selection_counts(ctx: Dict[str, Any], plan: Dict[str, Any], raw: Any = None) -> Dict[str, Any]:
    """v79 (A2): where the candidates went -- offered (the code's), parsed (reached the selection as soluzioni), valued
    (model-argued ones the kept reply valued), barred (contest an «integrato» element, unlabelled), unverifiable,
    minoritaria, the main line chosen (its candidate id, «fallback» when the selection inserted one) and whether it is
    the fallback (the posture's or the remedies table's line, no candidate of the code's)."""
    cands = ctx.get("candidates") or []
    vals = valuations(raw)
    sols, main, _ = selection(ctx, plan)
    m = sols[main] if sols else {}
    inserted = bool(m.get("fallback")) and not m.get("candidate")
    return {"offered": len(cands), "parsed": sum(1 for t in plan.get("soluzioni") or [] if t.get("candidate")),
            "valued": sum(1 for c in cands if c.get("argued", "model") == "model" and c["id"] in vals),
            "model_argued": sum(1 for c in cands if c.get("argued", "model") == "model"),
            "barred": sum(1 for t in sols if t.get("barred")),
            "unverifiable": sum(1 for t in sols if t.get("unverifiable")),
            "minoritaria": sum(1 for t in sols if t.get("minoritaria")),
            "chosen": "fallback" if inserted else str(m.get("candidate") or ""),
            "chosen_kind": str(m.get("kind") or ("fallback" if inserted else "")),
            "main_is_fallback": inserted,
            # v79 (D4): the general part's defence lines, the ranked articles graded into candidates
            "defence_lines": sum(1 for c in cands if c.get("kind") in ("defence", "scriminante")),
            "ranked_candidates": sum(1 for c in cands
                                     if ((ctx.get("records") or {}).get(c.get("article") or "") or {}).get("ranked_candidate"))}


# --- v73: the objections, written for the selected line --------------------------------------------------------------------

OBJECTIONS_TOKENS = 1400
OBJECTIONS_SYSTEM = (   # my draft: the glue of the objections call; his rules follow it (writer_rules_text)
    "Ricevi il CASO, la POSIZIONE del cliente, le SCHEDE (per ogni norma gli elementi con il loro id e il loro esito "
    "sui fatti), la VALUTAZIONE DAL REGISTRO e la strategia già scelta dal codice: la LINEA PRINCIPALE, la LINEA "
    "SUBORDINATA e le TESI ESCLUSE. Scrivi le tre obiezioni più forti che {who} opporrà alla LINEA PRINCIPALE, "
    "ciascuna con la risposta. Ogni risposta si regge sulla linea principale, o sulla subordinata, e sugli elementi "
    "delle SCHEDE: non usare mai una TESI ESCLUSA, né come risposta né come argomento, e non negare un elemento "
    "«integrato». Non attribuire orientamenti alla giurisprudenza. Rispondi SOLO con un JSON:\n"
    "{{\"obiezioni\": [{{\"obiezione\": \"…\", \"risposta\": \"…\"}}]}}")


def selection(ctx: Dict[str, Any], plan: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int, Optional[int]]:
    """(soluzioni, the main line, the subordinate) as compose selects them -- the same function on the same inputs, on
    a copy (the plan untouched)."""
    import copy
    sols, main = coherent_solutions(copy.deepcopy(plan), ctx.get("records") or {}, ctx.get("assessment") or {},
                                    ctx.get("riti") or [], [], ctx.get("posture") or {}, ctx.get("circumstances"))
    return sols, main, subordinate_of(sols, main)


def _line(t: Dict[str, Any]) -> str:
    out = t["tesi"].rstrip(".")
    if t.get("minoritaria"):
        out += " (tesi minoritaria)"
    if t.get("contesta"):
        out += f"; contesta {', '.join(t['contesta'])}"
    if t.get("poggia_su"):
        out += f"; si regge su {', '.join(t['poggia_su'])}"
    return out + (f". Argomento: {t['argomento']}" if t.get("argomento") else ".")


def excluded_theses(sols: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [t for t in sols if t.get("barred") or t.get("unverifiable")]


def objections_material(ctx: Dict[str, Any], sols: Sequence[Dict[str, Any]], main: int, sub: Optional[int]) -> str:
    a = ctx.get("assessment") or {}
    out = [f"CASO:\n{ctx.get('case') or ''}",
           f"POSIZIONE: {ctx.get('client') or 'la parte assistita'}, {ctx.get('posture_text') or 'non indicata'}",
           f"SCHEDE:\n{records_block(ctx.get('records') or {}, ctx.get('circumstances')) or '(nessuna)'}",
           "VALUTAZIONE DAL REGISTRO: " + f"esito probabile {a.get('esito')}; grado di certezza {a.get('grado')}; "
           f"obiettivo realistico {a.get('obiettivo')} ({a.get('why')})",
           f"LINEA PRINCIPALE: {_line(sols[main])}",
           f"LINEA SUBORDINATA: {_line(sols[sub]) if sub is not None else '(nessuna)'}"]
    barred_ = excluded_theses(sols)
    out.append("TESI ESCLUSE:\n" + ("\n".join(
        f"- {t['tesi'].rstrip('.')}: " + (f"contesta {', '.join(t['barred'])}, integrato" if t.get("barred")
                                          else "non dichiara gli elementi") for t in barred_) or "(nessuna)"))
    if registry(ctx):                   # v78: the kept passages, material of the objections too
        out.append(registry(ctx))
    return "\n\n".join(out)


NOT_APPLIED = ("Nota: il metodo di analisi per elementi non è stato applicato a questo parere: {why}. Il parere segue "
               "la struttura ordinaria.")      # my draft (v76): the method never steps aside silently
NOT_APPLIED_WHY = {   # my drafts (v76)
    "placeholder": "il registro non offre una linea fondata sugli elementi dei fatti narrati",
    "undecomposed": "l'art. {art} (l'addebito o la domanda) non si scompone in elementi verificabili",
    "unread": "il testo dell'art. {art} (l'addebito o la domanda) non è stato recuperato",
}


def not_applied_note(why: str) -> str:
    return NOT_APPLIED.format(why=str(why or NOT_APPLIED_WHY["placeholder"]).rstrip("."))


def placeholder_main(ctx: Dict[str, Any], plan: Dict[str, Any]) -> str:
    """v75: the selection's main line when it is the placeholder («Tesi da costruire …», fallback_thesis with no
    posture line and no containment) -- a failure state: the method refuses the answer (the standard body is written,
    logged); "" otherwise."""
    sols, main, _ = selection(ctx, plan)
    return sols[main]["tesi"] if sols[main].get("placeholder") else ""


def objections_plan(ctx: Dict[str, Any], plan: Dict[str, Any], call: Optional[Callable] = None,
                    review_call: Optional[Callable] = None) -> Dict[str, Any]:
    """v73: the objections written for the selected line -- one call after the selection (selection(): compose's own),
    given the main line, the subordinate and the barred list; an answer that argues a barred thesis (same_line) is
    dropped, logged. The plan with them as its «obiezioni» (none when the call fails: compose's fallbacks then, the
    record's and the selection's). Timed into ctx["timings"]["objections_s"], logged into the thesis log."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from .llm_reply import parse_json_reply
    sols, main, sub = selection(ctx, plan)
    penal = (ctx.get("posture") or {}).get("family") != "civile"
    system = OBJECTIONS_SYSTEM.format(who="l'accusa" if penal else "la controparte")
    rules = writer_rules_text(ctx.get("rito") or "", ctx.get("areas") or ())
    messages = [SystemMessage(content=system + ("\n\n" + rules if rules else "")),
                HumanMessage(content=objections_material(ctx, sols, main, sub))]
    t0 = time.time()
    raw: Any = {}
    text, error = "", ""
    try:
        if call is not None:
            raw = call(messages)
            text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
        else:
            from . import defensive_generation as dg
            text = dg._call_chat(messages, max_tokens=OBJECTIONS_TOKENS,
                                 **({"thinking": True} if thinking_supported() else {}))
        raw = parse_json_reply(text) if isinstance(raw, str) or call is None else raw
    except Exception as exc:          # noqa: BLE001 -- no objections from the call: compose's fallbacks answer
        logger.warning("lawyer_method: the objections call failed: %s", exc)
        error = f"{type(exc).__name__}: {exc}"[:300]
        raw = {}
    got = clean_objections((raw or {}).get("obiezioni") if isinstance(raw, dict) else raw)
    excluded = excluded_theses(sols)
    kept, dropped = [], []
    for o in got:
        hit = next((t["tesi"] for t in excluded if same_line(o.get("risposta") or "", t["tesi"])), "")
        (dropped if hit else kept).append({**o, **({"argues": hit[:160]} if hit else {})})
    replies = bind_replies(kept, sols[main], elements_by_id(ctx.get("records") or {}, ctx.get("circumstances")),
                           review_call)                    # v75b: no reply negates past the main line's «contesta»
    seconds = round(time.time() - t0, 1)
    ctx.setdefault("timings", {})["objections_s"] = seconds
    entry = {"main": sols[main]["tesi"], "subordinate": sols[sub]["tesi"] if sub is not None else "",
             "barred": [t["tesi"] for t in excluded], "kept": len(kept), "dropped": dropped, "seconds": seconds,
             "error": error, "reply_binding": replies}
    objections_log(ctx, messages, text, raw, entry)
    _log("objections", entry, seconds * 1000)
    return {**plan, "obiezioni": [{"obiezione": o["obiezione"], "risposta": o.get("risposta") or ""}
                                  for o in kept][:OBJECTIONS]}


def objections_log(ctx: Dict[str, Any], messages: Sequence[Any], raw_reply: str, parsed: Any,
                   entry: Dict[str, Any]) -> None:
    """The objections call into the thesis log (its «objections» key): input, reply, what was kept and dropped."""
    path = ctx.get("thesis_log")
    if not path:
        return
    try:
        with open(path, encoding="utf-8") as f:
            log = json.load(f)
        log["objections"] = {**entry, "max_tokens": OBJECTIONS_TOKENS, "raw_reply": raw_reply,
                             "parsed": parsed if isinstance(parsed, (dict, list)) else None,
                             "messages": [{"role": "system" if i == 0 else "user",
                                           "content": getattr(m, "content", str(m))} for i, m in enumerate(messages)]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=1, default=str)
    except Exception as exc:          # noqa: BLE001 -- the log never fails the answer
        logger.warning("lawyer_method: objections not logged: %s", exc)


def thinking_supported() -> bool:
    """dg._call_chat takes thinking= (ai_chat with v70): passed only then, never to an older one."""
    from . import defensive_generation as dg
    import inspect
    try:
        return "thinking" in inspect.signature(dg._call_chat).parameters
    except (TypeError, ValueError):
        return False


# --- the coherence check --------------------------------------------------------------------------------------------------

_FAVOUR_RE = re.compile(r"\besito\s+(?:sarà\s+|appare\s+|è\s+)?(?:favorevole|positivo)|\bbuone\s+(?:probabilità|"
                        r"possibilità)|\bprobabil\w*\s+(?:assoluzion\w*|proscioglim\w*|accogliment\w*|vittoria)|"
                        r"\b(?:sarà|verrà)\s+(?:assolt\w*|prosciolt\w*)|\bcerto\s+(?:accogliment\w*|esito)|"
                        r"\bfondate\s+possibilità\s+di\s+(?:assoluzion|accogliment)", re.I)
_SENT_SPLIT = re.compile(r"(?<!\bn\.)(?<!\bnn\.)(?<!\bco\.)(?<!\bart\.)(?<!\bartt\.)(?<!\blett\.)(?<!\bss\.)"
                         r"(?<!\bcfr\.)(?<!\bsez\.)(?<!\bCass\.)(?<=[.;!?])\s+(?=[A-ZÀ-ÖØ-Ý«\"“(\[*\d-])")
_ABBREV_END_RE = re.compile(r"\b(?:[a-z]\.){2,3}$", re.I)
_ELEMENT_ID_RE = re.compile(r"^[RC]\d{1,3}$")

# v71: a thesis is checked by the elements it declares (contesta / poggia_su), never by its words
BARRED_LABEL = "tesi che nega un elemento integrato (non sostenibile)"
UNVERIFIABLE_LABEL = "tesi non verificabile: non dichiara gli elementi"
INTEGRATED_NOTE = "contesta un elemento integrato"


def elements_by_id(records: Dict[str, Dict[str, Any]], circ: Optional[Dict[str, Dict[str, Any]]] = None
                   ) -> Dict[str, Dict[str, Any]]:
    """{element id: {article, text, status}} of the record's live elements and the circumstances'."""
    out = {}
    for label, rec in list((records or {}).items()) + list((circ or {}).items()):
        for r in _live(rec):
            if r.get("id"):
                out[str(r["id"])] = {"article": label, "text": _short(r.get("text"), 140),
                                     "status": r.get("status") or DA_PROVARE, "side": rec.get("side") or ""}
    return out


def declared_ids(value: Any, known: Dict[str, Dict[str, Any]], dropped: Optional[List[str]] = None) -> List[str]:
    """The element ids a thesis declares, kept only when the record holds them (the others into dropped)."""
    out = []
    for x in value if isinstance(value, list) else []:
        x = str(x or "").strip().upper()
        if _ELEMENT_ID_RE.match(x) and x in known:
            out.append(x)
        elif x and dropped is not None:
            dropped.append(x)
    return list(dict.fromkeys(out))


def barred(thesis: Dict[str, Any], known: Dict[str, Dict[str, Any]]) -> List[str]:
    """The «integrato» elements the thesis contests -- any one bars it from the main line."""
    return [x for x in thesis.get("contesta") or [] if (known.get(x) or {}).get("status") == PROVATO]


def posture_fallback(posture: Dict[str, Any]) -> str:
    """The posture table's main line when every thesis is barred («fallback_main» of the cell's entry)."""
    return str(((posture or {}).get("entry") or {}).get("fallback_main") or "")


def fallback_thesis(a: Dict[str, Any], riti: Sequence[str], posture: Optional[Dict[str, Any]] = None
                    ) -> Dict[str, Any]:
    main = posture_fallback(posture or {})
    if main:
        return {"tesi": main[0].upper() + main[1:], "pro": "non contesta alcun elemento integrato" + (
                    f"; {'; '.join(x.split(':')[0] for x in riti[:3])}" if riti else ""),
                "contro": "presuppone l'accertamento della condotta", "realismo": ALTO, "minoritaria": False,
                "contesta": [], "poggia_su": [], "fallback": True}
    civil = remedies().get("fallback", {}).get(str((posture or {}).get("role") or "")) \
        if _family_key(posture or {}) not in ("penale", "") else ""
    if civil:                   # v76: a civil role's line from the remedies table -- containment is penal-only
        return {"tesi": civil, "pro": "non nega i fatti narrati", "contro": "non porta al rigetto integrale della domanda",
                "realismo": MEDIO, "minoritaria": False, "contesta": [], "poggia_su": [], "fallback": True,
                "kind": "civil_fallback"}
    if a.get("obiettivo") == CONTENIMENTO and _family_key(posture or {}) in ("penale", ""):
        return {"tesi": "Contenimento della pena: riti alternativi, attenuanti e commisurazione della pena",
                "pro": "non contesta alcun elemento integrato" + (
                    f"; {'; '.join(x.split(':')[0] for x in riti[:3])}" if riti else ""),
                "contro": "presuppone l'accertamento della responsabilità", "realismo": ALTO, "minoritaria": False,
                "contesta": [], "poggia_su": [], "fallback": True}
    return {"tesi": "Tesi da costruire sugli elementi ancora da provare", "pro": "gli elementi aperti",
            "contro": "il materiale non basta a fondarla", "realismo": BASSO, "minoritaria": False, "contesta": [],
            "poggia_su": [], "fallback": True, "placeholder": True}     # v75: a failure state, never printed


def coherent_solutions(plan: Dict[str, Any], records: Dict[str, Dict[str, Any]], a: Dict[str, Any],
                       riti: Sequence[str] = (), log: Optional[List[Dict[str, Any]]] = None,
                       posture: Optional[Dict[str, Any]] = None, circ: Optional[Dict[str, Dict[str, Any]]] = None
                       ) -> Tuple[List[Dict[str, Any]], int]:
    """(solutions, index of the main one), by the ids each thesis declares (v71, rule 1 as the PO restated it):
      - a thesis declaring no element (contesta and poggia_su empty) is unverifiable (UNVERIFIABLE_LABEL): never main;
      - a thesis contesting an element the record marks «integrato» must say it is a minority one: labelled
        «minoritaria», it stays, realismo basso (INTEGRATED_NOTE); unlabelled, it is barred (BARRED_LABEL), never main;
      - a minority thesis is main only when the record offers no sustainable non-minority line -- and with the charge
        fully integrato the posture's «ammettere la condotta … contenere la pena» always is one (posture_fallback);
      - among the eligible, the model's choice when it is one, else the most realistic; none: the fallback line."""
    known = elements_by_id(records, circ)
    sols = [dict(s_) for s_ in plan.get("soluzioni") or []]
    for t in sols:
        # v79 (A1): a thesis the code enumerated is verifiable by construction -- its ids, or none, are the code's (an
        # eccezione, a remedy with nothing proven yet); only a thesis of the model's own (no candidate id) declaring
        # no element is unverifiable. Before v79 those candidates were lost here, and the fallback led.
        if not (t.get("contesta") or t.get("poggia_su")) and not t.get("candidate"):
            t["unverifiable"] = True
            if log is not None:
                log.append({"tesi": t["tesi"][:160], "unverifiable": True})
            continue
        hit = barred(t, known)
        if not hit:
            continue
        if t.get("minoritaria"):
            t.update({"contests_integrated": hit, "realismo": BASSO})
            if log is not None:
                log.append({"tesi": t["tesi"][:160], "minority_contests": hit})
        else:
            t.update({"barred": hit, "realismo": BASSO})
            if log is not None:
                log.append({"tesi": t["tesi"][:160], "barred": hit, "elements": {x: known[x]["text"] for x in hit}})
    eligible = [i for i, t in enumerate(sols) if not t.get("barred") and not t.get("unverifiable")]
    majority = [i for i in eligible if not sols[i].get("minoritaria")]
    wanted = (plan.get("tesi_principale") or 1) - 1
    proven = lambda t: sum(1 for x in t.get("poggia_su") or [] if (known.get(x) or {}).get("status") == PROVATO)
    # v76: by realism, then by the proven elements it rests on, then the candidates' order -- a line with nothing
    # proven never leads over an «alto» one; the model's own pick only in the old path (no code candidates)
    rank = lambda pool: sorted(pool, key=lambda i: (REALISMO.index(sols[i]["realismo"]), -proven(sols[i]), i))
    if any(t.get("candidate") for t in sols):
        wanted = -1

    def add_fallback() -> int:
        sols.insert(0, fallback_thesis(a, riti, posture))
        if log is not None:
            log.append({"fallback": sols[0]["tesi"]})
        return 0
    if majority:
        main = wanted if wanted in majority else rank(majority)[0]
    elif a.get("all_integrated"):           # the containment line is a sustainable non-minority one: it goes first
        main = add_fallback()
    elif eligible:
        main = wanted if wanted in eligible else rank(eligible)[0]
    else:
        main = add_fallback()
    return sols, main


def coherent_text(text: str, records: Dict[str, Dict[str, Any]], a: Dict[str, Any],
                  log: Optional[List[Dict[str, Any]]] = None) -> str:
    """The text without a sentence promising a favourable outcome the record does not give (the record's elements are
    checked through the theses' declared ids, never by matching words)."""
    out = []
    for para in (text or "").split("\n"):
        kept = []
        for sentence in _SENT_SPLIT.split(para):
            if a.get("esito") != FAVOREVOLE and _FAVOUR_RE.search(sentence):
                if log is not None:
                    log.append({"sentence": sentence[:200], "why": "promette un esito favorevole che il registro non dà"})
                continue
            kept.append(sentence)
        out.append(" ".join(kept))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


# v71: orientations are the code's (orientations_section) -- a sentence of the writer attributing one to the case law is
# never printed
_ORIENTATION_CLAIM_RE = re.compile(
    r"\b(?:orientament\w*|indirizz\w*|giurisprudenz\w*)\s+(?:\w+\s+){0,2}(?:prevalent\w*|maggioritari\w*|"
    r"minoritari\w*|consolidat\w*|costant\w*|dominant\w*|pacific\w*|recent\w*|di legittimità)|"
    r"\bSezioni\s+Unite\b|\bla\s+(?:Suprema\s+)?Corte\s+(?:di\s+[Cc]assazione\s+)?(?:ha|afferma|ritiene|ammette|"
    r"esclude|richiede)|\bsecondo\s+la\s+(?:giurisprudenza|Cassazione|Corte)|\bla\s+Cassazione\b", re.I)


def orientation_claims_out(text: str, log: Optional[List[str]] = None) -> str:
    out = []
    for para in (text or "").split("\n"):
        kept = []
        for sentence in _SENT_SPLIT.split(para):
            if _ORIENTATION_CLAIM_RE.search(sentence):
                if log is not None:
                    log.append(sentence[:200])
                continue
            kept.append(sentence)
        out.append(" ".join(kept))
    return "\n".join(out)


# v71: the procedural lines agree with the charge's procedibilità -- no remissione / 162-ter for a d'ufficio offence
_REMISSION_RE = re.compile(r"\bremission\w*|\b162-ter\b|\bcondotte\s+riparatorie\b", re.I)
_OFFENCE_REF_RE = re.compile(r"\b(\d{3}(?:-(?:bis|ter|quater|quinquies))?)\s*(?:,\s*co\.\s*\d+\s*,?\s*)?c\.p\.(?!p)", re.I)
_REMISSION_ARTICLES = {"152", "162-ter", "62", "62-bis", "133"}


def procedibility(records: Dict[str, Dict[str, Any]], rows: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    """{label: «ufficio» | «querela»} of the record's offences, from the procedural store (procedural_facts): a querela
    offence whose d'ufficio condition the record proves is d'ufficio."""
    from . import procedural_facts
    facts = procedural_facts.by_number(rows or [])
    out = {}
    for label, rec in (records or {}).items():
        number, _, code = label.partition(" ")
        if code != "c.p." or number not in facts:
            continue
        f = facts[number]
        if rec.get("procedibilita"):
            f = procedural_facts.with_statuses(f, rec["procedibilita"])
        if f.get("querela") is False or (f.get("querela") and f.get("ufficio_esito") == PROVATO):
            out[label] = "ufficio"
        elif f.get("querela"):
            out[label] = "querela"
    return out


def procedibility_out(text: str, procedibilita: Dict[str, str], charge: str, log: Optional[List[str]] = None) -> str:
    """A line or sentence on remissione / 162-ter dropped when it is about a d'ufficio offence: the offences it names
    (the remissione's own articles aside), else the charge."""
    if not procedibilita:
        return text
    out = []
    for line in (text or "").split("\n"):
        if not _REMISSION_RE.search(line):
            out.append(line)
            continue
        kept = []
        for sentence in _SENT_SPLIT.split(line):
            if _REMISSION_RE.search(sentence):
                named = [f"{n} c.p." for n in _OFFENCE_REF_RE.findall(sentence) if n not in _REMISSION_ARTICLES]
                about = named or ([charge] if charge else [])
                if about and all(procedibilita.get(x) == "ufficio" for x in about):
                    if log is not None:
                        log.append(sentence[:200])
                    continue
            kept.append(sentence)
        if kept:
            out.append(" ".join(kept))
    return "\n".join(out)


def _words(text: str) -> set:
    return {w[:5] for w in re.findall(r"[a-zà-ü]{4,}", (text or "").lower())}


def same_line(a: str, b: str) -> bool:
    """Two strategy lines say the same thing: half their words shared, or the shorter one's nearly all in the other."""
    x, y = _words(a), _words(b)
    if not x or not y:
        return False
    common = len(x & y)
    return common / len(x | y) >= 0.5 or common / min(len(x), len(y)) >= 0.8


# --- «da verificare»: out of the body, into the last section --------------------------------------------------------------

_PAREN_RE = re.compile(r"\s*\((?:([^()\n]*?);\s*)?(?:orientamento\s+)?da verificare(?::\s*([^()\n]*))?\)", re.I)
_COLON_RE = re.compile(r"(?<![«\w])da verificare\s*:\s*([^.\n;]+)[.;]?", re.I)
_TAIL_RE = re.compile(r"\s*[,;—–-]\s*(?:(?:l['’]|un['’]?\s*)?orientamento\s+)?da verificare\b\.?", re.I)
_BARE_RE = re.compile(r"\(\s*(?:orientamento\s+)?da verificare\s*\)|\b(?:orientamento\s+)?da verificare\b", re.I)
POINT_REF = "v. punto {n}"


_REF_RE = re.compile(r"\s*\(\s*v\. punto \d+\s*\)|\bv\. punto \d+\.?", re.I)
_MARK_COLON_RE = re.compile(r"(?<![«\w])(?:(?:l['’]|un['’]?\s*)?orientamento\s+)?da verificare\s*:\s*", re.I)
_ITEM_RE = re.compile(r"^\s*(?:[-•*]\s+|\d+[.)]\s+)?(?:\*\*)?")
# the marker as the sentence's predicate («la data è da verificare»): the point keeps it, the body loses the verb with it
_COPULA_RE = re.compile(r"\s+\b(?:è|sono|resta|restano|rimane|rimangono|va|vanno)\s+(?:ancora\s+)?da verificare\b", re.I)
_KEEP = "\x00"


def _sentence_at(text: str, start: int, end: int) -> str:
    """v74: the whole sentence (or the open question) the marker at [start, end) sits in -- the line's sentences cut
    where the answer's own split cuts them (abbreviations kept), a list item's bullet dropped."""
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    line = text[line_start:line_end if line_end >= 0 else len(text)]
    at = start - line_start
    pieces, begin = [], 0
    for m in _SENT_SPLIT.finditer(line):
        pieces.append((begin, m.start()))
        begin = m.end()
    pieces.append((begin, len(line)))
    k = next(i for i, (b, e) in enumerate(pieces) if at < e or i == len(pieces) - 1)
    if k and not point_sentence(line[pieces[k][0]:pieces[k][1]]):     # the marker alone after the open question
        return line[pieces[k - 1][0]:pieces[k][1]]
    return line[pieces[k][0]:pieces[k][1]]


def point_sentence(sentence: str) -> str:
    """v74: a point as the reader reads it -- the sentence with its markers and references taken out (a «(…; da
    verificare: X)» keeps its words, a «da verificare: X» keeps X), closed by its own stop or question mark."""
    x = re.sub(r"^\s*\*\*[^*\n]{1,60}\.\*\*\s+(?=\S)", "", sentence or "")     # a «**Risposta.**» label, not the point
    x = _REF_RE.sub("", x)
    x = _COPULA_RE.sub(lambda m: m.group(0).replace("da verificare", _KEEP), x)
    x = _PAREN_RE.sub(lambda m: " (" + "; ".join(_plain(g) for g in m.groups() if _plain(g)) + ")"
                      if any(_plain(g) for g in m.groups()) else "", x)
    x = _MARK_COLON_RE.sub("", x)
    x = _TAIL_RE.sub("", x)
    x = _BARE_RE.sub("", x)
    x = _ITEM_RE.sub("", x)
    x = re.sub(r"\*\*", "", x)
    x = re.sub(r"\(\s*\)", "", x)
    x = re.sub(r"\s+([.,;:?!)])", r"\1", x)
    x = re.sub(r"([:—–-])\s*([.?!])$", r"\2", " ".join(x.split()))
    x = x.strip(" ,;:—–-").replace(_KEEP, "da verificare")
    if x and x[-1] not in ".?!":
        x += "."
    return x[0].upper() + x[1:] if x else x


def same_point(a: str, b: str) -> bool:
    """v74: two points say the same: equal once plain, or nearly every word of the shorter one in the other."""
    if _plain(a).rstrip(".;:?").lower() == _plain(b).rstrip(".;:?").lower():
        return True
    x, y = _words(a), _words(b)
    return min(len(x), len(y)) >= 3 and len(x & y) / min(len(x), len(y)) >= 0.85


def collect_points(sections: List[Tuple[str, str]], points: Optional[List[str]] = None
                   ) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Every «da verificare» marker of the sections taken out: its point collected (numbered once, the same point
    once), in the body only the reference «(v. punto N)» -- the reader still sees what is open, the text says it once.
    v74: the point is the whole sentence (or the open question) that carried the marker, never a clause of it
    (point_sentence); a point that says what an earlier one says is merged into it (same_point: the fuller kept)."""
    points = list(points or [])

    def number(point: str) -> int:
        for i, p in enumerate(points, 1):
            if same_point(p, point):
                if len(point) > len(p) and _words(p) <= _words(point):
                    points[i - 1] = point
                return i
        points.append(point)
        return len(points)

    out = []
    for heading, text in sections:
        def point_at(m) -> int:
            start = m.start() + len(m.group(0)) - len(m.group(0).lstrip())     # the marker, not the newline before it
            point = point_sentence(_sentence_at(text, start, m.end()))
            return number(point) if point else 0          # v75b: no sentence, no point (never the heading)

        def ref(m, form: str, none: str = "") -> str:
            n = point_at(m)
            return form.format(ref=POINT_REF.format(n=n)) if n else none

        def paren(m):
            prefix = _plain(m.group(1))
            return ref(m, f" ({prefix}; {{ref}})" if prefix else " ({ref})", f" ({prefix})" if prefix else "")

        def colon(m):
            return ref(m, "{ref}.")

        def tail(m):
            return ref(m, " ({ref}).", ".")

        def bare(m):
            return ref(m, "({ref})")

        def copula(m):
            return ref(m, " ({ref})")

        for pattern, repl in ((_PAREN_RE, paren), (_COLON_RE, colon), (_TAIL_RE, tail), (_COPULA_RE, copula),
                              (_BARE_RE, bare)):
            text = pattern.sub(repl, text)
        text = re.sub(r"\.\s*\.", ".", text)
        text = re.sub(r"[ \t]+([.,;])", r"\1", text)
        out.append((heading, text))
    return out, points


def points_section(points: Sequence[str]) -> str:
    return "\n".join(f"{i}. {p}" for i, p in enumerate(points, 1)) if points else \
        "Nessun punto resta da verificare sui fatti narrati."


# --- the layout: his ten sections -----------------------------------------------------------------------------------------

NORMS_FROM = ("Norme applicate ai fatti", "Norme")
PROCEDURE_FROM = ("Profili processuali",)
DROPPED = ("Risposta e strategia", "Sintesi e strategia", "Obiezioni e repliche", "Conclusioni")
ANALYSIS_TITLE = "**Analisi per elementi**"      # his «Norme e analisi per elementi»: the element tables under the norms
STRATEGY_TITLE = "**Strategia**"


def solutions_section(sols: List[Dict[str, Any]], main: int, plan: Dict[str, Any], riti: Sequence[str],
                      stragiudiziale: str, log: Optional[Dict[str, Any]] = None) -> str:
    """The table (the main thesis first, labelled; a minority one declared; a barred one with BARRED_LABEL and the
    elements it contests) and the strategy: the main line -- the posture's containment line carries the riti --, the
    subordinate line only when it differs from the main one (same_line), the stragiudiziale option."""
    ordered = [main] + [i for i in range(len(sols)) if i != main]
    lines = ["| Tesi | Pro | Contro | Realismo |", "|---|---|---|---|"]
    for i in ordered:
        s_ = sols[i]
        label = s_["tesi"] + (" (tesi principale)" if i == main else "") + (" (tesi minoritaria)"
                                                                           if s_.get("minoritaria") else "")
        if s_.get("barred"):
            label += f" ({BARRED_LABEL}: {', '.join(s_['barred'])})"
        if s_.get("contests_integrated"):
            label += f" ({INTEGRATED_NOTE}: {', '.join(s_['contests_integrated'])})"
        if s_.get("unverifiable"):
            label += f" ({UNVERIFIABLE_LABEL})"
        lines.append(f"| {_cell(label)} | {_cell(s_.get('pro'))} | {_cell(s_.get('contro'))} | {_cell(s_['realismo'])} |")
    main_text = sols[main]["tesi"]
    strategy = [STRATEGY_TITLE, "", f"- Linea principale: {main_text.rstrip('.')}."]
    with_main = bool(sols[main].get("fallback"))         # the containment line: the riti are its content
    if with_main:
        strategy += [f"  - {x}" for x in riti]
    sub = plan.get("subordinata") or ""
    if sub and same_line(sub, main_text):
        if log is not None:
            log.setdefault("subordinate_omitted", []).append(sub[:200])
        sub = ""
    sub_riti = [] if with_main else list(riti)
    if sub or sub_riti:
        strategy.append(f"- Linea subordinata: {sub.rstrip('.') + '.' if sub else ''}".rstrip())
        strategy += [f"  - {x}" for x in sub_riti]
    if plan.get("stragiudiziale") and log is not None and not same_line(plan["stragiudiziale"], stragiudiziale or ""):
        log["stragiudiziale_model_dropped"] = str(plan["stragiudiziale"])[:200]      # v76: the table's line only
    strategy.append(f"- Opzione stragiudiziale: {(stragiudiziale or '').rstrip('.')}.")
    return "\n".join(lines + [""] + strategy)


def objections_section(plan: Dict[str, Any], fallback: Sequence[Dict[str, str]] = ()) -> str:
    items = list(plan.get("obiezioni") or [])
    for f in fallback:
        if len(items) >= OBJECTIONS:
            break
        if f.get("obiezione") and all(f["obiezione"] != x["obiezione"] for x in items):
            items.append(f)
    items = items[:OBJECTIONS]
    if not items:
        return "Il materiale non indica obiezioni della controparte sui fatti narrati."
    return "\n\n".join(f"{OBJECTION_HEAD.format(n=i)} {o['obiezione'].rstrip('.')}.\n{ANSWER_HEAD} "
                         f"{o.get('risposta') or NO_ANSWER}" for i, o in enumerate(items, 1))


OBJECTION_HEAD, ANSWER_HEAD = "**Obiezione {n}.**", "**Risposta.**"
NO_ANSWER = "Va costruita sui fatti che le indagini difensive o l'istruttoria porteranno."
OBJ_PROVEN = {"penale": ("L'accusa sosterrà che è provato l'elemento «{el}»: {fact}",
                         "I fatti narrati lo sostengono: la difesa non lo nega, ma ne trae le conseguenze sulla "
                         "qualificazione e sulla pena (v. Soluzioni a confronto)."),
              "civile": ("La controparte sosterrà che è provato l'elemento «{el}»: {fact}",
                         "I fatti narrati lo sostengono: la difesa non lo nega e si concentra sugli elementi che restano "
                         "da provare e sulla misura della domanda.")}
OBJ_OPEN = {"penale": ("L'accusa cercherà di provare l'elemento «{el}»",
                       "Il caso non lo dice: la prova spetta all'accusa, e la difesa ne contesta la sufficienza."),
            "civile": ("La controparte cercherà di provare l'elemento «{el}»",
                       "Il caso non lo dice: la prova spetta a chi lo afferma (art. 2697 c.c.).")}
OBJ_DENIED = ("La controparte negherà l'elemento «{el}»", "Va provato: {missing}.")


def objections_fallback(records: Dict[str, Dict[str, Any]], a: Dict[str, Any], penal: bool,
                        by_id: Dict[str, Dict[str, Any]], titles_: Sequence[str] = ()) -> List[Dict[str, str]]:
    """Objections from the record when the method plan gives fewer than three: defending, the main charge's proven
    elements (conceded, answered on qualification and pena) and its open ones (the burden); claiming, the client's
    open elements (the proof to acquire); then our objections' titles."""
    out = []
    label = a.get("article") or ""
    rec = (records or {}).get(label) or {}
    family = "penale" if penal else "civile"
    for r in _live(rec):
        el = _short(r.get("text"), 100)
        if rec.get("side") == "threat" and r.get("status") == PROVATO:
            q, ans = OBJ_PROVEN[family]
            out.append({"obiezione": q.format(el=el, fact=_sentence_text(r.get("supporting") or [], by_id)
                                              or "i fatti narrati").rstrip("."), "risposta": ans})
        elif rec.get("side") == "threat" and r.get("status") == DA_PROVARE:
            q, ans = OBJ_OPEN[family]
            out.append({"obiezione": q.format(el=el), "risposta": ans})
        elif rec.get("side") != "threat" and r.get("status") == DA_PROVARE:
            out.append({"obiezione": OBJ_DENIED[0].format(el=el),
                        "risposta": OBJ_DENIED[1].format(missing=_short(r.get("missing") or r.get("text"), 120))})
    out += [{"obiezione": t, "risposta": ""} for t in titles_]
    return out


MASSIMA_CHARS = 420
NO_POINTS_LINE = ORIENTATION_TO_VERIFY


def _massima(text: str) -> str:
    text = _plain(text)
    if len(text) <= MASSIMA_CHARS:
        return text
    cut = text[:MASSIMA_CHARS]
    end = max(cut.rfind(". "), cut.rfind("; "))
    return (cut[:end + 1] if end > MASSIMA_CHARS // 2 else cut.rsplit(" ", 1)[0]) + " …"


def retrieved_ruling(pronuncia: str, rulings: Sequence[Any]) -> Optional[Dict[str, Any]]:
    """The retrieved ruling the model names (its document id, or its label / number), with a massima to quote."""
    text = _plain(pronuncia)
    if not text:
        return None
    for r in rulings or []:
        if not isinstance(r, dict) or not r.get("massima"):
            continue
        if text == str(r.get("id") or "") or ruling_ok(text, [str(r.get("label") or "")]):
            return r
    return None


def orientations_section(plan: Dict[str, Any], rulings: Sequence[Any], log: Optional[List[str]] = None,
                         points: Optional[List[str]] = None) -> str:
    """Code-rendered (v71): per controverted point the model names, a retrieved ruling -- its label, its document id,
    its massima quoted -- or the single line «orientamento da verificare» (the point then in «Punti da verificare»).
    The model's own orientation text is never printed."""
    lines = []
    for o in plan.get("orientamenti") or []:
        point = _plain(o.get("punto") or o.get("tema")).rstrip(".:")
        if not point:
            continue
        found = retrieved_ruling(o.get("pronuncia") or "", rulings)
        if found:
            lines.append(f"- {point}: {found.get('label')} (documento {found.get('id')}): «{_massima(found['massima'])}»")
            continue
        if o.get("pronuncia") and log is not None:
            log.append(o["pronuncia"])
        lines.append(f"- {point}: {ORIENTATION_TO_VERIFY}")
        if points is not None:
            points.append(f"Orientamento sul punto: {point}")
    if not lines and points is not None:
        points.append("Orientamenti sui punti controversi")
    return "\n".join(lines) if lines else NO_POINTS_LINE


def orientations_text(ctx: Dict[str, Any], plan: Dict[str, Any], rulings: Sequence[Any], log: Optional[List[str]] = None,
                      points: Optional[List[str]] = None) -> str:
    """§5: v78 (RULINGS_RECORD), the registry's -- the rulings with their quoted passage, the computed orientations, the
    lawyer's pronunce di riferimento, the corpus line (rulings.section_text); else v71's (orientations_section)."""
    if ctx.get("rulings_record"):
        from . import rulings as rulings_mod
        return rulings_mod.section_text(ctx["rulings_record"], points)
    return orientations_section(plan, rulings, log, points)


def question_section(plan: Dict[str, Any], a: Dict[str, Any], records: Dict[str, Dict[str, Any]], client: str) -> str:
    if plan.get("questione_giuridica"):
        return plan["questione_giuridica"]
    label = a.get("article") or ""
    rec = (records or {}).get(label) or {}
    if not label:
        return "La questione giuridica è da verificare: i fatti narrati non indicano la norma principale."
    rubrica = f" ({_plain(rec.get('rubrica'))})" if rec.get("rubrica") else ""
    if rec.get("charge_hypothesis"):       # v79 (C4): no charge stated -- never «contestato»
        return (f"Se i fatti integrino l'art. {label}{rubrica}, {HYPOTHESIS_LABEL}, e quale "
                + ("difesa sia sostenibile." if rec.get("side") == "threat" else
                   f"tutela spetti a {client or 'la parte assistita'}."))
    if rec.get("side") == "threat":
        return f"Se i fatti integrino l'art. {label}{rubrica} contestato a {client or 'la parte assistita'} e quale difesa sia sostenibile."
    return f"Se i fatti integrino l'art. {label}{rubrica} e quale tutela spetti a {client or 'la parte assistita'}."


SUMMARY_LINES = 5          # his «Sintesi della soluzione (max 5 righe)»


def summary_section(plan: Dict[str, Any], sols: List[Dict[str, Any]], main: int, a: Dict[str, Any],
                    client: str) -> str:
    """The method plan's summary, at most SUMMARY_LINES sentences; without one, the main line and the record's
    assessment."""
    text = _plain(plan.get("sintesi"))
    if text:
        return " ".join(_SENT_SPLIT.split(text)[:SUMMARY_LINES])
    who = client or "la parte assistita"
    esito = {FAVOREVOLE: f"favorevole a {who}", SFAVOREVOLE: f"sfavorevole a {who}", INCERTO: "incerto"}.get(
        a.get("esito"), str(a.get("esito") or ""))
    return (f"Linea principale: {sols[main]['tesi'].rstrip('.')}. Esito probabile {esito} "
            f"({_plain(a.get('why')).rstrip('.')}); obiettivo realistico: {a.get('obiettivo')}.")


def final_check(answer: Dict[str, str], ctx: Dict[str, Any], log: Dict[str, Any], plan: Dict[str, Any]
                ) -> List[Dict[str, Any]]:
    """His CONTROLLO FINALE as the coherence checklist: each item answered from the code's own check, beside the
    model's answer to it (plan["controllo_finale"]); logged with the answer."""
    items = list(data().get("controllo_finale") or [])
    model = {str(x.get("voce")): x for x in plan.get("controllo_finale") or []}
    records = ctx.get("records") or {}
    norms = answer.get("norme", "")
    from . import issue_records
    counted = [x for x, r in records.items() if issue_records.counted(x, str(r.get("rubrica") or "")) and _live(r)
               and not r.get("applicazione")]
    main_row = next((ln for ln in answer.get("soluzioni", "").split("\n") if "(tesi principale)" in ln), "")
    code = [
        all(t in answer.get("fatti", "") for t in ("Fatti certi", "Fatti ipotizzabili", "Fatti mancanti")),
        all(f"**Art. {x}**" in norms for x in counted),
        bool(main_row) and BARRED_LABEL not in main_row and UNVERIFIABLE_LABEL not in main_row,
        not log.get("circumstance_claims"),
        not log.get("rulings_out") and not log.get("orientation_claims_out"),
        bool(ctx.get("temporal")),
        not log.get("subordinate_omitted") and not log.get("procedibility_out"),
        bool(ctx.get("assessment")) and not any("favorevole" in str(c.get("why")) for c in log.get("coherence") or []),
    ]
    how = ["le tre parti dei fatti, dalla sola narrazione", "una tabella per ogni norma con elementi",
           "la tesi principale dichiara i suoi elementi e non nega un elemento integrato senza dirsi minoritaria", "nessuna circostanza senza requisiti",
           "nessuna pronuncia fuori dal materiale, nessun orientamento scritto dal modello",
           "riga sulla disciplina applicabile", "subordinata diversa dalla principale, procedibilità coerente",
           "esito dal registro, nessuna promessa"]
    out = []
    for i, item in enumerate(items, 1):
        ok = code[i - 1] if i <= len(code) else None
        m = model.get(str(i)) or {}
        out.append({"voce": i, "domanda": item, "codice": ok, "come": how[i - 1] if i <= len(how) else "",
                    "modello": m.get("esito") or "", "nota": m.get("nota") or ""})
    return out


# --- v72: what follows from the code's selection -- never from the model's plan ---------------------------------------

NOTES_TITLE = "NOTE DI INQUADRAMENTO (dell'avvocato)"
MAIN_LINE, SUB_LINE, BARRED_LINE = "**Linea principale**", "**Linea subordinata**", "**Tesi non sostenibili**"
STEPS_MAX = 5


def eligible(t: Dict[str, Any]) -> bool:
    return not t.get("barred") and not t.get("unverifiable")


def subordinate_of(sols: Sequence[Dict[str, Any]], main: int) -> Optional[int]:
    """The subordinate line: the most realistic eligible thesis after the main one that is not the same line."""
    pool = [i for i, t in enumerate(sols) if i != main and eligible(t) and not same_line(t["tesi"], sols[main]["tesi"])]
    return sorted(pool, key=lambda i: REALISMO.index(sols[i]["realismo"]))[0] if pool else None


def _open_ids(ctx: Dict[str, Any]) -> List[str]:
    return [m["elemento"] for m in (ctx.get("facts") or {}).get("mancanti") or [] if m.get("elemento")]


def conclusions_from_selection(a: Dict[str, Any], sols: Sequence[Dict[str, Any]], main: int, sub: Optional[int],
                               ctx: Dict[str, Any], client: str) -> str:
    """§9 from the selection: the record's assessment (what it depends on: the elements still to prove; the residual
    risk: the charge or the claim when it is fully integrato, else those elements), the main line with its argument,
    the subordinate one, the theses barred or unverifiable."""
    penal = (ctx.get("posture") or {}).get("family") != "civile"
    art = a.get("article") or ""
    open_ = _open_ids(ctx)
    depends = f"la prova degli elementi {', '.join(open_)}" if open_ else ""
    if a.get("all_integrated") and art:
        risk = f"la condanna per l'art. {art}" if penal else f"l'accoglimento della domanda avversaria (art. {art})"
    elif open_:
        risk = (f"che l'accusa provi gli elementi {', '.join(open_)}" if (ctx.get("posture") or {}).get("role") in ACCUSED
                else f"che non si provino gli elementi {', '.join(open_)}")
    else:
        risk = ""
    m = sols[main]
    lines = [assessment_text(a, client, m["tesi"], depends, risk)]
    lines.append(f"{MAIN_LINE}: {m['tesi'].rstrip('.')}" + (f" — {m['argomento'].rstrip('.')}" if m.get("argomento")
                                                           else "") + ".")
    if sub is not None:
        lines.append(f"{SUB_LINE}: {sols[sub]['tesi'].rstrip('.')}" + (" (tesi minoritaria)" if sols[sub].get("minoritaria")
                                                                       else "") + ".")
    out = [f"«{t['tesi'].rstrip('.')}» (contesta {', '.join(t['barred'])}, integrato)" for t in sols if t.get("barred")]
    out += [f"«{t['tesi'].rstrip('.')}» (non dichiara gli elementi)" for t in sols if t.get("unverifiable")]
    if out:
        lines.append(f"{BARRED_LINE}: " + "; ".join(out) + ".")
    return "\n".join(lines)


STEP_TEXTS = {   # my drafts: v73, the next steps of the selected line, one per element it contests or rests on
    "acquire": ("Acquisire la prova: {fact}", "decide l'elemento {id} dell'art. {art}, su cui si regge la linea "
                                              "principale"),
    "document": ("Documentare l'elemento «{el}» ({id}) dell'art. {art}", "la linea principale si regge su di esso"),
    "contest_open": ("Verificare quale prova {who} ha di: {fact}",
                     "decide l'elemento {id} dell'art. {art}, che la linea principale contesta e il caso non prova"),
    "contest_excluded": ("Fissare i fatti che escludono l'elemento «{el}» ({id}) dell'art. {art}",
                         "la linea principale si fonda sulla sua esclusione"),
    "contest_integrated": ("Verificare se i fatti integrano davvero l'elemento «{el}» ({id}) dell'art. {art}",
                           "la linea principale lo contesta, ma i fatti narrati lo integrano (tesi minoritaria)"),
    "riti": ("Valutare i riti alternativi e le attenuanti", "contenere la pena ({riti})"),
}


def steps_from_selection(ctx: Dict[str, Any], sols: Sequence[Dict[str, Any]], main: int) -> List[Dict[str, str]]:
    """v73: the next steps only from the selected line's own elements -- each element it rests on (poggia_su): to
    acquire the proof of (its missing fact) when the record does not prove it, to document when it does; each one it
    contests (contesta): the counterparty's proof to verify, the facts excluding it to fix, or -- integrato, a minority
    line -- whether the facts really integrate it. The containment line rests on no element: its steps are its own
    content, the riti and the attenuanti. An article without decomposed elements yields none; nothing from the other
    theses, nor from the record's other missing facts."""
    m = sols[main]
    penal = (ctx.get("posture") or {}).get("family") != "civile"
    who = "l'accusa" if penal else "la controparte"
    known = elements_by_id(ctx.get("records") or {}, ctx.get("circumstances"))
    missing = {x["elemento"]: x for x in (ctx.get("facts") or {}).get("mancanti") or [] if x.get("elemento")}
    steps: List[Dict[str, str]] = []

    def add(key: str, **kw: str) -> None:
        passo, scopo = STEP_TEXTS[key]
        steps.append({"passo": passo.format(**kw), "scopo": scopo.format(**kw)})
    if m.get("fallback") or m.get("kind") == "contenimento":      # v75b: its own content, no element to prove
        if ctx.get("riti"):
            add("riti", riti="; ".join(x.split(":")[0] for x in ctx["riti"][:2]))
        return steps[:STEPS_MAX]
    for eid in m.get("poggia_su") or []:
        e = known.get(eid)
        if not e:
            continue
        el = _short(e["text"], 100)
        if e["status"] == PROVATO:
            add("document", el=el, id=eid, art=e["article"])
        else:
            fact = (missing.get(eid) or {}).get("fatto") or f"l'elemento «{el}» ({eid})"
            add("acquire", fact=fact, id=eid, art=e["article"])
    for eid in m.get("contesta") or []:
        e = known.get(eid)
        if not e or eid in (m.get("poggia_su") or []):
            continue
        key = {PROVATO: "contest_integrated", NON_RISULTA: "contest_excluded"}.get(e["status"], "contest_open")
        el = _short(e["text"], 100)
        add(key, el=el, id=eid, art=e["article"], who=who,
            fact=(missing.get(eid) or {}).get("fatto") or f"l'elemento «{el}» ({eid})")
    return steps[:STEPS_MAX]


def thesis_objections(sols: Sequence[Dict[str, Any]], main: int, ctx: Dict[str, Any]) -> List[Dict[str, str]]:
    """v72: objections from the selection, when the model's and the record's are fewer than three -- the main line's
    own weak point, the counterparty against the main line, the integrated elements it will lean on."""
    penal = (ctx.get("posture") or {}).get("family") != "civile"
    who = "L'accusa" if penal else "La controparte"
    m = sols[main]
    out = []
    if m.get("contro"):
        out.append({"obiezione": m["contro"], "risposta": m.get("argomento") or m.get("pro") or ""})
    out.append({"obiezione": f"{who} contesterà la linea principale: {m['tesi'].rstrip('.')}",
                "risposta": m.get("argomento") or m.get("pro") or ""})
    integrated = [x for x, e in elements_by_id(ctx.get("records") or {}).items() if e["status"] == PROVATO
                  and e.get("side") == "threat"]
    if integrated:
        out.append({"obiezione": f"{who} sosterrà che gli elementi integrati ({', '.join(integrated)}) bastano",
                    "risposta": "La difesa non li nega: la discussione si sposta sulla linea scelta e sulla misura "
                                "dell'esito (v. Soluzioni a confronto)."})
    if m.get("poggia_su"):
        out.append({"obiezione": f"{who} negherà gli elementi su cui si regge la linea principale "
                                 f"({', '.join(m['poggia_su'])})",
                    "risposta": "Vanno provati: v. i passi successivi nelle Conclusioni."})
    out.append({"obiezione": f"{who} contesterà i fatti narrati su cui il parere si fonda",     # the last resort
                "risposta": "Sono i fatti come il cliente li riferisce: vanno provati con i mezzi indicati nei passi "
                            "successivi."})
    return out


def compose(sections: Sequence[Tuple[str, str]], ctx: Dict[str, Any], plan: Dict[str, Any],
            profiles: Sequence[str] = (), log: Optional[Dict[str, Any]] = None) -> str:
    """The answer in the ten sections of his STRUTTURA DELL'OUTPUT, from our checked sections (sections_of the draft),
    the context (prepare()) and the method plan; his CONTROLLO FINALE answered and logged (final_check)."""
    log = log if log is not None else {}
    records = ctx.get("records") or {}
    a = ctx.get("assessment") or {}
    client = ctx.get("client") or ""
    rulings = ctx.get("rulings") or []
    circ = ctx.get("circumstances") or {}
    by_title = {}
    others: List[Tuple[str, str]] = []
    head = ""
    for t, x in sections:
        if not t:
            head = x.strip()
        elif t in NORMS_FROM or t in PROCEDURE_FROM:
            by_title[t] = x.strip()
        elif t not in DROPPED:
            others.append((t, x.strip()))
    claims: List[Dict[str, Any]] = []
    rulings_out: List[str] = []
    orientation_claims: List[str] = []
    procedural_out: List[str] = []
    procedibilita = ctx.get("procedibilita") or {}

    from . import fact_terms
    certi = list((ctx.get("facts") or {}).get("certi") or [])
    accused = (ctx.get("posture") or {}).get("role") in ACCUSED
    fact_wording: List[Dict[str, Any]] = []

    scope_log: List[str] = []           # v78: rulings outside their article / element
    articles_log: List[str] = []        # v78: the plan's sentences citing an article outside the scope

    from . import rule_statements
    rule_src = rule_statements.sources(ctx)
    rules_log: List[Dict[str, str]] = []        # v79 (B1): the rule statements removed, with their reason
    objections_dropped: List[str] = []          # v79 (B2): an objection or a reply the checks emptied

    def prose(text: str, rules: bool = True) -> str:   # every written paragraph: no orientation of its own, retrieved
        text = orientation_claims_out(text, orientation_claims)            # rulings only, circumstances with their
        text = unretrieved_rulings_out(text, rulings, rulings_out)         # requisites, no remissione for a d'ufficio
        text = rulings_scope_out(text, rulings, scope_log)                 # charge; v78: on its article and element
        text = procedibility_out(text, procedibilita, a.get("article") or "", procedural_out)
        text = fact_terms.apply(text, certi, ctx.get("case") or "", client, accused, fact_wording)     # v74
        text = circumstance_claims(text, circ, claims)
        return rule_statements.check(text, rule_src, rules_log) if rules else text      # v79 (B1): grounded rules
    charged_log: List[str] = []         # v75b: the charged article's paragraph from its sussunzione rows only
    norms = prose(charged_blocks(next((by_title[t] for t in NORMS_FROM if by_title.get(t)), ""),
                                 ctx.get("charged_prose") or {}, charged_log))
    quesiti = [(t, x) for t, x in others if t.endswith("?")]
    normative = [(t, x) for t, x in others if not t.endswith("?")]
    norms_parts = [norms] if norms else []
    norms_parts += [f"**{t}**\n\n{prose(x)}" for t, x in normative if x]
    if ctx.get("subsumption"):
        norms_parts.append(f"{ANALYSIS_TITLE}\n\n{ctx['subsumption']}")
    if profiles:
        from . import interaction_copy as C
        norms_parts.append(f"**{C.OTHER_PROFILES_SECTION}**\n\n" + "\n".join(profiles))
    procedure = next((by_title[t] for t in PROCEDURE_FROM if by_title.get(t)), "")
    extra = [x for x in [ctx.get("temporal")] + [f"Litisconsorzio: {x}" for x in ctx.get("litisconsorzio") or []] if x]
    procedure = "\n\n".join([p for p in [prose(procedure)] + extra if p]) or "Nessun profilo processuale calcolato sul caso."
    coherence: List[Dict[str, Any]] = []
    sols, main = coherent_solutions(plan, records, a, ctx.get("riti") or [], coherence, ctx.get("posture") or {}, circ)
    if any(t.get("candidate") for t in sols):   # v76: a thesis the model added (no candidate id) is never displayed
        added = [t["tesi"][:160] for i, t in enumerate(sols) if not t.get("candidate") and not t.get("fallback")
                 and i != main]
        if added:
            keep = [i for i, t in enumerate(sols) if t.get("candidate") or t.get("fallback") or i == main]
            main, sols = keep.index(main), [sols[i] for i in keep]
            log["added_theses_dropped"] = added
    if sols[main].get("placeholder"):       # v75: never printed -- "" keeps the standard body (the caller's), logged
        log["refused"] = {"why": "placeholder", "tesi": sols[main]["tesi"]}
        _log("compose_refused", log["refused"])
        return ""
    a = settle_objective(a, sols, records, circ)
    scope = article_scope(ctx, [norms, procedure] + list(by_title.values()))       # v78: what record_check passed

    def planned(text: str, code: bool = False) -> str:     # v78: the plan's prose -- the checks above, then the
        return articles_out(prose(text, rules=not code), scope, articles_log)      # articles' scope (v79: a code-
    for t in sols:                          # the table's cells are prose too          # written text: its sources)
        code = t.get("argued") == "code" or (bool(t.get("fallback")) and not t.get("candidate"))
        t["pro"], t["contro"] = planned(t.get("pro") or "", code), planned(t.get("contro") or "", code)
        t["argomento"] = planned(coherent_text(t.get("argomento") or "", records, a, coherence), code)
    sub = subordinate_of(sols, main)        # v72: the subordinate line, the summary, the conclusions and the next steps
    objections_reviewed = []                # v78: the objection itself reviewed too, not only its answer
    for o in plan.get("obiezioni") or []:
        said = planned(o.get("obiezione") or "")
        if not said:
            articles_log.append(_removed_note(o))
            objections_dropped.append(_removed_note(o))
            continue
        reply = o.get("risposta") or ""
        answered = planned(coherent_text(reply, records, a, coherence))
        if reply.strip() and not answered:  # v79 (B2): a reply the checks emptied drops its objection -- the
            objections_dropped.append(_removed_note({"obiezione": said}, "reply"))      # the fallback fills §7
            continue
        objections_reviewed.append({**o, "obiezione": said, "risposta": answered})
    plan = {**plan, "subordinata": sols[sub]["tesi"] if sub is not None else "",      # are the selection's -- never
            "stragiudiziale": planned(plan.get("stragiudiziale") or ""),                 # the model's plan
            "obiezioni": objections_reviewed,
            "questione_giuridica": planned(plan.get("questione_giuridica") or "")}
    conclusions = [conclusions_from_selection(a, sols, main, sub, ctx, client)]
    steps = steps_text(steps_from_selection(ctx, sols, main))
    if steps:
        conclusions.append(steps)
    conclusions += [f"**{t}**\n\n{prose(x)}" for t, x in quesiti if x]
    fallback_objections = list(ctx.get("objections_fallback") or []) + thesis_objections(sols, main, ctx)
    log["objections_from"] = {"model": len(plan.get("obiezioni") or []),
                              "fallback": max(0, OBJECTIONS - len(plan.get("obiezioni") or []))}
    orientation_points: List[str] = []
    layout_log: Dict[str, Any] = {}
    keyed = [
        ("sintesi", summary_section({}, sols, main, a, client)),
        ("fatti", facts_section(ctx.get("facts") or {})),
        ("questione", question_section(plan, a, records, client)),
        ("norme", "\n\n".join(norms_parts) or "Nessuna norma applicabile nel materiale recuperato."),
        ("orientamenti", orientations_text(ctx, plan, rulings, rulings_out, orientation_points)),
        ("soluzioni", solutions_section(sols, main, plan, ctx.get("riti") or [], ctx.get("stragiudiziale") or "",
                                        layout_log)),
        ("obiezioni", objections_section(plan, fallback_objections)),
        ("processuali", procedure),
        ("conclusioni", "\n\n".join(conclusions)),
    ]
    log.update({"rule_statements_removed": rules_log, "objections_dropped": objections_dropped,
                "coherence": coherence, "rulings_out": rulings_out, "circumstance_claims": claims,
                "rulings_scope_out": scope_log, "articles_out_of_scope": articles_log,
                "fact_wording": fact_wording, "charged_prose": charged_log,
                "orientation_claims_out": orientation_claims, "procedibility_out": procedural_out, **layout_log})
    log["controllo_finale"] = final_check(dict(keyed), ctx, log, plan)
    # «orientamento da verificare» stays where the section prints it (v71); its point is collected all the same
    exempt = title("orientamenti")
    # v75b: §10 holds only the open questions -- the uncertainties the text marked (whole sentences), the dates the
    # ratione temporis needs, the case's questions still unanswered; an orientamento to verify stays in §5
    collected, points = collect_points([(title(k), x) for k, x in keyed if title(k) != exempt],
                                       list(ctx.get("points") or []))
    for q in ctx.get("open_questions") or []:
        if not any(same_point(q, p) for p in points):
            points.append(q)
    texts = dict(collected)
    body = [(title(k), x if title(k) == exempt else texts[title(k)]) for k, x in keyed]
    body.append((title("verificare"), points_section(points)))
    log.update({"points": len(points), "main_thesis": sols[main]["tesi"][:160], "esito": a.get("esito"),
                "obiettivo": a.get("obiettivo")})
    _log("compose", log)
    joined = "\n\n".join(f"**{i}. {t}**\n\n{x}" for i, (t, x) in enumerate(body, 1))
    # v79 (B3): what the nightly's output check reads -- the sources, the scope, and the code-written texts (built from
    # the sources: never a rule statement to trace)
    code_texts = [x for k, x in keyed if k in ("sintesi", "fatti", "conclusioni")] + \
        [f"{o.get('obiezione') or ''} {o.get('risposta') or ''}" for o in fallback_objections] + \
        [f"{t['tesi']} {t.get('pro') or ''} {t.get('contro') or ''}" for t in sols] + [ctx.get("subsumption") or ""] + \
        [ctx.get("temporal") or "", ctx.get("stragiudiziale") or ""] + list(ctx.get("litisconsorzio") or []) + \
        list(ctx.get("riti") or []) + list(points)
    ctx["output_sources"] = {**rule_statements.serializable(rule_src),
                             "scope": sorted(scope | article_scope(ctx, code_texts + [head])),
                             "code": [_plain(x) for x in code_texts if _plain(x)]}
    return f"{head}\n\n{joined}" if head else joined


# --- the context, before writing ------------------------------------------------------------------------------------------

def attach_notes(records: Dict[str, Dict[str, Any]], circ: Dict[str, Dict[str, Any]], rows: Sequence[Dict[str, Any]]
                 ) -> Tuple[Dict[str, str], Dict[str, str]]:
    """v72: each record's and circumstance's note di inquadramento (article_notes.for_label: the store's, else the
    file's) into rec["note_di_inquadramento"]; ({label: note}, {label: source})."""
    from . import article_notes
    notes, sources = {}, {}
    for label, rec in list((records or {}).items()) + list((circ or {}).items()):
        note, source = article_notes.for_label(label, rows)
        if note:
            rec["note_di_inquadramento"] = note
            notes[label], sources[label] = note, source
    return notes, sources


def notes_block(records: Dict[str, Dict[str, Any]]) -> str:
    """The records' notes for a writer's material ("" with none)."""
    lines = [f"- art. {x}: {r['note_di_inquadramento']}" for x, r in (records or {}).items()
             if r.get("note_di_inquadramento")]
    return f"{NOTES_TITLE}:\n" + "\n".join(lines) if lines else ""


def posture_text(posture: Dict[str, Any]) -> str:
    parts = [f"{k} {posture.get(k)}" for k in ("rito", "role", "stage") if posture.get(k) and posture.get(k) != "sconosciuto"]
    return ", ".join(parts)


def charged_text(records: Dict[str, Dict[str, Any]], rows: Sequence[Dict[str, Any]]) -> str:
    """The main charged article's text (its pena), from the rows."""
    label = main_threat(records)
    if not label:
        return ""
    number = label.split()[0]
    for r in rows or []:
        ref = r.get("_article") or ()
        if ref and str(ref[0]) == number and "c.p" in str(ref[1]) and "c.p.p" not in str(ref[1]):
            return str((r.get("s") or {}).get("plain_text") or "")
    return ""


def prepare(state: Dict[str, Any], posture: Dict[str, Any], rows: Sequence[Dict[str, Any]], case: str, client: str,
            main: str = "", order: Sequence[str] = (), rulings: Sequence[str] = (), lines: Sequence[str] = (),
            abrogated: Sequence[str] = (), objections: Sequence[str] = (), link_call=None,
            frame_areas: Sequence[str] = (), questions: Sequence[Dict[str, Any]] = (), case_id_: str = ""
            ) -> Dict[str, Any]:
    """The method's context: the narrative's facts, the circumstances (graded: one call), the assessment, the
    subordinate line, the ratione temporis line, the litisconsorzio lines, the charge's procedibilità and the
    stragiudiziale option it allows. rulings: the retrieved ones, {id, label, massima} -- the only citable."""
    all_records = state.get("records") or {}
    sentences = state.get("sentences") or []
    penal = (posture or {}).get("family") != "civile"
    by_id = {s["id"]: s for s in sentences}
    notes, note_sources = attach_notes(all_records, {}, rows)       # v72: the lawyer's notes, into the records
    view_log: Dict[str, List[str]] = {}
    every = method_records(all_records, view_log)                   # v74: the decomposed fattispecie only
    # v79 (D2): the ranked candidates are graded and enumerated (ctx["records"]), never what the answer prints or
    # assesses (the record below: §2, §4, the assessment, the objections' fallback)
    records = {x: r for x, r in every.items() if not r.get("ranked_candidate")}
    circ = circumstance_records(circumstance_candidates(case, records, posture or {}))
    circ_log: Dict[str, Any] = {}
    if circ and sentences:
        try:
            circ_log = grade_circumstances(circ, sentences, call=link_call)
        except Exception as exc:       # noqa: BLE001 -- ungraded: every element stays «da provare»
            circ_log = {"error": f"{type(exc).__name__}: {exc}"[:200]}
    circ_notes, circ_sources = attach_notes({}, circ, rows)
    notes, note_sources = {**notes, **circ_notes}, {**note_sources, **circ_sources}
    a = assessment(records, posture or {}, main, client)
    riti = riti_lines(posture or {}, charged_text(records, rows))
    procedibilita = procedibility(records, rows)
    article_texts = [str((r.get("s") or {}).get("plain_text") or "") for r in rows or [] if r.get("_article")]
    facts_log: Dict[str, Any] = {}
    facts = facts_split(sentences, records, penal, questions, article_texts, facts_log)
    note = aggravante_note(circ, records)
    when, when_points = temporal(case, posture or {}, abrogated)
    ctx = {"case": case, "case_id": case_id_, "client": client, "rito": (posture or {}).get("rito") or "",
           "posture": posture or {}, "areas": list(frame_areas or ()), "procedibilita": procedibilita,
           "notes": notes,
           "posture_text": posture_text(posture or {}), "records": every, "circumstances": circ,
           "facts": facts, "assessment": a, "riti": riti,
           "subsumption": subsumption_section(records, order, by_id, circ, note),
           "temporal": when, "points": when_points,
           "litisconsorzio": litisconsorzio_lines(case, records, posture or {}),
           "stragiudiziale": stragiudiziale_line(posture or {}, procedibilita.get(a.get("article") or "", "")),
           "rulings": [r for r in rulings if isinstance(r, dict) and r.get("id") and r.get("massima")],
           "lines": list(lines),
           "objections_fallback": objections_fallback(records, a, penal, by_id, objections),
           "open_questions": [point_sentence(str(q.get("text") or "")) for q in questions or []      # v75b: §10
                              if isinstance(q, dict) and q.get("text")],
           "charged_prose": charged_prose(records, by_id),                    # v75b: from its sussunzione rows
           "store_fields": _store_fields(rows)}                               # v79 (B1): a rule statement's sources
    _log("prepare", {"assessment": a, "circumstances": {x: [r.get("status") for r in rec.get("requisites") or []]
                                                        for x, rec in circ.items()}, "circumstances_link": circ_log,
                     "aggravante_note": note, "riti": len(riti), "temporal": ctx["temporal"][:200],
                     "litisconsorzio": len(ctx["litisconsorzio"]), "facts": {k: len(v) for k, v in ctx["facts"].items()},
                     "facts_excluded": facts_log, "procedibilita": procedibilita, "rulings": len(ctx["rulings"]),
                     "notes": note_sources, "records": view_log})
    return ctx


def _store_fields(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    try:
        from . import rule_statements
        return rule_statements.store_fields(rows)
    except Exception as exc:          # noqa: BLE001 -- no store fields: a rule traces to the elements or passages
        logger.warning("lawyer_method: store fields not read: %s", exc)
        return {}


def method_records(records: Dict[str, Dict[str, Any]], log: Optional[Dict[str, List[str]]] = None
                   ) -> Dict[str, Dict[str, Any]]:
    """v74: the records the method reasons on -- every one split into elements as the charge is: a fattispecie (or a
    cause di giustificazione) with at least one element read from the article, only those elements kept (the
    whole-comma fallback is a text, not an element). Never a procedural code's article nor c.p. Libro I's (the gate:
    issue_records.decomposed_by_gate); an article that cannot be decomposed gets no record here, so no row in the
    sussunzione, the fatti mancanti or the passi. log: {"gated", "undecomposed"} (the latter: records with a text but
    no element)."""
    from . import issue_records
    log = log if log is not None else {}
    out: Dict[str, Dict[str, Any]] = {}
    for label, rec in (records or {}).items():
        if rec.get("not_fattispecie") or not issue_records.decomposed_by_gate(
                label, str(rec.get("rubrica") or ""), defence=rec.get("decomposed_for") == issue_records.DEFENCE_WHY):
            log.setdefault("gated", []).append(label)
            continue
        if rec.get("undecomposed") or not issue_records.decomposed(rec):
            if rec.get("requisites") or rec.get("undecomposed"):
                log.setdefault("undecomposed", []).append(label)
            continue
        out[label] = {**rec, "requisites": [r for r in rec["requisites"] if r.get("kind") == "requisito"]}
    for label in [x for x, r in out.items() if r.get("ranked_candidate")]:     # v79 (D2): a candidate that grades
        live = _live(out[label])                                               # to nothing, gone without a word
        if not any(r.get("status") == PROVATO for r in live) or any(r.get("status") == NON_RISULTA for r in live):
            log.setdefault("ranked_dropped", []).append(label)
            del out[label]
    noise = [x for x, r in out.items() if fact_free(r) and not kept_without_facts(r) and not r.get("ranked_candidate")]
    if noise and len(noise) < len(out):     # v78: tabled with «nessun fatto» rows only -- out, unless nothing else is left
        log["no_fact"] = noise
        out = {x: r for x, r in out.items() if x not in noise}
    return out


def fact_free(rec: Dict[str, Any]) -> bool:
    """v78: no case sentence touches any of the record's elements -- none supports, none contradicts, none «provato»
    or «non risulta» (590-bis, 583-quater, 2044 c.c. on a victim's lesioni: «nessun fatto» on every row)."""
    live = _live(rec)
    return bool(live) and all(r.get("status") == DA_PROVARE and not r.get("supporting") and not r.get("contradicting")
                              for r in live)


def kept_without_facts(rec: Dict[str, Any]) -> bool:
    """v78: a record the method keeps with no fact: the charge or the claim (rec["charged"], or decomposed as one), the
    counterparty's claim (side threat), an article the user named."""
    return bool(rec.get("charged") or rec.get("side") == "threat" or rec.get("named_due")
                or str(rec.get("decomposed_for") or "").startswith(("nominata dall'utente", "addebito", "domanda")))


# --- the nightly's yardstick: the answer's structure ----------------------------------------------------------------------

_SECTION_RE = re.compile(r"(?m)^\*\*(\d+)\.\s+([^*\n]+)\*\*\s*$")


def answer_sections(answer: str) -> List[Tuple[str, str]]:
    parts = _SECTION_RE.split(answer or "")
    return [(parts[i + 1].strip(), parts[i + 2].strip()) for i in range(1, len(parts) - 2, 3)]


_BOLD_HEAD_RE = re.compile(r"\*\*[^*]{1,80}\*\*\s*")


def output_check(answer: str, out: Dict[str, Any]) -> Dict[str, List[str]]:
    """v79 (B3): the delivered parere checked as the nightly reads it -- {"unsourced": the rule statements with no
    source, "out_of_scope": the articles cited outside the answer's scope}, from compose's ctx["output_sources"]; §5
    (the code's rulings and passages), the tables, the footer and the code-written sentences are not read."""
    from . import citations, rule_statements
    from .. import version
    src = rule_statements.from_serializable(out or {})
    scope = set((out or {}).get("scope") or [])
    code = [str(x) for x in (out or {}).get("code") or []]
    unsourced, outside = [], []
    skip = title("orientamenti")
    for t, text in answer_sections(version.strip_footer(answer or "")):
        if t == skip:
            continue
        for line in (text or "").split("\n"):
            if line.lstrip().startswith("|"):
                continue
            for piece in _citation_rejoin(_SENT_SPLIT.split(line)):
                sentence = _BOLD_HEAD_RE.sub("", piece).strip().lstrip("-• ").strip()
                if not sentence or any(_plain(sentence) in c for c in code):
                    continue
                if rule_statements.is_rule(sentence) and not rule_statements.source_of(sentence, src):
                    unsourced.append(sentence)
                outside += [citations.label(r) for r in citations.article_refs(sentence)
                            if scope and citations.label(r) not in scope]
    return {"unsourced": unsourced, "out_of_scope": list(dict.fromkeys(outside))}


def structure_checks(answer: str, cell: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """[{check, ok, why}]: the ten sections present and non-empty, in order; §4's tables; §6's table with its realismo,
    the stragiudiziale option; three objections; §9's assessment lines; no «da verificare» outside §10; and by the
    cell: a civil claimant before filing has mediazione, ATP and 700 in §8, an imputato the riti alternativi."""
    cell = cell or {}
    secs = answer_sections(answer)
    by = {t: x for t, x in secs}
    names = titles()
    out = []

    def check(name: str, ok: bool, why: str = "") -> None:
        out.append({"check": name, "ok": bool(ok), "why": "" if ok else why})

    check("sections", [t for t, _ in secs] == names, f"found {[t for t, _ in secs]}")
    for t in names:
        check(f"non_empty:{t}", bool((by.get(t) or "").strip()), "empty or missing")
    summary = by.get(title("sintesi")) or ""
    check("summary_max_lines", len([x for x in _SENT_SPLIT.split(summary) if x.strip()]) <= SUMMARY_LINES,
          f"more than {SUMMARY_LINES} sentences")
    sub = by.get(title("norme")) or ""
    check("subsumption_table", "| Elemento | Fatto che lo integra | Esito |" in sub, "no element table")
    check("no_count_sentence", not re.search(r"requisiti provati:?\s*\d+\s+su\s+\d+", answer or "", re.I),
          "a «requisiti provati N su M» line")
    sol = by.get(title("soluzioni")) or ""
    check("solutions_table", "| Tesi | Pro | Contro | Realismo |" in sol, "no solutions table")
    check("stragiudiziale", "Opzione stragiudiziale:" in sol, "no stragiudiziale option in the strategy")
    obj = by.get(title("obiezioni")) or ""
    check("three_objections", len(re.findall(r"(?m)^\*\*Obiezione \d+\.\*\*", obj)) >= OBJECTIONS,
          "fewer than three objections")
    con = by.get(title("conclusioni")) or ""
    for line in ("**Esito probabile**", "**Grado di certezza**", "**Obiettivo realistico**"):
        check(f"conclusions:{line.strip('*')}", line in con, "missing")
    check("obiettivo_label", bool(re.search(r"\*\*Obiettivo realistico\*\*: (?:esito pieno|esito parziale|contenimento)",
                                            con)), "no esito pieno / parziale / contenimento")
    from . import rulings as rulings_mod
    orient = by.get(title("orientamenti")) or ""
    registry_layout = rulings_mod.CORPUS_LINE in orient         # v78: §5 the registry's (rulings.section_text)
    body = "\n".join(("" if registry_layout else
                      re.sub(rf"(?m)^- [^\n]*: {ORIENTATION_TO_VERIFY}$|^{ORIENTATION_TO_VERIFY}$", "", x))
                     if t == title("orientamenti") else x for t, x in secs if t != title("verificare"))
    check("markers_collected", not re.search(r"\bda verificare\b", body, re.I), "«da verificare» in the body")
    if registry_layout:
        bad = rulings_mod.section_lines_ok(orient)
        check("orientations_code_rendered", not bad, f"lines not rendered by the code: {bad[:2]}")
        check("no_massima", not re.search(r"\bmassim[ae]\b", answer or "", re.I), "the word «massima» in the answer")
    else:
        check("orientations_code_rendered",
              all(re.match(rf"^- [^\n]*: (?:{ORIENTATION_TO_VERIFY}|.+\(documento [^)]+\): «.+»)$", ln)
                  for ln in orient.split("\n") if ln.strip() and ln.strip() != ORIENTATION_TO_VERIFY),
              "an orientation line not rendered by the code")
    proc = by.get(title("processuali")) or ""
    family, role, stage = cell.get("family"), cell.get("role"), cell.get("stage")
    if cell.get("rito", "civile") in ("civile", "famiglia") and family == "civile" and role == "attore" \
            and stage == "pre_filing":
        for name, rx in (("mediazione", r"mediazion"), ("atp", r"696"), ("urgenza_700", r"\b700\b")):
            check(f"pre_filing:{name}", re.search(rx, proc, re.I), f"§8 without {name}")
    if role == "imputato":
        check("riti_alternativi", re.search(r"abbreviato", answer or "", re.I) and re.search(r"444|patteggiament|"
                                                                                            r"applicazione della pena",
                                                                                            answer or "", re.I),
              "no abbreviato / patteggiamento")
    if family in ("penale", "civile"):
        check("ratione_temporis", "Disciplina applicabile" in proc, "no ratione temporis line in §8")
    return out
