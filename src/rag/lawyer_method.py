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
  obiezioni     his Fase 6: the three strongest objections, each answered (v79.14.1, item 6: with every element of the
                main record «da provare», one model-written objection on the strongest element, grounded, or none --
                strongest_open, all_open_section; the per-element template never repeated)
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


# --- v79.12.1: a defence record is never written with the charge template ----------------------------------------------

# the general-part records that are a defence line, never a charge: issue_records.DEFENCE_ARTICLES (the scriminanti
# 50-54, excess, attempt, aberratio, tenuità) and the putative scriminante (59) and art. 384 c.p. -- review: never
# causation (40, 41) nor fault (43): their elements integrated establish the charge, so they keep the charge's
# template; only 41 co. 2 (a later cause sufficient by itself, which breaks the causal link) is a defence record
DEFENCE_RECORDS = ("59", "384")
NOT_DEFENCE_RECORDS = ("40", "41", "43")
DEFENCE_COMMAS = {"41": "2"}
EXCULPATORY = ("41", "50", "51", "52", "53", "54", "59", "384", "131-bis")   # all integrato: the full outcome
DEFENCE_WORDS = {   # my drafts (v79.12.1): a defence record's own wording -- never «per l'accusa», never a conviction
    "kind": {"41": "interruzione del nesso causale", "50": "scriminante", "51": "scriminante", "52": "scriminante",
             "53": "scriminante", "54": "scriminante",
             "59": "scriminante putativa", "55": "eccesso colposo", "384": "causa di non punibilità",
             "131-bis": "particolare tenuità del fatto"},
    "kind_default": "difesa di parte generale",
    "effect": {"scriminante": "il fatto è scriminato e non è punibile",
               "scriminante putativa": "la scriminante, anche solo erroneamente supposta, esclude la punibilità",
               "eccesso colposo": "il fatto si punisce solo a titolo di colpa, se la legge lo prevede come delitto "
                                  "colposo",
               "causa di non punibilità": "il fatto non è punibile",
               "interruzione del nesso causale": "il nesso causale con l'evento è interrotto",
               "particolare tenuità del fatto": "la punibilità è esclusa per la particolare tenuità del fatto"},
    "effect_default": "la difesa sull'art. {art} regge",
    "all": "tutti gli elementi dell'art. {art} ({kind}) risultano integrati: {effect}",
    "excluded": "un elemento dell'art. {art} ({kind}) è escluso dai fatti: la difesa su questa norma non regge",
    "partly": "{proved} elementi su {total} dell'art. {art} ({kind}) risultano provati; gli altri restano da provare "
              "dalla difesa",
    "open": "gli elementi dell'art. {art} ({kind}) restano da provare dalla difesa",
    "depends": "la prova, da parte della difesa, degli elementi {ids}",
    "risk_all": "che il giudice non riconosca i presupposti dell'art. {art}{charge}",
    "risk_open": "che la difesa non provi gli elementi {ids} dell'art. {art}{charge}",
    "risk_excluded": "che l'addebito{charge_paren} resti senza la difesa dell'art. {art}",
    "risk_charge": " (resterebbe l'addebito dell'art. {charge})",
    "changes": "{head}: da provare dalla difesa; se provato la difesa sull'art. {art} regge, se escluso cade",
    "obj_proven": ("L'accusa contesterà che ricorra l'elemento «{el}» dell'art. {art}",
                   "I fatti narrati lo sostengono: {fact}."),
    "obj_open": ("L'accusa negherà l'elemento «{el}» dell'art. {art}", "Va provato dalla difesa: {missing}."),
    "prose_open": "restano da provare dalla difesa {items}",
}
_COMMA_TAIL_RE = re.compile(r"\s+co(?:mma|\.)\s*\d+.*$", re.IGNORECASE)


def defence_base(label: str) -> str:
    """v79.12.1: the c.p. number of a defence record («52» for «52 c.p.» and «52 c.p. co. 2»), "" for any other --
    review: "" for causation and fault (NOT_DEFENCE_RECORDS) but the defence comma of one («41 c.p. co. 2»)."""
    from . import issue_records
    from .plan_then_write import _code_key
    label = str(label or "").strip()
    number, _, code = _COMMA_TAIL_RE.sub("", label).partition(" ")
    if _code_key(code.strip()) != "c.p.":
        return ""
    if number in NOT_DEFENCE_RECORDS:
        comma = re.search(r"\bco(?:mma|\.)\s*(\d+)", label, re.IGNORECASE)
        return number if comma and DEFENCE_COMMAS.get(number) == comma.group(1) else ""
    return number if number in issue_records.DEFENCE_ARTICLES or number in DEFENCE_RECORDS else ""


def defence_record(label: str, rec: Optional[Dict[str, Any]] = None) -> bool:
    """v79.12.1: a defence record (52-55, 59, 384, 131-bis …, a later comma of one too): its outcome, its open elements
    and its risk are written with DEFENCE_WORDS, never with the charge's template."""
    return bool(defence_base(label) or defence_base(str((rec or {}).get("comma_of") or "")))


def defence_kind(label: str) -> str:
    return DEFENCE_WORDS["kind"].get(defence_base(label), DEFENCE_WORDS["kind_default"])


def defence_assessment(records: Dict[str, Dict[str, Any]], label: str, client: str = "", charge: str = ""
                       ) -> Dict[str, Any]:
    """v79.12.1: the outcome read on a defence record -- every element integrato: favourable (the full outcome for an
    exculpatory one, EXCULPATORY; partial for the excess, the attempt, the aberratio); one excluded: unfavourable (the
    defence does not hold); some proven: uncertain, the rest «da provare dalla difesa»; none: uncertain. a["defence"]
    marks it (the risk and what it depends on are DEFENCE_WORDS' too); a["charge"] the charge it answers, if any."""
    rec = (records or {}).get(label) or {}
    number = defence_base(label) or defence_base(str(rec.get("comma_of") or ""))
    kind = DEFENCE_WORDS["kind"].get(number, DEFENCE_WORDS["kind_default"])
    proved, excluded, total = _counts(rec)
    full = PIENO if number in EXCULPATORY else PARZIALE
    charge = charge if charge and charge != label and not defence_record(charge) else ""
    base = {"article": label, "defence": True, "charge": charge}
    if excluded:
        return {**base, "esito": SFAVOREVOLE, "grado": MEDIO, "obiettivo": PARZIALE,
                "why": DEFENCE_WORDS["excluded"].format(art=label, kind=kind)}
    if total and proved == total:
        effect = DEFENCE_WORDS["effect"].get(kind) or DEFENCE_WORDS["effect_default"].format(art=label)
        return {**base, "esito": FAVOREVOLE, "grado": ALTO if total > 1 else MEDIO, "obiettivo": full,
                "defence_integrated": True,
                "why": DEFENCE_WORDS["all"].format(art=label, kind=kind, effect=effect)}
    if proved:
        return {**base, "esito": INCERTO, "grado": MEDIO, "obiettivo": full,
                "why": DEFENCE_WORDS["partly"].format(proved=proved, total=total, art=label, kind=kind)}
    return {**base, "esito": INCERTO, "grado": BASSO, "obiettivo": full,
            "why": DEFENCE_WORDS["open"].format(art=label, kind=kind)}


def defence_risk(a: Dict[str, Any], open_: Sequence[str]) -> Tuple[str, str]:
    """v79.12.1: (depends, risk) of a defence assessment -- never a conviction under the defence article."""
    art, charge = str(a.get("article") or ""), str(a.get("charge") or "")
    tail = DEFENCE_WORDS["risk_charge"].format(charge=charge) if charge else ""
    depends = DEFENCE_WORDS["depends"].format(ids=", ".join(open_)) if open_ else ""
    if a.get("esito") == SFAVOREVOLE:
        return depends, DEFENCE_WORDS["risk_excluded"].format(art=art, charge_paren=f" (art. {charge})" if charge
                                                              else "")
    if open_:
        return depends, DEFENCE_WORDS["risk_open"].format(ids=", ".join(open_), art=art, charge=tail)
    return depends, DEFENCE_WORDS["risk_all"].format(art=art, charge=tail)


def changes(label: str, rec: Dict[str, Any], req: Dict[str, Any], penal: bool) -> str:
    """What a missing fact would change, by the element's id: for the charge or the claim against the client, whether
    that element of the article stands; else whether the client's norm applies."""
    head = f"l'elemento {req.get('id')} dell'art. {label} («{_short(req.get('text'), 90)}»)"
    if defence_record(label, rec):          # v79.12.1: a defence record's own words, never the charge's
        return DEFENCE_WORDS["changes"].format(head=head, art=label)
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
    from . import issue_records
    certi, ipotesi, excluded = [], [], []
    for s_ in sentences or []:
        if not narrative(s_, article_texts):
            if _plain(s_.get("text")) and s_.get("source") != "sistema":
                excluded.append(_short(s_.get("text"), 80))
            continue
        if issue_records.stated_certain(s_):        # v79.8 (C1): as §4's «provato»; v79.12.7 (item 7): an act too
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
# v79.14.3 (item 1): an element only another party's allegation supports (issue_records.contested) -- its «Esito» and
# its «Fatto che lo integra» cell (my drafts)
CONTESTED_ESITO = "contestato (allegazione {di_parte})"
CONTESTED_FACT = "nessun fatto certo; solo l'allegazione: {fact}"


def esito_label(r: Dict[str, Any]) -> str:
    """The element's «Esito»: ESITO's, or v79.14.3 (item 1) CONTESTED_ESITO for a contested one."""
    from . import issue_records
    if issue_records.contested(r):
        return CONTESTED_ESITO.format(di_parte=issue_records.of_party(r.get("contested_by")))
    status = r.get("status") or DA_PROVARE
    return ESITO.get(status, status)


def _sentence_text(ids: Sequence[str], by_id: Dict[str, Dict[str, Any]]) -> str:
    texts = [_short(by_id[i].get("text"), 160) for i in ids or [] if i in by_id]
    return " / ".join(texts[:2])


def subsumption_rows(rec: Dict[str, Any], by_id: Dict[str, Dict[str, Any]]) -> List[Tuple[str, str, str]]:
    """[(elemento, fatto che lo integra, esito)] of the record's live elements."""
    out = []
    from . import issue_records
    for r in _live(rec):
        status = r.get("status") or DA_PROVARE
        if status == PROVATO:
            # v79.14.3 (item 1): the certain facts first -- an allegation never shown as what integrates it
            sup = [i for i in r.get("supporting") or [] if i in by_id]
            sure = [i for i in sup if issue_records.certain_fact(by_id[i], r.get("text") or "")]
            fact = _sentence_text(sure or sup, by_id) or "i fatti estratti dal caso"
        elif issue_records.contested(r):        # v79.14.3 (item 1): the allegation, said as one
            fact = CONTESTED_FACT.format(fact=_sentence_text(r.get("contested_ids") or [], by_id) or
                                         ", ".join(r.get("contested_ids") or []))
        elif status == NON_RISULTA:
            fact = _sentence_text(r.get("contradicting") or [], by_id) or "i fatti estratti lo escludono"
        elif r.get("fit_open") and _fit_on():   # v79.13.0: the applicability judgment's «dipende» -- the question
            from .applicability import FIT_OPEN_ROW                                                    # decides it
            fact = FIT_OPEN_ROW.format(question=_short(r["fit_open"], 160))
        else:
            fact = "nessun fatto del caso" + (f": manca {_short(r.get('missing'), 120)}" if r.get("missing") else "")
        text = _short(r.get("text"), 160)
        if r.get("branch") and status == PROVATO:       # v79.6 (item 4): the branch that applies
            text = f"{text} (alternativa che ricorre: «{_short(r['branch'], 80)}»)"
        if r.get("jurisprudential") and r.get("source"):    # v79.12.7 (item 1): not in the text -- its source named
            text = f"{text} ({JURISPRUDENTIAL_ROW.format(source=_short(r['source'], 120))})"
        out.append((text, fact, esito_label(r)))
    return out


JURISPRUDENTIAL_ROW = "elemento non scritto nella norma, richiesto da {source}"     # my draft (v79.12.7, item 1)
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
                parts.append(DEFENCE_WORDS["prose_open"].format(items=", ".join(items))
                             if status == DA_PROVARE and defence_record(label, records[label])      # v79.12.1
                             else CHARGED_PROSE[status].format(items=", ".join(items)))
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


# --- v79.14.2 (B): the verified frame binds the charge / claim set ----------------------------------------------------
# With CASE_FRAME on and the frame verified on the rito (or a penal stage) and the role (case_frame.held / bound: the
# verified fields only), the records the method may offer as the charge or the claim come from the frame's side of the
# law: a penal rito (penale, penale_gdp; or, with no rito verified, the stage «indagini») with the client imputato /
# indagato -- penal sources only (penal_source: the c.p., the c.p.p., a penal provision as case_frame.is_penal reads it --
# 2621-2642 c.c., a penal law's penal articles --, a law data/institution_laws.json flags «penal» that case_frame's
# penal_provisions does not list); a civil rito (CIVIL_BINDING_RITI) with a verified role -- never a c.p. article. Every
# other record stays in the records as context (§2's schede, §4, «Altri profili») marked FRAME_EXCLUDED: never the
# charge (_charged, main_threat, the assessment), never a thesis candidate (thesis_candidates), so never «Linea
# principale» nor «Esito probabile»; the thesis call's SCHEDE never label it «addebito o domanda». The frame not
# verified on the rito or the role: no binding (the 79.14.1 path). The charge set emptied by the binding: no fallback to
# the excluded -- the charge takes the corpus line's path (BINDING_EMPTY_LINE in §1, the assessment's why).
# Counters frame_binding_applied, frame_binding_excluded (the articles), frame_binding_empty; log «frame_binding».
FRAME_EXCLUDED = "frame_excluded"
FRAME_BINDING_COUNTERS = ("frame_binding_applied", "frame_binding_excluded", "frame_binding_empty")
CIVIL_BINDING_RITI = ("civile", "lavoro", "famiglia")
PENAL_SOURCE_CODES = ("cp", "cpp")
# my drafts (v79.14.2): the corpus line's subject when the frame names no charge or claim
BINDING_MATERIA = ("questo addebito", "questa domanda")     # a penal binding's, a civil binding's


def binding_mode(holder: Optional[Dict[str, Any]]) -> str:
    """v79.14.2 (B): «penale», «civile» or "" (no binding) from the held frame's verified fields only."""
    from . import case_frame
    if not isinstance(holder, dict) or case_frame.held(holder) is None or case_frame.role_unknown_held(holder):
        return ""
    role = case_frame.bound(holder, "ruolo")
    if not role:
        return ""
    rito, stage = case_frame.bound(holder, "rito"), case_frame.bound(holder, "stadio")
    if role == "imputato" and (rito in case_frame.PENAL_RITI or (rito is None and stage == "indagini")):
        return "penale"
    if rito in CIVIL_BINDING_RITI and role != "imputato":
        return "civile"
    return ""


def _law_key(text: Any) -> str:
    return re.sub(r"[\s.]", "", str(text or "")).lower()


def penal_flagged_laws() -> List[str]:
    """v79.14.2 (B): data/institution_laws.json's laws flagged «penal» that are not codes (their key, compacted)."""
    return [_law_key(e["law"]) for e in institution_laws() if e.get("penal") and not code_law(str(e["law"]))]


def penal_source(label: str) -> bool:
    """v79.14.2 (B): an article a penal binding may offer as the charge -- the c.p.'s, the c.p.p.'s, a penal provision
    (case_frame.is_penal: 2621-2642 c.c., a penal law's penal articles), or an article of a law
    data/institution_laws.json flags «penal» when case_frame's penal_provisions does not list that law (it decides
    otherwise, by its ranges)."""
    from . import case_frame
    a = case_frame.parse_article(label)
    if not a.get("number"):
        return False
    lead = case_frame.lead_code(a.get("rest"))
    if lead in PENAL_SOURCE_CODES:
        return True
    if case_frame.is_penal(a):
        return True
    if lead or any(e["_rx"].search(a.get("rest") or "") for e in case_frame._penal_laws()):
        return False
    key = _law_key(a.get("rest"))
    return any(law and law in key for law in penal_flagged_laws())


def civil_excluded(label: str) -> bool:
    """v79.14.2 (B): a civil binding's excluded article -- the c.p.'s."""
    from . import case_frame
    return case_frame.lead_code(case_frame.parse_article(label).get("rest")) == "cp"


def binding_allows(mode: str, label: str) -> bool:
    if mode == "penale":
        return penal_source(label)
    if mode == "civile":
        return not civil_excluded(label)
    return True


def frame_excluded(rec: Any) -> bool:
    return isinstance(rec, dict) and bool(rec.get(FRAME_EXCLUDED))


def frame_binding(state: Optional[Dict[str, Any]], records: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """v79.14.2 (B): the binding on the method's records (prepare's, before the assessment), in place: every earlier
    mark cleared, then each record the mode does not allow marked FRAME_EXCLUDED. {mode, applied, charge_set (the
    records the charge or the claim against the client was: side «threat» or charged), kept, excluded (the charge set's
    excluded -- thesis_candidates and prepare add the candidates and the held charge they refuse), empty (a charge set
    the binding emptied)}."""
    for rec in (records or {}).values():
        if isinstance(rec, dict):
            rec.pop(FRAME_EXCLUDED, None)
    mode = binding_mode(state)
    out: Dict[str, Any] = {"mode": mode, "applied": bool(mode), "charge_set": [], "kept": [], "excluded": [],
                           "empty": False, "rescued": False}
    if not mode:
        return out
    for label, rec in (records or {}).items():
        if isinstance(rec, dict) and not binding_allows(mode, label):
            rec[FRAME_EXCLUDED] = mode
    charge_set = [x for x, r in (records or {}).items() if isinstance(r, dict)
                  and (r.get("side") == "threat" or r.get("charged"))]
    out["charge_set"] = charge_set
    out["kept"] = [x for x in charge_set if not frame_excluded(records[x])]
    out["excluded"] = [x for x in charge_set if frame_excluded(records[x])]
    out["empty"] = bool(charge_set) and not out["kept"]
    out["frame_charge"] = frame_charge(state, mode)
    out["charge_in_corpus"] = charge_in_corpus(out["frame_charge"])
    if out["empty"]:
        rescue(out, records, (state or {}).get("records") if isinstance(state, dict) else None)
    return out


# --- v79.14.3 (item 3): an empty plan with a verified charge ---------------------------------------------------------
# The binding emptied the charge set while the verified frame names its charge (addebito_o_domanda's article) and that
# article is a code's (code_law) or a law data/institution_laws.json lists: the theses are built on that article's
# elements -- its record (prepare's, by its label) marked the threat (rec["frame_rescued"]), so the charge, the
# assessment and the contests (its «da provare» and «contestato» elements, item 1) are its own -- never a fall through to
# «Ammettere la condotta» (never without ADMISSION_VERIFIED) nor the corpus line. frame_binding_rescued counts it; logged
# «frame_binding_rescue». The record missing (or with no live element): no rescue (logged with its reason) -- and still
# no corpus line for a code article or a law in the corpus (charge_in_corpus).
def frame_charge(state: Optional[Dict[str, Any]], mode: str = "") -> str:
    """v79.14.3 (item 3): the verified frame's charge or claim article (case_frame.bound «addebito_o_domanda»), "" when
    none, or when the binding's own side of the law refuses it."""
    from . import case_frame
    label = case_frame.bound(state, "addebito_o_domanda") if isinstance(state, dict) else None
    label = str(label or "")
    return label if label and binding_allows(mode, label) else ""


def charge_view(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """v79.19.0 (item 2): the charge the method reads -- the settled frame's (case_frame.bound «addebito_o_domanda»:
    one source of truth, its rule «case_frame», its alternatives the frame's other charges), else the charge rule's
    (state["charge"]: no verified frame charge)."""
    from . import case_frame, posture as posture_mod
    held = dict((state or {}).get("charge") or {}) if isinstance(state, dict) else {}
    label = case_frame.bound(state, "addebito_o_domanda") if isinstance(state, dict) else None
    if not label:
        return held
    if held.get("article") == label:
        return {**held, "alternatives": case_frame.alternative_charges(state)}
    return {**held, "article": label, "rule": posture_mod.FRAME_CHARGE_RULE,
            "alternatives": case_frame.alternative_charges(state)}


def charge_in_corpus(label: str) -> bool:
    """v79.14.3 (item 3): the charge's article is a code's (code_law: c.p., c.c., c.p.p. …) or of a law
    data/institution_laws.json lists (the laws the corpus is read for) -- the corpus line never names it."""
    from . import case_frame
    if not label:
        return False
    rest = str(case_frame.parse_article(label).get("rest") or "")
    if code_law(rest) or case_frame.lead_code(rest):
        return True
    key = _law_key(rest)
    return bool(key) and any(_law_key(e.get("law")) and _law_key(e.get("law")) in key for e in institution_laws())


def _same_label(a: str, b: str) -> bool:
    return _law_key(a) == _law_key(b)


def rescue(out: Dict[str, Any], records: Dict[str, Dict[str, Any]],
           read: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
    """v79.14.3 (item 3), in place: the emptied charge set rescued on the frame's verified charge (see above). read:
    the flow's records -- the charge's record the method's view left out as noise (method_records: «nessun fatto» rows
    beside other records) is taken back into records through method_records alone, as the charge always is."""
    label = out.get("frame_charge") or ""
    why = ""
    if not label:
        why = "il quadro non indica un addebito verificato"
    elif not out.get("charge_in_corpus"):
        why = "l'addebito non è un articolo dei codici né di una legge del corpus"
    found = next((x for x in (records or {}) if _same_label(x, label)), "") if not why else ""
    if not why and not found and isinstance(read, dict):
        flow_label = next((x for x in read if _same_label(x, label)), "")
        if flow_label and isinstance(read.get(flow_label), dict):
            taken = method_records({flow_label: read[flow_label]})
            if isinstance(records, dict) and taken.get(flow_label):
                records[flow_label] = taken[flow_label]
                found = flow_label
    rec = (records or {}).get(found) if found else None
    if not why and not isinstance(rec, dict):
        why = "nessuna scheda dell'articolo tra le norme lette"
    elif not why and (rec.get("ranked_candidate") or frame_excluded(rec) or not any(r.get("id") for r in _live(rec))):
        why = "la scheda dell'articolo non ha elementi da contestare"
    if why:
        out["rescued"] = False
        _log("frame_binding_rescue", {"charge": label, "rescued": False, "why": why})
        return
    rec.pop("unprinted", None)
    rec["side"], rec["frame_rescued"] = "threat", True
    out.update(rescued=True, rescued_charge=found, kept=[found], empty=False)
    _log("frame_binding_rescue", {"charge": found, "rescued": True,
                                  "elements": [r.get("id") for r in _live(rec) if r.get("id")]})


def candidate_subjects(records: Dict[str, Dict[str, Any]]) -> Callable[[Dict[str, Any]], List[str]]:
    """v79.14.2 (B): a candidate's articles -- its own, a remedy's list, else the records of the elements it contests
    or rests on (schema_shadow's reading)."""
    owner = {str(r.get("id")): x for x, rec in (records or {}).items() for r in (rec or {}).get("requisites") or []
             if isinstance(r, dict) and r.get("id")}

    def subject(c: Dict[str, Any]) -> List[str]:
        arts = [c["article"]] if c.get("article") else list(c.get("articles") or [])
        return arts or list(dict.fromkeys(owner[str(i)] for i in list(c.get("contesta") or [])
                                          + list(c.get("poggia_su") or []) if str(i) in owner))
    return subject


def binding_excluded(ctx: Dict[str, Any], labels: Sequence[str]) -> None:
    """v79.14.2 (B): more articles the binding kept from the charge (a candidate, the held charge), into the list."""
    b = ctx.get("frame_binding")
    if isinstance(b, dict):
        b["excluded"] = list(dict.fromkeys(list(b.get("excluded") or []) + [str(x) for x in labels if x]))


def binding_materia(state: Optional[Dict[str, Any]], mode: str) -> str:
    """v79.14.2 (B): what the corpus line names for an emptied charge -- the verified frame's charge or claim
    description, else its institution, else BINDING_MATERIA's words."""
    from . import case_frame
    got = case_frame.held(state) if isinstance(state, dict) else None
    view = case_frame.intake_view(got) if got is not None else {}
    charge = view.get("charge") or {}
    return _plain(charge.get("descrizione")) or _plain(view.get("institution")) or \
        BINDING_MATERIA[1 if mode == "civile" else 0]


def bound_assessment(a: Dict[str, Any], b: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """v79.14.2 (B): the assessment of a charge set the binding emptied -- no article, its why the corpus line's
    («il corpus non contiene ancora informazioni sufficienti su …»); any other assessment as it is. v79.14.3 (item 3):
    never the corpus line's why for a charge the corpus holds (charge_in_corpus: a code article, a listed law)."""
    if not (b or {}).get("empty") or a.get("article") or (b or {}).get("charge_in_corpus"):
        return a
    materia = b.get("materia") or BINDING_MATERIA[1 if b.get("mode") == "civile" else 0]
    return {**a, "frame_binding_empty": True,
            "why": CORPUS_LINE.split(":")[0].format(materia=materia).replace("Il corpus", "il corpus", 1)}


def binding_empty_line(ctx: Dict[str, Any]) -> str:
    """v79.14.2 (B): §1's corpus line for a charge set the binding emptied (CORPUS_LINE, with its stop); "" else."""
    b = ctx.get("frame_binding") or {}
    if not b.get("empty") or b.get("charge_in_corpus"):     # v79.14.3 (item 3): never for a charge the corpus holds
        return ""
    line = CORPUS_LINE.format(materia=b.get("materia") or BINDING_MATERIA[1 if b.get("mode") == "civile" else 0]) + "."
    return "" if corpus_line_on_code(line, ctx) else line   # (item 3) nor on a materia whose law is a code


def method_counters(counters: Dict[str, Any], ctx: Dict[str, Any]) -> None:
    """v79.14.3: the method's counters into block_counters (several clients: the largest count, the lists unioned) --
    (item 1) elements_contestato, theses_from_contestato; (item 3) frame_binding_rescued; (item 4) excluded_dropped (the
    excluded and fit-removed articles kept out of §4, §5, §7 and «Fatti mancanti»; compose adds what it dropped)."""
    for k, v in contested_counts(ctx).items():
        counters[k] = max(int(counters.get(k) or 0), int(v))
    counters["frame_binding_rescued"] = max(int(counters.get("frame_binding_rescued") or 0),
                                            int(bool((ctx.get("frame_binding") or {}).get("rescued"))))
    counters["excluded_dropped"] = list(dict.fromkeys(list(counters.get("excluded_dropped") or [])
                                                      + list(ctx.get("excluded_dropped") or [])))


def binding_counters(counters: Dict[str, Any], b: Optional[Dict[str, Any]]) -> None:
    """v79.14.2 (B): the binding's counters into block_counters (several clients: any applied or emptied, the excluded
    unioned)."""
    if not isinstance(b, dict):
        return
    counters["frame_binding_applied"] = max(int(counters.get("frame_binding_applied") or 0), int(bool(b.get("applied"))))
    counters["frame_binding_empty"] = max(int(counters.get("frame_binding_empty") or 0), int(bool(b.get("empty"))))
    counters["frame_binding_excluded"] = list(dict.fromkeys(list(counters.get("frame_binding_excluded") or [])
                                                           + list(b.get("excluded") or [])))


def excluded_labels(ctx: Dict[str, Any]) -> List[str]:
    """v79.14.3 (item 4): the articles excluded means excluded applies to -- the records the frame's binding excluded,
    the charge set and candidates it refused, the norms a verified «no» removed."""
    b = ctx.get("frame_binding") or {}
    return list(dict.fromkeys(list(ctx.get("excluded_articles") or []) + list(b.get("excluded") or [])
                              + list(ctx.get("fit_removed") or [])))


def cites_excluded(text: str, labels: Sequence[str]) -> List[str]:
    """v79.14.3 (item 4): the excluded articles a text cites (by number and code, however spelt), in labels' order."""
    if not labels or not str(text or "").strip():
        return []
    from . import posture as posture_mod
    cited = posture_mod.sentence_articles(str(text))
    return [x for x in dict.fromkeys(labels) if any(_same_label(x, c) for c in cited)]


def excluded_blocks_out(norms: str, labels: Sequence[str], log: Optional[List[str]] = None) -> str:
    """v79.14.3 (item 4): §4's per-article blocks (a «**Art. N …**» heading to the next bold line, as charged_blocks
    reads them) of an excluded article dropped whole; every other line as it is. (review) Any bold heading whose cited
    articles are all excluded («**Il risarcimento (art. 2043 c.c.)**») heads such a block too."""
    from . import citations, posture as posture_mod
    if not labels or not norms:
        return norms
    out: List[str] = []
    skipping = False
    for line in norms.split("\n"):
        if line.strip().startswith("**"):
            skipping = False
            refs = citations.article_refs(line) if _ART_HEAD_RE.match(line) else []
            label = f"{refs[0][0]} {refs[0][1]}" if refs else ""
            hit = next((x for x in labels if label and _same_label(x, label)), "")
            cited = posture_mod.sentence_articles(line) if not hit else []
            if cited and all(any(_same_label(x, c) for x in labels) for c in cited):
                hit = next(x for x in labels if any(_same_label(x, c) for c in cited))
            if hit:
                skipping = True
                if log is not None:
                    log.append(hit)
                continue
        if not skipping:
            out.append(line)
    # a paragraph outside any heading whose cited articles are all excluded is that article's scheda too
    kept: List[str] = []
    for para in "\n".join(out).split("\n\n"):
        cited = posture_mod.sentence_articles(para) if not para.strip().startswith("**") else []
        gone = [c for c in cited if any(_same_label(x, c) for x in labels)]
        if cited and len(gone) == len(cited):
            if log is not None:
                log += [x for x in labels if any(_same_label(x, c) for c in gone)]
            continue
        kept.append(para)
    return "\n\n".join(kept)


def _charged(records: Dict[str, Dict[str, Any]]) -> List[str]:
    """The charged records, in their order -- v79.12.1: never a defence record (a scriminante is no charge); v79.14.2
    (B): never one the frame's binding excludes."""
    out = sorted(((rec.get("charged"), label) for label, rec in (records or {}).items() if rec.get("charged")
                  and not defence_record(label, rec) and not frame_excluded(rec)), key=lambda x: x[0])
    return [label for _, label in out]


def circumstance_candidates(case: str, records: Dict[str, Dict[str, Any]], posture: Dict[str, Any],
                            triggered: Sequence[str] = ()) -> List[Dict[str, Any]]:
    """The table's circumstances to grade (penal postures only): every aggravante for the charged offence -- named by the
    case, matched by its patterns or tied to the charged article --, and the attenuanti the case's words point to or the
    imputato's posture always considers (62-bis). v79.12.7 (item 2): triggered, the attenuanti the facts trigger
    (penal_lines.circumstance_triggers: a diverbio for 62 n. 2, a concrete step for 62 n. 6) -- candidates too."""
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
        elif c.get("kind") == "attenuante" and (hit or c["label"] in set(triggered or ()) or (
                c["label"] == "62-bis c.p." and posture.get("role") == "imputato")):
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


_EXCL_EDGE_RE = re.compile(r"[,;:]\s|\s(?:mentre|ma|invece|però|tuttavia)\s", re.I)
_SUBJECTLESS_RE = re.compile(r"^\s*(?:non|è|sono|va|vanno|risulta|risultano|resta|restano|deve|devono|appare|appaiono|"
                             r"può|possono|potrà|potranno|ricorre|ricorrono|spetta|spettano)\b", re.I)


def _excl_clauses(sentence: str) -> List[Tuple[str, str]]:
    """(review) The sentence's clauses with the separator after each -- split at «, », «; », «: » and «mentre», «ma» …
    outside parentheses (an article's «art. 62, n. 2, c.p.» is one clause)."""
    out: List[Tuple[str, str]] = []
    depth, at, i = 0, 0, 0
    while i < len(sentence):
        ch = sentence[i]
        if ch in "(«":
            depth += 1
        elif ch in ")»":
            depth = max(0, depth - 1)
        elif depth == 0:
            m = _EXCL_EDGE_RE.match(sentence, i)
            if m:
                out.append((sentence[at:i], m.group(0)))
                at = i = m.end()
                continue
        i += 1
    out.append((sentence[at:], ""))
    return out


def circumstance_exclusions(text: str, circ: Dict[str, Dict[str, Any]], log: Optional[List[Dict[str, Any]]] = None,
                            rewrite: bool = True) -> str:
    """v79.12.7 (item 2): never «esclusa» for a circumstance the facts do not exclude -- a written clause naming a
    circumstance of the table (its article and number, or its name: _claim_re) and saying it is excluded
    (penal_lines.says_excluded: «esclusa», «non ricorre», «non spetta», «vanno negate» …; never «non può essere
    esclusa») while the record does not exclude it (penal_lines.excluded: no element «non risulta» by a sentence of the
    case) is replaced by CIRC_UNEXCLUDED, its open elements named. rewrite False: the text unchanged, only the log (the
    final count). (review) Per clause: an exclusion clause names the circumstance itself, or -- subjectless («…, è
    esclusa») -- the nearest clause before it that names one; the sentence's other clauses stay («La legittima difesa è
    esclusa, mentre l'attenuante … resta da valutare» is untouched), each circumstance excluded is rewritten (two in one
    sentence: two lines). An attenuante always; an aggravante only when the record holds it (penal_lines.
    aggravanti_held): the defence arguing an aggravante the record never held does not apply is no unfounded exclusion,
    and rewriting it would raise that aggravante against the client."""
    from . import penal_lines
    table = {c["label"]: c for c in data().get("circumstances") or []}
    held_aggr = set(penal_lines.aggravanti_held(circ or {}))
    out = []
    for para in (text or "").split("\n"):
        sentences = []
        for sentence in _SENT_SPLIT.split(para):
            if not penal_lines.says_excluded(sentence):
                sentences.append(sentence)
                continue
            clauses = _excl_clauses(sentence)
            names = [[label for label, c in table.items() if _claim_re(c).search(cl)] for cl, _ in clauses]
            drop: set = set()
            hits: List[str] = []
            for i, (cl, _) in enumerate(clauses):
                if not penal_lines.says_excluded(cl):
                    continue
                own, subject = names[i], i
                if not own and _SUBJECTLESS_RE.match(cl):
                    subject = next((j for j in range(i - 1, -1, -1) if names[j]), -1)
                    own = names[subject] if subject >= 0 else []
                for label in own:
                    c = table[label]
                    rec = (circ or {}).get(label)
                    if penal_lines.excluded(rec):
                        continue
                    if c.get("kind") == "aggravante" and label not in held_aggr:
                        continue
                    if label not in hits:
                        hits.append(label)
                    drop |= set(range(min(i, subject), i + 1))    # the parenthetical between them too
            if not hits:
                sentences.append(sentence)
                continue
            new = []
            for label in hits:
                c, rec = table[label], (circ or {}).get(label)
                open_ = [_short(r.get("text"), 90) for r in _live(rec) if r.get("status") != PROVATO] if rec \
                    else [_short(e, 90) for e in c.get("elements") or []]
                if log is not None:
                    log.append({"circumstance": label, "sentence": _short(sentence, 200)})
                name, art = penal_lines.circumstance_name(label, rec, c)
                new.append(penal_lines.CIRC_UNEXCLUDED.format(
                    name=name, article=art, open=penal_lines.CIRC_UNEXCLUDED_OPEN.format(
                        open="; ".join(open_)) if open_ else ""))
            if not rewrite:
                sentences.append(sentence)
                continue
            rest = "".join(cl + sep for i, (cl, sep) in enumerate(clauses) if i not in drop)
            rest = re.sub(r"^\s*(?:mentre|ma|invece|però|tuttavia)\b\s*", "", rest.strip(" ,;:"), flags=re.I).strip(" ,;:")
            if len(rest.split()) >= 3:
                rest = rest[0].upper() + rest[1:]
                new.insert(0, rest if re.search(r"[.!?]$", rest) else rest + ".")
            sentences.append(" ".join(new))
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
    if charged:     # v79.12.4 (item 3): the charged article with the highest maximum pena (posture.mark_sides' mark)
        return next((x for x in charged if records[x].get("exposure")), charged[0])
    return next((x for x, r in records.items() if r.get("side") == "threat" and _live(r) and not frame_excluded(r)
                 and issue_records.counted(x, str(r.get("rubrica") or ""))), "")


HYPOTHESIS_LABEL = "ipotesi di reato (addebito non indicato nel quesito)"     # v79 (C4), the PO's words
NO_CHARGE_WHY = {"penale": "l'addebito non è indicato nel quesito e i fatti narrati non ne individuano uno",
                 "civile": "la domanda avversaria non è indicata nel quesito e i fatti narrati non ne individuano una"}


def assessment(records: Dict[str, Dict[str, Any]], posture: Dict[str, Any], main: str = "", client: str = "",
               article: str = "", event: str = "") -> Dict[str, Any]:
    """v79 (C4): a charge the first read's hypothesis gave (rec["charge_hypothesis"]) is labelled HYPOTHESIS_LABEL in
    the why -- §1 and §9 print it --; no charge at all is said in the reader's words (NO_CHARGE_WHY), never «il
    registro non contiene …». v79.10.3 (C2, review): event -- the certain facts' (event_of); the qualification
    alternative the outcome names fits it (event_mismatch)."""
    a = _assessment(records, posture, main, client, article, event)
    if a.get("article") and defence_record(a["article"], (records or {}).get(a["article"])):
        # v79.12.1: a defence record's outcome is never the charge template's (all integrato -> favourable)
        a = defence_assessment(records, a["article"], client, main_threat(records or {}))
    rec = (records or {}).get(a.get("article") or "") or {}
    if rec.get("charge_hypothesis"):
        a = {**a, "hypothesis": True, "why": f"{HYPOTHESIS_LABEL}: art. {a['article']}; {a['why']}"}
    return a


def _assessment(records: Dict[str, Dict[str, Any]], posture: Dict[str, Any], main: str = "", client: str = "",
                article: str = "", event: str = "") -> Dict[str, Any]:
    """{"esito", "grado", "obiettivo", "why", "article"}: the probable outcome read from the record, not from the side
    we write for. Defending (imputato, convenuto): the charge's or the claim's main article -- every element proven:
    unfavourable, the goal is containing it; most proven: unfavourable, a partial outcome (another qualification,
    no aggravante, the attenuanti); an element excluded: favourable, the full outcome. Claiming: the client's main
    article -- applicable: favourable; partly proven: uncertain; nothing proven or excluded: unfavourable."""
    who = client or "la parte assistita"
    role = str((posture or {}).get("role") or "")
    records = records or {}
    if role in ACCUSED:
        label = article if article in records and not frame_excluded(records[article]) else \
            main_threat(records)        # v79 (E1): the main line's article; v79.14.2 (B): never a frame-excluded one
        if not label:                   # v71: no charge on the record -- never «esito pieno»
            return {"esito": INCERTO, "grado": BASSO, "obiettivo": PARZIALE, "article": "",
                    "why": NO_CHARGE_WHY["civile" if (posture or {}).get("family") == "civile" else "penale"]}
        proved, excluded, total = _counts(records[label])
        weapons = [x for x, r in records.items() if r.get("side") == "weapon" and r.get("alternative")
                   and _counts(r)[0] and not _counts(r)[1] and not event_mismatch(x, event)]     # v79.10.3 (C2)
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
                           "provare per l'accusa" if criminal(posture) else
                           f"la maggior parte degli elementi dell'art. {label} risulta provata"}
        return {"esito": INCERTO, "grado": BASSO, "obiettivo": PIENO, "article": label,
                "why": f"la maggior parte degli elementi dell'art. {label} resta da provare"}
    # v78: a victim's outcome is the offence of the querela -- the charge (rec["charged"]), whatever the main topic;
    # v79.12.4 (item 1): a penal article only -- a civil one is a remedy under it, never the outcome's article
    from . import issue_records, posture as posture_mod
    victim = role in VICTIM_ROLES
    penal_ok = (lambda x: posture_mod.penal_article(x)) if victim else (lambda x: True)
    victim_charge = next((x for x in _charged(records) if _live(records[x]) and penal_ok(x)), "") if victim else ""
    if article in records and not victim and not frame_excluded(records[article]):     # v79 (E1): the main line's
        victim_charge = article                                                         # article (v79.14.2 B: bound)
    label = victim_charge or (main if main in records and penal_ok(main) and not frame_excluded(records[main]) else next(
        (x for x, r in records.items() if r.get("side") != "threat" and _live(r) and penal_ok(x)
         and not issue_records.defence_article(x) and not frame_excluded(r)), ""))
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


def main_article(t: Dict[str, Any], records: Dict[str, Dict[str, Any]], known: Dict[str, Dict[str, Any]]) -> str:
    """v79 (E1): the article a selected line is about -- its own (a riqualificazione, a claim, a remedy's), else the
    article of the elements it contests or rests on; a fattispecie of the record only ("" otherwise: the containment
    line, a defence of the general part)."""
    from . import issue_records
    ids = list(t.get("contesta") or []) + list(t.get("poggia_su") or [])
    art = str(t.get("article") or "") or next(((known.get(x) or {}).get("article") for x in ids if known.get(x)), "")
    rec = (records or {}).get(art or "") or {}
    return art if rec and issue_records.counted(art, str(rec.get("rubrica") or "")) else ""


def forecast(ctx: Dict[str, Any], t: Dict[str, Any], a: Dict[str, Any], log: Optional[Dict[str, Any]] = None
             ) -> Dict[str, Any]:
    """v79 (E1): the assessment computed on the chosen main line's article -- a victim's or a parte civile's stays on
    the querela's offence (v78b); a line on no fattispecie (the containment, a general-part defence) keeps the
    charge's. Logged (log["forecast"])."""
    posture = ctx.get("posture") or {}
    if posture.get("role") in VICTIM_ROLES:
        return a
    records = ctx.get("records") or {}
    cand = next((c for c in ctx.get("candidates") or [] if t.get("candidate") and c.get("id") == t.get("candidate")), {})
    if cand.get("kind") == "eccezione" and cand.get("reduction") and cand.get("article"):
        b = reduction_assessment(cand, records, ctx.get("client") or "")     # v79.10.2 (B1): on the reduction's article
        if log is not None:
            log["forecast"] = {"was": a.get("article"), "now": b["article"], "esito": b.get("esito"), "reduction": True}
        return b
    if cand.get("kind") in ("scriminante", "defence") and _live(records.get(cand.get("article") or "") or {}) \
            and defence_record(cand["article"], records.get(cand["article"])):
        # v79.12.1: a defence main line is assessed on its own record with its own words -- every element integrato,
        # the outcome is favourable; never the charge's template, never left on the charge's outcome
        b = defence_assessment(records, cand["article"], ctx.get("client") or "",
                               a.get("article") if a.get("article") and not a.get("defence") else "")
        if log is not None:
            log["forecast"] = {"was": a.get("article"), "now": b["article"], "esito": b.get("esito"), "defence": True}
        return b
    art = main_article(t, records, elements_by_id(records, ctx.get("circumstances")))
    if not art or art == a.get("article") or frame_excluded(records.get(art)):     # v79.14.2 (B): never «Esito
        return a                                                                    # probabile» on an excluded one
    b = assessment(records, posture, client=ctx.get("client") or "", article=art,
                   event=event_of(certain_text(ctx)))           # v79.10.3 (C2, review): the event's alternative only
    if log is not None:
        log["forecast"] = {"was": a.get("article"), "now": art, "esito": b.get("esito")}
    return b


# v79.10.2 (B1), my drafts: the forecast of a reduction line, on the reduction's own article
REDUCTION_WHY = {"excluded": "un presupposto della riduzione (art. {art}) è escluso dai fatti",
                 "proven": "i presupposti della riduzione (art. {art}) risultano provati: {demand} si riduce",
                 "partly": "{proved} presupposti su {total} della riduzione (art. {art}) risultano provati",
                 "needs": "{demand} resta; la sua riduzione (art. {art}) dipende da {need}",
                 "open": "{demand} resta; la sua riduzione (art. {art}) dipende dai fatti ancora da provare",
                 "demand": "la pretesa", "demand_art": "la pretesa (art. {claim})"}


def reduction_assessment(c: Dict[str, Any], records: Dict[str, Dict[str, Any]], client: str = "") -> Dict[str, Any]:
    """v79.10.2 (B1): the forecast of a reduction line (1384 c.c. for a penalty) on its own article -- the demand stays,
    the goal is its reduction (esito parziale): the reduction's record with every element proven -> favourable; one
    excluded -> unfavourable; else uncertain; no record -> uncertain, what it needs from the table («requires»)."""
    art = str(c.get("article") or "")
    claim = str(c.get("claim") or "")
    demand = REDUCTION_WHY["demand_art"].format(claim=claim) if claim else REDUCTION_WHY["demand"]
    base = {"obiettivo": PARZIALE, "article": art, "reduction": True}
    rec = (records or {}).get(art) or {}
    if _live(rec):
        proved, excluded, total = _counts(rec)
        if excluded:
            return {**base, "esito": SFAVOREVOLE, "grado": MEDIO, "why": REDUCTION_WHY["excluded"].format(art=art)}
        if total and proved == total:
            return {**base, "esito": FAVOREVOLE, "grado": MEDIO,
                    "why": REDUCTION_WHY["proven"].format(art=art, demand=demand)}
        return {**base, "esito": INCERTO, "grado": MEDIO,
                "why": REDUCTION_WHY["partly"].format(proved=proved, total=total, art=art)}
    need = _plain(c.get("requires")).rstrip(".")
    why = REDUCTION_WHY["needs"].format(demand=demand, art=art, need=need) if need else \
        REDUCTION_WHY["open"].format(demand=demand, art=art)
    return {**base, "esito": INCERTO, "grado": BASSO, "why": why}


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

# v79.10.3 (C1): the charge's pena as the procedural store states it (procedural_facts.extract: the first comma's) --
# the reclusione or the arresto with its bounds, «non inferiore a» up to the general maximum (art. 23 c.p.), the
# ergastolo, a pecuniary pena alone; the general minimum where the text states none (artt. 23, 25 c.p.)
_DETENTION_MIN = {"reclusione": 15 / 365, "arresto": 5 / 365}


def pena_of(text: str) -> Dict[str, Any]:
    """v79.10.3 (C1): {known, low, high, ergastolo, pecuniary, text} of the first pena the text states, in years -- the
    first detention or ergastolo it names, a pecuniary pena only when it names neither; known False when it states
    none (the rite then «da verificare sulla pena edittale»)."""
    from . import procedural_facts as pf
    t = " ".join(str(text or "").split())
    out: Dict[str, Any] = {"known": False, "low": None, "high": None, "ergastolo": False, "pecuniary": False,
                           "contravvenzione": False, "text": t}     # (review, v79.12.7) the arresto: a contravvenzione
    q, unit = pf._QTY, r"(mes[ei]|ann[oi]|giorni)"
    found = []
    m = re.search(r"\bergastolo\b", t, re.I)
    if m:
        found.append((m.start(), {"known": True, "ergastolo": True}))
    m = re.search(r"\b(reclusione|arresto) da " + q + r"(?: " + unit + r")? a " + q + r" " + unit, t, re.I)
    if m:
        found.append((m.start(), {"known": True, "low": pf._years(m.group(2), m.group(3) or m.group(5)),
                                  "high": pf._years(m.group(4), m.group(5)),
                                  "contravvenzione": m.group(1).lower() == "arresto"}))
    m = re.search(r"\b(reclusione|arresto) fino a " + q + r" " + unit, t, re.I)
    if m:
        found.append((m.start(), {"known": True, "low": _DETENTION_MIN[m.group(1).lower()],
                                  "high": pf._years(m.group(2), m.group(3)),
                                  "contravvenzione": m.group(1).lower() == "arresto"}))
    m = re.search(r"\breclusione non inferiore (?:a " + q + r" " + unit + r"|ad? (anni|mesi) " + q + r")", t, re.I)
    if m:
        low = pf._years(m.group(1), m.group(2)) if m.group(1) else pf._years(m.group(4), m.group(3))
        found.append((m.start(), {"known": True, "low": low, "high": float(pf.MAX_RECLUSIONE)}))
    if found:
        got = min(found, key=lambda x: x[0])[1]
        return {**out, **got} if got.get("ergastolo") or got.get("high") is not None else out
    if re.search(r"\b(?:multa|ammenda)\s+(?:da|fino\s+a|non\s+inferiore)\b", t, re.I):
        return {**out, "known": True, "pecuniary": True, "low": 0.0, "high": 0.0}
    return out


AVAILABLE, TO_VERIFY_RITE, PRECLUDED = "ammesso", "da verificare", "precluso"
# v79.10.7: the patteggiamento whose best case is within the limit -- never simply «ammesso»: the pena in concreto decides
POSSIBLE = "possibile"
ADMITTED = (AVAILABLE, POSSIBLE)                                # what a written sentence may offer (unavailable_rites)
RITE_TO_VERIFY = "da verificare sulla pena edittale"            # the brief's words: never «ammesso» without the pena
RITE_TO_VERIFY_LINE = "{name}: da verificare sulla pena edittale del reato contestato ({condition})."   # my draft
RITI_PRECLUDED_TITLE = "RITI PRECLUSI DALLA PENA DEL REATO CONTESTATO (non proporli):"     # my draft (writer material)
RITI_PRECLUDED_HEAD = "Riti alternativi preclusi dalla pena edittale"                       # my draft (§8)
# my draft (§10); (review) in the rito's own words -- a §10 point naming a rito without them is one rites_out drops
RITE_POINT = "L'accesso a {names}: da verificare sulla pena edittale del reato contestato."
# my draft; v79.10.7: «{word}» the rito's own gender (the table's «precluded_word», «preclusa» by default); «min» the
# best case (best_case_text), «ergastolo» with the rule's source when the table names one (the abbreviato's 438 co. 1-bis)
RITE_PRECLUDED = {"max": "{name}: {word}, il massimo edittale del reato contestato supera {n} anni.",
                  # v79.12.5 (item 8): never the draft's own status in the printed line -- it stays in the table's
                  # «status» («bozza — da verificare dall'avvocato …»)
                  "max_listed": "{name}: {word}, il massimo edittale del reato contestato supera {n} anni e il reato "
                                "non figura tra quelli indicati dall'{source}.",
                  # v79.12.7 (item 6): the computation shown, step by step (rite_computation), never only asserted
                  "min": "{name}: {word}, perché anche nel caso più favorevole la pena supera {n} anni. Calcolo: "
                         "{computation}.",
                  "ergastolo": "{name}: {word}, il reato contestato è punito con l'ergastolo{source_paren}."}
# v79.10.7 (my draft): the best case's words -- the minimum, the attenuanti held, the rite's third, the result
BEST_CASE = "il minimo edittale di {low}, {attenuanti} di un terzo per il rito, scende a {best}"
BEST_CASE_ATTENUANTI = "diminuito per {which} e"
BEST_CASE_NONE = "senza attenuanti integrate o da provare e diminuito"
# a rito named in a sentence, by its id; what says it is not available (a sentence naming it with one of these words
# does not offer it)
RITE_NAMES = {"patteggiamento": re.compile(r"\b444(?:\s+ss\.)?\s*(?:,\s*co(?:mma|\.)\s*\d+(?:-bis)?\s*,?\s*)?c\.p\.p\.|"
                                           r"\bpatteggia\w*|\bapplicazione\s+della\s+pena\s+su\s+richiesta", re.I),
              "messa_alla_prova": re.compile(r"\b168-bis\b|\b464-bis\b|\bmessa\s+alla\s+prova\b", re.I),
              "abbreviato": re.compile(r"\b(?:giudizio|rito)\s+abbreviato\b", re.I)}
# (review) a modal negation counts only on the rito's access («non può chiedere», «non può essere ammessa»), never on
# something else the sentence says of it («che non può essere revocato»); «da verificare» only as the rito's own words
_NOT_OFFERED_RE = re.compile(r"\bpreclus\w*|\besclus\w*|\binammissibil\w*|"
                             r"\bnon\s+(?:è\s+|sono\s+|risulta\s+|appare\s+)?(?:ammess\w*|ammissibil\w*|consentit\w*|possibil\w*|praticabil\w*|accessibil\w*|"
                             r"esperibil\w*|applicabil\w*|percorribil\w*)|\bnon\s+(?:può|possono|potrà|potranno|potrebbe|"
                             r"potrebbero)\s+(?:essere\s+)?(?:\w+\s+){0,2}?(?:chie\w*|richie\w*|propo\w*|ammess\w*|applica\w*|"
                             r"concess\w*|acced\w*|otten\w*|accordat\w*|dispo\w*|benefic\w*|fruir\w*|avvaler\w*)|"
                             r"\bnon\s+(?:consente|ammette)\b", re.I)
# (review) the clauses a sentence joins with «;», «, ma», «, però», «, mentre»: each clause offering a rito on its own
_CLAUSE_SPLIT = re.compile(r";|,\s*(?=(?:ma|però|mentre)\b)|\s(?=ma\s)", re.I)


# v79.10.7: art. 444 c.p.p. is judged on the pena in concreto, after the circumstances, then reduced by up to a third:
# the best case is the edittale minimum less each attenuante the record holds as integrato or da provare (its table's
# «max_reduction»: up to a third for a common one, art. 65 n. 3 c.p., 62-bis included; one of special effect, by its
# own measure, first -- art. 63, co. 3, c.p. --, the common ones after, each on the pena the previous one left; with
# two or more never below a quarter of the minimum, art. 67, co. 2, c.p.), then the rite's third
RITE_REDUCTION = 1 / 3
COMMON_REDUCTION = 1 / 3
ATTENUANTI_FLOOR = 1 / 4


def _fraction(value: Any, default: float) -> float:
    """«1/3» -> 0.333…; a number as it is; anything else the default."""
    text = str(value if value is not None else "").strip()
    m = re.fullmatch(r"(\d+)\s*/\s*(\d+)", text)
    if m and int(m.group(2)):
        return int(m.group(1)) / int(m.group(2))
    try:
        return float(text) if text else default
    except ValueError:
        return default


def attenuante_measures(labels: Sequence[str]) -> List[Dict[str, Any]]:
    """v79.10.7: [{label, reduction, special}] of the attenuanti given, in the order art. 63 c.p. applies them -- those
    of special effect first, then the common ones (the table's «effect»); the measure is the table's «max_reduction»,
    a third (art. 65 n. 3 c.p.) for one the table does not hold. 62-bis counts once (art. 62-bis, co. 1, c.p.)."""
    table = {str(c.get("label")): c for c in data().get("circumstances") or [] if c.get("kind") == "attenuante"}
    out = []
    for label in dict.fromkeys(str(x) for x in labels or [] if x):
        c = table.get(label) or {}
        out.append({"label": label, "reduction": _fraction(c.get("max_reduction"), COMMON_REDUCTION),
                    "special": c.get("effect") == "speciale"})
    return sorted(out, key=lambda x: not x["special"])


def best_case(low: float, attenuanti: Sequence[str] = ()) -> Dict[str, Any]:
    """v79.10.7: the patteggiamento's best case on the edittale minimum (years) -- {after_circumstances, best, applied}:
    each attenuante held (attenuante_measures, in art. 63's order) at its full measure, never below a quarter of the
    minimum with two or more (art. 67, co. 2, c.p.), then the rite's third (art. 444, co. 1, c.p.p.)."""
    measures = attenuante_measures(attenuanti)
    pena = float(low)
    for x in measures:
        pena *= 1 - x["reduction"]
    if len(measures) > 1:
        pena = max(pena, float(low) * ATTENUANTI_FLOOR)
    return {"after_circumstances": pena, "best": pena * (1 - RITE_REDUCTION), "applied": [x["label"] for x in measures]}


# v79.12.7 (item 6): the riti computed, not asserted -- the steps in order (my drafts): the edittale minimum, each
# attenuante held (art. 63's order, its measure), the floor of art. 67, co. 2, the rite's own reduction, the abbreviato's
# further sixth on non-impugnazione (art. 442, co. 2-bis, c.p.p.), the result against the patteggiamento's limit; the PM's
# consent (art. 444, co. 1, c.p.p.), unlikely for the offences the table lists (data: «consent_unlikely_offences»)
COMP_BASE = "minimo edittale {low}"
COMP_NONE = "senza attenuanti integrate o da provare"
COMP_ATTENUANTE = "meno fino a {fraction} per l'attenuante dell'art. {label}: {value}"
COMP_FLOOR = "non meno di un quarto del minimo (art. 67, co. 2, c.p.): {value}"
COMP_RITE = "meno {fraction} per il rito ({source}): {value}"
COMP_FURTHER = "meno un ulteriore {noun} se l'imputato non impugna la sentenza ({source}): {value}"
COMP_LIMIT = {"within": "entro {n} anni", "over": "oltre {n} anni"}
CONSENT_LINE = "Richiede il consenso del pubblico ministero ({source})."
CONSENT_UNLIKELY_LINE = ("Richiede il consenso del pubblico ministero ({source}), improbabile per un delitto come quello "
                         "contestato (art. {number} c.p.).")
FRACTION_WORDS = {(1, 2): ("la metà", "metà"), (1, 3): ("un terzo", "terzo"), (1, 4): ("un quarto", "quarto"),
                  (1, 6): ("un sesto", "sesto"), (2, 3): ("due terzi", "due terzi")}


def fraction_words(value: Any, default: float) -> Tuple[float, str, str]:
    """v79.12.7 (item 6): (the fraction, its words «un terzo», its noun «terzo») of «1/3» (FRACTION_WORDS)."""
    f = _fraction(value, default)
    from fractions import Fraction
    q = Fraction(f).limit_denominator(12)
    words = FRACTION_WORDS.get((q.numerator, q.denominator)) or (f"{q.numerator}/{q.denominator}",) * 2
    return f, words[0], words[1]


def computable(p: Dict[str, Any]) -> bool:
    """v79.12.7 (item 6): a pena the riti can be computed on -- known, a reclusione with its minimum (never the ergastolo
    nor a pena pecuniaria alone)."""
    return bool(p.get("known")) and not p.get("ergastolo") and not p.get("pecuniary") and \
        p.get("low") is not None and float(p["low"]) > 0


def rite_computation(r: Dict[str, Any], low: float, attenuanti: Sequence[str] = (), contravvenzione: bool = False
                     ) -> Dict[str, Any]:
    """v79.12.7 (item 6): one rito's computation on the edittale minimum (years) -- {steps, text, result, after_rite}:
    the minimum (COMP_BASE, BEST_CASE_NONE without an attenuante held), each attenuante held in art. 63's order at its
    measure (attenuante_measures), the floor of art. 67, co. 2 with two or more, the rite's reduction (the row's
    «computation.reduction», a third by default), the abbreviato's further sixth (its «further»); the patteggiamento's
    result read against its limit (COMP_LIMIT). The same numbers decide the status (_rite_gate): what is printed is what
    is judged. (review) contravvenzione: the row's «reduction_contravvenzione» when it has one (the abbreviato's half,
    art. 442, co. 2, c.p.p.) in place of the delitti's reduction."""
    comp = r.get("computation") or {}
    if contravvenzione and comp.get("reduction_contravvenzione"):
        comp = {**comp, "reduction": comp["reduction_contravvenzione"]}
    measures = attenuante_measures(attenuanti)
    pena = float(low)
    steps = [COMP_BASE.format(low=years_text(low)) + ("" if measures else f", {COMP_NONE}")]
    for x in measures:
        pena *= 1 - x["reduction"]
        _, words, _ = fraction_words(x["reduction"], COMMON_REDUCTION)
        steps.append(COMP_ATTENUANTE.format(fraction=words, label=x["label"], value=years_text(pena)))
    if len(measures) > 1 and pena < float(low) * ATTENUANTI_FLOOR:
        pena = float(low) * ATTENUANTI_FLOOR
        steps.append(COMP_FLOOR.format(value=years_text(pena)))
    reduction, words, _ = fraction_words(comp.get("reduction"), RITE_REDUCTION)
    pena *= 1 - reduction
    after_rite = pena
    steps.append(COMP_RITE.format(fraction=words, source=comp.get("reduction_source") or "", value=years_text(pena))
                 .replace(" ()", ""))
    if comp.get("further"):
        further, _, noun = fraction_words(comp.get("further"), 0.0)
        pena *= 1 - further
        steps.append(COMP_FURTHER.format(noun=noun, source=comp.get("further_source") or "", value=years_text(pena))
                     .replace(" ()", ""))
    limit = r.get("max_years_after_reduction")
    if limit is not None:
        steps[-1] += ", " + COMP_LIMIT["over" if after_rite > float(limit) + 1e-9 else "within"].format(n=int(limit))
    return {"steps": steps, "text": "; ".join(steps), "result": pena, "after_rite": after_rite,
            "applied": [x["label"] for x in measures]}


def consent_line(r: Dict[str, Any], number: str = "") -> str:
    """v79.12.7 (item 6): the patteggiamento's line on the PM's consent -- always required (art. 444, co. 1, c.p.p.),
    unlikely for an offence the row lists («consent_unlikely_offences»: data, a draft for the lawyer)."""
    comp = r.get("computation") or {}
    if not comp.get("consent_source"):
        return ""
    if number and number in {str(x) for x in comp.get("consent_unlikely_offences") or []}:
        return CONSENT_UNLIKELY_LINE.format(source=comp["consent_source"], number=number)
    return CONSENT_LINE.format(source=comp["consent_source"])


def years_text(years: float) -> str:
    """v79.10.7: a pena in years as the lawyer writes it -- «10 anni», «6 anni e 8 mesi», «4 anni, 5 mesi e 10 giorni»."""
    total = round(float(years) * 12 * 30)               # days, by thirty-day months (the practice of the computation)
    y, rest = divmod(total, 360)
    m, d = divmod(rest, 30)
    parts = [f"{y} ann{'o' if y == 1 else 'i'}" if y else "", f"{m} mes{'e' if m == 1 else 'i'}" if m else "",
             f"{d} giorn{'o' if d == 1 else 'i'}" if d else ""]
    parts = [x for x in parts if x] or ["0 giorni"]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " e " + parts[-1]


def best_case_text(low: float, attenuanti: Sequence[str] = ()) -> str:
    """v79.10.7: the best case in words (BEST_CASE) -- the minimum, the attenuanti counted, the rite's third, the result."""
    got = best_case(low, attenuanti)
    labels = got["applied"]
    if labels:
        which = (f"l'attenuante dell'art. {labels[0]}" if len(labels) == 1 else
                 "le attenuanti degli artt. " + ", ".join(x.replace(" c.p.", "") for x in labels[:-1]) + " e "
                 + labels[-1])
        att = BEST_CASE_ATTENUANTI.format(which=which)
    else:
        att = BEST_CASE_NONE
    return BEST_CASE.format(low=years_text(low), attenuanti=att, best=years_text(got["best"]))


def _rite_gate(r: Dict[str, Any], p: Dict[str, Any], number: str, attenuanti: Sequence[str] = ()) -> Tuple[str, str]:
    """v79.10.3 (C1): (status, why) of one rito of the table for the charge's pena (pena_of) and article number -- a
    listed offence (art. 550, co. 2, c.p.p.) admitted to the messa alla prova whatever its pena; the pena unknown, «da
    verificare»; else the table's bounds. v79.10.7: the abbreviato excluded for an offence punished with the ergastolo
    (art. 438, co. 1-bis, c.p.p.: «excluded_ergastolo»), the pena unknown «da verificare» as for the others; the
    patteggiamento on its best case (best_case: the attenuanti the record holds, attenuanti), POSSIBLE within the
    limit, never AVAILABLE."""
    caps = ("max_edittale_years", "max_years_after_reduction")
    if r.get("excluded_ergastolo"):
        if not p.get("known"):
            return TO_VERIFY_RITE, "unknown"
        return (PRECLUDED, "ergastolo") if p.get("ergastolo") else (AVAILABLE, "pena")
    if r.get("always") or not any(r.get(k) is not None for k in caps):
        return AVAILABLE, "always"
    if number and number in {str(x) for x in r.get("listed_offences") or []}:
        return AVAILABLE, "listed"
    if not p.get("known"):
        return TO_VERIFY_RITE, "unknown"
    if p.get("ergastolo"):
        return PRECLUDED, "ergastolo"
    limit = r.get("max_years_after_reduction")
    if p.get("pecuniary"):
        return (POSSIBLE, "pecuniary") if limit is not None else (AVAILABLE, "pecuniary")
    cap = r.get("max_edittale_years")
    if cap is not None:
        if p.get("high") is None:
            return TO_VERIFY_RITE, "unknown"
        if p["high"] > cap:
            return PRECLUDED, "max_listed" if r.get("listed_offences") else "max"
    if limit is not None:
        if p.get("low") is None:
            return TO_VERIFY_RITE, "unknown"
        best = rite_computation(r, p["low"], attenuanti, bool(p.get("contravvenzione")))["after_rite"] \
            if r.get("computation") else \
            best_case(p["low"], attenuanti)["best"]          # v79.12.7 (item 6): the printed numbers decide
        if best > limit + 1e-9:
            return PRECLUDED, "min"
        return POSSIBLE, "pena"
    return AVAILABLE, "pena"


def rite_statuses(pena_text: str = "", number: str = "", attenuanti: Sequence[str] = ()) -> List[Dict[str, Any]]:
    """v79.10.3 (C1): every rito of the table (data/lawyer_method.json, riti_alternativi) with its status for the
    charge -- {id, name, status (AVAILABLE / POSSIBLE / TO_VERIFY_RITE / PRECLUDED), why, line, rite}: the table's line
    when admitted, RITE_TO_VERIFY_LINE when the pena is unknown, a RITE_PRECLUDED line when the pena precludes it.
    pena_text: the store's «pena» of the charge (or its article's text); number: the charge's c.p. article ("" for
    another code). v79.10.7: attenuanti, the attenuanti the record holds as integrato or da provare (held_attenuanti) --
    the patteggiamento's best case (best_case); POSSIBLE its «possible_line» with the best case in words."""
    p = pena_of(pena_text)
    out = []
    for r in data().get("riti_alternativi") or []:
        line = str(r.get("line") or "")
        name = line.split(":")[0]
        status, why = _rite_gate(r, p, str(number or ""), attenuanti)
        # v79.12.7 (item 6): a rito with a computation (the row's «computation») on a pena it can be computed on prints
        # it, step by step -- what the status says is what the numbers show
        comp = rite_computation(r, p["low"], attenuanti, bool(p.get("contravvenzione"))) \
            if r.get("computation") and computable(p) else None
        if status == AVAILABLE:
            text = str(r.get("listed_line") or line) if why == "listed" else \
                str(r.get("computed_line")).format(computation=comp["text"]) if comp and r.get("computed_line") \
                else line
        elif status == POSSIBLE:
            text = str(r.get("possible_pecuniary_line") or line) if why == "pecuniary" else \
                str(r.get("possible_line") or line).format(
                    best=best_case_text(p["low"], attenuanti), computation=comp["text"] if comp else "",
                    consent=consent_line(r, str(number or ""))).strip()
        elif status == TO_VERIFY_RITE:
            cond = str(r.get("to_verify") or "")
            text = RITE_TO_VERIFY_LINE.format(name=name, condition=cond if _ABBREV_END_RE.search(cond) else
                                              cond.rstrip("."))        # v79.10.7: «c.p.p.» keeps its dot
        else:
            n = r.get("max_edittale_years") if why.startswith("max") else r.get("max_years_after_reduction")
            src = str(r.get("excluded_source") or "")
            text = RITE_PRECLUDED[why].format(
                name=name, word=str(r.get("precluded_word") or "preclusa"), n=int(n) if n is not None else "",
                source=r.get("listed_source") or "", provisional=str(r.get("provisional") or "").strip(),
                source_paren=f" ({src})" if src else "",
                best=best_case_text(p["low"], attenuanti) if why == "min" else "",
                computation=comp["text"] if comp else "").replace(" .", ".")
            text = text[:-1] if text.endswith("..") else text         # v79.12.5: «… c.p.p.» ends the sentence
            # (review, v79.12.7 item 6): the PM's consent, unlikely for the offences the row lists, said with the
            # precluded computation too -- the brief's line for 584-class offences, whatever the numbers give
            unlikely = consent_line(r, str(number or "")) if comp and why == "min" else ""
            if unlikely and str(number or "") in {str(x) for x in (r.get("computation") or {}).get(
                    "consent_unlikely_offences") or []}:
                text = f"{text} {unlikely}"
        out.append({"id": str(r.get("id") or ""), "name": name, "status": status, "why": why, "line": text,
                    "rite": not r.get("sentencing"),
                    # v79.12.7 (item 6): the computation the line must show, and its result (the counters read them)
                    "computation": comp["text"] if comp else "", "computable": bool(comp),
                    "result": comp["after_rite"] if comp else None,
                    "limit": r.get("max_years_after_reduction"),
                    # (review) the figures a written sentence on this rito may state: the computation's, the
                    # edittale bounds, the limit -- in days (rite_prose_out)
                    "figures": sorted({figure_days(x) for x in [p.get("low"), p.get("high"),
                                                                 r.get("max_years_after_reduction")] if x is not None}
                                      | set(figures_in(comp["text"]) if comp else []))})
    return out


def held_attenuanti(circumstances: Optional[Dict[str, Dict[str, Any]]]) -> List[str]:
    """v79.10.7: the attenuanti the record holds as integrato or da provare -- each attenuante circumstance record that
    is not excluded (an element «non risulta»: its verdict «non applicabile», never counted) and that the record holds:
    (review) one of its elements «provato» or with a supporting sentence of the case. A candidate no fact touches --
    the 62-bis the imputato's posture always lists, or any ungraded one -- is not held: the record says nothing of it,
    and the best case is the minimum without it (584 c.p. precluded without a fact for the generiche). v79.12.7 (item
    2): a circumstance a fact of the case triggers (rec["triggered_by"]: the diverbio for 62 n. 2, the concrete step for
    62 n. 6, the favourable element for 62-bis) is held, «da provare», unless the record excludes it."""
    from . import issue_records
    out = []
    for label, rec in (circumstances or {}).items():
        if (rec or {}).get("kind") != "attenuante":
            continue
        if issue_records._verdict(rec) == issue_records.NOT_APPLICABLE:
            continue
        reqs = [r for r in rec.get("requisites") or [] if r.get("kind", "requisito") == "requisito"]
        if any(r.get("status") == PROVATO or r.get("supporting") or (r.get("link") or {}).get("supporting")
               for r in reqs) or rec.get("triggered_by"):     # v79.12.7 (item 2): a fact that triggers it is support
            out.append(str(label))
    return out


def _rites_posture(posture: Optional[Dict[str, Any]]) -> bool:
    """The imputato's subordinate line: penal intake rito, imputato, before the sentence (v70, v75)."""
    return (posture or {}).get("role") == "imputato" and (posture or {}).get("stage") not in ("impugnazione",
                                                                                              "esecuzione") \
        and _family_key(posture or {}) == "penale"          # v75: the intake rito's, never the frame's area


def riti_lines(posture: Dict[str, Any], charged_text: str = "", number: str = "", attenuanti: Sequence[str] = ()
               ) -> List[str]:
    """The imputato's subordinate line (penal postures, imputato, before the sentence): the riti the charge's pena
    allows. v79.10.3 (C1): a rito the pena precludes is no longer a line here (riti_precluded: §8 says so); the pena unknown,
    the rito «da verificare sulla pena edittale», never as admitted."""
    if not _rites_posture(posture):
        return []
    return [x["line"] for x in rite_statuses(charged_text, number, attenuanti) if x["status"] != PRECLUDED]


def riti_precluded(posture: Dict[str, Any], charged_text: str = "", number: str = "", attenuanti: Sequence[str] = ()
                   ) -> List[str]:
    """v79.10.3 (C1): the riti the charge's pena precludes, each with why -- §8's lines, never under a strategy line."""
    if not _rites_posture(posture):
        return []
    return [x["line"] for x in rite_statuses(charged_text, number, attenuanti) if x["status"] == PRECLUDED]


def riti_names(riti: Sequence[str], n: int) -> str:
    """v79.10.3 (C1): the first n riti admitted, by name («; ») -- one «da verificare sulla pena edittale» is never
    named among what the containment line offers."""
    return "; ".join([x.split(":")[0] for x in riti or [] if RITE_TO_VERIFY not in x][:n])


def charge_pena(label: str, rows: Sequence[Dict[str, Any]]) -> Tuple[str, str]:
    """v79.10.3 (C1): (the charge's pena from the procedural store, its c.p. number) -- procedural_facts.by_number (the
    store's entry, else computed from the row's text); ("", "") for no charge, ("", number) when the store holds none,
    ("", "") for another code's article (the store reads the c.p. alone: the pena then unknown)."""
    from . import procedural_facts
    number, _, code = str(label or "").partition(" ")
    if not number or code.strip() != "c.p.":
        return "", ""
    facts = procedural_facts.by_number(rows or []).get(number) or {}
    return str(facts.get("pena") or ""), number


def charge_rites(records: Dict[str, Dict[str, Any]], rows: Sequence[Dict[str, Any]], posture: Dict[str, Any],
                 article: str = "", circumstances: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """v79.10.3 (C1): the riti's statuses for the case's charge (main_threat, else the given article, else the intake's
    first charge) in a penal matter; [] outside one. One computation for the method's lines (prepare), §8's block
    (render_computed.stage_blocks) and the plan's subordinate lines (posture.apply_plan). v79.10.7: circumstances, the
    case's circumstance records -- the attenuanti held (held_attenuanti) give the patteggiamento's best case; each
    status carries them («attenuanti»)."""
    if _family_key(posture or {}) != "penale":
        return []
    from . import posture as posture_mod, issue_records
    # v79.12.4 (item 3): among several charged articles the one with the highest maximum pena from the store sets the
    # riti, never the list's order (posture.heaviest; main_threat's, then the order, when the store states none)
    threats = [x for x in _charged(records or {}) if (records or {})[x].get("side") == "threat" and _live(records[x])
               and issue_records.counted(x, str(records[x].get("rubrica") or ""))]
    label = posture_mod.heaviest([main_threat(records or {})] + threats, rows) if threats else ""
    label = label or main_threat(records or {}) or article or next(iter(posture_mod.intake_charges(posture)), "")
    pena_text, number = charge_pena(label, rows)
    held = held_attenuanti(circumstances)
    return [{**x, "charge": label, "pena": pena_text, "attenuanti": held}
            for x in rite_statuses(pena_text, number, held)]


def rite_of(text: str) -> str:
    """v79.10.3 (C1): the rito a line names (RITE_NAMES), "" for none."""
    return next((k for k, rx in RITE_NAMES.items() if rx.search(text or "")), "")


def unavailable_rites(statuses: Sequence[Dict[str, Any]]) -> set:
    """v79.10.3 (C1): the riti (RITE_NAMES ids) no status admits -- several clients' statuses together: a rito one of
    them is admitted to is never removed from the shared text."""
    admitted = {x.get("id") for x in statuses or [] if x.get("status") in ADMITTED}      # v79.10.7: POSSIBLE too
    unavailable = {x.get("id") for x in statuses or [] if x.get("status") not in ADMITTED and x.get("id") in RITE_NAMES}
    return unavailable - admitted


# v79.10.7 (review): a rito only «possibile» (the patteggiamento's best case within the limit) is never written as
# simply available -- a clause saying so («è ammesso», «è sicuramente accessibile») without a condition goes
_UNCONDITIONAL_RE = re.compile(r"\b(?:ammess\w*|ammissibil\w*|disponibil\w*|accessibil\w*|consentit\w*|esperibil\w*|"
                               r"praticabil\w*|garantit\w*|certament\w*|sicurament\w*|senz['’]altro)\b", re.I)
_CONDITIONAL_RE = re.compile(r"\bpossibil\w*|\bdipend\w*|\bin\s+concreto\b|\bse\b|\bqualora\b|\bove\b|\beventual\w*|"
                             r"\bsubordinat\w*\s+(?:a|al|alla|all['’]|ai|alle)\b|\bcondizion\w*|"
                             r"\bda\s+verificare\b", re.I)


def possible_only_rites(statuses: Sequence[Dict[str, Any]]) -> set:
    """v79.10.7 (review): the riti (RITE_NAMES ids) some status holds «possibile» and none simply admits."""
    possible = {x.get("id") for x in statuses or [] if x.get("status") == POSSIBLE and x.get("id") in RITE_NAMES}
    return possible - {x.get("id") for x in statuses or [] if x.get("status") == AVAILABLE}


def _offers(sentence: str, bad: set, possible: set = frozenset()) -> bool:
    """A clause naming a rito not admitted that neither says it is not (_NOT_OFFERED_RE) nor «da verificare sulla
    pena edittale» -- (review) per clause, the negation or the words in the rito's own clause. v79.10.7 (review): or a
    clause naming a rito only «possibile» as simply available (_UNCONDITIONAL_RE) with no condition
    (_CONDITIONAL_RE)."""
    for c in _CLAUSE_SPLIT.split(sentence):
        if _NOT_OFFERED_RE.search(c) or RITE_TO_VERIFY in c.lower():
            continue
        if any(RITE_NAMES[k].search(c) for k in bad):
            return True
        if any(RITE_NAMES[k].search(c) for k in possible) and _UNCONDITIONAL_RE.search(c) \
                and not _CONDITIONAL_RE.search(c):
            return True
    return False


def _pieces_of(line: str) -> List[str]:
    """A line's sentences -- a table row's cells' sentences (its «|» kept apart)."""
    if line.lstrip().startswith("|"):
        return [x for cell in line.split("|") for x in _SENT_SPLIT.split(cell)]
    return _SENT_SPLIT.split(line)


_OWN_LEAD_RE = re.compile(r"^\s*(?:[-•*]|\d+[.)])?\s*(?:\*\*)?\s*")


def _own_norm(text: str) -> str:
    return " ".join(_OWN_LEAD_RE.sub("", str(text or "")).replace("**", "").split()).strip()


def own_rite_sentence(piece: str, statuses: Sequence[Dict[str, Any]]) -> bool:
    """(review, v79.12.7) A sentence of the code's own rito lines (a status's line: its computation steps, the PM's
    consent) -- never a written offer, never a written figure: the gate and the figures check leave it alone."""
    norm = _own_norm(piece)
    return len(norm) >= 20 and any(norm in _own_norm(x.get("line")) for x in statuses or [])


def rites_offered_unavailable(text: str, statuses: Sequence[Dict[str, Any]]) -> List[str]:
    """v79.10.3 (C1): the sentences of the text naming a rito the charge does not admit (precluded, or its pena
    unknown) without saying so (_NOT_OFFERED_RE: «preclusa», «non è ammesso», «da verificare» …) -- the counter
    rites_unavailable_offered; must be none. (review, v79.12.7) The code's own lines aside (own_rite_sentence): the
    precluded patteggiamento's «Calcolo: … meno un terzo per il rito (art. 444 …)» names it to show why."""
    bad, possible = unavailable_rites(statuses), possible_only_rites(statuses)
    if not bad and not possible:
        return []
    return [x.strip()[:200] for line in (text or "").split("\n") for x in _pieces_of(line)
            if _offers(x, bad, possible) and not own_rite_sentence(x, statuses)]


_BARE_ITEM_RE = re.compile(r"^\s*(?:[-•*]|\d+[.)])?\s*(?:\*\*)?\s*$")


def rites_out(text: str, statuses: Sequence[Dict[str, Any]], log: Optional[List[str]] = None) -> str:
    """v79.10.3 (C1): a written sentence offering a rito the charge does not admit (rites_offered_unavailable) dropped,
    each logged -- the code's own lines say what is precluded or still to verify; in a table row, the cell's sentence
    alone."""
    bad, possible = unavailable_rites(statuses), possible_only_rites(statuses)
    if not (bad or possible) or not rites_offered_unavailable(text, statuses):
        return rite_prose_out(text, statuses, log)

    def kept(piece: str) -> bool:
        if _offers(piece, bad, possible) and not own_rite_sentence(piece, statuses):
            if log is not None:
                log.append(piece.strip()[:200])
            return False
        return True
    out = []
    for line in (text or "").split("\n"):
        if line.lstrip().startswith("|"):
            cells = [(" " + " ".join(x.strip() for x in _SENT_SPLIT.split(cell) if kept(x)) + " ", True)
                     if _offers(cell, bad, possible) else (cell, False) for cell in line.split("|")]
            if not any(hit and not c.strip() for c, hit in cells):      # (review) a cell left empty: the row goes
                out.append("|".join(c for c, _ in cells))
            continue
        split = _SENT_SPLIT.split(line)
        pieces = [x for x in split if kept(x)]
        if not line.strip() or pieces and not (len(pieces) < len(split) and _BARE_ITEM_RE.match(" ".join(pieces))):
            out.append(" ".join(pieces))                # (review) never an orphan «2.» where its item went
    return rite_prose_out("\n".join(out), statuses, log)


# (review, v79.12.7 item 6): a written sentence on a computed rito -- the figures it states must be the computation's
# (or the edittale bounds, the limit); a reduction or a result it asserts with no figure is the computation asserted, not
# shown (data/penal_lines.json «rite_prose»: claim_patterns, reference_patterns)
_NUM_WORDS = {"un": 1, "uno": 1, "una": 1, "due": 2, "tre": 3, "quattro": 4, "cinque": 5, "sei": 6, "sette": 7,
              "otto": 8, "nove": 9, "dieci": 10, "undici": 11, "dodici": 12, "tredici": 13, "quattordici": 14,
              "quindici": 15, "sedici": 16, "diciassette": 17, "diciotto": 18, "diciannove": 19, "venti": 20,
              "ventuno": 21, "ventiquattro": 24, "trenta": 30}
_QTY_WORD = r"(\d+|" + "|".join(sorted(_NUM_WORDS, key=len, reverse=True)) + r")"
_FIGURE_RE = re.compile(r"\b" + _QTY_WORD + r"\s+(ann[oi]|mes[ei]|giorn[oi])\b"
                        r"(?:\s*(?:,|e)\s*" + _QTY_WORD + r"\s+(mes[ei]|giorn[oi])\b)?"
                        r"(?:\s*(?:,|e)\s*" + _QTY_WORD + r"\s+(giorn[oi])\b)?", re.I)
_UNIT_DAYS = {"a": 360, "m": 30, "g": 1}


def figure_days(years: Any) -> int:
    """A pena in years as days, by thirty-day months (years_text's practice)."""
    return int(round(float(years) * 360))


def figures_in(text: str) -> List[int]:
    """The pena figures a text states («6 anni e 8 mesi», «due anni»), each in days."""
    out = []
    for m in _FIGURE_RE.finditer(text or ""):
        days = 0
        for i in range(1, 7, 2):
            q, unit = m.group(i), m.group(i + 1)
            if q and unit:
                days += int(_NUM_WORDS.get(q.lower(), 0) if not q.isdigit() else q) * _UNIT_DAYS[unit[0].lower()]
        out.append(days)
    return out


def _rite_prose_rx(key: str) -> Optional["re.Pattern"]:
    from . import penal_lines
    pats = [p for p in ((penal_lines.data().get("rite_prose") or {}).get(key) or []) if p]
    try:
        return re.compile("|".join(f"(?:{p})" for p in pats), re.I) if pats else None
    except re.error as exc:
        logger.warning("lawyer_method: rite_prose %s: %s", key, exc)
        return None


def rite_prose_issue(piece: str, statuses: Sequence[Dict[str, Any]]) -> str:
    """«mismatch» when a written sentence naming a computed rito states a figure its computation does not print (nor
    the edittale bounds, the limit); «uncomputed» when it asserts a reduction or a result with no figure and no
    reference to the computation; "" otherwise -- the code's own lines never."""
    computed = [x for x in statuses or [] if x.get("computable") and x.get("computation") and x.get("id") in RITE_NAMES]
    if not computed or own_rite_sentence(piece, statuses):
        return ""
    named = [x for x in computed if RITE_NAMES[x["id"]].search(piece)]
    if not named:
        return ""
    allowed = {d for x in named for d in x.get("figures") or []}
    figures = figures_in(piece)
    if any(all(abs(f - a) > 1 for a in allowed) for f in figures):
        return "mismatch"
    claim, ref = _rite_prose_rx("claim_patterns"), _rite_prose_rx("reference_patterns")
    if not figures and claim is not None and claim.search(piece) and not (ref is not None and ref.search(piece)):
        return "uncomputed"
    return ""


def rite_prose_out(text: str, statuses: Sequence[Dict[str, Any]], log: Optional[List[str]] = None,
                   rewrite: bool = True) -> str:
    """The written sentences on a computed rito that state a figure the computation does not print, or assert its
    reduction with none (rite_prose_issue), dropped -- the code's rito line shows the computation -- each logged
    («mismatch: …», «uncomputed: …»); outside the tables. rewrite False: the text unchanged, only the log."""
    if not text or not any(x.get("computable") for x in statuses or []):
        return text

    def kept(piece: str) -> bool:
        issue = rite_prose_issue(piece, statuses)
        if issue and log is not None:
            log.append(f"{issue}: {piece.strip()[:200]}")
        return not issue or not rewrite
    out = []
    for line in text.split("\n"):
        if line.lstrip().startswith("|"):
            out.append(line)
            continue
        split = _SENT_SPLIT.split(line)
        pieces = [x for x in split if kept(x)]
        if len(pieces) == len(split):
            out.append(line)
        elif pieces and not _BARE_ITEM_RE.match(" ".join(pieces)):
            out.append(" ".join(pieces))
    return "\n".join(out) if rewrite else text


def precluded_block(lines: Sequence[str], procedure: str = "") -> str:
    """v79.10.3 (C1): §8's line of the riti the charge's pena precludes (each with why) -- those the section does not
    already say precluded; "" with none."""
    said = [x for x in lines or [] if x and not any(rx.search(ln) and re.search(r"\bpreclus", ln, re.I)
                                                    for ln in (procedure or "").split("\n")
                                                    for rx in [RITE_NAMES.get(rite_of(x)) or re.compile(r"$^")])]
    return f"{RITI_PRECLUDED_HEAD}:\n" + "\n".join(f"- {x}" for x in said) if said else ""


def rite_stage_lines(statuses: Sequence[Dict[str, Any]]) -> List[str]:
    """v79.10.3 (C1): §8's «Riti alternativi» block for the charge -- each rito's line as its status gives it (admitted,
    «da verificare sulla pena edittale», precluded with why); the sentencing line is no rito."""
    return [x["line"] for x in statuses or [] if x.get("rite", True)]


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


CRIMINAL_ROLES = ("imputato", "indagato", "persona_offesa", "parte_civile")


def criminal(posture: Optional[Dict[str, Any]]) -> bool:
    """v79 (E2): a criminal matter -- the penal family, or no family and a criminal role: «l'accusa», «aggravante»
    and the penal lines only then."""
    family = str((posture or {}).get("family") or "")
    return family == "penale" or (not family and str((posture or {}).get("role") or "") in CRIMINAL_ROLES)


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
    with no remissione and no 162-ter («penale_imputato_ufficio»). v79.10.3 (C1): the remissione and 162-ter only when
    the store says the charge is a querela offence -- d'ufficio, an irrevocable querela or nothing read, the variant
    without them; the victim's likewise («penale_offesa_ufficio»)."""
    table = data().get("stragiudiziale") or {}
    family, role = _family_key(posture), str((posture or {}).get("role") or "")
    querela = charge_procedibilita == "querela"
    if family == "penale":
        if role in ("imputato", "sconosciuto", ""):
            key = "penale_imputato" if querela else "penale_imputato_ufficio"
            return table.get(key) or table.get("penale_imputato") or ""
        return table.get("penale_offesa" if querela else "penale_offesa_ufficio") or table.get("penale_offesa") or ""
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
    from . import topic_groups              # v79.12.5 (item 10): the general clauses, the stage table's articles
    labels += list(topic_groups.default_admitted())
    written = [ctx.get("case") or "", ctx.get("temporal") or "", ctx.get("stragiudiziale") or ""]
    written += [f"{c.get('tesi') or ''} {c.get('articles') or ''}" for c in ctx.get("candidates") or []]
    written += list(ctx.get("litisconsorzio") or []) + list(ctx.get("riti") or []) + list(texts or [])
    scope = {_norm_label(x) for x in labels} | {citations.label(r) for t in written for r in citations.article_refs(t)}
    if ctx.get("fit_removed"):          # v79.13.0: a norm the applicability judgment removed is never in the scope
        scope -= {_norm_label(x) for x in ctx["fit_removed"]}
    gated = act_gated_labels(ctx, scope)     # v79.14.1 (item 3): no c.p.a. article without the administrative act
    if gated:
        scope -= gated
        ctx["act_scope_refused"] = sorted(gated)
    return scope


def act_gated_labels(ctx: Dict[str, Any], labels: Sequence[str]) -> set:
    """v79.14.1 (item 3): the labels of an act-gated code (the c.p.a.) when the case states no administrative act
    against the client (module_act) -- never one the user names in the case (its article, or the code by name)."""
    from . import citations, family_gate
    if module_act(ctx):
        return set()
    case = str(ctx.get("case") or "")
    if family_gate._cpa_named(case):
        return set()
    named = {citations.label(r) for r in citations.article_refs(case)}
    return {x for x in labels if family_gate.act_gated_code(str(x).partition(" ")[2]) and x not in named}


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


def _sentence_list(text: str) -> List[str]:
    """The sentences _by_sentence hands its review, in the same order."""
    out: List[str] = []
    for line in (text or "").split("\n"):
        if line.strip():
            out += [x for x in _citation_rejoin(_SENT_SPLIT.split(line)) if x.strip()]
    return out


def orphans_out(before: str, after: str, events: Optional[List[Dict[str, Any]]] = None, site: str = "") -> str:
    """v79.15.1 (item 4): `after` (the checks' output of `before`) with every sentence that opens with an anaphoric
    connective right after a sentence the checks removed taken out too (analysis_step.orphans; a run of them likewise)
    -- never a «Inoltre, …» left after its sentence went; into events as analysis_step.dependent_event."""
    from . import analysis_step
    gone = analysis_step.orphans(_sentence_list(before), _sentence_list(after))
    if not gone:
        return after
    at = [-1]

    def review(x: str) -> str:
        at[0] += 1
        if at[0] in gone:
            if events is not None:
                events.append(analysis_step.dependent_event(site, x))
            return ""
        return x
    return _by_sentence(after, review)


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
    "le OPZIONI STRAGIUDIZIALI. Non aggiungere tesi e non cambiarle: valuta OGNI TESI CANDIDATA, con il suo id (T1, "
    "T2…) come chiave, e indica per ciascuna il realismo (alto, medio, basso; per una tesi minoritaria: basso); per "
    "quelle da argomentare (non quelle segnate «argomentata dal codice», di cui dai solo il realismo) scrivi anche gli "
    "argomenti a favore e contrari e l'argomento in due-tre frasi. Gli argomenti di una tesi possono negare "
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
    # v79.14.3 (item 1): an element only another party's allegation supports -- the defence client's primary thesis
    "contested": "{who} deve provare «{el}» ({id}) dell'art. {art}: i fatti narrano solo un'allegazione {di_parte}",
    "integrated": "Contestare l'elemento «{el}» ({id}) dell'art. {art}, che i fatti integrano",
    "requalify": "Riqualificazione nell'art. {art}{rubrica}",
    # v79.12.7 (item 1): an element the note or a ruling states and the text does not -- contesting it is sostenibile
    "jurisprudential": "L'elemento «{el}» ({id}) dell'art. {art}, richiesto da {source}, non è provato",
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
    "riduzione": "Chiedere {label} ({articles})",          # v79.10.2 (B1): a reduction, the resisting party's (kind eccezione)
    "riconvenzionale": "Domanda riconvenzionale sull'art. {art}{rubrica}",
    "defence": "Difesa sull'art. {art}{rubrica}",          # v79 (D1): a general-part defence with its elements proven
    "transazione": "Proporre una transazione che chiuda la lite (artt. 1965 ss. c.c.)",     # v79.8 (B3)
}
VICTIM_ROLES = ("persona_offesa", "parte_civile")
KNOWN_ROLES = ("imputato", "indagato", "convenuto", "attore", "persona_offesa", "parte_civile", "terzo")     # v79.6
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


# --- v79.10.3 (C2): an alternative qualification fits the event --------------------------------------------------------------

EVENTS_FILE = os.path.join(os.path.dirname(DATA_FILE), "event_offences.json")
_EVENTS: Dict[str, Any] = {}


def event_table() -> Dict[str, Any]:
    """v79.10.3 (C2): data/event_offences.json (read once; {} when missing, logged)."""
    if "d" not in _EVENTS:
        try:
            with open(EVENTS_FILE, encoding="utf-8") as f:
                _EVENTS["d"] = json.load(f)
        except (OSError, ValueError) as exc:
            logger.warning("lawyer_method: %s not read: %s", EVENTS_FILE, exc)
            _EVENTS["d"] = {}
    return _EVENTS["d"]


def event_of(certain_text: str) -> str:
    """v79.10.3 (C2): the event the certain facts narrate (data/event_offences.json, in its order: «morte», then
    «lesioni» unless a death) -- the wordings that name it without its happening (a threat of death, a danger, an
    attempt) cut out first; "" for none."""
    table = event_table()
    text = str(certain_text or "")
    for rx in table.get("not_events") or []:
        text = re.sub(rx, " ", text, flags=re.I)
    # (review) a death dated by its year alone before the latest year the facts state is the story's background («il
    # padre, morto nel 2015, …»), never the event
    latest = max([int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", text)] or [0])
    for rx in table.get("background_deaths") or []:
        text = re.sub(rx, lambda m: " " if int(m.group("year")) < latest else m.group(0), text, flags=re.I)
    seen: List[str] = []
    for e in table.get("events") or []:
        if any(re.search(rx, text, re.I) for rx in e.get("patterns") or []):
            seen.append(str(e.get("event") or ""))
    for e in table.get("events") or []:
        name = str(e.get("event") or "")
        if name in seen and not set(e.get("unless") or []) & set(seen):
            return name
    return ""


def offence_family(label: str) -> str:
    """v79.10.3 (C2): the table's family of a c.p. article («575 c.p.» -> «omicidio»); "" for none or another code."""
    number, _, code = str(label or "").partition(" ")
    if code.strip() != "c.p.":
        return ""
    return next((f for f, arts in (event_table().get("families") or {}).items() if number in {str(x) for x in arts}),
                "")


def event_mismatch(label: str, event: str) -> bool:
    """v79.10.3 (C2): the article's family is among the event's «unfit» ones -- with a death, no lesioni (590 …); with
    injuries and no death, no omicidio (575, 589 …). An article in no family, or no event, never mismatches."""
    if not event:
        return False
    family = offence_family(label)
    entry = next((e for e in event_table().get("events") or [] if e.get("event") == event), {})
    return bool(family) and family in (entry.get("unfit") or [])


def _in_range(label: str, code: str, span: Sequence[int]) -> bool:
    from .plan_then_write import _code_key
    parts = str(label or "").split(" ", 1)
    m = re.match(r"\d+", parts[0]) if len(parts) == 2 else None
    return bool(m) and _code_key(parts[1]) == _code_key(code) and span[0] <= int(m.group(0)) <= span[1]


def matter_of(label: str) -> Dict[str, Any]:
    """v75: the claim's matter by its article's place in the code (remedies.json «matters»); {} when none holds it."""
    return next((m for m in remedies().get("matters") or [] if _in_range(label, m["code"], m["range"])), {})


def _fits(r: Dict[str, Any], claim: str, demand: str) -> bool:
    """v79.10.2 (B1): an eccezioni entry answers this claim or this demand -- always, without «on» and «demand»."""
    on, want = list(r.get("on") or []), str(r.get("demand") or "")
    if not on and not want:
        return True
    return bool(claim and any(same_article(claim, x) or claim == x for x in on)) or bool(want and want == demand)


def resisting_eccezioni(claim: str, case: str) -> List[Tuple[Dict[str, Any], str]]:
    """v79.10.2 (B1): [(entry, its own article or "")] -- the convenuto's eccezioni: the claim's matter's entries that fit
    it (_fits), and every matter's reduction («kind»: «riduzione») the claim or the case's demand (intake.demand_kind)
    calls for, the reductions first; each entry once."""
    from . import intake as intake_mod
    demand = intake_mod.demand_kind(case)
    own = [r for r in (matter_of(claim).get("eccezioni") or []) if _fits(r, claim, demand)] if claim else []
    reductions = [r for m in remedies().get("matters") or [] for r in m.get("eccezioni") or []
                  if r.get("kind") == "riduzione" and (r.get("on") or r.get("demand")) and _fits(r, claim, demand)]
    out: List[Tuple[Dict[str, Any], str]] = []
    for r in reductions + [r for r in own if r.get("kind") != "riduzione"]:
        if not any(r is x for x, _ in out):
            out.append((r, str(r.get("article") or "") if r.get("kind") == "riduzione" else ""))
    return out


MODEL_ARGUED = ("contest_excluded", "contest_open", "contest_integrated", "requalify")    # v74: the model argues these
CODE_ARGUMENTS = {   # my drafts (v74): the argument of a thesis that contests no element, written from the record
    "contenimento": {"pro": "Non nega gli elementi che i fatti integrano ({ids}): sposta la difesa sulla "
                            "qualificazione e sulla misura della pena",
                     "contro": "Non evita la condanna per l'art. {art}: ne contiene l'esito",
                     "argomento": "Tutti gli elementi dell'art. {art} risultano integrati ({ids})"},
    # v79.7 (my drafts): the containment line when the charge is not fully integrato -- the subordinate line
    "contenimento_open": {"pro": "Non dipende dall'esito delle contestazioni sugli elementi: resta utile se "
                                 "{what} risulta provato",
                          "contro": "Presuppone l'accertamento della responsabilità per {what}",
                          "argomento": "Se gli elementi contestati risultano provati, la difesa si sposta sulla "
                                       "qualificazione e sulla misura della pena"},
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


TRANSAZIONE_ARGUMENT = {   # my drafts (v79.8, B3)
    "pro": "Non dipende dall'esito del giudizio sulla domanda",
    "contro": "Richiede reciproche concessioni (art. 1965 c.c.)",
    "argomento": "La transazione chiude la lite sulla domanda ({art}) con reciproche concessioni ed evita il rischio "
                 "e i tempi del giudizio"}


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
    who = "La controparte" if not criminal(posture) else (       # v79 (E2): «l'accusa» in a criminal matter only
        "La difesa dell'imputato" if posture.get("role") in VICTIM_ROLES else "L'accusa")
    art = c.get("article") or ""
    texts = CODE_ARGUMENTS.get(c["kind"]) or {}
    if c["kind"] == "contest_fit":              # v79.13.0: the applicability judgment's «no» on a kept norm
        from .applicability import FIT_CONTEST_ARGUMENT, clause, contest_pro
        return {"pro": _sentence_join([contest_pro(c)]),                    # v79.14.2 (A): the facts it quotes
                "contro": _sentence_join([FIT_CONTEST_ARGUMENT["contro"].format(who=who)]),
                "argomento": _sentence_join([FIT_CONTEST_ARGUMENT["argomento"].format(
                    perche=clause(c.get("perche")))]),
                "realismo": MEDIO}
    if c["kind"] == "transazione":              # v79.8 (B3): the convenuto's way out of the claim
        return {"pro": _sentence_join([TRANSAZIONE_ARGUMENT["pro"]]),
                "contro": _sentence_join([TRANSAZIONE_ARGUMENT["contro"]]),
                "argomento": _sentence_join([TRANSAZIONE_ARGUMENT["argomento"].format(
                    art=f"art. {art}" if art else "la domanda")]), "realismo": MEDIO}
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
        if not (ctx.get("assessment") or {}).get("all_integrated"):     # v79.7 (item 2): offered on every charge
            texts = CODE_ARGUMENTS["contenimento_open"]
        goal = str((posture.get("entry") or {}).get("goal") or "")
        riti = riti_names(ctx.get("riti") or [], 2)          # v79.10.3 (C1): the admitted ones only
        att = [x for x, r in circ.items() if r.get("kind") == "attenuante"
               and any(q.get("status") == PROVATO for q in _live(r))]
        what = f"l'art. {art}" if art else "l'addebito"
        arg = [texts["argomento"].format(art=art, ids=ids, what=what)]
        arg += [CODE_ARGUMENTS["goal"].format(goal=goal)] if goal else []
        arg += [CODE_ARGUMENTS["riti"].format(riti=riti)] if riti else []
        arg += [CODE_ARGUMENTS["attenuanti"].format(items=", ".join(f"art. {x}" for x in att))] if att else []
        return {"pro": _sentence_join([texts["pro"].format(ids=ids, what=what)]),
                "contro": _sentence_join([texts["contro"].format(art=art, what=what)]),
                "argomento": _sentence_join(arg),
                "realismo": ALTO if (ctx.get("assessment") or {}).get("all_integrated") else MEDIO}
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


def family_context(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """v79.9 (C1): the case's family context (family_gate.context), once per ctx. v79.10 (A5, review): with the
    retrieval gate's overrides (ctx["family_overrides"], prepare: state["family_gate"]["overrides"]) -- a candidate on a
    document the retrieval admitted through a door is not out of family (family_gate.carried_override)."""
    if "family_ctx" not in ctx:
        from . import family_gate
        ctx["family_ctx"] = family_gate.context(ctx.get("case") or "", ctx.get("family_words") or "",
                                                ctx.get("institutions") or [])
        ctx["family_ctx"]["overrides"] = list(ctx.get("family_overrides") or [])     # v79.10 (A5, review)
    return ctx["family_ctx"]


def family_gate_ok(label: str, fam: Dict[str, Any], named: Sequence[str] = ()) -> bool:
    from . import family_gate
    return family_gate.label_admitted(label, fam, [x for x in named if x])[0]


def family_of_label(label: str) -> str:
    from . import family_gate
    return family_gate.classify(str(label).partition(" ")[2])[0]


def label_code(label: str) -> str:
    """v79.9 (A3): an article label's code («582 c.p.» -> «c.p.»; «2 l. 604/1966» -> «l. 604/1966»)."""
    return str(label or "").strip().partition(" ")[2].strip().lower()


def differing_elements(charge: Dict[str, Any], alt: Dict[str, Any]) -> List[str]:
    """v79.9 (A3): the elements that tell the alternative from the charge -- each live element of either record whose
    own nouns the other record's elements do not hold (the charge's distinctive element first)."""
    from . import issue_records
    nouns = lambda rec: set().union(*[issue_records.element_nouns(r.get("text") or "") for r in _live(rec)] or [set()])
    mine, theirs = nouns(charge), nouns(alt)
    out = [r.get("text") or "" for r in _live(charge) if r.get("distinctive")]
    for rec, other in ((alt, mine), (charge, theirs)):
        for r in _live(rec):
            own = issue_records.element_nouns(r.get("text") or "")
            if own and not own <= other and (r.get("text") or "") not in out:
                out.append(r.get("text") or "")
    return [x for x in out if x]


def names_element(argument: str, elements: Sequence[str]) -> bool:
    """v79.9 (A3): the model's argument names one of the differing elements -- shares one of its own nouns."""
    from . import issue_records
    said = issue_records.element_nouns(argument or "")
    return any(said & issue_records.element_nouns(e) for e in elements or [])


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
    ctx["module_proven"] = False        # v79.12.5 (item 11, review): set below when a module line rests on proven elements

    def add(kind: str, tesi: str, contesta=(), poggia_su=(), label: str = SUSTAINABLE, **extra):
        rule = extra.pop("admitted_by", "") or ADMITTED_BY.get(kind, "")     # v79.8 (B1): the rule that admits it
        out.append({"kind": kind, "tesi": tesi, "contesta": list(contesta), "poggia_su": list(poggia_su),
                    "label": label, "admitted_by": rule, **extra})
    fatt = lambda x: issue_records.counted(x, str((records.get(x) or {}).get("rubrica") or ""))   # v74: fattispecie only
    named = lambda rec: bool(rec.get("named_due") or rec.get("decomposed_for") == "nominata dall'utente")      # v76
    role = str(posture.get("role") or "")
    victim, civil_role = role in VICTIM_ROLES, role in CIVIL_ROLES
    known_role = role in KNOWN_ROLES                # v79.6 (item 5): an unknown role gets neither side's wording
    offender = known_role and not victim and not civil_role        # v75: the imputato's set
    threats = [charge] if (records.get(charge) or {}).get("side") == "threat" and fatt(charge) \
        and not frame_excluded(records.get(charge)) else []
    threats += [x for x, r in records.items() if r.get("side") == "threat" and x not in threats and fatt(x)
                and not frame_excluded(r)]                  # v79.14.2 (B): the frame's side of the law only
    trace: Dict[str, Any] = {"role": role, "charge": charge, "threats": list(threats), "threats_from": "side"}
    if not threats and role in ACCUSED:     # v79.7 (item 2): no record marked «threat» -- the charged ones, else the
        threats = [x for x in _charged(records) if fatt(x) and _live(records[x])]            # assessment's article
        if not threats and charge in records and fatt(charge) and _live(records[charge]):
            threats = [charge]
        trace.update(threats=list(threats), threats_from="charged" if threats else "",
                     no_threat=_no_threat_why(records, charge, fatt))
    ctx["candidates_trace"] = trace
    contests = {"excluded": [], "open": [], "integrated": []}
    in_order: List[Tuple[str, Dict[str, Any]]] = []
    expired_out: List[str] = []         # v79.12.4 (item 7): the claims refused on a DECORSO term
    for label in threats:
        for r in _live(records[label]):
            if not r.get("id"):
                continue
            kind = {NON_RISULTA: "excluded", DA_PROVARE: "open", PROVATO: "integrated"}.get(r.get("status"), "open")
            contests[kind].append((label, r))
            in_order.append((label, r))
    distinctive = next((r["id"] for r in _live(records.get(charge) or {}) if r.get("distinctive") and r.get("id")), "")
    known = elements_by_id(records, circ)
    event = event_of(ctx.get("certain_text") or "") if offender else ""      # v79.10.3 (C2): the certain facts' event
    if event:
        trace["event"] = event
    from . import penal_lines
    # v79.12.7 (item 5): the causation lines' realism -- capped on an ordinary consequence with no trigger fact
    cap = penal_lines.causation_cap(case_facts_text(ctx)) if offender else {"capped": False}
    trace["causation_cap"] = cap

    def contest(label: str, r: Dict[str, Any]) -> None:
        kind = {NON_RISULTA: "excluded", PROVATO: "integrated"}.get(r.get("status"), "open")
        if r.get("jurisprudential"):        # v79.12.7 (item 1): a jurisprudential element -- sostenibile, its source
            add("contest_open", CANDIDATE_TEXTS["jurisprudential"].format(           # named, never minoritaria
                el=_short(r.get("text"), 80), id=r["id"], art=label, source=r.get("source") or ""),
                contesta=[r["id"]], label=SUSTAINABLE, jurisprudential=True, sources=list(r.get("sources") or []),
                admitted_by=JURISPRUDENTIAL_ADMITTED)
            return
        causal = kind == "open" and bool(r.get("event_narrated"))     # v79.12.7 (item 5): the causal link contested
        if kind == "open" and issue_records.contested(r):       # v79.14.3 (item 1): «l'accusa deve provare «…»»
            add("contest_open", CANDIDATE_TEXTS["contested"].format(
                who=contested_who(posture), el=_short(r.get("text"), 80), id=r["id"], art=label,
                di_parte=issue_records.of_party(r.get("contested_by"))),
                contesta=[r["id"]], label=SUSTAINABLE, contested=True, contested_by=str(r.get("contested_by") or ""),
                **({"causation": True, "causation_capped": bool(cap.get("capped"))} if causal else {}))
            return
        add(f"contest_{kind}", CANDIDATE_TEXTS[kind].format(el=_short(r.get("text"), 80), id=r["id"], art=label),
            contesta=[r["id"]], label=MINORITY if kind == "integrated" else SUSTAINABLE,
            **({"causation": True, "causation_capped": bool(cap.get("capped"))} if causal else {}))
    # v79.7 (item 2): the imputato's contests in the elements' order (an «integrato» one labelled minoritaria);
    # another role's as before -- the excluded and open ones first, the integrated ones last. v79.14.3 (item 1): for a
    # defence client (the accused, the convenuto / resistente) the contested elements are the primary theses -- first
    # (the accused's), first after the excluded ones (another defending role's); the rest in their order
    defending = offender or role in DEFENDING_ROLES
    first = lambda pairs: [x for x in pairs if issue_records.contested(x[1])] + \
        [x for x in pairs if not issue_records.contested(x[1])]          # noqa: E731
    for label, r in (first(in_order) if offender else contests["excluded"] + (
            first(contests["open"]) if defending else contests["open"])):
        contest(label, r)
    if victim:                  # v75: the victim's set -- never the offender's circumstances
        _victim_candidates(ctx, add, fatt)
    for label, rec in records.items():
        if not offender:
            break
        if label == charge or label in threats or not (rec.get("alternative") or rec.get("alternative_facts")
                                                       or rec.get("ranked_candidate")):     # v79 (D2)
            continue
        if issue_records.contradicted(rec):     # v75b: the record decides, never the read's «sì / no»; v79.7 (item
            continue                            # 2): an alternative the first read listed, its elements open or proven
        if not issue_records.assertable(rec) and not rec.get("alternative"):
            continue
        if not fatt(label) or not issue_records.decomposed(rec):    # v74: a decomposed, sound fattispecie record only
            continue
        base = charge if charge in records else (threats[0] if threats else charge)
        if label_code(label) != label_code(base):       # v79.9 (A3): into an offence of the charge's own code only
            trace.setdefault("requalify_out_of_code", []).append(label)
            continue
        if event_mismatch(label, event):                # v79.10.3 (C2): never an offence the event does not fit
            trace.setdefault("requalify_refused_event", []).append(label)
            continue
        differs = differing_elements(records.get(base) or {}, rec)
        if not differs:                                 # v79.9 (A3): nothing to name -- the same fattispecie
            trace.setdefault("requalify_no_difference", []).append(label)
            continue
        # v79.12.7 (item 4): a listed riqualificazione (584 -> 589/590) names the element that differs and the facts
        # bearing on it; with no fact it is not offered
        rule = penal_lines.requalify_rule(base, label)
        facts = penal_lines.requalify_facts(rule, ctx.get("sentences") or []) if rule else []
        if rule and not facts:
            trace.setdefault("requalify_no_facts", []).append(label)
            continue
        if rule:
            differs = [rule["charge_element"], rule["alternative_element"]] + [x for x in differs if x not in (
                rule["charge_element"], rule["alternative_element"])]
        proven = [r["id"] for r in _live(rec) if r.get("status") == PROVATO and r.get("id")]
        contesta = [distinctive] if distinctive else []
        minority = any((known.get(x) or {}).get("status") == PROVATO for x in contesta)
        text = penal_lines.requalify_text(rule, label, _rubrica_tail(rec), base, facts) if rule else \
            CANDIDATE_TEXTS["requalify"].format(art=label, rubrica=_rubrica_tail(rec))
        add("requalify", text, contesta=contesta,
            poggia_su=proven, label=MINORITY if minority else SUSTAINABLE, article=label, charge=base,
            differs=differs, **({"facts": facts, "requalify_rule": rule.get("id")} if rule else {}))
    for label, rec in records.items():
        if not offender:
            break
        if issue_records.justification(rec.get("comma_of") or label):
            if rec.get("comma_of") and not touched(rec):    # v79.9 (D5): a later comma, a thesis of its own only when
                continue                                    # a fact touches it
            proven = [r["id"] for r in _live(rec) if r.get("status") == PROVATO and r.get("id")]
            ids = proven or (_ids(rec) if by_concept(rec) else [])     # v79.7 (item 3): named by the facts
            if ids and not any(r.get("status") == NON_RISULTA for r in _live(rec)):
                add("scriminante", CANDIDATE_TEXTS["scriminante"].format(art=label, rubrica=_rubrica_tail(rec)),
                    poggia_su=ids, article=label)
    for label, rec in records.items():     # v79 (D1): a general-part defence (causation, fault, excess, attempt,
        if not offender:                    # aberratio, tenuità) with its elements proven and none excluded
            break
        if never_defence(label):            # v79.12.4 (item 5): 40, 43 -- their elements proven are the accusation's
            trace.setdefault("never_defence", []).append(label)
            continue
        if issue_records.defence_article(label) and not issue_records.justification(label):
            proven = [r["id"] for r in _live(rec) if r.get("status") == PROVATO and r.get("id")]
            ids = proven or (_ids(rec) if by_concept(rec) else [])     # v79.7 (item 3): named by the facts
            if ids and not any(r.get("status") == NON_RISULTA for r in _live(rec)):
                add("defence", CANDIDATE_TEXTS["defence"].format(art=label, rubrica=_rubrica_tail(rec)),
                    poggia_su=ids, article=label,
                    **({"causation": True, "causation_capped": bool(cap.get("capped"))}     # v79.12.7 (item 5)
                       if penal_lines.causation_line(label) else {}))
    for label, rec in circ.items():
        if rec.get("kind") != "attenuante" or not offender:      # v75: the offender's attenuanti, the offender's only
            continue
        proven = [r["id"] for r in _live(rec) if r.get("status") == PROVATO and r.get("id")]
        if proven and not any(r.get("status") == NON_RISULTA for r in _live(rec)):
            add("attenuante", CANDIDATE_TEXTS["attenuante"].format(art=label, rubrica=_rubrica_tail(rec)),
                poggia_su=proven, article=label)
    if known_role and not victim and (role == "convenuto" or role not in ACCUSED):
        own_kind = "riconvenzionale" if role == "convenuto" else "claim"      # v75: the convenuto's own claim
        modules = active_modules(ctx)                       # v79.8 (B2): the first read's institution
        ctx["claim_modules"] = [m["id"] for m in modules]
        ctx["module_stages"] = {m["id"]: m.get("stage") or "" for m in modules}           # v79.9 (B1)
        consumer = professional_counterparty(ctx)                                          # v79.9 (B3)
        families = read_families(ctx)                       # v79.10.1 (A2): the first read's institution families
        ctx["claim_families"] = families
        admitted, refused = [], {}
        for label, rec in records.items():
            if rec.get("side") == "threat" or (own_kind == "riconvenzionale" and rec.get("alternative")):
                continue
            live = _live(rec)
            proven = [r["id"] for r in live if r.get("status") == PROVATO and r.get("id")]
            open_ = [r for r in live if r.get("status") == DA_PROVARE and r.get("id")]
            if any(r.get("status") == NON_RISULTA for r in live) or not fatt(label):
                continue
            rule, group, rank = family_admission(label, modules, consumer, families,     # v79.8 (B1); v79.10.1 (A2)
                                                 sale_contract(str(ctx.get("case") or "")))      # v79.12.5 (10)
            if not rule:
                refused[label] = group
                continue
            admitted.append((rank, label, rec, rule, group, proven, open_))
        admitted.sort(key=lambda x: x[0])
        # v79.12.4 (item 7): the action the computed terms mark DECORSO -- a decadenza refuses the claim (and a module
        # line so refused gives no proven line: general obligations law follows), a prescrizione (the counterparty must
        # raise it, art. 2938 c.c.) keeps it, minoritaria
        # (review) a decadenza refuses only when the facts settle it (barring_terms' «firm»: not a denuncia the case
        # may just not mention, nor one the statute excuses -- the defect acknowledged or concealed, artt. 1495 co. 2,
        # 1667 co. 2 c.c.); else the claim is kept, minoritaria. A module line barred by any DECORSO term gives no
        # proven line: general obligations law follows -- minoritaria under the module's terms (the general remedies
        # for the same defect follow the special terms), never sustainable
        from . import fact_signals
        barring = {x[1]: fact_signals.barring_terms(x[1], ctx.get("terms")) for x in admitted}
        excused = bool(_DEFECT_EXCUSED_RE.search(str(ctx.get("case") or "") + " " + str(ctx.get("certain_text") or "")))
        decaduto = {x for x, e in barring.items() if any(
            t["kind"] != "prescrizione" and t.get("firm", True) and not (excused and t["id"] in EXCUSABLE_TERMS)
            for t in e)}
        module_proven = any(g == "module" and p and not barring[x] for _, x, _, _, g, p, _ in admitted)
        ctx["module_proven"] = module_proven        # v79.12.5 (item 11, review): main_outside reads it
        inherited = [] if module_proven else list({t["id"]: {**t, "inherited": True} for _, x, _, _, g, p, _ in admitted
                                                  if g == "module" and p for t in barring[x]}.values())
        claims = []
        ctx["claims_expired_minority"] = []
        for _, label, rec, rule, group, proven, open_ in admitted:
            if group == "general" and module_proven:     # B2: general obligations law, subordinate to the module
                refused[label] = CLAIM_REFUSED["subordinate"]
                continue
            expired = barring[label] or (inherited if group == "general" else [])
            if label in decaduto:
                refused[label] = CLAIM_REFUSED["expired"].format(terms=", ".join(
                    f"{e['id']}, {e['ref']}" if e.get("ref") else e["id"] for e in expired))
                expired_out.append(label)
                continue
            standing = MINORITY if expired else SUSTAINABLE
            if expired:
                ctx["claims_expired_minority"].append(label)
            if proven:
                add(own_kind, CANDIDATE_TEXTS[own_kind].format(art=label, rubrica=_rubrica_tail(rec)),
                    poggia_su=proven, article=label, admitted_by=rule, label=standing,
                    **({"expired": [e["id"] for e in expired]} if expired else {}))
                claims.append((label, proven, expired))
            elif (role == "attore" or named(rec)) and open_ and not rec.get("ranked_candidate"):   # v75b; v76: named
                add(own_kind, CANDIDATE_TEXTS[f"{own_kind}_open"].format(      # article always, when decomposed
                    art=label, rubrica=_rubrica_tail(rec),
                    open=", ".join(f"«{_short(r.get('text'), 60)}» ({r['id']})" for r in open_[:3])),
                    poggia_su=[r["id"] for r in open_], article=label, admitted_by=rule, label=standing,
                    **({"expired": [e["id"] for e in expired]} if expired else {}))
                claims.append((label, [], expired))
        ctx["claims_expired"] = expired_out
        if expired_out:
            _log("claims_refused_expired", {x: refused[x] for x in expired_out})
        ctx["claims_refused"] = refused
        ctx["claims_refused_family"] = sorted(x for x, why in refused.items() if why.startswith(FAMILY_REFUSED_HEAD))
        # v79.8.5: does any article of the client's confer an action, whatever its elements' status? v79.10.1 (A2):
        # admitted as the claims are -- an article outside the institution or its family confers none here
        ctx["claims_any_admitted"] = any(x[1] not in expired_out for x in admitted) or any(
            family_admission(x, modules, consumer, families, sale_contract(str(ctx.get("case") or "")))[0]
            for x, r in records.items()
            if r.get("side") != "threat" and fatt(x) and x not in expired_out)      # v79.12.4 (item 7)
        if refused:
            _log("claims_refused", refused)
        if ctx["claims_refused_family"]:
            _log("claims_refused_family", {"families": families, "articles": {
                x: article_family(x) for x in ctx["claims_refused_family"]}})
        if role == "attore":    # v75: the remedies of the claim's matter
            for label, proven, expired in claims[:1]:
                matter = matter_of(label)
                for r in (matter.get("remedies") or []):
                    # v79.12.4 (item 7, review): a claim kept minoritaria on a DECORSO term -- its remedies too (a
                    # remedy's «label» is its own text: the standing rides on «minority», is_minority)
                    add("remedy", CANDIDATE_TEXTS["remedy"].format(**r), poggia_su=proven, article=label,
                        admitted_by=f"remedies.json: {matter.get('id')}", **r,
                        **({"minority": True, "expired": [e["id"] for e in expired]} if expired else {}))
    if role == "convenuto":     # v75: the eccezioni of the claim's matter (the claim against the client)
        claim = next((x for x in threats if matter_of(x)), "")
        # v79.6: the eccezione di prescrizione stays (E4: claims only -- a prescription article is never a claim,
        # issue_records.counted). v79.10.2 (B1): the party resisting a demand -- its reductions first (1384 for a
        # penalty: the claim one of «on», or the case's demand the entry's), with no claim record too; an entry with
        # «on» or «demand» only then; a reduction's line is on its own article, the forecast's (reduction_assessment)
        for r, own in resisting_eccezioni(claim, str(ctx.get("case") or "")):
            add("eccezione", CANDIDATE_TEXTS["riduzione" if own else "eccezione"].format(**r), article=own or claim,
                claim=claim, reduction=bool(own), **{k: r[k] for k in ("label", "articles", "requires") if k in r})
        add("transazione", CANDIDATE_TEXTS["transazione"], article=claim or charge)    # v79.8 (B3): its set
    ctx.pop("no_action", None)
    ctx.pop("no_action_kind", None)
    if known_role and not victim and role not in ACCUSED and not out and expired_out:
        # v79.12.4 (item 7, review): the norm confers the action, the term has run -- said so, never «nessuna
        # attribuisce un'azione»
        ctx["no_action"] = NO_ACTION_EXPIRED_LINE.format(istituto=action_subject(ctx), terms=expired_terms_text(
            [t for x in expired_out for t in fact_signals_barring(x, ctx)]))
        ctx["no_action_kind"] = "expired"                   # v79.12.5 (item 2)
        _log("no_action", {"line": ctx["no_action"], "refused": sorted(ctx.get("claims_refused") or {}),
                           "expired": list(expired_out)})
    elif known_role and not victim and role not in ACCUSED and not out and not ctx.get("claims_any_admitted"):
        # v79.8.5 (B1): no norm confers an action. v79.10.1 (A1): never a candidate, never «Linea principale» -- the
        # method steps aside to the prose answer as 79.6 did (no candidate: the placeholder, method_refusal), and the
        # sentence is said once, in v76's note (a norm that does, with an element excluded, keeps v76's own why)
        ctx["no_action"] = NO_ACTION_LINE.format(istituto=action_subject(ctx))
        ctx["no_action_kind"] = "none"                      # v79.12.5 (item 2)
        _log("no_action", {"line": ctx["no_action"], "refused": sorted(ctx.get("claims_refused") or {})})
    for e in ctx.get("fit_contests") or []:     # v79.13.0: a charged or named norm the judgment says does not govern
        add("contest_fit", fit_contest_tesi(e), article=e["article"], passage=e.get("passage") or "",   # the facts
            perche=e.get("perche_printed") or "", fit_facts=list(e.get("facts") or []),
            fact_quotes=list(e.get("fact_quotes") or []))                  # v79.14.2 (A)
    if not offender:
        for label, r in contests["integrated"]:
            contest(label, r)
    else:                       # v79.7 (item 2): the containment line, always the imputato's and always last -- the
        main = posture_fallback(posture) or CONTAINMENT_LINE        # enumeration is never empty
        add("contenimento", main[0].upper() + main[1:],         # resting on the charge's elements when they are
            poggia_su=_ids(records.get(threats[0] if threats else charge) or {}) if a.get("all_integrated") else [],
            fallback=True)                                          # all integrato; else on none (a subordinate line)
    if _family_key(posture) not in ("penale", ""):       # v76: a penal line never under a civil intake rito
        dropped = [c["kind"] for c in out if c["kind"] in PENAL_KINDS]
        out[:] = [c for c in out if c["kind"] not in PENAL_KINDS]
        if dropped:
            _log("penal_candidates_dropped", {"rito": _family_key(posture), "kinds": dropped})
    # v79.9 (C1): every candidate's article inside the case's families (family_gate) -- a riqualificazione, a ranked
    # candidate, a claim from a document the case's families do not admit is no candidate; the charge's own are kept
    fam = family_context(ctx)
    gated = [c for c in out if c.get("article") and not family_gate_ok(c["article"], fam, threats + [charge])]
    if gated:
        ctx["family_dropped"] = [{"article": c["article"], "kind": c["kind"],
                                  "family": family_of_label(c["article"])} for c in gated]
        out[:] = [c for c in out if c not in gated]
        _log("candidates_out_of_family", ctx["family_dropped"])
    if (ctx.get("frame_binding") or {}).get("applied"):     # v79.14.2 (B): no candidate on an excluded article
        subject = candidate_subjects(records)
        barred = [c for c in out if not c.get("fallback") and any(frame_excluded(records.get(x)) or (
            x not in records and not binding_allows(ctx["frame_binding"]["mode"], x)) for x in subject(c))]
        if barred:
            binding_excluded(ctx, [x for c in barred for x in subject(c)])
            out[:] = [c for c in out if c not in barred]
            _log("frame_binding_candidates", {"mode": ctx["frame_binding"]["mode"],
                                              "dropped": [{"kind": c["kind"], "articles": subject(c)} for c in barred]})
    keep = [c for c in out if c.get("fallback")]
    rest = [c for c in out if not c.get("fallback")]
    if offender:                # v79.7: past the cap, the minority contests go first, the containment line never
        rest = sorted(rest, key=lambda c: CAP_PRIORITY.get(c["kind"], 3))
    capped = keep + rest[:max(0, CANDIDATES_MAX - len(keep))]
    capped = [c for c in out if c in capped]
    if not capped and offender:
        _log("candidates_empty", trace)
    for i, c in enumerate(capped, 1):
        c["id"] = f"T{i}"
        c["argued"] = "model" if c["kind"] in MODEL_ARGUED else "code"      # v74: the model argues contests only
        if c["argued"] == "code":
            c["code_argument"] = code_argument(c, ctx)
            if c.get("causation_capped"):       # v79.12.7 (item 5): the ordinary consequence -- the nexus holds
                c["code_argument"] = {**c["code_argument"], "contro": _sentence_join([penal_lines.CAUSATION_CONTRO]),
                                      "realismo": BASSO}
    return capped


JURISPRUDENTIAL_ADMITTED = "elemento di fonte giurisprudenziale dell'addebito"       # v79.12.7 (item 1)
DEFENDING_ROLES = ("convenuto", "resistente")       # v79.14.3 (item 1): a civil client resisting the claim
CONTESTED_WHO = ("L'accusa", "La controparte")      # my drafts: who must prove a contested element


def contested_who(posture: Dict[str, Any]) -> str:
    """v79.14.3 (item 1): who must prove a contested element -- the accusation in a penal matter, else the other side."""
    return CONTESTED_WHO[0] if criminal(posture or {}) else CONTESTED_WHO[1]


def contested_counts(ctx: Dict[str, Any]) -> Dict[str, int]:
    """v79.14.3 (item 1): elements_contestato (the printed records' live contested elements, the excluded aside) and
    theses_from_contestato (the plan's candidates on one)."""
    from . import issue_records
    records = ctx.get("records") or {}
    n = sum(1 for x, rec in records.items() if isinstance(rec, dict) and not frame_excluded(rec)
            and not rec.get("ranked_candidate") and not rec.get("unprinted")
            for r in _live(rec) if issue_records.contested(r))
    return {"elements_contestato": n,
            "theses_from_contestato": sum(1 for c in ctx.get("candidates") or [] if c.get("contested"))}


def _fit_on() -> bool:
    """v79.13.0 (review): APPLICABILITY_FIT -- a «dipende» mark read only with the flag on."""
    from .applicability import enabled
    return enabled()


def fit_contest_tesi(e: Dict[str, Any]) -> str:
    """v79.13.0: the thesis line of a kept norm the applicability judgment says does not govern the facts."""
    from .applicability import FIT_CONTEST_TESI, clause
    return FIT_CONTEST_TESI.format(art=e["article"], perche=clause(e.get("perche_printed")))


def fit_kept(state: Dict[str, Any], records: Dict[str, Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    """v79.13.0: each «no» the applicability judgment gave a norm the user names or the text charges (kept), as
    ("contest", item) -- the client's line: a charged norm (the judgment's «charged», a record charged) or a threat's
    record -- or ("objection", item): the other side's against the client's own norm. item: the judgment with its
    «perche_printed» (fact ids out). With or without a method record."""
    from . import applicability
    fit = state.get("applicability") or {}
    charged = set(_charged(records)) | {str(x) for x in fit.get("charged") or []}
    out: List[Tuple[str, Dict[str, Any]]] = []
    for e in fit.get("kept_named") or []:
        label = str(e.get("article") or "")
        if not label:
            continue
        rec = records.get(label) or {}
        item = {**e, "article": label,
                "perche_printed": applicability.printable(e.get("perche"), fit.get("facts") or [])}
        out.append(("contest" if label in charged or rec.get("side") == "threat" else "objection", item))
    return out


def fit_kept_lines(state: Dict[str, Any], posture: Dict[str, Any], records: Dict[str, Dict[str, Any]]) -> List[str]:
    """v79.13.0 (review): the kept named or charged norms' «no» as the prose answer prints them -- the contest's thesis,
    or the objection with its answer; [] with the flag off."""
    from . import applicability
    if not applicability.enabled():
        return []
    out = []
    for kind, item in fit_kept(state, records):
        if kind == "contest":
            out.append(fit_contest_tesi(item).rstrip(".") + ".")
        else:
            q, _ = applicability.FIT_OBJECTION
            out.append(q.format(who=_other_side(posture or {}), art=item["article"],
                                perche=applicability.clause(item["perche_printed"])).rstrip(".") + ". "
                       + applicability.objection_answer(item).rstrip(".") + ".")      # v79.14.2 (A)
    return out


def _other_side(posture: Dict[str, Any]) -> str:
    """v79.13.0: who contests the client's own norm -- code_argument's «who»."""
    return "La controparte" if not criminal(posture) else (
        "La difesa dell'imputato" if posture.get("role") in VICTIM_ROLES else "L'accusa")


def post_fact_lines(ctx: Dict[str, Any]) -> List[str]:
    """v79.12.7 (item 3): §6's favourable elements after the fact (penal_lines.post_fact_lines)."""
    from . import penal_lines
    return penal_lines.post_fact_lines(ctx)


def case_facts_text(ctx: Dict[str, Any]) -> str:
    """v79.12.7 (item 5): the case's facts as the causation cap reads them -- every sentence of the case (a pathology
    the defence alleges is still a fact to prove), else the certain facts' text, else the case."""
    sents = " ".join(str(s.get("text") or "") for s in ctx.get("sentences") or [] if isinstance(s, dict))
    return sents or str(ctx.get("certain_text") or "") or str(ctx.get("case") or "")


# v79.8.5 (B1, the PO's words): a claimant with no norm conferring an action -- said; v79.8.6: reworded, the E2 rule
# (no «registro») absolute; v79.10.1 (A1): never the main line -- the sentence of v76's note in the prose answer, once
NO_ACTION_LINE = "Tra le norme esaminate per {istituto} nessuna attribuisce un'azione al cliente"
NO_ACTION_CORE = "nessuna attribuisce un'azione al cliente"
# v79.12.4 (item 7, review; my draft): every claim refused on a DECORSO decadenza -- the action exists, the term barred it
NO_ACTION_EXPIRED_LINE = "Tra le norme esaminate per {istituto} l'azione del cliente è preclusa dal termine decorso: {terms}"
# v79.12.4 (item 7, review): the statute excuses a late denuncia when the defect was acknowledged or concealed (artt.
# 1495 co. 2, 1667 co. 2 c.c.) -- the denuncia terms then never refuse a claim (kept, minoritaria)
_DEFECT_EXCUSED_RE = re.compile(r"\b(?:riconosc\w*|riconobb\w*|ammess[oaie]\b|ammett\w*|ammise|occult\w*|nascost\w*|"
                                r"celat\w*)", re.IGNORECASE)
EXCUSABLE_TERMS = ("denuncia_vizi_vendita", "denuncia_difformita_appalto")


def fact_signals_barring(label: str, ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """v79.12.4 (item 7): fact_signals.barring_terms of the article on the context's computed terms."""
    from . import fact_signals
    return fact_signals.barring_terms(label, ctx.get("terms"))


def expired_terms_text(entries: Sequence[Dict[str, Any]]) -> str:
    """v79.12.4 (item 7): the DECORSO terms in words, each once -- «<label> (<ref>)» from fact_signals.TERMS."""
    from . import fact_signals
    out: List[str] = []
    for e in entries or []:
        term = next((t for t in fact_signals.TERMS if t.id == e.get("id")), None)
        text = f"{term.label} ({e.get('ref') or term.ref})" if term else str(e.get("ref") or e.get("id") or "")
        if text and text not in out:
            out.append(text)
    return "; ".join(out)


def is_minority(c: Dict[str, Any]) -> bool:
    """v79.12.4 (item 7, review): the candidate is minoritaria -- its label (the code's standing), or «minority» on a
    remedy (whose «label» is its own text) of a claim kept minoritaria on a DECORSO term."""
    return c.get("label") == MINORITY or bool(c.get("minority"))
NO_ACTION_SUBJECT = "i fatti narrati"


def action_subject(ctx: Dict[str, Any]) -> str:
    """v79.8.5: the institution the no-action line names -- the first read's, else the active module's, else the
    remedies matter of the refused articles, else «i fatti narrati»."""
    inst = institution_names(ctx.get("institutions") or [])      # v79.12.1: read_institutions' dicts too
    if inst:
        return inst[0].strip()
    if ctx.get("claim_modules"):
        return str(ctx["claim_modules"][0])
    matter = next((matter_of(x).get("id") for x in ctx.get("claims_refused") or {} if matter_of(x)), "")
    return matter or NO_ACTION_SUBJECT


# v79.12.5 (item 2, the PO's words): no admitted claim keeps the ten sections -- §1 says the corpus is not enough yet,
# §6 lists the articles read (no main line, no forecast), §5 and §8 stay. The prose step-aside only when the first read
# finds no institution at all. The same line fires in §1 of any parere whose first-read institution has no module and
# nothing graded in its family (corpus_line)
CORPUS_LINE = ("Il corpus non contiene ancora informazioni sufficienti su {materia}: il parere si limita alle norme dei "
               "codici")
NO_CLAIM_HEAD = "Nessuna linea è proposta e nessuna previsione di esito: {why}."      # my draft (79.12.5, item 2)
NO_CLAIM_EXPIRED_WHY = "l'azione del cliente è preclusa dal termine decorso indicato nella sintesi"     # my draft
NO_CLAIM_READ = "Norme lette:"                                                         # my draft
NO_CLAIM_NONE_READ = "Nessuna norma è stata letta sui fatti narrati."                  # my draft
NO_CLAIM_ARTICLE_WHY = {   # my drafts: why an article read gives the client no line (CLAIM_REFUSED is never printed)
    "expired": "azione preclusa dal termine decorso",
    "refused": "non attribuisce un'azione al cliente sui fatti narrati",
    "threat": "pretesa della controparte",
    "excluded": "un elemento non risulta dai fatti narrati",
    "unproven": "nessun elemento provato dai fatti narrati"}
NO_CLAIM_QUESTION = "Se le norme lette sui fatti narrati attribuiscano a {client} un'azione in materia di {materia}."  # my draft
NO_CLAIM_OBJECTIONS = "Senza una linea proposta non vi sono obiezioni della controparte da esaminare."   # my draft
NO_CLAIM_CONCLUSION = ("Il parere non propone una linea né una previsione di esito: si limita alle norme lette sui "
                       "fatti narrati.")                                                # my draft


def no_claim(ctx: Optional[Dict[str, Any]]) -> str:
    """v79.12.5 (item 2): «expired» or «none» when the client has no admitted claim and the answer keeps the ten
    sections -- every claim barred by a DECORSO term (the honesty line naming the terms, always), or no norm conferring
    an action while the first read found an institution; "" otherwise (a claim admitted, or no institution at all: the
    prose step-aside of 79.10.1 A1)."""
    if not ctx or not ctx.get("no_action"):
        return ""
    kind = str(ctx.get("no_action_kind") or "none")
    if kind == "expired":
        return kind
    return kind if institution_names(ctx.get("institutions") or []) else ""


# v79.14.1 (item 2): §1 of a no-claim answer when the honesty line does not fire -- the institution's governing law is a
# code or in the archive: the norms were read, none gives the client a line (my draft)
NO_CLAIM_SUMMARY = ("Le norme lette su {materia} non offrono a {client} una linea da proporre sui fatti narrati: il "
                    "parere le riporta con il motivo di ciascuna, senza una tesi né una previsione.")
INSTITUTION_LAWS_FILE = os.path.join(os.path.dirname(DATA_FILE), "institution_laws.json")
_INSTITUTION_LAWS: Dict[str, Any] = {}


def institution_laws() -> List[Dict[str, Any]]:
    """v79.14.1 (item 2): data/institution_laws.json «institutions» compiled ({name, law, _rx}); [] when missing
    (logged)."""
    if "d" not in _INSTITUTION_LAWS:
        try:
            with open(INSTITUTION_LAWS_FILE, encoding="utf-8") as f:
                raw = json.load(f).get("institutions") or []
        except (OSError, ValueError) as exc:
            logger.warning("lawyer_method: %s not read: %s", INSTITUTION_LAWS_FILE, exc)
            raw = []
        _INSTITUTION_LAWS["d"] = [{**e, "_rx": [re.compile(p, re.IGNORECASE) for p in e.get("patterns") or []]}
                                  for e in raw if isinstance(e, dict) and e.get("law")]
    return _INSTITUTION_LAWS["d"]


def code_law(law: str) -> bool:
    """v79.14.1 (item 2): the law is one of the codes (family_gate.ordinary_code: c.c., c.p., c.p.c., c.p.p., c.p.a., the
    Constitution, the disposizioni di attuazione)."""
    from . import family_gate
    return bool(str(law or "").strip()) and family_gate.ordinary_code(law)


def governing_law(inst: Any, ctx: Dict[str, Any]) -> Tuple[str, str]:
    """v79.14.1 (item 2): (the institution's governing law, where it was read) -- data/institution_laws.json's entry
    naming it («tabella»), else the institution module it names («modulo»: the code when its leading action is a code
    article, else its first law), else the code of a record whose rubrica is its rubricated name («rubrica»), else a
    law or an article its own words cite («citazione»); ("", "") when none (unknown: the line never fires)."""
    from . import citations, corpus_laws
    names = institution_names([inst])
    text = " ".join(names)
    if not text:
        return "", ""
    for e in institution_laws():
        if any(rx.search(text) for rx in e["_rx"]):
            return str(e["law"]), "tabella"
    for m in institution_modules().get("modules") or []:
        if module_hit(m, text, context=False):
            # v79.14.1 (item 2, review): the module's leading action decides -- a code article first, the code; a special
            # law's first (responsabilità sanitaria: art. 7 L. 24/2017), the module's first law
            first = next((str(a).partition(" ")[2] for a in m.get("actions") or []), "")
            if first and code_law(first):
                return first, "modulo"
            if m.get("laws"):
                return str(m["laws"][0]), "modulo"
            code = next((str(a).partition(" ")[2] for a in m.get("actions") or []
                         if code_law(str(a).partition(" ")[2])), "")
            if code:
                return code, "modulo"
    rub = {_plain(x).lower() for x in names}
    for label, rec in (ctx.get("records") or {}).items():
        if isinstance(rec, dict) and _plain(rec.get("rubrica")).lower() in rub and _plain(rec.get("rubrica")):
            return str(label).partition(" ")[2], "rubrica"
    for name in names:
        refs = citations.article_refs(name)
        if refs:
            return str(refs[0][1]), "citazione"
        m = re.search(r"\b(?:l\.|d\.\s*lgs\.|d\.\s*l\.|d\.\s*p\.\s*r\.)\s*(?:n\.\s*)?\d{1,4}\s*/\s*(?:19|20)\d{2}",
                      name, re.IGNORECASE)
        if m and corpus_laws.cited(m.group(0))[0]:
            return m.group(0), "citazione"
    return "", ""


def law_in_archive(law: str, known: Optional[Sequence[str]] = None) -> Optional[bool]:
    """v79.14.1 (item 2): the law is in the archive -- one of its archive names (data/corpus_expected.json, or a
    data/special_laws.json entry of the same number and year) among the graph's primary document names; None when the
    names are unknown (the graph not read)."""
    from . import corpus_laws, special_laws
    names = corpus_laws.names() if known is None else known
    if names is None:
        return None
    if corpus_laws.present(law, names):
        return True
    key = corpus_laws.law_for(law) or law
    if key != law and corpus_laws.present(key, names):
        return True
    m = re.search(r"(\d{1,4})\s*/\s*((?:19|20)\d{2})\b", str(law or ""))
    r = None if m else re.search(r"\b((?:19|20)\d{2})\s*/\s*(\d{1,4})\b", str(law or ""))   # a regulation: «2016/679»
    number, year = (m.group(1), m.group(2)) if m else ((r.group(2), r.group(1)) if r else ("", ""))
    have = {corpus_laws.norm(n) for n in names}
    for entry in special_laws.laws():
        same = bool(number) and (str(entry.get("number") or ""), str(entry.get("year") or "")) == (number, year)
        if not same:
            continue
        if special_laws.exact(entry):           # an exact entry by its archive names only
            if have & set(special_laws._archive(entry) or []):
                return True
        elif any(entry["_code"].search(n) for n in have):       # the others by their expression on a name
            return True
    return False


def corpus_gap(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """v79.14.1 (item 2): the first first-read institution whose governing law (governing_law) is outside the codes and
    missing from the archive (law_in_archive False) -- {materia, law, from}; {} when none. Each institution's reading
    is kept in ctx["corpus_gap_log"] ({institution: {law, from, code, in_archive}}): a code institution never fires."""
    log: Dict[str, Any] = {}
    out: Dict[str, Any] = {}
    for inst in ctx.get("institutions") or []:
        name = (institution_names([inst]) or [""])[0]
        if not name:
            continue
        law, how = governing_law(inst, ctx)
        code = code_law(law)
        present = law_in_archive(law) if law and not code else None
        log[name] = {"law": law, "from": how, "code": code, "in_archive": present}
        if not out and law and not code and present is False:
            out = {"materia": name, "law": law, "from": how}
    ctx["corpus_gap_log"] = log
    return out


def corpus_line(ctx: Dict[str, Any]) -> str:
    """v79.12.5 (item 2): CORPUS_LINE (with its stop). v79.14.1 (item 2): only when a first-read institution's governing
    law is outside the codes AND missing from the archive (corpus_gap) -- never for a code institution (legittima
    difesa: art. 52 c.p.), never when the law is unknown or the graph's names are; the line names that institution.
    "" otherwise."""
    if not institution_names(ctx.get("institutions") or []):
        return ""
    gap = corpus_gap(ctx)
    # v79.19.0 (item 6): the honesty line only when the retrieval returned nothing for the pretesa -- a record read on
    # the matter (any fitted article) means the parere is built on it, never on the corpus's silence
    if gap and any(isinstance(r, dict) and not r.get("ranked_candidate") and not r.get("unprinted")
                   for r in (ctx.get("records") or {}).values()):
        ctx["corpus_line_skipped"] = "retrieval returned records for the pretesa"
        return ""
    return CORPUS_LINE.format(materia=gap["materia"]) + "." if gap else ""


_CORPUS_LINE_RE = re.compile(re.escape(CORPUS_LINE.split("{materia}")[0]) + r"(.+?)" +
                             re.escape(CORPUS_LINE.split("{materia}")[1]))


def corpus_line_on_code(text: str, ctx: Optional[Dict[str, Any]] = None) -> List[str]:
    """v79.14.1 (item 2): the honesty lines of the text whose materia is a code institution (its governing law a code:
    governing_law on the materia's own words, else on the first-read institution of that name) -- must be []."""
    out = []
    for m in _CORPUS_LINE_RE.finditer(text or ""):
        materia = m.group(1).strip()
        inst = next((x for x in (ctx or {}).get("institutions") or [] if materia in institution_names([x])), materia)
        law, _ = governing_law(inst, ctx or {})
        if code_law(law):
            out.append(materia)
    return out


def no_claim_summary(ctx: Dict[str, Any], kind: str) -> str:
    """v79.12.5 (item 2): §1 with no admitted claim -- the honesty line naming the terms (expired), else CORPUS_LINE;
    "" when the missing-laws line is said (missing_laws_line: it names the laws the corpus lacks and already says the
    parere is limited to the code -- the two never both). v79.14.1 (item 2): CORPUS_LINE only when corpus_line fires
    (the governing law outside the codes and missing from the archive); else NO_CLAIM_SUMMARY -- the norms read on the
    institution give the client no line to propose (the reasons are §6's)."""
    if kind == "expired":
        return str(ctx.get("no_action") or "").rstrip(".") + "."
    if missing_laws_line(ctx):
        return ""
    return corpus_line(ctx) or NO_CLAIM_SUMMARY.format(materia=action_subject(ctx),
                                                       client=ctx.get("client") or "la parte assistita")


def no_claim_solutions(ctx: Dict[str, Any], kind: str) -> str:
    """v79.12.5 (item 2): §6 with no admitted claim -- why no line is proposed (the no-action sentence, its one
    appearance), then the articles read, each with why it gives the client no line; no table, no main line, no forecast."""
    line = str(ctx.get("no_action") or "")
    why = NO_CLAIM_EXPIRED_WHY if kind == "expired" else (line[:1].lower() + line[1:]).rstrip(".")
    refused = ctx.get("claims_refused") or {}
    expired = set(ctx.get("claims_expired") or [])
    items = []
    for label, rec in (ctx.get("records") or {}).items():
        if not isinstance(rec, dict) or rec.get("ranked_candidate") or rec.get("unprinted"):
            continue
        live = _live(rec)
        reason = ("expired" if label in expired else "threat" if rec.get("side") == "threat" else
                  "refused" if label in refused else
                  "excluded" if any(r.get("status") == NON_RISULTA for r in live) else "unproven")
        items.append(f"- Art. {label}{_rubrica_tail(rec)}: {NO_CLAIM_ARTICLE_WHY[reason]}.")
    body = "\n".join([NO_CLAIM_READ] + items) if items else NO_CLAIM_NONE_READ
    return f"{NO_CLAIM_HEAD.format(why=why)}\n\n{body}"


# v79.8 (B1): every candidate records the rule that admits it -- a claim, a riconvenzionale and a remedy theirs
# (claim_admission, remedies.json); the others by their kind. A candidate with none is counted (must be 0).
ADMITTED_BY = {"contest_fit": "giudizio di applicabilità sul testo della norma",              # v79.13.0
               "contest_excluded": "elemento dell'addebito o della domanda", "contest_open": "elemento dell'addebito o "
               "della domanda", "contest_integrated": "elemento dell'addebito o della domanda (tesi minoritaria)",
               "requalify": "qualificazione alternativa", "scriminante": "difesa di parte generale",
               "defence": "difesa di parte generale", "attenuante": "circostanza con elementi provati",
               "contenimento": "linea dell'imputato (tabella delle posizioni)",
               "claim_offence": "persona offesa: il reato contestato", "parte_civile": "persona offesa: artt. 74-79 c.p.p.",
               "sequestro": "persona offesa: artt. 253, 321, 262 c.p.p.", "aggravante": "persona offesa: aggravante provata",
               "eccezione": "remedies.json: eccezioni della materia", "transazione": "convenuto: art. 1965 c.c."}
CLAIM_REFUSED = {   # v79.8 (B1, B2): why an article is no claim -- logged, never printed
    "limiting": "norma che limita o misura la domanda ({why}): non conferisce un'azione",
    "none": "nessuna regola la ammette: non conferisce un'azione al ruolo del cliente",
    "outside": "fuori dall'istituto della prima lettura",
    "subordinate": "norma generale, subordinata: il modulo dell'istituto dà una linea provata",
    "non_action": "{why}: non conferisce un'azione al cliente",             # v79.9 (B2)
    "consumer": "codice del consumo: nessuna controparte professionale tra i fatti certi",     # v79.9 (B3)
    "family": "fuori dalla famiglia dell'istituto della prima lettura ({family}; prima lettura: {read})",  # v79.10.1 (A2)
    "expired": "termine decorso ({terms}): l'azione è preclusa",                                       # v79.12.4 (7)
    "consumer_sale": "codice del consumo, artt. 128-135: solo per una vendita"}     # v79.12.5 (10)
FAMILY_REFUSED_HEAD = CLAIM_REFUSED["family"].split(" (")[0]
MODULES_FILE = os.path.join(os.path.dirname(DATA_FILE), "institution_modules.json")
_MODULES: Dict[str, Any] = {}


def institution_modules() -> Dict[str, Any]:
    """data/institution_modules.json (read once; {} when missing, logged)."""
    if "d" not in _MODULES:
        try:
            with open(MODULES_FILE, encoding="utf-8") as f:
                _MODULES["d"] = json.load(f)
        except (OSError, ValueError) as exc:
            logger.warning("lawyer_method: %s not read: %s", MODULES_FILE, exc)
            _MODULES["d"] = {}
    return _MODULES["d"]


def same_article(label: str, spec: str) -> bool:
    """v79.8: the record's label is the module's article -- the same number and code. v79.10 (A4, archive names): a
    special law's action («7 l. 241/1990») through its data/corpus_expected.json law: the label's code is one of the
    law's archive names («7 Legge sul Procedimento Amministrativo 2026», the label the pipeline builds for an article of
    the archive document) or the same law's citation («7 L. 241/1990») -- never by a number in a name («7 Legge 7
    agosto 1990, n. 241» is not it); an unmapped law's action matches its citation only. A code the corpus_expected
    «codes» map («29 c.p.a.»): by its sigla, or the label's code one of its archive names."""
    from . import corpus_laws
    a_num, _, a_code = str(label or "").partition(" ")
    b_num, _, b_code = str(spec or "").partition(" ")
    if a_num.lower() != b_num.lower():
        return False
    law = corpus_laws.law_for(b_code)
    if law:
        return corpus_laws.law_for(a_code) == law
    if corpus_laws.cited(b_code)[0]:                # a law no corpus_expected entry holds: its citation, as before
        return _norm_label(label) == _norm_label(spec)
    code = " ".join(b_code.split())
    if code in archive_codes() and corpus_laws.law_of_name(a_code) == code:
        return True
    return _norm_label(label) == _norm_label(spec)


def archive_codes() -> List[str]:
    """v79.10 (A4): the codes data/corpus_expected.json maps to archive names («c.p.a.»)."""
    from . import corpus_laws
    return list((corpus_laws.expected().get("codes") or {}))


def certain_text(ctx: Dict[str, Any]) -> str:
    """v79.9 (B1): the case's certain facts as one text -- prepare's (the record's sentences, certain_fact), else the
    case's sentences stated as accertato."""
    if ctx.get("certain_text") is not None:
        return str(ctx["certain_text"])
    from . import issue_records
    return " ".join(str(x.get("text") or "") for x in issue_records.sentences(ctx.get("case") or "")
                    if x.get("certainty") == issue_records.ACCERTATO and not x.get("act"))


# v79.9 (B3): a professional counterparty -- a business named as a certain fact; a private seller negates it
_PROFESSIONAL_RE = re.compile(
    r"\b(?:impres[ae]|societ[àa]|s\.?r\.?l\.?|s\.?p\.?a\.?|ditta|negozi\w*|concessionari\w*|rivendit\w*|commerciant\w*|"
    r"aziend\w*|professionist\w*|venditore\s+professional\w*|e-?commerce|catena\s+di\s+negozi|grande\s+distribuzione|"
    r"supermercat\w*|ipermercat\w*|operatore\s+(?:telefonic|di\s+telefonia)\w*|compagnia\s+(?:telefonic|aere)\w*|gestore\s+"
    r"(?:telefonic|di\s+(?:luce|gas|energia))\w*|fornitore\s+di\s+(?:energia|luce|gas|servizi)|banca|istituto\s+di\s+credito|"
    r"compagnia\s+(?:di\s+)?assicura\w*|agenzia\s+(?:di\s+)?viaggi|negozio\s+online|sito\s+(?:di\s+)?(?:vendita|e-?commerce)|"
    r"store\s+online|piattaforma\s+(?:di\s+)?(?:vendita|e-?commerce))\b", re.IGNORECASE)
_PRIVATE_RE = re.compile(r"\b(?:venditore\s+privato|venditrice\s+privata|un\s+privato|una\s+privata|da\s+(?:un\s+)?privat[oi]|"
                         r"tra\s+privati|privato\s+cittadino|non\s+(?:è|era)\s+(?:un\s+)?(?:professionista|commerciante|"
                         r"un['’]impresa)|annunci\w*\s+(?:di|tra)\s+privat\w*)\b", re.IGNORECASE)


def consumer_code(label: str) -> bool:
    """v79.9 (B3): a consumer-code article. v79.10 (A4, archive names): its code is the d.lgs. 206/2005 by the archive
    name or the citation (corpus_laws.law_for), never by an expression on a document's name."""
    from . import corpus_laws
    return corpus_laws.law_for(str(label or "").partition(" ")[2]) == CONSUMER_LAW


CONSUMER_LAW = "d.lgs. 206/2005"


# v79.12.5 (item 10): the counterparty's qualification -- «impresa, società, ditta, S.r.l., S.p.A., professionista,
# azienda» name a business wherever the case says them (a party's own description is a certain fact, whichever
# sentence carries it); a private seller's sign still excludes it
_QUALIFIER_RE = re.compile(r"\b(?:impres[ae]|societ[àa]|ditt[ae]|s\.\s?r\.\s?l\.?|srl|s\.\s?p\.\s?a\.?|spa|"
                           r"professionist[aie]|aziend[ae])(?!\w)", re.IGNORECASE)     # never «aziendale», «dittatura»


# a qualification someone only asserts or supposes («Tizio sostiene che Alfa sia un'impresa») is no certain fact
_REPORTED_RE = re.compile(r"\b(?:sostien\w*|sosteneva|afferm\w*|ritien\w*|riteneva|crede\w*|pensa\w*|forse|"
                          r"probabilmente|potrebbe|potrebbero|sembr\w*|pare|parrebbe|presunt\w*|asserit\w*|se)\b",
                          re.IGNORECASE)


# v79.12.5 (item 10, review): a business word that does not describe the counterparty -- the client's own business
# («Assisto la società Beta S.r.l.», «il mio cliente, titolare di una ditta», «sono titolare di una ditta», «la mia
# ditta»), the client's employer («lavora come dipendente presso un'azienda»), a negated qualification («non è una
# società né una ditta») -- read in the clause before the word; and a word in a question («Era una ditta?»)
_CLIENT_BUSINESS_RE = re.compile(
    r"(?:\b(?:assist[oiae]\w*|rappresent[oiae]\w*|difend[oiae]\w*|patrocin\w*)\s+(?:(?:la|il|lo|l['’]|una|un['’]?)\s*)?"
    r"(?:[\w.]+\s+){0,2}"
    r"|\b(?:sono|siamo|client[ei]\w*\s*,|assistit[oa]\s*,|io\s*,)\s*(?:(?:anche|il|la)\s+)?(?:titolar\w*|soci[oa]?|"
    r"amministrator\w*|legale\s+rappresentante|dipendent\w*|impiegat\w*)\b[^.;:!?]{0,25}"
    r"|\b(?:dipendent[ei]|impiegat[oaie]|assunt[oaie]|alle\s+dipendenze|lavor(?:a|ava|ano|o|avo|ato)\s+(?:come|presso|per|alle))"
    r"\b[^.;:!?]{0,30}"
    r"|\b(?:mi[ao]|nostr[ao])\s+(?:(?:piccol[ao]|propri[ao])\s+)?"
    r"|\b(?:non\s+(?:è|era|sia|fosse|risulta)|né)\b[^.;:!?]{0,25})$", re.IGNORECASE)


def _describes_counterparty(text: str, m: "re.Match") -> bool:
    """v79.12.5 (item 10, review): the business word m names the counterparty -- not asserted or supposed in its
    clause (_REPORTED_RE), not the client's own business, employer or a negated qualification (_CLIENT_BUSINESS_RE),
    not in a question."""
    stops = [x.end() for x in re.finditer(r"[.;:!?\n]\s", text[:m.start()])]
    before = text[stops[-1] if stops else 0:m.start()]
    after = re.search(r"[.;!?\n](?:\s|$)|$", text[m.end():])
    if after and after.group(0).startswith("?"):
        return False
    return not (_REPORTED_RE.search(before) or _CLIENT_BUSINESS_RE.search(before))


def _counterparty_word(text: str, rx: "re.Pattern") -> str:
    """v79.12.5 (item 10, review): the first word of rx in text that names the counterparty (_describes_counterparty)."""
    return next((m.group(0) for m in rx.finditer(text or "") if _describes_counterparty(text, m)), "")


def counterparty_qualifier(ctx: Dict[str, Any]) -> str:
    """v79.12.5 (item 10): the qualifier word the case names a business by (_QUALIFIER_RE) in any of its sentences --
    never one only asserted or supposed in its clause (_REPORTED_RE before it); "" with none or with a private seller's
    sign (_PRIVATE_RE) in the case. Review fix: a word that describes the client's own business, its employer, a
    negated qualification or sits in a question names no counterparty (_describes_counterparty)."""
    case = str(ctx.get("case") or "")
    if _PRIVATE_RE.search(case) or _PRIVATE_RE.search(certain_text(ctx)):
        return ""
    return _counterparty_word(case, _QUALIFIER_RE)


def professional_counterparty(ctx: Dict[str, Any]) -> bool:
    """v79.9 (B3): the consumer qualification's condition -- a business as a certain fact, no sign of a private seller.
    v79.12.5 (item 10): a qualifier word anywhere in the case is one (counterparty_qualifier); (review) either way a
    word that names the client's business, its employer or a negated one is none (_describes_counterparty)."""
    text = certain_text(ctx)
    if _counterparty_word(text, _PROFESSIONAL_RE) and not _PRIVATE_RE.search(text):
        return True
    return bool(counterparty_qualifier(ctx))


# v79.12.5 (item 10): the consumer code's scope on a consumer-professional contract -- its general titles (artt. 1-38)
# and Parte VI (141 ff.) for any such contract, the sale block (128-135 and their -bis ...) only for a sale
CONSUMER_GENERAL = ((1, 38), (141, 10 ** 6))
CONSUMER_SALE = (128, 135)
CONSUMER_SALE_ONLY = CLAIM_REFUSED["consumer_sale"]
_SALE_RE = re.compile(r"\b(?:compravendit\w*|vendit\w*|vend(?:e|uto|uta|uti|ute|ono|eva|ere)\b|acquist\w*|"
                      r"comprat[oaie]\b|compr(?:a|ò|ano)\b|"
                      # review fix: the other ways a purchase is told -- «ho ordinato online un frigorifero», «ho preso
                      # uno smartphone nel negozio», «ho pagato 900 euro … per una lavatrice, consegnata …»
                      r"ordinat[oaie]\b|ordin(?:a|ò|ano)\b|(?:ho|ha|hanno|abbiamo|avevo|aveva)\s+(?:preso|pagato|ricevuto)\b|"
                      r"consegnat[oaie]\b|difett\w*\s+di\s+conformit\w*|garanzia\s+(?:legale|di\s+conformit\w*))", re.IGNORECASE)
# v79.12.5 (item 10, review): a service or a supply is no sale of goods, though told with a sale's word («ha acquistato
# un abbonamento in palestra», «il venditore di energia»): a sale word whose object (the rest of its clause, at most
# 60 characters) is one of these is not counted
_SERVICE_RE = re.compile(r"\b(?:abbonament\w*|serviz[io]\w*|fornitur\w*|energia|luce|gas|acqua|utenz\w*|palestr\w*|"
                         r"pacchett\w*|viagg\w*|vacanz\w*|vol[oi]\b|bigliett\w*|soggiorn\w*|polizz\w*|assicura\w*|"
                         r"mutu[oi]\b|finanziament\w*|prestit\w*|conto\s+corrente|telefon\w*|internet|connession\w*|"
                         r"cors[oi]\b|iscrizion\w*|riparazion\w*|manutenzion\w*|installazion\w*|appalt\w*|lavori\b|"
                         r"ristrutturazion\w*|prestazion\w*|consulenz\w*|trasport\w*|noleggi\w*|locazion\w*|affitt\w*|"
                         r"streaming|contenut[oi]\s+digital\w*|servizi\s+digital\w*|software|licenz\w*)", re.IGNORECASE)


def sale_contract(text: str) -> bool:
    """v79.12.5 (item 10): the case is a sale (_SALE_RE: a sale's words, the buying). Review fix: also an order, a
    purchase told as «preso» or «pagato», a delivery, the conformity guarantee; never a sale word whose object is a
    service or a supply (_SERVICE_RE in the rest of its clause)."""
    text = text or ""
    for m in _SALE_RE.finditer(text):
        tail = re.split(r"[.;:!?\n]", text[m.end():m.end() + 60], maxsplit=1)[0]
        if not _SERVICE_RE.search(tail):
            return True
    return False


# v79.12.5 (item 10, review): Capo I's sale of goods runs to 135-septies (d.lgs. 170/2021); 135-octies to 135-vicies
# ter (Capo I-bis) are digital content and services -- no sale block
_DIGITAL_SUFFIX_RE = re.compile(r"(?:octies|novies|\w*decies|\w*vicies)\b")


def consumer_scope(number: Any, sale: bool) -> str:
    """v79.12.5 (item 10): "" when a consumer-code article is in scope on a consumer-professional contract, else why --
    the general titles (1-38) and 141 ff. always, the sale block (128-135-septies) only for a sale; the rest (Capo
    I-bis's digital content among them) as before (the family rules, the module)."""
    m = re.match(r"\s*(\d+)(?:\s*[-‐ ]\s*([a-z]+))?", str(number or "").lower())
    if not m:
        return ""
    n, suffix = int(m.group(1)), m.group(2) or ""
    if any(lo <= n <= hi for lo, hi in CONSUMER_GENERAL):
        return ""
    if CONSUMER_SALE[0] <= n <= CONSUMER_SALE[1] and not (n == CONSUMER_SALE[1] and _DIGITAL_SUFFIX_RE.match(suffix)):
        return "" if sale else CONSUMER_SALE_ONLY
    return ""


def module_stage(module: Dict[str, Any], ctx: Dict[str, Any]) -> str:
    """v79.9 (B1): the module's stage -- the first whose patterns a certain fact matches, else the first whose events
    the facts' events hold (consegna, fine lavori, scoperta vizio, denuncia/reclamo: what the terms read); "" with none."""
    text = certain_text(ctx)
    stages = module.get("stages") or []
    for st in stages:
        if any(re.search(p, text, re.IGNORECASE) for p in st.get("patterns") or []):
            return str(st["id"])
    events = set(ctx.get("events") or [])
    return next((str(st["id"]) for st in stages if events & set(st.get("events") or [])), "")


def staged(module: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    """v79.9 (B1): the module as the case reads it -- its stage's actions first, then the rest in its order, then the
    conditional actions a certain fact triggers; the stage and the conditions met recorded."""
    stage = module_stage(module, ctx)
    first = next((list(st.get("actions") or []) for st in module.get("stages") or [] if st["id"] == stage), [])
    text = certain_text(ctx)
    met = [c for c in module.get("conditional") or []
           if any(re.search(p, text, re.IGNORECASE) for p in c.get("patterns") or [])]
    actions = list(dict.fromkeys(first + list(module.get("actions") or []) +
                                 [a for c in met for a in c.get("actions") or []]))
    return {**module, "actions": actions, "stage": stage, "conditions_met": [c["id"] for c in met]}


def module_hit(module: Dict[str, Any], text: str, context: bool = True) -> List[str]:
    """v79.10 (A4): the words of the text that activate the module -- one of its «patterns»; with «weak_patterns», else
    at least «weak_min» (default 2) different weak patterns' words. [] when none: a module without weak patterns is
    active on any pattern, as before («il Comune» alone never activates the amministrativo module). v79.12.1: its
    «context_patterns» (the words a damage or a claim also uses: the hospital, the surgeon) only with context -- the
    first read's institution names; never the user's words alone (context False)."""
    for p in list(module.get("patterns") or []) + (list(module.get("context_patterns") or []) if context else []):
        m = re.search(p, text or "", re.IGNORECASE)
        if m:
            return [m.group(0)]
    weak = [m.group(0) for m in (re.search(p, text or "", re.IGNORECASE) for p in module.get("weak_patterns") or [])
            if m]
    need = max(1, int(module.get("weak_min") or 2))
    return weak if weak and len(weak) >= need else []


# v79.12.1: the signals that switch a module on, in their order -- the first read's institution (its name and rubricated
# name), the user's words (never a context pattern), a module article the request names (review: after the user's
# words, the brief's order -- an incidental article the user cites never replaces the module the case's words name);
# then a reviewed schema's rapporto (v79.11, ELEMENT_SCHEMAS). Never a claim sentence or a damage word alone.
MODULE_SIGNALS = ("istituto", "parole dell'utente", "articolo", "schema")


def institution_names(items: Sequence[Any]) -> List[str]:
    """v79.12.1: the first read's institutions as names -- a read_institutions dict (interactive_flow) by its name and
    its rubricated name, never its claim (the hypothesis sentence: «risarcimento del danno per le cure in ospedale»); a
    string as it is (a name)."""
    out: List[str] = []
    for x in items or []:
        parts = [x.get("name"), x.get("rubricated")] if isinstance(x, dict) else [x]
        for part in parts:
            text = " ".join(str(part or "").split())
            if text and text not in out:
                out.append(text)
    return out


_SPECIAL_ACTION_RE = re.compile(r"^(\d+(?:-[a-z]+)?)\s+(?:l\.|d\.lgs\.|d\.l\.|d\.p\.r\.)\s*(\d+)/(\d{4})$", re.IGNORECASE)
# v79.12.1 (review): a character of the citation's sentence -- a dot too, unless it ends the sentence (a dot before a
# capital or at the end): «art. 7 l. 241/1990», «art. 2 d.lgs. 23/2015», «art. 31 del d.P.R. n. 380/2001»
_CITATION_CHAR = r"(?:[^.;\n]|\.(?!\s+(?-i:[A-ZÀ-Ý])|\s*$))"


def named_module_articles(module: Dict[str, Any], text: str) -> List[str]:
    """v79.12.1: the module's action articles the text names (a code article by its citation, same_article; a special
    law's «7 l. 24/2017» by «art. 7» and «24/2017» or «n. 24 … 2017» within the sentence) -- never a «general» one
    (1218, 1453, 2043 confer an action in every matter: naming one names no institution). Review: the abbreviated
    special-law forms too («l.», «d.lgs.», «d.P.R.»: _CITATION_CHAR crosses their dots)."""
    if not text:
        return []
    from . import citations
    general = institution_modules().get("general") or []
    cited = [citations.label(r) for r in citations.article_refs(text)]
    out = []
    for spec in module.get("actions") or []:
        if any(same_article(spec, g) for g in general):
            continue
        m = _SPECIAL_ACTION_RE.match(str(spec))
        if m:
            art, num, year = (re.escape(x) for x in m.groups())
            law = rf"(?:\b{num}\s*/\s*{year}\b|\bn\.?\s*{num}\b{_CITATION_CHAR}{{0,40}}?\b{year}\b|\b{year}\s*,\s*n\.?\s*{num}\b)"
            if re.search(rf"\bart(?:icol[oi]|t?\.)?\s*{art}(?![\w-]){_CITATION_CHAR}{{0,60}}?{law}", text, re.IGNORECASE):
                out.append(str(spec))
        elif any(same_article(c, spec) for c in cited):
            out.append(str(spec))
    return out


def module_input(case: str, institutions: Sequence[Any]) -> Dict[str, Any]:
    """v79.12.1: what active_modules reads, the same at the retrieval stage (family_gate.context) and the answer stage
    (prepare): the case's text and the first read's institutions as interactive_flow.read_institutions gives them
    (dicts: name, rubricated, claim -- never names only)."""
    return {"case": case or "", "institutions": list(institutions or [])}


def _module_signal(mods: Sequence[Dict[str, Any]], ctx: Dict[str, Any]) -> Tuple[str, List[Tuple[Dict[str, Any],
                                                                                                   List[str]]]]:
    """v79.12.1: (the signal, [(module, its words)]) -- the first of MODULE_SIGNALS but the schema that finds a module (a
    module requiring a professional counterparty only with one); ("", []) when none does."""
    listed = institution_names(ctx.get("institutions") or [])
    names = " ".join(listed)
    case = str(ctx.get("case") or "")

    def by_institution(m: Dict[str, Any]) -> List[str]:     # the institutions that name it (else the words that hit)
        words = module_hit(m, names, context=True) if names else []
        return ([n for n in listed if module_hit(m, n, context=True)] or words) if words else []
    readers = (("istituto", by_institution),
               ("parole dell'utente", lambda m: module_hit(m, case, context=False) if case else []),
               ("articolo", lambda m: named_module_articles(m, case)))
    act = module_act(ctx)
    for signal, read in readers:
        hit = [(m, words) for m, words in ((m, read(m)) for m in mods) if words]
        hit = [(m, w) for m, w in hit if m.get("requires") != "professional_counterparty" or professional_counterparty(ctx)]
        # v79.14.1 (item 3): the amministrativo module only with an administrative act against the client -- a module
        # article the request names aside (the user's own naming); a refusal recorded (modules_refused_no_act)
        refused = [m for m, _ in hit if not act and signal != "articolo" and str(m.get("id")) in act_modules()]
        if refused:
            seen = ctx.setdefault("modules_refused_no_act", [])
            seen += [str(m["id"]) for m in refused if str(m["id"]) not in seen]
            hit = [(m, w) for m, w in hit if m not in refused]
        if hit:
            return signal, hit
    return "", []


def act_modules() -> List[str]:
    """v79.14.1 (item 3): the modules that need the administrative act (data/administrative_act.json «modules»)."""
    from . import family_gate
    return list(family_gate.act_table()["modules"])


def module_act(ctx: Dict[str, Any]) -> List[str]:
    """v79.14.1 (item 3): the administrative act against the client in the case's words or the first read's institution
    names (family_gate.administrative_act), once per ctx (ctx["admin_act_words"])."""
    if "admin_act_words" not in ctx:
        from . import family_gate
        ctx["admin_act_words"] = family_gate.administrative_act(family_gate.act_text(ctx.get("case") or "",
                                                                                      ctx.get("institutions") or []))
    return list(ctx["admin_act_words"])


def active_modules(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """v79.8 (B2): the institution modules of the case -- those whose patterns the first read's institutions name
    (ctx["institutions"]), else the case's own words. v79.9 (B1, B3): each staged (its stage's actions first, its
    conditional ones when met); a module requiring a professional counterparty only with one. v79.12.1: the signals in
    their order (MODULE_SIGNALS) -- the first read's institution by its names (never its claim sentence), the user's
    words without the context patterns (never a damage word alone), a module article the request names; each module
    carries «signal» and «signal_words» (the counter modules_active)."""
    mods = institution_modules().get("modules") or []
    signal, hit = _module_signal(mods, ctx)
    if hit:
        schema_module_disagree(ctx, mods, [m for m, _ in hit])          # v79.11 (6): logged only, ELEMENT_SCHEMAS
        return [{**staged(m, ctx), "signal": signal, "signal_words": list(words)} for m, words in hit]
    found = schema_modules(ctx, mods)                       # v79.11 (6): ELEMENT_SCHEMAS only; [] off
    if found:
        _log("schema_modules", {"modules": [m["id"] for m in found], "rapporti": schema_rapporti(ctx, True)})
    shadow = [m["id"] for m in schema_modules(ctx, mods, reviewed_only=False) if m not in found]
    if shadow:                                              # an unreviewed schema's module: logged, never active
        _log("schema_modules_shadow", {"modules": shadow, "rapporti": schema_rapporti(ctx)})
    rapporti = schema_rapporti(ctx, True) if found else {}
    return [{**staged(m, ctx), "signal": "schema", "signal_words": list(rapporti.get(m["id"]) or [])} for m in found]


def modules_active(modules: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    """v79.12.1: the counter modules_active -- {module id: «signal: words»} (the signal that switched each on)."""
    return {str(m["id"]): f"{m.get('signal') or '?'}: " + ", ".join(f"«{w}»" for w in m.get("signal_words") or [])
            for m in modules or []}


def schema_rapporti(ctx: Dict[str, Any], reviewed_only: bool = False) -> Dict[str, List[str]]:
    """v79.11 (6): ELEMENT_SCHEMAS -- {rapporto: [articles]} of the schemas of ctx["records"] (rec["schema"]) whose side
    is not «threat» (the client's norms); {} with the flag off or no records (family_gate.context passes none).
    reviewed_only: only the lawyer's reviewed schemas (rec["schema"]["reviewed"] True)."""
    from . import element_schemas
    if not element_schemas.enabled():
        return {}
    out: Dict[str, List[str]] = {}
    for label, rec in (ctx.get("records") or {}).items():
        if not isinstance(rec, dict) or rec.get("side") == "threat" or not isinstance(rec.get("schema"), dict):
            continue
        if reviewed_only and rec["schema"].get("reviewed") is not True:
            continue
        rapporto = rec["schema"].get("rapporto")
        if isinstance(rapporto, str) and rapporto.strip():
            out.setdefault(rapporto.strip(), []).append(label)
    return out


def schema_modules(ctx: Dict[str, Any], mods: Sequence[Dict[str, Any]],
                   reviewed_only: bool = True) -> List[Dict[str, Any]]:
    """v79.11 (6): when the patterns find no module -- the modules whose id is a rapporto of the records' REVIEWED
    schemas (schema_rapporti): an active module refuses and subordinates claims, and an unreviewed schema never
    excludes, so a generated schema's module is only logged (schema_modules_shadow, reviewed_only=False). A
    professional counterparty still required where the module requires one; [] with the flag off."""
    rapporti = schema_rapporti(ctx, reviewed_only)
    hit = [m for m in mods if m.get("id") in rapporti]
    return [m for m in hit if m.get("requires") != "professional_counterparty" or professional_counterparty(ctx)]


def schema_module_disagree(ctx: Dict[str, Any], mods: Sequence[Dict[str, Any]], hit: Sequence[Dict[str, Any]]) -> None:
    """v79.11 (6): the patterns found modules -- a schema rapporto naming another module is logged
    (schema_module_disagree), and changes nothing."""
    rapporti = schema_rapporti(ctx)
    if not rapporti:
        return
    ids = {m.get("id") for m in mods}
    found = [m.get("id") for m in hit]
    other = {r: arts for r, arts in rapporti.items() if r in ids and r not in found}
    if other:
        _log("schema_module_disagree", {"patterns": found, "schema": other})


def claim_admission(label: str, modules: Sequence[Dict[str, Any]], consumer: bool = True,
                    sale: bool = True) -> Tuple[str, str, Tuple]:
    """v79.8 (B1): (the rule admitting a claim on the article, its group, its rank) -- an active module's action
    (group «module», the module's order), «general» obligations law (1218, 1453, 2043: subordinate), a remedies.json
    matter's action («matter»); a limiting rule or none: ("", why, ()). v79.9: an active module's «non_actions» (2118,
    2119 for a dismissed worker) never; a consumer-code article only with a professional counterparty (consumer)."""
    d = institution_modules()
    for spec, why in (d.get("limiting") or {}).items():
        if same_article(label, spec):
            return "", CLAIM_REFUSED["limiting"].format(why=why), ()
    for m in modules:                                               # v79.9 (B2)
        for spec, why in (m.get("non_actions") or {}).items():
            if same_article(label, spec):
                return "", CLAIM_REFUSED["non_action"].format(why=why), ()
    if not consumer and consumer_code(label):                       # v79.9 (B3)
        return "", CLAIM_REFUSED["consumer"], ()
    if consumer_code(label) and consumer_scope(label, sale):       # v79.12.5 (item 10): the sale block, a sale only
        return "", CLAIM_REFUSED["consumer_sale"], ()
    for i, m in enumerate(modules):
        for j, spec in enumerate(m.get("actions") or []):
            if same_article(label, spec):
                return f"modulo {m['id']}", "module", (0, i, j)
    matter = matter_of(label)
    if any(same_article(label, spec) for spec in matter.get("actions") or []) and \
            not any(same_article(label, g) for g in d.get("general") or []):
        return f"remedies.json: {matter.get('id')}", "matter", (1, 0, 0)
    for j, spec in enumerate(d.get("general") or []):
        if same_article(label, spec):
            return "norma generale delle obbligazioni", "general", (2, 0, j)
    return "", CLAIM_REFUSED["none"], ()


def article_family(label: str) -> str:
    """v79.10.1 (A2): a claim article's institution family -- the module whose actions hold it (any module, active or
    not: 1490 is «vendita», 2051 «custodia»), else its remedies.json matter (1418 «contratti»); "" when neither."""
    for m in institution_modules().get("modules") or []:
        if any(same_article(label, spec) for spec in m.get("actions") or []):
            return str(m["id"])
    return str(matter_of(label).get("id") or "")


def article_families(label: str) -> List[str]:
    """v79.10.1 (A2, review): the families an article belongs to -- its own (article_family), then the remedies.json
    matters holding that module among their «member_modules» (custodia inside the fatti illeciti: a read naming the
    tort admits 2052 for a dog's bite); a typed contract's module is no member of «contratti» (1490 stays «vendita»)."""
    family = article_family(label)
    if not family:
        return []
    return [family] + [str(m["id"]) for m in remedies().get("matters") or []
                       if family in (m.get("member_modules") or []) and str(m["id"]) != family]


def read_families(ctx: Dict[str, Any]) -> Optional[List[str]]:
    """v79.10.1 (A2): the first read's institution families -- the modules its institutions name (module_hit, never
    filtered by «requires»: the institution has its module even when the module is not active), then the remedies.json
    matters whose «patterns» its words hold ([] when they name none: no matter article is then a claim). None with no
    first-read institution (the read failed or named none): there is no family to bound the matter table, which admits
    as in v79.8 (the case's own words are no family: «proprietario» names a party, not the proprietà)."""
    from . import family_gate
    insts = family_gate.institution_texts(ctx.get("institutions") or [])
    if not insts:
        return None
    text = " ".join(insts)
    out = [str(m["id"]) for m in institution_modules().get("modules") or [] if module_hit(m, text)]
    out += [str(m["id"]) for m in remedies().get("matters") or []
            if any(re.search(p, text, re.IGNORECASE) for p in m.get("patterns") or [])]
    return list(dict.fromkeys(out))


def family_admission(label: str, modules: Sequence[Dict[str, Any]], consumer: bool = True,
                     families: Optional[Sequence[str]] = None, sale: bool = True) -> Tuple[str, str, Tuple]:
    """v79.10.1 (A2): claim_admission, then the institution's bounds on what a remedies.json matter's actions admit
    («matter»): with a module active, nothing (v79.8 B2: «outside»); else only an article of the first read's family
    (article_families in read_families) -- and where the read's institution has a module (named, not active), only
    that module's; with no first-read institution (families None) as v79.8. ("", why, ()) when refused."""
    rule, group, rank = claim_admission(label, modules, consumer, sale)
    if not rule or group != "matter":
        return rule, group, rank
    if modules:                                                     # v79.8 (B2): nothing outside the institution
        return "", CLAIM_REFUSED["outside"], ()
    if families is None:
        return rule, group, rank
    mod_ids = {str(m["id"]) for m in institution_modules().get("modules") or []}
    allowed = [f for f in families if f in mod_ids] or list(families)
    if any(f in allowed for f in article_families(label)):
        return rule, group, rank
    return "", CLAIM_REFUSED["family"].format(family=article_family(label) or "nessuna",
                                              read=", ".join(families) or "nessuna"), ()


CONTAINMENT_LINE ="Contenimento della pena: riti alternativi, attenuanti e commisurazione della pena"   # my draft
CAP_PRIORITY = {"contest_excluded": 0, "contest_open": 0, "contest_fit": 0,                # v79.13.0: contest_fit
                "requalify": 1, "scriminante": 2, "defence": 2,
                "attenuante": 3, "contest_integrated": 4}       # v79.7: what the cap keeps first (the containment always)
NO_THREAT_WHY = {   # v79.7 (item 2): why no record is the charge's -- logged and counted, never printed
    "no_record": "nessuna scheda dell'addebito (art. {art}) tra le schede del metodo",
    "unmarked": "la scheda dell'art. {art} non è segnata come addebito (posture.mark_sides)",
    "no_elements": "la scheda dell'art. {art} non ha elementi con id",
    "no_charge": "nessun addebito: né dall'intake, né dal testo, né dalla prima lettura"}


def _ids(rec: Dict[str, Any]) -> List[str]:
    return [r["id"] for r in _live(rec) if r.get("id")]


def by_concept(rec: Dict[str, Any]) -> bool:
    """v79.7 (item 3): a general-part defence the first read or the facts name by its concept («legittima difesa»):
    its line is offered with its elements still to prove (none excluded)."""
    from . import issue_records
    return rec.get("decomposed_for") == issue_records.DEFENCE_WHY and rec.get("defence_by") == "concetto"


def _no_threat_why(records: Dict[str, Dict[str, Any]], charge: str, fatt: Callable[[str], bool]) -> str:
    charged = _charged(records)
    art = charge or (charged[0] if charged else "")
    if not art:
        return "no_charge"
    rec = records.get(art)
    if rec is None or not fatt(art):
        return "no_record"
    if not _ids(rec):
        return "no_elements"
    return "unmarked"


def prescription_cited(text: str) -> bool:
    """v79 (E4): the text cites a prescription or decadenza article (c.c. 2934-2969)."""
    from . import citations, issue_records
    return any(issue_records.prescription(citations.label(r)) for r in citations.article_refs(text or ""))


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
    from . import posture as posture_mod
    # v79.12.4 (item 1): the charge is the querela's offence and only it -- a civil article (2043 c.c. …) is a remedy
    # under it, never the claim's article, whatever its elements
    charged = [x for x in _charged(records) if fatt(x) and _live(records[x]) and posture_mod.penal_article(x)]
    event = event_of(ctx.get("certain_text") or "")          # v79.10.3 (C2): an uncharged offence fits the event --
    proven = [x for x, r in records.items() if fatt(x) and r.get("side") != "threat"       # (review) preferred, never
              and posture_mod.penal_article(x)                                           # the victim left with none
              and any(q.get("status") == PROVATO for q in _live(r))]
    main = charged[0] if charged else next((x for x in proven if not event_mismatch(x, event)), proven[0] if proven
                                           else "")
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
                     + (" [argomentata dal codice: indica solo il realismo]" if c.get("argued") == "code" else "")
                     + (" [l'argomento deve nominare l'elemento che distingue le due fattispecie: "     # v79.9 (A3)
                        + "; ".join(f"«{_short(x, 80)}»" for x in c["differs"][:3]) + "]"
                        if c.get("kind") == "requalify" and c.get("differs") else ""))
    return "\n".join(lines)


def records_block(records: Dict[str, Dict[str, Any]], circ: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
    """The thesis call's SCHEDE. v79.14.3 (item 4): never a record the frame's binding excluded (excluded means
    excluded: no scheda, no context); (item 1) a contested element's esito as the table prints it (esito_label)."""
    lines = []
    for label, rec in list((records or {}).items()) + list((circ or {}).items()):
        live = _live(rec)
        if not live or frame_excluded(rec):
            continue
        side = {"threat": "addebito o domanda contro il cliente", "weapon": "norma del cliente"}.get(
            str(rec.get("side") or ""), "circostanza" if rec.get("circumstance") else "")
        lines.append(f"- art. {label}" + (f" — {_plain(rec.get('rubrica'))}" if rec.get("rubrica") else "")
                     + (f" [{side}]" if side else ""))
        lines += [f"  · [{r.get('id')}] {_short(r.get('text'), 140)}: {esito_label(r)}" for r in live]
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
    if ctx.get("riti_precluded"):           # v79.10.3 (C1): never proposed -- the pena of the charge precludes them
        parts.append(RITI_PRECLUDED_TITLE + "\n" + "\n".join(f"- {x}" for x in ctx["riti_precluded"]))
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


_CANDIDATE_KEY_RE = re.compile(r"^\W*(?:(?:tesi|candidat[ao]|t)[\W_]*)?(?:t[\W_]*)?(\d{1,2})(?!\d)", re.I)
_VALUATION_FIELDS = ("realismo", "pro", "contro", "argomento")


def candidate_key(key: Any) -> str:
    """v79: a valuation's candidate id as the reply wrote it -- «T1», «t1», «[T1]», «T1 - Contestare …» -> «T1»;
    v79.7: «T 1», «T01», «Tesi 1», «tesi_T1», «candidata 1», «1» too. Any other key upper-cased as it is (so a thesis
    the model adds is still seen, and dropped)."""
    text = " ".join(str(key or "").split())
    m = _CANDIDATE_KEY_RE.match(text)
    return f"T{int(m.group(1))}" if m else text.upper()


def _valuation(v: Any) -> bool:
    return isinstance(v, dict) and any(f in v for f in _VALUATION_FIELDS)


def valuations(raw: Any, candidates: Sequence[Dict[str, Any]] = ()) -> Dict[str, Dict[str, Any]]:
    """v79 (A1): the reply's «valutazioni» keyed by candidate id, in any shape the model gives them -- an object keyed
    by id, or a list of objects carrying their id («id», «tesi» or «candidata»); a value that is not an object dropped.
    v79.7 (item 1): a key or an id that is the candidate's own text (its «tesi») is that candidate's; a list whose
    items carry no id at all is in the candidates' order; a list of one-key objects ([{"T1": {…}}, …]); with no
    «valutazioni», top-level keys that read as candidate ids."""
    by_text = {_plain(c.get("tesi")).lower().rstrip("."): c["id"] for c in candidates or [] if c.get("id")}

    def key_of(k: Any) -> str:
        return by_text.get(_plain(k).lower().rstrip(".")) or candidate_key(k)
    raw = raw if isinstance(raw, dict) else {}
    vals = raw.get("valutazioni", raw.get("valutazione"))
    if vals is None:
        vals = {k: v for k, v in raw.items() if _CANDIDATE_KEY_RE.match(str(k)) and _valuation(v)} or None
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(vals, dict):
        for k, v in vals.items():
            if isinstance(v, dict):
                out.setdefault(key_of(k), v)
    elif isinstance(vals, list):
        items = [v for v in vals if isinstance(v, dict)]
        named = [next((v[f] for f in ("id", "tesi", "candidata", "candidate") if v.get(f)), "") for v in items]
        for i, (v, k) in enumerate(zip(items, named), 1):
            if not k and len(v) == 1 and isinstance(next(iter(v.values())), dict):      # [{"T1": {…}}, …]
                k, v = next(iter(v.items()))
            elif not k and not any(named):          # no item names its candidate: the candidates' order
                k = f"T{i}"
            if k:
                out.setdefault(key_of(k), v)
    return out


# v79.12.7: what a candidate carries into its solution -- the jurisprudential element and its sources (item 1), the
# riqualificazione's facts (item 4), the causation line and its cap (item 5)
SOLUTION_CARRIED = ("jurisprudential", "sources", "facts", "causation", "causation_capped")


def apply_valuations(raw: Any, candidates: Sequence[Dict[str, Any]], log: Optional[Dict[str, Any]] = None
                     ) -> List[Dict[str, Any]]:
    """v72: the soluzioni -- the code's candidates, each with the model's pro, contro, realismo and argument (a
    minority one: realismo basso); a thesis the model adds (a key no candidate has) dropped, into log["dropped"].
    v79.7 (item 1): every candidate's realismo is the model's -- a code-argued one keeps the code's text (v74: nothing
    the model wrote can negate an element), never the code's fixed realismo; one the code rates «basso» (nothing proven
    yet, an eccezione with no presupposto in the facts) at most «medio»; the code's when the model gave none."""
    raw = raw if isinstance(raw, dict) else {}
    vals = valuations(raw, candidates)
    ids = {c["id"] for c in candidates}
    dropped = [k for k in vals if k not in ids]
    if raw.get("soluzioni"):
        dropped.append("soluzioni")
    out = []
    ignored = [c["id"] for c in candidates if c.get("code_argument") and
               any(_plain((vals.get(c["id"]) or {}).get(f)) for f in BIND_FIELDS)]
    unnamed = []
    for c in candidates:
        model = vals.get(c["id"]) or {}
        code = c.get("code_argument") or {}
        if c["kind"] == "requalify" and not code and not names_element(
                " ".join(_plain(model.get(f)) for f in ("argomento", "pro")), c.get("differs") or []):
            unnamed.append(c["id"])     # v79.9 (A3): an argument that names no differing element: not offered
            continue
        v = code or model                   # v74: a code-argued candidate's argument is never the model's
        given = _plain(model.get("realismo")).lower()
        real = given if given in REALISMO else _plain(code.get("realismo")).lower()
        real = real if real in REALISMO else (BASSO if is_minority(c) else MEDIO)
        if code.get("realismo") == BASSO and real == ALTO:
            real = MEDIO
        if is_minority(c) or c.get("causation_capped"):     # v79.12.7 (item 5): an ordinary consequence -- basso
            real = BASSO
        out.append({"tesi": c["tesi"], "pro": _plain(v.get("pro")), "contro": _plain(v.get("contro")), "realismo": real,
                    "argomento": _plain(v.get("argomento")), "minoritaria": is_minority(c),
                    "contesta": list(c["contesta"]), "poggia_su": list(c["poggia_su"]), "candidate": c["id"],
                    "kind": c["kind"], "argued": c.get("argued") or "model", "valued": given in REALISMO,
                    **({"fallback": True} if c.get("fallback") else {}),
                    **{k: c[k] for k in SOLUTION_CARRIED if c.get(k)}})       # v79.12.7 (items 1, 4, 5)
    if log is not None:
        log["dropped"] = dropped
        log["code_argued_ignored"] = ignored
        log["requalify_unnamed"] = unnamed
    return out


def score_sample(raw: Any, candidates: Sequence[Dict[str, Any]], known: Dict[str, Dict[str, Any]]
                 ) -> Tuple[float, List[str]]:
    """v72: one reply of the thesis call, scored by the code -- every candidate valued and none added (valid ids), no
    candidate resting on an excluded element rated above «basso», no minority one above «basso»; an unreadable reply:
    -inf. (v73: the objections are no longer the thesis call's -- objections_plan writes them after the selection.)"""
    if not isinstance(raw, dict) or not raw:
        return float("-inf"), ["reply not readable"]
    vals = valuations(raw, candidates)
    ids = {c["id"] for c in candidates}         # v79.7 (item 1): every candidate valued, the code-argued ones too
    keys = set(vals)
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
        if real != BASSO and is_minority(c):
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


SAMPLES = 1                    # v72: best-of-3 on the thesis call; v79.8 (A3): one -- score_sample cannot tell the
                               # samples apart (every one valued, the same score), so two of three replies were waste.
                               # best-of-3 comes back only with a scorer that discriminates between them
SAMPLE_TEMPERATURE = 0.7
DETERMINISTIC_ENV = "THESIS_DETERMINISTIC"     # v79 (A3): one sample at temperature 0 (nightly / replay --deterministic)


def deterministic() -> bool:
    return (os.getenv(DETERMINISTIC_ENV) or "").strip().lower() in ("1", "true", "yes", "on")


def sampling() -> Tuple[int, float]:
    """(replies, temperature) of the thesis call: SAMPLES at SAMPLE_TEMPERATURE (v79.8: one); v79 (A3): one at 0 under
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
    ruling_elements(ctx)                # v79.12.7 (item 1): the kept passages' jurisprudential elements, before the lines
    ctx["candidates"] = thesis_candidates(ctx)
    if schemas_enabled():               # v79.11 (6): the schemas' admission, shadow only -- the candidates untouched
        try:
            ctx["schema_shadow"] = schema_shadow(ctx)
        except Exception as exc:        # noqa: BLE001 -- a shadow never fails the plan
            logger.warning("lawyer_method: schema shadow failed: %s", exc)
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
    _log("plan", {"candidates": len(ctx["candidates"]),
                  "candidate_articles": [", ".join(candidate_subjects(ctx.get("records") or {})(c))     # v79.14.2
                                         for c in ctx["candidates"]],                                    # (B)
                  "frame_binding": {k: (ctx.get("frame_binding") or {}).get(k) for k in
                                    ("mode", "applied", "excluded", "empty", "rescued")},     # v79.14.3: rescued
                  "contestato": contested_counts(ctx),                                        # v79.14.3 (item 1)
                  "samples": [x["score"] for x in scored], "best": best,
                  "choice": ctx["thesis_choice"], "counts": ctx["selection_counts"],
                  "dropped": clean_log.get("dropped"),
                  "orientamenti": len(plan["orientamenti"]), "seconds": seconds, "log": ctx["thesis_log"]},
         seconds * 1000)
    return plan


def schemas_enabled() -> bool:
    from . import element_schemas
    return element_schemas.enabled()


def ruling_elements(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """v79.12.7 (item 1): the jurisprudential elements a kept ruling passage states (penal_lines.jurisprudential_
    elements on the rulings record's «kept»), into ctx["records"]; an element added refreshes the assessment and §4's
    tables (prepare's inputs, ctx["refresh_inputs"]) -- the facts' «mancanti» stay prepare's. A penal posture only."""
    record = ctx.get("rulings_record") or {}
    if not record.get("kept") or not criminal(ctx.get("posture")):
        return []
    from . import penal_lines
    log: Dict[str, Any] = {}
    got = penal_lines.jurisprudential_elements(ctx.get("records") or {}, record.get("kept") or [], schemas_enabled(),
                                               circ=ctx.get("circumstances"), log=log,
                                               id_floor=int((ctx.get("refresh_inputs") or {}).get("id_floor") or 0))
    ctx["jurisprudential"] = list(ctx.get("jurisprudential") or []) + got
    if any(x.get("how") == "added" for x in got):
        refresh_after_elements(ctx)
    return got


def max_element_id(*groups: Dict[str, Dict[str, Any]]) -> int:
    """v79.12.7 (item 1): the highest «R<n>» id of the state's and the method's records -- an element added later
    never takes an id a record holds."""
    from . import penal_lines
    return penal_lines.max_id(*groups)


def printed_records(records: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """The records the answer prints and assesses (prepare's): no ranked candidate, nothing unprinted."""
    return {x: r for x, r in (records or {}).items() if not r.get("ranked_candidate") and not r.get("unprinted")}


def refresh_after_elements(ctx: Dict[str, Any]) -> None:
    """v79.12.7 (item 1): an element added after prepare -- the assessment and §4's tables computed again on prepare's
    inputs (ctx["refresh_inputs"]: the main line's anchor, §3's order, the aggravante note)."""
    inputs = ctx.get("refresh_inputs") or {}
    records = printed_records(ctx.get("records") or {})
    by_id = {s["id"]: s for s in ctx.get("sentences") or [] if isinstance(s, dict) and s.get("id")}
    ctx["assessment"] = bound_assessment(assessment(records, ctx.get("posture") or {}, inputs.get("main") or "",
                                                    ctx.get("client") or "", event=event_of(ctx.get("certain_text") or "")),
                                         ctx.get("frame_binding"))         # v79.14.2 (B): an emptied charge set
    ctx["subsumption"] = subsumption_section(records, inputs.get("order") or [], by_id, ctx.get("circumstances"),
                                             inputs.get("aggravante_note") or "")


# v79.11 (6, 7): the candidates' kinds by the schema's admission categories
SHADOW_CATEGORIES = {"claim": ("claim", "riconvenzionale", "remedy", "claim_offence", "parte_civile"),
                     "defence": ("scriminante", "defence", "attenuante"),
                     "contest": ("contest_excluded", "contest_open", "contest_integrated", "eccezione", "requalify")}


def _shadow_line(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The record's schema (rec["schema"], the summary decomposition kept) as the line shadow_admission reads; None
    without one (an article read on the text path)."""
    s = rec.get("schema") if isinstance(rec, dict) else None
    if not isinstance(s, dict):
        return None
    return {"schema": {"tipo": list(s.get("tipo") or []), "legittimato": list(s.get("legittimato") or []),
                       "rapporto": s.get("rapporto")}, "reviewed": s.get("reviewed") is True,
            "source": s.get("source") or ""}


def _admits(label: str, articles: Sequence[str]) -> bool:
    return any(a == label or same_article(label, a) for a in articles)


def schema_shadow(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """v79.11 (6, 7): what the schemas would admit (element_schemas.shadow_admission) for every record of ctx["records"]
    -- those thesis_candidates filters out before admission too -- against the current candidates, by article and
    category (SHADOW_CATEGORIES): {"items", "current": {cat: [articles]}, "schema": {cat: [articles]}, "agree" (the
    record-categories both read alike, records with a schema), "false_exclusions": [(article, cat, why)] (admitted now,
    refused by the schema), "extra": [(article, cat)] (the schema's only), "no_schema": [articles], "recall_hit",
    "recall_of", "candidate_recall"}; logged (schema_admission). Shadow only: nothing in ctx changes but the caller's
    ctx["schema_shadow"]."""
    from . import element_schemas
    role = str((ctx.get("posture") or {}).get("role") or "")
    owner = {str(r.get("id")): label for label, rec in (ctx.get("records") or {}).items()
             for r in (rec or {}).get("requisites") or [] if r.get("id")}
    current: Dict[str, List[str]] = {cat: [] for cat in SHADOW_CATEGORIES}
    for c in ctx.get("candidates") or []:
        cat = next((k for k, kinds in SHADOW_CATEGORIES.items() if c.get("kind") in kinds), "")
        if not cat:
            continue
        # a candidate's articles: its own, a remedy's list, else (a contest) the records of the elements it names
        arts = [c["article"]] if c.get("article") else list(c.get("articles") or []) or \
            [owner[str(i)] for i in list(c.get("contesta") or []) + list(c.get("poggia_su") or []) if str(i) in owner]
        for a in arts:
            if a and str(a) not in current[cat]:
                current[cat].append(str(a))
    schema: Dict[str, List[str]] = {cat: [] for cat in SHADOW_CATEGORIES}
    items, no_schema, false_ex, extra = [], [], [], []
    agree = hit = of = 0
    admitted, rejected = [], []
    for label, rec in (ctx.get("records") or {}).items():
        item = element_schemas.shadow_admission(label, _shadow_line(rec), role, (rec or {}).get("side"))
        items.append(item)
        if not item.get("schema"):
            no_schema.append(label)
            continue
        for cat in SHADOW_CATEGORIES:
            now, then = _admits(label, current[cat]), bool(item.get(cat))
            why = "; ".join(w for w in item.get("why") or [] if str(w).startswith(f"{cat}:"))
            if then:
                schema[cat].append(label)
                admitted.append((label, cat, bool(item.get("conditional"))))
            else:
                rejected.append((label, cat, why))
            if now:
                of += 1
                hit += then
            if now == then:
                agree += 1
            elif now:
                false_ex.append((label, cat, why))
            else:
                extra.append((label, cat))
    out = {"items": items, "current": current, "schema": schema, "agree": agree, "false_exclusions": false_ex,
           "extra": extra, "no_schema": no_schema, "recall_hit": hit, "recall_of": of,
           "candidate_recall": round(hit / of, 3) if of else None}
    _log("schema_admission", {"role": role, "admitted": admitted, "rejected": rejected, "false_exclusions": false_ex,
                              "extra": extra, "no_schema": no_schema, "candidate_recall": out["candidate_recall"]})
    return out


COUNT_KEYS = ("offered", "parsed", "valued", "barred", "unverifiable", "minoritaria", "chosen", "main_is_fallback")


def general_obligations(label: str) -> bool:
    """v79.12.5 (item 11): general obligations law -- institution_modules.json «general» (1218, 1453, 2043)."""
    return any(same_article(label or "", g) for g in institution_modules().get("general") or [])


def module_proven_line(ctx: Dict[str, Any]) -> bool:
    """v79.12.5 (item 11): the institution's module yields a proven line -- a claim or riconvenzionale candidate
    admitted by an active module («modulo …») resting on a proven element. Review fix: thesis_candidates' own reading
    (ctx «module_proven»: a module line on proven elements, not barred) when it ran; else a module candidate whose
    poggia_su is proven -- never an open one («…, provando …»: its poggia_su lists the elements still to prove)."""
    if "module_proven" in ctx:
        return bool(ctx["module_proven"])
    return any(c.get("kind") in ("claim", "riconvenzionale") and str(c.get("admitted_by") or "").startswith("modulo ")
               and c.get("poggia_su") and ", provando " not in str(c.get("tesi") or "")
               for c in ctx.get("candidates") or [])


def main_outside(ctx: Dict[str, Any], main: Dict[str, Any]) -> bool:
    """v79.8 (B4): the main line is a claim (or a remedy) on an article outside the first read's institution modules,
    while one is active. v79.12.5 (item 11, row 24): general obligations law (general_obligations: 1453 on a
    preliminare) is accepted when the institution's module yields nothing proven (module_proven_line) -- the line is
    right there, so the counter measures only a main line outside the institution while it had one of its own."""
    cand = next((c for c in ctx.get("candidates") or [] if c.get("id") == main.get("candidate")), {})
    art = str(cand.get("article") or main.get("article") or "")
    if art and general_obligations(art) and not module_proven_line(ctx):
        return False
    if (line_pertinence(main, ctx) or (0, 0))[0]:      # v79.12.1 (row 16): its norm outside the first read's
        return True                                     # institution (must be 0: a line inside it is promoted)
    mods = [m for m in institution_modules().get("modules") or [] if m["id"] in (ctx.get("claim_modules") or [])]
    if not mods or cand.get("kind") not in ("claim", "riconvenzionale", "remedy"):
        return False
    return not any(same_article(cand.get("article") or "", a) for m in mods for a in m.get("actions") or [])


def selection_counts(ctx: Dict[str, Any], plan: Dict[str, Any], raw: Any = None) -> Dict[str, Any]:
    """v79 (A2): where the candidates went -- offered (the code's), parsed (reached the selection as soluzioni), valued
    (v79.7: the candidates the kept reply gave a realismo, every kind), barred (contest an «integrato» element, unlabelled), unverifiable,
    minoritaria, the main line chosen (its candidate id, «fallback» when the selection inserted one) and whether it is
    the fallback (the posture's or the remedies table's line, no candidate of the code's). v79.11 (6): main_admitted_by
    (the main line's candidate's admitted_by, "" when none) and, with ELEMENT_SCHEMAS, schema_shadow's numbers."""
    cands = ctx.get("candidates") or []
    vals = valuations(raw, cands)
    sols, main, _ = selection(ctx, plan)
    m = sols[main] if sols else {}
    inserted = bool(m.get("fallback")) and not m.get("candidate")
    main_cand = next((c for c in cands if m.get("candidate") and c.get("id") == m.get("candidate")), {})
    shadow = ctx.get("schema_shadow") if isinstance(ctx.get("schema_shadow"), dict) else None
    out = {"offered": len(cands), "parsed": sum(1 for t in plan.get("soluzioni") or [] if t.get("candidate")),
            "valued": sum(1 for c in cands if _plain((vals.get(c["id"]) or {}).get("realismo")).lower() in REALISMO),
            "model_argued": sum(1 for c in cands if c.get("argued", "model") == "model"),
            "barred": sum(1 for t in sols if t.get("barred")),
            "unverifiable": sum(1 for t in sols if t.get("unverifiable")),
            "minoritaria": sum(1 for t in sols if t.get("minoritaria")),
            "chosen": "fallback" if inserted else str(m.get("candidate") or ""),
            "chosen_kind": str(m.get("kind") or ("fallback" if inserted else "")),
            "main_is_fallback": inserted,
            # v79.7 (item 2): where the charge's contests came from (a record marked «threat», else the charged
            # records) and, when none was marked, why (NO_THREAT_WHY)
            "samples": len((ctx.get("thesis_choice") or {}).get("scores") or []),      # v79.8 (A5): replies used
            # v79.8 (B4): candidates no rule admits (must be 0); the main line's norm outside the read's institution
            "unadmitted": sum(1 for c in cands if not c.get("admitted_by")),
            # v79.10.1 (A2): the client's articles a matter's actions admitted, refused outside the read's family
            "claims_refused_family": len(ctx.get("claims_refused_family") or []),
            # v79.10.1 (A3): the lines never_main kept out of the main line (circumstances, «elemento escluso» fragments)
            "main_demoted": sum(1 for t in sols if t.get("main_demoted")),
            # v79.10.2 (B1): the resisting party's reductions offered (1384 for a penalty)
            "reductions_offered": sum(1 for c in cands if c.get("reduction")),
            "main_outside_institution": main_outside(ctx, m),
            # v79.12.1 (row 16): the main line promoted over realismo's by pertinence (where it stood: from)
            "promoted_by_pertinence": int(bool(m.get("promoted_by_pertinence"))),
            "promoted_from": str((m.get("promoted_by_pertinence") or {}).get("from") or ""),
            "threats_from": str((ctx.get("candidates_trace") or {}).get("threats_from") or ""),
            "no_threat": str((ctx.get("candidates_trace") or {}).get("no_threat") or ""),
            # v79 (D4): the general part's defence lines, the ranked articles graded into candidates
            "defence_lines": sum(1 for c in cands if c.get("kind") in ("defence", "scriminante")),
            "ranked_candidates": sum(1 for c in cands
                                     if ((ctx.get("records") or {}).get(c.get("article") or "") or {}).get("ranked_candidate")),
            # v79.9 (A3): a riqualificazione into another code than the charge's (must be 0); those the gate refused;
            # those whose argument named no differing element (not offered)
            "riqualificazione_out_of_code": sum(1 for c in cands if c.get("kind") == "requalify" and
                                                label_code(c.get("article")) != label_code(c.get("charge"))),
            "requalify_refused_code": len((ctx.get("candidates_trace") or {}).get("requalify_out_of_code") or []),
            # v79.10.3 (C2): a riqualificazione the certain facts' event does not fit (a death: never 590; injuries and
            # no death: never 575 / 589) -- must be 0; those the gate refused
            "requalify_event_mismatch": sum(1 for c in cands if c.get("kind") == "requalify" and event_mismatch(
                c.get("article") or "", event_of(ctx.get("certain_text") or ""))) + int(event_mismatch(   # (review) and
                (ctx.get("assessment") or {}).get("alternative") or "", event_of(ctx.get("certain_text") or ""))),  # §9's
            "requalify_refused_event": len((ctx.get("candidates_trace") or {}).get("requalify_refused_event") or []),
            "requalify_unnamed": sum(1 for c in cands if c.get("kind") == "requalify" and c.get("id") not in
                                     {t.get("candidate") for t in plan.get("soluzioni") or []}),
            # v79.9 (C2): a candidate from a document outside the case's families (must be 0); those the gate dropped
            "candidates_out_of_family": sum(1 for c in cands if c.get("article") and not family_gate_ok(
                c["article"], family_context(ctx), [str((ctx.get("assessment") or {}).get("article") or "")] +
                [x for x, r in (ctx.get("records") or {}).items() if r.get("side") == "threat"])),
            "candidates_family_dropped": len(ctx.get("family_dropped") or [])}
    out.update(item_counts(ctx, sols, main, m))                                     # v79.12.4 (items 1, 4, 5, 7)
    out.update(penal_counts(ctx, sols, m))                                          # v79.12.7 (items 1, 3, 4, 5)
    # v79.11 (6): the rule admitting the main line's candidate ("" when none: a fallback, no candidate) -- always; the
    # schemas' shadow admission in numbers (ELEMENT_SCHEMAS: only when method_plan computed it)
    out["main_admitted_by"] = str(main_cand.get("admitted_by") or "")
    if shadow is not None:
        out["schema_shadow"] = {k: (len(v) if isinstance(v, list) else v) for k, v in shadow.items()
                                if k in ("agree", "false_exclusions", "extra", "recall_hit", "recall_of")}
    return out


PENAL_COUNT_KEYS = ("jurisprudential_elements", "jurisprudential_minoritaria", "jurisprudential_main",
                    "absorbed_titles", "absorbed_as_autonomous", "soccorso_post_fact", "requalify_no_facts",
                    "requalify_without_facts", "causation_capped", "causation_cap_lifted", "causation_over_basso",
                    "causation_replies_rewritten",
                    # v79.12.7 (items 2, 6)
                    "attenuanti_from_facts", "attenuanti_to_verify", "attenuanti_excluded_by_facts",
                    "bilanciamento_lines", "attenuanti_fixed_block", "rites_computed", "rites_without_computation",
                    "rites_computation_mismatch")


def penal_counts(ctx: Dict[str, Any], sols: Sequence[Dict[str, Any]], m: Dict[str, Any]) -> Dict[str, int]:
    """v79.12.7: the nightly's counters of items 1, 3, 4 and 5, from the candidates and the selection --
      jurisprudential_elements      the elements a note, a kept passage or a reviewed schema states and the text does
                                    not, in the records (inferred, «da provare»);
      jurisprudential_minoritaria   a contest of one labelled minoritaria (must be 0);
      jurisprudential_main          the main line contests one;
      absorbed_titles               the titles absorbed or made inapposite (581/582 beside 584, 589/590 kept as the
                                    alternative only, 593 where a soccorso is narrated);
      absorbed_as_autonomous        such a title still offered as autonomous -- a record printed or threatened, a
                                    candidate on it but the riqualificazione (must be 0);
      soccorso_post_fact            the soccorso sentences turned into a favourable element after the fact;
      requalify_no_facts            the riqualificazioni of a listed pair not offered for lack of facts on the
                                    differing element;
      requalify_without_facts       one offered with none (must be 0);
      causation_capped              the causation lines capped (an ordinary consequence, no trigger fact);
      causation_cap_lifted          those left uncapped by a trigger fact (pathology, medical error, abnormal cause);
      causation_over_basso          a capped line rated above «basso» (must be 0);
      causation_replies_rewritten   the objections' replies denying the causal link replaced (objections_plan sets it
                                    after the selection);
      attenuanti_from_facts         (item 2) the attenuanti lines printed from a fact of the case (a diverbio, a
                                    concrete step named, a favourable element);
      attenuanti_to_verify          those printed «da verificare» for lack of a fact (62-bis);
      attenuanti_excluded_by_facts  those printed «esclusa dai fatti» (the record excludes them by a sentence);
      bilanciamento_lines           the art. 69 c.p. line printed (an aggravante in the record);
      attenuanti_fixed_block        the fixed attenuanti block still among the printed lines (must be 0);
      rites_computed                (item 6) the riti printed with their computation (patteggiamento, abbreviato);
      rites_without_computation     a rito with a computable pena printed without it (must be 0);
      rites_computation_mismatch    a status the printed result contradicts -- «possibile» over the limit, «preclusa»
                                    within it (must be 0)."""
    from . import penal_lines
    cands = ctx.get("candidates") or []
    trace = ctx.get("candidates_trace") or {}
    causal = [c for c in cands if c.get("causation")]
    return {
        "jurisprudential_elements": len({(x.get("article"), x.get("id")) for x in ctx.get("jurisprudential") or []}),
        "jurisprudential_minoritaria": sum(1 for c in cands if c.get("jurisprudential") and is_minority(c)),
        "jurisprudential_main": int(bool(m.get("jurisprudential"))),
        "absorbed_titles": len(ctx.get("absorbed") or []),
        "absorbed_as_autonomous": penal_lines.absorbed_as_autonomous(ctx),
        "soccorso_post_fact": len(ctx.get("post_fact") or []),
        "requalify_no_facts": len(trace.get("requalify_no_facts") or []),
        "requalify_without_facts": sum(1 for c in cands if c.get("kind") == "requalify" and c.get("requalify_rule")
                                       and not c.get("facts")),
        "causation_capped": sum(1 for c in causal if c.get("causation_capped")),
        "causation_cap_lifted": sum(1 for c in causal if not c.get("causation_capped")
                                    and (trace.get("causation_cap") or {}).get("triggers")),
        "causation_over_basso": sum(1 for t in sols if t.get("causation_capped") and t.get("realismo") != BASSO),
        "causation_replies_rewritten": int(ctx.get("causation_replies_rewritten") or 0),
        **attenuanti_counts(ctx), **rite_counts(ctx)}


def attenuanti_counts(ctx: Dict[str, Any]) -> Dict[str, int]:
    """v79.12.7 (item 2): penal_counts' attenuanti counters, from prepare's attenuanti_log and the printed lines."""
    how = list(((ctx.get("attenuanti_log") or {}).get("lines") or {}).values())
    fixed = {x["line"] for x in ctx.get("rites") or [] if not x.get("rite", True)} if ctx.get("attenuanti_log") else set()
    return {"attenuanti_from_facts": sum(1 for x in how if x in ("facts", "step")),
            "attenuanti_to_verify": how.count("to_verify"), "attenuanti_excluded_by_facts": how.count("excluded"),
            "bilanciamento_lines": int(bool((ctx.get("attenuanti_log") or {}).get("aggravanti"))),
            "attenuanti_fixed_block": sum(1 for x in ctx.get("riti") or [] if x in fixed)}


def rite_counts(ctx: Dict[str, Any]) -> Dict[str, int]:
    """v79.12.7 (item 6): the riti printed (ctx["rites"], §6 and §8) with a computable pena -- each with its computation
    in its line (rites_computed), none without (rites_without_computation), none whose status the printed result
    contradicts (rites_computation_mismatch)."""
    done = without = mismatch = 0
    for x in ctx.get("rites") or []:
        if not x.get("rite", True) or not x.get("computable"):
            continue
        if x.get("computation") and x["computation"] in str(x.get("line") or ""):
            done += 1
        else:
            without += 1
        limit, result = x.get("limit"), x.get("result")
        if limit is not None and result is not None and x.get("status") in (POSSIBLE, PRECLUDED):
            over = float(result) > float(limit) + 1e-9
            mismatch += int(over != (x.get("status") == PRECLUDED and x.get("why") == "min"))
    return {"rites_computed": done, "rites_without_computation": without, "rites_computation_mismatch": mismatch}


def item_counts(ctx: Dict[str, Any], sols: Sequence[Dict[str, Any]], main: int, m: Dict[str, Any]) -> Dict[str, int]:
    """v79.12.4: the nightly's counters of items 1, 4, 5 and 7, from the selection --
      victim_charge_civil         the persona offesa's or parte civile's outcome on a civil article (must be 0);
      main_fragment               the main line a fragment contest (escluso_fragment) while another line was eligible
                                  (must be 0);
      main_other_threat           the main line a contest of a threat record that is not the charge while a line of the
                                  charge's was eligible (must be 0);
      event_narrated_proven       the elements naming the certain facts' event set «provato» (narrated_event_proven);
      event_narrated_unproven     those still «da provare» (must be 0);
      event_narrated_causal       the elements stating the narrated event with its cause («cagiona la morte»), still
                                  «da provare»: the causal link is what remains to prove (informative);
      defence_lines_40_43         a defence or scriminante candidate on art. 40 or 43 c.p. (must be 0);
      main_weak_defence           the main line a defence or scriminante with fewer than half its elements proven while
                                  another line was eligible (must be 0);
      claims_refused_expired      the client's claims refused on a term DECORSO (a decadenza);
      claims_expired_minority     those kept minoritaria (a prescrizione; a decadenza the facts do not settle or the
                                  statute excuses; general obligations law under a barred module's terms);
      claims_expired_sustainable  a claim on an article a DECORSO term bars offered as sustainable (must be 0)."""
    from . import fact_signals, posture as posture_mod
    cands = ctx.get("candidates") or []
    records = ctx.get("records") or {}
    known = elements_by_id(records, ctx.get("circumstances"))
    a = ctx.get("assessment") or {}
    charges = charge_articles(records, a, ctx)
    others = [t for i, t in enumerate(sols) if i != main and eligible(t)]
    role = str((ctx.get("posture") or {}).get("role") or "")
    claims = [c for c in cands if c.get("kind") in ("claim", "riconvenzionale", "remedy") and c.get("article")]
    return {
        "victim_charge_civil": int(role in VICTIM_ROLES and bool(a.get("article"))
                                   and not posture_mod.penal_article(str(a.get("article")))),
        "main_fragment": int(bool(m) and escluso_fragment(m, known) and any(
            not escluso_fragment(t, known) and t.get("kind") not in CIRCUMSTANCE_KINDS for t in others)),
        "main_other_threat": int(bool(m) and other_threat_contest(m, known, charges) and any(
            not other_threat_contest(t, known, charges) for t in others)),
        "event_narrated_proven": len(ctx.get("event_proven") or []),
        "event_narrated_unproven": event_unproven(records, str(ctx.get("certain_text") or "")),
        "event_narrated_causal": event_causal_open(records, str(ctx.get("certain_text") or "")),
        "defence_lines_40_43": sum(1 for c in cands if c.get("kind") in DEFENCE_KINDS
                                   and never_defence(str(c.get("article") or ""))),
        "main_weak_defence": int(bool(m) and m.get("kind") in DEFENCE_KINDS and not half_proven(m, known)
                                 and bool(others)),
        "claims_refused_expired": len(ctx.get("claims_expired") or []),
        "claims_expired_minority": len(ctx.get("claims_expired_minority") or []),
        "claims_expired_sustainable": sum(1 for c in claims if not is_minority(c) and (
            fact_signals.barring_terms(c["article"], ctx.get("terms")) or c.get("expired")))}


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
                                    ctx.get("riti") or [], [], ctx.get("posture") or {}, ctx.get("circumstances"),
                                    ctx=ctx)                    # v79.12.1 (row 16): pertinence before realismo
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


def objections_material(ctx: Dict[str, Any], sols: Sequence[Dict[str, Any]], main: int, sub: Optional[int],
                        party: Optional[Dict[str, Any]] = None) -> str:
    """The objections call's material; v79.14.1 (item 6): the POSIZIONE is the objection_party's (the frame's client
    and role under CASE_FRAME)."""
    a = ctx.get("assessment") or {}
    party = party or _party_of(ctx)
    where = posture_text({**(ctx.get("posture") or {}), "role": party.get("role") or ""}) if party.get("frame") else \
        ctx.get("posture_text")
    out = [f"CASO:\n{ctx.get('case') or ''}",
           f"POSIZIONE: {party.get('client') or ctx.get('client') or 'la parte assistita'}, {where or 'non indicata'}",
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
    "placeholder": "i fatti narrati non offrono una linea fondata sugli elementi delle norme",
    "undecomposed": "l'art. {art} (l'addebito o la domanda) non si scompone in elementi verificabili",
    "unread": "il testo dell'art. {art} (l'addebito o la domanda) non è stato recuperato",
}   # v79.10.1 (A1): with no norm conferring an action, the why is the no-action sentence (method_refusal)

# v79.12.1: a damages-measure article (1223-1227 c.c.: misura, obbligazioni pecuniarie, prevedibilità, valutazione
# equitativa, concorso del creditore) measures a claim -- it is never «l'addebito o la domanda» whose unread or
# undecomposed text makes the method step aside
DAMAGES_MEASURE = tuple(f"{n} c.c." for n in range(1223, 1228))


def damages_measure(label: str) -> bool:
    return _norm_label(label) in {_norm_label(x) for x in DAMAGES_MEASURE}


def not_applied_note(why: str) -> str:
    return NOT_APPLIED.format(why=str(why or NOT_APPLIED_WHY["placeholder"]).rstrip("."))


def placeholder_main(ctx: Dict[str, Any], plan: Dict[str, Any]) -> str:
    """v75: the selection's main line when it is the placeholder («Tesi da costruire …», fallback_thesis with no
    posture line and no containment) -- a failure state: the method refuses the answer (the standard body is written,
    logged); "" otherwise."""
    sols, main, _ = selection(ctx, plan)
    return sols[main]["tesi"] if sols[main].get("placeholder") else ""


def client_refusal(ctx: Optional[Dict[str, Any]], plan: Dict[str, Any]) -> Dict[str, Any]:
    """v79.10.1 (A1): one client's refusal -- {"tesi", "why", "no_action", "lines"} when its main line is the
    placeholder, {} otherwise. The why: with no norm conferring an action on the claimant (ctx["no_action"]) the
    no-action sentence -- its one appearance in the answer, never «Linea principale», never with a forecast -- else
    v76's placeholder why. "lines": the lines the method would have printed that the prose keeps (the missing-laws
    line, v79.9 B2: a dismissal with its special laws out of the archive)."""
    if ctx is None:
        return {}
    tesi = placeholder_main(ctx, plan)
    if not tesi:
        return {}
    line = str(ctx.get("no_action") or "")
    if line and no_claim(ctx):          # v79.12.5 (item 2): the ten sections stay (compose: no_claim)
        return {}
    why = (line[:1].lower() + line[1:]) if line else NOT_APPLIED_WHY["placeholder"]
    return {"tesi": tesi, "why": why, "no_action": bool(line), "lines": [x for x in [missing_laws_line(ctx)] if x]}


def method_refusal(items: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]]) -> Dict[str, Any]:
    """v79.10.1 (A1): the method steps aside to the prose answer (v76's note, the standard body) as 79.6 did -- the
    first client's refusal (client_refusal) when every client's main line is the placeholder, {} otherwise. Review:
    with several clients, one with a line keeps the method for its own; the others are said by
    refused_clients_notes, never dropped silently and never at the cost of another client's sections."""
    got = [client_refusal(ctx, plan) for ctx, plan in items if ctx is not None]
    return got[0] if got and all(got) else {}


NOT_APPLIED_CLIENT = "Nota: il metodo di analisi per elementi non è stato applicato alla posizione di {names}: {why}."
# my draft (v79.10.1, A1 review): a several-client parere where only some clients have a line


def refused_clients_notes(refused: Sequence[Tuple[str, Dict[str, Any]]]) -> List[str]:
    """v79.10.1 (A1, review): the notes for the clients of a several-client parere whose main line is the placeholder
    while another client's is not -- one note per why (the no-action sentence once), naming its clients, then their
    kept lines (the missing-laws line), each once."""
    by_why: Dict[str, List[str]] = {}
    lines: List[str] = []
    for name, r in refused:
        by_why.setdefault(str(r.get("why") or NOT_APPLIED_WHY["placeholder"]).rstrip("."), []).append(name)
        lines += [x for x in r.get("lines") or [] if x not in lines]
    return [NOT_APPLIED_CLIENT.format(names=_names(names), why=why) for why, names in by_why.items()] + lines


def no_action_problems(answer: str) -> List[str]:
    """v79.10.1 (A1): the no-action sentence at most once, never on a «Linea principale» line, never in a paragraph with
    a forecast («Esito probabile») -- [] when the answer keeps the rule (the nightly's check no_action_prose)."""
    text = answer or ""
    low = text.lower()
    out = []
    n = low.count(NO_ACTION_CORE)
    if n > 1:
        out.append(f"{n} volte")
    for para in re.split(r"\n\s*\n", text):
        if NO_ACTION_CORE not in para.lower():
            continue
        if any(NO_ACTION_CORE in ln.lower() and "linea principale" in ln.lower() for ln in para.split("\n")):
            out.append("come «Linea principale»")
        if re.search(r"esito\s+probabile", para, re.IGNORECASE):
            out.append("con una previsione")
    return out


# --- v79.12.5 (item 10): the two-turn cell -- «approfondisci il punto N» after the parere -----------------------------------

_POINT_ASK_RE = re.compile(r"\bpunto\s+(\d{1,2})\b", re.IGNORECASE)
_POINT_LINE_RE = re.compile(r"^\s*(\d{1,2})\.\s+(.+)$")
FOLLOWUP_SECTIONS_MAX = 2       # a follow-up printing more section heads than this is the whole parere again
FOLLOWUP_OTHER_SHARE = 0.6      # another point counts as covered when this share of its own words is in the follow-up


def asked_point(request: str) -> int:
    """The N of «approfondisci il punto N», 0 without one."""
    m = _POINT_ASK_RE.search(request or "")
    return int(m.group(1)) if m else 0


def answer_points(answer: str) -> Dict[int, str]:
    """{N: text} of §10's numbered points («Punti da verificare»: points_section)."""
    body = dict(answer_sections(answer or "")).get(title("verificare")) or ""
    return {int(m.group(1)): m.group(2).strip() for m in (_POINT_LINE_RE.match(x) for x in body.split("\n")) if m}


def followup_scope(first: str, second: str, request: str, records_before: Optional[Sequence[str]] = None,
                   records_after: Optional[Sequence[str]] = None) -> List[str]:
    """v79.12.5 (item 10): the follow-up to «approfondisci il punto N» checked against the parere -- [] when it keeps
    its scope: point N exists in §10; the follow-up is on that topic (its words, rule_statements.stems: two shared, or
    every one of a shorter point); only that topic (no other point's own words, FOLLOWUP_OTHER_SHARE of them, three at
    least); nothing unrequested (never the whole parere again -- more than FOLLOWUP_SECTIONS_MAX section heads --, no
    article cited that neither the parere nor the case's records hold); the case's records kept (records_before all in
    records_after, when both are given). Each problem in words."""
    from . import citations, rule_statements
    out: List[str] = []
    n = asked_point(request)
    points = answer_points(first)
    if not n or n not in points:
        return [f"punto {n or '?'} assente nel parere ({len(points)} punti)"]
    own = rule_statements.stems(points[n])
    said = rule_statements.stems(second)
    shared = len(own & said)
    if not (shared >= 2 or (own and shared == len(own))):
        out.append(f"fuori tema: {shared} parole del punto {n}")
    for j, text in points.items():
        unique = rule_statements.stems(text) - own
        if j != n and len(unique) >= 3 and len(unique & said) >= FOLLOWUP_OTHER_SHARE * len(unique):
            out.append(f"anche il punto {j}")
    heads = len(_SECTION_RE.findall(second or ""))
    if heads > FOLLOWUP_SECTIONS_MAX:
        out.append(f"il parere intero di nuovo ({heads} sezioni)")
    known = {citations.label(r) for r in citations.article_refs(first or "")} | {
        _norm_label(x) for x in records_before or []}
    extra = [x for x in dict.fromkeys(citations.label(r) for r in citations.article_refs(second or ""))
             if x not in known and _norm_label(x) not in known]
    if extra:
        out.append("articoli non richiesti: " + ", ".join(extra))
    if records_before is not None and records_after is not None:
        lost = [x for x in records_before if x not in set(records_after)]
        if lost:
            out.append("schede perse: " + ", ".join(lost))
    return out


def objections_plan(ctx: Dict[str, Any], plan: Dict[str, Any], call: Optional[Callable] = None,
                    review_call: Optional[Callable] = None) -> Dict[str, Any]:
    """v73: the objections written for the selected line -- one call after the selection (selection(): compose's own),
    given the main line, the subordinate and the barred list; an answer that argues a barred thesis (same_line) is
    dropped, logged. The plan with them as its «obiezioni» (none when the call fails: compose's fallbacks then, the
    record's and the selection's). Timed into ctx["timings"]["objections_s"], logged into the thesis log. v79.14.1
    (item 6): with every element of the main record «da provare» (strongest_open) the same call is asked for ONE
    objection on the strongest element (OBJECTIONS_ONE, STRONGEST_LINE), grounded in the record or the article, or
    none; the first objection of the reply on that element (on_element) is kept alone, into ctx["objection_all_open"]
    (compose then prints it only when the grounded-rule check removes nothing of it, and never adds a fallback)."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from .llm_reply import parse_json_reply
    sols, main, sub = selection(ctx, plan)
    if sols[main].get("placeholder"):   # v79.12.5 (item 2): no line to object to (no_claim, or the step-aside)
        return plan
    party = _party_of(ctx)              # v79.14.1 (item 6): the frame's client and role under CASE_FRAME
    who = OBJECTOR[party["side"]]       # v79 (E2): «l'accusa» in a criminal matter; the defence for a victim client
    system = OBJECTIONS_SYSTEM.format(who=who)
    # v79.14.1 (item 6): every element of the main record «da provare» -- the same call asks for ONE objection, on the
    # strongest element (strongest_open), grounded in the record's elements or the article, or none
    strong = strongest_open(ctx, sols, main)
    material = objections_material(ctx, sols, main, sub, party)
    if strong:
        system = system.replace(OBJECTIONS_THREE.format(who=who), OBJECTIONS_ONE.format(who=who))
        material += "\n\n" + STRONGEST_LINE.format(id=strong["id"], el=strong["text"], art=strong["article"],
                                                    why=STRONGEST_WHY[strong["why"]])
    rules = writer_rules_text(ctx.get("rito") or "", ctx.get("areas") or ())
    messages = [SystemMessage(content=system + ("\n\n" + rules if rules else "")),
                HumanMessage(content=material)]
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
    causation_replies(ctx, kept)                            # v79.12.7 (item 5): the subjective element, not the nexus
    if strong:          # v79.14.1 (item 6): the model's first objection on the strongest element, alone; else none
        pick = next((o for o in kept if on_element(f"{o.get('obiezione') or ''} {o.get('risposta') or ''}", strong)),
                    None)
        ctx["objection_all_open"] = {**strong, "written": bool(pick), "offered": len(kept)}
        kept = [pick] if pick else []
    seconds = round(time.time() - t0, 1)
    ctx.setdefault("timings", {})["objections_s"] = seconds
    entry = {"main": sols[main]["tesi"], "subordinate": sols[sub]["tesi"] if sub is not None else "",
             "barred": [t["tesi"] for t in excluded], "kept": len(kept), "dropped": dropped, "seconds": seconds,
             "error": error, "reply_binding": replies, "all_open": dict(ctx.get("objection_all_open") or {})}
    objections_log(ctx, messages, text, raw, entry)
    _log("objections", entry, seconds * 1000)
    return {**plan, "obiezioni": [{"obiezione": o["obiezione"], "risposta": o.get("risposta") or ""}
                                  for o in kept][:OBJECTIONS]}


def causation_replies(ctx: Dict[str, Any], objections: List[Dict[str, Any]]) -> int:
    """v79.12.7 (item 5): with a causation line capped (an ordinary consequence, no trigger fact: penal_lines.
    causation_cap), a reply denying the causal link (penal_lines.denies_causation) is replaced, in place, by the reply
    arguing the subjective element (penal_lines.CAUSATION_REPLY); the number replaced, into ctx (and logged)."""
    from . import penal_lines
    if not any(c.get("causation_capped") for c in ctx.get("candidates") or []):
        return 0
    n = 0
    for o in objections or []:
        if penal_lines.denies_causation(o.get("risposta") or ""):
            o["risposta"] = penal_lines.CAUSATION_REPLY + "."
            n += 1
    ctx["causation_replies_rewritten"] = int(ctx.get("causation_replies_rewritten") or 0) + n
    if isinstance(ctx.get("selection_counts"), dict):       # the state's thesis_counts (the same dict)
        ctx["selection_counts"]["causation_replies_rewritten"] = ctx["causation_replies_rewritten"]
    if n:
        _log("causation_replies", {"rewritten": n})
    return n


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


# v79.14.3 (item 3): «Ammettere la condotta, contestare l'inquadramento / contenere la pena» (the posture table's
# fallback_main) admits the conduct -- offered only when the verified frame says the client admits it. The frame
# (case_frame.FIELDS: parti, cliente, ruolo, rito, stadio, addebito_o_domanda, istituto, quesiti) has no field saying so,
# so the line is never generated -- with or without a held frame (no frame: no verified field says it either). Only a
# posture prepare marks ADMISSION_VERIFIED (frame_admits_conduct) offers it; the containment line that admits nothing
# (CONTAINMENT_LINE) takes its place everywhere else.
ADMISSION_VERIFIED = "admission_verified"


def frame_admits_conduct(state: Optional[Dict[str, Any]]) -> bool:
    """v79.14.3 (item 3): the held frame's verified fields say the client admits the conduct -- never: no field of the
    frame states it (see ADMISSION_VERIFIED)."""
    return False


def posture_fallback(posture: Dict[str, Any]) -> str:
    """The posture table's main line when every thesis is barred («fallback_main» of the cell's entry). v79.14.3 (item
    3): "" unless the posture is marked ADMISSION_VERIFIED (the verified frame says the client admits the conduct)."""
    if not (posture or {}).get(ADMISSION_VERIFIED):
        return ""
    return str(((posture or {}).get("entry") or {}).get("fallback_main") or "")


def fallback_thesis(a: Dict[str, Any], riti: Sequence[str], posture: Optional[Dict[str, Any]] = None
                    ) -> Dict[str, Any]:
    main = posture_fallback(posture or {})
    if not main and not (posture or {}).get(ADMISSION_VERIFIED) and ((posture or {}).get("entry") or {}).get("fallback_main") \
            and _family_key(posture or {}) in ("penale", ""):
        return {"tesi": CONTAINMENT_LINE,           # v79.14.3 (item 3): the containment line, admitting nothing
                "pro": "non contesta alcun elemento integrato" + (
                    f"; {riti_names(riti, 3)}" if riti_names(riti, 3) else ""),
                "contro": "presuppone l'accertamento della responsabilità", "realismo": ALTO, "minoritaria": False,
                "contesta": [], "poggia_su": [], "fallback": True}
    if main:
        return {"tesi": main[0].upper() + main[1:], "pro": "non contesta alcun elemento integrato" + (
                    f"; {riti_names(riti, 3)}" if riti_names(riti, 3) else ""),       # v79.10.3 (C1): admitted only
                "contro": "presuppone l'accertamento della condotta", "realismo": ALTO, "minoritaria": False,
                "contesta": [], "poggia_su": [], "fallback": True}
    civil = remedies().get("fallback", {}).get(str((posture or {}).get("role") or "")) \
        if _family_key(posture or {}) not in ("penale", "") else ""
    if civil:                   # v76: a civil role's line from the remedies table -- containment is penal-only
        return {"tesi": civil, "pro": "non nega i fatti narrati", "contro": "non porta al rigetto integrale della domanda",
                "realismo": MEDIO, "minoritaria": False, "contesta": [], "poggia_su": [], "fallback": True,
                "kind": "civil_fallback"}
    if a.get("obiettivo") == CONTENIMENTO and _family_key(posture or {}) in ("penale", ""):
        return {"tesi": CONTAINMENT_LINE,
                "pro": "non contesta alcun elemento integrato" + (
                    f"; {riti_names(riti, 3)}" if riti_names(riti, 3) else ""),       # v79.10.3 (C1): admitted only
                "contro": "presuppone l'accertamento della responsabilità", "realismo": ALTO, "minoritaria": False,
                "contesta": [], "poggia_su": [], "fallback": True}
    return {"tesi": "Tesi da costruire sugli elementi ancora da provare", "pro": "gli elementi aperti",
            "contro": "il materiale non basta a fondarla", "realismo": BASSO, "minoritaria": False, "contesta": [],
            "poggia_su": [], "fallback": True, "placeholder": True}     # v75: a failure state, never printed


# v79.10.1 (A3): the lines that hold the main line against a circumstance -- a contest, a defence, a riqualificazione
LEAD_KINDS = ("contest_excluded", "contest_open", "contest_integrated", "defence", "scriminante", "requalify")
CIRCUMSTANCE_KINDS = ("attenuante", "aggravante")
MAIN_DEMOTED_WHY = {"circumstance": "circostanza: mai linea principale finché esiste una contestazione, una difesa o "
                                    "una riqualificazione",
                    "fragment": "frammento di norma come «elemento escluso»: mai linea principale finché esiste una "
                                "contestazione, una difesa o una riqualificazione",
                    "attenuante": "attenuante: sempre subordinata",
                    # review: v79.7's containment line on a charge not fully integrato is the subordinate line
                    "containment": "contenimento con l'addebito non integrato: subordinato a una contestazione, a una "
                                   "difesa o a una riqualificazione sostenibile",
                    # v79.12.4 (items 4, 5): my drafts -- logged, never printed
                    "other_threat": "contestazione di una norma contro il cliente diversa dall'addebito: mai linea "
                                    "principale finché l'addebito ha una linea ammissibile",
                    "never_defence": "artt. 40 e 43 c.p.: i loro elementi provati sono dell'accusa, mai una difesa",
                    "weak_defence": "scriminante o difesa con meno della metà degli elementi provati: mai linea "
                                    "principale finché un'altra linea è ammissibile",
                    # v79.12.7 (item 1): my draft -- logged, never printed
                    "jurisprudential": "contenimento (riqualificazione, circostanze, pena): subordinato alla "
                                       "contestazione di un elemento di fonte giurisprudenziale ancora da provare",
                    # (review, v79.12.7 item 5): my draft -- logged, never printed
                    "causation_capped": "difesa sul nesso causale su una conseguenza ordinaria (realismo basso): mai "
                                        "linea principale finché un'altra linea è ammissibile"}
JURISPRUDENTIAL_SUBORDINATE = ("requalify", "contenimento") + ("attenuante", "aggravante")      # v79.12.7 (item 1)


def escluso_fragment(t: Dict[str, Any], known: Dict[str, Dict[str, Any]]) -> bool:
    """v79.10.1 (A3): a contest of an «escluso» element whose text is a fragment, no condition (rule_quote.fragment_why:
    a bare subject or object, a connective, a saving or scope clause, a modal permission). v79.12.4 (item 4): a contest
    of an open element («da provare») too -- contest_open as well as contest_excluded."""
    from . import rule_quote
    return t.get("kind") in ("contest_excluded", "contest_open") and any(
        rule_quote.fragment_why((known.get(x) or {}).get("text") or "") for x in t.get("contesta") or [])


# v79.12.4 (item 5): the general-part articles whose elements proven are the accusation's (causation, fault) -- never a
# defence line (the scriminante's or the defence's candidates, never_main)
NEVER_DEFENCE = ("40", "43")
DEFENCE_KINDS = ("defence", "scriminante")


def never_defence(label: str) -> bool:
    """v79.12.4 (item 5): art. 40 or 43 c.p. (any comma) -- its elements proven establish the charge, never a defence."""
    from .plan_then_write import _code_key
    number, _, code = _COMMA_TAIL_RE.sub("", str(label or "").strip()).partition(" ")
    return _code_key(code.strip()) == "c.p." and number in NEVER_DEFENCE


def line_article(t: Dict[str, Any], known: Dict[str, Dict[str, Any]]) -> str:
    """v79.12.4 (item 5): the line's own article -- its «article» when it carries one (a candidate), else (review: the
    soluzioni clean_method builds carry none) the article of the first element it rests on (poggia_su), else of the
    first it contests; "" for none."""
    return str(t.get("article") or "") or next((str((known.get(x) or {}).get("article") or "")
                                                for x in list(t.get("poggia_su") or []) + list(t.get("contesta") or [])
                                                if (known.get(x) or {}).get("article")), "")


def _line_elements(t: Dict[str, Any], known: Dict[str, Dict[str, Any]]) -> Tuple[int, int]:
    """v79.12.4 (item 5): (proven, total) of the elements of the line's own article (a scriminante's, a defence's)."""
    art = line_article(t, known)
    own = [e for e in known.values() if art and e.get("article") == art]
    return sum(1 for e in own if e.get("status") == PROVATO), len(own)


def half_proven(t: Dict[str, Any], known: Dict[str, Dict[str, Any]]) -> bool:
    """v79.12.4 (item 5): a scriminante or a defence line with at least half its elements proven (one with no element
    known is not judged: True)."""
    proved, total = _line_elements(t, known)
    return not total or proved * 2 >= total


def charge_articles(records: Dict[str, Dict[str, Any]], a: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None
                    ) -> List[str]:
    """v79.12.4 (item 4): the charge's articles -- the assessment's, the charge rule's (state["charge"]: its article and
    its exposure) and every charged record (rec["charged"]); [] when none is known."""
    charge = dict((ctx or {}).get("charge") or {})
    return [x for x in dict.fromkeys([str((a or {}).get("article") or ""), str(charge.get("article") or ""),
                                      str(charge.get("exposure") or "")] + _charged(records or {})) if x]


def other_threat_contest(t: Dict[str, Any], known: Dict[str, Dict[str, Any]], charges: Sequence[str]) -> bool:
    """v79.12.4 (item 4): a contest of a threat record that is not the charge -- every element it contests is on a
    record marked «threat» other than the charge's (charges: the charge, its exposure article)."""
    if not str(t.get("kind") or "").startswith("contest_") or not charges:
        return False
    arts = [(known.get(x) or {}) for x in t.get("contesta") or []]
    return bool(arts) and all(e.get("side") == "threat" and e.get("article") not in charges for e in arts)


def never_main(sols: Sequence[Dict[str, Any]], eligible: Sequence[int], known: Dict[str, Dict[str, Any]],
               all_integrated: bool = True, charges: Sequence[str] = ()) -> Dict[int, str]:
    """v79.10.1 (A3): {index: why} of the eligible lines kept out of the main line -- with one sample the realism is the
    model's alone (an attenuante's code argument «alto» led over a contest «medio»), so the kind decides first: while a
    contest, a defence or a riqualificazione is eligible, no circumstance (attenuante, aggravante) and no «elemento
    escluso» fragment leads; an attenuante never leads while any other line is eligible (the imputato's containment
    line always is). They stay among the soluzioni, subordinate. Review: with the charge not fully integrato
    (all_integrated False) the containment line -- v79.7's subordinate line there -- never leads over a non-minority
    contest, defence or riqualificazione either: the brief's test holds whatever the one sample's realism.
    v79.12.4: (item 4) a fragment contest (escluso_fragment: an open element's too) never leads while any other line is
    eligible -- the containment line counts, so a fragment never leads alone --; a contest of a threat record that is
    not the charge (other_threat_contest; charges: the charge's articles) never leads while the charge has an eligible
    line; (item 5) a defence or scriminante line on art. 40 or 43 c.p. never leads (never_defence), nor one with fewer
    than half its elements proven while another line is eligible (half_proven)."""
    circ = {i for i in eligible if sols[i].get("kind") in CIRCUMSTANCE_KINDS}
    frag = {i for i in eligible if escluso_fragment(sols[i], known)}
    other = {i for i in eligible if other_threat_contest(sols[i], known, charges)} - frag
    defences = {i for i in eligible if sols[i].get("kind") in DEFENCE_KINDS}
    not_def = {i for i in defences if never_defence(line_article(sols[i], known))}      # (review): by its elements
    weak = {i for i in defences - not_def if not half_proven(sols[i], known)}
    lead = [i for i in eligible if sols[i].get("kind") in LEAD_KINDS and i not in frag | other | not_def | weak]
    out: Dict[int, str] = {i: MAIN_DEMOTED_WHY["never_defence"] for i in not_def}
    if lead:
        out.update({i: MAIN_DEMOTED_WHY["circumstance"] for i in circ})
    if any(i not in frag | circ for i in eligible):     # v79.12.4 (item 4): the containment line counts as eligible
        out.update({i: MAIN_DEMOTED_WHY["fragment"] for i in frag})
    if any(i not in other | frag | circ | not_def for i in eligible):     # the charge has an eligible line
        out.update({i: MAIN_DEMOTED_WHY["other_threat"] for i in other if i not in out})
    if any(i not in weak | out.keys() for i in eligible):
        out.update({i: MAIN_DEMOTED_WHY["weak_defence"] for i in weak if i not in out})
    # (review, v79.12.7 item 5): a causation line capped at basso (an ordinary consequence, no trigger fact) never leads
    # while another line is eligible -- the containment line counts -- and never demotes the containment line
    capped = {i for i in eligible if sols[i].get("causation_capped") and i not in out}
    if any(i not in capped | out.keys() for i in eligible):
        out.update({i: MAIN_DEMOTED_WHY["causation_capped"] for i in capped})
    if not all_integrated and any(not sols[i].get("minoritaria") and i not in capped for i in lead):
        out.update({i: MAIN_DEMOTED_WHY["containment"] for i in eligible if sols[i].get("kind") == "contenimento"})
    others = [i for i in eligible if i not in circ and i not in out]
    if others:
        out.update({i: MAIN_DEMOTED_WHY["attenuante"] for i in circ
                    if sols[i].get("kind") == "attenuante" and i not in out})
    # v79.12.7 (item 1): while a sustainable contest of a jurisprudential element (not rated «basso») is eligible, the
    # containment lines -- a riqualificazione, a circumstance, the containment line -- never lead
    juris = [i for i in eligible if sols[i].get("jurisprudential") and not sols[i].get("minoritaria")
             and sols[i].get("realismo") != BASSO and i not in out]
    if juris:
        out.update({i: MAIN_DEMOTED_WHY["jurisprudential"] for i in eligible
                    if sols[i].get("kind") in JURISPRUDENTIAL_SUBORDINATE and i not in out})
    return out


def coherent_solutions(plan: Dict[str, Any], records: Dict[str, Dict[str, Any]], a: Dict[str, Any],
                       riti: Sequence[str] = (), log: Optional[List[Dict[str, Any]]] = None,
                       posture: Optional[Dict[str, Any]] = None, circ: Optional[Dict[str, Dict[str, Any]]] = None,
                       ctx: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], int]:
    """(solutions, index of the main one), by the ids each thesis declares (v71, rule 1 as the PO restated it):
      - a thesis declaring no element (contesta and poggia_su empty) is unverifiable (UNVERIFIABLE_LABEL): never main;
      - a thesis contesting an element the record marks «integrato» must say it is a minority one: labelled
        «minoritaria», it stays, realismo basso (INTEGRATED_NOTE); unlabelled, it is barred (BARRED_LABEL), never main;
      - a minority thesis is main only when the record offers no sustainable non-minority line -- and with the charge
        fully integrato the posture's «ammettere la condotta … contenere la pena» always is one (posture_fallback);
      - among the eligible, the model's choice when it is one, else the most realistic; none: the fallback line;
      - v79.10.1 (A3): never main -- a circumstance or an «elemento escluso» fragment while a contest, a defence or a
        riqualificazione is eligible, an attenuante while any other line is (never_main; marked «main_demoted»);
      - v79.12.1 (row 16): pertinence before realismo (ctx given: promote_by_pertinence) -- a line on a norm of the
        client's own outside the first read's institution, or a general one (1218, 2043, 1453), never leads over an
        eligible line inside it or more specific (a module article, a special comma): that line is promoted, marked
        «promoted_by_pertinence» with where it stood (subordinata, obiezione, risposta, soluzioni)."""
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
    charges = charge_articles(records, a, ctx)              # v79.12.4 (item 4): the charge's articles
    for t in sols:      # v79.12.4 (item 4): a contest of another threat record ranks after the charge's (subordinate_of)
        if other_threat_contest(t, known, charges):
            t["other_threat"] = True
    out_of_main = never_main(sols, eligible, known, bool(a.get("all_integrated")), charges)     # v79.10.1 (A3)
    for i, why in out_of_main.items():
        sols[i]["main_demoted"] = why
        if log is not None:
            log.append({"tesi": sols[i]["tesi"][:160], "main_demoted": why})
    eligible = [i for i in eligible if i not in out_of_main]
    majority =[i for i in eligible if not sols[i].get("minoritaria")]
    wanted = (plan.get("tesi_principale") or 1) - 1
    proven = lambda t: sum(1 for x in t.get("poggia_su") or [] if (known.get(x) or {}).get("status") == PROVATO)
    # v76: by realism, then by the proven elements it rests on, then the candidates' order -- a line with nothing
    # proven never leads over an «alto» one; the model's own pick only in the old path (no code candidates)
    by_realism = lambda i: (REALISMO.index(sols[i]["realismo"]), -proven(sols[i]), i)
    rank = lambda pool: sorted(pool, key=by_realism)
    if any(t.get("candidate") for t in sols):
        wanted = -1

    def add_fallback() -> int:
        sols.insert(0, fallback_thesis(a, riti, posture))
        if log is not None:
            log.append({"fallback": sols[0]["tesi"]})
        return 0
    if majority:
        main = wanted if wanted in majority else rank(majority)[0]
        main = promote_by_pertinence(sols, main, majority, plan, ctx, log, by_realism)       # v79.12.1 (row 16)
    elif a.get("all_integrated"):           # the containment line is a sustainable non-minority one: it goes first
        main = add_fallback()
    elif eligible:
        main = wanted if wanted in eligible else rank(eligible)[0]
        main = promote_by_pertinence(sols, main, eligible, plan, ctx, log, by_realism)       # v79.12.1 (row 16)
    else:
        main = add_fallback()
    return sols, main


# --- v79.12.1 (row 16): pertinence before realismo -----------------------------------------------------------------------

# the lines judged by pertinence: those resting on a norm of the client's own (a claim, a riconvenzionale, a remedy on
# the claim's article); a contest, a riqualificazione, a defence, the containment line, an eccezione are on the charge
# or the claim against the client -- the case's own institution by construction -- and are never moved by it
PERTINENCE_KINDS = ("claim", "riconvenzionale", "remedy")
PROMOTED_FROM = ("subordinata", "obiezione", "risposta", "soluzioni")


def first_institution_names(ctx: Dict[str, Any]) -> List[str]:
    """v79.12.1: the first read's institution -- its first institution and those of the same hypothesis (the same claim
    sentence), by name and rubricated name; a plain list of names: its first. [] with none."""
    items = [x for x in ctx.get("institutions") or [] if x]
    if not items:
        return []
    first = items[0]
    same = [x for x in items if isinstance(x, dict) and isinstance(first, dict) and not x.get("alternative")
            and str(x.get("claim") or "") == str(first.get("claim") or "")] if isinstance(first, dict) else [first]
    return institution_names(same or [first])


def _pertinence_modules(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    ids = list(ctx.get("claim_modules") or [])
    return [m for m in institution_modules().get("modules") or [] if m.get("id") in ids]


def inside_institution(label: str, ctx: Dict[str, Any]) -> Optional[bool]:
    """v79.12.1: the article is inside the first read's institution -- an action of an active module, a record the
    first read brought for it («primo esame: …»), filed under a topic of its name, rubricated as it, or of its family
    (article_families in the institution's read_families); None when neither an institution nor a module is known."""
    from .topic_step import name_match
    names = first_institution_names(ctx)
    mods = _pertinence_modules(ctx)
    if not names and not mods:
        return None
    if any(same_article(label, a) for m in mods for a in m.get("actions") or []):
        return True
    low = {_plain(n).lower() for n in names}
    rec = (ctx.get("records") or {}).get(label) or {}
    for k in rec.get("kept") or []:
        k = str(k)
        if k.startswith("primo esame:") and _plain(k.split(":", 1)[1]).lower() in low:
            return True
    if _plain(rec.get("topic")).lower() in low:
        return True
    if rec.get("rubrica") and any(name_match(n, str(rec.get("rubrica"))) for n in names):
        return True
    if names:
        fams = read_families({"institutions": names}) or []
        if set(article_families(label)) & set(fams):
            return True
    return False


def specificity(label: str, ctx: Dict[str, Any]) -> int:
    """v79.12.1: 0 a special comma's record or an active module's action, 2 a general norm (1218, 2043, 1453: lex
    generalis), 1 the rest."""
    rec = (ctx.get("records") or {}).get(label) or {}
    if rec.get("comma_of"):
        return 0
    group = claim_admission(label, _pertinence_modules(ctx))[1]
    return 0 if group == "module" else 2 if group == "general" or any(
        same_article(label, g) for g in institution_modules().get("general") or []) else 1


def line_pertinence(t: Dict[str, Any], ctx: Optional[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
    """v79.12.1: (outside, specificity) of a line judged by pertinence (PERTINENCE_KINDS) -- outside 1 when the first
    read's institution is known and the line's article is not inside it; None for a line it does not judge."""
    if not ctx:
        return None
    cand = next((c for c in ctx.get("candidates") or [] if t.get("candidate") and c.get("id") == t.get("candidate")), {})
    kind = cand.get("kind") or t.get("kind")
    art = str(cand.get("article") or t.get("article") or "")
    if kind not in PERTINENCE_KINDS or not art:
        return None
    return (1 if inside_institution(art, ctx) is False else 0, specificity(art, ctx))


def _named_in(text: str, t: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
    cand = next((c for c in ctx.get("candidates") or [] if t.get("candidate") and c.get("id") == t.get("candidate")), {})
    art = str(cand.get("article") or t.get("article") or "")
    number = art.split(" ", 1)[0]
    return bool(number) and bool(re.search(rf"\b(?:art(?:icolo|t)?\.?\s*)?{re.escape(number)}\b", str(text or "")))


def promote_by_pertinence(sols: List[Dict[str, Any]], main: int, pool: Sequence[int], plan: Dict[str, Any],
                          ctx: Optional[Dict[str, Any]], log: Optional[List[Dict[str, Any]]] = None,
                          by_realism: Optional[Callable[[int], Any]] = None) -> int:
    """v79.12.1 (row 16): the main line chosen first by pertinence -- when the line realismo chose is judged
    (line_pertinence) and another eligible line of the pool is more pertinent (inside the first read's institution
    against outside, then more specific), the most pertinent one leads -- review: among equally pertinent ones,
    realismo, then the proven count, then the pool's order decide (by_realism, coherent_solutions' rank key); it is
    marked «promoted_by_pertinence» {over, from} -- where it stood: the subordinate line, an
    objection or a reply of the plan naming its article, else among the soluzioni -- and logged."""
    if not ctx or main not in pool:
        return main
    if str((ctx.get("posture") or {}).get("role") or "") in VICTIM_ROLES:
        return main             # v79.12.4 (item 1): the victim's line is on the querela's offence -- never replaced
    have = line_pertinence(sols[main], ctx)
    if have is None:
        return main
    order = by_realism or (lambda i: pool.index(i))
    better = [(line_pertinence(sols[i], ctx), order(i), i) for i in pool if i != main]
    better = [(p, n, i) for p, n, i in better if p is not None and p < have]
    if not better:
        return main
    new = min(better)[2]
    where = "soluzioni"
    if subordinate_of(sols, main) == new:
        where = "subordinata"
    elif any(_named_in(o.get("obiezione") or "", sols[new], ctx) for o in plan.get("obiezioni") or []):
        where = "obiezione"
    elif any(_named_in(o.get("risposta") or "", sols[new], ctx) for o in plan.get("obiezioni") or []):
        where = "risposta"
    sols[new]["promoted_by_pertinence"] = {"over": sols[main].get("candidate") or sols[main]["tesi"][:80],
                                           "from": where}
    entry = {"tesi": sols[new]["tesi"][:160], "promoted_by_pertinence": where, "over": sols[main]["tesi"][:160]}
    if log is not None:
        log.append(entry)
    _log("promoted_by_pertinence", entry)
    return new


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


IRREVOCABLE = "querela irrevocabile"


def procedibility(records: Dict[str, Dict[str, Any]], rows: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    """{label: «ufficio» | «querela» | IRREVOCABLE} of the record's offences, from the procedural store
    (procedural_facts): a querela offence whose d'ufficio condition the record proves is d'ufficio; v79.10.3 (C1) one
    whose querela the store reads irrevocable is IRREVOCABLE (no remissione, no 162-ter)."""
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
        elif f.get("querela") and str(f.get("remissione") or "").startswith("querela irrevocabile"):
            out[label] = IRREVOCABLE        # v79.10.3 (C1): a querela no remissione extinguishes -- no 162-ter either
        elif f.get("querela"):
            out[label] = "querela"
    return out


def procedibility_out(text: str, procedibilita: Dict[str, str], charge: str, log: Optional[List[str]] = None,
                      penal: bool = False) -> str:
    """A line or sentence on remissione / 162-ter dropped when it is about a d'ufficio offence: the offences it names
    (the remissione's own articles aside), else the charge. v79.10.3 (C1): in a criminal matter (penal), dropped unless
    the store says every offence it is about is a querela offence (procedibility: «querela») -- d'ufficio, an
    irrevocable querela or nothing read alike, and no charge read at all; outside one, as before (only a d'ufficio
    offence's)."""
    if not procedibilita and not penal:
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
                if (not about or any(procedibilita.get(x) != "querela" for x in about)) if penal \
                        else about and all(procedibilita.get(x) == "ufficio" for x in about):   # (review) no charge read
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
_RITE_KEEP = "\x01"


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

        text = text.replace(RITE_TO_VERIFY, _RITE_KEEP)     # v79.10.3 (C1): the rito's own words stay on its line
        for pattern, repl in ((_PAREN_RE, paren), (_COLON_RE, colon), (_TAIL_RE, tail), (_COPULA_RE, copula),
                              (_BARE_RE, bare)):
            text = pattern.sub(repl, text)
        text = text.replace(_RITE_KEEP, RITE_TO_VERIFY)
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
# v79.12.7 (item 7): §6's strategy names the main line the table above prints (once per section) -- my draft
STRATEGY_MAIN_IN_TABLE = "Linea principale: la tesi indicata come principale nella tabella"
# v79.12.7 (item 7): a later answer of «Risposte ai quesiti» on the main line refers to the first that prints it
MAIN_LINE_SEE_ANSWER = "La linea principale è quella indicata nella risposta {n}."


def cenni_title(title_: str) -> bool:
    """v79.12.7 (item 7): a «Cenni: …» section's title (interaction_copy.CENNI_SECTION)."""
    from . import interaction_copy as C
    return str(title_ or "").strip().startswith(C.CENNI_SECTION.split("{")[0].strip())


def solutions_section(sols: List[Dict[str, Any]], main: int, plan: Dict[str, Any], riti: Sequence[str],
                      stragiudiziale: str, log: Optional[Dict[str, Any]] = None, post_fact: Sequence[str] = ()
                      ) -> str:
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
    # v79.12.7 (item 7): the table prints the main line -- the strategy names it, never prints it again
    strategy = [STRATEGY_TITLE, "", f"- {STRATEGY_MAIN_IN_TABLE}."]
    if sols[main].get("jurisprudential"):       # v79.12.7 (item 1): the element's source, a ruling's passage under it
        from . import penal_lines
        strategy += [f"  - {x.rstrip('.')}." for x in penal_lines.element_source_lines(sols[main])]
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
    strategy += [f"- {x.rstrip('.')}." for x in post_fact or []]       # v79.12.7 (item 3): a soccorso narrated
    strategy.append(f"- Opzione stragiudiziale: {(stragiudiziale or '').rstrip('.')}.")
    return "\n".join(lines + [""] + strategy)


def objections_section(plan: Dict[str, Any], fallback: Sequence[Dict[str, str]] = (),
                       verified: Sequence[Dict[str, str]] = ()) -> str:
    """§7. v79.15.1 (item 3, ANALYSIS_STEP): `verified` -- the verified analysis's objections
    (analysis_step.objection_items: every non-client allegation behind a «contestato» element, a claimant's exception
    elements) -- printed first and never cut; the plan's (and the fallback's) fill up to OBJECTIONS after them; the
    «no objections» line only when there is none of either."""
    items = list(verified or [])
    for o in plan.get("obiezioni") or []:
        if len(items) >= max(OBJECTIONS, len(verified or [])):
            break
        items.append(o)
    for f in fallback:
        if len(items) >= OBJECTIONS:
            break
        if f.get("obiezione") and all(f["obiezione"] != x["obiezione"] for x in items):
            items.append(f)
    items = items[:max(OBJECTIONS, len(verified or []))]
    if not items:
        return "Il materiale non indica obiezioni della controparte sui fatti narrati."
    return "\n\n".join(f"{OBJECTION_HEAD.format(n=i)} {o['obiezione'].rstrip('.')}.\n{ANSWER_HEAD} "
                         f"{o.get('risposta') or NO_ANSWER}" for i, o in enumerate(items, 1))


OBJECTION_HEAD, ANSWER_HEAD = "**Obiezione {n}.**", "**Risposta.**"

# --- v79.19.1 (items 6b, 6c): nothing the fit accepted goes unprinted; the rulings printed, counted -------------------
UNPRINTED_HEAD = "Altre norme applicabili ai fatti"       # my draft: the one-line home of a fitted record no block named


def fit_accepted(state: Dict[str, Any]) -> List[str]:
    """v79.19.1 (item 6b): the articles the applicability judgment accepted -- a verified «si» or «dipende»
    (state["applicability"]["judgments"]); [] without the judgment."""
    fit = (state or {}).get("applicability") if isinstance(state, dict) else None
    out: List[str] = []
    for j in (fit or {}).get("judgments") or []:
        if isinstance(j, dict) and j.get("verified") and j.get("article") and str(j.get("fit")) in ("si", "dipende"):
            label = str(j["article"])
            if label not in out:
                out.append(label)
    return out


def _names_article(text: str, label: str) -> bool:
    """The text names the article -- «art. 1218 c.c.», «Art. 1218 c.c.», «artt. 1218 e 1223 c.c.» (its number with the
    code), «**Art. 1218 c.c.**»; the label's spaces any whitespace."""
    if not text or not label:
        return False
    number, _, code = str(label).partition(" ")
    sep = r"\s*(?:,|\be\b|\bed\b|\bo\b)\s*"
    others = r"(?:\d+(?:-\w+)?" + sep + ")*"           # «artt. 1218 e 1223 c.c.» names both
    pattern = others + re.escape(number) + r"(?![\w-])" + (
        r"(?:" + sep + r"\d+(?:-\w+)?)*\s+" + re.escape(code).replace(r"\ ", r"\s+") if code else "")
    return re.search(r"(?i)\bartt?(?:\.|icol[oi])?\s*" + pattern + r"(?![\w-])", text) is not None


def unprinted_records(ctx: Dict[str, Any], records: Dict[str, Dict[str, Any]], text: str,
                      excluded: Sequence[str] = ()) -> List[str]:
    """v79.19.1 (item 6b): the fit-accepted articles (ctx["fit_accepted"]) that are a printed record (records: the
    answer's, not excluded, not removed by the fit) and that the §4 text does not name -- each as «art. X (rubrica)»."""
    out: List[str] = []
    for label in ctx.get("fit_accepted") or []:
        rec = next((r for x, r in (records or {}).items() if _same_label(x, label)), None)
        if not isinstance(rec, dict) or rec.get("dropped") or rec.get("ranked_candidate") or rec.get("unprinted"):
            continue
        if any(_same_label(x, label) for x in excluded or ()) or frame_excluded(rec):
            continue
        if _names_article(text or "", label):
            continue
        rubrica = _short(str(rec.get("rubrica") or ""), 80).rstrip(".")
        out.append(f"art. {label}" + (f" ({rubrica})" if rubrica else ""))
    return out


def rulings_printed(ctx: Dict[str, Any], rulings: Sequence[Any], excluded: Sequence[str] = ()) -> int:
    """v79.19.1 (item 6c): how many rulings §5 prints -- the registry record's kept ones (RULINGS_RECORD), else the
    retrieved rulings with an id and a massima; those on an excluded article left out, as orientations_text does."""
    def gone(label: Any) -> bool:
        return any(_same_label(x, str(label or "")) for x in excluded or ())
    rec = ctx.get("rulings_record")
    if rec:
        return sum(1 for r in rec.get("kept") or [] if isinstance(r, dict) and not gone(r.get("article")))
    return sum(1 for r in rulings or [] if isinstance(r, dict) and r.get("id") and r.get("massima")
               and not gone(r.get("article")))

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

# --- v79.14.1 (item 6): every element «da provare» -- one model-written objection on the strongest element, or none ----
ALL_OPEN_PREFIX = "Tutti gli elementi sono ancora da provare"
ALL_OPEN_ONE = "Tutti gli elementi sono ancora da provare: l'obiezione più forte {who} riguarda l'elemento «{el}»."
ALL_OPEN_NONE = ("Tutti gli elementi sono ancora da provare: nessuna obiezione {who} trova oggi fondamento negli "
                 "elementi della norma o negli atti.")                    # my drafts; both open with ALL_OPEN_PREFIX
ALL_OPEN_WHO = {"penale": "dell'accusa", "civile": "della controparte",
                "vittima": "della difesa dell'imputato",      # a penal client in VICTIM_ROLES: the defence objects
                "ignoto": "dell'altra parte"}                 # the frame's verified role «sconosciuto»: neither side
# who objects, in the objections call (OBJECTIONS_SYSTEM's {who}); the same keys as ALL_OPEN_WHO
OBJECTOR = {"penale": "l'accusa", "civile": "la controparte", "vittima": "la difesa dell'imputato",
            "ignoto": "l'altra parte"}
# the call's sentence asking for three, and the one replacing it when every element is open (my draft)
OBJECTIONS_THREE = "Scrivi le tre obiezioni più forti che {who} opporrà alla LINEA PRINCIPALE, ciascuna con la risposta."
OBJECTIONS_ONE = ("Tutti gli elementi della norma principale sono da provare: scrivi UNA sola obiezione che {who} "
                  "opporrà alla LINEA PRINCIPALE, sull'ELEMENTO PIÙ FORTE indicato, con la risposta. Ogni regola che "
                  "enunci, nell'obiezione o nella risposta, deve leggersi negli elementi delle SCHEDE o nel testo "
                  "dell'articolo; se nessuna obiezione è fondata così, rispondi {{\"obiezioni\": []}}.")
STRONGEST_LINE = "ELEMENTO PIÙ FORTE: {id} «{el}» dell'art. {art} ({why})"
STRONGEST_WHY = {"main_line": "l'elemento su cui si decide la linea principale",
                 "facts": "l'elemento che i fatti narrati toccano di più",
                 "distinctive": "l'elemento che distingue la fattispecie",
                 "first": "il primo elemento della norma"}
# a per-element «da provare» template (OBJ_DENIED, OBJ_OPEN, DEFENCE_WORDS["obj_open"])
OPEN_TEMPLATE_RE = re.compile(r"\b(?:negherà|cercherà\s+di\s+provare)\s+l['’]elemento\s+«([^»\n]*)")


def all_open(ctx: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """v79.14.1 (item 6): (the main record's label, its live elements) when every one of them is «da provare» -- the
    record of the assessment's article (the charge or the claim), at least one element; ("", []) otherwise."""
    label = str((ctx.get("assessment") or {}).get("article") or "")
    rec = (ctx.get("records") or {}).get(label) or {}
    live = [r for r in _live(rec) if r.get("id")]
    if not live or any((r.get("status") or DA_PROVARE) != DA_PROVARE for r in live):
        return "", []
    return label, live


def strongest_open(ctx: Dict[str, Any], sols: Sequence[Dict[str, Any]] = (), main: Optional[int] = None
                   ) -> Dict[str, str]:
    """v79.14.1 (item 6): with every element of the main record «da provare» (all_open), the strongest one -- {id,
    text, article, why}; {} otherwise. The pool: the elements of that record the selected main line contests, else
    those it rests on, else every live one. The strongest in the pool: the only one the main line turns on
    («main_line»); else the one the most narrated facts touch (its «supporting» sentences, one at least: «facts»); else
    the record's distinctive element («distinctive»); else the first the main line turns on («main_line») or the
    record's first («first»)."""
    label, live = all_open(ctx)
    if not live:
        return {}
    by_id = {str(r["id"]): r for r in live}
    m = sols[main] if sols and main is not None and 0 <= main < len(sols) else {}
    pool = [by_id[str(x)] for x in m.get("contesta") or [] if str(x) in by_id] or \
        [by_id[str(x)] for x in m.get("poggia_su") or [] if str(x) in by_id]
    from_line = bool(pool)
    pool = list({id(r): r for r in pool}.values()) or live
    facts = lambda r: len(r.get("supporting") or [])
    best = max(pool, key=lambda r: (facts(r), bool(r.get("distinctive")), -pool.index(r)))
    if from_line and len(pool) == 1:
        why = "main_line"
    elif facts(best):
        why = "facts"
    elif best.get("distinctive"):
        why = "distinctive"
    else:
        why = "main_line" if from_line else "first"
    return {"id": str(best["id"]), "text": _short(best.get("text"), 100), "article": label, "why": why}


def on_element(text: str, strong: Dict[str, str]) -> bool:
    """v79.14.1 (item 6): an objection (with its reply) is on the strongest element -- its id, or one of its own nouns
    (issue_records.element_nouns)."""
    from . import issue_records
    if not strong:
        return False
    if re.search(rf"(?<![\w]){re.escape(strong['id'])}(?![\w])", text or ""):
        return True
    return bool(issue_records.element_nouns(strong.get("text") or "") & issue_records.element_nouns(text or ""))


def objection_party(state: Optional[Dict[str, Any]], client: str, posture: Optional[Dict[str, Any]]
                    ) -> Dict[str, Any]:
    """v79.14.1 (item 6): for whom the objections are written -- {client, role, side, frame}. The method's client and
    the posture's role; with CASE_FRAME on and a verified frame held (case_frame.held), the frame's client and role:
    for the frame's own client (or the only client when the frame holds none; never a client of a multi-client answer
    when the frame binds no client) the frame's verified role, a different
    posture role logged frame_conflict («objections_plan») and losing; with one client only (no state["method_roles"])
    a client other than the frame's is logged the same way and the frame's client written for. Another client of a
    multi-client answer keeps its own (role_for_client's) role. «side» (objection_side): ALL_OPEN_WHO's and OBJECTOR's
    key."""
    posture = posture or {}
    out = {"client": str(client or ""), "role": str(posture.get("role") or ""), "frame": False}
    out["side"] = objection_side(posture, out["role"])
    from . import case_frame
    if not isinstance(state, dict) or case_frame.held(state) is None:
        return out
    f_client = case_frame.bound(state, "cliente") or ""
    f_role = case_frame.bound(state, "ruolo") or ""
    several = bool(state.get("method_roles"))
    if several and not (f_client and out["client"] and case_frame.same_value("cliente", out["client"], f_client)):
        return out          # another client, or a frame binding no client: its own role (role_for_client's), unlogged
    if f_client and out["client"] and not case_frame.same_value("cliente", out["client"], f_client):
        case_frame.conflict(state, "cliente", "objections_plan", out["client"], f_client)
    if f_client:
        out["client"] = f_client
    if f_role:
        if out["role"] and out["role"] != f_role:
            case_frame.conflict(state, "ruolo", "objections_plan", out["role"], f_role)
        out["role"] = f_role
    out["frame"] = True
    out["side"] = "ignoto" if case_frame.role_unknown_held(state) else objection_side(posture, out["role"])
    return out


def objection_side(posture: Optional[Dict[str, Any]], role: str = "") -> str:
    """v79.14.1 (item 6): who objects to the client's main line (ALL_OPEN_WHO's and OBJECTOR's key): «vittima» a penal
    client in VICTIM_ROLES (the imputato's defence), «penale» another penal client (the accusa), «civile» the
    controparte (_other_side's reading)."""
    if not criminal(posture):
        return "civile"
    return "vittima" if str(role or (posture or {}).get("role") or "") in VICTIM_ROLES else "penale"


def _party_of(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """The ctx's objection_party (prepare's), else the method's client and the posture's role (a ctx built by hand)."""
    got = ctx.get("objection_party")
    return got if isinstance(got, dict) and got.get("side") else objection_party(None, ctx.get("client") or "",
                                                                                 ctx.get("posture"))


def all_open_section(strong: Dict[str, str], objection: Optional[Dict[str, str]], penal: Any) -> str:
    """v79.14.1 (item 6): §7 with every element open -- the one objection, said on which element, or the line saying
    none is grounded; never the per-element template. «penal»: the objecting side's key (objection_side), or a bool."""
    side = penal if isinstance(penal, str) and penal in ALL_OPEN_WHO else ("penale" if penal else "civile")
    who = ALL_OPEN_WHO[side]
    if not objection:
        return ALL_OPEN_NONE.format(who=who)
    return ALL_OPEN_ONE.format(who=who, el=strong["text"]) + "\n\n" + objections_section({"obiezioni": [objection]})


def template_repeats(answer: str) -> int:
    """v79.14.1 (item 6): the per-element «da provare» templates repeated in each «Obiezioni e risposte» (each client's
    part on its own) -- every one where §7 is the all-open layout (ALL_OPEN_PREFIX: one model-written objection or
    none, never the template), and elsewhere (a mixed record: each open element its own objection) the same element's
    template printed again: must be 0."""
    n = 0
    for t, body in answer_sections(answer or ""):
        if t != title("obiezioni"):
            continue
        for part in re.split(r"(?m)^\*\*Nell'interesse di [^*\n]+\*\*\s*$", body):
            found = [" ".join(x.lower().split()) for x in OPEN_TEMPLATE_RE.findall(part)]
            n += len(found) if ALL_OPEN_PREFIX in part else len(found) - len(set(found))
    return n


def objections_fallback(records: Dict[str, Dict[str, Any]], a: Dict[str, Any], penal: bool,
                        by_id: Dict[str, Dict[str, Any]], titles_: Sequence[str] = ()) -> List[Dict[str, str]]:
    """Objections from the record when the method plan gives fewer than three: defending, the main charge's proven
    elements (conceded, answered on qualification and pena) and its open ones (the burden); claiming, the client's
    open elements (the proof to acquire); then our objections' titles. v79.14.1 (item 6, review): used only when some
    element is proven -- with every element open compose prints all_open_section instead, so each open element of a
    mixed record keeps its own objection (79.14.0's)."""
    out = []
    label = a.get("article") or ""
    rec = (records or {}).get(label) or {}
    family = "penale" if penal else "civile"
    defence = defence_record(label, rec)            # v79.12.1: the defence record's own objections
    for r in _live(rec):
        el = _short(r.get("text"), 100)
        if defence and r.get("status") == PROVATO:
            q, ans = DEFENCE_WORDS["obj_proven"]
            out.append({"obiezione": q.format(el=el, art=label),
                        "risposta": ans.format(fact=(_sentence_text(r.get("supporting") or [], by_id)
                                                     or "i fatti narrati").rstrip("."))})
        elif defence and r.get("status") == DA_PROVARE:
            q, ans = DEFENCE_WORDS["obj_open"]
            out.append({"obiezione": q.format(el=el, art=label),
                        "risposta": ans.format(missing=_short(r.get("missing") or r.get("text"), 120).rstrip("."))})
        elif defence:
            continue
        elif rec.get("side") == "threat" and r.get("status") == PROVATO:
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
                      points: Optional[List[str]] = None, excluded: Sequence[str] = (),
                      dropped: Optional[List[str]] = None) -> str:
    """§5: v78 (RULINGS_RECORD), the registry's -- the rulings with their quoted passage, the computed orientations, the
    lawyer's pronunce di riferimento, the corpus line (rulings.section_text); else v71's (orientations_section).
    v79.14.3 (item 4): never a ruling, an orientation or a point on an excluded article -- the record read through a view
    without them (the rulings' own record untouched), the v71 points and rulings filtered alike; dropped gets their
    articles."""
    if excluded:
        on = lambda label: next((x for x in excluded if _same_label(x, str(label or ""))), "")     # noqa: E731
        if ctx.get("rulings_record"):
            rec = ctx["rulings_record"]
            gone = [on(r.get("article")) for r in rec.get("kept") or []] + \
                [on(o.get("article")) for o in rec.get("orientations") or []] + [on(k) for k in rec.get("landmarks") or {}]
            if dropped is not None:
                dropped += [x for x in gone if x]
            view = {**rec, "kept": [r for r in rec.get("kept") or [] if not on(r.get("article"))],
                    "orientations": [o for o in rec.get("orientations") or [] if not on(o.get("article"))],
                    "landmarks": {k: v for k, v in (rec.get("landmarks") or {}).items() if not on(k)}}
            from . import rulings as rulings_mod
            return rulings_mod.section_text(view, points)
        kept = [r for r in rulings or [] if not (isinstance(r, dict) and on(r.get("article")))]
        points_in = [o for o in plan.get("orientamenti") or [] if not cites_excluded(
            f"{o.get('punto') or o.get('tema') or ''} {o.get('pronuncia') or ''}", excluded)]
        if dropped is not None:
            dropped += [on(r.get("article")) for r in rulings or [] if isinstance(r, dict) and on(r.get("article"))]
            dropped += [x for o in plan.get("orientamenti") or [] if o not in points_in for x in cites_excluded(
                f"{o.get('punto') or o.get('tema') or ''} {o.get('pronuncia') or ''}", excluded)]
        return orientations_section({**plan, "orientamenti": points_in}, kept, log, points)
    if ctx.get("rulings_record"):
        from . import rulings as rulings_mod
        return rulings_mod.section_text(ctx["rulings_record"], points)
    return orientations_section(plan, rulings, log, points)


# v79.10.6 (E1): a question put as a riqualificazione of the charge («sia qualificabile come percosse», «derubricazione»,
# «anziché», «in luogo del») -- the only way the read's alternative stands as the question without naming the charge
_REQUALIFY_RE = re.compile(r"(?i)\b(?:ri)?qualific|\bderubric|\banzich[eé]\b|\bin\s+luogo\s+d|\bin\s+alternativa\b|"
                           r"\bdiversa\s+(?:fattispecie|ipotesi)\b")


def other_charge(text: str, charge: str, records: Dict[str, Dict[str, Any]]) -> List[str]:
    """v79.10.5 (E1): the fattispecie a text names as its question that are not the held charge, when it does not name
    the charge -- [] when it names the charge (an alternative beside it is no replacement) or names no fattispecie.
    Named by number, or in words by a record's rubrica («il reato di truffa»). The client's own line beside the charge
    is no replacement either: a qualification the read put forward (rec["alternative"], a riqualificazione) and, outside
    the codes of offences, an article on the other side from the charge (the convenuto's exception to the claim, the
    other side's exception to the client's). v79.10.6 (E1): only a replacement of the charge for the same conduct is
    named -- a defence never: a riqualificazione the record carries (rec["alternative"] or rec["alternative_facts"]) when
    the text puts it as one (_REQUALIFY_RE: «qualificabile come», «derubricazione» -- «se i fatti integrino il reato di
    truffa» with the charge on 646 is a replacement, refused), a norm that only limits or measures a claim
    (data/institution_modules.json «limiting»: 1460, 1227, 1455 ...), and a scriminante or another uncounted article
    (issue_records.counted) pass through."""
    from . import issue_records, posture as posture_mod
    from .topic_step import GENERIC_STEMS
    if not charge or not text:
        return []
    cited = posture_mod.sentence_articles(text)
    if any(x == charge or same_article(x, charge) for x in cited):
        return []
    words = posture_mod._stems(text)

    def in_words(label: str) -> bool:
        stems = posture_mod._stems(str((records.get(label) or {}).get("rubrica") or ""))
        return bool(stems - GENERIC_STEMS) and stems <= words
    if in_words(charge):
        return []
    held = records.get(charge) or {}

    def defence(label: str) -> bool:
        rec = records.get(label) or {}
        if (rec.get("alternative") or rec.get("alternative_facts")) and _REQUALIFY_RE.search(text):
            return True             # v79.10.6 (E1): put as a riqualificazione, not as the question on its own
        if any(same_article(label, spec) for spec in (institution_modules().get("limiting") or {})):
            return True
        return not posture_mod.penal_article(label) and bool(rec.get("side")) and bool(held.get("side")) \
            and rec.get("side") != held.get("side")
    named = list(dict.fromkeys(cited + [x for x in records if x != charge and in_words(x)]))
    return [x for x in named if issue_records.counted(x, str((records.get(x) or {}).get("rubrica") or ""))
            and (x in records or label_code(x) == label_code(charge)) and not defence(x)]


def question_section(plan: Dict[str, Any], a: Dict[str, Any], records: Dict[str, Dict[str, Any]], client: str,
                     charge: str = "", log: Optional[Dict[str, Any]] = None) -> str:
    """§3, the legal question: the thesis call's, else the code's on the article. v79.10.5 (E1): with a charge held by
    rule (charge, in the records), the code's question is on it -- never on the main line's article the forecast moved
    to --, and the thesis call's question naming another fattispecie and not the charge is refused (other_charge: the
    code's question instead, logged in log["charge_refused"])."""
    held = charge if charge and charge in (records or {}) else ""
    if plan.get("questione_giuridica"):
        replaced = other_charge(str(plan["questione_giuridica"]), held, records or {})
        if not replaced:
            return plan["questione_giuridica"]
        if log is not None:
            log["charge_refused"] = {"charge": held, "named": replaced, "text": str(plan["questione_giuridica"])[:200]}
    label = held or a.get("article") or ""
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
                    client: str, charge: str = "", records: Optional[Dict[str, Dict[str, Any]]] = None,
                    log: Optional[Dict[str, Any]] = None) -> str:
    """The method plan's summary, at most SUMMARY_LINES sentences; without one, the main line and the record's
    assessment. v79.10.6 (E1): with a charge held by rule (charge, in records), a plan's summary naming another
    fattispecie and not the charge (other_charge) is refused like §3's question -- the code's summary instead, logged in
    log["charge_refused_summary"]. (compose passes no plan: §1 is the code's.)"""
    text = _plain(plan.get("sintesi"))
    held = charge if charge and charge in (records or {}) else ""
    replaced = other_charge(text, held, records or {}) if text else []
    if replaced and log is not None:
        log["charge_refused_summary"] = {"charge": held, "named": replaced, "text": text[:200]}
    if text and not replaced:
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
    # v79.12.4 (item 4): a contest of a threat record that is not the charge ranks after the charge's lines
    return sorted(pool, key=lambda i: (bool(sols[i].get("other_threat")), REALISMO.index(sols[i]["realismo"])))[0] \
        if pool else None


def _open_ids(ctx: Dict[str, Any]) -> List[str]:
    return [m["elemento"] for m in (ctx.get("facts") or {}).get("mancanti") or [] if m.get("elemento")]


def whole_citations(text: str, log: Optional[List[str]] = None) -> str:
    """v79.9 (D4): a ruling's reference printed whole or not at all -- every reference without its court's head
    («Sez. 4, n. 29558/2025», «4, n. 29558/2025») removed from the sentence it sits in (logged); §5 prints the
    record's estremi whole by construction. v79.12.7 (item 7): the reference's date with it, a date left orphaned before
    a closing parenthesis («…, 11.04.2025)») too, and never inside a «…» quote (text_hygiene.citation_spans)."""
    from . import text_hygiene
    if not text_hygiene.citation_spans(text or ""):
        return text or ""
    lines = []
    for line in text.split("\n"):          # line by line: a span never takes a line break, a line untouched stays as is
        spans = text_hygiene.citation_spans(line)
        out = line
        for a, b in reversed(spans):
            if log is not None:
                log.append(out[a:b].strip())
            gap = " " if out[a - 1:a].isalnum() and out[b:b + 1].isalnum() else ""     # never two words joined
            out = out[:a] + gap + out[b:]
        if spans:
            indent = out[:len(out) - len(out.lstrip())]
            out = re.sub(r"\(\s*\)|\[\s*\]", "", out)
            out = re.sub(r"[ \t]+([,.;:])", r"\1", out)
            out = re.sub(r"[,:;]\s*([.;:])", r"\1", out)
            out = indent + re.sub(r"[ \t]{2,}", " ", out.lstrip())
        lines.append(out)
    return "\n".join(lines)


def _with_line(text: str, line: Optional[str]) -> str:
    return f"{text}\n\n{line}" if line and text else (text or "")


# v79.9 (B2): a module whose special laws the graph lacks -- one fixed line in §1 and §9 (the PO's words). v79.12.1:
# once per answer (§1, or the step-aside prose's lines) and only for a module that fits (module_fits)
MISSING_LAWS_LINE = ("La disciplina speciale di {istituto} ({leggi}) non è nell'archivio: il parere si limita alle "
                     "norme del codice")


def module_fits(module: Dict[str, Any], ctx: Dict[str, Any]) -> str:
    """v79.12.1: "" when an active module fits the case, else why not -- switched on by the first read's institution or
    a module article the request names, it fits; by the user's words or a schema only outside a criminal matter (a
    penal narrative's words are its facts, not the institution) and, when the first read names institution families
    (read_families), only as one of them."""
    if module.get("signal") in ("istituto", "articolo"):
        return ""
    if criminal(ctx.get("posture")):
        return "materia penale: il modulo viene dalle sole parole del caso"
    families = read_families(ctx)
    if families and str(module.get("id")) not in families:
        return f"fuori dall'istituto della prima lettura ({', '.join(families)})"
    return ""


def missing_laws(ctx: Dict[str, Any], known: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """v79.9 (B2): {"istituto", "leggi"} of the first active module with a law the graph lacks (corpus_laws); {} with
    none or when the graph's names are unknown. v79.12.1: only a module that fits (module_fits; the others logged,
    missing_laws_unfit), with the signal that switched it on."""
    from . import corpus_laws
    unfit = {}
    for m in active_modules(ctx):
        gone = corpus_laws.missing(m.get("laws") or [], known) if m.get("laws") else []
        if not gone:
            continue
        why = module_fits(m, ctx)
        if why:
            unfit[m["id"]] = why
            continue
        if unfit:
            _log("missing_laws_unfit", unfit)
        return {"istituto": str(m.get("name") or m["id"]), "leggi": list(gone), "module": m["id"],
                "signal": str(m.get("signal") or "")}
    if unfit:
        _log("missing_laws_unfit", unfit)
    return {}


def missing_laws_line(ctx: Dict[str, Any]) -> str:
    gone = ctx.get("missing_laws") or {}
    return MISSING_LAWS_LINE.format(istituto=gone["istituto"], leggi=", ".join(gone["leggi"])) + "." if gone else ""


# v79.8 (E3): a civil case whose certain facts describe a reato -- one line in §1 and §9, never a second parere
PENAL_ROUTES = (("646 c.p.", "appropriazione indebita", r"\bsi\s+(?:è\s+)?appropriat\w*|\btrattien\w*\s+(?:per\s+s[ée]\s+)?"
                 r"(?:il|la|i|le|l['’])\s*(?:denar|somm|ben[ei]|merc|auto|vettur)|\bsi\s+rifiut\w*\s+di\s+restituir"),
                ("612 c.p.", "minaccia", r"\bminacci(?:at|av|ò|a\b|ando)\w*"),
                ("582 c.p.", "lesione personale", r"\b(?:ferit[oa]|aggredit[oa]|picchiat[oa]|frattur\w*|lesion[ie]\s+"
                 r"personal\w*)\b"),
                ("635 c.p.", "danneggiamento", r"\bha\s+(?:volontariamente\s+)?(?:distrutto|danneggiato|vandalizzat\w*)\b"),
                ("640 c.p.", "truffa", r"\btruff\w*|\braggir\w*|\bartifiz\w*"))
PENAL_ROUTE_LINE = ("I fatti narrati possono integrare anche il reato di {name} (art. {art}), {proc}: la via penale è "
                    "distinta da questo parere.")       # my drafts (v79.8, E3): §1
PENAL_ROUTE_CONCLUSION = "**Via penale**: {name} (art. {art}), {proc}; resta distinta dalle domande di questo parere."  # §9
PENAL_ROUTE_UNKNOWN = "procedibilità non indicata dalle fonti consultate"


# v79.9 (D3): the fraud indicators -- a false description or quality stated as certain, the thing never delivered, the
# counterparty vanished («profilo disattivato», «annuncio cancellato», «irreperibile»): a false description with either
# of the others, or the two others together, is a truffa; a thing lent and not returned, an appropriazione indebita
_FALSE_DESCRIPTION_RE = re.compile(
    r"\b(?:descritt|presentat|spacciat|pubblicizzat|garantit)[oaie]\s+(?:[\w'’]+\s+){0,3}?come\b|\bfalsament\w*|"
    r"\bin\s+realt[àa]\b|\bcontrariamente\s+a\s+quanto\s+(?:dichiarat|indicat|scritt|descritt)\w*|\bfals[oaie]\s+"
    r"(?:document|fattur|ricevut|certificat|dichiarazion|attestazion|garanzi)\w*|\bcontraffatt\w*|\bmai\s+incidentat\w*",
    re.IGNORECASE)
_NOT_DELIVERED_RE = re.compile(
    r"\bnon\s+(?:è\s+|sono\s+)?(?:mai\s+)?(?:stat[oaie]\s+)?(?:consegnat|spedit|inviat|arrivat|ricevut|recapitat)\w*|"
    r"\bnon\s+(?:ha\s+)?(?:mai\s+)?(?:consegna|spedisce|invia|riceve)\b|\bnon\s+(?:arriva|arrivano|è\s+mai\s+arrivat\w*)\b|"
    r"\bmai\s+(?:arrivat|consegnat|spedit|ricevut)\w*|\bmerce\s+mai\b", re.IGNORECASE)
_VANISHED_RE = re.compile(
    r"\bprofil\w*\s+(?:[\w'’]+\s+){0,3}?(?:è\s+stato\s+|risulta\s+|viene\s+)?(?:disattivat|cancellat|chius|rimoss|"
    r"eliminat)\w*|\bannunc\w*\s+(?:[\w'’]+\s+){0,3}?(?:è\s+stato\s+|risulta\s+|viene\s+)?(?:cancellat|rimoss|eliminat)\w*|"
    r"\birreperibil\w*|\bnon\s+risponde\w*\s+(?:più|al\s+telefono)|"
    r"\bsparit[oaie]\b|\bsi\s+(?:è|sono)\s+(?:resi?\s+)?(?:irreperibil|dileguat)\w*|\b(?:numero|utenza)\s+(?:è\s+)?"
    r"(?:disattivat|inesistent|staccat)\w*|\baccount\s+(?:è\s+stato\s+)?(?:chius|disattivat|cancellat)\w*|"
    r"\bl['’]?ha\s+bloccat[oa]|\bbloccat[oa]\s+su\s+(?:whatsapp|telegram|instagram|facebook)", re.IGNORECASE)
_LENT_RE = re.compile(r"\b(?:comodat\w*|prestat[oaie]|in\s+prestito|affidat[oaie]\s+in\s+custodia|dat[oaie]\s+in\s+"
                      r"(?:uso|custodia|deposito)|lasciat[oaie]\s+in\s+custodia)\b", re.IGNORECASE)
_NOT_RETURNED_RE = re.compile(r"\b(?:si\s+rifiut\w*\s+di\s+(?:restituir|ridar|riconsegnar)\w*|non\s+(?:l[aoie]\s+|l['’]\s*|"
                              r"gli\s+|le\s+)?(?:ha\s+)?(?:più\s+|mai\s+)?(?:restitui|ridat|riconsegna|resa|reso)\w*|"
                              r"(?:l[aoie]\s+)?tien\w*\s+per\s+s[ée])", re.IGNORECASE)


def fraud_indicators(facts: str) -> List[str]:
    """v79.9 (D3): which indicators the certain facts hold -- descrizione falsa, mancata consegna, controparte sparita."""
    return [name for name, rx in (("descrizione falsa", _FALSE_DESCRIPTION_RE), ("mancata consegna", _NOT_DELIVERED_RE),
                                  ("controparte sparita", _VANISHED_RE)) if rx.search(facts or "")]


def penal_route(sentences: Sequence[Dict[str, Any]], posture: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """v79.8 (E3): (article, name) of the first reato a civil case's certain facts describe; None in a criminal matter
    or with none. v79.9 (D3): the fraud indicators first (a false description with a non-delivery or a vanished
    counterparty, or those two together: 640), a thing lent and not returned (646), then the table."""
    from . import issue_records
    if criminal(posture):
        return None
    facts = " ".join(str(x.get("text") or "") for x in sentences or [] if issue_records.certain_fact(x))
    found = fraud_indicators(facts)
    if len(found) >= 2:
        return "640 c.p.", "truffa"
    if _LENT_RE.search(facts) and _NOT_RETURNED_RE.search(facts):
        return "646 c.p.", "appropriazione indebita"
    return next(((art, name) for art, name, rx in PENAL_ROUTES if re.search(rx, facts, re.IGNORECASE)), None)


def penal_route_line(art: str, name: str, procedibilita: str) -> Dict[str, str]:
    """v79.8 (E3): {"summary": §1's line, "conclusion": §9's} -- two wordings, so neither is taken for a repetition."""
    proc = _plain(procedibilita).rstrip(".") or PENAL_ROUTE_UNKNOWN
    proc = proc[:1].lower() + proc[1:]
    return {"summary": PENAL_ROUTE_LINE.format(name=name, art=art, proc=proc),
            "conclusion": PENAL_ROUTE_CONCLUSION.format(name=name, art=art, proc=proc)}


def main_open_ids(records: Dict[str, Dict[str, Any]], article: str) -> List[str]:
    """v79.8 (E1): the open elements of the main line's article -- «da provare», or «provato» on an allegazione alone
    (C1 makes none) --, the only ones «dipende da» lists."""
    return [r["id"] for r in _live((records or {}).get(article) or {}) if r.get("id") and
            (r.get("status") == DA_PROVARE or "allegazione" in str(r.get("why") or ""))]


def certainty_of(a: Dict[str, Any], records: Dict[str, Dict[str, Any]], log: Optional[Dict[str, Any]] = None
                 ) -> Dict[str, Any]:
    """v79.8 (E1): «grado di certezza» on the main line's article alone -- alto with none of its elements open, medio
    with one, basso otherwise; an assessment on no record article keeps its grado, never «alto» (nothing it depends on
    can be named). a["open"]: the open elements («dipende da»)."""
    art = str(a.get("article") or "")
    if art not in (records or {}):
        return {**a, "open": [], "grado": MEDIO if a.get("grado") == ALTO else a.get("grado")}
    open_ = main_open_ids(records, art)
    grado = ALTO if not open_ else MEDIO if len(open_) == 1 else BASSO
    if log is not None and grado != a.get("grado"):
        log["certainty"] = {"article": art, "was": a.get("grado"), "now": grado, "open": open_}
    return {**a, "grado": grado, "open": open_}


def conclusions_from_selection(a: Dict[str, Any], sols: Sequence[Dict[str, Any]], main: int, sub: Optional[int],
                               ctx: Dict[str, Any], client: str) -> str:
    """§9 from the selection: the record's assessment (what it depends on: the elements still to prove; the residual
    risk: the charge or the claim when it is fully integrato, else those elements), the main line with its argument,
    the subordinate one, the theses barred or unverifiable."""
    penal = criminal(ctx.get("posture"))                          # v79 (E2)
    art = a.get("article") or ""
    open_ = list(a["open"]) if "open" in a else _open_ids(ctx)      # v79.8 (E1): the main line's article's only
    depends = f"la prova degli elementi {', '.join(open_)}" if open_ else ""
    if a.get("defence") or defence_record(art, (ctx.get("records") or {}).get(art)):
        depends, risk = defence_risk(a, open_)          # v79.12.1: never a conviction under the defence article
    elif a.get("all_integrated") and art:
        risk = f"la condanna per l'art. {art}" if penal else f"l'accoglimento della domanda avversaria (art. {art})"
    elif open_:
        risk = (f"che l'accusa provi gli elementi {', '.join(open_)}" if penal and (ctx.get("posture") or {}).get("role") in ACCUSED
                else f"che non si provino gli elementi {', '.join(open_)}")
    else:
        risk = ""
    m = sols[main]
    # v79.12.7 (item 7): «Obiettivo realistico» no longer repeats the main line -- the next line prints it, once
    lines = [assessment_text(a, client, "", depends, risk)]
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
    penal = criminal(ctx.get("posture"))                          # v79 (E2)
    who = "l'accusa" if penal else "la controparte"
    known = elements_by_id(ctx.get("records") or {}, ctx.get("circumstances"))
    missing = {x["elemento"]: x for x in (ctx.get("facts") or {}).get("mancanti") or [] if x.get("elemento")}
    steps: List[Dict[str, str]] = []

    def add(key: str, **kw: str) -> None:
        passo, scopo = STEP_TEXTS[key]
        steps.append({"passo": passo.format(**kw), "scopo": scopo.format(**kw)})
    if m.get("fallback") or m.get("kind") == "contenimento":      # v75b: its own content, no element to prove
        if riti_names(ctx.get("riti") or [], 2):          # v79.10.3 (C1): the admitted ones only
            add("riti", riti=riti_names(ctx["riti"], 2))
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
    penal = criminal(ctx.get("posture"))                          # v79 (E2)
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


# --- v79.6 (item 1): the questions answered first --------------------------------------------------------------------------

ANSWERS_TITLE = "**Risposte ai quesiti**"
NO_SOURCE_ANSWER = "Non risulta dagli atti né dalle norme esaminate."        # the PO's words (v79.8.6)
ANSWER_SENTENCES = (2, 4)
_PROCEDURAL_Q_RE = re.compile(r"\b(?:rit[oi]|procedur\w*|procediment\w*|termin[ei]|entro\s+quando|scadenz\w*|tempi\b|"
                              r"impugn\w*|ricors\w*|appell\w*|competen\w*|giudice|tribunale|querela|udienz\w*|"
                              r"notific\w*|mediazion\w*|negoziazion\w*|opposizion\w*|davanti\s+a\s+chi|come\s+si\s+procede|"
                              r"prescri\w*|decadenz\w*)", re.IGNORECASE)
_ANSWER_ITEM_RE = re.compile(r"^(\d+)\.\s+«(.+?)»\s+—\s+(.*)$")


def _q_words(text: str) -> set:
    return {w[:5] for w in re.findall(r"[a-zà-ü]{4,}", str(text or "").lower())} - {"quali", "quale", "come", "posso",
                                                                                    "possi", "chied", "vorre", "sapere"}


def _first_sentences(text: str, n: int) -> List[str]:
    out = []
    for line in (text or "").split("\n"):
        line = _BOLD_HEAD_RE.sub("", line).strip().lstrip("-• ").strip()
        if not line or line.startswith("|"):
            continue
        out += [x.strip() for x in _citation_rejoin(_SENT_SPLIT.split(line)) if x.strip()]
    return out[:n]


def procedural_answer(question: str, procedure: str) -> List[str]:
    """The §8 sentences a procedural question asks about -- those sharing its words, in their order; ANSWER_SENTENCES
    at most."""
    want = _q_words(question)
    pool = _first_sentences(procedure, 60)
    hits = [x for x in pool if want & _q_words(x)]
    return hits[:ANSWER_SENTENCES[1]]


def answers_block(ctx: Dict[str, Any], quesiti: Sequence[Tuple[str, str]], procedure: str,
                  sols: Sequence[Dict[str, Any]], main: int, a: Dict[str, Any], client: str,
                  checked: Callable[[str], str], log: Optional[Dict[str, Any]] = None,
                  verified: Optional[Dict[str, Any]] = None) -> Tuple[str, List[str]]:
    """v79.6 (item 1): («Risposte ai quesiti», the code-written sentences) -- one numbered answer per question intake
    extracted (ctx["asked"]), in order, two to four sentences: the writer's own section on that question (checked as
    any written prose: Block B's grounding), a procedural question from §8, any other from the selection (the main line
    and the forecast, code-written); nothing grounded -> NO_SOURCE_ANSWER. Never a guess, never an offer to go on.
    v79.15.0 (build B, verified: analysis_step.view): the strategy sentence is the verified leading line (no forecast);
    a prose sentence stating what a norm provides with no verified affermazione_di_diritto is removed
    (analysis_step.qa_filter: ctx["qa_sentences_dropped"], log «qa_dropped»); the verified affermazioni printed after
    the answers. v79.15.1 (item 4): each answer reviewed by analysis_step.ProseReview (ctx["prose_scope"], compose's;
    else built here) -- such a sentence replaced by the first verified statement on its article (dropped only with
    none, and the anaphoric sentence after it with it); a sentence citing an excluded or an unfitted article dropped;
    the events into ctx["prose_review"], the counters (analysis_step.prose_counters) into ctx."""
    asked = [q for q in ctx.get("asked") or [] if str(q).strip()]
    by_title = {t: x for t, x in quesiti}
    for t, _ in quesiti:                # a writer's quesito intake did not list
        if not any(same_line(t, q) for q in asked):
            asked.append(t)
    if not asked:
        return "", []
    who = client or "la parte assistita"
    esito = {FAVOREVOLE: f"favorevole a {who}", SFAVOREVOLE: f"sfavorevole a {who}", INCERTO: "incerto"}.get(
        a.get("esito"), str(a.get("esito") or ""))
    strategy = [f"La linea principale è: {sols[main]['tesi'].rstrip('.')}.",
                f"L'esito probabile è {esito}: {_plain(a.get('why')).rstrip('.')}."] if sols else []
    qa_dropped: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    scope: Dict[str, Any] = {}
    if verified is not None:
        from . import analysis_step
        strategy = analysis_step.answer_strategy(verified)
        events = ctx.setdefault("prose_review", [])
        scope = ctx.get("prose_scope") or analysis_step.prose_scope(          # v79.15.1 (item 4)
            verified, excluded_labels(ctx), [ctx.get("case") or ""] + [str(q) for q in ctx.get("asked") or []])
    items, how = [], []
    said_at = 0                 # v79.12.7 (item 7): the answer that printed the main line -- the block prints it once
    for i, q in enumerate(asked, 1):
        written = next((x for t, x in by_title.items() if same_line(t, q) or _plain(t) == _plain(q)), "")
        if written and verified is not None:    # v79.15.1 (item 4): the writer's sentences reviewed first -- a rule
            review = analysis_step.ProseReview(verified, scope, analysis_step.SITE_QA, events)   # replaced before the
            mine = [y for y in (review(x) for x in _first_sentences(written, 60)) if y]     # checks could delete it --
            done = _first_sentences(checked(" ".join(mine)), 60)        # then the answer-wide checks; a sentence they
            gone = analysis_step.orphans(mine, done)                    # removed takes its anaphoric follower with it
            events += [analysis_step.dependent_event(analysis_step.SITE_QA, done[j]) for j in sorted(gone)]
            got, kind = [x for j, x in enumerate(done) if j not in gone][:ANSWER_SENTENCES[1]], "writer"
        elif written:
            got, kind = _first_sentences(checked(written), ANSWER_SENTENCES[1]), "writer"
        elif _PROCEDURAL_Q_RE.search(q):
            got, kind = procedural_answer(q, procedure), "procedura"
            if len(got) < ANSWER_SENTENCES[0]:
                got += strategy[:ANSWER_SENTENCES[0] - len(got)]
        else:
            got, kind = list(strategy), "selezione"
        if strategy and strategy[0] in got:
            if said_at:
                got = [MAIN_LINE_SEE_ANSWER.format(n=said_at) if x == strategy[0] else x for x in got]
            else:
                said_at = i
        if verified is not None and kind != "writer":   # v79.15.0 (build B): no unverified statement of what a norm
            see = MAIN_LINE_SEE_ANSWER.split("{")[0]    # provides -- the code's own lines kept, in their order;
            review = analysis_step.ProseReview(verified, scope, analysis_step.SITE_QA, events)    # v79.15.1 (item 4):
            got = [review.keep(x) if x in strategy or x.startswith(see) else review(x) for x in got]  # replaced
        got = [x for x in got if x][:ANSWER_SENTENCES[1]]
        if not got:
            got, kind = [NO_SOURCE_ANSWER], "nessuna fonte"
        how.append(kind)
        items.append(f"{i}. «{_plain(q).rstrip()}» — {' '.join(got)}")
    if log is not None:
        log["answers"] = how
    block = ANSWERS_TITLE + "\n\n" + "\n".join(items)
    if verified is not None:
        from . import analysis_step
        qa_dropped = [e for e in events if e.get("sede") == analysis_step.SITE_QA]
        ctx.update(analysis_step.prose_counters(events))        # v79.15.1 (item 4): every site's, compose's too
        ctx["qa_dropped"] = [e for e in qa_dropped if e.get("esito") == analysis_step.DROPPED]
        if qa_dropped:
            _log("qa_dropped", {"removed": qa_dropped})
        statements = analysis_step.statements_block(verified)
        if statements:
            block += "\n\n" + statements
    return block, strategy


def answers_of(answer: str) -> List[Tuple[str, str]]:
    """v79.6: [(question, answer)] of the parere's «Risposte ai quesiti»."""
    text = answer or ""
    at = text.find(ANSWERS_TITLE)
    if at < 0:
        return []
    block = text[at + len(ANSWERS_TITLE):]
    end = _SECTION_RE.search(block)
    block = block[:end.start()] if end else block
    return [(m.group(2), m.group(3).strip()) for m in (_ANSWER_ITEM_RE.match(x.strip()) for x in block.split("\n")) if m]


# --- v79 (E2): no internal wording reaches the reader ------------------------------------------------------------------------

_ELEMENT_PAREN_RE = re.compile(r"\s*\(\s*(?:elementi?\s+)?([RCP]\d{1,3}(?:\s*(?:,|e|ed)\s*[RCP]\d{1,3})*)\s*\)")
_ELEMENT_TOKEN_RE = re.compile(r"(?<![\w./«-])([RCP]\d{1,3})(?![\w/-])")
_REGISTRO_RE = re.compile(r"\b(?:il|dal|nel|del|al|sul)\s+registro\b|\bregistro\s+(?:delle\s+pronunce|non\b|dà\b)",
                          re.IGNORECASE)
_ACCUSA_RE = re.compile(r"\b([Ll])['’]accusa\b")
_AGGRAVANTE_RE = re.compile(r"\baggravant", re.IGNORECASE)


_ANY_REGISTRO_RE = re.compile(r"\bregistr[oi]\b", re.IGNORECASE)


def registro_hits(answer: str) -> List[str]:
    """v79.8.6: every «registro» in a delivered answer, any form, any section, tables included -- the nightly wants
    none."""
    text = answer or ""
    return [" ".join(text[max(0, m.start() - 40):m.end() + 40].split()) for m in _ANY_REGISTRO_RE.finditer(text)]


def wording_hits(text: str, known_ids: Sequence[str], criminal_matter: bool) -> List[str]:
    """v79 (E2): the internal wording a text shows the reader outside its tables -- an element code, the «registro»,
    and in a matter that is not criminal «l'accusa» and «aggravante». v79.12.7 (item 7): a ruling's stance mark
    («(contro)») too."""
    from . import text_hygiene
    out = []
    ids = set(known_ids)
    for line in (text or "").split("\n"):
        if line.lstrip().startswith("|"):
            continue
        out += [f"codice {m.group(1)}" for m in _ELEMENT_TOKEN_RE.finditer(line) if m.group(1) in ids]
        out += [f"«{x}»" for x in text_hygiene.stance_hits(line)]
        out += [f"«{m.group(0)}»" for m in _REGISTRO_RE.finditer(line)]
        if not criminal_matter:
            out += [f"«{m.group(0)}»" for m in _ACCUSA_RE.finditer(line)]
            out += [f"«{m.group(0)}…»" for m in _AGGRAVANTE_RE.finditer(line)]
    return out


def reader_wording(text: str, known: Dict[str, Dict[str, Any]], criminal_matter: bool) -> str:
    """v79 (E2): the text as the reader gets it, outside its tables -- an element code's parenthesis gone, a bare code
    its element's words («…»); in a matter that is not criminal «l'accusa» is «la controparte» and a sentence on an
    «aggravante» goes; a sentence on the «registro» goes."""
    def element(m: "re.Match") -> str:
        return f"«{_short(known[m.group(1)]['text'], 80)}»" if m.group(1) in known else m.group(0)

    def paren(m: "re.Match") -> str:
        ids = re.findall(r"[RCP]\d{1,3}", m.group(1))
        return "" if ids and all(x in known for x in ids) else m.group(0)

    def review(sentence: str) -> str:
        if _REGISTRO_RE.search(sentence) or (not criminal_matter and _AGGRAVANTE_RE.search(sentence)):
            return ""
        return sentence
    out = []
    for line in (text or "").split("\n"):
        if line.lstrip().startswith("|"):
            out.append(line)
            continue
        line = _ELEMENT_TOKEN_RE.sub(element, _ELEMENT_PAREN_RE.sub(paren, line))
        if not criminal_matter:
            line = _ACCUSA_RE.sub(lambda m: "La controparte" if m.group(1) == "L" else "la controparte", line)
        out.append(_by_sentence(line, review) if line.strip() else line)
    return "\n".join(out)


def legacy_plan(ctx: Dict[str, Any], sols: Sequence[Dict[str, Any]], main: int, sub: Optional[int],
                verified: Dict[str, Any]) -> Dict[str, Any]:
    """v79.15.0 (build B, ANALYSIS_STEP): what the old engine (the schema matcher, the thesis generator, the score)
    would have printed -- its main and subordinate line (article, thesis) and its candidates -- with the comparison
    against the verified analysis (analysis_step.compare: legacy_disagrees_line, legacy_disagrees_esito); logged
    «legacy_plan» by the flow, never rendered."""
    from . import analysis_step
    records = ctx.get("records") or {}
    known = elements_by_id(records, ctx.get("circumstances") or {})

    def line(i: Optional[int]) -> Optional[Dict[str, Any]]:
        if i is None or not sols or i >= len(sols):
            return None
        t = sols[i]
        return {"articolo": line_article(t, known) or None, "tesi": _short(t.get("tesi"), 200),
                "placeholder": bool(t.get("placeholder"))}
    out = {"linea_principale": line(main), "linea_subordinata": line(sub),
           "candidate_articles": [", ".join(candidate_subjects(records)(c)) for c in ctx.get("candidates") or []],
           "lines": [_short(t.get("tesi"), 120) for t in sols]}
    got = analysis_step.compare(verified, out, records)
    out.update(got)
    out["rendered"] = False
    return out


def compose(sections: Sequence[Tuple[str, str]], ctx: Dict[str, Any], plan: Dict[str, Any],
            profiles: Sequence[str] = (), log: Optional[Dict[str, Any]] = None) -> str:
    """The answer in the ten sections of his STRUTTURA DELL'OUTPUT, from our checked sections (sections_of the draft),
    the context (prepare()) and the method plan; his CONTROLLO FINALE answered and logged (final_check)."""
    log = log if log is not None else {}
    records = ctx.get("records") or {}
    a = ctx.get("assessment") or {}
    client = ctx.get("client") or ""
    # v79.15.0 (build B, ANALYSIS_STEP): the verified analysis (analysis_step.view) decides §1, §4's tables, §6, §9, the
    # counts and the answers' line -- the old engine below still runs, logged (ctx["legacy_plan"]), and renders none of them
    verified = ctx.get("verified_analysis") if isinstance(ctx.get("verified_analysis"), dict) else None
    from . import analysis_step
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
    exclusions: List[Dict[str, Any]] = []       # v79.12.7 (item 2): a circumstance called excluded, rewritten
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

    from . import answer_ledger, rule_statements
    removals = answer_ledger.Ledger()           # v79 (E3): compose is the last check -- its removals are «removed»

    def tracked(step: str, before: str, after: str) -> str:
        if before and before != after:
            removals.record(f"compose:{step}", [("", before)], [("", after)])
        return after
    rule_src = rule_statements.sources(ctx)
    rules_log: List[Dict[str, str]] = []        # v79 (B1): the rule statements removed, with their reason
    objections_dropped: List[str] = []          # v79 (B2): an objection or a reply the checks emptied

    from . import rulings as rulings_mod
    labels_log: List[str] = []                  # v79 (F1): a registry label that names no registry ruling
    rites_log: List[str] = []                   # v79.10.3 (C1): a written sentence offering a rito not admitted

    closed_list = stage_lines(ctx.get("posture") or {})     # v79.12.5 (item 10): §8's closed list, code-written
    closed_kept: List[str] = []                             # its sentences the rule check would have removed
    from . import text_hygiene
    known_ids = elements_by_id(records, circ)               # v79.12.7 (item 7): an element code no record holds
    unknown_log: List[str] = []

    # every written paragraph: no orientation of its own, retrieved
    def prose(text: str, rules: bool = True, keep: Sequence[str] = ()) -> str:
        text = rulings_mod.labels_out(text, ctx.get("rulings_record"), labels_log)     # v79 (F1): «[G1]» -> estremi
        text = orientation_claims_out(text, orientation_claims)            # rulings only, circumstances with their
        text = unretrieved_rulings_out(text, rulings, rulings_out)         # requisites, no remissione for a d'ufficio
        text = rulings_scope_out(text, rulings, scope_log)                 # charge; v78: on its article and element
        text = procedibility_out(text, procedibilita, a.get("article") or "", procedural_out,
                                 penal=criminal(ctx.get("posture")))     # v79.10.3 (C1): querela only, from the store
        text = rites_out(text, ctx.get("rites") or [], rites_log)      # v79.10.3 (C1): no rito the charge refuses
        text = fact_terms.apply(text, certi, ctx.get("case") or "", client, accused, fact_wording)     # v74
        text = circumstance_exclusions(text, circ, exclusions)     # v79.12.7 (item 2): never «esclusa» unfounded
        text = circumstance_claims(text, circ, claims)
        text = text_hygiene.unknown_codes_out(text, known_ids, unknown_log)  # v79.12.7 (item 7): an invented «R9»
        return rule_statements.check(text, rule_src, rules_log, keep, closed_kept) if rules else text   # v79 (B1)

    def checked(text: str, rules: bool = True, keep: Sequence[str] = ()) -> str:     # v79 (E3): prose, its removals
        return tracked("prose", text, prose(text, rules, keep))                     # into the ledger
    charged_log: List[str] = []         # v75b: the charged article's paragraph from its sussunzione rows only
    excluded = excluded_labels(ctx)     # v79.14.3 (item 4): excluded means excluded -- only «Altri profili» names them
    dropped_log: List[str] = []
    # v79.15.1 (item 4): the prose review's articles (the excluded, the fitted set, the user's) and its events -- §4's
    # prose, §7's model-written objections and the Q&A (answers_block) reviewed by analysis_step.ProseReview
    prose_events: List[Dict[str, Any]] = []
    ctx["prose_review"] = prose_events
    ctx["prose_scope"] = analysis_step.prose_scope(
        verified, excluded, [ctx.get("case") or ""] + [str(q) for q in ctx.get("asked") or []],
        list(circ)) if verified is not None else None
    # v79.15.1 (item 4): with the verified analysis, the prose reviewed first (analysis_step.ProseReview: a rule with
    # no verified twin replaced by its article's verified statement, else dropped with its anaphoric follower; an
    # excluded or unfitted article's sentence dropped), then `check` (the answer-wide checks); a sentence `check`
    # removed takes its anaphoric follower with it (orphans_out). Without it, `check` alone. replace=False (§7's
    # objection itself, v79.15.1 review): a rule with no verified twin dropped, never replaced by a statement of law.
    def reviewed(check: Callable[[str], str], text: str, site: str, replace: bool = True) -> str:
        if verified is None or not text:
            return check(text)
        mine = tracked("analysis_step", text, _by_sentence(text, analysis_step.ProseReview(
            verified, ctx["prose_scope"], site, prose_events, replace=replace)))
        return tracked("analysis_step", mine, orphans_out(mine, check(mine), prose_events, site))
    norms = reviewed(checked, charged_blocks(excluded_blocks_out(next((by_title[t] for t in NORMS_FROM
                                                                    if by_title.get(t)), ""), excluded, dropped_log),
                                             {} if verified is not None else ctx.get("charged_prose") or {},
                                             charged_log), analysis_step.SITE_NORMS)
    quesiti = [(t, x) for t, x in others if t.endswith("?")]
    normative = [(t, x) for t, x in others if not t.endswith("?")]
    norms_parts = [norms] if norms else []
    cenni_log: List[Dict[str, str]] = []        # v79.12.7 (item 7): a cenno's rule grounded in its article only

    def normative_text(t: str, x: str) -> str:
        if not cenni_title(t):
            return checked(x)
        # v79.12.7 (item 7): a «Cenni» section's rule statements on the article each cites -- its text (one comma) or
        # Block B's sources on it; naming none, its section's articles -- in place of Block B's answer-wide check
        done = checked(x, rules=False)
        return tracked("cenni", done, text_hygiene.cenni_check(done, ctx.get("cenni_texts") or {}, rule_src, cenni_log))
    norms_parts += [f"**{t}**\n\n" + reviewed(lambda y, t=t: normative_text(t, y),
                                              excluded_blocks_out(x, excluded, dropped_log), analysis_step.SITE_NORMS)
                    for t, x in normative if x and excluded_blocks_out(x, excluded)]
    norm_dropped = [e for e in prose_events if e.get("sede") == analysis_step.SITE_NORMS]
    ctx.update(analysis_step.prose_counters(prose_events))         # v79.15.1 (item 4); answers_block adds the Q&A's
    if norm_dropped:
        _log("norms_dropped", {"removed": norm_dropped})
    if verified is not None:            # v79.15.0 (build B): the verified verdicts only; the remedies named, no table
        norms_parts.append(f"{ANALYSIS_TITLE}\n\n{analysis_step.norms_tables(verified, list(circ))}")
    elif ctx.get("subsumption"):
        norms_parts.append(f"{ANALYSIS_TITLE}\n\n{ctx['subsumption']}")
    if profiles:
        from . import interaction_copy as C
        norms_parts.append(f"**{C.OTHER_PROFILES_SECTION}**\n\n" + "\n".join(profiles))
    # v79.19.1 (item 6b): every article the fit accepted («si» / «dipende») that is a record at the end is printed --
    # in §4 with its scheda (the tables, the writer's blocks) or at least here with one line, never silently absent;
    # records_unprinted counts what is still absent after that (must be 0), logged
    late = unprinted_records(ctx, records, "\n\n".join(norms_parts), excluded)
    if late:
        norms_parts.append(f"**{UNPRINTED_HEAD}**: " + "; ".join(late) + ".")
        _log("records_printed_late", {"articles": late})
    still = unprinted_records(ctx, records, "\n\n".join(norms_parts), excluded)
    ctx["records_unprinted"] = len(still)
    log["records_unprinted"] = still
    if still:
        _log("records_unprinted", {"articles": still})
    procedure = next((by_title[t] for t in PROCEDURE_FROM if by_title.get(t)), "")
    extra = [x for x in [ctx.get("temporal")] + [f"Litisconsorzio: {x}" for x in ctx.get("litisconsorzio") or []] if x]
    precluded = precluded_block(ctx.get("riti_precluded") or [], procedure)       # v79.10.3 (C1): §8 says them
    extra += [precluded] if precluded else []
    procedure = "\n\n".join([p for p in [checked(procedure, keep=closed_list)] + extra if p]) or \
        "Nessun profilo processuale calcolato sul caso."
    coherence: List[Dict[str, Any]] = []
    sols, main = coherent_solutions(plan, records, a, ctx.get("riti") or [], coherence, ctx.get("posture") or {}, circ,
                                    ctx=ctx)                    # v79.12.1 (row 16): pertinence before realismo
    if any(t.get("candidate") for t in sols):   # v76: a thesis the model added (no candidate id) is never displayed
        added = [t["tesi"][:160] for i, t in enumerate(sols) if not t.get("candidate") and not t.get("fallback")
                 and i != main]
        if added:
            keep = [i for i, t in enumerate(sols) if t.get("candidate") or t.get("fallback") or i == main]
            main, sols = keep.index(main), [sols[i] for i in keep]
            log["added_theses_dropped"] = added
    # v79.12.5 (item 2): no admitted claim -- the ten sections stay, with no main line and no forecast (no_claim)
    kind = no_claim(ctx) if sols[main].get("placeholder") else ""
    if verified is not None:            # v79.15.0 (review): the old engine's no-claim verdict decides no section (§3,
        log["no_claim_legacy"] = kind   # §6, §7, §9, the answers) with the verified analysis -- logged only
        kind = ""
    log["no_claim"] = kind
    ctx["no_claim_sections"] = kind
    # v79.15.0 (build B): with the verified analysis the old engine's placeholder decides nothing -- its forecast and
    # selection are only skipped (no_line), the sections are the analysis's
    no_line = bool(kind) or (verified is not None and bool(sols[main].get("placeholder")))
    if sols[main].get("placeholder") and not kind and verified is None:    # v75: never printed -- "" keeps the
        # standard body, logged
        log["refused"] = {"why": "placeholder", "tesi": sols[main]["tesi"]}
        _log("compose_refused", log["refused"])
        return ""
    if not no_line:
        a = settle_objective(forecast(ctx, sols[main], a, log), sols, records, circ)      # v79 (E1)
        a = certainty_of(a, records, log)                   # v79.8 (E1): the main line's article alone
    scope = article_scope(ctx, [norms, procedure] + list(by_title.values()))       # v78: what record_check passed

    def planned(text: str, code: bool = False) -> str:     # v78: the plan's prose -- the checks above, then the
        done = checked(text, rules=not code)                                      # articles' scope (v79: a code-
        return tracked("scope", done, articles_out(done, scope, articles_log))
    for t in sols:                          # the table's cells are prose too          # written text: its sources)
        code = t.get("argued") == "code" or (bool(t.get("fallback")) and not t.get("candidate"))
        t["pro"], t["contro"] = planned(t.get("pro") or "", code), planned(t.get("contro") or "", code)
        arg = t.get("argomento") or ""
        t["argomento"] = planned(tracked("coherence", arg, coherent_text(arg, records, a, coherence)), code)
    sub = subordinate_of(sols, main)        # v72: the subordinate line, the summary, the conclusions and the next steps
    strong = {} if no_line or verified is not None else strongest_open(ctx, sols, main)    # v79.14.1 (item 6): every
    # element open (v79.15.0 build B: the old engine's verdicts decide no §7 with the verified analysis)
    objections_reviewed = []                # v78: the objection itself reviewed too, not only its answer

    for o in plan.get("obiezioni") or []:
        gone = cites_excluded(f"{o.get('obiezione') or ''} {o.get('risposta') or ''}", excluded)
        if gone:                            # v79.14.3 (item 4): an objection on an excluded article -- never in §7
            dropped_log += gone
            tracked("objections", f"{o.get('obiezione') or ''} {o.get('risposta') or ''}".strip(), "")
            continue
        rules_before = len(rules_log)       # v79.14.1 (item 6): the rule check removed nothing of it -- grounded
        said = reviewed(planned, o.get("obiezione") or "", analysis_step.SITE_OBJECTIONS,     # v79.15.1 (item 4;
                        replace=False)      # review: the counterparty's objection never becomes a statement of law
        if not said:
            articles_log.append(_removed_note(o))
            objections_dropped.append(_removed_note(o))
            tracked("objections", f"{o.get('obiezione') or ''} {o.get('risposta') or ''}".strip(), "")
            continue
        reply = o.get("risposta") or ""
        answered = reviewed(lambda y: planned(tracked("coherence", y, coherent_text(y, records, a, coherence))), reply,
                            analysis_step.SITE_OBJECTIONS)                                    # v79.15.1 (item 4)
        if reply.strip() and not answered:  # v79 (B2): a reply the checks emptied drops its objection -- the
            objections_dropped.append(_removed_note({"obiezione": said}, "reply"))      # the fallback fills §7
            tracked("objections", said, "")
            continue
        objections_reviewed.append({**o, "obiezione": said, "risposta": answered,
                                    "_grounded": len(rules_log) == rules_before})
    all_open_pick: Optional[Dict[str, str]] = None
    if strong:      # v79.14.1 (item 6): one objection -- the model's first grounded one on the strongest element -- or none
        all_open_pick = next(({"obiezione": o["obiezione"], "risposta": o.get("risposta") or ""}
                              for o in objections_reviewed if o["_grounded"] and on_element(
                                  f"{o['obiezione']} {o.get('risposta') or ''}", strong)), None)
        log["objections_all_open"] = {**strong, "printed": bool(all_open_pick),
                                      "offered": len(plan.get("obiezioni") or []),
                                      "ungrounded": sum(1 for o in objections_reviewed if not o["_grounded"])}
        objections_reviewed = [all_open_pick] if all_open_pick else []
    objections_reviewed = [{k: v for k, v in o.items() if k != "_grounded"} for o in objections_reviewed]
    ctx["objection_grounded"] = int(bool(strong) and all_open_pick is not None)
    ctx["objection_none"] = int(bool(strong) and all_open_pick is None)
    plan = {**plan, "subordinata": sols[sub]["tesi"] if sub is not None else "",      # are the selection's -- never
            "stragiudiziale": planned(plan.get("stragiudiziale") or ""),                 # the model's plan
            "obiezioni": objections_reviewed,
            "questione_giuridica": planned(plan.get("questione_giuridica") or "")}
    if verified is not None:                # v79.15.0 (build B): §9 the verified analysis's
        conclusions = [analysis_step.conclusions_text(verified)]
    elif kind:                              # v79.12.5 (item 2): no line, no forecast, no steps
        conclusions = [NO_CLAIM_CONCLUSION]
    else:
        conclusions = [conclusions_from_selection(a, sols, main, sub, ctx, client)]
        steps = steps_text(steps_from_selection(ctx, sols, main))
        if steps:
            conclusions.append(steps)
    # v79.6 (item 1): the quesiti are answered first, in «Risposte ai quesiti» -- §9 no longer appends them
    fallback_objections = [] if kind or strong or verified is not None else \
        list(ctx.get("objections_fallback") or []) + thesis_objections(sols, main, ctx)     # v79.15.0 (build B): the
    # code's objections are built on the old engine's selection -- none with the verified analysis
    fallback_objections = [o for o in fallback_objections if not cites_excluded(      # v79.14.3 (item 4)
        f"{o.get('obiezione') or ''} {o.get('risposta') or ''}", excluded)]
    # v79.15.1 (item 3): §7's objections from the verified analysis -- the allegations, a claimant's exceptions
    verified_objections = analysis_step.objection_items(verified) if verified is not None else []
    log["objections_from"] = {"verified": len(verified_objections), "model": len(plan.get("obiezioni") or []),
                              "fallback": 0 if strong else max(0, OBJECTIONS - len(plan.get("obiezioni") or []))}
    # v79.19.1 (item 3): the verified analysis carries no objection and the model wrote none -- 79.15.3's objections
    # path (the code's, built on the record and the thesis: ctx["objections_fallback"] + thesis_objections, the one
    # that printed before the verified analysis took §7), never the empty line; counted objections_fallback_legacy
    ctx["objections_fallback_legacy"] = 0
    if verified is not None and not kind and not strong and not verified_objections and not plan.get("obiezioni"):
        try:
            legacy = list(ctx.get("objections_fallback") or []) + thesis_objections(sols, main, ctx)
        except Exception as exc:          # noqa: BLE001 -- the legacy path failing leaves §7 the record's objections
            _log("objections_fallback_legacy_failed", {"error": f"{type(exc).__name__}: {exc}"[:200]})
            legacy = list(ctx.get("objections_fallback") or [])
        legacy = [o for o in legacy if o.get("obiezione") and not cites_excluded(
            f"{o.get('obiezione') or ''} {o.get('risposta') or ''}", excluded)]
        if legacy:
            fallback_objections = legacy
            ctx["objections_fallback_legacy"] = 1
            log["objections_from"]["fallback_legacy"] = len(legacy)
            _log("objections_fallback_legacy", {"n": len(legacy), "first": str(legacy[0].get("obiezione"))[:120]})
    orientation_points: List[str] = []
    layout_log: Dict[str, Any] = {}
    # v79.12.5 (item 2): the corpus line in §1 -- with no admitted claim (no_claim_summary), else wherever the first
    # read's institution has no module and nothing graded in its family (corpus_line)
    general = "" if kind or missing_laws_line(ctx) else corpus_line(ctx)
    bound_line = binding_empty_line(ctx)                # v79.14.2 (B): the charge set the frame's binding emptied
    general = bound_line or general
    summary = analysis_step.summary_text(verified) if verified is not None else \
        no_claim_summary(ctx, kind) if kind else _with_line(
            summary_section({}, sols, main, a, client, (ctx.get("charge") or {}).get("article") or "", records, log),
            general)
    log["corpus_line"] = CORPUS_LINE.split(" su ")[0] in summary
    ctx["corpus_line"] = log["corpus_line"]
    # v79.14.1 (item 2): each institution's governing law as the line read it; the line on a code institution (must be 0)
    log["corpus_gap"] = dict(ctx.get("corpus_gap_log") or {})
    # v79.14.2 (B, review): the binding's corpus line counted too -- on a code charge (640 c.p. missing from the
    # records) it is the honesty line on a code institution, as interactive_flow.role_corpus_counted reads the answer
    ctx["corpus_line_code"] = len(corpus_line_on_code(summary, ctx))
    log["frame_binding_line"] = bool(bound_line) and bound_line in summary          # v79.14.2 (B)
    if verified is None:                # v79.15.0 (build B): §1 the verified analysis's only
        summary = "\n\n".join(x for x in (summary, (ctx.get("penal_route") or {}).get("summary"),     # v79.8 (E3)
                                           missing_laws_line(ctx)) if x)                             # v79.9 (B2)
    keyed = [
        ("sintesi", summary),
        ("fatti", facts_section(analysis_step.open_facts(verified, ctx.get("facts") or {}, records)
                                if verified is not None else ctx.get("facts") or {})),      # v79.15.0 (build B)
        ("questione", NO_CLAIM_QUESTION.format(client=client or "la parte assistita", materia=action_subject(ctx))
         if kind and not plan.get("questione_giuridica") else    # v79.12.5 (item 2): no article to frame it on
         question_section(plan, a, records, client, (ctx.get("charge") or {}).get("article") or "",
                          log)),                                                         # v79.10.5 (E1)
        ("norme", "\n\n".join(norms_parts) or "Nessuna norma applicabile nel materiale recuperato."),
        ("orientamenti", orientations_text(ctx, plan, rulings, rulings_out, orientation_points, excluded,
                                           dropped_log)),                                # v79.14.3 (item 4)
        ("soluzioni", analysis_step.solutions_text(verified) if verified is not None else   # v79.15.0 (build B)
         no_claim_solutions(ctx, kind) if kind else solutions_section(
            sols, main, plan, ctx.get("riti") or [], ctx.get("stragiudiziale") or "", layout_log,
            post_fact_lines(ctx))),                                         # v79.12.7 (item 3)
        ("obiezioni", NO_CLAIM_OBJECTIONS if kind else all_open_section(strong, all_open_pick,
            _party_of(ctx)["side"]) if strong else objections_section(plan, fallback_objections, **(   # v79.14.1
                {"verified": verified_objections} if verified is not None else {}))),    # (item 6); v79.15.1 (item 3)
        ("processuali", procedure),
        # v79.12.1: the missing-laws line once per answer -- §1's, never again in §9
        ("conclusioni", "\n\n".join(conclusions) if verified is not None else
         _with_line("\n\n".join(conclusions), (ctx.get("penal_route") or {}).get("conclusion"))),
    ]
    # v79.19.1 (item 6c): the rulings §5 prints (every printed article's, as 79.15.3 did: no filter of 79.16-79.18 on
    # this branch), counted per answer for the harness row
    ctx["rulings_printed"] = rulings_printed(ctx, rulings, excluded)
    log["rulings_printed"] = ctx["rulings_printed"]
    known_all = elements_by_id(records, circ)          # v79 (E2): what the reader reads, outside the tables and §5
    crim = criminal(ctx.get("posture"))
    residue: Dict[str, List[str]] = {}          # v79.12.7 (item 7): the stance marks, §5's codes, the answers' residue
    keyed = [(k, x if k == "orientamenti" else tracked("wording", x, reader_wording(x, known_all, crim)))
             for k, x in keyed]
    keyed = [(k, tracked("wording", x, text_hygiene.stance_out(text_hygiene.codes_out(
        x, known_all, residue.setdefault("codes", [])), residue.setdefault("stance", [])))) for k, x in keyed]
    headless_log: List[str] = []        # v79.9 (D4): a ruling's reference whole or not at all, every written section
    keyed = [(k, x if k == "orientamenti" else tracked("citations", x, whole_citations(x, headless_log)))
             for k, x in keyed]
    ctx["excluded_dropped"] = list(dict.fromkeys(list(ctx.get("excluded_dropped") or []) + dropped_log))
    log["excluded_dropped"] = ctx["excluded_dropped"]          # v79.14.3 (item 4)
    log.update({"rule_statements_removed": rules_log, "objections_dropped": objections_dropped,
                "registry_labels_out": labels_log,
                "coherence": coherence, "rulings_out": rulings_out, "circumstance_claims": claims,
                "rulings_scope_out": scope_log, "articles_out_of_scope": articles_log,
                "fact_wording": fact_wording, "charged_prose": charged_log,
                "orientation_claims_out": orientation_claims, "procedibility_out": procedural_out,
                "rites_out": rites_log,                                                      # v79.10.3 (C1)
                "headless_citations_out": headless_log, **layout_log,
                "closed_list_kept": closed_kept,                                             # v79.12.5 (item 10)
                "cenni_rules_removed": cenni_log, "residue_out": residue,                   # v79.12.7 (item 7)
                "unknown_codes_out": unknown_log})
    ctx["cenni_rules_removed"] = len(cenni_log)
    # v79.12.7 (item 7): the cenni's rule statements still ungrounded in the composed §4 (must be 0)
    left_cenni: List[Dict[str, str]] = []
    for part in norms_parts:
        head_, _, body_ = part.partition("\n\n")
        if cenni_title(head_.strip("*")):
            text_hygiene.cenni_check(body_, ctx.get("cenni_texts") or {}, rule_src, left_cenni, rewrite=False)
    ctx["cenni_rules_unsourced"] = len(left_cenni)
    ctx["closed_list_kept"] = len(closed_kept)
    # v79.12.7 (item 2): the written sentences calling a circumstance excluded that the record does not exclude --
    # rewritten (circumstance_exclusions_rewritten) and any left in the composed sections (must be 0)
    left: List[Dict[str, Any]] = []
    for _k, x in keyed:
        circumstance_exclusions(x, circ, left, rewrite=False)
    log["circumstance_exclusions"] = exclusions
    ctx["circumstance_exclusions_rewritten"] = len(exclusions)
    ctx["circumstance_excluded_unsupported"] = len(left)
    ctx["articles_out_of_scope"] = len(articles_log)        # v79.12.5 (item 10): what the scope check still flags
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
    answers, strategy_lines = answers_block(ctx, quesiti, texts.get(title("processuali")) or procedure,
                                            [] if kind else sols, main, a,      # v79.12.5 (item 2): no line
                                            client, checked, log, verified=verified)    # v79.6 (item 1): before §1
    if verified is not None:            # v79.15.1 (item 4): the review's counters over §4, §7 and the Q&A
        ctx.update(analysis_step.prose_counters(prose_events))
        log["prose_review"] = prose_events
        if prose_events:
            _log("prose_review", {"events": prose_events, **analysis_step.prose_counters(prose_events)})
    if answers:     # v79.12.7 (item 7): built after the sections' reader checks -- the path the codes leaked by
        answers = tracked("wording", answers, reader_wording(answers, known_all, crim))
        answers = tracked("wording", answers, text_hygiene.stance_out(text_hygiene.codes_out(
            answers, known_all, residue.setdefault("codes", [])), residue.setdefault("stance", [])))
        answers = tracked("citations", answers, whole_citations(answers, headless_log))
        joined = f"{answers}\n\n{joined}"
    # v79.12.7 (item 7): the main line printed at most once per section (tables included)
    if verified is not None or not kind:
        printed = ([("risposte", answers)] if answers else []) + list(body)
        log["main_line_repeated"] = text_hygiene.main_line_repeated(
            printed, analysis_step.main_tesi(verified) if verified is not None else sols[main]["tesi"])
        ctx["main_line_repeated"] = len(log["main_line_repeated"])
    if verified is not None:            # v79.15.0 (build B): the old engine's plan, logged and compared -- never printed
        ctx["legacy_plan"] = legacy_plan(ctx, sols, main, sub, verified)
    # v79 (B3): what the nightly's output check reads -- the sources, the scope, and the code-written texts (built from
    # the sources: never a rule statement to trace)
    code_texts = [x for k, x in keyed if k in ("sintesi", "fatti", "conclusioni")] + \
        [f"{o.get('obiezione') or ''} {o.get('risposta') or ''}" for o in fallback_objections + verified_objections] + \
        [f"{t['tesi']} {t.get('pro') or ''} {t.get('contro') or ''}" for t in sols] + [ctx.get("subsumption") or ""] + \
        [ctx.get("temporal") or "", ctx.get("stragiudiziale") or ""] + list(ctx.get("litisconsorzio") or []) + \
        list(ctx.get("riti") or []) + list(points) + strategy_lines + list(closed_list)     # v79.12.5 (item 10)
    log["ledger"] = [e for e in removals.entries if e["kind"] in ("removed", "rewritten")]     # v79 (E3)
    ctx["compose_removals"] = log["ledger"]
    ctx["output_sources"] = {**rule_statements.serializable(rule_src), "element_ids": sorted(known_all),
                             "criminal": crim,
                             "scope": sorted(scope | article_scope(ctx, code_texts + [head])),
                             "code": [_plain(x) for x in code_texts if _plain(x)]}
    if ctx.get("rulings_record"):           # v79.8 (E4): no registry label left in any section, §7 included
        joined = "\n".join(rulings_mod.labels_out(line, ctx["rulings_record"], labels_log)
                            if rulings_mod._LABEL_RE.search(line) else line for line in joined.split("\n"))
    return f"{head}\n\n{joined}" if head else joined


# --- v79.6 (item 2): more than one client ----------------------------------------------------------------------------------

CLIENT_HEAD = "**Nell'interesse di {name}**"
SHARED_NOTE = "_Sezione comune a {names}: la loro posizione su questo punto è la stessa._"
# v79.10.2 (B3): no section is per client whatever its text -- v79.6's §1 «always per client» is gone: a section
# identical for every client prints once with the note, never twice
PER_CLIENT_ALWAYS: Tuple[str, ...] = ()


def _section_key(text: str) -> str:
    """v79.10.2 (B3): a section's text as the merge compares it -- spaces and line breaks collapsed."""
    return " ".join(str(text or "").split())


def printed_twice(answer: str) -> List[str]:
    """v79.10.2 (B3): the client blocks («Nell'interesse di …») of a several-client parere whose text another client's
    block of the same section repeats -- [] when no section prints twice (the nightly's check clients_sections_once)."""
    out: List[str] = []
    text = answer or ""
    first = _SECTION_RE.search(text)
    prefix, body = (text[:first.start()], text[first.start():]) if first else (text, "")
    at = prefix.find(ANSWERS_TITLE)
    parts = [(ANSWERS_TITLE.strip("*"), prefix[at + len(ANSWERS_TITLE):])] if at >= 0 else []
    for t, section in parts + list(answer_sections(body)):
        blocks = re.split(r"\*\*Nell'interesse di [^*\n]+\*\*", section or "")
        seen = [_section_key(b) for b in blocks[1:] if _section_key(b)]
        if len(seen) != len(set(seen)):
            out.append(t)
    return out


def _names(names: Sequence[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " e " + names[-1]


def compose_clients(sections: Sequence[Tuple[str, str]], items: Sequence[Tuple[str, Dict[str, Any], Dict[str, Any]]],
                    profiles: Sequence[str] = (), log: Optional[Dict[str, Any]] = None) -> str:
    """v79.6 (item 2): the parere for several clients -- each client's answer composed on its own context and plan
    (compose), then merged section by section: §1 always per client («nell'interesse di …»); any other section shared
    when every client's is the same (SHARED_NOTE says so), per client where their positions differ. The answers to the
    quesiti likewise. The output check's sources are every client's."""
    log = log if log is not None else {}
    names = [n for n, _, _ in items]
    composed = []
    from . import rulings as rulings_mod
    for name, ctx, plan in items:
        own: Dict[str, Any] = {}
        text = compose(sections, ctx, plan, profiles, own)
        record = ctx.get("rulings_record") or items[0][1].get("rulings_record")
        labels: List[str] = []          # v79.8 (E4): a registry label left in any client's section (§7 too) replaced
        text = "\n".join(rulings_mod.labels_out(line, record, labels) if rulings_mod._LABEL_RE.search(line) else line
                          for line in text.split("\n"))
        if labels:
            own["registry_labels_out"] = labels
        composed.append((name, text, ctx))
        log.setdefault("clients", {})[name] = {k: own.get(k) for k in ("main_thesis", "esito", "answers",
                                                                       "charge_refused",     # v79.10.5 (E1)
                                                                       "charge_refused_summary")}
    heads, blocks, bodies = [], [], []
    for name, text, _ in composed:
        first = _SECTION_RE.search(text)
        prefix, body = (text[:first.start()], text[first.start():]) if first else (text, "")
        at = prefix.find(ANSWERS_TITLE)
        heads.append(prefix[:at].strip() if at >= 0 else prefix.strip())
        blocks.append(prefix[at:].strip() if at >= 0 else "")
        bodies.append(answer_sections(body))

    def merged(texts: Sequence[str], always: bool = False) -> str:
        # v79.10.2 (B3): the clients grouped by their text (spaces aside, _section_key) -- one group: once, with the
        # note; else each group's text once, under its clients' names («Nell'interesse di Alfa e Beta»): never twice
        keys = [_section_key(t) for t in texts]
        if not always and len(set(keys)) == 1:
            return f"{texts[0]}\n\n{SHARED_NOTE.format(names=_names(names))}" if keys[0] else ""
        groups: Dict[str, List[int]] = {}
        for i, k in enumerate(keys):
            if k:
                groups.setdefault(k, []).append(i)
        return "\n\n".join(f"{CLIENT_HEAD.format(name=_names([names[i] for i in ix]))}\n\n{texts[ix[0]]}"
                            for ix in groups.values())
    out = []
    if heads[0]:
        out.append(heads[0])
    if any(blocks):
        inner = [b[len(ANSWERS_TITLE):].strip() if b else "" for b in blocks]
        out.append(f"{ANSWERS_TITLE}\n\n{merged(inner)}")
    titles = list(dict.fromkeys(t for b in bodies for t, _ in b))       # v79.10.2 (B3): every client's sections
    shared, grouped = [], []
    for i, t in enumerate(titles):
        texts = [dict(b).get(t, "") for b in bodies]
        always = t in [title(k) for k in PER_CLIENT_ALWAYS]
        keys = [_section_key(x) for x in texts]
        if not always and len(set(keys)) == 1:
            shared.append(t)
        elif len({k for k in keys if k}) < len([k for k in keys if k]):
            grouped.append(t)
        out.append(f"**{i + 1}. {t}**\n\n{merged(texts, always)}")
    log["shared_sections"] = shared
    log["grouped_sections"] = grouped           # v79.10.2 (B3): some clients' text the same, printed once for them
    first_ctx = items[0][1]
    first_ctx["output_sources"] = merge_sources([c.get("output_sources") or {} for _, _, c in composed])
    # v79.10.5 (E3): each client's removals tagged with its client (answer_ledger.settle strikes them in its own text)
    first_ctx["compose_removals"] = [{**e, "client": name} for name, _, c in composed
                                     for e in c.get("compose_removals") or []]
    _log("compose_clients", {"clients": names, "shared": shared})
    return "\n\n".join(out)


def merge_sources(sources: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The output check's sources of several clients' answers, as one."""
    out: Dict[str, Any] = {"elements": {}, "store": {}, "passages": {}, "element_ids": [], "scope": [], "code": [],
                           "criminal": any(s.get("criminal", True) for s in sources) if sources else True}
    for s in sources:
        for key in ("elements", "passages"):
            for label, items in (s.get(key) or {}).items():
                out[key].setdefault(label, []).extend(items)
        for label, fields in (s.get("store") or {}).items():
            out["store"][label] = sorted(set(out["store"].get(label) or []) | set(fields))
        for key in ("element_ids", "scope", "code"):
            out[key] = list(dict.fromkeys(list(out[key]) + list(s.get(key) or [])))
    return out


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


def issue_records_certain(sentence: Dict[str, Any]) -> bool:
    from . import issue_records
    return issue_records.certain_fact(sentence)


def family_words(state: Dict[str, Any], case: str) -> str:
    """v79.9 (C1): the words a family's signal is read in -- the first read's (special_laws.read_text), else the case."""
    from . import special_laws
    read = state.get("read")
    return special_laws.read_text(read) if isinstance(read, dict) and read else (case or "")


def computed_terms(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """v79.12.4 (item 7): the terms the facts computed (fact_signals.compute_terms' entries: id, ref, status …) -- the
    analysis' fact_signals; [] without them."""
    analysis = state.get("analysis") or {}
    terms = ((analysis.get("fact_signals") or {}).get("terms") or []) if isinstance(analysis, dict) else []
    return [t for t in terms if isinstance(t, dict)]


def fact_events(state: Dict[str, Any]) -> List[str]:
    """v79.9 (B1): the kinds of the events the facts extraction read (fact_signals: consegna, fine lavori, scoperta
    vizio, denuncia/reclamo …) -- the ones the terms of §8 start from."""
    analysis = state.get("analysis") or {}
    facts = ((analysis.get("fact_signals") or {}).get("facts") or {}) if isinstance(analysis, dict) else {}
    return sorted({str(e.get("kind") or "").strip().lower() for e in facts.get("events") or []
                   if isinstance(e, dict) and e.get("kind")})


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
    penal = criminal(posture)                                     # v79 (E2): the penal wording only then
    by_id = {s["id"]: s for s in sentences}
    notes, note_sources = attach_notes(all_records, {}, rows)       # v72: the lawyer's notes, into the records
    view_log: Dict[str, List[str]] = {}
    every = method_records(all_records, view_log)                   # v74: the decomposed fattispecie only
    absorbed: Dict[str, Any] = {"absorbed": [], "post_fact": []}
    juris_log: Dict[str, Any] = {}
    if penal:
        from . import penal_lines
        # v79.12.7 (item 3): the absorbed titles out (581/582 beside 584), 589/590 beside 584 the riqualificazione
        # alternative only, 593 out where a soccorso is narrated (a favourable element after the fact instead)
        every, absorbed = penal_lines.absorb(every, sentences)
        # (review) the flow's records agree: its own paths (its riti, the anchor blocks, the citations) never print an
        # absorbed title as autonomous
        penal_lines.absorb_into(state.get("records"), absorbed)
        # v79.12.7 (item 1): an element the note states and the text does not (584: the prevedibilità in concreto of the
        # death) into the record, inferred, «da provare»; a reviewed schema carrying it used as is (ELEMENT_SCHEMAS)
        penal_lines.jurisprudential_elements(every, schemas_on=schemas_enabled(), circ=all_records, log=juris_log)
    # v79.12.4 (item 4): an element naming the event the certain facts narrate is never «da provare»
    event_proven = narrated_event_proven(every, " ".join(str(x.get("text") or "") for x in sentences
                                                         if issue_records_certain(x)))
    binding = frame_binding(state, every)       # v79.14.2 (B): the verified frame binds the charge set (marks every)
    # v79 (D2): the ranked candidates are graded and enumerated (ctx["records"]), never what the answer prints or
    # assesses (the record below: §2, §4, the assessment, the objections' fallback). v79.14.3 (item 3): after the
    # binding -- a record its rescue made the charge is printed
    records = {x: r for x, r in every.items() if not r.get("ranked_candidate") and not r.get("unprinted")}
    if frame_admits_conduct(state):         # v79.14.3 (item 3): «Ammettere la condotta» only when the verified frame
        posture = {**(posture or {}), ADMISSION_VERIFIED: True}         # says the client admits it -- else never
    # v79.14.3 (item 4): excluded means excluded -- the records the binding excluded (and, below, the norms a verified
    # «no» removed) never in §4's tables, «Fatti mancanti», §7's objections; only «Altri profili» may name them
    fit_gone = [str(x.get("article")) for x in (state.get("fit_removed") or []) if isinstance(x, dict) and x.get("article")]
    shown = {x: r for x, r in records.items() if not frame_excluded(r) and not any(_same_label(x, y) for y in fit_gone)}
    excluded_dropped = [x for x in records if x not in shown]
    # v79.12.7 (item 2): the attenuanti the facts trigger (data/penal_lines.json «circumstance_triggers»: a diverbio, a
    # concrete step of the riparazione, a favourable element) -- candidates, held «da provare» by the fact
    triggers: Dict[str, Dict[str, Any]] = {}
    if penal and (posture or {}).get("role") in ACCUSED:
        from . import penal_lines
        triggers = penal_lines.circumstance_triggers(sentences, absorbed.get("post_fact") or [])
    circ = circumstance_records(circumstance_candidates(case, records, posture or {}, list(triggers)))
    circ_log: Dict[str, Any] = {}
    if circ and sentences:
        try:
            circ_log = grade_circumstances(circ, sentences, call=link_call)
        except Exception as exc:       # noqa: BLE001 -- ungraded: every element stays «da provare»
            circ_log = {"error": f"{type(exc).__name__}: {exc}"[:200]}
    for label, got in triggers.items():          # v79.12.7 (item 2): the trigger facts, after the grading
        if label in circ:
            circ[label]["triggered_by"] = [f["sid"] for f in got.get("facts") or [] if f.get("sid")] or ["caso"]
            circ[label]["trigger_step"] = got.get("step") or ""
    circ_notes, circ_sources = attach_notes({}, circ, rows)
    notes, note_sources = {**notes, **circ_notes}, {**note_sources, **circ_sources}
    certain = " ".join(str(x.get("text") or "") for x in sentences if issue_records_certain(x))
    if binding["applied"] and main in records and frame_excluded(records[main]):    # v79.14.2 (B): never the main
        binding["excluded"] = list(dict.fromkeys(binding["excluded"] + [main]))     # line's anchor
        main = ""
    a = assessment(records, posture or {}, main, client, event=event_of(certain))      # v79.10.3 (C2, review)
    if binding["empty"]:        # v79.14.2 (B): no fallback to the excluded -- the corpus line's path for the charge
        binding["materia"] = binding_materia(state, binding["mode"])
        # v79.14.3 (item 3): never the corpus line on a materia whose governing law is a code (corpus_line_on_code)
        if corpus_line_on_code(CORPUS_LINE.format(materia=binding["materia"]) + ".",
                               {"institutions": read_institutions(state.get("read") or {}), "records": records}):
            binding["charge_in_corpus"] = True
        a = bound_assessment(a, binding)
    # v79.10.3 (C1): the riti the charge's pena from the procedural store admits (charge_rites); a precluded one is §8's.
    # (review) the gate on what is written (ctx["rites"]: compose's rites_out) is the imputato's before the sentence
    # alone -- the persona offesa's, an appeal's or the execution's answer may describe any rito
    rites = charge_rites(records, rows, posture or {}, a.get("article") or "", circ) if _rites_posture(posture) else []
    riti = [x["line"] for x in rites if x["status"] != PRECLUDED] if _rites_posture(posture) else []
    # v79.12.7 (item 2): the fixed attenuanti block (the table's sentencing line) replaced by the attenuanti the facts
    # give (penal_lines.attenuanti_lines: a fact narrated, the concrete step named, «da verificare» without one, never
    # «esclusa» unless the record excludes it; art. 69's line whenever an aggravante is in the record)
    att_log: Dict[str, Any] = {}
    if riti and penal:
        from . import penal_lines
        att = penal_lines.attenuanti_lines(triggers, circ, sentences, att_log)
        sentencing = {x["line"] for x in rites if not x.get("rite")}
        riti = [x for x in riti if x not in sentencing] + att
    riti_out = [x["line"] for x in rites if x["status"] == PRECLUDED] if _rites_posture(posture) else []
    procedibilita = procedibility(records, rows)
    article_texts = [str((r.get("s") or {}).get("plain_text") or "") for r in rows or [] if r.get("_article")]
    facts_log: Dict[str, Any] = {}
    facts = facts_split(sentences, shown, penal, questions, article_texts, facts_log)     # v79.14.3 (item 4)
    note = aggravante_note(circ, records) if penal else ""          # v79 (E2): never in a civil matter
    when, when_points = temporal(case, posture or {}, abrogated)
    unknown = [x["name"] for x in rites if x["status"] == TO_VERIFY_RITE] if riti else []
    if unknown:                 # v79.10.3 (C1): the rito «da verificare sulla pena edittale» -- its point in §10
        when_points = list(when_points) + [RITE_POINT.format(names=" e ".join(unknown))]
    from . import applicability as applicability_
    # v79.13.0 (APPLICABILITY_FIT): the judgment's effects, when it ran -- (review) read only with the flag on
    fit = (state.get("applicability") or {}) if applicability_.enabled() else {}
    fit_contests: List[Dict[str, Any]] = []
    fit_objections: List[Dict[str, str]] = []
    if fit:
        # a «dipende»: its condizione in §10 as the user's question (its norm still in the answer)
        when_points = list(when_points) + applicability_.questions(fit.get("open") or [], list(records),
                                                                   fit.get("facts") or [])
        # a «no» on a norm the user names or the text charges: kept -- the judgment the defence's contest (the
        # client's line against a threat or a charge; else the other side's objection to the client's own norm);
        # (review) also when the norm has no method record (gated, undecomposed, no fattispecie)
        for kind, item in fit_kept(state, records):
            if kind == "contest":
                fit_contests.append(item)
            else:
                q, _ = applicability_.FIT_OBJECTION
                fit_objections.append({"obiezione": q.format(who=_other_side(posture or {}), art=item["article"],
                                                            perche=applicability_.clause(item["perche_printed"])),
                                       "risposta": applicability_.objection_answer(item)})     # v79.14.2 (A)
    ctx = {"case": case, "case_id": case_id_, "client": client, "rito": (posture or {}).get("rito") or "",
           "posture": posture or {}, "areas": list(frame_areas or ()), "procedibilita": procedibilita,
           "notes": notes,
           "posture_text": posture_text(posture or {}), "records": every, "circumstances": circ,
           "objection_party": objection_party(state, client, posture),   # v79.14.1 (item 6): the frame's under CASE_FRAME
           "fit_accepted": fit_accepted(state) if fit else [],            # v79.19.1 (item 6b): printed, never silent
           "facts": facts, "assessment": a, "riti": riti,
           "rites": rites, "riti_precluded": riti_out,                        # v79.10.3 (C1)
           "subsumption": subsumption_section(shown, order, by_id, circ, note),       # v79.14.3 (item 4): shown
           "temporal": when, "points": when_points,
           "litisconsorzio": litisconsorzio_lines(case, records, posture or {}),
           "stragiudiziale": stragiudiziale_line(posture or {}, procedibilita.get(a.get("article") or "", "")),
           "rulings": [r for r in rulings if isinstance(r, dict) and r.get("id") and r.get("massima")],
           "lines": list(lines),
           "objections_fallback": [o for o in fit_objections + objections_fallback(shown, a, penal, by_id, objections)
                                   if not cites_excluded(f"{o.get('obiezione') or ''} {o.get('risposta') or ''}",
                                                         excluded_dropped + fit_gone)],      # v79.14.3 (item 4)
           "open_questions": [point_sentence(str(q.get("text") or "")) for q in questions or []      # v75b: §10
                              if isinstance(q, dict) and q.get("text")],
           "charged_prose": charged_prose(shown, by_id),                      # v75b: from its sussunzione rows
           "store_fields": _store_fields(rows),                               # v79 (B1): a rule statement's sources
           "records_view": view_log,                                          # v79.7: the ranked trace reads it
           "institutions": read_institutions(state.get("read") or {}),        # v79.8 (B2)
           "family_overrides": list((state.get("family_gate") or {}).get("overrides") or []),   # v79.10 (A5, review)
           # v79.9 (B1): the stage's sources -- the certain facts and the events the terms read
           "certain_text": certain,
           "events": fact_events(state),
           "family_words": family_words(state, case),                        # v79.9 (C1)
           "charge": charge_view(state),                                      # v79.10.5 (E1): held, by its rule
           "event_proven": event_proven,                                      # v79.12.4 (item 4)
           "terms": computed_terms(state),                                    # v79.12.4 (item 7)
           # v79.12.7: the case's sentences (the riqualificazione's facts, the soccorso), the absorbed titles, the
           # favourable elements after the fact, the jurisprudential elements, what method_plan's refresh needs
           "sentences": list(sentences), "absorbed": absorbed.get("absorbed") or [],
           "post_fact": absorbed.get("post_fact") or [], "jurisprudential": list(juris_log.get("jurisprudential") or []),
           "flow_records": state.get("records") if penal else None,          # (review) absorbed_as_autonomous
           # v79.12.7 (item 2): the attenuanti the facts trigger and how each was printed
           "circumstance_triggers": {k: [f.get("sid") for f in v.get("facts") or []] for k, v in triggers.items()},
           "attenuanti_log": att_log,
           "refresh_inputs": {"main": main, "order": list(order), "aggravante_note": note,
                              "id_floor": max_element_id(all_records, every)}}
    ctx["frame_binding"] = binding                  # v79.14.2 (B)
    ctx["excluded_dropped"] = list(excluded_dropped)        # v79.14.3 (item 4): compose adds what it drops
    ctx["excluded_articles"] = list(dict.fromkeys(excluded_dropped + fit_gone))
    held_charge = str((ctx["charge"] or {}).get("article") or "")
    if binding["applied"] and held_charge and not binding_allows(binding["mode"], held_charge):
        binding_excluded(ctx, [held_charge])        # the held charge on the other side of the law: no charge printed
        ctx["charge"] = {}
    if binding["applied"]:
        if isinstance(state, dict):
            binding_counters(state.setdefault("block_counters", {}), binding)
        _log("frame_binding", {k: binding.get(k) for k in ("mode", "charge_set", "kept", "excluded", "empty",
                                                          "materia", "frame_charge", "charge_in_corpus", "rescued")})
    if fit:                     # v79.13.0: the kept-named contests (thesis_candidates), the removed (out of scope)
        ctx["fit_contests"] = fit_contests
        ctx["fit_removed"] = [str(x.get("article")) for x in state.get("fit_removed") or [] if x.get("article")]
    _log("prepare", {"assessment": a, "circumstances": {x: [r.get("status") for r in rec.get("requisites") or []]
                                                        for x, rec in circ.items()}, "circumstances_link": circ_log,
                     "aggravante_note": note, "riti": len(riti),
                     "rites": {x["id"]: [x["status"], x["why"]] for x in rites}, "temporal": ctx["temporal"][:200],
                     "litisconsorzio": len(ctx["litisconsorzio"]), "facts": {k: len(v) for k, v in ctx["facts"].items()},
                     "facts_excluded": facts_log, "procedibilita": procedibilita, "rulings": len(ctx["rulings"]),
                     "notes": note_sources, "records": view_log,
                     "absorbed": ctx["absorbed"], "jurisprudential": ctx["jurisprudential"],       # v79.12.7
                     "circumstance_triggers": ctx["circumstance_triggers"], "attenuanti": att_log})
    return ctx


def read_institutions(read: Dict[str, Any]) -> List[Dict[str, Any]]:
    """v79.8 (B2): the first read's institutions. v79.12.1: as the retrieval stage reads them
    (interactive_flow.read_institutions: {name, rubricated, claim}, the main line's first, the alternatives after) --
    active_modules gets the same input at both stages (module_input), institutions and not names only."""
    from .interactive_flow import read_institutions as flow_read_institutions
    return flow_read_institutions(read or {})


def _store_fields(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    try:
        from . import rule_statements
        return rule_statements.store_fields(rows)
    except Exception as exc:          # noqa: BLE001 -- no store fields: a rule traces to the elements or passages
        logger.warning("lawyer_method: store fields not read: %s", exc)
        return {}


SCRIMINANTE_COMMA = "comma della scriminante"        # v79.9 (D5)


def scriminante_commi(reqs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """v79.9 (D5): a cause di giustificazione's elements -- every one past its base comma (the first) marked special
    (copies), so split_special gives it a record of its own («52 c.p. co. 2»)."""
    def n(r):
        try:
            return int(str(r.get("comma") or 1).split()[0])
        except ValueError:
            return 1
    base = min((n(r) for r in reqs), default=1)
    return [r if n(r) == base or r.get("special") else {**r, "special": SCRIMINANTE_COMMA} for r in reqs]


def touched(rec: Dict[str, Any]) -> bool:
    """v79.9 (D5): a fact of the case touches the record -- an element proven or excluded, or a sentence linked to one."""
    return any(r.get("status") in (PROVATO, NON_RISULTA) or r.get("supporting") or r.get("contradicting")
               for r in _live(rec))


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
        schema = rec.get("schema") if isinstance(rec.get("schema"), dict) and schemas_enabled() else None   # v79.11
        if rec.get("not_fattispecie") or not issue_records.decomposed_by_gate(
                label, str(rec.get("rubrica") or ""), defence=rec.get("decomposed_for") == issue_records.DEFENCE_WHY,
                **({"schema": schema} if schema is not None else {})):
            log.setdefault("gated", []).append(label)
            continue
        if rec.get("undecomposed") or not issue_records.decomposed(rec):
            if rec.get("requisites") or rec.get("undecomposed"):
                log.setdefault("undecomposed", []).append(label)
            continue
        # v79.8 (D1): a special comma is a record of its own, never elements of the base rule's; v79.9 (D5): a
        # scriminante's every comma after its base one too (52 co. 2-4) -- its thesis lists the base comma's elements
        reqs = [r for r in rec["requisites"] if r.get("kind") == "requisito"]
        if issue_records.justification(label):
            reqs = scriminante_commi(reqs)
        for x, part in issue_records.split_special(label, {**rec, "requisites": reqs}).items():
            if part["requisites"]:
                out[x] = part
            if x != label:
                log.setdefault("special_commi", []).append(x)
    for label in [x for x, r in out.items() if r.get("ranked_candidate")]:     # v79 (D2): a candidate that grades
        live = _live(out[label])                                               # to nothing, gone without a word
        if not any(r.get("status") == PROVATO for r in live) or any(r.get("status") == NON_RISULTA for r in live):
            log.setdefault("ranked_dropped", []).append(label)
            del out[label]
    noise = [x for x, r in out.items() if fact_free(r) and not kept_without_facts(r) and not r.get("ranked_candidate")]
    if noise and len(noise) < len(out):     # v78: tabled with «nessun fatto» rows only -- out, unless nothing else is left
        log["no_fact"] = noise
        for x in noise:     # v79.7 (item 2): an alternative the first read listed stays a riqualificazione candidate,
            if out[x].get("alternative"):           # never printed (prepare's record leaves it out)
                out[x]["unprinted"] = True
        out = {x: r for x, r in out.items() if x not in noise or r.get("unprinted")}
    return out


EVENT_PROVEN_WHY = "evento narrato dai fatti certi ({event})"        # v79.12.4 (item 4), my draft
# v79.12.4 (item 4, review): the event as an element's whole wording -- the bare event («la morte di una persona», «una
# lesione personale», «una malattia nel corpo o nella mente», «l'evento lesivo»), never one with a qualifier (gravity:
# «lesioni gravi», «superiore ai quaranta giorni»; means: «con armi»), nor a conduct that only mentions the victim's
# state (593: «trovando una persona ferita … omette»)
_DET_RE = r"(?:(?:la|le|il|lo|l['’]|una|un|uno|un['’])\s*)?"
_VICTIM_RE = (r"(?:\s+(?:di|del|della|dello|dell['’]|d['’])\s*" + _DET_RE +
              r"(?:uomo|donna|persona|person[ae]\s+offes[ae]|vittima|altr[oaie]|soggetto|individuo|minore|"
              r"lavorator[ei]|passeggero|pedone))?")
_BARE_EVENT = {
    "morte": r"(?:l['’]\s*evento\s+(?:morte|mortale|letale)|" + _DET_RE + r"(?:morte|decesso)" + _VICTIM_RE + ")",
    "lesioni": (r"(?:l['’]\s*evento\s+lesivo|" + _DET_RE + r"(?:lesion[ei](?:\s+personal[ei])?|malattia(?:\s+nel\s+"
                r"corpo\s+o\s+nella\s+mente)?)" + _VICTIM_RE + ")")}
# the causal link stated with it: a leading cause («cagiona [per colpa] [ad alcuno]»), a «dalla quale deriva» joining
# two bare events (582), a trailing consequence («come conseguenza degli atti»)
_CAUSAL_HEAD_RE = re.compile(r"^(?:chiunque\s+)?(?:(?:che\s+)?(?:cagion|caus|provoc|determin)\w*(?:\s+(?:per\s+colpa|"
                             r"ad\s+alcuno|a\s+taluno|ad\s+altri|a\s+una\s+persona|involontariamente))*\s+)",
                             re.IGNORECASE)
_CAUSAL_JOIN_RE = r",?\s+(?:dalla\s+quale|dal\s+quale|da\s+cui|dalle\s+quali|dai\s+quali)\s+deriv\w*\s+"
_CAUSAL_TAIL_RE = re.compile(r",?\s+(?:come|quale)\s+conseguenz\w*\b.*$|,?\s+(?:conseguent|derivant|cagionat|"
                             r"causat|provocat)\w*\s+(?:a|da|dal|dalla|dagli|dai|dall['’]).*$", re.IGNORECASE)


def event_element(text: str, event: str) -> str:
    """v79.12.4 (item 4, review): how the element states the event -- «bare» (its whole wording, _BARE_EVENT),
    «causal» (the bare event with its cause, _CAUSAL_*: «cagiona la morte di un uomo», «la morte come conseguenza
    degli atti»: the narration proves the event, never the causal link), "" otherwise (no event, a qualifier, a
    conduct)."""
    bare = _BARE_EVENT.get(str(event or ""))
    if not bare:
        return ""
    t = " ".join(str(text or "").lower().split()).strip(" .;:,«»\"'")
    if re.fullmatch(bare, t):
        return "bare"
    rest = _CAUSAL_TAIL_RE.sub("", _CAUSAL_HEAD_RE.sub("", t)).strip(" .;:,")
    if rest != t and re.fullmatch(f"{bare}(?:{_CAUSAL_JOIN_RE}{bare})?", rest):
        return "causal"
    return ""


def names_event(text: str, event: str) -> bool:
    """v79.12.4 (item 4): the element's whole wording is the event the certain facts narrate (event_element «bare»)."""
    return event_element(text, event) == "bare"


def narrated_event_proven(records: Dict[str, Dict[str, Any]], certain: str) -> List[str]:
    """v79.12.4 (item 4): an event the certain facts narrate is never «non provato» -- each live element «da provare»
    whose own words name that event (names_event: «la morte di un uomo», «l'evento lesivo» with a death or injuries
    among the certain facts; never one that joins it to its cause, «cagiona la morte», whose causal link the narration
    does not prove) is «provato», its why EVENT_PROVEN_WHY and r["event_narrated"]; an element «non risulta» (a fact
    against it) stays. Returns «<article>: <id or text>» of each, logged."""
    event = event_of(certain or "")
    if not event:
        return []
    out: List[str] = []
    causal: List[str] = []
    for label, rec in (records or {}).items():
        for r in _live(rec):
            if r.get("status") != DA_PROVARE or (r.get("fit_open") and _fit_on()):   # v79.13.0: a «dipende» stays
                continue
            how = event_element(str(r.get("text") or ""), event)
            if how == "bare":
                r.update(status=PROVATO, event_narrated=event, why=EVENT_PROVEN_WHY.format(event=event))
                out.append(f"{label}: {r.get('id') or _short(r.get('text'), 60)}")
            elif how == "causal":       # (review) the event narrated, the causal link still to prove -- marked, logged
                r["event_narrated"] = event
                causal.append(f"{label}: {r.get('id') or _short(r.get('text'), 60)}")
    if out or causal:
        _log("event_narrated_proven", {"event": event, "elements": out, "causal_open": causal})
    return out


def event_causal_open(records: Dict[str, Dict[str, Any]], certain: str) -> int:
    """v79.12.4 (item 4, review): the live elements stating the narrated event with its cause (event_element «causal»)
    still «da provare» -- the causal link is what remains to prove; the nightly's event_narrated_causal (informative)."""
    event = event_of(certain or "")
    return 0 if not event else sum(1 for rec in (records or {}).values() for r in _live(rec)
                                   if r.get("status") == DA_PROVARE
                                   and event_element(str(r.get("text") or ""), event) == "causal")


def event_unproven(records: Dict[str, Dict[str, Any]], certain: str) -> int:
    """v79.12.4 (item 4): the live elements naming the event the certain facts narrate still «da provare» -- must be 0."""
    event = event_of(certain or "")
    return 0 if not event else sum(1 for rec in (records or {}).values() for r in _live(rec)
                                   if r.get("status") == DA_PROVARE and names_event(str(r.get("text") or ""), event))


def fact_free(rec: Dict[str, Any]) -> bool:
    """v78: no case sentence touches any of the record's elements -- none supports, none contradicts, none «provato»
    or «non risulta» (590-bis, 583-quater, 2044 c.c. on a victim's lesioni: «nessun fatto» on every row)."""
    live = _live(rec)
    return bool(live) and all(r.get("status") == DA_PROVARE and not r.get("supporting") and not r.get("contradicting")
                              for r in live)


def kept_without_facts(rec: Dict[str, Any]) -> bool:
    """v78: a record the method keeps with no fact: the charge or the claim (rec["charged"], or decomposed as one), the
    counterparty's claim (side threat), an article the user named."""
    return bool(rec.get("charged") or rec.get("side") == "threat" or rec.get("named_due") or by_concept(rec)
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
    unsourced, outside, wording = [], [], []
    skip = title("orientamenti")
    block = "\n".join(f"{a}" for _, a in answers_of(answer))           # v79.6: the answers are checked too
    for t, text in [("risposte", block)] + answer_sections(version.strip_footer(answer or "")):
        if t == skip:
            continue
        wording += wording_hits(text, (out or {}).get("element_ids") or [], bool((out or {}).get("criminal", True)))
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
    return {"unsourced": unsourced, "out_of_scope": list(dict.fromkeys(outside)), "wording": wording}


def stage_lines(posture: Dict[str, Any]) -> List[str]:
    """v79.12.5 (item 10): the lines of the posture's closed list (render_computed.stage_blocks, every block), each
    without its bullet -- code-written: §8 keeps them whatever the rule check reads in them (the ATP line restored)."""
    try:
        from . import render_computed
        blocks = render_computed.stage_blocks(posture or {})
    except Exception:           # noqa: BLE001 -- no list, nothing kept beyond the check's own
        return []
    return [ln.strip().lstrip("-").strip() for _, body in blocks for ln in str(body).split("\n") if ln.strip()]


def no_claim_layout(solutions: str) -> bool:
    """v79.12.5 (item 2): §6 is the no-claim one (no_claim_solutions)."""
    return NO_CLAIM_HEAD.split(":")[0] in (solutions or "")


# --- v79.14.1 (item 8): the «requisiti provati N su M» count never in prose, on any path ------------------------------------
# v79.14.1 (item 8, review): the count in any of its wordings -- «requisiti provati: 0 su 5», «requisiti provati 0/5»,
# «i requisiti provati sono 0 su 5», «dei requisiti, provati 0 su 5», «risultano provati 0 requisiti su 5», «sono
# provati 0 su 5 requisiti», «sono provati 2 dei 5 requisiti»; never §9's assessment reason («3 elementi su 5 dell'art.
# 646 c.p. risultano provati; gli altri …»: the method's verdict on the named article, its order not a count's)
COUNT_SENTENCE_RE = re.compile(
    r"\b(?:requisiti|elementi)\b[^.;\n]{0,40}?\bprovati\s*(?:sono\s*)?:?\s*\d+\s*(?:su|/|di)\s*\d+|"
    r"\bprovati\s*:?\s*\d+\s+(?:(?:su|dei|degli|di)\s+\d+\s+(?:requisiti|elementi)\b|(?:requisiti|elementi)\s+(?:su|di)\s+"
    r"\d+)", re.IGNORECASE)
_COUNT_PAREN_RE = re.compile(r"[ \t]*\(([^()\n]{0,100})\)")


def count_out(text: str, log: Optional[List[str]] = None) -> str:
    """v79.14.1 (item 8): the count «requisiti provati N su M» out of the prose -- a parenthetical one cut (the sentence
    keeps its verdict: «Esito sui fatti del caso: da provare.»), a sentence still holding one removed; each into log."""
    found = log if log is not None else []

    def cut(m: "re.Match") -> str:
        if not COUNT_SENTENCE_RE.search(m.group(1)):
            return m.group(0)
        found.append(m.group(0).strip())
        return ""
    out = _COUNT_PAREN_RE.sub(cut, text or "")
    if not COUNT_SENTENCE_RE.search(out):
        return out if found else (text or "")

    def review(sentence: str) -> str:
        if COUNT_SENTENCE_RE.search(sentence):
            found.append(sentence.strip()[:160])
            return ""
        return sentence
    return _by_sentence(out, review)


def count_left(text: str) -> List[str]:
    """v79.14.1 (item 8): the counts still in a text (must be none in a delivered answer)."""
    return [m.group(0) for m in COUNT_SENTENCE_RE.finditer(text or "")]


_CLIENT_LAYOUT_RE = re.compile(r"(?m)^\*\*Nell'interesse di [^*\n]+\*\*\s*$|^_Sezione comune a [^\n]+_\s*$")


def structure_checks(answer: str, cell: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """[{check, ok, why}]: the ten sections present and non-empty, in order; §4's tables; §6's table with its realismo,
    the stragiudiziale option; three objections; §9's assessment lines; no «da verificare» outside §10; and by the
    cell: a civil claimant before filing has mediazione, ATP and 700 in §8, an imputato the riti alternativi."""
    cell = cell or {}
    raw_secs = answer_sections(answer)
    # v79.6 (item 2): a several-client parere's per-client heads and shared notes are layout, not section text
    secs = [(t, _CLIENT_LAYOUT_RE.sub("", x).strip()) for t, x in raw_secs]
    by = {t: x for t, x in secs}
    names = titles()
    out = []

    def check(name: str, ok: bool, why: str = "") -> None:
        out.append({"check": name, "ok": bool(ok), "why": "" if ok else why})

    check("sections", [t for t, _ in secs] == names, f"found {[t for t, _ in secs]}")
    for t in names:
        check(f"non_empty:{t}", bool((by.get(t) or "").strip()), "empty or missing")
    summary = dict(raw_secs).get(title("sintesi")) or ""
    parts = [x for x in re.split(r"(?m)^\*\*Nell'interesse di [^*\n]+\*\*$", summary) if x.strip()] or [summary]
    check("summary_max_lines", all(len([x for x in _SENT_SPLIT.split(p) if x.strip()]) <= SUMMARY_LINES
                                   for p in parts), f"more than {SUMMARY_LINES} sentences")
    sub = by.get(title("norme")) or ""
    check("subsumption_table", "| Elemento | Fatto che lo integra | Esito |" in sub, "no element table")
    check("no_count_sentence", not count_left(answer), "a «requisiti provati N su M» line")    # v79.14.1 (item 8)
    sol = by.get(title("soluzioni")) or ""
    con = by.get(title("conclusioni")) or ""
    if no_claim_layout(sol):    # v79.12.5 (item 2): no admitted claim -- no table, no line, no forecast, said so
        check("no_claim_no_forecast", not re.search(r"esito\s+probabile|linea\s+principale", answer or "", re.I)
              and NO_CLAIM_CONCLUSION in con, "a forecast or a main line with no admitted claim")
    else:
        check("solutions_table", "| Tesi | Pro | Contro | Realismo |" in sol, "no solutions table")
        check("stragiudiziale", "Opzione stragiudiziale:" in sol, "no stragiudiziale option in the strategy")
        obj = by.get(title("obiezioni")) or ""
        if ALL_OPEN_PREFIX in obj:      # v79.14.1 (item 6): every element open -- one objection or none, no template
            check("objections_all_open", not OPEN_TEMPLATE_RE.search(obj) and len(re.findall(
                r"(?m)^\*\*Obiezione \d+\.\*\*", obj)) <= obj.count(ALL_OPEN_PREFIX),
                "more than one objection, or the per-element template, with every element open")
        else:
            check("three_objections", len(re.findall(r"(?m)^\*\*Obiezione \d+\.\*\*", obj)) >= OBJECTIONS,
                  "fewer than three objections")
        check("objection_template_once", not template_repeats(answer), "the per-element «da provare» template "
                                                                        "repeated")     # v79.14.1 (item 6)
        for line in ("**Esito probabile**", "**Grado di certezza**", "**Obiettivo realistico**"):
            check(f"conclusions:{line.strip('*')}", line in con, "missing")
        check("obiettivo_label", bool(re.search(
            r"\*\*Obiettivo realistico\*\*: (?:esito pieno|esito parziale|contenimento)", con)),
            "no esito pieno / parziale / contenimento")
    from . import rulings as rulings_mod
    orient = by.get(title("orientamenti")) or ""
    registry_layout = rulings_mod.CORPUS_LINE in orient         # v78: §5 the registry's (rulings.section_text)
    body = "\n".join(("" if registry_layout else
                      re.sub(rf"(?m)^- [^\n]*: {ORIENTATION_TO_VERIFY}$|^{ORIENTATION_TO_VERIFY}$", "", x))
                     if t == title("orientamenti") else x for t, x in secs if t != title("verificare"))
    body = body.replace(RITE_TO_VERIFY, "")     # v79.10.3 (C1): a rito's «da verificare sulla pena edittale»: its own
    check("markers_collected", not re.search(r"\bda verificare\b", body, re.I), "«da verificare» in the body")
    # v79.8 (E1): «alto» never with something it depends on
    check("certainty_depends", not re.search(r"\*\*Grado di certezza\*\*: alto\s*—\s*dipende da", answer or "", re.I),
          "«Grado di certezza: alto» with a non-empty «dipende da»")
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
