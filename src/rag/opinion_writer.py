"""v79.19.9 (Part 2, rows 112-113 E): the opinion writer, behind OPINION_WRITER (default off: nothing is called, nothing
changes).

After the analysis step and render_computed.render, one chat call (max_tokens 3000; thinking asked, as the objections
ask it -- the LLM_THINKING setting decides) with:
  system: Antonelo's text of row 112 verbatim (SYSTEM_ROW_112), plus three lines (SYSTEM_RAILS): cite only the articles
          whose texts follow; no rulings nor ruling numbers; terms and rites only from the «Profili processuali
          (calcolati)» block;
  human:  the case text; the article texts of the live records graded 2 or 3 (the step's verified lines first), each
          article's whole text from the graph (the answer's rows), at most PER_ARTICLE_CHARS (9,000) per article and
          TOTAL_CHARS (45,000) in all; then the code-rendered procedural section under PROCEDURE_HEADING.
The prose then passes four code checks (checks), each logged under «opinion_writer»:
  articles_removed   a sentence citing an article (citations.article_refs, by key) not among the given texts: removed;
  rulings_stripped   «Cass.» / «sent.» / «ord.» followed by a number: the reference stripped (the rulings section stays
                     the code's);
  terms_flagged      a date or a term («entro N giorni/mesi/anni», «dal <date>», «prescrizione …», «decadenza …») not
                     present in the rendered procedural section: its sentence suffixed «[da verificare]»;
  facts_contested    the step's verified «provato» elements whose cited fact the prose marks as unproven («da provare»,
                     «non risulta», «allegazione»): counted, not edited.
The flow (interactive_flow.opinion_answer) prints the checked prose with the code-rendered procedural section as its
own titled section and, through step_aside_answer, the code-made rulings section; the page is still built and kept
(state «page_text» beside «opinion_text», the session result's «page_text» / «opinion_text», the event
interactive_opinion_writer {text, checks, …}). Retrieval, grading, records, the analysis step, the fit: untouched.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import feature_flags

logger = logging.getLogger(__name__)

FLAG = "OPINION_WRITER"
MAX_TOKENS = 3000
PER_ARTICLE_CHARS = 9000
TOTAL_CHARS = 45000
GRADES = (2, 3)
PROCEDURE_HEADING = "Profili processuali (calcolati)"
VERIFY_MARK = "[da verificare]"

# Antonelo's system text (strategy_review_ceiling_28nov.md, row 112, «The opinion writer we used»), verbatim.
SYSTEM_ROW_112 = (
    "Sei un avvocato italiano con lunga esperienza. Scrivi un parere legale per chi lo chiede nel caso, come lo "
    "consegneresti a un collega: in prosa, senza tabelle e senza elenchi di frammenti di norma.\n"
    "Il parere contiene, in quest'ordine: (1) la qualificazione giuridica dei fatti e la posizione dell'assistito; (2) "
    "la linea principale, argomentata applicando le norme ai fatti del caso, e una linea subordinata se esiste; (3) la "
    "risposta a ciascuna domanda posta nella richiesta (rimedi, contro chi, termini, oneri di prova, che cosa conviene, "
    "che cosa si può ottenere), una per una; (4) i profili processuali essenziali; (5) i rischi.\n"
    "Regole: sostieni la posizione dell'assistito; non contestare ciò che il caso dà per pacifico; cita solo norme di "
    "cui sei certo; se sotto trovi i testi di alcuni articoli, fonda il parere su quelli che servono, non attribuire "
    "loro contenuti che non hanno e ignora quelli che non c'entrano; non inventare sentenze né numeri di sentenze. "
    "Massimo 1.200 parole.")
# The three lines the brief adds (79.19.9).
SYSTEM_RAILS = (
    "Cita solo gli articoli i cui testi trovi sotto.\n"
    "Non citare sentenze né numeri di sentenze.\n"
    f"Per termini e riti usa esclusivamente il blocco \"{PROCEDURE_HEADING}\" riportato sotto, senza aggiungerne altri.")
SYSTEM = SYSTEM_ROW_112 + "\n" + SYSTEM_RAILS

CASE_HEADING = "CASO"
ARTICLES_HEADING = "TESTI DEGLI ARTICOLI"


def enabled() -> bool:
    return feature_flags.enabled(FLAG)


def _default_call(messages: List[Any], max_tokens: int) -> str:
    """The chat call: defensive_generation._call_chat with thinking asked where the installed _call_chat takes it."""
    from . import defensive_generation as dg
    from .lawyer_method import thinking_supported
    return dg._call_chat(messages, max_tokens=max_tokens, **({"thinking": True} if thinking_supported() else {}))


# --- the material -----------------------------------------------------------------------------------------------------------

def _key(label: str) -> Tuple[str, str]:
    from .interactive_flow import _key as key_, _parse_label
    return key_(_parse_label(label))


def line_labels(verified: Optional[Dict[str, Any]]) -> List[str]:
    """The step's verified lines' articles, leading first ([] without a verified view)."""
    a = ((verified or {}).get("analysis") or {}) if isinstance(verified, dict) else {}
    out = []
    for kind in ("linea_principale", "linea_subordinata"):
        x = a.get(kind) or {}
        if isinstance(x, dict) and x.get("articolo"):
            out.append(str(x["articolo"]))
    return out


def article_labels(state: Dict[str, Any], verified: Optional[Dict[str, Any]]) -> List[str]:
    """The live records graded 2 or 3, the step's lines first (a line's record whatever its grade), each once."""
    records = state.get("records") or {}
    keys_of = {_key(label): label for label in records}
    out: List[str] = []
    for label in line_labels(verified):
        have = keys_of.get(_key(label))
        if have and not (records[have] or {}).get("dropped") and have not in out:
            out.append(have)
    for label, rec in records.items():
        if isinstance(rec, dict) and not rec.get("dropped") and rec.get("grade") in GRADES and label not in out:
            out.append(label)
    return out


def article_texts(labels: Sequence[str], rows: Sequence[Dict[str, Any]], records: Dict[str, Any]
                  ) -> List[Dict[str, Any]]:
    """[{label, rubrica, text, chars, cut}] for the labels whose row holds a text (the graph's plain_text), each cut
    at PER_ARTICLE_CHARS, the whole at TOTAL_CHARS (an article past the total is left out, listed «omitted»)."""
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in rows or []:
        if r.get("_article"):
            from .interactive_flow import _key as key_
            by_key.setdefault(key_(r["_article"]), r)
    out: List[Dict[str, Any]] = []
    total = 0
    for label in labels:
        row = by_key.get(_key(label))
        sec = (row or {}).get("s") or {}
        text = " ".join(str(sec.get("plain_text") or "").split())
        if not text:
            out.append({"label": label, "rubrica": "", "text": "", "chars": 0, "cut": False, "omitted": "no_text"})
            continue
        cut = len(text) > PER_ARTICLE_CHARS
        text = text[:PER_ARTICLE_CHARS]
        if total + len(text) > TOTAL_CHARS:
            out.append({"label": label, "rubrica": "", "text": "", "chars": 0, "cut": False, "omitted": "total"})
            continue
        total += len(text)
        rubrica = str(sec.get("titolo_articolo") or (records.get(label) or {}).get("rubrica") or "").strip()
        out.append({"label": label, "rubrica": rubrica, "text": text, "chars": len(text), "cut": cut})
    return out


def human_message(case: str, texts: Sequence[Dict[str, Any]], rendered: str) -> str:
    parts = [f"{CASE_HEADING}:\n{(case or '').strip()}"]
    given = [t for t in texts if t.get("text")]
    if given:
        blocks = [f"ARTICOLO: {t['label']}" + (f" — {t['rubrica']}" if t.get("rubrica") else "") + f"\n{t['text']}"
                  for t in given]
        parts.append(f"{ARTICLES_HEADING}:\n\n" + "\n\n".join(blocks))
    if (rendered or "").strip():
        parts.append(f"{PROCEDURE_HEADING}:\n{rendered.strip()}")
    return "\n\n".join(parts)


# --- the four checks --------------------------------------------------------------------------------------------------------

_RULING_RE = re.compile(
    r"\(?\s*(?:Cass(?:azione)?\.?(?:\s*(?:pen|civ)\.?)?(?:\s*,?\s*[Ss]ez\.?\s*[\w.]+)?\s*,?\s*(?:n\.?\s*)?\d+[\w./-]*"
    r"(?:\s*/\s*\d{2,4})?|\b(?:sent|ord)\.\s*(?:n\.?\s*)?\d+[\w./-]*(?:\s*/\s*\d{2,4})?)\s*\)?", re.IGNORECASE)
_NUMBER_WORDS = (r"(?:\d+|un[oa]?|due|tre|quattro|cinque|sei|sette|otto|nove|dieci|undici|dodici|quindici|venti|trenta|"
                 r"quaranta|sessanta|novanta|cento|centoventi|centottanta)")
_TERM_RE = re.compile(
    rf"\bentro\s+(?:i\s+|il\s+|le\s+|gli\s+)?{_NUMBER_WORDS}\s+(?:giorni|mesi|anni|giorno|mese|anno)\b|"
    rf"\b{_NUMBER_WORDS}\s+(?:giorni|mesi|anni)\b|"
    r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b|"
    r"\b\d{1,2}\s+(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)"
    r"(?:\s+\d{4})?\b|"
    r"\bdal\s+\d{1,2}(?:[./]\d{1,2}[./]\d{2,4}|\s+(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|"
    r"settembre|ottobre|novembre|dicembre))\b|"
    r"\b(?:prescrizion[ei]|decadenz[ae])\b", re.IGNORECASE)
_UNPROVEN_RE = re.compile(r"\bda\s+provare\b|\bnon\s+risulta\b|\ballegazion[ei]\b", re.IGNORECASE)
_SENT_SPLIT = re.compile(r"(?<!\bn\.)(?<!\bart\.)(?<!\bartt\.)(?<!\bco\.)(?<!\bd\.lgs\.)(?<!\bd\.P\.R\.)(?<!\bd\.l\.)"
                         r"(?<!\bL\.)(?<!\bCass\.)(?<!\bsent\.)(?<!\bord\.)(?<=[.!?;])\s+(?=[A-ZÀ-ÖØ-Ý«\"“(\[*\d-])")


def _sentences(paragraph: str) -> List[str]:
    return [x for x in _SENT_SPLIT.split(paragraph) if x.strip()]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _term_present(term: str, rendered_norm: str) -> bool:
    """The term's words (its number and unit, its date, its «prescrizione» / «decadenza») are in the rendered section."""
    t = _norm(term)
    if t in rendered_norm:
        return True
    words = [w for w in re.findall(r"[\wàèéìòù./]+", t) if w not in ("entro", "dal", "i", "il", "le", "gli")]
    return bool(words) and all(w in rendered_norm for w in words)


def _stems(text: str) -> set:
    return {w[:5] for w in re.findall(r"[a-zà-ü]{4,}", (text or "").lower())}


def proven_facts(verified: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    """The step's verified «provato» elements with their cited facts' texts: [{article, element, fact}]."""
    v = verified if isinstance(verified, dict) else {}
    a = v.get("analysis") or {}
    sents = v.get("sentences") or {}
    out = []
    for article, els in (a.get("elementi") or {}).items():
        for e in els or []:
            if not isinstance(e, dict) or e.get("esito") != "provato":
                continue
            for sid in e.get("cita_fatti") or []:
                fact = str(sents.get(str(sid)) or "")
                if fact:
                    out.append({"article": str(article), "element": str(e.get("elemento") or ""), "fact": fact})
    return out


def checks(text: str, given: Sequence[str], rendered: str, verified: Optional[Dict[str, Any]]
           ) -> Tuple[str, Dict[str, Any]]:
    """The four code checks on the prose: (the text as printed, {"articles_removed": [...], "rulings_stripped": [...],
    "terms_flagged": [...], "facts_contested": [...]})."""
    from . import citations
    allowed = {_key(x) for x in given}
    rendered_norm = _norm(rendered)
    log: Dict[str, Any] = {"articles_removed": [], "rulings_stripped": [], "terms_flagged": [], "facts_contested": []}
    facts = proven_facts(verified)
    out_paragraphs: List[str] = []
    for paragraph in (text or "").split("\n"):
        if not paragraph.strip():
            out_paragraphs.append(paragraph)
            continue
        kept: List[str] = []
        for sentence in _sentences(paragraph):
            refs = citations.article_refs(sentence)
            outside = [citations.label(r) for r in refs if _key(citations.label(r)) not in allowed]
            if outside:
                log["articles_removed"].append({"articles": list(dict.fromkeys(outside)), "sentence": sentence[:200]})
                continue
            stripped = sentence
            for m in list(_RULING_RE.finditer(sentence)):
                log["rulings_stripped"].append(m.group(0).strip())
            stripped = _RULING_RE.sub("", stripped)
            stripped = re.sub(r"\(\s*\)", "", stripped)
            stripped = re.sub(r"\s+([,.;:])", r"\1", re.sub(r"\s{2,}", " ", stripped)).strip()
            terms = [m.group(0) for m in _TERM_RE.finditer(stripped)]
            missing = [t for t in terms if not _term_present(t, rendered_norm)]
            if missing and not stripped.endswith(VERIFY_MARK):
                log["terms_flagged"].append({"terms": list(dict.fromkeys(missing)), "sentence": stripped[:200]})
                stripped = f"{stripped} {VERIFY_MARK}"
            if facts and _UNPROVEN_RE.search(stripped):
                mine = _stems(stripped)
                for f in facts:
                    theirs = _stems(f["fact"])
                    if theirs and len(mine & theirs) >= max(2, min(3, len(theirs) // 2)):     # the fact restated
                        log["facts_contested"].append({**f, "sentence": stripped[:200]})
            kept.append(stripped)
        if kept:
            out_paragraphs.append(" ".join(kept))
    return "\n".join(out_paragraphs).strip(), log


# --- the call ---------------------------------------------------------------------------------------------------------------

def write(state: Dict[str, Any], verified: Optional[Dict[str, Any]], rows: Sequence[Dict[str, Any]], case: str,
          rendered: str, call: Optional[Callable[[List[Any], int], str]] = None) -> Dict[str, Any]:
    """The opinion on the answer's material: {"text" (checked), "raw", "articles" [labels given], "texts" [{label,
    chars, cut, omitted?}], "checks", "seconds", "error", "prompt_chars"}. A failed call: text "" and the error (the
    page is printed as today)."""
    from langchain_core.messages import HumanMessage, SystemMessage
    t0 = time.time()
    records = state.get("records") or {}
    labels = article_labels(state, verified)
    texts = article_texts(labels, rows, records)
    given = [t["label"] for t in texts if t.get("text")]
    human = human_message(case, texts, rendered)
    messages = [SystemMessage(content=SYSTEM), HumanMessage(content=human)]
    raw, error = "", ""
    try:
        raw = str((call or _default_call)(messages, MAX_TOKENS) or "")
    except Exception as exc:        # noqa: BLE001 -- no opinion: the page stands
        error = f"{type(exc).__name__}: {exc}"[:300]
        logger.warning("opinion_writer: the call failed: %s", error)
    text, log = checks(raw, given, rendered, verified) if raw.strip() else ("", {
        "articles_removed": [], "rulings_stripped": [], "terms_flagged": [], "facts_contested": []})
    return {"text": text, "raw": raw, "articles": given, "lines": line_labels(verified),
            "texts": [{k: v for k, v in t.items() if k != "text"} for t in texts], "checks": log,
            "seconds": round(time.time() - t0, 1), "error": error, "prompt_chars": len(SYSTEM) + len(human),
            "max_tokens": MAX_TOKENS}


def final_text(opinion: str, rendered: str) -> str:
    """The opinion as printed: the prose, then the code-rendered procedural section as its own titled section (the
    page's title, plan_then_write.SECTION_TITLES[4], which step_aside_answer leaves unchecked)."""
    from . import plan_then_write as ptw
    out = (opinion or "").strip()
    if (rendered or "").strip():
        out += f"\n\n**5. {ptw.SECTION_TITLES[4]}**\n\n{rendered.strip()}"
    return out
