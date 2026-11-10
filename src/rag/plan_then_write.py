"""PLAN_THEN_WRITE_ENABLED: the parere as a plan, then its sections written from the plan.

Flavio's handover of 22 Oct (6.3, "the main lever"): qwen2.5-32b is reliable on short focused
prompts, not on one long one. The lawyer's review of the held-out drafts found the wrong side
argued, the central question missed or conceded, rulings without court, section and number,
and no practical content. The PM's test cases of 23 Oct showed the same: the pupil's case was
argued for the school although the stance check had named the parents as the client, and the
burglary case's procedural question got "processo ordinario".

  1. The plan, one call at temperature 0, as JSON: the client and their goal, the central
     question in one sentence, an answer to each of the user's questions, and per issue the
     governing articles (from NORME, by their text), the rulings (from GIURISPRUDENZA, by full
     reference), the client's argument, the strongest opposing one and the rebuttal; then the
     subordinate requests and the weak points. Articles and rulings not among those given are
     removed in code. If the stance check (defensive_generation._opposing_arguments) finds an
     argument for the other side, the plan is made once more with that list.
  2. The practical checks of the case's posture and area (practical_checklist.py), given to
     the plan and to the procedural section.
  3. The plan checked before anything is written (check_plan, round 2 of the PM's cases):
     each issue's articles against their real heading and text, one short call per issue in
     parallel. An article that does not govern what the argument uses it for is removed
     (380/274 c.p.p. given as "procedura abbreviata", 657/658 c.p.c. for "sanzioni" in a
     right-of-way case); an issue whose argument contradicts its article's text is dropped
     ("la penale richiede la prova del danno" against art. 1382 c.c.).
  4. Six sections written in parallel, each call seeing only its own material; rulings are
     cited only where their passages are (an answer section citing a ruling it had not read
     quoted words the ruling does not contain).

Any failure (no plan, a plan without an argued issue, a section failing twice) returns None:
the pipeline then writes the parere in one call, as before.

PLAN_KEEP_APPLICABLE_ENABLED (heldout2, 24 Oct: MENTIONED 12.7/35 with the plan, 16.0 without; n4's
95/98 d.lgs. 36/2023 and sempronio's 581/582 lost): the plan keeps the breadth of the articles the
applicability check let through (_applicable: judged applicabile or alternativa, or exempt from it --
named in the case, found by their heading, cross-referenced -- in the case's areas). The plan is told
to place each under the issue it governs; the plan check never removes them (as the _vote ones);
those the plan left out make a last issue, "Altre norme rilevanti"; up to _MAX_ISSUES_WIDE issues when
_WIDE_APPLICABLE or more are applicable; sections 3 and 4 discuss every article of the plan by number
and code. Logged as case_analysis_plan_applicable: placed by the plan, added to the last issue.

The plan's articles are read back against the articles it was given (_refs_of), so a law named in
full ("art. 95 Codice degli Appalti 2026", the identifier the plan is shown) is not removed as
unreadable, as dg._refs_in alone did with every non-code law (d.lgs. 36/2023, l. 241, TUEL...).
"""

import logging
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from .ai_chat import last_reply
from . import defensive_generation as dg
from . import ai_chat, anchored_writing, article_text, citation_ledger, comma_material, grounded_review, method_candidates, practical_checklist, protected_sources, render_computed, repetition_guard, role_fit, rule_quote, section_fit
from .verbose_logger import exception_data


def _log_exception(stage: str, exc: BaseException, **detail: Any) -> None:
    """Every exception caught here, logged with its full traceback (pipeline_exception), whatever VERBOSE_LOGGING says."""
    dg.vlog("pipeline_exception", exception_data(f"plan_then_write.{stage}", exc, **detail))


logger = logging.getLogger(__name__)

_PLAN_TOKENS = 6000       # Leonardo on Qwen3: cut at 4000 twice. _call_chat lowers it if the context has less room
_PLAN_MIN_ISSUES = 2      # a cut plan with these complete needs no retry
_PLAN_CAPS = ("\n\nLa risposta precedente è stata troncata: rispondi di nuovo con lo stesso JSON, più conciso: al massimo 5 "
              "questioni, ogni campo al massimo 60 parole, nessuna ripetizione dei fatti.")
_MAX_ISSUES = 5
KEEP_APPLICABLE_FLAG = "PLAN_KEEP_APPLICABLE_ENABLED"
# PLAN_KEEP_CHECKED_ENABLED (diagnose_long, 25 Oct: 34 of the 60 expected articles lost were in the
# prompt and not in the plan, which placed 2-7 of 30-70 articles): every article the applicability
# check judged applicabile or alternativa -- not the exempt ones, not the unchecked ones -- is in the
# plan, under the issue it governs, up to _MAX_ISSUES_WIDE issues (_group_checked).
KEEP_CHECKED_FLAG = "PLAN_KEEP_CHECKED_ENABLED"
# With CHECK_GRADED_ENABLED (diagnose_long, 26 Oct: 263 articles cited without being expected came in through
# "Altre norme rilevanti"): only grade 2-3 articles, at most _GRADED_KEEP by grade then fusion rank, and one
# that fits no issue of the plan is left out, never put in a catch-all issue. Run 1302 (sempronio 581/582,
# h7 110 c.p., sempronio_no_hint 41 c.p. lost): with them the exempt articles linked to a forced one -- a
# cross-reference in its text, an article resolved for a central issue (_forced) -- up to _GRADED_KEEP.
_GRADED_KEEP = 18
# GRADER_EVIDENCE_ENABLED (run 1415: 20 of 122 expected articles lost at NOT_IN_PLAN): the grades come per
# issue of the case, so up to _GRADED_KEEP_BY_ISSUE are forced, offered to the plan grouped by that issue.
_GRADED_KEEP_BY_ISSUE = 18
_MAX_ISSUES_WIDE, _WIDE_APPLICABLE = 7, 8
# WRITER_RULES_ENABLED (the lawyer's blind review of two new cases, 26 Oct: both versions 3-5/10, with the same
# faults): sections cite an article only where it supports an argument (no "discuss every plan article", never
# "non pertinente/non applicabile"); the burden of proof only where the norm itself allocates it; no conduct,
# knowledge or intent of a party the case does not state; facts labelled from the client's side; a word budget
# per section and no cut section; no deep-dive links under the parere. After writing: a sentence citing an
# article only to say it does not apply is dropped, a per-article burden template whose norm allocates nothing is
# dropped, a sentence stating a fact about a party the case does not hold is flagged (review_section).
WRITER_RULES_FLAG = "WRITER_RULES_ENABLED"
# FACT_MARK_INLINE_ENABLED (default off; acts with WRITER_RULES_ENABLED. Run 20261026-0841: ~10 of a sample of 12
# marked sentences were arguments, quotes of the case, of a norm or a ruling, or evidence to gather): the "fact not
# in the case" marker written into the answer. Off, the flagged sentences are only logged.
FACT_MARK_FLAG = "FACT_MARK_INLINE_ENABLED"
# REVIEW_FIXES_ENABLED (the lawyer's review of the boat and pool cases, 26 Oct): the inquadramento names the facts
# that decide the case instead of labelling each one (the child's death came out "neutro" in one run, "sfavorevole"
# in the other); the procedural section writes every item of each block it is given (iure hereditatis offered
# twice, written never); a sentence stating as certain what the case does not say about a deadline, a report or a
# party's proof, knowledge or concealment is rewritten as a condition (rewrite_unsupported).
REVIEW_FIXES_FLAG = "REVIEW_FIXES_ENABLED"
OTHER_NORMS_TITLE = "Altre norme rilevanti"
_ISSUE_COUNT_WORDS = {5: "cinque", 7: "sette"}

_PLAN_SYSTEM = (
    "Sei un avvocato italiano con lunga esperienza di contenzioso. Prima di scrivere un parere ne prepari il "
    "PIANO, nell'interesse della parte assistita. Ricevi il caso, i quesiti, l'analisi dei fatti, le questioni "
    "giuridiche, le NORME (testi dalla banca dati, ciascuna con il suo identificativo tra parentesi quadre e, "
    "quando c'è, la sua rubrica), la GIURISPRUDENZA (passi di pronunce, ciascuna con il suo riferimento tra "
    "parentesi quadre) e le VERIFICHE pratiche per la fase e la materia del caso.\n\n"
    "REGOLE:\n"
    "- Tutto il piano è nell'interesse della parte assistita: chi chiede il parere o la strategia, o il "
    "cliente da difendere.\n"
    "- La questione centrale è la domanda da cui dipende l'esito per la parte assistita, in una frase: va "
    "risolta per prima e mai concessa.\n"
    "- Per ogni questione scegli le norme che la disciplinano leggendone la rubrica e il testo in NORME e "
    "indicale con l'identificativo esatto (\"art. 1384 c.c.\"); una norma va solo dove disciplina proprio ciò "
    "che l'argomento le attribuisce; mai norme che non sono in NORME, mai a memoria.\n"
    "- L'argomento non può affermare il contrario di ciò che dice il testo della norma su cui si fonda.\n"
    "- Se un quesito chiede quale rito, rimedio o iniziativa scegliere, rispondi confrontando le possibilità "
    "che le VERIFICHE e le NORME indicano per la fase del caso, con i loro limiti.\n"
    "- Pronunce: solo quelle di GIURISPRUDENZA, con il riferimento esatto tra parentesi quadre, e solo se il "
    "passo riguarda davvero la questione; altrimenti lista vuota. I fatti di una pronuncia non sono fatti "
    "del caso. Una pronuncia segnata come ORIENTAMENTO SFAVOREVOLE alla parte assistita va nell'obiezione "
    "della controparte, con la replica, mai nell'argomento.\n"
    "- L'argomento di ogni questione deve escludere, ridurre o contrastare la responsabilità o la pretesa "
    "contro la parte assistita, o sostenerne la pretesa. Un fatto che favorisce la controparte è "
    "un'obiezione da superare, non un argomento.\n"
    "- Considera le tesi alternative o subordinate quando i fatti lo consentono: una qualificazione diversa e "
    "meno grave, una riduzione, un rito o un rimedio più favorevole.\n"
    "- Non aggiungere fatti che il caso non contiene.\n\n"
    "Rispondi SOLO con un oggetto JSON:\n"
    '{"questioni": [{"titolo": "...", "norme": ["art. N codice"], "pronunce": ["riferimento"], '
    '"argomento": "...", "obiezione": "...", "replica": "..."}], '
    '"parte_assistita": "...", "posizione": "...", "obiettivo": "che cosa vuole ottenere", '
    '"questione_centrale": "una frase", '
    '"risposte": [{"quesito": "...", "risposta": "risposta diretta in una o due frasi"}], '
    '"subordinate": ["..."], "punti_deboli": ["..."]}\n'
    "Le questioni per prime, da due a cinque, in ordine di forza per la parte assistita. Sii conciso: ogni campo "
    "al massimo in due frasi, senza ripetere i fatti."
)

def _fell_back(step: str, error: Any, **detail) -> None:
    """Why this parere goes back to one call: the exact step and error, for plan_fallback_report.py."""
    dg.vlog("plan_then_write_fallback", {"step": step, "error": f"{error}"[:500], **detail})


def _argued_issues(raw: Any) -> int:
    """How many complete issues a salvaged plan holds (clean_plan keeps an issue only with its argument)."""
    issues = raw.get("questioni") if isinstance(raw, dict) else None
    return sum(1 for q in issues if isinstance(q, dict) and _text(q.get("argomento"))) if isinstance(issues, list) else 0


def _plan_json(messages: List[Any]) -> Any:
    """One plan call, its finish_reason logged. A plan is never lost to its length: the issues come first, so a cut reply
    still has complete ones, kept when there are _PLAN_MIN_ISSUES; else it is asked once more with hard caps, and cut again,
    whatever issues either reply completed (dg._cut_json). ValueError only with none: clean_plan then finds no plan."""
    last_reply.finish_reason, last_reply.max_tokens = None, None
    try:
        return dg._chat_json(messages, max_tokens=_PLAN_TOKENS, temperature=0, compact=_PLAN_CAPS,
                             enough=lambda raw: _argued_issues(raw) >= _PLAN_MIN_ISSUES)
    finally:
        dg.vlog("case_analysis_plan_call", {"finish_reason": getattr(last_reply, "finish_reason", None),
                                            "asked": _PLAN_TOKENS, "max_tokens": getattr(last_reply, "max_tokens", None),
                                            "prompt": sum(len(m.content) for m in messages) // 4})


def _remade(system: str, human: str, plan: Optional[Dict[str, Any]], removed: dict, raw: Any, known: set,
            references: Dict[str, List[Dict[str, Any]]], max_issues: int) -> tuple:
    """(plan, removed, raw) after asking for the plan again: the new one, unless the call failed or was cut to fewer issues
    than the first: then the first stands, and the code corrects its order (order_for_acquired)."""
    try:
        new_raw = _plan_json([SystemMessage(content=system), HumanMessage(content=human)])
        new_plan, new_removed = clean_plan(new_raw, known, references, max_issues)
    except Exception as exc:
        _log_exception("_remade", exc)
        logger.warning("plan-then-write: plan not remade, the first kept: %s", exc)
        dg.vlog("case_analysis_plan_remake", {"kept": "first", "why": str(exc)[:200]})
        return plan, removed, raw
    held = len((plan or {}).get("questioni") or [])
    if not new_plan or (getattr(last_reply, "salvaged", False) and len(new_plan["questioni"]) < held):
        dg.vlog("case_analysis_plan_remake", {"kept": "first", "why": f"remade with {len((new_plan or {}).get('questioni') or [])} issues of {held}"})
        return plan, removed, raw
    return new_plan, new_removed, new_raw


def _plan_brief(raw: Any) -> str:
    """The plan to remake, by its issues and articles. Not its JSON: on top of the whole prompt that took the room of the
    answer, and _call_chat lowers max_tokens to what the context has left (Leonardo v4: a remake cut at 5,040 characters)."""
    if not isinstance(raw, dict):
        return ""
    lines = [f"Questione centrale: {_text(raw.get('questione_centrale'))}"] if _text(raw.get("questione_centrale")) else []
    for q in raw.get("questioni") if isinstance(raw.get("questioni"), list) else []:
        if isinstance(q, dict):
            lines.append(f"- {_text(q.get('titolo'))} ({', '.join(_texts(q.get('norme')))})")
    return "\n".join(lines)


def _plan_system(max_issues: int = _MAX_ISSUES) -> str:
    system = _PLAN_SYSTEM.replace("da due a cinque", f"da due a {_ISSUE_COUNT_WORDS.get(max_issues, max_issues)}")
    if _writer_rules():
        system = system.replace("- Non aggiungere fatti che il caso non contiene.\n",
                                _STANCE_RULES + "- Non aggiungere fatti che il caso non contiene.\n")
    if _review_fixes():
        system = system.replace("- Non aggiungere fatti che il caso non contiene.\n",
                                _ACQUIRED_RULE + _QUESTIONS_RULE + "- Non aggiungere fatti che il caso non contiene.\n")
    if anchored_writing.enabled():        # ANCHORED_WRITING_ENABLED: each issue names the passages it rests on
        system = system.replace("- Non aggiungere fatti che il caso non contiene.\n",
                                anchored_writing.PLAN_RULE + "- Non aggiungere fatti che il caso non contiene.\n")
        system = system.replace('"norme": ["art. N codice"], ', '"norme": ["art. N codice"], ' + anchored_writing.PLAN_KEY)
    return system


# WRITER_RULES_ENABLED (the lawyer's review of the pool case's answer): nothing against the client or a co-client;
# a party with several roles in the facts is liable for each.
_STANCE_RULES = (
    "- Non proporre mai azioni, domande, denunce o querele contro la parte assistita o contro chi è assistito "
    "insieme a essa (per esempio uno dei familiari che chiedono il parere): un loro eventuale concorso di colpa è "
    "solo una difesa della controparte, da trattare come obiezione con la sua replica.\n"
    "- Se una parte ha più ruoli nei fatti (per esempio ha costruito un bene e poi lo gestisce o lo mantiene), "
    "valuta la sua responsabilità per ciascun ruolo, con le norme di ciascuno.\n")


_PLAN_CHECK_SYSTEM = (
    "Sei un giurista e controlli il piano di un parere prima che venga scritto. Ricevi una questione del piano "
    "(titolo e argomento della parte assistita) e le norme che il piano vi collega, ciascuna con la sua rubrica "
    "reale e il suo testo. Per ciascuna norma rispondi a due domande.\n"
    "- \"pertinente\": la norma disciplina davvero ciò per cui l'argomento la usa? Una norma sull'arresto in "
    "flagranza citata per il giudizio abbreviato, o una sulla licenza per finita locazione citata per una "
    "servitù, non è pertinente; una norma generale citata per il suo istituto lo è; la norma che definisce il "
    "reato o l'istituto di cui la questione tratta è pertinente anche quando l'argomento ne riguarda un aspetto "
    "(il tentativo, le circostanze, la prova).\n"
    "- \"contraddice\": l'argomento afferma il contrario di ciò che il testo della norma dice? Se il testo dice "
    "che la penale è dovuta indipendentemente dalla prova del danno, un argomento secondo cui la penale "
    "richiede la prova del danno la contraddice. Un argomento che ne chiede l'applicazione, la riduzione o "
    "un'eccezione prevista dal testo non la contraddice.\n"
    "Nel dubbio: pertinente true, contraddice false. Restituisci SOLO un array JSON, un oggetto per norma: "
    '[{"articolo": l\'identificativo dato, "pertinente": true o false, "contraddice": true o false, '
    '"perche": una frase}].'
)

# WRITER_RULES_ENABLED: an article about the inverse situation never governs the argument.
_PLAN_CHECK_DIRECTION = (
    "Non è pertinente nemmeno una norma che disciplina la situazione inversa a quella dell'argomento (il danno "
    "cagionato da un soggetto citato per il danno subito da quel soggetto; l'obbligo del conduttore citato per "
    "quello del locatore).\n")

SECTION_TITLES = ("Risposte ai quesiti", "Inquadramento e questione centrale", "Norme e giurisprudenza",
                  "Applicazione al caso", "Profili processuali e termini", "Conclusioni")


def _headings(article_rows: List[Dict[str, Any]]) -> Dict[tuple, str]:
    """{article: its real heading (rubrica)} for the rows that have one."""
    headings: Dict[tuple, str] = {}
    for row in article_rows:
        s = row.get("s") or {}
        heading = article_text.heading(s)
        if row.get("_article") and heading:
            headings.setdefault(row["_article"], heading)
    return headings


def _with_headings(block: str, headings: Dict[tuple, str]) -> str:
    """An article block (_source_block's "[art. N code]" labels) with each article's heading under its label."""
    for (number, code), heading in headings.items():
        label = f"[art. {number} {code}]\n"
        block = block.replace(label, f"{label}Rubrica: {heading}\n", 1)
    return block


def _checks(analysis: Dict[str, Any], article_rows: Optional[List[Dict[str, Any]]] = None,
            facts: str = "", request: str = "") -> List[str]:
    """The practical checks for the plan and the procedural section. CHECKLIST_GATED_ENABLED: those of
    the case's verified area x posture (method_candidates.case_lists), each only if every article it
    names went through the applicability check."""
    areas, posture = method_candidates.case_lists(analysis)
    side = _client_side(analysis)
    if not dg.feature_flags.enabled(method_candidates.CHECKLIST_GATED_FLAG):
        return _by_claim_posture(practical_checklist.checks_for(areas, posture, side), analysis)
    through = _through_check(article_rows or [], facts, request)
    return _by_claim_posture(practical_checklist.gated_checks(
        areas, posture, lambda ref: any(_same_article(ref, t) for t in through), side), analysis)


def _client_side(analysis: Dict[str, Any]) -> str:
    """ROLE_BLOCKS_ENABLED: "accused" / "victim" / "" for the client (fact_signals.client_side); "" with it off."""
    from . import fact_signals
    if not dg.feature_flags.enabled(fact_signals.ROLE_BLOCKS_FLAG):
        return ""
    return fact_signals.client_side(dg._stance_party(analysis or {}))


_RECEIVED_ITEM, _CLAIMING_ITEM = "Se la parte assistita riceve una pretesa", "Se è la parte assistita a pretendere: "


def _by_claim_posture(checks: List[str], analysis: Dict[str, Any]) -> List[str]:
    """FACT_REGIMES_ENABLED: the "respond to a claim received" item only when the case says one was received; a
    claimant client's own item stated as such (fact_signals.claim_posture)."""
    from . import fact_signals
    posture = (_signals(analysis) or {}).get("posture") if fact_signals.regimes_enabled() else None
    if not posture:
        return checks
    out = []
    for item in checks:
        if item.startswith(_RECEIVED_ITEM) and not posture.get("received"):
            continue
        if item.startswith(_CLAIMING_ITEM) and posture.get("claimant") and not posture.get("received"):
            item = item[len(_CLAIMING_ITEM):][0].upper() + item[len(_CLAIMING_ITEM) + 1:]
        out.append(item)
    return out


# --- 1. the plan -------------------------------------------------------------------------------

def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _texts(value: Any) -> List[str]:
    return [t for t in (_text(v) for v in (value if isinstance(value, list) else [])) if t]


def _matching_reference(text: str, references: Dict[str, List[Dict[str, Any]]]) -> Optional[str]:
    """The reference among `references` a plan's ruling names, by the reference itself or its
    decision number and year."""
    low = text.lower()
    numbers = dg._decision_numbers(text)
    for reference in references:
        if reference.lower() in low or numbers & dg._decision_numbers(reference):
            return reference
    return None


def _refs_of(text: str, known: set) -> List[tuple]:
    """The articles a plan item names: by sigla (dg._refs_in), and the known articles of a law
    named in full ("art. 95 Codice degli Appalti 2026", as NORME labels them), which
    dg._refs_in cannot read."""
    refs = dg._refs_in(text)
    low = (text or "").lower()
    for number, code in known:
        if code not in dg._CODE_PREFIXES and (number, code) not in refs and code.lower() in low \
                and re.search(rf"(?<![\w-]){re.escape(number)}(?![\w-])", text or "", re.IGNORECASE):
            refs.append((number, code))
    return refs


def clean_plan(raw: Any, known: set, references: Dict[str, List[Dict[str, Any]]],
               max_issues: int = _MAX_ISSUES) -> Tuple[Optional[dict], dict]:
    """(plan, removed): the model's plan with only articles among `known` and rulings among
    `references`; issues without an argument dropped. None when no argued issue is left."""
    removed: Dict[str, List[str]] = {"norme": [], "pronunce": []}
    if not isinstance(raw, dict):
        return None, removed
    issues = []
    for q in raw.get("questioni") if isinstance(raw.get("questioni"), list) else []:
        if not isinstance(q, dict) or not _text(q.get("argomento")):
            continue
        norme: List[tuple] = []
        for item in _texts(q.get("norme")):
            refs = _refs_of(item, known)
            if not refs:
                removed["norme"].append(item)
            for ref in refs:
                if ref not in known:
                    removed["norme"].append(f"{ref[0]} {ref[1]}")
                elif ref not in norme:
                    norme.append(ref)
        pronunce: List[str] = []
        for item in _texts(q.get("pronunce")):
            reference = _matching_reference(item, references)
            if not reference:
                removed["pronunce"].append(item)
            elif reference not in pronunce:
                pronunce.append(reference)
        issues.append({"titolo": _text(q.get("titolo")) or f"Questione {len(issues) + 1}", "norme": norme,
                       "pronunce": pronunce, "argomento": _text(q.get("argomento")),
                       "obiezione": _text(q.get("obiezione")), "replica": _text(q.get("replica")),
                       **({"passi": anchored_writing.parse_passi(q.get("passi"), known, references)}
                          if anchored_writing.enabled() else {})})
    if not issues:
        return None, removed
    answers = [{"quesito": _text(a.get("quesito")), "risposta": _text(a.get("risposta"))}
               for a in (raw.get("risposte") if isinstance(raw.get("risposte"), list) else [])
               if isinstance(a, dict) and _text(a.get("risposta"))]
    return {"parte_assistita": _text(raw.get("parte_assistita")), "posizione": _text(raw.get("posizione")),
            "obiettivo": _text(raw.get("obiettivo")), "questione_centrale": _text(raw.get("questione_centrale")),
            "risposte": answers, "questioni": issues[:max_issues],
            "subordinate": _texts(raw.get("subordinate")), "punti_deboli": _texts(raw.get("punti_deboli"))}, removed


def _voted(article_rows: List[Dict[str, Any]]) -> List[tuple]:
    """The articles the rulings on similar facts voted for (VOTE_POOL_ENABLED's _vote rows), best voted first."""
    rows = sorted((r for r in article_rows if r.get("_vote") and r.get("_article")),
                  key=lambda r: r.get("_vote_rank") or 0)
    return list(dict.fromkeys(r["_article"] for r in rows))


def _applicable(article_rows: List[Dict[str, Any]], facts: str, request: str, analysis: Dict[str, Any]) -> List[tuple]:
    """PLAN_KEEP_APPLICABLE_ENABLED: the articles the applicability check let through -- judged
    applicabile or alternativa, or exempt from it (named in the case, found by their heading,
    cross-referenced) -- whose code belongs to the case's areas or is named in its text
    (vote_pool's area rule), in the pipeline's order."""
    from . import vote_pool
    from .nodes.cypher import CROSSREF_SOURCE
    case = dg._case_text(facts, request)
    named = set(dg._refs_in(case))
    allowed = vote_pool.allowed_codes(vote_pool.filter_areas(analysis), vote_pool.named_codes(case))
    out = []
    for row in article_rows:
        ref = row.get("_article")
        if not ref or ref in out:
            continue
        through = _passed_check(row, named, CROSSREF_SOURCE)
        in_area = allowed is None or ref[1] not in vote_pool.MAIN_CODES or ref[1] in allowed
        if through and in_area:
            out.append(ref)
    return out


def _passed_check(row: Dict[str, Any], named: set, crossref_source: str) -> bool:
    """The row went through the applicability check: judged applicabile or alternativa, or exempt
    from it (named in the case, found by its rubrica, cross-referenced)."""
    return row.get("_applicability") in ("applicabile", "alternativa") or row.get("_article") in named \
        or bool(row.get("_heading_match")) or row.get("_source") == crossref_source


def _through_check(article_rows: List[Dict[str, Any]], facts: str, request: str) -> List[tuple]:
    """Every article that went through the applicability check (_passed_check), in the pipeline's order."""
    from .nodes.cypher import CROSSREF_SOURCE
    named = set(dg._refs_in(dg._case_text(facts, request)))
    return list(dict.fromkeys(r["_article"] for r in article_rows
                              if r.get("_article") and _passed_check(r, named, CROSSREF_SOURCE)))


def _checked(article_rows: List[Dict[str, Any]]) -> List[tuple]:
    """PLAN_KEEP_CHECKED_ENABLED: the articles the check judged applicabile or alternativa, in the
    pipeline's order; the exempt and the unchecked ones are not among them. CHECK_GRADED_ENABLED: _forced."""
    if not _graded():
        return list(dict.fromkeys(r["_article"] for r in article_rows
                                  if r.get("_article") and r.get("_applicability") in ("applicabile", "alternativa")))
    return list(_forced(article_rows))


def _forced(article_rows: List[Dict[str, Any]]) -> Dict[tuple, str]:
    """{article: why} the plan must hold (PLAN_KEEP_CHECKED_ENABLED with CHECK_GRADED_ENABLED), at most
    _GRADED_KEEP, in this order: the articles resolved for a central issue (_central) that the check did
    not grade 0, as grade 3; those graded 2 or 3; each by grade, then evidence tier, then fusion rank, then
    article id (evidence_tiers.forced_key: the same list every run) -- and right after each one, the
    cross-referenced articles its own text cites (CROSSREF rows). A demoted article (evidence tiers, Rule 2) is
    never forced. Past the cap a T1/T2 article stays in the material, unforced (logged as evidence_cutoff)."""
    from . import evidence_tiers
    from .nodes.cypher import CROSSREF_SOURCE

    from . import fact_issues
    own_issue = fact_issues.resolver_v2()      # a fact institution's articles get their own issue (place_fact_...)
    best: Dict[tuple, tuple] = {}
    why: Dict[tuple, str] = {}
    for index, row in enumerate(article_rows):
        ref, grade = row.get("_article"), row.get("_grade")
        if not ref or row.get("_demoted") or (own_issue and row.get("_fact_institution")):
            continue
        if row.get("_tier_why") == "fact rule":
            grade, reason = 3, "fact rule"
        elif row.get("_central"):
            grade, reason = 3, "central issue"
        elif isinstance(grade, int) and grade >= 2:
            reason = f"grade {grade}"
        else:
            continue
        key = evidence_tiers.forced_key(row, grade)
        if ref not in best or key < best[ref]:
            best[ref], why[ref] = key, reason
    crossrefs = list(dict.fromkeys(r["_article"] for r in article_rows
                                   if r.get("_article") and r.get("_source") == CROSSREF_SOURCE))
    out: Dict[tuple, str] = {}
    for ref in sorted(best, key=lambda r: best[r]):
        out.setdefault(ref, why[ref])
        cited = _cited_numbers(article_rows, ref)
        for child in crossrefs:
            if child[1] == ref[1] and child[0] in cited:
                out.setdefault(child, f"cross-reference of {ref[0]} {ref[1]}")
    # a fired rule's articles, its required ones too, are over and above the cap
    ruled = [ref for ref, reason in out.items() if reason == "fact rule"]
    others = [ref for ref in out if ref not in ruled]
    forced = {ref: out[ref] for ref in ruled + others[:max(0, _keep_cap() - len(ruled))]}
    rows_of = {r["_article"]: r for r in article_rows if r.get("_article")}
    if any(evidence_tiers.tier(r) for r in rows_of.values()):
        past = [rows_of[ref] for ref in others[max(0, _keep_cap() - len(ruled)):]]
        dg.vlog("evidence_cutoff", evidence_tiers.cutoff_log(
            "plan_forced", _keep_cap(), past, [r for r in past if evidence_tiers.evidenced(r)]))
    return forced


def _by_issue() -> bool:
    return dg.feature_flags.enabled(dg.GRADER_EVIDENCE_FLAG)


def _keep_cap() -> int:
    return _GRADED_KEEP_BY_ISSUE if _by_issue() else _GRADED_KEEP


def _issue_groups(reasons: Dict[tuple, str], issue_of: Dict[tuple, str]) -> Dict[str, List[tuple]]:
    """{issue: forced articles} in the order of the forced list: each by the issue the graded check named for
    it, a cross-reference with the article citing it, the rest under "altre"."""
    groups: Dict[str, List[tuple]] = {}
    home: Dict[tuple, str] = {}
    for ref, why in reasons.items():
        parent = re.match(r"cross-reference of (\S+) (.+)$", why)
        issue = home.get((parent.group(1), parent.group(2))) if parent else None
        issue = issue or (issue_of or {}).get(ref) or "altre"
        home[ref] = issue
        groups.setdefault(issue, []).append(ref)
    return groups


def _cited_numbers(article_rows: List[Dict[str, Any]], ref: tuple) -> set:
    """The article numbers an article's own text cites within its code, as broaden_article_context reads
    them (another act's articles left out)."""
    from .article_suffix import canonical_suffix
    from .nodes.retrieval import _CROSS_REF_NUM, _CROSS_REF_RUN, _OTHER_ACT_AFTER

    numbers = set()
    for row in article_rows:
        if row.get("_article") != ref:
            continue
        section = row.get("s") or {}
        text = section.get("plain_text") or section.get("abstract") or ""
        for run in _CROSS_REF_RUN.finditer(text):
            if _OTHER_ACT_AFTER.match(text, run.end(), run.end() + 160):
                continue
            for m in _CROSS_REF_NUM.finditer(run.group(0)):
                numbers.add(f"{m.group(1)}-{canonical_suffix(m.group(2))}" if m.group(2) else m.group(1))
    return numbers


def _forced_issues(plan: Dict[str, Any], reasons: Dict[tuple, str], issue_of: Dict[tuple, str]) -> List[Dict[str, Any]]:
    """The forced articles the plan holds, each with the plan issues that hold it and the issue the
    graded check named for it -- for the sections, which cite each only there (_scope_block)."""
    out = []
    for ref in reasons:
        titles = [q["titolo"] for q in plan["questioni"] if ref in q["norme"]]
        if titles:
            out.append({"ref": ref, "questioni": titles, "tema": (issue_of or {}).get(ref, "")})
    return out


def _discipline() -> bool:
    """PLAN_KEEP_CHECKED_ENABLED with CHECK_GRADED_ENABLED: the sections cite each forced article only in
    its issue (_scope_block) and no article their material lacks (_cite_only_material)."""
    return dg.feature_flags.enabled(KEEP_CHECKED_FLAG) and _graded()


def _writer_rules() -> bool:
    return dg.feature_flags.enabled(WRITER_RULES_FLAG)


def _review_fixes() -> bool:
    return dg.feature_flags.enabled(REVIEW_FIXES_FLAG)


def _graded() -> bool:
    return dg.feature_flags.enabled(dg.CHECK_GRADED_FLAG)


def _keep_all() -> bool:
    """Sections 3 and 4 discuss every article of the plan by name."""
    return dg.feature_flags.enabled(KEEP_APPLICABLE_FLAG) or dg.feature_flags.enabled(KEEP_CHECKED_FLAG)


def _same_article(a: tuple, b: tuple) -> bool:
    """One article, however the list and the pool name its law: the codes by sigla, any other law by a
    name one contains of the other ("Codice del Consumo" and "Codice del Consumo 2026")."""
    if a[0] != b[0]:
        return False
    if a[1] in dg._CODE_PREFIXES or b[1] in dg._CODE_PREFIXES:
        return a[1] == b[1]
    x, y = a[1].lower(), b[1].lower()
    return x in y or y in x


FACT_ISSUE_ARGUMENT = "{nome} ({norme}), per il fatto: {fatto}."


def place_fact_institutions(plan: Dict[str, Any], analysis: Dict[str, Any],
                            article_rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(plan, log), FACT_ISSUES_RESOLVER_V2_ENABLED: the articles each fact's institution reached
    (analysis["fact_institutions"]) in the plan under that institution's issue -- an existing issue holding one of
    them, else a new issue of its own, after the others -- outside the forced list's slots (_forced). Only an article
    with a row in the material is placed. Log: reached, placed, the issues added and merged."""
    return _place_institutions(plan, analysis.get("fact_institutions") or [], article_rows)


def place_rule_institutions(plan: Dict[str, Any], analysis: Dict[str, Any], article_rows: List[Dict[str, Any]]
                            ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(plan, log): a fired fact rule's articles are T1 and argued: each in an issue, the one holding one of them, else a
    new one, before order_for_acquired sets recognition before any constitutive remedy. Being in the material is not
    being in the plan (Leonardo, 1 Nov: usucapione fired and appeared only among the terms). The articles a rule
    requires (fact_signal_rules.yaml, requires) go with the recognition of the right, when the material holds them."""
    plan, log = _place_institutions(plan, analysis.get("rule_institutions") or [], article_rows)
    have = {r["_article"] for r in article_rows if r.get("_article")}
    held = {ref for q in plan["questioni"] for ref in q["norme"]}
    home = next((q for q in plan["questioni"] if _issue_kind(q) == "recognition"), None)
    for ref in analysis.get("rule_required") or []:
        ref = tuple(ref)
        if ref in have and ref not in held and home is not None:
            home["norme"].append(ref)
            held.add(ref)
            log["placed"].append(f"{ref[0]} {ref[1]}")
    return plan, log


def _place_institutions(plan: Dict[str, Any], institutions: List[Dict[str, Any]], article_rows: List[Dict[str, Any]]
                        ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    have = {r["_article"] for r in article_rows if r.get("_article")}
    issues = [{**q, "norme": list(q["norme"])} for q in plan["questioni"]]
    reached: List[tuple] = []
    placed: List[tuple] = []
    added: List[str] = []
    merged: Dict[str, str] = {}
    for inst in institutions:
        refs = [tuple(r) for r in inst.get("refs") or []]
        reached += [r for r in refs if r not in reached]
        refs = [r for r in refs if r in have]
        if not refs:
            continue
        home = next((q for q in issues if set(q["norme"]) & set(refs)), None)
        name = " ".join(str(inst.get("nome") or "").split())
        if home is None:
            home = {"titolo": name[:1].upper() + name[1:], "norme": [], "pronunce": [],
                    "argomento": inst.get("argomento") or FACT_ISSUE_ARGUMENT.format(
                        fatto=inst.get("fatto") or name, nome=name,
                        norme=", ".join(f"art. {n} {_code_key(c)}" for n, c in refs)),
                    "obiezione": "", "replica": "", **({"passi": []} if anchored_writing.enabled() else {})}
            issues.append(home)
            added.append(home["titolo"])
        else:
            merged[name] = home["titolo"]
        for ref in refs:
            if ref not in home["norme"]:
                home["norme"].append(ref)
            if ref not in placed:
                placed.append(ref)
    log = {"reached": [f"{n} {c}" for n, c in reached], "placed": [f"{n} {c}" for n, c in placed],
           "issues_added": added, "merged_into": merged}
    return {**plan, "questioni": issues}, log


def _with_other_norms(plan: Dict[str, Any], applicable: List[tuple]) -> Tuple[Dict[str, Any], List[tuple], List[tuple]]:
    """(plan, placed, added): the applicable articles no issue holds, as a last issue."""
    held = {ref for q in plan["questioni"] for ref in q["norme"]}
    placed = [ref for ref in applicable if ref in held]
    added = [ref for ref in applicable if ref not in held]
    if added:
        plan = {**plan, "questioni": plan["questioni"] + [{
            "titolo": OTHER_NORMS_TITLE, "norme": added, "pronunce": [],
            "argomento": "Le altre norme che si applicano ai fatti del caso, da esaminare una per una: che cosa "
                         "prevedono e come incidono sulla posizione della parte assistita.",
            "obiezione": "", "replica": ""}]}
    return plan, placed, added


# PLAN_RULE_INCLUDE (v45, the NOT_IN_PLAN stage of diagnose_long): after the plan's own steps, an article of the
# prompt material that no issue holds and no fired rule excludes goes into the plan by rule, no call: graded 3 always,
# graded 2 only when one or more of the retrieval signals frame, distill and institution found it (never another search
# alone, never a regime candidate alone). v46: a row the check exempts by its rubrica (_heading_match) or as a cross-
# reference carries no grade and counts as a grade 2 (unless GRADE_ALL_PROMPT graded it); graded 1, when two or more
# distinct signals among frame, distill, institution and ruling_applied found it (RULE_INCLUDE_GRADE1_SIGNALS: similar
# rulings apply it -- voted, a confirmed map hypothesis, or evidence_tiers.MIN_RULINGS rulings); graded 0, never. It goes under the issue whose title and argument share the most words with the
# issue the grader named for it (else its rubrica), else under a last "Ulteriori norme applicabili" issue, and is
# tagged plan_source "rule" (plan["plan_source"]): the writer may use it, _omitted never forces a paragraph for it. One
# check_plan removed (off-topic, or of an issue contradicting it) is never added back: skipped, check_plan_removed.
RULE_INCLUDE_FLAG = "PLAN_RULE_INCLUDE"
RULE_INCLUDE_TITLE = "Ulteriori norme applicabili"
RULE_INCLUDE_SIGNALS = ("frame", "distill", "institution")
RULE_INCLUDE_GRADE1_SIGNALS = RULE_INCLUDE_SIGNALS + ("ruling_applied",)


def _include_signals(grade1: bool = False) -> Tuple[str, ...]:
    """The signals PLAN_RULE_INCLUDE counts; with EDGE_EXPAND, its "edge" one too (v48)."""
    from . import edge_expand
    base = RULE_INCLUDE_GRADE1_SIGNALS if grade1 else RULE_INCLUDE_SIGNALS
    return base + ((edge_expand.SIGNAL, edge_expand.SIGNAL_ANCHORS) if edge_expand.enabled() else ())
RULE_SOURCE = "rule"
RULE_INCLUDE_GRADE1_MAX = 4
RULE_INCLUDE_GRADE1_TOKENS = 4000


def _added_tokens(rows: List[Dict[str, Any]]) -> int:
    """What an article inserted by rule adds to a section prompt, estimated: its text as TESTI DELLE NORME gives it plus
    its passage, each at most its cap."""
    text = max((" ".join(str((r.get("s") or {}).get("plain_text") or "").split()) for r in rows), key=len, default="")
    return dg._estimate_tokens(text[:dg._ARTICLE_CHARS]) + dg._estimate_tokens(text[:anchored_writing.PASSAGE_CAP])


def _retrieval_signals(rows: List[Dict[str, Any]]) -> List[str]:
    """Every retrieval signal an article's rows carry: the fusion's searches (_signals), the institution resolution
    (_institution), a regime candidate (_regime_candidate), similar rulings applying it (ruling_applied: _vote, _map,
    or evidence_tiers.MIN_RULINGS rulings in _vote_stats)."""
    from . import evidence_tiers
    out = [name for r in rows for name in r.get("_signals") or []]
    out += ["institution" for r in rows if r.get("_institution")]
    out += ["regime_candidate" for r in rows if r.get("_regime_candidate")]
    out += ["ruling_applied" for r in rows if r.get("_vote") or r.get("_map")
            or int((r.get("_vote_stats") or {}).get("rulings") or 0) >= evidence_tiers.MIN_RULINGS]
    return list(dict.fromkeys(out))


def rule_inserted(plan: Dict[str, Any]) -> set:
    """The plan's articles PLAN_RULE_INCLUDE put there (plan_source "rule")."""
    return {ref for q in plan.get("questioni") or [] for ref in q["norme"]
            if (plan.get("plan_source") or {}).get(f"{ref[0]} {ref[1]}") == RULE_SOURCE}


def rule_include() -> bool:
    return dg.feature_flags.enabled(RULE_INCLUDE_FLAG)


def include_by_rule(plan: Dict[str, Any], article_rows: List[Dict[str, Any]], articles: str,
                    headings: Dict[tuple, str], issue_of: Dict[tuple, str],
                    excluded: Sequence[tuple] = (), check_removed: Sequence[tuple] = ()
                    ) -> Tuple[Dict[str, Any], List[Dict[str, str]], List[Dict[str, str]]]:
    """(plan, inserted, skipped): the plan with every row of the prompt material (`articles`, the draft prompt's block)
    absent from its issues and excluded by no fired rule that PLAN_RULE_INCLUDE's rule admits (grade 3; grade 2 with one
    or more of RULE_INCLUDE_SIGNALS; grade 1 with two or more of RULE_INCLUDE_GRADE1_SIGNALS; an ungraded row exempt by
    heading or as a cross-reference counts as grade 2, its entry carrying "exempt"), in one issue each, tagged plan_source "rule"; inserted, one entry per article with
    where it went, its grade, its signals and why. In the pipeline's order, the best grade of the article's rows
    counting. One among check_removed (the articles check_plan took out) stays out, in skipped ("check_plan_removed"),
    and so does a grade 2 without the signals."""
    from .nodes.cypher import CROSSREF_SOURCE

    held = {ref for q in plan["questioni"] for ref in q["norme"]}
    grades: Dict[tuple, int] = {}
    rows_of: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in article_rows:
        ref, grade = row.get("_article"), row.get("_grade")
        if ref:
            rows_of.setdefault(ref, []).append(row)
        if ref and isinstance(grade, int) and grade >= 1 and f"[art. {ref[0]} {ref[1]}]" in articles:
            grades[ref] = max(grade, grades.get(ref, 0))
    exempt: Dict[tuple, str] = {}       # ungraded rows the check exempts: a grade 2
    for ref, rows in rows_of.items():
        if any(isinstance(r.get("_grade"), int) for r in rows) or f"[art. {ref[0]} {ref[1]}]" not in articles:
            continue
        why = ("heading" if any(r.get("_heading_match") for r in rows) else
               "crossref" if any(r.get("_source") == CROSSREF_SOURCE for r in rows) else "")
        if why:
            exempt[ref], grades[ref] = why, 2
    signals = {ref: _retrieval_signals(rows_of.get(ref, [])) for ref in grades}
    wanted = [ref for ref in grades if ref not in held and not any(_same_ref(ref, tuple(x)) for x in excluded)]
    skipped: List[Dict[str, Any]] = []
    admitted = []
    for ref in wanted:
        entry = {"article": f"{ref[0]} {ref[1]}", "grade": str(grades[ref]), "signals": signals[ref],
                 **({"exempt": exempt[ref]} if ref in exempt else {})}
        strong = [name for name in signals[ref] if name in _include_signals()]
        if any(_same_ref(ref, tuple(x)) for x in check_removed):
            skipped.append({**entry, "reason": "check_plan_removed"})
        elif grades[ref] == 2 and not strong:
            skipped.append({**entry, "reason": "grade 2 without any of " + "/".join(_include_signals())})
        elif grades[ref] == 1 and len({n for n in signals[ref] if n in _include_signals(True)}) < 2:
            skipped.append({**entry, "reason": "grade 1 without two of " + "/".join(_include_signals(True))})
        else:
            admitted.append(ref)
    # v48: grade 1 at most RULE_INCLUDE_GRADE1_MAX per case, most signals first, within RULE_INCLUDE_GRADE1_TOKENS of
    # added prompt (v47: 13 grade-1 rows in h1, l3, n3, the section prompts over the context)
    ones = sorted((ref for ref in admitted if grades[ref] == 1 and ref not in exempt),
                  key=lambda ref: -len({n for n in signals[ref] if n in _include_signals(True)}))
    spent, kept_ones = 0, set()
    for ref in ones:
        cost = _added_tokens(rows_of.get(ref, []))
        if len(kept_ones) < RULE_INCLUDE_GRADE1_MAX and spent + cost <= RULE_INCLUDE_GRADE1_TOKENS:
            kept_ones.add(ref)
            spent += cost
        else:
            skipped.append({"article": f"{ref[0]} {ref[1]}", "grade": "1", "signals": signals[ref], "tokens": cost,
                            "reason": f"grade-1 cap: {RULE_INCLUDE_GRADE1_MAX} per case within "
                                      f"{RULE_INCLUDE_GRADE1_TOKENS} added tokens"})
    wanted = [ref for ref in admitted if grades[ref] != 1 or ref in exempt or ref in kept_ones]
    if not wanted:
        return plan, [], skipped
    issues = [{**q, "norme": list(q["norme"])} for q in plan["questioni"]]
    inserted: List[Dict[str, str]] = []
    rest: List[tuple] = []
    for ref in wanted:
        words = _title_stems(issue_of.get(ref) or headings.get(ref, ""))
        best = max(issues, key=lambda q: len(words & _title_stems(f"{q['titolo']} {q['argomento']}")), default=None)
        shared = words & _title_stems(f"{best['titolo']} {best['argomento']}") if best else set()
        entry = {"article": f"{ref[0]} {ref[1]}", "grade": str(grades[ref]), "signals": signals[ref],
                 **({"exempt": exempt[ref]} if ref in exempt else {}),
                 "matched_on": "grader issue" if issue_of.get(ref) else "rubrica"}
        if shared:
            best["norme"].append(ref)
            inserted.append({**entry, "issue": best["titolo"],
                             "reason": f"grade {grades[ref]}, shares {', '.join(sorted(shared))} with the issue"})
        else:
            rest.append(ref)
            inserted.append({**entry, "issue": RULE_INCLUDE_TITLE,
                             "reason": f"grade {grades[ref]}, no issue shares a word with "
                                       f"«{issue_of.get(ref) or headings.get(ref, '') or '-'}»"})
    if rest:
        issues.append({"titolo": RULE_INCLUDE_TITLE, "norme": rest, "pronunce": [],
                       "argomento": "Le altre norme che il materiale giudica pertinenti ai fatti del caso, da esaminare "
                                    "una per una: che cosa prevedono e come incidono sulla posizione della parte assistita.",
                       "obiezione": "", "replica": ""})
    source = {**(plan.get("plan_source") or {}), **{e["article"]: RULE_SOURCE for e in inserted}}
    return {**plan, "questioni": issues, "plan_source": source}, inserted, skipped


PLAN_CAP_FLAG = "PLAN_CAP"
PLAN_CAP_MAX = 12          # substantive articles in the plan the writer reads (env PLAN_CAP_MAX overrides)
AVAILABLE_KEY = "_plan_available"


def plan_cap_max() -> int:
    import os
    try:
        return max(1, int(os.getenv("PLAN_CAP_MAX") or PLAN_CAP_MAX))
    except ValueError:
        return PLAN_CAP_MAX


def _cap_tier(ref: tuple, grade: Optional[int], signals: List[str], named: bool, by_rule: bool) -> int:
    """PLAN_CAP's order: 0 named by the user (always in), 1 grade 3, 2 grade 2 with two or more signals, 3 a rule
    insertion, 4 the plan's other articles."""
    if named:
        return 0
    if grade == 3:
        return 1
    if grade == 2 and len(signals) >= 2:
        return 2
    return 3 if by_rule else 4


def cap_plan(plan: Dict[str, Any], article_rows: List[Dict[str, Any]], named: Sequence[tuple],
             cap: int = PLAN_CAP_MAX) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """PLAN_CAP (v49): (plan, log) with at most `cap` substantive articles (a procedural code's stay, uncounted): the
    articles the user named always, then grade 3, grade 2 with two or more retrieval signals, the rule insertions, the
    plan's others -- each tier by signal count, then grade, then the plan's order. The rest leave the plan's issues
    and stay in the material tagged available (AVAILABLE_KEY, plan["disponibili"]) for the citation closure only: no
    writer's prompt names them (v50; v49 listed them in sections 3 and 4). An ungraded row exempt by heading or as a
    cross-reference counts as grade 2."""
    from .nodes.cypher import CROSSREF_SOURCE
    refs = _plan_refs(plan)
    substantive = [r for r in refs if not role_fit.PROCEDURAL_CODES.search(str(r[1]))]
    rows_of: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in article_rows:
        if row.get("_article"):
            rows_of.setdefault(tuple(row["_article"]), []).append(row)
    by_rule = rule_inserted(plan)

    def grade(ref) -> Optional[int]:
        rows = rows_of.get(tuple(ref)) or []
        grades = [r["_grade"] for r in rows if isinstance(r.get("_grade"), int)]
        if grades:
            return max(grades)
        return 2 if any(r.get("_heading_match") or r.get("_source") == CROSSREF_SOURCE for r in rows) else None
    info = {}
    for i, ref in enumerate(substantive):
        signals = _retrieval_signals(rows_of.get(tuple(ref)) or [])
        is_named = any(_same_ref(ref, n) for n in named)
        info[ref] = {"tier": _cap_tier(ref, grade(ref), signals, is_named, ref in by_rule), "grade": grade(ref),
                     "signals": signals, "order": i}
    ranked = sorted(substantive, key=lambda r: (info[r]["tier"], -len(info[r]["signals"]), -(info[r]["grade"] or 0),
                                                info[r]["order"]))
    keep = [r for r in ranked if info[r]["tier"] == 0]
    keep += [r for r in ranked if info[r]["tier"] > 0][:max(0, cap - len(keep))]
    available = [r for r in ranked if r not in keep]

    def entry(r):
        return {"article": f"{r[0]} {r[1]}", "tier": info[r]["tier"], "grade": info[r]["grade"],
                "signals": info[r]["signals"]}
    log = {"before": len(substantive), "after": len(keep), "cap": cap, "procedural": len(refs) - len(substantive),
           "kept": [entry(r) for r in keep], "available": [entry(r) for r in available]}
    if not available:
        return plan, log
    out = {**plan, "questioni": [{**q, "norme": [r for r in q["norme"] if r not in available]} for q in plan["questioni"]],
           "disponibili": [f"{n} {c}" for n, c in available]}
    out["plan_source"] = {k: v for k, v in (plan.get("plan_source") or {}).items()
                          if k not in out["disponibili"]}
    for ref in available:
        for row in rows_of.get(tuple(ref)) or []:
            row[AVAILABLE_KEY] = True
    return out, log


def _group_checked(plan: Dict[str, Any], checked: List[tuple], headings: Dict[tuple, str],
                   max_issues: int, issue_of: Optional[Dict[tuple, str]] = None,
                   catch_all: bool = True, reasons: Optional[Dict[tuple, str]] = None
                   ) -> Tuple[Dict[str, Any], List[tuple], Dict[tuple, str], List[tuple]]:
    """(plan, placed, grouped, added): every checked article in the plan (PLAN_KEEP_CHECKED_ENABLED).
    One the plan left out goes under the issue whose title and argument share the most words with its
    rubrica (and with the issue the graded check named for it, issue_of); with none sharing a word, into
    a last "Altre norme rilevanti" issue, or, at max_issues issues, into the last one -- or, without
    catch_all (CHECK_GRADED_ENABLED), nowhere: it is not forced into the plan. With reasons (_forced): a
    cross-reference goes under the issue of the article that cites it, and an article of a central issue
    sharing no word with any issue under the first one."""
    held = {ref for q in plan["questioni"] for ref in q["norme"]}
    placed = [ref for ref in checked if ref in held]
    issues = [{**q, "norme": list(q["norme"])} for q in plan["questioni"]]
    grouped: Dict[tuple, str] = {}
    rest: List[tuple] = []

    def stems(text):
        return {w[:5] for w in re.findall(r"\w{4,}", (text or "").lower()) if w not in _HEADING_STOP}
    for ref in (r for r in checked if r not in held):
        why = (reasons or {}).get(ref, "")
        parent = re.match(r"cross-reference of (\S+) (.+)$", why)
        home = next((q for q in issues if parent and (parent.group(1), parent.group(2)) in q["norme"]), None)
        if home is None:
            words = stems(f"{headings.get(ref, '')} {(issue_of or {}).get(ref, '')}")
            best = max(issues, key=lambda q: len(words & stems(f"{q['titolo']} {q['argomento']}")))
            home = best if words & stems(f"{best['titolo']} {best['argomento']}") else \
                issues[0] if why == "central issue" else None
        if home is not None:
            home["norme"].append(ref)
            grouped[ref] = home["titolo"]
        else:
            rest.append(ref)
    plan = {**plan, "questioni": issues}
    if not catch_all:
        return plan, placed, grouped, []
    if rest and len(issues) >= max_issues:
        issues[-1]["norme"] += rest
        grouped.update({ref: issues[-1]["titolo"] for ref in rest})
        rest = []
    if rest:
        plan, _, rest = _with_other_norms(plan, rest)
    return plan, placed, grouped, rest


# v60f: the interactive answer's plan -- its issues fixed (the chosen topics, each with its articles' records:
# interactive_flow, issue_records.plan_issues) and the user's own question (its angle)
PLAN_ISSUES_KEY = "plan_issues"
PLAN_ANGLE_KEY = "plan_angle"
FIXED_ISSUES_TITLE = "QUESTIONI SCELTE (già decise, con le norme e i requisiti di ciascuna)"
FIXED_ISSUES_RULE = ("Le questioni sono ESATTAMENTE quelle di QUESTIONI SCELTE, con i loro titoli e le loro norme, "
                     "nell'ordine dato: non aggiungerne, non toglierne, non spostare norme dall'una all'altra. Per "
                     "ciascuna scrivi argomento, obiezione e replica sui requisiti indicati e sui fatti che li "
                     "sostengono o li contraddicono; un requisito senza fatti che lo sostengano resta da provare. Sii "
                     "conciso: ogni campo al massimo in due frasi, senza ripetere i fatti.")
FIXED_ACQUIRED_RULE = ("- Un diritto è già acquisito solo se DIRITTI GIÀ ACQUISITI lo indica (una regola dei fatti "
                       "che il caso afferma); altrimenti è da provare, nella questione scelta che lo riguarda.\n")
ANGLE_TITLE = "DOMANDA DELL'UTENTE"
ANGLE_RULE = "Il piano risponde a questa domanda per prima, in «risposte»."


def _fixed_block(analysis: Dict[str, Any]) -> str:
    issues = (analysis or {}).get(PLAN_ISSUES_KEY) or []
    if not issues:
        return ""
    parts = []
    for i, issue in enumerate(issues, 1):
        norms = ", ".join(f"art. {n} {_code_key(c)}" for n, c in issue.get("norme") or [])
        parts.append(f"{i}. {issue.get('titolo')} — norme: {norms}\n{issue.get('scheda') or ''}".rstrip())
    return f"{FIXED_ISSUES_TITLE}:\n" + "\n".join(parts) + "\n\n"


def _plan_human(facts: str, request: str, analysis: Dict[str, Any], articles: str, rulings: str,
                principles: str, checks: List[str], voted: Optional[List[tuple]] = None,
                applicable: Optional[List[tuple]] = None, must_place: Any = False,
                groups: Optional[Dict[str, List[tuple]]] = None) -> str:
    base = dg._parere_human(facts, analysis, articles, rulings, request, principles)
    base = base[:base.rfind("Redigi ora il parere.")].rstrip() if "Redigi ora il parere." in base else base
    verifiche = "\n".join(f"- {c}" for c in checks)
    # VOTE_POOL_ENABLED: the articles the rulings on similar facts cite, for the plan to weigh
    # issue by issue (prod, 24 Oct: 1158, 1218, 1453, 56, 378 ... found only this way).
    votate = ", ".join(f"[art. {n} {c}]" for n, c in voted or [])
    applicabili = ", ".join(f"[art. {n} {c}]" for n, c in applicable or [])
    by_issue = ""
    if groups and applicable and must_place == "graded":      # GRADER_EVIDENCE_ENABLED: by the grader's issue
        by_issue = ("NORME PIÙ PERTINENTI AL CASO, per la questione che disciplinano:\n"
                    + "\n".join(f"- {issue}: " + ", ".join(f"[art. {n} {c}]" for n, c in refs)
                                for issue, refs in groups.items())
                    + "\nMetti ciascuna tra le norme della questione del piano che corrisponde a quella indicata; una "
                    "che non disciplina nessuna questione del piano resta fuori.\n\n")
    if (analysis or {}).get(PLAN_ISSUES_KEY):   # v60f: issues fixed -- no list that asks to add or regroup an issue
        votate, applicabili, by_issue = "", "", ""
    terms = _terms(analysis)
    qualified, tools = _qualifications(analysis), _tools(analysis)
    damages, roles = _damages(analysis), _roles(analysis)
    fixes = _review_fixes()
    questions = "\n".join(f"{i}. {q}" for i, q in enumerate(explicit_questions(request, facts), 1)) if fixes else ""
    fixed = bool((analysis or {}).get(PLAN_ISSUES_KEY))
    acquired = "\n".join(f"- {a}" for a in acquired_rights(analysis, stated_only=fixed)) if fixes else ""
    defendants_rule = _DEFENDANTS_RULE
    if fixed:                   # v60g (P6): a party the case does not identify is «da individuare»
        from .writer_rules import DEFENDANTS_RULE as defendants_rule
    return (f"PARTE ASSISTITA (dall'analisi del caso): {dg._stance_party(analysis)}\n\n{base}\n\n"
            + (f"{_QUESTIONS_TITLE}:\n{questions}\n\n" if questions else "")
            + (f"{_ACQUIRED_TITLE}:\n{acquired}\n\n" if acquired else "")
            + (f"{_CONCLUSIONS_TITLE}:\n{_conclusions(analysis)}\n{_CONCLUSIONS_RULE}\n\n" if _conclusions(analysis) else "")
            + (f"{_DEFENDANTS_TITLE}:\n{_defendants(analysis)}\n{defendants_rule}\n\n" if _defendants(analysis) else "")
            + (f"{_QUALIFIED_TITLE}:\n{qualified}\n{_QUALIFIED_RULE}\n\n" if qualified else "")
            + (f"{_ROLES_TITLE}:\n{roles}\n\n" if roles else "")
            + (f"{_DAMAGES_TITLE}:\n{damages}\n{_DAMAGES_RULE}\n\n" if damages else "")
            + (f"VERIFICHE PER LA FASE E LA MATERIA:\n{verifiche}\n\n" if verifiche else "")
            + (f"{_TOOLS_TITLE}:\n{tools}\n{_TOOLS_RULE}\n\n" if tools else "")
            + (f"{_CRIMINAL_TITLE}:\n{_tools(analysis, criminal=True)}\n\n" if _tools(analysis, criminal=True) else "")
            + (f"{_TERMS_TITLE}:\n{terms}\n{_terms_rule()}\n\n" if terms else "")
            + (by_issue if by_issue else
               f"NORME PIÙ PERTINENTI AL CASO: {applicabili}. Metti ciascuna tra le norme della questione che "
               "disciplina; una che non disciplina nessuna questione del piano resta fuori.\n\n"
               if applicabili and must_place == "graded" else
               f"NORME GIUDICATE APPLICABILI AI FATTI: {applicabili}. Ognuna deve comparire nel piano, tra le "
               "norme della questione che disciplina: raggruppale per questione, aggiungendo una questione se "
               "nessuna di quelle che hai la disciplina; non lasciarne fuori nessuna.\n\n"
               if applicabili and must_place else
               f"NORME APPLICABILI AI FATTI: {applicabili}. Metti ciascuna tra le norme della questione che "
               "disciplina; quelle che non disciplinano nessuna questione saranno trattate a parte, alla fine.\n\n"
               if applicabili else "")
            + (f"NORME APPLICATE DALLA GIURISPRUDENZA SU FATTI SIMILI (sono in NORME): {votate}. Per ogni "
               "questione valuta se una di queste la disciplina, leggendone rubrica e testo; se sì, mettila tra "
               "le sue norme, altrimenti lasciala fuori.\n\n" if votate else "")
            + (f"{ANGLE_TITLE}: {analysis[PLAN_ANGLE_KEY]}\n{ANGLE_RULE}\n\n" if (analysis or {}).get(PLAN_ANGLE_KEY) else "")
            + _fixed_block(analysis)
            + "Prepara ora il piano.")


# Words too common in headings and issue titles to tie them together.
_HEADING_STOP = {"disposizioni", "articolo", "codice", "norme", "norma", "generali", "comuni", "particolari", "speciali",
                 "della", "delle", "dello", "degli", "nella", "nelle", "altri", "altre", "casi", "pena", "pene",
                 "reato", "reati", "delitto", "delitti", "questione", "responsabilità", "diritto", "diritti"}
# Legal words that headings of every area share: alone they never tie an article to an issue. The lawyer's
# review of the boat case (26 Oct): art. 492 c.c. ("Garanzia", the heir's) and art. 2254 ("Garanzia e rischi
# dei conferimenti") were kept for "Garanzia delle cose vendute" on the word "garanzia" alone.
_HEADING_GENERIC = {"garanzia", "garanzie", "danno", "danni", "risarcimento", "contratto", "contratti",
                    "obbligazione", "obbligazioni", "effetti", "termini", "condizioni", "azione", "azioni",
                    "esclusione", "tutela", "provvedimenti", "procedimento"}


def _heading_shares_title(heading: str, title: str) -> bool:
    """The article's heading and the issue's title share a content word (5-letter stems):
    "Furto in abitazione e furto con strappo" and "Tentativo di furto in abitazione" do;
    "Garanzia" (art. 492 c.c.) and "Garanzia per vizi della barca" do not."""
    def stems(text):
        return {w[:5] for w in re.findall(r"\w{4,}", (text or "").lower())
                if w not in _HEADING_STOP and w not in _HEADING_GENERIC}
    return bool(stems(heading) & stems(title))


def _yes(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _no(value: Any) -> bool:
    return value is False or str(value).strip().lower() == "false"


def _issue_check(issue: Dict[str, Any], article_rows: List[Dict[str, Any]], headings: Dict[tuple, str]) -> Any:
    """The model's verdicts on one issue's articles (pertinent, contradicted)."""
    blocks = []
    for ref in issue["norme"]:
        text = " ".join(" ".join(str((r.get("s") or {}).get("plain_text") or r.get("_passage")
                                     or (r.get("s") or {}).get("abstract") or "").split())
                        for r in article_rows if r.get("_article") == ref)
        blocks.append(f"[art. {ref[0]} {ref[1]}]\nRubrica: {headings.get(ref) or '-'}\nTesto: {article_text.clause_cut(text, 1200) or '-'}")
    human = (f"QUESTIONE: {issue['titolo']}\nARGOMENTO DELLA PARTE ASSISTITA: {issue['argomento']}\n\n"
             "NORME COLLEGATE:\n" + "\n\n".join(blocks))
    system = _PLAN_CHECK_SYSTEM.replace('Nel dubbio: pertinente', _PLAN_CHECK_DIRECTION + 'Nel dubbio: pertinente') \
        if _writer_rules() else _PLAN_CHECK_SYSTEM
    return dg._chat_json([SystemMessage(content=system), HumanMessage(content=human)],
                         max_tokens=250 + 180 * len(issue["norme"]), temperature=0)


def check_plan(plan: Dict[str, Any], article_rows: List[Dict[str, Any]], headings: Dict[tuple, str],
               protected: Optional[List[tuple]] = None) -> Tuple[Optional[Dict[str, Any]], dict]:
    """(plan, checked): each issue's articles checked in parallel against their heading and text.
    Off-topic articles are removed; an issue whose argument contradicts its article is dropped;
    answers and subordinate requests naming a removed article go too (the writer then answers
    from the issues). A failed check keeps its issue as it was. None when no issue is left.
    An article voted by similar rulings (_vote) is never removed and never drops its issue:
    the verdict is only logged ("kept_by_vote"); so is one of `protected` (PLAN_KEEP_APPLICABLE_
    ENABLED's applicable articles, "kept_as_applicable")."""
    checked: Dict[str, List[str]] = {"off_topic": [], "contradicted": [], "kept_by_heading": [], "kept_by_vote": []}
    if protected is not None:
        checked["kept_as_applicable"] = []
    voted = set(_voted(article_rows))
    keepers = voted | set(protected or ())

    def keep_log(ref):
        return checked["kept_by_vote"] if ref in voted else checked["kept_as_applicable"]
    issues = [q for q in plan["questioni"] if q["norme"]]
    if not issues:
        return plan, checked

    def run(issue):
        try:
            return _issue_check(issue, article_rows, headings)
        except Exception as exc:
            _log_exception("run", exc)
            logger.warning("plan-then-write: plan check failed for %r, issue kept: %s", issue["titolo"], exc)
            return None

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(issues)) as pool:
        results = dict(zip((id(q) for q in issues), pool.map(run, issues)))
    dg.vlog("case_analysis_plan_check_llm", duration_ms=(time.time() - t0) * 1000)
    kept, removed_refs = [], set()
    for q in plan["questioni"]:
        verdicts = results.get(id(q))
        by_ref = {}
        for v in verdicts if isinstance(verdicts, list) else []:
            refs = _refs_of(str(v.get("articolo") or ""), set(q["norme"])) if isinstance(v, dict) else []
            if refs:
                by_ref[refs[0]] = v
        for n, c in q["norme"]:
            if (n, c) in keepers and _yes(by_ref.get((n, c), {}).get("contraddice")):
                keep_log((n, c)).append(f"{n} {c} (contraddice)")
        if any(_yes(by_ref.get(ref, {}).get("contraddice")) for ref in q["norme"] if ref not in keepers):
            checked["contradicted"].append(q["titolo"])
            removed_refs.update(q["norme"])
            continue
        # Round 4 of the PM's cases: the check called 624-bis c.p. off-topic for "tentativo di furto
        # in abitazione" in all four burglary runs, and no answer named the charge. An article whose
        # heading names what the issue is about stays.
        judged_off = [ref for ref in q["norme"] if _no(by_ref.get(ref, {}).get("pertinente"))]
        kept_by_heading = [ref for ref in judged_off if _heading_shares_title(headings.get(ref, ""), q["titolo"])]
        kept_as_keepers = [ref for ref in judged_off if ref in keepers and ref not in kept_by_heading]
        off = [ref for ref in judged_off if ref not in kept_by_heading and ref not in keepers]
        checked["off_topic"] += [f"{n} {c}" for n, c in off]
        checked["kept_by_heading"] += [f"{n} {c}" for n, c in kept_by_heading]
        for n, c in kept_as_keepers:
            keep_log((n, c)).append(f"{n} {c} (fuori tema)")
        removed_refs.update(off)
        kept.append({**q, "norme": [ref for ref in q["norme"] if ref not in off]})
    if not kept:
        return None, checked
    still_used = {ref for q in kept for ref in q["norme"]}
    gone = removed_refs - still_used

    def names_gone(text: str) -> bool:
        return any(ref in gone for ref in _refs_of(text, gone))

    answers = [] if checked["contradicted"] else [a for a in plan["risposte"] if not names_gone(a["risposta"])]
    return {**plan, "questioni": kept, "risposte": answers,
            "subordinate": [s for s in plan["subordinate"] if not names_gone(s)]}, checked


# --- REVIEW_FIXES_ENABLED, the live answer of 26 Oct (a long narrative on a locked dirt road) ---------------------
# 1. A right the facts show ALREADY acquired (servitù per destinazione del padre di famiglia, usucapione, a contract)
#    is argued first, its recognition and protection (accertamento, confessoria); a remedy that creates the right
#    anew (passaggio coattivo, with its indennità) only as subordinate -- the answer led with art. 1051 c.c.
# 5. Every explicit question of the request gets its own answer and a treatment ("una richiesta anche
#    sanzionatoria?" got none).
# 6. Two issues on the same governing article or with near-identical titles are one ("Passaggio coattivo" and
#    "Diritto di passaggio", both art. 1051 c.c.); an evidence list identical to an earlier one is not repeated.

_ACQUIRED_RULE = (
    "- Se i fatti o i DIRITTI GIÀ ACQUISITI indicano che la parte assistita ha già acquistato il diritto (servitù "
    "per destinazione del padre di famiglia, usucapione, un contratto), la prima questione è il riconoscimento e "
    "la tutela di quel diritto (accertamento, azione confessoria, cessazione delle turbative); un rimedio che "
    "costituisce il diritto ex novo (passaggio coattivo con indennità, costituzione giudiziale di una servitù) va "
    "solo in via subordinata, dopo, e non è la questione centrale.\n"
    "- Le domande subordinate sono ciò che chiede la parte assistita: mai le difese della controparte (un concorso di "
    "colpa dell'assistito va tra i punti deboli o nelle obiezioni).\n")
_QUESTIONS_RULE = (
    "- Ogni quesito dei QUESITI ESPLICITI ha la sua voce in \"risposte\", con il quesito ripreso, e una questione o "
    "una subordinata che lo tratta; se chiede sanzioni o penalità per chi non adempie, considera le misure "
    "coercitive indicate negli STRUMENTI PROCESSUALI.\n")
_ACQUIRED_TITLE = "DIRITTI GIÀ ACQUISITI SECONDO I FATTI"
_QUESTIONS_TITLE = "QUESITI ESPLICITI"

_ACQUIRED_RE = re.compile(r"destinazione del padre di famiglia|usucapi|servit\w* (?:costituita|acquisita|nata) "
                          r"(?:per|con|da)|diritto (?:già )?acquisit|in forza (?:del|di un) (?:contratto|titolo|atto)",
                          re.IGNORECASE)
_CONSTITUTIVE_RE = re.compile(r"passaggio coattiv|servit\w* coattiv|costituzion\w* (?:coattiv|giudizial|della servit)|"
                              r"costituire (?:la|una) servit|indennit\w* (?:per il|di) passaggio", re.IGNORECASE)
_RECOGNITION_RE = re.compile(r"accertament|confessori|riconosciment|destinazione del padre|usucapi|esistenza della "
                             r"servit|tutela (?:del|della) (?:diritto|servit)|cessazione delle turbative|reintegr",
                             re.IGNORECASE)
_CONSTITUTIVE_ARTICLES = [[1032, 1032], [1051, 1057]]
_RECOGNITION_ARTICLES = [[1031, 1031], [1061, 1062], [1079, 1079], [1158, 1167], [1168, 1170]]


def _civil_in(ref: tuple, ranges: List[List[int]]) -> bool:
    from . import fact_signals
    return _code_key(ref[1]) == "c.c." and fact_signals._in_ranges(ref[0], ranges)


def _issue_kind(issue: Dict[str, Any]) -> str:
    """"recognition" (of a right already acquired), "constitutive" (a remedy creating it) or "": by the title,
    else by the articles."""
    title = issue.get("titolo") or ""
    if _RECOGNITION_RE.search(title):
        return "recognition"
    if _CONSTITUTIVE_RE.search(title):
        return "constitutive"
    norms = issue.get("norme") or []
    recognition = any(_civil_in(r, _RECOGNITION_ARTICLES) for r in norms)
    constitutive = any(_civil_in(r, _CONSTITUTIVE_ARTICLES) for r in norms)
    return "recognition" if recognition and not constitutive else "constitutive" if constitutive and not recognition else ""


def acquired_rights(analysis: Dict[str, Any], stated_only: bool = False) -> List[str]:
    """The rights the case shows already acquired: the fired fact rules marked acquired_right (their institution and
    why), then the analysis' questions and institutions naming one. stated_only (v60g P6: the interactive plan with its
    issues fixed): the fired rules only -- a question's name is no fact."""
    out = []
    for rule in (_signals(analysis) or {}).get("fired") or []:
        if rule.get("acquired_right") and not rule.get("unstated"):      # unstated: interactive_flow.mark_unstated
            out.append(f"{', '.join(rule.get('institutions') or [])}: {rule.get('why') or ''}".strip(": "))
    for q in [] if stated_only else (analysis or {}).get("questioni") or []:
        names = [q.get("titolo") or ""] + [i.get("nome") or "" for i in q.get("istituti") or [] if isinstance(i, dict)]
        out += [n for n in names if n and _ACQUIRED_RE.search(n) and not _CONSTITUTIVE_RE.search(n)]
    return list(dict.fromkeys(out))


_QUESTION_STOP = {"richiesta", "richiesto", "possibile", "esserci", "essere", "anche", "quali", "quale", "posso",
                  "possiamo", "potrebbe", "potrei", "cliente", "propri", "proprio", "propria", "nostro", "nostra",
                  "fare", "questo", "questa", "dovrebbe", "chiedere", "allora", "ancora", "quanto", "come"}


def explicit_questions(request: str, facts: str = "") -> List[str]:
    """The questions the request asks, each sentence ending with "?" (else the facts' own, when the request has none)."""
    for text in (request, facts):
        found = [" ".join(q.split()) for q in re.findall(r"[^.?!\n]*\?", text or "")]
        found = [q for q in found if len(_question_stems(q)) >= 2]
        if found:
            return list(dict.fromkeys(found))
    return []


def _question_stems(text: str) -> set:
    return {w[:6] for w in re.findall(r"[a-zàèéìòù]{5,}", (text or "").lower()) if w not in _QUESTION_STOP}


def _answers(question: str, text: str) -> bool:
    """`text` takes up the question: half its content words at least, and a question about sanctions for
    non-compliance answered with them (fact_signals.asks_sanction)."""
    from . import fact_signals
    stems = _question_stems(question)
    found = _question_stems(text)
    if fact_signals.asks_sanction(question) and not re.search(r"sanzion|coerci|614-bis|penalit|astreinte",
                                                               text or "", re.IGNORECASE):
        return False
    return bool(stems) and len(stems & found) * 2 >= len(stems)


def _plan_problems(plan: Dict[str, Any], analysis: Dict[str, Any], questions: List[str]) -> List[str]:
    """What the plan gets wrong on items 1 and 5: a constitutive remedy argued without (or as the central question
    instead of) the recognition of the right already acquired; an explicit question with no answer or treatment."""
    problems = []
    kinds = [_issue_kind(q) for q in plan["questioni"]]
    central_constitutive = bool(_CONSTITUTIVE_RE.search(plan.get("questione_centrale") or ""))
    if acquired_rights(analysis) and ("constitutive" in kinds or central_constitutive):
        if "recognition" not in kinds:
            problems.append("manca la questione del riconoscimento e della tutela del diritto già acquisito "
                            f"({'; '.join(acquired_rights(analysis))[:300]}): va per prima, il rimedio costitutivo "
                            "solo in via subordinata")
        if central_constitutive:
            problems.append("la questione centrale è un rimedio costitutivo: va posta sul riconoscimento del diritto "
                            "già acquisito")
    treated = " ".join(f"{q['titolo']} {q['argomento']}" for q in plan["questioni"]) + " " + " ".join(plan["subordinate"])
    for question in questions:
        if not any(_answers(question, f"{a['quesito']} {a['risposta']}") for a in plan["risposte"]):
            problems.append(f"il quesito «{question}» non ha la sua risposta in \"risposte\"")
        elif not _answers(question, treated):
            problems.append(f"il quesito «{question}» non è trattato da nessuna questione o subordinata")
    return problems


# PLAN_PATCH_REDO (v46, c1: a whole second _plan_json call, 6,000 max_tokens and 29 s, for one explicit question without
# its answer): when every problem _plan_problems finds is a question with no answer in "risposte", only those answers are
# asked for (one call, _PATCH_TOKENS), from the case and the plan as made, and added to its "risposte"; an answer that
# does not come back, or a failed call, leaves the full redo as before. Any other problem (an issue missing, the central
# question, a question no issue treats) is structural: the full redo. Logged as case_analysis_plan_patch.
PLAN_PATCH_FLAG = "PLAN_PATCH_REDO"
_PATCH_TOKENS = 600
_PATCH_SYSTEM = (
    "Sei un avvocato italiano. Ti vengono dati il CASO, il PIANO di un parere già deciso (parte assistita, questione "
    "centrale, questioni con le loro norme) e alcuni QUESITI del cliente che il piano non ha ancora risposto. Per "
    "ciascun quesito scrivi la risposta diretta in una o due frasi, coerente con il PIANO e con le sue norme, senza "
    "aggiungere fatti che il CASO non contiene e senza citare norme che il PIANO non ha. Rispondi solo con un array "
    "JSON: [{\"quesito\": \"il quesito ripreso\", \"risposta\": \"...\"}].")


def _unanswered(plan: Dict[str, Any], questions: List[str]) -> List[str]:
    """The explicit questions with no answer in the plan's "risposte" (_plan_problems' first kind)."""
    return [q for q in questions if not any(_answers(q, f"{a['quesito']} {a['risposta']}") for a in plan["risposte"])]


def _patch_answers(plan: Dict[str, Any], questions: List[str], facts: str, request: str,
                   analysis: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """(plan, whether every question now has its answer): PLAN_PATCH_REDO's one call."""
    t0 = time.time()
    brief = "\n".join(
        [f"Parte assistita: {plan.get('parte_assistita') or '-'}", f"Questione centrale: {plan.get('questione_centrale') or '-'}"]
        + [f"{k}. {q['titolo']} (norme: {', '.join(f'art. {n} {c}' for n, c in q['norme']) or '-'}): {q['argomento']}"
           for k, q in enumerate(plan["questioni"], 1)])
    human = (f"CASO:\n{dg._case_text(facts, request)[:6000]}\n\nPIANO:\n{brief}\n\nQUESITI:\n"
             + "\n".join(f"{k}. {q}" for k, q in enumerate(questions, 1)))
    added: List[Dict[str, str]] = []
    try:
        raw = dg._chat_json([SystemMessage(content=_PATCH_SYSTEM), HumanMessage(content=human)],
                            max_tokens=_PATCH_TOKENS, temperature=0)
        items = [x for x in raw if isinstance(x, dict)] if isinstance(raw, list) else []
        for k, question in enumerate(questions):
            reply = next((x for x in items if _answers(question, str(x.get("quesito") or ""))), None) \
                or (items[k] if k < len(items) and not str(items[k].get("quesito") or "").strip() else None)
            answer = " ".join(str((reply or {}).get("risposta") or "").split())
            if answer:
                added.append({"quesito": question, "risposta": answer})
    except Exception as exc:
        _log_exception("_patch_answers", exc)
    patched = {**plan, "risposte": list(plan["risposte"]) + added}
    complete = not _unanswered(patched, questions)
    dg.vlog("case_analysis_plan_patch", {"kind": "answers", "questions": questions, "answered": [a["quesito"] for a in added],
                                         "complete": complete, "then": "kept" if complete else "full redo"},
            duration_ms=(time.time() - t0) * 1000)
    return (patched, True) if complete else (plan, False)


# REVIEW_FIXES_ENABLED, v48 (the PO's yellow on Leonardo: art. 1079 c.c. gives the confessoria, the end of the
# obstacles, the rimessione in pristino and damages; the answer used two): the plan's main remedy article -- its first
# article, outside the subordinate issues and the rule-inserted rows, whose text lists three or more remedies, split on
# "può" / "oltre" -- has each remedy in the plan's "risposte" (a missing one added to the answer citing it, else to the
# first) and in the writer's material with the rule to address each (REMEDIES_TITLE). Logged as
# case_analysis_plan_remedies.
REMEDIES_TITLE = "RIMEDI DELLA NORMA PRINCIPALE"
REMEDIES_RULE = "Tratta ciascuno di questi rimedi, uno per uno, con la sua norma: nessuno va omesso."
_REMEDY_SPLIT_RE = re.compile(r"(?<!\bnon )\b(?:può|possono|oltre)\b", re.IGNORECASE)
_REMEDY_LEAD_RE = re.compile(r"^(?:\s|anche|altresì|inoltre|chiedere|domandare|ottenere|il|la|lo|i|gli|le|l['’])+", re.IGNORECASE)
_REMEDY_MIN = 3


def remedies_of(text: str) -> List[str]:
    """The remedies an article's text lists ("può farne riconoscere ... e può far cessare ... Può anche chiedere la
    rimessione ..., oltre il risarcimento dei danni"): the part after each "può" / "oltre", to its first clause end."""
    parts = _REMEDY_SPLIT_RE.split(" ".join((text or "").split()))[1:]
    out = []
    for part in parts:
        head = re.split(r"[.;:]|,\s|\s+e\s+(?=può\b)|\s+contro\s+chi\b", part.strip(), maxsplit=1)[0]
        head = re.sub(r"(?:\s+e)+$", "", _REMEDY_LEAD_RE.sub("", head).strip(" ,"))
        if len(head.split()) >= 2 and head.lower() not in (x.lower() for x in out):
            out.append(head)
    return out


def main_remedies(plan: Dict[str, Any], article_rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """{"article", "ref", "remedies"} of the plan's main remedy article, or None."""
    texts = {}
    for row in article_rows or []:
        ref = row.get("_article")
        if ref and ref not in texts:
            texts[ref] = " ".join(str((row.get("s") or {}).get("plain_text") or "").split())
    by_rule = rule_inserted(plan)
    for q in plan.get("questioni") or []:
        if str(q.get("titolo") or "").lower().startswith("in via subordinata"):
            continue
        for ref in q["norme"]:
            if ref in by_rule:
                continue
            text = texts.get(ref) or next((t for k, t in texts.items() if _same_ref(k, ref)), "")
            found = remedies_of(text)
            if len(found) >= _REMEDY_MIN:
                return {"article": f"art. {ref[0]} {_code_key(ref[1])}", "ref": ref, "remedies": found}
    return None


_REMEDY_VERBS = {"farne", "farla", "farlo", "farli", "chiedere", "domandare", "ottenere", "agire"}


def _remedy_named(remedy: str, text: str) -> bool:
    """The remedy's head word (its first content word past a bare verb: riconoscere, cessare, rimessione, risarcimento)
    is in the text, by its stem."""
    words = [w for w in re.findall(r"[a-zà-ù]{5,}", remedy.lower()) if w not in _REMEDY_VERBS]
    have = {w[:6] for w in re.findall(r"[a-zà-ù]{5,}", (text or "").lower())}
    return bool(words) and words[0][:6] in have


def enumerate_remedies(plan: Dict[str, Any], article_rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """(plan, log): the main remedy article's remedies all in the plan's answers, the plan carrying them ("rimedi")."""
    found = main_remedies(plan, article_rows)
    if not found:
        return plan, None
    answers = [dict(a) for a in plan.get("risposte") or []]
    said = " ".join(f"{a.get('quesito', '')} {a.get('risposta', '')}" for a in answers)
    missing = [r for r in found["remedies"] if not _remedy_named(r, said)]
    target = None
    if missing and answers:
        number = found["ref"][0]
        target = next((a for a in answers if re.search(rf"\b{re.escape(number)}\b", a.get("risposta") or "")), answers[0])
        target["risposta"] = (f"{target.get('risposta', '').rstrip()} I rimedi dell'{found['article']}: "
                              + "; ".join(found["remedies"]) + ".").strip()
    log = {"article": found["article"], "remedies": found["remedies"], "missing": missing,
           "added_to": (f"risposta {answers.index(target) + 1}" + (f" ({target['quesito'][:60]})" if target.get("quesito") else ""))
           if target else None}
    dg.vlog("case_analysis_plan_remedies", log)
    return {**plan, "risposte": answers, "rimedi": {"article": found["article"], "remedies": found["remedies"]}}, log


def _remedies_block(plan: Dict[str, Any]) -> str:
    r = plan.get("rimedi") or {}
    if not r.get("remedies"):
        return ""
    return (f"{REMEDIES_TITLE}:\n- {r['article']}: " + "; ".join(r["remedies"]) + f".\n{REMEDIES_RULE}")


# PLAN_PATCH_REDO, v48 (14 of 32 v47 cases: a second whole _plan_json call, 6,000 max_tokens, for the stance check's
# opposing arguments): each opposing argument goes into the plan as made as the objection of the issue it belongs to,
# with the client's reply, and the issue's own argument restated for the client when it was the opposing one -- one call
# of _OPPOSING_TOKENS. A failed call, or an argument left without its issue, objection and reply: the full redo as before.
_OPPOSING_TOKENS = 800
_OPPOSING_SYSTEM = (
    "Sei un avvocato italiano. Ti vengono dati il CASO, il PIANO di un parere già deciso, con le sue questioni numerate, "
    "e alcuni ARGOMENTI che una revisione ha trovato favorevoli alla controparte, non alla parte assistita. Per ciascun "
    "argomento indica la questione del PIANO a cui appartiene e riformulalo come obiezione della controparte, con la "
    "replica della parte assistita; se l'argomento della questione era proprio quello, riscrivilo a favore della parte "
    "assistita, altrimenti lascialo vuoto. Non aggiungere fatti che il CASO non contiene e non citare norme che il "
    "PIANO non ha. Rispondi solo con un array JSON: [{\"argomento_contrario\": \"ripreso\", \"questione\": numero, "
    "\"obiezione\": \"...\", \"replica\": \"...\", \"argomento\": \"... o vuoto\"}].")


def _patch_opposing(plan: Dict[str, Any], opposing: List[Dict[str, str]], facts: str, request: str,
                    analysis: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """(plan, whether every opposing argument now sits in an issue as its objection with a reply)."""
    t0 = time.time()
    issues = plan["questioni"]
    brief = "\n".join(
        [f"Parte assistita: {plan.get('parte_assistita') or '-'}", f"Questione centrale: {plan.get('questione_centrale') or '-'}"]
        + [f"{k}. {q['titolo']}: {q.get('argomento') or '-'} (obiezione: {q.get('obiezione') or '-'}; replica: "
           f"{q.get('replica') or '-'})" for k, q in enumerate(issues, 1)])
    human = (f"CASO:\n{dg._case_text(facts, request)[:6000]}\n\nPIANO:\n{brief}\n\nARGOMENTI:\n"
             + "\n".join(f"{k}. {o['argomento']}: {o.get('perche') or ''}" for k, o in enumerate(opposing, 1)))
    placed: List[Dict[str, Any]] = []
    try:
        raw = dg._chat_json([SystemMessage(content=_OPPOSING_SYSTEM), HumanMessage(content=human)],
                            max_tokens=_OPPOSING_TOKENS, temperature=0)
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, dict):
                continue
            try:
                k = int(str(item.get("questione")).strip()) - 1
            except ValueError:
                continue
            objection, reply = _text(item.get("obiezione")), _text(item.get("replica"))
            if 0 <= k < len(issues) and objection and reply:
                placed.append({"issue": k, "obiezione": objection, "replica": reply,
                               "argomento": _text(item.get("argomento")),
                               "opposing": _text(item.get("argomento_contrario"))})
    except Exception as exc:
        _log_exception("_patch_opposing", exc)
    complete = bool(placed) and len(placed) >= len(opposing)
    out = plan
    if complete:
        new = [dict(q) for q in issues]
        for p in placed:
            q = new[p["issue"]]
            q["obiezione"] = "; ".join(x for x in (q.get("obiezione"), p["obiezione"]) if x)
            q["replica"] = "; ".join(x for x in (q.get("replica"), p["replica"]) if x)
            if p["argomento"]:
                q["argomento"] = p["argomento"]
        out = {**plan, "questioni": new}
    dg.vlog("case_analysis_plan_patch", {"kind": "opposing", "opposing": [o["argomento"] for o in opposing],
                                         "placed": [{"issue": issues[p["issue"]]["titolo"], **{k: v for k, v in p.items()
                                                                                              if k != "issue"}}
                                                    for p in placed],
                                         "complete": complete, "then": "kept" if complete else "full redo"},
            duration_ms=(time.time() - t0) * 1000)
    return out, complete


def _title_stems(title: str) -> set:
    return {w[:5] for w in re.findall(r"\w{4,}", (title or "").lower()) if w not in _HEADING_STOP}


def _near_titles(a: str, b: str) -> bool:
    x, y = _title_stems(a), _title_stems(b)
    return bool(x and y) and (x <= y or y <= x or len(x & y) * 3 >= 2 * len(x | y))


def merge_duplicate_issues(plan: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """(plan, merged): an issue whose governing (first) article is an earlier issue's, or whose title is nearly the
    same, merged into that one -- its articles and rulings added, its argument kept only when the first has none."""
    out: List[Dict[str, Any]] = []
    merged = []
    for q in plan["questioni"]:
        twin = next((p for p in out if (p["norme"] and q["norme"] and _same_article(p["norme"][0], q["norme"][0]))
                     or _near_titles(p["titolo"], q["titolo"])), None)
        if twin is None:
            out.append(dict(q))
            continue
        twin["norme"] = twin["norme"] + [r for r in q["norme"] if r not in twin["norme"]]
        twin["pronunce"] = twin["pronunce"] + [r for r in q["pronunce"] if r not in twin["pronunce"]]
        for key in ("argomento", "obiezione", "replica"):
            twin[key] = twin.get(key) or q.get(key) or ""
        merged.append({"into": twin["titolo"], "merged": q["titolo"]})
    return {**plan, "questioni": out}, merged


_SUBORDINATE_PREFIX = "In via subordinata: "


def _rule_refs(analysis: Dict[str, Any]) -> Tuple[tuple, ...]:
    return tuple(tuple(r) for inst in (analysis or {}).get("rule_institutions") or [] for r in inst.get("refs") or [])


def one_issue_per_article(plan: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(plan, log): every step placed its articles on its own; last, an article is argued in one issue. An issue whose
    main article an earlier one holds is merged into it, its articles and rulings added; an article repeated further on
    leaves that issue, which refers back to the first (Leonardo v4: 1079 twice, once subordinate to itself)."""
    kept: List[Dict[str, Any]] = []
    merged, moved = [], []
    for q in plan["questioni"]:
        twin = next((p for p in kept if q["norme"] and q["norme"][0] in p["norme"]), None)
        if twin is not None:
            twin["norme"] = twin["norme"] + [r for r in q["norme"] if r not in twin["norme"]]
            twin["pronunce"] = twin["pronunce"] + [r for r in q["pronunce"] if r not in twin["pronunce"]]
            merged.append({"into": twin["titolo"], "merged": q["titolo"]})
            continue
        kept.append({**q, "norme": list(q["norme"]), "pronunce": list(q["pronunce"])})
    seen: Dict[tuple, str] = {}
    issues = []
    for q in kept:
        repeated = [r for r in q["norme"] if r in seen]
        q["norme"] = [r for r in q["norme"] if r not in seen]
        for ref in repeated:
            moved.append({"article": f"{ref[0]} {ref[1]}", "stays_in": seen[ref], "left": q["titolo"]})
            q["argomento"] = f"{q['argomento']} Per l'art. {ref[0]} {_code_key(ref[1])} si rinvia alla questione «{seen[ref]}»."
        seen.update({r: q["titolo"] for r in q["norme"]})
        issues.append(q)
    titles = {q["titolo"] for q in issues}
    forced = [{**f, "questioni": [t for t in f["questioni"] if t in titles]} for f in plan.get("forzate") or []]
    return {**plan, "questioni": issues, **({"forzate": forced} if "forzate" in plan else {})}, {"merged": merged, "moved": moved}


_MIXED_REMEDY_TITLE = "Costituzione giudiziale del diritto"
_MIXED_REMEDY_ARGUMENT = ("Solo se il diritto già acquisito non fosse riconosciuto: il rimedio che lo costituisce con sentenza "
                          "e contro indennità, che non è il titolo del diritto acquisito.")


def _split_mixed(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """An issue holding recognition and constitutive articles: the first stay, the others become their own issue."""
    out = []
    for q in issues:
        norms = q.get("norme") or []
        remedy = [r for r in norms if _civil_in(r, _CONSTITUTIVE_ARTICLES)]
        if remedy and any(_civil_in(r, _RECOGNITION_ARTICLES) for r in norms):
            out.append({**q, "norme": [r for r in norms if r not in remedy]})
            out.append({"titolo": _MIXED_REMEDY_TITLE, "norme": remedy, "pronunce": [], "argomento": _MIXED_REMEDY_ARGUMENT,
                        "obiezione": "", "replica": "", **({"passi": []} if "passi" in q else {})})
        else:
            out.append(q)
    return out


_DEFENCE_RE = re.compile(r"concorso (?:di colpa|colposo|del danneggiato)|\b1227\b|colpa (?:del|della) (?:cliente|assistit|danneggiat)",
                         re.IGNORECASE)


def without_defences(plan: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """(plan, moved): the subordinate requests are what the client asks; the other side's defences (the client's concorso
    di colpa) go among the weak points."""
    moved = [s for s in plan["subordinate"] if _DEFENCE_RE.search(s)]
    if not moved:
        return plan, []
    return {**plan, "subordinate": [s for s in plan["subordinate"] if s not in moved],
            "punti_deboli": list(plan["punti_deboli"]) + moved}, moved


def order_for_acquired(plan: Dict[str, Any], acquired: List[str], rule_refs: Tuple[tuple, ...] = ()
                       ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(plan, log): with a right already acquired, the recognition issues first, a constitutive remedy last and
    titled as subordinate (and named in the plan's subordinate requests); an issue arguing both is split, the remedy
    going last; an issue on a fired rule's articles counts as recognition; a central question naming a constitutive remedy
    becomes the first issue; the plan says which right is acquired ("diritto_acquisito", for every section)."""
    if not acquired:
        return plan, {}
    plan = {**plan, "questioni": _split_mixed(plan["questioni"])}
    kinds = [_issue_kind(q) or ("recognition" if set(q.get("norme") or []) & set(rule_refs) else "") for q in plan["questioni"]]
    central_remedy = bool(_CONSTITUTIVE_RE.search(plan.get("questione_centrale") or ""))
    if "constitutive" not in kinds and not central_remedy:
        return plan, {}
    first = [q for q, k in zip(plan["questioni"], kinds) if k == "recognition"]
    middle = [q for q, k in zip(plan["questioni"], kinds) if k == ""]
    last = [{**q, "titolo": q["titolo"] if re.search(r"subordin", q["titolo"], re.IGNORECASE)
             else _SUBORDINATE_PREFIX + q["titolo"]} for q, k in zip(plan["questioni"], kinds) if k == "constitutive"]
    issues = first + middle + last
    subordinate = list(plan["subordinate"]) + [
        f"{q['titolo'][len(_SUBORDINATE_PREFIX):] if q['titolo'].startswith(_SUBORDINATE_PREFIX) else q['titolo']}: "
        "solo se il diritto già acquisito non fosse riconosciuto" for q in last
        if not any(_near_titles(q["titolo"], s) for s in plan["subordinate"])]
    log = {"acquired": acquired, "before": [q["titolo"] for q in plan["questioni"]],
           "after": [q["titolo"] for q in issues]}
    central = {"questione_centrale": issues[0]["titolo"]} if central_remedy and first else {}
    return {**plan, **central, "questioni": issues, "subordinate": subordinate, "diritto_acquisito": "; ".join(acquired)}, log


def split_adverse(plan: Dict[str, Any], references: Dict[str, List[Dict[str, Any]]]) -> Tuple[Dict[str, Any], List[str]]:
    """(plan, moved): RULING_FIT_ENABLED's adverse rulings (_fit_adverse) out of each issue's supporting rulings
    into its "pronunce_contrarie", cited only in the objection with the reply."""
    adverse = {ref for ref, rows in references.items() if any(r.get("_fit_adverse") for r in rows)}
    moved: List[str] = []
    issues = []
    for q in plan["questioni"]:
        against = [p for p in q["pronunce"] if p in adverse]
        moved += against
        issues.append({**q, "pronunce": [p for p in q["pronunce"] if p not in adverse],
                       "pronunce_contrarie": list(dict.fromkeys(q.get("pronunce_contrarie", []) + against))})
    return {**plan, "questioni": issues, "pronunce_contrarie": sorted(adverse)}, list(dict.fromkeys(moved))


def _arguments_text(plan: Dict[str, Any]) -> str:
    """The plan's arguments as the stance check reads a section 3."""
    return "3. Argomenti\n" + "\n".join(f"#### {q['titolo']}\n{q['argomento']}" for q in plan["questioni"])


PLAN_RETRY_EVENT = "case_analysis_plan_retry"


def make_plan(facts: str, request: str, analysis: Dict[str, Any], articles: str, rulings: str, principles: str,
              article_rows: List[Dict[str, Any]], references: Dict[str, List[Dict[str, Any]]],
              _retry: bool = True) -> Optional[Dict[str, Any]]:
    """The cleaned plan, made once more if its arguments favour the other side, then checked
    (check_plan); None on failure. v49 (v48 sempronio_no_hint: no plan, the parere in one call): a plan left without a
    usable issue is made once more from the same inputs (_plan_json again, then every step after it) before falling
    back to one call; logged as case_analysis_plan_retry with its outcome. A failed plan call is not retried."""
    known = {r["_article"] for r in article_rows if r.get("_article")}
    headings = _headings(article_rows)
    voted = [ref for ref in _voted(article_rows) if f"[art. {ref[0]} {ref[1]}]" in articles]
    keep_checked = dg.feature_flags.enabled(KEEP_CHECKED_FLAG)
    keep_all = dg.feature_flags.enabled(KEEP_APPLICABLE_FLAG) and not keep_checked
    reasons = _forced(article_rows) if keep_checked and _graded() else None
    applicable = (list(reasons) if reasons is not None else _checked(article_rows) if keep_checked
                  else _applicable(article_rows, facts, request, analysis) if keep_all else [])
    max_issues = _MAX_ISSUES_WIDE if keep_checked or len(applicable) >= _WIDE_APPLICABLE else _MAX_ISSUES
    fixed = list((analysis or {}).get(PLAN_ISSUES_KEY) or [])
    if fixed:                       # v60f: the issues are the chosen topics -- the plan argues them, never selects
        max_issues = max(1, len(fixed))
    system = _plan_system(max_issues)
    if fixed:
        system = re.sub(r"Le questioni per prime, da due a \w+, in ordine di forza per la parte assistita\.",
                        FIXED_ISSUES_RULE, system)
        system = system.replace(_ACQUIRED_RULE.split("\n- ")[0] + "\n", FIXED_ACQUIRED_RULE)       # v60g (P6)
    issue_of = {r["_article"]: r.get("_grade_issue") or "" for r in article_rows if r.get("_article")}
    groups = _issue_groups(reasons, issue_of) if reasons is not None and _by_issue() else None
    human = _plan_human(facts, request, analysis, _with_headings(articles, headings), rulings, principles,
                        _checks(analysis, article_rows, facts, request), voted, applicable,
                        must_place="graded" if keep_checked and _graded() else keep_checked, groups=groups)
    messages = [SystemMessage(content=system), HumanMessage(content=protected_sources.with_rule(human))]
    t0 = time.time()
    problems: List[str] = []
    try:
        raw = _plan_json(messages)
        plan, removed = clean_plan(raw, known, references, max_issues)
        opposing = dg._opposing_arguments(facts, request, analysis, _arguments_text(plan)) if plan else []
        patched_opposing = False
        if opposing and dg.feature_flags.enabled(PLAN_PATCH_FLAG):
            # PLAN_PATCH_REDO (v48): the opposing arguments become objections of the plan as made, one short call
            plan, patched_opposing = _patch_opposing(plan, opposing, facts, request, analysis)
        if opposing and not patched_opposing:
            listing = "\n".join(f"- {o['argomento']}: {o['perche']}" for o in opposing)
            retry = (f"{human}\n\nIl tuo primo piano:\n{_plan_brief(raw)}\n\nUna revisione ha "
                     f"trovato che questi argomenti giovano alla controparte, non a {dg._stance_party(analysis)}:\n"
                     f"{listing}\n\nRifai il piano: ogni argomento deve escludere, ridurre o contrastare la "
                     "responsabilità o la pretesa contro la parte assistita, o sostenerne la pretesa; un fatto "
                     "che favorisce la controparte va trattato come obiezione da superare.")
            plan, removed, raw = _remade(system, retry, plan, removed, raw, known, references, max_issues)
        problems = _plan_problems(plan, analysis, explicit_questions(request, facts)) if plan and _review_fixes() else []
        unanswered = _unanswered(plan, explicit_questions(request, facts)) if problems else []
        if problems and dg.feature_flags.enabled(PLAN_PATCH_FLAG) and len(unanswered) == len(problems):
            # PLAN_PATCH_REDO: only answers missing -- asked for alone and put in the plan, no full redo
            plan, patched = _patch_answers(plan, unanswered, facts, request, analysis)
            if patched:
                problems = []
        if problems:
            listing = "\n".join(f"- {p}" for p in problems)
            again = (f"{human}\n\nIl tuo piano:\n{_plan_brief(raw)}\n\nUna revisione ha trovato:\n"
                     f"{listing}\n\nRifai il piano correggendo questi punti, con le stesse regole.")
            plan, removed, raw = _remade(system, again, plan, removed, raw, known, references, max_issues)
    except Exception as exc:
        _log_exception("make_plan", exc)
        logger.warning("plan-then-write: no plan, the parere is written in one call: %s", exc)
        _fell_back("plan call", exc)
        return None
    finally:
        dg.vlog("case_analysis_plan_llm", duration_ms=(time.time() - t0) * 1000)
    checked: Dict[str, List[str]] = {"off_topic": [], "contradicted": [], "kept_by_heading": [], "kept_by_vote": []}
    check_removed: set = set()
    if plan:
        before = set(_plan_refs(plan))
        plan, checked = check_plan(plan, article_rows, headings, applicable if keep_all or keep_checked else None)
        check_removed = before - (set(_plan_refs(plan)) if plan else set())     # off-topic, or of a contradicted issue
    fixes: Dict[str, Any] = {}
    if plan:
        plan, rule_log = place_rule_institutions(plan, analysis, article_rows)
        dg.vlog("case_analysis_plan_rule_institutions", rule_log)
        plan, moved_defences = without_defences(plan)
    if plan and _review_fixes():
        plan, merged = merge_duplicate_issues(plan)
        plan, order = order_for_acquired(plan, acquired_rights(analysis), _rule_refs(analysis))
        fixes = {"problems": problems, "merged": merged, "order": order, "defences_moved": moved_defences,
                 "questions": explicit_questions(request, facts),
                 "unanswered": [q for q in explicit_questions(request, facts)
                                if not any(_answers(q, f"{a['quesito']} {a['risposta']}") for a in plan["risposte"])]}
    if plan and dg.feature_flags.enabled(dg.RULING_FIT_FLAG):
        plan, moved = split_adverse(plan, references)
        fixes["adverse_moved"] = moved
    if plan and keep_checked:
        plan, placed, grouped, added = _group_checked(plan, applicable, headings, max_issues, issue_of,
                                                      catch_all=not _graded(), reasons=reasons)
        if reasons is not None:
            plan["forzate"] = _forced_issues(plan, reasons, issue_of)
        dg.vlog("case_analysis_plan_applicable", {
            "mode": "checked", "applicable": [f"{n} {c}" for n, c in applicable], "max_issues": max_issues,
            "why": {f"{n} {c}": why for (n, c), why in (reasons or {}).items()},
            "placed": [f"{n} {c}" for n, c in placed],
            "grouped": {f"{n} {c}": title for (n, c), title in grouped.items()},
            "added_to_last_issue": [f"{n} {c}" for n, c in added]})
    if plan and keep_all:
        plan, placed, added = _with_other_norms(plan, applicable)
        dg.vlog("case_analysis_plan_applicable", {
            "applicable": [f"{n} {c}" for n, c in applicable], "max_issues": max_issues,
            "placed": [f"{n} {c}" for n, c in placed], "added_to_last_issue": [f"{n} {c}" for n, c in added]})
    from . import fact_issues
    if plan and fact_issues.resolver_v2():
        plan, placed_log = place_fact_institutions(plan, analysis, article_rows)
        dg.vlog("case_analysis_plan_fact_institutions", placed_log)
        if placed_log["issues_added"] and _review_fixes():         # a new issue may be a recognition or a remedy
            plan, _ = order_for_acquired(plan, acquired_rights(analysis), _rule_refs(analysis))
    if plan and rule_include():
        try:            # logged and skipped on any failure: the plan as it was
            plan, included, skipped = include_by_rule(plan, article_rows, articles, headings, issue_of,
                                                      statement_context(analysis).get("excluded") or (), check_removed)
            dg.vlog("case_analysis_plan_rule_include", {"inserted": included, "count": len(included), "skipped": skipped})
        except Exception as exc:
            _log_exception("make_plan.include_by_rule", exc)
    if plan and dg.feature_flags.enabled(PLAN_CAP_FLAG):
        try:            # PLAN_CAP (v49): at most PLAN_CAP_MAX substantive articles; the rest available, by name
            plan, cap_log = cap_plan(plan, article_rows, dg._refs_in(dg._case_text(facts, request)), plan_cap_max())
            dg.vlog("case_analysis_plan_cap", cap_log)
        except Exception as exc:
            _log_exception("make_plan.cap_plan", exc)
    dg.vlog("case_analysis_plan", {
        "issues": len(plan["questioni"]) if plan else 0, "replanned": bool(opposing),
        "opposing": [o["argomento"] for o in opposing],
        "norme": sorted({f"{n} {c}" for q in (plan or {}).get("questioni", []) for n, c in q["norme"]}),
        "pronunce": sorted({p for q in (plan or {}).get("questioni", []) for p in q["pronunce"]}),
        "removed": removed, "checked": checked, "references_available": len(references),
        "voted_offered": [f"{n} {c}" for n, c in voted], **({"review_fixes": fixes} if fixes else {})})
    if plan:
        plan, single = one_issue_per_article(plan)
        dg.vlog("case_analysis_plan_final", {
            "central": plan.get("questione_centrale"), "issues": [
                {"titolo": q["titolo"], "norme": [f"{n} {c}" for n, c in q["norme"]]} for q in plan["questioni"]], **single})
    if plan is None and _retry:
        logger.warning("plan-then-write: the plan has no usable issue, made once more from the same inputs")
        t1 = time.time()
        again = make_plan(facts, request, analysis, articles, rulings, principles, article_rows, references, _retry=False)
        dg.vlog(PLAN_RETRY_EVENT, {"why": "plan without a usable issue", "removed": removed,
                                   "outcome": "plan" if again else "no plan",
                                   "issues": len(again["questioni"]) if again else 0},
                duration_ms=(time.time() - t1) * 1000)
        return again
    if plan is None:
        logger.warning("plan-then-write: the plan has no usable issue, the parere is written in one call")
        _fell_back("plan without a usable issue", "no argued issue left, after one retry", removed=removed)
        return None
    if not plan["parte_assistita"]:
        plan["parte_assistita"] = str(analysis.get("parte_assistita") or "").strip()
    if _review_fixes():
        try:            # v48: every remedy of the main remedy article in the answers
            plan, _ = enumerate_remedies(plan, article_rows)
        except Exception as exc:
            _log_exception("enumerate_remedies", exc)
    gated = [ref for ref in _plan_refs(plan) if f"{ref[0]} {ref[1]}" in (analysis.get("action_gate") or {})]
    if gated:           # REGIME_CANDIDATES: an action's article resolved for a client not in the action's role
        plan = {**plan, "norme_contrarie": list(dict.fromkeys(list(plan.get("norme_contrarie") or [])
                                                              + [f"{n} {c}" for n, c in gated]))}
        dg.vlog("case_analysis_action_gate_plan", {"adverse": [f"{n} {c}" for n, c in gated]})
    if render_computed.enabled():
        try:            # RENDER_COMPUTED: the action gate's articles out of the remedies, into the objections block
            plan, moved = render_computed.move_gated(plan, analysis)
            if moved:
                dg.vlog("case_analysis_gated_objections", {"moved": moved})
        except Exception as exc:
            _log_exception("move_gated", exc)
    from . import edge_expand
    if edge_expand.enabled():
        edge_expand.note_plan(plan)         # EDGE_EXPAND's reach log
    if role_fit.mode() != "off":
        try:            # ROLE_FIT: whom each substantive article works for; the counterparty's tagged adverse (shadow: logged)
            plan, _ = role_fit.check(plan, article_rows, dg._case_text(facts, request), analysis.get("fact_signals"),
                                     shadow=role_fit.mode() == "shadow")
        except Exception as exc:
            _log_exception("role_fit", exc)
    return plan


# --- 3. the sections -------------------------------------------------------------------------

def _plan_text(plan: Dict[str, Any], answers: bool = False, issues: str = "titles", subordinate: bool = False,
               weak: bool = False, rulings: bool = True) -> str:
    """The plan as the writer reads it; `issues` is "titles", "full" or "" (none); the rulings
    only for a section that also has their passages."""
    party = plan["parte_assistita"] or "la parte assistita"
    lines = [f"Parte assistita: {party}" + (f" ({plan['posizione']})" if plan["posizione"] else "")]
    if plan["obiettivo"]:
        lines.append(f"Obiettivo: {plan['obiettivo']}")
    if plan["questione_centrale"]:
        lines.append(f"Questione centrale: {plan['questione_centrale']}")
    if plan.get("diritto_acquisito"):
        lines.append(f"Diritto già acquisito: {plan['diritto_acquisito']}. Prima il suo riconoscimento e la sua "
                     "tutela; il rimedio che costituisce il diritto ex novo solo in via subordinata.")
    if answers and plan["risposte"]:
        lines.append("Risposte ai quesiti:")
        lines += [f"- {a['quesito']} -> {a['risposta']}" if a["quesito"] else f"- {a['risposta']}"
                  for a in plan["risposte"]]
    if issues:
        lines.append("Questioni, in ordine di forza:")
        for i, q in enumerate(plan["questioni"], 1):
            norms = ", ".join(f"art. {n} {c}" + (f" ({role_fit.ADVERSE_MARK})" if role_fit.is_adverse(plan, (n, c)) else "")
                              for n, c in q["norme"]) or "-"
            cited = f"; pronunce: {', '.join(q['pronunce']) or '-'}" if rulings else ""
            if rulings and q.get("pronunce_contrarie"):
                cited += (f"; pronunce della controparte, solo nell'obiezione con la replica: "
                          f"{', '.join(q['pronunce_contrarie'])}")
            anchors = f"; passi: {', '.join(q['ancore'])}" if q.get("ancore") else ""     # ANCHORED_WRITING_ENABLED
            lines.append(f"{i}. {q['titolo']} (norme: {norms}{cited}{anchors})")
            if issues == "full":
                lines += [f"   Argomento: {q['argomento']}"]
                lines += [f"   Obiezione della controparte: {q['obiezione']}"] if q["obiezione"] else []
                lines += [f"   Replica: {q['replica']}"] if q["replica"] else []
            else:
                lines.append(f"   {q['argomento']}")
    if subordinate and plan["subordinate"]:
        lines += ["Tesi o domande subordinate:"] + [f"- {s}" for s in plan["subordinate"]]
    if weak and plan["punti_deboli"]:
        lines += ["Punti deboli:"] + [f"- {s}" for s in plan["punti_deboli"]]
    return "PIANO:\n" + "\n".join(lines)


def _plan_refs(plan: Dict[str, Any]) -> List[tuple]:
    return list(dict.fromkeys(ref for q in plan["questioni"] for ref in q["norme"]))


def _article_texts(article_rows: List[Dict[str, Any]], refs: List[tuple], chars: int) -> str:
    rows = sorted((r for r in article_rows if r.get("_article") in refs), key=lambda r: refs.index(r["_article"]))
    return dg._source_block(rows, chars)


def _ruling_texts(plan: Dict[str, Any], references: Dict[str, List[Dict[str, Any]]], chars: int) -> str:
    cited = list(dict.fromkeys(p for q in plan["questioni"] for p in q["pronunce"] + q.get("pronunce_contrarie", [])))
    return dg._source_block([references[p][0] for p in cited if references.get(p)], chars)


def _section_system(plan: Dict[str, Any], lang: str, sections_note: str = "", writer_rules: str = "") -> str:
    """sections_note (v60f): the answer's own sections in place of the parere's six (the interactive answer has its own
    list, one number, one set of titles). writer_rules (v60g d): the interactive answer's de-conflicted rule set
    (writer_rules.system_rules) in place of _WRITER_RULES."""
    from .language import language_display_name
    party = plan["parte_assistita"] or "la parte assistita"
    position = f" ({plan['posizione']})" if plan["posizione"] else ""
    rules = _writer_rules()
    return (
        "Sei un avvocato italiano con lunga esperienza di contenzioso. Scrivi UNA sezione di un parere motivato "
        f"nell'interesse di {party}{position}, seguendo il PIANO già deciso: stessa parte, stessa questione "
        + ("centrale, stessi argomenti; delle sue norme e pronunce usa quelle che sostengono gli argomenti.\n\n"
           if rules else "centrale, stessi argomenti, stesse norme e pronunce.\n\n") +
        "REGOLE:\n"
        "- Scrivi solo il testo della sezione, senza il suo titolo. "
        + (sections_note or "Il parere ha sei sezioni (risposte ai "
           "quesiti; inquadramento e fatti; norme e giurisprudenza; applicazione al caso; profili processuali e "
           "termini; conclusioni)") + ": non ripetere ciò che spetta alle altre, sii essenziale.\n"
        "- Non concedere mai un elemento contestato dell'accusa o della pretesa: una posizione che lo presuppone "
        "va presentata solo in via subordinata.\n"
        + ("- Norme: cita solo quelle del PIANO o dei TESTI forniti, sempre con numero e codice o legge come vi "
           "compaiono, mai il numero da solo; nessun articolo a memoria; riporta tra virgolette solo frasi presenti "
           "nei testi forniti.\n" if _discipline() else
           "- Norme: cita solo quelle del PIANO o dei TESTI forniti, sempre con il codice (\"art. 1384 c.c.\", mai "
           "\"art. 1384\" da solo); riporta tra virgolette solo frasi presenti nei testi forniti.\n") +
        "- Pronunce: cita solo quelle del PIANO, con il riferimento esatto indicato; senza riferimento non "
        "richiamare orientamenti giurisprudenziali. I fatti di una pronuncia non sono fatti del caso.\n"
        "- Non aggiungere fatti, date, importi o atti che il caso non contiene: se un dato manca, indica che cosa "
        "va verificato.\n"
        + (writer_rules or (_WRITER_RULES.format(party=party) if rules else "")) +
        "- Parafrasa le norme con parole tue: il testo di una norma si riporta solo tra «…», mai fuori dalle virgolette né "
        "dentro una parentesi.\n"
        "- Prosa argomentata in paragrafi; elenchi puntati solo per prove da acquisire e attività da compiere.\n"
        f"- Scrivi in {language_display_name(lang)}."
    )


_WRITER_RULES = (
    "- Cita una norma solo dove sostiene un argomento: non serve citare tutte le norme del PIANO. Non scrivere mai "
    "che una norma è «non pertinente», «non applicabile» o «non rilevante»: una norma che non serve si omette. "
    "Escludere una norma su cui la controparte o l'accusa fonda la pretesa contro {party} è invece un argomento, "
    "da motivare sui fatti.\n"
    "- Onere della prova: indicalo solo quando la norma stessa pone una presunzione o ripartisce l'onere, "
    "riportandone le parole; mai una formula generica («l'onere della prova ricade su…») norma per norma.\n"
    "- Non attribuire a una parte condotte, conoscenze, intenzioni o qualità che il CASO non afferma (che sapesse, "
    "che abbia omesso, che abbia agito in malafede): se servono all'argomento, presentale come circostanze da "
    "accertare. I fatti di una pronuncia non sono mai fatti del caso.\n"
    "- Una pronuncia si cita solo dove i fatti del CASO ne integrano i presupposti, indicando quale fatto li "
    "integra; se nessun fatto del caso li integra, la pronuncia si omette.\n"
    "- Termini (giorni, mesi, anni): solo quelli del materiale fornito, mai a memoria.\n"
    "- Non proporre mai azioni, domande, denunce o querele contro {party} o contro chi è assistito insieme; un loro "
    "eventuale concorso di colpa è solo una difesa della controparte, da superare.\n"
    "- Una parte con più ruoli nei fatti risponde per ciascun ruolo: esaminali uno per uno.\n"
    "- L'art. 1051 c.c. non è mai il titolo di un acquisto: il passaggio coattivo si costituisce con sentenza e contro "
    "indennità; non scrivere che una servitù è stata acquistata in base a esso. L'uso di un passaggio non fa presumere "
    "una servitù: non parlare di «presunzione di servitù», indica come si è acquistata (titolo, usucapione, "
    "destinazione del padre di famiglia).\n"
    "- Le domande subordinate sono quelle di {party}: mai le difese della controparte, come un suo concorso di colpa.\n"
    "- La negatoria (art. 949 c.c.) è l'azione del proprietario che nega diritti altrui: chi vanta un diritto sul fondo "
    "altrui agisce in confessoria (art. 1079 c.c.), mai in negatoria.\n"
    "- L'art. 1223 c.c. misura il danno, non lo fonda: fuori dal contratto il fondamento è l'art. 2043 c.c. (o l'art. "
    "1079 c.c. per il titolare della servitù). Mai «inadempimento» senza un contratto.\n"
    "- L'art. 1159 c.c. richiede un titolo trascritto proveniente da chi non è proprietario: se i fatti non lo mostrano "
    "non si applica; non dirlo mai decorso.\n"
    "- Un'azione possessoria il cui termine è decorso è preclusa: mai proponibile, neppure in via subordinata.\n"
    "- La servitù dell'art. 1062 c.c. nasce dalla separazione dei due fondi con l'opera visibile già presente, non "
    "dall'uso.\n"
    "- I TERMINI CALCOLATI sono l'unica fonte: nessuna sezione dice soddisfatto un termine che risulta non verificabile "
    "o incerto, né il contrario.\n"
    "- Il parere è completo in sé: non rimandare ad approfondimenti successivi.\n"
    "- Una norma si nomina sempre con numero e codice: mai le parole del suo testo o della sua rubrica al posto del numero.\n"
    "- Tratta solo gli strumenti che servono: nessuna riga su uno strumento, un'azione o un adempimento che non si applica "
    "(né «non pertinente», né «non necessario», né «non praticabile»). Sulla mediazione scrivi solo ciò che è tra le "
    "CONCLUSIONI DI DIRITTO; se lì non c'è, non dire che è obbligatoria.\n"
    "- Un'azione il cui termine è decorso compare una sola volta, tra i Termini, come preclusa; mai tra gli strumenti.\n"
)


def _facts_list(analysis: Dict[str, Any]) -> str:
    return "\n".join(f"- {f.get('fatto', '')}" for f in analysis.get("fatti", []) if f.get("fatto"))


def _procedural_refs(plan: Dict[str, Any], analysis: Dict[str, Any]) -> List[tuple]:
    """The procedural section's own articles: the posture's method list and the procedural tools', not in the plan."""
    refs = _plan_refs(plan)
    posture = method_candidates.case_lists(analysis)[1]
    return [ref for ref in method_candidates.candidate_refs([], posture) + _tool_refs(analysis) if ref not in refs]


def _term_sources(analysis: Dict[str, Any]) -> List[Dict[str, str]]:
    """The legal terms' articles with their text (fact_signals.attach_term_sources), each once: [{"ref", "text"}]."""
    from . import fact_signals
    if not fact_signals.regimes_enabled():
        return []
    seen: Dict[str, str] = {}
    for term in (_signals(analysis) or {}).get("terms") or []:
        for source in term.get("sources") or []:
            seen.setdefault(source["ref"], source["text"])
    return [{"ref": ref, "text": text} for ref, text in seen.items()]


def anchor_catalogue(plan: Dict[str, Any], analysis: Dict[str, Any], article_rows: List[Dict[str, Any]],
                     references: Dict[str, List[Dict[str, Any]]]) -> Optional["anchored_writing.Catalogue"]:
    """ANCHORED_WRITING_ENABLED: the passages of the final plan and of the procedural section (anchored_writing.build);
    None with the flag off."""
    if not anchored_writing.enabled():
        return None
    return anchored_writing.build(plan, article_rows, references, _procedural_refs(plan, analysis),
                                  _term_sources(analysis))


def section_jobs(plan: Dict[str, Any], facts: str, request: str, analysis: Dict[str, Any],
                 article_rows: List[Dict[str, Any]], references: Dict[str, List[Dict[str, Any]]],
                 principles: str, keep_all: bool = False,
                 anchors: Optional["anchored_writing.Catalogue"] = None,
                 rendered: Optional[str] = None) -> List[Tuple[str, str, str, int]]:
    """(title, instruction, material, max_tokens) for each section, in order. With keep_all
    (PLAN_KEEP_APPLICABLE_ENABLED) sections 3 and 4 discuss every article of the plan by name. With anchors
    (ANCHORED_WRITING_ENABLED) every section gets its passages and the anchor rule, the procedural section its
    stage's procedure vocabulary. RULE_QUOTE: sections 3 and 4 get REGOLE DELLE NORME DEL PIANO (rule_quote). With
    rendered (RENDER_COMPUTED: the procedural section's text, rendered by template) there is no procedural job: sections
    1, 4 and 6 get the rendered text as read-only context, sections 3 and 4 the objections block."""
    party = plan["parte_assistita"] or "la parte assistita"
    case = f"CASO:\n{dg._case_text(facts, request)}"
    frame = analysis.get("frame") or {}
    tasks = "\n".join(f"- {t}" for t in frame.get("tasks") or []) or request.strip()
    facts_list = _facts_list(analysis)
    refs = _plan_refs(plan)
    norme_refs = _norme_refs(plan)       # NORME_SCOPE: section 3's own articles
    norme_only = (" Tratta solo le norme indicate nel PIANO di questa sezione: le altre norme del caso sono "
                  "discusse nell'applicazione al caso e nei profili processuali, qui non vanno elencate."
                  if plan.get(NORME_SCOPE_KEY) is not None else "")
    headings = _headings(article_rows)
    # Article text per section within a fixed total, so that a plan naming many articles still
    # leaves each call well inside the context with the case text (up to ~5,000 tokens).
    def per_article(total_chars: int, most: int) -> int:
        return max(300, min(most, total_chars // max(1, len(refs))))

    def texts(wanted: List[tuple], chars: int) -> str:
        return _with_headings(_article_texts(article_rows, wanted, chars), headings)
    posture = method_candidates.case_lists(analysis)[1]
    procedural = _procedural_refs(plan, analysis)
    rules = _writer_rules()
    qualified, tools = _qualifications(analysis), _tools(analysis)
    damages, roles = _damages(analysis), _roles(analysis)
    no_rulings = " In questa sezione non citare pronunce."

    def block(title: str, body: str) -> str:
        return f"{title}:\n{body}" if body.strip() else ""
    every = (" Esamina una per una TUTTE le norme del PIANO, nessuna esclusa, ciascuna indicata con numero e "
             + ("codice o legge esattamente come nel PIANO." if _discipline() else
                "codice o legge come nel PIANO (\"art. 95 Codice degli Appalti 2026\", \"art. 1784 c.c.\").")
             if keep_all and not rules else "")
    scope = _scope_block(plan)
    scoped = (" Ogni norma di NORME E LORO QUESTIONE va citata solo a proposito della questione indicata accanto, "
              "mai a sostegno di un'altra." if scope else "")
    every_issue = (f" Per la questione \"{OTHER_NORMS_TITLE}\", una o due frasi per ciascuna norma, con numero e "
                   "codice, su come si applica ai fatti." if keep_all and not rules else "")
    # WRITER_RULES_ENABLED: the facts labelled from the client's side; the burden only where the norm sets it.
    labels = (f"favorevole se fonda o sostiene la pretesa o la difesa di {party} (anche quando descrive un danno, "
              f"un difetto o un torto che {party} ha subito), sfavorevole se la controparte può usarlo contro "
              f"{party}, neutro altrimenti; e perché, in una frase." if rules else
              f"favorevole, sfavorevole o neutro per {party}, e perché, in una frase.")
    facts_part = (f"in un paragrafo, i fatti che decidono l'esito: quelli che fondano la pretesa o la difesa di "
                  f"{party} e quelli che la controparte può usare contro {party}, ciascuno con il perché in poche "
                  "parole. Non elencare né etichettare uno per uno i fatti del caso." if _review_fixes() else
                  f"ogni fatto del caso valutato dal punto di vista di {party}: {labels}")
    norms = ("Per le norme del PIANO su cui si fondano gli argomenti: che cosa stabilisce ciascuna, con parole tue "
             "(il testo solo tra «…»), e, solo se la norma stessa pone una presunzione o ripartisce l'onere della prova, quella "
             "regola con le sue parole" if rules else
             "Per ogni norma del PIANO: che cosa stabilisce, citando il testo fornito, e come si ripartisce l'onere "
             "della prova")
    criminal = _tools(analysis, criminal=True)
    present = {"termini": bool(_terms(analysis)), "strumenti": bool(tools), "profilo_penale": bool(criminal),
               "voci_di_danno": bool(damages)}
    blocks = ([(title, _criminal_what(analysis) if key == "profilo_penale" else what)
               for key, title, what in _SECTION5_BLOCKS if present[key]]
              if _qualifications_or_regimes(analysis) else [])
    regimes = ((f" Tieni conto delle {_QUALIFIED_TITLE} e dei regimi speciali che comportano." if qualified else "")
               + (" Tratta uno per uno, ciascuno in un breve paragrafo aperto dal suo titolo in grassetto, senza "
                  "ometterne nessuno: " + "; ".join(f"**{title}**: {what}" for title, what in blocks) + "."
                  if blocks else ""))
    applied = ((f" Per ogni parte di {_ROLES_TITLE}, la responsabilità per ciascun ruolo." if roles else "")
               + (f" Tieni conto delle {_DAMAGES_TITLE}." if damages else ""))

    fixed = block(_CONCLUSIONS_TITLE, f"{_conclusions(analysis)}\n{_CONCLUSIONS_RULE}" if _conclusions(analysis) else "")
    to_sue = block(_DEFENDANTS_TITLE, f"{_defendants(analysis)}\n{_DEFENDANTS_RULE}" if _defendants(analysis) else "")

    def material(*parts: str) -> str:
        # PROTECTED_PARAPHRASE_ENABLED: a section given a protected passage gets the rule with it.
        return protected_sources.with_rule("\n\n".join(p for p in parts if p))
    fixes = _review_fixes()
    questions = explicit_questions(request, facts) if fixes else []
    asked = ("\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1)))
    one_each = (" Ai QUESITI ESPLICITI rispondi uno per uno, nell'ordine, ciascuno in un paragrafo che si apre "
                "riprendendo il quesito; nessuno va omesso, e per una richiesta di sanzioni o penalità contro chi non "
                "adempie indica le misure coercitive degli strumenti processuali." if questions else "")
    adverse = (f" Una pronuncia tra quelle «della controparte» si presenta solo come argomento della controparte, "
               f"seguito dalla replica di {party}; mai a sostegno di {party}."
               if any(q.get("pronunce_contrarie") for q in plan["questioni"]) else "")
    # ROLE_FIT: a norm marked as the counterparty's
    adverse += (f" Una norma segnata «{role_fit.ADVERSE_MARK}» si presenta solo come argomento della controparte, "
                f"seguito dalla replica di {party}; mai a sostegno di {party}." if plan.get("norme_contrarie") else "")
    remedies = _remedies_block(plan)          # REVIEW_FIXES_ENABLED: the main remedy article's remedies, each
    # RULE_QUOTE: each plan article's operative comma and its requisites, with the rule (section 3: its own articles)
    rules_norme = rule_quote.material_block(plan, article_rows, analysis, norme_refs) if rule_quote.enabled() else ""
    rules_all = rule_quote.material_block(plan, article_rows, analysis) if rule_quote.enabled() else ""
    quote = (f" Per ogni norma segui {rule_quote.TITLE}: riporta il comma tra «…» con il suo numero o resta dentro le sue "
             "parole; per ogni requisito numerato indica se è «provato» o «da provare» sui fatti.")
    # RENDER_COMPUTED: the action gate's articles only as the other side's; the rendered procedural section read only
    objections = render_computed.objections_block(plan) if rendered is not None else ""
    objected = (f" Le norme di {render_computed.OBJECTIONS_TITLE} solo come argomento della controparte, con la replica."
                if objections else "")
    read_only = render_computed.read_only_block(rendered or "")
    evidence = (" Le prove da acquisire di una questione che coincidono con quelle di una questione precedente non si "
                "ripetono: si rinvia a quella." if fixes else "")

    jobs = [
        (SECTION_TITLES[0],
         "SEZIONE: Risposte ai quesiti. Una risposta diretta e motivata a ciascun quesito, nell'ordine in cui è "
         "posto, a partire dalle risposte del PIANO quando ci sono, altrimenti dalle sue questioni; se la "
         "questione centrale non è tra i quesiti, rispondi prima a essa. Ogni risposta in due-quattro frasi, con "
         "la norma su cui si fonda." + one_each + scoped + no_rulings,
         material(case, fixed, block("QUESITI", tasks), block(_QUESTIONS_TITLE, asked),
                  block(_TOOLS_TITLE, tools if questions and any(fact_signals_asks(q) for q in questions) else ""),
                  _plan_text(plan, answers=True, rulings=False), scope, remedies, read_only),
         600 + 150 * max(0, len(questions) - 1)),
        (SECTION_TITLES[1],
         "SEZIONE: Inquadramento e questione centrale. La qualificazione giuridica della vicenda (la norma "
         "principale e quelle alternative o subordinate, e perché), la questione centrale e perché decide "
         f"l'esito; poi {facts_part} Non illustrare le "
         "norme nel dettaglio (sezione 3) né sviluppare gli argomenti (sezione 4)." + scoped + no_rulings,
         material(case, fixed, block("FATTI DEL CASO", facts_list), block(_QUALIFIED_TITLE, qualified),
                  _plan_text(plan, rulings=False), scope,
                  block("TESTI DELLE NORME", texts(refs, per_article(8000, 700)))), 1000),
        (SECTION_TITLES[2],
         f"SEZIONE: Norme e giurisprudenza. {norms}; per ogni pronuncia del PIANO: il principio che "
         "afferma, con il suo riferimento, e che cosa comporta per questo caso, senza svolgere gli argomenti "
         "(sezione 4). Solo norme e pronunce del PIANO; tra virgolette solo frasi dei testi e dei passi forniti."
         + every + scoped + adverse + norme_only + (quote if rules_norme else "") + objected,
         material(case, fixed, _plan_text(_scoped_plan(plan)), scope, rules_norme, objections,
                  block("TESTI DELLE NORME", texts(norme_refs, per_article(15000, dg._ARTICLE_CHARS))),
                  block("PASSI DELLE PRONUNCE", _ruling_texts(plan, references, 1200)),
                  block("PRINCIPI DALLA GIURISPRUDENZA", principles)), 1600 if keep_all else 1200),
        (SECTION_TITLES[3],
         "SEZIONE: Applicazione al caso. Per ogni questione del PIANO, nell'ordine, con un sottotitolo: "
         f"l'argomento di {party} applicato ai fatti del caso, con le norme e le pronunce del piano; le prove da "
         "acquisire; l'obiezione più forte della controparte e la replica. Poi le tesi subordinate del PIANO. "
         "Prosa, senza ripetere per ogni questione lo stesso schema a voci; tra virgolette solo frasi dei testi e "
         "dei passi forniti." + every_issue + scoped + applied + adverse + evidence + (quote if rules_all else "")
         + objected,
         material(case, fixed, to_sue, block("FATTI DEL CASO", facts_list), block(_QUALIFIED_TITLE, qualified),
                  block(_ROLES_TITLE, roles), block(_DAMAGES_TITLE, damages),
                  _plan_text(plan, issues="full", subordinate=True), scope, remedies, rules_all, objections, read_only,
                  block("TESTI DELLE NORME", texts(refs, per_article(9000, 600))),
                  block("PASSI DELLE PRONUNCE", _ruling_texts(plan, references, 600))), 2200 if keep_all else 1800),
        (SECTION_TITLES[4],
         "SEZIONE: Profili processuali e termini. La fase in cui si trova la vicenda e che cosa consente di fare "
         f"a {party}: gli atti da compiere e i loro termini, con l'articolo che li prevede; a chi rivolgersi o chi "
         "convenire; le voci di danno o le domande da proporre, anche in via subordinata. Usa le VERIFICHE solo "
         "se i fatti del caso le rendono pertinenti, dicendo perché, e solo quelle che la legge consente a "
         f"{party} nella sua posizione; un termine che dipende da una data che il caso non indica è una verifica "
         "da fare." + (f" {_terms_rule()}" if _terms(analysis) else "") + regimes + no_rulings,
         material(case, fixed, to_sue, block("FASE", frame.get("procedural_posture") or ""), block(_QUALIFIED_TITLE, qualified),
                  block(_TERMS_TITLE, _terms(analysis)), block(_TERM_TEXTS_TITLE, _term_texts(analysis)),
                  _plan_text(plan, issues="", subordinate=True, rulings=False), remedies,
                  block(_TOOLS_TITLE, tools), block(_CRIMINAL_TITLE, criminal), block(_DAMAGES_TITLE, damages),
                  block("VERIFICHE PER QUESTA FASE E MATERIA",
                        "\n".join(f"- {c}" for c in _checks(analysis, article_rows, facts, request))),
                  block("TESTI DELLE NORME PROCESSUALI", texts(procedural, 500))), 900 + 250 * len(blocks)),
        (SECTION_TITLES[5],
         "SEZIONE: Conclusioni. In due o tre paragrafi: gli argomenti in ordine di forza, come nel PIANO; una "
         f"stima motivata del rischio, tenendo conto dei punti deboli; la condotta consigliata a {party}."
         + no_rulings,
         material(fixed, _plan_text(plan, subordinate=True, weak=True, rulings=False), read_only), 500),
    ]
    if rendered is not None:            # RENDER_COMPUTED: the procedural section is rendered, not written
        jobs = [job for job in jobs if job[0] != SECTION_TITLES[4]]
    if anchors is not None:
        jobs = _with_anchors(jobs, anchors, posture)
    jobs = [_with_budget(job) for job in jobs] if rules else jobs
    for title, _, material_text, _ in jobs:        # COMMA_MATERIAL_ENABLED: the commi each section was given
        comma_material.log_material(f"section {title}", material_text, article_rows)
    return jobs


def _with_anchors(jobs: List[Tuple[str, str, str, int]], anchors: "anchored_writing.Catalogue",
                  posture: str) -> List[Tuple[str, str, str, int]]:
    """ANCHORED_WRITING_ENABLED: each section's material with its passages (the plan's; for the procedural section the
    procedural ones too, and its stage's procedure vocabulary with its rule), its instruction with the anchor rule."""
    out = []
    for title, instruction, material, tokens in jobs:
        ids = list(anchors.plan_ids)
        extra = ""
        if title == SECTION_TITLES[4]:
            ids = anchors.procedural_ids + ids
            vocabulary = practical_checklist.procedure_vocabulary(posture)
            if vocabulary:
                extra = f"{practical_checklist.VOCABULARY_TITLE}:\n" + "\n".join(f"- {v}" for v in vocabulary)
                instruction += " " + practical_checklist.VOCABULARY_RULE
        if "non citare pronunce" in instruction:           # a section without rulings gets no ruling passage
            ids = [i for i in ids if i not in anchors.rulings(ids)]
        anchors.section_ids[title] = list(dict.fromkeys(ids))
        block = anchors.block(ids)
        material = "\n\n".join(p for p in (material, extra, block) if p)
        out.append((title, instruction + (anchored_writing.ANCHOR_RULE if block else ""), material, tokens + 150))
    return out


def fact_signals_asks(question: str) -> bool:
    from . import fact_signals
    return bool(fact_signals.asks_sanction(question))


def _with_budget(job: Tuple[str, str, str, int]) -> Tuple[str, str, str, int]:
    """WRITER_RULES_ENABLED: the section's words, said in its instruction, within more tokens than the old budget
    (Italian runs ~1.6 tokens a word), so it is never cut; the conclusions get their own."""
    title, instruction, material, tokens = job
    tokens, words = (900, 300) if title == SECTION_TITLES[5] else (int(tokens * 1.3), int(tokens * 0.5))
    return title, f"{instruction} Al massimo {words} parole; chiudi sempre l'ultima frase.", material, tokens


# --- T3: citations only from the section's own material (_discipline) ----------------------------
# Run 1302: articles written from memory, in no section's material (1384 c.c. in 4 answers, 156/159/256
# c.p.p.). After each section is written, a citation of an article its material does not hold and the
# case does not name is taken out with its whole sentence. Logged as case_analysis_citation_discipline.
# The live answer of 26 Oct: a sentence rewritten without its citation left fragments ("Inoltre, conferma che
# ...", "... un risarcimento per il danno subito, in conformità."), and a "Questa disposizione è pertinente ..."
# whose article had gone. No rewriting any more: the sentence goes, then drop_dangling.
# A period after these words does not end a sentence ("art. 5", "Cass. pen., sez. V", "n. 190/2026").
_ABBREVIATIONS = {"art", "artt", "sez", "sezz", "sent", "ord", "n", "nn", "co", "lett", "cass", "pen", "civ", "cfr",
                  "pag", "pp", "d", "lgs", "dl", "dpr", "l", "ss", "es", "un", "cost", "trib", "app", "rv", "cit",
                  "vol", "proc", "cod", "reg", "dott", "avv", "sig", "prof", "cd", "rel"}


def _ends_sentence(text: str, i: int) -> bool:
    """The punctuation at text[i] ends a sentence: not after an abbreviation, and followed by the end, a
    line break, or a word that does not start in lower case or with a digit."""
    word = re.search(r"(\w+)$", text[:i])
    if text[i] == "." and word and word.group(1).lower() in _ABBREVIATIONS:
        return False
    rest = text[i + 1:]
    nxt = rest.lstrip(" \t")
    if not rest or not rest[0].isspace():
        return not rest
    return not nxt or nxt[0] == "\n" or not (nxt[0].islower() or nxt[0].isdigit())


# v60f (X4): the writer's paragraph mark «[N]» at a line's start -- never part of the sentence after it
_PARAGRAPH_MARK_RE = re.compile(r"\[\d{1,2}\][ \t]*")


def _sentence_span(text: str, start: int, end: int) -> Tuple[int, int]:
    """The sentence around text[start:end], within its line."""
    s = start
    while s > 0 and text[s - 1] != "\n" and not (text[s - 1] in ".!?" and _ends_sentence(text, s - 1)):
        s -= 1
    while s < start and text[s].isspace():
        s += 1
    mark = _PARAGRAPH_MARK_RE.match(text, s, start)     # v60f (X4): a paragraph's «[N]» is no sentence's
    if mark and (s == 0 or text[s - 1] == "\n"):
        s = mark.end()
    e = end
    while e < len(text) and text[e] != "\n" and not (text[e] in ".!?" and _ends_sentence(text, e)):
        e += 1
    return s, min(len(text), e + 1) if e < len(text) and text[e] in ".!?" else e


def _material_refs(material: str, documents: List[Dict[str, Any]]) -> List[tuple]:
    """Every article a text cites with its code or law, as the report counts them (dg._cited_articles)."""
    return dg._cited_articles(material or "", set(), documents)


def _code_key(code: str) -> str:
    """A code by its sigla, also when written as its document's name ("Codice Civile 2026" -> "c.c.")."""
    low = str(code or "").lower()           # v60e: «Codice della Strada 2026» as «Codice della strada»
    return next((abbr for abbr, prefix in dg._CODE_PREFIXES.items() if low.startswith(prefix.lower())), code)


def _same_ref(a: tuple, b: tuple) -> bool:
    """_same_article, a code named by its document read as its sigla."""
    return _same_article((a[0], _code_key(a[1])), (b[0], _code_key(b[1])))


# Not a bare number, though no act of the corpus was read after it: another law, named or not ("art. 3 l. 241/1990" when
# no document carries those numbers), or a private act ("art. 5 del contratto", never a citation of law).
_COMMA_LEAD_RE = re.compile(r"\s*,?\s*(?:(?:co\.|comma|commi)\s*\d+(?:\s*,?\s*(?:lett\.|lettera|n\.)\s*\w+)?\s*,?\s*)?",
                            re.IGNORECASE)
_SAME_ACT_RE = re.compile(r"(?:del(?:la|lo)?\s+|dell['’]\s*)?(?:medesim|stess|predett|citat|suddett|codice\b)", re.IGNORECASE)
_LAW_ACT_RE = re.compile(r"(?:del(?:la|lo)?\s+|dell['’]\s*)?(?:l\.|legge\b|d\.?\s*lgs|decreto\b|d\.\s*l\.|d\.?\s*p\.?\s*r\.?|r\.\s*d\.|"
                         r"t\.\s*u\.|testo unico|tuel\b|cost\.|costituzione|cedu\b|tfue\b|regolamento\s+\(?(?:ue|ce)|direttiva)",
                         re.IGNORECASE)
_OTHER_ACT_RE = re.compile(r"(?:del|della|dello|dell['’]|dei|degli|delle)\s*[^\W\d]", re.IGNORECASE)
# A code mentioned anywhere, even with no article before it ("il codice penale punisce..."); with its dots, as "cc" may
# be anything.
_CODE_MENTION_RE = re.compile(
    r"(?<![\w.])(?:c\.\s?p\.\s?c\.|c\.\s?p\.\s?p\.|c\.\s?p\.\s?a\.|c\.\s?d\.\s?s\.|c\.\s?c\.|c\.\s?p\.|cod\.\s*civ\.|cod\.\s*pen\.|"
    r"codice (?:di procedura civile|di procedura penale|del processo amministrativo|della strada|civile|penale))",
    re.IGNORECASE)
_QUOTES = (("«", "»"), ("“", "”"))


def _quote_span(text: str, at: int) -> Optional[Tuple[int, int]]:
    """(start, end) of the quotation holding position `at`, inside its paragraph, or None."""
    start = max(text.rfind("\n\n", 0, at), 0)
    stop = text.find("\n\n", at)
    stop = len(text) if stop < 0 else stop
    for opening, closing in _QUOTES:
        i = text.rfind(opening, start, at)
        if i >= 0 and text.rfind(closing, i, at) < 0:
            j = text.find(closing, at, stop)
            return i, stop if j < 0 else j + 1
    straight = [m.start() for m in re.finditer('"', text[start:stop])]
    before = [start + q for q in straight if start + q < at]
    after = [start + q for q in straight if start + q >= at]
    return (before[-1], after[0] + 1) if len(before) % 2 and after else None


def _near_code(text: str, start: int, end: int, explicit: List[Tuple[int, int, str]]) -> Optional[str]:
    """The code of the explicit reference nearest to a bare number, in its sentence or in the quotation holding it (with
    the sentences the quotation opens and closes in: "l'art. 644 c.p. punisce «... dall'articolo 643 ...»")."""
    low, high = _sentence_span(text, start, end)
    quote = _quote_span(text, start)
    if quote:
        low, high = min(low, _sentence_span(text, quote[0], quote[0] + 1)[0]), max(high, _sentence_span(text, quote[1] - 1, quote[1])[1])
    near = [(max(a - end, start - b, 0), code) for a, b, code in explicit if a < high and b > low and not (a >= start and b <= end)]
    return min(near, key=lambda item: item[0])[1] if near else None


def read_citations(text: str, documents: List[Dict[str, Any]], known: Optional[set] = None
                   ) -> List[Tuple[int, int, List[str], List[Optional[tuple]], List[str]]]:
    """(start, end, numbers, their refs, how each was resolved: "explicit", "nearest", "material" or "unresolved") of each
    citation of a law in text, the one reading of a citation. A number
    takes the code or law after it (dg._code_after, the words really naming a document); a bare one the nearest explicit
    reference in its sentence or quotation (_near_code), unless that article is not among `known` when exactly one
    known article has the number; else that one known article. None: not resolved, never a guess -- several or none
    known, or a law named but not found. An article of a private act ("art. 5 del contratto") is no citation at all."""
    runs, explicit = [], []
    for m in dg._ARTICLE_RUN_RE.finditer(text or ""):
        tail = text[m.end():m.end() + 120]
        code = dg._code_after(tail, documents)
        rest = tail[_COMMA_LEAD_RE.match(tail).end():]
        kind = "explicit" if code else "bare" if _SAME_ACT_RE.match(rest) else "law" if _LAW_ACT_RE.match(rest) \
            else "other" if _OTHER_ACT_RE.match(rest) else "bare"
        if kind == "other":
            continue
        runs.append((m, kind, code))
        if code:
            explicit.append((m.start(), m.end(), code))
    explicit += [(m.start(), m.end(), dg._code_of(m.group(0)) or next(abbr for name, abbr in dg._CODE_NAMES
                                                                      if m.group(0).lower() == name))
                 for m in _CODE_MENTION_RE.finditer(text or "")]
    out = []
    lists = {start: (end, refs) for start, end, refs in dg._article_lists(text or "")}
    for m, kind, code in runs:
        if m.start() in lists:      # "artt. 55 c.p. e 52 c.p.", "artt. 1176, comma 2, e 1218 c.c.": each its own code
            end, refs = lists[m.start()]
            out.append((m.start(), end, [n for n, _ in refs], list(refs), ["explicit"] * len(refs)))
            continue
        numbers = [dg._article_num(n) for n in dg._RUN_NUMBER_RE.findall(m.group(1))]
        near = _near_code(text, m.start(), m.end(), explicit) if kind == "bare" else None
        refs: List[Optional[tuple]] = []
        hows: List[str] = []
        for n in numbers:
            same = [k for k in known or () if k[0] == n and (kind != "law" or k[1] not in dg._CODE_PREFIXES)]
            if code:
                refs.append((n, code))
                hows.append("explicit")
            elif near and (not same or len(same) > 1 or any(_same_ref((n, near), k) for k in same)):
                refs.append((n, near))
                hows.append("nearest")
            else:
                refs.append(same[0] if len(same) == 1 else None)
                hows.append("material" if len(same) == 1 else "unresolved")
        out.append((m.start(), m.end(), numbers, refs, hows))
    return out


def _citations(text: str, documents: List[Dict[str, Any]], known: Optional[set] = None) -> List[Tuple[int, int, List[tuple]]]:
    """(start, end, refs) of each citation in text with the numbers read_citations resolved."""
    out = []
    for start, end, _, refs, _ in read_citations(text, documents, known):
        if any(refs):
            out.append((start, end, [r for r in refs if r]))
    return out


def _unsupported(text: str, allowed: List[tuple], documents: List[Dict[str, Any]]) -> List[Tuple[int, int, List[tuple]]]:
    """(start, end, refs) of each citation in text naming an article not among allowed."""
    out = []
    for start, end, refs in _citations(text, documents):
        bad = [r for r in refs if not any(_same_ref(r, a) for a in allowed)]
        if bad:
            out.append((start, end, bad))
    return out


def _citing(text: str, targets: List[tuple], documents: List[Dict[str, Any]],
            known: Optional[set] = None) -> List[Tuple[int, int, List[tuple]]]:
    """(start, end, refs) of each citation in text naming one of targets."""
    out = []
    for start, end, refs in _citations(text, documents, known):
        hit = [r for r in refs if any(_same_ref(r, t) for t in targets)]
        if hit:
            out.append((start, end, hit))
    return out


def _cite_only_material(title: str, text: str, material: str, named: List[tuple],
                        documents: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """(text, log): the section without the citations of articles its material does not hold and the
    case does not name: each sentence holding one dropped, then the fragments left without their norm
    (drop_dangling)."""
    allowed = _material_refs(material, documents) + list(named)
    text, log = _rewrite_out(title, text, _unsupported(text, allowed, documents))
    return _cleaned_after(title, text, log)


def remove_citations(text: str, targets: List[tuple], documents: List[Dict[str, Any]], known: Optional[set] = None,
                     title: str = "parere") -> Tuple[str, List[Dict[str, Any]]]:
    """(text, log): text without any citation of targets (FLAGGED_REMOVE_ENABLED): a heading line naming
    one ("**Art. 757 c.c. ...**") goes with its paragraph, then each sentence citing one is dropped whole
    (_rewrite_out), then the fragments left without their norm (drop_dangling)."""
    if not targets:
        return text, []
    text, log = _without_article_headings(text, targets, documents, known, title)
    text, more = _rewrite_out(title, text, _citing(text, targets, documents, known))
    return _cleaned_after(title, text, log + more)


def _cleaned_after(title: str, text: str, log: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """After a removal: the fragments it left without their norm dropped too (drop_dangling), logged with it."""
    if not log:
        return text, log
    text, dangling = drop_dangling(text, title)
    return text, log + dangling


_HEADING_LINE_RE = re.compile(r"\s*(#{1,6}\s+.+|\*\*[^*\n]+\*\*|__[^_\n]+__)(.*)$")


def _without_article_headings(text: str, targets: List[tuple], documents: List[Dict[str, Any]],
                              known: Optional[set], title: str) -> Tuple[str, List[Dict[str, Any]]]:
    """text without the heading lines that name one of targets: a heading alone on its line goes with the
    paragraph under it; a bold lead-in followed by text on the same line goes with that line."""
    lines, out, log, i = text.split("\n"), [], [], 0
    while i < len(lines):
        m = _HEADING_LINE_RE.match(lines[i])
        hit = _citing(m.group(1), targets, documents, known) if m else []
        if not hit:
            out.append(lines[i])
            i += 1
            continue
        removed = [lines[i]]
        i += 1
        if not m.group(2).strip(" .:;–-"):
            while i < len(lines) and not lines[i].strip():
                i += 1
            while i < len(lines) and lines[i].strip() and not _HEADING_LINE_RE.match(lines[i]):
                removed.append(lines[i])
                i += 1
        log.append({"section": title, "removed": [f"{n} {c}" for n, c in dict.fromkeys(r for _, _, rs in hit for r in rs)],
                    "how": "heading and paragraph dropped", "sentence": " ".join(removed).strip()[:300], "now": ""})
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)), log


def _rewrite_out(title: str, text: str, found: List[Tuple[int, int, List[tuple]]]) -> Tuple[str, List[Dict[str, Any]]]:
    """(text, log): each sentence holding one of the citations found dropped whole (the live answer of 26 Oct: a
    rewriting without the citation left fragments)."""
    if not found:
        return text, []
    spans: List[List[Any]] = []
    for start, end, refs in found:
        s, e = _sentence_span(text, start, end)
        if spans and s < spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], e)
            spans[-1][2] += [r for r in refs if r not in spans[-1][2]]
        else:
            spans.append([s, e, list(refs)])
    log, out, pos = [], [], 0
    for s, e, refs in spans:
        out.append(text[pos:s])
        through = max(e, with_orphans(text, s, e))
        pos = through
        if s == 0 or text[s - 1] == "\n":
            while pos < len(text) and text[pos] in " \t":      # a dropped first sentence: no leading blank
                pos += 1
        log.append({"section": title, "removed": [f"{n} {c}" for n, c in refs], "how": "sentence dropped",
                    "sentence": text[s:e].strip()[:300], "now": ""})
        if through > e:
            log.append({"section": title, "kind": "dangling", "how": "sentence dropped", "sentence": text[e:through].strip()[:300]})
    return _tidy("".join(out) + text[pos:]), log


def _tidy(text: str) -> str:
    """Blank runs and empty list items left by a removal collapsed."""
    text = re.sub(r"[ \t]+\n", "\n", re.sub(r"(?m)^[ \t]*(?:[-•*]|\d+[.)])[ \t]*$\n?", "", text))
    return re.sub(r"(?<=\S)[ \t]{2,}", " ", re.sub(r"\n{3,}", "\n\n", text)).strip()


# --- drop_dangling: the fragments a removal leaves ------------------------------------------------------------
# A sentence pointing back to a norm or a ruling ("Questa disposizione è pertinente ...", "Tale principio ..."),
# or with no subject at all ("Inoltre, conferma che ..."), that has no norm of its own and nothing before it, under
# the same heading, naming one; and a tail with nothing after its preposition ("..., in conformità.").
_NORM_MENTION_RE = re.compile(
    r"\bart(?:icol[oi]|t)?\.?\s*\d+|\bn\.\s*\d+/\d{2,4}\b|\bcass(?:azione)?\b|\bcorte\b|\bd\.\s?lgs\b|\blegge\b|"
    r"\bcodice\b|\bc\.\s?[cp]\.|\bcost\.", re.IGNORECASE)
_BACK_REFERENCE_RE = re.compile(
    r"^\W*(?:(?:inoltre|infatti|pertanto|quindi|dunque|anche),?\s+)?(?:quest[aoei]|tal[ei]|dett[aoei]|la stessa|lo stesso|"
    r"la medesima|il medesimo)\s+(?:disposizion|norm|articol|principi|pronunc|sentenz|ordinanz|orientament|"
    r"previsione|regola|massim)\w*", re.IGNORECASE)
_NO_SUBJECT_RE = re.compile(
    r"^\W*(?:(?:inoltre|infatti|altresì|anche|in via (?:subordinata|principale|alternativa)|pertanto|quindi|tuttavia|"
    r"peraltro),?\s+)?(?:conferma|stabilisce|prevede|dispone|afferma|ribadisce|precisa|"
    r"chiarisce|sancisce)\s+(?:che|il|la|l'|lo|i|gli|le|un|una|altresì|inoltre)\b"
    # v38: what is left of "L'azione negatoria (art. 949) potrebbe essere esercitata da Leonardo per ...", and of a clause
    # cut after its verb. With its subject after the verb ("può essere proposta domanda") it stands.
    r"|^\W*(?:(?:inoltre|infatti|altresì|anche|in via (?:subordinata|principale|alternativa)|pertanto|quindi|tuttavia|"
    r"peraltro),?\s+)?(?:potrebbe(?:ro)?|può|possono|dovrebbe(?:ro)?|deve|devono|dovrà|sarà|sarebbe)\s+(?:(?:anche\s+)?essere\s+)?"
    r"(?:(?:esercitat|propost|invocat|azionat|promoss|intentat|richiest|dedott|eccepit|applicat|fatt)[aoie]\b"
    r"(?=\s+(?:da|per|contro|entro|nei|nel|solo|soltanto|anche|in)\b|\s*[,.;]|\s*$)|"
    r"(?:applicarsi|operare|invocarsi|rilevare|trovare applicazione)\s*[.;]?\s*$)", re.IGNORECASE)
# A sentence that leans on the one before it: when that one goes, it goes too.
_LEANS_RE = re.compile(_BACK_REFERENCE_RE.pattern + r"|^\W*ciò\b", re.IGNORECASE)
_EMPTY_TAIL_RE = re.compile(
    r"(?:,\s*|\s+)(?:in conformità|ai sensi|in virtù|come previsto|secondo quanto (?:previsto|disposto|stabilito)|"
    r"a norma|in base|come stabilito|conformemente)(?:\s+(?:a|al|alla|allo|ai|alle|all'|dal|dalla|dall'|del|della|dell'|"
    r"di|degli|delle|dei))?\s*(?=[.;:!?]\s*$)", re.IGNORECASE)


def with_orphans(text: str, start: int, end: int) -> int:
    """The end of a removal from `end`, past the sentences it orphans: each next one, in the paragraph or opening the
    next, that starts on a back-reference ("Questo principio si applica ...") with no norm of its own."""
    while True:
        pos = end
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text) or _HEADING_LINE_RE.match(text[pos:text.find("\n", pos) if "\n" in text[pos:] else len(text)]):
            return end
        _, stop = _sentence_span(text, pos, pos)
        sentence = text[pos:stop]
        if not _LEANS_RE.search(sentence) or _NORM_MENTION_RE.search(sentence):
            return end
        end = stop


def _dangling(sentence: str, context: bool) -> bool:
    if _NORM_MENTION_RE.search(sentence):
        return False
    return bool(_NO_SUBJECT_RE.search(sentence)) or (not context and bool(_BACK_REFERENCE_RE.search(sentence)))


def drop_dangling(text: str, title: str = "") -> Tuple[str, List[Dict[str, Any]]]:
    """(text, log): the sentences pointing back to a norm or a ruling with none named in them or before them under
    the same heading, and those with no subject ("Inoltre, conferma che ..."), dropped; an empty tail ("...,
    in conformità.") cut from its sentence. A paragraph left with nothing goes too."""
    if not text:
        return text, []
    out_lines, log, context = [], [], False
    for line in text.split("\n"):
        m = _HEADING_LINE_RE.match(line)
        if m and not m.group(2).strip():
            context = bool(_NORM_MENTION_RE.search(line))
            out_lines.append(line)
            continue
        spans = _sentence_spans(line)
        if not spans:
            out_lines.append(line)
            continue
        keep: List[str] = []
        changed = False
        for s, e in spans:
            sentence = line[s:e].strip()
            if _dangling(sentence, context):
                log.append({"section": title, "kind": "dangling", "how": "sentence dropped", "sentence": sentence[:300]})
                changed = True
                continue
            cut = _EMPTY_TAIL_RE.sub("", sentence)
            if cut != sentence:
                log.append({"section": title, "kind": "dangling", "how": "empty tail cut", "sentence": sentence[:300],
                            "now": cut[:300]})
                sentence, changed = cut, True
            context = context or bool(_NORM_MENTION_RE.search(sentence))
            keep.append(sentence)
        if not changed:
            out_lines.append(line)                      # untouched: kept exactly as written
        elif keep:
            out_lines.append(line[:spans[0][0]] + " ".join(keep))
    return (_tidy("\n".join(out_lines)) if log else text), log


# FACT_SIGNALS_ENABLED: the terms computed in code from the case's dates (fact_signals.compute_terms), given
# to the plan and to the procedural section as facts, so an expired action is never proposed as open.
_TERMS_TITLE = "TERMINI CALCOLATI DALLE DATE DEL CASO"
_TERMS_RULE = ("Sono fatti, calcolati dalle date del caso: un'azione il cui termine è DECORSO non va proposta come "
               "praticabile (indicala come preclusa e proponi le alternative); un termine INCERTO va indicato come "
               "verifica da fare; un'azione non soggetta a prescrizione non ha termine.")


# FACT_REGIMES_ENABLED: every applicable term with its article, also when the case gives no dates (DA VERIFICARE).
_TERMS_RULE_REGIMES = ("Indica sempre ciascuno di questi termini con la sua durata, da quando decorre e la norma "
                       "tra parentesi quadre, anche quando il caso non dà le date: un termine DA VERIFICARE o "
                       "INCERTO è una verifica da fare (quale data accertare); un'azione il cui termine è DECORSO non "
                       "va proposta come praticabile (indicala come preclusa e proponi le alternative). Nessun altro "
                       "termine che non sia qui o nei testi forniti; un'azione non soggetta a prescrizione non ha termine. "
                       "Un termine con «inizio del possesso da accertare» non è maturato: non scrivere mai che il possesso "
                       "dura da oltre vent'anni né che l'usucapione è compiuta; indicalo come verifica da fare (quando è "
                       "iniziato il possesso).")
_TERM_TEXTS_TITLE = "TESTI DELLE NORME SUI TERMINI"
_QUALIFIED_TITLE = "QUALIFICHE DELLE PARTI"
_QUALIFIED_RULE = ("Sono dedotte dai fatti: considera i regimi speciali che comportano (per il consumatore, la "
                   "garanzia legale di conformità e il foro del consumatore del Codice del Consumo).")
_TOOLS_TITLE = "STRUMENTI PROCESSUALI INDICATI DAI FATTI"
_TOOLS_RULE = "Valutali per la fase del caso: i fatti indicati accanto li rendono pertinenti."


def _signals(analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return (analysis or {}).get("fact_signals")


_CONCLUSIONS_TITLE = "CONCLUSIONI DI DIRITTO ACCERTATE DAI FATTI"
_CONCLUSIONS_RULE = "Sono punti fermi come il testo delle norme: riportali dove servono e non scrivere mai il contrario."
_DEFENDANTS_TITLE = "PARTI DA CONVENIRE"
_DEFENDANTS_RULE = "Sono ricavate dai fatti: indicale tutte per nome, mai come «eventuali» proprietari o terzi."


def _conclusions(analysis: Dict[str, Any]) -> str:
    from . import fact_signals
    return fact_signals.conclusions_block(_signals(analysis), analysis)


def _defendants(analysis: Dict[str, Any]) -> str:
    from . import fact_signals
    return fact_signals.parties_block(_signals(analysis))


def _qualifications_or_regimes(analysis: Dict[str, Any]) -> bool:
    """FACT_REGIMES_ENABLED ran on the case (its signals carry its keys)."""
    from . import fact_signals
    return fact_signals.regimes_enabled() and "tools" in (_signals(analysis) or {})


def _terms(analysis: Dict[str, Any]) -> str:
    from . import fact_signals
    return fact_signals.terms_block(_signals(analysis), refs=fact_signals.regimes_enabled())


def _terms_rule() -> str:
    from . import fact_signals
    return _TERMS_RULE_REGIMES if fact_signals.regimes_enabled() else _TERMS_RULE


def _term_texts(analysis: Dict[str, Any]) -> str:
    """The terms' articles with their text (fact_signals.attach_term_sources), each once."""
    from . import fact_signals
    if not fact_signals.regimes_enabled():
        return ""
    seen: Dict[str, str] = {}
    for term in (_signals(analysis) or {}).get("terms") or []:
        for source in term.get("sources") or []:
            seen.setdefault(source["ref"], source["text"])
    return "\n\n".join(f"[{ref}]\n{text}" for ref, text in seen.items())


def _qualifications(analysis: Dict[str, Any]) -> str:
    from . import fact_signals
    return fact_signals.qualifications_block(_signals(analysis)) if fact_signals.regimes_enabled() else ""


def _tools(analysis: Dict[str, Any], criminal: bool = False) -> str:
    from . import fact_signals
    return fact_signals.tools_block(_signals(analysis), criminal) if fact_signals.regimes_enabled() else ""


_CRIMINAL_TITLE = "PROFILO PENALE DEL DANNO ALLA PERSONA"
# FACT_REGIMES_ENABLED (run 20261026-0841: terms stated 8/22, tools cited 6/33, criminal route 2/4): the procedural
# section treats each block it is given in its own short paragraph, under its own bold title (_SECTION5_BLOCKS);
# one that it leaves out gets the section written once more (write_sections), and what was offered and what
# written is logged as case_analysis_material_use.
_SECTION5_BLOCKS = (
    ("termini", "Termini", "ogni termine di " + "TERMINI CALCOLATI DALLE DATE DEL CASO" + ", con durata, decorrenza e norma"),
    ("strumenti", "Strumenti processuali", "ogni strumento di STRUMENTI PROCESSUALI INDICATI DAI FATTI, con la sua "
                                           "norma e a che cosa serve in questo caso"),
    ("profilo_penale", "Profilo penale", "il reato configurabile con la sua norma, la procedibilità, la "
                                         "costituzione di parte civile, le prove del procedimento penale e la "
                                         "prescrizione dell'azione civile"),
    ("voci_di_danno", "Voci di danno", "ciascuna delle VOCI DI DANNO DA CONSIDERARE che spettano alla parte assistita"),
)


def _criminal_what(analysis: Dict[str, Any]) -> str:
    """ROLE_BLOCKS_ENABLED: what the criminal profile's paragraph treats, for the client's side
    (fact_signals.fit_to_role); the side-free text otherwise."""
    from . import fact_signals
    tool = next((t for t in (_signals(analysis) or {}).get("tools") or [] if t.get("id") == fact_signals.CRIMINAL_TOOL), {})
    return tool.get("what") or next(what for key, _, what in _SECTION5_BLOCKS if key == "profilo_penale")


_NUMBER_WORDS = {1: "un|uno|una", 2: "due", 3: "tre", 4: "quattro", 5: "cinque", 6: "sei", 8: "otto", 10: "dieci",
                 12: "dodici", 15: "quindici", 20: "venti", 26: "ventisei", 30: "trenta", 45: "quarantacinque",
                 60: "sessanta", 90: "novanta", 120: "centoventi"}
_UNIT_STEMS = {"anno": "ann", "anni": "ann", "mese": "mes", "mesi": "mes", "giorno": "giorn", "giorni": "giorn"}


def _length_written(length: str, text: str) -> bool:
    """A term's length ("8 giorni", "26 mesi", "1 anno") in the text, in figures or in words."""
    m = re.search(r"(\d+)\s+(anno|anni|mese|mesi|giorno|giorni)\b", length or "")
    if not m:
        return False
    n = int(m.group(1))
    number = f"{n}" + (f"|{_NUMBER_WORDS[n]}" if n in _NUMBER_WORDS else "")
    return bool(re.search(rf"\b(?:{number})\s+{_UNIT_STEMS[m.group(2)]}\w*", text or "", re.IGNORECASE))


def _cited_number(number: str, text: str) -> bool:
    return bool(re.search(rf"\bartt?\.\s*(?:[\w-]+\s*(?:,|e)\s*)*{re.escape(number)}\b", text or "", re.IGNORECASE))


_DAMAGE_HEADS = (("iure proprio", r"iure proprio|rapporto parentale|perdita del rapporto"),
                 ("iure hereditatis", r"iure hereditatis|danno terminale|catastrofale"),
                 ("assicurazioni", r"assicura"))


def material_use(analysis: Dict[str, Any], text: str) -> Dict[str, Dict[str, List[str]]]:
    """{block: {"offered": [...], "written": [...]}} of the procedural section's blocks, read off its text: a term by
    its length, a tool by one of its articles, the criminal route by 589/590 c.p., a head of damage by its words."""
    from . import fact_signals
    if not fact_signals.regimes_enabled():
        return {}
    signals = _signals(analysis) or {}
    out: Dict[str, Dict[str, List[str]]] = {}
    terms = signals.get("terms") or []
    if terms:
        out["termini"] = {"offered": [t["id"] for t in terms],
                          "written": [t["id"] for t in terms if _length_written(t.get("length", ""), text) or (
                              not t.get("length") and any(_cited_number(n, text) for n in re.findall(r"\d+", t.get("ref") or "")))]}
    tools = [t for t in signals.get("tools") or [] if t["id"] != fact_signals.CRIMINAL_TOOL]
    if tools:
        out["strumenti"] = {"offered": [t["id"] for t in tools],
                            "written": [t["id"] for t in tools if any(_cited_number(r[0], text) for r in t["refs"])]}
    if any(t["id"] == fact_signals.CRIMINAL_TOOL for t in signals.get("tools") or []):
        out["profilo_penale"] = {"offered": [fact_signals.CRIMINAL_TOOL],
                                 "written": [fact_signals.CRIMINAL_TOOL] if _cited_number("589", text)
                                 or _cited_number("590", text) else []}
    if signals.get("damages"):
        out["voci_di_danno"] = {"offered": [name for name, _ in _DAMAGE_HEADS],
                                "written": [name for name, pattern in _DAMAGE_HEADS
                                            if re.search(pattern, text or "", re.IGNORECASE)]}
    return out


def _missing_blocks(use: Dict[str, Dict[str, List[str]]]) -> List[str]:
    """The blocks the section left out: no item written, or for the terms (with REVIEW_FIXES_ENABLED, for every
    block) any item left out."""
    every = _review_fixes()
    return [block for block, u in use.items()
            if u["offered"] and (not u["written"] or ((every or block == "termini")
                                                      and len(u["written"]) < len(u["offered"])))]


_DAMAGES_TITLE = "VOCI DI DANNO DA CONSIDERARE"
_DAMAGES_RULE = "Una regola generale: valuta ciascuna voce sui fatti del caso, senza aggiungere fatti."
_ROLES_TITLE = "PARTI CON PIÙ RUOLI NEI FATTI"


def _damages(analysis: Dict[str, Any]) -> str:
    from . import fact_signals
    return fact_signals.damages_block(_signals(analysis)) if fact_signals.regimes_enabled() else ""


def _roles(analysis: Dict[str, Any]) -> str:
    from . import fact_signals
    return fact_signals.roles_block(_signals(analysis)) if fact_signals.regimes_enabled() else ""


def _tool_refs(analysis: Dict[str, Any]) -> List[tuple]:
    from . import fact_signals
    return fact_signals.tool_refs(_signals(analysis)) if fact_signals.regimes_enabled() else []


def _scope_block(plan: Dict[str, Any]) -> str:
    """NORME E LORO QUESTIONE (_discipline): each forced article with the plan issue(s) that hold it and
    the issue the graded check named for it (run 1302: 41/40/42 c.p., 1227 c.c., 590 c.p. cited for
    arguments they do not govern)."""
    if not _discipline() or not plan.get("forzate"):
        return ""
    lines = []
    for item in plan["forzate"]:
        n, c = item["ref"]
        titles = " o ".join(f"«{t}»" for t in item["questioni"])
        lines.append(f"- art. {n} {c}" + (f" ({item['tema']})" if item.get("tema") else "") + f": solo in {titles}")
    return "NORME E LORO QUESTIONE:\n" + "\n".join(lines)


def _without_heading(text: str, title: str) -> str:
    """The section's text without a heading line the model repeated anyway."""
    text = (text or "").strip()
    first, _, rest = text.partition("\n")
    bare = re.sub(r"^[#>*\s]*(?:\d+\.\s*)?|[*#:\s]+$", "", first)
    return rest.strip() if bare.lower() == title.lower() else text


# --- WRITER_RULES_ENABLED: after writing ------------------------------------------------------------

_CUT_TAIL_RE = re.compile(r"(?:[,;(\-–]|\b(?:e|o|ed|od|di|del|della|dei|delle|che|il|lo|la|i|gli|le|un|una|per|con|a|"
                          r"da|in|su|non|ma|se|al|alla|nel|nella|sul|sulla|è|sono))\s*$", re.IGNORECASE)
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-•*]|\d+[.)])\s+")


_RULE_TAIL_RE = re.compile(r"(?:\n[ \t]*(?:-{3,}|\*{3,}|_{3,})[ \t]*)+$|^[ \t]*(?:-{3,}|\*{3,}|_{3,})[ \t]*$")


def _looks_cut(text: str) -> bool:
    """The section stops mid-sentence (its token budget ran out): the last line does not end a sentence, and is
    not a short list item or a heading. A trailing rule line ("---") is no cut (v47: the one-call parere puts one after
    every section, 21 sections counted cut)."""
    body = _RULE_TAIL_RE.sub("", (text or "").rstrip()).rstrip()
    if not body or body[-1] in ".!?»”\")]:":
        return False
    last = body.split("\n")[-1].strip()
    if last.startswith(("#", "**")) and last.endswith("**"):
        return False
    if _LIST_ITEM_RE.match(last) and len(last) < 200 and not _CUT_TAIL_RE.search(last):
        return False
    return True


# WRITER_LOOP_GUARD (v46: c1's and p1's "Norme e giurisprudenza" repeated one sentence on art. 52 c.p. until its tokens ran
# out; _complete_sections then wrote it again at 1.5x the tokens, c1's looping again, 45 s, still cut): a reply where a
# run of _LOOP_NGRAM words comes back _LOOP_REPEATS times or more is cut where the run first comes back, trimmed to its
# last whole sentence, its sentences said twice kept once; the section is marked looped, and _complete_sections keeps it
# as it is. On every section call and every rewrite of one (_complete_sections, _cover_blocks). Logged per section as
# case_analysis_writer_loops.
# NORME_SPLIT (v48, h3 v47: "Norme e giurisprudenza" over 23 articles cut at 2,275 tokens, then rewritten at 1.5x): a
# plan with more than NORME_SPLIT_MIN articles has the section written in two concurrent calls over disjoint halves of
# them, each within NORME_SPLIT_SHARE of the tokens and words; a half cut at its length is trimmed to its last sentence,
# never written again. Logged as case_analysis_norme_split.
NORME_SPLIT_FLAG = "NORME_SPLIT"
NORME_SPLIT_MIN = 8
NORME_SPLIT_SHARE = 0.6


def norme_split(plan: Dict[str, Any]) -> bool:
    return dg.feature_flags.enabled(NORME_SPLIT_FLAG) and len(_norme_refs(plan)) > NORME_SPLIT_MIN


# NORME_SCOPE (v51, Leonardo v50: "Norme e giurisprudenza" enumerated 13 articles, 843 / 1059 / 1027 among them, one
# sentence each): the section writes only the questioni's primary norms -- graded 3, named by the user, a regime
# candidate's -- at most NORME_SCOPE_MAX, the user's first, then grade 3, then the regime's, each in the plan's order.
# The plan's other articles stay in the plan for "Applicazione al caso" and the procedural section, never listed in
# section 3. None of them primary: section 3 keeps every plan article. Logged per case as case_analysis_norme_scope.
NORME_SCOPE_FLAG = "NORME_SCOPE"
NORME_SCOPE_MAX = 8
NORME_SCOPE_KEY = "norme_scope"


def norme_scope_enabled() -> bool:
    return dg.feature_flags.enabled(NORME_SCOPE_FLAG)


def norme_scope(plan: Dict[str, Any], article_rows: List[Dict[str, Any]], named: Sequence[tuple],
                most: int = NORME_SCOPE_MAX) -> Tuple[List[tuple], Dict[str, Any]]:
    """(the section 3 articles, log): the plan's primary norms by NORME_SCOPE's rule."""
    rows_of: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in article_rows:
        if row.get("_article"):
            rows_of.setdefault(tuple(row["_article"]), []).append(row)

    def why(ref) -> str:
        rows = rows_of.get(tuple(ref)) or []
        if any(_same_ref(ref, n) for n in named):
            return "named"
        if any(r.get("_grade") == 3 for r in rows):
            return "grade 3"
        if any(r.get("_regime_candidate") for r in rows):
            return "regime"
        return ""
    refs = _plan_refs(plan)
    reasons = {ref: why(ref) for ref in refs}
    order = ("named", "grade 3", "regime")
    ranked = sorted((r for r in refs if reasons[r]), key=lambda r: (order.index(reasons[r]), refs.index(r)))
    scope = [r for r in refs if r in ranked[:most]]          # back in the plan's order
    label = lambda r: f"{r[0]} {r[1]}"
    return scope, {"scope": [label(r) for r in scope], "why": {label(r): reasons[r] for r in scope},
                   "cut": [label(r) for r in ranked[most:]], "others": [label(r) for r in refs if r not in scope],
                   "max": most}


def _norme_refs(plan: Dict[str, Any]) -> List[tuple]:
    """Section 3's articles: the NORME_SCOPE set when the plan has one, else all the plan's."""
    if plan.get(NORME_SCOPE_KEY) is not None:
        return [tuple(r) for r in plan[NORME_SCOPE_KEY]]
    return _plan_refs(plan)


def _scoped_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """The plan as section 3 reads it: each questione with only its NORME_SCOPE norms."""
    if plan.get(NORME_SCOPE_KEY) is None:
        return plan
    keep = set(_norme_refs(plan))
    return {**plan, "questioni": [{**q, "norme": [r for r in q["norme"] if tuple(r) in keep]} for q in plan["questioni"]]}


WRITER_LOOP_FLAG = "WRITER_LOOP_GUARD"
_LOOP_NGRAM = 8
_LOOP_REPEATS = 3


def _loop_guard() -> bool:
    return dg.feature_flags.enabled(WRITER_LOOP_FLAG)


def find_loop(text: str) -> Optional[Dict[str, Any]]:
    """{"at": where the looping block first comes back, "ngram", "count", "period"}, or None. A loop: a run of
    _LOOP_NGRAM words found _LOOP_REPEATS times or more (not overlapping) where two successive gaps between its
    occurrences are equal and the words of the two blocks the same -- the reply saying one block again and again. A
    formula the writer reuses with other words in between ("Questa norma, applicabile al caso di Tizio, potrebbe ...",
    one per article; c1's and p1's "Applicazione al caso") is no loop."""
    words = [(m.group(0).lower(), m.start()) for m in re.finditer(r"\w+", text or "")]
    tokens = [w for w, _ in words]
    seen: Dict[Tuple[str, ...], List[int]] = {}
    for i in range(len(tokens) - _LOOP_NGRAM + 1):
        gram = tuple(tokens[i:i + _LOOP_NGRAM])
        at = seen.setdefault(gram, [])
        if not at or i - at[-1] >= _LOOP_NGRAM:
            at.append(i)
    best = None
    for gram, at in seen.items():
        if len(at) < _LOOP_REPEATS:
            continue
        for a, b, c in zip(at, at[1:], at[2:]):
            if b - a == c - b and tokens[a:b] == tokens[b:c]:
                if best is None or b < best[0]:
                    best = (b, gram, len(at), b - a)
                break
    if best is None:
        return None
    b, gram, count, period = best
    return {"at": words[b][1], "ngram": " ".join(gram), "count": count, "period": period}


def unloop(text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """(text, the loop or None): with WRITER_LOOP_GUARD, a looping reply cut where it first repeats, trimmed to its last
    whole sentence, each sentence said twice kept once (find_loop)."""
    if not _loop_guard():
        return text, None
    loop = find_loop(text)
    if loop is None:
        return text, None
    kept = _trim_to_sentence(text[:loop["at"]].rstrip())
    seen, drops = set(), []
    for s, e in _sentence_spans(kept):
        key = " ".join(re.findall(r"\w+", kept[s:e].lower()))
        if key in seen:
            drops.append((s, e))
        seen.add(key)
    kept = _drop_spans(kept, drops) if drops else kept
    return kept, {**loop, "chars_before": len(text or ""), "chars_after": len(kept), "repeated_sentences": len(drops)}


def _trim_to_sentence(text: str) -> str:
    """text up to its last complete sentence (a cut list item goes whole)."""
    body = (text or "").rstrip()
    head, _, last = body.rpartition("\n")
    if head and _LIST_ITEM_RE.match(last):
        return head.rstrip()
    for i in range(len(body) - 1, -1, -1):
        if body[i] in ".!?" and _ends_sentence(body, i):
            return body[:i + 1]
    return body


def _sentence_spans(text: str) -> List[Tuple[int, int]]:
    """(start, end) of each sentence of text, within its line; headings and lines without words are skipped."""
    spans, pos = [], 0
    for line in text.split("\n"):
        start, end = pos, pos + len(line)
        pos = end + 1
        stripped = line.strip()
        if not re.search(r"[a-zà-ù]{3}", stripped, re.IGNORECASE) or (
                _HEADING_LINE_RE.match(line) and not _HEADING_LINE_RE.match(line).group(2).strip(" .:;–-")):
            continue
        s = start + len(line) - len(line.lstrip())
        mark = _PARAGRAPH_MARK_RE.match(text, s, end)     # v60f (X4): a paragraph's «[N]» is no sentence's
        if mark:
            s = mark.end()
        for i in range(s, end):
            if text[i] in ".!?" and _ends_sentence(text, i):
                if text[s:i + 1].strip():
                    spans.append((s, i + 1))
                s = i + 1
                while s < end and text[s] in " \t":
                    s += 1
        if s < end and text[s:end].strip():
            spans.append((s, end))
    return spans


def _drop_spans(text: str, spans: List[Tuple[int, int]]) -> str:
    """text without the spans and the sentences they orphan (with_orphans); a list item or line left with no words goes."""
    spans = sorted((s, with_orphans(text, s, e)) for s, e in spans)
    merged: List[Tuple[int, int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(e, merged[-1][1]))
        else:
            merged.append((s, e))
    for s, e in reversed(merged):
        line_start = text.rfind("\n", 0, s) + 1
        line_end = text.find("\n", e)
        line_end = len(text) if line_end < 0 else line_end
        if not text[line_start:s].strip() and not text[e:line_end].strip():      # the whole line: it goes
            text = text[:line_start] + text[line_end + 1:] if line_end < len(text) else text[:max(0, line_start - 1)]
        else:
            text = text[:s] + text[e:]
    lines = [ln.rstrip() for ln in text.split("\n") if not re.fullmatch(r"\s*(?:[-•*]|\d+[.)])?\s*[.;:]?\s*", ln)
             or not ln.strip()]
    return re.sub(r"(?<=\S)[ \t]{2,}", " ", re.sub(r"\n{3,}", "\n\n", "\n".join(lines))).strip()


_FACT_MARK = " [fatto non indicato nel caso: da verificare]"


def _mark_spans(text: str, spans: List[Tuple[int, int]]) -> str:
    """_FACT_MARK at the end of each span, before its closing punctuation."""
    for s, e in sorted(spans, reverse=True):
        at = e - 1 if text[e - 1] in ".!?;:" else e
        text = text[:at] + _FACT_MARK + text[at:]
    return text


# A sentence saying an article does not apply, is not relevant, falls outside the case.
_NOT_APPLY_RE = re.compile(
    r"\bnon\s+(?:è|e'|sono|risulta\w*|appar\w+|sembr\w+|pare|può\s+dirsi|si\s+ritiene)?\s*(?:direttamente\s+|"
    r"propriamente\s+|strettamente\s+|qui\s+)?(?:pertinent|applicabil|rilevant|invocabil|conferent)\w*"
    r"|\bnon\s+(?:si\s+applica\w*|trova\w*\s+applicazione|rileva\w*|riguard\w+\s+(?:il|la|questo|questa|lo)\s+"
    r"(?:caso|fattispecie|vicenda))|\binapplicabil\w*|\birrilevant\w*|\besul[ao]\w*|\bestrane[oaie]\s+(?:al|alla|ai|alle)\s+"
    r"(?:caso|fattispecie|vicenda|questione)|\bnon\s+(?:viene|vengono)\s+in\s+(?:rilievo|considerazione)", re.IGNORECASE)
# Without the review call: only the remarks on relevance, never an argument that a norm does not apply.
_NOT_RELEVANT_RE = re.compile(
    r"\bnon\s+(?:è|e'|sono|risulta\w*|appar\w+|sembr\w+)?\s*(?:direttamente\s+|propriamente\s+|strettamente\s+)?"
    r"(?:pertinent|rilevant|conferent)\w*|\besul[ao]\w*\s+(?:dal|dalla|dai|dalle)\s+(?:caso|fattispecie|vicenda|"
    r"questione|tema)|\bestrane[oaie]\s+(?:al|alla|ai|alle)\s+(?:caso|fattispecie|vicenda|questione)", re.IGNORECASE)
_BURDEN_RE = re.compile(
    r"\bl?['’]?\s*onere\s+(?:della\s+prova|probatorio|di\s+(?:provare|dimostrare))\b[^.;]{0,60}?\b(?:ricad\w+|grav\w+|"
    r"incomb\w+|spett\w+|è\s+a\s+carico|resta\s+a\s+carico|è\s+in\s+capo|pes\w+)", re.IGNORECASE)
# What a norm's own text says when it allocates the burden or sets a presumption.
_ALLOCATION_RE = re.compile(
    r"presum|presunzion|prova\s+contraria|salvo\s+che\s+(?:non\s+)?provi|se\s+non\s+prova|non\s+prova|deve\s+provare|"
    r"devono\s+provare|tenut\w\s+a\s+provare|onere\s+della\s+prova|fino\s+a\s+prova|a\s+meno\s+che\s+(?:non\s+)?provi|"
    r"se\s+prova|chi\s+vuol\s+far\s+valere|deve\s+dimostrare|prova\s+di\s+aver", re.IGNORECASE)

_REVIEW_SYSTEM = (
    "Controlli una sezione di un parere legale, a frasi numerate, confrontandola con il CASO. Segnala solo le "
    "frasi di due tipi.\n"
    "- \"non_applicabile\": la frase cita una norma soltanto per dire che non è pertinente, non si applica o non "
    "rileva nel caso, senza usarla per un argomento. Non segnalare la frase che esclude una norma su cui la "
    "controparte o l'accusa fonda la pretesa contro la parte assistita: è un argomento.\n"
    "- \"fatto_non_nel_caso\": solo una frase che afferma come avvenuta, senza riserve, una condotta, una "
    "conoscenza o un'intenzione di una parte (che abbia fatto, omesso, saputo o voluto qualcosa) che il CASO non "
    "dice né rende evidente, compresi i fatti di una pronuncia attribuiti alle parti del caso. Non segnalare mai: un "
    "argomento («la parte può sostenere, eccepire, dimostrare»), un'ipotesi o una verifica («se risultasse», "
    "«occorre accertare»), una frase che riprende il testo del CASO, una citazione di una norma o di una pronuncia, "
    "un elenco di prove da acquisire, una qualificazione giuridica dei fatti narrati.\n"
    "- \"contro_assistito\": la frase propone un'azione, una domanda, una denuncia o una querela contro la PARTE "
    "ASSISTITA o contro chi è assistito insieme a essa. Non segnalare la frase che riporta un'obiezione della "
    "controparte per replicarvi.\n"
    'Rispondi SOLO con un array JSON, vuoto se nessuna frase va segnalata: '
    '[{"i": 3, "tipo": "fatto_non_nel_caso", "perche": "una frase"}]'
)


# What a "fact not in the case" flag is never about (run 20261026-0841's false positives), checked in code.
_ARGUES_RE = re.compile(
    r"\b(?:può|possono|potrà|potranno|potrebbe|potrebbero|dovrà|dovranno|deve|devono|occorre|occorrerà|occorrerebbe|"
    r"bisogna|conviene|va|vanno|si\s+potrà|si\s+può|è\s+possibile|sarà\s+possibile)\s+(?:\w+\s+){0,2}?(?:sostenere|"
    r"eccepire|dimostrare|provare|argomentare|affermare|contestare|rilevare|dedurre|allegare|accertare|verificare|"
    r"chiedere|invocare|far\s+valere|osservare|obiettare|replicare)\b|\b(?:se|qualora|ove|laddove|nel\s+caso\s+in\s+cui)\s",
    re.IGNORECASE)
_SOURCE_RE = re.compile(r"\bcass\.|\bsent\.|\bord\.|\bn\.\s*\d+/\d{2,4}\b|\bsez\.|\bcorte\b|\btribunale\b",
                        re.IGNORECASE)
_EVIDENCE_RE = re.compile(r"\bacquisir|\bda\s+acquisire|\bprodurre\b|\bchiedere\s+copia|\bprove\s+da|\braccogliere|"
                          r"\bdocumentazione\b|\btestimon|\bperizi|\bconsulenza\s+tecnica|\bfotograf", re.IGNORECASE)
_QUOTED_RE = re.compile(r"[«“\"]([^»”\"]{12,})[»”\"]")


def _words(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-zà-ù]{4,}", (text or "").lower())]


def _from_the_case(sentence: str, case_text: str) -> bool:
    """The sentence restates the case: a quoted passage of it, or (fuzzy) most of its content words are the case's."""
    case_norm = " ".join(_words(case_text))
    for quoted in _QUOTED_RE.findall(sentence):
        if len(_words(quoted)) >= 3 and " ".join(_words(quoted)) in case_norm:
            return True
    words = [w for w in _words(sentence) if w not in _HEADING_STOP]
    case_words = set(_words(case_text)) | {w[:6] for w in _words(case_text)}
    hits = sum(1 for w in words if w in case_words or w[:6] in case_words)
    return len(words) >= 4 and hits / len(words) >= 0.8


def _fact_flag_excluded(sentence: str, case_text: str) -> Optional[str]:
    """Why a "fact not in the case" flag on the sentence is not kept, or None: it argues, lists evidence to gather,
    quotes or cites a norm or a ruling, or restates the case."""
    if re.match(r"\s*(?:[-•*]|\d+[.)])\s+", sentence) or _EVIDENCE_RE.search(sentence):
        return "evidence to gather"
    if _ARGUES_RE.search(sentence):
        return "argues or supposes"
    if _cites(sentence) or _SOURCE_RE.search(sentence):
        return "cites a source"
    if _QUOTED_RE.search(sentence) and not _from_the_case(sentence, case_text):
        return "quotes a norm or ruling"
    if _from_the_case(sentence, case_text):
        return "restates the case"
    return None


def _cites(sentence: str) -> bool:
    return bool(dg._ARTICLE_RUN_RE.search(sentence or ""))


def _row_text(article_rows: List[Dict[str, Any]], ref: tuple) -> str:
    return " ".join(" ".join(str((r.get("s") or {}).get("plain_text") or (r.get("s") or {}).get("abstract") or "").split())
                    for r in article_rows if r.get("_article") and _same_ref(r["_article"], ref))


def _burden_templates(title: str, text: str, spans: List[Tuple[int, int]], article_rows: List[Dict[str, Any]],
                      documents: List[Dict[str, Any]], known: set) -> Dict[int, str]:
    """{span index: why} of the sentences stating who bears the burden of proof although neither a quoted
    passage nor the text of the articles they cite (or the previous sentence of their paragraph cites) allocates
    it or sets a presumption. A sentence citing nothing, with nothing cited before it, is kept outside the
    norms section."""
    out: Dict[int, str] = {}
    for k, (s, e) in enumerate(spans):
        sentence = text[s:e]
        if not _BURDEN_RE.search(sentence):
            continue
        quoted = " ".join(re.findall(r"[«“\"]([^»”\"]{8,})[»”\"]", sentence))
        if quoted and _ALLOCATION_RE.search(quoted):
            continue
        refs = [r for _, _, rs in _citations(sentence, documents, known) for r in rs]
        j = k - 1
        while not refs and j >= 0 and "\n\n" not in text[spans[j][1]:s]:
            refs = [r for _, _, rs in _citations(text[spans[j][0]:spans[j][1]], documents, known) for r in rs]
            j -= 1
        if not refs:
            if title == SECTION_TITLES[2]:
                out[k] = "burden stated without a norm"
            continue
        if any(_same_ref(r, ("2697", "c.c.")) or _ALLOCATION_RE.search(_row_text(article_rows, r)) for r in refs):
            continue
        out[k] = "burden template: " + ", ".join(f"{n} {c}" for n, c in refs) + " allocate nothing"
    return out


_ACQUIRED_BY_1051_RE = re.compile(r"\b(?:acquist|acquisit|sort[ao]\b|nat[ao]\b)[^;]{0,80}?\b1051\b|\b1051\b[^;]{0,60}?\b(?:acquist|acquisit)",
                                  re.IGNORECASE)
_SERVITUDE_PRESUMED_RE = re.compile(r"presunzion\w* (?:di|della|dell['’]) ?(?:esistenza (?:di|della) )?servit|servit\w* present",
                                    re.IGNORECASE)
_DENIED_RE = re.compile(r"\b(?:non|né|nessuna?|alcuna?|senza|esclus\w*)\b", re.IGNORECASE)


_NEGATORIA_RE = re.compile(r"\bnegatori\w*|\b949\b", re.IGNORECASE)
_DAMAGE_MEASURE_RE = re.compile(r"misur|quantific|liquid|conseguenz\w* immediat|perdita|mancato guadagno|danno emergente|lucro",
                                re.IGNORECASE)
_DAMAGE_BASIS_RE = re.compile(r"\b(?:2043|2056|1218|1079)\b")
_SATISFIED_RE = re.compile(r"soddisfatt|maturat|compiut|decors|integrat|raggiunt|perfezionat|superat", re.IGNORECASE)
# v42: an acquisition stated as done, for a term the dates leave open. Not "giurisprudenza consolidata".
_ACQUIRED_DONE_RE = re.compile(r"\b(?:acquisit[aoie]|acquisizione|consolidat[aoie]|confermat[aoie]|confermando|perfezionat[aoie]|maturat[aoie])\b",
                               re.IGNORECASE)
_SETTLED_LAW_RE = re.compile(r"giurisprudenz|orientament|principio|cassazione", re.IGNORECASE)
_UNSETTLED_RE = re.compile(r"non verificabil|da verificare|incert", re.IGNORECASE)
_UNSETTLED_STATUS = ("NON VERIFICABILE", "INCERTO", "DA VERIFICARE")
_POSSESSORY_RE = re.compile(r"reintegr|manutenzion|possessori", re.IGNORECASE)
_PROPOSED_RE = re.compile(r"proponibil|esperibil|propor|agire|in via subordinata", re.IGNORECASE)
_PRECLUDED_RE = re.compile(r"preclus|decadut|decadenz|tardiv|inammissib", re.IGNORECASE)
_FROM_USE_RE = re.compile(r"(?:dall|con l|per l|mediante l)['’]uso|uso (?:ventennal|prolungat|protratt|continuat)", re.IGNORECASE)


def statement_context(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """What _wrong_statement checks a sentence against: {} without typed facts."""
    from . import answer_invariants, fact_signals
    signals = _signals(analysis) or {}
    if not signals.get("facts"):
        return {}
    return {"servitude": any(re.search(r"servit", a, re.IGNORECASE) for a in acquired_rights(analysis)),
            "contract": bool(signals["facts"].get("contracts")),
            "terms": [{k: t.get(k) for k in ("id", "ref", "length", "status", "from", "start_unknown")}
                      for t in signals.get("terms") or []],
            "mediation": (signals.get("mediation") or {}).get("required"),
            "split_owner": str(fact_signals._the_split(signals["facts"]).get("owner") or ""),
            "conclusions": [{k: r.get(k) for k in ("id", "conclusion_subject", "conclusion_unless")}
                            for r in signals.get("fired") or [] if r.get("conclusion") and not r.get("unstated")],
            "defendants": bool(signals.get("defendants")),
            "excluded": [tuple(x) for r in signals.get("fired") or [] for x in r.get("excludes") or []],
            "excluded_by": [{"rule": r.get("id"), "article": tuple(x)} for r in signals.get("fired") or []
                            for x in r.get("excludes") or []],
            "case_durations": sorted(answer_invariants.durations(fact_signals._texts(signals["facts"])))}


def _term_marker(term: Dict[str, Any]) -> Optional[re.Pattern]:
    """How a sentence names a term: its article, or, for a twenty years' term, those words."""
    numbers = re.findall(r"\b\d{3,4}\b", str(term.get("ref") or ""))
    twenty = re.search(r"20|vent", str(term.get("length"))) and term.get("id") != "non_uso_servitu"   # usucapione's words, not its
    words = list(numbers) + (["ventenn\\w*", "vent['’]anni", "venti anni", "20 anni"] if twenty else [])
    words += ["usucapi\\w*"] if str(term.get("id") or "").startswith("usucapione") else []
    return re.compile(r"\b(?:" + "|".join(words) + r")", re.IGNORECASE) if words else None


def _term_conflict(sentence: str, terms: List[Dict[str, Any]]) -> str:
    """Computed terms are the single source: a sentence calling satisfied a term they leave open, or the reverse."""
    for term in terms:
        marker = _term_marker(term)
        if not marker or not marker.search(sentence):
            continue
        settled = _SATISFIED_RE.search(sentence) and not _DENIED_RE.search(sentence)
        done = _ACQUIRED_DONE_RE.search(sentence) and not _DENIED_RE.search(sentence) and not _SETTLED_LAW_RE.search(sentence) \
            and not _CONDITION_RE.search(sentence)
        if term.get("status") in _UNSETTLED_STATUS and (settled or done) and not _UNSETTLED_RE.search(sentence):
            return f"the computed term {term.get('id')} is {term.get('status')}"
        if term.get("status") == "DECORSO" and _UNSETTLED_RE.search(sentence):
            return f"the computed term {term.get('id')} has run"
    return ""


# Terms the case's actions have by name: a sentence giving such an action another term goes ("confessoria ... 10 anni").
_ACTION_NAMES = {"azione_confessoria": r"confessori", "azione_negatoria": r"negatori", "azione_rivendicazione": r"rivendic",
                 "possessoria": r"possessori|reintegrazion|manutenzion", "non_uso_servitu": r"non uso"}
_STATES_TERM_RE = re.compile(r"prescri|decad|\btermine\b|\bentro\b|estingu", re.IGNORECASE)
_NO_TERM_RE = re.compile(r"\bnon\s+(?:si\s+|è\s+|sono\s+)?(?:prescriv|soggett|soggiac)|imprescrittibil", re.IGNORECASE)


def _other_term(sentence: str, terms: List[Dict[str, Any]], in_case: Sequence[tuple]) -> str:
    """A duration stated for an action with a computed entry, found neither among the computed terms nor in the case."""
    from . import answer_invariants
    if not _STATES_TERM_RE.search(sentence) or _NO_TERM_RE.search(sentence):
        return ""
    allowed = {tuple(d) for d in in_case} | {d for t in terms for d in answer_invariants.durations(str(t.get("length") or ""))}
    stray = answer_invariants.durations(sentence) - allowed
    named = next((t["id"] for t in terms if t.get("id") in _ACTION_NAMES
                  and re.search(_ACTION_NAMES[t["id"]], sentence, re.IGNORECASE)), "")
    return f"a term the computed terms do not give for {named}" if stray and named else ""


_DENIES_RE = re.compile(r"\bnon\s+(?:è\s+|sono\s+|risulta\s+)?(?:applicabil|invocabil|configurabil|maturat|sussist|ricorr|"
                        r"oper[ai]\b|si\s+applica|spett|può\s+(?:operare|applicarsi|invocarsi))|\binapplicabil|"
                        r"(?<!non è )(?<!non sono )\besclus[aoie]\b|\bsi\s+esclude\b|\bda\s+escludere\b|\bprecluso\b|\bpreclusa\b",
                        re.IGNORECASE)
_THE_OTHER_SIDE_RE = re.compile(r"controparte|obiezion|tesi avversaria|potrebbe(?:ro)?\s+(?:sostenere|obiettare|eccepire)",
                                re.IGNORECASE)


def _denied_conclusion(sentence: str, conclusions: List[Dict[str, Any]]) -> str:
    """A fired rule's conclusion denied as a plain statement: an objection or a condition is not a denial."""
    if not _DENIES_RE.search(sentence) or _THE_OTHER_SIDE_RE.search(sentence) or _CONDITION_RE.search(sentence):
        return ""
    for rule in conclusions:
        unless = rule.get("conclusion_unless")
        if re.search(rule["conclusion_subject"], sentence, re.IGNORECASE) and not (unless and re.search(unless, sentence, re.IGNORECASE)):
            return f"denies the conclusion of rule {rule.get('id')}"
    return ""


_MEDIATION_DENIED_RE = re.compile(r"\bnon\s+(?:è\s+|sarà\s+|risulta\s+|appare\s+)?(?:necessari|obbligatori|richiest|dovut|occorre|serve|"
                                  r"condizione)|\bfacoltativ|\besclus[aoie]\b|\bsuperflu", re.IGNORECASE)
_MEDIATION_MANDATORY_RE = re.compile(r"obbligatori|condizione di procedibilit", re.IGNORECASE)
_TOOL_NAMES_RE = re.compile(r"sfratt|decreto ingiuntivo|ingiunzion|sequestr|diffida|pignorament|accertamento tecnico|"
                            r"provvedimento d['’]urgenza|azione possessoria|denuncia di nuova opera|denuncia di danno temuto|licenza",
                            re.IGNORECASE)
_TOOL_NOT_USED_RE = re.compile(r"\bnon\s+(?:è\s+|sono\s+|risulta\s+|risultano\s+|appare\s+|appaiono\s+)?(?:pertinent|praticabil|"
                               r"necessari|applicabil|utilizzabil|esperibil|opportun|utili?\b|rilevant|percorribil)|\bsuperflu|\binutil|"
                               r"\birrilevant", re.IGNORECASE)
_NON_USE_NAMED_RE = re.compile(r"non\s+uso|estinzion|estingu|\b1073\b", re.IGNORECASE)
_TWENTY_RE = re.compile(r"ventenn|vent['’]anni|venti anni|20 anni", re.IGNORECASE)
# v48 (the PO's red on Leonardo: "usata ... per oltre vent'anni" with no start of the possession in the facts)
_TWENTY_RUN_RE = re.compile(r"(?:oltre|più di|piu di|da almeno|ben)\s+(?:vent['’]anni|venti anni|20 anni|un ventennio)|"
                            r"(?:ventennio|vent['’]anni|venti anni)\s+(?:è\s+|sono\s+|risulta\s+)?(?:già\s+)?"
                            r"(?:maturat|compiut|decors|trascors|integrat)|usucapion\w*\s+(?:è\s+|risulta\s+)?(?:già\s+)?"
                            r"(?:maturat|compiut|intervenut|perfezionat)", re.IGNORECASE)
_ALIENATION_OF_RE = re.compile(r"alienazion\w*\s+(?:del\s+(?:fondo|terreno|podere)\s+)?(?:di|da parte di)\s+([A-ZÀ-Ü]\w+)")


def _non_use_as_usucapione(sentence: str, terms: List[Dict[str, Any]]) -> str:
    """The non-use term's date used as if it started another twenty years: it only extinguishes the servitude."""
    term = next((t for t in terms if t.get("id") == "non_uso_servitu"), None)
    year = re.search(r"\d{4}", str(((term or {}).get("from") or {}).get("date") or ""))
    if year and year.group(0) in sentence and _TWENTY_RE.search(sentence) and not _NON_USE_NAMED_RE.search(sentence):
        return "the 20 years from the last use are the servitude's extinction by non-use, not usucapione"
    return ""


def _case_statement(sentence: str, context: Dict[str, Any]) -> str:
    """What the computed facts forbid: mediation against its computed answer, a tool written only to say it does not
    apply, art. 1054 c.c. tied to another sale than the split of the land."""
    if "mediazion" in sentence.lower():
        required = context.get("mediation")
        if required and _MEDIATION_DENIED_RE.search(sentence):
            return "mediation is a condition of the action here (art. 5 d.lgs. 28/2010)"
        if required is False and _MEDIATION_MANDATORY_RE.search(sentence):
            return "the case's subject is not among those of art. 5 d.lgs. 28/2010"
    if _TOOL_NAMES_RE.search(sentence) and _TOOL_NOT_USED_RE.search(sentence) and not _THE_OTHER_SIDE_RE.search(sentence):
        return "a line on a tool that does not apply"
    if any(t.get("start_unknown") for t in context.get("terms") or []) and _TWENTY_RUN_RE.search(sentence) \
            and not _NON_USE_NAMED_RE.search(sentence) and not _UNSETTLED_RE.search(sentence) \
            and not _THE_OTHER_SIDE_RE.search(sentence) and not _CONDITION_RE.search(sentence):
        return "the computed term leaves the possession's start unknown: no twenty years can be said to have run"
    owner = context.get("split_owner")
    if owner and re.search(r"\b1054\b|altro contraente", sentence, re.IGNORECASE):
        named = _ALIENATION_OF_RE.search(sentence)
        if named and named.group(1) != owner:
            return f"art. 1054 c.c. goes with the split of {owner}'s land"
    return ""


# USUCAPIONE_CONDITIONAL (v51, Leonardo v50: "la servitù di passaggio, acquisita per destinazione del padre di famiglia
# e per uso continuo e pacifico" in sections 1, 4 and 6, with "inizio del possesso da accertare" among the computed
# terms): when a computed term leaves the possession's start unknown, a writer sentence asserting the acquisition by
# usucapione or a matured possession -- "acquisita", "maturata", "compiuto", "oltre / superiore a venti anni",
# "ventennale compiuto", next to usucapione, continuous use or possession -- is rewritten to the conditional form:
# "acquisita" -> "acquisibile", "già" out, "oltre venti anni" -> "venti anni", and after its usucapione words
# CONDITIONAL_CLAUSE. A denied, conditional, unsettled or counterparty's sentence, or one on the non-use term, is left
# as written. Applied after writing (write_sections) and in final_checks; logged as case_analysis_usucapione_conditional.
USUCAPIONE_CONDITIONAL_FLAG = "USUCAPIONE_CONDITIONAL"
CONDITIONAL_CLAUSE = " (se il possesso ventennale risulta compiuto: l'inizio del possesso è da accertare)"
_USUCAPIONE_WORDS_RE = re.compile(r"usucapion\w*(?:\s+(?:ordinaria|ventennale|abbreviata))?|"
                                  r"uso\s+continu\w*(?:\s+e\s+pacific\w*)?|"
                                  r"possesso\s+(?:continu\w*|ventennale|pacific\w*)(?:\s+e\s+(?:pacific|continu)\w*)?",
                                  re.IGNORECASE)
_ASSERTED_RE = re.compile(r"\b(?:acquisit|acquistat|maturat|compiut)[aoie]\b|"
                          r"\b(?:oltre|superiore\s+a|più\s+di|piu\s+di)\s+(?:vent['’]anni|venti\s+anni|20\s+anni|un\s+ventennio|"
                          r"il\s+ventennio)|\bventennale\s+compiut", re.IGNORECASE)
_NOT_ASSERTED_RE = re.compile(r"\bnon\s+(?:è\s+|sono\s+|risulta\s+|risultano\s+|può\s+dirsi\s+)?(?:ancora\s+)?(?:stat[aoie]\s+)?"
                              r"(?:acquisit|acquistat|maturat|compiut)|\bacquisibil|\bacquistabil", re.IGNORECASE)
_MORE_THAN_TWENTY_RE = re.compile(r"\b(?:oltre|superiore\s+a|più\s+di|piu\s+di)\s+(?=vent['’]anni|venti\s+anni|20\s+anni)",
                                  re.IGNORECASE)


def usucapione_asserted(sentence: str, context: Optional[Dict[str, Any]]) -> str:
    """USUCAPIONE_CONDITIONAL: why the sentence asserts an acquisition by usucapione the computed terms leave open, or
    "" (the flag off, no term with an unknown start, or a sentence that does not assert it)."""
    if not dg.feature_flags.enabled(USUCAPIONE_CONDITIONAL_FLAG):
        return ""
    if not any(t.get("start_unknown") for t in (context or {}).get("terms") or []):
        return ""
    if not _USUCAPIONE_WORDS_RE.search(sentence) or not _ASSERTED_RE.search(sentence) \
            or _NOT_ASSERTED_RE.search(sentence) or _UNSETTLED_RE.search(sentence) or _CONDITION_RE.search(sentence) \
            or _THE_OTHER_SIDE_RE.search(sentence) or _NON_USE_NAMED_RE.search(sentence):
        return ""
    return "usucapione asserted with the possession's start unknown"


def conditional_form(sentence: str) -> str:
    """The sentence asserting usucapione in the conditional form (USUCAPIONE_CONDITIONAL)."""
    out = re.sub(r"\bgià\s+(?=(?:stat[aoie]\s+)?(?:acquisit|acquistat|maturat|compiut))", "", sentence, flags=re.IGNORECASE)
    out = re.sub(r"\b(?:stat[aoie]\s+)?(acquisit|acquistat)([aoie])\b",
                 lambda m: ("acquisibil" if m.group(1).lower() == "acquisit" else "acquistabil")
                 + ("i" if m.group(2).lower() in "ie" else "e"), out, flags=re.IGNORECASE)
    out = _MORE_THAN_TWENTY_RE.sub("", out)
    m = _USUCAPIONE_WORDS_RE.search(out)
    return out[:m.end()] + CONDITIONAL_CLAUSE + out[m.end():] if m else out


def usucapione_conditional(title: str, text: str, context: Optional[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """(text, log): each sentence usucapione_asserted flags rewritten to conditional_form."""
    if not dg.feature_flags.enabled(USUCAPIONE_CONDITIONAL_FLAG):
        return text, []
    changes, log = {}, []
    for s, e in _sentence_spans(text or ""):
        sentence = text[s:e]
        why = usucapione_asserted(sentence, context)
        if why:
            new = conditional_form(sentence)
            changes[(s, e)] = new
            log.append({"section": title, "kind": why, "how": "made conditional", "sentence": sentence.strip()[:300],
                        "now": new.strip()[:300]})
    return (grounded_review.apply(text, changes) or text) if changes else text, log


def usucapione_sections(sections: List[Tuple[str, str]], context: Optional[Dict[str, Any]],
                        stage: str) -> List[Tuple[str, str]]:
    """usucapione_conditional on each section, logged once."""
    out, log = [], []
    for title, text in sections:
        new, entries = usucapione_conditional(title, text, context)
        out.append((title, new))
        log += entries
    if log:
        dg.vlog("case_analysis_usucapione_conditional", {"stage": stage, "entries": log, "count": len(log)})
    return out


_MAYBE_PARTIES_RE = re.compile(r"\beventual\w+\s+(?:altr\w+\s+)?(?:com?proprietar|proprietar|confinant|contitolar|litisconsort|"
                               r"terz|convenut|soggett|parti)", re.IGNORECASE)


def _wrong_statement(sentence: str, context: Optional[Dict[str, Any]] = None) -> str:
    """Statements of law the writer must never make, by their form: a servitude acquired under art. 1051 c.c. (it is
    constituted by judgment against indemnity), a servitude presumed from use, art. 1062 c.c. arising from use, art. 1223
    c.c. as the basis of damages; with the case's context, the negatoria for one claiming a right on another's land,
    a breach with no contract, art. 1159 c.c. or a time-barred possessoria as open, a term against its computed status.
    Denied, they are right."""
    context = context or {}
    terms = context.get("terms") or []
    conflict = _term_conflict(sentence, terms) or _other_term(sentence, terms, context.get("case_durations") or []) \
        or _denied_conclusion(sentence, context.get("conclusions") or []) or _non_use_as_usucapione(sentence, terms) \
        or _case_statement(sentence, context)
    if conflict:
        return conflict
    if context.get("defendants") and _MAYBE_PARTIES_RE.search(sentence):
        return "the parties to sue are named in the facts"
    if _DENIED_RE.search(sentence):
        return ""
    if re.search(r"\b1062\b|padre di famiglia", sentence, re.IGNORECASE) and _FROM_USE_RE.search(sentence):
        return "art. 1062 c.c.: the servitude arises from the separation, not from use"
    if re.search(r"\b1223\b", sentence) and not _DAMAGE_BASIS_RE.search(sentence) and not _DAMAGE_MEASURE_RE.search(sentence):
        return "art. 1223 c.c. measures damages, it does not found them"
    if context.get("servitude") and _NEGATORIA_RE.search(sentence) and not _THE_OTHER_SIDE_RE.search(sentence):
        return "the holder of a right on another's land acts in confessoria (art. 1079 c.c.)"
    if context.get("contract") is False and re.search(r"\binadempiment", sentence, re.IGNORECASE):
        return "a breach with no contract in the facts"
    if context and re.search(r"\b1159\b|usucapione abbreviata", sentence, re.IGNORECASE) and _SATISFIED_RE.search(sentence) \
            and not any(t.get("id") == "usucapione_abbreviata" for t in terms):
        return "art. 1159 c.c. needs a transcribed title from a non-owner, which the facts do not show"
    if any(t.get("id") == "possessoria" and t.get("status") == "DECORSO" for t in terms) and _POSSESSORY_RE.search(sentence) \
            and _PROPOSED_RE.search(sentence) and not _PRECLUDED_RE.search(sentence):
        return "a time-barred possessoria is precluded"
    if _ACQUIRED_BY_1051_RE.search(sentence):
        return "art. 1051 c.c. constitutes by judgment: nothing is acquired under it"
    if _SERVITUDE_PRESUMED_RE.search(sentence):
        return "no presumption of a servitude from use"
    return ""


def _review_call(title: str, text: str, spans: List[Tuple[int, int]], case_text: str,
                 party: str = "") -> Optional[List[Dict[str, Any]]]:
    """The review's flags [{i (0-based span index), tipo, perche}], or None when the call fails."""
    numbered = [(k, text[s:e].strip()) for k, (s, e) in enumerate(spans) if len(text[s:e].strip()) >= 25]
    if not numbered:
        return []
    listing = "\n".join(f"{n}. {sentence}" for n, (_, sentence) in enumerate(numbered, 1))
    human = (f"PARTE ASSISTITA: {party}\n\n" if party else "") + \
        f"CASO:\n{(case_text or '')[:12000]}\n\nSEZIONE «{title}»:\n{listing}"
    try:
        raw = dg._chat_json([SystemMessage(content=_REVIEW_SYSTEM), HumanMessage(content=human)],
                            max_tokens=200 + 60 * min(len(numbered), 30), temperature=0)
    except Exception as exc:
        _log_exception("_review_call", exc)
        logger.warning("writer rules: review of %r failed: %s", title, exc)
        return None
    out = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict) and str(item.get("i", "")).isdigit() and 1 <= int(item["i"]) <= len(numbered):
            out.append({"i": numbered[int(item["i"]) - 1][0], "tipo": str(item.get("tipo") or ""),
                        "perche": " ".join(str(item.get("perche") or "").split())[:200]})
    return out


def review_section(title: str, text: str, case_text: str, article_rows: Optional[List[Dict[str, Any]]] = None,
                   documents: Optional[List[Dict[str, Any]]] = None, party: str = "",
                   context: Optional[Dict[str, Any]] = None,
                   plan: Optional[Dict[str, Any]] = None) -> Tuple[str, List[Dict[str, Any]]]:
    """(text, log), WRITER_RULES_ENABLED: dropped, each sentence citing an article only to say it does not apply
    (the review call, and _NOT_APPLY_RE in the sentence; without the call, only a remark on relevance,
    _NOT_RELEVANT_RE) and each burden template whose norm allocates nothing (_burden_templates); flagged with
    _FACT_MARK (only with FACT_MARK_INLINE_ENABLED; otherwise logged), each sentence the review finds asserting a
    party's conduct, knowledge or intent the case does not hold, unless _fact_flag_excluded; dropped, each
    sentence proposing an action against the client (party) or a co-client. GROUNDED_EXCLUSION (with the plan): a
    sentence excluding an article on a ground grounded_exclusion accepts is not dropped as a non-application."""
    rows, documents = article_rows or [], documents or []
    known = {r["_article"] for r in rows if r.get("_article")}
    spans = _sentence_spans(text)
    if not spans:
        return text, []
    flags = _review_call(title, text, spans, case_text, party)
    drop: Dict[int, Tuple[str, str]] = {}
    mark: Dict[int, str] = {}
    filtered: List[Dict[str, Any]] = []
    def grounded(k):
        # the span alone, or with the one before it: the splitter ends a sentence at "575 c.p. (Omicidio) ..."
        if not plan:
            return False
        whole = [text[spans[k][0]:spans[k][1]]] + ([text[spans[k - 1][0]:spans[k][1]]] if k else [])
        return any(grounded_exclusion(w, plan, rows, context, documents, where=title) is not None for w in whole)
    for k, (s, e) in enumerate(spans):
        sentence = text[s:e]
        if flags is None and _cites(sentence) and _NOT_RELEVANT_RE.search(sentence) and not grounded(k):
            drop[k] = ("non_application", "a remark on relevance (no review call)")
    for flag in flags or []:
        k, sentence = flag["i"], text[spans[flag["i"]][0]:spans[flag["i"]][1]]
        if flag["tipo"] == "non_applicabile" and _cites(sentence) and _NOT_APPLY_RE.search(sentence) \
                and not grounded(k):
            drop[k] = ("non_application", flag["perche"])
        elif flag["tipo"] == "fatto_non_nel_caso":
            excluded = _fact_flag_excluded(sentence, case_text)
            if excluded:
                filtered.append({"section": title, "kind": "fact_flag_filtered", "how": "kept", "why": excluded,
                                 "sentence": sentence.strip()[:300]})
            else:
                mark[k] = flag["perche"]
        elif flag["tipo"] == "contro_assistito" and party:
            drop[k] = ("against_client", flag["perche"])
    for k, why in _burden_templates(title, text, spans, rows, documents, known).items():
        drop.setdefault(k, ("burden", why))
    for k, (s, e) in enumerate(spans):
        why = _wrong_statement(text[s:e], context)
        if why:
            drop.setdefault(k, ("wrong_statement", why))
    log = [{"section": title, "kind": kind, "how": "sentence dropped", "why": why,
            "sentence": text[spans[k][0]:spans[k][1]].strip()[:300]} for k, (kind, why) in sorted(drop.items())]
    inline = dg.feature_flags.enabled(FACT_MARK_FLAG)
    log += [{"section": title, "kind": "fact_not_in_case", "how": "marked" if inline else "logged", "why": why,
             "sentence": text[spans[k][0]:spans[k][1]].strip()[:300]} for k, why in sorted(mark.items()) if k not in drop]
    log += filtered
    marked = _mark_spans(text, [spans[k] for k in mark if k not in drop]) if inline else text
    # the marks shift the offsets: drop on the marked text by the same sentences, found again
    if drop:
        spans_after = _sentence_spans(marked)
        dropped = {text[spans[k][0]:spans[k][1]].strip() for k in drop}
        marked = _drop_spans(marked, [(s, e) for s, e in spans_after if marked[s:e].strip() in dropped])
    return marked, log


def standard_blocks(analysis: Dict[str, Any]) -> Dict[str, str]:
    """REVIEW_FIXES_ENABLED (round 3: the pool case's 589 c.p. never named, even after the rewrite): the block's
    standard text, appended to the procedural section when it is still missing after _cover_blocks -- for the
    criminal route, PROCEDURAL_TOOLS' text with the articles that apply to the case (589 and/or 590 c.p., 76
    c.p.p., 2947 co. 3 c.c.)."""
    from . import fact_signals
    if not _review_fixes():
        return {}
    tool = next((t for t in (_signals(analysis) or {}).get("tools") or []
                 if t.get("id") == fact_signals.CRIMINAL_TOOL), None)
    if not tool:
        return {}
    refs = ", ".join(f"art. {n}{', terzo comma,' if n == '2947' else ''} {c}" for n, c in tool.get("refs") or [])
    return {fact_signals.CRIMINAL_TOOL: f"**Profilo penale**\n\n{tool['text']}" + (f" Norme: {refs}." if refs else "")}


def _with_standard_blocks(sections: List[Tuple[str, str]], i: int,
                          coverage: Callable[[str], Dict[str, Dict[str, List[str]]]],
                          standard: Optional[Dict[str, str]]) -> List[Tuple[str, str]]:
    """The procedural section with the standard text of each block it still leaves out (standard_blocks)."""
    still = [b for b in _missing_blocks(coverage(sections[i][1])) if b in (standard or {})]
    if not still:
        return sections
    out = list(sections)
    out[i] = (sections[i][0], sections[i][1].rstrip() + "".join(f"\n\n{standard[b]}" for b in still))
    dg.vlog("case_analysis_standard_blocks", {"appended": still})
    return out


def _cover_blocks(sections: List[Tuple[str, str]], jobs: List[Tuple[str, str, str, int]], system: str,
                  coverage: Callable[[str], Dict[str, Dict[str, List[str]]]],
                  standard: Optional[Dict[str, str]] = None,
                  loops: Optional[List[Dict[str, Any]]] = None) -> List[Tuple[str, str]]:
    """The procedural section written once more when it leaves out a block of its material (_missing_blocks); the
    version that writes more items of the blocks is kept. A block still missing then gets its standard text
    (standard_blocks, REVIEW_FIXES_ENABLED)."""
    i = next((k for k, (title, _) in enumerate(sections) if title == SECTION_TITLES[4]), None)
    if i is None or not sections[i][1]:
        return sections
    use = coverage(sections[i][1])
    missing = _missing_blocks(use)
    if not missing:
        return sections
    title, instruction, material, max_tokens = jobs[i]
    lacking = "; ".join(f"{block}: " + ", ".join(sorted(set(u["offered"]) - set(u["written"])))
                        for block, u in use.items() if block in missing)
    try:
        again, loop = unloop(dg._call_chat([
            SystemMessage(content=f"{system}\n\n{instruction} La stesura precedente ometteva: {lacking}. Tratta "
                                  "ciascun blocco, uno per uno."),
            HumanMessage(content=material)], max_tokens=max_tokens))
        if loop and loops is not None:
            loops.append({"section": title, "call": "rewrite (blocks)", **{k: v for k, v in loop.items() if k != "at"}})
        again = _without_heading(again, title)
    except Exception as exc:
        _log_exception("_cover_blocks", exc)
        logger.warning("plan-then-write: procedural section not rewritten for its blocks: %s", exc)
        return _with_standard_blocks(sections, i, coverage, standard)

    def written(u):
        return sum(len(v["written"]) for v in u.values())
    if again and written(coverage(again)) > written(use):
        out = list(sections)
        out[i] = (title, again)
        return _with_standard_blocks(out, i, coverage, standard)
    return _with_standard_blocks(sections, i, coverage, standard)


def _complete_sections(sections: List[Tuple[str, str]], jobs: List[Tuple[str, str, str, int]],
                       system: str, looped: Sequence[str] = (),
                       loops: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[Tuple[str, str]], List[Dict[str, Any]]]:
    """WRITER_RULES_ENABLED: a section cut short (_looks_cut) written once more with half again its tokens; still
    cut, trimmed to its last complete sentence. WRITER_LOOP_GUARD: a looped section (`looped`, by title) is never
    written again, only trimmed; a rewrite that loops is cut like a section (unloop), into `loops`."""
    cut = [i for i, (title, text) in enumerate(sections) if _looks_cut(text) and title not in looped]
    log: List[Dict[str, Any]] = []
    out = list(sections)
    for i, (title, text) in enumerate(sections):
        if title in looped and _looks_cut(text):
            out[i] = (title, _trim_to_sentence(text))
            log.append({"section": title, "kind": "cut", "how": "looped: trimmed, not rewritten", "sentence": text[-160:]})
    sections = out
    if not cut:
        return sections, log

    def again(i):
        title, instruction, material, max_tokens = jobs[i]
        try:
            text, loop = unloop(dg._call_chat([SystemMessage(content=f"{system}\n\n{instruction}"),
                                               HumanMessage(content=material)], max_tokens=max_tokens * 3 // 2))
            if loop and loops is not None:
                loops.append({"section": title, "call": "rewrite (cut)", **{k: v for k, v in loop.items() if k != "at"}})
            return _without_heading(text, title)
        except Exception as exc:
            _log_exception("again", exc)
            logger.warning("writer rules: section %r cut, rewriting failed: %s", title, exc)
            return ""
    with ThreadPoolExecutor(max_workers=len(cut)) as pool:
        redone = dict(zip(cut, pool.map(again, cut)))
    out = list(sections)
    for i, text in redone.items():
        title, old = sections[i]
        if text and not _looks_cut(text):
            out[i], how = (title, text), "rewritten with more tokens"
        else:
            out[i], how = (title, _trim_to_sentence(text or old)), "trimmed to the last sentence"
        log.append({"section": title, "kind": "cut", "how": how, "sentence": old[-160:]})
        # the citation ledger: each article the cut section cited that its new version no longer does
        new = out[i][1]
        lost = [a for a in citation_ledger.cited(old) if a not in citation_ledger.cited(new)]
        for a in lost:
            first = next((old[s:e] for s, e in _sentence_spans(old) if a in citation_ledger.cited(old[s:e])), old)
            citation_ledger.record("complete_sections", first, f"cut: {how}", now=new, only=[a])
    return out, log


def review_parere(draft: str, case_text: str, article_rows: List[Dict[str, Any]], analysis: Dict[str, Any]) -> str:
    """WRITER_RULES_ENABLED on a parere written in one call (no plan): review_section over the whole draft."""
    try:
        text, log = review_section("parere", draft, case_text, article_rows, [], dg._stance_party(analysis),
                                   statement_context(analysis))
    except Exception as exc:
        _log_exception("review_parere", exc)
        logger.warning("writer rules: review of the one-call parere failed, draft kept: %s", exc)
        return draft
    dg.vlog("case_analysis_writer_rules", _writer_log(log))
    return text or draft


_guarded_step = None      # set below: _step, which write_sections wraps with the trace


def _writer_log(log: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"entries": log, "cut": sum(1 for e in log if e["kind"] == "cut"),
            "non_application": sum(1 for e in log if e["kind"] == "non_application"),
            "burden": sum(1 for e in log if e["kind"] == "burden"),
            "facts_flagged": sum(1 for e in log if e["kind"] == "fact_not_in_case"),
            "facts_filtered": dict(Counter(e["why"] for e in log if e["kind"] == "fact_flag_filtered")),
            "against_client": sum(1 for e in log if e["kind"] == "against_client")}


def _trace_comma_words(step: str, sections: List[Tuple[str, str]], rows: List[Dict[str, Any]]) -> None:
    """Logged as comma_words_trace: the sentences that carry a comma's words around or instead of a citation after each step,
    so a run shows the step they first appear in."""
    from . import text_tidy
    parts = [p for found, _ in text_tidy._parts_by_article(rows).values() for p in found]
    if not parts:
        return
    found = [text[s:e].strip()[:200] for _, text in sections for s, e in _sentence_spans(text)
             if text_tidy.comma_words_cited(text[s:e].strip(), parts) or text_tidy.comma_words_left(text[s:e].strip(), parts)]
    if found:
        dg.vlog("comma_words_trace", {"step": step, "count": len(found), "sentences": found[:10]})


def _step(name: str, sections: List[Tuple[str, str]],
          run: Callable[[List[Tuple[str, str]]], Optional[List[Tuple[str, str]]]]) -> List[Tuple[str, str]]:
    """One step after writing, fail-safe: its sections, or -- on an exception, an empty result or a section lost --
    the very sections it was given (the same object), logged as post_writing_step_failed. v60: a stage of the request's
    stage meter, when one is active (stage_meter.mark: a no-op otherwise)."""
    from .stage_meter import mark
    mark(f"review:{name}")
    try:
        out = run(sections)
        if out is None or len(out) != len(sections):
            raise ValueError(f"{len(out) if out is not None else 'no'} sections back for {len(sections)}")
        from . import answer_ledger         # v60f: what the step removed, when an answer's ledger is active; v60g:
        kept = answer_ledger.note(name, sections, out)      # with a deferring one, its removals put back (proposed)
        return kept if kept is not None and len(kept) == len(out) else out
    except Exception as exc:
        _log_exception("_step", exc)
        logger.exception("plan-then-write: step %s failed, the sections as they were: %s", name, exc)
        dg.vlog("post_writing_step_failed", {"step": name, "error": f"{type(exc).__name__}: {exc}"[:500]})
        return sections


_guarded_step = _step


def write_sections(plan: Dict[str, Any], jobs: List[Tuple[str, str, str, int]], lang: str,
                   named: Optional[List[tuple]] = None,
                   documents: Optional[List[Dict[str, Any]]] = None,
                   article_rows: Optional[List[Dict[str, Any]]] = None, case_text: str = "",
                   coverage: Optional[Callable[[str], Dict[str, Dict[str, List[str]]]]] = None,
                   standard: Optional[Dict[str, str]] = None,
                   questions: Optional[List[str]] = None,
                   checked: Optional[List[tuple]] = None,
                   anchors: Optional["anchored_writing.Catalogue"] = None,
                   context: Optional[Dict[str, Any]] = None, sections_note: str = "",
                   writer_rules: str = "") -> Optional[List[tuple]]:
    """[(title, text)] for the jobs, written in parallel; a failed section is tried once more.
    None if one still fails or comes back empty. _discipline(): each section then loses the citations
    its material does not hold (_cite_only_material; named: the articles the case names).
    _writer_rules(): a cut section is written again or trimmed, then each is reviewed (review_section).
    coverage (FACT_REGIMES_ENABLED: material_use): the procedural section, leaving out a block it was given, is
    written once more naming what it left out, and the version covering more is kept.
    Then (the live answer of 26 Oct): a sentence citing an adverse ruling (RULING_FIT_ENABLED) as support dropped;
    the fragments left without their norm dropped (drop_dangling, with any of the citation checks); an evidence
    list identical to an earlier one replaced by a reference to it, and the answers written once more when one of
    the explicit `questions` goes unanswered (REVIEW_FIXES_ENABLED). Last, GROUNDED_REVIEW_ENABLED: every
    rule-sentence checked against its section's material (grounded_review.review; `checked`, strict: the articles
    it may keep cited, grounded_review.strict_refs)."""
    system = _section_system(plan, lang, sections_note, writer_rules)

    def _step(name, secs, run):       # each step, and what it leaves of an article's words in place of its citation
        done = _guarded_step(name, secs, run)
        _trace_comma_words(name, done, article_rows or [])
        return done

    loops: List[Dict[str, Any]] = []        # WRITER_LOOP_GUARD: the sections whose reply looped
    # v48: every section's material within the model's context before its call (section_fit.fit); impossible, the
    # parere is written in one call
    rule_refs = sorted(rule_inserted(plan))
    fit_logs: List[Dict[str, Any]] = []
    try:
        fitted = []
        for title, instruction, material, max_tokens in jobs:
            material, fit_log = section_fit.fit(title, f"{system}\n\n{instruction}", material, max_tokens, plan,
                                                rule_refs, anchors)
            fit_logs.append(fit_log)
            fitted.append((title, instruction, material, max_tokens))
        jobs = fitted
    except section_fit.SectionTooLarge as exc:
        _log_exception("section_fit", exc)
        dg.vlog("case_analysis_section_fit", {"sections": fit_logs, "impossible": str(exc)})
        _fell_back("section fit", exc)
        return None
    if any(log["dropped"] for log in fit_logs):
        dg.vlog("case_analysis_section_fit", {"sections": [log for log in fit_logs if log["dropped"]]})
    refitted: Dict[str, str] = {}           # the sections refitted after a context-length error
    trimmed: set = set()                    # NORME_SPLIT: a half cut at its length, trimmed, never rewritten

    def call(title, instruction, material, max_tokens, half: str = ""):
        """One section call; on a context-length error the material refitted (section_fit.refit) and the call made
        once more, never with the identical prompt."""
        def once(material):
            messages = [SystemMessage(content=f"{system}\n\n{instruction}"), HumanMessage(content=material)]
            last_reply.finish_reason = None
            text = dg._call_chat(messages, max_tokens=max_tokens)
            finish = getattr(last_reply, "finish_reason", None)
            dg.vlog("case_analysis_section_call", {"section": title, "finish_reason": finish, "chars": len(text or ""),
                                                   "max_tokens": max_tokens, **({"half": half} if half else {})})
            if not (text or "").strip():
                raise ValueError(f"empty reply (finish_reason={finish})")
            text, loop = unloop(text)
            if loop:
                loops.append({"section": title, "call": "section" + (f" {half}" if half else ""),
                              **{k: v for k, v in loop.items() if k != "at"}})
            return text, finish
        try:
            return once(material)
        except Exception as exc:
            _log_exception("write", exc)
            sizes = ai_chat._overflow_sizes(str(exc), [SystemMessage(content=f"{system}\n\n{instruction}"),
                                                       HumanMessage(content=material)])
            if sizes:
                try:
                    material, log = section_fit.refit(title, f"{system}\n\n{instruction}", material, max_tokens, plan,
                                                      sizes, rule_refs, anchors)
                except section_fit.SectionTooLarge as none:
                    raise ValueError(f"section {title!r}: context exceeded, nothing left to drop ({none})") from exc
                dg.vlog("case_analysis_section_fit", {"sections": [{**log, **({"half": half} if half else {})}]})
                if not half:
                    refitted[title] = material
                logger.warning("plan-then-write: section %r over the context, refitted and tried once more", title)
            else:
                logger.warning("plan-then-write: section %r failed, trying once more: %s", title, exc)
            try:
                return once(material)
            except Exception as again:
                _log_exception("write", again)
                raise ValueError(f"section {title!r}: {again}") from again

    def write(job):
        title, instruction, material, max_tokens = job
        if title == SECTION_TITLES[2] and norme_split(plan):
            return _norme_halves(title, instruction, material, max_tokens)
        return call(title, instruction, material, max_tokens)[0]

    def _norme_halves(title, instruction, material, max_tokens):
        """NORME_SPLIT: the section in two concurrent calls over disjoint halves of the plan's articles (the rulings
        with the first); a half cut at its length trimmed to its last sentence, never written again."""
        refs = _norme_refs(plan)
        first, second = refs[:(len(refs) + 1) // 2], refs[(len(refs) + 1) // 2:]
        materials = section_fit.halves(material, first, second, anchors)
        tokens = max(400, int(max_tokens * NORME_SPLIT_SHARE))

        def _norme_half(k):
            own = (first, second)[k]
            words = re.sub(r"Al massimo (\d+) parole",
                           lambda m: f"Al massimo {int(int(m.group(1)) * NORME_SPLIT_SHARE)} parole", instruction)
            ask = (f"{words} In questa parte della sezione tratta SOLO queste norme del PIANO, una per una: "
                   + ", ".join(f"art. {n} {_code_key(c)}" for n, c in own)
                   + "; le altre norme del PIANO sono trattate in un'altra parte: non nominarle."
                   + ("" if k == 0 else " Non citare pronunce. Non scrivere un'introduzione: comincia dalla prima norma."))
            text, finish = call(title, ask, materials[k], tokens, half=f"{k + 1}/2")
            cut = finish == "length"
            if cut:
                text = _trim_to_sentence(text)
            return {"text": text, "finish_reason": finish, "trimmed": cut, "articles": [f"{n} {c}" for n, c in own],
                    "chars": len(text)}
        with ThreadPoolExecutor(max_workers=2) as pool:
            done = list(pool.map(_norme_half, (0, 1)))
        if any(d["trimmed"] for d in done):
            trimmed.add(title)
        dg.vlog("case_analysis_norme_split", {"halves": [{k: v for k, v in d.items() if k != "text"} for d in done],
                                              "max_tokens": tokens})
        return "\n\n".join(d["text"].strip() for d in done if d["text"].strip())

    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            texts = list(pool.map(write, jobs))
    except Exception as exc:
        _log_exception("write_sections", exc)
        logger.warning("plan-then-write: a section failed twice, the parere is written in one call: %s", exc)
        _fell_back("section call", exc)
        return None
    finally:
        dg.vlog("case_analysis_sections_llm", duration_ms=(time.time() - t0) * 1000)
    if refitted:
        jobs = [(t, i, refitted.get(t, m), k) for t, i, m, k in jobs]
    sections = [(job[0], _without_heading(text, job[0])) for job, text in zip(jobs, texts)]
    # v51, no flag: within a section, a sentence an earlier one already says goes (repetition_guard)
    sections = _step("repetition_guard", sections, lambda secs: repetition_guard.guard_sections(secs, "written")[0])
    if dg.feature_flags.enabled(USUCAPIONE_CONDITIONAL_FLAG):
        sections = _step("usucapione_conditional", sections, lambda secs: usucapione_sections(secs, context, "written"))
    _trace_comma_words("written", sections, article_rows or [])
    looped = {entry["section"] for entry in loops} | trimmed
    # every step after writing is fail-safe (_step): an exception keeps the sections as they were before it
    if coverage is not None:
        sections = _step("cover_blocks", sections,
                         lambda secs: _cover_blocks(secs, jobs, system, coverage, standard, loops=loops))
    writer_log: List[Dict[str, Any]] = []
    if _writer_rules() and all(text for _, text in sections):
        def complete(secs):
            done, log = _complete_sections(secs, jobs, system, looped=looped, loops=loops)
            writer_log.extend(log)
            return done
        sections = _step("complete_sections", sections, complete)
    if loops:
        dg.vlog("case_analysis_writer_loops", {"loops": loops, "sections": sorted({e["section"] for e in loops})})
    if _discipline() and all(text for _, text in sections):
        def discipline(secs):
            t1 = time.time()
            with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                done = list(pool.map(lambda pair: _cite_only_material(pair[0][0], pair[1], pair[0][2], named or [],
                                                                      documents or []), zip(jobs, (t for _, t in secs))))
            removed = [entry for _, log in done for entry in log]
            # the dangling fragments dropped with a removal (drop_dangling) carry no "removed" list
            dg.vlog("case_analysis_citation_discipline", {
                "removed": removed, "count": sum(len(e.get("removed") or []) for e in removed),
                "rephrased": sum(1 for e in removed if e.get("how") == "rephrased"),
                "dropped": sum(1 for e in removed if e.get("how") == "sentence dropped" and e.get("kind") != "dangling"),
                "dangling": sum(1 for e in removed if e.get("kind") == "dangling")},
                duration_ms=(time.time() - t1) * 1000)
            citation_ledger.record_entries("discipline", removed)
            return [(title, text or original) for (title, original), (text, _) in zip(secs, done)]
        sections = _step("citation_discipline", sections, discipline)
    if _writer_rules() and all(text for _, text in sections):
        def review(secs):      # the writer rules' review, the role check (sentences against the client) among them
            t2 = time.time()
            with ThreadPoolExecutor(max_workers=len(secs)) as pool:
                reviewed = list(pool.map(lambda pair: review_section(pair[0], pair[1], case_text, article_rows,
                                                                     documents, plan.get("parte_assistita") or "",
                                                                     context, plan=plan),
                                         secs))
            writer_log.extend(entry for _, log in reviewed for entry in log)
            dg.vlog("case_analysis_writer_rules", _writer_log(writer_log), duration_ms=(time.time() - t2) * 1000)
            citation_ledger.record_entries("writer_review", [e for _, log in reviewed for e in log])
            return [(title, text or original) for (title, original), (text, _) in zip(secs, reviewed)]
        sections = _step("writer_review", sections, review)
    if _review_fixes() and all(text for _, text in sections):
        def fixes(secs):
            t3 = time.time()
            with ThreadPoolExecutor(max_workers=len(secs)) as pool:
                fixed = list(pool.map(lambda pair: rewrite_unsupported(pair[0], pair[1], case_text, article_rows),
                                      secs))
            entries = [entry for _, log in fixed for entry in log]
            dg.vlog("case_analysis_unsupported_claims", {
                "entries": entries, "rewritten": sum(1 for e in entries if e.get("how") == "rewritten")},
                duration_ms=(time.time() - t3) * 1000)
            return [(title, text or original) for (title, original), (text, _) in zip(secs, fixed)]
        sections = _step("unsupported_claims", sections, fixes)
    if any(not text for _, text in sections):
        logger.warning("plan-then-write: an empty section, the parere is written in one call")
        _fell_back("empty section after the checks", "a step left a section without text",
                   sections=[title for title, text in sections if not text])
        return None
    sections = _step("after_checks", sections, lambda secs: _after_checks(plan, secs, jobs, system, questions or []))
    if anchors is not None:
        # ANCHORED_WRITING_ENABLED: each rule-sentence checked against its anchors' passages; failing, rewritten once
        # from them, kept or removed by what it cites; the anchors stripped (anchored_writing.py). On a failure the
        # sections as they were, only without their anchors (never shown to a reader).
        t5 = time.time()

        def anchored(secs):
            verified = anchored_writing.verify(secs, anchors, documents or [], article_rows or [],
                                               exempt=list((standard or {}).values()), case_text=case_text,
                                               materials={job[0]: job[2] for job in jobs})
            return [(title, text or original) for (title, original), (_, text) in zip(secs, verified)]
        before = sections
        sections = _step("anchored_writing", sections, anchored)
        if sections is before:
            sections = [(title, anchored_writing.strip_anchors(text)) for title, text in sections]
        dg.vlog("anchored_writing_llm", duration_ms=(time.time() - t5) * 1000)
    if grounded_review.enabled():
        # GROUNDED_REVIEW_ENABLED: each rule-sentence judged against its own section's material; the ones it does
        # not hold rewritten from it, as pointers, or (filler) dropped (grounded_review.py).
        t4 = time.time()

        def grounded(secs):
            reviewed = grounded_review.review(secs, jobs, documents or [], article_rows or [], checked=checked,
                                              topics=[q.get("titolo") or "" for q in plan.get("questioni") or []])
            return [(title, text or original) for (title, original), (_, text) in zip(secs, reviewed)]
        sections = _step("grounded_review", sections, grounded)
        dg.vlog("grounded_review_llm", duration_ms=(time.time() - t4) * 1000)
    return sections


_OBJECTION_RE = re.compile(r"controparte|obiezion|obiett|potrebbe (?:invocare|sostenere|richiamare)|contrari|sfavorevol|"
                           r"replica|tuttavia|diversamente|non si attaglia|si distingue|a differenza|invece|"
                           r"non (?:è|sono) (?:applicabil|pertinent)", re.IGNORECASE)


def adverse_as_support(title: str, text: str, adverse: List[str]) -> Tuple[str, List[Dict[str, Any]]]:
    """(text, log): each sentence citing an adverse ruling (by its number, "14015/2026") with no word of an
    objection or a reply dropped: an adverse ruling is only the opponent's argument."""
    numbers = {n for ref in adverse for n in re.findall(r"\b\d{1,6}/\d{4}\b", ref)}
    if not numbers:
        return text, []
    drop, log = [], []
    for s, e in _sentence_spans(text):
        sentence = text[s:e]
        hit = sorted(n for n in numbers if re.search(rf"(?<![\d/]){re.escape(n)}(?![\d/])", sentence))
        if hit and not _OBJECTION_RE.search(sentence):
            drop.append((s, e))
            log.append({"section": title, "kind": "adverse_as_support", "how": "sentence dropped", "ruling": hit,
                        "sentence": " ".join(sentence.split())[:300]})
    return (_drop_spans(text, drop) if drop else text), log


_EVIDENCE_HEAD_RE = re.compile(r"^\s*(?:#{1,6}\s*)?[*_]*\s*prove\s+da\s+acquisire\s*:?\s*[*_]*\s*:?\s*$", re.IGNORECASE)


def dedupe_evidence(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """(text, log): an evidence list ("Prove da acquisire:" and its items) identical to an earlier one replaced by
    one line referring to the issue where it first appears."""
    lines, out, log = text.split("\n"), [], []
    seen: Dict[tuple, str] = {}
    heading, i = "", 0
    while i < len(lines):
        line = lines[i]
        m = _HEADING_LINE_RE.match(line)
        if m and not m.group(2).strip() and not _EVIDENCE_HEAD_RE.match(line):
            heading = re.sub(r"^[#*_\s]+|[*_\s]+$", "", line)
        if not _EVIDENCE_HEAD_RE.match(line):
            out.append(line)
            i += 1
            continue
        j, items = i + 1, []
        while j < len(lines) and (not lines[j].strip() or _LIST_ITEM_RE.match(lines[j])):
            if lines[j].strip():
                items.append(" ".join(_LIST_ITEM_RE.sub("", lines[j]).lower().split()).rstrip(".;"))
            j += 1
        key = tuple(sorted(items))
        if items and key in seen:
            out.append(f"Prove da acquisire: le stesse indicate per «{seen[key]}»." if seen[key]
                       else "Prove da acquisire: le stesse indicate sopra.")
            if j < len(lines):
                out.append("")
            log.append({"kind": "evidence_repeated", "how": "list replaced by a reference", "under": heading,
                        "same_as": seen[key], "items": len(items)})
        else:
            if items:
                seen[key] = heading
            out.extend(lines[i:j])
        i = j
    return ("\n".join(out) if log else text), log


def _cover_questions(sections: List[Tuple[str, str]], jobs: List[Tuple[str, str, str, int]], system: str,
                     questions: List[str]) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    """The answers section written once more when it leaves an explicit question unanswered (_answers); the version
    answering more is kept."""
    i = next((k for k, (title, _) in enumerate(sections) if title == SECTION_TITLES[0]), None)
    if i is None or not questions:
        return sections, {}
    missing = [q for q in questions if not _answers(q, sections[i][1])]
    log: Dict[str, Any] = {"questions": questions, "unanswered": missing, "rewritten": False}
    if not missing:
        return sections, log
    title, instruction, material, max_tokens = jobs[i]
    try:
        again = _without_heading(dg._call_chat([
            SystemMessage(content=f"{system}\n\n{instruction} La stesura precedente non rispondeva a: "
                                  + " | ".join(missing) + ". Rispondi a ciascun quesito, uno per uno."),
            HumanMessage(content=material)], max_tokens=max_tokens), title)
    except Exception as exc:
        _log_exception("_cover_questions", exc)
        logger.warning("review fixes: answers not rewritten for the questions: %s", exc)
        return sections, log
    still = [q for q in questions if not _answers(q, again or "")]
    if again and len(still) < len(missing):
        out = list(sections)
        out[i] = (title, again)
        log.update(rewritten=True, still_unanswered=still)
        return out, log
    log["still_unanswered"] = missing
    return sections, log


def _after_checks(plan: Dict[str, Any], sections: List[Tuple[str, str]], jobs: List[Tuple[str, str, str, int]],
                  system: str, questions: List[str]) -> List[Tuple[str, str]]:
    """The checks on the written sections of the live answer of 26 Oct (write_sections' docstring)."""
    adverse = plan.get("pronunce_contrarie") or []
    if adverse:
        done = [adverse_as_support(title, text, adverse) for title, text in sections]
        entries = [e for _, log in done for e in log]
        dg.vlog("case_analysis_adverse_rulings", {"adverse": adverse, "entries": entries, "dropped": len(entries)})
        sections = [(title, text or original) for (title, original), (text, _) in zip(sections, done)]
    if _discipline() or _writer_rules() or dg.feature_flags.enabled("FLAGGED_REMOVE_ENABLED"):
        done = [drop_dangling(text, title) for title, text in sections]
        entries = [e for _, log in done for e in log]
        dg.vlog("case_analysis_dangling", {"entries": entries, "count": len(entries)})
        sections = [(title, text or original) for (title, original), (text, _) in zip(sections, done)]
    if _review_fixes():
        i = next((k for k, (title, _) in enumerate(sections) if title == SECTION_TITLES[3]), None)
        if i is not None:
            text, log = dedupe_evidence(sections[i][1])
            if log:
                dg.vlog("case_analysis_evidence_dedup", {"entries": log})
                sections = sections[:i] + [(sections[i][0], text)] + sections[i + 1:]
        if questions:
            sections, log = _cover_questions(sections, jobs, system, questions)
            dg.vlog("case_analysis_questions", log)
    return sections


# REVIEW_FIXES_ENABLED: the sentences that may state as certain what the case does not say -- a deadline met,
# missed or not yet run, a report made or not, a party's proof, knowledge or concealment (the boat case: "i vizi
# non sono stati denunciati entro il termine di 8 giorni"; the pool case: "poiché l'incidente è recente, il diritto
# non è ancora prescritto"). Only these go to the check; the review's own fact flag missed both.
_UNSUPPORTED_RE = re.compile(
    r"\b(?:è|sono|sia|siano|era|erano)\s+(?:ancora\s+|già\s+)?(?:prescritt|decadut|decors|scadut)\w*"
    r"|\bentro\s+(?:il\s+)?termin|\bnei\s+termini\b|\btempestiv|\bin\s+tempo\s+utile|\brecent"
    r"|\b(?:è|sono)\s+stat[aeio]\s+(?:denunciat|contestat|comunicat|segnalat)\w*"
    r"|\bnon\s+ha\s+(?:mai\s+)?(?:dimostrato|provato|fornito\s+(?:la\s+)?prova|adottato|denunciato|contestato|"
    r"segnalato)|\bha\s+(?:occultato|nascosto|taciuto|reso\s+difficile)|\bera\s+(?:a\s+conoscenza|consapevole)"
    r"|\bsapeva\b|\bin\s+mala\s+fede"
    # round 1 (26 Oct): "poiché Tizietta è deceduta immediatamente, non si trasmette alcun danno da morte agli eredi"
    r"|\b(?:mort|decedut|deceduto|decesso|annegat)\w*\b[^.;]{0,40}\b(?:immediat|sul\s+colpo|istantane|subito)"
    r"|\b(?:immediat\w*|sul\s+colpo|istantane\w*)\b[^.;]{0,40}\b(?:mort|deced|decesso|annegat)", re.IGNORECASE)
_UNSUPPORTED_SYSTEM = (
    "Sei un avvocato italiano e rivedi un parere. Ti vengono dati il CASO e alcune frasi del parere, numerate. Una "
    "frase è da correggere solo se afferma come certo, riguardo ai fatti di questo caso, qualcosa che il CASO non "
    "dice: che un termine sia stato rispettato, sia scaduto o non sia ancora decorso; che una denuncia, una "
    "contestazione o una comunicazione sia o non sia avvenuta, o quando; quando sono accaduti i fatti; che una "
    "parte abbia o non abbia provato, saputo, nascosto o taciuto qualcosa; quanto tempo una persona sia rimasta "
    "in vita o cosciente. Non è da correggere una frase che "
    "enuncia una regola, un termine di legge o un onere della prova, che pone una condizione o che riporta ciò che "
    "il CASO dice. Riscrivi ogni frase da correggere presentando quel punto come condizione o come circostanza da "
    "verificare (\"se la denuncia è avvenuta entro otto giorni dalla scoperta...\", \"spetterà al venditore "
    "provare di avere ignorato senza colpa i vizi\"), lasciando invariati il resto della frase e le norme e le "
    "pronunce che cita. Rispondi solo con un array JSON, un oggetto per frase: {\"i\": numero, "
    "\"da_correggere\": true o false, \"frase\": \"la frase riscritta, solo se da correggere\"}."
)
_CLAIM_REFS_RE = re.compile(r"\bartt?\.\s*(\d+(?:-[a-z]+)?)|\bn\.\s*(\d+/\d{4})", re.IGNORECASE)


def _claim_refs(sentence: str) -> set:
    return {a or b for a, b in _CLAIM_REFS_RE.findall(sentence or "")}


# Round 3 (the boat case v7): the check rewrote a correct rule -- "La denuncia non è necessaria se il venditore ha
# riconosciuto ... o l'ha occultato" (art. 1495 co. 2) got a condition appended. A sentence stating a condition, or
# restating the text of an article it cites, is never sent to the check.
_CONDITION_RE = re.compile(r"\b(?:se|qualora|ove|purché|purche|salvo che|nel caso in cui|a meno che)\b",
                           re.IGNORECASE)
_VERBATIM_SHARE = 0.7


def _content_stems(text: str) -> set:
    return {w[:6] for w in re.findall(r"[a-zàèéìòù]{5,}", (text or "").lower())}


def _states_a_rule(sentence: str, article_rows: Optional[List[Dict[str, Any]]]) -> bool:
    """A condition marker, or most of the sentence's content words in the text of an article it cites (a quote
    or a near-verbatim restatement of the norm)."""
    if _CONDITION_RE.search(sentence):
        return True
    cited = _claim_refs(sentence)
    stems = _content_stems(re.sub(r"\bartt?\.\s*\d+\S*|\b[cC]\.\s?[cpCP]\.\S*", " ", sentence))
    if not cited or len(stems) < 4:
        return False
    for row in article_rows or []:
        number = str((row.get("_article") or ("",))[0])
        if number and number in cited:
            text = (row.get("s") or {}).get("plain_text") or ""
            if len(stems & _content_stems(text)) >= _VERBATIM_SHARE * len(stems):
                return True
    return False


def rewrite_unsupported(title: str, text: str, case_text: str,
                        article_rows: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, List[Dict[str, Any]]]:
    """(text, log), REVIEW_FIXES_ENABLED: the sentences matching _UNSUPPORTED_RE that do not restate the case
    (_from_the_case) nor state a rule or a condition (_states_a_rule: never rewritten, round 3), checked in one
    call; each the check finds stating as certain what the case does not say is replaced by its rewriting as a
    condition -- kept as it was when the rewriting is empty, loses an article or a ruling the sentence cites, or
    is out of proportion with it."""
    spans = [(s, e) for s, e in _sentence_spans(text)
             if _UNSUPPORTED_RE.search(text[s:e]) and not _from_the_case(text[s:e], case_text)
             and not _states_a_rule(text[s:e], article_rows)]
    if not spans:
        return text, []
    numbered = "\n".join(f"{k}. {' '.join(text[s:e].split())}" for k, (s, e) in enumerate(spans, 1))
    try:
        verdicts = dg._chat_json([SystemMessage(content=_UNSUPPORTED_SYSTEM),
                                  HumanMessage(content=f"CASO:\n{case_text[:3000]}\n\nFRASI DEL PARERE:\n{numbered}")],
                                 max_tokens=300 + 240 * len(spans), temperature=0)
    except Exception as exc:
        _log_exception("rewrite_unsupported", exc)
        logger.warning("review fixes: claims of %r not checked: %s", title, exc)
        return text, []
    replace: Dict[int, str] = {}
    log: List[Dict[str, Any]] = []
    for v in verdicts if isinstance(verdicts, list) else []:
        if not isinstance(v, dict) or not _yes(v.get("da_correggere")):
            continue
        try:
            k = int(v.get("i")) - 1
        except (TypeError, ValueError):
            continue
        if not 0 <= k < len(spans) or k in replace:
            continue
        old = text[spans[k][0]:spans[k][1]]
        new = " ".join(str(v.get("frase") or "").split())
        usable = bool(new) and _claim_refs(old) <= _claim_refs(new) and 0.5 * len(old) <= len(new) <= 2.5 * len(old)
        if usable:
            replace[k] = new
        log.append({"section": title, "kind": "unsupported_claim", "how": "rewritten" if usable else "kept",
                    "sentence": " ".join(old.split())[:300], "rewritten": new[:300]})
    for k in sorted(replace, reverse=True):
        s, e = spans[k]
        text = text[:s] + replace[k] + text[e:]
    citation_ledger.record_entries("rewrite_unsupported", [{**x, "now": x["rewritten"]} for x in log
                                                           if x.get("how") == "rewritten"])
    return text, log


# PLAN_ARTICLES_COVER_ENABLED (v37: 1382, 1079, 191, 582, 320 c.p., 21-quinquies and 3 l. 241, 252 TUEL, 1218 planned, never
# written): a plan article graded 2 or more that no section cites, or one never sent to the grader (the case names it, its
# rubrica matched, a cross-reference), gets one short call: one or two sentences for its issue, from its text.
COVER_FLAG = "PLAN_ARTICLES_COVER_ENABLED"
_COVER_TOKENS = 260
_COVER_TEXT_CHARS = 1500
_COVER_SYSTEM = (
    "Sei un avvocato italiano e completi un parere: una norma del piano non è stata trattata. Scrivi una o due frasi, in "
    "prosa, su come la norma serve la questione indicata per i fatti del CASO. Fondati solo sul TESTO DELLA NORMA, senza "
    "aggiungere regole, fatti o date. Cita la norma una volta, esattamente come indicata; nessun altro articolo, nessuna "
    "pronuncia. Non scrivere mai che la norma non si applica o non è pertinente. Rispondi solo con le frasi.")


def covering() -> bool:
    return dg.feature_flags.enabled(COVER_FLAG)


def _worth_covering(rows: List[Dict[str, Any]]) -> bool:
    """Graded 2 or more: never one with no grade (v42: 2160, 1052 and 1038 c.c. written into a servitude case)."""
    return any(isinstance(r.get("_grade"), int) and r["_grade"] >= 2 for r in rows)


def _omitted(plan: Dict[str, Any], sections: List[Tuple[str, str]], article_rows: List[Dict[str, Any]],
             documents: List[Dict[str, Any]], excluded: Sequence[tuple] = (),
             conclusions: Sequence[str] = ()) -> List[Tuple[tuple, Dict[str, Any], Dict[str, Any]]]:
    """[(article, its issue, its row)] of the plan's articles worth covering that no section cites and no fired rule's
    conclusion excludes; then (PARTIES_ENFORCED) each article a line of the conclusions block cites that no section
    cites, whatever its grade, the line as its issue (its row {} when none was read: _cover_one refuses, "no text")."""
    known = {r["_article"] for r in article_rows if r.get("_article")}
    cited = [r for _, text in sections for _, _, refs in _citations(text, documents, known) for r in refs]
    out = []
    by_rule = rule_inserted(plan)        # PLAN_RULE_INCLUDE: the writer may use them, no paragraph is forced
    for ref in _plan_refs(plan):
        if ref in by_rule:
            continue
        rows = [r for r in article_rows if r.get("_article") and _same_ref(r["_article"], ref)]
        if rows and not any(_same_ref(ref, c) for c in cited) and _worth_covering(rows) \
                and not any(_same_ref(ref, tuple(x)) for x in excluded):
            out.append((ref, next(q for q in plan["questioni"] if ref in q["norme"]), rows[0]))
    held = _plan_refs(plan)
    for line in conclusions:
        for _, _, refs in _citations(line, documents, known):
            for ref in refs:
                if any(_same_ref(ref, c) for c in cited + held + [o[0] for o in out]) \
                        or any(_same_ref(ref, tuple(x)) for x in excluded):
                    continue
                rows = [r for r in article_rows if r.get("_article") and _same_ref(r["_article"], ref)]
                out.append((ref, {"titolo": _CONCLUSIONS_TITLE.capitalize(), "argomento": line, "conclusione": True},
                            rows[0] if rows else {}))
    return out


def conclusion_lines(analysis: Dict[str, Any]) -> List[str]:
    """PARTIES_ENFORCED: the conclusions block's lines, for the coverage check (_omitted); [] with the flag off."""
    from . import fact_signals
    if not dg.feature_flags.enabled(fact_signals.PARTIES_ENFORCED_FLAG):
        return []
    return [line[2:] if line.startswith("- ") else line for line in _conclusions(analysis).split("\n") if line.strip()]


_NOT_CONNECTED_RE = re.compile(r"\bnon\s+(?:si\s+)?(?:ricollega|riguarda|attiene|disciplina|concerne|rileva|incide)\b|\bestrane[aoie]\b|"
                               r"\bnon\s+(?:è|sono)\s+(?:direttamente|strettamente)\b|\bsolo\s+indirettamente\b", re.IGNORECASE)


# GROUNDED_EXCLUSION (v46, p1 in v45: plan cover's "L'art. 575 c.p. ... non si applica ... ai sensi dell'art. 55 c.p."
# refused for citing 55): excluding an article on a ground is an argument, not a remark on relevance. A sentence saying
# article A does not apply is accepted, by plan cover and by the writer review, when A is in the plan with grade 2 or more,
# the ground is another article B of the plan the sentence cites, or a fired fact rule whose conclusion excludes A, and
# the sentence cites nothing but A and B. Every other "does not apply" sentence is refused (cover) or dropped (review) as
# before. Logged as grounded_exclusion_accepted.
GROUNDED_EXCLUSION_FLAG = "GROUNDED_EXCLUSION"
# "cites nothing but A and B" read strictly: every number before a code sigla, and every number of an "artt." list, which
# read_citations and _refs_in miss after the first ("degli artt. 55 c.p. e 52 c.p.", "artt. 55 e 52 c.p.").
_SIGLA = r"c\.\s?p\.\s?p\.|c\.\s?p\.\s?c\.|c\.\s?p\.\s?a\.|c\.\s?p\.|c\.\s?c\."
_NUMBER_SIGLA_RE = re.compile(r"\b(\d+(?:-[a-z]+)?)\s*,?\s*(" + _SIGLA + r")", re.IGNORECASE)
_ARTT_LIST_RE = re.compile(r"\bartt\.\s*((?:\d+(?:-[a-z]+)?(?:\s*(?:" + _SIGLA + r"))?\s*(?:,|\be\b|\bed\b)\s*)+"
                           r"\d+(?:-[a-z]+)?)\s*(" + _SIGLA + r")", re.IGNORECASE)


def _strict_refs(sentence: str) -> List[tuple]:
    text = sentence or ""
    out = [(m.group(1), m.group(2)) for m in _NUMBER_SIGLA_RE.finditer(text)      # not "comma 2, c.p.", "n. 1, c.c."
           if not re.search(r"\b(?:comma|co\.|n\.|nn\.|lett\.|numero)\s*$", text[:m.start()], re.IGNORECASE)]
    for m in _ARTT_LIST_RE.finditer(text):
        out += [(n, m.group(2)) for n in re.findall(r"\b\d+(?:-[a-z]+)?\b", m.group(1))]
    return out


def grounded_exclusion(sentence: str, plan: Optional[Dict[str, Any]], article_rows: List[Dict[str, Any]],
                       context: Optional[Dict[str, Any]], documents: Optional[List[Dict[str, Any]]] = None,
                       excluded: Optional[tuple] = None, where: str = "plan cover") -> Optional[Dict[str, Any]]:
    """{"A", "B", "B_ref", "rule", "grade"} when GROUNDED_EXCLUSION accepts the sentence's exclusion of an article (A:
    `excluded` when given, else the first article the sentence cites that qualifies), else None."""
    if not dg.feature_flags.enabled(GROUNDED_EXCLUSION_FLAG) or not plan:
        return None
    if not (_NOT_APPLY_RE.search(sentence or "") or _NOT_RELEVANT_RE.search(sentence or "")
            or _NOT_CONNECTED_RE.search(sentence or "")):
        return None
    rows = article_rows or []
    known = {r["_article"] for r in rows if r.get("_article")}
    refs = [r for _, _, _, rs, _ in read_citations(sentence, documents or [], known) for r in rs]
    if not refs or any(r is None for r in refs):
        return None
    distinct: List[tuple] = []
    for r in refs + _strict_refs(sentence):
        if not any(_same_ref(r, d) for d in distinct):
            distinct.append(r)
    if len(distinct) > 2:
        return None
    planned = [tuple(r) for q in plan.get("questioni") or [] for r in q.get("norme") or []]

    def in_plan(r):
        return any(_same_ref(r, p) for p in planned)

    def grade(r):
        grades = [row["_grade"] for row in rows if row.get("_article") and _same_ref(row["_article"], r)
                  and isinstance(row.get("_grade"), int)]
        return max(grades) if grades else None
    candidates = [d for d in distinct if excluded is None or _same_ref(d, excluded)]
    for a in candidates:
        g = grade(a)
        if not in_plan(a) or (g or 0) < 2:
            continue
        others = [d for d in distinct if not _same_ref(d, a)]
        accepted = None
        if others:
            if in_plan(others[0]):
                accepted = {"B": f"{others[0][0]} {others[0][1]}", "B_ref": others[0], "rule": None}
        else:
            rule = next((x.get("rule") for x in (context or {}).get("excluded_by") or []
                         if _same_ref(tuple(x.get("article") or ("", "")), a)), None)
            if rule:
                accepted = {"B": None, "B_ref": None, "rule": rule}
        if accepted:
            out = {"A": f"{a[0]} {a[1]}", "grade": g, **accepted}
            dg.vlog("grounded_exclusion_accepted", {**{k: v for k, v in out.items() if k != "B_ref"}, "where": where,
                                                    "sentence": " ".join(sentence.split())[:200]})
            return out
    return None


def _cover_refusal(text: str, ref: tuple, documents: List[Dict[str, Any]], article_rows: List[Dict[str, Any]],
                   plan: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None) -> str:
    """Why these sentences cannot go in, "" when they can: they must cite the article and, besides it, only the plan's
    articles (v49: 320 c.p. explained through 318 and 319, both in the plan), and not say it does not apply -- unless
    GROUNDED_EXCLUSION accepts them (grounded_exclusion: the article excluded on a ground, the ground's article B then
    allowed too)."""
    known = {r["_article"] for r in article_rows if r.get("_article")}
    grounded = grounded_exclusion(text, plan, article_rows, context, documents, excluded=ref) if plan else None
    allowed = [ref] + ([grounded["B_ref"]] if grounded and grounded.get("B_ref") else []) + (_plan_refs(plan) if plan else [])
    refs = [r for _, _, rs in _citations(text, documents, known) for r in rs]
    if not refs or not any(_same_ref(r, ref) for r in refs) or not all(any(_same_ref(r, a) for a in allowed) for r in refs) \
            or any(r is None for _, _, _, rs, _ in read_citations(text, documents, known) for r in rs):
        return "cites another article, or not this one"
    if dg._decision_numbers(text):
        return "cites a ruling"
    if not grounded and (_NOT_APPLY_RE.search(text) or _NOT_RELEVANT_RE.search(text) or _NOT_CONNECTED_RE.search(text)):
        return "says it does not apply"
    from . import comma_material, text_tidy
    rows = [r for r in article_rows if r.get("_article") and _same_ref(r["_article"], ref)]
    parts = [p for r in rows for p in comma_material.article_parts((r.get("s") or {}).get("plain_text"))[0]]
    if text_tidy.comma_words_cited(text, parts) or text_tidy.comma_words_left(text, parts):
        return "copies the comma's words"
    return ""


def _cover_one(ref: tuple, issue: Dict[str, Any], row: Dict[str, Any], plan: Dict[str, Any], documents: List[Dict[str, Any]],
               article_rows: List[Dict[str, Any]], case_text: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """{"article", "issue", "how": "inserted", "sentence"} or {..., "how": "refused", "why"}: the call, then the checks of
    every written sentence on the reply (review_section, rewrite_unsupported, the grounded judge, the qualification
    check); a failed call or a check that cannot answer refuses."""
    from . import answer_invariants
    label = f"art. {ref[0]} {ref[1]}"
    entry = {"article": f"{ref[0]} {ref[1]}", "issue": issue.get("titolo") or "", "how": "refused"}
    body = " ".join(str((row.get("s") or {}).get("plain_text") or "").split())
    if not body:
        return {**entry, "why": "no text"}
    source = comma_material.trim(body, _COVER_TEXT_CHARS, f"{issue.get('titolo') or ''} {issue.get('argomento') or ''}")
    heading = article_text.heading(row.get("s"))
    human = (f"CASO:\n{case_text[:3000]}\n\nQUESTIONE: {issue.get('titolo') or ''}\n{issue.get('argomento') or ''}\n\n"
             f"NORMA DA CITARE: {label}" + (f" ({heading})" if heading else "") + f"\n\nTESTO DELLA NORMA:\n{source}")
    try:
        reply = dg._call_chat([SystemMessage(content=_COVER_SYSTEM), HumanMessage(content=human)], max_tokens=_COVER_TOKENS)
    except Exception as exc:
        _log_exception("_cover_one", exc)
        return {**entry, "why": f"call failed: {type(exc).__name__}"}
    text = " ".join((reply or "").split())
    spans = _sentence_spans(text)[:2]
    text = text[:spans[-1][1]].strip() if spans else ""
    if not text or _looks_cut(text):
        return {**entry, "why": "empty or cut reply", "sentence": text[:300]}
    why = _cover_refusal(text, ref, documents, article_rows, plan, context)
    if why:
        return {**entry, "why": why, "sentence": text[:300]}
    title = SECTION_TITLES[3]
    reply = text
    if _writer_rules():
        text, review = review_section(title, text, case_text, article_rows, documents, plan.get("parte_assistita") or "",
                                      context, plan=plan)
        # v46: a flag acts on the insertion as on a writer sentence. What would drop a writer sentence refuses it, with
        # the review's reason; a fact the case does not hold is marked when FACT_MARK_INLINE_ENABLED marks a writer's,
        # else (a writer's is only logged) the insertion is refused: it is optional text, not a sentence of the section.
        dropped = [e for e in review if e.get("how") == "sentence dropped"]
        if dropped:
            return {**entry, "why": f"dropped by the review: {dropped[0]['kind']}: {dropped[0].get('why') or ''}".strip(),
                    "sentence": reply[:300]}
        logged = [e for e in review if e.get("kind") == "fact_not_in_case" and e.get("how") == "logged"]
        if logged:
            return {**entry, "why": f"review flag: fatto_non_nel_caso: {logged[0].get('why') or ''}".strip(),
                    "sentence": reply[:300]}
    if _review_fixes():
        text, _ = rewrite_unsupported(title, text, case_text, article_rows)
    why = (_cover_refusal(text, ref, documents, article_rows, plan, context) or _wrong_statement(text, context)) \
        if text else "dropped by the review"
    if why:
        return {**entry, "why": why, "sentence": (text or reply)[:300]}
    if grounded_review.enabled():
        judged = grounded_review._grounded_judge(f"[{label}]\n{source}", [text])
        if grounded_review.verdict_of(judged.get(0))[1] in ("assente", "contraddetta", None):
            return {**entry, "why": "not supported by the article's text", "sentence": text[:300]}
    bad, answered = answer_invariants.judge([{"ref": ref, "heading": heading or "-", "sentence": text,
                                              "opening": article_text.clause_cut(body, 300)}], dg._chat_json)
    if 0 not in answered or bad:
        return {**entry, "why": "qualification check: " + (bad.get(0) or "no verdict"), "sentence": text[:300]}
    return {**entry, "how": "inserted", "sentence": text}


def _insert_for_issue(section: str, title: str, sentence: str) -> str:
    """The section with the sentence after the first paragraph under the subheading closest to the issue's title; no such
    subheading, at its end."""
    lines = section.split("\n")
    stems = _title_stems(title)
    heads = [(len(stems & _title_stems(line)), -i, i) for i, line in enumerate(lines) if _HEADING_LINE_RE.match(line)
             and not _HEADING_LINE_RE.match(line).group(2).strip()]
    best = max(heads, default=(0, 0, -1))
    if not best[0]:
        return f"{section.rstrip()}\n\n{sentence}"
    i = best[2] + 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    while i < len(lines) and lines[i].strip() and not _HEADING_LINE_RE.match(lines[i]):
        i += 1
    return "\n".join(lines[:i] + ["", sentence] + lines[i:])


def cover_plan_articles(plan: Dict[str, Any], sections: List[Tuple[str, str]], article_rows: List[Dict[str, Any]],
                        documents: List[Dict[str, Any]], case_text: str,
                        context: Optional[Dict[str, Any]] = None,
                        conclusions: Sequence[str] = ()) -> List[Tuple[str, str]]:
    """The sections with one or two sentences for each plan article omitted, in the issue's place in "Applicazione al
    caso". Logged as case_analysis_plan_articles, every insertion and refusal."""
    t0 = time.time()
    missing = _omitted(plan, sections, article_rows, documents, (context or {}).get("excluded") or (), conclusions)
    if not missing:
        dg.vlog("case_analysis_plan_articles", {"missing": [], "inserted": 0, "entries": []})
        return sections
    with ThreadPoolExecutor(max_workers=min(4, len(missing))) as pool:
        entries = list(pool.map(lambda m: _cover_one(*m, plan, documents, article_rows, case_text, context), missing))
    out = list(sections)
    i = next((k for k, (title, _) in enumerate(out) if title == SECTION_TITLES[3]), None)
    for entry in entries:
        if entry["how"] != "inserted":
            continue
        if i is None:
            entry.update(how="refused", why="no section for it")
            continue
        out[i] = (out[i][0], _insert_for_issue(out[i][1], entry["issue"], entry["sentence"]))
        entry["section"] = out[i][0]
    dg.vlog("case_analysis_plan_articles", {"missing": [e["article"] for e in entries], "entries": entries,
                                            "inserted": sum(1 for e in entries if e["how"] == "inserted")},
            duration_ms=(time.time() - t0) * 1000)
    return out


def for_the_plan(plan: Dict[str, Any], analysis: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(plan, analysis) as the sections read them: who claims a right on another's land has no negatoria (art. 949 c.c. is
    the owner's), and of the actions with no prescription only those the plan uses are listed."""
    from . import fact_signals
    signals = _signals(analysis)
    if not signals:
        return plan, analysis
    if statement_context(analysis).get("servitude"):
        plan = {**plan, "questioni": [{**q, "norme": [r for r in q["norme"] if not (r[0] == "949" and _code_key(r[1]) == "c.c.")]}
                                      for q in plan["questioni"]]}
    used = " ".join(f"{q.get('titolo') or ''} {q.get('argomento') or ''} {' '.join(n for n, _ in q['norme'])}" for q in plan["questioni"])

    def listed(term):
        if term.get("status") != fact_signals.NO_PRESCRIPTION_STATUS:
            return True
        number = re.search(r"\d+", term.get("ref") or "")
        return bool(re.search(_ACTION_NAMES[term["id"]], used, re.IGNORECASE)) or bool(number and re.search(rf"\b{number.group(0)}\b", used))
    return plan, {**analysis, "fact_signals": {**signals, "terms": [t for t in signals.get("terms") or [] if listed(t)]}}


_PRECLUDED_TOOLS_HEAD_RE = re.compile(r"^\W*strumenti processuali\b", re.IGNORECASE)


def _precluded_in_tools(text: str, terms: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """The sentences under the "Strumenti processuali" lead naming an action whose computed term has run."""
    names = [_ACTION_NAMES[t["id"]] for t in terms if t.get("status") == "DECORSO" and t.get("id") in _ACTION_NAMES]
    if not names:
        return []
    out, pos = [], 0
    inside = False
    for paragraph in re.split(r"(\n\s*\n)", text):
        if paragraph.strip() and not paragraph.isspace():
            if paragraph.lstrip().startswith(("**", "#")):
                inside = bool(_PRECLUDED_TOOLS_HEAD_RE.match(paragraph))
            if inside:
                out += [(pos + s, pos + e) for s, e in _sentence_spans(paragraph)
                        if any(re.search(n, paragraph[s:e], re.IGNORECASE) for n in names)]
        pos += len(paragraph)
    return out


_CLAUSE_SPLIT_RE = re.compile(r"(?<=[,;:])\s+")


def trim_clause(sentence: str, bad) -> str:
    """The sentence without the clauses (split at , ; :) `bad` finds, when what is left still cites a plan article,
    reads as a sentence and is no longer bad; "" else (v49: a plan article's sentence trimmed, not dropped)."""
    parts = _CLAUSE_SPLIT_RE.split(sentence.strip())
    if len(parts) < 2:
        return ""
    kept = [p for p in parts if not bad(p)]
    if not kept or len(kept) == len(parts):
        return ""
    out = " ".join(kept).rstrip(" ,;:")
    out = out[0].upper() + out[1:] if out else out
    if out and out[-1] not in ".!?":
        out += "."
    from . import text_tidy
    return out if citation_ledger.plan_cited(out) and not bad(out) and not text_tidy.unreadable(out) \
        and len(re.findall(r"\w{3,}", out)) >= 5 else ""


def final_checks(sections: List[Tuple[str, str]], article_rows: List[Dict[str, Any]],
                 context: Optional[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """The checks written in code, on the sections as the last model step left them, before plan cover can write a lost
    article back: an article's words around or instead of a citation (text_tidy.against_articles), the statements the
    writer must never make (_wrong_statement: the rewrites after the review may bring one back), an action already
    time-barred among the tools. Logged as case_analysis_final_checks."""
    from . import text_tidy
    out, entries = [], []
    terms = (context or {}).get("terms") or []
    for title, text in sections:
        text, log = text_tidy.against_articles(text, article_rows, "final_checks")
        entries += [{**e, "section": title} for e in log]
        text, made = usucapione_conditional(title, text, context)      # USUCAPIONE_CONDITIONAL: the rewrites' too
        entries += made
        drops = [(s, e) for s, e in _sentence_spans(text) if _wrong_statement(text[s:e], context)]
        tools = [span for span in _precluded_in_tools(text, terms) if span not in drops]
        trims: Dict[Tuple[int, int], str] = {}
        for s, e in drops + tools:         # v49: a plan article's sentence loses the offending clause, never itself
            sentence = text[s:e]
            kind = _wrong_statement(sentence, context) or "a time-barred action among the tools"
            if not citation_ledger.plan_cited(sentence):
                entries.append({"section": title, "kind": kind, "sentence": sentence.strip()[:300]})
                continue
            bad = (lambda c: bool(_wrong_statement(c, context))) if (s, e) in drops else \
                (lambda c: bool(_precluded_in_tools(f"**Strumenti processuali**\n\n{c}", terms)))
            trimmed = trim_clause(sentence, bad)
            if trimmed:
                trims[(s, e)] = trimmed
                entries.append({"section": title, "kind": f"{kind}: clause trimmed", "sentence": sentence.strip()[:300],
                                "now": trimmed[:300]})
                citation_ledger.kept("final_checks", sentence, kind, trimmed)
            else:           # nothing to trim it to: dropped, the reason logged in full
                entries.append({"section": title, "kind": kind, "sentence": sentence.strip()[:300],
                                "why": "a plan article's sentence: no clause to trim it to"})
        dropped = [span for span in drops + tools if span not in trims]
        if trims:
            text = grounded_review.apply(text, {**trims, **{span: None for span in dropped}}) or text
        elif dropped:
            text = _drop_spans(text, dropped) or text
        out.append((title, text))
    dg.vlog("case_analysis_final_checks", {"entries": entries, "count": len(entries)})
    citation_ledger.record_entries("final_checks", entries)
    return out


# TOOL_CITES_PROTECTED (v51, Leonardo v50: "La trascrizione è necessaria per garantire la stabilità del diritto e la sua
# opposibilità." -- the trascrizione tool's sentence without its art. 2653, n. 1, c.c.): the articles of the procedural
# tools (the criminal profile's included) and of the computed terms get the plan articles' protection
# (citation_ledger.protect): a sentence citing one is trimmed or kept, never dropped, by answer_tidy, final_checks and
# anchored_writing. Heads of damage are not tools.
TOOL_CITES_PROTECTED_FLAG = "TOOL_CITES_PROTECTED"


def protected_tool_refs(analysis: Dict[str, Any]) -> List[tuple]:
    """TOOL_CITES_PROTECTED: the tools' and the terms' articles; [] with the flag off."""
    if not dg.feature_flags.enabled(TOOL_CITES_PROTECTED_FLAG):
        return []
    from . import citation_closure
    signals = _signals(analysis) or {}
    tools = [tuple(r) for t in signals.get("tools") or [] for r in t.get("refs") or []]
    return list(dict.fromkeys(tools + citation_closure.term_refs(signals)))


def write_parere(facts: str, request: str, analysis: Dict[str, Any], article_rows: List[Dict[str, Any]],
                 ruling_rows: List[Dict[str, Any]], articles: str, rulings: str, principles: str,
                 lang: str = "it", documents: Optional[List[Dict[str, Any]]] = None,
                 store: Optional["article_text.ArticleStore"] = None) -> Optional[str]:
    """The parere from a plan, section by section, or None (the caller writes it in one call).
    `articles` and `rulings` are the draft prompt's own blocks (_fit_parere_prompt); `documents` (the
    corpus' names) let _discipline() read a citation of a law by its name; `store`, the request's
    article rows already read (article_text.ArticleStore), spares the commi a second read."""
    references: Dict[str, List[Dict[str, Any]]] = {}
    for row in ruling_rows:
        reference = dg._ruling_reference(row)
        if reference:
            references.setdefault(reference, []).append(row)
    plan = make_plan(facts, request, analysis, articles, rulings, principles, article_rows, references)
    if plan is None:
        return None
    case = dg._case_text(facts, request)
    pipeline_analysis = analysis
    plan, analysis = for_the_plan(plan, analysis)
    rendered = None
    if render_computed.enabled():
        try:            # RENDER_COMPUTED: the procedural section from the computed block, by template
            party = f"{plan.get('parte_assistita') or ''} {plan.get('posizione') or ''}".strip()
            rendered = render_computed.render(analysis, party) or None      # nothing computed: the model writes it
            if rendered:
                pipeline_analysis[render_computed.RENDERED_KEY] = rendered
            dg.vlog("case_analysis_render_computed", {"blocks": [t for t, _ in render_computed.blocks(analysis, party)],
                                                      "chars": len(rendered or ""), "fallback": rendered is None,
                                                      "gated": plan.get("obiezioni_norme") or []})
        except Exception as exc:
            _log_exception("render_computed", exc)
            rendered = None
    if rule_quote.enabled():
        try:
            dg.vlog("case_analysis_rule_quote_material", rule_quote.material_log(plan, article_rows, analysis))
        except Exception as exc:
            _log_exception("rule_quote.material_log", exc)
    in_plan = _plan_refs(plan)
    # v49: a sentence citing one is never dropped by tidy / final checks / anchors; TOOL_CITES_PROTECTED (v51): nor one
    # citing a procedural tool's or a computed term's article
    citation_ledger.protect(in_plan + [r for r in protected_tool_refs(analysis) if r not in in_plan])
    for row in article_rows:             # text_tidy.headed_norms: the articles a paragraph of the norms section may be about
        if row.get("_article") in in_plan:
            row["_in_plan"] = True
    # COMMA_MATERIAL_ENABLED: each article's commi chosen for the plan issue that places it; a comma the plan
    # or the case names, read now when the row lacks it (one read, no model call).
    commi = comma_material.for_plan(plan, article_rows, case, store=store)
    if commi:
        dg.vlog("comma_material_plan", commi)
    named = list(dict.fromkeys(dg._refs_in(case) + dg._cited_articles(case, set(), documents or [])))
    if norme_scope_enabled():            # NORME_SCOPE: section 3 writes the primary norms only
        try:
            scope, scope_log = norme_scope(plan, article_rows, named)
            if scope:                    # no primary norm: section 3 keeps the plan's, as without the flag
                plan = {**plan, NORME_SCOPE_KEY: scope}
            else:
                scope_log["fallback"] = "no primary norm: every plan article"
            dg.vlog("case_analysis_norme_scope", scope_log)
        except Exception as exc:
            _log_exception("norme_scope", exc)
    use_of = (lambda text: material_use(analysis, text)) if material_use(analysis, "") else None
    anchors = anchor_catalogue(plan, analysis, article_rows, references)
    if anchors is not None:     # a failing statement citing one of these is rewritten from it or kept, never removed
        # v49: the plan's articles too (h2 v48: 1382 c.c., exempt by heading, removed as an unanchored citation)
        anchors.checked = list(dict.fromkeys(list(grounded_review.strict_refs(article_rows, named, analysis)) + in_plan))
    sections = write_sections(plan, section_jobs(plan, facts, request, analysis, article_rows, references,
                                                 principles, _keep_all(), anchors, rendered), lang, named, documents,
                              article_rows, case, coverage=use_of if rendered is None else None,
                              standard=standard_blocks(analysis),
                              questions=explicit_questions(request, facts) if _review_fixes() else [],
                              checked=grounded_review.strict_refs(article_rows, named, analysis)
                              if grounded_review.strict() or grounded_review.abstain() else None, anchors=anchors,
                              context=statement_context(analysis))
    if sections is None:
        return None
    context = statement_context(analysis)
    sections = _step("final_checks", sections, lambda secs: final_checks(secs, article_rows, context))
    if covering():
        sections = _step("cover_plan_articles", sections, lambda secs: cover_plan_articles(
            plan, secs, article_rows, documents or [], case, context, conclusion_lines(analysis)))
    if rule_quote.enabled():
        def quoted(secs):       # RULE_QUOTE: a rule sentence stays within its comma, or becomes its quotation
            done, log = rule_quote.check_sections(secs, plan, article_rows, analysis, documents or [])
            dg.vlog("case_analysis_rule_quote", log)
            return done
        sections = _step("rule_quote", sections, quoted)
    if plan.get("obiezioni_norme"):
        def objections_only(secs):      # RENDER_COMPUTED: the action gate's articles only in an objection or a reply
            done = [render_computed.gated_as_support(title, text, plan, documents) for title, text in secs]
            entries = [e for _, log in done for e in log]
            dg.vlog("case_analysis_gated_as_support", {"entries": entries, "dropped": len(entries)})
            return [(title, text or original) for (title, original), (text, _) in zip(secs, done)]
        sections = _step("gated_as_support", sections, objections_only)
    sections = _step("repetition_guard", sections, lambda secs: repetition_guard.guard_sections(secs, "rewritten")[0])
    if rendered is not None:            # RENDER_COMPUTED: the rendered procedural section in its place, untouched
        at = min(4, len(sections))
        sections = sections[:at] + [(SECTION_TITLES[4], rendered)] + sections[at:]
    if use_of is not None:
        try:          # counted only: never fails the answer
            final = use_of(next((text for title, text in sections if title == SECTION_TITLES[4]), ""))
            dg.vlog("case_analysis_material_use", {
                "offered": {b: u["offered"] for b, u in final.items()},
                "written": {b: u["written"] for b, u in final.items()}, "missing": _missing_blocks(final)})
        except Exception as exc:
            _log_exception("write_parere", exc)
            logger.warning("plan-then-write: material use not counted: %s", exc)
    return "\n\n".join(f"**{i}. {title}**\n\n{text}" for i, (title, text) in enumerate(sections, 1))
