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

import json
import logging
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from . import defensive_generation as dg
from . import comma_material, grounded_review, method_candidates, practical_checklist, protected_sources

logger = logging.getLogger(__name__)

_PLAN_TOKENS = 2000
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
    '{"parte_assistita": "...", "posizione": "...", "obiettivo": "che cosa vuole ottenere", '
    '"questione_centrale": "una frase", '
    '"risposte": [{"quesito": "...", "risposta": "risposta diretta in una o due frasi"}], '
    '"questioni": [{"titolo": "...", "norme": ["art. N codice"], "pronunce": ["riferimento"], '
    '"argomento": "...", "obiezione": "...", "replica": "..."}], '
    '"subordinate": ["..."], "punti_deboli": ["..."]}\n'
    "Le questioni, da due a cinque, in ordine di forza per la parte assistita."
)

def _plan_system(max_issues: int = _MAX_ISSUES) -> str:
    system = _PLAN_SYSTEM.replace("da due a cinque", f"da due a {_ISSUE_COUNT_WORDS.get(max_issues, max_issues)}")
    if _writer_rules():
        system = system.replace("- Non aggiungere fatti che il caso non contiene.\n",
                                _STANCE_RULES + "- Non aggiungere fatti che il caso non contiene.\n")
    if _review_fixes():
        system = system.replace("- Non aggiungere fatti che il caso non contiene.\n",
                                _ACQUIRED_RULE + _QUESTIONS_RULE + "- Non aggiungere fatti che il caso non contiene.\n")
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
        heading = " ".join(str(s.get("titolo_articolo") or s.get("title") or "").split())
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
                       "obiezione": _text(q.get("obiezione")), "replica": _text(q.get("replica"))})
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

    best: Dict[tuple, tuple] = {}
    why: Dict[tuple, str] = {}
    for index, row in enumerate(article_rows):
        ref, grade = row.get("_article"), row.get("_grade")
        if not ref or row.get("_demoted"):
            continue
        if row.get("_central"):
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
    forced = dict(list(out.items())[:_keep_cap()])
    rows_of = {r["_article"]: r for r in article_rows if r.get("_article")}
    if any(evidence_tiers.tier(r) for r in rows_of.values()):
        past = [rows_of[ref] for ref in list(out)[_keep_cap():]]
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
    terms = _terms(analysis)
    qualified, tools = _qualifications(analysis), _tools(analysis)
    damages, roles = _damages(analysis), _roles(analysis)
    fixes = _review_fixes()
    questions = "\n".join(f"{i}. {q}" for i, q in enumerate(explicit_questions(request, facts), 1)) if fixes else ""
    acquired = "\n".join(f"- {a}" for a in acquired_rights(analysis)) if fixes else ""
    return (f"PARTE ASSISTITA (dall'analisi del caso): {dg._stance_party(analysis)}\n\n{base}\n\n"
            + (f"{_QUESTIONS_TITLE}:\n{questions}\n\n" if questions else "")
            + (f"{_ACQUIRED_TITLE}:\n{acquired}\n\n" if acquired else "")
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
        blocks.append(f"[art. {ref[0]} {ref[1]}]\nRubrica: {headings.get(ref) or '-'}\nTesto: {text[:1200] or '-'}")
    human = (f"QUESTIONE: {issue['titolo']}\nARGOMENTO DELLA PARTE ASSISTITA: {issue['argomento']}\n\n"
             "NORME COLLEGATE:\n" + "\n\n".join(blocks))
    system = _PLAN_CHECK_SYSTEM.replace('Nel dubbio: pertinente', _PLAN_CHECK_DIRECTION + 'Nel dubbio: pertinente') \
        if _writer_rules() else _PLAN_CHECK_SYSTEM
    return dg._chat_json([SystemMessage(content=system), HumanMessage(content=human)],
                         max_tokens=150 + 120 * len(issue["norme"]), temperature=0)


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
    "solo in via subordinata, dopo, e non è la questione centrale.\n")
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


def acquired_rights(analysis: Dict[str, Any]) -> List[str]:
    """The rights the case shows already acquired: the fired fact rules marked acquired_right (their institution and
    why), then the analysis' questions and institutions naming one."""
    out = []
    for rule in (_signals(analysis) or {}).get("fired") or []:
        if rule.get("acquired_right"):
            out.append(f"{', '.join(rule.get('institutions') or [])}: {rule.get('why') or ''}".strip(": "))
    for q in (analysis or {}).get("questioni") or []:
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


def order_for_acquired(plan: Dict[str, Any], acquired: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(plan, log): with a right already acquired, the recognition issues first, a constitutive remedy last and
    titled as subordinate (and named in the plan's subordinate requests); the plan says which right is acquired
    ("diritto_acquisito", for every section)."""
    kinds = [_issue_kind(q) for q in plan["questioni"]]
    if not acquired or "constitutive" not in kinds:
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
    return {**plan, "questioni": issues, "subordinate": subordinate, "diritto_acquisito": "; ".join(acquired)}, log


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


def make_plan(facts: str, request: str, analysis: Dict[str, Any], articles: str, rulings: str, principles: str,
              article_rows: List[Dict[str, Any]], references: Dict[str, List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """The cleaned plan, made once more if its arguments favour the other side, then checked
    (check_plan); None on failure."""
    known = {r["_article"] for r in article_rows if r.get("_article")}
    headings = _headings(article_rows)
    voted = [ref for ref in _voted(article_rows) if f"[art. {ref[0]} {ref[1]}]" in articles]
    keep_checked = dg.feature_flags.enabled(KEEP_CHECKED_FLAG)
    keep_all = dg.feature_flags.enabled(KEEP_APPLICABLE_FLAG) and not keep_checked
    reasons = _forced(article_rows) if keep_checked and _graded() else None
    applicable = (list(reasons) if reasons is not None else _checked(article_rows) if keep_checked
                  else _applicable(article_rows, facts, request, analysis) if keep_all else [])
    max_issues = _MAX_ISSUES_WIDE if keep_checked or len(applicable) >= _WIDE_APPLICABLE else _MAX_ISSUES
    system = _plan_system(max_issues)
    issue_of = {r["_article"]: r.get("_grade_issue") or "" for r in article_rows if r.get("_article")}
    groups = _issue_groups(reasons, issue_of) if reasons is not None and _by_issue() else None
    human = _plan_human(facts, request, analysis, _with_headings(articles, headings), rulings, principles,
                        _checks(analysis, article_rows, facts, request), voted, applicable,
                        must_place="graded" if keep_checked and _graded() else keep_checked, groups=groups)
    messages = [SystemMessage(content=system), HumanMessage(content=protected_sources.with_rule(human))]
    t0 = time.time()
    problems: List[str] = []
    try:
        raw = dg._chat_json(messages, max_tokens=_PLAN_TOKENS, temperature=0)
        plan, removed = clean_plan(raw, known, references, max_issues)
        opposing = dg._opposing_arguments(facts, request, analysis, _arguments_text(plan)) if plan else []
        if opposing:
            listing = "\n".join(f"- {o['argomento']}: {o['perche']}" for o in opposing)
            retry = (f"{human}\n\nIl tuo primo piano:\n{json.dumps(raw, ensure_ascii=False)}\n\nUna revisione ha "
                     f"trovato che questi argomenti giovano alla controparte, non a {dg._stance_party(analysis)}:\n"
                     f"{listing}\n\nRifai il piano: ogni argomento deve escludere, ridurre o contrastare la "
                     "responsabilità o la pretesa contro la parte assistita, o sostenerne la pretesa; un fatto "
                     "che favorisce la controparte va trattato come obiezione da superare.")
            raw = dg._chat_json([SystemMessage(content=system), HumanMessage(content=retry)],
                                max_tokens=_PLAN_TOKENS, temperature=0)
            plan, removed = clean_plan(raw, known, references, max_issues)
        problems = _plan_problems(plan, analysis, explicit_questions(request, facts)) if plan and _review_fixes() else []
        if problems:
            listing = "\n".join(f"- {p}" for p in problems)
            again = (f"{human}\n\nIl tuo piano:\n{json.dumps(raw, ensure_ascii=False)}\n\nUna revisione ha trovato:\n"
                     f"{listing}\n\nRifai il piano correggendo questi punti, con le stesse regole.")
            raw_again = dg._chat_json([SystemMessage(content=system), HumanMessage(content=again)],
                                      max_tokens=_PLAN_TOKENS, temperature=0)
            plan_again, removed_again = clean_plan(raw_again, known, references, max_issues)
            if plan_again:
                plan, removed, raw = plan_again, removed_again, raw_again
    except Exception as exc:
        logger.warning("plan-then-write: no plan, the parere is written in one call: %s", exc)
        return None
    finally:
        dg.vlog("case_analysis_plan_llm", duration_ms=(time.time() - t0) * 1000)
    checked: Dict[str, List[str]] = {"off_topic": [], "contradicted": [], "kept_by_heading": [], "kept_by_vote": []}
    if plan:
        plan, checked = check_plan(plan, article_rows, headings, applicable if keep_all or keep_checked else None)
    fixes: Dict[str, Any] = {}
    if plan and _review_fixes():
        plan, merged = merge_duplicate_issues(plan)
        plan, order = order_for_acquired(plan, acquired_rights(analysis))
        fixes = {"problems": problems, "merged": merged, "order": order,
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
    dg.vlog("case_analysis_plan", {
        "issues": len(plan["questioni"]) if plan else 0, "replanned": bool(opposing),
        "opposing": [o["argomento"] for o in opposing],
        "norme": sorted({f"{n} {c}" for q in (plan or {}).get("questioni", []) for n, c in q["norme"]}),
        "pronunce": sorted({p for q in (plan or {}).get("questioni", []) for p in q["pronunce"]}),
        "removed": removed, "checked": checked, "references_available": len(references),
        "voted_offered": [f"{n} {c}" for n, c in voted], **({"review_fixes": fixes} if fixes else {})})
    if plan is None:
        logger.warning("plan-then-write: the plan has no usable issue, the parere is written in one call")
        return None
    if not plan["parte_assistita"]:
        plan["parte_assistita"] = str(analysis.get("parte_assistita") or "").strip()
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
            norms = ", ".join(f"art. {n} {c}" for n, c in q["norme"]) or "-"
            cited = f"; pronunce: {', '.join(q['pronunce']) or '-'}" if rulings else ""
            if rulings and q.get("pronunce_contrarie"):
                cited += (f"; pronunce della controparte, solo nell'obiezione con la replica: "
                          f"{', '.join(q['pronunce_contrarie'])}")
            lines.append(f"{i}. {q['titolo']} (norme: {norms}{cited})")
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


def _section_system(plan: Dict[str, Any], lang: str) -> str:
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
        "- Scrivi solo il testo della sezione, senza il suo titolo. Il parere ha sei sezioni (risposte ai "
        "quesiti; inquadramento e fatti; norme e giurisprudenza; applicazione al caso; profili processuali e "
        "termini; conclusioni): non ripetere ciò che spetta alle altre, sii essenziale.\n"
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
        + (_WRITER_RULES.format(party=party) if rules else "") +
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
    "- Il parere è completo in sé: non rimandare ad approfondimenti successivi.\n"
)


def _facts_list(analysis: Dict[str, Any]) -> str:
    return "\n".join(f"- {f.get('fatto', '')}" for f in analysis.get("fatti", []) if f.get("fatto"))


def section_jobs(plan: Dict[str, Any], facts: str, request: str, analysis: Dict[str, Any],
                 article_rows: List[Dict[str, Any]], references: Dict[str, List[Dict[str, Any]]],
                 principles: str, keep_all: bool = False) -> List[Tuple[str, str, str, int]]:
    """(title, instruction, material, max_tokens) for each section, in order. With keep_all
    (PLAN_KEEP_APPLICABLE_ENABLED) sections 3 and 4 discuss every article of the plan by name."""
    party = plan["parte_assistita"] or "la parte assistita"
    case = f"CASO:\n{dg._case_text(facts, request)}"
    frame = analysis.get("frame") or {}
    tasks = "\n".join(f"- {t}" for t in frame.get("tasks") or []) or request.strip()
    facts_list = _facts_list(analysis)
    refs = _plan_refs(plan)
    headings = _headings(article_rows)
    # Article text per section within a fixed total, so that a plan naming many articles still
    # leaves each call well inside the context with the case text (up to ~5,000 tokens).
    def per_article(total_chars: int, most: int) -> int:
        return max(300, min(most, total_chars // max(1, len(refs))))

    def texts(wanted: List[tuple], chars: int) -> str:
        return _with_headings(_article_texts(article_rows, wanted, chars), headings)
    posture = method_candidates.case_lists(analysis)[1]
    procedural = [ref for ref in method_candidates.candidate_refs([], posture) + _tool_refs(analysis)
                  if ref not in refs]
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
    norms = ("Per le norme del PIANO su cui si fondano gli argomenti: che cosa stabilisce ciascuna, citando il testo "
             "fornito, e, solo se la norma stessa pone una presunzione o ripartisce l'onere della prova, quella "
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
    evidence = (" Le prove da acquisire di una questione che coincidono con quelle di una questione precedente non si "
                "ripetono: si rinvia a quella." if fixes else "")

    jobs = [
        (SECTION_TITLES[0],
         "SEZIONE: Risposte ai quesiti. Una risposta diretta e motivata a ciascun quesito, nell'ordine in cui è "
         "posto, a partire dalle risposte del PIANO quando ci sono, altrimenti dalle sue questioni; se la "
         "questione centrale non è tra i quesiti, rispondi prima a essa. Ogni risposta in due-quattro frasi, con "
         "la norma su cui si fonda." + one_each + scoped + no_rulings,
         material(case, block("QUESITI", tasks), block(_QUESTIONS_TITLE, asked),
                  block(_TOOLS_TITLE, tools if questions and any(fact_signals_asks(q) for q in questions) else ""),
                  _plan_text(plan, answers=True, rulings=False), scope), 600 + 150 * max(0, len(questions) - 1)),
        (SECTION_TITLES[1],
         "SEZIONE: Inquadramento e questione centrale. La qualificazione giuridica della vicenda (la norma "
         "principale e quelle alternative o subordinate, e perché), la questione centrale e perché decide "
         f"l'esito; poi {facts_part} Non illustrare le "
         "norme nel dettaglio (sezione 3) né sviluppare gli argomenti (sezione 4)." + scoped + no_rulings,
         material(case, block("FATTI DEL CASO", facts_list), block(_QUALIFIED_TITLE, qualified),
                  _plan_text(plan, rulings=False), scope,
                  block("TESTI DELLE NORME", texts(refs, per_article(8000, 700)))), 1000),
        (SECTION_TITLES[2],
         f"SEZIONE: Norme e giurisprudenza. {norms}; per ogni pronuncia del PIANO: il principio che "
         "afferma, con il suo riferimento, e che cosa comporta per questo caso, senza svolgere gli argomenti "
         "(sezione 4). Solo norme e pronunce del PIANO; tra virgolette solo frasi dei testi e dei passi forniti."
         + every + scoped + adverse,
         material(case, _plan_text(plan), scope,
                  block("TESTI DELLE NORME", texts(refs, per_article(15000, dg._ARTICLE_CHARS))),
                  block("PASSI DELLE PRONUNCE", _ruling_texts(plan, references, 1200)),
                  block("PRINCIPI DALLA GIURISPRUDENZA", principles)), 1600 if keep_all else 1200),
        (SECTION_TITLES[3],
         "SEZIONE: Applicazione al caso. Per ogni questione del PIANO, nell'ordine, con un sottotitolo: "
         f"l'argomento di {party} applicato ai fatti del caso, con le norme e le pronunce del piano; le prove da "
         "acquisire; l'obiezione più forte della controparte e la replica. Poi le tesi subordinate del PIANO. "
         "Prosa, senza ripetere per ogni questione lo stesso schema a voci; tra virgolette solo frasi dei testi e "
         "dei passi forniti." + every_issue + scoped + applied + adverse + evidence,
         material(case, block("FATTI DEL CASO", facts_list), block(_QUALIFIED_TITLE, qualified),
                  block(_ROLES_TITLE, roles), block(_DAMAGES_TITLE, damages),
                  _plan_text(plan, issues="full", subordinate=True), scope,
                  block("TESTI DELLE NORME", texts(refs, per_article(9000, 600))),
                  block("PASSI DELLE PRONUNCE", _ruling_texts(plan, references, 600))), 2200 if keep_all else 1800),
        (SECTION_TITLES[4],
         "SEZIONE: Profili processuali e termini. La fase in cui si trova la vicenda e che cosa consente di fare "
         f"a {party}: gli atti da compiere e i loro termini, con l'articolo che li prevede; a chi rivolgersi o chi "
         "convenire; le voci di danno o le domande da proporre, anche in via subordinata. Usa le VERIFICHE solo "
         "se i fatti del caso le rendono pertinenti, dicendo perché, e solo quelle che la legge consente a "
         f"{party} nella sua posizione; un termine che dipende da una data che il caso non indica è una verifica "
         "da fare." + (f" {_terms_rule()}" if _terms(analysis) else "") + regimes + no_rulings,
         material(case, block("FASE", frame.get("procedural_posture") or ""), block(_QUALIFIED_TITLE, qualified),
                  block(_TERMS_TITLE, _terms(analysis)), block(_TERM_TEXTS_TITLE, _term_texts(analysis)),
                  _plan_text(plan, issues="", subordinate=True, rulings=False),
                  block(_TOOLS_TITLE, tools), block(_CRIMINAL_TITLE, criminal), block(_DAMAGES_TITLE, damages),
                  block("VERIFICHE PER QUESTA FASE E MATERIA",
                        "\n".join(f"- {c}" for c in _checks(analysis, article_rows, facts, request))),
                  block("TESTI DELLE NORME PROCESSUALI", texts(procedural, 500))), 900 + 250 * len(blocks)),
        (SECTION_TITLES[5],
         "SEZIONE: Conclusioni. In due o tre paragrafi: gli argomenti in ordine di forza, come nel PIANO; una "
         f"stima motivata del rischio, tenendo conto dei punti deboli; la condotta consigliata a {party}."
         + no_rulings,
         _plan_text(plan, subordinate=True, weak=True, rulings=False), 500),
    ]
    jobs = [_with_budget(job) for job in jobs] if rules else jobs
    for title, _, material_text, _ in jobs:        # COMMA_MATERIAL_ENABLED: the commi each section was given
        comma_material.log_material(f"section {title}", material_text, article_rows)
    return jobs


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


def _sentence_span(text: str, start: int, end: int) -> Tuple[int, int]:
    """The sentence around text[start:end], within its line."""
    s = start
    while s > 0 and text[s - 1] != "\n" and not (text[s - 1] in ".!?" and _ends_sentence(text, s - 1)):
        s -= 1
    while s < start and text[s].isspace():
        s += 1
    e = end
    while e < len(text) and text[e] != "\n" and not (text[e] in ".!?" and _ends_sentence(text, e)):
        e += 1
    return s, min(len(text), e + 1) if e < len(text) and text[e] in ".!?" else e


def _material_refs(material: str, documents: List[Dict[str, Any]]) -> List[tuple]:
    """Every article a text cites with its code or law, as the report counts them (dg._cited_articles)."""
    return dg._cited_articles(material or "", set(), documents)


def _code_key(code: str) -> str:
    """A code by its sigla, also when written as its document's name ("Codice Civile 2026" -> "c.c.")."""
    return next((abbr for abbr, prefix in dg._CODE_PREFIXES.items() if code.startswith(prefix)), code)


def _same_ref(a: tuple, b: tuple) -> bool:
    """_same_article, a code named by its document read as its sigla."""
    return _same_article((a[0], _code_key(a[1])), (b[0], _code_key(b[1])))


def _citations(text: str, documents: List[Dict[str, Any]], known: Optional[set] = None) -> List[Tuple[int, int, List[tuple]]]:
    """(start, end, refs) of each citation in text, with its code or law (dg._code_after); a number without
    one counts when exactly one of `known` has it (as dg._cited_articles reads a draft)."""
    out = []
    for m in dg._ARTICLE_RUN_RE.finditer(text or ""):
        code = dg._code_after(text[m.end():m.end() + 120], documents)
        numbers = [dg._article_num(n) for n in dg._RUN_NUMBER_RE.findall(m.group(1))]
        if code:
            refs = [(n, code) for n in numbers]
        else:
            refs = [same[0] for n in numbers for same in [[k for k in known or () if k[0] == n]] if len(same) == 1]
        if refs:
            out.append((m.start(), m.end(), refs))
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
        pos = e
        if s == 0 or text[s - 1] == "\n":
            while pos < len(text) and text[pos] in " \t":      # a dropped first sentence: no leading blank
                pos += 1
        log.append({"section": title, "removed": [f"{n} {c}" for n, c in refs], "how": "sentence dropped",
                    "sentence": text[s:e].strip()[:300], "now": ""})
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
    r"^\W*(?:(?:inoltre|infatti|altresì|anche),?\s+)?(?:conferma|stabilisce|prevede|dispone|afferma|ribadisce|precisa|"
    r"chiarisce|sancisce)\s+(?:che|il|la|l'|lo|i|gli|le|un|una|altresì|inoltre)\b", re.IGNORECASE)
_EMPTY_TAIL_RE = re.compile(
    r"(?:,\s*|\s+)(?:in conformità|ai sensi|in virtù|come previsto|secondo quanto (?:previsto|disposto|stabilito)|"
    r"a norma|in base|come stabilito|conformemente)(?:\s+(?:a|al|alla|allo|ai|alle|all'|dal|dalla|dall'|del|della|dell'|"
    r"di|degli|delle|dei))?\s*(?=[.;:!?]\s*$)", re.IGNORECASE)


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
               "verifica da fare.")


# FACT_REGIMES_ENABLED: every applicable term with its article, also when the case gives no dates (DA VERIFICARE).
_TERMS_RULE_REGIMES = ("Indica sempre ciascuno di questi termini con la sua durata, da quando decorre e la norma "
                       "tra parentesi quadre, anche quando il caso non dà le date: un termine DA VERIFICARE o "
                       "INCERTO è una verifica da fare (quale data accertare); un'azione il cui termine è DECORSO non "
                       "va proposta come praticabile (indicala come preclusa e proponi le alternative). Nessun altro "
                       "termine che non sia qui o nei testi forniti.")
_TERM_TEXTS_TITLE = "TESTI DELLE NORME SUI TERMINI"
_QUALIFIED_TITLE = "QUALIFICHE DELLE PARTI"
_QUALIFIED_RULE = ("Sono dedotte dai fatti: considera i regimi speciali che comportano (per il consumatore, la "
                   "garanzia legale di conformità e il foro del consumatore del Codice del Consumo).")
_TOOLS_TITLE = "STRUMENTI PROCESSUALI INDICATI DAI FATTI"
_TOOLS_RULE = "Valutali per la fase del caso: i fatti indicati accanto li rendono pertinenti."


def _signals(analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return (analysis or {}).get("fact_signals")


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
                          "written": [t["id"] for t in terms if _length_written(t.get("length", ""), text)]}
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


def _looks_cut(text: str) -> bool:
    """The section stops mid-sentence (its token budget ran out): the last line does not end a sentence, and is
    not a short list item or a heading."""
    body = (text or "").rstrip()
    if not body or body[-1] in ".!?»”\")]:":
        return False
    last = body.split("\n")[-1].strip()
    if last.startswith(("#", "**")) and last.endswith("**"):
        return False
    if _LIST_ITEM_RE.match(last) and len(last) < 200 and not _CUT_TAIL_RE.search(last):
        return False
    return True


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
    """text without the spans; a list item or line left with no words goes."""
    for s, e in sorted(spans, reverse=True):
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
                            max_tokens=120 + 40 * min(len(numbered), 30), temperature=0)
    except Exception as exc:
        logger.warning("writer rules: review of %r failed: %s", title, exc)
        return None
    out = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict) and str(item.get("i", "")).isdigit() and 1 <= int(item["i"]) <= len(numbered):
            out.append({"i": numbered[int(item["i"]) - 1][0], "tipo": str(item.get("tipo") or ""),
                        "perche": " ".join(str(item.get("perche") or "").split())[:200]})
    return out


def review_section(title: str, text: str, case_text: str, article_rows: Optional[List[Dict[str, Any]]] = None,
                   documents: Optional[List[Dict[str, Any]]] = None, party: str = "") -> Tuple[str, List[Dict[str, Any]]]:
    """(text, log), WRITER_RULES_ENABLED: dropped, each sentence citing an article only to say it does not apply
    (the review call, and _NOT_APPLY_RE in the sentence; without the call, only a remark on relevance,
    _NOT_RELEVANT_RE) and each burden template whose norm allocates nothing (_burden_templates); flagged with
    _FACT_MARK (only with FACT_MARK_INLINE_ENABLED; otherwise logged), each sentence the review finds asserting a
    party's conduct, knowledge or intent the case does not hold, unless _fact_flag_excluded; dropped, each
    sentence proposing an action against the client (party) or a co-client."""
    rows, documents = article_rows or [], documents or []
    known = {r["_article"] for r in rows if r.get("_article")}
    spans = _sentence_spans(text)
    if not spans:
        return text, []
    flags = _review_call(title, text, spans, case_text, party)
    drop: Dict[int, Tuple[str, str]] = {}
    mark: Dict[int, str] = {}
    filtered: List[Dict[str, Any]] = []
    for k, (s, e) in enumerate(spans):
        sentence = text[s:e]
        if flags is None and _cites(sentence) and _NOT_RELEVANT_RE.search(sentence):
            drop[k] = ("non_application", "a remark on relevance (no review call)")
    for flag in flags or []:
        k, sentence = flag["i"], text[spans[flag["i"]][0]:spans[flag["i"]][1]]
        if flag["tipo"] == "non_applicabile" and _cites(sentence) and _NOT_APPLY_RE.search(sentence):
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
                  standard: Optional[Dict[str, str]] = None) -> List[Tuple[str, str]]:
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
        again = _without_heading(dg._call_chat([
            SystemMessage(content=f"{system}\n\n{instruction} La stesura precedente ometteva: {lacking}. Tratta "
                                  "ciascun blocco, uno per uno."),
            HumanMessage(content=material)], max_tokens=max_tokens), title)
    except Exception as exc:
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
                       system: str) -> Tuple[List[Tuple[str, str]], List[Dict[str, Any]]]:
    """WRITER_RULES_ENABLED: a section cut short (_looks_cut) written once more with half again its tokens; still
    cut, trimmed to its last complete sentence."""
    cut = [i for i, (_, text) in enumerate(sections) if _looks_cut(text)]
    if not cut:
        return sections, []

    def again(i):
        title, instruction, material, max_tokens = jobs[i]
        try:
            return _without_heading(dg._call_chat([SystemMessage(content=f"{system}\n\n{instruction}"),
                                                   HumanMessage(content=material)], max_tokens=max_tokens * 3 // 2), title)
        except Exception as exc:
            logger.warning("writer rules: section %r cut, rewriting failed: %s", title, exc)
            return ""
    with ThreadPoolExecutor(max_workers=len(cut)) as pool:
        redone = dict(zip(cut, pool.map(again, cut)))
    out, log = list(sections), []
    for i, text in redone.items():
        title, old = sections[i]
        if text and not _looks_cut(text):
            out[i], how = (title, text), "rewritten with more tokens"
        else:
            out[i], how = (title, _trim_to_sentence(text or old)), "trimmed to the last sentence"
        log.append({"section": title, "kind": "cut", "how": how, "sentence": old[-160:]})
    return out, log


def review_parere(draft: str, case_text: str, article_rows: List[Dict[str, Any]], analysis: Dict[str, Any]) -> str:
    """WRITER_RULES_ENABLED on a parere written in one call (no plan): review_section over the whole draft."""
    try:
        text, log = review_section("parere", draft, case_text, article_rows, [], dg._stance_party(analysis))
    except Exception as exc:
        logger.warning("writer rules: review of the one-call parere failed, draft kept: %s", exc)
        return draft
    dg.vlog("case_analysis_writer_rules", _writer_log(log))
    return text or draft


def _writer_log(log: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"entries": log, "cut": sum(1 for e in log if e["kind"] == "cut"),
            "non_application": sum(1 for e in log if e["kind"] == "non_application"),
            "burden": sum(1 for e in log if e["kind"] == "burden"),
            "facts_flagged": sum(1 for e in log if e["kind"] == "fact_not_in_case"),
            "facts_filtered": dict(Counter(e["why"] for e in log if e["kind"] == "fact_flag_filtered")),
            "against_client": sum(1 for e in log if e["kind"] == "against_client")}


def write_sections(plan: Dict[str, Any], jobs: List[Tuple[str, str, str, int]], lang: str,
                   named: Optional[List[tuple]] = None,
                   documents: Optional[List[Dict[str, Any]]] = None,
                   article_rows: Optional[List[Dict[str, Any]]] = None, case_text: str = "",
                   coverage: Optional[Callable[[str], Dict[str, Dict[str, List[str]]]]] = None,
                   standard: Optional[Dict[str, str]] = None,
                   questions: Optional[List[str]] = None,
                   checked: Optional[List[tuple]] = None) -> Optional[List[tuple]]:
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
    system = _section_system(plan, lang)

    def write(job):
        title, instruction, material, max_tokens = job
        messages = [SystemMessage(content=f"{system}\n\n{instruction}"), HumanMessage(content=material)]
        try:
            return dg._call_chat(messages, max_tokens=max_tokens)
        except Exception as exc:
            logger.warning("plan-then-write: section %r failed, trying once more: %s", title, exc)
            return dg._call_chat(messages, max_tokens=max_tokens)

    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            texts = list(pool.map(write, jobs))
    except Exception as exc:
        logger.warning("plan-then-write: a section failed twice, the parere is written in one call: %s", exc)
        return None
    finally:
        dg.vlog("case_analysis_sections_llm", duration_ms=(time.time() - t0) * 1000)
    sections = [(job[0], _without_heading(text, job[0])) for job, text in zip(jobs, texts)]
    if coverage is not None:
        sections = _cover_blocks(sections, jobs, system, coverage, standard)
    writer_log: List[Dict[str, Any]] = []
    if _writer_rules() and all(text for _, text in sections):
        sections, writer_log = _complete_sections(sections, jobs, system)
    if _discipline() and all(text for _, text in sections):
        t1 = time.time()
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            done = list(pool.map(lambda pair: _cite_only_material(pair[0][0], pair[1], pair[0][2], named or [],
                                                                  documents or []), zip(jobs, (t for _, t in sections))))
        removed = [entry for _, log in done for entry in log]
        dg.vlog("case_analysis_citation_discipline", {
            "removed": removed, "count": sum(len(e["removed"]) for e in removed),
            "rephrased": sum(1 for e in removed if e["how"] == "rephrased"),
            "dropped": sum(1 for e in removed if e["how"] == "sentence dropped" and e.get("kind") != "dangling"),
            "dangling": sum(1 for e in removed if e.get("kind") == "dangling")},
            duration_ms=(time.time() - t1) * 1000)
        sections = [(title, text or original) for (title, original), (text, _) in zip(sections, done)]
    if _writer_rules() and all(text for _, text in sections):
        t2 = time.time()
        with ThreadPoolExecutor(max_workers=len(sections)) as pool:
            reviewed = list(pool.map(lambda pair: review_section(pair[0], pair[1], case_text, article_rows, documents,
                                                                 plan.get("parte_assistita") or ""), sections))
        writer_log += [entry for _, log in reviewed for entry in log]
        dg.vlog("case_analysis_writer_rules", _writer_log(writer_log), duration_ms=(time.time() - t2) * 1000)
        sections = [(title, text or original) for (title, original), (text, _) in zip(sections, reviewed)]
    if _review_fixes() and all(text for _, text in sections):
        t3 = time.time()
        with ThreadPoolExecutor(max_workers=len(sections)) as pool:
            fixed = list(pool.map(lambda pair: rewrite_unsupported(pair[0], pair[1], case_text, article_rows),
                                  sections))
        entries = [entry for _, log in fixed for entry in log]
        dg.vlog("case_analysis_unsupported_claims", {
            "entries": entries, "rewritten": sum(1 for e in entries if e["how"] == "rewritten")},
            duration_ms=(time.time() - t3) * 1000)
        sections = [(title, text or original) for (title, original), (text, _) in zip(sections, fixed)]
    if any(not text for _, text in sections):
        logger.warning("plan-then-write: an empty section, the parere is written in one call")
        return None
    sections = _after_checks(plan, sections, jobs, system, questions or [])
    if grounded_review.enabled():
        # GROUNDED_REVIEW_ENABLED: each rule-sentence judged against its own section's material; the ones it does
        # not hold rewritten from it, as pointers, or (filler) dropped (grounded_review.py).
        t4 = time.time()
        try:
            reviewed = grounded_review.review(sections, jobs, documents or [], article_rows or [], checked=checked)
            sections = [(title, text or original) for (title, original), (_, text) in zip(sections, reviewed)]
        except Exception as exc:
            logger.warning("grounded review failed, sections kept: %s", exc)
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
                                 max_tokens=200 + 160 * len(spans), temperature=0)
    except Exception as exc:
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
    return text, log


def write_parere(facts: str, request: str, analysis: Dict[str, Any], article_rows: List[Dict[str, Any]],
                 ruling_rows: List[Dict[str, Any]], articles: str, rulings: str, principles: str,
                 lang: str = "it", documents: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
    """The parere from a plan, section by section, or None (the caller writes it in one call).
    `articles` and `rulings` are the draft prompt's own blocks (_fit_parere_prompt); `documents` (the
    corpus' names) let _discipline() read a citation of a law by its name."""
    references: Dict[str, List[Dict[str, Any]]] = {}
    for row in ruling_rows:
        reference = dg._ruling_reference(row)
        if reference:
            references.setdefault(reference, []).append(row)
    plan = make_plan(facts, request, analysis, articles, rulings, principles, article_rows, references)
    if plan is None:
        return None
    case = dg._case_text(facts, request)
    # COMMA_MATERIAL_ENABLED: each article's commi chosen for the plan issue that places it; a comma the plan
    # or the case names, read now when the row lacks it (one read, no model call).
    commi = comma_material.for_plan(plan, article_rows, case)
    if commi:
        dg.vlog("comma_material_plan", commi)
    named = list(dict.fromkeys(dg._refs_in(case) + dg._cited_articles(case, set(), documents or [])))
    use_of = (lambda text: material_use(analysis, text)) if material_use(analysis, "") else None
    sections = write_sections(plan, section_jobs(plan, facts, request, analysis, article_rows, references,
                                                 principles, _keep_all()), lang, named, documents,
                              article_rows, case, coverage=use_of, standard=standard_blocks(analysis),
                              questions=explicit_questions(request, facts) if _review_fixes() else [],
                              checked=grounded_review.strict_refs(article_rows, named, analysis)
                              if grounded_review.strict() else None)
    if sections is None:
        return None
    if use_of is not None:
        final = use_of(next((text for title, text in sections if title == SECTION_TITLES[4]), ""))
        dg.vlog("case_analysis_material_use", {
            "offered": {b: u["offered"] for b, u in final.items()}, "written": {b: u["written"] for b, u in final.items()},
            "missing": _missing_blocks(final)})
    return "\n\n".join(f"**{i}. {title}**\n\n{text}" for i, (title, text) in enumerate(sections, 1))
