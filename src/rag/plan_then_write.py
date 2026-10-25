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
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from . import defensive_generation as dg
from . import method_candidates, practical_checklist

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
# that fits no issue of the plan is left out, never put in a catch-all issue.
_GRADED_KEEP = 12
_MAX_ISSUES_WIDE, _WIDE_APPLICABLE = 7, 8
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
    return _PLAN_SYSTEM.replace("da due a cinque", f"da due a {_ISSUE_COUNT_WORDS.get(max_issues, max_issues)}")


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
    if not dg.feature_flags.enabled(method_candidates.CHECKLIST_GATED_FLAG):
        return practical_checklist.checks_for(areas, posture)
    through = _through_check(article_rows or [], facts, request)
    return practical_checklist.gated_checks(areas, posture, lambda ref: any(_same_article(ref, t) for t in through))


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
    pipeline's order; the exempt and the unchecked ones are not among them. CHECK_GRADED_ENABLED: those
    graded 2 or 3, at most _GRADED_KEEP, by grade, then fusion rank, then the pipeline's order."""
    if not _graded():
        return list(dict.fromkeys(r["_article"] for r in article_rows
                                  if r.get("_article") and r.get("_applicability") in ("applicabile", "alternativa")))
    best: Dict[tuple, tuple] = {}
    for index, row in enumerate(article_rows):
        ref, grade = row.get("_article"), row.get("_grade")
        if ref and isinstance(grade, int) and grade >= 2:
            key = (-grade, row.get("_fused_rank") or 10 ** 6, index)
            best[ref] = min(best.get(ref, key), key)
    return sorted(best, key=lambda ref: best[ref])[:_GRADED_KEEP]


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
                   catch_all: bool = True) -> Tuple[Dict[str, Any], List[tuple], Dict[tuple, str], List[tuple]]:
    """(plan, placed, grouped, added): every checked article in the plan (PLAN_KEEP_CHECKED_ENABLED).
    One the plan left out goes under the issue whose title and argument share the most words with its
    rubrica (and with the issue the graded check named for it, issue_of); with none sharing a word, into
    a last "Altre norme rilevanti" issue, or, at max_issues issues, into the last one -- or, without
    catch_all (CHECK_GRADED_ENABLED), nowhere: it is not forced into the plan."""
    held = {ref for q in plan["questioni"] for ref in q["norme"]}
    placed = [ref for ref in checked if ref in held]
    issues = [{**q, "norme": list(q["norme"])} for q in plan["questioni"]]
    grouped: Dict[tuple, str] = {}
    rest: List[tuple] = []

    def stems(text):
        return {w[:5] for w in re.findall(r"\w{4,}", (text or "").lower()) if w not in _HEADING_STOP}
    for ref in (r for r in checked if r not in held):
        words = stems(f"{headings.get(ref, '')} {(issue_of or {}).get(ref, '')}")
        best = max(issues, key=lambda q: len(words & stems(f"{q['titolo']} {q['argomento']}")))
        if words & stems(f"{best['titolo']} {best['argomento']}"):
            best["norme"].append(ref)
            grouped[ref] = best["titolo"]
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
                applicable: Optional[List[tuple]] = None, must_place: Any = False) -> str:
    base = dg._parere_human(facts, analysis, articles, rulings, request, principles)
    base = base[:base.rfind("Redigi ora il parere.")].rstrip() if "Redigi ora il parere." in base else base
    verifiche = "\n".join(f"- {c}" for c in checks)
    # VOTE_POOL_ENABLED: the articles the rulings on similar facts cite, for the plan to weigh
    # issue by issue (prod, 24 Oct: 1158, 1218, 1453, 56, 378 ... found only this way).
    votate = ", ".join(f"[art. {n} {c}]" for n, c in voted or [])
    applicabili = ", ".join(f"[art. {n} {c}]" for n, c in applicable or [])
    return (f"PARTE ASSISTITA (dall'analisi del caso): {dg._stance_party(analysis)}\n\n{base}\n\n"
            + (f"VERIFICHE PER LA FASE E LA MATERIA:\n{verifiche}\n\n" if verifiche else "")
            + (f"NORME PIÙ PERTINENTI AL CASO: {applicabili}. Metti ciascuna tra le norme della questione che "
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


def _heading_shares_title(heading: str, title: str) -> bool:
    """The article's heading and the issue's title share a content word (5-letter stems):
    "Furto in abitazione e furto con strappo" and "Tentativo di furto in abitazione" do."""
    def stems(text):
        return {w[:5] for w in re.findall(r"\w{4,}", (text or "").lower()) if w not in _HEADING_STOP}
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
    return dg._chat_json([SystemMessage(content=_PLAN_CHECK_SYSTEM), HumanMessage(content=human)],
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
    applicable = (_checked(article_rows) if keep_checked
                  else _applicable(article_rows, facts, request, analysis) if keep_all else [])
    max_issues = _MAX_ISSUES_WIDE if keep_checked or len(applicable) >= _WIDE_APPLICABLE else _MAX_ISSUES
    system = _plan_system(max_issues)
    human = _plan_human(facts, request, analysis, _with_headings(articles, headings), rulings, principles,
                        _checks(analysis, article_rows, facts, request), voted, applicable,
                        must_place="graded" if keep_checked and _graded() else keep_checked)
    messages = [SystemMessage(content=system), HumanMessage(content=human)]
    t0 = time.time()
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
    except Exception as exc:
        logger.warning("plan-then-write: no plan, the parere is written in one call: %s", exc)
        return None
    finally:
        dg.vlog("case_analysis_plan_llm", duration_ms=(time.time() - t0) * 1000)
    checked: Dict[str, List[str]] = {"off_topic": [], "contradicted": [], "kept_by_heading": [], "kept_by_vote": []}
    if plan:
        plan, checked = check_plan(plan, article_rows, headings, applicable if keep_all or keep_checked else None)
    if plan and keep_checked:
        issue_of = {r["_article"]: r.get("_grade_issue") or "" for r in article_rows if r.get("_article")}
        plan, placed, grouped, added = _group_checked(plan, applicable, headings, max_issues, issue_of,
                                                      catch_all=not _graded())
        dg.vlog("case_analysis_plan_applicable", {
            "mode": "checked", "applicable": [f"{n} {c}" for n, c in applicable], "max_issues": max_issues,
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
        "voted_offered": [f"{n} {c}" for n, c in voted]})
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
    if answers and plan["risposte"]:
        lines.append("Risposte ai quesiti:")
        lines += [f"- {a['quesito']} -> {a['risposta']}" if a["quesito"] else f"- {a['risposta']}"
                  for a in plan["risposte"]]
    if issues:
        lines.append("Questioni, in ordine di forza:")
        for i, q in enumerate(plan["questioni"], 1):
            norms = ", ".join(f"art. {n} {c}" for n, c in q["norme"]) or "-"
            cited = f"; pronunce: {', '.join(q['pronunce']) or '-'}" if rulings else ""
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
    cited = list(dict.fromkeys(p for q in plan["questioni"] for p in q["pronunce"]))
    return dg._source_block([references[p][0] for p in cited if references.get(p)], chars)


def _section_system(plan: Dict[str, Any], lang: str) -> str:
    from .language import language_display_name
    party = plan["parte_assistita"] or "la parte assistita"
    position = f" ({plan['posizione']})" if plan["posizione"] else ""
    return (
        "Sei un avvocato italiano con lunga esperienza di contenzioso. Scrivi UNA sezione di un parere motivato "
        f"nell'interesse di {party}{position}, seguendo il PIANO già deciso: stessa parte, stessa questione "
        "centrale, stessi argomenti, stesse norme e pronunce.\n\n"
        "REGOLE:\n"
        "- Scrivi solo il testo della sezione, senza il suo titolo. Il parere ha sei sezioni (risposte ai "
        "quesiti; inquadramento e fatti; norme e giurisprudenza; applicazione al caso; profili processuali e "
        "termini; conclusioni): non ripetere ciò che spetta alle altre, sii essenziale.\n"
        "- Non concedere mai un elemento contestato dell'accusa o della pretesa: una posizione che lo presuppone "
        "va presentata solo in via subordinata.\n"
        "- Norme: cita solo quelle del PIANO o dei TESTI forniti, sempre con il codice (\"art. 1384 c.c.\", mai "
        "\"art. 1384\" da solo); riporta tra virgolette solo frasi presenti nei testi forniti.\n"
        "- Pronunce: cita solo quelle del PIANO, con il riferimento esatto indicato; senza riferimento non "
        "richiamare orientamenti giurisprudenziali. I fatti di una pronuncia non sono fatti del caso.\n"
        "- Non aggiungere fatti, date, importi o atti che il caso non contiene: se un dato manca, indica che cosa "
        "va verificato.\n"
        "- Prosa argomentata in paragrafi; elenchi puntati solo per prove da acquisire e attività da compiere.\n"
        f"- Scrivi in {language_display_name(lang)}."
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
    procedural = [ref for ref in method_candidates.candidate_refs([], posture) if ref not in refs]
    no_rulings = " In questa sezione non citare pronunce."

    def block(title: str, body: str) -> str:
        return f"{title}:\n{body}" if body.strip() else ""
    every = (" Esamina una per una TUTTE le norme del PIANO, nessuna esclusa, ciascuna indicata con numero e "
             "codice o legge come nel PIANO (\"art. 95 Codice degli Appalti 2026\", \"art. 1784 c.c.\")."
             if keep_all else "")
    every_issue = (f" Per la questione \"{OTHER_NORMS_TITLE}\", una o due frasi per ciascuna norma, con numero e "
                   "codice, su come si applica ai fatti." if keep_all else "")

    def material(*parts: str) -> str:
        return "\n\n".join(p for p in parts if p)

    return [
        (SECTION_TITLES[0],
         "SEZIONE: Risposte ai quesiti. Una risposta diretta e motivata a ciascun quesito, nell'ordine in cui è "
         "posto, a partire dalle risposte del PIANO quando ci sono, altrimenti dalle sue questioni; se la "
         "questione centrale non è tra i quesiti, rispondi prima a essa. Ogni risposta in due-quattro frasi, con "
         "la norma su cui si fonda." + no_rulings,
         material(case, block("QUESITI", tasks), _plan_text(plan, answers=True, rulings=False)), 600),
        (SECTION_TITLES[1],
         "SEZIONE: Inquadramento e questione centrale. La qualificazione giuridica della vicenda (la norma "
         "principale e quelle alternative o subordinate, e perché), la questione centrale e perché decide "
         f"l'esito; poi ogni fatto del caso valutato dal punto di vista di {party}: favorevole, sfavorevole o "
         f"neutro per {party}, e perché, in una frase. Non illustrare le norme nel dettaglio (sezione 3) né "
         "sviluppare gli argomenti (sezione 4)." + no_rulings,
         material(case, block("FATTI DEL CASO", facts_list), _plan_text(plan, rulings=False),
                  block("TESTI DELLE NORME", texts(refs, per_article(8000, 700)))), 1000),
        (SECTION_TITLES[2],
         "SEZIONE: Norme e giurisprudenza. Per ogni norma del PIANO: che cosa stabilisce, citando il testo "
         "fornito, e come si ripartisce l'onere della prova; per ogni pronuncia del PIANO: il principio che "
         "afferma, con il suo riferimento, e che cosa comporta per questo caso, senza svolgere gli argomenti "
         "(sezione 4). Solo norme e pronunce del PIANO; tra virgolette solo frasi dei testi e dei passi forniti."
         + every,
         material(case, _plan_text(plan),
                  block("TESTI DELLE NORME", texts(refs, per_article(15000, dg._ARTICLE_CHARS))),
                  block("PASSI DELLE PRONUNCE", _ruling_texts(plan, references, 1200)),
                  block("PRINCIPI DALLA GIURISPRUDENZA", principles)), 1600 if keep_all else 1200),
        (SECTION_TITLES[3],
         "SEZIONE: Applicazione al caso. Per ogni questione del PIANO, nell'ordine, con un sottotitolo: "
         f"l'argomento di {party} applicato ai fatti del caso, con le norme e le pronunce del piano; le prove da "
         "acquisire; l'obiezione più forte della controparte e la replica. Poi le tesi subordinate del PIANO. "
         "Prosa, senza ripetere per ogni questione lo stesso schema a voci; tra virgolette solo frasi dei testi e "
         "dei passi forniti." + every_issue,
         material(case, block("FATTI DEL CASO", facts_list), _plan_text(plan, issues="full", subordinate=True),
                  block("TESTI DELLE NORME", texts(refs, per_article(9000, 600))),
                  block("PASSI DELLE PRONUNCE", _ruling_texts(plan, references, 600))), 2200 if keep_all else 1800),
        (SECTION_TITLES[4],
         "SEZIONE: Profili processuali e termini. La fase in cui si trova la vicenda e che cosa consente di fare "
         f"a {party}: gli atti da compiere e i loro termini, con l'articolo che li prevede; a chi rivolgersi o chi "
         "convenire; le voci di danno o le domande da proporre, anche in via subordinata. Usa le VERIFICHE solo "
         "se i fatti del caso le rendono pertinenti, dicendo perché, e solo quelle che la legge consente a "
         f"{party} nella sua posizione; un termine che dipende da una data che il caso non indica è una verifica "
         "da fare." + no_rulings,
         material(case, block("FASE", frame.get("procedural_posture") or ""),
                  _plan_text(plan, issues="", subordinate=True, rulings=False),
                  block("VERIFICHE PER QUESTA FASE E MATERIA",
                        "\n".join(f"- {c}" for c in _checks(analysis, article_rows, facts, request))),
                  block("TESTI DELLE NORME PROCESSUALI", texts(procedural, 500))), 900),
        (SECTION_TITLES[5],
         "SEZIONE: Conclusioni. In due o tre paragrafi: gli argomenti in ordine di forza, come nel PIANO; una "
         f"stima motivata del rischio, tenendo conto dei punti deboli; la condotta consigliata a {party}."
         + no_rulings,
         _plan_text(plan, subordinate=True, weak=True, rulings=False), 500),
    ]


def _without_heading(text: str, title: str) -> str:
    """The section's text without a heading line the model repeated anyway."""
    text = (text or "").strip()
    first, _, rest = text.partition("\n")
    bare = re.sub(r"^[#>*\s]*(?:\d+\.\s*)?|[*#:\s]+$", "", first)
    return rest.strip() if bare.lower() == title.lower() else text


def write_sections(plan: Dict[str, Any], jobs: List[Tuple[str, str, str, int]], lang: str) -> Optional[List[tuple]]:
    """[(title, text)] for the jobs, written in parallel; a failed section is tried once more.
    None if one still fails or comes back empty."""
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
    if any(not text for _, text in sections):
        logger.warning("plan-then-write: an empty section, the parere is written in one call")
        return None
    return sections


def write_parere(facts: str, request: str, analysis: Dict[str, Any], article_rows: List[Dict[str, Any]],
                 ruling_rows: List[Dict[str, Any]], articles: str, rulings: str, principles: str,
                 lang: str = "it") -> Optional[str]:
    """The parere from a plan, section by section, or None (the caller writes it in one call).
    `articles` and `rulings` are the draft prompt's own blocks (_fit_parere_prompt)."""
    references: Dict[str, List[Dict[str, Any]]] = {}
    for row in ruling_rows:
        reference = dg._ruling_reference(row)
        if reference:
            references.setdefault(reference, []).append(row)
    plan = make_plan(facts, request, analysis, articles, rulings, principles, article_rows, references)
    if plan is None:
        return None
    sections = write_sections(plan, section_jobs(plan, facts, request, analysis, article_rows, references,
                                                 principles, _keep_all()), lang)
    if sections is None:
        return None
    return "\n\n".join(f"**{i}. {title}**\n\n{text}" for i, (title, text) in enumerate(sections, 1))
