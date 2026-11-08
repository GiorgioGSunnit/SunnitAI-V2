"""ROLE_FIT (v48, default false; the PO's red on Leonardo: art. 949 c.c., the owner's negatoria, argued for the client who
claims a servitude on another's land): after the plan, one concurrent call per substantive plan article already graded 2
or more, asking only whom the norm works for -- the client, the counterparty or both -- from the client's side
(client_side), the parties' roles and the client's positions (fact_signals.client_roles). An article answered
"controparte" stays in the material, tagged adverse (plan["norme_contrarie"]): the plan text marks it and the writer
rule (ADVERSE_NORMS_RULE) presents it only as the counterparty's argument, followed by the client's reply. A failed call
leaves its article as it was; an article without its text is not asked. Logged per article as case_analysis_role_fit.

ROLE_FIT=shadow: the same calls and log, nothing tagged and no writer rule; each answer carries the role conditions'
own verdict where they have one ("deterministic", fact_signals.deterministic_side) and whether it "agrees", for a run to
measure ROLE_FIT against them and against the eval's forbidden lists (diagnose_long) at no risk to the answers."""

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from . import feature_flags

logger = logging.getLogger(__name__)

FLAG = "ROLE_FIT"
WORKERS = 8
MAX_TOKENS = 150
TEXT_CHARS = 1500
ANSWERS = ("cliente", "controparte", "entrambi")
# procedural articles work for whoever acts: not asked
PROCEDURAL_CODES = re.compile(r"c\.p\.c|c\.p\.p|c\.p\.a|procedura|processo|disp\. att", re.IGNORECASE)
ADVERSE_MARK = "norma della controparte"
ROLE_WORDS = {"claims_servitude": "vanta una servitù o un diritto di passaggio sul fondo altrui",
              "resists_servitude": "è il proprietario che contesta una servitù vantata da altri",
              "claims_ownership": "rivendica la proprietà o la restituzione del bene",
              "possessor": "possiede o ha posseduto il bene o il passaggio",
              "landlocked_owner": "è proprietario di un fondo senza accesso alla via pubblica"}

SYSTEM = (
    "Sei un avvocato italiano. Ti vengono dati il CASO, la PARTE ASSISTITA (il cliente) con la sua posizione, i ruoli "
    "delle parti e il testo di una NORMA. Rispondi a una sola domanda: questa norma, sui fatti di questo caso, opera a "
    "favore di chi? «cliente» se fonda o sostiene la pretesa o la difesa del cliente; «controparte» se è uno strumento "
    "o una difesa della controparte contro il cliente (per esempio l'azione che spetta al proprietario contro chi "
    "vanta un diritto sul suo fondo, quando il cliente è chi lo vanta); «entrambi» se può servire all'uno e all'altro. "
    "Rispondi solo con un oggetto JSON: {\"a_favore\": \"cliente\" | \"controparte\" | \"entrambi\", \"perche\": "
    "\"una frase\"}.")


def mode() -> str:
    """"on" (ROLE_FIT=true), "shadow" (ROLE_FIT=shadow: asked and logged, nothing tagged, no writer rule) or "off"."""
    import os
    if str(os.getenv(FLAG) or "").strip().lower() == "shadow":
        return "shadow"
    return "on" if feature_flags.enabled(FLAG) else "off"


def enabled() -> bool:
    return mode() == "on"


def _grade(rows: List[Dict[str, Any]]) -> Optional[int]:
    grades = [r.get("_grade") for r in rows if isinstance(r.get("_grade"), int)]
    return max(grades) if grades else None


def candidates(plan: Dict[str, Any], article_rows: List[Dict[str, Any]]) -> List[Tuple[tuple, Dict[str, Any]]]:
    """[(ref, its row)] of the plan's substantive articles graded 2 or more, in plan order."""
    rows_of: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in article_rows or []:
        if row.get("_article"):
            rows_of.setdefault(tuple(row["_article"]), []).append(row)
    out = []
    for q in plan.get("questioni") or []:
        for ref in q.get("norme") or []:
            ref = tuple(ref)
            rows = rows_of.get(ref) or []
            if any(r == ref for r, _ in out) or PROCEDURAL_CODES.search(str(ref[1])):
                continue
            grade = _grade(rows)
            if grade is not None and grade >= 2:
                out.append((ref, rows[0]))
    return out


def _parties(signals: Optional[Dict[str, Any]]) -> str:
    facts = (signals or {}).get("facts") or {}
    return "\n".join(f"- {p.get('name')}: {p.get('role') or '-'}" for p in facts.get("parties") or [] if p.get("name"))


def human(case_text: str, party: str, signals: Optional[Dict[str, Any]], ref: tuple, row: Dict[str, Any]) -> str:
    from . import article_text, fact_signals
    facts = (signals or {}).get("facts") or {}
    roles = sorted(fact_signals.client_roles(facts, (signals or {}).get("fired") or []) - {"any"})
    s = row.get("s") or {}
    text = article_text.clause_cut(" ".join(str(s.get("plain_text") or "").split()), TEXT_CHARS)
    side = facts.get("client_side") or party
    return (f"CASO:\n{case_text[:4000]}\n\nPARTE ASSISTITA: {side}"
            + (f"\nPOSIZIONE DEL CLIENTE (dai fatti): il cliente {'; '.join(ROLE_WORDS.get(r, r) for r in roles)}"
               if roles else "")
            + (f"\n\nRUOLI DELLE PARTI:\n{_parties(signals)}" if _parties(signals) else "")
            + f"\n\nNORMA: art. {ref[0]} {ref[1]}" + (f" ({article_text.heading(s)})" if article_text.heading(s) else "")
            + f"\n{text}")


def _answer(raw: Any) -> Tuple[str, str]:
    if not isinstance(raw, dict):
        return "", ""
    said = str(raw.get("a_favore") or "").strip().lower()
    said = next((a for a in ANSWERS if said.startswith(a)), "")
    return said, " ".join(str(raw.get("perche") or "").split())[:300]


def check(plan: Dict[str, Any], article_rows: List[Dict[str, Any]], case_text: str,
          signals: Optional[Dict[str, Any]], shadow: bool = False) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """(plan with "norme_contrarie" added to, per-article log). shadow (ROLE_FIT=shadow): the plan as it was, the log
    with each answer's agreement with the role conditions alone (fact_signals.deterministic_side)."""
    from . import defensive_generation as dg
    asked = candidates(plan, article_rows)
    if not asked:
        return plan, []
    party = plan.get("parte_assistita") or ""

    def _role_fit_one(item):
        ref, row = item
        if not " ".join(str((row.get("s") or {}).get("plain_text") or "").split()):
            return {"article": f"{ref[0]} {ref[1]}", "answer": "no text", "why": "not asked: the article's text is missing"}
        try:
            raw = dg._chat_json([SystemMessage(content=SYSTEM), HumanMessage(content=human(case_text, party, signals,
                                                                                        ref, row))],
                                max_tokens=MAX_TOKENS, temperature=0)
            said, why = _answer(raw)
            return {"article": f"{ref[0]} {ref[1]}", "answer": said or "unreadable", "why": why}
        except Exception as exc:
            dg._log_exception("role_fit", exc)
            return {"article": f"{ref[0]} {ref[1]}", "answer": "failed", "why": str(exc)[:200]}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=min(WORKERS, len(asked))) as pool:
        log = list(pool.map(_role_fit_one, asked))
    from . import fact_signals
    roles = fact_signals.client_roles((signals or {}).get("facts"), (signals or {}).get("fired") or [])
    for (ref, _), e in zip(asked, log):
        side = fact_signals.deterministic_side(ref, roles)
        if side:
            e["deterministic"] = side
            if e["answer"] in ANSWERS:
                e["agrees"] = e["answer"] == side or (e["answer"] == "entrambi" and side == "cliente")
    adverse = [e["article"] for e in log if e["answer"] == "controparte"]
    compared = [e for e in log if "agrees" in e]
    dg.vlog("case_analysis_role_fit", {"mode": "shadow" if shadow else "on", "articles": log, "adverse": adverse,
                                       "agreement": {"compared": len(compared),
                                                     "agree": sum(1 for e in compared if e["agrees"])}},
            duration_ms=(time.time() - t0) * 1000)
    if shadow or not adverse:
        return plan, log
    return {**plan, "norme_contrarie": list(dict.fromkeys(list(plan.get("norme_contrarie") or []) + adverse))}, log


def is_adverse(plan: Dict[str, Any], ref: tuple) -> bool:
    return f"{ref[0]} {ref[1]}" in (plan.get("norme_contrarie") or [])
