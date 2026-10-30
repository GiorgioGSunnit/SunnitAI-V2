"""Evidence tiers of the candidate articles (27 Oct), computed once in the fusion pool (CASE_POOL_FUSION_ENABLED) and
carried on each row as _tier / _tier_why; every rule below acts only on rows that carry a tier, inside the flag of
the step it changes, so a run without the fusion pool behaves as before.

  T1  named in the request or the case; resolved from a fired fact rule (or a procedural tool / regime the facts call
      for) or from a fact's institution (FACT_ISSUES_RESOLVER_V2_ENABLED); found by an exact heading match of one of
      the analysis' institutions.
  T2  applied by MIN_RULINGS or more of the top similar rulings of the case's area (the vote or the map), or found by
      two or more independent channels.
  T3  one channel only.

Rule 1, evidence before rank at every cut-off: each capacity limit orders by tier, then grade, then fusion rank
(cut_key); a T1/T2 candidate is never dropped for capacity or for being unchecked -- past a limit it stays in the
material, ungraded. Capacity cuts reach T3 only.
Rule 2, removal needs agreement: a T1/T2 article is removed (grade 0, "incompatibile", a second opinion "no", a
qualification mismatch) only when two independent judgements agree, or one quotes a contradicting fact found in the
case's text; a single unverified negative only demotes it (_demoted: never forced into the plan, still given to the
writer). T3 as before.
Rule 3, deterministic among equals: the plan's forced list orders by grade, tier, fusion rank, article id
(forced_key); every tie-break ends on the article id.

Every cut-off is logged as evidence_cutoff (stage, limit, what was cut with its tier, any T1/T2 dropped -- target
none) and every removal or demotion as evidence_removal (stage, article, tier, judgements, verified fact).
"""

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

T1, T2, T3 = 1, 2, 3
MIN_RULINGS = 3


def tier(row: Dict[str, Any]) -> Optional[int]:
    """The row's tier, or None when tiers were not computed (the rules then do not apply)."""
    value = row.get("_tier")
    return value if value in (T1, T2, T3) else None


def evidenced(row: Dict[str, Any]) -> bool:
    """T1 or T2."""
    return tier(row) in (T1, T2)


def tiers_by_ref(rows: Iterable[Dict[str, Any]]) -> Dict[tuple, int]:
    """{article: its best tier over its rows}, for the rows that carry one."""
    out: Dict[tuple, int] = {}
    for row in rows:
        ref, t = row.get("_article"), tier(row)
        if ref and t:
            out[ref] = min(out.get(ref, t), t)
    return out


def ref_key(ref: Any) -> tuple:
    """A fixed order of article ids: code, number, suffix ("614-bis c.p.c." after "614 c.p.c.")."""
    number, code = (tuple(ref) + ("", ""))[:2] if ref else ("", "")
    m = re.match(r"(\d+)(.*)", str(number))
    return (str(code), int(m.group(1)) if m else 10 ** 6, m.group(2) if m else str(number))


def _grade(row: Dict[str, Any]) -> int:
    grade = row.get("_grade")
    if isinstance(grade, int):
        return grade
    return {"applicabile": 2, "alternativa": 2, "marginale": 1}.get(str(row.get("_applicability") or ""), -1)


def cut_key(row: Dict[str, Any]) -> tuple:
    """Rule 1: tier, then grade (higher first), then fusion rank, then article id."""
    return (tier(row) or T3, -_grade(row), row.get("_fused_rank") or 10 ** 6, ref_key(row.get("_article")))


def forced_key(row: Dict[str, Any], grade: int) -> tuple:
    """Rule 3: grade (higher first), tier, fusion rank, article id."""
    return (-grade, tier(row) or T3, row.get("_fused_rank") or 10 ** 6, ref_key(row.get("_article")))


def tier_of(row: Dict[str, Any], named: set, fact_refs: set, exact_refs: set) -> Tuple[int, str]:
    """(tier, why) of one row from what found it."""
    ref = row.get("_article")
    if ref in named:
        return T1, "named in the case"
    if row.get("_fact_institution"):           # FACT_ISSUES_RESOLVER_V2_ENABLED: reached from a fact's institution
        return T1, "fact institution"
    if ref in fact_refs or row.get("_regime"):
        return T1, "fact rule"
    if ref in exact_refs or row.get("_heading_exact"):
        return T1, "exact heading of an institution"
    rulings = int((row.get("_vote_stats") or {}).get("rulings") or 0)
    if rulings >= MIN_RULINGS:
        return T2, f"applied by {rulings} similar rulings"
    if row.get("_multi_signal") or len(set(row.get("_signals") or [])) >= 2:
        return T2, "two or more channels"
    return T3, "one channel"


def annotate(rows: List[Dict[str, Any]], named: Iterable[tuple] = (), fact_refs: Iterable[tuple] = (),
             exact_refs: Iterable[tuple] = ()) -> Dict[str, Any]:
    """Each row's _tier and _tier_why (an article's rows all get its best tier); rows that already carry a tier
    keep it. Returns the log: {"T1": [...], "T2": [...], "T3": n, "why": {...}}."""
    named, fact_refs, exact_refs = set(named), set(fact_refs), set(exact_refs)
    best: Dict[tuple, Tuple[int, str]] = {}
    for row in rows:
        ref = row.get("_article")
        if not ref:
            continue
        found = (row["_tier"], row.get("_tier_why") or "") if tier(row) else tier_of(row, named, fact_refs, exact_refs)
        if ref not in best or found[0] < best[ref][0]:
            best[ref] = found
    for row in rows:
        ref = row.get("_article")
        if ref in best:
            row["_tier"], row["_tier_why"] = best[ref]
    label = lambda ref: f"{ref[0]} {ref[1]}"  # noqa: E731
    return {"T1": sorted(label(r) for r, (t, _) in best.items() if t == T1),
            "T2": sorted(label(r) for r, (t, _) in best.items() if t == T2),
            "T3": sum(1 for t, _ in best.values() if t == T3),
            "why": {label(r): why for r, (t, why) in best.items() if t != T3}}


def cutoff_log(stage: str, limit: Any, cut: Sequence[Dict[str, Any]], kept_aside: Sequence[Dict[str, Any]] = (),
               dropped: Sequence[Dict[str, Any]] = ()) -> Dict[str, Any]:
    """The evidence_cutoff entry: the rows cut (by tier), the T1/T2 ones kept aside instead (ungraded, unforced)
    and any T1/T2 dropped all the same (target none)."""
    def items(rows):
        seen, out = set(), []
        for row in rows:
            ref = row.get("_article")
            if ref and ref not in seen:
                seen.add(ref)
                out.append({"article": f"{ref[0]} {ref[1]}", "tier": tier(row)})
        return out
    return {"stage": stage, "limit": limit, "cut": items(cut), "kept_aside": items(kept_aside),
            "t12_dropped": [i for i in items(dropped) if i["tier"] in (T1, T2)]}


def removal_entry(row: Dict[str, Any], how: str, judgements: int, why: str = "",
                  verified_fact: bool = False) -> Dict[str, Any]:
    ref = row.get("_article") or ("", "")
    return {"article": f"{ref[0]} {ref[1]}", "tier": tier(row), "how": how, "judgements": judgements,
            "verified_fact": verified_fact, "why": str(why)[:200]}


def fact_in_case(fact: str, case_text: str) -> bool:
    """A contradicting fact a judgement quotes is in the case's own text (most of its longer words)."""
    from .defensive_generation import _found_in
    return bool(str(fact or "").strip()) and _found_in(str(fact), case_text or "")
