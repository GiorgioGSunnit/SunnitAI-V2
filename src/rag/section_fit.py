"""Section prompts within the model's context (v48; v47's l3, h1, n3, n7: every section call over 32,768 tokens, the
parere written in one call).

fit(): before a section call, its material shrinks until the estimated prompt (system, instruction and material, at
defensive_generation._estimate_tokens' 3 characters a token, more than the tokenizer uses) plus its max_tokens and a
margin fits MODEL_CONTEXT_TOKENS. What goes, in this order, one piece at a time, the largest first:
  1. the rows PLAN_RULE_INCLUDE put in the plan (plan_source "rule"): their text in TESTI DELLE NORME, their passages;
  2. the extra passages of PASSI CON ANCORA: an article's second comma, then the passages of an article that is not the
     first (governing) norm of its issue;
  3. the rulings: their passages (PASSI DELLE PRONUNCE, a ruling's anchor), then PRINCIPI DALLA GIURISPRUDENZA.
Never the case, the facts, the plan or another block. Nothing left to drop and still too long: SectionTooLarge (the
parere then goes to the one-call fallback). A dropped passage leaves the section's anchors (Catalogue.section_ids), so
anchored_writing.verify never counts it as given. Logged per section as case_analysis_section_fit.

refit(): after a context-length error, the same with the budget the server's own count implies (its prompt tokens
over our estimate); a refit that changes nothing raises, never the identical prompt again.

halves(): NORME_SPLIT's two materials over disjoint article sets (plan_then_write.write_sections)."""

import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

CONTEXT_ENV = "MODEL_CONTEXT_TOKENS"
DEFAULT_CONTEXT = 32768
MARGIN_TOKENS = 300                 # ai_chat._RETRY_MARGIN_TOKENS
TEXTS_TITLES = ("TESTI DELLE NORME",)
PASSAGES_TITLE = "PASSI CON ANCORA"
RULINGS_TITLE = "PASSI DELLE PRONUNCE"
PRINCIPLES_TITLE = "PRINCIPI DALLA GIURISPRUDENZA"

_TITLE_RE = re.compile(r"(?m)^([A-ZÀÈÉÌÒÙ][A-ZÀÈÉÌÒÙ0-9 ’'\-/().,]{2,}):\n")
_ENTRY_RE = re.compile(r"\n\n(?=\[art\. )")
_ENTRY_HEAD_RE = re.compile(r"^\[art\. ([^\]]+)\]")
_PASSAGE_RE = re.compile(r"^\[(A\d+)\] (.+?): ")
_PASSAGE_ART_RE = re.compile(r"^art\. (.+?)(?:, co\. [\w-]+)?$")
_RULING_ENTRY_RE = re.compile(r"\n\n(?=\[)")


class SectionTooLarge(ValueError):
    pass


def context_tokens() -> int:
    try:
        return int(os.getenv(CONTEXT_ENV, "") or DEFAULT_CONTEXT)
    except ValueError:
        return DEFAULT_CONTEXT


def _estimate(text: str) -> int:
    from . import defensive_generation as dg
    return dg._estimate_tokens(text)


def key(label: str) -> str:
    return " ".join(str(label).lower().replace(".", " ").split())


def ref_keys(ref: Sequence[str]) -> set:
    from . import plan_then_write as ptw
    return {key(f"{ref[0]} {ref[1]}"), key(f"{ref[0]} {ptw._code_key(ref[1])}")}


# --- the material's blocks ------------------------------------------------------------------------------------------

def blocks(material: str) -> List[List[str]]:
    """[[title, body]] in order; text before the first title under title ""."""
    out: List[List[str]] = []
    marks = list(_TITLE_RE.finditer(material or ""))
    if not marks or marks[0].start() > 0:
        head = (material or "")[:marks[0].start() if marks else None].strip("\n")
        if head:
            out.append(["", head])
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(material)
        out.append([m.group(1), material[m.end():end].strip("\n")])
    return out


def join(parts: List[List[str]]) -> str:
    return "\n\n".join((f"{t}:\n{b}" if t else b) for t, b in parts if b.strip())


class Pieces:
    """The droppable pieces of a material: article entries of the texts' blocks, passage lines, ruling entries."""

    def __init__(self, material: str, anchors=None):
        self.parts = blocks(material)
        self.anchors = anchors

    def text(self) -> str:
        return join(self.parts)

    def _passage_ref(self, pid: str, label: str) -> Tuple[Optional[set], bool]:
        """(the passage's article keys or None for a ruling, ruling?)."""
        p = (self.anchors.passages.get(pid) if self.anchors is not None else None) or {}
        if p.get("ruling") or p.get("kind") == "ruling":
            return None, True
        if p.get("ref"):
            return ref_keys(p["ref"]), False
        m = _PASSAGE_ART_RE.match(label)
        return ({key(m.group(1))}, False) if m else (None, True)

    def entries(self) -> List[Dict[str, Any]]:
        """Every droppable piece: {"block", "index", "kind" (text / passage / ruling / principles), "keys", "pid",
        "label", "chars"}."""
        out = []
        for b, (title, body) in enumerate(self.parts):
            if title in TEXTS_TITLES:
                for i, entry in enumerate(_ENTRY_RE.split(body)):
                    m = _ENTRY_HEAD_RE.match(entry)
                    if m:
                        out.append({"block": b, "index": i, "kind": "text", "keys": {key(m.group(1))}, "pid": None,
                                    "label": f"art. {m.group(1)}", "chars": len(entry)})
            elif title == PASSAGES_TITLE:
                for i, line in enumerate(body.split("\n")):
                    m = _PASSAGE_RE.match(line)
                    if m:
                        keys, ruling = self._passage_ref(m.group(1), m.group(2))
                        out.append({"block": b, "index": i, "kind": "ruling" if ruling else "passage",
                                    "keys": keys or set(), "pid": m.group(1), "label": f"[{m.group(1)}] {m.group(2)}",
                                    "chars": len(line)})
            elif title == RULINGS_TITLE:
                for i, entry in enumerate(_RULING_ENTRY_RE.split(body)):
                    out.append({"block": b, "index": i, "kind": "ruling", "keys": set(), "pid": None,
                                "label": entry.split("\n", 1)[0][:80], "chars": len(entry)})
            elif title == PRINCIPLES_TITLE:
                out.append({"block": b, "index": None, "kind": "principles", "keys": set(), "pid": None,
                            "label": PRINCIPLES_TITLE, "chars": len(body)})
        return out

    def drop(self, entries: List[Dict[str, Any]]) -> None:
        """Drop these pieces (entries() of the current parts)."""
        by_block: Dict[int, set] = {}
        for e in entries:
            by_block.setdefault(e["block"], set()).add(e["index"])
        for b, gone in by_block.items():
            title, body = self.parts[b]
            if None in gone:
                self.parts[b][1] = ""
            elif title in TEXTS_TITLES:
                self.parts[b][1] = "\n\n".join(x for i, x in enumerate(_ENTRY_RE.split(body)) if i not in gone)
            elif title == PASSAGES_TITLE:
                self.parts[b][1] = "\n".join(x for i, x in enumerate(body.split("\n")) if i not in gone)
            elif title == RULINGS_TITLE:
                self.parts[b][1] = "\n\n".join(x for i, x in enumerate(_RULING_ENTRY_RE.split(body)) if i not in gone)
        # a block left with its title alone goes
        self.parts = [[t, b] for t, b in self.parts
                      if b.strip() and not (t == PASSAGES_TITLE and not any(_PASSAGE_RE.match(x) for x in b.split("\n")))]


def _order(pieces: Pieces, plan: Dict[str, Any], rule_refs: Sequence[tuple]) -> List[List[Dict[str, Any]]]:
    """The droppable pieces, in their three tiers (fit's order), each tier the largest first; a rule row's text and
    passages go together."""
    entries = pieces.entries()
    rule_keys = {k for ref in rule_refs for k in ref_keys(ref)}
    governing = {k for q in plan.get("questioni") or [] if q.get("norme") for k in ref_keys(q["norme"][0])}
    rule: Dict[str, List[Dict[str, Any]]] = {}
    for e in entries:
        if e["kind"] in ("text", "passage") and e["keys"] & rule_keys:
            rule.setdefault(sorted(e["keys"] & rule_keys)[0], []).append(e)
    taken = {id(e) for group in rule.values() for e in group}
    rule_groups = sorted(rule.values(), key=lambda g: -sum(e["chars"] for e in g))
    passages = [e for e in entries if e["kind"] == "passage" and id(e) not in taken]
    seen: set = set()
    second, other = [], []
    for e in passages:                       # in the block's order: an article's first passage comes first
        k = frozenset(e["keys"])
        if k in seen:
            second.append(e)
        else:
            seen.add(k)
            if not e["keys"] & governing:
                other.append(e)
    extra = sorted(second, key=lambda e: -e["chars"]) + sorted(other, key=lambda e: -e["chars"])
    rulings = sorted((e for e in entries if e["kind"] == "ruling"), key=lambda e: -e["chars"]) + \
        [e for e in entries if e["kind"] == "principles"]
    return [[g for g in rule_groups], [[e] for e in extra], [[e] for e in rulings]]


def fit(title: str, prompt: str, material: str, max_tokens: int, plan: Dict[str, Any], rule_refs: Sequence[tuple] = (),
        anchors=None, budget: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
    """(material, log): the material within the budget (the context less max_tokens and MARGIN_TOKENS; prompt: the
    system and instruction); SectionTooLarge when dropping every droppable piece is not enough."""
    limit = budget if budget is not None else context_tokens() - max_tokens - MARGIN_TOKENS
    fixed = _estimate(prompt)
    before = fixed + _estimate(material)
    log = {"section": title, "budget": limit, "before": before, "after": before, "dropped": [], "fits": before <= limit}
    if before <= limit:
        return material, log
    pieces = Pieces(material, anchors)
    tier_names = ("rule row", "extra passage", "ruling")
    for tier, groups in zip(tier_names, _order(pieces, plan, rule_refs)):
        for group in groups:
            current = fixed + _estimate(pieces.text())
            if current <= limit:
                break
            # the entries of the current parts that are this group (indexes shift as pieces go)
            live = pieces.entries()
            wanted = {(e["block"], e["index"], e["label"]) for e in group}
            hits = [e for e in live if (e["block"], e["index"], e["label"]) in wanted]
            if not hits:      # indexes moved: match by label and kind
                labels = {(e["kind"], e["label"]) for e in group}
                hits = [e for e in live if (e["kind"], e["label"]) in labels]
            pieces.drop(hits)
            for e in hits:
                log["dropped"].append({"kind": tier, "what": e["label"], "tokens": _estimate("x" * e["chars"])})
                if e.get("pid") and anchors is not None and title in anchors.section_ids:
                    anchors.section_ids[title] = [i for i in anchors.section_ids[title] if i != e["pid"]]
    out = pieces.text()
    log["after"] = fixed + _estimate(out)
    log["fits"] = log["after"] <= limit
    if not log["fits"]:
        raise SectionTooLarge(f"section {title!r}: {log['after']} estimated tokens for a budget of {limit} with "
                              "nothing left to drop")
    return out, log


def refit(title: str, prompt: str, material: str, max_tokens: int, plan: Dict[str, Any], sizes: Tuple[int, int],
          rule_refs: Sequence[tuple] = (), anchors=None) -> Tuple[str, Dict[str, Any]]:
    """After a context-length error ((server limit, its prompt tokens), ai_chat._overflow_sizes): fit() again within
    the budget the server's count implies; SectionTooLarge when that changes nothing."""
    limit, counted = sizes
    ours = _estimate(prompt) + _estimate(material)
    ratio = max(1.0, counted / max(1, ours))
    budget = int((limit - max_tokens - MARGIN_TOKENS) / ratio)
    out, log = fit(title, prompt, material, max_tokens, plan, rule_refs, anchors, budget=min(budget, ours - 1))
    log.update(refit=True, server_tokens=counted, ratio=round(ratio, 2))
    if out == material:
        raise SectionTooLarge(f"section {title!r}: the refit dropped nothing")
    return out, log


def halves(material: str, first: Sequence[tuple], second: Sequence[tuple], anchors=None) -> Tuple[str, str]:
    """NORME_SPLIT: (material of the first half, of the second): each with the texts and passages of its own articles
    only; the rulings (their passages and principles) with the first half alone."""
    out = []
    for own, other in ((first, second), (second, first)):
        pieces = Pieces(material, anchors)
        other_keys = {k for ref in other for k in ref_keys(ref)} - {k for ref in own for k in ref_keys(ref)}
        gone = [e for e in pieces.entries() if e["kind"] in ("text", "passage") and e["keys"] & other_keys]
        if own is second:
            gone += [e for e in pieces.entries() if e["kind"] in ("ruling", "principles")]
        pieces.drop(gone)
        out.append(pieces.text())
    return out[0], out[1]
