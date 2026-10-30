"""ANCHORED_WRITING_ENABLED (default false): every rule the parere states is written from a passage named by an anchor.

fetch_audit's provenance by section: the sections that receive only the plan (Risposte ai quesiti, Conclusioni)
state rules from memory, and the others state rules their texts do not hold. With the flag:

  1. The plan names, for each issue, the passages its argument rests on ("passi": an article's comma, "art. 2051
     c.c. co. 1", or a ruling's reference), read by clean_plan. Once the plan is final, build() numbers them [A1],
     [A2], ...: each "passo" whose article is still among the issue's norms, then, for a norm the plan left without
     one, the comma that shares most words with the issue (its title, argument, objection and reply), and each
     ruling of the issue with its passage. The procedural section's own texts (its procedural articles and the legal
     terms' articles) get anchors of their own.
  2. Every section gets the passages it may rest on (block()): the plan's for every section -- so Risposte ai
     quesiti and Conclusioni, which got only the plan, now get the commi the plan cites for each issue, compact --
     the procedural ones too for the procedural section, and the plan's text shows each issue's anchors. The rule
     (ANCHOR_RULE): each sentence stating a rule, an institution, a procedural step or a holding ends with the anchor
     of its passage.
  3. After the sections, verify(): a sentence with anchors passes when every anchor exists, every article or ruling
     it cites is its anchors' own, and at least MIN_OVERLAP of its content words are in the anchors' passages; a
     sentence citing an article or a ruling without an anchor fails. A failing sentence is rewritten once from its
     anchors' passages (or the passages of the article it cites) and checked again; still failing, or with no
     passage to rewrite it from, it is removed. Then every anchor is stripped.
  4. The procedural section gets its stage's procedure vocabulary (practical_checklist.PROCEDURE_VOCABULARY) and the
     rule to name no procedural institution outside it.

Logged as anchored_writing: the passages, and per section the sentences anchored / citing without an anchor /
failed (by reason) / rewritten / removed. Off, nothing changes.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from . import feature_flags
from .verbose_logger import vlog

logger = logging.getLogger(__name__)

FLAG = "ANCHORED_WRITING_ENABLED"
MIN_OVERLAP = 0.25        # share of a sentence's content words that must be in its anchors' passages
PASSAGE_CHARS = 900
REWRITE_MARK = "FRASI DA RISCRIVERE:"   # fetch_audit cuts a rewrite call's material here (as grounded_review's)

ANCHOR_RE = re.compile(r"\[\s*(A\d+(?:\s*[,;]\s*A\d+)*)\s*\]")
_STRIP_RE = re.compile(r"[ \t]*\[\s*A\d+(?:\s*[,;]\s*A\d+)*\s*\]")
_COMMA_RE = re.compile(r"\bco(?:mma)?\.?\s*(\d+)", re.IGNORECASE)
_RULING_NUMBER_RE = re.compile(r"(?<![\d/])(\d{1,6}/\d{4})(?![\d/])")

BLOCK_TITLE = "PASSI CON ANCORA"
ANCHOR_RULE = (
    " Ogni frase che enuncia una regola, un istituto, un passaggio processuale o ciò che una pronuncia afferma termina "
    "con l'ancora del passo di PASSI CON ANCORA su cui si fonda, tra parentesi quadre ([A3], o [A3, A5]): cita la "
    "norma o la pronuncia di quel passo e ne riprende il contenuto. Una regola che nessun passo sostiene non si "
    "scrive.")
PLAN_RULE = (
    "- Per ogni questione indica in \"passi\" i commi delle sue norme e le pronunce su cui si fonda l'argomento: "
    "l'identificativo della norma con il comma (\"art. 2051 c.c. co. 1\") o il riferimento della pronuncia.\n")
PLAN_KEY = '"passi": ["art. N codice co. K", "riferimento"], '

REWRITE_SYSTEM = (
    "Sei un avvocato italiano e correggi un parere. Ricevi alcuni PASSI di norme e pronunce, ciascuno con la sua "
    "ancora, e alcune FRASI del parere, numerate, ciascuna con le ancore dei passi da cui va riscritta. Riscrivi ogni "
    "frase dicendo solo ciò che quei passi dicono, con le loro parole o parafrasandoli, citando la norma o la "
    "pronuncia di quei passi (numero e codice, o riferimento) e nessun'altra, nello stesso punto del ragionamento. "
    "Se i passi non sostengono la frase, rispondi con la frase vuota. Non usare ciò che sai. Rispondi SOLO con un "
    "array JSON: [{\"i\": numero, \"frase\": \"la frase riscritta, o vuota\"}].")


def enabled() -> bool:
    return feature_flags.enabled(FLAG)


# --- helpers ------------------------------------------------------------------------------------------------------

_STOP = set("""della delle dello degli nella nelle nello negli sulla sulle dalla dalle alla alle allo agli come anche
quando quale quali questo questa sono essere viene deve devono può possono ogni altro altra loro tale tali articolo
comma codice norma legge ovvero nonché caso parte""".split())


def stems(text: str) -> set:
    return {w[:6] for w in re.findall(r"[a-zà-ù]{4,}", (text or "").lower()) if w not in _STOP}


def strip_anchors(text: str) -> str:
    return _STRIP_RE.sub("", text or "")


def _row_text(row: Dict[str, Any]) -> str:
    s = row.get("s") or {}
    return row.get("_passage") or " ".join(str(s.get("plain_text") or s.get("abstract") or "").split())


def _commi(row: Dict[str, Any]) -> Tuple[List[str], bool]:
    """(the article's commi, numbered): COMMA_MATERIAL's when the row holds them, else its text split; numbered False
    for a sentence split (its parts are not commi)."""
    from . import comma_material
    if row.get(comma_material.COMMI_KEY):
        return list(row[comma_material.COMMI_KEY]), row.get(comma_material.FROM_KEY) != "sentences"
    parts, how = comma_material.split_commi(_row_text(row))
    return parts, how not in ("sentences", "none")


def best_comma(commi: Sequence[str], focus: str) -> int:
    """The index of the comma sharing most content words with focus (the first on a tie or none)."""
    words = stems(focus)
    best, score = 0, -1
    for i, comma in enumerate(commi):
        shared = len(words & stems(comma))
        if shared > score:
            best, score = i, shared
    return best


def parse_passi(value: Any, known: set, references: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """The plan issue's "passi": [{"ref", "comma"}] for an article among `known` (its comma or None), [{"ruling"}]
    for a reference among `references`; anything else left out."""
    from . import plan_then_write as ptw
    out: List[Dict[str, Any]] = []
    for item in ptw._texts(value):
        refs = [r for r in ptw._refs_of(item, known) if r in known]
        if refs:
            m = _COMMA_RE.search(item)
            entry = {"ref": refs[0], "comma": int(m.group(1)) if m else None}
        else:
            reference = ptw._matching_reference(item, references)
            entry = {"ruling": reference} if reference else None
        if entry and entry not in out:
            out.append(entry)
    return out


# --- the passages ---------------------------------------------------------------------------------------------------

class Catalogue:
    """The passages, by anchor id: {"id", "kind" (article / ruling), "ref" or "ruling", "label", "text"}; the plan's
    ids (each issue's in plan["questioni"][i]["ancore"]) and the procedural section's."""

    def __init__(self):
        self.passages: Dict[str, Dict[str, Any]] = {}
        self.plan_ids: List[str] = []
        self.procedural_ids: List[str] = []
        self._by_key: Dict[tuple, str] = {}

    def add(self, key: tuple, kind: str, label: str, text: str, **extra) -> Optional[str]:
        text = " ".join((text or "").split())[:PASSAGE_CHARS]
        if not text:
            return None
        if key in self._by_key:
            return self._by_key[key]
        pid = f"A{len(self.passages) + 1}"
        self.passages[pid] = {"id": pid, "kind": kind, "label": label, "text": text, **extra}
        self._by_key[key] = pid
        return pid

    def of_ref(self, ref: tuple) -> List[str]:
        from . import plan_then_write as ptw
        return [pid for pid, p in self.passages.items() if p.get("ref") and ptw._same_ref(p["ref"], ref)]

    def of_ruling(self, number: str) -> List[str]:
        return [pid for pid, p in self.passages.items() if p.get("ruling") and number in _RULING_NUMBER_RE.findall(p["ruling"])]

    def block(self, ids: Iterable[str]) -> str:
        lines = [f"[{pid}] {self.passages[pid]['label']}: {self.passages[pid]['text']}"
                 for pid in dict.fromkeys(ids) if pid in self.passages]
        return f"{BLOCK_TITLE}:\n" + "\n".join(lines) if lines else ""

    def log(self) -> List[Dict[str, Any]]:
        return [{"id": p["id"], "label": p["label"], "chars": len(p["text"])} for p in self.passages.values()]


def _article_passage(cat: Catalogue, row: Dict[str, Any], comma: Optional[int], focus: str) -> Optional[str]:
    from . import plan_then_write as ptw
    ref = row["_article"]
    commi, numbered = _commi(row)
    if not commi:
        return None
    k = comma - 1 if comma and numbered and 1 <= comma <= len(commi) else best_comma(commi, focus)
    label = f"art. {ref[0]} {ptw._code_key(ref[1])}" + (f", co. {k + 1}" if numbered and len(commi) > 1 else "")
    return cat.add(("article", ref, k), "article", label, commi[k], ref=ref, comma=k + 1 if numbered else None)


def build(plan: Dict[str, Any], article_rows: List[Dict[str, Any]], references: Dict[str, List[Dict[str, Any]]],
          procedural: Sequence[tuple] = (), terms: Sequence[Dict[str, str]] = ()) -> Catalogue:
    """The passages of the final plan, each issue's anchors set in plan["questioni"][i]["ancore"] (see the module
    docstring), then the procedural articles' and the terms' (`terms`: [{"ref", "text"}])."""
    from . import plan_then_write as ptw
    cat = Catalogue()
    from . import comma_material
    rows: Dict[tuple, Dict[str, Any]] = {}
    for row in article_rows:
        ref = row.get("_article")
        if ref and ref not in rows and (_row_text(row) or row.get(comma_material.COMMI_KEY)):
            rows[ref] = row
    for q in plan.get("questioni") or []:
        focus = " ".join(str(q.get(k) or "") for k in ("titolo", "argomento", "obiezione", "replica"))
        ids: List[str] = []
        planned = {}
        for p in q.get("passi") or []:
            if p.get("ref"):
                planned.setdefault(p["ref"], []).append(p.get("comma"))
        for ref in q.get("norme") or []:
            row = rows.get(ref) or next((r for k, r in rows.items() if ptw._same_ref(k, ref)), None)
            if row is None:
                continue
            for comma in planned.get(ref) or [None]:
                pid = _article_passage(cat, row, comma, focus)
                if pid:
                    ids.append(pid)
        for reference in list(q.get("pronunce") or []) + list(q.get("pronunce_contrarie") or []):
            hits = references.get(reference) or []
            if hits:
                pid = cat.add(("ruling", reference), "ruling", reference, _row_text(hits[0]), ruling=reference)
                if pid:
                    ids.append(pid)
        q["ancore"] = list(dict.fromkeys(ids))
        cat.plan_ids += [i for i in q["ancore"] if i not in cat.plan_ids]
    for ref in procedural:
        row = rows.get(ref) or next((r for k, r in rows.items() if ptw._same_ref(k, ref)), None)
        if row is not None:
            commi, _ = _commi(row)
            pid = cat.add(("procedural", ref), "article", f"art. {ref[0]} {ptw._code_key(ref[1])}",
                          " ".join(commi) or _row_text(row), ref=ref)
            if pid:
                cat.procedural_ids.append(pid)
    for term in terms:
        refs = [r for _, _, rs in ptw._citations(f"art. {term.get('ref', '')}", []) for r in rs]
        if refs and term.get("text"):
            pid = cat.add(("term", refs[0]), "article", f"art. {refs[0][0]} {ptw._code_key(refs[0][1])}",
                          term["text"], ref=refs[0])
            if pid and pid not in cat.procedural_ids:
                cat.procedural_ids.append(pid)
    return cat


# --- the check ------------------------------------------------------------------------------------------------------

def check(sentence: str, ids: Sequence[str], cat: Catalogue, documents: Optional[List[Dict[str, Any]]] = None,
          known: Optional[set] = None) -> Optional[str]:
    """Why the sentence fails, or None: "unknown anchor", "citation not its anchor's", "overlap" (fewer than
    MIN_OVERLAP of its content words in its anchors' passages), "unanchored citation" (no anchor, cites an article
    or a ruling)."""
    from . import plan_then_write as ptw
    body = strip_anchors(sentence)
    refs = [r for _, _, rs in ptw._citations(body, documents or [], known) for r in rs]
    rulings = _RULING_NUMBER_RE.findall(body)
    if not ids:
        return "unanchored citation" if refs or rulings else None
    if any(i not in cat.passages for i in ids):
        return "unknown anchor"
    passages = [cat.passages[i] for i in ids]
    for ref in refs:
        if not any(p.get("ref") and ptw._same_ref(p["ref"], ref) for p in passages):
            return "citation not its anchor's"
    for number in rulings:
        if not any(p.get("ruling") and number in _RULING_NUMBER_RE.findall(p["ruling"]) for p in passages):
            return "citation not its anchor's"
    words = stems(body)            # numbers and code sigle carry no stem
    if not words:
        return None
    shared = words & stems(" ".join(p["text"] for p in passages))
    return "overlap" if len(shared) / len(words) < MIN_OVERLAP else None


def _anchor_rewrite(cat: Catalogue, items: List[Tuple[str, List[str]]]) -> Dict[int, str]:
    """{index: the rewritten sentence ("" when the passages do not support it)} for one section; {} on a failure."""
    from . import defensive_generation as dg
    ids = list(dict.fromkeys(i for _, anchors in items for i in anchors))
    human = (f"PASSI:\n" + "\n".join(f"[{i}] {cat.passages[i]['label']}: {cat.passages[i]['text']}" for i in ids)
             + f"\n\n{REWRITE_MARK}\n"
             + "\n".join(f"{n}. {s} [passi: {', '.join(a)}]" for n, (s, a) in enumerate(items, 1)))
    try:
        raw = dg._chat_json([SystemMessage(content=REWRITE_SYSTEM), HumanMessage(content=human)],
                            max_tokens=200 + 140 * len(items), temperature=0)
    except Exception as exc:
        logger.warning("anchored writing: rewrite failed, the failing sentences are removed: %s", exc)
        return {}
    out: Dict[int, str] = {}
    for item in raw if isinstance(raw, list) else []:
        try:
            k = int(item.get("i")) - 1
        except (TypeError, ValueError, AttributeError):
            continue
        if 0 <= k < len(items) and k not in out:
            out[k] = " ".join(str(item.get("frase") or "").split())
    return out


def _source_ids(sentence: str, ids: List[str], cat: Catalogue, documents, known) -> List[str]:
    """The passages a failing sentence is rewritten from: its own existing anchors, else those of what it cites."""
    from . import plan_then_write as ptw
    own = [i for i in ids if i in cat.passages]
    if own:
        return own
    body = strip_anchors(sentence)
    out = [pid for _, _, rs in ptw._citations(body, documents or [], known) for r in rs for pid in cat.of_ref(r)]
    out += [pid for n in _RULING_NUMBER_RE.findall(body) for pid in cat.of_ruling(n)]
    return list(dict.fromkeys(out))


def verify(sections: List[Tuple[str, str]], cat: Catalogue, documents: Optional[List[Dict[str, Any]]] = None,
           article_rows: Optional[List[Dict[str, Any]]] = None,
           exempt: Sequence[str] = ()) -> List[Tuple[str, str]]:
    """The sections with every sentence checked against its anchors (check), a failing one rewritten once from its
    passages and checked again, else removed; every anchor stripped. Sentences of `exempt` texts (the standard blocks
    written in code) are not checked."""
    from . import plan_then_write as ptw
    known = {r["_article"] for r in article_rows or [] if r.get("_article")}
    exempt_text = " ".join(" ".join(t.split()) for t in exempt)
    plans = []
    for title, text in sections:
        failing = []
        stats = {"section": title, "sentences": 0, "anchored": 0, "unanchored_citing": 0, "failed": {},
                 "rewritten": 0, "removed": 0}
        for s, e in ptw._sentence_spans(text):
            sentence = text[s:e]
            stats["sentences"] += 1
            if exempt_text and " ".join(strip_anchors(sentence).split()) in exempt_text:
                continue
            ids = [i.strip() for m in ANCHOR_RE.finditer(sentence) for i in re.split(r"[,;]", m.group(1))]
            stats["anchored"] += bool(ids)
            why = check(sentence, ids, cat, documents, known)
            stats["unanchored_citing"] += why == "unanchored citation"
            if why:
                stats["failed"][why] = stats["failed"].get(why, 0) + 1
                failing.append((s, e, sentence, why, _source_ids(sentence, ids, cat, documents, known)))
        plans.append((title, text, failing, stats))

    def rewrite(plan):
        _, _, failing, _ = plan
        items = [(strip_anchors(sentence).strip(), src) for _, _, sentence, _, src in failing if src]
        return _anchor_rewrite(cat, items) if items else {}
    with ThreadPoolExecutor(max_workers=max(1, min(6, len(plans)))) as pool:
        replies = list(pool.map(rewrite, plans))
    out, entries, stats_out = [], [], []
    for (title, text, failing, stats), reply in zip(plans, replies):
        changes: Dict[Tuple[int, int], Optional[str]] = {}
        n = 0
        for s, e, sentence, why, src in failing:
            new = None
            if src:
                new = reply.get(n) or None
                n += 1
                if new and check(strip_anchors(new), src, cat, documents, known):
                    new = None
            how = "rewritten" if new else "removed"
            stats[how] += 1
            if new and sentence.rstrip()[-1:] in ".!?;:" and new[-1:] not in ".!?;:":
                new += sentence.rstrip()[-1]
            changes[(s, e)] = new
            entries.append({"section": title, "why": why, "how": how, "sentence": strip_anchors(sentence)[:400],
                            "from": src, **({"new": new[:400]} if new else {})})
        if changes:
            from . import grounded_review
            text = grounded_review.apply(text, changes)
            if stats["removed"]:
                text, _ = ptw.drop_dangling(text, title)
        out.append((title, strip_anchors(text).strip() or strip_anchors(text)))
        stats_out.append(stats)
    vlog("anchored_writing", {"passages": cat.log(), "sections": stats_out, "entries": entries,
                              "totals": {k: sum(st[k] for st in stats_out)
                                         for k in ("sentences", "anchored", "unanchored_citing", "rewritten",
                                                   "removed")}})
    return out
