"""GROUNDED_REVIEW_ENABLED (default false): every rule the parere states rests on the material of its section.

scripts/fetch_audit.py on run 0812 (judged): 28% of the sentences stating a rule, an institution, a procedural step
or a ruling's holding (879/3174) were not in the material of the writer call that wrote them, and 27% of the cited
claims (397/1497) cited an article the writer was never given. With the flag, after the sections are written
(plan_then_write.write_sections, after its other checks):

  1. JUDGE    each section's sentences, in batches, against that section's own material, at temperature 0, with the
              judge fetch_audit uses (JUDGE_SYSTEM is its PROVENANCE_SYSTEM): kind (regola / istituto / procedura /
              massima / altro) and verdict (supportata / assente / contraddetta).
  2. FETCH    an article a not-supported sentence cites that its material does not hold is read (one read for the
              whole parere, no model call), and the sentence judged again with that text: supported, it is kept
              as written.
  3. REWRITE  the others, one call per section: from the material when the material supports a version of the
              sentence; else as a pointer ("va verificata la disciplina dell'art. ...") keeping every citation;
              dropped only when it is pure filler and cites nothing. A rewriting that loses a citation is replaced
              by the pointer written in code. Nothing is ever added that the material does not hold, and a
              supported sentence is never touched.

Logged as grounded_review: per section the sentences judged and the verdicts, per sentence touched what was done
(rewritten / pointer / dropped / kept / kept_fetched), the articles read and their text (fetch_audit counts a fetched
article as given to the section that fetched it). Off, nothing runs.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from . import feature_flags
from .verbose_logger import vlog

logger = logging.getLogger(__name__)

FLAG = "GROUNDED_REVIEW_ENABLED"
BATCH = 12                 # sentences per judge call
MATERIAL_CHARS = 14000     # material per call, at most (fetch_audit's --material-chars)
FETCH_CHARS = 2500         # text of a fetched article
MIN_WORDS = 6              # shorter sentences are not judged (fetch_audit's MIN_WORDS)
REWRITE_MARK = "FRASI DA RISCRIVERE:"   # fetch_audit cuts a rewrite call's material here
VERDICTS = ("supportata", "assente", "contraddetta")


def enabled() -> bool:
    return feature_flags.enabled(FLAG)


# The judge of scripts/fetch_audit.py (PROVENANCE_SYSTEM), word for word: the review and the audit measure the same.
JUDGE_SYSTEM = (
    "Sei un revisore legale. Ricevi il MATERIALE che è stato dato a chi ha scritto un parere (testi di norme, "
    "passi di sentenze, elenchi di termini e strumenti) e alcune FRASI di quel parere, numerate. Per ogni frase "
    "decidi prima il tipo: \"regola\" (enuncia una regola giuridica o il contenuto di una norma), \"istituto\" "
    "(definisce o qualifica un istituto), \"procedura\" (un passaggio, un termine o un atto processuale), "
    "\"massima\" (ciò che una sentenza ha deciso o affermato), \"altro\" (fatti del caso, valutazioni, "
    "consigli pratici senza una regola). Per le frasi non \"altro\" decidi l'esito confrontandole SOLO con il "
    "MATERIALE, non con ciò che sai: \"supportata\" se il materiale contiene quella regola o quel contenuto (anche "
    "con altre parole), \"assente\" se il materiale non la contiene (anche se è vera), \"contraddetta\" se il "
    "materiale dice il contrario. Rispondi SOLO con un array JSON: [{\"i\": numero, \"tipo\": ..., \"esito\": "
    "\"supportata\"|\"assente\"|\"contraddetta\"|null, \"prova\": \"fino a 15 parole del materiale, o vuoto\"}].")

REWRITE_SYSTEM = (
    "Sei un avvocato italiano e correggi un parere. Ricevi il MATERIALE dato a chi ha scritto una sua sezione (testi "
    "di norme, passi di sentenze, termini e strumenti) e alcune FRASI di quella sezione, numerate, che enunciano una "
    "regola, un istituto, un passaggio processuale o una massima che il MATERIALE non contiene, o contraddice. Per "
    "ogni frase scegli una sola azione:\n"
    "- \"riscritta\": se il MATERIALE sostiene una versione della frase, riscrivila dicendo solo ciò che il "
    "MATERIALE dice (con le sue parole o parafrasandolo), nello stesso punto del ragionamento; se il MATERIALE dice "
    "il contrario, scrivi ciò che dice il MATERIALE.\n"
    "- \"rinvio\": se il MATERIALE non sostiene la frase, riscrivila come un punto da verificare (\"Va verificata "
    "la disciplina dell'art. ... quanto a ...\"), senza affermare la regola.\n"
    "- \"riempitivo\": solo se la frase non dice nulla di utile al parere e non cita alcuna norma o pronuncia.\n"
    "Regole: conserva OGNI riferimento normativo e giurisprudenziale della frase (articolo, codice o legge, "
    "pronuncia), con le stesse parole; non aggiungere norme, pronunce, termini, importi o fatti che il MATERIALE o "
    "la frase non contengono; non usare ciò che sai. Una frase, della stessa lunghezza o più breve. Rispondi SOLO "
    "con un array JSON: [{\"i\": numero, \"azione\": \"riscritta\"|\"rinvio\"|\"riempitivo\", \"frase\": \"la frase "
    "nuova, vuota per riempitivo\"}].")


# --- pure helpers ---------------------------------------------------------------------------------------------------

# A list bullet or a bold lead ("**Profilo penale**: ") stays in place; only the sentence after it is judged.
_LEAD_RE = re.compile(r"^\s*(?:(?:[-•*]|\d+[.)])\s+)?(?:\*\*[^*\n]{1,80}\*\*\s*[:.]?\s*)?")


def _words(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))


def candidates(text: str) -> List[Tuple[int, int, str]]:
    """(start, end, sentence) of the sentences worth judging: within each span of plan_then_write._sentence_spans,
    after its bullet or bold lead, at least MIN_WORDS words."""
    from . import plan_then_write as ptw
    out = []
    for s, e in ptw._sentence_spans(text):
        lead = _LEAD_RE.match(text[s:e])
        start = s + (lead.end() if lead else 0)
        sentence = text[start:e].strip()
        if _words(sentence) >= MIN_WORDS:
            out.append((start, start + len(text[start:e].rstrip()), sentence))
    return out


def batches(items: Sequence[Any], size: int = BATCH) -> List[List[Any]]:
    return [list(items[i:i + size]) for i in range(0, len(items), size)]


def trim_material(material: str, cited: Sequence[str], limit: int = MATERIAL_CHARS) -> str:
    """The material within `limit` characters: the blocks naming an article the sentences cite first (as
    fetch_audit's _trim_material), then the rest in order."""
    if len(material or "") <= limit:
        return material or ""
    blocks = re.split(r"\n(?=\[)", material)
    first = [b for b in blocks if any(c and c in b[:80] for c in cited)]
    rest = [b for b in blocks if b not in first]
    out, size = [], 0
    for b in first + rest:
        b = b[: max(0, limit - size)]
        if b:
            out.append(b)
            size += len(b)
        if size >= limit:
            break
    return "\n".join(out)


def _label(ref: tuple) -> str:
    from . import plan_then_write as ptw
    return f"art. {ref[0]} {ptw._code_key(ref[1])}"


def pointer(refs: Sequence[tuple]) -> str:
    """The pointer written in code: "Su questo punto va verificata la disciplina dell'art. 2051 c.c."."""
    labels = list(dict.fromkeys(_label(r) for r in refs))
    if not labels:
        return ""
    if len(labels) == 1:
        text = f"Su questo punto va verificata la disciplina dell'{labels[0]}"
    else:
        text = ("Su questo punto va verificata la disciplina degli artt. "
                + ", ".join(lab.replace("art. ", "", 1) for lab in labels[:-1]) + " e "
                + labels[-1].replace("art. ", "", 1))
    return text if text.endswith(".") else text + "."


def keeps_citations(old: str, new: str) -> bool:
    """Every article number and ruling number of `old` is still in `new`."""
    from . import plan_then_write as ptw
    return ptw._claim_refs(old) <= ptw._claim_refs(new)


def _closing(old: str, new: str) -> str:
    """new ending as old ended (a sentence's own punctuation, or none for a list item without it)."""
    new = new.strip()
    if old.rstrip()[-1:] in ".!?;:" and new[-1:] not in ".!?;:":
        new += old.rstrip()[-1]
    return new


def choose(old: str, reply: Optional[Dict[str, Any]], refs: Sequence[tuple]) -> Tuple[str, Optional[str]]:
    """(how, new text) for one not-supported sentence from the rewrite's reply: rewritten / pointer (the model's,
    or the code's when the model's loses a citation or is out of proportion) / dropped (filler citing nothing) /
    kept (nothing usable and nothing cited: never invent a pointer for it)."""
    from . import plan_then_write as ptw
    action = str((reply or {}).get("azione") or "").strip().lower()
    new = " ".join(str((reply or {}).get("frase") or "").split())
    cites = bool(ptw._claim_refs(old)) or bool(refs)
    if action == "riempitivo" and not cites:
        return "dropped", None
    usable = bool(new) and keeps_citations(old, new) and len(new) <= 2.5 * len(old) + 100
    if usable and action in ("riscritta", "rinvio"):
        return ("rewritten" if action == "riscritta" else "pointer"), _closing(old, new)
    if refs:
        return "pointer", _closing(old, pointer(refs))
    return "kept", None


def apply(text: str, changes: Dict[Tuple[int, int], Optional[str]]) -> str:
    """text with each (start, end) replaced by its new sentence, or dropped (None) with its line when nothing else
    is left on it."""
    from . import plan_then_write as ptw
    pieces, drops, pos, size = [], [], 0, 0
    for (s, e), new in sorted(changes.items()):
        pieces.append(text[pos:s])
        size += s - pos
        piece = text[s:e] if new is None else new
        if new is None:                 # its place in the new text, after the replacements before it
            drops.append((size, size + len(piece)))
        pieces.append(piece)
        size += len(piece)
        pos = e
    pieces.append(text[pos:])
    text = "".join(pieces)
    return ptw._drop_spans(text, drops) if drops else text


# --- model calls ----------------------------------------------------------------------------------------------------

def _parse_items(raw: Any, n: int) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for item in raw if isinstance(raw, list) else []:
        try:
            k = int(item.get("i")) - 1
        except (TypeError, ValueError, AttributeError):
            continue
        if 0 <= k < n and k not in out:
            out[k] = item
    return out


def _grounded_judge(material: str, sentences: List[str]) -> Dict[int, Dict[str, Any]]:
    """{index: {"tipo", "esito", "prova"}} for one batch; {} on a failure (nothing is then changed)."""
    from . import defensive_generation as dg
    human = f"MATERIALE:\n{material}\n\nFRASI:\n" + "\n".join(f"{n}. {s}" for n, s in enumerate(sentences, 1))
    try:
        raw = dg._chat_json([SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=human)],
                            max_tokens=120 + 70 * len(sentences), temperature=0)
    except Exception as exc:
        logger.warning("grounded review: judge failed, batch kept: %s", exc)
        return {}
    return _parse_items(raw, len(sentences))


def _grounded_rewrite(material: str, sentences: List[str]) -> Dict[int, Dict[str, Any]]:
    """{index: {"azione", "frase"}} for one section's not-supported sentences; {} on a failure."""
    from . import defensive_generation as dg
    human = f"MATERIALE:\n{material}\n\n{REWRITE_MARK}\n" + "\n".join(f"{n}. {s}" for n, s in enumerate(sentences, 1))
    try:
        raw = dg._chat_json([SystemMessage(content=REWRITE_SYSTEM), HumanMessage(content=human)],
                            max_tokens=200 + 140 * len(sentences), temperature=0)
    except Exception as exc:
        logger.warning("grounded review: rewrite failed, sentences handled in code: %s", exc)
        return {}
    return _parse_items(raw, len(sentences))


def verdict_of(item: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    """(kind, verdict) of a judge item, as fetch_audit reads it: no verdict for "altro" or an unknown one."""
    kind = str((item or {}).get("tipo") or "").strip().lower() or None
    verdict = (str((item or {}).get("esito") or "").strip().lower() or None) if kind and kind != "altro" else None
    return kind, (verdict if verdict in VERDICTS else None)


# --- the fetch ------------------------------------------------------------------------------------------------------

def fetch_blocks(refs: List[tuple], documents: List[Dict[str, Any]], session=None) -> Dict[tuple, str]:
    """{ref: its "[art. N code]" block} for the refs found in the corpus, in one read (the app's driver when no
    session is given); {} on a failure."""
    from . import defensive_generation as dg
    if not refs:
        return {}

    def read(s):
        return dg._fetch_any(s, documents, list(refs))
    try:
        if session is None:
            import neo4j
            from .main import NEO4J_DATABASE, driver
            with driver.session(database=NEO4J_DATABASE, default_access_mode=neo4j.READ_ACCESS) as own:
                rows = read(own)
        else:
            rows = read(session)
    except Exception as exc:
        logger.warning("grounded review: cited articles not read: %s", exc)
        return {}
    from . import plan_then_write as ptw
    out: Dict[tuple, str] = {}
    for ref in refs:
        mine = [r for r in rows if r.get("_article") and ptw._same_ref(r["_article"], ref)]
        if mine:
            block = dg._source_block(mine, FETCH_CHARS)
            if block.strip():
                out[ref] = block
    return out


# --- the review -----------------------------------------------------------------------------------------------------

def review(sections: List[Tuple[str, str]], jobs: List[Tuple[str, str, str, int]],
           documents: Optional[List[Dict[str, Any]]] = None, article_rows: Optional[List[Dict[str, Any]]] = None,
           fetch: Optional[Callable[[List[tuple]], Dict[tuple, str]]] = None) -> List[Tuple[str, str]]:
    """The sections with their not-supported rule-sentences rewritten from their material, as pointers, or (filler
    citing nothing) dropped; any failure keeps the section as it was. `fetch` (tests) replaces fetch_blocks."""
    from . import plan_then_write as ptw
    documents = documents or []
    known = {r["_article"] for r in article_rows or [] if r.get("_article")}
    material_of = {job[0]: job[2] for job in jobs}
    fetch = fetch or (lambda refs: fetch_blocks(refs, documents))

    # 1. judge every section's sentences, all batches in parallel
    found = {i: candidates(text) for i, (_, text) in enumerate(sections) if text}
    tasks = [(i, chunk) for i, cands in found.items() for chunk in batches(list(range(len(cands))))]

    def judge(task):
        i, chunk = task
        sentences = [found[i][k][2] for k in chunk]
        refs = [_label(r) for s in sentences for _, _, rs in ptw._citations(s, documents, known) for r in rs]
        material = trim_material(material_of.get(sections[i][0], ""), refs)
        got = _grounded_judge(material, sentences)
        return {chunk[n]: item for n, item in got.items()}
    verdicts: Dict[int, Dict[int, Dict[str, Any]]] = {i: {} for i in found}
    if tasks:
        with ThreadPoolExecutor(max_workers=min(8, len(tasks))) as pool:
            for (i, _), got in zip(tasks, pool.map(judge, tasks)):
                verdicts[i].update(got)

    stats, flagged = [], []          # flagged: (section index, candidate index, kind, verdict, refs)
    for i, cands in found.items():
        counts = {"section": sections[i][0], "sentences": len(cands), "judged": len(verdicts[i]), "rule": 0,
                  **{v: 0 for v in VERDICTS}}
        for k in range(len(cands)):
            kind, verdict = verdict_of(verdicts[i].get(k))
            if kind and kind != "altro":
                counts["rule"] += 1
            if verdict:
                counts[verdict] += 1
            if verdict in ("assente", "contraddetta"):
                refs = list(dict.fromkeys(r for _, _, rs in ptw._citations(cands[k][2], documents, known) for r in rs))
                flagged.append((i, k, kind, verdict, refs))
        stats.append(counts)
    entries: List[Dict[str, Any]] = []
    if not flagged:
        vlog("grounded_review", _log(stats, entries, {}, []))
        return sections

    # 2. the articles cited that the section's material does not hold: one read, then judged again with their text
    given = {i: ptw._material_refs(material_of.get(sections[i][0], ""), documents) for i in found}

    def not_given(i, refs):
        return [r for r in refs if not any(ptw._same_ref(r, g) for g in given[i])]
    wanted = list(dict.fromkeys(r for i, _, _, verdict, refs in flagged if verdict == "assente" for r in not_given(i, refs)))
    fetched = fetch(wanted) if wanted else {}
    fetched = {ref: block for ref, block in (fetched or {}).items() if block}
    missing = [r for r in wanted if r not in fetched]

    def extra_of(i, refs):
        return [ref for ref in not_given(i, refs) if ref in fetched]
    again = [(n, item) for n, item in enumerate(flagged) if item[3] == "assente" and extra_of(item[0], item[4])]

    def rejudge(entry):
        _, (i, k, _, _, refs) = entry
        blocks = "\n\n".join(fetched[r] for r in extra_of(i, refs))
        got = _grounded_judge(trim_material(f"{blocks}\n\n{material_of.get(sections[i][0], '')}",
                                            [_label(r) for r in refs]), [found[i][k][2]])
        return verdict_of(got.get(0))[1]
    supported_now = set()
    if again:
        with ThreadPoolExecutor(max_workers=min(8, len(again))) as pool:
            for (n, _), verdict in zip(again, pool.map(rejudge, again)):
                if verdict == "supportata":
                    supported_now.add(n)

    # 3. the rest rewritten, one call per section, its material with the articles read for it
    todo: Dict[int, List[int]] = {}
    for n, (i, k, kind, verdict, refs) in enumerate(flagged):
        if n in supported_now:
            entries.append({"section": sections[i][0], "kind": kind, "verdict": verdict, "how": "kept_fetched",
                            "sentence": found[i][k][2][:400], "fetched": [_label(r) for r in extra_of(i, refs)]})
        else:
            todo.setdefault(i, []).append(n)

    def rewrite(i):
        items = [flagged[n] for n in todo[i]]
        extra = list(dict.fromkeys(r for _, _, _, _, refs in items for r in extra_of(i, refs)))
        blocks = "\n\n".join(fetched[r] for r in extra)
        material = trim_material(f"{blocks}\n\n{material_of.get(sections[i][0], '')}".strip(),
                                 [_label(r) for _, _, _, _, refs in items for r in refs])
        return _grounded_rewrite(material, [found[i][k][2] for _, k, _, _, _ in items])
    order = sorted(todo)
    replies: Dict[int, Dict[int, Dict[str, Any]]] = {}
    if order:
        with ThreadPoolExecutor(max_workers=min(6, len(order))) as pool:
            replies = dict(zip(order, pool.map(rewrite, order)))
    out = list(sections)
    for i in order:
        changes: Dict[Tuple[int, int], Optional[str]] = {}
        for m, n in enumerate(todo[i]):
            _, k, kind, verdict, refs = flagged[n]
            s, e, sentence = found[i][k]
            how, new = choose(sentence, replies.get(i, {}).get(m), refs)
            entry = {"section": sections[i][0], "kind": kind, "verdict": verdict, "how": how,
                     "sentence": sentence[:400], "fetched": [_label(r) for r in extra_of(i, refs)]}
            if new is not None:
                entry["new"] = new[:400]
            entries.append(entry)
            if how == "dropped":
                changes[(s, e)] = None
            elif new is not None:
                changes[(s, e)] = new
        if changes:
            text = apply(sections[i][1], changes)
            out[i] = (sections[i][0], text or sections[i][1])
    vlog("grounded_review", _log(stats, entries, fetched, missing))
    return out


def _log(stats: List[Dict[str, Any]], entries: List[Dict[str, Any]], fetched: Dict[tuple, str],
         missing: List[tuple]) -> Dict[str, Any]:
    hows = ("rewritten", "pointer", "dropped", "kept", "kept_fetched")
    totals = {key: sum(s[key] for s in stats) for key in ("sentences", "judged", "rule") + VERDICTS}
    return {"sections": stats, "entries": entries, "totals": totals,
            "counts": {h: sum(1 for e in entries if e["how"] == h) for h in hows},
            "fetched": {_label(ref): block for ref, block in fetched.items()},
            "not_found": [_label(r) for r in missing]}


def totals(events: Sequence[Sequence[Any]]) -> Dict[str, int]:
    """The grounded_review counts of a run's events ([name, data, ...]), summed: sentences judged, rule-sentences,
    verdicts, and what was done."""
    out: Dict[str, int] = {}
    for event in events:
        if len(event) < 2 or event[0] != "grounded_review" or not isinstance(event[1], dict):
            continue
        for key, value in list((event[1].get("totals") or {}).items()) + list((event[1].get("counts") or {}).items()):
            out[key] = out.get(key, 0) + int(value or 0)
        out["fetched"] = out.get("fetched", 0) + len(event[1].get("fetched") or {})
        out["not_found"] = out.get("not_found", 0) + len(event[1].get("not_found") or [])
    return out
