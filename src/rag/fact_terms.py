"""v74: facts named with the case's words -- the term-consistency check extended to fact nouns.

The act and the parties are referred to with the wording of the fatti certi:

  - an act noun (data/fact_terms.json: classes of distinct acts or objects, each member one act with its synonyms) the
    writer uses where the fatti certi name another member of its class -- a «pugno» for the case's «schiaffo», a
    «ceffone» for it -- and the case never uses: replaced by the case's own noun, its article re-inflected («un pugno»
    -> «uno schiaffo», «del pugno» -> «dello schiaffo»); a form the replacement cannot inflect (a plural, a bare noun),
    or a class whose certain facts name two members (which one the writer meant is unknown): the sentence dropped;
  - a word prejudging the accused («l'aggressore», «del reo») when the client is the accused and the fatti certi name
    them: replaced by the client's name («di Sempronio»); a form it cannot rewrite (a plural): the sentence dropped.

A noun the case itself uses anywhere is never a paraphrase. apply(text, …) works sentence by sentence and logs each
change {how: replaced | dropped, from, to, sentence}; apply_sections runs it over the answer's sections.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data",
                         "fact_terms.json")
_DATA: Dict[str, Any] = {}
_SPLIT_RE = re.compile(r"(?<=[.;!?])\s+(?=[A-ZÀ-ÖØ-Ý«\"“(\[*\d-])")
_APOS = "['’]"
# the determiner before a noun: (preposition, definite) -- «un/uno/una/un'» indefinite, the definite articles alone or
# fused with di, a, da, in, su (and «col»)
_DETERMINERS = {
    "un": ("", False), "uno": ("", False), "una": ("", False), "un'": ("", False),
    "il": ("", True), "lo": ("", True), "la": ("", True), "l'": ("", True),
    "del": ("di", True), "dello": ("di", True), "della": ("di", True), "dell'": ("di", True),
    "al": ("a", True), "allo": ("a", True), "alla": ("a", True), "all'": ("a", True),
    "dal": ("da", True), "dallo": ("da", True), "dalla": ("da", True), "dall'": ("da", True),
    "nel": ("in", True), "nello": ("in", True), "nella": ("in", True), "nell'": ("in", True),
    "sul": ("su", True), "sullo": ("su", True), "sulla": ("su", True), "sull'": ("su", True),
    "col": ("con", True),
}
_FUSED = {("di", "il"): "del", ("di", "lo"): "dello", ("di", "la"): "della", ("di", "l'"): "dell'",
          ("a", "il"): "al", ("a", "lo"): "allo", ("a", "la"): "alla", ("a", "l'"): "all'",
          ("da", "il"): "dal", ("da", "lo"): "dallo", ("da", "la"): "dalla", ("da", "l'"): "dall'",
          ("in", "il"): "nel", ("in", "lo"): "nello", ("in", "la"): "nella", ("in", "l'"): "nell'",
          ("su", "il"): "sul", ("su", "lo"): "sullo", ("su", "la"): "sulla", ("su", "l'"): "sull'",
          ("con", "il"): "col", ("con", "lo"): "con lo", ("con", "la"): "con la", ("con", "l'"): "con l'"}


def _det_pattern(det: str) -> str:
    return rf"\b{re.escape(det[:-1])}{_APOS}" if det.endswith("'") else rf"\b{re.escape(det)}\s+"


_DET_RE = "(?:" + "|".join(_det_pattern(d) for d in sorted(_DETERMINERS, key=len, reverse=True)) + ")"


def data() -> Dict[str, Any]:
    if "d" not in _DATA:
        try:
            with open(DATA_FILE, encoding="utf-8") as f:
                _DATA["d"] = json.load(f)
        except (OSError, ValueError) as exc:
            logger.warning("fact_terms: %s not read: %s", DATA_FILE, exc)
            _DATA["d"] = {}
    return _DATA["d"]


def _word(form: str) -> str:
    return rf"(?<!\w){re.escape(form)}(?!\w)"


def _uses(text: str, form: str) -> bool:
    return bool(form) and re.search(_word(form), text or "", re.IGNORECASE) is not None


def _det_key(det: str) -> str:
    return re.sub(r"\s+", "", det.lower().replace("’", "'"))


def article(form: Dict[str, str], prep: str, definite: bool) -> str:
    """The determiner of the case's noun in the writer's construction: «uno schiaffo», «dello schiaffo», «l'auto»."""
    noun = form["form"]
    if definite:
        art = form.get("art") or "il"
        head = _FUSED.get((prep, art), f"{prep} {art}" if prep else art)
        return f"{head}{noun}" if head.endswith("'") else f"{head} {noun}"
    if form.get("g") == "f":
        return f"un'{noun}" if noun[:1].lower() in "aeiouàèéìòù" else f"una {noun}"
    return f"uno {noun}" if form.get("art") == "lo" else f"un {noun}"


def case_forms(certi: Sequence[str], case: str) -> List[Dict[str, Any]]:
    """Per class: the members the fatti certi name (their first form used), and the forms the case uses anywhere."""
    text = " \n".join(certi or [])
    out = []
    for cls in data().get("classes") or []:
        named = []
        for member in cls.get("members") or []:
            used = next((f for f in member if _uses(text, f["form"]) or _uses(text, f.get("plural") or "")), None)
            if used:
                named.append(used)
        anywhere = {f["form"] for m in cls.get("members") or [] for f in m
                    if _uses(case, f["form"]) or _uses(case, f.get("plural") or "")}
        if named:
            out.append({"id": cls["id"], "named": named, "anywhere": anywhere, "members": cls.get("members") or [],
                        "siblings": cls.get("siblings") or "drop"})
    return out


def _noun(form: str) -> str:
    """The noun with its determiner, when it has one (group «det»)."""
    return rf"(?:(?P<det>{_DET_RE})|(?<![\w'’])){re.escape(form)}(?!\w)"


def _drop(log: List[Dict[str, Any]], where: str, sentence: str, form: str, why: str, to: str = "") -> None:
    log.append({"section": where, "how": "dropped", "from": form, "to": to, "why": why, "sentence": sentence.strip()[:300]})


def _fix_sentence(sentence: str, classes: Sequence[Dict[str, Any]], client: str, accused: bool,
                  log: List[Dict[str, Any]], where: str) -> Optional[str]:
    """The sentence with the case's words, or None (dropped). Per class: the writer's forms the case never uses --
    a synonym of the case's noun always replaced; another member of the class replaced only in a class whose members
    are one act told differently (siblings «replace»), else the sentence dropped (it states an object the case does
    not); a sentence naming the case's noun beside it is a contrast, kept as written."""
    out = sentence
    for cls in classes:
        found = [(i, f, form, plural) for i, member in enumerate(cls["members"]) for f in member
                 if f["form"] not in cls["anywhere"]
                 for form, plural in ((f["form"], False), (f.get("plural") or "", True)) if form and _uses(out, form)]
        if not found:
            continue
        if any(_uses(out, x) for x in cls["anywhere"]):
            log.append({"section": where, "how": "kept", "from": found[0][2], "to": "", "why": "contrapposto al fatto "
                        "del caso", "sentence": sentence.strip()[:300]})
            continue
        if len(cls["named"]) != 1:
            _drop(log, where, sentence, found[0][2], "più fatti certi della stessa classe")
            return None
        own = cls["named"][0]
        own_member = next(i for i, m in enumerate(cls["members"]) if own in m)
        siblings = {i for i, *_ in found if i != own_member}
        if siblings and (cls.get("siblings") != "replace" or len(siblings) > 1):
            _drop(log, where, sentence, found[0][2], "un oggetto che il caso non nomina" if cls.get("siblings") !=
                  "replace" else "più atti diversi da quello del caso")
            return None
        for _, f, form, plural in found:
            if plural:
                _drop(log, where, sentence, form, "forma plurale", own["form"])
                return None

            def swap(m):
                det = m.group("det")
                if not det:
                    raise _Undeclinable()
                prep, definite = _DETERMINERS[_det_key(det)]
                new = article(own, prep, definite)
                return new[0].upper() + new[1:] if det[:1].isupper() else new
            try:
                new_out = re.sub(_noun(form), swap, out, flags=re.IGNORECASE)
                if new_out == out:          # «quell'automobile»: a determiner the replacement cannot inflect
                    raise _Undeclinable()
            except _Undeclinable:
                _drop(log, where, sentence, form, "nessun articolo da flettere", own["form"])
                return None
            log.append({"section": where, "how": "replaced", "from": form, "to": own["form"],
                        "sentence": sentence.strip()[:300], "now": new_out.strip()[:300]})
            out = new_out
    if accused and client:
        for word in data().get("accusatory") or []:
            if word.endswith("e") and _uses(out, word[:-1] + "i"):
                _drop(log, where, sentence, word, "forma plurale", client)
                return None
            pattern = re.compile(rf"(?P<det>{_DET_RE}){re.escape(word)}(?!\w)", re.IGNORECASE)
            if not pattern.search(out):
                continue

            def name(m):
                prep, _ = _DETERMINERS[_det_key(m.group("det"))]
                new = f"{prep} {client}" if prep else client
                return new[0].upper() + new[1:] if m.group("det")[:1].isupper() else new
            new_out = pattern.sub(name, out)
            log.append({"section": where, "how": "replaced", "from": word, "to": client,
                        "sentence": sentence.strip()[:300], "now": new_out.strip()[:300]})
            out = new_out
    return out


class _Undeclinable(Exception):
    pass


def apply(text: str, certi: Sequence[str], case: str, client: str = "", accused: bool = False,
          log: Optional[List[Dict[str, Any]]] = None, where: str = "") -> str:
    """The text with the case's words for its facts (the module's doc), line by line, sentence by sentence."""
    log = log if log is not None else []
    classes = case_forms(certi, case)
    if not classes and not (accused and client):
        return text
    lines = []
    for line in (text or "").split("\n"):
        kept = []
        for sentence in _SPLIT_RE.split(line):
            fixed = _fix_sentence(sentence, classes, client, accused, log, where)
            if fixed is not None:
                kept.append(fixed)
        if kept or not line.strip():
            lines.append(" ".join(kept))
    return "\n".join(lines)


def apply_sections(sections: List[Tuple[str, str]], certi: Sequence[str], case: str, client: str = "",
                   accused: bool = False, log: Optional[List[Dict[str, Any]]] = None) -> List[Tuple[str, str]]:
    return [(t, apply(x, certi, case, client, accused, log, t)) for t, x in sections]
