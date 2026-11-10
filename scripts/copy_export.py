#!/usr/bin/env python3
"""v60h (10): every user-facing Italian string Code has drafted since the interactive flow began, in one table for the
PO (key, text, where it appears) -- docs/copy_review.md. Re-runnable; reads the source only.

  1. src/rag/interaction_copy.py: each constant (and each entry of a dict of texts) whose comment says Code drafted
     it -- «my draft» / «my drafts», or a version tag after the PO's v54 texts (v55 … v60h) with no «PO» in it; the
     PO's own texts (v54, verbatim; «the PO's text», «the PO's words») are left out. Where: the functions that read it
     (C.KEY in src/rag), and the copy's own section (round 1, the questions, round 2, the answer, closing).
  2. The user-facing strings Code drafted outside the copy file (EXTRA: §4's procedural lines, the closed-list
     mediazione and negoziazione texts, the check's and the read-back's marks): every string literal of 12 characters or
     more, with a space, in the named constants and functions.

  python scripts/copy_export.py                    # docs/copy_review.md
  python scripts/copy_export.py --out /tmp/copy.md
"""

import argparse
import ast
import io
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COPY = ROOT / "src" / "rag" / "interaction_copy.py"
SRC = ROOT / "src" / "rag"
_DRAFT_RE = re.compile(r"\bmy drafts?\b", re.IGNORECASE)
_PO_RE = re.compile(r"\bPO\b(?!\s+review)")
_VERSION_RE = re.compile(r"\bv(5[5-9]|6\d)[a-z]?\b")
# (module, name, where): constants or functions of other modules whose strings reach the user
EXTRA = (
    ("render_computed", "listed_mediation", "Risposta, «Profili processuali»: la mediazione obbligatoria (lista chiusa)"),
    ("render_computed", "negotiation_of", "Risposta, «Profili processuali»: la negoziazione assistita"),
    ("render_computed", "NEGOTIATION_TITLE", "Risposta, «Profili processuali»: titolo del paragrafo"),
    ("render_computed", "PROCEDURE_TITLE", "Risposta, «Profili processuali»: titolo del paragrafo dei reati"),
    ("render_computed", "_fact_lines", "Risposta, «Profili processuali»: «Procedibilità e competenza»"),
    ("procedural_facts", "QUERELA_DEFAULT", "Risposta, «Procedibilità e competenza»: termine di querela"),
    ("procedural_facts", "REMISSIONE_DEFAULT", "Risposta, «Procedibilità e competenza»: remissione"),
    ("procedural_facts", "UFFICIO_DEFAULT", "Risposta, «Procedibilità e competenza»: procedibilità"),
    ("procedural_facts", "competenza", "Risposta, «Procedibilità e competenza»: competenza"),
    ("procedural_facts", "extract", "Risposta, «Procedibilità e competenza»: procedibilità e remissione"),
    ("record_check", "check", "Risposta, ogni sezione: segno su una frase da verificare", r"^\s*\(da verificare.*\)$"),
    ("readback", "UNVERIFIED", "Risposta: segno su un fatto della parte non trovato nel caso"),
)


def _comments(source: str):
    """({line: full-line comment}, {line: inline comment}) of the source."""
    full, inline = {}, {}
    lines = source.splitlines()
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT:
            n = tok.start[0]
            (full if lines[n - 1].lstrip().startswith("#") else inline)[n] = tok.string.lstrip("#").strip()
    return full, inline


def _above(comments, line):
    """The contiguous block of full-line comments right above a line, top first."""
    block, n = [], line - 1
    while n in comments:
        block.insert(0, comments[n])
        n -= 1
    return " ".join(block)


def _group(comments, starts, line, key):
    """The full-line comment block that heads the run of assignments a line is in (code lines between, no blank line):
    it holds for this key when the run's first key shares this key's first word (ANGLE_STATED_CHOSEN, ANGLE_STATED)."""
    n = line - 1
    first = key
    while n >= 1 and n not in comments and n in starts:
        if starts[n]:
            first = starts[n]
        n -= 1
    if n not in comments:
        return ""
    return _above(comments, n + 1) if first.split("_")[0] == key.split("_")[0] else ""


def _section(comments, line):
    """The copy's section a line falls in («--- round 1 ---» and the like)."""
    best = ""
    for n in sorted(comments):
        if n > line:
            break
        m = re.match(r"-{3}\s*(.+?)\s*-{3,}", comments[n])
        if m:
            best = m.group(1)
    return best


def _strings(node):
    """The string texts a value holds (a string, a tuple / dict of them), each once."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [s for e in node.elts for s in _strings(e)]
    if isinstance(node, ast.JoinedStr):
        return ["".join(v.value if isinstance(v, ast.Constant) else "{…}" for v in node.values)]
    return []


def _drafted(note: str) -> bool:
    if _DRAFT_RE.search(note):
        return True
    return bool(_VERSION_RE.search(note)) and not _PO_RE.search(note)


_USES = {}


def _uses():
    """{key: [module.function, …]} of every C.<KEY> in src/rag, each file read and parsed once."""
    if _USES:
        return _USES
    for path in sorted(SRC.glob("*.py")):
        if path.name == "interaction_copy.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "C." not in text:
            continue
        spans = sorted(((fn.lineno, fn.end_lineno, fn.name) for fn in ast.parse(text).body
                        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))), key=lambda x: x[0])
        for n, line in enumerate(text.splitlines(), 1):
            for key in re.findall(r"\bC\.([A-Z][A-Z0-9_]+)\b", line):
                fn = next((name for a, b, name in spans if a <= n <= b), "<module>")
                where = f"{path.stem}.{fn}"
                if where not in _USES.setdefault(key, []):
                    _USES[key].append(where)
    return _USES


def _users(key: str):
    """module.function of every use of C.<key> in src/rag."""
    return list(_uses().get(key, []))


_PLACES = {"round 1": "Primo turno", "the questions: one template per family (a family with no template is not asked)":
           "Primo turno, «Da chiarire»", "round 2": "Secondo turno / dopo una risposta", "the answer": "Risposta (parere)",
           "closing and follow-ups": "Chiusura e seguiti"}


def copy_rows():
    source = COPY.read_text(encoding="utf-8")
    comments, inline = _comments(source)
    tree = ast.parse(source)
    starts = {}                 # {line: the key assigned there, "" on a continuation line} of the module's code lines
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            for n in range(node.lineno, node.end_lineno + 1):
                starts[n] = node.targets[0].id if n == node.lineno else ""
    rows = []
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            continue
        key = node.targets[0].id
        if not key.isupper():
            continue
        note = " ".join(x for x in [_above(comments, node.lineno) or _group(comments, starts, node.lineno, key)]
                        + [inline.get(n, "") for n in range(node.lineno, node.end_lineno + 1)] if x)
        place = _PLACES.get(_section(comments, node.lineno), _section(comments, node.lineno))
        where = "; ".join([place] + _users(key))
        if isinstance(node.value, ast.Dict):
            for k, v in zip(node.value.keys, node.value.values):
                label = ast.literal_eval(k) if k is not None else "?"
                own = " ".join(x for x in (_above(comments, k.lineno), inline.get(v.end_lineno, "")) if x)
                if _PO_RE.search(own) or not (_drafted(own) or _drafted(note)):
                    continue
                texts = _strings(v)
                for i, text in enumerate(texts):
                    rows.append((f"{key}[{label!s}]" + (f"#{i + 1}" if len(texts) > 1 else ""), text, where))
            continue
        if _drafted(note):
            texts = _strings(node.value)
            for i, text in enumerate(texts):
                rows.append((key + (f"#{i + 1}" if len(texts) > 1 else ""), text, where))
    return rows


def extra_rows():
    rows = []
    for module, name, where, *keep in EXTRA:
        path = SRC / f"{module}.py"
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in tree.body:
            target = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                target = node.targets[0].id
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                target = node.name
            if target != name:
                continue
            found = []
            root = node.value if isinstance(node, ast.Assign) else node
            inner = {id(v) for sub in ast.walk(root) if isinstance(sub, ast.JoinedStr) for v in sub.values}
            body = getattr(root, "body", None) or []
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
                inner.add(id(body[0].value))            # the docstring
            for sub in ast.walk(root):
                if not isinstance(sub, (ast.Constant, ast.JoinedStr)) or id(sub) in inner:
                    continue
                for text in _strings(sub):
                    if text in found:
                        continue
                    if keep and not re.search(keep[0], text):
                        continue
                    if isinstance(node, ast.Assign) or (len(text) >= 12 and " " in text
                                                        and re.search(r"[a-zà-ù]{2,}[ :,.']+[a-zà-ù]", text)):
                        found.append(text)
            for i, s in enumerate(found):
                rows.append((f"{module}.{name}" + (f"#{i + 1}" if len(found) > 1 else ""), s, where))
    return rows


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "↵ ")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=str(ROOT / "docs" / "copy_review.md"))
    args = parser.parse_args(argv)
    rows = copy_rows() + extra_rows()
    lines = ["# Copy review: the strings Code drafted for the interactive flow", "",
             "Generated by `scripts/copy_export.py` (re-run after any copy change). One row per user-facing Italian "
             "string Code drafted since the interactive flow began (v55 on): the copy file's own entries "
             "(`src/rag/interaction_copy.py`, edit there) and the few written elsewhere (key `module.function`). "
             "`{…}` marks a slot the code fills. Which strings are Code's is read from the copy file's comments: an "
             "entry tagged «my draft», or with a version after the PO's v54 texts (v55 on) and no «PO», is listed; an "
             "untagged entry is taken as the PO's v54 text and left out.", "",
             f"{len(rows)} strings.", "",
             "| key | text | where it appears |", "|---|---|---|"]
    lines += [f"| `{k}` | {_cell(t)} | {_cell(w)} |" for k, t, w in rows]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(rows)} strings; {out}")


if __name__ == "__main__":
    sys.exit(main())
