"""v79.19.0 (item 1): no copyable example in any prompt -- a literal article («640 c.p.», «1218 c.c.», «700 c.p.c.»,
«585 c.p.p.», «116 c.p.a.»), an element or period id sample («A640cp:E1», «A640cp.1») or a few-shot party name in a
prompt text is what the model copies into an answer as if it were a fact of the case. The scan:
  - data/*.json: every string under a key whose path names a prompt (prompt_system, prompt, reask*, instruction*,
    system, rules, labels*, example*) -- data/lawyer_method.json's own sections (ruolo, fasi, sections, stile,
    honesty_rules, controllo_finale, modules) are the lawyer's method text, verbatim from docs/lawyer/metodo.md
    (tests/test_v70 keeps them in sync): his words, left to him -- listed here as the one exception;
  - src/**/*.py: every string constant (f-string parts included, docstrings excluded) assigned to a module-level name
    that says prompt (…SYSTEM, …RULE(S), …PROMPT, …EXAMPLES, …INSTRUCTION(S), …TEMPLATE, …_HEAD, …_LAW) or passed to
    SystemMessage(content=…) / HumanMessage(content=…), or inside a function whose name says prompt (prompt, system,
    instruction, messages, rules).
A placeholder («<numero> <codice>», «{art}») is no example: angle brackets and format slots are stripped first.
"""

import ast
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ARTICLE = re.compile(r"\b\d{1,4}(?:-\w+)?\s+c\.(?:c|p|p\.c|p\.p|p\.a)\.", re.IGNORECASE)
ART_N = re.compile(r"\bartt?\.\s*\d{1,4}\b", re.IGNORECASE)
ID_SAMPLE = re.compile(r"\bA\d{1,4}[a-z]+(?::E\d+|\.\d+)\b")
PLACEHOLDER = re.compile(r"<[^<>\n]{1,80}>|\{[a-zA-Z_][a-zA-Z_0-9.\[\]'\"]*\}")
PROMPT_KEY = re.compile(r"prompt|reask|instruction|^system$|rules|example", re.IGNORECASE)
PROMPT_NAME = re.compile(r"(SYSTEM|RULES?|PROMPT|EXAMPLES|INSTRUCTIONS?|TEMPLATE|_HEAD|_LAW)$")
PROMPT_FUNC = re.compile(r"prompt|system|instruction|messages|rules", re.IGNORECASE)
LAWYER_SECTIONS = ("ruolo", "fasi", "sections", "stile", "honesty_rules", "controllo_finale", "modules")
# names that say prompt but hold no prompt text (regular expressions, routing tables)
NOT_PROMPTS = {"_CODE_END_RE", "_LEAD_CODE_RE"}
# the document templates' drafting specs (src/rag/document_generation.py): the governing norm of the act to draft
# (an opposizione a decreto ingiuntivo, a locazione, a licenziamento) is the act's content, not an example of one
TEMPLATE_SPECS = re.compile(r"^src/rag/document_generation\.py:\d+:_(opposition|rental_\w+|employment_\w+)_system\(\)$")


def offending(text: str):
    clean = PLACEHOLDER.sub(" ", text)
    return ARTICLE.findall(clean) + ART_N.findall(clean) + ID_SAMPLE.findall(clean)


def data_prompt_strings():
    out = []
    for path in sorted((ROOT / "data").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))

        def walk(node, keys):
            if isinstance(node, dict):
                for k, v in node.items():
                    if str(k).startswith("_"):
                        continue
                    if path.name == "lawyer_method.json" and not keys and k in LAWYER_SECTIONS:
                        continue
                    walk(v, keys + [str(k)])
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    walk(v, keys + [str(i)])
            elif isinstance(node, str) and any(PROMPT_KEY.search(k) for k in keys):
                out.append((f"{path.name}:{'.'.join(keys)}", node))
        walk(data, [])
    return out


def _strings(node):
    """Every string constant inside an expression (concatenations, f-strings, tuples, calls)."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            yield sub.value


def src_prompt_strings():
    out = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc:
                    docstrings.add(doc)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                    and PROMPT_NAME.search(node.targets[0].id) and node.targets[0].id not in NOT_PROMPTS \
                    and not node.targets[0].id.endswith("_RE"):
                for s in _strings(node.value):
                    out.append((f"{path.relative_to(ROOT)}:{node.lineno}:{node.targets[0].id}", s))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id in ("SystemMessage", "HumanMessage"):
                for kw in node.keywords:
                    if kw.arg == "content":
                        for s in _strings(kw.value):
                            out.append((f"{path.relative_to(ROOT)}:{node.lineno}:{node.func.id}", s))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and PROMPT_FUNC.search(node.name):
                for s in _strings(node):
                    if s not in docstrings and len(s) > 40:
                        out.append((f"{path.relative_to(ROOT)}:{node.lineno}:{node.name}()", s))
    return out


@pytest.mark.parametrize("where,text", data_prompt_strings(), ids=lambda x: x if isinstance(x, str) and ":" in x
                         else "")
def test_no_copyable_example_in_the_data_prompts(where, text):
    assert not offending(text), (where, offending(text), text[:200])


def test_no_copyable_example_in_the_source_prompts():
    hits = [(where, offending(s), s[:160]) for where, s in src_prompt_strings()
            if offending(s) and not TEMPLATE_SPECS.match(where)]
    assert not hits, "\n".join(f"{w}: {o} -- {t}" for w, o, t in hits)


def test_the_scan_sees_the_prompts_it_guards():
    data = dict(data_prompt_strings())
    assert any(k.startswith("case_frame.json:prompt_system") for k in data)
    assert any(k.startswith("analysis_step.json:prompt_system") for k in data)
    assert any(k.startswith("applicability.json:prompt_system") for k in data)
    src = dict(src_prompt_strings())
    assert any("plan_then_write.py" in k and "_WRITER_RULES" in k for k in src)
    assert any("prompts.py" in k and "synthesis_system_message()" in k for k in src)
    assert any("defensive_generation.py" in k and "_GRADED_EXAMPLES" in k for k in src)
    assert offending("vedi l'art. 640 c.p. e l'id A640cp:E1") == ["640 c.p.", "art. 640", "A640cp:E1"]
    assert offending("art. <numero> <codice>, {art} c.c.") == []
