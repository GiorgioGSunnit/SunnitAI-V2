"""--render-from (v59): a stored --interactive-sim transcript re-rendered through the current code, no server, no model.

What the transcript stores is the output of each round; what the code made of it is re-made by the current functions on
what the transcript shows (the writer's paragraphs are kept as written -- no model is called):

  round 1   the listed topics (titles, articles, «Da chiarire» refs): a table driving a topic merged in and the topic named
            (topic_step._merge, named_by_table); a term's own topic no other topic carries out (topic_step.term_topics_out);
            the facts' term lines the listed topics carry (interactive_flow.carried_terms); the client question when no
            party is the client (interactive_flow.client_question); the questions' wording (question_wording).
  round 2   the chosen topics' names as round 1 now gives them.
  answer    §2: each requisite line through the current filter (rule_quote.requisite_kept, extends, REQUISITES_MIN); a
            table article now matched, bare in the transcript, rebuilt by norm_blocks from the local article text
            (answer_lint.COMMI_SOURCES) when there is one. §4: «Procedibilità e competenza» by render_computed
            .procedure_lines from the lines shown and the chosen topics' delitti; «Termini» the listed topics carry;
            «Mediazione» by render_computed.mediation_of. Then interactive_flow.answer_final, and the closing offer from
            the topics now listed.
Each case also gets "replay_log": what was changed and why.
"""

import glob
import os
import re
from pathlib import Path
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import answer_lint  # noqa: E402


# --- the local article texts (for a block the transcript never had) ---------------------------------------------------

_TEXTS: Optional[Dict[Tuple[str, str], Tuple[str, str]]] = None


def article_texts() -> Dict[Tuple[str, str], Tuple[str, str]]:
    """{(number, code): (the code as written, text)} of the local material (the longest copy of each)."""
    global _TEXTS
    if _TEXTS is not None:
        return _TEXTS
    out: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for pattern in answer_lint.COMMI_SOURCES:
        for path in glob.glob(pattern):
            text = open(path, encoding="utf-8", errors="replace").read()
            for m in re.finditer(r"(?m)^\[art\.\s+(\S+)\s+([^\]]+)\]\n(?:>>[^\n]*\n)*([^\n]*)", text):
                key = (m.group(1), answer_lint._code(m.group(2)))
                if "[…]" not in m.group(3) and len(m.group(3)) > len(out.get(key, ("", ""))[1]):
                    out[key] = (m.group(2), m.group(3))
    _TEXTS = out
    return out


_RUBRICHE: Optional[Dict[Tuple[str, str], str]] = None


def _rubrica(number: str, code: str) -> str:
    """The rubrica the local runs' records give the article (phase1_offline.rubrica_table)."""
    global _RUBRICHE
    if _RUBRICHE is None:
        import phase1_offline
        table = phase1_offline.rubrica_table(os.path.expanduser("~/SunnitAI-V2/diagnose_reports"))
        _RUBRICHE = {(str(k[0]), answer_lint._code(str(k[1]))): v for k, v in table.items()
                     if isinstance(k, tuple) and len(k) == 2}
    return _RUBRICHE.get((number, answer_lint._code(code)), "")


def _row(number: str, code: str, rubrica: str) -> Optional[Dict[str, Any]]:
    found = article_texts().get((number, answer_lint._code(code)))
    if not found:
        return None
    return {"_article": (number, found[0]), "s": {"plain_text": found[1],
                                                   "titolo_articolo": rubrica or _rubrica(number, code)}}


# --- round 1 -----------------------------------------------------------------------------------------------------------

def _terms_of(facts_text: str) -> List[Dict[str, Any]]:
    """The facts block's term sentences as the code's terms: [{"id", "ref", "label", "text"}]."""
    from src.rag import fact_signals
    by_label = {re.sub(r"\s*\(.*?\)", "", t.label).lower(): t for t in fact_signals.TERMS}
    out = []
    for sentence, refs, name in answer_lint._terms_in(facts_text):
        term = by_label.get(name)
        if term is not None:
            out.append({"id": term.id, "ref": term.ref, "label": term.label, "text": sentence})
        else:
            imp = next((x for x in fact_signals.NO_PRESCRIPTION if x[1] == name), None)
            out.append({"id": imp[0] if imp else name, "ref": imp[2] if imp else "", "label": name, "text": sentence})
    return out


def _topic(t: Dict[str, Any]) -> Dict[str, Any]:
    labels = [f"{n} {c}" for n, c in t["refs"]]
    return {"title": t["title"], "articles": labels, "refs": [tuple(r) for r in t["refs"]], "planned": len(labels),
            "merged": [], "facts": [], "questions": [], "source": "", "n": t["n"], "rest": t["rest"], "line": t["line"],
            "anchor": labels[0] if labels else ""}


def _driving_tables(topic: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The tables (topic_tables.TABLES) the round shows merged into the topic: the topic bears the table's name (or the
    sub-name an "anchors" pattern gives), or asks a question only that table asks (answer_lint's evidence)."""
    from collections import Counter
    from src.rag import interaction_copy as C, topic_tables
    families = Counter(q for table in topic_tables.TABLES for q in table.get("questions") or [])
    mine = {(n, answer_lint._code(c)) for n, c in topic["refs"]}
    out = []
    for table in topic_tables.TABLES:
        own = [(n, answer_lint._code(c)) for n, c in table["refs"]]
        if not set(own) & mine:
            continue
        shorts = [C.QUESTION_SHORT.get(q) for q in table.get("questions") or [] if families[q] == 1]
        names = [table["name"].lower()] + [named.lower() for _, _, named in table.get("anchors") or []]
        if topic["title"].lower() in names or any(s and s in topic["rest"] for s in shorts):
            out.append(table)
    return out


def replay_round1(case: Dict[str, Any], log: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """(the topics now listed, the round-1 pieces rebuilt) -- see the module docstring."""
    from src.rag import interaction_copy as C, interactive_flow as F, topic_step
    listed = [_topic(t) for t in answer_lint.topics(case["round1"])]
    # a table driving a topic: merged in (its articles the topic's own, after the anchor) and naming it
    for topic in listed:
        for table in _driving_tables(topic):
            title = topic_step._title(table["name"])
            own = [(n, c) for n, c in [(r[0], next((x[1] for x in topic["refs"] if x[0] == r[0]), r[1]))
                                       for r in table["refs"]]]
            topic_step._merge(topic, {"title": title, "own": own, "source": f"tabella {table['id']}", "facts": [],
                                      "merged": [], "questions": []})
            topic["articles"] = [f"{n} {c}" for n, c in topic["refs"]]
            before = topic["title"]
            topic_step.named_by_table(topic)
            if topic["title"] != before:
                log.append(f"topic «{before}» named «{topic['title']}» (named_by_table)")
        topic["matched"] = list(dict.fromkeys([topic["anchor"]] + [f"{n} {c}" for n, c in topic.get("own") or []]))
    terms = _terms_of(answer_lint.facts_block(case["round1"]))
    kept = topic_step.term_topics_out(listed, terms)
    for t in listed:
        if t not in kept:
            log.append(f"topic «{t['title']}» out (term_topics_out)")
    return kept, {"terms": terms, "all": listed}


def _questions(round1: str) -> List[str]:
    return [re.sub(r"^\d+\.\s+", "", q) for q in answer_lint.question_lines(round1)]


def _parties(round1: str) -> Tuple[Optional[str], List[Dict[str, str]]]:
    line = next((l for l in round1.splitlines() if l.startswith("Parti:")), None)
    if line is None:
        return None, []
    out = []
    for item in line[len("Parti:"):].strip().rstrip(".").split("; "):
        m = re.match(r"^(.*?)\s*(?:\((.*)\))?\s*(—\s*cliente)?$", item.strip())
        if m:
            role = (m.group(2) or "") + (", cliente" if m.group(3) and "cliente" not in (m.group(2) or "") else "")
            out.append({"name": m.group(1).strip(), "role": role})
    return line, out


def rebuild_round1(case: Dict[str, Any], topics: List[Dict[str, Any]], pieces: Dict[str, Any], log: List[str]) -> str:
    from src.rag import interaction_copy as C, interactive_flow as F
    text = case["round1"]
    old_qs = _questions(text)
    # the questions: those a kept topic refers to, or none refers to; the client's first when no party is the client
    refs_by_topic = {t["n"]: [int(x) for x in re.findall(r"\(domanda (\d+)\)", t["rest"])] for t in pieces["all"]}
    asked_by_any = {n for v in refs_by_topic.values() for n in v}
    kept_numbers = {n for t in topics for n in refs_by_topic.get(t["n"], [])}
    questions = [(i, q) for i, q in enumerate(old_qs, 1) if i in kept_numbers or i not in asked_by_any]
    for i, q in enumerate(old_qs, 1):
        if (i, q) not in questions:
            log.append(f"question {i} out (asked only by a topic now out)")
    _, parties = _parties(text)
    if parties and not any("Chi assisti" in q for _, q in questions):
        client = F.client_question({"parties": parties})
        if client:
            questions = [(0, client[0]["text"])] + questions
            log.append("client question added (client_question)")
    renumber = {old: new for new, (old, _) in enumerate(questions, 1) if old}
    worded = []
    for _, q in questions:
        family = "data_fatto" if q == C.QUESTIONS.get("data_fatto") else ""
        now = F.question_wording({"family": family, "text": q}, topics)
        if now != q:
            log.append("question wording: no criminal topic (question_wording)")
        worded.append(now)
    # the topic lines, renumbered, their «domanda N» refs too
    lines = []
    for new_n, t in enumerate(topics, 1):
        rest = re.sub(r"\(domanda (\d+)\)", lambda m: f"(domanda {renumber.get(int(m.group(1)), m.group(1))})", t["rest"])
        articles = F._articles_line({"articles": t["articles"], "planned": t["planned"]})
        lines.append(f"{new_n}. **{t['title']}** (artt. {articles}){rest}")
    body = text
    topic_block = re.search(r"(?ms)^1\.\s+\*\*.*?(?=\n\n)", body)
    if topic_block:
        body = body[:topic_block.start()] + "\n".join(lines) + body[topic_block.end():]
    # the facts' term lines the listed topics carry
    for term in pieces["terms"]:
        if term not in F.carried_terms(pieces["terms"], topics):
            body = body.replace(" " + term["text"], "").replace(term["text"] + " ", "").replace(term["text"], "")
            log.append(f"term line out: «{term['text'][:60]}» (carried_terms)")
    body = re.sub(r"\n[ \t]*\n(?=\n)", "\n", body)
    # the questions
    block = re.search(r"(?ms)(\*\*Da chiarire\*\*\s*\n\n)(.*?)(?=\n\n" + answer_lint.INSTRUCTION_START + ")", body)
    q_text = "\n".join(C.fill(C.QUESTION_LINE, n=i, text=q) for i, q in enumerate(worded, 1))
    if block:
        body = body[:block.start()] + (block.group(1) + q_text if worded else "").rstrip() + body[block.end():] \
            if worded else body[:block.start()].rstrip("\n") + body[block.end():]
    elif worded:
        closing = re.search(r"\n\n(?=" + answer_lint.INSTRUCTION_START + ")", body)
        if closing:
            body = body[:closing.start()] + f"\n\n{C.QUESTIONS_TITLE}\n\n{q_text}" + body[closing.start():]
    return re.sub(r"\n{3,}", "\n\n", body)


# --- the answer --------------------------------------------------------------------------------------------------------

_REQ_LINE_RE = re.compile(r"^-\s+(?:((?:co\.\s*\d+|testo)),\s+requisito|Requisito)\s+«(.*)»\s*:\s*(.*?)\.?\s*$")


def _blocks_refiltered(norms: str, log: List[str]) -> str:
    """Each article block's requisite lines through the current filter: a comma extending other articles' penalties its
    own line only; a requisite the filter drops gone; fewer than REQUISITES_MIN left, none."""
    from src.rag import interaction_copy as C, rule_quote
    lines = norms.split("\n")
    out: List[str] = []
    i = 0
    while i < len(lines):
        m = _REQ_LINE_RE.match(lines[i].strip())
        if not m:
            out.append(lines[i])
            i += 1
            continue
        group = []
        while i < len(lines) and _REQ_LINE_RE.match(lines[i].strip()):
            group.append(_REQ_LINE_RE.match(lines[i].strip()))
            i += 1
        by_label: "OrderedDict[str, list]" = OrderedDict()
        for g in group:
            by_label.setdefault(g.group(1) or C.COMMA_LABEL_ONLY, []).append(g)
        for label, items in by_label.items():
            texts = [g.group(2) for g in items]
            if any(rule_quote.extends(t) for t in texts):
                out.append(C.fill(C.COMMA_LINE, comma=label, status=items[0].group(3)))
                log.append(f"{label}: extends other penalties, its line only ({len(items)} requisites out)")
                continue
            kept = [g for g in items if rule_quote.requisite_kept(g.group(2))]
            if len(kept) < rule_quote.REQUISITES_MIN:
                kept = []
            if len(kept) < len(items):
                log.append(f"{label}: {len(items) - len(kept)} requisites out (requisite_kept)")
            out += [C.fill(C.REQUISITE_LINE, comma=label, requisite=g.group(2), status=g.group(3)) for g in kept]
    return "\n".join(out)


def _rebuilt_blocks(norms: str, topics: List[Dict[str, Any]], log: List[str]) -> str:
    """A chosen topic's matched article (a table's own) the transcript left bare or untreated: its block by norm_blocks
    from the local text, when there is one."""
    from src.rag import interactive_flow as F, rule_quote
    matched = [F._parse_label(a) for t in topics for a in t.get("matched") or []]
    for number, code in matched:
        key = (number, answer_lint._code(code))
        bare = re.search(rf"(?m)^\*\*Art\. {re.escape(number)} [^*\n]*\*\*\s*\n(?=\s*(?:\*\*Art\.|Altre norme|\Z))", norms)
        untreated = re.search(rf"Altre norme del tema non trattate:[^\n]*\bart\. {re.escape(number)} ", norms + " ")
        if not (bare or untreated):
            continue
        heading = re.search(rf"(?m)^\*\*Art\. {re.escape(number)} [^—*\n]*(?:— ([^*\n]*))?\*\*", norms)
        row = _row(number, code, heading.group(1) if heading and heading.group(1) else "")
        if row is None:
            log.append(f"art. {number} {code}: matched, no local text to rebuild its block")
            continue
        rules = rule_quote.article_rules({}, [row], {}, [row["_article"]])
        block = F.norm_blocks(rules, {}, [row], [row["_article"]], matched={F._key(tuple(row["_article"]))})
        if not block or "\n" not in block[0]:
            log.append(f"art. {number} {code}: matched, its block is its heading alone")
            continue
        if bare:
            norms = norms[:bare.start()] + block[0] + "\n\n" + norms[bare.end():]
        else:
            closing = re.search(r"(?m)^Altre norme del tema non trattate:", norms)
            norms = (norms[:closing.start()] + block[0] + "\n\n" + norms[closing.start():] if closing
                     else norms.rstrip() + "\n\n" + block[0])
            norms = re.sub(rf"(Altre norme del tema non trattate:[^\n]*?)(?:,\s*)?\bart\. {re.escape(number)} [^,\n]*?(?=,|\.?$)",
                           r"\1", norms, flags=re.M)
            norms = re.sub(r"(?m)^Altre norme del tema non trattate:\s*\.?\s*$\n?", "", norms)
        log.append(f"art. {number} {code}: matched (the table's), block rebuilt by norm_blocks")
    return norms


def _rules_of(norms: str) -> List[Dict[str, Any]]:
    """The §2 articles whose commi the local material numbers, as article_rules' shape: {"ref", "commi"} -- the texts
    where the material has them, the numbers it lists (commi non riportati) with no text."""
    from src.rag import rule_quote
    index = answer_lint.commi_index()
    out = []
    for m in re.finditer(r"(?m)^\*\*Art\.\s+(\S+)\s+(.+?)(?:\s+—\s+.*?)?\*\*\s*$", norms or ""):
        key = (m.group(1), answer_lint._code(m.group(2)))
        if key not in index:
            continue
        found = article_texts().get(key)
        texts = {k: t for k, t in (rule_quote.commi_of(found[1]) if found else []) if k}
        out.append({"ref": (m.group(1), m.group(2).strip()), "commi": [(k, texts.get(k, "")) for k in range(1, index[key] + 1)]})
    return out


def _commi_checked(norms: str, log: List[str]) -> str:
    """§2's paragraphs through interactive_flow.commi_moved: a «co. N» its article does not have moved or dropped."""
    from src.rag import interactive_flow as F
    rules = _rules_of(norms)
    if not rules:
        return norms
    parts = re.split(r"(?m)^(?=\*\*Art\.\s)|^(?=Altre norme del tema non trattate)", norms)
    heads = [p for p in parts if p.startswith("**Art.")]
    refs = [tuple(re.match(r"\*\*Art\.\s+(\S+)\s+(.+?)(?:\s+—\s+.*?)?\*\*", p).groups()) for p in heads]
    by_key = {F._key(r["ref"]): r for r in rules}
    placed, code_lines = {}, {}
    for i, p in enumerate(heads, 1):
        first, _, rest = p.partition("\n")
        chunks = [c.strip("\n") for c in re.split(r"\n[ \t]*\n", rest) if c.strip()]
        code_lines[i] = [first] + [c for c in chunks if all(l.startswith("- ") for l in c.split("\n"))]
        placed[i] = [c for c in chunks if c not in code_lines[i]]
    moved: List[Dict[str, Any]] = []
    placed = F.commi_moved(placed, refs, by_key, moved)
    if not moved:
        return norms
    log += [f"«co. {m['comma']}» under {m['from']}: " + (f"moved to {m['to']}" if m["to"] else "dropped")
            + " (commi_moved)" for m in moved]
    blocks = ["\n".join([code_lines[i][0]] + ["\n".join(code_lines[i][1:])] if len(code_lines[i]) > 1 else [code_lines[i][0]])
              + "".join(f"\n\n{x}" for x in placed.get(i, [])) for i in range(1, len(heads) + 1)]
    tail = [p for p in parts if p.startswith("Altre norme")]
    lead = [p for p in parts if p not in heads and p not in tail and p.strip()]
    return "\n\n".join([x.strip("\n") for x in lead] + blocks + [x.strip("\n") for x in tail])


def _phantoms_out(sections: List[Tuple[str, str]], norms: str, log: List[str]) -> List[Tuple[str, str]]:
    """interactive_flow.phantom_commi_out with the §2 articles' commi (_rules_of)."""
    from src.rag import interactive_flow as F
    found: List[Dict[str, Any]] = []
    out = F.phantom_commi_out(sections, _rules_of(norms), found)
    log += [f"sentence citing a comma its article does not have dropped: «{x['sentence'][:60]}» (phantom_commi_out)"
            for x in found]
    return out


def _procedural(text: str, chosen: List[Dict[str, Any]], listed: List[Dict[str, Any]], log: List[str]) -> str:
    """§4 through the current renderer's pieces -- see the module docstring."""
    from src.rag import interaction_copy as C, render_computed, rule_quote, topic_step, topic_tables
    from src.rag import interactive_flow as F
    blocks = OrderedDict((m.group(1), m.group(2).strip("\n")) for m in
                         re.finditer(r"(?ms)^\*\*([^*\n]+)\*\*\s*\n(.*?)(?=^\*\*[^*\n]+\*\*\s*$|\Z)", text or ""))
    refs = [F._parse_label(a) for t in chosen for a in t.get("matched") or [t["anchor"]] if a]
    mediation = render_computed.mediation_of({"topic_refs": refs})
    if mediation.get("required") and "Mediazione" not in blocks:
        blocks = OrderedDict([("Mediazione", C.MEDIATION or mediation["text"])] + list(blocks.items()))
        log.append(f"§4 Mediazione: the topics' matter is {mediation['subject']} (mediation_of)")
    if "Termini" in blocks:
        lines = blocks["Termini"].split("\n")
        terms = []
        for line in lines:
            parsed = _terms_of(re.sub(r"^\s*-\s*", "", line))
            terms.append((line, parsed[0] if parsed else None))
        kept = [line for line, term in terms if term is None or topic_step.carries_term(term, listed)]
        for line, term in terms:
            if line not in kept:
                log.append(f"§4 term out: «{line.strip()[:60]}» (carries_term)")
        if kept:
            blocks["Termini"] = "\n".join(kept)
        else:
            del blocks["Termini"]
    offences = []
    for t in chosen:
        for label in t.get("matched") or [t["anchor"]]:
            ref = F._parse_label(label)
            if rule_quote.offence(ref):
                rubrica = next((r for r in [_rubrica_of(text, ref[0])] if r), "")
                offences.append((ref[0], rubrica or topic_tables.CRIMINAL_PROCEDURE.get((ref[0], "c.p."), ("",))[0]
                                 or t["title"]))
    # v60 (v59-n1: 43, 45 c.p. chosen, no delitto, the frame not penale): chosen topics all in c.p./c.p.p. are criminal
    criminal = bool(refs) and all(answer_lint._code(c) in ("c.p.", "c.p.p.") for _, c in refs)
    if PROCEDURE in blocks or offences or criminal:
        found, shown, dated = [], [], True
        for line in (blocks.get(PROCEDURE) or "").split("\n"):
            m = re.match(r"^-\s+(.*?)\s+\(art\.\s*(\S+)\s+c\.p\.\):\s*procedibil", line)
            if m and (m.group(2), "c.p.") in topic_tables.CRIMINAL_PROCEDURE:
                found.append((m.group(2), "c.p."))
            elif m:
                shown.append((m.group(2), m.group(1)))
            if line.startswith("- Data del fatto: non nota"):
                dated = False
            many = re.match(r"^-\s+(.*):\s*procedibili d'ufficio", line)
            if many:
                shown += [(x.group(2), x.group(1)) for x in re.finditer(r"(.*?)\s+\(art\.\s*(\S+)\s+c\.p\.\)(?:;\s*|$)",
                                                                         many.group(1))]
        if not blocks.get(PROCEDURE):
            dated = False           # a §4 the transcript did not have: the date of the fact is unknown to it
        names = {n: r for n, r in shown}
        merged = list(dict.fromkeys(shown + [(n, names.get(n) or r) for n, r in offences]))
        merged = [x for i, x in enumerate(merged) if x[0] not in [y[0] for y in merged[:i]]]
        lines = render_computed.procedure_lines(found, merged, dated)
        now = "\n".join(lines)
        if now != blocks.get(PROCEDURE):
            log.append("§4 Procedibilità e competenza re-rendered (procedure_lines)")
        blocks[PROCEDURE] = now
    return "\n\n".join(f"**{k}**\n\n{v}" for k, v in blocks.items() if v.strip())


PROCEDURE = "Procedibilità e competenza"


def _rubrica_of(answer: str, number: str) -> str:
    m = re.search(rf"\*\*Art\. {re.escape(number)} c\.p\. — ([^*\n]+)\*\*", answer or "")
    return m.group(1).strip() if m else ""


def replay_answer(case: Dict[str, Any], topics: List[Dict[str, Any]], chosen: List[Dict[str, Any]], log: List[str]) -> str:
    from src.rag import interaction_copy as C, interactive_flow as F
    answer = case["answer"] or ""
    if not answer.strip():
        return answer
    body = re.sub(r"(?ms)^Se vuoi, posso approfondire:.*\Z", "", answer).rstrip()
    secs = answer_lint.sections(body)
    if not secs:
        return answer
    out = []
    norms_rubriche = body
    for title, text in secs:
        text = _stamps_rechecked(text, log)           # v60c (6): one stem with the rubrica or the first comma clears
        if title == F.ANSWER_TITLES[1]:
            text = _abrogated_out(text, log)          # v60c (8)
            text = _blocks_refiltered(text, log)
            text = _rebuilt_blocks(text, chosen, log)
            text = _commi_checked(text, log)
        if title == F.ANSWER_TITLES[3]:
            text = _procedural(text, chosen, topics, log)
        out.append((title, text))
    if F.ANSWER_TITLES[3] not in [t for t, _ in out]:
        procedural = _procedural("", chosen, topics, log)
        if procedural:
            at = next((i for i, (t, _) in enumerate(out) if t == F.ANSWER_TITLES[4]), len(out))
            out = out[:at] + [(F.ANSWER_TITLES[3], procedural)] + out[at:]
            log.append("§4 added: the chosen topics' delitti or matter (criminal_procedure, mediation_of)")
    out = _phantoms_out(out, dict(out).get(F.ANSWER_TITLES[1], ""), log)
    final_log: List[Dict[str, Any]] = []
    draft = F.answer_final(F._join([(t, x) for t, x in out if x.strip()]), final_log)
    log += [f"final: {x}" for x in final_log]
    draft = _read_back(draft, case, log)              # v60c (6, 7, 9): the read-back pass, on the case's own text
    others = [C.fill(C.CLOSING_TOPIC, n=i, title=t["title"]) for i, t in enumerate(topics, 1) if t not in chosen]
    others = [o for o in others if o]
    if others:
        draft += "\n\n" + C.fill_parts(C.CLOSING, topics=C.LIST_JOIN.join(others))
    del norms_rubriche
    return draft


# --- v60c: the rubrica stamps, abrogated articles, the read-back pass --------------------------------------------------

_STAMP_RE = re.compile(r"\s*\[DA VERIFICARE: l'art\. (\S+) ([^\]]+?) riguarda «([^»]*)»\]")


def _stamps_rechecked(text: str, log: List[str]) -> str:
    """A rubrica stamp the one-stem rule clears (the citing clause shares a stem with the rubrica or the article's
    first comma, the local text) is removed."""
    from src.rag import defensive_generation as dg
    out, last = [], 0
    for m in _STAMP_RE.finditer(text or ""):
        key = (m.group(1), answer_lint._code(m.group(2)))
        found = article_texts().get(key)
        start = max(text.rfind(". ", 0, m.start()), text.rfind("\n", 0, m.start())) + 1
        stop = [i for i in (text.find(". ", m.end()), text.find("\n", m.end())) if i >= 0]
        clause = text[start:m.start()] + " " + text[m.end():min(stop) if stop else len(text)]
        cleared = dg.one_stem_clears(dg._citing_clause(clause, key), m.group(3), dg.first_comma(found[1] if found else ""))
        out.append(text[last:m.start()] if cleared else text[last:m.end()])
        last = m.end()
        if cleared:
            log.append(f"stamp art. {key[0]} {key[1]} cleared ({', '.join(cleared[:3])})")
    return "".join(out) + (text or "")[last:] if out else text


def _abrogated_out(norms: str, log: List[str]) -> str:
    """§2 without the blocks -- and the «non trattate» items -- of articles whose local text is their abrogation."""
    from src.rag.article_text import is_abrogated
    for m in list(re.finditer(r"(?ms)^\*\*Art\.\s+(\S+)\s+([^—*\n]+?)(?:\s+—[^*\n]*)?\*\*\s*\n.*?(?=^\*\*Art\.|^Altre norme|\Z)",
                              norms or ""))[::-1]:
        found = article_texts().get((m.group(1), answer_lint._code(m.group(2))))
        if found and is_abrogated(found[1]):
            norms = norms[:m.start()] + norms[m.end():]
            log.append(f"art. {m.group(1)} {m.group(2).strip()}: abrogated, its block out")
    for number, code in [k for k, v in article_texts().items() if is_abrogated(v[1])]:
        line = re.search(rf"(?m)^Altre norme del tema non trattate:.*\bart\. {re.escape(number)} ", norms or "")
        if line:
            norms = re.sub(rf"(Altre norme del tema non trattate:[^\n]*?)(?:,\s*)?\bart\. {re.escape(number)} [^,\n]*?(?=,|\.?$)",
                           r"\1", norms, flags=re.M)
            log.append(f"art. {number} {code}: abrogated, out of «non trattate»")
    return re.sub(r"(?m)^Altre norme del tema non trattate:\s*\.?\s*$\n?", "", norms)


_CASES: Optional[Dict[str, str]] = None


def case_text_of(cid: str) -> str:
    """The case's own text (tests/eval_narrative_cases.json), "" for a case not in it."""
    global _CASES
    if _CASES is None:
        import json
        path = Path(__file__).resolve().parents[1] / "tests" / "eval_narrative_cases.json"
        try:
            _CASES = {str(c.get("id")): str(c.get("question") or "") for c in json.loads(path.read_text("utf-8"))["cases"]}
        except (OSError, ValueError, KeyError):
            _CASES = {}
    return _CASES.get(str(cid or "").split("@")[0], "")


def _read_back(draft: str, case: Dict[str, Any], log: List[str]) -> str:
    from src.rag import interactive_flow as F, readback
    text = case_text_of(case.get("_id", ""))
    if not text:
        log.append("read-back skipped: no case text")
        return draft
    cited = {(n, answer_lint._code(c)) for n, c in re.findall(r"\bart\.\s*(\d+(?:-[a-z]+)?)\s+(c\.\s*[cp]\.(?:\s*[pca]\.)?)",
                                                         draft, re.IGNORECASE)}
    texts = article_texts()
    articles = {f"{n} {c.replace(' ', '')}": f"{_rubrica(n, c)} {texts[(n, c)][1]}" for n, c in cited if (n, c) in texts}
    heads = {f"{n} {answer_lint._code(c).replace(' ', '')}": h for n, c, h in re.findall(
        r"(?m)^\*\*Art\.\s+(\S+)\s+([^—*\n]+?)\s+—\s+([^*\n]+)\*\*", draft)}
    rubriche = {k: heads.get(k) or _rubrica(*k.split(" ", 1)) for k in articles}
    procedural = dict(answer_lint.sections(draft)).get(F.ANSWER_TITLES[3], "")
    sources = [text, procedural] + [v for v in articles.values()]
    rb: Dict[str, Any] = {}
    out = readback.read_back(draft, sources, {}, (F.ANSWER_TITLES[3],), rb, articles=articles, case=text,
                             rubriche=rubriche)
    for key in ("ungrounded", "contradictions", "not_in_case"):
        for item in rb.get(key) or []:
            if key == "not_in_case":
                log.append(f"read-back: {'calcolato' if not item.get('not_in_case') else 'non risulta'} "
                           f"{(item.get('not_in_case') or item.get('calculated'))}")
            else:
                log.append(f"read-back {key}: «{str(item.get('dropped'))[:80]}»")
    return out


def replay_case(case: Dict[str, Any], cid: str = "") -> Dict[str, Any]:
    case = {**case, "_id": cid}
    log: List[str] = []
    old_chosen = [t["n"] for t in answer_lint.chosen(case)]
    topics, pieces = replay_round1(case, log)
    round1 = rebuild_round1(case, topics, pieces, log)
    chosen = [t for t in topics if t["n"] in old_chosen]
    numbers = [i for i, t in enumerate(topics, 1) if t in chosen]
    from src.rag import interaction_copy as C
    round2 = case["round2"]
    m = re.search(r"Tratterò:\s*([^.]+)\.", round2 or "")
    if m:
        round2 = round2[:m.start(1)] + "; ".join(t["title"] for t in chosen) + round2[m.end(1):]
    answer = replay_answer(case, topics, chosen, log)
    return {**case, "round1": round1, "round2": round2, "answer": answer,
            "choice": ", ".join(map(str, numbers)) or case.get("choice", ""), "replay_log": log}


def replay_file(path: str) -> "OrderedDict[str, dict]":
    return OrderedDict((cid, replay_case(case, cid)) for cid, case in answer_lint.read_cases(path).items())


def write_file(source: str, cases: "OrderedDict[str, dict]", target: str) -> None:
    """The transcript with each case's rounds re-rendered and its replay log, in the source's layout."""
    lines = [f"# Re-rendered from {os.path.basename(source)} (scripts/interactive_review.py --render-from)", ""]
    for cid, case in cases.items():
        lines += [f"## {cid}", "", "Replay: " + ("; ".join(case.get("replay_log") or []) or "no change"), "",
                  "### round 1 (round1, replay) — user: (case)", "", case["round1"], "",
                  f"### round 2 (round2, replay) — user: {case.get('choice', '')}", "", case["round2"], "",
                  "### round 3 (answer, replay) — user: procedi", "", case["answer"], ""]
    with open(target, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
