"""Phase 1 offline: RULE_QUOTE, RENDER_COMPUTED and TOPIC_STEP replayed on a diagnose_long run's dumps -- no model, no
database, nothing written but the report.

Per case of RUN/cases/<id>.json (and its llm/ dumps):
  A  RULE_QUOTE: the plan's articles (case_analysis_plan_final, norme_contrarie) with the commi the writer was given
     (PASSI CON ANCORA's "(co. K)" passages first, then TESTI DELLE NORME, material.txt's ARTICOLI); the comma selected
     for the client and its requisites (the block sections 3 and 4 would get); then rule_quote.check_sections on the
     answer as the run wrote it: how many rule sentences would be replaced, of which articles, each before / after.
     --show "leonardo:1054,1062" prints those articles' block and every sentence citing them, before and after.
  B  RENDER_COMPUTED: the procedural section render_computed.render makes from the run's computed block (the case's
     fact signals, the regime tools, the criminal profile for the client's side, the terms and tools the section was
     given -- read off its llm dump), next to what the run wrote; and which plan articles the action gate moves to
     the objections block.
  C  TOPIC_STEP: topic_step.build on the run's frame, analysis and pool (its multi-signal and regime-candidate rows,
     fusion ranks), with the recall of the expected norms -- an approximation of --topics-only, which reruns retrieval.

    venv/bin/python scripts/phase1_offline.py RUN [--show leonardo:1054,1062] [--render h3,p1] [--topics h3,p1]
        [--only h3,p1,leonardo] [--out RUN/phase1_offline.md]
"""

import argparse
import glob
import importlib.util
import json
import os
import re
import sys
from collections import Counter

for _key, _value in (("NEO4J_URI", "bolt://offline"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                     ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(_key, _value)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_SECTION_RE = re.compile(r"(?m)^\*\*(\d+)\.\s*(.+?)\*\*[ \t]*$")
_BLOCK_TITLE_RE = re.compile(r"(?m)^([A-ZÀÈÉÌÒÙ][A-ZÀÈÉÌÒÙ0-9 ’'\-/().,]{2,}):\n")
_PASSAGE_RE = re.compile(r"^\[A\d+\] art\. (\S+) (.+?)(?:, co\. (\d+))?: (.*)$")
_TEXT_HEAD_RE = re.compile(r"^\[art\. (\S+) ([^\]]+)\]\s*$")
_CO_RE = re.compile(r"\(co\. (\d+)\)\s*")


_INFERRED = ("REGIME_CANDIDATES", "PARTIES_ENFORCED", "ROLE_BLOCKS_ENABLED")


def _flags(record: dict, dumps: list) -> None:
    """The run's flags as the environment, with this phase's on; the ones the record does not list read off what the
    run did: a regime's tool or its candidates (REGIME_CANDIDATES), the parties line (PARTIES_ENFORCED), the side
    (ROLE_BLOCKS_ENABLED)."""
    from src.rag import fact_signals
    for name in _INFERRED:
        os.environ.pop(name, None)
    for name, value in (record.get("flags") or {}).items():
        if value is not None:
            os.environ[name] = str(value)
    for name in ("RULE_QUOTE", "RENDER_COMPUTED", "TOPIC_STEP"):
        os.environ[name] = "true"
    regimes = _last(record, "case_analysis_fact_regimes") or {}
    regime_tools = {t["id"] for t in fact_signals.PROCEDURAL_TOOLS if t.get("regime")}
    if set(regimes.get("tools") or []) & regime_tools or _events(record, "case_analysis_regime_candidates"):
        os.environ["REGIME_CANDIDATES"] = "true"
    s5 = next((user for stage, _, user, _ in dumps if stage.startswith("section_profili_processuali")), "")
    if "Parti da convenire:" in (_blocks(s5).get("CONCLUSIONI DI DIRITTO ACCERTATE DAI FATTI") or ""):
        os.environ["PARTIES_ENFORCED"] = "true"
    if regimes.get("side"):
        os.environ["ROLE_BLOCKS_ENABLED"] = "true"


def _dl():
    spec = importlib.util.spec_from_file_location("diagnose_long", os.path.join(ROOT, "scripts", "diagnose_long.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _events(record: dict, name: str) -> list:
    return [d for e, d, _ in record.get("events") or [] if e == name]


def _last(record: dict, name: str):
    found = _events(record, name)
    return found[-1] if found else None


def _dumps(case_dir: str) -> list:
    """[(stage, system, user, reply)] of the case's llm dumps, in call order."""
    out = []
    for path in sorted(glob.glob(os.path.join(case_dir, "llm", "*.txt"))):
        text = open(path, encoding="utf-8").read()
        stage = re.sub(r"^\d+_", "", os.path.basename(path)[:-4])
        parts = re.split(r"(?m)^===== (SYSTEM|USER|REPLY) =====\n", text)
        got = {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1, 2)}
        out.append((stage, got.get("SYSTEM", ""), got.get("USER", ""), got.get("REPLY", "")))
    return out


def _blocks(material: str) -> dict:
    marks = list(_BLOCK_TITLE_RE.finditer(material or ""))
    return {m.group(1): material[m.end():marks[i + 1].start() if i + 1 < len(marks) else len(material)].strip("\n")
            for i, m in enumerate(marks)}


def _norm_code(code: str) -> str:
    from src.rag.plan_then_write import _code_key
    return _code_key(code.strip())


def _add_commi(store: dict, ref: tuple, body: str, comma=None) -> None:
    """Into store[ref] {k: text}: a "(co. K) ..." body split on its labels; a single comma's body under comma; an
    unlabelled body under 0 (the text as given, commi unknown)."""
    body = re.sub(r"^\d{1,3}\.\s+", "", " ".join(body.split()))
    labelled = list(_CO_RE.finditer(body))
    have = store.setdefault(ref, {})
    if labelled:
        for i, m in enumerate(labelled):
            end = labelled[i + 1].start() if i + 1 < len(labelled) else len(body)
            text = re.sub(r"\s*\[commi non riportati:[^\]]*\]\s*$", "", body[m.end():end]).strip()
            text = re.sub(r"^\d{1,3}\.\s+", "", text)
            k = int(m.group(1))
            if text and len(text) > len(have.get(k, "")):
                have[k] = text
    elif comma:
        if len(body) > len(have.get(int(comma), "")):
            have[int(comma)] = body
    elif body and 0 not in have:
        have[0] = body


def article_commi(case_dir: str, dumps: list) -> dict:
    """{(number, code): {k: text}} of every article the writer was given, by its passages first."""
    store: dict = {}
    for stage, _, user, _ in dumps:
        if not stage.startswith("section"):
            continue
        blocks = _blocks(user)
        for line in (blocks.get("PASSI CON ANCORA") or "").split("\n"):
            m = _PASSAGE_RE.match(line.strip())
            if m:
                _add_commi(store, (m.group(1), _norm_code(m.group(2))), m.group(4), m.group(3))
        for title in ("TESTI DELLE NORME", "TESTI DELLE NORME PROCESSUALI"):      # the terms' texts are cut at 700
            body = blocks.get(title) or ""
            entries = re.split(r"\n(?=\[art\. )", body)
            for entry in entries:
                head, _, text = entry.partition("\n")
                m = _TEXT_HEAD_RE.match(head.strip())
                if m and text.strip() and "[…]" not in text:
                    _add_commi(store, (m.group(1), _norm_code(m.group(2))), text)
    material = os.path.join(case_dir, "material.txt")
    if os.path.isfile(material):
        text = open(material, encoding="utf-8").read().split("========== SENTENZE")[0]
        for entry in re.split(r"\n(?=\[art\. )", text):
            head, _, body = entry.partition("\n")
            m = _TEXT_HEAD_RE.match(head.strip())
            if m:
                body = "\n".join(line for line in body.split("\n") if not line.startswith(">>")).strip()
                if body:
                    _add_commi(store, (m.group(1), _norm_code(m.group(2))), body)
    return store


def rows_from(store: dict) -> list:
    """Article rows as rule_quote reads them: numbered commi ("2. ...") when known, else the text as given."""
    rows = []
    for ref, commi in store.items():
        numbered = {k: t for k, t in commi.items() if k}
        text = "\n".join(f"{k}. {t}" for k, t in sorted(numbered.items())) if numbered else commi.get(0, "")
        if text:
            rows.append({"_article": ref, "s": {"plain_text": text}, "_known_commi": sorted(numbered)})
    return rows


def _party(dumps: list) -> tuple:
    for stage, system, _, _ in dumps:
        if stage.startswith("section"):
            m = re.search(r"nell'interesse di (.+?)(?: \(([^)]*)\))?, seguendo", system)
            if m:
                return m.group(1), m.group(2) or ""
    return "", ""


def plan_of(record: dict, dumps: list) -> dict:
    final = _last(record, "case_analysis_plan_final") or {}
    issues = [{"titolo": q.get("titolo") or "", "argomento": "", "obiezione": "", "replica": "",
               "norme": [(lab.split(" ", 1)[0], _norm_code(lab.split(" ", 1)[1])) for lab in q.get("norme") or [] if " " in lab]}
              for q in final.get("issues") or []]
    party, position = _party(dumps)
    contrary = [lab for d in _events(record, "case_analysis_action_gate_plan") for lab in d.get("adverse") or []]
    for d in _events(record, "case_analysis_role_fit"):
        contrary += [a.get("article") for a in d.get("articles") or [] if isinstance(a, dict)
                     and a.get("side") == "controparte" and a.get("article")]
    return {"questioni": issues, "parte_assistita": party, "posizione": position,
            "norme_contrarie": list(dict.fromkeys(c for c in contrary if c))}


def analysis_of(record: dict, plan: dict) -> dict:
    gate = {d["article"]: d["reason"] for d in _events(record, "case_analysis_action_gate") if not d.get("fits")}
    signals = json.loads(json.dumps(record.get("fact_signals") or {}))
    return {"frame": record.get("frame") or {}, "questioni": record.get("questioni") or [], "fact_signals": signals,
            "action_gate": gate, "parte_assistita": plan.get("parte_assistita") or ""}


def sections_of(draft: str) -> list:
    marks = list(_SECTION_RE.finditer(draft or ""))
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(draft)
        body = draft[m.end():end].strip("\n")
        if i + 1 == len(marks):
            body = body.split("\n\n⚠️")[0]
        out.append((m.group(2).strip(), body.strip()))
    return out


# --- A ---------------------------------------------------------------------------------------------------------------

def replay_a(record: dict, dumps: list, case_dir: str) -> dict:
    from src.rag import rule_quote
    plan = plan_of(record, dumps)
    analysis = analysis_of(record, plan)
    store = article_commi(case_dir, dumps)
    rows = rows_from(store)
    sections = sections_of(record.get("draft") or "")
    checked, log = rule_quote.check_sections(sections, plan, rows, analysis, [])
    rules = rule_quote.article_rules(plan, rows, analysis)
    from src.rag.plan_then_write import _plan_refs
    missing = [f"{n} {c}" for n, c in _plan_refs(plan) if not any(r["ref"] == (n, c) for r in rules)]
    return {"plan": plan, "analysis": analysis, "rows": rows, "sections": sections, "checked": checked, "log": log,
            "rules": rules, "no_text": missing, "store": store}


def _sentences_citing(sections: list, number: str) -> list:
    from src.rag.plan_then_write import _sentence_spans
    out = []
    for title, text in sections:
        for s, e in _sentence_spans(text):
            sentence = text[s:e]
            if re.search(rf"\bart(?:t)?\.\s*(?:[\w-]+\s*(?:,|e)\s*)*{re.escape(number)}\b", sentence):
                out.append((title, " ".join(sentence.split())))
    return out


def show_articles(a: dict, numbers: list) -> list:
    from src.rag import rule_quote
    lines = []
    for number in numbers:
        rule = next((r for r in a["rules"] if r["ref"][0] == number), None)
        lines += ["", f"#### art. {number} " + (rule["ref"][1] if rule else "(no text in the dumps)")]
        if rule:
            known = next((r["_known_commi"] for r in a["rows"] if r["_article"] == rule["ref"]), [])
            lines += [f"commi in the dumps: {known or 'unnumbered text'}; selected for the client: {rule['selected']} "
                      f"({rule['how']}); client roles {rule['client']}, other side {rule['counter']}",
                      "", "Block for sections 3 and 4:", "```",
                      rule_quote._entry(rule, a["plan"].get("parte_assistita") or "la parte assistita"), "```"]
        before = _sentences_citing(a["sections"], number)
        after = {(e["section"], e["sentence"]): e for e in a["log"]["entries"]}
        names = rule_quote.party_names(a["analysis"], a["plan"])
        lines.append("")
        lines.append(f"Sentences citing it in the run's answer ({len(before)}):")
        for title, sentence in before:
            entry = after.get((title, sentence[:400]))
            kind = "rule sentence" if rule_quote.rule_sentence(sentence, names) else "application (names a party or no rule verb)"
            lines.append(f"- [{title}] BEFORE ({kind}): {sentence}")
            if entry:
                lines.append(f"  AFTER ({entry['how']}, {entry['why']}): {entry['now']}")
            elif rule and kind == "rule sentence":
                verdict = rule_quote.judge(sentence, rule)
                lines.append(f"  AFTER: kept ({verdict['why']})")
            else:
                lines.append("  AFTER: unchanged (not a rule sentence)")
    return lines


# --- B ---------------------------------------------------------------------------------------------------------------

def replay_b(record: dict, dumps: list) -> dict:
    from src.rag import fact_signals, render_computed
    plan = plan_of(record, dumps)
    analysis = analysis_of(record, plan)
    signals = analysis["fact_signals"]
    s5 = next(((system, user) for stage, system, user, _ in dumps if stage.startswith("section_profili_processuali")), None)
    blocks = _blocks(s5[1]) if s5 else {}
    ran_tools = set((_last(record, "case_analysis_fact_regimes") or {}).get("tools") or [])
    fact_signals.fit_to_role(signals, f"{plan['parte_assistita']} ({plan['posizione']})")
    have = {t["id"] for t in signals.get("tools") or []}
    keyed = [t for t in fact_signals.regime_tools(signals, analysis) if t["id"] not in have]
    given_tools = [line[2:] for line in (blocks.get("STRUMENTI PROCESSUALI INDICATI DAI FATTI") or "").split("\n")
                   if line.startswith("- ")]
    for tool in keyed:          # the text the run gave (its article's rubrica, read from the corpus)
        number = tool["refs"][0][0] if tool.get("refs") else ""
        line = next((g for g in given_tools if re.search(rf"\(art\. {re.escape(number)}\b", g)), "")
        if line:
            tool["text"] = line.split(" Nel caso:")[0].strip()
    signals["tools"] = [t for t in list(signals.get("tools") or []) + keyed if not ran_tools or t["id"] in ran_tools]
    given_terms = blocks.get("TERMINI CALCOLATI DALLE DATE DEL CASO") or ""
    if s5:
        signals["terms"] = [t for t in signals.get("terms") or [] if t.get("text") and t["text"][:60] in given_terms]
    gated_plan, moved = render_computed.move_gated(plan, analysis)
    written = next((text for title, text in sections_of(record.get("draft") or "")
                    if title.startswith("Profili processuali")), "")
    party = f"{plan['parte_assistita']} {plan['posizione']}".strip()
    return {"rendered": render_computed.render(analysis, party), "written": written, "moved": moved, "party": party,
            "objections": render_computed.objections_block(gated_plan), "had_dump": bool(s5)}


# --- C ---------------------------------------------------------------------------------------------------------------

def rubrica_table(reports: str) -> dict:
    """{(number, code): rubrica} from every local run's records: their pools' rubriche and the articles their
    institutions resolved to -- never the expected articles' (expected_info), the answer key."""
    out = {}
    for path in glob.glob(os.path.join(reports, "**", "cases", "*.json"), recursive=True):
        try:
            record = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(record, dict):
            continue
        items = [(p.get("label"), p.get("rubrica")) for p in record.get("pool") or [] if isinstance(p, dict)]
        items += [(a.get("label"), a.get("rubrica")) for i in record.get("institutions") or [] if isinstance(i, dict)
                  for a in i.get("articles") or [] if isinstance(a, dict)]
        for label, rubrica in items:
            if label and rubrica and " " in label:
                number, code = label.split(" ", 1)
                out.setdefault((number, code), rubrica)
    return out


_LINKS = {}


def links_for(links_dir: str, reports: str):
    """topic_step.CsvLinks from article_links' units.csv and refers_to.csv and the rubrica table, read once."""
    import csv
    from src.rag import topic_step
    if "links" not in _LINKS:
        units = [(r["code"], r["level"], r["unit name"], r["article"])
                 for r in csv.DictReader(open(os.path.join(links_dir, "units.csv"), encoding="utf-8"))]
        refers = [(r["from article"], r["to article"])
                  for r in csv.DictReader(open(os.path.join(links_dir, "refers_to.csv"), encoding="utf-8"))]
        table = rubrica_table(reports)
        _LINKS["links"] = topic_step.CsvLinks(units, refers, table)
        _LINKS["size"] = len(table)
    return _LINKS["links"]


def replay_c(record: dict, dl, ev, links=None) -> dict:
    from src.rag import topic_step
    fused = (record.get("fusion") or {}).get("fused") or []
    rank = {lab: i for i, lab in enumerate(fused, 1)}
    rows = []
    for p in record.get("pool") or []:
        number, _, code = p["label"].partition(" ")
        rows.append({"_article": (number, code), "_multi_signal": bool(p.get("multi_signal")),
                     "_regime_candidate": "regime_candidate" in (p.get("sources") or []),
                     "_fused_rank": rank.get(p["label"]), "s": {"titolo_articolo": p.get("rubrica") or ""}})
    analysis = {"frame": record.get("frame") or {}, "questioni": record.get("questioni") or [],
                "fact_signals": record.get("fact_signals") or {}, "area": record.get("area"),
                "institution_refs": {}}
    for inst in record.get("institutions") or []:       # the run's resolutions, as the pipeline now keeps them
        if isinstance(inst, dict) and inst.get("nome"):
            analysis["institution_refs"].setdefault(inst["nome"], []).extend(a["label"] for a in inst.get("articles") or [])
    case_text = (record.get("frame") or {}).get("facts") or []
    topics = topic_step.build(analysis, rows, links, " ".join(map(str, case_text)))
    return {"topics": topics, "recall": dl.topic_recall(topics, record.get("expected") or [], ev)
            if record.get("expected") else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run")
    parser.add_argument("--only", help="comma-separated case ids (default: every case)")
    parser.add_argument("--show", action="append", default=[], metavar="CASE:N,N",
                        help="articles to print before/after for a case (a case id or a part of it)")
    parser.add_argument("--render", default="h3,p1", help="cases whose rendered procedural section is printed")
    parser.add_argument("--topics", default="h3,p1", help="cases whose topics are printed")
    parser.add_argument("--out")
    parser.add_argument("--links", help="article_links' output folder (units.csv, refers_to.csv); default: "
                                        "<run>/../article_links_out")
    args = parser.parse_args()
    run = os.path.abspath(args.run)
    cases_dir = os.path.join(run, "cases")
    ids = sorted(n[:-5] for n in os.listdir(cases_dir) if n.endswith(".json") and not n.endswith(".input.json"))
    if args.only:
        ids = [i for i in ids if i in set(args.only.split(","))]
    shows = {}
    for item in args.show or ["leonardo:1054,1062"]:
        case, _, numbers = item.partition(":")
        shows[case] = [n.strip() for n in numbers.split(",") if n.strip()]
    renders, topics_for = set(args.render.split(",")), set(args.topics.split(","))
    dl = _dl()
    links_dir = args.links or os.path.join(os.path.dirname(run), "article_links_out")
    links = links_for(links_dir, os.path.dirname(run)) if os.path.isfile(os.path.join(links_dir, "units.csv")) else None
    sys.path.insert(0, ROOT)
    ev = dl._eval_module()

    out = [f"# Phase 1 offline replay of {run}", ""]
    table = ["## A. RULE_QUOTE: rule sentences the check would replace", "",
             "| case | rule sentences judged | replaced | by article | plan articles without text in the dumps |",
             "|---|---|---|---|---|"]
    detail_a, detail_b, detail_c, recall_rows = [], [], [], []
    low_share = []
    totals = Counter()
    for cid in ids:
        record = json.load(open(os.path.join(cases_dir, f"{cid}.json"), encoding="utf-8"))
        if record.get("kind") == "topics":
            continue
        case_dir = os.path.join(cases_dir, cid)
        dumps = _dumps(case_dir)
        _flags(record, dumps)
        try:
            a = replay_a(record, dumps, case_dir)
        except Exception as exc:
            table.append(f"| {cid} | error: {type(exc).__name__}: {exc} | | | |")
            continue
        log = a["log"]
        totals["judged"] += log["judged"]
        totals["replaced"] += log["replaced"]
        by = ", ".join(f"{k} ×{v}" for k, v in log["by_article"].items()) or "-"
        table.append(f"| {cid} | {log['judged']} | {log['replaced']} | {by} | {', '.join(a['no_text']) or '-'} |")
        low_share += [(cid, e) for e in log["entries"] if e.get("share") is not None and e["share"] < 0.2]
        if log["entries"]:
            detail_a += ["", f"### {cid}: replacements"]
            for e in log["entries"]:
                detail_a += [f"- [{e['section']}] art. {e['article']} ({e['how']}; {e['why']})",
                             f"  REJECTED: {e['sentence']}", f"  NOW: {e['now']}"]
        for key, numbers in shows.items():
            if key in cid:
                detail_a += ["", f"### {cid}: before / after"] + show_articles(a, numbers)
        if cid in renders:
            b = replay_b(record, dumps)
            detail_b += ["", f"### {cid}: §5 rendered by template" + ("" if b["had_dump"] else
                                                                       " (no §5 dump: every computed term and tool)"),
                         f"client: {b['party']}",
                         "", "```", b["rendered"] or "(nothing computed: the model would write it)", "```", "",
                         f"Articles the action gate moves to the objections block: "
                         + (", ".join(f"{m['article']} (from «{m['issue']}»: {m['reason']})" for m in b["moved"]) or "none"),
                         *(["", "```", b["objections"], "```"] if b["objections"] else []),
                         "", f"<details><summary>§5 as the run wrote it ({len(b['written'])} chars)</summary>", "",
                         b["written"], "", "</details>"]
        try:
            c = replay_c(record, dl, ev, links)
        except Exception as exc:
            c = {"topics": [], "recall": None, "error": f"{type(exc).__name__}: {exc}"}
        if c.get("recall"):
            rc = c["recall"]
            recall_rows.append(f"| {cid} | {len(c['topics'])} | {rc['expected']} | {rc['in_topics']} ({rc['share']}) | "
                               f"{rc['in_top3']} ({rc['share_top3']}) | "
                               + (", ".join("/".join(f"{x} {y}" for x, y in alts) for alts in rc["missing"]) or "-") + " |")
            totals["expected"] += rc["expected"]
            totals["in_topics"] += rc["in_topics"]
            totals["in_top3"] += rc["in_top3"]
        if cid in topics_for:
            detail_c += ["", f"### {cid}", "", "```", dl.topics_text(c["topics"]), "```"]
    table.append(f"| **all** | {totals['judged']} | {totals['replaced']} | | |")
    low = [(cid, e) for cid, e in low_share]
    table += ["", f"### Articles with share < 20% ({len(low)} replacements)", "",
              "| case | article | share | section | sentence |", "|---|---|---|---|---|"]
    table += [f"| {cid} | {e['article']} | {e['share']:.0%} | {e['section']} | {e['sentence'][:160]} |" for cid, e in low]
    by = Counter(e["article"] for _, e in low)
    table += ["", "By article: " + ", ".join(f"{a} ×{n}" for a, n in by.most_common())]
    out += table + detail_a
    out += ["", "## B. RENDER_COMPUTED"] + detail_b
    out += ["", "## C. TOPIC_STEP on the run's pool (approximation of --topics-only)", "",
            f"Links: {'units.csv + refers_to.csv, rubriche of ' + str(_LINKS.get('size')) + ' articles from the local runs' if links else 'none'}.", "",
            "| case | topics | expected | in topics | in top 3 | missing |", "|---|---|---|---|---|---|"] + recall_rows
    if totals["expected"]:
        out.append(f"| **all** | | {totals['expected']} | {totals['in_topics']} "
                   f"({totals['in_topics'] / totals['expected']:.3f}) | {totals['in_top3']} "
                   f"({totals['in_top3'] / totals['expected']:.3f}) | |")
    out += detail_c
    text = "\n".join(out) + "\n"
    path = args.out or os.path.join(run, "phase1_offline.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print(f"Written to {path}")


if __name__ == "__main__":
    main()
