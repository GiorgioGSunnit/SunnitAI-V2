"""INTERACTIVE_FLOW, review of an --interactive-sim report (no database, no model).

    python scripts/interactive_review.py ~/SunnitAI-V2/diagnose_reports/v54-interactive.md \
        --facts ~/SunnitAI-V2/diagnose_reports/v52-candidate [--out review.md]

Reads the report's per-case rounds (round 1, round 2, the answer) and writes:
  1. the checks of each case's lists and answer against the interactive rules: R1 a generic rubrica as anchor, R1b an
     anchor whose rubrica shares no word with its title, R3 an open term with no topic / a decided term as a topic, R4 more
     than five questions / a question not from the copy's templates / a question on the other side's possession, R5 a
     gated action left in a title, a duplicate topic; L1 a 1062 topic without its two questions, L2 the §2 status lines,
     L3 one heading per article (1061 its own), L4 an objection placeholder / no works objection while that question
     is open, L5 round 2's parties line shorter than the procedural section's, L6 the copy (inizio del, improcedibilità,
     altri profili by Capo); and the extras (cited, not expected);
  2. per case with topic recall under 50%: each expected norm outside every topic, its rubrica and Capo, the
     institution that resolved it in the facts run (its institutions), the fact rules and regimes whose ranges hold it --
     what a missing regime or rule table would have to anchor;
  3. per case with no question: the question families the facts run's extracted facts would raise (interactive_flow
     .questions, every family), its open terms and the fact each runs from, its undated events -- the families to write;
  4. (v55) --rules: why those fact rules did not fire -- each condition, what the extraction holds for it, whether the
     case text states it.
--facts: a single-shot run of the same cases (its records carry the extracted facts, the frame, the institutions).
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def parse_report(path: str) -> dict:
    """{case: {"table": row cells, "round1", "round2", "answer", "header": text before the rounds}}."""
    text = open(path, encoding="utf-8").read()
    head, *parts = re.split(r"(?m)^## ", text)
    table = {}
    for line in head.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("| ") and cells and cells[0] not in ("case", "**all**") and not cells[0].startswith("---"):
            table[cells[0]] = cells
    out = {}
    for part in parts:
        cid, _, body = part.partition("\n")
        rounds = re.split(r"(?m)^### ", body)
        case = {"header": rounds[0], "table": table.get(cid.strip()), "round1": "", "round2": "", "answer": ""}
        for r in rounds[1:]:
            title, _, rest = r.partition("\n")
            kind = re.search(r"\((\w+),", title)
            kind = kind.group(1) if kind else ""
            key = {"round1": "round1", "round2": "round2", "answer": "answer"}.get(kind)
            if key:
                case[key] = rest.strip()
        out[cid.strip()] = case
    return out


def topics_of(round1: str) -> list:
    """[(n, title, [articles], line)] of round 1's offered topics."""
    out = []
    block = round1.split("**Argomenti individuati**", 1)[-1].split("**Altri profili**")[0].split("Ecco i fatti")[0]
    for m in re.finditer(r"(?m)^(\d+)\. \*\*(.+?)\*\* \(artt\. ([^)]*)\)(.*)$", block):
        arts = []
        for group in m.group(3).split(";"):
            g = group.strip()
            code = re.search(r"([a-zA-Z].*)$", re.sub(r"^[\d\s,a-z-]*?(?=\s[a-zA-Z.]+\.?$)", "", g))
            nums = re.findall(r"\b\d+(?:-[a-z]+)?\b", g)
            codem = re.search(r"\s((?:c\.|d\.|l\.|Cod|cod|L\.|D)[^\d]*)$", g)
            arts += [f"{n} {codem.group(1).strip() if codem else ''}".strip() for n in nums]
        out.append((int(m.group(1)), m.group(2), arts, m.group(0)))
    return out


def others_of(round1: str) -> list:
    if "**Altri profili**" not in round1:
        return []
    block = round1.split("**Altri profili**", 1)[1].split("Ecco i fatti")[0]
    return [l[2:].strip() for l in block.splitlines() if l.startswith("- ")]


def questions_of(round1: str) -> list:
    if "**Da chiarire**" not in round1:
        return []
    block = round1.split("**Da chiarire**", 1)[1].split("Dimmi su quale")[0]
    return [re.sub(r"^\d+\.\s*", "", l).strip() for l in block.splitlines() if re.match(r"^\d+\.\s", l)]


def _template_res():
    from src.rag import interaction_copy as C
    out = []
    for key, tmpl in C.QUESTIONS.items():
        pattern = re.escape(tmpl)
        for slot in re.findall(r"\\\{(\w+)\\\}", pattern):
            pattern = pattern.replace(r"\{" + slot + r"\}", r".+?")
        out.append((key, re.compile("^" + pattern + "$")))
    return out


def check_case(cid: str, case: dict, rubriche: dict, expected: list, ev, dl) -> dict:
    from src.rag import topic_step
    issues = []
    r1, r2, ans = case["round1"], case["round2"], case["answer"]
    topics = topics_of(r1)
    qs = questions_of(r1)
    templates = _template_res()

    def rub(label):
        n, _, c = label.partition(" ")
        return rubriche.get((n, c)) or rubriche.get((n, topic_step._code(c))) or ""
    # R1 / R1b
    for n, title, arts, _ in topics:
        if not arts:
            continue
        rubrica = rub(arts[0])
        if rubrica and topic_step.generic_rubrica(rubrica):
            issues.append(f"R1 topic {n} «{title}» anchored on a generic rubrica: {arts[0]} «{rubrica}»")
        elif rubrica and not (set(topic_step.content_words(title)) & set(topic_step.content_words(rubrica))) \
                and not re.search(r"usucapion|termine", title, re.I):
            issues.append(f"R1b topic {n} «{title}»: anchor {arts[0]} «{rubrica}» shares no word with the title")
    # R3
    facts_block = r1.split("Ecco i fatti", 1)[-1].split("**Da chiarire**")[0]
    titles = " ".join(t[1].lower() for t in topics)
    if "L'usucapione va verificata" in facts_block and "usucapion" not in titles:
        issues.append("R3 the usucapione term is open but no usucapione topic")
    for m in re.finditer(r"(?:^|\. )([A-Z][^.:]{3,80}): [^.]*?(DA VERIFICARE|INCERTO|NON VERIFICABILE)", facts_block):
        label = m.group(1).strip().lower()
        if not any(len(set(topic_step.content_words(label)) & set(topic_step.content_words(t[1]))) for t in topics):
            issues.append(f"R3 open term «{m.group(1).strip()}» ({m.group(2)}) has no topic")
    if re.search(r"decadenza dall'azione di reintegrazione|azione possessoria", titles) and "ormai preclusa" in facts_block:
        issues.append("R3 a decided term (possessoria DECORSO) is a topic")
    # R4
    if len(qs) > 5:
        issues.append(f"R4 {len(qs)} questions (more than 5)")
    for q in qs:
        if not any(rx.match(q) for _, rx in templates):
            issues.append(f"R4 question not from a template: «{q[:90]}»")
    # R5 and duplicates
    if re.search(r"confessori\w* e negatori|negatoria", titles):
        issues.append("R5 a title keeps the negatoria (check the action gate)")
    seen = Counter(tuple(sorted(topic_step.content_words(t[1]))) for t in topics)
    stems = [{w[:6] for w in topic_step.content_words(t[1])} for t in topics]
    for i in range(len(topics)):
        for j in range(i + 1, len(topics)):
            if stems[i] and stems[j] and (stems[i] <= stems[j] or stems[j] <= stems[i]):
                issues.append(f"R5b near-duplicate topics: {topics[i][0]} «{topics[i][1]}» / {topics[j][0]} «{topics[j][1]}»")
    # L1
    if any(a.startswith("1062 ") for t in topics for a in t[2][:1]):
        if not any(q.startswith("Per la destinazione del padre di famiglia") for q in qs):
            issues.append("L1 1062 topic without the works-at-separation question")
        if not any(q.startswith("Con quale atto e in quale anno") for q in qs):
            issues.append("L1 1062 topic without the separation-act question")
    # L2 / L3 (the answer's §2)
    norme = re.search(r"\*\*2\. Norme applicate ai fatti\*\*(.*?)(?=\n\*\*\d+\. |\Z)", ans, re.S)
    if norme:
        body = norme.group(1)
        if not re.search(r"requisito «.+?»: (?:provato|da provare|non risulta)\.", body):
            issues.append("L2 §2 has no code-printed status lines")
        heads = re.findall(r"(?m)^\*\*Art\. (\d+(?:-[a-z]+)?) ([^*—]+?)(?: — [^*]+)?\*\*\s*$", body)
        dup = [h for h, k in Counter(heads).items() if k > 1]
        if dup:
            issues.append(f"L3 §2 repeats a heading: {dup}")
        if "1061" in body and not any(h[0] == "1061" for h in heads):
            issues.append("L3 §2 discusses 1061 without its own heading")
    # L4
    if re.search(r"Replica da sviluppare", ans):
        issues.append("L4 a placeholder reply reached the answer")
    if any(q.startswith("Per la destinazione del padre di famiglia") for q in qs) and \
            "le opere non erano visibili al momento della separazione" not in ans and ans:
        issues.append("L4 the works question is open but there is no works objection")
    if re.search(r"(?m)^\*\*\d+\. La controparte potrà obiettare", ans):
        issues.append("L4b objection titles numbered like section headings")
    # L5
    full = re.search(r"\*\*Parti da convenire\*\*\s*\n\s*\n(.+)", ans)
    line2 = re.search(r"Parti da convenire: ([^.]+(?:\([^)]*\))?[^.]*)\.", r2)
    if full and line2 and len(line2.group(1)) + 10 < len(re.sub(r"^Parti da convenire(?: in giudizio)?:\s*", "",
                                                                full.group(1).strip())):
        issues.append(f"L5 round 2's parties «{line2.group(1)[:60]}» shorter than the procedural section's line")
    # L6
    if re.search(r"\bnel inizio del|\bil inizio del", r1, re.I):
        issues.append("L6 an «inizio del» event not in the copy's form")
    if "condizione di procedibilità" in ans:
        issues.append("L6 mediation says «condizione di procedibilità», not «improcedibilità»")
    if others_of(r1) and not any(o.startswith("Capo ") for o in others_of(r1)):
        issues.append("L6 altri profili labelled by title, not by Capo")
    extras = dl.extras_of(ans, expected, ev) if ans else []
    return {"issues": issues, "topics": topics, "questions": qs, "others": others_of(r1), "extras": extras}


def load_facts_run(run: str) -> dict:
    out = {}
    cases = os.path.join(run, "cases")
    for name in os.listdir(cases):
        if name.endswith(".json"):
            try:
                r = json.load(open(os.path.join(cases, name), encoding="utf-8"))
            except ValueError:
                continue
            if isinstance(r, dict) and r.get("kind") in ("eval", None):
                out[r["id"]] = r
    return out


def capo_of(units: dict, label: str) -> str:
    return units.get(label, "")


def _ranges_hold(number: str, ranges) -> bool:
    m = re.match(r"\d+", number)
    return bool(m) and any(lo <= int(m.group(0)) <= hi for lo, hi in ranges or [])


def missing_report(cid: str, case: dict, rec: dict, expected: list, ev, dl, rubriche: dict, units: dict) -> list:
    """The expected norms outside every topic, and what would have anchored them."""
    from src.rag import fact_signals, interactive_flow, topic_step
    ekey = dl._ekey_fn(ev)
    topics = topics_of(case["round1"])
    inside = {ekey(a) for t in topics for a in t[2] if " " in a}
    rules = fact_signals.load_rules()
    rows = []
    institutions = [(i.get("nome"), [a.get("label") for a in i.get("articles") or []], i.get("method"))
                    for i in (rec or {}).get("institutions") or [] if isinstance(i, dict)]
    for item in expected:
        alts = [str(a).lower() for a in item.get("any_of") or [item["article"]]]
        if any((a, item["code"]) in inside for a in alts):
            continue
        a0 = alts[0]
        code = dl._EVAL_CODES.get(item["code"], item["code"])
        label = f"{a0} {code}"
        resolved = [f"{n} [{m}]" for n, arts, m in institutions if any(ekey(x) == (a0, item["code"]) for x in arts if x)]
        fired = {f.get("id") for f in ((rec or {}).get("fact_signals") or {}).get("fired") or []}
        rule_hits = [r["id"] + (" (fired)" if r["id"] in fired else " (not fired)") for r in rules
                     if (r.get("codice") or "") == code and _ranges_hold(a0, r.get("ranges"))]
        regime_hits = [g["id"] for g in fact_signals.REGIME_CANDIDATES
                       if any(str(n) == a0 and topic_step._code(c) == topic_step._code(code) for n, c in g.get("refs") or [])]
        rows.append({"article": label, "rubrica": rubriche.get((a0, code)) or "", "capo": units.get(label, ""),
                     "resolved_by": resolved, "rules": rule_hits, "regimes": regime_hits,
                     "areas": ((rec or {}).get("frame") or {}).get("areas") or []})
    return rows


_DAMAGE_KINDS = ("reato", "lesione", "incidente", "danneggiamento", "decesso")
_ACT_KINDS = ("provvedimento", "notifica")
_FACT_DATE_TERMS = ("querela", "prescrizione_extracontrattuale", "prescrizione_circolazione")
_NOTICE_TERMS = ("prescrizione_ordinaria", "denuncia_vizi_vendita", "denuncia_difformita_appalto",
                 "denuncia_conformita_consumo", "denuncia_gravi_difetti")
_ACT_TERMS = ("ricorso_tar", "ricorso_tar_appalti")


def question_report(rec: dict) -> dict:
    """The question families the extracted facts trigger, each with its trigger (the flow asks one only for a topic it
    serves, so this is what the facts would allow): possesso_inizio (a client-side possession with no start),
    opera/atto_separazione (a work, or a split owner, in a real-rights case), accesso_alternativo (a right of way, or an
    access whose interclusion is unknown), data_fatto (an open prescription/querela term, or an undated offence or
    damage), data_messa_in_mora (an open contract term, or an undated notice in a contract case), atto_notifica (v55: an
    open TAR term -- the flow asks it then -- or, as a family it would need, an undated act with no open term)."""
    from src.rag import fact_signals, interactive_flow as flow
    signals = (rec or {}).get("fact_signals") or {}
    facts = signals.get("facts") or {}
    areas = ((rec or {}).get("frame") or {}).get("areas") or []
    by_id = {t.id: t for t in fact_signals.TERMS}
    open_terms = [(t.get("id"), t.get("status"), getattr(by_id.get(t.get("id")), "since", "") or "")
                  for t in signals.get("terms") or [] if t.get("status") in ("DA VERIFICARE", "INCERTO", "NON VERIFICABILE")]
    open_ids = {t for t, _, _ in open_terms}
    events = facts.get("events") or []
    undated = [e for e in events if not str(e.get("date") or "").strip()]
    found = {}

    def asks(family):
        try:
            return flow._family_questions(family, facts, signals, "")
        except Exception:
            return []
    if asks("possesso_inizio"):
        found["possesso_inizio"] = "a client-side possession with no start"
    if "diritti_reali" in areas:
        if (facts.get("works_or_signs") or facts.get("split_into_parts")) and asks("opera_separazione"):
            found["opera_separazione"] = "a work in a real-rights case, its existence at the separation unknown"
        if fact_signals._the_split(facts) and asks("atto_separazione"):
            found["atto_separazione"] = "a split owner, the act not stated"
    if fact_signals._right_of_way(facts) or any(not str(a.get("landlocked") or "").strip() for a in facts.get("access") or []):
        found["accesso_alternativo"] = "a right of way with the interclusion unknown"
    damage = [e for e in undated if (e.get("kind") or "") in _DAMAGE_KINDS]
    if open_ids & set(_FACT_DATE_TERMS) or damage:
        found["data_fatto"] = ("open term " + ", ".join(sorted(open_ids & set(_FACT_DATE_TERMS))) if open_ids & set(_FACT_DATE_TERMS)
                               else f"{len(damage)} undated offence/damage event(s): «{damage[0].get('what', '')[:50]}»")
    notice = [e for e in undated if (e.get("kind") or "") in ("diffida", "denuncia/reclamo")]
    if open_ids & set(_NOTICE_TERMS) or (facts.get("contracts") and notice):
        found["data_messa_in_mora"] = "open contract term" if open_ids & set(_NOTICE_TERMS) else "an undated notice"
    missing = {}
    acts = [e for e in undated if (e.get("kind") or "") in _ACT_KINDS]
    if open_ids & set(_ACT_TERMS) and asks("atto_notifica"):
        found["atto_notifica"] = "open term " + ", ".join(sorted(open_ids & set(_ACT_TERMS)))
    elif "amministrativo" in areas and acts:
        missing["atto_notifica (no open term)"] = f"undated act «{acts[0].get('what', '')[:50]}»"
    return {"families": found, "missing": missing, "open_terms": open_terms, "areas": areas,
            "undated": len(undated), "events": len(events)}


# what each predicate reads in the extracted facts, and the words the case text would state it with
_READS = {"fact": ("events", "works_or_signs", "injuries", "contracts", "proceedings", "property_history", "possession",
                   "money", "client_side"), "request": ("requests",), "role": ("parties",), "role_count": ("parties",),
          "age_under": ("parties",), "contract": ("contracts",), "proceeding": ("proceedings",), "injury": ("injuries",),
          "event": ("events",), "term": ("terms",), "work": ("works_or_signs",), "possession": ("possession",),
          "landlocked": ("access",), "common_previous_owner": ("property_history",), "money_over_threshold": ("money",)}
_AGE_WORDS = r"\b\d{1,2} anni\b|minor|bambin|ragazz|alunn|studente|scolar"


def _text_pattern(name: str, arg) -> str:
    """The pattern the case text would state a condition with ("" when the predicate has none)."""
    from src.rag import fact_signals
    if isinstance(arg, dict) and (arg.get("pattern") or arg.get("type")):
        return arg.get("pattern") or arg.get("type")
    if name in ("role", "role_count"):
        roles = arg if name == "role" else arg["roles"]
        return "|".join(re.escape(r[:6]) for r in roles)
    if name == "age_under":
        return _AGE_WORDS
    if name == "injury":
        return fact_signals._DAMAGE
    return ""


def _condition_report(condition: dict, facts: dict, terms: list, case_text: str) -> list:
    """[(text, matched, in the extraction, in the case text)] for a condition (an "any" or "not": each part)."""
    from src.rag import fact_signals
    (name, arg), = condition.items()
    if name == "any":
        return [row for c in arg for row in _condition_report(c, facts, terms, case_text)]
    if name == "not":
        return [(f"NON ({t})", not m, x, y) for t, m, x, y in _condition_report(arg, facts, terms, case_text)]
    matched = fact_signals.holds(condition, facts, terms)
    keys = _READS.get(name, ())
    if name == "term":
        held = "; ".join(f"{t.get('id')} {t.get('status')}" for t in terms) or "no term computed"
    else:
        filled = [k for k in keys if facts.get(k)]
        held = (fact_signals._evidence(condition, facts, terms) or ("fields " + ", ".join(filled) + " filled, none "
                                                                    "matching" if filled else "")) or \
            f"empty: {', '.join(keys)}"
    pattern = _text_pattern(name, arg)
    stated = ""
    if pattern:
        m = re.search(pattern, (case_text or "").lower())
        stated = f"yes: «…{case_text[max(0, m.start() - 40):m.end() + 40].strip()}…»" if m else "no"
    return [(fact_signals.describe([condition]), matched, held, stated or "-")]


def rules_report(recs: dict, targets: dict, dl) -> list:
    """Per case and rule of targets ({case: [rule ids]}): each condition, whether it held on the run's extracted facts,
    what the extraction holds for it, and whether the case text states it -- a condition that fails on an empty field
    the text does state is the extraction's miss; one the text does not state is a fact to ask."""
    from src.rag import fact_signals
    rules = {r["id"]: r for r in fact_signals.load_rules()}
    L = []
    for cid, ids in targets.items():
        rec = recs.get(cid) or {}
        signals = rec.get("fact_signals") or {}
        facts, terms = signals.get("facts") or {}, signals.get("terms") or []
        areas = (rec.get("frame") or {}).get("areas") or []
        text = str((dl._eval_case(cid) or {}).get("question") or "")
        L += [f"### {cid} (areas {', '.join(areas) or '-'})", ""]
        if not facts:
            L += ["No extracted facts in the run's record.", ""]
            continue
        explained = {r["id"]: r for r in fact_signals.explain_rules(facts, terms, areas=areas)}
        for rid in ids:
            rule, ex = rules.get(rid), explained.get(rid) or {}
            if not rule:
                L += [f"- **{rid}**: no such rule", ""]
                continue
            L += [f"**{rid}** — {'fired' if ex.get('fired') else 'not fired'}"
                  + (f" ({ex['skipped']})" if ex.get("skipped") else ""), "",
                  "| condition | held | the extraction | the case text states it |", "|---|---|---|---|"]
            for condition in rule["when"]:
                for t, m, held, stated in _condition_report(condition, facts, terms, text):
                    L.append(f"| {t} | {'yes' if m else '**no**'} | {held} | {stated} |".replace("\n", " "))
            L.append("")
    return L


def lint_main(argv: list) -> None:
    """--lint <files>: the output lint (answer_lint) of transcripts, one finding per line (file, case, section, rule,
    excerpt), then the per-rule and per-file tables. --render-from <files>: the same transcripts re-rendered first
    (render_replay: the stored writer output through the current renderer and post-checks), then linted."""
    import answer_lint
    parser = argparse.ArgumentParser(prog="interactive_review.py --lint")
    parser.add_argument("--lint", nargs="*", default=None)
    parser.add_argument("--render-from", nargs="*", default=None)
    parser.add_argument("--out", help="write the re-rendered transcripts into this folder (--render-from)")
    parser.add_argument("--quiet", action="store_true", help="the tables only")
    args = parser.parse_args(argv)
    files = []
    for f in (args.render_from if args.render_from is not None else args.lint) or []:
        if os.path.isdir(f):
            files += sorted(os.path.join(d, x) for d, _, xs in os.walk(f) for x in xs if x.endswith(".md"))
        else:
            files.append(f)
    findings = []
    for path in files:
        if args.render_from is not None:
            import render_replay
            cases = render_replay.replay_file(path)
            if args.out:
                os.makedirs(args.out, exist_ok=True)
                name = answer_lint.file_label(path)
                target = os.path.join(args.out, name + (".md" if not name.endswith(".md") else ""))
                render_replay.write_file(path, cases, target)
            findings += answer_lint.lint_file(path, cases)
        else:
            findings += answer_lint.lint_file(path)
    if not args.quiet:
        for f in findings:
            print("\t".join(f))
        print()
    print(answer_lint.tables(findings, files))


def main() -> None:
    if any(a in ("--lint", "--render-from") for a in sys.argv[1:]):
        return lint_main(sys.argv[1:])
    import phase1_offline as p1
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("report")
    parser.add_argument("--facts", required=True, help="a single-shot run of the same cases (v52-candidate)")
    parser.add_argument("--links", help="article_links' output folder (default: <facts>/../article_links_out)")
    parser.add_argument("--out")
    parser.add_argument("--rules", default="c1:contatto_sociale_scuola,responsabilita_sanitaria_civile;"
                        "c3:risoluzione_contratto;h1:contatto_sociale_scuola,responsabilita_sanitaria_civile;"
                        "n7:difformita_appalto,rovina_gravi_difetti",
                        help="section 4, why these rules did not fire: case:rule,rule;case:rule")
    args = parser.parse_args()
    dl = p1._dl()
    ev = dl._eval_module()
    facts_run = os.path.abspath(os.path.expanduser(args.facts))
    reports = os.path.dirname(facts_run)
    rubriche = p1.rubrica_table(reports)
    links_dir = args.links or os.path.join(reports, "article_links_out")
    units = {}
    if os.path.isfile(os.path.join(links_dir, "units.csv")):
        for r in csv.DictReader(open(os.path.join(links_dir, "units.csv"), encoding="utf-8")):
            if r["level"].upper() == "CAPO":
                units.setdefault(f"{r['article']} {r['code']}", r["unit name"])
    cases = parse_report(os.path.expanduser(args.report))
    recs = load_facts_run(facts_run)
    from src.rag import interactive_flow
    L = [f"# Review of {args.report}", "", "## 1. Checks per case", ""]
    tally = Counter()
    per_case = {}
    for cid, case in cases.items():
        expected = (dl._eval_case(cid) or {}).get("expected") or (recs.get(cid) or {}).get("expected") or []
        res = check_case(cid, case, rubriche, expected, ev, dl)
        per_case[cid] = res
        for i in res["issues"]:
            tally[i.split(" ", 1)[0]] += 1
    L += ["| check | cases |", "|---|---|"] + [f"| {k} | {v} |" for k, v in sorted(tally.items())] + [""]
    for cid, res in per_case.items():
        L.append(f"- **{cid}** ({len(res['topics'])} topics, {len(res['questions'])} questions, "
                 f"{len(res['extras'])} extras): " + ("; ".join(res["issues"]) or "no issue"))
    L += ["", "## 2. Topic recall under 50%: the expected norms outside every topic", ""]
    by_area = defaultdict(list)
    for cid, case in cases.items():
        cells = case["table"] or []
        rec_cell = cells[8] if len(cells) > 8 else ""
        m = re.match(r"(\d+)/(\d+)", rec_cell)
        if not m or int(m.group(2)) == 0 or int(m.group(1)) * 2 >= int(m.group(2)):
            continue
        expected = (dl._eval_case(cid) or {}).get("expected") or []
        rows = missing_report(cid, case, recs.get(cid), expected, ev, dl, rubriche, units)
        areas = (rows[0]["areas"] if rows else ((recs.get(cid) or {}).get("frame") or {}).get("areas")) or ["?"]
        L += [f"### {cid} — recall {rec_cell}; areas {', '.join(areas)}", "",
              "| expected | rubrica | Capo | resolved in the facts run by | rules / regimes holding it |", "|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['article']} | {r['rubrica'] or '-'} | {r['capo'] or '-'} | {', '.join(r['resolved_by']) or '-'} | "
                     f"{', '.join(r['rules'] + r['regimes']) or 'NONE'} |")
            by_area[areas[0]].append((cid, r))
        L.append("")
    L += ["### By the case's first area: norms with no rule or regime holding them", ""]
    for area, items in sorted(by_area.items()):
        bare = [(c, r) for c, r in items if not (r["rules"] or r["regimes"])]
        unfired = [(c, r) for c, r in items if r["rules"] and not any(x.endswith("(fired)") for x in r["rules"])
                   and not r["regimes"]]
        L.append(f"- **{area}**: no table holds {len(bare)} of {len(items)} — " + ("; ".join(
            f"{r['article']} «{r['rubrica'][:50]}» ({c})" for c, r in bare) or "-"))
        if unfired:
            L.append(f"  - a rule holds it but did not fire: " + "; ".join(
                f"{r['article']} [{', '.join(r['rules'])}] ({c})" for c, r in unfired))
    L += ["", "## 3. Cases with no question: the families that would apply", ""]
    L += ["| case | areas | families with a template the facts trigger | families with no template yet | open terms "
          "(runs from) | undated events |", "|---|---|---|---|---|---|"]
    for cid, res in per_case.items():
        if res["questions"]:
            continue
        q = question_report(recs.get(cid))
        L.append(f"| {cid} | {', '.join(q['areas'])} | "
                 + ("; ".join(f"{k}: {v}" for k, v in q["families"].items()) or "none") + " | "
                 + ("; ".join(f"{k}: {v}" for k, v in q["missing"].items()) or "-") + " | "
                 + ("; ".join(f"{i} {s} ({since or '?'})" for i, s, since in q["open_terms"]) or "-") + " | "
                 + f"{q['undated']} of {q['events']} |")
    targets = {c: r.split(",") for c, _, r in (x.partition(":") for x in args.rules.split(";") if x)}
    L += ["", "## 4. Why these rules did not fire (the facts run's extracted facts)", ""] + rules_report(recs, targets, dl)
    out = args.out or os.path.join(os.path.dirname(os.path.expanduser(args.report)), "interactive_review.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
