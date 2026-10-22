"""Read-only: do the ruling cards pick better rulings than the live search, on the criminal cases?

Step 3 of the ruling-cards plan (22 Oct 2026). For each criminal eval case (expected
articles in c.p. / c.p.p. only; tuning cases, plus h5-h9 with --include-heldout):

  BEFORE  the rulings the live pipeline gives the writer: the retrieval part of
          run_case_analysis_pipeline, copied here (analysis, article selection, the
          second ruling search on the selected articles, _ruling_passages). There
          are no draft calls; it uses the same flags as the app's .env.
  AFTER   the cards (exp_ruling_cards.py) whose applied articles meet the articles
          the same run selected. They are ranked by the fixed weights below, set
          before any result was seen: 3 per shared article, 1 per fact or issue
          keyword found in the case, 2 if the ruling favours the client's side. The
          side comes from the analysis' "posizione"; the top 5 are taken, as many
          as the writer reads (_PASSAGE_RULINGS).

Both are scored on the same yardsticks, none of which the AFTER ranking sees:

  - criminal: the ruling is a criminal one (step 1's branch). AFTER is criminal by
    construction, since only criminal rulings were carded;
  - on topic: the ruling cites at least one of the case's EXPECTED articles (the
    eval's answer key, from step 1's citations);
  - full reference: court, number and date (+ city for lower courts), from the card,
    else from step 1's fields;
  - client's side: the card says it favours the client (unknown for rulings
    without a card).

Run from the app folder (prod), with exp_ruling_links.py next to this script, after
the cards run has finished (it calls the model for the analysis and the article
choice, about 30-60 s per case):
    cd /opt/chatbot && venv/bin/python /tmp/exp_ruling_search.py
    ... --include-heldout      ... --only p1,p2

Nothing is written to the database. Only counts are printed.
"""

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("exp_ruling_links", os.path.join(_HERE, "exp_ruling_links.py"))
rl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rl)   # also makes every Neo4j session read-only, and loads the app

import src.rag.defensive_generation as dg  # noqa: E402

TOP = dg._PASSAGE_RULINGS
CODE_ABBR = {"cp": "c.p.", "cpp": "c.p.p."}
W_ARTICLE, W_KEYWORD, W_SIDE = 3, 1, 2
_DEFENDANT_RE = re.compile(r"imputat|indagat|accusat|condannat|difes|arrestat|autore\s+del", re.IGNORECASE)
_VICTIM_RE = re.compile(r"persona\s+offesa|parte\s+civile|vittim|danneggiat|denunciant|querelant|famil", re.IGNORECASE)


# --- Pure helpers (tests/test_exp_ruling_search.py) ------------------------------------

def criminal_cases(cases: list, include_heldout: bool, only=None) -> list:
    """The cases whose expected articles are all c.p. / c.p.p.; held-out ones only on request."""
    out = []
    for c in cases:
        codes = {e.get("code") for e in c.get("expected") or []}
        if not codes or not codes <= set(CODE_ABBR):
            continue
        if c.get("heldout") and not include_heldout:
            continue
        if only and c["id"] not in only:
            continue
        out.append(c)
    return out


def expected_keys(case: dict) -> set:
    return {f"{rl._article_num(str(e['article']))} {CODE_ABBR[e['code']]}" for e in case["expected"]}


def client_side(posizione: str) -> str:
    """'imputato' / 'accusa' from the analysis' position of the client, '' when unclear."""
    text = posizione or ""
    defendant, victim = bool(_DEFENDANT_RE.search(text)), bool(_VICTIM_RE.search(text))
    return "imputato" if defendant and not victim else "accusa" if victim and not defendant else ""


def _hit(keyword: str, low: str) -> bool:
    stems = [w[:5] for w in re.findall(r"\w{4,}", keyword.lower())]
    return bool(stems) and all(s in low for s in stems)


def rank_cards(cards: list, selected: set, case_terms: str, side: str, top: int = TOP) -> list:
    """The ids of the `top` cards sharing an applied article with `selected`, by the fixed
    weights; ties: full reference first, then id (stable)."""
    low = " ".join(case_terms.lower().split())
    scored = []
    for c in cards:
        shared = len(set(c.get("articoli_applicati") or []) & selected)
        if not shared:
            continue
        keywords = (c.get("parole_chiave_fatti") or []) + (c.get("parole_chiave_questioni") or []) \
            + (c.get("argomento_decisivo") or [])
        score = W_ARTICLE * shared + W_KEYWORD * sum(_hit(k, low) for k in keywords) \
            + (W_SIDE if side and c.get("a_favore_di") == side else 0)
        scored.append((-score, not c.get("riferimento_completo"), c["id"]))
    return [doc_id for _, _, doc_id in sorted(scored)[:top]]


def measure(ids: list, links: dict, cards: dict, expected: set, side: str) -> dict:
    """The yardsticks over a list of ruling ids (see the module docstring)."""
    m = {"rulings": len(ids), "criminal": 0, "on_topic": 0, "full_reference": 0, "client_side": 0,
         "side_known": 0, "not_a_ruling": 0}
    for doc_id in ids:
        link, card = links.get(doc_id), cards.get(doc_id)
        if link is None:              # commentary (Codice Penale Commentato) or another document type
            m["not_a_ruling"] += 1
            continue
        m["criminal"] += link["branch"] == "penale"
        m["on_topic"] += bool(set(link["articles"]) & expected)
        m["full_reference"] += bool(card["riferimento_completo"] if card else rl.full_reference(link))
        if card and side:
            m["side_known"] += card["a_favore_di"] != "non chiaro"
            m["client_side"] += card["a_favore_di"] == side
    return m


def reference(link: dict, card: dict) -> str:
    """The card's reference, else one built from the Italgiure file name, else the id."""
    if card and card.get("riferimento"):
        return card["riferimento"]
    italgiure = (link or {}).get("italgiure")
    if italgiure:
        court = "Cass. pen." if italgiure["kind"] == "snpen" else "Cass. civ."
        return f"{court}, Sez. {italgiure['section']}, n. {italgiure['number']}/{italgiure['year']}"
    return "(not a ruling: commentary or other document)" if link is None else link["id"]


def show_lines(label: str, ids: list, links: dict, cards: dict, expected: set) -> list:
    """One line per ruling: reference, the side its card says it favours, whether it cites /
    applies an expected article. No text."""
    out = [f"   {label}:"]
    for doc_id in ids:
        link, card = links.get(doc_id), cards.get(doc_id)
        cites = bool(link and set(link["articles"]) & expected)
        applies = sorted(set((card or {}).get("articoli_applicati") or []) & expected)
        side = (card or {}).get("a_favore_di") or "-"
        out.append(f"      {reference(link, card):58s} favours: {side:10s} cites expected: {'yes' if cites else 'no ':3s}"
                   + (f"  applies: {', '.join(applies)}" if applies else ""))
    if not ids:
        out.append("      (none)")
    return out


def _cell(m: dict, key: str) -> str:
    return f"{m[key]}/{m['rulings']}"


def report_lines(rows: list) -> list:
    keys = ("criminal", "on_topic", "full_reference", "client_side")
    out = [f"{'case':18s} {'side':9s} | {'BEFORE: criminal / on topic / full ref / client side':52s} | AFTER",
           "-" * 120]
    totals = {"before": dict.fromkeys(keys + ("rulings",), 0), "after": dict.fromkeys(keys + ("rulings",), 0)}
    for case_id, side, before, after in rows:
        out.append(f"{case_id:18s} {side or '?':9s} | "
                   + "   ".join(f"{_cell(before, k):>5s}" for k in keys) + f"   (+{before['not_a_ruling']} commentary)"
                   + " " * 6 + "| " + "   ".join(f"{_cell(after, k):>5s}" for k in keys))
        for name, m in (("before", before), ("after", after)):
            for k in keys + ("rulings",):
                totals[name][k] += m[k]
    out.append("-" * 120)
    for name in ("before", "after"):
        t = totals[name]
        out.append(f"{name.upper():6s} total rulings {t['rulings']}: "
                   + ", ".join(f"{k.replace('_', ' ')} {rl._pct(t[k], t['rulings'])}" for k in keys))
    return out


# --- The live part ------------------------------------------------------------------------

def _default_cases_path() -> Path:
    for candidate in (Path(rl.ROOT) / "tests" / "eval_narrative_cases.json", Path("/root/eval/eval_narrative_cases.json")):
        if candidate.exists():
            return candidate
    return Path("/root/eval/eval_narrative_cases.json")


def live_ruling_ids(session, driver, database: str, question: str):
    """The retrieval part of run_case_analysis_pipeline, copied: (analysis, selected
    article keys, the ids of the rulings the writer reads)."""
    analysis = dg._analyse_case_for_pipeline(question, "it", "")
    v2 = dg.feature_flags.enabled(dg.CASE_ANALYSIS_FLAG)
    fusion = v2 and dg.feature_flags.enabled(dg._CASE_POOL_FUSION_FLAG)
    select_fn = dg._select_articles_fusion if fusion else (dg._select_articles_v2 if v2 else dg._select_articles)
    picked = select_fn(session, driver, database, question, "", analysis)
    picked = dg._add_method_candidates(session, driver, database, analysis, picked)
    selected = picked["selected"]
    second = dg._search_case_law(session, analysis, numbers=[n for n, _ in selected[:4]],
                                 exclude_docs={(r.get("d") or {}).get("id") for r in picked["similar"]})
    rows = dg._ruling_passages(session, dg._reading_order(picked["similar"], second, picked["first_rulings"]),
                               selected, analysis)
    keys = {f"{rl._article_num(str(n))} {code}" for n, code in selected}
    return analysis, keys, [(r.get("d") or {}).get("id") for r in rows]


def _case_terms(question: str, analysis: dict) -> str:
    parts = [question] + list(analysis.get("ricerca_fatti") or [])
    for q in analysis.get("questioni") or []:
        parts += [q.get("titolo") or ""] + [w for w in (q.get("parole_chiave") or []) if isinstance(w, str)]
    return " ".join(str(p) for p in parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cases", default=str(_default_cases_path()))
    parser.add_argument("--cards", default="/tmp/ruling_cards_v3.jsonl")
    parser.add_argument("--links", default="/tmp/ruling_links.jsonl")
    parser.add_argument("--include-heldout", action="store_true", help="add h5-h9 (measure only)")
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--show", action="store_true", help="list each case's rulings, before and after (no text)")
    args = parser.parse_args()

    from src.rag.main import NEO4J_DATABASE, driver

    links = {r["id"]: r for r in (json.loads(line) for line in open(args.links, encoding="utf-8"))}
    cards = {}
    for line in open(args.cards, encoding="utf-8"):
        card = json.loads(line)
        if not card.get("errore"):
            cards[card["id"]] = card
    cases = criminal_cases(json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"], args.include_heldout,
                           set(args.only.split(",")) if args.only else None)
    print(f"{len(cases)} criminal cases, {len(cards)} cards, {len(links)} rulings in step 1's labels", file=sys.stderr)

    rows, shown = [], []
    with driver.session(database=NEO4J_DATABASE) as session:
        for case in cases:
            t0 = time.time()
            analysis, selected, before_ids = live_ruling_ids(session, driver, NEO4J_DATABASE, case["question"])
            side = client_side(analysis.get("posizione") or "")
            after_ids = rank_cards(list(cards.values()), selected, _case_terms(case["question"], analysis), side)
            expected = expected_keys(case)
            rows.append((case["id"], side, measure(before_ids, links, cards, expected, side),
                         measure(after_ids, links, cards, expected, side)))
            if args.show:
                shown += [f"## {case['id']}  (client: {side or 'unclear'}; expected: {', '.join(sorted(expected))}; "
                          f"selected by the pipeline: {', '.join(sorted(selected)[:12])})"]
                shown += show_lines("BEFORE (today)", before_ids, links, cards, expected)
                shown += show_lines("AFTER (cards)", after_ids, links, cards, expected) + [""]
            print(f"   {case['id']}: {len(before_ids)} before, {len(after_ids)} after, "
                  f"{len(selected)} articles selected, {time.time() - t0:.0f}s", file=sys.stderr)

    if shown:
        print("\n".join(shown))
    print("\n".join(report_lines(rows)))
    print("\nDone. Nothing was written to the database.")


if __name__ == "__main__":
    main()
