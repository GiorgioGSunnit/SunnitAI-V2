"""Read-only: do the ruling cards pick better rulings than the live search, on the criminal cases?

Step 3 of the ruling-cards plan (22 Oct 2026). For each criminal eval case (expected
articles in c.p. / c.p.p. only; tuning cases, plus h5-h9 with --include-heldout):

  BEFORE  the rulings the live pipeline gives the writer: the retrieval part of
          run_case_analysis_pipeline, copied here (analysis, article selection, the
          second ruling search on the selected articles, _ruling_passages). There
          are no draft calls; it uses the same flags as the app's .env.
  AFTER   the cards (exp_ruling_cards.py) found by two routes and ranked together
          (see rank_cards): the article route (cards applying one of the case's
          CENTRAL articles) and the issue route (cards whose phrases contain one of
          the case's central issues by name: "tentata rapina", "colpa medica"). The
          top 5 are taken, as many as the writer reads (_PASSAGE_RULINGS).
          The issue route is opt-in (--issue-route): on the tuning cases (23 Oct,
          same cached pipeline outputs) it scored 18/50 on topic vs 20/50 by
          articles alone. It depends on the same analysis, which, when it names the
          wrong crime (fraud read as "estorsione, falso"), misleads both routes,
          and generic issue names ("nesso causale", "dolo") pull in other crimes.

The case's central articles, from the same run: the ones its text names and the ones
the analysis' legal issues resolved to ("omicidio preterintenzionale" -> 584 c.p.) weigh
1; the model's chosen pool 0.5. The generic method candidates (81, 110, 133 c.p. ...)
and the rest of the 46-80-article pool are not used. The first run (23 Oct) used that
whole pool at 3 points per shared article: sentencing rulings won everywhere, the same
one in all 10 cases. An article also weighs by how rare it is among the cards (art. 584
outweighs art. 81, applied in 268 of 2,476), and so does a keyword (motivazione, in 506
cards, almost nothing).

Both lists are scored on the same yardsticks, none of which the AFTER ranking sees:

  - criminal: the ruling is a criminal one (step 1's branch). AFTER is criminal by
    construction, since only criminal rulings were carded;
  - on topic: the ruling cites at least one of the case's EXPECTED articles (the eval's
    answer key, from step 1's citations). "available": how many carded rulings cite
    one at all, so a ranking miss shows apart from a data gap;
  - full reference: court, number and date, from the card, else from step 1's fields;
  - client's side: the card says it favours the client (unknown for rulings without a
    card).

The pipeline part takes about 75 s per case and calls the model, so its outputs are
saved to --cache; --use-cache ranks again from them in seconds, with no model call.
Tune on the tuning cases; run --include-heldout once, after, to confirm.

Run from the app folder (prod), with exp_ruling_links.py next to this script:
    cd /opt/chatbot && venv/bin/python /tmp/exp_ruling_search.py --show
    ... --use-cache --show       ... --include-heldout      ... --only p1,p2

Nothing is written to the database. Only references, labels and counts are printed.
"""

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("exp_ruling_links", os.path.join(_HERE, "exp_ruling_links.py"))
rl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rl)   # loads the app; main() makes every Neo4j session read-only

import src.rag.defensive_generation as dg  # noqa: E402

TOP = dg._PASSAGE_RULINGS
CODE_ABBR = {"cp": "c.p.", "cpp": "c.p.p.", "cc": "c.c.", "cpc": "c.p.c."}
# A case belongs to a branch when all its expected articles are of that branch's codes.
BRANCH_CODES = {"penale": {"cp", "cpp"}, "civile": {"cc", "cpc"}}
W_ARTICLE, W_KEYWORD, W_SIDE = 3, 1, 2
CENTRAL, CHOSEN = 1.0, 0.5
# A central article the ruling cites (step 1) but its card does not list as applied: still a
# sign the ruling is about it (q_long, 23 Oct: 5 rulings cite 584, one card applies it).
CITED_ONLY = 0.3
# The issue route: the case's central issues by NAME ("tentata rapina", "colpa medica") against
# the cards' own phrases. The held-out check (23 Oct) found the analysis names the issues well
# but maps them to wrong article numbers ("tentata rapina aggravata" -> 612, "concorso" -> 81),
# so the article route alone missed rulings the database has (63-127 per case).
W_ISSUE = 3
ISSUE_FRAME, ISSUE_INSTITUTION, ISSUE_QUESTION = 1.0, 0.7, 0.5
_ISSUE_STOP = {"reato", "reati", "delitto", "delitti", "penale", "penali", "civile", "civili", "articolo",
               "codice", "della", "dello", "delle", "degli", "nella", "nelle", "sulla", "sulle", "questione",
               "configurabilità", "responsabilità", "qualificazione", "giuridica"}
_DEFENDANT_RE = re.compile(r"imputat|indagat|accusat|condannat|difes|arrestat|autore\s+del", re.IGNORECASE)
_VICTIM_RE = re.compile(r"persona\s+offesa|parte\s+civile|vittim|danneggiat|denunciant|querelant|famil", re.IGNORECASE)


# --- Pure helpers (tests/test_exp_ruling_search.py) ------------------------------------

def criminal_cases(cases: list, include_heldout: bool, only=None) -> list:
    """The cases whose expected articles are all c.p. / c.p.p.; held-out ones only on request."""
    return branch_cases(cases, "penale", include_heldout, only)


def branch_cases(cases: list, branch: str, include_heldout: bool, only=None) -> list:
    """The cases whose expected articles are all of the branch's codes (BRANCH_CODES); held-out
    ones only on request."""
    out = []
    for c in cases:
        codes = {e.get("code") for e in c.get("expected") or []}
        if not codes or not codes <= BRANCH_CODES[branch]:
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


def ref_key(ref) -> str:
    """('584', 'c.p.') -> '584 c.p.' (a pipeline ref), a card's key unchanged."""
    if isinstance(ref, str):
        return ref
    number, code = ref[0], ref[1]
    return f"{rl._article_num(str(number))} {code}"


def central_weights(picked: dict) -> dict:
    """The case's central articles and their weight, from the pipeline's own picks: named in
    the case text or resolved from its legal issues CENTRAL, the model's chosen pool CHOSEN.
    The method candidates (generic articles added by area and posture) are left out."""
    method = {ref_key(r) for r in picked.get("method_candidates") or []}
    weights = {}
    for ref in picked.get("chosen") or []:
        key = ref_key(ref)
        if key not in method:
            weights[key] = CHOSEN
    for ref in list(picked.get("named") or []) + list(picked.get("institutions_resolved") or []):
        weights[ref_key(ref)] = CENTRAL
    return weights


def rarity(counts: Counter, total: int) -> dict:
    """item -> log(total / count): an article or keyword on every card weighs ~0."""
    return {item: math.log(total / n) for item, n in counts.items() if n}


def card_rarities(cards: list) -> tuple:
    """(article rarity, keyword rarity scaled to 0..1) over the cards."""
    n = max(len(cards), 2)
    articles = rarity(Counter(a for c in cards for a in set(c.get("articoli_applicati") or [])), n)
    keywords = Counter(k for c in cards for k in set(_keywords(c)))
    scale = math.log(n)
    return articles, {k: v / scale for k, v in rarity(keywords, n).items()}


def _keywords(card: dict) -> list:
    return (card.get("parole_chiave_fatti") or []) + (card.get("parole_chiave_questioni") or []) \
        + (card.get("argomento_decisivo") or [])


def _hit(keyword: str, low: str) -> bool:
    stems = [w[:5] for w in re.findall(r"\w{4,}", keyword.lower())]
    return bool(stems) and all(s in low for s in stems)


def issue_stems(name: str) -> list:
    """The 5-letter stems of an issue name's content words ("Tentata rapina aggravata" ->
    tenta, rapin, aggra), without the generic ones (reato, penale, responsabilità ...)."""
    words = [w for w in re.findall(r"\w{4,}", (name or "").lower()) if w not in _ISSUE_STOP]
    return list(dict.fromkeys(w[:5] for w in words))


def _issue_text(card: dict) -> str:
    return " ".join(_keywords(card)).lower()


def issues_of(run: dict) -> list:
    """[(issue name, weight)] for a case: the cached "issues", else (older cache entries)
    its institutions' names."""
    if run.get("issues"):
        return [tuple(item) for item in run["issues"]]
    return [(i.get("nome") or "", ISSUE_INSTITUTION) for i in run.get("istituti") or [] if isinstance(i, dict)]


def issue_scores(cards: list, issues: list) -> Counter:
    """card id -> W_ISSUE x weight x rarity for every case issue whose stems are all in the
    card's phrases (an issue on more than half the cards says nothing and is skipped)."""
    n = max(len(cards), 2)
    texts = {c["id"]: _issue_text(c) for c in cards}
    out, seen = Counter(), set()
    for name, weight in issues:
        stems = issue_stems(name)
        if not stems or tuple(stems) in seen:
            continue
        seen.add(tuple(stems))
        hits = [cid for cid, text in texts.items() if all(s in text for s in stems)]
        if not hits or len(hits) > n / 2:
            continue
        rarity_of_issue = math.log(n / len(hits))
        for cid in hits:
            out[cid] += W_ISSUE * weight * rarity_of_issue
    return out


def score_card(card: dict, central: dict, low_terms: str, side: str, article_rarity: dict,
               keyword_rarity: dict, issue_score: float = 0.0) -> float:
    """W_ARTICLE x weight x rarity per central article the card applies (x CITED_ONLY when the
    ruling only cites it: card["_cited"], step 1's list), plus its issue route score, W_KEYWORD
    x rarity per keyword found in the case, W_SIDE if it favours the client. 0 when neither
    route finds it: the card is not about this case."""
    applied = set(card.get("articoli_applicati") or [])
    shared = [a for a in applied if a in central]
    cited_only = [a for a in set(card.get("_cited") or []) - applied if a in central]
    if not shared and not cited_only and not issue_score:
        return 0.0
    score = W_ARTICLE * sum(central[a] * article_rarity.get(a, 1.0) for a in shared) + issue_score
    score += W_ARTICLE * CITED_ONLY * sum(central[a] * article_rarity.get(a, 1.0) for a in cited_only)
    score += W_KEYWORD * sum(keyword_rarity.get(k, 1.0) for k in set(_keywords(card)) if _hit(k, low_terms))
    return score + (W_SIDE if side and card.get("a_favore_di") == side else 0)


def rank_cards(cards: list, central: dict, case_terms: str, side: str, top: int = TOP,
               rarities: tuple = None, issues: list = ()) -> list:
    """The ids of the `top` best-scoring cards (score_card, with the issue route for
    `issues`), ties broken by full reference first, then id (stable)."""
    article_rarity, keyword_rarity = rarities or card_rarities(cards)
    low = " ".join(case_terms.lower().split())
    by_issue = issue_scores(cards, issues) if issues else Counter()
    scored = []
    for c in cards:
        s = score_card(c, central, low, side, article_rarity, keyword_rarity, by_issue.get(c["id"], 0.0))
        if s > 0:
            scored.append((-s, not c.get("riferimento_completo"), c["id"]))
    return [doc_id for _, _, doc_id in sorted(scored)[:top]]


def available_on_topic(cards: dict, links: dict, expected: set) -> int:
    """How many carded rulings cite one of the expected articles: the most AFTER could find."""
    return sum(1 for doc_id in cards if set((links.get(doc_id) or {}).get("articles") or []) & expected)


def measure(ids: list, links: dict, cards: dict, expected: set, side: str, branch: str = "penale",
            party: str = "") -> dict:
    """The yardsticks over a list of ruling ids (see the module docstring). "criminal" counts the
    rulings of the case's branch; for a civil case the client's side is the card's favoured party
    against the client's role (src/rag/ruling_cards.favours_client)."""
    m = {"rulings": len(ids), "criminal": 0, "on_topic": 0, "full_reference": 0, "client_side": 0,
         "side_known": 0, "not_a_ruling": 0}
    for doc_id in ids:
        link, card = links.get(doc_id), cards.get(doc_id)
        if link is None:              # commentary (Codice Penale Commentato) or another document type
            m["not_a_ruling"] += 1
            continue
        m["criminal"] += link["branch"] == branch
        m["on_topic"] += bool(set(link["articles"]) & expected)
        m["full_reference"] += bool(card["riferimento_completo"] if card else rl.full_reference(link))
        if card and branch == "civile" and party.strip():
            from src.rag import ruling_cards as rc
            m["side_known"] += bool(card.get("parte_favorita"))
            m["client_side"] += rc.favours_client(rc._trim(card), "", party)
        elif card and side:
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


def report_lines(rows: list, branch_word: str = "criminal") -> list:
    """rows: (case id, side, before, after, available). branch_word labels the branch column."""
    keys = ("criminal", "on_topic", "full_reference", "client_side")
    out = [f"{'case':18s} {'side':9s} | {'BEFORE: ' + branch_word + ' / on topic / full ref / client side':52s} | "
           f"AFTER  (available: carded rulings citing an expected article)", "-" * 130]
    totals = {"before": dict.fromkeys(keys + ("rulings",), 0), "after": dict.fromkeys(keys + ("rulings",), 0)}
    for case_id, side, before, after, available in rows:
        out.append(f"{case_id:18s} {side or '?':9s} | "
                   + "   ".join(f"{_cell(before, k):>5s}" for k in keys) + f"   (+{before['not_a_ruling']} commentary)"
                   + " " * 6 + "| " + "   ".join(f"{_cell(after, k):>5s}" for k in keys) + f"   (available {available})")
        for name, m in (("before", before), ("after", after)):
            for k in keys + ("rulings",):
                totals[name][k] += m[k]
    out.append("-" * 130)
    for name in ("before", "after"):
        t = totals[name]
        out.append(f"{name.upper():6s} total rulings {t['rulings']}: "
                   + ", ".join(f"{(branch_word if k == 'criminal' else k.replace('_', ' '))} "
                               f"{rl._pct(t[k], t['rulings'])}" for k in keys))
    return out


# --- The live part ------------------------------------------------------------------------

def _default_cases_path() -> Path:
    for candidate in (Path(rl.ROOT) / "tests" / "eval_narrative_cases.json", Path("/root/eval/eval_narrative_cases.json")):
        if candidate.exists():
            return candidate
    return Path("/root/eval/eval_narrative_cases.json")


def live_ruling_ids(session, driver, database: str, question: str) -> dict:
    """The retrieval part of run_case_analysis_pipeline, copied: what the ranking and the
    report need from it (the analysis' terms and side, the selected and central articles,
    the ids of the rulings the writer reads)."""
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
    resolved = resolved_institutions(session, driver, database, analysis)
    central = central_weights({**picked, "institutions_resolved":
                               list(picked.get("institutions_resolved") or []) + resolved})
    istituti = [i for q in analysis.get("questioni") or [] for i in (q.get("istituti") or []) if isinstance(i, dict)]
    return {"terms": _case_terms(question, analysis), "side": client_side(analysis.get("posizione") or ""),
            "party": f"{analysis.get('parte_assistita') or ''} {analysis.get('posizione') or ''}".strip(),
            "selected": sorted({ref_key(r) for r in selected}), "central": central, "istituti": istituti,
            "issues": case_issues(analysis), "before_ids": [(r.get("d") or {}).get("id") for r in rows]}


def case_issues(analysis: dict) -> list:
    """[(name, weight)]: the frame's central issues, the analysis' institutions, its question titles."""
    issues = [(c.get("nome") or "", ISSUE_FRAME)
              for c in ((analysis.get("frame") or {}).get("central_issue_candidates") or []) if isinstance(c, dict)]
    for q in analysis.get("questioni") or []:
        issues += [(i.get("nome") or "", ISSUE_INSTITUTION) for i in q.get("istituti") or [] if isinstance(i, dict)]
        issues.append((q.get("titolo") or "", ISSUE_QUESTION))
    return [[name, weight] for name, weight in issues if name.strip()]


def resolved_institutions(session, driver, database: str, analysis: dict) -> list:
    """Every article the analysis' institutions resolve to, the same lookup as
    _select_articles_v2 (database only, no model call). Its own "institutions_resolved"
    keeps only the ones no earlier step had found: on 23 Oct p2's "Truffa" -> 640 c.p. was
    resolved but missing there, so 640 never counted as central."""
    from src.rag.doc_lookup import _fetch_doc_names
    documents = _fetch_doc_names(driver, database)
    found = []
    for q in analysis.get("questioni") or []:
        for istituto in q.get("istituti") or []:
            try:
                rows = dg._resolve_institution(session, documents, istituto, q.get("parole_chiave"))
            except Exception:
                continue
            found += [row["_article"] for row in rows if row.get("_article")]
    return list(dict.fromkeys(found))


def _case_terms(question: str, analysis: dict) -> str:
    parts = [question] + list(analysis.get("ricerca_fatti") or [])
    for q in analysis.get("questioni") or []:
        parts += [q.get("titolo") or ""] + [w for w in (q.get("parole_chiave") or []) if isinstance(w, str)]
    return " ".join(str(p) for p in parts)


def main() -> None:
    rl.read_only()     # exp_ruling_links no longer does it on import (f2a4756)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cases", default=str(_default_cases_path()))
    parser.add_argument("--cards", default="/tmp/ruling_cards_v3.jsonl")
    parser.add_argument("--links", default="/tmp/ruling_links.jsonl")
    parser.add_argument("--cache", default="/tmp/ruling_search_cache.json",
                        help="the pipeline's outputs per case: written by a live run, read with --use-cache")
    parser.add_argument("--use-cache", action="store_true", help="rank from --cache, no pipeline or model call")
    parser.add_argument("--include-heldout", action="store_true", help="add h5-h9 (measure only)")
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--show", action="store_true", help="list each case's rulings, before and after (no text)")
    parser.add_argument("--issue-route", action="store_true",
                        help="also rank by the case's issues by name (off by default: no gain on the tuning cases)")
    parser.add_argument("--branch", default="penale", choices=("penale", "civile"),
                        help="the cases whose expected articles are all of this branch's codes")
    parser.add_argument("--production", action="store_true",
                        help="AFTER = the chatbot's own choice (src/rag/ruling_cards.py: anchors, side, phase)")
    args = parser.parse_args()

    links = {r["id"]: r for r in (json.loads(line) for line in open(args.links, encoding="utf-8"))}
    cards = {}
    for line in open(args.cards, encoding="utf-8"):
        card = json.loads(line)
        if not card.get("errore"):
            card["_cited"] = (links.get(card["id"]) or {}).get("articles") or []
            cards[card["id"]] = card
    cases = branch_cases(json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"], args.branch,
                         args.include_heldout, set(args.only.split(",")) if args.only else None)
    cache = json.loads(Path(args.cache).read_text(encoding="utf-8")) if os.path.exists(args.cache) else {}
    if args.use_cache:
        missing = [c["id"] for c in cases if c["id"] not in cache]
        if missing:
            sys.exit(f"--use-cache: no cached pipeline output for {', '.join(missing)}; run once without it")
    print(f"{len(cases)} {args.branch} cases, {len(cards)} cards, {len(links)} rulings in step 1's labels"
          + (f", pipeline outputs from {args.cache}" if args.use_cache else ""), file=sys.stderr)

    rarities = card_rarities(list(cards.values()))
    if args.production:
        from src.rag import ruling_cards as rc
        store = rc.CardStore([rc._trim(c) for c in cards.values()])
    rows, shown = [], []

    no_issues = [c["id"] for c in cases if args.use_cache and not cache[c["id"]].get("issues")]
    if no_issues and args.issue_route:
        print(f"   cached without issue names (older run): {', '.join(no_issues)} -- only their institutions' "
              "names are used; run once without --use-cache to cache the rest", file=sys.stderr)

    def one_case(case, run):
        expected = expected_keys(case)
        issues = issues_of(run) if args.issue_route else []
        if args.production:
            party = run.get("party") or ""
            ranked = rc.rank(store, run["central"], run["terms"], run["side"], party)
            after_ids = [c["id"] for c, _, _ in rc.choose(ranked, run["side"], party)]
        else:
            after_ids = rank_cards(list(cards.values()), run["central"], run["terms"], run["side"],
                                   rarities=rarities, issues=issues)
        party = run.get("party") or ""
        rows.append((case["id"], run["side"],
                     measure(run["before_ids"], links, cards, expected, run["side"], args.branch, party),
                     measure(after_ids, links, cards, expected, run["side"], args.branch, party),
                     available_on_topic(cards, links, expected)))
        if args.show:
            central = sorted(run["central"], key=lambda k: (-run["central"][k], k))
            shown.extend([f"## {case['id']}  (client: {run['side'] or 'unclear'}; expected: {', '.join(sorted(expected))}; "
                          f"central: {', '.join(central[:14]) or '-'}; "
                          f"issues: {'; '.join(name for name, _ in issues[:8]) or '-'})"]
                         + show_lines("BEFORE (today)", run["before_ids"], links, cards, expected)
                         + show_lines("AFTER (cards)", after_ids, links, cards, expected) + [""])

    if args.use_cache:
        for case in cases:
            one_case(case, cache[case["id"]])
    else:
        from src.rag.main import NEO4J_DATABASE, driver
        with driver.session(database=NEO4J_DATABASE) as session:
            for case in cases:
                t0 = time.time()
                run = live_ruling_ids(session, driver, NEO4J_DATABASE, case["question"])
                cache[case["id"]] = run
                Path(args.cache).write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                one_case(case, run)
                print(f"   {case['id']}: {len(run['before_ids'])} before, {len(run['central'])} central of "
                      f"{len(run['selected'])} selected articles, {time.time() - t0:.0f}s", file=sys.stderr)

    if shown:
        print("\n".join(shown))
    print("\n".join(report_lines(rows, "criminal" if args.branch == "penale" else "civil")))
    print(f"\nDone. Nothing was written to the database. Pipeline outputs: {args.cache}")


if __name__ == "__main__":
    main()
