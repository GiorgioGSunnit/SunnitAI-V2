"""Read-only experiment: does distilling a long narrative into neutral legal
facts improve retrieval, before touching the pipeline? No prod changes, no
DB writes -- Neo4j sessions only read, and the only side effect is the LLM
calls themselves (qwen2.5-32b-awq, same model the app already uses).

Run from the app folder so it uses the app's code, .env and venv:
    cd /opt/chatbot && venv/bin/python scripts/exp_distill.py
    ... --only sempronio,l1,l2             # just these cases
    ... --workers 4                        # cases in parallel (default 4)
    ... --vector-index section_embeddings_v2   # override the vector index name
    ... --k 20                             # retrieved per call; recall@10 and @k are both reported
    ... --out /root/exp_distill            # where the table and full detail land (default ./exp_distill_runs)

Methods compared, per case (code-aware recall: the article number AND the
code/law it belongs to, same scoring vocabulary as tests/eval_narrative.py):

  BASELINE           vector + fulltext search on the raw case text, merged
                      by best rank across the two.
  DISTILL             one LLM call, temperature 0: the case as 5-8 short
                      neutral bullets (fixed slots: soggetti e ruoli;
                      condotta; elementi sull'intenzione; evento/esito;
                      nesso causale; condotta successiva; posizione
                      processuale; cosa si chiede), no article numbers, no
                      code names. Retrieved per bullet and on all bullets
                      joined; merged by (how many of those calls found it,
                      then best rank among them).
  DISTILL_QUESTIONS  DISTILL's bullets, plus a second LLM call: 3-5 neutral
                      legal questions the facts raise, still no numbers;
                      retrieved per question too, merged into the same
                      hit-count-then-rank order.
  CURRENT            the live case-analysis v2 pipeline as it runs today
                      (_analyse_case_v2 + _select_articles_v2: institution
                      name -> real article, never a number from memory).

...and one composite:

  UNION              CURRENT + BASELINE + DISTILL, deduped (first
                      occurrence wins, in that priority order). Reported
                      with its own candidate count, recall@10/@k.

Round 5 (18 Oct) dropped RULINGS and UNION's numeric-neighbour/rulings-
added variants: on prod, neither changed UNION's recall (33/47 either
way) and RULINGS alone found only 9/47 -- not worth the extra calls.
Round 5 instead found UNION keeps ~all its candidates after the
applicability check (noise, not selection: SELECTED's recall@20 matched
UNION's, with ~16 non-expected articles still surviving per case) and
that recall@10 (25/47) trails recall@20 (33/47) -- the right articles are
usually IN the pool, just not ranked near the top. Four ranking methods
on the UNION pool answer that, reported at k=8 and k=12 (recall, noise,
and the latency each one adds):

  RERANK_BULLETS     the existing reranker (RERANKER_URL, qwen3-reranker-
                      8b), scoring each candidate's real text (titolo_
                      articolo + plain_text, truncated) against a query
                      built from the distilled bullets. Called directly
                      against the endpoint, not through reranker.
                      rerank_results (which also dedupes/injects/truncates
                      for the live answer -- this wants a plain score per
                      candidate). Ties -> more of {current, baseline,
                      distill} agreeing on it wins.
  RERANK_RAW          the same, query = the raw case text instead.
  LLM_RELEVANCE       one batched LLM call (temperature 0, qwen2.5-32b),
                      scoring every candidate 0-3 for RELEVANCE to the
                      facts (not mere compatibility), given the bullets
                      and each candidate's real text. Ties -> RERANK_
                      BULLETS' score.
  FUSION_BASELINE     no extra call: rank by how many of {current,
                      baseline, distill} found the article, then by the
                      best rank among those that did (_merge_by_hits).

Case set: sempronio, sempronio_no_hint, p1, p2, p3, c1, c2, c3, l1, l2, l3,
l4 -- all from tests/eval_narrative_cases.json (one source of truth with
tests/eval_narrative.py; l1-l4 were defined only here until 2026-10-19).
cp_short, cc_short, q_long and dottrina are deliberately left out of this
experiment -- they stay in tests/eval_narrative.py's own regression run.

2026-10-18 data fixes (now in tests/eval_narrative_cases.json itself, see
its _about): l2's "guaribile in 40 giorni" -> "60 giorni" (583 c.p.'s
aggravation needs MORE than 40 days, so 40 itself didn't actually support
the expected article). c3's expected dropped "art. 5 l392" (L. 392/1978
isn't in the corpus).

Round 8 (19 Oct): CASE_POOL_FUSION_ENABLED shipped FUSION_BASELINE's ranking
as src.rag.defensive_generation._select_articles_fusion, but as a PORT, not
a shared function -- the Cypher it used for BASELINE/DISTILL's own search
did a naive concatenation instead of a true best-rank merge, and its CURRENT
signal was institution-resolution alone rather than this script's CURRENT
(the full _select_articles_v2 selection), neither of which this script's own
measured numbers ever tested without. Diagnosed via check_applicability.py
--fusion on prod (a book and three wrong-code institution matches in the
ranked top 12, where this script had measured Sempronio at 5/5 @12). Fixed
by sharing, not porting: _merge_best_rank/_merge_by_hits below are now
aliases for dg._fusion_best_rank_merge/dg._fusion_merge_by_hits, and
_distill_bullets is dg._fusion_distill_bullets -- the same three functions
_select_articles_fusion itself calls, so the ranking algorithms and the
bullet-distillation prompt can't silently diverge again. One difference
remains, deliberately: _select_articles_fusion's own BASELINE/DISTILL search
(dg._fusion_vector_search/_fusion_fulltext_search) restricts to primary
sources and rejects book/manual names, while this script's BASELINE/DISTILL
columns stay unscoped (they're answering a different, standalone question --
how good is retrieval with no restriction at all). FUSION_BASELINE here is
therefore a close preview of the live pool, not an exact one; re-run it
against prod to re-validate the 27/47 @8 / 32/47 @12 numbers under that
narrower pool.

Round 10 (20 Oct) adds TIMELINE, a retrieval method built from the story's
own sequence of events rather than from fixed legal slots:

  TIMELINE           one LLM call, temperature 0, fixed JSON schema: the
                      narrative as an ordered list of small steps {n, actor,
                      action, object, means, result, relevance 0-2, terms:
                      2-4 statutory-style Italian terms, no article numbers}.
                      Steps with relevance 0 are dropped. Each kept step is
                      searched on "action + result + terms" through two
                      channels: (a) primary codes/laws, vector and fulltext
                      as two separate ranked lists; (b) rulings
                      (document_type='interpretation'), vector + fulltext,
                      ranked by document, then the articles those rulings
                      cite anywhere in their text (dg._refs_in, the same
                      citation reader as dg._articles_cited_in_rulings),
                      ordered by the best-ranked ruling citing each one, then
                      by how often it is cited. Every list from every step
                      and channel is merged by reciprocal rank fusion
                      (score = sum of 1/(60+rank)), with no cutoff before
                      the merge.
  TIMELINE_CURRENT   RRF of TIMELINE's ranking and CURRENT's, one list each
                      (CURRENT added as just one more list next to ~3 per
                      step would barely move the ranking).

Both are compared with BASELINE, DISTILL, CURRENT, UNION and FUSION_BASELINE
in one table: recall and noise at @10/@k, and per-case latency (composites
= the sum of their parts, since this script runs them one after another).

Round 11 (20 Oct). TIMELINE scored 19/47 @20 (TIMELINE_CURRENT 23/47,
FUSION_BASELINE 28/47, down from 34 since the re-ingest): the model
decomposes the facts fine, but lay facts don't share words with statutory
text. Two additions:

  SITUATION              each kept TIMELINE step, told plainly (actor,
                          action, object, means, result), and the raw case
                          text, searched against the situation index
                          (scripts/build_situation_index.py: 3-5 plain-
                          Italian situations per article, embedded, linked
                          to the article's main Section) -> articles, merged
                          by RRF. Queries get Qwen's query instruction and go
                          through the plain-text embedding client, so this
                          channel is right whatever EMBEDDING_QUERY_MODE is.
                          Latency includes TIMELINE's LLM call (its steps).
  SITUATION_FUSION       RRF(SITUATION, FUSION_BASELINE).
  FUSION_CLEAN           FUSION_BASELINE with BASELINE's and DISTILL's text
                          searches restricted to primary documents, minus
                          templates, contract models and books by name
                          (_NOT_A_LAW_RE, extendable with --exclude-docs) --
                          the 20 Oct briefing's §5.1 fusion-noise finding.
                          Same bullets as DISTILL; same merges. Every
                          document name the filter removed is reported, so
                          the rule can be checked against what it cut.
  SITUATION_FUSION_CLEAN RRF(SITUATION, FUSION_CLEAN) -- the combination a
                          cleaned-up live pool would actually use.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.path.join(ROOT, "tests"), os.getcwd()):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.getcwd(), ".env"))

import neo4j  # noqa: E402

_open_session = neo4j.Driver.session


def _read_only_session(self, **config):
    config["default_access_mode"] = neo4j.READ_ACCESS
    return _open_session(self, **config)


neo4j.Driver.session = _read_only_session   # every session in this process is read-only
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

import eval_narrative as ev  # noqa: E402  (tests/eval_narrative.py: art_of, code_of_document)
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

import src.rag.defensive_generation as dg  # noqa: E402
from src.rag.ai_chat import embedding_model  # noqa: E402
from src.rag.lookups import _sanitize_fulltext_query  # noqa: E402

DEFAULT_CASES = Path(__file__).resolve().parent.parent / "tests" / "eval_narrative_cases.json"

# --- The 12 cases, all from eval_narrative_cases.json (one source of truth,
# since 2026-10-19: l1-l4 used to be defined here too, duplicating what's now
# also in that file -- moved there to stop the two from drifting apart) ------
_CARRIED_OVER = ["sempronio", "sempronio_no_hint", "p1", "p2", "p3", "c1", "c2", "c3",
                "l1", "l2", "l3", "l4"]

METHODS = ["baseline", "distill", "distill_questions", "current"]
# Reported separately from METHODS: a composite built from the ones above,
# not an independent retrieval method, so it gets its own section instead
# of sharing the recall@10/@k table.
UNION_METHODS = ["union", "selected"]
# Round 10: TIMELINE and its fusion with CURRENT, compared against the other
# retrieval methods and the UNION/FUSION composites in one table.
TIMELINE_METHODS = ["timeline", "timeline_current"]
COMPARISON_METHODS = ["baseline", "distill", "current", "union", "fusion_baseline"] + TIMELINE_METHODS
# Latency of a composite = the methods it is built from (run sequentially here).
_COMPOSITE_LATENCY = {
    "union": ("current", "baseline", "distill"),
    "fusion_baseline": ("current", "baseline", "distill"),
    "timeline_current": ("timeline", "current"),
}
# Round 11.
SITUATION_METHODS = ["fusion_clean", "situation", "situation_fusion", "situation_fusion_clean"]
COMPARISON_METHODS += SITUATION_METHODS
_COMPOSITE_LATENCY.update({
    "fusion_clean": ("current", "baseline_clean", "distill_llm", "distill_clean_search"),
    "situation": ("timeline_llm", "situation_search"),
    "situation_fusion": ("timeline_llm", "situation_search", "current", "baseline", "distill"),
    "situation_fusion_clean": ("timeline_llm", "situation_search", "current", "baseline_clean", "distill_llm",
                               "distill_clean_search"),
})
_SITUATION_INSTRUCTION = (
    "Instruct: Given an account of events in plain Italian, retrieve the descriptions of concrete situations "
    "that match it\nQuery: "
)
_RRF_K = 60
_TIMELINE_MAX_STEPS = 20
_PRIMARY_TYPES = ["primary"]
_RULING_TYPES = ["interpretation"]
# Ranking methods applied to the UNION pool, reported at k=8/12 (recall,
# noise, added latency) rather than the @10/@k the retrieval methods use --
# see the module docstring, round 5.
RANKING_METHODS = ["rerank_bullets", "rerank_raw", "llm_relevance", "fusion_baseline"]

_DISTILL_QUESTIONS_SYSTEM = (
    "Dati questi fatti giuridici, scrivi da 3 a 5 domande giuridiche neutre che i fatti pongono (per "
    "esempio: \"quale reato integra un'azione violenta non diretta a uccidere da cui deriva la morte?\"). "
    "Non usare MAI un numero di articolo né il nome di un codice o di una legge. Restituisci SOLO un array "
    "JSON di stringhe."
)

_LLM_RELEVANCE_SYSTEM = (
    "Sei un avvocato italiano. Dati i fatti di un caso (come punti) e una lista di articoli candidati con "
    "il loro testo reale, valuta la RILEVANZA di ciascun articolo per questi fatti, su una scala da 0 a 3: "
    "0 = non rilevante, 1 = rilevante solo per l'area del diritto, 2 = rilevante per uno degli istituti in "
    "gioco, 3 = direttamente applicabile ai fatti specifici. Valuta la rilevanza per impostare un parere, "
    "non la compatibilità letterale articolo per articolo. Restituisci SOLO un array JSON di oggetti "
    "{\"articolo\": \"<identificativo esatto tra parentesi quadre>\", \"punteggio\": 0, 1, 2 o 3}."
)

_TIMELINE_SYSTEM = (
    "Sei un avvocato italiano. Scomponi il racconto in una sequenza ORDINATA di piccoli passi, un fatto per "
    "passo, nell'ordine in cui sono avvenuti. Per ogni passo restituisci un oggetto con esattamente queste "
    "chiavi: \"n\" (numero progressivo), \"actor\" (chi agisce), \"action\" (cosa fa), \"object\" (la "
    "cosa o la persona su cui agisce, la vittima se c'è; \"\" se assente), \"means\" (mezzo o modalità; "
    "\"\" se assente), \"result\" (esito o conseguenza; \"\" se assente), \"relevance\" (0 = "
    "giuridicamente irrilevante, 1 = contesto utile, 2 = giuridicamente decisivo), \"terms\" (da 2 a 4 "
    "espressioni giuridiche italiane nello stile del linguaggio normativo che descrivono il passo, per esempio "
    "\"impossessamento di cosa mobile altrui\", \"violenza alla persona\", \"lesione personale\"). Non "
    "usare MAI un numero di articolo né il nome di un codice o di una legge. Restituisci SOLO un array JSON."
)

# RERANKER_URL/RERANKER_MODEL: same defaults as src/rag/reranker.py. Called
# directly here (not through reranker.rerank_results, which also dedupes,
# injects protected sources and truncates for the live answer -- this wants
# a plain score per UNION candidate) and regardless of RERANKER_ENABLED,
# since this experiment is specifically testing whether reranking helps.
_RERANKER_URL = os.getenv("RERANKER_URL", "http://217.160.8.129:8002/v1/rerank")
_RERANKER_MODEL = os.getenv("RERANKER_MODEL", "reranker")

_ABBR_TO_EVAL_CODE = {"c.p.": "cp", "c.c.": "cc", "c.p.c.": "cpc", "c.p.p.": "cpp"}
_EVAL_CODE_TO_SEARCH_NAME = {
    "cp": "c.p.", "cc": "c.c.", "cpc": "c.p.c.", "cpp": "c.p.p.",
    "consumo": "Codice del Consumo", "l392": "L. 392/1978",
}
# --- Pure helpers (unit-tested in tests/test_exp_distill.py) ---------------

def _row_ref(row):
    """A retrieved Section row -> (eval code key, article number), or None
    when either can't be read (not a Section, missing/unknown document)."""
    code = ev.code_of_document(row.get("doc_name"))
    art = ev.art_of(row.get("name"))
    if not code or not art:
        return None
    return (code, art)


def _ranked_refs(rows):
    """rows (already ordered by score, descending) -> [(code, article), ...]
    in rank order, first occurrence kept."""
    seen, ordered = set(), []
    for row in rows:
        ref = _row_ref(row)
        if ref and ref not in seen:
            seen.add(ref)
            ordered.append(ref)
    return ordered


# _merge_best_rank/_merge_by_hits used to be this script's own copies of
# src/rag/defensive_generation.py's merge algorithms -- a port, not a
# shared function. Round 8, 19 Oct: that let the two drift (dg's own
# _fusion_text_search did a naive concatenation instead of a true best-
# rank merge, a bug this script's FUSION_BASELINE never had and so never
# caught). Both names are now aliases for dg's functions -- pure,
# format-agnostic tuple merges, so aliasing them here changes nothing
# about how this script's (eval code, article) refs are merged, it just
# makes it impossible for the two to silently diverge again.
_merge_best_rank = dg._fusion_best_rank_merge
_merge_by_hits = dg._fusion_merge_by_hits


def _code_key(abbr_or_name):
    """CURRENT's own (number, abbr-or-document-name) tag -> the eval's code
    key ("cp", "cc", ...), for the four/six sigle directly, or by matching
    the real document name (Codice del Consumo, l. 392/1978, ...) the same
    way tests/eval_narrative.py already does."""
    return _ABBR_TO_EVAL_CODE.get(abbr_or_name) or ev.code_of_document(abbr_or_name)


def _dedupe_ordered(*ranked_lists):
    """Union of refs across the given lists, first occurrence wins -- used
    for UNION, where the lists are already each a method's own priority
    order (CURRENT's curated selection first, then BASELINE, then DISTILL),
    not a score to re-merge."""
    seen, ordered = set(), []
    for ranked in ranked_lists:
        for ref in ranked:
            if ref not in seen:
                seen.add(ref)
                ordered.append(ref)
    return ordered


def _ref_label(ref):
    """(eval code, article) -> "<article> <sigla>", the identificativo
    shown to the reranker/LLM and parsed back from the LLM's reply."""
    code, article = ref
    return f"{article} {_EVAL_CODE_TO_SEARCH_NAME.get(code, code)}"


def _signal_counts(union_refs, current, baseline, distill):
    """How many of {current, baseline, distill} contain each UNION ref --
    RERANK's tie-break and FUSION_BASELINE's whole ranking."""
    sets = (set(current), set(baseline), set(distill))
    return {ref: sum(1 for s in sets if ref in s) for ref in union_refs}


def _noise(ranked_refs, expected, k):
    """How many of the top k refs are NOT one of the expected articles --
    SELECTED's cost side: recall alone can't tell a tight, precise set from
    one padded with irrelevant candidates the applicability check let through."""
    expected_set = set()
    for e in expected:
        if "any_of" in e:
            expected_set.update((e["code"], a) for a in e["any_of"])
        else:
            expected_set.add((e["code"], e["article"]))
    return sum(1 for ref in ranked_refs[:k] if ref not in expected_set)


def _recall(ranked_refs, expected, k):
    """(found, total) among `expected` within the top k of ranked_refs.
    An entry with "any_of" (c2's Codice del Consumo range) counts as found
    if any one of those articles, under that code, is in the top k."""
    top = set(ranked_refs[:k])
    found = 0
    for e in expected:
        if "any_of" in e:
            hit = any((e["code"], a) in top for a in e["any_of"])
        else:
            hit = (e["code"], e["article"]) in top
        found += int(hit)
    return found, len(expected)


def _rrf_scores(*ranked_lists, k0=_RRF_K):
    """{ref: sum over the lists containing it of 1/(k0 + rank)}, rank from 1."""
    scores = {}
    for ranked in ranked_lists:
        for rank, ref in enumerate(ranked, start=1):
            scores[ref] = scores.get(ref, 0.0) + 1.0 / (k0 + rank)
    return scores


def _rrf(*ranked_lists, k0=_RRF_K):
    """Reciprocal rank fusion: refs by RRF score, descending; ties keep the
    order in which the refs were first seen across the lists."""
    scores = _rrf_scores(*ranked_lists, k0=k0)
    first_seen = {ref: i for i, ref in enumerate(scores)}
    return sorted(scores, key=lambda ref: (-scores[ref], first_seen[ref]))


def _relevance(value):
    """A step's relevance as 0, 1 or 2; anything unreadable counts as 0."""
    try:
        rel = int(value)
    except (TypeError, ValueError):
        return 0
    return rel if rel in (0, 1, 2) else 0


def _parse_timeline(parsed):
    """The model's reply -> every step, normalised (strings stripped,
    relevance 0-2, terms: 2-4 strings without digits -- an article number
    slipping into a term would turn a description into a lookup)."""
    if not isinstance(parsed, list):
        return []
    steps = []
    for i, item in enumerate(parsed[:_TIMELINE_MAX_STEPS], start=1):
        if not isinstance(item, dict):
            continue
        terms = item.get("terms")
        terms = [t.strip() for t in terms if isinstance(t, str) and t.strip() and not any(c.isdigit() for c in t)] \
            if isinstance(terms, list) else []
        steps.append({
            "n": item.get("n") if isinstance(item.get("n"), int) else i,
            **{key: str(item.get(key) or "").strip() for key in ("actor", "action", "object", "means", "result")},
            "relevance": _relevance(item.get("relevance")),
            "terms": terms[:4],
        })
    return steps


def _kept_steps(steps):
    return [step for step in steps if step["relevance"] > 0]


def _timeline_query(step):
    """What a kept step is searched on: action + result + terms."""
    return " ".join(part for part in [step["action"], step["result"], *step["terms"]] if part)


def _cited_ranking(ruling_ids, cites_by_doc):
    """Articles cited by these rulings (ranked best first), ordered by the
    best rank of a ruling citing each one, then by how many times these
    rulings cite it in total. cites_by_doc: {doc id: {ref: citation count}}."""
    best, count = {}, {}
    for rank, doc_id in enumerate(ruling_ids, start=1):
        for ref, n in (cites_by_doc.get(doc_id) or {}).items():
            best.setdefault(ref, rank)
            count[ref] = count.get(ref, 0) + n
    return sorted(best, key=lambda ref: (best[ref], -count[ref]))


# Not a code or a law, by name: dg._NOT_A_PRIMARY_SOURCE_RE's books, plus
# textbooks and the templates/contract models the 20 Oct briefing found in
# the pool (c3: "LOCAZIONE ABITATIVA DI NATURA TRANSITORIA"). Deliberately
# not "contratt" alone: "Codice dei contratti pubblici" is a law.
_NOT_A_LAW_RE = re.compile(
    dg._NOT_A_PRIMARY_SOURCE_RE.pattern
    + r"|fondamenti|lineamenti|istituzioni di|lezioni di|compendio|corso di diritto|parte generale|parte speciale"
    r"|fac.?simile|template|\bmodell[oi] d|schema di contratto|contratto tipo|scrittura privata|natura transitoria"
    r"|locazione abitativa",
    re.IGNORECASE,
)


def _is_law_name(doc_name, exclude_re=_NOT_A_LAW_RE):
    return bool(doc_name) and not exclude_re.search(doc_name)


def _step_story(step):
    """A TIMELINE step told plainly, for the situation index: who did what,
    to whom, how, with what result -- no legal terms."""
    return " ".join(part for part in (step["actor"], step["action"], step["object"], step["means"], step["result"])
                    if part)


# --- Retrieval ---------------------------------------------------------

# doc_types scopes a search to those Document.document_type values; the index
# is then queried for k * _SCOPED_OVERFETCH nodes so the filter still leaves k.
_SCOPED_OVERFETCH = 5


def _doc_match(doc_types):
    return ("MATCH (d:Document)-[:CONTAINS]->(node) WHERE d.document_type IN $types " if doc_types
            else "OPTIONAL MATCH (d:Document)-[:CONTAINS]->(node) ")


def _vector_search(session, text, index, k, warned, doc_types=None, limit=None):
    try:
        embedding = list(embedding_model.embed_query(text))
    except Exception as exc:
        logging.warning("embedding failed: %s", exc)
        return []
    try:
        return session.run(
            "CALL db.index.vector.queryNodes($index, $fetch, $embedding) YIELD node, score "
            "WHERE 'Section' IN labels(node) AND node.plain_text IS NOT NULL AND node.plain_text <> '' "
            + _doc_match(doc_types) +
            "RETURN node.name AS name, d.name AS doc_name, d.id AS doc_id, score ORDER BY score DESC LIMIT $k",
            index=index, k=limit or k, fetch=(limit or k) * _SCOPED_OVERFETCH if doc_types else k,
            embedding=embedding, types=doc_types or [],
        ).data()
    except Exception as exc:
        if index not in warned:
            warned.add(index)
            logging.warning("vector search on index %r failed (does it exist?): %s", index, exc)
        return []


def _fulltext_search(session, text, k, doc_types=None, limit=None):
    q = _sanitize_fulltext_query(text)
    if not q:
        return []
    return session.run(
        "CALL db.index.fulltext.queryNodes('section_fulltext', $q, {limit: $fetch}) YIELD node, score "
        + _doc_match(doc_types) +
        "RETURN node.name AS name, d.name AS doc_name, d.id AS doc_id, score ORDER BY score DESC LIMIT $k",
        q=q, k=limit or k, fetch=(limit or k) * _SCOPED_OVERFETCH if doc_types else k, types=doc_types or [],
    ).data()


def _retrieve(session, text, index, k, warned):
    return _ranked_refs(_vector_search(session, text, index, k, warned) + _fulltext_search(session, text, k))


def _clean_rows(rows, exclude_re, excluded):
    """rows whose document is a code or a law; the names of the others go into `excluded`."""
    kept = []
    for row in rows:
        if _is_law_name(row.get("doc_name"), exclude_re):
            kept.append(row)
        elif row.get("doc_name"):
            excluded.add(row["doc_name"])
    return kept


def _clean_searches(session, text, index, k, warned, exclude_re, excluded):
    """FUSION_CLEAN's (vector refs, fulltext refs) for `text`: primary documents
    only, minus non-laws by name; 3k rows fetched so k refs survive the filter."""
    vector = _clean_rows(_vector_search(session, text, index, k, warned, _PRIMARY_TYPES, limit=3 * k),
                         exclude_re, excluded)
    fulltext = _clean_rows(_fulltext_search(session, text, k, _PRIMARY_TYPES, limit=3 * k), exclude_re, excluded)
    return _ranked_refs(vector)[:k], _ranked_refs(fulltext)[:k]


def _retrieve_clean(session, text, index, k, warned, exclude_re, excluded):
    """_retrieve, cleaned: the same vector-then-fulltext order DISTILL's own calls use."""
    vector, fulltext = _clean_searches(session, text, index, k, warned, exclude_re, excluded)
    return list(dict.fromkeys(vector + fulltext))


_situation_embedder = None   # set in main(): the plain-text client build_situation_index.py embeds with


def _situation_search(session, text, index, k, warned):
    """Articles whose situations match `text`, best first: 5k situations
    fetched (3-5 per article), each mapped to its article's main Section."""
    try:
        embedding = list(_situation_embedder.embed_query(_SITUATION_INSTRUCTION + text))
    except Exception as exc:
        logging.warning("situation embedding failed: %s", exc)
        return []
    try:
        rows = session.run(
            "CALL db.index.vector.queryNodes($index, $fetch, $embedding) YIELD node, score "
            "MATCH (s:Section)-[:HAS_SITUATION]->(node) MATCH (d:Document)-[:CONTAINS]->(s) "
            "RETURN s.name AS name, d.name AS doc_name, score ORDER BY score DESC",
            index=index, fetch=5 * k, embedding=embedding,
        ).data()
    except Exception as exc:
        if index not in warned:
            warned.add(index)
            logging.warning("situation search on index %r failed (built yet? scripts/build_situation_index.py): %s",
                            index, exc)
        return []
    return _ranked_refs(rows)


# --- UNION candidates' real text, and the two new ranking methods ----------

def _ref_of_row(row):
    """A fetched article row (dg._fetch_any's shape: _article = (number,
    abbr-or-document-name)) -> (eval code, article), or None."""
    article = row.get("_article")
    if not article:
        return None
    code = _code_key(article[1])
    return (code, article[0]) if code else None


def _union_texts(rows, chars=500):
    """{ref: "titolo_articolo. plain_text"[:chars]} from UNION's fetched
    rows -- a ref with several rows (fragments of one article) gets them
    joined; a ref with none gets no entry (callers treat that as "")."""
    by_ref = {}
    for row in rows:
        ref = _ref_of_row(row)
        if not ref:
            continue
        s = row.get("s") or {}
        titolo = (s.get("titolo_articolo") or "").strip()
        plain = (s.get("plain_text") or "").strip()
        text = f"{titolo}. {plain}" if titolo else plain
        by_ref.setdefault(ref, []).append(text)
    return {ref: " ".join(texts)[:chars] for ref, texts in by_ref.items()}


def _rerank_scores(query, documents, timeout=20):
    """relevance_score per document, in the same order as `documents`, from
    the live reranker endpoint -- see the module docstring for why this
    calls it directly rather than through reranker.rerank_results."""
    if not documents:
        return []
    import requests
    payload = {
        "model": _RERANKER_MODEL,
        "query": ("Instruct: Given a legal query in Italian, retrieve the most relevant legal document "
                 f"sections\nQuery: {query}"),
        "documents": documents,
    }
    resp = requests.post(
        _RERANKER_URL,
        headers={"Authorization": f"Bearer {os.getenv('LLM_API_KEY', '')}", "Content-Type": "application/json"},
        json=payload, timeout=timeout,
    )
    resp.raise_for_status()
    scores = [0.0] * len(documents)
    for result in resp.json().get("results") or []:
        idx = result.get("index")
        if isinstance(idx, int) and 0 <= idx < len(scores):
            scores[idx] = result.get("relevance_score", 0.0)
    return scores


def _rank_by_reranker(union_refs, texts, query, signal_counts):
    """UNION, ranked by reranker score against `query` (descending); ties
    broken by how many of {current, baseline, distill} found the ref."""
    docs = [texts.get(ref, "") for ref in union_refs]
    scores = dict(zip(union_refs, _rerank_scores(query, docs)))
    return sorted(union_refs, key=lambda ref: (-scores.get(ref, 0.0), -signal_counts.get(ref, 0))), scores


def _llm_relevance_scores(bullets_text, union_refs, texts):
    """{ref: 0-3} from one batched LLM call -- empty for a ref with no text
    (nothing to show the model) or if the whole call fails or the reply
    doesn't parse; callers treat a missing ref as score 0."""
    label_of = {ref: _ref_label(ref) for ref in union_refs}
    listing = "\n\n".join(f"[{label_of[ref]}]\n{texts[ref]}" for ref in union_refs if texts.get(ref))
    if not listing:
        return {}
    human = f"FATTI:\n{bullets_text or '(nessun punto distillato)'}\n\nARTICOLI CANDIDATI:\n{listing}"
    raw = dg._chat([SystemMessage(content=_LLM_RELEVANCE_SYSTEM), HumanMessage(content=human)],
                   max_tokens=2000, temperature=0)
    parsed = dg.parse_json_reply(raw)
    if not isinstance(parsed, list):
        return {}
    label_to_ref = {v: k for k, v in label_of.items()}
    scores = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        ref = label_to_ref.get(str(item.get("articolo") or "").strip())
        if ref is None:
            continue
        try:
            scores[ref] = int(item.get("punteggio"))
        except (TypeError, ValueError):
            continue
    return scores


def _rank_by_llm_relevance(union_refs, texts, bullets_text, reranker_scores):
    """UNION, ranked by LLM relevance score (descending); ties broken by
    RERANK_BULLETS' score. Fails open to an unchanged order on any error."""
    try:
        scores = _llm_relevance_scores(bullets_text, union_refs, texts)
    except Exception as exc:
        logging.warning("LLM relevance scoring failed: %s", exc)
        scores = {}
    ranked = sorted(union_refs, key=lambda ref: (-scores.get(ref, 0), -reranker_scores.get(ref, 0.0)))
    return ranked, scores


# --- The methods ----------------------------------------------------

# dg._fusion_distill_bullets/_FUSION_DISTILL_SYSTEM were ported from this
# script's own _distill_bullets/_DISTILL_SYSTEM when CASE_POOL_FUSION_
# ENABLED shipped (round 7, 19 Oct). Calling that function here instead of
# keeping the two prompts as separate (if identical) copies is what makes
# DISTILL/DISTILL_QUESTIONS/FUSION_BASELINE's bullets the same ones the
# live fusion pool would distill for the same case.
_distill_bullets = dg._fusion_distill_bullets


def _distill_questions(bullets):
    if not bullets:
        return []
    human = "\n".join(f"- {b['testo']}" for b in bullets)
    raw = dg._chat([SystemMessage(content=_DISTILL_QUESTIONS_SYSTEM), HumanMessage(content=human)],
                   max_tokens=400, temperature=0)
    parsed = dg.parse_json_reply(raw)
    if not isinstance(parsed, list):
        return []
    return [str(q).strip() for q in parsed if isinstance(q, str) and q.strip()][:5]


def _run_current(session, driver, database, text):
    analysis = (dg._analyse_case_structured(text, "it") if dg.feature_flags.enabled(dg.STRUCTURED_ANALYSIS_FLAG)
                else dg._analyse_case_v2(text, "it"))
    picked = dg._select_articles_v2(session, driver, database, text, "", analysis)
    seen, refs = set(), []
    for number, abbr in picked.get("selected") or []:
        code = _code_key(abbr)
        if not code:
            continue
        ref = (code, number)
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)
    return refs, analysis


def _to_fetch_ref(ref):
    """(eval code, article) -> (article, search name) for dg._fetch_any, or
    None for a code this experiment has no search name for."""
    code, article = ref
    name = _EVAL_CODE_TO_SEARCH_NAME.get(code)
    return (article, name) if name else None


def _fetch_union_rows(session, documents, union_refs):
    """UNION's refs, fetched for their real (titolo_articolo, plain_text)
    rows -- shared by SELECTED (the applicability check) and the two
    reranking methods (the text they score), so each ref is fetched once."""
    fetch_refs = [r for r in (_to_fetch_ref(ref) for ref in union_refs) if r]
    return dg._fetch_any(session, documents, fetch_refs) if fetch_refs else []


def _selected_refs(text, rows, union_refs):
    """union_refs, through the live applicability check
    (dg._filter_applicable_articles) -- the same check run_case_analysis_
    pipeline applies today, read-only and standalone here. Order preserved,
    just with the dropped ones removed."""
    if not rows:
        return []
    kept_rows = dg._filter_applicable_articles(text, "", rows)
    kept = {ref for ref in (_ref_of_row(r) for r in kept_rows) if ref}
    return [ref for ref in union_refs if ref in kept]


def _timeline_steps(text):
    parsed = dg._chat_json([SystemMessage(content=_TIMELINE_SYSTEM), HumanMessage(content=text[:6000])],
                           max_tokens=2500, temperature=0)
    return _parse_timeline(parsed)


def _ruling_ids(session, query, index, k, warned):
    """Ruling documents for a query, best rank first: vector and fulltext
    hits on interpretation sections, by document, best-rank merged."""
    def doc_order(rows):
        return list(dict.fromkeys(row["doc_id"] for row in rows if row.get("doc_id")))
    return _merge_best_rank(doc_order(_vector_search(session, query, index, k, warned, _RULING_TYPES)),
                            doc_order(_fulltext_search(session, query, k, _RULING_TYPES)))


def _citations_by_doc(session, doc_ids):
    """{doc id: {(eval code, article): times cited}} over each ruling's whole
    text -- dg._refs_in, the same reader dg._articles_cited_in_rulings uses."""
    ids = [i for i in dict.fromkeys(doc_ids) if i]
    if not ids:
        return {}
    cites = {}
    for row in session.run(
        "MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE d.id IN $ids RETURN d.id AS id, s.plain_text AS text",
        ids=ids,
    ).data():
        for number, abbr in dg._refs_in(row.get("text") or ""):
            code = _code_key(abbr)
            if code:
                per_doc = cites.setdefault(row["id"], {})
                per_doc[(code, number)] = per_doc.get((code, number), 0) + 1
    return cites


def _run_timeline(session, text, index, k, warned):
    """TIMELINE: (ranking, every step, the ranked lists per kept step, LLM seconds)."""
    t0 = time.time()
    steps = _timeline_steps(text)
    llm_seconds = time.time() - t0
    per_step = []
    for step in _kept_steps(steps):
        query = _timeline_query(step)
        per_step.append({
            "n": step["n"], "query": query,
            "primary_vector": _ranked_refs(_vector_search(session, query, index, k, warned, _PRIMARY_TYPES)),
            "primary_fulltext": _ranked_refs(_fulltext_search(session, query, k, _PRIMARY_TYPES)),
            "rulings": _ruling_ids(session, query, index, k, warned),
        })
    cites = _citations_by_doc(session, [doc_id for s in per_step for doc_id in s["rulings"]])
    lists = []
    for s in per_step:
        s["ruling_citations"] = _cited_ranking(s["rulings"], cites)
        lists += [s["primary_vector"], s["primary_fulltext"], s["ruling_citations"]]
    return _rrf(*lists), steps, per_step, llm_seconds


def _run_case(driver, database, case, vector_index, k, warned, situation_index="situation_embeddings_v2",
              exclude_re=_NOT_A_LAW_RE):
    text = case["question"]
    detail = {}
    with driver.session(database=database) as session:
        from src.rag.doc_lookup import _fetch_doc_names
        documents = _fetch_doc_names(driver, database)

        latency = {}
        t0 = time.time()
        baseline = _merge_best_rank(
            _ranked_refs(_vector_search(session, text, vector_index, k, warned)),
            _ranked_refs(_fulltext_search(session, text, k)),
        )
        latency["baseline"] = time.time() - t0

        t0 = time.time()
        try:
            bullets = _distill_bullets(text)
        except Exception as exc:
            logging.warning("%s: distill (bullets) failed: %s", case["id"], exc)
            bullets = []
        latency["distill_llm"] = time.time() - t0
        detail["bullets"] = bullets
        joined = " ".join(b["testo"] for b in bullets)
        bullet_lists = [_retrieve(session, b["testo"], vector_index, k, warned) for b in bullets]
        if joined:
            bullet_lists.append(_retrieve(session, joined, vector_index, k, warned))
        distill = _merge_by_hits(*bullet_lists) if bullet_lists else []
        latency["distill"] = time.time() - t0

        # FUSION_CLEAN's sources: the same searches, cleaned (round 11).
        excluded = set()
        t0 = time.time()
        baseline_clean = _merge_best_rank(*_clean_searches(session, text, vector_index, k, warned, exclude_re,
                                                           excluded))
        latency["baseline_clean"] = time.time() - t0
        t0 = time.time()
        clean_lists = [_retrieve_clean(session, b["testo"], vector_index, k, warned, exclude_re, excluded)
                       for b in bullets]
        if joined:
            clean_lists.append(_retrieve_clean(session, joined, vector_index, k, warned, exclude_re, excluded))
        distill_clean = _merge_by_hits(*clean_lists) if clean_lists else []
        latency["distill_clean_search"] = time.time() - t0
        detail["clean_excluded_documents"] = sorted(excluded)

        try:
            questions = _distill_questions(bullets)
        except Exception as exc:
            logging.warning("%s: distill (questions) failed: %s", case["id"], exc)
            questions = []
        detail["questions"] = questions
        question_lists = [_retrieve(session, q, vector_index, k, warned) for q in questions]
        distill_questions = _merge_by_hits(*bullet_lists, *question_lists) if (bullet_lists or question_lists) \
            else []

        t0 = time.time()
        try:
            current, analysis = _run_current(session, driver, database, text)
        except Exception as exc:
            logging.warning("%s: CURRENT (case-analysis v2) failed: %s", case["id"], exc)
            current, analysis = [], {}
        latency["current"] = time.time() - t0
        detail["current_analysis"] = analysis

        t0 = time.time()
        try:
            timeline, steps, per_step, latency["timeline_llm"] = _run_timeline(
                session, text, vector_index, k, warned)
        except Exception as exc:
            logging.warning("%s: TIMELINE failed: %s", case["id"], exc)
            timeline, steps, per_step = [], [], []
        latency["timeline"] = time.time() - t0
        detail["timeline"] = steps
        detail["timeline_searches"] = [
            {"n": s["n"], "query": s["query"], "rulings": len(s["rulings"]),
             **{key: [_ref_label(ref) for ref in s[key][:10]]
                for key in ("primary_vector", "primary_fulltext", "ruling_citations")}}
            for s in per_step]
        timeline_current = _rrf(timeline, current)

        t0 = time.time()
        situation_lists = [_situation_search(session, _step_story(step), situation_index, k, warned)
                           for step in _kept_steps(steps)]
        situation_lists.append(_situation_search(session, text, situation_index, k, warned))
        situation = _rrf(*situation_lists)
        latency["situation_search"] = time.time() - t0
        detail["situation_searches"] = [[_ref_label(ref) for ref in refs[:10]] for refs in situation_lists]

        union = _dedupe_ordered(current, baseline, distill)
        union_rows = _fetch_union_rows(session, documents, union)
        texts = _union_texts(union_rows)

        try:
            selected = _selected_refs(text, union_rows, union)
        except Exception as exc:
            logging.warning("%s: applicability check on UNION failed: %s", case["id"], exc)
            selected = union

        signal_counts = _signal_counts(union, current, baseline, distill)

        t0 = time.time()
        try:
            rerank_bullets, rerank_bullets_scores = (_rank_by_reranker(union, texts, joined, signal_counts)
                                                     if joined else (union, {}))
        except Exception as exc:
            logging.warning("%s: RERANK_BULLETS failed: %s", case["id"], exc)
            rerank_bullets, rerank_bullets_scores = union, {}
        latency["rerank_bullets"] = time.time() - t0

        t0 = time.time()
        try:
            rerank_raw, _ = _rank_by_reranker(union, texts, text, signal_counts)
        except Exception as exc:
            logging.warning("%s: RERANK_RAW failed: %s", case["id"], exc)
            rerank_raw = union
        latency["rerank_raw"] = time.time() - t0

        t0 = time.time()
        llm_relevance, llm_relevance_scores = _rank_by_llm_relevance(union, texts, joined, rerank_bullets_scores)
        latency["llm_relevance"] = time.time() - t0

        t0 = time.time()
        fusion_baseline = _merge_by_hits(current, baseline, distill)
        latency["fusion_baseline"] = time.time() - t0
        fusion_clean = _merge_by_hits(current, baseline_clean, distill_clean)

    detail["llm_relevance_scores"] = {_ref_label(ref): score for ref, score in llm_relevance_scores.items()}
    return {"baseline": baseline, "distill": distill, "distill_questions": distill_questions,
            "current": current, "union": union, "selected": selected,
            "rerank_bullets": rerank_bullets, "rerank_raw": rerank_raw,
            "llm_relevance": llm_relevance, "fusion_baseline": fusion_baseline,
            "timeline": timeline, "timeline_current": timeline_current,
            "fusion_clean": fusion_clean, "situation": situation,
            "situation_fusion": _rrf(situation, fusion_baseline),
            "situation_fusion_clean": _rrf(situation, fusion_clean),
            "union_candidate_count": len(union), "latency": latency, "detail": detail}


# --- Reporting -------------------------------------------------------------

def _recall_table(results, cases, k, methods):
    lines = []
    header = f"{'case':<18s}" + "".join(f"{m:<26s}" for m in methods)
    lines.append(header)
    totals = {m: [0, 0, 0, 0] for m in methods}
    for case in cases:
        r = results[case["id"]]
        cells = []
        for m in methods:
            f10, t10 = _recall(r[m], case["expected"], 10)
            fk, tk = _recall(r[m], case["expected"], k)
            cells.append(f"{f10}/{t10}@10  {fk}/{tk}@{k}".ljust(26))
            totals[m][0] += f10; totals[m][1] += t10; totals[m][2] += fk; totals[m][3] += tk
        lines.append(f"{case['id']:<18s}" + "".join(cells))
    lines.append("")
    lines.append(f"{'TOTAL':<18s}" + "".join(
        f"{totals[m][0]}/{totals[m][1]}@10  {totals[m][2]}/{totals[m][3]}@{k}".ljust(26) for m in methods))
    return lines


def _noise_latency_table(results, cases, methods, ks):
    """recall and noise at each k in `ks`, plus added latency, for a set of
    ranking methods applied to the UNION pool."""
    lines = []
    col = 18 + 6 * len(ks)
    header = f"{'case':<10s}" + "".join(f"{m:<{col}s}" for m in methods)
    lines.append(header)
    totals = {m: {k: [0, 0, 0] for k in ks} for m in methods}  # found, total, noise
    latency_totals = {m: 0.0 for m in methods}
    for case in cases:
        r = results[case["id"]]
        cells = []
        for m in methods:
            parts = []
            for k in ks:
                found, total = _recall(r[m], case["expected"], k)
                noise = _noise(r[m], case["expected"], k)
                parts.append(f"{found}/{total}@{k} n{noise}")
                totals[m][k][0] += found; totals[m][k][1] += total; totals[m][k][2] += noise
            cells.append("  ".join(parts).ljust(col))
            latency_totals[m] += r["latency"].get(m, 0.0)
        lines.append(f"{case['id']:<10s}" + "".join(cells))
    lines.append("")
    lines.append(f"{'TOTAL':<10s}" + "".join(
        ("  ".join(f"{totals[m][k][0]}/{totals[m][k][1]}@{k} n{totals[m][k][2]}" for k in ks)).ljust(col)
        for m in methods))
    lines.append("")
    lines.append("total added latency across all cases (seconds): " + ", ".join(
        f"{m}={latency_totals[m]:.1f}s" for m in methods))
    return lines


def _method_latency(r, method):
    parts = _COMPOSITE_LATENCY.get(method, (method,))
    return sum(r["latency"].get(m, 0.0) for m in parts)


def _comparison_lines(results, cases, k):
    """Retrieval methods and composites side by side: totals (recall, noise,
    mean latency per case), then found@10/found@k per case."""
    lines = [f"{'method':<24s}{'recall@10':<12s}{'recall@' + str(k):<12s}{'noise@10':<10s}"
             f"{'noise@' + str(k):<10s}{'latency/case':<14s}"]
    for m in COMPARISON_METHODS:
        f10 = fk = total = n10 = nk = 0
        for case in cases:
            r = results[case["id"]]
            a, t = _recall(r[m], case["expected"], 10)
            b, _ = _recall(r[m], case["expected"], k)
            f10 += a; fk += b; total += t
            n10 += _noise(r[m], case["expected"], 10)
            nk += _noise(r[m], case["expected"], k)
        latency = sum(_method_latency(results[c["id"]], m) for c in cases) / len(cases)
        lines.append(f"{m:<24s}{f'{f10}/{total}':<12s}{f'{fk}/{total}':<12s}{n10:<10d}{nk:<10d}{latency:>6.1f}s")
    lines.append("")
    lines.append("per case, found@10/found@" + str(k) + ":")
    lines.append(f"{'case':<18s}" + "".join(f"{m:<24s}" for m in COMPARISON_METHODS))
    for case in cases:
        r = results[case["id"]]
        cells = []
        for m in COMPARISON_METHODS:
            a, t = _recall(r[m], case["expected"], 10)
            b, _ = _recall(r[m], case["expected"], k)
            cells.append(f"{a}/{b} of {t}".ljust(24))
        lines.append(f"{case['id']:<18s}" + "".join(cells))
    lines.append("")
    lines.append("TIMELINE's own LLM call, mean per case: "
                 f"{sum(results[c['id']]['latency'].get('timeline_llm', 0.0) for c in cases) / len(cases):.1f}s "
                 "(the rest of TIMELINE's latency is its searches). Composite latency = sum of its parts; "
                 "SITUATION's includes that LLM call (its steps).")
    excluded = sorted({name for c in cases for name in results[c["id"]]["detail"].get("clean_excluded_documents", [])})
    lines.append("")
    lines.append(f"FUSION_CLEAN removed {len(excluded)} document(s) by name (check none is a code or a law):")
    lines += [f"   - {name}" for name in excluded]
    return lines


def _timeline_lines(case_id, detail):
    lines = [f"   timeline ({case_id}):"]
    for step in detail.get("timeline") or []:
        mark = "  " if step["relevance"] else "x "
        lines.append(f"     {mark}{step['n']}. [rel {step['relevance']}] {step['actor']} | {step['action']} | "
                     f"{step['object']} | {step['means']} | {step['result']} | terms: {step['terms']}")
    return lines


def _table_lines(results, cases, k):
    lines = [f"TIMELINE vs the other methods -- recall@10/@{k} (code-aware), noise (non-expected articles in "
             f"the top 10/{k}), latency:", ""]
    lines += _comparison_lines(results, cases, k)
    lines.append("")
    lines.append(f"recall@10 and @{k}, found/expected, code-aware:")
    lines.append("")
    lines += _recall_table(results, cases, k, METHODS)
    lines.append("")
    lines.append("UNION (current + baseline + distill, deduped) and SELECTED (UNION through the live "
                "applicability check) -- recall, plus candidate count and SELECTED's noise (non-expected "
                "articles kept):")
    lines.append("")
    lines += _recall_table(results, cases, k, UNION_METHODS)
    lines.append("")
    header = f"{'case':<18s}{'union candidates':<20s}{'selected candidates':<22s}{'selected noise@' + str(k):<18s}"
    lines.append(header)
    for case in cases:
        r = results[case["id"]]
        noise = _noise(r["selected"], case["expected"], k)
        lines.append(f"{case['id']:<18s}{r['union_candidate_count']:<20d}{len(r['selected']):<22d}{noise:<18d}")
    lines.append("")
    lines.append("Ranking methods on the UNION pool -- recall and noise ('n') at k=8 and k=12 (the number of "
                "articles a draft can realistically use), plus added latency per case:")
    lines.append("")
    lines += _noise_latency_table(results, cases, RANKING_METHODS, (8, 12))
    return lines


def _detail_lines(results, cases, k):
    lines = []
    for case in cases:
        r = results[case["id"]]
        lines.append(f"\n## {case['id']}")
        lines.append(f"   expected: {case['expected']}")
        for m in METHODS + UNION_METHODS + RANKING_METHODS + TIMELINE_METHODS + SITUATION_METHODS:
            top = r[m][:10]
            lines.append(f"   {m} top 10: {[f'{a} {c}' for c, a in top]}")
        if r["detail"].get("bullets"):
            lines.append("   distilled bullets:")
            for b in r["detail"]["bullets"]:
                lines.append(f"     [{b['categoria']}] {b['testo']}")
        if r["detail"].get("questions"):
            lines.append("   questions:")
            for q in r["detail"]["questions"]:
                lines.append(f"     - {q}")
        if r["detail"].get("clean_excluded_documents"):
            lines.append("   FUSION_CLEAN excluded: " + str(r["detail"]["clean_excluded_documents"]))
        if r["detail"].get("timeline"):
            lines += _timeline_lines(case["id"], r["detail"])
            lines.append("     (x = relevance 0, discarded)")
        if r["detail"].get("llm_relevance_scores"):
            lines.append("   LLM relevance scores: " + str(r["detail"]["llm_relevance_scores"]))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--workers", type=int, default=4, help="cases run in parallel")
    parser.add_argument("--vector-index", default="section_embeddings_v2")
    parser.add_argument("--k", type=int, default=20, help="top-k retrieved per call; also the second recall@")
    parser.add_argument("--out", default="exp_distill_runs")
    parser.add_argument("--situation-index", default="situation_embeddings_v2")
    parser.add_argument("--exclude-docs", help="extra regex: documents whose name matches it are cut from FUSION_CLEAN")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    from src.rag import embedding_config
    print(f"embedding query mode: {embedding_config.QUERY_MODE}")
    if not embedding_config.text_mode() and args.vector_index.endswith("_v2"):
        print(f"WARNING: EMBEDDING_QUERY_MODE={embedding_config.QUERY_MODE} sends token IDs, but {args.vector_index} "
              "holds plain-text vectors -- every vector search except SITUATION's is comparing gibberish. "
              "Run with EMBEDDING_QUERY_MODE=text.", flush=True)
    global _situation_embedder
    from scripts.build_situation_index import plain_text_embedder
    _situation_embedder = plain_text_embedder()
    exclude_re = (re.compile(_NOT_A_LAW_RE.pattern + "|" + args.exclude_docs, re.IGNORECASE) if args.exclude_docs
                  else _NOT_A_LAW_RE)

    carried = {c["id"]: c for c in json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]}
    cases = [carried[cid] for cid in _CARRIED_OVER if cid in carried]
    if args.only:
        wanted = {c.strip() for c in args.only.split(",")}
        cases = [c for c in cases if c["id"] in wanted]
    if not cases:
        parser.error("no cases selected")

    from src.rag.main import NEO4J_DATABASE, driver

    warned = set()
    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_case, driver, NEO4J_DATABASE, case, args.vector_index, args.k, warned,
                               situation_index=args.situation_index, exclude_re=exclude_re): case
                  for case in cases}
        for future in as_completed(futures):
            case = futures[future]
            print(f"... {case['id']} done", flush=True)
            results[case["id"]] = future.result()
            print("\n".join(_timeline_lines(case["id"], results[case["id"]]["detail"])), flush=True)

    table = _table_lines(results, cases, args.k)
    detail = _detail_lines(results, cases, args.k)
    print("\n".join(table))
    print("\n".join(detail))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "table.txt").write_text("\n".join(table) + "\n" + "\n".join(detail) + "\n", encoding="utf-8")
    (out_dir / "results.json").write_text(
        json.dumps({"vector_index": args.vector_index, "k": args.k,
                   "cases": {c["id"]: c["expected"] for c in cases}, "results": results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\nWrote {out_dir / 'table.txt'} and {out_dir / 'results.json'}. Nothing was written to the database.")


if __name__ == "__main__":
    main()
