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
"""

import argparse
import json
import logging
import os
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


# --- Retrieval ---------------------------------------------------------

def _vector_search(session, text, index, k, warned):
    try:
        embedding = list(embedding_model.embed_query(text))
    except Exception as exc:
        logging.warning("embedding failed: %s", exc)
        return []
    try:
        return session.run(
            "CALL db.index.vector.queryNodes($index, $k, $embedding) YIELD node, score "
            "WHERE 'Section' IN labels(node) AND node.plain_text IS NOT NULL AND node.plain_text <> '' "
            "OPTIONAL MATCH (d:Document)-[:CONTAINS]->(node) "
            "RETURN node.name AS name, d.name AS doc_name, score ORDER BY score DESC",
            index=index, k=k, embedding=embedding,
        ).data()
    except Exception as exc:
        if index not in warned:
            warned.add(index)
            logging.warning("vector search on index %r failed (does it exist?): %s", index, exc)
        return []


def _fulltext_search(session, text, k):
    q = _sanitize_fulltext_query(text)
    if not q:
        return []
    return session.run(
        "CALL db.index.fulltext.queryNodes('section_fulltext', $q, {limit: $k}) YIELD node, score "
        "OPTIONAL MATCH (d:Document)-[:CONTAINS]->(node) "
        "RETURN node.name AS name, d.name AS doc_name, score ORDER BY score DESC",
        q=q, k=k,
    ).data()


def _retrieve(session, text, index, k, warned):
    return _ranked_refs(_vector_search(session, text, index, k, warned) + _fulltext_search(session, text, k))


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
    analysis = dg._analyse_case_v2(text, "it")
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


def _run_case(driver, database, case, vector_index, k, warned):
    text = case["question"]
    detail = {}
    with driver.session(database=database) as session:
        from src.rag.doc_lookup import _fetch_doc_names
        documents = _fetch_doc_names(driver, database)

        baseline = _merge_best_rank(
            _ranked_refs(_vector_search(session, text, vector_index, k, warned)),
            _ranked_refs(_fulltext_search(session, text, k)),
        )

        try:
            bullets = _distill_bullets(text)
        except Exception as exc:
            logging.warning("%s: distill (bullets) failed: %s", case["id"], exc)
            bullets = []
        detail["bullets"] = bullets
        joined = " ".join(b["testo"] for b in bullets)
        bullet_lists = [_retrieve(session, b["testo"], vector_index, k, warned) for b in bullets]
        if joined:
            bullet_lists.append(_retrieve(session, joined, vector_index, k, warned))
        distill = _merge_by_hits(*bullet_lists) if bullet_lists else []

        try:
            questions = _distill_questions(bullets)
        except Exception as exc:
            logging.warning("%s: distill (questions) failed: %s", case["id"], exc)
            questions = []
        detail["questions"] = questions
        question_lists = [_retrieve(session, q, vector_index, k, warned) for q in questions]
        distill_questions = _merge_by_hits(*bullet_lists, *question_lists) if (bullet_lists or question_lists) \
            else []

        try:
            current, analysis = _run_current(session, driver, database, text)
        except Exception as exc:
            logging.warning("%s: CURRENT (case-analysis v2) failed: %s", case["id"], exc)
            current, analysis = [], {}
        detail["current_analysis"] = analysis

        union = _dedupe_ordered(current, baseline, distill)
        union_rows = _fetch_union_rows(session, documents, union)
        texts = _union_texts(union_rows)

        try:
            selected = _selected_refs(text, union_rows, union)
        except Exception as exc:
            logging.warning("%s: applicability check on UNION failed: %s", case["id"], exc)
            selected = union

        signal_counts = _signal_counts(union, current, baseline, distill)
        latency = {}

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

    detail["llm_relevance_scores"] = {_ref_label(ref): score for ref, score in llm_relevance_scores.items()}
    return {"baseline": baseline, "distill": distill, "distill_questions": distill_questions,
            "current": current, "union": union, "selected": selected,
            "rerank_bullets": rerank_bullets, "rerank_raw": rerank_raw,
            "llm_relevance": llm_relevance, "fusion_baseline": fusion_baseline,
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


def _table_lines(results, cases, k):
    lines = [f"recall@10 and @{k}, found/expected, code-aware:", ""]
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
        for m in METHODS + UNION_METHODS + RANKING_METHODS:
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

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
        futures = {pool.submit(_run_case, driver, NEO4J_DATABASE, case, args.vector_index, args.k, warned): case
                  for case in cases}
        for future in as_completed(futures):
            case = futures[future]
            print(f"... {case['id']} done", flush=True)
            results[case["id"]] = future.result()

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
