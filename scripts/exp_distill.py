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
  RULINGS            the articles cited across the top rulings (document_
                      type='interpretation') by vector similarity to the
                      raw text and to the distilled bullets, ranked by how
                      many of those rulings cite them.

...and four composites, reported separately with their own candidate
count (and, for SELECTED, noise -- non-expected articles kept):

  UNION              CURRENT + BASELINE + DISTILL, deduped (first
                      occurrence wins, in that priority order).
  UNION_NEIGHBOURS    UNION plus each article's numeric neighbours (N-1,
                      N+1, N-bis for a bare number) as low-priority
                      additions -- not "same Capo" (no chapter/title
                      property exists on a Section to group by), see the
                      comment by _NEIGHBOUR_RE.
  UNION_RULINGS      UNION plus RULINGS.
  SELECTED           UNION's real article text, through the live
                      applicability check (dg._filter_applicable_articles)
                      -- the same check run_case_analysis_pipeline applies
                      today, standalone and read-only here.

Case set: sempronio, sempronio_no_hint, p1, p2, p3, c1, c2, c3 (from
tests/eval_narrative_cases.json) plus four new long narrative cases (l1-l4,
defined below). cp_short, cc_short, q_long and dottrina are deliberately
left out of this experiment -- they stay in tests/eval_narrative.py's own
regression run.

2026-10-18 data fixes: l2's "guaribile in 40 giorni" -> "60 giorni" (583
c.p.'s aggravation needs MORE than 40 days, so 40 itself didn't actually
support the expected article). c3's expected dropped "art. 5 l392" -- see
tests/eval_narrative_cases.json's _about for why (L. 392/1978 isn't in the
corpus).
"""

import argparse
import json
import logging
import os
import re
import sys
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

# --- The 8 cases carried over from eval_narrative_cases.json ---------------
_CARRIED_OVER = ["sempronio", "sempronio_no_hint", "p1", "p2", "p3", "c1", "c2", "c3"]

# --- Four new long narrative cases, code-aware expected articles -----------
_NEW_LONG_CASES = [
    {
        "id": "l1",
        "question": (
            "Tizio, 34 anni, disoccupato, trascorre le giornate tra il bar sotto casa e la palestra. Un "
            "pomeriggio di marzo telefona a Maria, 81 anni, vedova, che vive sola a Ferrara e ha una nipote "
            "che studia a Bologna. Tizio si presenta come “il dottor Rossi dell’ufficio antifrode "
            "della sua banca”, le spiega con tono cordiale che sul suo conto sono stati rilevati "
            "movimenti sospetti e che, per bloccarli, deve comunicargli il codice utente e il codice che "
            "riceverà via SMS. Maria, preoccupata, glieli detta. Con quei codici Tizio accede al suo "
            "home banking da un internet point e dispone due bonifici per complessivi 14.500 euro verso un "
            "conto intestato a un prestanome. La nipote se ne accorge una settimana dopo. Il candidato, "
            "assunte le vesti del legale della persona offesa, individui i reati configurabili e le "
            "circostanze rilevanti."
        ),
        "expected": [{"code": "cp", "article": a} for a in ["640", "640-ter", "494", "61"]],
    },
    {
        # "guaribile in 60 giorni" (originally 40): 583 c.p.'s aggravation
        # requires a healing time of MORE than 40 days -- 40 itself doesn't
        # trigger it, so the original wording made the expected article wrong.
        "id": "l2",
        "question": (
            "Caio, dopo una serata passata in discoteca con alcuni amici che lo hanno lasciato senza "
            "passaggio, decide di tornare a piedi. Passando davanti a una villetta con le luci spente, "
            "forza la porta finestra del retro con un cacciavite ed entra. In camera da letto riempie uno "
            "zaino con gioielli e due orologi. Il proprietario, Sempronio, rientrato prima del previsto da "
            "una cena, lo sorprende nel corridoio; Caio, per guadagnare la fuga con la refurtiva, lo spinge "
            "con forza contro il mobile dell’ingresso. Sempronio riporta una frattura del polso "
            "guaribile in 60 giorni. Caio viene fermato poco dopo da una pattuglia con lo zaino. Il "
            "candidato, assunte le vesti del difensore di Caio, qualifichi i fatti e indichi la linea "
            "difensiva."
        ),
        "expected": [{"code": "cp", "article": a} for a in ["628", "624-bis", "582", "583"]],
    },
    {
        "id": "l3",
        "question": (
            "La signora Bianchi, proprietaria di un appartamento al terzo piano di un condominio del 1970 a "
            "Torino, nota a novembre macchie di umidità sul soffitto del bagno, che peggiorano dopo ogni "
            "pioggia. Ha appena rifatto la cucina e ha in programma di affittare l’immobile a studenti. "
            "Un idraulico da lei incaricato accerta che l’acqua proviene dalla colonna di scarico "
            "condominiale, deteriorata. Scrive tre volte all’amministratore, che risponde di dover "
            "attendere l’assemblea di primavera. Nel frattempo il controsoffitto crolla in parte e "
            "l’inquilino che doveva entrare a gennaio rinuncia. Quali azioni può intraprendere la "
            "signora Bianchi, nei confronti di chi e per quali danni?"
        ),
        "expected": [{"code": "cc", "article": a} for a in ["2051", "1117", "1130", "2043"]],
    },
    {
        "id": "l4",
        "question": (
            "Marco e Giulia sono sposati da dodici anni e hanno due figli di 7 e 10 anni; vivono nella casa "
            "di proprietà di Giulia a Bari, ereditata dai genitori. Marco, dirigente commerciale, viaggia "
            "spesso per lavoro. A settembre Giulia scopre, leggendo alcuni messaggi sul tablet di famiglia, "
            "che Marco ha da un anno una relazione con una collega; dopo una lite violenta Marco lascia la "
            "casa e si trasferisce in un residence. I figli frequentano la scuola vicino casa e il calcio il "
            "martedì e il giovedì. Giulia vuole separarsi, ottenere che i figli restino con lei e un "
            "contributo economico. Il candidato, assunte le vesti del legale di Giulia, illustri gli "
            "strumenti e le domande da proporre."
        ),
        "expected": [{"code": "cc", "article": a} for a in ["151", "156", "337-ter", "337-sexies"]],
    },
]

METHODS = ["baseline", "distill", "distill_questions", "current", "rulings"]
# Reported separately from METHODS: each is a composite built from the ones
# above, not an independent retrieval method, so it gets its own section
# in the report instead of sharing the recall@10/@k table.
UNION_METHODS = ["union", "union_neighbours", "union_rulings", "selected"]

_DISTILL_SYSTEM = (
    "Sei un avvocato italiano. Riscrivi il caso che segue come fatti giuridici neutri, in 5-8 punti brevi, "
    "uno per ciascuna di queste categorie quando il caso la prevede (saltala se non si applica): "
    "\"soggetti e ruoli\", \"condotta\", \"elementi sull'intenzione\", \"evento/esito\", \"nesso causale\", "
    "\"condotta successiva\", \"posizione processuale\", \"cosa si chiede\"; per i fatti penali aggiungi, "
    "se pertinente, \"circostanze\" (per esempio: vittima in condizione di minorata difesa o particolarmente "
    "vulnerabile, provocazione, danno riparato, recidiva, numero di persone coinvolte). Per ciascun punto "
    "usa il termine giuridico tipico con cui il codice descrive quella condotta, quell'evento o quello "
    "stato soggettivo, non una parafrasi generica: per esempio, se i fatti lo giustificano, \"artifizi o "
    "raggiri\" (non \"inganno\"), \"induzione in errore\", \"violenza o minaccia\", \"impossessamento\", "
    "\"sottrazione\", \"custodia\" o \"detenzione\" della cosa, \"dolo\" o \"colpa\" (con la forma specifica: "
    "negligenza, imprudenza, imperizia), \"inadempimento\", \"mora\", \"vizio della cosa\", \"nesso di "
    "causalità\". Elimina i dettagli irrilevanti (nomi propri non necessari, descrizioni di contesto, "
    "aggettivi, orari e luoghi non giuridicamente rilevanti). Non usare MAI un numero di articolo né il "
    "nome di un codice o di una legge: descrivi solo i fatti, con il vocabolario giuridico italiano. "
    "Restituisci SOLO un array JSON di oggetti {\"categoria\": una delle etichette sopra, \"testo\": il "
    "punto, breve}."
)

_DISTILL_QUESTIONS_SYSTEM = (
    "Dati questi fatti giuridici, scrivi da 3 a 5 domande giuridiche neutre che i fatti pongono (per "
    "esempio: \"quale reato integra un'azione violenta non diretta a uccidere da cui deriva la morte?\"). "
    "Non usare MAI un numero di articolo né il nome di un codice o di una legge. Restituisci SOLO un array "
    "JSON di stringhe."
)

_ABBR_TO_EVAL_CODE = {"c.p.": "cp", "c.c.": "cc", "c.p.c.": "cpc", "c.p.p.": "cpp"}
_EVAL_CODE_TO_SEARCH_NAME = {
    "cp": "c.p.", "cc": "c.c.", "cpc": "c.p.c.", "cpp": "c.p.p.",
    "consumo": "Codice del Consumo", "l392": "L. 392/1978",
}
# Numeric-only neighbour expansion: N-1, N+1, and N's own "-bis" variant for
# a bare-numbered article. "Same Capo" grouping (what item 5 actually asked
# for) isn't available -- Section nodes carry no chapter/title property in
# this corpus, only the article number in s.name -- so this is a narrower
# neighbourhood by design, not an oversight, and kept deliberately small (3
# lookups per article, not every suffix on both sides) given the latency
# budget (round 5, item 3): an already-slow pipeline doesn't need another
# multiplier.
_NEIGHBOUR_RE = re.compile(r"^(\d+)$")


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


def _merge_best_rank(*ranked_lists):
    """Union of refs across the given ranked lists, ordered by the best
    (lowest) rank any one of them gives it."""
    best = {}
    for ranked in ranked_lists:
        for rank, ref in enumerate(ranked, start=1):
            if ref not in best or rank < best[ref]:
                best[ref] = rank
    return sorted(best, key=lambda ref: best[ref])


def _merge_by_hits(*ranked_lists):
    """Union of refs, ordered by (how many of the given ranked lists
    contain it, descending), then by the best rank among those that do --
    a ref several bullets/questions independently retrieve outranks one
    only a single call happened to surface, even at a better rank."""
    hits, best_rank = {}, {}
    for ranked in ranked_lists:
        for rank, ref in enumerate(ranked, start=1):
            hits[ref] = hits.get(ref, 0) + 1
            if ref not in best_rank or rank < best_rank[ref]:
                best_rank[ref] = rank
    return sorted(hits, key=lambda ref: (-hits[ref], best_rank[ref]))


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


def _neighbour_numbers(article):
    """[N-1, N+1, "N-bis"] for a bare-numbered article, else [] -- see the
    module-level comment by _NEIGHBOUR_RE for why this is numeric-only and
    deliberately narrow."""
    m = _NEIGHBOUR_RE.match(article)
    if not m:
        return []
    n = int(m.group(1))
    return [str(x) for x in (n - 1, n + 1) if x > 0] + [f"{n}-bis"]


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


def _rulings_search(session, text, index, k, warned):
    """Up to k ruling (document_type='interpretation') Section rows closest
    to `text` by vector similarity, with their own plain_text (to read which
    articles they cite). The vector index isn't scoped to rulings alone, so
    a plain top-k would mostly come back statute sections that happen to
    read similarly to the facts; asking for a wider raw_k and filtering by
    document_type afterwards is still one query, just a bigger one."""
    try:
        embedding = list(embedding_model.embed_query(text))
    except Exception as exc:
        logging.warning("embedding failed: %s", exc)
        return []
    raw_k = max(k * 10, 200)
    try:
        return session.run(
            "CALL db.index.vector.queryNodes($index, $raw_k, $embedding) YIELD node, score "
            "WHERE 'Section' IN labels(node) AND node.plain_text IS NOT NULL AND node.plain_text <> '' "
            "MATCH (d:Document)-[:CONTAINS]->(node) WHERE d.document_type = 'interpretation' "
            "RETURN node.plain_text AS plain_text, score ORDER BY score DESC LIMIT $k",
            index=index, raw_k=raw_k, k=k, embedding=embedding,
        ).data()
    except Exception as exc:
        if index not in warned:
            warned.add(index)
            logging.warning("vector search on index %r failed (does it exist?): %s", index, exc)
        return []


def _rulings_refs(session, text, bullets_joined, vector_index, k, warned):
    """Articles cited across the top rulings for `text` and (if given)
    `bullets_joined`, ranked by how many of those rulings cite them, then by
    the best similarity rank among those that do."""
    rows = _rulings_search(session, text, vector_index, k, warned)
    if bullets_joined:
        rows = rows + _rulings_search(session, bullets_joined, vector_index, k, warned)
    counts, best_rank = {}, {}
    for rank, row in enumerate(rows, start=1):
        for number, abbr in dg._refs_in(row.get("plain_text") or ""):
            code = _code_key(abbr)
            if not code:
                continue
            ref = (code, number)
            counts[ref] = counts.get(ref, 0) + 1
            if ref not in best_rank or rank < best_rank[ref]:
                best_rank[ref] = rank
    return sorted(counts, key=lambda ref: (-counts[ref], best_rank[ref]))


# --- The methods ----------------------------------------------------

def _distill_bullets(text):
    raw = dg._chat([SystemMessage(content=_DISTILL_SYSTEM), HumanMessage(content=text[:6000])],
                   max_tokens=700, temperature=0)
    parsed = dg.parse_json_reply(raw)
    if not isinstance(parsed, list):
        return []
    return [{"categoria": str(b.get("categoria") or ""), "testo": str(b.get("testo") or "").strip()}
            for b in parsed if isinstance(b, dict) and str(b.get("testo") or "").strip()][:8]


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


def _selected_refs(session, documents, text, union_refs):
    """UNION's refs, fetched for their real text and passed through the live
    applicability check (dg._filter_applicable_articles) -- the same check
    run_case_analysis_pipeline applies today, read-only and standalone here.
    Order preserved from union_refs, just with the dropped ones removed."""
    fetch_refs = [r for r in (_to_fetch_ref(ref) for ref in union_refs) if r]
    if not fetch_refs:
        return []
    rows = dg._fetch_any(session, documents, fetch_refs)
    kept_rows = dg._filter_applicable_articles(text, "", rows)
    kept = set()
    for row in kept_rows:
        article = row.get("_article")
        if article:
            code = _code_key(article[1])
            if code:
                kept.add((code, article[0]))
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

        try:
            rulings = _rulings_refs(session, text, joined, vector_index, k, warned)
        except Exception as exc:
            logging.warning("%s: rulings search failed: %s", case["id"], exc)
            rulings = []

        union = _dedupe_ordered(current, baseline, distill)

        neighbours = []
        for code, article in union:
            for neighbour in _neighbour_numbers(article):
                neighbours.append((code, neighbour))
        union_neighbours = _dedupe_ordered(union, neighbours)
        union_rulings = _dedupe_ordered(union, rulings)

        try:
            selected = _selected_refs(session, documents, text, union)
        except Exception as exc:
            logging.warning("%s: applicability check on UNION failed: %s", case["id"], exc)
            selected = union

    return {"baseline": baseline, "distill": distill, "distill_questions": distill_questions,
            "current": current, "rulings": rulings, "union": union,
            "union_neighbours": union_neighbours, "union_rulings": union_rulings,
            "selected": selected, "union_candidate_count": len(union), "detail": detail}


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


def _table_lines(results, cases, k):
    lines = [f"recall@10 and @{k}, found/expected, code-aware:", ""]
    lines += _recall_table(results, cases, k, METHODS)
    lines.append("")
    lines.append("UNION (current + baseline + distill, deduped), its numeric-neighbour and rulings-added "
                "variants, and SELECTED (UNION through the live applicability check) -- recall, plus "
                "candidate count and SELECTED's noise (non-expected articles kept):")
    lines.append("")
    lines += _recall_table(results, cases, k, UNION_METHODS)
    lines.append("")
    header = f"{'case':<18s}{'union candidates':<20s}{'selected candidates':<22s}{'selected noise@' + str(k):<18s}"
    lines.append(header)
    for case in cases:
        r = results[case["id"]]
        noise = _noise(r["selected"], case["expected"], k)
        lines.append(f"{case['id']:<18s}{r['union_candidate_count']:<20d}{len(r['selected']):<22d}{noise:<18d}")
    return lines


def _detail_lines(results, cases, k):
    lines = []
    for case in cases:
        r = results[case["id"]]
        lines.append(f"\n## {case['id']}")
        lines.append(f"   expected: {case['expected']}")
        for m in METHODS + UNION_METHODS:
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
    cases = [carried[cid] for cid in _CARRIED_OVER if cid in carried] + _NEW_LONG_CASES
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
