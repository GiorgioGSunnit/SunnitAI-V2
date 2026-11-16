#!/usr/bin/env python3
"""v60 (F): the lawyer's rubric on final answers -- ten rows, each scored 0-1 by the model on the answer and the case
text, then a table. For unseen cases the PO supplies; the benchmark cases (tests/eval_narrative_cases.json) are marked
«regression only» and left out of the acceptance mean (they shaped the tables: they are no yardstick).

Inputs, either:
  --cases DIR          one case per subfolder DIR/<id>/ with case.txt (the case as the user wrote it, the upload's text
                       included) and answer.md (the final answer); or files DIR/<id>.case.txt + DIR/<id>.answer.md
  --transcript FILE    an interactive transcript (diagnose_long --interactive-sim's interactive.md, or the server's run
                       directory): each case's answer, with --case-texts JSON {id: case text}
  --live DIR           v60i (10): the live transcripts as the chat shows them (DIR/<id>.txt: round 1, then the parere):
                       each one's parere, from its first section on; the case text from --case-texts, else
                       DIR/<id>.case.txt, else none (the rows that need it -- the invented facts -- are then the model's
                       guess, flagged in the table)

The table: one row per case, then for each case its rows with our number and an empty column for the PO's (ours first,
the PO's after: the PO scores without seeing ours being the point).

  python scripts/rubric_score.py --cases ~/po_cases --out rubric.md
  python scripts/rubric_score.py --transcript diagnose_reports/runs/v60-all/interactive.md --case-texts cases.json
  python scripts/rubric_score.py --live diagnose_reports/live --out diagnose_reports/live/rubric.md

v71: the scorer defaults to «non risolto» -- a point is resolved only with a sentence quoted from the new parere, found
there verbatim; the thesis points (the critique file's tesi_attesa: his §1 and §7.2) are scored from the soluzioni
table (which thesis is principale, which is labelled minoritaria); the model's own verdicts are logged, never counted.

v70 (4), --critique: the lawyer's critiques are the yardstick, not test fixtures. Each answer is scored against the
critique of the same case (--critique DIR holding <id>.critique.json / .md / .txt / .docx, or one FILE with --id). A
.critique.json (scripts/critique_input.py: his points verbatim, each with a fixed id, his expected strategy, the case's
text) is scored point by point by its ids -- the same points every batch, so --baseline matches them; a free text has
its points listed by the model. Each point: resolved, partly, or the error repeated (a merit: kept or lost; his
expected strategy: adopted, partly or absent); and three scores, 0-1 -- tesi (the thesis and the realistic outcome the critique asks for), errori di categoria (none of the
category errors it names: a wrong qualification, a circumstance without its requisites, a norm out of place, an outcome
flattering the client), coerenza (no contradiction of the kind it names: a thesis denying a proven element, conclusions
against the assessment). --json-out keeps the result; --baseline <that JSON> prints the deltas, and every point that
was resolved and is repeated now (a regression between batches: a score can rise while a section falls to zero).

  python scripts/rubric_score.py --transcript <run>/interactive.md --critique docs/lawyer/critiques --json-out v70.json
  python scripts/rubric_score.py --transcript <run>/interactive.md --critique docs/lawyer/critiques --baseline v70.json
(a structured critique carries its case's text: no --case-texts needed)

Needs the model (the server's .env): one call per case.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# (key, the table's column, what 1 means -- the model reads it)
ROWS = (
    ("strumento", "strumento/atto", "indica lo strumento o l'atto giudiziario giusto per l'obiettivo del cliente "
                                    "(azione, ricorso, querela, opposizione, difesa…)"),
    ("qualificazione", "qualificazione", "qualifica correttamente i fatti (l'istituto o il reato giusto)"),
    ("procedibilita", "procedibilità e termini", "indica correttamente procedibilità, termini e decadenze rilevanti "
                                                 "(o dice che vanno verificati quando il caso non li permette)"),
    ("norme_fuori_luogo", "norme fuori luogo", "nessuna norma citata fuori luogo (1 = nessuna; 0 = più d'una "
                                               "palesemente estranea)"),
    ("fatti_inventati", "fatti inventati", "nessun fatto che il caso non dice è affermato come vero (1 = nessuno)"),
    ("strategia", "strategia principale/subordinata", "propone una linea principale e una subordinata coerenti"),
    ("controparte", "controparte", "anticipa la linea probabile della controparte e la contrasta"),
    ("prova", "prova", "indica i fatti da provare e le prove da raccogliere"),
    ("contraddizioni", "contraddizioni", "nessuna contraddizione interna (1 = nessuna)"),
    ("ruolo_cliente", "ruolo del cliente", "scrive dal punto di vista della parte giusta (il cliente indicato o, se "
                                           "non indicato, lo dichiara)"),
)
SYSTEM = (
    "Sei un avvocato senior che valuta il parere scritto da un collega su un caso. Per ciascuna voce della griglia dai un "
    "punteggio da 0 a 1 (0, 0.25, 0.5, 0.75 o 1) e una motivazione di una frase. Valuta solo ciò che il parere scrive, "
    "sul caso dato. Rispondi SOLO con un JSON {chiave: {\"score\": numero, \"why\": frase}} con queste chiavi:\n"
    + "\n".join(f"- {key}: 1 = {meaning}" for key, _, meaning in ROWS))
SCORE_TOKENS = 1400


def benchmark_ids() -> set:
    try:
        data = json.loads((ROOT / "tests" / "eval_narrative_cases.json").read_text(encoding="utf-8"))
        return {str(c.get("id")) for c in data.get("cases") or [] if c.get("id")}
    except (OSError, ValueError):
        return set()


def from_dir(folder: str) -> Dict[str, Tuple[str, str]]:
    """{id: (case text, answer)}."""
    out = {}
    base = Path(folder)
    for sub in sorted(p for p in base.iterdir() if p.is_dir()):
        case, answer = sub / "case.txt", sub / "answer.md"
        if case.is_file() and answer.is_file():
            out[sub.name] = (case.read_text(encoding="utf-8"), answer.read_text(encoding="utf-8"))
    for case in sorted(base.glob("*.case.txt")):
        cid = case.name[:-len(".case.txt")]
        answer = base / f"{cid}.answer.md"
        if answer.is_file():
            out[cid] = (case.read_text(encoding="utf-8"), answer.read_text(encoding="utf-8"))
    return out


def from_transcript(path: str, texts: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
    import answer_lint
    out = {}
    for cid, case in answer_lint.read_cases(path).items():
        if case.get("answer"):
            out[cid] = (texts.get(cid, ""), case["answer"])
    return out


# v60i (10): where the parere starts in a live transcript -- the old pipeline's «⚖️ BOZZA …» header, or the first
# section of either answer («1. Risposte ai quesiti», «1. Risposta e strategia»)
_PARERE_START_RE = re.compile(r"(?m)^\s*(?:⚖️|\*{0,2}1\.\s+Rispost[ae]\b)")


def answer_of(transcript: str) -> str:
    """The parere of a live transcript: from where it starts (_PARERE_START_RE) to the end; "" when none."""
    m = _PARERE_START_RE.search(transcript or "")
    return transcript[m.start():].strip() if m else ""


def from_live(folder: str, texts: Optional[Dict[str, str]] = None) -> Dict[str, Tuple[str, str]]:
    """{id: (case text, parere)} of DIR/<id>.txt; the case text from texts, else DIR/<id>.case.txt, else ""."""
    out = {}
    base = Path(folder)
    for path in sorted(base.glob("*.txt")):
        if path.name.endswith(".case.txt"):
            continue
        cid = path.stem
        answer = answer_of(path.read_text(encoding="utf-8"))
        if not answer:
            continue
        case = (texts or {}).get(cid)
        if case is None and (base / f"{cid}.case.txt").is_file():
            case = (base / f"{cid}.case.txt").read_text(encoding="utf-8")
        out[cid] = (case or "", answer)
    return out


def po_tables(scored: Dict[str, Dict[str, Dict[str, Any]]], no_case: Sequence[str] = ()) -> str:
    """v60i (10): per case, each row with our number and an empty column for the PO's."""
    out = ["", "## Per case: our number, then the PO's", ""]
    for cid, rows in scored.items():
        out.append(f"**{cid}**" + (" (no case text: «fatti inventati» is the model's guess)" if cid in no_case else ""))
        out += ["", "| row | ours | PO |", "|---|---|---|"]
        out += [f"| {col} | {'—' if rows[k]['score'] is None else format(rows[k]['score'], 'g')} |  |"
                for k, col, _ in ROWS]
        out.append("")
    return "\n".join(out)


def clean(raw: Any) -> Dict[str, Dict[str, Any]]:
    """The model's reply as {key: {"score": 0-1 or None, "why"}} for every row."""
    out = {}
    for key, _, _ in ROWS:
        item = raw.get(key) if isinstance(raw, dict) else None
        score = None
        why = ""
        if isinstance(item, dict):
            try:
                score = min(1.0, max(0.0, float(item.get("score"))))
            except (TypeError, ValueError):
                score = None
            why = " ".join(str(item.get("why") or "").split())[:200]
        out[key] = {"score": score, "why": why}
    return out


# --- v70 (4): scored against the lawyer's critique ------------------------------------------------------------------------

CRITIQUE_SCORES = (
    ("tesi", "tesi", "il parere adotta la tesi che la critica indica come corretta e un esito realistico (non un esito "
                     "favorevole che i fatti non sostengono)"),
    ("errori_di_categoria", "errori di categoria", "il parere non ripete nessuno degli errori di categoria che la "
                                                   "critica segnala (qualificazione sbagliata, circostanze senza i loro "
                                                   "requisiti, norme fuori luogo, istituti confusi)"),
    ("coerenza", "coerenza", "il parere non ha contraddizioni del tipo che la critica segnala (una tesi che nega un "
                             "elemento provato, conclusioni contrarie alla valutazione, sezioni che si smentiscono)"),
)
POINT_OUTCOMES = ("risolto", "parziale", "ripetuto", "non pertinente")
# the outcomes by the point's category: good, partial, bad (and none)
OUTCOMES_BY_CATEGORY = {"pregio": ("mantenuto", "", "perso"), "attesa": ("adottato", "parziale", "assente")}
DEFAULT_OUTCOMES = ("risolto", "parziale", "ripetuto")
NOT_RESOLVED = "non risolto"          # v71: every point's default -- a good outcome needs a quoted, verified sentence
EVIDENCE_MIN_CHARS = 20
ALL_OUTCOMES = set(POINT_OUTCOMES) | {x for v in OUTCOMES_BY_CATEGORY.values() for x in v if x}


def outcomes(categoria: str) -> tuple:
    return OUTCOMES_BY_CATEGORY.get(categoria, DEFAULT_OUTCOMES)


def grade(point: Dict[str, Any]) -> str:
    """good | partial | bad | none -- the point's outcome read in its category's terms."""
    good, partial, bad = outcomes(point.get("categoria") or "")
    if point.get("esito") == NOT_RESOLVED:
        return "bad"
    return {good: "good", bad: "bad"}.get(point.get("esito"), "partial" if partial and point.get("esito") == partial
                                          else "none")
CRITIQUE_SYSTEM = (
    "Sei un avvocato senior. Ricevi la CRITICA che un avvocato ha scritto su un parere precedente per lo stesso caso, il "
    "CASO e un NUOVO PARERE. La critica è il metro: elenca ogni suo punto (una carenza o un errore che il collega ha "
    "indicato) e per ciascuno di' se il nuovo parere lo risolve («risolto»), solo in parte («parziale»), ripete l'errore "
    "(«ripetuto») o se il punto non riguarda il nuovo parere («non pertinente»), con una frase di motivazione. Poi dai "
    "tre punteggi da 0 a 1 (0, 0.25, 0.5, 0.75 o 1), ciascuno con una frase:\n"
    + "\n".join(f"- {key}: 1 = {meaning}" for key, _, meaning in CRITIQUE_SCORES)
    + "\nValuta solo ciò che il nuovo parere scrive. Rispondi SOLO con un JSON: {\"punti\": [{\"punto\": \"…\", "
    "\"categoria\": \"tesi|categoria|coerenza|altro\", \"esito\": \"risolto|parziale|ripetuto|non pertinente\", "
    "\"why\": \"…\"}], " + ", ".join(f'\"{k}\": {{\"score\": numero, \"why\": frase}}' for k, _, _ in CRITIQUE_SCORES)
    + "}")
CRITIQUE_POINTS_SYSTEM = (
    "Sei un avvocato senior. Ricevi i PUNTI della critica che un avvocato ha scritto su un parere precedente per lo "
    "stesso caso (ciascuno con il suo id e la sua categoria), il CASO e un NUOVO PARERE. La critica è il metro. Per "
    "ciascun punto, con una frase di motivazione, l'esito sul nuovo parere: per le categorie tesi, categoria, coerenza, "
    "processuale, mancanza: «risolto» (il nuovo parere non commette l'errore o contiene ciò che mancava), «parziale», "
    "«ripetuto» o «non pertinente»; per «pregio»: «mantenuto» o «perso»; per «attesa» (la strategia che l'avvocato "
    "consiglia): «adottato», «parziale» o «assente». Per ogni esito positivo o parziale riporta in «citazione» la frase "
    "esatta del nuovo parere che lo dimostra, copiata parola per parola: senza una citazione che si trova nel parere "
    "l'esito non conta. Poi tre punteggi da 0 a 1 (0, 0.25, 0.5, 0.75 o 1), ciascuno con "
    "una frase:\n" + "\n".join(f"- {key}: 1 = {meaning}" for key, _, meaning in CRITIQUE_SCORES)
    + "\nValuta solo ciò che il nuovo parere scrive. Rispondi SOLO con un JSON: {\"punti\": {\"<id>\": {\"esito\": "
    "\"…\", \"citazione\": \"…\", \"why\": \"…\"}, …}, " + ", ".join(f'\"{k}\": {{\"score\": numero, \"why\": frase}}' for k, _, _ in CRITIQUE_SCORES)
    + "}")
CRITIQUE_TOKENS = 2400
CRITIQUE_POINT_TOKENS = 90             # per point, on top of CRITIQUE_TOKENS
CRITIQUE_SUFFIXES = (".critique.json", ".critique.md", ".critique.txt", ".critique.docx")


def read_text(path: Path) -> Any:
    """A critique: a .critique.json as its dict (critique_input.py), .docx through python-docx, else UTF-8 text."""
    if path.name.endswith(".critique.json"):
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() == ".docx":
        import docx
        return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    return path.read_text(encoding="utf-8")


def load_critiques(path: str, cid: str = "") -> Dict[str, Any]:
    """{id: critique}: a folder's <id>.critique.{md,txt,docx}, or one file (its id: --id, else the name before
    «.critique»)."""
    base = Path(os.path.expanduser(path))
    out = {}
    if base.is_dir():
        for p in sorted(base.iterdir()):
            suffix = next((x for x in CRITIQUE_SUFFIXES if p.name.endswith(x)), None)
            if suffix:
                out[p.name[:-len(suffix)]] = read_text(p)
    elif base.is_file():
        name = base.name.split(".critique")[0] if ".critique" in base.name else base.stem
        out[cid or name] = read_text(base)
    return out


def critique_case(critique: Any) -> str:
    """The case text a structured critique carries ("" for a free text)."""
    return str(critique.get("case") or "") if isinstance(critique, dict) else ""


def _norm(text: str) -> str:
    """For the verbatim check: no markdown emphasis, one kind of quote and apostrophe, single spaces, lower case."""
    text = re.sub(r"[*_`]+", "", str(text or ""))
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("«", '"').replace("»", '"')
    return " ".join(text.split()).lower().strip(" .;:")


def verified(quote: str, answer: str) -> bool:
    """v71: the quote is in the answer word for word (_norm), and long enough to prove something."""
    q = _norm(quote)
    return len(q) >= EVIDENCE_MIN_CHARS and q in _norm(answer)


# --- v71: the thesis points, scored from the soluzioni table --------------------------------------------------------------

_TABLE_HEAD = "| Tesi | Pro | Contro | Realismo |"


def solutions_rows(answer: str) -> List[Dict[str, Any]]:
    """The soluzioni table's rows: {tesi (the first cell), principale, minoritaria, barred}."""
    lines = (answer or "").split("\n")
    start = next((i for i, ln in enumerate(lines) if ln.strip().startswith(_TABLE_HEAD)), None)
    if start is None:
        return []
    out = []
    for ln in lines[start + 2:]:
        if not ln.strip().startswith("|"):
            break
        cell = ln.strip().strip("|").split("|")[0].strip()
        out.append({"tesi": cell, "principale": "(tesi principale)" in cell, "minoritaria": "(tesi minoritaria)" in cell,
                    "barred": "non sostenibile" in cell or "non verificabile" in cell})
    return out


def _alternatives(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The accepted main theses of a «principale» spec: its «alternative», else the spec itself."""
    return [x for x in spec.get("alternative") or [] if isinstance(x, dict)] or [spec]


def thesis_outcome(kind: str, spec: Dict[str, Any], rows: List[Dict[str, Any]]) -> Tuple[str, str]:
    """(esito, why) of a thesis point from the table. «principale»: the main row matches one of the accepted theses,
    with the label it requires («minoritaria») -- risolto; matched without the label, or an accepted thesis present but
    not main -- parziale; else non risolto. «esclusa»: the thesis the critique rejects is not the main row and, where
    it appears, is labelled -- minoritaria or non sostenibile -- risolto; else non risolto."""
    main = next((r for r in rows if r["principale"]), None)
    if not rows or main is None:
        return NOT_RESOLVED, "nessuna tabella delle soluzioni con una tesi principale"
    if kind == "esclusa":
        rx = re.compile(spec.get("pattern") or r"$^", re.IGNORECASE)
        if rx.search(main["tesi"]):
            return NOT_RESOLVED, f"la tesi principale è quella che la critica esclude: {main['tesi'][:120]}"
        bare = next((r for r in rows if rx.search(r["tesi"]) and not (r["minoritaria"] or r["barred"])), None)
        if bare:
            return NOT_RESOLVED, f"la tesi esclusa compare senza dirsi minoritaria: {bare['tesi'][:120]}"
        return "risolto", f"tesi principale: {main['tesi'][:120]}"
    accepted = _alternatives(spec)
    for alt in accepted:
        if re.search(alt.get("pattern") or r"$^", main["tesi"], re.IGNORECASE):
            if alt.get("minoritaria") and not main["minoritaria"]:
                return "parziale", f"la tesi principale ({alt.get('nome') or 'attesa'}) non è dichiarata minoritaria"
            return "risolto", f"tesi principale: {main['tesi'][:120]}"
    other = next((r for r in rows for alt in accepted if re.search(alt.get("pattern") or r"$^", r["tesi"],
                                                                    re.IGNORECASE)), None)
    if other:
        return "parziale", f"una tesi attesa c'è ma non è la principale: {other['tesi'][:120]}"
    return NOT_RESOLVED, f"nessuna tesi attesa; principale: {main['tesi'][:120]}"


def clean_critique(raw: Any, given: Optional[List[Dict[str, Any]]] = None, answer: str = "",
                   expected: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """{"points": [...], "scores": {...}, "model_scores": {...}}. v71: every point defaults to «non risolto»; a good or
    partial outcome counts only with its «citazione» found verbatim in the answer (verified); the thesis points
    (expected["punti"]) are scored from the soluzioni table (thesis_outcome); the model's verdicts are kept («modello»)
    but never counted. The scores: tesi -- the thesis points'; errori di categoria and coerenza -- the share of their
    points resolved (partial: half)."""
    raw = raw if isinstance(raw, dict) else {}
    expected = expected or {}
    thesis_points = expected.get("punti") or {}
    rows = solutions_rows(answer)
    points = []
    source = []
    if given is not None:
        got = raw.get("punti") if isinstance(raw.get("punti"), dict) else {}
        source = [(p, got.get(p["id"]) if isinstance(got.get(p["id"]), dict) else {}) for p in given]
    elif isinstance(raw.get("punti"), list):
        source = [({"punto": " ".join(str(p.get("punto")).split())[:240],
                    "categoria": str(p.get("categoria") or "altro").strip().lower()[:20]}, p)
                  for p in raw["punti"] if isinstance(p, dict) and str(p.get("punto") or "").strip()]
    for p, item in source:
        cat = p.get("categoria") or "altro"
        said = str(item.get("esito") or "").strip().lower()
        quote = " ".join(str(item.get("citazione") or "").split())
        model = {"esito": said, "citazione": quote[:300], "why": " ".join(str(item.get("why") or "").split())[:240]}
        good, partial, _ = outcomes(cat)
        if p.get("id") in thesis_points:
            esito, why = thesis_outcome(thesis_points[p["id"]], expected.get(thesis_points[p["id"]]) or {}, rows)
            how = "tabella delle soluzioni"
        elif said in (good, partial) and said:
            ok = verified(quote, answer)
            esito, why = (said, model["why"]) if ok else (NOT_RESOLVED, "nessuna citazione verificata nel parere")
            how = "citazione verificata" if ok else "citazione assente o non trovata"
        elif said == "non pertinente":
            esito, why, how = "non pertinente", model["why"], "dichiarato non pertinente"
        else:
            esito, why, how = NOT_RESOLVED, model["why"] or "nessun esito positivo dimostrato", "default"
        points.append({**({"id": p["id"]} if p.get("id") else {}), "punto": p["punto"], "categoria": cat,
                       "esito": esito, "why": why, "come": how, "modello": model})
    model_scores: Dict[str, Dict[str, Any]] = {}
    for key, _, _ in CRITIQUE_SCORES:
        item = raw.get(key) if isinstance(raw.get(key), dict) else {}
        try:
            value = min(1.0, max(0.0, float(item.get("score"))))
        except (TypeError, ValueError):
            value = None
        model_scores[key] = {"score": value, "why": " ".join(str(item.get("why") or "").split())[:240]}

    def share(selected: List[Dict[str, Any]]) -> Optional[float]:
        counted_ = [x for x in selected if grade(x) != "none"]
        if not counted_:
            return None
        return round(sum(1.0 if grade(x) == "good" else 0.5 if grade(x) == "partial" else 0.0 for x in counted_)
                     / len(counted_), 2)
    scores = {
        "tesi": {"score": share([x for x in points if x.get("id") in thesis_points]),
                 "why": "dalla tabella delle soluzioni: " + "; ".join(f"{x['id']} {x['esito']}" for x in points
                                                                    if x.get("id") in thesis_points)},
        "errori_di_categoria": {"score": share([x for x in points if x["categoria"] == "categoria"]),
                                "why": "quota dei punti «categoria» risolti con citazione verificata"},
        "coerenza": {"score": share([x for x in points if x["categoria"] == "coerenza"]),
                     "why": "quota dei punti «coerenza» risolti con citazione verificata"},
    }
    return {"points": points, "scores": scores, "model_scores": model_scores}


def critique_score(critique: Any, case_text: str, answer: str, caller=None) -> Dict[str, Any]:
    """One model call (caller: a stand-in taking (system, human), returning the parsed JSON, for tests). A structured
    critique (dict): its points by id, its own case text when none is given; a free text: the model lists the points."""
    case_text = case_text.strip() or critique_case(critique)
    given = critique.get("points") if isinstance(critique, dict) else None
    expected = critique.get("tesi_attesa") if isinstance(critique, dict) else None
    if given is not None:
        listed = "\n".join(f"[{p['id']}] ({p.get('categoria') or 'altro'}) {p['punto']}" for p in given)
        head = f"GIUDIZIO COMPLESSIVO: {critique.get('giudizio') or ''}\n\nPUNTI:\n{listed}"
        system, tokens = CRITIQUE_POINTS_SYSTEM, CRITIQUE_TOKENS + CRITIQUE_POINT_TOKENS * len(given)
    else:
        head, system, tokens = f"CRITICA:\n{str(critique).strip()}", CRITIQUE_SYSTEM, CRITIQUE_TOKENS
    human = f"{head}\n\nCASO:\n{case_text or '(testo del caso non fornito)'}\n\nNUOVO PARERE:\n{answer.strip()}"
    if caller is None:
        from langchain_core.messages import HumanMessage, SystemMessage
        from src.rag import defensive_generation as dg
        raw = dg._chat_json([SystemMessage(content=system), HumanMessage(content=human)], max_tokens=tokens,
                            temperature=0)
    else:
        raw = caller(system, human)
    return clean_critique(raw, given, answer, expected)


def _key(point: str) -> str:
    return re.sub(r"\W+", " ", point.lower()).strip()[:80]


def critique_table(results: Dict[str, Dict[str, Any]], baseline: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
    """Per case: the three scores (with the baseline's delta), the points by outcome; then each point; then the
    regressions -- points the baseline resolved that are repeated now."""
    baseline = baseline or {}
    cols = [col for _, col, _ in CRITIQUE_SCORES]
    lines = ["| case | " + " | ".join(cols) + " | risolti / mantenuti / adottati | parziali | ripetuti / persi / "
             "assenti |", "|---" * (len(cols) + 4) + "|"]
    regressions = []
    for cid, r in results.items():
        cells = []
        for key, _, _ in CRITIQUE_SCORES:
            v = r["scores"][key]["score"]
            old = ((baseline.get(cid) or {}).get("scores") or {}).get(key, {}).get("score")
            delta = f" ({v - old:+g})" if v is not None and old is not None and v != old else ""
            cells.append("—" if v is None else f"{v:g}{delta}")
        count = {g: sum(1 for p in r["points"] if grade(p) == g) for g in ("good", "partial", "bad")}
        lines.append(f"| {cid} | " + " | ".join(cells) + f" | {count['good']} | {count['partial']} | {count['bad']} |")
        before = {p.get("id") or _key(p["punto"]): grade(p) for p in (baseline.get(cid) or {}).get("points") or []}
        for p in r["points"]:
            if grade(p) == "bad" and before.get(p.get("id") or _key(p["punto"])) in ("good", "partial"):
                regressions.append((cid, p))
    out = lines + ["", "## Points", ""]
    for cid, r in results.items():
        out.append(f"**{cid}**")
        out += [f"- [{p['esito']}] " + (f"{p['id']} " if p.get("id") else "") + f"({p['categoria']}) {p['punto']} — "
                f"{p['why'] or '—'} ({p.get('come') or '—'}; modello: {(p.get('modello') or {}).get('esito') or '—'})"
                for p in r["points"]] or ["- —"]
        out += [f"- {col}: {r['scores'][k]['why'] or '—'}" for k, col, _ in CRITIQUE_SCORES]
        out.append("")
    if baseline:
        out += ["## Regressions against the baseline", ""]
        out += [f"- {cid}: " + (f"{p['id']} " if p.get("id") else "") + f"{p['punto']} (was good, now "
                f"{p['esito']}: {p['why'] or '—'})" for cid, p in regressions] \
            or ["- none"]
    return "\n".join(out)


def score(case_text: str, answer: str, caller=None) -> Dict[str, Dict[str, Any]]:
    """One model call (caller: a stand-in taking (system, human) and returning the parsed JSON, for tests)."""
    human = f"CASO:\n{case_text.strip() or '(testo del caso non fornito)'}\n\nPARERE:\n{answer.strip()}"
    if caller is None:
        from langchain_core.messages import HumanMessage, SystemMessage
        from src.rag import defensive_generation as dg
        raw = dg._chat_json([SystemMessage(content=SYSTEM), HumanMessage(content=human)], max_tokens=SCORE_TOKENS,
                            temperature=0)
    else:
        raw = caller(SYSTEM, human)
    return clean(raw)


def table(scored: Dict[str, Dict[str, Dict[str, Any]]], bench: set) -> str:
    head = "| case | " + " | ".join(col for _, col, _ in ROWS) + " | total |"
    lines = [head, "|---" * (len(ROWS) + 2) + "|"]
    accepted = []
    for cid, rows in scored.items():
        values = [rows[k]["score"] for k, _, _ in ROWS]
        total = sum(v for v in values if v is not None)
        tag = f"{cid} (benchmark: regression only)" if cid in bench else cid
        lines.append(f"| {tag} | " + " | ".join("—" if v is None else f"{v:g}" for v in values) + f" | {total:g}/10 |")
        if cid not in bench:
            accepted.append(rows)
    if accepted:
        means = []
        for k, _, _ in ROWS:
            got = [r[k]["score"] for r in accepted if r[k]["score"] is not None]
            means.append(f"{sum(got) / len(got):.2f}" if got else "—")
        total = sum(float(m) for m in means if m != "—")
        lines.append(f"| **mean, unseen cases ({len(accepted)})** | " + " | ".join(means) + f" | {total:.2f}/10 |")
    notes = ["", "## Why", ""]
    for cid, rows in scored.items():
        notes.append(f"**{cid}**")
        notes += [f"- {col}: {rows[k]['why'] or '—'}" for k, col, _ in ROWS]
        notes.append("")
    return "\n".join(lines + notes)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cases", help="a folder of cases: <id>/case.txt + <id>/answer.md")
    parser.add_argument("--transcript", help="an interactive transcript (.md, or a run directory holding interactive.md)")
    parser.add_argument("--live", help="a folder of live transcripts: <id>.txt (round 1, then the parere)")
    parser.add_argument("--case-texts", help="JSON {id: case text} for --transcript or --live")
    parser.add_argument("--out", help="write the table here (default: stdout)")
    parser.add_argument("--critique", help="v70: the lawyer's critiques -- a folder of <id>.critique.{md,txt,docx}, or "
                                           "one file (with --id); each answer scored against its case's critique")
    parser.add_argument("--id", default="", help="the case id of a single --critique file")
    parser.add_argument("--answer", help="with a single --critique file: the answer to score (a .md or a live .txt)")
    parser.add_argument("--case", help="with --answer: the case text file")
    parser.add_argument("--json-out", help="--critique: write the results as JSON (the next run's --baseline)")
    parser.add_argument("--baseline", help="--critique: a previous --json-out, for the deltas and the regressions")
    args = parser.parse_args(argv)
    if args.critique:
        critique_main(args)
        return
    texts = json.loads(Path(os.path.expanduser(args.case_texts)).read_text(encoding="utf-8")) if args.case_texts else {}
    if args.cases:
        cases = from_dir(os.path.expanduser(args.cases))
    elif args.live:
        cases = from_live(os.path.expanduser(args.live), texts)
    elif args.transcript:
        path = os.path.expanduser(args.transcript)
        path = os.path.join(path, "interactive.md") if os.path.isdir(path) else path
        cases = from_transcript(path, texts)
    else:
        parser.error("--cases, --live or --transcript")
    bench = benchmark_ids()
    if bench & set(cases):
        print(f"note: {len(bench & set(cases))} benchmark case(s) scored for regression only, out of the mean",
              file=sys.stderr)
    scored = {}
    for cid, (text, answer) in cases.items():
        try:
            scored[cid] = score(text, answer)
        except Exception as exc:
            print(f"{cid}: scoring failed: {exc}", file=sys.stderr)
            scored[cid] = clean({})
    no_case = [cid for cid, (text, _) in cases.items() if not text.strip()]
    if no_case:
        print(f"note: no case text for {len(no_case)} case(s): {', '.join(no_case)}", file=sys.stderr)
    out = table(scored, bench) + "\n" + po_tables(scored, no_case)
    if args.out:
        Path(args.out).write_text(out + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(out)


def critique_main(args) -> None:
    critiques = load_critiques(args.critique, args.id)
    if not critiques:
        raise SystemExit(f"no critique in {args.critique}")
    texts = json.loads(Path(os.path.expanduser(args.case_texts)).read_text(encoding="utf-8")) if args.case_texts else {}
    texts = {**{cid: critique_case(c) for cid, c in critiques.items() if critique_case(c)}, **texts}
    if args.answer:
        answer = Path(os.path.expanduser(args.answer)).read_text(encoding="utf-8")
        answer = answer_of(answer) or answer
        case = Path(os.path.expanduser(args.case)).read_text(encoding="utf-8") if args.case else ""
        cases = {next(iter(critiques)): (case, answer)}
    elif args.cases:
        cases = from_dir(os.path.expanduser(args.cases))
    elif args.live:
        cases = from_live(os.path.expanduser(args.live), texts)
    elif args.transcript:
        path = os.path.expanduser(args.transcript)
        cases = from_transcript(os.path.join(path, "interactive.md") if os.path.isdir(path) else path, texts)
    else:
        raise SystemExit("--critique needs --answer, --cases, --live or --transcript")
    missing = sorted(set(critiques) - set(cases))
    if missing:
        print(f"note: no answer for the critique(s) {', '.join(missing)}", file=sys.stderr)
    results = {}
    for cid in sorted(set(critiques) & set(cases)):
        case, answer = cases[cid]
        try:
            results[cid] = critique_score(critiques[cid], case, answer)
        except Exception as exc:
            print(f"{cid}: scoring failed: {exc}", file=sys.stderr)
            results[cid] = clean_critique({}, (critiques[cid] or {}).get("points") if isinstance(critiques[cid], dict)
                                          else None, cases[cid][1])
    baseline = json.loads(Path(os.path.expanduser(args.baseline)).read_text(encoding="utf-8")) if args.baseline else None
    out = critique_table(results, baseline)
    if args.json_out:
        Path(os.path.expanduser(args.json_out)).write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n",
                                                           encoding="utf-8")
    if args.out:
        Path(args.out).write_text(out + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(out)


if __name__ == "__main__":
    main()
