#!/usr/bin/env python3
"""v76b: the decomposer (issue_records._requisites -- the records' own split) run over every fattispecie article of the
corpus: c.p. Libro II, c.c. Libri IV and V (totals per Titolo), the special laws of data/special_laws.json. One row per
article -- article | rubrica | elements | first element | failure reason (fewer than two elements) --, a totals row per
code (and per c.c. Titolo): the share with two elements or more, the share with none. «Fattispecie»: what the records
count (issue_records.counted: never a definitional or procedural rubrica, never c.p. Libro I).

  python scripts/decompose_report.py --env .env --out decompose.md --json-out decompose.json      # the store
  python scripts/decompose_report.py --pdf DIR --out decompose.md                                 # the codes' PDFs

--env reads the graph (the server). --pdf reads the 2026 PDFs the corpus was built from (pdftotext; the c.p., the
c.c. and the special laws whose file PDF_FILES names) -- the same split, offline: the shares can differ slightly from
the store's (the PDF's unwrapping is this script's). --failures N: the failure rows shown per code (default all).
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k"), ("LLM_API_KEY", "k")):
    os.environ.setdefault(key, value)

CP_BOOK_II = (241, 649)
CC_TITOLI = OrderedDict([      # c.c. Libro IV and V, per Titolo
    ("IV.I obbligazioni in generale", (1173, 1320)), ("IV.II contratti in generale", (1321, 1469)),
    ("IV.III singoli contratti", (1470, 1986)), ("IV.IV promesse unilaterali", (1987, 1991)),
    ("IV.V titoli di credito", (1992, 2027)), ("IV.VI gestione di affari", (2028, 2032)),
    ("IV.VII pagamento dell'indebito", (2033, 2040)), ("IV.VIII arricchimento senza causa", (2041, 2042)),
    ("IV.IX fatti illeciti", (2043, 2059)), ("V.I attività professionali", (2060, 2081)),
    ("V.II lavoro nell'impresa", (2082, 2221)), ("V.III lavoro autonomo", (2222, 2238)),
    ("V.IV lavoro subordinato in particolari rapporti", (2239, 2246)), ("V.V società", (2247, 2510)),
    ("V.VI imprese cooperative", (2511, 2548)), ("V.VII associazione in partecipazione", (2549, 2554)),
    ("V.VIII azienda", (2555, 2574)), ("V.IX opere dell'ingegno", (2575, 2594)), ("V.X concorrenza", (2595, 2642)),
])
PDF_FILES = {     # the code or the special_laws.json id -> a substring of its PDF's file name
    "c.p.": "Codice Penale - 2026", "c.c.": "Codice Civile - 2026",
    "codice_della_strada": "codice-strada", "codice_del_consumo": "codice-consumo",
    "testo_unico_bancario": "testo-unico-bancario", "testo_unico_finanza": "testo-unico-finanza",
    "testo_unico_edilizia": "testo-unico-edilizia", "testo_unico_immigrazione": "testo-unico-immigrazione",
    "testo_unico_stupefacenti": "testo-unico-stupefacenti", "testo_unico_sicurezza_lavoro": "testo-unico-sicurezza-lavoro",
    "testo_unico_enti_locali": "testo-unico-enti-locali", "testo_unico_imposte_redditi": "testo-unico-imposte-redditi",
    "dlgs_231_2001": "Responsabilita Amministrativa Societa Enti", "gdpr": "gdpr-24-01-2019",
}

# --- the PDFs ----------------------------------------------------------------------------------------------------------------

_SUFFIX = r"(?:[-\s]?(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|nonies|decies|undecies|duodecies|terdecies|" \
          r"quaterdecies|quinquiesdecies|sexiesdecies|septiesdecies))?"
_HEAD_RE = re.compile(rf"^\s*Art(?:icolo)?\.?\s+(\d+{_SUFFIX})\s*\.?\s*(.*)$", re.IGNORECASE)
_SKIP_RE = re.compile(r"^\s*(?:LIBRO|TITOLO|CAPO|SEZIONE|PARTE|Libro|Titolo|Capo|Sezione|Parte)\b|^\s*In vigore dal\b|"
                      r"Altalex|Collana Codici|^\s*\d{1,4}\s*$|^\s*Pagina\s+\d+|^\s*['‘]?CODICE\s+(?:PENALE|CIVILE)\b|"
                      r"^\s*Testo risultante\b|©\s*[Cc]opyright|^\s*Wolters Kluwer\s*$|^\s*\d{2}/\d{2}/\d{4}\s*$",
                      re.UNICODE)
# an abrogated article: «L'articolo che recitava: …», a quoted old rubrica, a body of dots «(………)» or «[abrogato]»
_ABROGATED_RUBRICA_RE = re.compile(r"^\s*(?:L['’]articolo che recitava|[“\"«])", re.IGNORECASE)
_ABROGATED_BODY_RE = re.compile(r"^\W*(?:\(\s*[.…]+\s*\)\W*$|\[?\s*abrogat\w*\s*\]?\W*$|(?:\d+[\w-]*\.\s*)?abrogat[oa]\s+da|"
                                r"\[[^\]]*\]\W*$)", re.IGNORECASE)     # «Abrogato dal …», a body all «[ … ]» (an old text)
_NOTE_START_RE = re.compile(r"^\s*(?:Note:|\(\d{1,3}\)\s|\[\d{1,3}\]\s)")
_MARK_RE = re.compile(r"\s*\[\d{1,4}\]|\s*\(\d{1,3}\)(?=[\s.,;:]|$)")
_COMMA_NO_RE = re.compile(rf"^\s*(\d+{_SUFFIX})\.\s*$", re.IGNORECASE)


def pdf_text(pdf: Path, cache: Path) -> str:
    out = cache / (re.sub(r"\W+", "_", pdf.stem) + ".txt")
    if not out.exists():
        subprocess.run(["pdftotext", str(pdf), str(out)], check=True)
    return out.read_text(encoding="utf-8", errors="replace")


def _paragraphs(lines: Sequence[str]) -> str:
    """The body's paragraphs (one per line, as the store keeps commi): a comma number alone on its line opens one, a line
    after a full stop that starts with a capital or a number opens one, anything else continues the paragraph."""
    paras: List[str] = []
    pending_no = ""
    for raw in lines:
        line = " ".join(_MARK_RE.sub("", raw).split())
        if not line:
            continue
        m = _COMMA_NO_RE.match(line)
        if m:
            pending_no = m.group(1)
            continue
        if pending_no:
            paras.append(f"{pending_no}. {line}")
            pending_no = ""
        elif paras and not (re.search(r"[.;:]$", paras[-1]) and re.match(r"[A-ZÀ-Ü0-9«\"(]", line)):
            paras[-1] = f"{paras[-1]} {line}"
        else:
            paras.append(line)
    return "\n".join(paras)


def pdf_articles(text: str) -> Iterable[Tuple[str, str, str]]:
    """(number, rubrica, body) of each article of an extracted code: the heading «Art. N. Rubrica» (or the rubrica on
    the next line), the body up to the next heading, its notes, headings and page furniture left out."""
    lines = text.splitlines()
    i = 0
    seen = set()
    while i < len(lines):
        m = _HEAD_RE.match(lines[i])
        if not m:
            i += 1
            continue
        number = re.sub(r"\s+", "-", m.group(1).strip().lower()).replace("--", "-")
        rubrica = _MARK_RE.sub("", m.group(2)).strip()
        rubrica = re.sub(r"^\([^)]*\)\s*", "", rubrica).strip(" .")         # «(Legge 26 giugno 1990 …)»
        j = i + 1
        while j < len(lines) and re.match(r"^\s*[a-zà-ü]", lines[j]) and len(lines[j].split()) <= 10 \
                and not rubrica.endswith("."):          # a wrapped rubrica's second line («… e pascolo / abusivo»)
            rubrica = f"{rubrica} {_MARK_RE.sub('', lines[j]).strip(' .')}"
            j += 1
        if not rubrica:
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and not _HEAD_RE.match(lines[j]) and not _COMMA_NO_RE.match(lines[j].strip()):
                rubrica = _MARK_RE.sub("", lines[j]).strip(" .")
                j += 1
        body: List[str] = []
        noted = False
        while j < len(lines) and not _HEAD_RE.match(lines[j]):
            line = lines[j]
            if _NOTE_START_RE.match(line):
                noted = True
            if not noted and not _SKIP_RE.search(line):
                body.append(line)
            j += 1
        if number not in seen:              # a later «Art. N» inside a note is an old wording: the first one counts
            seen.add(number)
            yield number, rubrica, _paragraphs(body)
        i = j


# --- the store ---------------------------------------------------------------------------------------------------------------

def store_articles(code: str, law: Optional[Dict[str, Any]] = None) -> Iterable[Tuple[str, str, str]]:
    """(number, rubrica, plain text) of each article of a code (its document prefix) or a special law (the documents
    whose name its «code» expression matches), read as scripts/fill_procedural_store.py reads them."""
    import neo4j
    from src.rag import article_text, defensive_generation as dg
    from src.rag.main import NEO4J_DATABASE, driver
    query = ("MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE d.id IN $docs AND s.titolo_articolo IS NOT NULL "
             "RETURN DISTINCT d.id AS doc, s.name AS name")
    with driver.session(database=NEO4J_DATABASE, default_access_mode=neo4j.READ_ACCESS) as session:
        store = article_text.ArticleStore()
        if law is None:
            prefix = dg._CODE_PREFIXES.get(code)
            docs = store.prefix_docs(session, [prefix])[prefix] if prefix else []
        else:
            docs = [r["id"] for r in session.run("MATCH (d:Document) RETURN d.id AS id, d.name AS name").data()
                    if law["_code"].search(" ".join(str(r.get("name") or "").split()))]
        numbers = sorted({(r["doc"], article_text.article_number(r["name"])) for r in
                          session.run(query, docs=docs).data() if article_text.article_number(r["name"])})
        for i in range(0, len(numbers), 200):
            for (doc, number), rows in article_text.fetch_many(session, numbers[i:i + 200], store).items():
                if rows:
                    s = rows[0].get("s") or {}
                    yield str(number), str(s.get("titolo_articolo") or ""), str(s.get("plain_text") or "")


# --- the report --------------------------------------------------------------------------------------------------------------

def _base(number: str) -> int:
    m = re.match(r"\d+", number)
    return int(m.group(0)) if m else -1


def in_scope(code: str, number: str) -> Optional[str]:
    """The article's group in the report (the code, or its c.c. Titolo); None outside the scope."""
    n = _base(number)
    if code == "c.p.":
        return "c.p. Libro II" if CP_BOOK_II[0] <= n <= CP_BOOK_II[1] else None
    if code == "c.c.":
        return next((f"c.c. {t}" for t, (lo, hi) in CC_TITOLI.items() if lo <= n <= hi), None)
    return code


def failure(label: str, rubrica: str, text: str, elements: int) -> str:
    """Why the article gives fewer than two elements -- the pattern, never the article."""
    from src.rag import rule_quote as R
    if elements >= 2:
        return ""
    commi = R.commi_of(text)
    if not commi:
        return "testo vuoto"
    first = commi[0][1]
    low = first.lower()
    if re.search(r"\babrogat[oa]\b", low) and len(first) < 120:
        return "abrogato"
    if R.refers_penalties(first):
        return "rinvio ad altre norme o pene"
    if R.DEFINITION_RE.search(" ".join(R._COMMA_HEAD_RE.sub("", first, count=1).split())):
        return "definizione o principio"
    number, code = label.split(" ", 1)
    penal = R.offence((number, code)) or R.penal_provision(first)
    if penal:
        clauses = R.penal_clauses(first)
        if not clauses:
            return "pena senza clausola del fatto"
        if len(clauses) == 1:
            return "una sola clausola" + (" breve" if len(clauses[0].split()) < 3 else "")
        return "oltre 8 clausole" if len(clauses) > R.REQUISITES_MAX else "clausole scartate"
    reqs = R.requisites(first)
    if len(reqs) > R.REQUISITES_MAX:
        return "oltre 8 requisiti"
    if len(reqs) == 1:
        return "un solo requisito (connettivo)"
    if R._CONSEQUENCE_RE.search(first):
        return "conseguenza senza clausole prima"
    if R._CONNECTOR_RE.search(first):
        return "connettivo senza clausola utile"
    return "nessun connettivo né conseguenza"


def decompose(code: str, number: str, rubrica: str, text: str) -> Dict[str, Any]:
    from src.rag import issue_records
    label = f"{number} {code}"
    row = {"_article": (number, code), "s": {"plain_text": text, "titolo_articolo": rubrica}}
    try:
        reqs = [r for r in issue_records._requisites(row, {}) if r.get("kind") == "requisito"]
    except Exception as exc:        # noqa: BLE001 -- a crash is a failure reason too
        return {"article": label, "rubrica": rubrica, "elements": 0, "first": "", "failure": f"errore: {exc}"[:120]}
    return {"article": label, "rubrica": rubrica, "elements": len(reqs), "first": reqs[0]["text"] if reqs else "",
            "failure": failure(label, rubrica, text, len(reqs))}


def run(source: str, pdf_dir: Optional[Path], cache: Path, codes: Sequence[str]) -> List[Dict[str, Any]]:
    from src.rag import issue_records, special_laws
    laws = {law["id"]: law for law in special_laws.laws()}
    out = []
    for code in codes:
        law = laws.get(code)
        if source == "pdf":
            name = PDF_FILES.get(code)
            pdf = next((p for p in sorted(pdf_dir.glob("*.pdf")) if name and name in p.name), None) if pdf_dir else None
            if pdf is None:
                print(f"{code}: no PDF in {pdf_dir} -- skipped (the store has it: --env)", file=sys.stderr)
                continue
            articles = pdf_articles(pdf_text(pdf, cache))
        else:
            articles = store_articles(code, law)
        label_code = code if law is None else law["id"]
        for number, rubrica, text in articles:
            group = in_scope(code, number)
            if group is None or not issue_records.counted(f"{number} {code}", rubrica):
                continue
            if _ABROGATED_RUBRICA_RE.match(rubrica) or not text.strip() or _ABROGATED_BODY_RE.match(text):
                out.append({"article": f"{number} {label_code}", "rubrica": rubrica, "elements": 0, "first": "",
                            "failure": "abrogato o senza testo", "excluded": True, "group": group})
                continue
            row = decompose(label_code if law else code, number, rubrica, text)
            row["group"] = group
            out.append(row)
    return out


def totals(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
        code = r["group"].split(" ", 1)[0] if r["group"].startswith(("c.c.", "c.p.")) else "leggi speciali"
        if r["group"].startswith("c.c. "):
            groups.setdefault("c.c. Libri IV e V", []).append(r)
        if code == "leggi speciali":
            groups.setdefault("leggi speciali (tutte)", []).append(r)
    out = []
    for name, every in groups.items():
        items = [r for r in every if not r.get("excluded")]
        n = max(1, len(items))
        out.append({"group": name, "articles": len(items), "excluded": len(every) - len(items),
                    "two_or_more": round(100 * sum(r["elements"] >= 2 for r in items) / n, 1),
                    "none": round(100 * sum(r["elements"] == 0 for r in items) / n, 1),
                    "one": round(100 * sum(r["elements"] == 1 for r in items) / n, 1),
                    "reasons": Counter(r["failure"] for r in items if r["failure"]).most_common(8)})
    return out


def _cell(x: Any) -> str:
    return " ".join(str(x or "").replace("|", "/").split())[:140]


def report(rows: Sequence[Dict[str, Any]], tots: Sequence[Dict[str, Any]], failures: Optional[int]) -> str:
    lines = ["# Decomposizione degli articoli-fattispecie", "", "## Totali", "",
             "| gruppo | articoli | ≥2 elementi | 0 elementi | 1 elemento | esclusi (abrogati) | motivi principali |",
             "|---|---|---|---|---|---|---|"]
    for t in tots:
        lines.append(f"| {t['group']} | {t['articles']} | {t['two_or_more']}% | {t['none']}% | {t['one']}% | "
                     f"{t['excluded']} | "
                     + _cell("; ".join(f"{k} ({v})" for k, v in t["reasons"])) + " |")
    by_group: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for r in rows:
        by_group.setdefault(r["group"], []).append(r)
    for group, items in by_group.items():
        shown = [r for r in items if r["failure"]]
        shown = shown[:failures] if failures is not None else items
        lines += ["", f"## {group}", "", "| articolo | rubrica | elementi | primo elemento | motivo |", "|---|---|---|---|---|"]
        lines += [f"| {r['article']} | {_cell(r['rubrica'])} | {r['elements']} | {_cell(r['first'])} | {r['failure']} |"
                  for r in shown]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--env", help="the server's .env: the graph")
    src.add_argument("--pdf", type=Path, help="a folder of the codes' PDFs")
    ap.add_argument("--cache", type=Path, default=Path(os.getenv("TMPDIR", "/tmp")) / "decompose_report_cache")
    ap.add_argument("--codes", help="comma separated: c.p., c.c. and special_laws.json ids (default all)")
    ap.add_argument("--failures", type=int, help="failure rows shown per group (default: every row)")
    ap.add_argument("--out")
    ap.add_argument("--json-out")
    args = ap.parse_args(argv)
    if args.env:
        from dotenv import load_dotenv
        load_dotenv(args.env, override=True)
    from src.rag import special_laws
    codes = [c.strip() for c in args.codes.split(",")] if args.codes else ["c.p.", "c.c."] + [
        law["id"] for law in special_laws.laws()]
    args.cache.mkdir(parents=True, exist_ok=True)
    rows = run("pdf" if args.pdf else "store", args.pdf, args.cache, codes)
    tots = totals(rows)
    text = report(rows, tots, args.failures)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"totals": tots, "rows": rows}, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    for t in tots:
        print(f"{t['group']:52} {t['articles']:5}  ≥2: {t['two_or_more']:5}%  0: {t['none']:5}%  1: {t['one']:5}%  "
              f"esclusi {t['excluded']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
