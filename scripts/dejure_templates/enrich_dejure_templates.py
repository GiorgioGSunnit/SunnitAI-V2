"""Build catalog entries for the converted DEJURE templates.

Reads the converted DOCX files (and their source PDFs, for the DEJURE collection
label) and writes catalog entries in the same shape as the 484 already on the
server: filename, codice, categorie, sottocategorie, tipo_atto, label,
description, fields, flagged_issues, sections.

  sections     taken FROM THE DOCUMENT (heading + its text), using the same
               heading rule the generation code uses to pick a section, so the
               reference content is the real template rather than a rewrite.
  description  one sentence, from the LLM.
  fields       snake_case names of the data to collect, from the LLM.
  categorie    chosen by the LLM from the categories already used in that area,
               or proposed when the area is new.

Nothing is uploaded. Output: catalog_new_entries.json + review.tsv.
Resumable: every LLM answer is cached in enrich_cache.json.

    python scripts/enrich_dejure_templates.py --limit 3     # try a few
    python scripts/enrich_dejure_templates.py               # all of them
"""
import argparse, glob, hashlib, json, os, re, sys, time
import unicodedata as ud

import fitz
import httpx
from docx import Document
from dotenv import load_dotenv

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONVERTED = r"C:\Users\anton\Downloads\downloads\downloads\converted"
PDFS = r"C:\Users\anton\Downloads\downloads\downloads\test folder"
SERVER_CATALOG = r"C:\Users\anton\Downloads\catalog_enriched_server.json"
OUT_DIR = os.path.join(REPO, "scripts", "out")
CACHE = os.path.join(OUT_DIR, "enrich_cache.json")

load_dotenv(os.path.join(REPO, ".env"))
LLM_BASE = (os.getenv("LLM_BASE_URL") or "").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL")
LLM_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")

# DEJURE collection -> area (codice). Two areas are new: the catalog has nothing
# for company law or tax litigation.
COLLECTION_TO_CODICE = {
    "FORMULARIO DEGLI ATTI PENALI": "Codice di procedura penale",
    "FORMULARIO DEL PROCESSO CIVILE": "Codice di procedura civile",
    "FORMULARIO DELLE ESECUZIONI CIVILI": "Codice di procedura civile",
    "FORMULARIO DELLA RESPONSABILITÀ CIVILE": "Codice di procedura civile",
    "FORMULARIO DELLE FAMIGLIE": "Codice di procedura civile",
    "FORMULARIO DEL PROCESSO AMMINISTRATIVO": "Codice del processo amministrativo",
    "FORMULARIO DEL LAVORO": "Diritto del Lavoro",
    "FORMULARIO CONTRATTI": "Contratti e Atti Stragiudiziali",
    "FORMULARIO APPALTI": "Contratti e Atti Stragiudiziali",
    "FORMULARIO CONDOMINIO E LOCAZIONE": "Contratti e Atti Stragiudiziali",
    "FORMULARIO AGRARIO": "Contratti e Atti Stragiudiziali",
    "FORMULARIO RISOLUZIONE ALTERNATIVE CONTROVERSIE": "Arbitrato e Procedure Alternative",
    "FORMULARIO DELLA CRISI D'IMPRESA E INSOLVENZA": "Diritto Concorsuale",
    "FORMULARIO DELLE SOCIETÀ": "Diritto Societario",
    "FORMULARIO DEL CONTENZIOSO TRIBUTARIO": "Diritto Tributario",
    "FORMULARIO DEL CONTENZIOSO BANCARIO": "Codice di procedura civile",
}

# A court act keeps its procedural code; a contract, clause or letter from the
# same collection belongs with the out-of-court acts.
OUT_OF_COURT = re.compile(
    r"^(contratto|clausola|accordo|lettera|comunicazione|invito|proposta|"
    r"dichiarazione|avviso|comodato|anticipazione|estratto|preventivo)\b", re.I)
COURT_COLLECTIONS = {"FORMULARIO DEL CONTENZIOSO BANCARIO", "FORMULARIO AGRARIO",
                     "FORMULARIO CONDOMINIO E LOCAZIONE"}

CODICE_PREFIX = {
    "Codice di procedura civile": "civile",
    "Codice di procedura penale": "penale",
    "Codice del processo amministrativo": "amministrativo",
    "Contratti e Atti Stragiudiziali": "contratti",
    "Diritto del Lavoro": "lavoro",
    "Arbitrato e Procedure Alternative": "arbitrato",
    "Diritto Amministrativo Sanzionatorio": "sanzionatorio",
    "Diritto Concorsuale": "concorsuale",
    "Diritto Societario": "societario",
    "Diritto Tributario": "tributario",
}


def repair_name(name):
    for enc in ("cp437", "cp850", "cp1252"):
        try:
            return ud.normalize("NFC", name.encode(enc).decode("utf-8"))
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return ud.normalize("NFC", name)


def slugify(text):
    text = ud.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", text.lower())).strip("-")


def collection_of(pdf_path):
    text = fitz.open(pdf_path)[0].get_text()
    m = re.search(r"(?m)^\s*(FORMULARIO[^\n]*)$", text, re.I)
    return " ".join(m.group(1).split()).upper() if m else ""


def is_heading(par):
    """Same rule as _is_heading_paragraph in src/rag/document_generation.py,
    minus placeholder-only lines: "[DA COMPILARE]" alone is upper case but is a
    blank to fill, not a section heading."""
    text = par.text.strip()
    if not text or len(text) >= 100:
        return False
    if len(re.sub(r"\[DA COMPILARE\]|[^A-Za-zÀ-ÿ]", "", text)) < 3:
        return False
    style = (par.style.name or "") if par.style else ""
    if style.lower().startswith("heading") or style.lower() == "title":
        return True
    if text.isupper() and any(c.isalpha() for c in text):
        return True
    runs = [r for r in par.runs if r.text.strip()]
    return bool(runs) and all(r.bold for r in runs)


def sections_of(docx_path):
    """Sections straight from the template: heading + the text under it. Note
    paragraphs are left out - they are commentary, not part of the act."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from convert_dejure_templates import find_note_blocks

    doc = Document(docx_path)
    paras = [p for p in doc.paragraphs if p.text.strip()]
    _, blocks = find_note_blocks(paras)
    notes = {i for b in blocks for i in b}

    sections, heading, buf = [], "Intestazione", []
    for i, par in enumerate(paras):
        if i in notes:
            continue
        if is_heading(par):
            if buf:
                sections.append({"heading": heading, "content": "\n".join(buf)})
            heading, buf = par.text.strip(), []
        else:
            buf.append(par.text.strip())
    if buf:
        sections.append({"heading": heading, "content": "\n".join(buf)})
    return [s for s in sections if s["content"].strip()]


# The whole of `sections` goes into the prompt of every generation request for
# that template, so very long templates cost tokens on every use. They are kept
# in full anyway: trimming them would make generation produce a less complete
# act than the template it is based on. The cap is a safety net for the larger
# ingest, where a single runaway file should not blow the context.
SECTIONS_BUDGET = 40000
LONG_TEMPLATE = 10000    # flagged so the cost is visible, never trimmed


def trim_sections(sections, budget=SECTIONS_BUDGET):
    """Keep long templates from bloating every generation prompt. Median
    template is ~2000 chars and untouched; only 12 of 103 are longer than 6000.
    Trimming keeps each section's heading and opening, which is what the
    generation prompt uses the sections for."""
    total = sum(len(s["content"]) for s in sections)
    if total <= budget:
        return sections, False
    share = budget / total
    out = []
    for s in sections:
        keep = max(200, int(len(s["content"]) * share))
        content = s["content"]
        if len(content) > keep:
            cut = content.rfind(" ", 0, keep)
            content = content[:cut if cut > 200 else keep].rstrip() + "\n(sezione abbreviata)"
        out.append({"heading": s["heading"], "content": content})
    return out, True


def act_text(sections, limit=6000):
    text = "\n\n".join("%s\n%s" % (s["heading"], s["content"]) for s in sections)
    return text[:limit]


_cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}


def ask_llm(prompt_system, prompt_user, max_tokens=700):
    key = hashlib.sha1((prompt_system + "||" + prompt_user).encode("utf-8")).hexdigest()
    if key in _cache:
        return _cache[key]
    r = httpx.post(LLM_BASE.rstrip("/") + "/chat/completions", timeout=300,
                   headers={"Authorization": f"Bearer {LLM_KEY}"} if LLM_KEY else {},
                   json={"model": LLM_MODEL, "max_tokens": max_tokens, "temperature": 0,
                         "messages": [{"role": "system", "content": prompt_system},
                                      {"role": "user", "content": prompt_user}]})
    r.raise_for_status()
    out = r.json()["choices"][0]["message"]["content"]
    _cache[key] = out
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(_cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
    return out


def ascii_field(name):
    """Catalog field names are plain ASCII snake_case: citta_residenza, not
    città_residenza."""
    name = ud.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"_+", "_", re.sub(r"[^a-z0-9_]+", "_", name.lower())).strip("_")
    # the model occasionally slips into Spanish/Portuguese
    for wrong, right in (("sociedade", "societa"), ("sociedad", "societa"),
                         ("nombre", "nome"), ("empresa", "societa"), ("fecha", "data")):
        name = name.replace(wrong, right)
    return name


def match_category(proposed, known):
    """Reuse a category already in the catalog when the LLM says the same thing
    in different words - "RESPONSABILE CIVILE..." for "13. RESPONSABILE
    CIVILE...", "Eseguzione forzata" for "Esecuzione forzata"."""
    import difflib

    def norm(s):
        s = ud.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()
        return re.sub(r"^\d+\.\s*", "", s).strip()

    if not proposed:
        return proposed
    target = norm(proposed)
    for k in known:
        if norm(k) == target:
            return k
    close = difflib.get_close_matches(target, [norm(k) for k in known], n=1, cutoff=0.88)
    if close:
        return next(k for k in known if norm(k) == close[0])
    return proposed


def describe(label, codice, known_categories, text, strict=False):
    new_area = not known_categories or strict
    system = (
        "Sei un avvocato italiano che cataloga modelli di atti giuridici. "
        "Rispondi SOLO con un oggetto JSON valido, senza markdown, con le chiavi:\n"
        '  "description": una frase che dice cosa fa l\'atto e chi lo presenta, max 25 parole;\n'
        '  "fields": 5-10 nomi di campo in snake_case per i dati che l\'utente deve fornire '
        '(es. "nome_ricorrente", "tribunale", "data_notifica"), dedotti dai segnaposto [DA COMPILARE];\n'
        '  "categoria": la categoria che raggruppa l\'atto;\n'
        '  "sottocategoria": una sottocategoria più specifica.\n'
        + ("I nomi dei campi sono in italiano. "
           "La categoria deve raggruppare atti simili per fase o oggetto "
           "(es. \"Assemblee e verbali\", \"Operazioni sul capitale\", "
           "\"Atti introduttivi\"): NON usare come categoria il nome dell'area "
           "(\"%s\"), che è già indicato a parte." % codice
           if new_area else
           "Per categoria usa ESATTAMENTE una di quelle già in uso, se adatta:\n"
           + "\n".join("  - " + c for c in known_categories)
           + "\nAltrimenti proponine una nuova, breve e nello stesso stile.")
    )
    user = "Area: %s\nTitolo del modello: %s\n\nTesto del modello:\n%s" % (codice, label, text)
    raw = ask_llm(system, user)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    data = json.loads(m.group(0) if m else raw)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N templates")
    ap.add_argument("--no-llm", action="store_true", help="deterministic fields only")
    args = ap.parse_args()

    server = json.load(open(SERVER_CATALOG, encoding="utf-8"))
    cats_by_codice = {}
    for e in server:
        cats_by_codice.setdefault(e["codice"], set()).update(c for c in e["categorie"] if c)
    existing_files = {e["filename"] for e in server}

    pdfs = {repair_name(os.path.basename(p))[:-4]: p for p in glob.glob(PDFS + r"\*.pdf")}
    docs = sorted(f for f in glob.glob(CONVERTED + r"\*.docx") if not os.path.basename(f).startswith("~$"))
    if args.limit:
        docs = docs[:args.limit]

    entries, rows, t0 = [], [], time.time()
    for n, path in enumerate(docs, 1):
        label = os.path.basename(path)[:-5]
        collection = collection_of(pdfs[label])
        codice = COLLECTION_TO_CODICE.get(collection, "Contratti e Atti Stragiudiziali")
        if collection in COURT_COLLECTIONS and OUT_OF_COURT.match(label):
            codice = "Contratti e Atti Stragiudiziali"
        if collection == "FORMULARIO DELLE SOCIETÀ":
            codice = "Diritto Societario"

        sections, trimmed = trim_sections(sections_of(path))
        # The whole title goes in the slug. The picker builds its second line by
        # stripping the title's slug off the filename (_derive_sublabel); a
        # truncated slug fails that strip, and the row then shows the full title
        # twice - which is what pushed the picker off screen.
        filename = "%s__%s.docx" % (CODICE_PREFIX[codice], slugify(label))
        flagged = []
        if trimmed:
            flagged.append("sezioni abbreviate (modello oltre %d caratteri)" % SECTIONS_BUDGET)
        elif sum(len(s["content"]) for s in sections) > LONG_TEMPLATE:
            flagged.append("modello lungo: sezioni complete nel prompt di generazione")
        if len(act_text(sections)) < 400:
            flagged.append("testo molto breve")
        if filename in existing_files:
            flagged.append("filename già presente nel catalogo")

        entry = {
            "filename": filename,
            "codice": codice,
            "categorie": [],
            "sottocategorie": [],
            "tipo_atto": label,
            "label": label,
            "description": "",
            "fields": [],
            "flagged_issues": flagged,
            "sections": sections,
        }
        if not args.no_llm:
            try:
                known = sorted(cats_by_codice.get(codice, []))
                text = act_text(sections)
                data = describe(label, codice, known, text)
                # one retry when the answer is unusable: no category, or the
                # area's own name offered as the category
                cat = str(data.get("categoria", "")).strip()
                if not cat or cat.lower() == codice.lower():
                    data = describe(label, codice, known, text, strict=True)
                known = sorted(cats_by_codice.get(codice, []))
                entry["description"] = str(data.get("description", "")).strip()
                entry["fields"] = [f for f in (ascii_field(x) for x in data.get("fields", [])) if f][:10]
                entry["categorie"] = ([match_category(str(data["categoria"]).strip(), known)]
                                      if data.get("categoria") else [])
                entry["sottocategorie"] = [str(data["sottocategoria"]).strip()] if data.get("sottocategoria") else []
                # categories proposed for the two new areas feed the next files
                if entry["categorie"]:
                    cats_by_codice.setdefault(codice, set()).add(entry["categorie"][0])
            except Exception as exc:
                entry["flagged_issues"].append("LLM: %s" % str(exc)[:80])
                print("  ! %s: %s" % (label[:50], str(exc)[:80]), file=sys.stderr)

        entries.append(entry)
        rows.append([label, collection, codice, entry["categorie"][0] if entry["categorie"] else "",
                     str(len(sections)), str(len(act_text(sections))), entry["description"],
                     ", ".join(entry["fields"]), "; ".join(entry["flagged_issues"])])
        print("  %3d/%d  %-55s %s" % (n, len(docs), label[:55], codice), file=sys.stderr)

    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(entries, open(os.path.join(OUT_DIR, "catalog_new_entries.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    with open(os.path.join(OUT_DIR, "review.tsv"), "w", encoding="utf-8") as f:
        f.write("titolo\tcollezione\tarea\tcategoria\tsezioni\tcaratteri\tdescrizione\tcampi\tproblemi\n")
        for r in rows:
            f.write("\t".join(x.replace("\t", " ").replace("\n", " ") for x in r) + "\n")
    print("\n%d entries in %.0fs -> %s" % (len(entries), time.time() - t0, OUT_DIR))


if __name__ == "__main__":
    main()
