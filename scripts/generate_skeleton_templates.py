"""
Bulk skeleton template generator for SunnitAI document generation.

For each (codice, categoria, sottocategoria, tipo_atto) entry in
master_template_list.json, calls the chatbot's LLM to generate:
  - a short classifier description (for the 3-stage LLM classifier)
  - a structured skeleton (ordered sections, each either fixed
    boilerplate text or a [DA COMPILARE: ...] placeholder)
  - a list of structured fields (matching the existing
    DOCUMENT_TYPE_REGISTRY "fields" convention)

Outputs:
  - One .docx skeleton file per entry, in --output-dir
  - A single catalog.json indexing every generated template
    (filename, codice, categoria, tipo_atto, label, description,
    fields) — this is what feeds the 3-stage classifier and,
    eventually, the system user's template metadata.

Run with --limit N first to pilot a small batch before the full run.
Safe to interrupt and resume — skips entries whose .docx already
exists in --output-dir.

Usage:
  python3 generate_skeleton_templates.py --limit 5          # pilot
  python3 generate_skeleton_templates.py                    # full run
  python3 generate_skeleton_templates.py --workers 10        # parallel
"""

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dotenv import load_dotenv

load_dotenv('/opt/chatbot/.env')

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

MASTER_LIST_PATH = os.path.join(os.path.dirname(__file__), "master_template_list.json")


def slugify(text: str, max_len: int = 60) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text[:max_len].rstrip("-")


def codice_slug(codice: str) -> str:
    if "civile" in codice.lower():
        return "civile"
    if "penale" in codice.lower():
        return "penale"
    if "amministrativo" in codice.lower():
        return "amministrativo"
    return slugify(codice)


def build_prompt(entry: dict) -> str:
    categorie = "; ".join(entry["categorie"])
    sottocategorie = "; ".join(s for s in entry["sottocategorie"] if s)
    return (
        "Sei un esperto di diritto processuale italiano. Genera la STRUTTURA "
        "(non il contenuto sostanziale) di un atto giuridico italiano del tipo "
        "seguente, da usare come template compilabile da un avvocato.\n\n"
        f"Codice di riferimento: {entry['codice']}\n"
        f"Categoria/fase processuale: {categorie}\n"
        f"Sottocategoria: {sottocategorie or 'N/A'}\n"
        f"Tipo di atto: {entry['tipo_atto']}\n\n"
        "Genera ESATTAMENTE questo JSON, senza testo prima o dopo, senza "
        "markdown code fences:\n\n"
        "{\n"
        '  "label": "Nome ufficiale e leggibile dell\'atto",\n'
        '  "description": "Descrizione breve (max 40 parole) dell\'atto, utile '
        'a un classificatore automatico per riconoscere quando un utente lo sta '
        'richiedendo. Includere termini che un avvocato userebbe.",\n'
        '  "sections": [\n'
        '    {"heading": "Nome sezione (es. Intestazione, Premessa, In fatto, '
        'In diritto, Conclusioni, P.Q.M.)", '
        '"content": "Testo fisso standard, oppure [DA COMPILARE: istruzioni '
        'chiare su cosa inserire qui]"}\n'
        "  ],\n"
        '  "fields": ["elenco di 4-10 campi strutturati chiave necessari, es. '
        'nome_parte, tribunale, numero_rg, data_udienza"]\n'
        "}\n\n"
        "Regole importanti:\n"
        "- LINGUA: rispondi INTERAMENTE e SOLO in italiano. Non utilizzare MAI "
        "l'inglese, nemmeno singole parole o frasi, in nessuna sezione.\n\n"
        "- CALIBRAZIONE DELLA STRUTTURA — segui questo modello di ragionamento "
        "per decidere quante sezioni usare e quanto devono essere sostanziali:\n\n"
        "  TIPO A — atti introduttivi o di merito (es. Atto di citazione, "
        "Ricorso, Comparsa di risposta, Memoria difensiva di merito, Atto di "
        "appello): questi atti ARGOMENTANO una posizione davanti al giudice e "
        "richiedono SEMPRE sezioni sostanziali dedicate a: esposizione dei "
        "fatti rilevanti della controversia, motivi/ragioni di diritto a "
        "sostegno della domanda o difesa, eventuali mezzi di prova/richieste "
        "istruttorie, e conclusioni/richieste specifiche al giudice. Un atto "
        "di questo tipo SENZA queste sezioni sostanziali è incompleto e "
        "inutilizzabile nella pratica forense reale — non limitarti a sezioni "
        "generiche come 'Corpo del testo'.\n"
        "  ATTENZIONE: la 'vocatio in ius' (citazione della controparte a "
        "comparire entro un termine) è specifica SOLO dell'Atto di citazione "
        "che introduce un giudizio civile ex novo — NON includerla in altri "
        "atti (ricorsi, memorie, istanze cautelari, atti di appello, ecc.) "
        "che si inseriscono in un procedimento già pendente. Per questi "
        "ultimi, le conclusioni devono contenere solo le richieste specifiche "
        "pertinenti a QUEL tipo di atto (es. annullamento, sospensione, "
        "accoglimento di un'istanza), mai una citazione a comparire.\n\n"
        "  TIPO B — atti procedurali semplici (es. Procura, Elezione di "
        "domicilio, Rinuncia al mandato, Richiesta copie, Notifica): sono "
        "brevi comunicazioni o conferimenti di potere, senza argomentazione "
        "di merito. Bastano 3-5 sezioni essenziali (intestazione, oggetto "
        "dell'atto, eventuali specifiche, sottoscrizione) — non aggiungere "
        "'In fatto'/'In diritto'/'P.Q.M.' se l'atto non ne contiene davvero.\n\n"
        "  Per il tipo di atto richiesto qui sotto, valuta la sua VERA natura "
        "prima di scegliere la struttura: introduce un giudizio o argomenta "
        "una posizione di merito davanti a un giudice? → usa il modello A, con "
        "sezioni sostanziali e placeholder specifici sui fatti e sul diritto. "
        "È una comunicazione procedurale, un conferimento di potere o una "
        "richiesta semplice? → usa il modello B.\n\n"
        "- NON inventare contenuto legale sostanziale (fatti, argomentazioni, "
        "norme specifiche del caso) — solo struttura e placeholder, anche per "
        "gli atti di tipo A: il placeholder deve guidare l'avvocato su COSA "
        "scrivere (es. '[DA COMPILARE: esporre i fatti rilevanti che hanno "
        "dato origine alla controversia, in ordine cronologico]'), non "
        "inventare i fatti stessi.\n"
        "- 'Conclusioni' e 'P.Q.M.' coprono lo STESSO contenuto (la richiesta "
        "formale al giudice) — non creare MAI entrambe come sezioni separate. "
        "Usa UNA SOLA sezione per questo scopo: chiamala 'P.Q.M.' se l'atto "
        "termina tipicamente con questa formula nella prassi forense, "
        "altrimenti chiamala 'Conclusioni'. Non duplicare.\n"
        "- Ogni placeholder [DA COMPILARE: ...] deve avere un'istruzione chiara "
        "e specifica, non generica.\n"
        "- Includi sempre intestazione (autorità destinataria, parti) e, se "
        "pertinente al tipo di atto, formula di sottoscrizione finale.\n"
    )


def call_llm(prompt: str, timeout: int = 45) -> dict | None:
    if not LLM_BASE_URL:
        return None
    try:
        resp = requests.post(
            f"{LLM_BASE_URL}/chat/completions",
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 1500,
            },
            headers={
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip markdown code fences if the model added them anyway
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except Exception as e:
        print(f"  [LLM ERROR] {e}")
        return None


def build_docx(skeleton: dict, output_path: str) -> bool:
    """Build a .docx file from the skeleton JSON using python-docx."""
    try:
        from docx import Document
        from docx.shared import Pt
        from docx.enum.text import WD_ALIGN_PARAGRAPH
    except ImportError:
        print("  python-docx not installed. Run: pip install python-docx --break-system-packages")
        return False

    doc = Document()

    title = doc.add_heading(skeleton.get("label", "Atto"), level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    for section in skeleton.get("sections", []):
        heading = section.get("heading", "")
        content = section.get("content", "")
        if heading:
            doc.add_heading(heading, level=2)
        if content:
            p = doc.add_paragraph(content)
            if "[DA COMPILARE" in content:
                for run in p.runs:
                    run.italic = True

    fields = skeleton.get("fields", [])
    if fields:
        doc.add_heading("Campi strutturati", level=2)
        for f in fields:
            doc.add_paragraph(f, style="List Bullet")

    doc.save(output_path)
    return True


def flag_quality_issues(entry: dict, skeleton: dict) -> list[str]:
    """Deterministic checks for known LLM failure patterns this script has
    hit in testing. Returns a list of human-readable reasons this entry
    should get priority attention during the lawyer review pass — does NOT
    auto-fix anything, since silently rewriting legal language is riskier
    than flagging it for a human to look at."""
    issues = []
    is_citazione = "citazione" in entry["tipo_atto"].lower()
    full_text = json.dumps(skeleton, ensure_ascii=False).lower()

    if not is_citazione and ("vocatio in ius" in full_text or "termine a comparire" in full_text):
        issues.append("Contiene 'vocatio in ius' / 'termine a comparire' ma non è un Atto di citazione")

    # Crude English-leak check: common English function words that
    # shouldn't appear in a clean Italian legal document
    english_markers = [" the ", " pursuant to", " hereby", " shall ", " whereas "]
    if any(m in full_text for m in english_markers):
        issues.append("Possibile testo in inglese rilevato")

    pqm_section = next((s for s in skeleton.get("sections", []) if "p.q.m" in s.get("heading", "").lower()), None)
    conclusioni_section = next((s for s in skeleton.get("sections", []) if "conclusioni" in s.get("heading", "").lower()), None)
    if pqm_section and conclusioni_section:
        # Observed pattern: when both sections exist separately, P.Q.M.
        # reliably drifts into unrelated content (cauzione, notification
        # details, mezzi di prova content — different wrong topic each
        # time, same root cause: nothing distinct left to say). More
        # reliable signal than matching specific wrong phrases.
        issues.append("Contiene sia 'Conclusioni' che 'P.Q.M.' come sezioni separate — verificare che P.Q.M. non contenga contenuto fuori tema")
    elif pqm_section:
        content = pqm_section.get("content", "").lower()
        off_topic_markers = ["cauzione", "persona qualificata per ricevere notifiche", "domicilio", "mezzi di prova", "istruttorie"]
        if any(m in content for m in off_topic_markers):
            issues.append("Sezione P.Q.M. sembra fuori tema (verificare che riformuli la richiesta al giudice)")

    return issues


def process_entry(entry: dict, output_dir: str) -> dict:
    codice_s = codice_slug(entry["codice"])
    tipo_s = slugify(entry["tipo_atto"])
    filename = f"{codice_s}__{tipo_s}.docx"
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        return {"status": "skipped_exists", "filename": filename}

    prompt = build_prompt(entry)
    skeleton = call_llm(prompt)
    if not skeleton:
        return {"status": "llm_failed", "filename": filename, "entry": entry}

    ok = build_docx(skeleton, output_path)
    if not ok:
        return {"status": "docx_failed", "filename": filename, "entry": entry}

    issues = flag_quality_issues(entry, skeleton)

    catalog_entry = {
        "filename": filename,
        "codice": entry["codice"],
        "categorie": entry["categorie"],
        "sottocategorie": entry["sottocategorie"],
        "tipo_atto": entry["tipo_atto"],
        "label": skeleton.get("label", entry["tipo_atto"]),
        "description": skeleton.get("description", ""),
        "fields": skeleton.get("fields", []),
        "flagged_issues": issues,
    }
    return {"status": "ok", "filename": filename, "catalog_entry": catalog_entry}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process only first N entries (pilot mode)")
    parser.add_argument("--workers", type=int, default=5, help="Concurrent LLM workers")
    parser.add_argument("--output-dir", default="/opt/chatbot/data/system_templates_pilot", help="Where to save .docx files")
    parser.add_argument("--master-list", default=MASTER_LIST_PATH, help="Path to master_template_list.json")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.master_list, encoding="utf-8") as f:
        entries = json.load(f)

    if args.limit:
        entries = entries[: args.limit]

    print(f"Processing {len(entries)} entries with {args.workers} workers")
    print(f"Output dir: {args.output_dir}\n")

    catalog_path = os.path.join(args.output_dir, "catalog.json")
    catalog = []
    if os.path.exists(catalog_path):
        with open(catalog_path, encoding="utf-8") as f:
            catalog = json.load(f)

    ok_count = 0
    skip_count = 0
    fail_count = 0
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_entry, e, args.output_dir): e for e in entries}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result["status"] == "ok":
                ok_count += 1
                catalog.append(result["catalog_entry"])
            elif result["status"] == "skipped_exists":
                skip_count += 1
            else:
                fail_count += 1
                print(f"  [FAIL] {result['filename']} ({result['status']})")

            if i % 10 == 0 or i == len(entries):
                elapsed = time.time() - t_start
                print(f"  Progress: {i}/{len(entries)} | OK: {ok_count} | Skipped: {skip_count} | Failed: {fail_count} | {elapsed:.0f}s elapsed")

    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)

    flagged = [c for c in catalog if c.get("flagged_issues")]
    flagged_path = os.path.join(args.output_dir, "flagged_for_review.json")
    with open(flagged_path, "w", encoding="utf-8") as f:
        json.dump(flagged, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/60:.1f} minutes.")
    print(f"Generated: {ok_count} | Skipped (already existed): {skip_count} | Failed: {fail_count}")
    print(f"Flagged for priority review: {len(flagged)} of {len(catalog)}")
    print(f"Catalog saved to: {catalog_path}")
    print(f"Flagged entries saved to: {flagged_path}")


if __name__ == "__main__":
    main()
