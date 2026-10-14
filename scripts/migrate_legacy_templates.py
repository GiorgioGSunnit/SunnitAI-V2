"""
Migrates the legacy 20 hardcoded document templates into the
catalog-based system used by the 383-template pipeline.

What this script does:
  1. RETIRES 4 templates already covered by the 383:
       opposition_act, appeal, power_of_attorney, memoria_difensiva
     These are simply removed from DOCUMENT_TYPE_REGISTRY and _SYSTEM_FN
     in document_generation.py — the 383 classifier handles them.

  2. MIGRATES 16 templates into a new 4th codice:
       "Contratti e Atti Stragiudiziali"
     For each:
       - Calls the existing Italian _xxx_system() function
       - Parses the numbered section structure directly (no LLM calls)
       - Generates a skeleton .docx using the same build_docx() used for the 383
       - Adds the entry to catalog_enriched.json

  3. Updates document_generation.py to:
       - Remove the 16 migrated keys from DOCUMENT_TYPE_REGISTRY
       - Remove the 4 retired keys from DOCUMENT_TYPE_REGISTRY
       - Remove all 20 from the _SYSTEM_FN dict inside generate_document()
       - Fix the fallback extract_document_fields call to not crash when
         "opposition_act" is no longer in the registry

Run from /opt/chatbot:
  python3 scripts/migrate_legacy_templates.py \
      --templates-dir /opt/chatbot/data/system_templates \
      --dry-run          # preview without writing anything

  python3 scripts/migrate_legacy_templates.py \
      --templates-dir /opt/chatbot/data/system_templates
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

from src.rag.document_generation import (
    _opposition_system,
    _memoria_difensiva_system,
    _rental_basic_system,
    _rental_standard_system,
    _rental_student_system,
    _rental_transitional_system,
    _rental_free_rent_system,
    _rental_commercial_system,
    _rental_cancellation_system,
    _insurance_cancellation_system,
    _insurance_declaration_system,
    _employment_dismissal_appeal_system,
    _employment_termination_system,
    _franchising_contract_system,
    _demand_letter_system,
    _appeal_system,
    _power_of_attorney_system,
    _sale_agreement_system,
    _verbale_assemblea_system,
    _nota_contestazione_system,
    DOCUMENT_TYPE_REGISTRY,
)

# ── Retirement list: covered by the 383, simply removed ──────────────────────

RETIRE_KEYS = {
    "opposition_act",
    "appeal",
    "power_of_attorney",
    "memoria_difensiva",
}

# ── Migration list: 16 templates → "Contratti e Atti Stragiudiziali" ─────────

NEW_CODICE = "Contratti e Atti Stragiudiziali"

MIGRATE_ENTRIES = [
    # ── Contratti di Locazione ────────────────────────────────────────────
    {
        "key": "rental_basic",
        "fn": _rental_basic_system,
        "categorie": ["1. CONTRATTI DI LOCAZIONE"],
        "tipo_atto": "Contratto di Locazione con Cedolare Secca",
        "filename": "contratti__contratto-di-locazione-cedolare-secca.docx",
    },
    {
        "key": "rental_standard",
        "fn": _rental_standard_system,
        "categorie": ["1. CONTRATTI DI LOCAZIONE"],
        "tipo_atto": "Contratto di Locazione Abitativa (3+2)",
        "filename": "contratti__contratto-di-locazione-abitativa-3-2.docx",
    },
    {
        "key": "rental_student",
        "fn": _rental_student_system,
        "categorie": ["1. CONTRATTI DI LOCAZIONE"],
        "tipo_atto": "Locazione Abitativa per Studenti Universitari",
        "filename": "contratti__locazione-studenti-universitari.docx",
    },
    {
        "key": "rental_transitional",
        "fn": _rental_transitional_system,
        "categorie": ["1. CONTRATTI DI LOCAZIONE"],
        "tipo_atto": "Locazione Abitativa di Natura Transitoria",
        "filename": "contratti__locazione-natura-transitoria.docx",
    },
    {
        "key": "rental_free_rent",
        "fn": _rental_free_rent_system,
        "categorie": ["1. CONTRATTI DI LOCAZIONE"],
        "tipo_atto": "Contratto di Locazione a Canone Libero (4+4)",
        "filename": "contratti__contratto-di-locazione-canone-libero.docx",
    },
    {
        "key": "rental_commercial",
        "fn": _rental_commercial_system,
        "categorie": ["1. CONTRATTI DI LOCAZIONE"],
        "tipo_atto": "Contratto di Locazione ad Uso Commerciale",
        "filename": "contratti__contratto-di-locazione-uso-commerciale.docx",
    },
    {
        "key": "rental_cancellation",
        "fn": _rental_cancellation_system,
        "categorie": ["1. CONTRATTI DI LOCAZIONE"],
        "tipo_atto": "Disdetta Contratto di Locazione",
        "filename": "contratti__disdetta-contratto-locazione.docx",
    },
    # ── Atti Assicurativi ─────────────────────────────────────────────────
    {
        "key": "insurance_cancellation",
        "fn": _insurance_cancellation_system,
        "categorie": ["2. ATTI ASSICURATIVI"],
        "tipo_atto": "Disdetta Polizza Assicurativa",
        "filename": "contratti__disdetta-polizza-assicurativa.docx",
    },
    {
        "key": "insurance_declaration",
        "fn": _insurance_declaration_system,
        "categorie": ["2. ATTI ASSICURATIVI"],
        "tipo_atto": "Dichiarazione Sostitutiva di Polizza Assicurativa",
        "filename": "contratti__dichiarazione-sostitutiva-polizza.docx",
    },
    # ── Diritto del Lavoro ────────────────────────────────────────────────
    {
        "key": "employment_dismissal_appeal",
        "fn": _employment_dismissal_appeal_system,
        "categorie": ["3. DIRITTO DEL LAVORO"],
        "tipo_atto": "Impugnativa di Licenziamento",
        "filename": "contratti__impugnativa-licenziamento.docx",
    },
    {
        "key": "employment_termination",
        "fn": _employment_termination_system,
        "categorie": ["3. DIRITTO DEL LAVORO"],
        "tipo_atto": "Lettera di Licenziamento per Giusta Causa",
        "filename": "contratti__lettera-licenziamento-giusta-causa.docx",
    },
    # ── Contratti Commerciali ─────────────────────────────────────────────
    {
        "key": "franchising_contract",
        "fn": _franchising_contract_system,
        "categorie": ["4. CONTRATTI COMMERCIALI"],
        "tipo_atto": "Contratto di Franchising",
        "filename": "contratti__contratto-di-franchising.docx",
    },
    {
        "key": "sale_agreement",
        "fn": _sale_agreement_system,
        "categorie": ["4. CONTRATTI COMMERCIALI"],
        "tipo_atto": "Contratto di Compravendita",
        "filename": "contratti__contratto-di-compravendita.docx",
    },
    # ── Atti Condominiali ─────────────────────────────────────────────────
    {
        "key": "verbale_assemblea",
        "fn": _verbale_assemblea_system,
        "categorie": ["5. ATTI CONDOMINIALI"],
        "tipo_atto": "Verbale di Assemblea Condominiale",
        "filename": "contratti__verbale-assemblea-condominiale.docx",
    },
    # ── Diffide e Messa in Mora ───────────────────────────────────────────
    {
        "key": "demand_letter",
        "fn": _demand_letter_system,
        "categorie": ["6. DIFFIDE E MESSA IN MORA"],
        "tipo_atto": "Diffida / Lettera di Messa in Mora",
        "filename": "contratti__diffida-lettera-messa-in-mora.docx",
    },
    # ── Contestazioni e Dichiarazioni ─────────────────────────────────────
    {
        "key": "nota_contestazione",
        "fn": _nota_contestazione_system,
        "categorie": ["7. CONTESTAZIONI E DICHIARAZIONI"],
        "tipo_atto": "Nota alla Contestazione",
        "filename": "contratti__nota-alla-contestazione.docx",
    },
]


# ── Section extraction ────────────────────────────────────────────────────────

def extract_sections_from_prompt(text: str) -> list[dict]:
    """Parse the numbered section list from an Italian system prompt string."""
    lines = text.split('\n')
    sections = []
    current_name = None
    current_desc_parts = []
    STOP_PREFIXES = ('Usa ', 'IMPORTANTE', 'È VIETATO', 'Solo ', 'Non aggiungere', 'Scrivi ')

    for line in lines:
        m = re.match(r'^\d+\.\s+([A-ZÀÈÌÒÙÉÁÍÓÚ\s/,()\-]+?)\s+—\s+(.*)', line)
        if m:
            if current_name:
                desc = ' '.join(current_desc_parts).strip()
                desc = re.split(r'\s+Usa\s', desc)[0].strip()
                sections.append({
                    "heading": current_name.strip().title(),
                    "content": f"[DA COMPILARE: {desc}]",
                })
            current_name = m.group(1).strip()
            current_desc_parts = [m.group(2).strip()]
        elif current_name and line.strip():
            if any(line.strip().startswith(p) for p in STOP_PREFIXES):
                break
            current_desc_parts.append(line.strip())

    if current_name:
        desc = ' '.join(current_desc_parts).strip()
        desc = re.split(r'\s+Usa\s', desc)[0].strip()
        if desc:
            sections.append({
                "heading": current_name.strip().title(),
                "content": f"[DA COMPILARE: {desc}]",
            })

    return sections


# ── .docx generation ──────────────────────────────────────────────────────────

def build_docx(tipo_atto: str, sections: list[dict], fields: list[str], output_path: str):
    doc = Document()
    h1 = doc.add_heading(tipo_atto, level=1)
    h1.style.font.size = Pt(14)

    for section in sections:
        h2 = doc.add_heading(section["heading"], level=2)
        h2.style.font.size = Pt(12)
        p = doc.add_paragraph(section["content"])

    # Campi strutturati
    doc.add_heading("Campi strutturati", level=2)
    for field in fields:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(field)

    doc.save(output_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates-dir", default="/opt/chatbot/data/system_templates")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    catalog_path = os.path.join(args.templates_dir, "catalog_enriched.json")
    with open(catalog_path, encoding="utf-8") as f:
        catalog = json.load(f)

    existing_filenames = {e["filename"] for e in catalog}
    new_entries = []
    skipped = []

    for entry in MIGRATE_ENTRIES:
        key = entry["key"]
        filename = entry["filename"]

        if filename in existing_filenames:
            skipped.append(filename)
            continue

        registry_entry = DOCUMENT_TYPE_REGISTRY.get(key, {})
        fields = registry_entry.get("fields", [])
        label = registry_entry.get("label", entry["tipo_atto"])

        # Extract sections from the Italian system prompt
        system_text = entry["fn"]("it")
        sections = extract_sections_from_prompt(system_text)

        if not sections:
            print(f"  [WARN] No sections extracted for {key} — skipping")
            continue

        output_path = os.path.join(args.templates_dir, filename)

        if args.dry_run:
            print(f"[DRY RUN] Would create: {filename}")
            print(f"  Sections ({len(sections)}): {[s['heading'] for s in sections]}")
            print(f"  Fields: {fields}")
        else:
            build_docx(entry["tipo_atto"], sections, fields, output_path)

        catalog_entry = {
            "filename": filename,
            "codice": NEW_CODICE,
            "categorie": entry["categorie"],
            "sottocategorie": [None],
            "tipo_atto": entry["tipo_atto"],
            "label": label,
            "description": f"Template per: {label}",
            "fields": fields,
            "sections": sections,
            "flagged_issues": [],
        }
        new_entries.append(catalog_entry)

    if not args.dry_run and new_entries:
        catalog.extend(new_entries)
        with open(catalog_path, "w", encoding="utf-8") as f:
            json.dump(catalog, f, ensure_ascii=False, indent=2)

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Done.")
    print(f"Migrated: {len(new_entries)} templates")
    print(f"Retired (no catalog entry created): {len(RETIRE_KEYS)} templates")
    if skipped:
        print(f"Skipped (already in catalog): {len(skipped)}")
    if new_entries:
        print(f"\nNew entries:")
        for e in new_entries:
            print(f"  {e['filename']} ({e['categorie'][0]})")
    print(f"\nTotal catalog entries: {len(catalog) + len(new_entries) if args.dry_run else len(catalog)}")
    print()
    print("Next step: patch document_generation.py to remove the 20 legacy")
    print("keys from DOCUMENT_TYPE_REGISTRY and _SYSTEM_FN, then restart chatbot.")


if __name__ == "__main__":
    main()
