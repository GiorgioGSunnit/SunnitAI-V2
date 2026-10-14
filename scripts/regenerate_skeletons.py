#!/usr/bin/env python3
"""
Regenerate sparse skeleton templates in catalog_enriched.json and matching .docx files.

Skeletons that contain only [DA COMPILARE: ...] instructions are replaced with
real Italian legal prose containing [DA COMPILARE] placeholders inline.

Usage:
    cd /opt/chatbot && source venv/bin/activate
    python3 scripts/regenerate_skeletons.py --batch-size 50 --batch 0
    python3 scripts/regenerate_skeletons.py --batch-size 50 --batch 1
    ...

    Or run a specific file:
    python3 scripts/regenerate_skeletons.py --filename penale__nomina-difensore-di-fiducia.docx

    Or dry-run to see what would be regenerated:
    python3 scripts/regenerate_skeletons.py --dry-run --batch-size 50 --batch 0

The script is resumable: it skips entries that already have real prose content
(i.e. section content with >30 chars of text outside [DA COMPILARE...] blocks).
Use --force to regenerate even already-rich entries.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/opt/chatbot/logs/skeleton_regen.log", mode="a"),
    ],
)
logger = logging.getLogger("skeleton_regen")

CATALOG_PATH = Path("/opt/chatbot/data/system_templates/catalog_enriched.json")
SKELETON_DIR = Path("/opt/chatbot/data/system_templates")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://217.160.8.129:8000/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = "qwen2.5-32b-awq"
MAX_TOKENS = 2000


# ── LLM call ─────────────────────────────────────────────────────────────────

def call_llm(system: str, human: str, max_tokens: int = MAX_TOKENS) -> str:
    import urllib.request

    payload = json.dumps({
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": human},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }).encode()

    req = urllib.request.Request(
        f"{LLM_BASE_URL}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLM_API_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"].strip()


# ── Skeleton detection ────────────────────────────────────────────────────────

def is_instruction_only(entry: dict) -> bool:
    """Return True if all section content is just [DA COMPILARE: ...] instructions."""
    for s in entry.get("sections", []):
        content = s.get("content", "")
        stripped = re.sub(r"\[DA COMPILARE[^\]]*\]", "", content).strip()
        if len(stripped) > 30:
            return False
    return True


# ── Skeleton generation ───────────────────────────────────────────────────────

def generate_skeleton_sections(entry: dict) -> list[dict]:
    """
    Call the LLM to generate real Italian legal prose for each section.
    Returns a new sections list with proper [DA COMPILARE] placeholders inline.
    """
    label = entry.get("label") or entry.get("tipo_atto", "")
    codice = entry.get("codice", "")
    description = entry.get("description", "")
    fields = entry.get("fields", [])
    existing_sections = entry.get("sections", [])
    section_headings = [s["heading"] for s in existing_sections
                        if s["heading"] != "Campi strutturati"]

    fields_desc = ", ".join(fields) if fields else "nessuno specificato"
    headings_list = "\n".join(f"- {h}" for h in section_headings)

    system = (
        "Sei un avvocato esperto di diritto processuale italiano. "
        "Devi scrivere il contenuto testuale di uno schema di atto giuridico italiano. "
        "Ogni sezione deve contenere testo legale reale in italiano giuridico formale, "
        "con i dati variabili sostituiti dal placeholder [DA COMPILARE]. "
        "\n\n"
        "REGOLE FONDAMENTALI:\n"
        "1. Scrivi testo legale vero e completo, non istruzioni su cosa scrivere\n"
        "2. Usa [DA COMPILARE] inline nel testo per ogni dato variabile\n"
        "3. NON scrivere '[DA COMPILARE: descrizione]' — solo '[DA COMPILARE]'\n"
        "4. Il testo deve essere formalmente corretto e usabile direttamente\n"
        "5. Rispetta la struttura delle sezioni indicate\n"
        "6. Rispondi SOLO con JSON valido, nessun testo aggiuntivo\n"
        "\n"
        "Esempio di output CORRETTO per una sezione 'Intestazione' di una nomina:\n"
        '{"heading": "Intestazione", "content": "Il/La sottoscritto/a [DA COMPILARE], '
        'nato/a a [DA COMPILARE] il [DA COMPILARE], residente in [DA COMPILARE], '
        'C.F. [DA COMPILARE]"}\n'
        "\n"
        "Esempio di output SBAGLIATO:\n"
        '{"heading": "Intestazione", "content": "[DA COMPILARE: inserire nome e cognome, '
        'data di nascita, residenza e codice fiscale]"}'
    )

    human = (
        f"Tipo di atto: {label}\n"
        f"Codice: {codice}\n"
        f"Descrizione: {description}\n"
        f"Campi disponibili (dati che l'utente può fornire): {fields_desc}\n"
        f"\nSezioni da compilare:\n{headings_list}\n"
        f"\nGenera il contenuto per ogni sezione (escludi 'Campi strutturati'). "
        f"Rispondi con un array JSON:\n"
        f'[{{"heading": "NomeSezione", "content": "testo legale con [DA COMPILARE] inline"}}, ...]'
    )

    raw = call_llm(system, human)

    # Strip markdown fences
    clean = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    try:
        sections = json.loads(clean)
        if not isinstance(sections, list):
            raise ValueError("Expected a list")
        # Validate each section has heading and content
        for s in sections:
            if "heading" not in s or "content" not in s:
                raise ValueError(f"Missing heading or content in section: {s}")
        return sections
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("JSON parse failed for %s: %s — raw: %s", label, e, raw[:200])
        # Try to extract JSON array from response
        match = re.search(r"\[.*\]", clean, re.DOTALL)
        if match:
            try:
                sections = json.loads(match.group())
                return sections
            except Exception:
                pass
        raise RuntimeError(f"Could not parse LLM response for {label}: {raw[:300]}")


# ── DOCX writer ───────────────────────────────────────────────────────────────

def write_docx(entry: dict, sections: list[dict], path: Path) -> None:
    """Write a skeleton .docx file with Heading 1 (title), Heading 2 + Normal (sections),
    and Heading 2 + List Bullet (fields)."""
    from docx import Document
    from docx.shared import Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = Document()

    # Title
    label = entry.get("label") or entry.get("tipo_atto", "")
    title = doc.add_heading(label, level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Sections (real prose)
    for s in sections:
        if s.get("heading") == "Campi strutturati":
            continue
        doc.add_heading(s["heading"], level=2)
        doc.add_paragraph(s.get("content", "[DA COMPILARE]"))

    # Campi strutturati (field names)
    fields = entry.get("fields", [])
    if fields:
        doc.add_heading("Campi strutturati", level=2)
        for f in fields:
            doc.add_paragraph(f, style="List Bullet")

    doc.save(str(path))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Regenerate sparse skeleton templates")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Number of skeletons to process per run (default: 50)")
    parser.add_argument("--batch", type=int, default=0,
                        help="Batch index (0-based, default: 0)")
    parser.add_argument("--filename", type=str, default=None,
                        help="Process a single specific filename")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even already-rich skeletons")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be regenerated without making changes")
    args = parser.parse_args()

    # Load catalog
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)

    # Select targets
    if args.filename:
        targets = [e for e in catalog if e["filename"] == args.filename]
        if not targets:
            logger.error("Filename %r not found in catalog", args.filename)
            sys.exit(1)
    else:
        if args.force:
            candidates = catalog
        else:
            candidates = [e for e in catalog if is_instruction_only(e)]

        logger.info("Total candidates for regeneration: %d", len(candidates))

        start = args.batch * args.batch_size
        end = start + args.batch_size
        targets = candidates[start:end]
        logger.info(
            "Processing batch %d: entries %d-%d (%d total)",
            args.batch, start, min(end, len(candidates)), len(targets)
        )

    if not targets:
        logger.info("No targets in this batch — all done or batch out of range.")
        return

    if args.dry_run:
        print(f"\nDRY RUN — would regenerate {len(targets)} skeletons:")
        for e in targets:
            print(f"  {e['filename']}")
        return

    # Build filename→index map for fast catalog updates
    catalog_index = {e["filename"]: i for i, e in enumerate(catalog)}

    success = 0
    failed = 0

    for i, entry in enumerate(targets):
        fname = entry["filename"]
        docx_path = SKELETON_DIR / fname
        logger.info("[%d/%d] Regenerating %s", i + 1, len(targets), fname)

        try:
            # Generate new sections
            new_sections = generate_skeleton_sections(entry)

            # Add back Campi strutturati to sections list
            fields = entry.get("fields", [])
            if fields:
                new_sections_with_fields = new_sections + [{
                    "heading": "Campi strutturati",
                    "content": "\n".join(fields),
                }]
            else:
                new_sections_with_fields = new_sections

            # Update catalog in memory
            idx = catalog_index[fname]
            catalog[idx]["sections"] = new_sections_with_fields

            # Write .docx
            write_docx(entry, new_sections, docx_path)

            logger.info("  ✓ Done — %d sections, docx written to %s", len(new_sections), docx_path)
            success += 1

            # Save catalog after every entry (resumable)
            with open(CATALOG_PATH, "w", encoding="utf-8") as f:
                json.dump(catalog, f, ensure_ascii=False, indent=2)

            # Small delay to avoid hammering the inference server
            time.sleep(0.5)

        except Exception as exc:
            logger.error("  ✗ FAILED %s: %s", fname, exc)
            failed += 1
            continue

    logger.info(
        "\nBatch %d complete — %d succeeded, %d failed",
        args.batch, success, failed
    )
    logger.info(
        "Next batch: python3 scripts/regenerate_skeletons.py "
        "--batch-size %d --batch %d",
        args.batch_size, args.batch + 1
    )


if __name__ == "__main__":
    main()
