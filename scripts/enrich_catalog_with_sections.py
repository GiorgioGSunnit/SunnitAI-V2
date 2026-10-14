"""
One-time enrichment: parses each already-generated skeleton .docx back
into a structured sections list (heading + content per section) and
writes catalog_enriched.json — same as catalog.json but with a "sections"
field added to every entry.

Pure parsing, no LLM calls, no risk of altering content — reconstructs
exactly what build_docx() wrote, just back into JSON form so the runtime
generation code doesn't need to re-parse .docx files on every request.

Usage:
  python3 enrich_catalog_with_sections.py --templates-dir /opt/chatbot/data/system_templates
"""

import argparse
import json
import os

from docx import Document


def extract_sections(path: str) -> list[dict]:
    doc = Document(path)
    sections = []
    current = None
    for p in doc.paragraphs:
        style = p.style.name
        text = p.text.strip()
        if not text:
            continue
        if style == "Heading 2":
            if text == "Campi strutturati":
                # Stop before the trailing "Campi strutturati" bullet list —
                # that's redundant with the entry's existing "fields" key.
                current = None
                continue
            current = {"heading": text, "content": ""}
            sections.append(current)
        elif style == "Normal" and current is not None:
            current["content"] = (current["content"] + " " + text).strip() if current["content"] else text
    return sections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates-dir", default="/opt/chatbot/data/system_templates")
    args = parser.parse_args()

    catalog_path = os.path.join(args.templates_dir, "catalog.json")
    enriched_path = os.path.join(args.templates_dir, "catalog_enriched.json")

    with open(catalog_path, encoding="utf-8") as f:
        catalog = json.load(f)

    enriched = []
    failed = []
    for entry in catalog:
        path = os.path.join(args.templates_dir, entry["filename"])
        if not os.path.exists(path):
            failed.append(entry["filename"])
            continue
        sections = extract_sections(path)
        new_entry = dict(entry)
        new_entry["sections"] = sections
        enriched.append(new_entry)

    with open(enriched_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    print(f"Enriched {len(enriched)} entries with structured sections")
    if failed:
        print(f"Failed to find on disk: {len(failed)}")
        for f in failed:
            print(f"  {f}")
    print(f"Saved to: {enriched_path}")

    # Quick sanity check on one entry
    if enriched:
        sample = enriched[0]
        print(f"\nSample entry ({sample['filename']}):")
        print(f"  {len(sample['sections'])} sections extracted")
        for s in sample["sections"][:2]:
            print(f"    - {s['heading']}: {s['content'][:60]}...")


if __name__ == "__main__":
    main()
