"""
Post-hoc structural scan of already-generated skeleton .docx files.

Checks for a pattern observed during manual review: when a skeleton has
BOTH a separate "Conclusioni" heading AND a separate "P.Q.M." heading,
the P.Q.M. section reliably drifts into unrelated/wrong content (seen:
cauzione details, notification-recipient details, mezzi di prova content
— a different wrong topic each time, but always the same root cause:
P.Q.M. has nothing distinct left to say once Conclusioni already covered
the request).

Does not call the LLM and does not modify any .docx — purely flags
affected files in flagged_for_review.json for the lawyer review pass.

Usage:
  python3 scan_pqm_redundancy.py --templates-dir /opt/chatbot/data/system_templates
"""

import argparse
import json
import os
import re

from docx import Document


def scan_file(path: str) -> bool:
    """Returns True if the file has both a Conclusioni and a P.Q.M. heading."""
    doc = Document(path)
    headings = [
        p.text.strip().lower()
        for p in doc.paragraphs
        if p.style.name.startswith("Heading")
    ]
    has_conclusioni = any("conclusioni" in h for h in headings)
    has_pqm = any(re.search(r"p\.?\s*q\.?\s*m\.?", h) for h in headings)
    return has_conclusioni and has_pqm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates-dir", default="/opt/chatbot/data/system_templates")
    args = parser.parse_args()

    catalog_path = os.path.join(args.templates_dir, "catalog.json")
    flagged_path = os.path.join(args.templates_dir, "flagged_for_review.json")

    with open(catalog_path, encoding="utf-8") as f:
        catalog = json.load(f)
    with open(flagged_path, encoding="utf-8") as f:
        flagged = json.load(f)

    already_flagged_filenames = {f["filename"] for f in flagged}
    by_filename = {c["filename"]: c for c in catalog}

    newly_flagged = []
    for entry in catalog:
        path = os.path.join(args.templates_dir, entry["filename"])
        if not os.path.exists(path):
            continue
        if scan_file(path):
            if entry["filename"] not in already_flagged_filenames:
                entry.setdefault("flagged_issues", [])
                entry["flagged_issues"].append(
                    "Contiene sia 'Conclusioni' che 'P.Q.M.' come sezioni separate "
                    "— verificare che P.Q.M. non contenga contenuto fuori tema"
                )
                newly_flagged.append(entry)

    all_flagged = flagged + newly_flagged
    with open(flagged_path, "w", encoding="utf-8") as f:
        json.dump(all_flagged, f, ensure_ascii=False, indent=2)

    # Also update catalog.json with the new flagged_issues
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)

    print(f"Scanned {len(catalog)} files")
    print(f"Newly flagged (Conclusioni+P.Q.M. redundancy): {len(newly_flagged)}")
    print(f"Total flagged (all reasons): {len(all_flagged)}")
    for e in newly_flagged:
        print(f"  {e['filename']}")


if __name__ == "__main__":
    main()
