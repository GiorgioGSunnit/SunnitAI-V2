"""
Regenerates only the entries currently in flagged_for_review.json, using
the (now-fixed) prompt in generate_skeleton_templates.py. Replaces their
old .docx files and catalog.json entries in place — does not touch any
of the other already-good entries.

After regenerating, re-runs the P.Q.M. redundancy scan on just the
regenerated files to confirm the fix actually worked before declaring
them resolved.

Usage:
  python3 regenerate_flagged.py --templates-dir /opt/chatbot/data/system_templates
"""

import argparse
import json
import os

from generate_skeleton_templates import process_entry
from scan_pqm_redundancy import scan_file


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

    flagged_filenames = {f["filename"] for f in flagged}
    print(f"Regenerating {len(flagged_filenames)} flagged entries...")

    # Delete the old .docx files so process_entry doesn't skip them
    for fname in flagged_filenames:
        path = os.path.join(args.templates_dir, fname)
        if os.path.exists(path):
            os.remove(path)

    # Build entry dicts in the shape process_entry expects, from catalog data
    entries_to_regen = [
        {
            "codice": c["codice"],
            "categorie": c["categorie"],
            "sottocategorie": c["sottocategorie"],
            "tipo_atto": c["tipo_atto"],
        }
        for c in catalog
        if c["filename"] in flagged_filenames
    ]

    new_results = []
    for entry in entries_to_regen:
        result = process_entry(entry, args.templates_dir)
        if result["status"] == "ok":
            new_results.append(result["catalog_entry"])
        else:
            print(f"  [FAIL] {entry['tipo_atto']} -> {result['status']}")

    # Replace old catalog entries with new ones, by filename
    new_by_filename = {r["filename"]: r for r in new_results}
    catalog = [new_by_filename.get(c["filename"], c) for c in catalog]

    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)

    # Re-scan the regenerated files specifically to confirm the fix worked
    still_bad = []
    for r in new_results:
        path = os.path.join(args.templates_dir, r["filename"])
        if scan_file(path):
            still_bad.append(r["filename"])

    print(f"\nRegenerated: {len(new_results)}")
    print(f"Still showing Conclusioni+P.Q.M. redundancy after fix: {len(still_bad)}")
    for fname in still_bad:
        print(f"  {fname}")

    # Rebuild flagged_for_review.json: keep only entries still genuinely
    # flagged (still_bad, plus any non-P.Q.M. flagged entries we didn't touch)
    remaining_flagged = [
        f for f in flagged
        if f["filename"] not in flagged_filenames or f["filename"] in still_bad
    ]
    with open(flagged_path, "w", encoding="utf-8") as f:
        json.dump(remaining_flagged, f, ensure_ascii=False, indent=2)
    print(f"\nflagged_for_review.json now has {len(remaining_flagged)} entries")


if __name__ == "__main__":
    main()
