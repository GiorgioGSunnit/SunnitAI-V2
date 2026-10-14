"""
Bulk-registers all generated skeleton templates (system_templates/*.docx)
under the document-generation system user account, using the EXACT same
storage convention as the real upload_document() endpoint in
src/chatbot/routes/documents.py:

  - Files are copied to PRIVATE_DOCS_BASE/{tenant_id}/{user_id}/{uuid4()}.docx
  - A matching row is inserted into user_documents via create_user_document(),
    with original_filename set to the template's friendly label

This means these 383 templates show up through the existing
get_user_documents() retrieval path exactly like any document a real
user uploaded through the UI — no schema changes needed.

Run from /opt/chatbot so the existing src/ package imports resolve.

Usage:
  python3 scripts/register_system_templates.py \
      --templates-dir /opt/chatbot/data/system_templates \
      --user-id f9ac7571-bda9-4b78-bd85-f312ec18d87b \
      --tenant-id 072c56c5-0757-4385-9655-3cc3d869fc4b
"""

import argparse
import json
import os
import shutil
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants import PRIVATE_DOCS_BASE, SUPPORTED_EXTENSIONS
from src.db.base import SessionLocal
from src.db.crud import create_user_document, get_user_documents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates-dir", default="/opt/chatbot/data/system_templates")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without writing anything")
    args = parser.parse_args()

    user_id = uuid.UUID(args.user_id)
    tenant_id = uuid.UUID(args.tenant_id)

    catalog_path = os.path.join(args.templates_dir, "catalog.json")
    with open(catalog_path, encoding="utf-8") as f:
        catalog = json.load(f)

    print(f"Found {len(catalog)} templates in catalog.json")

    db = SessionLocal()

    # Avoid re-registering templates already present for this user (safe to
    # re-run) — check existing original_filename values first.
    existing = get_user_documents(db, user_id, tenant_id)
    existing_filenames = {d.original_filename for d in existing}
    print(f"User already has {len(existing)} documents on file")

    registered = 0
    skipped_existing = 0
    skipped_missing = 0
    skipped_bad_ext = 0

    target_folder = os.path.join(PRIVATE_DOCS_BASE, str(tenant_id), str(user_id))
    if not args.dry_run:
        os.makedirs(target_folder, exist_ok=True)

    for entry in catalog:
        label = entry.get("label") or entry["tipo_atto"]
        original_filename = f"{label}.docx"

        if original_filename in existing_filenames:
            skipped_existing += 1
            continue

        src_path = os.path.join(args.templates_dir, entry["filename"])
        if not os.path.exists(src_path):
            print(f"  [MISSING] {entry['filename']}")
            skipped_missing += 1
            continue

        ext = os.path.splitext(src_path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            print(f"  [BAD EXT] {entry['filename']} ({ext})")
            skipped_bad_ext += 1
            continue

        unique_name = f"{uuid.uuid4()}{ext}"
        storage_path = os.path.join(target_folder, unique_name)

        if args.dry_run:
            print(f"  [DRY RUN] would copy {entry['filename']} -> {storage_path} (original_filename={original_filename!r})")
            registered += 1
            continue

        shutil.copy2(src_path, storage_path)
        file_size = os.path.getsize(storage_path)

        create_user_document(
            db,
            user_id=user_id,
            tenant_id=tenant_id,
            original_filename=original_filename,
            storage_path=storage_path,
            file_size_bytes=file_size,
        )
        registered += 1

    db.close()

    print(f"\nRegistered: {registered}")
    print(f"Skipped (already existed): {skipped_existing}")
    print(f"Skipped (file missing on disk): {skipped_missing}")
    print(f"Skipped (bad extension): {skipped_bad_ext}")
    if args.dry_run:
        print("\nThis was a DRY RUN — nothing was actually written. Re-run without --dry-run to commit.")


if __name__ == "__main__":
    main()
