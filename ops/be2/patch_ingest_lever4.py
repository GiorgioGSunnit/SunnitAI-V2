"""Lever 4 on be2 (prepared 7 Nov 2026): the cheap path for Cassazione rulings, behind one setting.

INGEST_SKIP_ANALYSIS=1 in a worker's environment file (/opt/sunnitai-be/.env.worker2 for worker 2 only) makes that
worker ingest a file without any chat-model call. The four places that call the model per file become no-ops:

  1. lex_package/analisi_parallel.py  chunk analysis (~50 calls per ruling): every chunk is kept raw, as the
     module already does for chunks under 30 characters; requirement / core_text / pattern_type stay empty, so
     the section abstract becomes the first 200 words of the section text. One journal line per document:
     "[SKIP ANALYSIS] INGEST_SKIP_ANALYSIS set: N chunks kept raw, no LLM analysis".
  2. lex_package/utils/flatten.py      title / number / date extraction from the first pages (1 call): skipped;
     the document keeps the sidecar's display name and the date found in it, document_number stays NULL.
  3. lex_package/utils/post_process.py _generate_section_abstract (short sections): returns the abstract as is.
  4. lex_package/utils/post_process.py _generate_section_title (sections without a parser title): no title.

Embeddings are still computed (the measured 2025 no-analysis rulings went in this way, minus their error strings).
Without the setting nothing changes: every edit is guarded by the environment variable.

The script checks the three files against the checksums of be2's copy after lever 3 (6 Nov), refuses to touch a
file that differs, backs each up with a dated suffix, applies, and can undo.

    sudo python3 /home/sunnit/patch_ingest_lever4.py --check    # checksums only, changes nothing
    sudo python3 /home/sunnit/patch_ingest_lever4.py            # apply (prints what changed)
    sudo python3 /home/sunnit/patch_ingest_lever4.py --undo     # restore the backups
    echo 'INGEST_SKIP_ANALYSIS=1' | sudo tee -a /opt/sunnitai-be/.env.worker2 && sudo systemctl restart sunnitai-ingest-2
    # off again: delete that line from .env.worker2 and restart worker 2. Worker 1 never reads .env.worker2.

Nothing else in the files is touched; no service is restarted by this script.
"""
import hashlib
import shutil
import sys
import datetime as dt

ROOT = "/opt/sunnitai-be"
LEX = f"{ROOT}/src/be/src/lex_package"
SKIP = 'os.environ.get("INGEST_SKIP_ANALYSIS", "").strip().lower() in ("1", "true", "yes")'
FILES = {
    f"{LEX}/analisi_parallel.py": {
        "md5": "db2e23f06aaed0dec2ba943aa53c881f",          # after lever 3 (6 Nov 2026)
        "edits": [
            ("    _pre_filter = len(flat_inputs)\n"
             "    _keep  = [i for i, c in enumerate(flat_chunk_contents) if len(c.strip()) >= 30]\n"
             "    _short = [i for i, c in enumerate(flat_chunk_contents) if len(c.strip()) < 30]\n"
             "    _skipped = _pre_filter - len(_keep)\n",
             "    _pre_filter = len(flat_inputs)\n"
             "    _keep  = [i for i, c in enumerate(flat_chunk_contents) if len(c.strip()) >= 30]\n"
             "    _short = [i for i, c in enumerate(flat_chunk_contents) if len(c.strip()) < 30]\n"
             f"    if {SKIP}:\n"
             "        # Cheap path (Nov 2026 Cassazione batches): no chat-model call for any chunk. Every comma keeps\n"
             "        # its raw text, as the short chunks below already do; requirement / core_text / pattern_type\n"
             "        # stay empty, so the section abstract becomes the first 200 words of the text.\n"
             "        logger.info(\"[SKIP ANALYSIS] INGEST_SKIP_ANALYSIS set: %d chunks kept raw, no LLM analysis\", _pre_filter)\n"
             "        _keep, _short = [], list(range(_pre_filter))\n"
             "    _skipped = _pre_filter - len(_keep)\n"),
        ],
    },
    f"{LEX}/utils/flatten.py": {
        "md5": "0a4ec483afa574986bf6df89ee61754d",
        "edits": [
            ("import hashlib\nimport logging\nimport re\n", "import hashlib\nimport logging\nimport os\nimport re\n"),
            ("    extracted_metadata = None\n"
             "    if pdf_path:\n"
             "        try:\n"
             "            from lex_package.utils.metadata_extraction import extract_document_metadata\n",
             "    extracted_metadata = None\n"
             f"    if pdf_path and {SKIP}:\n"
             "        # Cheap path: no chat-model call for title / number / date; the sidecar name and its date are used\n"
             "        logger.info(\"[SKIP ANALYSIS] metadata extraction skipped for '%s'\", document_name)\n"
             "    elif pdf_path:\n"
             "        try:\n"
             "            from lex_package.utils.metadata_extraction import extract_document_metadata\n"),
        ],
    },
    f"{LEX}/utils/post_process.py": {
        "md5": "2303c3e6a95a44fe3bdd8ce39a2830cf",
        "edits": [
            ("    import os, requests\n"
             "    llm_base = os.getenv(\"LLM_BASE_URL\", \"\").rstrip(\"/\")\n"
             "    if not llm_base:\n"
             "        return existing_abstract\n",
             "    import os, requests\n"
             "    llm_base = os.getenv(\"LLM_BASE_URL\", \"\").rstrip(\"/\")\n"
             f"    if not llm_base or {SKIP}:\n"
             "        return existing_abstract   # INGEST_SKIP_ANALYSIS: cheap path, no model call\n"),
            ("    import os, requests\n"
             "    llm_base = os.getenv(\"LLM_BASE_URL\", \"\").rstrip(\"/\")\n"
             "    if not llm_base:\n"
             "        return \"\"\n",
             "    import os, requests\n"
             "    llm_base = os.getenv(\"LLM_BASE_URL\", \"\").rstrip(\"/\")\n"
             f"    if not llm_base or {SKIP}:\n"
             "        return \"\"   # INGEST_SKIP_ANALYSIS: cheap path, no model call\n"),
        ],
    },
}
SUFFIX = ".bak_lever4_" + dt.date.today().strftime("%d%b").lower()


def md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def check():
    ok = True
    for path, spec in FILES.items():
        have = md5(path)
        same = have == spec["md5"]
        ok &= same
        print(f"{'OK  ' if same else 'DIFF'} {path} md5 {have}" + ("" if same else f" (expected {spec['md5']})"))
    print("ready to apply" if ok else "STOP: a file is not the expected copy; nothing would be changed")
    return 0 if ok else 2


def apply():
    for path, spec in FILES.items():
        have = md5(path)
        if have != spec["md5"]:
            sys.exit(f"STOP: {path} is not the expected copy (md5 {have}); nothing changed")
    texts = {}
    for path, spec in FILES.items():
        with open(path, encoding="utf-8") as fh:
            s = fh.read()
        for old, new in spec["edits"]:
            if s.count(old) != 1:
                sys.exit(f"STOP: expected text not found exactly once in {path}: {old[:60]!r}; nothing changed")
        texts[path] = s
    for path, spec in FILES.items():
        backup = path + SUFFIX
        shutil.copy2(path, backup)
        s = texts[path]
        for old, new in spec["edits"]:
            s = s.replace(old, new)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(s)
        print(f"patched {path} ({len(spec['edits'])} edits); backup {backup}")
    print("done: the setting INGEST_SKIP_ANALYSIS=1 in .env.worker2 + restart of sunnitai-ingest-2 turns the cheap path on")


def undo():
    for path in FILES:
        backup = path + SUFFIX
        try:
            shutil.copy2(backup, path)
            print(f"restored {path} from {backup}")
        except FileNotFoundError:
            print(f"no backup for {path} with today's suffix ({backup}): look for *.bak_lever4_*")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--check" in args:
        sys.exit(check())
    undo() if "--undo" in args else apply()
