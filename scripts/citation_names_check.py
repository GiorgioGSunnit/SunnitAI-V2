#!/usr/bin/env python3
"""v60e: every code name in the store read back by the one citation reading (src/rag/citations.self_check): «art. N
<name>», «art. N, comma 2, <name>», «artt. N e M <name>» and the §2 heading «**Art. N <name> — …**». Read only.

  python scripts/citation_names_check.py          # prints the failing names and forms; exit 1 when any
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from src.rag import citations
    from src.rag.doc_lookup import _fetch_doc_names
    from src.rag.main import NEO4J_DATABASE, driver
    documents = _fetch_doc_names(driver, NEO4J_DATABASE)
    failures = citations.self_check(documents)
    for f in failures:
        print(f"{f['name']}: {f['form']} -> {f['read']}")
    print(f"{len(citations.store_names(documents))} names, {len(failures)} failing forms")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
