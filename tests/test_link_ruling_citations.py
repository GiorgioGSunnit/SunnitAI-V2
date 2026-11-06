"""link_ruling_citations.py: the incremental selection (--new-only reads only rulings without the stamp) and the
stamp's name, which the nightly line and the health report rely on."""
import importlib.util
import os
from pathlib import Path

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("link_ruling_citations", HERE.parent / "scripts" / "link_ruling_citations.py")
lrc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lrc)


def test_full_pass_reads_every_ruling_and_new_only_skips_the_stamped_ones():
    assert lrc.STAMP == "cites_linked_at"
    assert lrc.ruling_query(False) == \
        "MATCH (d:Document {document_type: 'interpretation'}) RETURN d.id AS id ORDER BY id"
    assert lrc.ruling_query(True) == \
        "MATCH (d:Document {document_type: 'interpretation'}) WHERE d.cites_linked_at IS NULL RETURN d.id AS id ORDER BY id"
    assert lrc.ruling_query(True, 50).endswith(" ORDER BY id LIMIT $n")
    assert lrc.ruling_query(False, 0).endswith(" ORDER BY id")

    assert "--new-only" in lrc.__doc__ and "cites_linked_at" in lrc.__doc__
