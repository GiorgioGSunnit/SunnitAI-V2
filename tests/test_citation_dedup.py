"""_extract_citations' section key: (name, normalized text[:200]), not name alone.

A numbered article's up to 7 duplicate Section nodes (same name, same text,
different id) must still collapse to one citation. Commentato/Fiscalita
chunks derive `name` from a truncated prose snippet and can share one across
genuinely distinct chunks (different text) — keying on name alone silently
dropped one chunk's content (or left a doc with sections: [] when the
surviving chunk failed the length filter). Keying on (name, text[:200])
collapses true duplicates and keeps distinct chunks apart. Falls back to
node_id only when a row has neither a name nor any text.
"""

from src.rag.answer_processing import _extract_citations


def _row(sid, name="584.0.0", text="plain text content here, long enough", doc="LEGAL_DOC::cp"):
    return {
        "d": {"id": doc, "name": "Codice Penale 2026"},
        "s": {"id": f"DOCUMENT_SECTION::{sid}", "name": name, "plain_text": text},
    }


def test_true_duplicate_copies_collapse_to_one_section():
    rows = [_row(f"dup{i}") for i in range(6)]
    citations = _extract_citations(rows)
    assert len(citations) == 1
    assert len(citations[0]["sections"]) == 1


def test_true_duplicates_collapse_despite_whitespace_differences():
    rows = [
        _row("e", name="584.0.0", text="Chiunque partecipa a una rissa e' punito"),
        _row("f", name="584.0.0", text="Chiunque   partecipa a una rissa e' punito  "),
    ]
    citations = _extract_citations(rows)
    assert len(citations) == 1
    assert len(citations[0]["sections"]) == 1


def test_distinct_section_names_are_not_collapsed():
    rows = [
        _row("a", name="corte_cassazione.1", text="testo A lungo abbastanza"),
        _row("b", name="corte_cassazione.2", text="testo B lungo abbastanza"),
    ]
    citations = _extract_citations(rows)
    assert len(citations) == 1
    assert len(citations[0]["sections"]) == 2


def test_same_truncated_name_but_different_text_are_not_collapsed():
    """The regression this fix targets: two distinct dottrina/commentato
    chunks sharing one truncated-prose name (e.g. both named after the same
    cited ruling) must both survive, since their actual text differs."""
    rows = [
        _row("c", name="2011) , . Difforme, Cass.", text="commento A, tutt'altro contenuto"),
        _row("d", name="2011) , . Difforme, Cass.", text="commento B, contenuto completamente diverso"),
    ]
    citations = _extract_citations(rows)
    assert len(citations) == 1
    assert len(citations[0]["sections"]) == 2


def test_rows_without_a_name_or_text_fall_back_to_node_id():
    rows = [
        {"d": {"id": "LEGAL_DOC::cp", "name": "Codice Penale 2026"},
         "s": {"id": "DOCUMENT_SECTION::g", "name": "", "plain_text": ""}},
    ]
    # No name and no text: fails the length gate regardless of key, so the
    # document entry exists with no sections — the fallback just avoids a
    # crash/collision on an empty key, not a content assertion here.
    citations = _extract_citations(rows)
    assert len(citations) == 1
    assert citations[0]["sections"] == []
