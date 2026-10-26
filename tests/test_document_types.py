"""scripts/document_types.py: a type proposed for documents with none, from the name, else the Sections."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "document_types", Path(__file__).resolve().parent.parent / "scripts" / "document_types.py")
document_types = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(document_types)


@pytest.mark.parametrize("name, expected", [
    ("Codice della Navigazione 2026", "primary"),
    ("D.Lgs. 81/2008 Testo unico sicurezza", "primary"),
    ("CCNL Commercio Terziario", "ccnl"),
    ("Cass. civ., sez. III, sent. n. 1234/2025", "interpretation"),
    ("Manuale di diritto privato", "special"),
])
def test_the_name_proposes_a_type(name, expected):
    assert document_types.propose(name, [])[0] == expected


def test_article_shaped_sections_propose_primary_and_nothing_else_proposes_nothing():
    assert document_types.propose("Raccolta 2026", ["1.0.0", "2.0.0", "2.1.0", "3"])[0] == "primary"
    assert document_types.propose("Raccolta 2026", ["premessa", "capitolo_1", "1.0.0"])[0] is None


def test_set_takes_only_known_types():
    assert document_types.parse_overrides(["abc=special"]) == {"abc": "special"}
    with pytest.raises(SystemExit):
        document_types.parse_overrides(["abc=secondary"])
