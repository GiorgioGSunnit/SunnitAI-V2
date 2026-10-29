"""scripts/quarantine_docs.py: hiding private test uploads from every search, never deleting (29 Oct)."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("quarantine_docs", HERE.parent / "scripts" / "quarantine_docs.py")
qd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qd)


def test_kind_of():
    assert qd.kind_of("Verbale di Sommarie Informazioni - X") == ("private-looking", "verbale di sommarie")
    assert qd.kind_of("2026-05-timesheet-x-firmato") == ("private-looking", "timesheet")
    assert qd.kind_of("Ordinanza 43/2026 (ECLI:IT:COST:2026:43)") == ("court ruling", "ordinanza")
    assert qd.kind_of("Codice Civile 2026") == ("other", "")
    assert qd.kind_of("Legge sulle locazioni abitative 2023") == ("other", "")
    assert qd.kind_of(None) == ("other", "")


def test_read_ids(tmp_path):
    f = tmp_path / "ids.tsv"
    f.write_text("# id\tkind\tname\nLEGAL_DOC::a\tprivate-looking\tX\n\nLEGAL_DOC::b  # keep a note\n"
                 "LEGAL_DOC::a\tagain\n", encoding="utf-8")
    assert qd.read_ids(str(f)) == ["LEGAL_DOC::a", "LEGAL_DOC::b"]


def test_the_visibility_filter_hides_a_quarantined_document():
    from src.rag.answer_processing import _visibility_filter
    where = _visibility_filter()
    # visibility 'private' and owner_id 'quarantine': no user id or tenant id is 'quarantine'
    assert "coalesce(d.visibility, 'public') = 'public'" in where and "d.owner_id = $user_id" in where
