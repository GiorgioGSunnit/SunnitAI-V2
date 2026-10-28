"""scripts/audit_ccnl.py: reading CCNL names, sections and pay tables; the audit and its report on a fake
read-only session. Invented documents only."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "audit_ccnl", Path(__file__).resolve().parent.parent / "scripts" / "audit_ccnl.py")
audit_ccnl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit_ccnl)


@pytest.mark.parametrize("name, kind", [
    ("CCNL Commercio Terziario Distribuzione e Servizi 30/03/2024", "ccnl"),
    ("Ipotesi di accordo di rinnovo CCNL Metalmeccanici 5 febbraio 2021", "ipotesi"),
    ("Testo consolidato CCNL Turismo 2023", "testo consolidato"),
    ("Accordo di rinnovo del CCNL Studi Professionali", "accordo di rinnovo"),
    ("Accordo economico Pulizie multiservizi", "accordo"),
    ("Verbale di accordo Logistica", "verbale"),
    ("Protocollo d'intesa Edilizia", "protocollo"),
    ("Contratto collettivo nazionale di lavoro Chimici", "ccnl"),
    ("Tessili 2022", "other"),
])
def test_the_agreement_type_is_read_from_the_name(name, kind):
    assert audit_ccnl.agreement_type(name) == kind


@pytest.mark.parametrize("name, expected", [
    ("CCNL Commercio 30/03/2024", ("2024-03-30", "numeric")),
    ("Ipotesi di accordo 5 febbraio 2021", ("2021-02-05", "words")),
    ("CCNL Turismo 2019 - rinnovo 12.07.2023", ("2023-07-12", "numeric")),     # the last date written
    ("Testo consolidato 2023", ("2023", "year")),
    ("CCNL Chimici", (None, "none")),
])
def test_the_date_is_read_from_the_name(name, expected):
    assert audit_ccnl.name_date(name) == expected


def test_versions_of_one_contract_share_a_sector_key():
    a = audit_ccnl.sector_key({"name": "CCNL Metalmeccanici Industria 5 febbraio 2021"})
    b = audit_ccnl.sector_key({"name": "Ipotesi di accordo di rinnovo CCNL Metalmeccanici Industria 2024"})
    assert a == b == "metalmeccanici industria"
    assert audit_ccnl.sector_key({"name": "x", "sector": "Commercio"}) == "commercio"


@pytest.mark.parametrize("name, shaped", [
    ("12", True), ("12.0.0", True), ("12-bis", True), ("Art. 12", True), ("Articolo 5", True),
    ("Premessa", False), ("Allegato A", False), ("", False)])
def test_article_shaped_section_names(name, shaped):
    assert audit_ccnl.article_shaped(name) is shaped


def test_table_shapes():
    assert audit_ccnl.table_shape("| livello | minimo |\n|---|---|\n| 1 | 1.900,00 |") == "markdown table"
    rows = "Livello 1  1.950,00  2.010,00\nLivello 2  1.800,00  1.850,00\nLivello 3  1.700,00  1.740,00"
    assert audit_ccnl.table_shape(rows) == "rows of amounts"
    assert audit_ccnl.table_shape("Il minimo del 2° livello è di € 1.800,00 mensili.") == "amounts in prose"
    assert audit_ccnl.table_shape("Le parti si incontrano.") == "no amounts"


def test_examples_cover_each_type_then_the_largest_sectors():
    docs = [{"id": f"m{i}", "type": "ccnl", "sector": "metalmeccanici", "date": f"202{i}", "sections": i}
            for i in range(4)]
    docs += [{"id": "t1", "type": "testo consolidato", "sector": "turismo", "date": "2023", "sections": 50},
             {"id": "c1", "type": "ccnl", "sector": "commercio", "date": "2024", "sections": 80}]
    # one per type (the most sections), then the largest sector's latest; never a document twice
    assert [d["id"] for d in audit_ccnl.pick_examples(docs, 4)] == ["t1", "c1", "m3"]
    assert [d["id"] for d in audit_ccnl.pick_examples(docs, 1)] == ["t1"]


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def data(self):
        return self._rows

    def single(self):
        return self._rows[0]


class FakeSession:
    """Answers each of the audit's queries by a fragment of it; records that every query is a read."""
    DOCS = [{"id": "LEGAL_DOC::c1", "name": "CCNL Commercio 30/03/2024", "sector": "Commercio",
             "valid_until": "2027-03-31", "sections": 3, "headed": 2, "embedded": 3,
             "names": ["1", "2", "Allegato A"]},
            {"id": "LEGAL_DOC::m1", "name": "Ipotesi di accordo di rinnovo CCNL Metalmeccanici 5 febbraio 2021",
             "sector": None, "valid_until": None, "sections": 0, "headed": 0, "embedded": 0, "names": []}]

    def __init__(self):
        self.queries = []

    def run(self, query, **params):
        self.queries.append(query)
        if "labels(d) AS labels" in query:
            return _Result([{"labels": ["Document", "LEGAL_DOC"], "n": 2}])
        if "UNWIND keys(d)" in query:
            return _Result([{"k": "name", "n": 2}, {"k": "sector", "n": 1}, {"k": "valid_until", "n": 1},
                            {"k": "previous_id", "n": 1}])
        if "RETURN d.`" in query:
            return _Result([{"v": "Commercio"}])
        if "collect(s.name)" in query:
            return _Result(self.DOCS)
        if "UNWIND keys(s)" in query:
            return _Result([{"k": "name", "n": 3}, {"k": "titolo_articolo", "n": 2}])
        if "minim" in query:
            return _Result([{"doc": "CCNL Commercio", "section": "Allegato A", "heading": "Minimi",
                             "text": "| livello | minimo |\n|---|---|\n| 1 | 1.900,00 |"}])
        if "labels(x)" in query:
            return _Result([{"labels": ["Section"], "n": 3}])
        if "UNION ALL" in query:
            return _Result([{"dir": "out", "type": "AMENDS", "other": "ccnl", "n": 1}])
        if "type(r) <> 'CONTAINS'" in query:
            return _Result([{"type": "AMENDS", "n": 1}])
        if "properties(d)" in query:
            return _Result([{"p": {"name": "CCNL Commercio", "embedding_v2": [0.1] * 8, "sector": "Commercio"}}])
        if "ORDER BY coalesce(s.order" in query:
            return _Result([{"name": "1", "heading": "Sfera di applicazione", "title": None,
                             "text": "Il presente contratto si applica...", "keys": ["name", "plain_text"]}])
        raise AssertionError(query)


def test_the_audit_and_its_report_on_a_fake_graph():
    session = FakeSession()
    result = audit_ccnl.audit(session, 2, 3, "embedding_v2")
    assert result["documents"] == 2 and result["agreement_types"] == {"ccnl": 1, "ipotesi": 1}
    assert result["sections"]["documents_without_sections"] == 1
    assert result["sections"]["article_shaped_share"] == round(2 / 3, 3)
    assert result["pay_sections"]["shapes"] == {"markdown table": 1}
    assert result["predecessor_keys"] == {"previous_id": 1}
    assert "embedding_v2" not in result["examples"][0]["properties"]
    assert not any(word in q.upper() for q in session.queries for word in ("CREATE ", "MERGE ", "SET ", "DELETE "))
    text = "\n".join(audit_ccnl.report(result, "embedding_v2"))
    for heading in ("## 1 Documents", "## 2 Naming", "## 3 Versions per sector", "## 4 Sections", "## 5 Pay tables",
                    "## 6 Links", "## 7 Representative documents"):
        assert heading in text
    assert "| out | AMENDS | ccnl | 1 |" in text and "Sfera di applicazione" in text
