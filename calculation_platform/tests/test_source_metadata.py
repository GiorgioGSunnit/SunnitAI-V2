"""Enforces the precision of legal-source metadata: every legal citation
must be independently findable online — a stable URL or, at minimum, an
exact Gazzetta Ufficiale issue reference — so a future verification pass
(automated or human) can check each value against its official source.
"""

from app.main import engine
from app.schemas.citation import Citation


def _legal_definitions():
    return [d for d in engine.registry.definitions() if d.category == "legal_it"]


def test_every_legal_calculator_has_at_least_one_citation():
    for definition in _legal_definitions():
        assert definition.citations, f"{definition.id} has no citations"


def test_every_legal_citation_is_precisely_searchable():
    for definition in _legal_definitions():
        for raw in definition.citations:
            citation = Citation(**raw)
            assert citation.official, f"{definition.id}: non-official citation {citation.reference!r}"
            assert citation.publisher, f"{definition.id}: citation missing publisher: {citation.reference!r}"
            # findable online: a stable permalink, or an exact G.U. issue
            has_pointer = bool(citation.url) or (
                citation.source_name and "Gazzetta Ufficiale" in citation.source_name
            )
            assert has_pointer, f"{definition.id}: citation not searchable online: {citation.reference!r}"


def test_every_legal_parameter_value_carries_sourced_citations():
    for parameter_id, entries in engine.parameter_store._values.items():
        if not parameter_id.startswith("legal_it"):
            continue
        for pv in entries:
            assert pv.source, f"{parameter_id} entry from {pv.effective_from} has no source"
            assert pv.official, f"{parameter_id} entry from {pv.effective_from} not marked official"
            assert pv.citations, f"{parameter_id} entry from {pv.effective_from} has no citations"
            for citation in pv.citations:
                has_pointer = bool(citation.url) or (
                    citation.source_name and "Gazzetta Ufficiale" in citation.source_name
                )
                assert has_pointer, (
                    f"{parameter_id}: citation not searchable online: {citation.reference!r}"
                )


def test_citation_urls_point_at_official_domains_only():
    official_domains = ("normattiva.it", "gazzettaufficiale.it", "agenziaentrate.gov.it",
                        "mef.gov.it", "inps.it", "istat.it")
    for definition in engine.registry.definitions():
        for raw in definition.citations:
            citation = Citation(**raw)
            if citation.url:
                assert any(domain in citation.url for domain in official_domains), (
                    f"{definition.id}: citation URL is not an official source: {citation.url}"
                )
