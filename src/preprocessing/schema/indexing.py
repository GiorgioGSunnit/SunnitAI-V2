"""Entry point for writing Neo4j schema constraints and indexes."""

from ..write_kg import relabel_legacy_nodes, write_indexing


def main() -> None:
    write_indexing()
