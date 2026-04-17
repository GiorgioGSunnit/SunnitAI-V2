"""Entry point for writing Neo4j schema constraints and indexes."""

from ..write_kg import write_indexing


def main() -> None:
    write_indexing()
