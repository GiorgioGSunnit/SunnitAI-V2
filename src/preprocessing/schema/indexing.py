from ..write_kg import write_indexing


def main() -> None:
    """Create Neo4j schema constraints and indexes."""
    write_indexing()
