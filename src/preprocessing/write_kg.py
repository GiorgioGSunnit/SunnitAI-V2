import json
from typing import List

from ..utils.db_client import DBClient
from .schema.schema import SCHEMA_CONSTRAINTS_AND_INDEXES, KnowledgeGraphExtraction


def save_to_neo4j(extraction: KnowledgeGraphExtraction) -> None:
    """
    Salva nodi e relazioni estratti dal LLM in Neo4j.

    Args:
        extraction: Oggetto KnowledgeGraphExtraction contenente nodes e relationships
    """
    with DBClient() as db:
        # 1. Salva nodi
        for node in extraction.nodes:
            cypher = f"""
            MERGE (n:{node.label} {{id: $id}})
            SET n.embedding_text = $embedding_text,
                n.page = $page,
                n += $properties
            """
            db.query(
                cypher,
                {
                    "id": node.id,
                    "embedding_text": node.embedding_text,
                    "page": node.page,
                    "properties": node.properties,
                },
            )

        # 2. Salva relazioni
        for rel in extraction.relationships:
            cypher = f"""
            MATCH (source {{id: $source_id}})
            MATCH (target {{id: $target_id}})
            MERGE (source)-[r:{rel.type}]->(target)
            SET r += $properties
            """
            db.query(
                cypher,
                {
                    "source_id": rel.source_id,
                    "target_id": rel.target_id,
                    "properties": rel.properties,
                },
            )


def write_indexing():
    """
    Crea constraints e indexes in Neo4j come definito in SCHEMA_CONSTRAINTS_AND_INDEXES.
    """
    with DBClient() as db:
        # Splitta le query per linee e esegui quelle non vuote
        queries = [
            q.strip()
            for q in SCHEMA_CONSTRAINTS_AND_INDEXES.split(";")
            if q.strip() and not q.strip().startswith("//")
        ]

        for query in queries:
            # Rimuovi i commenti inline
            query_lines = [
                line
                for line in query.split("\n")
                if line.strip() and not line.strip().startswith("//")
            ]
            clean_query = "\n".join(query_lines)

            if clean_query:
                db.query(clean_query)
        print("done")


if __name__ == "__main__":
    write_indexing()
