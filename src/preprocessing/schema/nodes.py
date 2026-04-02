"""
event
organization
person
legal_doc
"""

from datetime import date

from enums import *
from pydantic import BaseModel, Field


class Node(BaseModel):
    id: str = Field(
        ...,
        description="Unique identifier. Normalize strings using only lowercase (e.g., 'ministry-of-health').",
    )
    label: Literal["ORGANIZATION", "EVENT", "PERSON", "LEGAL_DOC"] = Field(
        ...,
        description="One of the following depending of the node nature: ORGANIZATION, EVENT, PERSON, LEGAL_DOC",
    )
    properties: dict = Field(
        default_factory=dict,
        description="Specific attributes like date, value_kwd, type, name, status, depending on the type of entity.",
    )
    # Il testo CRUCIALE per LightRAG
    embedding_text: str = Field(
        ...,
        description="The rich narrative description to be vectorized. Follow templates.",
    )


from typing import Literal


class Relationship(BaseModel):
    source_id: str = Field(..., description="ID of the source node")
    target_id: str = Field(..., description="ID of the target node")

    # QUI APPLICHIAMO IL VINCOLO DEL PAPER:
    type: Literal[
        "ANNOUNCES",  # Org -> Event (Chi emette)
        "PARTICIPATES_IN",  # Person/Org -> Event (Chi partecipa/vince/subisce)
        "AFFILIATED_WITH",  # Person -> Org o Org -> Org (Gerarchie, possesso)
        "REGULATED_BY",  # Event/Org -> Legal_Doc (Leggi applicabili)
    ] = Field(..., description="Strictly select one of these relationship types.")

    properties: dict = Field(
        default_factory=dict,
        description="CRITICAL: Use this for nuance. E.g., for PARTICIPATES_IN, set role='Winner' or role='Liquidator'. For ANNOUNCES, set date='YYYY-MM-DD'.",
    )


# Schema setup queries for Neo4j
SCHEMA_CONSTRAINTS_AND_INDEXES = """
// 1. Vincoli di Unicità (Per evitare duplicati durante il parsing massivo)
CREATE CONSTRAINT unique_org_id IF NOT EXISTS FOR (o:ORGANIZATION) REQUIRE o.id IS UNIQUE;
CREATE CONSTRAINT unique_person_id IF NOT EXISTS FOR (p:PERSON) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT unique_event_id IF NOT EXISTS FOR (e:EVENT) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT unique_doc_id IF NOT EXISTS FOR (d:LEGAL_DOC) REQUIRE d.id IS UNIQUE;

// Indice per Eventi (Tender, Aste)
CREATE VECTOR INDEX event_vector_index IF NOT EXISTS
FOR (n:EVENT) ON (n.embedding)
OPTIONS {indexConfig: {
 `vector.dimensions`: 1536,
 `vector.similarity_function`: 'cosine'
}};

// Indice per Organizzazioni
CREATE VECTOR INDEX org_vector_index IF NOT EXISTS
FOR (n:ORGANIZATION) ON (n.embedding)
OPTIONS {indexConfig: {
 `vector.dimensions`: 1536,
 `vector.similarity_function`: 'cosine'
}};


// Indice per Persone
CREATE VECTOR INDEX person_vector_index IF NOT EXISTS
FOR (n:PERSON) ON (n.embedding)
OPTIONS {indexConfig: {
 `vector.dimensions`: 1536,
 `vector.similarity_function`: 'cosine'
}};

// Unico indice per cercare ID o Nomi in tutto il database
CREATE FULLTEXT INDEX global_fulltext_index IF NOT EXISTS
FOR (n:EVENT|ORGANIZATION|PERSON|LEGAL_DOC)
ON EACH [n.id, n.ref_number, n.name_en]
"""
