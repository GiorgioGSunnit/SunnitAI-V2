"""Data models for the RAG pipeline."""

from __future__ import annotations

from typing import List, Optional, Set

from pydantic import BaseModel, Field

from ..preprocessing.schema.schema import entities as schema_entities


def _collect_schema_labels() -> List[str]:
    return [item["label"] for item in schema_entities]

# A list of all valid labels from the schema.
# This list can be generated dynamically from schema.py if needed.
_VALID_LABELS = {
    "Document", "LegalAct", "Article", "Clause", "Institution", "Person", 
    "Company", "CourtCase", "Court", "LegalParty", "Section", "Meeting", 
    "Resolution", "Tender", "Award", "Contract", "ChangeOrder", "Auction", 
    "Asset", "Penalty", "Complaint", "Vote", "Correction", "Addendum", "Topic"
}

_ALLOWED_LABELS_DESC = (
    "Document, LegalAct, Article, Clause, Institution, Person, Company, CourtCase, "
    "Court, LegalParty, Section, Meeting, Resolution, Tender, Award, Contract, "
    "ChangeOrder, Auction, Asset, Penalty, Complaint, Vote, Correction, Addendum, Topic"
)


class TypedEntity(BaseModel):
    """Represents a single typed entity extracted from the user query."""

    id: str = Field(description="A temporary, unique identifier for this node (e.g., 'node1', 'node2').")
    # Use plain str so the LLM cannot break JSON parsing with invented labels
    # (e.g. Country, Location). Invalid labels are dropped later in graph_nodes.
    label: str = Field(
        description=f"The Neo4j node label. MUST be one of: {_ALLOWED_LABELS_DESC}. Never invent other label names."
    )
    properties: dict = Field(
        description="Key-value pairs of properties to match. Must adhere to the schema for the given label.",
        additionalProperties=False
    )

class TypedRelationship(BaseModel):
    """Represents a relationship between two nodes, identified by temporary IDs."""
    source_id: str = Field(description="The temporary ID of the source node.")
    target_id: str = Field(description="The temporary ID of the target node.")
    type: str = Field(description="The type of the relationship (e.g., 'ISSUED_BY').")

class ExtractedGraph(BaseModel):
    """Represents the graph structure (nodes and relationships) extracted from the query."""
    nodes: List[TypedEntity] = Field(description="A list of all structured nodes.")
    relationships: List[TypedRelationship] = Field(
        description="A list of relationships connecting the nodes.", default_factory=list
    )

class DocumentEntities(BaseModel):
    """Represents all entities and relationships extracted from a user's query."""

    graph: ExtractedGraph = Field(
        description="The graph structure extracted from the query."
    )
    
    @classmethod
    def allowed_labels(cls) -> Set[str]:
        return _VALID_LABELS




