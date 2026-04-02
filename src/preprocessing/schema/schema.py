"""
event
organization
person
legal_doc
"""

import os
from datetime import date
from typing import List, Literal

from dotenv import load_dotenv

from .enums import *
from pydantic import BaseModel, Field

load_dotenv()

# Vector dimension depends on the embedding model:
#   - OpenAI text-embedding-3-small: 1536
#   - BAAI/bge-large-en-v1.5: 1024
#   - all-MiniLM-L6-v2: 384
VECTOR_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))


class Node(BaseModel):
    id: str = Field(
        ...,
        description="Unique identifier. Normalize strings using only lowercase (e.g., 'ministry-of-health').",
    )
    label: Literal[
        "ORGANIZATION", "EVENT", "PERSON", "LEGAL_DOC",
        "LOCATION", "LEGAL_CONCEPT", "ROLE", "DATE",
        "LEGAL_ACTION", "DOCUMENT_SECTION"
    ] = Field(
        ...,
        description="One of the following depending on the node nature: ORGANIZATION, EVENT, PERSON, LEGAL_DOC, LOCATION, LEGAL_CONCEPT, ROLE, DATE, LEGAL_ACTION, DOCUMENT_SECTION",
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
    page: int = Field(
        ...,
        description="If the node is an event, this is the value of the field 'page_number' of the json, you should save it for reference. For other node types, use -1 or a relevant page number if applicable.",
    )


class Relationship(BaseModel):
    source_id: str = Field(..., description="ID of the source node")
    target_id: str = Field(..., description="ID of the target node")
    type: Literal[
        "ANNOUNCES",          # ORGANIZATION -> EVENT (e.g., "Company announces merger")
        "PARTICIPATES_IN",    # PERSON/ORGANIZATION -> EVENT (e.g., "CEO participates in trial")
        "AFFILIATED_WITH",    # PERSON -> ORGANIZATION or ORGANIZATION -> ORGANIZATION (e.g., "Subsidiary of")
        "REGULATED_BY",       # EVENT/ORGANIZATION/LEGAL_ACTION -> LEGAL_DOC (e.g., "Event is regulated by law X")
        "HAS_ROLE",           # PERSON -> ROLE (e.g., "Person has role 'CEO' in Organization")
        "LOCATED_AT",         # EVENT/ORGANIZATION/LEGAL_ACTION -> LOCATION (e.g., "Trial located at Court Y")
        "REFERS_TO",          # LEGAL_DOC/DOCUMENT_SECTION -> LEGAL_CONCEPT (e.g., "Document refers to 'Privacy'")
        "OCCURS_ON",          # EVENT/LEGAL_ACTION -> DATE (e.g., "Trial occurs on 2025-11-15")
        "CONTAINS",           # LEGAL_DOC -> DOCUMENT_SECTION (e.g., "Document contains section 'Article 5'")
        "REGULATES",          # LEGAL_DOC -> LEGAL_CONCEPT (e.g., "Law regulates 'Data Protection'")
        "IS_PART_OF",         # EVENT -> EVENT or DOCUMENT_SECTION -> LEGAL_DOC (e.g., "Hearing is part of Trial")
    ] = Field(
        ...,
        description="""
        Type of relationship. Choose from:
        - ANNOUNCES: ORGANIZATION -> EVENT
        - PARTICIPATES_IN: PERSON/ORGANIZATION -> EVENT
        - AFFILIATED_WITH: PERSON -> ORGANIZATION or ORGANIZATION -> ORGANIZATION
        - REGULATED_BY: EVENT/ORGANIZATION/LEGAL_ACTION -> LEGAL_DOC
        - HAS_ROLE: PERSON -> ROLE
        - LOCATED_AT: EVENT/ORGANIZATION/LEGAL_ACTION -> LOCATION
        - REFERS_TO: LEGAL_DOC/DOCUMENT_SECTION -> LEGAL_CONCEPT
        - OCCURS_ON: EVENT/LEGAL_ACTION -> DATE
        - CONTAINS: LEGAL_DOC -> DOCUMENT_SECTION
        - REGULATES: LEGAL_DOC -> LEGAL_CONCEPT
        - IS_PART_OF: EVENT -> EVENT or DOCUMENT_SECTION -> LEGAL_DOC
        """
    )
    properties: dict = Field(
        default_factory=dict,
        description="""
        Use this for nuance. Examples:
        - For PARTICIPATES_IN: {"role": "Defendant", "status": "Active"}
        - For ANNOUNCES: {"date": "2025-11-15", "medium": "Press Release"}
        - For HAS_ROLE: {"role_type": "Executive", "start_date": "2025-01-01"}
        - For LOCATED_AT: {"location_type": "Courtroom", "address": "Via Roma 1"}
        - For OCCURS_ON: {"time": "09:00:00"}
        - For IS_PART_OF: {"relationship_type": "Sub-event"}
        - For REGULATED_BY: {"regulation_type": "Compliance", "jurisdiction": "Italy"}
        - For REFERS_TO: {"reference_type": "Definition", "context": "Article 3"}
        - For CONTAINS: {"section_type": "Clause", "order": 1}
        - For REGULATES: {"scope": "National", "effective_date": "2025-01-01"}
        """,
    )


class KnowledgeGraphExtraction(BaseModel):
    """
    Container model for strict GraphRAG extraction.
    Enforces separation between Entities definition and Relations linkage.
    """

    nodes: List[Node] = Field(
        ...,
        description="List of all unique entities identified in the text chunk. No duplicates allowed.",
    )
    relationships: List[Relationship] = Field(
        ...,
        description="List of all directed edges connecting the identified nodes. Ensure source_id and target_id exist in the nodes list.",
    )


def _build_vector_index(name: str, label: str) -> str:
    return (
        f"CREATE VECTOR INDEX {name} IF NOT EXISTS\n"
        f"FOR (n:{label}) ON (n.embedding)\n"
        f"OPTIONS {{indexConfig: {{\n"
        f" `vector.dimensions`: {VECTOR_DIMENSIONS},\n"
        f" `vector.similarity_function`: 'cosine'\n"
        f"}}}};"
    )


SCHEMA_CONSTRAINTS_AND_INDEXES = "\n\n".join([
    # 1. Uniqueness constraints
    "CREATE CONSTRAINT unique_org_id IF NOT EXISTS FOR (o:ORGANIZATION) REQUIRE o.id IS UNIQUE;",
    "CREATE CONSTRAINT unique_person_id IF NOT EXISTS FOR (p:PERSON) REQUIRE p.id IS UNIQUE;",
    "CREATE CONSTRAINT unique_event_id IF NOT EXISTS FOR (e:EVENT) REQUIRE e.id IS UNIQUE;",
    "CREATE CONSTRAINT unique_doc_id IF NOT EXISTS FOR (d:LEGAL_DOC) REQUIRE d.id IS UNIQUE;",
    "CREATE CONSTRAINT unique_location_id IF NOT EXISTS FOR (l:LOCATION) REQUIRE l.id IS UNIQUE;",
    "CREATE CONSTRAINT unique_date_id IF NOT EXISTS FOR (dt:DATE) REQUIRE dt.id IS UNIQUE;",
    "CREATE CONSTRAINT unique_legal_concept_id IF NOT EXISTS FOR (lc:LEGAL_CONCEPT) REQUIRE lc.id IS UNIQUE;",
    "CREATE CONSTRAINT unique_role_id IF NOT EXISTS FOR (r:ROLE) REQUIRE r.id IS UNIQUE;",
    "CREATE CONSTRAINT unique_legal_action_id IF NOT EXISTS FOR (la:LEGAL_ACTION) REQUIRE la.id IS UNIQUE;",
    "CREATE CONSTRAINT unique_document_section_id IF NOT EXISTS FOR (ds:DOCUMENT_SECTION) REQUIRE ds.id IS UNIQUE;",
    # 2. Vector indexes
    _build_vector_index("event_vector_index", "EVENT"),
    _build_vector_index("org_vector_index", "ORGANIZATION"),
    _build_vector_index("person_vector_index", "PERSON"),
    _build_vector_index("legal_doc_vector_index", "LEGAL_DOC"),
    _build_vector_index("legal_concept_vector_index", "LEGAL_CONCEPT"),
    _build_vector_index("role_vector_index", "ROLE"),
    _build_vector_index("legal_action_vector_index", "LEGAL_ACTION"),
    # 3. Full-text index
    (
        "CREATE FULLTEXT INDEX global_fulltext_index IF NOT EXISTS\n"
        "FOR (n:EVENT|ORGANIZATION|PERSON|LEGAL_DOC|LOCATION|DATE|LEGAL_CONCEPT|ROLE|LEGAL_ACTION|DOCUMENT_SECTION)\n"
        "ON EACH [n.id, n.ref_number, n.name, n.name_en, n.title, n.description]\n"
        "OPTIONS {indexConfig: {`fulltext.analyzer`: 'standard'}};"
    ),
])


# ---------------------------------------------------------------------------
# Entity & relationship definitions used by the RAG pipeline
# ---------------------------------------------------------------------------

entities = [
    {"label": "Document", "key": ["document_id"], "properties": ["document_title", "document_date", "issue_number", "volume_number", "document_type", "language", "authors", "publisher", "document_summary", "total_pages", "processed"]},
    {"label": "LegalAct", "key": ["act_type", "act_number", "act_year"], "properties": ["title", "subject", "effective_date"]},
    {"label": "Article", "key": ["parent_act_key", "index"], "properties": ["heading", "text_en", "text_ar", "version"]},
    {"label": "Clause", "key": ["parent_article_key", "index"], "properties": ["text_en", "text_ar"]},
    {"label": "Institution", "key": ["normalized_name"], "properties": ["name", "abbr"]},
    {"label": "Person", "key": ["normalized_name"], "properties": ["name", "role", "tenure_from", "tenure_to"]},
    {"label": "Company", "key": ["normalized_name"], "properties": ["name", "reg_id", "status", "legal_form"]},
    {"label": "CourtCase", "key": ["document_id", "chunk_id"], "properties": ["case_number", "title"]},
    {"label": "Court", "key": ["normalized_name"], "properties": ["name"]},
    {"label": "LegalParty", "key": ["normalized_name"], "properties": ["display_name"]},
    {"label": "Section", "key": ["document_id", "chunk_id"], "properties": ["title", "text_en", "text_ar"]},
    {"label": "Meeting", "key": ["meeting_id"], "properties": ["type", "date", "location"]},
    {"label": "Resolution", "key": ["resolution_id"], "properties": ["title", "text", "date"]},
    {"label": "Tender", "key": ["tender_id"], "properties": ["title", "ref_no", "subject", "issuer", "deadline"]},
    {"label": "Award", "key": ["award_id"], "properties": ["title", "date", "amount", "currency", "recommendation_text"]},
    {"label": "Contract", "key": ["contract_id"], "properties": ["title", "ref_no", "start_date", "end_date", "amount", "currency"]},
    {"label": "ChangeOrder", "key": ["change_order_id"], "properties": ["date", "reason", "amount_delta", "currency"]},
    {"label": "Auction", "key": ["auction_id"], "properties": ["title", "date", "location", "terms", "conditions"]},
    {"label": "Asset", "key": ["asset_id"], "properties": ["type", "description", "location"]},
    {"label": "Penalty", "key": ["penalty_id"], "properties": ["type", "amount", "currency", "reason", "date"]},
    {"label": "Complaint", "key": ["complaint_id"], "properties": ["type", "date", "subject"]},
    {"label": "Vote", "key": ["vote_id"], "properties": ["motion", "for_count", "against_count", "abstain_count", "result"]},
    {"label": "Correction", "key": ["correction_id"], "properties": ["text", "date"]},
    {"label": "Addendum", "key": ["addendum_id"], "properties": ["text", "date"]},
    {"label": "Topic", "key": ["label"], "properties": []},
]

relations = [
    {"type": "PUBLISHED_IN", "from": "LegalAct", "to": "Document", "properties": []},
    {"type": "ISSUED_BY", "from": "LegalAct", "to": "Institution", "properties": []},
    {"type": "SIGNED_BY", "from": "LegalAct", "to": "Person", "properties": []},
    {"type": "APPOINTS", "from": "LegalAct", "to": "Person", "properties": ["role"]},
    {"type": "APPOINTS", "from": "Institution", "to": "Person", "properties": ["role"]},
    {"type": "HAS_ARTICLE", "from": "LegalAct", "to": "Article", "properties": []},
    {"type": "HAS_CLAUSE", "from": "Article", "to": "Clause", "properties": []},
    {"type": "HAS_VERSION", "from": "Article", "to": "Article", "properties": ["relation"]},
    {"type": "AMENDS", "from": "LegalAct", "to": "LegalAct", "properties": []},
    {"type": "REPEALS", "from": "LegalAct", "to": "LegalAct", "properties": []},
    {"type": "REFERENCES", "from": "LegalAct", "to": "LegalAct", "properties": ["text"]},
    {"type": "HAS_TOPIC", "from": "LegalAct", "to": "Topic", "properties": []},
    {"type": "HAS_LEGAL_FORM", "from": "Company", "to": "Topic", "properties": []},
    {"type": "ADJUDICATED_BY", "from": "CourtCase", "to": "Court", "properties": []},
    {"type": "INVOLVES", "from": "CourtCase", "to": "LegalParty", "properties": ["role"]},
    {"type": "HAS_CHUNK", "from": "Document", "to": "Section", "properties": ["source_chunk_type"]},
    {"type": "HAS_CHUNK", "from": "Document", "to": "CourtCase", "properties": ["source_chunk_type"]},
    {"type": "HAS_CHUNK", "from": "Document", "to": "Company", "properties": ["source_chunk_type"]},
    {"type": "HAS_CHUNK", "from": "Document", "to": "Tender", "properties": ["source_chunk_type"]},
    {"type": "HAS_CHUNK", "from": "Document", "to": "Award", "properties": ["source_chunk_type"]},
    {"type": "HAS_CHUNK", "from": "Document", "to": "Penalty", "properties": ["source_chunk_type"]},
    {"type": "HAS_CHUNK", "from": "Document", "to": "Auction", "properties": ["source_chunk_type"]},
    {"type": "HAS_CHUNK", "from": "Document", "to": "LegalAct", "properties": ["source_chunk_type"]},
    {"type": "MENTIONS", "from": "Section", "to": "Institution", "properties": []},
    {"type": "HOLDS_MEETING", "from": "Company", "to": "Meeting", "properties": []},
    {"type": "PASSES_RESOLUTION", "from": "Meeting", "to": "Resolution", "properties": []},
    {"type": "HAS_VOTE", "from": "Meeting", "to": "Vote", "properties": []},
    {"type": "ISSUES_TENDER", "from": "Institution", "to": "Tender", "properties": []},
    {"type": "REMOVED_FROM_TENDER", "from": "Tender", "to": "Company", "properties": []},
    {"type": "AWARDED_TO", "from": "Award", "to": "Company", "properties": []},
    {"type": "AWARD_OF", "from": "Award", "to": "Tender", "properties": []},
    {"type": "RESULTS_IN_CONTRACT", "from": "Award", "to": "Contract", "properties": []},
    {"type": "AMENDS_CONTRACT", "from": "ChangeOrder", "to": "Contract", "properties": []},
    {"type": "ISSUED_BY", "from": "ChangeOrder", "to": "Institution", "properties": []},
    {"type": "ANNOUNCES_AUCTION", "from": "Institution", "to": "Auction", "properties": []},
    {"type": "AUCTION_OF", "from": "Auction", "to": "Asset", "properties": []},
    {"type": "AUCTION_ORGANIZED_FOR", "from": "Auction", "to": "Company", "properties": []},
    {"type": "IMPOSED_BY", "from": "Penalty", "to": "Institution", "properties": []},
    {"type": "IMPOSED_ON", "from": "Penalty", "to": "Company", "properties": []},
    {"type": "FILED_BY", "from": "Complaint", "to": "Person", "properties": []},
    {"type": "FILED_AGAINST", "from": "Complaint", "to": "Institution", "properties": []},
    {"type": "CORRECTS", "from": "Correction", "to": "Document", "properties": []},
    {"type": "ADDENDS", "from": "Addendum", "to": "Document", "properties": []},
]
