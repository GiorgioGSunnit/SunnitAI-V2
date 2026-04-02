from dataclasses import dataclass
from typing import List


@dataclass
class LookupConfig:
    # Lookup config: pairs a label with the property used in equality checks
    label: str
    property: str


BTREE_LOOKUPS: List[LookupConfig] = [
    LookupConfig("Document", "document_id"),
    LookupConfig("Company", "normalized_name"),
    LookupConfig("Institution", "normalized_name"),
    LookupConfig("Person", "normalized_name"),
    LookupConfig("Court", "normalized_name"),
    LookupConfig("LegalParty", "normalized_name"),
    LookupConfig("Meeting", "meeting_id"),
    LookupConfig("Resolution", "resolution_id"),
    LookupConfig("Tender", "tender_id"),
    LookupConfig("Award", "award_id"),
    LookupConfig("Contract", "contract_id"),
    LookupConfig("ChangeOrder", "change_order_id"),
    LookupConfig("Auction", "auction_id"),
    LookupConfig("Asset", "asset_id"),
    LookupConfig("Penalty", "penalty_id"),
    LookupConfig("Complaint", "complaint_id"),
    LookupConfig("Vote", "vote_id"),
    LookupConfig("Correction", "correction_id"),
    LookupConfig("Addendum", "addendum_id"),
    LookupConfig("Topic", "label"),
]

FULLTEXT_INDEXES: List[str] = [
    "entity_names",
    "legal_text_keywords",
    "event_keywords",
    "document_keywords",
]
ENTITY_VECTOR_INDEXES = [
    "document_embeddings",
    "legalact_embeddings",
    "article_embeddings",
    "clause_embeddings",
    "section_embeddings",
    "institution_embeddings",
    "person_embeddings",
    "company_embeddings",
    "court_embeddings",
    "courtcase_embeddings",
    "legalparty_embeddings",
    "tender_embeddings",
    "award_embeddings",
    "contract_embeddings",
    "changeorder_embeddings",
    "auction_embeddings",
    "penalty_embeddings",
]
CONTEXT_VECTOR_INDEXES = [
    "document_embeddings",
    "section_embeddings",
    "legalact_embeddings",
    "penalty_embeddings",
    "courtcase_embeddings",
]
CONTEXT_NODE_LIMIT = 10
VECTOR_K = 3
